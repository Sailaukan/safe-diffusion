# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra.utils
import lightning as L
import torch
from transformers.models.bert.configuration_bert import BertConfig
from genmol.backbone import TimeConditionedBertForMaskedLM
from genmol.diffusion import LogLinearNoiseSchedule, UniformDiscreteDiffusion
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution, UniformTimeDistribution

from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info


class SafeUDLM(L.LightningModule):
    """Lightning module for SAFE uniform discrete diffusion."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id

        self.backbone = self._build_backbone()
        self.diffusion = self._build_diffusion()
        # Keep the historical attribute name so existing task code keeps working.
        self.mdlm = self.diffusion
        self.ema = self._build_ema()
        self.register_buffer('train_loss_ema_value', torch.tensor(float('nan')), persistent=False)

    def _build_backbone(self):
        return TimeConditionedBertForMaskedLM(BertConfig.from_dict(dict(self.config.model)))

    def _build_time_distribution(self):
        if self.config.training.antithetic_sampling:
            return AntitheticUniformTimeDistribution(
                sampling_eps=self.config.training.sampling_eps,
            )
        return UniformTimeDistribution(min_t=self.config.training.sampling_eps, max_t=1.0)

    def _build_diffusion(self):
        diffusion_config = self.config.get('diffusion', {})
        sampling_config = self.config.get('sampling', {})
        noise_eps = self.config.noise.get('eps', self.config.training.sampling_eps)
        noise_schedule = LogLinearNoiseSchedule(eps=noise_eps)
        return UniformDiscreteDiffusion(
            vocab_size=int(self.config.model.vocab_size),
            mask_index=self.mask_index,
            pad_index=self.pad_index,
            bos_index=self.bos_index,
            eos_index=self.eos_index,
            time_distribution=self._build_time_distribution(),
            noise_schedule=noise_schedule,
            zero_recon_loss=diffusion_config.get('zero_recon_loss', True),
            freeze_special_tokens=diffusion_config.get('freeze_special_tokens', True),
            sampling_steps=sampling_config.get('steps', 128),
            sampling_eps=sampling_config.get('eps', self.config.training.sampling_eps),
            final_denoise=sampling_config.get('final_denoise', True),
        )

    def _build_ema(self):
        if self.config.training.ema > 0:
            return ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        return None

    def on_load_checkpoint(self, checkpoint):
        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            {'_target_': 'transformers.get_constant_schedule_with_warmup',
             'num_warmup_steps': int(self.config.optim.get('warmup_steps', 2500))},
             optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self.backbone.parameters())

    def on_before_optimizer_step(self, optimizer):
        if not self.config.training.get('log_grad_norm', True):
            return
        log_every = int(self.config.trainer.get('log_every_n_steps', 10))
        if log_every > 0 and self.trainer.global_step % log_every != 0:
            return

        total_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.backbone.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().float().norm(2)
            total_norm_sq += grad_norm.square()
        self.log(name='grad_norm',
                 value=total_norm_sq.sqrt(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)

    def _update_loss_ema(self, loss):
        decay = float(self.config.training.get('loss_ema_decay', 0.98))
        detached_loss = loss.detach().to(
            device=self.train_loss_ema_value.device,
            dtype=self.train_loss_ema_value.dtype,
        )
        if torch.isnan(self.train_loss_ema_value).item():
            self.train_loss_ema_value.copy_(detached_loss)
        else:
            self.train_loss_ema_value.mul_(decay).add_(detached_loss, alpha=1 - decay)

    def _backbone_logits(self, x, attention_mask=None, timesteps=None):
        autocast_enabled = self.device.type == 'cuda'
        device_type = 'cuda' if autocast_enabled else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.float32, enabled=autocast_enabled):
            return self.backbone(x, attention_mask=attention_mask, timesteps=timesteps).logits

    def forward(self, x, attention_mask=None, timesteps=None):
        return self._backbone_logits(x, attention_mask=attention_mask, timesteps=timesteps)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        frozen_mask = self.diffusion.get_frozen_token_mask(input_ids)
        # sample time
        t = self.diffusion.sample_time(input_ids.shape[0], device=input_ids.device)
        sigma = self.diffusion.time_conditioning(t)
        # forward process to add uniform discrete noise
        xt = self.diffusion.forward_process(input_ids, t, frozen_mask=frozen_mask)
        logits = self._backbone_logits(xt, attention_mask=attention_mask, timesteps=sigma)
        # compute loss
        loss_terms = self.diffusion.loss_terms(
            logits=logits,
            x0=input_ids,
            xt=xt,
            t=t,
            mask=attention_mask,
            frozen_mask=frozen_mask,
            global_mean=self.config.training.global_mean_loss,
        )
        if self.config.training.global_mean_loss:
            loss = loss_terms.loss
        else:
            loss = loss_terms.loss.mean()
        self._update_loss_ema(loss)
        log_mask = loss_terms.loss_mask.to(logits.dtype)
        denom = log_mask.sum().clamp_min(1)
        diffusion_loss = (loss_terms.diffusion_loss * log_mask).sum() / denom
        reconstruction_loss = (loss_terms.reconstruction_loss * log_mask).sum() / denom
        self.log(name='train_loss',
                 value=loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        self.log(name='train_diffusion_loss',
                 value=diffusion_loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log(name='train_reconstruction_loss',
                 value=reconstruction_loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log(name='train_loss_ema',
                 value=self.train_loss_ema_value,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        return loss


# Backward-compatible alias for older imports/checkpoints.
GenMol = SafeUDLM
