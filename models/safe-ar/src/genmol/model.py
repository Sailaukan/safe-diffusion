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


from __future__ import annotations

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F

from genmol.backbone import SafeCausalDiTLM
from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info


class SafeAR(L.LightningModule):
    """Lightning module for autoregressive SAFE molecule generation."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id
        self.vocab_size = int(self.config.model.vocab_size)

        self.backbone = self._build_backbone()
        self.ema = self._build_ema()
        self.register_buffer("train_loss_ema_value", torch.tensor(float("nan")), persistent=False)

    def _build_backbone(self) -> SafeCausalDiTLM:
        if self.config.get("diffusion", {}).get("engine", "ar") != "ar":
            raise ValueError("SAFE-AR expects diffusion.engine=ar")
        if self.config.get("diffusion", {}).get("parameterization", "ar") != "ar":
            raise ValueError("SAFE-AR expects diffusion.parameterization=ar")
        if self.config.model.get("time_conditioning", False):
            raise ValueError("SAFE-AR expects model.time_conditioning=False")
        return SafeCausalDiTLM(self.config.model, vocab_size=self.vocab_size)

    def _build_ema(self):
        if self.config.training.ema > 0:
            return ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        return None

    def on_load_checkpoint(self, checkpoint):
        if self.ema and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if "sampler" not in checkpoint:
            checkpoint["sampler"] = {}
        sampler = getattr(self.trainer.train_dataloader, "sampler", None)
        if hasattr(sampler, "state_dict"):
            sampler_state_dict = sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get("random_state", None)
        else:
            checkpoint["sampler"]["random_state"] = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            {
                "_target_": "transformers.get_constant_schedule_with_warmup",
                "num_warmup_steps": int(self.config.optim.get("warmup_steps", 2500)),
            },
            optimizer=optimizer,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "lr",
        }
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
        del optimizer
        if not self.config.training.get("log_grad_norm", True):
            return
        log_every = int(self.config.trainer.get("log_every_n_steps", 10))
        if log_every > 0 and self.trainer.global_step % log_every != 0:
            return

        total_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.backbone.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().float().norm(2)
            total_norm_sq += grad_norm.square()
        self.log(
            name="grad_norm",
            value=total_norm_sq.sqrt(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    def _update_loss_ema(self, loss: torch.Tensor) -> None:
        decay = float(self.config.training.get("loss_ema_decay", 0.98))
        detached_loss = loss.detach().to(
            device=self.train_loss_ema_value.device,
            dtype=self.train_loss_ema_value.dtype,
        )
        if torch.isnan(self.train_loss_ema_value).item():
            self.train_loss_ema_value.copy_(detached_loss)
        else:
            self.train_loss_ema_value.mul_(decay).add_(detached_loss, alpha=1 - decay)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return self.backbone(input_ids, sigma=None)

    def _shift_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        input_tokens = input_ids[:, :-1]
        target_tokens = input_ids[:, 1:]
        if attention_mask is None:
            target_mask = target_tokens.ne(self.pad_index)
        else:
            target_mask = attention_mask[:, 1:].to(torch.bool)
        return input_tokens, target_tokens, target_mask

    def _token_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        max_target = int(targets.max().detach().item())
        if max_target >= self.vocab_size:
            raise ValueError(
                f"Batch contains token id {max_target}, but model vocab_size is "
                f"{self.vocab_size}. If using bracket SAFE, set "
                "training.use_bracket_safe=True so the config vocab is expanded."
            )
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

    def training_step(self, batch, batch_idx):
        del batch_idx
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        input_tokens, target_tokens, target_mask = self._shift_batch(input_ids, attention_mask)

        logits = self.forward(input_tokens)
        token_loss = self._token_loss(logits, target_tokens)
        masked_token_loss = token_loss * target_mask.to(token_loss.dtype)

        if self.config.training.global_mean_loss:
            loss = masked_token_loss.sum() / target_mask.sum().clamp_min(1)
        else:
            per_sample_loss = masked_token_loss.sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
            loss = per_sample_loss.mean()

        self._update_loss_ema(loss)
        self.log(
            name="train_loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            name="train_nll",
            value=loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            name="train_loss_ema",
            value=self.train_loss_ema_value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def _mask_sampling_logits(
        self,
        logits: torch.Tensor,
        generated_steps: int,
        min_new_tokens: int,
        ban_special_tokens: bool,
    ) -> torch.Tensor:
        logits = logits.clone()
        if ban_special_tokens:
            for token_id in (self.pad_index, self.bos_index, self.mask_index):
                if token_id is not None and 0 <= token_id < logits.shape[-1]:
                    logits[:, token_id] = -torch.inf
        if generated_steps < min_new_tokens and self.eos_index is not None:
            logits[:, self.eos_index] = -torch.inf
        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        randomness: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
            top_values, _ = torch.topk(logits, k=top_k, dim=-1)
            logits = logits.masked_fill(logits < top_values[:, [-1]], -torch.inf)

        temperature = max(float(temperature), 1e-6)
        scaled_logits = logits / temperature
        if randomness <= 0:
            return scaled_logits.argmax(dim=-1)

        noise = torch.distributions.Gumbel(0, 1).sample(scaled_logits.shape).to(scaled_logits.device)
        return (scaled_logits + float(randomness) * noise).argmax(dim=-1)

    def _prepare_prefix(self, prefix_ids: torch.Tensor | None, num_samples: int) -> torch.Tensor:
        if prefix_ids is None:
            return torch.full((num_samples, 1), self.bos_index, dtype=torch.long, device=self.device)

        prefix_ids = prefix_ids.to(device=self.device, dtype=torch.long)
        if prefix_ids.ndim == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        if prefix_ids.shape[0] == 1 and num_samples > 1:
            prefix_ids = prefix_ids.repeat(num_samples, 1)
        if prefix_ids.shape[0] != num_samples:
            raise ValueError("prefix_ids batch size must be 1 or num_samples")

        rows = []
        for row in prefix_ids:
            row = row[row.ne(self.pad_index)]
            if row.numel() == 0 or row[0].item() != self.bos_index:
                row = torch.cat([row.new_tensor([self.bos_index]), row])
            if row[-1].item() == self.eos_index:
                row = row[:-1]
            rows.append(row)
        max_len = max(row.numel() for row in rows)
        padded = prefix_ids.new_full((num_samples, max_len), self.pad_index)
        for idx, row in enumerate(rows):
            padded[idx, : row.numel()] = row
        return padded

    @torch.no_grad()
    def sample_ids(
        self,
        num_samples: int | None = None,
        prefix_ids: torch.Tensor | None = None,
        max_length: int | None = None,
        temperature: float = 1.0,
        randomness: float = 1.0,
        min_new_tokens: int = 0,
        top_k: int | None = None,
        stop_at_eos: bool = True,
        ban_special_tokens: bool = True,
    ) -> torch.Tensor:
        self.eval()
        if num_samples is None:
            num_samples = int(self.config.sampling.get("batch_size", 1))
        if max_length is None:
            max_length = int(
                self.config.model.get(
                    "max_position_embeddings",
                    self.config.model.get("length", 256),
                )
            )
        max_length = min(max_length, int(self.backbone.max_position_embeddings))
        min_new_tokens = max(int(min_new_tokens), 0)

        x = self._prepare_prefix(prefix_ids, num_samples)
        if x.shape[1] >= max_length:
            return x[:, :max_length]

        prefix_len = x.ne(self.pad_index).sum(dim=1)
        finished = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        while x.shape[1] < max_length:
            model_input = x
            logits = self.forward(model_input)[:, -1]
            generated_steps = int((x.ne(self.pad_index).sum(dim=1) - prefix_len).min().item())
            logits = self._mask_sampling_logits(
                logits=logits,
                generated_steps=generated_steps,
                min_new_tokens=min_new_tokens,
                ban_special_tokens=ban_special_tokens,
            )
            next_token = self._sample_next_token(
                logits=logits,
                temperature=temperature,
                randomness=randomness,
                top_k=top_k,
            )
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_index),
                next_token,
            )
            x = torch.cat([x, next_token[:, None]], dim=1)
            if self.eos_index is not None:
                finished |= next_token.eq(self.eos_index)
            if stop_at_eos and bool(finished.all()):
                break
        return x


# Backward-compatible alias for existing experiment scripts.
GenMol = SafeAR
