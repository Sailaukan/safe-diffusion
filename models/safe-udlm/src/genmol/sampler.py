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

import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pickle
import random
import tempfile
from contextlib import suppress
from pathlib import Path

import safe as sf
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.model import SafeUDLM
from genmol.utils.utils_chem import Slicer, filter_by_substructure, mix_sequences, safe_to_smiles


ROOT_DIR = Path(__file__).resolve().parents[2]
LENGTH_DISTRIBUTION_PATH = ROOT_DIR / 'data' / 'len.pk'

DEFAULT_SOFTMAX_TEMP = 1.2
DEFAULT_RANDOMNESS = 2.0
DEFAULT_DE_NOVO_SOFTMAX_TEMP = 0.8
DEFAULT_DE_NOVO_RANDOMNESS = 0.5
DEFAULT_DE_NOVO_MIN_ADD_LEN = 40
DEFAULT_FRAGMENT_MIN_ADD_LEN = 30
DEFAULT_COMPLETION_MIN_ADD_LEN = 18


def _checkpoint_has_time_conditioning(checkpoint) -> bool:
    return any('sigma_map' in name for name in checkpoint.get('state_dict', {}))


def _patch_legacy_time_conditioning_config(checkpoint) -> None:
    if _checkpoint_has_time_conditioning(checkpoint):
        return

    model_cfg = checkpoint.get('hyper_parameters', {}).get('config', {}).get('model')
    if model_cfg is None:
        return

    if OmegaConf.is_config(model_cfg):
        OmegaConf.set_struct(model_cfg, False)
        model_cfg['time_conditioning'] = False
        OmegaConf.set_struct(model_cfg, True)
    else:
        model_cfg['time_conditioning'] = False


def _load_checkpoint_for_lightning(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    _patch_legacy_time_conditioning_config(checkpoint)

    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        torch.save(checkpoint, tmp_path)
        return SafeUDLM.load_from_checkpoint(tmp_path, map_location=device, strict=False)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)


def load_model_from_path(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_checkpoint_for_lightning(path, device)
    model = model.to(device)
    model.eval()
    model.backbone.eval()
    if model.ema:
        model.ema.store(model.backbone.parameters())
        model.ema.copy_to(model.backbone.parameters())
    return model


class Sampler:
    def __init__(self, path):
        self.model = load_model_from_path(path)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.diffusion = self.model.diffusion
        self.mdlm = self.diffusion
        self.diffusion.to_device(self.model.device)
        
    @torch.no_grad()
    def generate(
        self,
        x,
        softmax_temp=DEFAULT_SOFTMAX_TEMP,
        randomness=DEFAULT_RANDOMNESS,
        fix=True,
        gamma=0,
        w=2,
        **kwargs,
    ):
        template = x.to(self.model.device)
        attention_mask = template != self.pad_index
        model_attention_mask = attention_mask.long()
        editable_mask = (template == self.model.mask_index) & attention_mask
        frozen_mask = attention_mask & ~editable_mask

        if editable_mask.any():
            x = self.diffusion.initialize_sample(template, editable_mask)
        else:
            x = template.clone()

        num_steps = max(self.diffusion.get_num_steps_confidence(x), 2)
        timestep_grid = self.diffusion.get_sampling_timesteps(self.model.device, num_steps=num_steps)
        
        for i in range(num_steps):
            t = timestep_grid[i].expand(x.shape[0])
            sigma_t = self.diffusion.time_conditioning(t)
            logits = self.model(x, attention_mask=model_attention_mask, timesteps=sigma_t)

            if gamma and w:
                x_poor = self.diffusion.degrade_context(x, frozen_mask, gamma)
                logits_poor = self.model(x_poor, attention_mask=model_attention_mask, timesteps=sigma_t)
                logits = w * logits + (1 - w) * logits_poor

            x = self.diffusion.step_confidence(
                logits,
                x,
                i,
                num_steps,
                softmax_temp,
                randomness,
                editable_mask=editable_mask,
                timestep_grid=timestep_grid,
            )

        if editable_mask.any() and self.diffusion.final_denoise:
            sigma_0 = torch.zeros(x.shape[0], device=x.device)
            logits = self.model(x, attention_mask=model_attention_mask, timesteps=sigma_0)
            if gamma and w:
                x_poor = self.diffusion.degrade_context(x, frozen_mask, gamma)
                logits_poor = self.model(x_poor, attention_mask=model_attention_mask, timesteps=sigma_0)
                logits = w * logits + (1 - w) * logits_poor
            x = self.diffusion.final_denoise_step(logits, x, editable_mask, softmax_temp=softmax_temp, randomness=0.0)
            
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _load_length_distribution(self):
        with open(LENGTH_DISTRIBUTION_PATH, 'rb') as f:
            return pickle.load(f)

    def _sample_insert_length(self, current_length, min_add_len, mask_len, seq_len_list):
        if mask_len is not None:
            return max(int(mask_len), 1)
        return max(random.choice(seq_len_list) - current_length, min_add_len)

    def _insert_mask(self, x, num_samples, min_add_len=18, mask_len=None, **kwargs):
        x = x[0]
        seq_len_list = None if mask_len is not None else self._load_length_distribution()
        x_new = []
        for _ in range(num_samples):
            add_seq_len = self._sample_insert_length(len(x), min_add_len, mask_len, seq_len_list)
            masks = torch.full((add_seq_len,), self.model.mask_index, dtype=x.dtype)
            x_new.append(torch.hstack([x[:-1], masks, x[-1:]]))
        pad_len = max([len(xx) for xx in x_new])
        x_new = [
            torch.hstack([xx, torch.full((pad_len - len(xx),), self.pad_index, dtype=xx.dtype)])
            for xx in x_new
        ]
        return torch.stack(x_new)
    
    @torch.no_grad()
    def de_novo_generation(
        self,
        num_samples=1,
        softmax_temp=DEFAULT_DE_NOVO_SOFTMAX_TEMP,
        randomness=DEFAULT_DE_NOVO_RANDOMNESS,
        min_add_len=DEFAULT_DE_NOVO_MIN_ADD_LEN,
        **kwargs,
    ):
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp=softmax_temp, randomness=randomness)
    
    def fragment_linking_onestep(
        self,
        fragment,
        num_samples=1,
        softmax_temp=DEFAULT_SOFTMAX_TEMP,
        randomness=DEFAULT_RANDOMNESS,
        gamma=0,
        min_add_len=DEFAULT_FRAGMENT_MIN_ADD_LEN,
        **kwargs,
    ):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        samples = self.generate(x, softmax_temp=softmax_temp, randomness=randomness, gamma=gamma)
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(
        self,
        fragment,
        num_samples=1,
        softmax_temp=DEFAULT_SOFTMAX_TEMP,
        randomness=DEFAULT_RANDOMNESS,
        gamma=0,
        min_add_len=DEFAULT_FRAGMENT_MIN_ADD_LEN,
        **kwargs,
    ):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        prefix_samples = self.generate(x, softmax_temp=softmax_temp, randomness=randomness, gamma=gamma)

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        suffix_samples = self.generate(x, softmax_temp=softmax_temp, randomness=randomness, gamma=gamma)
        
        samples = filter_by_substructure(mix_sequences(prefix_samples, suffix_samples,
                                                      *fragment.split('.'), num_samples), fragment)
        return samples
        
    def fragment_completion(
        self,
        fragment,
        num_samples=1,
        apply_filter=True,
        softmax_temp=DEFAULT_SOFTMAX_TEMP,
        randomness=DEFAULT_RANDOMNESS,
        gamma=0,
        min_add_len=DEFAULT_COMPLETION_MIN_ADD_LEN,
        mask_len=None,
        **kwargs,
    ):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len, mask_len=mask_len)
        samples = self.generate(x, softmax_temp=softmax_temp, randomness=randomness, gamma=gamma)

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len=30, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except Exception:
            return smiles
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, **kwargs):
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        frag_idx = random.randint(0, len(special_token_idx) - 2)
        mask_start_idx = special_token_idx[frag_idx] + 1
        mask_end_idx = special_token_idx[frag_idx + 1]
        num_insert_mask = random.randint(5, 15)
        num_insert_mask = max(
            1,
            min(
                num_insert_mask,
                self.model.config.model.max_position_embeddings - x.shape[-1] + mask_end_idx - mask_start_idx,
            ),
        )
        x = torch.hstack([x[:, :mask_start_idx],
                          torch.full((1, num_insert_mask), self.model.mask_index),
                          x[:, mask_end_idx:]])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles
