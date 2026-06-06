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

import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import safe as sf
import torch
from rdkit import Chem

from genmol.model import SafeAR
from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.utils.utils_chem import Slicer, filter_by_substructure, mix_sequences, safe_to_smiles


DEFAULT_DE_NOVO_SOFTMAX_TEMP = 0.8
DEFAULT_DE_NOVO_RANDOMNESS = 0.5
DEFAULT_DE_NOVO_MIN_ADD_LEN = 40


def load_model_from_path(path: str, device: torch.device | str | None = None) -> SafeAR:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeAR.load_from_checkpoint(path, map_location=device, strict=False)
    model.to(device)
    model.backbone.eval()
    if model.ema:
        model.ema.store(model.backbone.parameters())
        model.ema.copy_to(model.backbone.parameters())
    return model


class Sampler:
    def __init__(self, path: str):
        self.model = load_model_from_path(path)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer(".")["input_ids"][1]
        self.pad_index = self.model.tokenizer.pad_token_id

    def _safe_to_smiles(self, samples: list[str], fix: bool = True) -> list[str]:
        if self.model.config.training.get("use_bracket_safe"):
            smiles = [safe_to_smiles(bracketsafe2safe(sample), fix=fix) for sample in samples]
        else:
            smiles = [safe_to_smiles(sample, fix=fix) for sample in samples]
        return [sorted(smi.split("."), key=len)[-1] for smi in smiles if smi]

    def _encode_prefix(self, safe_prefix: str, num_samples: int) -> torch.Tensor:
        batch = self.model.tokenizer(
            [safe_prefix],
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.model.max_position_embeddings,
        )
        prefix_ids = batch["input_ids"][0]
        prefix_ids = prefix_ids[prefix_ids.ne(self.pad_index)]
        if prefix_ids[-1].item() == self.model.eos_index:
            prefix_ids = prefix_ids[:-1]
        return prefix_ids.unsqueeze(0).repeat(num_samples, 1)

    @torch.no_grad()
    def generate_safe(
        self,
        num_samples: int = 1,
        safe_prefix: str | None = None,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        min_new_tokens: int = 0,
        max_length: int | None = None,
        top_k: int | None = None,
        **_,
    ) -> list[str]:
        prefix_ids = None
        if safe_prefix:
            prefix_ids = self._encode_prefix(safe_prefix, num_samples)

        sample_ids = self.model.sample_ids(
            num_samples=num_samples,
            prefix_ids=prefix_ids,
            max_length=max_length,
            temperature=softmax_temp,
            randomness=randomness,
            min_new_tokens=min_new_tokens,
            top_k=top_k,
            stop_at_eos=True,
            ban_special_tokens=True,
        )
        return [
            sample.strip()
            for sample in self.model.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
        ]

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor | None = None,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        fix: bool = True,
        min_new_tokens: int = 0,
        **kwargs,
    ) -> list[str]:
        if x is None:
            return self.de_novo_generation(
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_add_len=min_new_tokens,
                fix=fix,
                **kwargs,
            )

        x = x.detach().cpu()
        safe_samples: list[str] = []
        for row in x:
            keep = row.ne(self.pad_index)
            if self.model.mask_index is not None:
                keep &= row.ne(self.model.mask_index)
            if self.model.eos_index is not None:
                eos_positions = row.eq(self.model.eos_index).nonzero(as_tuple=True)[0]
                if eos_positions.numel():
                    keep[eos_positions[0] :] = False
            prefix_ids = row[keep]
            if prefix_ids.numel() == 0:
                safe_samples.extend(
                    self.generate_safe(
                        num_samples=1,
                        softmax_temp=softmax_temp,
                        randomness=randomness,
                        min_new_tokens=min_new_tokens,
                        **kwargs,
                    )
                )
                continue
            safe_prefix = self.model.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            safe_samples.extend(
                self.generate_safe(
                    num_samples=1,
                    safe_prefix=safe_prefix,
                    softmax_temp=softmax_temp,
                    randomness=randomness,
                    min_new_tokens=min_new_tokens,
                    **kwargs,
                )
            )
        return self._safe_to_smiles(safe_samples, fix=fix)

    @torch.no_grad()
    def de_novo_generation(
        self,
        num_samples: int = 1,
        softmax_temp: float = DEFAULT_DE_NOVO_SOFTMAX_TEMP,
        randomness: float = DEFAULT_DE_NOVO_RANDOMNESS,
        min_add_len: int = DEFAULT_DE_NOVO_MIN_ADD_LEN,
        fix: bool = True,
        **kwargs,
    ) -> list[str]:
        safe_samples = self.generate_safe(
            num_samples=num_samples,
            softmax_temp=softmax_temp,
            randomness=randomness,
            min_new_tokens=min_add_len,
            **kwargs,
        )
        return self._safe_to_smiles(safe_samples, fix=fix)

    def _fragment_prefix(self, fragment: str, onestep: bool = False) -> str:
        del onestep
        if self.model.config.training.get("use_bracket_safe"):
            return BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True) + "."
        converter = sf.SAFEConverter(slicer=None, ignore_stereo=True)
        return converter.encoder(fragment, allow_empty=True) + "."

    def fragment_linking_onestep(
        self,
        fragment,
        num_samples: int = 1,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        min_add_len: int = 10,
        **kwargs,
    ) -> list[str]:
        safe_prefix = self._fragment_prefix(fragment, onestep=True)
        samples = self._safe_to_smiles(
            self.generate_safe(
                num_samples=num_samples,
                safe_prefix=safe_prefix,
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_new_tokens=min_add_len,
                **kwargs,
            )
        )
        return filter_by_substructure(samples, fragment)

    def fragment_linking(
        self,
        fragment,
        num_samples: int = 1,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        min_add_len: int = 10,
        **kwargs,
    ) -> list[str]:
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        if "." not in encoded_fragment:
            return self.fragment_linking_onestep(
                fragment,
                num_samples=num_samples,
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_add_len=min_add_len,
                **kwargs,
            )
        prefix, suffix = encoded_fragment.split(".", maxsplit=1)
        prefix_samples = self._safe_to_smiles(
            self.generate_safe(
                num_samples=num_samples,
                safe_prefix=prefix + ".",
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_new_tokens=min_add_len,
                **kwargs,
            )
        )
        suffix_samples = self._safe_to_smiles(
            self.generate_safe(
                num_samples=num_samples,
                safe_prefix=suffix + ".",
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_new_tokens=min_add_len,
                **kwargs,
            )
        )
        samples = mix_sequences(prefix_samples, suffix_samples, *fragment.split("."), num_samples)
        return filter_by_substructure(samples, fragment)

    def fragment_completion(
        self,
        fragment,
        num_samples: int = 1,
        apply_filter: bool = True,
        softmax_temp: float = 1.0,
        randomness: float = 1.0,
        min_add_len: int = 10,
        **kwargs,
    ) -> list[str]:
        if "*" not in fragment:
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)

        safe_prefix = self._fragment_prefix(fragment)
        samples = self._safe_to_smiles(
            self.generate_safe(
                num_samples=num_samples,
                safe_prefix=safe_prefix,
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_new_tokens=min_add_len,
                **kwargs,
            )
        )
        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len: int = 30, **kwargs):
        encoded_smiles = sf.SAFEConverter(
            slicer=self.slicer,
            ignore_stereo=True,
        ).encoder(smiles, allow_empty=True)
        fragments = [fragment for fragment in encoded_smiles.split(".") if fragment]
        if not fragments:
            return smiles

        cut_idx = random.randint(1, len(fragments))
        safe_prefix = ".".join(fragments[:cut_idx]) + "."
        token_count = len(self.model.tokenizer(safe_prefix)["input_ids"])
        min_new_tokens = max(0, min_len - token_count)
        samples = self._safe_to_smiles(
            self.generate_safe(
                num_samples=1,
                safe_prefix=safe_prefix,
                min_new_tokens=min_new_tokens,
                **kwargs,
            )
        )
        return samples[0] if samples else smiles

    def addmask(self, smiles, num_edit: int = 3, **kwargs):
        del num_edit
        return self.mask_modification(smiles, **kwargs)

    def remask(self, smiles, input_ids=None, **kwargs):
        del input_ids
        return self.mask_modification(smiles, **kwargs)
