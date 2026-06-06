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

import datasets
import torch
from rdkit import RDLogger
from safe.tokenizer import SAFETokenizer

from genmol.utils.bracket_safe_converter import safe2bracketsafe

RDLogger.DisableLog('rdApp.*')


TEXT_COLUMNS = ('input', 'safe', 'text', 'inputs', 'sequence')


def _checkpoint_step(filename: str) -> int | None:
    stem, ext = os.path.splitext(filename)
    if ext != '.ckpt' or not stem.isdigit():
        return None
    return int(stem)


def get_last_checkpoint(save_dir):
    if not os.path.isdir(save_dir):
        return None

    checkpoints = [
        filename
        for filename in os.listdir(save_dir)
        if _checkpoint_step(filename) is not None
    ]
    if not checkpoints:
        return None

    last_filename = max(checkpoints, key=_checkpoint_step)
    return os.path.join(save_dir, last_filename)


def get_tokenizer():
    tokenizer = SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()
    tokenizer.add_tokens(['<', '>'])   # for bracket_safe
    return tokenizer


def get_example_text(example):
    for key in TEXT_COLUMNS:
        value = example.get(key)
        if isinstance(value, str):
            return value

    for key, value in example.items():
        if isinstance(value, str):
            return value

    raise KeyError(
        f'Could not find a string SAFE column in example. '
        f'Available keys: {list(example.keys())}'
    )


class Collator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = bool(config.training.get('use_bracket_safe'))
    
    def __call__(self, examples):
        texts = [get_example_text(example) for example in examples]
        if self.use_bracket_safe:
            texts = [safe2bracketsafe(text) for text in texts]

        batch = self.tokenizer(texts,
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']
        return batch
    

class UserDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(data_path, encoding='utf-8') as f:
            self.safe_list = [line.rstrip('\n') for line in f]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, index):
        return {'input': self.safe_list[index]}


def _common_loader_kwargs(config):
    num_workers = int(config.loader.num_workers)
    return {
        'batch_size': config.loader.batch_size,
        'collate_fn': Collator(config),
        'num_workers': num_workers,
        'pin_memory': config.loader.pin_memory,
        'persistent_workers': num_workers > 0,
    }


def get_dataloader(config):
    loader_kwargs = _common_loader_kwargs(config)
    if config.data == 'safe':
        return torch.utils.data.DataLoader(
            datasets.load_dataset('datamol-io/safe-gpt', streaming=True, split='train'),
            shuffle=False,  # streaming
            **loader_kwargs,
        )

    return torch.utils.data.DataLoader(
        UserDataset(config.data),
        shuffle=True,
        **loader_kwargs,
    )
