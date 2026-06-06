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
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import hydra
import lightning as L
import omegaconf
import torch
from genmol.model import SafeMDLM
from genmol.utils.utils_data import get_dataloader, get_last_checkpoint


def _wandb_sweep_run_id() -> str:
    run_id = os.environ.get('WANDB_RUN_ID')
    if run_id:
        return run_id

    run_id = uuid.uuid4().hex[:8]
    os.environ['WANDB_RUN_ID'] = run_id
    return run_id


def _hydra_run_dir(default_dir: str) -> str:
    if os.environ.get('WANDB_SWEEP_ID'):
        return os.path.join('ckpt', 'wandb_sweeps', _wandb_sweep_run_id())
    return default_dir


def _register_resolvers() -> None:
    omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
    omegaconf.OmegaConf.register_new_resolver('hydra_run_dir', _hydra_run_dir, replace=True)
    omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
    omegaconf.OmegaConf.register_new_resolver('eval', eval, replace=True)
    omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)


def _build_wandb_logger(config):
    if config.wandb.name is None and not os.environ.get('WANDB_SWEEP_ID'):
        return None

    wandb_kwargs = omegaconf.OmegaConf.to_object(config.wandb)
    if wandb_kwargs.get('name') is None:
        wandb_kwargs.pop('name')

    return L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        **wandb_kwargs,
    )


_register_resolvers()


@hydra.main(version_base=None,
    config_path="../configs",
    config_name="base",
)
def train(config):
    if config.get('seed') is not None:
        L.seed_everything(int(config.seed), workers=True)

    wandb_logger = _build_wandb_logger(config)

    if config.training.get('use_bracket_safe'):
        config.model.vocab_size += 2

    model = SafeMDLM(config)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    
    train_dataloader = get_dataloader(config)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=[hydra.utils.instantiate(config.callback)],
        strategy=hydra.utils.instantiate({'_target_': 'lightning.pytorch.strategies.DDPStrategy',
                                          'find_unused_parameters': False}),
        logger=wandb_logger,
        enable_progress_bar=True)
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    

if __name__ == '__main__':
    train()
