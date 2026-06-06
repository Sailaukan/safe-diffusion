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

def clean_checkpoint(checkpoint, accumulate_grad_batches):
    # Copied from BSD-3 https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    fit_loop = checkpoint['loops']['fit_loop']
    optimizer_steps = fit_loop['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
    batch_progress = fit_loop['epoch_loop.batch_progress']

    batch_progress['total']['completed'] = optimizer_steps['total']['completed'] * accumulate_grad_batches
    batch_progress['current']['completed'] = optimizer_steps['current']['completed'] * accumulate_grad_batches
    # _batches_that_stepped tracks global optimizer steps, not local micro-batches.
    fit_loop['epoch_loop.state_dict']['_batches_that_stepped'] = optimizer_steps['total']['completed']


def fast_forward_info(checkpoint):
    # Copied from BSD-3 https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    fit_loop = checkpoint['loops']['fit_loop']
    fast_forward_epochs = fit_loop['epoch_progress']['current']['completed']
    fast_forward_batches = fit_loop['epoch_loop.batch_progress']['current']['completed']
    return fast_forward_epochs, fast_forward_batches
