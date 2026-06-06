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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import numpy as np
import pandas as pd
import yaml
from rdkit import DataStructs, Chem, RDLogger
from rdkit.Chem import AllChem
from tdc import Oracle, Evaluator

from genmol.sampler import Sampler

RDLogger.DisableLog('rdApp.*')


def resolve_project_path(path):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


TASKS = [
    'linker_design',
    'motif_extension',
    'scaffold_decoration',
    'superstructure_generation',
    'linker_design_onestep',
]


def get_distance(smiles, df):
    if 'MOL' not in df:
        df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]
    
    fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 1024)
    return np.mean(DataStructs.BulkTanimotoSimilarity(fps, df['FPS'].tolist(), returnDistance=True))


def get_sampling_task(demo, config, task):
    if task in ('linker_design', 'scaffold_morphing'):
        return 'linker_design', demo.fragment_linking, config['linker_design']
    if task in ('motif_extension', 'scaffold_decoration', 'superstructure_generation'):
        return task, demo.fragment_completion, config[task]
    if task == 'linker_design_onestep':
        return 'linker_design', demo.fragment_linking_onestep, config['linker_design_onestep']
    raise ValueError(f'Unknown fragment task: {task}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='hparams.yaml')
    config_name = parser.parse_args().config
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_name)
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_samples = config['num_samples']
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    demo = Sampler(resolve_project_path(config['model_path']))
    data = pd.read_csv(resolve_project_path('data/fragments.csv'))

    for task in TASKS:
        data_column, sampling_method, sampling_config = get_sampling_task(demo, config, task)
        validity, uniqueness, diversity, distance, quality = [], [], [], [], []
        for original, fragment in zip(data['smiles'], data[data_column]):
            samples = sampling_method(fragment, num_samples, **sampling_config)
            if len(samples) == 0:
                validity.append(0)
                uniqueness.append(0)
                diversity.append(0)
                distance.append(0)
                quality.append(0)
                continue
            df = pd.DataFrame({'smiles': samples, 'qed': oracle_qed(samples), 'sa': oracle_sa(samples)})
            validity.append(len(df['smiles']) / num_samples)
            df = df.drop_duplicates('smiles')
            uniqueness.append(len(df['smiles']) / len(samples))
            if len(df['smiles']) == 1:
                diversity.append(0)
            else:
                diversity.append(evaluator(df['smiles']))
            distance.append(get_distance(original, df))
            df = df[df['qed'] >= 0.6]
            df = df[df['sa'] <= 4]
            quality.append(len(df) / num_samples)

        print(f'{task}')
        print(f'\tValidity:\t{np.mean(validity)}')
        print(f'\tUniqueness:\t{np.mean(uniqueness)}')
        print(f'\tDiversity:\t{np.mean(diversity)}')
        print(f'\tDistance:\t{np.mean(distance)}')
        print(f'\tQuality:\t{np.mean(quality)}')
        print('-' * 50)


if __name__ == '__main__':
    main()
