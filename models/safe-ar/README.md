# SAFE-AR

SAFE-AR trains an autoregressive molecule generator over SAFE molecular strings. It uses the same SAFE tokenizer, data loader, SMILES conversion, and evaluation scripts as the existing SAFE-MDLM and SAFE-UDLM repositories, but uses a causal next-token objective instead of a diffusion objective.

## Model

- Representation: SAFE strings from `datamol-io/safe-gpt`
- Tokenizer: `safe.tokenizer.SAFETokenizer`
- Objective: predict `x[t + 1]` from `x[:t]`
- Backbone: DDG-style causal transformer configured through `configs/base.yaml`
- Sampling: starts from BOS and samples until EOS or `model.max_position_embeddings`

SAFE-MDLM and SAFE-UDLM checkpoints are not compatible with SAFE-AR. Train SAFE-AR from scratch before comparing the three SAFE models.

## Training

```bash
cd models/safe-ar
export PYTHONPATH="$PWD/src"
python scripts/train.py
```

For the 50k-step batch-256 run:

```bash
sbatch scripts/slurm_train_50k_bs256.sh
```

The default config trains on the streaming `datamol-io/safe-gpt` dataset. To train on a local newline-delimited SAFE file, pass its path as `data=/path/to/file.safe`.

## Generation

```python
from genmol.sampler import Sampler

sampler = Sampler("ckpt/train/checkpoints/50000.ckpt")
smiles = sampler.de_novo_generation(num_samples=100, softmax_temp=0.8, randomness=1.0)
```

Denovo and fragment experiment entrypoints are under `scripts/exps/`.
