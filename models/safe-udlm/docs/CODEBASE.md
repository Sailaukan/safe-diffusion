# Codebase Map

This repository has three main layers:

- `configs/`: Hydra configuration for training, sampling, callbacks, and optimizer settings.
- `src/genmol/`: the installable SAFE-UDLM Python package. Core model code lives in `model.py`, diffusion logic in `diffusion.py`, sampling in `sampler.py`, and shared helpers under `utils/`.
- `scripts/`: executable entrypoints for training, preprocessing, and experiment evaluation.

Generated artifacts should stay outside the source tree and are ignored by git:

- `ckpt/` for checkpoints and Hydra run directories.
- `logs/` for Slurm/stdout/stderr logs.
- `wandb/` for local W&B metadata.
- `outputs/`, `.hydra/`, `__pycache__/`, and package build metadata.

When adding new experiment code, keep reusable logic in `src/genmol/` and keep scripts thin. Scripts should parse arguments, load configuration, call package APIs, and write outputs.
