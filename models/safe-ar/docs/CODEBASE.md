# SAFE-AR Codebase

- `configs/base.yaml`: default training, model, sampling, and checkpoint config.
- `src/genmol/model.py`: `SafeAR`, the Lightning module with next-token AR loss and sampling.
- `src/genmol/backbone.py`: causal transformer backbone for SAFE token logits.
- `src/genmol/sampler.py`: checkpoint loading, denovo generation, fragment-prefix generation, and SMILES decoding.
- `src/genmol/utils/`: shared SAFE data, chemistry, checkpoint, and EMA helpers.
- `scripts/train.py`: Hydra training entrypoint.
- `scripts/exps/`: denovo, fragment, lead, and PMO experiment entrypoints reused for SAFE model comparisons.
