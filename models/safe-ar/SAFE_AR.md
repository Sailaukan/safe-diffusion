# SAFE-AR Migration Note

This repository is the SAFE-AR workflow built for comparison with SAFE-MDLM and SAFE-UDLM.

The shared SAFE-specific pieces are intentionally aligned across the three repositories:

- SAFE tokenizer from `datamol-io/safe-gpt`
- SAFE dataset streaming and local SAFE text-file loading
- optional bracket SAFE conversion
- SAFE-to-SMILES decoding and the existing experiment scripts

The AR-specific settings are:

- `diffusion.engine: ar`
- `diffusion.parameterization: ar`
- `model.time_conditioning: False`

Training uses standard next-token negative log likelihood:

```text
input:  BOS x0 x1 ... xN
target: x0  x1 ... xN EOS
```

Padded targets are excluded from the loss. Sampling starts from BOS, bans PAD/BOS/MASK from normal generation, and stops when EOS is sampled or the configured maximum sequence length is reached.
