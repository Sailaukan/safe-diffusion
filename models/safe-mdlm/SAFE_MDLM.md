# SAFE-MDLM Migration Note

This repository is the SAFE-MDLM workflow built with MDLM diffusion dynamics from `kuleshov-group/discrete-diffusion-guidance`.

The SAFE-specific pieces are intentionally kept aligned for comparison with SAFE-UDLM:

- SAFE tokenizer from `datamol-io/safe-gpt`
- SAFE dataset streaming and custom SAFE text-file loading
- BERT masked-language-model backbone size and training workflow
- de novo, fragment, PMO, and lead-optimization sampler APIs

The model-specific change is the diffusion engine:

- `diffusion.type: absorbing_state`
- `diffusion.parameterization: subs`
- `training.T: 0`
- `model.time_conditioning: False`
- `diffusion.zero_recon_loss: False`

These are the upstream DDG settings for MDLM. Editable spans are represented as `[MASK]` tokens in the latent state, and the SUBS parameterization copies already unmasked tokens while predicting clean tokens only for masked positions.

SAFE-UDLM checkpoints are not architecture-compatible with SAFE-MDLM. Train SAFE-MDLM from scratch, then compare it against SAFE-UDLM using the same data, sampler settings, and evaluation scripts.
