# SAFE-MDLM Model Card

SAFE-MDLM is a codebase for training masked discrete diffusion models on SAFE molecular strings. This folder does not include a trained SAFE-MDLM checkpoint or benchmarked model release.

## Intended Use

- Train SAFE-MDLM from scratch on SAFE molecular sequence data.
- Compare SAFE-MDLM against SAFE-UDLM under matched tokenizer, data, sampler, and evaluation settings.
- Run de novo generation, fragment-conditioned generation, PMO, and lead-optimization workflows after training a checkpoint.

## Model Family

- Representation: SAFE molecular strings.
- Tokenizer: `datamol-io/safe-gpt` via `safe-mol`.
- Diffusion: absorbing-state masked discrete diffusion.
- Parameterization: SUBS.
- Time conditioning: disabled, following the MDLM setting in `kuleshov-group/discrete-diffusion-guidance`.

## Limitations

No model-quality claims are made until SAFE-MDLM is trained and evaluated. SAFE-UDLM checkpoints are not architecture-compatible with this code path.
