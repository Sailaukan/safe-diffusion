# SAFE-AR Model Card

SAFE-AR is a codebase for training autoregressive language models on SAFE molecular strings. This folder does not include a trained checkpoint or benchmarked model release.

## Intended Use

- Train SAFE-AR from scratch on SAFE molecular sequence data.
- Compare SAFE-AR against SAFE-MDLM and SAFE-UDLM under matched tokenizer, data, sampler, and evaluation settings.
- Generate candidate molecules for research workflows that include downstream chemistry validation.

## Out of Scope

Generated molecules are not validated drug candidates. Any generated structure must be checked with domain-specific chemistry, safety, synthesis, and experimental workflows before use.

## Training Data

The default config streams `datamol-io/safe-gpt`. Local newline-delimited SAFE files can be used by overriding `data`.

## Evaluation

No model-quality claims are made until SAFE-AR is trained and evaluated. Use the denovo and fragment scripts in `scripts/exps/` for comparisons with the existing SAFE diffusion models.
