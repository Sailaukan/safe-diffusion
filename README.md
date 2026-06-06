# SAFE Diffusion

Single-repository workspace for several SAFE and discrete-diffusion model codebases.

## Models

| Path | Model | Purpose |
| --- | --- | --- |
| `models/safe-mdlm/` | SAFE-MDLM | SAFE molecule generation with masked discrete diffusion. |
| `models/safe-udlm/` | SAFE-UDLM | SAFE molecule generation with uniform discrete diffusion. |
| `models/safe-ar/` | SAFE-AR | SAFE molecule generation with an autoregressive objective. |
| `models/safe-duo/` | SAFE-Duo | Duo, Duo++, and diffusion baseline experiments for text/image generation. |

## How To Use

Each model keeps its original layout, README, dependencies, scripts, configs, and licenses. Work from the model directory you need:

```bash
cd models/safe-mdlm
bash env/setup.sh
```

For `safe-duo`, follow its local README:

```bash
cd models/safe-duo
pip install -r requirements.txt
```

Do not mix environments unless you have checked dependency compatibility. The model repositories pin different versions of PyTorch, Lightning, Transformers, and related packages.

## Repository Layout

```text
models/
  safe-mdlm/
  safe-udlm/
  safe-ar/
  safe-duo/
UPSTREAMS.md
README.md
```

## Provenance

The imported source repositories and commit hashes are listed in `UPSTREAMS.md`.
# safe-diffusion
