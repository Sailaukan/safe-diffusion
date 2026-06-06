#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/safe-mdlm-mpl-${SLURM_JOB_ID:-manual}
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${MPLCONFIGDIR}"

echo "[$(date -Is)] SAFE-MDLM training start"
echo "host=$(hostname) slurm_job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  scripts/train.py \
  hydra.run.dir=ckpt/train_50k_bs256 \
  wandb.name=safe-mdlm_50k_bs256 \
  trainer.devices=1 \
  loader.global_batch_size=256 \
  loader.batch_size=16 \
  trainer.accumulate_grad_batches=16 \
  trainer.max_steps=50000 \
  loader.num_workers=8 \
  callback.every_n_train_steps=5000

echo "[$(date -Is)] SAFE-MDLM training finished"
