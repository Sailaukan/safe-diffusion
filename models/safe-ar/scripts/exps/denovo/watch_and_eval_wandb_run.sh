#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?run id is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-}"
if [[ -n "${CONDA_BIN}" ]]; then
  WANDB_BIN="${WANDB_BIN:-${CONDA_BIN}/wandb}"
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_BIN}/python}"
else
  WANDB_BIN="${WANDB_BIN:-wandb}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
SWEEP_PATH="${SWEEP_PATH:-sajlaukansahnazar-mbzuai/safe-ar-sweeps/v2ujglhn}"
RUN_PATH="${RUN_PATH:-sajlaukansahnazar-mbzuai/safe-ar-sweeps/${RUN_ID}}"

CKPT="${PROJECT_ROOT}/ckpt/wandb_sweeps/${RUN_ID}/checkpoints/10000.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/ckpt/wandb_sweeps/${RUN_ID}/denovo_10000"
LOG="${PROJECT_ROOT}/logs/denovo_${RUN_ID}_watch.log"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

cd "${PROJECT_ROOT}"
echo "[$(date -Is)] watcher started for ${RUN_ID}"
echo "[$(date -Is)] checkpoint: ${CKPT}"

"${WANDB_BIN}" sweep --pause "${SWEEP_PATH}" || true

while [[ ! -s "${CKPT}" ]]; do
  echo "[$(date -Is)] waiting for 10000.ckpt"
  sleep 60
done

echo "[$(date -Is)] found ${CKPT}"
echo "[$(date -Is)] waiting for sweep training process to exit"
while pgrep -af "scripts/train.py" | grep -q "wandb.project=safe-ar-sweeps"; do
  pgrep -af "scripts/train.py" | grep "wandb.project=safe-ar-sweeps" || true
  sleep 30
done

echo "[$(date -Is)] starting denovo evaluation"
"${PYTHON_BIN}" scripts/exps/denovo/eval_chunked.py \
  --model-path "${CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-path "${RUN_PATH}" \
  --num-samples 1000 \
  --chunk-size 100 \
  --softmax-temp 0.5 \
  --randomness 0.5 \
  --min-add-len 40

echo "[$(date -Is)] denovo evaluation complete"
