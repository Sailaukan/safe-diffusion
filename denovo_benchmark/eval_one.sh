#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL="${1:?Usage: bash denovo_benchmark/eval_one.sh MODEL [CHECKPOINT]}"
CHECKPOINT="$(require_checkpoint "${MODEL}" "${2:-}")"
MODEL_DIR="$(model_dir "${MODEL}")"
RUN_DIR="$(run_dir "${MODEL}")"

mkdir -p "${RUN_DIR}"

export SAFE_GPT_REVISION
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "${MODEL}" in
  safe-ar|safe-mdlm|safe-udlm)
    "${PYTHON_BIN}" "${BENCHMARK_DIR}/sample_genmol_denovo.py" \
      "--model=${MODEL}" \
      "--model-dir=${MODEL_DIR}" \
      "--metrics-dir=${REPO_ROOT}/models/safe-duo" \
      "--checkpoint=${CHECKPOINT}" \
      "--output=${RUN_DIR}/denovo_metrics.json" \
      "--records-output=${RUN_DIR}/denovo_records.csv" \
      "--num-samples=${SAMPLE_COUNT}" \
      "--batch-size=${SAMPLE_BATCH_SIZE}" \
      "--steps=${SAMPLING_STEPS}" \
      "--temperature=${GENMOL_SOFTMAX_TEMP}" \
      "--randomness=${GENMOL_RANDOMNESS}" \
      "--min-add-len=${GENMOL_MIN_ADD_LEN}" \
      "--seed=${SEED}" \
      "--device=${DEVICE}"
    ;;
  safe-duo)
    cd "${MODEL_DIR}"
    "${PYTHON_BIN}" molecule_benchmark.py sample-denovo \
      "--checkpoint=${CHECKPOINT}" \
      "--output=${RUN_DIR}/denovo_metrics.json" \
      "--records-output=${RUN_DIR}/denovo_records.csv" \
      "--num-samples=${SAMPLE_COUNT}" \
      "--batch-size=${SAMPLE_BATCH_SIZE}" \
      "--steps=${SAMPLING_STEPS}" \
      "--eps=${SAMPLING_EPS}" \
      "--device=${DEVICE}"
    ;;
esac

