#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL="${1:?Usage: bash denovo_benchmark/train_one.sh MODEL}"
MODEL_DIR="$(model_dir "${MODEL}")"
RUN_DIR="$(run_dir "${MODEL}")"
RESUME_BOOL="$(bool_override "${RESUME}")"

prepare_run_dir "${MODEL}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs"

export SAFE_GPT_REVISION
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

case "${MODEL}" in
  safe-ar|safe-mdlm|safe-udlm)
    cd "${MODEL_DIR}"
    "${PYTHON_BIN}" scripts/train.py \
      "data=safe" \
      "++seed=${SEED}" \
      "hydra.run.dir=${RUN_DIR}" \
      "loader.global_batch_size=${GLOBAL_BATCH_SIZE}" \
      "loader.batch_size=${MICRO_BATCH_SIZE}" \
      "loader.num_workers=${NUM_WORKERS}" \
      "trainer.max_steps=${MAX_STEPS}" \
      "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}" \
      "callback.dirpath=${RUN_DIR}/checkpoints" \
      "callback.every_n_train_steps=${CHECKPOINT_INTERVAL}"
    ;;
  safe-duo)
    cd "${MODEL_DIR}"
    "${PYTHON_BIN}" main.py \
      "mode=train" \
      "seed=${SEED}" \
      "data=safe-gpt-v1" \
      "model=safe_bert_size" \
      "algo=${DUO_ALGO}" \
      "wandb=null" \
      "hydra.run.dir=${RUN_DIR}/hydra" \
      "checkpointing.save_dir=${RUN_DIR}" \
      "checkpointing.resume_from_ckpt=${RESUME_BOOL}" \
      "checkpointing.resume_ckpt_path=${RUN_DIR}/checkpoints/last.ckpt" \
      "data.cache_dir=${DATA_CACHE_DIR}" \
      "data.tokenizer_revision=${SAFE_GPT_REVISION}" \
      "data.train_revision=${SAFE_GPT_REVISION}" \
      "data.valid_revision=${SAFE_GPT_REVISION}" \
      "loader.global_batch_size=${GLOBAL_BATCH_SIZE}" \
      "loader.eval_global_batch_size=${GLOBAL_BATCH_SIZE}" \
      "loader.batch_size=${MICRO_BATCH_SIZE}" \
      "loader.eval_batch_size=${SAMPLE_BATCH_SIZE}" \
      "loader.num_workers=${NUM_WORKERS}" \
      "trainer.max_steps=${MAX_STEPS}" \
      "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}" \
      "trainer.val_check_interval=${CHECKPOINT_INTERVAL}" \
      "callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_INTERVAL}"
    ;;
esac

