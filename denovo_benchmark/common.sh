#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
COLLECT_PYTHON_BIN="${COLLECT_PYTHON_BIN:-${PYTHON_BIN}}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARK_DIR}/runs}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/tmp/data}"

# Keep user-level packages out of benchmark jobs by default. In particular,
# a stale ~/.local torchvision can be discovered ahead of the conda env and
# crash transformers imports with missing torchvision C++ operators.
if [[ "${ALLOW_PYTHON_USER_SITE:-0}" == "1" ]]; then
  unset PYTHONNOUSERSITE
else
  export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
fi

SAFE_GPT_REVISION="${SAFE_GPT_REVISION:-b83175cd7394}"
SEED="${SEED:-1}"

MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
RESUME="${RESUME:-0}"

SAMPLE_COUNT="${SAMPLE_COUNT:-10000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"
SAMPLING_EPS="${SAMPLING_EPS:-1e-3}"
GENMOL_SOFTMAX_TEMP="${GENMOL_SOFTMAX_TEMP:-0.8}"
GENMOL_RANDOMNESS="${GENMOL_RANDOMNESS:-0.5}"
GENMOL_MIN_ADD_LEN="${GENMOL_MIN_ADD_LEN:-40}"
DEVICE="${DEVICE:-auto}"

DUO_ALGO="${DUO_ALGO:-duo_base}"

ALL_MODELS="${ALL_MODELS:-safe-ar safe-mdlm safe-udlm safe-duo}"

model_dir() {
  case "$1" in
    safe-ar|safe-mdlm|safe-udlm|safe-duo)
      printf '%s/models/%s\n' "${REPO_ROOT}" "$1"
      ;;
    *)
      printf 'Unknown model: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

python_bin_for_model() {
  case "$1" in
    safe-ar)
      printf '%s\n' "${SAFE_AR_PYTHON_BIN:-${PYTHON_BIN}}"
      ;;
    safe-mdlm)
      printf '%s\n' "${SAFE_MDLM_PYTHON_BIN:-${PYTHON_BIN}}"
      ;;
    safe-udlm)
      printf '%s\n' "${SAFE_UDLM_PYTHON_BIN:-${PYTHON_BIN}}"
      ;;
    safe-duo)
      printf '%s\n' "${SAFE_DUO_PYTHON_BIN:-${PYTHON_BIN}}"
      ;;
    *)
      printf 'Unknown model: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

run_dir() {
  printf '%s/%s\n' "${RUN_ROOT}" "$1"
}

checkpoint_dir() {
  printf '%s/checkpoints\n' "$(run_dir "$1")"
}

bool_override() {
  if [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]; then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

has_checkpoint() {
  local dir
  dir="$(checkpoint_dir "$1")"
  [[ -d "${dir}" ]] && find "${dir}" -maxdepth 1 -type f -name '*.ckpt' -print -quit | grep -q .
}

latest_checkpoint() {
  local dir
  dir="$(checkpoint_dir "$1")"
  [[ -d "${dir}" ]] || return 0
  if [[ -f "${dir}/last.ckpt" ]]; then
    printf '%s\n' "${dir}/last.ckpt"
    return 0
  fi
  find "${dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V | tail -n 1
}

require_checkpoint() {
  local model="$1"
  local checkpoint="${2:-}"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(latest_checkpoint "${model}")"
  fi
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    printf 'No checkpoint found for %s under %s\n' "${model}" "$(checkpoint_dir "${model}")" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

prepare_run_dir() {
  local model="$1"
  local dir
  dir="$(run_dir "${model}")"
  mkdir -p "${dir}"
  if has_checkpoint "${model}" && [[ "$(bool_override "${RESUME}")" != "true" ]]; then
    printf 'Refusing to train %s because checkpoints already exist in %s.\n' "${model}" "$(checkpoint_dir "${model}")" >&2
    printf 'Set RESUME=1 to continue, or set RUN_ROOT to a fresh directory.\n' >&2
    return 1
  fi
}
