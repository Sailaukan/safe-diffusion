#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARK_DIR}/common.sh"

mkdir -p "${RUN_ROOT}"

for model in ${ALL_MODELS}; do
  printf '\n==> Training %s\n' "${model}"
  bash "${BENCHMARK_DIR}/train_one.sh" "${model}"
done

