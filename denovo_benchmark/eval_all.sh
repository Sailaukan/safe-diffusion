#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARK_DIR}/common.sh"

for model in ${ALL_MODELS}; do
  printf '\n==> Evaluating %s\n' "${model}"
  bash "${BENCHMARK_DIR}/eval_one.sh" "${model}"
done

"${PYTHON_BIN}" "${BENCHMARK_DIR}/collect_results.py" \
  "--run-root=${RUN_ROOT}" \
  "--output=${RUN_ROOT}/summary.json" \
  ${ALL_MODELS}

