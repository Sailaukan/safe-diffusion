#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ROOT_SAFE="${ROOT_SAFE:-${PROJECT_ROOT}}"
ROOT_GENMOL="${ROOT_GENMOL:-${PROJECT_ROOT}/../genmol}"
SAFE_PY="${SAFE_PY:-python}"
GENMOL_PY="${GENMOL_PY:-python}"
OUT_DIR="${OUT_DIR:-${ROOT_SAFE}/ckpt/denovo_compare_50k}"

mkdir -p "$OUT_DIR"
export MPLCONFIGDIR="$OUT_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "=== Environment ==="
date
hostname
echo "SAFE python: $SAFE_PY"
echo "GENMOL python: $GENMOL_PY"

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "nvidia-smi not found"
fi

echo "=== SAFE-AR denovo ==="
cd "$ROOT_SAFE"
"$SAFE_PY" scripts/exps/denovo/run.py -c hparams.yaml | tee "$OUT_DIR/safe_ar_denovo.log"

echo "=== GenMol denovo ==="
cd "$ROOT_GENMOL"
"$GENMOL_PY" scripts/exps/denovo/run.py -c hparams_local_50000.yaml | tee "$OUT_DIR/genmol_denovo.log"
