#!/usr/bin/env bash
#SBATCH --job-name=safe_mdlm_50k
#SBATCH -N 1
#SBATCH --ntasks=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

srun -N1 -n1 "${SCRIPT_DIR}/train_50k_bs256.sh"
