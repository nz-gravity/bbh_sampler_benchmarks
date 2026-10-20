#!/usr/bin/env bash
#SBATCH --job-name=bbh-benchmark
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run --locked python -m benchmark.run --run "${1:?Usage: sbatch run.slurm run_b [--resume]}" "${@:2}"
