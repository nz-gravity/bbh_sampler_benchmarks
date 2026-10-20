#!/usr/bin/env bash
#SBATCH --job-name=bbh-catalogue
#SBATCH --account=oz200
#SBATCH --array=0-99%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-bbh-catalogue-%A_%a.out
#SBATCH --error=slurm-bbh-catalogue-%A_%a.err

# Submit manually from the repository root: sbatch ozstar/catalogue.slurm run_m
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_id=${1:-run_m}
[[ "$run_id" =~ ^run_[a-z]+$ ]] || { echo "Invalid run ID: $run_id" >&2; exit 2; }
injection_id=${SLURM_ARRAY_TASK_ID:?Use a Slurm job array}
[[ "$injection_id" =~ ^[0-9]+$ ]] || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBFS:-${TMPDIR:-/tmp}}/bbh-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint USR1 TERM INT
"$PWD/.venv/bin/python" -m benchmark.batch --run "$run_id" --injection "$injection_id" \
    --npool "${SLURM_CPUS_PER_TASK:-2}" &
child=$!
status=0
while true; do
    wait "$child" && status=0 || status=$?
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
printf -v event 'injection_%03d' "$injection_id"
lock="$PWD/studies/catalogue/$event/.work/$run_id/.running.lock"
if [[ -f "$lock" && "$(cat "$lock")" == "$(printf 'pid=%s\njob=%s' "$child" "$SLURM_JOB_ID")" ]]; then
    rm -- "$lock"
fi
exit "$status"
