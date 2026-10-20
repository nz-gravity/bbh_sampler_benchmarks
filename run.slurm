#!/usr/bin/env bash
#SBATCH --job-name=bbh-benchmark
#SBATCH --account=oz200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:USR1@180

set -euo pipefail
run_id=${1:?Usage: sbatch run.slurm run_b [--resume]}
shift
[[ "$run_id" =~ ^run_[a-z]+$ ]] || { echo "Invalid run ID: $run_id" >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
python="$PWD/.venv/bin/python"
[[ -x "$python" ]] || { echo "Run bash ozstar/setup_env.sh on the login node first." >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${JOBFS:-${TMPDIR:-/tmp}}/bbh-matplotlib-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"
printf 'Run: %s | Job: %s | Node: %s | CPUs: %s\n' \
    "$run_id" "${SLURM_JOB_ID:-manual}" "${SLURMD_NODENAME:-unknown}" "${SLURM_CPUS_PER_TASK:-1}"

# Ask the Bilby parent to checkpoint before Slurm reaches the time limit.
# Do not signal the worker pool directly while the parent is writing state.
child=""
checkpoint() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap checkpoint USR1 TERM INT
"$python" -m benchmark.run --run "$run_id" "$@" &
child=$!
status=0
while true; do
    wait "$child" && status=0 || status=$?
    # A trapped signal interrupts wait before checkpointing finishes.
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
# Bilby can exit without running Python finally blocks. Match both the process
# and the Slurm job: different nodes can have identical process IDs.
lock="$PWD/.work/$run_id/.running.lock"
if [[ -n "${SLURM_JOB_ID:-}" && -f "$lock" && "$(cat "$lock")" == "$(printf 'pid=%s\njob=%s' "$child" "$SLURM_JOB_ID")" ]]; then
    rm -- "$lock"
fi
exit "$status"
