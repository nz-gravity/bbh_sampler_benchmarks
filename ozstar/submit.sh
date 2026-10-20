#!/usr/bin/env bash
# Preview by default. Execute this script manually on the login node.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
submit=0
resume=0
requested=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) submit=1 ;;
        --resume) resume=1 ;;
        --help|-h)
            echo "Usage: bash ozstar/submit.sh [--submit] [--resume] [run_id ...]"
            echo "Default: preview unfinished planned Dynesty runs. Explicit IDs may also select MCMC."
            echo "Resources: SLURM_ACCOUNT=oz200 BBH_CPUS=4 BBH_MEM=8G BBH_TIME=24:00:00"
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) requested+=("$1") ;;
    esac
    shift
done
python="$repo/.venv/bin/python"
[[ -x "$python" ]] || { echo "Run bash ozstar/setup_env.sh first." >&2; exit 2; }
# Validate the entire selection before submitting any jobs.
selected=$("$python" - "$resume" ${requested[@]+"${requested[@]}"} <<'PY'
from pathlib import Path
import sys
from benchmark.registry import load_registry, resolve_run

registry = load_registry()
resume = sys.argv[1] == "1"
explicit = sys.argv[2:]
selected = explicit or [name for name, run in registry.items()
                        if run["status"] == "planned" and run["sampler"] == "dynesty"]
valid = []
for name in dict.fromkeys(selected):
    run = resolve_run(name, registry)
    if run["status"] != "planned":
        raise SystemExit(f"{name} already has imported results; use a new run ID")
    if (Path("results") / name).exists():
        if explicit:
            raise SystemExit(f"{name} already has output; refusing to overwrite")
        print(f"Skipping {name}: output already exists", file=sys.stderr)
        continue
    work = Path(".work") / name
    if work.exists() and not resume:
        raise SystemExit(f"{name} has working output; select it explicitly with --resume")
    if resume and not list(work.glob("*_resume.pickle")):
        raise SystemExit(f"{name} has no checkpoint to resume")
    valid.append(name)
print("\n".join(valid))
PY
)
if [[ -z "$selected" ]]; then echo "No new Dynesty jobs to submit."; exit 0; fi
if [[ "$submit" == 1 ]]; then command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on the cluster login node." >&2; exit 2; }; fi
while IFS= read -r run_id; do
    command=(sbatch --parsable --export=ALL --chdir="$repo" --job-name="bbh-$run_id"
        --account="${SLURM_ACCOUNT:-oz200}" --cpus-per-task="${BBH_CPUS:-4}"
        --mem="${BBH_MEM:-8G}" --time="${BBH_TIME:-24:00:00}" "$repo/run.slurm" "$run_id")
    if [[ "$resume" == 1 ]]; then command+=(--resume); fi
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$submit" == 1 ]]; then "${command[@]}"; fi
done <<< "$selected"
