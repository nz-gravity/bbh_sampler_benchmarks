#!/usr/bin/env bash
# Run manually on the OzSTAR/NT login node before submitting jobs.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
command -v uv >/dev/null || { echo "uv must be available on PATH before setup." >&2; exit 2; }
export UV_PROJECT_ENVIRONMENT="$repo/.venv"
uv sync --locked --python 3.13 --no-dev
# Validate the committed strain, PSDs, priors and dependency versions. Do not
# regenerate the noise, or run any sampler, on the login node.
"$repo/.venv/bin/python" -m benchmark.injection
printf '\nEnvironment and fixed dataset checked. Preview jobs with:\n  bash ozstar/submit.sh\n'
