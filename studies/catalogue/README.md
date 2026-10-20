# 100-injection study

The fixed-injection benchmark remains under `results/`. This separate study applies **`run_m`** from `configs/runs.yaml` to the 100 rows in `injections.csv`.

- Same intrinsic priors, detectors, waveform, analysis band and fixed-extrinsic treatment as the benchmark.
- Each injection uses its own parameters, start time and original noise seed equal to `inj_id`. The sampling seed is always 42.
- Saved strain and PSD arrays are reused across samplers and restarts. Preparation validates their checksums, priors, dependency versions and injection likelihoods.
- Row 0 reuses the verified completed `run_m` result, including its full accumulated runtime.
- Completed outputs contain at most 1000 posterior samples, log evidence and its uncertainty, runtime, likelihood-call count and core count.

Prepare and validate the datasets without inference:

```bash
uv run python -m benchmark.batch --prepare
```

Run locally, one injection at a time with two workers. Repeating the command skips verified completed cases and resumes checkpoints with the same worker count:

```bash
uv run python -m benchmark.batch --run run_m --all --npool 2
```

The row-0 measurement was 218 seconds of sampling; 99 similar cases would take about six hours. Other injections may take substantially longer. The low live-point count and short walks are exploratory settings, not validated evidence or posterior accuracy for all injections.

For OzSTAR, run these commands manually from the repository root:

```bash
bash ozstar/setup_env.sh
sbatch ozstar/catalogue.slurm run_m
```

The array allows eight concurrent cases, each with two CPUs, 8 GiB and a two-hour time limit. Completed cases are skipped; resubmitting a case resumes its checkpoint with the same worker count. No packages are installed in batch jobs.

Inspect a case or rebuild the summary:

```bash
uv run python -m benchmark.batch --run run_m --injection 7 --npool 2
uv run python -m benchmark.batch --run run_m --status
```

Outputs are `injection_007/strain.npz`, its settings/checksum JSON, and `injection_007/results/run_m/{posterior.csv,summary.json}`. Logs are under `logs/run_m/`. `summary_run_m.csv` distinguishes completed, running, checkpointed, failed and pending cases; evidence and runtime columns describe completed cases only. Evidence values for different injections should not be interpreted as sampler evidence differences for a common likelihood.

An interrupted local controller resumes on the next `--all` invocation. After a hard kill, remove a stale `.batch_run_m.lock` or case `.running.lock` only after confirming its process has stopped. Cases that failed before a checkpoint require their working directory to be moved aside before restarting.
