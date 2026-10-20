# Single-BBH sampler benchmark

Compare initial posterior samplers with **Bilby + Dynesty, 4000 live points** (`run_a`), using one fixed injection, dataset and set of priors.

## Injection

[configs/bbh.yaml](configs/bbh.yaml) contains the exact injection and likelihood settings. [configs/bbh.prior](configs/bbh.prior) contains the priors.

- H1 + L1; IMRPhenomPv2; 4 seconds sampled at 4096 Hz.
- Analysis band **20–2048 Hz**, matching `run_a`; reference frequency 20 Hz.
- Eight intrinsic parameters sampled; extrinsics fixed. No phase, time or distance marginalization.
- Every run loads the identical strain and PSD arrays from [data/bbh_strain.npz](data/bbh_strain.npz). The data settings, checksum and dependency versions are in [data/bbh_strain.json](data/bbh_strain.json).

The dataset and prior file are committed. `uv.lock` pins the environment. Tests regenerate the dataset from the fixed noise seed and require **byte-identical data** and matching priors. Sampler seeds never change the data.

## Configurations

All sampler settings are in [configs/runs.yaml](configs/runs.yaml). Refresh this table with `uv run python -m benchmark.run --update-readme`.

<!-- RUNS:START -->
| Run | Sampler | nlive | Exploration | Stopping | Seed | Status |
|---|---|---:|---|---|---|---|
| run_a | dynesty | 4000 | rwalk, bound=live, nact=20, walks=100 | dlogz=0.1 | unknown | imported |
| run_b | dynesty | 1000 | act-walk, bound=live, nact=2, walks=100 | dlogz=0.5 | 42 | planned |
| run_c | dynesty | 500 | act-walk, bound=live, nact=2, walks=100 | dlogz=0.5 | 42 | planned |
| run_d | dynesty | 250 | act-walk, bound=live, nact=2, walks=100 | dlogz=0.5 | 42 | planned |
| run_e | dynesty | 500 | rslice, bound=multi, slices=3 | dlogz=0.5 | 42 | planned |
| run_f | bilby_mcmc | — | 8 temperatures, gwA | nsamples=2000 | unknown | imported |
| run_g | dynesty | 2000 | rwalk, bound=live, nact=20, walks=100 | dlogz=0.1 | unknown | imported |
| run_h | dynesty | 1000 | rslice, bound=live, slices=3 | dlogz=0.1 | unknown | imported |
| run_i | dynesty | 100 | rwalk, bound=live, nact=20, walks=100 | dlogz=0.1 | unknown | imported |
| run_j | bilby_mcmc | — | 8 temperatures, gwA | nsamples=2000 | 42 | planned |
<!-- RUNS:END -->

Existing results have unknown sampler seeds; these are explicitly `null`. To repeat a configuration, copy it to a new run ID, set `status: planned` and choose an integer seed. Completed runs are never overwritten.

## Run

```bash
uv sync --locked --dev
uv run python -m benchmark.injection --check
uv run python -m benchmark.run --list
uv run python -m benchmark.run --run run_b
```

Use `--npool 4` for four workers, or `sbatch run.slurm run_b` on a cluster. Resume an interrupted run with `--run run_b --resume`. Checkpoints stay in ignored `.work/` directories until completion, then are removed. After a hard kill, remove `.running.lock` only after confirming the job has stopped.

## Small results for Git

Each completed run keeps only:

```text
results/run_a/
├── posterior.csv   # Up to 1000 equal-weight posterior samples
└── summary.json    # lnZ, uncertainty, cost, configuration and likelihood identity
```

Posterior thinning uses a fixed random seed and samples without replacement. Runs with fewer than 1000 samples retain all of them. Log evidence and its uncertainty are taken from the **full inference**, before thinning. Full Bilby result files are not retained. The CSV and JSON files are included by `.gitignore` and ready to commit.

## Compare combinations

```bash
uv run python -m benchmark.compare --reference run_a
uv run python -m benchmark.compare --reference run_a --run run_f --run run_h
uv run pytest
```

Repeat `--run` to choose any combination. Comparisons read only the compact files and save an overlaid eight-parameter `corner.png`, evidence/cost plot, and `summary.csv` under `plots/against_run_a/`. Selected combinations have separate subdirectories. These portable samples and summaries can also drive a future docs page.

Evidence differences require matching data, priors and likelihood conventions. Joint-space support screens supplement the corner contours. Thinning can discard rare modes; the retained rows are not a convergence test.
