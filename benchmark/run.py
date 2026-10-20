"""Run or list configurations for the shared BBH experiment."""

import argparse
import os
import math
from pathlib import Path
import time

from .injection import build_likelihood, data_manifest, settings
from .registry import ROOT, load_registry, read_json, resolve_run, run_table, update_readme, write_json
from .results import save_result


def result_metrics(result):
    stats = result.meta_data.get("run_statistics", {})
    elapsed = result.sampling_time
    if hasattr(elapsed, "total_seconds"):
        elapsed = elapsed.total_seconds()
    def available(value, cast=float):
        return cast(value) if value is not None and math.isfinite(value) else None
    return {
        "log_evidence": available(result.log_evidence),
        "log_evidence_err": available(result.log_evidence_err),
        "log_noise_evidence": available(result.log_noise_evidence),
        "sampling_seconds": available(elapsed),
        "likelihood_evaluations": available(getattr(result, "num_likelihood_evaluations", None) or stats.get("nlikelihood"), int),
        "ncores": available(stats.get("ncores"), int),
    }


def execute_run(run_id, *, resume=False, npool=1, results_root=None, registry=None):
    """Sample in temporary working output, then keep only a compact completed result."""
    import bilby
    import numpy as np
    import shutil

    resolved = resolve_run(run_id, registry)
    if resolved["seed"] is None:
        raise ValueError(f"{run_id} has an unknown imported seed; copy it to a new run ID with an explicit seed")
    if npool < 1:
        raise ValueError("npool must be positive")
    results_root = Path(results_root or ROOT / "results")
    directory = results_root / run_id
    if directory.exists():
        raise FileExistsError(f"{run_id} already has output; create a new run ID")
    work = results_root.parent / ".work" / run_id
    config = {"run": resolved, "likelihood_id": data_manifest()["likelihood_id"], "npool": npool}
    snapshot = work / "config.json"
    if snapshot.exists():
        if read_json(snapshot) != config:
            raise ValueError("Run configuration, likelihood or resources changed; create a new run ID")
        if not resume:
            raise FileExistsError(f"{run_id} was started; use --resume")
        if not list(work.glob("*_resume.pickle")):
            raise FileNotFoundError(f"No checkpoint to resume for {run_id}")
    elif work.exists() and any(work.iterdir()):
        raise FileExistsError(f"Unrecognized working output: {work}")
    elif resume:
        raise FileNotFoundError(f"No checkpoint to resume for {run_id}")
    work.mkdir(parents=True, exist_ok=True)
    lock = work / ".running.lock"
    with lock.open("x") as stream:
        stream.write(f"pid={os.getpid()}\n")
        if job_id := os.environ.get("SLURM_JOB_ID"):
            stream.write(f"job={job_id}\n")
    try:
        if not snapshot.exists():
            write_json(snapshot, config)
        likelihood, priors = build_likelihood()
        bilby.core.utils.random.seed(resolved["seed"])
        np.random.seed(resolved["seed"])
        sampler_seed = {"seed": resolved["seed"]} if resolved["sampler"] == "dynesty" else {}
        start = time.perf_counter()
        result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler=resolved["sampler"],
            outdir=str(work), label=run_id, npool=npool, save=False,
            resume=resume, use_ratio=True, injection_parameters=settings()["injection"],
            **sampler_seed, **resolved["kwargs"],
        )
        metrics = {**result_metrics(result), "invocation_seconds": time.perf_counter() - start}
        summary = save_result(directory, result.posterior, resolved, metrics)
    finally:
        lock.unlink()
    shutil.rmtree(work)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run")
    actions.add_argument("--list", action="store_true")
    actions.add_argument("--update-readme", action="store_true", help="Refresh the run table from configs/runs.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--npool", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()
    try:
        registry = load_registry()
        if args.list:
            print(run_table(registry))
        elif args.update_readme:
            update_readme()
        else:
            execute_run(args.run, resume=args.resume, npool=args.npool, registry=registry)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
