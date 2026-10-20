"""Repeat a registered sampler over the separate 100-injection catalogue."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np
import pandas as pd

from .compare import write_csv
from .injection import DATA, build_likelihood, data_manifest, generate_data, settings
from .registry import ROOT, resolve_run
from .results import load_result
from .run import execute_run

STUDY = ROOT / "studies/catalogue"


def registered_run(run_id):
    resolved = resolve_run(run_id)
    if resolved["status"] != "planned" or resolved["seed"] != 42:
        raise ValueError("Select a reproducible planned run with sampling seed 42")
    source = ROOT / "results" / run_id
    if source.exists():
        summary, _ = load_result(source)
        if summary["config"] != resolved:
            raise ValueError("Completed run configuration changed; create a new run ID")
    return resolved


def catalogue(study=STUDY):
    # Match the original parser, including its floating-point roundoff.
    rows = pd.read_csv(Path(study) / "injections.csv", index_col="inj_id")
    if rows.index.tolist() != list(range(100)) or set(rows.columns) != set(settings()["injection"]):
        raise ValueError("Expected exactly injections 0–99 and the canonical injection columns")
    if not np.all(np.isfinite(rows.to_numpy(dtype=float))):
        raise ValueError("Injection parameters must be finite")
    return rows


def injection_config(injection_id, rows):
    if injection_id not in rows.index:
        raise ValueError(f"Unknown injection: {injection_id}")
    config = deepcopy(settings())
    config["injection"] = rows.loc[injection_id].to_dict()
    config["noise_seed"] = int(injection_id)
    config["start_time"] = config["injection"]["geocent_time"] - 2
    return config


def event_directory(study, injection_id):
    return Path(study) / f"injection_{injection_id:03d}"


def prepare_injection(injection_id, *, study=STUDY, rows=None):
    rows = catalogue(study) if rows is None else rows
    config = injection_config(injection_id, rows)
    path = event_directory(study, injection_id) / "strain.npz"
    if path.exists() or path.with_suffix(".json").exists():
        data_manifest(path, config=config)
    elif config == settings():
        # Preserve the exact archived data for row zero on every platform.
        data_manifest()
        path.parent.mkdir(parents=True, exist_ok=True)
        for source, target in ((DATA, path), (DATA.with_suffix(".json"), path.with_suffix(".json"))):
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    else:
        generate_data(path, config=config)
    return path, config


def completed_result(directory, run_id, data_path, config):
    summary, _ = load_result(directory)
    manifest = data_manifest(data_path, config=config)
    if summary["config"] != resolve_run(run_id) or summary["likelihood_id"] != manifest["likelihood_id"]:
        raise ValueError(f"Result configuration or injection mismatch: {directory}")
    if summary["evidence_convention"] != "full_log_likelihood" or not np.isclose(
            summary["log_noise_evidence"], manifest["noise_log_likelihood"], rtol=0, atol=1e-7):
        raise ValueError(f"Result evidence convention mismatch: {directory}")
    return summary


def run_one(injection_id, run_id, *, study=STUDY, npool=2):
    registered_run(run_id)
    data_path, config = prepare_injection(injection_id, study=study)
    root = event_directory(study, injection_id)
    directory = root / "results" / run_id
    if directory.exists():
        return completed_result(directory, run_id, data_path, config)
    source = ROOT / "results" / run_id
    if config == settings() and source.exists():
        summary = completed_result(source, run_id, data_path, config)
        directory.mkdir(parents=True)
        for name in ("summary.json", "posterior.csv"):
            shutil.copyfile(source / name, directory / name)
        print(f"Injection {injection_id}: reused verified {run_id}", flush=True)
        return summary
    work = root / ".work" / run_id
    resume = any(work.glob("*_resume.pickle"))
    return execute_run(run_id, npool=npool, resume=resume, results_root=root / "results",
                       data_path=data_path, injection_config=config)


def collect_status(run_id, *, study=STUDY):
    rows = catalogue(study)
    records = []
    for injection_id in rows.index:
        root = event_directory(study, injection_id)
        record = {"injection_id": int(injection_id), "run_id": run_id, "status": "pending"}
        directory = root / "results" / run_id
        work = root / ".work" / run_id
        if directory.exists():
            try:
                summary = completed_result(directory, run_id, root / "strain.npz", injection_config(injection_id, rows))
                record.update(status="completed", **{key: summary.get(key) for key in (
                    "log_evidence", "log_evidence_err", "sampling_seconds", "invocation_seconds",
                    "likelihood_evaluations", "ncores", "posterior_rows")})
            except (OSError, ValueError, KeyError) as error:
                record.update(status="invalid", note=str(error))
        elif (work / ".running.lock").exists():
            record["status"] = "running"
        elif any(work.glob("*_resume.pickle")):
            record["status"] = "checkpointed"
        elif work.exists():
            record["status"] = "failed"
        records.append(record)
    write_csv(Path(study) / f"summary_{run_id}.csv", records)
    return records


def run_all(run_id, *, study=STUDY, npool=2):
    """One analysis at a time, each in a fresh process; retain logs and checkpoints."""
    registered_run(run_id)
    rows = catalogue(study)
    study = Path(study).resolve()
    lock = study / f".batch_{run_id}.lock"
    with lock.open("x") as stream:
        stream.write(f"pid={os.getpid()}\n")
    failures = []
    child = None
    injection_id = None

    def clear_child_lock():
        if child is None or child.poll() is None:
            return
        running = event_directory(study, injection_id) / ".work" / run_id / ".running.lock"
        if running.exists() and running.read_text().strip() == f"pid={child.pid}":
            running.unlink()

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        raise SystemExit(130)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        logs = study / "logs" / run_id
        logs.mkdir(parents=True, exist_ok=True)
        for injection_id in rows.index:
            log = logs / f"injection_{injection_id:03d}.log"
            with log.open("a") as stream:
                environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                               "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
                child = subprocess.Popen([
                    sys.executable, "-m", "benchmark.batch", "--study", str(study),
                    "--run", run_id, "--injection", str(injection_id), "--npool", str(npool),
                ], stdout=stream, stderr=subprocess.STDOUT, env=environment)
                print(f"Injection {injection_id:03d}: pid={child.pid}; {log}", flush=True)
                code = child.wait()
                # Bilby can exit without Python finally blocks. Clear only the
                # lock belonging to the stopped child, never another worker.
                clear_child_lock()
                if code:
                    failures.append(int(injection_id))
                collect_status(run_id, study=study)
                print(f"Injection {injection_id:03d}: exit={code}", flush=True)
    finally:
        clear_child_lock()
        lock.unlink()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run_m")
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--npool", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--injection", type=int)
    actions.add_argument("--all", action="store_true")
    actions.add_argument("--prepare", action="store_true", help="Save and validate data for all 100 injections; no inference")
    actions.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        registered_run(args.run)
        if args.npool < 1:
            raise ValueError("npool must be positive")
        if args.prepare:
            rows = catalogue(args.study)
            for injection_id in rows.index:
                path, config = prepare_injection(injection_id, study=args.study, rows=rows)
                likelihood, priors = build_likelihood(path, config=config)
                priors.sample_subset_constrained_as_array(priors.non_fixed_keys, size=1)
                if not np.isfinite(likelihood.log_likelihood(parameters=config["injection"])):
                    raise ValueError(f"Non-finite injection likelihood: {injection_id}")
                print(f"Prepared injection {injection_id:03d}", flush=True)
        elif args.status:
            records = collect_status(args.run, study=args.study)
            print(pd.Series([row["status"] for row in records]).value_counts().to_string())
            print(f"Saved {args.study / f'summary_{args.run}.csv'}")
        elif args.all:
            failures = run_all(args.run, study=args.study, npool=args.npool)
            if failures:
                parser.exit(1, f"Failed injections (see logs): {failures}\n")
        else:
            run_one(args.injection, args.run, study=args.study, npool=args.npool)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
