import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from benchmark.registry import ROOT, load_registry


@pytest.fixture
def cluster_checkout(tmp_path):
    repo = tmp_path / "cluster checkout"
    for name in ("ozstar/submit.sh", "ozstar/catalogue.slurm", "run.slurm", "configs/runs.yaml", "benchmark/__init__.py", "benchmark/registry.py"):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    return repo


def invoke(repo, *args, env=None):
    return subprocess.run(["bash", str(repo / "ozstar/submit.sh"), *args],
                          env={**os.environ, **(env or {})}, capture_output=True, text=True)


def test_default_preview_and_explicit_mcmc(cluster_checkout):
    result = invoke(cluster_checkout)
    assert result.returncode == 0, result.stderr
    planned = [name for name, run in load_registry().items()
               if run["status"] == "planned" and run["sampler"] == "dynesty"]
    assert len(result.stdout.splitlines()) == len(planned)
    for name in planned:
        assert f"bbh-{name}" in result.stdout
    assert "run_j" not in result.stdout
    assert "--account=oz200" in result.stdout
    assert invoke(cluster_checkout, "run_j").returncode == 0
    assert invoke(cluster_checkout, "run_f").returncode != 0


def test_selection_is_validated_before_any_submission(cluster_checkout, tmp_path):
    capture = tmp_path / "submissions.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(f'''#!{sys.executable}
import json,sys
with open({str(capture)!r}, "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
print("12345")
''')
    sbatch.chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "BBH_CPUS": "8", "SLURM_ACCOUNT": "test-account"}
    invalid = invoke(cluster_checkout, "--submit", "run_b", "unknown", env=env)
    assert invalid.returncode != 0
    assert not capture.exists()
    result = invoke(cluster_checkout, "--submit", "run_b", env=env)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert "--cpus-per-task=8" in args
    assert "--account=test-account" in args
    assert args[-1] == "run_b"


def test_completed_jobs_are_skipped_and_resume_is_explicit(cluster_checkout):
    (cluster_checkout / "results/run_b").mkdir(parents=True)
    result = invoke(cluster_checkout)
    assert result.returncode == 0
    assert "bbh-run_b" not in result.stdout
    assert invoke(cluster_checkout, "run_b").returncode != 0
    work = cluster_checkout / ".work/run_c"
    work.mkdir(parents=True)
    (work / "run_c_resume.pickle").write_bytes(b"test checkpoint; never loaded")
    assert invoke(cluster_checkout, "run_c").returncode != 0
    resumed = invoke(cluster_checkout, "--resume", "run_c")
    assert resumed.returncode == 0
    assert resumed.stdout.rstrip().endswith("--resume")
    assert invoke(cluster_checkout, "--resume", "run_d").returncode != 0


@pytest.mark.parametrize("foreign_lock", [False, True])
@pytest.mark.parametrize("job_script", ["run.slurm", "ozstar/catalogue.slurm"])
def test_job_forwards_checkpoint_signal_and_only_clears_its_own_lock(cluster_checkout, foreign_lock, job_script):
    relative_work = "studies/catalogue/injection_007/.work/run_b" if "catalogue" in job_script else ".work/run_b"
    work = cluster_checkout / relative_work
    python = cluster_checkout / ".venv/bin/python"
    python.write_text(f'''#!{sys.executable}
import os,signal,time
from pathlib import Path
work=Path({relative_work!r})
work.mkdir(parents=True)
job="different-job" if {foreign_lock!r} else os.environ["SLURM_JOB_ID"]
(work/".running.lock").write_text(f"pid={{os.getpid()}}\\njob={{job}}\\n")
def checkpoint(*args):
    (work/"run_b_resume.pickle").write_bytes(b"checkpoint written")
    os._exit(130)
signal.signal(signal.SIGTERM,checkpoint)
Path("ready").touch()
while True:time.sleep(0.01)
''')
    python.chmod(0o755)
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(cluster_checkout), "SLURM_JOB_ID": "12345",
           "SLURM_CPUS_PER_TASK": "4", "SLURM_ARRAY_TASK_ID": "7", "JOBFS": str(cluster_checkout / "scratch")}
    process = subprocess.Popen(["bash", str(cluster_checkout / job_script), "run_b"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (cluster_checkout / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (cluster_checkout / "ready").exists()
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert (work / "run_b_resume.pickle").read_bytes() == b"checkpoint written"
        assert (work / ".running.lock").exists() == foreign_lock
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
