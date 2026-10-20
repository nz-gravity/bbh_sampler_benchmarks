from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import bilby
import numpy as np
import pandas as pd
import pytest

from benchmark.injection import settings
from benchmark.registry import load_registry, read_json
from benchmark.run import execute_run


def fake_result():
    return SimpleNamespace(meta_data={"run_statistics": {"nlikelihood": 123, "ncores": 1}},
                           sampling_time=timedelta(seconds=2), log_evidence=-16170.0,
                           log_evidence_err=0.1, log_noise_evidence=-16293.358364584721,
                           posterior=pd.DataFrame([settings()["injection"]] * 20))


def test_every_planned_sampler_receives_identical_likelihood(tmp_path, monkeypatch):
    seen = []
    def sampler(**kwargs):
        likelihood = kwargs["likelihood"]
        seen.append((kwargs["sampler"], kwargs["priors"].copy(),
                     [ifo.frequency_domain_strain.copy() for ifo in likelihood.interferometers],
                     likelihood.log_likelihood(parameters=settings()["injection"])))
        assert kwargs["use_ratio"] is True
        assert kwargs["save"] is False
        assert kwargs["resume"] is False
        if kwargs["sampler"] == "dynesty":
            assert kwargs["seed"] == 42
        return fake_result()
    monkeypatch.setattr(bilby, "run_sampler", sampler)
    for run_id, config in load_registry().items():
        if config["status"] == "planned":
            summary = execute_run(run_id, results_root=tmp_path / "results")
            snapshot = read_json(tmp_path / "results" / run_id / "summary.json")
            assert snapshot["config"]["seed"] == 42
            assert not (tmp_path / ".work" / run_id).exists()
            assert summary["likelihood_evaluations"] == 123
            assert summary["sampling_seconds"] == 2.0
            assert snapshot["sampling_seconds"] == 2.0
            assert snapshot["ncores"] == 1
            assert snapshot["invocation_seconds"] >= 0
            with pytest.raises(FileExistsError):
                execute_run(run_id, results_root=tmp_path / "results")
    assert {row[0] for row in seen} == {"dynesty", "bilby_mcmc"}
    for _, priors, arrays, log_likelihood in seen:
        assert priors == seen[0][1]
        assert log_likelihood == seen[0][3]
        for actual, expected in zip(arrays, seen[0][2]):
            np.testing.assert_array_equal(actual, expected)


def test_resume_requires_identical_snapshot_and_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    def interrupted(**kwargs):
        lock = Path(kwargs["outdir"]) / ".running.lock"
        assert lock.read_text().splitlines()[1] == "job=12345"
        (Path(kwargs["outdir"]) / "run_b_resume.pickle").write_text("test checkpoint; never deserialized")
        raise KeyboardInterrupt
    monkeypatch.setattr(bilby, "run_sampler", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute_run("run_b", results_root=tmp_path / "results")
    assert not (tmp_path / ".work/run_b/.running.lock").exists()
    with pytest.raises(FileExistsError, match="--resume"):
        execute_run("run_b", results_root=tmp_path / "results")
    changed = deepcopy(load_registry())
    changed["run_b"]["seed"] = 43
    with pytest.raises(ValueError, match="changed"):
        execute_run("run_b", results_root=tmp_path / "results", resume=True, registry=changed)
    with pytest.raises(ValueError, match="changed"):
        execute_run("run_b", results_root=tmp_path / "results", resume=True, npool=2)
    def resumed(**kwargs):
        assert kwargs["resume"] is True
        return fake_result()
    monkeypatch.setattr(bilby, "run_sampler", resumed)
    monkeypatch.setenv("SLURM_JOB_ID", "67890")
    execute_run("run_b", results_root=tmp_path / "results", resume=True)


def test_imported_runs_cannot_be_executed(tmp_path):
    with pytest.raises(ValueError, match="imported"):
        execute_run("run_a", results_root=tmp_path / "results")


@pytest.mark.parametrize("run_id", [key for key, run in load_registry().items() if run["status"] == "planned"])
def test_registered_options_and_dynesty_rng_are_accepted_without_sampling(run_id, tmp_path):
    from bilby.core.sampler.dynesty import Dynesty
    from bilby.bilby_mcmc.sampler import Bilby_MCMC
    from benchmark.injection import build_likelihood

    likelihood, priors = build_likelihood()
    config = load_registry()[run_id]
    dynesty = config["sampler"] == "dynesty"
    sampler_class = Dynesty if dynesty else Bilby_MCMC
    seed = {"seed": config["seed"]} if dynesty else {}
    # Real initialization verifies constrained prior draws and times likelihood
    # evaluations. soft_init=True skips exactly the path used at job startup.
    sampler = sampler_class(likelihood, priors, outdir=str(tmp_path), label="preflight",
                            **seed, **config["kwargs"])
    if dynesty:
        assert sampler.kwargs["rstate"].bit_generator.state == np.random.default_rng(42).bit_generator.state
