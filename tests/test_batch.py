from copy import deepcopy
from datetime import timedelta
import shutil
from types import SimpleNamespace

import bilby
import numpy as np
import pandas as pd
import pytest

from benchmark import batch
from benchmark.injection import DATA, build_likelihood, data_manifest, settings
from benchmark.registry import ROOT, resolve_run, sha256


@pytest.fixture
def study(tmp_path):
    shutil.copyfile(batch.STUDY / "injections.csv", tmp_path / "injections.csv")
    return tmp_path


def test_catalogue_and_original_noise_seeds(study):
    rows = batch.catalogue(study)
    assert len(rows) == 100
    assert batch.injection_config(0, rows) == settings()
    for injection_id in rows.index:
        config = batch.injection_config(injection_id, rows)
        assert config["noise_seed"] == injection_id
        assert config["start_time"] == config["injection"]["geocent_time"] - 2
        assert config["prior_file"] == settings()["prior_file"]
        assert config["maximum_frequency"] == 2048
        assert config["fixed_parameters"] == settings()["fixed_parameters"]
    with pytest.raises(ValueError, match="Unknown injection"):
        batch.injection_config(100, rows)
    rows.iloc[:-1].to_csv(study / "injections.csv")
    with pytest.raises(ValueError, match="exactly"):
        batch.catalogue(study)


def test_prepared_injections_are_distinct_and_reproducible(study, tmp_path):
    first, config0 = batch.prepare_injection(0, study=study)
    second, config1 = batch.prepare_injection(1, study=study)
    assert first.read_bytes() == DATA.read_bytes()
    assert data_manifest(first, config=config0) == data_manifest()
    assert sha256(second) != sha256(first)
    replay = tmp_path / "replay"
    replay.mkdir()
    shutil.copyfile(study / "injections.csv", replay / "injections.csv")
    repeated, _ = batch.prepare_injection(1, study=replay)
    assert repeated.read_bytes() == second.read_bytes()
    likelihood, priors = build_likelihood(second, config=config1)
    assert np.isfinite(likelihood.log_likelihood(parameters=config1["injection"]))
    for name in config1["fixed_parameters"]:
        assert priors[name].peak == config1["injection"][name]
    changed = deepcopy(config1)
    changed["noise_seed"] = 2
    with pytest.raises(ValueError, match="different experiment"):
        data_manifest(second, config=changed)


def test_row_zero_reuses_complete_result_and_preserves_cost(study, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Row zero must reuse verified inference, not rerun it")
    monkeypatch.setattr(bilby, "run_sampler", forbidden)
    summary = batch.run_one(0, "run_m", study=study)
    assert summary["sampling_seconds"] == 218.008574
    assert summary["config"] == resolve_run("run_m")
    output = study / "injection_000/results/run_m"
    for name in ("summary.json", "posterior.csv"):
        assert (output / name).read_bytes() == (ROOT / "results/run_m" / name).read_bytes()
    assert batch.run_one(0, "run_m", study=study) == summary
    records = batch.collect_status("run_m", study=study)
    assert records[0]["status"] == "completed"
    assert records[1]["status"] == "pending"


def test_batch_passes_each_injection_to_shared_runner(study, monkeypatch):
    seen = []
    def sampler(**kwargs):
        seen.append(kwargs)
        config = batch.injection_config(1, batch.catalogue(study))
        assert kwargs["injection_parameters"] == config["injection"]
        assert kwargs["seed"] == 42
        assert kwargs["npool"] == 2
        assert kwargs["save"] is False
        assert kwargs["resume"] is False
        return SimpleNamespace(meta_data={"run_statistics": {"ncores": 2, "nlikelihood": 100}},
                               sampling_time=timedelta(seconds=7), log_evidence=-100,
                               log_evidence_err=0.5, log_noise_evidence=kwargs["likelihood"].noise_log_likelihood(),
                               posterior=pd.DataFrame([config["injection"]] * 20))
    monkeypatch.setattr(bilby, "run_sampler", sampler)
    summary = batch.run_one(1, "run_m", study=study)
    path, config = batch.prepare_injection(1, study=study)
    assert summary["likelihood_id"] == data_manifest(path, config=config)["likelihood_id"]
    assert summary["likelihood_id"] != data_manifest()["likelihood_id"]
    assert summary["sampling_seconds"] == 7
    assert summary["ncores"] == 2
    assert batch.run_one(1, "run_m", study=study) == summary
    assert len(seen) == 1
    assert not (study / "injection_001/.work/run_m").exists()
    assert batch.collect_status("run_m", study=study)[1]["status"] == "completed"


def test_controller_continues_after_failure_and_removes_own_lock(study, monkeypatch):
    rows = batch.catalogue(study).iloc[:2]
    monkeypatch.setattr(batch, "catalogue", lambda study: rows)
    calls = []
    class Child:
        pid = 12345
        def __init__(self, args, **kwargs):
            calls.append(args)
            self.code = int(args[args.index("--injection") + 1]) == 0
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            root = batch.event_directory(study, len(calls) - 1) / ".work/run_m"
            root.mkdir(parents=True)
            (root / ".running.lock").write_text(f"pid={self.pid}\n")
        def wait(self):
            return self.code
        def poll(self):
            return self.code
    monkeypatch.setattr(batch.subprocess, "Popen", Child)
    assert batch.run_all("run_m", study=study) == [0]
    assert len(calls) == 2
    assert not (study / ".batch_run_m.lock").exists()
    assert not list(study.glob("injection_*/.work/run_m/.running.lock"))


def test_changed_completed_settings_are_rejected_before_preparing_data(study, monkeypatch):
    changed = deepcopy(resolve_run("run_m"))
    changed["kwargs"]["walks"] = 21
    monkeypatch.setattr(batch, "resolve_run", lambda run_id: changed)
    with pytest.raises(ValueError, match="configuration changed"):
        batch.run_one(1, "run_m", study=study)
    assert not (study / "injection_001").exists()
