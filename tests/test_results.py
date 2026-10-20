import numpy as np
import pandas as pd
import pytest

from benchmark.registry import resolve_run
from benchmark.results import PARAMETERS, load_result, save_result


def test_thinning_is_deterministic_lossless_and_does_not_change_evidence(tmp_path):
    points = np.random.default_rng(9).normal(size=(2000, 8))
    posterior = pd.DataFrame(points, columns=PARAMETERS)
    metrics = {"log_evidence": -16170.605700338061, "log_evidence_err": 0.07434949879178701}
    for name in ("first", "second"):
        save_result(tmp_path / name, posterior, resolve_run("run_a"), metrics)
    summary, actual = load_result(tmp_path / "first")
    indices = np.sort(np.random.default_rng(0).choice(2000, 1000, replace=False))
    np.testing.assert_array_equal(actual.to_numpy(), points[indices])
    assert summary["log_evidence"] == metrics["log_evidence"]
    assert summary["log_evidence_err"] == metrics["log_evidence_err"]
    assert (tmp_path / "first/posterior.csv").read_bytes() == (tmp_path / "second/posterior.csv").read_bytes()
    assert {p.name for p in (tmp_path / "first").iterdir()} == {"posterior.csv", "summary.json"}
    with pytest.raises(FileExistsError):
        save_result(tmp_path / "first", posterior, resolve_run("run_a"), metrics)


def test_small_posteriors_are_not_upsampled(tmp_path):
    posterior = {name: np.arange(12, dtype=float) for name in PARAMETERS}
    save_result(tmp_path / "small", posterior, resolve_run("run_i"), {"log_evidence": -10, "log_evidence_err": 0.2})
    summary, actual = load_result(tmp_path / "small")
    assert summary["posterior_rows"] == len(actual) == 12
