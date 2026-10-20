import numpy as np

from benchmark.compare import compare, features, joint_diagnostics
from benchmark.registry import read_json, resolve_run, write_json
from benchmark.results import PARAMETERS, save_result


def sample_result(directory, run_id="run_a", error=0.1):
    points = np.random.default_rng(5).normal(size=(40, 8))
    posterior = {name: points[:, i] for i, name in enumerate(PARAMETERS)}
    return save_result(directory, posterior, resolve_run(run_id), {
        "log_evidence": -16170, "log_evidence_err": error,
        "log_noise_evidence": -16293.358364584721,
    })


def test_joint_screen_detects_missing_correlated_mode():
    rng = np.random.default_rng(23)
    low, high = rng.normal(-3, 0.1, (1200, 8)), rng.normal(3, 0.1, (1200, 8))
    reference = np.concatenate([low, high])
    missing, _ = joint_diagnostics(reference, low)
    matching, _ = joint_diagnostics(reference, reference.copy())
    assert missing["reference_uncovered_fraction"] > 0.35
    assert matching["reference_uncovered_fraction"] < 0.1
    assert missing["region_total_variation"] > 0.4


def test_periodic_embedding_has_no_boundary_seam():
    points = np.zeros((2, 8))
    points[0, 6:], points[1, 6:] = 1e-6, 2 * np.pi - 1e-6
    assert np.linalg.norm(features(points)[0] - features(points)[1]) < 1e-5


def test_missing_reference_and_incomplete_results_are_reported(tmp_path):
    (tmp_path / "results/run_c").mkdir(parents=True)
    (tmp_path / "results/run_c/summary.json").write_text('{"incomplete":')
    rows = compare(results_root=tmp_path / "results", output=tmp_path / "plots", make_plots=False)
    assert rows[0]["status"] == "missing"
    assert next(r for r in rows if r["run_id"] == "run_c")["status"] == "unreadable"


def test_unverified_evidence_is_never_differenced(tmp_path):
    sample_result(tmp_path / "results/run_a")
    sample_result(tmp_path / "results/run_c", "run_c")
    path = tmp_path / "results/run_c/summary.json"
    summary = read_json(path)
    summary["evidence_convention"] = "likelihood_ratio"
    write_json(path, summary)
    rows = compare(run_ids=["run_c"], results_root=tmp_path / "results", output=tmp_path / "plots", make_plots=False)
    assert rows[0]["evidence_comparable"] is True
    assert rows[1]["evidence_comparable"] is False
    assert "delta_log_evidence" not in rows[1]


def test_missing_evidence_error_is_graceful(tmp_path):
    sample_result(tmp_path / "results/run_a", error=None)
    rows = compare(run_ids=["run_c"], results_root=tmp_path / "results", output=tmp_path / "plots", make_plots=False)
    assert rows[0]["evidence_comparable"] is False
    assert "delta_log_evidence" not in rows[0]
