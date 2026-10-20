import numpy as np
import pytest

from benchmark.compare import compare, features, joint_diagnostics
from benchmark.registry import read_json, resolve_run, write_json
from benchmark.results import PARAMETERS, save_result


@pytest.mark.parametrize("reverse", [False, True])
def test_corner_marginals_have_unit_area_and_never_clip_earlier_curves(tmp_path, monkeypatch, reverse):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from benchmark import compare as plotting

    # Different widths expose overlay clipping; different row counts expose
    # raw-count histograms that visually underweight the smaller posterior.
    monkeypatch.setattr(plotting, "PARAMETERS", PARAMETERS[:2])
    rng = np.random.default_rng(6)
    center = np.array([14.8, 0.63])
    runs = [("narrow", rng.normal(center, [0.015, 0.03], size=(1000, 2))),
            ("broad", rng.normal(center, [0.06, 0.12], size=(200, 2)))]
    if reverse:
        runs.reverse()
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: captured.append(figure))
    plotting.corner_plot(dict(runs), tmp_path / "corner.png")
    axes = np.array(captured[0].axes).reshape(2, 2)
    for axis in axes.diagonal():
        assert len(axis.patches) == 2
        colours = [patch.get_edgecolor() for patch in axis.patches]
        assert colours[0] != colours[1]
        for patch in axis.patches:
            vertices = patch.get_xy()
            area = abs(np.trapezoid(vertices[:, 1], vertices[:, 0]))
            assert area == pytest.approx(1.0)
            assert np.max(vertices[:, 1]) < axis.get_ylim()[1]


def sample_result(directory, run_id="run_a", error=0.1):
    points = np.random.default_rng(5).normal(size=(40, 8))
    posterior = {name: points[:, i] for i, name in enumerate(PARAMETERS)}
    return save_result(directory, posterior, resolve_run(run_id), {
        "log_evidence": -16170, "log_evidence_err": error,
        "log_noise_evidence": -16293.358364584721,
        "sampling_seconds": 41394.719138, "ncores": 16,
        "likelihood_evaluations": 217823576,
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
    assert rows[0]["sampling_seconds"] == 41394.719138
    assert rows[0]["ncores"] == 16
    assert rows[0]["likelihood_evaluations"] == 217823576
    assert rows[1]["evidence_comparable"] is False
    assert "delta_log_evidence" not in rows[1]


def test_missing_evidence_error_is_graceful(tmp_path):
    sample_result(tmp_path / "results/run_a", error=None)
    rows = compare(run_ids=["run_c"], results_root=tmp_path / "results", output=tmp_path / "plots", make_plots=False)
    assert rows[0]["evidence_comparable"] is False
    assert "delta_log_evidence" not in rows[0]
