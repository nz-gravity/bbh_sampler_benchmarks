"""Compare available Bilby posteriors, evidence, and cost against a registered reference."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .injection import data_manifest, settings
from .registry import ROOT, load_registry, resolve_run
from .results import PARAMETERS, load_result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def verified_identity(run_id, summary, registry):
    if summary["config"] != resolve_run(run_id, registry):
        raise ValueError("Stored run configuration differs from configs/runs.yaml")
    manifest = data_manifest()
    if summary["likelihood_id"] != manifest["likelihood_id"]:
        raise ValueError("Likelihood or priors differ")
    if summary["evidence_convention"] != "full_log_likelihood":
        raise ValueError("Evidence convention differs")
    if not np.isclose(summary["log_noise_evidence"], manifest["noise_log_likelihood"], rtol=0, atol=1e-7):
        raise ValueError("Noise likelihood differs")
    return summary["likelihood_id"]


def features(points):
    """Embed periodic coordinates without creating a seam at 0/2pi."""
    return np.column_stack([points[:, :6], np.sin(points[:, 6:]), np.cos(points[:, 6:])])


def joint_diagnostics(reference, candidate):
    """Joint-space coverage and reference-defined region occupancies, not convergence tests."""
    from scipy.cluster.vq import kmeans2, vq
    from scipy.spatial import cKDTree

    ref, test = features(reference), features(candidate)
    mean = ref.mean(axis=0)
    eig, vectors = np.linalg.eigh(np.cov(ref, rowvar=False))
    transform = vectors @ np.diag(1 / np.sqrt(np.maximum(eig, 1e-8)))
    ref, test = (ref - mean) @ transform, (test - mean) @ transform
    count = min(1000, len(test), len(ref) // 2)
    if count < 10:
        return {"joint_diagnostic_note": "Too few samples for a joint support screen"}, []
    rng = np.random.default_rng(2026)
    indices = rng.choice(len(ref), 2 * count, replace=False)
    train, probe = ref[indices[:count]], ref[indices[count:]]
    other = test[rng.choice(len(test), count, replace=False)]
    radius = np.quantile(cKDTree(train).query(probe)[0], 0.95)
    uncovered = np.mean(cKDTree(other).query(probe)[0] > radius)
    centers, _ = kmeans2(ref[rng.choice(len(ref), min(5000, len(ref)), replace=False)], 8, iter=30, minit="++", seed=2026)
    ref_region, _ = vq(ref, centers)
    test_region, _ = vq(test, centers)
    occupancies = [{"region": i, "reference_fraction": float(np.mean(ref_region == i)),
                    "candidate_fraction": float(np.mean(test_region == i))} for i in range(8)]
    return {
        "reference_uncovered_fraction": float(uncovered),
        "joint_screen_sample_size": count,
        "region_total_variation": float(sum(abs(r["reference_fraction"] - r["candidate_fraction"]) for r in occupancies) / 2),
        "empty_reference_regions": sum(r["reference_fraction"] >= 0.01 and r["candidate_fraction"] == 0 for r in occupancies),
        "joint_diagnostic_note": "Heuristic: reference NN baseline is 5%; regions are not certified modes; weight error can resemble support loss",
    }, occupancies


def corner_plot(posteriors, output, *, bounds=None):
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    truth = [settings()["injection"][name] for name in PARAMETERS]
    combined = np.vstack([*posteriors.values(), truth])
    if bounds is None:
        bounds = [(float(combined[:, i].min()), float(combined[:, i].max())) for i in range(len(PARAMETERS))]
    options = dict(labels=PARAMETERS, range=bounds, bins=30, plot_datapoints=False,
                   plot_density=False, smooth=1, levels=(0.5, 0.9), quiet=True)
    fig, handles = None, []
    marginal_limits = np.zeros(len(PARAMETERS))
    for index, (name, points) in enumerate(posteriors.items()):
        color = "black" if index == 0 else f"C{index - 1}"
        markers = {"truths": truth, "truth_color": "grey"} if index == 0 else {}
        # A fresh dict is needed: corner adds the curve colour to hist_kwargs.
        fig = corner.corner(points, fig=fig, color=color, hist_kwargs={"density": True}, **markers, **options)
        axes = np.asarray(fig.axes).reshape(len(PARAMETERS), len(PARAMETERS))
        # With explicit ranges, corner resets these limits on every overlay.
        # Keep the largest density limit so later curves cannot clip earlier ones.
        marginal_limits = np.maximum(marginal_limits, [axis.get_ylim()[1] for axis in axes.diagonal()])
        handles.append(Line2D([], [], color=color, label=name))
    for axis, upper in zip(axes.diagonal(), marginal_limits):
        axis.set_ylim(0, upper)
    handles.append(Line2D([], [], color="grey", label="injection"))
    fig.legend(handles=handles, loc="upper right")
    fig.savefig(output, dpi=120)
    plt.close(fig)


def cost_plot(rows, output):
    import matplotlib.pyplot as plt

    present = [r for r in rows if r.get("status") == "available"]
    if not present:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, field, label in zip(axes, ("delta_log_evidence", "likelihood_evaluations", "sampling_seconds"),
                                   ("Δ log evidence", "Likelihood evaluations", "Sampling time (s)")):
        values = [r for r in present if finite_number(r.get(field)) and (field == "delta_log_evidence" or r[field] > 0)]
        x = np.arange(len(values))
        if field == "delta_log_evidence":
            axis.errorbar(x, [r[field] for r in values], yerr=[r["delta_log_evidence_err"] for r in values], fmt="o")
            axis.axhline(0, color="grey", linewidth=1)
        else:
            axis.bar(x, [r[field] for r in values])
            axis.set_yscale("log")
        axis.set_xticks(x, [r["run_id"] for r in values], rotation=45)
        axis.set_ylabel(label)
    fig.suptitle("Wall times use different core counts; see summary.csv")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def compare(reference_id="run_a", run_ids=None, *, results_root=None, output=None, make_plots=True):
    registry = load_registry()
    resolve_run(reference_id, registry)
    selected = list(dict.fromkeys([reference_id, *(run_ids if run_ids else registry)]))
    for name in selected:
        resolve_run(name, registry)
    directory = Path(output or ROOT / "plots" / f"against_{reference_id}")
    if output is None and run_ids:
        directory = directory / ("selected_" + "_".join(run_ids))
    directory.mkdir(parents=True, exist_ok=True)
    rows, loaded = [], {}
    for name in selected:
        row = {"run_id": name, "sampler": registry[name]["sampler"], "status": "missing", "note": ""}
        rows.append(row)
        try:
            result_directory = Path(results_root or ROOT / "results") / name
            if not result_directory.exists():
                continue
            summary, posterior = load_result(result_directory)
            points = posterior[list(PARAMETERS)].to_numpy()
            row.update(status="available", **{key: summary.get(key) for key in (
                "log_evidence", "log_evidence_err", "log_noise_evidence", "sampling_seconds",
                "likelihood_evaluations", "ncores", "posterior_rows", "original_posterior_rows")})
            identity = None
            try:
                identity = verified_identity(name, summary, registry)
            except (ValueError, KeyError, OSError, TypeError) as error:
                row["note"] = str(error)
            loaded[name] = (row, summary, points, identity)
        except (ValueError, KeyError, OSError, TypeError) as error:
            row.update(status="unreadable", note=str(error))
    reference = loaded.get(reference_id)
    for name, (row, raw, points, identity) in loaded.items():
        comparable = bool(reference and identity and identity == reference[3])
        valid_evidence = comparable and all(finite_number(v) for v in (
            row.get("log_evidence", np.nan), row.get("log_evidence_err", np.nan),
            reference[0].get("log_evidence", np.nan), reference[0].get("log_evidence_err", np.nan)))
        valid_evidence = valid_evidence and row["log_evidence_err"] >= 0 and reference[0]["log_evidence_err"] >= 0
        row["evidence_comparable"] = valid_evidence
        if valid_evidence:
            row["delta_log_evidence"] = row["log_evidence"] - reference[0]["log_evidence"]
            row["delta_log_evidence_err"] = 0.0 if name == reference_id else float(np.hypot(row["log_evidence_err"], reference[0]["log_evidence_err"]))
        elif not row["note"]:
            row["note"] = "Reference missing/unverified or likelihood/evidence conventions differ; no evidence difference computed"
        if reference and comparable:
            try:
                if name != reference_id:
                    diagnostics, regions = joint_diagnostics(reference[2], points)
                    row.update(diagnostics)
                    write_csv(directory / f"{name}_regions.csv", regions)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row["plot_note"] = str(error)
    write_csv(directory / "summary.csv", rows)
    if make_plots:
        cost_plot(rows, directory / "evidence_cost.png")
        posteriors = {name: item[2] for name, item in loaded.items() if reference and item[3] and item[3] == reference[3]}
        if posteriors:
            corner_plot(posteriors, directory / "corner.png")
    print(f"Saved {directory / 'summary.csv'}")
    for row in rows:
        delta = row.get("delta_log_evidence")
        delta_text = f"ΔlogZ={delta:+.3f}" if delta is not None else "ΔlogZ unavailable"
        print(f"{row['run_id']}: {row['status']}; {delta_text}" + (f"; {row['note']}" if row["note"] else ""))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="run_a")
    parser.add_argument("--run", action="append", dest="runs", help="Compare only this run; may be repeated")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
    try:
        compare(args.reference, args.runs, make_plots=not args.no_plots)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
