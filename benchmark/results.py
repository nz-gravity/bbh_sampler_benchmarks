"""Small, versionable posterior and evidence files for each run."""

from pathlib import Path

import numpy as np
import pandas as pd

from .injection import data_manifest
from .registry import read_json, sha256, write_json

PARAMETERS = ("chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl")
MAX_SAMPLES = 1000


def save_result(directory, posterior, config, metrics, *, likelihood_id=None):
    """Uniformly select at most 1000 equal-weight posterior rows, without replacement."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    columns = list(PARAMETERS) + [key for key in ("log_likelihood", "log_prior") if key in posterior]
    posterior = pd.DataFrame(posterior)[columns]
    if posterior.empty or not np.all(np.isfinite(posterior.to_numpy())):
        raise ValueError("Posterior must contain finite samples")
    indices = np.sort(np.random.default_rng(0).choice(len(posterior), min(MAX_SAMPLES, len(posterior)), replace=False))
    directory.mkdir(parents=True)
    path = directory / "posterior.csv"
    posterior.iloc[indices].to_csv(path, index=False, float_format="%.17g")
    summary = {
        "config": config, "likelihood_id": likelihood_id or data_manifest()["likelihood_id"],
        "evidence_convention": "full_log_likelihood", **metrics,
        "posterior_rows": len(indices), "original_posterior_rows": len(posterior),
        "posterior_sha256": sha256(path),
    }
    write_json(directory / "summary.json", summary)
    return summary


def load_result(directory):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    path = directory / "posterior.csv"
    if sha256(path) != summary["posterior_sha256"]:
        raise ValueError("Posterior checksum mismatch")
    posterior = pd.read_csv(path, float_precision="round_trip")
    if len(posterior) != summary["posterior_rows"] or not 0 < len(posterior) <= MAX_SAMPLES:
        raise ValueError("Invalid posterior row count")
    if not set(PARAMETERS) <= set(posterior) or not np.all(np.isfinite(posterior.to_numpy())):
        raise ValueError("Invalid posterior columns or samples")
    return summary, posterior
