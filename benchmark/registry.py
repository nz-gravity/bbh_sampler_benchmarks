"""Small run registry and immutable, JSON-serializable configuration snapshots."""

import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate IDs/settings instead of silently taking the last value."""


def _unique_mapping(loader, node):
    pairs = loader.construct_pairs(node, deep=True)
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"Duplicate registry key: {key}")
        mapping[key] = value
    return mapping


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_registry(path=None):
    registry = yaml.load(Path(path or ROOT / "configs/runs.yaml").read_text(), Loader=UniqueLoader)
    if not isinstance(registry, dict) or not registry:
        raise ValueError("Run registry must be a nonempty mapping")
    for run_id, run in registry.items():
        if not re.fullmatch(r"run_[a-z]+", run_id):
            raise ValueError(f"Invalid run ID: {run_id}")
        if run.get("sampler") not in {"dynesty", "bilby_mcmc"}:
            raise ValueError(f"Unsupported sampler in {run_id}")
        if run.get("status") not in {"planned", "imported"}:
            raise ValueError(f"Missing planned/imported status in {run_id}")
        seed = run.get("seed")
        if seed is None and run["status"] != "imported":
            raise ValueError(f"A new run needs a random seed: {run_id}")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError(f"Invalid random seed: {run_id}")
        kwargs = run.get("kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError(f"Missing sampler kwargs: {run_id}")
        forbidden = {"outdir", "label", "sampler", "priors", "likelihood", "seed", "npool", "resume", "use_ratio"}
        if forbidden & kwargs.keys():
            raise ValueError(f"Reserved runner settings in {run_id}: {forbidden & kwargs.keys()}")
        if run["sampler"] == "dynesty" and (type(kwargs.get("nlive")) is not int or kwargs["nlive"] <= 0):
            raise ValueError(f"Invalid nlive: {run_id}")
    if registry.get("run_a", {}).get("kwargs", {}).get("nlive") != 4000:
        raise ValueError("run_a is reserved for Dynesty 4000")
    if registry["run_a"]["sampler"] != "dynesty":
        raise ValueError("run_a must use Dynesty")
    return registry


def resolve_run(run_id, registry=None):
    registry = registry if registry is not None else load_registry()
    if run_id not in registry:
        raise ValueError(f"Unknown run {run_id}; use --list")
    return {"run_id": run_id, **registry[run_id]}


def run_table(registry=None):
    registry = registry if registry is not None else load_registry()
    lines = ["| Run | Sampler | nlive | Exploration | Stopping | Seed | Status |",
             "|---|---|---:|---|---|---|---|"]
    for name, run in registry.items():
        kw = run["kwargs"]
        if run["sampler"] == "dynesty":
            settings = [kw.get("sample", "unknown"), f"bound={kw.get('bound', 'unknown')}"]
            for key in (("slices",) if kw.get("sample") == "rslice" else ("nact", "walks")):
                if key in kw and kw[key] is not None:
                    settings.append(f"{key}={kw[key]}")
            stopping = f"dlogz={kw.get('dlogz', 'unknown')}"
        else:
            settings = [f"{kw.get('ntemps', '?')} temperatures", str(kw.get("proposal_cycle", "unknown"))]
            stopping = f"nsamples={kw.get('nsamples', 'unknown')}"
        seed = "unknown" if run["seed"] is None else str(run["seed"])
        lines.append(f"| {name} | {run['sampler']} | {kw.get('nlive', '—')} | {', '.join(settings)} | {stopping} | {seed} | {run['status']} |")
    return "\n".join(lines)


def update_readme():
    path = ROOT / "README.md"
    text = path.read_text()
    start, end = "<!-- RUNS:START -->", "<!-- RUNS:END -->"
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + "\n" + run_table() + "\n" + end + after)
