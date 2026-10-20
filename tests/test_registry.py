from copy import deepcopy

import pytest
import yaml

from benchmark.registry import ROOT, load_registry, read_json, resolve_run, run_table


def write_registry(tmp_path, value):
    path = tmp_path / "runs.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def test_registry_and_generated_readme_agree():
    registry = load_registry()
    assert registry["run_a"]["kwargs"]["nlive"] == 4000
    assert registry["run_a"]["kwargs"]["nact"] == 20
    assert run_table(registry) in (ROOT / "README.md").read_text()
    assert all(run["seed"] == 42 for run in registry.values() if run["status"] == "planned")


def test_completed_configurations_match_registry():
    registry = load_registry()
    for path in (ROOT / "results").glob("run_*/summary.json"):
        summary = read_json(path)
        assert summary["config"] == resolve_run(path.parent.name, registry)


@pytest.mark.parametrize("change", [
    lambda d: d["run_b"].update(seed=None),
    lambda d: d["run_b"].update(seed=-1),
    lambda d: d["run_a"]["kwargs"].update(nlive=500),
    lambda d: d["run_b"]["kwargs"].update(nlive=0),
    lambda d: d["run_b"]["kwargs"].update(outdir="elsewhere"),
    lambda d: d["run_b"].update(sampler="unsupported"),
])
def test_bad_registry_is_rejected(tmp_path, change):
    registry = deepcopy(load_registry())
    change(registry)
    with pytest.raises(ValueError):
        load_registry(write_registry(tmp_path, registry))


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "runs.yaml"
    path.write_text("run_a: {}\nrun_a: {}\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_registry(path)


def test_unknown_run_is_clear():
    with pytest.raises(ValueError, match="Unknown run"):
        resolve_run("run_missing")
