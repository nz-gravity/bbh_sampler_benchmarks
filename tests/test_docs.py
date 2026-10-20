from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest

from benchmark import docs
from benchmark.injection import data_manifest
from benchmark.registry import load_registry, read_json, resolve_run, write_json
from benchmark.results import PARAMETERS, save_result


def result(directory, name):
    points = np.random.default_rng(17).normal(size=(40, len(PARAMETERS)))
    posterior = {parameter: points[:, i] for i, parameter in enumerate(PARAMETERS)}
    save_result(directory / name, posterior, resolve_run(name), {
        "log_evidence": -16170.0, "log_evidence_err": 0.1,
        "log_noise_evidence": data_manifest()["noise_log_likelihood"],
    })


def test_site_refuses_missing_reference_and_mismatched_results(tmp_path):
    results = tmp_path / "results"
    result(results, "run_g")
    with pytest.raises(ValueError, match="run_a.*required"):
        docs.build(tmp_path / "site", results_root=results)
    result(results, "run_a")
    path = results / "run_g/summary.json"
    summary = read_json(path)
    summary["config"]["kwargs"]["nlive"] = 17
    write_json(path, summary)
    with pytest.raises(ValueError, match="configuration differs"):
        docs.build(tmp_path / "site", results_root=results)
    assert not (tmp_path / "site/index.html").exists()


def test_differences_preserve_omitted_settings_and_hide_sampling_seed():
    registry = load_registry()
    assert docs.differences(registry["run_a"], registry["run_g"]) == [("nlive", "4000", "2000")]
    changes = {key: (before, after) for key, before, after in docs.differences(registry["run_a"], registry["run_b"])}
    assert "seed" not in changes
    assert changes["proposals"] == ("not set", '["diff", "volumetric"]')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.ids.update([attrs["id"]] if "id" in attrs else [])
        for key in ("href", "src"):
            if key in attrs:
                self.urls.append(attrs[key])


def test_site_links_and_plots_share_verified_reference_and_axis_bounds(tmp_path, monkeypatch):
    results = tmp_path / "results"
    for name in ("run_a", "run_f", "run_g"):
        result(results, name)
    calls = []

    def capture_plot(posteriors, output, *, bounds):
        calls.append((posteriors, bounds))
        Path(output).write_bytes(b"plot placeholder")

    monkeypatch.setattr(docs, "corner_plot", capture_plot)
    output = docs.build(tmp_path / "site", results_root=results)
    assert len(calls) == 2
    assert all(list(posteriors)[0] == "dynesty_4000 (run_a)" for posteriors, _ in calls)
    np.testing.assert_array_equal(next(iter(calls[0][0].values())), next(iter(calls[1][0].values())))
    assert calls[0][1] == calls[1][1]
    page = (output / "index.html").read_text()
    assert "<code>seed</code>" not in page
    assert "New runs use sampling seed 42." in page
    assert "No result yet" in page
    assert 'data-comparison="run_b"' not in page
    assert page.index('data-comparison="run_g"') < page.index('data-comparison="run_f"')
    links = Links()
    links.feed(page)
    for url in links.urls:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        if parts.fragment:
            assert unquote(parts.fragment) in links.ids, url
        if parts.path:
            assert (output / unquote(parts.path)).exists(), url
