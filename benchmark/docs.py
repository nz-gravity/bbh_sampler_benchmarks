"""Build a static GitHub Pages site from verified compact results; never run inference."""

import argparse
from html import escape
import json
from pathlib import Path
import shutil
from string import Template

import numpy as np

from .compare import corner_plot, verified_identity
from .injection import settings
from .registry import ROOT, load_registry
from .results import PARAMETERS, load_result

REFERENCE = "run_a"


def value_text(value):
    return "not recorded" if value is None else json.dumps(value, ensure_ascii=False)


def run_label(run):
    if run["sampler"] == "dynesty":
        return f"dynesty_{run['kwargs']['nlive']}"
    return "bilby_mcmc"


def configuration(run):
    return {"sampler": run["sampler"], **run["kwargs"]}


def differences(reference, run):
    baseline, candidate = configuration(reference), configuration(run)
    return [(key, value_text(baseline[key]) if key in baseline else "not set",
             value_text(candidate[key]) if key in candidate else "not set")
            for key in sorted(baseline.keys() | candidate.keys())
            if key not in baseline or key not in candidate or baseline[key] != candidate[key]]


def code(value):
    return f"<code>{escape(str(value))}</code>"


def changes_html(delta):
    if not delta:
        return "Baseline"
    if len(delta) == 1:
        key, before, after = delta[0]
        return f"{escape(key)}: {escape(before)} → {escape(after)}"
    values = "".join(f"<dt>{escape(key)}</dt><dd>{escape(before)} → {escape(after)}</dd>"
                     for key, before, after in delta)
    return f'<details class="table-differences"><summary>{len(delta)} changed settings</summary><dl>{values}</dl></details>'


def build(output=None, *, results_root=None):
    registry = load_registry()
    results_root = Path(results_root or ROOT / "results")
    loaded = {}
    for name in registry:
        directory = results_root / name
        if directory.exists():
            summary, posterior = load_result(directory)
            verified_identity(name, summary, registry)
            loaded[name] = (summary, posterior[list(PARAMETERS)].to_numpy())
    if REFERENCE not in loaded:
        raise ValueError("Verified dynesty_4000 (run_a) results are required to build the site")

    output = Path(output or ROOT / "_site").resolve()
    # Write only generated files; do not clear a user-supplied destination.
    output.mkdir(parents=True, exist_ok=True)
    assets, downloads = output / "assets", output / "downloads"
    assets.mkdir(exist_ok=True)
    downloads.mkdir(exist_ok=True)
    for name in ("style.css", "main.js"):
        shutil.copyfile(ROOT / "docs" / name, output / name)
    for name in ("runs.yaml", "bbh.yaml", "bbh.prior"):
        shutil.copyfile(ROOT / "configs" / name, downloads / name)

    truth = [settings()["injection"][name] for name in PARAMETERS]
    combined = np.vstack([*(points for _, points in loaded.values()), truth])
    bounds = [(float(combined[:, i].min()), float(combined[:, i].max())) for i in range(len(PARAMETERS))]
    baseline = registry[REFERENCE]
    table, choices, panels = [], [], []
    candidates = sorted((name for name in loaded if name != REFERENCE),
                        key=lambda name: registry[name]["sampler"] != "dynesty")
    for name, run in registry.items():
        kw = run["kwargs"]
        delta = differences(baseline, run)
        if run["sampler"] == "dynesty":
            exploration = f"{kw.get('sample', 'default')} · {kw.get('bound', 'default')} bound"
            stopping = f"dlogz = {kw.get('dlogz', 'default')}"
        else:
            exploration = f"{kw.get('ntemps', '?')} temperatures · {kw.get('proposal_cycle', 'default')}"
            stopping = f"nsamples = {kw.get('nsamples', 'default')}"
        status = ("Reference" if name == REFERENCE else "View plot ↗") if name in loaded else "No result yet"
        status_html = (f'<a class="plot-link" href="#{name}">{status}</a>' if name in candidates
                       else f'<span class="badge {"reference" if name == REFERENCE else "pending"}">{status}</span>')
        change = changes_html(delta)
        table.append(f'<tr class="{"baseline" if name == REFERENCE else ""}"><th scope="row">'
                     f'{code(run_label(run))}<small>{name}</small></th><td>{kw.get("nlive", "—")}</td>'
                     f'<td>{escape(exploration)}</td><td>{code(stopping)}</td>'
                     f'<td class="changes">{change}</td><td>{status_html}</td></tr>')

    for name in loaded:
        target = downloads / name
        target.mkdir(exist_ok=True)
        for filename in ("posterior.csv", "summary.json"):
            shutil.copyfile(results_root / name / filename, target / filename)

    for name in candidates:
        run, (summary, points) = registry[name], loaded[name]
        label = run_label(run)
        plot_name = f"{name}_vs_{REFERENCE}.png"
        corner_plot({"dynesty_4000 (run_a)": loaded[REFERENCE][1], f"{label} ({name})": points},
                    assets / plot_name, bounds=bounds)
        choices.append(f'<a href="#{name}" data-comparison="{name}">{escape(label)}'
                       f'<span>{name}</span></a>')
        changes = "".join(f"<tr><th scope=\"row\">{code(key)}</th><td>{code(before)}</td><td>{code(after)}</td></tr>"
                          for key, before, after in differences(baseline, run))
        if not changes:
            changes = '<tr><td colspan="3">No supplied settings differ from the reference.</td></tr>'
        panels.append(f'''<article class="comparison" id="{name}" aria-labelledby="title-{name}">
          <div class="comparison-heading"><div><p class="eyebrow">{name} / run_a</p>
            <h3 id="title-{name}">{escape(label)} <span>vs</span> dynesty_4000</h3></div>
            <a class="button" href="assets/{plot_name}" target="_blank" rel="noopener">Open full-size plot ↗</a></div>
          <div class="legend"><span><i class="line candidate"></i>{escape(label)}</span>
            <span><i class="line"></i>dynesty_4000 · reference</span><span><i class="line truth"></i>Injection</span></div>
          <a class="plot" href="assets/{plot_name}" target="_blank" rel="noopener" aria-label="Open {escape(label)} corner plot at full size">
            <img src="assets/{plot_name}" width="2172" height="2172" loading="lazy"
              alt="Eight-parameter corner plot: {escape(label)} ({name}) in blue over dynesty_4000 (run_a) in black; injection in grey."></a>
          <div class="plot-footer"><span>{summary['posterior_rows']:,} candidate / {loaded[REFERENCE][0]['posterior_rows']:,} reference samples</span>
            <a href="downloads/{name}/posterior.csv" download>Posterior CSV ↓</a>
            <a href="downloads/{name}/summary.json" download>Run summary ↓</a></div>
          <details class="setting-details"><summary>Exact settings differences <span>{len(differences(baseline, run))} changed</span></summary>
            <p>Supplied settings only. “Not set” means omitted, rather than a known sampler default. New runs use sampling seed 42.</p>
            <div class="table-scroll" tabindex="0" role="region" aria-label="Settings differences for {name}">
              <table class="diff-table"><thead><tr><th scope="col">Setting</th><th scope="col">dynesty_4000</th><th scope="col">{escape(label)}</th></tr></thead><tbody>{changes}</tbody></table>
            </div></details>
        </article>''')

    config = settings()
    context = {
        "run_count": str(len(registry)), "available_count": str(len(loaded)),
        "comparison_count": str(len(candidates)), "rows": "\n".join(table),
        "choices": "\n".join(choices), "panels": "\n".join(panels) or '<p>No candidate posteriors are available yet.</p>',
        "detectors": escape(" + ".join(config["detectors"])),
        "waveform": escape(config["waveform_approximant"]),
        "duration": f'{config["duration"]:g}', "sample_rate": str(config["sampling_frequency"]),
        "min_frequency": f'{config["minimum_frequency"]:g}',
        "max_frequency": f'{config["maximum_frequency"]:g}',
        "parameter_count": str(len(PARAMETERS)),
    }
    template = Template((ROOT / "docs" / "template.html").read_text())
    (output / "index.html").write_text(template.substitute(context))
    (output / ".nojekyll").touch()
    print(f"Built {output}: {len(candidates)} comparisons against dynesty_4000")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    build(args.output)


if __name__ == "__main__":
    main()
