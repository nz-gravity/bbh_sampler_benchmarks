"""One injection, one saved strain realization, and one likelihood constructor."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from .registry import ROOT, fingerprint, read_json, sha256, write_json

CONFIG = ROOT / "configs/bbh.yaml"
DATA = ROOT / "data/bbh_strain.npz"
CORE_PACKAGES = ("bilby", "lalsuite", "numpy", "scipy")


def settings():
    return yaml.safe_load(CONFIG.read_text())


def scientific_config():
    config = settings()
    return {"settings": config, "prior_sha256": sha256(ROOT / config["prior_file"])}


def _waveform(config):
    import bilby

    return bilby.gw.WaveformGenerator(
        duration=config["duration"], sampling_frequency=config["sampling_frequency"],
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"waveform_approximant": config["waveform_approximant"],
                            "reference_frequency": config["reference_frequency"]},
    )


def _interferometers(config):
    import bilby

    ifos = bilby.gw.detector.InterferometerList(config["detectors"])
    for ifo in ifos:
        ifo.minimum_frequency = config["minimum_frequency"]
        ifo.maximum_frequency = config["maximum_frequency"]
    return ifos


def _assemble(config, ifos, waveform):
    import bilby

    priors = bilby.gw.prior.BBHPriorDict(str(ROOT / config["prior_file"]))
    for name in config["fixed_parameters"]:
        priors[name] = config["injection"][name]
    priors.convert_floats_to_delta_functions()
    priors.validate_prior(config["duration"], config["minimum_frequency"])
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos, waveform, priors=priors,
        time_marginalization=config["marginalization"]["time"],
        phase_marginalization=config["marginalization"]["phase"],
        distance_marginalization=config["marginalization"]["distance"],
    )
    return likelihood, priors


def data_manifest(data_path=DATA):
    data_path = Path(data_path)
    manifest = read_json(data_path.with_suffix(".json"))
    if manifest["scientific_config"] != scientific_config():
        raise ValueError("BBH configuration/prior changed: saved strain belongs to a different experiment")
    if manifest["data_sha256"] != sha256(data_path):
        raise ValueError("Saved strain checksum mismatch")
    expected = fingerprint({k: manifest[k] for k in ("scientific_config", "data_sha256", "core_versions", "geometry")})
    if expected != manifest["likelihood_id"]:
        raise ValueError("Saved likelihood identity mismatch")
    return manifest


def geometry(ifo):
    return {name: getattr(ifo, name) for name in (
        "length", "latitude", "longitude", "elevation", "xarm_azimuth", "yarm_azimuth", "xarm_tilt", "yarm_tilt")}


def build_likelihood(data_path=DATA):
    """Load immutable strain and PSD arrays; no run can regenerate or reseed noise."""
    import bilby

    config = settings()
    manifest = data_manifest(data_path)
    actual = {name: version(name) for name in CORE_PACKAGES}
    if actual != manifest["core_versions"]:
        raise ValueError("Likelihood dependency versions differ from the saved data manifest; use uv sync --locked")
    ifos = _interferometers(config)
    with np.load(data_path, allow_pickle=False) as data:
        for ifo in ifos:
            if geometry(ifo) != manifest["geometry"][ifo.name]:
                raise ValueError(f"Detector geometry changed: {ifo.name}")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=data["frequency"], psd_array=data[f"{ifo.name}_psd"])
            ifo.set_strain_data_from_frequency_domain_strain(
                data[f"{ifo.name}_strain"], sampling_frequency=config["sampling_frequency"],
                duration=config["duration"], start_time=config["start_time"])
            ifo.meta_data["parameters"] = config["injection"].copy()
    likelihood, priors = _assemble(config, ifos, _waveform(config))
    if not np.isclose(likelihood.noise_log_likelihood(), manifest["noise_log_likelihood"], rtol=0, atol=1e-9):
        raise ValueError("Saved data noise likelihood changed")
    return likelihood, priors


def generate_data(data_path=DATA):
    """Explicit, deterministic one-time preparation. Refuse to replace existing data."""
    import bilby

    data_path = Path(data_path)
    if data_path.exists() or data_path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to replace {data_path}")
    config = settings()
    bilby.core.utils.random.seed(config["noise_seed"])
    np.random.seed(config["noise_seed"])
    ifos, waveform = _interferometers(config), _waveform(config)
    for ifo in ifos:
        if Path(ifo.power_spectral_density.asd_file).name != config["psd"]:
            raise ValueError(f"Unexpected default PSD for {ifo.name}")
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=config["sampling_frequency"], duration=config["duration"], start_time=config["start_time"])
    ifos.inject_signal(waveform_generator=waveform, parameters=config["injection"])
    likelihood, _ = _assemble(config, ifos, waveform)
    arrays = {"frequency": ifos[0].frequency_array}
    for ifo in ifos:
        arrays[f"{ifo.name}_strain"] = ifo.frequency_domain_strain
        arrays[f"{ifo.name}_psd"] = ifo.power_spectral_density_array
    data_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent generators overwriting each other.
    with data_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {
        "scientific_config": scientific_config(), "data_sha256": sha256(data_path),
        "core_versions": {name: version(name) for name in CORE_PACKAGES},
        "geometry": {ifo.name: geometry(ifo) for ifo in ifos},
        "noise_log_likelihood": likelihood.noise_log_likelihood(),
        "evidence_convention": "Bilby full log likelihood (noise constant included); use_ratio=True during sampling",
    }
    manifest["likelihood_id"] = fingerprint({k: manifest[k] for k in ("scientific_config", "data_sha256", "core_versions", "geometry")})
    write_json(data_path.with_suffix(".json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--generate", action="store_true", help="Generate the shared data once; never runs inference")
    actions.add_argument("--check", action="store_true", help="Regenerate in a temporary directory and require byte-identical data")
    args = parser.parse_args()
    if args.generate:
        generate_data()
    elif args.check:
        from tempfile import TemporaryDirectory

        expected = data_manifest()
        with TemporaryDirectory() as directory:
            actual = generate_data(Path(directory) / "bbh_strain.npz")
        if actual["data_sha256"] != expected["data_sha256"]:
            raise ValueError("Regenerated dataset differs from the committed dataset")
        print("Dataset regenerated byte-for-byte; configuration and priors match.")
    else:
        likelihood, priors = build_likelihood()
        print(f"Shared likelihood: {data_manifest()['likelihood_id']}")
        print(f"Sampled parameters: {', '.join(priors.non_fixed_keys)}")
        print(f"Noise log likelihood: {likelihood.noise_log_likelihood():.12f}")


if __name__ == "__main__":
    main()
