import numpy as np
import pytest

from benchmark.injection import DATA, build_likelihood, data_manifest, generate_data, settings
from benchmark.registry import ROOT, sha256
from benchmark.results import load_result


@pytest.fixture(scope="module")
def shared_likelihood():
    return build_likelihood()


def test_saved_data_and_prior_contract(shared_likelihood):
    likelihood, priors = shared_likelihood
    assert list(priors.non_fixed_keys) == ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]
    assert priors["chirp_mass"].minimum == 14
    assert priors["chirp_mass"].maximum == 20
    assert priors["mass_ratio"].minimum == 0.125
    assert {ifo.name for ifo in likelihood.interferometers} == {"H1", "L1"}
    assert all(ifo.minimum_frequency == 20 and ifo.maximum_frequency == 2048 for ifo in likelihood.interferometers)
    assert not likelihood.time_marginalization
    assert not likelihood.phase_marginalization
    assert not likelihood.distance_marginalization
    assert likelihood.noise_log_likelihood() == pytest.approx(-16293.358364584721, abs=1e-9)
    assert data_manifest()["data_sha256"] == sha256(DATA)


def test_deterministic_generation_and_no_overwrite(tmp_path, shared_likelihood):
    path = tmp_path / "bbh_strain.npz"
    generate_data(path)
    with np.load(DATA) as expected, np.load(path) as actual:
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert sha256(path) == sha256(DATA)
    reloaded, priors = build_likelihood(path)
    parameters = settings()["injection"]
    assert reloaded.log_likelihood(parameters=parameters) == shared_likelihood[0].log_likelihood(parameters=parameters)
    with pytest.raises(FileExistsError):
        generate_data(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        build_likelihood(path)


@pytest.mark.parametrize("run_id", ["run_a", "run_f", "run_g", "run_h", "run_i"])
def test_compact_posterior_uses_exact_same_likelihood_and_priors(run_id, shared_likelihood):
    summary, posterior = load_result(ROOT / "results" / run_id)
    likelihood, priors = shared_likelihood
    for index in np.linspace(0, len(posterior) - 1, 32).astype(int):
        row = posterior.iloc[index].to_dict()
        parameters = {**settings()["injection"], **row}
        assert likelihood.log_likelihood_ratio(parameters=parameters) == pytest.approx(row["log_likelihood"], abs=1e-7, rel=0)
        sampled = {name: row[name] for name in priors.non_fixed_keys}
        assert priors.ln_prob(sampled, axis=0) == pytest.approx(row["log_prior"], abs=1e-7, rel=0)
    assert summary["likelihood_id"] == data_manifest()["likelihood_id"]


def test_prior_file_is_unchanged():
    assert sha256(ROOT / settings()["prior_file"]) == "4c8d3dd0b4b7535498529269e15596412a74ade8b346a30ec5d8a61de018c121"
