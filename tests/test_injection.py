from copy import deepcopy

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
    repeated = tmp_path / "repeated.npz"
    generate_data(repeated)
    # Fixed seeds must reproduce identical bytes on this platform. Compiled
    # waveform libraries can round differently across operating systems/CPUs.
    assert sha256(path) == sha256(repeated)
    with np.load(DATA) as expected, np.load(path) as actual:
        assert actual.files == expected.files
        for key in expected.files:
            finite = np.isfinite(expected[key])
            # PSDs use infinity outside their supported band; check those entries
            # exactly and exclude them from the roundoff scale.
            np.testing.assert_array_equal(actual[key][~finite], expected[key][~finite])
            # Bound absolute roundoff relative to each array's physical scale;
            # numpy's default absolute tolerance would dwarf strain/PSD values.
            tolerance = 64 * np.finfo(float).eps * np.max(np.abs(expected[key][finite]), initial=0)
            np.testing.assert_allclose(actual[key][finite], expected[key][finite], rtol=0, atol=tolerance)
    reloaded, priors = build_likelihood(path)
    parameters = settings()["injection"]
    assert reloaded.log_likelihood(parameters=parameters) == pytest.approx(
        shared_likelihood[0].log_likelihood(parameters=parameters), rel=0, abs=1e-9)
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


@pytest.mark.parametrize("size,shape", [(None, (8, 1)), (1, (8, 1)), (2, (8, 2))])
def test_constrained_draws_keep_shapes_and_constraints(shared_likelihood, size, shape):
    # Bilby copies the priors before sampler initialization; keep compatibility
    # through that copy as well as through the likelihood's original priors.
    priors = shared_likelihood[1].copy()
    keys = priors.non_fixed_keys
    samples = priors.sample_subset_constrained_as_array(keys, size=size)
    assert samples.shape == shape
    assert np.all(priors.evaluate_constraints(dict(zip(keys, samples))))


def test_single_draw_still_rejects_invalid_constraints(shared_likelihood, monkeypatch):
    priors = deepcopy(shared_likelihood[1])
    priors["mass_1"].maximum = 40
    # Test a stricter constraint on a copy only: reject a high primary mass,
    # then accept an equal-mass draw. The experiment's priors are unchanged.
    draws = iter([{"chirp_mass": 20.0, "mass_ratio": 0.125},
                  {"chirp_mass": 14.0, "mass_ratio": 1.0}])
    monkeypatch.setattr(priors, "sample_subset", lambda **kwargs: next(draws))
    samples = priors.sample_subset_constrained(["chirp_mass", "mass_ratio"], size=1)
    np.testing.assert_array_equal(samples["chirp_mass"], [14.0])
    np.testing.assert_array_equal(samples["mass_ratio"], [1.0])
