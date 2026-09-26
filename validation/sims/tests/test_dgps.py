"""Tests of validation/sims/dgps.py."""

from __future__ import annotations

import numpy as np
import pytest

from validation.sims import dgps

# A small, hand-built diagnostics dict, independent of the committed
# diagnostics.json, so these tests do not depend on running diagnostics.py.
FAKE_DIAGNOSTICS = {
    "D1": {"sigma_lo": 0.5, "sigma_hi": 1.0},
    "D6": {"sigma": 1.0},
    "D3-bin": {"lambda": 1.6, "mean_f": 0.0},
}


@pytest.mark.parametrize(
    ("name", "point", "expected"),
    [
        ("D1", (0.0, 0.0, 0.0), 0.0),
        ("D1", (1.0, 1.0, 1.0), 1.0 + 2.0 - 1.0),
        ("D2", (0.0, 0.0), 1.0),  # first two hinges 0, third hinge 2*(0.5-0)=1
        ("D5", (0.0, 0.0, 0.7), 0.7),  # both hinges 0, linear term x3
        ("D6", (0.3, 0.9), 0.0),
    ],
)
def test_f_hand_values(name, point, expected):
    dgp = dgps.REGISTRY[name]
    p = dgp.p
    x = np.zeros(p)
    x[: len(point)] = point
    X = x.reshape(1, -1)
    got = float(dgp.f(X)[0])
    assert got == pytest.approx(expected, abs=1e-12)


def test_d2_hinge_hand_value():
    # f_D2 = 2(x1-0.3)+ - 3(x1-0.7)+ + 2(0.5-x2)+ ; at x1=0.8, x2=0.5:
    # 2*0.5 - 3*0.1 + 2*0 = 1.0 - 0.3 = 0.7
    X = np.array([[0.8, 0.5]])
    assert dgps.f_d2(X)[0] == pytest.approx(0.7, abs=1e-12)


def test_d4_friedman1_hand_value():
    X = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    expected = 10 * np.sin(np.pi * 0.25) + 20 * 0.0 + 10 * 0.5 + 5 * 0.5
    assert dgps.f_d4(X)[0] == pytest.approx(expected, abs=1e-12)


def test_d7_and_d8_reuse_d3_and_d4_functional_form():
    rng = np.random.default_rng(0)
    x10 = rng.uniform(size=(5, 10))
    assert np.array_equal(dgps.f_d7(x10), dgps.f_d3(x10))
    assert np.array_equal(dgps.f_d8(x10), dgps.f_d4(x10))


@pytest.mark.parametrize(
    ("name", "relevant"),
    [
        ("D1", (0, 1, 2)),
        ("D2", (0, 1)),
        ("D3", (0, 1, 2)),
        ("D4", (0, 1, 2, 3, 4)),
        ("D5", (0, 1, 2)),
        ("D6", ()),
        ("D7", (0, 1, 2)),
        ("D8", (0, 1, 2, 3, 4)),
        ("D3-bin", (0, 1, 2)),
        ("D4-bin", (0, 1, 2, 3, 4)),
    ],
)
def test_relevant_covariates(name, relevant):
    assert dgps.REGISTRY[name].relevant == relevant


def test_registry_covariate_counts():
    assert dgps.REGISTRY["D7"].p == 50
    for name, dgp in dgps.REGISTRY.items():
        if name != "D7":
            assert dgp.p == 10, name


def test_sample_independent_uniform_bounds_and_shape():
    rng = np.random.default_rng(0)
    X = dgps.sample_independent_uniform(rng, 1000, 10)
    assert X.shape == (1000, 10)
    assert (X >= 0.0).all() and (X <= 1.0).all()


def test_sample_copula_uniform_is_uniform_margin_and_correlated():
    rng = np.random.default_rng(0)
    X = dgps.sample_copula_uniform(rng, 200_000, 10)
    assert X.shape == (200_000, 10)
    assert (X >= 0.0).all() and (X <= 1.0).all()
    # Uniform margins: mean about 0.5, sd about 1/sqrt(12) = 0.2887.
    assert X[:, 0].mean() == pytest.approx(0.5, abs=0.01)
    assert X[:, 0].std() == pytest.approx(1 / np.sqrt(12), abs=0.01)
    # The plan: latent correlation 0.6 gives about 0.58 between uniform margins.
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert corr == pytest.approx(0.582, abs=0.02)


def test_all_cells_counts_51_cells_as_planned():
    cells = dgps.all_cells()
    assert len(cells) == 51
    two_level = [c for c in cells if c.dgp not in ("D6", "D3-bin", "D4-bin")]
    assert len(two_level) == 7 * 3 * 2
    d6 = [c for c in cells if c.dgp == "D6"]
    assert len(d6) == 3 and all(c.noise is None for c in d6)
    binary = [c for c in cells if c.dgp in ("D3-bin", "D4-bin")]
    assert len(binary) == 2 * 3 and all(c.noise is None for c in binary)
    assert len(set(cells)) == 51  # every cell name is unique
    assert len({c.name for c in cells}) == 51


def test_cell_name_round_trips_its_fields():
    cell = dgps.Cell("D4", 1000, "lo")
    assert cell.name == "D4_n01000_lo"
    assert str(cell) == cell.name
    no_noise = dgps.Cell("D6", 200, None)
    assert no_noise.name == "D6_n00200"


def test_generate_regression_dgp_uses_the_right_sigma():
    dgp = dgps.REGISTRY["D1"]
    rng = np.random.default_rng(0)
    X, y, truth = dgps.generate(dgp, rng, 5000, "lo", FAKE_DIAGNOSTICS)
    assert X.shape == (5000, 10)
    assert np.array_equal(truth, dgp.f(X))
    resid_sd = (y - truth).std()
    assert resid_sd == pytest.approx(0.5, abs=0.05)  # sigma_lo in FAKE_DIAGNOSTICS


def test_generate_d6_uses_fixed_sigma_one():
    dgp = dgps.REGISTRY["D6"]
    rng = np.random.default_rng(0)
    _x, y, truth = dgps.generate(dgp, rng, 5000, None, FAKE_DIAGNOSTICS)
    assert np.array_equal(truth, np.zeros(5000))
    assert y.std() == pytest.approx(1.0, abs=0.05)


def test_generate_binary_dgp_returns_bernoulli_draws():
    dgp = dgps.REGISTRY["D3-bin"]
    rng = np.random.default_rng(0)
    _x, y, mu = dgps.generate(dgp, rng, 2000, None, FAKE_DIAGNOSTICS)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert (mu > 0.0).all() and (mu < 1.0).all()
    # The empirical rate of 1s should track the mean fitted probability.
    assert y.mean() == pytest.approx(mu.mean(), abs=0.05)


def test_generate_rejects_a_noise_level_the_dgp_does_not_have():
    dgp = dgps.REGISTRY["D6"]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="noise level"):
        dgps.generate(dgp, rng, 100, "lo", FAKE_DIAGNOSTICS)


def test_load_diagnostics_missing_file_gives_a_clear_error(tmp_path):
    missing = tmp_path / "diagnostics.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        dgps.load_diagnostics(missing)


def test_committed_diagnostics_json_has_every_registry_key():
    diagnostics = dgps.load_diagnostics()
    for name in dgps.REGISTRY:
        assert name in diagnostics, name
