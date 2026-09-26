"""Tests for blackbox.py. Every test here calls ``Rscript blackbox.R`` for
real, so all of them need R and the earth (and nnet) packages; they are
marked ``external`` (gate A/B and CI exclude it; gate C and manual runs
include it)."""

import blackbox as bb
import numpy as np
import pytest

pytestmark = pytest.mark.external


@pytest.fixture
def hinge_data():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(size=(n, 1))
    y = (
        2 * np.maximum(0, x[:, 0] - 0.3)
        - 3 * np.maximum(0, x[:, 0] - 0.7)
        + rng.normal(scale=0.05, size=n)
    )
    return x, y


class TestGetGcv:
    def test_is_vectorized_over_the_subset_size(self):
        gcv = bb.get_gcv([10, 5, 3, 3.5], [1, 2, 3, 4], 2, 60)
        assert gcv.shape == (4,)
        assert np.all(np.isfinite(gcv))

    def test_penalty_minus_one_gives_rss_over_n(self):
        gcv = bb.get_gcv([10, 5], [1, 2], -1, 60)
        assert gcv == pytest.approx([10 / 60, 5 / 60])

    def test_penalty_minus_one_is_bit_exact_against_rss_over_n(self):
        # Round-2 review, PR #36 finding 2: digits = I(17) reverted to NA in
        # blackbox.R's write_result rounds to 15 significant digits
        # (jsonlite 2.0.0), which the rel=1e-6 approx() above is far too
        # loose to catch. penalty = -1 is defined as GCV = RSS / ncases
        # exactly (get_gcv's own docstring), with no fitting or rounding of
        # its own on either side, so this checks bit equality, not approx.
        rss = 0.22733602246716966
        n = 60.0
        gcv = bb.get_gcv([rss], [3], -1.0, n)
        assert gcv[0] == rss / n

    def test_effective_parameters_at_or_above_n_gives_infinite_gcv(self):
        gcv = bb.get_gcv([10.0], [1], 2.0, 1)
        assert np.isinf(gcv[0])

    def test_a_larger_penalty_never_lowers_the_gcv_at_fixed_rss_and_terms(self):
        # Any sensible GCV penalizes the term count more as the penalty
        # grows; this checks that qualitative property (not a specific
        # formula, which get_gcv exists precisely so other code need not
        # reimplement) at a fixed RSS and a term count above 1, where the
        # penalty has something to bite on.
        low = bb.get_gcv([8.0], [4], 1.0, 60)[0]
        high = bb.get_gcv([8.0], [4], 4.0, 60)[0]
        assert high >= low

    def test_more_terms_at_the_same_rss_never_lowers_the_gcv(self):
        fewer = bb.get_gcv([8.0], [3], 2.0, 60)[0]
        more = bb.get_gcv([8.0], [6], 2.0, 60)[0]
        assert more >= fewer


class TestFitBxDirs:
    def test_returns_a_real_forward_basis(self, hinge_data):
        x, y = hinge_data
        result = bb.fit_bx_dirs(
            x,
            y,
            {
                "degree": 1,
                "pmethod": "none",
                "nk": 11,
                "thresh": 0,
                "minspan": 1,
                "endspan": 1,
                "fast.k": 0,
                "Auto.linpreds": False,
            },
        )
        n_terms = result["dirs"].shape[0]
        assert result["bx"].shape == (len(y), n_terms)
        assert result["cuts"].shape == (n_terms, 1)
        assert result["dirs"][0].tolist() == [0]  # row 0 is the intercept
        assert np.array_equal(result["bx"][:, 0], np.ones(len(y)))
        assert len(result["selected_terms"]) <= n_terms

    def test_accepts_a_multi_response_y(self, hinge_data):
        x, y = hinge_data
        Y = np.column_stack([y, -y])
        result = bb.fit_bx_dirs(x, Y, {"degree": 1, "nk": 11})
        assert result["bx"].shape[0] == len(y)


class TestPruningPass:
    def _fixed_basis(self, x, y):
        """A real forward-only fit's bx/dirs, for the pruning pass to prune."""
        result = bb.fit_bx_dirs(
            x,
            y,
            {
                "degree": 1,
                "pmethod": "none",
                "nk": 11,
                "thresh": 0,
                "minspan": 1,
                "endspan": 1,
                "fast.k": 0,
                "Auto.linpreds": False,
            },
        )
        return result["bx"], result["dirs"]

    def test_prunes_a_real_forward_basis(self, hinge_data):
        x, y = hinge_data
        bx, dirs = self._fixed_basis(x, y)
        result = bb.pruning_pass(x, y, bx, dirs, penalty=2.0)
        n_terms = bx.shape[1]
        assert result["rss_per_subset"].shape == (n_terms,)
        assert result["gcv_per_subset"].shape == (n_terms,)
        assert result["prune_terms"].shape == (n_terms, n_terms)
        assert 1 <= len(result["selected_terms"]) <= n_terms
        # rss.per.subset is indexed by subset size, from 1 term (index 0,
        # the worst fit) up to the full basis (the last index, the best).
        assert result["rss_per_subset"][0] >= result["rss_per_subset"][-1]

    def test_nprune_caps_the_selected_size(self, hinge_data):
        x, y = hinge_data
        bx, dirs = self._fixed_basis(x, y)
        result = bb.pruning_pass(x, y, bx, dirs, penalty=2.0, nprune=3)
        assert len(result["selected_terms"]) <= 3


class TestLmFit:
    def test_matches_numpy_lstsq(self, hinge_data):
        x, y = hinge_data
        X = np.column_stack([np.ones(len(y)), x[:, 0]])
        result = bb.lm_fit(X, y)
        expected, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert result["coefficients"].ravel() == pytest.approx(expected, rel=1e-8)
        assert result["rank"] == 2

    def test_residuals_have_the_shape_of_y(self, hinge_data):
        x, y = hinge_data
        X = np.column_stack([np.ones(len(y)), x[:, 0]])
        result = bb.lm_fit(X, y)
        assert result["residuals"].shape == (len(y), 1)

    def test_a_duplicated_column_gives_none_not_nan_for_the_aliased_coefficient(
        self, hinge_data
    ):
        # Rank-deficient x (two identical columns): R's lm.fit gives the
        # aliased coefficient as NA, a real "not estimable" marker, not the
        # same as a NaN; _desanitize must decode it to None, and lm_fit's
        # dtype=object coefficients must keep that None instead of numpy's
        # usual dtype=float cast silently turning it into nan.
        x, y = hinge_data
        X = np.column_stack([np.ones(len(y)), x[:, 0], x[:, 0]])
        result = bb.lm_fit(X, y)
        assert result["rank"] == 2
        coefficients = result["coefficients"].ravel().tolist()
        assert coefficients.count(None) == 1
        estimated = [c for c in coefficients if c is not None]
        assert len(estimated) == 2
        assert not any(np.isnan(c) for c in estimated)


class TestPredictEarth:
    def test_predicts_outside_the_training_range(self, hinge_data):
        x, y = hinge_data
        newx = np.array([[-1.0], [0.5], [2.0]])
        pred = bb.predict_earth(x, y, newx, earth_args={"degree": 1, "nk": 11})
        assert pred.shape == (3, 1)
        assert np.all(np.isfinite(pred))


class TestGlmFit:
    def test_matches_statsmodels_style_iwls_on_a_small_case(self):
        rng = np.random.default_rng(1)
        n = 200
        x = rng.uniform(size=n)
        X = np.column_stack([np.ones(n), x])
        p = 1 / (1 + np.exp(-(2 * x - 1)))
        y = (rng.uniform(size=n) < p).astype(float)
        result = bb.glm_fit(X, y)
        # No independent GLM solver is available here without a new
        # dependency, so this checks the fitted probabilities are sane and
        # improve on the intercept-only log loss, not exact coefficients
        # (test_glm.py, a later task, compares against this against pymars).
        fitted = result["fitted_values"]
        assert fitted.shape == (n,)
        assert np.all((fitted > 0) & (fitted < 1))
        base_rate = y.mean()
        ll_model = -np.mean(y * np.log(fitted) + (1 - y) * np.log1p(-fitted))
        ll_base = -np.mean(y * np.log(base_rate) + (1 - y) * np.log1p(-base_rate))
        assert ll_model < ll_base


class TestMultinomFit:
    def test_probabilities_sum_to_one_and_favor_the_true_class(self):
        rng = np.random.default_rng(2)
        n = 150
        x = rng.uniform(-3, 3, size=n)
        X = np.column_stack([np.ones(n), x])
        # A clearly separated 3-class problem by construction.
        y = np.where(x < -1, "lo", np.where(x > 1, "hi", "mid"))
        result = bb.multinom_fit(X, y)
        assert set(result["levels"]) == {"hi", "lo", "mid"}
        assert result["fitted"].shape == (n, 3)
        assert np.allclose(result["fitted"].sum(axis=1), 1.0)
        predicted = [result["levels"][i] for i in result["fitted"].argmax(axis=1)]
        accuracy = np.mean([p == truth for p, truth in zip(predicted, y, strict=True)])
        assert accuracy > 0.9

    def test_a_level_named_na_stays_the_string_na(self):
        # A class label that is itself the string "NA" is not R's missing
        # value; write_json's na="string" spells both the same way on the
        # wire ("NA"), so only treating "levels" as a string-only field (not
        # decoding sentinel tokens in it at all) tells them apart.
        rng = np.random.default_rng(3)
        n = 150
        x = rng.uniform(-3, 3, size=n)
        X = np.column_stack([np.ones(n), x])
        y = np.where(x < -1, "NA", np.where(x > 1, "hi", "mid"))
        result = bb.multinom_fit(X, y)
        assert set(result["levels"]) == {"hi", "NA", "mid"}
        assert all(level is not None for level in result["levels"])
