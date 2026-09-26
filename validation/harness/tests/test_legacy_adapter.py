"""Tests for legacy_adapter.py. Every test in TestFitLegacy needs the legacy
venv (`validation/legacy/make_venv.sh`'s `.venv-legacy`, with mars-earth
1.0.4), so it is marked ``external`` (gate A/B and CI exclude it; gate C and
manual runs include it); it is not skipped when the venv happens to be
missing (``fit_legacy()`` then raises ``FileNotFoundError`` with the command
to run)."""

import legacy_adapter
import numpy as np
import pytest
from legacy_adapter import fit_legacy


@pytest.mark.external
class TestFitLegacy:
    def _data(self, rng, n=100):
        X = rng.uniform(size=(n, 2))
        y = 2 * np.maximum(0, X[:, 0] - 0.3) * np.maximum(
            0, X[:, 1] - 0.4
        ) + 0.2 * rng.normal(size=n)
        return X, y

    def test_result_has_the_documented_schema(self):
        X, y = self._data(np.random.default_rng(0))
        result = fit_legacy(X, y, max_degree=2, penalty=3.0)
        expected_keys = {
            "dirs",
            "cuts",
            "selected_terms",
            "prune_terms",
            "coef",
            "rss",
            "gcv",
            "fwd_dirs",
            "fwd_cuts",
            "fwd_rss",
            "rss_per_subset",
            "gcv_per_subset",
            "n_terms",
        }
        assert expected_keys <= set(result)
        assert result["selected_terms"] is None  # not applicable; see the docstring
        assert result["prune_terms"] is None
        n_terms = result["n_terms"]
        assert np.array(result["dirs"]).shape == (n_terms, 2)
        assert np.array(result["cuts"]).shape == (n_terms, 2)
        assert np.array(result["coef"]).shape == (n_terms, 1)
        assert np.isfinite(result["rss"])
        assert np.isfinite(result["gcv"])

    def test_forward_snapshot_has_at_least_as_many_terms_as_the_final_model(self):
        # record_.fwd_basis_[-1] (pre-pruning) can only have more terms than
        # the pruned basis_, never fewer.
        X, y = self._data(np.random.default_rng(1))
        result = fit_legacy(X, y, max_degree=2, penalty=3.0)
        assert np.array(result["fwd_dirs"]).shape[0] >= result["n_terms"]

    def test_forward_rss_path_is_monotonically_non_increasing(self):
        # Adding a column to a least-squares fit cannot raise its RSS; this
        # holds regardless of record_.fwd_rss_'s own step granularity (it
        # need not match fwd_dirs's row count, since the legacy code logs
        # the two at different points in the forward pass).
        X, y = self._data(np.random.default_rng(1))
        result = fit_legacy(X, y, max_degree=2, penalty=3.0)
        fwd_rss = np.asarray(result["fwd_rss"])
        assert len(fwd_rss) >= 2
        assert np.all(np.diff(fwd_rss) <= 1e-8)

    def test_degree_one_gives_only_direction_codes_0_1_and_minus_1(self):
        # max_degree=1 (the plan's default): no linear (code 2) or product
        # terms in dirs, since allow_linear defaults to True in the legacy
        # code but only matters when it actually fires; this checks the
        # simplest, unambiguous case (no linear terms requested).
        X, y = self._data(np.random.default_rng(2))
        result = fit_legacy(X, y, max_degree=1, allow_linear=False)
        codes = {c for row in result["dirs"] for c in row}
        assert codes <= {-1, 0, 1}

    def test_pruning_trace_lists_are_present_and_finite(self):
        X, y = self._data(np.random.default_rng(3))
        result = fit_legacy(X, y, max_degree=1)
        assert len(result["rss_per_subset"]) > 0
        assert len(result["gcv_per_subset"]) > 0
        assert all(np.isfinite(result["rss_per_subset"]))

    def test_a_bad_keyword_argument_raises_runtimeerror(self):
        X, y = self._data(np.random.default_rng(4))
        with pytest.raises(RuntimeError):
            fit_legacy(X, y, not_a_real_kwarg=1)


def test_missing_legacy_venv_raises_filenotfounderror(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy_adapter, "LEGACY_PYTHON", tmp_path / "no-such-python")
    with pytest.raises(FileNotFoundError, match=r"make_venv\.sh"):
        fit_legacy(np.zeros((5, 1)), np.zeros(5))
