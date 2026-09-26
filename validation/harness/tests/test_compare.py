"""Pure-Python tests for compare.py; no R needed. Boundary values are exact
(right at a documented tolerance), so a change to a tolerance constant or to
a comparison operator (> vs >=, relative vs absolute) fails a test here."""

import numpy as np
import pytest
from compare import (
    Difference,
    compare_fit,
    compare_forward_steps,
    condition_number,
    is_near_tie,
)


class TestConditionNumber:
    def test_identity_is_one(self):
        assert condition_number(np.eye(3)) == pytest.approx(1.0)

    def test_matches_a_hand_computed_value(self):
        # diag(1, 1e6): singular values 1e6 and 1, so kappa = 1e6 / 1.
        assert condition_number(np.diag([1.0, 1e6])) == pytest.approx(1e6, rel=1e-9)


class TestIsNearTie:
    def test_exactly_at_the_threshold_is_not_a_tie(self):
        # "differ by less than 1e-7 times the RSS before the step": equal to
        # the threshold does not qualify. best=0 avoids floating-point
        # rounding from adding then subtracting a much larger number, so the
        # gap here is exactly rel * rss_before.
        rss_before = 100.0
        gap = 1e-7 * rss_before
        assert is_near_tie(best=0.0, second_best=gap, rss_before=rss_before) is False

    def test_just_below_the_threshold_is_a_tie(self):
        rss_before = 100.0
        gap = 1e-7 * rss_before * 0.999999
        assert is_near_tie(best=0.0, second_best=gap, rss_before=rss_before) is True

    def test_just_above_the_threshold_is_not_a_tie(self):
        rss_before = 100.0
        gap = 1e-7 * rss_before * 1.000001
        assert is_near_tie(best=0.0, second_best=gap, rss_before=rss_before) is False

    def test_missing_second_best_is_not_a_tie(self):
        assert is_near_tie(best=10.0, second_best=None, rss_before=100.0) is False

    def test_zero_rss_before_is_not_a_tie(self):
        assert is_near_tie(best=0.0, second_best=0.0, rss_before=0.0) is False

    def test_order_does_not_matter(self):
        assert is_near_tie(
            best=10.0, second_best=10.0 - 1e-10, rss_before=1.0
        ) is is_near_tie(best=10.0 - 1e-10, second_best=10.0, rss_before=1.0)


class TestCompareFitSelectedTerms:
    def test_same_set_in_different_order_is_no_difference(self):
        a = {"selected_terms": [0, 3, 1]}
        b = {"selected_terms": [1, 0, 3]}
        assert compare_fit(a, b) == []

    def test_different_set_is_reported_with_sorted_choices(self):
        a = {"selected_terms": [0, 1, 2]}
        b = {"selected_terms": [0, 1, 3]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert diffs[0].field == "selected_terms"
        assert diffs[0].a == [0, 1, 2]
        assert diffs[0].b == [0, 1, 3]

    def test_missing_on_either_side_is_skipped_not_an_error(self):
        assert compare_fit({"selected_terms": None}, {"selected_terms": [0, 1]}) == []
        assert compare_fit({"selected_terms": [0, 1]}, {}) == []


class TestCompareFitPerSubsetArrays:
    @pytest.mark.parametrize("field_name", ["rss_per_subset", "gcv_per_subset"])
    def test_exactly_at_the_relative_tolerance_passes(self, field_name):
        b = [1.0, 2.0, 100.0]
        a = [v * (1 + 1e-8) for v in b]
        assert compare_fit({field_name: a}, {field_name: b}) == []

    @pytest.mark.parametrize("field_name", ["rss_per_subset", "gcv_per_subset"])
    def test_just_over_the_relative_tolerance_fails(self, field_name):
        b = [1.0, 2.0, 100.0]
        a = [v * (1 + 2e-8) for v in b]
        diffs = compare_fit({field_name: a}, {field_name: b})
        assert len(diffs) == 1
        assert diffs[0].field == field_name
        assert diffs[0].tolerance == pytest.approx(1e-8)

    def test_mismatched_lengths_are_skipped_not_an_error(self):
        assert (
            compare_fit(
                {"rss_per_subset": [1.0, 2.0]}, {"rss_per_subset": [1.0, 2.0, 3.0]}
            )
            == []
        )

    def test_empty_arrays_are_skipped(self):
        assert compare_fit({"rss_per_subset": []}, {"rss_per_subset": []}) == []


class TestCompareFitCoefficients:
    def _coef_pair(self, rel_error):
        b = np.array([[1.0], [2.0], [-3.0]])
        a = b * (1 + rel_error) if rel_error else b
        # normwise relative = ||a - b|| / ||b||; scale rel_error to hit it exactly.
        return a.tolist(), b.tolist()

    def test_exactly_at_the_normwise_tolerance_passes(self):
        a, b = self._coef_pair(1e-6)
        assert compare_fit({"coef": a}, {"coef": b}) == []

    def test_just_over_the_normwise_tolerance_fails(self):
        a, b = self._coef_pair(2e-6)
        diffs = compare_fit({"coef": a}, {"coef": b})
        assert len(diffs) == 1
        assert diffs[0].field == "coef"

    def test_kappa_at_the_limit_is_still_compared(self):
        a, b = self._coef_pair(2e-6)  # would fail, if compared
        diffs = compare_fit({"coef": a}, {"coef": b}, kappa=1e5)
        assert len(diffs) == 1

    def test_kappa_just_over_the_limit_skips_the_comparison(self):
        a, b = self._coef_pair(2e-6)  # would fail, if compared
        diffs = compare_fit({"coef": a}, {"coef": b}, kappa=1e5 * 1.0001)
        assert diffs == []

    def test_no_kappa_given_always_compares(self):
        a, b = self._coef_pair(2e-6)
        diffs = compare_fit({"coef": a}, {"coef": b})
        assert len(diffs) == 1


class TestCompareFitGcvRsqGrsq:
    def test_gcv_uses_relative_tolerance_not_absolute(self):
        # A large absolute gap (5e-3) that is well inside the 1e-8 relative
        # tolerance at this scale (5e-3 / 1e6 = 5e-9): passes only if the
        # comparison is truly relative, not absolute.
        a, b = {"gcv": 1e6 + 5e-3}, {"gcv": 1e6}
        assert compare_fit(a, b) == []

    def test_gcv_just_over_the_relative_tolerance_fails(self):
        a, b = {"gcv": 1e6 * (1 + 2e-8)}, {"gcv": 1e6}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1 and diffs[0].field == "gcv"

    @pytest.mark.parametrize("field_name", ["rsq", "grsq"])
    def test_rsq_and_grsq_use_absolute_tolerance_not_relative(self, field_name):
        # A tiny absolute gap (1e-10) that would fail badly under a relative
        # check at this scale (1e-10 / 2e-10 = 0.5): passes only if the
        # comparison is truly absolute.
        a, b = {field_name: 2e-10}, {field_name: 1e-10}
        assert compare_fit(a, b) == []

    @pytest.mark.parametrize("field_name", ["rsq", "grsq"])
    def test_rsq_and_grsq_just_over_the_absolute_tolerance_fails(self, field_name):
        a, b = {field_name: 0.5 + 2e-8}, {field_name: 0.5}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1 and diffs[0].field == field_name

    def test_kappa_over_the_rss_limit_skips_gcv_rsq_grsq(self):
        a = {"gcv": 2.0, "rsq": 0.0, "grsq": 0.0}
        b = {"gcv": 1.0, "rsq": 1.0, "grsq": 1.0}  # all would fail, if compared
        assert compare_fit(a, b, kappa=1e6 * 1.0001) == []


class TestCompareFitFittedAndPredictions:
    @pytest.mark.parametrize("field_name", ["fitted", "pred_test"])
    def test_exactly_at_the_sd_scaled_tolerance_passes(self, field_name):
        sd_y = 4.0
        a, b = {field_name: [1.0, 2.0]}, {field_name: [1.0, 2.0 - 1e-8 * sd_y]}
        assert compare_fit(a, b, sd_y=sd_y) == []

    @pytest.mark.parametrize("field_name", ["fitted", "pred_test"])
    def test_just_over_the_sd_scaled_tolerance_fails(self, field_name):
        sd_y = 4.0
        a, b = {field_name: [1.0, 2.0]}, {field_name: [1.0, 2.0 - 2e-8 * sd_y]}
        diffs = compare_fit(a, b, sd_y=sd_y)
        assert len(diffs) == 1 and diffs[0].field == field_name

    def test_without_sd_y_the_comparison_is_skipped_entirely(self):
        a, b = {"fitted": [1.0, 100.0]}, {"fitted": [1.0, 2.0]}  # wildly different
        assert compare_fit(a, b) == []


class TestCompareFitGlm:
    def test_glm_false_skips_glm_fields_even_if_very_different(self):
        a = {"glm_coef": [10.0], "pred_train": [0.99]}
        b = {"glm_coef": [-10.0], "pred_train": [0.01]}
        assert compare_fit(a, b, glm=False) == []

    def test_glm_true_compares_glm_coef_with_relative_tolerance(self):
        a = {"glm_coef": [[1.0 * (1 + 2e-5)]]}
        b = {"glm_coef": [[1.0]]}
        diffs = compare_fit(a, b, glm=True)
        assert len(diffs) == 1 and diffs[0].field == "glm_coef"

    def test_glm_true_compares_fitted_probabilities_with_absolute_tolerance(self):
        a = {"pred_train": [0.5 + 2e-7]}
        b = {"pred_train": [0.5]}
        diffs = compare_fit(a, b, glm=True)
        assert len(diffs) == 1 and diffs[0].field == "pred_train_prob"


class TestCompareForwardSteps:
    def _step(self, parent=0, pred=0, knot=0.5, kind="hinge", **extra):
        return {"parent": parent, "pred": pred, "knot": knot, "kind": kind, **extra}

    def test_all_matching_steps_gives_no_mismatch_or_tie(self):
        a = [self._step(), self._step(parent=1)]
        b = [self._step(), self._step(parent=1)]
        result = compare_forward_steps(a, b)
        assert result.matched == [0, 1]
        assert result.first_mismatch is None
        assert result.first_near_tie is None
        assert result.differences == []

    def test_a_genuine_mismatch_stops_the_comparison_there(self):
        a = [self._step(), self._step(knot=0.7)]
        b = [self._step(), self._step(knot=0.9)]
        result = compare_forward_steps(a, b)
        assert result.matched == [0]
        assert result.first_mismatch == 1
        assert result.first_near_tie is None
        assert len(result.differences) == 1
        assert result.differences[0].step == 1

    def test_knot_is_compared_exactly_not_with_a_tolerance(self):
        a = [self._step(knot=0.30000001)]
        b = [self._step(knot=0.30000002)]
        result = compare_forward_steps(a, b)
        assert result.first_mismatch == 0

    def test_a_mismatch_explained_by_a_near_tie_is_not_a_mismatch(self):
        a = [
            self._step(
                knot=0.5, best_rss=10.0, second_best_rss=10.0 + 1e-9, rss_before=100.0
            )
        ]
        b = [
            self._step(
                knot=0.9, best_rss=10.0, second_best_rss=10.0 + 1e-9, rss_before=100.0
            )
        ]
        result = compare_forward_steps(a, b)
        assert result.first_near_tie == 0
        assert result.first_mismatch is None
        assert result.differences == []

    def test_a_matching_step_that_is_also_a_near_tie_still_stops_there(self):
        # Both sides picked the same candidate, but it was only barely ahead
        # of the runner-up, so later steps are not reliably comparable.
        tied_step = self._step(
            best_rss=10.0, second_best_rss=10.0 + 1e-9, rss_before=100.0
        )
        a = [tied_step, self._step(parent=5)]
        b = [tied_step, self._step(parent=9)]
        result = compare_forward_steps(a, b)
        assert result.first_near_tie == 0
        assert 0 in result.matched
        assert result.first_mismatch is None

    def test_stops_at_the_shorter_logs_length_with_no_error(self):
        a = [self._step()]
        b = [self._step(), self._step(parent=1)]
        result = compare_forward_steps(a, b)
        assert result.matched == [0]
        assert result.first_mismatch is None
        assert result.first_near_tie is None

    def test_missing_rss_fields_never_produce_a_false_tie(self):
        a = [self._step(knot=0.1)]
        b = [self._step(knot=0.2)]
        result = compare_forward_steps(a, b)
        assert result.first_mismatch == 0
        assert result.first_near_tie is None


def test_difference_is_a_plain_dataclass_with_the_documented_fields():
    d = Difference(
        field="x", a=1, b=2, metric=0.5, tolerance=0.1, step=3, dataset="S01"
    )
    assert (d.field, d.a, d.b, d.metric, d.tolerance, d.step, d.dataset) == (
        "x",
        1,
        2,
        0.5,
        0.1,
        3,
        "S01",
    )
    assert d.label is None
