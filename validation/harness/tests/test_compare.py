"""Pure-Python tests for compare.py; no R needed. Boundary values are exact
(right at a documented tolerance), so a change to a tolerance constant or to
a comparison operator (> vs >=, relative vs absolute) fails a test here.
TestStepsFromTrace parses two stored trace logs (also used by
test_trace_parse.py); it needs no R either, since these are committed
files."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from compare import (
    COEF_NORMWISE_REL,
    KAPPA_RSS_LIMIT,
    Difference,
    compare_defaults,
    compare_fit,
    compare_forward_steps,
    condition_number,
    is_near_tie,
    removed_sequence,
    steps_from_trace,
)
from trace_parse import parse_trace

_HARNESS_DIR = Path(__file__).resolve().parents[1]
TRACE9 = _HARNESS_DIR.parent / "legacy" / "out" / "trace9.txt"
TRACE_DEGREE2 = Path(__file__).resolve().parent / "data" / "trace_degree2.txt"


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

    def test_mismatched_lengths_are_reported_not_hidden(self):
        # A length (shape) mismatch must never look like agreement.
        diffs = compare_fit(
            {"rss_per_subset": [1.0, 2.0]}, {"rss_per_subset": [1.0, 2.0, 3.0]}
        )
        assert len(diffs) == 1
        assert diffs[0].field == "rss_per_subset"
        assert "shape mismatch" in diffs[0].detail
        assert diffs[0].metric is None  # no meaningful magnitude for this

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

    def test_kappa_just_over_the_limit_scales_the_tolerance_only_slightly(self):
        # Just past the limit, the tolerance is barely scaled (kappa /
        # limit is barely above 1), so a gap that fails at the base
        # tolerance still fails here, but now labeled "numeric".
        a, b = self._coef_pair(2e-6)  # fails at the base tolerance (1e-6)
        diffs = compare_fit({"coef": a}, {"coef": b}, kappa=1e5 * 1.0001)
        assert len(diffs) == 1
        assert diffs[0].label == "numeric"
        assert diffs[0].tolerance == pytest.approx(COEF_NORMWISE_REL * 1.0001)

    def test_a_larger_kappa_scales_the_tolerance_further(self):
        # A large enough kappa scales the tolerance past a gap that fails
        # at the base tolerance, showing the comparison still ran (and
        # passed at the scaled tolerance), not that it was skipped.
        a, b = self._coef_pair(2e-6)  # fails at the base tolerance
        diffs = compare_fit({"coef": a}, {"coef": b}, kappa=1e5 * 3)
        assert diffs == []

    def test_kappa_scaling_cannot_hide_a_huge_gap(self):
        a, b = self._coef_pair(1.0)  # fails at any reasonable tolerance
        diffs = compare_fit({"coef": a}, {"coef": b}, kappa=1e5 * 1.0001)
        assert len(diffs) == 1
        assert diffs[0].label == "numeric"
        assert diffs[0].tolerance > COEF_NORMWISE_REL

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

    def test_kappa_over_the_rss_limit_scales_gcv_rsq_grsq_and_labels_numeric(self):
        # A huge, unambiguous gap: fails even at the scaled tolerance, so
        # the comparison plainly still ran above the limit (it did not
        # just skip and report nothing), each labeled "numeric".
        a = {"gcv": 2.0, "rsq": 0.0, "grsq": 0.0}
        b = {"gcv": 1.0, "rsq": 1.0, "grsq": 1.0}
        diffs = compare_fit(a, b, kappa=1e6 * 1.0001)
        assert {d.field for d in diffs} == {"gcv", "rsq", "grsq"}
        assert all(d.label == "numeric" for d in diffs)


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


HINGE_PAIR = frozenset({1, -1})
LINEAR = frozenset({2})


class TestCompareForwardSteps:
    def _step(self, parent=0, pred=0, knot=0.5, direction=HINGE_PAIR, **extra):
        return {
            "parent": parent,
            "pred": pred,
            "knot": knot,
            "direction": direction,
            **extra,
        }

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

    @pytest.mark.parametrize(
        "a_step,b_step",
        [
            # Mutation: dropping "parent" from the comparison would miss this.
            ({"parent": 0}, {"parent": 5}),
            # Mutation: dropping "pred" would miss this.
            ({"pred": 0}, {"pred": 5}),
            # Mutation: dropping "direction" (or the old "kind") would miss
            # this: a single right hinge is not the same choice as a pair,
            # even at the identical knot (the F5 case the matched mode has
            # to show, per the tolerance-table review).
            ({"direction": frozenset({1})}, {"direction": HINGE_PAIR}),
        ],
    )
    def test_a_mismatch_in_any_one_key_alone_is_caught(self, a_step, b_step):
        a = [self._step(**a_step)]
        b = [self._step(**b_step)]
        result = compare_forward_steps(a, b)
        assert result.first_mismatch == 0, (
            "a mismatch confined to one key must still be caught"
        )

    def test_a_mismatch_explained_by_a_near_tie_is_reported_as_a_tie_not_hidden(self):
        # Finding 4: a near-tie with different choices is not silently
        # accepted; it becomes a Difference labeled "tie", carrying both
        # candidate RSS values, so DIFFERENCES.md has something to show.
        a = [
            self._step(
                knot=0.5,
                best_rss=10.0,
                second_best_rss=10.0 + 1e-9,
                rss_before=100.0,
                flags={"bx1g": True, "cov_col_g": True, "tol_g": True, "max_g": True},
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
        assert len(result.differences) == 1
        d = result.differences[0]
        assert d.label == "tie"
        assert d.candidate_rss == (10.0, 10.000000001)
        assert d.flags == {
            "bx1g": True,
            "cov_col_g": True,
            "tol_g": True,
            "max_g": True,
        }

    def test_near_tie_on_side_b_only_is_still_caught(self):
        # Mutation: checking only side a's candidates for a near-tie would
        # miss a tie that only side b's log shows.
        a = [
            self._step(knot=0.5, best_rss=10.0, second_best_rss=50.0, rss_before=100.0)
        ]
        b = [
            self._step(
                knot=0.9, best_rss=10.0, second_best_rss=10.0 + 1e-9, rss_before=100.0
            )
        ]
        result = compare_forward_steps(a, b)
        assert result.first_near_tie == 0

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

    def test_a_length_mismatch_with_no_earlier_stop_is_reported(self):
        # Finding 3: the plan expects the two forward passes to sometimes
        # end at different sizes; that must be visible, not silently
        # truncated to the shorter log with nothing said about it.
        a = [self._step()]
        b = [self._step(), self._step(parent=1)]
        result = compare_forward_steps(a, b)
        assert result.matched == [0]
        assert result.first_mismatch is None
        assert result.first_near_tie is None
        assert len(result.differences) == 1
        assert result.differences[0].field == "forward_step_count"
        assert result.differences[0].a == 1 and result.differences[0].b == 2

    def test_no_length_mismatch_difference_when_a_real_mismatch_already_stopped_it(
        self,
    ):
        # The length-mismatch check must not add a second, redundant
        # Difference on top of a genuine structural mismatch already found.
        a = [self._step(knot=0.1)]
        b = [self._step(knot=0.2), self._step(parent=1)]
        result = compare_forward_steps(a, b)
        assert result.first_mismatch == 0
        assert len(result.differences) == 1
        assert result.differences[0].field == "forward_step"

    def test_missing_rss_fields_never_produce_a_false_tie(self):
        a = [self._step(knot=0.1)]
        b = [self._step(knot=0.2)]
        result = compare_forward_steps(a, b)
        assert result.first_mismatch == 0
        assert result.first_near_tie is None

    def test_continuing_after_the_first_mismatch_would_be_a_bug(self):
        # Mutation: not `break`-ing after the first mismatch would compare
        # every remaining step too, reporting more than one difference for
        # what should be a single stop.
        a = [self._step(knot=0.1), self._step(knot=0.3), self._step(knot=0.5)]
        b = [self._step(knot=0.2), self._step(knot=0.4), self._step(knot=0.6)]
        result = compare_forward_steps(a, b)
        assert len(result.differences) == 1
        assert result.first_mismatch == 0


class TestStepsFromTrace:
    """steps_from_trace against the two committed traces test_trace_parse.py
    also uses; no R needed, since these are stored files."""

    def test_trace9_first_step_matches_the_summary_line(self):
        log = parse_trace(TRACE9, sample_var_y=None)
        steps = steps_from_trace(log)
        assert len(steps) == len(log.steps)
        first = steps[0]
        assert first["parent"] == 1 and first["pred"] == 1
        assert first["direction"] == HINGE_PAIR
        assert first["knot"] == pytest.approx(0.77466, abs=1e-3)
        assert first["best_rss"] is not None
        assert first["rss_before"] == pytest.approx(199.0)

    def test_degree2_trace_direction_is_linear_or_a_hinge_pair(self):
        log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
        steps = steps_from_trace(log)
        assert steps
        for step in steps:
            assert step["direction"] in (HINGE_PAIR, LINEAR, frozenset())

    def test_a_step_with_no_live_search_gives_an_empty_direction(self):
        # A degenerate step (every search skipped) must not crash; it is
        # simply unusable for a structural comparison at that index.
        fake_step = SimpleNamespace(
            searches=[SimpleNamespace(skipped_reason="x", pred=None)]
        )
        fake_log = SimpleNamespace(steps=[fake_step])
        steps = steps_from_trace(fake_log)
        assert steps == [
            {
                "parent": None,
                "pred": None,
                "direction": frozenset(),
                "knot": None,
                "best_rss": None,
                "second_best_rss": None,
                "rss_before": None,
                "flags": None,
            }
        ]


class TestRemovedSequence:
    def test_from_pruning_removed_directly(self):
        assert removed_sequence({"pruning_removed": [3, 1, 2]}) == [3, 1, 2]

    def test_derived_from_a_prune_terms_matrix(self):
        # Row i = the term indices present at subset size i + 1 (0-padded);
        # size 1 -> {1}, size 2 -> {1, 3}, size 3 -> {1, 2, 3}: term 2 is
        # added last going up in size, so it is removed first going down.
        prune_terms = [[1, 0, 0], [1, 3, 0], [1, 2, 3]]
        assert removed_sequence({"prune_terms": prune_terms}) == [2, 3]

    def test_neither_field_present_gives_none(self):
        assert removed_sequence({}) is None

    def test_ambiguous_prune_terms_gives_none_not_a_guess(self):
        # Two terms differ between consecutive rows: no single term can be
        # identified as "the one removed", so this must not guess.
        prune_terms = [[1, 0], [2, 3]]
        assert removed_sequence({"prune_terms": prune_terms}) is None


class TestCompareFitPruningRemoved:
    def test_same_sequence_is_no_difference(self):
        # b's matrix derives to [2, 3] (test_derived_from_a_prune_terms_
        # matrix above shows the same matrix and checks that directly).
        a = {"pruning_removed": [2, 3]}
        b = {"prune_terms": [[1, 0, 0], [1, 3, 0], [1, 2, 3]]}
        assert compare_fit(a, b) == []

    def test_a_different_removal_choice_is_reported_at_its_step(self):
        a = {"pruning_removed": [3, 1, 2]}
        b = {"pruning_removed": [3, 2, 1]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert diffs[0].field == "pruning_removed"
        assert diffs[0].step == 1

    def test_length_mismatch_is_reported(self):
        a = {"pruning_removed": [3, 1]}
        b = {"pruning_removed": [3, 1, 2]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert "length mismatch" in diffs[0].detail


class TestCompareFitForwardRssPath:
    def test_exactly_at_the_relative_tolerance_passes(self):
        b = [10.0, 5.0, 2.0]
        a = [v * (1 + 1e-8) for v in b]
        assert compare_fit({"fwd_rss": a}, {"fwd_rss": b}) == []

    def test_just_over_the_relative_tolerance_fails(self):
        b = [10.0, 5.0, 2.0]
        a = [v * (1 + 2e-8) for v in b]
        diffs = compare_fit({"fwd_rss": a}, {"fwd_rss": b})
        assert len(diffs) == 1 and diffs[0].field == "fwd_rss"

    def test_length_mismatch_is_reported_not_hidden(self):
        diffs = compare_fit({"fwd_rss": [1.0, 2.0]}, {"fwd_rss": [1.0, 2.0, 3.0]})
        assert len(diffs) == 1
        assert "shape mismatch" in diffs[0].detail

    def test_kappa_over_the_limit_scales_and_labels_numeric(self):
        b = [10.0, 5.0, 2.0]
        a = [v * (1 + 1e-3) for v in b]  # fails at the base tolerance
        diffs = compare_fit({"fwd_rss": a}, {"fwd_rss": b}, kappa=KAPPA_RSS_LIMIT * 1e6)
        assert diffs == []  # the huge kappa scales the tolerance well past 1e-3


class TestNeverHidesADifference:
    """Finding 3: NaN on one side, an infinity in an array, and a shape
    mismatch must all be reported, not silently pass."""

    def test_nan_on_one_side_is_a_difference_not_a_pass(self):
        assert compare_fit({"gcv": float("nan")}, {"gcv": 1.0}) != []
        assert compare_fit({"rsq": float("nan")}, {"rsq": 0.9}) != []
        assert (
            compare_fit({"glm_coef": [float("nan")]}, {"glm_coef": [1.0]}, glm=True)
            != []
        )

    def test_matching_nan_at_the_same_position_is_not_a_difference(self):
        assert compare_fit({"gcv": float("nan")}, {"gcv": float("nan")}) == []

    def test_an_infinity_in_an_array_does_not_hide_the_other_entries(self):
        # np.max would otherwise see inf - inf = NaN in the difference
        # array and hide a real mismatch at a different position.
        a = {"gcv_per_subset": [1.0, 2.0, float("inf")]}
        b = {"gcv_per_subset": [1.0, 5.0, float("inf")]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert diffs[0].field == "gcv_per_subset"
        assert diffs[0].metric is not None and diffs[0].metric > 0

    def test_matching_infinities_at_the_same_position_are_not_a_difference(self):
        a = {"gcv_per_subset": [1.0, float("inf")]}
        b = {"gcv_per_subset": [1.0, float("inf")]}
        assert compare_fit(a, b) == []

    def test_an_infinity_on_only_one_side_is_a_difference(self):
        a = {"gcv_per_subset": [1.0, float("inf")]}
        b = {"gcv_per_subset": [1.0, 2.0]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert "infinite" in diffs[0].detail

    def test_a_shape_mismatch_in_coef_is_reported_not_broadcast(self):
        # Mutation risk: numpy would silently broadcast (1, 1) against
        # (3, 1) rather than raise, so this must be checked explicitly.
        diffs = compare_fit({"coef": [[1.0]]}, {"coef": [[1.0], [2.0], [3.0]]})
        assert len(diffs) == 1
        assert "shape mismatch" in diffs[0].detail

    def test_without_sd_y_the_fitted_comparison_is_skipped_with_no_error(self):
        # Unlike the other fields, there is genuinely no way to compare
        # fitted values without knowing sd(y), so this one really is
        # skipped (not "hidden": there is nothing to hide, no shared
        # tolerance exists without it).
        a, b = {"fitted": [1.0, 100.0]}, {"fitted": [1.0, 2.0]}
        assert compare_fit(a, b) == []


class TestCompareFitNormwiseVsElementwiseMutation:
    def test_normwise_and_elementwise_metrics_can_disagree(self):
        # Mutation: replacing the normwise coefficient metric with the
        # elementwise max relative error would pass this case (one
        # coefficient near 0 makes its own relative error explode) even
        # though the normwise metric correctly says the vectors agree well.
        a = [[1000.0], [1000.0], [1e-9]]
        b = [[1000.0 * (1 + 1e-9)], [1000.0 * (1 + 1e-9)], [2e-9]]
        assert compare_fit({"coef": a}, {"coef": b}) == []


class TestCompareFitGlmPredTestMutation:
    def test_pred_test_probabilities_are_compared_when_glm_is_true(self):
        # Mutation: dropping "pred_test" from the GLM probability loop
        # (leaving only "pred_train") would miss this.
        a = {"pred_test": [0.5 + 2e-6]}
        b = {"pred_test": [0.5]}
        diffs = compare_fit(a, b, glm=True)
        assert len(diffs) == 1
        assert diffs[0].field == "pred_test_prob"


class TestCompareDefaults:
    def test_reports_rsq_gcv_and_term_count_for_each_side(self):
        a = {"rsq": 0.9, "gcv": 0.05, "selected_terms": [0, 1, 2]}
        b = {"rsq": 0.8, "gcv": 0.10, "dirs": [[0], [1]]}
        report = compare_defaults(a, b, dataset="S04")
        assert report == {
            "dataset": "S04",
            "a": {"rsq": 0.9, "gcv": 0.05, "n_terms": 3, "test_error": None},
            "b": {"rsq": 0.8, "gcv": 0.10, "n_terms": 2, "test_error": None},
        }

    def test_reports_test_error_when_y_test_is_given(self):
        a = {"pred_test": [1.0, 2.0, 3.0]}
        b = {"pred_test": [1.0, 2.0, 4.0]}
        report = compare_defaults(a, b, y_test=[1.0, 2.0, 3.0])
        assert report["a"]["test_error"] == pytest.approx(0.0)
        assert report["b"]["test_error"] == pytest.approx(1 / 3)


def test_difference_is_a_plain_dataclass_with_the_documented_fields():
    d = Difference(
        field="x",
        a=1,
        b=2,
        metric=0.5,
        tolerance=0.1,
        step=3,
        dataset="S01",
        candidate_rss=(1.0, 1.1),
        flags={"tol_g": False},
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
    assert d.candidate_rss == (1.0, 1.1)
    assert d.flags == {"tol_g": False}
    assert d.label is None
