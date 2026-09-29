"""Pure-Python tests for compare.py; no R needed. Boundary values are exact
(right at a documented tolerance), so a change to a tolerance constant or to
a comparison operator (> vs >=, relative vs absolute) fails a test here.
TestStepsFromTrace parses a stored trace log together with its fit's
dirs/cuts (committed alongside it, tests/data/trace_steps_fit.json); it
needs no R either, since these are committed files."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from compare import (
    COEF_NORMWISE_REL,
    FWD_RSS_REL,
    KAPPA_CAP,
    KAPPA_RSS_LIMIT,
    Difference,
    Skip,
    _dirs_row_groups,
    compare_defaults,
    compare_fit,
    compare_forward_steps,
    condition_number,
    is_near_tie,
    removed_sequence,
    steps_from_trace,
)
from trace_parse import parse_trace

_DATA_DIR = Path(__file__).resolve().parent / "data"
TRACE_STEPS = _DATA_DIR / "trace_steps.txt"
TRACE_STEPS_FIT = _DATA_DIR / "trace_steps_fit.json"
TRACE_STEPS_SLOTGAP = _DATA_DIR / "trace_steps_slotgap.txt"
TRACE_STEPS_SLOTGAP_FIT = _DATA_DIR / "trace_steps_slotgap_fit.json"


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


def _load_steps_fixture():
    """The committed trace + dirs/cuts pair (a real, black-box fit_earth.R
    run with trace = 9, degree = 2, pmethod = "none": one mirrored hinge
    pair on x0, one on x1, then a single unpaired hinge on x0, then a
    hinge pair on x1 nested under that single hinge -- both a pair and a
    single, and a two-level parent chain, in one fit)."""
    log = parse_trace(TRACE_STEPS, sample_var_y=None)
    fit = json.loads(TRACE_STEPS_FIT.read_text())
    return log, np.array(fit["dirs"]), np.array(fit["cuts"]), np.array(fit["x"])


class TestStepsFromTrace:
    """steps_from_trace against the committed trace_steps.txt/
    trace_steps_fit.json pair; no R needed, since these are stored files."""

    def test_a_trailing_rejected_step_with_no_new_term_is_dropped(self):
        # The fixture's trace has 5 FindTerm blocks, but the fit stopped at
        # 4 term-groups (7 terms plus the intercept): the 5th block ends
        # "reject (small DeltaRSq)" and added no row to dirs, so it has no
        # group to align to and must not appear as a placeholder either.
        log, dirs, cuts, _ = _load_steps_fixture()
        assert len(log.steps) == 5
        steps = steps_from_trace(log, dirs, cuts)
        assert len(steps) == 4

    def test_a_mirrored_pair_reports_both_signs(self):
        log, dirs, cuts, x = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        first = steps[0]
        assert first["parent"] == 0 and first["pred"] == 0  # the intercept, x0
        assert first["direction"] == HINGE_PAIR
        assert first["knot"] == cuts[1, 0]  # exact: row 1 (or 2)'s own cuts entry
        assert np.any(x[:, 0] == first["knot"])  # bit-exact, an observed x
        assert first["rss_before"] == pytest.approx(119.0)

    def test_an_unpaired_single_hinge_is_not_reported_as_a_pair(self):
        # Finding 2 (round-2 review): a single hinge earth did not mirror
        # (here, at a step where the paired candidate lost to a boundary
        # case) must not come out as {1, -1}; before this fix every hinge
        # winner did, hiding exactly this case.
        log, dirs, cuts, x = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        single = steps[2]
        assert single["direction"] in (frozenset({1}), frozenset({-1}))
        assert len(single["direction"]) == 1
        assert single["parent"] == 0 and single["pred"] == 0
        assert np.any(x[:, 0] == single["knot"])

    def test_knot_is_the_exact_cuts_value_not_the_trace_rounded_cut(self):
        # Finding 2: the trace's Cut has 5-6 significant digits; the knot
        # here must be cuts's full-precision value (bit-exactly an
        # observed x, per the harness's precision guarantee), not that
        # rounded text -- the trace file never even contains that many
        # digits, since it was printed by R at far lower precision.
        log, dirs, cuts, x = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        trace_text = TRACE_STEPS.read_text()
        for step in steps:
            assert np.any(x[:, step["pred"]] == step["knot"])
            assert repr(step["knot"]) not in trace_text

    def test_a_later_terms_parent_is_the_earlier_single_hinges_row(self):
        # Finding 2: parent/pred are 0-based dirs row/column indices (not
        # the trace's own 1-based internal slot numbers, which count
        # differently: trace_parse.py's docstring). The interaction term
        # here is built on top of the single hinge from the previous
        # (unpaired) step, at dirs row 5.
        log, dirs, cuts, _ = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        interaction = steps[3]
        assert interaction["parent"] == 5
        assert interaction["pred"] == 1
        assert interaction["direction"] == HINGE_PAIR

    def test_best_and_second_best_rss_are_distinct_on_a_real_step(self):
        # N06 (round-2 review): steps_from_trace setting second_best_rss
        # equal to best_rss (as if every candidate tied) would still pass
        # every test above; pin the actual, distinct values from a real
        # step so that specific mutation fails here.
        log, dirs, cuts, _ = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        first = steps[0]
        assert first["best_rss"] == pytest.approx(64.754)
        assert first["second_best_rss"] == pytest.approx(74.885)
        assert first["best_rss"] != first["second_best_rss"]
        assert first["best_rss"] < first["second_best_rss"]

    def test_every_step_is_one_of_the_valid_directions(self):
        log, dirs, cuts, _ = _load_steps_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        assert steps
        for step in steps:
            assert step["direction"] in (
                HINGE_PAIR,
                frozenset({1}),
                frozenset({-1}),
                LINEAR,
            )

    def test_more_dirs_groups_than_trace_steps_raises(self):
        # A dirs/cuts that does not belong to this trace at all (more row
        # groups than the trace has FindTerm blocks) must not silently
        # align them positionally; that would compare the wrong terms.
        dirs = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        cuts = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])
        fake_log = SimpleNamespace(steps=[SimpleNamespace(searches=[], term=2)])
        with pytest.raises(ValueError, match="row-groups"):
            steps_from_trace(fake_log, dirs, cuts)

    def test_a_parent_slot_not_mapping_to_an_earlier_row_raises(self):
        # A trace whose reported parent slot does not resolve to a row
        # earlier than this step's own group is not trustworthy input (the
        # slot<->row mapping trace_parse.py warns about has broken down);
        # raise rather than silently emit a nonsensical comparison.
        dirs = np.array([[0, 0], [1, 0], [-1, 0]])
        cuts = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0]])
        candidate = SimpleNamespace(rss=1.0, cut=0.5, best=True)
        # parent = 5 does not exist (only rows 0-2 do): 5 - 1 = 4 is past
        # this step's own group (row 1), so this must raise.
        search = SimpleNamespace(
            skipped_reason=None,
            pred=1,
            parent=5,
            linear=None,
            hinge=candidate,
            cases=[],
            rss_before=10.0,
        )
        fake_step = SimpleNamespace(searches=[search], term=2)
        fake_log = SimpleNamespace(steps=[fake_step])
        with pytest.raises(ValueError, match="parent slot"):
            steps_from_trace(fake_log, dirs, cuts)


def _load_slotgap_fixture():
    """A real degree-2 fit (numpy default_rng(303), 150 cases, 3 covariates,
    nk = 25, thresh = 0, pmethod = "none") with five consecutive single-
    hinge steps (dirs rows 13-17, each its own row group), then a step
    (the trace's "new term 20") whose winning candidate's parent is slot
    16 -- the step for "new term 16", itself a single hinge. Round-3
    review, PR #38 finding 1's own reproduction: ``parent_slot - 1`` gives
    row 15 (a different term, on a different predictor, that only
    happens to be an earlier row), not row 14, the actual parent."""
    log = parse_trace(TRACE_STEPS_SLOTGAP, sample_var_y=None)
    fit = json.loads(TRACE_STEPS_SLOTGAP_FIT.read_text())
    return log, np.array(fit["dirs"]), np.array(fit["cuts"]), np.array(fit["x"])


class TestStepsFromTraceSlotGap:
    """Round-3 review, PR #38 finding 1: ``parent_slot - 1`` holds only
    until the fit's first single-term step, because every forward step
    reserves two slots regardless of whether it adds one term or two;
    ``test_a_later_terms_parent_is_the_earlier_single_hinges_row`` above
    only exercises a parent that is the fit's first single hinge, where
    slot and row still happen to agree."""

    def test_a_parent_after_several_single_hinge_steps_is_mapped_correctly(
        self,
    ):
        log, dirs, cuts, _ = _load_slotgap_fixture()
        # Confirms the reviewer's own reproduction numbers before trusting
        # the fix: row 14 is h(x1 - ...) (the true parent), row 15 is a
        # same-shape-looking h(x2 - ...) that parent_slot - 1 = 16 - 1
        # would wrongly return instead.
        assert dirs[14].tolist() == [0, 1, 0]
        assert dirs[15].tolist() == [0, 0, 1]
        steps = steps_from_trace(log, dirs, cuts)
        assert steps[9]["parent"] == 14
        assert steps[9]["pred"] == 2

    def test_every_parent_in_this_fit_is_earlier_and_matches_the_masked_row(
        self,
    ):
        # A whole-fit sanity check beyond the one hand-picked step above:
        # every resolved parent must be a real, earlier dirs row whose
        # pattern equals the new row's with the predictor column removed
        # (steps_from_trace's own two post-hoc checks, exercised here
        # against real data rather than a fabricated counterexample).
        log, dirs, cuts, _ = _load_slotgap_fixture()
        steps = steps_from_trace(log, dirs, cuts)
        groups = _dirs_row_groups(dirs, cuts)
        for step, group in zip(steps, groups, strict=True):
            assert step["parent"] < group[0]
            expected = dirs[group[0]].copy()
            expected[step["pred"]] = 0
            assert dirs[step["parent"]].tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("parent_cuts", "groups"),
    [
        ((0.5, 0.5), [[1, 2], [3, 4]]),  # one parent: the pair h(x0-0.5)*h(+-(x1-0.3))
        ((0.5, 0.8), [[1, 2], [3], [4]]),  # two parents: two single hinges
    ],
)
def test_a_mirror_pair_has_one_parent(parent_cuts, groups):
    # Spec review of #48, item 7: rows 3 and 4 differ only in their code for
    # x1, but they are a pair only when their x0 cuts (their parents) agree.
    dirs = np.array([[0, 0], [1, 0], [-1, 0], [1, 1], [1, -1]])
    cuts = np.array([[0, 0], [0.5, 0], [0.5, 0], [parent_cuts[0], 0.3], [0, 0]])
    cuts[4] = [parent_cuts[1], 0.3]
    assert _dirs_row_groups(dirs, cuts) == groups


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


class TestCompareFitPruningNonNestedSubsets:
    """Finding 4 (round-2 review): S01_matched_d1's own prune_terms rows are
    not nested (the size-4 subset {1, 4, 5, 6} is not inside the size-5
    subset {1, 4, 5, 10, 11}, spec PRUNE-3 for one response), so
    removed_sequence returns None and the row was silently skipped. When
    both sides give prune_terms directly, the comparison must work from
    the per-size subsets themselves, nested or not."""

    # The exact S01_matched_d1 shape the review names: sizes 1..5, the last
    # two rows not nested in each other.
    NON_NESTED_PRUNE_TERMS: ClassVar[list[list[int]]] = [
        [1, 0, 0, 0, 0],
        [1, 4, 0, 0, 0],
        [1, 4, 5, 0, 0],
        [1, 4, 5, 6, 0],
        [1, 4, 5, 10, 11],
    ]

    def test_identical_non_nested_matrices_are_no_difference(self):
        # gen_fixtures.py's own reproducibility check: the same earth run
        # twice gives the same (non-nested) prune_terms both times.
        a = {"prune_terms": self.NON_NESTED_PRUNE_TERMS}
        b = {"prune_terms": [row[:] for row in self.NON_NESTED_PRUNE_TERMS]}
        assert compare_fit(a, b) == []

    def test_removed_sequence_cannot_derive_this_but_compare_fit_still_runs(self):
        # Pinning that this exact matrix is the ambiguous case
        # removed_sequence refuses to guess at (TestRemovedSequence covers
        # the mechanism directly); compare_fit must not inherit that
        # refusal now that it no longer routes through removed_sequence
        # when both sides give prune_terms.
        assert removed_sequence({"prune_terms": self.NON_NESTED_PRUNE_TERMS}) is None
        a = {"prune_terms": self.NON_NESTED_PRUNE_TERMS}
        b = {"prune_terms": [row[:] for row in self.NON_NESTED_PRUNE_TERMS]}
        assert compare_fit(a, b) == []

    def test_a_difference_at_a_non_nested_size_is_still_caught(self):
        a = {"prune_terms": self.NON_NESTED_PRUNE_TERMS}
        b_rows = [row[:] for row in self.NON_NESTED_PRUNE_TERMS]
        b_rows[3] = [1, 4, 5, 7, 0]  # size-4 subset differs: 6 vs 7
        b = {"prune_terms": b_rows}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert diffs[0].field == "pruning_removed"
        assert diffs[0].step == 3
        assert diffs[0].a == [1, 4, 5, 6]
        assert diffs[0].b == [1, 4, 5, 7]

    def test_a_row_count_mismatch_is_reported(self):
        a = {"prune_terms": self.NON_NESTED_PRUNE_TERMS}
        b = {"prune_terms": self.NON_NESTED_PRUNE_TERMS[:-1]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert "length mismatch" in diffs[0].detail

    def test_pruning_removed_direct_still_uses_the_nested_derivation(self):
        # pymars's own schema (new_adapter.py) has no prune_terms at all,
        # only pruning_removed directly; that path is unaffected.
        a = {"pruning_removed": [2, 3]}
        b = {"prune_terms": [[1, 0, 0], [1, 3, 0], [1, 2, 3]]}
        assert compare_fit(a, b) == []


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
        # KAPPA_CAP bounds the scale factor at KAPPA_CAP / KAPPA_RSS_LIMIT =
        # 1e8 / 1e6 = 100, so the tolerance here is 1e-8 * 100 = 1e-6, well
        # under 1e-3: this kappa, however huge, cannot make the gap pass.
        diffs = compare_fit({"fwd_rss": a}, {"fwd_rss": b}, kappa=KAPPA_RSS_LIMIT * 1e6)
        assert len(diffs) == 1
        assert diffs[0].label == "numeric"
        expected_tol = FWD_RSS_REL * (KAPPA_CAP / KAPPA_RSS_LIMIT)
        assert diffs[0].tolerance == pytest.approx(expected_tol)

    def test_kappa_scaling_is_capped_so_a_small_gap_still_passes(self):
        # The cap still lets a genuinely tiny gap through at a huge kappa,
        # showing the cap bounds the tolerance rather than zeroing it out.
        b = [10.0, 5.0, 2.0]
        tol_at_cap = FWD_RSS_REL * (KAPPA_CAP / KAPPA_RSS_LIMIT)
        a = [v * (1 + tol_at_cap * 0.5) for v in b]
        diffs = compare_fit({"fwd_rss": a}, {"fwd_rss": b}, kappa=KAPPA_RSS_LIMIT * 1e6)
        assert diffs == []

    def test_infinite_kappa_still_reports_a_gcv_of_1_against_100(self):
        # Round-2 review, PR #38 finding 1: a singular basis (for example
        # the legacy code's left hinge at the smallest knot) gives kappa(B)
        # = inf; without a cap, the scaled tolerance is also infinite, and
        # a gcv of 1 against 100 (or rsq of 0 against 1, or fitted of 0
        # against 9, or coef of 1 against 50) passes. It must not.
        diffs = compare_fit({"gcv": 1.0}, {"gcv": 100.0}, kappa=float("inf"))
        assert len(diffs) == 1
        assert diffs[0].field == "gcv"
        assert diffs[0].label == "numeric"

    def test_nan_kappa_is_treated_the_same_as_infinite(self):
        diffs = compare_fit({"gcv": 1.0}, {"gcv": 100.0}, kappa=float("nan"))
        assert len(diffs) == 1
        assert diffs[0].label == "numeric"


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

    def test_na_on_one_side_against_a_real_number_is_a_difference(self):
        a = {"gcv_per_subset": [1.0, None]}
        b = {"gcv_per_subset": [1.0, 2.0]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert "NA" in diffs[0].detail

    def test_na_on_one_side_against_nan_on_the_other_is_a_difference(self):
        # Finding 4 (round-2 review): np.asarray(..., dtype=float) turns
        # None into nan, the same bit pattern a genuine NaN has, so an R
        # NA (decoded to None by _desanitize) compared against an actual
        # NaN at the same position must not read as "both NaN, matching".
        a = {"gcv_per_subset": [1.0, None]}
        b = {"gcv_per_subset": [1.0, float("nan")]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert "NA" in diffs[0].detail

    def test_na_at_the_same_position_on_both_sides_is_not_a_difference(self):
        a = {"gcv_per_subset": [1.0, None]}
        b = {"gcv_per_subset": [1.0, None]}
        assert compare_fit(a, b) == []

    def test_na_in_coefficients_is_also_kept_distinct_from_nan(self):
        a = {"coef": [[1.0], [None]]}
        b = {"coef": [[1.0], [float("nan")]]}
        diffs = compare_fit(a, b)
        assert len(diffs) == 1
        assert diffs[0].field == "coef"
        assert "NA" in diffs[0].detail


class TestCompareFitSkips:
    """Finding 4 (round-2 review): a comparison that does not run (a field
    absent on one side, or one that needs sd_y when it was not given)
    leaves no record in the returned Difference list (correctly: it is
    not itself a disagreement), but must not vanish silently either -- the
    caller's skipped list, when given, gets a Skip naming the field and
    the reason."""

    def test_a_field_absent_on_one_side_is_recorded(self):
        skipped: list[Skip] = []
        diffs = compare_fit({"gcv_per_subset": [1.0, 2.0]}, {}, skipped=skipped)
        assert diffs == []
        assert len(skipped) == 1
        assert skipped[0].field == "gcv_per_subset"
        assert skipped[0].reason

    def test_a_field_absent_on_both_sides_is_not_recorded(self):
        skipped: list[Skip] = []
        diffs = compare_fit({}, {}, skipped=skipped)
        assert diffs == []
        assert skipped == []

    def test_sd_y_not_given_is_recorded_once_per_field_present_on_both_sides(self):
        skipped: list[Skip] = []
        diffs = compare_fit(
            {"fitted": [1.0, 2.0]}, {"fitted": [1.0, 2.0]}, skipped=skipped
        )
        assert diffs == []
        assert {s.field for s in skipped} == {"fitted"}
        assert "sd_y" in skipped[0].reason

    def test_skipped_defaults_to_none_and_nothing_is_collected(self):
        # The default call site (no skipped= argument) must behave exactly
        # as before: no error, no change to the returned differences.
        assert compare_fit({"fitted": [1.0, 2.0]}, {"fitted": [1.0, 2.0]}) == []

    def test_dataset_is_carried_onto_the_skip(self):
        skipped: list[Skip] = []
        compare_fit({"coef": [[1.0]]}, {}, dataset="S07", skipped=skipped)
        assert skipped[0].dataset == "S07"


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
