"""Pure-Python tests for trace_parse.py; no R needed. They parse two stored
logs: validation/legacy/out/trace9.txt (degree 1, one covariate, from the
seed prototype) and tests/data/trace_degree2.txt (degree 2, two covariates,
made once for this suite by a black-box fit_earth.R run with trace = 9)."""

import re
from pathlib import Path

import numpy as np
import pytest
from trace_parse import TraceParseError, parse_trace

_SUMMARY_CUT_RE = re.compile(r"x\d+\s+(\S+)")

HARNESS_DIR = Path(__file__).resolve().parents[1]
TRACE9 = HARNESS_DIR.parent / "legacy" / "out" / "trace9.txt"
_DATA_DIR = Path(__file__).resolve().parent / "data"
TRACE_DEGREE2 = _DATA_DIR / "trace_degree2.txt"
TRACE_LEVEL7 = _DATA_DIR / "trace_level7.txt"
TRACE_LEVEL8 = _DATA_DIR / "trace_level8.txt"
HINGE1D_TRAIN = HARNESS_DIR.parent / "legacy" / "data" / "hinge1d_1_train.csv"


def _var_y(csv_path: Path) -> float:
    y = np.genfromtxt(csv_path, delimiter=",", names=True)["y"]
    return float(np.var(y, ddof=1))


def test_trace9_step_count_and_case_counts():
    log = parse_trace(TRACE9, sample_var_y=None)
    # degree 1, hinge-only: forward steps add a pair of terms, so the step
    # numbers are 2, 4, .., 20 (validation/legacy/out/trace9.txt's own
    # "Reached nk 21" / "12 terms used" trailer).
    assert [s.term for s in log.steps] == list(range(2, 21, 2))
    # p = 1, so every step has exactly one live (parent, predictor) search,
    # each scanning the same 198 candidate cases (minspan = endspan = 1).
    for step in log.steps:
        live = [sr for sr in step.searches if sr.skipped_reason is None]
        assert len(live) == 1
        assert live[0].parent == 1 and live[0].pred == 1
        assert len(live[0].cases) == 198


def test_trace9_rss_before_first_hinge_is_n_minus_1_on_the_standardized_scale():
    # S01 has 200 cases; a y standardized to sample variance 1 has total sum
    # of squares n - 1 = 199, which is exactly what earth reports as the
    # RSS before any hinge is added to the intercept-only model.
    log = parse_trace(TRACE9, sample_var_y=None)
    first_search = log.steps[0].searches[0]
    assert first_search.rss_before == pytest.approx(199.0)
    assert first_search.min_span == 1
    assert first_search.end_span == 1


def test_trace9_rss_scaling_matches_the_real_forward_rss_path():
    # Cross-check against a real fit: driver.run_earth's fwd_rss (computed
    # independently, by refitting lm.fit on the cumulative bx columns) after
    # both hinges of the first pair is the same number, to display
    # precision, as this trace's winning first-step candidate's Rss times
    # the real sample variance of y.
    var_y = _var_y(HINGE1D_TRAIN)
    log = parse_trace(TRACE9, sample_var_y=var_y)
    winner = log.steps[0].searches[0].hinge
    assert winner is not None and winner.best
    assert winner.rss == pytest.approx(2.8826, abs=5e-5)


def test_trace9_flags_and_accepted():
    # trace9.txt is at minspan = 1, so every printed case was evaluated
    # (there are no iSpan rows); accepted is the AND of the four flags, and
    # is False exactly on the TolG = 0 rows, which were still evaluated.
    log = parse_trace(TRACE9, sample_var_y=None)
    cases = log.steps[0].searches[0].cases
    assert all(c.evaluated for c in cases)
    rejected = [c for c in cases if not c.tol_g]
    accepted = [c for c in cases if c.accepted]
    assert rejected and accepted
    assert all(c.accepted is False for c in rejected)
    assert all(c.bx1g and c.cov_col_g and c.tol_g and c.max_g for c in accepted)
    # "best" tracks a running best-so-far within the scan, so it must appear
    # at least once, and never on a rejected (TolG = 0) row.
    assert any(c.best for c in cases)
    assert not any(c.best for c in rejected)


def test_trace9_summary_line_is_captured_verbatim():
    log = parse_trace(TRACE9, sample_var_y=None)
    last = log.steps[-1]
    assert last.term == 20
    assert last.summary_line is not None
    assert last.summary_line.strip().startswith("20")
    assert "x0" in last.summary_line


def test_degree2_trace_has_skips_and_multi_search_steps():
    log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
    assert len(log.steps) >= 3
    # p = 2 and degree = 2: once more than one term exists, several (parent,
    # predictor) pairs compete in one step, unlike the degree-1, p=1 file.
    multi = [s for s in log.steps if len(s.searches) > 1]
    assert multi
    reasons = {
        sr.skipped_reason
        for step in log.steps
        for sr in step.searches
        if sr.skipped_reason is not None
    }
    assert reasons  # at least one skip, of some reason
    # A skipped search has no scan and no candidate.
    for step in log.steps:
        for sr in step.searches:
            if sr.skipped_reason is not None:
                assert sr.cases == []
                assert sr.linear is None and sr.hinge is None


def test_degree2_trace_parent_and_pred_can_both_vary():
    log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
    parents = {sr.parent for step in log.steps for sr in step.searches}
    preds = {
        sr.pred for step in log.steps for sr in step.searches if sr.pred is not None
    }
    assert len(parents) > 1
    assert preds == {1, 2}


def test_degree2_trace_winning_candidate_matches_its_steps_summary_line():
    # Every step's summary_line names the Cut earth actually chose; the
    # search whose hinge (or linear) candidate is tagged best=True in file
    # order last should be the one that summary line reports (both are
    # earth's own rounded display of the same double, at different
    # precisions, so this compares them as numbers, not as text).
    log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
    for step in log.steps:
        winners = [
            sr
            for sr in step.searches
            if (sr.hinge and sr.hinge.best) or (sr.linear and sr.linear.best)
        ]
        assert winners, f"term {step.term} has no tagged winner"
        last_winner = winners[-1]
        if last_winner.hinge and last_winner.hinge.best:
            cut = last_winner.hinge.cut
        else:
            cut = last_winner.linear.cut
        assert step.summary_line is not None
        m = _SUMMARY_CUT_RE.search(step.summary_line)
        assert m, step.summary_line
        assert cut == pytest.approx(float(m[1]), abs=1e-3)


def test_degree2_trace_keeps_span_skipped_cases_as_unevaluated():
    # Finding 4 of the round-1 review: cases earth visits but does not
    # evaluate (--FindKnot--Case N iSpan K bx1 V, the minspan/endspan rule)
    # were silently dropped; trace_degree2.txt (minspan = endspan = 1) still
    # has some, from interaction terms whose parent needs nStartSpan cases
    # skipped before a candidate is legal.
    log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
    all_cases = [c for step in log.steps for sr in step.searches for c in sr.cases]
    skipped = [c for c in all_cases if not c.evaluated]
    evaluated = [c for c in all_cases if c.evaluated]
    assert skipped, "expected at least one span-skipped case in this fixture"
    assert evaluated
    for c in skipped:
        assert c.i_span is not None and c.bx1 is not None
        assert c.cut is None and c.rss is None and c.rss_delta is None
        assert c.bx1g is None and c.accepted is None
    for c in evaluated:
        assert c.cut is not None and c.rss is not None


def test_degree2_trace_accepted_is_none_without_evaluation():
    log = parse_trace(TRACE_DEGREE2, sample_var_y=None)
    skipped = [
        c
        for step in log.steps
        for sr in step.searches
        for c in sr.cases
        if not c.evaluated
    ]
    assert skipped and all(c.accepted is None for c in skipped)


def test_findknot_block_raises_on_an_unrecognized_line(tmp_path):
    bad = tmp_path / "bad_trace.txt"
    bad.write_text(
        "|FindTerm: Searching for new term 2    RssDelta 0 MaxLegalRssDelta 1\n"
        "|Parent 1  Pred 1  Case   -1 Cut     0.5< Rss 10       RssDeltaLin 1\n"
        "--FindKnotBegin-- iPred 1 iNewCol 3 RssBeforeAddingHinge 10 nMinSpan 1"
        " nEndSpan 1 nStartSpan 1\n"
        "--FindKnot--Case  5 something entirely unexpected\n"
        "--FindKnotEnd--\n"
    )
    with pytest.raises(TraceParseError, match="matches neither"):
        parse_trace(bad)


class TestTraceLevel7:
    """trace = 7: one summary line per combo, no linear/hinge phase split,
    no FindKnotBegin/End block at all."""

    def test_parses_without_error_and_keeps_no_per_case_detail(self):
        log = parse_trace(TRACE_LEVEL7, sample_var_y=None)
        assert len(log.steps) >= 3
        all_cases = [c for step in log.steps for sr in step.searches for c in sr.cases]
        assert all_cases == []
        live = [
            sr
            for step in log.steps
            for sr in step.searches
            if sr.skipped_reason is None
        ]
        assert live
        for sr in live:
            assert sr.linear is None
            assert sr.hinge is not None
            assert sr.rss_before is None

    def test_winning_cut_matches_the_summary_line(self):
        log = parse_trace(TRACE_LEVEL7, sample_var_y=None)
        step = log.steps[0]
        assert step.term == 2
        winner = step.searches[0]
        assert winner.hinge.best
        m = _SUMMARY_CUT_RE.search(step.summary_line)
        assert m and winner.hinge.cut == pytest.approx(float(m[1]), abs=1e-3)


class TestTraceLevel8:
    """trace = 8: the linear pre-check has no RssDeltaLin field, and
    FindKnot rows have no bx1G/CovColG/TolG/MaxG flags; span-skipped cases
    are not printed at all, so every printed case is evaluated."""

    def test_parses_without_error(self):
        log = parse_trace(TRACE_LEVEL8, sample_var_y=None)
        assert len(log.steps) >= 3

    def test_linear_precheck_has_no_rss_delta(self):
        log = parse_trace(TRACE_LEVEL8, sample_var_y=None)
        first_search = log.steps[0].searches[0]
        assert first_search.linear is not None
        assert first_search.linear.rss_delta is None

    def test_cases_have_no_flags_but_are_evaluated(self):
        log = parse_trace(TRACE_LEVEL8, sample_var_y=None)
        cases = log.steps[0].searches[0].cases
        assert cases
        assert all(c.evaluated for c in cases)
        assert all(c.bx1g is None for c in cases)
        assert all(c.accepted is None for c in cases)

    def test_winning_cut_matches_the_summary_line(self):
        log = parse_trace(TRACE_LEVEL8, sample_var_y=None)
        step = log.steps[0]
        winner = step.searches[0]
        assert winner.hinge.best
        m = _SUMMARY_CUT_RE.search(step.summary_line)
        assert m and winner.hinge.cut == pytest.approx(float(m[1]), abs=1e-3)


def test_sample_var_y_none_keeps_the_standardized_scale():
    raw = parse_trace(TRACE9, sample_var_y=None)
    scaled = parse_trace(TRACE9, sample_var_y=2.0)
    raw_rss = raw.steps[0].searches[0].cases[0].rss
    scaled_rss = scaled.steps[0].searches[0].cases[0].rss
    assert scaled_rss == pytest.approx(raw_rss * 2.0)
