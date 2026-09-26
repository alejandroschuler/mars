"""Pure-Python tests for trace_parse.py; no R needed. They parse two stored
logs: validation/legacy/out/trace9.txt (degree 1, one covariate, from the
seed prototype) and tests/data/trace_degree2.txt (degree 2, two covariates,
made once for this suite by a black-box fit_earth.R run with trace = 9)."""

import re
from pathlib import Path

import numpy as np
import pytest
from trace_parse import parse_trace

_SUMMARY_CUT_RE = re.compile(r"x\d+\s+(\S+)")

HARNESS_DIR = Path(__file__).resolve().parents[1]
TRACE9 = HARNESS_DIR.parent / "legacy" / "out" / "trace9.txt"
TRACE_DEGREE2 = Path(__file__).resolve().parent / "data" / "trace_degree2.txt"
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


def test_trace9_flags_and_evaluated():
    log = parse_trace(TRACE9, sample_var_y=None)
    cases = log.steps[0].searches[0].cases
    rejected = [c for c in cases if not c.tol_g]
    accepted = [c for c in cases if c.evaluated]
    assert rejected and accepted
    assert all(not c.evaluated for c in rejected)
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


def test_sample_var_y_none_keeps_the_standardized_scale():
    raw = parse_trace(TRACE9, sample_var_y=None)
    scaled = parse_trace(TRACE9, sample_var_y=2.0)
    raw_rss = raw.steps[0].searches[0].cases[0].rss
    scaled_rss = scaled.steps[0].searches[0].cases[0].rss
    assert scaled_rss == pytest.approx(raw_rss * 2.0)
