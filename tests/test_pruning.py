"""Tests of pymars/_pruning.py (docs/algorithm.md: PRUNE-1 to PRUNE-9, with
GCV-2, GCV-5 to GCV-7, LIMIT-3, LA-4, W-1 to W-5, RESP-1, CORE-3 and EDGE-6).

The earth fixtures are ``validation/fixtures/components/pruning_fixed_basis``
(``earth:::pruning.pass`` on an earth forward basis, for one response and for
three) and the S datasets (``prune.terms``, ``rss.per.subset``,
``gcv.per.subset``, the selected terms, the coefficients and the statistics of
earth 5.3.4). The tests build earth's forward basis again from its ``dirs`` and
``cuts`` (TERM-3) and prune it. Tolerances, from the plan's tolerance table:
the exact terms at every size, relative 1e-8 for RSS and GCV, absolute 1e-8
for RSq and GRSq, normwise relative 1e-6 for coefficients where κ(B) ≤ 1e5,
and 1e-8·sd(y) for fitted values.

The explicit answers come from a refit of every subset by numpy's SVD solver,
which follows the steps of PRUNE-3 as the spec writes them. The inputs from
``_case`` and from the fixtures are read-only, so a function that writes into
them fails.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pymars import _gcv, _linalg, _terms
from pymars import _pruning as pr

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "fixtures"
S_FITS = sorted(
    p.stem
    for p in FIXTURES.glob("S*.json")
    if p.name.startswith("S")
    and "error" not in json.loads(p.read_text(encoding="utf-8"))["result"]
)

seeds = st.integers(0, 2**32 - 1)
# (seed, n, m, k, weighted): n rows, m terms and k responses.
cases = st.tuples(
    seeds, st.integers(10, 40), st.integers(1, 8), st.integers(1, 3), st.booleans()
)


def _frozen(*arrays):
    out = []
    for a in arrays:
        a = None if a is None else np.array(a, dtype=np.float64)
        if a is not None:
            a.flags.writeable = False
        out.append(a)
    return out


def _case(seed, n, m, k, weighted):
    """An intercept, then hinge and linear columns, as in a MARS basis."""
    rng = np.random.default_rng(seed)
    cols = [np.ones(n)]
    for j in range(1, m):
        x = rng.standard_normal(n)
        t = np.quantile(x, rng.uniform(0.2, 0.6))
        cols.append(x if j % 3 == 0 else np.maximum((x - t) * (-1) ** j, 0.0))
    B = np.column_stack(cols)
    Y = B @ rng.standard_normal((m, k)) + rng.standard_normal((n, k))
    w = rng.uniform(0.2, 3.0, n) if weighted else None
    return _frozen(B, Y[:, 0] if k == 1 else Y, w)


def _rss(B, Y, w, cols):
    """The weighted RSS of Y on the columns ``cols`` of B, summed (LA-1)."""
    sw = np.ones((B.shape[0], 1)) if w is None else np.sqrt(w)[:, None]
    A, V = sw * B[:, sorted(cols)], sw * Y.reshape(B.shape[0], -1)
    return float(np.sum((V - A @ np.linalg.lstsq(A, V, rcond=None)[0]) ** 2))


def _spec_pass(B, Y, w):
    """PRUNE-2 to PRUNE-4 as the spec writes them, with a refit of every subset.

    Returns ``removed``, R[m], T[m] and the smallest relative gap between two
    choices that the rules must tell apart (a drop or an offer), so that a test
    can leave out near-ties (OQ-2).
    """
    M = B.shape[1]
    several = Y.ndim == 2 and Y.shape[1] >= 2
    order = list(range(M))
    best, sets = [np.inf] * M, [None] * M
    gap = np.inf

    def offer():
        nonlocal gap
        for m in range(1, M + 1):
            U = frozenset(order[:m])
            r = _rss(B, Y, w, U)
            if sets[m - 1] is not None and sets[m - 1] != U:
                gap = min(gap, abs(r - best[m - 1]) / best[m - 1])
            if r < best[m - 1]:
                best[m - 1], sets[m - 1] = r, U

    if several:
        best[M - 1], sets[M - 1] = _rss(B, Y, w, order), frozenset(order)
    else:
        offer()
    removed = []
    for pos in range(M, 1, -1):
        drops = [
            (_rss(B, Y, w, set(order[:pos]) - {order[i]}), order[i], i)
            for i in range(1, pos)
        ]
        rss = sorted(d[0] for d in drops)
        if len(rss) > 1:
            gap = min(gap, (rss[1] - rss[0]) / rss[0])
        r, term, i = min(drops, key=lambda d: (d[0], -d[1]))
        removed.append(term)
        order.insert(pos - 1, order.pop(i))
        if several:
            best[pos - 2], sets[pos - 2] = r, frozenset(order[: pos - 1])
        else:
            offer()
    return removed, best, sets, gap


def _sets(subsets):
    return [frozenset(np.flatnonzero(row).tolist()) for row in subsets]


def assert_rel(actual, expected, rtol):
    """Equal infinities, and finite values within the relative tolerance."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=0)


def _earth_case(fixture):
    """Earth's forward basis, Y, w and penalty of one S fixture.

    earth stores the smallest x at a linear factor, pymars 0.0 (TERM-2); a
    factor response is one indicator response per level (RESP-1); the default
    penalty is that of GCV-4.
    """
    r = fixture["result"]
    dirs, cuts = np.array(r["dirs"]), np.array(r["cuts"], dtype=float)
    cuts[(dirs == 0) | (dirs == 2)] = 0.0
    B = _terms.basis_matrix(np.array(fixture["inputs"]["X"]), dirs, cuts)
    y = fixture["inputs"]["y"]
    if isinstance(y[0], str):
        Y = np.column_stack([[v == level for v in y] for level in r["levels"]])
    else:
        Y = np.array(y)
    args = fixture["earth_args"]
    penalty = args.get("penalty", _gcv.default_penalty(args["degree"]))
    return [*_frozen(B, Y), fixture["inputs"]["weights"], penalty]


@pytest.mark.parametrize("case", ["one_response", "several_responses"])
def test_component_fixture(load_fixture, case):
    """PRUNE-3 to PRUNE-5 against earth:::pruning.pass on a degree-2 basis of
    12 terms (one response) and 11 terms (three responses). Seed 138 makes the
    one-response rule and the several-responses rule give different subsets at
    sizes 5 to 7 in both cases, so every size is compared."""
    fixture = load_fixture("components/pruning_fixed_basis")
    c = fixture[case]
    B = _terms.basis_matrix(np.array(fixture["X"]), c["dirs"], c["cuts"])
    res = pr.pruning_pass(*_frozen(B, c["y"]), penalty=fixture["penalty"])
    earth_sets = [frozenset(t - 1 for t in row if t) for row in c["prune_terms"]]
    assert _sets(res.subsets) == earth_sets
    assert_rel(res.rss_per_size, c["rss_per_subset"], 1e-8)
    assert_rel(res.gcv_per_size, c["gcv_per_subset"], 1e-8)
    np.testing.assert_array_equal(res.selected + 1, c["selected_terms"])


def test_the_rule_follows_the_number_of_response_columns(load_fixture):
    """PRUNE-3: K counts the columns of Y. A constant second column adds 0 to
    every RSS, so the removals stay the same, but the nested rule of K ≥ 2 then
    misses earth's one-response subsets at the sizes the fixture names."""
    fixture = load_fixture("components/pruning_fixed_basis")
    c = fixture["one_response"]
    B = _terms.basis_matrix(np.array(fixture["X"]), c["dirs"], c["cuts"])
    y = np.array(c["y"])
    M = B.shape[1]
    one = pr.pruning_pass(B, y[:, None], penalty=2.0)
    two = pr.pruning_pass(B, np.column_stack([y, np.full_like(y, 3.0)]), penalty=2.0)
    np.testing.assert_array_equal(one.removed, two.removed)
    differ = np.flatnonzero((one.subsets != two.subsets).any(axis=1)) + 1
    assert differ.tolist() == c["k1_vs_k2_diverge_at_sizes"]
    everything = frozenset(range(M))
    nested = [everything - set(two.removed[: M - m]) for m in range(1, M + 1)]
    assert _sets(two.subsets) == nested
    assert np.all(one.rss_per_size <= two.rss_per_size * (1 + 1e-12))


@pytest.mark.parametrize("name", S_FITS)
def test_s_fixture_pruning(load_fixture, name):
    """PRUNE-3 to PRUNE-5 on earth's forward basis of each S fit: the subset of
    every size, and its RSS; with no weights also the GCV and the selected
    terms. Departures: earth reports sums of squares below about 1e-10 as 0
    (PRUNE-4), ignores weights that are all equal (GCV-8) and uses the number
    of cases in its GCV (W-2), so those values are not compared. earth keeps a
    zero weight as a tiny one (W-3), which moves its RSS by at most 6e-9."""
    fixture = load_fixture(name)
    B, Y, w, penalty = _earth_case(fixture)
    r = fixture["result"]
    res = pr.pruning_pass(B, Y, w, penalty=penalty)
    earth_sets = [frozenset(t - 1 for t in row if t) for row in r["prune_terms"]]
    assert _sets(res.subsets) == earth_sets
    rss = np.array(r["rss_per_subset"])
    scale = w[0] if w is not None and len(set(w)) == 1 else 1.0
    shown = rss > 0.0
    assert_rel(res.rss_per_size[shown] / scale, rss[shown], 1e-8)
    if w is None:
        assert_rel(res.gcv_per_size[shown], np.array(r["gcv_per_subset"])[shown], 1e-8)
        np.testing.assert_array_equal(res.selected + 1, r["selected_terms"])


@pytest.mark.parametrize("name", S_FITS)
def test_s_fixture_final_fit(load_fixture, name):
    """PRUNE-8 on earth's selected terms: coefficients (κ ≤ 1e5), fitted values
    and RSS; with no weights also the GCV, RSq and GRSq (W-2, GCV-8 as above).
    A fit whose RSS earth reports as 0 is left out (PRUNE-4, GCV-7)."""
    fixture = load_fixture(name)
    B, Y, w, penalty = _earth_case(fixture)
    r = fixture["result"]
    if r["rss"] == 0.0:
        return
    sel = np.array(r["selected_terms"]) - 1
    fit = pr.final_fit(B, Y, sel, w, penalty=penalty)
    coef = np.array(r["coef"]).reshape(fit.coef.shape)
    if np.linalg.cond(B[:, sel]) <= 1e5:
        assert np.linalg.norm(fit.coef - coef) <= 1e-6 * np.linalg.norm(coef)
    fitted = np.array(r["fitted"]).reshape(B.shape[0], -1)
    assert np.max(np.abs(B[:, sel] @ fit.coef - fitted)) <= 1e-8 * np.std(Y)
    scale = w[0] if w is not None and len(set(w)) == 1 else 1.0
    assert_rel(fit.rss / scale, r["rss"], 1e-8)
    if w is None:
        assert_rel(fit.gcv, r["gcv"], 1e-8)
        assert abs(fit.rsq - r["rsq"]) <= 1e-8
        assert abs(fit.grsq - r["grsq"]) <= 1e-8


def test_s_fixture_list():
    """The collection found the fixtures."""
    assert len(S_FITS) >= 130


@given(cases)
def test_matches_refits_of_every_subset(case):
    """PRUNE-2 to PRUNE-5 against the steps of the spec with a refit of every
    subset, for one and several responses, with and without weights: the same
    removals, the same subsets (not nested in general for K = 1, nested for
    K ≥ 2), the same RSS and GCV, and the intercept in every subset."""
    B, Y, w = _case(*case)
    removed, best, sets, gap = _spec_pass(B, Y, w)
    assume(gap > 1e-9)
    res = pr.pruning_pass(B, Y, w, penalty=3.0)
    M = B.shape[1]
    np.testing.assert_array_equal(res.removed, removed)
    assert _sets(res.subsets) == sets
    assert_rel(res.rss_per_size, best, 1e-8)
    N = B.shape[0] if w is None else np.sum(w)
    sizes = np.arange(1, M + 1)
    assert_rel(res.gcv_per_size, _gcv.gcv(np.array(best), sizes, 3.0, N), 1e-8)
    assert res.removed.dtype == np.int64 and res.removed.shape == (M - 1,)
    assert sorted(res.removed.tolist()) == list(range(1, M))
    assert res.subsets.dtype == bool and res.subsets.shape == (M, M)
    assert res.subsets[:, 0].all() and (res.subsets.sum(axis=1) == sizes).all()
    m_star = int(np.argmin(res.gcv_per_size)) + 1
    assert res.selected_size == m_star
    np.testing.assert_array_equal(res.selected, sorted(sets[m_star - 1]))
    assert res.selected.dtype == np.int64


def test_one_term():
    """CORE-3: with M_f = 1, ``removed`` is empty, ``rss_per_size`` is [TSS]
    and ``subsets`` is [[True]]."""
    y = np.array([1.0, 2.0, 4.0, 7.0])
    res = pr.pruning_pass(np.ones((4, 1)), y, penalty=2.0)
    assert res.removed.shape == (0,) and res.subsets.tolist() == [[True]]
    assert_rel(res.rss_per_size, [_gcv.tss(y)], 1e-14)
    assert_rel(res.gcv_per_size, [_gcv.gcv(_gcv.tss(y), 1, 2.0, 4.0)], 1e-14)
    assert res.selected_size == 1 and res.selected.tolist() == [0]


@pytest.mark.parametrize("k", [1, 2])
def test_exact_ties_and_constant_responses(k):
    """PRUNE-3 ties: when every response is constant every RSS is 0 exactly
    (Conventions), so each stage removes the term with the largest index and
    the first offer of each size stays. GCV-7: the fit is degenerate, so every
    GCV is +∞, and PRUNE-5 selects the smallest size."""
    B, _, _ = _case(5, 20, 6, 1, False)
    Y = np.full((20, k), 2.5)
    res = pr.pruning_pass(B, Y[:, 0] if k == 1 else Y, penalty=2.0)
    assert res.removed.tolist() == [5, 4, 3, 2, 1]
    np.testing.assert_array_equal(res.subsets, np.tri(6, dtype=bool))
    assert res.rss_per_size.tolist() == [0.0] * 6
    assert np.isposinf(res.gcv_per_size).all()
    assert res.selected_size == 1 and res.selected.tolist() == [0]
    fit = pr.final_fit(B, Y, [0], penalty=2.0)
    assert fit.coef.tolist() == [[2.5] * k] and fit.rss == 0.0
    assert np.isposinf(fit.gcv) and fit.rsq == 0.0 and fit.grsq == 0.0


def test_a_constant_response_adds_nothing():
    """A constant response adds 0 exactly to every RSS, and its coefficients
    are its value and zeros (Conventions, PRUNE-8)."""
    B, y, _ = _case(8, 25, 5, 1, False)
    Y = np.column_stack([y, np.full(25, -1.5), 2 * y])
    two = pr.pruning_pass(B, Y[:, [0, 2]], penalty=2.0)
    three = pr.pruning_pass(B, Y, penalty=2.0)
    np.testing.assert_array_equal(three.removed, two.removed)
    np.testing.assert_array_equal(three.subsets, two.subsets)
    np.testing.assert_array_equal(three.rss_per_size, two.rss_per_size)
    fit = pr.final_fit(B, Y, three.selected, penalty=2.0)
    col = fit.coef[:, 1]
    assert col[0] == -1.5 and not col[1:].any()


def test_an_equal_offer_keeps_the_earlier_subset(monkeypatch):
    """PRUNE-3 for one response: an offered prefix replaces T[m] only when its
    RSS is lower, so when every offer gives the same RSS, the prefixes of the
    starting order stay, although the stages reorder the terms."""
    B, y, _ = _case(6, 30, 6, 1, False)
    moved = pr.pruning_pass(B, y, penalty=2.0).removed
    assert moved.tolist() != [5, 4, 3, 2, 1]
    monkeypatch.setattr(_linalg, "prefix_rss", lambda Z, rss: np.arange(6.0, 0.0, -1))
    res = pr.pruning_pass(B, y, penalty=2.0)
    np.testing.assert_array_equal(res.removed, moved)
    np.testing.assert_array_equal(res.subsets, np.tri(6, dtype=bool))


def test_selected_size_is_the_smallest_with_the_lowest_gcv(monkeypatch):
    """PRUNE-5, PRUNE-6: ties go to the smaller size, and ``nprune`` bounds the
    range; PRUNE-7: ``pmethod="none"`` keeps the first min(nprune, M_f) terms."""
    B, y, _ = _case(3, 30, 5, 1, False)
    monkeypatch.setattr(_gcv, "gcv", lambda *a: np.array([3.0, 1.0, 2.0, 1.0, 5.0]))
    res = pr.pruning_pass(B, y, penalty=2.0)
    assert res.selected_size == 2
    np.testing.assert_array_equal(res.selected, np.flatnonzero(res.subsets[1]))
    assert pr.pruning_pass(B, y, penalty=2.0, nprune=1).selected_size == 1
    none = pr.pruning_pass(B, y, penalty=2.0, pmethod="none", nprune=4)
    assert none.selected_size == 4 and none.selected.tolist() == [0, 1, 2, 3]
    none = pr.pruning_pass(B, y, penalty=2.0, pmethod="none")
    assert none.selected_size == 5 and none.selected.tolist() == [0, 1, 2, 3, 4]


@given(cases, st.integers(1, 9), st.sampled_from(["backward", "none"]))
def test_nprune_and_pmethod_change_only_the_selection(case, nprune, pmethod):
    """PRUNE-6, PRUNE-7: the stages and the records are those of
    ``pmethod="backward"`` without ``nprune``."""
    B, Y, w = _case(*case)
    base = pr.pruning_pass(B, Y, w, penalty=2.0)
    res = pr.pruning_pass(B, Y, w, penalty=2.0, pmethod=pmethod, nprune=nprune)
    for a, b in zip(res[:4], base[:4], strict=True):
        np.testing.assert_array_equal(a, b)
    assert res.selected_size <= min(nprune, B.shape[1])
    if pmethod == "none":
        assert res.selected_size == min(nprune, B.shape[1])


def test_nprune_with_penalty_minus_one_selects_nprune_terms():
    """PRUNE-6, bb07.19: with d = -1 the GCV is RSS/N, which falls as m grows,
    so ``nprune`` = k selects exactly k terms."""
    B, y, _ = _case(11, 40, 8, 1, False)
    res = pr.pruning_pass(B, y, penalty=-1.0)
    assert np.all(np.diff(res.rss_per_size) < 0)
    assert_rel(res.gcv_per_size, res.rss_per_size / 40.0, 1e-15)
    for k in (1, 3, 8, 20):
        assert pr.pruning_pass(B, y, penalty=-1.0, nprune=k).selected_size == min(k, 8)


@given(cases)
def test_final_fit_is_least_squares_on_the_selected_terms(case):
    """PRUNE-8: the coefficients of LA-4 on the selected columns, the RSS of
    those coefficients, and GCV-2, GCV-5, GCV-6 with M = m*; for
    ``pmethod="backward"`` the RSS is ``rss_per_size[m* - 1]``."""
    B, Y, w = _case(*case)
    res = pr.pruning_pass(B, Y, w, penalty=2.0)
    fit = pr.final_fit(B, Y, res.selected, w, penalty=2.0)
    lm = _linalg.lm_fit(B[:, res.selected], Y, w)
    want = lm.coef.reshape(fit.coef.shape)
    assert np.linalg.norm(fit.coef - want) <= 1e-12 * np.linalg.norm(want)
    assert fit.coef.shape == (res.selected_size, 1 if Y.ndim == 1 else Y.shape[1])
    assert_rel(fit.rss, res.rss_per_size[res.selected_size - 1], 1e-8)
    N, tau = _gcv.total_weight(B.shape[0], w)
    m, tss = res.selected_size, _gcv.tss(Y, w)
    assert fit.gcv == _gcv.gcv(fit.rss, m, 2.0, N, tau)
    assert fit.rsq == _gcv.rsq(fit.rss, tss, m)
    assert fit.grsq == _gcv.grsq(fit.rss, tss, m, 2.0, N, tau)


def test_final_fit_with_pmethod_none_describes_the_first_terms():
    """PRUNE-7, PRUNE-8 (a departure): with m* < M_f the statistics are those of
    terms 0 to m* - 1, not of T[m*]."""
    B, y, _ = _case(4, 30, 7, 1, False)
    res = pr.pruning_pass(B, y, penalty=2.0, pmethod="none", nprune=3)
    fit = pr.final_fit(B, y, res.selected, penalty=2.0)
    assert_rel(fit.rss, _rss(B, y, None, [0, 1, 2]), 1e-10)
    assert fit.rss >= res.rss_per_size[2]


def test_degenerate_weight_sum():
    """GCV-7: with N ≤ 1 the fit is degenerate, so the GCV is +∞, while the RSS
    of a response that is not constant is computed."""
    B = np.ones((2, 1))
    y, w = np.array([0.0, 1.0]), np.array([0.25, 0.5])
    res = pr.pruning_pass(B, y, w, penalty=2.0)
    assert_rel(res.rss_per_size, [_gcv.tss(y, w)], 1e-14)
    assert np.isposinf(res.gcv_per_size).all()
    fit = pr.final_fit(B, y, [0], w, penalty=2.0)
    assert_rel(fit.coef, [[2 / 3]], 1e-14)
    assert np.isposinf(fit.gcv)


def test_rows_with_zero_weight_are_not_cases():
    """W-3: a row with zero weight is dropped first, so a response that is
    constant on the other rows makes the fit degenerate (GCV-7): every RSS is
    0 exactly and every GCV +∞."""
    B, _, _ = _case(2, 12, 3, 1, False)
    y = np.full(12, 4.0)
    y[5] = -1.0
    w = np.ones(12)
    w[5] = 0.0
    res = pr.pruning_pass(B, y, w, penalty=2.0)
    assert res.rss_per_size.tolist() == [0.0] * 3
    assert np.isposinf(res.gcv_per_size).all()
    fit = pr.final_fit(B, y, [0], w, penalty=2.0)
    assert fit.coef.tolist() == [[4.0]] and fit.rss == 0.0


@given(seeds, st.integers(8, 20), st.integers(1, 6), st.booleans())
def test_integer_weights_equal_repeated_rows(seed, n, m, several):
    """W-1, W-2: an integer weight counts as that many copies of the row, a
    zero weight drops the row (W-3), and unit weights equal no weights (W-5)."""
    B, Y, _ = _case(seed, n, m, 2 if several else 1, False)
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 4, n).astype(float)
    w[:m] = 1.0  # keep enough rows
    assume(np.linalg.matrix_rank(B[w > 0]) == m)
    rep = np.repeat(np.arange(n), w.astype(int))
    got = pr.pruning_pass(B, Y, w, penalty=2.0)
    want = pr.pruning_pass(B[rep], Y[rep], penalty=2.0)
    np.testing.assert_array_equal(got.removed, want.removed)
    np.testing.assert_array_equal(got.subsets, want.subsets)
    assert_rel(got.rss_per_size, want.rss_per_size, 1e-10)
    assert_rel(got.gcv_per_size, want.gcv_per_size, 1e-10)
    assert got.selected_size == want.selected_size
    fit, fit_rep = (
        pr.final_fit(B, Y, got.selected, w, penalty=2.0),
        pr.final_fit(B[rep], Y[rep], got.selected, penalty=2.0),
    )
    assert_rel(fit.rss, fit_rep.rss, 1e-10)
    unit = pr.pruning_pass(B, Y, np.ones(n), penalty=2.0)
    none = pr.pruning_pass(B, Y, penalty=2.0)
    for a, b in zip(unit, none, strict=True):
        np.testing.assert_array_equal(a, b)


@given(cases, st.integers(-60, 60))
def test_a_power_of_two_in_y_scales_the_sums_exactly(case, j):
    """EDGE-6: Y times 2^j gives the same stages and every RSS and GCV times
    4^j, bit for bit."""
    B, Y, w = _case(*case)
    base = pr.pruning_pass(B, Y, w, penalty=2.0)
    res = pr.pruning_pass(B, np.ldexp(Y, j), w, penalty=2.0)
    np.testing.assert_array_equal(res.removed, base.removed)
    np.testing.assert_array_equal(res.subsets, base.subsets)
    np.testing.assert_array_equal(res.rss_per_size, np.ldexp(base.rss_per_size, 2 * j))
    np.testing.assert_array_equal(res.gcv_per_size, np.ldexp(base.gcv_per_size, 2 * j))


@given(cases)
def test_row_order_and_column_scale_do_not_change_the_pass(case):
    """Conventions: the rows' order does not matter, and a positive factor on a
    column changes no RSS (PRUNE-3 compares spans only)."""
    B, Y, w = _case(*case)
    *_, gap = _spec_pass(B, Y, w)
    assume(gap > 1e-9)
    base = pr.pruning_pass(B, Y, w, penalty=2.0)
    perm = np.random.default_rng(case[0]).permutation(B.shape[0])
    scale = np.ones(B.shape[1])
    scale[1:] = np.random.default_rng(case[0]).uniform(1e-3, 1e3, B.shape[1] - 1)
    res = pr.pruning_pass(
        (B * scale)[perm], Y[perm], None if w is None else w[perm], penalty=2.0
    )
    np.testing.assert_array_equal(res.removed, base.removed)
    np.testing.assert_array_equal(res.subsets, base.subsets)
    assert_rel(res.rss_per_size, base.rss_per_size, 1e-9)


def test_errors():
    """PRUNE-1, LIMIT-3, GCV-1 and the input checks raise ValueError."""
    B, y, _ = _case(1, 12, 3, 1, False)
    bad = [
        lambda: pr.pruning_pass(B, y, penalty=2.0, pmethod="exhaustive"),
        lambda: pr.pruning_pass(B, y, penalty=2.0, nprune=0),
        lambda: pr.pruning_pass(B, y, penalty=2.0, nprune=True),
        lambda: pr.pruning_pass(B, y, penalty=-0.5),
        lambda: pr.pruning_pass(B[:, 1:], y, penalty=2.0),
        lambda: pr.pruning_pass(B + np.eye(12, 3), y, penalty=2.0),
        lambda: pr.pruning_pass(B[0], y, penalty=2.0),
        lambda: pr.pruning_pass(B, y[:-1], penalty=2.0),
        lambda: pr.pruning_pass(B, np.empty((12, 0)), penalty=2.0),
        lambda: pr.pruning_pass(B, np.where(y > 0, np.inf, y), penalty=2.0),
        lambda: pr.pruning_pass(B, y, -np.ones(12), penalty=2.0),
        lambda: pr.pruning_pass(B, y, np.ones(11), penalty=2.0),
        lambda: pr.pruning_pass(B, y, np.zeros(12), penalty=2.0),
        lambda: pr.pruning_pass(B[:2], y[:2], penalty=2.0),
        lambda: pr.final_fit(B, y, [1, 2], penalty=2.0),
        lambda: pr.final_fit(B, y, [0, 2, 1], penalty=2.0),
        lambda: pr.final_fit(B, y, [0, 1, 1], penalty=2.0),
        lambda: pr.final_fit(B, y, [0, 3], penalty=2.0),
        lambda: pr.final_fit(B, y, [], penalty=2.0),
        lambda: pr.final_fit(B, y, [0.0, 1.0], penalty=2.0),
    ]
    for call in bad:
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError, match="independent"):
        pr.pruning_pass(np.column_stack([B, np.zeros(12)]), y, penalty=2.0)


def _exact(B, Y, w, cols):
    """Weighted least squares of Y on the columns ``cols`` of B in rational
    arithmetic: the coefficients, one row per response, and the RSS summed
    over the responses (LA-1)."""
    A = [[Fraction(v) for v in row] for row in B[:, cols]]
    W = [Fraction(v) for v in (np.ones(B.shape[0]) if w is None else w)]
    n, m = len(A), len(cols)
    beta, rss = [], Fraction(0)
    for col in np.reshape(Y, (n, -1)).T:
        y = [Fraction(v) for v in col]
        E = [  # the normal equations [G | g], solved by Gauss-Jordan
            [sum(W[i] * A[i][r] * A[i][c] for i in range(n)) for c in range(m)]
            + [sum(W[i] * A[i][r] * y[i] for i in range(n))]
            for r in range(m)
        ]
        for c in range(m):
            E[c] = [v / E[c][c] for v in E[c]]
            for r in range(m):
                if r != c:
                    E[r] = [a - E[r][c] * b for a, b in zip(E[r], E[c], strict=True)]
        b = [E[r][m] for r in range(m)]
        beta.append([float(v) for v in b])
        fit = [sum(A[i][c] * b[c] for c in range(m)) for i in range(n)]
        rss += sum(W[i] * (y[i] - fit[i]) ** 2 for i in range(n))
    return np.array(beta), float(rss)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_a_large_mean_keeps_the_sums_of_squares_exact(k, weighted):
    """LA-5 for responses whose means are 1e13 times their spread: every subset
    holds the intercept (PRUNE-3), so centering Y changes no RSS. Each RSS of
    the records (PRUNE-4), the TSS (GCV-5), and the final RSS and coefficients
    (PRUNE-8) match rational arithmetic to relative 1e-8. Without the
    centering the RSS errors reach 1e-3 and the TSS errors 1e-6."""
    rng = np.random.default_rng(0)
    x = rng.uniform(size=50)
    B = np.column_stack([np.ones(50), np.maximum(x - 0.5, 0), np.maximum(0.3 - x, 0)])
    means = np.array([1e13, -4e12, 7e12])[:k]
    Y = means + np.sin(6 * x)[:, None] + 0.1 * rng.normal(size=(50, k))
    Y = Y[:, 0] if k == 1 else Y
    w = rng.uniform(0.5, 2.0, 50) if weighted else None
    res = pr.pruning_pass(B, Y, w, penalty=2.0)
    exact = [_exact(B, Y, w, np.flatnonzero(row))[1] for row in res.subsets]
    assert_rel(res.rss_per_size, exact, 1e-8)
    assert_rel(_gcv.tss(Y, w), exact[0], 1e-8)
    fit = pr.final_fit(B, Y, res.selected, w, penalty=2.0)
    beta, rss = _exact(B, Y, w, res.selected)
    assert_rel(fit.rss, rss, 1e-8)
    assert_rel(fit.coef, beta.T, 1e-8)
