"""Tests of pymars/_pruning.py (docs/algorithm.md: PRUNE-1 to PRUNE-9, with
GCV-2, GCV-5 to GCV-7, LIMIT-3, LA-4, W-1 to W-5, RESP-1, CORE-3 and EDGE-6).

The earth fixtures are ``validation/fixtures/components/pruning_fixed_basis``
(``earth:::pruning.pass`` on an earth forward basis, for one response and for
three) and the S datasets (``prune.terms``, ``rss.per.subset``,
``gcv.per.subset``, the selected terms, the coefficients and the statistics of
earth 5.3.4). The tests build earth's forward basis again from its ``dirs`` and
``cuts`` (TERM-3) and prune it. Tolerances, from the plan's tolerance table:
the exact terms at every size, relative 1e-8 for RSS and GCV (relative 1e-7
where earth keeps a zero weight, W-3), absolute 1e-8 for RSq and GRSq,
normwise relative 1e-6 for coefficients where κ(B) ≤ 1e5, and 1e-8·sd(y) for
fitted values. Against exact arithmetic the bound is LA-5's
ε(V*, V*) = 1e-8 V* + c sqrt(TSS V*) + (c/2)² TSS, with c = 1e-14.

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


def _spec_pass(B, Y, w, refit=_rss):
    """PRUNE-2 to PRUNE-4 as the spec writes them, with a refit of every subset
    by ``refit`` (numpy's lstsq, or rational arithmetic).

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
            r = refit(B, Y, w, U)
            if sets[m - 1] is not None and sets[m - 1] != U:
                gap = min(gap, abs(r - best[m - 1]) / best[m - 1])
            if r < best[m - 1]:
                best[m - 1], sets[m - 1] = r, U

    if several:
        best[M - 1], sets[M - 1] = refit(B, Y, w, order), frozenset(order)
    else:
        offer()
    removed = []
    for pos in range(M, 1, -1):
        drops = [
            (refit(B, Y, w, set(order[:pos]) - {order[i]}), order[i], i)
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


def assert_la5(actual, exact, tss):
    """LA-5 with R = V*: every RSS within ε(V*, V*) of its exact value, for
    the total sum of squares tss (GCV-5)."""
    actual, exact = np.asarray(actual, float), np.asarray(exact, float)
    bound = 1e-8 * exact + 1e-14 * np.sqrt(tss * exact) + (1e-14 / 2) ** 2 * tss
    assert np.all(np.abs(actual - exact) <= bound), (actual, exact)


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
    rtol = 1e-7 if w is not None and np.any(w == 0) else 1e-8  # W-3
    assert_rel(res.rss_per_size[shown] / scale, rss[shown], rtol)
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
    rtol = 1e-7 if w is not None and np.any(w == 0) else 1e-8  # W-3
    assert_rel(fit.rss / scale, r["rss"], rtol)
    if w is None:
        assert_rel(fit.gcv, r["gcv"], 1e-8)
        assert abs(fit.rsq - r["rsq"]) <= 1e-8
        assert abs(fit.grsq - r["grsq"]) <= 1e-8


def test_s_fixture_list():
    """The collection found the fixtures."""
    assert len(S_FITS) >= 130


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
    """A constant response becomes 0 exactly, so it changes no removal and no
    subset, and every RSS only by the order of the additions; its
    coefficients are its value and zeros (Conventions, PRUNE-8)."""
    B, y, _ = _case(8, 25, 5, 1, False)
    Y = np.column_stack([y, np.full(25, -1.5), 2 * y])
    two = pr.pruning_pass(B, Y[:, [0, 2]], penalty=2.0)
    three = pr.pruning_pass(B, Y, penalty=2.0)
    np.testing.assert_array_equal(three.removed, two.removed)
    np.testing.assert_array_equal(three.subsets, two.subsets)
    assert_rel(three.rss_per_size, two.rss_per_size, 1e-14)
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
        lambda: pr.pruning_pass(B, y, penalty=2.0, X=np.zeros((12, 2))),
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


def _exact_basis(X, dirs, cuts):
    """The columns of the terms (TERM-3) as Fractions: every factor and every
    product is exact."""
    Bx = np.empty((X.shape[0], dirs.shape[0]), dtype=object)
    for i, k in np.ndindex(Bx.shape):
        Bx[i, k] = Fraction(1)
        for j in np.flatnonzero(dirs[k]):
            x, c = Fraction(X[i, j]), Fraction(float(cuts[k, j]))
            Bx[i, k] *= {1: max(x - c, 0), -1: max(c - x, 0), 2: x}[int(dirs[k, j])]
    return Bx


def _linear_factor_case(rng, weights):
    """Terms 1, (x0 - c0)+, x1, x0, x1·h0, h0·(x2 - c2)+, x2·h0, x1·x2 and
    x1·x2·h0 on X = 2^36 + U (U uniform on (0, 1)), Y of a near-exact fit:
    the rounded basis B, the exact basis of TERM-3 as Fractions, Y, w and the
    keywords of ``pruning_pass`` that describe the terms. The last two terms
    have a parent with a linear factor (x1, x1·h0), and x2·h0 and x1·x2 are
    separated from their parents by other terms."""
    U = rng.uniform(size=(40, 3))
    X = 2.0**36 + U
    dirs = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [2, 0, 0],
            [1, 2, 0],
            [1, 0, 1],
            [1, 0, 2],
            [0, 2, 2],
            [1, 2, 2],
        ]
    )
    cuts = np.zeros(dirs.shape)
    cuts[[1, 4, 5, 6, 8], 0] = np.sort(X[:, 0])[12]
    cuts[5, 2] = np.sort(X[:, 2])[20]

    h = np.maximum(U[:, 0] - 0.3, 0)
    Y = 2 * h + 3 * h * U[:, 1] + U[:, 1] * U[:, 2] + U[:, 0]
    Y = Y + 4 * h * U[:, 1] * U[:, 2] + 0.01 * rng.normal(size=40)
    w = None if weights == "none" else rng.uniform(0.5, 2, 40)
    if w is not None:  # not a case (W-3), so it must not set m_j (LA-4)
        X[0], w[0] = 0.5, 0.0
    B = _terms.basis_matrix(X, dirs, cuts)
    Bx = _exact_basis(X, dirs, cuts)
    return B, Bx, Y, w, {"X": X, "dirs": dirs, "cuts": cuts}


def _pair_case(basis):
    """PR #96's review: the pair at one knot under a shifted linear parent,
    1, x0, x1, x0·(x1 - t)+, x0·(t - x1)+, x0 = 2^36 + U0, x1 = U1. The two
    hinges differ by x1 - t on every case, so their monomials and u1 are
    dependent unless (t - x)+ is written (x - t)+ - x + t. Or a hinge that is
    linear on the cases: 1, x1, x0·(x1 - t)+, x0·x1 with t = min x1. Or both
    covariates shifted, x0 = 1000 + U0 and x1 = 5 + U1, with 1, x0, x1,
    x0·(t - x1)+ at t = median x1, and x0·(x1 - 4.5)+, linear on the cases
    with t - m_1 ≠ 0 (the round-1 recheck of PR #96)."""
    if basis == "shifted hinges at 1000":
        rng = np.random.default_rng(0)
        U = rng.uniform(size=(20, 2))
        X = np.column_stack([1000 + U[:, 0], 5 + U[:, 1]])
        t = float(np.median(X[:, 1]))
        dirs = np.array([[0, 0], [2, 0], [0, 2], [2, -1], [2, 1]])
        cuts = np.array([[0, 0], [0, 0], [0, 0], [0, t], [0, 4.5]])
        Y = U[:, 0] * U[:, 1] + U[:, 1] + 0.01 * rng.normal(size=20)
        B = _terms.basis_matrix(X, dirs, cuts)
        kw = {"X": X, "dirs": dirs, "cuts": cuts}
        return B, _exact_basis(X, dirs, cuts), Y, None, kw
    rng = np.random.default_rng(1)
    U = rng.uniform(size=(30, 2))
    X = np.column_stack([2.0**36 + U[:, 0], U[:, 1]])
    t = float(np.sort(U[:, 1])[15])
    dirs = np.array([[0, 0], [2, 0], [0, 2], [2, 1], [2, -1]])
    cuts = np.array([[0, 0], [0, 0], [0, 0], [0, t], [0, t]])
    if basis != "a pair under x0 at 2^36":
        t = float(U[:, 1].min())
        dirs = np.array([[0, 0], [0, 2], [2, 1], [2, 2]])
        cuts = np.array([[0, 0], [0, 0], [0, t], [0, 0]])
    Y = U[:, 0] * np.maximum(U[:, 1] - t, 0) + U[:, 1] + 0.01 * rng.normal(size=30)
    B = _terms.basis_matrix(X, dirs, cuts)
    return B, _exact_basis(X, dirs, cuts), Y, None, {"X": X, "dirs": dirs, "cuts": cuts}


@pytest.mark.parametrize(
    ("k", "weights", "basis"),
    [
        (1, "none", "hinges"),
        (3, "none", "hinges"),
        (1, "uniform", "hinges"),
        (3, "uniform", "hinges"),
        (1, "far first", "hinges"),
        (1, "none", "binary at 33"),
        (1, "far first", "binary at 33"),
        (1, "far first", "x and a hinge at 33"),
        (1, "none", "linear factors at 2^36"),
        (1, "uniform", "linear factors at 2^36"),
        (1, "none", "a pair under x0 at 2^36"),
        (1, "none", "a hinge linear on the cases at 2^36"),
        (1, "none", "shifted hinges at 1000"),
    ],
)
def test_a_large_mean_keeps_the_sums_of_squares_exact(k, weights, basis):
    """LA-5 for responses whose means are 1e13 times their spread: every subset
    holds the intercept (PRUNE-3), so centering Y changes no RSS. Each RSS of
    the records (PRUNE-4), the TSS (GCV-5) and the final RSS (PRUNE-8) match
    rational arithmetic to LA-5's ε, and the coefficients to normwise
    1e-6. Without the centering the RSS errors reach 1e-3 and the TSS errors
    1e-6. In the "far first" case of the hinges the first 30 cases lie 2e13
    below the other 20 and have weight 1e-30 (W-6 allows it), so the shift
    must be the data value nearest the weighted mean: the first value, the
    value nearest the unweighted mean, or an unweighted mean of the
    differences misses by 1e-6 or more. The "binary at 33" cases have a
    column of B far from 0 (#84): x binary at 33 +- 0.25 and a near-exact
    fit, RSq = 1 - 1e-12. The pass centers the columns after the intercept
    too; without that, the RSS of [1, x] misses by 2.8e-8. With "far first"
    there, 30 more cases at x = 1e7 have weight 1e-40; in the last case the
    first 10 of B = [1, x, (x - 33)+] lie at x = -1e9 with weight 1e-30. So
    the columns need the weighted rule of Y: an unweighted centering, a shift
    by the first row or by the unweighted mean misses by 1e-3 or more. In the
    "linear factors at 2^36" cases X = 2^36 + U, with U in (0, 1) and a
    near-exact fit (#84, #105; ``_linear_factor_case``). B is the rounded
    product of the exact basis, and rounding x·h to 53 bits hides the part of
    x1·h0 that x1 spreads over h0: the RSS of subsets that hold both missed by
    1e-5 with B alone, and by 1.7e-4 with the terms rebuilt from x - a (PR
    #96's first version), through x1·x2·h0, whose parent x1·h0 has a linear
    factor too. Both functions take X, dirs and cuts and work in the shifted
    atoms of LA-4, every RSS is compared with the exact basis of TERM-3, and
    the final fit keeps the selected x1·h0, x2·h0, x1·x2 and x1·x2·h0, all
    of which LA-4 without the shift drops. With weights, row 0 sits at
    x = 0.5 with weight 0: not a case (W-3), so it must not set m_j; when it
    did, the RSS missed LA-5 by a ratio of 6e4 (PR #96's review). The "pair
    under x0" case is the pair at one knot under a shifted linear factor
    (``_pair_case``), which missed by 2.5e-4 relative when the two hinges
    were separate monomials; a hinge that is linear on the cases missed the
    same way, unless it is written as that line."""
    rng = np.random.default_rng({"hinges": 0, "binary at 33": 189}.get(basis, 1))
    kw, Bx = {}, None
    if basis == "linear factors at 2^36":
        B, Bx, Y, w, kw = _linear_factor_case(rng, weights)
    elif basis.endswith(("at 2^36", "at 1000")):
        B, Bx, Y, w, kw = _pair_case(basis)
    elif basis == "binary at 33":
        x = np.where(rng.random(109) < 0.5, 32.98083136553032, 33.48083136553032)
        w = None
        if weights == "far first":
            x = np.concatenate([x, np.full(30, 1e7)])
            w = np.concatenate([np.ones(109), np.full(30, 1e-40)])
        B = np.column_stack([np.ones(x.size), x])
        Y = 2.5564 * x + 1e-6 * rng.standard_normal(x.size)
    elif basis == "x and a hinge at 33":
        x, w = 33 + rng.uniform(-0.25, 0.25, 60), rng.uniform(0.5, 2, 60)
        x[:10], w[:10] = -1e9, 1e-30
        B = np.column_stack([np.ones(60), x, np.maximum(x - 33, 0)])
        Y = 2.5 * x + 1e-6 * rng.standard_normal(60)
    else:
        x = rng.uniform(size=50)
        B = np.column_stack(
            [np.ones(50), np.maximum(x - 0.5, 0), np.maximum(0.3 - x, 0)]
        )
        means = np.array([1e13, -4e12, 7e12])[:k]
        Y = means + np.sin(6 * x)[:, None] + 0.1 * rng.normal(size=(50, k))
        w = None if weights == "none" else rng.uniform(0.5, 2.0, 50)
        if weights == "far first":
            Y[:30], w[:30] = -means, 1e-30
        Y = Y[:, 0] if k == 1 else Y
    res = pr.pruning_pass(B, Y, w, penalty=2.0, **kw)
    Bf, B = B, (B if Bx is None else Bx)  # the exact basis, where there is one
    exact = [_exact(B, Y, w, np.flatnonzero(row))[1] for row in res.subsets]
    assert_la5(res.rss_per_size, exact, exact[0])
    if Bx is not None:  # the stages choose as the spec does in rational arithmetic
        removed, _, sets, _ = _spec_pass(
            Bx, Y, w, lambda B, Y, w, cols: _exact(B, Y, w, sorted(cols))[1]
        )
        assert res.removed.tolist() == removed
        assert _sets(res.subsets) == sets
    assert_la5([_gcv.tss(Y, w)], [exact[0]], exact[0])
    fit = pr.final_fit(Bf, Y, res.selected, w, penalty=2.0, **kw)
    beta, rss = _exact(B, Y, w, res.selected)
    assert_la5([fit.rss], [rss], exact[0])
    assert np.linalg.norm(fit.coef - beta.T) <= 1e-6 * np.linalg.norm(beta)


def test_fit_mars_passes_the_terms_to_the_pruning_pass():
    """LA-5 end to end (#84): a degree-2 fit of 40 cases at X = 2^23 + 8·U has
    linear factors in its terms, and each rss_per_size of the fit matches
    rational arithmetic on the exact basis of its own kept terms to relative
    1e-8. Before the pass took the terms, this case (seed 11) missed by 1.1e-8."""
    from pymars import _core

    rng = np.random.default_rng(11)
    noise = rng.choice([0.01, 0.2, 1.0])
    U = rng.uniform(size=(40, 3))
    h = np.maximum(U[:, 0] - 0.3, 0)
    Y = 2 * h + 3 * h * U[:, 1] + U[:, 1] * U[:, 2] + noise * rng.normal(size=40)
    X = 2.0**23 + 8 * U
    fit = _core.fit_mars(X, Y, None, _core.MarsParams(max_degree=2, max_terms=13))
    kept = fit.forward.kept
    dirs, cuts = fit.forward.dirs[kept], fit.forward.cuts[kept]
    assert (dirs == 2).any()
    Bx = _exact_basis(X, dirs, cuts)
    exact = [_exact(Bx, Y, None, np.flatnonzero(r))[1] for r in fit.pruning.subsets]
    assert_rel(fit.pruning.rss_per_size, exact, 1e-8)


@pytest.mark.parametrize("shift", [0.5, 1e4, 1e10])
@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("shift2", [0.0, 2.0**20])
def test_the_final_fit_applies_la4_after_the_shift(shift, copy, shift2):
    """LA-4 after the exact shift in the final fit (PRUNE-8), on the examples
    of the spec with #75's data, x0 = shift + (2, 0, 0, 2, 1, 2, 1, 2, 1, 0).
    The terms 1, x0, x0·x2, x2 all count, whatever the order of x0·x2 and x2
    (without the shift, LA-4 drops x0 and x2 at 1e10). With x3 = 3·x2 + 1, the
    term x0·x3 after 1, x0, x3, x0·x2 is dependent at every shift; x3 is one
    symbol with x2 (#99), else the RSS of the other four misses by 1e-6 at
    1e10. With x2 shifted by 2^20 too, x3 is a copy of a shifted covariate,
    and its linear factor is x3 itself, not x3 - m_3 (PR #96's review: an
    O(1) error). The RSS and the coefficients are those of the kept terms in
    rational arithmetic, to 1e-8 and to normwise 1e-6."""
    x0 = shift + np.array([2, 0, 0, 2, 1, 2, 1, 2, 1, 0.0])
    x2 = shift2 + np.array([1, 0, 0, 2, 2, 1, 0, 0, 1, 2.0])
    X = np.column_stack([x0, np.zeros(10), x2, 3 * x2 + 1])
    rows = [[0, 0, 0, 0], [2, 0, 0, 0], [2, 0, 2, 0], [0, 0, 2, 0]]
    if copy:
        rows = [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 2], [2, 0, 2, 0], [2, 0, 0, 2]]
    dirs = np.array(rows)
    cuts = np.zeros(dirs.shape)
    y = np.random.default_rng(3).normal(size=10)
    B = _terms.basis_matrix(X, dirs, cuts)
    sel = np.arange(len(rows))
    kw = {"X": X, "dirs": dirs, "cuts": cuts}
    fit = pr.final_fit(B, y, sel, None, penalty=2.0, **kw)
    kept = fit.coef[:, 0] != 0.0
    assert kept.tolist() == [True] * (len(rows) - copy) + [False] * copy
    Bx = _exact_basis(X, dirs, cuts)
    beta, exact = _exact(Bx, y, None, sel[kept])
    assert_rel(fit.rss, exact, 1e-8)
    assert np.linalg.norm(fit.coef[kept, 0] - beta[0]) <= 1e-6 * np.linalg.norm(beta)
