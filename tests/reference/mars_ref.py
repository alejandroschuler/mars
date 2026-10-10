"""Reference implementation of the pymars 2.0 fitting algorithm (the test oracle).

This module implements ``docs/algorithm.md`` (spec v2) literally, so that the
oracle tests can compare the fast code in ``pymars/`` with it
(VALIDATION_PLAN.md, "Tests and validation folders" and "Fast path"). It was
written from the spec alone. It imports nothing from ``pymars``, and its
author has not read the fast code, so that the two implementations fail
independently.

How it computes, and how that differs from the fast code:

- Every column is built explicitly from X, and every RSS is the residual of
  an explicit weighted least-squares projection [LA-1]: a column-pivoted
  Householder QR (``scipy.linalg.qr``) of the columns, with no updating
  formula and no suffix sum.
- The collinearity ratio [LA-3] is an explicit regression of the centered
  hinge column on the centered current columns.
- Dependence follows LA-4 after its exact shift of large-mean covariates:
  the reduced echelon form of the terms' exact expansions decides it, and
  in exact arithmetic when a covariate is shifted (``la4_dependent``). The
  projections use columns from an exact change of basis (``Conditioned``),
  so that a linear factor does not cost digits [LA-1, LA-5].
- The pruning pass refits every subset that its rules consider [PRUNE-9].
- The forward pass evaluates every candidate of every visited parent with
  its explicit columns, and follows the queue, the slots and the stopping
  rules of the spec one step at a time.
- ``fit_mars`` is the entry point: it returns the fields of ``MarsFit`` as
  the dict of CORE-5.
- Two knot scans are here: a loop that follows KNOT-3 step by step (unit
  weights), and the cumulative-weight scan of KNOT-6, so that each checks
  the other.

It is slow on purpose, and practical up to n <= 300, p <= 6 and
max_degree <= 3 [CORE-7]. Y and each column of X are scaled by powers of
2 first [EDGE-6, EDGE-7], and the knots and coefficients are scaled back;
sums of squares are formed from the scaled values as they are.
Bracketed IDs such as [GCV-2] cite the rules of the spec. Indices are
0-based, as in the spec. No function changes its inputs.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import numbers
from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import scipy.linalg

ALPHA = 0.05  # the probability in Friedman's span formulas [Notation]
DEPENDENT_TOL = 1e-7  # the dependency test for coefficients [LA-4]
EXACT_FIT = 1e-14  # the exact-fit band of FWD-5
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Weights [W-3, W-4]


def weight_sum(w) -> float:
    """N or N_b: the exactly rounded sum of the weights (``math.fsum``) [W-4]."""
    return math.fsum(np.asarray(w, dtype=np.float64).ravel().tolist())


def weight_tol(N: float) -> float:
    """tau_N = min(1e-8 * N, 0.1), the tolerance for sums of weights [W-4]."""
    return min(1e-8 * N, 0.1)


def snap(value: float, tau_N: float) -> float:
    """``value``, or the integer that replaces it when it is within tau_N [W-4]."""
    nearest = math.floor(value + 0.5)
    return float(nearest) if abs(value - nearest) <= tau_N else float(value)


# ---------------------------------------------------------------------------
# Terms [TERM-1 to TERM-4]


def factor(code: int, v: np.ndarray, c: float) -> np.ndarray:
    """The factor f(code, v, c) of TERM-3: (v - c)+, (c - v)+ or v."""
    v = np.asarray(v, dtype=np.float64)
    if code == 1:
        return np.maximum(v - c, 0.0)
    if code == -1:
        return np.maximum(c - v, 0.0)
    if code == 2:
        return v.copy()
    raise ValueError(f"unknown factor code {code}")


def basis_matrix(X, dirs, cuts) -> np.ndarray:
    """The n x M matrix B with B[i, k] = B_k(x_i) [TERM-3].

    The factors of a term multiply in increasing covariate order; nothing is
    clipped, so the formula holds outside the range of the training data.
    """
    X = np.asarray(X, dtype=np.float64)
    dirs = np.asarray(dirs)
    cuts = np.asarray(cuts, dtype=np.float64)
    B = np.ones((X.shape[0], dirs.shape[0]))
    for k in range(dirs.shape[0]):
        for j in np.flatnonzero(dirs[k]):
            B[:, k] = B[:, k] * factor(int(dirs[k, j]), X[:, j], float(cuts[k, j]))
    return B


def term_degree(dirs_row) -> int:
    """The degree of a term: its number of factors [TERM-4]."""
    return int(np.count_nonzero(dirs_row))


# ---------------------------------------------------------------------------
# GCV and fit statistics [GCV-1 to GCV-7]; the term limit [LIMIT-1]


def default_penalty(max_degree: int) -> float:
    """d for ``penalty=None``: 2 when max_degree is 1, else 3 [GCV-4]."""
    return 2.0 if max_degree == 1 else 3.0


def effective_params(M: int, penalty: float) -> float:
    """C(M) = M + d (M - 1) / 2 for d >= 0, and 0 for d = -1 [GCV-1]."""
    if penalty == -1:
        return 0.0
    return M + penalty * (M - 1) / 2


def gcv(rss: float, M: int, penalty: float, N: float, tau_N: float) -> float:
    """GCV(RSS, M) = RSS / (N (1 - C/N)^2) when C(M) < N - tau_N, else +inf [GCV-2]."""
    c_m = effective_params(M, penalty)
    if c_m < N - tau_N:
        return rss / (N * (1.0 - c_m / N) ** 2)
    return math.inf


def rsq(rss: float, tss: float) -> float:
    """RSq = 1 - RSS / TSS [GCV-5]."""
    return 1.0 - rss / tss


def grsq(rss: float, M: int, tss: float, penalty: float, N: float, tau_N: float):
    """GRSq = 1 - GCV(RSS, M) / GCV(TSS, 1), and -inf when GCV(RSS, M) is +inf
    [GCV-6]."""
    num = gcv(rss, M, penalty, N, tau_N)
    if num == math.inf:
        return -math.inf
    return 1.0 - num / gcv(tss, 1, penalty, N, tau_N)


def default_max_terms(p: int) -> int:
    """M_max for ``max_terms=None``: min(200, max(20, 2p)) + 1 [LIMIT-1]."""
    return min(200, max(20, 2 * p)) + 1


# ---------------------------------------------------------------------------
# Spans [SPAN-1 to SPAN-6]


def auto_minspan(p: int, Nb: float) -> int:
    """L = max(1, trunc(-log2(-ln(1 - alpha) / (p N_b)) / 2.5)) [SPAN-1].

    N_b must be positive: the caller asks for spans only for a parent with
    an active case.
    """
    return max(1, math.trunc(-math.log2(-math.log(1.0 - ALPHA) / (p * Nb)) / 2.5))


def auto_endspan(p: int) -> int:
    """E = trunc(3 - log2(alpha / p)) [SPAN-2]."""
    return math.trunc(3.0 - math.log2(ALPHA / p))


def adjusted_endspan(E: int, parent_degree: int, adjust_endspan: float) -> int:
    """E1 = E + floor(a E + 0.5) for a parent of degree >= 1, else E [SPAN-4]."""
    if parent_degree == 0:
        return E
    return E + math.floor(adjust_endspan * E + 0.5)


def scan_endspan(E1: int, N: float, tau_N: float) -> int:
    """E* = max(1, min(E1, floor((N + tau_N) / 2) - 1)) [SPAN-5]."""
    return max(1, min(E1, math.floor((N + tau_N) / 2) - 1))


def search_spans(
    p: int,
    parent_degree: int,
    Nb: float,
    N: float,
    tau_N: float,
    minspan: int | None = None,
    endspan: int | None = None,
    adjust_endspan: float = 2.0,
) -> tuple[int, int]:
    """(L, E*) for one knot search: a user value replaces the automatic L or E
    [SPAN-3], the adjustment and the cap apply to both kinds [SPAN-4, SPAN-5],
    and the minspan is not capped [SPAN-6]. N_b, the weight of the parent's
    active cases, must be positive when the minspan is automatic."""
    L = auto_minspan(p, Nb) if minspan is None else int(minspan)
    E = auto_endspan(p) if endspan is None else int(endspan)
    return L, scan_endspan(adjusted_endspan(E, parent_degree, adjust_endspan), N, tau_N)


# ---------------------------------------------------------------------------
# Candidate knots [KNOT-1 to KNOT-6]


def case_order(x: np.ndarray, active: np.ndarray) -> np.ndarray:
    """The order of KNOT-2: ascending x, and among equal x the inactive cases first."""
    return np.lexsort((np.asarray(active, dtype=np.int8), np.asarray(x)))


def scan_start(N: float, minspan: int, endspan: int) -> int:
    """c0 = E* + ceil(g / 2), the start of the counter of a scan, with
    g = D - L floor(D / L) and D = N - 2 E* - 1 [KNOT-3, KNOT-6]."""
    L, E = int(minspan), int(endspan)
    D = N - 2 * E - 1
    return E + math.ceil((D - L * math.floor(D / L)) / 2)


def knot_scan_unit(x, active, minspan: int, endspan: int) -> list[float]:
    """The knot list of KNOT-3 for unit weights, in scan order, repeats included.

    This follows the three steps of KNOT-3 one position at a time; knot_scan
    is the cumulative-weight form of KNOT-6, and the tests check that the two
    agree for unit weights.
    """
    x = np.asarray(x, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    n = x.shape[0]
    if not active.any():
        return []
    order = case_order(x, active)
    xs, act = x[order], active[order]
    v = x[active].max()
    L, E = int(minspan), int(endspan)
    counter = scan_start(n, L, E)
    knots = []
    for q in range(n, E + 1, -1):  # q = n, n - 1, ..., E* + 2 (1-based)
        t = xs[q - 2]  # x_(q-1)
        if t >= v:
            continue
        if act[q - 1]:  # a_q
            counter -= 1
            if counter == 0:
                knots.append(float(t) + 0.0)  # + 0.0: a zero knot has no sign
                counter = L
    return knots


def _smallest_from(guess: int, holds) -> int:
    """The smallest k >= 0 with holds(k), for a condition that holds from some
    k on, found from a guess a step or so away. The steps are bounded, so a
    guess that is far off fails at once instead of running on."""
    k = max(0, guess)
    for _ in range(4):
        if k > 0 and holds(k - 1):
            k -= 1
        elif not holds(k):
            k += 1
        else:
            return k
    raise AssertionError(f"the guess {guess} is not within a few steps")


def knot_scan(
    x,
    active,
    w,
    minspan: int,
    endspan: int,
    N: float,
    tau_N: float,
    distinct: bool = False,
):
    """The knot list of KNOT-6 (cumulative weight), in scan order, repeats included.

    The scan visits u = N, N - 1, ... while u >= E* + 2 - tau_N; visit k
    (u = N - k) takes t = x(u - 1) and a(u) from the cases that hold u - 1
    and u. Those two cases stay the same over stretches of visits, so the
    visits of a stretch are counted at once (KNOT-7): the counter moves in
    every visit of a stretch or in none, and a knot is appended at the
    c0-th move and at every L-th move after it, where the counter of KNOT-3
    reaches 0. The state of the scan takes time O(n log n) and memory O(n),
    whatever N is, so the whole call takes time O(n log n + K) and memory
    O(n + K), where K is the length of the returned list (every listing of a
    knot is kept). With ``distinct=True`` only the first listing of each
    value is kept, which is ``distinct_knots`` of the list, and the call takes
    O(n log n) time and O(n) memory. The tests check this form against a loop
    over single visits.
    """
    x = np.asarray(x, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    w = np.asarray(w, dtype=np.float64)
    if not active.any():
        return []
    L, E = int(minspan), int(endspan)
    bound = E + 2 - tau_N

    def first_visit(level: float) -> int:
        # The smallest k >= 0 with N - k <= level; N - k is exact in float64.
        return _smallest_from(math.ceil(N - level), lambda k: N - k <= level)

    # the visits are k = 0, ..., visits - 1: the first k that is not a visit
    visits = _smallest_from(math.floor(N - bound) + 1, lambda k: N - k < bound)
    if visits == 0:
        return []
    order = case_order(x, active)
    xs, act = x[order], active[order]
    n = xs.shape[0]
    # The case that holds u is the smallest q with u <= W_q + tau_N, else the
    # last case; first[q] is the first visit at which case q or a lower one
    # holds u, so it does not increase with q.
    first = np.array([first_visit(level) for level in np.cumsum(w[order]) + tau_N])

    def holder(k: int) -> int:  # the case that holds u = N - k
        return min(int(np.searchsorted(-first, -k, side="left")), n - 1)

    marks = {0, visits}
    for f in first.tolist():
        marks.update(b for b in (f - 1, f) if 0 < b < visits)
    marks = sorted(marks)
    v = x[active].max()
    c0 = scan_start(N, L, E)
    knots, moves = [], 0
    for begin, end in itertools.pairwise(marks):
        t = xs[holder(begin + 1)]
        if not (t < v and act[holder(begin)]):
            continue
        stretch = end - begin
        hit = c0 + L * max(0, -(-(moves + 1 - c0) // L))  # the next knot's move
        if hit <= moves + stretch:
            if not distinct:
                knots += [float(t) + 0.0] * ((moves + stretch - hit) // L + 1)
            elif not knots or knots[-1] != t:
                knots.append(float(t) + 0.0)
        moves += stretch
    return knots


def distinct_knots(knot_list) -> list[float]:
    """The candidate knots: the distinct values of a knot list, largest first.

    A scan lists knots from the top down, so the first listing of each value
    keeps the order of FWD-5 (the knots from the largest down).
    """
    out: list[float] = []
    for t in knot_list:
        if not out or t != out[-1]:
            out.append(float(t))
    return out


# ---------------------------------------------------------------------------
# Linear algebra [LA-1 to LA-4, LA-7]


def _orthonormal_basis(A: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the numerical span of the columns of A.

    The columns are scaled to unit norm (zero columns are left out), then
    factored by a column-pivoted Householder QR; the rank counts the
    diagonal entries of R above max(n, m) * eps times the largest.
    """
    n = A.shape[0]
    norms = np.sqrt(np.sum(A * A, axis=0))
    nonzero = norms > 0
    if not nonzero.any():
        return np.zeros((n, 0))
    unit = A[:, nonzero] / norms[nonzero]
    Q, R, _ = scipy.linalg.qr(unit, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.count_nonzero(diag > max(unit.shape) * EPS * diag[0]))
    return Q[:, :rank]


def _residual(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """V minus its projection on the orthonormal columns of Q, done twice."""
    R = V - Q @ (Q.T @ V)
    return R - Q @ (Q.T @ R)


class Projector:
    """The w-orthogonal projection on the span of the columns of B [LA-1].

    Every vector is scaled by sqrt(w), so that the weighted problem becomes
    an ordinary one. When column 0 of B is a nonzero constant (the intercept
    of every RSS in the spec), every other column and every projected vector
    is first centered at its weighted mean: that changes no span and no
    residual, and it keeps a large mean from costing digits.
    """

    def __init__(self, B, w):
        B = np.asarray(B, dtype=np.float64)
        self._w = np.asarray(w, dtype=np.float64)
        self._wsum = float(np.sum(self._w))
        self._sw = np.sqrt(self._w)
        first = B[:, 0] if B.shape[1] else None
        self.centered = (
            first is not None and first[0] != 0 and bool(np.all(first == first[0]))
        )
        if self.centered:
            A = np.column_stack([self._sw, self.scaled(B[:, 1:])])
        else:
            A = self.scaled(B)
        self.Q = _orthonormal_basis(A)

    @property
    def rank(self) -> int:
        """The numerical rank of B."""
        return self.Q.shape[1]

    def scaled(self, V) -> np.ndarray:
        """sqrt(w) times V, centered first when B holds the intercept."""
        V = np.asarray(V, dtype=np.float64)
        if self.centered:
            V = V - (self._w @ V) / self._wsum
        return V * (self._sw if V.ndim == 1 else self._sw[:, None])

    def residual(self, V) -> np.ndarray:
        """The sqrt(w)-scaled residual of V (n,) or (n, m) on the span of B."""
        return _residual(self.Q, self.scaled(V))

    def rss(self, Y) -> float:
        """RSS(B) of LA-1: the weighted residual sum of squares, summed over the
        columns of Y."""
        return float(np.sum(self.residual(Y) ** 2))


def rss(B, Y, w) -> float:
    """RSS(U) of LA-1 for the columns U of B: the sum over the responses of
    ||Y_k - P_U Y_k||_w^2. Dependent columns do not change it."""
    return Projector(B, w).rss(Y)


def collinearity_ratio(h, G, w) -> float:
    """rho(t) = ||h - P_G h||_w^2 / ||h - hbar||_w^2 [LA-3].

    G holds the intercept, so this is 1 - R^2 of the centered hinge column
    regressed on the centered columns of G. The caller rejects a constant h
    before the call.
    """
    P = Projector(G, w)
    if not P.centered:
        raise ValueError("column 0 of G must be the intercept")
    return float(np.sum(P.residual(h) ** 2) / np.sum(P.scaled(h) ** 2))


def weighted_variance(v, w, N: float) -> float:
    """sigma_v^2 = sum w_i (v_i - vbar)^2 / N, with vbar the weighted mean, and 0
    exactly for a constant v [Notation, Conventions]."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if np.all(v == v[0]):
        return 0.0
    mean = float(w @ v) / float(np.sum(w))
    return float(w @ (v - mean) ** 2) / N


def dependent_columns(B, w) -> np.ndarray:
    """The dependent columns of B, as a boolean mask [LA-4].

    The columns are taken in their order. A column is dependent when the
    w-norm of its part orthogonal to the kept earlier columns is less than
    1e-7 times its own w-norm, not centered. A column of zeros is dependent.
    """
    B = np.asarray(B, dtype=np.float64)
    A = B * np.sqrt(np.asarray(w, dtype=np.float64))[:, None]
    Q = np.zeros((A.shape[0], 0))
    dependent = np.zeros(A.shape[1], dtype=bool)
    for j in range(A.shape[1]):
        column = A[:, j]
        size = float(np.linalg.norm(column))
        rest = _residual(Q, column)
        rest_size = float(np.linalg.norm(rest))
        if size == 0.0 or rest_size < DEPENDENT_TOL * size:
            dependent[j] = True
        else:
            Q = np.column_stack([Q, rest / rest_size])
    return dependent


def lstsq_coef(B, Y, w) -> np.ndarray:
    """Weighted least-squares coefficients of Y on the columns of B [LA-4].

    A dependent column (dependent_columns) gets the coefficient 0, and the
    kept columns get the least-squares solution, by a column-pivoted QR of
    the sqrt(w)-scaled kept columns. When column 0 is the intercept, Y is
    centered first and the mean is added back to its coefficient, which
    changes nothing but the rounding. Returns shape (M, K) for Y of shape
    (n, K), and (M,) for a 1-D Y.
    """
    B = np.asarray(B, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    Y2 = Y.reshape(Y.shape[0], -1)
    coef = np.zeros((B.shape[1], Y2.shape[1]))
    keep = ~dependent_columns(B, w)
    if keep.any():
        shift = np.zeros(Y2.shape[1])
        if keep[0] and np.all(B[:, 0] == 1.0):
            shift = (w @ Y2) / np.sum(w)
        sw = np.sqrt(w)
        Q, R, piv = scipy.linalg.qr(
            B[:, keep] * sw[:, None], mode="economic", pivoting=True
        )
        solution = scipy.linalg.solve_triangular(R, Q.T @ ((Y2 - shift) * sw[:, None]))
        beta = np.empty_like(solution)
        beta[piv] = solution
        if keep[0]:
            beta[0] = beta[0] + shift
        coef[keep] = beta
    return coef if Y.ndim > 1 else coef[:, 0]


# ---------------------------------------------------------------------------
# The pruning pass [PRUNE-1 to PRUNE-8]


def select_size(gcv_per_size, m_max: int) -> int:
    """m*: the smallest m in 1..m_max at which gcv_per_size[m - 1] is lowest
    [PRUNE-5]; tied minima go to the smaller size."""
    best = 1
    for m in range(2, m_max + 1):
        if gcv_per_size[m - 1] < gcv_per_size[best - 1]:
            best = m
    return best


def prune(
    B,
    Y,
    w,
    *,
    penalty: float,
    N: float,
    tau_N: float,
    pmethod: str = "backward",
    nprune: int | None = None,
    columns_of=None,
    coef_of=None,
) -> dict:
    """The pruning pass on the kept forward terms [PRUNE-2 to PRUNE-8].

    B is the n x M_f matrix of the kept terms in forward order (column 0 is
    the intercept), Y is n x K and w the weights; the results are in the
    units of Y. Every subset that the rules consider is refitted from its
    columns [PRUNE-9], once: the RSS of each set is kept for reuse.

    Returns the fields of ``PruningRecord`` (removed, rss_per_size,
    gcv_per_size, subsets, selected_size) together with ``selected`` (the
    pruning indices of the selected terms, increasing), ``coef`` (m*, K),
    ``rss``, ``gcv``, ``rsq`` and ``grsq`` of the final model [PRUNE-8], and
    ``tss``.

    ``columns_of``, when given, maps the sorted indices of a subset of the
    terms to columns that span what the terms span; every RSS then uses them
    in place of the columns of B, which keeps the digits that a large
    covariate mean would cost in the float columns of B [LA-5]
    (``Conditioned.span_columns``). ``coef_of(selected, Y, w)``, when given,
    returns the final coefficients and the mask of the selected terms that
    LA-4 keeps (``Conditioned.coef``); else they come from B (lstsq_coef).

    The fit must not be degenerate: ``fit_mars`` applies EDGE-1 and the
    degenerate values of GCV-7 before it calls this function, so here
    N > 1 and some response is not constant. The final rss is RSS(selected
    terms) of LA-1, the centered projection on the columns of the selected
    terms that LA-4 keeps, not the RSS of the rounded coefficients
    [PRUNE-8], so it meets LA-5 also when Y has a large mean. Cost:
    either rule refits O(M_f^2) distinct subsets, each in O(n M_f (M_f + K))
    time, so O(n M_f^3 (M_f + K)) in all, and the cache holds O(M_f^2) values.
    """
    B = np.asarray(B, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(B.shape[0], -1)
    w = np.asarray(w, dtype=np.float64)
    if pmethod not in ("backward", "none"):
        raise ValueError(f"pmethod must be 'backward' or 'none', not {pmethod!r}")
    if nprune is not None and nprune < 1:
        raise ValueError(f"nprune must be None or at least 1, not {nprune!r}")
    Mf = B.shape[1]
    memo: dict[frozenset, float] = {}
    if columns_of is None:

        def columns_of(rows):
            return B[:, rows]

    def rss_of(terms) -> float:
        key = frozenset(terms)
        if key not in memo:
            memo[key] = rss(columns_of(sorted(key)), Y, w)
        return memo[key]

    best_rss = [math.inf] * (Mf + 1)  # R[m], m = 1..Mf
    best_set: list[frozenset] = [frozenset()] * (Mf + 1)  # T[m]
    removed: list[int] = []

    def lowest_removal(terms) -> int:
        # The term (never the intercept) whose removal leaves the lowest RSS;
        # equal RSS values go to the term with the largest index. The terms
        # come in increasing index order (the first pos entries of the working
        # order stay increasing, as does the current set of K >= 2).
        choice, choice_rss = -1, math.inf
        for term in terms:
            if term == 0:
                continue
            value = rss_of(set(terms) - {term})
            if value < choice_rss or (value == choice_rss and term > choice):
                choice, choice_rss = term, value
        return choice

    if Y.shape[1] == 1:
        # One response: the working order and its offers [PRUNE-3].
        order = list(range(Mf))

        def offer():
            for m in range(1, Mf + 1):
                value = rss_of(order[:m])
                if value < best_rss[m]:
                    best_rss[m], best_set[m] = value, frozenset(order[:m])

        offer()
        for pos in range(Mf, 1, -1):
            term = lowest_removal(order[:pos])
            removed.append(term)
            i = order.index(term)
            order = order[:i] + order[i + 1 : pos] + [term] + order[pos:]
            offer()
    else:
        # Two or more responses: plain backward elimination [PRUNE-3].
        current = list(range(Mf))
        best_rss[Mf], best_set[Mf] = rss_of(current), frozenset(current)
        while len(current) > 1:
            term = lowest_removal(current)
            removed.append(term)
            current.remove(term)
            best_rss[len(current)] = rss_of(current)
            best_set[len(current)] = frozenset(current)

    rss_per_size = np.array(best_rss[1:], dtype=np.float64)
    gcv_per_size = np.array(
        [gcv(best_rss[m], m, penalty, N, tau_N) for m in range(1, Mf + 1)]
    )
    subsets = np.zeros((Mf, Mf), dtype=bool)
    for m in range(1, Mf + 1):
        subsets[m - 1, sorted(best_set[m])] = True

    if pmethod == "backward":  # [PRUNE-5, PRUNE-6]
        m_max = Mf if nprune is None else min(Mf, int(nprune))
        size = select_size(gcv_per_size, m_max)
        selected = sorted(best_set[size])
    else:  # [PRUNE-7]
        size = Mf if nprune is None else min(int(nprune), Mf)
        selected = list(range(size))

    # The final model [PRUNE-8]; the intercept-only model has RSq = GRSq = 0
    # by definition [GCV-7]. The returned coefficients are the least-squares
    # fit on the selected columns that LA-4 keeps, so their RSS is the RSS of
    # the projection on those columns [LA-1], which the Projector computes
    # centered; Y - BS coef on the raw Y would cancel at a large mean [LA-5].
    if coef_of is None:
        coef = lstsq_coef(B[:, selected], Y, w)
        keep = ~dependent_columns(B[:, selected], w)
    else:
        coef, keep = coef_of(selected, Y, w)
    final_rss = rss_of(np.array(selected)[keep].tolist())
    tss = best_rss[1]
    final_gcv = gcv(final_rss, size, penalty, N, tau_N)
    if size == 1:
        final_rsq = final_grsq = 0.0
    else:
        final_rsq = rsq(final_rss, tss)
        final_grsq = grsq(final_rss, size, tss, penalty, N, tau_N)
    return {
        "removed": np.array(removed, dtype=np.int64),
        "rss_per_size": rss_per_size,
        "gcv_per_size": gcv_per_size,
        "subsets": subsets,
        "selected_size": int(size),
        "selected": np.array(selected, dtype=np.int64),
        "coef": coef,
        "rss": final_rss,
        "gcv": final_gcv,
        "rsq": final_rsq,
        "grsq": final_grsq,
        "tss": tss,
    }


# ---------------------------------------------------------------------------
# Parameters [CORE-2]


def _is_int(value) -> bool:
    """An int field takes a Python int or a numpy integer, never a bool [CORE-2]."""
    return isinstance(value, numbers.Integral) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_real(value) -> bool:
    """A float field takes any real number type [CORE-2]."""
    return isinstance(value, numbers.Real)


@dataclasses.dataclass(frozen=True)
class Params:
    """The fields of ``MarsParams`` with their defaults, checked as CORE-2 says:
    the constructor raises ValueError, naming the field, for a value outside
    its range."""

    max_degree: int = 1
    max_terms: int | None = None
    penalty: float | None = None
    thresh: float = 0.001
    minspan: int | None = None
    endspan: int | None = None
    adjust_endspan: float = 2.0
    auto_linpreds: bool = True
    fast_k: int = 20
    fast_beta: float = 1.0
    pmethod: str = "backward"
    nprune: int | None = None

    def __post_init__(self):
        def whole(v, low, optional=False):
            return (optional and v is None) or (_is_int(v) and v >= low)

        def real(v, low):
            return _is_real(v) and math.isfinite(v) and v >= low

        d = self.penalty
        rules = [
            ("max_degree", whole(self.max_degree, 1), "an int >= 1"),
            ("max_terms", whole(self.max_terms, 1, True), "None or an int >= 1"),
            (
                "penalty",
                d is None or real(d, 0) or (_is_real(d) and d == -1),
                "None, -1 or a finite float >= 0",
            ),
            ("thresh", real(self.thresh, 0), "a finite float >= 0"),
            ("minspan", whole(self.minspan, 1, True), "None or an int >= 1"),
            ("endspan", whole(self.endspan, 1, True), "None or an int >= 1"),
            ("adjust_endspan", real(self.adjust_endspan, 0), "a finite float >= 0"),
            (
                "auto_linpreds",
                isinstance(self.auto_linpreds, (bool, np.bool_)),
                "a bool",
            ),
            ("fast_k", whole(self.fast_k, 0), "an int >= 0"),
            ("fast_beta", real(self.fast_beta, 0), "a finite float >= 0"),
            (
                "pmethod",
                isinstance(self.pmethod, str) and self.pmethod in ("backward", "none"),
                "'backward' or 'none'",
            ),
            ("nprune", whole(self.nprune, 1, True), "None or an int >= 1"),
        ]
        for name, ok, allowed in rules:
            if not ok:
                raise ValueError(
                    f"{name} must be {allowed}, not {getattr(self, name)!r}"
                )

    def resolved_max_terms(self, p: int) -> int:
        """M_max: the given max_terms, or the default of LIMIT-1."""
        return default_max_terms(p) if self.max_terms is None else int(self.max_terms)

    def resolved_penalty(self) -> float:
        """d: the given penalty, or the default of GCV-4."""
        if self.penalty is None:
            return default_penalty(self.max_degree)
        return float(self.penalty)


def as_params(params=None) -> Params:
    """``params`` as Params: None (all defaults), a Params, a mapping of field
    names, or an object with the fields as attributes (a ``MarsParams``)."""
    if params is None:
        return Params()
    if isinstance(params, Params):
        return params
    names = [field.name for field in dataclasses.fields(Params)]
    if isinstance(params, Mapping):
        unknown = sorted(set(params) - set(names))
        if unknown:
            raise ValueError(f"unknown parameters: {unknown}")
        return Params(**params)
    missing = [name for name in names if not hasattr(params, name)]
    if missing:
        raise ValueError(f"params has no field {missing}")
    return Params(**{name: getattr(params, name) for name in names})


# ---------------------------------------------------------------------------
# The forward pass [FWD-1 to FWD-11, STOP-1 to STOP-6, FAST-1 to FAST-6]

# Termination codes [CORE-4]
DEGENERATE, NO_ROOM, GRSQ_NEG_INF, GRSQ_LOW = 0, 1, 2, 3
RSQ_CHANGE_SMALL, RSQ_HIGH, NO_GAIN, TERM_LIMIT = 4, 5, 6, 7

# Candidate kinds [CORE-3]
NO_KIND, PAIR, SINGLE, LINEAR = 0, 1, 2, 3


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One candidate of a forward step [FWD-3]. Occurrences of one kind that
    add the same rows share one Candidate, that of the first [FWD-12]."""

    parent: int  # the forward index of the parent term
    variable: int
    kind: int  # PAIR, SINGLE or LINEAR
    knot: float  # NaN for a linear candidate
    rss: float  # the RSS of LA-2


def collinearity_tol(steps_taken: int) -> float:
    """tau of LA-3: 0.01 while the pass has taken at most 6 steps (so in steps
    1 to 7), and 1e-5 from step 8 on."""
    return 0.01 if steps_taken <= 6 else 1e-5


def max_legal(rss_path) -> float:
    """MaxLegal_s = min(1.01 RSS_s, 10 Delta_s), and 1.01 RSS_0 at the first
    step [FWD-4]; ``rss_path`` is RSS_0, ..., RSS_s."""
    if len(rss_path) == 1:
        return 1.01 * rss_path[0]
    return min(1.01 * rss_path[-1], 10.0 * (rss_path[-2] - rss_path[-1]))


def is_legal(kind: int, reduction: float, limit: float) -> bool:
    """A knot candidate is legal when 0 < reduction <= MaxLegal_s, a linear
    candidate when its reduction is positive, whatever its size [FWD-4]."""
    if kind == LINEAR:
        return reduction > 0
    return 0 < reduction <= limit


def reduction(c, rss_s: float) -> float:
    """The RSS reduction of candidate c, RSS_s minus its RSS, after the
    exact-fit band of FWD-5: RSS_s exactly when its computed RSS is at most
    EXACT_FIT RSS_s."""
    return rss_s if c.rss <= EXACT_FIT * rss_s else rss_s - c.rss


def queue_value(candidates, searchable: bool, rss_s: float, limit: float) -> float:
    """lambda_e of a searched entry [FAST-5]: the largest legal reduction among
    its candidates, after the exact-fit band, linear candidates of any size
    included; 0 when none is legal, and -1 when no covariate could be
    searched."""
    if not searchable:
        return -1.0
    return max(
        (
            reduction(c, rss_s)
            for c in candidates
            if is_legal(c.kind, reduction(c, rss_s), limit)
        ),
        default=0.0,
    )


def candidate_key(c) -> tuple:
    """How the candidate log names a candidate [CORE-3]: the parent, the
    variable, the kind and the knot, with None as the knot of a linear
    candidate. Two occurrences of one kind that add the same rows are one
    candidate, named by the first [FWD-12, merge_key]."""
    return (c.parent, c.variable, c.kind, None if c.kind == LINEAR else c.knot)


def covariate_variances(X, w, N: float) -> list[float]:
    """sigma_v^2 of every covariate, with divisor N [Notation, LA-7]."""
    return [weighted_variance(X[:, j], w, N) for j in range(X.shape[1])]


def queue_table(entries, steps_taken: int, fast_beta: float) -> list[int]:
    """The table of FAST-2: the entry numbers (1-based), in table order.

    ``entries[e - 1]`` is (lambda_e, kappa_e). rank_e is the 0-based place of
    entry e by lambda, largest first, equal values in increasing entry
    number; AgedRank_e = rank_e + fast_beta (kappa_prev - kappa_e); the table
    sorts by AgedRank, smallest first, equal values in increasing rank.
    Before the first step it holds the intercept alone.
    """
    if steps_taken == 0:
        return [1]
    kappa_prev = 2 * steps_taken
    by_value = sorted(range(1, len(entries) + 1), key=lambda e: (-entries[e - 1][0], e))
    rank = {e: r for r, e in enumerate(by_value)}
    aged = {e: rank[e] + fast_beta * (kappa_prev - entries[e - 1][1]) for e in by_value}
    return sorted(by_value, key=lambda e: (aged[e], rank[e]))


def window(table, fast_k: int) -> list[int]:
    """The rows that a step visits [FAST-3]: the first nu = max(3, fast_k), or
    every row when fast_k = 0 or nu >= M."""
    if fast_k == 0:
        return list(table)
    return list(table[: max(3, fast_k)])


# ---------------------------------------------------------------------------
# Dependence after the exact shift [LA-1, LA-2, LA-4, FWD-11, PRUNE-8]
#
# A linear factor x_j carries the mean of x_j into its term, and a float
# column loses the digits of a large mean. So LA-4 judges dependence after an
# exact change of basis. Covariate j has the shift m_j, its smallest value
# when that is larger in size than its range, else 0, and u_j = x_j - m_j.
# Every term is a polynomial in the atoms u_j, (x_j - t)+ and (t - x_j)+ (a
# hinge is its own atom; x_j = u_j + m_j), expanded into monomials with
# exact rational coefficients. The reduced echelon form of a set of terms,
# in LA-4's order of the monomials, is unique, so it does not depend on the
# order of the terms, and it decides dependence (la4_dependent). Its rows can
# carry a factor m_j on a monomial that is not a pivot, so the projections of
# the forward pass, the subsets of the pruning pass and the final fit use the
# rows of another exact elimination of the same terms, which span the same
# (Basis) [LA-1].
# Covariates whose columns u_j are equal bit for bit (an exact copy, or a
# copy shifted by a constant that the shift removes) share one symbol, so a
# product of such copies is exact too (#99).

U_ATOM, PLUS_ATOM, MINUS_ATOM = 0, 1, 2  # the kinds of atoms, in LA-4's order


def _large_mean(X) -> set:
    """The covariates whose smallest value is larger in size than their range."""
    low, high = X.min(axis=0), X.max(axis=0)
    return {j for j in range(X.shape[1]) if abs(low[j]) > high[j] - low[j]}


def monomial_order(monomial) -> tuple:
    """The key of LA-4's total order of monomials (tuples of atoms, sorted):
    by degree, highest first, then lexicographically on the atoms. An atom
    is (symbol, kind, knot - m_j)."""
    return (-len(monomial), monomial)


class Atoms:
    """The atoms of LA-4 on the data X: the shifts m_j, the symbols, the
    expansion of a term and the column of a monomial."""

    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.float64)
        p = self.X.shape[1]
        large = _large_mean(self.X)
        self.m = [float(self.X[:, j].min()) if j in large else 0.0 for j in range(p)]
        self.U = self.X - np.array(self.m)
        self.symbol = [
            next(i for i in range(j + 1) if np.array_equal(self.U[:, i], self.U[:, j]))
            for j in range(p)
        ]
        self.plain = not large  # every term is one monomial with coefficient 1
        self._columns: dict = {}

    def atom(self, j: int, code: int, knot: float) -> tuple:
        s = self.symbol[j]
        if code == 2:
            return (s, U_ATOM, Fraction(0))
        kind = PLUS_ATOM if code == 1 else MINUS_ATOM
        return (s, kind, Fraction(float(knot)) - Fraction(self.m[j]))

    def expansion(self, row, cut, rewrite: bool = False) -> dict:
        """The term (rows of dirs and cuts) as {monomial: coefficient}. With
        ``rewrite``, a factor (t - x_j)+ is written as (x_j - t)+ - u_j +
        (t - m_j), so that an elimination sees that a pair of hinges spans
        x_j (for the projections only; LA-4 keeps each hinge as its atom)."""
        out = {(): Fraction(1)}
        for j in np.flatnonzero(row):
            code = int(row[j])
            if rewrite and code == -1:
                plus = self.atom(j, 1, cut[j])
                parts = {plus: Fraction(1), self.atom(j, 2, 0.0): Fraction(-1)}
                parts[None] = plus[2]
            else:
                parts = {self.atom(j, code, cut[j]): Fraction(1)}
            if code == 2 and self.m[j] != 0.0:
                parts[None] = Fraction(self.m[j])  # x_j = u_j + m_j
            new: dict = {}
            for mono, f in out.items():
                for a, g in parts.items():
                    key = mono if a is None else tuple(sorted((*mono, a)))
                    new[key] = new.get(key, 0) + f * g
            out = new
        return {mono: f for mono, f in out.items() if f != 0}

    def column(self, mono) -> np.ndarray:
        """The monomial on the data, its factors multiplied in key order."""
        if mono not in self._columns:
            column = np.ones(self.X.shape[0])
            for s, kind, knot in mono:
                if kind == U_ATOM:
                    column = column * self.U[:, s]
                else:
                    t = float(knot + Fraction(self.m[s]))
                    column = column * factor(
                        1 if kind == PLUS_ATOM else -1, self.X[:, s], t
                    )
            self._columns[mono] = column
        return self._columns[mono]

    def evaluate(self, row, absolute: bool = False) -> np.ndarray:
        """sum_q c_q column(q) of a row {monomial: c_q}; with ``absolute``,
        sum_q |c_q| |column(q)|, the scale of its rounding."""
        out = np.zeros(self.X.shape[0])
        for mono in sorted(row, key=monomial_order):
            c = float(row[mono])
            col = self.column(mono)
            out = out + (abs(c) * np.abs(col) if absolute else c * col)
        return out


class Echelon:
    """The reduced echelon form of a set of expansions, in LA-4's order:
    {pivot monomial: row}, each row with coefficient 1 at its pivot and 0 at
    the other pivots."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def copy(self) -> Echelon:
        return Echelon(self.rows)

    def reduce(self, e) -> dict:
        """e minus its components along the rows, exactly."""
        v = dict(e)
        for p, r in self.rows.items():
            f = v.get(p, 0)
            if f != 0:
                v = _combine(v, -f, r)
        return v

    def add(self, e):
        """Adds e; returns its new pivot, or None when e is in the span."""
        v = self.reduce(e)
        if not v:
            return None
        p = min(v, key=monomial_order)
        v = {q: f / v[p] for q, f in v.items()}
        for q, r in self.rows.items():
            if r.get(p, 0) != 0:
                self.rows[q] = _combine(r, -r[p], v)
        self.rows[p] = v
        return p

    def ordered(self) -> list:
        """The rows, the intercept's (degree 0) first."""
        return [self.rows[p] for p in sorted(self.rows, key=lambda p: (len(p), p))]


def _combine(a: dict, f, b: dict) -> dict:
    """a + f b, without zero entries."""
    out = dict(a)
    for q, g in b.items():
        value = out.get(q, 0) + f * g
        if value != 0:
            out[q] = value
        else:
            out.pop(q, None)
    return out


def _dyadic(v) -> tuple[list, int]:
    """Integers a_i and e with v_i = a_i 2^-e exactly, for floats v_i."""
    ratios = [float(x).as_integer_ratio() for x in v]
    e = max(d.bit_length() for _, d in ratios) - 1
    return [a << (e - d.bit_length() + 1) for a, d in ratios], e


def _exact_rest_sq(atoms: Atoms, new: dict, others: list, w) -> Fraction:
    """The squared w-norm of the part of the evaluated row ``new`` orthogonal
    to the evaluated rows ``others``, in exact arithmetic on the float64
    monomial columns and weights: a Gram matrix of the monomials, then the
    Schur complement. Cost: O(n K^2) integer products for K monomials and
    O(k^3) operations on fractions for k rows."""
    monos = sorted(set(new).union(*others), key=monomial_order)
    W, ew = _dyadic(w)
    ints = {q: _dyadic(atoms.column(q)) for q in monos}
    G: dict = {}
    for i, a in enumerate(monos):
        A, ea = ints[a]
        for b in monos[i:]:
            Bv, eb = ints[b]
            total = sum(x * y * z for x, y, z in zip(W, A, Bv, strict=True))
            G[a, b] = G[b, a] = Fraction(total, 1 << (ew + ea + eb))
    vectors = [*others, new]

    def inner(x, y):
        return sum(f * g * G[a, b] for a, f in x.items() for b, g in y.items())

    k = len(vectors)
    H = [[inner(x, y) for y in vectors] for x in vectors]
    for i in range(k - 1):
        pivot = H[i][i]
        if pivot == 0:
            continue  # a positive semidefinite row of zeros
        for r in range(i + 1, k):
            f = H[r][i] / pivot
            if f != 0:
                for c in range(i + 1, k):
                    H[r][c] -= f * H[i][c]
    return H[k - 1][k - 1]


def la4_dependent(atoms: Atoms, S: Echelon, e: dict, w) -> bool:
    """LA-4's test of a new column, the expansion e, against the kept set S.

    Dependent when e lies in the span of S's expansions. Otherwise the
    reduced echelon form of S with e has one new pivot, whose evaluated
    reduced row is the new part; the column is dependent when the new part
    is 0 at every case, or when the w-norm of its part orthogonal to the
    other evaluated reduced rows is less than 1e-7 times the w-norm of the
    evaluated pivot monomial (of the new part, when the monomial is 0 at
    every case). With a shifted covariate the orthogonal part is computed
    in exact arithmetic (_exact_rest_sq), since the other rows can carry
    large factors m_j; without one, every row is a monomial, and this is the
    plain test of LA-4 on the columns, in float64.
    """
    v = S.reduce(e)
    if not v:
        return True
    p = min(v, key=monomial_order)
    new = {q: f / v[p] for q, f in v.items()}
    others = [r if r.get(p, 0) == 0 else _combine(r, -r[p], new) for r in S.ordered()]
    sw = np.sqrt(np.asarray(w, dtype=np.float64))
    part = atoms.evaluate(new)
    if not np.any(part):
        return True
    size = float(np.linalg.norm(sw * atoms.column(p)))
    if size == 0.0:
        size = float(np.linalg.norm(sw * part))
    threshold = DEPENDENT_TOL * size
    if not atoms.plain:
        return _exact_rest_sq(atoms, new, others, w) < Fraction(threshold) ** 2
    b = sw * part
    if not others:
        return float(np.linalg.norm(b)) < threshold
    A = np.column_stack([atoms.evaluate(o) for o in others]) * sw[:, None]
    return float(np.linalg.norm(_residual(_orthonormal_basis(A), b))) < threshold


def la4_kept(X, dirs, cuts, w, atoms: Atoms | None = None) -> np.ndarray:
    """The terms that FWD-11 keeps: in order, each term that LA-4 does not
    find dependent on the kept earlier ones (a boolean mask)."""
    atoms = Atoms(X) if atoms is None else atoms
    S, keep = Echelon(), []
    for row, cut in zip(dirs, cuts, strict=True):
        e = atoms.expansion(row, cut)
        keep.append(not la4_dependent(atoms, S, e, w))
        if keep[-1]:
            S.add(e)
    return np.array(keep, dtype=bool)


class Basis:
    """Columns for the projections that span what a list of expansions spans:
    Gaussian elimination in exact arithmetic, each kept row scaled to 1 at its
    pivot, the monomial with the largest share of its column (the first in
    LA-4's order among equal shares). Its rows, evaluated, have coefficients
    of size about 1 on monomials whose values are of the size of the range,
    so no digits go to a mean [LA-5]. The reduced echelon form of LA-4 decides
    dependence; its rows can carry large factors m_j on monomials that are not
    pivots, so they are not used for projections. The pivots depend on the
    order of the expansions; the span does not."""

    def __init__(self, atoms: Atoms, expansions=()):
        self.atoms = atoms
        self.pivots: list = []  # (pivot, row), in order
        for e in expansions:
            self.add(e)

    def reduce(self, e, record=None) -> dict:
        """e minus its components along the rows in order; with a list
        ``record``, the multiple of each row is appended to it."""
        v = dict(e)
        for p, r in self.pivots:
            f = v.get(p, 0)
            if record is not None:
                record.append(f)
            if f != 0:
                v = _combine(v, -f, r)
        return v

    def add(self, e) -> bool:
        v = self.reduce(e)
        if not v:
            return False
        size = {q: abs(v[q]) * float(np.linalg.norm(self.atoms.column(q))) for q in v}
        p = max(sorted(v, key=monomial_order), key=lambda q: size[q])
        self.pivots.append((p, {q: f / v[p] for q, f in v.items()}))
        return True

    def columns(self) -> np.ndarray:
        return np.column_stack([self.atoms.evaluate(r) for _, r in self.pivots])


class Conditioned:
    """The kept terms (rows ``dirs``, ``cuts``, columns ``B``) after LA-4's
    exact shift. ``echelon`` is their reduced echelon form, for LA-4's test
    of a new term (``la4_dependent``); ``columns`` span what the terms span
    (``Basis``) [LA-1]; ``linear_column`` is the new part of a linear
    candidate b x; ``span_columns`` and ``coef`` serve the pruning pass and
    the final fit. Without a shifted covariate every term is one monomial,
    and the columns are those of B. Cost: with M terms of degree d,
    O(M^2 2^d) operations on fractions and O(n M 2^d) on floats."""

    def __init__(self, X, dirs, cuts, B=None, atoms: Atoms | None = None):
        self.atoms = Atoms(X) if atoms is None else atoms
        self._terms = [
            self.atoms.expansion(r, c) for r, c in zip(dirs, cuts, strict=True)
        ]
        self._rewritten = [
            self.atoms.expansion(r, c, rewrite=True)
            for r, c in zip(dirs, cuts, strict=True)
        ]
        self.echelon = Echelon()
        for e in self._terms:
            self.echelon.add(e)
        if B is None:
            B = basis_matrix(self.atoms.X, np.array(dirs), np.array(cuts))
        self._B = np.asarray(B, dtype=np.float64)
        self._basis = None
        if self.atoms.plain:
            self.columns = self._B
        else:
            self._basis = Basis(self.atoms, self._rewritten)
            self.columns = self._basis.columns()

    def linear_expansion(self, parent_row, parent_cut, j, rewrite=False) -> dict:
        """The expansion of b x_j, for the parent's rows of dirs and cuts."""
        row = np.array(parent_row).copy()
        row[j] = 2
        cut = np.array(parent_cut, dtype=np.float64).copy()
        cut[j] = 0.0
        return self.atoms.expansion(row, cut, rewrite)

    def linear_column(self, parent_row, parent_cut, j, b) -> np.ndarray:
        """b x_j minus a combination of the kept terms, so that it has the
        same A_w of LA-7 and spans with them what b x_j spans: b (x_j - min
        x_j) without a shifted covariate, else b x_j reduced exactly by the
        rows of ``columns`` and evaluated (0 when it is in their span)."""
        if self.atoms.plain:
            x = self.atoms.X[:, j]
            return b * (x - x.min())
        e = self.linear_expansion(parent_row, parent_cut, j, rewrite=True)
        v = self._basis.reduce(e)
        return self.atoms.evaluate(v)

    def span_columns(self, rows) -> np.ndarray:
        """Columns that span what the terms in ``rows`` span (pruning indices,
        increasing, the intercept first)."""
        rows = list(rows)
        if self.atoms.plain:
            return self._B[:, rows]
        return Basis(self.atoms, [self._rewritten[k] for k in rows]).columns()

    def coef(self, rows, Y, w) -> tuple[np.ndarray, np.ndarray]:
        """The least-squares coefficients of Y on the terms in ``rows``
        [PRUNE-8], and the mask of the terms that LA-4 keeps (in order, each
        against the kept earlier ones); a dependent term gets 0. Without a
        shifted covariate this is ``lstsq_coef`` on the kept columns of B.
        Otherwise the fit is made on the columns R of the kept terms' Basis,
        gamma, and term k = sum_i T[k, i] R_i exactly, so the coefficients
        solve T' coef = gamma, in exact arithmetic, then round to float64."""
        rows = list(rows)
        S, keep = Echelon(), []
        for k in rows:
            keep.append(not la4_dependent(self.atoms, S, self._terms[k], w))
            if keep[-1]:
                S.add(self._terms[k])
        keep = np.array(keep, dtype=bool)
        Y2 = np.asarray(Y, dtype=np.float64).reshape(len(w), -1)
        coef = np.zeros((len(rows), Y2.shape[1]))
        kept = [k for k, ok in zip(rows, keep, strict=True) if ok]
        if self.atoms.plain:
            coef[keep] = lstsq_coef(self._B[:, kept], Y2, w)
            return coef, keep
        basis = Basis(self.atoms, [self._rewritten[k] for k in kept])
        gamma = lstsq_coef(basis.columns(), Y2, w)
        T = []
        for k in kept:
            T.append([])
            basis.reduce(self._rewritten[k], record=T[-1])
        Tt = [list(r) for r in zip(*T, strict=True)]
        for col in range(Y2.shape[1]):
            g = [Fraction(float(x)) for x in gamma[:, col]]
            coef[np.flatnonzero(keep), col] = [float(x) for x in _solve_exact(Tt, g)]
        return coef, keep


def _solve_exact(A: list, b: list) -> list:
    """The solution of the nonsingular system A x = b, in exact arithmetic."""
    k = len(b)
    M = [[*A[i], b[i]] for i in range(k)]
    for c in range(k):
        r = next((i for i in range(c, k) if M[i][c] != 0), None)
        if r is None:
            raise ValueError("the exact system for the coefficients is singular")
        M[c], M[r] = M[r], M[c]
        for i in range(k):
            if i != c and M[i][c] != 0:
                f = M[i][c] / M[c][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c], strict=True)]
    return [M[i][k] / M[i][i] for i in range(k)]


def _parent_candidates(
    k,
    parent_row,
    X,
    Y,
    w,
    B,
    P_B,
    e_B,
    sigma2,
    N,
    tau_N,
    params,
    tau,
    conditioned=None,
    parent_cut=None,
    met=None,
    rss_s=None,
):
    """The candidates of parent term k (column k of B), in the order of FWD-5,
    and whether some covariate could be searched for it [FWD-2, FWD-3].

    ``met`` maps the candidates that earlier searches of the step met, by
    their kind and added rows (``merge_key``), to the Candidate of the first
    occurrence, or to None when LA-3 rejected it there. A candidate met
    before is that Candidate, with its parent, its RSS and so its legality,
    or no candidate at all when it was rejected [FWD-12]; the dict is
    updated with the new ones. ``rss_s`` is RSS_s, the RSS of a candidate
    whose new columns all count 0 (by default the RSS of P_B).

    For each covariate that the parent lacks, in increasing order: the kind
    of the search by the pymars rule of LA-7; in a pair search the linear
    candidate, with RSS(B + {b x}); then the knots of KNOT-6 from the
    largest down, without the ones that LA-3 rejects, each with RSS(G + {h})
    [LA-2], where G is B, and B with b x in a pair search. Each hinge column
    is built and projected on G explicitly; its RSS is that of the residual.
    The column b x is formed as b (x - min x), which spans the same with B.
    When ``conditioned`` is the Conditioned of the terms of B, P_B must be
    the projector of its columns, which G uses in place of B, and b x is its
    linear_column (``parent_cut`` is the cut row of the parent). The hinge
    column h is used as it is: when a large mean would cost digits of its
    part that is new to G, that part is a small fraction of h, and LA-3
    rejects it. With ``conditioned``, LA-4 tests b x against the kept terms
    and each hinge that LA-3 accepts against G; a dependent new column
    counts as 0 in the RSS [LA-2], so a linear candidate whose b x is
    dependent has the RSS of B and is not legal. A hinge's test reuses its
    orthogonal part from LA-3 when that is clearly above 1e-7 of the w-norm
    of its pivot monomial (la4_dependent decides the others). Cost: up to n
    knots per covariate, each projected on up to M + 1 columns, so O(p n^2
    M) time, and O(n^2) memory for the hinge matrix.
    """
    p = X.shape[1]
    b = B[:, k]
    own = [j for j in range(p) if parent_row[j] != 0]
    covariates = [j for j in range(p) if parent_row[j] == 0]
    if not covariates:
        return [], False
    active = b > 0  # [KNOT-1]
    spans = None
    if active.any():
        Nb = snap(weight_sum(w[active]), tau_N)
        spans = search_spans(
            p,
            len(own),
            Nb,
            N,
            tau_N,
            params.minspan,
            params.endspan,
            params.adjust_endspan,
        )
    B_proj = B if conditioned is None else conditioned.columns
    met = {} if met is None else met
    rss_s = float(np.sum(e_B**2)) if rss_s is None else rss_s
    cut_row = np.zeros(p) if parent_cut is None else parent_cut
    out = []

    def first(candidate, j, kind, knot):
        # the first occurrence decides [FWD-12]
        key = merge_key(parent_row, cut_row, j, kind, knot, X, params.auto_linpreds)
        if key not in met:
            met[key] = candidate
        if met[key] is not None:
            out.append(met[key])

    for j in covariates:
        x = X[:, j]
        # b x enters only through its span with B, which holds b, so b (x - m)
        # with m the smallest x serves in its place: the same A_w, the same
        # linear candidate and the same G in exact arithmetic, and x - m is
        # exact when x has a large mean, so no digits go to the mean [LA-5]
        bx_dependent, S = False, None
        if conditioned is None:
            bx = b * (x - x.min())
        else:
            bx = conditioned.linear_column(parent_row, parent_cut, j, b)
            e_bx = conditioned.linear_expansion(parent_row, cut_row, j)
            S = conditioned.echelon.copy()
            bx_dependent = la4_dependent(conditioned.atoms, S, e_bx, w)  # [LA-2]
        V = [*own, j]
        pair = all(sigma2[v] > 0 for v in V) and (
            P_B.rss(bx) >= 0.01 * math.prod(sigma2[v] for v in V)
        )
        if pair and not bx_dependent:
            P_G = Projector(np.column_stack([B_proj, bx]), w)
            e_G = P_G.residual(Y)
            if S is not None:
                S.add(e_bx)
        else:
            P_G, e_G = P_B, e_B
        if pair:
            value = rss_s if bx_dependent else float(np.sum(e_G**2))
            first(Candidate(k, j, LINEAR, math.nan, value), j, LINEAR, math.nan)
        if spans is None:
            continue
        knots = knot_scan(x, active, w, *spans, N, tau_N, distinct=True)
        if not knots:
            continue
        t = np.array(knots)
        H = b[:, None] * np.maximum(x[:, None] - t[None, :], 0.0)
        H_perp = P_G.residual(H)
        size = np.sum(H_perp**2, axis=0)
        spread = np.sum(P_G.scaled(H) ** 2, axis=0)
        constant = np.all(H[0] == H, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            accept = ~constant & (size / spread >= tau)  # [LA-3]
        kind = PAIR if pair else SINGLE
        for i in range(len(t)):
            c = None  # a knot that LA-3 rejects is no candidate
            if accept[i]:
                # a dependent hinge counts 0: the RSS of G
                value = rss_s if P_G is P_B else float(np.sum(e_G**2))
                if S is None or not _hinge_dependent(
                    conditioned,
                    S,
                    parent_row,
                    cut_row,
                    j,
                    float(t[i]),
                    H[:, i],
                    size[i],
                    w,
                ):
                    h = H_perp[:, i]
                    residual = e_G - np.outer(h, (h @ e_G) / size[i])
                    value = float(np.sum(residual**2))
                c = Candidate(k, j, kind, float(t[i]), value)
            first(c, j, kind, float(t[i]))
    return out, True


def _hinge_dependent(conditioned, S, parent_row, parent_cut, j, t, h, size, w):
    """LA-4's test of the hinge b (x_j - t)+ against S [LA-2]. ``size`` is the
    squared w-norm of its part orthogonal to G, from LA-3; when it is above
    1e-7 of the w-norm of the pivot monomial by more than a bound on its
    rounding, the hinge counts, else la4_dependent decides."""
    atoms = conditioned.atoms
    row = np.array(parent_row).copy()
    row[j] = 1
    cut = np.array(parent_cut, dtype=np.float64).copy()
    cut[j] = t
    sw = np.sqrt(w)
    e = atoms.expansion(row, cut)
    if atoms.plain:
        pivot = float(np.linalg.norm(sw * h))
    else:
        v = S.reduce(e)
        if not v:
            return True
        pivot = float(np.linalg.norm(sw * atoms.column(min(v, key=monomial_order))))
        if pivot == 0.0:
            return la4_dependent(atoms, S, e, w)  # the new part's norm decides
    bound = 1e3 * EPS * float(np.linalg.norm(sw * h))
    if math.sqrt(size) > DEPENDENT_TOL * pivot + bound:
        return False
    return la4_dependent(atoms, S, e, w)


def merge_key(parent_row, parent_cut, j, kind, knot, X, auto_linpreds) -> tuple:
    """The kind of a candidate and the rows of dirs and cuts that it adds, in
    order [FWD-6]: candidates with equal keys are one candidate [FWD-12]."""
    if kind == PAIR:
        codes = [(1, knot), (-1, knot)]
    elif kind == SINGLE:
        codes = [(1, knot)]
    elif auto_linpreds:
        codes = [(2, 0.0)]
    else:
        codes = [(1, float(np.min(X[:, j])))]
    rows = []
    for code, cut in codes:
        row, cuts = list(np.asarray(parent_row).tolist()), list(map(float, parent_cut))
        row[j], cuts[j] = code, cut + 0.0
        rows.append((tuple(row), tuple(cuts)))
    return (kind, tuple(rows))


def _first_best(candidates, rss_s):
    """The candidate with the largest reduction after the exact-fit band;
    equal reductions go to the one found first [FWD-5]."""
    best = None
    for c in candidates:
        if best is None or reduction(c, rss_s) > reduction(best, rss_s):
            best = c
    return best


def _new_terms(c, X, columns, dirs, cuts, auto_linpreds):
    """The terms that candidate c adds, in order, as (dirs row, cuts row,
    column) [FWD-6, TERM-2, TERM-6]."""
    b, x = columns[c.parent], X[:, c.variable]

    def term(code, knot):
        row, cut = dirs[c.parent].copy(), cuts[c.parent].copy()
        row[c.variable] = code
        cut[c.variable] = 0.0 if code == 2 else knot
        return row, cut, b * factor(code, x, knot)

    if c.kind == PAIR:
        return [term(1, c.knot), term(-1, c.knot)]
    if c.kind == SINGLE:
        return [term(1, c.knot)]
    if auto_linpreds:
        return [term(2, 0.0)]
    return [term(1, float(x.min()))]


def _candidate_log(log) -> dict:
    """The CandidateLog of CORE-3 from (chosen, second, RSS_s) triples, one per
    step: second_rss is RSS_s minus the second's reduction after the band of
    FWD-5, and best_rss the computed RSS of the new basis."""

    def field(name, none, dtype):
        return np.array(
            [none if s is None else getattr(s, name) for _, s, _ in log], dtype
        )

    return {
        "best_rss": np.array([c.rss for c, _, _ in log], dtype=np.float64),
        "second_rss": np.array(
            [math.inf if s is None else r - reduction(s, r) for _, s, r in log],
            dtype=np.float64,
        ),
        "second_parent": field("parent", -1, np.int64),
        "second_variable": field("variable", -1, np.int64),
        "second_knot": np.array(
            [math.nan if s is None or s.kind == LINEAR else s.knot for _, s, _ in log],
            dtype=np.float64,
        ),
        "second_kind": field("kind", NO_KIND, np.int8),
    }


def forward_pass(
    X, Y, w, params=None, *, N, tau_N, tss, record_candidates=False, trace=None
):
    """The forward pass of a fit that is not degenerate [FWD-1 to FWD-11], with
    its stopping rules [STOP-1 to STOP-6] and the queue [FAST-1 to FAST-6].

    X is n x p, Y n x K and w the positive weights; N and tau_N are those of
    W-4, and tss is the RSS of the intercept alone, in the units of Y.
    Returns (record, B): the fields of ForwardRecord [CORE-3] as a dict, and
    the n x M_a matrix of the columns of the terms. When ``trace`` is a
    list, each step appends a dict with its queue table, the visited
    entries, the searched parents (forward indices), the queue entries after
    the search, and the chosen candidate. Cost: a step searches up to M
    parents, so it takes O(p n^2 M^2) time at worst, and O(n (p + M) + n^2)
    memory.
    """
    params = as_params(params)
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    Y = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    w = np.asarray(w, dtype=np.float64)
    max_terms = params.resolved_max_terms(p)
    penalty = params.resolved_penalty()
    sigma2 = covariate_variances(X, w, N)
    columns, dirs, cuts = [np.ones(n)], [np.zeros(p, dtype=np.int8)], [np.zeros(p)]
    parent, step, rss_path = [-1], [0], [float(tss)]
    slots = {1: 0}  # slot -> forward index of its term [FWD-9]
    entries = [[math.inf, 0]]  # entry e at index e - 1: [lambda_e, kappa_e]
    log = []
    atoms, kept = Atoms(X), [0]  # the terms that LA-4 keeps [LA-1, FWD-11]
    s = 0
    while True:
        kappa = 2 * (s + 1)
        if 1 + 2 * (s + 1) > max_terms:  # [STOP-1]
            termination = NO_ROOM if s == 0 else TERM_LIMIT
            break
        B = np.column_stack(columns)
        conditioned = Conditioned(
            X, [dirs[i] for i in kept], [cuts[i] for i in kept], B[:, kept], atoms
        )
        P_B = Projector(conditioned.columns, w)
        e_B = P_B.residual(Y)
        rss_s = rss_path[-1]
        limit = max_legal(rss_path)
        tau = collinearity_tol(s)
        table = queue_table(entries, s, params.fast_beta)
        visited = window(table, params.fast_k)
        found, searched, met, seen = [], [], {}, set()
        for e in visited:
            k = slots.get(e)
            if k is None or term_degree(dirs[k]) >= params.max_degree:
                continue  # skipped; the entry keeps its values [FAST-4]
            cands, searchable = _parent_candidates(
                k,
                dirs[k],
                X,
                Y,
                w,
                B,
                P_B,
                e_B,
                sigma2,
                N,
                tau_N,
                params,
                tau,
                conditioned,
                cuts[k],
                met,
                rss_s,
            )
            entries[e - 1] = [queue_value(cands, searchable, rss_s, limit), kappa]
            found.extend(c for c in cands if id(c) not in seen)
            seen.update(id(c) for c in cands)
            searched.append(k)
        legal = [c for c in found if is_legal(c.kind, reduction(c, rss_s), limit)]
        chosen = _first_best(legal, rss_s)
        if trace is not None:
            trace.append(
                {
                    "step": s + 1,
                    "table": table,
                    "visited": visited,
                    "searched": searched,
                    "entries": [tuple(v) for v in entries],
                    "chosen": chosen,
                }
            )
        # The stopping rules, in the order STOP-3, STOP-4, STOP-2, STOP-5. A
        # step without a legal candidate counts as one term and no gain.
        M = len(columns)
        if chosen is None:
            rss_new, M_new = rss_s, M + 1
        else:
            rss_new, M_new = chosen.rss, M + (2 if chosen.kind == PAIR else 1)
        rsq_gain = rsq(rss_new, tss) - rsq(rss_s, tss)
        grsq_new = grsq(rss_new, M_new, tss, penalty, N, tau_N)
        if params.thresh > 0 and grsq_new < -10:  # [STOP-3]
            termination = GRSQ_NEG_INF if grsq_new < -1000 else GRSQ_LOW
            break
        if rsq_gain < params.thresh:  # [STOP-4]
            termination = RSQ_CHANGE_SMALL
            break
        if chosen is None:  # [STOP-2]
            termination = NO_GAIN
            break
        new = _new_terms(chosen, X, columns, dirs, cuts, params.auto_linpreds)
        S = conditioned.echelon.copy()
        for i, (row, cut, column) in enumerate(new):  # [FWD-9, FAST-1, TERM-6]
            e = atoms.expansion(row, cut)
            if not la4_dependent(atoms, S, e, w):  # [FWD-11]
                kept.append(len(columns))
                S.add(e)
            slots[kappa + i] = len(columns)
            columns.append(column)
            dirs.append(row)
            cuts.append(cut)
            parent.append(chosen.parent)
            step.append(s + 1)
            entries.append([math.inf, kappa])
        s += 1
        rss_path.append(chosen.rss)
        if record_candidates:
            others = [c for c in legal if c is not chosen]  # merged already [FWD-12]
            log.append((chosen, _first_best(others, rss_s), rss_s))
        floor = 1e-10 * tss / (N - 1)
        if rsq(chosen.rss, tss) >= 1 - params.thresh or chosen.rss < floor:
            termination = RSQ_HIGH  # [STOP-5]
            break
    B = np.column_stack(columns)
    dropped = np.ones(len(columns), dtype=bool)  # [FWD-11], as tested in the steps
    dropped[kept] = False
    record = {
        "dirs": np.array(dirs, dtype=np.int8).reshape(-1, p),
        "cuts": np.array(cuts, dtype=np.float64).reshape(-1, p),
        "kept": np.flatnonzero(~dropped).astype(np.int64),
        "dropped": np.flatnonzero(dropped).astype(np.int64),
        "parent": np.array(parent, dtype=np.int64),
        "step": np.array(step, dtype=np.int64),
        "rss": np.array(rss_path, dtype=np.float64),
        "termination": int(termination),
        "candidates": _candidate_log(log) if record_candidates else None,
    }
    return record, B


# ---------------------------------------------------------------------------
# The fit [CORE-1, CORE-3, CORE-5, W-3, EDGE-1, EDGE-6, GCV-7]


def _unscale(value, power: int):
    """``value`` times 2**power [EDGE-6]: exact while the result is a normal
    float64, rounded in the subnormal range, 0 below it and inf above the
    largest float64, as the arithmetic without the scaling would give."""
    with np.errstate(over="ignore", under="ignore"):
        out = np.ldexp(np.asarray(value, dtype=np.float64), power)
    return float(out) if out.ndim == 0 else out


def x_scale_powers(X) -> np.ndarray:
    """j_v of EDGE-7 for each column v: the largest |x_iv| times 2**j_v lies
    in [1, 2); 0 for a column of zeros."""
    top = np.max(np.abs(X), axis=0) if X.shape[0] else np.zeros(X.shape[1])
    return np.array(
        [0 if t == 0.0 else 1 - math.frexp(float(t))[1] for t in top], dtype=np.int64
    )


def _coef_back(coef, dirs, jx, jy: int) -> np.ndarray:
    """The coefficients of the fit on the scaled X and Y on the original
    scales: row k times 2 to the sum of j_v over the covariates of term k,
    and divided by 2**jy, in one ldexp [EDGE-7, EDGE-6]. ValueError when a
    result is not finite, or when a nonzero coefficient becomes 0 or a
    subnormal, whatever scale causes it [EDGE-7]."""
    power = (np.asarray(dirs) != 0) @ jx - jy
    with np.errstate(over="ignore", under="ignore"):
        out = np.ldexp(coef, power[:, None])
    lost = (coef != 0) & (np.abs(out) < np.finfo(np.float64).tiny)
    if not np.all(np.isfinite(out)) or lost.any():
        raise ValueError(
            "the scale of X is out of range (or that of y): a coefficient "
            "leaves the normal range of float64 when it is scaled back"
        )
    return out


def y_scale_power(Y) -> int:
    """j of EDGE-6: D 2**j lies in [1, 2), where D is the largest |Y_ik|;
    0 when D = 0."""
    D = float(np.max(np.abs(Y)))
    return 0 if D == 0.0 else 1 - math.frexp(D)[1]


def case_weights(w, n: int) -> np.ndarray:
    """The weights as an (n,) float64 array [W-5, W-6]: None is 1 for every
    row; a Python or numpy int or float scalar is given to every row; a bool
    (Python or numpy) and a 0-d array raise ValueError."""
    if w is None:
        return np.ones(n)
    if isinstance(w, (bool, np.bool_)):
        raise ValueError("a bool is not a weight (W-6)")
    if isinstance(w, np.ndarray) and w.ndim == 0:
        raise ValueError("a 0-d array is not a weight; give a scalar or 1-D array")
    if isinstance(w, (int, float, np.integer, np.floating)):
        return np.full(n, float(w))
    return np.asarray(w, dtype=np.float64)


def total_weight(w) -> float:
    """N = math.fsum of the weights; ValueError when the sum overflows or is
    2^52 or more, the bound of W-4 [W-6]."""
    try:
        N = weight_sum(w)
    except OverflowError:
        N = math.inf
    if not N < 2.0**52:
        raise ValueError(f"the total weight is out of range: {N!r}, not below 2^52")
    return N


def fit_mars(X, Y, w=None, params=None, *, record_candidates=False) -> dict:
    """The fit of CORE-1, returned as the dict of CORE-5.

    X is (n, p); Y is (n, K), or (n,) for K = 1 [RESP-2]; w is (n,) with a
    positive sum, a scalar for every row, or None for w_i = 1 exactly [W-5,
    W-6]; a total weight that W-6 rejects raises ValueError before EDGE-6.
    Otherwise the input is valid, as the estimators check it first [CORE-1].
    Rows with zero weight are dropped before anything else [W-3]. Y is
    multiplied by 2**j before any sum, and every value on the scale of Y is
    scaled back [EDGE-6]. The
    keys follow CORE-3; ``forward``, ``pruning`` and ``candidates`` are
    nested dicts and ``termination`` is the integer code [CORE-5].
    """
    params = as_params(params)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y.reshape(Y.shape[0], -1)
    w = case_weights(w, X.shape[0])
    rows = w > 0
    X, Y, w = X[rows], Y[rows], w[rows]
    n, p = X.shape
    N0 = total_weight(w)
    tau_N = weight_tol(N0)
    N = snap(N0, tau_N)
    j = y_scale_power(Y)
    Ys = np.ldexp(Y, j)
    constant = bool(np.all(Y[0] == Y))  # every response is constant
    tss = 0.0  # by definition when every response is constant [GCV-7]
    if not constant:
        # the scaled Y is below 2 in absolute value, so only the weights can
        # make the TSS overflow; that is the error below, not a warning
        with np.errstate(over="ignore", invalid="ignore"):
            tss = rss(np.ones((n, 1)), Ys, w)
        if not (math.isfinite(tss) and tss >= np.finfo(np.float64).tiny):
            raise ValueError(
                "the scale of y or of the weights is out of range: the weighted "
                f"total sum of squares of the scaled y is {tss!r}"
            )
    common = {
        "n_eff": N,
        "max_terms": params.resolved_max_terms(p),
        "penalty": params.resolved_penalty(),
    }
    if N <= 1 or constant:
        return _degenerate_fit(Y, Ys, w, j, constant, tss, p, record_candidates, common)
    jx = x_scale_powers(X)
    X = np.ldexp(X, jx)  # the fit runs on the scaled X [EDGE-7]
    forward, B = forward_pass(
        X,
        Ys,
        w,
        params,
        N=N,
        tau_N=tau_N,
        tss=tss,
        record_candidates=record_candidates,
    )
    kept = forward["kept"]
    conditioned = Conditioned(
        X, forward["dirs"][kept], forward["cuts"][kept], B[:, kept]
    )
    pruned = prune(
        B[:, kept],
        Ys,
        w,
        penalty=common["penalty"],
        N=N,
        tau_N=tau_N,
        pmethod=params.pmethod,
        nprune=params.nprune,
        columns_of=conditioned.span_columns,
        coef_of=conditioned.coef,
    )
    selected = kept[pruned["selected"]]
    # the knots back on the original scale, exactly [EDGE-7]
    cuts = np.ldexp(forward["cuts"], -jx)
    forward = {**forward, "cuts": cuts, "rss": _unscale(forward["rss"], -2 * j)}
    if forward["candidates"] is not None:
        log = forward["candidates"]
        forward["candidates"] = {
            **log,
            "best_rss": _unscale(log["best_rss"], -2 * j),
            "second_rss": _unscale(log["second_rss"], -2 * j),
            "second_knot": np.ldexp(
                log["second_knot"], -jx[np.maximum(log["second_variable"], 0)]
            ),
        }
    coef = _coef_back(pruned["coef"], forward["dirs"][selected], jx, j)
    return {
        "dirs": forward["dirs"][selected],
        "cuts": cuts[selected],
        "coef": coef,
        "selected": selected,
        "rss": _unscale(pruned["rss"], -2 * j),
        "gcv": _unscale(pruned["gcv"], -2 * j),
        "rsq": pruned["rsq"],
        "grsq": pruned["grsq"],
        **common,
        "forward": forward,
        "pruning": {
            "removed": pruned["removed"],
            "rss_per_size": _unscale(pruned["rss_per_size"], -2 * j),
            "gcv_per_size": _unscale(pruned["gcv_per_size"], -2 * j),
            "subsets": pruned["subsets"],
            "selected_size": pruned["selected_size"],
        },
    }


def _degenerate_fit(Y, Ys, w, j, constant, tss, p, record_candidates, common) -> dict:
    """The intercept alone, when N <= 1 or every response is constant [EDGE-1,
    GCV-7]: its coefficient is the weighted mean of each response, gcv is
    +inf, rsq and grsq are 0, and TSS and rss are 0 exactly when every
    response is constant, else computed. The rss of the intercept model is
    its TSS, computed centered [LA-5]."""
    n = Ys.shape[0]
    if constant:
        coef = Y[:1].copy()  # the weighted mean of a constant is its value
        final_rss = 0.0
    else:
        coef = _unscale(lstsq_coef(np.ones((n, 1)), Ys, w), -j)
        final_rss = tss  # the RSS of the intercept model
    tss_row = np.array([_unscale(tss, -2 * j)])
    return {
        "dirs": np.zeros((1, p), dtype=np.int8),
        "cuts": np.zeros((1, p)),
        "coef": coef,
        "selected": np.array([0], dtype=np.int64),
        "rss": _unscale(final_rss, -2 * j),
        "gcv": math.inf,
        "rsq": 0.0,
        "grsq": 0.0,
        **common,
        "forward": {
            "dirs": np.zeros((1, p), dtype=np.int8),
            "cuts": np.zeros((1, p)),
            "kept": np.array([0], dtype=np.int64),
            "dropped": np.zeros(0, dtype=np.int64),
            "parent": np.array([-1], dtype=np.int64),
            "step": np.array([0], dtype=np.int64),
            "rss": tss_row,
            "termination": DEGENERATE,
            "candidates": _candidate_log([]) if record_candidates else None,
        },
        "pruning": {
            "removed": np.zeros(0, dtype=np.int64),
            "rss_per_size": tss_row.copy(),
            "gcv_per_size": np.array([math.inf]),
            "subsets": np.ones((1, 1), dtype=bool),
            "selected_size": 1,
        },
    }
