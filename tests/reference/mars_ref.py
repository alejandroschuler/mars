"""Reference implementation of the pymars 2.0 fitting algorithm (the test oracle).

This module implements ``docs/algorithm.md`` (spec v1) literally, so that the
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
max_degree <= 3 [CORE-7]. Sums of squares are formed from the values as
they are, so values whose squares leave the normal range of float64 (above
about 1e150 or below about 1e-150 in absolute value, in a covariate or in a
product of factors) are outside its range; Y is scaled first [EDGE-6].
Bracketed IDs such as [GCV-2] cite the rules of the spec. Indices are
0-based, as in the spec. No function changes its inputs.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import numbers
from collections.abc import Mapping

import numpy as np
import scipy.linalg

ALPHA = 0.05  # the probability in Friedman's span formulas [Notation]
DEPENDENT_TOL = 1e-7  # the dependency test for coefficients [LA-4]
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

    The fit must not be degenerate: ``fit_mars`` applies EDGE-1 and the
    degenerate values of GCV-7 before it calls this function, so here
    N > 1 and some response is not constant. The final rss is the RSS of the
    returned coefficients, computed as the centered projection on the columns
    that LA-4 keeps, so it meets LA-5 also when Y has a large mean. Cost:
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

    def rss_of(terms) -> float:
        key = frozenset(terms)
        if key not in memo:
            memo[key] = rss(B[:, sorted(key)], Y, w)
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
    BS = B[:, selected]
    coef = lstsq_coef(BS, Y, w)
    final_rss = rss(BS[:, ~dependent_columns(BS, w)], Y, w)
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
    """One candidate of a forward step [FWD-3]; two candidates differ when they
    differ in the parent, the variable, the kind or the knot [CORE-3]."""

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


def queue_value(candidates, searchable: bool, rss_s: float, limit: float) -> float:
    """lambda_e of a searched entry [FAST-5]: the largest legal reduction among
    its candidates, linear candidates of any size included; 0 when none is
    legal, and -1 when no covariate could be searched."""
    if not searchable:
        return -1.0
    return max(
        (rss_s - c.rss for c in candidates if is_legal(c.kind, rss_s - c.rss, limit)),
        default=0.0,
    )


def candidate_key(c) -> tuple:
    """What tells two candidates apart [CORE-3]: the parent, the variable, the
    kind and the knot, with None as the knot of a linear candidate."""
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


def _parent_candidates(
    k, parent_row, X, Y, w, B, P_B, e_B, sigma2, N, tau_N, params, tau
):
    """The candidates of parent term k (column k of B), in the order of FWD-5,
    and whether some covariate could be searched for it [FWD-2, FWD-3].

    For each covariate that the parent lacks, in increasing order: the kind
    of the search by the pymars rule of LA-7; in a pair search the linear
    candidate, with RSS(B + {b x}); then the knots of KNOT-6 from the
    largest down, without the ones that LA-3 rejects, each with RSS(G + {h})
    [LA-2], where G is B, and B with b x in a pair search. Each hinge column
    is built and projected on G explicitly; its RSS is that of the residual.
    Cost: up to n knots per covariate, each projected on up to M + 1
    columns, so O(p n^2 M) time, and O(n^2) memory for the hinge matrix.
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
    out = []
    for j in covariates:
        x = X[:, j]
        bx = b * x
        V = [*own, j]
        pair = all(sigma2[v] > 0 for v in V) and (
            P_B.rss(bx) >= 0.01 * math.prod(sigma2[v] for v in V)
        )
        if pair:
            P_G = Projector(np.column_stack([B, bx]), w)
            e_G = P_G.residual(Y)
            out.append(Candidate(k, j, LINEAR, math.nan, float(np.sum(e_G**2))))
        else:
            P_G, e_G = P_B, e_B
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
        for i in np.flatnonzero(accept):
            h = H_perp[:, i]
            residual = e_G - np.outer(h, (h @ e_G) / size[i])
            kind = PAIR if pair else SINGLE
            out.append(Candidate(k, j, kind, float(t[i]), float(np.sum(residual**2))))
    return out, True


def _first_best(candidates, rss_s):
    """The candidate with the largest reduction; equal reductions go to the one
    found first [FWD-5]."""
    best = None
    for c in candidates:
        if best is None or rss_s - c.rss > rss_s - best.rss:
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
    """The CandidateLog of CORE-3 from (chosen, second) pairs, one per step."""

    def field(name, none, dtype):
        return np.array(
            [none if s is None else getattr(s, name) for _, s in log], dtype
        )

    return {
        "best_rss": np.array([c.rss for c, _ in log], dtype=np.float64),
        "second_rss": field("rss", math.inf, np.float64),
        "second_parent": field("parent", -1, np.int64),
        "second_variable": field("variable", -1, np.int64),
        "second_knot": np.array(
            [math.nan if s is None or s.kind == LINEAR else s.knot for _, s in log],
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
    s = 0
    while True:
        kappa = 2 * (s + 1)
        if 1 + 2 * (s + 1) > max_terms:  # [STOP-1]
            termination = NO_ROOM if s == 0 else TERM_LIMIT
            break
        B = np.column_stack(columns)
        P_B = Projector(B, w)
        e_B = P_B.residual(Y)
        rss_s = rss_path[-1]
        limit = max_legal(rss_path)
        tau = collinearity_tol(s)
        table = queue_table(entries, s, params.fast_beta)
        visited = window(table, params.fast_k)
        found, searched = [], []
        for e in visited:
            k = slots.get(e)
            if k is None or term_degree(dirs[k]) >= params.max_degree:
                continue  # skipped; the entry keeps its values [FAST-4]
            cands, searchable = _parent_candidates(
                k, dirs[k], X, Y, w, B, P_B, e_B, sigma2, N, tau_N, params, tau
            )
            entries[e - 1] = [queue_value(cands, searchable, rss_s, limit), kappa]
            found.extend(cands)
            searched.append(k)
        legal = [c for c in found if is_legal(c.kind, rss_s - c.rss, limit)]
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
            termination = GRSQ_NEG_INF if grsq_new == -math.inf else GRSQ_LOW
            break
        if rsq_gain < params.thresh:  # [STOP-4]
            termination = RSQ_CHANGE_SMALL
            break
        if chosen is None:  # [STOP-2]
            termination = NO_GAIN
            break
        new = _new_terms(chosen, X, columns, dirs, cuts, params.auto_linpreds)
        for i, (row, cut, column) in enumerate(new):  # [FWD-9, FAST-1, TERM-6]
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
            others = [c for c in legal if candidate_key(c) != candidate_key(chosen)]
            log.append((chosen, _first_best(others, rss_s)))
        floor = 1e-10 * tss / (N - 1)
        if rsq(chosen.rss, tss) >= 1 - params.thresh or chosen.rss < floor:
            termination = RSQ_HIGH  # [STOP-5]
            break
    B = np.column_stack(columns)
    dropped = dependent_columns(B, w)  # [FWD-11]
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


def y_scale_power(Y) -> int:
    """j of EDGE-6: D 2**j lies in [1, 2), where D is the largest |Y_ik|;
    0 when D = 0."""
    D = float(np.max(np.abs(Y)))
    return 0 if D == 0.0 else 1 - math.frexp(D)[1]


def fit_mars(X, Y, w=None, params=None, *, record_candidates=False) -> dict:
    """The fit of CORE-1, returned as the dict of CORE-5.

    X is (n, p); Y is (n, K), or (n,) for K = 1 [RESP-2]; w is (n,) with a
    positive sum, or None for w_i = 1 exactly [W-5]. The input is valid,
    as the estimators check it first [CORE-1]. Rows with zero weight are
    dropped before anything else [W-3]. Y is multiplied by 2**j before any
    sum, and every value on the scale of Y is scaled back [EDGE-6]. The
    keys follow CORE-3; ``forward``, ``pruning`` and ``candidates`` are
    nested dicts and ``termination`` is the integer code [CORE-5].
    """
    params = as_params(params)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y.reshape(Y.shape[0], -1)
    w = np.ones(X.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    rows = w > 0
    X, Y, w = X[rows], Y[rows], w[rows]
    n, p = X.shape
    N0 = weight_sum(w)
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
    pruned = prune(
        B[:, kept],
        Ys,
        w,
        penalty=common["penalty"],
        N=N,
        tau_N=tau_N,
        pmethod=params.pmethod,
        nprune=params.nprune,
    )
    selected = kept[pruned["selected"]]
    forward = {**forward, "rss": _unscale(forward["rss"], -2 * j)}
    if forward["candidates"] is not None:
        log = forward["candidates"]
        forward["candidates"] = {
            **log,
            "best_rss": _unscale(log["best_rss"], -2 * j),
            "second_rss": _unscale(log["second_rss"], -2 * j),
        }
    return {
        "dirs": forward["dirs"][selected],
        "cuts": forward["cuts"][selected],
        "coef": _unscale(pruned["coef"], -j),
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
