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
- Two knot scans are here: a loop that follows KNOT-3 step by step (unit
  weights), and the cumulative-weight scan of KNOT-6, so that each checks
  the other.

It is slow on purpose, and practical up to n <= 300, p <= 6 and
max_degree <= 3 [CORE-7]. Bracketed IDs such as [GCV-2] cite the rules of
the spec. Indices are 0-based, as in the spec. No function changes its
inputs.
"""

from __future__ import annotations

import math

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
    """L = max(1, trunc(-log2(-ln(1 - alpha) / (p N_b)) / 2.5)) [SPAN-1]."""
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
    and the minspan is not capped [SPAN-6]."""
    L = auto_minspan(p, Nb) if minspan is None else int(minspan)
    E = auto_endspan(p) if endspan is None else int(endspan)
    return L, scan_endspan(adjusted_endspan(E, parent_degree, adjust_endspan), N, tau_N)


# ---------------------------------------------------------------------------
# Candidate knots [KNOT-1 to KNOT-6]


def case_order(x: np.ndarray, active: np.ndarray) -> np.ndarray:
    """The order of KNOT-2: ascending x, and among equal x the inactive cases first."""
    return np.lexsort((np.asarray(active, dtype=np.int8), np.asarray(x)))


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
    g = (n - 2 * E - 1) % L
    counter = E + (g + 1) // 2  # E* + ceil(g / 2)
    knots = []
    for q in range(n, E + 1, -1):  # q = n, n - 1, ..., E* + 2 (1-based)
        t = xs[q - 2]  # x_(q-1)
        if t >= v:
            continue
        if act[q - 1]:  # a_q
            counter -= 1
            if counter == 0:
                knots.append(float(t))
                counter = L
    return knots


def knot_scan(x, active, w, minspan: int, endspan: int, N: float, tau_N: float):
    """The knot list of KNOT-6 (cumulative weight), in scan order, repeats included.

    The visits u = N, N - 1, ... while u >= E* + 2 - tau_N are listed at once.
    At each visit t = x(u - 1), the counter moves when t < v and a(u) = 1,
    and a knot is appended at the c0-th move and at every L-th move after it,
    which is where the counter of KNOT-3 reaches 0.
    """
    x = np.asarray(x, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    w = np.asarray(w, dtype=np.float64)
    n = x.shape[0]
    L, E = int(minspan), int(endspan)
    if not active.any():
        return []
    last = math.floor(N - (E + 2 - tau_N))
    if last < -1:
        return []
    u = N - np.arange(last + 2, dtype=np.float64)
    u = u[u >= E + 2 - tau_N]
    if u.size == 0:
        return []
    order = case_order(x, active)
    xs, act = x[order], active[order]
    upper = np.cumsum(w[order]) + tau_N  # W_q + tau_N
    # the case that holds u: the smallest q with u <= W_q + tau_N, else case n
    holds = np.minimum(np.searchsorted(upper, u, side="left"), n - 1)
    holds_below = np.minimum(np.searchsorted(upper, u - 1.0, side="left"), n - 1)
    t = xs[holds_below]
    moves = (t < x[active].max()) & act[holds]
    D = N - 2 * E - 1
    g = D - L * math.floor(D / L)
    c0 = E + math.ceil(g / 2)
    count = np.cumsum(moves)
    hits = moves & (count >= c0) & ((count - c0) % L == 0)
    return [float(value) for value in t[hits]]


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
