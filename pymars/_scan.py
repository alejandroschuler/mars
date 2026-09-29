"""The knot scan of one knot search, with an error bound for every knot, the
explicit values of one knot, and the explicit rebuild of the winner.

Spec: ``docs/algorithm.md``, "Linear algebra contract" (LA-1 to LA-3, LA-5,
LA-7) and "Forward pass" (FWD-3); the plan's section "Fast path" (Friedman
1991, eq. 52) sets the method. ``_forward`` calls these functions for each
parent b and covariate x that a step searches.

The problem. Q (n x r) is an orthonormal basis of the columns G of the search
(the current columns B in a single-hinge search, B and b·x in a pair search,
LA-7), with column 0 the intercept direction, and E (n x K) holds the
residuals of the responses on G, so QᵀE = 0. The hinge column of a knot t is
h = b·(x - t)₊, and its part orthogonal to Q is h⊥ = h - QQᵀh. Adding h to G
lowers the RSS by Σ_k (hᵀE_k)² / ‖h⊥‖² (LA-2; hᵀE_k = h⊥ᵀE_k since QᵀE = 0),
with ‖h⊥‖² = D - Σ_{c≥1} (q_cᵀh)², where D = ‖h‖² - (q₀ᵀh)² is the centered
sum of squares of h (q₀ is the intercept direction); rho of LA-3 is
‖h⊥‖² / D.

The sums. Sort the cases by x. A knot is a data value, t = x_{s-1} for the
0-based suffix start s (``_knots.KnotCandidates.split``), and h is 0 below s.
With the gaps g_j = x_j - x_{j-1} ≥ 0, x_q - t = Σ_{j=s..q} g_j for q ≥ s, so
for any vector v

    hᵀv = Σ_{j≥s} g_j·P_v(j),   P_v(j) = Σ_{q≥j} b_q·v_q,

and, with W(j) = Σ_{q≥j} b_q² and H(j) = Σ_{j'≥j} g_{j'}·W(j'),

    ‖h‖² = Σ_{j≥s} g_j·(2·H(j + 1) + g_j·W(j)).

These are the suffix sums of the plan's derivation, written in the gaps of x:
every knot comes from a few passes over the n cases, in O(n·(r + K)); a shift
of x changes no sum, which is what centering x does in the plan; and ‖h‖² is
a sum of terms ≥ 0.

Accuracy. u = 2⁻⁵³ is the unit roundoff, and gamma_k = k·u/(1 - k·u).

- The suffix sums add blocks of ⌈√m⌉ rows and carry the block totals, so a
  sum of m terms has an error of at most gamma_{2⌈√m⌉}·Σ|terms|, where a running
  sum has gamma_m.
- For the intercept parent (``intercept=True``: b = √w, and Q[:, 0] = √w/√N),
  D also comes from terms ≥ 0. With W_s the weight of the cases s to n - 1,
  W_pre that of the cases below s and S1 = Σ_{q≥s} w_q (x_q - t),

      D = SSW_s + W_pre·S1² / (W_s·(W_pre + W_s)),

  where SSW_s is the weighted sum of squares of x over the cases s to n - 1
  about their mean, from Welford's recursion written in the gaps
  (``_centered``). For another parent D = ‖h‖² - (q₀ᵀh)², which cancels when
  h is nearly constant.

What cannot be avoided is the subtraction ‖h⊥‖² = D - Σ_{c≥1} (q_cᵀh)², which
loses about log10(1/rho) digits, and the terms q_cᵀh carry the rounding of
sums over h, of size ‖h‖. So when h lies far from its own mean (‖h‖²/D large,
for example a few low values below a tight bulk) and rho is small, the error
of the scan can exceed LA-5. ``knot_scan`` therefore returns with each value
a bound on its error: the worst-case first-order rounding of the sums, plus
the departure of Q and E from exact orthonormality and orthogonality, which
the caller guarantees (the preconditions ‖QᵀQ - I‖₂ ≤ (r + 1)·gamma_n and
‖QᵀE_k‖ ≤ √r·gamma_n·‖E_k‖, which Gram-Schmidt applied twice gives); the bound
is doubled for the second-order terms. It bounds the distance from exact
arithmetic on the given Q and E, not the rounding that Q and E carry as a
basis and residuals of the data, which is far inside LA-5. The caller decides
with a scan value only when that value, moved by its bound either way,
decides the same, and otherwise with the explicit values of ``exact_knot``
(LA-5).

Weights. With case weights, b is √w·b, and Q and E are the
basis and the residuals of the √w-scaled problem (plan: Fast path); the
formulas are unchanged, and no division by a weight occurs.

Numerics. float64; no function writes into its inputs; no absolute epsilon.
The scan multiplies the gaps and b by the powers of 2 that bring their
largest values to [0.5, 1), which changes neither rho nor the reduction.
When the gaps or b span a wider range than about 2^450, the squares of the
small ones can leave the normal range of float64, where rounding is no longer
relative, so a knot whose scaled ‖h‖² or D is below 2^-900 gets infinite
bounds, and the caller values it with ``exact_knot``. A column that is 0 at
every case is found from the inputs (no q ≥ s with b_q ≠ 0 and
x_q > x_{s-1}), not from its sums (Conventions). Tied rows take the input
order, so in floating point a permutation of tied rows changes the sums by
rounding (KNOT-2 holds in exact arithmetic). Memory is O(n·(r + K)).

Public names, for ``_forward``:

- ``hinge_products``: hᵀV and ‖h‖² for every knot (eq. 52).
- ``knot_scan``, ``KnotScan``: rho and the RSS reduction of every knot, with
  their error bounds.
- ``exact_knot``: rho and the reduction of one knot from its column.
- ``rebuild``, ``Rebuild``: the chosen candidate's columns, orthogonalized
  twice, and its RSS (LA-1).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pymars import _linalg

FloatArray = npt.NDArray[np.float64]

#: The unit roundoff of float64.
UNIT_ROUNDOFF = 2.0**-53
#: Below this scaled ‖h‖² or D, a bound would not cover underflow: far above
#: the subnormal range (2^-1022), so that n·2^-1022 stays below u·2^-900.
SCALED_FLOOR = 2.0**-900


class KnotScan(NamedTuple):
    """``knot_scan``: one entry per knot. ``ratio`` is rho of LA-3, 0.0 when D
    is not positive; ``gain`` is the RSS reduction of adding h to G, summed over
    the responses, 0.0 where ‖h⊥‖² is not positive. ``ratio_err`` and
    ``gain_err`` bound the errors of the two (+∞ when the rounding could make D
    or ‖h⊥‖² vanish, or when underflow could reach them). A column h that is 0
    at every case, which LA-3 rejects as constant, gets 0.0 for all four."""

    ratio: FloatArray
    gain: FloatArray
    ratio_err: FloatArray
    gain_err: FloatArray


class Rebuild(NamedTuple):
    """``rebuild``: ``Q`` with the new orthonormal columns appended, the
    residuals ``resid`` (n, K) of the responses on it, and ``rss``, their
    sum of squares (LA-1)."""

    Q: FloatArray
    resid: FloatArray
    rss: float


def _gamma(k: float) -> float:
    """gamma_k = k·u/(1 - k·u), the bound of k roundings in a row."""
    return k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)


def _suffix(a: FloatArray) -> FloatArray:
    """Return the suffix sums of a along axis 0: out[j] = Σ_{q≥j} a[q].

    From the end, the rows are added in blocks of B = ⌈√m⌉, and the block
    totals are carried from block to block, so every output goes through at
    most 2⌈√m⌉ - 1 roundings. Complexity: O(m·c) time and memory for an (m, c)
    array.
    """
    m = a.shape[0]
    B = math.isqrt(max(m - 1, 0)) + 1
    nb = -(-m // B)
    rest = a.shape[1:]
    rev = np.concatenate((a[::-1], np.zeros((nb * B - m, *rest))))
    within = np.cumsum(rev.reshape((nb, B, *rest)), axis=1)
    carry = np.zeros((nb, *rest))
    np.cumsum(within[:-1, -1], axis=0, out=carry[1:])
    return (within + carry[:, None]).reshape((nb * B, *rest))[:m][::-1]


def _products(
    g: FloatArray, b: FloatArray, V: FloatArray, split: npt.NDArray[np.int64]
) -> tuple[FloatArray, FloatArray]:
    """hᵀV and ‖h‖² at every split, from the gaps g (module docstring)."""
    HV = _suffix(g[:, None] * _suffix(b[:, None] * V)[1:])
    W = _suffix(b * b)[1:]  # W(j) for j = 1..n - 1
    H = _suffix(g * W)
    F = _suffix(g * (2.0 * np.append(H[1:], 0.0) + g * W))
    return HV[split - 1], F[split - 1]


def _centered(g: FloatArray, w: FloatArray, split: npt.NDArray[np.int64]) -> FloatArray:
    """D of the intercept parent at every split, from terms ≥ 0.

    w (n,) holds the weights in the order of x (1 without weights). Welford's
    recursion SSW_j = SSW_{j+1} + w_j·W_{j+1}/W_j·(x̄_{j+1} - x_j)², with
    x̄_{j+1} - x_j = S1(j + 1)/W_{j+1} in the gaps, gives the suffix sums of
    squares; then D = SSW_s + W_pre·S1(s)²/(W_s·(W_pre + W_s)) (module
    docstring). With unit weights every W is an exact count. Complexity: O(n).
    """
    Wsuf = _suffix(w)  # W_j, j = 0..n - 1
    Wpre = _suffix(w[::-1])[::-1]  # the weight of the cases 0..j
    S1 = _suffix(g * Wsuf[1:])  # S1[j - 1] = S1(j), j = 1..n - 1
    SSW = np.append(_suffix(w[:-1] * S1**2 / (Wsuf[1:] * Wsuf[:-1])), 0.0)
    Ws, Wp, S = Wsuf[split], Wpre[split - 1], S1[split - 1]
    return SSW[split] + Wp * S**2 / (Ws * (Wp + Ws))


def _to_unit(a: FloatArray) -> FloatArray:
    """a times the power of 2 that puts its largest |value| in [0.5, 1)."""
    top = float(np.max(np.abs(a))) if a.size else 0.0
    return np.ldexp(a, -int(np.frexp(top)[1])) if top > 0.0 else a


def _check(
    x: npt.ArrayLike, b: npt.ArrayLike, V: npt.ArrayLike, split: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """The checks of ``hinge_products``, which ``knot_scan`` shares."""
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    split = np.asarray(split)
    n = x.shape[0]
    if x.ndim != 1 or b.shape != x.shape or V.ndim != 2 or V.shape[0] != n:
        raise ValueError("x and b must be 1-D and V 2-D, with one row per case")
    if not np.isfinite(x).all() or np.any(x[1:] < x[:-1]):
        raise ValueError("x must be finite and sorted in ascending order")
    if not (np.isfinite(b).all() and np.isfinite(V).all()):
        raise ValueError("b and V must be finite")
    if split.size and not np.issubdtype(split.dtype, np.integer):
        raise ValueError("every split must be an integer")
    split = split.astype(np.int64)
    if split.ndim != 1 or np.any(split < 1) or np.any(split > n - 1):
        raise ValueError(f"every split must be in 1..{n - 1}")
    return x, b, V, split


def hinge_products(
    x: npt.ArrayLike, b: npt.ArrayLike, V: npt.ArrayLike, split: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return hᵀV (k, c) and ‖h‖² (k,) for the hinge of every knot (eq. 52).

    x (n,) is the covariate sorted in ascending order, b (n,) the parent
    column and V (n, c) the vectors, both in the order of x. ``split`` (k,)
    holds integer suffix starts s, 1 ≤ s ≤ n - 1, and the hinge of s is
    h_q = b_q·(x_q - x_{s-1}) for q ≥ s and 0 below: the knot x_{s-1} with
    ``_knots.candidate_knots``'s ``split``. Raises ValueError for unsorted or
    nonfinite input, mismatched shapes or a split that is not an integer in
    the range. Complexity: O(n·c) time and memory.
    """
    x, b, V, split = _check(x, b, V, split)
    return _products(np.diff(x), b, V, split)


def knot_scan(
    x: npt.ArrayLike,
    b: npt.ArrayLike,
    Q: npt.ArrayLike,
    E: npt.ArrayLike,
    split: npt.ArrayLike,
    *,
    intercept: bool = False,
) -> KnotScan:
    """Return rho (LA-3) and the RSS reduction (LA-2) of the hinge of every
    knot, with a bound on the error of each.

    x, b and ``split`` are as in ``hinge_products``; Q (n, r), r ≥ 1, is an
    orthonormal basis of G with column 0 the intercept direction, and E (n,)
    or (n, K) the residuals of the responses on G; both have their rows in
    the order of x, and they meet the preconditions of the module docstring.
    For a pair search the caller passes G = B and b·x (FWD-3), so the
    reduction is that of the hinge given b·x. ``intercept=True`` says that b
    is √w for the intercept parent (1 without weights), and then D comes from
    terms ≥ 0. The bounds are those of the module docstring; the caller
    applies the tolerance tau and the legality rules with them.
    Complexity: O(n·(r + K)) time and memory.
    """
    Q = np.asarray(Q, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    E = E[:, None] if E.ndim == 1 else E
    if Q.ndim != 2 or Q.shape[1] < 1 or E.ndim != 2:
        raise ValueError("Q must be 2-D, intercept direction first; E (n,) or (n, K)")
    n, r = Q.shape
    part = Q[:, 1:] if intercept else Q
    x, b0, V, split = _check(x, b, np.hstack((part, E)), split)
    g, b = _to_unit(np.diff(x)), _to_unit(b0)
    HV, F = _products(g, b, V, split)
    k = part.shape[1]
    he = HV[:, k:]
    if intercept:
        c = HV[:, :k]
        D = _centered(g, b * b, split)
    else:
        c0, c = HV[:, 0], HV[:, 1:k]
        D = F - c0**2
    s2 = np.sum(c**2, axis=1)
    perp = D - s2
    ratio = np.divide(perp, D, out=np.zeros_like(D), where=D > 0.0)
    num = np.sum(he**2, axis=1)
    gain = np.divide(num, perp, out=np.zeros_like(perp), where=perp > 0.0)

    # The error bounds (module docstring), per knot.
    lam = 2 * (math.isqrt(n - 1) + 1)  # the roundings of one blocked suffix sum
    u, g_hv, g_n = UNIT_ROUNDOFF, _gamma(2 * lam + 4), _gamma(n + 2)
    dQ = (r + 1) * g_n  # the precondition on ‖QᵀQ - I‖₂
    sqF = np.sqrt(F)
    e_c = (g_hv + dQ) * sqF  # the error of each q_cᵀh
    if intercept:  # and the departure of Q[:, 0] from b/‖b‖, one rounding per row
        errD = _gamma(7 * lam + 20) * D + 2.0 * u * np.sqrt(F * np.abs(D))
    else:
        errD = _gamma(3 * lam + 8) * F + 2.0 * np.abs(c0) * e_c + e_c**2
        errD += (dQ + 2.0 * u) * F
    errP = errD + 2.0 * np.sum(np.abs(c), axis=1) * e_c + (r - 1) * e_c**2
    errP += dQ * s2 + 4.0 * u * (np.abs(D) + s2)
    normE = np.sqrt(np.sum(E**2, axis=0))
    e_k = np.outer(sqF, normE) * (g_hv + math.sqrt(r) * g_n)
    errN = np.sum(2.0 * np.abs(he) * e_k + e_k**2, axis=1) + 2.0 * E.shape[1] * u * num
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_err = np.where(
            errD < D, (errP + np.abs(ratio) * errD) / (D - errD), np.inf
        )
        gain_err = np.where(errP < perp, (errN + gain * errP) / (perp - errP), np.inf)
    ratio_err = 2.0 * (ratio_err + 4.0 * u * np.abs(ratio))
    gain_err = 2.0 * (gain_err + 4.0 * u * gain)
    lost = (F < SCALED_FLOOR) | (np.abs(D) < SCALED_FLOOR)  # underflow can reach
    ratio_err[lost], gain_err[lost] = np.inf, np.inf
    # h is 0 at every case, a constant column (LA-3), when no case q ≥ s has
    # b_q ≠ 0 and x_q > x_{s-1}: a count over the inputs.
    nonzero = np.append(np.cumsum((b0 != 0.0)[::-1])[::-1], 0)
    zero = nonzero[np.searchsorted(x, x[split - 1], side="right")] == 0
    for a in (ratio, gain, ratio_err, gain_err):
        a[zero] = 0.0
    return KnotScan(ratio, gain, ratio_err, gain_err)


def exact_knot(
    Q: npt.ArrayLike, E: npt.ArrayLike, h: npt.ArrayLike, w: npt.ArrayLike | None = None
) -> tuple[float, float]:
    """Return rho (LA-3) and the RSS reduction (LA-2) of one hinge column,
    computed from the column itself.

    Q (n, r) is the √w-scaled orthonormal basis of G, with the intercept, and E
    (n,) or (n, K) the √w-scaled residuals on G; h (n,) is the unscaled hinge
    column and w the weights, or None. rho is
    ``_linalg.collinearity_ratio(Q, h, w)``; the reduction is Σ_k (qᵀE_k)², with
    q the unit vector of √w·h orthogonalized twice against Q
    (``_linalg.gram_schmidt``), so the residuals' own departure from QᵀE = 0
    does not enter; it is 0.0 when that part is exactly 0. For a column that
    lies in the span only up to rounding the reduction is rounding noise, as
    in ``rebuild``, so the caller decides LA-3 on rho first.
    Complexity: O(n·(r + K)).
    """
    Q = np.asarray(Q, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    rho = _linalg.collinearity_ratio(Q, h, w)
    v = h if w is None else np.sqrt(np.asarray(w, dtype=np.float64)) * h
    q = _linalg.gram_schmidt(Q, v).q
    return rho, 0.0 if q is None else float(np.sum(np.square(q @ E)))


def rebuild(
    Q: npt.ArrayLike, Y: npt.ArrayLike, columns: npt.ArrayLike
) -> Rebuild | None:
    """Append the chosen candidate's columns to Q; return None when the part of
    a column orthogonal to the others is exactly 0.

    ``columns`` (n, a) are the new columns in the order of FWD-6's span (b·x
    before the hinge of a pair), √w-scaled in weighted fits. Each is
    orthogonalized twice against Q and the columns before it
    (``_linalg.gram_schmidt``). A column that lies in the span only up to
    rounding gets a unit vector of rounding noise, so the caller tests a
    hinge with LA-3 first. Y (n,) or (n, K) are the responses, centered
    (√w·(Y - Ȳ_w) with weights): the residuals come from Y again,
    orthogonalized twice, so rounding does not build up over the steps, but a
    large mean of Y would keep the rounding of its projection.
    Complexity: O(n·(r + a)·(a + K)) time, O(n·(r + a + K)) memory.
    """
    Q = np.asarray(Q, dtype=np.float64)
    columns = np.asarray(columns, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    for v in columns.T:
        q = _linalg.gram_schmidt(Q, v).q
        if q is None:
            return None
        Q = np.column_stack((Q, q))
    resid, _ = _linalg.orthogonalize(Q, Y[:, None] if Y.ndim == 1 else Y)
    return Rebuild(Q, resid, float(np.sum(resid * resid)))
