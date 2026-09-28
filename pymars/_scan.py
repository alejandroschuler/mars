"""The knot scan of one knot search, and the explicit rebuild of the winner.

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
with ‖h⊥‖² = ‖h‖² - ‖Qᵀh‖², and the ratio of LA-3 is
rho = ‖h⊥‖² / (‖h‖² - (q₀ᵀh)²), where the denominator is the centered sum of
squares of h, since q₀ is the intercept direction.

The sums. Sort the cases by x. A knot is a data value, t = x_{s-1} for the
0-based suffix start s (``_knots.KnotCandidates.split``), and h is 0 below s.
With the gaps g_j = x_j - x_{j-1} ≥ 0, x_q - t = Σ_{j=s..q} g_j for q ≥ s, so
for any vector v

    hᵀv = Σ_{j≥s} g_j·P_v(j),   P_v(j) = Σ_{q≥j} b_q·v_q,

and, with W(j) = Σ_{q≥j} b_q² and H(j) = Σ_{j'≥j} g_{j'}·W(j'),

    ‖h‖² = Σ_{j≥s} g_j·(2·H(j + 1) + g_j·W(j)).

These are the suffix sums of the plan's derivation, rewritten in the gaps of
x: every knot comes from two passes over the n cases, in O(n·(r + K)); a
shift of x changes no sum, which is what centering x does in the plan; and
‖h‖² is a sum of terms ≥ 0, so it has no cancellation. The one subtraction
left is ‖h⊥‖² = ‖h‖² - ‖Qᵀh‖², which loses precision when h lies nearly in
the span of Q; the collinearity test (rho ≥ tau ≥ 1e-5, LA-3) bounds that loss,
and ``rebuild`` computes the chosen candidate again explicitly.

Weights (stage 3 of T11). With case weights, b is √w·b, and Q and E are the
basis and the residuals of the √w-scaled problem (plan: Fast path); the
formulas are unchanged, and no division by a weight occurs.

Numerics. float64; no function writes into its inputs; no absolute epsilon.
Memory is O(n·(r + K)) for the temporary sums.

Public names, for ``_forward``:

- ``hinge_products``: hᵀV and ‖h‖² for every knot (eq. 52).
- ``knot_scan``, ``KnotScan``: rho and the RSS reduction of every knot.
- ``rebuild``, ``Rebuild``: the chosen candidate's columns, orthogonalized
  twice, and its RSS (LA-1).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pymars import _linalg

FloatArray = npt.NDArray[np.float64]


class KnotScan(NamedTuple):
    """``knot_scan``: one entry per knot. ``ratio`` is rho of LA-3, 0.0 when the
    centered sum of squares of h is not positive (a constant h has exactly
    0); ``gain`` is the RSS reduction of adding h to G, summed over the
    responses, 0.0 where ‖h⊥‖² is not positive."""

    ratio: FloatArray
    gain: FloatArray


class Rebuild(NamedTuple):
    """``rebuild``: ``Q`` with the new orthonormal columns appended, the
    residuals ``resid`` (n, K) of the responses on it, and ``rss``, their
    sum of squares (LA-1)."""

    Q: FloatArray
    resid: FloatArray
    rss: float


def _suffix(a: FloatArray) -> FloatArray:
    """Return the suffix sums of a along axis 0: out[j] = Σ_{q≥j} a[q]."""
    return np.cumsum(a[::-1], axis=0)[::-1]


def hinge_products(
    x: npt.ArrayLike, b: npt.ArrayLike, V: npt.ArrayLike, split: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return hᵀV (k, c) and ‖h‖² (k,) for the hinge of every knot (eq. 52).

    x (n,) is the covariate sorted in ascending order, b (n,) the parent
    column and V (n, c) the vectors, both in the order of x. ``split`` (k,)
    holds the suffix starts s, 1 ≤ s ≤ n - 1, and the hinge of s is
    h_q = b_q·(x_q - x_{s-1}) for q ≥ s and 0 below: the knot x_{s-1} with
    ``_knots.candidate_knots``'s ``split``. Raises ValueError for unsorted or
    nonfinite x, mismatched shapes or a split outside the range.
    Complexity: O(n·c) time and memory.
    """
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    split = np.asarray(split, dtype=np.int64)
    n = x.shape[0]
    if x.ndim != 1 or b.shape != x.shape or V.ndim != 2 or V.shape[0] != n:
        raise ValueError("x and b must be 1-D and V 2-D, with one row per case")
    if not np.isfinite(x).all() or np.any(x[1:] < x[:-1]):
        raise ValueError("x must be finite and sorted in ascending order")
    if split.ndim != 1 or np.any(split < 1) or np.any(split > n - 1):
        raise ValueError(f"every split must be in 1..{n - 1}")
    g = np.diff(x)
    HV = _suffix(g[:, None] * _suffix(b[:, None] * V)[1:])
    W = _suffix(b * b)[1:]  # W(j) for j = 1..n - 1
    H = _suffix(g * W)
    F = _suffix(g * (2.0 * np.append(H[1:], 0.0) + g * W))
    return HV[split - 1], F[split - 1]


def knot_scan(
    x: npt.ArrayLike,
    b: npt.ArrayLike,
    Q: npt.ArrayLike,
    E: npt.ArrayLike,
    split: npt.ArrayLike,
) -> KnotScan:
    """Return rho (LA-3) and the RSS reduction (LA-2) of the hinge of every knot.

    x, b and ``split`` are as in ``hinge_products``; Q (n, r), r ≥ 1, is an
    orthonormal basis of G with column 0 the intercept direction, and E (n,)
    or (n, K) the residuals of the responses on G; both have their rows in
    the order of x. For a pair search the caller passes G = B and b·x
    (FWD-3), so the reduction is that of the hinge given b·x. The values are
    those of the sums, accurate to LA-5 when rho ≥ tau; the caller applies the
    tolerance tau and the legality rules. Complexity: O(n·(r + K)) time and
    memory.
    """
    Q = np.asarray(Q, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    E = E[:, None] if E.ndim == 1 else E
    if Q.ndim != 2 or Q.shape[1] < 1 or E.ndim != 2:
        raise ValueError("Q must be 2-D, intercept direction first; E (n,) or (n, K)")
    r = Q.shape[1]
    HV, F = hinge_products(x, b, np.hstack((Q, E)), split)
    D = F - HV[:, 0] ** 2
    perp = D - np.sum(HV[:, 1:r] ** 2, axis=1)
    ratio = np.divide(perp, D, out=np.zeros_like(D), where=D > 0.0)
    num = np.sum(HV[:, r:] ** 2, axis=1)
    gain = np.divide(num, perp, out=np.zeros_like(perp), where=perp > 0.0)
    return KnotScan(ratio, gain)


def rebuild(
    Q: npt.ArrayLike, Y: npt.ArrayLike, columns: npt.ArrayLike
) -> Rebuild | None:
    """Append the chosen candidate's columns to Q; return None if one is in the span.

    ``columns`` (n, a) are the new columns in the order of FWD-6's span (b·x
    before the hinge of a pair), √w-scaled in weighted fits. Each is
    orthogonalized twice against Q and the columns before it
    (``_linalg.gram_schmidt``); a column whose orthogonal part is exactly 0
    gives None. Y (n,) or (n, K) are the responses, centered or not, and the
    residuals come from Y again, orthogonalized twice, so rounding does not
    build up over the steps. Complexity: O(n·(r + a)·(a + K)) time,
    O(n·(r + a + K)) memory.
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
