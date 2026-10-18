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
‖QᵀE_k‖ ≤ √r·gamma_n·‖E_k‖). The forward pass meets them with Gram-Schmidt
twice for the columns of its basis, and for a pair search's b·x with one pass
when that pass keeps more than half of ‖v‖² and a second otherwise (DGKS),
whose departure is a small multiple of r·u, far inside the margin;
``test_every_scan_gets_q_and_e_that_meet_its_preconditions`` checks the
margin on the oracle's draws. The bound
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

from pymars import _kernels, _linalg

FloatArray = npt.NDArray[np.float64]

#: The unit roundoff of float64.
UNIT_ROUNDOFF = 2.0**-53
#: Below this scaled ‖h‖² or D, a bound would not cover underflow: far above
#: the subnormal range (2^-1022), so that n·2^-1022 stays below u·2^-900.
SCALED_FLOOR = 2.0**-900
# _suffix adds whole rows of its blocks for arrays with at least this many
# columns (a speed choice only: both ways give the same bits).
_WIDE = 6


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
    residuals ``resid`` (n, K) of the responses on it, and ``rss``, their sum
    of squares (LA-1)."""

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
    if a.ndim == 2 and a.shape[1] >= _WIDE:
        # The same additions in the same order, so the same bits, but one row
        # of every block at a time: a cumsum along the strided block axis is
        # slow, and adds of whole contiguous (nb, c) rows are not.
        T = np.empty((B, nb, *rest))
        np.moveaxis(T, 0, 1)[...] = rev.reshape((nb, B, *rest))
        for i in range(1, B):
            np.add(T[i - 1], T[i], out=T[i])
        carry = np.zeros((nb, *rest))
        np.cumsum(T[-1, :-1], axis=0, out=carry[1:])
        T += carry
        return np.moveaxis(T, 0, 1).reshape((nb * B, *rest))[:m][::-1]
    within = np.cumsum(rev.reshape((nb, B, *rest)), axis=1)
    carry = np.zeros((nb, *rest))
    np.cumsum(within[:-1, -1], axis=0, out=carry[1:])
    return (within + carry[:, None]).reshape((nb * B, *rest))[:m][::-1]


def _products(
    g: FloatArray, b: FloatArray, V: FloatArray, split: npt.NDArray[np.int64]
) -> tuple[FloatArray, FloatArray]:
    """hᵀV and ‖h‖² at every split, from the gaps g (module docstring)."""
    return _hv(g, b, V, split), _norms(g, b, split)


def _hv(
    g: FloatArray, b: FloatArray, V: FloatArray, split: npt.NDArray[np.int64]
) -> FloatArray:
    """hᵀV at every split. Each column of V is summed on its own, so a
    column's values do not depend on the other columns."""
    return _suffix(g[:, None] * _suffix(b[:, None] * V)[1:])[split - 1]


def _norms(g: FloatArray, b: FloatArray, split: npt.NDArray[np.int64]) -> FloatArray:
    """‖h‖² at every split."""
    W = _suffix(b * b)[1:]  # W(j) for j = 1..n - 1
    H = _suffix(g * W)
    F = _suffix(g * (2.0 * np.append(H[1:], 0.0) + g * W))
    return F[split - 1]


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


def _scan_hv(g, b, Q, q0, E, o, split) -> FloatArray:
    """hᵀV for V = [Q[o, q0:], E] at every split: the compiled kernel when
    ``_kernels.ENABLED``, else ``_hv`` on V (Q then in the order of x, o
    unused)."""
    if _kernels.ENABLED:
        return _kernels.hv(g, b, Q, q0, E, o, split)
    return _hv(g, b, np.hstack((Q[:, q0:], E)), split)


def _scan_norms(g: FloatArray, b: FloatArray, split) -> FloatArray:
    """``_norms``, compiled when ``_kernels.ENABLED``; the same bits."""
    return (_kernels.norms if _kernels.ENABLED else _norms)(g, b, split)


def _scan_centered(g: FloatArray, w: FloatArray, split) -> FloatArray:
    """``_centered``, compiled when ``_kernels.ENABLED``; the same bits."""
    return (_kernels.centered if _kernels.ENABLED else _centered)(g, w, split)


def _to_unit(a: FloatArray) -> FloatArray:
    """a times the power of 2 that puts its largest |value| in [0.5, 1)."""
    top = float(np.max(np.abs(a))) if a.size else 0.0
    return np.ldexp(a, -int(np.frexp(top)[1])) if top > 0.0 else a


def unit_gaps(x: FloatArray) -> FloatArray:
    """The gaps diff(x) of a sorted x, times the power of 2 that puts the
    largest in [0.5, 1), as ``knot_scan`` uses them. Complexity: O(n)."""
    return _kernels.unit(x, True) if _kernels.ENABLED else _to_unit(np.diff(x))


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
    _checked: bool = False,
    _cache: tuple[dict, int] | None = None,
    _order: npt.NDArray[np.int64] | None = None,
    _gaps: FloatArray | None = None,
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
    ``_checked=True`` skips the checks of ``hinge_products``, for the forward
    pass, whose x, b and split are float64 and int64 arrays that meet them.
    ``_cache = (store, m)``, for the forward pass, keeps in ``store`` the
    K-vectors ‖h‖² and D (not g and b, which cost O(n) to recompute), which
    depend only on x, b and split, and hᵀq for the first m columns of Q
    (after the intercept column with ``intercept=True``), which the caller
    promises are the same at the next call with this store; a later call
    computes hᵀq only for the other columns. Each column is summed on its own
    (``_hv``), so the store changes no value.
    ``_order`` (n,), for the forward pass, says that the rows of Q and E are
    in the caller's order and that row q in the order of x is row
    ``_order[q]``; the compiled scan (``_kernels``) then reads Q at those
    rows without a gathered copy. The compiled and the numpy scan
    agree within the error bounds (``_kernels``). ``_gaps``, for the forward pass, is
    ``unit_gaps(x)``, which depends on x alone, so the pass keeps it per
    covariate.
    Complexity: O(n·(r + K)) time and memory.
    """
    Q = np.asarray(Q, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    E = E[:, None] if E.ndim == 1 else E
    if Q.ndim != 2 or Q.shape[1] < 1 or E.ndim != 2:
        raise ValueError("Q must be 2-D, intercept direction first; E (n,) or (n, K)")
    n, r = Q.shape
    fast = _kernels.ENABLED
    if _order is not None:  # O(n·K); the kernels read Q at the rows o
        E = E[_order]
        if not fast:
            Q = Q[_order]
    o = np.arange(n) if _order is None else _order  # read only by the kernels
    q0 = 1 if intercept else 0  # the columns of Q that enter hᵀV
    k = r - q0
    if _cache is None and not _checked:
        x, b, _, split = _check(x, b, np.hstack((Q[:, q0:], E)), split)
    b0 = b
    if _gaps is not None:
        g, b = _gaps, (_kernels.unit(b0, False) if fast else _to_unit(b0))
    else:
        g, b = _kernels.prep(x, b0) if fast else (_to_unit(np.diff(x)), _to_unit(b0))
    if _cache is None:
        HV = _scan_hv(g, b, Q, q0, E, o, split)
        F = _scan_norms(g, b, split)
        D = _scan_centered(g, b * b, split) if intercept else None
    else:  # the forward pass's own arrays, as with _checked
        store, m = _cache
        if "F" not in store:
            store.update(F=_scan_norms(g, b, split), m=0)
            store["D"] = _scan_centered(g, b * b, split) if intercept else None
            store["HV"] = np.empty((split.shape[0], 0))
        F, D = store["F"], store["D"]
        known = store["m"]
        assert known <= m <= k, "the kept columns are a prefix of Q"
        new = _scan_hv(g, b, Q, q0 + known, E, o, split)
        HV = np.hstack((store["HV"], new))
        store["HV"], store["m"] = HV[:, :m], m
    e2 = np.sum(E * E, axis=0)  # the squared norms of the responses
    lam = 2 * (math.isqrt(n - 1) + 1)  # the roundings of one blocked suffix sum
    g_n = _gamma(n + 2)
    cst = np.array(
        [
            UNIT_ROUNDOFF,
            _gamma(2 * lam + 4),  # of the sums hᵀV and ‖h‖²
            (r + 1) * g_n,  # dQ, the precondition on ‖QᵀQ - I‖₂
            _gamma(7 * lam + 20) if intercept else _gamma(3 * lam + 8),  # of D
            _gamma(2 * lam + 4) + math.sqrt(r) * g_n,  # the factor of e_k
            SCALED_FLOOR,
        ]
    )
    if fast:
        return KnotScan(
            *_kernels.finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst)
        )
    return _finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst)


def _finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst) -> KnotScan:
    """rho, the reduction and their error bounds (module docstring) from HV,
    which holds hᵀq for the k columns of Q that enter (column 0 is q₀ unless
    ``intercept``) and then hᵀE, ‖h‖² (F), the intercept parent's D, and the
    squared norms e2 of the columns of E; ``cst`` holds u, the gamma of the
    sums, dQ, the gamma of D, the factor of e_k and the floor. The numpy form
    of ``_kernels.finish``. Complexity: O(k·c + n)."""
    u, g_hv, dQ, gD, ce, _ = cst  # the floor is SCALED_FLOOR
    he = HV[:, k:]
    if intercept:
        c = HV[:, :k]
    else:
        c0, c = HV[:, 0], HV[:, 1:k]
        D = F - c0**2
    s2 = np.sum(c**2, axis=1)
    perp = D - s2
    ratio = np.divide(perp, D, out=np.zeros_like(D), where=D > 0.0)
    num = np.sum(he**2, axis=1)
    gain = np.divide(num, perp, out=np.zeros_like(perp), where=perp > 0.0)

    sqF = np.sqrt(F)
    e_c = (g_hv + dQ) * sqF  # the error of each q_cᵀh
    if intercept:  # and the departure of Q[:, 0] from b/‖b‖, one rounding per row
        errD = gD * D + 2.0 * u * np.sqrt(F * np.abs(D))
    else:
        errD = gD * F + 2.0 * np.abs(c0) * e_c + e_c**2
        errD += (dQ + 2.0 * u) * F
    errP = errD + 2.0 * np.sum(np.abs(c), axis=1) * e_c + (r - 1) * e_c**2
    errP += dQ * s2 + 4.0 * u * (np.abs(D) + s2)
    e_k = np.outer(sqF, np.sqrt(e2)) * ce
    errN = np.sum(2.0 * np.abs(he) * e_k + e_k**2, axis=1) + 2.0 * he.shape[1] * u * num
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
    Q: npt.ArrayLike,
    E: npt.ArrayLike,
    h: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    N: float | None = None,
) -> tuple[float, float]:
    """Return rho (LA-3) and the RSS reduction (LA-2) of one hinge column,
    computed from the column itself.

    Q (n, r) is the √w-scaled orthonormal basis of G, with the intercept, and E
    (n,) or (n, K) the √w-scaled residuals on G; h (n,) is the unscaled hinge
    column and w the weights, or None; N is Σw (``math.fsum(w)`` when None).
    One Gram-Schmidt of √w·h against Q (``_linalg.cgs2``) gives all three
    values: rho = ‖h⊥‖²/‖h - h̄‖_w², as ``_linalg.collinearity_ratio`` gives it
    (0.0 for an h that is constant over the cases); LA-4's test of h against G
    (``_linalg.dependent`` on ‖h⊥‖ and ‖√w·h‖); and the reduction Σ_k (qᵀE_k)²
    with q = h⊥/‖h⊥‖, so the residuals' own departure from QᵀE = 0 does not
    enter. A dependent h has the reduction 0.0 (LA-2), so the rounding noise of
    a column in the span never counts. h is first multiplied by a power of 2,
    which changes none of the values and keeps the sums of squares in range.
    Complexity: O(n·(r + K)).
    """
    Q = np.asarray(Q, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    wv = np.ones(h.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    cases = h if w is None else h[wv > 0.0]
    if cases.size == 0 or np.all(cases == cases[0]):
        return 0.0, 0.0  # constant over the cases: LA-3 rejects it
    hs = _to_unit(h if w is None else np.where(wv > 0.0, h, 0.0))  # W-3
    N = math.fsum(wv) if N is None else N
    d = hs - float(wv @ hs) / N
    centered = float(wv @ (d * d))
    o = _linalg.cgs2(Q, hs if w is None else np.sqrt(wv) * hs)
    rho = o.norm**2 / centered if centered > 0.0 else 0.0
    if _linalg.dependent(o.norm, o.vnorm):
        return rho, 0.0
    return rho, float(np.sum(np.square(o.perp @ E))) / o.norm**2


def rebuild(
    Q: npt.ArrayLike,
    Y: npt.ArrayLike,
    columns: npt.ArrayLike,
    *,
    _buffer: FloatArray | None = None,
) -> Rebuild | None:
    """Append the chosen candidate's columns to Q; return None when the part of
    a column orthogonal to the others is exactly 0.

    ``columns`` (n, a) are the new columns in the order of FWD-6's span (b·x
    before the hinge of a pair), √w-scaled in weighted fits. Each is
    orthogonalized twice against Q and the columns before it
    (``_linalg.cgs2``). A column that lies in the span only up to rounding
    gets a unit vector of rounding noise, so the caller tests a hinge with
    LA-3 first. Y (n,) or (n, K) are the responses, centered (√w·(Y - Ȳ_w)
    with weights): the residuals come from Y again, orthogonalized twice, so
    rounding does not build up over the steps, but a large mean of Y would
    keep the rounding of its projection. ``_buffer``, for the forward pass, is
    an (n, c) C-order array, c ≥ r + a: Q and the new columns go into it, and
    ``Rebuild.Q`` is a view of it.
    Complexity: O(n·(r + a)·(a + K)) time, O(n·(r + a + K)) memory.
    """
    Q = np.asarray(Q, dtype=np.float64)
    columns = np.asarray(columns, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, r = Q.shape
    a = columns.shape[1]
    if _buffer is None:
        _buffer = np.empty((n, r + a))
    _buffer[:, :r] = Q
    for i in range(a):
        o = _linalg.cgs2(_buffer[:, : r + i], columns[:, i], twice=True)
        if o.norm == 0.0:
            return None
        _buffer[:, r + i] = o.perp / o.norm
    Q = _buffer[:, : r + a]
    resid, _ = _linalg.orthogonalize(Q, Y[:, None] if Y.ndim == 1 else Y)
    return Rebuild(Q, resid, float(np.sum(resid * resid)))
