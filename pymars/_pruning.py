"""The pruning pass: the best subset of each size, the size with the lowest GCV,
and the least squares of the selected terms.

Spec: ``docs/algorithm.md``, "Pruning pass" (PRUNE-1 to PRUNE-9), and the rules
that it calls: LA-1 and LA-4 (least squares), GCV-1 to GCV-7 (the GCV and the
fit statistics), LIMIT-3 (``nprune``), W-2 to W-5 (weights), RESP-1 (several
responses) and CORE-3 (the pruning record). The plan's sections "Pruning pass"
and "Fast path" set the method.

Input. The pass works on the M_f terms that the forward pass kept (FWD-11), as
the n x M_f matrix B of their columns in forward order; column 0 is the
intercept, a column of ones (TERM-3). The columns must be linearly independent,
as FWD-11 makes them. Every index here is a pruning index, 0 to M_f - 1
(PRUNE-2); the core maps it to a forward index with ``ForwardRecord.kept``. Y
is (n,) or (n, K); w is (n,), or None, which means w_i = 1 exactly (W-5). A
row with zero weight is not a case, and both functions drop it first (W-3).
N = Σw and τ_N follow W-4 (``_gcv.total_weight``).

Method (PRUNE-9). One Householder QR of the √w-scaled B in the working order
gives R and Z = Qᵀ(√w·Y) (``_linalg.r_factor``). A stage reads the RSS of each
prefix of the order and the RSS of each drop of one term from R and Z
(``prefix_rss``, ``drop_costs``), then moves the removed term with a QR of the
rows that change (``move_column``); no subset is refit.

Numerics. float64 throughout; no function writes into its inputs; no absolute
epsilon; the tie rules are fixed: the term added last when two drops give the
same RSS (PRUNE-3), the earlier offer when two subsets of one size do, and the
smaller size when two sizes give the same GCV (PRUNE-5). Every subset holds
the intercept, so subtracting a constant from a response changes no RSS in
exact arithmetic. Each response is centered first, so that the rounding of
the projections is of the size of the spread of Y, not of its mean (LA-5);
a response that is constant over the cases (Conventions: all its values are
equal) becomes 0 exactly and adds 0 to every sum. For the same reason
``pruning_pass`` centers each column of B after the intercept in the same
way before the QR: a column far from 0 relative to its spread (x near 33 with
a spread of 0.25, say) would otherwise make the factor ill-conditioned, and
the RSS of a near-exact fit would lose about 1e-8 of its relative accuracy
(#84). That is not enough for a term with a linear factor x_j of a
covariate whose mean is far from 0 relative to its spread: the rounding of
x_j·b hides the part that x_j spreads over b, and with b in the subset that
part is all that counts (1e-5 of the RSS at 2^36). So when X and the terms
(``dirs``, ``cuts``) are given and LA-4 shifts such a covariate, the pass
writes the terms exactly in the shifted atoms of LA-4 (u_j = x_j - m_j and
the hinges), factors the columns of those monomials once, and works on exact
eliminations of the terms in each working order (``_Shifted``), and the
final fit applies LA-4 after the shift (``_shifted_fit``). Y enters only through
differences, products and sums, so multiplying Y by a power of 2 (EDGE-6)
multiplies every RSS and GCV by its square and changes no other bit. Memory
is O(n·(M_f + K) + M_f²), and O(n·(P + K) + M_f·P) for P monomials.

Public functions, for ``_core``:

- ``pruning_pass``: the stages, the records and the selected terms (PRUNE-1 to
  PRUNE-7).
- ``final_fit``: the coefficients and statistics of the selected terms
  (PRUNE-8).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pymars import _gcv, _linalg, _terms

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: PRUNE-1: the supported pruning methods.
PMETHODS = ("backward", "none")


class PruningPass(NamedTuple):
    """``pruning_pass(B, Y, w, ...)``, the fields of ``PruningRecord`` (CORE-3)
    and the selected terms.

    ``removed`` (M_f - 1,) int64: the term removed at each stage (PRUNE-4).
    ``rss_per_size``, ``gcv_per_size`` (M_f,): R[m] and GCV(R[m], m) at index
    m - 1. ``subsets`` (M_f, M_f) bool: row m - 1 marks the terms of T[m].
    ``selected_size``: m*. ``selected`` (m*,) int64: the selected terms,
    increasing (PRUNE-5, PRUNE-7).
    """

    removed: IntArray
    rss_per_size: FloatArray
    gcv_per_size: FloatArray
    subsets: BoolArray
    selected_size: int
    selected: IntArray


class FinalFit(NamedTuple):
    """``final_fit(B, Y, selected, w, ...)``: ``coef`` (m*, K), the weighted
    least-squares coefficients, and the ``rss``, ``gcv``, ``rsq`` and ``grsq``
    of the returned model (PRUNE-8)."""

    coef: FloatArray
    rss: float
    gcv: float
    rsq: float
    grsq: float


def _cases(
    B: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray | None, float, float]:
    """Check the input, drop the rows with zero weight (W-3), and return B, Y as
    (n, K), w, N and τ_N (W-4). Complexity: O(n·(M + K))."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[1] < 1 or B.shape[0] < 1:
        raise ValueError(
            f"B must be a 2-D array with at least one column, not {B.shape}"
        )
    n = B.shape[0]
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != n or Y.shape[1] < 1:
        raise ValueError(f"Y must have shape ({n},) or ({n}, K), not {Y.shape}")
    if not (np.isfinite(B).all() and np.isfinite(Y).all()):
        raise ValueError("B and Y must be finite")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (n,) or not (np.isfinite(w).all() and (w >= 0.0).all()):
            raise ValueError(f"w must be finite and nonnegative, with shape ({n},)")
        keep = w > 0.0
        if not keep.any():
            raise ValueError("every weight is zero")
        if not keep.all():
            B, Y, w = B[keep], Y[keep], w[keep]
    if np.any(B[:, 0] != 1.0):
        raise ValueError("column 0 of B must be the intercept, a column of ones")
    N, tau = _gcv.total_weight(B.shape[0], w)
    return B, Y, w, N, tau


def _centered(Y: FloatArray, w: FloatArray | None) -> tuple[FloatArray, FloatArray]:
    """Return Y minus one constant per response, and the constants.

    Each response is shifted by its data value nearest its weighted mean μ (the
    first such case in a tie), then by the weighted mean of the differences.
    The shift a is a data value, so the differences are exact for values
    within a factor of 2 of it, and N·(a - μ)² ≤ TSS, so the rounding of the
    others is of the size of the spread, whatever the weights and the order
    of the cases. A constant response becomes 0 exactly, and its constant is
    its value. Complexity: O(n·K).
    """
    near = np.abs(Y - np.average(Y, axis=0, weights=w)).argmin(axis=0)
    anchor = Y[near, np.arange(Y.shape[1])]
    D = Y - anchor
    mean = np.average(D, axis=0, weights=w)
    return D - mean, anchor + mean


# The kept terms in shifted atoms (LA-4, LA-5). An atom is (j, kind, knot): kind
# 0 is u_j = x_j - m_j, 1 the hinge (x_j - t)₊ and 2 the hinge (t - x_j)₊, the
# key order of LA-4. A monomial is a sorted tuple of atoms, () for the
# intercept, and a term is {monomial index: Fraction}.
_KIND = {_terms.LINEAR: 0, _terms.PLUS: 1, _terms.MINUS: 2}


class _Monomials(NamedTuple):
    """The terms as exact polynomials in the atoms of LA-4. ``keys`` the
    monomials of LA-4 (atoms u_j, (x_j - t)₊ and (t - x_j)₊), ``terms`` (M,),
    term k as {index in keys: Fraction}; ``canon`` each of ``keys`` as an exact
    polynomial in the canonical monomials, whose columns are ``E`` (n, P), not
    centered, and ``cterms`` the terms in them; the cases ``X`` and the shifts
    ``m``. The canonical atoms are u_j and (x_j - t)₊ only, so that the
    columns of a pair at one knot and of u_j are not dependent on the data."""

    keys: list
    terms: list
    canon: list
    E: FloatArray
    cterms: list
    ckeys: list
    X: FloatArray
    m: FloatArray


def _times(poly: dict, parts: dict) -> dict:
    """The product of {monomial: Fraction} and {atom or None: Fraction}, None
    standing for the constant 1. Complexity: O(len(poly)·len(parts))."""
    new: dict = {}
    for e, f in poly.items():
        for a, g in parts.items():
            key = e if a is None else tuple(sorted((*e, a)))
            new[key] = new.get(key, 0) + f * g
    return {e: f for e, f in new.items() if f != 0}


def _atom_column(X: FloatArray, m: FloatArray, atom: tuple) -> FloatArray:
    j, kind, t = atom
    if kind == 0:
        return X[:, j] - m[j]
    return _terms.factor(_terms.PLUS if kind == 1 else _terms.MINUS, X[:, j], t)


def _shifts(X: FloatArray) -> FloatArray:
    """m_j of LA-4: the smallest value of covariate j over the cases when its
    absolute value is larger than the range of x_j, else 0. Complexity: O(n·p)."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    return np.where(np.abs(lo) > hi - lo, lo, 0.0)


def _copies(X: FloatArray, used: list) -> dict:
    """The covariates among ``used`` that are exact affine copies of an earlier
    one at the cases, x_j = a·x_r + b in rational arithmetic with a ≠ 0, as
    {j: (r, a, b)}, r the first of its class. A copy of a covariate is one
    symbol with it (#99), so that a product with a large m_j on the copy
    cancels exactly against the same product on the original (LA-4, LA-5).
    Complexity: O(n·u²) for u used covariates, and O(n) rational operations
    for each candidate pair."""
    out: dict = {}
    reps: list = []
    for j in used:
        x = X[:, j]
        lo, hi = int(np.argmin(x)), int(np.argmax(x))
        if x[lo] == x[hi]:
            continue
        for r in reps:
            xr = X[:, r]
            i0, i1 = int(np.argmin(xr)), int(np.argmax(xr))
            a = (Fraction(float(x[i1])) - Fraction(float(x[i0]))) / (
                Fraction(float(xr[i1])) - Fraction(float(xr[i0]))
            )
            b = Fraction(float(x[i0])) - a * Fraction(float(xr[i0]))
            if a == 0 or not np.allclose(
                x, float(a) * xr + float(b), rtol=1e-12, atol=0
            ):
                continue
            if all(
                Fraction(float(v)) == a * Fraction(float(u)) + b
                for v, u in zip(x, xr, strict=True)
            ):
                out[j] = (r, a, b)
                break
        else:
            reps.append(j)
    return out


def _linear(j: int, m: FloatArray, copies: dict) -> dict:
    """u_j = x_j - m_j in the atoms of the symbol of x_j: for a copy x_j = a·x_r
    + b, a·u_r + (a·m_r + b - m_j). Complexity: O(1)."""
    if j not in copies:
        return {(j, 0, 0.0): Fraction(1)}
    r, a, b = copies[j]
    const = a * Fraction(float(m[r])) + b - Fraction(float(m[j]))
    return {(r, 0, 0.0): a} | ({None: const} if const else {})


def _factor_parts(j: int, kind: int, t: float, m: FloatArray, copies: dict) -> dict:
    """A factor as {atom: Fraction}, None standing for the constant 1: x_j =
    u_j + m_j for a linear factor (in u_r for a copy of x_r, ``_linear``), the
    hinge for a hinge. Complexity: O(1)."""
    if kind == 0:
        parts = _linear(j, m, copies)
        const = parts.pop(None, Fraction(0)) + Fraction(float(m[j]))
        return parts | ({None: const} if const else {})
    return {(j, kind, t): Fraction(1)}


def _canonical(atom: tuple, X: FloatArray, m: FloatArray, copies: dict) -> dict:
    """An atom in the canonical atoms, exactly on the cases: (t - x)₊ =
    (x - t)₊ - u_j + (t - m_j); a hinge that is linear on every case is that
    line, and one that is 0 on every case is 0. Complexity: O(n)."""
    j, kind, t = atom
    if kind == 0:
        return {atom: Fraction(1)}
    x = X[:, j]
    lin = _linear(j, m, copies)
    const = Fraction(t) - Fraction(float(m[j]))  # t - m_j
    if (np.all(x <= t) and kind == 1) or (np.all(x >= t) and kind == 2):
        return {}
    if np.all(x >= t) or np.all(x <= t):  # the line ±(x_j - t) on every case
        sign = 1 if kind == 1 else -1
        out = {q: sign * f for q, f in lin.items()}
        out[None] = out.get(None, 0) - sign * const
        return {q: f for q, f in out.items() if f != 0}
    if kind == 1:
        return {atom: Fraction(1)}
    out = {(j, 1, t): Fraction(1)} | {q: -f for q, f in lin.items()}
    out[None] = out.get(None, 0) + const
    return {q: f for q, f in out.items() if f != 0}


def _monomials(X: FloatArray, dirs: IntArray, cuts: FloatArray) -> _Monomials:
    """Write each term exactly in the atoms of LA-4: a linear factor x_j is
    u_j + m_j, a hinge is its own atom (``_factor_parts``; exact copies of a
    covariate are one symbol, ``_copies``), and each monomial in the canonical
    atoms (``_canonical``). A term with L linear factors of shifted covariates
    has at most 3^L·2^h canonical monomials for h hinges. u_j = x_j - m_j is
    exact in float64 (Sterbenz) when |m_j| ≥ 2·range, and its rounding is of
    the size of the range otherwise, so no column of E carries the factor m_j.
    Complexity: O(M·3^d) rational operations, O(n·d·P) for E, and O(n·p·d) and
    that of ``_copies`` for the atoms."""
    m = _shifts(X)
    copies = _copies(X, np.flatnonzero(np.any(dirs != 0, axis=0)).tolist())
    index: dict = {}
    keys: list = []
    terms: list = []
    for row, cut in zip(dirs, cuts, strict=True):
        poly: dict = {(): Fraction(1)}
        for j in np.flatnonzero(row):
            kind = _KIND[int(row[j])]
            t = float(cut[j]) if kind else 0.0
            poly = _times(poly, _factor_parts(int(j), kind, t, m, copies))
        term = {}
        for key, f in poly.items():
            term[index.setdefault(key, len(index))] = f
            if len(keys) < len(index):
                keys.append(key)
        terms.append(term)
    atoms: dict = {}
    cindex: dict = {}
    cols: list = []
    canon: list = []
    ckeys: list = []
    for key in keys:
        poly = {(): Fraction(1)}
        for atom in key:
            if atom not in atoms:
                atoms[atom] = _canonical(atom, X, m, copies)
            poly = _times(poly, atoms[atom])
        out = {}
        for ckey, f in poly.items():
            if ckey not in cindex:
                cindex[ckey] = len(cols)
                ckeys.append(ckey)
                col = np.ones(X.shape[0])
                for atom in ckey:
                    col = col * _atom_column(X, m, atom)
                cols.append(col)
            out[cindex[ckey]] = f
        canon.append(out)
    cterms = [_compose(t, canon) for t in terms]
    return _Monomials(keys, terms, canon, np.column_stack(cols), cterms, ckeys, X, m)


def _compose(poly: dict, canon: list) -> dict:
    """A polynomial in the monomials of LA-4 as one in the canonical monomials.
    Complexity: O(Σ len(canon[q]))."""
    out: dict = {}
    for q, f in poly.items():
        for c, g in canon[q].items():
            out[c] = out.get(c, 0) + f * g
    return {c: f for c, f in out.items() if f != 0}


def _shifted(X: FloatArray, dirs: IntArray) -> bool:
    """Whether a term has a linear factor of a covariate that LA-4 shifts.
    Complexity: O(n·p + M·p)."""
    m = _shifts(X)
    return bool(np.any((dirs == _terms.LINEAR) & (m != 0.0)))


class _Reduced(NamedTuple):
    """``_reduce``: ``cols`` g_k as {monomial: Fraction}, ``pivots`` their
    pivots, and ``inverse`` T = U⁻¹ as columns {position: Fraction}, where
    c_k = g_k + Σ_l U_lk·g_l, U unit upper triangular."""

    cols: list
    pivots: list
    inverse: list


def _reduce(terms: list, share: FloatArray) -> _Reduced:
    """Exact Gaussian elimination of the terms in their order, in coefficient
    space: g_k is c_k minus its parts along the earlier g_l, so that g_k is 0
    at their pivots, and its pivot is the monomial with the largest share,
    |coefficient|·``share`` (the first by index among equals). So the prefixes
    of g span what the prefixes of the terms span, and a large m_j stays in
    the coefficient of a monomial that an earlier term holds: x_j·b after b is
    u_j·b. Terms of different monomials do not interact, so the work is that
    of the terms that share monomials. Complexity: O(M·f·3^d) rational
    operations, f the largest number of terms on one set of monomials."""
    where: dict = {}
    G: list = []
    pivots: list = []
    inverse: list = []
    for k, c in enumerate(terms):
        v = dict(c)
        upper: dict = {}
        while hits := [where[q] for q in v if q in where]:
            r = min(hits)
            f = v[pivots[r]] / G[r][pivots[r]]
            for q, g in G[r].items():
                v[q] = v.get(q, 0) - f * g
            v = {q: g for q, g in v.items() if g != 0}
            upper[r] = f
        if not v:
            raise ValueError("the columns of B must be linearly independent (FWD-11)")
        p = max(sorted(v), key=lambda q: abs(float(v[q])) * share[q])
        where[p] = k
        G.append(v)
        pivots.append(p)
        t: dict = {k: Fraction(1)}  # T[:, k] = e_k - Σ_r U_rk·T[:, r]
        for r, f in upper.items():
            for i, g in inverse[r].items():
                t[i] = t.get(i, 0) - f * g
        inverse.append({i: g for i, g in t.items() if g != 0})
    return _Reduced(G, pivots, inverse)


def _scaled_columns(red: _Reduced, share: FloatArray, P: int) -> tuple:
    """g_k as a dense float (P, M) matrix, each column times 2^-e_k so that its
    largest share is in [0.5, 1), and the exponents e (M,). Complexity: O(P·M)."""
    G = np.zeros((P, len(red.cols)))
    e = np.zeros(len(red.cols), dtype=np.int64)
    for k, g in enumerate(red.cols):
        q = np.fromiter(g, dtype=np.int64, count=len(g))
        v = np.array([float(g[i]) for i in q])
        e[k] = np.frexp(np.max(np.abs(v) * share[q]))[1]
        G[q, k] = np.ldexp(v, -e[k])
    return G, e


def _inverse_rows(red: _Reduced, e: npt.NDArray[np.int64]) -> FloatArray:
    """T·2^-e as a dense float matrix, each row times a power of 2 so that its
    largest entry is in [0.5, 1); a row's scale changes no drop cost.
    Complexity: O(M²)."""
    M = len(red.cols)
    T = np.zeros((M, M))
    for k, col in enumerate(red.inverse):
        for i, g in col.items():
            T[i, k] = float(g)
    T = np.ldexp(T, -e[None, :])
    return np.ldexp(T, -np.frexp(np.max(np.abs(T), axis=1))[1][:, None])


class _Plain:
    """The factor of PRUNE-9 for the columns of B: one QR, then downdates
    (``_linalg.prefix_rss``, ``drop_costs`` and ``move_column``)."""

    def __init__(self, R: FloatArray, Z: FloatArray, rss: float):
        self.R, self.Z, self.rss = R, Z, rss

    def prefix(self) -> FloatArray:
        return _linalg.prefix_rss(self.Z, self.rss)

    def drops(self, pos: int) -> FloatArray:
        return _linalg.drop_costs(self.R, self.Z, pos)

    def move(self, i: int, j: int) -> None:
        self.R, self.Z = _linalg.move_column(self.R, self.Z, i, j)


class _Shifted:
    """The factor of PRUNE-9 for terms with a linear factor of a shifted
    covariate (LA-4, LA-5): the terms are exact polynomials in the monomials
    of shifted atoms (``_monomials``), whose columns carry no large mean.

    One QR of the √w-scaled centered monomial columns gives R_E and z_E; a QR
    of R_E times the reduced terms of the forward order (``_reduce``) gives an
    orthonormal basis Q_0 of the span of all terms, the RSS of all terms, and
    H = Q_0ᵀR_E. For a working order, the terms are reduced in that order and
    A = H·G is factored, A = Q_A·R_A, with Z = Q_Aᵀ·Q_0ᵀz_E: the prefixes of
    A span what the prefixes of the order span, and no column of A carries a
    factor m_j that the earlier terms hold, so each prefix RSS keeps LA-5. The
    RSS increase of a drop comes from the rows of (R_A·U)⁻¹ = T·R_A⁻¹, with T
    = U⁻¹ exact (``_reduce``), so a term's coefficient m_j never multiplies a
    rounded column. Complexity: O(n·P·(P + K)) once, P ≤ M·3^d monomials, and
    per working order O(M³ + M²·(P + K)) floats and the rational work of
    ``_reduce``."""

    def __init__(self, mono: _Monomials, Yc: FloatArray, w: FloatArray | None):
        n, P = mono.E.shape
        self.terms, self.P = mono.cterms, P
        sw = np.ones(n) if w is None else np.sqrt(w)
        Ec = mono.E.copy()
        rest = [q for q, key in enumerate(mono.ckeys) if key]
        if rest:  # the intercept's monomial stays 1 (every subset holds it)
            Ec[:, rest] = _centered(Ec[:, rest], w)[0]
        As = sw[:, None] * Ec
        self.share = np.sqrt(np.sum(np.square(As), axis=0))
        Q, R = np.linalg.qr(As)
        resid, z = _linalg.orthogonalize(Q, sw[:, None] * Yc)
        red = _reduce(self.terms, self.share)
        G, _ = _scaled_columns(red, self.share, P)
        Q0, _ = np.linalg.qr(R @ G)
        resid0, self.z = _linalg.orthogonalize(Q0, z)
        self.rss = float(np.sum(np.square(resid)) + np.sum(np.square(resid0)))
        self.H = Q0.T @ R
        self.order = np.arange(len(self.terms))
        self._factor(red)

    def _factor(self, red: _Reduced | None = None) -> None:
        if red is None:
            red = _reduce([self.terms[o] for o in self.order], self.share)
        G, self.e = _scaled_columns(red, self.share, self.P)
        Q, self.R = np.linalg.qr(self.H @ G)
        if np.any(np.diag(self.R) == 0.0):
            raise ValueError("the columns of B must be linearly independent (FWD-11)")
        self.Z = Q.T @ self.z
        self.red = red

    def prefix(self) -> FloatArray:
        return _linalg.prefix_rss(self.Z, self.rss)

    def drops(self, pos: int) -> FloatArray:
        # The columns of A are scaled already, so R needs no scaling here.
        W = scipy.linalg.solve_triangular(self.R[:pos, :pos], np.eye(pos))
        rows = _inverse_rows(self.red, self.e)[:pos, :pos]
        rows = np.ldexp(rows, -np.frexp(np.max(np.abs(rows), axis=1))[1][:, None]) @ W
        beta = rows @ self.Z[:pos]
        num = np.square(beta) if beta.ndim == 1 else np.sum(np.square(beta), axis=1)
        return num / np.sum(np.square(rows), axis=1)

    def move(self, i: int, j: int) -> None:
        self.order = np.insert(np.delete(self.order, i), j, self.order[i])
        self._factor()


def _stages(
    f: _Plain | _Shifted, M: int, several: bool
) -> tuple[IntArray, FloatArray, BoolArray]:
    """Run the stages of PRUNE-3 on the factor ``f`` of the forward order.

    Returns ``removed``, R[m] for m = 1, …, M_f and the rows of T[m]. With
    ``several`` false (K = 1), each new working order is offered: a prefix of
    size m replaces T[m] when its RSS is lower than R[m], so the earlier offer
    wins a tie. With ``several`` true (K ≥ 2), T[m] is the set left after
    M_f - m removals. Complexity: O(M_f³·(M_f + K)) in all, at most
    O(M_f²·(M_f + K)) per stage for ``_Plain``; see ``_Shifted`` for it.
    """
    order = np.arange(M)
    offered = f.prefix()
    best = offered.copy()
    subsets = np.tri(M, dtype=bool)  # the prefixes of the starting order
    removed = np.empty(M - 1, dtype=np.int64)
    for pos in range(M, 1, -1):
        # Step 1: the RSS without each term at positions 2, …, pos (0-based 1 to
        # pos - 1); ties go to the term with the largest index.
        drop = offered[pos - 1] + f.drops(pos)[1:]
        tied = 1 + np.flatnonzero(drop == drop.min())
        i = int(tied[np.argmax(order[tied])])
        # Step 2: record the term and move it to position pos.
        removed[M - pos] = order[i]
        f.move(i, pos - 1)
        order = np.insert(np.delete(order, i), pos - 1, order[i])
        # Step 3: offer the new order.
        offered = f.prefix()
        sizes = [pos - 2] if several else np.flatnonzero(offered < best)
        for m in sizes:
            best[m] = offered[m]
            subsets[m] = False
            subsets[m, order[: m + 1]] = True
    return removed, best, subsets


def _terms_of(
    B: FloatArray,
    keep: BoolArray | None,
    X: npt.ArrayLike | None,
    dirs: npt.ArrayLike | None,
    cuts: npt.ArrayLike | None,
) -> _Monomials | None:
    """Check X, dirs and cuts, and return the monomials of the terms when a term
    has a linear factor of a shifted covariate (LA-4), else None."""
    if (X is None) != (dirs is None) or (X is None) != (cuts is None):
        raise ValueError("X, dirs and cuts are given together or not at all")
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float64)
    dirs, cuts = _terms.check_terms(dirs, cuts)
    n = keep.size if keep is not None else B.shape[0]
    if X.ndim != 2 or X.shape[0] != n or not np.isfinite(X).all():
        raise ValueError(
            f"X must be finite and 2-D with one row per case, not {X.shape}"
        )
    if dirs.shape != (B.shape[1], X.shape[1]) or np.any(dirs[0] != 0):
        raise ValueError(
            "dirs and cuts must be (M_f, p), as B and X say, from the intercept"
        )
    if keep is not None:
        X = X[keep]
    return _monomials(X, dirs, cuts) if _shifted(X, dirs) else None


def _positive(w: npt.ArrayLike | None) -> BoolArray | None:
    """The rows with positive weight, before ``_cases`` drops the others."""
    if w is None:
        return None
    wa = np.asarray(w, dtype=np.float64)
    return wa > 0.0 if wa.ndim == 1 and np.all(np.isfinite(wa)) else None


def pruning_pass(
    B: npt.ArrayLike,
    Y: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    penalty: float,
    pmethod: str = "backward",
    nprune: int | None = None,
    X: npt.ArrayLike | None = None,
    dirs: npt.ArrayLike | None = None,
    cuts: npt.ArrayLike | None = None,
) -> PruningPass:
    """Run the pruning pass on the kept forward terms (PRUNE-1 to PRUNE-7).

    PRUNE-3: with one response (Y 1-D or with one column), the stages offer the
    prefixes of a working order, leaps-style, and T[m] is the subset of size m
    with the lowest RSS among those offered, so the sets need not be nested;
    with K ≥ 2 columns, the pass is plain backward elimination on the summed
    RSS. K counts the columns of Y, constant ones included. At every stage the
    removed term is the one whose drop gives the lowest RSS, ties going to the
    term with the largest index, and the intercept (term 0) is never removed.
    PRUNE-4: the records; ``gcv_per_size`` follows GCV-2 with N = Σw (W-2),
    and it is +∞ for a degenerate fit, N ≤ 1 or every response constant
    (GCV-7). PRUNE-5: with ``pmethod="backward"`` the selected size is the
    smallest m ≤ min(M_f, ``nprune``) with the lowest GCV, and the selected
    terms are T[m*]. PRUNE-6: ``nprune`` changes only that range. PRUNE-7: with
    ``pmethod="none"`` the records are the same, m* = min(``nprune``, M_f), and
    the selected terms are 0, …, m* - 1.

    B (n, M_f), Y and w are as in the module docstring, with M_f ≤ n; the
    penalty d is resolved (GCV-1, GCV-4) and ``nprune`` is None or an integer
    ≥ 1 (LIMIT-3). ``X`` (n, p), ``dirs`` and ``cuts`` (M_f, p) are optional,
    and all three or none: the covariates and the terms that made B (TERM-3).
    When a term has a linear factor of a covariate that LA-4 shifts, the
    pass works on the terms in shifted atoms (``_Shifted``), so that every RSS
    keeps LA-5 whatever the mean of x; else on B itself (``_Plain``). Raises
    ValueError for input outside these ranges. Complexity: O(n·M_f·(M_f + K))
    for the factor and O(M_f³·(M_f + K)) for the stages, at most O(n·M_f²) per
    stage for K ≤ M_f (PRUNE-9); with shifted terms, O(n·P·(P + K)) once for
    P ≤ M_f·3^d monomials and O(M_f³ + M_f²·(P + K)) per stage, plus the
    exact elimination of ``_reduce``; memory O(n·(P + K) + M_f·P).
    """
    if pmethod not in PMETHODS:
        raise ValueError(f"pmethod must be 'backward' or 'none', not {pmethod!r}")
    keep = _positive(w)
    B, Y, w, N, tau = _cases(B, Y, w)
    M = B.shape[1]
    m_max = _gcv.nprune_limit(M, nprune)
    mono = _terms_of(B, keep, X, dirs, cuts)
    Yc = _centered(Y, w)[0]
    if mono is not None:
        factor: _Plain | _Shifted = _Shifted(mono, Yc, w)
    else:
        Bc = B.copy()
        if M > 1:
            Bc[:, 1:] = _centered(B[:, 1:], w)[0]  # the same spans (LA-5, #84)
        R, Z, rss = _linalg.r_factor(Bc, Yc, w)
        # fit_mars cannot reach this: FWD-11 drops a column constant over the cases.
        if np.any(np.diag(R) == 0.0):
            raise ValueError("the columns of B must be linearly independent (FWD-11)")
        factor = _Plain(R, Z, rss)
    removed, rss_per_size, subsets = _stages(factor, M, several=Y.shape[1] >= 2)
    gcv_per_size = _gcv.gcv(rss_per_size, np.arange(1, M + 1), penalty, N, tau)
    if _gcv.is_degenerate(Y, N):
        gcv_per_size = np.full(M, np.inf)
    if pmethod == "backward":
        size = int(np.argmin(gcv_per_size[:m_max])) + 1
        selected = np.flatnonzero(subsets[size - 1])
    else:
        size = m_max
        selected = np.arange(size)
    return PruningPass(
        removed, rss_per_size, gcv_per_size, subsets, size, selected.astype(np.int64)
    )


def final_fit(
    B: npt.ArrayLike,
    Y: npt.ArrayLike,
    selected: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    penalty: float,
    X: npt.ArrayLike | None = None,
    dirs: npt.ArrayLike | None = None,
    cuts: npt.ArrayLike | None = None,
) -> FinalFit:
    """Return the coefficients and statistics of the selected terms (PRUNE-8).

    ``coef`` holds the weighted least-squares coefficients of Y on the columns
    ``selected`` of B, in increasing term order, one column per response, with
    LA-4 for dependent columns (``_linalg.lm_fit``). The fit is of the
    centered responses, and the intercept's coefficient takes back their
    constants, so a constant response gets its value there and 0 for the other
    terms, exactly. ``rss`` is the weighted RSS of the centered fit: that of
    the returned coefficients, up to the rounding of the intercept's
    coefficient. ``gcv``, ``rsq`` and ``grsq`` follow GCV-2 and GCV-5 to GCV-7
    with that RSS, M = m* and N = Σw. With ``pmethod="none"`` and m* < M_f the
    selected terms are not T[m*], so ``rss`` differs from
    ``rss_per_size[m* - 1]`` (PRUNE-7).

    B, Y and w are as for ``pruning_pass``; ``selected`` is an increasing
    integer array that starts with the intercept, 0. Raises ValueError
    otherwise. With ``X``, ``dirs`` and ``cuts`` of all M_f terms (as for
    ``pruning_pass``) and a selected term with a linear factor of a covariate
    that LA-4 shifts, the fit applies LA-4 after the exact shift and ``rss`` is
    the RSS of the projection (``_shifted_fit``), so that a large mean of x
    neither drops a term nor costs the RSS its accuracy (LA-5).
    Complexity: O(n·m*·(m* + K)) time, O(n·(m* + K)) memory; with shifted
    terms see ``_shifted_fit``.
    """
    keep = _positive(w)
    B, Y, w, N, tau = _cases(B, Y, w)
    sel = np.asarray(selected)
    if (
        sel.ndim != 1
        or sel.size == 0
        or not np.issubdtype(sel.dtype, np.integer)
        or sel[0] != 0
        or np.any(np.diff(sel) <= 0)
        or sel[-1] >= B.shape[1]
    ):
        raise ValueError(
            f"selected must be increasing term indices below {B.shape[1]}, from 0"
        )
    m = sel.size
    mono = _terms_of(B, keep, X, dirs, cuts)
    if mono is not None:
        mono = mono._replace(
            terms=[mono.terms[k] for k in sel], cterms=[mono.cterms[k] for k in sel]
        )
    Yc, constants = _centered(Y, w)
    if mono is not None:
        coef, rss = _shifted_fit(mono, Yc, w)
    else:
        fit = _linalg.lm_fit(B[:, sel], Yc, w)
        coef, rss = fit.coef, fit.rss
    coef[0] += constants  # column 0 is the intercept, a column of ones
    gcv = _gcv.gcv(rss, m, penalty, N, tau)
    if _gcv.is_degenerate(Y, N):
        gcv = np.inf
    tss = _gcv.tss(Y, w)
    rsq = _gcv.rsq(rss, tss, m)
    grsq = _gcv.grsq(rss, tss, m, penalty, N, tau)
    return FinalFit(coef, rss, float(gcv), rsq, grsq)


def _subtract(r: dict, f: Fraction, row: dict) -> dict:
    out = dict(r)
    for q, g in row.items():
        out[q] = out.get(q, 0) - f * g
    return {q: g for q, g in out.items() if g != 0}


def _shifted_fit(
    mono: _Monomials, Yc: FloatArray, w: FloatArray | None
) -> tuple[FloatArray, float]:
    """Least squares of Yc on the terms of ``mono`` with LA-4 after the exact
    shift (PRUNE-8): the coefficients (M, K), 0 for a dependent term, and
    the RSS of the projection.

    LA-4: the terms are taken in order. A term is dependent when its expansion
    is in the span of those of the kept terms. Else the reduced echelon form
    of the kept terms with it (monomials by degree, highest first, then by
    their atoms) has one new pivot p, whose reduced row is the term's new
    part, and the term is dependent when the w-norm of the part of the new
    part orthogonal to the other reduced rows is less than 1e-7·‖p‖_w, or,
    when p is 0 at every case, less than 1e-7 times the w-norm of the new
    part, or when the new part is 0 at every case. The orthogonal part comes
    from the rows reduced again as in ``_reduce``, so never from a column
    that carries m_j. The coefficients solve the least
    squares on the reduced kept terms, g, and b = T·g with T = U⁻¹ exact; the
    intercept's takes the constants of the centered monomials.
    Complexity: O(n·m²·(m + P)) and O(m²·f·3^d) rational operations for m
    terms and P monomials.
    """
    n, P = mono.E.shape
    sw = np.ones(n) if w is None else np.sqrt(w)
    Ec = mono.E.copy()
    shift = np.zeros(P)
    rest = [q for q, key in enumerate(mono.ckeys) if key]
    if rest:
        Ec[:, rest], shift[rest] = _centered(Ec[:, rest], w)
    As = sw[:, None] * Ec
    share = np.sqrt(np.sum(np.square(As), axis=0))
    Ew = sw[:, None] * mono.E

    def pivot_norm(q: int) -> float:  # the w-norm of a monomial of LA-4
        col = sw.copy()
        for atom in mono.keys[q]:
            col = col * _atom_column(mono.X, mono.m, atom)
        return float(scipy.linalg.norm(col))

    rank = [(-len(key), key) for key in mono.keys]
    rows: dict = {}  # the reduced echelon form: pivot -> row, pivot coefficient 1
    kept: list = []
    basis: list = []
    for k, c in enumerate(mono.terms):
        v = dict(c)
        for p, r in rows.items():  # the rows are 0 at each other's pivots
            if v.get(p, 0):
                v = _subtract(v, v[p], r)
        if not v:
            continue
        p = min(v, key=lambda q: rank[q])
        new = {q: f / v[p] for q, f in v.items()}
        others = {
            p2: _subtract(r, r[p], new) if r.get(p, 0) else r for p2, r in rows.items()
        }
        cnew = _compose(new, mono.canon)
        try:  # an exact dependence on the cases leaves cnew in the span
            red = _reduce(
                [*(_compose(r, mono.canon) for r in others.values()), cnew], share
            )
        except ValueError:
            continue
        G, e = _scaled_columns(red, share, P)
        A = As @ G
        Q, _ = np.linalg.qr(A[:, :-1])
        perp, _ = _linalg.orthogonalize(Q, A[:, -1])
        norm = math.ldexp(float(scipy.linalg.norm(perp)), int(e[-1]))
        base = pivot_norm(p)
        if base == 0.0:
            base = float(
                scipy.linalg.norm(Ew[:, list(cnew)] @ [float(f) for f in cnew.values()])
            )
        if base == 0.0 or norm < _linalg.LM_TOL * base:
            continue
        rows = {**others, p: new}
        kept.append(k)
        basis.append(mono.cterms[k])
    red = _reduce(basis, share)
    G, e = _scaled_columns(red, share, P)
    Qa, Ra = np.linalg.qr(As @ G)
    resid, z = _linalg.orthogonalize(Qa, sw[:, None] * Yc)
    gamma = np.ldexp(scipy.linalg.solve_triangular(Ra, z), -e[:, None])
    T = np.zeros((len(kept), len(kept)))
    for j, col in enumerate(red.inverse):
        for i, f in col.items():
            T[i, j] = float(f)
    coef = np.zeros((len(mono.terms), Yc.shape[1]))
    coef[kept] = T @ gamma
    coef[0] -= shift @ (np.ldexp(G, e[None, :]) @ gamma)  # Ec = E - 1·shiftᵀ
    return coef, float(np.sum(np.square(resid)))
