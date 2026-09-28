"""Terms of a MARS model: the ``dirs`` and ``cuts`` arrays and the basis matrix.

Spec: ``docs/algorithm.md``, "Terms" (TERM-1 to TERM-6). The plan's section
"Term structure" sets the layout, which is that of earth's ``dirs`` and
``cuts`` outputs.

A model with M terms on p covariates is two (M, p) arrays. ``dirs`` (int8)
holds one code per term and covariate: ``ABSENT`` (0) when the covariate is
not in the term, ``PLUS`` (+1) for the factor (x - c)₊, ``MINUS`` (-1) for
(c - x)₊ and ``LINEAR`` (2) for x. ``cuts`` (float64) holds the knots c, and
0.0 wherever the code is 0 or 2 (TERM-2). Row 0 is the intercept, with every
code 0 (TERM-1). One row describes a whole product term, so no covariate
appears twice in a term (TERM-4).

Numerics. float64 throughout; no function writes into its inputs; nothing is
clipped, so a term extrapolates outside the range of the training data
(TERM-3). Memory is O(n·M) for the basis matrix and O(M·p) for the terms.

Public names, for ``_scan``, ``_forward``, ``_pruning``, ``_core`` and the
estimators:

- ``ABSENT``, ``PLUS``, ``MINUS``, ``LINEAR``: the codes (TERM-1).
- ``intercept_terms``, ``check_terms``, ``child_term``: build and check term
  tables (TERM-1, TERM-2, TERM-4, TERM-6).
- ``factor``, ``basis_matrix``: evaluate factors and terms (TERM-3).
- ``term_degrees``, ``term_variables``: degree and covariates (TERM-4).
- ``term_label``, ``term_labels``: labels for ``summary()`` (TERM-5).
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Int8Array = npt.NDArray[np.int8]

#: TERM-1: the covariate is not in the term.
ABSENT = 0
#: TERM-1: the factor (x - c)₊.
PLUS = 1
#: TERM-1: the factor (c - x)₊.
MINUS = -1
#: TERM-1: the linear factor x.
LINEAR = 2
_CODES = (ABSENT, PLUS, MINUS, LINEAR)
_FACTOR_CODES = (PLUS, MINUS, LINEAR)


def intercept_terms(p: int) -> tuple[Int8Array, FloatArray]:
    """Return ``dirs`` and ``cuts`` of the intercept-only model on p covariates.

    Both arrays have shape (1, p) and hold zeros: row 0 is the intercept, with
    every code 0 (TERM-1, TERM-2). Complexity: O(p).
    """
    p = operator.index(p)
    if p < 1:
        raise ValueError(f"p must be at least 1, not {p}")
    return np.zeros((1, p), dtype=np.int8), np.zeros((1, p), dtype=np.float64)


def check_terms(
    dirs: npt.ArrayLike, cuts: npt.ArrayLike
) -> tuple[Int8Array, FloatArray]:
    """Check a term table and return it as int8 ``dirs`` and float64 ``cuts``.

    ``dirs`` and ``cuts`` must be 2-D arrays of one shape (M, p) (TERM-1).
    Every code must be 0, +1, -1 or 2 exactly (a float such as 1.0 is
    accepted); every cut must be finite, and 0.0 wherever the code is 0 or 2
    (TERM-2). Row 0 need not be the intercept: a row of zeros anywhere is an
    intercept column. Raises ValueError otherwise. The returned ``dirs`` is a
    new array; the returned ``cuts`` may share memory with the input, and
    neither is written to here. Complexity: O(M·p).
    """
    d = np.asarray(dirs)
    c = np.asarray(cuts, dtype=np.float64)
    if d.ndim != 2 or c.shape != d.shape:
        raise ValueError(
            f"dirs and cuts must be 2-D arrays of one shape, not {d.shape}, {c.shape}"
        )
    if not np.isin(d, _CODES).all():
        raise ValueError("every code in dirs must be 0, 1, -1 or 2")
    d8 = d.astype(np.int8)
    if not np.isfinite(c).all():
        raise ValueError("every cut must be finite")
    if np.any(c[(d8 == ABSENT) | (d8 == LINEAR)] != 0.0):
        raise ValueError("cuts must be 0.0 wherever dirs is 0 or 2 (TERM-2)")
    return d8, c


def child_term(
    dirs_row: npt.ArrayLike,
    cuts_row: npt.ArrayLike,
    variable: int,
    code: int,
    cut: float = 0.0,
) -> tuple[Int8Array, FloatArray]:
    """Return the row of the term that multiplies a parent term by one factor.

    ``dirs_row`` and ``cuts_row`` (p,) describe the parent. The new term is the
    parent times the factor of ``code`` (``PLUS``, ``MINUS`` or ``LINEAR``) in
    covariate ``variable`` with knot ``cut``; its column is the parent's column
    times ``factor(code, X[:, variable], cut)`` (TERM-6). The covariate must not
    be in the parent, since no covariate appears twice in a term (TERM-4). A
    linear factor stores the cut 0.0 whatever ``cut`` is (TERM-2); a hinge needs
    a finite cut. Raises ValueError otherwise. The inputs are not changed; the
    rows returned are new. Complexity: O(p).
    """
    d = np.array(dirs_row, dtype=np.int8)
    c = np.array(cuts_row, dtype=np.float64)
    if d.ndim != 1 or c.shape != d.shape:
        raise ValueError("dirs_row and cuts_row must be 1-D arrays of one shape")
    variable = operator.index(variable)
    if not 0 <= variable < d.shape[0]:
        raise ValueError(f"variable must be in 0 to {d.shape[0] - 1}, not {variable}")
    if code not in _FACTOR_CODES:
        raise ValueError(f"code must be 1, -1 or 2, not {code}")
    if d[variable] != ABSENT:
        raise ValueError(f"covariate {variable} is already in the parent term (TERM-4)")
    cut = float(cut)
    if code != LINEAR and not np.isfinite(cut):
        raise ValueError("a hinge needs a finite cut")
    d[variable] = code
    c[variable] = 0.0 if code == LINEAR else cut
    return d, c


def factor(code: int, x: npt.ArrayLike, cut: float) -> FloatArray:
    """Return one factor of a term at the values x of its covariate (TERM-3).

    f(+1, x, c) = (x - c)₊, f(-1, x, c) = (c - x)₊ and f(2, x, c) = x, at every
    finite x, with nothing clipped. The result is a new float64 array of the
    shape of x. Raises ValueError for another code. Complexity: O(len(x)).
    """
    x = np.asarray(x, dtype=np.float64)
    if code == PLUS:
        return np.maximum(x - cut, 0.0)
    if code == MINUS:
        return np.maximum(cut - x, 0.0)
    if code == LINEAR:
        return x.copy()
    raise ValueError(f"code must be 1, -1 or 2, not {code}")


def basis_matrix(
    X: npt.ArrayLike, dirs: npt.ArrayLike, cuts: npt.ArrayLike
) -> FloatArray:
    """Return the basis matrix B with B[i, k] = B_k(x_i) (TERM-3).

    X is (n, p); ``dirs`` and ``cuts`` are (M, p) and are checked by
    ``check_terms``. Term k is the product of the factors
    f(dirs[k, j], X[:, j], cuts[k, j]) over the j with dirs[k, j] ≠ 0, taken in
    increasing j; the empty product, for the intercept, is 1. Outside the range
    of the training data the formula still holds, since nothing is clipped. B
    is a new (n, M) float64 array in Fortran order, so that each column is
    contiguous. Raises ValueError when X is not 2-D or has not p columns.
    Complexity: O(n·Σ_k deg_k) ≤ O(n·M·max_degree) time, O(n·M) memory.
    """
    d, c = check_terms(dirs, cuts)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != d.shape[1]:
        raise ValueError(
            f"X must be 2-D with {d.shape[1]} columns, not shape {X.shape}"
        )
    B = np.ones((X.shape[0], d.shape[0]), dtype=np.float64, order="F")
    for k, j in zip(*np.nonzero(d), strict=True):
        B[:, k] *= factor(int(d[k, j]), X[:, j], float(c[k, j]))
    return B


def term_degrees(dirs: npt.ArrayLike) -> IntArray:
    """Return the degree of each term: its number of nonzero codes (TERM-4).

    A linear factor counts as one, and the intercept has degree 0. ``dirs`` is
    (M, p); the result is a new int64 array of shape (M,). Complexity: O(M·p).
    """
    d = np.asarray(dirs)
    if d.ndim != 2:
        raise ValueError(f"dirs must be 2-D, not {d.ndim}-D")
    return np.count_nonzero(d, axis=1).astype(np.int64)


def term_variables(dirs_row: npt.ArrayLike) -> IntArray:
    """Return the covariates of one term, in increasing order (TERM-4).

    ``dirs_row`` is one row (p,) of ``dirs``; the result is a new int64 array,
    empty for the intercept. Complexity: O(p).
    """
    d = np.asarray(dirs_row)
    if d.ndim != 1:
        raise ValueError(f"dirs_row must be 1-D, not {d.ndim}-D")
    return np.flatnonzero(d).astype(np.int64)


def term_label(
    dirs_row: npt.ArrayLike,
    cuts_row: npt.ArrayLike,
    feature_names: Sequence[str] | None = None,
) -> str:
    """Return the label of one term, as ``summary()`` shows it (TERM-5).

    The label joins the labels of the factors with ``*``, in increasing
    covariate order: ``h(NAME-C)`` for +1, ``h(C-NAME)`` for -1 and ``NAME`` for
    2, where C is the knot formatted with ``.6g``. NAME is
    ``feature_names[j]`` when names are given, else ``x`` followed by the
    0-based column index. The intercept's label is ``(Intercept)``. A negative
    knot gives labels such as ``h(x0--1.5)``, as in earth. Complexity: O(p).
    """
    d = np.asarray(dirs_row)
    c = np.asarray(cuts_row, dtype=np.float64)
    if d.ndim != 1 or c.shape != d.shape:
        raise ValueError("dirs_row and cuts_row must be 1-D arrays of one shape")
    if feature_names is not None and len(feature_names) != d.shape[0]:
        raise ValueError(
            f"feature_names must have {d.shape[0]} names, not {len(feature_names)}"
        )
    parts = []
    for j in np.flatnonzero(d):
        name = f"x{j}" if feature_names is None else str(feature_names[j])
        code = int(d[j])
        if code == PLUS:
            parts.append(f"h({name}-{float(c[j]):.6g})")
        elif code == MINUS:
            parts.append(f"h({float(c[j]):.6g}-{name})")
        elif code == LINEAR:
            parts.append(name)
        else:
            raise ValueError(f"code must be 0, 1, -1 or 2, not {code}")
    return "*".join(parts) if parts else "(Intercept)"


def term_labels(
    dirs: npt.ArrayLike,
    cuts: npt.ArrayLike,
    feature_names: Sequence[str] | None = None,
) -> list[str]:
    """Return ``term_label`` of every row of ``dirs`` and ``cuts`` (TERM-5).

    Complexity: O(M·p).
    """
    d, c = check_terms(dirs, cuts)
    return [term_label(d[k], c[k], feature_names) for k in range(d.shape[0])]
