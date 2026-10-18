"""Compiled loops of the knot scan (numba), bit-equal to the numpy code of
``_scan``.

Spec: ``docs/algorithm.md``, "Linear algebra contract" (LA-2, LA-3, LA-5),
through ``_scan``, whose module docstring derives the sums. These kernels
change no value: each one does the additions and products of its numpy
counterpart in ``_scan`` in the same order, so every fit is bit-equal with
or without numba (``validation/bench/hash_fits.py`` checks this over 533
fits on both paths).

- ``hv`` replaces ``_scan._hv`` and the gather ``Q[o]`` of the caller: one
  pass from the last row to the first, which forms b·v, its blocked suffix
  sum P, g·P and the blocked suffix sum of that, and reads the second sum at
  the splits. It needs no n x c temporary.
- ``norms`` and ``centered`` replace ``_scan._norms`` and ``_scan._centered``
  with the same steps and O(n) temporaries.

The order of the sums is that of ``_scan._suffix``: from the end, rows are
added in blocks of B = isqrt(m - 1) + 1, block totals are carried from block
to block (the carry of block 1 is the total of block 0 itself, then
carry + total), and every output is its in-block partial sum plus the carry
of its block (0.0 for block 0). numba compiles without fastmath, so LLVM
neither reassociates nor contracts a·b + c into a fused multiply-add; the
test ``test_the_kernels_round_every_product`` checks that on a case where a
contraction would change the result.

``AVAILABLE`` is True when numba imports; ``ENABLED`` (default
``AVAILABLE``) selects the kernels in ``_scan``; tests and
``hash_fits.py --no-numba`` set it to False to run the numpy code. The
compiled code is cached on disk (``cache=True``), so only the first call in
a fresh environment pays the compilation.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba
except ImportError:  # numba is optional; ``_scan`` then uses its numpy code
    numba = None

AVAILABLE = numba is not None
ENABLED = AVAILABLE


def _blocks(m: int) -> int:
    """The block size of ``_scan._suffix`` for m rows."""
    return math.isqrt(max(m - 1, 0)) + 1


def _suffix1(a, B):
    """``_scan._suffix`` of a 1-D array with block size B. Complexity: O(m)."""
    m = a.shape[0]
    out = np.empty(m)
    carry = 0.0
    within = 0.0
    for p in range(m):
        q = m - 1 - p
        if p % B == 0:
            if p == B:
                carry = within
            elif p > B:
                carry = carry + within
            within = a[q]
        else:
            within = within + a[q]
        out[q] = within + carry
    return out


def _hv(g, b, Q, q0, E, o, split, B1, B2):
    """hᵀV at every split, V = [Q[o, q0:], E] (``_scan._hv``). g (n - 1,),
    b (n,), E (n, K) and o (n,) are in the order of x; Q is in the caller's
    order and is read at the rows o. Complexity: O(n·c + k·log k)."""
    n = b.shape[0]
    nq = Q.shape[1] - q0
    c = nq + E.shape[1]
    k = split.shape[0]
    out = np.empty((k, c))
    if k == 0:
        return out
    rank = np.argsort(split)
    nxt = k - 1  # the largest split not yet read
    w1 = np.zeros(c)
    c1 = np.zeros(c)
    w2 = np.zeros(c)
    c2 = np.zeros(c)
    for q in range(n - 1, -1, -1):
        p1 = n - 1 - q  # the position from the end, of P (n rows)
        new1 = p1 % B1 == 0
        if new1 and p1 == B1:
            for j in range(c):
                c1[j] = w1[j]
        elif new1 and p1 > B1:
            for j in range(c):
                c1[j] = c1[j] + w1[j]
        if q >= 1:  # row i = q - 1 of the second sum (n - 1 rows), g_i·P(q)
            p2 = p1
            new2 = p2 % B2 == 0
            if new2 and p2 == B2:
                for j in range(c):
                    c2[j] = w2[j]
            elif new2 and p2 > B2:
                for j in range(c):
                    c2[j] = c2[j] + w2[j]
        bq = b[q]
        row = o[q]
        gq = g[q - 1] if q >= 1 else 0.0
        for j in range(c):
            v = bq * (Q[row, q0 + j] if j < nq else E[q, j - nq])
            w1[j] = v if new1 else w1[j] + v
            if q >= 1:
                y = gq * (w1[j] + c1[j])
                w2[j] = y if new2 else w2[j] + y
        while nxt >= 0 and split[rank[nxt]] == q:
            for j in range(c):
                out[rank[nxt], j] = w2[j] + c2[j]
            nxt -= 1
    return out


def _norms(g, b, split, Bn, Bm):
    """‖h‖² at every split (``_scan._norms``). Complexity: O(n)."""
    m = g.shape[0]
    W = _suffix1(b * b, Bn)[1:]
    H = _suffix1(g * W, Bm)
    Hn = np.empty(m)
    Hn[: m - 1] = H[1:]
    Hn[m - 1] = 0.0
    F = _suffix1(g * (2.0 * Hn + g * W), Bm)
    return F[split - 1]


def _centered(g, w, split, Bn, Bm):
    """D of the intercept parent at every split (``_scan._centered``).
    Complexity: O(n)."""
    n = w.shape[0]
    Wsuf = _suffix1(w, Bn)
    Wpre = _suffix1(w[::-1].copy(), Bn)[::-1]
    S1 = _suffix1(g * Wsuf[1:], Bm)
    SSW = np.empty(n)
    SSW[: n - 1] = _suffix1(w[:-1] * (S1 * S1) / (Wsuf[1:] * Wsuf[:-1]), Bm)
    SSW[n - 1] = 0.0
    k = split.shape[0]
    out = np.empty(k)
    for i in range(k):
        s = split[i]
        Ws, Wp, S = Wsuf[s], Wpre[s - 1], S1[s - 1]
        out[i] = SSW[s] + Wp * (S * S) / (Ws * (Wp + Ws))
    return out


if AVAILABLE:
    _jit = numba.njit(cache=True, fastmath=False, nogil=True)
    _suffix1 = _jit(_suffix1)
    _hv = _jit(_hv)
    _norms = _jit(_norms)
    _centered = _jit(_centered)


def hv(g, b, Q, q0, E, o, split):
    """hᵀV (k, c) for V = [Q[o, q0:], E] at every split, as ``_scan._hv``
    gives for the gathered V. Complexity: O(n·c + k·log k) time, O(c + k·c)
    memory."""
    n = b.shape[0]
    return _hv(g, b, Q, q0, E, o, split, _blocks(n), _blocks(n - 1))


def norms(g, b, split):
    """‖h‖² at every split, as ``_scan._norms``. Complexity: O(n)."""
    n = b.shape[0]
    if split.shape[0] == 0:  # also n = 1, where g is empty
        return np.empty(0)
    return _norms(g, b, split, _blocks(n), _blocks(n - 1))


def centered(g, w, split):
    """D of the intercept parent at every split, as ``_scan._centered``.
    Complexity: O(n)."""
    n = w.shape[0]
    if split.shape[0] == 0:  # also n = 1, where g is empty
        return np.empty(0)
    return _centered(g, w, split, _blocks(n), _blocks(n - 1))
