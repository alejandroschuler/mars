"""Compiled loops of the knot scan (numba).

Spec: ``docs/algorithm.md``, "Linear algebra contract" (LA-2, LA-3, LA-5),
through ``_scan``, whose module docstring derives the sums and the error
bounds. Each kernel has a numpy counterpart in ``_scan``, which runs when
numba is missing (the ``nonumba`` CI job). Bit-identity of the two is not a
rule (#145): the suffix sums keep the blocked order of ``_scan._suffix``, so
the error bounds keep their form, and the two forms agree within those
bounds. So the two paths give the same models except at near-ties (STOP-7),
where they can choose different terms.

- ``hv`` replaces ``_scan._hv`` and the gather ``Q[o]`` of the caller: one
  pass from the last row to the first, which forms b·v, its blocked suffix
  sum P, g·P and the blocked suffix sum of that, and reads the
  second sum at the splits. It needs no n x c temporary.
- ``norms`` and ``centered`` replace ``_scan._norms`` and ``_scan._centered``
  with the same steps and O(n) temporaries.
- ``prep`` gives the scaled gaps and parent (``_scan._to_unit``), and
  ``finish`` (kernel B) gives rho, the reduction, both error bounds, the
  floor and the zero mask per knot in one pass, in place of about 30 numpy
  calls (``_scan._finish``).
- ``cgs2`` is ``_linalg.cgs2``'s Gram-Schmidt of one column against Q: a pass
  for the new columns' coefficients, then one fused pass that forms v - Q·C,
  its norm and Qᵀ(v - Q·C), and a second such pass only when DGKS asks for it.
  Each row's dot product runs in column order, 4 rows at a time so that the
  chains overlap. It reads Q 1 or 2 times where two BLAS matrix-vector
  products read it twice per pass (measured, n = 100,000 and 12 columns:
  0.53 ms against 1.1 ms; at 10,000 the two are equal).

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
#: The rows of Q that ``hv`` gathers at a time.
TILE = 128
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


def _rank(split):
    """The order of ``split`` from the smallest to the largest: with no sort
    when it is monotone, as ``_knots.candidate_knots`` gives it (largest
    first). Complexity: O(k) when monotone, else O(k·log k)."""
    k = split.shape[0]
    up = True
    down = True
    for i in range(1, k):
        if split[i] < split[i - 1]:
            up = False
        elif split[i] > split[i - 1]:
            down = False
    if up:
        return np.arange(k)
    if down:
        return np.arange(k - 1, -1, -1)
    return np.argsort(split)


def _hv(g, b, Q, q0, E, o, split, B1, B2):
    """hᵀV at every split, V = [Q[o, q0:], E] (``_scan._hv``). g (n - 1,),
    b (n,), E (n, K) and o (n,) are in the order of x; Q is in the caller's
    order and is read at the rows o. The blocks of the two suffix sums are
    counted down, with no modulo per row, and the loops over the columns have
    no branch.
    Complexity: O(n·c), plus O(k·log k) when ``split`` is not monotone."""
    n = b.shape[0]
    nq = Q.shape[1] - q0
    nE = E.shape[1]
    c = nq + nE
    k = split.shape[0]
    out = np.empty((k, c))
    if k == 0:
        return out
    rank = _rank(split)
    nxt = k - 1  # the largest split not yet read
    w1 = np.zeros(c)
    c1 = np.zeros(c)
    w2 = np.zeros(c)
    c2 = np.zeros(c)
    left1 = 0  # rows left in the current block of P (n rows)
    blocks1 = 0  # blocks of P started so far
    left2 = 0  # the same for the second sum (n - 1 rows)
    blocks2 = 0
    # The rows of Q are gathered TILE rows at a time into buf, a loop of
    # independent loads, so the cache misses of the random rows overlap.
    buf = np.empty((TILE, nq))
    left_t = 0  # rows of buf not yet used
    t = 0  # the row of buf of row q
    for q in range(n - 1, -1, -1):
        if left_t == 0:
            left_t = min(TILE, q + 1)
            for i in range(left_t):
                row = o[q - i]
                for j in range(nq):
                    buf[i, j] = Q[row, q0 + j]
            t = 0
        new1 = left1 == 0
        if new1:  # carry: the total of block 0, then carry + total
            if blocks1 == 1:
                for j in range(c):
                    c1[j] = w1[j]
            elif blocks1 > 1:
                for j in range(c):
                    c1[j] = c1[j] + w1[j]
            blocks1 += 1
            left1 = B1
        left1 -= 1
        bq = b[q]
        if new1:
            for j in range(nq):
                w1[j] = bq * buf[t, j]
            for j in range(nE):
                w1[nq + j] = bq * E[q, j]
        else:
            for j in range(nq):
                w1[j] = w1[j] + bq * buf[t, j]
            for j in range(nE):
                w1[nq + j] = w1[nq + j] + bq * E[q, j]
        if q >= 1:  # row i = q - 1 of the second sum, g_i·P(q)
            new2 = left2 == 0
            if new2:
                if blocks2 == 1:
                    for j in range(c):
                        c2[j] = w2[j]
                elif blocks2 > 1:
                    for j in range(c):
                        c2[j] = c2[j] + w2[j]
                blocks2 += 1
                left2 = B2
            left2 -= 1
            gq = g[q - 1]
            if new2:
                for j in range(c):
                    w2[j] = gq * (w1[j] + c1[j])
            else:
                for j in range(c):
                    w2[j] = w2[j] + gq * (w1[j] + c1[j])
        while nxt >= 0 and split[rank[nxt]] == q:
            for j in range(c):
                out[rank[nxt], j] = w2[j] + c2[j]
            nxt -= 1
        t += 1
        left_t -= 1
    return out


def _scale(a, out, start):
    """out = a[1:] - a[:-1] when ``start`` is 1, else a, each times the
    power of 2^e that puts its largest |value| in [0.5, 1). While 2^e is
    normal, the product by it is the correctly rounded a·2^e, which is what
    ``np.ldexp`` gives; otherwise ``math.ldexp`` scales each value.
    Complexity: O(n)."""
    m = out.shape[0]
    top = 0.0
    for i in range(m):
        v = a[i + 1] - a[i] if start == 1 else a[i]
        out[i] = v
        top = max(top, abs(v))
    if top > 0.0:
        e = -math.frexp(top)[1]
        if -1022 <= e <= 1023:
            f = math.ldexp(1.0, e)
            for i in range(m):
                out[i] = out[i] * f
        else:
            for i in range(m):
                out[i] = math.ldexp(out[i], e)


def _prep(x, b0):
    """g = diff(x) and b = b0, each scaled by ``_scale``
    (``_scan._to_unit``). Complexity: O(n)."""
    n = x.shape[0]
    g = np.empty(n - 1)
    _scale(x, g, 1)
    b = np.empty(n)
    _scale(b0, b, 0)
    return g, b


def _finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst):
    """``_scan._finish`` per knot, in one pass: rho, the reduction, their
    error bounds, the underflow floor and the zero mask. HV holds hᵀq for the
    k columns of Q that enter (column 0 is q₀ unless ``intercept``), then hᵀE;
    D is the intercept parent's D (else unused); e2 the squared norms of E;
    ``cst`` the constants (u, gamma_hv, dQ, the gamma of D, the factor of
    e_k, the floor). Complexity: O(k·c + n)."""
    u, g_hv, dQ, gD, ce, floor = cst[0], cst[1], cst[2], cst[3], cst[4], cst[5]
    m = HV.shape[0]
    nE = HV.shape[1] - k
    n = x.shape[0]
    ratio = np.zeros(m)
    gain = np.zeros(m)
    ratio_err = np.zeros(m)
    gain_err = np.zeros(m)
    normE = np.sqrt(e2)
    last = -1  # the last case with b0 ≠ 0
    for q in range(n - 1, -1, -1):
        if b0[q] != 0.0:
            last = q
            break
    first = 0 if intercept else 1
    for i in range(m):
        # h is 0 at every case (LA-3: constant) when no case q ≥ s has
        # b0_q ≠ 0 and x_q > x_{s-1}; x is sorted, so the last such case decides
        if last < 0 or x[last] <= x[split[i] - 1]:
            continue
        Fi = F[i]
        if intercept:
            c0 = 0.0
            Di = D[i]
        else:
            c0 = HV[i, 0]
            Di = Fi - c0 * c0
        s2 = 0.0
        sa = 0.0
        for j in range(first, k):
            v = HV[i, j]
            s2 += v * v
            sa += abs(v)
        perp = Di - s2
        rho = perp / Di if Di > 0.0 else 0.0
        num = 0.0
        for j in range(k, k + nE):
            num += HV[i, j] * HV[i, j]
        gi = num / perp if perp > 0.0 else 0.0
        sqF = math.sqrt(Fi)
        e_c = (g_hv + dQ) * sqF
        if intercept:
            errD = gD * Di + 2.0 * u * math.sqrt(Fi * abs(Di))
        else:
            errD = gD * Fi + 2.0 * abs(c0) * e_c + e_c * e_c
            errD += (dQ + 2.0 * u) * Fi
        errP = errD + 2.0 * sa * e_c + (r - 1) * e_c * e_c
        errP += dQ * s2 + 4.0 * u * (abs(Di) + s2)
        errN = 0.0
        for j in range(nE):
            e_k = sqF * normE[j] * ce
            errN += 2.0 * abs(HV[i, k + j]) * e_k + e_k * e_k
        errN += 2.0 * nE * u * num
        re = (errP + abs(rho) * errD) / (Di - errD) if errD < Di else math.inf
        ge = (errN + gi * errP) / (perp - errP) if errP < perp else math.inf
        if Fi < floor or abs(Di) < floor:  # underflow can reach
            re = math.inf
            ge = math.inf
        ratio[i] = rho
        gain[i] = gi
        ratio_err[i] = 2.0 * (re + 4.0 * u * abs(rho))
        gain_err[i] = 2.0 * (ge + 4.0 * u * gi)
    return ratio, gain, ratio_err, gain_err


def _first(Q, v, coef, known):
    """coef[j] = Σ_i Q[i, j]·v[i] for the columns j ≥ known. The rows go 4 at a
    time, and the loop over the columns vectorizes. Complexity: O(n·(r - known))."""
    n = v.shape[0]
    r = Q.shape[1]
    for j in range(known, r):
        coef[j] = 0.0
    i = 0
    while i + 4 <= n:
        v0, v1, v2, v3 = v[i], v[i + 1], v[i + 2], v[i + 3]
        for j in range(known, r):
            a = coef[j] + Q[i, j] * v0 + Q[i + 1, j] * v1
            coef[j] = a + Q[i + 2, j] * v2 + Q[i + 3, j] * v3
        i += 4
    while i < n:
        for j in range(known, r):
            coef[j] = coef[j] + Q[i, j] * v[i]
        i += 1


def _project(Q, c, v, out, acc, accumulate):
    """out = v - Q·c, and acc += Qᵀ·out when ``accumulate``; returns Σv² and
    Σout². One pass over the rows of Q, 4 rows at a time: each row's dot
    product runs in column order, and the 4 rows' chains overlap. out may be
    v. Complexity: O(n·r)."""
    n = v.shape[0]
    r = Q.shape[1]
    a0 = a1 = a2 = a3 = 0.0
    b0 = b1 = b2 = b3 = 0.0
    i = 0
    while i + 4 <= n:
        s0 = s1 = s2 = s3 = 0.0
        for j in range(r):
            cj = c[j]
            s0 += Q[i, j] * cj
            s1 += Q[i + 1, j] * cj
            s2 += Q[i + 2, j] * cj
            s3 += Q[i + 3, j] * cj
        v0, v1, v2, v3 = v[i], v[i + 1], v[i + 2], v[i + 3]
        p0, p1, p2, p3 = v0 - s0, v1 - s1, v2 - s2, v3 - s3
        out[i], out[i + 1], out[i + 2], out[i + 3] = p0, p1, p2, p3
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
        b0 += p0 * p0
        b1 += p1 * p1
        b2 += p2 * p2
        b3 += p3 * p3
        if accumulate:
            for j in range(r):
                t = acc[j] + Q[i, j] * p0 + Q[i + 1, j] * p1
                acc[j] = t + Q[i + 2, j] * p2 + Q[i + 3, j] * p3
        i += 4
    while i < n:
        s0 = 0.0
        for j in range(r):
            s0 += Q[i, j] * c[j]
        v0 = v[i]
        p0 = v0 - s0
        out[i] = p0
        a0 += v0 * v0
        b0 += p0 * p0
        if accumulate:
            for j in range(r):
                acc[j] = acc[j] + Q[i, j] * p0
        i += 1
    return (a0 + a1) + (a2 + a3), (b0 + b1) + (b2 + b3)


def _cgs2(Q, v, coef, known, perp, total, twice):
    """Classical Gram-Schmidt of v against the columns of Q (n, r), with a
    second pass when the first leaves less than half of ‖v‖² (DGKS).
    coef[:known] holds Qᵀv for the leading columns already; ``_first`` fills
    the others. perp gets v⊥ and total the coefficients of both passes.
    Returns ‖v‖ and ‖v⊥‖. 2 or 3 passes over Q, plus one over its new
    columns. Complexity: O(n·r)."""
    r = Q.shape[1]
    if known < r:
        _first(Q, v, coef, known)
    c2 = np.zeros(r)
    nv, n1 = _project(Q, coef, v, perp, c2, True)
    if twice or n1 + n1 < nv:
        _, n1 = _project(Q, c2, perp, perp, c2, False)
        for j in range(r):
            total[j] = coef[j] + c2[j]
    else:
        for j in range(r):
            total[j] = coef[j]
    return math.sqrt(nv), math.sqrt(n1)


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
    # error_model="numpy": a division by 0.0 gives inf or nan, as in numpy
    # (an intercept parent whose w = b·b underflows to 0), not an exception.
    _jit = numba.njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    _suffix1 = _jit(_suffix1)
    _rank = _jit(_rank)
    _hv = _jit(_hv)
    _scale = _jit(_scale)
    _prep = _jit(_prep)
    _finish = _jit(_finish)
    _norms = _jit(_norms)
    _centered = _jit(_centered)
    _first = _jit(_first)
    _project = _jit(_project)
    _cgs2 = _jit(_cgs2)


def hv(g, b, Q, q0, E, o, split):
    """hᵀV (k, c) for V = [Q[o, q0:], E] at every split, as ``_scan._hv``
    gives for the gathered V. Complexity: O(n·c + k·log k) time, O(c + k·c) memory."""
    n = b.shape[0]
    # A split outside 1..n - 1 would leave its row of the output unset; the
    # forward pass never passes one, and hinge_products checks it. The check
    # is O(k), next to the O(n·c) kernel, and holds under ``python -O`` too.
    if split.size and (split.min() < 1 or split.max() > n - 1):
        raise ValueError(f"every split must be in 1 to {n - 1}")
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


def unit(a, gaps):
    """``_scan._to_unit`` of diff(a) when ``gaps``, else of a, in one compiled
    pass. Complexity: O(n)."""
    out = np.empty(a.shape[0] - 1 if gaps else a.shape[0])
    _scale(a, out, 1 if gaps else 0)
    return out


def prep(x, b0):
    """``_scan._to_unit`` of diff(x) and of b0, in one compiled pass. The
    scaling is by a power of 2, so the values are those of the numpy code.
    Complexity: O(n)."""
    return _prep(x, b0)


def finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst):
    """``_scan._finish`` in one compiled pass (kernel B). Its sums over the
    columns run in order, where numpy may sum in pairs, so a value can differ
    from the numpy one by rounding inside its error bound.
    Complexity: O(k·c + n)."""
    if D is None:
        D = F  # unused without the intercept parent
    return _finish(HV, F, D, intercept, k, r, e2, x, b0, split, cst)


def cgs2(Q, v, coef, known, twice):
    """``_linalg.cgs2`` compiled: returns (v⊥, coef, total, ‖v‖, ‖v⊥‖), with
    coef[:known] given. Complexity: O(n·r) time, O(n + r) memory."""
    n, r = Q.shape
    perp = np.empty(n)
    total = np.empty(r)
    nv, n1 = _cgs2(Q, v, coef, known, perp, total, twice)
    return perp, coef, total, nv, n1
