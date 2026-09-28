"""The forward pass: the steps, the stopping rules and the forward record.

Spec: ``docs/algorithm.md``, "Forward pass" (FWD-1 to FWD-11), "Stopping
rules" (STOP-1 to STOP-7), LA-2 to LA-7, KNOT, SPAN, LIMIT-2, EDGE-1, EDGE-6,
CORE-3 (the forward record and the candidate log) and CORE-4 (termination).
The plan's sections "Fast path" and "Cost model" set the method; ``_scan``
scores the knots of one search.

Stage 1 of T11 (issue #13) supports ``max_degree=1`` (the intercept is the
only parent), one response, no weights and ``fast_k=0``; other values raise
NotImplementedError. Interactions come in stage 2, and Fast MARS, weights and
several responses in stage 3.

Method. Each column of X is sorted once, and the knots of the intercept are
found once per covariate (KNOT-7). The pass keeps an orthonormal basis Q of
the current columns (column 0 the intercept direction) and the residuals E.
A step searches each covariate: Gram-Schmidt of b·x gives A of LA-7, and so
the kind of the search; ``_scan.knot_scan`` scores every knot at once. The
best legal candidate (FWD-4, FWD-5) is built again explicitly
(``_scan.rebuild``), its hinge is tested again with the explicit ratio of
LA-3 (``_linalg.collinearity_ratio``), and its reduction is checked against
FWD-4 with the explicit RSS; a candidate that fails is left out and the step
is searched again, so the chosen candidate always passes FWD-4 and LA-3 with
its explicit values.

Numerics. Y is multiplied by a power of 2 first (EDGE-6) and centered (FWD-10,
LA-6); every RSS in the record is on the original scale. No absolute epsilon.
Ties follow FWD-5. No input is written to. Complexity: O(p·n·log n) for the
sorts, O(p·n·r) per step for rank r, O(n·M²) for FWD-11; memory
O(n·(p + M_max)).

Public names, for ``_core``:

- ``forward_pass``: the pass (FWD-1 to FWD-11, STOP-1 to STOP-5).
- ``ForwardPass``, ``CandidateLog``: the fields of ``ForwardRecord`` and
  ``CandidateLog`` (CORE-3).
- ``Termination``: the codes of CORE-4.
- ``KIND_NONE``, ``KIND_PAIR``, ``KIND_HINGE``, ``KIND_LINEAR``: the
  candidate kinds of the log (CORE-3).
"""

from __future__ import annotations

import enum
import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pymars import _gcv, _knots, _linalg, _scan, _terms

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: CORE-3: the kinds of a candidate in the candidate log.
KIND_NONE, KIND_PAIR, KIND_HINGE, KIND_LINEAR = 0, 1, 2, 3
#: FWD-4: a knot's reduction is at most min(1.01·RSS_s, 10·Δ_s).
MAX_LEGAL_RSS, MAX_LEGAL_DELTA = 1.01, 10.0
#: STOP-3: the pass stops when GRSq' is below this value.
GRSQ_FLOOR = -10.0
#: STOP-5: the pass stops when RSS_s < 1e-10·TSS/(N - 1).
RSS_FLOOR = 1e-10


class Termination(enum.IntEnum):
    """CORE-4: why the forward pass stopped (earth's codes where earth has one)."""

    DEGENERATE = 0
    NO_ROOM = 1
    GRSQ_NEG_INF = 2
    GRSQ_LOW = 3
    RSQ_CHANGE_SMALL = 4
    RSQ_HIGH = 5
    NO_GAIN = 6
    TERM_LIMIT = 7


class CandidateLog(NamedTuple):
    """CORE-3, one entry per step: ``best_rss`` (S,), equal to ``rss[1:]``;
    ``second_rss`` (S,), +∞ when the step had no other legal candidate;
    ``second_parent`` and ``second_variable`` (S,) int64, -1 for none;
    ``second_knot`` (S,), NaN for a linear term or none; ``second_kind``
    (S,) int8, a ``KIND_*`` code. Different candidates differ in the parent,
    the covariate, the kind or the knot value (FWD-8)."""

    best_rss: FloatArray
    second_rss: FloatArray
    second_parent: IntArray
    second_variable: IntArray
    second_knot: FloatArray
    second_kind: npt.NDArray[np.int8]


class ForwardPass(NamedTuple):
    """``forward_pass``: the fields of ``ForwardRecord`` (CORE-3).

    ``dirs`` (M_a, p) int8 and ``cuts`` (M_a, p): every term added, in order,
    row 0 the intercept. ``kept`` (M_f,) and ``dropped`` int64: the terms that
    FWD-11 keeps and drops. ``parent`` and ``step`` (M_a,) int64 (TERM-6).
    ``rss`` (S + 1,): TSS, then the RSS after each step. ``termination``
    (CORE-4). ``candidates``: a ``CandidateLog``, or None.
    """

    dirs: npt.NDArray[np.int8]
    cuts: FloatArray
    kept: IntArray
    dropped: IntArray
    parent: IntArray
    step: IntArray
    rss: FloatArray
    termination: Termination
    candidates: CandidateLog | None


class _Candidate(NamedTuple):
    """A candidate of a step: its reduction on the scaled Y, with a bound
    ``err`` on its error (0.0 for an explicit value), whether it is legal for
    sure (``sure``: no error within the bounds can change its LA-3 and FWD-4
    decisions), its place in the order of FWD-5, and what it adds."""

    reduction: float
    order: tuple[int, int, int]  # (parent rank, covariate, place in the search)
    parent: int
    variable: int
    kind: int
    knot: float  # NaN for a linear term
    err: float = 0.0
    sure: bool = True


class _Search(NamedTuple):
    """The search of one covariate x on the parent b: its kind (LA-7), the
    linear candidate's reduction (0.0 in a single-hinge search), and the
    orthonormal basis and residuals on G (B, and b·x in a pair search)."""

    x: FloatArray
    b: FloatArray
    pair: bool
    lin: float
    Q: FloatArray
    E: FloatArray


def _top_two(candidates: list[_Candidate]) -> list[_Candidate]:
    """Return the best two, by reduction and then by the order of FWD-5.
    Complexity: O(c·log c) for c candidates."""
    return _ranked(candidates)[:2]


def _ranked(candidates: list[_Candidate]) -> list[_Candidate]:
    """The candidates by reduction, then by the order of FWD-5."""
    return sorted(candidates, key=lambda c: (-c.reduction, c.order))


def _second_largest(values: list[float]) -> float:
    """The second largest value, or -inf when there are fewer than two."""
    return sorted(values, reverse=True)[1] if len(values) > 1 else -math.inf


class _Pass:
    """The state of one forward pass on the scaled, centered responses."""

    def __init__(self, X: FloatArray, Yc: FloatArray, tss: float, params: dict):
        self.X, self.Yc, self.tss, self.params = X, Yc, tss, params
        n, p = X.shape
        self.N, self.tau = _gcv.total_weight(n)
        self.variances = _linalg.weighted_variances(X)
        self.order = np.argsort(X, axis=0, kind="stable")
        self.ones = np.ones(n)  # the intercept, the only parent in stage 1
        L, E = _knots.search_spans(
            p,
            0,
            self.N,
            self.N,
            minspan=params["minspan"],
            endspan=params["endspan"],
            adjust_endspan=params["adjust_endspan"],
            tau=self.tau,
        )
        active = np.ones(n, dtype=bool)
        self.knots = [
            _knots.candidate_knots(
                X[self.order[:, j], j], active, L, E, total=self.N, tau=self.tau
            )
            for j in range(p)
        ]
        self.Q = np.full((n, 1), 1.0 / math.sqrt(n))
        self.E, _ = _linalg.orthogonalize(self.Q, Yc)
        self.rss = [tss]
        self.dirs, self.cuts = _terms.intercept_terms(p)
        self.parent, self.step_of = [-1], [0]

    def max_legal(self) -> float:
        """FWD-4: MaxLegal_s = min(1.01·RSS_s, 10·Δ_s), and 1.01·RSS_0 first.
        Complexity: O(1)."""
        rss = self.rss
        if len(rss) == 1:
            return MAX_LEGAL_RSS * rss[0]
        return min(MAX_LEGAL_RSS * rss[-1], MAX_LEGAL_DELTA * (rss[-2] - rss[-1]))

    def setup(self, j: int) -> _Search:
        """The search of covariate j on the intercept: Gram-Schmidt of b·x gives
        A of LA-7, and so the kind; a pair search adds b·x to G.
        Complexity: O(n·r)."""
        x, b = self.X[:, j], self.ones
        gs = _linalg.gram_schmidt(self.Q, b * x)
        pair = gs.q is not None and _linalg.pair_search(gs.norm**2, self.variances[[j]])
        if not pair:
            return _Search(x, b, False, 0.0, self.Q, self.E)
        proj = gs.q @ self.E
        E = self.E - np.outer(gs.q, proj)
        return _Search(
            x, b, True, float(np.sum(proj**2)), np.column_stack((self.Q, gs.q)), E
        )

    def search(self, j: int, excluded: set) -> list[_Candidate]:
        """Pass 1 for covariate j: every candidate that can be one of the step's
        best two by the scan's values and error bounds (FWD-3, FWD-4, LA-3),
        ranked, leaving out the places in ``excluded``. A knot is dropped when
        its bounds make it illegal for sure, and kept when its upper bound
        reaches the second largest lower bound of the sure candidates.
        Complexity: O(n·r)."""
        sr = self.setup(j)
        o, kc = self.order[:, j], self.knots[j]
        scan = _scan.knot_scan(
            sr.x[o], sr.b[o], sr.Q[o], sr.E[o], kc.split, intercept=True
        )
        red, err = scan.gain + sr.lin, scan.gain_err
        tau = _linalg.collinearity_tolerance(len(self.rss))  # this is step s
        top = self.max_legal()
        possible = (scan.ratio + scan.ratio_err >= tau) & (red + err > 0.0)
        possible &= red - err <= top
        sure = (scan.ratio - scan.ratio_err >= tau) & (red - err > 0.0)
        sure &= red + err <= top
        for parent, var, place in excluded:
            if (parent, var) == (0, j) and place > 0:
                possible[place - 1] = False
        found = []
        if sr.pair and sr.lin > 0.0 and (0, j, 0) not in excluded:
            found.append(_Candidate(sr.lin, (0, j, 0), 0, j, KIND_LINEAR, math.nan))
        floor = _second_largest(
            [*(red - err)[possible & sure], *(c.reduction for c in found)]
        )
        kind = KIND_PAIR if sr.pair else KIND_HINGE
        for i in np.flatnonzero(possible & (red + err >= floor)):
            t, e = float(kc.knots[i]), float(err[i])
            c = _Candidate(
                float(red[i]), (0, j, i + 1), 0, j, kind, t, e, bool(sure[i])
            )
            found.append(c)
        return _ranked(found)

    def refine(self, cands: list[_Candidate]) -> list[_Candidate]:
        """Pass 2: the explicit values of the knots (``_scan.exact_knot``), with
        LA-3 and FWD-4 decided on them; the illegal ones are left out, and the
        linear candidates, whose values are explicit already, stay.
        Complexity: O(n·r) per covariate and O(n·(r + K)) per knot."""
        out = [c for c in cands if c.kind == KIND_LINEAR]
        s, top = len(self.rss), self.max_legal()
        for j in sorted({c.variable for c in cands if c.kind != KIND_LINEAR}):
            sr = self.setup(j)
            for c in cands:
                if c.variable != j or c.kind == KIND_LINEAR:
                    continue
                h = sr.b * _terms.factor(_terms.PLUS, sr.x, c.knot)
                rho, gain = _scan.exact_knot(sr.Q, sr.E, h)
                red = gain + sr.lin
                if not _linalg.knot_rejected(rho, s) and 0.0 < red <= top:
                    out.append(c._replace(reduction=red, err=0.0, sure=True))
        return out

    def columns(self, c: _Candidate) -> tuple[FloatArray, FloatArray | None]:
        """Return the columns that span the candidate's terms (b·x before the
        hinge of a pair, FWD-6) and its hinge column, or None. Complexity:
        O(n)."""
        x = self.X[:, c.variable]
        h = None if c.kind == KIND_LINEAR else _terms.factor(_terms.PLUS, x, c.knot)
        cols = {KIND_PAIR: (x, h), KIND_HINGE: (h,), KIND_LINEAR: (x,)}[c.kind]
        return np.column_stack(cols), h

    def check(self, c: _Candidate) -> _scan.Rebuild | None:
        """Build the candidate again explicitly and apply LA-3 and FWD-4 to it
        with the explicit values; return the rebuild, or None when it fails.
        Complexity: O(n·r·K)."""
        cols, h = self.columns(c)
        rb = _scan.rebuild(self.Q, self.Yc, cols)
        if rb is None:
            return None
        if h is not None:
            r = self.Q.shape[1]
            G = rb.Q[:, : r + 1 if c.kind == KIND_PAIR else r]
            if _linalg.knot_rejected(_linalg.collinearity_ratio(G, h), len(self.rss)):
                return None
        reduction = self.rss[-1] - rb.rss
        if reduction <= 0.0:
            return None
        if c.kind != KIND_LINEAR and reduction > self.max_legal():
            return None
        return rb

    def best(self) -> tuple[_Candidate | None, _scan.Rebuild | None, _Candidate | None]:
        """Search the step and return the chosen candidate, its rebuild and the
        second-best candidate (FWD-2 to FWD-5, FWD-8). Pass 1 scans every
        covariate; every candidate whose upper bound reaches the second largest
        lower bound of the sure ones is then valued explicitly (pass 2), so the
        best two and every LA-3 and FWD-4 decision about them rest on explicit
        values. Complexity: O(p·n·r) for pass 1, repeated only when a check
        fails, and O(n·r) for each covariate and knot of pass 2."""
        excluded: set = set()
        while True:
            kept = [c for j in range(self.X.shape[1]) for c in self.search(j, excluded)]
            floor = _second_largest([c.reduction - c.err for c in kept if c.sure])
            top = _top_two(
                self.refine([c for c in kept if c.reduction + c.err >= floor])
            )
            if not top:
                return None, None, None
            rb = self.check(top[0])
            if rb is not None:
                return top[0], rb, top[1] if len(top) > 1 else None
            excluded.add(top[0].order)

    def add(self, c: _Candidate, rb: _scan.Rebuild) -> None:
        """Append the candidate's terms (FWD-6) and take its basis and RSS.
        Complexity: O(M·p) for the copies of the term table."""
        auto = self.params["auto_linpreds"]
        if c.kind == KIND_PAIR:
            new = [(_terms.PLUS, c.knot), (_terms.MINUS, c.knot)]
        elif c.kind == KIND_HINGE:
            new = [(_terms.PLUS, c.knot)]
        elif auto:
            new = [(_terms.LINEAR, 0.0)]
        else:
            new = [(_terms.PLUS, _knots.linear_option_knot(self.X[:, c.variable]))]
        s = len(self.rss)
        for code, cut in new:
            d, k = _terms.child_term(self.dirs[0], self.cuts[0], c.variable, code, cut)
            self.dirs = np.vstack((self.dirs, d))
            self.cuts = np.vstack((self.cuts, k))
            self.parent.append(c.parent)
            self.step_of.append(s)
        self.Q, self.E = rb.Q, rb.resid
        self.rss.append(rb.rss)


def _stop(
    st: _Pass, chosen: _Candidate | None, rss_new: float, m_new: int
) -> Termination | None:
    """STOP-3, STOP-4 and STOP-2, in this order, before the candidate is added.

    RSq' - RSq_s is computed as (RSS_s - RSS')/TSS, equal to it in exact
    arithmetic and free of the rounding of 1 - RSS/TSS near RSq = 1.
    Complexity: O(1).
    """
    prm, tss = st.params, st.tss
    grsq = _gcv.grsq(rss_new, tss, m_new, prm["penalty"], st.N, st.tau)
    if prm["thresh"] > 0.0 and grsq < GRSQ_FLOOR:
        return Termination.GRSQ_NEG_INF if grsq == -math.inf else Termination.GRSQ_LOW
    if (st.rss[-1] - rss_new) / tss < prm["thresh"]:
        return Termination.RSQ_CHANGE_SMALL
    if chosen is None:
        return Termination.NO_GAIN
    return None


def _rsq_high(st: _Pass, rss: float) -> bool:
    """STOP-5 after a step: RSq_s ≥ 1 - thresh, computed as RSS_s/TSS ≤ thresh
    (the same in exact arithmetic), or RSS_s < 1e-10·TSS/(N - 1).
    Complexity: O(1)."""
    floor = RSS_FLOOR * st.tss / (st.N - 1.0)
    return rss / st.tss <= st.params["thresh"] or rss < floor


def _run(st: _Pass, max_terms: int, log: list) -> Termination:
    """The steps of FWD-1 to FWD-9 with the stopping rules in the spec's order.
    Complexity: O(S·p·n·r) for S steps."""
    while True:
        s = len(st.rss) - 1  # completed steps
        if 1 + 2 * (s + 1) > max_terms:  # STOP-1
            return Termination.NO_ROOM if s == 0 else Termination.TERM_LIMIT
        chosen, rb, second = st.best()
        m = st.dirs.shape[0]
        if chosen is None:
            rss_new, m_new = st.rss[-1], m + 1
        else:
            rss_new, m_new = rb.rss, m + (2 if chosen.kind == KIND_PAIR else 1)
        code = _stop(st, chosen, rss_new, m_new)
        if code is not None:
            return code
        rss_before = st.rss[-1]
        st.add(chosen, rb)
        log.append((rss_before, second))
        if _rsq_high(st, rss_new):  # STOP-5
            return Termination.RSQ_HIGH


def _log(rss: FloatArray, log: list, shift: int) -> CandidateLog:
    """The candidate log on the original scale (CORE-3, EDGE-6).
    Complexity: O(S)."""
    S = len(log)
    second_rss = np.full(S, math.inf)
    second_parent = np.full(S, -1, dtype=np.int64)
    second_variable = np.full(S, -1, dtype=np.int64)
    second_knot = np.full(S, math.nan)
    second_kind = np.zeros(S, dtype=np.int8)
    for i, (rss_before, c) in enumerate(log):
        if c is not None:
            second_rss[i] = np.ldexp(rss_before - c.reduction, shift)
            second_parent[i], second_variable[i] = c.parent, c.variable
            second_knot[i], second_kind[i] = c.knot, c.kind
    return CandidateLog(
        rss[1:].copy(),
        second_rss,
        second_parent,
        second_variable,
        second_knot,
        second_kind,
    )


def forward_pass(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    max_degree: int = 1,
    max_terms: int | None = None,
    penalty: float | None = None,
    thresh: float = 0.001,
    minspan: int | None = None,
    endspan: int | None = None,
    adjust_endspan: float = 2.0,
    auto_linpreds: bool = True,
    fast_k: int = 20,
    fast_beta: float = 1.0,
    record_candidates: bool = False,
) -> ForwardPass:
    """Run the forward pass on X (n, p) and Y (n,) or (n, 1); return its record.

    The keyword arguments are the fields of ``MarsParams`` (CORE-2), which
    the core checks; None resolves ``max_terms`` by LIMIT-1 and ``penalty``
    by GCV-4 (the penalty enters only GRSq' of STOP-3). A degenerate fit
    (EDGE-1) returns the intercept alone with code ``DEGENERATE``, and
    ``max_terms`` ≤ 2 gives ``NO_ROOM`` (STOP-1). Raises NotImplementedError
    outside stage 1 (``max_degree`` > 1, weights, several responses,
    ``fast_k`` ≠ 0; ``fast_beta`` acts from stage 3), and ValueError for
    nonfinite or mismatched input. EDGE-6's error for a scaled TSS that is
    not a positive normal number needs weights, and comes with them: with
    unit weights the scaled Y has a value of size at least 1.
    Complexity: O(p·n·log n + S·p·n·r + n·M²) time, O(n·(p + M_max)) memory.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if X.ndim != 2 or Y.ndim != 2 or Y.shape[0] != X.shape[0] or X.shape[0] < 1:
        raise ValueError(f"X (n, p) and Y (n, K) do not match: {X.shape}, {Y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X and Y must be finite")
    if max_degree != 1 or w is not None or Y.shape[1] != 1 or fast_k != 0:
        raise NotImplementedError(
            "T11 stage 1: max_degree=1, one response, no weights and fast_k=0 only"
        )
    n, p = X.shape
    max_terms = _gcv.default_max_terms(p) if max_terms is None else max_terms
    penalty = _gcv.default_penalty(max_degree) if penalty is None else penalty
    params = {
        "minspan": minspan,
        "endspan": endspan,
        "adjust_endspan": adjust_endspan,
        "auto_linpreds": auto_linpreds,
        "thresh": float(thresh),
        "penalty": penalty,
    }
    if _gcv.is_degenerate(Y, float(n)):  # EDGE-1, GCV-7
        return _intercept_only(
            p, _gcv.tss(Y), Termination.DEGENERATE, record_candidates
        )
    shift = 1 - int(np.frexp(np.max(np.abs(Y)))[1])  # EDGE-6: D·2^shift in [1, 2)
    Ys = np.ldexp(Y, shift)
    st = _Pass(X, Ys - Ys.mean(axis=0), _gcv.tss(Ys), params)
    log: list = []
    termination = _run(st, max_terms, log)
    rss = np.ldexp(np.array(st.rss), -2 * shift)
    B = _terms.basis_matrix(X, st.dirs, st.cuts)
    kept = _linalg.independent_columns(B)  # FWD-11
    return ForwardPass(
        st.dirs,
        st.cuts,
        np.flatnonzero(kept).astype(np.int64),
        np.flatnonzero(~kept).astype(np.int64),
        np.array(st.parent, dtype=np.int64),
        np.array(st.step_of, dtype=np.int64),
        rss,
        termination,
        _log(rss, log, -2 * shift) if record_candidates else None,
    )


def _intercept_only(
    p: int, tss: float, termination: Termination, record: bool
) -> ForwardPass:
    """The record of a pass that took no step (EDGE-1). Complexity: O(p)."""
    dirs, cuts = _terms.intercept_terms(p)
    empty = np.empty(0, dtype=np.int64)
    log = CandidateLog(
        np.empty(0), np.empty(0), empty, empty, np.empty(0), np.empty(0, dtype=np.int8)
    )
    return ForwardPass(
        dirs,
        cuts,
        np.zeros(1, dtype=np.int64),
        empty,
        np.array([-1], dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.array([tss]),
        termination,
        log if record else None,
    )
