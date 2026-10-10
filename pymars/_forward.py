"""The forward pass: the steps, the stopping rules and the forward record.

Spec: ``docs/algorithm.md``, "Forward pass" (FWD-1 to FWD-11), "Stopping
rules" (STOP-1 to STOP-7), LA-2 to LA-7, KNOT, SPAN, LIMIT-2, "Fast MARS"
(FAST-1 to FAST-6), "Weights" (W-1 to W-5, W-8), "Several responses"
(RESP-1 to RESP-3), EDGE-1, EDGE-6, CORE-3 (the forward record and the
candidate log) and CORE-4 (termination). The plan's
sections "Fast path" and "Cost model" set the method; ``_scan`` scores the
knots of one search.

Weights and several responses (W, RESP). Rows with zero weight are dropped
first (W-3). Least squares works on rows scaled by √w (the plan's "Fast
path"): Q is an orthonormal basis of the √w-scaled columns, E holds the
√w-scaled residuals of the K responses, and the scan gets √w·b for a parent
b. N = Σw replaces n in the spans, the knot scan, the GCV and the stopping
rules (W-2, W-4), and every reduction is summed over the responses (RESP-1).
Without weights w_i = 1 exactly, and every product with √w is exact (W-5).

Method. Each column of X is sorted once (KNOT-7). The pass keeps the column
of every term (TERM-3), an orthonormal basis Q of the current columns (column
0 the intercept direction) and the residuals E. A step searches the parents
that the queue gives (FWD-2): the first nu rows of the table (FAST-2, FAST-3;
nu = max(3, ``fast_k``), and every row with ``fast_k=0``), where entry e
stands for slot e, so a term in a slot above M, the number of terms, waits
(FAST-4, FWD-9, OQ-4), and a term at ``max_degree`` is skipped but keeps its
row. For a parent b and each covariate x that b does not hold, the knots
come from the active cases of b (KNOT-1 to KNOT-3) with the spans of b's
degree (SPAN-1 to SPAN-5, the adjusted endspan of SPAN-4 included).
Gram-Schmidt of b·(x - c) gives A of LA-7, and so the kind of the search,
where c is the covariate's middle value (the plan's "Fast path": center x
before the sums; the span does not change, since b is in B, and the term
added stays b·x). ``_scan.knot_scan`` scores every knot at once, with error
bounds (its intercept path for the intercept, its general path for other
parents), and pass 2 values explicitly every knot whose bounds could change
the step's best two or its parent's best (``best``), so that every choice,
the candidate log and the queue values of FAST-5 rest on explicit values. The
chosen candidate is built again (``_scan.rebuild``), and its reduction is
checked against FWD-4 with the rebuilt RSS; a candidate that fails is left
out, its search is done again without it, and the step chooses again.

Numerics. Y is multiplied by a power of 2 first (EDGE-6) and centered in two
steps, each response by its data value nearest its weighted mean and then by
the weighted mean (FWD-10, LA-6; ``_pruning`` does the same), so that the
rounding is of the size of the spread of Y and not of its mean, whatever the
weights; the TSS comes from the centered Y. Every RSS in the record is on the
original scale, where one can underflow to 0 or, for |y|·√n above about
1e154, overflow to +inf (EDGE-6 names the underflow). No absolute epsilon.
Ties follow FWD-5. No input is written to. Complexity: O(p·n·log n) for the
sorts, O(P·p·n·r) per step for P parents and rank r, O(n·M²) for FWD-11;
memory O(n·(p + M_max)).

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

from pymars import _gcv, _knots, _linalg, _pruning, _scan, _terms

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
#: A candidate RSS at most EXACT_FIT·RSS_s is an exact fit, 0 (issue #81).
#: The value is set by the rounding of a reduction Σ_k (qᵀE_k)² ≤ RSS_s: with
#: E orthogonal to Q and q a unit vector, both to about √r·u (Gram-Schmidt
#: twice), and blocked sums, it is a small multiple of u·RSS_s (u = 2^-53).
#: Measured, a candidate that fits exactly in exact arithmetic rounds to at
#: most 6.7e-16·RSS_s on #81's data, and to 2.4e-15·RSS_s for n up to 30000
#: with weights exp(U(-9, 9)). 1e-14 (about 45·u) covers that four times;
#: a wider band sets real candidates equal (a kink of size 1e-5 in 50 cases
#: lost its knot at 1e-12, #83).
EXACT_FIT = 1e-14
#: FAST-3: a positive fast_k below this value acts as this value.
FAST_K_MIN = 3


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
    ``second_parent`` and ``second_variable`` (S,) int64, -1 for none, the
    parent as a forward index (TERM-6); ``second_knot`` (S,), NaN for a
    linear term or none; ``second_kind`` (S,) int8, a ``KIND_*`` code.
    Different candidates differ in the parent, the covariate, the kind or
    the knot value (FWD-8). A second-best whose RSS lies within EXACT_FIT of
    0 is logged with ``second_rss`` = 0, so at an exact fit ``second_rss`` can
    be below ``best_rss``, which is the RSS of the rebuilt winner."""

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
    decisions), its place in the order of FWD-5, and what it adds. ``parent``
    is the parent's forward index (TERM-6)."""

    reduction: float
    order: tuple[int, int, int]  # (table row of the parent, covariate, place)
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
    """The state of one forward pass on the scaled, centered responses Yc
    (n, K) with the weights w (n,), all positive, or None for w_i = 1.

    The terms are numbered by their forward index k (TERM-6); ``cols[k]`` is
    the column of term k (TERM-3), ``degree[k]`` its degree (TERM-4) and
    ``spans[k]`` the minspan and endspan of a search on it as a parent
    (SPAN-1 to SPAN-5). ``slot[k]`` is the slot of term k and ``term_at``
    the term in each slot that holds one (FWD-9). The queue has one entry
    per term, 1-based: ``lam[e - 1]`` and ``kap[e - 1]`` are λ_e and κ_e of
    entry e (FAST-1, FAST-5). ``rows`` maps each parent of the current step
    to its row in the queue table, the first key of FWD-5. ``sw`` is √w, and
    ``Yw`` = √w·Yc the responses of the scaled rows.
    """

    def __init__(
        self,
        X: FloatArray,
        Yc: FloatArray,
        tss: float,
        params: dict,
        w: FloatArray | None = None,
    ):
        self.X, self.tss, self.params, self.w = X, tss, params, w
        n, p = X.shape
        self.N, self.tau = _gcv.total_weight(n, w)
        self.sw = np.ones(n) if w is None else np.sqrt(w)
        self.Yw = self.sw[:, None] * np.reshape(Yc, (n, -1))
        self.variances = _linalg.weighted_variances(X, w)
        self.order = np.argsort(X, axis=0, kind="stable")
        self.sorted_x = np.take_along_axis(X, self.order, axis=0)
        self.center = self.sorted_x[n // 2]  # a data value (FWD-10)
        self.cols = [np.ones(n)]
        self.degree = [0]
        self.spans = [self._spans(0, self.N)]
        L, E = self.spans[0]
        active = np.ones(n, dtype=bool)
        self.knots = [  # the intercept's knots, found once per covariate (KNOT-7)
            self._knots(j, active, L, E) for j in range(p)
        ]
        raw = float(n) if w is None else math.fsum(w)  # ‖√w‖², not snapped
        self.Q = (self.sw / math.sqrt(raw))[:, None]
        self.E, _ = _linalg.orthogonalize(self.Q, self.Yw)
        self.rss = [tss]
        self.dirs, self.cuts = _terms.intercept_terms(p)
        self.cond = _linalg.Conditioner(X, self.center)  # LA-5
        self.cond.append(self.dirs[0], self.cuts[0])
        self.parent, self.step_of = [-1], [0]
        self.slot, self.term_at = [1], {1: 0}  # slot 1 holds the intercept
        self.lam, self.kap = [math.inf], [0]
        self.rows = {0: 0}  # before the first step the table is the intercept

    def _knots(self, j: int, active: npt.NDArray[np.bool_], L: int, E: int):
        """The knots of covariate j for the active cases, in the order of x
        (KNOT-3, KNOT-6 with weights). Complexity: O(n)."""
        w = None if self.w is None else self.w[self.order[:, j]]
        return _knots.candidate_knots(
            self.sorted_x[:, j], active, L, E, w, total=self.N, tau=self.tau
        )

    def _spans(self, degree: int, active_weight: float) -> tuple[int, int]:
        """(L, E*) of a search on a parent of this degree and active weight
        (SPAN-1 to SPAN-5). Complexity: O(1)."""
        prm = self.params
        return _knots.search_spans(
            self.X.shape[1],
            degree,
            self.N,
            active_weight,
            minspan=prm["minspan"],
            endspan=prm["endspan"],
            adjust_endspan=prm["adjust_endspan"],
            tau=self.tau,
        )

    def max_legal(self) -> float:
        """FWD-4: MaxLegal_s = min(1.01·RSS_s, 10·Δ_s), and 1.01·RSS_0 first.
        Complexity: O(1)."""
        rss = self.rss
        if len(rss) == 1:
            return MAX_LEGAL_RSS * rss[0]
        return min(MAX_LEGAL_RSS * rss[-1], MAX_LEGAL_DELTA * (rss[-2] - rss[-1]))

    def table(self) -> list[int]:
        """FAST-2: the queue entries (1-based) in the order of the table.

        rank_e is the place of entry e by λ, largest first, ties by entry
        number; AgedRank_e = rank_e + fast_beta·(κ_prev - κ_e), with κ_prev
        the term number of the step just done; the table sorts the entries by
        AgedRank and then by rank. Before the first step the intercept's entry
        is the only one. Complexity: O(M·log M).
        """
        s = len(self.rss) - 1
        M = len(self.lam)
        rank = [0] * M
        for r, e in enumerate(sorted(range(M), key=lambda e: (-self.lam[e], e))):
            rank[e] = r
        beta, kappa_prev = self.params["fast_beta"], 2 * s
        aged = [rank[e] + beta * (kappa_prev - self.kap[e]) for e in range(M)]
        return [e + 1 for e in sorted(range(M), key=lambda e: (aged[e], rank[e]))]

    def parents(self) -> list[int]:
        """FAST-3 and FAST-4: the step visits the first nu rows of the table,
        nu = max(3, ``fast_k``) for ``fast_k`` ≥ 1 (so 1 and 2 act as 3), and
        every row for ``fast_k=0`` or nu ≥ M. Entry e stands for slot e. It is
        skipped when the slot is empty or its term's degree is ``max_degree``,
        and it keeps its row among the nu; otherwise that term is a parent of
        the step. Sets ``rows`` and returns the parents in table order.
        Complexity: O(M·log M)."""
        table, fast_k = self.table(), self.params.get("fast_k", 0)
        if fast_k >= 1:
            table = table[: max(FAST_K_MIN, fast_k)]
        rows = {}
        for row, e in enumerate(table):
            k = self.term_at.get(e)
            if k is not None and self.degree[k] < self.params["max_degree"]:
                rows[k] = row
        self.rows = rows
        return list(rows)

    def covariates(self, k: int) -> list[int]:
        """FWD-2: the covariates that a search on parent k takes, in
        increasing order: those that term k does not hold (TERM-4).
        Complexity: O(p)."""
        return np.flatnonzero(self.dirs[k] == _terms.ABSENT).tolist()

    def knots_of(self, k: int, j: int) -> _knots.KnotCandidates:
        """The candidate knots of parent k and covariate j (KNOT-1 to KNOT-3):
        the active cases of k are those where its column is positive.
        Complexity: O(n), and O(1) for the intercept, whose knots are kept."""
        if k == 0:
            return self.knots[j]
        L, E = self.spans[k]
        return self._knots(j, self.cols[k][self.order[:, j]] > 0.0, L, E)

    def _capped(self, red):
        """A reduction, or RSS_s where the candidate's RSS, RSS_s - red, is at
        most EXACT_FIT·RSS_s: an exact fit up to rounding, since an RSS is at
        least 0 (LA-1, LA-2). Such candidates then tie exactly, and FWD-5
        decides between them, not the rounding (issue #81). The change is at
        most EXACT_FIT·RSS_s, inside LA-5. Complexity: O(len(red))."""
        rss = self.rss[-1]
        return np.where(rss - red <= EXACT_FIT * rss, rss, red)

    def linear_column(self, k: int, j: int) -> FloatArray:
        """The unscaled column that stands for b·x_j in the projections, b the
        column of term k: it spans with B what b·x_j spans, since B holds b.
        That is b·(x_j - c_j), c_j the covariate's middle value (the plan's
        "Fast path"), unless b has a linear factor, or a hinge (t - x)₊, of a
        covariate with a large mean: then the reduced expansion of b·x_j
        (``_linalg.Conditioner``), so that no digits go to the mean (LA-5).
        Complexity: O(n) and, with a large mean, O(M·3^d) rational operations
        and O(n·d·3^d) for the column of a term of degree d."""
        if not self.cond.expands(self.dirs[k]):
            return self.cols[k] * (self.X[:, j] - self.center[j])
        return self.cond.linear(k, self.dirs[k], self.cuts[k], j)

    def setup(self, k: int, j: int) -> _Search:
        """The search of covariate j on parent k: Gram-Schmidt of b·(x - c)
        gives A of LA-7, and so the kind, where V holds the covariates of b
        and x; a pair search adds b·x to G, and its residuals are
        orthogonalized twice against G, as ``_scan.knot_scan`` requires.
        Every reduction goes through ``_capped``, here, in ``search`` and in
        ``refine``.
        Complexity: O(n·r·K)."""
        x, b = self.X[:, j], self.cols[k]
        gs = _linalg.gram_schmidt(self.Q, self.sw * self.linear_column(k, j))
        V = [*np.flatnonzero(self.dirs[k]).tolist(), j]
        pair = gs.q is not None and _linalg.pair_search(gs.norm**2, self.variances[V])
        if not pair:
            return _Search(x, b, False, 0.0, self.Q, self.E)
        Q = np.column_stack((self.Q, gs.q))
        lin = float(self._capped(float(np.sum((gs.q @ self.E) ** 2))))
        return _Search(x, b, True, lin, Q, _linalg.orthogonalize(Q, self.E)[0])

    def search(self, k: int, j: int, excluded: set) -> list[_Candidate]:
        """Pass 1 for parent k and covariate j: every candidate that can be one
        of the best two of its search by the scan's values and error bounds
        (FWD-3, FWD-4, LA-3), ranked, leaving out the places in ``excluded``.
        A knot is dropped when its bounds make it illegal for sure, and kept
        when its upper bound reaches the second largest lower bound of the
        sure candidates. Complexity: O(n·r)."""
        sr = self.setup(k, j)
        o, kc, row = self.order[:, j], self.knots_of(k, j), self.rows[k]
        bw = self.sw[o] * sr.b[o]  # √w·b in the order of x
        scan = _scan.knot_scan(
            self.sorted_x[:, j], bw, sr.Q[o], sr.E[o], kc.split, intercept=k == 0
        )
        red, err = self._capped(scan.gain + sr.lin), scan.gain_err
        tau = _linalg.collinearity_tolerance(len(self.rss))  # this is step s
        top = self.max_legal()
        possible = (scan.ratio + scan.ratio_err >= tau) & (red + err > 0.0)
        possible &= red - err <= top
        sure = (scan.ratio - scan.ratio_err >= tau) & (red - err > 0.0)
        sure &= red + err <= top
        for r, var, place in excluded:
            if (r, var) == (row, j) and place > 0:
                possible[place - 1] = False
        found = []
        if sr.pair and sr.lin > 0.0 and (row, j, 0) not in excluded:
            found.append(_Candidate(sr.lin, (row, j, 0), k, j, KIND_LINEAR, math.nan))
        floor = _second_largest(
            [*(red - err)[possible & sure], *(c.reduction for c in found)]
        )
        kind = KIND_PAIR if sr.pair else KIND_HINGE
        for i in np.flatnonzero(possible & (red + err >= floor)):
            t, e = float(kc.knots[i]), float(err[i])
            c = _Candidate(
                float(red[i]), (row, j, i + 1), k, j, kind, t, e, bool(sure[i])
            )
            found.append(c)
        return _ranked(found)

    def refine(self, cands: list[_Candidate]) -> list[_Candidate]:
        """Pass 2: the explicit values of the knots (``_scan.exact_knot``), with
        LA-3 and FWD-4 decided on them; the illegal ones are left out, and the
        linear candidates, whose values are explicit already, stay.
        Complexity: O(n·r) per search and O(n·(r + K)) per knot."""
        out = [c for c in cands if c.kind == KIND_LINEAR]
        s, top = len(self.rss), self.max_legal()
        for k, j in sorted(
            {(c.parent, c.variable) for c in cands if c.kind != KIND_LINEAR}
        ):
            sr = self.setup(k, j)
            for c in cands:
                if (c.parent, c.variable) != (k, j) or c.kind == KIND_LINEAR:
                    continue
                h = sr.b * _terms.factor(_terms.PLUS, sr.x, c.knot)
                rho, gain = _scan.exact_knot(sr.Q, sr.E, h, self.w)
                red = float(self._capped(gain + sr.lin))
                if not _linalg.knot_rejected(rho, s) and 0.0 < red <= top:
                    out.append(c._replace(reduction=red, err=0.0, sure=True))
        return out

    def columns(self, c: _Candidate) -> tuple[FloatArray, FloatArray | None]:
        """Return the √w-scaled columns that span the candidate's terms (b·x
        before the hinge of a pair, FWD-6) and its unscaled hinge column, or
        None. Complexity: O(n)."""
        x, b = self.X[:, c.variable], self.cols[c.parent]
        xc = self.linear_column(c.parent, c.variable)  # as in setup
        h = None
        if c.kind != KIND_LINEAR:
            h = b * _terms.factor(_terms.PLUS, x, c.knot)
            if self.cond.expands(self.dirs[c.parent]):  # LA-5
                d, k = _terms.child_term(
                    self.dirs[c.parent],
                    self.cuts[c.parent],
                    c.variable,
                    _terms.PLUS,
                    c.knot,
                )
                h = self.cond.hinge(d, k)
        cols = {KIND_PAIR: (xc, h), KIND_HINGE: (h,), KIND_LINEAR: (xc,)}[c.kind]
        return self.sw[:, None] * np.column_stack(cols), h

    def check(self, c: _Candidate) -> _scan.Rebuild | None:
        """Build the candidate again (``_scan.rebuild``) and apply FWD-4 to its
        rebuilt RSS; return the rebuild, or None when it fails. LA-3 is not
        tested again: pass 2 decided it on the same Gram-Schmidt of the same
        columns, so the ratio would be the same up to rounding. The rebuilt RSS and
        pass 2's reduction can differ in the last bits, so a knot at MaxLegal
        can fail here. Complexity: O(n·r·K)."""
        cols, _ = self.columns(c)
        rb = _scan.rebuild(self.Q, self.Yw, cols)
        if rb is None:
            return None
        reduction = self.rss[-1] - rb.rss
        if reduction <= 0.0:
            return None
        if c.kind != KIND_LINEAR and reduction > self.max_legal():
            return None
        return rb

    def best(self) -> tuple[_Candidate | None, _scan.Rebuild | None, _Candidate | None]:
        """Search the step and return the chosen candidate, its rebuild and the
        second-best candidate (FWD-2 to FWD-5, FWD-8); set the queue values of
        the parents searched (FAST-5).

        Pass 1 scans every parent and covariate. Pass 2 values explicitly
        every candidate whose upper bound reaches the second largest lower
        bound of the sure candidates of the step, or the largest one of its
        own parent, so the best two, each parent's best and every LA-3 and
        FWD-4 decision about them rest on explicit values. When the chosen
        candidate fails its check, it is left out, its search is done again,
        and the step chooses again from the values it has. FAST-5: each
        parent's entry gets κ_e = κ and λ_e = its best legal reduction, 0
        when it has none, and -1 when it holds every covariate.
        Complexity: O(P·p·n·r) for pass 1, with P parents, and O(n·r) for
        each search and knot of pass 2. Pass 2 values about one knot per
        parent and two for the step in an ordinary step; in the worst case,
        when y is (nearly) linear in one covariate, every pair knot of it
        reaches the floor and pass 2 values them all, O(n²·r/L) for that
        step (L the minspan)."""
        parents = self.parents()
        searched = {k: self.covariates(k) for k in parents}
        excluded: set = set()
        found = {
            (k, j): self.search(k, j, excluded) for k in parents for j in searched[k]
        }
        explicit: dict = {}  # order -> the candidate with explicit values, or None
        while True:
            kept = [c for cands in found.values() for c in cands]
            want = self._contenders(kept)
            new = [c for c in want if c.order not in explicit]
            explicit.update((c.order, None) for c in new)
            explicit.update((c.order, c) for c in self.refine(new))
            legal = [explicit[c.order] for c in want if explicit[c.order] is not None]
            top = _top_two(legal)
            rb = self.check(top[0]) if top else None
            if not top or rb is not None:
                break
            c = top[0]
            excluded.add(c.order)
            explicit[c.order] = None
            found[(c.parent, c.variable)] = self.search(c.parent, c.variable, excluded)
        kappa = 2 * len(self.rss)
        for k in parents:
            own = [c.reduction for c in legal if c.parent == k]
            e = self.slot[k] - 1
            self.lam[e] = max(own, default=0.0) if searched[k] else -1.0
            self.kap[e] = kappa
        if not top:
            return None, None, None
        return top[0], rb, top[1] if len(top) > 1 else None

    @staticmethod
    def _contenders(kept: list[_Candidate]) -> list[_Candidate]:
        """The candidates of pass 1 that pass 2 values: those whose upper bound
        reaches the second largest lower bound of the sure candidates (the
        step's best two, FWD-8) or the largest one of their own parent (its
        λ, FAST-5). Complexity: O(c) for c candidates."""
        floor = _second_largest([c.reduction - c.err for c in kept if c.sure])
        own: dict = {}
        for c in kept:
            if c.sure:
                own[c.parent] = max(own.get(c.parent, -math.inf), c.reduction - c.err)
        return [
            c
            for c in kept
            if c.reduction + c.err >= min(floor, own.get(c.parent, -math.inf))
        ]

    def add(self, c: _Candidate, rb: _scan.Rebuild) -> None:
        """Append the candidate's terms (FWD-6) in the slots κ and κ + 1
        (FWD-9), with one queue entry each (FAST-1: λ = +∞, κ_e = κ), and take
        its basis and RSS. Complexity: O(n + M·p) for the new columns and
        the copies of the term table."""
        auto = self.params["auto_linpreds"]
        x = self.X[:, c.variable]
        if c.kind == KIND_PAIR:
            new = [(_terms.PLUS, c.knot), (_terms.MINUS, c.knot)]
        elif c.kind == KIND_HINGE:
            new = [(_terms.PLUS, c.knot)]
        elif auto:
            new = [(_terms.LINEAR, 0.0)]
        else:
            new = [(_terms.PLUS, _knots.linear_option_knot(x))]
        s = len(self.rss)
        kappa = 2 * s
        dirs, cuts, b = self.dirs[c.parent], self.cuts[c.parent], self.cols[c.parent]
        for i, (code, cut) in enumerate(new):
            d, k = _terms.child_term(dirs, cuts, c.variable, code, cut)
            self.dirs = np.vstack((self.dirs, d))
            self.cuts = np.vstack((self.cuts, k))
            self.cond.append(d, k)
            col = b * _terms.factor(code, x, cut)
            self.cols.append(col)
            self.degree.append(self.degree[c.parent] + 1)
            active = col > 0.0  # KNOT-1; N_b is the weight of these cases (W-4)
            Nb = float(np.count_nonzero(active)) if self.w is None else None
            Nb = _gcv.weight_sum(self.w[active], self.tau) if Nb is None else Nb
            self.spans.append(self._spans(self.degree[-1], Nb))
            self.parent.append(c.parent)
            self.step_of.append(s)
            self.slot.append(kappa + i)
            self.term_at[kappa + i] = len(self.cols) - 1
            self.lam.append(math.inf)
            self.kap.append(kappa)
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
    Complexity: O(S·P·p·n·r) for S steps of P parents."""
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
            with np.errstate(over="ignore"):  # EDGE-6: can pass the range
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
    """Run the forward pass on X (n, p), Y (n,) or (n, K) and the case weights
    w (n,), or None for w_i = 1 exactly; return its record.

    The keyword arguments are the fields of ``MarsParams`` (CORE-2), which
    the core checks; None resolves ``max_terms`` by LIMIT-1 and ``penalty``
    by GCV-4 (the penalty enters only GRSq' of STOP-3). ``fast_k`` and
    ``fast_beta`` set the window and the ageing of the queue (FAST-2,
    FAST-3). Rows with zero weight are dropped first (W-3). A degenerate fit
    (EDGE-1) returns the intercept alone with code ``DEGENERATE``, and
    ``max_terms`` ≤ 2 gives ``NO_ROOM`` (STOP-1). Raises ValueError for
    nonfinite or mismatched input, weights that are negative or all 0, and,
    by EDGE-6, a scaled TSS that is not a positive normal float64 (possible
    only with extreme weights: without weights the scaled Y has a value of
    size at least 1).
    Complexity: O(p·n·log n + S·P·p·n·r·K + n·M²) time for S steps of P
    parents, O(n·(p + M_max + K)) memory.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if X.ndim != 2 or Y.ndim != 2 or Y.shape[0] != X.shape[0] or X.shape[0] < 1:
        raise ValueError(f"X (n, p) and Y (n, K) do not match: {X.shape}, {Y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X and Y must be finite")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (X.shape[0],) or not (np.isfinite(w).all() and (w >= 0).all()):
            raise ValueError(f"w must be finite and at least 0, shape ({len(X)},)")
        if not w.any():
            raise ValueError("every weight is zero; the weights need a positive sum")
        keep = w > 0.0  # W-3: a row with zero weight is not a case
        X, Y, w = (X, Y, w) if keep.all() else (X[keep], Y[keep], w[keep])
    n, p = X.shape
    max_terms = _gcv.default_max_terms(p) if max_terms is None else max_terms
    penalty = _gcv.default_penalty(max_degree) if penalty is None else penalty
    params = {
        "max_degree": max_degree,
        "minspan": minspan,
        "endspan": endspan,
        "adjust_endspan": adjust_endspan,
        "auto_linpreds": auto_linpreds,
        "thresh": float(thresh),
        "penalty": penalty,
        "fast_k": int(fast_k),
        "fast_beta": float(fast_beta),
    }
    # EDGE-1 comes before EDGE-6 here, so a degenerate fit never raises. The
    # core checks EDGE-6 first (its TSS is uncentered): the two differ only when
    # N <= 1 and Y is not constant with an out-of-range TSS, where the core raises.
    if _gcv.is_degenerate(Y, _gcv.total_weight(n, w)[0]):  # EDGE-1, GCV-7
        return _intercept_only(
            p, _gcv.tss(Y, w), Termination.DEGENERATE, record_candidates
        )
    shift = 1 - int(np.frexp(np.max(np.abs(Y)))[1])  # EDGE-6: D·2^shift in [1, 2)
    # FWD-10: shift each response by its data value nearest its weighted mean,
    # then by the weighted mean; the TSS too comes from Yc (CORE-3: rss[0]).
    Yc, _ = _pruning._centered(np.ldexp(Y, shift), w)
    tss = _gcv.tss(Yc, w)
    if not (math.isfinite(tss) and tss >= np.finfo(np.float64).tiny):
        raise ValueError(
            "the scale of y or of the weights is out of range: the total sum of "
            f"squares of y·2^{shift} is {tss}, not a positive normal float64 (EDGE-6)"
        )
    st = _Pass(X, Yc, tss, params, w)
    log: list = []
    termination = _run(st, max_terms, log)
    with np.errstate(over="ignore"):  # EDGE-6: an RSS above the range is inf
        rss = np.ldexp(np.array(st.rss), -2 * shift)
    B = _terms.basis_matrix(X, st.dirs, st.cuts)
    kept = _linalg.independent_columns(B, w)  # FWD-11
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
