"""The forward pass: the steps, the stopping rules and the forward record.

Spec: ``docs/algorithm.md``, "Forward pass" (FWD-1 to FWD-12), "Stopping
rules" (STOP-1 to STOP-7), LA-1 to LA-7, KNOT, SPAN, LIMIT-2, "Fast MARS"
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

Dependence (LA-1, LA-2, LA-4, FWD-11). A candidate's new column (b·x, a hinge)
that LA-4's float64 test (``_linalg.dependent``) finds dependent counts as 0: a
b·x that LA-2 does not count gives no linear candidate and stays out of the set
that LA-2 tests the hinges against and of F_{s+1}, but LA-3's G holds it in
every pair search (#164); a dependent hinge has the gain 0 and stays out of
F_{s+1} (FWD-13). The test uses
the distance to G that the search has already and the uncentered √w-norm of
the column itself (b·x, not the centered b·(x - c) of the projections). FWD-11
is the same test on the √w-scaled columns of the terms, in their order
(``_linalg.independent_columns``). On inputs beyond ordinary data (the spec's
Limits, LIM-1 to LIM-4) the decisions follow float64 rounding.

FWD-12: the occurrences of one single hinge or linear candidate in several
searches of a step (``_Pass.occurrences``) are one candidate, decided by the
first in the order of FWD-5 (``_Pass.decide``); the others keep its reduction for
their parent's λ, and the log's second best is of another key.

Numerics. Y is multiplied by a power of 2 first (EDGE-6) and centered in two
steps, each response by its data value nearest its weighted mean and then by
the weighted mean (FWD-10, LA-6; ``_pruning`` does the same), so that the
rounding is of the size of the spread of Y and not of its mean, whatever the
weights; the TSS comes from the centered Y. Every RSS in the record is on the
original scale, where one can underflow to 0 or, for |y|·√n above about
1e154, overflow to +inf (EDGE-6 names the underflow). No absolute epsilon.
Ties follow FWD-5. No input is written to. Complexity: O(p·n·log n) for the
sorts, O(P·p·n·r) per step for P parents and rank r, O(n·M²) for FWD-11;
memory O(n·(p + M_max)). The dependence tests add O(n·r) for each knot that
pass 2 values and O(n) for each pair search; merging adds O(p) for each
candidate that pass 2 values and O(n·r) for each alternate search that exists.

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
#: STOP-3: the pass stops when GRSq' is below this value, with code 3 (v2: code
#: 2 when GRSq' is below GRSQ_NEG_INF_FLOOR, -inf included).
GRSQ_FLOOR = -10.0
GRSQ_NEG_INF_FLOOR = -1000.0
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
    is the parent's forward index (TERM-6). ``key`` names the rows that a single
    hinge or a linear candidate on a parent of degree 1 or more adds, so that two
    occurrences of one candidate have one key (FWD-12)."""

    reduction: float
    order: tuple[int, int, int]  # (table row of the parent, covariate, place)
    parent: int
    variable: int
    kind: int
    knot: float  # NaN for a linear term
    err: float = 0.0
    sure: bool = True
    key: tuple | None = None  # FWD-12: (kind, rows added); None never merges


class _Setup(NamedTuple):
    """The scalar results of the search of one covariate x on the parent b in
    this step (``_Pass.setup``): its kind (LA-7), the linear candidate's
    reduction (0.0 in a single-hinge search), whether LA-4 finds b·x dependent
    on B in a pair search (``dep``: it counts as 0, LA-1, LA-2, so there is no
    linear candidate and LA-2 tests the hinges against B, while LA-3's G still
    holds b·x), and ‖v⊥‖ of v = √w·b·(x - c) against B."""

    pair: bool
    lin: float
    dep: bool
    norm: float


def _top_two(candidates: list[_Candidate]) -> list[_Candidate]:
    """Return the best two, by reduction and then by the order of FWD-5, of
    different keys: later occurrences of a merged candidate do not count
    (FWD-12, FWD-8). Complexity: O(c·log c) for c candidates."""
    out: list[_Candidate] = []
    seen: set = set()
    for c in _ranked(candidates):
        if c.key is not None:
            if c.key in seen:
                continue
            seen.add(c.key)
        out.append(c)
        if len(out) == 2:
            break
    return out


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

    The basis Q (n, r) is the leading r columns of a C-order buffer
    (``_Qbuf``), allocated once with room for the forward basis and 2 more
    columns: the forward pass only appends columns, and old columns never
    change. Column r is scratch: a pair search puts its unit vector of b·x
    there, so the scan reads G = Q and b·x with no copy. ``check`` rebuilds
    the winner into a second buffer (``_spare``: Q copied, then the new
    columns), so no search can write over it, and ``add`` swaps the two.
    ``_coefs[k, j]`` keeps Qᵀv of the first Gram-Schmidt pass of a search's
    v = √w·b·(x - c) across steps (r values; neither v nor old columns
    change), so the next step computes only the new columns' coefficients.
    ``_setups`` keeps the scalar results of each search of the step
    (``_Setup``), not its vectors, so memory stays O(n·(p + M)); ``basis``
    builds a pair search's q and residuals again when another search has used
    the scratch column since.
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
        # O(n·p), once: x - c with contiguous columns, and the scan's gaps
        self._xc = np.asfortranarray(X - self.center)
        self._gaps = [_scan.unit_gaps(self.sorted_x[:, j]) for j in range(p)]
        self.cols = [np.ones(n)]
        self._swcols = [self.sw]  # √w·b of each term, O(n·M)
        self._bx_norms: dict = {}  # (k, j) -> ‖√w·b·x‖, which never changes
        self.degree = [0]
        self.spans = [self._spans(0, self.N)]
        L, E = self.spans[0]
        active = np.ones(n, dtype=bool)
        self._knot_cache: dict = {}  # (k, j) -> the knots of term k > 0 (KNOT-7)
        self._knot_count = 0
        self._scan_stores: dict = {}  # (k, j) -> the store of knot_scan
        self._scan_size = 0
        self.knots = [  # the intercept's knots, found once per covariate (KNOT-7)
            self._knots(j, active, L, E) for j in range(p)
        ]
        raw = float(n) if w is None else math.fsum(w)  # ‖√w‖², not snapped
        self.wsum = raw
        width = min(params.get("max_terms", 21), n) + 2
        self._Qbuf = np.empty((n, width))  # C order: a row is contiguous
        self._spare = np.empty((n, width))
        self._Qbuf[:, 0] = self.sw / math.sqrt(raw)
        self._r = 1
        self._coefs: dict = {}  # (k, j) -> Qᵀv of the first pass, kept
        self._coef_count = 0  # values in _coefs, within _COEF_BUDGET
        self._setups: dict = {}  # (k, j) -> _Setup of this step
        self._scratch: tuple | None = None  # (k, j, E) of the q in column r
        self._E, _ = _linalg.orthogonalize(self.Q, self.Yw)
        self._e2: FloatArray | None = None  # ‖E_k‖², when a pair search needs it
        self.rss = [tss]
        self.dirs, self.cuts = _terms.intercept_terms(p)
        self.lin_knot = [_knots.linear_option_knot(X[:, j]) for j in range(p)]
        self.index = {self._row_key(self.dirs[0], self.cuts[0]): 0}  # FWD-12
        self.parent, self.step_of = [-1], [0]
        self.slot, self.term_at = [1], {1: 0}  # slot 1 holds the intercept
        self.lam, self.kap = [math.inf], [0]
        self.rows = {0: 0}  # before the first step the table is the intercept

    @property
    def Q(self) -> FloatArray:  # noqa: N802 (the spec's name)
        """The orthonormal basis of the √w-scaled forward basis F_s (FWD-13),
        column 0 the intercept direction: a view of the buffer."""
        return self._Qbuf[:, : self._r]

    @Q.setter
    def Q(self, Q: FloatArray) -> None:  # noqa: N802
        """Take a new basis: a rebuild into the spare buffer (``add``), which
        holds Q and appends, so the two buffers swap; or any other array
        (tests), which is copied in and forgets the kept coefficients.
        Complexity: O(n·r) for a copy, else O(1)."""
        r = Q.shape[1]
        if Q.base is self._spare:
            self._Qbuf, self._spare = self._spare, self._Qbuf
        elif Q.base is not self._Qbuf:
            self._reserve(r + 2)
            self._Qbuf[:, :r] = Q
            self._coefs, self._coef_count = {}, 0
        self._r = r
        self._reserve(r + 2)
        self._setups, self._scratch = {}, None

    @property
    def E(self) -> FloatArray:  # noqa: N802
        """The √w-scaled residuals of the responses on Q (n, K)."""
        return self._E

    @E.setter
    def E(self, E: FloatArray) -> None:  # noqa: N802
        self._E, self._e2 = E, None
        self._setups, self._scratch = {}, None

    def _reserve(self, width: int) -> None:
        """Make both buffers at least ``width`` columns wide, keeping Q.
        Complexity: O(n·r) when they grow (at most a few times a pass)."""
        n, have = self._Qbuf.shape
        if have < width:
            buf = np.empty((n, max(width, 2 * have)))
            buf[:, : self._r] = self._Qbuf[:, : self._r]
            self._Qbuf, self._spare = buf, np.empty(buf.shape)

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

    @staticmethod
    def _row_key(dirs_row: IntArray, cuts_row: FloatArray) -> tuple[bytes, bytes]:
        """The key of a term's row (TERM-6). Complexity: O(p)."""
        return dirs_row.tobytes(), (cuts_row + 0.0).tobytes()  # -0.0 is 0.0

    def _adds(self, kind: int, j: int, knot: float) -> tuple[int, float]:
        """The factor (code and cut) that a candidate of this kind adds on the
        covariate j (FWD-6). Complexity: O(1)."""
        if kind == KIND_HINGE:
            return _terms.PLUS, knot
        if self.params["auto_linpreds"]:
            return _terms.LINEAR, 0.0
        return _terms.PLUS, self.lin_knot[j]

    def _child(self, c: _Candidate) -> tuple[IntArray, FloatArray]:
        """The row of the term that candidate c adds (a single hinge or a
        linear candidate). Complexity: O(p)."""
        code, cut = self._adds(c.kind, c.variable, c.knot)
        return _terms.child_term(
            self.dirs[c.parent], self.cuts[c.parent], c.variable, code, cut
        )

    def keyed(self, c: _Candidate) -> _Candidate:
        """c with its FWD-12 key: that of the rows it adds, for a single hinge
        or a linear candidate on a parent of degree 1 or more (the others add
        rows that no other search adds). Complexity: O(p)."""
        if c.kind == KIND_PAIR or self.degree[c.parent] == 0:
            return c
        d, k = self._child(c)
        return c._replace(key=(c.kind, *self._row_key(d, k)))

    def occurrences(self, c: _Candidate) -> list[_Candidate]:
        """The occurrences of candidate c in this step, c included, in the order
        of FWD-5 (FWD-12): the searches of a parent that this step visits and a
        covariate g, of the same kind as c (single hinge, or pair search for a
        linear candidate), that add the same rows, with the same knot. The first
        one decides. Only the factors g of the added term that a search of c's
        kind adds can be the new one. Complexity: O(p·n·r) for each alternate
        search that exists, O(p) otherwise, and O(1) when c.key is None."""
        out = [c]
        if c.key is None:
            return out
        d, k = self._child(c)
        for g in np.flatnonzero(d).tolist():
            code, cut = self._adds(c.kind, g, float(k[g]))
            if g == c.variable or d[g] != code or k[g] != cut:
                continue
            d2, k2 = d.copy(), k.copy()
            d2[g], k2[g] = _terms.ABSENT, 0.0
            parent = self.index.get(self._row_key(d2, k2))
            if parent is None or parent not in self.rows:
                continue
            place = 0
            if c.kind == KIND_HINGE:
                at = np.flatnonzero(self.knots_of(parent, g).knots == cut)
                if at.size == 0:
                    continue
                place = int(at[0]) + 1
            if self.setup(parent, g).pair != (c.kind == KIND_LINEAR):
                continue
            order = (self.rows[parent], g, place)
            knot = cut if c.kind == KIND_HINGE else math.nan
            out.append(c._replace(order=order, parent=parent, variable=g, knot=knot))
        return sorted(out, key=lambda o: o.order)

    def decide(self, first: _Candidate) -> _Candidate | None:
        """The explicit values of the first occurrence of a merged candidate, or
        None when it is not legal (FWD-12: its LA-3 decision, its reduction and
        its legality hold for every occurrence). Complexity: O(n·(r + K))."""
        if first.kind == KIND_LINEAR:
            sr = self.setup(first.parent, first.variable)
            lin = sr.lin
            if not sr.pair or lin <= 0.0:
                return None
            return first._replace(reduction=lin, err=0.0, sure=True)
        out = self.refine([first])
        return out[0] if out else None

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

    # Budgets of the two caches, in multiples of n·(p + M) (a test lowers them).
    _KNOT_BUDGET = 1
    _SCAN_BUDGET = 4
    _COEF_BUDGET = 1

    def knots_of(self, k: int, j: int) -> _knots.KnotCandidates:
        """The candidate knots of parent k and covariate j (KNOT-1 to KNOT-3):
        the active cases of k are those where its column is positive.
        Complexity: O(n) the first time, and O(1) for the intercept and for
        the knots kept from an earlier step."""
        if k == 0:
            return self.knots[j]
        kc = self._knot_cache.get((k, j))
        if kc is None:
            L, E = self.spans[k]
            kc = self._knots(j, self.cols[k][self.order[:, j]] > 0.0, L, E)
            # A term's column and spans never change, so its knots are kept,
            # within a budget of n·(p + M) knots (2 arrays each, the knots and the
            # splits, so 2 values each) that keeps memory O(n·(p + M)).
            n, p = self.X.shape
            cap = 2 * self._KNOT_BUDGET * n * (p + len(self.cols))
            if self._knot_count + 2 * kc.knots.size <= cap:
                self._knot_cache[k, j] = kc
                self._knot_count += 2 * kc.knots.size
        return kc

    def _scan_store(self, k: int, j: int, knots: int) -> tuple[dict, int] | None:
        """The store of ``_scan.knot_scan`` for parent k and covariate j, with
        the number of leading columns of its Q that stay the same (the columns
        of the current basis; rebuilds only append), or None when keeping
        them (K·(m + 2) values for K knots: hᵀq, ‖h‖² and D) would pass a
        budget of 4·n·(p + M) values, so that memory stays
        O(n·(p + M)). The store changes no value (``knot_scan``).
        Complexity: O(1)."""
        m = self._r - (k == 0)
        store = self._scan_stores.get((k, j))
        old = 0 if store is None else store["size"]
        n, p = self.X.shape
        cap = self._SCAN_BUDGET * n * (p + len(self.cols))
        if self._scan_size - old + knots * (m + 2) > cap:
            if store is not None:
                del self._scan_stores[k, j]
                self._scan_size -= old
            return None
        if store is None:
            store = self._scan_stores[k, j] = {"size": 0}
        self._scan_size += knots * (m + 2) - old
        store["size"] = knots * (m + 2)
        return store, m

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
        column of term k: b·(x_j - c_j), c_j the covariate's middle value (the
        plan's "Fast path"). It spans with B what b·x_j spans, since B holds b.
        Complexity: O(n)."""
        return self.cols[k] * (self.X[:, j] - self.center[j])

    def _v(self, k: int, j: int) -> FloatArray:
        """v = √w·b·(x_j - c_j), the √w-scaled ``linear_column``, in the
        product order (√w·b)·(x - c). Complexity: O(n)."""
        return self._swcols[k] * self._xc[:, j]

    def setup(self, k: int, j: int) -> _Setup:
        """The search of covariate j on parent k: Gram-Schmidt of
        v = √w·b·(x - c) gives A of LA-7, and so the kind, where V holds the
        covariates of b and x. In a pair search LA-2 tests b·x with LA-4
        (``_linalg.dependent``): its distance to B is that of b·(x - c), and its
        norm that of b·x itself, not centered; when it counts, its unit vector q
        goes to the scratch column and the residuals lose their part along q
        (``_linalg.remove_direction``), for ``basis``. Every reduction goes
        through ``_capped``, here, in ``search`` and in ``refine``. The result
        is kept for the step. Complexity: O(n·(a + r)) the first time in a step,
        a the columns added since the search last ran (``_linalg.cgs2``), plus
        O(n·K) in a pair search; O(1) after."""
        sr = self._setups.get((k, j))
        if sr is not None:
            return sr
        o = _linalg.cgs2(self.Q, self._v(k, j), self._coefs.get((k, j)))
        self._keep_coefs(k, j, o.coef)
        V = [*np.flatnonzero(self.dirs[k]).tolist(), j]
        pair = o.norm > 0.0 and _linalg.pair_search(o.norm**2, self.variances[V])
        dep, lin = False, 0.0
        if pair:
            norm = self._bx_norms.get((k, j))
            if norm is None:
                norm = float(np.linalg.norm(self._swcols[k] * self.X[:, j]))
                self._bx_norms[k, j] = norm
            dep = _linalg.dependent(o.norm, norm)  # LA-2: b·x counts as 0
            if not dep:
                lin = float(self._capped(float(np.sum(self._extend(k, j, o) ** 2))))
        sr = self._setups[k, j] = _Setup(pair, lin, dep, o.norm)
        return sr

    def _keep_coefs(self, k: int, j: int, coef: FloatArray) -> None:
        """Keep Qᵀv of search (k, j) for the next step, within a budget of
        n·(p + M) values, so that memory stays O(n·(p + M)) however many
        searches the pass makes (O(P·p·M) values without it). Past the budget
        the entry is dropped, and the next call makes its full first pass,
        which gives the same values. Complexity: O(1)."""
        old = self._coefs.pop((k, j), None)
        if old is not None:
            self._coef_count -= old.size
        n, p = self.X.shape
        cap = self._COEF_BUDGET * n * (p + len(self.cols))
        if self._coef_count + coef.size <= cap:
            self._coefs[k, j] = coef
            self._coef_count += coef.size

    def _extend(self, k: int, j: int, o: _linalg.Orthogonal) -> FloatArray:
        """Put q = v⊥/‖v⊥‖ of a pair search in the scratch column and keep the
        residuals on G = (Q, q); return qᵀE. The rank-1 update keeps E's own
        departure from QᵀE = 0, of order u·‖E_k‖, which the scan's precondition
        allows while ‖E_k‖ keeps most of its size; when it loses more than half
        of its square for some k (q fits much of y), E is orthogonalized against
        G again, so that the departure stays small next to the new ‖E_k‖.
        Complexity: O(n·K), or O(n·r·K) in that case."""
        q = o.perp / o.norm
        E, t = _linalg.remove_direction(q, self._E)
        self._Qbuf[:, self._r] = q
        if self._e2 is None:
            self._e2 = np.einsum("ik,ik->k", self._E, self._E)
        if np.any(2.0 * np.einsum("ik,ik->k", E, E) < self._e2):
            E = _linalg.orthogonalize(self._Qbuf[:, : self._r + 1], self._E)[0]
        self._scratch = (k, j, E)
        return t

    def basis(self, k: int, j: int, la3: bool = False) -> tuple[FloatArray, FloatArray]:
        """The orthonormal basis Q and the residuals E on the columns that LA-2
        tests the hinges of the search of covariate j on parent k against: B,
        and b·x in a pair search where LA-2 counts it (LA-7, LA-2). With
        ``la3``, on LA-3's G instead, which holds b·x in every pair search,
        also where LA-2 does not count it (#164); there q = v⊥/‖v⊥‖ is
        orthogonalized against B once more, since ‖v⊥‖ can be far below ‖v‖.
        The scan requires both orthogonalized to working precision
        (``_scan.knot_scan``). When another search has used the scratch column
        since ``setup``, q comes again from the kept coefficients, with the
        same steps and so the same values. The arrays are valid until the next
        search. Complexity: O(1), or O(n·(r + K))."""
        sr = self.setup(k, j)
        if not sr.pair or (sr.dep and not la3):
            return self.Q, self._E
        if self._scratch is None or self._scratch[:2] != (k, j):
            v, known = self._v(k, j), self._coefs.get((k, j))
            o = _linalg.cgs2(self.Q, v, known)
            if sr.dep:
                o = _linalg.cgs2(self.Q, o.perp / o.norm, twice=True)
            self._extend(k, j, o)
        return self._Qbuf[:, : self._r + 1], self._scratch[2]

    def search(self, k: int, j: int, excluded: set) -> list[_Candidate]:
        """Pass 1 for parent k and covariate j: every candidate that can be one
        of the best two of its search by the scan's values and error bounds
        (FWD-3, FWD-4, LA-3), ranked, leaving out the places in ``excluded``.
        A knot is dropped when its bounds make it illegal for sure, and kept
        when its upper bound reaches the second largest lower bound of the
        sure candidates. Complexity: O(n·r)."""
        sr = self.setup(k, j)
        o, kc, row = self.order[:, j], self.knots_of(k, j), self.rows[k]
        bw = self._swcols[k][o]  # √w·b in the order of x

        def scan(Q, E):
            return _scan.knot_scan(
                self.sorted_x[:, j],
                bw,
                Q,
                E,
                kc.split,
                intercept=k == 0,
                _checked=True,
                _cache=self._scan_store(k, j, kc.split.shape[0]),
                _order=o,
                _gaps=self._gaps[j],
            )

        sc = scan(*self.basis(k, j))
        ratio, ratio_err = sc.ratio, sc.ratio_err
        if sr.pair and sr.dep:  # LA-3's G holds the b·x that LA-2 left out
            g = scan(*self.basis(k, j, la3=True))
            ratio, ratio_err = g.ratio, g.ratio_err
        red, err = self._capped(sc.gain + sr.lin), sc.gain_err
        tau = _linalg.collinearity_tolerance(len(self.rss))  # this is step s
        top = self.max_legal()
        possible = (ratio + ratio_err >= tau) & (red + err > 0.0)
        possible &= red - err <= top
        sure = (ratio - ratio_err >= tau) & (red - err > 0.0)
        sure &= red + err <= top
        for r, var, place in excluded:
            if (r, var) == (row, j) and place > 0:
                possible[place - 1] = False
        found = []
        if sr.pair and sr.lin > 0.0 and (row, j, 0) not in excluded:
            lin = _Candidate(sr.lin, (row, j, 0), k, j, KIND_LINEAR, math.nan)
            found.append(self.keyed(lin))
        floor = _second_largest(
            [*(red - err)[possible & sure], *(c.reduction for c in found)]
        )
        kind = KIND_PAIR if sr.pair else KIND_HINGE
        for i in np.flatnonzero(possible & (red + err >= floor)):
            t, e = float(kc.knots[i]), float(err[i])
            c = _Candidate(
                float(red[i]), (row, j, i + 1), k, j, kind, t, e, bool(sure[i])
            )
            found.append(self.keyed(c))
        return _ranked(found)

    def refine(self, cands: list[_Candidate]) -> list[_Candidate]:
        """Pass 2: the explicit values of the knots (``_scan.exact_knot``: one
        Gram-Schmidt of the hinge gives rho, LA-4's test of the hinge against
        the columns of LA-2 and the reduction, 0 for a dependent hinge, LA-2;
        a second one gives rho on LA-3's G where LA-2 left b·x out), with LA-3
        and FWD-4 decided on them; the illegal ones are left out, and the
        linear candidates, whose values are explicit already, stay.
        Complexity: O(n·(r + K)) per search and per knot."""
        out = [c for c in cands if c.kind == KIND_LINEAR]
        s, top = len(self.rss), self.max_legal()
        for k, j in sorted(
            {(c.parent, c.variable) for c in cands if c.kind != KIND_LINEAR}
        ):
            sr = self.setup(k, j)
            Q, E = self.basis(k, j)
            G = self.basis(k, j, la3=True) if sr.pair and sr.dep else None
            x, b = self.X[:, j], self.cols[k]
            for c in cands:
                if (c.parent, c.variable) != (k, j) or c.kind == KIND_LINEAR:
                    continue
                h = b * _terms.factor(_terms.PLUS, x, c.knot)
                rho, gain = _scan.exact_knot(Q, E, h, self.w, N=self.wsum)
                if G is not None:  # LA-3's G holds the b·x that LA-2 left out
                    rho = _scan.exact_knot(*G, h, self.w, N=self.wsum)[0]
                red = float(self._capped(gain + sr.lin))
                if not _linalg.knot_rejected(rho, s) and 0.0 < red <= top:
                    out.append(c._replace(reduction=red, err=0.0, sure=True))
        return out

    def columns(self, c: _Candidate) -> FloatArray:
        """Return the √w-scaled columns that span the candidate's terms (b·x
        before the hinge of a pair, FWD-6), without a column that LA-2 did not
        count (FWD-13): a dependent b·x, and a hinge that LA-4 finds dependent
        on the columns that LA-2 tests it against (``basis``), by the test of
        ``_scan.exact_knot`` on the same column up to a power of 2 (#164).
        Such a hinge adds only rounding noise to the RSS. LA-3 rejects it
        first unless its uncentered w-norm is more than √τ·1e7, about 3e4,
        times its centered one, since LA-3's G holds the columns that LA-2
        tests it against. Complexity: O(n·(r + K))."""
        x, b = self.X[:, c.variable], self.cols[c.parent]
        cols = []
        if self._spans_bx(c):
            cols.append(self.sw * self.linear_column(c.parent, c.variable))
        if c.kind != KIND_LINEAR:
            h = self.sw * (b * _terms.factor(_terms.PLUS, x, c.knot))
            o = _linalg.cgs2(self.basis(c.parent, c.variable)[0], h)
            if not _linalg.dependent(o.norm, o.vnorm):
                cols.append(h)
        return np.column_stack(cols) if cols else np.empty((len(x), 0))

    def _spans_bx(self, c: _Candidate) -> bool:
        """Whether b·x is among the candidate's new columns (FWD-6, LA-2)."""
        return c.kind != KIND_HINGE and not (
            c.kind == KIND_PAIR and self.setup(c.parent, c.variable).dep
        )

    def check(self, c: _Candidate) -> _scan.Rebuild | None:
        """Build the candidate again (``_scan.rebuild``) and apply FWD-4 to its
        rebuilt RSS; return the rebuild, or None when it fails. LA-3 is not
        tested again: pass 2 decided it on the same Gram-Schmidt of the same
        columns, so the ratio would be the same up to rounding. The rebuilt RSS and
        pass 2's reduction can differ in the last bits, so a knot at MaxLegal
        can fail here. Complexity: O(n·r·K)."""
        rb = _scan.rebuild(self.Q, self.Yw, self.columns(c), _buffer=self._spare)
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
        decided: dict = {}  # key -> (first occurrence with explicit values or None)
        while True:
            kept = [c for cands in found.values() for c in cands]
            want = self._contenders(kept)
            new = [c for c in want if c.key is None and c.order not in explicit]
            explicit.update((c.order, None) for c in new)
            explicit.update((c.order, c) for c in self.refine(new))
            legal = []
            for c in want:
                if c.key is None:
                    r = explicit[c.order]
                else:  # FWD-12: the first occurrence decides for all
                    if c.key not in decided:
                        decided[c.key] = self.decide(self.occurrences(c)[0])
                    r = decided[c.key]
                    if r is not None and r.order != c.order:
                        legal.append(r)
                        r = r._replace(
                            order=c.order, parent=c.parent, variable=c.variable
                        )
                if r is not None:
                    legal.append(r)
            top = _top_two(legal)
            rb = self.check(top[0]) if top else None
            if not top or rb is not None:
                break
            c = top[0]
            if c.key is None:
                dead = [c]
                explicit[c.order] = None
            else:
                dead = self.occurrences(c)
                decided[c.key] = None
            for d in dead:
                excluded.add(d.order)
            for k, j in {(d.parent, d.variable) for d in dead}:
                found[k, j] = self.search(k, j, excluded)
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
        step's best two, FWD-8, the occurrences of one merged candidate counted
        once, FWD-12) or the largest one of their own parent (its λ, FAST-5).
        Complexity: O(c) for c candidates."""
        lower: dict = {}  # the occurrences of a merged candidate count once
        for c in kept:
            if c.sure:
                at = c.key if c.key is not None else c.order
                lower[at] = max(lower.get(at, -math.inf), c.reduction - c.err)
        floor = _second_largest(list(lower.values()))
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
            self.index[self._row_key(d, k)] = len(self.cols)
            col = b * _terms.factor(code, x, cut)
            self.cols.append(col)
            self._swcols.append(col if self.w is None else self.sw * col)
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
        # Q holds F_{s+1} (FWD-13): ``columns`` leaves out a b·x or a hinge
        # that LA-2 did not count, so a pair of case (b) adds only its hinge's
        # direction.
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
        low = grsq < GRSQ_NEG_INF_FLOOR  # -inf too (STOP-3, FAST-6)
        return Termination.GRSQ_NEG_INF if low else Termination.GRSQ_LOW
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
        "max_terms": max_terms,
    }
    # EDGE-6 (v2): the scale of Y and the TSS check come before the degenerate
    # test of EDGE-1, as in the core, so that a degenerate fit whose scaled TSS
    # underflows raises here too. The check needs a response that is not constant.
    D = float(np.max(np.abs(Y)))
    shift = 0 if D == 0.0 else 1 - math.frexp(D)[1]  # D·2^shift in [1, 2)
    constant = bool(np.all(Y[0] == Y))
    if not constant:
        tss0 = _gcv.tss(np.ldexp(Y, shift), w)
        if not (math.isfinite(tss0) and tss0 >= np.finfo(np.float64).tiny):
            raise ValueError(
                "the scale of y or of the weights is out of range: the total sum "
                f"of squares of y·2^{shift} is {tss0}, not a positive normal "
                "float64 (EDGE-6)"
            )
    if _gcv.is_degenerate(Y, _gcv.total_weight(n, w)[0]):  # EDGE-1, GCV-7
        return _intercept_only(
            p, _gcv.tss(Y, w), Termination.DEGENERATE, record_candidates
        )
    # FWD-10: shift each response by its data value nearest its weighted mean,
    # then by the weighted mean; the TSS too comes from Yc (CORE-3: rss[0]).
    Yc, _ = _pruning._centered(np.ldexp(Y, shift), w)
    tss = _gcv.tss(Yc, w)
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
