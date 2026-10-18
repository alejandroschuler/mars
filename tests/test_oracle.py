"""The oracle tests: the fast code in ``pymars/`` against the reference
implementation in ``tests/reference/mars_ref.py``.

VALIDATION_PLAN.md, "Tests and validation folders" (``test_oracle.py``),
"Ties" and "What is compared, and the tolerances"; docs/algorithm.md, CORE-3
and CORE-5 (the records), LA-5 (the accuracy bounds) and STOP-7 (the
near-ties). Different agents wrote the two implementations from the spec, so
agreement on broad data is evidence for both, and a disagreement is a finding
for one of them. Every later change to the fast code must pass these tests.

What is compared:

- The forward pass: ``_forward.forward_pass`` against the forward record of
  ``mars_ref.fit_mars``, step by step: the terms of each step (parent,
  covariate, codes and knots, exactly: knots are data values), the RSS after
  each step within LA-5's eps (1e-8 of the RSS before the step plus the terms
  in sqrt(TSS V) and TSS), the second-best candidate of the candidate log (its
  identity exactly, its RSS by LA-5); the added rows, not their parents, since
  two programs can add the same rows from different parents (FWD-12); and,
  when every step agrees, the termination code and the kept terms (FWD-11).
- The pruning pass: ``_pruning.pruning_pass`` and ``final_fit`` against
  ``mars_ref.prune`` on random bases, with and without weights, for K = 1 and
  K >= 2: the removals and the subsets exactly (up to PRUNE-10's near-ties),
  the RSS and GCV of each size within LA-5's eps with R = V, the selected
  terms, and the final coefficients and statistics by the plan's tolerance
  table.
- The whole fit: ``compare_fits`` compares ``_core.fit_mars`` with
  ``mars_ref.fit_mars`` through ``MarsFit.from_dict`` on hypothesis data:
  the resolved values, the forward records as above, and then the pruning
  records, the selection and the final fit.

Near-ties. Where two candidates or a threshold are within rounding, the two
programs may choose differently, and the paths after two different choices
cannot be compared (plan, "Ties"). The comparison walks the steps while both
programs choose alike, also through a near-tie that both decide the same way.
At the first step where they differ, the reference's values decide whether
the step is a near-tie: the two chosen candidates are within 1e-7 of the RSS
before the step, or one of them lies in a band of LA-5 or STOP-7 (its
collinearity ratio, the pair rule's A_w, the limits of FWD-4, a stopping
threshold, a near-dependence). A near-tie ends the comparison of that fit,
and it is counted; any other difference fails. The fast code's RSS of its own
choice must still meet LA-5, so a wrong value cannot pass as a tie. With
``PYMARS_ORACLE_TALLY`` set to a folder, each process writes its counts there,
and gate C reports the share of fits that stop at a near-tie, by reason.

Spec v3 (#157). Every eps and delta takes cond and n (LA-5: c = max(1e-14,
u cond sqrt(n))), with cond that of the compared columns, unit-norm and
sqrt(w)-scaled, after LA-4's test (``compare.basis_cond``): the forward basis
F_s of FWD-13 for an RSS of the path, F_s with a candidate's new columns
(b x as the spec forms it, then the hinge) for a candidate, and the subset for
the pruning pass. Each bound is tried with cond = 1 first, and cond, an SVD,
is computed only where that fails. A value that misses its bound where cond
exceeds 1e6, or where an LA-4 ratio of the compared columns is within a
factor 2 of 1e-7, is a near-dependence (STOP-7): the comparison stops there
with the reason ``near-dependence``. The hypothesis cases draw only ordinary
data (spec, Limits; ``ordinary``), at every supported degree (1 to 3; the
fixture fits of degree 4 or more run at degree 3, so r_d = 30 holds for
them); ``test_limits_end_with_a_valid_model`` runs inputs outside it and
checks only that both programs end with a valid model.

``SUPPORTED`` lists the settings of the forward pass that the fast code
supports. The strategies and the fixture cases take every setting from it, so
no test is skipped, and each stage of T11 widens the tests by editing it.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from reference import mars_ref

from pymars import _core, _forward, _pruning

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "validation" / "fixtures"
# The fixture cases import the harness and the simulation code in validation/.
# pytest collects tests/ before validation/, and an installed pymars (the CI
# job with the lowest dependencies) leaves the repository root off sys.path;
# appending it keeps the installed pymars first.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from validation.harness import compare  # noqa: E402

# ---------------------------------------------------------------------------
# The settings of the forward pass that the oracle draws: every degree (T11
# stage 2), and fast_k > 0 and weights (stage 3), with STOP-7's band for two
# queue values (FAST-3) in _Step.queue_bands. Several responses come next.
# The pruning pass takes weights and several responses already.

SUPPORTED = {
    "max_degree": (1, 2, 3),
    "fast_k": (0, 1, 3, 5, 20),
    "weights": (False, True),
    "responses": (1,),
}

FORWARD_PARAMS = {f.name for f in dataclasses.fields(mars_ref.Params)} - {
    "pmethod",
    "nprune",
}

# LA-5 bounds each RSS V by eps(V, R, cond) = 1e-8 R + c sqrt(TSS V) +
# (c/2)^2 TSS, with c = max(1e-14, u cond sqrt(n)) (``compare.la5_eps``); two
# candidates within 1e-7 of the RSS before their step are a near-tie (plan,
# "Ties"); STOP-7's delta is 2 eps(R, R, cond) for the RSS R that it names;
# LA-5's bands for rho and A_w are 1e-6 of their thresholds. PRUNE-10: two
# subsets of one size are a near-tie when their RSS values differ by at most
# 1e-7 of the lower one plus 2 eps(V, V, cond), V the lower; above cond 1e6
# a value is a near-dependence (STOP-7).
TIE = 1e-7
BAND = 1e-6
PRUNE_TIE = 1e-7
EXACT_FIT = 1e-14  # FWD-5: a candidate RSS at most EXACT_FIT RSS_s is 0
COND_LIMIT = compare.COND_LIMIT
NEAR_DEPENDENCE = "near-dependence"


def eps(V: float, R: float, tss: float, cond: float, n: int) -> float:
    """LA-5's bound on the error of an RSS value V (V* in the rule), at the
    scale R, for the total sum of squares tss, the condition number cond of
    the columns that V projects onto, and n cases."""
    return compare.la5_eps(V, R, tss, cond, n)


def delta(R: float, tss: float, cond: float, n: int) -> float:
    """STOP-7's delta: the error of a difference of two values, each within
    eps(R, R, cond) of its exact value."""
    return 2 * eps(R, R, tss, cond, n)


class _Stop(Exception):  # noqa: N818
    """The comparison ends at a near-dependence of STOP-7: a value missed its
    LA-5 bound where LA-5 states none (module docstring)."""


def _dependence(columns: np.ndarray, w: np.ndarray, first: int = 0) -> bool:
    """STOP-7's near-dependence in the columns: an LA-4 ratio of a column from
    ``first`` on within a factor 2 of 1e-7, or cond of the kept columns above
    1e6 (LA-5)."""
    ratios = compare.la4_ratios(columns, w)[first:]
    return compare.is_near_dependence(ratios) or not (
        compare.basis_cond(columns, w) <= COND_LIMIT
    )


def _within(case, a, b, V, R, tss, n, columns, what: str) -> None:
    """a and b (b the reference's) within eps(V, R, cond) of LA-5. The bound
    is tried with cond = 1 first; only when that fails is cond computed from
    ``columns()``, a callable that returns (columns, w), since the cost is an
    SVD. A miss where the columns have a near-dependence raises _Stop; any
    other miss fails."""
    if a == b or abs(a - b) <= eps(V, R, tss, 1.0, n):
        return
    cols, w = columns()
    if abs(a - b) <= eps(V, R, tss, compare.basis_cond(cols, w), n):
        return
    if _dependence(cols, w):
        raise _Stop(what)
    _fail(case, f"{what}: fast {a!r}, reference {b!r} (LA-5)")


PAIR, SINGLE, LINEAR = mars_ref.PAIR, mars_ref.SINGLE, mars_ref.LINEAR
RSQ_HIGH = mars_ref.RSQ_HIGH
LIMIT_CODES = {mars_ref.NO_ROOM, mars_ref.TERM_LIMIT}

# The counts of the fits compared, by group (module docstring).
TALLY: collections.Counter = collections.Counter()


@pytest.fixture(scope="module", autouse=True)
def _write_tally():
    yield
    folder = os.environ.get("PYMARS_ORACLE_TALLY")
    if folder:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        path = Path(folder) / f"tally-{worker}.json"
        path.write_text(json.dumps(dict(TALLY), sort_keys=True, indent=1))


@dataclasses.dataclass(frozen=True)
class Outcome:
    """How a comparison ended: ``steps`` steps (or stages) compared, and the
    reason and the step of the near-tie that ended it, or None."""

    steps: int
    near_tie: str | None = None
    step: int | None = None
    plan_tie: bool = False  # a step of either log within TIE (plan, "Ties")


def _count(group: str, outcome: Outcome) -> None:
    TALLY[f"{group}: fits"] += 1
    TALLY[f"{group}: steps compared"] += outcome.steps
    if outcome.near_tie is not None:
        TALLY[f"{group}: near-tie stops"] += 1
        TALLY[f"{group}: near-tie stops ({outcome.near_tie})"] += 1
    TALLY[f"{group}: plan near-ties"] += outcome.plan_tie


def _plain(rec) -> dict | None:
    """A record as the dict of CORE-5: a NamedTuple of ``_forward`` or
    ``_pruning``, an object with ``to_dict`` (``MarsFit``), or a dict."""
    if rec is None or isinstance(rec, dict):
        return None if rec is None else dict(rec)
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    return {
        k: _plain(v) if hasattr(v, "_asdict") else v for k, v in rec._asdict().items()
    }


def _fail(case, message: str):
    raise AssertionError(f"{case.name}: {message}\n{case.reproduction()}")


def _close(case, a: float, b: float, tol: float, what: str) -> None:
    """a and b within an absolute bound; equal infinities pass."""
    if not (a == b or abs(a - b) <= tol):
        _fail(case, f"{what}: fast {a!r}, reference {b!r}, bound {tol:.3g}")


def _rel(case, a, b, rtol: float, what: str) -> None:
    """Equal infinities, and finite values within a relative tolerance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    fin = np.isfinite(b)
    if not np.array_equal(a[~fin], b[~fin]) or np.any(
        np.abs(a[fin] - b[fin]) > rtol * np.abs(b[fin])
    ):
        _fail(case, f"{what}: fast {a.tolist()}, reference {b.tolist()}, rtol {rtol}")


def _reproduction(params: dict, **arrays) -> str:
    """Lines that rebuild a case, for a report of a disagreement."""
    lines = [f"params = {params!r}"]
    for name, a in arrays.items():
        lines.append(f"{name} = {None if a is None else f'np.array({a.tolist()!r})'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A case: the data and the settings of one fit


@dataclasses.dataclass
class Case:
    """One fit's data and settings: X (n, p), Y (n, K), w (n,) or None, and
    ``params``, fields of MarsParams (CORE-2)."""

    name: str
    X: np.ndarray
    Y: np.ndarray
    w: np.ndarray | None
    params: dict

    def fast_forward(self):
        kw = {k: v for k, v in self.params.items() if k in FORWARD_PARAMS}
        Y = self.Y[:, 0] if self.Y.shape[1] == 1 else self.Y
        return _forward.forward_pass(self.X, Y, self.w, record_candidates=True, **kw)

    def reference_fit(self) -> dict:
        return mars_ref.fit_mars(
            self.X, self.Y, self.w, self.params, record_candidates=True
        )

    def reproduction(self) -> str:
        return _reproduction(self.params, X=self.X, Y=self.Y, w=self.w)

    @functools.cached_property
    def kept(self) -> SimpleNamespace:
        """The reference's view of the cases of the fit (W-3, W-4, EDGE-6,
        EDGE-7): the rows with positive weight, N, tau_N, the power of 2 on
        Y, and the powers of 2 on the columns of X, with Xs the scaled X on
        which the reference's searches (``searches``) run."""
        w = np.ones(len(self.X)) if self.w is None else self.w
        rows = w > 0
        N0 = mars_ref.weight_sum(w[rows])
        tau_N = mars_ref.weight_tol(N0)
        Y = self.Y[rows]
        j = mars_ref.y_scale_power(Y)
        jx = mars_ref.x_scale_powers(self.X[rows])
        return SimpleNamespace(
            X=self.X[rows],
            Xs=np.ldexp(self.X[rows], jx),
            jx=jx,
            Y=Y,
            Ys=np.ldexp(Y, j),
            w=w[rows],
            N=mars_ref.snap(N0, tau_N),
            tau_N=tau_N,
            j=j,
        )

    @functools.cached_property
    def _spied(self) -> tuple[list, list]:
        """The reference runs again with a spy on its search of one parent and
        with the trace of its forward pass (the queue of FAST-1 to FAST-5).
        The spy keeps only the candidates that each search found, so that it
        does not depend on the search's arguments."""
        calls, trace = [], []
        real, real_pass = mars_ref._parent_candidates, mars_ref.forward_pass

        def spy(*args, **kw):
            found, searchable = real(*args, **kw)
            calls.append(SimpleNamespace(found=found))
            return found, searchable

        def traced(*args, **kw):
            return real_pass(*args, **{**kw, "trace": trace})

        with (
            mock.patch.object(mars_ref, "_parent_candidates", spy),
            mock.patch.object(mars_ref, "forward_pass", traced),
        ):
            self.reference_fit()
        if calls and not trace:
            raise RuntimeError("the reference's forward pass left no trace")
        return calls, trace

    @property
    def searches(self) -> list:
        """Every search of the reference's forward pass, in order."""
        return self._spied[0]

    @property
    def trace(self) -> list:
        """The reference's trace: one dict per step (``mars_ref.forward_pass``)."""
        return self._spied[1]

    def searches_of(self, s: int) -> list:
        """The searches of step s (1-based), by the parents that the trace of
        each step lists as searched."""
        if s > len(self.trace):
            return []
        start = sum(len(t["searched"]) for t in self.trace[: s - 1])
        return self.searches[start : start + len(self.trace[s - 1]["searched"])]

    @functools.cached_property
    def _bases(self) -> dict:
        return {}


def _term_columns(case: Case, ref: dict) -> np.ndarray:
    """The columns of every term of the reference's forward record, on the
    scaled X that its searches ran on (EDGE-7, TERM-3)."""
    if "terms" not in case._bases:
        cuts = np.ldexp(np.asarray(ref["cuts"], dtype=float), case.kept.jx)
        B = mars_ref.basis_matrix(case.kept.Xs, np.asarray(ref["dirs"]), cuts)
        case._bases["terms"] = B
    return case._bases["terms"]


def _new_columns(B, X, parent: int, variable: int, kind: int, knot: float) -> list:
    """The new columns of a candidate in LA-2's order (FWD-13): b x, then the
    hinge b (x - t)+, for a pair; the hinge for a single hinge; b x for a
    linear candidate. b x is the product itself, as LA-2 states it."""
    b, x = B[:, parent], X[:, variable]
    if kind == LINEAR:
        return [b * x]
    h = b * np.maximum(x - knot, 0.0)
    return [b * x, h] if kind == PAIR else [h]


def _chosen_columns(case: Case, ref: dict, s: int) -> list:
    """The new columns of the candidate that the reference chose at step s,
    and which of them LA-2 counted (``Candidate.counted``, all when the
    reference does not say)."""
    c = case.trace[s - 1]["chosen"]
    new = _new_columns(
        _term_columns(case, ref), case.kept.Xs, c.parent, c.variable, c.kind, c.knot
    )
    counted = tuple(getattr(c, "counted", ())) or (True,) * len(new)
    return list(zip(new, counted, strict=True))


def _forward_basis(case: Case, ref: dict, s: int) -> np.ndarray:
    """The forward basis F_s of FWD-13 after s steps, on the scaled X: the
    intercept, then the new columns that LA-2 counted at each step, in its
    order, as the reference's trace records them."""
    if s not in case._bases:
        cols = [np.ones(len(case.kept.Xs))]
        for t in range(1, s + 1):
            cols += [v for v, counted in _chosen_columns(case, ref, t) if counted]
        case._bases[s] = np.column_stack(cols)
    return case._bases[s]


def _step_columns(case: Case, ref: dict, s: int) -> tuple[np.ndarray, np.ndarray]:
    """F_{s-1} and every new column of the reference's choice at step s, and
    the weights: the columns of the RSS after step s, with the columns that
    LA-2 may have dropped, so that a near-dependence among them shows."""
    new = [v for v, _ in _chosen_columns(case, ref, s)]
    return np.column_stack([_forward_basis(case, ref, s - 1), *new]), case.kept.w


# ---------------------------------------------------------------------------
# The forward pass


def _key(parent, variable, kind, knot) -> tuple:
    """A candidate as CORE-3 tells candidates apart (``mars_ref.candidate_key``)."""
    knot = None if kind == LINEAR else float(knot)
    return (int(parent), int(variable), int(kind), knot)


def _rows(rec: dict, s: int) -> list:
    """The rows that step s added: their dirs and cuts, without the parents.
    Two programs can add the same rows from different parents (FWD-12), so
    the comparison of a step looks at the rows."""
    return [t[:2] for t in _terms(rec, s)]


def _terms(rec: dict, s: int) -> list:
    """The terms that step s added: their dirs and cuts rows and parents."""
    return [
        (
            tuple(rec["dirs"][r].tolist()),
            tuple(rec["cuts"][r].tolist()),
            rec["parent"][r],
        )
        for r in np.flatnonzero(rec["step"] == s)
    ]


def _parent_gives(rec: dict, p: int, r: int) -> bool:
    """Term p is a parent of term r (TERM-1): the row of r with one covariate
    removed, in codes and cuts."""
    for v in np.flatnonzero(rec["dirs"][r]):
        d, c = rec["dirs"][r].copy(), rec["cuts"][r].copy()
        d[v], c[v] = 0, rec["cuts"][p][v]
        if (
            p < r
            and np.array_equal(rec["dirs"][p], d)
            and np.array_equal(rec["cuts"][p], c)
        ):
            return True
    return False


def _render(case: Case, rec: dict, key) -> list:
    """The terms that candidate ``key`` adds to the parent's row (FWD-6, TERM-1,
    TERM-2), in the form of ``_terms``."""
    k, v, kind, knot = key
    if kind == PAIR:
        codes = [(1, knot), (-1, knot)]
    elif kind == SINGLE:
        codes = [(1, knot)]
    elif case.params.get("auto_linpreds", True):
        codes = [(2, 0.0)]
    else:
        codes = [(1, float(case.kept.X[:, v].min()))]
    out = []
    for code, cut in codes:
        d, c = rec["dirs"][k].copy(), rec["cuts"][k].copy()
        d[v], c[v] = code, cut
        out.append((tuple(d.tolist()), tuple(c.tolist()), k))
    return out


def _choice(case: Case, rec: dict, s: int) -> list[tuple]:
    """The candidate that step s chose, as its possible keys: the keys whose
    terms by FWD-6 are the step's terms. With ``auto_linpreds=False`` a single
    hinge at the smallest x has the linear option's term, so it has two
    readings. Terms that no candidate gives fail."""
    rows = np.flatnonzero(rec["step"] == s)
    k = int(rec["parent"][rows[0]])
    v = int(np.flatnonzero(rec["dirs"][rows[0]] != rec["dirs"][k])[0])
    knot = float(rec["cuts"][rows[0], v])
    keys = [_key(k, v, kind, knot) for kind in (PAIR, SINGLE, LINEAR)]
    keys = [key for key in keys if _render(case, rec, key) == _terms(rec, s)]
    if not keys:
        _fail(case, f"step {s}: the terms {_terms(rec, s)} do not follow FWD-6")
    return keys


def _identical(step, a, b) -> bool:
    """Two candidates with equal columns, bit for bit: the same kind and knot,
    and equal columns of the parents and of the covariates. Each program then
    finds equal values, so FWD-5's order decides between them."""
    return (
        a[2:] == b[2:]
        and np.array_equal(step.B[:, a[0]], step.B[:, b[0]])
        and np.array_equal(step.kept.X[:, a[1]], step.kept.X[:, b[1]])
    )


def _second(log: dict, i: int) -> tuple | None:
    """The second-best candidate of step i + 1 of a candidate log, or None."""
    kind = int(log["second_kind"][i])
    if kind == 0:
        return None
    parent, variable = log["second_parent"][i], log["second_variable"][i]
    return _key(parent, variable, kind, log["second_knot"][i])


class _Step:
    """The reference's view of forward step s (1-based), on its scaled Y
    (EDGE-6): its legal candidates, the value of any candidate, and the bands
    of LA-5 and STOP-7 where a candidate's fate is a near-tie."""

    def __init__(self, case: Case, ref: dict, s: int):
        kept = case.kept
        self.case, self.kept = case, kept
        M = int(np.count_nonzero(ref["step"] < s))
        self.M, self.s, self.ref = M, s, ref
        self.n = len(kept.w)
        self.dirs = ref["dirs"][:M]
        calls = case.searches_of(s)
        # The reference searches the scaled X (EDGE-7): its columns, knots and
        # cuts are on that scale here; keys stay on the original scale.
        self.cuts = np.ldexp(ref["cuts"][:M], kept.jx)
        self.B = mars_ref.basis_matrix(kept.Xs, self.dirs, self.cuts)
        # Every candidate of the step is evaluated on F_{s-1} (FWD-13).
        self.F = _forward_basis(case, ref, s - 1)
        self.P_F = mars_ref.Projector(self.F, kept.w)
        self._cands: dict = {}
        self.sigma2 = mars_ref.covariate_variances(kept.Xs, kept.w, kept.N)
        self.tau = mars_ref.collinearity_tol(s - 1)
        self.path = np.ldexp(np.asarray(ref["rss"], dtype=float), 2 * kept.j)
        self.rss_s = self.path[s - 1]
        self.tss = self.path[0]
        self.limit = mars_ref.max_legal(list(self.path[:s]))
        # STOP-7's R: the RSS before step s for a candidate's RSS alone, one
        # step earlier for a reduction, two steps earlier for Delta_s.
        self.R_cand, self.R_red, self.R_delta = (
            self.path[max(s - i, 0)] for i in (1, 2, 3)
        )
        found = [c for call in calls for c in call.found]
        self.values = {self.key_of(c): c.rss for c in found}
        self.legal = [
            c
            for c in found
            if mars_ref.is_legal(c.kind, self.rss_s - c.rss, self.limit)
        ]
        self.legal_keys = {self.key_of(c) for c in self.legal}
        # FWD-12: occurrences of one kind that add the same rows are one
        # candidate, so a key of the fast code from another parent finds the
        # reference's value and legality through its rows
        self.by_rows = {self.rows_of(self.key_of(c)): self.key_of(c) for c in found}

    def key_of(self, c) -> tuple:
        """The key (``_key``) of a reference Candidate, its knot on the
        original scale of X (EDGE-7)."""
        knot = (
            c.knot
            if c.kind == LINEAR
            else math.ldexp(c.knot, -int(self.kept.jx[c.variable]))
        )
        return _key(c.parent, c.variable, c.kind, knot)

    def rows_of(self, key) -> tuple:
        """The kind and the rows (dirs, cuts) that a candidate adds (FWD-6)."""
        return (key[2], tuple(t[:2] for t in _render(self.case, self.ref, key)))

    def merged(self, key) -> tuple:
        """The reference's key of the candidate that ``key`` merges with
        (FWD-12), or ``key``."""
        return self.by_rows.get(self.rows_of(key), key)

    def scaled(self, key) -> tuple[np.ndarray, np.ndarray, float]:
        """The parent column, the covariate and the knot of a key, on the
        scaled X that the reference searched (EDGE-7)."""
        k, v, kind, knot = key
        x = self.kept.Xs[:, v]
        t = math.nan if kind == LINEAR else math.ldexp(knot, int(self.kept.jx[v]))
        return self.B[:, k], x, t

    def columns(self, key) -> SimpleNamespace:
        """F_{s-1} with the candidate's new columns (``_new_columns``):
        ``all`` holds every new column, ``kept`` only those that LA-4's test
        counts in LA-2's order, and ``first`` is the index of the first new
        column; ``cond`` is cond of LA-5 of the kept columns."""
        if key not in self._cands:
            k, v, kind, _ = key
            _, _, knot = self.scaled(key)
            new = _new_columns(self.B, self.kept.Xs, k, v, kind, knot)
            A = np.column_stack([self.F, *new])
            counted = compare.la4_ratios(A, self.kept.w) >= compare.LA4_TOL
            counted[: self.F.shape[1]] = True
            self._cands[key] = SimpleNamespace(
                all=A, kept=A[:, counted], first=self.F.shape[1], cond=None
            )
        c = self._cands[key]
        if c.cond is None:
            c.cond = compare.basis_cond(c.kept, self.kept.w)
        return c

    @functools.cached_property
    def cond_f(self) -> float:
        """cond of LA-5 of F_{s-1}, the basis of RSS_{s-1} and of the RSS
        values before it, whose bases are subsets of it."""
        return compare.basis_cond(self.F, self.kept.w)

    def cond(self, *keys) -> float:
        """The largest cond over F_{s-1} and the bases of the candidates, the
        cond of STOP-7 for a comparison of their values."""
        return max([self.cond_f, *(self.columns(k).cond for k in keys)])

    def dependent(self, key) -> bool:
        """A near-dependence of STOP-7 in the candidate's new columns: an LA-4
        ratio within a factor 2 of 1e-7, or cond above 1e6."""
        c = self.columns(key)
        return _dependence(c.all, self.kept.w, c.first)

    def delta(self, R: float, *keys) -> float:
        """STOP-7's delta for the RSS R (scaled Y), with cond of F_{s-1} and
        of the bases of the candidates ``keys``."""
        return delta(R, self.tss, self.cond(*keys), self.n)

    def best(self):
        """The reference's choice at this step, or None (FWD-4, FWD-5)."""
        return mars_ref._first_best(self.legal, self.rss_s)

    def value(self, key) -> float:
        """The RSS of LA-2 of a candidate: the reference's own value when it
        found the candidate, else a least-squares fit of the columns that
        LA-2 counts (``columns``)."""
        key = self.merged(key)
        if key in self.values:
            return self.values[key]
        return mars_ref.rss(self.columns(key).kept, self.kept.Ys, self.kept.w)

    def bands(self, key) -> list[str]:
        """The bands that the candidate lies in: A_w of its search within 1e-6
        of the pair rule's threshold (LA-7); its knot's rho within 1e-6 tau of
        tau (LA-3), for each kind of search that the pair rule allows; its
        reduction within 22 delta of MaxLegal or within 2 delta of 0 (FWD-4),
        or its RSS within 2 delta of the exact-fit edge (FWD-5; every
        candidate below about 4e-8 RSS_s, so the band never decides); a
        near-dependence of its new columns (STOP-7). Each delta takes cond of
        the candidate's basis, which holds those of the path's values."""
        k, v, kind, _ = key
        out, w = [], self.kept.w
        b, x, knot = self.scaled(key)
        V = [*np.flatnonzero(self.dirs[k]).tolist(), v]
        thr = 0.01 * math.prod(self.sigma2[u] for u in V)
        searchable = all(self.sigma2[u] > 0 for u in V)
        lin = b * x
        A = self.P_F.rss(lin)
        near_pair = searchable and abs(A - thr) <= BAND * thr
        if near_pair:
            out.append("LA-7")
        h = None if kind == LINEAR else b * np.maximum(x - knot, 0.0)
        kinds = [True, False] if near_pair else [searchable and thr <= A]
        if h is not None and not np.all(h == h[0]):
            for pair in kinds:
                G = np.column_stack([self.F, lin]) if pair else self.F
                rho = mars_ref.collinearity_ratio(h, G, w)
                if abs(rho - self.tau) <= BAND * self.tau:
                    out.append("LA-3")
        red = self.rss_s - self.value(key)
        if h is not None and abs(red - self.limit) <= 22 * self.delta(
            self.R_delta, key
        ):
            out.append("FWD-4 MaxLegal")
        if abs(red) <= 2 * self.delta(self.R_red, key):
            out.append("FWD-4 zero")
        rss_new = self.rss_s - red
        if abs(rss_new - EXACT_FIT * self.rss_s) <= 2 * self.delta(self.R_cand, key):
            out.append("FWD-5 exact fit")
        if self.dependent(key):
            out.append(NEAR_DEPENDENCE)
        return out

    def queue_bands(self) -> list[str]:
        """STOP-7's band of FAST-3, at step s or at any step before it, since
        the queues of the two programs can part at a step whose choice they
        still share: two stored values lambda within 2 delta of each other
        whose order changes the rows of the table that the step visits. An
        entry searched in step t has kappa = 2t and a reduction of that step,
        so delta = 2 eps(RSS_{t-2}), here with cond of F_{s-1} (the bases of
        the stored values are not kept). The table of step t comes from the entries
        after step t - 1 and one entry for each term that step added (FAST-1,
        FAST-2); it must be the table of the reference's trace."""
        fast_k = mars_ref.as_params(self.case.params).fast_k
        if fast_k == 0:
            return []
        trace, nu = self.case.trace, max(3, fast_k)
        for t in range(2, self.s + 1):
            added = int(np.count_nonzero(self.ref["step"] == t - 1))
            entries = [*trace[t - 2]["entries"], *[(math.inf, 2 * (t - 1))] * added]
            if nu >= len(entries):
                continue
            table = self._table(entries, t)
            if len(trace) >= t and table != list(trace[t - 1]["table"]):
                _fail(self.case, f"step {t}: queue table {table}, trace {trace[t - 1]}")
            values = [
                (e, lam, 4 * self.delta(self.path[max(kappa // 2 - 2, 0)]))
                for e, (lam, kappa) in enumerate(entries)
                if 0.0 <= lam < math.inf
            ]
            for i, (a, la, da) in enumerate(values):
                for b, lb, db in values[i + 1 :]:
                    band = max(da, db)
                    if abs(la - lb) > band:
                        continue
                    top, seen = max(la, lb), []
                    for va, vb in ((top + band, top), (top, top + band)):
                        order = list(entries)
                        order[a], order[b] = (va, entries[a][1]), (vb, entries[b][1])
                        seen.append(set(self._table(order, t)[:nu]))
                    if seen[0] != seen[1]:
                        return ["FAST-3"]
        return []

    def _table(self, entries, t: int) -> list[int]:
        """The table of FAST-2 for step t as 1-based entry numbers, with
        kappa_prev the term number of the step just done, 2 (t - 1)."""
        beta = mars_ref.as_params(self.case.params).fast_beta
        by_value = sorted(range(len(entries)), key=lambda e: (-entries[e][0], e))
        rank = {e: r for r, e in enumerate(by_value)}
        kprev = 2 * (t - 1)
        aged = {e: rank[e] + beta * (kprev - entries[e][1]) for e in rank}
        return [e + 1 for e in sorted(rank, key=lambda e: (aged[e], rank[e]))]

    def stop_bands(self, key) -> list[str]:
        """STOP-7's bands of STOP-3 and STOP-4 for the step that adds the
        candidate ``key``, or for a step without a legal candidate (None),
        which counts as one term and no gain."""
        prm, kept, tss = mars_ref.as_params(self.case.params), self.kept, self.path[0]
        keys = () if key is None else (key,)
        if key is None:
            rss_new, m_new, R = self.rss_s, self.M + 1, self.R_red
        else:
            rss_new, R = self.value(key), self.R_cand
            m_new = self.M + (2 if key[2] == PAIR else 1)
        d = prm.resolved_penalty()
        grsq = mars_ref.grsq(rss_new, m_new, tss, d, kept.N, kept.tau_N)
        out = []
        # GRSq' within 2 delta (1 - GRSq') / RSS' of -10 or of -1000 (STOP-3);
        # at -inf the band would be infinite, so it does not count there.
        dR = self.delta(R, *keys)
        width = 2 * dR * (1 - grsq) / rss_new if math.isfinite(grsq) else 0
        near = math.isfinite(grsq) and min(abs(grsq + 10), abs(grsq + 1000)) <= width
        if prm.thresh > 0 and near:
            out.append("STOP-3")
        gain = (self.rss_s - rss_new) / tss  # 0 exactly without a candidate
        band = 2 * self.delta(self.R_red, *keys) / tss
        if key is not None and abs(gain - prm.thresh) <= band:
            out.append("STOP-4")
        return out


def _grsq_new(case: Case, step: _Step, key) -> float:
    """GRSq' of the reference's step for the candidate ``key``, or for a step
    without a legal candidate (None), as STOP-3 says."""
    prm, kept = mars_ref.as_params(case.params), case.kept
    if key is None:
        rss_new, m_new = step.rss_s, step.M + 1
    else:
        rss_new, m_new = step.value(key), step.M + (2 if key[2] == PAIR else 1)
    d = prm.resolved_penalty()
    return mars_ref.grsq(rss_new, m_new, step.tss, d, kept.N, kept.tau_N)


def _stop5_bands(case: Case, ref: dict, s: int) -> list[str]:
    """STOP-7's bands of STOP-5 after step s: the new RSS within 2 delta of
    the floor 1e-10 TSS / (N - 1), or the new RSq within 2 delta / TSS of
    1 - thresh, with R the RSS before the step and cond that of F_s."""
    kept, thresh = case.kept, mars_ref.as_params(case.params).thresh
    path = np.ldexp(np.asarray(ref["rss"], dtype=float), 2 * kept.j)
    tss, rss, R = path[0], path[s], path[s - 1]
    out = []
    cond = compare.basis_cond(_forward_basis(case, ref, s), kept.w)
    d = delta(R, tss, cond, len(kept.w))
    if abs(rss - 1e-10 * tss / (kept.N - 1)) <= 2 * d:
        out.append("STOP-5 floor")
    if abs(thresh - rss / tss) <= 2 * d / tss:
        out.append("STOP-5 RSq")
    return out


def _join(reasons) -> str:
    return "+".join(sorted(set(reasons)))


def _diverged(case: Case, fast: dict, ref: dict, s: int) -> Outcome:
    """Step s chose differently: a near-tie, or a failure (module docstring).
    The fast code's choice must be legal in the reference or in a band, and
    either within TIE of the reference's choice, or the reference's choice
    in a band (so that the fast code may have left it out)."""
    step = _Step(case, ref, s)
    R = step.rss_s
    cf, cr = _choice(case, fast, s), _choice(case, ref, s)
    if any(_identical(step, a, b) for a in cf for b in cr):
        _fail(case, f"step {s}: {cf[0]} and {cr[0]} have equal columns (FWD-5)")
    vf, vr = step.value(cf[0]), step.value(cr[0])
    rss_f = math.ldexp(float(fast["rss"][s]), 2 * case.kept.j)
    _within(
        case,
        rss_f,
        vf,
        vf,
        R,
        step.tss,
        step.n,
        lambda: (step.columns(cf[0]).all, case.kept.w),
        f"step {s}: the fast RSS of {cf[0]}",
    )
    f_legal = any(step.merged(k) in step.legal_keys for k in cf)
    # the plan's tie, 1e-7 of the RSS before the step, widened by STOP-7's
    # delta for the two values, which matters where c sqrt(TSS / R) is large
    tie = TIE * R + step.delta(step.R_cand, cf[0], cr[0])
    if f_legal and vf < vr - tie:
        _fail(case, f"step {s}: the reference chose {cr[0]} over the better {cf[0]}")
    f_bands = [b for k in cf for b in step.bands(k)]
    r_bands = [b for k in cr for b in step.bands(k)]
    gap = vf - vr < tie
    if (f_legal or f_bands) and (gap or r_bands):
        return Outcome(s - 1, "gap" if f_legal and gap else _join(f_bands + r_bands), s)
    if queue := step.queue_bands():  # the two passes searched other parents
        return Outcome(s - 1, _join(queue + f_bands + r_bands), s)
    _fail(
        case,
        f"step {s}: fast {_terms(fast, s)}, reference {_terms(ref, s)}; values "
        f"{vf!r} and {vr!r} with RSS {R!r} before the step; the fast choice is "
        f"legal in the reference: {f_legal}; bands {f_bands} and {r_bands}",
    )


def _check_second(case: Case, fast: dict, ref: dict, s: int) -> str | None:
    """The second-best candidate of step s (CORE-3, FWD-8), after both programs
    chose alike; returns the near-tie that explains different seconds. The
    fast second must differ from the chosen candidate, and neither second may
    be better than the other by TIE when both are legal in the reference."""
    lf, lr, i = fast["candidates"], ref["candidates"], s - 1
    kf, kr = _second(lf, i), _second(lr, i)
    if kf == kr:
        fields = ("second_parent", "second_variable", "second_kind", "second_knot")
        raw = [np.array([log[f][i] for f in fields]) for log in (lf, lr)]
        if not np.array_equal(*raw, equal_nan=True):
            _fail(case, f"step {s}: the second's fields: fast {raw[0]}, ref {raw[1]}")
        tss, V = ref["rss"][0], lr["second_rss"][i]
        if not math.isfinite(V):
            _close(case, lf["second_rss"][i], V, 0.0, f"second, {s}")
            return None
        n = len(case.kept.w)
        _within(
            case,
            lf["second_rss"][i],
            V,
            V,
            ref["rss"][i],
            tss,
            n,
            lambda: (_Step(case, ref, s).columns(kr).all, case.kept.w),
            f"second, {s}",
        )
        return None
    step = _Step(case, ref, s)

    def fast_value(k) -> None:
        """The fast second's RSS against the reference's value of it."""
        vf = step.value(k)
        rss_f = math.ldexp(float(lf["second_rss"][i]), 2 * case.kept.j)
        _within(
            case,
            rss_f,
            vf,
            vf,
            step.rss_s,
            step.tss,
            step.n,
            lambda: (step.columns(k).all, case.kept.w),
            f"step {s}: the fast second {k}",
        )

    if kf is not None and kr is not None and step.rows_of(kf) == step.rows_of(kr):
        # one candidate from two parents (FWD-12): rows, not parent
        fast_value(kf)
        return None
    if kf is not None and step.rows_of(kf) in [
        step.rows_of(k) for k in _choice(case, fast, s)
    ]:
        # the fast second adds the chosen rows from another parent (FWD-12)
        _fail(case, f"step {s}: the fast second {kf} is the chosen candidate (FWD-8)")
    if kf is not None and kr is not None and _identical(step, kf, kr):
        _fail(case, f"step {s}: the seconds {kf} and {kr} have equal columns (FWD-5)")
    if kf is not None:
        fast_value(kf)
    f_bands = [] if kf is None else step.bands(kf)
    r_bands = [] if kr is None else step.bands(kr)
    if kf is None or kr is None:
        ok = bool(f_bands or r_bands)
    else:
        tie = TIE * step.rss_s + step.delta(step.R_cand, kf, kr)
        gap = abs(step.value(kf) - step.value(kr)) < tie
        f_ok = step.merged(kf) in step.legal_keys or f_bands
        r_ok = step.merged(kr) in step.legal_keys or r_bands
        ok = f_ok and r_ok and (gap or f_bands or r_bands)
    if not ok and (queue := step.queue_bands()):
        return _join(queue)
    if not ok:
        _fail(case, f"step {s}: second fast {kf}, reference {kr}; {f_bands}, {r_bands}")
    return _join(f_bands + r_bands) or "gap"


def _stopped_apart(case: Case, fast: dict, ref: dict, t: int) -> Outcome:
    """The two passes agree on t - 1 steps and then stop differently: a
    near-tie of STOP-7, or a failure. STOP-1 is exact. When one pass stopped
    by STOP-5 after step t - 1, only the bands of STOP-5 count; otherwise
    both searched step t, and the bands of its candidates count (the
    reference's best and each pass's choice): those of LA-3, LA-7 and FWD-4,
    which decide whether a legal candidate exists, and those of STOP-3 and
    STOP-4."""
    ends = [
        (len(rec["rss"]) - 1 == t - 1, int(rec["termination"])) for rec in (fast, ref)
    ]
    high = [done and code == RSQ_HIGH for done, code in ends]
    limit = [done and code in LIMIT_CODES for done, code in ends]
    if any(high) and not all(high):
        reasons = _stop5_bands(case, ref, t - 1)
    elif any(limit):
        reasons = []
    else:
        step = _Step(case, ref, t)
        best = step.best()
        keys = [] if best is None else [step.key_of(best)]
        for rec in (fast, ref):
            if len(rec["rss"]) - 1 >= t:
                keys += _choice(case, rec, t)
        reasons = [b for key in keys for b in step.bands(key) + step.stop_bands(key)]
        if best is None:
            reasons += step.stop_bands(None)
        reasons += step.queue_bands()
    if reasons:
        return Outcome(t - 1, _join(reasons), t)
    _fail(
        case,
        f"after {t - 1} equal steps: fast {len(fast['rss']) - 1} steps, code "
        f"{int(fast['termination'])}; reference {len(ref['rss']) - 1} steps, "
        f"code {ref['termination']}",
    )


#: CORE-3: the dtypes of the forward record and of the candidate log.
DTYPES = {
    **dict.fromkeys(
        ("cuts", "rss", "best_rss", "second_rss", "second_knot"), np.float64
    ),
    **dict.fromkeys(("kept", "dropped", "parent", "step"), np.int64),
    **dict.fromkeys(("second_parent", "second_variable"), np.int64),
    **dict.fromkeys(("dirs", "second_kind"), np.int8),
}


def compare_forward(case: Case, fast, ref) -> Outcome:
    """Compare two forward records (CORE-3) of one case, as the module
    docstring says; ``fast`` and ``ref`` are records or their dicts. The
    outcome also says whether the fit has a near-tie by the plan's count
    ("Ties"): a step, among those compared, where either log's best and
    second differ by less than TIE of the RSS before the step."""
    fast, ref = _plain(fast), _plain(ref)
    out = _compare_forward(case, fast, ref)
    S = min(len(fast["rss"]), len(ref["rss"]), out.step or math.inf) - 1
    plan = False
    for rec in (fast, ref):
        if rec["candidates"] is not None and S > 0:
            log, rss = _plain(rec["candidates"]), np.asarray(rec["rss"])
            gap = log["second_rss"][:S] - log["best_rss"][:S]
            plan |= bool(np.any(gap < TIE * rss[:S]))
    return dataclasses.replace(out, plan_tie=plan)


def _compare_forward(case: Case, fast: dict, ref: dict) -> Outcome:
    logs = fast["candidates"], ref["candidates"] = (
        _plain(fast["candidates"]),
        _plain(ref["candidates"]),
    )
    for rec, who in ((fast, "fast"), (ref, "reference")):
        for key, value in {**rec, **(rec["candidates"] or {})}.items():
            if key in DTYPES and np.asarray(value).dtype != DTYPES[key]:
                _fail(case, f"the {who} {key} has dtype {np.asarray(value).dtype}")
    tss, n = ref["rss"][0], len(case.kept.w)
    _close(case, fast["rss"][0], tss, eps(tss, tss, tss, 1.0, n), "TSS, rss[0]")
    S = min(len(fast["rss"]), len(ref["rss"])) - 1
    for s in range(1, S + 1):
        try:
            if _rows(fast, s) != _rows(ref, s):
                return _diverged(case, fast, ref, s)
            V, R = ref["rss"][s], ref["rss"][s - 1]
            what = f"RSS after step {s}"
            cols = functools.partial(_step_columns, case, ref, s)
            _within(case, fast["rss"][s], V, V, R, tss, n, cols, what)
            if None in logs:
                continue
            for rec, who in ((fast, "fast"), (ref, "reference")):
                if rec["candidates"]["best_rss"][s - 1] != rec["rss"][s]:
                    _fail(case, f"step {s}: the {who} log's best_rss is not rss[{s}]")
            if (reason := _check_second(case, fast, ref, s)) is not None:
                TALLY[f"second best differs at a near-tie ({reason})"] += 1
        except _Stop:
            return Outcome(s - 1, NEAR_DEPENDENCE, s)
    if len(fast["rss"]) != len(ref["rss"]) or fast["termination"] != ref["termination"]:
        return _stopped_apart(case, fast, ref, S + 1)
    for key in ("dirs", "cuts", "step"):
        if not np.array_equal(fast[key], ref[key]):
            _fail(case, f"{key}: fast {fast[key].tolist()}, ref {ref[key].tolist()}")
    if not np.array_equal(fast["kept"], ref["kept"]):
        # FWD-11's test of the terms at a near-dependence (STOP-7)
        if _dependence(_term_columns(case, ref), case.kept.w):
            return Outcome(S, NEAR_DEPENDENCE, S + 1)
        _fail(case, f"kept: fast {fast['kept'].tolist()}, ref {ref['kept'].tolist()}")
    if not np.array_equal(fast["dropped"], ref["dropped"]):
        _fail(case, f"dropped: fast {fast['dropped']}, ref {ref['dropped']}")
    for r, (pf, pr) in enumerate(zip(fast["parent"], ref["parent"], strict=True)):
        # FWD-12: a different parent is fine when the rows are the same and
        # each parent gives the row (its row, less one covariate).
        if pf != pr and not all(_parent_gives(ref, int(p), r) for p in (pf, pr)):
            _fail(case, f"parent of term {r}: fast {pf}, ref {pr}")
    return Outcome(S)


def check_forward(case: Case) -> Outcome:
    """Run both forward passes on the case and compare them."""
    return compare_forward(case, case.fast_forward(), case.reference_fit()["forward"])


# ---------------------------------------------------------------------------
# The pruning pass


def _working_sets(removed, Mf: int, several: bool) -> list:
    """The terms at positions 1 to pos at each stage pos = Mf, ..., 2 (PRUNE-3):
    the working order for one response, the current set for several."""
    order, out = list(range(Mf)), []
    for pos, term in zip(range(Mf, 1, -1), removed, strict=False):
        out.append(order[:pos])
        if several:
            order.remove(term)
        else:
            i = order.index(term)
            order = order[:i] + order[i + 1 : pos] + [term] + order[pos:]
    return out


def compare_pruning(case, fast: dict, ref: dict, B, Y, w) -> Outcome:
    """Compare two pruning passes on the basis B (PRUNE-2 to PRUNE-8, CORE-3).

    ``fast`` and ``ref`` hold the fields of PruningRecord, ``selected`` (in
    pruning indices) and ``coef``, ``rss``, ``gcv``, ``rsq`` and ``grsq`` of
    the final model; w holds the weights (ones for none). The removals are
    compared stage by stage. At the first stage where they differ, the
    reference's RSS of the two drops decides whether it is a near-tie (PRUNE-10:
    within PRUNE_TIE of the lower plus 2 eps(V, V, cond), cond the larger of
    the two subsets', or a subset at a near-dependence); then only the sizes
    m >= pos, which the earlier stages fix, are compared. An offer can tie as
    well: two subsets of one size, by the same bound. The selected size and
    the final fit are compared when no removal tied. An RSS below STOP-5's
    floor 1e-10 TSS / (N - 1) is an exact fit, whose value is rounding: two
    such values count as equal. A value that misses its bound at a
    near-dependence (STOP-7) ends the comparison there (spec v3: the RSS of
    a subset is that of the columns of B, without v2's shifted atoms).
    """
    Mf, several = B.shape[1], Y.shape[1] >= 2
    n = int(np.count_nonzero(w > 0))

    def columns(terms):
        return B[:, sorted(terms)]

    rss_of = functools.cache(lambda terms: mars_ref.rss(columns(terms), Y, w))
    cond_of = functools.cache(lambda terms: compare.basis_cond(columns(terms), w))
    dependent = functools.cache(lambda terms: _dependence(columns(terms), w))
    N = float(np.sum(w))  # a degenerate fit (N <= 1, EDGE-1) has no exact fits
    tss = ref["rss_per_size"][0]
    floor = 1e-10 * tss / (N - 1) if N > 1 else 0.0
    for rec, who in ((fast, "fast"), (ref, "reference")):
        for key, (dtype, shape) in {
            "removed": (np.int64, (Mf - 1,)),
            "rss_per_size": (np.float64, (Mf,)),
            "gcv_per_size": (np.float64, (Mf,)),
            "subsets": (np.bool_, (Mf, Mf)),
            "coef": (np.float64, (rec["selected_size"], Y.shape[1])),
        }.items():
            a = np.asarray(rec[key])
            if a.dtype != dtype or a.shape != shape:
                _fail(case, f"the {who} {key} is {a.dtype} {a.shape} (CORE-3, PRUNE-8)")

    def same(a, b, cond, near=False) -> bool:
        """Two RSS values of one subset: within LA-5's eps(V, V, cond) of the
        exact value (b stands for it; also an exact fit, whose value is
        rounding, as the third term of eps allows), or, where the subsets
        differ, PRUNE-10's near-tie bound."""
        if a <= floor and b <= floor:
            return True
        V = min(a, b) if near else b
        bound = eps(V, V, tss, cond, n) * (2 if near else 1)
        return abs(a - b) <= bound + (PRUNE_TIE * V if near else 0)

    def rtol(V, cond, near=False) -> float:
        """The relative bound on a GCV value whose RSS is V (GCV-2 passes
        the bound of the RSS on)."""
        if V <= 0:
            return math.inf
        bound = eps(V, V, tss, cond, n) * (2 if near else 1)
        return bound / V + (PRUNE_TIE if near else 0)

    def tied(Ta, Tb) -> bool:
        """PRUNE-10's near-tie of two subsets of one size, or a subset at a
        near-dependence (STOP-7)."""
        Ta, Tb = frozenset(Ta), frozenset(Tb)
        if dependent(Ta) or dependent(Tb):
            return True
        cond = max(cond_of(Ta), cond_of(Tb))
        return same(rss_of(Ta), rss_of(Tb), cond, near=True)

    first, near_tie = 1, None  # the smallest size that the compared stages fix
    stages = _working_sets(ref["removed"], Mf, several)
    for i, (a, b) in enumerate(zip(fast["removed"], ref["removed"], strict=True)):
        if a != b:
            U = set(stages[i])
            if not tied(U - {int(a)}, U - {int(b)}):
                _fail(case, f"stage pos={Mf - i}: fast removes {a}, reference {b}")
            first, near_tie = Mf - i, "removal"
            break
    for m in range(first, Mf + 1):
        Tf, Tr = (
            frozenset(np.flatnonzero(T[m - 1]).tolist())
            for T in (fast["subsets"], ref["subsets"])
        )
        near = Tf != Tr
        if near:
            if not tied(Tf, Tr):
                _fail(case, f"size {m}: fast {sorted(Tf)}, reference {sorted(Tr)}")
            near_tie = near_tie or f"subset of size {m}"
        cond = max(cond_of(Tf), cond_of(Tr))
        rf, rr = fast["rss_per_size"][m - 1], ref["rss_per_size"][m - 1]
        gf, gr = fast["gcv_per_size"][m - 1], ref["gcv_per_size"][m - 1]
        if not same(rf, rr, cond, near):
            if dependent(Tf) or dependent(Tr):  # STOP-7
                return Outcome(Mf - m, NEAR_DEPENDENCE, Mf - m + 1)
            _fail(case, f"rss_per_size[{m - 1}]: fast {rf!r}, reference {rr!r}")
        if not (rf <= floor and rr <= floor) or np.isinf(gf) or np.isinf(gr):
            bound = rtol(min(rf, rr) if near else rr, cond, near)
            _rel(case, gf, gr, bound, f"gcv_per_size[{m - 1}]")
    if first > 1:
        return Outcome(Mf - first, near_tie, Mf - first + 1)
    mf, mr = fast["selected_size"], ref["selected_size"]
    if mf != mr:
        gf, gr = ref["gcv_per_size"][mf - 1], ref["gcv_per_size"][mr - 1]
        exact = all(ref["rss_per_size"][m - 1] <= floor for m in (mf, mr))
        by_gcv = case.params.get("pmethod", "backward") == "backward"  # PRUNE-7
        if not (by_gcv and (exact or abs(gf - gr) <= PRUNE_TIE * max(gf, gr))):
            _fail(case, f"selected size: fast {mf}, reference {mr}")
        return Outcome(Mf - 1, "selected size", Mf)
    if not np.array_equal(fast["selected"], ref["selected"]):
        if near_tie is None:
            _fail(case, f"selected: fast {fast['selected']}, ref {ref['selected']}")
        return Outcome(Mf - 1, near_tie, Mf)
    sel = frozenset(np.asarray(ref["selected"]).tolist())
    BS, cond = B[:, ref["selected"]], cond_of(sel)
    try:
        _compare_final(
            case,
            fast,
            ref,
            BS,
            Y,
            w,
            lambda a, b: same(a, b, cond),
            lambda V: rtol(V, cond),
            floor,
        )
    except AssertionError:
        # PRUNE-8: LA-4's test of the selected columns at a near-dependence
        # can zero a column in one program only (STOP-7)
        if not dependent(sel):
            raise
        return Outcome(Mf - 1, NEAR_DEPENDENCE, Mf)
    return Outcome(Mf - 1, near_tie, None if near_tie is None else Mf)


def _compare_final(case, fast: dict, ref: dict, BS, Y, w, same, rtol, floor) -> None:
    """The final fit of the same terms (PRUNE-8), by the plan's tolerance
    table: per response, the coefficients normwise within 1e-6 where
    kappa(B) <= 1e5; the fitted values with their weighted mean removed
    within 1e-8 sd(y), scaled by kappa / 1e6 above 1e6, and the mean of their
    difference within rounding, 8 ulp of max |y| per term plus kappa u ||y - mean||
    (a fitted value near 1e13 has an ulp of 2e-3); the RSS of the fast
    coefficients, the reference's RSS plus the weighted sum of squares of
    the centered difference of the fitted values (least squares), within
    LA-5's eps of the reference's RSS plus the effect of coefficients 8 u |c|
    apart (PRUNE-8); the final RSS and GCV within the same
    bound (``same`` and ``rtol``, also for exact fits); RSq and GRSq within
    1e-8."""
    m, sw = BS.shape[1], np.sqrt(w)[:, None]
    kappa = np.linalg.cond(BS * sw) if m > 1 else 1.0
    cf = np.asarray(fast["coef"], dtype=float).reshape(m, -1)
    cr = np.asarray(ref["coef"], dtype=float).reshape(m, -1)
    if kappa <= 1e5 and np.any(
        np.linalg.norm(cf - cr, axis=0) > 1e-6 * np.linalg.norm(cr, axis=0)
    ):
        _fail(case, f"coefficients: fast {cf.tolist()}, reference {cr.tolist()}")
    mean = np.average(Y, axis=0, weights=w)
    sd = np.sqrt(np.average((Y - mean) ** 2, axis=0, weights=w))
    d = BS @ (cf - cr)
    d_mean = np.average(d, axis=0, weights=w)
    dc = d - d_mean
    if np.any(np.max(np.abs(dc), axis=0) > 1e-8 * sd * max(1, kappa / 1e6)):
        _fail(case, f"fitted values: kappa {kappa:.3g}, sd(y) {sd.tolist()}")
    u = np.finfo(float).eps / 2
    ulps = 8 * m * np.spacing(np.max(np.abs(Y), axis=0))
    rounding = ulps + kappa * u * np.linalg.norm((Y - mean) * sw, axis=0)
    if np.any(np.abs(d_mean) > rounding):
        _fail(case, f"mean of the fitted values: {d_mean.tolist()}, {rounding}")
    excess = float(np.sum(w[:, None] * dc**2))
    # The RSS of the returned coefficients can differ from the LA-1 value by
    # more than LA-5 allows, through the rounding of the coefficients
    # themselves, as PRUNE-8 reports (1.9e-9 to 0.97 of the RSS at means of
    # 1e10 to 1e15); #120 item 5: exactly rounded coefficients exceed eps at
    # means of 1e9 to 1e15. So the excess may also reach that of coefficients
    # 8 u |c_k| apart (u = 2^-53), (8 u sum_k |c_k| ||B_k - mean||_w)^2 per
    # response. The form is ours, not PRUNE-8's, and 8 is a margin of 8 over
    # the largest measured ratio (1.0, seed 273). The check cannot see an
    # error in a coefficient whose term's share |c_k| ||B_k - mean||_w of the
    # sum is below about 8 u (doubling term 2 at seed 273 passes); a bound per
    # coefficient would need the coefficients to be well conditioned, and
    # above kappa 1e5 the plan compares the fitted values only.
    Bc = BS - np.average(BS, axis=0, weights=w)
    spread = np.sqrt(np.sum(w[:, None] * Bc**2, axis=0))
    coef_rounding = float(
        np.sum((8 * u * (spread @ np.maximum(abs(cf), abs(cr)))) ** 2)
    )
    if not same(ref["rss"] + max(excess - coef_rounding, 0.0), ref["rss"]):
        _fail(case, f"RSS of the fast coefficients: {ref['rss']!r} + {excess!r}")
    if not same(fast["rss"], ref["rss"]):
        _fail(case, f"final rss: fast {fast['rss']!r}, reference {ref['rss']!r}")
    if not (fast["rss"] <= floor and ref["rss"] <= floor) or np.isinf(ref["gcv"]):
        _rel(case, fast["gcv"], ref["gcv"], rtol(ref["rss"]), "final gcv")
    for key in ("rsq", "grsq"):
        _close(case, fast[key], ref[key], 1e-8, f"final {key}")
        if m == 1 and not fast[key] == ref[key] == 0.0:  # by definition (GCV-7)
            _fail(case, f"{key} of the intercept alone: {fast[key]!r}, {ref[key]!r}")


# ---------------------------------------------------------------------------
# The whole fit


def compare_fits(case: Case, fast_fit, ref_fit) -> Outcome:
    """Compare ``_core.fit_mars`` with ``mars_ref.fit_mars`` on one case
    (CORE-1, CORE-3, CORE-5): ``fast_fit`` is ``_core.fit_mars(X, Y, w,
    MarsParams(**params), record_candidates=True)`` and ``ref_fit`` is
    ``MarsFit.from_dict`` of the reference's dict; either may also be a dict
    of CORE-5. The resolved values and the forward records
    are compared first; when the forward records agree to the end, the
    pruning records, the selected terms and the final fit."""
    f, r = _plain(fast_fit), _plain(ref_fit)
    for key in ("n_eff", "max_terms", "penalty"):
        if f[key] != r[key]:
            _fail(case, f"{key}: fast {f[key]!r}, reference {r[key]!r}")
    out = compare_forward(case, f["forward"], r["forward"])
    if out.near_tie is not None:
        return out
    kept, fwd = case.kept, _plain(r["forward"])
    dirs, cuts = fwd["dirs"][fwd["kept"]], fwd["cuts"][fwd["kept"]]
    B = mars_ref.basis_matrix(kept.X, dirs, cuts)

    def pruning(d):
        final = {k: d[k] for k in ("coef", "rss", "gcv", "rsq", "grsq")}
        sel = np.searchsorted(fwd["kept"], d["selected"])
        return {**_plain(d["pruning"]), **final, "selected": sel}

    pr = compare_pruning(case, pruning(f), pruning(r), B, kept.Y, kept.w)
    if pr.near_tie is None:
        for key in ("selected", "dirs", "cuts"):
            if not np.array_equal(f[key], r[key]):
                _fail(case, f"{key}: fast {f[key].tolist()}, ref {r[key].tolist()}")
    return Outcome(out.steps, pr.near_tie, pr.step, out.plan_tie)


@dataclasses.dataclass
class PruneCase:
    """A random basis for the pruning pass: B (n, M) with the intercept first,
    Y (n, K), w (n,) with zeros allowed or None, and the pass's settings."""

    name: str
    B: np.ndarray
    Y: np.ndarray
    w: np.ndarray | None
    params: dict

    def reproduction(self) -> str:
        return _reproduction(self.params, B=self.B, Y=self.Y, w=self.w)


# ---------------------------------------------------------------------------
# The fixture datasets (validation/fixtures), in the supported settings


def _earth_params(args: dict) -> dict:
    """earth's arguments as MarsParams fields (API-7, through the harness's
    name table); earth's 0 for minspan and endspan is None, automatic."""
    from validation.harness import names_map

    names = {e: p for p, e in names_map.EARTH_NAMES.items()}
    out = {names[e]: v for e, v in args.items()}
    for key in ("minspan", "endspan"):
        if out.get(key) == 0:
            out[key] = None
    return out


def _in_supported(name: str, X, Y, w, params: dict) -> list[Case]:
    """The case in the supported settings: an unsupported degree becomes the
    largest supported one below it, an unsupported fast_k the first supported
    one, unsupported weights none, and unsupported several responses one case
    per response."""
    params = dict(params)
    degree = params.get("max_degree", 1)
    if degree not in SUPPORTED["max_degree"]:
        params["max_degree"] = max(d for d in SUPPORTED["max_degree"] if d <= degree)
    if params.get("fast_k", 20) not in SUPPORTED["fast_k"]:
        params["fast_k"] = SUPPORTED["fast_k"][0]
    if w is not None and True not in SUPPORTED["weights"]:
        w = None
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    if Y.shape[1] in SUPPORTED["responses"]:
        return [Case(name, X, Y, w, params)]
    return [Case(f"{name}[y{k}]", X, Y[:, [k]], w, params) for k in range(Y.shape[1])]


def _basis_case(name: str, X, Y, w, dirs, cuts, args: dict) -> PruneCase | None:
    """earth's forward basis of one fixture fit (every term it added, TERM-3),
    with 0.0 where earth stores the smallest x at a linear factor (TERM-2),
    and earth's pruning settings; None for a constant response (EDGE-1),
    which the pruning pass does not receive."""
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    if np.all(Y[0] == Y):
        return None
    dirs, cuts = np.asarray(dirs), np.array(cuts, dtype=float)
    cuts[(dirs == 0) | (dirs == 2)] = 0.0
    B = mars_ref.basis_matrix(np.asarray(X, dtype=float), dirs, cuts)
    params = _earth_params(args)
    params = {
        "penalty": float(
            params.get("penalty", mars_ref.default_penalty(params.get("max_degree", 1)))
        ),
        "pmethod": params.get("pmethod", "backward"),
        "nprune": params.get("nprune"),
    }
    return PruneCase(name, B, Y, w, params)


def _fixture_cases() -> tuple[list[Case], list[PruneCase]]:
    """Every S dataset fixture, and every S15 draw rebuilt from the simulation
    DGPs (``gen_fixtures.s15_draws``): the forward cases, in the supported
    settings and without repeats, and earth's forward basis of each fit, in
    its own settings, for the pruning pass. String labels (S18) become one 0/1
    response per class, as the classifier's passes see them (GLM-1)."""
    from validation.harness import gen_fixtures
    from validation.sims import dgps, seeds

    cases, bases, seen = [], [], set()

    def add(new, basis):
        for c in new:
            w = None if c.w is None else c.w.tobytes()
            key = (c.X.tobytes(), c.Y.tobytes(), w, tuple(sorted(c.params.items())))
            if key not in seen:
                seen.add(key)
                cases.append(c)
        if basis is not None:
            bases.append(basis)

    for path in sorted(FIXTURES_DIR.glob("S*.json")):
        if not path.name.startswith("S"):  # a case-insensitive file system
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        inputs, y, r = d["inputs"], d["inputs"]["y"], d["result"]
        if isinstance(y[0], str):
            classes = sorted(set(y))
            y = [[float(v == c) for c in classes] for v in y]
        w = inputs["weights"]
        w = None if w is None else np.asarray(w, dtype=float)
        args, X = d["earth_args"], inputs["X"]
        basis = None
        if "error" not in r:
            basis = _basis_case(path.stem, X, y, w, r["dirs"], r["cuts"], args)
        add(_in_supported(path.stem, X, y, w, _earth_params(args)), basis)
    diagnostics = dgps.load_diagnostics()
    draws = json.loads((FIXTURES_DIR / "s15_draws.json").read_text(encoding="utf-8"))
    for draw in draws["draws"]:
        name, noise, rep = draw["dgp"], draw["noise"], draw["rep"]
        rng, _ = seeds.train_test_rngs(f"S15_{name}_{noise}", rep)
        X, y, _ = dgps.generate(dgps.REGISTRY[name], rng, draw["n"], noise, diagnostics)
        X, _ = gen_fixtures.scaled_matrix(X)
        args = gen_fixtures._s15_earth_args(draw["degree"], draw["mode_family"])
        tag = f"S15_draw{rep:03d}"
        basis = _basis_case(tag, X, y, None, draw["dirs"], draw["cuts"], args)
        add(_in_supported(tag, X, y, None, _earth_params(args)), basis)
    component = json.loads(
        (FIXTURES_DIR / "components" / "pruning_fixed_basis.json").read_text()
    )
    for key in ("one_response", "several_responses"):
        c = component[key]
        args = {"degree": 2, "penalty": component["penalty"]}
        name = f"pruning_fixed_basis[{key}]"
        X, dirs, cuts = component["X"], c["dirs"], c["cuts"]
        bases.append(_basis_case(name, X, c["y"], None, dirs, cuts, args))
    for b in [b for b in bases if b.Y.shape[1] >= 3]:  # the rule for K = 2 (PRUNE-3)
        bases.append(dataclasses.replace(b, name=f"{b.name}[y0, y1]", Y=b.Y[:, :2]))
    return cases, bases


#: The ordinary data of the spec's Limits: |weighted mean| <= R_D sd for each
#: covariate, by max_degree (1, 2, and 3 or more).
R_D = {1: 1e4, 2: 300.0, 3: 30.0}


def ordinary(X, w, max_degree: int) -> bool:
    """Whether X and w are ordinary data (spec, Limits), on the cases with
    positive weight: each covariate's weighted mean at most R_D times its
    weighted standard deviation (a constant covariate never enters a term,
    EDGE-4, so it is exempt); for two covariates that are not equal bit for
    bit, the LA-4 ratio of each against the intercept and the other at least
    1e-6; distinct values of a covariate at least 1e-9 of its range apart;
    the positive weights normal and within a factor 1e12."""
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
    rows = w > 0
    X, w = np.asarray(X, dtype=float)[rows], w[rows]
    if not len(w):
        return True
    tiny = np.finfo(float).tiny
    if w.min() < tiny or w.max() > 1e12 * w.min():
        return False
    live = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
    r_d = R_D[min(max_degree, 3)]
    for j in live:
        x = X[:, j]
        mean = float(w @ x) / float(w.sum())
        sd = math.sqrt(float(w @ (x - mean) ** 2) / float(w.sum()))
        if abs(mean) > r_d * sd or np.min(np.diff(np.unique(x))) < 1e-9 * np.ptp(x):
            return False
    one = np.ones(len(w))
    for a in live:
        for b in live:
            if a != b and not np.array_equal(X[:, a], X[:, b]):
                cols = np.column_stack([one, X[:, b], X[:, a]])
                if compare.la4_ratios(cols, w)[2] < 1e-6:
                    return False
    return True


FIXTURE_CASES, BASIS_CASES = _fixture_cases()
# The heaviest fixture fits (1,000 cases at degree 2 or 3, about 25 s) run at
# the thorough profile only, in gate C, so that gate B keeps its time.
THOROUGH = os.environ.get("HYPOTHESIS_PROFILE") == "thorough"
FIXTURE_CASES = [
    c
    for c in FIXTURE_CASES
    if THOROUGH or len(c.X) < 1000 or c.params.get("max_degree", 1) == 1
]
# Only ordinary data are compared (spec v3, Limits). Five fixture fits are
# not: S10 (x2 a near copy of x0 after the harness's scaling, LA-4 ratio
# 1.6e-9; LIM-2) in two modes, and S12 x + 1e6 (3.2e6 times the spread;
# LIM-1) in three. The pruning pass on earth's bases of them is still
# compared, where a near-dependence stops it (STOP-7).
FIXTURE_CASES = [
    c for c in FIXTURE_CASES if ordinary(c.X, c.w, c.params.get("max_degree", 1))
]


# ---------------------------------------------------------------------------
# Hypothesis data

DATA_KINDS = ("smooth", "hinge", "ties", "scaled", "shifted", "small")


def _settings():
    """MarsParams fields; those that T11 limits come from SUPPORTED."""
    return st.fixed_dictionaries(
        {
            "max_degree": st.sampled_from(SUPPORTED["max_degree"]),
            "max_terms": st.sampled_from(
                [None] * 4 + list(range(21, 0, -2)) + [8, 6, 4, 2]
            ),
            "penalty": st.sampled_from([None, -1.0, 0.0, 2.0, 3.0, 7.5]),
            "thresh": st.one_of(st.just(0.0), st.sampled_from([0.001, 0.01, 0.05])),
            "minspan": st.sampled_from([None, 1, 2, 5]),
            "endspan": st.sampled_from([None, 1, 2, 5]),
            "adjust_endspan": st.sampled_from([0.0, 0.5, 1.0, 1.5, 2.0]),
            "auto_linpreds": st.booleans(),
            "fast_k": st.sampled_from(SUPPORTED["fast_k"]),
            "fast_beta": st.sampled_from([0.0, 1.0, 2.5]),
        }
    )


def _truth(rng, X, smooth: bool) -> np.ndarray:
    """A smooth truth, or a sum of one to three hinges at data quantiles; now
    and then plus a linear part, which a linear term can take (FWD-6)."""
    f = rng.normal(0, 3) * X[:, rng.integers(X.shape[1])] * (rng.uniform() < 0.3)
    if smooth:
        a = rng.uniform(1, 8)
        return f + np.sin(a * X[:, 0]) + (X[:, -1] - 0.5) ** 2 + X[:, 0] * X[:, -1]
    for _ in range(rng.integers(1, 4)):
        j = rng.integers(X.shape[1])
        t = np.quantile(X[:, j], rng.uniform(0.1, 0.9))
        f += rng.normal(0, 3) * np.maximum(rng.choice([-1, 1]) * (X[:, j] - t), 0)
    return f


def _ordinary_shift(rng, X, max_degree: int, top: bool = False) -> np.ndarray:
    """X with each covariate moved so that its mean is r times its standard
    deviation, r up to half of R_D (log-uniform, or the top of that range),
    with a random sign; bitwise-equal columns move alike, so copies stay
    copies (FWD-5, EDGE-4), and constant columns stay."""
    X, done = X.copy(), {}
    half = R_D[min(max_degree, 3)] / 2
    for j in range(X.shape[1]):
        x = X[:, j]
        key = x.tobytes()
        if key not in done:
            sd = float(np.std(x))
            r = half if top else 10.0 ** rng.uniform(0, math.log10(half))
            done[key] = 0.0 if sd == 0 else rng.choice([-1, 1]) * r * sd - x.mean()
        X[:, j] = x + done[key]
    return X


@st.composite
def forward_cases(draw, kind: str) -> Case:
    """A case of the given kind, always ordinary data (``ordinary``; spec,
    Limits): ``smooth`` and ``hinge`` truths on uniform covariates; ``ties``,
    covariates on 2 to 20 levels, in about half of the draws with a bitwise
    copied column, or a constant column, or all constant, and now and then
    shifted within the ordinary bound (FWD-5, KNOT-2, LA-4, EDGE-3, EDGE-4);
    ``scaled``, covariates and y scaled by powers of 10 and shifted, the
    covariates within the ordinary bound (LA-7, EDGE-6, EDGE-7, FWD-10);
    ``shifted``, y + c with |c| of 1e10 or 1e13 and covariates at the top of
    the ordinary bound, where only centering keeps the sums of squares within
    LA-5 (FWD-10, LA-6, the plan's "Fast path"); ``small``, 1 to 15 cases,
    now and then with a constant y (EDGE-1, EDGE-2, STOP-3). The noise runs
    from none (exact fits and STOP-5) to as large as the signal. v2's copies
    at another shift, divided by 8 or one ulp off, and shifts of 1e6 to 1e10
    times the spread, are not ordinary data (LIM-1, LIM-2; #150), and
    ``test_limits_end_with_a_valid_model`` runs such inputs."""
    seed = draw(st.integers(0, 2**32 - 1))
    p = draw(st.integers(1, 3 if kind == "small" else 4))
    n = draw(st.integers(1, 15) if kind == "small" else st.integers(20, 120))
    params = draw(_settings())
    noise = draw(st.sampled_from([0.0, 1e-6, 0.01, 0.3, 1.0]))
    K = draw(st.sampled_from(SUPPORTED["responses"]))
    weighted = draw(st.sampled_from(SUPPORTED["weights"]))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    degree = params["max_degree"]
    if kind == "ties":
        levels = rng.choice([2, 3, 5, 10, 20])
        X = np.floor(X * levels) / levels
        if p >= 2 and rng.uniform() < 0.5:  # a bitwise copy (FWD-5, LA-4)
            X[:, 1] = X[:, 0]
        if p >= 3 and rng.uniform() < 0.3:
            X[:, 2] = X[0, 2]
        if rng.uniform() < 0.1:  # no covariate has a candidate (EDGE-3)
            X[:] = X[0]
        if rng.uniform() < 0.3:  # large means within the ordinary bound
            X = _ordinary_shift(rng, X, degree)
    smooth = kind == "smooth" or rng.uniform() < 0.3
    Y = np.column_stack([_truth(rng, X, smooth) for _ in range(K)])
    Y = Y + noise * np.std(Y) * rng.standard_normal((n, K))
    if kind == "small" and rng.uniform() < 0.1:  # a degenerate fit (EDGE-1)
        Y[:] = Y[0]
    if kind == "scaled":
        X = X * 10.0 ** rng.uniform(-6, 6, p)
        if rng.uniform() < 0.7:
            X = _ordinary_shift(rng, X, degree)
        Y = Y * 10.0 ** rng.uniform(-6, 6) + rng.choice(
            [0, 1, -1]
        ) * 10.0 ** rng.uniform(0, 10)
    if kind == "shifted":
        X = _ordinary_shift(rng, X, degree, top=True)
        Y = Y + rng.choice([0.0, 1e10, -1e10, 1e13, -1e13])
    w = None
    if weighted:
        w = [rng.uniform(0.2, 3.0, n), rng.integers(0, 4, n).astype(float)][
            rng.integers(2)
        ]
        w[0] = max(w[0], 1.0)
    assume(ordinary(X, w, degree))
    return Case(f"hypothesis {kind} seed={seed}", X, Y, w, params)


@st.composite
def pruning_cases(draw) -> PruneCase:
    """An intercept, then hinge and linear columns of Gaussian covariates, as
    in a forward basis; Y a linear combination of them plus noise, K = 1 to
    3, now and then shifted by 1e10 or 1e13, where only a centered TSS meets
    LA-5 (GCV-1, PRUNE-2); no weights, positive weights, or integer weights
    with zeros (W-3)."""
    seed = draw(st.integers(0, 2**32 - 1))
    M = draw(st.integers(1, 10))
    n = draw(st.integers(2 * M + 3, 60))
    K = draw(st.sampled_from([1, 2, 3]))
    weights = draw(st.sampled_from(["none", "positive", "integers"]))
    params = {
        "penalty": draw(st.sampled_from([-1.0, 0.0, 2.0, 3.0, 5.0])),
        "pmethod": draw(st.sampled_from(["backward", "none"])),
        "nprune": draw(
            st.sampled_from([None, 1, max(1, M // 2), max(1, M - 1), M + 1])
        ),
    }
    noise = draw(st.sampled_from([1e-3, 0.1, 1.0]))
    shift = draw(st.sampled_from([0.0] * 8 + [1e10, -1e10, 1e13, -1e13]))
    greedy = draw(st.booleans())
    rng = np.random.default_rng(seed)
    cols = [np.ones(n)]
    for _ in range(1, M):
        x = rng.standard_normal(n)
        t = np.quantile(x, rng.uniform(0.1, 0.7))
        cols.append([x, np.maximum(x - t, 0), np.maximum(t - x, 0)][rng.integers(3)])
    B = np.column_stack(cols)
    Y = B @ rng.standard_normal((M, K)) + noise * rng.standard_normal((n, K))
    Y = Y + shift
    w = {
        "none": None,
        "positive": rng.uniform(0.2, 3.0, n),
        "integers": rng.integers(0, 4, n).astype(float),
    }[weights]
    rows = np.ones(n, bool) if w is None else w > 0
    assume(rows.sum() > M + 1 and np.linalg.matrix_rank(B[rows]) == M)
    # A shift can round a response with little noise to a constant, a
    # degenerate fit (EDGE-1), which the pruning pass does not receive.
    assume(not np.all(Y[rows][0] == Y[rows]))
    # Half of the bases in the order of a greedy forward selection, as a
    # forward pass adds terms, so that the first offer of PRUNE-3 can beat
    # backward elimination; the others in their drawn order, so that the first
    # terms that pmethod="none" keeps differ from T[m] (PRUNE-7).
    ww, order, rest = np.where(rows, 1.0 if w is None else w, 0.0), [0], [*range(1, M)]
    while rest and greedy:
        best = min(rest, key=lambda j: mars_ref.rss(B[:, [*order, j]], Y, ww))
        order.append(best)
        rest.remove(best)
    return PruneCase(f"pruning seed={seed}", B[:, order + rest], Y, w, params)


def _on_binary_grids(design: str) -> tuple[np.ndarray, np.ndarray]:
    """Data whose values lie on binary grids, so that a shift by a power of 2
    is exact. "pairs" and "one covariate" have a hinge in x0 and a sine in the
    last covariate; in "linear", x0 is x1 plus noise and y = 5 (x1 - x0); in
    "interaction", hinges on x1 multiply x0 and a hinge on x2."""
    n = 200
    if design == "interaction":
        rng = np.random.default_rng(3)
        X = np.round(rng.uniform(size=(n, 3)) * 2**12) / 2**12
        y = 5 * np.maximum(X[:, 1] - 0.4, 0) * (X[:, 0] + np.maximum(X[:, 2] - 0.5, 0))
        return X, np.round((y + 0.1 * rng.normal(size=n)) * 2**10) / 2**10
    if design == "linear":
        rng = np.random.default_rng(3)
        x1 = np.round(rng.uniform(size=n) * 2**12) / 2**12
        x0 = np.round((x1 + 0.02 * rng.normal(size=n)) * 2**12) / 2**12
        y = 5 * (x1 - x0) + 1e-3 * rng.normal(size=n)
        return np.column_stack((x0, x1)), np.round(y * 2**20) / 2**20
    grid, p = (12, 2) if design == "pairs" else (6, 1)
    rng = np.random.default_rng(0)
    u = np.round(rng.uniform(size=n) * 2**grid) / 2**grid
    X = np.column_stack((u, rng.uniform(size=n)))[:, :p]
    y = np.sin(5 * X[:, -1]) + np.maximum(u - 0.5, 0.0) + 0.05 * rng.normal(size=n)
    return X, np.round(y * 2**8) / 2**8


def _merge_cases() -> list[Case]:
    """FWD-12 on discrete covariates with a product in the truth: the same rows
    come from two searches, as linear candidates (the cut that a search adds
    matters with ``auto_linpreds=False``) and as single hinges. The comparison
    covers the whole record and the candidate log, the recorded parent and the
    second best's parent and covariate included (the first occurrence in the
    order of FWD-5 decides). Seeds from the review of #111."""
    cases = []
    for seed in (619, 486, 613, 690):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([12, 20, 40, 100]))
        p = int(rng.choice([2, 3, 4]))
        levels = int(rng.choice([2, 3, 4, 6, 10]))
        X = np.floor(rng.uniform(size=(n, p)) * levels) / levels
        y = sum(
            rng.normal() * np.maximum(X[:, i] - np.median(X[:, i]), 0) for i in range(p)
        )
        y = y + rng.normal() * X[:, 0] * X[:, -1]
        y = y + rng.choice([0, 0.01, 0.3]) * rng.normal(size=n)
        kw = {
            "max_degree": int(rng.choice([2, 3])),
            "auto_linpreds": bool(rng.integers(2)),
            "fast_k": int(rng.choice([0, 3, 5, 20])),
            "thresh": 0.0,
            "max_terms": int(rng.choice([7, 13, 21])),
        }
        if rng.uniform() < 0.4:
            kw |= {"minspan": 1, "endspan": 1}
        cases.append(Case(f"merged candidates, seed {seed}", X, y, None, kw))
    return cases


def _designed_cases() -> list[Case]:
    """Designs that earlier tests and reviews built for one rule each; here the
    reference gives the answer."""
    rng = np.random.default_rng(0)
    x0 = rng.uniform(size=300)
    x1 = x0 + 0.02 * rng.normal(size=300)
    y = 5 * (np.maximum(x0 - 0.5, 0) - np.maximum(x1 - 0.5, 0))
    y = y + 0.001 * rng.normal(size=300)
    X = np.column_stack((x0, x1))
    spans = {"thresh": 0.0, "minspan": 1, "endspan": 1, "fast_k": 0}
    # FWD-4: a pair on x1 comes first, and then every pair on x0 reduces the
    # RSS by more than 10 Delta_1, while the linear term may (with either
    # form of FWD-6); the same with X + 2^11, about 7e3 times the spread, near
    # the top of ordinary data at degree 1 (LA-5 for shifted covariates).
    cases = [
        Case(f"FWD-4 cap, auto_linpreds={a}", X, y, None, spans | {"auto_linpreds": a})
        for a in (True, False)
    ]
    cases.append(
        Case("FWD-4 cap, X + 2^11", X + 2.0**11, y, None, spans | {"max_terms": 11})
    )
    # A few values far below a tight bulk (the scan's error bounds).
    n = 400
    x0 = np.r_[rng.uniform(0, 1, 12), 40 + rng.uniform(size=n - 12)]
    Xb = np.column_stack((x0, rng.uniform(size=n)))
    yb = np.maximum(x0 - 0.5, 0) + np.sin(6 * Xb[:, 1]) + 0.05 * rng.normal(size=n)
    cases.append(Case("values below a tight bulk", Xb, yb, None, {"fast_k": 0}))
    # LA-7: x1 is x0 plus noise, so A of x0 after a pair on x1 is near the
    # pair rule's threshold.
    for sd in (4e-3, 1e-3):
        x0 = rng.uniform(size=200)
        Xp = np.column_stack((x0, x0 + sd * rng.normal(size=200)))
        yp = np.maximum(x0 - 0.5, 0) + 5 * Xp[:, 1]
        cases.append(Case(f"LA-7, sd {sd}", Xp, yp, None, spans | {"max_terms": 5}))
    # STOP-5 before STOP-1: an exact fit.
    x = np.linspace(0, 1, 41)
    yx = 2 * np.maximum(x - x[12], 0.0)
    cases.append(Case("exact fit", x[:, None], yx, None, spans | {"max_terms": 3}))
    # FWD-5 and EDGE-4: a duplicated covariate (below).
    # Exact shifts by powers of 2 on binary grids, near the top of ordinary
    # data: x0 by 2^11 at degree 1 (A of LA-7, a pair's first column, the
    # linear candidate), and y by 2^44 (the TSS) [FWD-10]; at degree 2, x0 by
    # 2^6, about 220 times its spread, enters products as b x0 (FWD-6).
    for design, dx, dy in [
        ("one covariate", 2.0**11, 0.0),
        ("pairs", 2.0**11, 0.0),
        ("linear", 2.0**11, 0.0),
        ("pairs", 0.0, 2.0**44),
        ("interaction", 2.0**6, 0.0),
    ]:
        Xg, yg = _on_binary_grids(design)
        Xg[:, 0] += dx
        kw = spans | {"max_terms": 11}
        if design == "interaction":
            kw |= {"max_degree": 2, "auto_linpreds": False}
        cases.append(Case(f"{design}, x0 + {dx:g}, y + {dy:g}", Xg, yg + dy, None, kw))
    u = rng.uniform(size=100)
    Xd = np.column_stack((u, u, rng.uniform(size=100)))
    yd = np.sin(6 * u) + 0.1 * Xd[:, 2]
    cases.append(
        Case("a duplicated covariate", Xd, yd, None, {"fast_k": 0, "thresh": 0.0})
    )
    cases += _merge_cases()
    # STOP-3: GRSq' of the first candidate is -14.6, between -20 and -10: code 3
    rng = np.random.default_rng(2)
    Xs, ys = rng.uniform(size=(10, 2)), rng.normal(size=10)
    cases.append(Case("GRSq' between -20 and -10", Xs, ys, None, {"penalty": 5.0}))
    return [dataclasses.replace(c, Y=np.reshape(c.Y, (-1, 1))) for c in cases]


def _limit_cases() -> list[Case]:
    """Inputs outside ordinary data, one or two of each limit of the spec
    (LIM-1 to LIM-4), from the designs that v2's exact rules served (#110,
    #113, #130, #150): covariates at 1e7 to 2^46 times their spread, alone
    and in products at degree 2 and 3; x1 = x0 / 8 at another shift, and x1
    one ulp off x0 at one case; two values of a covariate one ulp apart, so
    that knots are; weights from 1e-300 to 1."""
    rng = np.random.default_rng(7)
    cases = []
    x = np.floor(rng.uniform(size=(40, 3)) * 5) / 5
    y = np.sin(4 * x[:, 0]) + x[:, 0] * x[:, 2] + 0.01 * rng.normal(size=40)
    deep = {"max_degree": 3, "fast_k": 0, "thresh": 0.0, "minspan": 1, "endspan": 1}
    for shift in (1e7, 2.0**26, 2.0**46):  # LIM-1
        cases.append(Case(f"LIM-1, X + {shift:g}", x + shift, y, None, deep))
    binary = np.column_stack([1e7 + (np.arange(60) % 2), rng.uniform(size=60)])
    yb = 2 * (binary[:, 0] - 1e7) + 3 * np.maximum(binary[:, 1] - 0.5, 0)
    cases.append(Case("LIM-1, a binary covariate + 1e7", binary, yb, None, {}))
    eighth = x.copy()  # LIM-2
    eighth[:, 1] = x[:, 0] / 8 + 2.0**26
    cases.append(Case("LIM-2, x1 = x0 / 8 + 2^26", eighth, y, None, deep))
    ulp = x + 1e9
    ulp[:, 1] = ulp[:, 0]
    ulp[3, 1] = np.nextafter(ulp[3, 1], np.inf)
    cases.append(Case("LIM-2, x1 one ulp off x0", ulp, y, None, deep))
    knots = rng.uniform(size=(50, 2))  # LIM-3
    knots[1, 0] = np.nextafter(knots[0, 0], np.inf)
    yk = np.maximum(knots[:, 0] - knots[0, 0], 0) * 1e3 + knots[:, 1]
    cases.append(Case("LIM-3, values one ulp apart", knots, yk, None, deep))
    w = 10.0 ** rng.choice([-300.0, -150.0, 0.0], 40)  # LIM-4
    w[0] = 1.0
    cases.append(Case("LIM-4, weights 1e-300 to 1", x, y, w, {"max_degree": 2}))
    return [dataclasses.replace(c, Y=np.reshape(c.Y, (-1, 1))) for c in cases]


# ---------------------------------------------------------------------------
# The tests


@pytest.mark.parametrize("kind", DATA_KINDS)
@given(data=st.data())
def test_forward_pass_on_hypothesis_data(kind, data):
    """The fast forward pass equals the reference's on data of each kind, up
    to the first near-tie (FWD-1 to FWD-11, STOP-1 to STOP-7, LA-2 to LA-7,
    KNOT-1 to KNOT-6, SPAN-1 to SPAN-5, LIMIT-2, EDGE-1, EDGE-6, CORE-3,
    CORE-4)."""
    case = data.draw(forward_cases(kind), label="case")
    _count(f"forward, hypothesis {kind}", check_forward(case))


@given(data=st.data())
def test_whole_fit_on_hypothesis_data(data):
    """_core.fit_mars equals the reference's fit_mars, read through
    MarsFit.from_dict, on data of every kind with pmethod and nprune drawn
    (CORE-1 to CORE-5, PRUNE-5 to PRUNE-8, EDGE-1, EDGE-6), up to the first
    near-tie of either pass."""
    kind = data.draw(st.sampled_from(DATA_KINDS), label="kind")
    case = data.draw(forward_cases(kind), label="case")
    case.params["pmethod"] = data.draw(st.sampled_from(["backward", "none"]))
    case.params["nprune"] = data.draw(st.sampled_from([None, 1, 3, 10]))
    params = _core.MarsParams(**case.params)
    fast = _core.fit_mars(case.X, case.Y, case.w, params, record_candidates=True)
    ref = _core.MarsFit.from_dict(case.reference_fit())
    _count("whole fit, hypothesis", compare_fits(case, fast, ref))


@pytest.mark.parametrize("case", _merge_cases(), ids=lambda c: c.name)
def test_merged_candidates_record_the_reference_parents(case):
    """FWD-12: the recorded parent of each term and the second best's parent,
    covariate and kind equal the reference's exactly (the comparison of
    ``check_forward`` takes the same rows from another parent as a tie)."""
    case = dataclasses.replace(case, Y=np.reshape(case.Y, (-1, 1)))
    fast, ref = _plain(case.fast_forward()), case.reference_fit()["forward"]
    assert fast["parent"].tolist() == ref["parent"].tolist()
    for key in ("second_parent", "second_variable", "second_kind"):
        got, want = _plain(fast["candidates"])[key], ref["candidates"][key]
        assert got.tolist() == want.tolist()


def test_whole_fit_of_two_cases_with_weights():
    """PRUNE-8: an exact fit of two weighted cases (hypothesis `small`, seed
    9004). The weighted mean of the fitted values is 0.14666045613757861 in
    exact arithmetic; the fast fit is 3.6e-16 from it and the reference 2.3e-15
    (about 10 ulp), so the mean of the difference is rounding of the reference's
    least squares, within 8 ulp of max |y| per term."""
    X = np.array(
        [
            [0.7019429236618134, 0.07319247226045111],
            [0.2140700088820472, 0.8836087384301462],
        ]
    )
    Y = np.array([[-0.3900669638755577], [1.5168667116401282]])
    w = np.array([2.2183432527174323, 0.8689535943607877])
    params = {"max_degree": 1, "adjust_endspan": 0.0, "fast_k": 0, "thresh": 0.0}
    case = Case("two cases", X, Y, w, params)
    fast = _core.fit_mars(X, Y, w, _core.MarsParams(**params), record_candidates=True)
    ref = _core.MarsFit.from_dict(case.reference_fit())
    compare_fits(case, fast, ref)


@pytest.mark.parametrize("case", _designed_cases(), ids=lambda c: c.name)
def test_forward_pass_on_designed_data(case):
    """The same comparison on designs made for single rules (``_designed_cases``),
    all ordinary data (spec, Limits)."""
    assert ordinary(case.X, case.w, case.params.get("max_degree", 1))
    _count("forward, designs", check_forward(case))


# The numpy fallback of the knot scan (_scan._centered) divides 0 by 0 where
# products of weights of 1e-300 underflow (LIM-4) and warns; the fit still
# ends with a valid model, which is all that Limits asks (reported on #160).
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("case", _limit_cases(), ids=lambda c: c.name)
def test_limits_end_with_a_valid_model(case):
    """Limits (LIM-1 to LIM-4): outside ordinary data both programs still end
    with a valid model of the float64 columns, and nothing more is compared:
    finite RSS, GCV not NaN, finite coefficients of the selected terms, and
    selected terms among the forward terms that FWD-11 kept."""
    assert not ordinary(case.X, case.w, case.params.get("max_degree", 1))
    params = _core.MarsParams(**case.params)
    fast = _plain(_core.fit_mars(case.X, case.Y, case.w, params))
    ref = _plain(_core.MarsFit.from_dict(case.reference_fit()))
    for fit in (fast, ref):
        kept = fit["forward"]["kept"]
        assert np.isfinite(fit["rss"]) and fit["rss"] >= 0
        assert not np.isnan(fit["gcv"])
        assert np.all(np.isfinite(fit["coef"]))
        assert set(np.asarray(fit["selected"]).tolist()) <= set(kept.tolist())


@pytest.mark.slow
@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda c: c.name)
def test_forward_pass_on_the_fixture_datasets(case):
    """The same comparison on every fixture dataset, S01 to S20 and the S15
    draws, in the supported settings."""
    group = "S15" if case.name.startswith("S15") else "S01 to S20"
    _count(f"forward, {group}", check_forward(case))


def check_pruning(case: PruneCase) -> Outcome:
    """Run both pruning passes and final fits on the case and compare them."""
    B, Y, w = case.B, case.Y, case.w
    ww = np.ones(len(B)) if w is None else w
    N0 = mars_ref.weight_sum(ww[ww > 0])
    tau_N = mars_ref.weight_tol(N0)
    ref = mars_ref.prune(
        B, Y, ww, N=mars_ref.snap(N0, tau_N), tau_N=tau_N, **case.params
    )
    Yf = Y[:, 0] if Y.shape[1] == 1 else Y
    pp = _pruning.pruning_pass(B, Yf, w, **case.params)
    ff = _pruning.final_fit(B, Yf, pp.selected, w, penalty=case.params["penalty"])
    return compare_pruning(case, {**pp._asdict(), **ff._asdict()}, ref, B, Y, ww)


@given(pruning_cases())
def test_pruning_pass_on_random_bases(case):
    """The fast pruning pass and final fit equal the reference's on random
    bases, with and without weights, for one and several responses (PRUNE-2
    to PRUNE-8, GCV-2, GCV-5, GCV-6, W-3, RESP-1), up to the first near-tie."""
    group = f"pruning, random, K {'= 1' if case.Y.shape[1] == 1 else '>= 2'}"
    group += ", y shifted" if np.abs(case.Y).min() > 1e9 else ""
    _count(group, check_pruning(case))


@pytest.mark.slow
@pytest.mark.parametrize("case", BASIS_CASES, ids=lambda c: c.name)
def test_pruning_pass_on_earth_forward_bases(case):
    """The same comparison on earth's forward basis of every fixture fit, at
    degrees 1 to 3, with the fixtures' weights and several responses."""
    group = case.name[:3] if case.name[:3] in ("S15", "pru") else "S01 to S20"
    group = {"pru": "components"}.get(group, group)
    _count(f"pruning, earth bases, {group}", check_pruning(case))


def test_a_parent_gives_a_row_by_removing_one_covariate():
    """FWD-12: the product h(x0 - .5) h(x1 - .3) comes from either factor as
    its parent, and from no other term."""
    rec = {
        "dirs": np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0]]),
        "cuts": np.array([[0, 0], [0.5, 0], [0, 0.3], [0.5, 0.3], [0.4, 0]]),
    }
    assert [_parent_gives(rec, p, 3) for p in range(5)] == [
        False,
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("shift", [0.0, 3e6])
def test_a_value_off_at_a_near_dependence_stops_the_comparison(shift):
    """STOP-7's near-dependence (spec v3): x0 = shift + a 0/1 pattern has the
    LA-4 ratio 0.5 / shift against the intercept, 1.7e-7 at shift 3e6: x0
    counts, within a factor 2 of 1e-7 (at 5e6 the ratio is 1e-7 itself, and
    the reference's float test drops x0, as LA-4 allows there). A fast
    RSS 1e-6 off the reference's, far outside LA-5, then stops the forward
    comparison after step 1 (the linear x0) and the pruning comparison at
    size 3, all columns with x0, as a near-dependence, while at shift 0 the
    same error fails."""
    rng = np.random.default_rng(1)
    X = np.column_stack([shift + (np.arange(60) % 2), rng.uniform(size=60)])
    y = 2 * (np.arange(60) % 2) + 3 * np.maximum(X[:, 1] - 0.5, 0)
    y = (y + 0.05 * rng.normal(size=60))[:, None]
    case = Case(f"x0 + {shift:g}", X, y, None, {"fast_k": 0})
    ref, fast = case.reference_fit()["forward"], _plain(case.fast_forward())
    assert ref["dirs"][1].tolist() == [2, 0]  # step 1 adds x0 linearly
    fast["rss"] = fast["rss"] * np.r_[1.0, 1 + 1e-6, np.ones(len(fast["rss"]) - 2)]
    B = np.column_stack([np.ones(60), X])
    prune = PruneCase(
        case.name, B, y, None, {"penalty": 2.0, "pmethod": "backward", "nprune": None}
    )
    ww = np.ones(60)
    pr = mars_ref.prune(
        B, y, ww, N=60.0, tau_N=mars_ref.weight_tol(60.0), **prune.params
    )
    off = {**pr, "rss_per_size": pr["rss_per_size"] * np.r_[1.0, 1.0, 1 + 1e-6]}
    if shift:
        assert compare_forward(case, fast, ref).near_tie == NEAR_DEPENDENCE
        assert compare_pruning(prune, off, pr, B, y, ww).near_tie == NEAR_DEPENDENCE
    else:
        with pytest.raises(AssertionError, match="RSS after step 1"):
            compare_forward(case, fast, ref)
        with pytest.raises(AssertionError, match=r"rss_per_size\[2\]"):
            compare_pruning(prune, off, pr, B, y, ww)
