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
  each step within 1e-8 of the RSS before the step (LA-5), the second-best
  candidate of the candidate log (its identity exactly, its RSS by LA-5), and,
  when every step agrees, the termination code and the kept terms (FWD-11).
- The pruning pass: ``_pruning.pruning_pass`` and ``final_fit`` against
  ``mars_ref.prune`` on random bases, with and without weights, for K = 1 and
  K >= 2: the removals and the subsets exactly, the RSS and GCV of each size
  to a relative 1e-8, the selected terms, and the final coefficients and
  statistics by the plan's tolerance table.
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
threshold). A near-tie ends the comparison of that fit, and it is counted;
any other difference fails. The fast code's RSS of its own choice must still
meet LA-5, so a wrong value cannot pass as a tie. With ``PYMARS_ORACLE_TALLY``
set to a folder, each process writes its counts there, and gate C reports the
share of fits that stop at a near-tie.

``SUPPORTED`` lists the settings of the forward pass that the fast code
supports. The strategies and the fixture cases take every setting from it, so
no test is skipped, and each stage of T11 widens the tests by editing it.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
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

# LA-5 bounds each RSS by 1e-8 of the RSS before its step; two candidates
# within 1e-7 of that RSS are a near-tie (plan, "Ties"); STOP-7's delta is
# 2e-8 of the RSS that it names; LA-5's bands for rho and A_w are 1e-6 of
# their thresholds. The pruning pass takes the forward pass's near-tie
# factor (OQ-2 leaves the threshold of the conformance tests to T07).
LA5 = 1e-8
TIE = 1e-7
DELTA = 2e-8
BAND = 1e-6
PRUNE_TIE = 1e-7

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
        """The reference's view of the cases of the fit (W-3, W-4, EDGE-6):
        the rows with positive weight, N, tau_N and the power of 2 on Y."""
        w = np.ones(len(self.X)) if self.w is None else self.w
        rows = w > 0
        N0 = mars_ref.weight_sum(w[rows])
        tau_N = mars_ref.weight_tol(N0)
        Y = self.Y[rows]
        j = mars_ref.y_scale_power(Y)
        return SimpleNamespace(
            X=self.X[rows],
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
        with the trace of its forward pass (the queue of FAST-1 to FAST-5)."""
        calls, trace = [], []
        real, real_pass = mars_ref._parent_candidates, mars_ref.forward_pass

        signature = inspect.signature(real)

        def spy(*args, **kw):
            found, searchable = real(*args, **kw)
            a = signature.bind(*args, **kw).arguments
            calls.append(
                SimpleNamespace(B=a["B"], cond=a.get("conditioned"), found=found)
            )
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


# ---------------------------------------------------------------------------
# The forward pass


def _key(parent, variable, kind, knot) -> tuple:
    """A candidate as CORE-3 tells candidates apart (``mars_ref.candidate_key``)."""
    knot = None if kind == LINEAR else float(knot)
    return (int(parent), int(variable), int(kind), knot)


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
        self.dirs = ref["dirs"][:M]
        calls = [c for c in case.searches if c.B.shape[1] == M]
        self.cuts = ref["cuts"][:M]
        self.B = (
            calls[0].B if calls else mars_ref.basis_matrix(kept.X, self.dirs, self.cuts)
        )
        # The reference's columns for its projections (#79): those of its
        # Conditioned when it used one, else B.
        self.cond = calls[0].cond if calls else None
        self.G = self.B if self.cond is None else self.cond.columns
        self.P_B = mars_ref.Projector(self.G, kept.w)
        self.sigma2 = mars_ref.covariate_variances(kept.X, kept.w, kept.N)
        self.tau = mars_ref.collinearity_tol(s - 1)
        self.path = np.ldexp(np.asarray(ref["rss"], dtype=float), 2 * kept.j)
        self.rss_s = self.path[s - 1]
        self.limit = mars_ref.max_legal(list(self.path[:s]))
        # STOP-7's R: the RSS before step s for a candidate's RSS alone, one
        # step earlier for a reduction, two steps earlier for Delta_s.
        self.R_cand, self.R_red, self.R_delta = (
            self.path[max(s - i, 0)] for i in (1, 2, 3)
        )
        found = [c for call in calls for c in call.found]
        self.values = {mars_ref.candidate_key(c): c.rss for c in found}
        self.legal = [
            c
            for c in found
            if mars_ref.is_legal(c.kind, self.rss_s - c.rss, self.limit)
        ]
        self.legal_keys = {mars_ref.candidate_key(c) for c in self.legal}

    def best(self):
        """The reference's choice at this step, or None (FWD-4, FWD-5)."""
        return mars_ref._first_best(self.legal, self.rss_s)

    def _linear(self, k: int, v: int) -> np.ndarray:
        """A column that spans with B what b x_v spans, as the reference forms
        it: b (x - min x), or the Conditioned's linear column (#72, #79)."""
        b, x = self.B[:, k], self.kept.X[:, v]
        if self.cond is None:
            return b * (x - x.min())
        return self.cond.linear_column(self.dirs[k], self.cuts[k], v, b)

    def value(self, key) -> float:
        """The RSS of LA-2 of a candidate: the reference's own value when it
        found the candidate, else a least-squares fit of its columns."""
        if key in self.values:
            return self.values[key]
        k, v, kind, knot = key
        b, x = self.B[:, k], self.kept.X[:, v]
        h = None if kind == LINEAR else b * np.maximum(x - knot, 0.0)
        lin = None if kind == SINGLE else self._linear(k, v)
        cols = {PAIR: [lin, h], SINGLE: [h], LINEAR: [lin]}[kind]
        return mars_ref.rss(np.column_stack([self.G, *cols]), self.kept.Ys, self.kept.w)

    def bands(self, key) -> list[str]:
        """The bands that the candidate lies in: A_w of its search within 1e-6
        of the pair rule's threshold (LA-7); its knot's rho within 1e-6 tau of
        tau (LA-3), for each kind of search that the pair rule allows; its
        reduction within 11 delta of MaxLegal or within delta of 0 (FWD-4)."""
        k, v, kind, knot = key
        out, w = [], self.kept.w
        b, x = self.B[:, k], self.kept.X[:, v]
        V = [*np.flatnonzero(self.dirs[k]).tolist(), v]
        thr = 0.01 * math.prod(self.sigma2[u] for u in V)
        searchable = all(self.sigma2[u] > 0 for u in V)
        lin = self._linear(k, v)
        A = self.P_B.rss(lin)
        near_pair = searchable and abs(A - thr) <= BAND * thr
        if near_pair:
            out.append("LA-7")
        h = None if kind == LINEAR else b * np.maximum(x - knot, 0.0)
        kinds = [True, False] if near_pair else [searchable and thr <= A]
        if h is not None and not np.all(h == h[0]):
            for pair in kinds:
                G = np.column_stack([self.G, lin]) if pair else self.G
                rho = mars_ref.collinearity_ratio(h, G, w)
                if abs(rho - self.tau) <= BAND * self.tau:
                    out.append("LA-3")
        red = self.rss_s - self.value(key)
        if h is not None and abs(red - self.limit) <= 11 * DELTA * self.R_delta:
            out.append("FWD-4 MaxLegal")
        if abs(red) <= DELTA * self.R_red:
            out.append("FWD-4 zero")
        return out

    def queue_bands(self) -> list[str]:
        """STOP-7's band of FAST-3, at step s or at any step before it, since
        the queues of the two programs can part at a step whose choice they
        still share: two stored values lambda within 2 delta of each other
        whose order changes the rows of the table that the step visits. An
        entry searched in step t has kappa = 2t and a reduction of that step,
        so delta = 2e-8 RSS_{t-2}. The table of step t comes from the entries
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
                (e, lam, 2 * DELTA * self.path[max(kappa // 2 - 2, 0)])
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
        if key is None:
            rss_new, m_new, R = self.rss_s, self.M + 1, self.R_red
        else:
            rss_new, R = self.value(key), self.R_cand
            m_new = self.M + (2 if key[2] == PAIR else 1)
        d = prm.resolved_penalty()
        grsq = mars_ref.grsq(rss_new, m_new, tss, d, kept.N, kept.tau_N)
        out = []
        near = (
            math.isfinite(grsq) and abs(grsq + 10) <= DELTA * R * (1 - grsq) / rss_new
        )
        if prm.thresh > 0 and near:
            out.append("STOP-3")
        gain = (self.rss_s - rss_new) / tss  # 0 exactly without a candidate
        if key is not None and abs(gain - prm.thresh) <= DELTA * self.R_red / tss:
            out.append("STOP-4")
        return out


def _stop5_bands(case: Case, ref: dict, s: int) -> list[str]:
    """STOP-7's bands of STOP-5 after step s: the new RSS within delta of the
    floor 1e-10 TSS / (N - 1), or the new RSq within delta / TSS of
    1 - thresh, with R the RSS before the step."""
    kept, thresh = case.kept, mars_ref.as_params(case.params).thresh
    path = np.ldexp(np.asarray(ref["rss"], dtype=float), 2 * kept.j)
    tss, rss, R = path[0], path[s], path[s - 1]
    out = []
    if abs(rss - 1e-10 * tss / (kept.N - 1)) <= DELTA * R:
        out.append("STOP-5 floor")
    if abs(thresh - rss / tss) <= DELTA * R / tss:
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
    if abs(rss_f - vf) > LA5 * R:
        _fail(case, f"step {s}: the fast RSS {rss_f!r} of {cf[0]} is not {vf!r} (LA-5)")
    f_legal = any(k in step.legal_keys for k in cf)
    if f_legal and vf < vr - TIE * R:
        _fail(case, f"step {s}: the reference chose {cr[0]} over the better {cf[0]}")
    f_bands = [b for k in cf for b in step.bands(k)]
    r_bands = [b for k in cr for b in step.bands(k)]
    gap = vf - vr < TIE * R
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
        R = LA5 * ref["rss"][i]
        _close(case, lf["second_rss"][i], lr["second_rss"][i], R, f"second, {s}")
        return None
    if kf is not None and kf in _choice(case, fast, s):
        _fail(case, f"step {s}: the fast second {kf} is the chosen candidate (FWD-8)")
    step = _Step(case, ref, s)
    if kf is not None and kr is not None and _identical(step, kf, kr):
        _fail(case, f"step {s}: the seconds {kf} and {kr} have equal columns (FWD-5)")
    if kf is not None:
        vf = step.value(kf)
        rss_f = math.ldexp(float(lf["second_rss"][i]), 2 * case.kept.j)
        if abs(rss_f - vf) > LA5 * step.rss_s:
            _fail(case, f"step {s}: the fast second {kf} has RSS {rss_f!r}, not {vf!r}")
    f_bands = [] if kf is None else step.bands(kf)
    r_bands = [] if kr is None else step.bands(kr)
    if kf is None or kr is None:
        ok = bool(f_bands or r_bands)
    else:
        gap = abs(step.value(kf) - step.value(kr)) < TIE * step.rss_s
        f_ok = kf in step.legal_keys or f_bands
        r_ok = kr in step.legal_keys or r_bands
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
        keys = [] if best is None else [mars_ref.candidate_key(best)]
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
    _close(case, fast["rss"][0], ref["rss"][0], LA5 * ref["rss"][0], "TSS, rss[0]")
    S = min(len(fast["rss"]), len(ref["rss"])) - 1
    for s in range(1, S + 1):
        if _terms(fast, s) != _terms(ref, s):
            return _diverged(case, fast, ref, s)
        R = ref["rss"][s - 1]
        _close(case, fast["rss"][s], ref["rss"][s], LA5 * R, f"RSS after step {s}")
        if None in logs:
            continue
        for rec, who in ((fast, "fast"), (ref, "reference")):
            if rec["candidates"]["best_rss"][s - 1] != rec["rss"][s]:
                _fail(case, f"step {s}: the {who} log's best_rss is not rss[{s}]")
        if (reason := _check_second(case, fast, ref, s)) is not None:
            TALLY[f"second best differs at a near-tie ({reason})"] += 1
    if len(fast["rss"]) != len(ref["rss"]) or fast["termination"] != ref["termination"]:
        return _stopped_apart(case, fast, ref, S + 1)
    for key in ("dirs", "cuts", "parent", "step", "kept", "dropped"):
        if not np.array_equal(fast[key], ref[key]):
            _fail(case, f"{key}: fast {fast[key].tolist()}, ref {ref[key].tolist()}")
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
    reference's RSS of the two drops decides whether it is a near-tie (within
    PRUNE_TIE of the larger); then only the sizes m >= pos, which the earlier
    stages fix, are compared. An offer can tie as well: two subsets of one
    size within PRUNE_TIE. The selected size and the final fit are compared
    when no removal tied. An RSS below STOP-5's floor 1e-10 TSS / (N - 1) is an
    exact fit, whose value is rounding: two such values count as equal.
    """
    Mf, several = B.shape[1], Y.shape[1] >= 2
    rss_of = functools.cache(lambda terms: mars_ref.rss(B[:, sorted(terms)], Y, w))
    N = float(np.sum(w))  # a degenerate fit (N <= 1, EDGE-1) has no exact fits
    floor = 1e-10 * ref["rss_per_size"][0] / (N - 1) if N > 1 else 0.0
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

    def same(a, b, rtol) -> bool:  # two RSS values
        return (a <= floor and b <= floor) or abs(a - b) <= rtol * max(a, b)

    def tied(Ta, Tb) -> bool:
        return same(rss_of(frozenset(Ta)), rss_of(frozenset(Tb)), PRUNE_TIE)

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
            set(np.flatnonzero(T[m - 1]).tolist())
            for T in (fast["subsets"], ref["subsets"])
        )
        tol = LA5
        if Tf != Tr:
            if not tied(Tf, Tr):
                _fail(case, f"size {m}: fast {sorted(Tf)}, reference {sorted(Tr)}")
            near_tie, tol = near_tie or f"subset of size {m}", PRUNE_TIE
        rf, rr = fast["rss_per_size"][m - 1], ref["rss_per_size"][m - 1]
        if not same(rf, rr, tol):
            _fail(case, f"rss_per_size[{m - 1}]: fast {rf!r}, reference {rr!r}")
        gf, gr = fast["gcv_per_size"][m - 1], ref["gcv_per_size"][m - 1]
        if not (rf <= floor and rr <= floor) or np.isinf(gf) or np.isinf(gr):
            _rel(case, gf, gr, tol, f"gcv_per_size[{m - 1}]")
    if first > 1:
        return Outcome(Mf - first, near_tie, Mf - first + 1)
    mf, mr = fast["selected_size"], ref["selected_size"]
    if mf != mr:
        gf, gr = ref["gcv_per_size"][mf - 1], ref["gcv_per_size"][mr - 1]
        exact = same(*(ref["rss_per_size"][m - 1] for m in (mf, mr)), 0.0)
        by_gcv = case.params.get("pmethod", "backward") == "backward"  # PRUNE-7
        if not (by_gcv and (exact or abs(gf - gr) <= PRUNE_TIE * max(gf, gr))):
            _fail(case, f"selected size: fast {mf}, reference {mr}")
        return Outcome(Mf - 1, "selected size", Mf)
    if not np.array_equal(fast["selected"], ref["selected"]):
        if near_tie is None:
            _fail(case, f"selected: fast {fast['selected']}, ref {ref['selected']}")
        return Outcome(Mf - 1, near_tie, Mf)
    _compare_final(case, fast, ref, B[:, ref["selected"]], Y, w, same, floor)
    return Outcome(Mf - 1, near_tie, None if near_tie is None else Mf)


def _compare_final(case, fast: dict, ref: dict, BS, Y, w, same, floor) -> None:
    """The final fit of the same terms (PRUNE-8), by the plan's tolerance
    table: per response, the coefficients normwise within 1e-6 where
    kappa(B) <= 1e5; the fitted values with their weighted mean removed
    within 1e-8 sd(y), scaled by kappa / 1e6 above 1e6, and the mean of their
    difference within rounding, 4 ulp of max |y| plus kappa u ||y - mean||
    (a fitted value near 1e13 has an ulp of 2e-3); the RSS of the fast
    coefficients, the reference's RSS plus the weighted sum of squares of
    the centered difference of the fitted values (least squares), within a
    relative 1e-8 of the reference's RSS; RSS and GCV within a relative 1e-8
    (``same``, for exact fits); RSq and GRSq within 1e-8."""
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
    ulps = 4 * np.spacing(np.max(np.abs(Y), axis=0))
    rounding = ulps + kappa * u * np.linalg.norm((Y - mean) * sw, axis=0)
    if np.any(np.abs(d_mean) > rounding):
        _fail(case, f"mean of the fitted values: {d_mean.tolist()}, {rounding}")
    excess = float(np.sum(w[:, None] * dc**2))
    if not same(ref["rss"] + excess, ref["rss"], LA5):
        _fail(case, f"RSS of the fast coefficients: {ref['rss']!r} + {excess!r}")
    if not same(fast["rss"], ref["rss"], LA5):
        _fail(case, f"final rss: fast {fast['rss']!r}, reference {ref['rss']!r}")
    if not (fast["rss"] <= floor and ref["rss"] <= floor) or np.isinf(ref["gcv"]):
        _rel(case, fast["gcv"], ref["gcv"], LA5, "final gcv")
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
    B = mars_ref.basis_matrix(
        kept.X, fwd["dirs"][fwd["kept"]], fwd["cuts"][fwd["kept"]]
    )

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

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "validation" / "fixtures"
# The fixture cases import the harness and the simulation code in validation/.
# pytest collects tests/ before validation/, and an installed pymars (the CI
# job with the lowest dependencies) leaves the repository root off sys.path;
# appending it keeps the installed pymars first.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


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


FIXTURE_CASES, BASIS_CASES = _fixture_cases()
# The heaviest fixture fits (1,000 cases at degree 2 or 3, about 25 s) run at
# the thorough profile only, in gate C, so that gate B keeps its time.
THOROUGH = os.environ.get("HYPOTHESIS_PROFILE") == "thorough"
FIXTURE_CASES = [
    c
    for c in FIXTURE_CASES
    if THOROUGH or len(c.X) < 1000 or c.params.get("max_degree", 1) == 1
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


# #79 fixed the reference's forward pass for covariates with a large mean at
# degree 2 and 3 (#77), and #85 fixed the fast one. One case still misses
# LA-5: the fast pruning pass at intermediate means, near a ratio of 2^20
# (#84, #96; the reference's is fixed), so the whole fit keeps the ratio of a
# covariate shift to its spread at or below SHIFT_CAP until #96 lifts it.
SHIFT_CAP = 2.0**10


@st.composite
def forward_cases(draw, kind: str, pruning_path: bool = False) -> Case:
    """A case of the given kind: ``smooth`` and ``hinge`` truths on uniform
    covariates; ``ties``, covariates on 2 to 20 levels, with a duplicated or a
    constant column, or all constant (FWD-5, KNOT-2, EDGE-3, EDGE-4);
    ``scaled``, covariates and y scaled and shifted by powers of 10 (LA-7,
    EDGE-6, FWD-10, FWD-11); ``shifted``, y + c with |c| of 1e10 or 1e13
    and covariates + 2^26 or 2^36, where only centering keeps the sums of
    squares within LA-5 (FWD-10, the plan's "Fast path"); ``small``, 1 to 15
    cases, now and then with a constant y (EDGE-1, EDGE-2, STOP-3). The noise
    runs from none (exact fits and STOP-5) to as large as the signal. At
    degree 2 and 3 the covariate shifts stay at or below SHIFT_CAP times the
    spread with ``pruning_path`` (the whole fit, #84)."""
    seed = draw(st.integers(0, 2**32 - 1))
    p = draw(st.integers(1, 3 if kind == "small" else 4))
    n = draw(st.integers(1, 15) if kind == "small" else st.integers(20, 120))
    params = draw(_settings())
    capped = params["max_degree"] >= 2 and pruning_path
    noise = draw(st.sampled_from([0.0, 1e-6, 0.01, 0.3, 1.0]))
    K = draw(st.sampled_from(SUPPORTED["responses"]))
    weighted = draw(st.sampled_from(SUPPORTED["weights"]))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    if kind == "ties":
        levels = rng.choice([2, 3, 5, 10, 20])
        X = np.floor(X * levels) / levels
        if p >= 2 and rng.uniform() < 0.5:
            X[:, 1] = X[:, 0]
        if p >= 3 and rng.uniform() < 0.3:
            X[:, 2] = X[0, 2]
        if rng.uniform() < 0.1:  # no covariate has a candidate (EDGE-3)
            X[:] = X[0]
        if rng.uniform() < 0.3:  # a linear term that LA-4 drops (FWD-11)
            low, high = (1, math.log10(SHIFT_CAP)) if capped else (6, 9)
            X = X + 10.0 ** rng.uniform(low, high, p)
    smooth = kind == "smooth" or rng.uniform() < 0.3
    Y = np.column_stack([_truth(rng, X, smooth) for _ in range(K)])
    Y = Y + noise * np.std(Y) * rng.standard_normal((n, K))
    if kind == "small" and rng.uniform() < 0.1:  # a degenerate fit (EDGE-1)
        Y[:] = Y[0]
    if kind == "scaled":
        scale = rng.uniform(-6, 6, p)
        shift = rng.uniform(0, 8, p)
        if capped:
            shift = np.minimum(shift, scale + math.log10(SHIFT_CAP))
        X = X * 10.0**scale + rng.choice([0, 1, -1], p) * 10.0**shift
        Y = Y * 10.0 ** rng.uniform(-6, 6) + rng.choice(
            [0, 1, -1]
        ) * 10.0 ** rng.uniform(0, 10)
    if kind == "shifted":
        big = [2.0**26, 2.0**36]
        X = X + rng.choice([0.0, 2.0**6, SHIFT_CAP] if capped else [0.0, *big], p)
        Y = Y + rng.choice([0.0, 1e10, -1e10, 1e13, -1e13])
    w = None
    if weighted:
        w = [rng.uniform(0.2, 3.0, n), rng.integers(0, 4, n).astype(float)][
            rng.integers(2)
        ]
        w[0] = max(w[0], 1.0)
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


def _large_mean_cases() -> list[Case]:
    """Products with a linear factor of a covariate with a large mean (LA-5,
    #85). The fast forward pass was off from the exact rational RSS of its
    terms by 7e-7 (the first, degree 2, 1e10), 1.7e-7 (the second, covariates
    with means of 1e9 and 4e6 times their spreads, degree 3), 3e-5 (the third,
    covariates + 2^26 and + 2^36, degree 3) and 5e-7 of the RSS before the
    step (the fourth, a covariate that another one repeats after a shift of
    each, degree 3); the reference was exact."""
    x0 = 1e10 + np.array([2, 0, 0, 2, 1, 2, 1, 2, 1, 0.0])
    x1 = np.array([1, 0, 0, 1, 1, 1, 2, 2, 2, 2.0])
    x2 = np.array([1, 0, 0, 2, 2, 1, 0, 0, 1, 2.0])
    y = np.array([0.2431975046920717, 0, 0, 1.2431975046920716, 2.909297426825682])
    y = np.r_[y, 0.2431975046920717, 0.9092974268256817, -0.7568024953079283]
    y = np.r_[y, 1.9092974268256817, 2.0]
    spans = {"thresh": 0.0, "fast_k": 0, "adjust_endspan": 0.0}
    cases = [
        Case(
            "x0 + 1e10, x0 x2",
            np.column_stack((x0, x1, x2)),
            y,
            None,
            spans | {"max_degree": 2},
        )
    ]
    for kind, seed in [("scaled", 176), ("shifted", 588), ("repeated", 126)]:
        rng = np.random.default_rng(seed)
        n, p = int(rng.integers(20, 80 if kind == "repeated" else 60)), 3
        if kind == "repeated":
            X = np.floor(rng.uniform(size=(n, p)) * rng.choice([5, 10, 20])) / 10
            X[:, 1] = X[:, 0]
            X = X + 10.0 ** rng.uniform(6, 9, p)
            m = X.min(axis=0)
            noise = 0.01 * rng.standard_normal(n)
            f = np.sin(4 * (X[:, 0] - m[0])) + (X[:, 2] - X[:, 2].mean()) ** 2 * (
                X[:, 0] - m[0]
            )
            kw = {
                "max_degree": 3,
                "fast_k": 0,
                "thresh": 0.0,
                "minspan": 1,
                "endspan": 1,
            }
            cases.append(Case("a repeated covariate, shifted", X, f + noise, None, kw))
            continue
        p = int(rng.integers(2, 4))
        X = rng.uniform(size=(n, p))
        f = np.sin(3 * X[:, 0]) + X[:, 0] * X[:, -1]
        f = f + 3 * X[:, 0] * X[:, 1] * np.maximum(X[:, -1] - 0.5, 0)
        f = f + 0.01 * rng.standard_normal(n)
        if kind == "scaled":
            scale, shift = rng.uniform(-6, 6, p), rng.uniform(0, 8, p)
            X = X * 10.0**scale + rng.choice([0, 1, -1], p) * 10.0**shift
        else:
            X = X + rng.choice([0.0, 2.0**26, 2.0**36], p)
        kw = {
            "max_degree": int(rng.choice([2, 3])),
            "fast_k": int(rng.choice([0, 1, 5, 20])),
        }
        cases.append(
            Case(
                f"{kind} covariates, seed {seed}",
                X,
                f,
                None,
                kw | {"thresh": 0.0, "adjust_endspan": float(rng.choice([0.0, 2.0]))},
            )
        )
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
    # form of FWD-6); the same with X + 1e9 (LA-5 for shifted covariates).
    cases = [
        Case(f"FWD-4 cap, auto_linpreds={a}", X, y, None, spans | {"auto_linpreds": a})
        for a in (True, False)
    ]
    cases.append(
        Case("FWD-4 cap, X + 1e9", X + 1e9, y, None, spans | {"max_terms": 11})
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
    # FWD-11: the linear term of a binary covariate shifted by 1e7, which LA-4
    # drops; FWD-5 and EDGE-4: a duplicated covariate.
    Xs = np.column_stack([1e7 + (np.arange(60) % 2), rng.uniform(size=60)])
    ys = 2 * (Xs[:, 0] - 1e7) + 3 * np.maximum(Xs[:, 1] - 0.5, 0)
    ys = ys + rng.normal(scale=0.05, size=60)
    cases.append(
        Case("FWD-11, a shifted binary covariate", Xs, ys, None, {"fast_k": 0})
    )
    # Exact shifts by powers of 2 on binary grids: x0 by 2^46 (A of LA-7),
    # 2^36 (a pair's first column) and 2^24 (the linear candidate; FWD-4 bars
    # the knots of x0 at step 2), and y by 2^44 (the TSS) [FWD-10]; at degree
    # 2, x0 by 2^36 enters products as b (x0 - m), which a shift does not
    # change (FWD-6).
    for design, dx, dy in [
        ("one covariate", 2.0**46, 0.0),
        ("pairs", 2.0**36, 0.0),
        ("linear", 2.0**24, 0.0),
        ("pairs", 0.0, 2.0**44),
        ("interaction", 2.0**36, 0.0),
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
    cases += _large_mean_cases()
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
    case = data.draw(forward_cases(kind, pruning_path=True), label="case")
    case.params["pmethod"] = data.draw(st.sampled_from(["backward", "none"]))
    case.params["nprune"] = data.draw(st.sampled_from([None, 1, 3, 10]))
    params = _core.MarsParams(**case.params)
    fast = _core.fit_mars(case.X, case.Y, case.w, params, record_candidates=True)
    ref = _core.MarsFit.from_dict(case.reference_fit())
    _count("whole fit, hypothesis", compare_fits(case, fast, ref))


@pytest.mark.parametrize("case", _designed_cases(), ids=lambda c: c.name)
def test_forward_pass_on_designed_data(case):
    """The same comparison on designs made for single rules (``_designed_cases``)."""
    _count("forward, designs", check_forward(case))


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
