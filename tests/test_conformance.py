"""Conformance of the new fitting code with earth (VALIDATION_PLAN.md,
"Correctness against earth"; T07).

Each implementation in ``IMPLEMENTATIONS`` fits every earth fixture: the
datasets S01 to S20 in their modes, and the 200 draws of S15. The test
compares, by the plan's tolerance table: the forward steps (the terms that
each step adds, exactly) up to the first step where the choices differ, and
the RSS after each step within 1e-8 of the RSS before it (LA-5); the
termination code; the pruning record on the same forward basis (the fit's
own, or the implementation's pruning pass on earth's forward basis when the
forward paths differ), with the subsets exact down to the first size where
they differ; and the selected terms, coefficients, GCV, RSq, GRSq, fitted
values and predictions, by ``compare.compare_fit`` with kappa(B).

A differing choice is a near-tie, labeled ``tie``, when the two choices are
different candidates with columns that are not bitwise equal, and their RSS
values differ by less than 1e-7 of the RSS before the step (plan: Ties); two
subsets of one size, when their RSS values differ by at most 1e-7 of the
lower one (the threshold of OQ-2). The comparison of the structure stops at the first
differing choice. Every other difference must be listed in
``validation/differences.json`` with its label and rules;
``validation/DIFFERENCES.md`` explains the entries and the special cases of
the plan: integer weights against the repeated rows (W-1), non-integer
weights through the fixed basis only (OQ-3), earth's zeroing of small sums
of squares (PRUNE-4, PRUNE-8), and the least-squares passes only for the
binary responses. Adding an implementation is one line.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from reference import mars_ref as ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "fixtures"
for _path in (ROOT / "validation" / "harness", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import compare  # noqa: E402
import names_map  # noqa: E402
import new_adapter  # noqa: E402

NEAR_TIE = compare.NEAR_TIE_REL  # 1e-7 (plan: Ties)
ZEROED_BELOW = 1e-9  # earth reports 0 below about 1e-10 (bb26.4: up to 4.83e-11)
LABELS = ("rule", "bug", "quirk", "tie", "numeric")
PYMARS_NAMES = {earth: pymars for pymars, earth in names_map.EARTH_NAMES.items()}


# ---------------------------------------------------------------------------
# The implementations


class Reference:
    """tests/reference/mars_ref.py, the oracle."""

    slow_from_n = 1000  # its fits with 1,000 cases take seconds

    def fit(self, X, Y, w, params: dict) -> dict:
        return ref.fit_mars(X, Y, w, params, record_candidates=True)

    def prune(self, B, Y, w, *, penalty, pmethod, nprune) -> dict:
        rows = w > 0
        N = ref.weight_sum(w[rows])
        N = ref.snap(N, ref.weight_tol(N))
        return ref.prune(
            B[rows],
            Y[rows],
            w[rows],
            penalty=penalty,
            N=N,
            tau_N=ref.weight_tol(N),
            pmethod=pmethod,
            nprune=nprune,
        )

    def basis_matrix(self, X, dirs, cuts) -> np.ndarray:
        return ref.basis_matrix(X, dirs, cuts)


class Fast:
    """The fast code (``pymars._core.fit_mars``, T12)."""

    slow_from_n = math.inf

    def fit(self, X, Y, w, params: dict) -> dict:
        from pymars import _core

        mars = _core.fit_mars(
            X, Y, w, _core.MarsParams(**params), record_candidates=True
        )
        return mars.to_dict()

    def prune(self, B, Y, w, *, penalty, pmethod, nprune) -> dict:
        from pymars import _pruning

        passed = _pruning.pruning_pass(
            B, Y, w, penalty=penalty, pmethod=pmethod, nprune=nprune
        )
        final = _pruning.final_fit(B, Y, passed.selected, w, penalty=penalty)
        return {**passed._asdict(), **final._asdict()}

    def basis_matrix(self, X, dirs, cuts) -> np.ndarray:
        from pymars import _terms

        return _terms.basis_matrix(X, dirs, cuts)


IMPLEMENTATIONS: dict[str, Any] = {"reference": Reference()}
# T12 adds the fast code: "fast": Fast(),


# ---------------------------------------------------------------------------
# The cases


@dataclass
class Case:
    """One comparison: the data of a fixture, the parameters, and the earth
    result that the fit is compared with."""

    name: str
    X: np.ndarray
    Y: np.ndarray
    w: np.ndarray  # ones when the fixture has no weights
    weighted: bool
    params: dict
    earth: dict
    X_test: np.ndarray | None = None
    repeat: np.ndarray | None = None  # earth's rows as indices of ours
    forward: bool = True  # False: only the pruning pass on earth's basis
    glm: bool = False


def _read(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def pymars_params(earth_args: dict) -> dict:
    """earth's arguments as MarsParams fields (API-7); earth's span 0 is
    automatic, None in pymars."""
    return {
        PYMARS_NAMES[key]: None
        if key in ("minspan", "endspan") and value == 0
        else value
        for key, value in earth_args.items()
    }


def load_case(name: str) -> Case:
    """The comparison of the dataset fixture ``name`` (module docstring)."""
    fixture = _read(name)
    inputs, earth = fixture["inputs"], fixture["result"]
    X = np.asarray(inputs["X"], dtype=float)
    labels = np.asarray(inputs["y"], dtype=object).ravel()
    if isinstance(labels[0], str):  # a factor: one indicator per level (GLM-1)
        Y = np.column_stack([(labels == v).astype(float) for v in earth["levels"]])
    else:
        Y = np.asarray(inputs["y"], dtype=float).reshape(len(X), -1)
    weighted = inputs["weights"] is not None
    w = np.asarray(inputs["weights"], dtype=float) if weighted else np.ones(len(X))
    repeat, forward = None, True
    if weighted and np.all(w == np.round(w)):
        repeat = np.repeat(np.arange(len(w)), w.astype(int))
        earth = _read(re.sub(r"_(matched|defaults)", r"_repeated_\1", name))["result"]
    elif weighted:
        forward = False
    X_test = inputs["X_test"]
    return Case(
        name=name,
        X=X,
        Y=Y,
        w=w,
        weighted=weighted,
        params=pymars_params(fixture["earth_args"]),
        earth=earth,
        X_test=None if X_test is None else np.asarray(X_test, dtype=float),
        repeat=repeat,
        forward=forward,
        glm=earth.get("glm_coef") is not None,
    )


def s15_cases() -> list[tuple[dict, Case]]:
    """The 200 draws of S15 and their comparisons. The fixture keeps earth's
    forward terms and pruning records (pmethod "none", PRUNE-7) but not the
    data, which its generator makes again from the seeds."""
    from gen_fixtures import _s15_earth_args, scaled_matrix

    from validation.sims import dgps, seeds

    diagnostics = dgps.load_diagnostics()
    out = []
    for draw in _read("s15_draws")["draws"]:
        rng, _ = seeds.train_test_rngs(
            f"S15_{draw['dgp']}_{draw['noise']}", draw["rep"]
        )
        X, y, _ = dgps.generate(
            dgps.REGISTRY[draw["dgp"]], rng, draw["n"], draw["noise"], diagnostics
        )
        X, _ = scaled_matrix(X)
        earth = {
            k: draw[k] for k in ("dirs", "cuts", "rss_per_subset", "gcv_per_subset")
        }
        case = Case(
            name=f"s15_draws/draw{draw['rep']:03d}",
            X=X,
            Y=y.reshape(-1, 1),
            w=np.ones(len(X)),
            weighted=False,
            params=pymars_params(_s15_earth_args(draw["degree"], draw["mode_family"])),
            earth=earth,
        )
        out.append((draw, case))
    return out


DATASETS = sorted(  # glob ignores case on Windows, so check the name too
    path.stem for path in FIXTURES.glob("S*.json") if path.name.startswith("S")
)


def _case_params() -> list:
    out = []
    for name in DATASETS:
        n = len(_read(name)["inputs"]["X"])
        for label, implementation in IMPLEMENTATIONS.items():
            slow = [pytest.mark.slow] if n >= implementation.slow_from_n else []
            out.append(pytest.param(label, name, marks=slow, id=f"{label}-{name}"))
    return out


# ---------------------------------------------------------------------------
# The comparison


def _wrss(B: np.ndarray, Y: np.ndarray, w: np.ndarray) -> float:
    """RSS(U) of LA-1 for the columns of B, by an SVD solve that is neither
    implementation's."""
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(B * sw, Y * sw, rcond=None)
    return float(np.sum((Y * sw - (B * sw) @ coef) ** 2))


def _kappa(B: np.ndarray, w: np.ndarray) -> float:
    return compare.condition_number(B * np.sqrt(w)[:, None])


def _label(dirs_row, cuts_row) -> str:
    """A term as TERM-5 writes it, with x numbered from 0."""
    parts = []
    for j, code in enumerate(dirs_row):
        c = f"{cuts_row[j]:.6g}"
        parts += {0: [], 1: [f"h(x{j}-{c})"], -1: [f"h({c}-x{j})"], 2: [f"x{j}"]}[
            int(code)
        ]
    return "*".join(parts) or "(Intercept)"


def _terms(dirs, cuts, rows) -> list[tuple]:
    """What a step adds: the codes of each row, and its cuts where the code
    is 1 or -1 (TERM-2)."""
    return [
        (
            tuple(int(v) for v in dirs[r]),
            tuple(
                float(c) if abs(v) == 1 else 0.0
                for v, c in zip(dirs[r], cuts[r], strict=True)
            ),
        )
        for r in rows
    ]


def _sources(dirs, cuts, rows) -> dict:
    """The (parent, covariate) pairs that the terms of a step can come from,
    each with the step's knot there, or None for a linear term (FWD-6). The
    parent is the first row without the covariate (TERM-6), and a pair's
    covariate is the column where its two rows differ."""
    first = rows[0]
    columns = np.flatnonzero(dirs[first])
    if len(rows) == 2:
        columns = np.flatnonzero(dirs[rows[0]] != dirs[rows[1]])
    out = {}
    for v in columns:
        parent_dirs, parent_cuts = dirs[first].copy(), cuts[first].copy()
        parent_dirs[v], parent_cuts[v] = 0, 0.0
        hinge = {int(dirs[r, v]) for r in rows} <= {1, -1}
        key = (*_terms([parent_dirs], [parent_cuts], [0]), int(v))
        out[key] = float(cuts[first, v]) if hinge else None
    return out


def _near_tie(a: tuple, b: tuple, band: float) -> bool:
    """Whether two differing steps are a near-tie: different candidates
    (another parent or covariate, or hinges at another knot; not another
    kind or order of the terms of one candidate, FWD-3, FWD-6, LA-7) whose
    new columns are not bitwise equal (FWD-5 breaks those ties by its
    order, as earth does, bb18.1) and whose RSS values are within ``band``.
    ``a`` and ``b`` are (dirs, cuts, rows, B, rss) of each side."""
    (dirs_a, cuts_a, rows_a, B_a, rss_a), (dirs_b, cuts_b, rows_b, B_b, rss_b) = a, b
    if not rows_a or not rows_b:  # FWD-11 dropped every term of a step
        return False
    src_a, src_b = _sources(dirs_a, cuts_a, rows_a), _sources(dirs_b, cuts_b, rows_b)
    same = any(
        None in (src_a[k], src_b[k]) or src_a[k] == src_b[k]
        for k in src_a.keys() & src_b.keys()
    )
    equal = len(rows_a) == len(rows_b) and np.array_equal(
        B_a[:, rows_a], B_b[:, rows_b]
    )
    return not same and not equal and abs(rss_a - rss_b) < band


def _forward(impl: Any, case: Case, ours: dict) -> tuple[list, bool]:
    """The forward steps and the RSS path; also whether the whole forward
    path matched (module docstring)."""
    earth = case.earth
    dirs_a, cuts_a = np.asarray(ours["dirs"]), np.asarray(ours["cuts"])
    dirs_b, cuts_b = np.asarray(earth["dirs"]), np.asarray(earth["cuts"], dtype=float)
    steps_a = [step["rows"] for step in ours["forward_steps"]]
    steps_b = compare._dirs_row_groups(dirs_b, cuts_b)
    B_a = impl.basis_matrix(case.X, dirs_a.astype(np.int8), cuts_a)
    B_b = impl.basis_matrix(case.X, dirs_b.astype(np.int8), cuts_b)
    rss, earth_rss = ours["forward_rss"], earth.get("fwd_rss")  # per step, per term
    diffs = []
    for s in range(1, min(len(steps_a), len(steps_b)) + 1):
        rows_a, rows_b = steps_a[s - 1], steps_b[s - 1]
        before = rss[s - 1]
        if _terms(dirs_a, cuts_a, rows_a) != _terms(dirs_b, cuts_b, rows_b):
            done_a = [0, *(r for rows in steps_a[: s - 1] for r in rows)]
            done_b = [0, *(r for rows in steps_b[: s - 1] for r in rows)]
            rss_a = _wrss(B_a[:, done_a + rows_a], case.Y, case.w)
            rss_b = _wrss(B_b[:, done_b + rows_b], case.Y, case.w)
            tie = _near_tie(
                (dirs_a, cuts_a, rows_a, B_a, rss_a),
                (dirs_b, cuts_b, rows_b, B_b, rss_b),
                NEAR_TIE * before,
            )
            diffs.append(
                compare.Difference(
                    field="forward",
                    step=s,
                    a=[_label(dirs_a[r], cuts_a[r]) for r in rows_a],
                    b=[_label(dirs_b[r], cuts_b[r]) for r in rows_b],
                    candidate_rss=(rss_a, rss_b),
                    label="tie" if tie else None,
                    detail=f"the RSS before the step is {before!r}",
                )
            )
            return diffs, False
        if earth_rss is not None:
            columns = [0, *(r for rows in steps_a[:s] for r in rows)]
            kappa = _kappa(B_a[:, columns], case.w)
            tol, numeric = compare._kappa_scale(
                compare.FWD_RSS_REL, kappa, compare.KAPPA_RSS_LIMIT
            )
            theirs = earth_rss[max(rows_b)]
            if abs(rss[s] - theirs) > tol * before:
                diffs.append(
                    compare.Difference(
                        field="forward_rss",
                        step=s,
                        a=rss[s],
                        b=theirs,
                        metric=abs(rss[s] - theirs) / before,
                        tolerance=tol,
                        label="numeric" if numeric else None,
                        detail="relative to the RSS before the step (LA-5)",
                    )
                )
    if len(steps_a) != len(steps_b):
        diffs.append(
            compare.Difference(
                field="forward",
                step=min(len(steps_a), len(steps_b)) + 1,
                a=f"{len(steps_a)} steps, termination {ours['termcond']}",
                b=f"{len(steps_b)} steps, termination {earth.get('termcond')}",
                detail="one forward pass stopped before the other",
            )
        )
        return diffs, False
    if "termcond" in earth and ours["termcond"] != earth["termcond"]:
        diffs.append(
            compare.Difference(
                field="termination", a=ours["termcond"], b=earth["termcond"]
            )
        )
    return diffs, True


def _fixed_basis(impl: Any, case: Case) -> dict:
    """The implementation's pruning pass and final fit on earth's forward
    basis (PRUNE-2 to PRUNE-8), in the common schema."""
    dirs = np.asarray(case.earth["dirs"]).astype(np.int8)
    cuts = np.asarray(case.earth["cuts"], dtype=float)
    B = impl.basis_matrix(case.X, dirs, cuts)
    params = ref.as_params(case.params)
    penalty = params.resolved_penalty()
    pruned = impl.prune(
        B, case.Y, case.w, penalty=penalty, pmethod=params.pmethod, nprune=params.nprune
    )
    subsets = np.asarray(pruned["subsets"], dtype=bool)
    return {
        "dirs": dirs,
        "cuts": cuts,
        "selected_terms": [int(t) + 1 for t in pruned["selected"]],
        "prune_terms": [list(np.flatnonzero(row) + 1) for row in subsets],
        "rss_per_subset": list(pruned["rss_per_size"]),
        "gcv_per_subset": list(pruned["gcv_per_size"]),
        "coef": np.asarray(pruned["coef"], dtype=float).reshape(-1, case.Y.shape[1]),
        **{k: float(pruned[k]) for k in ("rss", "gcv", "rsq", "grsq")},
        "penalty": penalty,
    }


def _pruning(case: Case, ours: dict, B: np.ndarray, zeroed: list) -> tuple[list, int]:
    """The subsets and the per-size values; the first size where the
    subsets differ, else 0. Values that earth reports as 0 (module
    docstring) go to ``zeroed``."""
    earth = case.earth
    Mf = len(ours["rss_per_subset"])
    if len(earth["rss_per_subset"]) != Mf:
        n = len(earth["rss_per_subset"])
        return [compare.Difference(field="pruning", a=Mf, b=n, detail="sizes")], Mf
    diffs, stop = [], 0
    if earth.get("prune_terms") is not None:
        for m in range(Mf, 0, -1):
            a = sorted(t for t in ours["prune_terms"][m - 1] if t)
            b = sorted(int(t) for t in earth["prune_terms"][m - 1] if t)
            if a != b:
                rss_a = _wrss(B[:, [t - 1 for t in a]], case.Y, case.w)
                rss_b = _wrss(B[:, [t - 1 for t in b]], case.Y, case.w)
                tie = abs(rss_a - rss_b) <= NEAR_TIE * min(rss_a, rss_b)
                diffs.append(
                    compare.Difference(
                        field="pruning",
                        step=m,
                        a=a,
                        b=b,
                        candidate_rss=(rss_a, rss_b),
                        label="tie" if tie else None,
                        detail="the subsets of this size differ",
                    )
                )
                stop = m
                break
    for name in ("rss_per_subset", "gcv_per_subset"):
        for m in range(Mf, stop, -1):
            a, b = float(ours[name][m - 1]), float(earth[name][m - 1])
            if b == 0.0 and 0.0 < a < ZEROED_BELOW:
                zeroed.append(f"{name}[{m}]")
            elif a != b:
                metric = abs(a - b) / abs(b) if math.isfinite(b) and b else math.inf
                if metric > compare.RSS_PER_SUBSET_REL:
                    diffs.append(
                        compare.Difference(
                            field=name,
                            step=m,
                            a=a,
                            b=b,
                            metric=metric,
                            tolerance=compare.RSS_PER_SUBSET_REL,
                        )
                    )
                    break
    return diffs, stop


def _earth_statistics(case: Case, size: int, penalty: float) -> dict:
    """earth's rss, gcv, rsq and grsq from its residuals (GCV-2, GCV-5,
    GCV-6), with its own number of cases."""
    Y = case.Y if case.repeat is None else case.Y[case.repeat]
    w = case.w if case.repeat is None else np.ones(len(case.repeat))
    rss = float(
        w
        @ (Y - np.asarray(case.earth["fitted"], dtype=float)) ** 2
        @ np.ones(Y.shape[1])
    )
    N = float(np.sum(w))
    tss = float(w @ (Y - w @ Y / N) ** 2 @ np.ones(Y.shape[1]))
    tau = ref.weight_tol(N)
    return {
        "gcv": ref.gcv(rss, size, penalty, N, tau),
        "rsq": ref.rsq(rss, tss),
        "grsq": ref.grsq(rss, size, tss, penalty, N, tau),
    }


def _sd(case: Case) -> float:
    """sd(y) for the fitted-value tolerance: the smallest weighted standard
    deviation of the responses (divisor N)."""
    mean = case.w @ case.Y / np.sum(case.w)
    return float(np.min(np.sqrt(case.w @ (case.Y - mean) ** 2 / np.sum(case.w))))


def _final(impl: Any, case: Case, ours: dict, B: np.ndarray, zeroed: list) -> list:
    """The selected terms and the final model, by compare.compare_fit."""
    earth = case.earth
    keys = ("selected_terms", "coef", "gcv", "rsq", "grsq")
    mine, theirs = {k: ours[k] for k in keys}, {k: earth[k] for k in keys}
    selected = [t - 1 for t in ours["selected_terms"]]
    reported = [
        k for k in ("rss", "gcv") if earth[k] == 0.0 and 0 < ours[k] < ZEROED_BELOW
    ]
    reported += [k for k in ("rsq", "grsq") if math.isnan(earth[k])]
    if reported:
        zeroed += reported
        theirs.update(_earth_statistics(case, len(selected), ours["penalty"]))
    if not case.glm:  # earth's fitted values of a GLM fit are probabilities
        coef = np.asarray(ours["coef"], dtype=float)
        fitted = B[:, selected] @ coef
        mine["fitted"] = fitted if case.repeat is None else fitted[case.repeat]
        theirs["fitted"] = earth["fitted"]
        if case.X_test is not None:
            dirs = np.asarray(ours["dirs"]).astype(np.int8)[selected]
            cuts = np.asarray(ours["cuts"], dtype=float)[selected]
            mine["pred_test"] = impl.basis_matrix(case.X_test, dirs, cuts) @ coef
            theirs["pred_test"] = earth["pred_test"]
    kappa = _kappa(B[:, selected], case.w)
    return compare.compare_fit(mine, theirs, kappa=kappa, sd_y=_sd(case))


def conform(impl: Any, case: Case, fit: dict | None = None) -> list:
    """Every difference between the implementation's fit (``fit``, or a new
    one) and earth's on ``case`` (module docstring)."""
    earth = case.earth
    if case.forward:
        if fit is None:
            w = case.w if case.weighted else None
            fit = impl.fit(case.X, case.Y, w, case.params)
        ours = new_adapter.mars_fit_to_common(fit)
        if ours["termcond"] == ref.DEGENERATE:  # EDGE-1, GCV-7
            keys = ("selected_terms", "coef")
            stats = [ours[k] for k in ("gcv", "rsq", "grsq", "termcond")]
            diffs = compare.compare_fit(
                {k: ours[k] for k in keys}, {k: earth[k] for k in keys}
            )
            if stats != [math.inf, 0.0, 0.0, 0]:  # not EDGE-1's values
                diffs.append(compare.Difference(field="edge_1", a=stats, b=None))
            earths = [earth[k] for k in ("gcv", "rsq", "grsq", "termcond")]
            if earths != stats:
                diffs.append(compare.Difference(field="degenerate", a=stats, b=earths))
            return diffs
        diffs, matched = _forward(impl, case, ours)
    else:
        diffs, matched = [], False
    if not matched:
        ours = _fixed_basis(impl, case)
    dirs = np.asarray(ours["dirs"]).astype(np.int8)
    B = impl.basis_matrix(case.X, dirs, np.asarray(ours["cuts"], dtype=float))
    zeroed: list = []
    more, stop = _pruning(case, ours, B, zeroed)
    diffs += more
    same = sorted(ours["selected_terms"]) == sorted(earth.get("selected_terms", []))
    if "selected_terms" in earth and (same or not stop):
        diffs += _final(impl, case, ours, B, zeroed)
    if zeroed:
        diffs.append(
            compare.Difference(
                field="earth_zeroing",
                a=zeroed,
                b=None,
                detail="earth reports these values as 0 or NaN",
            )
        )
    return diffs


# ---------------------------------------------------------------------------
# The expected differences


def expected_differences() -> list[dict]:
    path = ROOT / "validation" / "differences.json"
    return json.loads(path.read_text(encoding="utf-8"))["differences"]


def check(differences: list, implementation: str, case: str, entries: list) -> None:
    """Fail on a difference that no entry labels, unless the comparison
    labeled it ``tie``; on an entry for this case and implementation that
    did not occur (a ``tie`` entry may be absent); and on a ``bug`` entry.
    An entry's recorded choices must be those observed, and its candidate
    RSS values too, to a relative 1e-8."""
    mine = [
        e
        for e in entries
        if case in e["cases"]
        and implementation in e.get("implementations", [implementation])
    ]
    used, unlabeled = set(), []
    for d in differences:
        found = [
            e for e in mine if e["field"] == d.field and e.get("step") in (None, d.step)
        ]
        for e in found:
            used.add(e["id"])
            assert (e.get("pymars", d.a), e.get("earth", d.b)) == (d.a, d.b), e["id"]
            assert d.label != "tie" or e["label"] == "tie", f"{e['id']} is a near-tie"
            if "candidate_rss" in e:
                np.testing.assert_allclose(
                    d.candidate_rss, e["candidate_rss"], rtol=1e-8
                )
        if not found and d.label != "tie":
            unlabeled.append(d)
    stale = [e["id"] for e in mine if e["id"] not in used and e["label"] != "tie"]
    problems = [
        f"{what}: {items}"
        for what, items in (
            ("differences that validation/differences.json does not list", unlabeled),
            ("entries that did not occur", stale),
            ("bugs", [e["id"] for e in mine if e["label"] == "bug"]),
        )
        if items
    ]
    if problems:
        pytest.fail(f"{implementation} on {case}: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# The tests


@pytest.mark.parametrize(("implementation", "name"), _case_params())
def test_the_fit_conforms_to_earth(implementation, name):
    differences = conform(IMPLEMENTATIONS[implementation], load_case(name))
    check(differences, implementation, name, expected_differences())


def _linear_options(fit: dict, X: np.ndarray) -> list[int]:
    """The steps whose one term is the hinge at the smallest value of its
    covariate over all cases, the linear option with auto_linpreds=False
    (FWD-6)."""
    forward = fit["forward"]
    dirs, cuts, parent = forward["dirs"], forward["cuts"], forward["parent"]
    out = []
    for s in range(1, len(forward["rss"])):
        (terms,) = np.nonzero(forward["step"] == s)
        if len(terms) == 1:
            k = terms[0]
            v = np.flatnonzero(dirs[k] != dirs[parent[k]])[0]
            if dirs[k, v] == 1 and cuts[k, v] == X[:, v].min():
                out.append(s)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_the_s15_draws_conform_to_earth(implementation):
    # plan: Ties: at most 5 percent of the fits may stop at a near-tie; this
    # counts every fit with a near-tie in the implementation's own log up to
    # its first difference, whether or not the two choices differ there.
    impl, entries = IMPLEMENTATIONS[implementation], expected_differences()
    failures, near_ties = [], 0
    for draw, case in s15_cases():
        fit = impl.fit(case.X, case.Y, None, case.params)
        differences = conform(impl, case, fit)
        try:
            check(differences, implementation, case.name, entries)
        except pytest.fail.Exception as error:
            failures.append(str(error))
        for e in entries:  # FWD-11: the quirk needs earth's rank fix and a
            # linear option before the step (the note in the body of #44)
            if case.name in e["cases"] and "FWD-11" in e["rules"]:
                linear = _linear_options(fit, case.X)
                assert draw["rank_fix"] and not case.params.get(
                    "auto_linpreds", True
                ), e["id"]
                assert linear and linear[0] < e["step"], e["id"]
        log, rss = fit["forward"]["candidates"], fit["forward"]["rss"]
        steps = [d.step for d in differences if d.field == "forward"]
        first = min(steps, default=len(rss) - 1)
        gaps = (log["second_rss"] - log["best_rss"])[:first] / rss[:first]
        near_ties += bool(np.any(gaps < NEAR_TIE))
    assert not failures, "\n".join(failures)
    assert near_ties <= 0.05 * 200, f"{near_ties} of 200 fits stop at a near-tie"


def test_the_list_of_differences_is_complete_and_documented():
    # Every entry names existing cases, a label of the plan's triage, spec
    # rules that exist, and a note, and DIFFERENCES.md describes it by its id.
    # (An entry with a wrong field or step never matches, so it fails above.)
    spec = (ROOT / "docs" / "algorithm.md").read_text(encoding="utf-8")
    rules = set(re.findall(r"\*\*([A-Z]+-\d+)\*\*", spec))
    report = (ROOT / "validation" / "DIFFERENCES.md").read_text(encoding="utf-8")
    draws = {f"s15_draws/draw{d['rep']:03d}" for d in _read("s15_draws")["draws"]}
    entries = expected_differences()
    assert len({e["id"] for e in entries}) == len(entries)
    for e in entries:
        assert set(e["cases"]) <= set(DATASETS) | draws, e["id"]
        assert e["label"] in LABELS, e["id"]
        assert e["rules"] and set(e["rules"]) <= rules, e["id"]
        assert e["note"] and f"| {e['id']} |" in report, e["id"]


@functools.cache
def _reference_fit(name: str) -> dict:
    case = load_case(name)
    return ref.fit_mars(case.X, case.Y, None, case.params, record_candidates=True)


def _set(path: str, index, new):
    """A change to a fit dict: the entries ``index`` of the field ``path``
    (such as ``forward.rss``) become ``new(values)``."""

    def change(fit: dict) -> None:
        *outer, last = path.split(".")
        record = functools.reduce(lambda d, k: d[k], outer, fit)
        values = np.array(record[last])
        values[index] = new(values)
        record[last] = values

    return change


def _other_term(row):
    """A subset with its last term traded for the first term that it lacks."""
    row = row.copy()
    row[np.flatnonzero(row)[-1]], row[np.flatnonzero(~row)[0]] = False, True
    return row


def _move_pair(columns: int) -> list:
    """The pair of step 1 moved ``columns`` covariates to the right."""
    return [
        _set(f"forward.{k}", slice(1, 3), lambda v: np.roll(v[1:3], columns, axis=1))
        for k in ("dirs", "cuts")
    ]


S01, S10 = "S01_matched_d1", "S10_matched_d1"


@pytest.mark.parametrize(
    ("name", "changes", "found"),
    [
        # the knot of the pair of step 2 moved by 0.25
        (S01, [_set("forward.cuts", (slice(3, 5), 0), lambda c: c[3:5, 0] + 0.25)],
         [("forward", 2, None)]),
        # the pair of step 1 in the order (-1, +1) (FWD-6)
        (S01, [_set("forward.dirs", slice(1, 3), lambda d: d[[2, 1]])],
         [("forward", 1, None)]),
        # step 1 on the duplicate x1, a column bitwise equal to x0 (FWD-5)
        (S10, _move_pair(1), [("forward", 1, None)]),
        # step 1 on x2, x0 plus noise of 1e-9: a near-tie
        (S10, _move_pair(2), [("forward", 1, "tie")]),
        # the RSS after step 2 off by 2e-8, and by 5e-9, of the RSS before it
        (S01, [_set("forward.rss", 2, lambda r: r[2] + 2e-8 * r[1])],
         [("forward_rss", 2, None)]),
        (S01, [_set("forward.rss", 2, lambda r: r[2] + 5e-9 * r[1])], []),
        (S01, [_set("forward.termination", (), lambda t: 5)],
         [("termination", None, None)]),
        # the subset of size 3 with another term
        (S01, [_set("pruning.subsets", 2, lambda s: _other_term(s[2]))],
         [("pruning", 3, None)]),
        # rss.per.subset of size 5 off by a relative 2e-8, and by 5e-9
        (S01, [_set("pruning.rss_per_size", 4, lambda r: r[4] * (1 + 2e-8))],
         [("rss_per_subset", 5, None)]),
        (S01, [_set("pruning.rss_per_size", 4, lambda r: r[4] * (1 + 5e-9))], []),
        # the final model: a coefficient off by a relative 1e-4, the
        # intercept by 2e-8 sd(y), gcv by a relative 2e-8, rsq by 2e-8
        (S01, [_set("coef", (1, 0), lambda c: c[1, 0] * (1 + 1e-4))],
         [("coef", None, None), ("fitted", None, None), ("pred_test", None, None)]),
        (S01, [_set("coef", (0, 0), lambda c: c[0, 0] + 2e-8 * _sd(load_case(S01)))],
         [("fitted", None, None), ("pred_test", None, None)]),
        (S01, [_set("gcv", (), lambda g: g * (1 + 2e-8))], [("gcv", None, None)]),
        (S01, [_set("rsq", (), lambda r: r + 2e-8)], [("rsq", None, None)]),
    ],
)  # fmt: skip
def test_a_new_difference_fails_until_it_is_labeled(name, changes, found):
    # The suite fails on a difference that the list does not label, and
    # passes once an entry labels it (brief T07, item 3); a near-tie passes
    # unlisted, but not under an entry of another label. Each case plants
    # changes in the reference's fit of a fixture that matches earth, at or
    # just past a tolerance of the plan's table.
    fit = copy.deepcopy(_reference_fit(name))
    for change in changes:
        change(fit)

    class Planted(Reference):
        def fit(self, X, Y, w, params):
            return fit

    differences = conform(Planted(), load_case(name))
    assert [(d.field, d.step, d.label) for d in differences] == found
    entries = [
        {"id": f"X{i}", "cases": [name], "field": f, "step": s, "label": "rule"}
        for i, (f, s, _) in enumerate(found)
    ]
    if found and found[0][2] == "tie":
        check(differences, "reference", name, [])
        with pytest.raises(AssertionError, match="near-tie"):
            check(differences, "reference", name, entries)
        return
    if not found:
        return
    with pytest.raises(pytest.fail.Exception, match="does not list"):
        check(differences, "reference", name, [])
    check(differences, "reference", name, entries)
    with pytest.raises(pytest.fail.Exception, match="did not occur"):
        check([], "reference", name, entries)
    with pytest.raises(pytest.fail.Exception, match="bugs"):
        check(differences, "reference", name, [{**e, "label": "bug"} for e in entries])
    with pytest.raises(AssertionError):
        check(differences, "reference", name, [{**entries[0], "pymars": "?"}])
