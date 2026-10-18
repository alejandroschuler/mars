"""Profiling of the fast code for T20 (VALIDATION_PLAN.md, "Profiling").

One case per run, in a fresh process with one thread (source dev/env.sh,
run under nice -n 15). It prints the wall time of a first fit and of a second
fit in the same process (the difference is the one-time start cost), the time
of each phase of ``fit_mars`` and, with ``--prof``, writes cProfile stats of a
second fit and prints the top functions by own time.

Cases: the plan's baseline at 10,000 cases (``base10k``), the small-n cells
of REPORT.md (``n250``, ``n1000``), and pruning at a large covariate mean
(``shift3``: degree 3, every covariate shifted by 1e6, 2,000 cases).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import make_data

CASES = {
    "base10k": {"n": 10000, "p": 10, "degree": 2, "max_terms": 21, "shift": 0.0},
    "n250": {"n": 250, "p": 10, "degree": 2, "max_terms": 21, "shift": 0.0},
    "n1000": {"n": 1000, "p": 10, "degree": 2, "max_terms": 21, "shift": 0.0},
    "shift3": {"n": 2000, "p": 10, "degree": 3, "max_terms": 21, "shift": 1e6},
    "plain3": {"n": 2000, "p": 10, "degree": 3, "max_terms": 21, "shift": 0.0},
    # 150 forward terms of degree 3 (thresh = 0), with and without the shift.
    "shift3big": {
        "n": 2000,
        "p": 10,
        "degree": 3,
        "max_terms": 151,
        "shift": 1e6,
        "thresh": 0.0,
    },
    "plain3big": {
        "n": 2000,
        "p": 10,
        "degree": 3,
        "max_terms": 151,
        "shift": 0.0,
        "thresh": 0.0,
    },
    # The pruning pass alone on 150 random terms of degree 1 to 3 (the cost
    # noted on PR #96), at a covariate shift of 1e6 and with none.
    "prune150s": {
        "n": 2000,
        "p": 10,
        "degree": 3,
        "max_terms": 150,
        "shift": 1e6,
        "prune": True,
    },
    "prune150": {
        "n": 2000,
        "p": 10,
        "degree": 3,
        "max_terms": 150,
        "shift": 0.0,
        "prune": True,
    },
}


def random_terms(X, M, degree, seed=20):
    """M terms: the intercept and M - 1 random products of 1 to ``degree``
    factors (x, (x - t)+ or (t - x)+, t a data value) on distinct covariates."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n, p = X.shape
    dirs = np.zeros((M, p), dtype=np.int64)
    cuts = np.zeros((M, p))
    for k in range(1, M):
        for j in rng.choice(p, size=rng.integers(1, degree + 1), replace=False):
            dirs[k, j] = rng.choice([1, -1, 2])
            if dirs[k, j] != 2:
                cuts[k, j] = X[rng.integers(n), j]
    return dirs, cuts


def prune_case(c, prof, top):
    import numpy as np

    from pymars import _core, _gcv, _linalg, _pruning, _terms

    X, y, _ = make_data(c["n"], c["p"])
    X = X + c["shift"]
    Xs = np.ldexp(X, _core._column_exponents(X)[None, :])  # EDGE-7
    dirs, cuts = random_terms(Xs, c["max_terms"], c["degree"])
    t0 = time.perf_counter()
    keep = _linalg.independent_terms(Xs, dirs, cuts)  # FWD-11
    print(f"  independent_terms {time.perf_counter() - t0:.3f} s, kept {keep.sum()}")
    dirs, cuts = dirs[keep], cuts[keep]
    B = _terms.basis_matrix(Xs, dirs, cuts)
    pen = _gcv.default_penalty(c["degree"])
    kw = {"X": Xs, "dirs": dirs, "cuts": cuts}

    def run():
        t0 = time.perf_counter()
        pr = _pruning.pruning_pass(B, y, None, penalty=pen, **kw)
        t1 = time.perf_counter()
        _pruning.final_fit(B, y, pr.selected, None, penalty=pen, **kw)
        return t1 - t0, time.perf_counter() - t1, len(pr.selected)

    for i in range(2):
        tp, tf, m = run()
        print(f"  run {i}: pruning_pass {tp:.3f} s, final_fit {tf:.3f} s, selected {m}")
    if prof:
        pr = cProfile.Profile()
        pr.enable()
        run()
        pr.disable()
        pr.dump_stats(prof)
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
        print(s.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--prof", help="write cProfile stats of the second fit here")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    c = CASES[a.case]
    if c.get("prune"):
        prune_case(c, a.prof, a.top)
        return

    t0 = time.perf_counter()
    from pymars import _core, _forward, _pruning, _terms

    t_import = time.perf_counter() - t0
    X, y, _ = make_data(c["n"], c["p"])
    X = X + c["shift"]
    params = _core.MarsParams(
        max_degree=c["degree"], max_terms=c["max_terms"], thresh=c.get("thresh", 1e-3)
    )

    phases: dict[str, float] = defaultdict(float)

    def timed(mod, name):
        f = getattr(mod, name)

        def g(*args, **kw):
            s = time.perf_counter()
            try:
                return f(*args, **kw)
            finally:
                phases[name] += time.perf_counter() - s

        setattr(mod, name, g)

    t0 = time.perf_counter()
    m = _core.fit_mars(X, y, None, params)
    t_first = time.perf_counter() - t0
    for mod, name in [
        (_forward, "forward_pass"),
        (_terms, "basis_matrix"),
        (_pruning, "pruning_pass"),
        (_pruning, "final_fit"),
    ]:
        timed(mod, name)
    t0 = time.perf_counter()
    m = _core.fit_mars(X, y, None, params)
    t_second = time.perf_counter() - t0
    print(
        f"{a.case}: import {t_import:.3f} s, first fit {t_first:.3f} s, "
        f"second fit {t_second:.3f} s; terms {len(m.selected)}/"
        f"{m.forward.kept.size} (forward rows {m.forward.dirs.shape[0]}, "
        f"code {m.forward.termination})"
    )
    print("  phases: " + ", ".join(f"{k} {v:.3f}" for k, v in phases.items()))
    if a.prof:
        pr = cProfile.Profile()
        pr.enable()
        _core.fit_mars(X, y, None, params)
        pr.disable()
        pr.dump_stats(a.prof)
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(a.top)
        print(s.getvalue())


if __name__ == "__main__":
    main()
