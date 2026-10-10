"""One timed fit in a fresh process (VALIDATION_PLAN.md, "Benchmark design").

Run by ``run.py`` for ``pymars_fit`` (``pymars.fit_mars``), ``pymars_est``
(``pymars.EarthRegressor``) and ``legacy`` (mars-earth 1.0.4, in its own venv,
with ``-I``). The file uses only the standard library and numpy, so the legacy
venv can run it. It prints one JSON object on stdout.

Modes: ``time`` (wall time of the fit alone, imports and data loading
excluded), ``tracemalloc`` (peak Python-level allocation during the fit; it
slows the fit, so ``run.py`` never uses its time), ``profile`` (cProfile stats
of the fit written to ``--profile-out``, for T20).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--data", required=True, help=".npz with X, y and w")
    ap.add_argument("--degree", type=int, required=True)
    ap.add_argument("--max-terms", type=int, required=True)
    ap.add_argument("--weights", action="store_true")
    ap.add_argument(
        "--mode", default="time", choices=["time", "tracemalloc", "profile"]
    )
    ap.add_argument("--profile-out")
    a = ap.parse_args()

    import numpy as np

    warnings.filterwarnings("ignore")
    d = np.load(a.data)
    X, y = np.ascontiguousarray(d["X"]), np.ascontiguousarray(d["y"])
    w = np.ascontiguousarray(d["w"]) if a.weights else None

    if a.system == "pymars_fit":
        from pymars._core import MarsParams, fit_mars

        params = MarsParams(max_degree=a.degree, max_terms=a.max_terms)

        def fit():
            return fit_mars(X, y, w, params)

        def sizes(m):
            return len(m.selected), int(m.forward.kept.size)

    elif a.system == "pymars_est":
        import pymars

        def fit():
            est = pymars.EarthRegressor(max_degree=a.degree, max_terms=a.max_terms)
            return est.fit(X, y, sample_weight=w)

        def sizes(m):
            return len(m.mars_.selected), int(m.mars_.forward.kept.size)

    elif a.system == "legacy":
        if a.weights:
            sys.exit("the legacy code takes no weights")
        import pymars

        def fit():
            est = pymars.Earth(max_degree=a.degree, max_terms=a.max_terms)
            return est.fit(X, y)

        def sizes(m):
            return len(m.basis_), len(m.record_.fwd_basis_[-1])

    else:
        sys.exit(f"unknown system {a.system}")

    out: dict = {"system": a.system, "mode": a.mode}
    if a.mode == "time":
        t0 = time.perf_counter()
        m = fit()
        out["seconds"] = time.perf_counter() - t0
    elif a.mode == "tracemalloc":
        import tracemalloc

        tracemalloc.start()
        m = fit()
        out["py_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        m = fit()
        pr.disable()
        pr.dump_stats(a.profile_out)
    out["terms"], out["forward_terms"] = sizes(m)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
