"""Fit pymars and R earth on identical data and compare the fits.

Usage: python compare_earth.py [dataset ...]
Writes data/ and out/ next to this file and prints one block per case.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import pymars as earth
from pymars._basis import (
    ConstantBasisFunction,
    HingeBasisFunction,
    LinearBasisFunction,
)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
(HERE / "data").mkdir(exist_ok=True)
(HERE / "out").mkdir(exist_ok=True)


def make(name, n, seed):
    rng = np.random.default_rng(seed)
    if name == "hinge1d":
        X = rng.uniform(0, 1, (n, 1))
        f = 2 * np.maximum(0, X[:, 0] - 0.3) - 3 * np.maximum(0, X[:, 0] - 0.7)
        sigma = 0.1
    elif name == "friedman1":
        X = rng.uniform(0, 1, (n, 5))
        f = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
        sigma = 1.0
    elif name == "additive3":
        X = rng.uniform(-1, 1, (n, 3))
        f = np.maximum(0, X[:, 0]) + np.abs(X[:, 1]) - 0.5 * X[:, 2]
        sigma = 0.2
    else:
        raise KeyError(name)
    y = f + sigma * rng.standard_normal(n)
    return X, y, f


def write_csv(path, X, y):
    cols = [f"x{j}" for j in range(X.shape[1])] + ["y"]
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=",".join(cols),
               comments="", fmt="%.17g")


# Configurations. "matched" makes the two algorithms as close as their options
# allow: hinge-only terms, no fast MARS, no RSq stopping rule, every case a
# candidate knot, and pymars penalty = earth penalty / 2 so that both GCVs count
# the same effective parameters for a hinge-only model with an intercept.
CONFIGS = {
    "matched_d1": (
        dict(degree=1, penalty=2, nk=21, thresh=0, minspan=1, endspan=1,
             fast_k=0, Auto_linpreds=False, pmethod="backward"),
        dict(max_degree=1, penalty=1.0, max_terms=21, minspan=1, endspan=0,
             allow_linear=False),
    ),
    "matched_d1_fwd": (
        dict(degree=1, penalty=2, nk=21, thresh=0, minspan=1, endspan=1,
             fast_k=0, Auto_linpreds=False, pmethod="none"),
        None,  # forward pass only; compared against pymars record_.fwd_basis_
    ),
    "matched_d2": (
        dict(degree=2, penalty=3, nk=21, thresh=0, minspan=1, endspan=1,
             fast_k=0, Auto_linpreds=False, Adjust_endspan=1, pmethod="backward"),
        dict(max_degree=2, penalty=1.5, max_terms=21, minspan=1, endspan=0,
             allow_linear=False),
    ),
    "defaults_d1": (dict(degree=1), dict(max_degree=1)),
    "defaults_d2": (dict(degree=2), dict(max_degree=2)),
}


def r_config(cfg):
    out = {}
    for k, v in cfg.items():
        out[{"fast_k": "fast.k", "Auto_linpreds": "Auto.linpreds",
             "Adjust_endspan": "Adjust.endspan"}.get(k, k)] = v
    out["timing_reps"] = 1
    return out


def describe_py(bfs):
    """Map pymars basis functions to (var, cut, dir) tuples like earth's dirs/cuts."""
    terms = []
    for bf in bfs:
        parts = []
        node = bf
        while node is not None and not isinstance(node, ConstantBasisFunction):
            if isinstance(node, HingeBasisFunction):
                parts.append((node.variable_idx, round(float(node.knot_val), 10),
                              1 if node.is_right_hinge else -1))
            elif isinstance(node, LinearBasisFunction):
                parts.append((node.variable_idx, None, 2))
            node = node.parent1
        terms.append(tuple(sorted(parts)))
    return terms


def describe_r(dirs, cuts, idx):
    terms = []
    for i in idx:
        row = dirs[i - 1]
        parts = []
        for j, d in enumerate(row):
            if d == 0:
                continue
            parts.append((j, None if d == 2 else round(float(cuts[i - 1][j]), 10),
                          int(d)))
        terms.append(tuple(sorted(parts)))
    return terms


def run_case(name, n, seed, cfg_name):
    X, y, f = make(name, n, seed)
    Xt, yt, ft = make(name, 2000, seed + 1)
    tr, te = HERE / "data" / f"{name}_{seed}_train.csv", HERE / "data" / f"{name}_{seed}_test.csv"
    write_csv(tr, X, y)
    write_csv(te, Xt, yt)
    rcfg, pycfg = CONFIGS[cfg_name]
    cfg_path = HERE / "out" / f"{cfg_name}.json"
    cfg_path.write_text(json.dumps(r_config(rcfg)))
    out_path = HERE / "out" / f"{name}_{seed}_{cfg_name}.json"
    subprocess.run(["Rscript", str(HERE / "fit_earth.R"), str(tr), str(te),
                    str(cfg_path), str(out_path)], check=True)
    r = json.loads(out_path.read_text())
    r_dirs, r_cuts = np.array(r["dirs"]), np.array(r["cuts"])
    sel = r["selected"] if isinstance(r["selected"], list) else [r["selected"]]
    r_terms = describe_r(r_dirs, r_cuts, sel)

    if pycfg is None:  # forward-pass comparison against the matched pymars run
        pycfg = CONFIGS["matched_d1"][1]
        m = earth.Earth(**pycfg).fit(X, y)
        py_fwd = describe_py(m.record_.fwd_basis_[-1])
        # Distinct hinge knots, in order of first appearance.
        def knots(terms):
            seen = []
            for t in terms:
                for v, c, d in t:
                    if c is not None and (v, c) not in seen:
                        seen.append((v, c))
            return seen
        kp, kr = knots(py_fwd), knots(r_terms)
        k = 0
        while k < min(len(kp), len(kr)) and kp[k] == kr[k]:
            k += 1
        print(f"[{name} n={n} seed={seed} {cfg_name}] forward knots: pymars {len(kp)}, "
              f"earth {len(kr)}; identical prefix length {k}")
        print(f"   pymars first 6: {kp[:6]}")
        print(f"   earth  first 6: {kr[:6]}")
        print(f"   pymars fwd terms {len(py_fwd)} vs earth fwd terms {len(r_terms)}; "
              f"pymars fwd RSS path (first 6) {np.round(m.record_.fwd_rss_[:6], 4)}; "
              f"earth fwd RSS path (first 6 cols) {np.round(r['fwd_rss'][:6], 4)}")
        return

    t0 = time.perf_counter()
    m = earth.Earth(**pycfg).fit(X, y)
    t_py = time.perf_counter() - t0
    p_tr, p_te = m.predict(X), m.predict(Xt)
    py_terms = describe_py(m.basis_)
    sd = np.std(y)
    rsq_py = 1 - np.sum((y - p_tr) ** 2) / np.sum((y - y.mean()) ** 2)
    mse_te_py = np.mean((ft - p_te) ** 2)
    mse_te_r = np.mean((ft - np.array(r["pred_test"])) ** 2)
    same_terms = sorted(py_terms, key=str) == sorted(r_terms, key=str)
    print(f"[{name} n={n} seed={seed} {cfg_name}] terms pymars {len(py_terms)} earth "
          f"{len(r_terms)}; same term set={same_terms}")
    print(f"   RSq pymars {rsq_py:.5f} earth {r['rsq']:.5f}; GCV pymars {m.gcv_:.5g} "
          f"earth {r['gcv']:.5g}")
    print(f"   test MSE vs truth: pymars {mse_te_py:.4g} earth {mse_te_r:.4g}; "
          f"max|pred diff|/sd(y) test {np.max(np.abs(p_te - np.array(r['pred_test']))) / sd:.3g}")
    print(f"   time pymars {t_py:.2f}s earth {r['time_min']:.3f}s; earth termcond {r['termcond']}")
    if not same_terms:
        only_py = sorted(set(py_terms) - set(r_terms), key=str)[:4]
        only_r = sorted(set(r_terms) - set(py_terms), key=str)[:4]
        print(f"   only pymars: {only_py}")
        print(f"   only earth : {only_r}")


if __name__ == "__main__":
    cases = [
        ("friedman1", 200, 3, "defaults_d1"),
        ("friedman1", 200, 3, "defaults_d2"),
        ("friedman1", 200, 3, "matched_d2"),
        ("additive3", 200, 2, "defaults_d1"),
        ("hinge1d", 200, 1, "defaults_d1"),
    ]
    _old = [
        ("hinge1d", 200, 1, "matched_d1"),
        ("hinge1d", 200, 1, "matched_d1_fwd"),
        ("additive3", 200, 2, "matched_d1"),
        ("additive3", 200, 2, "matched_d1_fwd"),
        ("friedman1", 200, 3, "matched_d1"),
        ("friedman1", 200, 3, "matched_d1_fwd"),
        ("friedman1", 200, 3, "defaults_d1"),
        ("friedman1", 200, 3, "defaults_d2"),
    ]
    wanted = set(sys.argv[1:])
    for c in cases:
        if not wanted or c[0] in wanted:
            run_case(*c)
