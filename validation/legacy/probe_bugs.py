"""Quick probes of pymars behaviour that inform VALIDATION_PLAN.md.

Each probe prints one line: PROBE <name>: <finding>.
"""

from __future__ import annotations

import pickle
import time
import warnings

import numpy as np

import pymars as earth
from pymars._basis import ConstantBasisFunction

warnings.filterwarnings("ignore")


def friedman1(n, p=10, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    f = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    return X, f + sigma * rng.standard_normal(n), f


def basis_strs(m):
    return [str(b) for b in m.basis_]


def fwd_basis(m):
    return [str(b) for b in m.record_.fwd_basis_[-1]]


def probe(name, msg):
    print(f"PROBE {name}: {msg}", flush=True)


# --- weights -------------------------------------------------------------
X, y, _ = friedman1(200, p=5, seed=1)
m0 = earth.Earth(max_degree=1).fit(X, y)
for label, w in [
    ("w=1", np.ones(len(y))),
    ("w=10", 10 * np.ones(len(y))),
    ("w=1/n", np.ones(len(y)) / len(y)),
]:
    mw = earth.Earth(max_degree=1).fit(X, y, sample_weight=w)
    same = basis_strs(mw) == basis_strs(m0)
    probe(
        "weights_scale",
        f"{label}: n_terms={len(mw.basis_)} (unweighted {len(m0.basis_)}), "
        f"same basis as unweighted={same}, max|pred diff|="
        f"{np.max(np.abs(mw.predict(X) - m0.predict(X))):.3g}",
    )

# --- response scale -------------------------------------------------------
for s in [1e-9, 1e-6, 1e6, 1e9]:
    ms = earth.Earth(max_degree=1).fit(X, s * y)
    same = basis_strs(ms) == basis_strs(m0)
    probe(
        "y_scale",
        f"y*{s:g}: n_terms={len(ms.basis_)} (base {len(m0.basis_)}), same basis={same}",
    )

# --- penalty changes the forward pass? -----------------------------------------
fwd = {}
for pen in [0.0, 2.0, 3.0, 10.0]:
    mp = earth.Earth(max_degree=2, penalty=pen, max_terms=21).fit(X, y)
    fwd[pen] = fwd_basis(mp)
probe(
    "penalty_forward",
    "forward-pass basis identical across penalty 0/2/3/10: "
    + str(all(fwd[p] == fwd[0.0] for p in fwd))
    + "; forward sizes "
    + str({p: len(v) for p, v in fwd.items()}),
)

# --- intercept pruned, zero columns, collinearity ---------------------------
n_no_icpt = n_zero = n_rankdef = 0
reps = 30
for r in range(reps):
    Xr, yr, _ = friedman1(100, p=5, seed=100 + r)
    mr = earth.Earth(max_degree=2).fit(Xr, yr)
    if not any(isinstance(b, ConstantBasisFunction) for b in mr.basis_):
        n_no_icpt += 1
    B_fwd = np.column_stack(
        [b.transform(Xr, np.zeros_like(Xr, bool)) for b in mr.record_.fwd_basis_[-1]]
    )
    if np.any(np.all(B_fwd == 0, axis=0)):
        n_zero += 1
    if np.linalg.matrix_rank(B_fwd) < B_fwd.shape[1]:
        n_rankdef += 1
probe(
    "structure",
    f"of {reps} fits (n=100, p=5, degree 2): intercept pruned in {n_no_icpt}; "
    f"forward basis has an all-zero column in {n_zero}; "
    f"forward basis rank-deficient in {n_rankdef}",
)

# --- missing values ---------------------------------------------------------
Xm = X.copy()
rng = np.random.default_rng(3)
Xm[rng.uniform(size=Xm.shape) < 0.1] = np.nan
mm = earth.Earth(max_degree=1, allow_missing=True).fit(Xm, y)
pm = mm.predict(Xm)
probe(
    "missing",
    f"allow_missing fit: {len(mm.basis_)} terms; NaN predictions on training rows: "
    f"{int(np.isnan(pm).sum())} of {len(pm)} (rows with any NaN: "
    f"{int(np.isnan(Xm).any(axis=1).sum())})",
)
mm0 = earth.Earth(max_degree=1, allow_missing=True).fit(X, y)
probe(
    "missing_noop",
    "allow_missing=True on complete data gives same basis as False: "
    + str(basis_strs(mm0) == basis_strs(m0)),
)

# --- pickle -------------------------------------------------------------------
for est in [earth.Earth(), earth.EarthRegressor()]:
    est.fit(X, y)
    try:
        est2 = pickle.loads(pickle.dumps(est))
        ok = np.allclose(est2.predict(X), est.predict(X))
        probe("pickle", f"{type(est).__name__}: round trip ok={ok}")
    except Exception as exc:  # noqa: BLE001
        probe("pickle", f"{type(est).__name__}: FAILED {type(exc).__name__}: {exc}")

# --- determinism and invariances ---------------------------------------------
ma = earth.Earth(max_degree=2).fit(X, y)
mb = earth.Earth(max_degree=2).fit(X, y)
probe("determinism_same_process", str(basis_strs(ma) == basis_strs(mb)))
perm = np.random.default_rng(9).permutation(len(y))
mc = earth.Earth(max_degree=2).fit(X[perm], y[perm])
probe(
    "row_permutation",
    f"same basis={basis_strs(mc) == basis_strs(ma)}; max|pred diff|="
    f"{np.max(np.abs(mc.predict(X) - ma.predict(X))):.3g}",
)
cperm = np.array([4, 3, 2, 1, 0])
md = earth.Earth(max_degree=2).fit(X[:, cperm], y)
probe(
    "column_permutation",
    f"max|pred diff|={np.max(np.abs(md.predict(X[:, cperm]) - ma.predict(X))):.3g}",
)

# --- duplicated, constant, collinear columns -------------------------------------
Xdup = np.column_stack([X, X[:, 0]])
mdup = earth.Earth(max_degree=1).fit(Xdup, y)
probe(
    "duplicate_column",
    f"max|pred diff| vs no duplicate={np.max(np.abs(mdup.predict(Xdup) - m0.predict(X))):.3g}",
)
Xconst = np.column_stack([X, np.ones(len(y))])
try:
    mconst = earth.Earth(max_degree=1).fit(Xconst, y)
    probe(
        "constant_column",
        f"ok; max|pred diff|={np.max(np.abs(mconst.predict(Xconst) - m0.predict(X))):.3g}",
    )
except Exception as exc:  # noqa: BLE001
    probe("constant_column", f"FAILED {type(exc).__name__}: {exc}")

# --- tiny n -------------------------------------------------------------------
for n in [3, 5, 10]:
    Xt, yt, _ = friedman1(n, p=3, seed=5)
    try:
        mt = earth.Earth().fit(Xt, yt)
        probe(
            "tiny_n",
            f"n={n}: {len(mt.basis_)} terms, gcv={mt.gcv_}, finite preds="
            f"{np.all(np.isfinite(mt.predict(Xt)))}",
        )
    except Exception as exc:  # noqa: BLE001
        probe("tiny_n", f"n={n}: FAILED {type(exc).__name__}: {exc}")

# --- extreme x values ------------------------------------------------------------
Xe = X.copy()
Xe[:, 0] = Xe[:, 0] * 1e8
me = earth.Earth(max_degree=1).fit(Xe, y)
probe(
    "x_scale",
    f"x0*1e8: max|pred diff| vs unscaled={np.max(np.abs(me.predict(Xe) - m0.predict(X))):.3g}",
)

# --- binary outcomes ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

Xb, _, f = friedman1(400, p=5, seed=11)
pb = 1 / (1 + np.exp(-(f - f.mean()) / 3))
yb = (np.random.default_rng(12).uniform(size=len(pb)) < pb).astype(int)
glm = earth.GLMEarth(family="logistic").fit(Xb, yb)
B = glm.earth_._build_basis_matrix(Xb, glm.basis_, np.zeros_like(Xb, bool))
unpen = LogisticRegression(penalty=None, max_iter=5000).fit(B, yb)
probe(
    "glm_coef",
    f"GLMEarth coef (first 4) {np.round(glm.glm_.coef_.ravel()[:4], 3)} vs "
    f"unpenalized {np.round(unpen.coef_.ravel()[:4], 3)}; has predict_proba="
    f"{hasattr(glm, 'predict_proba')}",
)
clf = earth.EarthClassifier().fit(Xb, yb)
pp = clf.predict_proba(Xb)[:, 1]
Bc = clf.earth_._build_basis_matrix(Xb, clf.basis_, np.zeros_like(Xb, bool))
unpen_c = LogisticRegression(penalty=None, max_iter=5000).fit(Bc, yb)
probe(
    "classifier_proba",
    f"max|p_clf - p_unpenalized| on train={np.max(np.abs(pp - unpen_c.predict_proba(Bc)[:, 1])):.3f}",
)

# --- timing: growth in n ------------------------------------------------------------
times = {}
for n in [100, 200, 400]:
    Xn, yn, _ = friedman1(n, p=5, seed=2)
    t0 = time.perf_counter()
    earth.Earth(max_degree=1, max_terms=21).fit(Xn, yn)
    times[n] = time.perf_counter() - t0
ns = np.array(list(times))
ts = np.array(list(times.values()))
slope = np.polyfit(np.log(ns), np.log(ts), 1)[0]
probe(
    "timing_n",
    "p=5, degree 1, max_terms 21: "
    + ", ".join(f"n={k}: {v:.2f}s" for k, v in times.items())
    + f"; log-log slope {slope:.2f}",
)
