"""Second batch of pymars probes (tiny n, x scale, binary, weights trace, timing)."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

import pymars as earth

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


def probe(name, msg):
    print(f"PROBE {name}: {msg}", flush=True)


def s(m):
    return [str(b) for b in m.basis_]


X, y, _ = friedman1(200, p=5, seed=1)
m0 = earth.Earth(max_degree=1).fit(X, y)

# Where do the unweighted and unit-weight forward passes part ways?
m1 = earth.Earth(max_degree=1).fit(X, y, sample_weight=np.ones(len(y)))
f0 = [str(b) for b in m0.record_.fwd_basis_[-1]]
f1 = [str(b) for b in m1.record_.fwd_basis_[-1]]
k = next((i for i, (a, b) in enumerate(zip(f0, f1)) if a != b), min(len(f0), len(f1)))
probe(
    "unit_weights_trace",
    f"forward terms unweighted {len(f0)}, unit weights {len(f1)}; first difference at "
    f"term {k}: {f0[k] if k < len(f0) else None!r} vs {f1[k] if k < len(f1) else None!r}; "
    f"final gcv {m0.gcv_:.6g} vs {m1.gcv_:.6g}",
)
probe(
    "unit_weights_pruning",
    f"final terms unweighted {len(m0.basis_)} vs unit {len(m1.basis_)}; "
    f"pruning GCV traces equal length={len(m0.record_.pruning_trace_gcv_) == len(m1.record_.pruning_trace_gcv_)}",
)

# Response scaling: compare forward-pass length and final size.
for sc in [1e-9, 1.0, 1e9]:
    ms = earth.Earth(max_degree=1).fit(X, sc * y)
    probe(
        "y_scale_trace",
        f"y*{sc:g}: forward terms {len(ms.record_.fwd_basis_[-1])}, final terms "
        f"{len(ms.basis_)}, forward identical to base="
        f"{[str(b) for b in ms.record_.fwd_basis_[-1]] == f0}",
    )

# Tiny n.
for n in [3, 5, 8, 12]:
    Xt, yt, _ = friedman1(n, p=5, seed=5)
    try:
        mt = earth.Earth().fit(Xt, yt)
        probe(
            "tiny_n",
            f"n={n}: {len(mt.basis_)} terms, gcv={mt.gcv_:.4g}, finite preds="
            f"{np.all(np.isfinite(mt.predict(Xt)))}",
        )
    except Exception as exc:  # noqa: BLE001
        probe("tiny_n", f"n={n}: FAILED {type(exc).__name__}: {exc}")

# Extreme x values.
for sc in [1e-8, 1e8]:
    Xe = X.copy()
    Xe[:, 0] = Xe[:, 0] * sc
    me = earth.Earth(max_degree=1).fit(Xe, y)
    probe(
        "x_scale",
        f"x0*{sc:g}: final terms {len(me.basis_)} (base {len(m0.basis_)}); max|pred diff|="
        f"{np.max(np.abs(me.predict(Xe) - m0.predict(X))):.3g}",
    )
Xo = X.copy()
Xo[0, 0] = 1e6  # one gross outlier in x0
mo = earth.Earth(max_degree=1).fit(Xo, y)
probe("x_outlier", f"one x0 outlier: fit ok, {len(mo.basis_)} terms")

# Collinear (not identical) columns.
Xc = np.column_stack([X, X[:, 0] + 1e-9 * np.random.default_rng(0).standard_normal(len(y))])
mc = earth.Earth(max_degree=1).fit(Xc, y)
probe(
    "near_duplicate_column",
    f"max|pred diff| vs base={np.max(np.abs(mc.predict(Xc) - m0.predict(X))):.3g}; "
    f"max|coef|={np.max(np.abs(mc.coef_)):.3g} (base {np.max(np.abs(m0.coef_)):.3g})",
)

# Binary outcomes.
Xb, _, f = friedman1(400, p=5, seed=11)
pb = 1 / (1 + np.exp(-(f - f.mean()) / 3))
yb = (np.random.default_rng(12).uniform(size=len(pb)) < pb).astype(int)
glm = earth.GLMEarth(family="logistic").fit(Xb, yb)
B = glm.earth_._build_basis_matrix(Xb, glm.basis_, np.zeros_like(Xb, bool))
unpen = LogisticRegression(penalty=None, max_iter=5000).fit(B, yb)
probe(
    "glm_coef",
    f"GLMEarth coef (first 4) {np.round(glm.glm_.coef_.ravel()[:4], 3)} vs unpenalized "
    f"{np.round(unpen.coef_.ravel()[:4], 3)}; has predict_proba={hasattr(glm, 'predict_proba')}; "
    f"predict returns {np.unique(glm.predict(Xb))}",
)
clf = earth.EarthClassifier().fit(Xb, yb)
pp = clf.predict_proba(Xb)[:, 1]
Bc = clf.earth_._build_basis_matrix(Xb, clf.basis_, np.zeros_like(Xb, bool))
unpen_c = LogisticRegression(penalty=None, max_iter=5000).fit(Bc, yb)
probe(
    "classifier_proba",
    f"max|p_clf - p_unpenalized| on train={np.max(np.abs(pp - unpen_c.predict_proba(Bc)[:, 1])):.3f}",
)
y3 = np.digitize(f, np.quantile(f, [1 / 3, 2 / 3]))
clf3 = earth.EarthClassifier().fit(Xb, y3)
probe("multiclass", f"3-class fit ok; Earth fitted on integer-coded labels ({np.unique(y3)})")

# Timing: growth in n and in max_terms.
def tfit(**kw):
    t0 = time.perf_counter()
    earth.Earth(**kw).fit(Xn, yn)
    return time.perf_counter() - t0


times = {}
for n in [100, 200, 400]:
    Xn, yn, _ = friedman1(n, p=5, seed=2)
    times[n] = tfit(max_degree=1, max_terms=21)
ns, ts = np.array(list(times)), np.array(list(times.values()))
probe(
    "timing_n",
    "p=5, degree 1, max_terms 21: "
    + ", ".join(f"n={k}: {v:.2f}s" for k, v in times.items())
    + f"; log-log slope {np.polyfit(np.log(ns), np.log(ts), 1)[0]:.2f}",
)
Xn, yn, _ = friedman1(200, p=5, seed=2)
tt = {mt: tfit(max_degree=2, max_terms=mt) for mt in [11, 21, 31]}
probe(
    "timing_max_terms",
    "n=200, p=5, degree 2: "
    + ", ".join(f"max_terms={k}: {v:.2f}s" for k, v in tt.items()),
)
