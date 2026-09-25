# Legacy prototypes

These scripts produced the preliminary findings in `VALIDATION_PLAN.md` (F1 to F16 and the fit-time table). They test pymars 1.0.4, the legacy code, and they import its internal modules (`pymars._basis`, `pymars._util`), so they run only against that code: a checkout of the tag `legacy-1.0.4-head` with the `.pth` recipe in the plan's first appendix, or a venv with the `mars-earth==1.0.4` wheel.

| File | Purpose |
|---|---|
| `fit_earth.R` | Fits R earth on a CSV train and test pair with arguments from a JSON config and writes the fit as JSON |
| `compare_earth.py` | Fits pymars and earth on the same data in the matched and defaults modes; writes `data/` and `out/` |
| `probe_bugs.py`, `probe_bugs2.py` | Probes of the legacy behavior; each prints `PROBE <name>: <finding>` |
| `f3_mechanism.py` | The traced fits behind F3 (the pruned intercept) |
| `sk_checks.py` | The scikit-learn `check_estimator` results behind F11 |
| `timing.py`, `timing/` | The fit-time grid against earth |
| `blas_det.py`, `wheel_vs_head.py` | The determinism checks and the comparison of HEAD with the 1.0.4 wheel (F16) |
| `data/`, `out/`, `*.json` | The inputs and outputs of the runs above; `out/trace9.txt` is the earth trace behind F5 |

The harness task (T02 in the plan) turns these prototypes into `validation/harness/`. They use earth only as a black box: they call it and read its outputs.
