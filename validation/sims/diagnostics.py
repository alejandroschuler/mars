"""DGP diagnostics from a fixed one-million-case draw per DGP.

VALIDATION_PLAN.md, "Data-generating processes": "Before the pilot, one draw
of one million cases per DGP gives the diagnostics to report: the signal
variance Var f, the noise standard deviation sigma, the population R^2, the
share of Var f that the best linear approximation of f explains, and for the
binary DGPs the range of the true probability mu(X)."

Run once: ``uv run --frozen python -m validation.sims.diagnostics``. Writes
``diagnostics.json`` (committed, small) next to this file. ``dgps.generate``
reads it at simulation time for sigma (regression DGPs) and lambda and mean_f
(binary DGPs), so the 10**6-case draws happen only here, not on every fit.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.special import expit, logit

from . import dgps, seeds

N_DRAW = 1_000_000
LOGIT_095 = float(logit(0.95))  # ln 19, about 2.944


def _linear_r2_share(X: np.ndarray, f_values: np.ndarray) -> float:
    """Share of Var f that the best linear approximation of f (in mean
    squared error, over this draw) explains: 1 minus the residual variance of
    the OLS fit of f_values on X (with an intercept), over Var f.
    """
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, f_values, rcond=None)
    resid = f_values - design @ coef
    var_f = f_values.var(ddof=0)
    if var_f == 0.0:
        return float("nan")
    return float(1.0 - resid.var(ddof=0) / var_f)


def _regression_diagnostics(
    dgp: dgps.Dgp, rng: np.random.Generator, n_draw: int
) -> dict:
    X = dgp.sample_covariates(rng, n_draw)
    f_values = dgp.f(X)
    var_f = float(f_values.var(ddof=0))
    sd_f = float(np.sqrt(var_f))
    entry: dict = {"var_f": var_f, "sd_f": sd_f}
    if dgp.name == "D6":
        entry["sigma"] = 1.0
        entry["population_r2"] = 0.0
        entry["note"] = "f = 0 by construction; sigma fixed at 1, one noise level"
    else:
        entry["linear_r2_share"] = _linear_r2_share(X, f_values)
        for level, r2 in dgps.R_SQUARED.items():
            entry[f"population_r2_{level}"] = r2
            entry[f"sigma_{level}"] = sd_f * float(np.sqrt((1 - r2) / r2))
    if dgp.name == "D8":
        pairs = [
            float(np.corrcoef(X[:, i], X[:, j])[0, 1])
            for i in range(dgp.p)
            for j in range(i + 1, dgp.p)
        ]
        entry["empirical_corr_mean"] = float(np.mean(pairs))
        entry["empirical_corr_pair01"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    return entry


def _binary_diagnostics(dgp: dgps.Dgp, rng: np.random.Generator, n_draw: int) -> dict:
    X = dgp.sample_covariates(rng, n_draw)
    f_values = dgp.f(X)
    mean_f = float(f_values.mean())
    centered = f_values - mean_f
    q01, q99 = (float(v) for v in np.percentile(centered, [1, 99]))
    scale = max(abs(q01), q99)
    lam = LOGIT_095 / scale
    mu = expit(lam * centered)
    quantile_points = (1, 5, 25, 50, 75, 95, 99)
    return {
        "mean_f": mean_f,
        "q01": q01,
        "q99": q99,
        "lambda": lam,
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "mu_quantiles": {
            str(q): float(v)
            for q, v in zip(
                quantile_points, np.percentile(mu, quantile_points), strict=True
            )
        },
        "share_mu_in_0.3_0.7": float(np.mean((mu >= 0.3) & (mu <= 0.7))),
    }


def compute_diagnostics(n_draw: int = N_DRAW) -> dict:
    """Compute every DGP's diagnostics from one draw of size ``n_draw`` each.

    ``n_draw`` defaults to the plan's 10**6; tests pass a smaller value to
    check the formulas cheaply (the seed is the same fixed diagnostic seed
    either way, just fewer rows, so a test result is not the committed one).
    """
    result: dict = {"n_draw": n_draw, "logit_0.95": LOGIT_095}
    for name, dgp in dgps.REGISTRY.items():
        rng = seeds.diagnostic_rng(name)
        result[name] = (
            _binary_diagnostics(dgp, rng, n_draw)
            if dgp.binary
            else _regression_diagnostics(dgp, rng, n_draw)
        )
    return result


def main() -> None:
    result = compute_diagnostics()
    dgps.DIAGNOSTICS_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {dgps.DIAGNOSTICS_PATH}")
    lam_d3 = result["D3-bin"]["lambda"]
    lam_d4 = result["D4-bin"]["lambda"]
    corr_d8 = result["D8"]["empirical_corr_mean"]
    print(f"lambda D3-bin = {lam_d3:.4f} (plan: approximately 1.60)")
    print(f"lambda D4-bin = {lam_d4:.4f} (plan: approximately 0.28)")
    print(f"D8 mean covariate correlation = {corr_d8:.4f} (plan: approximately 0.58)")


if __name__ == "__main__":
    main()
