"""Performance measures for the statistical performance study.

VALIDATION_PLAN.md, "Performance measures". Excess risk and the binary
measures are computed per fit, on a held-out test set; `log_ratio` pairs two
arms' excess risks from the same repetition (the same training data, so the
comparison is paired), and `summarize.py` aggregates the per-repetition log
ratios into a mean and a Monte Carlo standard error.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.special import logit, xlogy
from sklearn.linear_model import LogisticRegression

PROBABILITY_CLIP = 1e-6


def _as_1d_matched(a, b, name_a: str, name_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Both inputs as float64, 1-D, the same shape, and finite.

    Without this, ``(f_hat.reshape(-1, 1) - f_true)`` (an (n, 1) prediction
    against an (n,) truth, which a DataFrame's ``to_numpy()`` produces
    routinely) broadcasts to an (n, n) array and every measure below returns
    a silently wrong number instead of raising.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must be 1-D arrays of the same shape, "
            f"got {a.shape} and {b.shape}"
        )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError(f"{name_a} and {name_b} must be finite (no NaN or inf)")
    return a, b


def excess_risk(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """R(f_hat) = mean((f_hat(X) - f(X))^2) on the test set."""
    f_hat, f_true = _as_1d_matched(f_hat, f_true, "f_hat", "f_true")
    return float(np.mean((f_hat - f_true) ** 2))


def log_ratio(risk_a: float, risk_b: float) -> float:
    """g_i = log R_a - log R_b for one repetition (natural log)."""
    return float(np.log(risk_a) - np.log(risk_b))


def clip_probabilities(
    mu_hat: np.ndarray, clip: float = PROBABILITY_CLIP
) -> tuple[np.ndarray, int]:
    """Clip fitted probabilities to [clip, 1 - clip]. Returns the clipped
    array and the count of values that fell outside that range.
    """
    mu_hat = np.asarray(mu_hat, dtype=np.float64)
    if mu_hat.ndim != 1:
        raise ValueError(f"mu_hat must be a 1-D array, got shape {mu_hat.shape}")
    if not np.isfinite(mu_hat).all():
        raise ValueError("mu_hat must be finite (no NaN or inf)")
    n_clipped = int(np.sum((mu_hat < clip) | (mu_hat > 1 - clip)))
    return np.clip(mu_hat, clip, 1 - clip), n_clipped


def excess_log_loss(
    mu_hat: np.ndarray, mu_true: np.ndarray, clip: float = PROBABILITY_CLIP
) -> tuple[float, int]:
    """E[KL(mu(X) || mu_hat(X))], the mean Bernoulli KL divergence, and the
    number of clipped values. Uses ``scipy.special.xlogy`` (0 * log(0) = 0 by
    convention) so a true mu of exactly 0 or 1 gives no NaN; this study's DGPs
    never produce one, but the formula should not depend on that.
    """
    mu_hat, mu_true = _as_1d_matched(mu_hat, mu_true, "mu_hat", "mu_true")
    mu_hat_c, n_clipped = clip_probabilities(mu_hat, clip)
    kl = (
        xlogy(mu_true, mu_true)
        - xlogy(mu_true, mu_hat_c)
        + xlogy(1 - mu_true, 1 - mu_true)
        - xlogy(1 - mu_true, 1 - mu_hat_c)
    )
    return float(np.mean(kl)), n_clipped


def excess_brier_score(mu_hat: np.ndarray, mu_true: np.ndarray) -> float:
    """E[(mu_hat(X) - mu(X))^2]."""
    mu_hat, mu_true = _as_1d_matched(mu_hat, mu_true, "mu_hat", "mu_true")
    return float(np.mean((mu_hat - mu_true) ** 2))


def calibration_slope(
    mu_hat: np.ndarray, y: np.ndarray, clip: float = PROBABILITY_CLIP
) -> tuple[float, int]:
    """The slope from an unpenalized logistic regression of the observed
    test-set outcome y on logit(mu_hat); 1.0 means good calibration. Returns
    (slope, n_clipped). The slope is NaN when it is not identified: y has
    only one class in the test set, or logit(mu_hat) has no spread (a
    constant fit, as from an intercept-only classifier) so the fitted slope
    would come only from the optimizer's path, not the data.
    """
    mu_hat, y = _as_1d_matched(mu_hat, y, "mu_hat", "y")
    mu_hat_c, n_clipped = clip_probabilities(mu_hat, clip)
    logit_mu_hat = logit(mu_hat_c)
    if len(np.unique(y)) < 2 or np.ptp(logit_mu_hat) == 0:
        return float("nan"), n_clipped
    # C=np.inf, penalty left at its default: scikit-learn's unpenalized fit
    # (VALIDATION_PLAN.md, "Behavior target"); passing penalty=None directly
    # is deprecated as of scikit-learn 1.8.
    model = LogisticRegression(C=np.inf)
    model.fit(logit_mu_hat.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), n_clipped


def irrelevant_covariates_used(
    covariates_used: Iterable[int], relevant: Sequence[int]
) -> int:
    """The number of distinct covariates the fit used (appear in a selected
    term) that are not among the DGP's relevant covariates.
    """
    relevant_set = set(relevant)
    return sum(1 for j in set(covariates_used) if j not in relevant_set)


def uses_any_irrelevant_covariate(
    covariates_used: Iterable[int], relevant: Sequence[int]
) -> bool:
    return irrelevant_covariates_used(covariates_used, relevant) > 0
