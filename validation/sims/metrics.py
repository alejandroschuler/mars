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
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

PROBABILITY_CLIP = 1e-6


def excess_risk(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """R(f_hat) = mean((f_hat(X) - f(X))^2) on the test set."""
    return float(
        np.mean((np.asarray(f_hat, dtype=np.float64) - np.asarray(f_true)) ** 2)
    )


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
    n_clipped = int(np.sum((mu_hat < clip) | (mu_hat > 1 - clip)))
    return np.clip(mu_hat, clip, 1 - clip), n_clipped


def excess_log_loss(
    mu_hat: np.ndarray, mu_true: np.ndarray, clip: float = PROBABILITY_CLIP
) -> tuple[float, int]:
    """E[KL(mu(X) || mu_hat(X))], the mean Bernoulli KL divergence, and the
    number of clipped values.
    """
    mu_hat_c, n_clipped = clip_probabilities(mu_hat, clip)
    mu = np.asarray(mu_true, dtype=np.float64)
    kl = mu * np.log(mu / mu_hat_c) + (1 - mu) * np.log((1 - mu) / (1 - mu_hat_c))
    return float(np.mean(kl)), n_clipped


def excess_brier_score(mu_hat: np.ndarray, mu_true: np.ndarray) -> float:
    """E[(mu_hat(X) - mu(X))^2]."""
    return float(
        np.mean((np.asarray(mu_hat, dtype=np.float64) - np.asarray(mu_true)) ** 2)
    )


def calibration_slope(
    mu_hat: np.ndarray, y: np.ndarray, clip: float = PROBABILITY_CLIP
) -> tuple[float, int]:
    """The slope from an unpenalized logistic regression of the observed
    test-set outcome y on logit(mu_hat); 1.0 means good calibration. Returns
    (slope, n_clipped); slope is NaN when y has only one class in the test
    set (cannot happen at n = 10,000 except in a degenerate DGP).
    """
    mu_hat_c, n_clipped = clip_probabilities(mu_hat, clip)
    logit_mu_hat = logit(mu_hat_c).reshape(-1, 1)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan"), n_clipped
    # C=np.inf, penalty left at its default: scikit-learn's unpenalized fit
    # (VALIDATION_PLAN.md, "Behavior target"); passing penalty=None directly
    # is deprecated as of scikit-learn 1.8.
    model = LogisticRegression(C=np.inf)
    model.fit(logit_mu_hat, y)
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
