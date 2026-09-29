"""The logistic refit of the classifier on the selected basis.

Spec: ``docs/algorithm.md``, "GLM refit for the classifier" (GLM-2 to GLM-5;
the estimator applies GLM-1, GLM-6 and GLM-7), LA-4 (dependent columns get
the coefficient 0), W-3 (rows with zero weight are dropped) and OQ-7 (the
solver is this module's choice; ``dev/DECISIONS.md`` gives the reason).

The model. Q ≥ 2 classes, coded 0 to Q - 1, in the reference-class form:
class 0 has the linear predictor 0, and class k ≥ 1 has η_k = B·β_k. With two
classes this is the binomial GLM with the logit link, and with Q ≥ 3 the
multinomial model of ``nnet::multinom``. The coefficients minimize

    Σ_i w_i·nll_i + (alpha/2)·Σ_k Σ_j g_jk²,

with nll_i = log Σ_k exp(η_ik) - η_i,y_i the negative log-likelihood of case i,
and g_jk the coefficients of the non-intercept columns after each column is
standardized to weighted mean 0 and Σ_i w_i z_ij² = N (GLM-2). The intercept
is not penalized, and alpha = 0 gives the maximum likelihood fit, which does
not depend on the standardization.

The columns. Column 0 of B is the intercept. A column that is constant over
the cases (all its values equal, "Conventions") and a column that LA-4 finds
dependent on earlier ones get the coefficient 0 and take no part in the fit.
If no other column is left, the fit is the intercept alone, whose
probabilities are the weighted class frequencies (GLM-5); no solver runs.

The solver. Newton's method on the standardized columns, from the
intercept-only fit, with a backtracking line search (Armijo) that needs a
strict decrease. It stops when half the Newton decrement, which estimates how
far the objective lies above its minimum, is at most ``DECREMENT_TOL`` times
the objective; it then takes one more full Newton step, which squares the
relative error of the coefficients, so they reach about machine precision. It
gives up after ``MAX_ITER`` steps, or when the line search finds no
decrease. Under separation the minimum is not attained: the iterates grow
without bound and the relative decrement stays near 1/2, so the fit ends at
``MAX_ITER`` without converging. A ConvergenceWarning that
suggests a positive ``glm_alpha`` follows when the fit did not converge, or
when some fitted probability is within 10·ε of 0 or 1 (GLM-4); the last
iterate is kept.

Numerics: float64; no function writes into its inputs; the log-likelihood,
the probabilities and 1 - p come from log-sum-exp forms, so they neither
overflow nor lose the small probabilities; every tolerance is relative.
Complexity: O(n·M·r) for the column checks (r the rank), then per Newton step
O(n·d²·(Q - 1)² + d³·(Q - 1)³) time for d kept columns, and O(n·(M + Q) +
d²·(Q - 1)²) memory.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.special
from sklearn.exceptions import ConvergenceWarning

from pymars import _linalg

__all__ = ["GlmFit", "fit_glm", "linear_predictors", "probabilities"]

FloatArray = npt.NDArray[np.float64]

#: The largest number of Newton steps.
MAX_ITER = 100
#: Converged when half the Newton decrement is at most this times the
#: objective; this lies well above the rounding of the objective (about
#: 1e-16 relative), so that the line search can still see the decrease.
DECREMENT_TOL = 1e-12
#: GLM-4: a probability within this of 0 or 1 warns (R's glm uses the same).
EXTREME_PROB = 10.0 * np.finfo(np.float64).eps
#: The Armijo constant and the most halvings of the line search.
_ARMIJO = 1e-4
_HALVINGS = 60


class GlmFit(NamedTuple):
    """The refit (GLM-2): ``coef`` is (M,) with two classes, the coefficients
    of class 1, and (M, Q) otherwise, with a first column of 0; ``converged``
    and ``n_iter`` describe the solver (``n_iter`` = 0 for the intercept-only
    fit of GLM-5); ``extreme`` is True when some fitted probability is within
    10·ε of 0 or 1 (GLM-4)."""

    coef: FloatArray
    converged: bool
    n_iter: int
    extreme: bool


def linear_predictors(B: npt.ArrayLike, coef: npt.ArrayLike) -> FloatArray:
    """Return the Q linear predictors B·coef, (n, Q) with a first column of 0,
    for coefficients of shape (M,) (two classes) or (M, Q)."""
    B = np.asarray(B, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        eta = B @ coef
        return np.column_stack([np.zeros_like(eta), eta])
    return B @ coef


def probabilities(eta: npt.ArrayLike) -> FloatArray:
    """Return the softmax of the rows of the (n, Q) linear predictors; each row
    sums to 1 (GLM-6). Complexity: O(n·Q)."""
    return scipy.special.softmax(np.asarray(eta, dtype=np.float64), axis=1)


def _complements(full: FloatArray, lse: FloatArray) -> FloatArray:
    """1 - P_ik for each class, as exp(log Σ_{l≠k} exp(η_il) - lse_i), which
    keeps its relative accuracy when P_ik is near 1. Complexity: O(n·Q²)."""
    Q = full.shape[1]
    out = np.empty_like(full)
    for k in range(Q):
        others = np.delete(full, k, axis=1)
        out[:, k] = np.exp(scipy.special.logsumexp(others, axis=1) - lse)
    return out


class _Problem:
    """The objective of GLM-2 on the design D = [1, Z] (n, d) of the kept,
    standardized columns, with Θ of shape (d, Q - 1), vectorized class by
    class (index k·d + j)."""

    def __init__(self, D, codes, Q, w, alpha):
        self.D, self.codes, self.Q, self.w, self.alpha = D, codes, Q, w, alpha
        n, d = D.shape
        self.d = d
        self.T = np.zeros((n, Q))
        self.T[np.arange(n), codes] = 1.0
        # the penalty acts on every coefficient but the intercepts (row 0)
        self.pen = np.ones((d, Q - 1))
        self.pen[0] = 0.0

    def _full(self, theta: FloatArray) -> FloatArray:
        eta = self.D @ theta
        return np.column_stack([np.zeros(eta.shape[0]), eta])

    def _nll(self, full: FloatArray) -> FloatArray:
        """nll_i = log(1 + Σ_{k≠y_i} exp(η_ik - η_i,y_i)), by ``logaddexp``, so
        that a case with a probability near 1 keeps its small nll_i exactly
        rather than as a difference of two large numbers."""
        rel = full - full[np.arange(full.shape[0]), self.codes][:, None]
        others = scipy.special.logsumexp(rel, axis=1, b=1.0 - self.T)
        return np.logaddexp(0.0, others)

    def value(self, theta: FloatArray) -> float:
        nll = self._nll(self._full(theta))
        return float(self.w @ nll) + 0.5 * self.alpha * float(
            np.sum(self.pen * theta**2)
        )

    def fitted(self, theta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """The (n, Q) predictors [0, η], probabilities P and complements 1 - P."""
        full = self._full(theta)
        lse = scipy.special.logsumexp(full, axis=1)
        return full, np.exp(full - lse[:, None]), _complements(full, lse)

    def derivatives(self, theta: FloatArray):
        """The value, the gradient (d·(Q - 1),) and the Hessian."""
        full, P, C = self.fitted(theta)
        f = float(self.w @ self._nll(full)) + 0.5 * self.alpha * float(
            np.sum(self.pen * theta**2)
        )
        # P - T, with P_ik - 1 = -(1 - P_ik) from the complement for k = y_i
        R = np.where(self.T > 0.0, -C, P)
        D, w, d, K1 = self.D, self.w, self.d, self.Q - 1
        G = D.T @ (w[:, None] * R[:, 1:]) + self.alpha * (self.pen * theta)
        H = np.empty((K1 * d, K1 * d))
        for k in range(K1):
            for m in range(k, K1):
                if k == m:
                    v = w * P[:, k + 1] * C[:, k + 1]
                else:
                    v = -w * P[:, k + 1] * P[:, m + 1]
                block = D.T @ (D * v[:, None])
                H[k * d : (k + 1) * d, m * d : (m + 1) * d] = block
                H[m * d : (m + 1) * d, k * d : (k + 1) * d] = block.T
        H[np.diag_indices_from(H)] += self.alpha * self.pen.T.ravel()
        return f, G.T.ravel(), H


def _newton_step(H: FloatArray, g: FloatArray) -> FloatArray:
    """Solve H·s = -g by Cholesky; a Hessian that is not positive definite (all
    probabilities saturated, say) takes the least-squares solution."""
    try:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(H), -g)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return scipy.linalg.lstsq(H, -g)[0]


def _solve(problem: _Problem, theta: FloatArray) -> tuple[FloatArray, bool, int]:
    """Newton's method with a backtracking line search (module docstring)."""
    shape = theta.shape
    for it in range(1, MAX_ITER + 1):
        f, g, H = problem.derivatives(theta)
        s = _newton_step(H, g)
        slope = float(g @ s)  # -λ², the directional derivative
        step = s.reshape(shape[::-1]).T
        if -0.5 * slope <= DECREMENT_TOL * f:
            # a last full step: near the minimum Newton's steps square the error
            return theta + step, True, it
        t = 1.0
        for _ in range(_HALVINGS):
            new = theta + t * step
            value = problem.value(new)
            # strictly lower, so that a step lost in rounding is not accepted
            if value < f and value <= f + _ARMIJO * t * slope:
                break
            t *= 0.5
        else:
            return theta, False, it - 1
        theta = new
    return theta, False, MAX_ITER


def fit_glm(
    B: npt.ArrayLike,
    codes: npt.ArrayLike,
    n_classes: int,
    w: npt.ArrayLike | None = None,
    alpha: float = 0.0,
) -> GlmFit:
    """Fit the logistic refit of GLM-2 on the columns of B.

    B is (n, M) with the intercept in column 0; ``codes`` (n,) holds the class
    of each row, 0 to ``n_classes`` - 1, and every class must occur among the
    rows with positive weight (GLM-1 makes it so); w is (n,), finite and ≥ 0,
    or None for weights 1; ``alpha`` ≥ 0 is ``glm_alpha``. Rows with zero
    weight are dropped (W-3). Constant and dependent columns get the
    coefficient 0 (LA-4, GLM-2, GLM-3); with no other column left, the
    probabilities are the weighted class frequencies (GLM-5). Issues the
    ConvergenceWarning of GLM-4. Complexity: see the module docstring.
    """
    B = np.asarray(B, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.intp)
    Q = int(n_classes)
    n, M = B.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    rows = w > 0.0
    B, codes, w = B[rows], codes[rows], w[rows]
    N = math.fsum(w)
    T = np.zeros((B.shape[0], Q))
    T[np.arange(B.shape[0]), codes] = 1.0
    freq = (w @ T) / N  # the weighted class frequencies, all positive
    base = np.log(freq[1:]) - math.log(freq[0])

    constant = np.all(B[0] == B, axis=0)
    keep = _linalg.independent_columns(B, w) & ~constant
    keep[0] = False
    cols = np.flatnonzero(keep)
    coef = np.zeros((M, Q))
    coef[0, 1:] = base
    if cols.size == 0:  # GLM-5: the intercept alone
        return GlmFit(coef[:, 1] if Q == 2 else coef, True, 0, False)

    A = B[:, cols]
    mean = (w @ A) / N
    sd = np.sqrt((w @ (A - mean) ** 2) / N)
    D = np.column_stack([np.ones(A.shape[0]), (A - mean) / sd])
    problem = _Problem(D, codes, Q, w, float(alpha))
    theta0 = np.zeros((D.shape[1], Q - 1))
    theta0[0] = base
    theta, converged, n_iter = _solve(problem, theta0)

    gamma = theta[1:]
    coef[cols, 1:] = gamma / sd[:, None]
    coef[0, 1:] = theta[0] - (mean / sd) @ gamma
    _, P, C = problem.fitted(theta)
    extreme = bool(np.any(P <= EXTREME_PROB) or np.any(C <= EXTREME_PROB))
    if not converged or extreme:
        why = [] if converged else ["did not converge"]
        if extreme:
            why.append("gave fitted probabilities numerically 0 or 1")
        warnings.warn(
            "The logistic refit of EarthClassifier "
            + " and ".join(why)
            + ", as with separated classes; the coefficients are those of the "
            "last step. A positive glm_alpha, such as 1e-4, gives an L2 penalty "
            "with a finite minimum.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return GlmFit(coef[:, 1] if Q == 2 else coef, converged, n_iter, extreme)
