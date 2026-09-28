"""pymars 2.0: multivariate adaptive regression splines with a scikit-learn API.

The fit follows the rules of the R package earth, as ``docs/algorithm.md``
states them, and is validated against earth (VALIDATION_PLAN.md).
``EarthRegressor`` is the regressor, and ``Earth`` is the same class, so that
``import pymars as earth; earth.Earth()`` works (API-2). The legacy code,
upstream commit d68b54a with the version string 1.0.4, is kept under the git
tag ``legacy-1.0.4-head``.
"""

from pymars._estimators import EarthRegressor

Earth = EarthRegressor

__all__ = ["Earth", "EarthRegressor", "__version__"]

__version__ = "2.0.0.dev0"
