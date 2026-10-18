"""pymars 2.0: multivariate adaptive regression splines with a scikit-learn API.

The fit follows the rules of the R package earth, as ``docs/algorithm.md``
states them, and is validated against earth (VALIDATION_PLAN.md).
``EarthRegressor`` is the regressor, and ``Earth`` is the same class, so that
``import pymars as earth; earth.Earth()`` works (API-2); ``EarthClassifier``
refits the selected terms with a logistic model (GLM-1 to GLM-7). The legacy code,
upstream commit d68b54a with the version string 1.0.4, is kept under the git
tag ``legacy-1.0.4-head``.

The types of the fit are public too: ``mars_`` is a ``MarsFit``, which holds a
``ForwardRecord`` (with a ``Termination`` and, when recorded, a
``CandidateLog``) and a ``PruningRecord``; ``MarsParams`` holds the parameters
of a fit (CORE-2 to CORE-5). ``MarsFit.to_dict`` gives a dict with the key
``"format"`` (``DICT_FORMAT``), which ``MarsFit.from_dict`` checks.
"""

from pymars._core import (
    DICT_FORMAT,
    CandidateLog,
    ForwardRecord,
    MarsFit,
    MarsParams,
    PruningRecord,
    Termination,
)
from pymars._estimators import EarthClassifier, EarthRegressor

Earth = EarthRegressor

__all__ = [
    "DICT_FORMAT",
    "CandidateLog",
    "Earth",
    "EarthClassifier",
    "EarthRegressor",
    "ForwardRecord",
    "MarsFit",
    "MarsParams",
    "PruningRecord",
    "Termination",
    "__version__",
]

__version__ = "2.0.0.dev0"
