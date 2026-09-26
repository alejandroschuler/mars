"""List every failing scikit-learn estimator check for each pymars estimator."""

from __future__ import annotations

import warnings

import sklearn
from sklearn.utils.estimator_checks import check_estimator

import pymars as earth

warnings.filterwarnings("ignore")
print("sklearn", sklearn.__version__)
for est in [
    earth.Earth(),
    earth.EarthRegressor(),
    earth.EarthClassifier(),
    earth.GLMEarth(),
]:
    results = check_estimator(est, on_fail=None)
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{type(est).__name__}: {len(results)} checks, {len(failed)} failed")
    for r in failed:
        msg = str(r["exception"]).strip().splitlines()
        first = msg[0][:150] if msg else ""
        print(f"  - {r['check_name']}: {type(r['exception']).__name__}: {first}")
