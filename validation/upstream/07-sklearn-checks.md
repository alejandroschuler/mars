# Title: Earth fails 5 of 48 scikit-learn estimator checks, and GLMEarth fails 13 of 59

## Summary

`check_estimator` in scikit-learn 1.9.1, run on the code of the tag `legacy-1.0.4-head` (the same checks give 6 failures out of 41 on the 1.0.4 wheel):

- `Earth`: 5 of 48 fail.
- `EarthRegressor` and `EarthClassifier`: 1 fails each, `check_fit2d_predict1d`.
- `GLMEarth`: 13 of 59 fail.

No test runs `check_estimator` on `Earth`, which is the class that the documentation presents. The wrappers hide `minspan`, `endspan`, `allow_missing`, `categorical_features` and `feature_importance_type`. Fixing the checks also makes the estimators work in scikit-learn pipelines, `clone`, `GridSearchCV` and stacking, which is where most users meet them.

## Reproduction

```python
import warnings
import sklearn
from sklearn.utils.estimator_checks import check_estimator
import pymars as earth
warnings.filterwarnings("ignore")
print("sklearn", sklearn.__version__)
for est in [earth.Earth(), earth.EarthRegressor(), earth.EarthClassifier(), earth.GLMEarth()]:
    results = check_estimator(est, on_fail=None)
    failed = [r for r in results if r["status"] == "failed"]
    print(f"{type(est).__name__}: {len(results)} checks, {len(failed)} failed")
    for r in failed:
        print("  -", r["check_name"])
```

Output (the Python code of the tag first on `sys.path`):

```
sklearn 1.9.1
Earth: 48 checks, 5 failed
  - check_no_attributes_set_in_init
  - check_estimators_unfitted
  - check_n_features_in_after_fitting
  - check_mixin_order
  - check_fit_check_is_fitted
EarthRegressor: 59 checks, 1 failed
  - check_fit2d_predict1d
EarthClassifier: 62 checks, 1 failed
  - check_fit2d_predict1d
GLMEarth: 59 checks, 13 failed
  - check_estimators_unfitted
  - check_do_not_raise_errors_in_init_or_set_params
  - check_n_features_in_after_fitting
  - check_estimators_nan_inf
  - check_regressors_train
  - check_regressors_train
  - check_regressors_train
  - check_regressor_data_not_an_array
  - check_regressors_no_decision_function
  - check_fit_idempotent
  - check_fit_check_is_fitted
  - check_n_features_in
  - check_fit2d_predict1d
```

What each failure says:

- `Earth` sets learned attributes in `__init__`, raises `RuntimeError` instead of `NotFittedError` before `fit`, does not check the number of features in `predict`, lists its mixin and `BaseEstimator` in the wrong order, and passes `check_is_fitted` before `fit`.
- `GLMEarth` inherits from `EarthRegressor` but fits a classifier, so the regressor checks fail with "Unknown label type: continuous"; it also forwards `**kwargs` in a way that `get_params` and `set_params` cannot handle.

The weight check `check_sample_weight_equivalence_on_dense_data` passes today. It is worth keeping.

## Suggestion

Use `validate_data` and `check_is_fitted`, set learned attributes only in `fit`, raise `NotFittedError`, and list the mixins before `BaseEstimator`, as scikit-learn requires. Run `check_estimator` on every public estimator in the tests, and keep the list of expected failures short and accurate. At present the expected-failure lists in `tests/test_sklearn_compat.py` are stale (see the issue on the reference tests).

The 2.0 code in the fork passes every check on its estimators, and its tests run them with `parametrize_with_checks`.
