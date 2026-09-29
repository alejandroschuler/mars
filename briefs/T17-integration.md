# Brief: T17 the integration tests

Read `briefs/COMMON.md` (same folder) first. Issue: T17, #19. Reviewers: 1 (`single`). Model: Sonnet. Start after T13 (#73) and T14 (#78) have merged; both have.

## Read first

- `VALIDATION_PLAN.md`: "Definition of done" items 2 and 3, "Behavior target: scikit-learn first", "scikit-learn compatibility" (the list of integration tests), "Determinism", "Missing values" and "Categorical inputs" (the OneHotEncoder recipe and its errors).
- `docs/algorithm.md` on `origin/main`: API-1 to API-7, ERR-1 to ERR-4, GLM-6, W-6 and W-7.
- `pymars/_estimators.py` and `tests/test_estimators.py`, `tests/test_sklearn_checks.py` and `tests/test_glm.py`, so that you do not repeat what they test.

## Scope

One new file, `tests/test_integration.py`, and nothing else unless a test finds a bug (then stop, keep the failing test in a work-in-progress commit, and report it with a minimal reproduction).

## What to test

The plan's list, with `EarthRegressor` and `EarthClassifier` at their defaults and at `max_degree=2`, on small data (each test under about 2 s at the `ci` profile):

- a Pipeline with a ColumnTransformer and `OneHotEncoder(drop="first")` on a pandas frame with numeric, string and categorical columns; a string column without the encoder raises the ValueError that names OneHotEncoder, if the spec says so;
- GridSearchCV with `n_jobs=2` over `max_degree` and `penalty`;
- `cross_val_predict(..., method="predict_proba")` for the classifier, rows summing to 1;
- StackingRegressor, StackingClassifier, CalibratedClassifierCV and TransformedTargetRegressor;
- pandas feature names (`feature_names_in_`, a mismatch warning or error at predict as scikit-learn does it);
- `sample_weight` through metadata routing (`sklearn.set_config(enable_metadata_routing=True)` inside a context, with `set_fit_request`), in a Pipeline and in `cross_validate`, giving the same fit as a direct `fit(..., sample_weight=w)`;
- pickle and `clone` round trips that keep predictions bit for bit;
- `import pymars as earth; earth.Earth().fit(X, y)` works;
- determinism within one process: two fits give the same predictions bit for bit.

## Test rules

Meaningful tests only (COMMON.md "Tests"): each test checks what the plan lists, and asserts a result (a prediction, a shape, an equality with a direct fit), not only that no error is raised. No mutation campaign is needed for this task; the reviewer samples a few.

## Report

When the pull request is ready: 20 lines or fewer, with the number and head, the gate results, the run time of the new file, and any bug found.
