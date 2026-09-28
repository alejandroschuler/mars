# Brief: T13 the estimators (pymars/_estimators.py, pymars/__init__.py)

Read `briefs/COMMON.md` (same folder) first, including the scratch-file and mutation-check rules. Issue: T13, #15. Reviewers: 2 (`spec`, `adversarial`). Model: the strongest. Start after T12's `_core.py` and the reference's `fit_mars` (PR #49) have merged.

## Scope

- T13: the shared estimator code, `EarthRegressor`, and `Earth` (the same class object as `EarthRegressor`); `pymars/__init__.py` exports `EarthRegressor`, `EarthClassifier` (once T14 adds it), `Earth` and `__version__`, and nothing else.
- T14 (later): `_glm.py` and `EarthClassifier`. Put the code that both estimators share (parameter handling, validation, weights, fitted attributes, `basis_matrix`, `summary`, tags) in one place so that T14 only adds the classifier's parts.

## Read first

- `docs/algorithm.md` on `origin/main`: API-1 to API-7, ERR-1 to ERR-4, W (above all W-6 and the UserWarning for non-integer weights), CORE-1, CORE-2 and CORE-6, EDGE-1, RESP (several responses), TERM-3 and TERM-5.
- `VALIDATION_PLAN.md`: "Behavior target: scikit-learn first", "scikit-learn compatibility, Python versions and determinism", "Public API", "Core API".
- scikit-learn's developer guide for estimators (reading the web page is fine), for version 1.6 and the latest: `validate_data`, `__sklearn_tags__`, `_check_sample_weight`, metadata routing. Use only public API.

## What to build

- `fit(X, y, sample_weight=None)`: `validate_data`, the weight checks, the ERR rules (the OneHotEncoder message of ERR-2, `allow_missing=True` raises NotImplementedError with the issue link of ERR-3), `MarsParams` built from the parameters (ERR-4: `__init__` only stores them), a 1-D or 2-D y (RESP), and the call `_core.fit_mars(...)` through the module (CORE-6). Fitted attributes of API-3, `predict` (API-4), `basis_matrix(X)`, `summary()`, and the tags of API-5.
- No `coef_`, no `transform`, no `max_iter` (API-6).

## Tests

- `tests/test_sklearn_checks.py`: `parametrize_with_checks` on `EarthRegressor()` and `EarthRegressor(max_degree=2)`, with no expected failures. Until the fast core covers degree 2 and weights (T11 stages 2 and 3), run the checks with the reference in place of the fast core through CORE-6 (`monkeypatch.setattr(pymars._core, "fit_mars", adapter)`, with `MarsFit.from_dict`), and also with the fast core for the settings it supports. Say in the pull request which runs use which core.
- Unit tests for every ERR and API rule, the weights (integer weights against repeated rows, zero weights against removed rows, the UserWarning rule), pandas input with feature names, a 2-D y with one column, pickling, `clone`, and `Earth is EarthRegressor`.
- Mutation checks with `python -B` and a fresh copy per mutant, in your own scratchpad subfolder.

## Report

30 lines or fewer when the pull request is ready: the number and head, the gate results, which checks ran on which core, the mutation results, and any spec question (raise it on #44).
