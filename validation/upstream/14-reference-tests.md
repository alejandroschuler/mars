# Title: The reference regression tests compare the code with its own stored output, at loose tolerances

## Summary

`tests/test_reference_regression.py` fits the model on ten small cases and compares the result with `tests/fixtures/reference_regression_cases.json`, which holds earlier output of the same code. It does not compare with another implementation. The details:

- The fixture stores the `basis` and `coef` of each case, but the test never compares them. It only asserts that the fitted basis is not empty and that it has as many coefficients as terms (lines 258-266 on the tag `legacy-1.0.4-head`).
- Predictions are compared at `atol=1e-12` for seven cases, but at 5e-2, 6e-1 and 6e-1 for `penalty_sensitive_1d`, `mixed_3d` and `interaction_2d`. The metrics `gcv`, `rss` and `mse` are compared at 5e-2, 6e-1 and 1.5 for those cases. A prediction error of 0.6 and a metric error of 1.5 are large next to the scale of these data.
- The tolerances were widened in nine commits on 2026-05-03, which fits the cross-run flakiness described in the issue on the absolute stopping rule.
- One fixture, `missingness_2d`, has no intercept, so the test locks in the behavior described in the issue on the pruned intercept.

The expected-failure lists in the scikit-learn tests are stale as well (`tests/test_sklearn_compat.py`, lines 186-194 and 339-350):

- With scikit-learn 1.9.1, only `check_fit2d_predict1d` fails for `EarthRegressor` and for `EarthClassifier` at default settings, and the lists have 6 and 9 entries.
- One entry in each list is a garbled name (`check_fit_score_takes_y_दानी`) that matches no check, so it does nothing.
- `tests/test_earth.py` turns a failure into an expected failure at run time (`pytest.xfail` around line 160, when no interaction term is selected), so that test cannot fail.

## Evidence

Read from the files on the tag; no run is needed. For the claim about the checks, see the output in the issue on the scikit-learn checks.

## Suggestion

- Compare against output that does not come from the code under test. earth's results on the same inputs are the natural choice: store the inputs and earth's terms, coefficients, GCV and predictions as JSON together with the R and earth versions, and compare the terms exactly and the numbers at a tolerance set by the conditioning of the basis.
- Compare the stored terms and coefficients in the existing test.
- Keep one tolerance rule for all cases, and treat a case that needs a looser tolerance as a finding to look into.
- Replace the expected-failure lists by the real list (one entry at present), and remove the run-time `xfail`.

The fork has such a conformance suite (`validation/harness/`, `validation/fixtures/`, `tests/test_conformance.py`), and it is offered in the note on the 2.0 code.
