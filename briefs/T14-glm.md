# Brief: T14 the GLM refit and the classifier (pymars/_glm.py, EarthClassifier)

Read `briefs/COMMON.md` (same folder) first, including the scratch-file and mutation-check rules. Issue: T14, #16. Reviewers: 2 (`spec`, `adversarial`). Model: the strongest. Start after T13 (#73, `pymars/_estimators.py`) has merged.

## Read first

- `docs/algorithm.md` on `origin/main`: GLM-1 to GLM-7, API-2, API-4 (the classifier's methods), W (above all W-3, zero weights dropped first), LA-4 (collinear columns get coefficient 0), OQ-7 (T14 picks the solver), and the departure table rows for GLM-2.
- `VALIDATION_PLAN.md`: "Behavior target: scikit-learn first" (the rows "Multiclass outcomes", "GLM penalty and separation" and "Degenerate inputs"), "Binary outcomes", the tolerance table (GLM coefficients and fitted probabilities: relative 1e-5, absolute 1e-7), "Target modules" (`_glm.py`, `_estimators.py`) and "Public API".
- The merged code you build on, by its docstrings: `pymars/_estimators.py` (the shared estimator code that T13 put in one place for you), `pymars/_core.py` (`fit_mars`, `MarsFit`), `pymars/_linalg.py`.
- `tests/test_conformance.py` on main already reads the GLM fields of the earth fixtures (`glm_coef`, the probabilities as fitted values); reuse its loaders.

## Scope

- `pymars/_glm.py`: the refit of GLM-2 on the selected columns B_S. Binomial with the logit link for 2 classes; one multinomial model in the reference-class form for 3 or more. Weights. The optional L2 penalty `glm_alpha` on the weighted, standardized non-intercept columns. Columns with weighted variance 0, and columns that LA-4 finds dependent, get coefficient 0. GLM-4's ConvergenceWarning (it suggests `glm_alpha`), GLM-5's intercept-only case. Pick the solver (OQ-7): scikit-learn's LogisticRegression with `C=np.inf` (no `penalty` argument; it is deprecated from 1.8), scipy, or a short IRLS / Newton with a line search. It must reach the minimum of GLM-2 to relative 1e-5 against R's `glm` on the fixtures (GLM-3), and work on scikit-learn 1.6 and the latest release. Record the choice and its reason in `dev/DECISIONS.md`.
- `EarthClassifier` in `pymars/_estimators.py`: GLM-1 (`check_classification_targets`, `classes_` from rows with positive weight, one 0/1 response for 2 classes, Q indicator responses for Q ≥ 3), GLM-6 (`predict_proba`, `decision_function`, `predict` with the first class on a tie), GLM-7 (the one-class error), the fitted attributes `classes_` and `glm_`, `summary()` with the GLM coefficients, the classifier tags. Export it from `pymars/__init__.py`.
- Tests: `tests/test_glm.py` and the classifier's part of the scikit-learn checks (`parametrize_with_checks` on `EarthClassifier()`, with no expected failures).
- Only these files: `pymars/_glm.py`, `pymars/_estimators.py`, `pymars/__init__.py`, `tests/test_glm.py`, `tests/test_sklearn_checks.py` (the classifier's entries), `dev/DECISIONS.md` (one entry). About 800 changed lines at most; split into two PRs (the refit, then the classifier) if it grows past that.

## Dependency on T11 stage 3

Q ≥ 3 classes need K = Q responses, and weights need W in the fast forward pass; both come with T11 stage 3 (#13), which is in progress on the branch `t11-stage3`. Until it merges, test the multiclass path and the weights through the reference core (as T13 did for its checks), skip only what needs the fast core with a message that names #13, and match the skip on that message only (see #76, item 3).

## Tests

- Against the earth fixtures S14 (binary, degree 2), S18 (three classes) and S20 (separation): on earth's selected basis, the binomial coefficients and fitted probabilities within relative 1e-5 and absolute 1e-7 of earth's `glm.coefficients` where earth converged without a warning; for S20, both warn and the probabilities are within 1e-3 of 0 or 1 where earth's are. For S18 the terms come from the multi-response pass; compare the multinomial probabilities with `nnet::multinom` on earth's selected basis if the fixture has them, and otherwise say so in the PR body and propose the fixture change as a follow-up.
- String, boolean and {-1, 1} labels give the same fit as 0/1 (GLM-1). Zero weights equal dropping the rows (W-3). The `glm_alpha` penalty is invariant to a rescaling of a column (the standardization). A dependent column gets coefficient 0.
- Meaningful tests only (COMMON.md "Tests"). Prefer one comparison on a fixture that pins several rules over one test per rule.

## Report

When the pull request is ready: 30 lines or fewer, with the pull request number and head, the gate results, the solver and its reason, the largest errors against earth on S14, S18 and S20, the mutation results, and any spec question (raise it on #44).
