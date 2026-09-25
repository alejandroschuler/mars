# Validation and improvement plan for pymars

Draft for review, 2026-09-25. Branch `validation-plan`, based on upstream commit `d68b54a`. No library code has changed.

## Purpose and short answer

The `supervised-learning` skill in the stats-research plugin (ctml-skills) recommends MARS (multivariate adaptive regression splines) as the spline learner in super learner libraries. In R that learner is `earth`. This plan asks whether `pymars` (PyPI name `mars-earth`, import name `pymars`) can fill that role in Python, what must change first, and how to show the result to other people.

The short answer is no, not yet, but the gap can be closed. The core search works: with settings that make the two programs comparable, pymars picked the same ten knots in the same order as earth on Friedman's first test function (Friedman #1 below), and the residual sum of squares agreed with earth's to about 13 significant digits after every step. The problems sit in the rules and defaults around that search. Four of them give wrong or unusable results in common use: sample weights, pruning of the intercept, missing values, and the refit for binary outcomes.

Speed is the second problem. A fit takes roughly 760 to 12,000 times as long as with earth for 250 to 2,000 training cases and 10 covariates, and at the larger sizes the time grows almost with the square of the number of cases. Neither the updating formula of Friedman (1991) nor the Fast MARS method of Friedman (1993) is implemented. The existing reference tests cannot detect these problems, because they compare pymars only with its own stored outputs, never check the terms or the coefficients, and accept prediction errors up to 0.6.

The plan has a setup step (Phase 0) and five phases of work:

1. Build a harness that fits pymars and earth on the same data, and describe the current behavior. This phase makes no library changes.
2. Fix the correctness bugs, with the tests written first.
3. Replace the full least-squares solve for every candidate with Friedman's updating formula, and show that the selected models do not change.
4. Run a simulation study of held-out error against earth.
5. Write a validation report and prepare what to offer upstream.

[Phases, exit criteria and review points](#phases-exit-criteria-and-review-points) gives the exit criteria and the points where you review the work.

## Terms and abbreviations

| Term | Meaning |
|---|---|
| MARS | Multivariate adaptive regression splines (Friedman 1991) |
| earth | The R package for MARS by Milborrow and coauthors, version 5.3.4; the reference in this plan. `Earth` in code font is pymars' class. |
| Super learner | An ensemble that combines several learners with weights chosen by cross-validation; its library is the list of learners it combines |
| X, y, n, p | The covariate matrix and the response vector of a training set, with n cases (rows) and p covariates (columns). In the simulation sections X also denotes a random covariate vector and Y a random response. |
| (z)₊ | max(0, z) |
| Hinge, knot | A function of one covariate x of the form (x − t)₊ (the right hinge) or (t − x)₊ (the left hinge); t is the knot. earth calls a knot a cut. |
| Term | One column of a MARS model: the intercept, a hinge, a linear term, or a product of these |
| Parent | The existing term that a new hinge or linear term multiplies. A case is active for a parent when the parent is nonzero at that case. |
| Linear term | A covariate times a parent, with no knot |
| Pair | The two hinges (x − t)₊ and (t − x)₊ with the same knot and the same parent |
| Degree | The number of covariates multiplied together in a term. A model's degree limit (`max_degree` in pymars, `degree` in earth) caps it; degree 1 gives an additive model, and a term of degree 2 or more is an interaction term. |
| Basis matrix B | The n × M matrix whose columns are the M terms of a model evaluated at the training cases |
| Candidate | A term or pair that the forward pass could add at one step, given by a parent, a covariate and, for a hinge, a knot |
| Term limit | The largest number of terms the forward pass may create: `nk` in earth, `max_terms` in pymars, M_max in formulas |
| Collinearity tolerance | earth's rule that rejects a candidate knot when the existing terms already explain almost all of the new hinge (F5) |
| Forward pass, pruning pass | The two stages of MARS: terms are added greedily, then deleted one at a time |
| RSS | Residual sum of squares on the training data |
| R² | 1 − RSS divided by the total sum of squares of y about its mean. The simulation also uses the population R², defined under [Data-generating processes](#data-generating-processes). |
| GCV | Generalized cross-validation, RSS / (n (1 − C/n)²) when C < n and infinite when C ≥ n, where C is the effective number of parameters of the model ([Code reading against the references](#code-reading-against-the-references) compares the definitions of C) |
| GRSq | earth's generalized R²: 1 − GCV / (GCV of the intercept-only model) |
| Conformance suite | The tests that compare pymars with earth, described under [Correctness against earth](#correctness-against-earth) |
| DGP | Data-generating process of a simulation |
| Arm, cell | In the simulation, an arm is one learner with fixed settings, and a cell is one combination of DGP, sample size and noise level |
| Monte Carlo standard error | The standard error of a simulation summary that comes from the finite number of repetitions |
| BLAS | Basic Linear Algebra Subprograms, the library that numpy and R call for matrix arithmetic |
| SVD, QR | Singular value decomposition and QR decomposition; two ways to solve a least-squares problem |
| GLM | Generalized linear model |
| CI | Continuous integration: tests that run on every change |

## Decisions for you

| ID | Decision | Recommendation |
|---|---|---|
| Q1 | Where the work happens | In the fork (`alejandroschuler/mars`), on branches, one pull request per fix. Open upstream issues first, and upstream pull requests only after you approve them. |
| Q2 | Behavior target | Earth-compatible defaults and rules, each one documented, with every departure from Friedman (1991) listed. Some earth rules are not in the papers: the collinearity tolerance, the automatic linear terms (`Auto.linpreds`) and Fast MARS with 20 parents (`fast.k = 20`). Adopting them is a choice. |
| Q3 | License | earth is GPL-3 and pymars is Apache-2.0. Implement from the papers and from Milborrow's notes, use earth only as a black-box reference in tests, and do not port earth's C source code. This is not legal advice. |
| Q4 | Missing values | Make `allow_missing=True` raise an error until it works. earth does not accept missing predictors either. Fix it later with py-earth's design if you need it. |
| Q5 | Compute for the simulation | Run the current-pymars arms on 15 cells at 200 cases (13 regression cells without D7, and 2 binary cells) and on 4 regression cells at 1,000 cases, about 80 core-hours. Leave out D7 (the DGP with 50 covariates, defined under [Data-generating processes](#data-generating-processes)) for the current code, where one fit would take about 40 minutes at 200 cases. Run the full grid after Phase 3. |
| Q6 | Scope of the clean-up | Either propose upstream to split out the code that does not fit models, or keep a lean fork. [What to remove or split out](#what-to-remove-or-split-out) lists the candidates. |

## Scope, versions and environment

The plan tests two versions of the code. HEAD `d68b54a` of `edithatogo/mars` has the version string 1.0.4, builds with maturin (a Rust build tool), and requires Python 3.10 or later. The PyPI wheel `mars_earth-1.0.4-py3-none-any.whl` (2026-04-17) is pure Python and requires Python 3.9 or later. The fitting files differ in many lines between the two, mostly refactors. They gave identical terms and predictions on 12 test fits (6 seeds at degree 1 and 2), and the conformance suite runs on both.

`Earth.fit` is pure Python on every platform. The compiled Rust module sets `_SUPPORTS_TRAINING_PARITY = False` (`rust-runtime/src/python.rs:88`), so the Rust trainer is never used. `Earth.predict` goes through the Rust runtime on Linux and Windows when the extension is built (`pymars/runtime.py:39`), and never on macOS. The PyPI wheel contains no extension, so pip users always get the Python path; one Linux CI job ([Python and dependency versions](#python-and-dependency-versions)) covers the Rust path.

The reference is R 4.4.3 with earth 5.3.4 (GPL-3), whose pruning pass uses a bundled copy of the leaps code. The written references are Friedman (1991), Friedman (1993), Milborrow's "Notes on the earth package" (2024-10-01, installed with earth as `doc/earth-notes.pdf`), and section 9.4 of *The Elements of Statistical Learning* (Hastie, Tibshirani and Friedman) for the GCV cost of a knot. The archived py-earth package serves only as a reference for missing values and for its GCV formula.

The preliminary work ran on macOS 15 (Darwin 24.6), Apple silicon with 10 cores, and uv 0.11.11, with Python 3.12.13, numpy 2.5.3 (Accelerate BLAS), scipy 1.18.1 and scikit-learn 1.9.1. A source install needs a Rust toolchain because of maturin, and this machine has none, so the venv (`.venv`, ignored by git) imports the worktree through a `.pth` file. The existing test suite passes on Python 3.12.13, 3.13.5 and 3.14.7 with the same counts: 344 passed, 4 skipped and 7 expected failures, in 115 to 131 s.

## Preliminary findings

The evidence comes from scripts in the session scratchpad; [the first appendix](#appendix-reproducing-the-preliminary-findings) says how to reproduce each finding. The Kind column says whether a finding gives a wrong result (or an unusable feature), a departure that changes the fitted model, or a hygiene problem (compatibility, tests, packaging). The finding IDs stay fixed, because the tests, issues and fixes will refer to them.

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| F1 | Sample weights w change the model when they should not. pymars uses `sum(w)` as the sample size in the GCV denominator, in the default term limit and in a cap on the size of a candidate model (`_forward.py:178-185`, `:216-221`, `:667`; `_pruning.py:87`). Weights that sum to 1 make that size 1, so the default term limit becomes 1 and the fit is intercept-only. Weights of 10 make it 2,000 for 200 cases, which shrinks the GCV factor (1 − C/n)⁻² from about 2 to about 1.06 for C near 60, and the fit grows from 9 to 16 terms. Unit weights leave the size at n and still change the model, through the rounding effect in F6: the weighted code computes the RSS differently. earth counts cases and treats equal weights as no weights. | probes `weights_scale`, `unit_weights_trace` | Wrong result |
| F2 | The forward pass picks the candidate with the lowest GCV and uses RSS only to break ties (`_forward.py:706`). All pairs of hinges at one step have the same effective number of parameters C, so GCV ranks them as RSS does, but a single linear candidate raises C by 1 while a pair raises it by 2 + 2d, where d is the GCV penalty. A larger penalty therefore favors linear terms, and it also rules candidates out sooner, since GCV is infinite once C ≥ n. As a result the penalty changes which knots the forward pass finds: penalties 0, 2, 3 and 10 gave four different forward bases. Friedman's forward algorithm and earth pick the largest RSS reduction, and Milborrow's notes state that the penalty does not change the knot positions. | probe `penalty_forward` | Departure |
| F3 | The pruning pass can delete the intercept: `_pruning.py:297-317` protects it only when it is the last term. pymars treats the intercept like any other term. In the four such fits that were traced, the model was full rank when the intercept left, and removing the intercept raised the RSS less than removing any other term (by 0.015 to 3.9), because the other terms, many of them linear terms that also add 1 to the effective number of parameters C, nearly spanned the constant. This happened in 11 of 30 degree-2 fits (100 cases, 5 covariates), and one checked-in fixture has no intercept (`missingness_2d` in `tests/fixtures/reference_regression_cases.json`). Friedman's pruning algorithm and earth never remove it. | probe `structure` | Wrong result |
| F4 | The GCV penalty follows a different convention from earth's. In pymars the effective number of parameters is C = M + d·H (`_util.py:82`), where M is the number of terms, H the number of hinge terms and d the penalty. This charges d once per hinge term, close to Friedman (1991), who charges d once per nonconstant term, but pymars charges nothing for linear terms and counts columns instead of the rank. earth and py-earth charge d once per knot instead: C = M + d·(M − 1)/2, which is d/2 per term after the intercept. When every term after the intercept is a hinge, the penalty part of C at the same d is therefore twice as large in pymars (d per hinge against d/2), and the total cost of a hinge is 1 + d against 1 + d/2. The default d is 3 at every degree in pymars, where Friedman and earth use 2 for additive models (degree 1) and 3 otherwise. The docstring says the formula follows py-earth, which it does not. | code; py-earth `_util.pyx`; `earth:::get.gcv` | Departure |
| F5 | Every forward step adds both hinges of a pair, even when the parent and the variable already appear together in the model. For a parent b, b·(t − x)₊ = b·(x − t)₊ − b·x + t·b, and b·x is already in the span when an earlier pair on the same parent and variable is present, because b·(x − s)₊ − b·(s − x)₊ = b·x − s·b. The second hinge is then an exact linear combination of the first new hinge and existing columns, and 16 of 30 forward bases were rank-deficient. earth adds a single hinge in that case. It also rejects a candidate knot when the existing columns explain more than 99% of the variation of the new hinge column (the tolerance is 0.01 while earth's internal column counter is below 15, then 1e-5). One observed divergence traces to that tolerance: with `trace = 9`, earth rejected pymars' fourth knot (0.148) at each step while its counter ran 3, 4, 6, …, 14, and chose it when the counter reached 16. | probe `structure`; earth trace | Departure |
| F6 | The forward pass stops when the best-GCV candidate fails to lower the RSS by machine epsilon (about 2.2e-16), an absolute number (`_forward.py:240`). For an RSS of 4 or more, adjacent doubles are more than that epsilon apart, so subtracting it changes nothing, and the test only asks whether the computed RSS fell at all; when the true decrease is tiny, that turns on the last bits of two SVD solves. Multiplying y by 1e-9 scales the RSS by 1e-18, so an RSS near 200 becomes about 2e-16, the size of the epsilon itself. Multiplying y by 1e-9 or 1e9, or one column of X by 1e-8 or 1e8, removed one forward term and changed the final model. Unit weights had the same effect. This may explain the cross-run flakiness behind nine commits on 2026-05-03 that widened the reference-test tolerances. earth stops on relative rules: R² changes by less than `thresh = 0.001`, R² reaches 0.999, or GRSq falls below −10. | probes `y_scale_trace`, `x_scale` | Departure |
| F7 | By default `minspan_alpha = endspan_alpha = 0`, so every distinct value is a candidate knot, except the largest when the intercept is the parent. earth and py-earth set the spacing between knots (minspan) and the margin at each end (endspan) with Friedman's equations (43) and (45), in which a small probability α (0.05 by default) sets how strongly the rule guards against knots fitted to runs of noise. pymars counts distinct values where earth counts cases, and it rounds where earth truncates. pymars also lacks earth's larger endspan for interaction terms (`Adjust.endspan`) and does not center the grid of candidate knots. | code | Departure |
| F8 | With `allow_missing=True`, every row with a missing value gets a NaN prediction (85 of 200 training rows in the probe). Hinge and linear terms return NaN for a missing input (`_basis.py:261`, `:433`). Candidates are also scored on different subsets of rows, because each candidate drops its own rows with NaN (`_forward.py:94-99`). | probe `missing` | Wrong result |
| F9 | For binary outcomes, `GLMEarth` and `EarthClassifier` refit the basis with scikit-learn's L2-penalized logistic regression at its default `C=1` (`glm.py:30`, `_sklearn_compat.py:359`); this `C` is scikit-learn's inverse penalty strength, unrelated to the effective number of parameters C. Against an unpenalized fit on the same basis, the coefficients shrank by about 20% and the fitted probabilities moved by up to 0.08. `GLMEarth` has no `predict_proba`, predicts labels, inherits from a regressor and fails 13 scikit-learn checks. `EarthClassifier` treats a three-class response as a regression on the integer label codes. earth fits an unpenalized GLM on the selected terms and uses one indicator response per class. | probes `glm_coef`, `classifier_proba`, `multiclass` | Wrong result |
| F10 | For every candidate (each parent, variable and distinct value), the forward pass rebuilds the whole basis matrix and solves a new least-squares problem by SVD (`_forward.py:662`, `:677-684`). One degree-2 fit with 300 cases and 5 covariates took 15.7 s, made 87,450 calls to `lstsq` and 1.2 million hinge evaluations, and spent 99.6% of its time in the forward pass. There is no updating formula (Friedman 1991, eq. 52) and no Fast MARS (Friedman 1993). The Rust trainer runs the same search, solves the normal equations, and is switched off. See [the fit-time table](#fit-time-against-earth). | cProfile; timing grid | Wrong result (unusable at large n) |
| F11 | The documented class `Earth` fails 5 of 48 checks in scikit-learn's `check_estimator` (version 1.9.1). It sets learned attributes in `__init__`, raises `RuntimeError` instead of `NotFittedError` before fit, does not check the number of features in `predict`, lists its mixins in the wrong order, and passes `check_is_fitted` before fit. `EarthRegressor` and `EarthClassifier` fail only `check_fit2d_predict1d`. The wrappers hide `minspan`, `endspan`, `allow_missing`, `categorical_features` and `feature_importance_type`, and no test runs `check_estimator` on `Earth`. | `sk_checks.py` | Hygiene |
| F12 | The dependency bounds are wrong. The code passes `ensure_all_finite`, a keyword that scikit-learn added in version 1.6, while the package declares `scikit-learn>=1.0.0`; with scikit-learn 1.5.2, `Earth().fit` fails with `TypeError`. The wheel's metadata says Python 3.9 or later, and its PyPI classifiers (the package's metadata tags) list 3.8 to 3.12; HEAD says 3.10 or later but configures ruff and ty for 3.9. matplotlib is a required dependency, and `import pymars` takes 1.2 s because it loads matplotlib. | venv with scikit-learn 1.5.2 | Hygiene |
| F13 | `tests/test_reference_regression.py` compares pymars with its own stored outputs. It never compares the stored terms or coefficients, and it accepts prediction errors up to 0.6 and metric errors up to 1.5. The expected-failure lists in the `check_estimator` tests are stale: at default settings only `check_fit2d_predict1d` fails for either wrapper, and one entry in each list is a garbled name that matches no check. One test in `test_earth.py` turns a failure into an expected failure at run time. | code | Hygiene |
| F14 | The core search is sound. The matched mode (see [Comparison modes](#comparison-modes)) used hinge terms only, degree 1, a pymars penalty equal to half the earth penalty, no R² stopping rule, no Fast MARS, and every case a candidate knot. With them, pymars reproduced earth's forward pass on Friedman #1 (200 cases, 5 covariates): the same 10 knots in the same order, and an RSS path that agreed at all 11 points to a relative difference of 3e-14. At degree 2 both reached R² 0.95330, with GCV 1.4568 against 1.4569, and some knots one data point apart. | `compare_earth.py` | None |
| F15 | With each package at its defaults, the results differ more. In one draw of Friedman #1 (200 cases, 5 covariates, degree 2), the test mean squared error against the true function was 0.78 for pymars and 0.37 for earth. On two other test functions pymars did better or about as well. One draw per case is only a preliminary check; [the simulation study](#statistical-performance-study) designs the full comparison. | `compare_earth.py` | None |
| F16 | Determinism on one machine is good: repeated fits, 1 against 10 BLAS threads, and permuted rows or columns all gave identical models. F6 shows that the stop can still change with rounding, so results may differ between BLAS libraries and platforms. | `blas_det.py`, probes | Hygiene |

### Fit time against earth

Friedman #1 with 10 covariates (5 of them irrelevant), each package at its defaults, on Apple silicon with one thread. The earth time is the minimum of 3 runs and the pymars time is one run. earth times near 0.003 s are close to the timer resolution, so the first ratios are rough.

| n | Degree | earth (s) | pymars (s) | pymars / earth |
|---|---|---|---|---|
| 250 | 1 | 0.003 | 2.3 | about 760 |
| 500 | 1 | 0.003 | 9.4 | about 3,100 |
| 1,000 | 1 | 0.006 | 27.0 | about 4,500 |
| 2,000 | 1 | 0.009 | 108.1 | about 12,000 |
| 250 | 2 | 0.006 | 21.0 | about 3,500 |
| 500 | 2 | 0.012 | 47.2 | about 3,900 |
| 1,000 | 2 | 0.023 | 130.5 | about 5,700 |
| 2,000 | 2 | 0.046 | 502.1 | about 11,000 |

## Code reading against the references

The GCV of a model is RSS / (n (1 − C/n)²) when C < n, where C is the effective number of parameters. The factor (1 − C/n)² is zero at C = n and grows again for C > n, so the expression would favor models with C > n; both programs therefore set GCV to infinity when C ≥ n, which rules such models out. The sources differ only in how they count C. Let M be the number of terms including the intercept, H the number of hinge terms among them, d the GCV penalty, and r the rank of the basis matrix (the number of linearly independent terms); M, H, d and r keep these meanings for the rest of the document.

Friedman (1991) uses C = r + d·(M − 1): the penalty is paid once per nonconstant term, with d = 3 in general and d = 2 for additive models. Section 9.4 of *The Elements of Statistical Learning* writes C = r + d·K instead, where K is the number of knots, so the penalty is paid once per knot, and a knot usually carries two terms. earth and py-earth use the per-knot form with r = M and K = (M − 1)/2, a count they keep even after a single hinge, a linear term or pruning breaks a pair. pymars uses C = M + d·H: per term as in Friedman, but only for hinge terms, and with columns in place of the rank.

The tables below list every place where the fitting code departs from Friedman (1991, 1993) or from earth 5.3.4. "Paper" means Friedman (1991) unless the row says otherwise, code locations are at HEAD, and the last column links the test that checks the row.

### Forward pass

| Topic | Paper | earth 5.3.4 | pymars | Test |
|---|---|---|---|---|
| Parent terms | Every term whose degree is below the maximum | The same when `fast.k = 0`; by default only the best 20, chosen by a priority queue | Every eligible term (`_forward.py:522-524`) | [interaction rules](#interaction-rules) |
| Variables for a parent | Only variables not already in the parent | The same | The same (`:525-528`) | [interaction rules](#interaction-rules) |
| Selection criterion | Largest RSS reduction (Algorithm 2) | Largest RSS reduction; the penalty has no effect | Lowest GCV, RSS as a tie-break (`:706-709`); the penalty changes the forward pass (F2) | [comparison modes](#comparison-modes) |
| Terms added per step | A pair of hinges | A pair; a single hinge when the parent and variable already appear together; a single linear term when the best knot is at the smallest value (`Auto.linpreds`) | Always a pair, or a linear candidate that competes by GCV (`:535-564`) (F5) | [linear terms](#linear-terms) |
| Collinear candidates | Not discussed | A knot is rejected when the existing columns explain more than 99% of the variation of the new column (the 99% becomes 1 − 1e-5 once earth's internal column counter reaches 15) | No check; `lstsq` returns a minimum-norm solution for a rank-deficient basis | [triage](#triage-of-differences) |
| Lower end of the knot range | The first endspan cases are skipped (eq. 45) | The scan over sorted cases stops at index endspan and counts all cases, including those where the parent is zero | endspan distinct values are skipped among the active cases (`:474-476`) | [knot sets](#candidate-knot-sets) |
| Upper end | The last endspan cases are skipped | The largest case is never a knot, and the spacing counts from the top, from a centered offset | For the intercept parent the largest value is dropped (`:481-482`); nothing more for other parents | [knot sets](#candidate-knot-sets) |
| Spacing | Every minspan-th active case (eq. 43) | The same, counted in cases, with the grid centered in the available range | After each allowed knot, the next minspan − 1 distinct values are skipped, starting from the first eligible value (`:509-516`) | [knot sets](#candidate-knot-sets) |
| Default spans | The probability α = 0.05 in eq. 43 and 45 | α = 0.05, with the resulting spans truncated to integers; endspan doubled for interaction terms (`Adjust.endspan = 2`) | α = 0, so both spans are 0; spans rounded instead of truncated (`:420-432`, `:487-507`) (F7) | [knot sets](#candidate-knot-sets) |
| Knot at the smallest active value | Excluded by endspan | Not in the knot scan; evaluated separately as the linear option | Allowed, and the left hinge is then zero on every active case | [edge cases](#edge-cases) |
| Stopping | Term limit | Term limit `nk`; R² changes by less than `thresh`; R² reaches 1 − `thresh`; GRSq below −10; no gain in R² | Term limit, or the best-GCV candidate does not lower RSS by machine epsilon (`:238-253`) (F6) | [invariance](#invariance-tests) |
| Default term limit | Left to the user | min(200, max(20, 2p)) + 1 | min(max(1, ⌊n⌋ − 1), max(21, 2p + 1)), with n replaced by `sum(w)` when weighted, and no cap at 200 (`:216-221`) | [weights](#sample-weights) |
| Updating formula | Eq. 52: after one sort, constant work per knot move for each existing term | Used for unweighted fits; a full regression per knot for weighted fits | None; a full rebuild and an SVD solve per candidate (F10) | [speed](#speed-and-scaling) |
| Fast MARS | Friedman (1993): a priority queue that scores only the most promising parents, with an ageing factor and an interval for re-scoring all variables | `fast.k = 20` parents and `fast.beta = 1` ageing; no re-scoring interval | None | [speed](#speed-and-scaling) |
| Scaling of y | Not discussed | y is scaled to mean 0 and standard deviation 1 during the forward pass, for numerical stability | None; comparisons against an absolute epsilon (F6) | [invariance](#invariance-tests) |
| Ties between candidates | Not discussed | The first candidate found is kept (strict `>`); knots are scanned from the largest value down, predictors in index order | The first found is kept (strict `<`); knots are scanned from the smallest value up, predictors in index order | [ties](#ties) |

### Pruning pass

| Topic | Paper | earth 5.3.4 | pymars | Test |
|---|---|---|---|---|
| Term removed at each step | The one whose removal hurts the fit least (Algorithm 3); all candidates have the same size, so RSS and GCV order them the same way | The one whose removal gives the lowest RSS (leaps backward elimination) | The one whose removal gives the lowest GCV (`_pruning.py:340`); this differs when candidates carry different costs, as hinge and linear terms do | [pruning test](#pruning-of-a-fixed-basis) |
| Intercept | Never removed | Never removed | Can be removed (F3) | [pruning test](#pruning-of-a-fixed-basis) |
| Choice of model size | Lowest GCV | `which.min` over sizes, so ties go to the smaller model | Strict `<` while shrinking, so ties go to the larger model (`:352`) | [pruning test](#pruning-of-a-fixed-basis) |
| Final coefficients | Least squares | `lm.fit` on the selected columns (a rank-revealing QR) | `lstsq` (SVD, minimum norm) | [coefficients](#coefficients-of-fixed-terms) |
| GCV (effective number of parameters C, rank r, penalty d, hinge terms H) | C = r + d·(M − 1), with d = 3, or 2 for additive models | C = M + d·(M − 1)/2 with n the number of cases; `penalty = -1` means GCV = RSS/n | C = M + d·H with n = `sum(w)` when weighted; counts columns, not the rank; no special case for −1 (F4) | [GCV test](#gcv-function) |

### Inputs and outcomes

| Topic | earth 5.3.4 | pymars | Test |
|---|---|---|---|
| Weights | Regresses √w·y on √w times the covariates; ignores weights when it computes minspan and endspan; replaces zero weights by very small ones; treats equal weights as no weights | √w scaling in the solves; the minspan formula uses the weighted count of active cases; zero-weight rows cannot hold knots; `sum(w)` as the sample size (F1) | [weights](#sample-weights) |
| Missing predictors | Not supported (`na.fail`) | NaN predictions, and candidates scored on different row subsets (F8) | [missing values](#missing-values) |
| Categorical predictors | Factors are expanded with R's contrast coding (one treatment dummy for each level after the first) and then treated as numeric 0/1 columns | Label-encoded, with one indicator candidate per level; an unseen level is mapped to the most frequent level without a warning (`_categorical.py:51-54`) | [categorical inputs](#categorical-inputs) |
| GLM | Unpenalized `glm` on the selected columns | L2-penalized scikit-learn fit (F9) | [binary outcomes](#binary-outcomes) |
| Several responses | A shared basis, with RSS and GCV summed over responses; one indicator response per class for a factor with three or more levels | Not supported; three classes are regressed on integer codes (F9) | [binary outcomes](#binary-outcomes) |
| Variable importance | `evimp`: counts and GCV or RSS changes over the pruning sequence | `nb_subsets` is close to earth's count; `gcv` and `rss` use the forward-pass gains of the terms that survive, which differs | Low priority |

## Correctness against earth

Phase 1 builds and runs these tests, and every change in Phases 2 and 3 must pass them before it merges.

### Harness

A Python driver writes each dataset as CSV with 17 significant digits, runs earth through `Rscript`, and reads the results back as JSON. It records the versions of R, earth, Python, numpy, scikit-learn and the BLAS library, and the pymars commit.

From earth the harness collects `dirs`, `cuts`, `selected.terms`, `prune.terms`, `rss.per.subset`, `gcv.per.subset`, `coefficients`, `glm.coefficients`, `rsq`, `grsq`, `termcond`, the fitted values and the predictions on a test set. With `pmethod = "none"` it also computes the RSS path of the forward pass. With `trace = 7` to `9`, earth prints every case that each knot search visits: whether the case was evaluated, the cut, the RSS with that knot, and four flags that give the reason a knot was skipped or rejected. The flag `bx1G` says whether the parent is positive at the case, `CovColG` whether the new column has positive variance, `TolG` whether the knot passes the collinearity tolerance, and `MaxG` whether the RSS reduction is below a safety maximum. The traced RSS is on the scale of the standardized response, so the harness multiplies it by the sample variance of y.

From pymars the harness collects `basis_`, `coef_`, `gcv_`, `rss_`, `record_.fwd_basis_`, `record_.fwd_rss_` and `record_.pruning_trace_*`. pymars has no candidate log, so the harness rebuilds the forward state after each recorded step and calls pymars' own `_generate_candidates` and `_calculate_rss_and_coeffs` on it; this needs no library change. In Phase 2 a small tracing option replaces this.

A prototype (`fit_earth.R`, `compare_earth.py`) exists in the session scratchpad and produced F14 and F15.

### Comparison modes

The matched mode uses settings under which both programs should make the same choices:

- hinge terms only (earth `Auto.linpreds = FALSE`, pymars `allow_linear = False`);
- no Fast MARS (`fast.k = 0`) and no R² stopping rule (`thresh = 0`);
- every case except the largest a candidate knot (`minspan = 1`, and earth `endspan = 1` against pymars `endspan = 0`). For the intercept parent these give the same candidate knots: earth scans every case except the smallest and the largest and evaluates the smallest separately as its linear option, while pymars scans every distinct value except the largest;
- no larger endspan for interaction terms (`Adjust.endspan = 1`);
- a pymars penalty equal to half the earth penalty.

For the intercept parent, these settings make the two programs rank candidates the same way. With hinge terms only, every forward candidate adds two hinges, so all candidates at one step have the same effective number of parameters C, and pymars' GCV ranks them as earth's RSS does. The penalty rule then makes the two GCVs equal for choosing the model size. In a hinge-only model with an intercept, every term after the intercept is a hinge, so the number of hinge terms is H = M − 1. Substituting this into pymars' formula gives C = M + d_pymars·(M − 1), where d_pymars is the pymars penalty. earth's formula for the same model gives C = M + d_earth·(M − 1)/2, where d_earth is the earth penalty. The two are equal for every M when d_pymars = d_earth/2. This equality breaks after a step where earth adds a single hinge and pymars adds a pair whose second hinge is redundant (F5): pymars then has one more column than earth, so its M and its C are larger. A second source of redundant columns is the knot at the smallest value, where pymars' left hinge is zero on every case.

In the matched mode, expect identical choices except in these cases:

- earth's collinearity tolerance or its single-hinge rule applies (F5);
- the step is a near-tie ([Ties](#ties) defines them);
- the parent is not the intercept (degree 2 and above), because the two programs treat the ends of the knot range differently (see the forward-pass table);
- n is small. pymars never picks a candidate with infinite GCV, so its forward pass stops once every candidate's effective number of parameters C is at least n. With n = 20 and a pymars penalty d_pymars = 1 (earth's penalty 2 halved), a model with K pairs (one knot each) has M = 1 + 2K terms and H = 2K hinges, so C = M + H = 1 + 4K. C < 20 needs K ≤ 4 (K = 5 gives C = 21), so pymars stops at 1 + 2·4 = 9 terms; its default term limit, min(19, 21) = 19, does not bind. earth stops by its own rules: in a check with 20 cases and `thresh = 0` it stopped at 7 terms because no new term raised R² (its termination code 6). The two forward passes can therefore end at different sizes;
- pruning reaches a tie between model sizes, or pymars removes the intercept (F3).

The earth-compatible mode exists after Phase 2, when pymars gets an option that applies earth's rules and defaults. With each package at its defaults, expect identical choices except at near-ties.

The defaults mode runs each package at its own defaults. The structures will differ, so this mode compares only R², GCV, test error and the number of terms. [The simulation study](#statistical-performance-study) extends it to many datasets.

### Test datasets

All datasets are deterministic and are stored with the fixtures. Continuous covariates have no repeated values unless the case is about ties.

| Case | Content | Purpose |
|---|---|---|
| S01 | One covariate, two true knots, 200 cases | Knot recovery; minspan 1, 5 and automatic; endspan 1, 10 and automatic |
| S02 | S01 with 20 and 50 cases | Span formulas and stopping rules at small n |
| S03 | Three covariates: (x1)₊, \|x2\|, and a linear term in x3 | Pairs against single hinges; linear terms |
| S04 | Friedman #1 with 5 and 10 covariates, 200 and 1,000 cases | Interactions and irrelevant covariates, degree 1 and 2 |
| S05 | A pure degree-2 hinge product | Interaction search; `Adjust.endspan` |
| S06 | A degree-3 product | Degree 3 |
| S07 | A linear truth | `Auto.linpreds` against pymars' linear candidates |
| S08 | Integer covariates with 10 levels | Repeated x values: distinct values against cases |
| S09 | A 0/1 covariate and a 4-level categorical covariate | Categorical coding |
| S10 | A duplicated column, a near-duplicate (x1 plus noise with standard deviation 1e-9), a constant column | Tie-breaks across predictors; collinearity |
| S11 | 3, 5, 8 and 12 cases | Degenerate sizes |
| S12 | x times 1e-8 and 1e8, x plus 1e6, y times 1e-9 and 1e9 | Invariance to scale and shift |
| S13 | Random positive weights, integer weights, weights with zeros, equal weights | Weights |
| S14 | A binary response from a logistic truth, 5 covariates, 500 cases | GLM refit |
| S15 | 200 small draws from the simulation DGPs | Rates: the share of fits that agree, the step of the first divergence, and its cause |

### What is compared, and the tolerances

Below, κ(B) is the 2-norm condition number of the basis matrix B.

| Quantity | When compared | Tolerance |
|---|---|---|
| Parent, variable, direction and knot of each forward step | Matched and earth-compatible modes, up to the first near-tie | Exact. Knots are observed data values, so they are compared as numbers. |
| RSS after each forward step | Same structure and κ(B) at most 1e6 | Relative 1e-8 |
| Term removed at each pruning step, `rss.per.subset`, `gcv.per.subset` | Same forward basis | Exact term; relative 1e-8 |
| Selected terms | Matched and earth-compatible modes | Exact set |
| Coefficients | Same terms and κ(B) at most 1e5 | Normwise relative 1e-6: the norm of the difference between the two coefficient vectors over the norm of earth's coefficient vector β (one coefficient near 0 has no useful relative bound); above κ(B) = 1e5, compare fitted values instead |
| GCV, R², GRSq | Same terms and κ(B) at most 1e6, after the penalty conversion | Relative 1e-8 for GCV; absolute 1e-8 for R² and GRSq |
| Fitted values and predictions on new data | Same terms and κ(B) at most 1e6 | Largest absolute difference at most 1e-8 × sd(y) |
| GLM coefficients and fitted probabilities | Same columns | Relative 1e-5; absolute 1e-7 |
| Anything, when the structures differ (defaults mode) | Always | No tolerance; report R², GCV, test error and the number of terms |

Above these κ(B) limits the harness compares fitted values with a tolerance scaled by κ(B), and labels any difference `numeric`.

These values allow for the fact that the two programs use double precision but different linear algebra: earth uses orthogonal updates and the QR decomposition in `lm.fit`, pymars uses SVD, and earth standardizes y. For identical terms, the usual rounding-error bounds for a backward-stable least-squares solve are normwise. With κ = κ(B), the condition number of the basis matrix, they are of order κ·u·‖y‖ for the fitted values, 2κ·u·‖y‖/‖e‖ in relative terms for the RSS (the factor 2 because the RSS is a squared norm), and κ·u + κ²·u·tan θ in relative terms for the coefficients. Here u ≈ 1.1e-16 is the unit roundoff (half the machine epsilon of F6), e the residual vector, ŷ the fitted values and tan θ = ‖e‖/‖ŷ‖. In Friedman #1 with 200 cases, the ratio of ‖y‖ to the residual norm ‖e‖ is about 14, so the RSS bound stays below 1e-8 up to κ of about 3e6, and the coefficient bound is about 8e-8 at κ = 1e5. The fitted-value bound must also be converted to the largest entry in units of sd(y): since ‖y‖ is about √n times the root mean square of y, which is about 3·sd(y) here, the factor is about 43 at n = 200, and the bound stays below 1e-8·sd(y) up to κ of about 2e6. The table's limits sit below these values. The F14 run showed 3e-14, far inside them, while a modeling difference such as one different knot changes the RSS by far more.

### Ties

When x has repeated values, pymars scores distinct values and earth scans cases. With `minspan = 1` the candidate knot values are the same. With a larger minspan they differ by design; S08 measures that difference, and the report documents it.

Exact ties in the criterion come from duplicated columns or symmetric designs. The two programs break ties between knots in opposite directions: earth keeps the largest tied knot and pymars the smallest. Both keep the predictor with the lower index. S10 checks the predictor rule, and the earth-compatible mode adopts earth's knot rule.

Near-ties come from rounding. At each forward step the harness takes the best and the second-best candidate RSS from both logs. If they differ by less than 1e-7 times the RSS before the step, the step counts as a near-tie. The two programs' values for one candidate RSS may differ by up to 1e-8 of that RSS, which is at most 1e-8 of the RSS before the step, so a flip in order needs a gap below about 2e-8 of the RSS before the step; the threshold covers that five times over. Either choice then passes, and the structural comparison of that fit stops, because the paths after two different choices cannot be compared. The harness reports how many fits stopped this way. If more than 5% of the fits in S15 stop at a near-tie, the tolerance or the designs need review.

### Triage of differences

Each difference goes into `validation/DIFFERENCES.md`. An entry gives the dataset, the step, both choices, both candidate RSS values, earth's flags, and one of five labels:

- `rule`: an earth rule that pymars lacks, such as the collinearity tolerance. Decision Q2 settles whether pymars adopts it or documents it.
- `bug`: a pymars bug. It gets a failing test in Phase 1 and a fix in Phase 2.
- `quirk`: an earth behavior that looks accidental, such as the lower endspan counting inactive cases. The report documents it, and pymars copies it only if decision Q2 says so.
- `tie`: a near-tie.
- `numeric`: a numerical difference, such as a rank-deficient solve.

Phase 1 ends when every difference in S01 to S15 has a label, and every `bug` item has a failing test, marked as an expected failure with its finding ID.

### Component tests

Each test covers one component, so that a failure points at one place.

#### GCV function

pymars' GCV is compared with `earth:::get.gcv` over a grid: RSS values, 1 to 41 terms, penalties 0 to 6 and −1, and n from 10 to 100,000, after the penalty conversion, which assumes a hinge-only model (the number of hinge terms H = M − 1). Two parts of the grid behave differently by design. With penalty −1, earth sets the effective number of parameters C to 0, so its GCV is RSS/n, while pymars has no special case, so those cells are expected failures. Where C ≥ n, both programs return infinity.

#### Candidate knot sets

An earth log with `trace = 9` gives the set of evaluated cuts for each knot search. The test compares it with pymars' `_get_allowable_knot_values` for the same parent and variable, over a grid: minspan automatic, 1, 3, 10 and −3 (in earth a negative minspan asks for at most that many evenly spaced knots per variable, here 3); endspan automatic, 1 and 5; degree 1 and 2; `Adjust.endspan` 1 and 2; and 20, 200 and 2,000 cases.

#### Pruning of a fixed basis

earth's forward basis has no exact collinearity. The test builds the same terms in pymars and runs both pruning passes on them: earth's internal `earth:::pruning.pass` and pymars' `PruningPasser`. It compares the term removed at each step, `rss.per.subset`, `gcv.per.subset` and the selected size.

#### Coefficients of fixed terms

pymars' least-squares coefficients are compared with `lm.fit` on the same columns.

#### Prediction at new points

The basis is evaluated at new points, including points outside the training range, and compared with `predict.earth`.

#### Linear terms

S07 runs with earth `Auto.linpreds = TRUE` against pymars `allow_linear = True`.

#### Interaction rules

No variable may appear twice in a product, linear terms can be parents, and `Adjust.endspan` applies to interaction terms.

### Binary outcomes

earth fits `earth(x, y, degree = 2, glm = list(family = binomial))`. Its terms come from the least-squares passes on the 0/1 response, and `glm` then fits the selected columns. The pymars counterparts are `GLMEarth(family="logistic")` and `EarthClassifier`. The harness compares three things:

1. The selected terms, which should match the regression fit on the 0/1 response, so the rules of [the comparison modes](#comparison-modes) apply.
2. The GLM coefficients and fitted probabilities for the same columns, against earth's `glm.coefficients` and against an unpenalized fit in Python, with relative tolerance 1e-5. In scikit-learn 1.9 an unpenalized fit needs `C=np.inf`, because the `penalty` argument is deprecated since version 1.8.
3. The held-out log loss, in the simulation.

The tests also cover string, boolean and {−1, 1} labels; separation, where earth warns that fitted probabilities are numerically 0 or 1; and three classes, against earth's model with one indicator response per class. With the current code the second comparison fails by design (F9).

### Sample weights

Fits must be invariant to the scale of the weights: the fit with weights w must equal the fit with weights c·w for every constant c in {1e-6, 1/n, 10, 1e6}. The reason is that multiplying w by c > 0 multiplies every weighted RSS by c, so the minimizing coefficients do not change. Every comparison then stays the same as long as n counts cases, every threshold is relative (the absolute epsilon of F6 is not), and the span formulas count cases (pymars' minspan formula uses the weighted count today). Unit weights must give the same model as no weights, and zero weights the same coefficients as removing those rows. The GCV rule for zero-weight rows needs a decision, since earth counts all rows.

In the matched mode the harness compares weighted fits with earth for random positive weights, integer weights and weights with zeros. earth's weighted forward pass solves a full regression at each knot, so it is slow but exact.

Integer weights and repeated rows must give the same coefficients for a fixed basis, since Σ w_i (y_i − B_iβ)² over the rows equals the plain RSS of the data with row i repeated w_i times; here B_i is row i of the basis matrix B and β the coefficient vector. The whole MARS fit need not agree, because the spans and the GCV count cases. With the current code, the invariance tests fail (F1).

### Missing values

earth does not accept missing values in x, so it cannot be the reference here. With no missing values, `allow_missing=True` must give the same model as `False`; this passes today. Predictions must be finite for rows with missing values; this fails today (F8). When values are missing completely at random in an irrelevant covariate, that covariate and its missingness indicator must be selected no more often than the same covariate when it is fully observed.

The semantics check follows py-earth's published design, in which a hinge on a covariate is multiplied by an indicator that the covariate is present, and missingness indicators can enter as terms. If py-earth builds in an old Python (for example 3.8 through uv), the harness compares with it directly. If not, it compares with earth on an augmented design: missing values set to 0, presence indicators added as columns, and the degree raised by one so that earth can multiply a hinge by an indicator. That design can represent the same functions, although earth's search over it differs, so this comparison checks predictions and model quality rather than exact terms.

### Categorical inputs

pymars with `categorical_features=[j]` is compared, in the matched mode, with earth given one-hot dummies for the same column j (one 0/1 column per level). The spans then match, but the candidate costs do not: in pymars an indicator raises C by 1 and a hinge pair by 2 + 2d, and pymars ranks candidates by GCV (F2), while earth ranks them by RSS. Exact agreement is therefore expected only after Phase 2 moves pymars to RSS selection; before that, differences get the label `rule`. With treatment dummies earth has no column for the reference level, so the paths can also differ for that reason. At predict time pymars maps an unseen level to the most frequent level without a warning; decide whether that should be an error or a warning. The tests also pass pandas categorical and string columns through `Earth` and `EarthRegressor`.

### Edge cases

| Case | earth (reference) | pymars today | Wanted after the fixes |
|---|---|---|---|
| Constant column | Never used | Never used; the same predictions | The same |
| Duplicated column | The lower index is used | The lower index is used; the same predictions | The same |
| Near-duplicate column | The collinearity tolerance applies | Minimum-norm solution; coefficients of similar size | Tolerance as in earth |
| Constant y | Intercept only | To measure | Intercept only |
| 3, 5, 8 and 12 cases | To measure | 2, 4, 3 and 4 terms. With 5 cases no hinge can enter at all: a model with a hinge has M ≥ 2 terms and H ≥ 1 hinge terms, so its effective number of parameters C = M + 3H is at least 5 = n and its GCV is infinite. The 4 terms are the intercept and three linear terms, which leaves one residual degree of freedom. | As in earth, and never more terms than the cases support |
| n < p; p = 1; p = 200 | The term limit is capped at 201 | No cap on the term limit | A cap as in earth |
| Scale and shift (S12) | Invariant | The model changes (F6) | Invariant |
| One outlier in x at 1e6 | To measure | The fit completes, with 7 terms | As in earth |
| Inf or NaN when missing values are not allowed | An error | A scikit-learn error | A clear error |
| One million rows | To measure | To measure (memory of the rebuild per candidate) | Memory linear in n |

### Invariance tests

These property-based tests (with the hypothesis library) need no reference:

- Row order does not change the model.
- Column order changes only the labels, apart from ties.
- Shifting and scaling a covariate x to a + s·x, with shift a and scale s > 0, gives the same terms with transformed knots, because (a + s·x − (a + s·t))₊ = s·(x − t)₊, and the same predictions up to rounding (relative 1e-8; after a shift of 1e6, a covariate on [0, 1] keeps about 10 significant digits, which is enough). With s < 0 the hinges mirror, since the same expression equals |s|·(t − x)₊; this holds only apart from ties and from the rules that treat the two ends of the range differently (the largest value is never a knot, and the spacing grid starts from one end).
- Shifting and scaling y to a + s·y, with shift a and scale s ≠ 0, gives the same terms: the intercept absorbs a, every RSS scales by s², so the rankings and the relative stopping rules do not change, and the other coefficients scale by s. The predictions transform the same way.
- Multiplying the weights by a constant gives the same model.
- Adding a constant column or a duplicated column leaves the predictions unchanged, provided `max_terms` and both spans are fixed, because p enters the default term limit and the span formulas.

With the current code, the tests on the scale of y, the scale of x and the scale of the weights fail (F1, F6).

## Statistical performance study

This section follows the design-and-report-simulations skill: claims first, then goals, mockups of the tables and figures, the DGPs, the build and the pilot. The learner settings follow the supervised-learning skill.

### Claims

The claims use excess risk. For a function f̂ fitted to training data, the excess risk R(f̂) = E[(f̂(X) − f(X))²] is the mean squared difference between the fit and the true regression function f at a new covariate draw X; the expectation averages over the new X with the fit held fixed. For binary outcomes the analogue is the excess log loss, defined under [Performance measures](#performance-measures).

- The conformance claim: in the matched mode, pymars reproduces earth's forward pass and pruning pass, except at documented rule differences and near-ties. The evidence is the conformance suite, so this claim needs no simulation.
- The gap claim: with each package at its defaults, current pymars has a higher excess risk than earth when the truth has interactions, and the gap shrinks when pymars runs with earth's settings (the P-ear arm).
- The parity claim: after Phases 2 and 3, pymars at its defaults has an excess risk within a factor of 1.05 of earth's, in either direction, for every DGP and sample size in the grid, and an excess log loss within 5% for binary outcomes. Its boundary is still to be found. Two likely places are small samples with many irrelevant covariates, where the collinearity rule and Fast MARS matter most, and correlated covariates. Where the results contradict the parity claim, the claim gains a condition, and the report shows those cells as prominently as the others.
- The sparsity claim: on pure noise and with many irrelevant covariates, pymars selects no more terms and no more irrelevant covariates than earth.
- The speed claim: fixed pymars fits within a stated factor of earth's time. The evidence is the benchmark ([Speed and scaling](#speed-and-scaling)).

### Arms and DGPs at a glance

The arms are the learners under comparison; [Learners and settings](#learners-and-settings) gives their full settings.

- E-def is earth at its defaults with degree 2, the reference.
- E-pym is earth moved to pymars' settings, an ablation.
- P-cur is pymars as users get it today, with degree 2.
- P-ear is pymars moved to earth's settings within its current options, an ablation.
- P-fix is pymars after Phases 2 and 3, at its defaults.
- OLS (ordinary least squares with the covariates as linear terms) is a floor, and HGB (scikit-learn's histogram gradient boosting) is the Python learner the skill uses today in place of MARS.

The regression DGPs are D1 (linear), D2 (additive hinges), D3 (additive smooth), D4 (Friedman #1), D5 (a hinge interaction), D6 (pure noise), D7 (D3 with 50 covariates) and D8 (D4 with correlated covariates). D3-bin and D4-bin are binary versions of D3 and D4. [Data-generating processes](#data-generating-processes) defines them.

### Goals

For the gap claim, the measure is the ratio of excess risks, P-cur over E-def, with the ablation arms P-ear and E-pym. It runs on D1 to D8 at both noise levels with 200 cases, and on D3, D4, D5 and D8 at the low-noise level with 1,000 cases, to check that the gap persists at a larger n. The contrast counts as resolved when the mean log ratio is at least 3 Monte Carlo standard errors away from 0. The full run is sized for 5 standard errors (see [Pilot and number of repetitions](#pilot-and-number-of-repetitions)), so that a true gap of 5 standard errors shows at 3 or more with probability about 0.98: the estimate is then approximately normal with mean 5 and standard deviation 1 in units of the standard error, and P(Z ≥ −2) = Φ(2) ≈ 0.977.

For the parity claim, the measure is the same ratio, P-fix over E-def, at 200, 1,000 and 5,000 cases, tested for equivalence with a margin Δ = log 1.05 on the log scale. Equivalence is shown when the interval for the mean log ratio, 3 Monte Carlo standard errors on each side, lies inside ±Δ, that is when the interval for the ratio lies inside [1/1.05, 1.05] ≈ [0.952, 1.05]. The binary DGPs D3-bin and D4-bin get the same test on the excess log loss.

For the sparsity claim, the measures are the median number of selected terms, the share of fits that use at least one irrelevant covariate, and the mean number of irrelevant covariates used, on D4, D6, D7 and D8 at 200 cases, where overfitting shows most.

### Mockups

The captions come first, and every display names the claim it serves. To fill these displays, the design needs interaction DGPs (for the gap claim), a noise DGP and a DGP with many covariates (for the sparsity claim), two noise levels, three sample sizes, the two ablation arms, and binary versions of one additive DGP and one interaction DGP.

#### Ratio table

Caption: excess risk of the current pymars and of the two ablation arms relative to earth at 200 cases and the low-noise level, as a ratio of geometric means over repetitions, with an interval of ±3 Monte Carlo standard errors on the log scale, the width the decision rules use. Values above 1 favor earth. The first column tests the gap claim, and the two ablation columns show whether settings or rules cause the gap. The D7 row has no P-cur or P-ear value, because the current code is too slow there. The appendix repeats the table for the four cells run at 1,000 cases.

| DGP | P-cur / E-def | P-ear / E-def | E-pym / E-def |
|---|---|---|---|
| D1 linear | | | |
| D2 additive hinges | | | |
| D3 additive smooth | | | |
| D4 Friedman #1 | | | |
| D5 hinge interaction | | | |
| D6 pure noise | | | |
| D7 many covariates | | | |
| D8 correlated | | | |

#### Equivalence figure

Caption: log ratio of excess risk, P-fix over E-def, with intervals of ±3 Monte Carlo standard errors, the width the equivalence rule uses; one row per DGP, one panel per sample size, point shape by noise level. A shaded band marks the equivalence margin of ±log 1.05. The figure supports the parity claim across the whole grid.

#### Per-repetition box plots

Caption: box plots of the per-repetition log ratios against E-def, at 200 cases for P-cur, P-ear, E-pym and P-fix. They show whether a few bad fits drive the means in [the ratio table](#ratio-table) and in [the equivalence figure](#equivalence-figure).

#### Selection table

Caption: model size and selection on D4, D6, D7 and D8 at 200 cases: the median number of terms with its interquartile range, the share of fits that use any irrelevant covariate, and the mean number of irrelevant covariates used, for E-def, P-cur and P-fix. The table supports the sparsity claim.

#### Binary-outcome table

Caption: the ratio of excess log loss (pymars over earth) and the calibration slope (both defined under [Performance measures](#performance-measures)), for D3-bin and D4-bin, for `EarthClassifier` and `GLMEarth` as they are today (200 cases) and for P-fix (all three sample sizes). The table supports the parity claim for binary outcomes and measures the effect of F9 (the penalized refit).

Appendix tables give every cell with its Monte Carlo standard error, and the failure counts.

### Data-generating processes

Covariates are X ~ Unif[0, 1]^p, independent, with p = 10 unless stated; D8 uses a Gaussian copula with latent correlation 0.6 mapped to uniform margins, which gives a correlation of (6/π)·arcsin(0.3) ≈ 0.58 between the uniform covariates. The outcome is Y = f(X) + σε, where f is the regression function, ε ~ N(0, 1) and σ is the noise standard deviation. For each DGP, σ is set so that the population R², Var f / (Var f + σ²), is 0.8 (low noise) or 0.3 (high noise); here Var f is the variance of f(X). Solving for the noise standard deviation gives σ = sd(f)·√((1 − R²)/R²), where sd(f) = √(Var f) is the signal standard deviation, which is sd(f)/2 at R² = 0.8 and about 1.53·sd(f) at R² = 0.3. The two levels bracket easy and hard problems, since a DGP on which every method is perfect, or every method fails, cannot separate the methods. D6 has population R² = 0 by construction and uses σ = 1, so it has one noise level only.

| DGP | f(x) | Purpose |
|---|---|---|
| D1 linear | x1 + 2 x2 − x3 | Linear terms; the nonlinear share of Var f is 0 |
| D2 additive hinges | 2 (x1 − 0.3)₊ − 3 (x1 − 0.7)₊ + 2 (0.5 − x2)₊ | A truth inside the MARS class; knot recovery |
| D3 additive smooth | sin(2π x1) + 2 (x2 − 0.5)² + exp(x3) | Curvature approximated by hinges |
| D4 Friedman #1 | 10 sin(π x1 x2) + 20 (x3 − 0.5)² + 10 x4 + 5 x5 | The standard benchmark: a smooth interaction and 5 irrelevant covariates |
| D5 hinge interaction | 4 (x1 − 0.4)₊ (x2 − 0.5)₊ + x3 | An interaction inside the MARS class; `Adjust.endspan` |
| D6 pure noise | 0 | Pruning and overfitting |
| D7 many covariates | D3 with p = 50, so 47 irrelevant covariates | Selection among many covariates; run time in p |
| D8 correlated | D4 with correlated covariates | Collinearity and near-ties |
| D3-bin | μ(X) = expit(λ (f_D3(X) − E[f_D3(X)])) | A binary outcome with additive structure |
| D4-bin | The same construction with f_D4 | A binary outcome with an interaction |

In D3-bin and D4-bin, Y is Bernoulli with probability μ(X) = P(Y = 1 given X), f_D3 and f_D4 are the regression functions of D3 and D4, and expit(z) = 1/(1 + e^(−z)). The scale λ is the largest value for which the 1st and 99th percentiles of μ(X), computed on the diagnostic draw, lie inside [0.05, 0.95]. Because expit is increasing and λ > 0, those percentiles are expit(λ·q₀₁) and expit(λ·q₉₉), where q₀₁ and q₉₉ are the 1st and 99th percentiles of f(X) − E[f(X)]. Both move away from 0.5 as λ grows, so λ = logit(0.95) / max(|q₀₁|, q₉₉), with logit(0.95) = ln 19 ≈ 2.94. On a draw of one million this gives λ ≈ 1.60 for D3 and 0.28 for D4. Bounding the extremes of f instead would give about 1.26 and 0.19 and would leave more probabilities near 0.5: the share of μ(X) in [0.3, 0.7] would be 51% instead of 41% for D3, and 62% instead of 45% for D4.

Before the pilot, one draw of one million cases per DGP gives the diagnostics to report: the signal variance Var f, the noise standard deviation σ, the population R², the share of Var f that the best linear approximation of f explains, and for the binary DGPs the range of the true probability μ(X).

The design is fully factorial over DGP, sample size (200, 1,000 and 5,000 cases) and noise level for the regression DGPs. That gives 7 × 3 × 2 = 42 cells, plus 3 cells for D6 (one noise level) and 2 × 3 = 6 binary cells, 51 cells in all.

### Learners and settings

The supervised-learning skill requires every hyperparameter to be fixed or tuned, with a reason. All are fixed here, because the claims are about the learner as a library would include it, and the skill states that MARS works well with one set of hyperparameters.

| Arm | Settings | Reason |
|---|---|---|
| E-def | earth with `degree = 2` and all other arguments at their defaults: penalty 3, `nk` = min(200, max(20, 2p)) + 1, `thresh = 0.001`, automatic minspan and endspan, `fast.k = 20`, `pmethod = "backward"` | The reference. Degree 2 follows the skill's advice to set the interaction degree high enough and keep pruning on. |
| E-pym | earth with `degree = 2` and pymars' settings: penalty 6 (the per-knot value that equals pymars' 3 per hinge), `thresh = 0`, `minspan = 1`, `endspan = 1`, `fast.k = 0`, `Adjust.endspan = 1` | Ablation: earth moved to pymars' settings. The match is approximate, because pymars charges nothing for a linear term and earth charges d/2. |
| P-cur | pymars `Earth(max_degree=2)`, other arguments at their defaults | The library as users get it today |
| P-ear | pymars with earth's settings within the current options: `max_degree=2`, `penalty=1.5`, `minspan_alpha = endspan_alpha = 0.05` | Ablation: pymars moved to earth's settings. The default term limits already agree in this grid (21 terms at p = 10, 101 at p = 50). |
| P-fix | pymars after Phases 2 and 3, at its defaults | The candidate for the skill |
| OLS | Linear regression with the covariates as linear terms | A floor, as the skill advises |
| HGB | `HistGradientBoostingRegressor` or `HistGradientBoostingClassifier`, learning rate 0.1, up to 1,000 iterations with early stopping, fixed `random_state` | The Python learner the skill uses today in place of MARS |

For binary outcomes the arms are E-def with `glm = list(family = binomial)`, `EarthClassifier` and `GLMEarth` as they are today, P-fix, logistic regression with the covariates as linear terms, and the HGB classifier.

The supervised-learning skill's edge rule says that a setting chosen by cross-validation must not sit at the edge of its tuning grid; if it does, the grid moves in that direction. It applies to HGB, whose early stopping acts as a tuning grid for the number of iterations: if it often reaches 1,000 iterations, the learning rate goes up instead of the cap.

An optional tuned arm, reported only in the appendix, runs 5-fold cross-validation over degree 1, 2 and 3 and penalty 2, 3 and 4, for E-def and P-fix on D4 (Friedman #1) and D5 (hinge interaction). It tests the skill's statement that one setting works well. Degree 1 is a hard limit and exempt from the edge rule. If most repetitions choose degree 3, penalty 2 or penalty 4, the grid moves in that direction.

### Performance measures

The first measure is the failure count: attempted and completed fits for each arm and cell, with the error type.

The primary measure is the excess risk R(f̂), estimated on a fresh test draw of 10,000 points in each repetition. For two arms in repetition i, for example P-fix with fit f̂_P,i and E-def with fit f̂_E,i, the log ratio is g_i = log R(f̂_P,i) − log R(f̂_E,i), with natural logarithms. It is paired, because both arms see the same training data. Let n_sim be the number of repetitions, and let ḡ and s_g be the mean and standard deviation of the log ratios g_i over them. The report gives the ratio exp(ḡ) and the interval exp(ḡ ± 3 s_g/√n_sim), whose half-width is the 3 Monte Carlo standard errors that the decision rules use. Since ḡ is the mean of log R(f̂_P,i) minus the mean of log R(f̂_E,i), exp(ḡ) is the ratio of the geometric means of the two excess risks. The log scale turns the contrast into a ratio and tames the long right tail of squared errors. The excess risk is also the gain in held-out squared error over the true function. Writing Y − f̂ = (Y − f) + (f − f̂) gives (Y − f̂)² − (Y − f)² = 2(Y − f)(f − f̂) + (f − f̂)², and the cross term has expectation 0 because E[Y − f(X) given X and the training data] = 0; so E[(Y − f̂(X))² − (Y − f(X))²] = E[(f̂(X) − f(X))²].

For binary outcomes, μ(X) is the true probability of Y = 1 and μ̂(X) the fitted one. The excess log loss is E[KL(μ(X) ‖ μ̂(X))], where KL is the Kullback-Leibler divergence between two Bernoulli distributions. It equals the expected log loss of μ̂ minus that of μ, because given X the difference of the two log losses has expectation μ log(μ/μ̂) + (1 − μ) log((1 − μ)/(1 − μ̂)), which is that divergence. The excess Brier score E[(μ̂(X) − μ(X))²] follows from the same add-and-subtract step as the excess risk; the appendix tables report it. The calibration slope is the coefficient from a logistic regression of Y on the logit of the fitted probability, logit(μ̂(X)), in the test set, where logit(q) = log(q/(1 − q)); a slope of 1 means good calibration. Fitted probabilities are clipped to [1e-6, 1 − 1e-6], and the report gives the number of clipped values.

The secondary measures are the number of selected terms, the use of irrelevant covariates (in D4 Friedman #1, D7 with many covariates and D8 with correlated covariates), and the fit time of each arm.

### Build

The layout follows the skill's Python scaffold: `validation/sims/dgps.py`, `learners.py`, `run.py` and `summarize.py`. One pipeline runs the pilot, the full run and every display.

Each dataset's seed comes from its name (DGP, sample size, noise level and repetition) through numpy's `default_rng`, and the test set comes from a second stream of the same seed. No seed depends on the position in a loop, so the pilot repetitions become the first repetitions of the full run.

earth runs through `Rscript` in blocks. Python writes the datasets for a block of repetitions, one R process fits them all and writes the predictions, and the R start-up time (about 0.3 s) is paid once per block.

Predictions are cached on local disk, in a directory that git ignores. The cache key hashes the training and test data, the learner name and settings, the learner's source code, and the package versions (the pymars commit and the earth version). The per-repetition results on disk are the record; the cache only saves time.

Each fit gets one BLAS thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` set to 1), and the cells run in parallel with joblib.

### Pilot and number of repetitions

The pilot runs in three steps:

1. Two repetitions per cell check that the code runs.
2. One large draw per DGP gives the diagnostics listed under [Data-generating processes](#data-generating-processes).
3. 100 repetitions size the full run. For the gap claim they run at 200 cases on D4 (Friedman #1) and D5 (hinge interaction) at both noise levels. For the parity claim they run after Phase 3 at 1,000 cases on D3, D4, D5 and D7 at both noise levels. The sparsity claim is descriptive and uses the repetitions of the other contrasts.

The sizing uses the skill's `pilot_check.py`. Let n_pilot be the number of pilot repetitions, and let ḡ_p and s_p be the pilot mean and standard deviation of the log ratios g_i; they estimate the true mean of g_i and its standard deviation. After n_sim repetitions, the Monte Carlo standard error of the full-run mean ḡ is about s_p/√n_sim.

A difference contrast (the gap claim) is decided by the 3-standard-error rule under [Goals](#goals), and the full run is sized so that the planned gap is k = 5 Monte Carlo standard errors: |ḡ_p| ≥ k·s_p/√n_sim. The standard error falls as n_sim grows, so the smallest adequate n_sim is where the inequality binds; squaring and solving gives n_sim ≥ (k·s_p/|ḡ_p|)². The plan also replaces |ḡ_p| by the lower bound max(0, |ḡ_p| − 1.96·s_p/√n_pilot), and `pilot_check.py` reports the resulting n_sim as `n_sim_safe`; this guards against a pilot that overstates the gap by chance.

An equivalence contrast (the parity claim) is shown when the interval ḡ ± k·s_g/√n_sim, with k = 3, lies inside ±Δ, the equivalence margin; that is, when |ḡ| + k·s_g/√n_sim ≤ Δ. For a pilot gap |ḡ_p| below Δ this gives n_sim ≥ (k·s_p/(Δ − |ḡ_p|))². The plan sizes for an estimated gap of up to Δ/2, which gives n_sim ≥ (2k·s_p/Δ)². With the equivalence margin Δ = log 1.05 ≈ 0.0488, a pilot standard deviation s_p = 0.15 needs about 340 repetitions and s_p = 0.3 about 1,360; doubling s_p quadruples n_sim. The chance of success depends on the true mean of g_i: at that size it is about 0.997 when the true gap is 0 and about 0.5 when it is Δ/2. If the pilot's |ḡ_p| is already Δ or more, no n_sim can show equivalence, and the parity claim gains a condition for that cell. Four multipliers appear in this plan, each for its own rule: 3 for the displayed intervals and for both decision rules, 1.96 for the pilot's lower bound on the gap, k = 5 for sizing the difference contrasts and k = 3 for sizing the equivalence contrasts.

The pilot's own statistic z = ḡ_p / (s_p/√n_pilot), the pilot mean divided by its standard error, decides whether it can size the run. For a difference contrast, the lower bound above is zero exactly when |z| ≤ 1.96, and then `n_sim_safe` is infinite. So if |z| is below about 2, the pilot cannot tell the gap from zero; then the pilot grows or the design changes, and the report says so.

### Compute budget

earth takes 0.003 to 0.05 s per fit, which is negligible. The current-pymars arms P-cur and P-ear take about 15 s at 200 cases, 130 s at 1,000 cases and about 8 minutes at 2,000 cases per degree-2 fit with 10 covariates ([the fit-time table](#fit-time-against-earth)). The 200-case figure is extrapolated: n² scaling from the 21 s at 250 cases gives 13 s, and the measured growth from 250 to 500 cases gives 16 s. A full run of 300 repetitions on the 13 regression cells other than D7 at 1,000 cases, with two arms, would need about 280 core-hours.

D7 is far more expensive for the current code. With 50 covariates the default term limit is 101, and the cost model under [Speed and scaling](#speed-and-scaling) grows with p and with the fourth power of the number of terms. At 200 cases the rule GCV = ∞ when C ≥ n binds first. With K pairs, M = 1 + 2K and H = 2K, so with d = 3 the effective number of parameters is C = 1 + 8K, and C < 200 needs K ≤ 24, which stops P-cur near 49 terms (linear terms, which add only 1 to C each, are ignored here). The 10-covariate fits stop at their limit of 21 terms, so one D7 fit costs about 5 × (49/21)⁴ ≈ 150 times as much, about 40 minutes. D7 therefore runs only for the earth arms and P-fix.

The current-pymars arms are capped at 200 repetitions to fit the budget; if the pilot asks for more, the report says so. They run on the 13 other regression cells at 200 cases (about 22 core-hours), on the 2 binary cells at 200 cases (about 2 core-hours, since `EarthClassifier` and `GLMEarth` share one basis fit), and on 4 cells at 1,000 cases: D3, D4, D5 and D8 at the low-noise level (about 58 core-hours). They do not run at 5,000 cases. This is decision Q5.

Phase 3 targets less than 1 s per fit at 1,000 cases. If P-fix meets that target and the earth-style stopping rules keep D7 models small, the full grid of 51 cells, several hundred repetitions and 7 arms takes a few tens of core-hours; the 5,000-case cells cost about 5 times as much per fit, since the fast algorithm's cost, O(p·n·M_max³) under [Cost model](#cost-model), is linear in n. The Phase 3 benchmark gives the measured figures.

## Speed and scaling

### Cost model

Let M_max be the term limit. In pymars today, a forward step with M terms at degree 2 or more scores about M·p·n candidates, because each of the M terms can be the parent, each of the p covariates the variable, and each of up to n distinct values the knot. Each candidate rebuilds an n × (M + 2) basis matrix, at cost O(n·M) for terms of low degree, and solves it by SVD, at cost O(n·M²). One step therefore costs O(M·p·n) × O(n·M²) = O(p·n²·M³). Each step adds one or two terms, so a pass has between M_max/2 and M_max steps, and summing the per-step cost over them gives O(p·n²·M_max⁴), because the sum of M³ over those steps grows like M_max⁴. At degree 1 only the intercept can be a parent, so a step scores about p·n candidates and the pass costs O(p·n²·M_max³).

The measured times fit this model only roughly ([the fit-time table](#fit-time-against-earth)). From 1,000 to 2,000 cases the time grew by a factor of 4.0 at degree 1 and 3.8 at degree 2, close to n². The smaller doublings varied from 2.2 to 4.1, because the forward pass stopped at different sizes (14 to 21 terms at degree 1) and because a fixed Python overhead per candidate, multiplied by the roughly n candidates, adds a part of the cost that grows only linearly in n.

With Friedman's updating formula (1991, eq. 52), as in earth, each covariate is sorted once. For one parent and one variable, a sweep over all candidate knots keeps running sums, and each knot move updates M inner products, so the sweep costs O(n·M). One step covers M parents and p variables, so it costs O(p·n·M²), and the whole pass costs O(p·n·M_max³). Fast MARS (Friedman 1993) cuts the parents scored per step from M to earth's `fast.k`.

The two per-step costs differ by a factor of order n·M, about 2 × 10⁴ at n = 1,000 and M = 21. The measured ratios to earth (4,500 and 5,700 at 1,000 cases) are of that order once constant factors are included. The model also predicts that the ratio doubles with n; from 1,000 to 2,000 cases it grew by 2.7 at degree 1 and 1.9 at degree 2.

### Benchmark design

The benchmark varies one factor at a time around 1,000 cases, 10 covariates, degree 2 and 21 terms:

- cases 250, 500, 1,000, 2,000, 5,000, 10,000 and 100,000;
- covariates 5, 10, 20, 50 and 100;
- degree 1, 2 and 3;
- term limits 11, 21, 41 and 81;
- weights off and on.

earth runs at its defaults and with `fast.k = 0`, to separate the effect of Fast MARS from the effect of the updating formula. Each point is the median of 3 runs with one thread, recording wall time and peak memory (`tracemalloc` in Python and `/usr/bin/time -l` for both programs), with a timeout of 30 minutes per fit. The report gives the log-log slope for each factor together with the model sizes reached, because early stops change the amount of work.

### Profiling

One cProfile run is done (F10). Phase 3 adds `line_profiler` on `_find_best_candidate_addition`, `_build_basis_matrix` and the `transform` methods of the basis functions, a `py-spy` flame graph of a long fit, and counters for the candidates per step, the `lstsq` calls and the basis evaluations.

### Fast MARS status

Fast MARS is not implemented in Python or in Rust. There is no priority queue, no `fast_k` or `fast_beta` parameter, and no updating formula: every candidate is refit from scratch.

### Speed work

Phase 3 makes these changes in order.

1. Keep the current basis matrix B and a QR factor of it between steps, instead of rebuilding them for every candidate.
2. Score all knots of one parent and variable at once, with suffix sums over the sorted variable (Friedman 1991, eq. 52). The derivation follows this list.
3. Use the same sums for weighted fits. Here pymars can be faster than earth, which falls back to a full regression at each knot.
4. Add Fast MARS with `fast_k` and `fast_beta`, with defaults 20 and 1 as in earth.
5. Prune with QR downdating, which updates the factorization when a column is removed instead of refitting. The gain is small, because pruning takes less than 1% of the time now.
6. Clean up three slow spots: the missing-value scan is a Python loop over every entry of X in both fit and predict (`earth.py:531-539`, `:637-645`); every `predict` call builds a model specification (`earth.py:715-726`); and the fitted model stores the training data (`X_original_`).

The derivation behind the second change goes as follows. Let Q be an n × r matrix with orthonormal columns that span the current columns of the basis matrix B (r is its rank), so QᵀQ = I, and let e = y − (fitted values) be the current residual vector, so Qᵀe = 0. A new column h enters the fit only through its part orthogonal to Q, h⊥ = h − QQᵀh. The two parts of h = QQᵀh + h⊥ are orthogonal, so ‖h‖² = ‖QQᵀh‖² + ‖h⊥‖², and ‖QQᵀh‖² = hᵀQQᵀQQᵀh = ‖Qᵀh‖² because QᵀQ = I; hence ‖h⊥‖² = ‖h‖² − ‖Qᵀh‖². Because Qᵀe = 0, h⊥ᵀe = hᵀe − hᵀQQᵀe = hᵀe. The new model spans Q and h⊥, with h⊥ orthogonal to Q, so its fitted values are the old ones plus c·h⊥ with c = h⊥ᵀe/‖h⊥‖², and its residual e − c·h⊥ is orthogonal to h⊥. By Pythagoras the new RSS is ‖e‖² − (h⊥ᵀe)²/‖h⊥‖², so adding h lowers the RSS by

(h⊥ᵀe)² / ‖h⊥‖² = (hᵀe)² / (‖h‖² − ‖Qᵀh‖²).

For a candidate hinge, let b be the parent column (b_i is its value for case i), x the covariate, and t the knot, so the new column is h_t with entries b_i (x_i − t)₊. For any vector v (the residual e or one column of the orthonormal basis Q), the needed inner product is

h_tᵀv = Σ_i b_i v_i (x_i − t)₊ = Σ_{i: x_i > t} b_i v_i x_i − t Σ_{i: x_i > t} b_i v_i.

The first equality is the definition of the hinge column h_t, and the second holds because (x_i − t)₊ is x_i − t when x_i > t and 0 otherwise. After the cases are sorted by x, both sums over {i: x_i > t} are suffix cumulative sums, so one pass gives them for every candidate t at once, at cost O(n). The squared norm has the same form, because (x_i − t)₊² = x_i² − 2t·x_i + t² when x_i > t and 0 otherwise; multiplying by b_i² and summing gives

‖h_t‖² = Σ_{i: x_i > t} b_i² x_i² − 2t Σ_{i: x_i > t} b_i² x_i + t² Σ_{i: x_i > t} b_i².

So the RSS reductions of all knots for one parent and variable cost O(n·r), at most O(n·M): two suffix sums for each of the r + 1 vectors v (the residual and the r columns of Q), three for the norm, and O(r) work per knot to form ‖Qᵀh_t‖², the squared projection of the hinge column h_t onto the current fit, from the r inner products. The RSS drop is not monotone in t, so the best knot comes from a scan over all candidates. For a pair on a new combination of parent and variable, the linear column b·x enters first. The identity b·(t − x)₊ = b·(x − t)₊ − b·x + t·b shows that the pair spans the same space as (b·x, b·(x − t)₊) once the parent column b is in the model. A pair's RSS reduction is therefore the reduction from b·x plus the reduction from b·(x − t)₊ given b·x. Only the second part depends on t, so it alone picks the knot, and pairs on different combinations are compared by the sum. When b·x is already in the span, the first part is zero, which is F5's single-hinge case.

The subtraction ‖h‖² − ‖Qᵀh‖² loses precision when the candidate column h lies almost in the span of the orthonormal basis Q. earth's collinearity tolerance tests the same quantity: in its knot search it computes this difference on centered columns, divides it by the centered squared norm of h (its squared distance from its own mean), and compares the ratio with the tolerance. That ratio is 1 − R² of h regressed on the existing columns, which is why the findings table speaks of the share of variation explained. The fast code needs the same guard.

With case weights w_i, least squares works on rescaled rows: y_i becomes √w_i·y_i and row i of the basis matrix B becomes √w_i times itself, and Q and e come from the rescaled problem. The rescaled candidate column has entries √w_i·b_i (x_i − t)₊, so its inner product with any rescaled vector ṽ (the rescaled residual or a column of Q) is Σ_{i: x_i > t} √w_i b_i ṽ_i (x_i − t), the same suffix sums with √w_i b_i ṽ_i in place of b_i v_i; the squared norm uses w_i b_i² in place of b_i². No division by a weight is needed, so zero weights are allowed, and the sort order of x does not change, so the formula still applies.

Every speed change must pass one gate: on the conformance suite and on S15 (the 200 small draws) it must select the same terms and knots as the slow path, with RSS within relative 1e-8, apart from near-ties counted as under [Ties](#ties). The slow path stays in the test suite as a reference implementation.

The target for the speed claim is a fit time within 10 times earth's at 10,000 cases, 10 covariates and degree 2, in pure numpy. The report states the factor that is reached.

## scikit-learn compatibility, Python versions and determinism

### scikit-learn

The test suite gets `parametrize_with_checks` for `Earth`, `EarthRegressor`, `EarthClassifier` and `GLMEarth` (or its replacement), with no expected failures after Phase 2; F11 gives the current results. Integration tests cover:

- `clone`;
- `Pipeline` with a scaler;
- `cross_val_score` and `GridSearchCV` with `n_jobs=2`, which needs pickling (pickling works today);
- `StackingRegressor` and `StackingClassifier`, the closest scikit-learn analogue of a super learner;
- `TransformedTargetRegressor` and `CalibratedClassifierCV`;
- pandas input with feature names, and `set_output`;
- sample weights through metadata routing (`set_fit_request(sample_weight=True)`).

### Python and dependency versions

Each of Python 3.10, 3.11, 3.12, 3.13 and 3.14 gets a uv venv; uv must download the 3.10 and 3.11 interpreters, which needs your OK. Each Python version gets two dependency sets: the newest versions, and the lowest direct versions (`uv pip install --resolution lowest-direct`). Today the lowest set would pick scikit-learn 1.0 and fail (F12). The fix is to declare scikit-learn 1.6 or later, or to add a shim for the older keyword `force_all_finite`.

The metadata must agree: `requires-python`, the PyPI classifiers, ruff's `target-version` and ty's `python-version`. CI runs on Linux, macOS and Windows, and one Linux job builds the Rust extension and checks that the Rust predict path matches the Python path to 1e-12.

### Determinism

The fit is repeated in the same process and in a new process, with 1 BLAS thread and with all threads, with different BLAS libraries (Accelerate on macOS; OpenBLAS and MKL on Linux, through conda-forge or Docker), with numpy 1.26 and 2.x, with rows and columns permuted, and with the scale tests under [Invariance tests](#invariance-tests). The criterion is identical terms and knots, with RSS within relative 1e-10.

F6, the stop on an absolute epsilon, predicts failures across BLAS libraries today. After the relative stopping rules and the collinearity tolerance ([Improvements, ranked](#improvements-ranked)), these tests should pass except at near-ties.

## Improvements, ranked

The ranking is by value to someone who needs a correct and fast MARS in Python.

| Rank | Change | Findings | Value | Effort | Changes fitted models? |
|---|---|---|---|---|---|
| 1 | Weights: GCV and the size limits count cases, equal weights mean no weights, and fits are invariant to the scale of the weights | F1 | Critical | Hours | Weighted fits only |
| 2 | Never prune the intercept | F3 | Critical | Hours | Yes |
| 3 | Missing values: raise an error for `allow_missing=True` until it works, then adopt py-earth's design | F8 | Critical (silent NaN) | Hours now, a day or two later | No |
| 4 | Binary outcomes: an unpenalized logistic refit, `predict_proba`, a classifier class built on `ClassifierMixin`, and several classes as one indicator response per class or a clear error | F9 | High for super learners | Hours to a day | Classifiers |
| 5 | Forward selection by RSS; the penalty acts only in pruning; pruning removes the term with the smallest RSS increase | F2 | High | Hours | Yes |
| 6 | A single hinge when the parent and variable already appear together; rejection of near-collinear candidates; linear terms by the knot-at-minimum rule; the rank in GCV | F5 | High | A day or two | Yes |
| 7 | GCV per knot as in earth and py-earth; default penalty 2 at degree 1 and 3 above; `penalty = -1` | F4 | High | Hours | Yes |
| 8 | Relative stopping rules (`thresh`, R², GRSq) with no absolute epsilon; the default term limit as earth's `nk` | F6 | High | Hours | Yes |
| 9 | Knot spacing: eq. 43 and 45 with α = 0.05 by default, counted in cases and truncated, with `Adjust.endspan` and a centered grid | F7 | Medium to high | A day or two | Yes |
| 10 | The fast forward pass (eq. 52) in numpy, then Fast MARS | F10 | Very high for speed | Several days | No, apart from near-ties |
| 11 | scikit-learn compliance for `Earth`; every parameter exposed in the wrappers, or `Earth` itself as the regressor | F11 | Medium | Hours | No |
| 12 | Dependency bounds and metadata; matplotlib optional; lazy imports | F12 | Medium | Hours | No |
| 13 | Tests: fixtures from earth with strict tolerances, invariance tests, `parametrize_with_checks`, and no stale expected failures | F13 | High (prevents regressions) | A day or two | No |
| 14 | The three slow spots: the missing-value loop, the model specification built per predict call, and the stored training data | F10 | Low to medium | Hours | No |

Ranks 1 to 4 come first because they give wrong or unusable results in common use: weights appear in targeted maximum likelihood estimation and in inverse-probability weighting, and super learners need probabilities. Ranks 5 to 9 move the models toward the reference, and they belong in one release because each one changes the fixtures. Rank 10 is the largest change, and it is what makes the library usable above a few thousand cases. It comes after ranks 5 to 9, so the fast code implements the final rules only once. If speed matters more to you than exact earth behavior, rank 10 can move up.

## What to remove or split out

The repository holds much more code outside the fitting path than in it:

- the fitting modules and scikit-learn wrappers: 3,639 lines of Python;
- the runtime, portable-specification, cluster and accelerator modules: 2,185 lines;
- the Rust crate: 3,551 lines;
- bindings for R, Julia, Go, TypeScript and C#: 2,464 lines, plus 1,066 lines of Go at the root;
- 275 files under `conductor/`, 173 files under `docs/`, and 21 GitHub workflows.

Keep `Earth`, `EarthRegressor`, `EarthClassifier` (fixed), the GLM refit option, the portable JSON export (small, and useful for storing models), and the plots as an optional extra.

Split three groups into separate packages or repositories: the cluster and accelerator modules with the high-performance-computing documents; the Rust runtime and trainer; and the language bindings with the Go code. The Rust trainer is switched off, and the maturin build backend makes a source install need a Rust compiler, which works against the pure-Python goal.

Remove these:

- `EarthCV`, a thin wrapper around `cross_val_score`; if wanted, replace it with a pruning size chosen by cross-validation, like earth's `pmethod = "cv"`;
- the helpers in `_missing.py`, which the fit does not use;
- stray files at the root: `r.pdf` (the manual of the R binding), two PNG files, `bandit-report.json`, `safety-report.json`, `.pypi.json`, `.fork_status` and `go.mod`;
- the agent planning files (`conductor/`, `.agents/`, `.gemini/`, `QWEN.md` and `SESSION_LOGS.md`), from the package repository.

Switch the core back to a pure-Python build backend (hatchling or setuptools), so a source install needs no compiler, and drop matplotlib from the required dependencies. For the ctml-skills use case a lean fork is enough, whatever upstream decides.

## Reporting, and what to offer upstream

### In the repository

All of this goes in the fork, under `validation/`:

- `README.md` says how to run everything and lists the versions used.
- `harness/` holds `fit_earth.R`, the Python driver and the trace parsers.
- `fixtures/` holds the inputs and earth outputs as JSON, with the R and earth versions, and one command regenerates them. These fixtures replace the self-referential ones as the regression gate, with the tolerances under [What is compared, and the tolerances](#what-is-compared-and-the-tolerances).
- `DIFFERENCES.md` is the triaged catalogue described under [Triage of differences](#triage-of-differences).
- `sims/` and `bench/` hold the code, the raw per-repetition results and the summaries.
- `REPORT.md`, or a Quarto document rendered to HTML, is organized by claim, in the order conformance, gap, parity, sparsity and speed. Each claim gets its ADEMP description (aims, data-generating processes, estimands, methods and performance measures), its evidence, and a statement of what the evidence shows. Every table and figure names the claim it serves.
- An optional CI job installs R and earth (`r-lib/actions/setup-r`), regenerates the fixtures and runs the conformance suite on every pull request that touches the fitting code.

### For the ctml-skills supervised-learning skill

A short note gives the verdict, the settings to use (for example `pymars.EarthRegressor(max_degree=2)` after the fixes), and a proposed edit to the MARS row and to the Python paragraph, which today says that MARS has no maintained Python implementation. The skill's "very fast" for MARS gets a qualification for Python, based on the benchmark.

### Upstream

Each of these waits for your approval:

1. One issue per confirmed bug (F1, F3, F6, F8, F9, F11 and F12), one per departure from earth (F2, F4, F5 and F7), one on speed (F10) and one on the reference tests (F13), each with a minimal reproduction and, where it applies, the earth output.
2. Pull requests from the fork, one per fix, each with its test.
3. The earth conformance harness and fixtures, as an optional CI job.
4. The fast forward pass, as a separate pull request.
5. The validation report.
6. An issue that proposes to split out the code that does not fit models, for discussion before any code moves.

Upstream's own planning excludes R earth as a validation gate and excludes benchmarks across implementations (`conductor/tracks/reference_regression_validation_20260420/spec.md`). Its "parity audit" compared documentation only and rated weights and missing values as compatible (`docs/parity_audit_repo_gap_matrix.md`). The work should therefore reach upstream as added evidence, in small changes that are easy to review.

## Phases, exit criteria and review points

| Phase | Content | Exit criterion | Review point |
|---|---|---|---|
| 0 | Setup: environments | venvs for Python 3.10 to 3.14 (after your OK for the downloads); R and earth versions recorded; the suite passes | None |
| 1 | The harness and a description of current behavior, with no library changes | The conformance suite runs on HEAD and on the 1.0.4 wheel; `DIFFERENCES.md` is complete; every `bug` item has a failing test | You review `DIFFERENCES.md` and decide Q2 (the behavior target) |
| 2 | The changes ranked 1 to 9 and 11 to 14 under [Improvements, ranked](#improvements-ranked), tests first | All tests pass; the earth-compatible mode matches earth at its defaults on S01 to S15 except at near-ties; the invariance tests pass | You review the diff |
| 3 | Speed (rank 10) | The same models as in Phase 2 on the conformance suite; a benchmark report; the speed claim met, or the gap stated | You review the benchmark |
| 4 | Simulation | A pilot report with the number of repetitions for each contrast; the full run; every mockup filled | You review the report draft |
| 5 | The report and the upstream material | `REPORT.md` reviewed by you; issues and pull requests drafted but not opened | You approve each upstream item |

pymars counts as validated for the skill when no difference in `DIFFERENCES.md` is unexplained, the invariance tests pass, the parity claim holds or the report narrows it to the cells where it holds, the speed result is stated as measured, `check_estimator` passes for every public estimator, and CI passes on Python 3.10 to 3.14 and on three operating systems.

## Appendix: reproducing the preliminary findings

The environment, from the worktree root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy scikit-learn matplotlib pandas pytest pytest-benchmark hypothesis click rich
echo "$PWD" > .venv/lib/python3.12/site-packages/pymars_worktree.pth
.venv/bin/python -m pytest -q -p no:cacheprovider
```

The scripts are in the session scratchpad and are not committed; Phase 1 turns them into `validation/`. The probe names in the findings table are the labels that `probe_bugs.py` and `probe_bugs2.py` print before each result.

| Script | Findings |
|---|---|
| `probe_bugs.py`, `probe_bugs2.py` | F1, F2, F3, F5, F6, F8, F9, F16, and the tiny-n and scaled-input probes |
| `sk_checks.py` | F11 |
| `compare_earth.py`, `fit_earth.R` | F14, F15, and the earth trace for F5 |
| `timing.py` | [The fit-time table](#fit-time-against-earth) |
| `blas_det.py`, `wheel_vs_head.py` | F16, and the identical fits of HEAD and the 1.0.4 wheel |
| A venv with scikit-learn 1.5.2 | F12 |

The Python snippet reproduces F1, where weights that sum to 1 give an intercept-only model:

```python
import numpy as np
import pymars

rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = 10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3] + 5*X[:, 4] + rng.normal(size=200)
print(len(pymars.Earth().fit(X, y).basis_))                                    # 9 terms
print(len(pymars.Earth().fit(X, y, sample_weight=np.full(200, 1/200)).basis_))  # 1 term
```

The R snippet reproduces the earth trace in F5, where earth rejects pymars' fourth knot (0.148) with `TolG 0`:

```r
library(earth)
d <- read.csv("data/hinge1d_1_train.csv")   # written by compare_earth.py
earth(x = as.matrix(d["x0"]), y = d$y, degree = 1, penalty = 2, nk = 21,
      thresh = 0, minspan = 1, endspan = 1, fast.k = 0,
      Auto.linpreds = FALSE, pmethod = "none", trace = 9)
```

## Appendix: references

- Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics* 19(1), 1-67.
- Friedman, J. H. (1993). Fast MARS. Technical Report 110, Department of Statistics, Stanford University.
- Milborrow, S. (2024). Notes on the earth package. Vignette of the R package earth, version 5.3.4.
- Milborrow, S., Hastie, T., Tibshirani, R., Miller, A. and Lumley, T. earth: Multivariate Adaptive Regression Splines. R package version 5.3.4 (GPL-3).
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd edition, section 9.4.
- Morris, T. P., White, I. R. and Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine* 38(11), 2074-2102.
- py-earth (archived): https://github.com/scikit-learn-contrib/py-earth
