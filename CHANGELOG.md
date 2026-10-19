# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0.dev0] - unreleased

The rewrite of pymars has started, in the fork alejandroschuler/mars. The new fitting code follows a written specification and is validated against the R package earth, as VALIDATION_PLAN.md describes. The git tag `legacy-1.0.4-head` keeps the legacy code, upstream commit d68b54a with the version string 1.0.4.

### Added
- `EarthRegressor` (also `pymars.Earth`) and `EarthClassifier`, scikit-learn estimators that fit MARS by the rules of `docs/algorithm.md`. The parameters map to earth's arguments (`max_degree`, `max_terms`, `penalty`, `thresh`, `minspan`, `endspan`, `adjust_endspan`, `auto_linpreds`, `fast_k`, `fast_beta`, `pmethod`, `nprune`), and the classifier adds `glm_alpha`, the L2 penalty of the logistic refit, with the default 1 (the strength of scikit-learn's `LogisticRegression` at `C=1`). At 0 the refit is earth's unpenalized one, and it warns on scikit-learn's breast_cancer, iris and wine data (#175); use `glm_alpha=0` to match earth.
- Fitted attributes `dirs_`, `cuts_`, `term_coef_`, `rss_`, `gcv_`, `rsq_`, `grsq_`, `max_terms_`, `penalty_` and `mars_`; for the classifier `classes_` and `glm_`. Methods `predict`, `predict_proba`, `decision_function`, `basis_matrix` and `summary`.
- Frequency weights through `sample_weight`: an integer weight acts as repeated rows, and a zero weight removes the row. Several responses share one set of terms. The classifier refits the selected terms with a binomial or multinomial logistic model.
- Validation: a reference implementation written from the specification; both it and the fast code are compared with earth 5.3.4 on 133 dataset fixtures and 200 random draws, with a triage of every difference found (`validation/DIFFERENCES.md`: 7 differences, each a documented rule or an earth quirk, none a bug, the same for both); oracle tests of the fast code against the reference, scikit-learn's estimator checks, integration tests with pipelines, searches and stacking, a benchmark against earth (`validation/bench/REPORT.md`), and a simulation study against earth and the 1.0.4 release over 51 settings (`validation/REPORT.md`).
- numba-compiled loops for the knot scan (#148, #159), used wherever numba has wheels. Intel Macs run the numpy code, which gives the same models except at near-ties (STOP-7 of `docs/algorithm.md`), where the two can choose different terms.
- The types of the fit are public: `MarsFit` (the type of `mars_`), `ForwardRecord`, `PruningRecord`, `CandidateLog`, `Termination` and `MarsParams`. `MarsFit.to_dict()` has the key `"format"` (`pymars.DICT_FORMAT`), which `MarsFit.from_dict` checks.
- Documentation: `docs/algorithm.md` (the specification), `docs/usage.md`, `docs/weights.md` and `docs/differences.md`.

### Changed
- The build backend is hatchling. The wheel is pure Python and depends on numpy 1.24.4 or later, scipy 1.9.3 or later, scikit-learn 1.6 or later and, except on macOS x86_64 (Intel Macs, where numba has no wheels, so the slower numpy code runs), numba 0.63 or later. It supports Python 3.10 to 3.14. The numpy floor is 1.24.4 because the OpenBLAS in numpy 1.23.5 gives wrong matrix products on some CPUs (#88).
- A lean CI workflow replaces the 21 inherited workflows.
- Release fixes from the architecture reviews (#177): a bad parameter's error names the estimator ("EarthRegressor: penalty must be ..."), and an unknown keyword such as `degree=2` names the estimator class. Weights that are not integers warn only when their mean is below 0.5 or above 2, so cross-validation folds of rescaled weights do not warn. The version string has one place, `pymars/__init__.py`, and the sdist no longer ships the tests, which need the fixtures in `validation/`.
- Spec v3 (#157, #158): all numerics are float64 with tolerances. The exact rational arithmetic that v2 used for nearly dependent columns and large covariate means is gone from the fast code and the reference (#160 to #162), with about 1,100 lines of the fast code. The "Limits" section of `docs/algorithm.md` (and of `docs/differences.md`, in plain words) names the inputs where rounding can change the model.
- Speed passes (#124, #148, #156, #159, #162, #167): the T20 profiling passes, a numba kernel for the knot scan, compiled bounds and masks, the removal of the exact paths, and compiled setup, refine and search steps of the forward pass. A fit with 10,000 cases, 10 covariates and degree 2 now takes 1.7 times as long as earth in a fresh process and about the same time in a warm one, down from 7.7 times; at 100,000 cases pymars takes 0.76 times earth's time. The numpy code (Intel Macs) takes 2.6 to 2.7 times earth's time at 10,000 cases.

### Removed
- All legacy code in `pymars/`, with `EarthCV`, `GLMEarth`, `CategoricalImputer`, `categorical_features`, `feature_importance_type`, the plots, `explain.py`, the portable JSON model format and the command-line tool.
- `pymars_runtime/`, the Rust crate, the bindings for R, Julia, Go, TypeScript and C#, and the Go code at the root.
- The old tests, whose fixtures compared the code only with its own stored outputs.
- The documentation site, the agent and planning files, the packaging recipes, the scripts, and the release and security tooling.

## [1.0.1] - 2025-11-08
### Fixed
- Addressed sklearn deprecation warnings by updating `force_all_finite` parameter to `ensure_all_finite` with backward compatibility
- Improved documentation with comprehensive tutorial, API reference, and usage examples
- Updated license from MIT to Apache 2.0
- Fixed GitHub Pages workflow to properly deploy documentation to docs folder on main branch

## [1.0.0-beta.1] - 2025-02-02
### Added
- Initial release of pure Python MARS implementation
- Full scikit-learn compatibility with EarthRegressor and EarthClassifier
- Generalized Linear Models support with GLMEarth
- Cross-validation helper with EarthCV
- Feature importance calculations (nb_subsets, gcv, rss)
- Missing value and categorical feature support
- Plotting utilities for diagnostics
- Comprehensive test suite with >90% coverage
- State-of-the-art CI/CD pipeline with automated testing, linting, type checking, and security scanning
- Property-based testing with Hypothesis
- Performance benchmarking with pytest-benchmark
- Advanced interpretability tools (partial dependence, ICE plots, model explanations)
- Enhanced command-line interface for model fitting, prediction, and scoring
- Comprehensive documentation and development guidelines
- Automated release workflow to PyPI
- Code coverage integration with Codecov

[1.0.0]: https://github.com/pymars/pymars/releases/tag/v1.0.0
