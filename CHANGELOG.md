# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0.dev0] - unreleased

The rewrite of pymars has started, in the fork alejandroschuler/mars. The new fitting code follows a written specification and is validated against the R package earth, as VALIDATION_PLAN.md describes. The git tag `legacy-1.0.4-head` keeps the legacy code, upstream commit d68b54a with the version string 1.0.4.

### Added
- `EarthRegressor` (also `pymars.Earth`) and `EarthClassifier`, scikit-learn estimators that fit MARS by the rules of `docs/algorithm.md`. The parameters map to earth's arguments (`max_degree`, `max_terms`, `penalty`, `thresh`, `minspan`, `endspan`, `adjust_endspan`, `auto_linpreds`, `fast_k`, `fast_beta`, `pmethod`, `nprune`), and the classifier adds `glm_alpha`.
- Fitted attributes `dirs_`, `cuts_`, `term_coef_`, `rss_`, `gcv_`, `rsq_`, `grsq_`, `max_terms_`, `penalty_` and `mars_`; for the classifier `classes_` and `glm_`. Methods `predict`, `predict_proba`, `decision_function`, `basis_matrix` and `summary`.
- Frequency weights through `sample_weight`: an integer weight acts as repeated rows, and a zero weight removes the row. Several responses share one set of terms. The classifier refits the selected terms with a binomial or multinomial logistic model.
- Validation: a reference implementation written from the specification, compared with earth 5.3.4 on 133 dataset fixtures and 200 random draws at commit 2a00063 with a triage of every difference found (`validation/DIFFERENCES.md`, none is a bug); oracle tests of the fast code against the reference (its direct comparison with earth is pending, C07), scikit-learn's estimator checks, integration tests with pipelines, searches and stacking, and a benchmark against earth (`validation/bench/REPORT.md`).
- Documentation: `docs/algorithm.md` (the specification), `docs/usage.md`, `docs/weights.md` and `docs/differences.md`.

### Changed
- The build backend is hatchling. The wheel is pure Python and depends on numpy 1.24.4 or later, scipy 1.9.3 or later, scikit-learn 1.6 or later and, except on macOS x86_64 (Intel Macs, where numba has no wheels, so the slower numpy code runs), numba 0.63 or later. It supports Python 3.10 to 3.14. The numpy floor is 1.24.4 because the OpenBLAS in numpy 1.23.5 gives wrong matrix products on some CPUs (#88).
- A lean CI workflow replaces the 21 inherited workflows.

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

## [Unreleased]

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

[Unreleased]: https://github.com/pymars/pymars/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pymars/pymars/releases/tag/v1.0.0
