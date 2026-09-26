# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0.dev0] - unreleased

The rewrite of pymars has started, in the fork alejandroschuler/mars. The new fitting code will follow a written specification and will be checked against the R package earth, as VALIDATION_PLAN.md describes. This version has no estimators yet. The git tag `legacy-1.0.4-head` keeps the legacy code, upstream commit d68b54a with the version string 1.0.4.

### Changed
- The build backend is hatchling. The package is pure Python and depends only on numpy 1.23.5 or later, scipy 1.9.3 or later and scikit-learn 1.6 or later. It supports Python 3.10 to 3.14.
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
