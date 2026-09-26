# pymars

pymars fits multivariate adaptive regression splines (MARS, Friedman 1991) in pure Python, with a scikit-learn API. The distribution name is `mars-earth` and the import name is `pymars`.

Version 2 is a rewrite, and it is not usable yet. The new fitting code follows a written specification and is checked against the R package [earth](https://CRAN.R-project.org/package=earth). [VALIDATION_PLAN.md](VALIDATION_PLAN.md) describes the work, and [dev/DECISIONS.md](dev/DECISIONS.md) lists the choices made so far. Until the estimators arrive, the package holds only its version number.

## Install

```bash
pip install git+https://github.com/alejandroschuler/mars
```

The package is pure Python and needs no compiler. It supports Python 3.10 to 3.14 and depends only on numpy, scipy and scikit-learn 1.6 or later. Version 2 has no PyPI release.

When the estimators are in place, the usage will follow py-earth:

```python
import pymars as earth

model = earth.Earth(max_degree=2).fit(X, y)
```

## Legacy code

The legacy code, upstream commit d68b54a with the version string 1.0.4, is kept under the git tag `legacy-1.0.4-head`. The PyPI release mars-earth 1.0.4 differs from that commit in many lines; `validation/legacy/make_venv.sh` installs the release for comparisons. Version 2 removes the legacy code, together with the Rust runtime, the language bindings, the command-line tool and the portable model format. [CHANGELOG.md](CHANGELOG.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Agents also follow [AGENTS.md](AGENTS.md).

## License and attribution

pymars is licensed under the [Apache License 2.0](LICENSE). This fork builds on [mars](https://github.com/edithatogo/mars) by Dylan A Mordaunt. Its API follows [py-earth](https://github.com/scikit-learn-contrib/py-earth), the archived scikit-learn-contrib package. The method is from Friedman (1991, 1993). The tests use the R package earth (GPL-3) by Stephen Milborrow and coauthors as a black-box reference, and pymars contains no earth code. To cite pymars, use [CITATION.cff](CITATION.cff).
