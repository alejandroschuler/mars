# pymars

pymars fits multivariate adaptive regression splines (MARS, Friedman 1991) in pure Python, with a scikit-learn API. The distribution name is `mars-earth` and the import name is `pymars`.

Version 2 is a rewrite. The fitting code follows a written specification, [docs/algorithm.md](docs/algorithm.md), and is validated against the R package [earth](https://CRAN.R-project.org/package=earth) as described under Status and validation below. It has no release on PyPI yet.

## Install

```bash
pip install git+https://github.com/alejandroschuler/mars
```

The package is pure Python and needs no compiler. It supports Python 3.10 to 3.14 and depends only on numpy, scipy and scikit-learn 1.6 or later.

## Regression

```python
import numpy as np
import pymars as earth
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.uniform(size=(500, 5))
y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
     + 10 * X[:, 3] + 5 * X[:, 4] + rng.normal(size=500))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = earth.Earth(max_degree=2).fit(X_train, y_train)
print(model.summary())            # the selected terms, their coefficients and the fit statistics
print(model.score(X_test, y_test))
print(model.predict(X_test[:3]))
```

`Earth` is the same class as `EarthRegressor`, so `import pymars as earth` gives py-earth's spelling. The parameters carry earth's rules and defaults. [docs/usage.md](docs/usage.md) lists them with their earth names.

## Classification

`EarthClassifier` fits the terms by least squares on 0/1 (or indicator) responses and then refits the selected terms with a logistic model.

```python
import numpy as np
from pymars import EarthClassifier

rng = np.random.default_rng(1)
X = rng.uniform(size=(600, 4))
p = 1 / (1 + np.exp(-6 * (np.abs(X[:, 0] - 0.5) + X[:, 1] - 0.8)))
y = rng.uniform(size=600) < p

clf = EarthClassifier(max_degree=2).fit(X[:400], y[:400])
print(clf.classes_)                      # [False  True]
print(clf.predict_proba(X[400:403]))     # one row per case, each row sums to 1
print(clf.score(X[400:], y[400:]))
```

## scikit-learn

Both estimators follow the scikit-learn API (`fit`, `predict`, `score`, `get_params`, `clone`, pickle, `sample_weight`), pass `parametrize_with_checks`, and work in pipelines and searches.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pymars as earth

rng = np.random.default_rng(0)
X = rng.uniform(size=(300, 4))
y = np.maximum(X[:, 0] - 0.5, 0) * 8 + X[:, 1] + rng.normal(scale=0.1, size=300)

pipe = Pipeline([("scale", StandardScaler()), ("earth", earth.Earth())])
search = GridSearchCV(pipe, {"earth__max_degree": [1, 2], "earth__penalty": [2, 3, 4]}, cv=3)
search.fit(X, y)
print(search.best_params_)
```

Columns of strings or categories raise an error that names `OneHotEncoder`; [docs/usage.md](docs/usage.md) shows the pipeline. Missing values are not supported yet (issue #27).

## Status and validation

Two implementations of the specification exist: the fast code in `pymars/`, and a slow reference in `tests/reference/` that an agent wrote from the specification alone. The reference is compared with earth 5.3.4 on 133 dataset fixtures and 200 random draws; every difference found there has a label, and none is a bug. Those counts are from the reference at commit 2a00063 of `main`. The fast code is compared with the reference by the oracle tests, and its direct comparison with earth is pending (C07); T22b updates this text after it. [docs/differences.md](docs/differences.md) lists the differences that a user can meet, and [validation/DIFFERENCES.md](validation/DIFFERENCES.md) has the full triage. [VALIDATION_PLAN.md](VALIDATION_PLAN.md) describes the work, and [dev/DECISIONS.md](dev/DECISIONS.md) lists the choices made.

Speed: in the benchmark of [validation/bench/REPORT.md](validation/bench/REPORT.md), a fit with 10,000 cases, 10 covariates, degree 2 and 21 terms takes 7.7 times as long as earth at its defaults for `EarthRegressor` (4.3 s against 0.56 s; 7.0 times for the internal `fit_mars`, at commit `ae9d199`; later changes to the fast code are not in that run yet). Weights: `sample_weight` means case counts; see [docs/weights.md](docs/weights.md).

## Legacy code

The legacy code, upstream commit d68b54a with the version string 1.0.4, is kept under the git tag `legacy-1.0.4-head`. The PyPI release mars-earth 1.0.4 differs from that commit in many lines; `validation/legacy/make_venv.sh` installs the release for comparisons. Version 2 removes the legacy code, together with the Rust runtime, the language bindings, the command-line tool and the portable model format. [CHANGELOG.md](CHANGELOG.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Agents also follow [AGENTS.md](AGENTS.md).

## License and attribution

pymars is licensed under the [Apache License 2.0](LICENSE). This fork builds on [mars](https://github.com/edithatogo/mars) by Dylan A Mordaunt. Its API follows [py-earth](https://github.com/scikit-learn-contrib/py-earth), the archived scikit-learn-contrib package. The method is from Friedman (1991, 1993). The tests use the R package earth (GPL-3) by Stephen Milborrow and coauthors as a black-box reference, and pymars contains no earth code. To cite pymars, use [CITATION.cff](CITATION.cff).
