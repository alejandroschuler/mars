# pymars

pymars fits multivariate adaptive regression splines (MARS, Friedman 1991) in Python, with a scikit-learn API. The distribution name is `mars-earth` and the import name is `pymars`.

Version 2 is a rewrite. The fitting code follows a written specification, [docs/algorithm.md](docs/algorithm.md), and is validated against the R package [earth](https://CRAN.R-project.org/package=earth) as described under Status and validation below. It has no release on PyPI yet.

## Install

```bash
pip install git+https://github.com/alejandroschuler/mars
```

The wheel is pure Python and needs no compiler. It supports Python 3.10 to 3.14 and depends on numpy, scipy and scikit-learn 1.6 or later, and on numba 0.63 or later, which compiles the knot scan on first use and speeds up the fit. numba has no wheels for Intel Macs (macOS x86_64), so pymars does not install numba there and runs the numpy code, which takes up to about 3 times as long as the numba code (see Status and validation). The two give the same models up to rounding, so they can differ only at near-ties, where two candidates are equal up to rounding.

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

Two implementations of the specification exist: the fast code in `pymars/`, and a slow reference in `tests/reference/` that an agent wrote from the specification alone. Both compute in float64; the specification's "Limits" section names the rare inputs where rounding can change the model (see [docs/differences.md](docs/differences.md)). Both are compared with earth 5.3.4 on 133 dataset fixtures and 200 random draws: 124 fixtures and 197 draws agree with earth step by step, and the other 12 comparisons show 7 differences, each a documented rule or an earth quirk that pymars does not copy, none a bug. The fast code shows exactly the same differences as the reference. [docs/differences.md](docs/differences.md) lists the differences that a user can meet, and [validation/DIFFERENCES.md](validation/DIFFERENCES.md) has the full triage. [VALIDATION_PLAN.md](VALIDATION_PLAN.md) describes the work, and [dev/DECISIONS.md](dev/DECISIONS.md) lists the choices made.

Accuracy: in a simulation of 51 settings (8 regression and 2 binary data-generating processes, 200 to 5,000 cases, two noise levels), `EarthRegressor(max_degree=2)` and `EarthClassifier(max_degree=2)` match earth's excess risk within 5 percent in 48 settings. On pure noise pymars is 3 to 8 percent worse than earth at every sample size, and it keeps slightly more noise terms; the study does not show which rule causes this. Two settings are not decided. [validation/REPORT.md](validation/REPORT.md) has the study, claim by claim.

Speed: in the benchmark of [validation/bench/REPORT.md](validation/bench/REPORT.md), a fit with 10,000 cases, 10 covariates, degree 2 and 21 terms takes 1.7 times as long as earth at its defaults for `EarthRegressor` in a fresh process (0.81 s against 0.46 s), and about the same time (0.92 times, within the noise of the two measurements) for later fits in the same process. The numpy code that Intel Macs run takes 2.6 to 2.7 times earth's time there. At 100,000 cases pymars with numba is faster than earth (5.8 s against 7.6 s, a factor of 0.76 fresh and 0.67 warm), and the numpy code takes 2.0 times. At 250 cases earth is faster, by a factor of 21 in a fresh process and 3.5 in a warm one, since pymars has a fixed cost of about 0.06 s per fit and about 0.3 s more for the first fit in a process. Weights: `sample_weight` means case counts; see [docs/weights.md](docs/weights.md).

## Legacy code

The legacy code, upstream commit d68b54a with the version string 1.0.4, is kept under the git tag `legacy-1.0.4-head`. The PyPI release mars-earth 1.0.4 differs from that commit in many lines; `validation/legacy/make_venv.sh` installs the release for comparisons. Version 2 removes the legacy code, together with the Rust runtime, the language bindings, the command-line tool and the portable model format. [CHANGELOG.md](CHANGELOG.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Agents also follow [AGENTS.md](AGENTS.md).

## License and attribution

pymars is licensed under the [Apache License 2.0](LICENSE). This fork builds on [mars](https://github.com/edithatogo/mars) by Dylan A Mordaunt. Its API follows [py-earth](https://github.com/scikit-learn-contrib/py-earth), the archived scikit-learn-contrib package. The method is from Friedman (1991, 1993). The tests use the R package earth (GPL-3) by Stephen Milborrow and coauthors as a black-box reference, and pymars contains no earth code. To cite pymars, use [CITATION.cff](CITATION.cff).
