# Using pymars

pymars has two estimators, `EarthRegressor` (also `pymars.Earth`) and `EarthClassifier`. Both follow the scikit-learn API. The spec, [algorithm.md](algorithm.md), has the rule behind each item; this page cites the rule ids (for example GCV-4) so that you can look them up.

```python
import pymars as earth
from pymars import EarthClassifier, EarthRegressor   # earth.Earth is EarthRegressor
```

## Parameters

Both estimators take the parameters below. `None` means earth's automatic value. The third column is the argument of R's `earth()` that the parameter maps to (API-7).

| Parameter | Default | earth | Meaning |
|---|---|---|---|
| `max_degree` | 1 | `degree` | The largest number of factors in a term. |
| `max_terms` | None | `nk` | The term limit of the forward pass. None gives min(200, max(20, 2p)) + 1 for p columns (LIMIT-1). Each forward step counts 2 toward the limit (LIMIT-2). |
| `penalty` | None | `penalty` | The GCV cost per knot. -1 turns the penalty off; otherwise 0 or more. None gives 2 at degree 1 and 3 above (GCV-4). |
| `thresh` | 0.001 | `thresh` | The forward pass stops when RSq grows by less than this (STOP-4). |
| `minspan` | None | `minspan` | The number of cases between knots. None is earth's automatic value (earth's 0). Negative values are not offered. |
| `endspan` | None | `endspan` | The number of cases kept beyond the end knots. None is automatic. |
| `adjust_endspan` | 2.0 | `Adjust.endspan` | The factor on the endspan for interaction terms (SPAN-4). |
| `auto_linpreds` | True | `Auto.linpreds` | Use a linear term where a hinge sits at the smallest value (FWD-6). |
| `fast_k` | 20 | `fast.k` | How many parents the Fast MARS queue searches in a step. 0 searches all of them. 1 and 2 act as 3, as in earth (FAST-3). |
| `fast_beta` | 1.0 | `fast.beta` | The ageing factor of the Fast MARS queue (FAST-2). |
| `pmethod` | `"backward"` | `pmethod` | `"backward"` prunes by GCV. `"none"` keeps the forward terms (PRUNE-1). earth's other methods are not offered. |
| `nprune` | None | `nprune` | The largest number of terms, intercept included, that pruning may keep. None is no limit. |
| `allow_missing` | False | none | True raises NotImplementedError. Missing values are not supported; see issue #27. Impute first, for example with `SimpleImputer` in a pipeline. |
| `glm_alpha` (classifier only) | 1.0 | none | The L2 penalty of the logistic refit, on columns standardized to mean 0 and variance 1. The default 1 has the strength of scikit-learn's `LogisticRegression` at `C=1` and gives a finite fit when the classes are separated. `glm_alpha=0` is the plain fit of earth's `glm = list(family = binomial)`: use it to match earth (GLM-2). |

earth arguments that pymars does not offer: `linpreds`, `allowed`, `newvar.penalty`, `nfold` and `Scale.y` (API-7). The estimators have no `coef_`, no `transform`, no `max_iter` and no `feature_importances_` (API-6). A parameter outside its range raises ValueError in `fit` (ERR-4).

`fit(X, y, sample_weight=None)` returns the estimator. `sample_weight` holds case counts; read [weights.md](weights.md).

## Fitted attributes

Names that end in an underscore exist only after `fit` (API-3).

| Attribute | Meaning |
|---|---|
| `n_features_in_` | The number of columns of X. |
| `feature_names_in_` | The column names, only after a fit on a table with string column names. |
| `dirs_`, `cuts_` | The selected terms, two arrays of shape (M, p). `dirs_[k, j]` is 0 when covariate j is not in term k, +1 for the factor (x_j - cut)+, -1 for (cut - x_j)+, and 2 for the linear factor x_j. Row 0 is the intercept. `cuts_` holds the knots, and 0.0 where there is no knot (TERM-1, TERM-2). |
| `term_coef_` | The least-squares coefficients of the terms: shape (M,) for a 1-D y, and (M, K) for K responses. |
| `rss_`, `gcv_`, `rsq_`, `grsq_` | The residual sum of squares, the GCV, RSq and GRSq of the returned model (PRUNE-8). |
| `max_terms_`, `penalty_` | The term limit and the penalty that the fit used, after `None` is resolved. |
| `mars_` | The whole `MarsFit`: the forward record (terms added, RSS after each step, termination code), the pruning record (RSS and GCV of every subset size), and the selected terms (CORE-3). Its types are exported: `pymars.MarsFit`, `ForwardRecord`, `PruningRecord`, `CandidateLog`, `Termination` and `MarsParams`. For example `est.mars_.pruning.gcv_per_size` gives the GCV of each subset size, and `est.mars_.forward.termination` says why the forward pass stopped. |
| `classes_`, `glm_` (classifier only) | The sorted class labels, and the coefficients of the logistic refit. |

The functions of the model are `predict`, `score` (from the scikit-learn mixins), `basis_matrix(X)` and `summary()`. `basis_matrix(X)` returns the (n, M) matrix of the selected terms at X. Nothing is clipped outside the range of the training data (TERM-3). The regressor's `predict(X)` is `basis_matrix(X) @ term_coef_`.

## summary

`summary()` returns a string with one line per selected term and two lines of statistics (API-4, TERM-5). A term's label joins its factors with `*`: `h(x0-0.45)` is (x0 - 0.45)+, `h(0.45-x0)` is (0.45 - x0)+, and a bare name is a linear term. After a fit on a table with string column names, the labels use those names; otherwise they use `x0`, `x1`, and so on (numbered from 0, where earth numbers from 1).

```python
import numpy as np
import pandas as pd
import pymars as earth

rng = np.random.default_rng(1)
df = pd.DataFrame({"age": rng.uniform(20, 70, 400), "dose": rng.uniform(0, 1, 400)})
y = 3 * np.maximum(df["age"] - 45, 0) + 10 * df["dose"] + rng.normal(size=400)

model = earth.Earth().fit(df, y)
print(model.feature_names_in_)   # ['age' 'dose']
print(model.summary())
```

The statistics line gives `rss`, `gcv`, `rsq` and `grsq`. The last line gives the termination code of the forward pass, for example `termination 4 (RSQ_CHANGE_SMALL)` or `termination 7 (TERM_LIMIT)`. For the classifier the coefficients are those of the logistic refit, one column per class (one column with two classes), and the statistics belong to the least-squares passes.

## Several responses

A 2-D y with K columns gives one set of terms for all responses. `term_coef_` has shape (M, K), and `predict` returns shape (n, K). The criterion sums the sums of squares over the responses without rescaling them, so a response in larger units counts more, as in earth (RESP-1, RESP-3). A 2-D y with one column gives the same fit as a 1-D y, and the predictions keep the shape of y (RESP-2).

## Classification

`EarthClassifier` runs the forward and pruning passes on a 0/1 response (two classes) or on one indicator response for each class (three or more). It then refits the selected terms with a logistic model: a binomial GLM for two classes, and one multinomial model for three or more, so that the rows of `predict_proba` sum to 1 (GLM-1, GLM-2). The labels can be numbers, strings or booleans. Classes that appear only on rows of weight 0 do not count.

- `predict_proba(X)` has shape (n, Q) in the order of `classes_`.
- `decision_function(X)` returns the linear predictor of `classes_[1]`, shape (n,), with two classes. With three or more it returns all Q predictors, shape (n, Q), and the first column is 0.
- `predict(X)` returns the class of largest probability, and the first such class on a tie.
- The default `glm_alpha=1` shrinks the refit. `glm_alpha=0` is earth's unpenalized refit, which has no finite minimum when the classes are separated. The fit then issues a ConvergenceWarning and keeps its last step (GLM-4). This also happens on scikit-learn's breast_cancer, iris and wine data, with coefficients up to about 480 (issue #175). A positive value, the default 1 or a weaker one such as 1e-4, gives a finite fit. Use `glm_alpha=0` to reproduce earth.
- y must be 1-D. A column vector is raveled with a DataConversionWarning. Multi-output classification is not supported.

## Text and categorical columns

A column that holds strings, bytes or pandas categories raises ValueError that names `OneHotEncoder`. Encode such columns first. `drop="first"` gives earth's coding of factors (ERR-2).

```python
import numpy as np
import pandas as pd
import pymars as earth
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(1)
n = 400
df = pd.DataFrame({"age": rng.uniform(20, 70, n), "dose": rng.uniform(0, 1, n),
                   "arm": rng.choice(["a", "b", "c"], n)})
y = (3 * np.maximum(df["age"] - 45, 0) + 10 * df["dose"]
     + 5 * (df["arm"] == "b") + rng.normal(size=n))

encode = make_column_transformer(
    (OneHotEncoder(drop="first"), ["arm"]),
    remainder="passthrough",
    verbose_feature_names_out=False,
)
pipe = make_pipeline(encode, earth.Earth(max_degree=2)).fit(df, y)
print(pipe[-1].summary())
```

The pipeline passes a numpy array to the estimator, so the terms are named `x0`, `x1`, and so on, in the order of the transformed columns.

## Pipelines and grid search

The estimators clone, pickle, and take `sample_weight` through metadata routing. A pickle is meant to be loaded by the same version of pymars: it holds the private module paths of the classes, which can change in any release. To keep a fit across versions, save `est.mars_.to_dict()` (arrays and plain numbers), and rebuild it with `pymars.MarsFit.from_dict`. The dict has the key `"format"`, which changes when a key or a dtype changes; `from_dict` raises ValueError for a format that it does not read. In a `Pipeline` the parameters have the prefix of the step name, for example `earth__max_degree`.

```python
import numpy as np
import pymars as earth
from sklearn.model_selection import GridSearchCV

rng = np.random.default_rng(1)
X = rng.uniform(size=(400, 2))
y = 3 * np.maximum(X[:, 0] - 0.5, 0) + X[:, 1] + rng.normal(scale=0.2, size=400)

grid = GridSearchCV(
    earth.Earth(),
    {"max_degree": [1, 2], "penalty": [2, 3, 4]},
    cv=3,
)
grid.fit(X, y)
print(grid.best_params_)
```

`max_degree` and `penalty` change the model that the pruning pass picks, so they are the usual parameters to search over.

## What the fit costs

The core runs in Python with numpy, and with numba-compiled loops in the knot scan wherever numba is installed. pip installs numba with pymars on every platform that has numba wheels. It does not on Intel Macs, on PyPy, on Windows ARM64 with Python 3.10, 3.11 or 3.15, nor on a Python version newer than the latest numba release supports (3.15 at the time of writing). There the fit uses the slower numpy code, and the first fit of each process issues one `pymars.NumbaUnavailableWarning` (a `UserWarning`) that says so. The two give the same models except at near-ties (STOP-7 of [algorithm.md](algorithm.md)), where two candidates are equal up to rounding and the two can choose different terms. numba compiles the loops once and keeps them in a disk cache; the first fit in a process loads them, which adds about 0.3 s.

To install numba yourself, run `pip install numba` in an environment where it has wheels. On a platform where pip fails on numba (musllinux, 32-bit Python, or a CPU other than x86-64 and ARM64), install pymars with `pip install --no-deps git+https://github.com/alejandroschuler/mars` and then `pip install numpy scipy scikit-learn`. To run the numpy code without the warning, filter it:

```python
import warnings
import pymars

warnings.filterwarnings("ignore", category=pymars.NumbaUnavailableWarning)
```

In the benchmark ([validation/bench/REPORT.md](../validation/bench/REPORT.md); 10 covariates, degree 2, 21 terms, one thread), a fit with 10,000 cases takes 0.81 s in a fresh process and 0.43 s in a process that has already fitted, against 0.46 s for earth; with 100,000 cases it takes about 5.1 to 5.8 s, against 7.6 s for earth. The numpy code takes 1.2 s at 10,000 cases and 15 s at 100,000, 2.6 and 2.0 times earth's time. Below about 10,000 cases a fixed cost of about 0.06 s per fit dominates, so earth is faster there (0.02 s at 250 cases). Above that the time grows a little less than linearly in the number of cases. Weights cost little extra (0.45 s against 0.39 s at 1,000 cases). For many small fits, as in cross-validation, fit them in one process. The memory of the fit is O(n·(p + nk)).
