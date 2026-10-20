# Title: The pruning pass can delete the intercept

## Summary

`_pruning.py:297-317` protects the intercept only when it is the last term left. Otherwise the pruning pass treats it like any other term, and it can remove it when that gives the lowest GCV. The returned model then has no `(Intercept)` term. Friedman's pruning algorithm and earth never remove the intercept.

In the cases that I traced, the model was full rank when the intercept left. Removing it raised the RSS by less than removing any other term (by 0.015 to 3.9), because the other terms, many of them linear terms that also add 1 to the effective number of parameters, nearly span the constant.

The fixture `missingness_2d` in `tests/fixtures/reference_regression_cases.json` has no intercept either (its basis is `is_missing(x0)`, `is_missing(x1)`), so the current tests lock in the behavior.

## Reproduction

On `mars-earth` 1.0.4. Thirty Friedman #1 samples (100 cases, 5 covariates), degree 2, default settings:

```python
import numpy as np, pymars
from pymars._basis import ConstantBasisFunction

def friedman1(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 5))
    y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
         + 5*X[:, 4] + rng.standard_normal(n))
    return X, y

missing = []
for seed in range(100, 130):
    X, y = friedman1(100, seed)
    m = pymars.Earth(max_degree=2).fit(X, y)
    if not any(isinstance(b, ConstantBasisFunction) for b in m.basis_):
        missing.append(seed)
print("fits without an intercept:", len(missing), "of 30; seeds", missing)
X, y = friedman1(100, missing[0])
m = pymars.Earth(max_degree=2).fit(X, y)
print(len(m.basis_), "terms, first:", m.basis_[0])
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
```

Output:

```
fits without an intercept: 11 of 30; seeds [102, 105, 108, 114, 115, 118, 120, 123, 125, 127, 129]
11 terms, first: (Intercept) * x3
```

(The first basis function prints as `(Intercept) * x3`: it is a linear term whose parent is the intercept. The constant term itself is absent.)

earth, on the first sample (`data.csv` from above):

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
m <- earth(x, y, degree = 2)
cat("earth:", length(m$selected.terms), "terms; first coefficient name:", rownames(m$coefficients)[1], "\n")
```

Output:

```
earth: 13 terms; first coefficient name: (Intercept)
```

## Scale

In the fork's conformance runs (`validation/DIFFERENCES_legacy.md`), 102 of 332 legacy fits end without an intercept: 63 of 147 at the defaults, 27 of 150 in the matched mode, 11 of 27 in the span-grid fits and 1 of 6 on unscaled inputs. Among 100 random draws at the defaults, 43 do.

## Suggestion

Never remove the intercept in the pruning pass, as in Friedman (1991) and in earth. The 2.0 code in the fork does this and has a test for it.
