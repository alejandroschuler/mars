# Title: The forward pass stops on an absolute epsilon, so the fit depends on the units of y and x

## Summary

`_forward.py:240` stops the forward pass when the best candidate (by GCV) lowers the RSS by at most machine epsilon (about 2.2e-16). That is an absolute number. For an RSS of 4 or more, adjacent doubles are more than that epsilon apart, so subtracting it changes nothing, and the test only asks whether the computed RSS fell at all. When the true decrease is tiny, the answer turns on the last bits of two SVD solves. When the RSS itself is near 2e-16 (for example y times 1e-9), the test stops the search at once.

earth stops on relative rules: R-squared changes by less than `thresh = 0.001`, R-squared reaches 0.999, or GRSq falls below -10. Its fit does not change with the units of y or x.

The same effect probably explains the cross-run flakiness behind the commits on 2026-05-03 that widened the tolerances of the reference tests.

## Reproduction

On `mars-earth` 1.0.4, degree 1, default settings. Friedman #1, 200 cases, 5 covariates.

```python
import numpy as np, pymars
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(200))
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
base = [str(b) for b in pymars.Earth(max_degree=1).fit(X, y).record_.fwd_basis_[-1]]
for s in [1e-9, 1.0, 1e9]:
    f = [str(b) for b in pymars.Earth(max_degree=1).fit(X, s*y).record_.fwd_basis_[-1]]
    m = pymars.Earth(max_degree=1).fit(X, s*y)
    print(f"y * {s:g}: forward terms {len(f)}, final terms {len(m.basis_)}, "
          f"forward pass equals the unscaled one: {f == base}")
Xs = X.copy(); Xs[:, 0] *= 1e8
m = pymars.Earth(max_degree=1).fit(Xs, y)
print("x0 * 1e8: final terms", len(m.basis_))
```

Output:

```
y * 1e-09: forward terms 10, final terms 8, forward pass equals the unscaled one: False
y * 1: forward terms 11, final terms 9, forward pass equals the unscaled one: True
y * 1e+09: forward terms 10, final terms 8, forward pass equals the unscaled one: False
x0 * 1e8: final terms 8
```

earth on the same data (`data.csv` from above):

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
for (s in c(1e-9, 1, 1e9)) {
  m <- earth(x, s * y, degree = 1)
  cat("earth, y *", s, ":", length(m$selected.terms), "final terms; forward terms", nrow(m$dirs), "\n")
}
```

Output:

```
earth, y * 1e-09 : 10 final terms; forward terms 16 
earth, y * 1 : 10 final terms; forward terms 16 
earth, y * 1e+09 : 10 final terms; forward terms 16 
```

Multiplying y by 1e-9 scales the RSS by 1e-18. In a fit whose starting RSS is about 2.6e-17 (the fixture `S12_y_1em9` in the fork), the code returns the intercept alone in all three of the modes that were run, where earth fits the base knot. Unit weights are affected as well: the weighted code computes the RSS differently, and the stop test sees other bits.

## Suggestion

Use a relative stopping rule: a change of R-squared below a threshold, as in earth, or a decrease of the RSS relative to the total sum of squares. Do not compare an RSS with an absolute number. The 2.0 code in the fork does this, and its conformance tests include y and x rescaled by 1e-9 and 1e9.
