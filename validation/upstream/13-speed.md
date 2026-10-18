# Title: A fit takes three to four orders of magnitude longer than earth, and the cost grows almost with the square of n

## Summary

For every candidate (each parent, variable and distinct value), the forward pass rebuilds the whole basis matrix and solves a new least-squares problem by SVD (`_forward.py:662`, `:677-684`). It has no updating formula (Friedman 1991, equation 52) and no Fast MARS (Friedman 1993). The Rust trainer in `rust-runtime/` runs the same search, solves the normal equations, and is switched off.

One degree-2 fit with 300 cases and 5 covariates, below, takes 12.8 seconds, where earth takes under 0.01 seconds. A profile of such a fit shows 87,458 calls to `numpy.linalg.lstsq` and 1.3 million calls to the hinge `transform`, and the forward pass takes nearly all of the time.

## Reproduction

On `mars-earth` 1.0.4, one BLAS thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`), degree 2, default settings.

```python
import time
import numpy as np, pymars
rng = np.random.default_rng(2)
n, p = 300, 5
X = rng.uniform(size=(n, p))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(n))
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
t = time.perf_counter()
m = pymars.Earth(max_degree=2).fit(X, y)
print(f"pymars 1.0.4, n={n}, p={p}, degree 2: {time.perf_counter()-t:.1f} s, {len(m.basis_)} terms")
```

Output (this machine, Apple silicon; times vary by a few seconds between runs, and 15.7 s was seen on another run):

```
pymars 1.0.4, n=300, p=5, degree 2: 12.8 s, 16 terms
```

earth on the same data:

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
t <- system.time(m <- earth(x, y, degree = 2))[["elapsed"]]
cat(sprintf("earth, same data: %.3f s, %d terms\n", t, length(m$selected.terms)))
```

Output:

```
earth, same data: 0.009 s, 15 terms
```

## Scale

Friedman #1 with 10 covariates (5 of them irrelevant), each package at its defaults, one thread. The earth time is the minimum of 3 runs and the pymars time is one run. earth times near 0.003 s are close to the timer resolution, so the first ratios are rough.

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

At the larger sizes the time grows almost with the square of n. Libraries that tune MARS inside cross-validation (a super learner, for example) multiply these times by the number of folds and settings.

## Suggestion

Use the updating formula of Friedman (1991) and the Fast MARS queue of Friedman (1993), so that a candidate knot costs O(1) updates instead of a new solve. The 2.0 code in the fork does this and is compared with earth for speed in the validation report.
