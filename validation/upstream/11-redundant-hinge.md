# Title: Every forward step adds both hinges of a pair, and the forward basis is often rank deficient

## Summary

Each forward step adds both hinges of a pair, even when the parent and the variable already appear together in the model. For a parent b, b·(t − x)₊ = b·(x − t)₊ − b·x + t·b. The term b·x is already in the span when an earlier pair on the same parent and variable is present, because b·(x − s)₊ − b·(s − x)₊ = b·x − s·b. The second hinge is then an exact linear combination of the first new hinge and existing columns.

Some consequences: the model carries a column that adds nothing, it uses one term of the term limit, it adds to the effective number of parameters in the GCV, and the least-squares solve is rank deficient. In 16 of 30 forward bases in the reproduction below, the basis is rank deficient.

earth adds a single hinge in that case. It also rejects a candidate knot when the existing columns explain more than 99 percent of the variation of the new column (a tolerance of 0.01 while its internal column counter is small, then 1e-5). One divergence in a trace run (`trace = 9`) comes from that tolerance: earth rejected the legacy code's fourth knot (0.148) at each step while its column counter ran from 3 to 14, and chose it when the counter reached 16.

## Reproduction

On `mars-earth` 1.0.4. Thirty Friedman #1 samples (100 cases, 5 covariates), degree 2, default settings. The block counts the samples whose forward basis has fewer independent columns than columns, and saves the first such sample for earth.

```python
import numpy as np, pymars

def friedman1(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 5))
    y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
         + 5*X[:, 4] + rng.standard_normal(n))
    return X, y

def forward_rank(X, y):
    m = pymars.Earth(max_degree=2).fit(X, y)
    cols = m.record_.fwd_basis_[-1]
    B = np.column_stack([b.transform(X, np.zeros_like(X, bool)) for b in cols])
    return B.shape[1], np.linalg.matrix_rank(B)

first = None
deficient = 0
for seed in range(100, 130):
    ncol, rank = forward_rank(*friedman1(100, seed))
    if rank < ncol:
        deficient += 1
        if first is None:
            first = seed
            print("seed", seed, ": forward basis has", ncol, "columns, rank", rank)
print("rank-deficient forward bases:", deficient, "of 30")
X, y = friedman1(100, first)
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
```

Output:

```
seed 102 : forward basis has 21 columns, rank 20
rank-deficient forward bases: 16 of 30
```

earth on that sample (`data.csv` from above), forward pass only:

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
m <- earth(x, y, degree = 2, pmethod = "none")
cat("earth forward basis:", ncol(m$bx), "columns, rank", qr(m$bx)$rank, "\n")
```

Output:

```
earth forward basis: 20 columns, rank 20
```

## Scale

In the fork's conformance runs (`validation/DIFFERENCES_legacy.md`), this is the first difference from earth in 126 fits, and it happens before any other difference in 37 of the 150 matched-mode fits, most often at step 2. Along the steps where both programs agree, the RSS agrees to 8e-15 relative. At a choice divergence, earth's collinearity tolerance rejected the legacy knot in all 113 of the choices of this kind, 81 of them random draws.

A knot close to the end of the data can leave a term fitted to a single case. In one random draw (D5, degree 1, 200 cases) the legacy code selects `h(0.0425783 - x9)`, which is positive at 1 of the 200 training cases, with coefficient 114,773. Its test error is 126,243, against 0.037 for earth.

## Suggestion

Add a single hinge when the parent and the variable already appear together, and reject candidates whose new column is almost in the span of the existing columns, as in Friedman (1991) and earth. The 2.0 code in the fork follows earth's rules here, with each one stated in `docs/algorithm.md`.
