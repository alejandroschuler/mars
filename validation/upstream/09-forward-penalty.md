# Title: The GCV penalty changes which knots the forward pass finds

## Summary

The forward pass (`_forward.py:706`) picks the candidate with the lowest GCV and uses the RSS only to break ties. All pairs of hinges at one step have the same effective number of parameters, so GCV ranks them as the RSS does. A single linear candidate is different: it raises the effective number of parameters by 1, where a pair raises it by 2 + 2d (d is the penalty). A larger penalty therefore favors linear terms, and it rules candidates out sooner, since GCV is infinite once the effective number of parameters reaches n. The penalty thus changes which knots the forward pass finds.

Friedman's forward algorithm and earth pick the candidate with the largest RSS reduction. In earth the forward pass is the same for every penalty, as the run below shows.

## Reproduction

On `mars-earth` 1.0.4. Friedman #1, 200 cases, 5 covariates, degree 2, 21 terms, penalties 0, 2, 3 and 10:

```python
import numpy as np, pymars
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(200))
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
fwd = {}
for pen in [0.0, 2.0, 3.0, 10.0]:
    m = pymars.Earth(max_degree=2, penalty=pen, max_terms=21).fit(X, y)
    fwd[pen] = [str(b) for b in m.record_.fwd_basis_[-1]]
    print("penalty", pen, ": forward pass has", len(fwd[pen]), "terms;",
          "same as penalty 0:", fwd[pen] == fwd[0.0])
```

Output:

```
penalty 0.0 : forward pass has 21 terms; same as penalty 0: True
penalty 2.0 : forward pass has 21 terms; same as penalty 0: False
penalty 3.0 : forward pass has 21 terms; same as penalty 0: False
penalty 10.0 : forward pass has 21 terms; same as penalty 0: False
```

earth, on the same data, with the forward pass alone (`pmethod = "none"`) and the other forward settings fixed:

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
ref <- NULL
for (pen in c(0, 2, 3, 10)) {
  m <- earth(x, y, degree = 2, nk = 21, pmethod = "none", penalty = pen,
             fast.k = 0, thresh = 0, Auto.linpreds = FALSE)
  key <- paste(c(m$dirs), c(m$cuts))
  if (is.null(ref)) ref <- key
  cat("earth, penalty", pen, ": forward pass has", nrow(m$dirs), "terms; same as penalty 0:",
      identical(key, ref), "\n")
}
```

Output:

```
earth, penalty 0 : forward pass has 18 terms; same as penalty 0: TRUE 
earth, penalty 2 : forward pass has 18 terms; same as penalty 0: TRUE 
earth, penalty 3 : forward pass has 18 terms; same as penalty 0: TRUE 
earth, penalty 10 : forward pass has 18 terms; same as penalty 0: TRUE 
```

## Scale

In the fork's conformance runs (`validation/DIFFERENCES_legacy.md`), this is the first difference from earth in 104 of the legacy fits, 82 of them random draws at the defaults, where the legacy code picks a linear term by GCV and earth picks the pair with the lower RSS. It also stops the forward pass early at small n: when every candidate has an effective number of parameters of at least n, the GCV is infinite and the search ends while earth's goes on.

## Suggestion

Choose the forward candidate by the RSS reduction, and use GCV only for the backward pass and for the optional stopping rule on GRSq, as in Friedman (1991) and earth. The 2.0 code in the fork does this.
