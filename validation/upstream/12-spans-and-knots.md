# Title: The defaults for minspan_alpha and endspan_alpha are 0, so the candidate knots are not Friedman's

## Summary

By default `minspan_alpha = endspan_alpha = 0`, so every distinct value of a covariate is a candidate knot, except the largest when the intercept is the parent. earth and py-earth set the spacing between knots (minspan) and the margin at each end (endspan) with Friedman's equations (43) and (45). In those equations a small probability α (0.05 by default) sets how strongly the rule guards against knots that are fitted to runs of noise.

Further differences in the knot rules:

- pymars counts distinct values where earth counts cases;
- pymars rounds where earth truncates;
- earth uses a larger endspan for interaction terms (`Adjust.endspan`), and pymars has no such rule;
- earth centers the grid of candidate knots, and pymars does not.

With no margin at the ends, a knot can sit at the 2nd smallest value of a covariate, and a hinge there is fitted to very few cases.

## Reproduction

On `mars-earth` 1.0.4. One covariate and a response of pure noise, so that the forward pass picks knots anywhere it is allowed to. The block fits degree 1 with 21 terms, no linear terms, and reports the position of the lowest and the highest knot among the sorted cases (1 is the smallest case).

```python
import re
import numpy as np, pymars
rng = np.random.default_rng(7)
n = 100
x = rng.uniform(size=(n, 1))
y = rng.standard_normal(n)  # pure noise, so the forward pass picks knots anywhere
np.savetxt("data.csv", np.column_stack([x, y]), delimiter=",", header="x1,y", comments="")
m = pymars.Earth(max_degree=1, allow_linear=False, max_terms=21).fit(x, y)
knots = sorted({float(k) for b in m.record_.fwd_basis_[-1]
                for k in re.findall(r"(?:max\(0, x0 - |max\(0, )([0-9.]+)", str(b))})
order = np.sort(x[:, 0])
ranks = [int(np.searchsorted(order, k)) + 1 for k in knots]
print("legacy defaults: knot ranks (1 = smallest case) from", min(ranks), "to", max(ranks),
      "of", n)
```

Output:

```
legacy defaults: knot ranks (1 = smallest case) from 3 to 100 of 100
```

earth at its defaults for the spans, on the same data (`data.csv` from above), with the other forward settings matched:

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1, drop = FALSE]); y <- d$y
m <- earth(x, y, degree = 1, nk = 21, pmethod = "none", thresh = 0, fast.k = 0, Auto.linpreds = FALSE)
k <- sort(unique(m$cuts[m$cuts != 0]))
r <- sapply(k, function(v) sum(x[, 1] < v) + 1)
cat("earth defaults: knot ranks from", min(r), "to", max(r), "of", nrow(x), "\n")
```

Output:

```
earth defaults: knot ranks from 8 to 92 of 100
```

## Scale

In the fork's conformance runs (`validation/DIFFERENCES_legacy.md`), the first difference from earth is a knot that is in one program's candidate set and not in the other's, in 82 fits. In 61 of them, earth's minspan and endspan grid leaves out the legacy knot, or the legacy code's own thinned grid leaves out earth's knot. This covers the defaults and every span setting of the two span-grid datasets. The other 21 come from a quirk of earth's scan for a hinge parent, which the 2.0 code in the fork documents.

## Suggestion

Default `minspan_alpha` and `endspan_alpha` to 0.05, and follow earth's rules for counting cases, truncation, the larger endspan for interactions and the centered grid. If a different default is intended, say so in the docstring and the documentation. The 2.0 code in the fork follows earth's rules.
