# Title: The effective number of parameters follows a different convention than earth's, and the docstring says it follows py-earth

## Summary

The effective number of parameters in `_util.py:82` is C = M + d·H, where M is the number of terms, H the number of hinge terms and d the penalty. This charges d once for each hinge term. It charges nothing for a linear term, and it counts columns instead of the rank.

earth, and py-earth, charge d once per knot instead: C = M + d·(M − 1)/2, which is d/2 for every term after the intercept. When every term after the intercept is a hinge, the penalty part of C at the same d is twice as large in pymars (d per hinge against d/2), and the total cost of a hinge is 1 + d against 1 + d/2. Friedman (1991) charges d once per nonconstant basis function, which is closer to pymars for hinges but still charges linear terms.

Two more differences add up with this one:

- the default d is 3 at every degree in pymars, where Friedman and earth use 2 for additive models (degree 1) and 3 otherwise;
- the docstring says that the formula follows py-earth, which it does not.

The convention matters because the GCV decides the size of the pruned model. The same data and the same terms give different selected sizes.

## Reproduction

On `mars-earth` 1.0.4. A degree-1 fit with only hinge terms (`allow_linear=False`) and `penalty=2`. The first block shows that the code's `gcv_` is the RSS formula with C = M + d·H, and what earth's convention would give for the same model. The second block shows that earth's reported GCV is the same formula with C = M + d·(M − 1)/2.

```python
import numpy as np, pymars
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(200))
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
n = len(y)
m = pymars.Earth(max_degree=1, penalty=2.0, allow_linear=False).fit(X, y)
M = len(m.basis_)
H = M - 1  # every term after the intercept is a hinge here
print("terms", M, "hinges", H, "rss", m.rss_, "gcv_", m.gcv_)
for name, C in [("pymars C = M + d*H", M + 2.0*H),
                ("earth C = M + d*(M-1)/2", M + 2.0*(M-1)/2)]:
    print(f"{name}: C = {C:g}, gcv = {m.rss_/n/(1-C/n)**2:.6f}")
```

Output:

```
terms 13 hinges 12 rss 536.636765356276 gcv_ 4.039570667742683
pymars C = M + d*H: C = 37, gcv = 4.039571
earth C = M + d*(M-1)/2: C = 25, gcv = 3.504567
```

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
m <- earth(x, y, degree = 1, penalty = 2, Auto.linpreds = FALSE)
M <- length(m$selected.terms)
cat("earth: terms", M, "rss", m$rss, "gcv", m$gcv, "\n")
cat("earth C = M + d*(M-1)/2 =", M + 2*(M-1)/2, "gives gcv",
    m$rss / 200 / (1 - (M + 2*(M-1)/2)/200)^2, "\n")
```

Output:

```
earth: terms 10 rss 570.2508 gcv 3.481278 
earth C = M + d*(M-1)/2 = 19 gives gcv 3.481278 
```

(The two fits select different numbers of terms, 13 and 10. The point of the blocks is the formula, not the fits: each program's reported GCV equals its own formula, and the two formulas give different values for the same terms.)

## Scale

In 5 fits of the fork's conformance runs, both programs add the same forward terms, but at the sizes where both pruning paths keep the same subset, their GCVs differ by up to 4.5 percent, and the selected sizes differ. For example, with one pair of hinges and no linear term, at size 3 the legacy C is 3 + 3·2 = 9 and earth's is 3 + 2·2/2 = 5. In the matched mode, where the penalty is halved, the two conventions agree for hinge-only models with an intercept, and no fit disagrees.

## Suggestion

State the convention in the docstring and the documentation, and choose it on purpose. To follow earth, use C = M + d·(M − 1)/2 with d = 2 at degree 1 and 3 otherwise. The 2.0 code in the fork follows earth and keeps the departures from it in a table (`docs/algorithm.md`, "Departures from earth").
