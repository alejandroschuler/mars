# Sample weights

`sample_weight` in `fit` holds case counts (frequency weights). This page says what that means for the fit. The rule ids point to [algorithm.md](algorithm.md), "Weights" (W-1 to W-8).

## The meaning

An integer weight w gives the same fit as w copies of the row. A zero weight gives the same fit as removing the row. Weights that are all 1 give the same fit as no weights (W-1, W-5).

The equality holds up to rounding. The weighted fit and the fit on repeated rows run different float64 sums, so where two candidates are equal up to rounding (a near-tie), the two fits can choose differently (W-1, STOP-7).

```python
import numpy as np
import pymars as earth

rng = np.random.default_rng(1)
n = 400
X = rng.uniform(size=(n, 2))
y = 3 * np.maximum(X[:, 0] - 0.5, 0) + X[:, 1] + rng.normal(scale=0.2, size=n)
w = rng.integers(0, 4, n)                  # integer weights, some of them 0

weighted = earth.Earth().fit(X, y, sample_weight=w)
rows = np.repeat(np.arange(n), w)
repeated = earth.Earth().fit(X[rows], y[rows])
print(np.allclose(weighted.predict(X), repeated.predict(X)))   # True
print(weighted.gcv_, repeated.gcv_)                            # the same to rounding
```

## The scale of the weights matters

Weights times a constant c act like c copies of each row (W-8). The sample size in the formulas below is the sum of the weights, so doubling every weight changes the GCV, the spans and the fit. earth differs here: it counts cases, and it ignores weights that are all equal.

Do not pass importance weights or survey weights as they are. Rescale them to `w * n / w.sum()` first, so that they sum to n. When some weight is not an integer and the mean of the weights differs from 1 by more than 1e-6 (relative), `fit` issues a UserWarning that says this (W-7). Integer weights never warn. The test uses all n rows as given, zero weights included.

## Zero weights

Rows of weight 0 are dropped before anything else (W-3). They count neither in n and N, nor in the spans, nor in the knot search. This is the same fit as one on the data without those rows, and it is a departure from earth, which replaces a zero weight by a very small positive weight, so that the row still counts as a case in its spans and knots.

For the classifier, a class that appears only on rows of weight 0 is not a class of the model, and the refit and the ConvergenceWarning ignore those rows (W-3, GLM-1, GLM-4).

A tiny positive weight is not a zero weight. Weights so small next to the others that their products with squared values underflow in float64, such as 1e-300 next to 1, leave those rows in N and in the spans but give them no effect on the sums of squares ([differences.md](differences.md), LIM-4). Keep the positive weights within a factor of about 1e12 of each other, or set the tiny ones to 0.

If every weight is 0, `fit` raises ValueError (the message matches `weight.*zero`). A weight below 0 or not finite, a weight array that is not 1-D with one entry per row, a bool, a 0-d array, and a total weight of 2^52 or more raise ValueError too (W-6). A scalar number is the weight of every row.

## How the weights enter the fit

N is the sum of the weights, taken in a way that does not depend on the order of the rows (W-4). It takes the place of the number of cases everywhere (W-2):

- **GCV.** RSS is the weighted sum of squares, the sum of w_i r_i². GCV = RSS / (N (1 - C(M)/N)²), where C(M) = M + d (M - 1) / 2 for penalty d, and GCV is infinite when C(M) is within a tolerance of N or above it (GCV-1, GCV-2, GCV-8). RSq uses the weighted total sum of squares about the weighted mean, and GRSq uses the GCV of the model and of the intercept alone, both with N (GCV-5, GCV-6).
- **Spans and knots.** The minspan formula uses the weight of the active cases, and the endspan is capped by N. The knot search runs on cumulative weight, so that a case of weight w covers w positions of the scan (SPAN-1, SPAN-5, KNOT-6). earth ignores weights here.
- **Forward pass.** The reduction of RSS that picks each candidate uses weighted sums, and so do the collinearity test and the choice between a pair of hinges and a single hinge (LA-3, LA-7, W-8).
- **Pruning.** Every subset of the pruning pass is fit by weighted least squares, so RSS(U) and the GCV of each size use the weights and N (PRUNE-2 to PRUNE-4). The final coefficients are the weighted least-squares fit of y on the selected columns (PRUNE-8).
- **Classification.** The logistic refit maximizes the weighted log-likelihood (GLM-2).

The default term limit does not depend on n or on the weights (LIMIT-1).

## What was checked

Integer weights, zeros included, were compared with earth without weights on the repeated rows, since earth's own weighted fit uses another meaning. The 12 such fits whose response is not constant agree with earth. For non-integer weights, only the pruning pass on earth's forward basis was compared with weighted earth, and the two agree ([validation/DIFFERENCES.md](../validation/DIFFERENCES.md)). Where the fits have zero weights, earth's sums of squares differ from pymars's by up to about 6e-9 relative, because earth still gives those rows a tiny weight (W-3).
