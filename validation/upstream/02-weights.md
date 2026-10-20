# Title: sample_weight changes the fit with the scale of the weights, and the span rules count weights differently

## Summary

`Earth.fit(X, y, sample_weight=w)` treats `w` as frequency weights in some places and as something else in others. As a result, the fit depends on the scale of `w`, and weights that are all equal do not give the unweighted fit. earth counts cases and treats equal weights as no weights, so its fit does not change when all weights are multiplied by a constant.

This is not about the choice of meaning. Frequency weights are what scikit-learn's `check_sample_weight_equivalence_on_dense_data` expects, and they are a fine choice. The issue is that the code does not apply that meaning everywhere.

## Where the code disagrees with itself

On the tag `legacy-1.0.4-head` (the `1.0.4` wheel has no `sample_weight` argument):

- the GCV denominator, the default term limit and a cap on the size of a candidate model use `sum(w)` as the sample size (`_forward.py:178-185`, `:216-221`, `:667`; `_pruning.py:87`);
- the minspan formula uses weighted counts;
- the endspan rule counts distinct values;
- with unit weights the size is still `n`, but the model still changes, because the weighted code computes the RSS by another route (see the issue on the absolute stopping rule).

## Reproduction

Run on the Python code of the tag `legacy-1.0.4-head`.

```python
import numpy as np, pymars
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(200))
np.savetxt("data.csv", np.column_stack([X, y]), delimiter=",",
           header="x1,x2,x3,x4,x5,y", comments="")
base = pymars.Earth(max_degree=1).fit(X, y)
print("no weights:", len(base.basis_), "terms")
for label, w in [("w = 1", np.ones(200)), ("w = 10", 10*np.ones(200)),
                 ("w = 1/n", np.ones(200)/200)]:
    m = pymars.Earth(max_degree=1).fit(X, y, sample_weight=w)
    print(label, len(m.basis_), "terms; same terms as unweighted:",
          [str(b) for b in m.basis_] == [str(b) for b in base.basis_],
          "; max |pred diff| = %.3g" % np.abs(m.predict(X) - base.predict(X)).max())
```

Output:

```
no weights: 9 terms
w = 1 8 terms; same terms as unweighted: False ; max |pred diff| = 0.463
w = 10 16 terms; same terms as unweighted: False ; max |pred diff| = 3.89
w = 1/n 1 terms; same terms as unweighted: False ; max |pred diff| = 13.2
```

Weights of 1/n make the sample size 1, so the default term limit is 1 and the fit is the intercept alone. Under frequency weights that is the expected reading (the data then count as one case), but it surprises users who pass normalized weights, and nothing warns them. Weights of 10 make the sample size 2,000 for 200 cases, which shrinks the GCV penalty factor from about 2 to about 1.06, and the fit grows from 9 to 16 terms, as it would for 10 copies of the data.

earth, on the same data (`data.csv` from the block above):

```r
suppressMessages(library(earth))
d <- read.csv("data.csv"); x <- as.matrix(d[, 1:5]); y <- d$y
b <- earth(x, y, degree = 1)
cat("earth, no weights:", length(b$selected.terms), "terms\n")
for (w in list(rep(1, 200), rep(10, 200), rep(1/200, 200))) {
  m <- earth(x, y, degree = 1, weights = w)
  cat("earth, w[1] =", w[1], ":", length(m$selected.terms), "terms; same as unweighted:",
      identical(m$selected.terms, b$selected.terms) &&
        max(abs(predict(m, x) - predict(b, x))) < 1e-8, "\n")
}
```

Output:

```
earth, no weights: 10 terms
earth, w[1] = 1 : 10 terms; same as unweighted: TRUE 
earth, w[1] = 10 : 10 terms; same as unweighted: TRUE 
earth, w[1] = 0.005 : 10 terms; same as unweighted: TRUE 
```

## Suggestion

Pick one meaning and apply it in every rule. For frequency weights that means `sum(w)` in the GCV denominator, in the default term limit, in the minspan and endspan rules and in the size cap, and it means that integer weights give the same fit as repeated rows. A warning when the weights sum to less than the number of cases would also help.

The 2.0 code in the fork follows this rule and documents it (`docs/weights.md`); it is offered in a separate note. In the fork's conformance runs on weighted data, the weighted fit matches the fit on repeated rows, and its forward pass matches earth's on the repeated rows.

## Evidence

`validation/DIFFERENCES_legacy.md` in the fork, section "F1, weights": 16 weighted fits on the tag, of which 14 with integer weights add the same forward terms as the code on repeated rows in 13 cases, and 8 of the 14 select other terms, with prediction differences up to 14.3.
