# Title: With allow_missing=True, every row with a missing value gets a NaN prediction

## Summary

With `allow_missing=True`, the fit accepts NaN in `X`, but `predict` returns NaN for every row that has a NaN. Hinge and linear terms return NaN for a missing input (`_basis.py:261`, `:433`), and the `is_missing(...)` indicator terms that the fit adds do not rescue the product. In the probe below, 85 of 200 training rows have a NaN, and all 85 get a NaN prediction.

During the forward pass, each candidate also drops the rows with a NaN in its own columns (`_forward.py:94-99`), so candidates are scored on different subsets of rows and their RSS values are not comparable.

earth does not accept NaN in `x`, so there is no earth output to compare. The reference for the intended behavior is the documentation of the option: a model that was fitted with missing values should predict for rows with missing values.

## Reproduction

On `mars-earth` 1.0.4. Friedman #1 with 10 percent of the entries of `X` set to NaN at random.

```python
import numpy as np, pymars
rng = np.random.default_rng(1)
X = rng.uniform(size=(200, 5))
y = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3]
     + 5*X[:, 4] + rng.standard_normal(200))
Xm = X.copy()
Xm[np.random.default_rng(3).uniform(size=Xm.shape) < 0.1] = np.nan
m = pymars.Earth(max_degree=1, allow_missing=True).fit(Xm, y)
p = m.predict(Xm)
print("rows with a missing value:", int(np.isnan(Xm).any(axis=1).sum()), "of 200")
print("NaN predictions on the training rows:", int(np.isnan(p).sum()))
```

Output:

```
rows with a missing value: 85 of 200
NaN predictions on the training rows: 85
```

## Suggestion

Either make the missing-value path work, with a rule that treats a term as zero (or as its missing-indicator) for a missing input and scores all candidates on the same rows, or remove `allow_missing` and let the estimator raise a clear error, so that users impute or use scikit-learn's `SimpleImputer` with `add_indicator=True`. The 2.0 code in the fork raises.
