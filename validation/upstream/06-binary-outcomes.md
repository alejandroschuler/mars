# Title: GLMEarth and EarthClassifier refit with an L2 penalty, and three classes are fitted as a regression

## Summary

1. `GLMEarth` and `EarthClassifier` refit the selected basis with scikit-learn's `LogisticRegression` at its default `C=1` (`glm.py:30`, `_sklearn_compat.py:359`). This `C` is scikit-learn's inverse penalty strength. It has nothing to do with the effective number of parameters C of the GCV. Against an unpenalized fit on the same basis, the coefficients shrink by about 20 percent and the fitted probabilities move by up to 0.08. earth fits an unpenalized GLM on the selected terms.
2. `GLMEarth` has no `predict_proba`, predicts labels, inherits from a regressor, and fails 13 scikit-learn estimator checks (see the issue on the checks).
3. `EarthClassifier` with three classes fits the integer class codes as one regression. earth fits one indicator response per class.

## Reproduction

On `mars-earth` 1.0.4.

```python
import warnings
import numpy as np, pymars
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
X = rng.uniform(size=(400, 5))
f = (10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3] + 5*X[:, 4])
yb = (np.random.default_rng(12).uniform(size=400) < 1/(1+np.exp(-(f-f.mean())/3))).astype(int)
clf = pymars.EarthClassifier().fit(X, yb)
B = clf.earth_._build_basis_matrix(X, clf.basis_, np.zeros_like(X, bool))
unpen = LogisticRegression(penalty=None, max_iter=5000).fit(B, yb)
p, q = clf.predict_proba(X)[:, 1], unpen.predict_proba(B)[:, 1]
print("max |p(EarthClassifier) - p(unpenalized logistic on the same basis)| = %.3f" % np.abs(p - q).max())
glm = pymars.GLMEarth(family="logistic").fit(X, yb)
print("GLMEarth has predict_proba:", hasattr(glm, "predict_proba"), "; predict returns", np.unique(glm.predict(X)))
y3 = np.digitize(f, np.quantile(f, [1/3, 2/3]))
c3 = pymars.EarthClassifier().fit(X, y3)
print("3 classes: Earth fitted on the integer codes", np.unique(y3), "->", type(c3.earth_).__name__,
      "with a single response; basis terms:", len(c3.earth_.basis_))
```

Output:

```
max |p(EarthClassifier) - p(unpenalized logistic on the same basis)| = 0.080
GLMEarth has predict_proba: False ; predict returns [0 1]
3 classes: Earth fitted on the integer codes [0 1 2] -> Earth with a single response; basis terms: 8
```

When the classes are nearly separable the penalty matters more. In the fork's runs (`validation/DIFFERENCES_legacy.md`, S14 and S20), the penalty moves the fitted probabilities by up to 0.60 and 0.65.

## Suggestion

Refit with an unpenalized logistic regression, or expose the penalty as an explicit parameter whose default is no penalty, so that the probabilities follow earth's `glm` step. Give the classifier `predict_proba` and a classifier base class, and fit one response per class for more than two classes. The 2.0 code in the fork merges `GLMEarth` into `EarthClassifier` and does this.
