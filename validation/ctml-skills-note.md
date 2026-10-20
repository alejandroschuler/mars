# Note for the ctml-skills supervised-learning skill

This note is for the maintainer of the `supervised-learning` skill in the stats-research plugin. It proposes edits to the skill's MARS row and its Python paragraph, based on the validation of pymars 2.0 in this repository (`validation/REPORT.md`). Nothing in the ctml-skills repository has been changed.

## Verdict

pymars 2.0 can be the MARS learner in a Python super learner library.

- On earth's own fixtures it takes the same forward steps and the same pruning path as earth, except at 4 documented rule differences and near-ties (`validation/DIFFERENCES.md`).
- In the simulation study, its excess risk is within 5 percent of earth's on 51 of 51 cells in a rerun after the pair-rule change of #185 (48 of 51 before), including every cell with interactions, many irrelevant covariates, correlated covariates or a binary outcome.
- Before #185 one cell failed (pure noise at 200 cases, 7.6 percent worse than earth) and two were not decided. The cause was the pair rule of LA-7 (#173), and after #185 all 51 cells are equivalent. On lognormal covariates the new rule costs up to 1.9 percent of excess risk on one function (D4 at 200 cases), a known limit in the report.
- The legacy pymars (the `mars-earth` 1.0.4 wheel on PyPI) was 1.2 to 1.6 times worse than earth on every nonlinear truth at 200 cases, and 69 to 1,400 times slower than pymars 2.0 at 250 to 2,000 cases. The skill's current advice to install from the repository and not from PyPI stays right.

## Settings to use

```python
from pymars import EarthRegressor, EarthClassifier

EarthRegressor(max_degree=2)
EarthClassifier(max_degree=2)
```

Install it with:

```
pip install git+https://github.com/alejandroschuler/mars
```

All other parameters stay at their defaults, which are earth's, except the classifier's `glm_alpha` (below). `max_degree` must be set, because its default is 1, as in earth.

## Weights

pymars treats `sample_weight` as frequency weights, as scikit-learn does. Inverse-probability weights should be rescaled to mean 1 before the fit (`w * n / w.sum()`), and pymars warns when they are not. Zero weights drop the case.

## Speed

At 10 covariates and degree 2, against earth at its defaults, on one thread:

| Cases | pymars, first fit in a process | pymars, later fits | pymars without numba |
|---|---|---|---|
| 250 | 21 times earth | 3.5 times | 8 times |
| 10,000 | 1.7 times | about the same as earth | 2.6 times |
| 100,000 | 0.76 times | 0.67 times | 2.0 times |

pymars uses numba where numba has wheels. Intel Macs and Python 3.16 and later use the slower numpy code, which gives the same models except at near-ties; the first fit of such a process issues one warning. At a few hundred cases a fit costs about 0.06 s, plus about 0.3 s for the first fit in a process, against earth's 0.02 s. In a cross-validated library at small n, MARS is therefore not the cheapest learner in Python, although it is still far from the cost of boosting with tuning.

## Proposed edits to the skill

The MARS row (line 16 today):

> | **MARS** (multivariate adaptive regression splines; `earth` in R, `pymars` in Python) | Very fast | Works well with one set of hyperparameters. Set the interaction degree high enough and keep pruning on. |

could read:

> | **MARS** (multivariate adaptive regression splines; `earth` in R, `pymars` in Python) | Very fast in R. In Python, about as fast as earth from 10,000 cases; at a few hundred cases a fit costs about 0.06 s | Works well with one set of hyperparameters. Set the interaction degree high enough and keep pruning on. |

The Python paragraph (line 23 today) could add, after "as in earth.":

> Its `sample_weight` is a frequency weight, so rescale inverse-probability weights to mean 1. For binary outcomes, the logistic refit has an L2 penalty by default (`glm_alpha=1.0`, the strength of scikit-learn's `LogisticRegression` at `C=1`); `glm_alpha=0` is earth's unpenalized refit, the earth-compatible setting. The validation used `glm_alpha=0`: at a few hundred cases its probabilities were overconfident (calibration slopes 0.36 to 0.44 at 200 cases, for pymars and earth alike). The validation did not test the penalized default.

Line 131 ("Among spline learners, MARS is very fast and works with one setting.") holds for both implementations from about 10,000 cases. At 1,000 cases pymars takes 1.9 to 7.9 times earth's time, so the line could add "in Python, from about 10,000 cases".

## Open items

Both defaults of issue #175 are decided (2026-10-19): `EarthClassifier` has `glm_alpha=1.0`, with 0 as the earth-compatible setting, and numba stays a dependency where it has wheels. The settings and the Python paragraph above follow them.
