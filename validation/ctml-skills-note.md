# Note for the ctml-skills supervised-learning skill

This note is for the maintainer of the `supervised-learning` skill in the stats-research plugin. It proposes edits to the skill's MARS row and its Python paragraph, based on the validation of pymars 2.0 in this repository (`validation/REPORT.md`). Nothing in the ctml-skills repository has been changed.

## Verdict

pymars 2.0 can be the MARS learner in a Python super learner library.

- On earth's own fixtures it takes the same forward steps and the same pruning path as earth, except at 7 documented rule differences and near-ties (`validation/DIFFERENCES.md`).
- In the simulation study, its excess risk is within 5 percent of earth's on 48 of 51 cells, including every cell with interactions, many irrelevant covariates, correlated covariates or a binary outcome.
- One cell fails: on pure noise at 200 cases it is 7.6 percent worse than earth. On pure noise it is 3 to 8 percent worse at every sample size, and it keeps slightly more noise terms than earth; the study does not show which rule causes this (#173). Two other cells are not decided.
- The legacy pymars (the `mars-earth` 1.0.4 wheel on PyPI) was 1.2 to 1.6 times worse than earth on every nonlinear truth at 200 cases, and 68 to 1,400 times slower. The skill's current advice to install from the repository and not from PyPI stays right.

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

All other parameters stay at their defaults, which are earth's. `max_degree` must be set, because its default is 1, as in earth.

## Weights

pymars treats `sample_weight` as frequency weights, as scikit-learn does. Inverse-probability weights should be rescaled to mean 1 before the fit (`w * n / w.sum()`), and pymars warns when they are not. Zero weights drop the case.

## Speed

At 10 covariates and degree 2, against earth at its defaults, on one thread:

| Cases | pymars, first fit in a process | pymars, later fits | pymars without numba |
|---|---|---|---|
| 250 | 21 times earth | 3.5 times | 8 times |
| 10,000 | 1.7 times | about the same as earth | 2.6 times |
| 100,000 | 0.76 times | 0.67 times | 2.0 times |

pymars uses numba where numba has wheels. Intel Macs use the slower numpy code, which gives the same models except at near-ties. At a few hundred cases a fit costs about 0.06 s, plus about 0.3 s for the first fit in a process, against earth's 0.02 s. In a cross-validated library at small n, MARS is therefore not the cheapest learner in Python, although it is still far from the cost of boosting with tuning.

## Proposed edits to the skill

The MARS row (line 16 today):

> | **MARS** (multivariate adaptive regression splines; `earth` in R, `pymars` in Python) | Very fast | Works well with one set of hyperparameters. Set the interaction degree high enough and keep pruning on. |

could read:

> | **MARS** (multivariate adaptive regression splines; `earth` in R, `pymars` in Python) | Very fast in R. In Python, about as fast as earth from 10,000 cases; at a few hundred cases a fit costs about 0.06 s | Works well with one set of hyperparameters. Set the interaction degree high enough and keep pruning on. |

The Python paragraph (line 23 today) could add, after "as in earth.":

> Its `sample_weight` is a frequency weight, so rescale inverse-probability weights to mean 1. For binary outcomes at a few hundred cases, the default logistic refit is unpenalized, as in earth, and can overfit; `glm_alpha`, an L2 penalty, helps there.

Line 131 ("Among spline learners, MARS is very fast and works with one setting.") holds for both implementations at moderate and large n, so it needs no change.

## Open items

Two defaults wait for a decision (issue #175): whether `EarthClassifier` keeps the unpenalized refit (`glm_alpha=0`) as its default, and whether numba stays a required dependency. If either changes, the settings and the Python paragraph above change with it.
