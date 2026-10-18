# Upstream drafts

These files are drafts. Nobody has posted them, and nothing here is sent to `edithatogo/mars` or anywhere else. The user decides later whether to post any of them, and where.

Each draft starts with a title line (`# Title: ...`) and has a body that can be pasted into an issue. The posting order below puts the offer of the 2.0 code first, because the issues refer to it.

| File | Kind | Finding | One line |
|---|---|---|---|
| `01-offer-pymars-2.md` | note | all | Offers the 2.0 code, the earth conformance harness and fixtures, and the validation report as one piece of added evidence. |
| `02-weights.md` | issue | F1 | `sample_weight` changes the fit with its scale, and the formulas for the sample size disagree with each other. |
| `03-pruned-intercept.md` | bug | F3 | The pruning pass can delete the intercept. |
| `04-absolute-stop.md` | bug | F6 | The forward pass stops on an absolute epsilon, so the fit depends on the units of y and x. |
| `05-missing-values.md` | bug | F8 | With `allow_missing=True`, every row with a missing value gets a NaN prediction. |
| `06-binary-outcomes.md` | bug | F9 | `GLMEarth` and `EarthClassifier` refit with an L2 penalty, and three classes are fitted as a regression. |
| `07-sklearn-checks.md` | bug | F11 | `Earth` fails 5 of 48 scikit-learn estimator checks and `GLMEarth` fails 13 of 59. |
| `08-dependency-bounds.md` | bug | F12 | The declared bounds are wrong: scikit-learn 1.5 fails, and the Python versions disagree. |
| `09-forward-penalty.md` | departure | F2 | The GCV penalty changes which knots the forward pass finds. |
| `10-gcv-convention.md` | departure | F4 | The effective number of parameters follows another convention than earth's and Friedman's, and the docstring says otherwise. |
| `11-redundant-hinge.md` | departure | F5 | Every step adds both hinges of a pair, and the forward basis is often rank deficient. |
| `12-spans-and-knots.md` | departure | F7 | `minspan_alpha` and `endspan_alpha` default to 0, so the candidate knots are not the ones of Friedman's rule. |
| `13-speed.md` | issue | F10 | A fit takes 3 to 4 orders of magnitude longer than earth, and the cost grows almost with the square of n. |
| `14-reference-tests.md` | issue | F13 | The reference tests compare the code with its own stored output at loose tolerances. |
| `15-split-out-code.md` | proposal | none | Proposes to split out or remove the code that does not fit models. |

## How the drafts were checked

Every Python and R block in `02` to `14` was run, and the output shown in each draft is the output of that run. The runs used the `mars-earth` 1.0.4 wheel in `.venv-legacy` (Python 3.12.13, numpy 2.5.3, scikit-learn 1.9.1), R 4.4.3 and earth 5.3.4 on macOS, Apple silicon, with one BLAS thread. `02` (weights) and `07` (scikit-learn checks) need `sample_weight` and the current estimators, which the 1.0.4 wheel lacks or differs on, so they ran on the Python code of the tag `legacy-1.0.4-head` (upstream commit `d68b54a`) in the same venv, with that code first on `PYTHONPATH`. `08` ran in a second venv with scikit-learn 1.5.2. The first appendix of `VALIDATION_PLAN.md` and `validation/DIFFERENCES_legacy.md` give the full evidence for each finding, including the counts over the fixtures.

earth is used only as a black box in every block: the blocks call it and read the fields of its result.

## Before posting

- The numbers in `01-offer-pymars-2.md` about the 2.0 code are copied from `docs/differences.md` and `validation/DIFFERENCES.md` as of the commit that added these drafts. Refresh them from `validation/REPORT.md` and from `docs/differences.md` on the day of posting, and add the speed and the simulation results from the report.
- Replace the placeholder for the link to the fork (`<link to the fork>`) by the address of the repository, and say which commit or tag holds the 2.0 code.
- Post the issues one at a time. The note in `01` is written to stand alone and links to nothing that is unpublished.
