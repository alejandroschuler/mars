# Differences from earth

pymars follows the rules of R's earth (5.3.4) where they fit scikit-learn, and it departs from them where they do not. This page lists the differences that a user can meet, in plain words. The full list is the "Departures from earth" table of [algorithm.md](algorithm.md), and the differences found by running both programs on the fixtures are triaged in [validation/DIFFERENCES.md](../validation/DIFFERENCES.md). The decisions behind them are in [dev/DECISIONS.md](../dev/DECISIONS.md).

Each item has one of the labels of the plan's triage:

- **rule**: an earth rule that pymars replaces by another documented rule, on purpose.
- **quirk**: an earth behavior that looks accidental. The spec says for each whether pymars copies it.
- **tie**: a near-tie, where two choices are equal up to rounding and either program may take either.
- **numeric**: a numerical difference, such as a rank-deficient solve.

None of the differences below is a bug. On the fixtures, both the fast code and the reference implementation agree with earth in full on 124 of 133 dataset fixtures and 197 of 200 random draws, and the fast code shows exactly the same differences as the reference. The other 9 dataset fixtures and 3 draws fall under the entries E1 to E7 of the triage. They are covered below, and each has the label rule or quirk (no tie and no numeric difference occurred).

## Weights

- **rule** Weights are case counts. earth counts cases in its GCV and ignores weights in its spans and knots; pymars uses the sum of the weights everywhere, so an integer weight w is the same as w copies of the row. The scale of the weights therefore changes the fit, where earth ignores weights that are all equal. See [weights.md](weights.md). (W-2, GCV-8)
- **rule** A zero weight removes the row. earth gives such a row a very small weight, so the row still counts as a case. (W-3)
- **rule** The classifier's warning about probabilities near 0 or 1 looks only at rows of positive weight, where R's `glm` also tests the rows of weight 0. (GLM-4)

## The fit

- **quirk, not copied** The fit does not depend on the units of the covariates (E1). earth chooses between a pair of hinges and a single hinge by comparing an absolute number with 0.01, so a covariate in other units can change its choice. pymars scales that number by the variances of the covariates, and it agrees with earth when the covariates have variance 1. (LA-7)
- **quirk, not copied** The fit does not depend on the row order. earth's knots can depend on it where equal values have different activity. pymars fixes the order. (KNOT-2)
- **rule** Exact ties between candidates, and between subsets in pruning, follow a fixed order, where earth lets rounding decide. Candidates that fit the data exactly (within 1e-14 of the residual sum of squares) tie, and the fixed order decides. (FWD-5, PRUNE-3, FWD-12)
- **rule** A linear column whose mean is large next to its range counts as a term, where R's `lm.fit` can drop it as a copy of the intercept. (LA-4)
- **quirk, not copied** (E5 to E7) With `auto_linpreds=False`, earth sometimes adds a hidden term that takes a slot and a queue entry, and removes it at the end. pymars has no hidden term. In 3 of 200 random draws this made earth take another term at a late step; earth's choice had the lower residual sum of squares on the common basis in each. (FWD-11)
- **quirk, not copied** With several responses, earth stops at an exact fit by an absolute floor, so its stop depends on the units of y. pymars uses a floor relative to the total sum of squares. (STOP-5)
- **quirk, copied** A hinge or linear term that a step adds uses two slots of the term limit, so `max_terms=21` allows at most 10 steps. The queue of Fast MARS addresses parents by slot, so the newest terms can wait a few steps before they are searched. `fast_k` of 1 or 2 acts as 3. The full list of copied quirks is in the "Quirks" section of [algorithm.md](algorithm.md).
- **quirk, not copied** `pmethod="none"` with `nprune` keeps the first `nprune` forward terms, and the statistics `rss_`, `gcv_`, `rsq_` and `grsq_` are those of that model. earth reports the statistics of the backward subset of that size, which are not those of the model it returns. (PRUNE-7, PRUNE-8)

## Statistics and degenerate data

- **rule** A constant y gives the intercept alone, with gcv infinite, rsq and grsq 0, and termination code 0 (`DEGENERATE`). earth gives gcv 0 and rsq and grsq NaN, and its termination code depends on n. One case gives the same result in pymars and an error in earth. (E4; GCV-7, EDGE-1)
- **rule** pymars reports the computed sums of squares, where earth reports `rss`, `gcv` and their per-subset values as 0 below about 1e-10 in units of y squared, so that its rsq and grsq can be 1 or NaN. pymars uses no absolute epsilon, so a response in tiny units still gives useful statistics. (E2, E3; PRUNE-4)
- **rule** A penalty above 1000 is allowed; earth gives an error. A minspan larger than n is allowed. (CORE-2, SPAN-3)

## Parameters and inputs

- **rule** `None` means automatic for `minspan` and `endspan` (earth uses 0), and `nprune=None` is earth's NULL. Negative `minspan`, which asks earth for evenly spaced knots, is not offered. `pmethod` is `"backward"` or `"none"`; earth's `"exhaustive"`, `"forward"`, `"seqrep"` and `"cv"` are not offered. `linpreds`, `allowed`, `newvar.penalty`, `nfold` and `Scale.y` are not offered. Issues #28 to #33 track these as `later`; `Scale.y` has none, since pymars does not scale responses separately (FWD-10).
- **rule** Strings and categories are an error that names `OneHotEncoder`, where earth expands factors itself. `OneHotEncoder(drop="first")` gives earth's coding. (ERR-2)
- **rule** Missing values raise an error (`allow_missing=True` raises NotImplementedError); earth has `na.action = na.fail` only. Issue #27 tracks support. (ERR-3)
- **rule** Covariates are numbered from 0 in the term labels of `summary()`, where earth numbers from 1, and the terms appear in increasing covariate order. A linear term stores 0.0 in `cuts_`, where earth stores the smallest value of the covariate. These affect labels and storage only; the basis is the same. (TERM-2, TERM-5)

## Classification

- **rule** With three or more classes, earth fits one binomial GLM for each class, and its probabilities need not sum to 1. pymars fits one multinomial model, so each row of `predict_proba` sums to 1. With two classes both fit the same binomial GLM (the refit agrees with R's `glm` to a relative 1e-7 or better). (GLM-2)
- **rule** The logistic refit has no penalty in earth. `glm_alpha=0`, the default, gives the same unpenalized fit; a positive value is an addition. (API-7)
- **rule** Probabilities near 0 or 1 are the computed ones, where R reports 2.2e-16 for every linear predictor above 30 in absolute value. The ConvergenceWarning keeps earth's threshold. (GLM-4)

## Limits of float64 arithmetic

pymars computes in float64, as earth does. On ordinary data rounding changes the model only at a near-tie, where two choices are equal up to rounding. On the four kinds of input below, rounding can decide which term enters, so pymars and earth, and pymars on two platforms or with and without numba, can give different models. The fit still ends with a valid model and raises no new error. The ids point to the "Limits" section of [algorithm.md](algorithm.md).

- **LIM-1** A covariate whose mean is far larger than its spread, such as a year or a timestamp. A linear term in such a covariate looks like a copy of the intercept once the mean is about 1e7 times the spread, and a product of two or three such terms at about 3,000 and 200 times. pymars then leaves the term out, as earth's final fit does. Subtract the mean (or a typical value) before the fit if such terms matter; at degree 2 or more this changes which models are possible, so pymars does not do it for you.
- **LIM-2** Two covariates that are equal, or a linear function of each other, except for the last few digits at some rows. Which of them enters then follows rounding. Exact copies are not a problem: a fixed order decides between them.
- **LIM-3** Covariate values that differ only in the last few digits. A knot at one of them can give a term with a very large coefficient, so predictions away from the data can be far off even though the fitted values are accurate.
- **LIM-4** Weights so small next to the others, such as 1e-300 next to 1, that their products with squared values underflow to 0. Those rows then have no effect on the fit.

The validation of pymars (the oracle tests, the comparison with earth and [validation/REPORT.md](../validation/REPORT.md)) uses ordinary data, as the "Limits" section defines it.

## Differences that are not changes of rule

Two things differ in how the comparison was made, and not in the fit. The harness gives earth the covariates divided by their standard deviation (LA-7), so that both programs see the same matrix. And fits with zero weights were compared with earth to a relative 1e-7 in their sums of squares, because earth's tiny weights shift them by up to about 6e-9 (W-3).
