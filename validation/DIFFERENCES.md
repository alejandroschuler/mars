# Differences between the new code and earth

This file triages every difference between the new fitting code (pymars 2.0) and earth 5.3.4 on the fixtures in `validation/fixtures/` (VALIDATION_PLAN.md, "Triage of differences"). The machine-readable list is `validation/differences.json`; `tests/test_conformance.py` compares each implementation with earth and fails on a difference that the list does not hold, or on an entry that no longer occurs. Each row below is one entry of that list, by its id.

Status: the reference implementation (`tests/reference/mars_ref.py`) at commit 2a00063 of `main`, with R 4.4.3 and earth 5.3.4 on macOS (arm64). The fast code joins the comparison when `pymars._core.fit_mars` lands (T12); the entries hold for every implementation unless they name one.

## What is compared

The test module's docstring gives the rules. In short:

- Every dataset fixture (133 files: S01 to S20 in their modes) and the 200 draws of S15. The fixtures give earth the covariates divided by their standard deviation (LA-7), except S12's raw mode.
- The forward steps: the terms that each step adds, exactly, up to the first step where the two choices differ; the RSS after each matching step within 1e-8 of the RSS before it (LA-5); the termination code.
- The pruning record on the same forward basis: the fit's own when the forward paths match, else the implementation's pruning pass on earth's forward basis. The subsets of each size exactly, and the RSS and GCV of each size within a relative 1e-8.
- The selected terms, the coefficients, GCV, RSq, GRSq, the fitted values and the predictions at new points, with the plan's tolerances and κ(B) limits (`compare.compare_fit`).

Where the plan compares a fixture otherwise:

- Integer weights (S13 and S16): pymars with the weights against earth without weights on the repeated rows, the `_repeated` fixtures (W-1). The 12 such fits whose response is not constant agree with earth. earth's own weighted fits of these data are not compared, because earth counts cases and ignores the weights in its spans and knots (W-2, W-3).
- Non-integer weights (S13_nonint, weights that sum to n): only the pruning pass on earth's weighted forward basis, against weighted earth (OQ-3). Both modes agree.
- The binary responses (S14, S20): the least-squares passes. The GLM refit and its probabilities are T14's (`s18_multinom.json` too).

## Near-ties

A step where the two choices differ is a near-tie when their RSS values on the basis before the step differ by less than 1e-7 of the RSS before the step (plan: Ties). A size where the pruning subsets differ is a near-tie when their RSS values differ by at most 1e-7 of the lower one. This is the threshold that OQ-2 asks T07 to fix: each program's RSS of one subset is within 1e-8 of that RSS (the tolerance for `rss.per.subset`), so a flip needs a gap below about 2e-8, and 1e-7 covers it five times over, as in the forward pass. A near-tie is labeled `tie` and passes, and the comparison of that fit's structure stops there.

A near-tie in the implementation's own candidate log at a step where both programs chose the same term does not stop the comparison, since the paths after the same choice can still be compared. This matters: in S10_matched_d1 the duplicated column makes the best and the second-best candidate tie exactly at 6 of its 10 steps, and within 1e-7 at the other 4, and both programs break these ties the same way (FWD-5, bb18.1). In all, 8 of the 131 whole fits have such a step in the reference's log (S04 with 1,000 cases, S07 matched, S10), and all 8 agree with earth.

No difference on the fixtures is a near-tie. The near-ties of LA-5 and STOP-7 (a collinearity ratio or a pair-rule value at its threshold, a reduction at MaxLegal, a stopping rule at its bound) need values that the candidate log does not hold, so a difference of that kind fails the test and gets its label here by hand. None occurred.

## Results

Counts by label, over the comparisons of the reference:

| Label | Entries | Comparisons |
|---|---|---|
| rule | 4 | 9 |
| quirk | 3 | 3 |
| tie | 0 | 0 |
| numeric | 0 | 0 |
| bug | 0 | 0 |

By dataset: S11 1 comparison (E3), S12 4 (E1, E2), S13 4 (E4), S15 3 (E5 to E7). Every other comparison agrees with earth in full: 124 of the 133 dataset fixtures, and 197 of the 200 S15 draws.

| Id | Cases | Step | pymars | earth | Candidate RSS (pymars, earth) | earth's flags | Label | Rules |
|---|---|---|---|---|---|---|---|---|
| E1 | S12_x_1em8_raw_d1 | forward 1 | the pair h(x0-4.08098e-09), h(4.08098e-09-x0) | the single hinge h(x0-4.08098e-09) | 0.36376841094440215, 0.3678187937537397 (before: 25.870380638504514) | no trace | rule | LA-7 |
| E2 | S12_y_1em9 in the defaults, matched and raw modes | the statistics | the computed values | rss, gcv and every rss.per.subset and gcv.per.subset 0; rsq, grsq NaN | | | rule | PRUNE-4, PRUNE-8 |
| E3 | S11_n03_matched_d1 | rss.per.subset of size 3 | 1.8e-63 | 0 | | | rule | PRUNE-4, PRUNE-8 |
| E4 | S13_constant_y_weighted and its repeated rows, defaults and matched modes | the whole fit | degenerate: gcv +inf, rsq and grsq 0, code 0 | gcv 0, rsq and grsq NaN, code 4 (defaults) or 7 (matched) | | | rule | EDGE-1, GCV-7, CORE-4 |
| E5 | S15 draw 21 | forward 7 | h(2.8584-x0)*h(x1-1.1211) and its mirror | h(3.24208-x7)*h(x9-0.664554) and its mirror | 176.64601298766252, 176.63399682138416 (before: 184.2567300017018) | rank fix printed | quirk | FWD-11, OQ-6, FAST-4 |
| E6 | S15 draw 33 | forward 7 | h(3.3732-x0)*h(x9-2.62033) and its mirror | h(2.03371-x2)*h(x9-0.951963) and its mirror | 151.0823446122114, 148.705730583402 (before: 154.94244645559493) | rank fix printed | quirk | FWD-11, OQ-6, FAST-4 |
| E7 | S15 draw 61 | forward 10 | the single hinge h(x7-0.609737)*h(x9-0.738223) | h(x8-0.813011)*h(x9-0.537277) and its mirror | 120.5298001805806, 119.36386027987149 (before: 123.536576669401) | rank fix printed | quirk | FWD-11, OQ-6, FAST-4 |

The dataset fixtures hold no trace, so earth's flags are not known there; E1 needs none, since the rule decides it. The S15 draws with the rank fix hold no parsed trace steps either (`steps_error`: the hidden term moves the slots that `compare.steps_from_trace` maps), only the flag that the trace printed the rank fix.

### E1: the kind of a search in raw units (rule, LA-7)

The raw mode of S12 keeps x in its own units, here 1e-8 times the base x. earth runs a single-hinge search because A, the RSS of x on the intercept in these units, is 1.4e-15, below 0.01. pymars compares A with 0.01 times the variance of x (A is 150 variances) and runs a pair search at the same knot. earth's single hinge is not a candidate of pymars at this step. The pruning pass on earth's forward basis agrees with earth. In the other raw fits (x times 1e8, x plus 1e6, y times 1e-9 and 1e9) the forward passes agree, since the scale of y does not enter the rule and A is far from 0.01 there.

### E2 and E3: earth reports small sums of squares as 0 (rule, PRUNE-4 and PRUNE-8)

In S12_y_1em9, y is 1e-9 times the base y, so every sum of squares is below 3e-17. earth reports rss, gcv, rss.per.subset and gcv.per.subset as 0, and rsq and grsq as NaN (bb26.9). pymars reports the computed values, since it uses no absolute epsilon. In S11_n03_matched_d1 the model of size 3 interpolates the three cases: its RSS is 1.8e-63 in pymars and 0 in earth. The comparison leaves out the values that earth reports as 0 (below 1e-9 in pymars) and takes earth's rss, gcv, rsq and grsq from its residuals. Everything else agrees: the forward path, the subsets, the selected terms, the coefficients and the fitted values.

### E4: a constant response (rule, EDGE-1, GCV-7, CORE-4)

y is 3 at every case, so the pymars fit is degenerate: the intercept alone, gcv +inf, rsq and grsq 0, and termination code 0. earth reports gcv 0 and rsq and grsq NaN. Its total sum of squares is 8.6e-29 from rounding, not 0, so its forward pass runs: it stops at once at its defaults (code 4), and it takes 10 steps to the term limit in the matched mode (code 7). Both select the intercept alone, with the coefficient 3 (earth: 2.9999999999999982). earth stops with an error on the weighted data (bb10.9), so the weighted fits are compared with earth on the repeated rows.

### E5 to E7: earth's hidden term (quirk, FWD-11 and OQ-6)

All three are S15 draws at degree 2 in the matched mode (`Auto.linpreds = FALSE`), and earth's trace prints the rank fix. FWD-11 says that T07 labels such a difference `quirk`. The mechanism is the same in all three. A linear-option step (the hinge at the smallest x, FWD-6) made earth add its hidden term, which fills the empty second slot of that step (FWD-9). A later step then adds a pair whose second term sits in a slot above the number of pymars's terms, so pymars does not search it as a parent yet (FAST-4), while earth, with its hidden term, has one term more and does. earth's choice has the lower RSS on the common basis in all three.

- E5 (the DGP D1 at the high noise level): the linear option of step 5; earth's parent at step 7 is h(3.24208-x7), the second term of step 6, in slot 13, while pymars has 12 terms.
- E6 (the DGP D6): the linear option of step 4; earth's parent at step 7 is h(2.03371-x2), the second term of step 6, in slot 13, while pymars has 12 terms.
- E7 (the DGP D6): the linear options of steps 2 and 6; earth's parent at step 10 is h(x9-0.537277), the second term of step 9, in slot 18, while pymars has 17 terms.

## S15

The 200 draws come from the simulation DGPs D1 to D6 and D8 (n = 200, p = 10), at degree 1 and 2, half in the matched mode and half at the defaults, all with `pmethod = "none"`. The fixture keeps earth's forward terms and its pruning records, not the fitted values, so the comparison covers the forward path and `rss.per.subset` and `gcv.per.subset`. Every fit takes 10 steps and stops at the term limit.

- The share of fits that agree: 197 of 200 (98.5 %). The subsets' RSS and GCV values agree to a relative 7.5e-15 at most.
- The first divergence: at step 7 (E5, E6) and step 10 (E7), all three by earth's hidden term (`quirk`, FWD-11). 11 draws print the rank fix, all at degree 2 in the matched mode; the other 8 agree with earth.
- The share of fits that stop at a near-tie: 0 of 200. Counting also a near-tie where both programs chose the same term, 1 of 200 (0.5 %): draw 94, step 9, where the best and the second-best RSS differ by 9.3e-8 of the RSS before the step. Both are below the 5 % at which the plan asks for a review, and the test checks the larger count.

## Spec questions

Raised on #44 (spec v2), one comment each: the pruning near-tie threshold that closes OQ-2; the reading that only a near-tie with different choices stops the comparison; a record of the LA-5 and STOP-7 near-ties in the candidate log, so that the tests can label them; and the S15 evidence on the hidden term (OQ-6) together with the slot rule (FAST-4, OQ-4).
