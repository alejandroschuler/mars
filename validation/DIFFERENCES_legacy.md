# The legacy code against earth, S01 to S20

This file describes the legacy code against earth 5.3.4 on the datasets S01 to S20 of `VALIDATION_PLAN.md` ("Test datasets"). The legacy code is mars-earth 1.0.4, the PyPI wheel, and HEAD `d68b54a` (the tag `legacy-1.0.4-head`). The plan describes the legacy code and does not fix it; pymars 2.0 replaces it. This is the P1 exit item: every difference has a label from "Triage of differences" (`rule`, `bug`, `quirk`, `tie` or `numeric`), with a finding ID from "Preliminary findings" (F1 to F16) where one applies, and a new ID (F17 onward) where none does.

`validation/legacy/conformance_legacy.py` makes every number here; the last section gives the commands. `validation/legacy/conformance_legacy.json` holds the per-fit results, with one sentence of evidence for each difference.

## What was run

- **Fits.** Every dataset fixture in `validation/fixtures/` (132 fits of S01 to S20, in the modes that each dataset's purpose needs), S19's factor through the legacy `categorical_features` path, and the 200 S15 draws. earth's side is the fixture's result, or a new black-box run where the fixtures lack the settings: the matched mode at degrees 2 and 3, and the S15 draws in the legacy matched mode.
- **Matched mode** ("Comparison modes"). Legacy: `allow_linear=False`, `minspan=1`, `endspan=0`, `penalty` half of earth's, `max_terms` equal to `nk`. earth: `Auto.linpreds = FALSE`, `fast.k = 0`, `thresh = 0`, `minspan = 1`, `endspan = 1`, `Adjust.endspan = 0`, `nk = 21`, `penalty = 2`.
- **Defaults mode.** Each program at its own defaults, at the same degree.
- **Span grid** (S01, S02). earth at its defaults with `minspan` 1, 5 or automatic and `endspan` 1, 10 or automatic; the legacy code at its defaults with the same spans. earth's automatic span maps to the legacy formula with α = 0.05 (`minspan_alpha`, `endspan_alpha`), and earth endspan e to legacy endspan e − 1, as in the matched mode. This map gives the same knots only at e = 1. At e = 10 the legacy knots start one value lower, at sorted position E where earth's lowest knot is at E + 1, and earth caps its endspan at ⌊n/2⌋ − 1 (SPAN-5), which applies in S02_n020.
- **Raw** (S12). earth at its defaults on the unscaled inputs, against the legacy defaults on the same inputs. Every other mode uses the fixtures' scaled matrix (each covariate divided by its standard deviation, LA-7).
- **Weights** (S13, S16). The 1.0.4 wheel's `Earth.fit` takes no `sample_weight`; HEAD's does. So the weighted fits run on HEAD. For integer weights earth's side is its unweighted fit on the repeated rows (frequency weights, as in "Sample weights"); for `S13_nonint` it is earth's weighted fit.
- **Binary and class responses.** S14 and S20 use `EarthClassifier`, whose inner `Earth` fits the 0/1 response by least squares, as earth does before its `glm` step. S18 uses `EarthClassifier` on the three class labels.

### A correction to the plan's matched mode

The plan sets `Adjust.endspan = 1` for "no larger endspan for interaction terms". earth's endspan for a parent other than the intercept is E + ⌊a·E + 0.5⌋ (docs/algorithm.md, SPAN-4), so a = 1 doubles it. A black-box `trace = 9` run on S05 (endspan 1) confirms it: the searches with a hinge parent print `nEndSpan 1`, `2` and `3` at `Adjust.endspan` 0, 1 and 2. This task uses 0. The fixtures' `matched_d2` and `matched_d3` use earth's default 2 and S05's `matched_d2_adjust1` uses 1, so every matched fit at degree 2 or 3 here is a new earth run.

## How a fit is compared

A term is compared as its set of factors (variable, direction and knot, or a linear factor), so no row or slot numbers are needed. Each forward step of the two programs adds a set of terms, and the steps are compared in order:

- **same**: both add the same terms;
- **structure**: earth adds one term of the legacy pair, a single hinge or the linear option at the same knot, where the legacy code adds both hinges (F5);
- **choice**: the programs add different terms. The comparison of the forward pass stops at the first choice divergence.

Along the steps that agree, the RSS after each step is compared at relative 1e-8, scaled by κ(B) above 1e6 ("What is compared, and the tolerances"). When the two forward passes add the same terms throughout, the pruning pass is compared size by size, then the GCVs, the selected terms and, if those agree, the coefficients, the GCV and the predictions, with `compare.py`. The pruning differences are listed in the order of their effect on the selected model: the intercept missing from the legacy model (F3); else, when the selected terms differ, other GCVs at a size where both paths keep the same subset (F4), other subsets at a selected size, or a tie between sizes; then the path's other differences, such as the intercept dropped only below the selected size (F3). When the forward passes differ, the later differences follow from the first one, and only summary numbers are kept: the number of terms, R², GCV and the largest difference in predictions.

At the first choice divergence, the script gathers the evidence from both sides:

- the legacy side: its own record of the search at that step, with the chosen candidate, the second best by its criterion, and its own RSS and GCV for earth's choice, or why earth's choice was not a legacy candidate (for example a knot outside the legacy knot set);
- the earth side: a black-box run with `trace = 9` and `nk` = 2s + 1, so that step s is the last one it searches (LIMIT-2). It says whether earth searched the legacy choice's parent and variable, and whether that search was for a pair or a single hinge (LA-7). It also says whether earth visited and evaluated the legacy knot, and gives earth's four flags there (`bx1G`, `CovColG`, `TolG`, `MaxG`). The script checks that the probe run repeats the full run's first terms (the same `dirs` and `cuts` rows), and all 312 probes do;
- the RSS of both choices, from a least-squares fit on earth's basis before the step.

The near-tie rule of "Ties" comes first: a divergence is a `tie` when the two choices' RSS are within 1e-7 of the RSS before the step, in the legacy code's full-precision log or on earth's basis. earth's trace prints only 5 or 6 significant digits, so it cannot decide a near-tie. Otherwise the side that passed over the term with the lower RSS, on earth's basis, gives the cause. When earth's choice has the lower RSS, the cause is on the legacy side: earth's knot is not a legacy candidate, or the legacy GCV ranked it lower (F2). When the legacy choice has the lower RSS, the cause is in earth's trace: the collinearity tolerance, a single-hinge search, or a knot outside earth's grid. The table lists the causes:

| Cause at the first divergence | Label | ID |
|---|---|---|
| earth adds one term where the legacy code adds a pair | rule | F5 |
| earth rejects the legacy knot by its collinearity tolerance (`TolG 0`, LA-3) | rule | F5 |
| earth searches that parent and variable for a single hinge (LA-7), without the linear part of the legacy pair | rule | F5 |
| the legacy parent is the redundant second hinge of an earlier pair, which earth never added | rule | F5 |
| the legacy knot is outside earth's knot grid: the spans, or the parent's largest active value (KNOT-3, KNOT-4) | rule | F7 |
| earth's knot scan for a hinge parent: a knot only just below a case where the parent is positive, so possibly at the value of a case where it is zero (KNOT-3, KNOT-4) | quirk | F7 |
| earth's choice is not a legacy candidate (a knot outside the legacy knot set) | rule | F7 |
| the legacy code ranks by GCV, and earth's choice had the lower RSS | rule | F2 |
| every legacy candidate has C ≥ n, or n or more columns, so the legacy code stops | rule | F2 |
| the best legacy candidate by GCV is already in the model's span (its computed gain is rounding), and the legacy code stops on it | rule | F2 |
| the best legacy candidate lowers the RSS by a positive amount below machine epsilon | rule | F6 |
| earth stops by its relative rules (`thresh`), which the legacy code lacks | rule | F6 |

The pruning pass (F3, F4), the GLM refit (F9) and the new findings (F17 to F19) get their labels in the sections below. The script also has verdicts for earth rules that no fit here showed (the MaxLegal limit of FWD-4, the slot rule of FAST-4, a better pruning subset by PRUNE-3); they would need new IDs.
## Results

There are 334 cases: 132 fixture fits, S19's factor in 2 codings, and 200 S15 draws. The 16 weighted cases run on HEAD as well, which gives 350 rows. Of these, 18 rows are fits that the legacy code cannot make (the wheel has no weights, and no version takes several responses), and 3 have no difference at all: S19's factor as numeric codes, and a constant response at the defaults (twice). The other 347 have at least one difference. No difference is unexplained, and none is `numeric`.

<!-- generated: conformance_legacy.py report --markdown, the count tables -->
| Label | First difference of a fit | All |
|---|---|---|
| bug | 6 | 111 |
| quirk | 23 | 23 |
| rule | 318 | 357 |
| tie | 0 | 2 |

| Finding | First difference of a fit | All |
|---|---|---|
| F1 | 16 | 16 |
| F2 | 104 | 108 |
| F3 | 3 | 104 |
| F4 | 2 | 5 |
| F5 | 126 | 158 |
| F6 | 7 | 7 |
| F7 | 82 | 82 |
| F9 | 2 | 6 |
| F17 | 1 | 1 |
| F18 | 2 | 2 |
| F19 | 2 | 2 |
| (tie) | 0 | 2 |
<!-- end generated -->

Most first differences are `rule`: the legacy code lacks a rule of earth's (F2, F5, F6 and F7). The `bug` entries are the intercept that the pruning pass removes (F3), the GLM refit (F9) and the categorical path (F17); as first differences, F3 comes first in 3 fits and F9 in 2. The `quirk` entries are earth's knot scan for a hinge parent (F7) and its forward pass on a constant response (F19).

S15 measures the rates. In the matched mode, 2 of the 100 draws make the same choices at every step (apart from single hinges against pairs, F5). The others diverge at a median of step 3, for these causes: earth's collinearity tolerance (F5) in 81, earth's knot scan for a hinge parent (a quirk, F7) in 16, and a near-tie in 1. At the defaults, every draw diverges at step 1 or 2: the legacy code picks a linear term by GCV (F2) in 82, and a knot outside earth's grid (F7) in 18.

<!-- generated: the S15 table -->
| S15 mode | Fits | No choice divergence | First divergence: step | Cause |
|---|---|---|---|---|
| matched | 100 | 2 | 1: 27, 2: 19, 3: 17, 4: 13, 5: 10, 6: 4, 7: 5, 8: 1, 9: 1, 10: 1 | quirk F7: 16, rule F5: 81, tie: 1 |
| defaults | 100 | 0 | 1: 94, 2: 6 | rule F2: 82, rule F7: 18 |
<!-- end generated -->

Only 1 of the 200 S15 fits (0.5 percent) stops at a near-tie, below the plan's review threshold of 5 percent ("Ties").

## The findings

### Confirmed in part

**F1, weights.** The 1.0.4 wheel's `Earth.fit` has no `sample_weight`, so its 16 weighted fits fail (`rule`). HEAD takes weights as frequency weights in the forward pass: its 14 fits with integer weights add the same forward terms as the legacy code on the repeated rows in 13 cases. But 8 of the 14 select other terms, and in 6 of them the predictions differ, by up to 14.3 (S13_int_random, matched).

The fits diverge by rounding, in two ways. In S13_unit, with every weight 1, the pruning paths agree down to 13 of 21 terms. At 12, several removals of redundant hinge columns (F5) give the same RSS, 0.839653995611, and rounding picks a different one in the weighted code. In S16_weighted at the defaults, the best candidate by GCV at step 8 is the linear term x0, which is already in the model's span. Its computed gain is exactly 0 with weights and 3.4e-13 without, so the weighted fit stops after 7 steps and the other goes on to 9 (F2 with F6: the stop test looks at the best candidate by GCV). Against earth on the repeated rows, HEAD's weighted fits get the same labels as the unweighted legacy fits. So the runs reach part of F1: with integer weights the forward pass mostly matches repeated rows, and unit weights change the pruning by rounding. They do not reach weights that sum to 1, nor the span formulas, because the spans here are 0 or 1.

<!-- generated: the weights table -->
| HEAD with weights | Steps (weights, repeated) | Same forward terms | Same selected terms | Largest test difference |
|---|---|---|---|---|
| S13_constant_y_weighted/defaults_d1 | 0, 0 | True | True | 1.33e-15 |
| S13_constant_y_weighted/matched_d1 | 0, 0 | True | True | 1.33e-15 |
| S13_equal2/defaults_d1 | 1, 1 | True | True | 0 |
| S13_equal2/matched_d1 | 10, 10 | True | False | 1.68e-13 |
| S13_int_random/defaults_d1 | 1, 1 | True | True | 1.33e-15 |
| S13_int_random/matched_d1 | 10, 10 | True | False | 14.3 |
| S13_int_zeros/defaults_d1 | 1, 1 | True | True | 0 |
| S13_int_zeros/matched_d1 | 10, 10 | True | False | 0.416 |
| S13_unit/defaults_d1 | 1, 1 | True | True | 0 |
| S13_unit/matched_d1 | 10, 10 | True | False | 0.265 |
| S16_weighted/defaults_d1 | 7, 9 | False | False | 5.35 |
| S16_weighted/defaults_d2 | 12, 12 | True | False | 10.9 |
| S16_weighted/matched_d1 | 10, 10 | True | False | 4.55 |
| S16_weighted/matched_d2 | 10, 10 | True | False | 3.27e-11 |
<!-- end generated -->

### Confirmed

**F2, the GCV criterion.** The legacy code picks the candidate with the lowest GCV, so a linear term, which costs one column and no penalty, often beats a pair with the lower RSS. This is the first difference in 101 fits, 82 of them S15 draws at the defaults. In S04 (5 covariates, 200 cases) at the defaults, step 1: the legacy code adds x3, with GCV 16.02 and RSS 3140.3, and earth adds the pair h(x3-1.89005) and h(1.89005-x3), whose RSS is 3127.7 but whose legacy GCV is 17.15. The same rule stops the legacy code at small n: in the 4 fits of S11 in the matched mode, every legacy candidate has C ≥ n, an infinite GCV, or at least n columns, and the legacy forward pass stops while earth's goes on. The GCV ranking also stops the legacy code in the three S02 span-grid fits with minspan 1 (50 cases): the best candidate by GCV is the linear term x0, already in the model's span, whose computed gain is ±6e-17. The stop test (F6) looks only at that candidate, so the forward pass ends after 2 steps although a pair would lower the RSS by about 5 percent. These are labeled F2, since no threshold would let a zero gain through.

**F3, the pruned intercept.** The legacy pruning pass protects the intercept only as the last term. 102 of the 332 legacy fits end without an intercept: 63 of 147 at the defaults, 27 of 150 in the matched mode, 11 of 27 in the span grid and 1 of 6 raw. Among the S15 draws it is 43 of 100 at the defaults and 16 of 100 matched. In 2 more fits (S13_int_random, on HEAD and on repeated rows) the intercept leaves the legacy path only at size 1, below the selected size, and the selected model keeps it; there F4 comes first. earth keeps the intercept in every fit.

**F4, the GCV convention.** In 5 fits at the defaults both programs add the same forward terms, but at the sizes where both pruning paths keep the same subset, their GCVs differ, by up to 4.5 percent. Two differences add up. The legacy C = M + d·H charges d for each hinge term and nothing for a linear term, where earth charges d/2 for each term after the intercept. And the legacy default d is 3 at every degree, where earth's is 2 at degree 1. In S13_int_random and S13_int_zeros (on HEAD with weights and on repeated rows) the terms are one hinge pair and no linear term: at size 3 the legacy C is 3 + 3·2 = 9 and earth's 3 + 2·2/2 = 5. The selected sizes differ: 2 against 3 in S13_int_random, where F4 comes first, and 1 against 3 in S13_int_zeros, whose legacy model has no intercept (F3 first). In S19_dummies the terms are the three linear dummies: at size 4 the legacy C is 4 and earth's 7, and the selected sizes are 3 against 4 (F3 first). In the matched mode the halved penalty makes the two conventions equal for hinge-only models with an intercept, and no matched fit disagrees with that.

**F5, pairs, single hinges and the collinearity tolerance.** Where earth adds one term, a single hinge in a single-hinge search (LA-7) or the linear option, the legacy code adds a pair whose second hinge is redundant. This happens before any other difference in 37 of the 150 matched fits, most often at step 2. The redundant hinge leaves the span unchanged: along the steps where both programs agree, the RSS agrees to 8e-15 relative. The one exception is S10, with a covariate equal to x0 plus noise of size 1e-9. There the legacy pair adds a column that earth's single-hinge search leaves out, and the RSS after step 2 is 0.53925 against 0.54109.

At a choice divergence, earth's collinearity tolerance (LA-3, `TolG 0`) rejected the legacy knot in all 113 of the F5 choices, 81 of them S15 draws. In S04 (5 covariates, 200 cases) in the matched mode, the legacy code's first knot is x3's 9th-lowest value (trace case 9). With x3, it gives RSS 3119.1, and earth, which rejects it, adds the pair at x3 = 1.89005 with RSS 3127.7. A knot this close to the end of the data can leave a term fitted to a single case. In S15 draw 32 (D5, degree 1), the legacy code selects h(0.0425783-x9), which is positive at 1 of the 200 training cases, with coefficient 114,773; its test error is 126,243 against earth's 0.037.

**F6, the absolute stop.** The legacy forward pass stops when its best candidate by GCV lowers the RSS by at most machine epsilon. With y times 1e-9 (S12_y_1em9) the RSS starts at 2.6e-17, so the legacy code returns the intercept alone in all three modes, where earth fits the base knot. In 4 fits at the defaults (S08, S09 and two of S11), earth stops first, by its relative rules (`thresh`, or GRSq), and the legacy code goes on. On S12 the legacy code is not invariant to the scale of y or of x, while earth is on these data:

<!-- generated: the S12 table -->
| S12 mode | Variant | Legacy: terms, knots | earth: terms, knots |
|---|---|---|---|
| raw_d1 | S12_base | 2, same | 2, same |
| raw_d1 | S12_x_1e8 | 2, same | 2, same |
| raw_d1 | S12_x_1em8 | 2, same | 2, same |
| raw_d1 | S12_x_plus_1e6 | 2, differs | 2, same |
| raw_d1 | S12_y_1e9 | 2, same | 2, same |
| raw_d1 | S12_y_1em9 | 1, differs | 2, same |
| defaults_d1 | S12_base | 2, same | 2, same |
| defaults_d1 | S12_x_1e8 | 2, same | 2, same |
| defaults_d1 | S12_x_1em8 | 2, differs | 2, same |
| defaults_d1 | S12_x_plus_1e6 | 2, same | 2, same |
| defaults_d1 | S12_y_1e9 | 2, differs | 2, same |
| defaults_d1 | S12_y_1em9 | 1, differs | 2, same |
| matched_d1 | S12_base | 2, same | 2, same |
| matched_d1 | S12_x_1e8 | 3, differs | 2, same |
| matched_d1 | S12_x_1em8 | 2, same | 2, same |
| matched_d1 | S12_x_plus_1e6 | 2, same | 2, same |
| matched_d1 | S12_y_1e9 | 2, same | 2, same |
| matched_d1 | S12_y_1em9 | 1, differs | 2, same |
<!-- end generated -->

"Same" means the selected terms have the base variant's knots, in its units, to relative 1e-8. In the scaled modes the x variants give the same matrix up to rounding, so their differences, for example the knot 0.408098 in place of 0.404473, come from rounding at close candidates.

**F7, the spans and the knot sets.** The first difference is a knot that is in one program's candidate set and not the other's in 82 fits. In 61 of them it is a `rule`: earth's minspan and endspan grid (KNOT-3) leaves out the legacy knot, or the legacy code's own thinned grid (span grid, S08 with minspan 5) leaves out earth's knot. This covers the defaults, where the legacy spans are 0, and every span-grid setting of S01 and S02. In the other 21 it is a `quirk` of earth's scan for a hinge parent (KNOT-3, KNOT-4). There, a knot is the value just below a case where the parent is positive, so earth skips a legacy knot whose next case has the parent at zero, or takes the value of such a case. This is the first difference in 5 of the 9 matched fits at degree 2 or 3 (S04 three times, S05, S06), and in 16 of the 50 S15 draws at degree 2.

**F9, binary and class responses.** `EarthClassifier` refits the selected basis with scikit-learn's L2-penalized logistic regression (`C=1`). On its own basis, the penalty moves the fitted probabilities by up to 0.003 against an unpenalized refit in S14 at the defaults, by 0.60 in S14 in the matched mode, and by 0.56 and 0.65 in S20, whose classes are separable. With three classes (S18), the classifier's inner fit regresses the integer class codes, where earth fits one indicator response per class.

**F10, speed.** The slowest legacy fit took 141 s (S04 with 10 covariates and 1,000 cases, degree 2, at the defaults), and 132 s in the matched mode. This matches the plan's 130 s.

### Qualified

**F14, the core search.** Where both programs make the same choice, the RSS after each step agrees to 8e-15 relative, at 383 of the 384 steps compared; the other is step 2 of S10 (F5, above). But the choices diverge early. Of the 50 matched fixture fits, 6 make the same choices throughout: S04 with 5 covariates and 1,000 cases, S04 with 10 covariates and 200 cases, S09, S13_nonint, S18 (whose responses differ by design) and S19_dummies. Of the S15 draws, 2 of 100 do. On S04 at degree 1, which is F14's Friedman #1, one fit diverges at step 1 (the collinearity tolerance), two keep all 10 knots, and one diverges at step 6. So F14 holds for the steps before the first knot that earth rejects by a rule that the legacy code lacks.

**F15, the defaults.** The S15 draws were made to measure conformance rates: they pool the DGPs D1 to D6 and D8, both noise levels and degrees 1 and 2, at n = 200. On them, at the defaults, every draw diverges at step 1 or 2. The legacy code's test error, against the true function at 1,000 new points per draw, is lower than earth's in 63 of 100 draws (a binomial standard error of about 5 points), with a median ratio of 0.905. It is more than 10 times earth's in 1 draw, the largest ratio being 216. In the matched mode it is lower in 39 of 100, with a median ratio of 1.16, and more than 10 times earth's in 10 draws; the largest ratio there is 3.4e6, from a term fitted to one case (F5). So the fits differ more at the defaults, as F15 says. Prediction quality is for the simulation study to settle.

### Not tested here

F8 (missing values) needs inputs that earth does not accept. F11 (the scikit-learn checks), F12 (the dependency bounds) and F13 (the reference tests) are not about fits. The runs touch F11 in one place: `EarthClassifier` takes no `minspan` or `endspan`, so its matched mode relies on its defaults, which give the same knots as `minspan = 1` and `endspan = 0`. F16 (determinism) is covered by HEAD against the wheel below.

## New findings

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| F17 | `categorical_features` on a column of strings gives the intercept alone. The candidates are made from the raw values of `X_fit_original` (the strings), but each indicator compares them with the label-encoded values (floats), so every candidate column is zero. With numeric codes the same data give earth's factor fit. | S19's factor: 1 term, fitted values off earth's factor fit by up to 2.04; as codes 0 to 3: 4 terms, fitted values within 3e-14 of earth's. The snippet below, on the wheel and on HEAD. | Wrong result (`bug`) |
| F18 | Several responses are not supported: `Earth.fit` raises `ValueError: y should be a 1d array` for a 2-D y. earth fits a shared basis. | S17 in both modes; the same on HEAD. | Missing feature (`rule`) |
| F19 | On a constant response with `thresh = 0`, earth's forward pass adds terms on rounding noise (a forward RSS of about 1e-28), in S13 up to the term limit, and its pruning pass removes them all; on other data it can stop after one step (code 6). The legacy code stops at once. Both return the intercept. | S13_constant_y_weighted in the matched mode, on repeated rows and on HEAD with weights. The spec lists earth's other quirks on a constant response (GCV-7, EDGE-1). | earth `quirk` |

F17 on the wheel and on HEAD (a checkout of the tag first on `sys.path`):

```python
import numpy as np, pymars
rng = np.random.default_rng(0)
labels = rng.choice(["a", "b", "c", "d"], size=200)
y = np.array([{"a": 0.0, "b": 1.0, "c": -1.5, "d": 2.5}[v] for v in labels])
y = y + rng.normal(scale=0.1, size=200)
strings = labels.reshape(-1, 1).astype(object)
codes = np.array([["abcd".index(v)] for v in labels], dtype=float)
print(pymars.Earth(categorical_features=[0]).fit(strings, y).basis_)  # [Intercept]
print(pymars.Earth(categorical_features=[0]).fit(codes, y).basis_)    # 4 terms
```

## HEAD and the wheel

HEAD's Python code runs in the wheel's venv, with a worktree of the tag first on `sys.path`; it needs no Rust part for fitting. On the 243 unweighted cases run with both (S01, S04, the repeated-row cases of S13 and S16, S15, and S19's factor), HEAD and the wheel give identical fits: the same terms, steps, pruning path, coefficients and fitted values, bit for bit. They differ in one respect: HEAD takes `sample_weight`, and the wheel does not (F1). F17 and F18 hold on both.

## Every fit

The table has one row per fit of S01 to S20. "Code" is the version that ran it: the 1.0.4 wheel, or HEAD for the weighted cases. The first difference and the next one are listed; `conformance_legacy.json` has every difference of a fit, with its sentence of evidence, and the S15 draws one by one. "Terms" is the number of selected terms, legacy then earth. A structure difference is one term against a pair at the same knot; a choice difference is two different terms; a stop difference is a forward pass that ends first; a design difference is a response that the two programs model differently.

<!-- generated: conformance_legacy.py report --markdown, the first table -->
| Fit | Code | First difference | Label | ID | Next difference | Label | ID | Terms |
|---|---|---|---|---|---|---|---|---|
| S01/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 4, 3 |
| S01/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 8, 6 |
| S01/span_m1_e1 | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 4, 4 |
| S01/span_m1_e10 | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 4, 3 |
| S01/span_m1_eauto | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 4, 4 |
| S01/span_m5_e1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 4 |
| S01/span_m5_e10 | wheel | choice at step 1 | rule | F7 | none | |  | 3, 3 |
| S01/span_m5_eauto | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 3, 4 |
| S01/span_mauto_e1 | wheel | choice at step 2 | rule | F7 | pruning | bug | F3 | 2, 4 |
| S01/span_mauto_e10 | wheel | choice at step 1 | rule | F7 | none | |  | 3, 3 |
| S01/span_mauto_eauto | wheel | choice at step 2 | rule | F7 | none | |  | 3, 3 |
| S02_n020/defaults_d1 | wheel | choice at step 1 | rule | F2 | none | |  | 3, 2 |
| S02_n020/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 3, 3 |
| S02_n020/span_m1_e1 | wheel | choice at step 1 | rule | F2 | none | |  | 3, 3 |
| S02_n020/span_m1_e10 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S02_n020/span_m1_eauto | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S02_n020/span_m5_e1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 3 |
| S02_n020/span_m5_e10 | wheel | choice at step 1 | rule | F2 | none | |  | 2, 2 |
| S02_n020/span_m5_eauto | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S02_n020/span_mauto_e1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 3 |
| S02_n020/span_mauto_e10 | wheel | choice at step 1 | rule | F2 | none | |  | 2, 2 |
| S02_n020/span_mauto_eauto | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S02_n050/defaults_d1 | wheel | choice at step 2 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 2, 3 |
| S02_n050/span_m1_e1 | wheel | structure at step 2 | rule | F5 | stop at step 3 | rule | F2 | 2, 3 |
| S02_n050/span_m1_e10 | wheel | structure at step 2 | rule | F5 | stop at step 3 | rule | F2 | 2, 3 |
| S02_n050/span_m1_eauto | wheel | structure at step 2 | rule | F5 | stop at step 3 | rule | F2 | 2, 3 |
| S02_n050/span_m5_e1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/span_m5_e10 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/span_m5_eauto | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/span_mauto_e1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/span_mauto_e10 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S02_n050/span_mauto_eauto | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 3 |
| S03/defaults_d1 | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 6, 7 |
| S03/matched_d1 | wheel | structure at step 4 | rule | F5 | choice at step 5 | rule | F5 | 9, 7 |
| S03/matched_d1_linear | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 6, 7 |
| S04_p05_n0200/defaults_d1 | wheel | choice at step 1 | rule | F2 | none | |  | 8, 10 |
| S04_p05_n0200/defaults_d2 | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 16, 18 |
| S04_p05_n0200/matched_d1 | wheel | choice at step 1 | rule | F5 | none | |  | 14, 12 |
| S04_p05_n0200/matched_d2 | wheel | choice at step 1 | rule | F5 | none | |  | 17, 18 |
| S04_p05_n1000/defaults_d1 | wheel | choice at step 1 | rule | F2 | none | |  | 9, 13 |
| S04_p05_n1000/defaults_d2 | wheel | choice at step 1 | rule | F2 | none | |  | 17, 18 |
| S04_p05_n1000/matched_d1 | wheel | structure at step 6 | rule | F5 | none | |  | 15, 15 |
| S04_p05_n1000/matched_d2 | wheel | choice at step 6 | quirk | F7 | none | |  | 17, 18 |
| S04_p10_n0200/defaults_d1 | wheel | choice at step 1 | rule | F2 | none | |  | 9, 10 |
| S04_p10_n0200/defaults_d2 | wheel | choice at step 1 | rule | F2 | none | |  | 17, 15 |
| S04_p10_n0200/matched_d1 | wheel | structure at step 6 | rule | F5 | none | |  | 11, 10 |
| S04_p10_n0200/matched_d2 | wheel | choice at step 6 | quirk | F7 | none | |  | 16, 16 |
| S04_p10_n1000/defaults_d1 | wheel | choice at step 1 | rule | F2 | none | |  | 13, 15 |
| S04_p10_n1000/defaults_d2 | wheel | choice at step 1 | rule | F2 | none | |  | 14, 18 |
| S04_p10_n1000/matched_d1 | wheel | choice at step 6 | rule | F5 | none | |  | 14, 14 |
| S04_p10_n1000/matched_d2 | wheel | choice at step 6 | quirk | F7 | none | |  | 17, 17 |
| S05/defaults_d2 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 1, 2 |
| S05/matched_d2 | wheel | choice at step 2 | quirk | F7 | pruning | bug | F3 | 2, 3 |
| S06/defaults_d3 | wheel | choice at step 4 | rule | F2 | pruning | bug | F3 | 4, 9 |
| S06/matched_d3 | wheel | choice at step 2 | quirk | F7 | none | |  | 9, 10 |
| S07/defaults_d1 | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 2, 5 |
| S07/matched_d1 | wheel | choice at step 1 | rule | F5 | none | |  | 5, 5 |
| S07/matched_d1_linear | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 8, 5 |
| S08/defaults_d1 | wheel | stop at step 2 | rule | F6 | none | |  | 3, 2 |
| S08/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 1, 2 |
| S08/matched_d1_minspan5 | wheel | choice at step 1 | rule | F7 | none | |  | 3, 2 |
| S09/defaults_d1 | wheel | stop at step 5 | rule | F6 | pruning | bug | F3 | 4, 5 |
| S09/matched_d1 | wheel | structure at step 1 | rule | F5 | pruning | bug | F3 | 4, 5 |
| S10/defaults_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 3, 2 |
| S10/matched_d1 | wheel | structure at step 2 | rule | F5 | rss at step 2 | rule | F5 | 8, 7 |
| S11_n03/defaults_d1 | wheel | stop at step 1 | rule | F6 | pruning | bug | F3 | 1, 1 |
| S11_n03/matched_d1 | wheel | stop at step 1 | rule | F2 | none | |  | 1, 1 |
| S11_n05/defaults_d1 | wheel | stop at step 1 | rule | F6 | none | |  | 2, 1 |
| S11_n05/matched_d1 | wheel | stop at step 1 | rule | F2 | none | |  | 1, 2 |
| S11_n08/defaults_d1 | wheel | choice at step 1 | rule | F2 | pruning | bug | F3 | 1, 2 |
| S11_n08/matched_d1 | wheel | stop at step 2 | rule | F2 | pruning | bug | F3 | 1, 2 |
| S11_n12/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S11_n12/matched_d1 | wheel | structure at step 2 | rule | F5 | stop at step 3 | rule | F2 | 4, 4 |
| S12_base/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_base/matched_d1 | wheel | choice at step 2 | rule | F5 | none | |  | 2, 2 |
| S12_base/raw_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_x_1e8/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_x_1e8/matched_d1 | wheel | choice at step 2 | rule | F5 | none | |  | 3, 2 |
| S12_x_1e8/raw_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_x_1em8/defaults_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 2 |
| S12_x_1em8/matched_d1 | wheel | choice at step 2 | rule | F5 | none | |  | 2, 2 |
| S12_x_1em8/raw_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_x_plus_1e6/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_x_plus_1e6/matched_d1 | wheel | choice at step 2 | rule | F5 | none | |  | 2, 2 |
| S12_x_plus_1e6/raw_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 2 |
| S12_y_1e9/defaults_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 2, 2 |
| S12_y_1e9/matched_d1 | wheel | choice at step 2 | rule | F5 | none | |  | 2, 2 |
| S12_y_1e9/raw_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 2, 2 |
| S12_y_1em9/defaults_d1 | wheel | stop at step 1 | rule | F6 | none | |  | 1, 2 |
| S12_y_1em9/matched_d1 | wheel | stop at step 1 | rule | F6 | none | |  | 1, 2 |
| S12_y_1em9/raw_d1 | wheel | stop at step 1 | rule | F6 | none | |  | 1, 2 |
| S13_constant_y_weighted/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_constant_y_weighted/defaults_d1 | head | none | |  | none | |  | 1, 1 |
| S13_constant_y_weighted/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_constant_y_weighted/matched_d1 | head | stop at step 1 | quirk | F19 | none | |  | 1, 1 |
| S13_constant_y_weighted_repeated/defaults_d1 | wheel | none | |  | none | |  | 1, 1 |
| S13_constant_y_weighted_repeated/matched_d1 | wheel | stop at step 1 | quirk | F19 | none | |  | 1, 1 |
| S13_equal2/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_equal2/defaults_d1 | head | choice at step 1 | rule | F7 | pruning | bug | F3 | 1, 2 |
| S13_equal2/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_equal2/matched_d1 | head | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 8, 6 |
| S13_equal2_repeated/defaults_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 1, 2 |
| S13_equal2_repeated/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 8, 6 |
| S13_int_random/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_int_random/defaults_d1 | head | pruning | rule | F4 | pruning | bug | F3 | 2, 3 |
| S13_int_random/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_int_random/matched_d1 | head | structure at step 2 | rule | F5 | choice at step 5 | rule | F5 | 11, 11 |
| S13_int_random_repeated/defaults_d1 | wheel | pruning | rule | F4 | pruning | bug | F3 | 2, 3 |
| S13_int_random_repeated/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 5 | rule | F5 | 7, 11 |
| S13_int_zeros/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_int_zeros/defaults_d1 | head | pruning | bug | F3 | pruning | rule | F4 | 1, 3 |
| S13_int_zeros/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_int_zeros/matched_d1 | head | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 6, 6 |
| S13_int_zeros_repeated/defaults_d1 | wheel | pruning | bug | F3 | pruning | rule | F4 | 1, 3 |
| S13_int_zeros_repeated/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 3 | rule | F5 | 6, 6 |
| S13_nonint/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_nonint/defaults_d1 | head | choice at step 1 | rule | F7 | none | |  | 2, 3 |
| S13_nonint/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_nonint/matched_d1 | head | structure at step 2 | rule | F5 | none | |  | 2, 3 |
| S13_unit/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_unit/defaults_d1 | head | choice at step 1 | rule | F7 | pruning | bug | F3 | 1, 2 |
| S13_unit/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S13_unit/matched_d1 | head | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 4, 2 |
| S13_unit_repeated/defaults_d1 | wheel | choice at step 1 | rule | F7 | pruning | bug | F3 | 1, 2 |
| S13_unit_repeated/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 4 | rule | F5 | 4, 2 |
| S14/defaults_d2 | wheel | choice at step 1 | rule | F2 | glm | bug | F9 | 8, 10 |
| S14/matched_d2 | wheel | choice at step 1 | rule | F5 | glm | bug | F9 | 12, 11 |
| S16_weighted/defaults_d1 | wheel | error | rule | F1 | none | |  |  |
| S16_weighted/defaults_d1 | head | choice at step 1 | rule | F7 | none | |  | 11, 15 |
| S16_weighted/defaults_d2 | wheel | error | rule | F1 | none | |  |  |
| S16_weighted/defaults_d2 | head | choice at step 1 | rule | F7 | none | |  | 18, 19 |
| S16_weighted/matched_d1 | wheel | error | rule | F1 | none | |  |  |
| S16_weighted/matched_d1 | head | choice at step 3 | rule | F5 | none | |  | 15, 14 |
| S16_weighted/matched_d2 | wheel | error | rule | F1 | none | |  |  |
| S16_weighted/matched_d2 | head | choice at step 3 | rule | F5 | none | |  | 19, 21 |
| S16_weighted_repeated/defaults_d1 | wheel | choice at step 1 | rule | F7 | none | |  | 12, 15 |
| S16_weighted_repeated/defaults_d2 | wheel | choice at step 1 | rule | F7 | none | |  | 17, 19 |
| S16_weighted_repeated/matched_d1 | wheel | choice at step 3 | rule | F5 | none | |  | 15, 14 |
| S16_weighted_repeated/matched_d2 | wheel | choice at step 3 | rule | F5 | none | |  | 19, 21 |
| S17/defaults_d1 | wheel | error | rule | F18 | none | |  |  |
| S17/matched_d1 | wheel | error | rule | F18 | none | |  |  |
| S18/defaults_d1 | wheel | design at step 1 | bug | F9 | pruning | bug | F3 | 3, 3 |
| S18/matched_d1 | wheel | design at step 1 | bug | F9 | none | |  | 2, 3 |
| S19_dummies/defaults_d1 | wheel | pruning | bug | F3 | pruning | rule | F4 | 3, 4 |
| S19_dummies/matched_d1 | wheel | structure at step 1 | rule | F5 | pruning | bug | F3 | 3, 4 |
| S20/defaults_d1 | wheel | choice at step 1 | rule | F7 | glm | bug | F9 | 4, 4 |
| S20/matched_d1 | wheel | structure at step 2 | rule | F5 | choice at step 6 | rule | F5 | 11, 7 |
| S19_factor/strings | wheel | final | bug | F17 | none | |  | 1 |
| S19_factor/codes | wheel | none | |  | none | |  | 4 |
<!-- end generated -->

## Making the numbers again

From the repository root, with R and earth 5.3.4 on the path, after `. dev/env.sh` and `validation/legacy/make_venv.sh`:

```bash
git worktree add --detach ../legacy-head legacy-1.0.4-head  # HEAD's Python code
S=validation/legacy/conformance_legacy.py
O=validation/runs/t18
uv run --frozen --group validation python $S run --out $O
uv run --frozen --group validation python $S run --out $O --code head \
    --head-path ../legacy-head --only '^(S01/|S04_|S13_|S15/|S16_|S19_factor/)'
uv run --frozen --group validation python $S probe --out $O
uv run --frozen --group validation python $S probe --out $O --code head \
    --only '^(S13_|S16_)'
uv run --frozen --group validation python $S report --out $O \
    --json validation/legacy/conformance_legacy.json --markdown $O/tables.md
```

The tables between the `generated` comments are `tables.md`. On one core the wheel's fits took 43 minutes and HEAD's 39 (the slowest single fit 141 s); the new earth fits, the probes and the report take about a minute. `run` skips the cases it has finished, so a stopped run resumes. The fits here were made first, with the same fitting code as this commit (`legacy_fit.py` and the `run` command have not changed since); the probes and the report were then made again with the final code. The runs used Python 3.12.13, numpy 2.5.3 with Accelerate, scikit-learn 1.9.1, R 4.4.3 and earth 5.3.4 on macOS, Apple silicon.
