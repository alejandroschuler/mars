# Differences between the new code and earth

This file triages every difference between the new fitting code (pymars 2.0) and earth 5.3.4 on the fixtures in `validation/fixtures/` (VALIDATION_PLAN.md, "Triage of differences"). The machine-readable list is `validation/differences.json`; `tests/test_conformance.py` compares each implementation with earth and fails on a difference that the list does not hold, or on an entry that no longer occurs. Each row below is one entry of that list, by its id.

The results below are for both implementations, the reference (`tests/reference/mars_ref.py`) and the fast code (`pymars._core.fit_mars`), at the head of the pull request for #173 (spec v3 with the LA-7 threshold 0.01·Π_{v∈V}(12σ_v²); earlier results were at commit 43f956f of `main`), against the stored fixtures made with R 4.4.3 and earth 5.3.4 on macOS (arm64). An entry holds for every implementation unless it names some. The fast code has no difference from earth that the reference does not share: on all 333 comparisons (133 dataset fixtures and 200 S15 draws, one whole fit each) the two implementations show the same differences, at the same steps, with the same choices and the same candidate RSS values (to a relative 1e-8, which the test checks for each entry). C07 turned the fast code on in the suite.

## What is compared

The docstring of `tests/test_conformance.py` states the rules. The test compares:

- Every dataset fixture (133 files: S01 to S20 in their modes) and the 200 draws of S15. The fixtures give earth the covariates divided by their standard deviation (LA-7), except S12's raw mode; on that matrix every σ_v² is 1, so pymars's pair threshold is 12^|V| times earth's (E1).
- The forward steps: the terms that each step adds, exactly, up to the first step where the two choices differ; the RSS after each matching step within 1e-8 of the RSS before it plus LA-5's terms in √(TSS·V) and TSS (LA-5); the termination code.
- The pruning record on the same forward basis: the fit's own when the forward paths match, else the implementation's pruning pass on earth's forward basis. The subsets of each size exactly, and the RSS and GCV of each size within LA-5's ε with R = V.

Spec v3 (#157) gives LA-5's ε the factor c = max(1e-14, u·cond·√n), with cond the condition number of the compared columns, √w-scaled to unit norm after LA-4's test (`compare.basis_cond`), and n the number of cases. The test passes cond and n at each of its three uses of ε. On the fixtures cond is far below 1e6 and n at most 1,000, so c stays near 1e-14 and no result below changed.
- The selected terms, the coefficients, GCV, RSq, GRSq, the fitted values and the predictions at new points, with the plan's tolerances and κ(B) limits (`compare.compare_fit`).

Where the plan compares a fixture otherwise:

- Integer weights (S13 and S16): pymars with the weights against earth without weights on the repeated rows, the `_repeated` fixtures (W-1). The 12 such fits whose response is not constant agree with earth. earth's own weighted fits of these data are not compared, because earth counts cases and ignores the weights in its spans and knots (W-2, W-3).
- Non-integer weights (S13_nonint, weights that sum to n): only the pruning pass on earth's weighted forward basis, against weighted earth (OQ-3). Both modes agree.
- The binary responses (S14, S20): the least-squares passes only. T14 compares the GLM refit and its probabilities, and uses `s18_multinom.json` for three classes.

## Near-ties

A step where the two choices differ is a near-tie when their RSS values on the basis before the step differ by less than 1e-7 of the RSS before the step (plan: Ties), when they are different candidates (another parent or covariate, or hinges at another knot), and when their new columns are not bitwise equal. A difference in the kind, the codes or the order of the terms of one candidate (FWD-3, FWD-6, LA-7) is not a near-tie, and neither is a tie between bitwise-equal columns, which FWD-5 breaks by its order as earth does (bb18.1). The test also fails when an entry that is not a `tie` matches a difference that the comparison labels `tie`, which bounds the band from above. A size where the pruning subsets differ is a near-tie when their RSS values differ by at most 1e-7 of the lower one plus 2ε of LA-5 (PRUNE-10), or when either subset has cond above 1e6. Under spec v3 a differing forward step is also labeled `tie` at a near-dependence (STOP-7): an LA-4 ratio of the step's columns within a factor 2 of 1e-7, or cond above 1e6. No fixture difference is of that kind. This is the threshold that OQ-2 asks T07 to fix: each program's RSS of one subset is within 1e-8 of that RSS (the tolerance for `rss.per.subset`), so a flip needs a gap below about 2e-8, and 1e-7 covers it five times over, as in the forward pass. A near-tie is labeled `tie` and passes, and the comparison of that fit's structure stops there.

A near-tie in the implementation's own candidate log at a step where both programs chose the same term does not stop the comparison, since the paths after the same choice can still be compared. S10_matched_d1 shows why this matters. There the duplicated column makes the best and the second-best candidate tie exactly at 6 of its 10 steps, and within 1e-7 at the other 4, and both programs break these ties the same way (FWD-5, bb18.1). In all, 8 of the 131 whole fits have such a step in the log of either implementation (S04 with 1,000 cases, S07 matched, S10), and all 8 agree with earth. The fast code's log shows the same 8 fits, and 1 of the 200 S15 draws, as the reference's does.

No difference on the fixtures is a near-tie. The near-ties of LA-5 and STOP-7 (a collinearity ratio or a pair-rule value at its threshold, a reduction at MaxLegal, a stopping rule at its bound) need values that the candidate log does not hold, so a difference of that kind fails the test and gets its label here by hand. None occurred.

Two fixture datasets are outside ordinary data as spec v3 defines it (Limits): S10 has a covariate that is a near copy of another after the harness's scaling (measured LA-4 ratio 1.6e-9 for the near copy against the intercept and the original, 200 cases; the bitwise copy gives 5.5e-17, and it is exempt as bitwise equal; LIM-2), and S12's x + 1e6 is 3.2e6 times its spread (LIM-1). Both still agree with earth here, but a difference on them would not be a defect: the spec allows the two programs, and earth, to differ there. The oracle tests (`tests/test_oracle.py`) leave their forward passes out.

## Results

Counts by label, over the comparisons of each implementation (the two counts are equal, so the table gives one):

| Label | Entries | Comparisons |
|---|---|---|
| rule | 4 | 33 |
| quirk | 0 | 0 |
| tie | 0 | 0 |
| numeric | 0 | 0 |
| bug | 0 | 0 |

By dataset: S04 2 comparisons (E1), S06 2 (E1), S11 1 (E3), S12 4 (E1, E2), S13 4 (E4), S14 1 (E1), S15 19 (E1). Every other comparison agrees with earth in full: 119 of the 133 dataset fixtures, and 181 of the 200 S15 draws, for the reference and for the fast code alike. Before #173 the counts were 12 comparisons, 124 of 133 and 197 of 200: the threshold of LA-7 turned 21 agreements into differences (5 datasets and 16 draws) and gave three differences (draws 21, 33 and 61) an earlier first step.

| Id | Cases | Step | pymars | earth | Candidate RSS (pymars, earth) | earth's flags | Label | Rules |
|---|---|---|---|---|---|---|---|---|
| E1 | 25 cases: S04_p05_n0200_{defaults,matched}_d2, S06_{defaults,matched}_d3, S12_x_1em8_raw_d1, S14_matched_d2 and 19 S15 draws (the table below) | forward, steps 1 to 10 | the choice of the single-hinge search, or for S12 the pair | earth's choice | per case, below | no trace for the datasets; the draws 21, 33, 61, 69, 105, 153 and 169 print the rank fix | rule | LA-7 |
| E2 | S12_y_1em9 in the defaults, matched and raw modes | the statistics | the computed values | rss, gcv and every rss.per.subset and gcv.per.subset 0; rsq, grsq NaN | | | rule | PRUNE-4, PRUNE-8 |
| E3 | S11_n03_matched_d1 | rss.per.subset of size 3 | 1.8e-63 | 0 | | | rule | PRUNE-4, PRUNE-8 |
| E4 | S13_constant_y_weighted and its repeated rows, defaults and matched modes | the whole fit | degenerate: gcv +inf, rsq and grsq 0, code 0 | gcv 0, rsq and grsq NaN, code 4 (defaults) or 7 (matched) | | | rule | EDGE-1, GCV-7, CORE-4 |

The dataset fixtures hold no trace, so earth's flags are not known there; E1 needs none, since LA-7 decides it. The S15 draws with the rank fix hold no parsed trace steps either (`steps_error`: the hidden term moves the slots that `compare.steps_from_trace` maps), only the flag that the trace printed the rank fix.

### E1: the kind of a search (rule, LA-7)

LA-7 (#173) makes pymars run a pair search when A_w is at least 0.01 times the product of (12σ_v²) over the covariates v of the term b·x. This is earth's rule, A at least 0.01, only where each covariate's sample variance is 1/12, the population variance of a uniform on [0, 1], and it does not depend on units. The harness divides each covariate by its σ_v, so on its matrix every σ_v² is 1, and the threshold of pymars is 0.01·12^|V|: 0.12 for a term of one covariate, 1.44 for two, 17.3 for three. earth's is 0.01. A search whose A_w lies in between is a pair search in earth, where the linear column b·x is in the RSS of every knot, and a single-hinge search in pymars. The harness's matrix is the one where the rule is stricter than earth's. On uniform data the sample variances are only near 1/12, so the two programs still choose differently in a minority of fits: in four settings of covariates in [0, 1] (uniform at two sample sizes, Beta(0.5, 0.5) and binary), 172 to 191 of 200 fits were equal in the adversarial review of PR #185, and in the pure-noise cell D6 at n = 200, 99.5 percent of 7,500. In D6 the rule removes the extra terms that pymars kept (#173). With skewed covariates (lognormal margins) the rule costs a little excess risk against the rule before it, which made earth's choices on those data: 1.9 percent on D4 at n = 200 and 0.7 percent at n = 1,000, resolved, and no loss was resolved in the other DGPs tested (`validation/REPORT.md`, "A known limit").

The first difference of each case below is the first step where the choices differ. In the 24 cases of the datasets and draws other than S12's raw mode, the same fit agreed with earth up to that step before #173, so the threshold decides it. The candidate of earth has the lower RSS in all 24. The comparison stops at that step, and the pruning pass on earth's forward basis agrees with earth in all 25 cases. The reference and the fast code show the same difference in every case, with the same candidate RSS values.

| Case | Step | pymars | earth | Candidate RSS (pymars, earth) |
|---|---|---|---|---|
| S04_p05_n0200_defaults_d2 | 8 | h(x0-2.37265)*h(x1-1.9027) | h(x0-1.69594)*h(x1-0.533593), h(x0-1.69594)*h(0.533593-x1) | 213.26, 211.171 (before: 235.074) |
| S04_p05_n0200_matched_d2 | 9 | h(x2-0.948602)*h(1.89005-x3), h(0.948602-x2)*h(1.89005-x3) | h(x0-1.62379)*h(x1-0.525359), h(x0-1.62379)*h(0.525359-x1) | 190.598, 184.8 (before: 206.109) |
| S06_defaults_d3 | 2 | h(x0-0.701277)*h(x2-3.23584) | x0*h(x2-3.23584) | 1.16263, 1.16167 (before: 1.19995) |
| S06_matched_d3 | 2 | h(x0-0.525596)*h(x2-3.23584) | h(x0-2.95406)*h(x2-3.23584), h(2.95406-x0)*h(x2-3.23584) | 1.16131, 1.15321 (before: 1.19995) |
| S12_x_1em8_raw_d1 | 1 | h(x0-4.08098e-09), h(4.08098e-09-x0) | h(x0-4.08098e-09) | 0.363768, 0.367819 (before: 25.8704) |
| S14_matched_d2 | 4 | h(x0-0.729043)*h(-0.27027-x1), h(0.729043-x0)*h(-0.27027-x1) | h(x0-1.4904)*h(x3-0.357811), h(x0-1.4904)*h(0.357811-x3) | 105.149, 105.131 (before: 106.663) |
| s15_draws/draw021 | 5 | h(x7-3.17981), h(3.17981-x7) | h(x1-0.0147268)*h(x2-3.08562) | 193.758, 193.412 (before: 202.154) |
| s15_draws/draw027 | 7 | h(x3-2.82974)*h(3.26525-x5) | h(x3-2.82974)*h(3.26525-x5), h(2.82974-x3)*h(3.26525-x5) | 16291.8, 16285.2 (before: 16753.1) |
| s15_draws/draw033 | 4 | h(3.3732-x0)*h(x2-0.873451), h(3.3732-x0)*h(0.873451-x2) | h(x0-3.3732)*h(x7-0.0162102) | 169.478, 167.737 (before: 172.837) |
| s15_draws/draw035 | 3 | h(x9-3.07787), h(3.07787-x9) | h(x1-3.21609)*x8 | 216.808, 210.711 (before: 222.803) |
| s15_draws/draw047 | 8 | h(x0-1.64748)*h(3.31701-x9) | x4*h(x9-3.31701) | 168.036, 167.729 (before: 171.506) |
| s15_draws/draw053 | 3 | h(x1-2.33006), h(2.33006-x1) | h(x0-3.17714)*h(x1-0.00449311) | 68.2684, 68.2062 (before: 70.9465) |
| s15_draws/draw059 | 8 | h(x5-1.18505)*h(x6-2.34279), h(1.18505-x5)*h(x6-2.34279) | h(x7-3.29596)*x9 | 7828.73, 7795.37 (before: 8095.7) |
| s15_draws/draw061 | 6 | h(x1-3.01181), h(3.01181-x1) | h(x1-0.00056534)*h(x6-3.24085) | 133.888, 133.252 (before: 136.535) |
| s15_draws/draw069 | 4 | h(x3-2.58608)*h(x6-2.05701), h(x3-2.58608)*h(2.05701-x6) | h(x1-0.00880282)*h(x4-3.05925) | 18097.1, 18020.2 (before: 18636.2) |
| s15_draws/draw091 | 6 | h(x1-1.69429)*h(2.45285-x7), h(1.69429-x1)*h(2.45285-x7) | h(x1-3.35181)*x6 | 183.404, 181.001 (before: 190.106) |
| s15_draws/draw095 | 9 | h(3.2674-x2)*h(x7-2.79922), h(3.2674-x2)*h(2.79922-x7) | h(0.193491-x4)*x6 | 40.742, 39.5657 (before: 41.7446) |
| s15_draws/draw105 | 10 | h(x2-0.758156)*h(x8-2.7665), h(x2-0.758156)*h(2.7665-x8) | h(x1-2.98943)*h(x8-0.0233684) | 192.667, 191.447 (before: 198.31) |
| s15_draws/draw113 | 6 | h(1.48749-x1)*h(x9-0.969394), h(1.48749-x1)*h(0.969394-x9) | h(x4-3.03965)*h(x8-0.00719697) | 66.2474, 66.1089 (before: 68.2445) |
| s15_draws/draw125 | 10 | h(x0-1.35594)*h(x9-0.599591) | h(x0-1.35594)*h(x9-0.599591), h(1.35594-x0)*h(x9-0.599591) | 12468.9, 12467.4 (before: 13066.2) |
| s15_draws/draw139 | 7 | h(x1-2.59373)*h(3.16777-x8), h(2.59373-x1)*h(3.16777-x8) | x4*h(x8-3.16777) | 15108.5, 15069.3 (before: 15799.6) |
| s15_draws/draw151 | 9 | h(x1-0.649867)*h(x6-0.812857), h(x1-0.649867)*h(0.812857-x6) | x0*h(x2-3.02716) | 41.4227, 41.0909 (before: 42.1018) |
| s15_draws/draw153 | 7 | h(2.87517-x4)*h(x7-2.63252), h(2.87517-x4)*h(2.63252-x7) | h(x3-0.0124945)*h(x6-3.25126) | 16628.7, 16619.9 (before: 17126.9) |
| s15_draws/draw169 | 4 | h(1.59952-x1)*h(x8-0.959608), h(1.59952-x1)*h(0.959608-x8) | h(x0-3.16186)*h(x3-0.00683095) | 70.4606, 69.8425 (before: 72.9183) |
| s15_draws/draw171 | 10 | h(x1-0.883915)*h(3.22273-x6), h(0.883915-x1)*h(3.22273-x6) | h(x6-3.22273)*x9 | 8807.83, 8788.17 (before: 9032.28) |

S12_x_1em8_raw_d1 is the case that E1 held before #173, then with the label `quirk`. The raw mode of S12 keeps x in its own units, here 1e-8 times the base x. earth runs a single-hinge search because A, the RSS of x on the intercept in these units, is 1.4e-15, below 0.01. pymars compares A_w with 0.01·(12σ_v²) of x, and A is 150 variances, so it runs a pair search at the same knot. The cause is earth's dependence on the units of x, which the Quirks section of the spec lists among the quirks that pymars does not copy, and the difference is the departure of LA-7, so E1 now holds it under `rule`. earth's single hinge is not a candidate of pymars at this step. In the other raw fits (x times 1e8, x plus 1e6, y times 1e-9 and 1e9) the forward passes agree, since the scale of y does not enter the rule and A is far from 0.01 there.

The draws 21, 33 and 61 held E5 to E7 before #173. Their first difference now comes at step 5, 4 and 6, earlier than the hidden term's, so those entries no longer occur (see below).

### E2 and E3: earth reports small sums of squares as 0 (rule, PRUNE-4 and PRUNE-8)

In S12_y_1em9, y is 1e-9 times the base y, so every sum of squares is below 3e-17. earth reports rss, gcv, rss.per.subset and gcv.per.subset as 0, and rsq and grsq as NaN (bb26.9). pymars reports the computed values, since it uses no absolute epsilon. In S11_n03_matched_d1 the model of size 3 interpolates the three cases: its RSS is 1.8e-63 in pymars and 0 in earth. The comparison leaves out the values that earth reports as 0 (below 1e-9 in pymars) and takes earth's rss, gcv, rsq and grsq from its residuals. Everything else agrees: the forward path, the subsets, the selected terms, the coefficients and the fitted values.

### E4: a constant response (rule, EDGE-1, GCV-7, CORE-4)

y is 3 at every case, so the pymars fit is degenerate: the intercept alone, gcv +inf, rsq and grsq 0, and termination code 0. The test checks these values of EDGE-1 in every degenerate fit. earth reports gcv 0, rsq and grsq NaN, and 0 for every rss.per.subset, and its forward pass runs anyway: it stops at once at its defaults (code 4), and it takes 10 steps to the term limit in the matched mode (code 7). The fixture's `fwd_rss[0]`, 8.6e-29, is the RSS of the harness's `lm.fit` on the intercept, not a value that earth reports. Both select the intercept alone, with the coefficient 3 (earth: 2.9999999999999982). earth stops with an error on the weighted data (bb10.9), so the weighted fits are compared with earth on the repeated rows.

### E5 to E7, retired by #173: earth's hidden term (quirk, FWD-11 and OQ-6)

These three entries no longer occur. Under the threshold of LA-7 (E1) the draws 21, 33 and 61 differ from earth earlier, at steps 5, 4 and 6, and the comparison stops there. What follows records what they showed when the threshold was 0.01·Π_{v∈V} σ_v² (commit 43f956f of `main`): earth's hidden term, which spec OQ-6 and FWD-11 describe, is still a quirk that pymars does not copy. Of the 11 draws that print the rank fix, 7 differ from earth now (21, 33, 61, 69, 105, 153 and 169), and no entry holds the hidden term.

All three are S15 draws at degree 2 in the matched mode (`Auto.linpreds = FALSE`), and earth's trace prints the rank fix. FWD-11 says that T07 labels such a difference `quirk`, and each first divergence comes after the first linear-option step of its fit, as a note in #44 asks. The S15 test checks both conditions for every entry that cites FWD-11. The mechanism is the same in all three. A linear-option step (the hinge at the smallest x, FWD-6) made earth add its hidden term, which fills the empty second slot of that step (FWD-9). A later term then sits in a slot above the number of pymars's terms, so pymars does not search it as a parent yet (FAST-4), while earth, with its hidden term, has one term more and does. earth's choice has the lower RSS on the common basis in all three.

A scratch copy of the reference that adds one empty queue entry (FAST-1) after the right linear option follows earth's whole path in all three draws, which confirms the mechanism. With an entry after every linear option, the copy also agrees with earth on the other 8 draws with the rank fix, but not on draw 61 (E7).

- E5 (the DGP D1 at the high noise level): the linear option of step 5; earth's parent at step 7 is h(3.24208-x7), the second term of step 6, in slot 13, while pymars has 12 terms.
- E6 (the DGP D6): the linear option of step 4; earth's parent at step 7 is h(2.03371-x2), the second term of step 6, in slot 13, while pymars has 12 terms.
- E7 (the DGP D6): linear options at steps 2 and 6, but only one of them added the hidden term, and these data do not say which. earth's parent at step 10 is h(x9-0.537277), the first term of step 9, in slot 18: pymars, with 17 terms, does not search it, and earth, with 18, does, but not slot 19. With entries after both steps, the copy searches slot 19 and takes a linear option whose RSS, 119.08, is below earth's choice, 119.36, which earth did not do. So in one fit one linear option added the hidden term and the other did not (OQ-6).

## S15

The 200 draws come from the simulation DGPs D1 to D6 and D8 (n = 200, p = 10), at degree 1 and 2, half in the matched mode and half at the defaults, all with `pmethod = "none"`. The fixture keeps earth's forward terms and its pruning records, not the data or the fitted values, so the comparison covers the forward path and `rss.per.subset` and `gcv.per.subset`, on data that the test makes again from the seeds. On another platform D8's copula (a matrix product and `ndtr`) can give a value that differs by an ulp from the fixture machine's, so the test takes each of earth's knots as the data value within a relative 1e-12 of it. On the fixture machine all 4,736 knots are data values already. Every fit takes 10 steps and stops at the term limit.

- The share of fits that agree: 181 of 200 (90.5 %); 197 of 200 (98.5 %) before #173. The subsets' RSS and GCV values agree to a relative 6.0e-15 at most (the test's own measure), on earth's forward basis where the forward paths differ.
- The first divergence: at steps 3 to 10 (2, 3, 1, 3, 3, 2, 2 and 3 draws at steps 3 to 10), all 19 by the threshold of LA-7 (`rule`, E1). 11 draws print the rank fix, all at degree 2 in the matched mode; 7 of them differ from earth (E1), and 4 agree.
- The share of fits that stop at a near-tie: 0 of 200. Counting also a near-tie where both programs chose the same term, 1 of 200 (0.5 %): draw 94, step 9, where the best and the second-best RSS differ by 9.3e-8 of the RSS before the step. Both are below the 5 % at which the plan asks for a review, and the test checks the larger count.

## Spec questions

Raised on #44 (spec v2), one comment each:

- [the pruning near-tie threshold](https://github.com/alejandroschuler/mars/issues/44#issuecomment-5877383632), which could close OQ-2;
- [the reading that only a near-tie with different choices stops the comparison](https://github.com/alejandroschuler/mars/issues/44#issuecomment-5877384033);
- [a record of the LA-5 and STOP-7 near-ties in the candidate log](https://github.com/alejandroschuler/mars/issues/44#issuecomment-5877384385), so that the tests can label them;
- [the S15 evidence on the hidden term](https://github.com/alejandroschuler/mars/issues/44#issuecomment-5877384746) (OQ-6), together with the slot rule (FAST-4, OQ-4), and [its correction for draw 61](https://github.com/alejandroschuler/mars/issues/44#issuecomment-5879039462).
