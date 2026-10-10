# Black-box experiments with earth

Each experiment answers one question about the behavior of the R package earth 5.3.4, which `docs/algorithm.md` cites by its ID. Each has an R script `bbNN_<slug>.R`, which makes its own small deterministic data and runs in under a minute, and its saved output `bbNN_<slug>.out`. An output ends with `CHECK` lines, one per claim, computed by the script.

Clean room: the scripts use earth only as a black box. They call its functions and read return values, printed output and `trace` logs. They never print an R function body, never read earth's C or R source, and quote nothing from earth's documentation.

To run one experiment again, from this folder:

```bash
. ../../dev/env.sh && /usr/local/bin/Rscript bb01_gcv_grid.R > bb01_gcv_grid.out 2>&1
```

The outputs were made with R 4.4.3 and earth 5.3.4 on macOS 15 (Apple silicon).

A CHECK line that ends in TRUE states a result that the spec may cite. A line marked HYPOTHESIS or ALTERNATIVE records a rule that the experiment refuted; its FALSE result is expected, and a TRUE line states what earth does instead. The spec cites only TRUE lines.

Notes for the trace parser (T02):

- Trace lines number cases from 0, in earth's sort order. Among equal values of x that order is not R's `order()`, so do not align cases with `order(x)` when x has ties (bb06.8).
- Parent numbers in trace lines (`|Parent P`, the Par column of the step table and the Fast MARS queue) count slots, not `dirs` rows. Step s fills slots 2s and 2s + 1; a step that adds one term leaves slot 2s + 1 empty, so after it the slot numbers are larger than the `dirs` rows. The Terms column of the step table counts `dirs` rows. bb14 and bb11 map slots to rows with the step table.
- `iNewCol` is K, the term number searched, in a single-hinge search, and K + 1 in a pair search (bb11.5).
- Long traces: `capture.output` is slow for traces of many thousand lines; bb05 and bb14 `sink()` to a temporary file instead.

| ID | Question | Spec rules |
|---|---|---|
| bb01 | the formula of `earth:::get.gcv` over a grid of RSS, terms, penalties and n; the penalties that `earth()` accepts (−1, 0 to 1000) | GCV-1, GCV-2, CORE-2 |
| bb02 | the GCV, RSq and GRSq of fitted models; weights; several responses; the default penalty | GCV-1 to GCV-8, W-2 |
| bb03 | the default `nk`, and how steps count toward it | LIMIT-1, LIMIT-2 |
| bb05 | the knot scan for the intercept, and the automatic spans; a knot at a repeated minimum | SPAN-1 to SPAN-6, KNOT-3 to KNOT-5 |
| bb06 | the knot scan for other parents: activity, adjusted endspan and its float64 form, ties, row order | SPAN-1, SPAN-4, SPAN-5, KNOT-1 to KNOT-4 |
| bb07 | the pruning pass on a fixed basis: the prefix rule, `nprune`, `pmethod` | PRUNE-2 to PRUNE-7 |
| bb08 | exact ties in the pruning pass, in the removed term and in the size | PRUNE-3, PRUNE-5 |
| bb09 | the final coefficients, the cut stored for a linear factor, and R's rule for dependent columns | TERM-1 to TERM-3, LA-4, PRUNE-8 |
| bb10 | the termination codes; a constant y, with and without weights | CORE-4, LIMIT-2, GCV-7, EDGE-1 |
| bb11 | the collinearity tolerance and when it changes | LA-3 |
| bb12 | how the forward pass chooses a candidate: the RSS reduction, the penalty, the limit MaxLegalRssDelta | FWD-3, FWD-4 |
| bb13 | the terms that a forward step adds: pairs, single hinges, the linear candidate | FWD-3, FWD-6 |
| bb14 | when earth runs a pair search and when a single-hinge search; the RSS of a single-hinge search | LA-2, LA-7, KNOT-5 |
| bb15 | the pruning pass with two or more responses; `pmethod = "none"` with `nprune` | PRUNE-3, PRUNE-4, PRUNE-7, LIMIT-3 |
| bb16 | the Fast MARS queue: ranks, ageing, `fast_k`, slots, updates | FAST-1 to FAST-6, FWD-2, STOP-2 |
| bb17 | how earth scales the response in the forward pass | FWD-10, RESP-3 |
| bb18 | exact ties between candidates in the forward pass | FWD-5, EDGE-4 |
| bb19 | the stopping rules and their order | STOP-1 to STOP-6 |
| bb20 | several responses in the forward pass | RESP-1, RESP-2, STOP-6, FWD-4 |
| bb21 | earth's GLM refit: binomial, weights, separation, a 3-level factor, label types | GLM-1, GLM-2, GLM-4 |
| bb22 | small and degenerate inputs, constant and duplicated covariates (also at degree 2), non-finite values | EDGE-1 to EDGE-5, ERR-1 |
| bb23 | the removal of linearly dependent terms at the end of the forward pass | FWD-11 |
| bb24 | whether the 10x limit on RSS reductions applies to linear candidates | FWD-4, FAST-5 |
| bb25 | the stop when a step has no legal candidate at thresh > 0, and when the Fast MARS window ends a degree-1 pass; codes 2, 3 and 4 by GRSq(RSS, M + 1) | STOP-2 to STOP-4, FAST-6, CORE-4 |
| bb26 | the stop at an exact fit: the RSS floor relative to TSS/(n − 1), earth's rounding near it at large n, and the absolute floor with two responses; the statistics that earth reports as 0 or NaN for small values | STOP-5, STOP-7, CORE-4, PRUNE-4, PRUNE-8 |
| bb27 | earth's hidden term after some linear-option steps with `Auto.linpreds = FALSE`: the kind of parent does not decide it, and the column order can change it | FWD-11, FAST-1 |
| bb28 | whether R reads doubles back exactly from 17-digit decimal and from hexadecimal strings | LA-7 |
| bb29 | one case far above the others in x: the collinearity test rejects every knot; the mirror x to −x keeps them (spec v2) | LA-3, Conventions |
| bb30 | the termination code for a GRSq′ below −10: code 2 below −1000, finite values included (spec v2) | STOP-3, CORE-4, FAST-6 |
| bb31 | when R's `glm`, which earth's refit calls, warns that fitted probabilities are numerically 0 or 1: for a linear predictor above 30 in absolute value (spec v2) | GLM-4 |
