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
| bb01 | the formula of `earth:::get.gcv` over a grid of RSS, terms, penalties and n; the negative penalties that `earth()` accepts | GCV-1, GCV-2 |
| bb02 | the GCV, RSq and GRSq of fitted models; weights; several responses; the default penalty | GCV-1 to GCV-8, W-2 |
| bb03 | the default `nk`, and how steps count toward it | LIMIT-1, LIMIT-2 |
| bb05 | the knot scan for the intercept, and the automatic spans; a knot at a repeated minimum | SPAN-1 to SPAN-6, KNOT-3 to KNOT-5 |
| bb06 | the knot scan for other parents: activity, adjusted endspan and its float64 form, ties, row order | SPAN-1, SPAN-4, SPAN-5, KNOT-1 to KNOT-4 |
| bb07 | the pruning pass on a fixed basis: the prefix rule, `nprune`, `pmethod` | PRUNE-2 to PRUNE-7 |
| bb08 | exact ties in the pruning pass, in the removed term and in the size | PRUNE-3, PRUNE-5 |
| bb09 | the final coefficients, the cut stored for a linear factor, and R's rule for dependent columns | TERM-1 to TERM-3, LA-4, PRUNE-8 |
| bb10 | the termination codes | CORE-4, LIMIT-2 |
| bb11 | the collinearity tolerance and when it changes | LA-3 |
| bb14 | when earth runs a pair search and when a single-hinge search; the RSS of a single-hinge search | LA-2, LA-7, KNOT-5 |
| bb15 | the pruning pass with two or more responses; `pmethod = "none"` with `nprune` | PRUNE-3, PRUNE-4, PRUNE-7, LIMIT-3 |
