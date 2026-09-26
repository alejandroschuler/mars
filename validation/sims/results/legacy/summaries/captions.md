# Display captions

## `ratio_table.csv`

Ratio table. Excess risk of the current pymars and of the two ablation arms relative to earth, at 200 cases and the low-noise level, as a ratio of geometric means over repetitions, with an interval of +/-3 Monte Carlo standard errors on the log scale, the width the decision rules use. Values above 1 favor earth. The first column tests the GAP CLAIM; the two ablation columns show whether settings or rules cause the gap. The D7 row has no P-cur or P-ear value: the legacy code is too slow there.

## `ratio_table_n1000.csv`

Appendix ratio table at 1,000 cases (see ratio_table.csv), for the GAP CLAIM: D3, D4, D5 and D8 at the low-noise level are the mockup cells the full run covers at this size ('Goals'; the compute ledger's 'Legacy, 1,000 cases, the mockup cells'); the other DGPs show as missing unless the low-priority batch runs them too.

## `box_plots.png`

Per-repetition box plots. Log ratios against E-def, at 200 cases, for P-cur, P-ear, E-pym and P-fix. Show whether a few bad fits drive the means in the ratio table and the equivalence figure (the GAP and PARITY CLAIMS).

## `selection_table.csv`

Selection table. Model size and selection on D4, D6, D7 and D8 at 200 cases: the median number of terms with its interquartile range, the share of fits that use any irrelevant covariate, and the mean number of irrelevant covariates used, for E-def, P-cur and P-fix. Supports the SPARSITY CLAIM.

## `binary_outcome_table.csv`

Binary-outcome table. Ratio of excess log loss (pymars over earth) and the calibration slope, for D3-bin and D4-bin, for EarthClassifier and GLMEarth as they are in 1.0.4 (200 cases) and for P-fix (200, 1,000 and 5,000 cases). Supports the PARITY CLAIM for binary outcomes and measures the effect of finding F9 (the penalized refit).

## `appendix_table.csv`

Appendix: every cell run so far, with its Monte Carlo standard error and its failure count, for whichever claim's evidence that cell belongs to.
