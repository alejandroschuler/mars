# Validation report for pymars 2.0

This report checks five claims about pymars 2.0, the rewrite in this fork, against the R package earth 5.3.4 and against the legacy pymars (the mars-earth 1.0.4 wheel). It follows the order of `VALIDATION_PLAN.md`, "Reporting, and what to offer upstream": conformance, gap, parity, sparsity and speed. Each claim has its own section with the design (ADEMP: aims, data-generating processes, estimands, methods, performance measures), the evidence, and a plain statement of what the evidence shows. Every table and figure says which claim it serves.

## The verdicts

| Claim | Verdict |
|---|---|
| [Conformance](#1-conformance): pymars reproduces earth's forward and pruning passes, except at documented rule differences and near-ties | Holds on the fixtures. 119 of 133 dataset fixtures and 181 of 200 random draws agree with earth in full; the other 33 comparisons show 4 labeled differences, all of the kind `rule`, none a bug, and the fast code shows exactly the same differences as the reference. On the harness's data the pair rule of #185 (E1) is stricter than earth's, which costs 21 agreements (5 dataset fixtures and 16 draws). |
| [Gap](#2-gap): the legacy pymars has a higher excess risk than earth when the truth has interactions, and the gap shrinks with earth's settings | Narrowed. At 200 cases and low noise the legacy code is 1.2 to 1.6 times worse than earth on every nonlinear DGP, additive ones included, and it stays worse on D3, D4 and D8 at 1,000 cases. Earth's settings do not close the gap, so the legacy code's rules cause most of it; on D8, and on D4 at 1,000 cases, the legacy settings alone give a part of it. |
| [Parity](#3-parity): pymars 2.0 is within a factor 1.05 of earth's excess risk on every cell | Holds, after a fix. The study at the freeze commit showed equivalence in 48 of 51 cells: on pure noise pymars was worse than earth at every n, by 7.6 percent at 200 cases (interval 1.051 to 1.101, a failure), and D1 at 5,000 cases and high noise was not decided. The cause was LA-7's pair rule (#173), which #185 changed. A rerun of the whole grid after #185 shows equivalence in 51 of 51 cells: D6 is at 0.997 at 200 cases, and no cell with signal loses. Known limit: with skewed (lognormal) covariates, which the grid does not have, the new rule is 1.9 percent worse than the old one on D4 at 200 cases ([A known limit](#a-known-limit-skewed-covariates)), inside the margin. |
| [Sparsity](#4-sparsity): on pure noise and with many irrelevant covariates, pymars selects no more terms and no more irrelevant covariates than earth | Holds. No resolved excess on D4, D7 and D8 (the claim has no margin, so this is the most a test can show). On pure noise pymars now selects as many terms as earth: 4.87 against 4.87 at 200 cases, and differences of 0.005 and 0.000 terms at 1,000 and 5,000 cases, none resolved. At the freeze it kept 0.11 more terms and 0.11 more irrelevant covariates per fit at 200 cases, because of the same pair rule. |
| [Speed](#5-speed): pymars 2.0 fits within a stated factor of earth's time | At 10,000 cases, 10 covariates and degree 2, `EarthRegressor` with numba takes 1.7 times earth's time in a fresh process and 0.92 times in a warm one; the numpy code (Intel Macs, and the plan's target of 10 times "in pure numpy") takes 2.6 and 2.7 times. At 100,000 cases the numba path is faster than earth (0.76 and 0.67) and the numpy path 2.0 times slower. At 250 cases the numba path is 21 times slower than earth in a fresh process and 3.5 times in a warm one, because of a fixed cost per fit. |

No fit failed in any arm of the simulation ([failure counts](#failure-counts)).

## Versions and data

- pymars 2.0, arm P-fix: the freeze commit `0274c5a` (branch `sim-freeze-1`; spec v3, numba 0.68.0 knot scan). This is the original study. Its parity and sparsity results showed a failure on pure noise, which issue #173 traced to the pair rule of LA-7. PR #185 (merged as `ab923ca`) changed that rule. Arm P-fix-185 is a rerun of the P-fix arm over the whole grid at `286d994`, the head of #185 when the run started; `ab923ca` differs from it in one comment and one docstring of `pymars/_linalg.py`. The rerun also holds the changes made after the freeze: #164 (LA-3's G with a dependent b·x), #167 (the compiled forward-pass steps), the release fixes of #177 and the `glm_alpha` changes of #181 and #183 (the arm keeps `glm_alpha=0`). So the rerun measures all changes since the freeze together. The phase 1 diagnosis of #173 attributes the differences on pure noise to LA-7, and the rerun does not separate the pair rule from the other changes.
- What each result rests on. The conformance counts are for the head of #185. The gap tables (Section 2) are for the freeze code, and so are their P-fix columns; the claim is about the legacy code. The parity and sparsity verdicts are for the rerun, and the freeze results stay in the report as the original study. The speed numbers are those of `main` at `95212fa`, which adds the compiled forward-pass steps of K145c (#167) and the near-span fixes of #172; the benchmark was not run again after #185.
- earth 5.3.4 on R 4.4.3. The legacy code: the mars-earth 1.0.4 wheel from PyPI (`validation/legacy/make_venv.sh`).
- Machine: Apple M1 Pro, 10 cores, macOS; Python 3.12.13, numpy 2.5.3. Every fit used one thread.
- Simulation results: `validation/sims/results/full/` (T21: P-fix and E-def, 94,500 fits), `validation/sims/results/legacy/` (T04: the legacy arms, E-pym, OLS and HGB, 49,440 fits), `validation/sims/results/t173-grid/` (the P-fix rerun after #185, 47,250 fits, paired with the stored E-def of T21) and `validation/sims/results/skew/` (the skewed-covariate study, 126,000 fits). The E-def fits are in both of the first two runs with the same seeds, and the 5,300 shared fits give identical values.
- Tables and figures: `validation/sims/report/make_report.py` builds them from those four `results.csv` files (`tables/`, `figures/`). The numbers in this report are copied from `tables/`.

## 1. Conformance

**Aim.** In matched comparisons, pymars takes the same forward steps and the same pruning path as earth and gives the same final model, except where a documented rule of the spec differs from earth on purpose, or at a near-tie, where two choices are equal up to rounding. A skeptic would reject the claim if a difference occurred that no spec rule explains.

**Data.** The fixtures in `validation/fixtures/`: 133 dataset fixtures (S01 to S20 in their modes: defaults, matched settings, raw units, weights, binary responses, degenerate data, degree 1 to 3) and the 200 random draws of S15, each fitted once by earth 5.3.4 through `Rscript`. `VALIDATION_PLAN.md`, "Test datasets", gives each dataset's purpose.

**Estimands.** For each fit: the term that each forward step adds, the RSS after each step, the termination code, the subsets of each size in pruning with their RSS and GCV, the selected terms, the coefficients, GCV, RSq, GRSq, the fitted values and the predictions.

**Methods.** Two implementations of `docs/algorithm.md`: the fast code (`pymars._core.fit_mars`) and the reference (`tests/reference/mars_ref.py`), which an agent wrote from the spec alone. `tests/test_conformance.py` compares each with earth.

**Performance measures.** Each difference from earth, with its label: `rule` (a documented rule that departs from earth), `quirk` (an earth behavior that pymars does not copy), `tie` (a near-tie: RSS values within 1e-7 of the RSS before the step), `numeric`, or `bug`. The tolerances are those of the plan's table: the forward terms exactly, the RSS values within 1e-8 relative plus the LA-5 terms, and so on (`validation/DIFFERENCES.md`, "What is compared").

**Evidence.** The table below counts the differences. It supports the claim if no difference has the label `bug`.

| Label | Entries | Comparisons |
|---|---|---|
| rule | 4 | 33 |
| quirk | 0 | 0 |
| tie | 0 | 0 |
| numeric | 0 | 0 |
| bug | 0 | 0 |

*Table 1.1. Differences between pymars and earth on 333 comparisons (133 dataset fixtures and 200 S15 draws, one whole fit each), the same for the fast code and the reference, at the head of PR #185. Supports the conformance claim. `validation/DIFFERENCES.md` explains the entries E1 to E4 one by one. Before #185 the table had 3 `rule` entries (8 comparisons) and 4 `quirk` entries (4 comparisons).*

119 of the 133 dataset fixtures and 181 of the 200 draws agree with earth in full (124 and 197 before #185). The 4 `rule` entries are:

- **E1, the kind of a search (25 comparisons: 6 dataset fixtures and 19 draws).** LA-7 makes pymars run a pair search when A_w ≥ 0.01·Π_{v∈V}(12σ_v²), a threshold that does not depend on the units of the covariates, and a single-hinge search otherwise; earth compares A with 0.01 in raw units. The harness divides each covariate by its standard deviation before it gives the data to earth, so on that matrix every σ_v² is 1 and pymars's threshold is 12^|V| times earth's. Where A lies between the two, earth runs a pair search and pymars a single-hinge search. In 24 of the 25 comparisons the same fit agreed with earth up to that step before #185. E1 was a `quirk` entry before #185, for S12's raw case alone.
- **E4**, a constant response (pymars reports gcv infinite and rsq 0, earth gcv 0 and rsq NaN), and **E2 and E3**, earth's zeroing of sums of squares below about 1e-10.

Earth's hidden term with `auto_linpreds=False` (E5 to E7, 3 `quirk` entries before #185) no longer occurs as an entry: in those 3 draws the first difference now comes earlier, at E1's step, so the comparison stops before the hidden term acts. The hidden term is still an earth quirk that pymars does not copy (`docs/differences.md`), and no fixture shows it alone. Fewer fixtures agree with earth than before #185 because the new rule is stricter than earth's on a matrix with unit variances. Dividing each covariate by σ_v·√12 in the harness would make the two thresholds equal; that needs the fixtures made again with R, and it is not done.

The fast code shows the same differences at the same steps as the reference, with the same candidate RSS values to 1e-8 (C07, PR #128; rechecked under spec v3 in PRs #160 to #162, and at the head of PR #185 for the counts above). The oracle tests compare the fast code with the reference on further random data: under spec v3, 0.4 percent of forward passes and 0.5 percent of whole fits stop at a near-tie, against the plan's limit of 5 percent (PR #160).

**Statement.** On the fixtures, pymars reproduces earth step by step, and every difference is a documented choice: E1 to E4, all of the kind `rule`. The fixtures stay inside the spec's ordinary data except S10 (a near copy) and S12 (a covariate shifted by 3.2e6 times its spread), which still agree. Outside ordinary data (the spec's Limits, LIM-1 to LIM-4) the claim does not apply: the spec allows the two programs to differ there.

## The simulation study (claims 2 to 4)

The gap, parity and sparsity claims share one design, from `VALIDATION_PLAN.md`, "Statistical performance study". This section describes it once.

### Data-generating processes

Covariates are X ~ Unif[0, 1]^p, independent, with p = 10 unless stated. The outcome is Y = f(X) + σε with ε ~ N(0, 1). The noise σ is set so that the population R², Var f / (Var f + σ²), is 0.8 (low noise, `lo`) or 0.3 (high noise, `hi`); the values come from one draw of 10^6 cases (`validation/sims/diagnostics.json`). The two levels bracket an easy and a hard problem, because a DGP on which every method is perfect, or every method fails, cannot separate the methods.

| DGP | f(x) | sd(f) | Linear share of Var f | Why it is here |
|---|---|---|---|---|
| D1 linear | x1 + 2 x2 − x3 | 0.71 | 1.00 | A linear truth: MARS should add linear terms and stop |
| D2 additive hinges | 2 (x1 − 0.3)₊ − 3 (x1 − 0.7)₊ + 2 (0.5 − x2)₊ | 0.44 | 0.80 | A truth inside the MARS class; knot recovery |
| D3 additive smooth | sin(2π x1) + 2 (x2 − 0.5)² + exp(x3) | 0.87 | 0.71 | Curvature approximated by hinges |
| D4 Friedman #1 | 10 sin(π x1 x2) + 20 (x3 − 0.5)² + 10 x4 + 5 x5 | 4.88 | 0.75 | The standard benchmark: an interaction and 5 irrelevant covariates |
| D5 hinge interaction | 4 (x1 − 0.4)₊ (x2 − 0.5)₊ + x3 | 0.35 | 0.83 | An interaction inside the MARS class |
| D6 pure noise | 0, with σ = 1 (one noise level) | 0 | | Overfitting and pruning |
| D7 many covariates | D3 with p = 50 | 0.87 | 0.71 | Selection among 47 irrelevant covariates |
| D8 correlated | D4 with a Gaussian copula, latent correlation 0.6 (0.58 between the uniforms) | 6.47 | 0.87 | Collinearity and near-ties |
| D3-bin, D4-bin | Y ~ Bernoulli(expit(λ (f(X) − E f(X)))) with f of D3 or D4 | | | Binary outcomes, additive and with an interaction |

For the binary DGPs, λ is the largest scale that keeps the 1st and 99th percentiles of the true probability μ(X) inside [0.05, 0.95]: λ = 1.60 for D3-bin and 0.28 for D4-bin, which puts 41 and 45 percent of μ(X) in [0.3, 0.7]. The grid is fully factorial over DGP, sample size (200, 1,000 and 5,000 cases) and noise level: 51 cells. Each dataset's seed comes from its cell name and repetition number, so all arms see the same training and test data in a repetition, and a pilot's repetitions are the first repetitions of the full run.

### Methods (the arms)

| Arm | Settings | Why it is here |
|---|---|---|
| E-def | earth, `degree = 2`, all else at its defaults (penalty 3, `nk` = min(200, max(20, 2p)) + 1, `thresh = 0.001`, automatic spans, `fast.k = 20`, backward pruning); for binary outcomes `glm = list(family = binomial)` | The reference for every ratio |
| P-fix | pymars 2.0 `EarthRegressor(max_degree=2)` or `EarthClassifier(max_degree=2)`, all else at its defaults at the freeze (so `glm_alpha=0`; see the note under Binary outcomes) | The candidate at the freeze commit `0274c5a`: the original study (parity, sparsity) |
| P-fix-185 (shown as "after #185") | The same settings and the same datasets at `286d994`, with the pair rule of #185; run over the whole grid, paired with the stored E-def | The candidate after the fix: the rerun (parity, sparsity) |
| P-cur | the 1.0.4 wheel, `Earth(max_degree=2)`, all else at its defaults | The library as users had it (gap) |
| P-ear | the 1.0.4 wheel with earth's settings where it has them: `penalty=1.5`, `minspan_alpha = endspan_alpha = 0.05` | Ablation: do the settings explain the gap? |
| E-pym | earth with the legacy code's settings: penalty 6, `thresh = 0`, `minspan = endspan = 1`, `fast.k = 0`, `Adjust.endspan = 1` | Ablation: earth moved to the legacy settings |
| EarthClassifier, GLMEarth | the 1.0.4 classifiers at their defaults (binary DGPs, 200 cases) | The legacy code for binary outcomes |

OLS and histogram gradient boosting (HGB) also ran as floors (T04). No claim here needs them, so this report does not show them; their numbers are in `validation/sims/results/legacy/summaries/`. The optional tuned arm of the plan did not run. All hyperparameters are fixed, because the claims are about each library at its defaults. The legacy arms did not run on D7 or at 5,000 cases, where one legacy fit takes 40 minutes or more (`VALIDATION_PLAN.md`, "Compute ledger").

### Estimands and performance measures

The estimand of each fit is its excess risk R(f̂) = E[(f̂(X) − f(X))²], the mean squared distance from the true regression function at a new X, estimated on a fresh test draw of 10,000 cases in each repetition. It equals the gain in held-out squared error over the true f, so lower is better. For binary outcomes it is the excess log loss E[KL(μ(X) ‖ μ̂(X))], with fitted probabilities clipped to [1e-6, 1 − 1e-6].

Two arms are compared by the paired log ratio g_i = log R_A,i − log R_B,i in repetition i. With n_sim repetitions, mean ḡ and standard deviation s_g, the report gives the ratio exp(ḡ), which is the ratio of the geometric means of the two excess risks, and the interval exp(ḡ ± 3 s_g/√n_sim): 3 Monte Carlo standard errors, the width that the decision rules use. The log scale turns the contrast into a ratio and tames the long right tail of squared errors.

- A difference contrast (the gap claim) is resolved when ḡ is at least 3 standard errors from 0. Resolved cells carry a `*` in the tables.
- An equivalence contrast (the parity claim) is shown when the whole interval lies inside [1/1.05, 1.05], that is when |ḡ| + 3 s_g/√n_sim ≤ log 1.05. It is a failure when the whole interval lies outside, and not decided otherwise.

The sparsity measures are the number of selected terms, whether a fit uses any irrelevant covariate, and how many it uses (on D4, D7 and D8 the irrelevant covariates are those that f does not use; on D6 every covariate is irrelevant). The binary table adds the calibration slope: the coefficient of a logistic regression of the test Y on logit(μ̂(X)), where 1 means good calibration.

### Repetitions

The legacy arms ran at the plan's caps: 300 repetitions at 200 cases and 200 at 1,000 cases (EarthClassifier 300, GLMEarth 20). GLMEarth ran 20 repetitions per binary cell only to confirm that it gives the same predictions as EarthClassifier in 1.0.4; it does, exactly, on all 40 (`validation/sims/results/legacy/README.md`), so its rows add no information beyond EarthClassifier's. Their pilot (`validation/sims/pilot_legacy.md`) showed that the caps resolve the large gaps but not every contrast; the gap tables show which.

P-fix and E-def ran at the size that a post-freeze pilot of 100 repetitions per cell asked for, by the plan's equivalence rule n_sim = (2·3·s_p / log 1.05)², rounded up to tiers of 300, 600, 1,300 and 3,000 (`validation/sims/results/full/pilot_sizing.csv`). Seven cells failed the equivalence rule in that run. A reviewer found most of them under-sized for the spread seen in the full run, and the seven were extended on the same frozen code, with nested seeds, to the larger of the plan's rule and the count needed at the observed gap: D1 at 1,000 cases (900 at high noise, 750 at low), D1 at 5,000 cases and high noise (400, then 600), D5 at 200 cases and low noise (900), D6 at 200 cases (7,500), D6 at 1,000 cases (3,300, then 3,800) and D6 at 5,000 cases (12,400).

These cells were extended because they had failed at an interim look, and two of them twice. That is a form of optional stopping: a cell that is extended after it fails has a higher chance to pass than the nominal rule gives. The sizes were fixed by the rule before each extension, and the 3-standard-error width keeps the error rate of one look small, but the four cells that passed only after an extension (D1 at 1,000 cases, both noise levels; D5 at 200 cases and low noise; D6 at 5,000 cases) are weaker evidence than the others. No further extension will run.

The rerun after #185 used exactly these sizes in every cell (the repetitions `0:n_sim` of each cell, extended cells included), and extended none. Its verdicts therefore come from sizes that its own results did not set, and the caveat above does not apply to them in the same way. The rerun fits the same datasets as the study, so the two are paired, and they are not independent studies.

## 2. Gap

**Aim.** At their defaults, the legacy pymars (P-cur) has a higher excess risk than earth (E-def) when the truth has interactions (D4, D5, D8), and the gap shrinks when the legacy code runs with earth's settings (P-ear). The two ablations ask whether the settings or the rules cause the gap: if P-ear is close to E-def, or E-pym (earth with the legacy settings) is as bad as P-cur, the settings explain it.

**Design.** All regression DGPs at 200 cases and both noise levels, and D3, D4, D5 and D8 at 1,000 cases and low noise, to check that the gap persists at a larger n. Arms P-cur, P-ear and E-pym, each against E-def; P-fix is added on the same repetitions for comparison. Measures and decision rule as above.

**Evidence.** The ratio table at 200 cases and low noise is the main display. Values above 1 favor earth.

| DGP | P-cur / E-def | P-ear / E-def | E-pym / E-def | P-fix / E-def |
|---|---|---|---|---|
| D1 linear | 0.794* [0.702, 0.897] | 1.002 [0.903, 1.112] | 0.598* [0.493, 0.725] | 1.005 [0.973, 1.038] |
| D2 additive hinges | 1.575* [1.417, 1.750] | 1.611* [1.472, 1.764] | 0.822* [0.678, 0.998] | 0.999 [0.992, 1.005] |
| D3 additive smooth | 1.551* [1.422, 1.691] | 1.447* [1.353, 1.548] | 0.979 [0.873, 1.098] | 1.008 [0.994, 1.022] |
| D4 Friedman #1 | 1.251* [1.154, 1.357] | 1.191* [1.114, 1.274] | 1.142 [0.978, 1.332] | 0.999 [0.991, 1.007] |
| D5 hinge interaction | 1.527* [1.300, 1.794] | 1.392* [1.248, 1.553] | 0.793* [0.671, 0.937] | 1.027 [0.987, 1.068] |
| D6 pure noise | 1.374* [1.132, 1.667] | 1.742* [1.461, 2.077] | 0.159* [0.096, 0.264] | 1.023 [0.919, 1.139] |
| D7 many covariates | not run (too slow) | not run (too slow) | 0.600* [0.520, 0.691] | 1.004 [0.987, 1.022] |
| D8 correlated | 1.205* [1.118, 1.299] | 1.115* [1.038, 1.197] | 1.141* [1.028, 1.265] | 1.032* [1.001, 1.065] |

*Table 2.1. Ratio of excess risk to E-def (geometric means over 300 repetitions, 200 cases, low noise; D6 has one noise level), with ±3 Monte Carlo standard errors on the log scale; `*` marks a ratio at least 3 standard errors from 1. Supports the gap claim (first column) and tests its cause (the two ablation columns). P-fix is the arm at the freeze commit, before #185, on the same 300 repetitions, so its values differ slightly from the parity table; Table A.1b has the ratios after #185.*

The first column shows the gap: the legacy code is worse than earth on every nonlinear DGP, by factors of 1.21 to 1.58, and better on the linear D1. The gap is as large on the additive D2 and D3 as on the interaction DGPs, so "when the truth has interactions" is the wrong condition. P-ear stays far from 1 on the same DGPs, so earth's settings do not repair the legacy code. Paired directly against P-cur (Table A.4), P-ear is better only on D8 (0.93) and worse on D1 and D6 (1.26 and 1.27), where earth's smaller penalty lets the legacy code overfit more. E-pym does not reproduce the gap either: earth with the legacy settings is as good as or better than earth at its defaults on most DGPs, and much better on D6 (0.16), where its larger penalty prunes the noise away. E-pym is resolved worse than E-def on D8 (1.14), and at 1,000 cases also on D4 (1.07) and D8 (1.16; Table A.3). On D8 at 1,000 cases that is about half of the legacy code's gap on the log scale (log 1.162 / log 1.319 = 0.54), so there the legacy settings alone explain a part of the gap.

At high noise (Table A.2) the gap stays on the additive DGPs (D2 1.33, D3 1.41) but is not resolved on the interaction DGPs (D4 1.02, D5 1.06, D8 0.91), where the pilot had already said that 300 repetitions could not separate them. At 1,000 cases (Table A.3) it persists on D3, D4 and D8 (1.28 to 1.38) and reverses on D5 (0.76).

![Box plots of per-repetition log ratios against E-def at 200 cases](sims/report/figures/per_rep_boxes.png)

*Figure 2.1. Per-repetition log ratio of excess risk against E-def at 200 cases, for P-cur, P-ear, E-pym and P-fix, by DGP; blue low noise, orange high noise; boxes are quartiles, whiskers the 5th and 95th percentiles; the axis is cut at −4 and 3 (D6's E-pym box runs further left). It shows whether a few bad fits drive the means of Table 2.1 and of Figure 3.1.*

The legacy arms' boxes are wide and shifted, so the gap is a shift of the whole distribution, and a few bad fits do not drive it. The P-fix boxes have collapsed to a line at 0, because 72 to 100 percent of P-fix fits give the same excess risk as earth to 1e-9 (the "equal fits" column of Table A.1).

**Statement.** The legacy pymars has a higher excess risk than earth on every nonlinear DGP at 200 cases and low noise, additive or not, by factors of 1.2 to 1.6, and on D3, D4 and D8 at 1,000 cases. Earth's settings do not close the gap, and earth with the legacy settings opens only a part of it, on D8 and on D4 at 1,000 cases, so the legacy code's rules and implementation cause most of the gap and its settings a part of it on D4 and D8. The claim therefore changes from "when the truth has interactions" to "when the truth is nonlinear", with the condition that at high noise and 200 cases only the additive DGPs show a resolved gap. pymars 2.0 has no such gap on the same repetitions (last column).

## 3. Parity

**Aim.** pymars 2.0 (P-fix) at its defaults (`glm_alpha=0`, the default at the freeze, for the binary cells) with `max_degree=2` has an excess risk within a factor of 1.05 of earth's (E-def), in either direction, on every cell of the grid, and an excess log loss within 5 percent on the binary DGPs. The plan named two likely boundaries: small samples with many irrelevant covariates, and correlated covariates.

**Design.** All 51 cells, P-fix against E-def, with 300 to 12,400 repetitions per cell as described under [Repetitions](#repetitions). The test is the equivalence rule with the margin log 1.05. The study ran twice on the same cells, seeds and repetitions. The original study used P-fix at the freeze commit. It failed on pure noise, issue #173 found the cause, and PR #185 changed the rule. Then the whole grid ran again with P-fix at `286d994` (arm P-fix-185, "after #185"), against the stored E-def fits. The verdict rests on the rerun.

**The rerun in ADEMP.** *Aim:* with the pair rule of #185, pymars is within a factor 1.05 of earth in every cell, and no cell with signal gets worse; D6 is the cell that should improve. *Data:* the 51 cells and the datasets of the study. *Estimand:* the excess risk of each fit (excess log loss for the binary cells). *Methods:* P-fix at `286d994` against the stored E-def; earth, the data and the test sets do not change with pymars's code, so the E-def fits are those of the study. *Performance measures:* the paired log ratio and the equivalence rule above, and an acceptance test that was fixed before the rerun ran (the brief of #173): (1) in every cell with signal the ratio does not rise above the upper bound of the study's cell, and no cell goes from equivalent to another verdict; (2) D6's ratio falls at every n.

### The study at the freeze commit

**Evidence.** The equivalence figure shows every cell.

![Ratio of excess risk P-fix over E-def for every cell](sims/report/figures/parity.png)

*Figure 3.1. The study at the freeze commit. Ratio of excess risk (excess log loss for D3-bin and D4-bin), P-fix over E-def, with ±3 Monte Carlo standard errors, one row per DGP and one panel per sample size; the shaded band is the equivalence margin [1/1.05, 1.05]; circles low noise, squares high noise, diamonds one level; blue equivalent, orange not decided, red not equivalent. Supports the parity claim across the grid; Table A.1 has every value and the number of repetitions.*

Equivalence is shown in 48 of the 51 cells. The three other cells:

| DGP | noise | n | reps | P-fix / E-def | equal fits | decision |
|---|---|---|---|---|---|---|
| D1 linear | hi | 5,000 | 600 | 1.038 [1.019, 1.056] | 81% | not decided |
| D6 pure noise | na | 200 | 7,500 | 1.076 [1.051, 1.101] | 85% | not equivalent |
| D6 pure noise | na | 1,000 | 3,800 | 1.030 [1.007, 1.053] | 78% | not decided |

*Table 3.1. The cells where equivalence is not shown at the freeze commit, from Table A.1. "equal fits" is the share of repetitions where the two excess risks agree to 1e-9. These cells limited the parity claim before #185.*

- **D6 at 200 cases is a failure.** On pure noise pymars is 7.6 percent worse than earth, and the whole interval [1.051, 1.101] lies above the margin. With 7,500 repetitions it is one of the most precise cells in the study.
- **D1 at 5,000 cases and high noise is not decided** at its 600 repetitions. The interval [1.019, 1.056] crosses the upper margin; at the observed spread about 1,300 repetitions would decide it.
- **D6 at 1,000 cases is not decided** at its 3,800 repetitions. The interval [1.007, 1.053] crosses the upper margin; about 5,000 repetitions would decide it.

Two patterns stand out. First, the cells off the margin are on D6 and D1, the DGPs with the least nonlinear signal, and not where the plan expected them: D7 with many covariates and D8 with correlated covariates pass at every n. Second, the differences lean one way. In 46 of the 51 cells the two programs differ in at least one repetition; in 38 of those 46 pymars has the larger excess risk, the interval lies wholly above 1 in 12, and wholly below 1 in none. In the equivalent cells the lean is at most about 3 percent, apart from D6 at 5,000 cases (4.0 percent).

The fits differ in a minority of repetitions (a median of 11 percent of repetitions per cell, at most 28 percent for the regression DGPs), and the ratio comes from those. On D6 at 200 cases the fits differ in 15 percent of repetitions; in those, pymars has more terms than earth in 53 percent, fewer in 29 percent, and the larger excess risk in 69 percent. On the binary DGPs the share of equal fits is lower (46 to 63 percent) because the logistic refits of the same terms differ in the last digits; the ratios there are still within 2.2 percent.

The study at the freeze did not say why pymars keeps more terms on noise. Issue #173 asked that, and the next part gives the answer.

### The cause on noise: LA-7's pair rule (#173)

Phase 1 of #173 compared the forward passes of pymars (at `7cf85d0`) and of earth, which ran as a black box with its trace of the forward pass, in the repetitions of D6 at 200 cases where the two choose different models. There are 1,145 of them among the 7,500, and in all 1,145 the first difference is LA-7's pair rule. pymars had run a pair search where earth runs a single-hinge search. The threshold of pymars, 0.01·Π_{v∈V}σ_v² over the covariates of the candidate term b·x, is 144 times smaller than earth's raw 0.01 at degree 2 for Unif[0, 1] covariates, whose variance is 1/12. So pymars ran pair searches more often, and it often added b·x on a parent that is active on few cases (1,053 of the 1,145). Two counterfactuals agree: pymars with earth's raw threshold gives earth's model in 7,500 of 7,500 repetitions, and earth fitted on the covariates divided by their standard deviation gives pymars's model. None of the other causes that phase 1 checked (a near-tie broken the other way, the candidate queue of Fast MARS, pruning or GCV) is the first difference in any of the 1,145. The evidence is on #173.

The project owner chose the unit-free threshold 0.01·Π_{v∈V}(12σ_v²). It is earth's 0.01 for covariates with variance 1/12 and it does not change with the units of the covariates. On D6 at 200 cases it gave a ratio of 0.997 [0.993, 1.001] with 99.5 percent equal fits before the whole grid ran again. PR #185 made it the rule.

### The rerun after #185

![Ratio of excess risk P-fix after #185 over E-def for every cell](sims/report/figures/parity_rerun.png)

*Figure 3.2. The rerun after #185. As Figure 3.1, for P-fix at `286d994`: ratio of excess risk (excess log loss for D3-bin and D4-bin) over the stored E-def, with ±3 Monte Carlo standard errors, on the cells and repetitions of Figure 3.1. All 51 cells are blue, and their intervals lie inside the band. Supports the parity claim after the fix; Table A.1b has every value.*

Equivalence is shown in 51 of the 51 cells. The three cells that were not shown at the freeze are equivalent: D6 at 200 and 1,000 cases and D1 at 5,000 cases and high noise. D6 at 5,000 cases, equivalent by a hair at the freeze, now has an interval of width 0.0003. Table 3.2 compares these four cells with the study.

| DGP | noise | n | reps | freeze: P-fix / E-def | after #185: P-fix / E-def | decision | equal fits to E-def | paired change |
|---|---|---|---|---|---|---|---|---|
| D1 linear | hi | 5,000 | 600 | 1.038 [1.019, 1.056] | 1.000 [1.000, 1.000] | not decided to equivalent | 81% to 100% | 0.964 [0.947, 0.981] |
| D6 pure noise | na | 200 | 7,500 | 1.076 [1.051, 1.101] | 0.997 [0.993, 1.001] | not equivalent to equivalent | 85% to 99% | 0.927 [0.905, 0.949] |
| D6 pure noise | na | 1,000 | 3,800 | 1.030 [1.007, 1.053] | 1.001 [0.999, 1.003] | not decided to equivalent | 78% to 97% | 0.972 [0.951, 0.994] |
| D6 pure noise | na | 5,000 | 12,400 | 1.040 [1.031, 1.050] | 1.000 [1.000, 1.000] | equivalent | 77% to 100% | 0.961 [0.953, 0.969] |

*Table 3.2. The four cells that changed most between the study at the freeze and the rerun after #185, from Table A.1 and Table A.1b. "equal fits" is the share of repetitions where pymars and earth agree to 1e-9, at the freeze and after #185. "paired change" is the ratio of pymars's excess risk after #185 to its excess risk at the freeze on the same datasets (geometric mean over the repetitions, ±3 Monte Carlo standard errors); below 1 means that pymars improved. Supports the parity claim and the diagnosis of #173: the cells that failed are the cells that changed.*

The acceptance test of the rerun holds.

1. In the 48 cells with signal the ratio lies between 0.998 and 1.002, and the highest upper bound is 1.009 (D3 at 200 cases and low noise). No such cell has a ratio above the upper bound of the study's cell, and none goes from equivalent to another verdict. In 7 of them the ratio is above the point value of the study, all by less than 0.0012 (from 0.9989 to 0.9999 at the freeze to 0.99998 to 1.0000 now).
2. D6's ratio falls at every n: from 1.076 to 0.997 at 200 cases, from 1.030 to 1.001 at 1,000 cases and from 1.040 to 1.000 at 5,000 cases. On the same datasets the excess risk of pymars on D6 falls by 7.3 percent at 200 cases (paired change 0.927, interval 0.905 to 0.949), by 2.8 percent at 1,000 cases and by 3.9 percent at 5,000 cases.

The lean that the study found, with pymars the worse of the two more often than not, is gone. No interval lies wholly above 1 or wholly below 1 (12 lay wholly above 1 at the freeze). In 32 of the 51 cells the two programs differ in at least one repetition (46 at the freeze). The share of repetitions with the same excess risk has a median of 99 percent per cell (89 percent at the freeze); in the regression cells it is 96 to 100 percent (72 to 100 percent at the freeze). On the binary DGPs it is 46 to 66 percent (46 to 63 percent at the freeze), because the logistic refits of the same terms differ in the last digits; the ratios there are 1.000 to 1.001.

Three limits of the rerun.

- **It ran at `286d994`, which also holds #164 and #167** from after the freeze, so the change between the two studies is the change of all of them. The phase 1 diagnosis attributes the noise differences to LA-7. It ran at `7cf85d0`, which holds #164 and #167, and found the differences there, and the raw threshold removed them.
- **The rule is earth's only where each covariate has the sample variance 1/12.** On uniform data the two programs still differ in a minority of fits: 99.5 percent of the 7,500 fits of D6 at 200 cases are equal, and 172 to 191 of 200 in four settings of covariates in [0, 1] that the review of #185 ran.
- **The grid has Unif[0, 1] covariates only.** A later section tests skewed covariates, where the new rule costs a little against the old one ([A known limit](#a-known-limit-skewed-covariates)).

**Statement.** pymars 2.0 at its defaults (`glm_alpha=0` for the binary cells) with `max_degree=2` matches earth's excess risk within 5 percent on all 51 cells after the pair rule of #185, including every interaction, many-covariate, correlated-covariate and binary cell, and no cell with signal lost against the study at the freeze. The study at the freeze matched earth on 48 of the 51 cells and failed on pure noise, where pymars was worse than earth at every sample size (7.6 percent at 200 cases). The cause was the pair threshold of LA-7, far below earth's on uniform covariates, and the rule of #185 removes it: D6's ratio is 0.997, 1.001 and 1.000 at 200, 1,000 and 5,000 cases. On pure noise both excess risks are small, so the ratio is one of two small numbers. The claim holds for covariates in [0, 1], as the grid has them, and with the known limit on skewed covariates.

## 4. Sparsity

**Aim.** On pure noise and with many irrelevant covariates, pymars selects no more terms and no more irrelevant covariates than earth. Overfitting shows most at small n, so the cells are D4, D6, D7 and D8 at 200 cases.

**Design.** The repetitions of the parity and gap runs, and of the parity rerun after #185; no extra fits. For P-fix at the freeze, P-fix after #185 and P-cur the per-repetition differences from E-def are paired, and their means get ±3 Monte Carlo standard errors. Table 4.1 shows the cells at 200 cases for every arm. Table 4.2 shows pure noise at every sample size, where the freeze showed an excess.

**Evidence.**

| DGP | noise | n | arm | terms, median (IQR) | terms, mean | uses an irrelevant covariate | irrelevant covariates, mean | minus E-def: terms / irrelevant covariates |
|---|---|---|---|---|---|---|---|---|
| D4 Friedman #1 | lo | 200 | E-def | 14 (13 to 15) | 13.95 | 77% | 1.06 |  |
| D4 Friedman #1 | lo | 200 | P-cur | 15 (13 to 16) | 14.64 | 100% | 2.83 | +0.69 [+0.21, +1.17] / +1.77 [+1.56, +1.98] |
| D4 Friedman #1 | lo | 200 | P-fix (freeze) | 14 (13 to 15) | 13.98 | 76% | 1.05 | +0.03 [-0.03, +0.08] / -0.01 [-0.03, +0.01] |
| D4 Friedman #1 | lo | 200 | P-fix (after #185) | 14 (13 to 15) | 13.95 | 77% | 1.06 | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |
| D4 Friedman #1 | hi | 200 | E-def | 10 (8 to 11) | 9.62 | 88% | 1.52 |  |
| D4 Friedman #1 | hi | 200 | P-cur | 11 (9 to 13) | 11.03 | 100% | 3.27 | +1.41 [+0.82, +2.00] / +1.75 [+1.52, +1.98] |
| D4 Friedman #1 | hi | 200 | P-fix (freeze) | 10 (8 to 11) | 9.70 | 88% | 1.54 | +0.08 [-0.06, +0.22] / +0.02 [-0.03, +0.08] |
| D4 Friedman #1 | hi | 200 | P-fix (after #185) | 10 (8 to 11) | 9.63 | 88% | 1.52 | +0.01 [-0.01, +0.02] / +0.00 [-0.01, +0.01] |
| D6 pure noise | na | 200 | E-def | 4 (3 to 7) | 4.87 | 92% | 3.41 |  |
| D6 pure noise | na | 200 | P-cur | 9 (7 to 11) | 8.82 | 100% | 6.56 | +4.00 [+3.32, +4.67] / +3.12 [+2.73, +3.50] |
| D6 pure noise | na | 200 | P-fix (freeze) | 5 (3 to 7) | 4.98 | 92% | 3.52 | +0.11 [+0.07, +0.15] / +0.11 [+0.09, +0.14] |
| D6 pure noise | na | 200 | P-fix (after #185) | 4 (3 to 7) | 4.87 | 92% | 3.40 | -0.01 [-0.01, +0.00] / -0.00 [-0.01, +0.00] |
| D7 many covariates | lo | 200 | E-def | 22 (20 to 25) | 22.56 | 100% | 10.17 |  |
| D7 many covariates | lo | 200 | P-cur | not run (too slow) |  |  |  |  |
| D7 many covariates | lo | 200 | P-fix (freeze) | 22 (20 to 25) | 22.51 | 100% | 10.15 | -0.05 [-0.41, +0.31] / -0.02 [-0.23, +0.19] |
| D7 many covariates | lo | 200 | P-fix (after #185) | 22 (20 to 25) | 22.49 | 100% | 10.16 | -0.07 [-0.21, +0.07] / -0.01 [-0.04, +0.02] |
| D7 many covariates | hi | 200 | E-def | 19 (16 to 23) | 19.74 | 100% | 10.41 |  |
| D7 many covariates | hi | 200 | P-cur | not run (too slow) |  |  |  |  |
| D7 many covariates | hi | 200 | P-fix (freeze) | 20 (16 to 23) | 19.88 | 100% | 10.51 | +0.14 [-0.33, +0.62] / +0.11 [-0.17, +0.39] |
| D7 many covariates | hi | 200 | P-fix (after #185) | 20 (16 to 23) | 19.73 | 100% | 10.39 | -0.01 [-0.04, +0.03] / -0.01 [-0.04, +0.02] |
| D8 correlated | lo | 200 | E-def | 13 (11 to 14) | 12.65 | 80% | 1.28 |  |
| D8 correlated | lo | 200 | P-cur | 14 (13 to 16) | 14.22 | 99% | 2.88 | +1.65 [+1.16, +2.14] / +1.62 [+1.41, +1.83] |
| D8 correlated | lo | 200 | P-fix (freeze) | 13 (11 to 14) | 12.72 | 81% | 1.30 | +0.07 [-0.01, +0.15] / +0.01 [-0.03, +0.05] |
| D8 correlated | lo | 200 | P-fix (after #185) | 13 (11 to 14) | 12.66 | 80% | 1.28 | +0.01 [-0.03, +0.05] / +0.00 [-0.01, +0.01] |
| D8 correlated | hi | 200 | E-def | 8 (6 to 10) | 8.11 | 86% | 1.78 |  |
| D8 correlated | hi | 200 | P-cur | 11 (9 to 12) | 10.60 | 100% | 3.25 | +2.45 [+1.88, +3.02] / +1.53 [+1.31, +1.76] |
| D8 correlated | hi | 200 | P-fix (freeze) | 8 (6 to 10) | 8.09 | 87% | 1.79 | -0.01 [-0.15, +0.12] / +0.01 [-0.04, +0.07] |
| D8 correlated | hi | 200 | P-fix (after #185) | 8 (6 to 10) | 8.12 | 86% | 1.78 | +0.01 [-0.02, +0.04] / -0.00 [-0.01, +0.00] |

*Table 4.1. Model size and selection at 200 cases: the median and the mean number of terms (intercept included), with the interquartile range of the terms, the share of fits that use any irrelevant covariate, the mean number of irrelevant covariates used, and the mean paired difference from E-def with ±3 Monte Carlo standard errors. "P-fix (freeze)" is the study at the freeze commit and "P-fix (after #185)" the rerun at `286d994`. 300 repetitions (D6: 7,500 for P-fix, P-fix after #185 and E-def and 300 for P-cur; D8: 600 for P-fix, P-fix after #185 and E-def). P-cur did not run on D7. Supports the sparsity claim.*

The legacy code is much less sparse than earth: on pure noise it selects 4 more terms and 3 more irrelevant covariates per fit, and on D4 and D8 it uses an irrelevant covariate in nearly every fit. pymars 2.0 at the freeze shows no resolved excess on D4, D7 and D8, where every interval covers 0. On D6 it selected 0.11 more terms and 0.11 more irrelevant covariates per fit than earth (intervals [+0.07, +0.15] and [+0.09, +0.14]), about 2 percent of earth's mean model size of 4.87 terms. This agreed with the parity failure in the same cell: more noise terms give a larger excess risk. After #185 the difference on D6 is gone: pymars selects 4.87 terms on average, as earth does, and the differences are −0.005 terms ([−0.014, +0.003]) and −0.003 irrelevant covariates ([−0.008, +0.002]). On D4, D7 and D8 every interval still covers 0, and the intervals are narrower than at the freeze because the fits equal earth's more often. For example, D7 at high noise gives −0.01 terms [−0.04, +0.03] after #185 against +0.14 [−0.33, +0.62] at the freeze.

| DGP | noise | n | arm | terms, median (IQR) | terms, mean | uses an irrelevant covariate | irrelevant covariates, mean | minus E-def: terms / irrelevant covariates |
|---|---|---|---|---|---|---|---|---|
| D6 pure noise | na | 200 | E-def | 4 (3 to 7) | 4.87 | 92% | 3.41 |  |
| D6 pure noise | na | 200 | P-fix (freeze) | 5 (3 to 7) | 4.98 | 92% | 3.52 | +0.108 [+0.070, +0.146] / +0.113 [+0.086, +0.141] |
| D6 pure noise | na | 200 | P-fix (after #185) | 4 (3 to 7) | 4.87 | 92% | 3.40 | -0.005 [-0.014, +0.003] / -0.003 [-0.008, +0.002] |
| D6 pure noise | na | 1,000 | E-def | 6 (4 to 8) | 6.30 | 98% | 4.40 |  |
| D6 pure noise | na | 1,000 | P-fix (freeze) | 6 (4 to 8) | 6.35 | 98% | 4.45 | +0.049 [-0.003, +0.102] / +0.052 [+0.017, +0.086] |
| D6 pure noise | na | 1,000 | P-fix (after #185) | 6 (4 to 8) | 6.31 | 98% | 4.40 | +0.005 [-0.007, +0.017] / +0.002 [-0.006, +0.009] |
| D6 pure noise | na | 5,000 | E-def | 6 (4 to 9) | 6.42 | 96% | 4.35 |  |
| D6 pure noise | na | 5,000 | P-fix (freeze) | 6 (4 to 9) | 6.51 | 96% | 4.43 | +0.090 [+0.061, +0.120] / +0.082 [+0.065, +0.100] |
| D6 pure noise | na | 5,000 | P-fix (after #185) | 6 (4 to 9) | 6.42 | 96% | 4.35 | +0.000 [-0.001, +0.001] / +0.000 [-0.001, +0.001] |

*Table 4.2. Pure noise (D6) at every sample size: the same measures as Table 4.1, with three decimals for the differences, for earth, pymars at the freeze and pymars after #185. 7,500, 3,800 and 12,400 repetitions at 200, 1,000 and 5,000 cases. Supports the sparsity claim on pure noise, and it shows that the excess at the freeze is gone at every n.*

At the freeze pymars kept more terms or irrelevant covariates than earth at every n: 0.11 and 0.11 more at 200 cases, 0.05 more irrelevant covariates at 1,000 cases ([+0.02, +0.09]; terms +0.05 [−0.00, +0.10]), and 0.09 more terms ([+0.06, +0.12]) and 0.08 more irrelevant covariates ([+0.06, +0.10]) at 5,000 cases. After #185 the differences are +0.005 terms ([−0.007, +0.017]) and +0.002 irrelevant covariates at 1,000 cases and +0.000 at 5,000 cases, none resolved, and the intervals are no wider than 0.03 at any n.

**Statement.** When the truth has signal, also with 47 irrelevant covariates and with correlated covariates, the study finds no resolved excess of pymars 2.0 over earth in terms or in irrelevant covariates, at the freeze and after #185; the claim has no margin, so it cannot show more than that, and on D7 the intervals at the freeze were wide (for example [−0.33, +0.62] terms at high noise). On pure noise pymars kept slightly more terms than earth at every sample size at the freeze, and the cause was the pair rule of LA-7 (#173). After #185 pymars selects as many terms and as many irrelevant covariates as earth at every sample size, 4.87 terms against 4.87 at 200 cases, and the claim holds without the condition that the freeze gave it. The legacy code kept far more.

## 5. Speed

**Aim.** pymars 2.0 fits within a stated factor of earth's time. The plan's target was 10 times earth's time at 10,000 cases, 10 covariates and degree 2, "in pure numpy" (`VALIDATION_PLAN.md`, "Fast path"; definition of done, item 5). Since then numba became a dependency where it has wheels, so the report states both paths: numba, which most users run, and the numpy code, which Intel Macs run and which the target names.

**Design.** The benchmark of `validation/bench/` (`VALIDATION_PLAN.md`, "Benchmark design"): Friedman #1 data with fixed seeds, one factor at a time around 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms; cases from 250 to 100,000. Each point is the median of 3 runs, each in a fresh process with one thread, 4 cells at a time. pymars runs `EarthRegressor` and the internal `fit_mars` with numba; earth runs at its defaults and with `fast.k = 0`; the legacy code runs up to 2,000 cases (its times are from the T19 run at `ae9d199`, since that code does not change). The timer covers the fit alone, after the imports and the data loading. In a fresh process it still includes pymars' first-call costs, about 0.3 s, mostly loading numba's compiled code from its disk cache. So `validation/sims/report/warm_times.py` also times the first fit of a fresh process and the median of 3 later fits in the same process, as in cross-validation or a simulation, for the numba path and, with `pymars._kernels.ENABLED = False`, for the numpy path.

**Evidence.**

![Fit time against the number of cases](sims/report/figures/speed.png)

*Figure 5.1. Fit time against the number of cases, log-log, at 10 covariates, degree 2 and 21 terms: pymars 2.0 `EarthRegressor` (numba) in a fresh and in a warm process, earth at its defaults, and the legacy 1.0.4 code (up to 2,000 cases). Supports the speed claim. Table 5.1 adds the numpy path.*

| Cases | earth defaults (s) | numba, fresh (s) | numba, warm (s) | numpy, fresh (s) | numpy, warm (s) | Factor: numba fresh / warm | Factor: numpy fresh / warm | legacy 1.0.4 (s) |
|---|---|---|---|---|---|---|---|---|
| 250 | 0.018 | 0.373 | 0.063 | 0.150 | 0.148 | 21 / 3.5 | 8.3 / 8.2 | 25.8 |
| 1,000 | 0.049 | 0.388 | 0.094 | 0.244 | 0.238 | 7.9 / 1.9 | 5.0 / 4.9 | 144 |
| 10,000 | 0.464 | 0.809 | 0.425 | 1.22 | 1.26 | 1.7 / 0.92 | 2.6 / 2.7 | not run |
| 100,000 | 7.57 | 5.77 | 5.09 | 15.3 | 15.3 | 0.76 / 0.67 | 2.0 / 2.0 | not run |

*Table 5.1. Fit times of `EarthRegressor` and earth, and their ratio, at 10 covariates, degree 2 and 21 terms, on `main` at `95212fa`. Earth and numba fresh: median of 3 fresh processes in the benchmark (`validation/bench/REPORT.md`). numba warm, numpy fresh and numpy warm: `warm_times.py` (`warm_times.csv`, `warm_times_numpy.csv`), where "fresh" is the first fit of a fresh process and "warm" the median of 3 later fits; it ran one size at a time, while the benchmark ran 4 cells at a time. At 10,000 cases the first numba fit of the script took 0.69 s against 0.81 s in the benchmark, so the run conditions differ by about 15 percent, and a factor near 1 means "about equal". Supports the speed claim.*

At the plan's point, 10,000 cases, the numpy path takes 2.6 times earth's time in a fresh process and 2.7 times in a warm one, so the target of 10 "in pure numpy" is met. With numba, `EarthRegressor` takes 1.7 times earth's time in a fresh process (0.81 s against 0.46 s) and about the same time in a warm one (0.92 times); the internal `fit_mars` also takes 1.7 times. At 100,000 cases the numba path is faster than earth (5.8 s fresh and 5.1 s warm against 7.6 s), and the numpy path takes 2.0 times earth's time. At small n a fixed cost dominates the numba path: about 0.3 s more in the first fit of a process, and about 0.06 s per fit in a warm one, against earth's 0.02 s at 250 cases. The numpy path has no cache to load, so its first fit costs no more than the later ones, and below about 1,000 cases it is faster than the numba path in a fresh process. Against the legacy code, pymars 2.0 is 69 times faster at 250 cases and 1,400 times faster at 2,000 cases. From 10,000 to 100,000 cases the numba time grows by a factor of 7.1, a little less than linearly in n (`validation/bench/REPORT.md` has the other factors). With weights, pymars takes 0.45 s at 1,000 cases, where earth takes 16 s.

**Statement.** At 10,000 cases, 10 covariates and degree 2, pymars 2.0 fits within 2.7 times earth's time on the numpy path, which meets the plan's target of 10, and within 1.7 times with numba, or in about the same time in a warm process. At 100,000 cases the numba path is faster than earth and the numpy path 2.0 times slower. At a few hundred cases earth is 3.5 to 21 times faster than the numba path, because of a fixed cost of about 0.06 s per fit and about 0.3 s more in the first fit of a process. The legacy code was 69 to 1,400 times slower (250 to 2,000 cases).

## Failure counts

| arm | fits attempted | failed |
|---|---|---|
| E-def | 47,250 | 0 |
| E-pym | 5,300 | 0 |
| EarthClassifier | 600 | 0 |
| GLMEarth | 40 | 0 |
| HGB | 15,300 | 0 |
| OLS | 13,500 | 0 |
| P-cur | 4,700 | 0 |
| P-ear | 4,700 | 0 |
| P-fix | 47,250 | 0 |
| P-fix-185 | 47,250 | 0 |
| E-def (skewed covariates) | 42,000 | 0 |
| P-new (skewed covariates) | 42,000 | 0 |
| P-old (skewed covariates) | 42,000 | 0 |

*Fits attempted and failed per arm. E-def counts the T21 run only; the 5,300 E-def fits of the T04 run repeat the same seeds and also had no failure (`validation/sims/results/full/failure_counts.csv` and `validation/sims/results/legacy/failure_counts.csv` give them per cell). P-fix-185 is the rerun after #185, and the three arms marked "skewed covariates" are the study of the known limit (`validation/sims/results/t173-grid/failure_counts.csv` and `validation/sims/results/skew/failure_counts.csv`). Every number in this report comes from completed fits, and none failed.*

## Binary outcomes

The binary DGPs belong to the parity claim. This table adds the legacy classifiers and the calibration slope, which measures the effect of the legacy code's penalized refit (F9 in the plan).

| DGP | n | arm | reps | excess log loss / E-def | calibration slope, median |
|---|---|---|---|---|---|
| D3-bin | 200 | E-def | 1,300 | 1 (reference) | 0.44 |
| D3-bin | 200 | EarthClassifier | 300 | 0.520* [0.467, 0.578] | 1.11 |
| D3-bin | 200 | GLMEarth | 20 | 0.760 [0.418, 1.380] | 1.11 |
| D3-bin | 200 | P-fix | 1,300 | 1.022* [1.008, 1.037] | 0.43 |
| D3-bin | 1,000 | E-def | 300 | 1 (reference) | 0.78 |
| D3-bin | 1,000 | P-fix | 300 | 1.008 [0.994, 1.021] | 0.78 |
| D3-bin | 5,000 | E-def | 300 | 1 (reference) | 0.96 |
| D3-bin | 5,000 | P-fix | 300 | 1.000 [1.000, 1.000] | 0.96 |
| D4-bin | 200 | E-def | 300 | 1 (reference) | 0.36 |
| D4-bin | 200 | EarthClassifier | 300 | 0.365* [0.336, 0.396] | 1.10 |
| D4-bin | 200 | GLMEarth | 20 | 0.404* [0.277, 0.587] | 0.98 |
| D4-bin | 200 | P-fix | 300 | 1.011 [0.992, 1.031] | 0.36 |
| D4-bin | 1,000 | E-def | 300 | 1 (reference) | 0.78 |
| D4-bin | 1,000 | P-fix | 300 | 1.009 [0.994, 1.023] | 0.78 |
| D4-bin | 5,000 | E-def | 300 | 1 (reference) | 0.95 |
| D4-bin | 5,000 | P-fix | 300 | 1.005 [0.997, 1.013] | 0.95 |

*Table 6.1. Ratio of excess log loss to E-def with ±3 Monte Carlo standard errors (`*` resolved at 3 standard errors) and the median calibration slope on the test set (1 is well calibrated). Supports the parity claim for binary outcomes and measures the effect of the legacy penalized refit.*

pymars 2.0 matches earth on all six binary cells (Figure 3.1; after #185 the six ratios are 1.000 to 1.001, Figure 3.2), and its calibration matches earth's at every n. Both are overconfident at 200 cases (slopes 0.36 to 0.44) and close to calibrated at 5,000 cases (0.95 and 0.96), as expected of an unpenalized logistic refit of a selected basis. At 200 cases the legacy EarthClassifier has 0.37 to 0.52 of earth's excess log loss, and the legacy classifiers have slopes near 1. Their penalized refit, which shrinks the coefficients, is a likely cause; no ablation here tests it, and the legacy forward pass differs too. Both programs follow their documented method here. pymars 2.0 offers the penalty as `glm_alpha`. The binary part of the parity claim is about earth, and it holds.

**Note on the default of `glm_alpha` (issue #175).** The simulation and the conformance fixtures used `glm_alpha=0`, earth's unpenalized refit and the default at the freeze. The default is now 1, the strength of scikit-learn's `LogisticRegression` at `C=1`, because at 0 the refit warns on scikit-learn's breast_cancer, iris and wine data. The parity claim for binary outcomes holds at `glm_alpha=0`, the earth-compatible setting. No number in this report changed.

## A known limit: skewed covariates

The parity claim is about covariates in [0, 1], the only kind the grid has. This section tests one kind of skewed covariates, because the new pair rule of #185 is not earth's rule there. It has its own short ADEMP.

**Aim.** With skewed covariates, the excess risk of pymars with the pair rule of #185 is less than 5 percent above that of pymars with the rule before it, the margin of the parity claim, and above earth's. The review of #185 raised the question: in a first look at lognormal covariates, D4 at 200 cases gave a ratio of new to old of 1.055 [1.013, 1.099] over 300 repetitions, the worst of 24 cells, and a second look with fresh seeds gave 1.020 [0.994, 1.047] over 600. The two looks were small, so this study asks again with enough repetitions to resolve a loss of 1 percent. Subclaims, one for each DGP of the grid with signal: D1 (linear), D2 and D3 (additive), D4 (an interaction and irrelevant covariates, also at 1,000 cases), D5 (a hinge interaction), D7 (many covariates) and D8 (correlated covariates). A possible mechanism, which this study does not test: for a covariate with a long tail the variance comes from the tail, so the threshold 0.01·Π_{v∈V}(12σ_v²) is high next to the bulk of the data.

**Data-generating processes.** The covariates that the learners see are Z_j = exp(Φ⁻¹(U_j)), with lognormal margins, where U is the covariate vector of the grid's DGP, drawn in the same way (so D8 keeps its Gaussian copula). The truth stays f(U), so the truth as a function of Z is a monotone transform of each covariate, and the noise σ is the grid's, at the low noise level (population R² 0.8). The grid's DGPs are reused so that the truth is known and the DGPs have a reason of their own (see the table of DGPs above); lognormal margins are the plain example of a long right tail, and the monotone map keeps every DGP's role. D6 has no signal and is not part of this study. The sample size is 200 cases for every DGP, and 1,000 cases for D4 as well. The seeds come from the cell name with `skew` after the DGP (`D4skew_n00200_lo`), so the datasets differ from those of the grid, and all arms see the same data in a repetition.

**Estimand and methods.** The estimand is the excess risk of each fit on a fresh test draw of 10,000 cases of the same covariates, as in the grid. The arms: P-old, pymars at `fdaca4a` (the `main` before #185, with the old threshold 0.01·Π_{v∈V}σ_v²); P-new, pymars at `ac67b38` (the head of #185 when the run started, with the same behavior as the merge `ab923ca`); E-def, earth at its defaults with `degree = 2`. Both pymars arms are `EarthRegressor(max_degree=2)` at its defaults, and between the two commits the only change in `pymars/` is LA-7's threshold.

**Performance measures.** As in the grid: the paired log ratio, the ratio of geometric means with ±3 Monte Carlo standard errors, for new over old (the aim), new over earth and old over earth, and the share of repetitions with equal fits (agreement to 1e-9). A ratio is resolved when the interval excludes 1 (`*`). The decision rule was fixed before the runs: no material loss when every upper bound of new over old is below 1.05, the margin of the parity claim.

**Design.** A screen of all seven DGPs at 200 cases and 1,000 repetitions, with D4 at 1,000 cases and 1,000 repetitions as a pilot. Then a confirmation of D4 at 200 and at 1,000 cases with 7,000 repetitions each, a size fixed in advance from the spread of the log ratio, s_g (the screen's s_g of 0.224 at 200 cases asks for about 4,750 repetitions for a half-width of 0.01 on the ratio, and the pilot's 0.088 at 1,000 cases for about 700). The screen's interval for D8, [0.985, 1.081], did not decide the margin, so D8 at 200 cases was extended to 23,000 repetitions (its s_g of 0.49 asks for about 22,800). **That extension was not part of the fixed design**; it was added after the screen, and it is a look that followed an undecided interval (the caveat of [Repetitions](#repetitions) applies to it). The seeds nest, so the screen's repetitions are the first of the extended ones. In all, 126,000 fits (3 arms and 42,000 repetitions), none failed.

**Evidence.**

| DGP | n | reps | new / old | new / E-def | old / E-def | equal fits, new vs old | equal fits to E-def, old / new |
|---|---|---|---|---|---|---|---|
| D1 linear | 200 | 1,000 | 0.996 [0.987, 1.004] | 0.995 [0.985, 1.004] | 0.999 [0.995, 1.002] | 97% | 100% / 97% |
| D2 additive hinges | 200 | 1,000 | 0.995 [0.973, 1.017] | 0.995 [0.973, 1.017] | 1.000 [1.000, 1.000] | 90% | 100% / 90% |
| D3 additive smooth | 200 | 1,000 | 0.996 [0.985, 1.008] | 0.996 [0.985, 1.008] | 1.000 [1.000, 1.000] | 90% | 100% / 90% |
| D4 Friedman #1 | 200 | 7,000 | 1.019* [1.010, 1.027] | 1.019* [1.010, 1.027] | 1.000 [0.999, 1.001] | 62% | 100% / 62% |
| D4 Friedman #1 | 1,000 | 7,000 | 1.007* [1.003, 1.010] | 1.007* [1.003, 1.010] | 1.000 [1.000, 1.000] | 80% | 100% / 80% |
| D5 hinge interaction | 200 | 1,000 | 0.970 [0.939, 1.001] | 0.969 [0.939, 1.001] | 1.000 [0.997, 1.002] | 92% | 100% / 92% |
| D7 many covariates | 200 | 1,000 | 0.996 [0.973, 1.019] | 0.990 [0.967, 1.014] | 0.995 [0.987, 1.002] | 65% | 94% / 62% |
| D8 correlated | 200 | 23,000 | 1.009 [1.000, 1.018] | 1.009 [1.000, 1.018] | 1.000 [1.000, 1.000] | 45% | 100% / 45% |

*Table 7.1. Excess risk with lognormal covariates and low noise: the new pair rule (P-new, #185) over the old rule (P-old), over earth (E-def), and the old rule over earth, as ratios of geometric means with ±3 Monte Carlo standard errors (`*`: at least 3 standard errors from 1); the share of repetitions where P-new and P-old agree to 1e-9; and the share where each of P-old and P-new agrees with earth. Supports the aim: whether the new rule costs excess risk on skewed covariates, and by how much. D8's 23,000 repetitions are an extension that the fixed design did not include.*

On D4 the new rule is resolved worse than the old one: 1.019 [1.010, 1.027] at 200 cases (1.9 percent) and 1.007 [1.003, 1.010] at 1,000 cases (0.7 percent). The old rule gives earth's excess risk in these cells (old over earth 1.000), and the new rule has 62 and 80 percent equal fits, so the loss of the new rule against the old one is also its loss against earth. On D8 the ratio is 1.009 [1.000, 1.018], not resolved, after the extension. The other five DGPs are not resolved either, with point estimates from 0.970 to 0.996, so the study finds no loss on them. The largest upper bound of the eight cells is 1.027 (D4 at 200 cases). All eight are below 1.05, so the decision rule gives no material loss. In the screen, the two cells with the largest points, D4 at 200 cases (1.026) and D8 (1.032), moved towards 1 with more repetitions (1.019 and 1.009), as the worst cells of a screen do.

**Statement.** With lognormal covariates the pair rule of #185 costs excess risk against the old rule on D4: 1.9 percent at 200 cases and 0.7 percent at 1,000 cases, resolved. On D8 the cost is 0.9 percent and not resolved, after an extension that the fixed design did not include, and on the other five DGPs it is not resolved, with points from 0.970 to 0.996. All upper bounds are inside the 5 percent margin. The old rule gives earth's fits on these data and the new rule does not, so this limit is a difference from earth that the new rule creates, and the project owner accepted the rule with it. The study covers one family of skewed covariates (lognormal margins), low noise, 200 cases (D4 also 1,000), and the DGPs of the grid. It does not show the effect of heavier tails or other skews, of high noise, or of the mechanism above.

## Appendix

### A.1 Every parity cell at the freeze commit

| DGP | noise | n | reps | P-fix / E-def | equal fits | decision |
|---|---|---|---|---|---|---|
| D1 linear | hi | 200 | 3,000 | 1.019 [1.007, 1.032] | 86% | equivalent |
| D1 linear | hi | 1,000 | 900 | 1.028 [1.010, 1.047] | 81% | equivalent |
| D1 linear | hi | 5,000 | 600 | 1.038 [1.019, 1.056] | 81% | not decided |
| D1 linear | lo | 200 | 1,300 | 1.016 [0.998, 1.033] | 87% | equivalent |
| D1 linear | lo | 1,000 | 750 | 1.023 [1.005, 1.042] | 82% | equivalent |
| D1 linear | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 100% | equivalent |
| D2 additive hinges | hi | 200 | 300 | 1.010 [0.990, 1.031] | 92% | equivalent |
| D2 additive hinges | hi | 1,000 | 300 | 1.002 [0.993, 1.011] | 93% | equivalent |
| D2 additive hinges | hi | 5,000 | 300 | 1.002 [0.998, 1.006] | 98% | equivalent |
| D2 additive hinges | lo | 200 | 300 | 0.999 [0.992, 1.005] | 97% | equivalent |
| D2 additive hinges | lo | 1,000 | 300 | 1.002 [0.998, 1.007] | 97% | equivalent |
| D2 additive hinges | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 100% | equivalent |
| D3 additive smooth | hi | 200 | 300 | 1.008 [0.993, 1.023] | 89% | equivalent |
| D3 additive smooth | hi | 1,000 | 300 | 1.001 [0.994, 1.009] | 97% | equivalent |
| D3 additive smooth | hi | 5,000 | 300 | 1.000 [0.998, 1.001] | 99% | equivalent |
| D3 additive smooth | lo | 200 | 300 | 1.008 [0.994, 1.022] | 97% | equivalent |
| D3 additive smooth | lo | 1,000 | 300 | 0.999 [0.998, 1.001] | 99% | equivalent |
| D3 additive smooth | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 100% | equivalent |
| D3-bin | na | 200 | 1,300 | 1.022 [1.008, 1.037] | 61% | equivalent |
| D3-bin | na | 1,000 | 300 | 1.008 [0.994, 1.021] | 46% | equivalent |
| D3-bin | na | 5,000 | 300 | 1.000 [1.000, 1.000] | 56% | equivalent |
| D4 Friedman #1 | hi | 200 | 300 | 1.020 [0.997, 1.045] | 90% | equivalent |
| D4 Friedman #1 | hi | 1,000 | 300 | 1.003 [0.997, 1.009] | 94% | equivalent |
| D4 Friedman #1 | hi | 5,000 | 300 | 1.002 [0.996, 1.008] | 95% | equivalent |
| D4 Friedman #1 | lo | 200 | 300 | 0.999 [0.991, 1.007] | 95% | equivalent |
| D4 Friedman #1 | lo | 1,000 | 300 | 1.000 [0.992, 1.008] | 94% | equivalent |
| D4 Friedman #1 | lo | 5,000 | 300 | 1.000 [0.997, 1.002] | 91% | equivalent |
| D4-bin | na | 200 | 300 | 1.011 [0.992, 1.031] | 63% | equivalent |
| D4-bin | na | 1,000 | 300 | 1.009 [0.994, 1.023] | 48% | equivalent |
| D4-bin | na | 5,000 | 300 | 1.005 [0.997, 1.013] | 59% | equivalent |
| D5 hinge interaction | hi | 200 | 1,300 | 1.024 [1.004, 1.045] | 83% | equivalent |
| D5 hinge interaction | hi | 1,000 | 600 | 1.012 [0.998, 1.027] | 87% | equivalent |
| D5 hinge interaction | hi | 5,000 | 600 | 1.020 [1.007, 1.034] | 89% | equivalent |
| D5 hinge interaction | lo | 200 | 900 | 1.025 [1.002, 1.049] | 87% | equivalent |
| D5 hinge interaction | lo | 1,000 | 600 | 1.015 [0.998, 1.033] | 89% | equivalent |
| D5 hinge interaction | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 100% | equivalent |
| D6 pure noise | na | 200 | 7,500 | 1.076 [1.051, 1.101] | 85% | not equivalent |
| D6 pure noise | na | 1,000 | 3,800 | 1.030 [1.007, 1.053] | 78% | not decided |
| D6 pure noise | na | 5,000 | 12,400 | 1.040 [1.031, 1.050] | 77% | equivalent |
| D7 many covariates | hi | 200 | 300 | 1.013 [0.984, 1.042] | 72% | equivalent |
| D7 many covariates | hi | 1,000 | 300 | 1.005 [0.994, 1.016] | 86% | equivalent |
| D7 many covariates | hi | 5,000 | 300 | 1.005 [0.999, 1.012] | 87% | equivalent |
| D7 many covariates | lo | 200 | 300 | 1.004 [0.987, 1.022] | 80% | equivalent |
| D7 many covariates | lo | 1,000 | 300 | 1.001 [0.997, 1.006] | 91% | equivalent |
| D7 many covariates | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 100% | equivalent |
| D8 correlated | hi | 200 | 600 | 1.009 [0.986, 1.032] | 84% | equivalent |
| D8 correlated | hi | 1,000 | 300 | 1.005 [0.994, 1.017] | 87% | equivalent |
| D8 correlated | hi | 5,000 | 300 | 1.011 [1.000, 1.022] | 93% | equivalent |
| D8 correlated | lo | 200 | 600 | 1.022 [1.002, 1.042] | 87% | equivalent |
| D8 correlated | lo | 1,000 | 300 | 0.999 [0.978, 1.020] | 84% | equivalent |
| D8 correlated | lo | 5,000 | 300 | 0.999 [0.997, 1.001] | 100% | equivalent |

*Table A.1. P-fix over E-def in every cell, with ±3 Monte Carlo standard errors, the number of repetitions, the share of repetitions with equal excess risk (to 1e-9), and the equivalence decision at the margin 1.05. Supports the parity claim in the study at the freeze (the original study). The same numbers are in `validation/sims/results/full/parity.csv`. The cells where every fit is equal have intervals of width 0. D6 at 5,000 cases rounds to an upper bound of 1.050; its exact upper bound is 1.0495, 0.0004 inside the margin on the log scale, after an extension.*

### A.1b Every parity cell after #185

| DGP | noise | n | reps | freeze: P-fix / E-def | after #185: P-fix / E-def | decision | equal fits to E-def |
|---|---|---|---|---|---|---|---|
| D1 linear | hi | 200 | 3,000 | 1.019 [1.007, 1.032] | 1.000 [0.998, 1.001] | equivalent | 86% to 100% |
| D1 linear | hi | 1,000 | 900 | 1.028 [1.010, 1.047] | 1.002 [0.998, 1.005] | equivalent | 81% to 96% |
| D1 linear | hi | 5,000 | 600 | 1.038 [1.019, 1.056] | 1.000 [1.000, 1.000] | not decided to equivalent | 81% to 100% |
| D1 linear | lo | 200 | 1,300 | 1.016 [0.998, 1.033] | 1.002 [0.998, 1.005] | equivalent | 87% to 100% |
| D1 linear | lo | 1,000 | 750 | 1.023 [1.005, 1.042] | 0.999 [0.994, 1.003] | equivalent | 82% to 96% |
| D1 linear | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |
| D2 additive hinges | hi | 200 | 300 | 1.010 [0.990, 1.031] | 1.000 [1.000, 1.000] | equivalent | 92% to 100% |
| D2 additive hinges | hi | 1,000 | 300 | 1.002 [0.993, 1.011] | 1.000 [1.000, 1.000] | equivalent | 93% to 99% |
| D2 additive hinges | hi | 5,000 | 300 | 1.002 [0.998, 1.006] | 1.000 [1.000, 1.000] | equivalent | 98% to 100% |
| D2 additive hinges | lo | 200 | 300 | 0.999 [0.992, 1.005] | 1.000 [1.000, 1.000] | equivalent | 97% to 100% |
| D2 additive hinges | lo | 1,000 | 300 | 1.002 [0.998, 1.007] | 1.001 [0.999, 1.004] | equivalent | 97% to 99% |
| D2 additive hinges | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |
| D3 additive smooth | hi | 200 | 300 | 1.008 [0.993, 1.023] | 1.000 [1.000, 1.000] | equivalent | 89% to 100% |
| D3 additive smooth | hi | 1,000 | 300 | 1.001 [0.994, 1.009] | 1.000 [1.000, 1.000] | equivalent | 97% to 100% |
| D3 additive smooth | hi | 5,000 | 300 | 1.000 [0.998, 1.001] | 1.000 [1.000, 1.000] | equivalent | 99% to 100% |
| D3 additive smooth | lo | 200 | 300 | 1.008 [0.994, 1.022] | 1.001 [0.993, 1.009] | equivalent | 97% to 99% |
| D3 additive smooth | lo | 1,000 | 300 | 0.999 [0.998, 1.001] | 1.000 [1.000, 1.000] | equivalent | 99% to 100% |
| D3 additive smooth | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |
| D3-bin | na | 200 | 1,300 | 1.022 [1.008, 1.037] | 1.001 [0.997, 1.005] | equivalent | 61% to 66% |
| D3-bin | na | 1,000 | 300 | 1.008 [0.994, 1.021] | 1.000 [1.000, 1.001] | equivalent | 46% to 46% |
| D3-bin | na | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 56% to 56% |
| D4 Friedman #1 | hi | 200 | 300 | 1.020 [0.997, 1.045] | 0.999 [0.998, 1.001] | equivalent | 90% to 99% |
| D4 Friedman #1 | hi | 1,000 | 300 | 1.003 [0.997, 1.009] | 1.000 [1.000, 1.000] | equivalent | 94% to 98% |
| D4 Friedman #1 | hi | 5,000 | 300 | 1.002 [0.996, 1.008] | 1.000 [1.000, 1.000] | equivalent | 95% to 100% |
| D4 Friedman #1 | lo | 200 | 300 | 0.999 [0.991, 1.007] | 1.000 [1.000, 1.000] | equivalent | 95% to 100% |
| D4 Friedman #1 | lo | 1,000 | 300 | 1.000 [0.992, 1.008] | 1.000 [1.000, 1.000] | equivalent | 94% to 99% |
| D4 Friedman #1 | lo | 5,000 | 300 | 1.000 [0.997, 1.002] | 1.000 [1.000, 1.000] | equivalent | 91% to 100% |
| D4-bin | na | 200 | 300 | 1.011 [0.992, 1.031] | 1.000 [1.000, 1.001] | equivalent | 63% to 66% |
| D4-bin | na | 1,000 | 300 | 1.009 [0.994, 1.023] | 1.000 [1.000, 1.000] | equivalent | 48% to 49% |
| D4-bin | na | 5,000 | 300 | 1.005 [0.997, 1.013] | 1.000 [1.000, 1.000] | equivalent | 59% to 61% |
| D5 hinge interaction | hi | 200 | 1,300 | 1.024 [1.004, 1.045] | 0.999 [0.998, 1.001] | equivalent | 83% to 100% |
| D5 hinge interaction | hi | 1,000 | 600 | 1.012 [0.998, 1.027] | 1.000 [0.998, 1.001] | equivalent | 87% to 98% |
| D5 hinge interaction | hi | 5,000 | 600 | 1.020 [1.007, 1.034] | 1.000 [1.000, 1.000] | equivalent | 89% to 100% |
| D5 hinge interaction | lo | 200 | 900 | 1.025 [1.002, 1.049] | 1.001 [0.993, 1.009] | equivalent | 87% to 99% |
| D5 hinge interaction | lo | 1,000 | 600 | 1.015 [0.998, 1.033] | 1.000 [0.999, 1.000] | equivalent | 89% to 99% |
| D5 hinge interaction | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |
| D6 pure noise | na | 200 | 7,500 | 1.076 [1.051, 1.101] | 0.997 [0.993, 1.001] | not equivalent to equivalent | 85% to 99% |
| D6 pure noise | na | 1,000 | 3,800 | 1.030 [1.007, 1.053] | 1.001 [0.999, 1.003] | not decided to equivalent | 78% to 97% |
| D6 pure noise | na | 5,000 | 12,400 | 1.040 [1.031, 1.050] | 1.000 [1.000, 1.000] | equivalent | 77% to 100% |
| D7 many covariates | hi | 200 | 300 | 1.013 [0.984, 1.042] | 1.000 [0.998, 1.001] | equivalent | 72% to 99% |
| D7 many covariates | hi | 1,000 | 300 | 1.005 [0.994, 1.016] | 1.000 [1.000, 1.000] | equivalent | 86% to 99% |
| D7 many covariates | hi | 5,000 | 300 | 1.005 [0.999, 1.012] | 1.000 [1.000, 1.000] | equivalent | 87% to 100% |
| D7 many covariates | lo | 200 | 300 | 1.004 [0.987, 1.022] | 0.998 [0.995, 1.002] | equivalent | 80% to 98% |
| D7 many covariates | lo | 1,000 | 300 | 1.001 [0.997, 1.006] | 1.000 [1.000, 1.000] | equivalent | 91% to 100% |
| D7 many covariates | lo | 5,000 | 300 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |
| D8 correlated | hi | 200 | 600 | 1.009 [0.986, 1.032] | 1.000 [0.996, 1.004] | equivalent | 84% to 99% |
| D8 correlated | hi | 1,000 | 300 | 1.005 [0.994, 1.017] | 0.999 [0.995, 1.002] | equivalent | 87% to 96% |
| D8 correlated | hi | 5,000 | 300 | 1.011 [1.000, 1.022] | 1.000 [1.000, 1.000] | equivalent | 93% to 100% |
| D8 correlated | lo | 200 | 600 | 1.022 [1.002, 1.042] | 1.002 [0.998, 1.007] | equivalent | 87% to 98% |
| D8 correlated | lo | 1,000 | 300 | 0.999 [0.978, 1.020] | 0.998 [0.993, 1.004] | equivalent | 84% to 99% |
| D8 correlated | lo | 5,000 | 300 | 0.999 [0.997, 1.001] | 1.000 [1.000, 1.000] | equivalent | 100% to 100% |

*Table A.1b. The rerun after #185 beside the study of Table A.1, in every cell, on the same repetitions: P-fix over E-def at the freeze and P-fix at `286d994` over the same E-def fits, each with ±3 Monte Carlo standard errors; the equivalence decision at the margin 1.05 ("a to b" where it changed); the share of repetitions with equal excess risk to 1e-9 for the freeze and after #185. Supports the parity claim after the fix. The same numbers, with the paired changes and the acceptance test, are in `validation/sims/report/tables/parity_rerun.csv` and `parity_rerun_counts.json`. In the cells where every fit equals earth's the interval has width 0.*

### A.2 The gap at high noise, 200 cases

| DGP | P-cur / E-def | P-ear / E-def | E-pym / E-def | P-fix / E-def |
|---|---|---|---|---|
| D1 linear | 0.843* [0.757, 0.939] | 1.037 [0.935, 1.150] | 0.739* [0.622, 0.878] | 1.018 [0.965, 1.073] |
| D2 additive hinges | 1.329* [1.194, 1.480] | 1.493* [1.349, 1.652] | 0.800 [0.614, 1.041] | 1.010 [0.990, 1.031] |
| D3 additive smooth | 1.406* [1.277, 1.549] | 1.389* [1.276, 1.513] | 0.841* [0.742, 0.955] | 1.008 [0.993, 1.023] |
| D4 Friedman #1 | 1.023 [0.938, 1.116] | 1.077* [1.009, 1.149] | 0.998 [0.892, 1.117] | 1.020 [0.997, 1.045] |
| D5 hinge interaction | 1.062 [0.912, 1.236] | 1.207* [1.080, 1.350] | 0.816* [0.690, 0.964] | 1.036 [0.995, 1.079] |
| D7 many covariates | not run (too slow) | not run (too slow) | 0.651* [0.546, 0.776] | 1.013 [0.984, 1.042] |
| D8 correlated | 0.913 [0.820, 1.016] | 1.004 [0.930, 1.084] | 0.881 [0.754, 1.029] | 1.001 [0.974, 1.029] |

*Table A.2. As Table 2.1 at high noise (population R² 0.3), 300 repetitions. Supports the high-noise condition of the gap claim.*

### A.3 The gap at 1,000 cases, low noise

| DGP | P-cur / E-def | P-ear / E-def | E-pym / E-def | P-fix / E-def |
|---|---|---|---|---|
| D3 additive smooth | 1.376* [1.293, 1.464] | 1.276* [1.209, 1.346] | 0.954 [0.847, 1.075] | 0.999 [0.997, 1.002] |
| D4 Friedman #1 | 1.281* [1.202, 1.366] | 1.109* [1.053, 1.168] | 1.067* [1.012, 1.126] | 0.999 [0.988, 1.009] |
| D5 hinge interaction | 0.757* [0.672, 0.853] | 0.973 [0.864, 1.095] | 0.694* [0.556, 0.866] | 1.009 [0.977, 1.042] |
| D8 correlated | 1.319* [1.200, 1.449] | 1.128* [1.045, 1.217] | 1.162* [1.014, 1.333] | 1.002 [0.975, 1.030] |

*Table A.3. As Table 2.1 at 1,000 cases and low noise, 200 repetitions; the four cells that the plan runs for the legacy code at this size. Supports the claim that the gap persists at a larger n.*

### A.4 Settings of the legacy code

| DGP | noise | n | P-ear / P-cur |
|---|---|---|---|
| D1 linear | lo | 200 | 1.263* [1.159, 1.376] |
| D2 additive hinges | lo | 200 | 1.023 [0.954, 1.098] |
| D3 additive smooth | lo | 200 | 0.933 [0.869, 1.001] |
| D4 Friedman #1 | lo | 200 | 0.952 [0.889, 1.019] |
| D5 hinge interaction | lo | 200 | 0.912 [0.784, 1.060] |
| D6 pure noise | na | 200 | 1.268* [1.141, 1.410] |
| D8 correlated | lo | 200 | 0.925* [0.878, 0.974] |
| D1 linear | hi | 200 | 1.230* [1.140, 1.326] |
| D2 additive hinges | hi | 200 | 1.123* [1.046, 1.206] |
| D3 additive smooth | hi | 200 | 0.988 [0.921, 1.060] |
| D4 Friedman #1 | hi | 200 | 1.052 [0.974, 1.137] |
| D5 hinge interaction | hi | 200 | 1.137 [1.000, 1.292] |
| D8 correlated | hi | 200 | 1.100 [0.997, 1.213] |
| D3 additive smooth | lo | 1,000 | 0.927* [0.878, 0.979] |
| D4 Friedman #1 | lo | 1,000 | 0.866* [0.814, 0.920] |
| D5 hinge interaction | lo | 1,000 | 1.284* [1.178, 1.400] |
| D8 correlated | lo | 1,000 | 0.855* [0.793, 0.922] |

*Table A.4. The legacy code with earth's settings over the legacy code at its defaults, paired by repetition. Tests the second half of the gap claim, that earth's settings shrink the gap: they shrink it by 7 to 15 percent in the 5 resolved cells below 1 and enlarge it in the 5 resolved cells above 1.*

### A.5 Binary outcomes: excess Brier score and clipped probabilities

| DGP | n | arm | excess Brier score / E-def | clipped probabilities per fit, mean (share of fits with any) |
|---|---|---|---|---|
| D3-bin | 200 | E-def | 1 (reference) | 9.6 (15%) |
| D3-bin | 200 | EarthClassifier | 0.782* [0.720, 0.849] | 0.0 (0%) |
| D3-bin | 200 | GLMEarth | 1.059 [0.643, 1.744] | 0.0 (0%) |
| D3-bin | 200 | P-fix | 1.011* [1.002, 1.020] | 12.1 (17%) |
| D3-bin | 1,000 | E-def | 1 (reference) | 1.1 (5%) |
| D3-bin | 1,000 | P-fix | 1.002 [0.995, 1.010] | 1.3 (5%) |
| D3-bin | 5,000 | E-def | 1 (reference) | 0.0 (0%) |
| D3-bin | 5,000 | P-fix | 1.000* [1.000, 1.000] | 0.0 (0%) |
| D4-bin | 200 | E-def | 1 (reference) | 10.3 (15%) |
| D4-bin | 200 | EarthClassifier | 0.525* [0.495, 0.558] | 0.0 (0%) |
| D4-bin | 200 | GLMEarth | 0.595* [0.454, 0.779] | 0.0 (0%) |
| D4-bin | 200 | P-fix | 1.003 [0.990, 1.015] | 11.6 (17%) |
| D4-bin | 1,000 | E-def | 1 (reference) | 0.5 (2%) |
| D4-bin | 1,000 | P-fix | 1.005 [0.994, 1.017] | 0.5 (2%) |
| D4-bin | 5,000 | E-def | 1 (reference) | 0.0 (0%) |
| D4-bin | 5,000 | P-fix | 1.004 [0.998, 1.010] | 0.0 (0%) |

*Table A.5. The excess Brier score E[(μ̂(X) − μ(X))²] as a ratio to E-def with ±3 Monte Carlo standard errors (`*` resolved), and the number of fitted probabilities clipped to [1e-6, 1 − 1e-6] per fit (out of 10,000 test points), with the share of fits that clip any. The plan's "Performance measures" asks for both. The excess Brier score tells the same story as the excess log loss in Table 6.1. Earth and pymars 2.0 clip a few probabilities per fit at 200 cases (in 15 to 17 percent of fits) and almost none at 5,000; the clipped values enter the excess log loss at the clipping bound. The clipped counts come from the per-repetition files of the runs (`tables/clipped.csv`), since `results.csv` does not hold them.*

### A.6 How to rebuild

```sh
. dev/env.sh
uv run --frozen --group validation python validation/sims/report/make_report.py \
    --bench-run <a run folder of validation/bench/run.py> \
    --run-dirs <the T21 run folder> <the T04 run folder>
NUMBA_NUM_THREADS=1 nice -n 15 uv run --frozen --group validation python \
    validation/sims/report/warm_times.py --out validation/sims/report/warm_times.csv
NUMBA_NUM_THREADS=1 nice -n 15 uv run --frozen --group validation python \
    validation/sims/report/warm_times.py --numpy --out validation/sims/report/warm_times_numpy.csv
```

Without `--bench-run` the script uses the stored `validation/sims/report/bench_times.csv`, and without `--run-dirs` the stored `tables/clipped.csv`. The tables of the rerun and of the skewed-covariate study need only the committed results (`validation/sims/results/t173-grid/results.csv`, `validation/sims/results/skew/results.csv` and the two earlier files). The run folders of this report were `.worktrees/runner-t21/validation/runs/full` and `.worktrees/runner-legacy/validation/runs/legacy_full` for the T21 and T04 runs, `.worktrees/runner-t173/validation/runs/t173-grid` for the rerun, and `.worktrees/runner-t173-skew-old/validation/runs/skew` and `.worktrees/runner-t173-skew-new/validation/runs/skew` for the skewed-covariate study (all ignored by git). The results folders collect them with `validation/sims/results/t173-grid/analyze.py <run folder>` and `validation/sims/results/skew/analyze.py <old run folder> <new run folder>`, and `validation/sims/skew_sims.py` made the skewed-covariate fits. The simulation runs are described in `validation/sims/README.md`, `validation/sims/results/legacy/README.md` and `validation/sims/results/full/summary.md`, the docstrings of the two `analyze.py` files and `driver_*.sh`, and the benchmark in `validation/bench/README.md`.
