# Pilot check: legacy full run (T04 step 4, issue #6)

This note sizes the four gap-claim pilot cells, D4 and D5 at 200 cases and
both noise levels (`D4_n00200_lo`, `D4_n00200_hi`, `D5_n00200_lo`,
`D5_n00200_hi`), against `VALIDATION_PLAN.md`'s "Goals" and "Pilot and
number of repetitions". For each of the three difference contrasts the gap
claim uses (P-cur/E-def, P-ear/E-def, E-pym/E-def), the table below gives
the pilot mean and standard deviation of the paired log ratio of excess
risk, the pilot's own z statistic, and the repetitions the full run needs at
5 Monte Carlo standard errors (`n_sim`) and at the 1.96-lower-bounded gap
(`n_sim_safe`), both from `pilot_check.difference_n_sim` unchanged.

The full run (`validation/runs/legacy_full` in the runner worktree) did not
wait on this check. The executor started it at the plan's caps, 300
repetitions at 200 cases, right after the check run passed, instead of
holding it for a separate pilot. The statistics below use only repetitions 0
to 99 of that run. That subset is the pilot the plan names: each dataset's
seed comes from its cell name and repetition number, never from its
position in a loop, so repetition k is the same dataset whether it is read
as a 100-repetition pilot or pulled out of the finished 300-repetition run.
All 100 repetitions succeeded for every cell and arm here. No fit recorded
an error, so every row has n_pilot = 100 and no repetition was excluded.

## Pilot statistics

| Cell | Contrast | ḡ_p | s_p | z | n_sim | n_sim_safe | Cap (300) sufficient |
|---|---|---:|---:|---:|---:|---:|---|
| D4_n00200_lo | P-cur/E-def | 0.2733 | 0.5083 | 5.38 | 86.5 | 214.1 | yes |
| D4_n00200_lo | P-ear/E-def | 0.2026 | 0.3813 | 5.31 | 88.6 | 222.3 | yes |
| D4_n00200_lo | E-pym/E-def | 0.1380 | 0.8618 | 1.60 | 975.6 | inf | no |
| D4_n00200_hi | P-cur/E-def | 0.0233 | 0.3905 | 0.60 | 7041.1 | inf | no |
| D4_n00200_hi | P-ear/E-def | 0.1380 | 0.3454 | 4.00 | 156.6 | 603.3 | no |
| D4_n00200_hi | E-pym/E-def | 0.0392 | 0.6754 | 0.58 | 7404.8 | inf | no |
| D5_n00200_lo | P-cur/E-def | 0.3623 | 0.7625 | 4.75 | 110.7 | 320.8 | no |
| D5_n00200_lo | P-ear/E-def | 0.2947 | 0.5596 | 5.27 | 90.1 | 228.5 | yes |
| D5_n00200_lo | E-pym/E-def | -0.4514 | 0.7857 | -5.75 | 75.7 | 174.5 | yes |
| D5_n00200_hi | P-cur/E-def | 0.0443 | 0.6007 | 0.74 | 4594.6 | inf | no |
| D5_n00200_hi | P-ear/E-def | 0.2224 | 0.5863 | 3.79 | 173.8 | 744.1 | no |
| D5_n00200_hi | E-pym/E-def | -0.1863 | 0.7835 | -2.38 | 442.2 | 14323.6 | no |

`_lo`/`_hi` is the low/high noise level (population R² 0.8 / 0.3).
ḡ_p and s_p are the pilot mean and standard deviation (ddof=1) of the
per-repetition log ratio `metrics.log_ratio`; z = ḡ_p / (s_p / sqrt(100)).
"Cap sufficient" marks n_sim_safe <= 300, the plan's repetition cap at 200
cases.

## Cells where the pilot cannot size a run

Four of the twelve (cell, contrast) pairs have |z| < 2, so the pilot cannot
tell the gap from 0 and n_sim_safe is infinite:

- D4_n00200_lo, E-pym/E-def (z = 1.60)
- D4_n00200_hi, P-cur/E-def (z = 0.60)
- D4_n00200_hi, E-pym/E-def (z = 0.58)
- D5_n00200_hi, P-cur/E-def (z = 0.74)

## Whether the caps suffice

n_sim_safe stays at or under 300 for 4 of the 12 pairs: D4_n00200_lo for
P-cur/E-def and P-ear/E-def, and D5_n00200_lo for P-ear/E-def and
E-pym/E-def. The other 8 do not, for two different reasons. Four are the
|z| < 2 pairs above, where no n_sim is trustworthy. The other four clear
z = 2 but still need more than 300 repetitions: D4_n00200_hi P-ear/E-def
(603.3), D5_n00200_lo P-cur/E-def (320.8, just over the cap), D5_n00200_hi
P-ear/E-def (744.1) and D5_n00200_hi E-pym/E-def (14323.6).

This note only sizes the pilot against the plan's caps. It draws no
conclusion about the gap, parity or sparsity claims themselves; T22's report
interprets these numbers alongside the rest of the statistical performance
study.
