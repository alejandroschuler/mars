# Skewed covariates: PR #185's pair rule on lognormal data (issue #173)

The question comes from the adversarial review of #185. Every grid cell has Unif[0, 1] margins, so does the new threshold cost excess risk when the covariates have a heavy tail? The report's section "A known limit: skewed covariates" gives the ADEMP.

- Covariates: Z_j = exp(Φ⁻¹(U_j)), with U the grid's covariates (`dgps.generate`, the same draws; D8 keeps its copula). The truth is f(U). σ and the noise level are the grid's (low). The cell names are like `D4skew_n00200_lo`, so the seeds differ from the grid's, and they are the same in every worktree. `skew_sims.py` makes the data and runs the arms.
- Arms:
  - P-old is pymars at fdaca4a (main before #185), in the runner `.worktrees/runner-t173-skew-old`.
  - P-new is pymars at ac67b38 (the head of #185 when the run started, with the same behavior as ab923ca), and E-def is earth 5.3.4 at degree 2, both in the runner `.worktrees/runner-t173-skew-new`.
  - Between fdaca4a and ac67b38, the only change in `pymars/` is LA-7's threshold.
  - In the per-repetition files, P-old and P-new are both called `P-fix`.
  - Each runner used 4 workers, `nice -n 15`, no cache and `--resume`.
  - The three arms see the same data in every repetition: `analyze.py` stops on a different `data_sha256`, and `results.csv` keeps 12 digits of it as `data_hash`.
- Fits: 126,000 in all (84,000 in the new runner and 42,000 in the old), 0 failed. Wall times on 2026-10-19 (UTC): screen 21:08 to 21:15; D4 confirmation 21:16 to 21:24 (7.7 min); D8 extension 21:24 to 21:34 (9.8 min).
- The design was fixed before the runs:
  - a screen of D1 to D5, D7 and D8 at n = 200 with 1,000 repetitions, and D4 at n = 1,000 as a pilot;
  - the decision rule: no material loss when every upper bound of new/old is below 1.05;
  - a confirmation of D4 at n = 200 and n = 1,000 with 7,000 repetitions each.
- Added after the screen, and not in the fixed design: D8 at n = 200 was extended to 23,000 repetitions. Its screen interval [0.985, 1.081] did not decide, and sd(g) = 0.487 asks for 21,365 repetitions for a 3-standard-error half-width of 0.01 (3 sd(g) / 0.01 squared), or 22,755 with the screen's ratio of 1.032, which the extension rounded up to 23,000. The seeds nest, so the screen is the first 1,000 repetitions.
- The scripts ran as copies in the runners' `validation/runs/`, and `skew_sims.py` ran as the module `validation.runs.skew_sims`. The committed `validation/sims/skew_sims.py` is formatted and has an updated docstring, with the logic unchanged, so `script_sha256` in the manifests is the hash of the runner's copy. The drivers are the runners' copies, and they find their runner from their place in `validation/runs/`.
- Files: `results.csv` (columns dgp, n, noise, arm, rep, error, excess_risk, n_terms, data_hash), `manifest_old.json`, `manifest_new.json`, `failure_counts.csv`, `driver_skew_stage1.sh`, `driver_skew_stage2.sh`, `driver_log_summary_old.txt` and `driver_log_summary_new.txt`. `make_report.py` builds `tables/skew*`.
