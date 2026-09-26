# Brief: T03 simulation harness author

You build the simulation harness for the statistical performance study. Read `briefs/COMMON.md` (same folder) first; its rules hold for you. Issue: T03 (the executor gives you the number). Reviewer: 1 (`single`).

Before you write any code, invoke the skills `stats-research:design-and-report-simulations` and `stats-research:supervised-learning` with the Skill tool, and follow them. The plan was written with them; where the plan is more specific, the plan wins.

## Plan sections to read

"Statistical performance study" (all of it: "Claims", "Arms and DGPs at a glance", "Goals", "Mockups", "Data-generating processes", "Learners and settings", "Performance measures", "Build", "Pilot and number of repetitions", "Compute ledger"), "Long computations", "Concurrency and cores", and "Behavior target" (for the P-fix arm).

## Deliverables (all under `validation/sims/`)

- `dgps.py`: D1 to D8, D3-bin and D4-bin exactly as the plan defines them. X ~ Unif[0, 1]^p with p = 10 (D7: p = 50); D8 through a Gaussian copula with latent correlation 0.6. σ from the population R² (0.8 and 0.3): σ = sd(f)·√((1 − R²)/R²), with sd(f) from the diagnostic draw; D6 uses σ = 1 and one noise level. For the binary DGPs, λ = logit(0.95)/max(|q₀₁|, q₉₉) from the 1st and 99th percentiles of f(X) − E f(X) on the diagnostic draw. Each DGP returns X, y and the true f(X) (or μ(X)), and knows which covariates are relevant.
- `diagnostics.py`: one draw of 10⁶ cases per DGP, with a fixed seed; it writes `validation/sims/diagnostics.json` (committed, small): Var f, σ at both levels, the population R², the share of Var f that the best linear approximation explains, λ and the range and quantiles of μ(X) for the binary DGPs. Check the plan's values λ ≈ 1.60 (D3) and 0.28 (D4) and the correlation ≈ 0.58 for D8, and report any difference.
- Seeds from names: the training seed comes from the cell name (DGP, sample size, noise level) and the repetition index through a stable hash (not Python's `hash`), and `numpy.random.default_rng`; the test set (10,000 points) comes from a second stream of the same seed. No seed depends on a loop position, so pilot repetitions are the first repetitions of the full run.
- `learners.py`: the arms with fixed settings as in "Learners and settings": E-def, E-pym, P-cur, P-ear, P-fix, OLS and HGB, and the binary arms (E-def with `glm = list(family = binomial)`, the 1.0.4 `EarthClassifier` and `GLMEarth`, P-fix classifier, logistic regression with the covariates as linear terms and no penalty, the HGB classifier). HGB records `n_iter_` so the report can apply the edge rule. earth runs through `Rscript` in blocks (one R process fits a block of datasets and writes the predictions). The legacy arms run in `.venv-legacy` (made by `validation/legacy/make_venv.sh`) through a subprocess worker that fits several repetitions per process. P-fix imports `pymars.EarthRegressor` and `pymars.EarthClassifier` from the main venv; they do not exist yet, so the arm raises a clear error until they do, and the tests use a stand-in. Record for every MARS fit the number of selected terms and the covariates it uses.
- `run.py`: a command line with the cells, the arms, the repetition range, the number of workers, `--resume` and the output folder (under the ignored `validation/runs/`). Results are written per cell and arm, atomically (a temporary file, then `os.replace`), with a manifest (settings, versions, the pymars commit, the earth version, the source hash of each learner). `--resume` skips finished work. Predictions are cached on disk, keyed by a hash of the training and test data, the learner name and settings, the learner's source code and the package versions. Failures are recorded per fit with the error type, and never stop the run. Each worker uses one BLAS thread (`dev/env.sh`); the cells run in parallel with joblib; the command is meant to run under `nohup caffeinate -i nice -n 15`, with a PID file and a log in the output folder.
- Measures in `metrics.py`: the excess risk R(f̂) = mean of (f̂(X) − f(X))² on the test set; paired log ratios g_i between two arms; for binary outcomes the excess log loss (the mean Bernoulli KL divergence of μ̂ from μ), the excess Brier score, the calibration slope (logistic regression of Y on logit μ̂ in the test set), with probabilities clipped to [1e-6, 1 − 1e-6] and the number of clipped values counted; the number of terms; the use of irrelevant covariates; the fit time.
- `summarize.py`: every display of "Mockups" (the ratio table, the equivalence figure, the per-repetition box plots, the selection table, the binary-outcome table) and the appendix tables with Monte Carlo standard errors and failure counts, from the per-repetition results on disk. Captions come from the plan and name the claim each display serves. Mark cells that are not yet run as missing; never invent numbers.
- `pilot_check.py`: for a difference contrast, n_sim = (k·s_p/|ḡ_p|)² with k = 5, and `n_sim_safe` with |ḡ_p| replaced by max(0, |ḡ_p| − 1.96·s_p/√n_pilot) (infinite when that is 0); for an equivalence contrast with Δ = log 1.05 and k = 3, n_sim = (2k·s_p/Δ)², and the statement that no n_sim works when |ḡ_p| ≥ Δ; the pilot statistic z = ḡ_p/(s_p/√n_pilot).
- Tests in `validation/sims/tests/`: seeds are stable across runs and processes; the σ formula gives the target population R² on a large draw; the metrics are 0 for the true function and correct on a small hand case; `--resume` skips finished cells and redoes a cell whose file is incomplete; atomic writes; `pilot_check.py` formulas on hand cases. Tests that need R or the legacy venv are separate from the pure-Python tests and do not skip; say in the pull request which ones you ran.
- `validation/sims/README.md`: how to run the diagnostics, a smoke run, the pilot and the full run, and where the results go.

## Checks

- Gate A and gate B pass.
- The diagnostics run once, and `diagnostics.json` is committed.
- A smoke run of 2 repetitions per cell for the arms that exist now (E-def, E-pym, P-cur, P-ear, OLS, HGB; the D7 legacy arms excepted) at 200 cases completes with no failures, and `summarize.py` runs on it. Do not start longer runs: T04 does the pilot and the full run.
- New Python dependencies go only in the `validation` dependency group (update `uv.lock`).

## Size

If the change grows past about 800 lines that are not generated, split it into two pull requests: first the DGPs, the diagnostics, the seeds, the metrics and `pilot_check.py`; then the learners, `run.py` and `summarize.py`.

## Report

30 lines or fewer: the pull request numbers and head SHAs, the gate results, the diagnostics (λ for D3-bin and D4-bin, the D8 correlation), the smoke-run fit times per arm, and anything T04 should know.
