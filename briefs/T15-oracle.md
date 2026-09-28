# Brief: T15 the oracle tests and gate C (tests/test_oracle.py, dev/gate_c.sh)

Read `briefs/COMMON.md` (same folder) first, above all the "Tests" rules: the oracle test is the broad comparison that those rules prefer, so it should replace narrow unit tests, not add to them. Issue: T15, #17. Reviewers: 2 (`spec`, `adversarial`). Model: the strongest.

The oracle tests compare the fast code with the reference implementation (`tests/reference/mars_ref.py`, complete on main). Every later change to the fast code must pass them, and gate D (the phase exit before the freeze) rests on them.

## Read first

- `VALIDATION_PLAN.md` on `origin/main`: "Gates" (gate C), "Tests and validation folders" (`test_oracle.py`), "Ties" (the near-tie rule and its 1e-7 band), "What is compared, and the tolerances".
- `docs/algorithm.md`: CORE-3 and CORE-5 (the records), LA-5 (the accuracy bounds), STOP-7 (near-tie bands), and the Departures.
- The public APIs, by their docstrings: the reference (`fit_mars`, `forward_pass`, `prune` and the part-1 helpers) and the fast modules on main (`_gcv`, `_knots`, `_linalg`, `_pruning`, `_scan`, and `_forward` once PR #55 merges; `_core.fit_mars` once T12 merges).

## Rules for this task

- You change neither implementation. A disagreement that is not a near-tie is a finding: report it to the executor with a minimal reproduction, and mark nothing as expected.
- One place lists the settings that the fast code supports (degree 1 only, no weights, one response, `fast_k = 0` for T11 stage 1), so that each later stage of T11 widens the tests by editing that one list. Generate only supported settings; no skips.
- Near-ties: compare the forward records step by step up to the first step where either program's best and second-best candidate RSS differ by less than the plan's band (1e-7 of the RSS before the step); after that step, compare nothing structural for that fit, and count it. Report the share of fits that stop at a near-tie.

## What to build

- `tests/test_oracle.py`:
  - the fast forward pass against the reference forward pass on hypothesis data (a few strategies: smooth truths, hinge truths, ties, shifted and scaled covariates, small n) and on every fixture dataset in the supported settings: the terms, the RSS path (the LA-5 tolerance), the termination code, and the candidate log up to the first near-tie;
  - the fast pruning pass against the reference pruning pass on random bases, with and without weights and for K = 1 and K ≥ 2;
  - `_core.fit_mars` against the reference `fit_mars` through `MarsFit.from_dict` when T12 has merged (if T12 is not on main yet, leave a clearly named function that the T12 pull request, or you in a follow-up, connects; no skip);
  - replace existing narrow tests in `tests/test_forward.py`, `tests/test_pruning.py` and `tests/reference/` that only repeat what the oracle comparison now checks, and list the deletions in the pull request body (the executor's pruning pass, STATE.md item 5).
- Hypothesis profiles: `dev` (50), `ci` (200), `thorough` (2,000), as `tests/conftest.py` defines them; the oracle tests must finish in about 2 minutes at `ci`.
- `dev/gate_c.sh`: the oracle and invariance tests at the `thorough` profile; if the diff against `origin/main` touches `validation/harness/`, the fixtures made again with R and `gen_fixtures.py --check`; a log `<git-common-dir>/pymars-executor/gates/<head>.gateC.log` whose last line is `GATE C PASS <head>` or `GATE C FAIL <head>`, as `merge_pr.sh --require-gate-c` reads. The benchmark smoke of gate C comes with T19; say so in the script.

## Report

30 lines or fewer when the pull request is ready: the number and head, the gate results, the fits compared and the near-tie share, every disagreement with its reproduction, the tests you deleted, and the time at each profile.
