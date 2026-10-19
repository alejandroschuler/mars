# Brief: the two defaults of #175 (user decisions of 2026-10-19)

Read `COMMON.md` (same folder) first; its rules hold, except the paths. This executor is session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`, which runs in `<S>` = `/Users/aschuler/Documents/research/projects/pymars/.claude/worktrees/competent-poincare-5a4d49`. The Write and Edit tools work only inside `<S>`, so make your worktree with `PYMARS_WORKTREES=<S>/.worktrees bash /Users/aschuler/Documents/research/projects/pymars/dev/tools/new_worktree.sh u175-defaults` (read the script first). Gate logs stay in `<main>/.git/pymars-executor/gates/`.

The project is done on main; this is one small follow-up PR (`Closes #175`). Be lean: the fewest tests that pin the two changes, and no campaigns. The user's priorities for pymars: speed, predictive performance, and MARS as Friedman describes it.

## 1. EarthClassifier's default `glm_alpha` becomes 1.0

The user chose "Positive default, 1": the same strength as scikit-learn's LogisticRegression default (C=1). At 0, the unpenalized refit warns on breast_cancer, iris and wine, and its cross-validated log loss on breast_cancer is 0.685, against 0.105 at alpha 1 (#175).

- Change the default in `pymars/_estimators.py` (signature and docstring), and wherever the spec or the docs state the default (`docs/algorithm.md` only where it states the default value; the usage, differences and README pages; CHANGELOG).
- Document `glm_alpha=0` as the earth-compatible setting.
- `validation/REPORT.md`: one short note where the binary results are reported, saying that the simulation and the conformance fixtures used `glm_alpha=0`, the default at the freeze, and that the default is now 1. Change no number in the report.
- Tests: update the tests that rely on the old default by passing `glm_alpha=0` where they compare with earth; add one test that pins the new default.

## 2. numba: keep it by default, with a Python guard and a warning

The user chose to keep numba as a dependency wherever it has wheels, "assuming the user is alerted that it will be slow without numba".

- `pyproject.toml`: keep the marker that leaves out macOS x86_64, and also leave numba out on Python versions newer than the newest one that numba publishes wheels for. Check PyPI's numba page for the newest CPython wheels of the latest release (reading the page is allowed). Write the marker in PEP 508 syntax, which has no `not`; for example `python_version < "3.15" and (sys_platform != "darwin" or platform_machine != "x86_64")`. Put a comment next to it that says when to raise the bound.
- A warning when the numpy fallback runs: issue it once per process, at the first fit that uses the numpy path, not at import. The message says that numba is not available, that fits are then about 3 times slower, and how to install numba. Use a clear warning class (a subclass of UserWarning in pymars is fine), and document it in the usage page.
- Tests: one test that the warning comes once per process when the fallback runs (force the fallback, for example by monkeypatching the flag that says numba is available).

Then gate A on each commit, gate B on the head, open the PR (ready, not draft), and report in 15 lines or fewer: the PR number, the head SHA, the gate B result, the marker you chose and why, and any blocked command with its exact text. Reviewer: 1 (`single`); the executor starts it.
