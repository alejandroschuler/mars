# Brief: why pymars keeps more terms than earth on pure noise (issue #173)

Read `COMMON.md` (same folder) first; its rules hold, except the paths. This executor is session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`, which runs in `<S>` = `/Users/aschuler/Documents/research/projects/pymars/.claude/worktrees/competent-poincare-5a4d49`. The Write and Edit tools work only inside `<S>`, so make any worktree with `PYMARS_WORKTREES=<S>/.worktrees bash /Users/aschuler/Documents/research/projects/pymars/dev/tools/new_worktree.sh <branch>` (read the script first). Long jobs follow COMMON.md's "Long jobs" (detached, `nice -n 15`, at most 8 workers).

The user's priorities: speed, predictive performance, and MARS as Friedman describes it. The user's condition for this task: a fix must not hurt performance where the data have signal.

## Phase 1: diagnosis (this agent; report before any fix)

You act as the spec writer's helper: you may run earth as a black box, with `trace = 7` to `9`, and read Milborrow's notes and earth's help pages locally (`<main>/.git/pymars-executor/ref/`). You may not read earth's C code or print R function bodies (COMMON.md, "Clean room").

1. From `validation/sims/results/full/` on main, find the D6 (pure noise) repetitions at n = 200 where P-fix (pymars) and E-def (earth) select different models. The report says 15 percent of 7,500 differ. Rebuild about 30 of them from their seeds with the simulation code (`validation/sims/`), with the same settings as the study (P-fix uses `glm_alpha=0` only for binary outcomes; D6 is continuous).
2. For each, find the first step where the two programs differ: pymars's forward record and candidate log against earth's trace, then the pruning pass. Classify the cause:
   - the pair rule (pymars scales the comparison of a pair against a single hinge by the covariate's variance, earth works in raw units: E1 in `validation/DIFFERENCES.md`, LA-7 in the spec);
   - a near-tie broken the other way;
   - Fast MARS's candidate queue;
   - pruning or GCV;
   - something else (describe it).
3. Count the causes, and say which one explains the extra terms. If one rule explains most of them, say what change would make pymars follow earth there, and estimate, from the same reps, whether that change removes the gap.

Do not change pymars or the spec in phase 1. Write scratch scripts in your own scratchpad subfolder. Report to the executor in 30 lines or fewer: the counts by cause, three example seeds with the step and the values, the proposed change if any, and its expected effect.

## Phase 2 (only after the executor decides): a fix and its check

A rule change is a spec PR, then the fast code and the reference (separate authors, clean room). Its acceptance test is fixed in advance, so that the result cannot steer the test:
- the conformance fixtures agree with earth at least as often as on main;
- a rerun of the whole T21 parity grid (all 51 cells, the same seeds and repetitions as the study) shows, for every cell with signal, no loss: the ratio of pymars's excess risk to earth's does not rise by more than its Monte Carlo interval allows, and the cell's equivalence verdict does not change from pass to fail;
- D6's ratio falls.
