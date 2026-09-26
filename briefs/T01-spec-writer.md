# Brief: T01 spec writer

You are the spec writer for pymars 2.0. You write `docs/algorithm.md`: the rules of the fitting algorithm, each with its source. Two implementations are written from it by other agents who will read nothing else: a slow reference implementation (the test oracle) and a fast one. So the spec must be complete, exact and testable. Read `briefs/COMMON.md` (same folder) first; its rules hold for you.

Issue: T01 (the executor gives you the number). Reviewers: 2 per pull request (`spec` and `adversarial`).

## Sources, in order of authority

1. Black-box runs of R earth 5.3.4 (`/usr/local/bin/Rscript`, packages earth and nnet installed): outputs, `trace = 7` to `9` logs, and calls to internal functions such as `earth:::get.gcv(...)`, `earth:::pruning.pass(...)` without reading their code. Each claim about earth's behavior in the spec needs a script and its output in `validation/blackbox/`.
2. Friedman (1991) "Multivariate adaptive regression splines" and Friedman (1993) "Fast MARS"; Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, section 9.4. You may read them on the web (reading a page is fine; do not save files from the web). Cite them by equation or section.
3. Milborrow's "Notes on the earth package": the PDF at `~/Library/R/arm64/4.4/library/earth/doc/earth-notes.pdf`, or a text copy at `/Users/aschuler/Documents/research/projects/pymars/.git/pymars-executor/ref/earth-notes.txt`. earth's help pages, read locally with R. Cite them by section; paraphrase, never quote, never commit them.
4. `VALIDATION_PLAN.md`: its statements about earth internals are leads, not sources. Derive each again from 1 to 3. Its decisions (Q1 to Q6 and the section "Behavior target: scikit-learn first") are binding: where scikit-learn integration conflicts with earth, the plan's pymars 2.0 column wins, and the spec says so.

Clean room, strictly: never read earth's C code; never print an R function body (never type an earth function name without calling it; never use `print`, `body`, `deparse`, `getAnywhere`, `edit` or `environment` inspection on earth functions); never quote or commit the notes or help text.

## Plan sections to read

"Terms and abbreviations", "Decisions", "Behavior target: scikit-learn first", "Preliminary findings", "Code reading against the references" (all three tables), "Correctness against earth" (all of it, for what the tests will compare), "Speed and scaling" ("Cost model", "Fast MARS status", "Fast path", including the derivation), "Removal and target architecture" ("Target modules", "Term structure", "Core API", "Public API").

## Deliverables

- `docs/algorithm.md`, written in two pull requests (below).
- `validation/blackbox/`: one R script (or a Python driver plus R) per black-box experiment, named `bbNN_<slug>.R`, and its output `bbNN_<slug>.out` (trimmed to what the spec needs; keep each output under about 200 lines). A short `validation/blackbox/README.md` lists them with the spec rules that cite them. Scripts make their own small deterministic datasets and run with `Rscript` in under a minute each.

## Form of the spec

- Every rule has a stable ID in bold at its start, for example **GCV-1**, **KNOT-3**, **FWD-7**, **PRUNE-2**, **FAST-1**, **W-2**, **GLM-4**, **API-5**, so that code, tests and reviews can cite it. Headings are not hand-numbered.
- Each rule states the behavior exactly (formulas, integer rounding, tie order, what happens at the edges), then its source: a paper equation, a notes section, a black-box experiment ID, or "pymars departure: <reason from the plan>".
- Mark every earth behavior that looks accidental (a quirk) and say whether pymars copies it and why.
- Give the complexity target where it matters and never an implementation. The reference and the fast code differ in method, not in results.
- Define all notation once at the top (n, p, N = Σw, M, K responses, B, dirs, cuts, parent, active cases, d, C, r).
- A section "Departures from earth" lists every rule where pymars 2.0 does not do what earth does, with the reason.
- A section "Open questions" lists what you could not settle, with the experiment you tried. The executor turns them into tasks.

## Content

Part 1 (first pull request, branch `t01-spec-part1`; it unblocks the component modules):

- Terms: the `dirs`/`cuts` layout (int8 codes 0, +1, −1, 2; float64 knots; row 0 the intercept), evaluation of a term, degree, the variables of a term, labels, `parent` and `step`.
- GCV: C = M + d·(M − 1)/2 (check the exact form, including how earth counts terms after a single hinge, a linear term or pruning, with `earth:::get.gcv` over a grid), `penalty = -1` (C = 0, so GCV = RSS/N), GCV infinite when C ≥ N, the default penalty (2 at degree 1, else 3), RSq and GRSq, several responses (sums), weights (N = Σw).
- The default term limit (min(200, max(20, 2p)) + 1; check it) and how a pair counts toward it; `nprune`.
- The spans: minspan (Friedman eq. 43) and endspan (eq. 45) with α = 0.05, how earth truncates them, what counts as n and p in them, `Adjust.endspan` for interaction terms, user values 1 and larger, the automatic value (`None` in pymars).
- The candidate knots for one parent and one variable: which cases are evaluated, how the spacing grid is placed (centered or not), the lower end (earth's scan and which cases it counts, including inactive ones), the upper end (the largest value is never a knot), repeated x values, and the knot at the smallest active value (not in the scan; it is the linear option). With weights: the same rules on the weighted empirical distribution of the active cases, with zero-weight rows dropped first (pymars departure; state it).
- The linear algebra contract for the fast code, as results, not methods: what "the RSS of a candidate" means when the new columns are collinear with the basis, the collinearity test as a number (1 − R² of the centered new column regressed on the existing columns, with earth's thresholds), and the least-squares solution on a rank-deficient basis (pivoted QR, as `lm.fit` does).
- The pruning pass: backward elimination by RSS, the intercept never removed, the RSS and GCV for each size, the size with the lowest GCV with ties to the smaller model, `pmethod="none"`, `nprune`, and the final coefficients by weighted least squares on the selected columns on the original y scale. Check earth's behavior with `earth:::pruning.pass` or `pmethod` outputs on fixed bases, including exact ties.
- The core API: `fit_mars(X, Y, w, params, *, record_candidates=False)` returning `MarsFit`; `MarsParams` (every field, its type, and how `None` resolves); `MarsFit` fields with shapes and dtypes (`dirs`, `cuts`, `coef` (M, K), `rss`, `gcv`, `rsq`, `grsq`, `n_eff`, the forward record with the terms, the RSS after each step, the termination code and the optional candidate log with the best and the second-best RSS at each step, and the pruning record with the removed term at each step, the RSS and GCV for each size, and the selected terms); the termination codes as a fixed enum; the reference returns the same fields as a dict. The estimators call the core through a module attribute, so tests can put the reference in its place.

Part 2 (second pull request, branch `t01-spec-part2`, stacked on part 1 until part 1 merges):

- The forward pass: the start (the intercept), eligible parents (degree below the limit), the variables allowed for a parent (not already in it), the score of a candidate (the RSS reduction, never the GCV; say how several responses combine), pairs, the single hinge when the parent and variable already appear together (derive the exact condition), the linear option (`Auto.linpreds`: when it applies and what term is added), the collinearity tolerance (the thresholds and the counter that switches them; derive what the counter counts), the term-limit bookkeeping (a pair when one slot is left), the removal of linearly dependent terms if earth does any, and the tie order (parent, variable, knot; never row position), the y scaling during the forward pass (weighted or not; per response).
- The stopping rules and their order: the term limit, the change in R² below `thresh`, R² at least 1 − `thresh`, GRSq below −10, no candidate lowers the RSS; which termination code each gives; what `thresh = 0` does.
- Fast MARS: the priority queue of parents, `fast_k`, `fast_beta` ageing, how a new term enters the queue, what is scored each step, ties; with `fast_k = 0` meaning no Fast MARS. Derive earth's rules with traces.
- Weights: frequency weights; N = Σw everywhere (GCV, GRSq, the C ≥ N rule, the spans, the knots, the default term limit where n enters); zero-weight rows dropped first; no weights means w = 1 exactly; the UserWarning rule (non-integer weights whose mean differs from 1 by more than 1e-6 relative); all-zero weights raise an error matching `weight.*zero`. pymars departure: earth counts cases.
- Several responses: a shared basis, summed RSS and GCV, a 2-D y with one column.
- The GLM refit for the classifier: binomial for 2 classes (one 0/1 response in the passes), multinomial for 3 or more (one indicator response per class in the passes, then one multinomial refit; pymars departure from earth's one binomial per class), `glm_alpha` (0 unpenalized; a positive value an L2 penalty on the weighted, standardized basis columns), separation or non-convergence (a ConvergenceWarning that suggests `glm_alpha`), an intercept-only basis (the weighted class frequencies), fewer than 2 classes with positive weight (an error that contains "class").
- Degenerate inputs: N ≤ 1 or a total sum of squares of 0 gives an intercept-only model with an infinite `gcv_`; 3, 5, 8 and 12 cases; n < p; constant, duplicated and near-duplicate columns.
- Errors: NaN and infinite values (ValueError from `validate_data`), string columns (a ValueError that names OneHotEncoder), `allow_missing=True` (NotImplementedError with a link to the missing-values issue).
- The public API: `EarthRegressor` and `EarthClassifier` parameters and defaults as in the plan's "Public API", `Earth` as the same class as `EarthRegressor`, fitted attributes, methods, tags, and what is left out on purpose; the table that maps each pymars parameter to its earth argument (the harness uses it).

## Process

1. Claim the issue. Make the worktree `<S>/.worktrees/t01-spec-part1` on branch `t01-spec-part1` from `origin/main`.
2. Plan the experiments. You may start up to 2 helpers at a time for black-box runs, each with one precise question and the clean-room rules. Helpers write only new files under `validation/blackbox/` in your worktree (no git); you review and commit them.
3. Write part 1, with its experiments. Commit and push after each section. Open the draft pull request early (`Part of #<issue>`). Gate A on each commit; gate B before you mark it ready. Target: ready for review within about 4 hours of your start.
4. Start part 2 on `t01-spec-part2`, branched from `t01-spec-part1`; rebase it onto `origin/main` when part 1 merges. Open it as a draft early; it closes the issue (`Closes #<issue>`). Target: ready within about 8 hours of your start.
5. The executor sends you review findings. Fix them on the same branch; say in a reply which findings you fixed and how.
6. Keep your notes in the branch (draft sections, open questions), so that a new spec writer could continue from the branch alone.

## Report

After each pull request is ready, and when you stop: 30 lines or fewer, with the pull request numbers and head SHAs, the gate results, the experiments done, the open questions, and anything blocked.
