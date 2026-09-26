# Brief: T05 fixtures S01 to S20

You make the earth fixtures that the reference implementation (T06), the component tests (T08 to T10, T12 to T14) and the conformance tests (T07) read. Read `briefs/COMMON.md` (same folder) first. Issue: T05, #7. Reviewers: 2 (`spec`, `adversarial`).

## Read first

- `VALIDATION_PLAN.md` on `origin/main`: "Correctness against earth" (all of it, above all "Comparison modes", "Test datasets", "What is compared, and the tolerances", "Ties", "Component tests", "Binary outcomes", "Sample weights", "Categorical inputs", "Edge cases").
- `docs/algorithm.md` on `origin/main` (spec part 1; part 2 is PR #37, which you may read for the name table API-7): above all LA-7 (the harness step), W-4, KNOT, PRUNE-3, CORE-4 and the Departures.
- `validation/harness/` and `validation/README.md` on `origin/main`: the harness you build on (`gen_fixtures.py` registry, `driver.py`, `blackbox.py`, `trace_parse.py`, `compare.py`). S01 exists already.

## Rules that matter here

- Clean room: earth only as a black box, as in COMMON.md. Only the spec writer and its helpers run trace experiments for the spec; you record traces as fixture data for the knot-set tests (T08), which is allowed, but you do not interpret earth's internals beyond what the spec states.
- The harness step (spec LA-7): in the matched and earth-compatible modes, each non-constant covariate is divided by its standard deviation with divisor N (weighted, N = Σw), not centered; constant columns stay as they are, at moderate sizes; the same scaled matrix goes to both programs, and the fixture stores that matrix and the scale factors. S12, which is about scale and shift, also stores earth's fits on the raw variants, because earth is not scale invariant (spec bb14.4).
- Deterministic data: every dataset from a fixed seed; continuous covariates without repeated values unless the case is about ties.
- Size: keep `validation/fixtures/` small (aim for 10 MB or less in total): trim traces to the steps and searches the tests need, and use short test sets.

## Deliverables

1. Datasets S02 to S20 in the `gen_fixtures.py` registry, as the plan's "Test datasets" table defines them, each with the modes and settings its purpose needs: the matched mode (the same explicit arguments to both programs, for example `Auto.linpreds = FALSE` and `fast.k = 0`, as the plan's "Comparison modes" says for the new code) and the earth-compatible mode (earth's defaults) where they apply; the span grids of S01 and S02 (minspan 1, 5 and automatic; endspan 1, 10 and automatic); degrees 1, 2 and 3 where the case says; S13 and S16 with weights as the plan's "Sample weights" says (integer weights through repeated rows against unweighted earth; non-integer weights only on the fixed-basis path, with weights rescaled to Σw = n); S15 as 200 small draws from `validation/sims/dgps.py`; S17 (three responses), S18 (three classes, with `nnet::multinom` on earth's selected basis), S19 (a factor given to earth as a factor and as treatment dummies), S20 (a separable binary response). Where earth errors on an input (for example some weighted constant responses, spec bb10.9), record the error in the fixture instead of dropping the case.
2. Component fixtures in `validation/fixtures/components/`: the `earth:::get.gcv` grid (RSS values, 1 to 41 terms, penalties 0 to 6 and −1, n from 10 to 100,000); fixed-basis pruning on earth's forward bases (term removed at each step, `rss.per.subset`, `gcv.per.subset`, the selected size), for one and for several responses; `lm.fit` coefficients on fixed columns; `predict.earth` at new points, also outside the training range; R's `glm` (binomial) and `nnet::multinom` on fixed columns; trace-based candidate knot sets for the knot tests (the grid of the plan's "Candidate knot sets": minspan automatic, 1, 3 and 10, endspan automatic, 1 and 5, degree 1 and 2, `Adjust.endspan` 1 and 2, and 20, 200 and 2,000 cases; negative minspan is a `later` issue, so leave it out).
3. One command makes every fixture again, and `gen_fixtures.py --check` reports no difference.
4. `.github/workflows/earth-conformance.yml`: `workflow_dispatch` only, `permissions: contents: read`, no secrets; it installs R (`r-lib/actions/setup-r`, pinned to a commit SHA) and earth 5.3.4 and nnet, then runs `gen_fixtures.py --check`. It must not trigger on anything else.
5. Tests: a pure-Python test (runs in CI) that every fixture has the required fields, a versions block and finite or explicitly marked values; an `external` test that makes a few fixtures again and compares them.
6. `validation/README.md`: a table of the fixtures, what each is for, and the command to make them again.

## Size and splitting

The generated fixtures do not count toward the 800-line guide; the code does. If the code grows large, split into two pull requests: first the component fixtures (so T08 to T10 can start), then S02 to S20.

## Report

30 lines or fewer: the pull request numbers and heads, the gate results, the fixture sizes, the `--check` result, the cases where earth errors, and anything T06, T07 or T08 to T10 must know.
