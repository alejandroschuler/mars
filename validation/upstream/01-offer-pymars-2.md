# Title: Offer: a rewrite of the fitting code that is checked against R's earth, with its tests and report

Hello, and thank you for `mars`. I am writing with a piece of work that may be useful to the project, and with a set of separate issues that came out of it. Please take any of it, in whole or in part, or none of it.

## Why I looked

I wanted to know whether `mars-earth` can be the MARS learner in Python for a library of learners in supervised learning, where the usual choice is R's `earth`. So I ran the code against `earth` 5.3.4 on a set of datasets, with settings that make the two programs comparable, and I read the code against Friedman (1991, 1993). The project's own plans (`conductor/tracks/reference_regression_validation_20260420/spec.md`) put R `earth` and benchmarks across implementations out of scope for validation, and the parity audit compared documentation. That is a reasonable scope for a project of this size. The runs below are meant as added evidence, not as a criticism of that choice.

The core search is sound. In a matched mode (hinge terms only, degree 1, no linear terms, a penalty equal to half of earth's, every case a candidate knot) the 1.0.4 code picked the same ten knots in the same order as `earth` on Friedman's first test function, and the RSS agreed to about 13 significant digits after every step. The problems are in the rules and defaults around that search, and in speed. I have written them up as separate issues with a short reproduction each, run against 1.0.4 and, where it applies, against `earth`.

## What I can offer

I replaced the fitting code. The result is a package with the same distribution and import names (`mars-earth`, `pymars`), and it is in the fork at `<link to the fork>`. It has four parts:

1. **The 2.0 code.** A written specification (`docs/algorithm.md`) states each rule, with an id, and says where it follows `earth` and where it departs (`docs/differences.md`). The fitting code follows the specification. It has the Friedman (1991) updating formulas and the Fast MARS queue, and the estimators follow scikit-learn's conventions (`EarthRegressor`, which `Earth` aliases, and `EarthClassifier`). A slow reference implementation written from the specification alone (`tests/reference/`) serves as the test oracle, and the fast code is compared with it.
2. **The conformance harness and fixtures.** `validation/harness/` runs `earth` as a black box (it calls the program and reads its results) and compares fits step by step: the terms each forward step adds, the RSS after the step, the pruning path, the selected terms, the coefficients and the predictions, with tolerances that depend on the condition of the basis. `validation/fixtures/` holds 133 dataset fixtures (inputs and `earth` output as JSON, with the R and `earth` versions) and one command that makes them again. These replace the stored outputs of the current reference tests.
3. **The validation report.** `validation/REPORT.md` is organized by claim (conformance, the gap to `earth`, parity in prediction error, sparsity, speed), with the design of each study and what the evidence shows. `validation/DIFFERENCES.md` triages every difference from `earth` as a rule, a quirk, a tie or a numerical difference, and `validation/DIFFERENCES_legacy.md` does the same for 1.0.4.
4. **The tests.** The suite runs scikit-learn's `check_estimator` on the estimators, checks weights (an integer weight equals repeated rows), invariances (row and column order, units), and edge cases.

Numbers, from `validation/REPORT.md` and `validation/bench/REPORT.md` (refresh them if the code has moved on by the time you read this):

- Conformance: the reference implementation agrees with `earth` in full on 119 of 133 dataset fixtures and 181 of 200 random draws; the other differences each have a label and a reason in `validation/DIFFERENCES.md`, and all of them are documented rule differences. The fast code shows the same differences as the reference on all 333 comparisons.
- Parity in prediction error: equivalence to within a factor of 1.05 of `earth`'s excess risk is shown in 51 of 51 cells, after a change to the pair rule (LA-7) that removed a 7.6 percent gap on pure noise at 200 cases. On skewed covariates the change costs up to 1.9 percent on one function (a known limit in the report). The report says which cells are weaker evidence.
- Speed, at 10,000 cases, 10 covariates, degree 2: `EarthRegressor` takes 1.7 times `earth`'s time in a fresh process and about 0.9 times in a warm one with numba, and 2.6 and 2.7 times with the numpy fallback. The 1.0.4 code takes 144 s for 1,000 cases with the same settings, where the new code takes under a second.

## What would change for users

This is a major version, so I changed some behavior on purpose: `allow_missing` raises instead of returning NaN predictions, `GLMEarth` is merged into `EarthClassifier` (which refits without a penalty and has `predict_proba`), `categorical_features` is replaced by scikit-learn's `OneHotEncoder`, `sample_weight` is a frequency weight everywhere, and the code outside the fitting path is not part of the package. The last point is the subject of a separate proposal. Each change has a reason in `docs/differences.md`.

## About patches

After a rewrite there is no patch for each fix, because the findings in the separate issues come from rules that the old code did not have. I would rather offer the new code as a whole, and let the maintainers decide how to take it: as a replacement, as a new major version on a branch, or as a source of tests and fixtures to use against the current code. The fixtures and the harness do not depend on the new code. They run against any implementation that returns a fit in the common schema, and the harness has an adapter for 1.0.4.

## License and provenance

The new code is Apache-2.0, like the existing code. `earth` is GPL-3, so I used it only as a black box: I ran it, read its outputs and trace logs, and read its help pages. I did not read its source code. The file `AGENTS.md` in the fork states the rule.

## How the work was done

Most of the work, including the code, the tests and the reviews, was done by AI agents (Claude Code) under my direction in the fork, and the fork's pull requests show it. You may want to know that before you decide how much to trust any single claim, and the harness exists because I did not want anyone, me included, to take the claims on trust: every statement in the report can be reproduced from the fixtures with one command.

Thank you for reading. I am glad to answer questions, to split the work into smaller pieces, or to take it down if it is not wanted.
