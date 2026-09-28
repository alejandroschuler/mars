# Decisions

This file lists the choices made for the user, with their reasons, so that the user can read it instead of the code. It changes only through pull requests.

## The user's answers of 2026-09-25

The table "Decisions" in VALIDATION_PLAN.md has the full text.

- Q1, where the work happens: in the fork `alejandroschuler/mars`, run by an executor agent, with agents as authors and reviewers; nothing is posted to the upstream repository.
- Q2, the behavior target: earth's defaults and rules where they fit scikit-learn, and another reasonable default where they do not.
- Q3, the license: Apache-2.0 stays, with a clean room; earth is only a black-box reference in the tests.
- Q4, missing values: `allow_missing=True` raises an error, and a fork issue tracks support for later.
- Q5, compute: 10 cores for about 60 hours over the weekend, then 10 cores until the work is done.
- Q6, the scope: a lean fork that installs with `pip install git+https://github.com/alejandroschuler/mars` and works with scikit-learn by default.

## Bootstrap (T00)

- The distribution name stays `mars-earth`, and the import name stays `pymars`. The plan uses these names, and they keep `import pymars as earth` working. Version 2 has no PyPI release.
- The runtime dependencies are `numpy>=1.23.5`, `scipy>=1.9.3` and `scikit-learn>=1.6`, and nothing else. scikit-learn 1.6 is the first release with `validate_data` and `__sklearn_tags__`. It accepts numpy 1.19.5 and scipy 1.6.0, but those releases have no wheels for Python 3.10. numpy 1.23.2 and scipy 1.9.2 are the first releases with wheels for Python 3.10 and 3.11 on Linux, macOS (arm64) and Windows, and the floors are the last patch releases of those two series. pandas 2 accepts the numpy floor (it needs numpy 1.22.4 or later on Python 3.10, and 1.23.2 or later on 3.11), so the lowest-version CI job can install pandas for the tests.
- `.pre-commit-config.yaml` is removed. pre-commit is no longer a dev dependency, the hooks pinned ruff 0.9.6 (the lock has 0.16.9) and named deleted files, and installing the hooks clones repositories from GitHub, which the plan's download rule does not allow. Gate A runs the same ruff checks.
- ruff checks neither `validation/legacy/` nor Markdown files. The legacy prototypes stay as they were run. ruff 0.16 formats the Python code blocks in Markdown files, and VALIDATION_PLAN.md must not change.
- In the tests, warnings are errors. DeprecationWarning and PendingDeprecationWarning are ignored when another package raises them, and they are errors when pymars raises them.
- Gate B runs ruff and `uv lock --check` too. A gate B pass thus covers the lint part of gate A and shows that the lockfile is current. Gate B runs the fast tests on Python 3.13 and 3.14 at the hypothesis `dev` profile: those runs check version support, and the Python 3.12 run has the `ci` profile.
- The project venv uses Python 3.12, the version of the full gate B run. `dev/tools/new_worktree.sh` and gate B pass `--python 3.12` to uv.
- CI pins uv 0.11.11, the release that made `uv.lock`, and pins each action to a commit SHA. Update the uv pin when `uv.lock` is made with a newer uv.
- CODE_OF_CONDUCT.md named the removed SUPPORT.md as its enforcement contact. The contact is now the maintainer of this fork, through the owner's GitHub profile or a private report to the repository owner, with no email address.

## Spec (T01)

- With `pmethod="none"` and `nprune`, pymars keeps the first `nprune` forward terms, as earth does, and reports `rss_`, `gcv_`, `rsq_` and `grsq_` of that model. earth reports the statistics of the backward subset of that size instead, which do not describe the model it returns (spec PRUNE-7, PRUNE-8).
- earth chooses between a pair of hinges and a single hinge by comparing an absolute residual sum of squares with 0.01, so its fit depends on the units of the covariates. pymars scales that threshold by the variances of the term's covariates. Its fit then does not depend on the units, and it matches earth's rule on covariates with variance 1; the harness gives both programs the same covariates, each non-constant covariate divided by its standard deviation with divisor N (the weight sum), not centered, and constant covariates unchanged (spec LA-7).

## Executor process

- A pure rebase gets a re-approval in place of a fresh review. A small checker agent checks that `git range-diff` shows every commit as `=` against the old head, that gate B passed on the new head, and that CI is green. It then posts a verdict line for the new head, for every role the merge needs, citing the approvals of the old head.
- A fix that touches only tests, only to satisfy CI, of about 20 lines or less, gets the same narrow re-approval. A checker agent checks that the fix stays within that scope, that gate B passed, and that CI is green. It then posts a verdict line for every required role.
- Before a stacked pull request's lower part merges, the executor retargets the part above it to `main` by hand. `dev/tools/merge_pr.sh` refuses to merge while another open pull request still uses the branch being merged as its base, because `--delete-branch` would close that pull request.
