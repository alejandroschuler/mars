# Contributing

People and agents work the same way here. [AGENTS.md](AGENTS.md) adds the rules for agents.

## Set up

You need git and [uv](https://docs.astral.sh/uv/). Make a task worktree with its dev venv:

```bash
dev/tools/new_worktree.sh t07-conformance   # from origin/main
```

In a plain clone, do the same by hand:

```bash
. dev/env.sh
uv sync --frozen --group dev
```

`dev/env.sh` sets one thread for BLAS and OpenMP, so that results do not depend on the thread count, and it stops uv from downloading Python interpreters. Source it before tests, gates, simulations and benchmarks. The `validation` dependency group (`uv sync --group validation`) adds the packages for the harness and the simulations. They never ship.

## Gates

- `dev/gate_a.sh` runs ruff and the fast tests (`pytest -m "not slow"`) in a minute or less. Run it before every commit.
- `dev/gate_b.sh` runs the full suite at the hypothesis `ci` profile on Python 3.12 with at least 90 percent coverage, the fast tests on Python 3.13 and 3.14, and a wheel build with a smoke check. Run it on a clean checkout before you ask for a review. It writes its log to `<git-common-dir>/pymars-executor/gates/<sha>.gateB.log`.
- CI (`.github/workflows/ci.yml`) runs ruff, the fast tests on Linux, macOS and Windows, the slow tests, the lowest direct dependency versions on Python 3.10, and a wheel build.

## Tests

Tests live in `tests/` and use pytest and hypothesis. Mark a long test with `@pytest.mark.slow`; gate A skips it. `HYPOTHESIS_PROFILE` sets the number of examples: `dev` (50, the default), `ci` (200) or `thorough` (2,000). The fixture `load_fixture` reads `validation/fixtures/<name>.json`. Warnings are errors, except the deprecations that other packages raise.

## Conventions

- Branch names are `t<id>-<slug>`, from `origin/main`. Rebase before review and before merge. Force-push only to your own branch, with `--force-with-lease`.
- Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/). The squash merge uses the pull request title as its subject, so the title follows them too.
- A pull request body has the sections of [the template](.github/PULL_REQUEST_TEMPLATE.md). Reviewers use the checklist under "Pull requests and reviews" in VALIDATION_PLAN.md.
- `ruff format` formats the code, and `ruff check` lints it; the settings are in `pyproject.toml`.
- Every choice made for the user goes in [dev/DECISIONS.md](dev/DECISIONS.md), with its reason.

## Legacy code

`validation/legacy/make_venv.sh` makes the ignored venv `.venv-legacy` with mars-earth 1.0.4 from PyPI. The scripts in `validation/legacy/` run with its Python.
