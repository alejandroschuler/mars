#!/usr/bin/env bash
# Gate A (VALIDATION_PLAN.md, "Gates"): ruff and the fast tests, in 60 s or less
# on one core. Run it before every commit, from any folder of a checkout.
# testpaths also covers validation/sims/tests and validation/harness/tests;
# "external" (needs R/earth or .venv-legacy) is left to gate C and a direct
# pytest invocation.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
. dev/env.sh

uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen --group validation pytest -m "not slow and not external" -x -q
