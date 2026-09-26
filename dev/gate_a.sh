#!/usr/bin/env bash
# Gate A (VALIDATION_PLAN.md, "Gates"): ruff and the fast tests, in 60 s or less
# on one core. Run it before every commit, from any folder of a checkout.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
. dev/env.sh

uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen pytest -m "not slow" -x -q
