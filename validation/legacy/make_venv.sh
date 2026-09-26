#!/usr/bin/env bash
# Make the ignored venv .venv-legacy at the repository root with the legacy code:
# the mars-earth 1.0.4 wheel from PyPI, with the versions that the preliminary
# findings used (VALIDATION_PLAN.md, "Appendix: bootstrap commands and prompts").
# Safe to run again. Usage: validation/legacy/make_venv.sh
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
. "$root/dev/env.sh"
venv=$root/.venv-legacy

[ -x "$venv/bin/python" ] || uv venv "$venv" --python 3.12
uv pip install --python "$venv/bin/python" \
  "mars-earth==1.0.4" "scikit-learn==1.9.1" "numpy==2.5.3" pandas
"$venv/bin/python" -I -c 'import importlib.metadata as m, pymars
print("mars-earth", m.version("mars-earth"), "imports from", pymars.__file__)'
