"""Put ``validation/harness/`` on ``sys.path`` so its tests can import its
modules directly (``import driver``, ``import trace_parse``, ...), the same
way a script under ``validation/harness/`` would when run directly.

These tests live outside ``tests/`` because CI has no R
(``VALIDATION_PLAN.md``, "Tests and validation folders"), but they are in
pytest's ``testpaths``, so a plain ``pytest`` run from the repository root
collects them too. Only the tests marked ``external`` (they call ``Rscript``
or the legacy venv) are excluded from gate A, gate B and CI
(``-m "not external"``); the rest run everywhere. Run everything, including
the external tests: ``uv run --frozen --group validation pytest
validation/harness/tests``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
