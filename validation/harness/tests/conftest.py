"""Put ``validation/harness/`` on ``sys.path`` so its tests can import its
modules directly (``import driver``, ``import trace_parse``, ...), the same
way a script under ``validation/harness/`` would when run directly.

These tests live outside ``tests/`` because gate A, gate B and CI have no R
(``VALIDATION_PLAN.md``, "Tests and validation folders"): run them with
``uv run --frozen --group validation pytest validation/harness/tests``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
