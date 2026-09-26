"""Shared test setup: one BLAS thread, the hypothesis profiles and a fixture loader."""

import json
import os
from pathlib import Path

import pytest
from hypothesis import settings
from threadpoolctl import threadpool_limits

# One thread per BLAS and OpenMP library. The variables must be set before
# numpy loads, because Accelerate (macOS) reads them only then and threadpoolctl
# cannot change it later. dev/env.sh sets the same variables for every process.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "validation" / "fixtures"

# HYPOTHESIS_PROFILE picks the profile: dev (the default), ci (gate B and CI) or
# thorough (gate C). There is no deadline, because fit times vary with the load.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("thorough", max_examples=2000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE") or "dev")


@pytest.fixture(scope="session", autouse=True)
def _one_blas_thread():
    """Limit the libraries that threadpoolctl can control to one thread."""
    with threadpool_limits(limits=1):
        yield


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; the earth harness makes it")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def load_fixture():
    """Return a function that reads ``validation/fixtures/<name>.json``."""
    return _load_fixture
