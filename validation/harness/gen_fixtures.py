"""The fixture generator framework (VALIDATION_PLAN.md, "Correctness
against earth"): a registry of datasets and a registry of earth argument
sets ("modes"), which together make `validation/fixtures/<dataset>_<mode>
.json`, one file per (dataset, mode) pair, each with that pair's inputs,
earth arguments, `driver.run_earth` result and versions.

This registers one dataset, S01 ("one covariate, two true knots, 200
cases"), under two modes, to prove the pipeline; T05 adds S02 through S20.

Usage (also validation/README.md):
    python validation/harness/gen_fixtures.py           # write every fixture
    python validation/harness/gen_fixtures.py --check   # regenerate into a
                                                         # temp folder and
                                                         # report any diff
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class CheckResult:
    """``check()``'s outcome: ``problems`` (a difference in what a fixture
    claims about earth: its inputs, earth arguments or result, or a
    fixture missing outright) fail the check; ``notes`` (a ``versions``
    difference: this machine's R, earth, numpy, scikit-learn or BLAS is not
    the one that made the committed fixture) do not, since nothing about
    earth conformance follows from running the generator on different
    software versions.
    """

    problems: list[str]
    notes: list[str]


@dataclass
class Dataset:
    """One registered dataset: everything `driver.EarthJob` needs besides
    the earth arguments (which a mode supplies)."""

    id: str
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray | None = None
    factor_response: bool = False


DatasetFn = Callable[[], Dataset]
REGISTRY: dict[str, DatasetFn] = {}

# earth argument sets ("modes"), VALIDATION_PLAN.md's "Comparison modes":
# defaults_d1 is earth at its own defaults; matched_d1 is the legacy-code
# matched mode from the seed prototype (validation/legacy/compare_earth.py),
# hinge-only and every case a candidate knot, so the two implementations
# should make the same forward choices except at near-ties.
MODES: dict[str, dict[str, Any]] = {
    "defaults_d1": {"degree": 1},
    "matched_d1": {
        "degree": 1,
        "penalty": 2,
        "nk": 21,
        "thresh": 0,
        "minspan": 1,
        "endspan": 1,
        "fast.k": 0,
        "Auto.linpreds": False,
        "pmethod": "backward",
    },
}


def register(fn: DatasetFn) -> DatasetFn:
    """Decorator: register a dataset generator under its own return value's
    ``id`` (calling it once, at import time, to read that id)."""
    REGISTRY[fn().id] = fn
    return fn


@register
def s01() -> Dataset:
    """S01: one covariate, two true knots, 200 cases. Knot recovery."""
    rng = np.random.default_rng(1)
    n, n_test = 200, 50
    x = rng.uniform(0, 1, size=(n, 1))
    x_test = rng.uniform(0, 1, size=(n_test, 1))

    def f(x0: np.ndarray) -> np.ndarray:
        return 2 * np.maximum(0, x0 - 0.3) - 3 * np.maximum(0, x0 - 0.7)

    y = f(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return Dataset(id="S01", X=x, y=y, X_test=x_test)


def _sanitize_versions(versions: dict[str, Any]) -> dict[str, Any]:
    """Drop the parts of ``driver.versions()`` that vary by machine or by
    commit without saying anything about earth conformance: an absolute
    path (threadpoolctl's and R's BLAS ``filepath``/``la_library`` include
    the machine's home folder, which also does not belong in a public
    repository) becomes just its basename, and ``pymars_commit`` is left
    out of the fixture entirely (``check()`` reports it separately; a
    fixture regenerated on a later commit always differs there, by
    design).
    """
    out = dict(versions)
    out.pop("pymars_commit", None)
    if out.get("blas"):
        out["blas"] = [
            {
                **lib,
                "filepath": Path(lib["filepath"]).name if lib.get("filepath") else None,
            }
            for lib in out["blas"]
        ]
    if out.get("r_blas"):
        out["r_blas"] = {
            key: (Path(value).name if value else None)
            for key, value in out["r_blas"].items()
        }
    return out


def _fixture_payload(dataset_id: str, mode: str) -> dict[str, Any]:
    ds = REGISTRY[dataset_id]()
    earth_args = MODES[mode]
    job = driver.EarthJob(
        id=f"{dataset_id}_{mode}",
        X=ds.X,
        y=ds.y,
        X_test=ds.X_test,
        earth_args=earth_args,
        factor_response=ds.factor_response,
    )
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-") as tmp:
        result = driver.run_earth([job], workdir=Path(tmp))[job.id]
    return {
        "dataset": dataset_id,
        "mode": mode,
        "inputs": {
            "X": ds.X.tolist(),
            "y": ds.y.tolist(),
            "X_test": ds.X_test.tolist() if ds.X_test is not None else None,
        },
        "earth_args": earth_args,
        "result": result,
        "versions": _sanitize_versions(driver.versions(result)),
    }


def _write_fixture(dataset_id: str, mode: str, fixtures_dir: Path) -> Path:
    payload = _fixture_payload(dataset_id, mode)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{dataset_id}_{mode}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def make_all(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered (dataset, mode) fixture and return their
    paths, in registration order.

    ``fixtures_dir`` defaults to the module-level ``FIXTURES_DIR``, read
    inside this function (not bound as a default parameter value), so
    ``monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", ...)`` in a test
    changes what a plain ``make_all()`` call does.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    return [
        _write_fixture(dataset_id, mode, fixtures_dir)
        for dataset_id in REGISTRY
        for mode in MODES
    ]


def check(*, fixtures_dir: Path | None = None) -> CheckResult:
    """Regenerate every fixture into a temporary folder and compare each one
    against ``fixtures_dir`` (default: the module-level ``FIXTURES_DIR``,
    read dynamically the same way ``make_all`` does).

    Only ``inputs``, ``earth_args`` and ``result`` (what a fixture claims
    about earth) or a fixture missing outright can add to ``.problems``; a
    ``versions`` difference (this machine's R, earth, numpy, scikit-learn
    or BLAS is not the one that made the committed fixture) instead adds to
    ``.notes``, since ``_sanitize_versions`` already drops the one field
    (``pymars_commit``) that would differ on every run from a different
    commit by construction, and nothing about earth conformance follows
    from the rest of ``versions`` differing.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    problems: list[str] = []
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-check-") as tmp:
        fresh_paths = make_all(fixtures_dir=Path(tmp))
        for fresh in fresh_paths:
            committed = fixtures_dir / fresh.name
            if not committed.is_file():
                problems.append(f"{fresh.name}: missing from {fixtures_dir}")
                continue
            fresh_payload = json.loads(fresh.read_text(encoding="utf-8"))
            committed_payload = json.loads(committed.read_text(encoding="utf-8"))
            checked = ("dataset", "mode", "inputs", "earth_args", "result")
            if {k: fresh_payload.get(k) for k in checked} != {
                k: committed_payload.get(k) for k in checked
            }:
                problems.append(f"{fresh.name}: differs from the committed fixture")
            elif fresh_payload.get("versions") != committed_payload.get("versions"):
                notes.append(f"{fresh.name}: versions differ")
    return CheckResult(problems=problems, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="remake every fixture in a temp folder and report any diff",
    )
    args = parser.parse_args(argv)

    if args.check:
        result = check()
        for message in result.notes:
            print(message, file=sys.stderr)
        for message in result.problems:
            print(message, file=sys.stderr)
        if result.problems:
            print(
                f"{len(result.problems)} fixture(s) did not reproduce exactly",
                file=sys.stderr,
            )
            return 1
        print("every fixture reproduced exactly")
        return 0

    paths = make_all()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
