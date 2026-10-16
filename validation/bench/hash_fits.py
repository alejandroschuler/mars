"""Bit-equality check for T20's speed passes: a hash of every fast fit.

Each pass must leave the fits bit-equal to main. This script fits the fast
code (``pymars._core.fit_mars`` with the candidate log) on the dataset
fixtures, on the 200 draws of S15, and on 200 random small fits of its own
(weights, shifted covariates, tied values, degrees 1 to 3), and hashes each
``MarsFit.to_dict()`` as JSON (floats by ``repr`` and arrays by their bytes,
both exact). It writes the hashes to ``--out``; with ``--compare`` it lists
the cases whose hash differs from those of an earlier run and exits 1 if
there is one.

    . dev/env.sh
    nice -n 15 uv run --frozen --group validation python \\
        validation/bench/hash_fits.py --out <file> [--compare <file of main>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "tests", ROOT / "validation" / "harness", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _bits(v):
    """An array as its dtype, shape and bytes, so that equal means bit-equal."""
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]
    if isinstance(v, np.generic):
        return _bits(np.asarray(v))
    raise TypeError(f"cannot hash {type(v).__name__}")


def _digest(d: dict) -> str:
    text = json.dumps(d, sort_keys=True, allow_nan=True, default=_bits)
    return hashlib.sha256(text.encode()).hexdigest()


def random_cases(count: int = 200, seed: int = 22):
    """Small fits that reach the shifted, weighted and tied paths."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        n, p = int(rng.integers(20, 301)), int(rng.integers(2, 7))
        X = rng.uniform(0.0, 1.0, (n, p))
        if rng.random() < 0.3:  # tied values
            X[:, 0] = np.round(X[:, 0] * 5) / 5
        if rng.random() < 0.4:  # a covariate with a large mean
            X[:, 1] += 10.0 ** rng.integers(3, 11)
        y = np.sin(3 * X[:, 0]) + (X[:, -1] - 0.5) ** 2 + 0.1 * rng.standard_normal(n)
        w = rng.integers(1, 4, n).astype(float) if rng.random() < 0.4 else None
        params = {
            "max_degree": int(rng.integers(1, 4)),
            "max_terms": int(rng.choice([11, 21, 31])),
        }
        yield f"random/{i:03d}", X, y, w, params


def all_cases():
    from test_conformance import DATASETS, load_case, s15_cases

    for name in DATASETS:
        c = load_case(name)
        yield f"fixture/{name}", c.X, c.Y, (c.w if c.weighted else None), c.params
    for _, c in s15_cases():
        yield c.name, c.X, c.Y, None, c.params
    yield from random_cases()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args()
    from pymars import _core

    hashes: dict[str, str] = {}
    for name, X, Y, w, params in all_cases():
        try:
            fit = _core.fit_mars(
                X, Y, w, _core.MarsParams(**params), record_candidates=True
            )
        except Exception as e:  # an error is a result too; it must not change
            hashes[name] = f"error: {type(e).__name__}: {e}"
            continue
        hashes[name] = _digest(fit.to_dict())
    total = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    Path(a.out).write_text(json.dumps({"total": total, "cases": hashes}, indent=1))
    print(f"{len(hashes)} fits, combined hash {total}")
    if a.compare:
        old = json.loads(Path(a.compare).read_text())["cases"]
        diff = sorted(
            k for k in hashes.keys() | old.keys() if hashes.get(k) != old.get(k)
        )
        print(f"{len(diff)} differ from {a.compare}")
        for k in diff[:30]:
            print("  ", k)
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
