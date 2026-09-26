"""Deterministic seeding for the simulation harness.

VALIDATION_PLAN.md, "Build": each dataset's seed comes from its name (DGP,
sample size, noise level and repetition) through a stable hash, not Python's
builtin ``hash`` (randomized per process by ``PYTHONHASHSEED``, so it is not
stable across runs or processes), and ``numpy.random.default_rng``. The test
set comes from a second stream of the same seed. No seed depends on the
position in a loop, so the pilot repetitions are the first repetitions of the
full run: rerunning cell ``"D4_n00200_lo"`` repetition 7 anywhere, at any
time, gives the same training and test data.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(*parts: object) -> int:
    """A reproducible non-negative integer derived from ``parts``.

    Stable across runs, processes and interpreters, unlike the builtin
    ``hash()``. ``parts`` are joined with a separator byte that cannot appear
    in a ``str()`` of an int or name, so distinct part sequences never collide
    through concatenation (``("ab", "c")`` vs. ``("a", "bc")``).
    """
    text = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 8 bytes gives a seed well within the range numpy's SeedSequence accepts,
    # with negligible collision risk for the number of cells this study runs.
    return int.from_bytes(digest[:8], byteorder="big")


def cell_seed_sequence(cell_name: str, rep: int) -> np.random.SeedSequence:
    """The parent seed sequence for one (cell, repetition)."""
    return np.random.SeedSequence(stable_hash("cell", cell_name, rep))


def train_test_rngs(
    cell_name: str, rep: int
) -> tuple[np.random.Generator, np.random.Generator]:
    """The training-data generator and a second, independent stream for the
    10,000-point test set, both derived from one seed for ``(cell_name, rep)``.
    """
    train_seed, test_seed = cell_seed_sequence(cell_name, rep).spawn(2)
    return np.random.default_rng(train_seed), np.random.default_rng(test_seed)


def diagnostic_rng(dgp_name: str) -> np.random.Generator:
    """The fixed-seed generator for one DGP's one-million-case diagnostic
    draw (VALIDATION_PLAN.md, "Data-generating processes"). Independent of
    ``train_test_rngs``, so running the diagnostics again never perturbs any
    cell's training or test data.
    """
    return np.random.default_rng(stable_hash("diagnostic", dgp_name))
