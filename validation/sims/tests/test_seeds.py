"""Tests of validation/sims/seeds.py.

VALIDATION_PLAN.md, "Build": seeds must be stable across runs and processes
(not Python's randomized ``hash()``), and the test set must come from a
second, independent stream of the training seed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.sims import seeds

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stable_hash_is_deterministic_within_a_process():
    assert seeds.stable_hash("D4", 200, "lo", 3) == seeds.stable_hash(
        "D4", 200, "lo", 3
    )


def test_stable_hash_distinguishes_its_arguments():
    values = {seeds.stable_hash("D4", 200, "lo", rep) for rep in range(20)}
    assert len(values) == 20  # no collisions among 20 small, related inputs

    # Concatenation cannot make two different part sequences collide: the
    # separator byte (0x1f) cannot appear in str(int) or a DGP name.
    assert seeds.stable_hash("ab", "c") != seeds.stable_hash("a", "bc")


def test_stable_hash_matches_across_processes():
    """Not Python's builtin ``hash()``, which is randomized per process by
    PYTHONHASHSEED and would make this test flaky.
    """
    expected = seeds.stable_hash("D4", 200, "lo", 3)
    script = (
        "from validation.sims import seeds; "
        "print(seeds.stable_hash('D4', 200, 'lo', 3))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    assert int(out.stdout.strip()) == expected


def test_train_test_rngs_are_deterministic_and_independent():
    train1, test1 = seeds.train_test_rngs("D4_n00200_lo", 0)
    train2, test2 = seeds.train_test_rngs("D4_n00200_lo", 0)
    assert np.array_equal(train1.uniform(size=10), train2.uniform(size=10))
    assert np.array_equal(test1.uniform(size=10), test2.uniform(size=10))

    train, test = seeds.train_test_rngs("D4_n00200_lo", 0)
    train_draw = train.uniform(size=1000)
    test_draw = test.uniform(size=1000)
    assert not np.array_equal(train_draw, test_draw)


def test_seed_depends_on_cell_name_and_rep_not_loop_position():
    train_a, _ = seeds.train_test_rngs("D4_n00200_lo", 5)
    train_b, _ = seeds.train_test_rngs("D4_n00200_lo", 5)
    assert np.array_equal(train_a.uniform(size=5), train_b.uniform(size=5))

    train_rep5, _ = seeds.train_test_rngs("D4_n00200_lo", 5)
    train_rep6, _ = seeds.train_test_rngs("D4_n00200_lo", 6)
    assert not np.array_equal(train_rep5.uniform(size=5), train_rep6.uniform(size=5))

    train_d4, _ = seeds.train_test_rngs("D4_n00200_lo", 0)
    train_d5, _ = seeds.train_test_rngs("D5_n00200_lo", 0)
    assert not np.array_equal(train_d4.uniform(size=5), train_d5.uniform(size=5))


def test_diagnostic_rng_is_deterministic_and_differs_by_name():
    a1 = seeds.diagnostic_rng("D3")
    a2 = seeds.diagnostic_rng("D3")
    b = seeds.diagnostic_rng("D4")
    assert np.array_equal(a1.uniform(size=10), a2.uniform(size=10))
    assert not np.array_equal(a1.uniform(size=10), b.uniform(size=10))
