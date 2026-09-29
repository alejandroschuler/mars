"""Tests of the package metadata."""

import importlib.metadata
import re

import numpy as np

import pymars

# The canonical form of a PEP 440 version (PEP 440, appendix B).
CANONICAL_VERSION = re.compile(
    r"([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?"
    r"(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?"
)


def test_version_is_a_canonical_pep440_string():
    assert CANONICAL_VERSION.fullmatch(pymars.__version__)


def test_version_equals_the_installed_metadata():
    assert pymars.__version__ == importlib.metadata.version("mars-earth")


def test_runtime_dependencies_are_numpy_scipy_and_scikit_learn():
    requirements = importlib.metadata.requires("mars-earth") or []
    # The distribution name starts each requirement; a marker follows ";".
    names = {
        re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9._-]+", r).group()).lower()
        for r in requirements
        if ";" not in r
    }
    assert names == {"numpy", "scipy", "scikit-learn"}


def test_matmul_agrees_with_a_loop_at_the_shape_of_a_hinge_projection():
    """The BLAS that numpy uses computes A.T @ B correctly at the shape of the
    projections of a knot search (LA-1): n = 300 cases, 14 basis columns and
    298 hinge columns. numpy 1.23.5 bundles OpenBLAS 0.3.20, whose Cooperlake
    kernel gets the columns from 160 on wrong at this shape, and the fits then
    take other terms (issue #88); the numpy floor 1.24.4 in dev/DECISIONS.md
    excludes it. The loop is numpy's einsum without BLAS."""
    rng = np.random.default_rng(88)
    A = rng.standard_normal((300, 14))
    B = np.maximum(rng.uniform(size=(300, 1)) - np.linspace(1, 0, 298)[None, :], 0.0)
    loop = np.einsum("ij,ik->jk", A, B, optimize=False)
    # a bound on the rounding error of each column: 1e-12 of its sum of |a_i b_i|
    bound = 1e-12 * np.einsum("ij,ik->jk", np.abs(A), B, optimize=False).max(axis=0)
    wrong = np.flatnonzero(np.abs(A.T @ B - loop).max(axis=0) > bound)
    assert wrong.size == 0, f"A.T @ B is wrong in the columns {wrong.tolist()}"
