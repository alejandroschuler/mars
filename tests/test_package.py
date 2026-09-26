"""Tests of the package metadata."""

import importlib.metadata
import re

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
