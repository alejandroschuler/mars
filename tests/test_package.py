"""Tests of the package metadata."""

import importlib.metadata

from packaging.requirements import Requirement
from packaging.version import Version

import pymars


def test_version_is_a_normalized_pep440_string():
    assert str(Version(pymars.__version__)) == pymars.__version__


def test_version_equals_the_installed_metadata():
    assert pymars.__version__ == importlib.metadata.version("mars-earth")


def test_runtime_dependencies_are_numpy_scipy_and_scikit_learn():
    requirements = [Requirement(r) for r in importlib.metadata.requires("mars-earth")]
    names = {r.name for r in requirements if r.marker is None}
    assert names == {"numpy", "scipy", "scikit-learn"}
