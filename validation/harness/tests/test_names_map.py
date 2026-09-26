"""Pure-Python tests for names_map.py; no R needed."""

import pytest
from names_map import EARTH_NAMES, to_earth_args


def test_every_public_api_parameter_is_mapped():
    # VALIDATION_PLAN.md, "Public API": every earth-related constructor
    # parameter of EarthRegressor/EarthClassifier must have an entry here.
    expected = {
        "max_degree",
        "max_terms",
        "penalty",
        "thresh",
        "minspan",
        "endspan",
        "adjust_endspan",
        "auto_linpreds",
        "fast_k",
        "fast_beta",
        "pmethod",
        "nprune",
    }
    assert set(EARTH_NAMES) == expected


def test_concrete_values_pass_through_unchanged():
    args = to_earth_args(
        max_degree=2,
        penalty=3.0,
        thresh=0.001,
        pmethod="backward",
        fast_k=0,
        fast_beta=0.0,
        adjust_endspan=1.0,
        auto_linpreds=False,
    )
    assert args == {
        "degree": 2,
        "penalty": 3.0,
        "thresh": 0.001,
        "pmethod": "backward",
        "fast.k": 0,
        "fast.beta": 0.0,
        "Adjust.endspan": 1.0,
        "Auto.linpreds": False,
    }


def test_none_is_omitted_for_most_parameters():
    # Letting earth compute its own default (for example nk's or penalty's
    # formula) needs no value from this module at all.
    args = to_earth_args(max_terms=None, penalty=None, nprune=None, max_degree=1)
    assert args == {"degree": 1}


@pytest.mark.parametrize(
    "name,earth_name", [("minspan", "minspan"), ("endspan", "endspan")]
)
def test_none_span_becomes_earths_own_automatic_zero(name, earth_name):
    # ?earth (read locally): minspan/endspan's own default is the literal 0,
    # its sentinel for "calculate this internally".
    args = to_earth_args(**{name: None})
    assert args == {earth_name: 0}


def test_explicit_span_value_passes_through():
    args = to_earth_args(minspan=5, endspan=10)
    assert args == {"minspan": 5, "endspan": 10}


def test_unknown_parameter_name_is_rejected():
    with pytest.raises(ValueError, match="newvar_penalty"):
        to_earth_args(newvar_penalty=0.1)


def test_empty_call_gives_empty_args():
    assert to_earth_args() == {}
