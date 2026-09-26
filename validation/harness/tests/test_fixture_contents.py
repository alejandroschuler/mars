"""Pure-Python checks on every committed fixture under validation/fixtures/
(datasets and components): T05 brief (issue #7), deliverable 5, "a
pure-Python test (runs in CI) that every fixture has the required fields, a
versions block and finite or explicitly marked values".

These fixtures are already-generated JSON files, so this needs no R and no
``Rscript`` call, unlike the ``external`` tests in test_gen_fixtures.py that
regenerate them for real; it runs everywhere, including gate A, gate B and
CI, the same as trace_parse.py's and names_map.py's own tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"

# fit_earth.R's and blackbox.R's write_result spell Inf/-Inf/NaN as these
# exact strings (na = "string"; see driver.py's module docstring) before
# driver.py's/blackbox.py's _desanitize() turns them back into real floats;
# every committed fixture is written only after that conversion, so one of
# these tokens surviving as a bare leaf value (outside a string-only field
# such as a term name, a factor level or an id) means desanitization did not
# happen, not a legitimate value.
_STRAY_SENTINELS = frozenset({"Inf", "-Inf", "NaN"})
_STRING_ONLY_KEYS = frozenset(
    {
        "id",
        "term_names",
        "levels",
        "r_version",
        "earth_version",
        "error",
        "call",
        "dataset",
        "mode",
        "component",
        "extra",
        "label",
        "warnings",
        "dirs_colnames",
        "steps_error",
    }
)

# The result fields every non-error dataset fixture must carry (driver.py's
# module docstring; CORE-3/CORE-5, docs/algorithm.md). glm_coef and levels
# are legitimately null outside a GLM/factor fit, so they are not required
# to be non-null, only present, which the "documented top-level shape" test
# below checks through the full set of keys driver.run_earth always writes.
_REQUIRED_RESULT_FIELDS = (
    "dirs",
    "cuts",
    "selected_terms",
    "rss_per_subset",
    "gcv_per_subset",
    "coef",
    "rss",
    "rsq",
    "gcv",
    "grsq",
    "termcond",
    "fitted",
    "pred_train",
    "r_version",
    "earth_version",
    "warnings",
    "glm_converged",
)

_REQUIRED_VERSION_FIELDS = (
    "python",
    "numpy",
    "scikit_learn",
    "r_version",
    "earth_version",
)

# Keys whose numeric value may legitimately be non-finite (review round 1,
# #42 adversarial finding 7): GCV-2 gives +inf where the effective number
# of parameters is at or above N (gcv, and any per-size entry of
# gcv_per_subset, including gcv_grid's own per-cell "gcv" list); GCV-7 and
# earth's own departure from it (docs/algorithm.md, "Departures from
# earth") give NaN rsq/grsq for a degenerate or near-degenerate fit.
# Everything else (inputs, weights, scale, coefficients, fitted values,
# predictions, rss) must be finite.
_MAY_BE_NONFINITE_KEYS = frozenset({"gcv", "grsq", "rsq", "gcv_per_subset"})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_level_fixture_paths() -> list[Path]:
    """Every ``validation/fixtures/*.json`` file: a (dataset, mode) fixture
    (a ``"dataset"`` key) or a bespoke "extra" one (an ``"extra"`` key,
    gen_fixtures.py's ``EXTRA_REGISTRY``, for example S15's 200 draws);
    which is which is decided by content, not by a filename guess."""
    return sorted(p for p in FIXTURES_DIR.glob("*.json") if p.is_file())


def _dataset_fixture_paths() -> list[Path]:
    return [p for p in _top_level_fixture_paths() if "dataset" in _load(p)]


def _extra_fixture_paths() -> list[Path]:
    return [p for p in _top_level_fixture_paths() if "extra" in _load(p)]


def _component_fixture_paths() -> list[Path]:
    components_dir = FIXTURES_DIR / "components"
    return sorted(components_dir.glob("*.json")) if components_dir.is_dir() else []


def _all_fixture_paths() -> list[Path]:
    return _top_level_fixture_paths() + _component_fixture_paths()


def _walk_leaves(value: Any, under_string_only_key: bool) -> Iterator[tuple[Any, bool]]:
    """Every leaf (non-dict, non-list) value in a JSON-shaped structure,
    paired with whether it sits under a string-only key (directly, or
    nested inside one, for example a list of term names)."""
    if isinstance(value, dict):
        for key, v in value.items():
            nested = under_string_only_key or key in _STRING_ONLY_KEYS
            yield from _walk_leaves(v, nested)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_leaves(v, under_string_only_key)
    else:
        yield value, under_string_only_key


def _walk_numeric_leaves(
    value: Any, enclosing_key: str | None
) -> Iterator[tuple[float, str | None]]:
    """Every numeric (int or float, not bool) leaf, paired with the
    nearest enclosing dict key (a list item inherits its own key, so
    gcv_per_subset's entries all carry "gcv_per_subset")."""
    if isinstance(value, dict):
        for key, v in value.items():
            yield from _walk_numeric_leaves(v, key)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_numeric_leaves(v, enclosing_key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value, enclosing_key


def test_at_least_one_dataset_and_one_component_fixture_are_committed():
    assert _dataset_fixture_paths(), f"no dataset fixtures committed in {FIXTURES_DIR}"
    assert _component_fixture_paths(), (
        f"no component fixtures committed in {FIXTURES_DIR / 'components'}"
    )
    assert _extra_fixture_paths(), f"no extra fixtures committed in {FIXTURES_DIR}"


def test_every_fixture_is_valid_json():
    for path in _all_fixture_paths():
        _load(path)  # json.loads raises if this fixture is not valid JSON


def test_every_fixture_has_a_complete_versions_block():
    for path in _all_fixture_paths():
        versions = _load(path).get("versions")
        assert versions, f"{path.name}: no versions block"
        for key in _REQUIRED_VERSION_FIELDS:
            assert versions.get(key), f"{path.name}: versions.{key} is missing or empty"


def test_no_undesanitized_sentinel_strings_leak_through():
    for path in _all_fixture_paths():
        for leaf, under_string_only_key in _walk_leaves(_load(path), False):
            if under_string_only_key:
                continue
            assert leaf not in _STRAY_SENTINELS, (
                f"{path.name}: an un-desanitized {leaf!r} leaked through as a "
                "bare value (driver.py's/blackbox.py's _desanitize should "
                "have turned this into a real float already)"
            )


def test_every_numeric_value_is_finite_or_explicitly_marked():
    # Review round 1 (#42 adversarial finding 7): T05 brief, deliverable 5,
    # "finite or explicitly marked values" -- checked directly on the
    # numbers, not just on sentinel-string decoding (which a NaN or an
    # unmarked Infinity never touches in the first place).
    for path in _all_fixture_paths():
        for value, key in _walk_numeric_leaves(_load(path), None):
            if key in _MAY_BE_NONFINITE_KEYS:
                continue
            assert math.isfinite(value), (
                f"{path.name}: {key!r} holds the non-finite value {value!r}, "
                "not one of the fields explicitly allowed to "
                f"({sorted(_MAY_BE_NONFINITE_KEYS)})"
            )


def test_every_dataset_fixture_has_the_documented_top_level_shape():
    for path in _dataset_fixture_paths():
        payload = _load(path)
        for key in ("dataset", "mode", "inputs", "earth_args", "result", "versions"):
            assert key in payload, f"{path.name}: missing {key!r}"
        result = payload["result"]
        if "error" in result:
            # VALIDATION_PLAN.md, "Missing values"/T05 brief: where earth
            # errors on an input, the fixture records the error instead of
            # the usual result fields, an explicit marking rather than a
            # silently missing or non-finite value.
            assert result["error"], f"{path.name}: result.error is empty"
            assert result.get("r_version"), (
                f"{path.name}: an error result still needs r_version"
            )
            assert result.get("earth_version"), (
                f"{path.name}: an error result still needs earth_version"
            )
            continue
        for key in _REQUIRED_RESULT_FIELDS:
            assert key in result, f"{path.name}: result is missing {key!r}"


def test_every_component_fixture_names_itself():
    for path in _component_fixture_paths():
        payload = _load(path)
        assert payload.get("component") == path.stem, (
            f"{path.name}: payload['component'] is {payload.get('component')!r}, "
            "not this file's own name"
        )


def test_every_extra_fixture_names_itself():
    for path in _extra_fixture_paths():
        payload = _load(path)
        assert payload.get("extra") == path.stem, (
            f"{path.name}: payload['extra'] is {payload.get('extra')!r}, "
            "not this file's own name"
        )


# Review round 2 (#43 adversarial, non-blocking): "_fixture_payload ignores
# scale_override" and the several other ways a weighted/repeated pair could
# quietly drift apart "survive every test outside gate C" (which needs R).
# This one reads the committed JSON directly, so it runs in gate A, gate B
# and CI too, not only a `gen_fixtures.py --check` on a machine with R.
_WEIGHTED_REPEATED_PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("S13_int_zeros", "S13_int_zeros_repeated", ("matched_d1", "defaults_d1")),
    ("S13_int_random", "S13_int_random_repeated", ("matched_d1", "defaults_d1")),
    ("S13_unit", "S13_unit_repeated", ("matched_d1", "defaults_d1")),
    ("S13_equal2", "S13_equal2_repeated", ("matched_d1", "defaults_d1")),
    (
        "S13_constant_y_weighted",
        "S13_constant_y_weighted_repeated",
        ("matched_d1", "defaults_d1"),
    ),
    (
        "S16_weighted",
        "S16_weighted_repeated",
        ("matched_d1", "defaults_d1", "defaults_d2", "matched_d2"),
    ),
)


def test_committed_weighted_and_repeated_pairs_are_bit_identical():
    for weighted_id, repeated_id, modes in _WEIGHTED_REPEATED_PAIRS:
        for mode in modes:
            w_path = FIXTURES_DIR / f"{weighted_id}_{mode}.json"
            r_path = FIXTURES_DIR / f"{repeated_id}_{mode}.json"
            if not w_path.is_file() or not r_path.is_file():
                continue  # a constant-y case may hold an earth error instead
            w_payload, r_payload = _load(w_path), _load(r_path)
            w_in, r_in = w_payload["inputs"], r_payload["inputs"]
            label = f"{weighted_id}/{repeated_id} ({mode})"

            assert w_payload["scale"] == r_payload["scale"], (
                f"{label}: scale is not shared exactly"
            )

            weights = [int(x) for x in w_in["weights"]]
            rows = zip(w_in["X"], weights, strict=True)
            expanded_X = [row for row, w in rows if w for _ in range(w)]
            assert expanded_X == r_in["X"], (
                f"{label}: repeating the weighted X by its weights does not "
                "give the repeated fixture's X exactly"
            )
            ys = zip(w_in["y"], weights, strict=True)
            expanded_y = [y for y, w in ys if w for _ in range(w)]
            assert expanded_y == r_in["y"], (
                f"{label}: repeating the weighted y by its weights does not "
                "give the repeated fixture's y exactly"
            )
            assert w_in["X_test"] == r_in["X_test"], (
                f"{label}: X_test is not shared exactly"
            )
