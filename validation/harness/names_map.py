"""The table from pymars 2.0 parameter names to earth arguments.

``VALIDATION_PLAN.md``'s "Public API" section fixes the pymars names; its
"Behavior target" table says pymars' ``None`` means earth's own automatic
value. This module only maps names and resolves that ``None`` convention. It
never reproduces one of earth's internal default *formulas* (``nk``'s or
``penalty``'s, for example): omitting an argument already asks earth to
compute its own default, which is simpler and does not depend on reading any
earth internals.

The one exception is ``minspan``/``endspan``: earth's own default for both is
the literal value 0, its documented sentinel for "calculate this
internally" (read locally from ``?earth``, not quoted here, per
``VALIDATION_PLAN.md``, "Instruction files and clean room").

``docs/algorithm.md`` (written in parallel by the spec task) carries the
final name table; T07 checks this one against it, since a name here could
still move, and later parameters that the plan defers to `later` issues
(``newvar.penalty``, ``linpreds``, ``allowed``, a negative ``minspan``,
``nfold``) are not in this table at all yet.
"""

from __future__ import annotations

from typing import Any

# pymars parameter name -> earth argument name (VALIDATION_PLAN.md, "Public
# API" and "Correctness against earth"). Identically-named arguments
# (penalty, thresh, minspan, endspan, pmethod, nprune) are listed too, so this
# table is the single place that answers "what does earth call this".
EARTH_NAMES: dict[str, str] = {
    "max_degree": "degree",
    "max_terms": "nk",
    "penalty": "penalty",
    "thresh": "thresh",
    "minspan": "minspan",
    "endspan": "endspan",
    "adjust_endspan": "Adjust.endspan",
    "auto_linpreds": "Auto.linpreds",
    "fast_k": "fast.k",
    "fast_beta": "fast.beta",
    "pmethod": "pmethod",
    "nprune": "nprune",
}

# The pymars names whose None resolves to earth's own explicit automatic
# value (0) rather than to omitting the argument.
AUTOMATIC_IS_ZERO: frozenset[str] = frozenset({"minspan", "endspan"})


def to_earth_args(**pymars_params: Any) -> dict[str, Any]:
    """Build an ``earth_args`` dict (``driver.EarthJob.earth_args``) from
    pymars-style parameter names and values.

    Every keyword must be a key of ``EARTH_NAMES``. A ``None`` value is
    dropped, letting earth compute its own default, except for ``minspan``
    and ``endspan``, whose ``None`` becomes earth's own literal automatic
    value, 0.
    """
    unknown = set(pymars_params) - set(EARTH_NAMES)
    if unknown:
        raise ValueError(f"not in the pymars-to-earth name table: {sorted(unknown)}")
    earth_args: dict[str, Any] = {}
    for name, value in pymars_params.items():
        earth_name = EARTH_NAMES[name]
        if value is None:
            if name in AUTOMATIC_IS_ZERO:
                earth_args[earth_name] = 0
            continue
        earth_args[earth_name] = value
    return earth_args
