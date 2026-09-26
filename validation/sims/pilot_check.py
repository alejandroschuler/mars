"""Pilot sizing for the full simulation run.

VALIDATION_PLAN.md, "Pilot and number of repetitions". A pilot of n_pilot
repetitions gives the mean g_bar_p and standard deviation s_p of the
per-repetition log ratios g_i (``metrics.log_ratio``) for one arm comparison.
This module turns that into the number of repetitions n_sim the full run
needs, for a difference contrast (the gap claim) or an equivalence contrast
(the parity claim), and the pilot statistic z that says whether the pilot
itself was informative enough to size anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EQUIVALENCE_MARGIN = math.log(1.05)  # Delta, on the log scale
DIFFERENCE_K = 5
EQUIVALENCE_K = 3
LOWER_BOUND_Z = 1.96


def from_log_ratios(g: Sequence[float]) -> tuple[float, float, int]:
    """g_bar_p, s_p (sample standard deviation, ddof=1) and n_pilot from an
    array of per-repetition log ratios g_i.
    """
    values = np.asarray(list(g), dtype=np.float64)
    n_pilot = len(values)
    if n_pilot < 2:
        raise ValueError("need at least 2 pilot repetitions to estimate s_p")
    return float(values.mean()), float(values.std(ddof=1)), n_pilot


def pilot_z(g_bar_pilot: float, s_pilot: float, n_pilot: int) -> float:
    """z = g_bar_p / (s_p / sqrt(n_pilot)): the pilot mean over its own
    standard error. |z| below about 2 means the pilot cannot tell the gap
    from zero, and no n_sim derived from it can be trusted.
    """
    se = s_pilot / math.sqrt(n_pilot)
    if se == 0.0:
        return math.inf if g_bar_pilot != 0.0 else 0.0
    return g_bar_pilot / se


@dataclass(frozen=True)
class DifferenceSizing:
    """Sizing for a difference contrast (the gap claim)."""

    n_sim: float  # (k * s_p / |g_bar_p|)^2; inf if g_bar_p == 0
    n_sim_safe: float  # using the 1.96-lower-bounded gap; inf when that bound is 0
    z: float
    k: int = DIFFERENCE_K

    @property
    def pilot_too_small(self) -> bool:
        """True when the pilot's own |z| <= 1.96, so n_sim_safe is infinite:
        the pilot cannot tell the gap from zero. VALIDATION_PLAN.md: "then
        the pilot grows or the design changes, and the report says so."
        """
        return math.isinf(self.n_sim_safe)


def difference_n_sim(
    g_bar_pilot: float, s_pilot: float, n_pilot: int, k: int = DIFFERENCE_K
) -> DifferenceSizing:
    """n_sim for a difference contrast: the full run is sized so the planned
    gap is k Monte Carlo standard errors, n_sim >= (k * s_p / |g_bar_p|)^2.
    ``n_sim_safe`` replaces |g_bar_p| with the lower bound
    max(0, |g_bar_p| - 1.96 * s_p / sqrt(n_pilot)), guarding against a pilot
    that overstates the gap by chance; it is infinite exactly when that bound
    is 0, i.e. when the pilot's own |z| <= 1.96.
    """
    z = pilot_z(g_bar_pilot, s_pilot, n_pilot)
    gap = abs(g_bar_pilot)
    n_sim = math.inf if gap == 0.0 else (k * s_pilot / gap) ** 2
    lower_bound = max(0.0, gap - LOWER_BOUND_Z * s_pilot / math.sqrt(n_pilot))
    n_sim_safe = math.inf if lower_bound == 0.0 else (k * s_pilot / lower_bound) ** 2
    return DifferenceSizing(n_sim=n_sim, n_sim_safe=n_sim_safe, z=z, k=k)


@dataclass(frozen=True)
class EquivalenceSizing:
    """Sizing for an equivalence contrast (the parity claim)."""

    n_sim: float  # the plan's headline number, (2k * s_p / margin)^2
    n_sim_at_pilot_gap: float  # (k * s_p / (margin - |g_bar_p|))^2 at the observed gap
    feasible: bool  # False when |g_bar_p| >= margin: no n_sim can show equivalence
    margin: float = EQUIVALENCE_MARGIN
    k: int = EQUIVALENCE_K


def equivalence_n_sim(
    g_bar_pilot: float,
    s_pilot: float,
    margin: float = EQUIVALENCE_MARGIN,
    k: int = EQUIVALENCE_K,
) -> EquivalenceSizing:
    """n_sim for an equivalence contrast. The plan's headline number sizes
    for an assumed pilot gap of margin/2, n_sim = (2k*s_p/margin)^2 -
    independent of the pilot's own point estimate, so a lucky pilot cannot
    make it optimistic. Separately, no n_sim at all can show equivalence once
    the pilot's |g_bar_p| reaches margin (``feasible=False``);
    ``n_sim_at_pilot_gap``, (k*s_p/(margin - |g_bar_p|))^2, is the sharper,
    pilot-gap-dependent requirement, reported alongside for context.
    """
    gap = abs(g_bar_pilot)
    n_sim = (2 * k * s_pilot / margin) ** 2
    feasible = gap < margin
    n_sim_at_pilot_gap = (k * s_pilot / (margin - gap)) ** 2 if feasible else math.inf
    return EquivalenceSizing(
        n_sim=n_sim,
        n_sim_at_pilot_gap=n_sim_at_pilot_gap,
        feasible=feasible,
        margin=margin,
        k=k,
    )


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["difference", "equivalence"])
    parser.add_argument(
        "g_values",
        nargs="*",
        type=float,
        help="per-repetition log ratios g_i; if omitted, read one per line from stdin",
    )
    args = parser.parse_args()
    values = args.g_values or [float(line) for line in sys.stdin if line.strip()]
    g_bar, s_p, n_pilot = from_log_ratios(values)
    if args.kind == "difference":
        sizing = difference_n_sim(g_bar, s_p, n_pilot)
        print(f"g_bar_p={g_bar:.4f} s_p={s_p:.4f} n_pilot={n_pilot} z={sizing.z:.3f}")
        print(f"n_sim={sizing.n_sim:.1f} n_sim_safe={sizing.n_sim_safe:.1f}")
        if sizing.pilot_too_small:
            print("pilot too small: |z| <= 1.96, cannot tell the gap from zero")
    else:
        sizing = equivalence_n_sim(g_bar, s_p)
        print(f"g_bar_p={g_bar:.4f} s_p={s_p:.4f} n_pilot={n_pilot}")
        print(f"n_sim={sizing.n_sim:.1f} (planning number; margin={sizing.margin:.4f})")
        print(f"n_sim_at_pilot_gap={sizing.n_sim_at_pilot_gap:.1f}")
        if not sizing.feasible:
            print("infeasible: |g_bar_p| >= margin, no n_sim can show equivalence")


if __name__ == "__main__":
    main()
