"""The core of pymars: the parameters, the fit record and ``fit_mars``.

Spec: ``docs/algorithm.md``, "Core API" (CORE-1 to CORE-7). This part holds
``MarsParams`` (CORE-2), the records ``MarsFit``, ``ForwardRecord`` and
``PruningRecord`` (CORE-3), with ``_forward``'s ``CandidateLog`` (CORE-3) and
``Termination`` (CORE-4), and their dict form (CORE-5). ``fit_mars``
(CORE-1) comes in part 2 of T12 (issue #14).

Records. ``MarsFit``, ``ForwardRecord`` and ``PruningRecord`` are frozen
dataclasses (CORE-3). ``ForwardRecord`` is a frozen dataclass made from
``_forward.ForwardPass``, with the same fields, so that every record of a fit
has the same checks and the same dict form; ``CandidateLog`` and
``Termination`` are ``_forward``'s own, one definition of each. The
constructors check the dtypes and shapes of CORE-3 and store new read-only
arrays, so a fit shares no memory with its inputs. ``to_dict`` and
``from_dict`` give the dict form of CORE-5, which the reference returns.

Numerics: float64 throughout; no function writes into its inputs.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from pymars import _pruning
from pymars._forward import CandidateLog, Termination

__all__ = [
    "CandidateLog",
    "ForwardRecord",
    "MarsFit",
    "MarsParams",
    "PruningRecord",
    "Termination",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: CORE-2: the kind, the lower bound (None: no bound) and whether None is
#: allowed, per field.
_PARAMS: dict[str, tuple[str, float | None, bool]] = {
    "max_degree": ("int", 1, False),
    "max_terms": ("int", 1, True),
    "penalty": ("penalty", 0.0, True),
    "thresh": ("float", 0.0, False),
    "minspan": ("int", 1, True),
    "endspan": ("int", 1, True),
    "adjust_endspan": ("float", 0.0, False),
    "auto_linpreds": ("bool", None, False),
    "fast_k": ("int", 0, False),
    "fast_beta": ("float", 0.0, False),
    "pmethod": ("str", None, False),
    "nprune": ("int", 1, True),
}


def _param(name: str, value: Any) -> Any:
    """Return a field of ``MarsParams`` as a plain Python value, or raise
    ValueError that names the field (CORE-2)."""
    kind, low, optional = _PARAMS[name]
    if value is None and optional:
        return None
    where = f"MarsParams.{name}"
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{where} must be an integer, not {value!r}")
        if value < low:
            raise ValueError(f"{where} must be at least {low}, not {value!r}")
        return int(value)
    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{where} must be a bool, not {value!r}")
        return bool(value)
    if kind == "str":
        if not isinstance(value, str) or value not in _pruning.PMETHODS:
            raise ValueError(f"{where} must be 'backward' or 'none', not {value!r}")
        return str(value)
    if not isinstance(value, numbers.Real):  # CORE-2: any real number type
        raise ValueError(f"{where} must be a real number, not {value!r}")
    try:
        x = float(value)
    except OverflowError:
        x = math.inf
    if kind == "penalty" and x == -1.0:
        return x
    if not (math.isfinite(x) and x >= low):
        also = "-1, or " if kind == "penalty" else ""
        raise ValueError(f"{where} must be {also}finite and at least 0, not {value!r}")
    return x


@dataclasses.dataclass(frozen=True)
class MarsParams:
    """The parameters of a fit (CORE-2). None means the automatic value.

    The constructor checks every field and raises ValueError with a message
    that names the field. An int field takes a Python or numpy integer but not
    a bool; a float field takes any real number (``numbers.Real``, which
    includes a bool, as the reference reads CORE-2); ``auto_linpreds`` takes a
    Python or numpy bool; ``pmethod`` is ``"backward"`` or ``"none"``. The
    fields hold plain Python values (int, float, bool, str or None).
    Complexity: O(1).
    """

    max_degree: int = 1
    max_terms: int | None = None
    penalty: float | None = None
    thresh: float = 0.001
    minspan: int | None = None
    endspan: int | None = None
    adjust_endspan: float = 2.0
    auto_linpreds: bool = True
    fast_k: int = 20
    fast_beta: float = 1.0
    pmethod: str = "backward"
    nprune: int | None = None

    def __post_init__(self) -> None:
        for name in _PARAMS:
            object.__setattr__(self, name, _param(name, getattr(self, name)))


def _array(value: Any, dtype: type, shape: tuple, name: str) -> np.ndarray:
    """Return value as a new read-only array of ``dtype`` and ``shape``, where
    None in ``shape`` takes any length; raise ValueError when the values do
    not convert exactly (CORE-3, CORE-5). Complexity: O(size)."""
    a = np.asarray(value)
    kinds = {np.int8: "iu", np.int64: "iu", np.float64: "fiu", np.bool_: "b"}[dtype]
    if a.size and a.dtype.kind not in kinds:
        raise ValueError(
            f"{name} must hold {np.dtype(dtype).name} values, not {a.dtype}"
        )
    out = a.astype(dtype)  # a new array
    if a.size and not np.array_equal(out, a, equal_nan=dtype is np.float64):
        raise ValueError(f"{name} has values that are not {np.dtype(dtype).name}")
    if out.ndim != len(shape) or any(
        s is not None and s != t for s, t in zip(shape, out.shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, not {out.shape}")
    out.flags.writeable = False
    return out


def _scalar(value: Any, kind: type, name: str) -> Any:
    """Return value as a Python int or float; a bool is neither here."""
    abc = numbers.Integral if kind is int else numbers.Real
    if isinstance(value, bool) or not isinstance(value, abc):
        raise ValueError(f"{name} must be {kind.__name__}, not {value!r}")
    return kind(value)


def _increasing(a: np.ndarray, name: str) -> None:
    if np.any(np.diff(a) <= 0):
        raise ValueError(f"{name} must be increasing")


def _fields(d: Any, cls: type, name: str) -> dict:
    """Return the mapping d as a dict with exactly the fields of ``cls``."""
    if not isinstance(d, Mapping):
        raise ValueError(f"{name} must be a mapping, not {type(d).__name__}")
    names = (
        [f.name for f in dataclasses.fields(cls)]
        if dataclasses.is_dataclass(cls)
        else list(cls._fields)
    )
    if set(d) != set(names):
        raise ValueError(f"{name} must have the keys {sorted(names)}, not {sorted(d)}")
    return dict(d)


def _set(obj: Any, **values: Any) -> None:
    for key, value in values.items():
        object.__setattr__(obj, key, value)


@dataclasses.dataclass(frozen=True, eq=False)
class ForwardRecord:
    """The forward record (CORE-3), the fields of ``_forward.ForwardPass``.

    ``dirs`` (M_a, p) int8 and ``cuts`` (M_a, p): all terms added, in order.
    ``kept`` and ``dropped`` int64: the terms that FWD-11 keeps and drops, each
    increasing, together 0 to M_a - 1, with 0 kept. ``parent`` and ``step``
    (M_a,) int64 (TERM-6). ``rss`` (S + 1,): TSS, then the RSS after each step.
    ``termination``: a ``Termination`` (CORE-4). ``candidates``: a
    ``CandidateLog`` of S entries, or None. Complexity: O(M_a·p + S).
    """

    dirs: npt.NDArray[np.int8]
    cuts: FloatArray
    kept: IntArray
    dropped: IntArray
    parent: IntArray
    step: IntArray
    rss: FloatArray
    termination: Termination
    candidates: CandidateLog | None

    def __post_init__(self) -> None:
        dirs = _array(self.dirs, np.int8, (None, None), "forward.dirs")
        M, p = dirs.shape
        kept = _array(self.kept, np.int64, (None,), "forward.kept")
        dropped = _array(self.dropped, np.int64, (None,), "forward.dropped")
        rss = _array(self.rss, np.float64, (None,), "forward.rss")
        _increasing(kept, "forward.kept")
        _increasing(dropped, "forward.dropped")
        if p < 1 or rss.size < 1:
            raise ValueError("forward needs at least one covariate and one rss value")
        if kept[:1].tolist() != [0] or sorted([*kept, *dropped]) != list(range(M)):
            raise ValueError("forward.kept and dropped must split the terms, 0 kept")
        code = Termination(_scalar(self.termination, int, "forward.termination"))
        log = self.candidates
        if log is not None:
            if not isinstance(log, CandidateLog):
                raise ValueError("forward.candidates must be a CandidateLog or None")
            S = rss.size - 1
            dtypes = (np.float64, np.float64, np.int64, np.int64, np.float64, np.int8)
            log = CandidateLog(
                *(
                    _array(getattr(log, f), t, (S,), f"forward.candidates.{f}")
                    for f, t in zip(CandidateLog._fields, dtypes, strict=True)
                )
            )
        _set(
            self,
            dirs=dirs,
            cuts=_array(self.cuts, np.float64, (M, p), "forward.cuts"),
            kept=kept,
            dropped=dropped,
            parent=_array(self.parent, np.int64, (M,), "forward.parent"),
            step=_array(self.step, np.int64, (M,), "forward.step"),
            rss=rss,
            termination=code,
            candidates=log,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PruningRecord:
    """The pruning record (CORE-3), in pruning indices 0 to M_f - 1 (PRUNE-2).

    ``removed`` (M_f - 1,) int64 (PRUNE-4); ``rss_per_size`` and
    ``gcv_per_size`` (M_f,), index m - 1 for size m; ``subsets`` (M_f, M_f)
    bool, row m - 1 marking T[m]; ``selected_size`` m*, from 1 to M_f.
    Complexity: O(M_f²).
    """

    removed: IntArray
    rss_per_size: FloatArray
    gcv_per_size: FloatArray
    subsets: npt.NDArray[np.bool_]
    selected_size: int

    def __post_init__(self) -> None:
        rss = _array(self.rss_per_size, np.float64, (None,), "pruning.rss_per_size")
        M = rss.size
        size = _scalar(self.selected_size, int, "pruning.selected_size")
        if not 1 <= size <= M:
            raise ValueError(f"pruning.selected_size must be in 1 to {M}, not {size}")
        _set(
            self,
            removed=_array(self.removed, np.int64, (M - 1,), "pruning.removed"),
            rss_per_size=rss,
            gcv_per_size=_array(
                self.gcv_per_size, np.float64, (M,), "pruning.gcv_per_size"
            ),
            subsets=_array(self.subsets, np.bool_, (M, M), "pruning.subsets"),
            selected_size=size,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class MarsFit:
    """A fitted MARS model and its records (CORE-3).

    ``dirs`` (M, p) int8 and ``cuts`` (M, p): the selected terms, in increasing
    forward order; ``coef`` (M, K) (PRUNE-8); ``selected`` (M,) int64: their
    forward indices, increasing, from 0, all among ``forward.kept``; ``rss``,
    ``gcv``, ``rsq`` and ``grsq`` of the final model; ``n_eff`` = N = Σw
    after the drop; ``max_terms`` and ``penalty``, the resolved values;
    ``forward`` and ``pruning``, the records. The constructor checks these
    dtypes, shapes and relations. ``to_dict`` and ``from_dict`` convert to and
    from the dict of CORE-5. Complexity: O(M_a·p + M_f² + M·K).
    """

    dirs: npt.NDArray[np.int8]
    cuts: FloatArray
    coef: FloatArray
    selected: IntArray
    rss: float
    gcv: float
    rsq: float
    grsq: float
    n_eff: float
    max_terms: int
    penalty: float
    forward: ForwardRecord
    pruning: PruningRecord

    def __post_init__(self) -> None:
        fwd, prn = self.forward, self.pruning
        if not (isinstance(fwd, ForwardRecord) and isinstance(prn, PruningRecord)):
            raise ValueError("forward and pruning must be ForwardRecord, PruningRecord")
        p = fwd.dirs.shape[1]
        sel = _array(self.selected, np.int64, (None,), "selected")
        M = sel.size
        dirs = _array(self.dirs, np.int8, (M, p), "dirs")
        cuts = _array(self.cuts, np.float64, (M, p), "cuts")
        _increasing(sel, "selected")
        if sel[:1].tolist() != [0] or not np.isin(sel, fwd.kept).all():
            raise ValueError("selected must be kept forward terms, from 0")
        if not (
            np.array_equal(dirs, fwd.dirs[sel]) and np.array_equal(cuts, fwd.cuts[sel])
        ):
            raise ValueError("dirs and cuts must be the selected forward terms")
        if prn.rss_per_size.size != fwd.kept.size or prn.selected_size != M:
            raise ValueError("pruning must have M_f sizes and select M terms")
        stats = {
            k: _scalar(getattr(self, k), float, k)
            for k in ("rss", "gcv", "rsq", "grsq", "n_eff", "penalty")
        }
        _set(
            self,
            dirs=dirs,
            cuts=cuts,
            coef=_array(self.coef, np.float64, (M, None), "coef"),
            selected=sel,
            max_terms=_scalar(self.max_terms, int, "max_terms"),
            **stats,
        )
        if self.coef.shape[1] < 1:
            raise ValueError("coef must have at least one column")

    def to_dict(self) -> dict:
        """Return the fit as the nested dict of CORE-5: the field names as
        keys, ``forward``, ``pruning`` and ``candidates`` as dicts,
        ``termination`` as its integer code, and new arrays with the dtypes
        and shapes of CORE-3. Complexity: O(size of the fit)."""
        return _plain(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> MarsFit:
        """Build a fit from the dict of CORE-5, such as the reference returns.

        The keys must be the field names (a missing ``forward.candidates`` is
        None). Arrays may be numpy arrays or nested lists whose values convert
        exactly to the dtypes of CORE-3. Raises ValueError otherwise.
        Complexity: O(size of the fit).
        """
        d = _fields(d, cls, "the fit")
        fwd = d["forward"]
        if isinstance(fwd, Mapping) and "candidates" not in fwd:
            fwd = {**fwd, "candidates": None}
        fwd = _fields(fwd, ForwardRecord, "forward")
        if fwd["candidates"] is not None:
            log = _fields(fwd["candidates"], CandidateLog, "forward.candidates")
            fwd["candidates"] = CandidateLog(**log)
        d["forward"] = ForwardRecord(**fwd)
        d["pruning"] = PruningRecord(**_fields(d["pruning"], PruningRecord, "pruning"))
        return cls(**d)


def _plain(value: Any) -> Any:
    """The dict form of a record, with new arrays (CORE-5)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Termination):
        return int(value)
    if isinstance(value, CandidateLog):
        return {k: v.copy() for k, v in value._asdict().items()}
    if dataclasses.is_dataclass(value):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    return value
