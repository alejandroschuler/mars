"""The core of pymars: the parameters, the fit record and ``fit_mars``.

Spec: ``docs/algorithm.md``, "Core API" (CORE-1 to CORE-7), and the rules that
the core applies itself: W-3 and W-4 (the zero-weight drop and N), LIMIT-1 and
GCV-4 (the resolved ``max_terms`` and ``penalty``), EDGE-1, EDGE-6 and GCV-7
(degenerate fits and the power of 2 on Y), FWD-11 (the kept terms), PRUNE-5 to
PRUNE-8 (``nprune``, ``pmethod`` and the final fit) and RESP-1.

Who does what in ``fit_mars``:

- The core checks the arrays, drops the rows with zero weight (W-3), forms N
  and τ_N (W-4), resolves ``max_terms`` and ``penalty``, and builds the
  degenerate fit itself, since the forward pass does not run for it (EDGE-1).
- ``_forward.forward_pass`` is called through the module attribute, so that a
  test can replace it. It gets the original Y and the resolved values; it
  scales Y by its own power of 2 (EDGE-6) and centers it (FWD-10), so the core
  does neither for it, and its record is on the original scale (LA-6). It
  also picks the kept terms (FWD-11).
- ``_pruning.pruning_pass`` and ``_pruning.final_fit`` get the basis of the
  kept terms and Y times s = 2^j, the power of 2 of EDGE-6, which ``_pruning``
  leaves to its caller. The core multiplies their sums of squares and GCVs by
  1/s² and their coefficients by 1/s, which changes no bit unless a value
  leaves the normal range, and keeps their RSq and GRSq, which come from the
  scaled values. Pruning index m is forward index ``kept[m]``.

Records. ``MarsFit``, ``ForwardRecord`` and ``PruningRecord`` are frozen
dataclasses (CORE-3). ``ForwardRecord`` is a frozen dataclass made from
``_forward.ForwardPass``, with the same fields, so that every record of a fit
has the same checks and the same dict form; ``CandidateLog`` and
``Termination`` are ``_forward``'s own, one definition of each. The
constructors check the dtypes and shapes of CORE-3 and store new read-only
arrays, so a fit shares no memory with its inputs. ``to_dict`` and
``from_dict`` give the dict form of CORE-5, which the reference returns.

Numerics: float64 throughout; no function writes into its inputs; no absolute
epsilon; the tie rules are those of ``_forward`` and ``_pruning``.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from pymars import _forward, _gcv, _pruning, _terms
from pymars._forward import CandidateLog, Termination

__all__ = [
    "CandidateLog",
    "ForwardRecord",
    "MarsFit",
    "MarsParams",
    "PruningRecord",
    "Termination",
    "fit_mars",
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
#: The fields that go to the forward pass as they are (``max_terms`` and
#: ``penalty`` go resolved).
_FORWARD_PARAMS = (
    "max_degree",
    "thresh",
    "minspan",
    "endspan",
    "adjust_endspan",
    "auto_linpreds",
    "fast_k",
    "fast_beta",
)
#: EDGE-6: the scaled TSS must be at least the smallest positive normal float64.
_TINY = float(np.finfo(np.float64).tiny)


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


def _cases(
    X: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Check X (n, p), Y (n, K) or (n,) and w (n,) or None, and drop the rows
    with zero weight (W-3). The inputs are not copied unless rows are dropped
    or a dtype changes, and never written to. Complexity: O(n·(p + K))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or 0 in (*X.shape, *Y.shape):
        raise ValueError(f"X (n, p) and Y (n, K) do not match: {X.shape}, {Y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X and Y must be finite")
    if w is None:
        return X, Y, None
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (len(X),) or not (np.isfinite(w).all() and (w >= 0.0).all()):
        raise ValueError(f"w must be finite and at least 0, with shape ({len(X)},)")
    keep = w > 0.0
    if not keep.any():
        raise ValueError("every weight is zero; the weights need a positive sum")
    return (X, Y, w) if keep.all() else (X[keep], Y[keep], w[keep])


def _intercept_record(p: int, tss: float, record: bool) -> ForwardRecord:
    """The forward record of a degenerate fit: the intercept alone, rss [TSS],
    code ``DEGENERATE`` and an empty log when one is asked for (EDGE-1)."""
    dirs, cuts = _terms.intercept_terms(p)
    none, zero = np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    empty = np.zeros(0)
    log = CandidateLog(empty, empty, none, none, empty, np.zeros(0, dtype=np.int8))
    return ForwardRecord(
        dirs,
        cuts,
        zero,
        none,
        np.array([-1]),
        zero,
        np.array([tss]),
        Termination.DEGENERATE,
        log if record else None,
    )


def fit_mars(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    w: npt.ArrayLike | None,
    params: MarsParams,
    *,
    record_candidates: bool = False,
) -> MarsFit:
    """Fit a MARS model (CORE-1): the forward pass, the pruning pass and the
    final weighted least squares.

    X (n, p) and Y (n, K), or (n,) for K = 1, are finite float64 arrays; w is
    (n,), finite and ≥ 0 with a positive sum, or None for w_i = 1 exactly
    (W-5). The estimators check the data first (ERR-1); here bad shapes,
    nonfinite values and weights that are all 0 raise ValueError. The steps:

    1. drop the rows with zero weight (W-3); N = Σw by ``math.fsum``, snapped
       (W-4), is ``n_eff``;
    2. resolve ``max_terms`` (LIMIT-1) and ``penalty`` (GCV-4);
    3. j with D·2^j ∈ [1, 2) for D = max |Y| (EDGE-6); if some response is not
       constant and the TSS of Y·2^j is not a positive normal float64, raise
       ValueError (the scale of y or of the weights is out of range);
    4. a degenerate fit (N ≤ 1 or every response constant, EDGE-1, GCV-7) is
       the intercept alone, without the forward pass, with code
       ``DEGENERATE``; otherwise ``_forward.forward_pass`` gives the forward
       record (module docstring);
    5. the pruning pass on the kept terms (FWD-11, PRUNE-2), with ``pmethod``
       and ``nprune`` (PRUNE-5 to PRUNE-7), and the final fit of the selected
       terms (PRUNE-8), both on Y·2^j and scaled back; ``selected`` is in
       forward indices, ``kept[m]`` for pruning index m.

    The function is pure: it never writes into its inputs, and its output
    depends on the order of the rows only at near-ties. ``record_candidates``
    asks the forward pass for its candidate log (FWD-8). Complexity: the
    forward pass (CORE-7), plus O(n·(p + K)) for the checks and the scaling,
    O(n·M_f·max_degree) for the basis of the kept terms, and the pruning pass
    and the final fit, O(n·M_f·(M_f + K) + M_f³·(M_f + K)) (PRUNE-9); memory
    O(n·(p + M_max + K)).
    """
    if not isinstance(params, MarsParams):
        raise TypeError(f"params must be a MarsParams, not {type(params).__name__}")
    X, Y, w = _cases(X, Y, w)
    n, p = X.shape
    N, _ = _gcv.total_weight(n, w)
    max_terms = params.max_terms
    max_terms = _gcv.default_max_terms(p) if max_terms is None else max_terms
    penalty = params.penalty
    penalty = _gcv.default_penalty(params.max_degree) if penalty is None else penalty
    D = float(np.max(np.abs(Y)))
    j = 0 if D == 0.0 else 1 - math.frexp(D)[1]
    Ys = np.ldexp(Y, j)
    tss = _gcv.tss(Ys, w)
    if not np.all(Y[0] == Y) and not (math.isfinite(tss) and tss >= _TINY):
        raise ValueError(
            "the scale of y or of the weights is out of range: the total sum of "
            f"squares of y·2^{j} is {tss}, not a positive normal float64 (EDGE-6)"
        )
    if _gcv.is_degenerate(Y, N):
        forward = _intercept_record(p, float(np.ldexp(tss, -2 * j)), record_candidates)
    else:
        kw = {name: getattr(params, name) for name in _FORWARD_PARAMS}
        fp = _forward.forward_pass(
            X,
            Y,
            w,
            **kw,
            max_terms=max_terms,
            penalty=penalty,
            record_candidates=record_candidates,
        )
        forward = ForwardRecord(
            **{f.name: getattr(fp, f.name) for f in dataclasses.fields(ForwardRecord)}
        )
    kept = forward.kept
    B = _terms.basis_matrix(X, forward.dirs[kept], forward.cuts[kept])
    pruned = _pruning.pruning_pass(
        B, Ys, w, penalty=penalty, pmethod=params.pmethod, nprune=params.nprune
    )
    final = _pruning.final_fit(B, Ys, pruned.selected, w, penalty=penalty)
    selected = kept[pruned.selected]
    return MarsFit(
        dirs=forward.dirs[selected],
        cuts=forward.cuts[selected],
        coef=np.ldexp(final.coef, -j),
        selected=selected,
        rss=float(np.ldexp(final.rss, -2 * j)),
        gcv=float(np.ldexp(final.gcv, -2 * j)),
        rsq=final.rsq,
        grsq=final.grsq,
        n_eff=N,
        max_terms=max_terms,
        penalty=penalty,
        forward=forward,
        pruning=PruningRecord(
            removed=pruned.removed,
            rss_per_size=np.ldexp(pruned.rss_per_size, -2 * j),
            gcv_per_size=np.ldexp(pruned.gcv_per_size, -2 * j),
            subsets=pruned.subsets,
            selected_size=pruned.selected_size,
        ),
    )
