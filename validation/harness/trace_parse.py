"""Parse an earth ``trace = 7`` to ``9`` log into structured records.

This only reads printed text: the case-by-case output of earth's forward
pass, never its C or R source (the black-box use ``VALIDATION_PLAN.md``,
"Instruction files and clean room", allows). The line shapes below were read
off ``validation/legacy/out/trace9.txt`` (degree 1) and a fresh degree-2
trace made the same way (a black-box ``fit_earth.R`` run with
``trace = 9``), not off any documentation of earth's internals.

Log shape
---------
For each ``FindTerm: Searching for new term N`` block (one per forward
step), earth considers a series of (parent term, predictor) pairs. For each
pair it may:

- skip it outright (a parent-level or a (parent, predictor)-level ``skip
  (<reason>)`` line, with no further detail);
- report a linear-predictor candidate (a ``Case -1 ... RssDeltaLin ...``
  line) or say there is none (``no new form``), then always scan hinge
  knots between ``--FindKnotBegin--`` and ``--FindKnotEnd--`` (one
  ``--FindKnot--Case`` line per candidate case) and report the best hinge
  found (a ``Case ... RssDelta ...`` line).

A trailing ``best for term`` (or ``best for term (lin pred)``) on any of
these marks it as the best candidate seen so far, in file order; since a
later candidate can still beat it, only the last such tag in a step is the
step's actual winner (the table row that follows the step, kept verbatim in
``ForwardStep.summary_line``, is the authoritative record of what earth
picked).

RSS scale
---------
Every RSS and RSS-delta value in the log is on the scale of a y standardized
to sample variance 1: in ``validation/legacy/out/trace9.txt`` (200 cases),
``RssBeforeAddingHinge`` for the first split is exactly 199 = n - 1, which is
the total sum of squares of a unit-variance standardized y. Multiplying a
traced value by the real sample variance of y (``numpy.var(y, ddof=1)``, R's
``var()`` convention) then matches the real-scale RSS: for that same file,
the winning first-step candidate's ``Rss 31.029`` times ``var(y, ddof=1)``
(about 0.0929) is about 2.883, matching the real RSS after that step to the
trace's printed precision. ``parse_trace``'s ``sample_var_y`` argument does
this multiplication; pass ``None`` to keep the standardized values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FINDTERM_RE = re.compile(
    r"^\|FindTerm: Searching for new term\s+(?P<term>\d+)"
    r"\s+RssDelta\s+(?P<rss_delta>\S+)\s+MaxLegalRssDelta\s+(?P<max_legal>\S+)"
)
_SKIP_PRED_RE = re.compile(
    r"^\|Parent\s+(?P<parent>\d+)\s+Pred\s+(?P<pred>\d+)\s+skip \((?P<reason>.+)\)\s*$"
)
_SKIP_PARENT_RE = re.compile(
    r"^\|Parent\s+(?P<parent>\d+)\s+skip \((?P<reason>.+)\)\s*$"
)
_NO_NEW_FORM_RE = re.compile(
    r"^\|Parent\s+(?P<parent>\d+)\s+Pred\s+(?P<pred>\d+)\s+no new form\s*$"
)
# The linear pre-check ("RssDeltaLin") and the post-scan hinge outcome
# ("RssDelta") share this shape; the optional "Lin" group tells them apart.
_COMBO_RE = re.compile(
    r"^\|Parent\s+(?P<parent>\d+)\s+Pred\s+(?P<pred>\d+)\s+Case\s+(?P<case>-?\d+)"
    r"\s+Cut\s+(?P<cut>\S+?)<?\s+Rss\s+(?P<rss>\S+)\s+RssDelta(?P<lin>Lin)?"
    r"\s+(?P<rss_delta>\S+)\s*(?P<best>best for term(?: \(lin pred\))?)?\s*$"
)
_FINDKNOTBEGIN_RE = re.compile(
    r"^--FindKnotBegin--\s+iPred\s+(?P<pred>\d+)\s+iNewCol\s+(?P<new_col>\d+)"
    r"\s+RssBeforeAddingHinge\s+(?P<rss_before>\S+)\s+nMinSpan\s+(?P<minspan>\d+)"
    r"\s+nEndSpan\s+(?P<endspan>\d+)\s+nStartSpan\s+(?P<startspan>\d+)\s*$"
)
_FINDKNOT_CASE_RE = re.compile(
    r"^--FindKnot--Case\s+(?P<case>-?\d+)\s+RssWithKnot\s+(?P<rss>\S+)"
    r"\s+RssDelta\s+(?P<rss_delta>\S+)\s+Cut\s+(?P<cut>\S+?)<?"
    r"\s+bx1G\s+(?P<bx1g>\d)\s+CovColG\s+(?P<covcolg>\d)"
    r"\s+TolG\s+(?P<tolg>\d)\s+MaxG\s+(?P<maxg>\d)\s*(?P<best>best)?\s*$"
)
_FINDKNOTEND_RE = re.compile(r"^--FindKnotEnd--\s*$")


def _float(token: str) -> float:
    """Parse a trace numeric token; R prints ``Inf``/``-Inf``/``NaN`` as-is,
    and Python's ``float()`` already accepts those spellings."""
    return float(token.rstrip("<"))


@dataclass(frozen=True)
class CaseRecord:
    """One ``--FindKnot--Case`` row: a single candidate knot."""

    case: int
    cut: float
    rss: float
    rss_delta: float
    bx1g: bool
    cov_col_g: bool
    tol_g: bool
    max_g: bool
    best: bool

    @property
    def evaluated(self) -> bool:
        """Whether earth accepted this candidate as legal: derived from the
        four flags it printed, not a separate labeled field in the trace."""
        return self.bx1g and self.cov_col_g and self.tol_g and self.max_g


@dataclass(frozen=True)
class CandidateSummary:
    """A ``|Parent P  Pred Q  Case ...`` line: the linear pre-check, or the
    post-scan best-hinge outcome, for one (parent, predictor) pair."""

    case: int
    cut: float
    rss: float
    rss_delta: float
    best: bool


@dataclass
class KnotSearch:
    """Every candidate considered for one (parent, predictor) pair.

    ``pred`` is ``None`` when the parent itself was skipped before any
    predictor was considered. ``skipped_reason`` set means no scan happened
    at all (``linear``, ``hinge`` and ``cases`` all stay empty/``None``).
    """

    parent: int
    pred: int | None
    skipped_reason: str | None = None
    no_new_form: bool = False
    linear: CandidateSummary | None = None
    hinge: CandidateSummary | None = None
    cases: list[CaseRecord] = field(default_factory=list)
    # From --FindKnotBegin--, when a scan happened (None when skipped): the
    # span rules actually enforced for this search, which the candidate-knot-
    # sets component test compares with _get_allowable_knot_values.
    rss_before: float | None = None
    min_span: int | None = None
    end_span: int | None = None
    start_span: int | None = None


@dataclass
class ForwardStep:
    """One ``FindTerm: Searching for new term N`` block."""

    term: int
    rss_delta: float
    max_legal_rss_delta: float
    searches: list[KnotSearch] = field(default_factory=list)
    summary_line: str | None = None  # the table row that follows, verbatim


@dataclass
class TraceLog:
    steps: list[ForwardStep]
    sample_var_y: float | None


def parse_trace(path: str | Path, *, sample_var_y: float | None = None) -> TraceLog:
    """Parse a ``trace = 7`` to ``9`` log written by earth (for example by
    ``fit_earth.R``'s ``trace_file``).

    When ``sample_var_y`` is given, every RSS and RSS-delta value is
    multiplied by it, converting the log's standardized-y scale to the real
    scale (this module's docstring, "RSS scale", checks the factor); ``None``
    keeps the raw standardized values.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    def scaled(token: str) -> float:
        x = _float(token)
        return x if sample_var_y is None else x * sample_var_y

    steps: list[ForwardStep] = []
    step: ForwardStep | None = None
    search: KnotSearch | None = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        m = _FINDTERM_RE.match(line)
        if m:
            step = ForwardStep(
                term=int(m["term"]),
                rss_delta=scaled(m["rss_delta"]),
                max_legal_rss_delta=scaled(m["max_legal"]),
            )
            steps.append(step)
            search = None
            i += 1
            continue

        if step is not None and line.lstrip().startswith(f"{step.term} "):
            step.summary_line = line
            i += 1
            continue

        m = _SKIP_PRED_RE.match(line)
        if m:
            step.searches.append(
                KnotSearch(
                    parent=int(m["parent"]),
                    pred=int(m["pred"]),
                    skipped_reason=m["reason"],
                )
            )
            i += 1
            continue

        m = _SKIP_PARENT_RE.match(line)
        if m:
            step.searches.append(
                KnotSearch(
                    parent=int(m["parent"]), pred=None, skipped_reason=m["reason"]
                )
            )
            i += 1
            continue

        m = _NO_NEW_FORM_RE.match(line)
        if m:
            search = KnotSearch(
                parent=int(m["parent"]), pred=int(m["pred"]), no_new_form=True
            )
            step.searches.append(search)
            i += 1
            continue

        m = _COMBO_RE.match(line)
        if m:
            candidate = CandidateSummary(
                case=int(m["case"]),
                cut=_float(m["cut"]),
                rss=scaled(m["rss"]),
                rss_delta=scaled(m["rss_delta"]),
                best=m["best"] is not None,
            )
            if m["lin"]:
                search = KnotSearch(
                    parent=int(m["parent"]), pred=int(m["pred"]), linear=candidate
                )
                step.searches.append(search)
            else:
                search.hinge = candidate
            i += 1
            continue

        m = _FINDKNOTBEGIN_RE.match(line)
        if m:
            search.rss_before = scaled(m["rss_before"])
            search.min_span = int(m["minspan"])
            search.end_span = int(m["endspan"])
            search.start_span = int(m["startspan"])
            i += 1
            while not _FINDKNOTEND_RE.match(lines[i]):
                cm = _FINDKNOT_CASE_RE.match(lines[i])
                if cm:
                    search.cases.append(
                        CaseRecord(
                            case=int(cm["case"]),
                            cut=_float(cm["cut"]),
                            rss=scaled(cm["rss"]),
                            rss_delta=scaled(cm["rss_delta"]),
                            bx1g=cm["bx1g"] == "1",
                            cov_col_g=cm["covcolg"] == "1",
                            tol_g=cm["tolg"] == "1",
                            max_g=cm["maxg"] == "1",
                            best=cm["best"] is not None,
                        )
                    )
                i += 1
            i += 1  # past --FindKnotEnd--
            continue

        i += 1  # a header, blank, separator or "Call:" line we do not need

    return TraceLog(steps=steps, sample_var_y=sample_var_y)
