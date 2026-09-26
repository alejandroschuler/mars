"""Parse an earth ``trace = 7`` to ``9`` log into structured records.

This only reads printed text: the case-by-case output of earth's forward
pass, never its C or R source (the black-box use ``VALIDATION_PLAN.md``,
"Instruction files and clean room", allows). The line shapes below were read
off ``validation/legacy/out/trace9.txt`` (degree 1) and fresh traces made
the same way (black-box ``fit_earth.R`` runs at each of ``trace = 7``, ``8``
and ``9``, at degree 1 and 2, with and without automatic spans), not off
any documentation of earth's internals.

Log shape
---------
For each ``FindTerm: Searching for new term N`` block (one per forward
step), earth considers a series of (parent term, predictor) pairs, printed
as ``|Parent P  Pred Q ...``. **``P`` counts earth's own internal slots, not
rows of a ``dirs``/``cuts`` matrix built from the selected or forward
terms**: do not assume ``parent`` is a 0-based or 1-based index into such a
matrix without checking the mapping first. For each pair earth may:

- skip it outright (a parent-level or a (parent, predictor)-level ``skip
  (<reason>)`` line, with no further detail);
- at ``trace = 8`` or ``9``: report a linear-predictor candidate (a ``Case
  -1 ...`` line, with an ``RssDeltaLin`` field at ``trace = 9`` only) or say
  there is none (``no new form``), then scan hinge knots between
  ``--FindKnotBegin--`` and ``--FindKnotEnd--`` (one ``--FindKnot--Case``
  line per visited case) and report the best hinge found (a ``Case ...
  RssDelta ...`` line);
- at ``trace = 7``: report one summary line only, with no separate linear
  and hinge phases and no per-case detail at all (``search.linear`` stays
  ``None`` and ``search.cases`` stays empty; the outcome, whichever phase
  produced it, is stored as ``search.hinge``).

Within a ``trace = 9`` ``FindKnotBegin``/``FindKnotEnd`` block, a case earth
visits but does not evaluate (most often the minspan/endspan rule) prints
as ``--FindKnot--Case N iSpan K bx1 V``, not ``RssWithKnot ...``; this
parser keeps it too, as a ``CaseRecord`` with ``evaluated = False``. At
``trace = 8`` these span-skipped cases are not printed at all (only
evaluated cases are, without the four flags); at ``trace = 7`` there is no
per-case detail of any kind.

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

The trace prints RSS values at only 5 or 6 significant digits, so a near-tie
at the plan's 1e-7 relative threshold ("Ties") cannot be judged from these
values; that comparison needs the new code's own candidate log
(``record_candidates=True``), which carries full precision.
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
# One line per (parent, predictor) combo, three shapes sharing one pattern:
# trace 9's linear pre-check ("RssDeltaLin"), trace 8's linear pre-check (no
# RssDelta field at all) and the post-scan hinge outcome / trace 7's single
# summary line (both "RssDelta"). The optional groups tell them apart; see
# parse_trace's main loop for how "lin" is None but rss_delta is set is
# itself ambiguous between a post-scan update and a trace-7 standalone line.
_COMBO_RE = re.compile(
    r"^\|Parent\s+(?P<parent>\d+)\s+Pred\s+(?P<pred>\d+)\s+Case\s+(?P<case>-?\d+)"
    r"\s+Cut\s+(?P<cut>\S+?)<?\s+Rss\s+(?P<rss>\S+)"
    r"(?:\s+RssDelta(?P<lin>Lin)?\s+(?P<rss_delta>\S+))?"
    r"\s*(?P<best>best for term(?: \(lin pred\))?)?\s*$"
)
_FINDKNOTBEGIN_RE = re.compile(
    r"^--FindKnotBegin--\s+iPred\s+(?P<pred>\d+)\s+iNewCol\s+(?P<new_col>\d+)"
    r"\s+RssBeforeAddingHinge\s+(?P<rss_before>\S+)\s+nMinSpan\s+(?P<minspan>\d+)"
    r"\s+nEndSpan\s+(?P<endspan>\d+)\s+nStartSpan\s+(?P<startspan>\d+)\s*$"
)
# The four flags are only printed at trace = 9; at trace = 8 the line ends
# after Cut (and every visited case is evaluated, so there are no iSpan rows
# to match separately at that level).
_FINDKNOT_CASE_RE = re.compile(
    r"^--FindKnot--Case\s+(?P<case>-?\d+)\s+RssWithKnot\s+(?P<rss>\S+)"
    r"\s+RssDelta\s+(?P<rss_delta>\S+)\s+Cut\s+(?P<cut>\S+?)<?"
    r"(?:\s+bx1G\s+(?P<bx1g>\d)\s+CovColG\s+(?P<covcolg>\d)"
    r"\s+TolG\s+(?P<tolg>\d)\s+MaxG\s+(?P<maxg>\d))?"
    r"\s*(?P<best>best)?\s*$"
)
# A case earth visited but did not evaluate (trace = 9 only): no Rss, no
# flags, just how many cases short of a legal span it fell (iSpan) and the
# parent's raw value there (bx1).
_FINDKNOT_ISPAN_RE = re.compile(
    r"^--FindKnot--Case\s+(?P<case>-?\d+)\s+iSpan\s+(?P<i_span>\d+)"
    r"\s+bx1\s+(?P<bx1>\S+)\s*$"
)
_FINDKNOTEND_RE = re.compile(r"^--FindKnotEnd--\s*$")


def _float(token: str) -> float:
    """Parse a trace numeric token; R prints ``Inf``/``-Inf``/``NaN`` as-is,
    and Python's ``float()`` already accepts those spellings."""
    return float(token.rstrip("<"))


class TraceParseError(ValueError):
    """A line inside a FindKnot block matched neither the evaluated-case nor
    the visited-but-skipped shape (see this module's docstring); raised
    rather than silently dropped, so a trace shape this parser does not yet
    know about is never mistaken for one with fewer visited cases."""


@dataclass(frozen=True)
class CaseRecord:
    """One ``--FindKnot--Case`` row: a single visited case.

    A row printed as ``iSpan ... bx1 ...`` (earth skipped it without a full
    evaluation, most often the minspan/endspan rule) has ``evaluated =
    False``; only ``case``, ``i_span`` and ``bx1`` are meaningful, the rest
    stay ``None``. A row printed as ``RssWithKnot ...`` has ``evaluated =
    True``; at ``trace = 9`` it also carries the four flags and ``accepted``
    (their AND) is whether earth accepted it as a legal candidate (an
    evaluated row can still be rejected, most often by ``tol_g``, so
    ``accepted`` means "accepted", not "evaluated"); at ``trace = 8`` the
    flags are not printed, so the four flag fields and ``accepted`` stay
    ``None`` even though the case was evaluated.
    """

    case: int
    evaluated: bool
    best: bool = False
    i_span: int | None = None
    bx1: float | None = None
    cut: float | None = None
    rss: float | None = None
    rss_delta: float | None = None
    bx1g: bool | None = None
    cov_col_g: bool | None = None
    tol_g: bool | None = None
    max_g: bool | None = None

    @property
    def accepted(self) -> bool | None:
        """The AND of the four flags (whether earth accepted this candidate
        as legal); ``None`` when it was not evaluated, or when the flags
        were not printed at this trace level."""
        if not self.evaluated or self.bx1g is None:
            return None
        return bool(self.bx1g and self.cov_col_g and self.tol_g and self.max_g)


@dataclass(frozen=True)
class CandidateSummary:
    """A ``|Parent P  Pred Q  Case ...`` line: the linear pre-check, the
    post-scan best-hinge outcome, or (at ``trace = 7``) the one summary a
    (parent, predictor) pair gets. ``rss_delta`` is ``None`` only for a
    ``trace = 8`` linear pre-check, which does not print one."""

    case: int
    cut: float
    rss: float
    rss_delta: float | None
    best: bool


@dataclass
class KnotSearch:
    """Every candidate considered for one (parent, predictor) pair.

    ``pred`` is ``None`` when the parent itself was skipped before any
    predictor was considered. ``skipped_reason`` set means no scan happened
    at all (``linear``, ``hinge`` and ``cases`` all stay empty/``None``).
    At ``trace = 7`` the single reported outcome is stored as ``hinge``
    (``linear`` and ``cases`` stay empty), because the log does not say
    which phase, if either, it came from.
    """

    parent: int
    pred: int | None
    skipped_reason: str | None = None
    no_new_form: bool = False
    linear: CandidateSummary | None = None
    hinge: CandidateSummary | None = None
    cases: list[CaseRecord] = field(default_factory=list)
    # From --FindKnotBegin-- (trace 8 and 9 only; None at trace 7, where
    # there is no such block): the span rules actually enforced for this
    # search, which the candidate-knot-sets component test compares with
    # _get_allowable_knot_values.
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
    ``fit_earth.R``'s ``trace_file``); see this module's docstring for how
    much detail each level gives.

    When ``sample_var_y`` is given, every RSS and RSS-delta value is
    multiplied by it, converting the log's standardized-y scale to the real
    scale (this module's docstring, "RSS scale", checks the factor); ``None``
    keeps the raw standardized values.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    def scaled(token: str) -> float:
        x = _float(token)
        return x if sample_var_y is None else x * sample_var_y

    def scaled_opt(token: str | None) -> float | None:
        return None if token is None else scaled(token)

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
            parent, pred = int(m["parent"]), int(m["pred"])
            candidate = CandidateSummary(
                case=int(m["case"]),
                cut=_float(m["cut"]),
                rss=scaled(m["rss"]),
                rss_delta=scaled_opt(m["rss_delta"]),
                best=m["best"] is not None,
            )
            if m["lin"] or m["rss_delta"] is None:
                # trace 9's "RssDeltaLin" line, or trace 8's linear pre-check
                # (no RssDelta field at all): both open a new combo.
                search = KnotSearch(parent=parent, pred=pred, linear=candidate)
                step.searches.append(search)
            elif (
                search is not None
                and search.hinge is None
                and search.parent == parent
                and search.pred == pred
            ):
                # The post-scan update to the combo a linear line or "no new
                # form" just opened (trace 8 or 9).
                search.hinge = candidate
            else:
                # trace 7: one line is the whole combo, with no separate
                # linear/hinge phase and no FindKnotBegin/End block.
                search = KnotSearch(parent=parent, pred=pred, hinge=candidate)
                step.searches.append(search)
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
                            evaluated=True,
                            cut=_float(cm["cut"]),
                            rss=scaled(cm["rss"]),
                            rss_delta=scaled(cm["rss_delta"]),
                            bx1g=None if cm["bx1g"] is None else cm["bx1g"] == "1",
                            cov_col_g=None
                            if cm["covcolg"] is None
                            else cm["covcolg"] == "1",
                            tol_g=None if cm["tolg"] is None else cm["tolg"] == "1",
                            max_g=None if cm["maxg"] is None else cm["maxg"] == "1",
                            best=cm["best"] is not None,
                        )
                    )
                    i += 1
                    continue
                sm = _FINDKNOT_ISPAN_RE.match(lines[i])
                if sm:
                    search.cases.append(
                        CaseRecord(
                            case=int(sm["case"]),
                            evaluated=False,
                            i_span=int(sm["i_span"]),
                            bx1=_float(sm["bx1"]),
                        )
                    )
                    i += 1
                    continue
                raise TraceParseError(
                    f"line {i + 1} inside a FindKnot block matches neither "
                    f"the evaluated-case nor the visited-but-skipped shape: "
                    f"{lines[i]!r}"
                )
            i += 1  # past --FindKnotEnd--
            continue

        i += 1  # a header, blank, separator or "Call:" line we do not need

    return TraceLog(steps=steps, sample_var_y=sample_var_y)
