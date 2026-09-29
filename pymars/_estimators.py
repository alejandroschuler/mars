"""The scikit-learn estimators: ``EarthRegressor`` (also ``Earth``) and the code
that the classifier shares with it.

Spec: ``docs/algorithm.md``, "Public API" (API-1 to API-7), "Errors" (ERR-1 to
ERR-4), W-6 and W-7 (the weight checks), CORE-1, CORE-2 and CORE-6 (the call
of the core), EDGE-1, RESP-2 (the shape of y), TERM-3 (``basis_matrix``) and
TERM-5 (the labels in ``summary()``).

Who does what. The estimators check the parameters and the data, and the core
(``_core.fit_mars``) fits the model; it assumes valid input (CORE-1). ``fit``
runs these steps in this order:

1. the parameters: ``allow_missing=True`` raises NotImplementedError (ERR-3),
   and ``MarsParams`` checks the others (ERR-4); ``__init__`` only stores them;
2. a column of X that holds strings, bytes or pandas categories raises
   ValueError that names OneHotEncoder (ERR-2);
3. ``validate_data`` makes X a float64 array and checks X and y (ERR-1); it
   also sets ``n_features_in_`` and ``feature_names_in_``;
4. the weight checks of W-6 and the UserWarning of W-7;
5. ``_core.fit_mars``, looked up in the module at each fit (CORE-6), so that a
   test can put the reference in its place.

``_EarthBase`` holds what the regressor and the classifier share: the
parameters, the checks above, the fitted attributes that come from the
``MarsFit``, ``basis_matrix``, ``summary`` and the tags. A subclass adds its
``fit``, its ``predict`` and ``_summary_columns``.

Numerics: X and Y reach the core as float64, and no method writes into its
inputs. Complexity: the checks cost O(n·p), with Python-level work only for
object and string columns; the fit costs what the core costs (CORE-7);
``predict`` and ``basis_matrix`` cost O(n·M·(max_degree + K)) time and O(n·M)
memory, for M terms and K responses.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import warnings

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from pymars import _core, _terms

__all__ = ["MISSING_VALUES_ISSUE", "ONE_HOT_RECIPE", "EarthRegressor"]

#: API-1: the parameters that map one to one to the fields of ``MarsParams``.
_MARS_PARAMS = tuple(f.name for f in dataclasses.fields(_core.MarsParams))
#: ERR-3: the issue about missing values.
MISSING_VALUES_ISSUE = "https://github.com/alejandroschuler/mars/issues/27"
#: ERR-2: the recipe for text and categorical columns, earth's coding of factors.
ONE_HOT_RECIPE = (
    'make_column_transformer((OneHotEncoder(drop="first"), cols), '
    'remainder="passthrough")'
)
#: W-7: the weights warn when their mean differs from 1 by more than this.
_MEAN_WEIGHT_TOL = 1e-6


def _holds_text(values: np.ndarray) -> bool:
    return any(isinstance(v, (str, bytes)) for v in values)


def _text_columns(X: object) -> list:
    """Return the columns of X that hold strings, bytes or pandas categorical
    values (ERR-2): names for a pandas DataFrame, else 0-based indices.

    Sparse input, and input that is not 2-D, gives none, so that
    ``validate_data`` raises its own error for it. A list is read as an object
    array, so that numpy does not turn its numbers into strings. Complexity:
    O(n·p), with Python-level work only for object and string columns.
    """
    pd = sys.modules.get("pandas")  # a DataFrame means that pandas is loaded
    if pd is not None and isinstance(X, pd.DataFrame):
        return [
            name
            for j, (name, dtype) in enumerate(X.dtypes.items())
            if isinstance(dtype, pd.CategoricalDtype)
            or (dtype.kind in "OSU" and _holds_text(X.iloc[:, j].to_numpy()))
        ]
    if sp.issparse(X):
        return []
    A = np.asarray(X, dtype=object) if isinstance(X, (list, tuple)) else np.asarray(X)
    if A.ndim != 2:
        return []
    if A.dtype.kind in "SU":
        return list(range(A.shape[1]))
    if A.dtype.kind == "O":
        return [j for j in range(A.shape[1]) if _holds_text(A[:, j])]
    return []


def _check_numeric(X: object, owner: BaseEstimator) -> None:
    """Raise ValueError that names OneHotEncoder and ``ONE_HOT_RECIPE`` when a
    column of X holds strings, bytes or categories (ERR-2)."""
    cols = _text_columns(X)
    if cols:
        raise ValueError(
            f"{type(owner).__name__} needs numeric columns, but the columns "
            f"{cols} of X hold strings, bytes or categories. Encode them with "
            f"OneHotEncoder first, for example in a Pipeline with {ONE_HOT_RECIPE}, "
            "which gives earth's coding of factors."
        )


def _check_weights(sample_weight: npt.ArrayLike | None, n: int) -> np.ndarray | None:
    """Return the weights as a float64 array of shape (n,), or None (W-6, W-7).

    Raises ValueError unless the weights are 1-D with one weight per row of X,
    finite and at least 0, with some weight positive; for weights that are
    all 0 the message matches ``weight.*zero``. Issues a UserWarning when some
    weight is not an integer and the mean of the weights differs from 1 by
    more than 1e-6. The zero weights stay, since the core drops their rows
    (W-3). Complexity: O(n).
    """
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight)
    if w.shape != (n,):
        raise ValueError(
            "sample_weight must be 1-D with one weight per row of X, shape "
            f"({n},), not shape {w.shape}"
        )
    w = check_array(w, ensure_2d=False, dtype=np.float64, input_name="sample_weight")
    if (w < 0.0).any():
        raise ValueError("sample_weight must be at least 0")
    if not (w > 0.0).any():
        raise ValueError("every sample weight is zero; at least one must be positive")
    if (w != np.floor(w)).any():
        mean = math.fsum(w) / n
        if abs(mean - 1.0) > _MEAN_WEIGHT_TOL:
            warnings.warn(
                "sample_weight holds weights that are not integers, and their "
                f"mean is {mean:.6g}, not 1. pymars reads weights as case "
                "counts: a weight w acts as w copies of its row, so the scale "
                "of the weights changes the fit. For importance weights, "
                "rescale them to w * n / sum(w) first.",
                UserWarning,
                stacklevel=3,
            )
    return w


class _EarthBase(BaseEstimator):
    """What the estimators share: the parameters (API-1), the checks of
    ``fit`` (ERR-1 to ERR-4, W-6, W-7), the fitted attributes that come from
    the ``MarsFit`` (API-3), ``basis_matrix`` and ``summary`` (API-4), and the
    tags (API-5). A subclass defines ``fit``, ``predict`` and
    ``_summary_columns``."""

    def __init__(
        self,
        max_degree=1,
        max_terms=None,
        penalty=None,
        thresh=0.001,
        minspan=None,
        endspan=None,
        adjust_endspan=2.0,
        auto_linpreds=True,
        fast_k=20,
        fast_beta=1.0,
        pmethod="backward",
        nprune=None,
        allow_missing=False,
    ):
        self.max_degree = max_degree
        self.max_terms = max_terms
        self.penalty = penalty
        self.thresh = thresh
        self.minspan = minspan
        self.endspan = endspan
        self.adjust_endspan = adjust_endspan
        self.auto_linpreds = auto_linpreds
        self.fast_k = fast_k
        self.fast_beta = fast_beta
        self.pmethod = pmethod
        self.nprune = nprune
        self.allow_missing = allow_missing

    def _mars_params(self) -> _core.MarsParams:
        """Return the parameters as a ``MarsParams``, which checks them (ERR-4);
        ``allow_missing`` must be a bool, and True raises NotImplementedError
        (ERR-3)."""
        if not isinstance(self.allow_missing, (bool, np.bool_)):
            raise ValueError(
                f"allow_missing must be a bool, not {self.allow_missing!r}"
            )
        if self.allow_missing:
            raise NotImplementedError(
                "allow_missing=True is not supported yet; see "
                f"{MISSING_VALUES_ISSUE}. Impute the missing values first, for "
                "example with sklearn.impute.SimpleImputer in a Pipeline."
            )
        return _core.MarsParams(**{name: getattr(self, name) for name in _MARS_PARAMS})

    def _check_input(self, X: npt.ArrayLike) -> np.ndarray:
        """Return X of a fitted estimator as a float64 array: ERR-2, then
        ``validate_data``, which checks the values (ERR-1), the number of
        columns (ERR-4) and the names of the columns of ``fit``."""
        _check_numeric(X, self)
        return validate_data(self, X, reset=False, dtype=np.float64)

    def _fit_core(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        w: np.ndarray | None,
        params: _core.MarsParams,
    ) -> _core.MarsFit:
        """Call ``_core.fit_mars`` through the module (CORE-6) and set the
        fitted attributes that come from the fit (API-3)."""
        fit = _core.fit_mars(X, Y, w, params)
        self.mars_ = fit
        self.dirs_, self.cuts_ = fit.dirs, fit.cuts
        self.rss_, self.gcv_ = fit.rss, fit.gcv
        self.rsq_, self.grsq_ = fit.rsq, fit.grsq
        self.max_terms_, self.penalty_ = fit.max_terms, fit.penalty
        return fit

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted means that the core returned a fit, so an estimator whose
        first ``fit`` failed is still not fitted."""
        return hasattr(self, "mars_")

    def basis_matrix(self, X: npt.ArrayLike) -> np.ndarray:
        """Return the basis matrix of the selected terms at X (TERM-3, API-4).

        X has the columns of ``fit``. The result is a new float64 array of
        shape (n, M): column k is term k (row k of ``dirs_`` and ``cuts_``) at
        the rows of X; column 0, the intercept, is 1. Nothing is clipped
        outside the range of the training data. Complexity: O(n·p) for the
        checks and O(n·M·max_degree) for the terms; memory O(n·M).
        """
        check_is_fitted(self)
        return _terms.basis_matrix(self._check_input(X), self.dirs_, self.cuts_)

    def _summary_columns(self) -> tuple[np.ndarray, list[str]]:
        """The coefficients that ``summary`` shows, (M, C), and C headings."""
        raise NotImplementedError

    def summary(self) -> str:
        """Return a table of the selected terms and the statistics of the fit.

        One line per selected term, in the order of ``dirs_``: its label
        (TERM-5), which uses the names in ``feature_names_in_`` after a fit on
        a table with string column names, and its coefficients. Then a line
        with ``rss_``, ``gcv_``, ``rsq_`` and ``grsq_``, and a line with the
        termination code of the forward pass (CORE-4). Numbers use the format
        ``.6g`` (API-4). Complexity: O(M·(p + K)).
        """
        check_is_fitted(self)
        names = getattr(self, "feature_names_in_", None)
        labels = _terms.term_labels(self.dirs_, self.cuts_, names)
        coef, heads = self._summary_columns()
        rows = [["term", *heads]]
        rows += [
            [s, *(f"{c:.6g}" for c in r)] for s, r in zip(labels, coef, strict=True)
        ]
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines = [
            "  ".join(
                [row[0].ljust(widths[0])]
                + [v.rjust(n) for v, n in zip(row[1:], widths[1:], strict=True)]
            )
            for row in rows
        ]
        code = self.mars_.forward.termination
        lines.append(
            f"rss {self.rss_:.6g}  gcv {self.gcv_:.6g}  "
            f"rsq {self.rsq_:.6g}  grsq {self.grsq_:.6g}"
        )
        lines.append(f"termination {int(code)} ({code.name})")
        return "\n".join(lines)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False  # API-5: NaN raises (ERR-1)
        return tags


class EarthRegressor(RegressorMixin, _EarthBase):
    """Multivariate adaptive regression splines (MARS) by the rules of R's
    earth package, with the scikit-learn API. ``pymars.Earth`` is this class.

    The parameters map one to one to earth's arguments (``docs/algorithm.md``,
    API-7), and None means earth's automatic value.

    Parameters
    ----------
    max_degree : int, default=1
        The largest degree of a term (earth's ``degree``).
    max_terms : int or None, default=None
        The term limit of the forward pass (``nk``); None gives
        min(200, max(20, 2p)) + 1 (LIMIT-1).
    penalty : float or None, default=None
        The GCV penalty per knot, -1 or at least 0; None gives 2 at degree 1
        and 3 otherwise (GCV-4).
    thresh : float, default=0.001
        The forward pass stops when RSq changes by less than this (STOP-4).
    minspan, endspan : int or None, default=None
        The spans of the knot search (SPAN-1 to SPAN-3); None is automatic.
    adjust_endspan : float, default=2.0
        The factor on the endspan for interaction terms (SPAN-4).
    auto_linpreds : bool, default=True
        Add a linear term in place of a hinge at the smallest value (FWD-6).
    fast_k : int, default=20
        The number of parents that Fast MARS searches; 0 searches all (FAST-3).
    fast_beta : float, default=1.0
        The ageing factor of the Fast MARS queue (FAST-2).
    pmethod : {"backward", "none"}, default="backward"
        The pruning method (PRUNE-1).
    nprune : int or None, default=None
        The largest number of terms that the pruning pass may select (LIMIT-3).
    allow_missing : bool, default=False
        True raises NotImplementedError: missing values are not supported
        (issue #27).

    Attributes
    ----------
    n_features_in_ : int
        The number of columns of X in ``fit``.
    feature_names_in_ : ndarray of str
        The column names, only after a fit on a table with string names.
    dirs_ : ndarray of int8, shape (M, p)
        The selected terms (TERM-1); row 0 is the intercept.
    cuts_ : ndarray of float64, shape (M, p)
        Their knots.
    term_coef_ : ndarray, shape (M,) or (M, K)
        The coefficients of the terms: (M,) for a 1-D y, (M, K) for a 2-D y.
    rss_, gcv_, rsq_, grsq_ : float
        The statistics of the returned model (PRUNE-8).
    max_terms_ : int
        The term limit that the fit used.
    penalty_ : float
        The penalty that the fit used.
    mars_ : MarsFit
        The whole fit, with the forward and pruning records (CORE-3).

    Notes
    -----
    ``sample_weight`` holds case counts (frequency weights): an integer weight
    w gives the fit of w copies of the row, and a zero weight that of a removed
    row (W-1, W-3). The estimator has no ``coef_``, no ``transform`` and no
    ``max_iter`` (API-6); ``term_coef_`` and ``basis_matrix`` give the model.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Numeric covariates, finite. Columns of strings, bytes or pandas
            categories raise ValueError; encode them with OneHotEncoder first.
        y : array-like of shape (n,) or (n, K)
            One response, or K responses that share the terms (RESP-1).
        sample_weight : array-like of shape (n,), default=None
            Case counts, finite and at least 0, not all 0; None gives every
            row the weight 1 exactly (W-5).

        Returns
        -------
        self : EarthRegressor
            The fitted estimator.
        """
        # A failed refit must not leave the new names with the old model (API-3).
        for name in [a for a in vars(self) if a.endswith("_") and a[0] != "_"]:
            delattr(self, name)
        params = self._mars_params()  # ERR-3 and ERR-4 before the data
        _check_numeric(X, self)
        X, y = validate_data(
            self, X, y, dtype=np.float64, y_numeric=True, multi_output=True
        )
        w = _check_weights(sample_weight, X.shape[0])
        Y = np.asarray(y, dtype=np.float64)
        fit = self._fit_core(X, Y.reshape(Y.shape[0], -1), w, params)
        self.term_coef_ = fit.coef[:, 0] if Y.ndim == 1 else fit.coef
        return self

    def predict(self, X):
        """Return the fitted values at X: shape (n,) after a fit on a 1-D y,
        else (n, K), K = 1 included (API-4, RESP-2). Complexity: that of
        ``basis_matrix``, plus O(n·M·K)."""
        return self.basis_matrix(X) @ self.term_coef_

    def _summary_columns(self) -> tuple[np.ndarray, list[str]]:
        coef = self.term_coef_
        if coef.ndim == 1:
            return coef[:, None], ["coef"]
        return coef, [f"y{k}" for k in range(coef.shape[1])]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True  # API-5, RESP-1
        return tags
