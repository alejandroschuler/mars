"""Tests of pymars/_terms.py: the TERM rules of docs/algorithm.md.

The earth fixtures are ``validation/fixtures/components/predict_new_points``
and the S datasets (``dirs``, ``cuts``, coefficients, predictions and term
names from earth 5.3.4). Tolerance, from the plan's tolerance table: the
largest absolute difference of predictions at most 1e-8·sd(y).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pymars import _terms
from pymars._terms import ABSENT, LINEAR, MINUS, PLUS

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "fixtures"


def _s_fits(*, least_squares: bool) -> list[str]:
    """The S fixtures with an earth fit; with ``least_squares``, only those
    whose predictions are B·coef (no GLM refit, no factor response)."""
    names = []
    for path in sorted(FIXTURES.glob("S*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))["result"]
        if result.get("error") or result.get("dirs") is None:
            continue
        glm = result.get("glm_coef") is not None or result.get("levels") is not None
        if not (least_squares and glm):
            names.append(path.stem)
    return names


S_FITS = _s_fits(least_squares=False)
S_LEAST_SQUARES_FITS = _s_fits(least_squares=True)


def earth_terms(dirs, cuts):
    """earth's dirs and cuts as TERM-1 arrays: earth stores min(x) in cuts
    for a linear factor, pymars 0.0 (TERM-2)."""
    d = np.asarray(dirs, dtype=float).astype(np.int8)
    c = np.array(cuts, dtype=np.float64)
    c[(d == ABSENT) | (d == LINEAR)] = 0.0
    return d, c


def test_codes_and_intercept():
    """TERM-1: the codes, and row 0 of the intercept model."""
    assert (ABSENT, PLUS, MINUS, LINEAR) == (0, 1, -1, 2)
    dirs, cuts = _terms.intercept_terms(3)
    assert dirs.dtype == np.int8 and cuts.dtype == np.float64
    assert dirs.shape == cuts.shape == (1, 3)
    assert not dirs.any() and not cuts.any()
    with pytest.raises(ValueError):
        _terms.intercept_terms(0)


def test_basis_matrix_by_hand():
    """TERM-3: each factor, a product of degree 3, and the intercept."""
    X = np.array([[0.0, 5.0, -1.0], [1.0, 2.0, 3.0], [4.0, -2.0, 0.5]])
    dirs = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0], [1, -1, 2]], dtype=np.int8
    )
    cuts = np.array(
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0], [0.5, 1.0, 0]], dtype=float
    )
    expected = np.array(
        [
            [1.0, 0.0, 0.5, 5.0, 0.0],
            [1.0, 0.5, 0.0, 2.0, 0.0],
            [1.0, 3.5, 0.0, -2.0, 3.5 * 3.0 * 0.5],
        ]
    )
    B = _terms.basis_matrix(X, dirs, cuts)
    assert B.dtype == np.float64 and B.shape == (3, 5)
    np.testing.assert_array_equal(B, expected)


def test_basis_matrix_extrapolates():
    """TERM-3: nothing is clipped outside the range of the training data."""
    dirs = np.array([[0], [1], [-1], [2]], dtype=np.int8)
    cuts = np.array([[0.0], [0.5], [0.5], [0.0]])
    B = _terms.basis_matrix(np.array([[-100.0], [100.0]]), dirs, cuts)
    np.testing.assert_array_equal(B, [[1, 0, 100.5, -100], [1, 99.5, 0, 100]])


def test_basis_matrix_keeps_inputs():
    """Conventions: no function changes its inputs."""
    X = np.array([[0.2, 0.7], [0.9, 0.1]])
    dirs = np.array([[0, 0], [1, -1]], dtype=np.int8)
    cuts = np.array([[0.0, 0.0], [0.5, 0.4]])
    copies = X.copy(), dirs.copy(), cuts.copy()
    _terms.basis_matrix(X, dirs, cuts)
    for before, after in zip(copies, (X, dirs, cuts), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    ("dirs", "cuts", "message"),
    [
        ([[0, 1]], [[0.0]], "one shape"),
        ([0, 1], [0.0, 0.5], "one shape"),
        ([[0, 3]], [[0.0, 0.5]], "code"),
        ([[0, 1]], [[0.0, np.nan]], "finite"),
        ([[0, 1]], [[0.0, np.inf]], "finite"),
        ([[0.5, 1]], [[0.0, 0.5]], "code"),
        ([[0, 0]], [[0.0, 0.5]], "TERM-2"),
        ([[0, 2]], [[0.0, 0.5]], "TERM-2"),
    ],
)
def test_check_terms_rejects(dirs, cuts, message):
    """TERM-1 and TERM-2: codes 0, ±1 and 2; finite cuts, 0.0 for 0 and 2."""
    with pytest.raises(ValueError, match=message):
        _terms.check_terms(dirs, cuts)


def test_check_terms_accepts_float_codes():
    """earth's JSON dirs are floats; exact codes are accepted as int8."""
    dirs, _ = _terms.check_terms([[0.0, 1.0], [2.0, -1.0]], [[0, 0.5], [0, 0.25]])
    assert dirs.dtype == np.int8
    np.testing.assert_array_equal(dirs, [[0, 1], [2, -1]])


def test_basis_matrix_rejects_bad_x():
    dirs, cuts = _terms.intercept_terms(2)
    with pytest.raises(ValueError, match="columns"):
        _terms.basis_matrix(np.zeros((3, 3)), dirs, cuts)
    with pytest.raises(ValueError, match="2-D"):
        _terms.basis_matrix(np.zeros(3), dirs, cuts)


def test_factor():
    """TERM-3: f(+1, v, c) = (v - c)₊, f(-1, v, c) = (c - v)₊, f(2, v, c) = v."""
    x = np.array([-2.0, 0.5, 3.0])
    np.testing.assert_array_equal(_terms.factor(PLUS, x, 0.5), [0.0, 0.0, 2.5])
    np.testing.assert_array_equal(_terms.factor(MINUS, x, 0.5), [2.5, 0.0, 0.0])
    linear = _terms.factor(LINEAR, x, 0.5)
    np.testing.assert_array_equal(linear, x)
    assert linear is not x
    with pytest.raises(ValueError):
        _terms.factor(ABSENT, x, 0.0)


def test_degrees_and_variables():
    """TERM-4: the degree counts the nonzero codes, a linear factor as one."""
    dirs = np.array([[0, 0, 0], [1, 0, 0], [2, -1, 0], [1, 2, -1]], dtype=np.int8)
    np.testing.assert_array_equal(_terms.term_degrees(dirs), [0, 1, 2, 3])
    assert _terms.term_degrees(dirs).dtype == np.int64
    np.testing.assert_array_equal(_terms.term_variables(dirs[2]), [0, 1])
    assert _terms.term_variables(dirs[0]).size == 0
    with pytest.raises(ValueError):
        _terms.term_degrees(dirs[0])
    with pytest.raises(ValueError):
        _terms.term_variables(dirs)


def test_child_term():
    """TERM-6: the child is the parent times one new factor; TERM-2, TERM-4."""
    parent_d = np.array([1, 0, 0], dtype=np.int8)
    parent_c = np.array([0.5, 0.0, 0.0])
    d, c = _terms.child_term(parent_d, parent_c, 2, MINUS, 0.25)
    np.testing.assert_array_equal(d, [1, 0, -1])
    np.testing.assert_array_equal(c, [0.5, 0.0, 0.25])
    d, c = _terms.child_term(parent_d, parent_c, 1, LINEAR, 7.0)
    np.testing.assert_array_equal(d, [1, 2, 0])
    np.testing.assert_array_equal(c, [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(parent_d, [1, 0, 0])
    np.testing.assert_array_equal(parent_c, [0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="TERM-4"):
        _terms.child_term(parent_d, parent_c, 0, PLUS, 0.1)
    with pytest.raises(ValueError, match="code"):
        _terms.child_term(parent_d, parent_c, 1, ABSENT, 0.1)
    with pytest.raises(ValueError, match="finite"):
        _terms.child_term(parent_d, parent_c, 1, PLUS, math.nan)
    with pytest.raises(ValueError, match="variable"):
        _terms.child_term(parent_d, parent_c, 3, PLUS, 0.1)
    with pytest.raises(ValueError, match="1-D"):
        _terms.child_term(parent_d[None], parent_c[None], 1, PLUS, 0.1)


@pytest.mark.parametrize(
    ("dirs", "cuts", "names", "label"),
    [
        ([0, 0], [0, 0], None, "(Intercept)"),
        ([1, 0], [0.336266, 0], None, "h(x0-0.336266)"),
        ([0, -1], [0, 0.123456789], None, "h(0.123457-x1)"),
        ([1, 0], [-1.5, 0], None, "h(x0--1.5)"),
        ([0, 2], [0, 0], None, "x1"),
        ([-1, 1], [2e-10, 123456789.0], None, "h(2e-10-x0)*h(x1-1.23457e+08)"),
        ([2, 1], [0, 3.0], ["age", "dose"], "age*h(dose-3)"),
    ],
)
def test_term_label(dirs, cuts, names, label):
    """TERM-5: factors in increasing covariate order, knots with ``.6g``."""
    assert _terms.term_label(np.array(dirs), np.array(cuts, float), names) == label


def test_term_labels_errors():
    dirs, cuts = _terms.intercept_terms(2)
    assert _terms.term_labels(dirs, cuts) == ["(Intercept)"]
    with pytest.raises(ValueError, match="names"):
        _terms.term_labels(dirs, cuts, ["a"])
    with pytest.raises(ValueError, match="one shape"):
        _terms.term_label(dirs[0], cuts, None)
    with pytest.raises(ValueError, match="code"):
        _terms.term_label(np.array([3, 0]), np.zeros(2), None)


FLOATS = st.floats(-100, 100, allow_nan=False, allow_infinity=False)
#: Multiples of 1/64: sums and products of a few of them are exact.
DYADIC = st.integers(-6400, 6400).map(lambda k: k / 64)


@st.composite
def term_tables(draw, values=FLOATS):
    """Random X and terms: every code in every slot, finite knots."""
    n, p = draw(st.integers(1, 8)), draw(st.integers(1, 3))
    X = draw(hnp.arrays(np.float64, (n, p), elements=values))
    M = draw(st.integers(1, 6))
    dirs = np.zeros((M, p), dtype=np.int8)
    cuts = np.zeros((M, p))
    for k in range(1, M):
        for j in range(p):
            dirs[k, j] = draw(st.sampled_from([0, 1, -1, 2]))
            if dirs[k, j] in (PLUS, MINUS):
                cuts[k, j] = draw(values)
    return X, dirs, cuts


def scalar_term(x, d_row, c_row):
    """TERM-3 transcribed with Python floats, for one case and one term."""
    value = 1.0
    for v, code, c in zip(x, d_row, c_row, strict=True):
        if code == PLUS:
            value *= max(v - c, 0.0)
        elif code == MINUS:
            value *= max(c - v, 0.0)
        elif code == LINEAR:
            value *= v
    return value


@given(term_tables())
def test_basis_matrix_matches_scalar_formula(table):
    """TERM-3: B[i, k] = ∏_j f(dirs[k, j], x_ij, cuts[k, j]), in increasing j."""
    X, dirs, cuts = table
    B = _terms.basis_matrix(X, dirs, cuts)
    expected = [
        [scalar_term(x, d, c) for d, c in zip(dirs, cuts, strict=True)] for x in X
    ]
    np.testing.assert_array_equal(B, np.array(expected).reshape(B.shape))


@given(term_tables(), st.randoms(use_true_random=False))
def test_basis_matrix_row_order(table, random):
    """Conventions: permuting the rows of X permutes the rows of B."""
    X, dirs, cuts = table
    perm = list(range(X.shape[0]))
    random.shuffle(perm)
    B = _terms.basis_matrix(X, dirs, cuts)
    np.testing.assert_array_equal(_terms.basis_matrix(X[perm], dirs, cuts), B[perm])


@given(term_tables(DYADIC), st.integers(-20, 20))
def test_basis_matrix_scale(table, exponent):
    """Conventions: x_j·c with its knots·c scales each term with x_j by c.

    c is a power of 2 and the data are dyadic, so the identity is exact.
    """
    X, dirs, cuts = table
    c = 2.0**exponent
    X2, cuts2 = X.copy(), cuts.copy()
    X2[:, 0] *= c
    cuts2[:, 0] *= c
    scale = np.where(dirs[:, 0] != ABSENT, c, 1.0)
    np.testing.assert_array_equal(
        _terms.basis_matrix(X2, dirs, cuts2), _terms.basis_matrix(X, dirs, cuts) * scale
    )


@given(term_tables(DYADIC), st.data())
def test_child_column_is_parent_times_factor(table, data):
    """TERM-6: column k of B is column parent[k] times the new factor."""
    X, dirs, cuts = table
    k = data.draw(st.integers(0, dirs.shape[0] - 1))
    free = np.flatnonzero(dirs[k] == ABSENT)
    if free.size == 0:
        return
    j = int(data.draw(st.sampled_from(free.tolist())))
    code = data.draw(st.sampled_from([PLUS, MINUS, LINEAR]))
    cut = float(data.draw(st.sampled_from(X[:, j].tolist())))
    d, c = _terms.child_term(dirs[k], cuts[k], j, code, cut)
    parent = _terms.basis_matrix(X, dirs[k : k + 1], cuts[k : k + 1])[:, 0]
    child = _terms.basis_matrix(X, d[None], c[None])[:, 0]
    np.testing.assert_array_equal(child, parent * _terms.factor(code, X[:, j], c[j]))


def test_s_fixture_lists():
    """The collection found the fixtures: 131 fits, 125 of them least squares."""
    assert len(S_FITS) >= 131 and len(S_LEAST_SQUARES_FITS) >= 125


def test_predict_new_points(load_fixture):
    """TERM-3 against predict.earth at points inside and outside the range.

    The four cases: degree 1, degree 2, and a linear term against the hinge
    at the minimum that ``Auto.linpreds = FALSE`` uses, which differ only
    below the training minimum (FWD-6).
    """
    fixture = load_fixture("components/predict_new_points")
    for name in ("degree1", "degree2", "linear_auto", "linear_hinge"):
        case = fixture[name]
        dirs, cuts = earth_terms(case["dirs"], case["cuts"])
        selected = np.array(case["selected_terms"]) - 1
        B = _terms.basis_matrix(np.array(case["newx"]), dirs[selected], cuts[selected])
        pred = B @ np.array(case["coefficients"])
        tol = 1e-8 * float(np.std(case["y"], ddof=1))
        np.testing.assert_allclose(pred, np.array(case["pred"]), rtol=0, atol=tol)


@pytest.mark.parametrize("name", S_LEAST_SQUARES_FITS)
def test_s_fixture_predictions(load_fixture, name):
    """TERM-3: B·coef of earth's selected terms against earth's predictions,
    on the training rows and the test rows, within 1e-8·sd(y)."""
    fixture = load_fixture(name)
    result = fixture["result"]
    dirs, cuts = earth_terms(result["dirs"], result["cuts"])
    selected = np.array(result["selected_terms"]) - 1
    coef = np.array(result["coef"])
    y = np.array(fixture["inputs"]["y"], dtype=float).reshape(-1, coef.shape[1])
    tol = 1e-8 * np.std(y, axis=0, ddof=1)
    pairs = [("X", "pred_train")]
    if fixture["inputs"]["X_test"] is not None:
        pairs.append(("X_test", "pred_test"))
    for x_key, pred_key in pairs:
        B = _terms.basis_matrix(
            np.array(fixture["inputs"][x_key]), dirs[selected], cuts[selected]
        )
        error = np.abs(B @ coef - np.array(result[pred_key]).reshape(-1, coef.shape[1]))
        assert np.all(error.max(axis=0) <= tol), (pred_key, error.max(axis=0), tol)


@pytest.mark.parametrize("name", S_FITS)
def test_s_fixture_labels(load_fixture, name):
    """TERM-5 against earth's term names: equal for degree 0 and 1, and the
    same factors for higher degrees, where earth keeps the order of
    construction and pymars the covariate order (a departure)."""
    result = load_fixture(name)["result"]
    dirs, cuts = earth_terms(result["dirs"], result["cuts"])
    ours = _terms.term_labels(dirs, cuts)
    for label, earth, degree in zip(
        ours, result["term_names"], _terms.term_degrees(dirs), strict=True
    ):
        if degree <= 1:
            assert label == earth
        else:
            assert sorted(label.split("*")) == sorted(earth.split("*"))
