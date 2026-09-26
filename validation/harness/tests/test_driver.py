"""Tests for driver.py.

The CSV-writing and column-naming tests need no R. ``TestRunEarth`` and
``test_run_earth_reports_the_r_error`` call ``Rscript fit_earth.R`` for
real, so they need R and the earth package; both are marked ``external``
(gate A/B and CI exclude it; gate C and manual runs include it).
"""

import csv
import math

import numpy as np
import pytest
from driver import (
    EarthJob,
    _desanitize,
    _x_columns,
    _y_columns,
    run_earth,
    versions,
    write_csv,
)


def _basis_from_dirs_cuts(
    X: np.ndarray, dirs: np.ndarray, cuts: np.ndarray
) -> np.ndarray:
    """Rebuild earth's basis matrix from its dirs/cuts convention
    (VALIDATION_PLAN.md, "Term structure"): dirs code 0 absent, +1 a right
    hinge (x - c)+, -1 a left hinge (c - x)+, 2 linear (x); a row's nonzero
    codes multiply together into one term (one column). Used to check
    fit_earth.R's forward RSS path against an independently computed OLS
    fit, not against another call into R.
    """
    n = X.shape[0]
    B = np.ones((n, dirs.shape[0]))
    for term in range(dirs.shape[0]):
        for var in range(dirs.shape[1]):
            code = dirs[term, var]
            if code == 0:
                continue
            x = X[:, var]
            if code == 1:
                factor = np.maximum(0.0, x - cuts[term, var])
            elif code == -1:
                factor = np.maximum(0.0, cuts[term, var] - x)
            elif code == 2:
                factor = x
            else:
                raise ValueError(f"unknown dirs code {code!r}")
            B[:, term] *= factor
    return B


class TestDesanitize:
    def test_converts_inf_and_nan(self):
        out = _desanitize(["Inf", "-Inf", "NaN", 1.0])
        assert out[0] == math.inf
        assert out[1] == -math.inf
        assert math.isnan(out[2])
        assert out[3] == 1.0

    def test_na_becomes_none_distinct_from_nan(self):
        # Review round 1, finding 3: NA and NaN must stay distinguishable,
        # not both collapse to the same float NaN.
        out = _desanitize(["NA", "NaN"])
        assert out[0] is None
        assert isinstance(out[1], float) and math.isnan(out[1])

    def test_string_only_fields_are_never_converted_even_if_they_spell_a_sentinel(self):
        # A term name, an id, a factor level or a version string that
        # happens to equal "NA", "NaN" or "Inf" must stay that literal
        # string, not become None/NaN/inf.
        out = _desanitize(
            {
                "id": "NA",
                "term_names": ["NaN", "Inf"],
                "levels": ["NA", "b"],
                "r_version": "Inf",
                "earth_version": "NaN",
                "error": "NA",
                "call": "Inf",
            }
        )
        assert out == {
            "id": "NA",
            "term_names": ["NaN", "Inf"],
            "levels": ["NA", "b"],
            "r_version": "Inf",
            "earth_version": "NaN",
            "error": "NA",
            "call": "Inf",
        }

    def test_other_strings_pass_through_unchanged(self):
        assert _desanitize("(Intercept)") == "(Intercept)"
        assert _desanitize({"gcv": 1.5}) == {"gcv": 1.5}

    def test_recurses_into_nested_lists_and_dicts(self):
        out = _desanitize({"dirs": [["Inf", 0], [0, "NaN"]]})
        assert out["dirs"][0][0] == math.inf
        assert math.isnan(out["dirs"][1][1])


class TestWriteCsv:
    def test_numeric_columns_are_hex_floats(self, tmp_path):
        # Python-side check only: that write_csv emits float.hex() text,
        # which Python's own float() reads back exactly. Whether R's
        # read.csv also reads this text to the identical double (the actual
        # point of using hex floats; %.17g decimal text does not, on a
        # build with no long double) is checked directly in
        # test_precision.py's TestCsvHexFloatPrecision, which calls R.
        rng = np.random.default_rng(0)
        x = rng.uniform(size=20) * np.array([1e-8, 1e8] * 10)
        path = tmp_path / "d.csv"
        write_csv(path, {"x0": x})
        lines = path.read_text().splitlines()[1:]
        assert all(line.startswith(("0x", "-0x")) for line in lines)
        back = np.array([float.fromhex(line) for line in lines])
        assert np.array_equal(back, x)

    def test_writes_a_header_and_comma_separated_columns(self, tmp_path):
        path = tmp_path / "d.csv"
        write_csv(path, {"x0": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])})
        lines = path.read_text().splitlines()
        assert lines[0] == "x0,y"
        assert lines[1] == f"{(1.0).hex()},{(3.0).hex()}"
        assert lines[2] == f"{(2.0).hex()},{(4.0).hex()}"

    def test_string_columns_are_written_as_plain_text(self, tmp_path):
        path = tmp_path / "d.csv"
        write_csv(path, {"y": np.array(["a", "b", "c"])})
        assert path.read_text().splitlines() == ["y", "a", "b", "c"]

    def test_a_label_containing_a_comma_is_csv_quoted(self, tmp_path):
        path = tmp_path / "d.csv"
        write_csv(path, {"y": np.array(["plain", "has,comma", 'has"quote'])})
        lines = path.read_text().splitlines()
        assert lines[1:] == ["plain", '"has,comma"', '"has""quote"']
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        assert [r[0] for r in rows[1:]] == ["plain", "has,comma", 'has"quote']

    def test_rejects_mismatched_column_lengths(self, tmp_path):
        cols = {"x0": np.array([1.0, 2.0]), "y": np.array([1.0])}
        with pytest.raises(ValueError, match="same length"):
            write_csv(tmp_path / "d.csv", cols)


class TestColumnNaming:
    def test_x_columns_are_x0_x1(self):
        X = np.arange(6.0).reshape(3, 2)
        cols = _x_columns(X)
        assert list(cols) == ["x0", "x1"]
        assert np.array_equal(cols["x0"], [0.0, 2.0, 4.0])
        assert np.array_equal(cols["x1"], [1.0, 3.0, 5.0])

    def test_1d_x_is_n_samples_of_one_feature_not_one_sample_of_n_features(self):
        # np.atleast_2d(X) would instead give a (1, n) row, one sample.
        X = np.array([10.0, 20.0, 30.0])
        cols = _x_columns(X)
        assert list(cols) == ["x0"]
        assert np.array_equal(cols["x0"], [10.0, 20.0, 30.0])

    def test_1d_y_is_a_single_y_column(self):
        cols = _y_columns(np.array([1.0, 2.0, 3.0]), factor_response=False)
        assert list(cols) == ["y"]

    def test_2d_single_column_y_is_also_a_single_y_column(self):
        cols = _y_columns(np.array([[1.0], [2.0]]), factor_response=False)
        assert list(cols) == ["y"]

    def test_multi_response_y_is_y1_y2(self):
        Y = np.arange(6.0).reshape(3, 2)
        cols = _y_columns(Y, factor_response=False)
        assert list(cols) == ["y1", "y2"]
        assert np.array_equal(cols["y1"], [0.0, 2.0, 4.0])
        assert np.array_equal(cols["y2"], [1.0, 3.0, 5.0])

    def test_factor_response_is_a_single_y_column_of_labels(self):
        cols = _y_columns(np.array(["a", "b", "a"]), factor_response=True)
        assert list(cols) == ["y"]
        assert list(cols["y"]) == ["a", "b", "a"]

    def test_factor_response_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1-D"):
            _y_columns(np.zeros((3, 2)), factor_response=True)

    def test_y_rejects_more_than_2d(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _y_columns(np.zeros((2, 2, 2)), factor_response=False)


class TestVersions:
    def test_reports_python_side_versions(self):
        info = versions()
        assert info["python"]
        assert info["numpy"]
        assert info["scikit_learn"]
        assert isinstance(info["blas"], list)
        # Always a dict with these two keys, but not always populated: numpy
        # before 2.0 (the "lowest direct dependencies" CI job) cannot report
        # its own BLAS build metadata, so both are None there instead of a
        # name/version string; test_numpy_blas_reads_the_dicts_mode_build_
        # dependencies_entry pins the populated case without depending on
        # this machine's own numpy build.
        assert set(info["numpy_blas"]) == {"name", "version"}
        assert "r_version" not in info
        assert "r_blas" not in info

    def test_merges_in_the_earth_result_versions(self):
        info = versions({"r_version": "R version 4.4.3", "earth_version": "5.3.4"})
        assert info["r_version"] == "R version 4.4.3"
        assert info["earth_version"] == "5.3.4"

    def test_r_blas_is_none_without_earth_result(self):
        assert "r_blas" not in versions()

    def test_missing_rscript_gives_none_not_an_error(self):
        info = versions(
            {"r_version": "x", "earth_version": "y"}, rscript="no-such-r-binary"
        )
        assert info["r_blas"] is None

    @pytest.mark.external
    def test_r_blas_is_populated_when_rscript_is_real(self):
        info = versions({"r_version": "x", "earth_version": "y"})
        assert info["r_blas"]["la_library"]
        assert info["r_blas"]["blas"]

    def test_numpy_blas_reads_the_dicts_mode_build_dependencies_entry(
        self, monkeypatch
    ):
        def fake_show_config(mode=None):
            assert mode == "dicts"
            return {
                "Build Dependencies": {"blas": {"name": "fake_blas", "version": "9.9"}}
            }

        monkeypatch.setattr(np, "show_config", fake_show_config)
        assert versions()["numpy_blas"] == {"name": "fake_blas", "version": "9.9"}

    def test_numpy_blas_degrades_to_none_on_a_pre_2_0_numpy(self, monkeypatch):
        """numpy < 2.0's show_config() takes no arguments (and prints instead
        of returning a dict), so the "mode" keyword raises TypeError; this
        must not crash versions(), just leave numpy_blas empty (regression:
        the "lowest direct dependencies" CI job installs numpy 1.23.5, the
        floor in dev/DECISIONS.md, and hit exactly this)."""

        def old_show_config():
            return None

        monkeypatch.setattr(np, "show_config", old_show_config)
        assert versions()["numpy_blas"] == {"name": None, "version": None}


@pytest.mark.external
class TestRunEarth:
    """These call Rscript fit_earth.R for real; they need R and earth."""

    def _job(self, id_, rng, n=60):
        X = rng.uniform(size=(n, 1))
        y = (
            2 * np.maximum(0, X[:, 0] - 0.3)
            - 3 * np.maximum(0, X[:, 0] - 0.7)
            + rng.normal(scale=0.05, size=n)
        )
        return EarthJob(
            id=id_,
            X=X,
            y=y,
            earth_args={
                "degree": 1,
                "penalty": 2,
                "nk": 11,
                "thresh": 0,
                "minspan": 1,
                "endspan": 1,
                "fast.k": 0,
                "Auto.linpreds": False,
                "pmethod": "backward",
            },
        )

    def test_single_job_result_has_the_documented_schema(self, tmp_path):
        rng = np.random.default_rng(0)
        results = run_earth([self._job("s01", rng)], workdir=tmp_path)
        r = results["s01"]
        expected_keys = {
            "id",
            "dirs",
            "cuts",
            "term_names",
            "selected_terms",
            "prune_terms",
            "rss_per_subset",
            "gcv_per_subset",
            "coef",
            "glm_coef",
            "rss",
            "rsq",
            "gcv",
            "grsq",
            "termcond",
            "levels",
            "fitted",
            "pred_train",
            "pred_test",
            "fwd_rss",
            "r_version",
            "earth_version",
        }
        assert expected_keys <= set(r)
        assert r["id"] == "s01"
        assert r["glm_coef"] is None  # no glm family was requested
        assert r["pred_test"] is None  # no test set was requested
        dirs = np.array(r["dirs"])
        cuts = np.array(r["cuts"])
        assert dirs.ndim == 2 and cuts.shape == dirs.shape
        assert np.array(r["fitted"]).shape == (60, 1)
        assert r["r_version"] and r["earth_version"]

    def test_test_set_predictions_are_returned_when_requested(self, tmp_path):
        rng = np.random.default_rng(1)
        job = self._job("s01", rng)
        job.X_test = rng.uniform(size=(5, 1))
        r = run_earth([job], workdir=tmp_path)["s01"]
        assert np.array(r["pred_test"]).shape == (5, 1)

    def test_a_block_fits_every_job_in_one_r_process(self, tmp_path):
        rng = np.random.default_rng(2)
        jobs = [self._job(f"job{k}", rng, n=30 + 10 * k) for k in range(3)]
        results = run_earth(jobs, workdir=tmp_path)
        assert set(results) == {"job0", "job1", "job2"}
        for k, job in enumerate(jobs):
            assert np.array(results[job.id]["fitted"]).shape[0] == 30 + 10 * k

    def test_weights_are_forwarded(self, tmp_path):
        # Pin the exact relationship, not just finiteness (a one-line
        # mutation that drops args$weights <- train$w in fit_earth.R would
        # still return a finite rss, just the unweighted one): earth's own
        # weighted rss must equal the weighted sum of squared residuals on
        # the fitted values it returns, which only holds when the weights
        # it received were the ones actually used to fit.
        rng = np.random.default_rng(3)
        job = self._job("s01", rng)
        job.weights = rng.integers(1, 4, size=job.X.shape[0]).astype(float)
        r = run_earth([job], workdir=tmp_path)["s01"]
        fitted = np.array(r["fitted"]).ravel()
        resid = job.y - fitted
        expected_rss = float(np.sum(job.weights * resid**2))
        assert r["rss"] == pytest.approx(expected_rss, rel=1e-6)

        unweighted = EarthJob(id="s01", X=job.X, y=job.y, earth_args=job.earth_args)
        r_plain = run_earth([unweighted], workdir=tmp_path / "plain")["s01"]
        assert r["rss"] != pytest.approx(r_plain["rss"])

    def test_forward_rss_path_matches_an_independent_ols_refit(self, tmp_path):
        # Independent of fit_earth.R's own lm.fit/lm.wfit call: rebuild each
        # cumulative basis column from dirs/cuts (earth's dirs/cuts
        # convention: code 0 absent, +1 right hinge, -1 left hinge, 2
        # linear, a term's factors multiplied together) and solve by plain
        # numpy least squares. A one-line mutation that refits on the first
        # response column only, or a constant/near-constant path, would
        # fail either the value match or the "must decrease" check below.
        rng = np.random.default_rng(4)
        job = self._job("s01", rng)
        job.earth_args = {**job.earth_args, "pmethod": "none"}
        r = run_earth([job], workdir=tmp_path)["s01"]
        fwd = np.array(r["fwd_rss"])
        assert len(fwd) >= 2
        assert np.all(np.diff(fwd) <= 1e-8)
        assert fwd[0] > fwd[-1] * 1.5  # genuinely decreases, not a flat path

        dirs, cuts = np.array(r["dirs"]), np.array(r["cuts"])
        B = _basis_from_dirs_cuts(job.X, dirs, cuts)
        last_terms = B.shape[1]
        coef, *_ = np.linalg.lstsq(B[:, :last_terms], job.y, rcond=None)
        independent_rss = float(np.sum((job.y - B[:, :last_terms] @ coef) ** 2))
        assert fwd[-1] == pytest.approx(independent_rss, rel=1e-6)

    def test_include_forward_path_false_omits_it(self, tmp_path):
        rng = np.random.default_rng(5)
        job = self._job("s01", rng)
        job.include_forward_path = False
        r = run_earth([job], workdir=tmp_path)["s01"]
        assert r["fwd_rss"] is None

    def test_glm_family_adds_glm_coef(self, tmp_path):
        rng = np.random.default_rng(6)
        n = 80
        X = rng.uniform(size=(n, 2))
        p = 1 / (1 + np.exp(-(3 * X[:, 0] - 2 * X[:, 1])))
        y = (rng.uniform(size=n) < p).astype(float)
        job = EarthJob(
            id="glm", X=X, y=y, earth_args={"degree": 1}, glm_family="binomial"
        )
        r = run_earth([job], workdir=tmp_path)["glm"]
        assert r["glm_coef"] is not None
        probs = np.array(r["pred_train"]).ravel()
        assert np.all((probs >= 0) & (probs <= 1))

    def test_factor_response_reports_levels_and_skips_forward_path(self, tmp_path):
        rng = np.random.default_rng(7)
        n = 60
        X = rng.uniform(size=(n, 2))
        labels = rng.choice(["a", "b", "c"], size=n)
        job = EarthJob(
            id="fac", X=X, y=labels, earth_args={"degree": 1}, factor_response=True
        )
        r = run_earth([job], workdir=tmp_path)["fac"]
        assert r["levels"] == ["a", "b", "c"]
        assert r["fwd_rss"] is None
        assert np.array(r["coef"]).shape[1] == 3

    def test_multi_response_y_gives_one_coefficient_column_per_response(self, tmp_path):
        rng = np.random.default_rng(8)
        n = 60
        X = rng.uniform(size=(n, 2))
        Y = np.column_stack([X[:, 0] * 2, X[:, 1] ** 2])
        job = EarthJob(id="multi", X=X, y=Y, earth_args={"degree": 1})
        r = run_earth([job], workdir=tmp_path)["multi"]
        assert np.array(r["coef"]).shape[1] == 2
        assert np.array(r["fitted"]).shape == (60, 2)

    def test_trace_capture_writes_a_findknot_log(self, tmp_path):
        rng = np.random.default_rng(9)
        job = self._job("s01", rng)
        job.trace = 9
        run_earth([job], workdir=tmp_path)
        trace_file = tmp_path / "s01_trace.txt"
        assert trace_file.is_file()
        assert "FindKnotBegin" in trace_file.read_text()


@pytest.mark.external
def test_run_earth_reports_the_r_error(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(20, 1))
    y = X[:, 0]
    # An earth argument that does not exist raises inside fit_earth.R;
    # run_earth must surface that message, not a bare non-zero exit.
    job = EarthJob(id="bad", X=X, y=y, earth_args={"not_a_real_argument": 1})
    with pytest.raises(RuntimeError, match="earth failed for job 'bad'"):
        run_earth([job], workdir=tmp_path)
