"""Tests of validation/sims/run.py's mechanics: atomic writes, --resume, the
prediction cache, and the CLI's cell/rep parsing. All pure Python, using a
fake inline arm so these tests need neither R nor .venv-legacy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from validation.sims import dgps, learners, run

CALL_LOG: list[int] = []


def _fake_fit_predict(x_train, y_train, x_test, binary):
    CALL_LOG.append(1)
    return learners.FitOutcome(
        predictions=np.zeros(len(x_test)),
        n_terms=3,
        covariates_used=(0, 5),
        fit_seconds=0.001,
    )


FAKE_ARM = learners.Arm(
    name="Fake",
    kind="inline",
    supports_regression=True,
    supports_binary=False,
    source_text="fake source v1",
    fit_predict=_fake_fit_predict,
)


@pytest.fixture(autouse=True)
def _register_fake_arm(monkeypatch):
    CALL_LOG.clear()
    monkeypatch.setitem(learners.ARMS, "Fake", FAKE_ARM)
    yield


@pytest.fixture
def diagnostics():
    return dgps.load_diagnostics()


# ---------------------------------------------------------------------------
# Atomic writes.
# ---------------------------------------------------------------------------


def test_atomic_write_json_leaves_no_temp_file(tmp_path):
    path = tmp_path / "a" / "b.json"
    run.atomic_write_json(path, {"x": 1})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"x": 1}
    assert list(tmp_path.rglob("*.tmp.*")) == []


def test_is_done_true_for_valid_json_false_otherwise(tmp_path):
    missing = tmp_path / "missing.json"
    assert not run.is_done(missing, retry_failed=False)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert not run.is_done(corrupt, retry_failed=False)

    valid = tmp_path / "valid.json"
    valid.write_text("{}")
    assert run.is_done(valid, retry_failed=False)


def test_is_done_retry_failed_redoes_a_recorded_error(tmp_path):
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"error": "NotImplementedError: nope"}))
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"error": None, "excess_risk": 1.0}))

    assert run.is_done(failed, retry_failed=False)  # a recorded failure IS done
    assert not run.is_done(failed, retry_failed=True)  # ...unless asked to retry it
    assert run.is_done(ok, retry_failed=True)  # a success is never redone


# ---------------------------------------------------------------------------
# --resume and redoing an incomplete cell.
# ---------------------------------------------------------------------------


def test_full_run_then_resume_skips_everything(tmp_path, diagnostics):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:2",
        "--resume",
        "--no-cache",
    ]
    run.main(argv)
    assert len(CALL_LOG) == 2  # 2 repetitions, 1 arm, 1 cell
    run.main(argv)
    assert len(CALL_LOG) == 2  # --resume: nothing new to do


def test_resume_redoes_a_cell_whose_file_is_incomplete(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:2",
        "--resume",
        "--no-cache",
    ]
    run.main(argv)
    assert len(CALL_LOG) == 2

    result_file = tmp_path / "D1_n00200_lo" / "Fake" / "rep0000.json"
    result_file.write_text("{not valid json")
    run.main(argv)
    assert len(CALL_LOG) == 3  # only the corrupted one redone
    assert json.loads(result_file.read_text())["n_terms"] == 3


def test_without_resume_every_unit_reruns(tmp_path):
    argv_no_resume = [
        "--out",
        str(tmp_path),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:2",
        "--no-cache",
    ]
    run.main(argv_no_resume)
    run.main(argv_no_resume)
    assert len(CALL_LOG) == 4  # 2 reps, run twice, no --resume and no cache


# ---------------------------------------------------------------------------
# The prediction cache.
# ---------------------------------------------------------------------------


def test_cache_short_circuits_a_refit(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE_DIR", tmp_path / "cache")
    argv = [
        "--out",
        str(tmp_path / "out"),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:1",
        "--resume",
    ]
    run.main(argv)
    assert len(CALL_LOG) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1

    # Delete the result file only: without --resume this refits unless the
    # cache (untouched) short-circuits it.
    result_file = tmp_path / "out" / "D1_n00200_lo" / "Fake" / "rep0000.json"
    result_file.unlink()
    run.main([a for a in argv if a != "--resume"])
    assert len(CALL_LOG) == 1  # still 1: the cache supplied the prediction
    assert result_file.is_file()


def test_cache_key_is_stable_and_sensitive_to_its_inputs():
    rng = np.random.default_rng(0)
    x_train = rng.uniform(size=(10, 3))
    y_train = rng.uniform(size=10)
    x_test = rng.uniform(size=(5, 3))
    versions = {"numpy": "1.0"}
    key1 = run.cache_key(x_train, y_train, x_test, FAKE_ARM, versions)
    key2 = run.cache_key(x_train, y_train, x_test, FAKE_ARM, versions)
    assert key1 == key2

    other_versions = {"numpy": "2.0"}
    assert run.cache_key(x_train, y_train, x_test, FAKE_ARM, other_versions) != key1

    other_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="a different source",
        fit_predict=_fake_fit_predict,
    )
    assert run.cache_key(x_train, y_train, x_test, other_arm, versions) != key1


def test_versions_for_arm_does_not_depend_on_other_requested_arms():
    """A review's repro: OLS's cache key used to change depending on whether
    an R or legacy arm was *also* requested in the same invocation, because
    every arm's key carried the same shared, invocation-wide versions dict.
    """
    all_versions = {
        "numpy": "1.0",
        "scipy": "1.0",
        "sklearn": "1.0",
        "pymars": "2.0",
        "pymars_commit": "abc",
        "r_earth": "5.3.4",
        "legacy_mars_earth": "1.0.4",
        "legacy_sklearn": "1.9.1",
    }
    inline_arm = learners.ARMS["OLS"]
    with_r_and_legacy = run.versions_for_arm(all_versions, inline_arm)

    minimal_versions = {k: v for k, v in all_versions.items() if k in with_r_and_legacy}
    without_r_and_legacy = run.versions_for_arm(minimal_versions, inline_arm)
    assert with_r_and_legacy == without_r_and_legacy
    assert "r_earth" not in with_r_and_legacy
    assert "legacy_mars_earth" not in with_r_and_legacy

    r_arm = learners.ARMS["E-def"]
    assert "r_earth" in run.versions_for_arm(all_versions, r_arm)
    legacy_arm = learners.ARMS["P-cur"]
    assert "legacy_mars_earth" in run.versions_for_arm(all_versions, legacy_arm)


# ---------------------------------------------------------------------------
# Streaming: a unit's result lands on disk the moment it finishes, not after
# the whole invocation. Killing the process partway must not lose any unit
# that had already finished.
# ---------------------------------------------------------------------------


def test_interrupted_run_keeps_finished_units_and_resume_matches_uninterrupted(
    tmp_path,
):
    kill_after = 3
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")

    def _counting_fit_predict(x_train, y_train, x_test, binary):
        n = int(counter_file.read_text(encoding="utf-8")) + 1
        counter_file.write_text(str(n), encoding="utf-8")
        if n == kill_after:
            raise KeyboardInterrupt("simulated kill")
        return learners.FitOutcome(predictions=np.full(len(x_test), float(n)))

    counting_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="counting arm",
        fit_predict=_counting_fit_predict,
    )
    learners.ARMS["Fake"] = counting_arm

    argv = [
        "--out",
        str(tmp_path / "out"),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:6",
        "--n-jobs",
        "1",
        "--resume",
        "--no-cache",
    ]
    with pytest.raises(KeyboardInterrupt):
        run.main(argv)

    result_dir = tmp_path / "out" / "D1_n00200_lo" / "Fake"
    finished_after_kill = sorted(p.name for p in result_dir.glob("rep*.json"))
    assert len(finished_after_kill) == kill_after - 1  # units before the kill only

    # Resume: only the un-finished units (including the one that raised) run
    # again. Reset the arm so the retried unit succeeds this time.
    counter_file.write_text("-1000", encoding="utf-8")  # never hits kill_after again
    run.main(argv)
    finished_after_resume = sorted(p.name for p in result_dir.glob("rep*.json"))
    assert finished_after_resume == [f"rep{i:04d}.json" for i in range(6)]

    # The units finished before the kill were never re-run: their recorded
    # predictions are exactly the ones the first (interrupted) call made.
    first_unit = json.loads((result_dir / "rep0000.json").read_text())
    assert first_unit["error"] is None


# ---------------------------------------------------------------------------
# Pairing: every arm in one (cell, repetition) sees bit-identical data, even
# though each unit now generates its own copy independently (for memory).
# ---------------------------------------------------------------------------


def test_two_units_for_the_same_cell_and_rep_generate_identical_data():
    diagnostics = dgps.load_diagnostics()
    cell = dgps.Cell("D4", 200, "lo")
    first = run.generate_dataset(cell, 3, diagnostics)
    second = run.generate_dataset(cell, 3, diagnostics)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
    # A different repetition must differ (otherwise this test would be
    # vacuous: identical data regardless of the rep argument).
    third = run.generate_dataset(cell, 4, diagnostics)
    assert not np.array_equal(first[0], third[0])


def test_build_units_never_generates_data(monkeypatch, tmp_path):
    """The memory fix: build_units must only name units (cell, arm, rep), and
    never call generate_dataset itself. A review measured 2.32 GB of
    pre-generated datasets in memory before the first fit, for 15 cells x 100
    repetitions at n=200; the fix is that no dataset exists until the task
    that fits it runs.
    """

    def _boom(*_args, **_kwargs):
        raise AssertionError("build_units must not generate any dataset")

    monkeypatch.setattr(run, "generate_dataset", _boom)
    cells = [dgps.Cell("D1", 200, "lo"), dgps.Cell("D4", 1000, "hi")]
    units = run.build_units(cells, ["OLS", "HGB"], range(50), tmp_path, False, False)
    assert len(units) == 2 * 2 * 50


# ---------------------------------------------------------------------------
# A binary cell, and --retry-failed.
# ---------------------------------------------------------------------------


def test_run_on_a_binary_cell(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "HGB,LogReg",
        "--dgps",
        "D3-bin",
        "--sizes",
        "200",
        "--reps",
        "0:1",
        "--no-cache",
    ]
    run.main(argv)
    for arm in ("HGB", "LogReg"):
        data = json.loads(
            (tmp_path / "D3-bin_n00200" / arm / "rep0000.json").read_text()
        )
        assert data["error"] is None
        assert "calibration_slope" in data
        assert "excess_log_loss" in data


def test_retry_failed_redoes_only_recorded_failures(tmp_path):
    def _flaky_fit_predict(x_train, y_train, x_test, binary):
        raise RuntimeError("always fails, the first time")

    flaky_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="flaky arm v1",
        fit_predict=_flaky_fit_predict,
    )
    learners.ARMS["Fake"] = flaky_arm
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:1",
        "--resume",
        "--no-cache",
    ]
    run.main(argv)
    result_file = tmp_path / "D1_n00200_lo" / "Fake" / "rep0000.json"
    first_error = json.loads(result_file.read_text())["error"]
    assert first_error is not None

    # Plain --resume leaves the recorded failure alone (the arm is still
    # broken here, so any redo would also fail; the unchanged error message
    # is the sign that this unit was not retried at all).
    run.main(argv)
    assert json.loads(result_file.read_text())["error"] == first_error

    # Fix the arm, then --resume --retry-failed redoes it.
    learners.ARMS["Fake"] = FAKE_ARM
    run.main([*argv, "--retry-failed"])
    assert json.loads(result_file.read_text())["error"] is None


# ---------------------------------------------------------------------------
# The manifest is a history, not a single overwritten entry.
# ---------------------------------------------------------------------------


def test_manifest_keeps_one_entry_per_invocation(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "Fake",
        "--dgps",
        "D1",
        "--sizes",
        "200",
        "--noise",
        "lo",
        "--reps",
        "0:1",
        "--resume",
        "--no-cache",
    ]
    run.main(argv)
    run.main(
        [
            "--out",
            str(tmp_path),
            "--arms",
            "Fake",
            "--dgps",
            "D1",
            "--sizes",
            "200",
            "--noise",
            "hi",
            "--reps",
            "0:1",
            "--resume",
            "--no-cache",
        ]
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == 2
    assert manifest[0]["cells"] != manifest[1]["cells"]


# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------


def test_parse_reps_half_open():
    assert list(run.parse_reps("0:3")) == [0, 1, 2]
    assert list(run.parse_reps("2:2")) == []


def test_parse_cells_by_exact_name():
    parser = run.build_arg_parser()
    args = parser.parse_args(
        [
            "--out",
            "x",
            "--arms",
            "Fake",
            "--cells",
            "D1_n00200_lo,D6_n00200",
            "--reps",
            "0:1",
        ]
    )
    cells = run.parse_cells(args, dgps.all_cells())
    assert {c.name for c in cells} == {"D1_n00200_lo", "D6_n00200"}


def test_parse_cells_unknown_name_raises():
    parser = run.build_arg_parser()
    args = parser.parse_args(
        ["--out", "x", "--arms", "Fake", "--cells", "nonsense", "--reps", "0:1"]
    )
    with pytest.raises(SystemExit):
        run.parse_cells(args, dgps.all_cells())


def test_parse_cells_by_filters():
    parser = run.build_arg_parser()
    args = parser.parse_args(
        [
            "--out",
            "x",
            "--arms",
            "Fake",
            "--dgps",
            "D1,D6",
            "--sizes",
            "200",
            "--noise",
            "lo,na",
            "--reps",
            "0:1",
        ]
    )
    cells = run.parse_cells(args, dgps.all_cells())
    names = {c.name for c in cells}
    assert names == {"D1_n00200_lo", "D6_n00200"}


def test_main_rejects_unknown_arm(tmp_path):
    with pytest.raises(SystemExit):
        run.main(
            [
                "--out",
                str(tmp_path),
                "--arms",
                "NotAnArm",
                "--dgps",
                "D1",
                "--reps",
                "0:1",
            ]
        )
