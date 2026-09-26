"""Tests of validation/sims/run.py's mechanics: atomic writes, --resume, the
prediction cache, and the CLI's cell/rep parsing. All pure Python, using a
fake inline arm so these tests need neither R nor .venv-legacy.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import signal

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
    """A review found that this test, as it stood, passed with the data,
    ``arm.config`` and the arm's name all left out of the hash entirely
    (only the versions and source_text were ever varied): without the data
    in the key, every repetition of an arm would share one cache entry, and
    the cache would return the first repetition's predictions for all of
    them. Each assertion below isolates one input.
    """
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

    # The data: a different repetition's draw must not share a key.
    other_x_train = rng.uniform(size=(10, 3))
    assert run.cache_key(other_x_train, y_train, x_test, FAKE_ARM, versions) != key1
    other_y_train = rng.uniform(size=10)
    assert run.cache_key(x_train, other_y_train, x_test, FAKE_ARM, versions) != key1
    other_x_test = rng.uniform(size=(5, 3))
    assert run.cache_key(x_train, y_train, other_x_test, FAKE_ARM, versions) != key1

    # arm.config: two arms with the same name and source but different
    # settings (as E-def and E-pym, or P-cur and P-ear, do) must not collide.
    differently_configured_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text=FAKE_ARM.source_text,
        config={"penalty": 99},
        fit_predict=_fake_fit_predict,
    )
    assert (
        run.cache_key(x_train, y_train, x_test, differently_configured_arm, versions)
        != key1
    )

    # The arm's name: two arms that otherwise share everything (as two
    # differently-named registrations of the same code would) must not
    # collide either.
    differently_named_arm = learners.Arm(
        name="OtherFakeName",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text=FAKE_ARM.source_text,
        fit_predict=_fake_fit_predict,
    )
    assert (
        run.cache_key(x_train, y_train, x_test, differently_named_arm, versions) != key1
    )


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


def test_a_failed_fit_is_not_cached(tmp_path, monkeypatch):
    """_process_inline_unit only calls cache_put when outcome.ok. A review's
    mutation caching every outcome regardless of outcome.error passed every
    prior test, since none of them inspected the cache's own contents after
    a failing fit; a later, successful --retry-failed run would otherwise
    have replayed the cached failure forever.
    """
    monkeypatch.setattr(run, "CACHE_DIR", tmp_path / "cache")

    def _always_fails(x_train, y_train, x_test, binary):
        raise RuntimeError("always fails")

    failing_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="failing arm",
        fit_predict=_always_fails,
    )
    learners.ARMS["Fake"] = failing_arm

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
    ]  # cache left on: no --no-cache
    run.main(argv)

    result = json.loads(
        (tmp_path / "out" / "D1_n00200_lo" / "Fake" / "rep0000.json").read_text()
    )
    assert result["error"] is not None
    assert list((tmp_path / "cache").glob("*.json")) == []


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
# Failure isolation: one bad block, or one measure computation failure, must
# not stop the run. Both are Exception (not KeyboardInterrupt/SystemExit)
# paths, so unlike the kill test above, run.main must return normally.
# ---------------------------------------------------------------------------


def test_a_crashing_block_does_not_stop_the_run(tmp_path):
    """_process_block_chunk's own try/except turns a crashing run_block into
    a recorded error for every unit in that block, instead of letting the
    exception escape and abort every later block in the same invocation. A
    review's mutation (letting the exception propagate) still passed every
    prior test, since none of them used a block ("r"/"legacy") kind arm whose
    run_block itself raises.
    """

    def _always_crashes(jobs):
        raise RuntimeError("block subprocess exploded")

    crashing_arm = learners.Arm(
        name="Fake",
        kind="r",
        supports_regression=True,
        supports_binary=False,
        source_text="crashing block arm",
        run_block=_always_crashes,
    )
    learners.ARMS["Fake"] = crashing_arm

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
        "--no-cache",
    ]
    run.main(argv)  # must not raise

    result_dir = tmp_path / "D1_n00200_lo" / "Fake"
    for rep in range(2):
        data = json.loads((result_dir / f"rep{rep:04d}.json").read_text())
        assert data["error"] is not None
        assert "RuntimeError" in data["error"]


def test_a_measure_computation_failure_does_not_stop_the_run(tmp_path):
    """_write_measures's own try/except turns a measure-computation failure
    (metrics.py rejects a non-finite prediction) into a recorded error
    instead of letting it escape and abort the run. A review's mutation
    (removing that try/except) still passed every prior test, since no
    existing arm ever produced a non-finite prediction.
    """
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")

    def _nan_then_ok_fit_predict(x_train, y_train, x_test, binary):
        n = int(counter_file.read_text(encoding="utf-8")) + 1
        counter_file.write_text(str(n), encoding="utf-8")
        value = np.nan if n == 1 else 0.0
        return learners.FitOutcome(predictions=np.full(len(x_test), value))

    flaky_measures_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="nan then ok arm",
        fit_predict=_nan_then_ok_fit_predict,
    )
    learners.ARMS["Fake"] = flaky_measures_arm

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
        "0:2",
        "--n-jobs",
        "1",
        "--no-cache",
    ]
    run.main(argv)  # must not raise, despite rep 0's non-finite prediction

    result_dir = tmp_path / "out" / "D1_n00200_lo" / "Fake"
    rep0 = json.loads((result_dir / "rep0000.json").read_text())
    rep1 = json.loads((result_dir / "rep0001.json").read_text())
    assert rep0["error"] is not None  # metrics.py rejected the NaN prediction
    assert rep1["error"] is None
    assert "excess_risk" in rep1


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


def _probe_fit_predict(x_train, y_train, x_test, binary):
    """Records a hash of what it was actually given, not what it was asked
    for: a plain module-level function (not a closure), so it pickles safely
    across joblib's process boundary. The hash rides home inside the result
    file's ``extra`` field, since a probe running in a loky worker cannot
    hand data back to the test process any other way.
    """
    hasher = hashlib.sha256()
    for arr in (x_train, y_train, x_test):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return learners.FitOutcome(
        predictions=np.zeros(len(x_test)), extra={"data_hash": hasher.hexdigest()}
    )


def test_two_probe_arms_receive_identical_data_through_run_main(tmp_path):
    """Pairing, through the real run.main/joblib scheduling path, not by
    calling generate_dataset directly twice with the same arguments (a
    review found that does not exercise the scheduling code at all, and
    would still pass if every arm computed its own, different, effective
    repetition). Two independently registered arms each record a hash of the
    (x_train, y_train, x_test) they were given; the hashes must agree for
    every (cell, repetition).
    """
    probe_a = learners.Arm(
        name="ProbeA",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="probe a",
        fit_predict=_probe_fit_predict,
    )
    probe_b = learners.Arm(
        name="ProbeB",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="probe b",
        fit_predict=_probe_fit_predict,
    )
    learners.ARMS["ProbeA"] = probe_a
    learners.ARMS["ProbeB"] = probe_b
    try:
        run.main(
            [
                "--out",
                str(tmp_path),
                "--arms",
                "ProbeA,ProbeB",
                "--dgps",
                "D1",
                "--sizes",
                "200",
                "--noise",
                "lo",
                "--reps",
                "0:3",
                "--n-jobs",
                "1",  # joblib runs n_jobs=1 sequentially in this process, so a
                # dynamically-registered test arm (not in learners.ARMS at
                # module-import time) is visible; a real loky worker process,
                # used at n_jobs > 1, re-imports learners fresh and would not
                # see it. Still exercises the real run.main/build_units/
                # run_units/_process_inline_unit path, which is the point.
                "--no-cache",
            ]
        )
        for rep in range(3):
            a = json.loads(
                (
                    tmp_path / "D1_n00200_lo" / "ProbeA" / f"rep{rep:04d}.json"
                ).read_text()
            )
            b = json.loads(
                (
                    tmp_path / "D1_n00200_lo" / "ProbeB" / f"rep{rep:04d}.json"
                ).read_text()
            )
            assert a["extra"]["data_hash"] == b["extra"]["data_hash"], rep
    finally:
        learners.ARMS.pop("ProbeA", None)
        learners.ARMS.pop("ProbeB", None)


def test_generate_dataset_differs_across_repetitions_so_pairing_is_not_vacuous():
    """The pairing check above compares two probes' hashes per repetition;
    this is only a meaningful check because different repetitions really do
    get different data. If one arm silently used a different effective
    repetition than another (a review's named bug shape: "if each arm gets
    its own repetition offset in _process_inline_unit"), the hashes above
    would then disagree, exactly because of the fact checked here.
    """
    diagnostics = dgps.load_diagnostics()
    cell = dgps.Cell("D1", 200, "lo")
    rep2 = run.generate_dataset(cell, 2, diagnostics)
    rep3 = run.generate_dataset(cell, 3, diagnostics)
    assert not np.array_equal(rep2[0], rep3[0])
    assert not np.array_equal(rep2[1], rep3[1])


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
# Fixed configuration: the held-out test set size, one BLAS thread per
# process, and turning SIGTERM/SIGINT into a clean exit without leaking the
# handler into whatever process called main() (for example a pytest session).
# ---------------------------------------------------------------------------


def test_test_n_produces_10000_test_rows():
    """VALIDATION_PLAN.md sizes every arm's fit against a fixed, large held-
    out test set. A review's repro: changing TEST_N to 1,000 (a plausible
    typo for the smoke test's own train size) left every prior test green,
    because none of them checked the test set's own row count.
    """
    diagnostics = dgps.load_diagnostics()
    cell = dgps.Cell("D1", 200, "lo")
    _x_train, _y_train, x_test, y_test, truth_test = run.generate_dataset(
        cell, 0, diagnostics
    )
    assert run.TEST_N == 10_000
    assert len(x_test) == 10_000
    assert len(y_test) == 10_000
    assert len(truth_test) == 10_000


def test_blas_thread_env_vars_default_to_one_on_import(monkeypatch):
    """run.py sets one BLAS thread per process before numpy loads (module
    docstring), so a caller who forgot to source dev/env.sh still gets a
    single-threaded BLAS in every worker; a review measured 5 OpenMP threads
    per worker at --n-jobs 2 without this. setdefault must not override a
    value a caller already set (checked here by leaving one variable set to
    something else first); importlib.reload re-executes run.py's module-
    level for loop, the code under test, without needing a fresh process.
    """
    names = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "7")  # an explicit value must survive

    try:
        importlib.reload(run)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
        assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "7"  # untouched by setdefault
    finally:
        importlib.reload(run)  # re-run once more against the restored env


def test_main_installs_signal_handlers_while_running_and_restores_them_after(
    tmp_path,
):
    """Finding 22: main(), called directly (not only as `python -m ...`),
    must not leave its SIGTERM handler installed in the calling process once
    it returns; a review found a pytest session kept run.py's handler after
    test_run.py's own tests finished. A mid-run snapshot (not just a before/
    after comparison) also catches a mutation that deletes the install call
    entirely: without it, "handler never touched" and "handler installed
    then correctly restored" would look identical from outside.
    """
    sentinel_term = signal.getsignal(signal.SIGTERM)
    during_run: list = []

    def _snapshotting_fit_predict(x_train, y_train, x_test, binary):
        during_run.append(signal.getsignal(signal.SIGTERM))
        return learners.FitOutcome(predictions=np.zeros(len(x_test)))

    snapshotting_arm = learners.Arm(
        name="Fake",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text="snapshotting arm",
        fit_predict=_snapshotting_fit_predict,
    )
    learners.ARMS["Fake"] = snapshotting_arm

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
        "--no-cache",
    ]
    run.main(argv)

    assert during_run  # the fake arm's fit_predict really ran
    assert during_run[0] is not sentinel_term  # main() had installed its own
    assert signal.getsignal(signal.SIGTERM) is sentinel_term  # ...then restored it


# ---------------------------------------------------------------------------
# The lock file: two run.py invocations must not write into one output
# folder at once.
# ---------------------------------------------------------------------------


def test_acquire_lock_raises_when_one_is_already_held(tmp_path):
    lock_path = run.acquire_lock(tmp_path)
    assert lock_path == tmp_path / "run.lock"
    assert lock_path.is_dir()
    with pytest.raises(SystemExit, match=r"run\.lock"):
        run.acquire_lock(tmp_path)
    lock_path.rmdir()
    assert run.acquire_lock(tmp_path) == lock_path  # released: acquired again


def test_main_refuses_to_run_when_the_lock_already_exists(tmp_path):
    (tmp_path / "run.lock").mkdir()
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
        "--no-cache",
    ]
    with pytest.raises(SystemExit, match=r"run\.lock"):
        run.main(argv)


def test_main_releases_its_lock_on_a_clean_exit(tmp_path):
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
        "--no-cache",
    ]
    run.main(argv)
    assert not (tmp_path / "run.lock").exists()
    run.main(argv)  # would raise SystemExit here if the lock had leaked


# ---------------------------------------------------------------------------
# Block sizing: a legacy arm's own block is always exactly one repetition
# (finding 17), regardless of --block-size, since its interpreter start-up is
# negligible next to one fit and a larger block risks more finished work on a
# kill; an R (earth) arm keeps the requested block size, since its fits take
# milliseconds and batching amortizes Rscript's own start-up.
# ---------------------------------------------------------------------------


def test_block_size_for_is_always_1_for_a_legacy_arm():
    legacy_arm = learners.ARMS["P-cur"]
    r_arm = learners.ARMS["E-def"]
    assert run.block_size_for(legacy_arm, 20) == 1
    assert run.block_size_for(legacy_arm, 1) == 1
    assert run.block_size_for(r_arm, 20) == 20
    assert run.block_size_for(r_arm, 1) == 1


def test_legacy_arm_block_killed_after_its_first_fit_keeps_that_fit_on_disk(
    tmp_path,
):
    """Finding 17. block_size_for forces a legacy arm to block size 1, so a
    kill can only cost the one repetition whose own block is still open:
    every earlier repetition already has its own, separate, already-
    completed block, whose result file was written before the kill. Passing
    --block-size 20 here (batching every repetition into one block, a review
    measured 2 of 4 legacy fits lost that way) and getting only rep0000.json
    written proves block_size_for, not the flag, is what limits the damage:
    if block_size_for stopped forcing 1 for a "legacy" arm, this block would
    batch all 3 repetitions, run_block would never return (the kill happens
    inside it), and _process_block_chunk would then write no result file at
    all, not even rep0000's.
    """
    kill_after = 2
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")

    def _counting_run_block(jobs):
        outcomes = {}
        for job in jobs:
            n = int(counter_file.read_text(encoding="utf-8")) + 1
            counter_file.write_text(str(n), encoding="utf-8")
            if n == kill_after:
                raise KeyboardInterrupt("simulated kill mid-block")
            outcomes[job.job_id] = learners.FitOutcome(
                predictions=np.full(len(job.x_test), float(n))
            )
        return outcomes

    counting_legacy_arm = learners.Arm(
        name="Fake",
        kind="legacy",  # block_size_for forces block size 1 for this kind
        supports_regression=True,
        supports_binary=False,
        source_text="counting legacy arm",
        run_block=_counting_run_block,
    )
    learners.ARMS["Fake"] = counting_legacy_arm

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
        "0:3",
        "--n-jobs",
        "1",
        "--block-size",
        "20",  # would batch all 3 reps into one block if not forced to 1
        "--resume",
        "--no-cache",
    ]
    with pytest.raises(KeyboardInterrupt):
        run.main(argv)

    result_dir = tmp_path / "out" / "D1_n00200_lo" / "Fake"
    finished = sorted(p.name for p in result_dir.glob("rep*.json"))
    assert finished == ["rep0000.json"]
    assert json.loads((result_dir / "rep0000.json").read_text())["error"] is None


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
