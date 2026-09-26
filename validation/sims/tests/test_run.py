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
    assert not run.is_done(missing)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert not run.is_done(corrupt)

    valid = tmp_path / "valid.json"
    valid.write_text("{}")
    assert run.is_done(valid)


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
