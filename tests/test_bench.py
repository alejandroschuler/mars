"""The logic of the benchmark harness (validation/bench/): the grid, the result
records and the resume rule, the skip rule after a timeout, and the log-log
slope. The timed fits themselves are not run here."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "validation" / "bench"
sys.path.insert(0, str(BENCH))

import core  # noqa: E402
import run  # noqa: E402


def test_grid_is_one_factor_at_a_time_around_the_baseline():
    """Benchmark design: every series holds the baseline cell, varies one
    factor, and the legacy code stops at 2,000 cases and takes no weights."""
    series = core.build_series()
    for (system, factor), cells in series.items():
        assert core.Cell(system) in cells
        for c in cells:
            same = [k for k in core.BASE if getattr(c, k) == core.BASE[k]]
            assert len(same) >= len(core.BASE) - 1 or c == core.Cell(system)
            assert all(getattr(c, k) == core.BASE[k] for k in core.BASE if k != factor)
    legacy = [c for (s, _), cs in series.items() if s == "legacy" for c in cs]
    assert max(c.n for c in legacy) == 2000
    assert not any(c.weights for c in legacy)
    n_cells = [c.n for c in series[("pymars_fit", "n")]]
    assert n_cells == [250, 500, 1000, 2000, 5000, 10000, 100000]
    # the baseline is shared, so it appears once among the distinct cells
    assert core.all_cells(series).count(core.Cell("earth_default")) == 1


def test_data_are_fixed_by_the_seed_and_shared_by_cells_with_the_same_n_and_p():
    X1, y1, w1 = core.make_data(300, 7)
    X2, y2, w2 = core.make_data(300, 7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and np.array_equal(w1, w2)
    assert set(np.unique(w1)) <= {1.0, 2.0, 3.0}
    assert not np.array_equal(core.make_data(300, 8)[1], y1)


def test_cache_key_depends_on_cell_code_and_repetition_rule():
    c = core.Cell("pymars_fit")
    base = core.cell_key(c, "a")
    assert base == core.cell_key(core.Cell("pymars_fit"), "a")
    assert base != core.cell_key(c, "b")
    assert base != core.cell_key(core.Cell("pymars_fit", n=2000), "a")
    assert base != core.cell_key(c, "a", reps=1)
    assert base != core.cell_key(c, "a", timeout=60.0)


def test_loglog_slope_recovers_a_power_law_and_ignores_nonpositive_points():
    xs = [250.0, 500.0, 1000.0, 2000.0]
    ys = [3 * x**1.5 for x in xs]
    slope, k = core.loglog_slope([*xs, 4000.0], [*ys, 0.0])
    assert k == 4
    assert slope == pytest.approx(1.5)
    assert core.loglog_slope([1.0], [2.0]) is None
    assert core.loglog_slope([2.0, 2.0], [1.0, 3.0]) is None


def test_atomic_write_keeps_old_content_when_the_write_fails(tmp_path):
    path = tmp_path / "r.json"
    core.write_json_atomic(path, {"a": 1})
    with pytest.raises(TypeError):
        core.write_json_atomic(path, {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert core.read_result(tmp_path, "r") == {"a": 1}
    (tmp_path / "torn.json").write_text('{"a"')
    assert core.read_result(tmp_path, "torn") is None


def _runner(tmp_path, monkeypatch, behaviour, resume):
    args = argparse.Namespace(
        out=str(tmp_path), resume=resume, reps=2, timeout=5.0, profile=False, jobs=1
    )
    r = run.Runner(args)
    calls = []

    def fake(cmd, timeout):
        calls.append(cmd)
        if behaviour(cmd) == "timeout":
            return run.Measured(None, "", "", timeout, 0)
        out = json.dumps(
            {"seconds": 0.5, "py_peak_bytes": 8, "terms": 5, "forward_terms": 7}
        )
        return run.Measured(0, out, "", 0.5, 4096)

    monkeypatch.setattr(run, "run_measured", fake)
    monkeypatch.setattr(
        r, "command", lambda cell, mode, prof=None: ["w", str(cell.n), mode]
    )
    return r, calls


def test_a_timeout_skips_the_larger_cells_of_the_series(tmp_path, monkeypatch):
    r, _ = _runner(
        tmp_path,
        monkeypatch,
        lambda cmd: "timeout" if cmd[1] == "1000" else "ok",
        False,
    )
    cells = core.build_series(("legacy",))[("legacy", "n")]
    r.planned = len(cells)
    r.run_series(cells)
    code = r.codes["legacy"]
    status = {
        c.n: core.read_result(r.results, core.cell_key(c, code, 2, 5.0))["status"]
        for c in cells
    }
    assert status == {250: "ok", 500: "ok", 1000: "timeout", 2000: "skipped"}
    rec = core.read_result(r.results, core.cell_key(cells[0], code, 2, 5.0))
    assert rec["time_median"] == 0.5 and rec["terms"] == 5 and len(rec["times"]) == 2


def test_resume_skips_finished_cells_and_reruns_errors(tmp_path, monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, lambda cmd: "ok", resume=True)
    r.planned = 1
    cell = core.Cell("earth_default", n=250)
    key = core.cell_key(cell, r.codes["earth_default"], 2, 5.0)
    core.write_json_atomic(
        r.results / f"{key}.json", core.make_record(cell, key, "ok", time_median=9.0)
    )
    assert r.run_cell(cell)["time_median"] == 9.0
    assert calls == []
    core.write_json_atomic(
        r.results / f"{key}.json", core.make_record(cell, key, "error", error="x")
    )
    assert r.run_cell(cell)["status"] == "ok"
    assert (
        len(calls) == 2
    )  # two repetitions of the time mode only (earth has no memory run)
    assert json.loads((tmp_path / "manifest.json").read_text())["counts"] == {"ok": 1}
