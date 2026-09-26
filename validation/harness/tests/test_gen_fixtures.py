"""Tests for gen_fixtures.py.

The registry and dataset-generator tests need no R. TestMakeAllAndCheck
calls Rscript for real (through driver.run_earth), so it is marked
``external`` (gate A/B and CI exclude it; gate C and manual runs include
it)."""

import json
from pathlib import Path

import gen_fixtures
import numpy as np
import pytest


class TestRegistry:
    def test_s01_is_registered(self):
        assert "S01" in gen_fixtures.REGISTRY

    def test_s01_matches_its_own_id(self):
        assert gen_fixtures.REGISTRY["S01"]().id == "S01"

    def test_s01_is_deterministic(self):
        a, b = gen_fixtures.s01(), gen_fixtures.s01()
        assert np.array_equal(a.X, b.X)
        assert np.array_equal(a.y, b.y)
        assert np.array_equal(a.X_test, b.X_test)

    def test_s01_shape(self):
        ds = gen_fixtures.s01()
        assert ds.X.shape == (200, 1)
        assert ds.y.shape == (200,)
        assert ds.X_test.shape[1] == 1
        assert ds.X_test.shape[0] > 0

    def test_at_least_the_two_documented_modes_are_registered(self):
        assert {"defaults_d1", "matched_d1"} <= set(gen_fixtures.MODES)

    def test_matched_d1_is_hinge_only_every_case_a_candidate(self):
        # VALIDATION_PLAN.md, "Comparison modes": the legacy-code matched
        # settings this mode is seeded from.
        mode = gen_fixtures.MODES["matched_d1"]
        assert mode["Auto.linpreds"] is False
        assert mode["fast.k"] == 0
        assert mode["minspan"] == 1
        assert mode["endspan"] == 1


@pytest.mark.external
class TestMakeAllAndCheck:
    def test_make_all_writes_one_file_per_dataset_and_mode(self, tmp_path):
        paths = gen_fixtures.make_all(fixtures_dir=tmp_path)
        expected = {
            f"{dataset_id}_{mode}.json"
            for dataset_id in gen_fixtures.REGISTRY
            for mode in gen_fixtures.MODES
        }
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.is_file()

    def test_fixture_has_the_documented_top_level_shape(self, tmp_path):
        [path] = [
            p
            for p in gen_fixtures.make_all(fixtures_dir=tmp_path)
            if p.name == "S01_matched_d1.json"
        ]
        payload = json.loads(path.read_text())
        assert payload["dataset"] == "S01"
        assert payload["mode"] == "matched_d1"
        assert payload["earth_args"] == gen_fixtures.MODES["matched_d1"]
        assert len(payload["inputs"]["X"]) == 200
        assert len(payload["inputs"]["y"]) == 200
        assert payload["result"]["selected_terms"]
        assert payload["versions"]["earth_version"]

    def test_check_reports_nothing_when_fixtures_match(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert result.problems == []
        assert result.notes == []

    def test_check_reports_a_missing_fixture(self, tmp_path):
        paths = gen_fixtures.make_all(fixtures_dir=tmp_path)
        paths[0].unlink()
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert any("missing" in p for p in result.problems)

    def test_check_reports_a_modified_fixture(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        target = tmp_path / "S01_matched_d1.json"
        payload = json.loads(target.read_text())
        payload["result"]["gcv"] = payload["result"]["gcv"] + 1.0  # corrupt it
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert any(
            "S01_matched_d1.json" in p and "differs" in p for p in result.problems
        )

    def test_check_reports_a_versions_difference_as_a_note_not_a_problem(
        self, tmp_path
    ):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        target = tmp_path / "S01_matched_d1.json"
        payload = json.loads(target.read_text())
        payload["versions"]["python"] = "9.9.9"  # a plausible future version
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert result.problems == []
        assert any(
            "S01_matched_d1.json" in n and "versions differ" in n for n in result.notes
        )

    def test_sanitize_versions_drops_the_commit_and_shortens_paths(self):
        raw = {
            "pymars_commit": "abc123",
            "blas": [
                {
                    "internal_api": "openblas",
                    "filepath": "/home/alice/.venv/lib/libopenblas.so",
                }
            ],
            "r_blas": {
                "la_library": "/Users/alice/R/lib/libRlapack.dylib",
                "blas": "/Users/alice/R/lib/libRblas.0.dylib",
            },
        }
        out = gen_fixtures._sanitize_versions(raw)
        assert "pymars_commit" not in out
        assert out["blas"][0]["filepath"] == "libopenblas.so"
        assert out["r_blas"]["la_library"] == "libRlapack.dylib"
        assert out["r_blas"]["blas"] == "libRblas.0.dylib"
        assert "/home/alice" not in json.dumps(out)
        assert "/Users/alice" not in json.dumps(out)

    def test_main_check_exits_nonzero_on_a_difference(
        self, tmp_path, monkeypatch, capsys
    ):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        (tmp_path / "S01_matched_d1.json").write_text("{}")
        monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", tmp_path)
        assert gen_fixtures.main(["--check"]) == 1
        assert "did not reproduce" in capsys.readouterr().err

    def test_main_check_exits_zero_when_everything_matches(self, tmp_path, monkeypatch):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", tmp_path)
        assert gen_fixtures.main(["--check"]) == 0

    def test_the_committed_s01_fixtures_reproduce_exactly(self):
        # The actual check the brief asks for: the committed fixtures under
        # validation/fixtures/ regenerate with no diff.
        repo_fixtures = Path(gen_fixtures.FIXTURES_DIR)
        assert list(repo_fixtures.glob("S01_*.json")), (
            f"no committed S01 fixtures in {repo_fixtures}"
        )
        assert gen_fixtures.check(fixtures_dir=repo_fixtures).problems == []
