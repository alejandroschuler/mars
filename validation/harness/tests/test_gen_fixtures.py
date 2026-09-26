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
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
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
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
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
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", tmp_path)
        assert gen_fixtures.main(["--check"]) == 0

    def test_the_committed_fixtures_reproduce_exactly(self):
        # The actual check the brief asks for: every committed fixture under
        # validation/fixtures/ (datasets and components) regenerates with no
        # diff; check() covers both (its own docstring).
        repo_fixtures = Path(gen_fixtures.FIXTURES_DIR)
        assert list(repo_fixtures.glob("S01_*.json")), (
            f"no committed S01 fixtures in {repo_fixtures}"
        )
        assert list((repo_fixtures / "components").glob("*.json")), (
            f"no committed component fixtures in {repo_fixtures / 'components'}"
        )
        assert gen_fixtures.check(fixtures_dir=repo_fixtures).problems == []


class TestComponentRegistry:
    def test_expected_components_are_registered(self):
        assert {
            "gcv_grid",
            "pruning_fixed_basis",
            "lm_fit_coefficients",
            "predict_new_points",
            "classifier_refit",
            "knot_candidates",
        } <= set(gen_fixtures.COMPONENT_REGISTRY)

    def test_a_registered_component_is_a_callable_not_its_payload(self):
        # Unlike register (datasets), register_component must not call fn()
        # at decoration time: a component payload runs a real earth call, so
        # merely importing gen_fixtures must never touch Rscript.
        assert callable(gen_fixtures.COMPONENT_REGISTRY["gcv_grid"])


@pytest.mark.external
class TestMakeAllComponentsAndCheck:
    @pytest.fixture(scope="class")
    def component_fixtures(self, tmp_path_factory):
        # Module truly means "class" here: one real regeneration of every
        # component, reused read-only by every shape test below, instead of
        # each test paying its own Rscript cost.
        tmp = tmp_path_factory.mktemp("components")
        gen_fixtures.make_all_components(fixtures_dir=tmp)
        return {
            p.stem: json.loads(p.read_text())
            for p in (tmp / "components").glob("*.json")
        }

    def test_make_all_components_writes_one_file_per_component(self, tmp_path):
        paths = gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        expected = {f"{name}.json" for name in gen_fixtures.COMPONENT_REGISTRY}
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.is_file()
            assert p.parent.name == "components"

    def test_every_component_fixture_has_a_component_key_and_versions(
        self, component_fixtures
    ):
        for name, payload in component_fixtures.items():
            assert payload["component"] == name
            assert payload["versions"]["earth_version"]
            assert payload["versions"]["r_version"]

    def test_gcv_grid_covers_the_documented_penalties_and_case_counts(
        self, component_fixtures
    ):
        payload = component_fixtures["gcv_grid"]
        penalties = {cell["penalty"] for cell in payload["grid"]}
        assert {-1, 0, 1, 2, 3, 4, 5, 6} <= penalties
        assert len(payload["nterms"]) == 41
        assert min(c["ncases"] for c in payload["grid"]) <= 10
        assert max(c["ncases"] for c in payload["grid"]) >= 100_000
        assert any(
            v is not None and np.isinf(v)
            for cell in payload["grid"]
            for v in cell["gcv"]
        )

    def test_pruning_fixed_basis_has_one_and_several_response_cases(
        self, component_fixtures
    ):
        payload = component_fixtures["pruning_fixed_basis"]
        assert len(payload["one_response"]["selected_terms"]) >= 1
        assert np.array(payload["several_responses"]["y"]).ndim == 2
        # rss.per.subset is indexed by subset size, worst (1 term) to best.
        assert (
            payload["one_response"]["rss_per_subset"][0]
            >= payload["one_response"]["rss_per_subset"][-1]
        )

    def test_lm_fit_coefficients_has_a_rank_deficient_case(self, component_fixtures):
        payload = component_fixtures["lm_fit_coefficients"]
        by_label = {c["label"]: c for c in payload["cases"]}
        assert by_label["full_rank"]["rank"] == 3
        assert by_label["duplicated_column"]["coefficients"].count(None) == 1

    def test_predict_new_points_covers_outside_the_training_range(
        self, component_fixtures
    ):
        payload = component_fixtures["predict_new_points"]
        for case in (payload["degree1"], payload["degree2"]):
            x = np.array(case["x"])
            newx = np.array(case["newx"])
            assert newx.min() < x.min()
            assert newx.max() > x.max()
            assert np.all(np.isfinite(case["pred"]))

    def test_classifier_refit_has_binomial_and_multinomial_cases(
        self, component_fixtures
    ):
        payload = component_fixtures["classifier_refit"]
        fitted = np.array(payload["binomial"]["fitted_values"])
        assert np.all((fitted > 0) & (fitted < 1))
        probs = np.array(payload["multinomial"]["fitted"])
        assert probs.shape[1] == len(payload["multinomial"]["levels"]) == 3
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_knot_candidates_grid_covers_the_documented_axes(self, component_fixtures):
        cases = component_fixtures["knot_candidates"]["cases"]
        assert {c["degree"] for c in cases} == {1, 2}
        assert {c["minspan"] for c in cases} == {None, 1, 3, 10}
        assert {c["endspan"] for c in cases} == {None, 1, 5}
        assert {c["n"] for c in cases} >= {20, 200, 2000}
        assert all("FindKnotBegin" in c["trace_text"] for c in cases)

    def test_check_reports_nothing_when_components_match(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert result.problems == []

    def test_check_reports_a_modified_component_fixture(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        target = tmp_path / "components" / "gcv_grid.json"
        payload = json.loads(target.read_text())
        payload["grid"][0]["gcv"][0] = (payload["grid"][0]["gcv"][0] or 0.0) + 1.0
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert any("gcv_grid.json" in p and "differs" in p for p in result.problems)
