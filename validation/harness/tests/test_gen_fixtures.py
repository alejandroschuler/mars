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

    def test_s02_through_s20_are_registered(self):
        # Every dataset id gen_fixtures.py registers for S02 to S20 (T05,
        # issue #7); a dataset with several sizes or variants is several
        # ids (validation/README.md, "Dataset fixtures").
        expected = {
            "S02_n020",
            "S02_n050",
            "S03",
            "S04_p05_n0200",
            "S04_p05_n1000",
            "S04_p10_n0200",
            "S04_p10_n1000",
            "S05",
            "S06",
            "S07",
            "S08",
            "S09",
            "S10",
            "S11_n03",
            "S11_n05",
            "S11_n08",
            "S11_n12",
            "S12_base",
            "S12_x_1em8",
            "S12_x_1e8",
            "S12_x_plus_1e6",
            "S12_y_1em9",
            "S12_y_1e9",
            "S13_int_zeros",
            "S13_int_zeros_repeated",
            "S13_int_random",
            "S13_int_random_repeated",
            "S13_unit",
            "S13_unit_repeated",
            "S13_nonint",
            "S13_constant_y_weighted",
            "S13_constant_y_weighted_repeated",
            "S14",
            "S16_weighted",
            "S16_weighted_repeated",
            "S17",
            "S18",
            "S19_dummies",
            "S20",
        }
        assert expected <= set(gen_fixtures.REGISTRY)

    def test_every_registered_dataset_has_a_unique_deterministic_id(self):
        seen = set()
        for dataset_id, fn in gen_fixtures.REGISTRY.items():
            ds_a, ds_b = fn(), fn()
            assert ds_a.id == dataset_id, f"{dataset_id}: registered under another id"
            assert ds_a.id not in seen, f"{dataset_id}: id collides with another one"
            seen.add(ds_a.id)
            assert np.array_equal(ds_a.X, ds_b.X), f"{dataset_id}: X not deterministic"
            assert np.array_equal(ds_a.y, ds_b.y), f"{dataset_id}: y not deterministic"

    def test_every_dataset_mode_is_a_registered_mode(self):
        for dataset_id, modes in gen_fixtures.DATASET_MODES.items():
            assert dataset_id in gen_fixtures.REGISTRY, f"{dataset_id} not registered"
            for mode in modes:
                assert mode in gen_fixtures.MODES, f"{mode!r} unknown ({dataset_id})"

    def test_extras_are_registered(self):
        assert {"s15_draws", "s18_multinom", "s19_factor"} <= set(
            gen_fixtures.EXTRA_REGISTRY
        )


class TestScaledMatrix:
    def test_constant_column_is_left_as_is(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        X_scaled, scale = gen_fixtures.scaled_matrix(X)
        assert np.array_equal(X_scaled[:, 1], X[:, 1])
        assert scale[1] == 1.0

    def test_non_constant_column_gets_unit_population_variance(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_scaled, scale = gen_fixtures.scaled_matrix(X)
        assert np.var(X_scaled[:, 0]) == pytest.approx(1.0)
        assert scale[0] == pytest.approx(np.std(X[:, 0]))

    def test_is_not_centered(self):
        X = np.array([[10.0], [11.0], [12.0]])
        X_scaled, _ = gen_fixtures.scaled_matrix(X)
        assert np.mean(X_scaled[:, 0]) != pytest.approx(0.0)

    def test_weighted_variance_uses_divisor_n_not_n_minus_1(self):
        X = np.array([[1.0], [2.0], [3.0]])
        w = np.array([2.0, 1.0, 1.0])
        _, scale = gen_fixtures.scaled_matrix(X, w)
        mean = np.average(X[:, 0], weights=w)
        expected = np.sqrt(np.average((X[:, 0] - mean) ** 2, weights=w))
        assert scale[0] == pytest.approx(expected)

    def test_integer_weights_match_the_repeated_row_expansion(self):
        # W-1: an integer weight must give the same fit as repeated rows;
        # scaled_matrix is the harness step both take, so the two must
        # agree here too.
        X = np.array([[1.0], [2.0], [5.0]])
        w = np.array([2, 1, 3])
        _, scale_w = gen_fixtures.scaled_matrix(X, w.astype(float))
        X_rep = np.repeat(X, w, axis=0)
        _, scale_rep = gen_fixtures.scaled_matrix(X_rep)
        assert scale_w == pytest.approx(scale_rep)


class TestRawModes:
    def test_s12_datasets_include_a_raw_mode(self):
        # Review round 1, #43 finding 3: S12 also needs the LA-7-scaled
        # modes (without them, no S12 fixture gives pymars and earth the
        # same matrix), so this no longer requires *only* raw modes, just
        # that raw_d1 (earth's own defaults, unscaled) is still one of
        # them -- the one exception to LA-7 rescaling, since scaling away
        # S12's whole point (earth's raw scale/shift dependence) would
        # defeat it.
        for dataset_id in gen_fixtures.REGISTRY:
            if dataset_id.startswith("S12_"):
                modes = gen_fixtures.DATASET_MODES[dataset_id]
                assert set(modes) & gen_fixtures.RAW_MODES, (
                    f"{dataset_id}: has no raw mode left, so nothing shows "
                    "earth's own raw scale/shift dependence (bb14.4)"
                )


@pytest.mark.external
class TestMakeAllAndCheck:
    def test_make_all_writes_one_file_per_dataset_and_its_own_modes(self, tmp_path):
        paths = gen_fixtures.make_all(fixtures_dir=tmp_path)
        expected = {
            f"{dataset_id}_{mode}.json"
            for dataset_id in gen_fixtures.REGISTRY
            for mode in gen_fixtures.DATASET_MODES.get(
                dataset_id, gen_fixtures.DEFAULT_DATASET_MODES
            )
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
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
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
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
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
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
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
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
        expected = {f"{name}.json" for name in gen_fixtures.COMPONENT_REGISTRY}
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.is_file()
            assert p.parent.name == "components"

    def test_every_component_fixture_has_a_component_key_and_versions(
        self, component_fixtures
    ):
        # Review round 1 (#42 finding, adversarial, non-blocking): a
        # component fixture's versions block used to lack r_blas although
        # the dataset fixtures always have it.
        for name, payload in component_fixtures.items():
            assert payload["component"] == name
            assert payload["versions"]["earth_version"]
            assert payload["versions"]["r_version"]
            assert payload["versions"]["r_blas"]

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
        # Review round 1, #42 adversarial finding 3 (several responses) and
        # round 2, #42/#43 blocking finding 2 (one response, a regression):
        # both cases must actually separate PRUNE-3's K = 1 rule from its
        # K >= 2 rule, not just happen to agree with either.
        assert payload["one_response"]["k1_vs_k2_diverge_at_sizes"]
        assert payload["several_responses"]["k1_vs_k2_diverge_at_sizes"]

    def test_lm_fit_coefficients_has_a_rank_deficient_case(self, component_fixtures):
        payload = component_fixtures["lm_fit_coefficients"]
        by_label = {c["label"]: c for c in payload["cases"]}
        assert by_label["full_rank"]["rank"] == 3
        assert by_label["duplicated_column"]["coefficients"].count(None) == 1
        # Review round 1, #42 finding 5: near-duplicate columns on each
        # side of LA-4's 1e-7 threshold, and the centered/uncentered norm
        # case (bb09.9), and a weighted case (PRUNE-8's lm.wfit path).
        assert by_label["near_duplicate_above_1e-7"]["rank"] == 4
        assert by_label["near_duplicate_below_1e-7"]["coefficients"].count(None) == 1
        assert by_label["centered_vs_uncentered_norm"]["coefficients"].count(None) == 1
        assert by_label["weighted_full_rank"]["weights"] is not None
        assert by_label["weighted_full_rank"]["coefficients"].count(None) == 0

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
            # Review round 1, #42 finding 2: the model, not only pred.
            assert len(case["dirs"]) == len(case["cuts"])
            assert len(case["coefficients"]) == len(case["selected_terms"])

    def test_predict_new_points_linear_factor_diverges_below_the_minimum(
        self, component_fixtures
    ):
        # FWD-6: a linear term (Auto.linpreds = TRUE) extrapolates as a
        # straight line; a hinge at the training minimum (Auto.linpreds =
        # FALSE) extrapolates as a constant below it. The two must agree
        # inside the training range and diverge below the minimum.
        payload = component_fixtures["predict_new_points"]
        linear = np.array(payload["linear_auto"]["pred"]).ravel()
        hinge = np.array(payload["linear_hinge"]["pred"]).ravel()
        newx = np.array(payload["linear_auto"]["newx"]).ravel()
        x_min = np.array(payload["linear_auto"]["x"]).min()
        below = newx < x_min
        above = ~below
        assert below.any() and above.any()
        assert np.allclose(linear[above], hinge[above])
        assert not np.allclose(linear[below], hinge[below])

    def test_classifier_refit_has_binomial_and_multinomial_cases(
        self, component_fixtures
    ):
        payload = component_fixtures["classifier_refit"]
        fitted = np.array(payload["binomial"]["fitted_values"])
        assert np.all((fitted > 0) & (fitted < 1))
        probs = np.array(payload["multinomial"]["fitted"])
        assert probs.shape[1] == len(payload["multinomial"]["levels"]) == 3
        assert np.allclose(probs.sum(axis=1), 1.0)
        # Review round 1, #42/#43 finding 1: both references must have
        # actually converged (GLM-3/GLM-4).
        assert payload["binomial"]["converged"] is True
        assert payload["multinomial"]["convergence"] == 0

    def test_knot_candidates_grid_covers_the_documented_axes(self, component_fixtures):
        cases = component_fixtures["knot_candidates"]["cases"]
        assert {c["degree"] for c in cases} == {1, 2}
        assert {c["minspan"] for c in cases} == {None, 1, 3, 10}
        assert {c["endspan"] for c in cases} == {None, 1, 5}
        assert {c["n"] for c in cases} >= {20, 200, 2000}
        assert all("FindKnotBegin" in c["trace_text"] for c in cases)
        # Review round 1, #42 finding 4: endspan = 5 at n = 200 (SPAN-5's
        # cap does not hide Adjust.endspan there), for both its values.
        endspan5_at_200 = {
            c["adjust_endspan"] for c in cases if c["endspan"] == 5 and c["n"] == 200
        }
        assert endspan5_at_200 == {1.0, 2.0}
        # A degree-2, negative-x case exists (KNOT-1's negative-linear-
        # parent rule), whatever term structure earth actually picked.
        assert any(
            c["degree"] == 2 and min(min(row) for row in c["X"]) < 0 for c in cases
        )

    def test_check_reports_nothing_when_components_match(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert result.problems == []

    def test_check_reports_a_modified_component_fixture(self, tmp_path):
        gen_fixtures.make_all(fixtures_dir=tmp_path)
        gen_fixtures.make_all_components(fixtures_dir=tmp_path)
        gen_fixtures.make_all_extras(fixtures_dir=tmp_path)
        target = tmp_path / "components" / "gcv_grid.json"
        payload = json.loads(target.read_text())
        payload["grid"][0]["gcv"][0] = (payload["grid"][0]["gcv"][0] or 0.0) + 1.0
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result = gen_fixtures.check(fixtures_dir=tmp_path)
        assert any("gcv_grid.json" in p and "differs" in p for p in result.problems)
