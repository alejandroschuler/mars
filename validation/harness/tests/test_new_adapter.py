"""Tests for new_adapter.py, using a small fake MarsFit (pymars._core does
not exist yet); no R needed."""

from types import SimpleNamespace

from new_adapter import mars_fit_to_common


def _fake_object_fit():
    """An object-shaped fake, matching the module docstring's assumed
    MarsFit shape: fit.forward.steps, fit.pruning.removed, and so on."""
    forward = SimpleNamespace(
        steps=[
            SimpleNamespace(parent=0, pred=0, direction=1, knot=0.3),
            SimpleNamespace(parent=0, pred=0, direction=-1, knot=0.3),
        ],
        rss=[18.0, 10.0, 4.0],
        termcond=6,
        candidates=[(10.0, 10.5), (4.0, 4.2)],
    )
    pruning = SimpleNamespace(
        removed=[2],
        rss=[4.0, 6.0, 18.0],
        gcv=[0.05, 0.07, 0.2],
        selected=[0, 1],
    )
    return SimpleNamespace(
        dirs=[[0, 0], [1, 0], [-1, 0]],
        cuts=[[0.0, 0.0], [0.3, 0.0], [0.3, 0.0]],
        coef=[[1.0], [2.0], [-1.5]],
        rss=4.0,
        gcv=0.05,
        rsq=0.92,
        grsq=0.9,
        n_eff=200.0,
        forward=forward,
        pruning=pruning,
    )


def _fake_dict_fit():
    """The same fit, as nested dicts ("the reference returns the same
    fields as a dict")."""
    return {
        "dirs": [[0, 0], [1, 0], [-1, 0]],
        "cuts": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.0]],
        "coef": [[1.0], [2.0], [-1.5]],
        "rss": 4.0,
        "gcv": 0.05,
        "rsq": 0.92,
        "grsq": 0.9,
        "n_eff": 200.0,
        "forward": {
            "steps": [
                {"parent": 0, "pred": 0, "direction": 1, "knot": 0.3},
                {"parent": 0, "pred": 0, "direction": -1, "knot": 0.3},
            ],
            "rss": [18.0, 10.0, 4.0],
            "termcond": 6,
            "candidates": [(10.0, 10.5), (4.0, 4.2)],
        },
        "pruning": {
            "removed": [2],
            "rss": [4.0, 6.0, 18.0],
            "gcv": [0.05, 0.07, 0.2],
            "selected": [0, 1],
        },
    }


class TestObjectShapedFit:
    def test_top_level_fields(self):
        result = mars_fit_to_common(_fake_object_fit())
        assert result["dirs"] == [[0, 0], [1, 0], [-1, 0]]
        assert result["cuts"] == [[0.0, 0.0], [0.3, 0.0], [0.3, 0.0]]
        assert result["coef"] == [[1.0], [2.0], [-1.5]]
        assert result["rss"] == 4.0
        assert result["gcv"] == 0.05
        assert result["rsq"] == 0.92
        assert result["grsq"] == 0.9
        assert result["n_eff"] == 200.0

    def test_pruning_fields(self):
        result = mars_fit_to_common(_fake_object_fit())
        assert result["selected_terms"] == [0, 1]
        assert result["prune_terms"] is None
        assert result["rss_per_subset"] == [4.0, 6.0, 18.0]
        assert result["gcv_per_subset"] == [0.05, 0.07, 0.2]
        assert result["pruning_removed"] == [2]

    def test_forward_fields(self):
        result = mars_fit_to_common(_fake_object_fit())
        assert result["termcond"] == 6
        assert result["fwd_rss"] == [18.0, 10.0, 4.0]
        # direction coerces a bare code to a one-element frozenset, so it
        # compares equal to compare.py's own {1, -1}/{1}/{-1}/{2} shape; the
        # best/second-best candidate RSS are merged in from "candidates" by
        # position, not left as a separate list the caller must re-zip.
        assert result["forward_steps"] == [
            {
                "parent": 0,
                "pred": 0,
                "direction": frozenset({1}),
                "knot": 0.3,
                "best_rss": 10.0,
                "second_best_rss": 10.5,
                "rss_before": None,
                "flags": None,
            },
            {
                "parent": 0,
                "pred": 0,
                "direction": frozenset({-1}),
                "knot": 0.3,
                "best_rss": 4.0,
                "second_best_rss": 4.2,
                "rss_before": None,
                "flags": None,
            },
        ]
        assert result["forward_candidates"] == [(10.0, 10.5), (4.0, 4.2)]


class TestDictShapedFit:
    def test_matches_the_object_shaped_result(self):
        # The reference implementation returns its fields as a dict; the
        # adapter must give the identical common-schema result either way.
        assert mars_fit_to_common(_fake_dict_fit()) == mars_fit_to_common(
            _fake_object_fit()
        )


class TestForwardStepDirectionAndCandidates:
    def test_a_direction_already_given_as_a_set_passes_through(self):
        fit = SimpleNamespace(
            forward=SimpleNamespace(
                steps=[SimpleNamespace(parent=1, pred=2, direction={1, -1}, knot=0.5)],
            ),
        )
        result = mars_fit_to_common(fit)
        assert result["forward_steps"][0]["direction"] == frozenset({1, -1})

    def test_missing_candidates_gives_none_best_and_second_best(self):
        fit = SimpleNamespace(
            forward=SimpleNamespace(
                steps=[SimpleNamespace(parent=1, pred=2, direction=2, knot=0.5)],
            ),
        )
        result = mars_fit_to_common(fit)
        step = result["forward_steps"][0]
        assert step["best_rss"] is None and step["second_best_rss"] is None


class TestMissingSubRecords:
    def test_no_forward_or_pruning_gives_none_not_an_error(self):
        fit = SimpleNamespace(
            dirs=[[0]],
            cuts=[[0.0]],
            coef=[[3.0]],
            rss=0.0,
            gcv=0.0,
            rsq=1.0,
            grsq=1.0,
            n_eff=5.0,
        )
        result = mars_fit_to_common(fit)
        assert result["dirs"] == [[0]]
        for key in (
            "selected_terms",
            "termcond",
            "fwd_rss",
            "forward_steps",
            "forward_candidates",
            "rss_per_subset",
            "gcv_per_subset",
            "pruning_removed",
        ):
            assert result[key] is None

    def test_missing_dirs_and_cuts_is_none(self):
        result = mars_fit_to_common(SimpleNamespace())
        assert result["dirs"] is None
        assert result["cuts"] is None
        assert result["coef"] is None
