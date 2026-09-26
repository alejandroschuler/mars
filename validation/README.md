# Validation

Everything here checks pymars 2.0 against the R package earth, and reports
on it. `VALIDATION_PLAN.md` (repository root) is the plan; earth is used
only as a black box, per its "Instruction files and clean room" section.

## Requirements

The harness needs R (earth, jsonlite and nnet installed) on `PATH`, in
addition to the `validation` uv dependency group:

```bash
uv sync --frozen --group validation
Rscript -e 'library(earth); library(jsonlite); library(nnet)'  # sanity check
```

`legacy_adapter.py` also needs the legacy venv, made once:

```bash
validation/legacy/make_venv.sh
```

## `validation/harness/`

- `fit_earth.R`: fits R earth on a block of datasets described by a JSON
  config, and writes each result as JSON at full double precision. Reads
  its own config format from its module docstring; not normally run by
  hand (see `driver.py` below).
- `driver.py`: the Python entry point. `EarthJob` describes one dataset and
  earth call; `run_earth(jobs, workdir=...)` writes the data as CSV, fits
  every job in `jobs` with one `Rscript fit_earth.R` process (so R's
  start-up cost is paid once per block), and returns `{job.id: result}` in
  the schema `driver.py`'s module docstring documents. `versions()` records
  the Python-side versions (numpy, scikit-learn, the BLAS library through
  threadpoolctl, the pymars commit) and merges in a result's own
  `r_version`/`earth_version`.
- `blackbox.R`/`blackbox.py`: one function per black-box call into an
  earth-adjacent internal for the component tests (`get_gcv`,
  `pruning_pass`, `lm_fit`, `predict_earth`, `glm_fit`, `multinom_fit`);
  each writes a small JSON request and reads a JSON result back.
- `trace_parse.py`: `parse_trace(path, sample_var_y=...)` turns an earth
  `trace = 7` to `9` log into a `TraceLog` of `ForwardStep`s, each with its
  `KnotSearch`es (one per parent/predictor pair considered) and their
  `CaseRecord`s (one per candidate knot).
- `names_map.py`: `to_earth_args(**pymars_params)` maps pymars 2.0
  constructor parameter names to earth's own argument names, resolving
  `None` ("earth's automatic value") for the parameters that need it.
- `legacy_adapter.py`: `fit_legacy(X, y, **earth_kwargs)` fits the legacy
  code (mars-earth 1.0.4, `.venv-legacy`) in a subprocess and converts its
  result to the common schema.
- `new_adapter.py`: `mars_fit_to_common(fit)` converts a `MarsFit`
  (`pymars._core`, once it exists) to the common schema.
- `compare.py`: `compare_fit(a, b, ...)` compares two common-schema results
  by the tolerance table; `compare_forward_steps(a_steps, b_steps)` compares
  two forward-pass candidate logs up to the first near-tie.
- `gen_fixtures.py`: the fixture generator (its own section below).

Example: fit earth on one dataset with a trace, using `driver.py` directly.
These modules are plain scripts, not a `validation.harness` package (there
is no `__init__.py`), so a caller outside `validation/harness/` puts that
folder on `sys.path` first, the same way `validation/harness/tests/conftest.py`
does for the test suite.

```python
import sys

import numpy as np

sys.path.insert(0, "validation/harness")
from driver import EarthJob, run_earth

rng = np.random.default_rng(0)
X = rng.uniform(size=(200, 1))
y = 2 * np.maximum(0, X[:, 0] - 0.3) + rng.normal(scale=0.05, size=200)

job = EarthJob(
    id="example", X=X, y=y,
    earth_args={"degree": 1, "nk": 21, "pmethod": "backward"},
    trace=9,
)
result = run_earth([job], workdir="/tmp/example")["example"]
print(result["selected_terms"], result["gcv"])
```

Run the harness's own tests (in `validation/harness/tests/`, on
`testpaths` alongside `tests/` and `validation/sims/tests/`):

```bash
uv run --frozen --group validation pytest validation/harness/tests -v
```

A test that calls `Rscript` for real (`fit_earth.R` or `blackbox.R`) or
needs `.venv-legacy` is marked `external`; it is not skipped when R or the
legacy venv is missing, it fails, so gate A, gate B and CI all exclude it
(`-m "not external"`). Everything else, including the pure-Python parts
(`trace_parse.py`, `names_map.py`, and `driver.py`'s CSV writing and
column-naming helpers), needs no R and runs everywhere. Gate C and a plain
`pytest` invocation run everything, `external` tests included.

## Fixture generator

`gen_fixtures.py` writes `validation/fixtures/<dataset>_<mode>.json` for
every registered dataset and earth argument set ("mode"), each with that
pair's inputs, earth arguments, `driver.run_earth` result and versions:

```bash
python validation/harness/gen_fixtures.py           # write every fixture
python validation/harness/gen_fixtures.py --check   # remake them in a temp
                                                     # folder and diff
```

`--check` exits nonzero if any fixture fails to reproduce exactly (a
difference in `versions.pymars_commit` alone, from running it on a
different commit than the one that made the committed fixture, is expected
and not a regression). Register a new dataset by adding a `@register`-
decorated function returning a `Dataset` to `gen_fixtures.py`; register a
new mode by adding an entry to its `MODES` dict; a dataset not listed in
`DATASET_MODES` is generated under `DEFAULT_DATASET_MODES`
(`defaults_d1`/`matched_d1`).

The committed fixtures were made on macOS arm64. `--check` compares every
stored double exactly, so running it on a different OS or architecture (for
example `earth-conformance.yml`'s `ubuntu-latest`, x86_64) can fail from a
last-bit difference in a transcendental function or a BLAS routine alone,
with no change to the harness or the datasets. Exact reproduction is
reliable only on the machine that made the fixtures; a failure elsewhere
needs a by-hand look at which bits differ before it counts as a regression.

The fixtures are Python-only JSON, not strict RFC 8259: a field in
`test_fixture_contents.py`'s `_MAY_BE_NONFINITE_KEYS` (`gcv`, `grsq`,
`rsq`, `gcv_per_subset`, and `gcv_grid.json`'s own per-cell `gcv` list)
may hold a bare `Infinity`/`-Infinity`/`NaN` token where GCV-2 or GCV-7
gives one (a case count at or below the effective number of parameters;
a degenerate or near-degenerate fit). Python's `json.loads` reads these
back exactly; a strict reader (R's `jsonlite::fromJSON` included) does
not, and needs a sentinel pass, or a preprocessing step, first (review
round 1, #42 adversarial finding 7; round 2, #42 spec finding 4, added
`gcv_grid.json` to the fixtures this affects, since round 1 had noted
it only here on #43).

### Dataset fixtures

`VALIDATION_PLAN.md`, "Test datasets", defines S01 to S20 and what each is
for. A dataset with more than one size or variant (for example S04's 5 and
10 covariates, 200 and 1,000 cases) is registered as several dataset ids,
one file set per variant; `gen_fixtures.DATASET_MODES` names which modes
each dataset id is generated under (`DEFAULT_DATASET_MODES`,
`defaults_d1`/`matched_d1`, for a dataset not listed there).

In the matched and earth-compatible (`defaults_*`) modes, `inputs.X` is the
LA-7-scaled matrix (each non-constant covariate divided by its weighted
standard deviation, divisor N, not centered; `gen_fixtures.scaled_matrix`),
and the top-level `scale` field gives the per-column divisors used (`null`
for a raw mode, `gen_fixtures.RAW_MODES`, S12's own fits). Where earth
errors on an input, `result` is `{"error": ..., "r_version": ...,
"earth_version": ...}` instead of a full result (S13's weighted constant
response).

| Dataset id(s) | Modes | Purpose |
|---|---|---|
| `S01` | `defaults_d1`, `matched_d1`, the 9 `span_*` (minspan 1/5/auto x endspan 1/10/auto) | Knot recovery |
| `S02_n020`, `S02_n050` | same as S01 | Span formulas and stopping rules at small n |
| `S03` | `defaults_d1`, `matched_d1`, `matched_d1_linear` | Pairs against single hinges; linear terms |
| `S04_p05_n0200`, `S04_p05_n1000`, `S04_p10_n0200`, `S04_p10_n1000` | `defaults_d1`, `defaults_d2`, `matched_d1`, `matched_d2` | Interactions and irrelevant covariates, degree 1 and 2 |
| `S05` | `defaults_d2`, `matched_d2`, `matched_d2_adjust1` | Interaction search; `Adjust.endspan` |
| `S06` | `defaults_d3`, `matched_d3` | Degree 3 |
| `S07` | `defaults_d1`, `matched_d1`, `matched_d1_linear` | `Auto.linpreds` against pymars' linear candidates |
| `S08` | `defaults_d1`, `matched_d1`, `matched_d1_minspan5` | Repeated x values: distinct values against cases |
| `S09` | `defaults_d1`, `matched_d1` | Categorical coding through OneHotEncoder |
| `S10` | `defaults_d1`, `matched_d1` | Tie-breaks across predictors; collinearity |
| `S11_n03`, `S11_n05`, `S11_n08`, `S11_n12` | `defaults_d1`, `matched_d1` | Degenerate sizes |
| `S12_base`, `S12_x_1em8`, `S12_x_1e8`, `S12_x_plus_1e6`, `S12_y_1em9`, `S12_y_1e9` | `raw_d1` (no LA-7 rescaling, earth's own scale/shift dependence), `defaults_d1`, `matched_d1` | Invariance to scale and shift; earth is not scale invariant (bb14.4) |
| `S13_int_zeros`(`_repeated`), `S13_int_random`(`_repeated`), `S13_unit`(`_repeated`), `S13_equal2`(`_repeated`), `S13_nonint`, `S13_constant_y_weighted`(`_repeated`) | `matched_d1`, `defaults_d1` (every id, `S13_nonint` included) | Weights: repetition, removal, unit weights, weights all equal but not 1 (GCV-8/W-8), the fixed-basis path for non-integer weights, and a weighted constant response earth may error on |
| `S14` | `defaults_d2`, `matched_d2` (`glm_family="binomial"`) | GLM refit |
| `s15_draws` (an "extra", not a (dataset, mode) fixture) | `matched`/`defaults`, degree 1/2, alternated across reps | 200 draws from `validation/sims/dgps.py` (D1-D6, D8; n = 200, earth's own default term limit); each draw's `trace = 8` log is parsed into a compact per-step summary (`compare.steps_from_trace`) and dropped rather than stored whole. About 1 in 7 draws hits a gap in `steps_from_trace` (FAST-4's slot/row skew after a single-term step) and keeps `dirs`/`cuts`/`rss_per_subset`/`gcv_per_subset` with `steps = null` and `steps_error` set, rather than losing the draw or patching the harness here. Each step's `best_rss`/`second_best_rss`/`rss_before` are `trace = 8` text (at most 5 significant digits, on y standardized to variance 1, not `rss_per_subset`'s own scale in the same record): not precise enough for a 1e-7 near-tie call, which T07 should take from pymars's own candidate log instead (review round 2, #43 spec finding 3; also stored as `steps_rss_precision_note` in the fixture itself) |
| `S16_weighted`, `S16_weighted_repeated` | `matched_d1`, `defaults_d1`, `defaults_d2`, `matched_d2` | Frequency weights (S04's 5-covariate, 200-case data) against repeated rows |
| `S17` | `defaults_d1`, `matched_d1` | Several responses with a shared basis |
| `S18` | `defaults_d1`, `matched_d1`; plus the extra `s18_multinom` (`nnet::multinom` on earth's selected basis) | Multiclass terms, and probabilities against `nnet::multinom` |
| `S19_dummies` | `defaults_d1`, `matched_d1`; plus the extra `s19_factor` (earth on a genuine R factor column) | The OneHotEncoder recipe |
| `S20` | `defaults_d1`, `matched_d1` (`glm_family="binomial"`) | Separation warnings and fitted probabilities near 0 or 1 |

### Extra fixtures

Three bespoke, one-off fixtures do not fit the (dataset, mode) registry
(`gen_fixtures.EXTRA_REGISTRY`/`make_all_extras()`, `validation/fixtures/
<name>.json`, alongside the dataset fixtures): `s15_draws.json` (S15),
`s18_multinom.json` (S18's `nnet::multinom` comparison) and
`s19_factor.json` (S19's "factor" side). Register a new one by adding a
`@register_extra`-decorated function to `gen_fixtures.py`.

### Component fixtures

`validation/fixtures/components/<name>.json` each target one earth-adjacent
internal (`VALIDATION_PLAN.md`, "Component tests"), through `blackbox.py`
rather than a whole `earth()` fit compared end to end. Register a new one
by adding a `@register_component`-decorated function to `gen_fixtures.py`;
`make_all_components()`/`gen_fixtures.py --check` cover them the same way
as the dataset fixtures above.

| Fixture | Tests | For |
|---|---|---|
| `gcv_grid.json` | `earth:::get.gcv` (`blackbox.get_gcv`) over a grid of penalties (0 to 6, -1) and case counts (10 to 100,000), 1 to 41 terms | The GCV function (GCV-1, GCV-2) |
| `pruning_fixed_basis.json` | `earth:::pruning.pass` (`blackbox.pruning_pass`) on a real forward basis (`blackbox.fit_bx_dirs`, degree 2), for one response and for several | Pruning of a fixed basis (PRUNE-1 to PRUNE-9) |
| `lm_fit_coefficients.json` | R's `lm.fit`/`lm.wfit` (`blackbox.lm_fit`), full rank, a duplicated (rank-deficient) column, near-duplicate columns on each side of LA-4's 1e-7 threshold, a centered-vs-uncentered-norm case (bb09.9), and a weighted case | Coefficients of fixed terms (LA-4, PRUNE-8) |
| `predict_new_points.json` | `predict.earth` (`blackbox.predict_earth`) at points inside and outside the training range, degree 1 and 2, plus a linear-factor case (`Auto.linpreds = TRUE`) and a hinge-at-the-minimum case, each with new points below the training minimum (FWD-6); every case stores earth's `dirs`/`cuts`/`selected_terms`/coefficients alongside `pred` | Prediction at new points (TERM-3) |
| `classifier_refit.json` | R's `glm.fit` (binomial) and `nnet::multinom` (`blackbox.glm_fit`/`blackbox.multinom_fit`) on fixed columns | The GLM refit's coefficients and fitted probabilities ("Binary outcomes") |
| `knot_candidates.json` | earth's `trace = 9` case-by-case knot scan (`driver.run_earth`, trace text embedded per case), over the minspan/endspan/degree/`Adjust.endspan`/case-count grid | Candidate knot sets (KNOT-1 to KNOT-7); the comparison and the interpretation are T08's, not this fixture's (clean room) |

Regenerate all of them, and check they reproduce, with the same commands as
the dataset fixtures above (`gen_fixtures.py` writes and checks both).
