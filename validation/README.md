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

Run the harness's own tests (outside `tests/`, because CI has no R):

```bash
uv run --frozen --group validation pytest validation/harness/tests
```

Most of these tests call `Rscript` for real (`fit_earth.R` or
`blackbox.R`); they are not skipped when R is missing, so run them only
where R, earth and nnet are installed. The pure-Python parts
(`trace_parse.py`, `names_map.py`, and `driver.py`'s CSV writing and
column-naming helpers) have tests that need no R.
