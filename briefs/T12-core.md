# Brief: T12 the core (pymars/_core.py)

Read `briefs/COMMON.md` (same folder) first, including the scratch-file and mutation-check rules. Issue: T12, #14. Reviewers: 2 (`spec`, `adversarial`). Model: the strongest.

## Read first

- `docs/algorithm.md` on `origin/main` (spec v1): CORE-1 to CORE-7 above all; then W (weights: W-3 drops zero-weight rows, W-4 N = Σw with `math.fsum` and τ_N), LIMIT (term limit), GCV-4 and GCV-7, PRUNE-5 to PRUNE-8 (nprune, pmethod, the final fit and its statistics), FWD-11 (kept and dropped terms), RESP (several responses), EDGE (degenerate inputs) and ERR (the errors that belong to the core).
- `VALIDATION_PLAN.md`: "Core API", "Target modules", "Term structure".
- The modules you build on, by their docstrings: `pymars/_gcv.py`, `pymars/_linalg.py`, `pymars/_pruning.py` (merged), and `pymars/_forward.py` from T11 stage 1 (branch `origin/t11-forward-stage1`; its pull request comes soon). Its public names: `forward_pass(X, Y, w=None, *, <the MarsParams fields>, record_candidates=False) -> ForwardPass`, a NamedTuple with the fields of `ForwardRecord`; `Termination` (CORE-4); `CandidateLog`; the `KIND_*` codes.

## Independence

Do not read `tests/reference/` (the oracle). You may import and call the reference as a black box in tests (its full `fit_mars` comes with PR #49), but do not read its source. Never read earth's code; use the fixtures in `validation/fixtures/`. Do not change `_forward.py`, `_scan.py` or the merged modules. If you need a change there, write it in your report, and the executor passes it on.

## Branch

Make `t12-core` from `origin/t11-forward-stage1`, because `_core` imports `_forward`. When T11 stage 1 merges, rebase onto `origin/main` (your commits touch other files, so this is a plain rebase), and only then open the pull request with base `main`. If stage 1 changes during its review, rebase onto its new head. Push your branch after every step as usual.

## What to build

`pymars/_core.py`:

- `MarsParams` (CORE-2): a frozen dataclass; the constructor checks every field and raises ValueError that names the field. An int field takes a Python or numpy integer but not a bool; a float field takes any real number.
- `MarsFit`, `ForwardRecord`, `PruningRecord`, `CandidateLog` (CORE-3) and `Termination` (CORE-4). Re-use `_forward`'s `Termination` and `CandidateLog`; one definition of each. `ForwardRecord` can be `_forward.ForwardPass` or a frozen dataclass made from it; state your choice in the docstring.
- `MarsFit.to_dict()` and `MarsFit.from_dict(d)` (CORE-5): nested dicts, the integer termination code, the dtypes and shapes of CORE-3. `from_dict` must accept the dict that the reference returns.
- `fit_mars(X, Y, w, params, *, record_candidates=False)` (CORE-1): the zero-weight drop (W-3), N (W-4), the resolved `max_terms` and `penalty`, the degenerate path (EDGE-1, GCV-7), the forward pass through the module attribute (`_forward.forward_pass(...)`, so a test can replace it), the pruning pass on the kept terms (pruning index m is forward index `kept[m]`), `pmethod` and `nprune`, the final fit on the original scale (PRUNE-8), and `selected` in forward indices. Where the spec gives a step to the forward pass (for example the y prescaling of EDGE-6 happens inside `forward_pass`), do not do it twice; say in the docstring which module does what.
- Pure function, never write into the inputs, and state the complexity.

## Tests (`tests/test_core.py`)

- `MarsParams`: every field at its bounds, out of range, of the wrong type (a bool for an int field, a string), numpy integers; frozen.
- `to_dict` and `from_dict`: a round trip that keeps the dtypes and shapes, also for an intercept-only fit and without a candidate log.
- `fit_mars` with a stub forward pass (monkeypatch `pymars._forward.forward_pass`) that returns earth's forward basis from the fixtures (`validation/fixtures/components/` pruning cases, and the S fixtures' forward terms): the selected terms, `rss_per_size`, `gcv_per_size`, the coefficients, `rss`, `gcv`, `rsq` and `grsq` against earth, with the plan's tolerances (RSS and GCV relative 1e-8, coefficients normwise relative 1e-6 when κ(B) ≤ 1e5).
- Weights through the core: zero-weight rows give the same fit as rows removed; integer weights give the same fit as repeated rows; a positive rescaling of all weights, as the spec says for frequency weights.
- Degenerate inputs (EDGE, GCV-7): N ≤ 1, a constant response, `max_terms` ≤ 2; the code, the record and the statistics.
- End to end at degree 1 with the real stage-1 forward pass: S01 and S04 at degree 1 in the matched mode against earth (the forward steps, the pruning record, the selected terms, the coefficients); and against the reference's `fit_mars` through `from_dict` once PR #49 has merged.
- Mutation checks with `python -B` and a fresh copy per mutant, in your own scratchpad subfolder.

## Size and report

At most about 800 changed lines that are not generated; split into two pull requests if it grows (for example the types and `to_dict`/`from_dict` first). Report in 30 lines or fewer when the pull request is ready: the number and head, the gate results, the fixture comparisons with their largest errors, the mutation results, and any spec question (raise it on #44).
