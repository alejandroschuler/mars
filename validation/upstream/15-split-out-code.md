# Title: Proposal: split out or remove the code that does not fit models

## Summary

The repository holds much more code outside the fitting path than in it. I counted this at upstream commit `d68b54a` (the tag `legacy-1.0.4-head` in the fork):

- the fitting modules and the scikit-learn wrappers: 3,639 lines of Python;
- the runtime, portable-specification, cluster and accelerator modules: 2,185 lines (`runtime.py`, `cluster.py`, `cluster_worker.py`, `accelerator*.py`, `specialized_accelerator_backends.py`, `_model_spec.py`);
- the Rust crate, `rust-runtime/`: 3,551 lines;
- bindings for R, Julia, Go, TypeScript and C#: 2,464 lines, plus 1,066 lines of Go at the root (`cmd/`, `mars/`, `go.mod`);
- 275 files under `conductor/`, 173 files under `docs/`, and 21 GitHub workflows.

Outside the fitting path there are 2,185 more lines of Python, about 7,100 lines of Rust and binding code, and the planning and documentation files. That code has to be tested and kept up to date, and it hides the fit from a reader. In the validation work, none of it was needed to fit a model, and several parts were switched off or unused (the Rust trainer runs the same search as the Python code, solves the normal equations, and is off by default).

## Why it matters

- Review. A contributor or a reviewer who wants to check the fit has to find it among cluster workers, accelerator backends and language bindings.
- Tests. A test suite of this size mostly tests the surrounding code against itself, and it does not catch problems in the fit (see the issue on the reference tests).
- Packaging. The package pulls in matplotlib at import time (see the issue on the dependency bounds), and the metadata and the tool settings disagree about Python versions.
- Claims. A feature that is present but not validated invites users to rely on it.

## Proposal

1. Decide what `mars-earth` is for. My suggestion is: one scikit-learn estimator family (regressor and classifier) with a documented relation to `earth`.
2. Move what is outside that scope to their own repositories or to branches, with a link from the README: the bindings (R, Julia, Go, TypeScript, C#), the portable runtime and JSON export, the cluster and accelerator modules, the CLI, and the Rust crate until it is faster than the Python code and compared with `earth`.
3. Keep plotting and explanation tools as thin helpers on scikit-learn's own tools (`PartialDependenceDisplay`), or as an optional extra.
4. Move planning and agent files (`conductor/`, `.agents/`, `.gemini/`, session logs) out of the package repository.
5. Keep the tag `legacy-1.0.4-head` or an equivalent, so that nothing is lost.

I would do this first, before any change to the fitting code, so that later changes are easy to review. Alternatively the fitting code can be replaced as a whole (see the note on the 2.0 code), and the rest can be dropped in the same step, as the fork does: it deletes every file listed above, keeps the license and the attribution, and keeps the history in the tag.

I know that some of these parts may matter to people I am not aware of. If you use any of them, please say which, and I will not propose to remove it.
