# pymars 2.0 fitting algorithm (spec)

Status: part 1 in progress (T01, issue #3). Part 1 covers terms, the GCV, the term limit, the spans and knots, the linear algebra contract, the pruning pass and the core API. Part 2 adds the forward pass, the stopping rules, Fast MARS, weights, several responses, the GLM refit, degenerate inputs, errors and the public API.

This file states the rules of the fitting algorithm, each with its source. Two implementations are written from it: the reference implementation in `tests/reference/` (the test oracle) and the fast implementation in `pymars/`. They may differ in method, never in results, except where a rule allows a numerical tolerance.

## How to read this spec

- Every rule has a stable ID in bold at its start, such as **GCV-1**. Code, tests and reviews cite these IDs. An ID is never reused for a different rule; a rule that is withdrawn keeps its ID with the note "withdrawn".
- Each rule states the behavior, then its source in brackets:
  - [F91 eq. 30] or [F91 §3.8]: Friedman (1991); [F93 §3.1]: Friedman (1993); [ESL §9.4]: Hastie, Tibshirani and Friedman (2009).
  - [Notes §3]: Milborrow's "Notes on the earth package" (2024), paraphrased.
  - [bbNN]: a black-box experiment with earth 5.3.4, whose script and output are in `validation/blackbox/`; [bbNN.k] names one CHECK line of that output.
  - [pymars departure: reason]: pymars 2.0 does not do what earth does; the section [Departures from earth](#departures-from-earth) lists them all.
  - [plan: section]: a decision in `VALIDATION_PLAN.md` (Q1 to Q6 and "Behavior target: scikit-learn first").
- A **quirk** is an earth behavior that looks accidental. Each quirk is marked where it occurs, with the choice pymars makes.
- "Must" and "never" are binding. A complexity target is binding for the fast code only.

## Notation

| Symbol | Meaning |
|---|---|
| X | The covariate matrix, n × p, float64 |
| Y | The response matrix, n × K, float64; a 1-D y is Y with K = 1 |
| n | The number of rows (cases) used in the fit, after rows with zero weight are dropped ([W-3](#weights)) |
| p | The number of columns (covariates) of X |
| K | The number of responses (columns of Y) |
| w | The case weights, w_i > 0 after the drop; w_i = 1 exactly when no weights are given |
| N | The weight sum Σ w_i; N = n when no weights are given |
| M | The number of terms of a model, including the intercept |
| M_max | The term limit of the forward pass (`max_terms`; earth's `nk`) |
| B | The n × M basis matrix: column j is term j evaluated at the rows of X |
| `dirs`, `cuts` | The (M, p) arrays that define the terms ([Terms](#terms)) |
| parent | The existing term that a new factor multiplies |
| active cases | For a parent term b, the cases i with b(x_i) ≠ 0 |
| d | The GCV penalty per knot (`penalty`) |
| C | The effective number of parameters of a model, used in the GCV |
| r | The rank of B (the number of linearly independent columns) |
| α | The probability in Friedman's span formulas, fixed at 0.05 |
| RSS | The weighted residual sum of squares, Σ_k Σ_i w_i (Y_ik − Ŷ_ik)² |
| TSS | The weighted total sum of squares, Σ_k Σ_i w_i (Y_ik − Ȳ_k)², with Ȳ_k the weighted mean of column k |
| (z)₊ | max(0, z) |
| ⌊z⌋ | The largest integer at most z; trunc(z) rounds toward zero |

Indices are 0-based in this spec, as in numpy: terms 0 to M − 1 (term 0 is the intercept), variables 0 to p − 1, cases 0 to n − 1. earth numbers terms and variables from 1; the harness converts.

## Conventions for all rules

- All arithmetic is float64. No function changes its inputs in place.
- No rule uses an absolute epsilon. Every tolerance is relative to a scale that the rule names.
- Every choice between equal candidates follows a fixed tie rule that the spec states. No result may depend on the order of the rows of X, or on which of several rows with the same x value comes first.
- Memory is O(n·(p + M_max)) for the fast code.

## Terms

(pending)

## GCV and fit statistics

(pending)

## Term limit

(pending)

## Spans

(pending)

## Candidate knots

(pending)

## Linear algebra contract

(pending)

## Pruning pass

(pending)

## Core API

(pending)

## Forward pass

(part 2)

## Stopping rules

(part 2)

## Fast MARS

(part 2)

## Weights

(part 2; the weight rules that part 1 needs are stated where they apply)

## Several responses

(part 2)

## GLM refit for the classifier

(part 2)

## Degenerate inputs

(part 2)

## Errors

(part 2)

## Public API

(part 2)

## Departures from earth

(pending)

## Open questions

(pending)

## Sources

- Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics* 19(1), 1-67. Cited as F91.
- Friedman, J. H. (1993). Fast MARS. Technical Report 110, Department of Statistics, Stanford University. Cited as F93.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd edition, section 9.4. Cited as ESL.
- Milborrow, S. (2024). Notes on the earth package. Vignette of the R package earth 5.3.4. Cited as Notes, by section, and only paraphrased.
- The black-box experiments in `validation/blackbox/`, run with R 4.4.3 and earth 5.3.4 ([the README there](../validation/blackbox/README.md) lists them).
