# pymars 2.0 fitting algorithm (spec)

Status: part 1 (T01, issue #3). Part 1 covers terms, the GCV, the term limit, the spans and knots, the linear algebra contract, the pruning pass and the core API. Part 2 adds the forward pass, the stopping rules, Fast MARS, weights, several responses, the GLM refit, degenerate inputs, errors and the public API.

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
| active cases | For a parent term b, the cases i with b(x_i) > 0 ([KNOT-1](#candidate-knots)) |
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

**TERM-1** A model with M terms on p covariates is stored as two arrays of shape (M, p): `dirs` (int8) and `cuts` (float64). `dirs[k, j]` is 0 when covariate j is not in term k, +1 when term k has the factor (x_j − `cuts[k, j]`)₊, −1 when it has the factor (`cuts[k, j]` − x_j)₊, and 2 when it has the linear factor x_j. Row 0 is the intercept, with every code 0. [earth's `dirs` and `cuts`; bb09.4]

**TERM-2** `cuts[k, j]` is 0.0 wherever `dirs[k, j]` is 0 or 2. earth stores the smallest value of x_j there for a linear factor and never uses it [bb09.4, bb09.5]; the harness compares `cuts` only where `dirs` is ±1. [plan: Term structure]

**TERM-3** Term k at a point x ∈ ℝ^p is B_k(x) = ∏ f(dirs[k, j], x_j, cuts[k, j]) over the j with dirs[k, j] ≠ 0, where f(+1, v, c) = (v − c)₊, f(−1, v, c) = (c − v)₊ and f(2, v, c) = v. The empty product, for the intercept, is 1. The formula holds at every finite x, also outside the range of the training data; nothing is clipped. `basis_matrix(X, dirs, cuts)` returns the n × M float64 matrix B with B[i, k] = B_k(x_i). The factors may be multiplied in any order, which changes the result by rounding only. [bb09.4]

**TERM-4** The degree of term k is the number of nonzero entries in row k of `dirs`; a linear factor counts as one. The intercept has degree 0. A row has one entry per covariate, so no covariate appears twice in a term. [ESL §9.4 (each input at most once in a product); F91 Algorithm 2; bb06.7 (a linear term is a parent at degree 2)]

**TERM-5** Labels, for `summary()`. The label of term k joins the labels of its factors with `*`, in increasing covariate order: `h(NAME-C)` for +1, `h(C-NAME)` for −1 and `NAME` for 2, where C is the knot formatted with the Python format spec `.6g`. The intercept's label is `(Intercept)`. NAME is the column name when the estimator was fitted on a table with column names (`feature_names_in_`), else `x` followed by the 0-based column index. A negative knot gives labels such as `h(x0--1.5)`, as in earth. [earth's term names, such as `h(x1-0.336266)`; pymars departure: earth lists the factors in the order of construction and numbers x from 1]

**TERM-6** For every forward term k, the forward pass records `parent[k]`, the index of the term that the new factor multiplied (−1 for the intercept), and `step[k]`, the forward step that added the term, counted from 1 (0 for the intercept). Column k of B then equals column `parent[k]` times the new factor. [plan: Term structure]

## GCV and fit statistics

**GCV-1** The effective number of parameters of a model with M terms is C(M) = M + d·(M − 1)/2 when d ≥ 0, and C(M) = 0 when d < 0. M counts the terms: not the rank of B and not the number of knots, and a single hinge or a linear term counts as one term like any other. The intercept-only model has C = 1 for every d ≥ 0. [bb01.1, bb01.4, bb02.1 to bb02.4, bb02.10, bb02.22; the penalty per knot as in ESL §9.4, with (M − 1)/2 knots] earth treats every negative d as −1 [bb01, printed cells for −0.5 and −2]; this is a quirk, and pymars copies it.

**GCV-2** GCV(RSS, M) = RSS / (N·(1 − C(M)/N)²) when C(M) < N, and +∞ when C(M) ≥ N. The comparison C(M) < N is exact in float64. [F91 eq. 30; bb01.1 to bb01.3] pymars departure: N = Σw, where earth uses the number of cases also in weighted fits [bb02.12] ([W-2](#weights)).

**GCV-3** With K responses, RSS and TSS are sums over the responses, and GCV-2 is applied to the summed RSS. GCV is linear in RSS, so this equals the sum of the per-response GCVs. [bb02.23 to bb02.26; Notes §2.9]

**GCV-4** The default penalty (`penalty=None`) is d = 2 when `max_degree` is 1, and d = 3 otherwise. [bb02 (fits a and b print penalty 2 and 3); ESL §9.4: c = 3, and 2 for additive models]

**GCV-5** RSq = 1 − RSS/TSS, where TSS is the weighted total sum of squares about the weighted means, summed over the responses. [bb02.5, bb02.17, bb02.27; Notes §13.10]

**GCV-6** GRSq = 1 − GCV(RSS, M)/GCV(TSS, 1), with the model's d and N. So GRSq = −∞ when C(M) ≥ N. [bb02.6, bb02.28; Notes §13.11]

**GCV-7** For the intercept-only model, RSq = 0 and GRSq = 0 by definition. This covers TSS = 0 and N ≤ 1, where GCV-5 and GCV-6 give 0/0 or ∞/∞. For M > 1 the model has TSS > 0 and N > 1 ([Degenerate inputs](#degenerate-inputs)), and GCV-5 and GCV-6 apply. [pymars: the values are always defined]

**GCV-8** Weighted sums are not normalized: RSS = Σ w_i r_i² and TSS = Σ w_i (y_i − ȳ_w)², where ȳ_w is the weighted mean. earth does the same for weights that are not all equal [bb02.14, bb02.17], and it ignores weights that are all equal [bb02.15]. pymars never ignores weights ([W-1](#weights)).

## Term limit

**LIMIT-1** The default term limit (`max_terms=None`) is M_max = min(200, max(20, 2p)) + 1, with p the number of columns of X. It does not depend on n. [bb03.1, bb03.2; Notes §13.13]

**LIMIT-2** Every forward step counts 2 toward M_max, whether it adds a pair, a single hinge or a linear term. After s steps the counted size is 1 + 2s, and the model has at most 1 + 2s terms. A step is taken only when 1 + 2(s + 1) ≤ M_max, so the forward pass takes at most ⌊(M_max − 1)/2⌋ steps; when M_max is even, its last slot is never used; and when M_max ≤ 2, the forward pass takes no step. [bb03.4, bb03.7, bb03.9, bb03.10, bb10.1, bb11.5; Notes §13.14] A single hinge using two slots is a quirk; pymars copies it.

**LIMIT-3** `nprune` is None or an integer ≥ 1: the largest size, in terms including the intercept, that the pruning pass may select ([PRUNE-5](#pruning-pass)). None means no limit. [bb07.16 to bb07.18; Notes §13.15]

## Spans

In this section a knot search has a parent term b and a covariate x. N is the weight sum of all n cases, and N_b the weight of the active cases of b ([KNOT-1](#candidate-knots)); for the intercept, N_b = N. p is the number of columns of X, and α = 0.05.

**SPAN-1** The automatic minspan (`minspan=None`) is L = max(1, trunc(−log₂(−ln(1 − α)/(p·N_b)) / 2.5)). [F91 eq. 43; bb05.4, bb05.6 (truncated, not rounded), bb06.1 (with the parent's active count, not n)] earth counts the active cases where pymars uses their weight ([W-2](#weights)). The outer max(1, ·) never binds in earth, where p·N_b ≥ 1; it keeps L ≥ 1 for small weights. [pymars]

**SPAN-2** The automatic endspan (`endspan=None`) is E = trunc(3 − log₂(α/p)). It depends only on p. [F91 eq. 45; bb05.5]

**SPAN-3** A user `minspan` or `endspan` is an integer ≥ 1, used in place of L or E. earth reads 0 as automatic, and a negative minspan as a request for at most that many evenly spaced knots; pymars uses None for automatic and does not support a negative minspan ([Departures](#departures-from-earth)). earth raises an error when minspan exceeds n [bb05.9]; pymars accepts such a value, and the scan of [KNOT-3](#candidate-knots) then finds few knots [bb05.8]. [pymars departure: the folds of a cross-validation vary in size]

**SPAN-4** For a parent of degree 1 or more, the endspan becomes E₁ = E + ⌊a·E + 0.5⌋, where a = `adjust_endspan`, a float ≥ 0 (default 2.0). For the intercept, E₁ = E. [bb06.2 (36 settings), bb06.4] This is a quirk: earth describes the argument as a factor on the endspan, but the endspan it uses is E·(1 + a) rounded half up, so a = 1 doubles it. pymars copies the observed rule.

**SPAN-5** The endspan used in the scan is E* = max(1, min(E₁, ⌊N/2⌋ − 1)). The cap applies to user and automatic values alike, after the adjustment, and N counts all cases, not only the active ones. [bb05.5, bb05.7, bb06.2] earth uses n in place of N ([W-2](#weights)).

**SPAN-6** The minspan is not capped. [bb05.8]

## Candidate knots

This section fixes, for one parent b and one covariate x, the list of knot values at which the forward pass evaluates a hinge. The forward-pass rules (part 2) say which candidates it builds from these knots.

**KNOT-1** A case i is active for the parent b when b(x_i) > 0. For a parent with a linear factor, a case where the parent is negative is inactive, although the new column is nonzero there. Every case is active for the intercept. [bb06.5, bb06.7] Inactive negative cases are a quirk; pymars copies it.

**KNOT-2** Sort the n cases by x, ascending. Among cases with equal x, the inactive cases come before the active ones. Cases with equal x and equal activity may come in any order, because the scan does not tell them apart. [pymars departure: earth's order among equal values of x depends on the row order, and a permutation of the rows changes its knots, bb06.8, bb06.9; the fit must not depend on the row order, plan: Behavior target, "Ties and row order"]

**KNOT-3** The scan, with unit weights. Let x_(1) ≤ … ≤ x_(n) be the sorted values, a_q ∈ {0, 1} the activity of the case at position q (1-based), L the minspan ([SPAN-1](#spans), [SPAN-3](#spans)), E* the endspan ([SPAN-5](#spans)), and v the largest x among the active cases. A counter starts at S = E* + ⌈r/2⌉, where r = (n − 2E* − 1) mod L, the remainder in [0, L). Then, for q = n, n − 1, …, E* + 2 in turn:

1. let t = x_(q−1);
2. if t ≥ v, go on to the next q, and the counter does not change;
3. otherwise, if a_q = 1, decrease the counter by 1; if it is then 0, append t to the list and set the counter to L.

The list may repeat a value; the candidate knots are its distinct values. With no active case there is no knot. [bb05.1 to bb05.3 (240 settings with the intercept), bb06.3 and bb06.4 (120 searches), bb06.10 (80 searches with ties)]

**KNOT-4** These properties follow from KNOT-3. Tests may check them.

- No knot is at or above v, so the largest active value is never a knot.
- No knot is below x_(E*+1), so the E* lowest cases are never knots. For a parent other than the intercept these cases may be inactive, and then the lowest active values are not protected. This is a quirk [bb06, the printed example]; pymars copies it.
- A knot can be the value of an inactive case, since a knot is the value just below an active case in the full order [bb06.6]. This is a quirk; pymars copies it.
- At the top, only active cases move the counter, so the first knot lies just below the S-th active case that counts, from the top.
- For the intercept, the knots are every L-th value from the top of x_(E*+1), …, x_(n−E*); the slack (n − 2E* − 1) mod L is split between the two ends, with the larger half at the top [bb05.2, bb05.3].
- A value can be listed more than once, and each listing moves the counter.

**KNOT-5** The lowest knot, x_(E*+1), equals the smallest value x_(1) only when that value is repeated. Such a knot t gives b·(x − t)₊ = b·(x − t) at every case, which lies in the span of b and b·x; the collinearity test ([LA-3](#linear-algebra-contract)) rejects it [bb05, the example with a repeated minimum: TolG 0]. The linear term b·x, which earth reports as a knot at the minimum, is not a scan candidate: the forward-pass rules (part 2) evaluate it separately. [Notes §6.3]

**KNOT-6** With weights, the same scan runs on cumulative weight, so that an integer weight w_i gives the same list as w_i copies of row i. Order the cases as in KNOT-2, and let W_q be the weight of the first q cases (W_0 = 0, W_n = N); the case at position q covers the interval (W_{q−1}, W_q]. For a real u in (0, N], let x(u) and a(u) be the value and the activity of the case whose interval holds u. The scan visits u = N, N − 1, N − 2, … while u ≥ E* + 2, and at each u it takes steps 1 to 3 of KNOT-3 with t = x(u − 1) and a(u) in place of x_(q−1) and a_q. The start is S = E* + ⌈r/2⌉ with r = z − L·⌊z/L⌋ and z = N − 2E* − 1. N_b in [SPAN-1](#spans) is the weight of the active cases. With unit weights this is KNOT-3. [pymars departure: frequency weights, plan: Behavior target and Sample weights; earth ignores weights in the spans and knots, Notes §2.10]

**KNOT-7** The fast code sorts each column of X once, in O(p·n·log n), and builds the list for one parent and one covariate in O(n), also with weights: the activity is fixed within one case's interval, so the steps inside it can be counted at once. [plan: Fast path]

## Linear algebra contract

This section states results, not methods. ⟨u, v⟩_w = Σ_i w_i u_i v_i is the weighted inner product, and ‖u‖_w² = ⟨u, u⟩_w.

**LA-1** For a set S of columns of B, RSS(S) = Σ_k ‖Y_k − P_S Y_k‖_w², where P_S is the w-orthogonal projection onto the span of the columns in S. RSS(S) is defined also when these columns are linearly dependent, since dependent columns do not change the span. Every RSS in this spec, in the forward pass, in the pruning pass and in the records, is of this kind and on the original scale of Y. [least squares; bb09.1]

**LA-2** When the forward pass evaluates a candidate that adds the columns H to the current basis, the RSS of the candidate is RSS(current columns ∪ H), and its RSS reduction is the RSS before the step minus this value. A new column that lies in the span of the current columns reduces the RSS by 0. [LA-1]

**LA-3** The collinearity test. Let h = b·(x − t)₊ be the hinge column of a candidate knot t on parent b and covariate x. Let E be the current columns, together with the column b·x when the candidate is a pair (that is, when b·x is not in the span of the current columns). The ratio

ρ(t) = ‖h − P_E h‖_w² / ‖h − h̄‖_w²,

with h̄ the weighted mean of h, is 1 − R² of h regressed on E; E contains the intercept. The knot is rejected when ρ(t) < τ, or when ‖h − h̄‖_w = 0, and a rejected knot is not a candidate. The tolerance is τ = 0.01 while the forward pass has taken at most 6 steps, so in steps 1 to 7, and τ = 1e-5 from step 8 on. [bb11.1 (at the first step the rejected ratios reach 0.009964 and the accepted ones start at 0.010021), bb11.3 to bb11.6 (the change after 7 steps, for pairs and single hinges), bb11.7 (hinge parents at degree 2); bb11.2 (a rejected knot keeps the RSS of b·x alone); plan: F5] earth changes τ when an internal slot counter, 2s + 2 for a single hinge and 2s + 3 for a pair after s steps, reaches 16 [bb11.5, bb11.6]; in steps this is the rule above. This is a quirk; pymars copies it. With weights, earth may run a different code path for this test (Notes §2.10 and earth's help on `Force.weights`); pymars applies the test with the weighted inner product, so that integer weights match repeated rows. [pymars departure for weighted fits]

**LA-4** A rank-deficient solve. Where the spec asks for least-squares coefficients on columns that may be linearly dependent ([PRUNE-8](#pruning-pass)), take the columns in their order. A column is dependent when the w-norm of its part orthogonal to the kept earlier columns is less than 1e-7 times its w-norm, not centered. A dependent column gets the coefficient 0, and the other columns get the least-squares solution on the kept columns. This is what R's `lm.fit` and `lm.wfit` do with their default tolerance, where the dependent coefficient is NA. [bb09.7 to bb09.14]

**LA-5** The reference and the fast code compute each RSS of LA-1 to LA-3 to within a relative error of 1e-8 of the RSS before the step, and they make the same choices except at near-ties, as the plan's tolerance table and its section "Ties" define. [plan: What is compared, and the tolerances]

**LA-6** An implementation may scale each response (subtract a constant, divide by a positive constant) inside the forward pass for numerical stability. The scaling does not change the rankings, the collinearity test or the stopping rules beyond rounding, and every value in the records is reported on the original scale. [earth scales a single response in its forward pass, earth's help on `Scale.y`; part 2 states pymars's rule]

## Pruning pass

**PRUNE-1** `pmethod` is `"backward"` (the default) or `"none"`. earth's other methods are not supported ([Departures](#departures-from-earth)).

**PRUNE-2** The input is the M_f forward terms, numbered 0 to M_f − 1 in the order the forward pass added them (term 0 is the intercept), with Y, w, d and N. For each size m = 1, …, M_f the pass keeps the lowest RSS found so far, R[m], and its set of terms, T[m]; at the start R[m] = +∞.

**PRUNE-3** A working order o = (o_1, …, o_{M_f}) starts as (0, 1, …, M_f − 1). To offer the order: for m = 1, …, M_f, let U = {o_1, …, o_m}, and if RSS(U) < R[m], set R[m] = RSS(U) and T[m] = U. Offer the starting order. Then for pos = M_f, M_f − 1, …, 2 in turn:

1. among the positions i = 2, …, pos, choose the one for which RSS({o_1, …, o_pos} without o_i) is lowest; if several are equal, choose the one whose term has the largest index (the term added last);
2. record o_i as the term removed at this stage, and move it to position pos, so that o_{i+1}, …, o_pos move one place to the left;
3. offer the new order.

Position 1 always holds the intercept, so the intercept is never removed, and T[1] = {0}. The sets T[m] need not be nested. [bb07.12, bb07.24 (130 random bases and 16 ordinary fits), bb07.2 to bb07.5, bb07.13; F91 Algorithm 3 for plain backward deletion; earth uses code from the leaps package here, Notes §2.1 and §2.5] In earth, rounding decides an exact tie in step 1 [bb08.4 to bb08.10]; the pymars tie rule is a choice (a departure for exact ties only).

**PRUNE-4** The records are `removed`, the terms removed at the stages pos = M_f, …, 2 in that order; `rss_per_size[m − 1]` = R[m] and `gcv_per_size[m − 1]` = GCV(R[m], m) by [GCV-2](#gcv-and-fit-statistics), for m = 1, …, M_f; and the sets T[m]. [bb07.8, bb07.9]

**PRUNE-5** Let m_max = M_f, or min(M_f, `nprune`) when `nprune` is given. The selected size m* is the smallest m in 1, …, m_max at which `gcv_per_size[m − 1]` is lowest, and the selected terms are T[m*], in increasing order. [bb07.10, bb07.17, bb08.13 (tied minima go to the smaller size)]

**PRUNE-6** `nprune` changes only the range in PRUNE-5; the stages and the records are the same as without it. With d < 0, so that GCV = RSS/N, and `nprune` = k, earth selected exactly k terms in bb07.19, as expected when R[m] falls as m grows. [bb07.16 to bb07.19]

**PRUNE-7** With `pmethod="none"`, the stages and the records are computed as for `"backward"`, and all M_f forward terms are selected. [bb07.14, bb07.15]

**PRUNE-8** The final coefficients are the weighted least-squares coefficients of Y on the selected columns, in increasing term order and on the original scale of Y, with [LA-4](#linear-algebra-contract) for dependent columns; `coef` has shape (m*, K). The final `rss`, `gcv`, `rsq` and `grsq` follow [GCV-2](#gcv-and-fit-statistics) to [GCV-7](#gcv-and-fit-statistics) with M = m*, and `rss` equals `rss_per_size[m* − 1]` up to rounding. [bb09.1, bb09.1b, bb09.3; Notes §2.5]

**PRUNE-9** The fast code updates or downdates one factorization instead of refitting each subset; a stage may cost O(n·M_f²). The reference may refit every subset. [plan: Fast path]

## Core API

**CORE-1** `pymars._core.fit_mars(X, Y, w, params, *, record_candidates=False)` returns a `MarsFit`.

- X is a float64 array of shape (n, p), finite.
- Y is a float64 array of shape (n, K), with K ≥ 1, finite.
- w is a float64 array of shape (n,), finite and ≥ 0 with a positive sum, or None, which means w_i = 1 exactly.
- params is a `MarsParams`.

The function first drops the rows with zero weight ([W-3](#weights)), and it never changes its inputs. The estimators check the data before the call ([Errors](#errors)), so the core assumes valid input. The function is pure: the same inputs give the same output, and the output does not depend on the order of the rows, apart from near-ties.

**CORE-2** `MarsParams` is a frozen dataclass with the fields below. Its constructor raises ValueError, with a message that names the field, for a value outside the range. The resolved values of the fields that may be None are in `MarsFit`.

| Field | Type | Default | Allowed values | None means |
|---|---|---|---|---|
| `max_degree` | int | 1 | ≥ 1 | |
| `max_terms` | int or None | None | ≥ 1 | [LIMIT-1](#term-limit) |
| `penalty` | float or None | None | finite; a negative value means C = 0 | [GCV-4](#gcv-and-fit-statistics) |
| `thresh` | float | 0.001 | ≥ 0, finite | |
| `minspan` | int or None | None | ≥ 1 | [SPAN-1](#spans) |
| `endspan` | int or None | None | ≥ 1 | [SPAN-2](#spans) |
| `adjust_endspan` | float | 2.0 | ≥ 0, finite | |
| `auto_linpreds` | bool | True | | |
| `fast_k` | int | 20 | ≥ 0; 0 turns Fast MARS off | |
| `fast_beta` | float | 1.0 | ≥ 0, finite | |
| `pmethod` | str | `"backward"` | `"backward"` or `"none"` | |
| `nprune` | int or None | None | ≥ 1 | no limit |

An int field accepts a Python int or a numpy integer, and a bool is not an int here. A float field accepts any real number type.

**CORE-3** `MarsFit` is a frozen dataclass. M is the number of selected terms, M_f the number of forward terms and S the number of forward steps.

| Field | Type and shape | Content |
|---|---|---|
| `dirs` | int8 (M, p) | the selected terms, in increasing forward order |
| `cuts` | float64 (M, p) | their knots |
| `coef` | float64 (M, K) | [PRUNE-8](#pruning-pass) |
| `selected` | int64 (M,) | the forward indices of the selected terms, increasing, starting with 0 |
| `rss`, `gcv`, `rsq`, `grsq` | float | of the final model, [PRUNE-8](#pruning-pass) |
| `n_eff` | float | N = Σw after the drop |
| `max_terms` | int | the resolved M_max |
| `penalty` | float | the resolved d |
| `forward` | `ForwardRecord` | below |
| `pruning` | `PruningRecord` | below |

`ForwardRecord`:

| Field | Type and shape | Content |
|---|---|---|
| `dirs`, `cuts` | int8, float64 (M_f, p) | all forward terms, in the order added |
| `parent`, `step` | int64 (M_f,) | [TERM-6](#terms) |
| `rss` | float64 (S + 1,) | `rss[0]` = TSS, the RSS of the intercept alone; `rss[s]` = the RSS after step s |
| `termination` | `Termination` | [CORE-4](#core-api) |
| `candidates` | `CandidateLog` or None | present when `record_candidates=True` |

`CandidateLog`, one entry per step:

| Field | Type and shape | Content |
|---|---|---|
| `best_rss` | float64 (S,) | the RSS of the chosen candidate, equal to `rss[s]` |
| `second_rss` | float64 (S,) | the lowest RSS among the other candidates evaluated at the step; +∞ if there is none |
| `second_parent`, `second_variable` | int64 (S,) | the parent and the covariate of that candidate; −1 if there is none |
| `second_knot` | float64 (S,) | its knot; NaN for a linear term or when there is none |
| `second_kind` | int8 (S,) | 0 none, 1 pair, 2 single hinge, 3 linear term |

Two candidates are different when they differ in the parent, the covariate, the kind or the knot value, so a knot listed twice ([KNOT-3](#candidate-knots)) is one candidate. [plan: Harness, Ties]

`PruningRecord`:

| Field | Type and shape | Content |
|---|---|---|
| `removed` | int64 (M_f − 1,) | [PRUNE-4](#pruning-pass) |
| `rss_per_size`, `gcv_per_size` | float64 (M_f,) | index m − 1 is size m |
| `subsets` | bool (M_f, M_f) | row m − 1 marks the terms of T[m] |
| `selected_size` | int | m* |

With M_f = 1, `removed` is empty, `rss_per_size` is [TSS], and `subsets` is [[True]].

**CORE-4** `Termination` is an IntEnum. It uses earth's code where earth has one [bb10.1 to bb10.6]. Part 2 states when each code applies and in which order the rules are checked.

| Value | Name | The forward pass stopped because | earth |
|---|---|---|---|
| 0 | `DEGENERATE` | it did not run: N ≤ 1 or TSS = 0 ([Degenerate inputs](#degenerate-inputs)) | none |
| 1 | `NO_ROOM` | M_max ≤ 2, so no step fits ([LIMIT-2](#term-limit)) | 1 |
| 2 | `GRSQ_NEG_INF` | GRSq is −∞ | 2 |
| 3 | `GRSQ_LOW` | GRSq is below −10 | 3 |
| 4 | `RSQ_CHANGE_SMALL` | RSq changed by less than `thresh` | 4 |
| 5 | `RSQ_HIGH` | RSq reached 1 − `thresh` | 5 |
| 6 | `NO_GAIN` | no candidate lowers the RSS | 6 |
| 7 | `TERM_LIMIT` | the term limit was reached after at least one step | 7 |

**CORE-5** The reference implementation returns the same fields as a dict. The keys are the field names above; `forward`, `pruning` and `candidates` are nested dicts; `termination` is the integer code; the arrays have the dtypes and shapes above. `MarsFit.from_dict(d)` builds a `MarsFit` from such a dict, and `MarsFit.to_dict()` gives it back, so that tests can compare the two.

**CORE-6** The estimators call the core through its module: `pymars/_estimators.py` does `from pymars import _core` and calls `_core.fit_mars(...)` during `fit`, never a name bound at import. A test puts the reference in its place with `monkeypatch.setattr(pymars._core, "fit_mars", adapter)`, where `adapter` calls the reference and returns `MarsFit.from_dict(...)`. [plan: Core API]

**CORE-7** Targets for the fast code: a forward step costs O(p·n·r) for each parent it scores, with r the rank of the current basis ([plan: Cost model](../VALIDATION_PLAN.md#cost-model)); the pruning pass meets [PRUNE-9](#pruning-pass); memory is O(n·(p + M_max)). The reference has no targets, but it must handle n ≤ 300, p ≤ 6 and degree ≤ 3.

## Forward pass

(part 2)

## Stopping rules

(part 2)

## Fast MARS

(part 2)

## Weights

Part 1 states the weight rules that the sections above use. Part 2 completes this section.

**W-1** Weights are frequency weights: an integer weight w_i gives the same fit as w_i copies of row i, and weights that are all 1 give the same fit as no weights. No weights means w_i = 1 exactly. [plan: Behavior target, "Sample weights"]

**W-2** N = Σ w_i takes the place of the number of cases everywhere: in the GCV and its rule C ≥ N ([GCV-2](#gcv-and-fit-statistics)), in GRSq, in the spans ([SPAN-1](#spans), [SPAN-5](#spans)) and in the knot scan ([KNOT-6](#candidate-knots)); and every sum of squares is weighted. [pymars departure: earth uses the number of cases in the GCV and ignores weights in the spans and knots, bb02.12, Notes §2.10; it also ignores weights that are all equal, bb02.15]

**W-3** Rows with zero weight are dropped before anything else, so they count neither in n and N nor in the spans and the knot scan. [pymars departure: earth replaces a zero weight by a very small positive one, Notes §2.10, so such rows still count as cases in its spans and knots]

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

Each row names the rule where pymars 2.0 does not do what earth 5.3.4 does. Part 2 adds the forward pass, the GLM refit and the public API.

| Rules | earth 5.3.4 | pymars 2.0 | Reason |
|---|---|---|---|
| [GCV-2](#gcv-and-fit-statistics), [W-2](#weights) | the number of cases n in the GCV, also with weights | N = Σw | frequency weights (plan) |
| [GCV-8](#gcv-and-fit-statistics), [W-2](#weights) | weights that are all equal are ignored | used as given, so their scale matters | frequency weights |
| [SPAN-1](#spans), [SPAN-5](#spans), [KNOT-6](#candidate-knots) | weights ignored in the spans and knots | cumulative weight | frequency weights |
| [W-3](#weights) | a zero weight becomes a very small weight | the row is dropped | scikit-learn's weight checks |
| [KNOT-2](#candidate-knots) | the order among equal x values depends on the row order | inactive cases first, then active ones | the fit must not depend on the row order |
| [PRUNE-3](#pruning-pass) | rounding decides an exact tie | the term added last is removed | a fixed tie rule |
| [LA-3](#linear-algebra-contract) | the weighted code may skip the collinearity test | the test uses the weighted inner product | integer weights must match repeated rows |
| [SPAN-3](#spans) | a minspan above n is an error | accepted | folds of varying size |
| [SPAN-3](#spans) | minspan 0 or endspan 0 means automatic; a negative minspan asks for evenly spaced knots | None means automatic; negative values are not supported (a `later` issue) | scikit-learn style (plan) |
| [PRUNE-1](#pruning-pass) | `pmethod` also `"exhaustive"`, `"forward"`, `"seqrep"`, `"cv"` | `"backward"` and `"none"` only | plan (`later` issues) |
| [TERM-2](#terms) | a linear factor stores the smallest x value in `cuts` | 0.0 | representation only; the basis is the same |
| [TERM-5](#terms) | labels in the order of construction, covariates numbered from 1 | increasing covariate order, numbered from 0 | labels only |

## Quirks

Quirks that pymars copies, so that its fits match earth's: every negative penalty acts as −1 ([GCV-1](#gcv-and-fit-statistics)); a single hinge or a linear term uses two slots of the term limit ([LIMIT-2](#term-limit)); the adjusted endspan is E·(1 + a), rounded ([SPAN-4](#spans)); a negative value of a linear parent is inactive ([KNOT-1](#candidate-knots)); the lower end of the scan counts inactive cases, and a knot can be the value of an inactive case ([KNOT-4](#candidate-knots)); the collinearity tolerance changes after 7 steps ([LA-3](#linear-algebra-contract)).

The quirk that pymars does not copy: earth's knots can depend on the row order when equal x values have different activity ([KNOT-2](#candidate-knots)).

## Open questions

- **OQ-1** ([SPAN-4](#spans)). For a non-integer `adjust_endspan`, the forms E + ⌊a·E + 0.5⌋ and ⌊(1 + a)·E + 0.5⌋ can differ when a·E is a half that float64 represents inexactly. Both fit the 36 settings of bb06. pymars uses the first. An experiment that settles it needs values of a and E where the two forms differ in float64. The default a = 2 is not affected.
- **OQ-2** ([PRUNE-3](#pruning-pass)). earth decides an exact tie in the pruning pass by rounding [bb08]. The conformance tests (T07) should treat an exact or near tie in step 1 of PRUNE-3 like a near-tie in the forward pass, and stop the structural comparison there. The threshold for a pruning near-tie is for T07 to fix.
- **OQ-3** ([KNOT-6](#candidate-knots), [LA-3](#linear-algebra-contract)). For weights that are not integers, pymars's spans, knots and collinearity test have no earth counterpart. The tests check that integer weights give the same fit as repeated rows, and that the reference and the fast code agree for other weights. No experiment can settle these rules against earth.

## Sources

- Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics* 19(1), 1-67. Cited as F91.
- Friedman, J. H. (1993). Fast MARS. Technical Report 110, Department of Statistics, Stanford University. Cited as F93.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd edition, section 9.4. Cited as ESL.
- Milborrow, S. (2024). Notes on the earth package. Vignette of the R package earth 5.3.4. Cited as Notes, by section, and only paraphrased.
- The black-box experiments in `validation/blackbox/`, run with R 4.4.3 and earth 5.3.4 ([the README there](../validation/blackbox/README.md) lists them).
