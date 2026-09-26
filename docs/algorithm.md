# pymars 2.0 fitting algorithm (spec)

Status: spec v1, parts 1 and 2 (T01, issue #3). Part 1 covers terms, the GCV, the term limit, the spans and knots, the linear algebra contract, the pruning pass and the core API. Part 2 adds the forward pass, the stopping rules, Fast MARS, weights, several responses, the GLM refit, degenerate inputs, errors and the public API.

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
| S | The number of steps that the forward pass took |
| α | The probability in Friedman's span formulas, fixed at 0.05 |
| σ_v² | The weighted variance of covariate v over the n cases: Σ w_i (x_iv − x̄_v)² / N, with x̄_v the weighted mean |
| τ_N | The tolerance 1e-8·N for sums of weights ([W-4](#weights)) |
| RSS | The weighted residual sum of squares, Σ_k Σ_i w_i (Y_ik − Ŷ_ik)² |
| TSS | The weighted total sum of squares, Σ_k Σ_i w_i (Y_ik − Ȳ_k)², with Ȳ_k the weighted mean of column k |
| (z)₊ | max(0, z) |
| ⌊z⌋ | The largest integer at most z; trunc(z) rounds toward zero |

Indices are 0-based in this spec, as in numpy: terms 0 to M − 1 (term 0 is the intercept), variables 0 to p − 1, cases 0 to n − 1. earth numbers terms and variables from 1; the harness converts.

## Conventions for all rules

- All arithmetic is float64. No function changes its inputs in place.
- No rule uses an absolute epsilon. Every tolerance is relative to a scale that the rule names.
- A test whether data are constant compares the input values exactly, never a computed sum with 0: a response is constant when all its values over the n cases are equal, and a column is constant on a set of cases when all its values there are equal, each value computed as in [TERM-3](#terms). A sum of squares or a variance of constant values is then 0 by definition.
- Multiplying a covariate by a positive constant does not change the fit, apart from rounding and ties: the knots scale, and the other results stay the same ([LA-7](#linear-algebra-contract)). A shift of a covariate can change the fit when the covariate is a linear factor in a term of degree 2 or more, because a linear factor is x itself, and in earth too; shift-invariance tests (T16) must leave such fits out or use `auto_linpreds=False` ([LA-7](#linear-algebra-contract)).
- Every choice between equal candidates follows a fixed tie rule that the spec states. No result may depend on the order of the rows of X, or on which of several rows with the same x value comes first.
- Memory is O(n·(p + M_max)) for the fast code.

## Terms

**TERM-1** A model with M terms on p covariates is stored as two arrays of shape (M, p): `dirs` (int8) and `cuts` (float64). `dirs[k, j]` is 0 when covariate j is not in term k, +1 when term k has the factor (x_j − `cuts[k, j]`)₊, −1 when it has the factor (`cuts[k, j]` − x_j)₊, and 2 when it has the linear factor x_j. Row 0 is the intercept, with every code 0. [earth's `dirs` and `cuts`; bb09.4]

**TERM-2** `cuts[k, j]` is 0.0 wherever `dirs[k, j]` is 0 or 2. earth stores the smallest value of x_j there for a linear factor and never uses it [bb09.4, bb09.15]; the harness compares `cuts` only where `dirs` is ±1. [plan: Term structure]

**TERM-3** Term k at a point x ∈ ℝ^p is B_k(x) = ∏ f(dirs[k, j], x_j, cuts[k, j]) over the j with dirs[k, j] ≠ 0, where f(+1, v, c) = (v − c)₊, f(−1, v, c) = (c − v)₊ and f(2, v, c) = v. The empty product, for the intercept, is 1. The formula holds at every finite x, also outside the range of the training data; nothing is clipped. `basis_matrix(X, dirs, cuts)` returns the n × M float64 matrix B with B[i, k] = B_k(x_i). The factors may be multiplied in any order, which changes the result by rounding only. [bb09.4]

**TERM-4** The degree of term k is the number of nonzero entries in row k of `dirs`; a linear factor counts as one. The intercept has degree 0. A row has one entry per covariate, so no covariate appears twice in a term. [ESL §9.4 (each input at most once in a product); F91 Algorithm 2; bb06.7 (a linear term is a parent at degree 2)]

**TERM-5** `summary()` shows a label for each term. The label of term k joins the labels of its factors with `*`, in increasing covariate order: `h(NAME-C)` for +1, `h(C-NAME)` for −1 and `NAME` for 2, where C is the knot formatted with the Python format spec `.6g`. The intercept's label is `(Intercept)`. NAME is the column name when the estimator was fitted on a table with column names (`feature_names_in_`), else `x` followed by the 0-based column index. A negative knot gives labels such as `h(x0--1.5)`, as in earth. [earth's term names, such as `h(x1-0.336266)`; pymars departure: earth lists the factors in the order of construction and numbers x from 1]

**TERM-6** For every forward term k, the forward pass records `parent[k]`, the index of the term that the new factor multiplied, and `step[k]`, the forward step that added the term, counted from 1. For the intercept, `parent[0]` = −1 and `step[0]` = 0; a term whose parent is the intercept has `parent[k]` = 0. Column k of B equals column `parent[k]` times the new factor. [plan: Term structure]

## GCV and fit statistics

**GCV-1** The effective number of parameters of a model with M terms is C(M) = M + d·(M − 1)/2 when d ≥ 0, and C(M) = 0 when d = −1. M counts the terms: not the rank of B and not the number of knots, and a single hinge or a linear term counts as one term like any other. The intercept-only model has C = 1 for every d ≥ 0. [bb01.1, bb01.4, bb02.1 to bb02.4, bb02.10, bb02.22; the penalty per knot as in ESL §9.4, with (M − 1)/2 knots] A penalty below 0 other than −1 is invalid ([CORE-2](#core-api)), as in earth, whose fit stops with an error for −0.5, −2 and −10 [bb01.5].

**GCV-2** GCV(RSS, M) = RSS / (N·(1 − C(M)/N)²) when C(M) < N − τ_N, and +∞ otherwise, with N as in [W-4](#weights). [F91 eq. 30; bb01.1 to bb01.3] pymars departure: N = Σw, where earth uses the number of cases also in weighted fits [bb02.12] ([W-2](#weights)). A degenerate fit overrides this rule ([GCV-7](#gcv-and-fit-statistics)). With unit weights the test C(M) < N − τ_N differs from earth's C < n only when C is within 1e-8·n below n, which needs a penalty tuned to that precision.

**GCV-3** With K responses, RSS and TSS are sums over the responses, and GCV-2 is applied to the summed RSS. GCV is linear in RSS, so this equals the sum of the per-response GCVs. [bb02.23 to bb02.26; Notes §2.9]

**GCV-4** The default penalty (`penalty=None`) is d = 2 when `max_degree` is 1, and d = 3 otherwise. [bb02.30; ESL §9.4: c = 3, and 2 for additive models]

**GCV-5** RSq = 1 − RSS/TSS, where TSS is the weighted total sum of squares about the weighted means, summed over the responses. [bb02.5, bb02.17, bb02.27; Notes §13.10]

**GCV-6** GRSq = 1 − GCV(RSS, M)/GCV(TSS, 1), with the model's d and N. So GRSq = −∞ when C(M) ≥ N. [bb02.6, bb02.28; Notes §13.11]

**GCV-7** For the intercept-only model, RSq = 0 and GRSq = 0 by definition. A fit is degenerate when N ≤ 1 or when every response is constant (Conventions: all its values over the n cases are equal); then TSS = 0 by definition ([Degenerate inputs](#degenerate-inputs)). A degenerate fit is the intercept alone, and its `gcv` and `gcv_per_size[0]` are +∞, which overrides GCV-2. For M > 1 the fit is not degenerate, so N > 1 and some response is not constant, and GCV-5 and GCV-6 apply. [plan: Behavior target, "Degenerate inputs"; pymars departure: earth gives gcv 0 and rsq and grsq NaN for a constant y, bb10.7]

**GCV-8** Weighted sums are not normalized: RSS = Σ w_i r_i² and TSS = Σ w_i (y_i − ȳ_w)², where ȳ_w is the weighted mean. earth does the same for weights that are not all equal [bb02.14, bb02.17], and it ignores weights that are all equal [bb02.15]. pymars never ignores weights ([W-1](#weights)).

## Term limit

**LIMIT-1** The default term limit (`max_terms=None`) is M_max = min(200, max(20, 2p)) + 1, with p the number of columns of X. It does not depend on n. [bb03.1, bb03.2; Notes §13.13]

**LIMIT-2** Every forward step counts 2 toward M_max, whether it adds a pair, a single hinge or a linear term. After s steps the counted size is 1 + 2s, and the model has at most 1 + 2s terms. A step is taken only when 1 + 2(s + 1) ≤ M_max, so the forward pass takes at most ⌊(M_max − 1)/2⌋ steps; when M_max is even, its last slot is never used; and when M_max ≤ 2, the forward pass takes no step. [bb03.4, bb03.7, bb03.9, bb03.10, bb10.1, bb11.5; Notes §13.14] A single hinge using two slots is a quirk; pymars copies it.

**LIMIT-3** `nprune` is None or an integer ≥ 1: the largest size, in terms including the intercept, that the pruning pass may select ([PRUNE-5](#pruning-pass)); with `pmethod="none"` it keeps the first `nprune` forward terms ([PRUNE-7](#pruning-pass)). None means no limit. [bb07.16 to bb07.18, bb15.7; Notes §13.15]

## Spans

In this section a knot search has a parent term b and a covariate x. N is the weight sum of all n cases, and N_b the weight of the active cases of b ([KNOT-1](#candidate-knots)); for the intercept, N_b = N. p is the number of columns of X, and α = 0.05.

**SPAN-1** The automatic minspan (`minspan=None`) is L = max(1, trunc(−log₂(−ln(1 − α)/(p·N_b)) / 2.5)), with N_b as in [W-4](#weights). [F91 eq. 43; bb05.4, bb05.6 (truncated, not rounded), bb06.1 (with the parent's active count, not n)] earth counts the active cases where pymars uses their weight ([W-2](#weights)). The outer max(1, ·) never binds in earth, where p·N_b ≥ 1; it keeps L ≥ 1 for small weights. [pymars]

**SPAN-2** The automatic endspan (`endspan=None`) is E = trunc(3 − log₂(α/p)). It depends only on p. [F91 eq. 45; bb05.5]

**SPAN-3** A user `minspan` or `endspan` is an integer ≥ 1, used in place of L or E. earth reads 0 as automatic, and a negative minspan as a request for at most that many evenly spaced knots; pymars uses None for automatic and does not support a negative minspan ([Departures](#departures-from-earth)). earth raises an error when minspan exceeds n [bb05.9]; pymars accepts such a value, and the scan of [KNOT-3](#candidate-knots) then finds few knots [bb05.8]. [pymars departure: the folds of a cross-validation vary in size]

**SPAN-4** For a parent of degree 1 or more, the endspan becomes E₁ = E + ⌊a·E + 0.5⌋, computed in float64 in this form, where a = `adjust_endspan`, a float ≥ 0 (default 2.0). For the intercept, E₁ = E. [bb06.2 (36 settings), bb06.4, bb06.12 (the form: in the 5 settings where it and ⌊(1 + a)·E + 0.5⌋ differ in float64, earth uses this one)] This is a quirk: earth describes the argument as a factor on the endspan, but the endspan it uses is E·(1 + a) rounded half up, so a = 1 doubles it. pymars copies the observed rule.

**SPAN-5** The endspan used in the scan is E* = max(1, min(E₁, ⌊(N + τ_N)/2⌋ − 1)), with N as in [W-4](#weights). The cap applies to user and automatic values alike, after the adjustment, and N counts all cases, not only the active ones. [bb05.5, bb05.7, bb06.2] earth uses n in place of N ([W-2](#weights)).

**SPAN-6** The minspan is not capped. [bb05.8]

## Candidate knots

This section fixes, for one parent b and one covariate x, the list of knot values at which the forward pass evaluates a hinge. [FWD-3](#forward-pass) says which candidates the forward pass builds from these knots.

**KNOT-1** A case i is active for the parent b when b(x_i) > 0. For a parent with a linear factor, a case where the parent is negative is inactive, although the new column is nonzero there. Every case is active for the intercept. [bb06.5, bb06.7] Inactive negative cases are a quirk; pymars copies it.

**KNOT-2** Sort the n cases by x, ascending. Among cases with equal x, the inactive cases come before the active ones. Cases with equal x and equal activity may come in any order, because the scan does not tell them apart. [pymars departure: earth's order among equal values of x depends on the row order, and a permutation of the rows changes its knots, bb06.8, bb06.9; the fit must not depend on the row order, plan: Behavior target, "Ties and row order"]

**KNOT-3** With unit weights, the scan runs as follows. Let x_(1) ≤ … ≤ x_(n) be the sorted values, a_q ∈ {0, 1} the activity of the case at position q (1-based), L the minspan ([SPAN-1](#spans), [SPAN-3](#spans)), E* the endspan ([SPAN-5](#spans)), and v the largest x among the active cases. A counter starts at c₀ = E* + ⌈g/2⌉, where g = (n − 2E* − 1) mod L, the remainder in [0, L). Then, for q = n, n − 1, …, E* + 2 in turn:

1. let t = x_(q−1);
2. if t ≥ v, go on to the next q, and the counter does not change;
3. otherwise, if a_q = 1, decrease the counter by 1; if it is then 0, append t to the list and set the counter to L.

The list may repeat a value; the candidate knots are its distinct values. With no active case there is no knot. [bb05.1 to bb05.3 (240 settings with the intercept), bb06.3 and bb06.4 (120 searches), bb06.10 (80 searches with ties)]

**KNOT-4** These properties follow from KNOT-3. Tests may check them.

- No knot is at or above v, so the largest active value is never a knot.
- No knot is below x_(E*+1), so the E* lowest cases are never knots. For a parent other than the intercept these cases may be inactive, and then the lowest active values are not protected. This is a quirk [bb06.11]; pymars copies it.
- A knot can be the value of an inactive case, since a knot is the value just below an active case in the full order [bb06.6]. This is a quirk; pymars copies it.
- At the top, only active cases move the counter, so the first knot lies just below the c₀-th active case that counts, from the top.
- For the intercept and distinct x values, the knots are every L-th value from the top of x_(E*+1), …, x_(n−E*); the slack (n − 2E* − 1) mod L is split between the two ends, with the larger half at the top [bb05.2, bb05.3].
- A value can be listed more than once, and each listing moves the counter.

**KNOT-5** The lowest knot, x_(E*+1), equals the smallest value x_(1) only when that value is repeated. Such a knot t gives h = b·(x − t)₊ = b·x − t·b at every case. In a pair search ([LA-7](#linear-algebra-contract)) h lies in the span of b and b·x, so the collinearity test ([LA-3](#linear-algebra-contract)) rejects it [bb05.10]. In a single-hinge search b·x is not among the columns of the test, so the knot can pass, and adding its hinge is then the same as adding b·x [bb14.6]. The linear term b·x, which earth reports as a knot at the minimum, is not a scan candidate: [FWD-3](#forward-pass) evaluates it separately, in pair searches only. [Notes §6.3]

**KNOT-6** With weights, the same scan runs on cumulative weight, so that an integer weight w_i gives the same list as w_i copies of row i. Order the cases as in KNOT-2, and let W_q be the weight of the first q cases ([W-4](#weights); W_0 = 0, W_n = N); the case at position q covers the interval (W_{q−1}, W_q]. For a real u in (0, N], the case that holds u is the smallest q with u ≤ W_q + τ_N, and case n if there is none; let x(u) and a(u) be its value and its activity. The scan visits u = N, N − 1, N − 2, … while u ≥ E* + 2 − τ_N, and at each u it takes steps 1 to 3 of KNOT-3 with t = x(u − 1) and a(u) in place of x_(q−1) and a_q. The start is c₀ = E* + ⌈g/2⌉ with g = D − L·⌊D/L⌋ and D = N − 2E* − 1. N_b in [SPAN-1](#spans) is the weight of the active cases. With unit weights this is KNOT-3. [pymars departure: frequency weights, plan: Behavior target and Sample weights; earth ignores weights in the spans and knots, Notes §2.10]

**KNOT-7** The fast code sorts each column of X once, in O(p·n·log n), and builds the list for one parent and one covariate in O(n), also with weights: the activity is fixed within one case's interval, so the steps inside it can be counted at once. [plan: Fast path]

## Linear algebra contract

This section states results, not methods. ⟨u, v⟩_w = Σ_i w_i u_i v_i is the weighted inner product, and ‖u‖_w² = ⟨u, u⟩_w.

**LA-1** For a set U of columns of B, RSS(U) = Σ_k ‖Y_k − P_U Y_k‖_w², where P_U is the w-orthogonal projection onto the span of the columns in U. RSS(U) is defined also when these columns are linearly dependent, since dependent columns do not change the span. Every RSS in this spec, in the forward pass, in the pruning pass and in the records, is of this kind and on the original scale of Y. [least squares; bb09.1]

**LA-2** The RSS of a candidate depends on the kind of its search ([LA-7](#linear-algebra-contract)). With B the current columns, b the parent, x the covariate and h = b·(x − t)₊ the hinge of knot t: in a pair search, the knot t has RSS(B ∪ {b·x, h}) and the linear candidate has RSS(B ∪ {b·x}); in a single-hinge search, the knot t has RSS(B ∪ {h}), and there is no linear candidate. The RSS reduction of a candidate is the RSS before the step minus its RSS; a new column in the span of the current columns reduces the RSS by 0. [LA-1; bb14.5 (in a single-hinge search earth's traced RSS is RSS(B ∪ {h}), not RSS(B ∪ {b·x, h}))]

**LA-3** A knot is rejected when the current columns nearly explain its hinge column. Let h = b·(x − t)₊ be the hinge column of a candidate knot t on parent b and covariate x. Let G be the current columns B in a single-hinge search, and B together with b·x in a pair search ([LA-7](#linear-algebra-contract)). The ratio

ρ(t) = ‖h − P_G h‖_w² / ‖h − h̄‖_w²,

with h̄ the weighted mean of h, is 1 − R² of h regressed on G; G contains the intercept. The knot is rejected when h is constant (Conventions: all its values over the n cases are equal) or when ρ(t) < τ, and a rejected knot is not a candidate. The tolerance is τ = 0.01 while the forward pass has taken at most 6 steps, so in steps 1 to 7, and τ = 1e-5 from step 8 on. [bb11.1 (at the first step the rejected ratios reach 0.009964 and the accepted ones start at 0.010021), bb11.3 to bb11.6 (the change after 7 steps, for pairs and single hinges), bb11.7 (hinge parents at degree 2, with G by earth's search kind); bb11.2 (a rejected knot keeps the RSS of b·x alone); plan: F5] earth changes τ when an internal slot counter, 2s + 2 for a single hinge and 2s + 3 for a pair after s steps, reaches 16 [bb11.5, bb11.6]; in steps this is the rule above. This is a quirk; pymars copies it. With weights, earth may run a different code path for this test (Notes §2.10 and earth's help on `Force.weights`); pymars applies the test with the weighted inner product, so that integer weights match repeated rows. [pymars departure for weighted fits]

**LA-4** Where the spec asks for least-squares coefficients on columns that may be linearly dependent ([PRUNE-8](#pruning-pass)), take the columns in their order. A column is dependent when the w-norm of its part orthogonal to the kept earlier columns is less than 1e-7 times its w-norm, not centered. A dependent column gets the coefficient 0, and the other columns get the least-squares solution on the kept columns. This is what R's `lm.fit` and `lm.wfit` do with their default tolerance, where the dependent coefficient is NA. [bb09.7, bb09.8, bb09.10 to bb09.14; bb09.9: the same tolerance on the centered norm does not fit]

**LA-5** The reference and the fast code compute each RSS of LA-1 and LA-2 to within a relative error of 1e-8 of the RSS before the step, each ratio ρ(t) of LA-3 to within an absolute error of 1e-7·τ, and each A_w of LA-7 to within a relative error of 1e-7 of 0.01·Π σ_v². They make the same choices except at near-ties, as the plan's tolerance table and its section "Ties" define. A knot with |ρ(t) − τ| ≤ 1e-6·τ (LA-3), or a search with |A_w − 0.01·Π σ_v²| ≤ 1e-6·(0.01·Π σ_v²) (LA-7), is a near-tie: the two implementations may decide it differently, and the oracle and conformance tests treat the step like a near-tie between candidates. [plan: What is compared, and the tolerances; Ties]

**LA-6** An implementation may scale each response (subtract a constant, divide by a positive constant) inside the forward pass for numerical stability. The scaling does not change the rankings, the collinearity test or the stopping rules beyond rounding, and every value in the records is reported on the original scale. [earth scales a single response in its forward pass, earth's help on `Scale.y`; [FWD-10](#forward-pass) states pymars's rule]

**LA-7** Each search for a parent and a covariate is either a pair search or a single-hinge search. For a parent b and a covariate x, let A be the residual sum of squares of the column b·x regressed on the current columns B (the intercept is among them). earth, in unweighted fits, runs a pair search when A ≥ 0.01 and a single-hinge search when A < 0.01, where A is computed with x in its own units; 1 − R² of b·x plays no part [bb14.1 (1966 searches; the largest A of a single-hinge search is 0.0095 and the smallest A of a pair search 0.0107), bb14.2, bb14.3 (rescaling x2 moves A from 0.00990 to 0.01010 and the search from single-hinge to pair)]. So earth's fit depends on the units of the covariates [bb14.4], and the plan's edge-case row that calls earth invariant to scale and shift (S12) is corrected by this rule. With weights that are not all equal, earth runs pair searches also below A = 0.01 [bb14.7].

pymars departure: pymars runs a pair search when A_w ≥ 0.01·Π_{v∈V} σ_v², where A_w is the weighted residual sum of squares of b·x on B, V is the set of covariates of the term b·x (those of the parent and x), and σ_v² is the weighted variance of covariate v with divisor N (Notation); it runs a single-hinge search otherwise, and always when some v ∈ V is constant over the n cases (Conventions), where σ_v² = 0 by definition. This rule does not change when a covariate is multiplied by a positive constant, nor, for a parent without a linear factor, when a constant is added to a covariate. It is earth's rule when every covariate has σ_v² = 1. [pymars departure: the fit must not depend on the units of x, Conventions and plan: Invariance tests]

For the matched and earth-compatible comparisons, the harness (T02, T05, T07) divides each non-constant covariate by its σ_v (weighted, divisor N over the cases of the fit after zero-weight rows are dropped; for integer weights compared with repeated rows, from the repeated rows), does not center it, leaves constant covariates as they are, and gives the same scaled matrix to earth and to pymars, so that their knots compare directly. Centering would change the fit whenever a covariate is a linear factor in a term of degree 2 or more (Conventions). On raw data pymars and earth choose the kind of a search differently more often than on scaled data, so comparisons at the two programs' defaults on raw data differ more in structure.

## Pruning pass

**PRUNE-1** `pmethod` is `"backward"` (the default) or `"none"`. earth's other methods are not supported ([Departures](#departures-from-earth)).

**PRUNE-2** The input is the M_f forward terms, numbered 0 to M_f − 1 in the order the forward pass added them (term 0 is the intercept), with Y, w, d and N. For each size m = 1, …, M_f the pass keeps the lowest RSS found so far, R[m], and its set of terms, T[m]; at the start R[m] = +∞. [bb07.8; F91 Algorithm 3]

**PRUNE-3** The pass depends on the number K of response columns.

With one response (K = 1), a working order o = (o_1, …, o_{M_f}) starts as (0, 1, …, M_f − 1). To offer the order: for m = 1, …, M_f, let U = {o_1, …, o_m}, and if RSS(U) < R[m], set R[m] = RSS(U) and T[m] = U. Offer the starting order. Then for pos = M_f, M_f − 1, …, 2 in turn:

1. among the positions i = 2, …, pos, choose the one for which RSS({o_1, …, o_pos} without o_i) is lowest; if several are equal, choose the one whose term has the largest index (the term added last);
2. record o_i as the term removed at this stage, and move it to position pos, so that o_{i+1}, …, o_pos move one place to the left;
3. offer the new order.

Position 1 always holds the intercept, so the intercept is never removed, and T[1] = {0}. The sets T[m] need not be nested. [bb07.12, bb07.24 (130 random bases and 16 ordinary fits), bb07.2 to bb07.4, bb07.13, bb15.4; F91 Algorithm 3 for plain backward deletion; earth uses code from the leaps package here, Notes §2.1 and §2.5]

With two or more responses (K ≥ 2), the pass is plain backward elimination: start from all M_f terms; at each stage remove the term, other than the intercept, whose removal gives the lowest RSS (summed over the responses and weighted, [LA-1](#linear-algebra-contract)), with the same tie rule, until the intercept alone is left. T[m] is the set left after M_f − m removals, so the sets are nested, and `removed` is the sequence of removals. [bb15.1, bb15.2, bb15.5 (36 fits with 2 or 3 responses, with and without weights; bb15.3: the one-response rule fails in all 11 fits where the two rules differ), bb15.6 (earth names a different subset routine for several responses)] The classifier's binary refit runs its passes on one 0/1 response (K = 1), and its multiclass refit on one indicator per class (K ≥ 3); earth treats a 3-level factor response as several responses [bb15.6].

With M_f ≤ 2 the two rules give the same result. Using two algorithms by K is a quirk of earth that pymars copies. In earth, rounding decides an exact tie in step 1 for one response and in a removal for several: the removed term follows the data values, not term numbers or column positions [bb08.9, bb08.10 for K = 1; bb15.9, bb15.10 for K = 2]; the pymars tie rule is a choice (a departure for exact ties only).

**PRUNE-4** The records are `removed`, the terms removed at the stages pos = M_f, …, 2 in that order (for K ≥ 2, the removals in order); `rss_per_size[m − 1]` = R[m] and `gcv_per_size[m − 1]` = GCV(R[m], m) by [GCV-2](#gcv-and-fit-statistics), for m = 1, …, M_f; and the sets T[m]. [bb07.8, bb07.9, bb15.5]

**PRUNE-5** With `pmethod="backward"`: let m_max = M_f, or min(M_f, `nprune`) when `nprune` is given. The selected size m* is the smallest m in 1, …, m_max at which `gcv_per_size[m − 1]` is lowest, and the selected terms are T[m*], in increasing order. [bb07.10, bb07.17, bb08.13 (tied minima go to the smaller size)]

**PRUNE-6** `nprune` changes only the range in PRUNE-5; the stages and the records are the same as without it. With d < 0, so that GCV = RSS/N, and `nprune` = k, earth selected exactly k terms in bb07.19, as expected when R[m] falls as m grows. [bb07.16 to bb07.19]

**PRUNE-7** With `pmethod="none"`, the stages and the records are computed as for `"backward"`. The selected size is m* = min(`nprune`, M_f), or M_f when `nprune` is None, and the selected terms are the first m* forward terms, 0 to m* − 1. [bb07.14, bb07.15, bb15.7] earth selects the same terms, but for m* < M_f it reports the statistics of T[m*] (`rss.per.subset[m*]`) with the coefficients of its selected terms, so its statistics do not describe the model it returns [bb15.8]. This is a quirk; pymars departs and reports the statistics of the selected terms ([PRUNE-8](#pruning-pass)), because the fitted attributes must describe the returned model.

**PRUNE-8** The final coefficients are the weighted least-squares coefficients of Y on the selected columns, in increasing term order and on the original scale of Y, with [LA-4](#linear-algebra-contract) for dependent columns; `coef` has shape (m*, K). The final `rss` is RSS(selected terms) ([LA-1](#linear-algebra-contract)), and `gcv`, `rsq` and `grsq` follow [GCV-2](#gcv-and-fit-statistics) to [GCV-7](#gcv-and-fit-statistics) with that RSS and M = m*. With `pmethod="backward"`, and with `pmethod="none"` when m* = M_f, the selected terms are T[m*], so `rss` equals `rss_per_size[m* − 1]` up to rounding; with `pmethod="none"` and m* < M_f the two differ in general ([PRUNE-7](#pruning-pass)). [bb09.1, bb09.1b, bb09.3; Notes §2.5]

**PRUNE-9** The fast code updates or downdates one factorization instead of refitting each subset; a stage may cost O(n·M_f²). The reference may refit every subset. [plan: Fast path]

## Core API

**CORE-1** `pymars._core.fit_mars(X, Y, w, params, *, record_candidates=False)` returns a `MarsFit`. [plan: Core API]

- X is a float64 array of shape (n, p), finite.
- Y is a float64 array of shape (n, K), with K ≥ 1, finite.
- w is a float64 array of shape (n,), finite and ≥ 0 with a positive sum, or None, which means w_i = 1 exactly.
- params is a `MarsParams`.

The function first drops the rows with zero weight ([W-3](#weights)), and it never changes its inputs. The estimators check the data before the call ([Errors](#errors)), so the core assumes valid input. The function is pure: the same inputs give the same output, and the output does not depend on the order of the rows, apart from near-ties.

**CORE-2** `MarsParams` is a frozen dataclass with the fields below. Its constructor raises ValueError, with a message that names the field, for a value outside the range. `MarsFit` holds the resolved values of `max_terms` and `penalty`; the spans are resolved for each knot search ([Spans](#spans)), and `nprune=None` means no limit. [plan: Core API, Public API]

| Field | Type | Default | Allowed values | None means |
|---|---|---|---|---|
| `max_degree` | int | 1 | ≥ 1 | |
| `max_terms` | int or None | None | ≥ 1 | [LIMIT-1](#term-limit) |
| `penalty` | float or None | None | −1, or finite and ≥ 0 ([GCV-1](#gcv-and-fit-statistics)); earth rejects values above 1000 (bb01.6), so the harness sends none | [GCV-4](#gcv-and-fit-statistics) |
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

**CORE-3** `MarsFit` is a frozen dataclass. M is the number of selected terms, M_f the number of forward terms and S the number of forward steps. [plan: Core API]

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

**CORE-4** `Termination` is an IntEnum. It uses earth's code where earth has one [bb10.1 to bb10.6]. The [stopping rules](#stopping-rules) state when each code applies and in which order the rules are checked.

| Value | Name | The forward pass stopped because | earth |
|---|---|---|---|
| 0 | `DEGENERATE` | it did not run: N ≤ 1 or every response constant ([GCV-7](#gcv-and-fit-statistics)) | for a constant y, 2 at n = 3 and 4 at n = 12 and 100 (bb10.7); pymars departure |
| 1 | `NO_ROOM` | M_max ≤ 2, so no step fits ([LIMIT-2](#term-limit)) | 1 |
| 2 | `GRSQ_NEG_INF` | GRSq is −∞ | 2 |
| 3 | `GRSQ_LOW` | GRSq is below −10 | 3 |
| 4 | `RSQ_CHANGE_SMALL` | RSq changed by less than `thresh` | 4 |
| 5 | `RSQ_HIGH` | RSq reached 1 − `thresh` | 5 |
| 6 | `NO_GAIN` | no candidate lowers the RSS | 6 |
| 7 | `TERM_LIMIT` | the term limit was reached after at least one step | 7 |

**CORE-5** The reference implementation returns the same fields as a dict. The keys are the field names above; `forward`, `pruning` and `candidates` are nested dicts; `termination` is the integer code; the arrays have the dtypes and shapes above. `MarsFit.from_dict(d)` builds a `MarsFit` from such a dict, and `MarsFit.to_dict()` gives it back, so that tests can compare the two. [plan: Core API]

**CORE-6** The estimators call the core through its module: `pymars/_estimators.py` does `from pymars import _core` and calls `_core.fit_mars(...)` during `fit`, never a name bound at import. A test puts the reference in its place with `monkeypatch.setattr(pymars._core, "fit_mars", adapter)`, where `adapter` calls the reference and returns `MarsFit.from_dict(...)`. [plan: Core API]

**CORE-7** Targets for the fast code: a forward step costs O(p·n·r) for each parent it scores, with r the rank of the current basis ([plan: Cost model](../VALIDATION_PLAN.md#cost-model)); the pruning pass meets [PRUNE-9](#pruning-pass); memory is O(n·(p + M_max)). The reference has no targets, but it must handle n ≤ 300, p ≤ 6 and degree ≤ 3.

## Forward pass

The forward pass adds terms one step at a time. Step s + 1 follows s completed steps; its term number is K = 2(s + 1), and its terms go into the slots K and K + 1 ([FWD-9](#forward-pass)). RSS_s is the RSS after s steps, and Δ_s = RSS_{s−1} − RSS_s the reduction that step s achieved.

**FWD-1** The pass starts from the intercept alone: B holds the column of ones, RSS_0 = TSS, and s = 0. A degenerate fit does not run the pass ([EDGE-1](#degenerate-inputs)). [F91 Algorithm 2]

**FWD-2** Each step first applies [STOP-1](#stopping-rules). It then searches the parents that the queue gives ([FAST-3](#fast-mars), [FAST-4](#fast-mars)). For a parent b, it searches every covariate j = 0, …, p − 1 in increasing order, except the covariates of b (a covariate appears at most once in a term, [TERM-4](#terms)). A parent is searched only when its degree is below `max_degree`. [F91 Algorithm 2; bb16.13 (the degree test)]

**FWD-3** The search for a parent b and a covariate x is a pair search or a single-hinge search by [LA-7](#linear-algebra-contract). A pair search has the linear candidate b·x and one candidate for each knot of [KNOT-3](#candidate-knots) (or [KNOT-6](#candidate-knots)) that [LA-3](#linear-algebra-contract) does not reject. A single-hinge search has only the knot candidates. The RSS of each candidate is that of [LA-2](#linear-algebra-contract). The linear candidate is evaluated whatever `auto_linpreds` is; `auto_linpreds` only sets the term that it adds ([FWD-6](#forward-pass)). [bb12.3, bb13.4, bb13.5, bb14.5]

**FWD-4** A candidate is legal when its RSS reduction is positive and at most MaxLegal_s = min(1.01·RSS_s, 10·Δ_s); at the first step MaxLegal_0 = 1.01·RSS_0. Of the legal candidates of the step, the one with the largest RSS reduction is chosen. With several responses the reduction is summed over them ([RESP-1](#several-responses)). The GCV penalty plays no part. [bb12.1 (the forward pass is the same for penalties −1, 0, 2, 3 and 10), bb12.2, bb12.3, bb12.4 (the limit 10·Δ_s changes earth's choice), bb20.1; Notes §3] The limit 10·Δ_s is a quirk: a step may not reduce the RSS by more than 10 times what the previous step did. pymars copies it. The test on linear candidates is not observed in earth (OQ-5); pymars applies it to them too.

**FWD-5** Equal RSS reductions go to the candidate found first in this order: the parents in the order of the queue table ([FAST-2](#fast-mars)), the covariates in increasing index, within one search the linear candidate before the knots, and the knots from the largest down. So a tie between covariates goes to the lower index, and a tie between knots to the larger knot. earth decides exact ties the same way when the columns are bitwise equal [bb18.1], and otherwise by rounding [bb18.4]; the pymars order is fixed. [plan: Behavior target, "Ties and row order"; pymars departure for ties that earth decides by rounding]

**FWD-6** The chosen candidate adds these terms, in this order ([TERM-1](#terms)):

- a knot t of a pair search: b·(x − t)₊ (code +1), then b·(t − x)₊ (code −1), with the knot t [bb13.1];
- a knot t of a single-hinge search: b·(x − t)₊ (code +1) [bb13.3];
- the linear candidate: with `auto_linpreds=True`, b·x (code 2); with `auto_linpreds=False`, b·(x − m)₊ (code +1, knot m), where m is the smallest value of x over all n cases, not only the active ones [bb13.4, bb13.5, bb13.7; bb13.6 refutes the active cases; Notes §6.3].

For b·x and b·(x − m)₊ the fitted values and the RSS are the same; they differ outside the range of the training data.

**FWD-7** Each step counts two toward the term limit, whatever it adds ([LIMIT-2](#term-limit)).

**FWD-8** The forward pass records, for each step s, RSS_s in `forward.rss[s]`, and for each new term its parent and step ([TERM-6](#terms)). With `record_candidates=True` it records the candidate log of [CORE-3](#core-api). The second-best candidate of a step is the legal candidate with the largest RSS reduction among those that differ from the chosen one in the parent, the covariate, the kind or the knot value.

**FWD-9** Slots. Step s + 1 fills slot K with its first term and slot K + 1 with its second term, if any; a step that adds one term leaves slot K + 1 empty. Slot 1 holds the intercept. The queue of [Fast MARS](#fast-mars) addresses parents by slot. [bb16.13, bb16.14; bb11.5]

**FWD-10** Scaling. Where an implementation scales Y inside the pass ([LA-6](#linear-algebra-contract)), it may, with one response, subtract the weighted mean and divide by any positive constant; with several responses it may subtract each response's weighted mean and divide all responses by one common constant. It must not scale the responses by different constants, because the summed RSS weighs each response by its own units. [bb17.1, bb17.2 (earth scales one response to unit standard deviation), bb17.3a, bb17.4b (with several responses earth does not scale by default, and multiplying one response by 1000 changes its terms)] earth's `Scale.y = TRUE`, which scales each response to unit variance, is not offered.

## Stopping rules

The rules apply in this order at each step. RSq_s = 1 − RSS_s/TSS, and for the step's chosen candidate RSq′ and GRSq′ are those of the model that would result ([GCV-5](#gcv-and-fit-statistics), [GCV-6](#gcv-and-fit-statistics)), with M′ its real number of terms.

**STOP-1** Before the search: if 1 + 2(s + 1) > M_max, the pass stops, with code `NO_ROOM` (1) when s = 0 and `TERM_LIMIT` (7) otherwise. [LIMIT-2; bb10.1, bb19.2, bb19.7]

**STOP-2** After the search: if the step has no legal candidate, for example because no parent could be searched, the pass stops with code `NO_GAIN` (6). [bb19.9, bb16.23]

**STOP-3** If `thresh` > 0 and GRSq′ < −10, the candidate is not added, and the pass stops with code `GRSQ_NEG_INF` (2) when GRSq′ = −∞ and `GRSQ_LOW` (3) otherwise. GRSq′ uses the real number of terms M′ = M + 1 or M + 2, not the counted slots. [bb19.1, bb19.4, bb19.5, bb19.6, bb10.4 (the rule does not act when thresh = 0)]

**STOP-4** If RSq′ − RSq_s < `thresh`, the candidate is not added, and the pass stops with code `RSQ_CHANGE_SMALL` (4). [bb19.1, bb19.5 (STOP-3 comes first), bb19.8 (also at the last step that the term limit allows)]

**STOP-5** Otherwise the candidate is added (s grows by 1). If then RSq_s ≥ 1 − `thresh`, the pass stops with code `RSQ_HIGH` (5). This rule acts also when `thresh` = 0, for an exact fit. [bb19.2, bb19.7 (it comes before the term limit), bb10.6]

**STOP-6** With several responses, RSq and GRSq are the pooled values of [GCV-5](#gcv-and-fit-statistics) and [GCV-6](#gcv-and-fit-statistics). [bb20.4]

## Fast MARS

The queue decides which parents a step searches. It follows Friedman (1993) with earth's details. [F93 §3.0, §3.1]

**FAST-1** The queue has one entry for each term: entries 1 to T, where T is the number of terms after the last step (entry 1 is the intercept). Entry c holds a value R_c and a term number K_c. When a step adds a terms, it appends the entries T + 1 to T + a with R = +∞ and K_c = K, the step's term number. Entries are never removed. [bb16.1, bb16.17]

**FAST-2** Before a step, with K_prev the term number of the step just done: rank_c is the 0-based place of entry c when the entries are sorted by R_c, largest first, equal values in increasing entry number; AgedRank_c = rank_c + `fast_beta`·(K_prev − K_c); the table is the entries sorted by AgedRank, smallest first, equal values in increasing rank. Since K grows by 2 per step, `fast_beta` = 1 adds 2 per step that an entry waits. [bb16.2, bb16.4, bb16.7, bb16.9; bb16.3, bb16.5, bb16.6 and bb16.8 refute other orders]

**FAST-3** The step visits the first k rows of the table, where k = max(3, `fast_k`) when `fast_k` ≥ 1, and every row when `fast_k` = 0 or k ≥ T. So `fast_k` = 1 or 2 acts as 3; this is a quirk that pymars copies. [bb16.10, bb16.11, bb16.16]

**FAST-4** A visited entry c stands for slot c ([FWD-9](#forward-pass)), not for the c-th term. It is skipped when slot c is empty or when its term's degree equals `max_degree`; a skipped entry keeps its row among the k visited ones. Otherwise the term in slot c is searched as a parent. After a step that adds one term, the slots run ahead of T, so the newest terms, in slots above T, are not searched until T reaches their slot; this holds also with `fast_k` = 0. [bb16.13; bb16.12 refutes the reading by term, and bb16.15 refutes that `fast.k = 0` searches every eligible term: in 121 cases it did not] This is a quirk. pymars copies it, so that its fits match earth's; OQ-4 records the choice.

**FAST-5** Each searched entry gets K_c = K and R_c = the largest legal RSS reduction ([FWD-4](#forward-pass)) among its candidates in the step, or 0 when it has none, or −1 when no covariate could be searched for it. The chosen parent is updated too. Entries that were not visited or were skipped keep their values. [bb16.17, bb16.18, bb16.19, bb16.24] Only the order of the values R matters, so the scale of Y does not.

**FAST-6** At degree 1 only the intercept is ever searched. The defaults are `fast_k` = 20 and `fast_beta` = 1.0. [bb16.22, bb16.26]

## Weights

Weights are case weights, given as `sample_weight` to `fit`.

**W-1** Weights are frequency weights: an integer weight w_i gives the same fit as w_i copies of row i, and weights that are all 1 give the same fit as no weights. No weights means w_i = 1 exactly. [plan: Behavior target, "Sample weights"]

**W-2** N = Σ w_i ([W-4](#weights)) takes the place of the number of cases everywhere: in the GCV and its rule C ≥ N ([GCV-2](#gcv-and-fit-statistics)), in GRSq, in the spans ([SPAN-1](#spans), [SPAN-5](#spans)) and in the knot scan ([KNOT-6](#candidate-knots)); and every sum of squares is weighted. [pymars departure: earth uses the number of cases in the GCV and ignores weights in the spans and knots, bb02.12, Notes §2.10; it also ignores weights that are all equal, bb02.15]

**W-3** Rows with zero weight are dropped before anything else, so they count neither in n and N nor in the spans and the knot scan. [pymars departure: earth replaces a zero weight by a very small positive one, Notes §2.10, so such rows still count as cases in its spans and knots]

**W-4** Sums of weights are formed in float64 as follows. N = `math.fsum` of the weights of the n cases, and N_b = `math.fsum` of the weights of the active cases; these sums are exactly rounded, so they do not depend on the order of the cases. W_q is the running sum of the weights in the order of [KNOT-2](#candidate-knots), for example by `numpy.cumsum`; any method whose error stays below τ_N/10 is allowed. τ_N = 1e-8·N. Where N or N_b is within τ_N of an integer, that integer replaces it in every rule. Comparisons of sums with integers or grid points use τ_N as [GCV-2](#gcv-and-fit-statistics), [SPAN-5](#spans) and [KNOT-6](#candidate-knots) state. With integer weights every sum is exact, and τ_N changes nothing in the spans and the knots (GCV-2 notes its one effect there); and 100 cases of weight 0.1 (10 at each of 10 values) give the same spans and knots as 10 cases of weight 1. [pymars: two implementations must get the same knots for the same weights]

**W-5** No weights means w_i = 1 exactly, and then every sum of weights is exact, so the fit equals the fit with weights all 1. [plan: Behavior target]

**W-6** Weights must be a 1-D array of length n, finite and ≥ 0; otherwise `fit` raises ValueError. If every weight is 0, `fit` raises ValueError with a message that matches the regular expression `weight.*zero`. [plan: Behavior target, "Sample weights"; scikit-learn's weight checks]

**W-7** `fit` issues a UserWarning when some weight is not an integer and the mean of the given weights differs from 1 by more than 1e-6. The message says that the weights act as case counts, and that importance weights should be rescaled to w·n/Σw. Integer weights, zeros included, never warn. [plan: Behavior target, "Sample weights"]

**W-8** Every rule uses the weights as the sections above state: the weighted sums of squares ([GCV-8](#gcv-and-fit-statistics)), N and the tolerance τ_N ([W-2](#weights), [W-4](#weights)), the spans and the knots ([SPAN-1](#spans), [SPAN-5](#spans), [KNOT-6](#candidate-knots)), the collinearity test and the kind of a search ([LA-3](#linear-algebra-contract), [LA-7](#linear-algebra-contract)), and the weighted least squares of the final coefficients ([PRUNE-8](#pruning-pass)). Multiplying all weights by c changes the fit as c copies of the data would. [plan: Sample weights]

## Several responses

**RESP-1** With Y of shape (n, K), K ≥ 2, the responses share one basis. Every RSS and TSS is summed over the responses ([LA-1](#linear-algebra-contract), [GCV-3](#gcv-and-fit-statistics)); the forward pass chooses the candidate with the largest summed reduction; the pruning pass follows the K ≥ 2 branch of [PRUNE-3](#pruning-pass); the final coefficients are one column per response ([PRUNE-8](#pruning-pass)). [bb20.1, bb20.2, bb15.1; Notes §2.9]

**RESP-2** A 2-D Y with one column is K = 1 and gives the same fit as the 1-D y; the regressor's predictions keep the shape of y ([API-4](#public-api)). [bb20.3, bb02.29]

**RESP-3** The responses are not scaled to a common variance ([FWD-10](#forward-pass)): a response in larger units weighs more in the criterion, as in earth's default. [bb17.3a, bb17.4b]

## GLM refit for the classifier

`EarthClassifier` fits the terms by the least-squares passes and then refits a logistic model on the selected basis. `classes_` are the sorted unique labels of y.

**GLM-1** With two classes, the passes run on one 0/1 response, 1 for `classes_[1]` (K = 1). With C ≥ 3 classes, they run on C indicator responses, one per class in the order of `classes_` (K = C). [plan: Behavior target, "Multiclass outcomes"; bb21.4, bb15.6]

**GLM-2** The refit uses the selected columns B_S. With two classes it is a binomial GLM with the logit link; with C ≥ 3 classes it is one multinomial logistic model with a coefficient vector per class. The coefficients minimize Σ_i w_i·ℓ_i + (`glm_alpha`/2)·Σ_j γ_j², where ℓ_i is the negative log-likelihood of case i, and γ_j are the coefficients of the non-intercept columns after each is standardized to weighted mean 0 and weighted standard deviation 1 (for the multinomial model, the sum runs over all classes). The intercept is not penalized. With `glm_alpha` = 0 (the default) the fit is unpenalized. pymars departure for C ≥ 3: earth fits one binomial GLM per class, whose probabilities need not sum to 1. [plan: Behavior target, "GLM penalty and separation"]

**GLM-3** The solver must reach the minimum of GLM-2: with `glm_alpha` = 0 and two classes, its coefficients agree with R's `glm` on the same columns to a relative 1e-5 where `glm` converges without a warning. A column of B_S that is linearly dependent on earlier ones ([LA-4](#linear-algebra-contract)) gets coefficient 0. [plan: Binary outcomes]

**GLM-4** When the solver does not converge, or when some fitted probability is within 10·ε of 0 or of 1 (ε = 2.2e-16, as in R's `glm`), `fit` issues a sklearn ConvergenceWarning whose message suggests a positive `glm_alpha`, and keeps the last iterate. [plan: Behavior target; earth warns in the same case, bb21.3]

**GLM-5** If B_S is the intercept alone, the refit is not run: the probabilities are the weighted class frequencies. [plan: Behavior target]

**GLM-6** `predict_proba` returns, with two classes, the columns 1 − p and p, and with C ≥ 3 classes the softmax of the C linear predictors; each row sums to 1. `decision_function` returns the linear predictor of `classes_[1]` with two classes and the C linear predictors otherwise. `predict` returns the class of the largest probability, the first such class on a tie. [plan: Behavior target, scikit-learn's classifier contract]

**GLM-7** If fewer than 2 classes have positive weight, `fit` raises ValueError with a message that contains "class". [plan: Behavior target, "Degenerate inputs"]

## Degenerate inputs

**EDGE-1** A fit is degenerate when N ≤ 1, or when TSS = 0 (every response constant over the cases with positive weight). The forward pass does not run: the model is the intercept alone, with coefficient the weighted mean of each response, `gcv` = +∞, `rsq` = `grsq` = 0, termination `DEGENERATE` (0), M_f = 1 and the trivial pruning record ([CORE-3](#core-api), [GCV-7](#gcv-and-fit-statistics)). A single case gives such a fit, not an error. [plan: Behavior target, "Degenerate inputs"; pymars departure: earth stops with an error for n = 1 and returns gcv = 0 with code 4 for a constant y, bb22.1, bb22.2a]

**EDGE-2** Otherwise there is no special case for small n: the rules above decide. In earth, 2 to 5 cases give the intercept alone, because every candidate has GRSq = −∞ ([STOP-3](#stopping-rules)); 8 and 12 cases add terms. [bb22.1]

**EDGE-3** A constant covariate never enters a term: its knots are all at or above its largest value, and its linear column lies in the span of the parent. [KNOT-4, LA-7; bb22.3a]

**EDGE-4** An exact duplicate of a covariate is never used when the original has the lower index ([FWD-5](#forward-pass)); a near-duplicate is a different covariate, and the collinearity test ([LA-3](#linear-algebra-contract)) applies to it. [bb22.3b, bb18.1, bb22.3d]

**EDGE-5** n < p needs no special case, and p enters only the spans and the default term limit, which is at most 201. [bb22.4, LIMIT-1]

## Errors

**ERR-1** A NaN or an infinite value in X, y or `sample_weight` raises ValueError from scikit-learn's `validate_data` (or from the weight check of [W-6](#weights)). [plan: Behavior target, "NaN and infinite values"; earth also stops with an error, bb22.6a]

**ERR-2** A column of X that is not numeric, such as strings in a DataFrame, raises ValueError whose message names OneHotEncoder and the recipe `make_column_transformer((OneHotEncoder(drop="first"), cols), remainder="passthrough")`. [plan: Behavior target, "Categorical inputs"]

**ERR-3** `allow_missing=True` raises NotImplementedError in `fit`, with a link to the issue https://github.com/alejandroschuler/mars/issues/27. [plan: Q4, Missing values]

**ERR-4** A parameter outside its range raises ValueError in `fit`, from `MarsParams` ([CORE-2](#core-api)); `__init__` only stores the parameters. `predict`, `predict_proba`, `decision_function`, `basis_matrix` and `summary` before `fit` raise NotFittedError, and X with a different number of columns than in `fit` raises ValueError. [plan: scikit-learn compatibility]

## Public API

**API-1** `EarthRegressor(max_degree=1, max_terms=None, penalty=None, thresh=0.001, minspan=None, endspan=None, adjust_endspan=2.0, auto_linpreds=True, fast_k=20, fast_beta=1.0, pmethod="backward", nprune=None, allow_missing=False)`. The parameters map one to one to the fields of `MarsParams` ([CORE-2](#core-api)), and `None` means earth's automatic value. [plan: Public API]

**API-2** `EarthClassifier` takes the same parameters and `glm_alpha=0.0`, a float ≥ 0 ([GLM-2](#glm-refit-for-the-classifier)). `Earth` is the same class object as `EarthRegressor`, so `import pymars as earth; earth.Earth()` works. [plan: Public API]

**API-3** Fitted attributes: `n_features_in_`; `feature_names_in_` (only after a fit on a table with string column names); `dirs_`, `cuts_`, `gcv_`, `rss_`, `rsq_`, `grsq_` (from `MarsFit`); `max_terms_` and `penalty_`, the resolved values; `mars_`, the whole `MarsFit`; `term_coef_`, of shape (M,) for a 1-D y and (M, K) for a 2-D y. The classifier also has `classes_` and `glm_`, the refit's coefficients: shape (M,) with two classes and (M, C) otherwise. [plan: Public API]

**API-4** Methods: `fit(X, y, sample_weight=None)` returns self; `predict(X)` returns shape (n,) for a 1-D y and (n, K) for a 2-D y, K = 1 included; the classifier's `predict_proba` and `decision_function` follow [GLM-6](#glm-refit-for-the-classifier); `score` comes from the scikit-learn mixins; `basis_matrix(X)` returns the (n, M) matrix of the selected terms by [TERM-3](#terms); `summary()` returns a string with one line per selected term (its label, [TERM-5](#terms), and its coefficients) and the values of `rss_`, `gcv_`, `rsq_`, `grsq_` and the termination. `sample_weight` also reaches `fit` through metadata routing. [plan: Public API, scikit-learn compatibility]

**API-5** Tags, through `__sklearn_tags__`: the regressor declares `target_tags.multi_output = True`, and both estimators `input_tags.allow_nan = False`. The estimators use scikit-learn 1.6 or later and only its public API. [plan: Behavior target, "Version floor and tags"]

**API-6** Left out on purpose: `coef_`, `transform`, a `max_iter` parameter and `feature_importances_` (issue #33). [plan: Behavior target, "Coefficients and transform"]

**API-7** The names that the harness maps between pymars and earth:

| pymars | earth | Note |
|---|---|---|
| `max_degree` | `degree` | |
| `max_terms` | `nk` | None: earth's default, [LIMIT-1](#term-limit) |
| `penalty` | `penalty` | None: earth's default, [GCV-4](#gcv-and-fit-statistics); −1 or ≥ 0 |
| `thresh` | `thresh` | |
| `minspan` | `minspan` | None: 0 (automatic); earth's negative values are not offered (issue #28) |
| `endspan` | `endspan` | None: 0 (automatic) |
| `adjust_endspan` | `Adjust.endspan` | |
| `auto_linpreds` | `Auto.linpreds` | |
| `fast_k` | `fast.k` | |
| `fast_beta` | `fast.beta` | |
| `pmethod` | `pmethod` | `"backward"` or `"none"`; other values are not offered (issue #32) |
| `nprune` | `nprune` | None: NULL |
| `sample_weight` | `weights` | the meaning differs ([W-2](#weights)); compare through repeated rows |
| `glm_alpha` = 0 | `glm = list(family = binomial)` | earth has no penalty; C ≥ 3 classes differ ([GLM-2](#glm-refit-for-the-classifier)) |
| `allow_missing` | none | earth has only `na.action = na.fail` |
| none | `Scale.y` | pymars does not scale responses separately ([FWD-10](#forward-pass)) |
| none | `linpreds`, `allowed`, `newvar.penalty`, `nfold` | not offered (issues #30, #31, #29, #32) |

Before it runs earth in the matched and earth-compatible modes, the harness standardizes each covariate to weighted variance 1 ([LA-7](#linear-algebra-contract)), and it passes pymars's `fast_k` = 1 or 2 as earth's value unchanged, which earth also treats as 3 ([FAST-3](#fast-mars)).

## Departures from earth

Each row names the rule where pymars 2.0 does not do what earth 5.3.4 does.

| Rules | earth 5.3.4 | pymars 2.0 | Reason |
|---|---|---|---|
| [GCV-2](#gcv-and-fit-statistics), [W-2](#weights) | the number of cases n in the GCV, also with weights | N = Σw | frequency weights (plan) |
| [GCV-8](#gcv-and-fit-statistics), [W-2](#weights) | weights that are all equal are ignored | used as given, so their scale matters | frequency weights |
| [SPAN-1](#spans), [SPAN-5](#spans), [KNOT-6](#candidate-knots) | weights ignored in the spans and knots | cumulative weight | frequency weights |
| [W-3](#weights) | a zero weight becomes a very small weight | the row is dropped | scikit-learn's weight checks |
| [LA-7](#linear-algebra-contract) | a pair search when A ≥ 0.01, with x in its own units | a pair search when A_w ≥ 0.01·Π σ_v²; equal to earth's rule when every σ_v² = 1, so the harness gives both programs the same matrix, each non-constant covariate divided by σ_v (divisor N, not centered) | the fit must not depend on the units of x |
| [GCV-7](#gcv-and-fit-statistics), [CORE-4](#core-api) | a constant y gives gcv 0, rsq and grsq NaN, and termination code 2 (n = 3) or 4 | the intercept alone, gcv +∞, rsq and grsq 0, and code 0 (`DEGENERATE`) | plan: Behavior target, "Degenerate inputs" |
| [PRUNE-7](#pruning-pass), [PRUNE-8](#pruning-pass) | `pmethod="none"` with `nprune` < M_f reports the statistics of T[nprune] with the coefficients of the first `nprune` terms | the statistics of the selected terms | the fitted attributes must describe the returned model |
| [CORE-2](#core-api) | a penalty above 1000 is an error | any finite penalty ≥ 0 | no reason to reject it; the harness sends only values up to 1000 |
| [KNOT-2](#candidate-knots) | the order among equal x values depends on the row order | inactive cases first, then active ones | the fit must not depend on the row order |
| [PRUNE-3](#pruning-pass) | rounding decides an exact tie | the term added last is removed | a fixed tie rule |
| [LA-3](#linear-algebra-contract) | the weighted code may skip the collinearity test | the test uses the weighted inner product | integer weights must match repeated rows |
| [SPAN-3](#spans) | a minspan above n is an error | accepted | folds of varying size |
| [SPAN-3](#spans) | minspan 0 or endspan 0 means automatic; a negative minspan asks for evenly spaced knots | None means automatic; negative values are not supported (a `later` issue) | scikit-learn style (plan) |
| [PRUNE-1](#pruning-pass) | `pmethod` also `"exhaustive"`, `"forward"`, `"seqrep"`, `"cv"` | `"backward"` and `"none"` only | plan (`later` issues) |
| [FWD-5](#forward-pass) | rounding decides exact ties between candidates, unless their columns are bitwise equal | a fixed order: queue, covariate index, linear candidate, larger knot | a fixed tie rule |
| [GLM-2](#glm-refit-for-the-classifier) | C ≥ 3 classes: one binomial GLM per class, whose probabilities need not sum to 1 | one multinomial refit | `predict_proba` rows must sum to 1 (plan) |
| [EDGE-1](#degenerate-inputs) | one case stops with an error | the intercept alone, gcv +∞ | scikit-learn's one-sample check (plan) |
| [ERR-2](#errors) | factors are expanded to dummies | a non-numeric column is an error that names OneHotEncoder | plan: Behavior target, "Categorical inputs" |
| [TERM-2](#terms) | a linear factor stores the smallest x value in `cuts` | 0.0 | representation only; the basis is the same |
| [TERM-5](#terms) | labels in the order of construction, covariates numbered from 1 | increasing covariate order, numbered from 0 | labels only |

## Quirks

Quirks that pymars copies, so that its fits match earth's:

- a single hinge or a linear term uses two slots of the term limit ([LIMIT-2](#term-limit));
- the adjusted endspan is E·(1 + a), rounded ([SPAN-4](#spans));
- a negative value of a linear parent is inactive ([KNOT-1](#candidate-knots));
- the lower end of the scan counts inactive cases, and a knot can be the value of an inactive case ([KNOT-4](#candidate-knots));
- the collinearity tolerance changes after 7 steps ([LA-3](#linear-algebra-contract));
- the pruning pass uses one algorithm for one response and another for several ([PRUNE-3](#pruning-pass));
- a step may not reduce the RSS by more than 10 times the previous step's reduction ([FWD-4](#forward-pass));
- `fast_k` = 1 or 2 acts as 3 ([FAST-3](#fast-mars));
- the queue addresses parents by slot, so the newest terms can wait several steps before they are searched as parents ([FAST-4](#fast-mars)).

Quirks that pymars does not copy: earth's knots can depend on the row order when equal x values have different activity ([KNOT-2](#candidate-knots)); earth's choice between a pair and a single hinge depends on the units of x ([LA-7](#linear-algebra-contract)); with `pmethod="none"` and `nprune`, earth's statistics belong to another model than its coefficients ([PRUNE-7](#pruning-pass)).

## Open questions

- **OQ-1** ([SPAN-4](#spans)), closed. The float64 form of the adjusted endspan is E + ⌊a·E + 0.5⌋ [bb06.12].
- **OQ-2** ([PRUNE-3](#pruning-pass)). earth decides an exact tie in the pruning pass by rounding [bb08.9, bb08.10, bb15.9, bb15.10]. The conformance tests (T07) should treat an exact or near tie in step 1 of PRUNE-3, or in a removal for K ≥ 2, like a near-tie in the forward pass, and stop the structural comparison there. The comparison RSS(U) < R[m] when an order is offered can flip in the same way between two subsets of one size with nearly equal RSS; T07 treats it the same way. The threshold for a pruning near-tie is for T07 to fix.
- **OQ-3** ([KNOT-6](#candidate-knots), [LA-3](#linear-algebra-contract), [LA-7](#linear-algebra-contract)). For weights that are not integers, pymars's spans, knots, collinearity test and choice between a pair and a single hinge have no earth counterpart, since earth runs a different code path with weights [bb14.7]. The tests check that integer weights give the same fit as repeated rows, and that the reference and the fast code agree for other weights. No experiment can settle these rules against earth.
- **OQ-4** ([FAST-4](#fast-mars)), decided. earth's queue addresses parents by slot, so after a step that adds one term the newest terms are not searched as parents until the number of terms reaches their slot, even with `fast.k = 0` [bb16.13; bb16.15, a refuted HYPOTHESIS line, counts 121 such cases]. pymars copies this so that its fits match earth's. It is plainly accidental, and it can keep a good parent out of the search for several steps. The simulation study (T21, T22) could compare it with a search of all eligible parents; a change would be a spec v2 decision.
- **OQ-5** ([FWD-4](#forward-pass)). No experiment saw a linear candidate whose RSS reduction exceeded the limit 10·Δ_s, so it is not known whether earth applies the limit to linear candidates. pymars applies it to every candidate. An experiment that settles it needs a step whose best candidate is a linear term with a reduction above 10 times the previous one.
- **OQ-6** (the end of the forward pass). earth's notes say that the forward pass removes linearly dependent terms before the pruning pass [Notes §13.14]. Whether and when this happens is under test (bb23); with the collinearity test of [LA-3](#linear-algebra-contract) the basis should rarely be rank-deficient, and [LA-4](#linear-algebra-contract) covers a dependent column in the final solve.
- **OQ-7** ([GLM-3](#glm-refit-for-the-classifier)). The spec fixes the objective of the refit and its agreement with R's `glm`, not the solver or its convergence criterion; T14 picks the solver.

## Sources

- Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics* 19(1), 1-67. Cited as F91.
- Friedman, J. H. (1993). Fast MARS. Technical Report 110, Department of Statistics, Stanford University. Cited as F93.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd edition, section 9.4. Cited as ESL.
- Milborrow, S. (2024). Notes on the earth package. Vignette of the R package earth 5.3.4. Cited as Notes, by section, and only paraphrased.
- The black-box experiments in `validation/blackbox/`, run with R 4.4.3 and earth 5.3.4 ([the README there](../validation/blackbox/README.md) lists them).
