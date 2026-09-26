# bb03_nk.R
#
# Question: what is earth's default term limit (nk), and how do knot pairs (vs single
# hinges) count against it during the forward pass?
#
# Part 1: default nk vs candidate min(200,max(20,2p))+1, over covariate count p; then
#         whether nk is reduced for small n.
# Part 2: with a single covariate and a fixed nk, does the forward pass add a single
#         hinge, a full pair (exceeding nk), or stop, once few slots remain?
# Part 3: a single-hinge scenario (two knots on the same variable, same parent) -- does
#         a step that adds one term count as 1 or as 2 toward the term limit?

cat(R.version.string, "\n")
cat("earth", as.character(packageVersion("earth")), "\n\n")

suppressPackageStartupMessages(library(earth))

trace_step_lines <- function(out) out[!grepl("^(x\\[|y\\[|^$)", out)]
forward_complete_attempted <- function(out) {
  ln <- grep("Forward pass complete", out, value = TRUE)
  if (length(ln) == 0) return(NA_integer_)
  as.integer(sub(".*complete: ([0-9]+) terms?.*", "\\1", ln))
}

## ================= Part 1: default nk =================
cat("---- Part 1: default nk vs min(200,max(20,2p))+1 ----\n")
set.seed(1)
p_grid <- c(1, 5, 9, 10, 11, 50, 99, 100, 101, 150)
nk_match <- logical(0)
for (p in p_grid) {
  xx <- matrix(runif(60 * p), 60, p)
  yy <- xx[, 1] + rnorm(60)
  fit <- earth(xx, yy, thresh = 0.05)
  cand <- min(200, max(20, 2 * p)) + 1
  nk_match[as.character(p)] <- fit$nk == cand
  cat(sprintf("n=60 p=%-4d fit$nk=%-4d candidate=%-4d match=%s\n", p, fit$nk, cand, fit$nk == cand))
}

cat("\nSmall-n edge cases (candidate formula ignores n):\n")
set.seed(10)
x_n10p5 <- matrix(runif(10 * 5), 10, 5); y_n10p5 <- x_n10p5[, 1] + rnorm(10)
out_n10p5 <- capture.output(fit_n10p5 <- earth(x_n10p5, y_n10p5, trace = 2))
cand_n10p5 <- min(200, max(20, 2 * 5)) + 1
cat(sprintf("n=10 p=5:  fit$nk=%d  candidate(ignoring n)=%d\n", fit_n10p5$nk, cand_n10p5))
cat("  Forward pass line:", grep("Forward pass:", out_n10p5, value = TRUE), "\n")

set.seed(11)
x_n30p150 <- matrix(runif(30 * 150), 30, 150); y_n30p150 <- x_n30p150[, 1] + rnorm(30)
out_n30p150 <- capture.output(fit_n30p150 <- earth(x_n30p150, y_n30p150, trace = 2, thresh = 0.05))
cand_n30p150 <- min(200, max(20, 2 * 150)) + 1
cat(sprintf("n=30 p=150: fit$nk=%d  candidate(ignoring n)=%d\n", fit_n30p150$nk, cand_n30p150))
cat("  Forward pass line:", grep("Forward pass:", out_n30p150, value = TRUE), "\n")

nk_reduced_for_small_n <- (fit_n10p5$nk != cand_n10p5) || (fit_n30p150$nk != cand_n30p150)

## ================= Part 2: pairs and nk =================
cat("\n---- Part 2: pairs and nk (single covariate, thresh=0, pmethod=none) ----\n")
set.seed(2)
x1s <- sample((1:100) / 100)
yp2 <- sin(6 * x1s) + 0.05 * rnorm(100)
xp2 <- matrix(x1s, ncol = 1); colnames(xp2) <- "x1"

nrow_le_nk <- logical(0)
nrow_eq_nk_when_reached <- logical(0)
attempted_eq_nk_when_reached <- logical(0)
attempted_by_nk <- integer(0)
for (nk in 2:9) {
  out <- capture.output(
    fit <- earth(xp2, yp2, degree = 1, thresh = 0, pmethod = "none", minspan = 1, endspan = 1,
                 Auto.linpreds = FALSE, fast.k = 0, nk = nk, trace = 2)
  )
  attempted <- forward_complete_attempted(out)
  attempted_by_nk[as.character(nk)] <- attempted
  reached_nk <- fit$termcond == 7  # observed code for "reached nk" during exploration
  nrow_le_nk[as.character(nk)] <- nrow(fit$dirs) <= nk
  if (reached_nk) {
    nrow_eq_nk_when_reached[as.character(nk)] <- nrow(fit$dirs) == nk
    attempted_eq_nk_when_reached[as.character(nk)] <- identical(attempted, as.integer(nk))
  }
  cat(sprintf("nk=%d: nrow(dirs)=%d termcond=%d attempted(forward-pass-complete)=%s\n",
              nk, nrow(fit$dirs), fit$termcond, attempted))
  print(trace_step_lines(out))
  cat("\n")
}

## Refined rule for nk in 3..9 (where termcond==7, i.e. nk was actually reached by
## attempting full pairs): attempted == nk when (nk-1) is even (an integer number of
## pairs exactly fits the budget beyond the intercept); otherwise the forward pass
## stops one slot short (attempted == nk-1) rather than spending the odd leftover slot.
nk39 <- 3:9
expected_attempted <- ifelse((nk39 - 1) %% 2 == 0, nk39, nk39 - 1)
actual_attempted <- attempted_by_nk[as.character(nk39)]
refined_rule_ok <- identical(as.integer(actual_attempted), as.integer(expected_attempted))
cat("Refined rule check -- nk:", nk39, "\n")
cat("  attempted (actual):  ", actual_attempted, "\n")
cat("  attempted (expected):", expected_attempted, "\n")

cat("Repeat with Auto.linpreds=TRUE, nk=2,3:\n")
for (nk in 2:3) {
  out <- capture.output(
    fit <- earth(xp2, yp2, degree = 1, thresh = 0, pmethod = "none", minspan = 1, endspan = 1,
                 Auto.linpreds = TRUE, fast.k = 0, nk = nk, trace = 2)
  )
  cat(sprintf("nk=%d (Auto.linpreds=TRUE): nrow(dirs)=%d termcond=%d attempted=%s\n",
              nk, nrow(fit$dirs), fit$termcond, forward_complete_attempted(out)))
  print(trace_step_lines(out))
  cat("\n")
}

## ================= Part 3: single hinges and nk =================
cat("---- Part 3: does a single-hinge step count as 1 or 2 toward nk? ----\n")
set.seed(4)
x1t <- runif(100)
y3h <- pmax(x1t - 0.3, 0) + 2 * pmax(x1t - 0.7, 0) + 0.02 * rnorm(100)
xp3 <- matrix(x1t, ncol = 1); colnames(xp3) <- "x1"

# First, an unconstrained (nk default) run so we know how many real terms this DGP wants
out_free <- capture.output(
  fit_free <- earth(xp3, y3h, degree = 1, minspan = 1, endspan = 1, Auto.linpreds = FALSE,
                     pmethod = "none", trace = 2)
)
cat(sprintf("unconstrained: nrow(dirs)=%d (nk default=%d), termcond=%d\n",
            nrow(fit_free$dirs), fit_free$nk, fit_free$termcond))
print(trace_step_lines(out_free))

# Now cap nk so the limit bites exactly at the point the single hinge would be added:
# unconstrained used a pair (2) then a single hinge (1) => intercept+2+1 = 4 real terms.
# Set nk=4 (one slot after intercept+pair) and nk=5 to bracket the single-hinge step.
single_hinge_results <- list()
for (nk in c(4, 5)) {
  out <- capture.output(
    fit <- earth(xp3, y3h, degree = 1, minspan = 1, endspan = 1, Auto.linpreds = FALSE,
                 pmethod = "none", nk = nk, trace = 2)
  )
  attempted <- forward_complete_attempted(out)
  cat(sprintf("nk=%d: nrow(dirs)=%d termcond=%d attempted=%s\n", nk, nrow(fit$dirs), fit$termcond, attempted))
  print(trace_step_lines(out))
  cat("\n")
  single_hinge_results[[as.character(nk)]] <- fit
}
# Counting rule: compare how much nrow(dirs) grew (intercept+pair=3 baseline) once the
# single-hinge step is included vs excluded by nk, against how "attempted" grew.
grew_by_one_term_dirs <- nrow(single_hinge_results[["5"]]$dirs) - nrow(single_hinge_results[["4"]]$dirs)
grew_by_one_attempted <- forward_complete_attempted(capture.output(
  fit5 <- earth(xp3, y3h, degree = 1, minspan = 1, endspan = 1, Auto.linpreds = FALSE,
                pmethod = "none", nk = 5, trace = 2))) -
  forward_complete_attempted(capture.output(
    fit4 <- earth(xp3, y3h, degree = 1, minspan = 1, endspan = 1, Auto.linpreds = FALSE,
                  pmethod = "none", nk = 4, trace = 2)))
cat(sprintf("going from nk=4 to nk=5: nrow(dirs) grows by %d; 'attempted' (forward-pass-complete count) grows by %d\n",
            grew_by_one_term_dirs, grew_by_one_attempted))

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb03.1 %s fit$nk == min(200,max(20,2p))+1 for every p in {%s} at n=60\n",
            ifelse(all(nk_match), "TRUE", "FALSE"), paste(p_grid, collapse = ",")))
cat(sprintf("CHECK bb03.2 %s nk is not reduced below the formula for small n (n=10,p=5 and n=30,p=150)\n",
            ifelse(!nk_reduced_for_small_n, "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.3 %s HYPOTHESIS nk is reduced for small n (fit$nk < formula in at least one edge case)\n",
            ifelse(nk_reduced_for_small_n, "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.4 %s the forward pass never lets nrow(dirs) exceed nk, for nk in 2..9\n", ifelse(all(nrow_le_nk), "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.5 %s HYPOTHESIS whenever termcond signals the term limit was reached, nrow(dirs) == nk exactly\n",
            ifelse(length(nrow_eq_nk_when_reached) > 0 && all(nrow_eq_nk_when_reached), "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.6 %s HYPOTHESIS whenever termcond signals the term limit was reached, the counted terms (forward-pass-complete line) == nk\n",
            ifelse(length(attempted_eq_nk_when_reached) > 0 && all(attempted_eq_nk_when_reached), "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.7 %s going from nk=4 to nk=5 adds exactly one row to dirs (a single hinge, not a rejected pair)\n",
            ifelse(grew_by_one_term_dirs == 1, "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.8 %s HYPOTHESIS a single-hinge step is counted as 1 (not 2) toward the 'attempted' term budget too (attempted also grows by 1 from nk=4 to nk=5)\n",
            ifelse(grew_by_one_attempted == 1, "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.9 %s a single-hinge step is counted as a full pair (2 slots) against the budget even though only 1 real term results (attempted grows by 2, matching Part 2's pattern)\n",
            ifelse(grew_by_one_attempted == 2, "TRUE", "FALSE")))
cat(sprintf("CHECK bb03.10 %s for nk = 3..9 the counted terms == nk when (nk-1) is even; attempted == nk-1 when (nk-1) is odd (the forward pass always budgets 2 slots per pair attempt and never spends a lone leftover slot)\n",
            ifelse(refined_rule_ok, "TRUE", "FALSE")))
