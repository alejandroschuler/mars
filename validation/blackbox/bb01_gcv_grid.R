# bb01_gcv_grid.R
#
# Question: what is the exact formula behind earth:::get.gcv(rss, nterms, penalty, n)?
#
# Candidate formula:
#   C = nterms + penalty*(nterms-1)/2     for penalty >= 0
#   C = 0                                  for penalty == -1   (so candidate GCV = rss/n)
#   GCV = rss / (n*(1 - C/n)^2)   if C < n
#   GCV = Inf                      if C >= n
# (note C=0 already gives rss/n via the general formula, so penalty==-1 needs no special case)
#
# We check, over a grid of (rss, nterms, penalty, n):
#  bb01.1 numeric agreement with earth's actual get.gcv on all finite cells
#  bb01.2 Inf occurs in earth's output exactly when C >= n
#  bb01.3 hand-picked cells with C exactly equal to n also give non-finite GCV
#  bb01.4 nterms=1 gives C=1 (matches candidate) at every penalty >= 0
# We also print (no claim attached) get.gcv at penalty -0.5 and -2, where the
# candidate formula above is not defined.

cat(R.version.string, "\n")
cat("earth", as.character(packageVersion("earth")), "\n\n")

suppressPackageStartupMessages(library(earth))

safe_get_gcv <- function(rss, nterms, penalty, n) {
  tryCatch(list(value = earth:::get.gcv(rss, nterms, penalty, n), err = NA_character_),
           error = function(e) list(value = NA_real_, err = conditionMessage(e)))
}

candidate_C <- function(nterms, penalty) {
  ifelse(penalty == -1, 0, nterms + penalty * (nterms - 1) / 2)
}
candidate_gcv <- function(rss, nterms, penalty, n) {
  C <- candidate_C(nterms, penalty)
  ifelse(C < n, rss / (n * (1 - C / n)^2), Inf)
}

rss_grid <- c(1, 123.456)
nterms_grid <- 1:41
penalty_grid <- c(-1, 0, 0.5, 1, 2, 3, 4, 5, 6)
n_grid <- c(10, 11, 20, 50, 100, 1000, 100000)

grid <- expand.grid(rss = rss_grid, nterms = nterms_grid, penalty = penalty_grid, n = n_grid)

res <- mapply(function(rss, nterms, penalty, n) safe_get_gcv(rss, nterms, penalty, n),
              grid$rss, grid$nterms, grid$penalty, grid$n, SIMPLIFY = FALSE)
grid$actual <- vapply(res, function(r) r$value, numeric(1))
grid$err <- vapply(res, function(r) r$err, character(1))
grid$candidate <- candidate_gcv(grid$rss, grid$nterms, grid$penalty, grid$n)
grid$C <- candidate_C(grid$nterms, grid$penalty)

n_err <- sum(!is.na(grid$err))
cat("Grid cells:", nrow(grid), " ERROR cells:", n_err, "\n")
if (n_err > 0) {
  cat("Sample errors:\n")
  print(head(grid[!is.na(grid$err), c("rss", "nterms", "penalty", "n", "err")], 5))
}

## bb01.1: numeric agreement on finite cells
finite_mask <- is.finite(grid$actual) & is.finite(grid$candidate)
reldiff <- abs(grid$actual[finite_mask] - grid$candidate[finite_mask]) / abs(grid$candidate[finite_mask])
max_reldiff <- max(reldiff)
worst_idx <- which(finite_mask)[which.max(reldiff)]
cat(sprintf("\nMax relative difference over %d finite cells: %.17g\n", sum(finite_mask), max_reldiff))
cat("Worst cell:\n")
print(grid[worst_idx, c("rss", "nterms", "penalty", "n", "actual", "candidate", "C")], row.names = FALSE)

## bb01.2: Inf <=> C >= n
grid$actual_nonfinite <- !is.finite(grid$actual) & is.na(grid$err)
grid$C_ge_n <- grid$C >= grid$n
inf_mismatches <- grid$actual_nonfinite != grid$C_ge_n
n_inf_mismatch <- sum(inf_mismatches)
cat(sprintf("\nInf-vs-C>=n mismatches over full grid: %d\n", n_inf_mismatch))
if (n_inf_mismatch > 0) {
  print(grid[inf_mismatches, c("rss", "nterms", "penalty", "n", "actual", "C")], row.names = FALSE)
}

## bb01.3: hand-picked cells where C == n exactly
exact_cells <- data.frame(
  rss     = c(1,  1,   123.456, 1),
  nterms  = c(4,  5,   4,       21),
  penalty = c(4,  2.5, 4,       1),
  n       = c(10, 10,  10,      31)
)
exact_cells$C <- candidate_C(exact_cells$nterms, exact_cells$penalty)
stopifnot(all(exact_cells$C == exact_cells$n))  # sanity check on our hand-picked cells themselves
exact_res <- mapply(function(rss, nterms, penalty, n) safe_get_gcv(rss, nterms, penalty, n),
                     exact_cells$rss, exact_cells$nterms, exact_cells$penalty, exact_cells$n, SIMPLIFY = FALSE)
exact_cells$actual <- vapply(exact_res, function(r) r$value, numeric(1))
cat("\nHand-picked cells with C == n exactly:\n")
print(exact_cells, row.names = FALSE)
exact_all_nonfinite <- all(!is.finite(exact_cells$actual))
cat("All hand-picked C==n cells give non-finite actual GCV:", exact_all_nonfinite, "\n")

## bb01.4: nterms=1 => C=1 for every penalty >= 0, and earth matches rss/(n*(1-1/n)^2)
sub1 <- grid[grid$nterms == 1 & grid$penalty >= 0, ]
c_is_1 <- all(sub1$C == 1)
sub1_reldiff <- abs(sub1$actual - sub1$candidate) / abs(sub1$candidate)
max_sub1_reldiff <- max(sub1_reldiff, na.rm = TRUE)
cat(sprintf("\nnterms=1, penalty>=0: candidate C==1 for all %d cells: %s; max reldiff vs earth: %.17g\n",
            nrow(sub1), c_is_1, max_sub1_reldiff))

## For the record: penalty -0.5 and -2 (candidate formula not defined here; just observe)
misc <- expand.grid(rss = c(1, 123.456), nterms = c(1, 5, 20), penalty = c(-0.5, -2), n = c(10, 100))
misc_res <- mapply(function(rss, nterms, penalty, n) safe_get_gcv(rss, nterms, penalty, n),
                    misc$rss, misc$nterms, misc$penalty, misc$n, SIMPLIFY = FALSE)
misc$actual <- vapply(misc_res, function(r) r$value, numeric(1))
misc$err <- vapply(misc_res, function(r) r$err, character(1))
cat("\nget.gcv at penalty -0.5 and -2 (for the record; no claim):\n")
print(misc[, c("rss", "nterms", "penalty", "n", "actual", "err")], row.names = FALSE)

## CHECK lines
cat("\n")
cat(sprintf("CHECK bb01.1 %s max relative difference over finite cells is < 1e-12 (got %.3g)\n",
            ifelse(max_reldiff < 1e-12, "TRUE", "FALSE"), max_reldiff))
cat(sprintf("CHECK bb01.2 %s Inf occurs in earth's output exactly when candidate C >= n over the full grid\n",
            ifelse(n_inf_mismatch == 0, "TRUE", "FALSE")))
cat(sprintf("CHECK bb01.3 %s hand-picked cells with C == n exactly all give non-finite GCV\n",
            ifelse(exact_all_nonfinite, "TRUE", "FALSE")))
cat(sprintf("CHECK bb01.4 %s nterms=1 gives candidate C=1 at every penalty >= 0, matching earth to < 1e-12 (got %.3g)\n",
            ifelse(c_is_1 && max_sub1_reldiff < 1e-12, "TRUE", "FALSE"), max_sub1_reldiff))
