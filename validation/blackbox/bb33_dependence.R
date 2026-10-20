# bb33: earth's float64 dependence tests (spec v3, LA-4, issue #157).
# Question: which relative threshold does earth use to call a column linearly
# dependent (a) in the forward pass, for a linear candidate, (b) in the fix at
# the end of the forward pass, and (c) in its final least-squares fit?
# Data: n = 100, U = U(0, 1), z = N(0, 1) centered, x1 = S (U + c),
# x2 = x1 + 0.1 z, y = U + 5 z, degree 1, nk 5 (two steps), thresh 0,
# pmethod "none", fast.k 0. The z part of y is reachable only through the part
# of x2 orthogonal to (1, x1), whose relative size r falls like 1/S.
# r is ||part of x2 orthogonal to (1, x1)|| / ||x2||, uncentered ("unc"), and
# the same over the centered norm of x2 ("cen").
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
set.seed(1)
n <- 100
U0 <- runif(n); z <- rnorm(n); z <- z - mean(z)
ratio <- function(S, c) {
  x1 <- S * (U0 + c); x2 <- x1 + 0.1 * z
  r <- qr.resid(qr(cbind(1, x1)), x2)
  c(unc = sqrt(sum(r^2)) / sqrt(sum(x2^2)),
    cen = sqrt(sum(r^2)) / sqrt(sum((x2 - mean(x2))^2)))
}
run <- function(S, c) {
  x1 <- S * (U0 + c); x2 <- x1 + 0.1 * z; y <- U0 + 5 * z
  out <- capture.output(f <- tryCatch(
    earth(cbind(x1, x2), y, degree = 1, nk = 5, thresh = 0, pmethod = "none",
          fast.k = 0, trace = 2),
    error = function(e) conditionMessage(e)))
  fwd <- grep("After forward pass", out, value = TRUE)
  list(fixed = any(grepl("Fixed rank deficient", out)),
       error = is.character(f) && grepl("singular", f),
       fwd_rsq = as.numeric(sub(".*RSq ", "", fwd)),
       r = ratio(S, c))
}
cat("\nPart A: c = 0, S = 10^k\n")
A <- list()
for (k in seq(4, 14, by = 1)) {
  a <- run(10^k, 0); A[[length(A) + 1]] <- a
  cat(sprintf("S=1e%-3d r_unc=%.3g r_cen=%.3g forward RSq=%.4f fix=%s final lm.fit error=%s\n",
              k, a$r["unc"], a$r["cen"], a$fwd_rsq, a$fixed, a$error))
}
cat("\nPart B: bisection on log10 S for the fix, at offsets c = 0, 1, 10\n")
Bres <- list()
for (c in c(0, 1, 10)) {
  lo <- 4; hi <- 10
  for (i in 1:30) { mid <- (lo + hi) / 2; if (run(10^mid, c)$fixed) hi <- mid else lo <- mid }
  kept <- run(10^lo, c); fixed <- run(10^hi, c)
  Bres[[length(Bres) + 1]] <- list(c = c, kept = kept, fixed = fixed)
  cat(sprintf("c=%-3g last kept: r_unc=%.5g r_cen=%.4g (lm.fit error=%s); first fixed: r_unc=%.5g r_cen=%.4g\n",
              c, kept$r["unc"], kept$r["cen"], kept$error, fixed$r["unc"], fixed$r["cen"]))
}
cat("\nCHECK bb33.1", all(sapply(A, function(a) a$fwd_rsq >= 0.99)),
    "the forward RSq reaches at least 0.99 at every S from 1e4 to 1e14 (r_unc down to",
    sprintf("%.2g):", min(sapply(A, function(a) a$r["unc"]))),
    "the forward pass counts the near-dependent linear candidate however small its relative orthogonal part, and its RSq can exceed 1\n")
cat("CHECK bb33.2", all(sapply(Bres, function(b) abs(b$kept$r["unc"] / 1e-8 - 1) < 0.01 && abs(b$fixed$r["unc"] / 1e-8 - 1) < 0.01)),
    "the end-of-pass fix removes a term exactly when r_unc falls below 1e-8 (within 1 %) at the offsets 0, 1 and 10\n")
cat("CHECK bb33.3", diff(range(sapply(Bres, function(b) b$kept$r["cen"]))) > 1e-7,
    "at the switch the centered ratio varies with the offset (from",
    sprintf("%.2g to %.2g),", min(sapply(Bres, function(b) b$kept$r["cen"])), max(sapply(Bres, function(b) b$kept$r["cen"]))),
    "so the fix uses the uncentered norm, as lm.fit does (bb09.8)\n")
mid <- Filter(function(a) a$r["unc"] >= 1e-8 && a$r["unc"] < 1e-7, A)
cat("CHECK bb33.4", length(mid) > 0 && all(sapply(mid, function(a) !a$fixed && a$error)),
    "for 1e-8 <= r_unc < 1e-7 the fix keeps the term and earth's final lm.fit (tol 1e-7) stops with a singular-fit error\n")
