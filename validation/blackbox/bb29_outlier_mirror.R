# bb29: one outlier at the top of x, and the mirror x -> -x (issue #44, T16).
#
# x1 is 99 evenly spaced values in [0, 1] plus one case at +1e6; y is
# 2*(0.5 - x1)+ plus noise at the 99 cases and 0 at the outlier; x2 is noise.
# LA-3 tests the + hinge (x1 - t)+, which is about 1e6 at the outlier, so its
# 1 - R^2 on {1, x1} is about 1e-11 and every knot of x1 is rejected.
# A. With the outlier at +1e6, earth's forward pass adds no hinge on x1.
# B. With the outlier at -1e6 (the mirror of x1), it adds a hinge on x1 near 0.5.
# C. The mirror changes the fit, so x -> -x is not an invariance of earth.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
n0 <- 99; x1 <- (0:(n0 - 1)) / (n0 - 1)
set.seed(2901); e <- 0.05 * rnorm(n0); x2 <- runif(n0 + 1)
y <- c(2 * pmax(0.5 - x1, 0) + e, 0)
fwd <- function(x1full) {
  X <- cbind(x1 = x1full, x2 = x2)
  f <- earth(X, y, degree = 1, pmethod = "none", thresh = 1e-4)
  d <- f$dirs[, "x1"]; ct <- f$cuts[, "x1"]
  hinges <- ct[abs(d) == 1]
  cat(sprintf("  outlier at %+g: %d terms, codes on x1: %s; hinge knots on x1: %s; RSq %.4f\n",
    x1full[n0 + 1], nrow(f$dirs), paste(d, collapse = " "),
    if (length(hinges)) paste(signif(hinges, 6), collapse = " ") else "none", f$rsq))
  list(f = f, hinges = hinges, lin = any(d == 2))
}
A <- fwd(c(x1, 1e6))
B <- fwd(c(-x1, -1e6))
cat(sprintf("CHECK bb29.1 %s with one case at x1 = +1e6, the forward pass adds no hinge on x1 (x1 enters as a linear term: %s)\n",
  length(A$hinges) == 0, A$lin))
cat(sprintf("CHECK bb29.2 %s with x1 mirrored (the outlier at -1e6), the forward pass adds a hinge on x1 with a knot within 0.1 of -0.5\n",
  length(B$hinges) > 0 && any(abs(B$hinges + 0.5) < 0.1)))
cat(sprintf("CHECK bb29.3 %s the mirror changes the fit: RSq %.4f against %.4f\n",
  abs(A$f$rsq - B$f$rsq) > 0.1, A$f$rsq, B$f$rsq))
