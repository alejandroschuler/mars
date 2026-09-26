# bb25: how earth's forward pass ends when a step has no legal candidate, and
# when the Fast MARS window ends a degree-1 pass.
#
# A. With thresh > 0, a step without a legal candidate ends with code 4 (an
#    RSq change of 0 is below thresh), or with code 3 when the GRSq of the
#    unchanged RSS at M + 1 terms is below -10; code 6 needs thresh = 0.
#    The step table then prints a row with DeltaRSq 0, whose GRSq is checked
#    against 1 - GCV(RSS, M + 1)/GCV(TSS, 1).
# B. At degree 1 the window leaves out the intercept once nu entries rank
#    ahead of it. With pairs only and fast.beta = 1 this happens at the first
#    size M with M - 1 >= nu: 7, 11 and 21 terms for fast.k 5, 10 and 20.
#    When the steps add single hinges, older entries age past the intercept
#    and the window does not end the pass.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
run <- function(X, y, ...) {
  out <- capture.output(f <- quiet(earth(X, y, trace = 2, pmethod = "none", ...)))
  rows <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ ", out, value = TRUE)
  list(f = f, last = tail(rows, 1), code = f$termcond, terms = nrow(f$dirs),
    msg = trimws(grep("Reached|RSq changed|GRSq -Inf|No new term", out, value = TRUE)[1]))
}

cat("Part A\n")
set.seed(5101); n <- 300; X6 <- matrix(runif(n * 6), n, 6); colnames(X6) <- paste0("x", 1:6)
y6 <- 3 * sin(3 * X6[, 1]) + 2 * pmax(X6[, 2] - 0.4, 0) + X6[, 3]^2 + 0.5 * X6[, 4] + 0.3 * rnorm(n)
codesA <- sapply(c(0, 1e-6, 0.001, 0.01), function(th) {
  r <- run(X6, y6, degree = 1, fast.k = 3, thresh = th)
  cat(sprintf("  degree 1, fast.k 3, thresh %-6g: code %d, %d terms | %s\n", th, r$code, r$terms, r$msg))
  r$code })
grsq_rows <- function(X, y, pen, ...) {
  r <- run(X, y, penalty = pen, ...)
  tk <- strsplit(trimws(r$last), " +")[[1]]
  tss <- sum((y - mean(y))^2); n <- length(y)
  g <- 1 - earth:::get.gcv(r$f$rss, r$terms + 1, pen, n) / earth:::get.gcv(tss, 1, pen, n)
  cat(sprintf("  %-34s code %d, %d terms | last row GRSq %s, DeltaRSq %s | GRSq(RSS, M + 1) %.4f | %s\n",
    deparse(substitute(X)), r$code, r$terms, tk[2], tk[4], g, r$msg))
  c(code = r$code, printed = as.numeric(tk[2]), computed = g, delta = as.numeric(tk[4]))
}
set.seed(5201); x20 <- matrix(runif(60), 20, 3); colnames(x20) <- paste0("x", 1:3); y20 <- rnorm(20)
g1 <- grsq_rows(x20, y20, 3, nk = 21, thresh = 1e-6)
x14 <- matrix((1:14) / 14, ncol = 1, dimnames = list(NULL, "x1")); set.seed(8); y14 <- sin(6 * x14[, 1]) + 0.2 * rnorm(14)
g2 <- grsq_rows(x14, y14, 6)
c14_thr0 <- run(x14, y14, penalty = 6, thresh = 0)$code
cat(sprintf("  the same n = 14 fit with thresh 0: code %d\n", c14_thr0))

cat("Part B\n")
set.seed(4242); n <- 1500; p <- 24; X24 <- matrix(runif(n * p), n, p); colnames(X24) <- paste0("x", 1:p)
y24 <- rowSums(sapply(1:p, function(j) 3 * 0.93^(j - 1) * abs(X24[, j] - 0.5))) + 0.01 * rnorm(n)
win <- sapply(c(5, 10, 20), function(k) {
  r <- run(X24, y24, degree = 1, nk = 61, fast.k = k, thresh = 0)
  cat(sprintf("  24 covariates, pairs, fast.k %2d, nk 61, thresh 0: %2d terms, code %d\n", k, r$terms, r$code))
  c(k = k, terms = r$terms, code = r$code) })
r0 <- run(X24, y24, degree = 1, nk = 61, fast.k = 0, thresh = 0)
cat(sprintf("  the same with fast.k 0: %d terms, code %d\n", r0$terms, r0$code))
rdef <- run(X24, y24, degree = 1, nk = 41, fast.k = 5)
cat(sprintf("  fast.k 5 at the default thresh: %d terms, code %d\n", rdef$terms, rdef$code))
set.seed(20260926); n <- 1000; p <- 15; X15 <- matrix(runif(n * p), n, p); colnames(X15) <- paste0("x", 1:p)
y15 <- rowSums(sapply(1:p, function(j) sin(2 * pi * (j %% 4 + 1) * X15[, j]) / sqrt(j))) + 0.05 * rnorm(n)
s20 <- run(X15, y15, degree = 1, nk = 81, thresh = 0, fast.k = 20); s00 <- run(X15, y15, degree = 1, nk = 81, thresh = 0, fast.k = 0)
s03 <- run(X15, y15, degree = 1, nk = 81, thresh = 0, fast.k = 3)
cat(sprintf("  15 covariates, many single hinges, nk 81: fast.k 20 gives %d terms (code %d), fast.k 0 %d terms (code %d), fast.k 3 %d terms\n",
  s20$terms, s20$code, s00$terms, s00$code, s03$terms))

cat(sprintf("CHECK bb25.1 %s a step without a legal candidate ends with code 6 at thresh 0 and with code 4 at thresh 1e-6, 0.001 and 0.01\n",
  all(codesA == c(6, 4, 4, 4))))
cat(sprintf("CHECK bb25.2 %s the code is 3 when the GRSq of the unchanged RSS at M + 1 terms is below -10: the last row prints that GRSq (%.4f and %.4f) with DeltaRSq 0\n",
  g1["code"] == 3 && g2["code"] == 3 && abs(g1["printed"] - g1["computed"]) < 1e-3 && abs(g2["printed"] - g2["computed"]) < 1e-2 &&
  g1["delta"] == 0 && g2["delta"] == 0 && c14_thr0 == 6, g1["computed"], g2["computed"]))
cat(sprintf("CHECK bb25.3 %s at degree 1 with pairs, the window ends the pass at 7, 11 and 21 terms for fast.k 5, 10 and 20, the first size M with M - 1 >= nu\n",
  all(win["terms", ] == c(7, 11, 21)) && all(win["code", ] == 6) && r0$terms > 21))
cat(sprintf("CHECK bb25.4 %s when the steps add single hinges, fast.k 20 gives the fit of fast.k 0 (%d terms, code 7), while fast.k 3 stops early\n",
  identical(s20$f$dirs, s00$f$dirs) && identical(s20$f$cuts, s00$f$cuts) && s20$code == 7 && s03$terms < s20$terms, s20$terms))
