# bb10: earth's termination codes (fit$termcond) and the message that each
# prints at trace = 1, for each rule that can end the forward pass: the term
# limit before any step, the term limit after some steps, an RSq change below
# thresh, RSq at least 1 - thresh, GRSq below -10, GRSq of -Inf, and no new
# term that raises RSq. Each case prints its settings, termcond, the number of
# forward terms, and the trace line that names the condition.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

run <- function(label, x, y, ...) {
  out <- capture.output(f <- quiet(earth(x, y, trace = 1, pmethod = "none", ...)))
  if (is.character(f)) { cat(sprintf("%-28s %s\n", label, f)); return(NA) }
  msg <- grep("Reached|RSq changed|GRSq|No new term|no new term|increase|min GRSq|nk", out, value = TRUE)
  msg <- msg[!grepl("^After|Forward pass complete|^Prune|selected", msg)]
  cat(sprintf("%-28s termcond %d, %2d forward terms | %s\n", label, f$termcond, nrow(f$dirs), paste(trimws(msg), collapse = " / ")))
  f$termcond
}

set.seed(1); n <- 200; X <- matrix(runif(n * 5), n, 5)
yf <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n)
codes <- list()
codes$nk_start <- run("nk = 2 (no step fits)", X, yf, nk = 2)
codes$nk_start1 <- run("nk = 1", X, yf, nk = 1)
codes$nk <- run("nk = 9", X, yf, nk = 9)
codes$delta <- run("defaults (thresh 0.001)", X, yf)
codes$delta2 <- run("thresh 0.05", X, yf, thresh = 0.05)
x1 <- matrix((1:100) / 100, ncol = 1); ylin <- 3 * pmax(x1[, 1] - 0.5, 0)
codes$rsq <- run("exact hinge, no noise", x1, ylin, minspan = 1, endspan = 1)
codes$rsq2 <- run("exact hinge, thresh 0.1", x1, ylin + 0.01 * sin(1:100), thresh = 0.1)
set.seed(2); n <- 20; x20 <- matrix(runif(n * 3), n, 3); y20 <- rnorm(n)
codes$grsq0 <- run("n = 20, noise, thresh 0", x20, y20, thresh = 0, nk = 21, penalty = 3)
codes$grsq0_big <- run("n = 20, noise, pen 10, thr 0", x20, y20, thresh = 0, nk = 21, penalty = 10)
codes$grsq_small <- run("n = 20, noise, penalty -1", x20, y20, thresh = 0, nk = 21, penalty = -1)
codes$grsq <- run("n = 20, noise, penalty 3", x20, y20, nk = 21, penalty = 3, thresh = 1e-6)
codes$grsq_big <- run("n = 20, noise, penalty 10", x20, y20, nk = 21, penalty = 10, thresh = 1e-6)
set.seed(5); x40 <- matrix(runif(40 * 3), 40, 3); y40 <- rnorm(40)
codes$grsq40 <- run("n = 40, noise, penalty 6", x40, y40, nk = 41, penalty = 6, thresh = 1e-6, minspan = 1, endspan = 1)
set.seed(3); n <- 12; x12 <- matrix(runif(n), n, 1); y12 <- rnorm(n)
codes$norsq <- run("n = 12, one x, thresh 0", x12, y12, thresh = 0, nk = 21, penalty = -1, minspan = 1, endspan = 1)
set.seed(4); n <- 30; xa <- matrix(runif(n * 2), n, 2); ya <- xa[, 1] + 0.1 * rnorm(n)
codes$c <- run("n = 30, thresh 0, pen -1", xa, ya, thresh = 0, nk = 61, penalty = -1, minspan = 1, endspan = 1)
x0 <- matrix((1:50) / 50, ncol = 1)
codes$const_y <- run("constant y", x0, rep(1, 50))

tab <- unlist(codes)
cat("\ncodes seen:", paste(names(tab), tab, sep = "=", collapse = ", "), "\n")
cat(sprintf("CHECK bb10.1 %s the term limit gives code 1 when no step fits (nk <= 2) and code 7 after steps\n",
  isTRUE(tab["nk_start"] == 1) && isTRUE(tab["nk"] == 7)))
cat(sprintf("CHECK bb10.2 %s an RSq change below thresh gives code 4 (thresh 0.05; and a constant y at the defaults)\n",
  isTRUE(tab["delta2"] == 4) && isTRUE(tab["const_y"] == 4)))
cat(sprintf("CHECK bb10.3 %s RSq of at least 1 - thresh gives code 5\n", isTRUE(tab["rsq"] == 5) || isTRUE(tab["rsq2"] == 5)))
cat(sprintf("CHECK bb10.4 %s GRSq of -Inf gives code 2 and GRSq below -10 gives code 3, and neither rule acts when thresh = 0\n",
  isTRUE(tab["grsq_big"] == 2) && isTRUE(tab["grsq"] == 3) && isTRUE(tab["grsq40"] == 3) &&
  !any(tab[c("grsq0", "grsq0_big")] %in% c(2, 3))))
cat(sprintf("CHECK bb10.5 %s no new term that raises RSq gives code 6\n", all(tab[c("norsq", "grsq_small")] == 6)))
cat(sprintf("CHECK bb10.6 %s with thresh = 0 the rule RSq >= 1 - thresh still acts (an exact fit gives code 5)\n", isTRUE(tab["c"] == 5)))

# A constant y at several n: earth's termination code, rss, gcv, rsq and grsq
cy <- lapply(c(3, 12, 100), function(n) {
  set.seed(n); xc <- matrix(runif(2 * n), n, 2)
  out <- capture.output(f <- quiet(withCallingHandlers(earth(xc, rep(0.1, n)), warning = function(w) invokeRestart("muffleWarning"))))
  cat(sprintf("constant y = 0.1, n = %3d: termcond %d, terms %d, rss %g, gcv %g, rsq %s, grsq %s\n", n, f$termcond, nrow(f$dirs), f$rss, f$gcv, f$rsq, f$grsq))
  c(code = f$termcond, rss = f$rss, gcv = f$gcv, rsq_nan = is.nan(f$rsq), grsq_nan = is.nan(f$grsq))
})
cy <- do.call(rbind, cy)
cat(sprintf("CHECK bb10.7 %s for a constant y earth gives the intercept alone, rss 0 and gcv 0, rsq and grsq NaN, and termination code 2 at n = 3 and 4 at n = 12 and 100\n",
  all(cy[, "rss"] == 0) && all(cy[, "gcv"] == 0) && all(cy[, "rsq_nan"] == 1) && all(cy[, "grsq_nan"] == 1) && all(cy[, "code"] == c(2, 4, 4))))
