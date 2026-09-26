# bb17: how earth scales the response(s) y before the forward pass.
#
# earth's trace=9 log prints, at the top of every knot search block,
# "RssBeforeAddingHinge <rss>": the RSS of the current model (before the
# candidate hinge is added) on earth's internal scale. At the very first
# search (parent = intercept, before any term is added) this is the total
# sum of squares of y on that internal scale, so it lets us read off how y
# was scaled without ever looking at earth's source.
#
# Part 1 (one response, no weights): is the first RssBeforeAddingHinge equal
#   to n - 1, i.e. is y standardized to mean 0, sd 1 with the n-1 divisor?
#   Checked for n = 50, 200 and for a y with a large mean and scale.
# Part 2 (one response, weights w): RssBeforeAddingHinge is compared with
#   sum(w*(y-weighted.mean(y,w))^2) / s^2 for 5 candidate scales s: sd(y)
#   (plain, unweighted), the weighted sd with denominator sum(w)-1, with
#   sum(w), with n-1, and no scaling (s=1). Integer weights (1..3) and
#   continuous weights are both tried, over many random settings.
# Part 3 (three responses): is the first RssBeforeAddingHinge the sum over
#   responses of the unscaled TSS (no per-response scaling)? And does
#   Scale.y = TRUE change it to the sum of each response standardized to
#   sd 1 (so exactly K*(n-1)), even when the responses have wildly
#   different natural scales?
# Part 4: (a) for one response, does scaling y by 1e-9 or 1e9 leave dirs,
#   cuts and selected.terms unchanged? (b) for three responses, does
#   multiplying one response by 1000 change the terms chosen (it should,
#   if the criterion is the RAW summed RSS with no per-response scaling)?
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

# The first RssBeforeAddingHinge line in a trace=9 log: the same number is
# printed for every parent/predictor searched at the first step, so the
# first match in the whole log is exactly the initial (no-term-added) RSS.
rbh1 <- function(X, Y, ...) {
  out <- capture.output(f <- quiet(earth(X, Y, trace = 9, nk = 5, pmethod = "none", ...)))
  if (is.character(f)) return(NA_real_)
  line <- grep("RssBeforeAddingHinge", out, value = TRUE)[1]
  as.numeric(sub(".*RssBeforeAddingHinge ([-+0-9.e]+).*", "\\1", line))
}
rd <- function(a, b) abs(a - b) / abs(b)

## ================= Part 1: one response, no weights =================
cat("\n== Part 1: one response, no weights ==\n")
p1_case <- function(n, big) {
  set.seed(1000 + n)
  x <- matrix(runif(n * 2), n, 2)
  y0 <- x[, 1] + 0.3 * rnorm(n)
  y <- if (big) 1e6 + 1e3 * y0 else y0
  rbh <- rbh1(x, y)
  ok <- rd(rbh, n - 1) < 1e-6
  cat(sprintf("  n=%-4d %-14s RssBeforeAddingHinge=%.6f  n-1=%d  match=%s\n",
    n, if (big) "1e6+1e3*y" else "plain y", rbh, n - 1, ok))
  ok
}
ok_part1 <- all(sapply(c(50, 200), function(n) p1_case(n, FALSE)) , sapply(c(50, 200), function(n) p1_case(n, TRUE)))

## ================= Part 2: one response, weights =================
cat("\n== Part 2: one response, weights ==\n")
set.seed(20260925)
n_trials <- 24
ok_sdy <- ok_wsdm1 <- ok_wsdm <- ok_wsdn1 <- ok_noscale <- logical(n_trials)
for (i in seq_len(n_trials)) {
  n <- sample(20:80, 1)
  x <- matrix(runif(n * 2), n, 2)
  y <- x[, 1] + 0.3 * rnorm(n)
  w <- if (i %% 2 == 0) sample(1:3, n, replace = TRUE) else runif(n, 0.3, 6)
  rbh <- rbh1(x, y, weights = w)
  wm <- weighted.mean(y, w)
  raw <- sum(w * (y - wm)^2)
  tol <- 1e-4
  ok_sdy[i]     <- rd(rbh, raw / var(y)) < tol         # s = sd(y), unweighted
  ok_wsdm1[i]   <- rd(rbh, sum(w) - 1) < tol            # s^2 = raw/(sum(w)-1)
  ok_wsdm[i]    <- rd(rbh, sum(w)) < tol                # s^2 = raw/sum(w)
  ok_wsdn1[i]   <- rd(rbh, n - 1) < tol                 # s^2 = raw/(n-1)
  ok_noscale[i] <- rd(rbh, raw) < tol                   # s = 1
}
cat(sprintf("  %d trials (mixed integer 1:3 and continuous weights):\n", n_trials))
cat(sprintf("  matches sd(y) [plain, unweighted]      : %d/%d\n", sum(ok_sdy), n_trials))
cat(sprintf("  matches weighted sd, denom sum(w)-1    : %d/%d\n", sum(ok_wsdm1), n_trials))
cat(sprintf("  matches weighted sd, denom sum(w)      : %d/%d\n", sum(ok_wsdm), n_trials))
cat(sprintf("  matches weighted sd, denom n-1         : %d/%d\n", sum(ok_wsdn1), n_trials))
cat(sprintf("  matches no scaling (raw weighted TSS)  : %d/%d\n", sum(ok_noscale), n_trials))
# One worked example for the reader.
set.seed(2); n <- 30; x <- matrix(runif(n * 2), n, 2); y <- x[, 1] + 0.1 * rnorm(n)
w <- rep(1:3, length.out = n)
rbh_ex <- rbh1(x, y, weights = w)
wm_ex <- weighted.mean(y, w); raw_ex <- sum(w * (y - wm_ex)^2)
cat(sprintf("  example n=30 w=1:3 repeated: RssBeforeAddingHinge=%.6f | raw/var(y)=%.6f | sum(w)-1=%d | sum(w)=%g | n-1=%d | raw=%.6f\n",
  rbh_ex, raw_ex / var(y), sum(w) - 1, sum(w), n - 1, raw_ex))

## ================= Part 3: three responses =================
cat("\n== Part 3: three responses ==\n")
set.seed(7)
cfgs <- list(c(n = 40, K = 2), c(n = 90, K = 3), c(n = 150, K = 4), c(n = 60, K = 3), c(n = 200, K = 5))
ok_default <- ok_scaley <- logical(length(cfgs))
for (i in seq_along(cfgs)) {
  n <- cfgs[[i]]["n"]; K <- cfgs[[i]]["K"]
  X <- matrix(runif(n * 3), n, 3)
  scales <- 10^runif(K, -3, 3)  # wildly different natural scales per response
  Y <- sapply(seq_len(K), function(k) scales[k] * (X[, (k - 1) %% 3 + 1] + rnorm(n)))
  tss <- sum(apply(Y, 2, function(v) sum((v - mean(v))^2)))
  rbh_d <- rbh1(X, Y)
  rbh_s <- rbh1(X, Y, Scale.y = TRUE)
  ok_default[i] <- rd(rbh_d, tss) < 1e-4
  ok_scaley[i]  <- rd(rbh_s, K * (n - 1)) < 1e-4
  cat(sprintf("  n=%3d K=%d  default RssBeforeAddingHinge=%.6g (sum TSS=%.6g, match=%s) | Scale.y=TRUE -> %.6g (K*(n-1)=%d, match=%s)\n",
    n, K, rbh_d, tss, ok_default[i], rbh_s, K * (n - 1), ok_scaley[i]))
}

## ================= Part 4 =================
cat("\n== Part 4a: one response, invariance to scaling by 1e-9 / 1e9 ==\n")
set.seed(11)
inv_ok <- logical(5)
for (i in 1:5) {
  n <- sample(30:100, 1); x <- matrix(runif(n * 2), n, 2)
  y <- x[, 1] + 0.4 * x[, 2] + 0.2 * rnorm(n)
  f0 <- quiet(earth(x, y, pmethod = "backward"))
  fa <- quiet(earth(x, y * 1e-9, pmethod = "backward"))
  fb <- quiet(earth(x, y * 1e9, pmethod = "backward"))
  inv_ok[i] <- !any(sapply(list(fa, fb), is.character)) &&
    identical(f0$dirs, fa$dirs) && identical(f0$dirs, fb$dirs) &&
    isTRUE(all.equal(f0$cuts, fa$cuts)) && isTRUE(all.equal(f0$cuts, fb$cuts)) &&
    identical(f0$selected.terms, fa$selected.terms) && identical(f0$selected.terms, fb$selected.terms)
}
cat(sprintf("  dirs, cuts, selected.terms unchanged at 1e-9 and 1e9, in %d/5 configs\n", sum(inv_ok)))

cat("\n== Part 4b: three responses, does multiplying one response by 1000 change the terms? ==\n")
set.seed(13)
chg_ok <- logical(5)
for (i in 1:5) {
  n <- 150; X <- matrix(runif(n * 4), n, 4)
  y1 <- X[, 1] + 0.2 * rnorm(n); y2 <- X[, 2] + 0.2 * rnorm(n); y3 <- X[, 3] + 0.2 * rnorm(n)
  Y <- cbind(y1, y2, y3)
  f0 <- quiet(earth(X, Y, degree = 1, pmethod = "none", nk = 11))
  Yb <- Y; Yb[, 2] <- Yb[, 2] * 1000
  fb <- quiet(earth(X, Yb, degree = 1, pmethod = "none", nk = 11))
  chg_ok[i] <- !is.character(f0) && !is.character(fb) &&
    (!identical(f0$dirs, fb$dirs) || !identical(f0$selected.terms, fb$selected.terms))
}
cat(sprintf("  scaling one of three responses by 1000 changed dirs or selected.terms in %d/5 configs\n", sum(chg_ok)))

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb17.1 %s one response, no weights: the first RssBeforeAddingHinge equals n-1 for n=50,200 and for a y with a large mean and scale\n",
  ok_part1))
cat(sprintf("CHECK bb17.2 %s one response, weighted: RssBeforeAddingHinge equals sum(w*(y-weighted.mean(y,w))^2)/var(y), the PLAIN unweighted sd(y); none of the weighted-variance denominators (sum(w)-1, sum(w), n-1) or no-scaling match (%d,%d,%d,%d matches of %d)\n",
  all(ok_sdy) && sum(ok_wsdm1) == 0 && sum(ok_wsdm) == 0 && sum(ok_wsdn1) == 0 && sum(ok_noscale) == 0,
  sum(ok_wsdm1), sum(ok_wsdm), sum(ok_wsdn1), sum(ok_noscale), n_trials))
cat(sprintf("CHECK bb17.3a %s three responses, default: the first RssBeforeAddingHinge is the sum over responses of the unscaled TSS, in all %d configs, even with wildly different per-response scales\n",
  all(ok_default), length(cfgs)))
cat(sprintf("CHECK bb17.3b %s three responses, Scale.y=TRUE: the first RssBeforeAddingHinge is K*(n-1), i.e. each response is standardized to sd 1 before summing, in all %d configs\n",
  all(ok_scaley), length(cfgs)))
cat(sprintf("CHECK bb17.4a %s one response: scaling y by 1e-9 or 1e9 leaves dirs, cuts and selected.terms unchanged (%d/5 configs)\n",
  all(inv_ok), sum(inv_ok)))
cat(sprintf("CHECK bb17.4b %s three responses: multiplying one response by 1000 changes dirs or selected.terms, confirming the default criterion is the RAW summed RSS with no per-response scaling (%d/5 configs)\n",
  all(chg_ok), sum(chg_ok)))
