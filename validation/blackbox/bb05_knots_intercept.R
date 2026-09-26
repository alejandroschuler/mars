# bb05: which knots does earth evaluate for the intercept parent (degree 1)?
#
# For one variable x and the intercept parent, the first forward step logs,
# at trace = 9, every case that the knot search visits and every cut it
# evaluates. This script states a rule for that list and compares the rule
# with the logged list over many random settings (n, minspan, endspan, ties).
#
# The rule (1-based positions q in the cases sorted by x, all cases active):
#   counter <- nStartSpan
#   for q = n down to endspan + 2:
#     t <- x_(q-1)
#     if t equals the largest x: skip (the counter does not move)
#     else: counter <- counter - 1; if counter == 0: evaluate cut t, counter <- minspan
#   nStartSpan = endspan + ceiling(((n - 2*endspan - 1) mod minspan) / 2)
# where endspan is the capped value max(1, min(endspan, floor(n/2) - 1)),
# for a user value and for the automatic one. The script also checks that
# the automatic spans follow Friedman's equations (43) and (45) with
# alpha = 0.05, truncated, and that a minspan above n is an error.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

# Every earth call is wrapped so that a failure prints only its message.
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

first_scan <- function(x, y, minspan, endspan) {
  out <- capture.output(quiet(earth(matrix(x, ncol = 1), y, trace = 9, nk = 3,
    minspan = minspan, endspan = endspan, Auto.linpreds = FALSE,
    pmethod = "none", thresh = 0)))
  b <- grep("--FindKnotBegin--", out)[1]
  e <- grep("--FindKnotEnd--", out)[1]
  seg <- out[(b + 1):(e - 1)]
  ev <- grepl("RssWithKnot", seg)
  list(
    ms = as.integer(sub(".*nMinSpan ([0-9]+).*", "\\1", out[b])),
    es = as.integer(sub(".*nEndSpan ([0-9]+).*", "\\1", out[b])),
    start = as.integer(sub(".*nStartSpan ([0-9]+).*", "\\1", out[b])),
    cuts = as.numeric(sub(".*Cut +([-0-9.e]+).*", "\\1", seg[ev])),
    tolg = as.integer(sub(".*TolG ([01]).*", "\\1", seg[ev])),
    visited = as.integer(sub(".*Case +([0-9]+).*", "\\1", seg))
  )
}

cap_endspan <- function(n, es) max(1, min(es, floor(n / 2) - 1))
# With the cap, n - 2 es - 1 >= 0, so the remainder is never negative.
start_rule <- function(n, ms, es) es + ceiling(((n - 2 * es - 1) %% ms) / 2)

cuts_rule <- function(x, ms, es, start) {
  xs <- sort(x); n <- length(xs); counter <- start; cuts <- numeric(0)
  if (n >= es + 2) for (q in n:(es + 2)) {
    t <- xs[q - 1]
    if (t == xs[n]) next
    counter <- counter - 1
    if (counter == 0) { cuts <- c(cuts, t); counter <- ms }
  }
  cuts
}

auto_minspan <- function(n, p) trunc(-log2(-log(1 - 0.05) / (p * n)) / 2.5)
auto_endspan <- function(n, p) max(1, min(trunc(3 - log2(0.05 / p)), floor(n / 2) - 1))

# Part 1: user spans, distinct and tied x, many random settings.
set.seed(20260925)
n_cfg <- 0; n_ok_cuts <- 0; n_ok_start <- 0; n_ok_range <- 0; bad <- character(0)
kinds <- c("distinct", "pairs", "run_mid", "run_min", "run_max", "few_levels")
make_x <- function(kind, n) {
  switch(kind,
    distinct = sample(1:n),
    pairs = sample(rep(1:ceiling(n / 2), 2)[1:n]),
    run_mid = sample(c(1:(n - 5), rep(ceiling(n / 2), 5))),
    run_min = sample(c(rep(1, 4), 2:(n - 3))),
    run_max = sample(c(1:(n - 4), rep(n, 4))),
    few_levels = sample(rep(1:5, length.out = n)))
}
for (i in 1:240) {
  kind <- kinds[(i %% length(kinds)) + 1]
  n <- sample(8:60, 1); ms <- sample(1:8, 1); es <- sample(1:6, 1)
  x <- make_x(kind, n) + 0
  y <- sin(x / 3) + 0.2 * rnorm(n)
  r <- first_scan(x, y, ms, es)
  n_cfg <- n_cfg + 1
  es_eff <- cap_endspan(n, es)
  exp_start <- start_rule(n, ms, es_eff)
  exp_cuts <- cuts_rule(x, ms, es_eff, exp_start)
  ok_start <- r$start == exp_start && r$es == es_eff
  ok_cuts <- identical(r$cuts, exp_cuts)
  ok_range <- length(r$visited) == 0 || (max(r$visited) == n - 1 && min(r$visited) == es_eff + 1)
  n_ok_start <- n_ok_start + ok_start; n_ok_cuts <- n_ok_cuts + ok_cuts; n_ok_range <- n_ok_range + ok_range
  if (!(ok_start && ok_cuts && ok_range)) bad <- c(bad, sprintf("%s n=%d ms=%d es=%d start=%d/%d cuts=%s | rule=%s",
    kind, n, ms, es, r$start, exp_start, paste(r$cuts, collapse = ","), paste(exp_cuts, collapse = ",")))
}
cat(sprintf("part 1: %d settings; nStartSpan matches %d; cut list matches %d; visited range n-1..endspan+1 %d\n",
  n_cfg, n_ok_start, n_ok_cuts, n_ok_range))
if (length(bad)) cat("mismatches:\n", paste(" ", head(bad, 20)), sep = "\n")

# Two examples in full, for the reader.
show <- function(x, ms, es) {
  set.seed(1); r <- first_scan(x, sin(x / 3) + 0.2 * rnorm(length(x)), ms, es)
  cat(sprintf("sorted x: %s\n  minspan %d endspan %d nStartSpan %d -> cuts %s (TolG %s)\n",
    paste(sort(x), collapse = " "), r$ms, r$es, r$start, paste(r$cuts, collapse = ","), paste(r$tolg, collapse = "")))
}
set.seed(2); show(sample(1:20) + 0, 3, 1); show(sample(1:20) + 0, 4, 2)
set.seed(3); show(sample(c(1:16, rep(17, 4))) + 0, 3, 2); show(sample(c(rep(1, 4), 2:17)) + 0, 1, 1)

# Part 2: automatic spans (minspan = 0, endspan = 0) at the intercept.
res <- NULL
for (p in c(1, 2, 3, 5, 10, 50, 100)) for (n in c(3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 21, 22, 30, 50, 100, 200, 1000, 10000)) {
  if (n * p > 2e5) next
  set.seed(n + p); x <- matrix(runif(n * p), n, p); y <- x[, 1] + 0.1 * rnorm(n)
  out <- capture.output(quiet(earth(x, y, trace = 9, nk = 3, pmethod = "none")))
  l <- grep("--FindKnotBegin--", out, value = TRUE)[1]
  f <- grep("Forward pass: minspan", out, value = TRUE)[1]
  res <- rbind(res, data.frame(n = n, p = p,
    printed_ms = as.integer(sub(".*minspan ([0-9]+) .*", "\\1", f)),
    printed_es = as.integer(sub(".*endspan ([0-9]+) .*", "\\1", f)),
    scan_ms = as.integer(sub(".*nMinSpan ([0-9]+).*", "\\1", l)),
    scan_es = as.integer(sub(".*nEndSpan ([0-9]+).*", "\\1", l)),
    eq43 = -log2(-log(1 - 0.05) / (p * n)) / 2.5, eq45 = 3 - log2(0.05 / p),
    rule_ms = auto_minspan(n, p), rule_es = auto_endspan(n, p)))
}
print(res[res$n %in% c(3, 5, 8, 20, 21, 100, 10000) | res$p == 100, ], digits = 4, row.names = FALSE)
ok_ms <- all(res$printed_ms == res$rule_ms & res$scan_ms == res$rule_ms, na.rm = FALSE)
ok_es <- all(res$printed_es == res$rule_es & res$scan_es == res$rule_es)
trunc_not_round <- any(round(res$eq43) != trunc(res$eq43) & res$printed_ms == trunc(res$eq43))

# Part 3: a user endspan above the cap, a minspan wider than the range, and
# a minspan above n.
set.seed(4); x <- sample(1:20) + 0; y <- sin(x / 3) + 0.2 * rnorm(20)
r1 <- first_scan(x, y, 1, 12)
set.seed(4); x40 <- sample(1:40) + 0; y40 <- sin(x40 / 3) + 0.2 * rnorm(40)
r2 <- first_scan(x40, y40, 30, 1)
m3 <- quiet(earth(matrix(x, ncol = 1), y, minspan = 21, endspan = 1))
m4 <- quiet(earth(matrix(x, ncol = 1), y, minspan = 20, endspan = 1))
cat(sprintf("n=20 endspan=12: nEndSpan %d, cuts %s\n", r1$es, paste(r1$cuts, collapse = ",")))
cat(sprintf("n=40 minspan=30 endspan=1: nStartSpan %d, cuts %s\n", r2$start, paste(r2$cuts, collapse = ",")))
cat("n=20 minspan=21:", if (is.character(m3)) m3 else "no error", "\n")
cat("n=20 minspan=20:", if (is.character(m4)) m4 else "no error", "\n")

cat(sprintf("CHECK bb05.1 %s the visited cases run from n-1 down to capped endspan+1 in all %d settings\n", n_ok_range == n_cfg, n_cfg))
cat(sprintf("CHECK bb05.2 %s nEndSpan is the capped endspan and nStartSpan = endspan + ceiling(((n - 2 endspan - 1) mod minspan)/2) in all settings\n", n_ok_start == n_cfg))
cat(sprintf("CHECK bb05.3 %s the evaluated cut list (order and repeats) equals the rule in all settings\n", n_ok_cuts == n_cfg))
cat(sprintf("CHECK bb05.4 %s automatic minspan = trunc(eq. 43 with n cases and p columns)\n", ok_ms))
cat(sprintf("CHECK bb05.5 %s automatic endspan = max(1, min(trunc(eq. 45 with p), floor(n/2) - 1))\n", ok_es))
cat(sprintf("CHECK bb05.6 %s eq. 43 is truncated, not rounded (some cell has round != trunc)\n", trunc_not_round))
cat(sprintf("CHECK bb05.7 %s a user endspan is capped the same way (endspan 12 at n = 20 becomes 9)\n", r1$es == 9))
cat(sprintf("CHECK bb05.8 %s a minspan wider than the range follows the same rule (cuts %s)\n",
  identical(r2$cuts, cuts_rule(x40, 30, 1, start_rule(40, 30, 1))), paste(r2$cuts, collapse = ",")))
cat(sprintf("CHECK bb05.9 %s minspan above n is an error and minspan = n is not\n",
  is.character(m3) && grepl("minspan", m3) && !is.character(m4)))
