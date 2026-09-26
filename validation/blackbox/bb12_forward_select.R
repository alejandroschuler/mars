# bb12: how earth's forward pass chooses a candidate.
#
# A. With thresh = 0 (no GRSq stopping rules), the penalty does not change
#    the forward pass: the same dirs and cuts for penalty -1, 0, 2, 3 and 10.
# B. Each step prints MaxLegalRssDelta. Hypothesis: it is
#    min(1.01 * RSS before the step, 10 * the RSS reduction of the previous
#    step), and 1.01 * RSS at the first step.
# C. A knot whose RSS reduction exceeds MaxLegalRssDelta is flagged MaxG 0,
#    and the step chooses the candidate with the largest RSS reduction among
#    those with TolG 1 and MaxG 1, counting the linear option of each pair
#    search (its RssDeltaLin). The trace prints about 5 digits, so a match
#    means: the chosen candidate's printed reduction is the largest printed
#    valid reduction of the step.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
num <- function(pat, s) as.numeric(sub(pat, "\\1", s))

# Part A
set.seed(1); n <- 200; X <- matrix(runif(n * 5), n, 5); colnames(X) <- paste0("x", 1:5)
yf <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n)
same_all <- TRUE
for (deg in 1:2) {
  keys <- sapply(c(-1, 0, 2, 3, 10), function(pen) {
    f <- quiet(earth(X, yf, degree = deg, penalty = pen, thresh = 0, nk = 21, pmethod = "none"))
    paste(c(f$dirs, signif(f$cuts, 12)), collapse = ",")
  })
  cat(sprintf("part A: degree %d, forward bases for penalties -1, 0, 2, 3, 10 identical: %s\n", deg, length(unique(keys)) == 1))
  same_all <- same_all && length(unique(keys)) == 1
}

# Parts B and C: parse every step of a trace-9 log.
steps_of <- function(out) {
  s <- grep("Searching for new term", out)
  ends <- c(s[-1] - 1, length(out))
  lapply(seq_along(s), function(k) {
    seg <- out[s[k]:ends[k]]
    ml <- num(".*MaxLegalRssDelta ([-0-9.e]+).*", seg[1])
    b <- grep("--FindKnotBegin--", seg)
    rb <- if (length(b)) num(".*RssBeforeAddingHinge ([-0-9.e]+).*", seg[b[1]]) else NA
    # candidates: knot lines and the linear-option lines of pair searches
    kn <- grep("--FindKnot--Case .*RssWithKnot", seg, value = TRUE)
    lin <- grep("^\\|Parent .*Case +-1 .*RssDeltaLin", seg, value = TRUE)
    cand <- rbind(
      data.frame(kind = "knot", delta = num(".*RssDelta +([-0-9.e]+) Cut.*", kn),
        valid = grepl("TolG 1", kn) & grepl("MaxG 1", kn) & grepl("bx1G 1", kn) & grepl("CovColG 1", kn),
        maxg0 = grepl("MaxG 0", kn)),
      data.frame(kind = rep("linear", length(lin)), delta = num(".*RssDeltaLin +([-0-9.e]+).*", lin),
        valid = rep(TRUE, length(lin)), maxg0 = rep(FALSE, length(lin))))
    best <- grep("best for term", seg, value = TRUE)
    best <- best[!grepl("--FindKnot--", best)]
    chosen <- if (length(best)) tail(best, 1) else NA
    chosen_delta <- if (is.na(chosen)) NA else if (grepl("RssDeltaLin", chosen)) num(".*RssDeltaLin +([-0-9.e]+).*", chosen) else num(".*RssDelta +([-0-9.e]+).*", chosen)
    list(maxlegal = ml, rss_before = rb, cand = cand, chosen_delta = chosen_delta)
  })
}
analyze <- function(label, X, y, ...) {
  out <- capture.output(f <- quiet(earth(X, y, trace = 9, pmethod = "none", ...)))
  st <- steps_of(out)
  rb <- sapply(st, `[[`, "rss_before"); ml <- sapply(st, `[[`, "maxlegal")
  prev <- c(NA, -diff(rb))
  pred <- ifelse(is.na(prev), 1.01 * rb, pmin(1.01 * rb, 10 * prev))
  okB <- all(abs(ml - pred) <= 5e-4 * pmax(ml, 1e-12) + 1e-3, na.rm = TRUE)
  okC <- 0; tot <- 0; bind <- 0
  for (s in st) {
    if (is.na(s$chosen_delta) || !nrow(s$cand)) next
    v <- s$cand$delta[s$cand$valid]
    tot <- tot + 1
    okC <- okC + (abs(max(v) - s$chosen_delta) <= 1e-4 * max(abs(v), 1e-12) + 1e-9)
    bind <- bind + any(s$cand$maxg0 & s$cand$delta > s$chosen_delta)
  }
  cat(sprintf("%-40s steps %2d | MaxLegal formula holds: %s | chosen = best valid: %d of %d | steps where a MaxG 0 knot beat the choice: %d\n",
    label, length(st), okB, okC, tot, bind))
  list(okB = okB, okC = okC == tot, bind = bind, tab = data.frame(rss_before = rb, prev_delta = prev, maxlegal = ml, formula = pred))
}
n <- 600; x <- (1:n) / n; set.seed(2); x <- sample(x)
y1 <- sin(12 * x) + 0.5 * sin(40 * x) + 0.05 * rnorm(n)
r1 <- analyze("one x, sin, minspan 1", matrix(x, ncol = 1), y1, nk = 21, minspan = 1, endspan = 1, Auto.linpreds = FALSE, thresh = 0)
print(r1$tab, digits = 6, row.names = FALSE)
r2 <- analyze("Friedman #1, degree 1", X, yf, degree = 1, nk = 21, thresh = 0)
r3 <- analyze("Friedman #1, degree 2", X, yf, degree = 2, nk = 21, thresh = 0, fast.k = 0)
set.seed(5); n <- 300; X3 <- matrix(runif(n * 3), n, 3); colnames(X3) <- paste0("x", 1:3)
y3 <- 20 * pmax(X3[, 1] - 0.5, 0) * pmax(X3[, 2] - 0.5, 0) + 0.3 * X3[, 3] + 0.05 * rnorm(n)
r4 <- analyze("interaction after a weak step, degree 2", X3, y3, degree = 2, nk = 15, thresh = 0, fast.k = 0)
rs <- list(r1, r2, r3, r4)

cat(sprintf("CHECK bb12.1 %s with thresh = 0 the forward basis is the same for penalties -1, 0, 2, 3 and 10 (degree 1 and 2)\n", same_all))
cat(sprintf("CHECK bb12.2 %s MaxLegalRssDelta = min(1.01 * RSS before the step, 10 * the previous step's RSS reduction), and 1.01 * RSS at the first step\n",
  all(sapply(rs, `[[`, "okB"))))
cat(sprintf("CHECK bb12.3 %s each step chooses the largest RSS reduction among the knots with TolG 1 and MaxG 1 and the linear options\n",
  all(sapply(rs, `[[`, "okC"))))
cat(sprintf("CHECK bb12.4 %s in some steps a knot with MaxG 0 had a larger RSS reduction than the chosen candidate (%d steps)\n",
  sum(sapply(rs, `[[`, "bind")) > 0, sum(sapply(rs, `[[`, "bind"))))
