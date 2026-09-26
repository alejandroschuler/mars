# bb24: does earth's limit MaxLegalRssDelta = min(1.01 RSS, 10 x the previous
# step's reduction) apply to linear candidates?
#
# In each design, step 2's best candidate is a linear term b*x whose RSS
# reduction (RssDeltaLin in the trace) is far above the step's MaxLegal. If
# earth applied the limit to it, step 2 would take a smaller candidate.
# Designs: a hinge parent times a binary x2 (every knot of x2 is collinear),
# x1*x2 at earth's defaults, the same with Auto.linpreds = FALSE, and two
# continuous covariates.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
traced_out <- function(expr) {  # sink() is much faster than capture.output() for long traces
  tf <- tempfile(); zz <- file(tf, open = "wt"); sink(zz)
  tryCatch(force(expr), error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
  sink(); close(zz); out <- readLines(tf); unlink(tf); out
}
num <- function(pat, s) as.numeric(sub(pat, "\\1", s))

step2 <- function(label, X, y, ...) {
  out <- traced_out(f <- quiet(earth(X, y, trace = 9, pmethod = "none", ...)))
  s <- grep("Searching for new term", out)
  seg <- out[s[2]:(if (length(s) >= 3) s[3] - 1 else length(out))]
  ml <- num(".*MaxLegalRssDelta ([-0-9.e]+).*", seg[1])
  best <- grep("best for term", seg, value = TRUE); best <- best[!grepl("--FindKnot--", best)]
  chosen <- tail(best, 1)
  lin <- grepl("RssDeltaLin", chosen) && grepl("lin pred", chosen)
  red <- if (lin) num(".*RssDeltaLin +([-0-9.e]+).*", chosen) else num(".*RssDelta +([-0-9.e]+).*", chosen)
  kn <- grep("--FindKnot--Case .*RssWithKnot", seg, value = TRUE)
  best_legal_knot <- if (length(kn)) max(c(0, num(".*RssDelta +([-0-9.e]+) Cut.*", kn[grepl("MaxG 1", kn) & grepl("TolG 1", kn)]))) else NA
  cat(sprintf("%-38s MaxLegal %9.4g | chosen %s, reduction %9.4g | best legal knot %9.4g | terms %s\n", label, ml,
    if (lin) "linear" else "knot  ", red, best_legal_knot, paste(rownames(f$dirs), collapse = " ")))
  c(linear = lin, above = red > ml, ratio = red / ml)
}
set.seed(3202); n <- 400; x1 <- runif(n); x2 <- sample(rep(0:1, n / 2)) + 0
yA <- 5 * (x1 - 0.5) * (2 * x2 - 1) + 0.2 * sin(8 * x1) + 0.02 * rnorm(n)
set.seed(9001); n <- 2000; u1 <- runif(n, -1, 1); u2 <- sample(rep(c(-1, 1), n / 2)) + 0
yB <- u1 * u2 + 0.1 * u2 + 0.02 * rnorm(n)
set.seed(515); n <- 1500; a <- runif(n, -1, 1); b <- runif(n, -1, 1)
yC <- a * b + 0.08 * b + 0.01 * rnorm(n)
res <- rbind(
  step2("hinge parent x binary x2", cbind(x1, x2), yA, degree = 2, nk = 15, fast.k = 0, thresh = 0),
  step2("x1*x2 at earth's defaults", cbind(x1 = u1, x2 = u2), yB, degree = 2),
  step2("x1*x2, Auto.linpreds = FALSE", cbind(x1 = u1, x2 = u2), yB, degree = 2, Auto.linpreds = FALSE),
  step2("two continuous covariates", cbind(a = a, b = b), yC, degree = 2, nk = 7, thresh = 0, fast.k = 0))
cat(sprintf("CHECK bb24.1 %s in all 4 designs step 2 takes a linear candidate whose RSS reduction is above MaxLegalRssDelta (%.1f to %.1f times it), so the limit does not apply to linear candidates\n",
  all(res[, "linear"] == 1) && all(res[, "above"] == 1), min(res[, "ratio"]), max(res[, "ratio"])))
