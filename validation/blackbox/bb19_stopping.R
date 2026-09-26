# bb19: earth's stopping rules for the forward pass, and their order.
#
# The step table (trace 2) prints each step's best candidate with the RSq
# and GRSq of the model that would result, and marks a stop: "reject (small
# DeltaRSq)", "reject (negative GRSq)", "final (max RSq)", "final (reached
# nk)". Hypotheses:
# A. A GRSq below -10 (code 3) or of -Inf (code 2), and an RSq change below
#    thresh (code 4), reject the step's candidate and stop; RSq >= 1 - thresh
#    (code 5) and the term limit (code 7) keep it and stop.
# B. The GRSq of a candidate uses the model's real number of terms: a single
#    hinge adds 1, not the 2 slots that it counts toward nk.
# C. When several rules hold at one step: which code wins?
# D. Code 6: no candidate reduces the RSS.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
run <- function(label, x, y, ...) {
  out <- capture.output(f <- quiet(earth(x, y, trace = 2, pmethod = "none", ...)))
  st <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out, value = TRUE)
  last <- if (length(st)) tail(st, 1) else ""
  mark <- if (grepl("reject", last)) sub(".*(reject \\([^)]*\\)).*", "\\1", last) else if (grepl("final", last)) sub(".*(final \\([^)]*\\)).*", "\\1", last) else ""
  added <- sum(!grepl("reject", st))
  cat(sprintf("%-40s code %s | %2d forward terms | last step: %s | %s\n", label, f$termcond, nrow(f$dirs), mark,
    trimws(grep("Reached|RSq changed|GRSq -Inf|No new term", out, value = TRUE)[1])))
  list(code = f$termcond, nterms = nrow(f$dirs), mark = mark, last = last, fit = f)
}

set.seed(1); n <- 200; X <- matrix(runif(n * 5), n, 5); colnames(X) <- paste0("x", 1:5)
yf <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n)
set.seed(2); x20 <- matrix(runif(20 * 3), 20, 3); colnames(x20) <- paste0("x", 1:3); y20 <- rnorm(20)
x1 <- matrix((1:100) / 100, ncol = 1, dimnames = list(NULL, "x1"))
cat("Part A\n")
a4 <- run("RSq change below thresh (0.05)", X, yf, thresh = 0.05)
a3 <- run("GRSq below -10 (n 20, penalty 3)", x20, y20, nk = 21, penalty = 3, thresh = 1e-6)
a2 <- run("GRSq -Inf (n 20, penalty 10)", x20, y20, nk = 21, penalty = 10, thresh = 1e-6)
a5 <- run("RSq reaches 1 - thresh (exact hinge)", x1, 3 * pmax(x1[, 1] - 0.5, 0), minspan = 1, endspan = 1)
a7 <- run("term limit (nk 9)", X, yf, nk = 9, thresh = 0)
rejects <- a4$mark == "reject (small DeltaRSq)" && a3$mark == "reject (negative GRSq)" && a2$mark == "reject (negative GRSq)"
keeps <- grepl("final \\(max RSq", a5$mark) && grepl("final \\(reached nk", a7$mark)
# the rejected candidate is not in dirs: the fit keeps the terms of the steps before it
kept_ok <- a4$nterms == 11 && a3$nterms == 5 && a2$nterms == 3 && a5$nterms == 3 && a7$nterms == 9

# Part B: one covariate, so step 2 is a single hinge. n = 20, penalty 8:
# C(4) = 16 < 20 but C(5) = 21 >= 20.
cat("Part B\n")
set.seed(4); n <- 20; xb <- matrix(sort(runif(n)), ncol = 1, dimnames = list(NULL, "x1"))
yb <- 4 * pmax(xb[, 1] - 0.3, 0) - 6 * pmax(xb[, 1] - 0.6, 0) + 0.02 * rnorm(n)
out <- capture.output(fb <- quiet(earth(xb, yb, trace = 2, pmethod = "none", minspan = 1, endspan = 1, nk = 9,
  penalty = 8, thresh = 1e-6, Auto.linpreds = FALSE)))
st <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out, value = TRUE)
cat(st, sep = "\n")
g2 <- as.numeric(strsplit(trimws(st[2]), " +")[[1]][2]); r2 <- as.numeric(strsplit(trimws(st[2]), " +")[[1]][3])
tss <- sum((yb - mean(yb))^2)
grsq_M <- function(M) 1 - earth:::get.gcv((1 - r2) * tss, M, 8, n) / earth:::get.gcv(tss, 1, 8, n)
cat(sprintf("step 2 (a single hinge): printed GRSq %s; GRSq with M = 4 (real terms) %.4f, with M = 5 (slots) %.4f\n",
  format(g2), grsq_M(4), grsq_M(5)))
real_M <- is.finite(g2) && abs(g2 - grsq_M(4)) < 5e-4 && !is.finite(grsq_M(5))

# Part C: two rules at one step
cat("Part C\n")
c34 <- run("GRSq -13 and RSq change 0.027 < 0.05", x20, y20, nk = 21, penalty = 3, thresh = 0.05)
x12 <- matrix((1:12) / 12, ncol = 1, dimnames = list(NULL, "x1"))
c35 <- run("n 12, exact hinge, penalty 30", x12, 3 * pmax(x12[, 1] - 0.5, 0), minspan = 1, endspan = 1, penalty = 30)
c57 <- run("RSq reaches 1 - thresh at the last slot", x1, 3 * pmax(x1[, 1] - 0.5, 0), minspan = 1, endspan = 1, nk = 3)
# the small change comes at step 6 (bb19 part A), which is the last step that nk = 13 allows (1 + 2 * 6 = 13)
c47 <- run("small RSq change at the last step (nk 13)", X, yf, thresh = 0.05, nk = 13)

# Part D: no candidate reduces the RSS
cat("Part D\n")
d6 <- run("n 20, thresh 0", x20, y20, nk = 21, penalty = 3, thresh = 0)

cat(sprintf("CHECK bb19.1 %s a GRSq below -10, a GRSq of -Inf and an RSq change below thresh reject the step's candidate and stop (codes 3, 2, 4)\n", rejects))
cat(sprintf("CHECK bb19.2 %s RSq >= 1 - thresh and the term limit keep the step's candidate and stop (codes 5, 7)\n", keeps))
cat(sprintf("CHECK bb19.3 %s the forward terms are those of the steps before a rejected candidate (11, 5, 3), or up to the kept last step (3, 9)\n", kept_ok))
cat(sprintf("CHECK bb19.4 %s the GRSq of a single-hinge candidate uses the real number of terms (M = 4), not the counted slots (M = 5)\n", real_M))
cat(sprintf("CHECK bb19.5 %s when the GRSq rule and the RSq-change rule both hold, the code is %d\n", c34$code %in% c(3, 4), c34$code))
cat(sprintf("CHECK bb19.6 %s when RSq reaches 1 - thresh with a GRSq below -10, the code is %d (%s)\n", c35$code %in% c(2, 3, 5), c35$code, c35$mark))
cat(sprintf("CHECK bb19.7 %s when RSq reaches 1 - thresh at the last slot of nk, the code is %d\n", c57$code %in% c(5, 7), c57$code))
cat(sprintf("CHECK bb19.8 %s when the RSq change is small at the last step that nk allows, the candidate is rejected with code %d (%d terms)\n",
  c47$code == 4 && c47$nterms == 11, c47$code, c47$nterms))
cat(sprintf("CHECK bb19.9 %s with thresh = 0 the pass can stop because no candidate reduces the RSS (code 6)\n", d6$code == 6))
