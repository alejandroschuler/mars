# bb14: when does earth run a pair search, and when a single-hinge search?
#
# For a parent b and covariate x, let A = ||(I - P) (b*x)||^2, the residual
# sum of squares of the column b*x regressed on the current columns (the
# intercept included), with x on its raw scale. The hypothesis: in unweighted
# fits earth runs a pair search when A >= 0.01 and a single-hinge search when
# A < 0.01, and 1 - R^2 of b*x plays no part. A search's kind is read from the
# trace: iNewCol = K for a single-hinge search, K + 1 for a pair (bb11.5).
# Trace parent numbers count slots: every step takes two slots, and a
# one-term step leaves its second slot empty, so the script maps slots to
# dirs rows with the step lines (whose Terms column counts dirs rows).
#
# Part A: every search of several fits. Part B: moving A across 0.01 by the
# scale of x2 at a fixed 1 - R^2. Part C: the same data in two units gives
# different models. Part D: in a single-hinge search a knot's RSS is
# RSS(current + h), and a knot at a repeated minimum can pass the
# collinearity test. Part E: with weights that are not all equal, pair
# searches occur below A = 0.01.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
# Traces are long; sink() to a file is much faster than traced_out().
# The expression is a promise, so assignments inside it land in the caller.
traced_out <- function(expr) {
  tf <- tempfile(); zz <- file(tf, open = "wt"); sink(zz)
  tryCatch(force(expr), error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
  sink(); close(zz); out <- readLines(tf); unlink(tf)
  out
}
num <- function(pat, s) as.numeric(sub(pat, "\\1", s))

basis <- function(dirs, cuts, X) sapply(seq_len(nrow(dirs)), function(k) {
  v <- rep(1, nrow(X))
  for (j in which(dirs[k, ] != 0)) v <- v * switch(as.character(dirs[k, j]),
    "1" = pmax(X[, j] - cuts[k, j], 0), "-1" = pmax(cuts[k, j] - X[, j], 0), "2" = X[, j])
  v })
# Step lines: K GRSq RSq DeltaRSq Pred PredName Cut Terms [Par] Deg.
steps_of <- function(out) {
  i <- grep("^[0-9]+ +[-0-9.]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ +x[0-9]+ ", out)
  lapply(i, function(ii) {
    tk <- strsplit(trimws(sub(" final.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    list(K = as.integer(tk[1]), terms = if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)])
  })
}
slot_map <- function(st) {  # slot number -> dirs row
  m <- c(`1` = 1L); row <- 1L
  for (s in st) for (q in seq_along(s$terms)) { row <- row + 1L; m[as.character(s$K + q - 1)] <- row }
  m
}
searches <- function(X, y, w = NULL, ...) {
  out <- traced_out(f <- quiet(earth(X, y, weights = w, trace = 9, pmethod = "none", thresh = 0, fast.k = 0,
    Auto.linpreds = FALSE, minspan = 1, endspan = 1, ...)))
  st <- steps_of(out); smap <- slot_map(st); B <- basis(f$dirs, f$cuts, X)
  rows_before <- 1L + cumsum(c(0L, sapply(st, function(s) length(s$terms))))
  sl <- grep("Searching for new term", out); Ks <- num(".*new term ([0-9]+).*", out[sl])
  b <- grep("--FindKnotBegin--", out); par <- grep("^\\|Parent", out)
  res <- NULL; qrs <- list()
  for (bb in b) {
    si <- max(which(sl < bb)); P <- num("^\\|Parent +([0-9]+).*", out[max(par[par < bb])])
    pr <- smap[as.character(P)]; j <- num(".*iPred ([0-9]+).*", out[bb])
    if (is.na(pr) || si > length(rows_before)) next
    key <- as.character(si)
    if (is.null(qrs[[key]])) qrs[[key]] <- qr(B[, seq_len(rows_before[si]), drop = FALSE])
    l <- B[, pr] * X[, j]; e <- qr.resid(qrs[[key]], l)
    res <- rbind(res, data.frame(step = si, single = num(".*iNewCol ([0-9]+).*", out[bb]) == Ks[si],
      A = sum(e^2), one_minus_r2 = if (sum((l - mean(l))^2) > 0) sum(e^2) / sum((l - mean(l))^2) else 0))
  }
  res
}

# Part A
set.seed(1); n <- 200; X5 <- matrix(runif(n * 5), n, 5); colnames(X5) <- paste0("x", 1:5)
yf <- 10 * sin(pi * X5[, 1] * X5[, 2]) + 20 * (X5[, 3] - 0.5)^2 + 10 * X5[, 4] + 5 * X5[, 5] + rnorm(n)
set.seed(2); X3 <- matrix(runif(300 * 3), 300, 3); colnames(X3) <- paste0("x", 1:3)
y3 <- sin(8 * X3[, 1]) + 2 * pmax(X3[, 1] - 0.3, 0) * X3[, 2] + 0.5 * sin(6 * X3[, 3]) + 0.05 * rnorm(300)
set.seed(3); X4 <- matrix(runif(150 * 4) * c(1, 3, 0.5, 2), 150, 4, byrow = TRUE); colnames(X4) <- paste0("x", 1:4)
y4 <- 4 * pmax(X4[, 1] - 0.5, 0) * X4[, 2] + sin(3 * X4[, 3]) + 0.1 * rnorm(150)
all <- rbind(searches(X5, yf, degree = 2, nk = 31), searches(X5, yf, degree = 3, nk = 31),
  searches(X3, y3, degree = 2, nk = 31), searches(X3, y3, degree = 3, nk = 25), searches(X4, y4, degree = 2, nk = 25))
okA <- all(all$single == (all$A < 0.01))
cat(sprintf("part A: %d searches (%d single-hinge); largest A of a single-hinge search %.6g, smallest A of a pair search %.6g\n",
  nrow(all), sum(all$single), max(all$A[all$single]), min(all$A[!all$single])))
cat(sprintf("        1 - R^2 of b*x: single-hinge searches up to %.4g, pair searches down to %.4g\n",
  max(all$one_minus_r2[all$single]), min(all$one_minus_r2[!all$single])))

# Part B: A across 0.01 by the scale of x2, with 1 - R^2 fixed
set.seed(515); n <- 60; x1 <- runif(n); x2 <- runif(n)
yb <- 5 * pmax(x1 - 0.5, 0) + 0.3 * pmax(x1 - 0.5, 0) * x2 + 0.05 * rnorm(n)
kind_at <- function(s) {
  out <- traced_out(f <- quiet(earth(cbind(x1, x2 = s * x2), yb, degree = 2, trace = 9, nk = 5, minspan = 1,
    endspan = 1, Adjust.endspan = 0, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
  seg <- out[grep("Searching for new term 4 ", out):length(out)]
  a <- grep("^\\|Parent 2  Pred 2 ", seg)[1]; bg <- grep("--FindKnotBegin--", seg); bg <- bg[bg > a][1]
  B <- cbind(1, pmax(x1 - f$cuts[2, 1], 0), pmax(f$cuts[2, 1] - x1, 0)); l <- B[, 2] * s * x2
  e <- qr.resid(qr(B), l)
  c(A = sum(e^2), r2 = sum(e^2) / sum((l - mean(l))^2), single = as.numeric(grepl(" iNewCol 4 ", seg[bg])))
}
kb <- rbind(kind_at(0.218845), kind_at(0.221045))
print(data.frame(scale = c(0.218845, 0.221045), A = kb[, "A"], one_minus_r2 = kb[, "r2"], single = kb[, "single"] == 1), digits = 6, row.names = FALSE)
okB <- kb[1, "single"] == 1 && kb[2, "single"] == 0 && kb[1, "A"] < 0.01 && kb[2, "A"] >= 0.01 && abs(kb[1, "r2"] - kb[2, "r2"]) < 1e-9

# Part C: the same data in two units
set.seed(77); n <- 100; xc <- runif(n); yc <- 3 * pmax(xc - 0.3, 0) - 4 * pmax(xc - 0.7, 0) + 0.05 * rnorm(n)
fc1 <- quiet(earth(matrix(xc, ncol = 1, dimnames = list(NULL, "x")), yc, pmethod = "none"))
fc2 <- quiet(earth(matrix(0.01 * xc, ncol = 1, dimnames = list(NULL, "x")), yc, pmethod = "none"))
cat("part C: x      :", rownames(fc1$dirs), "\n        x / 100:", rownames(fc2$dirs), "\n")
okC <- !(nrow(fc1$dirs) == nrow(fc2$dirs) && all(fc1$dirs == fc2$dirs) && isTRUE(all.equal(fc1$cuts, 100 * fc2$cuts)))

# Part D: a single-hinge search, its RSS, and a knot at a repeated minimum
set.seed(3409); n <- 80; XD <- matrix(runif(2 * n), n, 2); XD[, 2] <- round(XD[, 2] * 4) / 4; colnames(XD) <- c("x1", "x2")
yd <- 5 * pmax(XD[, 1] - 0.4, 0) * XD[, 2] + 2 * pmax(0.7 - XD[, 2], 0) + sin(4 * XD[, 2]) + 0.1 * rnorm(n)
out <- traced_out(fd <- quiet(earth(XD, yd, degree = 2, trace = 9, nk = 31, minspan = 1, endspan = 1, Adjust.endspan = 0,
  Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
BD <- basis(fd$dirs, fd$cuts, XD)[, 1:7]
ys <- (yd - mean(yd)) / sd(yd); rss <- function(M) sum(qr.resid(qr(M), ys)^2)
seg <- out[grep("Searching for new term 8 ", out):grep("Searching for new term 10 ", out)]
a <- grep("^\\|Parent 7  Pred 2 ", seg)[1]; e <- which(grepl("FindKnotEnd", seg) & seq_along(seg) > a)[1]
bg <- which(grepl("--FindKnotBegin--", seg) & seq_along(seg) > a)[1]
bD <- BD[, 7]; lD <- bD * XD[, 2]; AD <- sum(qr.resid(qr(BD), lD)^2)
singleD <- grepl(" iNewCol 8 ", seg[bg])
getk <- function(t) grep(sprintf("Cut +%s +bx1G", t), seg[a:e], value = TRUE)[1]
k05 <- getk(0.5); k0 <- getk(0)
rss_h <- rss(cbind(BD, bD * pmax(XD[, 2] - 0.5, 0))); rss_lh <- rss(cbind(BD, lD, bD * pmax(XD[, 2] - 0.5, 0)))
cat(sprintf("part D: parent %s with x2: A = %.5f, 1 - R^2 = %.4f, single-hinge search: %s\n", rownames(fd$dirs)[7], AD,
  AD / sum((lD - mean(lD))^2), singleD))
cat(sprintf("        knot 0.5: traced RssWithKnot %s | RSS(current + h) %.5g | RSS(current + b*x2 + h) %.5g\n",
  sub(".*RssWithKnot +([-0-9.e]+).*", "\\1", k05), rss_h, rss_lh))
cat(sprintf("        knot 0 (the repeated minimum of x2): TolG %s\n", sub(".*TolG ([01]).*", "\\1", k0)))
okD1 <- singleD && AD < 0.01 && abs(num(".*RssWithKnot +([-0-9.e]+).*", k05) - rss_h) < 1e-3 * rss_h && abs(rss_h - rss_lh) > 1e-2
okD2 <- singleD && grepl("TolG 1", k0)

# Part E: weights that are not all equal
set.seed(8); wt <- runif(200, 0.5, 2)
allw <- rbind(searches(X5, yf, w = wt, degree = 2, nk = 31), searches(X3, y3, w = runif(300, 0.5, 2), degree = 2, nk = 31))
cat(sprintf("part E (weighted): %d searches; pair searches with A < 0.01: %d (smallest A of a pair search %.3g)\n",
  nrow(allw), sum(!allw$single & allw$A < 0.01), min(allw$A[!allw$single])))

cat(sprintf("CHECK bb14.1 %s in unweighted fits a search is a pair search exactly when A >= 0.01 (%d searches; bracket %.6g, %.6g)\n",
  okA, nrow(all), max(all$A[all$single]), min(all$A[!all$single])))
cat(sprintf("CHECK bb14.2 %s 1 - R^2 of b*x does not decide the kind: single-hinge searches reach %.3g and pair searches go down to %.3g\n",
  max(all$one_minus_r2[all$single]) > min(all$one_minus_r2[!all$single]), max(all$one_minus_r2[all$single]), min(all$one_minus_r2[!all$single])))
cat(sprintf("CHECK bb14.3 %s at a fixed 1 - R^2, rescaling x2 moves A from %.6f to %.6f and the search from single-hinge to pair\n", okB, kb[1, "A"], kb[2, "A"]))
cat(sprintf("CHECK bb14.4 %s the same data with x and with x / 100 give different models, beyond the scaling of the knots\n", okC))
cat(sprintf("CHECK bb14.5 %s in a single-hinge search the RSS of a knot is RSS(current columns + h), without b*x\n", okD1))
cat(sprintf("CHECK bb14.6 %s in a single-hinge search a knot at a repeated minimum can pass the collinearity test (TolG 1)\n", okD2))
cat(sprintf("CHECK bb14.7 %s with weights that are not all equal, earth runs pair searches below A = 0.01\n", sum(!allw$single & allw$A < 0.01) > 0))
