# bb30: the termination code when the step's GRSq' is below -10 (STOP-3,
# issue #44, T06 question 8). The hypothesis of spec v1: code 2 only when
# GRSq' = -Inf (C(M') >= n), else code 3. The alternative: code 2 whenever
# GRSq' < -1000.
# A. x = 0..7, y alternating +1/-1, the defaults otherwise. The step-1 pair has
#    M' = 3; its RSS' comes from a run at thresh 0 with nk 3 and pmethod none
#    (the forward choice does not depend on thresh or penalty, bb12.1). A scan of
#    the penalty moves GRSq' = 1 - GCV(RSS', 3)/GCV(TSS, 1) across -1000.
# B. A step without a legal candidate: two constant columns, y = 0..9, so
#    GRSq' = GRSq(TSS, 2); a scan of the penalty moves it across -1000.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
gcv <- function(rss, m, pen, n) earth:::get.gcv(rss, m, pen, n)
scan <- function(X, y, rss1, m1, pens) {
  n <- length(y); tss <- sum((y - mean(y))^2)
  t(sapply(pens, function(pen) {
    f <- earth(X, y, penalty = pen)
    g <- 1 - gcv(rss1, m1, pen, n) / gcv(tss, 1, pen, n)
    cat(sprintf("  penalty %-7g C(M') %-8g GRSq' %-12.6g termcond %d\n", pen, m1 + pen * (m1 - 1) / 2, g, f$termcond))
    c(pen = pen, grsq = g, code = f$termcond)
  }))
}
cat("Part A\n")
XA <- matrix(0:7, ncol = 1, dimnames = list(NULL, "x")); yA <- rep(c(1, -1), 4)
f0 <- earth(XA, yA, thresh = 0, nk = 3, pmethod = "none")
cat(sprintf("  step-1 model at thresh 0: %d terms, RSS' %.10g\n", nrow(f0$dirs), f0$rss))
pensA <- c(4.0, 4.7, 4.75, 4.78, 4.79, 4.8, 4.9)
rA <- scan(XA, yA, f0$rss, nrow(f0$dirs), pensA)
cat("Part B\n")
XB <- cbind(a = rep(1, 10), b = rep(2, 10)); yB <- 0:9
pensB <- c(10, 14, 15, 15.4, 15.5, 15.6, 16)
rB <- scan(XB, yB, sum((yB - mean(yB))^2), 2, pensB)
r <- rbind(rA, rB)
fin <- is.finite(r[, "grsq"])
cat(sprintf("CHECK bb30.1 %s with a finite GRSq' below -10, earth gives code 2 exactly when GRSq' < -1000 and code 3 otherwise (%d runs)\n",
  all(r[r[, "grsq"] < -10, "code"] == ifelse(r[r[, "grsq"] < -10, "grsq"] < -1000, 2, 3)), sum(r[, "grsq"] < -10)))
cat(sprintf("CHECK bb30.2 %s code 2 occurs with a finite GRSq' (so -Inf is not the condition)\n",
  any(fin & r[, "code"] == 2)))
cat(sprintf("CHECK bb30.3 %s the same switch applies to a step without a legal candidate (Part B)\n",
  all(rB[rB[, "grsq"] < -10, "code"] == ifelse(rB[rB[, "grsq"] < -10, "grsq"] < -1000, 2, 3)) && any(rB[, "code"] == 2) && any(rB[, "code"] == 3)))
cat(sprintf("CHECK bb30.4 %s HYPOTHESIS code 2 only when GRSq' = -Inf\n", all((r[, "code"] == 2) == !fin)))
