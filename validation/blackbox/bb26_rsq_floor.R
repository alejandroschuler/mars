# bb26: earth's code-5 stop below an exact fit.
#
# Hypothesis: besides RSq >= 1 - thresh, earth stops with code 5 when
# RSS/TSS < 1e-10/(n - 1), that is, when the RSS of y standardized to unit
# variance (divisor n - 1) is below 1e-10. The design is an exact hinge at a
# grid point plus noise k*g(x); for each n a bisection on k finds where the
# pass flips from stopping after the first pair (3 terms, code 5) to going on.
# The script reports 1 - RSq after the first pair on both sides of the flip.
# It also checks that the flip does not move when y is multiplied by 1e6 or
# 1e-6, and that thresh = 1e-13 acts as 0 while thresh = 1e-12 moves the flip.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

design <- function(n) {
  x <- (0:(n - 1)) / (n - 1); t0 <- x[round(0.4 * (n - 1)) + 1]; set.seed(n); o <- sample(n)
  list(x = x[o], hinge = 3 * pmax(x[o] - t0, 0), g = sin(37 * x[o]) * cos(11 * x[o]))
}
ratio_after_pair <- function(d, k, scale) {        # 1 - RSq after the first step
  y <- scale * (d$hinge + k * d$g)
  f <- quiet(earth(matrix(d$x, ncol = 1), y, nk = 3, minspan = 1, endspan = 1, thresh = 0, pmethod = "none"))
  sum(f$residuals^2) / sum((y - mean(y))^2)    # f$rss can be reported as 0 (part C)
}
stops <- function(d, k, scale, thresh) {
  y <- scale * (d$hinge + k * d$g)
  f <- quiet(earth(matrix(d$x, ncol = 1), y, nk = 21, minspan = 1, endspan = 1, thresh = thresh, pmethod = "none"))
  nrow(f$dirs) == 3 && f$termcond == 5
}
flip <- function(n, scale = 1, thresh = 0) {
  d <- design(n); lo <- log(1e-12); hi <- log(1e-3)   # stop at lo, go on at hi
  for (i in 1:40) { mid <- (lo + hi) / 2; if (stops(d, exp(mid), scale, thresh)) lo <- mid else hi <- mid }
  c(n = n, below = ratio_after_pair(d, exp(lo), scale), above = ratio_after_pair(d, exp(hi), scale), floor = 1e-10 / (n - 1))
}
tab <- rbind(flip(21), flip(57), flip(201), flip(1001))
print(tab, digits = 5)
# earth tests its own RSS of the standardized y, which carries rounding of a
# few times n * eps relative to the floor here, so the flip is compared within 3%.
rel <- abs(tab[, "below"] / tab[, "floor"] - 1)
cat(sprintf("relative distance of the flip from 1e-10/(n - 1): %s\n", paste(sprintf("%.2g", rel), collapse = ", ")))
inside <- all(rel < 0.03) && all(tab[, "above"] / tab[, "below"] < 1.001)
sc <- rbind(flip(201, 1e6), flip(201, 1e-6))
cat("n = 201 with y times 1e6 and times 1e-6:\n"); print(sc, digits = 5)
scale_free <- all(abs(sc[, "below"] / tab[3, "below"] - 1) < 1e-2)
th <- rbind(flip(201, thresh = 1e-13), flip(201, thresh = 1e-12))
cat("n = 201 with thresh 1e-13 and 1e-12:\n"); print(th, digits = 5)
thresh_ok <- abs(th[1, "below"] / tab[3, "below"] - 1) < 1e-3 && abs(th[2, "below"] / 1e-12 - 1) < 1e-2

# Part C: earth reports rss = 0 and rsq = 1 when the RSS is small in absolute
# terms, although its residuals are not 0.
d <- design(201); rep_tab <- NULL
for (k in 10^seq(-9, -2, by = 0.25)) {
  y <- d$hinge + k * d$g
  f <- quiet(earth(matrix(d$x, ncol = 1), y, nk = 3, minspan = 1, endspan = 1, thresh = 0, pmethod = "none"))
  rep_tab <- rbind(rep_tab, data.frame(rss_resid = sum(f$residuals^2), rss_reported = f$rss, rsq = f$rsq))
}
zero <- rep_tab$rss_reported == 0
cat(sprintf("part C: earth reports rss 0 for residual sums of squares up to %.3g and the true value from %.3g up (TSS %.4g)\n",
  max(rep_tab$rss_resid[zero]), min(rep_tab$rss_resid[!zero]), sum((d$hinge - mean(d$hinge))^2)))

cat(sprintf("CHECK bb26.1 %s at n = 21, 57, 201 and 1001 earth stops with code 5 when 1 - RSq falls below 1e-10/(n - 1), up to earth's rounding (every flip lies within 3%% of it)\n", inside))
cat(sprintf("CHECK bb26.2 %s the flip moves by less than 1%% when y is multiplied by 1e6 or 1e-6, so the rule is relative\n", scale_free))
cat(sprintf("CHECK bb26.3 %s with thresh = 1e-13 the flip stays at the floor, and with thresh = 1e-12 it moves to 1 - RSq = 1e-12\n", thresh_ok))
cat(sprintf("CHECK bb26.4 %s earth reports rss = 0 and rsq = 1 when its RSS is small in absolute terms (here up to %.3g), while the residuals give the true value\n",
  any(zero) && any(!zero) && all(rep_tab$rsq[zero] == 1) && max(rep_tab$rss_resid[zero]) < min(rep_tab$rss_resid[!zero]), max(rep_tab$rss_resid[zero])))
