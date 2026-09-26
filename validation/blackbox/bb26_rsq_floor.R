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
# Part C: earth's reported rss for a small RSS. Part D: earth's own rounding
# near the floor at n = 3001 and 30001. Part E: the floor with two responses.
# Part F: the statistics that earth reports as 0 or NaN at small scales of y.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

design <- function(n) {
  x <- (0:(n - 1)) / (n - 1); t0 <- x[round(0.4 * (n - 1)) + 1]; set.seed(n); o <- sample(n)
  list(x = x[o], hinge = 3 * pmax(x[o] - t0, 0), g = sin(37 * x[o]) * cos(11 * x[o]), t0 = t0)
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

# Part D: near the floor, earth's decision carries its own rounding. A grid of
# noise levels puts 1 - RSq (from the residuals of the exact pair basis) at
# 0.5 to 2 times the floor; design 2 has its knot at 0.7 and another g.
design2 <- function(n) {
  x <- (0:(n - 1)) / (n - 1); t0 <- x[round(0.7 * (n - 1)) + 1]; set.seed(n + 7); o <- sample(n)
  list(x = x[o], hinge = 2 * pmax(t0 - x[o], 0), g = cos(23 * x[o]) * sin(7 * x[o]), t0 = t0)
}
exact_ratio <- function(d, k) {   # (1 - RSq) of the exact pair basis, and that over the floor
  y <- d$hinge + k * d$g
  r <- lm.fit(cbind(1, pmax(d$x - d$t0, 0), pmax(d$t0 - d$x, 0)), y)$residuals
  u <- sum(r^2) / sum((y - mean(y))^2); c(u = u, ratio = u / (1e-10 / (length(y) - 1)))
}
scan_floor <- function(d, ratios) {
  k1 <- 1e-7 / sqrt(exact_ratio(d, 1e-7)["ratio"])
  t(sapply(ratios, function(q) { k <- k1 * sqrt(q); c(exact_ratio(d, k), stop = stops(d, k, 1, 0)) }))
}
fl3 <- 1e-10 / 3000; dev_tss <- 0; st_max <- 0; go_min <- Inf
for (w in 1:2) {
  sc3 <- scan_floor(if (w == 1) design(3001) else design2(3001), seq(0.5, 2, by = 0.01))
  st <- sc3[sc3[, "stop"] == 1, , drop = FALSE]; go <- sc3[sc3[, "stop"] == 0, , drop = FALSE]
  wrong <- c(st[st[, "u"] > fl3, "u"] - fl3, fl3 - go[go[, "u"] < fl3, "u"])
  dev_tss <- max(dev_tss, wrong); st_max <- max(st_max, st[, "ratio"]); go_min <- min(go_min, go[, "ratio"])
  cat(sprintf("part D, n = 3001, design %d: stops at up to %.3f times the floor, goes on at down to %.3f times it; %d of %d decisions against the rule\n",
    w, max(st[, "ratio"]), min(go[, "ratio"]), length(wrong), nrow(sc3)))
}
cat(sprintf("part D: the largest distance of a decision against the rule from the floor is %.3g * TSS (the floor is %.3g * TSS)\n", dev_tss, fl3))
sc30 <- scan_floor(design2(30001), c(1e-4, 1e-2, 0.5))
cat(sprintf("part D, n = 30001, design 2: at %s times the floor earth stops: %s\n", paste(signif(sc30[, "ratio"], 2), collapse = ", "),
  paste(as.logical(sc30[, "stop"]), collapse = ", ")))

# Part E: two responses with one exact hinge, plus noise k * G, at several
# scales of y. Where does earth stop at 3 terms with code 5?
n2 <- 201; x2 <- (0:(n2 - 1)) / (n2 - 1); t2 <- x2[141]; set.seed(208); x2 <- x2[sample(n2)]
H2 <- cbind(2 * pmax(t2 - x2, 0), 1.5 * pmax(t2 - x2, 0)); G2 <- cbind(cos(23 * x2) * sin(7 * x2), sin(29 * x2 + 1))
B2 <- cbind(1, pmax(x2 - t2, 0), pmax(t2 - x2, 0))
two_resp <- function(k, s) {
  Y <- s * (H2 + k * G2)
  f <- quiet(earth(matrix(x2, ncol = 1), Y, nk = 21, minspan = 1, endspan = 1, thresh = 0, pmethod = "none"))
  rss <- sum(sapply(1:2, function(j) lm.fit(B2, Y[, j])$residuals)^2)
  c(stop = nrow(f$dirs) == 3 && f$termcond == 5, rss = rss, u = rss / sum(scale(Y, scale = FALSE)^2))
}
ks <- 10^seq(-16, -2, by = 0.25); flipE <- NULL; patt <- list()
for (s in c(1e-4, 1e-3, 1e-2, 1, 1e2, 1e3)) {
  e <- t(sapply(ks, two_resp, s = s)); patt[[as.character(s)]] <- e[, "stop"]
  st <- e[e[, "stop"] == 1, , drop = FALSE]; go <- e[e[, "stop"] == 0, , drop = FALSE]
  flipE <- rbind(flipE, c(s = s, stops = nrow(st), stop_rss = if (nrow(st)) max(st[, "rss"]) else NA, go_rss = min(go[, "rss"]),
    stop_u = if (nrow(st)) max(st[, "u"]) else NA, go_u = min(go[, "u"]), switches = sum(diff(e[, "stop"]) != 0)))
}
cat("part E: two responses; per scale of y, the largest summed RSS and RSS/TSS at a stop and the smallest at a go-on, over", length(ks), "noise levels\n")
print(flipE, digits = 3)
mid <- flipE[flipE[, "s"] %in% c(1e-3, 1e-2, 1), , drop = FALSE]
absolute <- all(mid[, "switches"] == 1) && all(mid[, "stop_rss"] > 1e-11 & mid[, "stop_rss"] < 1e-10 & mid[, "go_rss"] > 1e-10 & mid[, "go_rss"] < 1e-9) &&
  max(mid[, "stop_u"]) / min(mid[, "stop_u"]) > 1e5
erratic <- flipE[flipE[, "s"] == 1e-4, "stops"] == 0 && all(flipE[flipE[, "s"] %in% c(1e2, 1e3), "switches"] >= 4)

# Part F: an ordinary fit at small scales of y
set.seed(3303); nF <- 300; XF <- matrix(runif(nF * 3), nF, 3); colnames(XF) <- paste0("x", 1:3)
yF <- sin(3 * XF[, 1]) + 0.5 * XF[, 2] + 2 * pmax(XF[, 3] - 0.6, 0) + 0.1 * rnorm(nF)
refF <- earth(XF, yF); fs <- list()
for (s in 10^(0:-8)) {
  f <- quiet(earth(XF, s * yF)); fs[[as.character(s)]] <- f
  cat(sprintf("part F: y times %-6g rss %-9.3g rsq %-7.4g gcv %-9.3g grsq %-7.4g | from the residuals: rss %.3g, gcv %.3g | zeros in rss.per.subset %d, in gcv.per.subset %d of %d | terms as at scale 1: %s\n",
    s, f$rss, f$rsq, f$gcv, f$grsq, sum(f$residuals^2), earth:::get.gcv(sum(f$residuals^2), length(f$selected.terms), 2, nF),
    sum(f$rss.per.subset == 0), sum(f$gcv.per.subset == 0), length(f$rss.per.subset), identical(f$selected.terms, refF$selected.terms)))
}
f5 <- fs[["1e-05"]]; f6 <- fs[["1e-06"]]
zeroed <- f5$rss > 0 && f5$gcv == 0 && is.nan(f5$grsq) && all(f5$gcv.per.subset == 0) && f6$rss == 0 && is.nan(f6$rsq) && all(f6$rss.per.subset == 0) &&
  all(sapply(fs, function(f) identical(f$selected.terms, refF$selected.terms))) && length(refF$selected.terms) > 1 &&
  length(refF$selected.terms) < length(refF$rss.per.subset)

cat(sprintf("CHECK bb26.1 %s at n = 21, 57, 201 and 1001 earth stops with code 5 when 1 - RSq falls below 1e-10/(n - 1), up to earth's rounding (every flip lies within 3%% of it)\n", inside))
cat(sprintf("CHECK bb26.2 %s the flip moves by less than 1%% when y is multiplied by 1e6 or 1e-6, so the rule is relative\n", scale_free))
cat(sprintf("CHECK bb26.3 %s with thresh = 1e-13 the flip stays at the floor, and with thresh = 1e-12 it moves to 1 - RSq = 1e-12\n", thresh_ok))
cat(sprintf("CHECK bb26.4 %s earth reports rss = 0 and rsq = 1 when its RSS is small in absolute terms (here up to %.3g), while the residuals give the true value\n",
  any(zero) && any(!zero) && all(rep_tab$rsq[zero] == 1) && max(rep_tab$rss_resid[zero]) < min(rep_tab$rss_resid[!zero]), max(rep_tab$rss_resid[zero])))
cat(sprintf("CHECK bb26.5 %s at n = 3001 earth's decision near the floor is not monotone: over two designs it stops at up to %.2f times the floor and goes on at down to %.2f times it, and every decision against the rule lies within %.2g * TSS of the floor, below 1e-13 * TSS\n",
  st_max > 1.1 && go_min < 1 && dev_tss < 1e-13, st_max, go_min, dev_tss))
cat(sprintf("CHECK bb26.6 %s at n = 30001 (design 2) earth does not stop by the floor at 1e-4, 0.01 or 0.5 times it\n", all(sc30[, "stop"] == 0)))
cat(sprintf("CHECK bb26.7 %s with two responses and y times 1e-3, 1e-2 and 1, earth stops with code 5 exactly when the summed RSS is below a value between 1e-11 and 1e-9 in the units of y, while RSS/TSS at the flip changes by a factor above 1e5: the floor is absolute\n", absolute))
cat(sprintf("CHECK bb26.8 %s with two responses and y times 1e-4 earth does not stop at 3 terms at any noise level, and with y times 100 and 1000 stops and go-ons alternate as the noise changes (%d and %d switches)\n",
  erratic, flipE[flipE[, "s"] == 1e2, "switches"], flipE[flipE[, "s"] == 1e3, "switches"]))
cat(sprintf("CHECK bb26.9 %s earth reports gcv 0 and grsq NaN at y times 1e-5, and rss 0 and rsq NaN at y times 1e-6, with gcv.per.subset and then rss.per.subset 0 at every size; the selected terms (%d of %d) are those at y times 1 at every scale down to 1e-8\n",
  zeroed, length(refF$selected.terms), length(refF$rss.per.subset)))
