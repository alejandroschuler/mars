# bb20: several responses in the forward pass.
#
# Data: n = 200, 5 uniform covariates, three responses that each depend
# mainly on a different covariate (x1, x2, x3); x4, x5 are noise.
#
# 1. At each forward step, is the chosen candidate the one with the largest
#    reduction of the RSS summed over the responses? Compared two ways: the
#    smallest RSS logged at trace=9 among every candidate scanned at that
#    step, against sum over responses of the lm.fit RSS of the basis that
#    was actually built (reconstructed from dirs/cuts, on earth's internal
#    scale, which bb17 found equals the raw scale here: no per-response
#    rescaling with the defaults).
# 2. Are dirs, cuts and the pruning path for cbind(y1,y2,y3) different from
#    each single-response fit (they should be, if the basis is shared)?
# 3. Does a 2-D y with one column give the same fit as the vector y (dirs,
#    cuts, coefficients, rss, gcv)?
# 4. With thresh = 0.05, does the forward pass stop by the RSq-change rule,
#    and does the per-step RSq printed at trace = 2 equal
#    1 - sum(rss)/sum(tss) summed over responses?
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

set.seed(20260925)
n <- 200; p <- 5
X <- matrix(runif(n * p), n, p)
colnames(X) <- paste0("x", 1:p)
y1 <- 3 * pmax(X[, 1] - 0.4, 0) + 0.2 * rnorm(n)
y2 <- 2 * pmax(0.6 - X[, 2], 0) + 0.2 * rnorm(n)
y3 <- 4 * X[, 3] + 0.2 * rnorm(n)
Y <- cbind(y1 = y1, y2 = y2, y3 = y3)

## ================= Part 1: chosen candidate = largest combined reduction =================
cat("== Part 1: chosen candidate has the largest reduction of the summed RSS ==\n")

# Reconstruct the basis matrix from dirs/cuts, in construction order (bb11's convention).
basis <- function(f, Xmat) sapply(seq_len(nrow(f$dirs)), function(k) {
  v <- rep(1, nrow(Xmat))
  for (j in which(f$dirs[k, ] != 0)) v <- v * switch(as.character(f$dirs[k, j]),
    "1" = pmax(Xmat[, j] - f$cuts[k, j], 0), "-1" = pmax(f$cuts[k, j] - Xmat[, j], 0), "2" = Xmat[, j])
  v
})
combined_rss <- function(Bm, Ymat) sum(sapply(seq_len(ncol(Ymat)), function(j) sum(lm.fit(Bm, Ymat[, j])$residuals^2)))

out <- capture.output(f1 <- quiet(earth(X, Y, degree = 1, trace = 9, nk = 11,
  minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0)))

B <- basis(f1, X)

# With several responses, whether a step is a single hinge or a mirrored
# pair depends on which predictor actually wins that step (a variable
# already used as a hinge parent earlier can be searched as a single from
# then on; bb11.5's "iNewCol == K" rule holds per candidate, but different
# predictors in the SAME step can disagree on it). So the width of each
# step is read directly off the winning path in dirs/cuts: two consecutive
# rows are one pair exactly when they share one variable with opposite
# +-1 codes and the same cut; otherwise a row is a lone single.
ft <- grep("FindTerm: Searching for new term", out)
ends <- c(ft[-1] - 1, length(out))
M <- nrow(f1$dirs)
is_pair <- sapply(seq_len(M - 1), function(k) {
  j1 <- which(f1$dirs[k, ] != 0); j2 <- which(f1$dirs[k + 1, ] != 0)
  length(j1) == 1 && length(j2) == 1 && j1 == j2 &&
    abs(f1$dirs[k, j1]) == 1 && f1$dirs[k, j1] == -f1$dirs[k + 1, j2] &&
    isTRUE(all.equal(f1$cuts[k, j1], f1$cuts[k + 1, j2]))
})
cum_terms <- integer(0); row <- 2L
while (row <= M) {
  width <- if (row <= length(is_pair) && is_pair[row]) 2L else 1L
  cum_terms <- c(cum_terms, row + width - 1L)
  row <- row + width
}
cat(sprintf("  %d forward steps in dirs (widths: %s), %d steps logged at trace=9\n",
  length(cum_terms), paste(diff(c(1L, cum_terms)), collapse = ","), length(ft)))

step_min <- sapply(seq_along(ft), function(i) {
  seg <- out[ft[i]:ends[i]]
  # "|Parent K ... skip (degree of term would be 2)" lines have no Rss field
  # at all (a non-intercept parent is skipped under degree=1): exclude them.
  par_lines <- grep("^\\|Parent", seg, value = TRUE)
  par_lines <- par_lines[grepl("Rss", par_lines)]
  rss_par  <- as.numeric(sub(".*[^a-zA-Z]Rss +([-+0-9.e]+).*", "\\1", par_lines))
  rss_knot <- as.numeric(sub(".*RssWithKnot +([-+0-9.e]+).*", "\\1", grep("RssWithKnot", seg, value = TRUE)))
  min(c(rss_par, rss_knot))
})
rebuilt <- sapply(cum_terms, function(m) combined_rss(B[, seq_len(m), drop = FALSE], Y))
match_ok <- abs(step_min - rebuilt) / pmax(abs(rebuilt), 1e-8) < 1e-3
cat("step  logged_min_RSS  rebuilt_lm.fit_RSS  match\n")
print(data.frame(step = seq_along(step_min), logged_min = signif(step_min, 7),
  rebuilt = signif(rebuilt, 7), match = match_ok), row.names = FALSE)
ok_item1 <- all(match_ok) && length(match_ok) > 0

## ================= Part 2: shared basis differs from each single-response fit =================
cat("\n== Part 2: combined fit vs each single-response fit ==\n")
fm <- quiet(earth(X, Y, degree = 1))
fy1 <- quiet(earth(X, y1, degree = 1))
fy2 <- quiet(earth(X, y2, degree = 1))
fy3 <- quiet(earth(X, y3, degree = 1))
cat(sprintf("combined:  %d forward terms, %d selected: %s\n", nrow(fm$dirs), length(fm$selected.terms),
  paste(rownames(fm$dirs)[fm$selected.terms], collapse = ", ")))
cat(sprintf("y1 alone:  %d forward terms, %d selected: %s\n", nrow(fy1$dirs), length(fy1$selected.terms),
  paste(rownames(fy1$dirs)[fy1$selected.terms], collapse = ", ")))
cat(sprintf("y2 alone:  %d forward terms, %d selected: %s\n", nrow(fy2$dirs), length(fy2$selected.terms),
  paste(rownames(fy2$dirs)[fy2$selected.terms], collapse = ", ")))
cat(sprintf("y3 alone:  %d forward terms, %d selected: %s\n", nrow(fy3$dirs), length(fy3$selected.terms),
  paste(rownames(fy3$dirs)[fy3$selected.terms], collapse = ", ")))
differs_from_all <- !identical(fm$dirs, fy1$dirs) && !identical(fm$dirs, fy2$dirs) && !identical(fm$dirs, fy3$dirs)
gcv_path_differs <- !isTRUE(all.equal(fm$gcv.per.subset, fy1$gcv.per.subset)) &&
  !isTRUE(all.equal(fm$gcv.per.subset, fy2$gcv.per.subset)) &&
  !isTRUE(all.equal(fm$gcv.per.subset, fy3$gcv.per.subset))
cat(sprintf("combined dirs differs from every single-response dirs: %s ; gcv.per.subset (the pruning path) differs too: %s\n",
  differs_from_all, gcv_path_differs))

## ================= Part 3: 1-column matrix y vs vector y =================
cat("\n== Part 3: 1-column matrix y vs vector y ==\n")
f_vec <- quiet(earth(X, y1, degree = 1))
f_mat <- quiet(earth(X, matrix(y1, ncol = 1), degree = 1))
same_onecol <- identical(f_vec$dirs, f_mat$dirs) && isTRUE(all.equal(f_vec$cuts, f_mat$cuts)) &&
  isTRUE(all.equal(f_vec$coefficients, f_mat$coefficients, check.attributes = FALSE)) &&
  isTRUE(all.equal(f_vec$rss, f_mat$rss)) && isTRUE(all.equal(f_vec$gcv, f_mat$gcv)) &&
  identical(f_vec$selected.terms, f_mat$selected.terms)
cat(sprintf("dirs/cuts/coefficients/rss/gcv/selected.terms all identical: %s\n", same_onecol))

## ================= Part 4: stopping rules with several responses =================
cat("\n== Part 4: thresh = 0.05, several responses ==\n")
out4 <- capture.output(f4 <- quiet(earth(X, Y, degree = 1, pmethod = "none", nk = 21, trace = 2, thresh = 0.05)))
term_msg <- grep("RSq changed|Reached|GRSq|No new term", out4, value = TRUE)
term_msg <- trimws(term_msg[!grepl("^After|Forward pass complete", term_msg)])
cat("termination message:", paste(term_msg, collapse = " / "), " | termcond:", f4$termcond, "\n")

b <- grep("GRSq +RSq", out4)[1]
blank_after <- grep("^$", out4); blank_after <- min(blank_after[blank_after > b])
rows <- out4[(b + 1):(blank_after - 1)]
step_tab <- do.call(rbind, lapply(rows, function(ln) {
  tok <- strsplit(trimws(ln), "\\s+")[[1]]
  if (length(tok) < 10) return(NULL)
  if (grepl("reject", ln)) return(NULL)  # a rejected candidate is never added to dirs
  data.frame(RSq = as.numeric(tok[3]), Terms = as.integer(tok[8]))
}))

B4 <- basis(f4, X)
tss4 <- sum(apply(Y, 2, function(v) sum((v - mean(v))^2)))
# earth's printed "Terms" counts forward terms only, excluding the intercept
# row of dirs, so the cumulative basis width is Terms + 1.
pred_rsq <- sapply(step_tab$Terms + 1L, function(m) 1 - combined_rss(B4[, seq_len(m), drop = FALSE], Y) / tss4)
rsq_diff <- abs(step_tab$RSq - pred_rsq)
cat("Terms  printed_RSq  1-sum(rss)/sum(tss)  abs_diff\n")
print(data.frame(Terms = step_tab$Terms, printed_RSq = step_tab$RSq, predicted = signif(pred_rsq, 6),
  abs_diff = signif(rsq_diff, 3)), row.names = FALSE)
ok_rsq_rule <- length(rsq_diff) > 0 && all(rsq_diff < 2e-4)
ok_thresh_termcond <- isTRUE(f4$termcond == 4) && any(grepl("RSq changed", term_msg))

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb20.1 %s at every forward step (%d steps), the smallest RSS logged at trace=9 across all scanned candidates equals sum over responses of the lm.fit RSS of the term actually added\n",
  ok_item1, length(match_ok)))
cat(sprintf("CHECK bb20.2 %s dirs for cbind(y1,y2,y3) differs from every single-response dirs, and so does the pruning path (gcv.per.subset)\n",
  differs_from_all && gcv_path_differs))
cat(sprintf("CHECK bb20.3 %s a 1-column matrix y gives a fit identical to the vector y in dirs, cuts, coefficients, rss and gcv\n",
  same_onecol))
cat(sprintf("CHECK bb20.4 %s with thresh=0.05 several responses stop by the RSq-change rule (termcond 4) and the per-step printed RSq equals 1 - sum(rss)/sum(tss) summed over responses (max abs diff %.2g over %d rows)\n",
  ok_thresh_termcond && ok_rsq_rule, if (length(rsq_diff)) max(rsq_diff) else NA, length(rsq_diff)))
