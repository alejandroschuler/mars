# bb02_gcv_fitted.R
#
# Question: what does earth actually feed into get.gcv() for a fitted model, and how
# do rsq/grsq, sample weights, and multiple response columns interact with rss/gcv
# bookkeeping (rss.per.subset, gcv.per.subset, rss.per.response, gcv.per.response)?
#
# Data: Friedman #1, n=200, 5 covariates U(0,1), y = 10 sin(pi x1 x2) + 20(x3-0.5)^2
#       + 10 x4 + 5 x5 + N(0,1).
#
# NOTE on fit$bx vs fit$dirs indexing (discovered while writing this script): dirs/cuts
# record every candidate hinge the forward pass considered, including "mirror" partners
# that turn out to be linearly redundant given their sibling and are never actually put
# in the regression basis. fit$bx only has columns for the real (non-redundant) terms,
# so it can have FEWER columns than dirs has rows, and its columns are not at the same
# index as the matching dirs row. We build an explicit name-based map (bx colnames vs
# dirs rownames) before ever indexing bx by a dirs-space index (selected.terms, or the
# intercept row).

cat(R.version.string, "\n")
cat("earth", as.character(packageVersion("earth")), "\n\n")

suppressPackageStartupMessages(library(earth))

safe_get_gcv <- function(rss, nterms, penalty, n) {
  tryCatch(earth:::get.gcv(rss, nterms, penalty, n), error = function(e) NA_real_)
}
maxreldiff <- function(a, b) max(abs(a - b) / abs(b))

# map from dirs-space row index (character) -> bx column index
dirs_to_bx_map <- function(fit) {
  m <- match(colnames(fit$bx), rownames(fit$dirs))
  if (anyNA(m)) cat("  ! warning: a bx column did not match any dirs row by name\n")
  setNames(seq_along(m), m)
}
bx_cols_for <- function(map, dirs_idx) {
  out <- map[as.character(dirs_idx)]
  if (anyNA(out)) cat("  ! warning: dirs index has no corresponding (non-redundant) bx column\n")
  unname(out)
}
intercept_dirs_row <- function(fit) {
  idx <- which(apply(fit$dirs, 1, function(r) all(r == 0)))
  if (length(idx) != 1) { cat("  ! unexpected: ", length(idx), " all-zero rows in dirs, using first\n"); idx <- idx[1] }
  idx
}

## ---- data: Friedman #1 --------------------------------------------------
set.seed(1)
n <- 200
x <- matrix(runif(n * 5), n, 5)
colnames(x) <- paste0("x", 1:5)
y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5] + rnorm(n)

## ================= Part 1: four fits =================
fit_a <- earth(x, y, degree = 1)
fit_b <- earth(x, y, degree = 2)

## fit (c), attempt 1: a predictor that is EXACTLY linear in the DGP (x1, coefficient 3)
## plus a hinge in x2. Hypothesis: Auto.linpreds gives x1 a dirs code of 2.
set.seed(2)
xc0 <- matrix(runif(n * 2), n, 2); colnames(xc0) <- c("x1", "x2")
yc0 <- 3 * xc0[, 1] + 5 * pmax(xc0[, 2] - 0.5, 0) + 0.01 * rnorm(n)
fit_c0 <- earth(xc0, yc0, degree = 1, Auto.linpreds = TRUE)
c0_has_2 <- any(fit_c0$dirs == 2)
cat("fit(c) attempt 1 (continuous-linear x1) dirs:\n"); print(fit_c0$dirs)
cat("attempt 1 dirs contains a 2:", c0_has_2, "(expected TRUE; a merely-linear continuous x1 still gets hinged)\n\n")

## fit (c), corrected: Auto.linpreds is about predictors that CANNOT take an interior
## knot, e.g. a two-level (0/1) predictor -- try that instead.
set.seed(2)
x1_bin <- rbinom(n, 1, 0.5)
x2c <- runif(n)
yc <- 3 * x1_bin + 5 * pmax(x2c - 0.5, 0) + 0.01 * rnorm(n)
xc <- cbind(x1 = x1_bin, x2 = x2c)
fit_c <- earth(xc, yc, degree = 1, Auto.linpreds = TRUE)
fit_c_off <- earth(xc, yc, degree = 1, Auto.linpreds = FALSE)
c_has_2 <- any(fit_c$dirs == 2)
cat("fit(c) corrected (two-level x1) dirs, Auto.linpreds=TRUE:\n"); print(fit_c$dirs)
cat("fit(c) corrected dirs, Auto.linpreds=FALSE (for comparison):\n"); print(fit_c_off$dirs)
cat("corrected dirs contains a 2 (Auto.linpreds=TRUE):", c_has_2, "\n")
cat("corrected dirs contains a 2 (Auto.linpreds=FALSE):", any(fit_c_off$dirs == 2),
    "(x1 is still coded linear -- row name has no h(...) -- just with a 1, not a 2)\n\n")

## fit (d): a fit where the forward pass adds a single hinge instead of a pair
set.seed(3)
xd <- matrix(runif(n * 1), n, 1); colnames(xd) <- "x1"
yd <- pmax(xd[, 1] - 0.3, 0) + 2 * pmax(xd[, 1] - 0.7, 0) + 0.05 * rnorm(n)
trace_d <- capture.output(
  fit_d <- earth(xd, yd, degree = 1, minspan = 1, endspan = 1, Auto.linpreds = FALSE,
                 pmethod = "none", trace = 2)
)

fits <- list(a = fit_a, b = fit_b, c = fit_c, d = fit_d)
ys   <- list(a = y,     b = y,     c = yc,    d = yd)

cat("---- Part 1: gcv.per.subset formula, rsq, grsq for fits a-d ----\n")
gcv_ok <- rsq_ok <- grsq_ok <- logical(0)
for (nm in names(fits)) {
  fit <- fits[[nm]]; yy <- ys[[nm]]
  K <- length(fit$gcv.per.subset)
  predicted <- vapply(seq_len(K), function(k) safe_get_gcv(fit$rss.per.subset[k], k, fit$penalty, n), numeric(1))
  rd <- maxreldiff(fit$gcv.per.subset, predicted)
  gcv_ok[nm] <- rd < 1e-12

  tss <- sum((yy - mean(yy))^2)
  cand_rsq <- 1 - fit$rss / tss
  rsq_ok[nm] <- abs(fit$rsq - cand_rsq) / abs(cand_rsq) < 1e-10

  gcv_null <- safe_get_gcv(tss, 1, fit$penalty, n)
  cand_grsq <- 1 - fit$gcv / gcv_null
  grsq_ok[nm] <- abs(fit$grsq - cand_grsq) / abs(cand_grsq) < 1e-10

  cat(sprintf("fit(%s): penalty=%.4g K=%d  gcv.per.subset maxreldiff=%.3g | rsq rd=%.3g | grsq rd=%.3g\n",
              nm, fit$penalty, K, rd,
              abs(fit$rsq - cand_rsq) / abs(cand_rsq), abs(fit$grsq - cand_grsq) / abs(cand_grsq)))
}

cat("\nfit(d) dirs:\n")
print(fit_d$dirs)
cat("fit(d) termcond:", fit_d$termcond, "\n")
d_nrow_minus1_odd <- (nrow(fit_d$dirs) - 1) %% 2 == 1
cat("naive test - nrow(dirs)-1 odd (would prove a single-hinge step, but misses an EVEN count of them):",
    d_nrow_minus1_odd, "\n")
cat("fit(d) trace=2 step table:\n")
print(trace_d[!grepl("^(x\\[|y\\[|^$)", trace_d)])

describe_terms <- function(fit, label) {
  sel <- fit$selected.terms
  dirs_sel <- fit$dirs[sel, , drop = FALSE]
  cuts_sel <- fit$cuts[sel, , drop = FALSE]
  is_intercept <- apply(dirs_sel, 1, function(r) all(r == 0))
  has_hinge <- apply(dirs_sel, 1, function(r) any(r == 1 | r == -1))
  has_linear <- apply(dirs_sel, 1, function(r) any(r == 2))
  n_hinge <- sum(has_hinge & !is_intercept)
  n_linear <- sum(has_linear & !has_hinge & !is_intercept)
  hinge_idx <- which(dirs_sel == 1 | dirs_sel == -1, arr.ind = TRUE)
  distinct_knots <- if (length(hinge_idx) == 0) 0 else nrow(unique(cbind(var = hinge_idx[, 2], cut = cuts_sel[hinge_idx])))
  cat(sprintf("fit(%s) selected model: M=%d  hinge_terms=%d  linear_terms=%d  distinct_knots=%d\n",
              label, length(sel), n_hinge, n_linear, distinct_knots))
  c(M = length(sel), knots = distinct_knots)
}
cat("\n")
desc_c <- describe_terms(fit_c, "c")
desc_d <- describe_terms(fit_d, "d")
d_fewer_terms_than_2x_knots <- unname((desc_d["M"] - 1) < 2 * desc_d["knots"])
cat("corrected test - fit(d): (M-1) < 2*distinct_knots (proves at least one single-hinge step):",
    d_fewer_terms_than_2x_knots, "\n")

## ================= Part 2: weights =================
cat("\n---- Part 2: weights ----\n")
w_all2 <- rep(2, n)
w_int  <- rep(1:3, length.out = n)
w_int10 <- 10 * w_int
wsets <- list(all2 = w_all2, int = w_int, int10 = w_int10)
wfits <- lapply(wsets, function(w) earth(x, y, degree = 1, weights = w))

cat(sprintf("mean(w): all2=%.6g int=%.6g int10=%.6g\n", mean(w_all2), mean(w_int), mean(w_int10)))

n_nrowx_ok <- n_sumw_ok <- character(0)
for (nm in names(wfits)) {
  fit <- wfits[[nm]]; w <- wsets[[nm]]
  K <- length(fit$gcv.per.subset)
  pred_nrowx <- vapply(seq_len(K), function(k) safe_get_gcv(fit$rss.per.subset[k], k, fit$penalty, nrow(x)), numeric(1))
  pred_sumw  <- vapply(seq_len(K), function(k) safe_get_gcv(fit$rss.per.subset[k], k, fit$penalty, sum(w)), numeric(1))
  rd_nrowx <- maxreldiff(fit$gcv.per.subset, pred_nrowx)
  rd_sumw  <- maxreldiff(fit$gcv.per.subset, pred_sumw)
  cat(sprintf("weights=%-5s: maxreldiff with n=nrow(x)=%d -> %.3g | with n=sum(w)=%.4g -> %.3g\n",
              nm, nrow(x), rd_nrowx, sum(w), rd_sumw))
  n_nrowx_ok[nm] <- rd_nrowx < 1e-10
  n_sumw_ok[nm] <- rd_sumw < 1e-10
}

cat("\nDoes a CONSTANT weight vector (all2) give a fit identical to no weights at all?\n")
const_w_is_unweighted <- isTRUE(all.equal(wfits$all2$rss, fit_a$rss)) &&
  isTRUE(all.equal(wfits$all2$gcv, fit_a$gcv)) &&
  isTRUE(all.equal(wfits$all2$rsq, fit_a$rsq)) &&
  identical(wfits$all2$dirs, fit_a$dirs) &&
  identical(wfits$all2$selected.terms, fit_a$selected.terms)
cat(sprintf("  fit_a (unweighted) rss=%.10g rsq=%.10g | wfits$all2 rss=%.10g rsq=%.10g | identical: %s\n",
            fit_a$rss, fit_a$rsq, wfits$all2$rss, wfits$all2$rsq, const_w_is_unweighted))

cat("\nrss.per.subset vs weighted RSS from lm(y ~ bx - 1, weights=w), at k=1 and k=selected:\n")
rss_raw_ok <- rss_normalized_ok <- logical(0)
for (nm in names(wfits)) {
  fit <- wfits[[nm]]; w <- wsets[[nm]]
  map <- dirs_to_bx_map(fit)
  sel <- fit$selected.terms
  configs <- list(list(k = 1, cols = bx_cols_for(map, intercept_dirs_row(fit))),
                  list(k = length(sel), cols = bx_cols_for(map, sel)))
  ok_raw <- ok_norm <- logical(0)
  for (cfg in configs) {
    bxk <- fit$bx[, cfg$cols, drop = FALSE]
    lmf <- lm(y ~ bxk - 1, weights = w)
    yhat <- fitted(lmf)
    rss_w <- sum(w * (y - yhat)^2)
    rss_w_mw <- rss_w / mean(w)
    actual <- fit$rss.per.subset[cfg$k]
    rd_w <- abs(actual - rss_w) / abs(rss_w)
    rd_mw <- abs(actual - rss_w_mw) / abs(rss_w_mw)
    cat(sprintf("  weights=%-5s k=%-3d actual=%.10g  sum(w*resid^2)=%.10g (rd=%.3g)  /mean(w)=%.10g (rd=%.3g)\n",
                nm, cfg$k, actual, rss_w, rd_w, rss_w_mw, rd_mw))
    ok_raw <- c(ok_raw, rd_w < 1e-8); ok_norm <- c(ok_norm, rd_mw < 1e-8)
  }
  rss_raw_ok[nm] <- all(ok_raw); rss_normalized_ok[nm] <- all(ok_norm)
}

cat("\ntss behind rsq: weighted sum(w*(y-weighted.mean(y,w))^2), undivided vs /mean(w):\n")
tss_raw_ok <- tss_normalized_ok <- logical(0)
for (nm in names(wfits)) {
  fit <- wfits[[nm]]; w <- wsets[[nm]]
  wmy <- weighted.mean(y, w)
  tss_w <- sum(w * (y - wmy)^2)
  tss_w_mw <- tss_w / mean(w)
  cand_raw <- 1 - fit$rss / tss_w
  cand_norm <- 1 - fit$rss / tss_w_mw
  cat(sprintf("  weights=%-5s actual rsq=%.10g  cand(tss undiv)=%.10g  cand(tss/mean(w))=%.10g\n",
              nm, fit$rsq, cand_raw, cand_norm))
  tss_raw_ok[nm] <- abs(fit$rsq - cand_raw) < 1e-8
  tss_normalized_ok[nm] <- abs(fit$rsq - cand_norm) < 1e-8
}

cat("\nscaling weights by 10 (int -> int10): structure and value invariance:\n")
struct_same <- identical(wfits$int$dirs, wfits$int10$dirs) &&
  identical(wfits$int$cuts, wfits$int10$cuts) &&
  identical(wfits$int$selected.terms, wfits$int10$selected.terms)
rsq_diff <- abs(wfits$int$rsq - wfits$int10$rsq)
grsq_diff <- abs(wfits$int$grsq - wfits$int10$grsq)
rss_diff <- abs(wfits$int$rss - wfits$int10$rss) / abs(wfits$int$rss)
gcv_diff <- abs(wfits$int$gcv - wfits$int10$gcv) / abs(wfits$int$gcv)
rss_ratio <- wfits$int10$rss / wfits$int$rss
gcv_ratio <- wfits$int10$gcv / wfits$int$gcv
cat(sprintf("  dirs/cuts/selected.terms identical: %s\n", struct_same))
cat(sprintf("  |rsq diff|=%.3g  |grsq diff|=%.3g  |rss reldiff|=%.3g  |gcv reldiff|=%.3g  rss ratio=%.6g  gcv ratio=%.6g\n",
            rsq_diff, grsq_diff, rss_diff, gcv_diff, rss_ratio, gcv_ratio))

## ================= Part 3: penalty -1 and 0 =================
cat("\n---- Part 3: penalty=-1 and penalty=0 on fit (a)'s data ----\n")
fit_pm1 <- earth(x, y, degree = 1, penalty = -1)
fit_p0  <- earth(x, y, degree = 1, penalty = 0)
rd_pm1 <- maxreldiff(fit_pm1$gcv.per.subset, fit_pm1$rss.per.subset / n)
K0 <- length(fit_p0$gcv.per.subset)
cand_p0 <- vapply(seq_len(K0), function(k) fit_p0$rss.per.subset[k] / (n * (1 - k / n)^2), numeric(1))
rd_p0 <- maxreldiff(fit_p0$gcv.per.subset, cand_p0)
cat(sprintf("penalty=-1: maxreldiff(gcv.per.subset, rss.per.subset/n) = %.3g\n", rd_pm1))
cat(sprintf("penalty=0:  maxreldiff(gcv.per.subset, rss/(n*(1-k/n)^2)) = %.3g\n", rd_p0))

## ================= Part 4: three response columns =================
cat("\n---- Part 4: three response columns ----\n")
set.seed(1)  # reproduce the same noise draw used above for column 3
y3 <- cbind(y, 2 * y + x[, 1], sin(3 * x[, 2]) + 0.1 * rnorm(n))
fit_m <- earth(x, y3, degree = 1)
M <- length(fit_m$selected.terms)
cat(sprintf("fit_m: M=%d penalty=%.4g\n", M, fit_m$penalty))

rss_sum_ok <- abs(fit_m$rss - sum(fit_m$rss.per.response)) / abs(fit_m$rss) < 1e-10
cat(sprintf("rss=%.10g  sum(rss.per.response)=%.10g  match: %s\n",
            fit_m$rss, sum(fit_m$rss.per.response), rss_sum_ok))

gcv_sum_ok <- abs(fit_m$gcv - sum(fit_m$gcv.per.response)) / abs(fit_m$gcv) < 1e-10
gcv_per_resp_pred <- vapply(seq_len(ncol(y3)), function(j) safe_get_gcv(fit_m$rss.per.response[j], M, fit_m$penalty, n), numeric(1))
gcv_per_resp_ok <- maxreldiff(fit_m$gcv.per.response, gcv_per_resp_pred) < 1e-10
cat(sprintf("gcv=%.10g  sum(gcv.per.response)=%.10g  match: %s\n", fit_m$gcv, sum(fit_m$gcv.per.response), gcv_sum_ok))
cat(sprintf("gcv.per.response == get.gcv(rss.per.response[j], M, penalty, n): maxreldiff=%.3g\n",
            maxreldiff(fit_m$gcv.per.response, gcv_per_resp_pred)))

map_m <- dirs_to_bx_map(fit_m)
sel_m <- fit_m$selected.terms
configs <- list(list(k = 1, cols = bx_cols_for(map_m, intercept_dirs_row(fit_m))),
                list(k = length(sel_m), cols = bx_cols_for(map_m, sel_m)))
cat("rss.per.subset[k] vs sum over responses of per-response RSS (via lm.fit):\n")
subset_multiresp_ok <- logical(0)
subset_gcv_multiresp_ok <- logical(0)
for (cfg in configs) {
  bxk <- fit_m$bx[, cfg$cols, drop = FALSE]
  rss_j <- vapply(seq_len(ncol(y3)), function(j) sum(lm.fit(bxk, y3[, j])$residuals^2), numeric(1))
  rss_sum <- sum(rss_j)
  actual <- fit_m$rss.per.subset[cfg$k]
  rd <- abs(actual - rss_sum) / abs(rss_sum)
  cat(sprintf("  k=%-3d actual=%.10g  sum_j(RSS_j)=%.10g  (rd=%.3g)\n", cfg$k, actual, rss_sum, rd))
  subset_multiresp_ok <- c(subset_multiresp_ok, rd < 1e-8)

  gcv_j <- vapply(seq_len(ncol(y3)), function(j) safe_get_gcv(rss_j[j], cfg$k, fit_m$penalty, n), numeric(1))
  gcv_sum_k <- sum(gcv_j)
  actual_gcv <- fit_m$gcv.per.subset[cfg$k]
  rd_gcv <- abs(actual_gcv - gcv_sum_k) / abs(gcv_sum_k)
  cat(sprintf("  k=%-3d actual gcv.per.subset=%.10g  sum_j(get.gcv(RSS_j,k,pen,n))=%.10g  (rd=%.3g)\n",
              cfg$k, actual_gcv, gcv_sum_k, rd_gcv))
  subset_gcv_multiresp_ok <- c(subset_gcv_multiresp_ok, rd_gcv < 1e-8)
}

tss_j <- apply(y3, 2, function(yy) sum((yy - mean(yy))^2))
has_rsq_per_resp <- "rsq.per.response" %in% names(fit_m)
has_grsq_per_resp <- "grsq.per.response" %in% names(fit_m)
cat(sprintf("\nfit_m has rsq.per.response: %s ; has grsq.per.response: %s\n", has_rsq_per_resp, has_grsq_per_resp))
hyp1_rsq <- 1 - sum(fit_m$rss.per.response) / sum(tss_j)
hyp2_rsq <- if (has_rsq_per_resp) mean(fit_m$rsq.per.response) else NA
cat(sprintf("rsq: actual=%.10g  hyp(1-sum(rss)/sum(tss))=%.10g  hyp(mean rsq.per.response)=%.10g\n",
            fit_m$rsq, hyp1_rsq, hyp2_rsq))
rsq_hyp1_ok <- abs(fit_m$rsq - hyp1_rsq) < 1e-8
rsq_hyp2_ok <- !is.na(hyp2_rsq) && abs(fit_m$rsq - hyp2_rsq) < 1e-8

gcv_null_j <- vapply(seq_len(ncol(y3)), function(j) safe_get_gcv(tss_j[j], 1, fit_m$penalty, n), numeric(1))
hyp1_grsq <- 1 - sum(fit_m$gcv.per.response) / sum(gcv_null_j)
hyp2_grsq <- if (has_grsq_per_resp) mean(fit_m$grsq.per.response) else NA
cat(sprintf("grsq: actual=%.10g  hyp(1-sum(gcv)/sum(gcv.null))=%.10g  hyp(mean grsq.per.response)=%.10g\n",
            fit_m$grsq, hyp1_grsq, hyp2_grsq))
grsq_hyp1_ok <- abs(fit_m$grsq - hyp1_grsq) < 1e-8
grsq_hyp2_ok <- !is.na(hyp2_grsq) && abs(fit_m$grsq - hyp2_grsq) < 1e-8

fit_1col <- earth(x, matrix(y, ncol = 1), degree = 1)
onecol_same <- isTRUE(all.equal(fit_1col$rss, fit_a$rss)) &&
  isTRUE(all.equal(fit_1col$gcv, fit_a$gcv)) &&
  isTRUE(all.equal(fit_1col$rsq, fit_a$rsq)) &&
  identical(fit_1col$dirs, fit_a$dirs) &&
  identical(fit_1col$selected.terms, fit_a$selected.terms)
cat(sprintf("\n1-column matrix y gives identical fit to vector y: %s\n", onecol_same))

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb02.1 %s gcv.per.subset matches get.gcv(rss.per.subset[k],k,penalty,n) for fit(a)\n", ifelse(gcv_ok["a"], "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.2 %s gcv.per.subset matches get.gcv(rss.per.subset[k],k,penalty,n) for fit(b)\n", ifelse(gcv_ok["b"], "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.3 %s gcv.per.subset matches get.gcv(rss.per.subset[k],k,penalty,n) for fit(c)\n", ifelse(gcv_ok["c"], "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.4 %s gcv.per.subset matches get.gcv(rss.per.subset[k],k,penalty,n) for fit(d)\n", ifelse(gcv_ok["d"], "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.5 %s rsq == 1-rss/tss for all of fits a-d\n", ifelse(all(rsq_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.6 %s grsq == 1-gcv/gcv.null for all of fits a-d\n", ifelse(all(grsq_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.7 %s HYPOTHESIS a continuous predictor that is exactly linear in the DGP gets dirs code 2 under Auto.linpreds\n", ifelse(c0_has_2, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.8 %s a two-level (0/1) predictor gets dirs code 2 under Auto.linpreds=TRUE\n", ifelse(c_has_2, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.9 %s HYPOTHESIS fit(d) has an odd non-intercept term count\n", ifelse(d_nrow_minus1_odd, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.10 %s fit(d): (M-1) < 2*distinct_knots, proving at least one single-hinge step\n", ifelse(d_fewer_terms_than_2x_knots, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.11 %s HYPOTHESIS get.gcv's n argument is sum(w) across weights={all2,int,int10}\n", ifelse(all(n_sumw_ok == "TRUE"), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.12 %s get.gcv's n argument is nrow(x), unaffected by weights\n", ifelse(all(n_nrowx_ok == "TRUE"), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.13 %s HYPOTHESIS rss.per.subset is the un-normalized weighted RSS sum(w*(y-yhat)^2), for ALL of {all2,int,int10}\n", ifelse(all(rss_raw_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.14 %s rss.per.subset is the un-normalized weighted RSS when the weights are not all equal (int, int10)\n", ifelse(all(rss_raw_ok[c("int", "int10")]), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.15 %s a CONSTANT weight vector (all2) gives a fit identical in every respect to no weights at all\n", ifelse(const_w_is_unweighted, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.16 %s HYPOTHESIS rsq's tss is the un-normalized weighted sum(w*(y-weighted.mean(y,w))^2), for ALL of {all2,int,int10}\n", ifelse(all(tss_raw_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.17 %s rsq's tss is the un-normalized weighted tss when the weights are not all equal (int, int10)\n", ifelse(all(tss_raw_ok[c("int", "int10")]), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.18 %s HYPOTHESIS scaling all (non-constant) weights by 10 leaves dirs/cuts/selected.terms unchanged\n", ifelse(struct_same, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.19 %s scaling all (non-constant) weights by 10 leaves rsq,grsq unchanged (ratio-invariant, <1e-8 abs diff)\n",
            ifelse(rsq_diff < 1e-8 && grsq_diff < 1e-8, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.20 %s scaling all (non-constant) weights by 10 scales raw rss,gcv by 10x (NOT invariant; ratio in [9.99,10.01])\n",
            ifelse(abs(rss_ratio - 10) < 0.01 && abs(gcv_ratio - 10) < 0.01, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.21 %s gcv.per.subset == rss.per.subset/n for penalty=-1 (maxreldiff=%.3g)\n", ifelse(rd_pm1 < 1e-10, "TRUE", "FALSE"), rd_pm1))
cat(sprintf("CHECK bb02.22 %s gcv.per.subset matches C=nterms rule for penalty=0 (maxreldiff=%.3g)\n", ifelse(rd_p0 < 1e-10, "TRUE", "FALSE"), rd_p0))
cat(sprintf("CHECK bb02.23 %s rss == sum(rss.per.response) for the 3-response fit\n", ifelse(rss_sum_ok, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.24 %s gcv == sum(gcv.per.response) AND gcv.per.response[j] == get.gcv(rss.per.response[j],M,penalty,n)\n", ifelse(gcv_sum_ok && gcv_per_resp_ok, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.25 %s rss.per.subset[k] == sum_j(per-response RSS of that subset) at k=1,k=selected\n", ifelse(all(subset_multiresp_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.26 %s gcv.per.subset[k] == sum_j(get.gcv(rss_jk,k,penalty,n)) at k=1,k=selected\n", ifelse(all(subset_gcv_multiresp_ok), "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.27 %s multi-response rsq == 1-sum(rss.per.response)/sum(tss.per.response)\n", ifelse(rsq_hyp1_ok, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.28 %s multi-response grsq == 1-sum(gcv.per.response)/sum(gcv.null.per.response)\n", ifelse(grsq_hyp1_ok, "TRUE", "FALSE")))
cat(sprintf("CHECK bb02.29 %s a 1-column matrix y gives an identical fit to the equivalent vector y\n", ifelse(onecol_same, "TRUE", "FALSE")))
