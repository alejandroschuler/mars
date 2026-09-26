# bb09_coef.R
# Question 1: are earth's final coefficients the least-squares coefficients of y
# on the selected basis columns bx (lm.fit, or lm.wfit with weights), on the
# original y scale, also with several responses? Are fitted.values bx %*% coef?
# Is bx the basis evaluated from dirs and cuts (products of pmax(x - cut, 0) for
# code 1, pmax(cut - x, 0) for code -1, and x for code 2)?
# Question 2: what is lm.fit's rule for a rank-deficient least-squares solve?
# At which ratio ||part of a column orthogonal to the earlier columns|| / ||column||
# does lm.fit drop the column, compared with its default tol = 1e-7? Which of
# two equal or nearly equal columns gets NA, and what are the other coefficients?
suppressMessages(library(earth))
# On failure print only the condition message (no call, no traceback).
earth_bb <- function(...) tryCatch(earth(...), error = function(e) stop(paste("ERROR:", conditionMessage(e)), call. = FALSE))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

checks <- character(0)
CHECK <- function(id, ok, desc) {
  checks <<- c(checks, sprintf("CHECK %s %s %s", id, if (isTRUE(ok)) "TRUE" else "FALSE", desc))
}
nrel <- function(a, b) sqrt(sum((a - b)^2)) / sqrt(sum(b^2))  # normwise relative difference

# ---------------------------------------------------------------- part 1
set.seed(1); n <- 200
x <- matrix(runif(n * 5), n, 5); colnames(x) <- paste0("x", 1:5)
y <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5] + rnorm(n)
w_int <- rep(1:3, length.out = n)
set.seed(2); w_unif <- runif(n, 0.5, 2)
Y3 <- cbind(y1 = y, y2 = 2 * y + x[, 1], y3 = sin(3 * x[, 2]))

eval_basis <- function(fit, x) {
  terms <- fit$selected.terms
  B <- sapply(terms, function(t) {
    col <- rep(1, nrow(x))
    for (p in seq_len(ncol(x))) {
      d <- fit$dirs[t, p]; ct <- fit$cuts[t, p]
      if (d == 1) col <- col * pmax(x[, p] - ct, 0)
      else if (d == -1) col <- col * pmax(ct - x[, p], 0)
      else if (d == 2) col <- col * x[, p]
      else if (d != 0) stop("unexpected dirs code")
    }
    col
  })
  matrix(B, nrow(x))
}

cases <- list(
  list(lab = "deg1 unweighted", deg = 1, Y = y, w = NULL),
  list(lab = "deg1 w=rep(1:3)", deg = 1, Y = y, w = w_int),
  list(lab = "deg1 w=runif", deg = 1, Y = y, w = w_unif),
  list(lab = "deg2 unweighted", deg = 2, Y = y, w = NULL),
  list(lab = "deg2 w=rep(1:3)", deg = 2, Y = y, w = w_int),
  list(lab = "deg2 w=runif", deg = 2, Y = y, w = w_unif),
  list(lab = "deg1 3 responses", deg = 1, Y = Y3, w = NULL),
  list(lab = "deg2 3 responses", deg = 2, Y = Y3, w = NULL),
  list(lab = "deg2 3 resp w=runif", deg = 2, Y = Y3, w = w_unif),
  list(lab = "deg1 linpreds=4:5", deg = 1, Y = y, w = NULL, lp = 4:5),
  list(lab = "deg2 linpreds=4:5 w=runif", deg = 2, Y = y, w = w_unif, lp = 4:5))
cat("case | nterms | code counts (1,-1,2) in selected dirs | coef vs LS | fitted vs bx%*%coef | fitted vs LS | bx vs dirs/cuts (max abs)\n")
ok_coef <- ok_fit <- ok_bx <- ok_names <- ok_bitwise <- TRUE; n_code2 <- 0; cut2 <- numeric(0); cut2_is_min <- TRUE
for (cs in cases) {
  lp <- if (is.null(cs$lp)) FALSE else cs$lp
  fit <- if (is.null(cs$w)) earth_bb(x = x, y = cs$Y, degree = cs$deg, linpreds = lp) else
    earth_bb(x = x, y = cs$Y, degree = cs$deg, weights = cs$w, linpreds = lp)
  bx <- fit$bx; Ym <- as.matrix(cs$Y)
  ls <- if (is.null(cs$w)) lm.fit(bx, Ym) else lm.wfit(bx, Ym, cs$w)
  cf <- as.matrix(fit$coefficients); lcf <- as.matrix(ls$coefficients)
  d_coef <- max(sapply(seq_len(ncol(Ym)), function(j) nrel(cf[, j], lcf[, j])))
  fv <- as.matrix(fit$fitted.values)
  d_fit <- max(sapply(seq_len(ncol(Ym)), function(j) nrel(fv[, j], drop(bx %*% cf[, j]))))
  d_fitls <- max(sapply(seq_len(ncol(Ym)), function(j) nrel(fv[, j], as.matrix(ls$fitted.values)[, j])))
  mine <- eval_basis(fit, x); d_bx <- max(abs(mine - bx)); r_bx <- nrel(mine, bx)
  nm_ok <- identical(colnames(bx), rownames(fit$dirs)[fit$selected.terms]) && identical(rownames(cf), colnames(bx))
  sd <- fit$dirs[fit$selected.terms, , drop = FALSE]
  cc <- c(sum(sd == 1), sum(sd == -1), sum(sd == 2)); n_code2 <- n_code2 + cc[3]
  if (cc[3] > 0) cut2 <- c(cut2, fit$cuts[fit$selected.terms, , drop = FALSE][sd == 2])
  for (k in which(apply(sd == 2, 1, any))) for (jj in which(sd[k, ] == 2))
    cut2_is_min <- cut2_is_min && fit$cuts[fit$selected.terms[k], jj] == min(x[, jj])
  cat(sprintf("%-20s | %2d | %2d %2d %2d | %.1e | %.1e | %.1e | %.1e (rel %.1e)\n", cs$lab, ncol(bx), cc[1], cc[2], cc[3],
              d_coef, d_fit, d_fitls, d_bx, r_bx))
  ok_coef <- ok_coef && d_coef < 1e-10; ok_fit <- ok_fit && d_fit < 1e-10 && d_fitls < 1e-10
  ok_bitwise <- ok_bitwise && identical(unname(cf), unname(lcf))
  ok_bx <- ok_bx && r_bx < 1e-10; ok_names <- ok_names && nm_ok
}
cat(sprintf("cuts stored at dirs code 2 entries: %s (the basis uses x, not x - cut)\n", paste(sprintf("%.6g", sort(unique(cut2))), collapse = " ")))
# Is the fit weighted as lm.wfit, not as lm.fit? (guards against a weighted check that passes by accident)
fw <- earth_bb(x = x, y = y, degree = 1, weights = w_unif)
d_unw <- nrel(fw$coefficients[, 1], lm.fit(fw$bx, y)$coefficients)
cat(sprintf("weighted fit (deg1 w=runif): coef vs unweighted lm.fit on the same bx differs by %.1e\n", d_unw))

# ---------------------------------------------------------------- part 2
cat("lm.fit rank rule: X = [1, c2, c3], c3 = c2 + eps*z, z orthogonal to 1 and c2, ||z|| = ||c2||\n")
set.seed(3); m <- 100
c1 <- rep(1, m); yv <- rnorm(m)
make_c3 <- function(c2, eps, z0) {
  z <- qr.resid(qr(cbind(c1, c2)), z0); z <- z * sqrt(sum(c2^2)) / sqrt(sum(z^2))
  c2 + eps * z
}
ratio <- function(c2, c3) {  # computed ||part of c3 orthogonal to (1, c2)|| / ||c3||, and the same over the centred norm
  rp <- sqrt(sum(qr.resid(qr(cbind(c1, c2)), c3)^2))
  c(full = rp / sqrt(sum(c3^2)), centred = rp / sqrt(sum((c3 - mean(c3))^2)))
}
scan <- function(c2, z0, epss, label) {
  cat(sprintf("  %s: ||c2 - mean|| / ||c2|| = %.3f\n", label, sqrt(sum((c2 - mean(c2))^2)) / sqrt(sum(c2^2))))
  cat("    eps  r(eps)=eps/sqrt(1+eps^2)  r computed  r over centred norm  rank  coef3_NA\n")
  out <- NULL
  for (eps in epss) {
    c3 <- make_c3(c2, eps, z0); f <- lm.fit(cbind(c1, c2, c3), yv); rr <- ratio(c2, c3)
    cat(sprintf("    %.8e  %.10e  %.10e  %.6e  %d  %s\n", eps, eps / sqrt(1 + eps^2), rr[1], rr[2], f$rank, is.na(f$coefficients[3])))
    out <- rbind(out, c(eps = eps, r = unname(rr[1]), rc = unname(rr[2]), rank = f$rank, na = unname(is.na(f$coefficients[3]))))
  }
  out
}
z0 <- rnorm(m)
grid <- 10^seq(-3, -11, by = -0.5)
fine <- 1e-7 * c(2, 1.01, 1.001, 1 + 1e-6, 1, 1 - 1e-6, 1 - 1e-4, 0.999, 0.99, 0.5)
cA <- rnorm(m)                 # nearly centred column
cB <- 1 + runif(m)             # column with a large mean: full and centred norms differ by a factor of about 5
sA <- scan(cA, z0, c(grid, fine), "c2 = rnorm")
sB <- scan(cB, z0, c(10^seq(-6, -9, by = -0.5), fine, 1e-7 * c(0.3, 0.2, 0.19, 0.18)), "c2 = 1 + runif")
thr <- function(s, col) { kept <- s[s[, "rank"] == 3, col]; drop <- s[s[, "rank"] == 2, col]; c(min_kept = min(kept), max_dropped = max(drop)) }
tA <- thr(sA, "r"); tB <- thr(sB, "r"); tBc <- thr(sB, "rc")
cat(sprintf("  threshold (full norm): rnorm c2: smallest kept r %.17g, largest dropped r %.17g; 1+runif c2: %.17g, %.17g\n", tA[1], tA[2], tB[1], tB[2]))
cat(sprintf("  threshold (centred norm) for 1+runif c2: smallest kept %.6e, largest dropped %.6e\n", tBc[1], tBc[2]))
rank_rule_full <- tA[2] < 1e-7 && tA[1] >= 1e-7 * (1 - 1e-6) && tB[2] < 1e-7 && tB[1] >= 1e-7 * (1 - 1e-6)
rank_rule_centred <- tBc[2] < 1e-7 && tBc[1] >= 1e-7 * (1 - 1e-6)

# identical columns and dependent columns in other positions
c2 <- cB; c4 <- runif(m)
f_dup <- lm.fit(cbind(c1, c2, c2), yv); f_ref <- lm.fit(cbind(c1, c2), yv)
cat(sprintf("  [1, c2, c2]: rank %d, coef NA = %s, pivot = %s; coef[1:2] - lm.fit([1, c2]) max abs %.1e, identical %s\n",
            f_dup$rank, paste(is.na(f_dup$coefficients), collapse = " "), paste(f_dup$qr$pivot, collapse = " "),
            max(abs(f_dup$coefficients[1:2] - f_ref$coefficients)), identical(unname(f_dup$coefficients[1:2]), unname(f_ref$coefficients))))
f_mid <- lm.fit(cbind(c1, c2, c2, c4), yv); f_ref2 <- lm.fit(cbind(c1, c2, c4), yv)
cat(sprintf("  [1, c2, c2, c4]: rank %d, coef NA = %s, pivot = %s; non-NA coef - lm.fit([1, c2, c4]) max abs %.1e, identical %s\n",
            f_mid$rank, paste(is.na(f_mid$coefficients), collapse = " "), paste(f_mid$qr$pivot, collapse = " "),
            max(abs(f_mid$coefficients[c(1, 2, 4)] - f_ref2$coefficients)), identical(unname(f_mid$coefficients[c(1, 2, 4)]), unname(f_ref2$coefficients))))
fit_same <- max(abs(f_mid$fitted.values - f_ref2$fitted.values))
c3 <- make_c3(c2, 1e-9, z0)
f_32 <- lm.fit(cbind(c1, c3, c2), yv); f_23 <- lm.fit(cbind(c1, c2, c3), yv)
cat(sprintf("  near-dependent pair (eps = 1e-9): [1, c3, c2] coef NA = %s; [1, c2, c3] coef NA = %s\n",
            paste(is.na(f_32$coefficients), collapse = " "), paste(is.na(f_23$coefficients), collapse = " ")))
f_lead <- lm.fit(cbind(c2, c2, c1), yv)
cat(sprintf("  [c2, c2, 1] (intercept last): coef NA = %s\n", paste(is.na(f_lead$coefficients), collapse = " ")))
# lm.wfit: the same rule on the sqrt(w)-scaled columns
wv <- runif(m, 0.5, 2); sw <- sqrt(wv)
wr <- function(eps) {
  c3 <- make_c3(c2, eps, z0); f <- lm.wfit(cbind(c1, c2, c3), yv, wv)
  rp <- sqrt(sum(qr.resid(qr(cbind(c1, c2) * sw), c3 * sw)^2)) / sqrt(sum((c3 * sw)^2))
  c(r_w = rp, rank = f$rank)
}
wa <- wr(1.2e-7); wb <- wr(0.8e-7)
cat(sprintf("  lm.wfit: weighted ratio %.4e -> rank %d; weighted ratio %.4e -> rank %d\n", wa[1], wa[2], wb[1], wb[2]))
wrule <- (wa[1] >= 1e-7) == (wa[2] == 3) && (wb[1] >= 1e-7) == (wb[2] == 3)

CHECK("bb09.1", ok_coef, "coefficients equal lm.fit / lm.wfit coefficients of y on bx (normwise rel 1e-10), all 11 cases")
CHECK("bb09.1b", ok_bitwise, "the coefficients are bitwise identical to the lm.fit / lm.wfit coefficients, all 11 cases")
CHECK("bb09.2", d_unw > 1e-6, "the weighted coefficients differ from unweighted lm.fit on the same bx (so bb09.1 tests the weights)")
CHECK("bb09.3", ok_fit, "fitted.values equal bx %*% coefficients and the lm.fit / lm.wfit fitted values (normwise rel 1e-10), all cases")
CHECK("bb09.4", ok_bx, "bx equals the basis evaluated from dirs and cuts (1: pmax(x-cut,0), -1: pmax(cut-x,0), 2: x), all cases")
CHECK("bb09.5", n_code2 > 0 && any(cut2 != 0), "the cases include dirs code 2 with nonzero stored cuts (so bb09.4 tests 'x, not x - cut' for code 2)")
CHECK("bb09.15", n_code2 > 0 && cut2_is_min, "the stored cut of every dirs code 2 entry equals the smallest value of that column of x")
CHECK("bb09.6", ok_names, "colnames(bx) are the dirs rownames of selected.terms, in that order, and match rownames(coefficients)")
CHECK("bb09.7", sum(sA[, "na"]) > 0 && all((sA[, "rank"] == 2) == sA[, "na"]), "lm.fit drops the near-dependent column (rank 2, coef[3] NA) for small eps")
CHECK("bb09.8", rank_rule_full, "lm.fit keeps c3 iff ||part orthogonal to earlier columns|| / ||c3|| >= tol = 1e-7 (full, uncentred norm)")
CHECK("bb09.9", rank_rule_centred, "ALTERNATIVE: the threshold 1e-7 applies to the ratio over the centred norm of c3")
CHECK("bb09.10", is.na(f_dup$coefficients[3]) && !any(is.na(f_dup$coefficients[1:2])), "with two identical columns the later one gets NA")
CHECK("bb09.11", max(abs(f_dup$coefficients[1:2] - f_ref$coefficients)) < 1e-12 * max(abs(f_ref$coefficients)) &&
        max(abs(f_mid$coefficients[c(1, 2, 4)] - f_ref2$coefficients)) < 1e-12 * max(abs(f_ref2$coefficients)) && fit_same < 1e-12,
      "the other coefficients (and fitted values) equal the fit without the dropped column (rel 1e-12)")
CHECK("bb09.12", is.na(f_mid$coefficients[3]) && identical(as.integer(f_mid$qr$pivot), c(1L, 2L, 4L, 3L)),
      "a dropped column in the middle is pivoted to the end; coefficients stay in the original column order with NA there")
CHECK("bb09.13", is.na(f_32$coefficients[3]) && is.na(f_23$coefficients[3]) && !any(is.na(f_32$coefficients[1:2])),
      "of two nearly equal columns the later in column order gets NA, whichever of c2, c3 comes first")
CHECK("bb09.14", wrule, "lm.wfit applies the same 1e-7 rule to the sqrt(w)-scaled columns")
cat(checks, sep = "\n")
