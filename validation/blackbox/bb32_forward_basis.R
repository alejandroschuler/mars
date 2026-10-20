# bb32: the basis of earth's forward pass after a step whose new term the
# dependence fix at the end of the pass removes (FWD-11, LA-1, issue #127).
# Question: after such a step, do the later steps and the forward RSq use a
# basis that holds the removed term's direction, or is the term dropped at once?
# Data: x1 = U(0, 1), x2 = M + z with z = U(0, 1), and
# y = 2 h(0.5 - x1) + 4 z h(0.5 - x1) + noise, degree 2, nk 11, thresh 0,
# pmethod "none" (so the returned model holds every term that the fix keeps).
# Step 2 adds the linear term h(c - x1) * x2 = M h(c - x1) + z h(c - x1). Its
# part orthogonal to the other columns is z h(c - x1)'s, about 1/M of its norm,
# while it carries most of the signal. M = 3e6: no fix. M = 1e8: the fix
# removes one term. M = 1e7: between the two.
# The script computes RSS values with the offset taken out exactly
# (x2 - M is exact, Sterbenz), so its own least-squares fits are well posed.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
set.seed(1)
n <- 60
x1 <- runif(n); z <- runif(n)
rsq_of <- function(B, y) 1 - sum(lm.fit(B, y)$residuals^2) / sum((y - mean(y))^2)
steps_rsq <- function(out) {
  # RSq column of the step table: lines "<terms> <GRSq> <RSq> <DeltaRSq> ..."
  rows <- grep("^[0-9]+ +-?[0-9.]+ +[0-9.]+ +[0-9.e+-]+ +[0-9]+ ", out, value = TRUE)
  as.numeric(sapply(strsplit(trimws(rows), " +"), `[`, 3))
}
res <- list()
for (M in c(3e6, 1e7, 1e8)) {
  x2 <- M + z
  zz <- x2 - M  # exact
  y <- 2 * pmax(0.5 - x1, 0) + 4 * z * pmax(0.5 - x1, 0) + 0.01 * rnorm(n)
  out <- capture.output(f <- tryCatch(
    earth(cbind(x1, x2), y, degree = 2, trace = 2, pmethod = "none", thresh = 0, nk = 11),
    error = function(e) conditionMessage(e)))
  fixmsg <- grep("Fixed rank deficient", out, value = TRUE)
  fwd <- steps_rsq(out)
  cat(sprintf("\nM = %g\n", M))
  cat(grep("^ +GRSq|^[0-9]+ ", out, value = TRUE), sep = "\n")
  cat(sprintf("  forward RSq after steps 1 to %d: %s\n", length(fwd), paste(fwd, collapse = " ")))
  cat(sprintf("  fix message: %s\n", if (length(fixmsg)) fixmsg else "(none)"))
  r <- list(M = M, fwd = fwd, fix = length(fixmsg) > 0, error = is.character(f))
  if (is.character(f)) {
    cat(sprintf("  earth stopped with the error: %s\n", f))
  } else {
    terms <- rownames(f$dirs)
    lin <- grep("^h\\(0.497699-x1\\)\\*x2$", terms, value = TRUE)
    H <- f$bx[, "h(0.497699-x1)"]
    keep <- setdiff(colnames(f$bx), "h(0.497699-x1)*x2")
    # every returned column, with h(c - x1) * x2 written as M H + zz H, so the
    # direction it adds is zz H
    B_kept <- f$bx[, keep, drop = FALSE]
    rsq_kept <- rsq_of(B_kept, y)
    rsq_with <- rsq_of(cbind(B_kept, zz * H), y)
    # the relative orthogonal part of h(c - x1) * x2 against the other columns
    qrk <- qr(B_kept)
    orth <- sqrt(sum(qr.resid(qrk, zz * H)^2)) / sqrt(sum((x2 * H)^2))
    cat(sprintf("  returned terms: %s\n", paste(terms, collapse = ", ")))
    cat(sprintf("  returned RSq %.6f; RSq of the returned columns without h(c-x1)*x2 %.6f, with its direction zz*H %.6f\n",
      f$rsq, rsq_kept, rsq_with))
    cat(sprintf("  relative orthogonal part of h(c-x1)*x2 against the other columns: %.3g\n", orth))
    r <- c(r, list(has_lin = length(lin) == 1, rsq = f$rsq, rsq_kept = rsq_kept,
      rsq_with = rsq_with, orth = orth))
  }
  res[[as.character(M)]] <- r
}
a <- res[["3e+06"]]; b <- res[["1e+07"]]; c8 <- res[["1e+08"]]
cat("\n")
cat(sprintf("CHECK bb32.1 %s at M = 1e8 the fix removes one term after the forward pass, and that term is h(c-x1)*x2, which step 2 added; at M = 3e6 there is no fix and the term is returned\n",
  c8$fix && !c8$has_lin && !a$fix && a$has_lin))
cat(sprintf("CHECK bb32.2 %s at M = 1e8 the forward RSq after step 2 and after every later step is at least 0.999, while the returned model, every term that the fix kept, has RSq %.4f < 0.95: the forward pass kept the removed term's direction in its basis for all later steps, and the fix changed no forward value\n",
  all(c8$fwd[-1] >= 0.999) && c8$rsq < 0.95, c8$rsq))
cat(sprintf("CHECK bb32.3 %s at M = 1e8 the RSq of the returned columns plus the removed term's direction zz*H, %.4f, equals the last forward RSq %.4f to the 4 digits of the trace, and without it equals the returned RSq: the forward RSS is that of the span of all forward terms\n",
  round(c8$rsq_with, 4) == tail(c8$fwd, 1) && abs(c8$rsq_kept - c8$rsq) < 1e-6, c8$rsq_with, tail(c8$fwd, 1)))
cat(sprintf("CHECK bb32.4 %s the removed term's relative orthogonal part against the other returned columns is %.2g at M = 1e8 and %.2g at M = 3e6, both below LA-4's 1e-7, yet only the first is removed; at M = 1e7 the fix removes nothing and earth stops with an error from its final least-squares fit: earth's fix has a tolerance of its own, between the two\n",
  c8$orth < 1e-7 && a$orth < 1e-7 && c8$fix && !a$fix && !b$fix && b$error, c8$orth, a$orth))
