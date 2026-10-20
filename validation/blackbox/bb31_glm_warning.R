# bb31: when the binomial refit warns that fitted probabilities are numerically
# 0 or 1 (GLM-4, issue #44, t14-glm on PR #78). earth's GLM coefficients are
# those of R's glm on the selected columns (bb21.1a), and earth passes glm's
# warnings on (bb21.3), so this experiment calls glm as a black box.
# Four cases with offsets c, -c, 0, 0 and y = 1, 0, 1, 0: by symmetry the
# fitted intercept is 0, so the fitted linear predictors are +c and -c. A scan
# of c from 29 to 35 moves the largest |eta| across 30 and across 33.7.
# Hypothesis (pymars v1): a warning when a fitted probability is within 10*eps
# of 0 or 1, which needs |eta| > -log(10*eps) = 33.7. Alternative: from |eta| > 30.
cat(R.version.string, "\n")
eps <- .Machine$double.eps
run <- function(cc) {
  msg <- character(0)
  d <- data.frame(y = c(1, 0, 1, 0), o = c(cc, -cc, 0, 0))
  f <- withCallingHandlers(glm(y ~ 1 + offset(o), family = binomial, data = d),
    warning = function(w) { msg <<- c(msg, conditionMessage(w)); invokeRestart("muffleWarning") })
  warned <- any(grepl("numerically 0 or 1", msg))
  eta <- max(abs(predict(f, type = "link")))
  cat(sprintf("  c %-6g: intercept %+.2e, largest |eta| %.4f, smallest fitted p %.4g, warning %s%s\n",
    cc, coef(f)[1], eta, min(fitted(f)), warned, if (length(msg)) paste0(" (", paste(unique(msg), collapse = "; "), ")") else ""))
  c(eta = eta, warned = warned)
}
grid <- c(29, 29.9, 29.99, 30.01, 30.1, 31, 33, 33.6, 33.8, 35)
r <- t(sapply(grid, run))
cat(sprintf("CHECK bb31.1 %s glm warns exactly when the largest |eta| is above 30 (%d fits)\n",
  all(r[, "warned"] == (r[, "eta"] > 30)), nrow(r)))
cat(sprintf("CHECK bb31.2 %s HYPOTHESIS glm warns exactly when a fitted probability is within 10*eps of 0 or 1 (|eta| > %.2f)\n",
  all(r[, "warned"] == (r[, "eta"] > -log(10 * eps))), -log(10 * eps)))
