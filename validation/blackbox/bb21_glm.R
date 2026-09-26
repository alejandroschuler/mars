# bb21: earth's GLM refit for a classifier (glm = list(family = binomial)).
#
# Data: n = 500, 5 uniform covariates; a binary y from a logistic truth with
# hinge effects, and (from the same x) a 3-class factor response.
#
# 1. Binary: is glm.coefficients equal (relative 1e-8) to glm(y ~ bx - 1,
#    family = binomial) on the fit's own bx? Are dirs/cuts/selected.terms the
#    same as earth(x, y, degree = 2) without glm (so the GLM does not change
#    the passes)? Does predict(fit, type = "response") equal
#    plogis(bx %*% glm.coefficients)?
# 2. Binary with weights (integers 1:3, and non-integer): does the glm match
#    glm(y ~ bx - 1, family = binomial, weights = w) exactly?
# 3. Separation: a y a hinge separates perfectly. What warnings does earth
#    print (condition messages only)? How close to 0/1 are the fitted
#    probabilities?
# 4. Three classes: a factor y with glm = list(family = binomial). How many
#    response/glm columns, is dirs shared (one pass), and is each column's
#    glm a separate binomial glm of that class indicator on bx (compare with
#    glm() fit alone)? Do the three probabilities of a row sum to 1? Compare
#    with nnet::multinom(yfactor ~ bx - 1) on the same bx.
# 5. A binary y given as a factor, as logical, and as 0/1: the same fit?
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
warncatch <- function(expr) {
  warns <- character(0)
  val <- withCallingHandlers(expr, warning = function(w) {
    warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning")
  })
  list(value = val, warnings = warns)
}

set.seed(20260925)
n <- 500; p <- 5
x <- matrix(runif(n * p), n, p); colnames(x) <- paste0("x", 1:p)
lp <- 2 * pmax(x[, 1] - 0.4, 0) - 3 * pmax(0.6 - x[, 2], 0) + x[, 3]
y01 <- rbinom(n, 1, plogis(lp))
lpA <- 2 * pmax(x[, 1] - 0.4, 0) - x[, 4]
lpB <- -1.5 * pmax(0.6 - x[, 3], 0) + x[, 5]
probs3 <- cbind(exp(lpA), exp(lpB), 1); probs3 <- probs3 / rowSums(probs3)
set.seed(20260926)
yfac <- factor(apply(probs3, 1, function(pr) sample(c("a", "b", "c"), 1, prob = pr)), levels = c("a", "b", "c"))
cat("n =", n, "| table(y01):", paste(names(table(y01)), table(y01), sep = "=", collapse = " "),
  "| table(yfac):", paste(names(table(yfac)), table(yfac), sep = "=", collapse = " "), "\n")

## ================= Part 1: binary GLM matches glm() on bx; passes unchanged =================
f1 <- quiet(earth(x, y01, degree = 2, glm = list(family = binomial)))
f1_nog <- quiet(earth(x, y01, degree = 2))
bx1 <- f1$bx
g1 <- glm(y01 ~ bx1 - 1, family = binomial)
coef_reldiff <- max(abs(coef(g1) - f1$glm.coefficients[, 1]) / pmax(abs(f1$glm.coefficients[, 1]), 1e-8))
same_passes <- identical(f1$dirs, f1_nog$dirs) && isTRUE(all.equal(f1$cuts, f1_nog$cuts)) &&
  identical(f1$selected.terms, f1_nog$selected.terms)
pr_resp <- predict(f1, type = "response")
manual_resp <- plogis(bx1 %*% f1$glm.coefficients)
resp_diff <- max(abs(pr_resp - manual_resp))
cat(sprintf("\npart 1: %d bx columns; max relative coef diff vs glm(y~bx-1,family=binomial): %.3g\n", ncol(bx1), coef_reldiff))
cat(sprintf("        same dirs/cuts/selected.terms as earth(x,y,degree=2) without glm: %s\n", same_passes))
cat(sprintf("        predict(fit,type=\"response\") vs plogis(bx %%*%% glm.coefficients): max abs diff %.3g\n", resp_diff))
ok1_coef <- coef_reldiff < 1e-8
ok1_passes <- same_passes
ok1_pred <- resp_diff < 1e-8

## ================= Part 2: weights =================
set.seed(101); wint <- sample(1:3, n, replace = TRUE)
set.seed(102); wnon <- runif(n, 0.5, 3)
f2i <- quiet(earth(x, y01, degree = 2, glm = list(family = binomial), weights = wint))
f2n <- suppressWarnings(quiet(earth(x, y01, degree = 2, glm = list(family = binomial), weights = wnon)))
g2i <- glm(y01 ~ f2i$bx - 1, family = binomial, weights = wint)
g2n <- suppressWarnings(glm(y01 ~ f2n$bx - 1, family = binomial, weights = wnon))
wi_diff <- max(abs(coef(g2i) - f2i$glm.coefficients[, 1]))
wn_diff <- max(abs(coef(g2n) - f2n$glm.coefficients[, 1]))
cat(sprintf("\npart 2: integer weights (1:3) max abs coef diff vs glm(weights=w): %.3g\n", wi_diff))
cat(sprintf("        non-integer weights max abs coef diff vs glm(weights=w): %.3g (both emit R's usual \"non-integer #successes\" warning, suppressed here)\n", wn_diff))
ok2 <- wi_diff < 1e-8 && wn_diff < 1e-8

## ================= Part 3: perfect separation =================
set.seed(42); n3 <- 200; x3 <- matrix(runif(n3 * 3), n3, 3); colnames(x3) <- paste0("x", 1:3)
ysep <- as.numeric(x3[, 1] > 0.5)  # a single hinge at 0.5 separates this exactly
r3 <- warncatch(quiet(earth(x3, ysep, degree = 1, glm = list(family = binomial))))
f3 <- r3$value
cat("\npart 3 (separation) warnings:\n")
for (w in r3$warnings) cat("  WARNING:", w, "\n")
if (!is.character(f3)) {
  pr3 <- predict(f3, type = "response")
  cat(sprintf("        fitted probabilities range [%.3g, %.3g] over n=%d; below 1e-6: %d; above 1-1e-6: %d\n",
    min(pr3), max(pr3), n3, sum(pr3 < 1e-6), sum(pr3 > 1 - 1e-6)))
}
ok3 <- !is.character(f3) && any(grepl("did not converge", r3$warnings)) &&
  any(grepl("numerically 0 or 1", r3$warnings)) && any(grepl("did not converge for response", r3$warnings)) &&
  min(pr3) < 1e-6 && max(pr3) > 1 - 1e-6

## ================= Part 4: three classes =================
f4 <- quiet(earth(x, yfac, degree = 2, glm = list(family = binomial)))
ncols4 <- ncol(f4$fitted.values)
bx4 <- f4$bx
maxdiff_cols <- sapply(seq_len(ncols4), function(k) {
  ind <- as.numeric(yfac == levels(yfac)[k])
  gk <- glm(ind ~ bx4 - 1, family = binomial)
  max(abs(coef(gk) - f4$glm.coefficients[, k]))
})
pr4 <- predict(f4, type = "response"); rs4 <- rowSums(pr4)
suppressMessages(library(nnet))
mn <- nnet::multinom(yfac ~ bx4 - 1, trace = FALSE, maxit = 1000)
prmn <- predict(mn, type = "probs")[, levels(yfac)]
mn_diff <- max(abs(pr4 - prmn))
cat(sprintf("\npart 4: %d response columns (%s); dirs shared across columns (one pass, %d forward terms)\n",
  ncols4, paste(colnames(f4$fitted.values), collapse = ","), nrow(f4$dirs)))
cat(sprintf("        per-class glm vs separate glm(indicator~bx-1,family=binomial): max abs diff %s\n",
  paste(signif(maxdiff_cols, 3), collapse = ", ")))
cat(sprintf("        row sums of the 3 fitted probabilities: range [%.4f, %.4f]; largest diff vs nnet::multinom on the same bx: %.4f\n",
  min(rs4), max(rs4), mn_diff))
ok4_cols <- ncols4 == nlevels(yfac)
ok4_glm <- all(maxdiff_cols < 1e-8)
ok4_sum1 <- isTRUE(all.equal(as.numeric(rs4), rep(1, length(rs4)), tolerance = 1e-6))

## ================= Part 5: binary y as 0/1, logical, factor =================
ylog <- as.logical(y01); yfac2 <- factor(y01, labels = c("no", "yes"))
f5_01 <- f1  # identical call to part 1's fit
f5_log <- quiet(earth(x, ylog, degree = 2, glm = list(family = binomial)))
f5_fac <- quiet(earth(x, yfac2, degree = 2, glm = list(family = binomial)))
same_dirs5 <- identical(f5_01$dirs, f5_log$dirs) && identical(f5_01$dirs, f5_fac$dirs)
same_glm5 <- isTRUE(all.equal(f5_01$glm.coefficients[, 1], f5_log$glm.coefficients[, 1], check.attributes = FALSE)) &&
  isTRUE(all.equal(f5_01$glm.coefficients[, 1], f5_fac$glm.coefficients[, 1], check.attributes = FALSE))
pr5_log <- predict(f5_log, type = "response"); pr5_fac <- predict(f5_fac, type = "response")
same_pred5 <- max(abs(pr_resp - pr5_log)) < 1e-10 && max(abs(pr_resp - pr5_fac)) < 1e-10
cat(sprintf("\npart 5: same dirs across 0/1, logical, 2-level factor: %s; same glm.coefficients: %s; same predictions: %s\n",
  same_dirs5, same_glm5, same_pred5))
cat(sprintf("        glm.coefficients column name: 0/1 -> \"%s\", logical -> \"%s\", factor -> \"%s\"\n",
  colnames(f5_01$glm.coefficients), colnames(f5_log$glm.coefficients), colnames(f5_fac$glm.coefficients)))
ok5 <- same_dirs5 && same_glm5 && same_pred5

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb21.1a %s glm.coefficients equal (relative 1e-8) glm(y ~ bx - 1, family=binomial) on the fit's own bx\n", ok1_coef))
cat(sprintf("CHECK bb21.1b %s the glm refit does not change dirs/cuts/selected.terms: identical to earth(x,y,degree=2) without glm\n", ok1_passes))
cat(sprintf("CHECK bb21.1c %s predict(fit,type=\"response\") equals plogis(bx %%*%% glm.coefficients) (max abs diff %.3g)\n", ok1_pred, resp_diff))
cat(sprintf("CHECK bb21.2 %s with integer (1:3) and non-integer weights, the glm refit matches glm(y~bx-1,family=binomial,weights=w) exactly\n", ok2))
cat(sprintf("CHECK bb21.3 %s under perfect separation earth's glm refit warns (glm.fit non-convergence, fitted probabilities numerically 0/1, and earth's own \"did not converge for response\" wrapper) and fitted probabilities reach both extremes\n", ok3))
cat(sprintf("CHECK bb21.4a %s a %d-level factor response gives %d response/glm columns sharing one dirs matrix (one forward+prune pass for all classes)\n", ok4_cols, nlevels(yfac), ncols4))
cat(sprintf("CHECK bb21.4b %s each column's glm is a separate binomial glm of that class indicator on bx, identical to fitting glm(indicator~bx-1,family=binomial) alone\n", ok4_glm))
cat(sprintf("CHECK bb21.4c %s HYPOTHESIS the three per-class fitted probabilities of a row sum to 1\n", ok4_sum1))
cat(sprintf("CHECK bb21.4d %s instead each class column is an independent one-vs-rest binomial glm with no sum-to-1 constraint: row sums range %.4f to %.4f, and the largest per-class probability difference from a proper nnet::multinom fit on the same bx is %.4f\n",
  !ok4_sum1, min(rs4), max(rs4), mn_diff))
cat(sprintf("CHECK bb21.5 %s a binary y given as 0/1, logical, or a 2-level factor gives identical dirs, glm.coefficients and predictions (only the response's column name changes)\n", ok5))
