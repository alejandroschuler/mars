# Fit earth on a CSV train/test pair with arguments from a JSON config.
# Usage: Rscript fit_earth.R train.csv test.csv config.json out.json
suppressMessages({
  library(earth)
  library(jsonlite)
})
a <- commandArgs(trailingOnly = TRUE)
tr <- read.csv(a[1])
te <- read.csv(a[2])
cfg <- fromJSON(a[3])
xcols <- grep("^x", names(tr), value = TRUE)
x <- as.matrix(tr[, xcols, drop = FALSE])
y <- tr$y
xt <- as.matrix(te[, xcols, drop = FALSE])
w <- if ("w" %in% names(tr)) tr$w else NULL
reps <- if (is.null(cfg$timing_reps)) 1 else cfg$timing_reps
cfg$timing_reps <- NULL
fam <- cfg$glm_family
cfg$glm_family <- NULL
args <- c(list(x = x, y = y), cfg)
if (!is.null(w)) args$weights <- w
if (!is.null(fam)) args$glm <- list(family = get(fam))
times <- numeric(reps)
for (r in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  fit <- do.call(earth, args)
  times[r] <- proc.time()[["elapsed"]] - t0
}
# RSS path over forward-pass terms, in the order the forward pass added them.
fwd_rss <- NULL
if (identical(cfg$pmethod, "none")) {
  bx <- fit$bx
  fwd_rss <- sapply(seq_len(ncol(bx)), function(j) {
    f <- lm.fit(bx[, 1:j, drop = FALSE], y)
    sum(f$residuals^2)
  })
}
pred_type <- if (is.null(fam)) "link" else "response"
out <- list(
  dirs = unname(fit$dirs), cuts = unname(fit$cuts),
  term_names = rownames(fit$dirs),
  selected = fit$selected.terms,
  coef = unname(as.numeric(fit$coefficients)),
  glm_coef = if (is.null(fam)) NULL else unname(as.numeric(fit$glm.coefficients)),
  rss = fit$rss, rsq = fit$rsq, gcv = fit$gcv, grsq = fit$grsq,
  rss_per_subset = fit$rss.per.subset, gcv_per_subset = fit$gcv.per.subset,
  termcond = fit$termcond, fwd_rss = fwd_rss,
  pred_train = as.numeric(predict(fit, x, type = pred_type)),
  pred_test = as.numeric(predict(fit, xt, type = pred_type)),
  time_min = min(times), time_all = times
)
write_json(out, a[4], digits = NA, auto_unbox = TRUE, null = "null")
