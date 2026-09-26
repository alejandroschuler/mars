# Fit R's earth on a block of train/test CSV pairs in one R process.
#
# One R process pays its startup cost once and fits every job in the block
# (VALIDATION_PLAN.md, "Build": "earth runs through Rscript in blocks"). earth
# is used only as a black box here: this script calls `earth()` and
# `predict()` and reads their outputs; it never reads earth's source.
#
# Usage: Rscript fit_earth_block.R <manifest.json>
#
# The manifest is {"jobs": [{"id", "train_csv", "test_csv", "out_json",
# "args", "glm_family"}, ...]}. "args" are earth() arguments (for example
# degree, penalty, thresh, minspan, endspan, "fast.k", "Adjust.endspan");
# "glm_family" is null for a regression job or "binomial" for a binary one.
# Each job's train CSV has columns x1..xp, y; its test CSV has x1..xp only.
# Each job writes its own out_json atomically (a temp file, then a rename),
# and one job's error never stops the rest of the block.
suppressMessages({
  library(earth)
  library(jsonlite)
})

write_result <- function(out_json, result) {
  tmp <- paste0(out_json, ".tmp.", Sys.getpid())
  write_json(result, tmp, digits = NA, auto_unbox = TRUE, null = "null")
  file.rename(tmp, out_json)
}

run_job <- function(job) {
  tr <- read.csv(job$train_csv)
  te <- read.csv(job$test_csv)
  xcols <- grep("^x", names(tr), value = TRUE)
  xcols <- xcols[order(as.integer(sub("^x", "", xcols)))]
  x <- as.matrix(tr[, xcols, drop = FALSE])
  y <- tr$y
  xt <- as.matrix(te[, xcols, drop = FALSE])

  args <- job$args
  fam <- job$glm_family
  if (!is.null(fam)) args$glm <- list(family = get(fam))
  fit_args <- c(list(x = x, y = y), args)

  t0 <- proc.time()[["elapsed"]]
  fit <- do.call(earth, fit_args)
  fit_seconds <- proc.time()[["elapsed"]] - t0

  pred_type <- if (is.null(fam)) "link" else "response"
  pred_test <- as.numeric(predict(fit, xt, type = pred_type))

  selected <- fit$selected.terms
  dirs_selected <- fit$dirs[selected, , drop = FALSE]
  used <- which(colSums(abs(dirs_selected)) > 0) - 1L # 0-indexed, matches Python

  list(
    predictions = pred_test,
    n_terms = length(selected),
    # I(): jsonlite's auto_unbox otherwise turns a length-1 (or length-0)
    # integer vector into a JSON scalar instead of an array, which breaks the
    # Python side's tuple(covariates_used) whenever a fit selects exactly one
    # covariate (or none).
    covariates_used = I(as.integer(used)),
    fit_seconds = fit_seconds
  )
}

main <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  manifest <- fromJSON(a[1], simplifyVector = FALSE)
  for (job in manifest$jobs) {
    result <- tryCatch(run_job(job), error = function(e) {
      list(error = paste0(class(e)[1], ": ", conditionMessage(e)))
    })
    write_result(job$out_json, result)
  }
}

main()
