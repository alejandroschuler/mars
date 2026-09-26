# Fit R's earth on a block of jobs in one R process.
#
# One R process pays its startup cost once and fits every job in the block
# (VALIDATION_PLAN.md, "Build": "earth runs through Rscript in blocks"). earth
# is used only as a black box here: this script calls `earth()` and
# `predict()` and reads their outputs; it never reads earth's source.
#
# Usage: Rscript fit_earth_block.R <manifest.json>
#
# The manifest is {"jobs": [{"id", "train_x", "train_y", "test_x", "n_train",
# "n_test", "p", "out_json", "args", "glm_family"}, ...]}. "train_x"/"test_x"
# are raw float64 files in R's own column-major order (learners.py writes
# them with numpy's Fortran order, so `matrix(readBin(...), nrow, ncol)`, the
# default byrow = FALSE, reconstructs them exactly); "train_y" is a flat
# float64 vector. This avoids a decimal round trip: reading a CSV of
# %.17g-formatted numbers back in R can be off by 1-2 ULP from the value
# Python wrote, which a review measured on real data, and every arm must see
# exactly the same numbers. "args" are earth() arguments (for example degree,
# penalty, thresh, minspan, endspan, "fast.k", "Adjust.endspan"); "glm_family"
# is null for a regression job or "binomial" for a binary one. Each job
# writes its own out_json atomically (a temp file, then a rename), and one
# job's error never stops the rest of the block.
suppressMessages({
  library(earth)
  library(jsonlite)
})

read_matrix <- function(path, nrow, ncol) {
  matrix(readBin(path, what = "double", n = nrow * ncol, size = 8), nrow = nrow, ncol = ncol)
}

read_vector <- function(path, n) {
  readBin(path, what = "double", n = n, size = 8)
}

write_result <- function(out_json, result) {
  tmp <- paste0(out_json, ".tmp.", Sys.getpid())
  # digits = I(17): digits = NA still rounds to 15 significant digits: a
  # review found this cost precision on the predictions written back.
  write_json(result, tmp, digits = I(17), auto_unbox = TRUE, null = "null")
  file.rename(tmp, out_json)
}

run_job <- function(job) {
  x <- read_matrix(job$train_x, job$n_train, job$p)
  y <- read_vector(job$train_y, job$n_train)
  xt <- read_matrix(job$test_x, job$n_test, job$p)

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
    # I(): jsonlite's auto_unbox otherwise turns a length-1 (or length-0)
    # vector into a JSON scalar instead of an array, which breaks the Python
    # side's tuple()/array() reconstruction whenever, for example, a fit
    # selects exactly one covariate, or the test set has exactly one row.
    predictions = I(pred_test),
    n_terms = length(selected),
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
