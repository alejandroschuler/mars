# Fit R earth on one or more datasets ("a block") described by a JSON config,
# and write each fit's results as JSON with full double precision.
#
# This script only calls earth() and reads what it returns; it never reads
# earth's C or R source (VALIDATION_PLAN.md, "Instruction files and clean
# room"). One process can fit several datasets, so the caller pays R's
# start-up time once (driver.py explains why).
#
# Usage: Rscript fit_earth.R <config.json>
#
# The config is a JSON object {"jobs": [<job>, ...]}. Each <job> is an object:
#   id                   a label, echoed back in the result
#   train_csv            path to a CSV with columns x0, x1, ... and the
#                        response column(s); read.csv() parses the %.17g
#                        text driver.py writes at full double precision
#   test_csv             optional CSV of the same shape, for predict()
#   x_cols               optional array of column names; default: every
#                        column of train_csv whose name starts with "x"
#   y_cols               optional array of response column names; default
#                        c("y"); more than one name means several responses
#                        (cbind into a matrix); factor_response requires
#                        exactly one name
#   weight_col           optional column name for case weights
#   factor_response      optional bool; if true, y is as.factor()'d before
#                        the fit, so earth treats it as a classification
#                        response (one indicator column per level)
#   earth_args           optional object forwarded as-is to earth(), using
#                        earth's own argument names (for example "nk",
#                        "Auto.linpreds", "Adjust.endspan"); names_map.py
#                        builds this object from pymars parameter names, so
#                        this script needs no knowledge of that mapping
#   glm_family           optional string, e.g. "binomial"; sets
#                        glm = list(family = <get(glm_family)>)
#   trace                optional int 0-9 (default 0), forwarded as earth's
#                        own trace argument
#   trace_file           path to write the trace text to, when trace > 0
#   include_forward_path optional bool (default true); if true and the
#                        response is not a factor, an extra fit with
#                        pmethod = "none" gives the forward-pass RSS path
#   out                  path to write this job's result JSON to
#
# Each result also carries the R and earth versions, so a fixture or a run
# log always says which versions produced it.
suppressMessages({
  library(earth)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

r_version <- R.version.string
earth_version <- as.character(utils::packageVersion("earth"))

write_result <- function(x, path) {
  # na = "string" writes Inf/-Inf/NaN/NA as those exact strings rather than
  # collapsing all of them to JSON null (jsonlite's default): a GCV of Inf
  # (S11's edge case, effective parameters at or above n) must stay
  # distinguishable from a merely missing value. driver.py's _desanitize()
  # converts these four strings back to real floats after json.loads().
  write_json(x, path, digits = NA, auto_unbox = TRUE, null = "null", na = "string")
}

# jsonlite serializes a genuine R matrix as a nested array with a shape that
# does not depend on its size (a (1, 1) or (M, 1) matrix still comes out as a
# nested array, never collapsed to a bare number or a flat list), so this only
# guards the NULL case; it never reads or reshapes the values.
mat_json <- function(x) if (is.null(x)) NULL else as.matrix(x)

# A plain vector, unlike a matrix, has auto_unbox collapse it to a bare scalar
# when it has length 1 (an intercept-only model, for example). jsonlite's I()
# marks it to stay an array at every length; still NULL when x is NULL.
vec_json <- function(x) if (is.null(x)) NULL else I(unname(x))

# The forward-pass RSS path: the RSS after each term is added, in the order
# the forward pass added it. earth has no direct output for this (its own
# rss.per.subset is indexed by *pruned* subset size, not by forward step), so
# this refits the cumulative columns of a pmethod = "none" basis with
# lm.fit(), exactly as the validation/legacy/ prototype did to produce the
# F14 and F15 findings in VALIDATION_PLAN.md.
forward_rss_path <- function(fit_args, y) {
  fwd_args <- fit_args
  fwd_args$pmethod <- "none"
  fwd_args$trace <- 0
  fwd_fit <- do.call(earth, fwd_args)
  bx <- fwd_fit$bx
  ymat <- as.matrix(y)
  vapply(seq_len(ncol(bx)), function(j) {
    sum(lm.fit(bx[, seq_len(j), drop = FALSE], ymat)$residuals^2)
  }, numeric(1))
}

read_xy <- function(csv_path, x_cols, y_cols, weight_col, factor_response) {
  d <- read.csv(csv_path)
  xcols <- x_cols %||% grep("^x", names(d), value = TRUE)
  x <- as.matrix(d[, xcols, drop = FALSE])
  if (isTRUE(factor_response)) {
    stopifnot("factor_response needs exactly one y column" = length(y_cols) == 1)
    y <- factor(d[[y_cols[1]]])
  } else if (length(y_cols) == 1) {
    y <- d[[y_cols[1]]]
  } else {
    y <- as.matrix(d[, y_cols, drop = FALSE])
  }
  w <- if (!is.null(weight_col) && weight_col %in% names(d)) d[[weight_col]] else NULL
  list(x = x, y = y, w = w, xcols = xcols)
}

fit_one <- function(job) {
  ycols <- unlist(job$y_cols) %||% "y"
  xcols_in <- unlist(job$x_cols)
  train <- read_xy(job$train_csv, xcols_in, ycols, job$weight_col, job$factor_response)

  args <- c(list(x = train$x, y = train$y), job$earth_args)
  if (!is.null(train$w)) {
    args$weights <- train$w
  }
  fam <- job$glm_family
  if (!is.null(fam)) {
    args$glm <- list(family = get(fam))
  }
  trace <- job$trace %||% 0
  args$trace <- trace

  if (trace > 0 && !is.null(job$trace_file)) {
    con <- file(job$trace_file, "wt")
    sink(con)
    fit <- tryCatch(do.call(earth, args), finally = {
      sink()
      close(con)
    })
  } else {
    fit <- do.call(earth, args)
  }

  fwd_rss <- NULL
  if ((job$include_forward_path %||% TRUE) && !isTRUE(job$factor_response)) {
    fwd_rss <- forward_rss_path(args, train$y)
  }

  pred_type <- if (is.null(fam)) "link" else "response"
  pred_train <- predict(fit, train$x, type = pred_type)
  pred_test <- NULL
  if (!is.null(job$test_csv)) {
    test <- read_xy(job$test_csv, train$xcols, ycols, job$weight_col, job$factor_response)
    pred_test <- predict(fit, test$x, type = pred_type)
  }

  list(
    id = job$id,
    dirs = mat_json(fit$dirs),
    cuts = mat_json(fit$cuts),
    term_names = vec_json(rownames(fit$dirs)),
    selected_terms = vec_json(as.integer(fit$selected.terms)),
    prune_terms = mat_json(fit$prune.terms),
    rss_per_subset = vec_json(as.numeric(fit$rss.per.subset)),
    gcv_per_subset = vec_json(as.numeric(fit$gcv.per.subset)),
    coef = mat_json(fit$coefficients),
    glm_coef = mat_json(fit$glm.coefficients),
    rss = unname(fit$rss),
    rsq = unname(fit$rsq),
    gcv = unname(fit$gcv),
    grsq = unname(fit$grsq),
    termcond = unname(fit$termcond),
    levels = vec_json(fit$levels),
    fitted = mat_json(fit$fitted.values),
    pred_train = mat_json(pred_train),
    pred_test = mat_json(pred_test),
    fwd_rss = vec_json(fwd_rss),
    r_version = r_version,
    earth_version = earth_version
  )
}

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) != 1) {
  stop("usage: Rscript fit_earth.R <config.json>")
}
cfg <- fromJSON(argv[1], simplifyVector = FALSE)
for (job in cfg$jobs) {
  result <- tryCatch(
    fit_one(job),
    error = function(e) {
      list(id = job$id, error = conditionMessage(e),
           r_version = r_version, earth_version = earth_version)
    }
  )
  write_result(result, job$out)
}
