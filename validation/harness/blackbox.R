# Black-box calls into a few of earth's internal functions, for the harness's
# component tests (VALIDATION_PLAN.md, "Component tests"). Each call passes
# arguments in with a JSON request and reads a return value out as JSON; none
# of them read earth's C or R source. Their argument names were learned with
# formals()/args(), which show no code (the allowed exception under
# VALIDATION_PLAN.md, "Instruction files and clean room").
#
# Usage: Rscript blackbox.R <request.json> <out.json>
# The request is {"call": "<name>", ...call-specific fields}; see blackbox.py
# for what each call needs and returns.
suppressMessages({
  library(earth)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

write_result <- function(x, path) {
  # See fit_earth.R's write_result: na = "string" keeps Inf/-Inf/NaN/NA
  # distinguishable from each other and from JSON null; blackbox.py's
  # _desanitize() undoes it after json.loads(). digits = I(17), not NA:
  # jsonlite 2.0.0's NA rounds to 15 significant digits, which does not
  # round-trip a double exactly.
  write_json(x, path, digits = I(17), auto_unbox = TRUE, null = "null", na = "string")
}
mat_json <- function(x) if (is.null(x)) NULL else as.matrix(x)
vec_json <- function(x) if (is.null(x)) NULL else I(unname(x))

# A JSON nested list of rows (blackbox.py's row-major encoding of a numpy
# array) back into a plain numeric matrix.
to_matrix <- function(rows) {
  if (is.null(rows) || length(rows) == 0) {
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }
  do.call(rbind, lapply(rows, function(row) as.numeric(unlist(row))))
}

calls <- list(
  # A grid of (rss.per.subset, ntermsVec) pairs at one penalty and case count;
  # get.gcv takes the whole vectors at once and returns one GCV per entry.
  get_gcv = function(req) {
    list(gcv = vec_json(earth:::get.gcv(
      unlist(req$rss_per_subset), unlist(req$nterms), req$penalty, req$ncases
    )))
  },

  # earth's backward pass on a fixed forward basis (bx, dirs): the caller
  # builds bx/dirs from a real fit (for example with pmethod = "none") and
  # passes them here so the pruning step alone can be compared.
  pruning_pass = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    bx <- to_matrix(req$bx)
    dirs <- to_matrix(req$dirs)
    res <- earth:::pruning.pass(
      x = x, y = y, bx = bx, pmethod = req$pmethod %||% "backward",
      penalty = req$penalty, nprune = req$nprune, trace = 0, dirs = dirs,
      Force.xtx.prune = isTRUE(req$force_xtx_prune),
      Exhaustive.tol = req$exhaustive_tol %||% 1e-10
    )
    list(
      rss_per_subset = vec_json(res$rss.per.subset),
      gcv_per_subset = vec_json(res$gcv.per.subset),
      prune_terms = mat_json(res$prune.terms),
      selected_terms = vec_json(as.integer(res$selected.terms))
    )
  },

  # Plain least squares on fixed columns (base R's lm.fit, not earth's; the
  # coefficients-of-fixed-terms component test compares pymars against this).
  lm_fit = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    if (ncol(y) == 1) {
      y <- y[, 1]
    }
    fit <- lm.fit(x, y)
    list(
      coefficients = mat_json(as.matrix(fit$coefficients)),
      residuals = mat_json(as.matrix(fit$residuals)),
      rank = fit$rank
    )
  },

  # A fresh earth fit on (x, y), predicted at newx; the prediction-at-new-
  # points component test compares this with pymars' basis_matrix()/predict.
  predict_earth = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    if (ncol(y) == 1) {
      y <- y[, 1]
    }
    args <- c(list(x = x, y = y), req$earth_args)
    fit <- do.call(earth, args)
    newx <- to_matrix(req$newx)
    list(pred = mat_json(predict(fit, newx, type = req$type %||% "link")))
  },

  # R's own glm() (or glm.fit()) on fixed columns, unpenalized: the reference
  # for the binomial refit's coefficients and fitted probabilities.
  glm_fit = function(req) {
    x <- to_matrix(req$x)
    y <- as.numeric(unlist(req$y))
    fam <- get(req$family %||% "binomial")()  # glm.fit needs a family object,
                                               # not the bare family function
    fit <- glm.fit(x, y, family = fam)
    list(
      coefficients = vec_json(as.numeric(fit$coefficients)),
      fitted_values = vec_json(as.numeric(fit$fitted.values))
    )
  },

  # A fresh earth fit on (x, y), returning its forward basis (bx, dirs,
  # cuts): the fixture generator's way to get a real fixed basis to prune or
  # to hand to lm_fit/multinom_fit, without adding bx to every fit_earth.R
  # result (test_blackbox.py's _fixed_basis helper already reaches past
  # driver.run_earth the same way, for the same reason: bx would duplicate
  # earth's own forward-pass output in every dataset fixture).
  fit_bx_dirs = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    if (ncol(y) == 1) {
      y <- y[, 1]
    }
    args <- c(list(x = x, y = y), req$earth_args)
    fit <- do.call(earth, args)
    list(
      bx = mat_json(fit$bx),
      dirs = mat_json(fit$dirs),
      cuts = mat_json(fit$cuts),
      selected_terms = vec_json(as.integer(fit$selected.terms))
    )
  },

  # nnet::multinom on fixed columns (earth's own selected basis, including
  # its intercept column, so "- 1" does not add a second one): the reference
  # for multiclass probabilities, which earth's one-binomial-glm-per-class
  # convention does not give directly (VALIDATION_PLAN.md, "Binary outcomes").
  multinom_fit = function(req) {
    suppressMessages(library(nnet))
    x <- to_matrix(req$x)
    y <- factor(unlist(req$y))
    df <- as.data.frame(x)
    df$.y <- y
    fit <- nnet::multinom(.y ~ . - 1, data = df, trace = FALSE)
    list(
      coefficients = mat_json(coef(fit)),
      levels = vec_json(levels(y)),
      fitted = mat_json(predict(fit, newdata = df, type = "probs"))
    )
  }
)

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) != 2) {
  stop("usage: Rscript blackbox.R <request.json> <out.json>")
}
req <- fromJSON(argv[1], simplifyVector = FALSE)
fn <- calls[[req$call]]
if (is.null(fn)) {
  stop(paste("unknown call:", req$call))
}
write_result(fn(req), argv[2])
