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

r_version <- R.version.string
earth_version <- as.character(utils::packageVersion("earth"))

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

# Run expr, collecting every warning's message instead of letting it print
# (review round 1, #42 findings 4/6): GLM-3/GLM-4 and the multinom
# reference are only valid "where glm converges without a warning", so a
# caller needs the warnings, not just the fitted object.
with_warnings <- function(expr) {
  messages <- character(0)
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      messages <<- c(messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = messages)
}

calls <- list(
  # No request fields; every call's result already carries r_version and
  # earth_version (below), so this just reads them off with nothing else in
  # the way, for a fixture's versions block (component fixtures have no
  # fit_earth.R result to take them from otherwise).
  versions = function(req) list(),

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

  # Plain least squares on fixed columns (base R's lm.fit/lm.wfit, not
  # earth's; the coefficients-of-fixed-terms component test compares
  # pymars against this). w (optional) routes to lm.wfit, the weighted
  # variant LA-4/PRUNE-8 need (review round 1, #42 finding 5).
  lm_fit = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    if (ncol(y) == 1) {
      y <- y[, 1]
    }
    w <- req$w
    fit <- if (is.null(w)) lm.fit(x, y) else lm.wfit(x, y, unlist(w))
    list(
      coefficients = mat_json(as.matrix(fit$coefficients)),
      residuals = mat_json(as.matrix(fit$residuals)),
      rank = fit$rank
    )
  },

  # A fresh earth fit with one column as a genuine R factor (plus optional
  # other numeric columns), the "factor" side of S19's comparison
  # (VALIDATION_PLAN.md, "Categorical inputs": "earth on the factor must
  # equal earth on the dummies"); driver.run_earth's CSV-based x is numeric
  # only, so this reaches past it for a plain data.frame(...) call, the
  # standard, documented way to hand earth a factor column, not internals.
  # Only a summary comes back (fitted values, gcv, term count), because a
  # factor fit's own dirs/cuts/bx are not comparable in shape with the
  # dummy-encoded fit's.
  earth_factor_fit = function(req) {
    y <- as.numeric(unlist(req$y))
    cols <- list()
    if (!is.null(req$labels)) {
      cols$.f <- factor(unlist(req$labels))
    }
    other <- req$other_x
    if (!is.null(other) && length(other) > 0) {
      other_x <- to_matrix(other)
      for (j in seq_len(ncol(other_x))) {
        cols[[paste0("x", j)]] <- other_x[, j]
      }
    }
    df <- as.data.frame(cols)
    args <- c(list(x = df, y = y), req$earth_args)
    fit <- do.call(earth, args)
    list(
      fitted = vec_json(as.numeric(fitted(fit))),
      gcv = fit$gcv,
      rsq = fit$rsq,
      nterms = nrow(fit$dirs)
    )
  },

  # A fresh earth fit on (x, y), predicted at newx; the prediction-at-new-
  # points component test compares this with pymars' basis_matrix()/predict.
  # Returns the model (dirs, cuts, selected_terms, coefficients) alongside
  # pred (review round 1, #42 finding 2): without them, matching pred alone
  # needs a whole forward-pass refit, not TERM-3 in isolation.
  predict_earth = function(req) {
    x <- to_matrix(req$x)
    y <- to_matrix(req$y)
    if (ncol(y) == 1) {
      y <- y[, 1]
    }
    args <- c(list(x = x, y = y), req$earth_args)
    fit <- do.call(earth, args)
    newx <- to_matrix(req$newx)
    list(
      pred = mat_json(predict(fit, newx, type = req$type %||% "link")),
      dirs = mat_json(fit$dirs),
      cuts = mat_json(fit$cuts),
      selected_terms = vec_json(as.integer(fit$selected.terms)),
      coefficients = mat_json(fit$coefficients)
    )
  },

  # R's own glm() (or glm.fit()) on fixed columns, unpenalized: the reference
  # for the binomial refit's coefficients and fitted probabilities. Returns
  # converged and warnings too (review round 1, #42 finding 6): GLM-3
  # compares coefficients only "where glm converges without a warning".
  # Review round 2 checked glm.fit's default control (epsilon = 1e-8,
  # maxit = 25) directly: tightening epsilon to 1e-15 moves the binomial
  # reference's coefficients by only 2e-8 to 3e-10, well inside GLM-3's
  # tolerance, so it was not the multinom bug's cause. Tightened here
  # anyway (defensive, same reasoning as multinom_fit's reltol) since it
  # costs nothing on a well-posed IRLS fit.
  glm_fit = function(req) {
    x <- to_matrix(req$x)
    y <- as.numeric(unlist(req$y))
    fam <- get(req$family %||% "binomial")()  # glm.fit needs a family object,
                                               # not the bare family function
    ctrl <- glm.control(epsilon = 1e-12, maxit = 100)
    run <- with_warnings(glm.fit(x, y, family = fam, control = ctrl))
    fit <- run$value
    list(
      coefficients = vec_json(as.numeric(fit$coefficients)),
      fitted_values = vec_json(as.numeric(fit$fitted.values)),
      converged = fit$converged,
      warnings = vec_json(run$warnings)
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
  # maxit raises nnet's default cap of 100 iterations when the caller's
  # design needs more (review round 1, #42/#43 finding 1/3/5: separable
  # labels never converge at all, whatever maxit is, so the fix is a
  # non-separable design plus checking convergence, not just a larger cap).
  # reltol (review round 2, #42/#43 blocking finding 1): convergence code 0
  # only means the objective stopped changing by more than reltol, and
  # nnet's own default (1e-8) is loose enough that the stopping point can
  # be off from GLM-2's actual minimum by more than GLM-3's tolerance, so
  # this defaults to something far tighter; a caller doing its own
  # stability check (gen_fixtures.py's _assert_multinom_is_stable) passes
  # a distinctly different reltol for the second fit.
  # Returns convergence (nnet's own code; 0 is converged) and warnings.
  multinom_fit = function(req) {
    suppressMessages(library(nnet))
    x <- to_matrix(req$x)
    y <- factor(unlist(req$y))
    df <- as.data.frame(x)
    df$.y <- y
    maxit <- req$maxit %||% 10000
    reltol <- req$reltol %||% 1e-15
    run <- with_warnings(
      nnet::multinom(
        .y ~ . - 1,
        data = df,
        trace = FALSE,
        maxit = maxit,
        reltol = reltol
      )
    )
    fit <- run$value
    list(
      coefficients = mat_json(coef(fit)),
      levels = vec_json(levels(y)),
      fitted = mat_json(predict(fit, newdata = df, type = "probs")),
      convergence = fit$convergence,
      warnings = vec_json(run$warnings)
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
result <- fn(req)
result$r_version <- r_version
result$earth_version <- earth_version
write_result(result, argv[2])
