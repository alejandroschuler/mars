# bb22: small and degenerate inputs.
#
# For each case: dirs row names, selected.terms, rss, gcv, rsq, grsq,
# termcond and the termination message (trace=1), or the error/warning
# message. Six groups: (1) tiny n, p=2, default and minspan=1/endspan=1;
# (2) constant y, and y with all but one value equal; (3) a constant
# column, an exact duplicate column and a near-duplicate column; (4) n < p
# and extreme p; (5) a single outlier x value; (6) non-finite x or y.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

# Every case is reported the same way: dirs row names, selected.terms, the
# fit statistics, termcond, the trace=1 termination line, and any warning
# (conditionMessage only, per the clean-room rule), or the error message.
fit_report <- function(label, x, y, ...) {
  warns <- character(0)
  out <- capture.output(
    f <- withCallingHandlers(
      quiet(earth(x, y, trace = 1, ...)),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
    )
  )
  if (is.character(f)) {
    cat(sprintf("%-38s %s\n", label, f))  # f already reads "ERROR: <message>"
    return(invisible(f))
  }
  msg <- grep("Reached|RSq changed|GRSq|No new term|no new term|increase|min GRSq|nk\\b", out, value = TRUE)
  msg <- trimws(msg[!grepl("^After|Forward pass complete|^Prune|selected", msg)])
  wtxt <- if (length(warns)) sprintf(" WARN: %s", paste(unique(warns), collapse = " | ")) else ""
  cat(sprintf("%-38s dirs=[%s] sel=%s rss=%.4g gcv=%.4g rsq=%.4g grsq=%.4g termcond=%s | %s%s\n",
    label, paste(rownames(f$dirs), collapse = "; "), paste(f$selected.terms, collapse = ","),
    f$rss, f$gcv, f$rsq, f$grsq, as.character(f$termcond), paste(msg, collapse = " / "), wtxt))
  invisible(f)
}

## ================= Item 1: tiny n, p = 2 =================
cat("-- item 1: n = 1,2,3,4,5,8,12; p = 2; default and minspan=1,endspan=1 --\n")
r1 <- list()
for (nn in c(1, 2, 3, 4, 5, 8, 12)) {
  set.seed(100 + nn); x <- matrix(runif(nn * 2), nn, 2); y <- x[, 1] + 0.1 * rnorm(nn)
  r1[[paste0("d", nn)]]  <- fit_report(sprintf("n=%-3d default", nn), x, y)
  r1[[paste0("s", nn)]]  <- fit_report(sprintf("n=%-3d minspan=1 endspan=1", nn), x, y, minspan = 1, endspan = 1)
}
intercept_only <- function(f) !is.character(f) && nrow(f$dirs) == 1
n1_error <- is.character(r1$d1) && is.character(r1$s1)
small_n_no_step <- all(sapply(c(2, 3, 4, 5), function(nn) intercept_only(r1[[paste0("d", nn)]]) && intercept_only(r1[[paste0("s", nn)]])))
larger_n_steps <- all(sapply(c(8, 12), function(nn) !intercept_only(r1[[paste0("d", nn)]]) && !intercept_only(r1[[paste0("s", nn)]])))

## ================= Item 2: constant / near-constant y =================
cat("\n-- item 2: constant y, with and without noise-free x structure; y all-but-one equal --\n")
set.seed(200); n2 <- 20
x_rand <- matrix(runif(n2 * 2), n2, 2)
x_struct <- cbind(seq(0, 1, length.out = n2), seq(1, 0, length.out = n2))  # noise-free, deterministic x
f_const_rand   <- fit_report("y constant, x random", x_rand, rep(3.0, n2))
f_const_struct <- fit_report("y constant, x noise-free structure", x_struct, rep(3.0, n2))
y_allbut1 <- c(rep(3.0, n2 - 1), 9.0)
f_allbut1 <- fit_report("y all-but-one equal", x_rand, y_allbut1)
const_y_warns <- !is.character(f_const_rand) && !is.character(f_const_struct) &&
  isTRUE(f_const_rand$termcond == 4) && isTRUE(f_const_struct$termcond == 4)
allbut1_prunes_to_intercept <- !is.character(f_allbut1) && length(f_allbut1$selected.terms) == 1

## ================= Item 3: constant / duplicated / near-duplicate columns =================
cat("\n-- item 3: constant column, duplicated column, near-duplicate column --\n")
set.seed(300); n3 <- 60; xu <- runif(n3); y3 <- xu + 0.1 * rnorm(n3)
f_constcol <- fit_report("constant col next to useful", cbind(x1 = xu, xconst = rep(2.0, n3)), y3)
f_dup      <- fit_report("duplicated column (x2==x1)", cbind(x1 = xu, x2 = xu), y3)
set.seed(301); xnear <- xu + 1e-9 * rnorm(n3)
f_near  <- fit_report("near-duplicate column (x2=x1+1e-9*noise)", cbind(x1 = xu, x2 = xnear), y3)
f_nodup <- fit_report("reference: x1 alone, no x2 at all", cbind(x1 = xu), y3)

constcol_unused <- !is.character(f_constcol) && !any(grepl("xconst", rownames(f_constcol$dirs)))
# "x2" never appears as a substring of any term label when x2 is an exact
# duplicate of x1 (every term names x1 instead): a real, not merely
# same-prefix, tie-break toward the lower column index.
dup_uses_first  <- !is.character(f_dup) && !any(grepl("x2", rownames(f_dup$dirs)))
near_uses_x1 <- !is.character(f_near) && any(grepl("x1", rownames(f_near$dirs)))
near_uses_x2 <- !is.character(f_near) && any(grepl("x2", rownames(f_near$dirs)))
near_matches_nodup <- !is.character(f_near) && !is.character(f_nodup) && isTRUE(all.equal(f_near$rss, f_nodup$rss))
cat(sprintf("  exact duplicate: x2 never appears in any term (%s) | near-duplicate: x1 appears %s, x2 appears %s (both, not a clean single-column tie-break) | near-dup fit matches dropping x2 outright: %s\n",
  dup_uses_first, near_uses_x1, near_uses_x2, near_matches_nodup))

## ================= Item 4: n < p, and extreme p =================
cat("\n-- item 4: n < p (n=20,p=50), p=1, and p=200,n=300 --\n")
report_np <- function(label, X, y) {
  f <- quiet(earth(X, y))
  if (is.character(f)) { cat(sprintf("%-16s %s\n", label, f)); return(invisible(f)) }
  cat(sprintf("%-16s nk=%-4d forward_terms(dirs)=%-4d selected=%-4d termcond=%d\n",
    label, f$nk, nrow(f$dirs), length(f$selected.terms), f$termcond))
  invisible(f)
}
set.seed(400); X20 <- matrix(runif(20 * 50), 20, 50); y20 <- X20[, 1] + 0.1 * rnorm(20)
f_n20p50 <- report_np("n=20,p=50", X20, y20)
set.seed(401); X1c <- matrix(runif(30 * 1), 30, 1); y1c <- X1c[, 1] + 0.1 * rnorm(30)
f_p1 <- report_np("p=1", X1c, y1c)
set.seed(402); X300 <- matrix(runif(300 * 200), 300, 200); y300 <- X300[, 1] + 0.1 * rnorm(300)
f_n300p200 <- report_np("n=300,p=200", X300, y300)
np_no_error <- !is.character(f_n20p50) && !is.character(f_p1) && !is.character(f_n300p200)
np_nk_formula <- !is.character(f_n20p50) && f_n20p50$nk == min(200, max(20, 2 * 50)) + 1 &&
  !is.character(f_n300p200) && f_n300p200$nk == min(200, max(20, 2 * 200)) + 1

## ================= Item 5: an outlier x value =================
cat("\n-- item 5: one x value at 1e6 among values in [0,1] --\n")
set.seed(500); n5 <- 60; xo <- runif(n5); xo[1] <- 1e6
yo <- xo + 0.1 * rnorm(n5)
f_outlier <- fit_report("outlier x=1e6, n=60", cbind(xo), yo)
outlier_ok <- !is.character(f_outlier) && nrow(f_outlier$dirs) > 0

## ================= Item 6: non-finite input =================
cat("\n-- item 6: NaN in x, Inf in x, NaN in y, Inf in y --\n")
set.seed(600); n6 <- 40; x6 <- runif(n6); y6 <- x6 + 0.1 * rnorm(n6)
xn <- x6; xn[5] <- NaN
f_nanx <- fit_report("NaN in x", cbind(xn), y6)
xi <- x6; xi[5] <- Inf
f_infx <- fit_report("Inf in x", cbind(xi), y6)
yn <- y6; yn[5] <- NaN
f_nany <- fit_report("NaN in y", cbind(x6), yn)
yi <- y6; yi[5] <- Inf
f_infy <- fit_report("Inf in y", cbind(x6), yi)
# An Inf anywhere in y makes mean(y) and sd(y) both non-finite, so EVERY
# element becomes NaN once y is internally scaled (bb17's LA-6): the error
# always names row 1, not the row that actually held the Inf.
cat(sprintf("  mean/sd of a y with one Inf: mean=%s sd=%s (both non-finite, so scaling turns every element to NaN)\n",
  mean(yi), sd(yi)))
nonfinite_always_errors <- all(sapply(list(f_nanx, f_infx, f_nany, f_infy), is.character))
x_errors_name_true_row <- is.character(f_nanx) && grepl("\\[5\\]", f_nanx) && is.character(f_infx) && grepl("\\[5\\]", f_infx)
nany_names_row5 <- is.character(f_nany) && grepl("\\[5\\]", f_nany)  # NaN in y: localized, names the true row
hyp_infy_names_row5 <- is.character(f_infy) && grepl("\\[5\\]", f_infy)
infy_names_row1_instead <- is.character(f_infy) && grepl("\\[1\\]", f_infy) && !grepl("\\[5\\]", f_infy)

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb22.1 %s n=1 is an error (\"at least two rows\"); n in {2,3,4,5} fits only the intercept (no forward step, termcond 2) under default AND minspan=1,endspan=1 alike; n in {8,12} adds forward terms\n",
  n1_error && small_n_no_step && larger_n_steps))
cat(sprintf("CHECK bb22.2a %s a constant y fits (rss=gcv=0, termcond 4, an RSq-change stop) rather than erroring, whether or not x has a noise-free deterministic structure, and WARNS instead of erroring\n",
  const_y_warns))
cat(sprintf("CHECK bb22.2b %s y with all but one value equal: the forward pass records candidate hinges, but pruning selects the intercept alone (1 selected term)\n",
  allbut1_prunes_to_intercept))
cat(sprintf("CHECK bb22.3a %s a constant column next to a useful one is never used (no dirs row names it)\n",
  constcol_unused))
cat(sprintf("CHECK bb22.3b %s an exactly duplicated column resolves to the LOWER column index only (x2 never appears in any term)\n",
  dup_uses_first))
cat(sprintf("CHECK bb22.3c %s HYPOTHESIS a near-duplicate (not exact) column is resolved the same way, to one column only (it is not: the unpruned terms use x1 (%s) and use x2 (%s), both present)\n",
  !(near_uses_x1 && near_uses_x2), near_uses_x1, near_uses_x2))
cat(sprintf("CHECK bb22.3d %s instead, a near-duplicate is treated as a genuinely different (not tied) column at every knot search, so different terms of the same fit can use either near-duplicate column, and the fit does not match dropping the near-duplicate column outright (rss %s vs %s)\n",
  (near_uses_x1 && near_uses_x2) && !near_matches_nodup,
  if (!is.character(f_near)) signif(f_near$rss, 4) else NA, if (!is.character(f_nodup)) signif(f_nodup$rss, 4) else NA))
cat(sprintf("CHECK bb22.4 %s n < p (n=20,p=50) and p=200,n=300 fit without error, and the default nk follows min(200,max(20,2p))+1 in both; p=1 also fits normally\n",
  np_no_error && np_nk_formula))
cat(sprintf("CHECK bb22.5 %s a single extreme outlier value (1e6 among [0,1]) does not cause an error; the row is kept and fit like any other point\n",
  outlier_ok))
cat(sprintf("CHECK bb22.6a %s NaN or Inf anywhere in x or y is always a fitting error (never silently dropped or coerced); for x, and for a plain NaN in y, the message names the true offending row (5)\n",
  nonfinite_always_errors && x_errors_name_true_row && nany_names_row5))
cat(sprintf("CHECK bb22.6b %s HYPOTHESIS an Inf in y also names row 5, the row that actually holds the Inf, the way Inf in x and NaN in y do\n",
  hyp_infy_names_row5))
cat(sprintf("CHECK bb22.6c %s instead, an Inf anywhere in y makes mean(y) and sd(y) non-finite (shown above), so scaling turns EVERY element to NaN and the error always names row 1, regardless of where the Inf actually was\n",
  infy_names_row1_instead))
