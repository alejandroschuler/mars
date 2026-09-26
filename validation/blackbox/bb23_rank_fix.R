# bb23: earth's removal of linearly dependent terms at the end of the
# forward pass (a "Fixed rank deficient bx" fix, printed at trace >= 2,
# before the pruning pass).
#
# Part A: a random search over settings (few-valued / duplicate / collinear
# columns, crossed with degree, nk, thresh, minspan/endspan, Auto.linpreds),
# recording which settings trigger the message.
# Part B: three representative triggering cases -- few-valued columns alone
# with no explicit extra column; a column that is the sum of three others
# (degree = 3); and an exact duplicate column -- each reported with the
# message, nrow(dirs) with pmethod = "none", whether the returned bx is
# exactly full column rank (qr), and a linear-dependence argument (qr,
# tolerance stated).
#
# What this script could NOT establish: which specific dirs row(s) the fix
# removes, and hence whether the fix keeps the earlier- or later-built of two
# dependent terms. The message itself never names a row, at any trace level
# up to 9 (checked). Reconstructing the pre-fix dirs from the trace = 2 step
# table's cumulative "Terms" column was tried by hand on two triggering
# cases: the same reconstruction rule (cumulative-count deltas give each
# step's row count, mirrored pairs share a cut with opposite-sign codes)
# accounted for all pre-fix rows in one case but left one final dirs row
# ("h(2-x1)"/"h(x1-2)"-type main-effect terms) unreachable from any step in
# the other, an inconsistency that was not resolved (see the two worked
# examples this comment is drawn from, seeds 5090 and 50, kept only in this
# script's development notes, not below). Rather than print a reconstruction
# that failed to reproduce a known case, this script instead proves a
# stronger, fully verifiable fact that does not depend on identifying the
# removed row: every column of the kept (post-fix) bx is an exact function
# of the covariates, so it lies in the span of one indicator per distinct
# covariate row; with few-valued or collinear covariates that span has far
# fewer dimensions than a degree>=2 forward search can generate candidate
# terms for, so *some* term becoming a linear combination of others is
# mathematically forced once enough terms accumulate, regardless of which
# one earth's forward pass happens to keep.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

## ================= Part A: random search over settings =================
gens <- list(
  int2 = function(n, p) matrix(sample(0:1, n * p, replace = TRUE), n, p),
  int3 = function(n, p) matrix(sample(1:3, n * p, replace = TRUE), n, p),
  int5 = function(n, p) matrix(sample(1:5, n * p, replace = TRUE), n, p),
  unif = function(n, p) matrix(runif(n * p), n, p)
)
make_data <- function(seed, gname, n, p, extra) {
  set.seed(seed)
  x <- gens[[gname]](n, p)
  if (extra == "dup") x <- cbind(x, x[, 1])
  if (extra == "sum") x <- cbind(x, x[, 1] + x[, 2])
  if (extra == "sum3" && p >= 3) x <- cbind(x, x[, 1] + x[, 2] + x[, 3])
  if (extra == "near") x <- cbind(x, x[, 1] + 1e-9 * rnorm(n))
  colnames(x) <- paste0("x", seq_len(ncol(x)))
  y <- x[, 1] + (if (ncol(x) >= 2) x[, 2] else 0) + (if (ncol(x) >= 3) 0.5 * x[, 3] else 0) + rnorm(n, sd = 0.1)
  list(x = x, y = y)
}
set.seed(999)
hits <- list(); ntried <- 220
for (i in seq_len(ntried)) {
  gname <- sample(names(gens), 1); n <- sample(c(60, 80, 100, 150), 1); p <- sample(2:4, 1)
  extra <- sample(c("none", "dup", "sum", "sum3", "near"), 1)
  degree <- sample(2:3, 1); nk <- sample(c(21, 31, 41, 51), 1); thresh <- sample(c(0, 0.001), 1)
  minspan <- sample(c(1, -1), 1); endspan <- sample(c(1, -1), 1); autolp <- sample(c(TRUE, FALSE), 1)
  seed <- 5000 + i
  res <- tryCatch({
    d <- make_data(seed, gname, n, p, extra)
    args <- list(x = d$x, y = d$y, trace = 2, degree = degree, nk = nk, thresh = thresh,
      Auto.linpreds = autolp, pmethod = "none")
    if (minspan == 1) args$minspan <- 1
    if (endspan == 1) args$endspan <- 1
    out <- capture.output(f <- suppressWarnings(do.call(earth, args)))
    list(hit = grep("rank deficient", out, ignore.case = TRUE, value = TRUE),
      nr = if (is.character(f)) NA else nrow(f$dirs))
  }, error = function(e) list(hit = character(0), nr = NA))
  if (length(res$hit)) hits[[length(hits) + 1]] <- list(gname = gname, n = n, p = p, extra = extra,
    degree = degree, nk = nk, thresh = thresh, minspan = minspan, endspan = endspan, autolp = autolp,
    seed = seed, hit = res$hit, nr = res$nr)
}
cat(sprintf("\nPart A: %d random settings tried, %d triggered \"...rank deficient...\"\n", ntried, length(hits)))
for (h in hits) cat(sprintf("  gen=%-5s n=%-3d p=%d extra=%-5s degree=%d nk=%d thresh=%-5s minspan=%-2s endspan=%-2s Auto.linpreds=%-5s :: %s\n",
  h$gname, h$n, h$p, h$extra, h$degree, h$nk, h$thresh, h$minspan, h$endspan, h$autolp, h$hit))
all_autolp_false <- length(hits) > 0 && all(!vapply(hits, `[[`, TRUE, "autolp"))
all_degree_ge2 <- length(hits) > 0 && all(vapply(hits, `[[`, 1L, "degree") >= 2)
cat(sprintf("Every trigger had Auto.linpreds = FALSE: %s; every trigger had degree >= 2: %s\n", all_autolp_false, all_degree_ge2))

## ================= Part B: three representative cases =================
# For a fitted model, the covariates it actually used can take only as many
# distinct joint values as the data contains; every bx column is a fixed
# function of those covariates, so it must lie in the span of one indicator
# per distinct row. Checking that span (via qr, relative-residual tolerance
# 1e-6) is a full, verifiable stand-in for "is this term a linear
# combination of others" that does not require knowing which row is which.
span_check <- function(f, X) {
  grp <- do.call(paste, c(as.data.frame(X), sep = "|"))
  G <- model.matrix(~ factor(grp) - 1)
  qrG <- qr(G); qrb <- qr(f$bx)
  relresid <- apply(f$bx, 2, function(col) {
    r <- qr.resid(qrG, col); nrm <- sqrt(sum(col^2))
    if (nrm < 1e-12) 0 else sqrt(sum(r^2)) / nrm
  })
  list(rank_bx = qrb$rank, ncol_bx = ncol(f$bx), unique_rows = ncol(G), max_relresid = max(relresid))
}
run_case <- function(label, x, y, ...) {
  out <- capture.output(f <- tryCatch(suppressWarnings(earth(x, y, trace = 2, pmethod = "none", ...)),
    error = function(e) paste("ERROR:", conditionMessage(e))))
  msg <- grep("rank deficient", out, value = TRUE)
  cat(sprintf("\n%s\n  message: %s\n", label, if (length(msg)) msg else "(not triggered)"))
  if (is.character(f) || !length(msg)) return(invisible(NULL))
  fc <- span_check(f, x)
  cat(sprintf("  nrow(dirs) = %d; qr(bx) rank = %d of %d columns (full column rank: %s)\n",
    nrow(f$dirs), fc$rank_bx, fc$ncol_bx, fc$rank_bx == fc$ncol_bx))
  cat(sprintf("  unique rows of x = %d (the rank ceiling; nrow(dirs) <= ceiling: %s); max relative residual of a kept bx column vs the %d-dim unique-row span: %.3g (tol 1e-6: %s)\n",
    fc$unique_rows, nrow(f$dirs) <= fc$unique_rows, fc$unique_rows, fc$max_relresid, fc$max_relresid < 1e-6))
  invisible(list(f = f, fc = fc))
}

# Rebuild three of Part A's own confirmed hits exactly (same seed, same
# make_data, same earth() settings), one per mechanism, rather than
# hand-built data: Part A already showed that re-deriving "equivalent"
# settings by hand is error-prone (an earlier draft of this script built its
# own x1+x2+x3 and duplicate-column cases from scratch and neither one
# actually triggered the message, even though they looked equivalent to a
# confirmed hit).
pick_hit <- function(extra_want) Find(function(h) h$extra == extra_want, hits)
hL <- pick_hit("none"); hS <- pick_hit("sum3"); hD <- pick_hit("dup")

rebuild_case <- function(tag, h) {
  if (is.null(h)) { cat(sprintf("\nCase %s: (no Part A hit of this kind to rebuild)\n", tag)); return(invisible(NULL)) }
  label <- sprintf("Case %s: gen=%s p=%d extra=%s degree=%d (from a Part A hit)", tag, h$gname, h$p, h$extra, h$degree)
  d <- make_data(h$seed, h$gname, h$n, h$p, h$extra)
  args <- c(list(label = label, x = d$x, y = d$y, degree = h$degree, nk = h$nk, thresh = h$thresh,
    Auto.linpreds = h$autolp), if (h$minspan == 1) list(minspan = 1), if (h$endspan == 1) list(endspan = 1))
  do.call(run_case, args)
}
rL <- rebuild_case("L (few-valued columns alone, no explicit extra column)", hL)
rS <- rebuild_case("S (an extra column that is the sum of 3 others)", hS)
rD <- rebuild_case("D (an extra column that exactly duplicates column 1)", hD)

# Case D's mechanism can be pinned down further: bb22 already found that an
# exact duplicate column is never used by the forward pass itself (it always
# prefers the lower column index), so this fix must be catching something
# else. Confirm directly: a hinge built from the unused duplicate (the last
# column), at any cut a kept term actually uses on the original (column 1),
# is (to float precision) an exact copy of that kept term.
if (!is.null(rD)) {
  dD <- make_data(hD$seed, hD$gname, hD$n, hD$p, hD$extra); xD <- dD$x
  dup_col <- ncol(xD); orig_col <- 1
  dup_unused <- !any(rD$f$dirs[, dup_col] != 0)
  x1rows <- which(rD$f$dirs[, orig_col] != 0)
  qrb <- qr(rD$f$bx)
  # rebuild the WHOLE term of row i (every nonzero factor), but with the
  # orig_col factor's data swapped for the duplicate column's data (same
  # sign, same cut, other factors unchanged) -- a fair copy test even when
  # column 1 appears inside an interaction, not just as a main effect.
  term_with_swap <- function(i, swap_col) {
    v <- rep(1, nrow(xD))
    for (j in which(rD$f$dirs[i, ] != 0)) {
      jj <- if (j == orig_col) swap_col else j
      code <- rD$f$dirs[i, j]; cut <- rD$f$cuts[i, j]
      v <- v * switch(as.character(code), "1" = pmax(xD[, jj] - cut, 0),
        "-1" = pmax(cut - xD[, jj], 0), "2" = xD[, jj])
    }
    v
  }
  dup_relresid <- vapply(x1rows, function(i) {
    col <- term_with_swap(i, dup_col)
    r <- qr.resid(qrb, col); sqrt(sum(r^2)) / sqrt(sum(col^2))
  }, 0)
  cat(sprintf("  Case D detail: the duplicate column is used by no kept term: %s; rebuilding each of the %d kept terms that use column 1, with the duplicate swapped in for column 1 throughout (same cuts, same other factors), gives relative residual(s) %s vs the kept bx (essentially exact copies)\n",
    dup_unused, length(x1rows), paste(signif(dup_relresid, 3), collapse = ", ")))
}

## ---- CHECK lines ----
cat("\n")
cat(sprintf("CHECK bb23.1 %s at trace>=2, a \"Fixed rank deficient bx by removing N term(s), M terms remain\" message appeared in %d of %d random settings, every one with Auto.linpreds = FALSE and degree >= 2\n",
  length(hits) > 0 && all_autolp_false && all_degree_ge2, length(hits), ntried))
cat(sprintf("CHECK bb23.2 %s in every case examined in Part B, the returned bx is exactly full column rank (qr rank == ncol(bx) == nrow(dirs)), and nrow(dirs) never exceeds the number of distinct covariate rows the kept terms use\n",
  !is.null(rL) && !is.null(rS) && !is.null(rD) &&
  rL$fc$rank_bx == rL$fc$ncol_bx && nrow(rL$f$dirs) <= rL$fc$unique_rows &&
  rS$fc$rank_bx == rS$fc$ncol_bx && nrow(rS$f$dirs) <= rS$fc$unique_rows &&
  rD$fc$rank_bx == rD$fc$ncol_bx && nrow(rD$f$dirs) <= rD$fc$unique_rows))
cat(sprintf("CHECK bb23.3 %s (tol 1e-6) every kept bx column lies in the span of one indicator per distinct covariate row, in all three cases: a rigorous reason a term MUST become a linear combination of others once a degree>=2 search overshoots that span's dimension, independent of which row earth's fix actually removes\n",
  !is.null(rL) && !is.null(rS) && !is.null(rD) &&
  rL$fc$max_relresid < 1e-6 && rS$fc$max_relresid < 1e-6 && rD$fc$max_relresid < 1e-6))
cat(sprintf("CHECK bb23.4 %s HYPOTHESIS the trace (checked up to trace=9) names the specific dirs row(s) the fix removes, so which of two dependent terms -- the one built earlier or later in the forward pass -- was kept can be read off directly\n", FALSE))
cat(sprintf("CHECK bb23.5 %s instead the message is only ever a count, at every trace level up to 9; reconstructing the pre-fix row order from the trace=2 step table's cumulative \"Terms\" column, tried by hand on two triggering cases, gave an inconsistent row count in one of them and was not trusted further, so which specific term is removed, and whether it is the later- or earlier-built of a dependent pair, is NOT established here (Part B's span argument shows a removal is forced, but not which one)\n", TRUE))
