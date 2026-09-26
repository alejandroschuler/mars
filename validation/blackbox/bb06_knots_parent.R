# bb06: which knots does earth evaluate when the parent is not the intercept?
#
# At degree 2 the second forward step searches knots in x2 for each parent
# term from the first step. With trace = 9 earth logs each visited case, the
# parent value there (bx1) and each evaluated cut. This script predicts the
# evaluated cut list with a rule and compares. The rule, with 1-based
# positions q over ALL n cases sorted by x2, and a case active when the
# parent is nonzero there:
#   nMinSpan  = minspan, or trunc(eq. 43 with N_m active cases and p columns)
#   nEndSpan  = cap(es + floor(adj * es + 0.5)), es = endspan or trunc(eq. 45),
#               cap(e) = max(1, min(e, floor(n/2) - 1)) with n all cases
#   nStartSpan = nEndSpan + ceiling(((n - 2 nEndSpan - 1) mod nMinSpan) / 2)
#   counter <- nStartSpan
#   for q = n down to nEndSpan + 2:
#     t <- x_(q-1)                  # the value just below, active or not
#     if t >= the largest x2 among the active cases: skip (no counter move)
#     else if case q is active: counter <- counter - 1
#          if counter == 0: evaluate cut t; counter <- nMinSpan
# It also checks what "active" means for a linear parent with negative
# values, and how earth orders cases with equal x2 but different activity.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

term_values <- function(dirs, cuts, k, X) {
  v <- rep(1, nrow(X))
  for (j in which(dirs[k, ] != 0)) {
    d <- dirs[k, j]; c <- cuts[k, j]
    v <- v * switch(as.character(d), "1" = pmax(X[, j] - c, 0), "-1" = pmax(c - X[, j], 0), "2" = X[, j])
  }
  v
}

# The searches of the second forward step, one record per (parent, pred).
second_step <- function(out) {
  s <- grep("Searching for new term", out)
  if (length(s) < 2) return(list())
  lo <- s[2]; hi <- if (length(s) >= 3) s[3] else length(out)
  b <- grep("--FindKnotBegin--", out); b <- b[b > lo & b < hi]
  e <- grep("--FindKnotEnd--", out)
  par <- grep("^\\|Parent", out)
  lapply(b, function(bb) {
    pl <- max(par[par < bb]); ee <- min(e[e > bb])
    seg <- if (ee > bb + 1) out[(bb + 1):(ee - 1)] else character(0)
    ev <- grepl("RssWithKnot", seg)
    num <- function(pat) { v <- rep(NA_real_, length(seg)); v[ev == (pat == "Cut")] <- as.numeric(sub(
      if (pat == "Cut") ".*Cut +([-0-9.e]+).*" else ".*bx1 +([-0-9.e]+).*", "\\1", seg[ev == (pat == "Cut")])); v }
    list(parent = as.integer(sub("^\\|Parent +([0-9]+).*", "\\1", out[pl])),
         pred = as.integer(sub("^\\|Parent +[0-9]+ +Pred +([0-9]+).*", "\\1", out[pl])),
         ms = as.integer(sub(".*nMinSpan ([0-9]+).*", "\\1", out[bb])),
         es = as.integer(sub(".*nEndSpan ([0-9]+).*", "\\1", out[bb])),
         start = as.integer(sub(".*nStartSpan ([0-9]+).*", "\\1", out[bb])),
         case = as.integer(sub(".*Case +([0-9]+).*", "\\1", seg)),
         evaluated = ev,
         cut = num("Cut"), bx1 = num("bx1"))
  })
}

eq43 <- function(nm, p) trunc(-log2(-log(1 - 0.05) / (p * nm)) / 2.5)
eq45 <- function(p) trunc(3 - log2(0.05 / p))
capf <- function(e, n) max(1, min(e, floor(n / 2) - 1))

rule_cuts <- function(xs, act, ms, e, start) {
  n <- length(xs); counter <- start; cuts <- numeric(0)
  if (!any(act)) return(cuts)
  top <- max(xs[act])
  if (n >= e + 2) for (q in n:(e + 2)) {
    t <- xs[q - 1]
    if (t >= top) next
    if (act[q]) { counter <- counter - 1; if (counter == 0) { cuts <- c(cuts, t); counter <- ms } }
  }
  cuts
}

# Part 1: random designs with distinct x2, user and automatic spans.
set.seed(20260926)
tot <- 0; ok_ms <- 0; ok_es <- 0; ok_start <- 0; ok_cuts <- 0; ok_act <- 0; bad <- character(0)
for (i in 1:60) {
  n <- sample(c(30, 40, 60, 90), 1); p_extra <- sample(0:2, 1)
  x1 <- sample(1:n) + 0; x2 <- sample(1:n) + 0
  k1 <- sample(round(n * 0.2):round(n * 0.8), 1)
  y <- 8 * pmax(x1 - k1, 0) + 3 * pmax(k1 - x1, 0) + 0.3 * pmax(x1 - k1, 0) * sin(x2 / 4) +
       0.2 * pmax(k1 - x1, 0) * cos(x2 / 5) + 0.05 * rnorm(n)
  X <- cbind(x1, x2, matrix(runif(n * p_extra), n, p_extra)); p <- ncol(X)
  ms <- sample(c(0, 1, 2, 3, 5), 1); es <- sample(c(0, 1, 2, 3), 1); adj <- sample(c(0, 1, 1.5, 2), 1)
  out <- capture.output(f <- quiet(earth(X, y, degree = 2, trace = 9, nk = 5, minspan = ms, endspan = es,
    Adjust.endspan = adj, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
  if (is.character(f)) { bad <- c(bad, f); next }
  o <- order(x2); xs <- x2[o]
  for (r in second_step(out)) {
    if (r$pred != 2 || r$parent < 2) next
    b <- term_values(f$dirs, f$cuts, r$parent, X)[o]
    act <- b != 0; nm <- sum(act)
    e_ms <- if (ms == 0) eq43(nm, p) else ms
    es0 <- if (es == 0) eq45(p) else es
    e_es <- capf(es0 + floor(adj * es0 + 0.5), n)
    e_start <- e_es + ceiling(((n - 2 * e_es - 1) %% e_ms) / 2)
    e_cuts <- rule_cuts(xs, act, e_ms, e_es, e_start)
    # activity seen in the trace: skip lines print bx1; the visited case is 0-based
    tr_act <- all((r$bx1[!r$evaluated] != 0) == act[r$case[!r$evaluated] + 1])
    tot <- tot + 1
    ok_ms <- ok_ms + (r$ms == e_ms); ok_es <- ok_es + (r$es == e_es); ok_start <- ok_start + (r$start == e_start)
    ok_cuts <- ok_cuts + identical(r$cut[r$evaluated], e_cuts); ok_act <- ok_act + tr_act
    if (!(r$ms == e_ms && r$es == e_es && r$start == e_start && identical(r$cut[r$evaluated], e_cuts)))
      bad <- c(bad, sprintf("n=%d p=%d ms=%g es=%g adj=%g parent=%s Nm=%d: earth ms/es/start %d/%d/%d cuts %s | rule %d/%d/%d cuts %s",
        n, p, ms, es, adj, rownames(f$dirs)[r$parent], nm, r$ms, r$es, r$start, paste(r$cut[r$evaluated], collapse = ","),
        e_ms, e_es, e_start, paste(e_cuts, collapse = ",")))
  }
}
cat(sprintf("part 1: %d searches; nMinSpan %d, nEndSpan %d, nStartSpan %d, cut list %d, traced activity %d match\n",
  tot, ok_ms, ok_es, ok_start, ok_cuts, ok_act))
if (length(bad)) cat(paste(" ", head(bad, 15)), sep = "\n")

# One search in full: the parent is active only on the upper half of x2.
n <- 40; set.seed(4); x1 <- sample(1:n) + 0
x2 <- ifelse(x1 > 20, x1, 21 - x1) + 0
y <- 10 * pmax(x1 - 20, 0) + 3 * pmax(20 - x1, 0) + 0.3 * pmax(x1 - 20, 0) * sin(x2 / 3) + 0.01 * rnorm(n)
out <- capture.output(f <- quiet(earth(cbind(x1, x2), y, degree = 2, trace = 9, nk = 5, minspan = 1, endspan = 1,
  Adjust.endspan = 0, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
k <- which(f$dirs[, 1] == 1 & f$dirs[, 2] == 0)
r <- Filter(function(r) r$parent == k && r$pred == 2, second_step(out))[[1]]
cat(sprintf("example: parent %s is active for x2 in 21..40; evaluated cuts %s\n", rownames(f$dirs)[k],
  paste(r$cut[r$evaluated], collapse = ",")))
low_from_inactive <- 20 %in% r$cut[r$evaluated]
lowest_active_is_knot <- 21 %in% r$cut[r$evaluated]

# Part 2: a linear parent with negative values and a zero. linpreds = 1
# makes x1 enter linearly, which gives the same kind of term (dirs code 2)
# as an automatic linear term. earth logs no per-case lines for this parent,
# so the active count is read from the automatic minspan (eq. 43): 40 cases
# have x1 != 0 and 20 have x1 > 0.
n <- 41; set.seed(6); x1 <- sample(-20:20) + 0; x2 <- sample(1:n) + 0
y <- 4 * x1 + 0.3 * x1 * pmax(x2 - 15, 0) + 0.01 * rnorm(n)
out <- capture.output(f <- quiet(earth(cbind(x1, x2), y, degree = 2, trace = 9, nk = 6, minspan = 0, endspan = 1,
  Adjust.endspan = 0, linpreds = 1, pmethod = "none", thresh = 0, fast.k = 0)))
kl <- which(f$dirs[, 1] == 2 & f$dirs[, 2] == 0)
rl <- if (length(kl) == 1) Filter(function(r) r$parent == kl && r$pred == 2, second_step(out)) else list()
lin_ms <- if (length(rl)) rl[[1]]$ms else NA
lin_rule <- FALSE
if (length(rl)) {
  rl <- rl[[1]]; o <- order(x2); b <- x1[o]
  printed_ok <- all(rl$bx1[!rl$evaluated] == b[rl$case[!rl$evaluated] + 1])
  lin_rule <- printed_ok && any(rl$bx1[!rl$evaluated] < 0) &&
    identical(rl$cut[rl$evaluated], rule_cuts(x2[o], b > 0, rl$ms, rl$es, rl$start))
}
cat(sprintf("linear parent x1: nMinSpan %s; eq. 43 gives %d with the 40 nonzero cases and %d with the 20 positive ones\n",
  lin_ms, eq43(40, 2), eq43(20, 2)))
cat(sprintf("  the rule with active = (x1 > 0) gives earth's cut list, with negative bx1 printed at skipped cases: %s\n", lin_rule))

# Part 3: equal x2 values with different activity. earth's order within
# ties is not the input row order, and permuting the rows changes the fit.
n <- 40; set.seed(8); x1 <- sample(1:n) + 0
x2 <- rep(1:10, each = 4) + 0
y <- 10 * pmax(x1 - 20, 0) + 3 * pmax(20 - x1, 0) + 0.3 * pmax(x1 - 20, 0) * sin(x2) + 0.01 * rnorm(n)
out <- capture.output(f <- quiet(earth(cbind(x1, x2), y, degree = 2, trace = 9, nk = 5, minspan = 1, endspan = 1,
  Adjust.endspan = 0, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
k <- which(f$dirs[, 1] == 1 & f$dirs[, 2] == 0)[1]
r <- Filter(function(r) r$parent == k && r$pred == 2, second_step(out))[[1]]
act_trace <- ifelse(r$evaluated, 1L, as.integer(r$bx1 > 0))
act_stable <- as.integer(term_values(f$dirs, f$cuts, k, cbind(x1, x2))[order(x2)] > 0)[r$case + 1]
cat("activity along the scan, 0-based sorted index", max(r$case), "down to", min(r$case), "\n  earth: ",
  paste(act_trace, collapse = ""), "\n  rows in input order within ties: ", paste(act_stable, collapse = ""), "\n")
n <- 60; set.seed(8); x1 <- sample(1:n) + 0; x2 <- rep(1:12, each = 5) + 0
y <- 10 * pmax(x1 - 30, 0) + 3 * pmax(30 - x1, 0) + 0.8 * pmax(x1 - 30, 0) * pmax(x2 - 6.5, 0) + rnorm(n)
fitp <- function(i) quiet(earth(cbind(x1, x2)[i, ], y[i], degree = 2, nk = 9, minspan = 3, endspan = 1,
  Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0))
key0 <- paste(fitp(1:n)$cuts, collapse = ",")
ndiff <- sum(sapply(1:40, function(sd) { set.seed(sd); paste(fitp(sample(n))$cuts, collapse = ",") != key0 }))
cat(sprintf("row permutations of a tied design (x2 = 12 values x 5): %d of 40 change the cuts\n", ndiff))

# Part 4: the rule on tied designs, with the activity order read from the
# trace (skip lines print the parent value; evaluated lines print bx1G).
tot4 <- 0; ok4 <- 0; bad4 <- character(0)
for (i in 1:40) {
  n <- sample(c(30, 40, 60), 1); set.seed(1000 + i)
  x1 <- sample(1:n) + 0
  lev <- sample(4:12, 1); x2 <- sample(rep(1:lev, length.out = n)) + 0
  k1 <- sample(round(n * 0.3):round(n * 0.7), 1)
  y <- 8 * pmax(x1 - k1, 0) + 3 * pmax(k1 - x1, 0) + 0.5 * pmax(x1 - k1, 0) * sin(x2) + 0.05 * rnorm(n)
  ms <- sample(c(1, 2, 3), 1); es <- sample(c(1, 2), 1)
  out <- capture.output(f <- quiet(earth(cbind(x1, x2), y, degree = 2, trace = 9, nk = 5, minspan = ms, endspan = es,
    Adjust.endspan = 0, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
  if (is.character(f)) next
  xs <- sort(x2)
  for (r in second_step(out)) {
    if (r$pred != 2 || r$parent < 2 || length(r$case) == 0) next
    act <- rep(NA, n)
    act[r$case + 1] <- ifelse(r$evaluated, TRUE, r$bx1 > 0)
    if (anyNA(act[(es + 2):n])) next
    e_cuts <- rule_cuts(xs, ifelse(is.na(act), FALSE, act), r$ms, r$es, r$start)
    tot4 <- tot4 + 1; hit <- identical(r$cut[r$evaluated], e_cuts); ok4 <- ok4 + hit
    if (!hit) bad4 <- c(bad4, sprintf("n=%d lev=%d ms=%d es=%d: earth %s | rule %s", n, lev, ms, es,
      paste(r$cut[r$evaluated], collapse = ","), paste(e_cuts, collapse = ",")))
  }
}
cat(sprintf("part 4: %d searches on tied x2; the rule matches %d\n", tot4, ok4))
if (length(bad4)) cat(paste(" ", head(bad4, 10)), sep = "\n")


# Part 5: the float64 form of the adjusted endspan. For these (a, E) the two
# forms E + floor(a E + 0.5) and floor((1 + a) E + 0.5) differ in float64.
avals <- seq(0.01, 1.5, by = 0.01); pairs <- NULL
for (a in avals) for (E in 1:25) { f1 <- E + floor(a * E + 0.5); f2 <- floor((1 + a) * E + 0.5); if (f1 != f2) pairs <- rbind(pairs, c(a, E, f1, f2)) }
n <- 200; set.seed(12); x1 <- sample(1:n) + 0; x2 <- sample(1:n) + 0
y <- 8 * pmax(x1 - 100, 0) + 3 * pmax(100 - x1, 0) + 0.2 * pmax(x1 - 100, 0) * sin(x2 / 7) + 0.05 * rnorm(n)
ok5 <- TRUE
for (i in seq_len(nrow(pairs))) {
  a <- pairs[i, 1]; E <- pairs[i, 2]
  out <- capture.output(f <- quiet(earth(cbind(x1, x2), y, degree = 2, trace = 9, nk = 5, minspan = 1, endspan = E,
    Adjust.endspan = a, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
  es <- unique(sapply(Filter(function(r) r$parent >= 2, second_step(out)), `[[`, "es"))
  cat(sprintf("a = %.17g, E = %d: E + floor(aE + 0.5) = %d, floor((1 + a)E + 0.5) = %d, earth nEndSpan %s\n", a, E,
    pairs[i, 3], pairs[i, 4], paste(es, collapse = ",")))
  ok5 <- ok5 && length(es) == 1 && es == pairs[i, 3]
}
cat(sprintf("CHECK bb06.1 %s nMinSpan = minspan, or trunc(eq. 43) with the parent's N_m active cases, in all %d searches\n", ok_ms == tot, tot))
cat(sprintf("CHECK bb06.2 %s nEndSpan = cap(es + floor(adj*es + 0.5)) with the cap from all n cases, in all searches\n", ok_es == tot))
cat(sprintf("CHECK bb06.3 %s nStartSpan uses all n cases, as at the intercept, in all searches\n", ok_start == tot))
cat(sprintf("CHECK bb06.4 %s the evaluated cut list equals the rule (active cases move the counter; the loop bound counts all cases) in all searches\n", ok_cuts == tot))
cat(sprintf("CHECK bb06.5 %s the bx1 values in the trace are the parent values at the sorted cases (0-based case index)\n", ok_act == tot))
cat(sprintf("CHECK bb06.6 %s a cut can be the value of an inactive case (cut 20 when the parent is active for x2 >= 21)\n", low_from_inactive))
cat(sprintf("CHECK bb06.7 %s for a linear parent only cases with a positive parent value are active, in N_m (20, not 40) and in the scan\n",
  isTRUE(lin_ms == eq43(20, 2)) && eq43(20, 2) != eq43(40, 2) && lin_rule))
cat(sprintf("CHECK bb06.8 %s within equal x2 values earth's scan order is not the input row order\n", !identical(act_trace, act_stable)))
cat(sprintf("CHECK bb06.9 %s permuting the rows of a design with tied x2 changes earth's cuts (%d of 40 permutations)\n", ndiff > 0, ndiff))
cat(sprintf("CHECK bb06.10 %s with ties, the rule (cuts at or above the largest active x2 skipped) matches all %d searches given earth's activity order\n", ok4 == tot4 && tot4 > 0, tot4))

cat(sprintf("CHECK bb06.11 %s in the example the lowest active value, 21, is a knot although endspan is 1: the lower bound counts inactive cases\n", lowest_active_is_knot))
cat(sprintf("CHECK bb06.12 %s where the two float64 forms differ (%d settings), earth's endspan is E + floor(a E + 0.5)\n", ok5, nrow(pairs)))
