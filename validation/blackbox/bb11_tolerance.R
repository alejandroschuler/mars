# bb11: earth's collinearity tolerance for a candidate knot (the TolG flag).
#
# For a candidate knot t on parent b and variable x, earth evaluates the pair
# as {b*x, b*(x - t)+}. The hypothesis: the knot is rejected (TolG 0) when
#   r(t) = 1 - R^2 of the centered hinge b*(x - t)+ regressed on the centered
#          existing columns and b*x
# is below a tolerance: 0.01 in the first steps, 1e-5 later, where "later" is
# set by the column counter iNewCol printed in the trace.
# Part A brackets the tolerance at the first step with many close ratios.
# Part B follows the tolerance over many steps and prints, per step, iNewCol,
# the largest rejected ratio and the smallest accepted one.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

# 1 - R^2 of h on the columns of E (the intercept is implied by centering).
ratio <- function(h, E) {
  hc <- h - mean(h)
  if (is.null(E) || ncol(E) == 0) return(1)
  Ec <- sweep(E, 2, colMeans(E))
  res <- qr.resid(qr(Ec), hc)
  sum(res^2) / sum(hc^2)
}

scans <- function(out) {
  b <- grep("--FindKnotBegin--", out); e <- grep("--FindKnotEnd--", out); par <- grep("^\\|Parent", out)
  lapply(b, function(bb) {
    pl <- max(par[par < bb]); ee <- min(e[e > bb]); seg <- out[(bb + 1):(ee - 1)]
    ev <- seg[grepl("RssWithKnot", seg)]
    sl <- max(grep("Searching for new term", out)[grep("Searching for new term", out) < bb])
    K <- as.integer(sub(".*new term ([0-9]+).*", "\\1", out[sl]))
    newcol <- as.integer(sub(".*iNewCol ([0-9]+).*", "\\1", out[bb]))
    list(line = bb, parent = as.integer(sub("^\\|Parent +([0-9]+).*", "\\1", out[pl])),
      K = K, steps_done = K / 2 - 1, single = newcol == K,
      newcol = as.integer(sub(".*iNewCol ([0-9]+).*", "\\1", out[bb])),
      cut = as.numeric(sub(".*Cut +([-0-9.e]+).*", "\\1", ev)),
      tolg = as.integer(sub(".*TolG ([01]).*", "\\1", ev)))
  })
}

# Part A: one variable, 3000 distinct values, first step.
n <- 3000; x <- (1:n) / n; set.seed(1); x <- sample(x)
y <- sin(6 * x) + 0.1 * rnorm(n)
out <- capture.output(f <- quiet(earth(matrix(x, ncol = 1), y, trace = 9, nk = 3, minspan = 1, endspan = 1,
  Auto.linpreds = FALSE, pmethod = "none", thresh = 0)))
s1 <- scans(out)[[1]]
r1 <- sapply(s1$cut, function(t) ratio(pmax(x - t, 0), cbind(x)))
rej <- r1[s1$tolg == 0]; acc <- r1[s1$tolg == 1]
cat(sprintf("part A: iNewCol %d, %d cuts evaluated, %d rejected\n", s1$newcol, length(s1$cut), length(rej)))
cat(sprintf("  largest rejected ratio %.8g, smallest accepted ratio %.8g\n", max(rej), min(acc)))
sepA <- max(rej) < 0.01 && min(acc) >= 0.01
# the alternative normalizations separate worse
rc <- sapply(s1$cut, function(t) ratio(pmax(t - x, 0), cbind(x)))
cat(sprintf("  the same test on (t - x)+ instead: rejected ratios up to %.4g, accepted down to %.4g\n",
  max(rc[s1$tolg == 0]), min(rc[s1$tolg == 1])))
# Rejected knots report the RSS with the linear part only.
lin_rss <- as.numeric(sub(".*Rss +([-0-9.e]+).*", "\\1", grep("^\\|Parent 1 +Pred 1 +Case +-1", out, value = TRUE)[1]))
rows <- grep("RssWithKnot", out, value = TRUE)
rss_rej <- as.numeric(sub(".*RssWithKnot +([-0-9.e]+).*", "\\1", rows[grepl("TolG 0", rows)]))
cat(sprintf("  rejected knots print RssWithKnot equal to the linear-only RSS %.6g: %s\n", lin_rss,
  all(abs(rss_rej - lin_rss) <= 1e-4 * lin_rss)))

# Parts B and C follow the tolerance over many steps. For each knot search
# the existing columns are the first m terms of dirs (terms are numbered in
# the order they were added), where m is found by matching the traced
# RssBeforeAddingHinge (on the scale of the standardized y) with lm fits.
# A pair search adds b*x to the regressors; a single-hinge search ("no new
# form": b*x is already in the span) does not.
basis <- function(f, X) sapply(1:nrow(f$dirs), function(k) {
  v <- rep(1, nrow(X))
  for (j in which(f$dirs[k, ] != 0)) v <- v * switch(as.character(f$dirs[k, j]),
    "1" = pmax(X[, j] - f$cuts[k, j], 0), "-1" = pmax(f$cuts[k, j] - X[, j], 0), "2" = X[, j])
  v })
follow <- function(X, y, nk) {
  out <- capture.output(f <- quiet(earth(X, y, trace = 9, nk = nk, minspan = 1, endspan = 1,
    Auto.linpreds = FALSE, pmethod = "none", thresh = 0)))
  B <- basis(f, X); ys <- (y - mean(y)) / sd(y)
  rss_m <- sapply(1:ncol(B), function(m) sum(qr.resid(qr(B[, 1:m, drop = FALSE]), ys)^2))
  tab <- NULL
  for (s in scans(out)) {
    before <- as.numeric(sub(".*RssBeforeAddingHinge ([-0-9.e]+).*", "\\1", out[s$line]))
    m <- which(abs(rss_m - before) <= 1e-4 * before)[1]
    if (is.na(m) || length(s$cut) == 0) next
    j <- as.integer(sub(".*iPred ([0-9]+).*", "\\1", out[s$line]))
    E <- B[, seq_len(m), drop = FALSE][, -1, drop = FALSE]
    if (!s$single) E <- cbind(E, X[, j])
    r <- sapply(s$cut, function(t) ratio(pmax(X[, j] - t, 0), E))
    tab <- rbind(tab, data.frame(iNewCol = s$newcol, steps_done = s$steps_done, single = s$single, terms_before = m, cuts = length(r),
      rejected = sum(s$tolg == 0),
      max_rej = if (any(s$tolg == 0)) max(r[s$tolg == 0]) else NA,
      min_acc = if (any(s$tolg == 1)) min(r[s$tolg == 1]) else NA))
  }
  tab
}
agg <- function(tab) do.call(rbind, lapply(split(tab, tab$iNewCol), function(d) data.frame(iNewCol = d$iNewCol[1],
  steps_done = d$steps_done[1], terms_before = paste(unique(d$terms_before), collapse = "/"),
  searches = nrow(d), single = sum(d$single), rejected = sum(d$rejected),
  max_rej = suppressWarnings(max(d$max_rej, na.rm = TRUE)), min_acc = suppressWarnings(min(d$min_acc, na.rm = TRUE)))))

n <- 600; x <- (1:n) / n; set.seed(2); x <- sample(x)
y <- sin(12 * x) + 0.5 * sin(40 * x) + 0.05 * rnorm(n)
tabB <- follow(matrix(x, ncol = 1), y, 41)
cat("part B: one variable, single hinges after the first pair\n"); print(agg(tabB), digits = 4, row.names = FALSE)

n <- 400; set.seed(3); X <- matrix(runif(n * 9), n, 9); colnames(X) <- paste0("x", 1:9)
y <- rowSums(sapply(1:9, function(j) (10 - j) * pmax(X[, j] - 0.3 - 0.04 * j, 0))) + 0.05 * rnorm(n)
tabC <- follow(X, y, 25)
cat("part C: nine variables, pairs on new variables\n"); print(agg(tabC), digits = 4, row.names = FALSE)

# Trace parent numbers count slots: every step takes two, and a one-term step
# leaves its second slot empty. The step lines (Terms counts dirs rows) give the
# map from slots to dirs rows.
steps_of <- function(out) {
  i <- grep("^[0-9]+ +[-0-9.]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ +x[0-9]+ ", out)
  lapply(i, function(ii) {
    tk <- strsplit(trimws(sub(" final.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    list(K = as.integer(tk[1]), terms = if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)])
  })
}
slot_map <- function(st) {
  m <- c(`1` = 1L); row <- 1L
  for (s in st) for (q in seq_along(s$terms)) { row <- row + 1L; m[as.character(s$K + q - 1)] <- row }
  m
}
# Part D: a hinge parent at degree 2. For each search with a parent other
# than the intercept, the ratio uses h = b*(x - t)+ and E = the existing
# columns plus b*x for a pair search.
follow2 <- function(X, y, nk) {
  out <- capture.output(f <- quiet(earth(X, y, degree = 2, trace = 9, nk = nk, minspan = 1, endspan = 1,
    Adjust.endspan = 0, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, fast.k = 0)))
  B <- basis(f, X); ys <- (y - mean(y)) / sd(y); smap <- slot_map(steps_of(out))
  rss_m <- sapply(1:ncol(B), function(m) sum(qr.resid(qr(B[, 1:m, drop = FALSE]), ys)^2))
  tab <- NULL
  for (s in scans(out)) {
    if (s$parent < 2 || length(s$cut) == 0) next
    s$parent <- unname(smap[as.character(s$parent)]); if (is.na(s$parent)) next
    before <- as.numeric(sub(".*RssBeforeAddingHinge ([-0-9.e]+).*", "\\1", out[s$line]))
    m <- which(abs(rss_m - before) <= 1e-4 * before)[1]
    if (is.na(m) || s$parent > m) next
    j <- as.integer(sub(".*iPred ([0-9]+).*", "\\1", out[s$line]))
    b <- B[, s$parent]
    E <- B[, seq_len(m), drop = FALSE][, -1, drop = FALSE]
    if (!s$single) E <- cbind(E, b * X[, j])
    r <- sapply(s$cut, function(t) ratio(b * pmax(X[, j] - t, 0), E))
    tab <- rbind(tab, data.frame(iNewCol = s$newcol, steps_done = s$steps_done, single = s$single, parent = s$parent,
      cuts = length(r), rejected = sum(s$tolg == 0),
      max_rej = if (any(s$tolg == 0)) max(r[s$tolg == 0]) else NA,
      min_acc = if (any(s$tolg == 1)) min(r[s$tolg == 1]) else NA))
  }
  tab
}
n <- 300; set.seed(7); X <- matrix(runif(n * 3), n, 3); colnames(X) <- paste0("x", 1:3)
y <- 6 * pmax(X[, 1] - 0.3, 0) * X[, 2] + 3 * pmax(0.6 - X[, 1], 0) * pmax(X[, 3] - 0.2, 0) + 2 * X[, 2] + 0.05 * rnorm(n)
tabD <- follow2(X, y, 21)
cat("part D: degree 2, searches whose parent is not the intercept\n")
print(tabD[, c("iNewCol", "steps_done", "single", "parent", "cuts", "rejected", "max_rej", "min_acc")], digits = 4, row.names = FALSE)
okD <- nrow(tabD) > 0 && sum(tabD$rejected) > 0 &&
  all(ifelse(tabD$iNewCol <= 15, tabD$max_rej < 0.01, tabD$max_rej < 1e-5), na.rm = TRUE) &&
  all(ifelse(tabD$iNewCol <= 15, tabD$min_acc >= 0.01, tabD$min_acc >= 1e-5), na.rm = TRUE)

tab <- rbind(tabB, tabC)
early <- tab[tab$iNewCol <= 15, ]; late <- tab[tab$iNewCol >= 16, ]
by_steps <- all(tab$steps_done[tab$iNewCol <= 15] <= 6) && all(tab$steps_done[tab$iNewCol >= 16] >= 7)
cat(sprintf("CHECK bb11.1 %s at the first step the knot is rejected exactly when 1 - R^2 of (x - t)+ on {1, x} is below 0.01 (bracket %.6g, %.6g)\n",
  sepA, max(rej), min(acc)))
cat(sprintf("CHECK bb11.2 %s a rejected knot keeps the RSS of the linear part alone (its RSS reduction is that of b*x)\n",
  all(abs(rss_rej - lin_rss) <= 1e-4 * lin_rss)))
cat(sprintf("CHECK bb11.3 %s while iNewCol <= 15, every rejected ratio is below 0.01 and every accepted one at least 0.01\n",
  all(early$max_rej < 0.01, na.rm = TRUE) && all(early$min_acc >= 0.01, na.rm = TRUE)))
cat(sprintf("CHECK bb11.4 %s once iNewCol >= 16, every rejected ratio is below 1e-5 and every accepted one at least 1e-5\n",
  nrow(late) > 0 && all(late$max_rej < 1e-5, na.rm = TRUE) && all(late$min_acc >= 1e-5, na.rm = TRUE)))
cat(sprintf("CHECK bb11.5 %s iNewCol is K for a single-hinge search and K + 1 for a pair search, K the term number searched (2, 4, 6, ...)\n",
  all((tab$iNewCol - 2 * (tab$steps_done + 1)) %in% c(0, 1))))
cat(sprintf("CHECK bb11.6 %s the tolerance is 0.01 in the first 7 forward steps and 1e-5 from the 8th, for pairs and single hinges alike (steps_done <= 6 vs >= 7)\n",
  by_steps))
cat(sprintf("CHECK bb11.7 %s for a hinge parent at degree 2 the same test holds with h = b*(x - t)+ and E = existing columns plus b*x (%d searches, %d rejected knots)\n",
  okD, nrow(tabD), sum(tabD$rejected)))
