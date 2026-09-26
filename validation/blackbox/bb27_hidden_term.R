# bb27: a hidden term in earth's forward pass with Auto.linpreds = FALSE.
#
# When the linear option wins a step and Auto.linpreds = FALSE, earth adds the
# one hinge b*(x - min)+ (bb13.5). In some fits the Fast MARS queue (trace 6)
# then grows by two entries although the step table lists one term; the
# final "terms used" count is one more than nrow(dirs); and the end-of-pass
# rank fix ("Fixed rank deficient bx by removing 1 term") removes one term.
# With Auto.linpreds = TRUE none of this happens. Part B counts how often the
# fix appears over random fits with each setting.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

queue_sizes <- function(out) {   # number of rows in each queue table
  h <- grep("^SortedQ", out)
  sapply(h, function(i) { j <- i + 1; k <- 0
    while (j <= length(out) && (grepl("^ +[0-9]+ +[0-9]+ +[0-9]+ +[-0-9.]+ +", out[j]) || grepl("^FastK", out[j]))) {
      if (!grepl("^FastK", out[j])) k <- k + 1; j <- j + 1 }
    k })
}
# one record per step: the listed terms, whether it is the linear option, its parent slot, and the queue growth
step_records <- function(out) {
  i <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out)
  q <- queue_sizes(out); h <- grep("^SortedQ", out)
  recs <- lapply(seq_along(i), function(k) {
    ii <- i[k]; tk <- strsplit(trimws(sub(" final.*| reject.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    nt <- length(if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)])
    after <- q[which(h > ii)[1]]; before <- if (k == 1) 1 else q[which(h > i[k - 1])[1]]
    data.frame(K = as.integer(tk[1]), linear = grepl("<$", tk[7]), parent_intercept = deg == 1, listed = nt,
      growth = if (is.na(after)) NA else after - before)
  })
  do.call(rbind, recs)
}
steps_terms <- function(out) {   # number of terms that each step lists in the step table
  i <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out)
  sapply(i, function(ii) { tk <- strsplit(trimws(sub(" final.*| reject.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    length(if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)]) })
}
fit_trace <- function(X, y, alp, ...) {
  out <- capture.output(f <- quiet(earth(X, y, degree = 2, trace = 6, pmethod = "none", Auto.linpreds = alp, ...)))
  used <- as.integer(sub(".*, ([0-9]+) terms used.*", "\\1", grep("terms used", out, value = TRUE)[1]))
  list(f = f, queue = queue_sizes(out), steps = steps_terms(out), fix = grep("Fixed rank deficient", out, value = TRUE),
    used = used, linear_step = any(grepl("<", sub(".*x[0-9] +", "", grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out, value = TRUE)))))
}

# Part A: one fit, both settings
set.seed(2024); n <- 200; b0 <- sample(c(-1, 1), n, TRUE) + 0; x1 <- runif(n, -1.7, 1.7); b2 <- sample(c(-1, 1), n, TRUE) + 0
y <- 6 * pmax(0.8 - x1, 0) + 3 * (b0 > 0) + (b0 > 0) * pmax(0.8 - x1, 0) + 0.01 * rnorm(n)
X <- cbind(x0 = b0, x1 = x1, x2 = b2)
rF <- fit_trace(X, y, FALSE, nk = 11, thresh = 0, fast.k = 0)
rT <- fit_trace(X, y, TRUE, nk = 11, thresh = 0, fast.k = 0)
for (nm in c("FALSE", "TRUE")) {
  r <- if (nm == "FALSE") rF else rT
  cat(sprintf("Auto.linpreds %-5s: terms per step %s | queue sizes %s | terms used %s | nrow(dirs) %d | %s\n", nm,
    paste(r$steps, collapse = " "), paste(r$queue, collapse = " "), r$used, nrow(r$f$dirs), if (length(r$fix)) trimws(r$fix) else "no rank fix"))
}
visible_F <- 1 + cumsum(rF$steps); visible_T <- 1 + cumsum(rT$steps)
extra_entry <- any(rF$queue[seq_along(visible_F)] > visible_F[seq_along(rF$queue)])
match_T <- all(rT$queue[seq_along(visible_T)] == visible_T[seq_along(rT$queue)])

# Part B: over random fits with Auto.linpreds = FALSE, which linear-option
# steps add a hidden queue entry (queue growth above the listed terms)?
set.seed(77); allrec <- NULL; cnt <- c(F_fits = 0, F_fix = 0, T_fits = 0, T_fix = 0)
for (i in 1:40) {
  n <- 150; Xr <- cbind(x0 = sample(c(-1, 1), n, TRUE) + 0, x1 = runif(n, -1, 1), x2 = sample(0:2, n, TRUE) + 0, x3 = runif(n))
  yr <- 2 * pmax(0.3 - Xr[, 2], 0) + (Xr[, 1] > 0) * (1 + Xr[, 3]) + sin(3 * Xr[, 4]) + 0.05 * rnorm(n)
  for (alp in c(FALSE, TRUE)) for (dg in c(2, 3)) {
    out <- capture.output(f <- quiet(earth(Xr, yr, degree = dg, trace = 6, pmethod = "none", Auto.linpreds = alp, nk = 15, thresh = 0, fast.k = 0)))
    rec <- step_records(out); rec$alp <- alp; rec$degree <- dg
    fix <- any(grepl("Fixed rank deficient", out))
    if (any(rec$linear)) { key <- if (alp) "T" else "F"; cnt[paste0(key, "_fits")] <- cnt[paste0(key, "_fits")] + 1
      cnt[paste0(key, "_fix")] <- cnt[paste0(key, "_fix")] + fix }
    allrec <- rbind(allrec, rec)
  }
}
lin <- allrec[allrec$linear & !is.na(allrec$growth), ]
lin$hidden <- lin$growth > lin$listed
print(with(lin, table(auto_linpreds = alp, parent_is_intercept = parent_intercept, hidden_entry = hidden)))
nonlin <- allrec[!allrec$linear & !is.na(allrec$growth), ]
cat(sprintf("part B: fits with a linear-option step: Auto.linpreds FALSE %d, with the rank fix %d; TRUE %d, with the rank fix %d; steps that are not the linear option with a hidden entry: %d of %d\n",
  cnt["F_fits"], cnt["F_fix"], cnt["T_fits"], cnt["T_fix"], sum(nonlin$growth > nonlin$listed), nrow(nonlin)))
rule_hidden <- with(lin, all(hidden == (!alp & !parent_intercept)))

# Part C: does the forward pass count the hidden term in the GRSq that it
# prints (and uses in its stopping rules)? Compare each step row's GRSq with
# GCV at the visible number of terms and at that number plus the hidden ones.
set.seed(31); n <- 120; z0 <- sample(c(0, 1), n, TRUE) + 0; z1 <- runif(n); z2 <- runif(n)
yz <- 2 * pmax(z1 - 0.4, 0) + 1.5 * z0 * pmax(0.4 - z1, 0) + 0.8 * z2 + 0.4 * rnorm(n)
out <- capture.output(f <- quiet(earth(cbind(x0 = z0, x1 = z1, x2 = z2), yz, degree = 2, trace = 2, pmethod = "none",
  Auto.linpreds = FALSE, nk = 15, thresh = 0, fast.k = 0, penalty = 3)))
rows <- grep("^[0-9]+ +[-0-9.inf]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ ", out, value = TRUE); tss <- sum((yz - mean(yz))^2)
M <- 1; hid <- 0; vis_ok <- TRUE; hid_off <- FALSE
for (r in rows) {
  tk <- strsplit(trimws(sub(" final.*| reject.*", "", r)), " +")[[1]]
  rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
  M <- M + length(if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)])
  if (grepl("<$", tk[7]) && deg >= 2) hid <- hid + 1
  rss <- (1 - as.numeric(tk[3])) * tss
  g <- function(m) 1 - earth:::get.gcv(rss, m, 3, n) / earth:::get.gcv(tss, 1, 3, n)
  vis_ok <- vis_ok && abs(as.numeric(tk[2]) - g(M)) < 2e-4
  if (hid > 0) hid_off <- hid_off || abs(as.numeric(tk[2]) - g(M + hid)) > 1e-2
}
cat(sprintf("part C: %d steps, %d with a hidden term; the printed GRSq matches the visible term count: %s; the count with hidden terms is off: %s\n",
  length(rows), hid, vis_ok, hid_off))

cat(sprintf("CHECK bb27.1 %s with Auto.linpreds = FALSE the queue grows by two entries after a step that lists one term, the fit uses one more term than dirs shows (%s against %d), and the rank fix removes it\n",
  extra_entry && isTRUE(rF$used == nrow(rF$f$dirs) + 1) && length(rF$fix) > 0, rF$used, nrow(rF$f$dirs)))
cat(sprintf("CHECK bb27.2 %s with Auto.linpreds = TRUE on the same data the queue sizes equal the visible term counts and no rank fix is printed\n",
  match_T && length(rT$fix) == 0))
cat(sprintf("CHECK bb27.3 %s over random fits at degree 2 and 3, a linear-option step adds a hidden queue entry exactly when Auto.linpreds = FALSE and the parent is not the intercept (%d steps), and no other step does\n",
  rule_hidden && sum(nonlin$growth > nonlin$listed) == 0, nrow(lin)))
cat(sprintf("CHECK bb27.4 %s the rank fix appears only with Auto.linpreds = FALSE (%d of %d such fits, and %d of %d with TRUE)\n",
  cnt["F_fix"] > 0 && cnt["T_fix"] == 0, cnt["F_fix"], cnt["F_fits"], cnt["T_fix"], cnt["T_fits"]))
cat(sprintf("CHECK bb27.5 %s the forward pass computes GRSq with the visible terms only; the hidden term does not count\n", vis_ok && hid_off && hid > 0))
