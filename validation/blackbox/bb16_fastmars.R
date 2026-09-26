# bb16: earth's Fast MARS priority queue (fast.k, fast.beta).
#
# At trace = 6 earth prints, after each forward step, its queue of parent
# terms: sorted position, entry number, nTermsForRssDelta (K_m), AgedRank and
# RssDelta. At trace = 7 it prints, for each step, every parent it visits:
# "|Parent P  Pred J ..." when it searches P, and "|Parent P  skip (degree of
# term would be D)" when it skips P. Trace = 9 adds every evaluated knot with
# its RSS reduction. The step that searches for term K (K = 2, 4, 6, ...) fills
# earth's internal slots K and K + 1, or only slot K when it adds one term (a
# single hinge or a linear term). T is the number of terms so far.
#
# The rules that this script compares with every printed table of 99 fits:
#   Q1 the queue entries are 1..T. A step that adds a terms appends entries
#      T+1..T+a with RssDelta = inf and K_m = K.
#   Q2 rank = 0-based place after sorting by RssDelta, largest first, equal
#      RssDelta by entry number; AgedRank = rank + fast.beta * (K - K_m), K of
#      the step just done (fast.beta = 0 prints -1.0). The table is sorted by
#      AgedRank, equal AgedRank by rank.
#   Q3 the next step visits the first k rows in table order, k = max(3,
#      trunc(fast.k)), all rows when trunc(fast.k) = 0. Entry c stands for
#      SLOT c. It is skipped when slot c is empty or holds a term whose degree
#      equals the degree argument, and searched otherwise.
#   Q4 a searched entry gets K_m = K and RssDelta = its best RSS reduction in
#      the step (0 when none is positive, -1 when no predictor is allowed).
#      Skipped and unvisited entries keep K_m and RssDelta.
#   Q5 a step that searches no entry adds no term and ends the forward pass.
# The script also tests refuted alternatives (HYPOTHESIS lines).
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

# Every earth call is wrapped so that a failure prints only its message and
# warnings keep only their message.
warn_msgs <- character(0)
quiet <- function(expr) withCallingHandlers(
  tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e))),
  warning = function(w) { warn_msgs <<- c(warn_msgs, conditionMessage(w)); invokeRestart("muffleWarning") })

# The forward-pass part of a trace.
fp <- function(out) out[grep("^Forward pass:", out)[1]:grep("^Forward pass complete", out)[1]]

# Step lines: K, number of terms added (a), parent slot, degree, rejected.
parse_steps <- function(reg) {
  do.call(rbind, lapply(grep("^[0-9]+ ", reg, value = TRUE), function(s) {
    tk <- strsplit(trimws(s), " +")[[1]]; K <- as.integer(tk[1])
    if (K == 1) return(data.frame(K = 1L, a = 1L, par = NA, deg = 0L, reject = FALSE))
    if (tk[5] == "-") return(data.frame(K = K, a = 0L, par = NA, deg = NA, reject = TRUE))
    ints <- integer(0)
    for (t in tk[-(1:7)]) if (grepl("^[0-9]+$", t)) ints <- c(ints, as.integer(t)) else break
    deg <- ints[length(ints)]; oth <- ints[-length(ints)]
    # A degree-1 term has the intercept as parent, so its Par column is blank.
    if (deg == 1) data.frame(K = K, a = length(oth), par = 1L, deg = deg, reject = FALSE)
    else data.frame(K = K, a = length(oth) - 1L, par = oth[length(oth)], deg = deg, reject = FALSE)
  }))
}

# Queue tables, each with the K of the step printed before it.
parse_tables <- function(reg) {
  lapply(grep("^SortedQ", reg), function(b) {
    j <- b + 1; rows <- NULL; fkline <- NA
    while (j <= length(reg) && (grepl("^ +[0-9]+ +[0-9]+ +[0-9]+ +[-0-9.]+ +[-+0-9.a-z]+ *$", reg[j]) || grepl("^FastK", reg[j]))) {
      if (grepl("^FastK", reg[j])) fkline <- as.integer(sub("^FastK ([0-9]+).*", "\\1", reg[j])) else {
        v <- strsplit(trimws(reg[j]), " +")[[1]]
        rows <- rbind(rows, data.frame(entry = as.integer(v[2]), Km = as.integer(v[3]), aged = as.numeric(v[4]), rss = as.numeric(v[5])))
      }
      j <- j + 1
    }
    stp <- max(grep("^[0-9]+ ", reg[1:b]))
    list(K = as.integer(sub("^([0-9]+) .*", "\\1", reg[stp])), tab = rows, fkline = fkline)
  })
}

# "|Considering parents P [v] ..." lines, keyed by the K of the step that follows.
parse_considering <- function(reg) {
  lapply(grep("^\\|Considering parents", reg), function(i) {
    s <- sub("^\\|Considering parents +", "", reg[i])
    m <- regmatches(s, gregexpr("[0-9]+ \\[[^]]*\\]", s))[[1]]
    nxt <- grep("^[0-9]+ ", reg); nxt <- nxt[nxt > i][1]
    list(K = as.integer(sub("^([0-9]+) .*", "\\1", reg[nxt])),
      entry = as.integer(sub(" .*", "", m)), val = as.numeric(sub(".*\\[(.*)\\]", "\\1", m)))
  })
}

# Visits per step from trace 7: entries in visiting order, and the printed
# degree D for entries skipped by degree (NA when searched).
parse_visits <- function(reg) {
  st <- grep("Searching for new term", reg); res <- list()
  for (i in seq_along(st)) {
    seg <- reg[st[i]:(if (i < length(st)) st[i + 1] - 1 else length(reg))]
    K <- as.integer(sub(".*new term ([0-9]+).*", "\\1", seg[1]))
    pl <- grep("^\\|Parent", seg, value = TRUE)
    par <- as.integer(sub("^\\|Parent +([0-9]+).*", "\\1", pl))
    u <- unique(par)
    D <- sapply(u, function(q) { s <- pl[par == q & grepl("skip \\(degree of term would be", pl)]
      if (length(s)) as.integer(sub(".*would be ([0-9]+).*", "\\1", s[1])) else NA_integer_ })
    res[[as.character(K)]] <- data.frame(entry = u, D = D)
  }
  res
}

# Q2: 0-based ranks and the sorted order.
rank_of <- function(entry, rss) { o <- order(-rss, entry); r <- integer(length(o)); r[o] <- seq_along(o) - 1L; r }

# Slot -> dirs row, from the number of terms that each step added.
slot_rows <- function(steps, nk) {
  sr <- rep(NA_integer_, nk + 2); sr[1] <- 1L; r <- 1L
  for (i in which(steps$K >= 2 & !steps$reject)) for (j in seq_len(steps$a[i]) - 1L) { r <- r + 1L; sr[steps$K[i] + j] <- r }
  sr
}

# One fit at trace 6 and trace 7, checked against Q1-Q5.
check_fit <- function(X, y, degree, nk, fk, fb, lp = FALSE, label = "") {
  o6 <- capture.output(f6 <- quiet(earth(X, y, degree = degree, nk = nk, fast.k = fk, fast.beta = fb, linpreds = lp, trace = 6, pmethod = "none", thresh = 0)))
  o7 <- capture.output(f7 <- quiet(earth(X, y, degree = degree, nk = nk, fast.k = fk, fast.beta = fb, linpreds = lp, trace = 7, pmethod = "none", thresh = 0)))
  r6 <- fp(o6); r7 <- fp(o7)
  steps <- parse_steps(r7); tabs <- parse_tables(r6); vis <- parse_visits(r7); cons <- parse_considering(r6)
  dg <- unname(rowSums(f7$dirs != 0)); sr <- slot_rows(steps, nk)
  slot_deg <- function(c) ifelse(is.na(sr[c]), 101L, dg[sr[c]])
  kk <- if (fk == 0) Inf else max(3, trunc(fk))
  s <- list(label = label, fk = fk, fb = fb, degree = degree, same = identical(f6$dirs, f7$dirs) && identical(f6$cuts, f7$cuts),
    terms_ok = nrow(f7$dirs) == sum(steps$a), ntab = length(tabs), rows = 0, aged = 0, order = 0,
    entries_ok = 0, fkline_ok = 0, nvis = 0, vis_ok = 0, skip_ok = 0, skip_row_ok = 0, skipD_ok = 0, nskipD = 0, unusedD = integer(0),
    ntrans = 0, trans_ok = 0, chosen_new = 0, nchosen = 0, cons_ok = 0, cons_val_ok = 0, ncons = 0,
    empty = 0, empty_reject = 0, reject_nonempty = 0, par_ok = 0, par_row_ok = 0, npar = 0,
    tieA = 0, tieA_info = 0, altA_ok = 0, tieR = 0, tieR_fin = 0, tieR_info = 0, altR_prev_ok = 0, altR_desc_ok = 0,
    fb0_rows = 0, fb0_neg1 = 0, ntab_aged = 0, alt_steps_ok = 0,
    int_missed = 0, eligible_unvisited = 0, cons_past = numeric(0), neg = integer(0), rss_vals = numeric(0), bad = character(0))
  Tn <- cumsum(steps$a); names(Tn) <- steps$K
  prev <- NULL
  for (i in seq_along(tabs)) {
    tb <- tabs[[i]]; t <- tb$tab; K <- tb$K; n <- nrow(t)
    # A rank vector r reproduces the table when the rows are in order(AgedRank,
    # then tie) and the printed AgedRank is r + fast.beta * (K - K_m); with
    # fast.beta = 0 earth prints -1.0 for every row.
    pa <- function(r, age = K - t$Km) if (fb == 0) rep(-1, n) else r + fb * age
    holds <- function(r, tie = r, age = K - t$Km) identical(t$entry, t$entry[order(r + fb * age, tie)]) && all(abs(t$aged - pa(r, age)) < 1e-9)
    rk <- rank_of(t$entry, t$rss); ex <- rk + fb * (K - t$Km)
    s$rows <- s$rows + n; s$aged <- s$aged + sum(abs(t$aged - pa(rk)) < 1e-9)
    ok_order <- identical(t$entry, t$entry[order(ex, rk)]); s$order <- s$order + ok_order
    if (!holds(rk)) s$bad <- c(s$bad, sprintf("%s K=%d table order/AgedRank", label, K))
    if (fb == 0) { s$fb0_rows <- s$fb0_rows + n; s$fb0_neg1 <- s$fb0_neg1 + sum(t$aged == -1) }
    if (fb > 0 && any(t$Km < K)) { s$ntab_aged <- s$ntab_aged + 1; s$alt_steps_ok <- s$alt_steps_ok + holds(rk, age = (K - t$Km) / 2) }
    s$entries_ok <- s$entries_ok + identical(sort(t$entry), seq_len(Tn[as.character(K)]))
    s$fkline_ok <- s$fkline_ok + (if (is.finite(kk) && n >= kk) isTRUE(tb$fkline == kk) else is.na(tb$fkline))
    s$neg <- c(s$neg, t$rss[t$rss < 0]); s$rss_vals <- c(s$rss_vals, t$rss)
    # ties in AgedRank (informative when entry order differs from rank order)
    for (g in unique(ex[duplicated(ex)])) { w <- which(ex == g); s$tieA <- s$tieA + 1
      s$tieA_info <- s$tieA_info + !identical(order(rk[w]), order(t$entry[w])) }
    s$altA_ok <- s$altA_ok + holds(rk, tie = t$entry)
    # ties in RssDelta; alternatives: by previous table position, by entry descending
    for (g in unique(t$rss[duplicated(t$rss)])) { w <- which(t$rss == g); s$tieR <- s$tieR + 1; s$tieR_fin <- s$tieR_fin + is.finite(g)
      if (!is.null(prev)) { pp <- match(t$entry[w], prev$entry); pp[is.na(pp)] <- 1000 + t$entry[w][is.na(pp)]
        s$tieR_info <- s$tieR_info + !identical(order(pp), order(t$entry[w])) } }
    if (!is.null(prev)) { pp <- match(t$entry, prev$entry); pp[is.na(pp)] <- 1000 + t$entry[is.na(pp)]
      o <- order(-t$rss, pp); r2 <- integer(n); r2[o] <- seq_len(n) - 1L; s$altR_prev_ok <- s$altR_prev_ok + holds(r2) }
    o <- order(-t$rss, -t$entry); r3 <- integer(n); r3[o] <- seq_len(n) - 1L
    s$altR_desc_ok <- s$altR_desc_ok + holds(r3)
    # Q3: the next step's visits
    Kn <- K + 2; v <- vis[[as.character(Kn)]]
    if (!is.null(v)) {
      exp_vis <- head(t$entry, if (is.finite(kk)) kk else n)
      s$nvis <- s$nvis + 1; okv <- identical(v$entry, exp_vis); s$vis_ok <- s$vis_ok + okv
      if (!okv) s$bad <- c(s$bad, sprintf("%s K=%d visits %s expected %s", label, Kn, paste(v$entry, collapse = ","), paste(exp_vis, collapse = ",")))
      sk <- !is.na(v$D)
      s$skip_ok <- s$skip_ok + identical(sk, slot_deg(v$entry) >= degree)
      s$skip_row_ok <- s$skip_row_ok + identical(sk, dg[v$entry] >= degree)
      s$nskipD <- s$nskipD + sum(sk); s$skipD_ok <- s$skipD_ok + sum(v$D[sk] == slot_deg(v$entry[sk]) + 1)
      s$unusedD <- c(s$unusedD, v$D[sk & is.na(sr[v$entry])])
      s$int_missed <- s$int_missed + !(1 %in% v$entry)
      # eligible terms whose slot has no entry (slot > T): never visited this step
      used <- which(!is.na(sr) & seq_along(sr) <= K + 1); s$eligible_unvisited <- s$eligible_unvisited + sum(used > n & dg[sr[used]] < degree)
      searched <- v$entry[!sk]
      st <- steps[steps$K == Kn, ]
      if (length(searched) == 0) { s$empty <- s$empty + 1; s$empty_reject <- s$empty_reject + isTRUE(st$reject) }
      if (isTRUE(st$reject) && length(searched) > 0) s$reject_nonempty <- s$reject_nonempty + 1
      # Q1, Q4: the table after step Kn
      nt <- Filter(function(z) z$K == Kn, tabs)
      if (length(nt)) {
        u <- nt[[1]]$tab; m <- match(t$entry, u$entry)
        keep <- !(t$entry %in% searched)
        ok <- all(u$Km[m[!keep]] == Kn) && all(u$Km[m[keep]] == t$Km[keep]) && all(u$rss[m[keep]] == t$rss[keep])
        new <- setdiff(u$entry, t$entry)
        ok <- ok && identical(sort(new), n + seq_len(st$a)) && all(is.infinite(u$rss[u$entry %in% new])) && all(u$Km[u$entry %in% new] == Kn)
        s$ntrans <- s$ntrans + 1; s$trans_ok <- s$trans_ok + ok
        if (!ok) s$bad <- c(s$bad, sprintf("%s K=%d transition", label, Kn))
        if (!is.na(st$par)) { s$nchosen <- s$nchosen + 1
          s$chosen_new <- s$chosen_new + (u$Km[u$entry == st$par] == Kn && u$rss[u$entry == st$par] == max(u$rss[u$entry %in% searched])) }
      }
    }
    prev <- t
  }
  # the "Considering parents" line before step K lists the visited entries
  for (cn in cons) { v <- vis[[as.character(cn$K)]]; if (cn$K < 4 || is.null(v)) next
    t <- Filter(function(z) z$K == cn$K - 2, tabs)[[1]]$tab; s$ncons <- s$ncons + 1
    s$cons_ok <- s$cons_ok + identical(cn$entry, v$entry)
    # each bracketed value is the RssDelta of the next table row; a bracket
    # past the end of the table holds a value that varies from run to run
    j <- seq_along(cn$entry) + 1; inside <- j <= nrow(t); nxt <- t$rss[j[inside]]; cv <- cn$val[inside]
    s$cons_val_ok <- s$cons_val_ok + isTRUE(all(ifelse(is.infinite(nxt), is.infinite(cv), abs(cv - nxt) <= 1e-5 * pmax(1, abs(nxt)))))
    s$cons_past <- c(s$cons_past, cn$val[!inside]) }
  # the step line's Par is the parent's slot: the new term = parent term x one new factor
  for (i in which(steps$K >= 2 & !steps$reject & steps$deg >= 2)) {
    r <- sum(steps$a[1:i]); pr_slot <- sr[steps$par[i]]; pr_row <- steps$par[i]
    is_parent <- function(p) { if (is.na(p) || p < 1 || p >= r) return(FALSE)
      d1 <- f7$dirs[r, ]; d0 <- f7$dirs[p, ]; nw <- which(d1 != 0 & d0 == 0)
      length(nw) == 1 && all(d1[-nw] == d0[-nw]) && all(f7$cuts[r, -nw][d0[-nw] != 0] == f7$cuts[p, -nw][d0[-nw] != 0]) }
    s$npar <- s$npar + 1; s$par_ok <- s$par_ok + is_parent(pr_slot); s$par_row_ok <- s$par_row_ok + is_parent(pr_row)
  }
  s$cuts <- f7$cuts; s$dirs <- f7$dirs; s$steps <- steps; s$tabs <- tabs; s$vis <- vis; s$sr <- sr; s$dg <- dg; s$nterms <- nrow(f7$dirs)
  s
}

# Data sets: four with 5 to 8 covariates, and one with a single covariate so
# that every hinge parent has no allowed predictor (exact ties at -1).
mk <- function(id) {
  if (id == "D1") { set.seed(101); n <- 200; X <- matrix(runif(n * 7), n, 7)
    y <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n) }
  if (id == "D2") { set.seed(102); n <- 400; X <- matrix(runif(n * 8), n, 8)
    y <- 5 * X[, 1] * X[, 2] * X[, 3] + 3 * sin(6 * X[, 4]) + 2 * pmax(X[, 5] - 0.4, 0) * X[, 1] +
      4 * pmax(X[, 6] - 0.5, 0) * pmax(0.7 - X[, 7], 0) + 0.1 * rnorm(n) }
  if (id == "D3") { set.seed(103); n <- 100; X <- matrix(runif(n * 6), n, 6)
    y <- 6 * X[, 1] + 3 * sin(8 * X[, 2]) + 4 * X[, 3] * X[, 4] + 0.2 * rnorm(n) }
  if (id == "D4") { set.seed(104); n <- 300; X <- matrix(runif(n * 5), n, 5)
    y <- 5 * X[, 1] * X[, 2] * X[, 3] + 3 * sin(6 * X[, 4]) + 2 * pmax(X[, 5] - 0.4, 0) * X[, 1] + 0.1 * rnorm(n) }
  if (id == "D5") { set.seed(105); n <- 150; X <- matrix(runif(n), n, 1); y <- sin(8 * X[, 1]) + 0.1 * rnorm(n) }
  colnames(X) <- paste0("x", seq_len(ncol(X))); list(X = X, y = y)
}
design <- list(D1 = c(deg = 2, nk = 21), D2 = c(deg = 3, nk = 41), D3 = c(deg = 2, nk = 31), D4 = c(deg = 3, nk = 31), D5 = c(deg = 2, nk = 21))

fits <- list()
for (id in names(design)) {
  d <- mk(id); de <- design[[id]]
  grid <- if (id == "D5") expand.grid(fk = c(0, 3, 20), fb = c(0.5, 1)) else
    rbind(expand.grid(fk = c(1, 2, 3, 5, 20), fb = c(0, 0.5, 1, 2)), data.frame(fk = c(0, 1000), fb = 1))
  for (g in seq_len(nrow(grid))) fits[[length(fits) + 1]] <- check_fit(d$X, d$y, de[["deg"]], de[["nk"]], grid$fk[g], grid$fb[g],
    label = sprintf("%s deg%d nk%d fk%g fb%g", id, de[["deg"]], de[["nk"]], grid$fk[g], grid$fb[g]))
}
# Degree 1 (D1), and a linear predictor at degree 2 (D3 with linpreds = 1).
d1 <- mk("D1")
for (fk in c(0, 3, 20)) fits[[length(fits) + 1]] <- check_fit(d1$X, d1$y, 1, 21, fk, 1, label = sprintf("D1 deg1 nk21 fk%g fb1", fk))
d3 <- mk("D3")
for (fk in c(0, 5)) fits[[length(fits) + 1]] <- check_fit(d3$X, d3$y, 2, 31, fk, 1, lp = 1, label = sprintf("D3lin deg2 nk31 fk%g fb1", fk))

S <- function(field) sum(sapply(fits, function(z) sum(z[[field]])))
cat(sprintf("%d fits (%d at degree 1, %d at degree 3), %d steps, %d one-term steps, %d queue tables, %d rows\n",
  length(fits), sum(sapply(fits, `[[`, "degree") == 1), sum(sapply(fits, `[[`, "degree") == 3),
  sum(sapply(fits, function(z) sum(z$steps$K >= 2))), sum(sapply(fits, function(z) sum(z$steps$K >= 2 & z$steps$a == 1))), S("ntab"), S("rows")))
by_id <- split(fits, sub(" .*", "", sapply(fits, `[[`, "label")))
cat("per data set: tables; rows with AgedRank = rule; tables in rule order; visits = rule; transitions = rule; steps that search no entry\n")
for (nm in names(by_id)) { z <- by_id[[nm]]; Sz <- function(f) sum(sapply(z, function(q) sum(q[[f]])))
  cat(sprintf("  %-6s fits %2d  tables %3d  rows %4d/%4d  order %3d/%3d  visits %3d/%3d  transitions %3d/%3d  empty %d\n", nm, length(z),
    Sz("ntab"), Sz("aged"), Sz("rows"), Sz("order"), Sz("ntab"), Sz("vis_ok"), Sz("nvis"), Sz("trans_ok"), Sz("ntrans"), Sz("empty"))) }
bad <- unlist(lapply(fits, `[[`, "bad"))
if (length(bad)) cat("mismatches:\n", paste(" ", head(bad, 15)), sep = "\n")

# Example 1 (D1, degree 2, fast.k 5, fast.beta 1): one table with the rule's
# rank and AgedRank, ties in AgedRank resolved by rank, not by entry number.
show_table <- function(z, K) {
  t <- Filter(function(q) q$K == K, z$tabs)[[1]]$tab; rk <- rank_of(t$entry, t$rss)
  cat(sprintf("%s, table after K = %d (degree of each entry's slot in brackets):\n", z$label, K))
  cat(sprintf("  %s\n", paste(sprintf("%d[d%s] Km %d rss %s rank %d aged %.1f", t$entry,
    ifelse(is.na(z$sr[t$entry]), "-", z$dg[z$sr[t$entry]]), t$Km, format(t$rss, digits = 6), rk, t$aged), collapse = "\n  ")))
}
zA <- Filter(function(z) z$label == "D1 deg2 nk21 fk5 fb1", fits)[[1]]
KA <- max(sapply(zA$tabs, `[[`, "K")); show_table(zA, KA)
cat(sprintf("  next step visits %s\n", paste(ifelse(is.na(zA$vis[[as.character(KA + 2)]]$D), zA$vis[[as.character(KA + 2)]]$entry,
  sprintf("%d(skip D=%d)", zA$vis[[as.character(KA + 2)]]$entry, zA$vis[[as.character(KA + 2)]]$D)), collapse = " ")))

# Example 2 (D3 with linpreds = 1): entry c is slot c, not dirs row c.
zB <- Filter(function(z) z$label == "D3lin deg2 nk31 fk0 fb1", fits)[[1]]
one <- zB$steps$K[zB$steps$K >= 2 & zB$steps$a == 1]
cat(sprintf("%s: one-term steps at K = %s; empty slots %s\n", zB$label, paste(one, collapse = ","),
  paste(setdiff(seq_len(max(which(!is.na(zB$sr)))), which(!is.na(zB$sr))), collapse = ",")))
for (K in names(zB$vis)[2:min(8, length(zB$vis))]) { v <- zB$vis[[K]]; Kp <- as.integer(K) - 2
  miss <- which(!is.na(zB$sr) & seq_along(zB$sr) <= Kp + 1 & seq_along(zB$sr) > nrow(Filter(function(q) q$K == Kp, zB$tabs)[[1]]$tab))
  miss <- miss[zB$dg[zB$sr[miss]] < 2]
  cat(sprintf("  K=%s visits %s%s\n", K, paste(ifelse(is.na(v$D), v$entry, sprintf("%d(D=%d)", v$entry, v$D)), collapse = " "),
    if (length(miss)) sprintf("; no entry for eligible slot(s) %s = dirs row(s) %s", paste(miss, collapse = ","), paste(zB$sr[miss], collapse = ",")) else "")) }

# Example 3 (D1, degree 1, fast.k 3): the intercept falls out of the window.
zC <- Filter(function(z) z$label == "D1 deg1 nk21 fk3 fb1", fits)[[1]]
show_table(zC, max(sapply(zC$tabs, `[[`, "K")))
cat(sprintf("  steps: %s; terms %d (fast.k 20: %d, fast.k 0: %d)\n", paste(ifelse(zC$steps$reject, paste0(zC$steps$K, " reject"), zC$steps$K), collapse = ", "),
  zC$nterms, Filter(function(z) z$label == "D1 deg1 nk21 fk20 fb1", fits)[[1]]$nterms, Filter(function(z) z$label == "D1 deg1 nk21 fk0 fb1", fits)[[1]]$nterms))

# Example 4 (D5, one covariate): exact ties at RssDelta -1.
zD <- Filter(function(z) z$label == "D5 deg2 nk21 fk20 fb1", fits)[[1]]
show_table(zD, max(sapply(zD$tabs, `[[`, "K")))

# Part 2: the stored RssDelta against the knot searches of trace 9. For each
# searched entry the best value is the largest printed RssDelta among its knot
# lines that pass all four guards (bx1G, CovColG, TolG, MaxG) and its linear
# candidates (RssDeltaLin); 0 when none is positive; -1 when every predictor
# is skipped as "pred is in parent". MaxG 0 marks a knot whose RssDelta is
# above the step's MaxLegalRssDelta, which is also checked here.
num_after <- function(s, key) as.numeric(sub(paste0(".*", key, " +([-0-9.e+]+|inf|-inf|nan).*"), "\\1", s))
t9_best <- function(o9) {
  reg <- fp(o9); st <- grep("Searching for new term", reg); res <- NULL; lim <- NULL
  for (i in seq_along(st)) {
    seg <- reg[st[i]:(if (i < length(st)) st[i + 1] - 1 else length(reg))]
    K <- as.integer(sub(".*new term ([0-9]+).*", "\\1", seg[1]))
    ib <- grep("FindKnotBegin", seg)
    ml <- num_after(seg[1], "MaxLegalRssDelta"); ik0 <- grep("RssWithKnot", seg); kv0 <- num_after(seg[ik0], "RssDelta")
    mg0 <- grepl("MaxG 0", seg[ik0])
    lim <- rbind(lim, data.frame(K = K, maxlegal = ml, before = if (length(ib)) num_after(seg[ib[1]], "RssBeforeAddingHinge") else NA,
      nknot = length(ik0), maxg_ok = sum(mg0 == (kv0 > ml * (1 + 1e-5))), nmaxg0 = sum(mg0)))
    ip <- grep("^\\|Parent", seg); if (!length(ip)) next
    who <- as.integer(sub("^\\|Parent +([0-9]+).*", "\\1", seg[ip]))
    owner <- function(idx) who[findInterval(idx, ip)]
    ipred <- ip[grepl("Pred", seg[ip])]; nopred <- grepl("pred is in parent", seg[ipred])
    il <- grep("RssDeltaLin", seg); ik <- grep("RssWithKnot", seg)
    legal <- grepl("bx1G 1 CovColG 1 TolG 1 MaxG 1", seg[ik])
    kv <- num_after(seg[ik], "RssDelta"); kcol <- as.integer(sub(".*iNewCol ([0-9]+).*", "\\1", seg[ib]))[findInterval(ik, ib)]
    lv <- num_after(seg[il], "RssDeltaLin")
    for (p in unique(owner(ipred))) {
      if (all(nopred[owner(ipred) == p])) { res <- rbind(res, data.frame(K = K, p = p, best = -1, how = "no allowed pred", above = NA)); next }
      w <- owner(ik) == p; kp <- kv[w & legal]; cp <- kcol[w & legal]; lp <- lv[owner(il) == p]
      bk <- if (length(kp)) max(kp) else -Inf; bl <- if (length(lp)) max(lp) else -Inf; best <- max(0, bk, bl)
      how <- if (best == 0) "none positive" else if (bl > bk) "linear" else if (cp[which.max(kp)] == K) "single hinge" else "pair"
      above <- suppressWarnings(max(kv[w & grepl("MaxG 0", seg[ik])]))
      res <- rbind(res, data.frame(K = K, p = p, best = best, how = how, above = above))
    }
  }
  list(best = res, lim = lim)
}
cmp9 <- NULL; lim9 <- NULL
cfs <- list(list("D1", 2, 21, 5, 1, FALSE), list("D3", 2, 31, 3, 0.5, FALSE), list("D3", 2, 31, 20, 1, 1), list("D4", 3, 21, 20, 2, FALSE), list("D5", 2, 21, 20, 1, FALSE))
for (ic in seq_along(cfs)) {
  cf <- cfs[[ic]]; d <- mk(cf[[1]])
  o6 <- capture.output(f <- quiet(earth(d$X, d$y, degree = cf[[2]], nk = cf[[3]], fast.k = cf[[4]], fast.beta = cf[[5]], linpreds = cf[[6]], trace = 6, pmethod = "none", thresh = 0)))
  o9 <- capture.output(f <- quiet(earth(d$X, d$y, degree = cf[[2]], nk = cf[[3]], fast.k = cf[[4]], fast.beta = cf[[5]], linpreds = cf[[6]], trace = 9, pmethod = "none", thresh = 0)))
  tb <- parse_tables(fp(o6)); r9 <- t9_best(o9); b9 <- r9$best
  for (i in seq_len(nrow(b9))) { t <- Filter(function(z) z$K == b9$K[i], tb); if (!length(t)) next
    t <- t[[1]]$tab; stored <- t$rss[t$entry == b9$p[i]]
    cmp9 <- rbind(cmp9, data.frame(run = ic, cfg = cf[[1]], K = b9$K[i], p = b9$p[i], stored = stored, best = b9$best[i], how = b9$how[i],
      above = is.finite(b9$above[i]) && b9$above[i] > stored, ok = abs(stored - b9$best[i]) <= 1e-4 * max(1e-3, abs(stored)))) }
  # MaxLegalRssDelta = min(1.01 * RSS before the step, 10 * the RssDelta chosen in the previous step)
  mine <- cmp9$run == ic; ch <- tapply(cmp9$stored[mine], cmp9$K[mine], max)
  L <- r9$lim; prevch <- ch[as.character(L$K - 2)]
  L$tenx <- 10 * prevch; L$expect <- ifelse(L$K == 2, 1.01 * L$before, pmin(1.01 * L$before, L$tenx))
  lim9 <- rbind(lim9, L[!is.na(L$expect), ])
}
cat(sprintf("part 2: %d searched entries in 5 fits; stored RssDelta = best legal value of trace 9 in %d\n", nrow(cmp9), sum(cmp9$ok)))
cat(sprintf("  scale: the first step of D5 starts from RssBeforeAddingHinge %s with n = %d (y scaled to sd 1, total SS n - 1)\n",
  sub(".*RssBeforeAddingHinge ([-0-9.e]+).*", "\\1", grep("RssBeforeAddingHinge", o9, value = TRUE)[1]), nrow(d$X)))
print(table(cmp9$how, cmp9$ok, dnn = c("best from", "match")))
if (any(!cmp9$ok)) print(head(cmp9[!cmp9$ok, ], 8), row.names = FALSE)
ex9 <- cmp9[cmp9$how %in% c("single hinge", "none positive", "no allowed pred") | cmp9$above, ]
print(ex9[!duplicated(paste(ex9$how, ex9$above)), c("cfg", "K", "p", "stored", "best", "how", "above")], row.names = FALSE)
ok_lim <- abs(lim9$maxlegal - lim9$expect) <= 2e-5 * lim9$maxlegal
cat(sprintf("  MaxLegalRssDelta = min(1.01 RSS, 10 x previous chosen RssDelta) in %d of %d steps (the 10x bound is the smaller in %d);\n  a knot is MaxG 0 iff its RssDelta > MaxLegalRssDelta in %d of %d knot lines (%d are MaxG 0)\n",
  sum(ok_lim), nrow(lim9), sum(lim9$K > 2 & lim9$tenx < 1.01 * lim9$before, na.rm = TRUE), sum(lim9$maxg_ok), sum(lim9$nknot), sum(lim9$nmaxg0)))

# Part 3: default fast.k and fast.beta, and small or fractional fast.k.
reg_of <- function(...) { o <- capture.output(f <- quiet(earth(d1$X, d1$y, degree = 2, nk = 21, trace = 6, pmethod = "none", thresh = 0, ...))); list(r = fp(o), f = f) }
def <- reg_of(); exp1 <- reg_of(fast.k = 20, fast.beta = 1)
same_default <- identical(def$r, exp1$r)
small <- lapply(c(1, 2, 2.9, 3, 3.5, 4, 0.5), function(k) reg_of(fast.k = k))
fkl <- sapply(small, function(z) { l <- grep("^FastK", z$r, value = TRUE); if (length(l)) sub("^FastK ([0-9]+).*", "\\1", l[1]) else "none" })
cat(sprintf("printed FastK for fast.k = 1, 2, 2.9, 3, 3.5, 4, 0.5: %s\n", paste(fkl, collapse = ", ")))
same_as3 <- all(sapply(small[c(1, 2, 3, 5)], function(z) identical(z$f$cuts, small[[4]]$f$cuts) && identical(z$f$dirs, small[[4]]$f$dirs)))
half_as0 <- identical(small[[7]]$r, reg_of(fast.k = 0)$r)
if (length(warn_msgs)) cat("warnings:", paste(unique(warn_msgs), collapse = " | "), "\n")

allf <- function(pred) all(sapply(fits, pred))
dg1 <- Filter(function(z) z$degree == 1, fits); fk0 <- Filter(function(z) z$fk == 0, fits)
inter_only <- all(sapply(dg1, function(z) all(sapply(z$vis[-1], function(v) all(v$entry[is.na(v$D)] == 1)))))
k_seq <- allf(function(z) { k <- z$steps$K[z$steps$K >= 2]; identical(k, 2L * seq_along(k)) })
n20 <- Filter(function(z) z$label == "D1 deg1 nk21 fk20 fb1", fits)[[1]]$nterms
f0_same <- all(sapply(c("D1", "D2", "D3", "D4"), function(nm) { z <- by_id[[nm]]
  a <- Filter(function(q) q$fk == 0, z)[[1]]; b <- Filter(function(q) q$fk == 1000, z)[[1]]; identical(a$cuts, b$cuts) && identical(a$dirs, b$dirs) }))
negs <- unlist(lapply(fits, `[[`, "neg"))
ck <- function(id, ok, fmt, ...) cat(sprintf(paste0("CHECK bb16.%s %s ", fmt, "\n"), id, ok, ...))
ck(1, S("entries_ok") == S("ntab") && allf(function(z) z$same && z$terms_ok && z$ntab >= 1),
  "every fit prints a queue table after each step that adds terms, at degrees 1, 2 and 3 (%d tables); its entries are exactly 1..T, T = terms so far", S("ntab"))
ck(2, S("aged") == S("rows") && S("fb0_neg1") == S("fb0_rows"),
  "AgedRank = rank + fast.beta * (K - K_m) in all %d rows, rank = 0-based place by RssDelta largest first; fast.beta = 0 prints -1.0 in all %d of its rows", S("rows"), S("fb0_rows"))
ck(3, S("altA_ok") == S("ntab"), "HYPOTHESIS equal AgedRank is ordered by entry number (holds in %d of %d tables)", S("altA_ok"), S("ntab"))
ck(4, S("order") == S("ntab"), "every table is sorted by AgedRank, equal AgedRank by rank (%d tie groups, %d where rank and entry order differ)", S("tieA"), S("tieA_info"))
ck(5, S("altR_prev_ok") == S("ntab") - length(fits), "HYPOTHESIS equal RssDelta keeps the previous table's order (holds in %d of %d tables)", S("altR_prev_ok"), S("ntab") - length(fits))
ck(6, S("altR_desc_ok") == S("ntab"), "HYPOTHESIS equal RssDelta is ranked by entry number descending (holds in %d of %d tables)", S("altR_desc_ok"), S("ntab"))
ck(7, S("aged") == S("rows") && S("order") == S("ntab") && S("tieR_info") > 0,
  "equal RssDelta is ranked by entry number ascending (%d tie groups, %d finite, e.g. 0 or -1; %d where the previous order differs)", S("tieR"), S("tieR_fin"), S("tieR_info"))
ck(8, S("alt_steps_ok") == S("ntab_aged"), "HYPOTHESIS age counts steps: AgedRank = rank + fast.beta * (K - K_m) / 2 (holds in %d of %d tables)", S("alt_steps_ok"), S("ntab_aged"))
ck(9, k_seq && S("aged") == S("rows"), "K = 2, 4, 6, ... in every fit, also after the %d one-term steps; age is K - K_m, so fast.beta = 1 adds 2 per step",
  sum(sapply(fits, function(z) sum(z$steps$K >= 2 & z$steps$a == 1))))
ck(10, S("vis_ok") == S("nvis"), "the next step visits the first max(3, trunc(fast.k)) rows in table order, all rows when trunc(fast.k) = 0 (%d steps)", S("nvis"))
ck(11, same_as3 && all(fkl[1:5] == "3") && fkl[6] == "4" && half_as0 && S("fkline_ok") == S("ntab"),
  "fast.k = 1, 2, 2.9 and 3.5 print FastK 3 and fit as fast.k = 3; fast.k = 0.5 fits as 0; the FastK line follows row max(3, trunc(fast.k)) in all tables")
ck(12, S("skip_row_ok") == S("nvis"), "HYPOTHESIS entry c is dirs row c: skipped iff the degree of row c equals the degree argument (holds in %d of %d steps)", S("skip_row_ok"), S("nvis"))
ck(13, S("skip_ok") == S("nvis") && S("skipD_ok") == S("nskipD"),
  "entry c is slot c: skipped iff slot c is empty or its degree equals the degree argument (%d steps); D = degree + 1 (%d of %d), %s if empty",
  S("nvis"), S("skipD_ok"), S("nskipD"), paste(unique(unlist(lapply(fits, `[[`, "unusedD"))), collapse = ","))
ck(14, S("par_ok") == S("npar") && S("par_row_ok") < S("npar"),
  "the step line's Par is the parent's slot (%d of %d interaction steps; read as a dirs row: %d)", S("par_ok"), S("npar"), S("par_row_ok"))
ck(15, sum(sapply(fk0, function(z) z$eligible_unvisited)) == 0,
  "HYPOTHESIS with fast.k = 0 every eligible term is searched every step (%d step-term cases of an eligible term in a slot above T, never visited)", sum(sapply(fk0, function(z) z$eligible_unvisited)))
ck(16, f0_same && all(sapply(fk0, function(z) z$vis_ok == z$nvis)), "fast.k = 0 visits every entry 1..T each step in table order, and gives the fit of fast.k = 1000")
ck(17, S("trans_ok") == S("ntrans"),
  "a searched entry gets K_m = K; skipped and unvisited entries keep K_m and RssDelta; entries T+1..T+a start at inf with K_m = K (%d of %d steps)", S("trans_ok"), S("ntrans"))
ck(18, S("chosen_new") == S("nchosen"), "the chosen parent is updated too: K_m = K and the largest RssDelta among the searched entries (%d of %d)", S("chosen_new"), S("nchosen"))
ck(19, all(cmp9$ok) && any(cmp9$above), "stored RssDelta = best legal trace-9 reduction (pairs, single hinges, linpreds terms; MaxG-0 knots excluded, %d cases); 0 if none; -1 if no allowed pred (%d of %d)",
  sum(cmp9$above), sum(cmp9$ok), nrow(cmp9))
ck(20, all(ok_lim) && sum(lim9$maxg_ok) == sum(lim9$nknot), "MaxLegalRssDelta = min(1.01 * RssBeforeAddingHinge, 10 * previous chosen RssDelta) (%d of %d steps), and MaxG 0 iff RssDelta is above it (%d of %d knots)", sum(ok_lim), nrow(lim9), sum(lim9$maxg_ok), sum(lim9$nknot))
ck(21, S("int_missed") > 0, "the intercept is entry 1 in every table, but the window can leave it out (%d steps without it)", S("int_missed"))
ck(22, inter_only && tail(zC$steps$reject, 1) && zC$nterms < n20,
  "at degree 1 only the intercept is searched; fast.k = 3 ends the pass by reject (no DeltaRsq) at %d terms (fast.k 20: %d)", zC$nterms, n20)
ck(23, S("empty_reject") == S("empty") && S("empty") > 0,
  "a step that searches no entry adds nothing and ends the pass by reject (no DeltaRsq) (%d of %d); %d more rejects searched only entries without a candidate", S("empty_reject"), S("empty"), S("reject_nonempty"))
ck(24, all(negs == -1) && length(negs) > 0, "the only negative RssDelta is -1 (%d rows); 0 occurs in %d rows", length(negs), sum(unlist(lapply(fits, `[[`, "rss_vals")) == 0))
past <- unlist(lapply(fits, `[[`, "cons_past"))
ck(25, S("cons_ok") == S("ncons") && S("cons_val_ok") == S("ncons"),
  "the Considering line lists the visited entries (%d of %d), but each bracket holds the RssDelta of the NEXT row (%d of %d); the last bracket of %d lines lies past the table and is not checked",
  S("cons_ok"), S("ncons"), S("cons_val_ok"), S("ncons"), length(past))
ck(26, same_default, "the defaults are fast.k = 20 and fast.beta = 1 (identical trace 6)")
