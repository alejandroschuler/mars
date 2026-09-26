# bb15: earth's pruning pass with two or more responses, and pmethod = "none"
# with nprune.
#
# A. With K >= 2 response columns, is prune.terms the nested path of plain
#    sequential backward elimination on the RSS summed over the responses
#    (weighted when there are weights), and not the prefix rule that holds for
#    one response (bb07.12)? Fixed bases (the linpreds trick of bb07) and
#    ordinary fits, with no weights, integer weights and non-integer weights.
# B. Which routine does earth name at trace = 3 for 1 and for several
#    responses, and for a factor response with 2 and with 3 levels?
# C. pmethod = "none" with nprune = k: which terms are selected?
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))

wrss <- function(B, Y, S, w) {  # RSS summed over the columns of Y, weighted
  B <- B[, S, drop = FALSE]
  if (is.null(w)) sum(lm.fit(B, Y)$residuals^2) else sum(w * lm.wfit(B, Y, w)$residuals^2)
}
backward <- function(B, Y, w) {  # plain sequential backward elimination
  # which.min breaks an exact tie by the lowest index, not by the spec's rule;
  # no exact tie occurs in parts A1 to A3 (part D looks at ties).
  K <- ncol(B); cur <- seq_len(K); sets <- vector("list", K); sets[[K]] <- cur
  for (s in K:2) {
    cand <- cur[cur != 1L]; r <- vapply(cand, function(j) wrss(B, Y, setdiff(cur, j), w), 0)
    cur <- setdiff(cur, cand[which.min(r)]); sets[[s - 1]] <- cur
  }
  sets
}
prefix <- function(B, Y, w) {  # PRUNE-3's prefix rule (bb07.12), on the same summed RSS
  K <- ncol(B); o <- seq_len(K); best <- rep(Inf, K); sets <- vector("list", K)
  offer <- function() for (m in seq_len(K)) { r <- wrss(B, Y, o[seq_len(m)], w); if (r < best[m]) { best[m] <<- r; sets[[m]] <<- sort(o[seq_len(m)]) } }
  offer()
  for (pos in K:2) {
    r <- vapply(2:pos, function(i) wrss(B, Y, o[seq_len(pos)][-i], w), 0)
    i <- (2:pos)[which.min(r)]; o <- append(o[-i], o[i], after = pos - 1); offer()
  }
  sets
}
pt_sets <- function(f) { P <- f$prune.terms; lapply(seq_len(nrow(P)), function(k) sort(as.integer(P[k, P[k, ] != 0]))) }
same <- function(a, b) length(a) == length(b) && all(mapply(function(u, v) identical(as.integer(u), as.integer(v)), a, b))
nested <- function(s) all(vapply(seq_len(length(s) - 1), function(k) all(s[[k]] %in% s[[k + 1]]), TRUE))
basis <- function(f, X) sapply(seq_len(nrow(f$dirs)), function(k) { v <- rep(1, nrow(X))
  for (j in which(f$dirs[k, ] != 0)) v <- v * switch(as.character(f$dirs[k, j]),
    "1" = pmax(X[, j] - f$cuts[k, j], 0), "-1" = pmax(f$cuts[k, j] - X[, j], 0), "2" = X[, j]); v })

tally <- data.frame()
record <- function(label, f, B, Y, w) {
  es <- pt_sets(f); bw <- backward(B, Y, w); pf <- prefix(B, Y, w)
  rss_ok <- all(abs(f$rss.per.subset - vapply(es, function(S) wrss(B, Y, S, w), 0)) <= 1e-9 * f$rss.per.subset)
  tally <<- rbind(tally, data.frame(label = label, K = ncol(as.matrix(Y)), weights = !is.null(w), backward = same(es, bw),
    prefix = same(es, pf), rules_differ = !same(bw, pf), nested = nested(es), rss_ok = rss_ok))
}

# Part A1: fixed bases through linpreds, several responses
for (i in 1:24) {
  set.seed(1500 + i); n <- sample(c(60, 100, 150), 1); p <- sample(6:10, 1)
  X <- matrix(runif(n * p), n, p); colnames(X) <- paste0("c", seq_len(p))
  X[, 3] <- pmax(X[, 1] - 0.4, 0); X[, 4] <- X[, 2] * X[, 1]
  f1 <- 2 * X[, 1] + X[, 3] - X[, 4]
  K <- if (i %% 2) 2 else 3
  Y <- cbind(f1 + rnorm(n), -f1 + X[, 5] + rnorm(n), sin(3 * X[, 2]) + 0.5 * rnorm(n))[, seq_len(K)]
  w <- switch(i %% 3 + 1, NULL, as.numeric(sample(1:3, n, TRUE)), runif(n, 0.5, 2))
  f <- quiet(earth(X, Y, weights = w, linpreds = TRUE, degree = 1, thresh = 0, nk = 2 * p + 1))
  if (is.character(f) || nrow(f$dirs) != p + 1) next
  ct <- apply(f$dirs[-1, , drop = FALSE], 1, function(r) which(r == 2))
  record("fixed basis", f, cbind(1, X[, ct, drop = FALSE]), Y, w)
}
# Part A2: ordinary fits with several responses
for (i in 1:12) {
  set.seed(2500 + i); n <- 250; p <- 4; X <- matrix(runif(n * p), n, p)
  g <- sin(3 * X[, 1]) * X[, 2] + 2 * pmax(X[, 3] - 0.5, 0) * pmax(X[, 1] - 0.3, 0) * X[, 2]
  K <- if (i %% 2) 2 else 3
  Y <- cbind(g + 0.3 * rnorm(n), 2 * g + X[, 1] + 0.5 * rnorm(n), cos(2 * X[, 2]) + 0.2 * rnorm(n))[, seq_len(K)]
  w <- switch(i %% 3 + 1, NULL, as.numeric(sample(1:3, n, TRUE)), runif(n, 0.5, 2))
  f <- quiet(earth(X, Y, weights = w, degree = 1 + i %% 3, nk = 31))
  if (is.character(f)) next
  record("ordinary fit", f, basis(f, X), Y, w)
}
# Part A3: the same ordinary designs with one response, for contrast
for (i in 1:6) {
  set.seed(2500 + i); n <- 250; p <- 4; X <- matrix(runif(n * p), n, p)
  g <- sin(3 * X[, 1]) * X[, 2] + 2 * pmax(X[, 3] - 0.5, 0) * pmax(X[, 1] - 0.3, 0) * X[, 2]
  y <- g + 0.3 * rnorm(n)
  f <- quiet(earth(X, y, degree = 1 + i %% 3, nk = 31))
  record("one response", f, basis(f, X), matrix(y), NULL)
}
agg <- aggregate(cbind(backward, prefix, rules_differ, nested, rss_ok) ~ label + K + weights, data = tally, FUN = sum)
agg$fits <- aggregate(backward ~ label + K + weights, data = tally, FUN = length)$backward
print(agg[order(agg$label, agg$K, agg$weights), ], row.names = FALSE)
multi <- tally[tally$K >= 2, ]; one <- tally[tally$K == 1, ]

# Part B: the routine named at trace = 3
set.seed(9); XB <- matrix(runif(100 * 3), 100, 3); z <- XB[, 1]
routine <- function(Y) {
  out <- capture.output(quiet(earth(XB, Y, trace = 3, nk = 11)))
  m <- grep("EvalSubsetsUsingXtx|leaps", out, value = TRUE)
  if (length(m)) trimws(m[1]) else "(no routine named)"
}
msgs <- c(one = routine(z + 0.3 * sin(9 * XB[, 2])), two = routine(cbind(z, z^2)), three = routine(cbind(z, z^2, sin(z))),
  factor2 = routine(factor(ifelse(z > 0.5, "a", "b"))), factor3 = routine(factor(cut(z, c(0, 0.3, 0.6, 1)))))
for (k in names(msgs)) cat(sprintf("trace 3, %-8s response: %s\n", k, msgs[[k]]))
xtx_multi <- all(grepl("EvalSubsetsUsingXtx", msgs[c("two", "three", "factor3")])) && !grepl("EvalSubsetsUsingXtx", msgs["one"]) &&
  !grepl("EvalSubsetsUsingXtx", msgs["factor2"])

# Part C: pmethod = "none" with nprune
set.seed(1); n <- 200; x <- matrix(runif(n * 5), n, 5)
yf <- 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5] + rnorm(n)
selC <- sapply(c(3, 5, 8), function(k) {
  fk <- quiet(earth(x, yf, pmethod = "none", nprune = k))
  identical(sort(as.integer(fk$selected.terms)), seq_len(k))
})
f3 <- quiet(earth(x, yf, pmethod = "none", nprune = 5)); fb <- quiet(earth(x, yf))
cat(sprintf("part C: %d forward terms; pmethod none with nprune 5 selects %s; the backward subset of size 5 is %s\n",
  nrow(f3$dirs), paste(sort(f3$selected.terms), collapse = " "), paste(sort(fb$prune.terms[5, 1:5]), collapse = " ")))
# The statistics that earth reports in that case, for one response, weights and two responses
statC <- function(Y, w = NULL, k = 6) {
  set.seed(12); n <- 150; X <- matrix(runif(n * 4), n, 4)
  Yv <- if (is.function(Y)) Y(X, n) else Y
  f <- quiet(earth(X, Yv, weights = w, degree = 2, pmethod = "none", nprune = k))
  Ym <- as.matrix(Yv); ww <- if (is.null(w)) rep(1, n) else w
  bx <- basis(f, X)[, sort(f$selected.terms), drop = FALSE]
  cf_ok <- isTRUE(all.equal(unname(as.matrix(f$coefficients)), unname(as.matrix(if (is.null(w)) lm.fit(bx, Ym)$coefficients else lm.wfit(bx, Ym, w)$coefficients)), tolerance = 1e-10))
  rss_model <- sum(ww * as.matrix(f$residuals)^2)
  c(first_k = identical(sort(as.integer(f$selected.terms)), seq_len(k)), coef_first_k = cf_ok,
    rss_is_Tk = isTRUE(all.equal(f$rss, f$rss.per.subset[k], tolerance = 1e-12)), rss_is_model = isTRUE(all.equal(f$rss, rss_model, tolerance = 1e-8)),
    rss = f$rss, rss_model = rss_model)
}
gy <- function(X, n) 4 * pmax(X[, 1] - 0.3, 0) * X[, 2] + sin(5 * X[, 3]) + 0.2 * rnorm(n)
set.seed(1); wC <- runif(150, 0.5, 2)
sc <- rbind(one = statC(gy), weighted = statC(gy, wC), two = statC(function(X, n) cbind(gy(X, n), cos(3 * X[, 4]) + 0.2 * rnorm(n))))
print(sc, digits = 6)
quirkC <- all(sc[, "first_k"] == 1) && all(sc[, "coef_first_k"] == 1) && all(sc[, "rss_is_Tk"] == 1) && !any(sc[, "rss_is_model"] == 1)

# Part D: exact ties with two responses. Rows come in pairs that swap two
# columns, so removing either column gives the same summed RSS in exact
# arithmetic. Which one does earth remove first, in three column orders?
tieD <- NULL
for (i in 1:8) {
  set.seed(4000 + i); m <- 40; a <- runif(m); b <- runif(m); z1 <- runif(m); z2 <- runif(m); e1 <- rnorm(m); e2 <- rnorm(m)
  Xd <- rbind(cbind(a, b, z1, z2), cbind(b, a, z1, z2))
  Yd <- rbind(cbind(3 * z1 + 0.1 * (a + b) + e1, 2 * z2 + e2), cbind(3 * z1 + 0.1 * (b + a) + e1, 2 * z2 + e2))  # a and b are the two weakest, and tied
  for (ord in list(1:4, c(2, 1, 3, 4), c(3, 4, 1, 2))) {
    Xo <- Xd[, ord, drop = FALSE]; colnames(Xo) <- c("a", "b", "z1", "z2")[ord]
    f <- quiet(earth(Xo, Yd, linpreds = TRUE, degree = 1, thresh = 0, nk = 9))
    if (is.character(f) || nrow(f$dirs) != 5) next
    ct <- colnames(Xo)[apply(f$dirs[-1, , drop = FALSE], 1, function(r) which(r == 2))]
    es <- pt_sets(f)
    first_removed <- setdiff(es[[5]], es[[4]])
    tieD <- rbind(tieD, data.frame(design = i, order = paste(colnames(Xo), collapse = ","), removed = ct[first_removed - 1],
      removed_term = first_removed))
  }
}
print(tieD, row.names = FALSE)
tied_rows <- tieD[tieD$removed %in% c("a", "b"), ]
by_col <- tapply(tied_rows$removed, tied_rows$design, function(v) length(unique(v)))
cat(sprintf("CHECK bb15.1 %s with 2 or 3 responses, prune.terms is plain backward elimination on the summed RSS (weighted with weights) in all %d fits\n",
  all(multi$backward), nrow(multi)))
cat(sprintf("CHECK bb15.2 %s with 2 or 3 responses the rows of prune.terms are nested in all fits\n", all(multi$nested)))
cat(sprintf("CHECK bb15.3 %s HYPOTHESIS the prefix rule (PRUNE-3 for one response) also holds with 2 or 3 responses (it holds in %d of the %d fits where the two rules differ)\n",
  all(multi$prefix[multi$rules_differ]), sum(multi$prefix[multi$rules_differ]), sum(multi$rules_differ)))
cat(sprintf("CHECK bb15.4 %s with one response the prefix rule holds and plain backward elimination fails where they differ (%d fits, %d differ)\n",
  all(one$prefix) && !any(one$backward[one$rules_differ]), nrow(one), sum(one$rules_differ)))
cat(sprintf("CHECK bb15.5 %s rss.per.subset is the summed (weighted) RSS of each row in all fits\n", all(tally$rss_ok)))
cat(sprintf("CHECK bb15.6 %s at trace 3 earth names EvalSubsetsUsingXtx for 2 and 3 responses and a 3-level factor, and no routine (leaps) for one response or a 2-level factor\n", xtx_multi))
cat(sprintf("CHECK bb15.7 %s with pmethod none and nprune = k (3, 5, 8), earth selects the first k forward terms\n", all(selC)))
cat(sprintf("CHECK bb15.8 %s with pmethod none and nprune = k < M_f earth returns the coefficients of the first k terms but reports rss = rss.per.subset[k], the RSS of the backward subset T[k], not of its returned model (one response, weights, two responses)\n", quirkC))
same_in_orders <- all(tapply(tieD$removed, tieD$design, function(v) length(unique(v)) == 1))
cat(sprintf("CHECK bb15.9 %s with two responses, in each tie design the three column orders give the same removed data column (%d designs, %d with a tied column removed first)\n",
  same_in_orders, length(unique(tieD$design)), length(unique(tied_rows$design))))
cat(sprintf("CHECK bb15.10 %s with two responses the removed tied column is a in some designs and b in others, at term 4 in some and 5 in others, so no rule by column or term number decides: rounding does\n",
  length(unique(tied_rows$removed)) > 1 && length(unique(tied_rows$removed_term)) > 1))
