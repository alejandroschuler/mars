# bb07_prune_fixed.R
# Question: on a fixed basis, what does earth's pruning pass (pmethod = "backward")
# compute? Is it a plain sequential backward elimination that removes, at each
# step, the non-intercept term whose removal gives the lowest RSS? Are the rows of
# prune.terms nested? Do rss.per.subset and gcv.per.subset follow from the
# subsets, and is the selected model the subset at which.min(gcv.per.subset)?
# Also: what do pmethod = "none", nprune = k, penalty = -1 with nprune = k and
# pmethod = "exhaustive" do?
# Method: earth is a black box. The basis is given as x with linpreds = TRUE,
# degree = 1, thresh = 0 and nk = 2 * ncol(x) + 1 (each linear term uses two nk
# slots in the forward pass, so nk = ncol(x) + 1 stops the forward pass early).
# All RSS values of my own come from lm.fit on the term columns.
suppressMessages(library(earth))
# On failure print only the condition message (no call, no traceback).
earth_bb <- function(...) tryCatch(earth(...), error = function(e) stop(paste("ERROR:", conditionMessage(e)), call. = FALSE))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

checks <- character(0)
CHECK <- function(id, ok, desc) {
  checks <<- c(checks, sprintf("CHECK %s %s %s", id, if (isTRUE(ok)) "TRUE" else "FALSE", desc))
}
rel <- function(a, b) max(abs(a - b) / pmax(abs(b), 1e-300))
fmt <- function(S) paste(S, collapse = " ")
same_sets <- function(a, b) length(a) == length(b) && all(mapply(function(u, v) identical(as.integer(u), as.integer(v)), a, b))

rss_of <- function(B, y, S) sum(lm.fit(B[, S, drop = FALSE], y)$residuals^2)
gcv_of <- function(rss, k, pen, n) vapply(seq_along(rss), function(i) earth:::get.gcv(rss[i], k[i], pen, n), 0)
fit_fixed <- function(X, y, ...) earth_bb(x = X, y = y, linpreds = TRUE, degree = 1, thresh = 0, nk = 2 * ncol(X) + 1, ...)

# TRUE when the forward pass entered every column once, linearly, and nothing else.
basis_ok <- function(fit, X) {
  d <- fit$dirs
  nrow(d) == ncol(X) + 1 && all(d[1, ] == 0) && all(d[d != 0] == 2) &&
    all(rowSums(d[-1, , drop = FALSE] != 0) == 1) &&
    identical(sort(as.integer(apply(d[-1, , drop = FALSE], 1, function(r) which(r != 0)))), seq_len(ncol(X)))
}
# Basis matrix with columns in earth's term order (term 1 is the intercept).
term_basis <- function(fit, X) {
  ct <- apply(fit$dirs[-1, , drop = FALSE], 1, function(r) which(r == 2))
  cbind(1, X[, ct, drop = FALSE])
}
pt_sets <- function(fit) {
  P <- fit$prune.terms
  lapply(seq_len(nrow(P)), function(k) sort(as.integer(P[k, P[k, ] != 0])))
}
pt_rowsize_ok <- function(fit) {
  P <- fit$prune.terms
  all(vapply(seq_len(nrow(P)), function(k) sum(P[k, ] != 0) == k && all(P[k, seq_len(k)] != 0), TRUE))
}

# Hypothesis: plain sequential backward elimination by lowest RSS.
backward_path <- function(B, y) {
  K <- ncol(B); cur <- seq_len(K); sets <- vector("list", K); rss <- numeric(K)
  sets[[K]] <- cur; rss[K] <- rss_of(B, y, cur); removed <- integer(0)
  for (s in K:2) {
    cand <- cur[cur != 1L]
    r <- vapply(cand, function(j) rss_of(B, y, setdiff(cur, j)), 0)
    j <- cand[which.min(r)]; removed <- c(removed, j)
    cur <- setdiff(cur, j); sets[[s - 1]] <- cur; rss[s - 1] <- min(r)
  }
  list(sets = sets, rss = rss, removed = removed)
}

# Corrected rule ("prefix rule"). A working order of the terms starts as the
# forward-pass order 1..K. Every prefix of the working order is a candidate: for
# each size m, keep the prefix of length m with the lowest RSS seen so far (a later
# prefix replaces it only if its RSS is strictly lower). Then for pos = K down to 2:
# among positions 2..pos, find the term whose removal from the first pos terms
# gives the lowest RSS, move it to position pos (the terms after it shift one
# place left) and offer every prefix of the new order again.
prefix_rule <- function(B, y, move = "shift") {
  K <- ncol(B); ord <- seq_len(K); best <- rep(Inf, K); sets <- vector("list", K)
  offer <- function() {
    for (m in seq_len(K)) {
      S <- ord[seq_len(m)]; r <- rss_of(B, y, S)
      if (r < best[m]) { best[m] <<- r; sets[[m]] <<- sort(S) }
    }
  }
  offer()
  for (pos in K:2) {
    r <- vapply(2:pos, function(i) rss_of(B, y, ord[seq_len(pos)][-i]), 0)
    i <- (2:pos)[which.min(r)]
    if (move == "shift") {
      ord <- append(ord[-i], ord[i], after = pos - 1)
    } else {  # alternative: swap the removed term with the term at position pos
      tmp <- ord[pos]; ord[pos] <- ord[i]; ord[i] <- tmp
    }
    offer()
  }
  list(sets = sets, rss = best)
}
# Alternative rule: for each size, the forward-order prefix or the backward-path
# subset, whichever has the lower RSS (no other prefixes).
prefix_or_path <- function(B, y) {
  K <- ncol(B); p <- backward_path(B, y)$sets
  lapply(seq_len(K), function(k) {
    a <- seq_len(k); b <- sort(p[[k]])
    if (rss_of(B, y, a) < rss_of(B, y, b)) a else b
  })
}

# ---------------------------------------------------------------- datasets
make_data <- function(id) {
  if (id == "D1") {
    set.seed(701); n <- 100; x1 <- runif(n); x2 <- runif(n); x3 <- runif(n)
    X <- cbind(h1a = pmax(x1 - 0.25, 0), h1b = pmax(0.7 - x1, 0), l2 = x2, h2a = pmax(x2 - 0.5, 0),
               l3 = x3, h3b = pmax(0.4 - x3, 0), p12 = pmax(x1 - 0.25, 0) * x2, p13 = pmax(0.7 - x1, 0) * x3)
    y <- 3 * X[, "h1a"] - 2 * X[, "h1b"] + 2 * X[, "l2"] + 1.5 * X[, "p12"] + rnorm(n, sd = 0.5)
  } else if (id == "D2") {
    set.seed(702); n <- 150; x1 <- runif(n); x2 <- runif(n); x3 <- runif(n)
    X <- cbind(l1 = x1, h1a = pmax(x1 - 0.5, 0), l2 = x2, h2b = pmax(0.6 - x2, 0), l3 = x3, p23 = x2 * x3)
    y <- 2 * X[, "l1"] + 4 * X[, "h1a"] - 3 * X[, "h2b"] + 2 * X[, "p23"] + X[, "l3"] + rnorm(n, sd = 0.2)
  } else if (id == "D3") {  # 12 columns, weak signal
    set.seed(703); n <- 200; x1 <- runif(n); x2 <- runif(n); x3 <- runif(n); x4 <- runif(n)
    X <- cbind(l1 = x1, h1a = pmax(x1 - 0.3, 0), h1b = pmax(0.6 - x1, 0), l2 = x2, h2a = pmax(x2 - 0.5, 0),
               h2b = pmax(0.4 - x2, 0), l3 = x3, h3a = pmax(x3 - 0.2, 0), l4 = x4, h4b = pmax(0.7 - x4, 0),
               p12 = pmax(x1 - 0.3, 0) * x2, p34 = x3 * pmax(0.7 - x4, 0))
    y <- 0.6 * x1 + 0.8 * pmax(x2 - 0.5, 0) + 0.5 * X[, "p34"] + rnorm(n, sd = 0.5)
  } else if (id == "D4") {
    set.seed(704); n <- 60; x1 <- runif(n); x2 <- runif(n); x3 <- runif(n)
    X <- cbind(l1 = x1, h1a = pmax(x1 - 0.4, 0), h1b = pmax(0.5 - x1, 0), l2 = x2, h2a = pmax(x2 - 0.3, 0),
               h2b = pmax(0.8 - x2, 0), l3 = x3, h3a = pmax(x3 - 0.6, 0), p12 = x1 * x2, p23 = pmax(x2 - 0.3, 0) * x3)
    y <- 1 + 2 * X[, "h1a"] - 1.5 * X[, "l2"] + 3 * X[, "h3a"] + 2 * X[, "p12"] + rnorm(n, sd = 0.4)
  }
  list(X = X, y = y)
}

acc <- list(basis = TRUE, icpt = TRUE, size1 = TRUE, rowsize = TRUE, nested = TRUE, path = TRUE,
            rsspath = TRUE, rssown = TRUE, gcv = TRUE, gcvown = TRUE, sel = TRUE, prefix = TRUE, le = TRUE)
analyse <- function(fit, X, y, verbose, label) {
  K <- nrow(fit$dirs); n <- length(y); pen <- fit$penalty
  bok <- basis_ok(fit, X)
  acc$basis <<- acc$basis && bok
  if (!bok) { cat(label, "fixed basis NOT entered: nrow(dirs) =", K, "\n"); return(invisible(NULL)) }
  B <- term_basis(fit, X); es <- pt_sets(fit); bp <- backward_path(B, y); pr <- prefix_rule(B, y)
  own_rss <- vapply(es, function(S) rss_of(B, y, S), 0)
  nested <- all(vapply(1:(K - 1), function(k) all(es[[k]] %in% es[[k + 1]]), TRUE))
  path_same <- same_sets(es, lapply(bp$sets, sort))
  wm <- which.min(fit$gcv.per.subset)
  sel_ok <- identical(sort(as.integer(fit$selected.terms)), es[[wm]])
  acc$icpt <<- acc$icpt && all(vapply(es, function(S) 1L %in% S, TRUE))
  acc$size1 <<- acc$size1 && identical(es[[1]], 1L)
  acc$rowsize <<- acc$rowsize && pt_rowsize_ok(fit)
  acc$nested <<- acc$nested && nested
  acc$path <<- acc$path && path_same
  acc$rsspath <<- acc$rsspath && rel(fit$rss.per.subset, bp$rss) < 1e-10
  acc$rssown <<- acc$rssown && rel(fit$rss.per.subset, own_rss) < 1e-10
  acc$gcv <<- acc$gcv && rel(fit$gcv.per.subset, gcv_of(fit$rss.per.subset, 1:K, pen, n)) < 1e-10
  acc$gcvown <<- acc$gcvown && rel(fit$gcv.per.subset, gcv_of(own_rss, 1:K, pen, n)) < 1e-10
  acc$sel <<- acc$sel && sel_ok
  acc$prefix <<- acc$prefix && same_sets(es, pr$sets)
  acc$le <<- acc$le && all(fit$rss.per.subset <= bp$rss * (1 + 1e-10))
  if (verbose) {
    cat(sprintf("%s: n=%d ncol=%d K=%d penalty=%g basis_ok=%s termcond=%d\n", label, n, ncol(X), K, pen, bok, fit$termcond))
    cat("  term -> column:", paste0(2:K, "=", colnames(B)[-1], collapse = " "), "\n")
    cat("  my backward removal order (term numbers):", fmt(bp$removed), "\n")
    cat("  size | earth prune.terms row | my backward subset | earth rss.per.subset | rel diff vs lm.fit of earth row\n")
    for (k in 1:K) {
      flag <- if (identical(es[[k]], as.integer(sort(bp$sets[[k]])))) "" else "  <-- differs"
      cat(sprintf("  %2d | %-28s | %-28s | %.17g | %.1e%s\n", k, fmt(es[[k]]), fmt(sort(bp$sets[[k]])),
                  fit$rss.per.subset[k], abs(fit$rss.per.subset[k] / own_rss[k] - 1), flag))
    }
    cat(sprintf("  nested=%s same_as_backward=%s prefix_rule_same=%s which.min(gcv)=%d selected=%s sel_ok=%s\n",
                nested, path_same, same_sets(es, pr$sets), wm, fmt(sort(fit$selected.terms)), sel_ok))
    cat(sprintf("  max rel diff: rss vs my path %.2e | rss vs lm.fit(earth rows) %.2e | gcv vs get.gcv(rss) %.2e\n",
                rel(fit$rss.per.subset, bp$rss), rel(fit$rss.per.subset, own_rss),
                rel(fit$gcv.per.subset, gcv_of(fit$rss.per.subset, 1:K, pen, n))))
  }
  invisible(list(K = K, wm = wm, B = B, es = es))
}

res <- list()
for (id in c("D1", "D2", "D3", "D4")) {
  d <- make_data(id); fit <- fit_fixed(d$X, d$y)
  res[[id]] <- c(analyse(fit, d$X, d$y, TRUE, id), list(fit = fit, X = d$X, y = d$y))
}
designed_nested <- acc$nested; designed_path <- acc$path; designed_prefix <- acc$prefix

# ------------------------------------------------ random bases (corrected rule)
nvalid <- 0; nnest <- 0; npath <- 0; nprefix <- 0; nswap <- 0; nporp <- 0; ntry <- 200
for (it in seq_len(ntry)) {
  set.seed(7100 + it)
  n <- sample(50:200, 1); p <- sample(6:12, 1); Z <- matrix(runif(n * 4), n, 4); used <- integer(0); cols <- list()
  for (j in seq_len(p)) {
    typ <- sample(1:4, 1); v <- sample(1:4, 1); t <- runif(1, 0.15, 0.85)
    if (typ == 1 && v %in% used) typ <- 2
    if (typ == 1) used <- c(used, v)
    cols[[j]] <- switch(typ, Z[, v], pmax(Z[, v] - t, 0), pmax(t - Z[, v], 0), pmax(Z[, v] - t, 0) * Z[, (v %% 4) + 1])
  }
  X <- do.call(cbind, cols); colnames(X) <- paste0("c", seq_len(p))
  y <- drop(X %*% (rnorm(p) * rbinom(p, 1, 0.6))) + rnorm(n, sd = runif(1, 0.1, 1))
  fit <- fit_fixed(X, y)
  if (!basis_ok(fit, X)) next  # the forward pass refused a column: not a fixed basis
  nvalid <- nvalid + 1

  analyse(fit, X, y, FALSE, "")
  B <- term_basis(fit, X); es <- pt_sets(fit)
  nnest <- nnest + all(vapply(1:(nrow(fit$dirs) - 1), function(k) all(es[[k]] %in% es[[k + 1]]), TRUE))
  npath <- npath + same_sets(es, lapply(backward_path(B, y)$sets, sort))
  nprefix <- nprefix + same_sets(es, prefix_rule(B, y)$sets)
  nswap <- nswap + same_sets(es, prefix_rule(B, y, "swap")$sets)
  nporp <- nporp + same_sets(es, prefix_or_path(B, y))
}
cat(sprintf("Random bases: %d of %d tries entered every column (others skipped); prune.terms nested in %d, equal to my backward path in %d, equal to the prefix rule in %d\n",
            nvalid, ntry, nnest, npath, nprefix))
cat(sprintf("  alternatives: prefix rule with swap instead of shift matches %d; 'forward prefix or backward path' matches %d (of %d)\n",
            nswap, nporp, nvalid))

# ------------------------------------------------ pmethod = "none" (D3)
d3 <- res$D3; K3 <- d3$K; f3 <- d3$fit; n3 <- length(d3$y)
fn <- fit_fixed(d3$X, d3$y, pmethod = "none")
cat("D3 pmethod='none': selected.terms =", fmt(fn$selected.terms), "\n")
cat("  rss.per.subset:", paste(sprintf("%.10g", fn$rss.per.subset), collapse = " "), "\n")
cat("  gcv.per.subset:", paste(sprintf("%.10g", fn$gcv.per.subset), collapse = " "), "\n")
none_all <- identical(sort(as.integer(fn$selected.terms)), seq_len(K3))
none_same <- identical(fn$prune.terms, f3$prune.terms) && identical(fn$rss.per.subset, f3$rss.per.subset) &&
  identical(fn$gcv.per.subset, f3$gcv.per.subset)
cat("  prune.terms, rss.per.subset, gcv.per.subset identical to pmethod='backward':", none_same, "\n")

# ------------------------------------------------ nprune = k (D3)
cat(sprintf("D3 default: selected size %d of %d; gcv.per.subset: %s\n", d3$wm, K3,
            paste(sprintf("%.6g", f3$gcv.per.subset), collapse = " ")))
np_full <- TRUE; np_sel <- TRUE; np_path <- TRUE
for (k in c(2, 3, 6, 9)) {
  fk <- fit_fixed(d3$X, d3$y, nprune = k)
  selk <- length(fk$selected.terms); want <- which.min(f3$gcv.per.subset[1:k])
  full_len <- nrow(fk$prune.terms) == K3 && length(fk$rss.per.subset) == K3 && length(fk$gcv.per.subset) == K3
  same_k <- identical(fk$prune.terms[1:k, ], f3$prune.terms[1:k, ]) && identical(fk$rss.per.subset[1:k], f3$rss.per.subset[1:k])
  same_all <- identical(fk$prune.terms, f3$prune.terms) && identical(fk$rss.per.subset, f3$rss.per.subset)
  cat(sprintf("  nprune=%d: dim(prune.terms)=%s length(rss.per.subset)=%d selected size=%d which.min(gcv[1:k])=%d rows<=k identical=%s all rows identical=%s\n",
              k, paste(dim(fk$prune.terms), collapse = "x"), length(fk$rss.per.subset), selk, want, same_k, same_all))
  np_full <- np_full && full_len; np_sel <- np_sel && selk == want &&
    identical(sort(as.integer(fk$selected.terms)), d3$es[[want]]); np_path <- np_path && same_k
}

# ------------------------------------------------ penalty = -1 (D3)
pm_sel <- TRUE; pm_gcv <- TRUE
for (k in c(2, 5, 9, 12)) {
  fm <- fit_fixed(d3$X, d3$y, penalty = -1, nprune = k)
  cat(sprintf("  penalty=-1 nprune=%d: selected size %d (terms %s)\n", k, length(fm$selected.terms), fmt(sort(fm$selected.terms))))
  pm_sel <- pm_sel && length(fm$selected.terms) == k
  pm_gcv <- pm_gcv && rel(fm$gcv.per.subset, fm$rss.per.subset / n3) < 1e-12
}
fm0 <- fit_fixed(d3$X, d3$y, penalty = -1)
cat(sprintf("  penalty=-1 no nprune: selected size %d of %d\n", length(fm0$selected.terms), K3))

# ------------------------------------------------ pmethod = "exhaustive" (small basis)
set.seed(7022); n <- 60; u1 <- runif(n); u2 <- runif(n)
Xe <- cbind(a = pmax(u1 - 0.3, 0), b = pmax(0.7 - u1, 0), c = u2, d = pmax(u2 - 0.4, 0), e = pmax(u1 - 0.3, 0) * u2, f = u1)
ye <- drop(Xe %*% (rnorm(6) * rbinom(6, 1, 0.7))) + rnorm(n, sd = 0.3)
fb <- fit_fixed(Xe, ye); fe <- fit_fixed(Xe, ye, pmethod = "exhaustive")
ex_basis <- basis_ok(fb, Xe) && basis_ok(fe, Xe)
Be <- term_basis(fb, Xe); Ke <- ncol(Be); eb <- pt_sets(fb); ee <- pt_sets(fe)
brute <- lapply(1:Ke, function(k) {
  if (k == 1) return(list(S = 1L, r = rss_of(Be, ye, 1L)))
  cm <- combn(2:Ke, k - 1); r <- apply(cm, 2, function(s) rss_of(Be, ye, c(1L, s)))
  list(S = sort(c(1L, cm[, which.min(r)])), r = min(r))
})
cat("Exhaustive on 6-column basis (n=60): size | backward row (rss) | exhaustive row (rss) | brute-force best\n")
ex_brute <- TRUE; ex_diff <- FALSE
for (k in 1:Ke) {
  rb <- rss_of(Be, ye, eb[[k]]); re <- rss_of(Be, ye, ee[[k]])
  ex_brute <- ex_brute && identical(ee[[k]], as.integer(brute[[k]]$S))
  if (!identical(eb[[k]], ee[[k]]) && re < rb) ex_diff <- TRUE
  cat(sprintf("  %d | %-14s (%.10g) | %-14s (%.10g) | %s\n", k, fmt(eb[[k]]), rb, fmt(ee[[k]]), re, fmt(brute[[k]]$S)))
}

# ------------------------------------------------ ordinary earth fits (hinge bases)
# The prefix rule applied to the full forward-pass basis, rebuilt from dirs and cuts
# (code 1: pmax(x - cut, 0), -1: pmax(cut - x, 0), 2: x), Friedman #1 data.
full_basis <- function(fit, x) {
  sapply(seq_len(nrow(fit$dirs)), function(t) {
    col <- rep(1, nrow(x))
    for (p in seq_len(ncol(x))) {
      d <- fit$dirs[t, p]; ct <- fit$cuts[t, p]
      if (d == 1) col <- col * pmax(x[, p] - ct, 0) else if (d == -1) col <- col * pmax(ct - x[, p], 0) else if (d == 2) col <- col * x[, p]
    }
    col
  })
}
nord <- 0; nord_ok <- 0; nord_nest <- 0
for (s in 1:8) {
  set.seed(s); n <- 200; xf <- matrix(runif(n * 5), n, 5); colnames(xf) <- paste0("x", 1:5)
  yf <- 10 * sin(pi * xf[, 1] * xf[, 2]) + 20 * (xf[, 3] - 0.5)^2 + 10 * xf[, 4] + 5 * xf[, 5] + rnorm(n)
  for (deg in 1:2) {
    fo <- earth_bb(x = xf, y = yf, degree = deg); Bo <- full_basis(fo, xf); eo <- pt_sets(fo)
    nord <- nord + 1; nord_ok <- nord_ok + same_sets(eo, prefix_rule(Bo, yf)$sets)
    nord_nest <- nord_nest + all(vapply(1:(length(eo) - 1), function(k) all(eo[[k]] %in% eo[[k + 1]]), TRUE))
  }
}
cat(sprintf("Ordinary earth fits (Friedman #1, n=200, seeds 1-8, degree 1 and 2, default settings): prefix rule reproduces prune.terms in %d of %d; rows nested in %d\n",
            nord_ok, nord, nord_nest))

# ------------------------------------------------ checks
CHECK("bb07.1", acc$basis && ex_basis, "fixed basis: every designed dataset entered each column once, linearly (dirs 2), nrow(dirs) = ncol(x)+1")
CHECK("bb07.2", acc$icpt, "the intercept (term 1) is in every row of prune.terms (designed + random bases)")
CHECK("bb07.3", acc$size1, "row 1 of prune.terms is the intercept alone")
CHECK("bb07.4", acc$rowsize, "row k of prune.terms lists exactly k terms (row k is the subset of size k)")
CHECK("bb07.5", designed_nested, "HYPOTHESIS rows of prune.terms are nested (size k subset inside size k+1 subset) in all designed datasets")
CHECK("bb07.6", designed_path, "HYPOTHESIS prune.terms equals my sequential backward elimination path in all designed datasets")
CHECK("bb07.7", acc$rsspath, "HYPOTHESIS rss.per.subset equals the RSS of my backward path (rel 1e-10), all datasets")
CHECK("bb07.8", acc$rssown, "rss.per.subset[k] equals lm.fit RSS of earth's own prune.terms row k (rel 1e-10), all datasets")
CHECK("bb07.9", acc$gcv && acc$gcvown, "gcv.per.subset[k] equals get.gcv(rss, k, penalty, n) (rel 1e-10), all datasets")
CHECK("bb07.10", acc$sel, "selected.terms equals the prune.terms row at which.min(gcv.per.subset), all datasets")
CHECK("bb07.11", d3$wm > 2 && d3$wm < K3 - 2, "weak-signal 12-column dataset D3 selects a size well inside the path")
CHECK("bb07.12", acc$prefix && nprefix == nvalid && nvalid >= 30, "CORRECTED the prefix rule reproduces prune.terms in all designed and random bases")
CHECK("bb07.13", acc$le, "earth's rss.per.subset at each size <= RSS of my backward path at that size")
CHECK("bb07.13a", nswap == nvalid, "ALTERNATIVE prefix rule with a swap (not a shift) reproduces prune.terms in all random bases")
CHECK("bb07.13b", nporp == nvalid, "ALTERNATIVE 'forward-order prefix or backward-path subset, lower RSS' reproduces prune.terms in all random bases")
CHECK("bb07.14", none_all, "pmethod='none' selects all K terms")
CHECK("bb07.15", none_same, "pmethod='none' returns prune.terms, rss.per.subset, gcv.per.subset identical to pmethod='backward'")
CHECK("bb07.16", np_full, "nprune=k does not shorten prune.terms, rss.per.subset or gcv.per.subset (all K sizes kept)")
CHECK("bb07.17", np_sel, "nprune=k selects the subset at which.min(gcv.per.subset[1:k]) of the fit without nprune")
CHECK("bb07.18", np_path, "nprune=k: prune.terms rows and rss.per.subset for sizes <= k are identical to the fit without nprune")
CHECK("bb07.19", pm_sel, "penalty=-1 with nprune=k selects exactly k terms")
CHECK("bb07.20", pm_gcv, "penalty=-1: gcv.per.subset equals rss.per.subset / n")
CHECK("bb07.21", length(fm0$selected.terms) == K3, "penalty=-1 without nprune selects all K terms")
CHECK("bb07.22", ex_brute, "pmethod='exhaustive': each prune.terms row is the brute-force lowest-RSS subset of its size")
CHECK("bb07.23", ex_diff, "pmethod='exhaustive' differs from pmethod='backward' at some size, with lower RSS there")
CHECK("bb07.24", nord_ok == nord, "CORRECTED the prefix rule also reproduces prune.terms of ordinary earth fits (full hinge basis from dirs/cuts)")
cat(checks, sep = "\n")
