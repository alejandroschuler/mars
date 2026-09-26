# bb08_prune_ties.R
# Question (a): when two terms tie exactly (in exact arithmetic) as the cheapest
# removal in earth's pruning pass, which one does earth remove: the one with the
# lower term number (entered earlier), the higher, the lower x column index, or
# the higher? Question (b): when two model sizes have exactly equal GCV, which
# size does earth select?
# Method: fixed basis as in bb07 (linpreds = TRUE, degree = 1, thresh = 0,
# nk = 2 * ncol(x) + 1). bb07 found that earth's pruning follows a "prefix rule"
# (backward elimination on a working order, with every prefix of the order
# offered as the subset of its size). Here I simulate that rule with lm.fit and
# force the removal of one or the other tied term; the member for which the
# simulation reproduces earth's prune.terms is the one earth removed. Ties among
# offered subsets (same size, RSS within 1e-9 relative) may be resolved either way.
# (a) uses two kinds of designs:
#  S: rows come in pairs (a, b, rest, y) and (b, a, rest, y), so swapping c1 and
#     c2 maps the data set to itself; values are on dyadic grids.
#  O: orthogonal 0/1 columns from a 64 x 64 Hadamard matrix with dyadic y, where
#     the two tied columns have equal coefficients.
# (b) builds a column whose coefficient in the full model is exactly zero, so that
# removing it leaves the RSS unchanged, and uses penalty = -1 (GCV = RSS/n).
suppressMessages(library(earth))
# On failure print only the condition message (no call, no traceback).
earth_bb <- function(...) tryCatch(earth(...), error = function(e) stop(paste("ERROR:", conditionMessage(e)), call. = FALSE))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")

checks <- character(0)
CHECK <- function(id, ok, desc) {
  checks <<- c(checks, sprintf("CHECK %s %s %s", id, if (isTRUE(ok)) "TRUE" else "FALSE", desc))
}
rss_of <- function(B, y, S) sum(lm.fit(B[, S, drop = FALSE], y)$residuals^2)
fit_fixed <- function(X, y, ...) earth_bb(x = X, y = y, linpreds = TRUE, degree = 1, thresh = 0, nk = 2 * ncol(X) + 1, ...)
basis_ok <- function(fit, X) {
  d <- fit$dirs
  nrow(d) == ncol(X) + 1 && all(d[d != 0] == 2) && all(rowSums(d[-1, , drop = FALSE] != 0) == 1) &&
    identical(sort(as.integer(apply(d[-1, , drop = FALSE], 1, function(r) which(r != 0)))), seq_len(ncol(X)))
}
TOL <- 1e-9

# Prefix rule of bb07; at a backward step, candidates within TOL of the lowest RSS
# are tied and `force` (a term number) is removed if it is among them. Returns all
# offered subsets per size and the tie steps.
run_prefix <- function(B, y, force) {
  K <- ncol(B); ord <- seq_len(K); offered <- vector("list", K); steps <- list()
  offer <- function() for (m in seq_len(K)) {
    S <- sort(ord[seq_len(m)]); offered[[m]][[length(offered[[m]]) + 1]] <<- list(S = S, r = rss_of(B, y, S))
  }
  offer()
  for (pos in K:2) {
    cand <- ord[2:pos]
    r <- vapply(2:pos, function(i) rss_of(B, y, ord[seq_len(pos)][-i]), 0)
    tied <- which(r <= min(r) * (1 + TOL))
    pick <- if (force %in% cand[tied]) force else cand[tied[1]]
    if (length(tied) > 1) {
      rest <- sort(r[-tied])
      steps[[length(steps) + 1]] <- list(pos = pos, terms = cand[tied], rss = r[tied],
                                         gap = if (length(rest)) rest[1] / min(r) - 1 else Inf)
    }
    i <- which(ord == pick)
    ord <- append(ord[-i], ord[i], after = pos - 1)
    offer()
  }
  list(offered = offered, steps = steps)
}
consistent <- function(run, es) {
  all(vapply(seq_along(es), function(k) {
    o <- run$offered[[k]]; rs <- vapply(o, function(x) x$r, 0); ok <- which(rs <= min(rs) * (1 + TOL))
    any(vapply(o[ok], function(x) identical(as.integer(x$S), as.integer(es[[k]])), TRUE))
  }, TRUE))
}

# ---------------------------------------------------------------- (a) designs
sym_design <- function(seed, dsc, proxy) {
  set.seed(seed); m <- 40
  g <- sample(4:14, m, TRUE) / 16; d <- sample(c(-2, -1, 1, 2), m, TRUE) * dsc
  w <- sample(0:16, m, TRUE) / 16; z <- sample(0:16, m, TRUE) / 16
  e <- sample(-8:8, m, TRUE) / 32; et <- sample(-4:4, m, TRUE) / 16
  xa <- as.vector(rbind(g + d, g - d)); xb <- as.vector(rbind(g - d, g + d))
  W <- rep(w, each = 2); Z <- rep(z, each = 2)
  X <- cbind(c1 = pmax(xa - 0.25, 0), c2 = pmax(xb - 0.25, 0), lw = W, hw = pmax(W - 0.5, 0),
             lz = Z, hz = pmax(0.625 - Z, 0), pwz = W * Z)
  if (proxy) X <- cbind(X, t = W + Z + rep(et, each = 2))
  y <- 2 * (X[, "c1"] + X[, "c2"]) + 1.5 * W + 1.5 * Z - 2 * X[, "hz"] + X[, "pwz"] + rep(e, each = 2)
  list(X = X, y = y)
}
H <- matrix(1, 1, 1); for (i in 1:6) H <- rbind(cbind(H, H), cbind(H, -H))  # 64 x 64 Sylvester Hadamard
a01 <- function(j) pmax(H[, j], 0)
orth_design <- function(pair, others, cp, co) {
  X <- cbind(c1 = a01(pair[1]), c2 = a01(pair[2]), sapply(others, a01))
  colnames(X)[-(1:2)] <- paste0("q", others)
  y <- cp * (X[, 1] + X[, 2]) + drop(X[, -(1:2)] %*% co) + 0.25 * H[, 8]
  list(X = X, y = y)
}
reorder <- function(dd, how) {
  cn <- colnames(dd$X); rest <- setdiff(cn, c("c1", "c2"))
  oc <- switch(how, "pair first" = c("c1", "c2", rest), "pair last" = c(rest, "c2", "c1"),
               "pair split" = c("c2", rest, "c1"), "pair reversed" = c("c2", "c1", rest), "all reversed" = rev(cn))
  list(X = dd$X[, oc], y = dd$y)
}
runs <- list()
add <- function(label, dd) runs[[length(runs) + 1]] <<- list(label = label, X = dd$X, y = dd$y)
for (cfg in list(list(801, 1 / 64, FALSE), list(802, 1 / 32, TRUE), list(803, 1 / 64, TRUE))) {
  dd <- sym_design(cfg[[1]], cfg[[2]], cfg[[3]])
  for (how in c("pair first", "pair last", "pair split")) add(sprintf("S%d %s", cfg[[1]], how), reorder(dd, how))
}
dd <- sym_design(801, 1 / 64, FALSE); sw <- as.vector(rbind(seq(2, 80, 2), seq(1, 79, 2)))
add("S801 pair first, rows swapped in pairs", list(X = dd$X[sw, ], y = dd$y[sw]))
add("S801 pair first, rows reversed", list(X = dd$X[80:1, ], y = dd$y[80:1]))
o1 <- orth_design(c(2, 3), 4:7, 1, c(3, 2.5, 2, 1.5))
for (how in c("pair first", "pair last", "pair reversed", "all reversed")) add(paste("O1", how), reorder(o1, how))
o2 <- orth_design(c(2, 3), 4:7, 2, c(3, 2.5, 1.5, 1))
for (how in c("pair first", "pair reversed")) add(paste("O2", how), reorder(o2, how))
o3 <- orth_design(c(5, 4), c(2, 3, 6, 7), 1, c(3, 2.5, 2, 1.5))
for (how in c("pair first", "pair reversed")) add(paste("O3", how), reorder(o3, how))

cat("(a) tie between two terms as the cheapest removal\n")
cat("run | term (col) of c1, c2 | tie at pos: lm.fit RSS without c1, without c2, bitwise equal, gap to next | earth removed\n")
res <- list()
for (R in runs) {
  X <- R$X; y <- R$y; fit <- fit_fixed(X, y); K <- nrow(fit$dirs); ok <- basis_ok(fit, X)
  ct <- apply(fit$dirs[-1, , drop = FALSE], 1, function(r) which(r == 2)); B <- cbind(1, X[, ct])
  tn <- colnames(X)[ct]; t1 <- which(tn == "c1") + 1L; t2 <- which(tn == "c2") + 1L
  k1 <- which(colnames(X) == "c1"); k2 <- which(colnames(X) == "c2")
  es <- lapply(seq_len(K), function(k) sort(as.integer(fit$prune.terms[k, 1:k])))
  r1 <- run_prefix(B, y, t1); r2 <- run_prefix(B, y, t2)
  st <- Filter(function(s) setequal(s$terms, c(t1, t2)), r1$steps)
  valid <- ok && length(st) == 1 && length(r1$steps) == 1 && st[[1]]$gap > 1e-6
  c1ok <- consistent(r1, es); c2ok <- consistent(r2, es)
  removed <- if (c1ok && !c2ok) "c1" else if (c2ok && !c1ok) "c2" else if (c1ok) "ambiguous" else "neither"
  if (length(st)) {
    s <- st[[1]]; rw1 <- s$rss[s$terms == t1]; rw2 <- s$rss[s$terms == t2]
    cat(sprintf("%-38s | c1 %d (%d), c2 %d (%d) | pos %d: %.17g %.17g %s gap %.1e | %s\n", R$label, t1, k1, t2, k2,
                s$pos, rw1, rw2, rw1 == rw2, s$gap, removed))
  } else {
    rw1 <- rw2 <- NA
    cat(sprintf("%-38s | c1 %d (%d), c2 %d (%d) | no pair tie in the backward steps | %s\n", R$label, t1, k1, t2, k2, removed))
  }
  if (removed %in% c("c1", "c2")) {
    tr <- if (removed == "c1") t1 else t2; to <- if (removed == "c1") t2 else t1
    kr <- if (removed == "c1") k1 else k2; ko <- if (removed == "c1") k2 else k1
    rr <- if (removed == "c1") rw1 else rw2; ro <- if (removed == "c1") rw2 else rw1
  } else tr <- to <- kr <- ko <- rr <- ro <- NA
  res[[length(res) + 1]] <- list(label = R$label, valid = valid, removed = removed, higher_term = tr > to,
                                 lower_col = kr < ko, lower_lmfit = rr < ro, bitwise = isTRUE(rw1 == rw2))
}
det <- Filter(function(r) r$removed %in% c("c1", "c2"), res)
cat(sprintf("determined in %d of %d runs; removed member had the higher term number in %d, the lower column index in %d, the lower lm.fit RSS in %d\n",
            length(det), length(res), sum(sapply(det, `[[`, "higher_term")), sum(sapply(det, `[[`, "lower_col")),
            sum(sapply(det, `[[`, "lower_lmfit"))))
lab <- sapply(res, `[[`, "label"); rem <- sapply(res, `[[`, "removed")
same_col_S <- all(vapply(c("S801", "S802", "S803"), function(p) length(unique(rem[startsWith(lab, paste(p, "pair")) & !grepl(",", lab)])) == 1, TRUE))
swap_flips <- rem[lab == "S801 pair first"] %in% c("c1", "c2") && rem[lab == "S801 pair first, rows swapped in pairs"] %in% c("c1", "c2") &&
  rem[lab == "S801 pair first"] != rem[lab == "S801 pair first, rows swapped in pairs"]

# ---------------------------------------------------------------- (b) GCV ties
cat("(b) exact ties between model sizes in gcv.per.subset, penalty = -1\n")
gcv_runs <- list()
X1 <- cbind(C = a01(2) + a01(3) + 0.5 * a01(7), a2 = a01(2), a3 = a01(3), a4 = a01(4), a5 = a01(5), a6 = a01(6))
y1 <- 3 * a01(2) + 2 * a01(3) + a01(4) + 0.5 * a01(5) + 0.25 * a01(6) + 0.125 * H[, 8]
gcv_runs[["G1 0/1 Hadamard, n=64"]] <- list(X = X1, y = y1)
X2 <- cbind(C = H[, 2] + H[, 3] + 0.5 * H[, 7], h2 = H[, 2], h3 = H[, 3], h4 = H[, 4], h5 = H[, 5], h6 = H[, 6])
y2 <- 3 * H[, 2] + 2 * H[, 3] + H[, 4] + 0.5 * H[, 5] + 0.25 * H[, 6] + 0.125 * H[, 8]
gcv_runs[["G2 +-1 Hadamard, n=64"]] <- list(X = X2, y = y2)
set.seed(805); n3 <- 100; u <- runif(n3); v <- runif(n3); wv <- runif(n3)
A <- pmax(u - 0.3, 0); Bh <- pmax(0.7 - u, 0); L <- v; Hv <- pmax(v - 0.5, 0); P <- u * wv
yy <- 2 * A + 3 * Bh + L + 2 * Hv + P + rnorm(n3, sd = 0.3)
D <- qr.resid(qr(cbind(1, A, Bh, L, Hv, P, yy)), rnorm(n3))  # orthogonal to the basis and to y
gcv_runs[["G3 random hinges, n=100"]] <- list(X = cbind(C = A + Bh + 0.3 * D / sqrt(mean(D^2)), A = A, Bh = Bh, L = L, Hv = Hv, P = P), y = yy)
g_tie <- TRUE; g_sel <- TRUE; g_in <- TRUE
for (nm in names(gcv_runs)) {
  X <- gcv_runs[[nm]]$X; y <- gcv_runs[[nm]]$y
  tr <- capture.output(fit <- fit_fixed(X, y, penalty = -1, trace = 2))
  K <- nrow(fit$dirs); ok <- basis_ok(fit, X)
  tn <- colnames(X)[apply(fit$dirs[-1, , drop = FALSE], 1, function(r) which(r == 2))]
  gv <- fit$gcv.per.subset; rs <- fit$rss.per.subset; tiedk <- which(gv == min(gv))
  zc <- which(tn == "C") + 1L
  # the zero-coefficient column: its lm.fit coefficient in the full model, and the lm.fit RSS change on removing it
  Bf <- cbind(1, X[, match(tn, colnames(X))]); cf <- lm.fit(Bf, y)$coefficients[zc]
  drss <- rss_of(Bf, y, setdiff(seq_len(K), zc)) - rss_of(Bf, y, seq_len(K))
  cat(sprintf("%s: basis_ok=%s terms %s; C is term %d; lm.fit coef of C %.2e, RSS change on removing C %.2e\n",
              nm, ok, paste(tn, collapse = " "), zc, cf, drss))
  cat(sprintf("  rss.per.subset[%d:%d] = %s\n", K - 2, K, paste(sprintf("%.17g", rs[(K - 2):K]), collapse = " ")))
  cat(sprintf("  gcv.per.subset[%d:%d] = %s\n", K - 2, K, paste(sprintf("%.17g", gv[(K - 2):K]), collapse = " ")))
  cat(sprintf("  sizes with gcv == min(gcv): %s | which.min: %d | selected size: %d | row %d omits C: %s\n",
              paste(tiedk, collapse = " "), which.min(gv), length(fit$selected.terms), K - 1, !(zc %in% fit$prune.terms[K - 1, ])))
  fl <- grep(sprintf("^%d ", 2 * zc - 2), tr, value = TRUE)
  if (length(fl)) cat("  forward trace row of C:", gsub("\\s+", " ", fl[1]), "\n")
  g_tie <- g_tie && ok && length(tiedk) >= 2
  g_sel <- g_sel && length(fit$selected.terms) == min(tiedk) && which.min(gv) == min(tiedk)
  g_in <- g_in && ok
}

CHECK("bb08.1", all(sapply(res, `[[`, "valid")), "(a) every run is a valid tie design: one backward step where c1 and c2 are the two cheapest removals, equal to 1e-9 in lm.fit, next candidate >1e-6 higher")
CHECK("bb08.2", all(sapply(res, `[[`, "bitwise")), "(a) HYPOTHESIS the two tied lm.fit RSS values are bitwise equal in every run")
CHECK("bb08.3", length(det) == length(res), "(a) HYPOTHESIS the member earth removed is determined (only one forced choice reproduces prune.terms) in every run")
CHECK("bb08.4", length(det) > 0 && all(sapply(det, `[[`, "higher_term")), "(a) HYPOTHESIS earth removes the tied term with the higher term number (entered later)")
CHECK("bb08.5", length(det) > 0 && !any(sapply(det, `[[`, "higher_term")), "(a) HYPOTHESIS earth removes the tied term with the lower term number (entered earlier)")
CHECK("bb08.6", length(det) > 0 && all(sapply(det, `[[`, "lower_col")), "(a) HYPOTHESIS earth removes the tied term with the lower x column index")
CHECK("bb08.7", length(det) > 0 && !any(sapply(det, `[[`, "lower_col")), "(a) HYPOTHESIS earth removes the tied term with the higher x column index")
CHECK("bb08.8", length(det) > 0 && all(sapply(det, `[[`, "lower_lmfit")), "(a) HYPOTHESIS earth removes the tied term whose removal has the lower lm.fit RSS (rounding agrees with lm.fit)")
CHECK("bb08.9", same_col_S, "(a) in each S design, earth removes the same data column in all three column orders")
CHECK("bb08.10", swap_flips, "(a) swapping the two rows of every pair (swaps the contents of c1 and c2) flips which named column earth removes")
CHECK("bb08.11", g_in, "(b) the forward pass entered the zero-coefficient column C in all three constructions")
CHECK("bb08.12", g_tie, "(b) an exact tie was built: gcv.per.subset has two or more bitwise-equal minima in all three constructions")
CHECK("bb08.13", g_sel, "(b) with tied minima earth selects the smallest tied size, which is which.min(gcv.per.subset)")
cat(checks, sep = "\n")
