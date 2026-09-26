# bb13: which terms a forward step adds.
#
# A. A pair is stored in dirs as the +1 hinge, then the -1 hinge, with the
#    same parent factors and the same knot.
# B. A (parent, covariate) search is a single-hinge search (iNewCol = K, the
#    term number searched; bb11.5) exactly when an existing term is the
#    parent times a factor in that covariate. Otherwise it is a pair search
#    (iNewCol = K + 1). A single-hinge step adds b*(x - t)+ (dirs +1).
# C. The linear option: with Auto.linpreds = TRUE a step whose best candidate
#    is the linear option adds b*x (dirs 2), one term; with FALSE it adds the
#    hinge b*(x - m)+ (dirs +1) with m a minimum of x, one term. Which minimum
#    (over all cases or over the parent's active cases) is read from cuts.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
num <- function(pat, s) as.numeric(sub(pat, "\\1", s))

# does term k equal parent row pr times one factor in covariate j?
is_child <- function(dirs, cuts, k, pr, j) {
  d <- dirs[k, ]; dp <- dirs[pr, ]
  if (d[j] == 0 || dp[j] != 0) return(FALSE)
  others <- setdiff(seq_along(d), j)
  all(d[others] == dp[others]) && all(cuts[k, others][d[others] %in% c(-1, 1)] == cuts[pr, others][d[others] %in% c(-1, 1)])
}

# Step lines (trace >= 2): K GRSq RSq DeltaRSq Pred PredName Cut Terms [Par] Deg.
# Terms uses the dirs numbering; Par, printed for degree >= 2, uses earth's
# internal numbering, where every step takes two slots (K and K + 1).
parse_steps <- function(out) {
  i <- grep("^[0-9]+ +[-0-9.]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ +x[0-9]+ ", out)
  lapply(i, function(ii) {
    tk <- strsplit(trimws(sub(" final.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    terms <- if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)]
    list(K = as.integer(tk[1]), pred = as.integer(tk[5]), linear = grepl("<$", tk[7]), terms = terms,
      par_internal = if (deg >= 2) rest[length(rest) - 1] else 1L)
  })
}
# internal slot number -> dirs row
slot_map <- function(st) {
  m <- c(`1` = 1L); row <- 1L
  for (s in st) { for (q in seq_along(s$terms)) { row <- row + 1L; m[as.character(s$K + q - 1)] <- row } }
  m
}
set.seed(1); n <- 200; X <- matrix(runif(n * 5), n, 5); colnames(X) <- paste0("x", 1:5)
yf <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n)
set.seed(2); X2 <- matrix(runif(300 * 3), 300, 3); colnames(X2) <- paste0("x", 1:3)
y2 <- sin(8 * X2[, 1]) + 2 * pmax(X2[, 1] - 0.3, 0) * X2[, 2] + 0.5 * sin(6 * X2[, 3]) + 0.05 * rnorm(300)
cfgs <- list(
  list("Friedman #1, degree 1", X, yf, degree = 1, nk = 31),
  list("Friedman #1, degree 2", X, yf, degree = 2, nk = 31),
  list("three x, degree 2", X2, y2, degree = 2, nk = 31),
  list("three x, degree 3", X2, y2, degree = 3, nk = 25))
okA <- TRUE; okB <- 0; totB <- 0; single_plus <- TRUE; npairs <- 0; nsingles <- 0
for (cf in cfgs) {
  lab <- cf[[1]]; Xc <- cf[[2]]; yc <- cf[[3]]; args <- cf[-(1:3)]
  out <- capture.output(f <- quiet(do.call(earth, c(list(x = Xc, y = yc, trace = 9, pmethod = "none", thresh = 0,
    fast.k = 0, Auto.linpreds = FALSE, minspan = 1, endspan = 1), args))))
  dirs <- f$dirs; cuts <- f$cuts; st <- parse_steps(out); smap <- slot_map(st)
  rows_before <- 1L + cumsum(c(0L, sapply(st, function(s) length(s$terms))))
  for (k in seq_along(st)) {
    s <- st[[k]]; pr <- smap[as.character(s$par_internal)]; j <- s$pred
    if (length(s$terms) == 2) {
      a <- s$terms[1]; b <- s$terms[2]; npairs <- npairs + 1
      okA <- okA && dirs[a, j] == 1 && dirs[b, j] == -1 && cuts[a, j] == cuts[b, j] &&
        all(dirs[a, -j] == dirs[pr, -j]) && all(dirs[b, -j] == dirs[pr, -j])
    } else if (!s$linear) {
      nsingles <- nsingles + 1
      single_plus <- single_plus && dirs[s$terms[1], j] == 1 && all(dirs[s$terms[1], -j] == dirs[pr, -j])
    }
  }
  # every knot search of the step: single-hinge search (iNewCol == K) vs the structural rule
  sl <- grep("Searching for new term", out); Ks <- num(".*new term ([0-9]+).*", out[sl])
  b <- grep("--FindKnotBegin--", out); par <- grep("^\\|Parent", out)
  for (bb in b) {
    si <- max(which(sl < bb)); Kk <- Ks[si]
    P <- num("^\\|Parent +([0-9]+).*", out[max(par[par < bb])]); j <- num(".*iPred ([0-9]+).*", out[bb])
    pr <- smap[as.character(P)]; if (is.na(pr)) next
    older <- setdiff(seq_len(rows_before[si]), pr)
    structural <- any(sapply(older, function(k) is_child(dirs, cuts, k, pr, j)))
    single <- num(".*iNewCol ([0-9]+).*", out[bb]) == Kk
    totB <- totB + 1; okB <- okB + (single == structural)
  }
  cat(sprintf("%-24s %2d terms, %2d steps\n", lab, nrow(dirs), length(st)))
}
cat(sprintf("parts A and B: %d pair steps, %d single-hinge steps, %d knot searches; iNewCol rule matches the structural rule in %d\n",
  npairs, nsingles, totB, okB))

# B, direction: in a one-covariate fit every step after the first is a single hinge
n <- 400; x <- (1:n) / n; set.seed(3); x <- sample(x)
ys <- sin(10 * x) + 0.05 * rnorm(n)
fs <- quiet(earth(matrix(x, ncol = 1), ys, nk = 21, minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("one covariate, dirs by term:", fs$dirs[, 1], "\n")
single_plus <- single_plus && all(fs$dirs[-(1:3), 1] == 1) && all(fs$dirs[2:3, 1] == c(1, -1))
# ... and with the knot near the low end, where (t - x)+ has little support
ys2 <- 5 * pmax(0.15 - x, 0) + 5 * pmax(0.08 - x, 0) + 0.01 * rnorm(n)
fs2 <- quiet(earth(matrix(x, ncol = 1), ys2, nk = 7, minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("low-end knots, dirs by term:", fs2$dirs[, 1], " cuts:", round(fs2$cuts[, 1], 4), "\n")
single_plus <- single_plus && all(fs2$dirs[-(1:3), 1] == 1)

# C: the linear option. x1 is binary, so every knot of x1 is rejected.
set.seed(3); n <- 100
x1 <- sample(c(0, 1), n, replace = TRUE) + 0; x2 <- runif(n); x3 <- sample(1:3, n, replace = TRUE) + 0
yl <- 2 * x1 + 3 * pmax(x2 - 0.4, 0) + 0.5 * x3 + 0.1 * rnorm(n)
ft <- quiet(earth(cbind(x1, x2, x3), yl, degree = 2, nk = 11, Auto.linpreds = TRUE, pmethod = "none", thresh = 0))
ff <- quiet(earth(cbind(x1, x2, x3), yl, degree = 2, nk = 11, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("Auto.linpreds TRUE, term 2:", rownames(ft$dirs)[2], "dirs", ft$dirs[2, ], "cuts", ft$cuts[2, ], "\n")
cat("Auto.linpreds FALSE, term 2:", rownames(ff$dirs)[2], "dirs", ff$dirs[2, ], "cuts", ff$cuts[2, ], "\n")
linT <- ft$dirs[2, 1] == 2 && all(ft$dirs[2, -1] == 0)
linF <- ff$dirs[2, 1] == 1 && ff$cuts[2, 1] == min(x1)
same_rest <- identical(ft$dirs[-2, ], ff$dirs[-2, ]) && identical(ft$cuts[-2, ], ff$cuts[-2, ])
# a hinge parent whose active cases do not reach the global minimum of a 3-level covariate
set.seed(4); n <- 300; z1 <- runif(n); z2 <- ifelse(z1 > 0.5, sample(1:2, n, TRUE), sample(0:2, n, TRUE)) + 0
yz <- 4 * pmax(z1 - 0.5, 0) + 3 * pmax(z1 - 0.5, 0) * z2 + 0.01 * rnorm(n)
fz <- quiet(earth(cbind(z1, z2), yz, degree = 2, nk = 7, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, minspan = 1, endspan = 1))
cat("hinge parent with a 3-level covariate (active cases have z2 in 1..2, global minimum 0):\n"); print(cbind(fz$dirs, fz$cuts))
kz <- which(fz$dirs[, 1] != 0 & fz$dirs[, 2] == 1)
cut_min <- if (length(kz)) fz$cuts[kz[1], 2] else NA

cat(sprintf("CHECK bb13.1 %s a pair step adds the +1 hinge, then the -1 hinge, with the same knot and the parent's factors (%d pair steps)\n", okA, npairs))
cat(sprintf("CHECK bb13.2 %s a search is a single-hinge search exactly when an existing term is its parent times a factor in its covariate (%d searches)\n", okB == totB, totB))
cat(sprintf("CHECK bb13.3 %s a single-hinge step adds b*(x - t)+ (dirs +1), also for knots near the low end\n", single_plus))
cat(sprintf("CHECK bb13.4 %s with Auto.linpreds = TRUE the linear option adds b*x as one term (dirs 2)\n", linT))
cat(sprintf("CHECK bb13.5 %s with Auto.linpreds = FALSE it adds the one hinge (x - min x)+ (dirs +1, cut = min x), and the other terms do not change\n", linF && same_rest))
cat(sprintf("CHECK bb13.6 %s for a hinge parent the Auto.linpreds = FALSE hinge sits at the smallest value among the active cases (cut %s, active minimum 1, global minimum 0)\n",
  isTRUE(cut_min == 1), cut_min))
