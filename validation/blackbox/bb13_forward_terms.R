# bb13: which terms a forward step adds.
#
# A. A pair step adds the +1 hinge, then the -1 hinge, with the same knot and
#    the parent's factors.
# B. A single-hinge step (bb14 on the part 1 branch says when a search is one)
#    adds b*(x - t)+ (dirs +1), also for knots near the low end.
# C. The linear option: with Auto.linpreds = TRUE a step whose best candidate
#    is the linear option adds b*x (dirs 2) as one term; with FALSE it adds
#    one hinge b*(x - m)+ (dirs +1). Which minimum m: over the parent's active
#    cases or over all cases?
# The step lines of trace 2 give, for each step, the terms added (the Terms
# column counts dirs rows), the covariate (Pred) and the Cut ("<" marks the
# linear option); Par counts slots, which the script maps to dirs rows.
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
traced_out <- function(expr) {  # sink() is much faster than capture.output() for long traces
  tf <- tempfile(); zz <- file(tf, open = "wt"); sink(zz)
  tryCatch(force(expr), error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
  sink(); close(zz); out <- readLines(tf); unlink(tf); out
}
parse_steps <- function(out) {
  i <- grep("^[0-9]+ +[-0-9.]+ +[-0-9.]+ +[-0-9.e]+ +[0-9]+ +[xz][0-9]+ ", out)
  lapply(i, function(ii) {
    tk <- strsplit(trimws(sub(" final.*", "", out[ii])), " +")[[1]]
    rest <- as.integer(tk[8:length(tk)]); deg <- tail(rest, 1)
    list(K = as.integer(tk[1]), pred = as.integer(tk[5]), linear = grepl("<$", tk[7]),
      terms = if (deg >= 2) rest[seq_len(length(rest) - 2)] else rest[seq_len(length(rest) - 1)],
      par_slot = if (deg >= 2) rest[length(rest) - 1] else 1L)
  })
}
slot_map <- function(st) {
  m <- c(`1` = 1L); row <- 1L
  for (s in st) for (q in seq_along(s$terms)) { row <- row + 1L; m[as.character(s$K + q - 1)] <- row }
  m
}

set.seed(1); n <- 200; X <- matrix(runif(n * 5), n, 5); colnames(X) <- paste0("x", 1:5)
yf <- 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5] + rnorm(n)
set.seed(2); X2 <- matrix(runif(300 * 3), 300, 3); colnames(X2) <- paste0("x", 1:3)
y2 <- sin(8 * X2[, 1]) + 2 * pmax(X2[, 1] - 0.3, 0) * X2[, 2] + 0.5 * sin(6 * X2[, 3]) + 0.05 * rnorm(300)
cfgs <- list(list("Friedman #1, degree 1", X, yf, 1, 31), list("Friedman #1, degree 2", X, yf, 2, 31),
  list("three x, degree 2", X2, y2, 2, 31), list("three x, degree 3", X2, y2, 3, 25))
okA <- TRUE; okB <- TRUE; npairs <- 0; nsingles <- 0
for (cf in cfgs) {
  out <- traced_out(f <- quiet(earth(cf[[2]], cf[[3]], degree = cf[[4]], nk = cf[[5]], trace = 2, pmethod = "none",
    thresh = 0, fast.k = 0, Auto.linpreds = FALSE, minspan = 1, endspan = 1)))
  dirs <- f$dirs; cuts <- f$cuts; st <- parse_steps(out); smap <- slot_map(st)
  for (s in st) {
    pr <- smap[as.character(s$par_slot)]; j <- s$pred
    if (length(s$terms) == 2) {
      a <- s$terms[1]; b <- s$terms[2]; npairs <- npairs + 1
      okA <- okA && dirs[a, j] == 1 && dirs[b, j] == -1 && cuts[a, j] == cuts[b, j] &&
        all(dirs[a, -j] == dirs[pr, -j]) && all(dirs[b, -j] == dirs[pr, -j])
    } else if (!s$linear) {
      nsingles <- nsingles + 1
      okB <- okB && dirs[s$terms[1], j] == 1 && all(dirs[s$terms[1], -j] == dirs[pr, -j])
    }
  }
  cat(sprintf("%-24s %2d terms, %2d steps\n", cf[[1]], nrow(dirs), length(st)))
}
cat(sprintf("parts A and B: %d pair steps and %d single-hinge steps\n", npairs, nsingles))

# B, direction, in a one-covariate fit where every step after the first is a single hinge
n <- 400; x <- (1:n) / n; set.seed(3); x <- sample(x)
fs <- quiet(earth(matrix(x, ncol = 1), sin(10 * x) + 0.05 * rnorm(n), nk = 21, minspan = 1, endspan = 1,
  Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("one covariate, dirs by term:", fs$dirs[, 1], "\n")
fs2 <- quiet(earth(matrix(x, ncol = 1), 5 * pmax(0.15 - x, 0) + 5 * pmax(0.08 - x, 0) + 0.01 * rnorm(n), nk = 7,
  minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("knots near the low end, dirs by term:", fs2$dirs[, 1], " cuts:", round(fs2$cuts[, 1], 4), "\n")
okB <- okB && all(fs$dirs[2:3, 1] == c(1, -1)) && all(fs$dirs[-(1:3), 1] == 1) && all(fs2$dirs[-(1:3), 1] == 1)

# C: the linear option. x1 is binary, so every knot of x1 is rejected.
set.seed(3); n <- 100
x1 <- sample(c(0, 1), n, replace = TRUE) + 0; x2 <- runif(n); x3 <- sample(1:3, n, replace = TRUE) + 0
yl <- 2 * x1 + 3 * pmax(x2 - 0.4, 0) + 0.5 * x3 + 0.1 * rnorm(n)
ft <- quiet(earth(cbind(x1, x2, x3), yl, degree = 2, nk = 11, Auto.linpreds = TRUE, pmethod = "none", thresh = 0))
ff <- quiet(earth(cbind(x1, x2, x3), yl, degree = 2, nk = 11, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
cat("Auto.linpreds TRUE, term 2:", rownames(ft$dirs)[2], "| dirs", ft$dirs[2, ], "\n")
cat("Auto.linpreds FALSE, term 2:", rownames(ff$dirs)[2], "| dirs", ff$dirs[2, ], "| cuts", ff$cuts[2, ], "\n")
linT <- ft$dirs[2, 1] == 2 && all(ft$dirs[2, -1] == 0)
linF <- ff$dirs[2, 1] == 1 && ff$cuts[2, 1] == min(x1)
same_rest <- identical(ft$dirs[-2, ], ff$dirs[-2, ]) && identical(ft$cuts[-2, ], ff$cuts[-2, ])
# a hinge parent whose active cases do not reach the global minimum of a 3-level covariate
set.seed(4); n <- 300; z1 <- runif(n); z2 <- ifelse(z1 > 0.5, sample(1:2, n, TRUE), sample(0:2, n, TRUE)) + 0
yz <- 4 * pmax(z1 - 0.5, 0) + 3 * pmax(z1 - 0.5, 0) * z2 + 0.01 * rnorm(n)
fz <- quiet(earth(cbind(z1, z2), yz, degree = 2, nk = 7, Auto.linpreds = FALSE, pmethod = "none", thresh = 0, minspan = 1, endspan = 1))
cat("hinge parent h(z1 - c), covariate z2 with values 0..2; its active cases have z2 in 1..2:\n")
print(cbind(fz$dirs, round(fz$cuts, 6)))
kz <- which(fz$dirs[, 1] != 0 & fz$dirs[, 2] == 1)
cut_min <- if (length(kz)) fz$cuts[kz[1], 2] else NA
c_par <- if (length(kz)) fz$cuts[kz[1], 1] else NA        # the knot of the parent h(z1 - c)
act_min <- min(z2[z1 > c_par]); glob_min <- min(z2)

cat(sprintf("CHECK bb13.1 %s a pair step adds the +1 hinge, then the -1 hinge, with the same knot and the parent's factors (%d pair steps)\n", okA, npairs))
cat(sprintf("CHECK bb13.3 %s a single-hinge step adds b*(x - t)+ (dirs +1), also for knots near the low end (%d single-hinge steps)\n", okB, nsingles))
cat(sprintf("CHECK bb13.4 %s with Auto.linpreds = TRUE the linear option adds b*x as one term (dirs 2)\n", linT))
cat(sprintf("CHECK bb13.5 %s with Auto.linpreds = FALSE it adds the one hinge (x - min x)+ (dirs +1, cut = min x), and the other terms do not change\n", linF && same_rest))
cat(sprintf("CHECK bb13.6 %s HYPOTHESIS for a hinge parent the Auto.linpreds = FALSE hinge sits at the smallest value among the active cases (%g)\n",
  isTRUE(cut_min == act_min), act_min))
cat(sprintf("CHECK bb13.7 %s it sits at the smallest value of the covariate over all cases (cut %g, global minimum %g)\n",
  isTRUE(cut_min == glob_min) && glob_min != act_min, cut_min, glob_min))
