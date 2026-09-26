# bb18: exact ties between candidates in earth's forward pass.
#
# earth searches the parents in the order of its queue (bb16), the covariates
# of a parent in index order, the linear candidate of a pair search before its
# knots, and the knots from the largest down (bb05, bb06). Hypothesis: a
# candidate replaces the best so far only when its RSS reduction is strictly
# larger, so an exact tie goes to the one found first: the lower covariate,
# and within a covariate the larger knot.
# A. x2 = x1 (a duplicate) and x2 = 1 - x1 (a mirror image): a pair on x2 at
#    knot 1 - t spans the same columns as a pair on x1 at knot t.
# B. y symmetric in x = 1..20 about 10.5: the knots t and 21 - t give the
#    same RSS in exact arithmetic. Which does earth choose, and are the two
#    traced reductions equal?
suppressMessages(library(earth))
cat(R.version.string, "| earth", as.character(packageVersion("earth")), "\n")
quiet <- function(expr) tryCatch(expr, error = function(e) paste("ERROR:", conditionMessage(e)))
num <- function(pat, s) as.numeric(sub(pat, "\\1", s))

# Part A
resA <- NULL
for (i in 1:20) {
  set.seed(1800 + i); n <- 60; x1 <- runif(n); y <- 3 * pmax(x1 - 0.4, 0) + sin(5 * x1) + 0.1 * rnorm(n)
  for (kind in c("duplicate", "mirror")) {
    x2 <- if (kind == "duplicate") x1 else 1 - x1
    f <- quiet(earth(cbind(x1, x2), y, degree = 1, nk = 7, minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0))
    used <- unique(which(f$dirs[-1, , drop = FALSE] != 0, arr.ind = TRUE)[, 2])
    resA <- rbind(resA, data.frame(kind = kind, seed = i, first_step_x1 = f$dirs[2, 1] != 0, used = paste(sort(used), collapse = ",")))
  }
}
print(table(kind = resA$kind, first_step_uses = ifelse(resA$first_step_x1, "x1", "x2")))

# Part B: symmetric data, one covariate, one step
sym_pick <- function(n, noise_seed) {
  x <- as.numeric(1:n); set.seed(noise_seed); e <- rnorm(ceiling(n / 2))
  half <- sin(x[1:ceiling(n / 2)] / 3) + 0.3 * e
  y <- c(half, rev(half))[1:n]              # y[i] = y[n + 1 - i]
  set.seed(noise_seed + 1); o <- sample(n)  # rows in a shuffled order
  out <- capture.output(f <- quiet(earth(matrix(x[o], ncol = 1, dimnames = list(NULL, "x")), y[o], trace = 9, nk = 3,
    minspan = 1, endspan = 1, Auto.linpreds = FALSE, pmethod = "none", thresh = 0)))
  kn <- grep("--FindKnot--Case .*RssWithKnot", out, value = TRUE)
  cut <- num(".*Cut +([-0-9.e]+).*", kn); rd <- num(".*RssDelta +([-0-9.e]+) Cut.*", kn)
  valid <- grepl("TolG 1", kn) & grepl("MaxG 1", kn)
  best <- max(rd[valid]); tied <- sort(cut[valid & rd == best])
  chosen <- f$cuts[2, 1]
  data.frame(n = n, seed = noise_seed, chosen = chosen, printed_ties = paste(tied, collapse = ","),
    mirror_of_chosen = n + 1 - chosen, mirror_evaluated = (n + 1 - chosen) %in% cut)
}
resB <- do.call(rbind, lapply(1:12, function(k) sym_pick(sample(c(20, 21, 30, 31), 1), 100 + k)))
print(resB, row.names = FALSE)
larger <- with(resB, chosen > mirror_of_chosen | chosen == mirror_of_chosen)

cat(sprintf("CHECK bb18.1 %s with an exact duplicate x2 = x1, the first step always uses x1, the lower index (20 of 20)\n",
  all(resA$first_step_x1[resA$kind == "duplicate"])))
cat(sprintf("CHECK bb18.2 %s HYPOTHESIS with a mirror image x2 = 1 - x1, the first step always uses x1 (%d of 20)\n",
  all(resA$first_step_x1[resA$kind == "mirror"]), sum(resA$first_step_x1[resA$kind == "mirror"])))
cat(sprintf("CHECK bb18.3 %s HYPOTHESIS in the symmetric designs the chosen knot is the larger of the two mirror knots (%d of %d)\n",
  all(larger), sum(larger), nrow(resB)))
nm <- sum(resA$first_step_x1[resA$kind == "mirror"])
cat(sprintf("CHECK bb18.4 %s beyond bitwise-identical columns, rounding decides earth's exact ties: the mirror designs use x1 in %d and x2 in %d of 20, and the symmetric designs take the smaller knot in %d of %d\n",
  nm > 0 && nm < 20 && sum(!larger) > 0 && sum(larger) > 0, nm, 20 - nm, sum(!larger), nrow(resB)))
