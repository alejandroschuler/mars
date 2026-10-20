# One timed earth() fit (black box: only earth() is called; no earth code is
# read or printed). Usage: Rscript earth_worker.R <data.csv> <earth_args.json>
# The CSV has columns x0, x1, ..., y and w; the JSON object holds earth's own
# argument names (built by validation/harness/names_map.py) and, optionally,
# "use_weights": true. Prints one JSON object on stdout.
suppressMessages({
  library(earth)
  library(jsonlite)
})
argv <- commandArgs(trailingOnly = TRUE)
d <- read.csv(argv[1])
cfg <- fromJSON(argv[2], simplifyVector = FALSE)
use_w <- isTRUE(cfg$use_weights)
cfg$use_weights <- NULL
x <- as.matrix(d[, grep("^x", names(d)), drop = FALSE])
args <- c(list(x = x, y = d$y), cfg)
if (use_w) args$weights <- d$w
t0 <- proc.time()[["elapsed"]]
fit <- do.call(earth, args)
elapsed <- proc.time()[["elapsed"]] - t0
cat(toJSON(list(
  seconds = elapsed,
  terms = length(fit$selected.terms),
  forward_terms = nrow(fit$dirs),
  earth_version = as.character(packageVersion("earth"))
), auto_unbox = TRUE, digits = I(17)), "\n")
