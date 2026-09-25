
suppressMessages(library(earth))
a <- commandArgs(trailingOnly=TRUE)
d <- read.csv(a[1]); x <- as.matrix(d[, grep('^x', names(d))]); y <- d$y
deg <- as.integer(a[2])
t <- sapply(1:3, function(i) system.time(f <<- earth(x, y, degree=deg))[['elapsed']])
cat(min(t), length(f$selected.terms), nrow(f$dirs), '\n')
