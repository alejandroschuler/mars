# bb28: does R read back a double exactly from a 17-significant-digit decimal
# string, and from a hexadecimal string? The harness (T02) moves the data from
# Python to R; exact knot comparisons need the same doubles on both sides.
# Base R only; earth is not involved.
cat(R.version.string, "\n")
set.seed(28); x <- c(runif(2500), rnorm(2500) * 10^runif(2500, -8, 8))
dec <- as.numeric(sprintf("%.17g", x))
hex <- as.numeric(sprintf("%a", x))
n_dec <- sum(dec != x); n_hex <- sum(hex != x)
cat(sprintf("of %d random doubles: %d differ after a 17-digit decimal round trip, %d after a hexadecimal round trip\n", length(x), n_dec, n_hex))
if (n_dec > 0) { i <- which(dec != x)[1]; cat(sprintf("example: %s is read back as %s (%.3g ulp)\n", sprintf("%.17g", x[i]), sprintf("%.17g", dec[i]),
  (dec[i] - x[i]) / (2^(floor(log2(abs(x[i]))) - 52)))) }
cat(sprintf("CHECK bb28.1 %s in this R, some doubles do not survive a 17-digit decimal round trip (%d of %d)\n", n_dec > 0, n_dec, length(x)))
cat(sprintf("CHECK bb28.2 %s every double survives a hexadecimal round trip (sprintf('%%a') then as.numeric)\n", n_hex == 0))
