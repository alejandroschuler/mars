"""Time pymars and earth default fits on Friedman #1 (p=10) as n grows.

Each pymars fit runs in its own subprocess with a timeout. Earth is timed in
one Rscript call per configuration on the same CSV.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
TIMEOUT = 600

FIT_SNIPPET = """
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import pymars as earth
d = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
X, y = d[:, :-1], d[:, -1]
t0 = time.perf_counter()
m = earth.Earth(max_degree=int(sys.argv[2])).fit(X, y)
print(time.perf_counter() - t0, len(m.basis_), len(m.record_.fwd_basis_[-1]))
"""

R_SNIPPET = """
suppressMessages(library(earth))
a <- commandArgs(trailingOnly=TRUE)
d <- read.csv(a[1]); x <- as.matrix(d[, grep('^x', names(d))]); y <- d$y
deg <- as.integer(a[2])
t <- sapply(1:3, function(i) system.time(f <<- earth(x, y, degree=deg))[['elapsed']])
cat(min(t), length(f$selected.terms), nrow(f$dirs), '\\n')
"""


def friedman1(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p))
    f = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4])
    return X, f + rng.standard_normal(n)


(HERE / "timing").mkdir(exist_ok=True)
(HERE / "timing" / "fit.py").write_text(FIT_SNIPPET)
(HERE / "timing" / "fit.R").write_text(R_SNIPPET)
results = []
timed_out = set()
for degree in [1, 2]:
    for n in [250, 500, 1000, 2000]:
        X, y = friedman1(n, 10, seed=n)
        csv = HERE / "timing" / f"f1_n{n}.csv"
        cols = [f"x{j}" for j in range(10)] + ["y"]
        np.savetxt(csv, np.column_stack([X, y]), delimiter=",", header=",".join(cols),
                   comments="", fmt="%.17g")
        r = subprocess.run(["Rscript", str(HERE / "timing" / "fit.R"), str(csv), str(degree)],
                           capture_output=True, text=True, check=True).stdout.split()
        row = {"n": n, "p": 10, "degree": degree, "earth_s": float(r[0]),
               "earth_terms": int(r[1]), "earth_fwd_terms": int(r[2])}
        if degree in timed_out:
            row["pymars_s"] = None
        else:
            try:
                out = subprocess.run([PY, str(HERE / "timing" / "fit.py"), str(csv), str(degree)],
                                     capture_output=True, text=True, timeout=TIMEOUT,
                                     check=True).stdout.split()
                row.update(pymars_s=float(out[0]), pymars_terms=int(out[1]),
                           pymars_fwd_terms=int(out[2]))
            except subprocess.TimeoutExpired:
                row["pymars_s"] = f">{TIMEOUT}"
                timed_out.add(degree)
        print(json.dumps(row), flush=True)
        results.append(row)
(HERE / "timing" / "results.json").write_text(json.dumps(results, indent=1))
