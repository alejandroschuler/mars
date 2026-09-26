# Black-box experiments with earth

Each experiment answers one question about the behavior of the R package earth 5.3.4, which `docs/algorithm.md` cites by its ID. Each has an R script `bbNN_<slug>.R`, which makes its own small deterministic data and runs in under a minute, and its saved output `bbNN_<slug>.out`. An output ends with `CHECK` lines, one per claim, computed by the script.

Clean room: the scripts use earth only as a black box. They call its functions and read return values, printed output and `trace` logs. They never print an R function body, never read earth's C or R source, and quote nothing from earth's documentation.

To run one experiment again, from this folder:

```bash
. ../../dev/env.sh && /usr/local/bin/Rscript bb01_gcv_grid.R > bb01_gcv_grid.out 2>&1
```

The outputs were made with R 4.4.3 and earth 5.3.4 on macOS 15 (Apple silicon).

| ID | Question | Spec rules |
|---|---|---|
