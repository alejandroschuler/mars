# Source this file before tests, gates, simulations and benchmarks: . dev/env.sh
# One thread for each BLAS and OpenMP library, so that results do not depend on
# the number of threads and parallel jobs do not compete for cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Use the installed Python interpreters, and never download one.
export UV_PYTHON_DOWNLOADS=never
