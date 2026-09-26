
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import pymars as earth
d = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
X, y = d[:, :-1], d[:, -1]
t0 = time.perf_counter()
m = earth.Earth(max_degree=int(sys.argv[2])).fit(X, y)
print(time.perf_counter() - t0, len(m.basis_), len(m.record_.fwd_basis_[-1]))
