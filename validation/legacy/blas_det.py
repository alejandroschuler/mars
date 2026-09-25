import sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import pymars as earth
out = []
for seed in range(8):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (150, 5))
    y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4] + rng.standard_normal(150)
    m = earth.Earth(max_degree=1).fit(X, y)
    out.append([str(b) for b in m.basis_] + [repr(float(m.gcv_))])
print(json.dumps(out))
