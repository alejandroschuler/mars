import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
import pymars as earth
res = []
for seed in range(6):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (120, 4))
    y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + rng.standard_normal(120)
    for deg in (1, 2):
        m = earth.Earth(max_degree=deg).fit(X, y)
        res.append({"seed": seed, "deg": deg, "basis": [str(b) for b in m.basis_],
                    "pred": np.round(m.predict(X[:10]), 10).tolist(), "gcv": m.gcv_})
print(json.dumps(res))
