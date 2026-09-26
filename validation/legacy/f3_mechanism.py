import warnings, numpy as np
warnings.filterwarnings("ignore")
import pymars as earth
from pymars._basis import ConstantBasisFunction, HingeBasisFunction
from pymars._util import calculate_gcv, gcv_penalty_cost_effective_parameters

def friedman1(n, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    f = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4]
    return X, f + rng.standard_normal(n)

def fit_stats(X, y, bfs, mm, d):
    B = np.column_stack([b.transform(X, mm) for b in bfs])
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    rss = float(np.sum((y - B @ coef) ** 2))
    H = sum(isinstance(b, HingeBasisFunction) for b in bfs)
    C = gcv_penalty_cost_effective_parameters(len(bfs), H, d, len(y))
    return rss, C, calculate_gcv(rss, len(y), C), np.linalg.matrix_rank(B), B.shape[1]

shown = 0
for r in range(30):
    X, y = friedman1(100, seed=100 + r)
    m = earth.Earth(max_degree=2).fit(X, y)
    if any(isinstance(b, ConstantBasisFunction) for b in m.basis_):
        continue
    mm = np.zeros_like(X, bool)
    tr = m.record_.pruning_trace_basis_functions_
    # find the step where the intercept leaves
    for k in range(1, len(tr)):
        had = any(isinstance(b, ConstantBasisFunction) for b in tr[k-1])
        has = any(isinstance(b, ConstantBasisFunction) for b in tr[k])
        if had and not has:
            before = tr[k-1]
            rss0, C0, g0, rank0, ncol0 = fit_stats(X, y, before, mm, 3.0)
            rows = []
            for j, b in enumerate(before):
                sub = before[:j] + before[j+1:]
                rss, C, g, rank, ncol = fit_stats(X, y, sub, mm, 3.0)
                rows.append((g, rss - rss0, C0 - C, type(b).__name__, str(b)[:45]))
            rows.sort()
            print(f"seed {100+r}: step {k}, model had {ncol0} columns, rank {rank0}; RSS {rss0:.4f}")
            for g, dr, dc, t, s in rows[:4]:
                print(f"   remove {t:24s} dRSS={dr:10.4g} dC={dc:4.1f} GCV={g:.5f}  {s}")
            break
    shown += 1
    if shown == 4:
        break
