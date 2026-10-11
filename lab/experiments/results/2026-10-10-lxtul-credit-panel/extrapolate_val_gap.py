"""Joint power-law fit of trainer-val curves with a shared per-step batch offset.

L_k(t) = E_k + A_k * (t/1000)^(-a_k) + d(t);  d(t) is shared by every run at step t (the val
batches at a step are the same in every run), sum d = 0. Fit on steps >= 1000 (LR flat from 1000).
"""
import re, sys, json
import numpy as np
from scipy.optimize import least_squares

S = "/home/wolfe/morph-scratch/"
RUNS = {"lx_a": "perf/fast5k/train.log", "lx_b": "perf/fastbf5k/train.log",
        "lx_graph": "perf/graph/p5k/train.log", "lx_noise": "perf/graph/noise/train.log",
        "onecell": "abq/train-onecell.log", "spdwin": "spdwin/train-r1.log",
        "nospan": "perf/nospan5k/train.log"}

def val(p):
    d = {}
    for l in open(S + p):
        m = re.search(r"\[VAL\s+(\d+)\] loss=([0-9.]+)", l)
        if m: d[int(m.group(1))] = float(m.group(2))
    return d

V = {k: val(v) for k, v in RUNS.items()}
steps = sorted(s for s in set.intersection(*[set(v) for v in V.values()]) if s >= 1000)
names = list(V)
Y = np.array([[V[k][s] for s in steps] for k in names])          # [R, T]
t = np.array(steps) / 1000.0
R, T = Y.shape

def unpack(p):
    E, A, a = p[:R], p[R:2*R], p[2*R:3*R]
    d = np.append(p[3*R:], -p[3*R:].sum())
    return E, A, a, d

def resid(p, Y=Y):
    E, A, a, d = unpack(p)
    return (E[:, None] + A[:, None] * t[None] ** (-a[:, None]) + d[None] - Y).ravel()

def fit(Y):
    p0 = np.concatenate([Y[:, -1] - 0.3, np.full(R, 1.0), np.full(R, 0.5), np.zeros(T - 1)])
    lb = np.concatenate([np.full(R, 0.0), np.full(R, 0.0), np.full(R, 0.02), np.full(T - 1, -1)])
    ub = np.concatenate([np.full(R, 6.0), np.full(R, 20.0), np.full(R, 3.0), np.full(T - 1, 1)])
    return least_squares(resid, p0, bounds=(lb, ub), args=(Y,)).x

def curve(p, k, tt):
    E, A, a, _ = unpack(p)
    return E[k] + A[k] * tt ** (-a[k])

p = fit(Y)
E, A, a, d = unpack(p)
res = resid(p).reshape(R, T)
print(f"steps {steps[0]}..{steps[-1]} ({T} points), residual rms {np.sqrt((res**2).mean()):.4f}")
print("run        E(asymptote)  A      alpha   L(5k)fit")
for i, k in enumerate(names):
    print(f"{k:10s} {E[i]:.3f}        {A[i]:.3f}  {a[i]:.3f}   {curve(p, i, 5.0):.4f}")

lx = [names.index(k) for k in ("lx_a", "lx_b")]
oc = names.index("onecell")
horizons = {"5k": 5, "10k": 10, "20k": 20, "50k": 50, "2B tok (325k)": 325}
def gaps(p):
    return {h: float(curve(p, oc, x) - np.mean([curve(p, j, x) for j in lx])) for h, x in horizons.items()}
g = gaps(p)

# Bootstrap over residuals within each run (keeps d(t) structure); refit.
rng = np.random.default_rng(0)
fitted = Y - res
B = []
for _ in range(200):
    Yb = fitted + res[:, rng.integers(0, T, T)]
    B.append(gaps(fit(Yb)))
print("\nonecell minus LXTUL (mean of lx_a, lx_b), extrapolated:")
for h in horizons:
    xs = np.array([b[h] for b in B])
    print(f"  {h:14s} {g[h]:+.4f}   bootstrap 10-90%: [{np.percentile(xs,10):+.4f}, {np.percentile(xs,90):+.4f}]  P(gap>0)={np.mean(xs>0):.2f}")
# Crossover step for the point fit
xx = np.logspace(np.log10(5), np.log10(5000), 4000)
gg = curve(p, oc, xx) - np.mean([curve(p, j, xx) for j in lx], axis=0)
cross = xx[np.argmax(gg > 0)] if (gg > 0).any() else None
print("point-fit crossover:", f"step {cross*1000:.0f} ({cross*1000*6144/1e9:.2f}B tokens)" if cross else "none before 5M steps")
xsb = []
for b in range(len(B)): pass

print("\n=== variant fits: one alpha shared by all runs, per-run E and A ===")
def fit_shared(Y, alpha=None):
    def r(q):
        E, A = q[:R], q[R:2*R]
        al = alpha if alpha is not None else q[2*R]
        dd = np.append(q[-(T-1):], -q[-(T-1):].sum())
        return (E[:, None] + A[:, None] * t[None] ** (-al) + dd[None] - Y).ravel()
    extra = [] if alpha is not None else [0.5]
    q0 = np.concatenate([Y[:, -1] - 0.3, np.full(R, 1.0), extra, np.zeros(T - 1)])
    lb = np.concatenate([np.zeros(R), np.zeros(R), [0.02] * len(extra), np.full(T - 1, -1)])
    ub = np.concatenate([np.full(R, 6), np.full(R, 50), [3.0] * len(extra), np.full(T - 1, 1)])
    q = least_squares(r, q0, bounds=(lb, ub)).x
    return q, np.sqrt((r(q) ** 2).mean())
for alpha in (None, 0.3, 0.5, 1.0):
    q, rms = fit_shared(Y, alpha)
    E, A = q[:R], q[R:2*R]; al = alpha if alpha is not None else q[2*R]
    c = lambda k, x: E[k] + A[k] * x ** (-al)
    gs = {h: c(oc, x) - np.mean([c(j, x) for j in lx]) for h, x in horizons.items()}
    print(f"alpha={'free' if alpha is None else alpha} (fit {al:.3f}) rms {rms:.4f}: " +
          "  ".join(f"{h} {v:+.4f}" for h, v in gs.items()))

print("\n=== the gap alone (batch noise cancels): gap(t) = g0 + g1*ln(t), steps >= 2000 ===")
gap = np.array([V['onecell'][s] - np.mean([V['lx_a'][s], V['lx_b'][s]]) for s in steps])
m = np.array(steps) >= 2000
X = np.vstack([np.ones(m.sum()), np.log(t[m])]).T
coef, *_ = np.linalg.lstsq(X, gap[m], rcond=None)
rs = gap[m] - X @ coef
se = np.sqrt((rs ** 2).sum() / (m.sum() - 2) * np.linalg.inv(X.T @ X).diagonal())
print(f"g0 {coef[0]:+.4f}  g1 {coef[1]:+.4f} per e-fold of steps (se {se[1]:.4f}); residual sd {rs.std(ddof=2):.4f}")
for h, x in horizons.items():
    print(f"  {h:14s} {coef[0] + coef[1]*np.log(x):+.4f}  (+/- {2*np.sqrt(se[0]**2 + (se[1]*np.log(x))**2):.4f} at 2 se, ignoring covariance)")
if coef[1] > 0:
    print(f"  crossover by this line at step {np.exp(-coef[0]/coef[1])*1000:.0f}")
