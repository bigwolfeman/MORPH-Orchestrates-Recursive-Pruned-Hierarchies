"""Numerical companion to the Lean theorems: the Euler k-curve of the L2-optimal flow-matching
field for (a) a Gaussian conditional and (b) a separated-mixture conditional.

The field is the exact conditional expectation E[z1 - z0 | z_t = z] in both cases (closed
form; no training), so the k-curve read here is the k-curve of the DESIGN, with no
optimisation gap. Source z0 ~ N(0, I). Euler as in `morph/model/tul_code.py::euler_sample`.

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 taskset -c 0,1 python sim/kcurve.py

Metrics per k (higher log-density = the endpoint looks like a real code to a coda that
trusts codes; residual = E||endpoint - independent true code||^2 / total code variance,
the instrument of `code_subspace_probe.py`: 2.0 an unconditional draw, 0 a perfect guess):

* `logp`   mean log-density of the endpoint under the TRUE conditional law
* `resid`  residual ratio against an independent draw from the conditional
* `radius` mean norm of (endpoint - conditional mean) over sqrt(D): the lambda_k of
           `eulerEndpoint_gauss` (Gaussian case)
* `cos_mu` (Gaussian only) mean cosine between the endpoint and the conditional mean: the
           share of the context signal in the DIRECTION the coda reads after `code_rmsnorm`
* `hit`    (mixture only) fraction of endpoints within 2 sigma sqrt(D) of some mode: the
           endpoint sits IN a mode rather than between modes
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
D = 16            # dimensions per cell (small: the effect is dimension-free)
N = 4000          # endpoints per setting
KS = [1, 2, 4, 8, 16, 64]


def euler(velocity, z0: np.ndarray, k: int) -> np.ndarray:
    z = z0.copy()
    dt = 1.0 / k
    for j in range(k):
        z = z + dt * velocity(z, j * dt)
    return z


def log_gauss(z: np.ndarray, mu: np.ndarray, var: float) -> np.ndarray:
    d = z.shape[-1]
    return -0.5 * ((z - mu) ** 2).sum(-1) / var - 0.5 * d * np.log(2 * np.pi * var)


# ---------------------------------------------------------------- (a) Gaussian conditional
def gauss_case(r2: float, s0: float = 1.0):
    """Conditional N(mu, s1^2 I) with total per-dim variance 1 and explained fraction r2:
    mu ~ N(0, r2 I) over contexts, s1^2 = 1 - r2."""
    s1sq = 1.0 - r2
    mu = rng.normal(size=(N, D)) * np.sqrt(r2)          # one context per row

    def a(t):
        return (t * s1sq - (1 - t) * s0 ** 2) / ((1 - t) ** 2 * s0 ** 2 + t * t * s1sq)

    def velocity(z, t):
        return a(t) * (z - t * mu) + mu

    z0 = rng.normal(size=(N, D)) * s0
    true = mu + rng.normal(size=(N, D)) * np.sqrt(s1sq)   # an independent true code
    total_var = D * 1.0
    def cos_mu(z):
        num = (z * mu).sum(-1)
        den = np.linalg.norm(z, axis=-1) * np.linalg.norm(mu, axis=-1) + 1e-12
        return float((num / den).mean())

    rows = []
    for k in KS:
        zk = euler(velocity, z0, k)
        rows.append(dict(k=k,
                         logp=float(log_gauss(zk, mu, s1sq).mean()),
                         resid=float(((zk - true) ** 2).sum(-1).mean() / total_var),
                         radius=float(np.linalg.norm(zk - mu, axis=-1).mean() / np.sqrt(D * s0 ** 2)),
                         cos_mu=cos_mu(zk)))
    exact = mu + (np.sqrt(s1sq) / s0) * z0               # the k = infinity endpoint
    rows.append(dict(k="inf",
                     logp=float(log_gauss(exact, mu, s1sq).mean()),
                     resid=float(((exact - true) ** 2).sum(-1).mean() / total_var),
                     radius=float(np.linalg.norm(exact - mu, axis=-1).mean() / np.sqrt(D * s0 ** 2)),
                     cos_mu=cos_mu(exact)))
    ref = float(log_gauss(true, mu, s1sq).mean())
    return rows, ref


# ---------------------------------------------------------------- (b) mixture conditional
def mixture_case(m_modes: int, sep: float, sw: float, s0: float = 1.0):
    """Conditional (1/M) sum_j N(mu_j, sw^2 I), the modes mu_j ~ N(0, sep^2 I) (shared across
    contexts for simplicity: the context is the mode set). The L2-optimal field is the
    posterior-weighted mixture of the per-component Gaussian fields."""
    mus = rng.normal(size=(m_modes, D)) * sep                 # [M, D]
    swsq = sw * sw

    def a(t):
        return (t * swsq - (1 - t) * s0 ** 2) / ((1 - t) ** 2 * s0 ** 2 + t * t * swsq)

    def velocity(z, t):
        st2 = (1 - t) ** 2 * s0 ** 2 + t * t * swsq
        # posterior over components given z_t = z: N(z; t mu_j, st2 I)
        logits = -0.5 * (((z[:, None, :] - t * mus[None]) ** 2).sum(-1)) / st2   # [N, M]
        logits -= logits.max(-1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(-1, keepdims=True)
        # per-component field a(t)(z - t mu_j) + mu_j, averaged
        v = a(t) * z + (w @ mus) * (1.0 - t * a(t))
        return v

    def logp(z):
        ll = log_gauss(z[:, None, :], mus[None], swsq) - np.log(m_modes)     # [N, M]
        mx = ll.max(-1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(ll - mx).sum(-1)))

    z0 = rng.normal(size=(N, D)) * s0
    comp = rng.integers(0, m_modes, size=N)
    true = mus[comp] + rng.normal(size=(N, D)) * sw
    mean = mus.mean(0)
    total_var = float(((true - mean) ** 2).sum(-1).mean())
    rows = []
    for k in KS:
        zk = euler(velocity, z0, k)
        dmin = np.sqrt(((zk[:, None, :] - mus[None]) ** 2).sum(-1)).min(-1)
        rows.append(dict(k=k,
                         logp=float(logp(zk).mean()),
                         resid=float(((zk - true) ** 2).sum(-1).mean() / total_var),
                         radius=float(np.linalg.norm(zk - mean, axis=-1).mean() / np.sqrt(D)),
                         hit=float((dmin < 2.0 * sw * np.sqrt(D)).mean())))
    ref = float(logp(true).mean())
    dtrue = np.sqrt(((true[:, None, :] - mus[None]) ** 2).sum(-1)).min(-1)
    hit_ref = float((dtrue < 2.0 * sw * np.sqrt(D)).mean())
    return rows, ref, hit_ref


def show(title, rows, ref, extra=""):
    print(f"\n== {title}  (true-code reference logp {ref:.3f}{extra})")
    print("   k     logp    resid  radius" + ("     hit" if "hit" in rows[0] else "")
          + ("  cos_mu" if "cos_mu" in rows[0] else ""))
    for r in rows:
        k = f"{r['k']:>4}"
        s = f"{k}  {r['logp']:8.3f}  {r['resid']:6.3f}  {r['radius']:6.3f}"
        if "hit" in r:
            s += f"  {r['hit']:6.3f}"
        if "cos_mu" in r:
            s += f"  {r['cos_mu']:6.3f}"
        print(s)
    k2 = next(r for r in rows if r["k"] == 2)
    k16 = next(r for r in rows if r["k"] == 16)
    print(f"   slope k2 -> k16: logp {k16['logp'] - k2['logp']:+.3f}, resid {k16['resid'] - k2['resid']:+.3f}")


if __name__ == "__main__":
    for r2 in (0.1, 0.5):
        rows, ref = gauss_case(r2)
        show(f"(a) Gaussian conditional, explained fraction R^2 = {r2}", rows, ref)

    for m, sep, sw in ((2, 2.0, 0.3), (8, 2.0, 0.3), (64, 2.0, 0.3), (4096, 2.0, 0.3),
                       (8, 2.0, 1.0), (8, 0.5, 0.3)):
        rows, ref, hit_ref = mixture_case(m, sep, sw)
        show(f"(b) mixture: M = {m} modes, mode spread {sep}, within-mode std {sw}",
             rows, ref, extra=f", true-code hit rate {hit_ref:.3f}")
