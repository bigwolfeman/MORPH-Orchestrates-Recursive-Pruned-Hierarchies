"""Numerical check of the parallel_loop Lean theorems (CPU, numpy, one thread).

Part 1 (PL2, Prune.lean): the linear-Gaussian model trained by minibatch SGD on sampled
(z, phi, T). Compares the trained weights and per-depth losses with the closed forms the
theorems give: parallel prunes the duplicate channel (par_minimizer, par_stationary_prunes),
keeps the unique channel at the series weight (par2_minimizer), its K-curve is the unique
share of the series one (unique_share), and the stochastic skip keeps c_p and pays
c_p^2 var at eval (drop_minimizer, drop_tax).

Part 2 (PL1, DepthLaw.lean): a NONLINEAR learned loop with a hard per-pass step bound,
    v_t = P tanh(W x_t + U z + b),  x_{t+1} = x_t + delta * v_t / sqrt(1 + ||v_t||^2)  =>  ||step|| < delta,
read by a norm-bounded linear reader (||R||_F <= rho). Target y = [kappa z1, eta (z2^2 - 1)], eta = 1.0 (unique job longer than a pass) and 0.3 (fits one):
the first coordinate is what a skip carries, the second only the loop can build. Arms:
series, parallel (skip V z), thin-core parallel (delta / 2), withheld skip (||V||_F <= kappa/2),
stochastic skip (drop V z with p = 0.5 at train). Trained by Adam with a Poisson(6) depth law
clamped to [1, 8]; the readout is the per-depth loss and K1-K6 = L(1) - L(6).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 taskset -c 15 python toy.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

T_MAX = 8
DECAY = 0.447  # Parcae's injection decay at init


def depth_law(mean: float = 6.0, t_max: int = T_MAX) -> np.ndarray:
    """Poisson(mean) clamped to [1, t_max]; weights for depths 1..t_max."""
    k = np.arange(0, 60)
    pmf = np.exp(-mean + k * math.log(mean) - np.array([math.lgamma(i + 1) for i in k]))
    w = np.zeros(t_max + 1)
    for i, p in zip(k, pmf):
        w[min(max(i, 1), t_max)] += p
    return w[1:] / w[1:].sum()


PI = depth_law()
DEPTHS = np.arange(1, T_MAX + 1)


# ------------------------------------------------------------------ part 1: linear-Gaussian
def part1(rng: np.random.Generator, out) -> None:
    s = (1 - DECAY ** DEPTHS) / (1 - DECAY)          # scale build of the loop output
    S1, S2 = (PI * s).sum(), (PI * s * s).sum()
    var = S2 - S1 ** 2
    kappa, eta, p_drop = 1.0, 0.5, 0.5
    q = S1 / S2
    print(f"PART 1  depth law Poisson(6) clamped [1,8]; s_T = (1-a^T)/(1-a), a = {DECAY}", file=out)
    print(f"  S1 {S1:.4f}  S2 {S2:.4f}  var {var:.5f}  q = S1/S2 {q:.4f}", file=out)

    def train(arm: str, steps: int = 30000, batch: int = 512, lr: float = 0.02) -> dict:
        # factorized weights: skip v, loop write beta_k, reader w_k (k = 1: z channel, 2: phi channel)
        p = {"v": 0.0, "w1": 0.3, "b1": 0.3, "w2": 0.3, "b2": 0.3}
        for it in range(steps):
            z = rng.standard_normal(batch)
            ph = rng.standard_normal(batch)
            T = rng.choice(DEPTHS, size=batch, p=PI)
            sT = (1 - DECAY ** T) / (1 - DECAY)
            skip = np.ones(batch)
            if arm == "series":
                skip[:] = 0.0
            elif arm == "drop":
                skip = (rng.random(batch) >= p_drop).astype(float)
            c1, c2 = p["w1"] * p["b1"], p["w2"] * p["b2"]
            yhat = skip * p["v"] * z + c1 * sT * z + c2 * sT * ph
            r = kappa * z + eta * ph - yhat                     # residual; loss = mean r^2
            g_v = -2 * np.mean(r * skip * z)
            g_c1 = -2 * np.mean(r * sT * z)
            g_c2 = -2 * np.mean(r * sT * ph)
            lr_t = lr * (0.1 if it > steps * 0.8 else 1.0)
            if arm != "series":
                p["v"] -= lr_t * g_v
            gw1, gb1 = g_c1 * p["b1"], g_c1 * p["w1"]
            gw2, gb2 = g_c2 * p["b2"], g_c2 * p["w2"]
            p["w1"] -= lr_t * gw1
            p["b1"] -= lr_t * gb1
            p["w2"] -= lr_t * gw2
            p["b2"] -= lr_t * gb2
        p["c1"], p["c2"] = p["w1"] * p["b1"], p["w2"] * p["b2"]
        return p

    def eval_losses(p: dict, skip_on: bool) -> np.ndarray:
        """Exact per-depth eval loss for orthonormal unit features."""
        v = p["v"] if skip_on else 0.0
        return (kappa - v - p["c1"] * s) ** 2 + (eta - p["c2"] * s) ** 2

    rows = []
    for arm in ("parallel", "series", "drop"):
        p = train(arm)
        L = eval_losses(p, skip_on=(arm != "series"))
        rows.append((arm, p, L))
    c_p = p_drop * kappa * S1 / ((1 - p_drop) * var + p_drop * S2)
    theory = {
        "parallel": dict(v=kappa, c1=0.0, c2=eta * q),
        "series": dict(v=0.0, c1=kappa * q, c2=eta * q),
        "drop": dict(v=kappa - c_p * S1, c1=c_p, c2=eta * q),
    }
    print("  arm       |  v (SGD / theory)   |  c1 (SGD / theory)  |  c2 (SGD / theory)  |"
          "  L@1     L@6    K1-K6   | K1-K6 theory", file=out)
    k_theory = {}
    for arm, p, L in rows:
        th = theory[arm]
        v_t = th["v"]
        Lt = (kappa - v_t - th["c1"] * s) ** 2 + (eta - th["c2"] * s) ** 2
        k_theory[arm] = Lt[0] - Lt[5]
        print(f"  {arm:9s} | {p['v']:+.4f} / {v_t:+.4f}  | {p['c1']:+.4f} / {th['c1']:+.4f}  |"
              f" {p['c2']:+.4f} / {th['c2']:+.4f}  | {L[0]:.4f} {L[5]:.4f} {L[0]-L[5]:+.4f} |"
              f" {k_theory[arm]:+.4f}", file=out)
    k_par, k_ser = rows[0][2][0] - rows[0][2][5], rows[1][2][0] - rows[1][2][5]
    print(f"  unique share: K_par / K_ser = {k_par / k_ser:.4f}; theory eta^2/(kappa^2+eta^2) = "
          f"{eta**2 / (kappa**2 + eta**2):.4f}", file=out)
    drop_avg = (PI * ((kappa - rows[2][1]["v"] - rows[2][1]["c1"] * s) ** 2)).sum()
    print(f"  stochastic skip: eval z-channel depth-avg loss {drop_avg:.5f}; theory c_p^2 var = "
          f"{c_p**2 * var:.5f}; plain parallel optimum 0", file=out)


# ------------------------------------------------------------------ part 2: bounded-step loop
class Adam:
    def __init__(self, params: dict, lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, lr: float) -> None:
        self.t += 1
        for k in grads:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def project_fro(M: np.ndarray, r: float) -> np.ndarray:
    n = np.linalg.norm(M)
    return M if n <= r else M * (r / n)


def part2(seed: int, out, eta: float, steps: int = 10000) -> dict:
    nz, d, h, ny = 3, 8, 64, 2
    kappa, rho = 3.0, 1.0
    arms = {
        "series": dict(delta=1.0, skip="none"),
        "parallel": dict(delta=1.0, skip="full"),
        "thin-core parallel": dict(delta=0.5, skip="full"),
        "withheld skip": dict(delta=1.0, skip="half"),
        "stochastic skip": dict(delta=1.0, skip="drop"),
    }
    res = {}
    for name, cfg in arms.items():
        rng = np.random.default_rng(seed)
        prm = {
            "W": rng.standard_normal((h, d)) * 0.3,
            "U": rng.standard_normal((h, nz)) * 0.5,
            "b": rng.standard_normal(h) * 0.5,
            "P": rng.standard_normal((d, h)) * 0.1,
            "R": rng.standard_normal((ny, d)) * 0.1,
            "V": np.zeros((ny, nz)),
        }
        opt = Adam(prm, 3e-3)
        delta, mode = cfg["delta"], cfg["skip"]

        def bstep(v):
            q = 1.0 + np.sum(v * v, axis=1, keepdims=True)
            return delta * v / np.sqrt(q), q

        def forward(z, T, skip_mask, need_cache):
            B = z.shape[0]
            x = np.zeros((B, d))
            xs, ths, vs = [x], [], []
            xT = np.zeros((B, d))
            for t in range(T_MAX):
                a = x @ prm["W"].T + z @ prm["U"].T + prm["b"]
                th = np.tanh(a)
                v = th @ prm["P"].T
                x = x + bstep(v)[0]
                ths.append(th)
                vs.append(v)
                xs.append(x)
                xT = np.where((T == t + 1)[:, None], x, xT)
            yhat = xT @ prm["R"].T + skip_mask[:, None] * (z @ prm["V"].T)
            return yhat, xT, (xs, ths, vs) if need_cache else None

        def target(z):
            return np.stack([kappa * z[:, 0], eta * (z[:, 1] ** 2 - 1)], axis=1)

        for it in range(steps):
            B = 256
            z = rng.standard_normal((B, nz))
            T = rng.choice(DEPTHS, size=B, p=PI)
            if mode == "none":
                sk = np.zeros(B)
            elif mode == "drop":
                sk = (rng.random(B) >= 0.5).astype(float)
            else:
                sk = np.ones(B)
            yhat, xT, (xs, ths, vs) = forward(z, T, sk, True)
            gy = 2 * (yhat - target(z)) / B                     # d mean ||r||^2 / d yhat
            g = {k: np.zeros_like(v) for k, v in prm.items()}
            g["R"] = gy.T @ xT
            g["V"] = (gy * sk[:, None]).T @ z
            gx = np.zeros((B, d))
            for t in range(T_MAX - 1, -1, -1):
                gx = gx + np.where((T == t + 1)[:, None], gy @ prm["R"], 0.0)  # read at x_{t+1}
                th, v = ths[t], vs[t]
                q = 1.0 + np.sum(v * v, axis=1, keepdims=True)
                gv = delta * (gx * q - v * np.sum(v * gx, axis=1, keepdims=True)) / q ** 1.5
                g["P"] += gv.T @ th
                ga = (gv @ prm["P"]) * (1 - th ** 2)
                g["W"] += ga.T @ xs[t]
                g["U"] += ga.T @ z
                g["b"] += ga.sum(0)
                gx = gx + ga @ prm["W"]
            if mode == "none":
                g["V"][:] = 0.0
            lr = 3e-3 if it < 0.75 * steps else 3e-4
            opt.step(prm, g, lr)
            prm["R"] = project_fro(prm["R"], rho)
            if mode == "half":
                prm["V"] = project_fro(prm["V"], kappa / 2)
            if mode == "none":
                prm["V"][:] = 0.0

        # eval: fixed rows, every depth, skip on (except series)
        erng = np.random.default_rng(10_000 + seed)
        z = erng.standard_normal((20000, nz))
        sk = np.zeros(len(z)) if mode == "none" else np.ones(len(z))
        L = []
        for T_eval in DEPTHS:
            yhat, _, _ = forward(z, np.full(len(z), T_eval), sk, False)
            L.append(float(np.mean(np.sum((yhat - target(z)) ** 2, axis=1))))
        # step-bound audit: largest step norm seen on the eval rows
        x = np.zeros((len(z), d))
        max_step = 0.0
        for _ in range(T_MAX):
            stp = bstep(np.tanh(x @ prm["W"].T + z @ prm["U"].T + prm["b"]) @ prm["P"].T)[0]
            max_step = max(max_step, float(np.linalg.norm(stp, axis=1).max()))
            x = x + stp
        res[name] = dict(L=np.array(L), max_step=max_step, delta=delta)
    return res


def main() -> None:
    out = sys.stdout
    print(f"depth law weights (T=1..8): {np.round(PI, 4).tolist()}", file=out)
    part1(np.random.default_rng(0), out)
    for eta in (1.0, 0.3):
        print(f"\nPART 2  bounded-step nonlinear loop, unique job eta = {eta} "
              f"({'longer than one pass for most rows' if eta > 0.5 else 'fits one pass for ~96 % of rows'}); "
              "per-depth eval loss (mean over 20k rows)", file=out)
        allres = [part2(seed, out, eta) for seed in (1, 2)]
        print("  arm                 seed | L@1     L@2     L@3     L@4     L@6     L@8    | K1-K6   "
              "| max step / delta", file=out)
        for name in allres[0]:
            for si, res in enumerate(allres):
                r = res[name]
                L = r["L"]
                print(f"  {name:20s} {si+1:>3d} | " + " ".join(f"{L[t-1]:.4f}" for t in (1, 2, 3, 4, 6, 8))
                      + f" | {L[0]-L[5]:+.4f} | {r['max_step']:.3f} / {r['delta']}", file=out)


if __name__ == "__main__":
    main()
