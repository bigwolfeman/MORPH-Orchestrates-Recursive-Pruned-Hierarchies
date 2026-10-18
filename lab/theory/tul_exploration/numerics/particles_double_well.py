"""Numerics for Particles/Depth.lean and Particles/Collapse.lean on the 1-D double well.

Pass: G(x) = tanh(beta * x) + c.  For beta > 1 and |c| < c_crit(beta) it has two stable
fixed points and one repelling one (the saddle x_u); past c_crit one basin disappears.
Starts: x0 ~ N(0, s0^2) ("like-init"), K particles per context, T passes.

Checks, each against the Lean statement it illustrates:
  1. passes_to_settle: the empirical first T with |x_T - x*| <= eps is >= the co-Lipschitz
     bound log(d0/eps)/log(1/lam), lam = min over the trajectory region of |G'| (here we use
     the local rate G'(x*) and report both; the theorem needs the region-wide lower rate).
  2. saddle_read_margin / passes_to_decide: the fraction of particles with |x_T - x_u| < m
     (undecided for a margin-m Lipschitz reader) vs T; it shrinks about like mu^-T,
     mu = G'(x_u).
  3. monotone_side_preserved: no particle ever changes side of x_u.
  4. particles_spread (contracting control, beta < 1): spread ratio std(x_T)/std(x_0) vs
     rho^T with rho = beta (sup |G'|).
  5. Per-context diversity: the fraction of particles in the upper basin vs c.  The basin
     set depends on c; beyond c_crit every particle lands in one basin.

CPU, one thread:
  OMP_NUM_THREADS=1 nice -n 19 taskset -c 15 python numerics/particles_double_well.py
"""
import math
import numpy as np

rng = np.random.default_rng(0)


def G(x, beta, c):
    return np.tanh(beta * x) + c


def dG(x, beta):
    return beta * (1.0 - np.tanh(beta * x) ** 2)


def fixed_points(beta, c, lo=-3.0, hi=3.0, n=200001):
    xs = np.linspace(lo, hi, n)
    f = G(xs, beta, c) - xs
    roots = []
    for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        a, b = xs[i], xs[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(G(a, beta, c) - a) == np.sign(G(m, beta, c) - m):
                a = m
            else:
                b = m
        r = 0.5 * (a + b)
        if not roots or abs(r - roots[-1]) > 1e-6:
            roots.append(r)
    return roots


def c_crit(beta):
    t = math.sqrt(1.0 - 1.0 / beta)
    x = math.atanh(t) / beta
    return abs(x - t)


def main():
    beta, c, s0, N = 2.0, 0.0, 1.0, 200000
    fps = fixed_points(beta, c)
    stable = [r for r in fps if abs(dG(r, beta)) < 1]
    unstable = [r for r in fps if abs(dG(r, beta)) > 1]
    xu = unstable[0]
    print(f"beta={beta} c={c}: fixed points {np.round(fps, 5)}; stable {np.round(stable, 5)};"
          f" saddle {xu:.5f}; G'(x*)={dG(stable[-1], beta):.4f}; G'(x_u)={dG(xu, beta):.4f};"
          f" c_crit={c_crit(beta):.4f}")

    x0 = rng.normal(0.0, s0, N)
    eps = 1e-3
    lam = dG(stable[-1], beta)
    mu = dG(xu, beta)
    x = x0.copy()
    side0 = np.sign(x0 - xu)
    settled_T = np.full(N, -1)
    print("T  frac_settled(eps=1e-3)  frac_undecided(|x-x_u|<0.25)  mu^-T*frac0  side_flips")
    und0 = np.mean(np.abs(x0 - xu) < 0.25)
    for T in range(0, 13):
        if T > 0:
            x = G(x, beta, c)
        target = np.where(x0 > xu, stable[-1], stable[0])
        newly = (settled_T < 0) & (np.abs(x - target) <= eps)
        settled_T[newly] = T
        und = np.mean(np.abs(x - xu) < 0.25)
        flips = int(np.sum(np.sign(x - xu) != side0))
        print(f"{T:2d} {np.mean(settled_T >= 0):.4f}                  {und:.5f}"
              f"                     {und0 * mu ** -T:.5f}      {flips}")

    # check 1: empirical settle time vs co-Lipschitz bound with the LOCAL rate
    target = np.where(x0 > xu, stable[-1], stable[0])
    d0 = np.abs(x0 - target)
    ok = (settled_T > 0) & (d0 > eps)
    bound_local = np.log(d0[ok] / eps) / np.log(1.0 / lam)
    viol = np.mean(settled_T[ok] < bound_local - 1e-9)
    print(f"settle-time vs local-rate bound log(d0/eps)/log(1/G'(x*)): violations {viol:.4f}"
          " (the theorem needs the REGION-wide co-Lipschitz rate; the local rate is not it)")
    # region-wide co-Lipschitz rate toward x* on the region the trajectories visit
    xs = np.linspace(xu + 1e-3, 3.0, 20001)
    ratio = np.abs(G(xs, beta, c) - stable[-1]) / np.abs(xs - stable[-1] + 1e-300)
    lam_region = ratio[np.abs(xs - stable[-1]) > 1e-6].min()
    sel = ok & (x0 > xu + 1e-3) & (x0 < 3.0)
    bound_region = np.log(d0[sel] / eps) / np.log(1.0 / lam_region)
    viol_r = np.mean(settled_T[sel] < bound_region - 1e-9)
    print(f"region-wide co-Lipschitz rate on (x_u, 3]: lam={lam_region:.4g}; bound violations"
          f" {viol_r:.4f} (must be 0)")

    # check 4: contracting control
    bc = 0.6
    x = rng.normal(0.0, s0, 50000)
    sd0 = x.std()
    print(f"\ncontracting control beta={bc} (sup|G'|={bc}): spread ratio vs rho^T")
    for T in range(1, 9):
        x = G(x, bc, 0.3)
        print(f"  T={T}: std ratio {x.std() / sd0:.3e}   rho^T {bc ** T:.3e}")

    # check 5: per-context diversity
    print(f"\nupper-basin fraction of particles vs c (beta={beta}, s0={s0}, T=8)")
    for cc in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        x = rng.normal(0.0, s0, 50000)
        for _ in range(8):
            x = G(x, beta, cc)
        nst = sum(1 for r in fixed_points(beta, cc) if abs(dG(r, beta)) < 1)
        print(f"  c={cc:+.2f}: stable points {nst}, upper-basin fraction {np.mean(x > 0):.3f},"
              f" spread std {x.std():.3f}")


if __name__ == "__main__":
    main()
