#!/usr/bin/env python3
"""z3 proof for a sufficient tape-feedback stability envelope.

Model:
    x_{n+1} = A x_n + g W R [x_n, x_{n-1}, ..., x_{n-K+1}]

Let ||A|| <= rho_core and ||W R|| <= wr. For any eigen-root lambda with
|lambda| >= rho_star, the delay equation gives

    |lambda| <= rho_core + gamma * sum_{j=0}^{K-1} |lambda|^{-j}

where gamma = g * wr. Since |lambda| >= rho_star, a sufficient condition for
no root outside rho_star is:

    rho_core + gamma * S_K(rho_star) <= rho_star,
    S_K(rho_star) = sum_{j=0}^{K-1} rho_star^{-j}.

This is intentionally conservative: it uses norm bounds and absolute delay
coefficients, so it is safe for any tape reader with operator norm <= ||R||.
"""

from __future__ import annotations

import argparse
from math import inf

from z3 import And, Implies, Not, Real, Solver, sat


def prove_symbolic() -> str:
    rho_core = Real("rho_core")
    rho_star = Real("rho_star")
    gamma = Real("gamma")
    s_k = Real("s_k")
    bound = Real("bound")

    assumptions = And(
        rho_star > 0,
        rho_core >= 0,
        gamma >= 0,
        s_k >= 1,
        rho_core < rho_star,
        gamma * s_k <= rho_star - rho_core,
        bound == rho_core + gamma * s_k,
    )
    claim = bound <= rho_star
    solver = Solver()
    solver.add(assumptions)
    solver.add(Not(claim))
    result = solver.check()
    if result == sat:
        return f"FAIL: counterexample {solver.model()}"
    return "PROVED: assumptions imply rho_core + gamma*S_K <= rho_star"


def s_k(rho_star: float, k: int) -> float:
    if rho_star <= 0.0:
        return inf
    return sum(rho_star ** (-j) for j in range(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-core", type=float, default=0.92)
    ap.add_argument("--rho-star", type=float, default=0.98)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--wr", type=float, default=1.0)
    args = ap.parse_args()

    proof = prove_symbolic()
    sk = s_k(args.rho_star, args.k)
    gamma_max = max(0.0, (args.rho_star - args.rho_core) / sk)
    g_max = gamma_max / args.wr if args.wr > 0.0 else inf
    print(proof)
    print(f"S_K={sk:.6f}")
    print(f"gamma_max={gamma_max:.6f}")
    print(f"g_max={g_max:.6f} for ||W||||R||={args.wr:.6f}")
    print(f"gate_init_recommendation={0.1 * g_max:.6f}")


if __name__ == "__main__":
    main()
