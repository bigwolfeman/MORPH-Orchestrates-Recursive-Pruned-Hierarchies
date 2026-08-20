#!/usr/bin/env python3
"""Toy probes for boundary-trajectory readout decisions.

The script estimates geometry and forecastability on synthetic boundary
trajectories. It intentionally avoids ML framework dependencies so the probes
are transparent and reproducible from numpy alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x.T @ x
    reg = lam * np.eye(xtx.shape[0], dtype=x.dtype)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x @ w


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def pca_stats(z: np.ndarray) -> dict[str, float]:
    x = z - z.mean(axis=0, keepdims=True)
    eig = np.linalg.eigvalsh((x.T @ x) / max(len(x) - 1, 1))[::-1]
    eig = np.maximum(eig, 0.0)
    total = float(eig.sum())
    pr = float(total**2 / max(float(np.sum(eig**2)), 1e-12))
    csum = np.cumsum(eig) / max(total, 1e-12)
    d90 = int(np.searchsorted(csum, 0.90) + 1)
    d99 = int(np.searchsorted(csum, 0.99) + 1)
    return {"participation_dim": pr, "pca_d90": d90, "pca_d99": d99}


def curvature(z: np.ndarray, stride: int = 4) -> np.ndarray:
    """Short-stride turning curvature, robust to boundary-level jitter."""
    v_prev = z[stride:-stride] - z[:-2 * stride]
    v_next = z[2 * stride :] - z[stride:-stride]
    dot = np.sum(v_prev * v_next, axis=1)
    denom = np.linalg.norm(v_prev, axis=1) * np.linalg.norm(v_next, axis=1) + 1e-8
    cos = np.clip(dot / denom, -1.0, 1.0)
    return 0.5 * (1.0 - cos)


def make_windows(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(k - 1, len(z) - 1):
        xs.append(z[i - k + 1 : i + 1].reshape(-1))
        ys.append(z[i + 1])
    return np.asarray(xs), np.asarray(ys)


def fourier_features(x: np.ndarray, rng: np.random.Generator, m: int = 96) -> np.ndarray:
    d = x.shape[1]
    idx = rng.choice(x.shape[0], size=min(256, x.shape[0]), replace=False)
    sample = x[idx]
    pair = sample[: len(sample) // 2] - sample[len(sample) // 2 : 2 * (len(sample) // 2)]
    sigma = float(np.median(np.linalg.norm(pair, axis=1))) if len(pair) else 1.0
    sigma = max(sigma, 1e-3)
    w = rng.normal(scale=1.0 / sigma, size=(d, m))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(m,))
    return np.sqrt(2.0 / m) * np.cos(x @ w + b)


def pca_poly_features(
    x_train: np.ndarray, x_all: np.ndarray, rank: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Low-rank order-aware nonlinear features for the M4 toy proxy."""
    mu = x_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_train - mu, full_matrices=False)
    basis = vt[: min(rank, vt.shape[0])].T
    p = (x_all - mu) @ basis
    feats = [p, p**2]
    pairs = []
    r = p.shape[1]
    for i in range(r):
        for j in range(i + 1, r):
            pairs.append((p[:, i] * p[:, j])[:, None])
    if pairs:
        feats.append(np.concatenate(pairs, axis=1))
    out = np.concatenate(feats, axis=1)
    return out[: len(x_train)], out


@dataclass
class SimConfig:
    n: int = 3200
    d: int = 16
    q: int = 4
    k: int = 6
    seed: int = 7


def random_embed(rng: np.random.Generator, q: int, d: int) -> np.ndarray:
    qmat, _ = np.linalg.qr(rng.normal(size=(d, q)))
    return qmat.T


def simulate(kind: str, cfg: SimConfig, pivot_p: float = 0.05, switch_mag: float = 0.9) -> np.ndarray:
    kind_seed = {
        "linear": 11,
        "piecewise": 23,
        "nonlinear": 37,
        "random_walk": 41,
        "switch_nonlinear": 53,
    }[kind]
    rng = np.random.default_rng(cfg.seed + kind_seed + int(1000 * pivot_p) + int(100 * switch_mag))
    emb = random_embed(rng, cfg.q, cfg.d)
    y = np.zeros((cfg.n, cfg.q))
    y[0] = rng.normal(scale=0.5, size=cfg.q)

    if kind == "linear":
        # Stationary second-order linear flow: position plus damped velocity.
        pos = rng.normal(scale=0.4, size=cfg.q)
        vel = rng.normal(scale=0.08, size=cfg.q)
        for t in range(cfg.n - 1):
            y[t] = pos
            acc = -0.018 * pos + 0.003 * rng.normal(size=cfg.q)
            pos = pos + vel + 0.004 * rng.normal(size=cfg.q)
            vel = 0.982 * vel + acc
        y[-1] = pos
    elif kind == "piecewise":
        rotations = []
        for _ in range(4):
            qmat, _ = np.linalg.qr(rng.normal(size=(cfg.q, cfg.q)))
            rotations.append((1.0 - 0.12 * switch_mag) * np.eye(cfg.q) + 0.12 * switch_mag * qmat)
        pos = rng.normal(scale=0.4, size=cfg.q)
        vel = rng.normal(scale=0.08, size=cfg.q)
        state = 0
        for t in range(cfg.n - 1):
            y[t] = pos
            if rng.random() < pivot_p:
                state = int(rng.integers(0, len(rotations)))
                vel = rotations[state] @ vel + 0.06 * switch_mag * rng.normal(size=cfg.q)
            acc = -0.018 * pos + 0.004 * rng.normal(size=cfg.q)
            pos = pos + vel + 0.004 * rng.normal(size=cfg.q)
            vel = 0.982 * vel + acc
        y[-1] = pos
    elif kind == "nonlinear":
        omega = np.linspace(0.015, 0.055, cfg.q)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=cfg.q)
        for t in range(cfg.n):
            base = phase + omega * t
            y[t] = np.sin(base) + 0.35 * np.sin(2.7 * base + 0.2)
        y += 0.015 * rng.normal(size=y.shape)
    elif kind == "random_walk":
        for t in range(cfg.n - 1):
            y[t + 1] = y[t] + 0.10 * rng.normal(size=cfg.q)
    elif kind == "switch_nonlinear":
        pos = rng.normal(scale=0.4, size=cfg.q)
        vel = rng.normal(scale=0.07, size=cfg.q)
        for t in range(cfg.n - 1):
            y[t] = pos
            cue = np.tanh(4.0 * pos[0] * pos[1])
            p_turn = np.clip(pivot_p * (1.0 + cue), 0.0, 0.9)
            if rng.random() < p_turn:
                theta = switch_mag * (0.35 + 0.25 * np.tanh(pos[2] * pos[3]))
                c, s = np.cos(theta), np.sin(theta)
                pair = vel[:2].copy()
                vel[:2] = np.array([c * pair[0] - s * pair[1], s * pair[0] + c * pair[1]])
                vel += 0.02 * switch_mag * rng.normal(size=cfg.q)
            nonlinear_acc = 0.010 * switch_mag * np.array(
                [
                    np.sin(2.0 * pos[1]),
                    np.tanh(3.0 * pos[0] * pos[2]),
                    np.sin(pos[0] + pos[3]),
                    np.tanh(pos[1] * pos[3]),
                ]
            )
            acc = -0.016 * pos + nonlinear_acc + 0.003 * rng.normal(size=cfg.q)
            pos = pos + vel + 0.004 * rng.normal(size=cfg.q)
            vel = 0.985 * vel + acc
        y[-1] = pos

    z = y @ emb + 0.012 * rng.normal(size=(cfg.n, cfg.d))
    if kind == "nonlinear":
        z += 0.12 * np.tanh(y @ rng.normal(size=(cfg.q, cfg.d)) / np.sqrt(cfg.q))
    return z


def forecast_metrics(z: np.ndarray, cfg: SimConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed + 123)
    k = cfg.k
    xk = z[k - 1 : -1]
    xprev = z[k - 2 : -2]
    y = z[k:]
    vel_direct = xk + (xk - xprev)
    lastk, y2 = make_windows(z, k)
    assert np.allclose(y, y2)
    n_train = int(0.7 * len(y))

    def fit_eval(x: np.ndarray, target: np.ndarray) -> float:
        w = ridge_fit(x[:n_train], target[:n_train])
        pred = ridge_predict(x[n_train:], w)
        return r2_score(target[n_train:], pred)

    pos = fit_eval(xk, y)
    vel = r2_score(y[n_train:], vel_direct[n_train:])
    lin_k = fit_eval(lastk, y)
    ff = fourier_features(lastk, rng, m=96)
    nonlinear_k = fit_eval(np.concatenate([lastk, ff], axis=1), y)
    return {
        "r2_zk": pos,
        "r2_zk_plus_v": vel,
        "r2_linear_lastk": lin_k,
        "r2_nonlinear_lastk": nonlinear_k,
        "linearity_L": lin_k / max(nonlinear_k, 1e-9),
    }


def residual_forecast_metrics(
    z: np.ndarray, cfg: SimConfig, local: int = 2, tape: int = 6, horizon: int = 8
) -> dict[str, float]:
    """Forecast future residual after subtracting a short recency baseline."""
    rng = np.random.default_rng(cfg.seed + 456)
    xs_local, xs_tape, ys = [], [], []
    start = max(local, tape) - 1
    stop = len(z) - horizon
    for i in range(start, stop):
        xs_local.append(z[i - local + 1 : i + 1].reshape(-1))
        xs_tape.append(z[i - tape + 1 : i + 1].reshape(-1))
        ys.append(z[i + horizon])
    x0 = np.asarray(xs_local)
    xk = np.asarray(xs_tape)
    y = np.asarray(ys)
    order = rng.permutation(len(y))
    n_train = int(0.7 * len(y))
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    w0 = ridge_fit(x0[train_idx], y[train_idx], lam=1e-2)
    resid_train = y[train_idx] - ridge_predict(x0[train_idx], w0)
    resid_test = y[test_idx] - ridge_predict(x0[test_idx], w0)

    wl = ridge_fit(xk[train_idx], resid_train, lam=1e-2)
    lin = ridge_predict(xk[test_idx], wl)

    _, poly = pca_poly_features(xk[train_idx], xk, rank=10)
    xn = np.concatenate([xk, poly], axis=1)
    wn = ridge_fit(xn[train_idx], resid_train, lam=1e-1)
    nonlin = ridge_predict(xn[test_idx], wn)

    r2_l = r2_score(resid_test, lin)
    r2_n = r2_score(resid_test, nonlin)
    r2_l_pos = max(0.0, r2_l)
    r2_n_pos = max(0.0, r2_n)
    return {
        "r2_cond_linear": r2_l,
        "r2_cond_nonlinear": r2_n,
        "linearity_cond_L": min(1.0, r2_l_pos / max(r2_n_pos, 1e-9)),
        "nonlinear_cond_advantage": r2_n - r2_l,
    }


def conditional_gaussian_mi(y: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> float:
    """Gaussian predictive-probe conditional MI lower bound in nats/dim.

    For residual covariance after linear probes,
    I(Y; X1 | X0) >= 0.5/d * log det Sigma(Y|X0) / det Sigma(Y|X0,X1).
    """
    n_train = int(0.7 * len(y))
    w0 = ridge_fit(x0[:n_train], y[:n_train], lam=1e-2)
    w01 = ridge_fit(np.concatenate([x0, x1], axis=1)[:n_train], y[:n_train], lam=1e-2)
    e0 = y[n_train:] - ridge_predict(x0[n_train:], w0)
    e01 = y[n_train:] - ridge_predict(np.concatenate([x0, x1], axis=1)[n_train:], w01)
    c0 = np.cov(e0, rowvar=False) + 1e-5 * np.eye(y.shape[1])
    c01 = np.cov(e01, rowvar=False) + 1e-5 * np.eye(y.shape[1])
    sign0, ld0 = np.linalg.slogdet(c0)
    sign1, ld1 = np.linalg.slogdet(c01)
    if sign0 <= 0 or sign1 <= 0:
        return float("nan")
    return float(0.5 * (ld0 - ld1) / y.shape[1])


def mi_probe(z: np.ndarray, cfg: SimConfig, local: int = 2, tape: int = 6, horizon: int = 8) -> dict[str, float]:
    xs_local, xs_tape, ys = [], [], []
    start = max(local, tape) - 1
    stop = len(z) - horizon
    for i in range(start, stop):
        xs_local.append(z[i - local + 1 : i + 1].reshape(-1))
        xs_tape.append(z[i - tape + 1 : i - local + 1].reshape(-1))
        ys.append(z[i + horizon])
    x0 = np.asarray(xs_local)
    x1 = np.asarray(xs_tape)
    y = np.asarray(ys)
    v = np.diff(z, axis=0)
    traj = []
    resid = []
    for i in range(start, stop):
        block = z[i - tape + 1 : i - local + 1]
        vv = v[i - tape + 1 : i - local + 1]
        traj.append(np.concatenate([block[-1], vv.mean(axis=0), vv[-1] - vv[0]]))
        resid.append((block - block.mean(axis=0, keepdims=True)).reshape(-1))
    traj = np.asarray(traj)
    resid = np.asarray(resid)
    return {
        "mi_tape_given_local_nats_per_dim": conditional_gaussian_mi(y, x0, x1),
        "mi_traj_given_local_nats_per_dim": conditional_gaussian_mi(y, x0, traj),
        "mi_resid_given_local_nats_per_dim": conditional_gaussian_mi(y, x0, resid),
    }


def summarize(kind: str, cfg: SimConfig, pivot_p: float, switch_mag: float) -> dict[str, float | str]:
    z = simulate(kind, cfg, pivot_p=pivot_p, switch_mag=switch_mag)
    curv = curvature(z)
    out: dict[str, float | str] = {"kind": kind}
    out.update(pca_stats(z))
    out.update(
        {
            "curv_median": float(np.median(curv)),
            "curv_p90": float(np.quantile(curv, 0.90)),
            "pivot_rate_pi": float(np.mean(curv > 0.20)),
        }
    )
    out.update(forecast_metrics(z, cfg))
    out.update(residual_forecast_metrics(z, cfg))
    out.update(mi_probe(z, cfg))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--crossover", action="store_true")
    args = ap.parse_args()
    cfg = SimConfig()
    if args.crossover:
        print("pivot_p\tswitch_mag\tpi\tL_cond\tm4_cond_advantage")
        for pivot_p in [0.02, 0.08, 0.18]:
            for switch_mag in [0.4, 0.8, 1.2, 1.6]:
                row = summarize("switch_nonlinear", cfg, pivot_p=pivot_p, switch_mag=switch_mag)
                adv = row["nonlinear_cond_advantage"]
                print(
                    "\t".join(
                        [
                            f"{pivot_p:.2f}",
                            f"{switch_mag:.1f}",
                            f"{row['pivot_rate_pi']:.4f}",
                            f"{row['linearity_cond_L']:.4f}",
                            f"{adv:.5f}",
                        ]
                    )
                )
        return
    rows = [
        summarize("linear", cfg, pivot_p=0.02, switch_mag=0.4),
        summarize("piecewise", cfg, pivot_p=0.02, switch_mag=0.9),
        summarize("piecewise", cfg, pivot_p=0.08, switch_mag=1.2),
        summarize("nonlinear", cfg, pivot_p=0.04, switch_mag=0.8),
        summarize("random_walk", cfg, pivot_p=0.02, switch_mag=0.4),
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        keys = [
            "kind",
            "participation_dim",
            "pca_d90",
            "pivot_rate_pi",
            "r2_zk",
            "r2_zk_plus_v",
            "r2_linear_lastk",
            "r2_nonlinear_lastk",
            "linearity_L",
            "r2_cond_linear",
            "r2_cond_nonlinear",
            "linearity_cond_L",
            "mi_tape_given_local_nats_per_dim",
            "mi_traj_given_local_nats_per_dim",
            "mi_resid_given_local_nats_per_dim",
        ]
        print("\t".join(keys))
        for row in rows:
            print("\t".join(str(round(row[k], 4)) if isinstance(row[k], float) else str(row[k]) for k in keys))


if __name__ == "__main__":
    main()
