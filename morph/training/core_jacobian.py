"""Realized spectral norm of the looped core map's Jacobian — sigma_max(J_core).

MORPH is a NESTED dynamical system (CLAUDE.md, and
`.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md`): the
optimizer sees only `grad_theta L`, which integrates over the inner trajectory
`h_{k+1} = f_theta(h_k)` and discards its structure. It is therefore blind to the inner
map's contractivity. Everything already logged about the core loop is a magnitude —
`core_gain` is `||h_new|| / ||h||` along the one direction the data happened to pick, and
`spec/sigma_max` is `sigma_max(W)` of a single weight matrix, an upper bound on one factor
of one block. Neither is the operator norm of the map.

This module measures the operator norm itself:

    sigma_max(J) = max over v of ||J v|| / ||v||,   J = d f_theta / d h  at the live h.

Two readings that the existing logs cannot tell apart:

* **magnitude** — sigma_max(J) grows, so the map itself became expansive; a spectral cap
  on the weights bounds it.
* **alignment** — sigma_max(J) was always above 1 and the realized backward direction
  rotated INTO the top singular direction; the weights need not have moved at all, and a
  spectral cap on their magnitude cannot help.

`sigma_max` plus the realized gain separates them, which is why both are reported.

Method. Power iteration on `J^T J` at the captured operating point. `J v` comes from the
double-backward identity (`g = J^T u` is linear in `u`, so `d g / d u` applied to `v` is
`J v`) — exact, no finite differences, and it needs no forward-mode rule, which matters
because the core's attention is a stack of custom modules. One forward builds the graph;
every power iteration then reuses it, so the cost is one forward plus `2 * n_iter`
backward passes over ONE core step.

Everything runs in fp32 with autocast disabled. Training runs the core in bf16, whose ~3
decimal digits cannot support power iteration; the operator being measured is the fp32
master-weight operator either way.

Restriction to the active set. A frozen slot's state is not updated, and a pad slot is not
real, so the probe measures `M J M` where `M` masks to the positions that iteration is
actually updating. Without the mask the top singular direction can sit entirely in the pad
subspace and the number means nothing.

Validation gate: `python -m morph.training.core_jacobian` → CORE_JACOBIAN_GATE_PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class JacobianResult:
    """One measurement of the core map at one operating point."""

    iter_idx: int
    sigma_step: float
    """sigma_max(J) of the WHOLE core step (injection + all n_core blocks)."""
    rms_step: float = float("nan")
    """Typical gain ||J||_F / sqrt(n) of the WHOLE core step — the gain a generic
    direction sees, which is what the realized backward gain measures."""
    sigma_blocks: list[float] = field(default_factory=list)
    """sigma_max(J) of each core block on its own, at that block's live input."""
    rms_blocks: list[float] = field(default_factory=list)
    """Typical gain of each core block on its own."""
    rel_change: float = float("nan")
    """|sigma_k - sigma_{k-1}| / sigma_k on the last power iteration — convergence."""
    n_iter: int = 0

    @property
    def block_product(self) -> float:
        """Product of the per-block sigmas — the submultiplicative bound on sigma_step."""
        p = 1.0
        for s in self.sigma_blocks:
            p *= s
        return p

    @property
    def rms_block_gain(self) -> float:
        """Geometric mean of the per-block typical gains — the operator-side counterpart
        of `preclip/core_block_gain`, which fits one uniform gain across the blocks."""
        if not self.rms_blocks:
            return float("nan")
        p = 1.0
        for s in self.rms_blocks:
            p *= max(s, 1e-30)
        return p ** (1.0 / len(self.rms_blocks))


def _jacobian_stats(fn, h0: Tensor, mask: Tensor, n_iter: int, seed: int,
                    n_probe: int = 8) -> tuple[float, float, float]:
    """Spectral norm AND typical gain of `d fn / d h` at `h0`, restricted to `mask`.

    Returns `(sigma_max, rel_change_on_the_last_power_iteration, rms_gain)`.

    Both numbers are needed and they answer different questions. `sigma_max` is the
    WORST-CASE amplification over all directions; a residual block can carry a large one
    in a direction the data never occupies (a slot whose carrier norm is small makes the
    RMSNorm Jacobian large there), so on its own it over-reads. `rms_gain` is
    `sqrt(E ||J v||^2 / ||v||^2)` over isotropic `v`, i.e. `||J||_F / sqrt(n)`, the gain a
    TYPICAL direction sees — and the realized backward gain the trainer logs is a typical
    direction, not the worst one. `sigma_max >= rms_gain` always; the two together say
    whether an expansive direction exists and whether the operator is expansive on
    average.

    `fn` must map a tensor shaped like `h0` to a tensor shaped like `h0`. `mask`
    broadcasts against `h0`.
    """
    h = h0.detach().float().requires_grad_(True)
    y = fn(h)
    if y.shape != h.shape:
        raise RuntimeError(f"core map is not an endomorphism: in {tuple(h.shape)} "
                           f"out {tuple(y.shape)} — the probe assumes h -> h")
    y = y.float()

    # u is the cotangent slot. `g = J^T u` is LINEAR in u, so d g / d u contracted with v
    # is exactly `J v` — the standard double-backward jvp. create_graph keeps the graph of
    # g so that second grad is available; retain_graph keeps BOTH graphs alive across every
    # power iteration, so the forward runs once.
    u = torch.zeros_like(y, requires_grad=True)
    (g,) = torch.autograd.grad(y, h, grad_outputs=u, create_graph=True)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(h0.shape, generator=gen).to(device=h0.device, dtype=torch.float32)
    v = v * mask
    v = v / (v.norm() + 1e-12)

    sigma = float("nan")
    prev = float("nan")
    rel = float("nan")
    for _ in range(max(1, n_iter)):
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v, retain_graph=True)
        jv = jv * mask
        s = float(jv.norm())
        prev, sigma = sigma, s
        if s < 1e-20:                      # the map annihilates this direction
            return 0.0, 0.0, 0.0
        (w,) = torch.autograd.grad(y, h, grad_outputs=jv, retain_graph=True)
        w = w * mask
        wn = w.norm()
        if float(wn) < 1e-20:
            break
        v = w / wn
        if prev == prev:                   # not NaN → we have two iterates to compare
            rel = abs(sigma - prev) / max(sigma, 1e-20)

    # Typical gain: Hutchinson over isotropic directions inside the mask. n is the number
    # of UNMASKED coordinates, which is what the estimate is normalised by.
    n_free = float(mask.expand_as(h0).sum())
    acc = 0.0
    for k in range(max(1, n_probe)):
        gk = torch.Generator(device="cpu").manual_seed(seed + 9973 * (k + 1))
        p = torch.randn(h0.shape, generator=gk).to(device=h0.device, dtype=torch.float32)
        p = p * mask
        (jp,) = torch.autograd.grad(g, u, grad_outputs=p, retain_graph=True)
        jp = jp * mask
        acc += float((jp * jp).sum()) / max(float((p * p).sum()), 1e-30)
    rms = (acc / max(1, n_probe)) ** 0.5
    return sigma, rel, rms


class CoreJacobianProbe:
    """sigma_max(J_core) at the operating points a live forward actually visited.

    Usage — the capture site in `MORPHTransformer._core_region` is
    Python-level no-ops until `_jac_capture` is a list:

        probe = CoreJacobianProbe(model)
        with probe.capture() as points:
            with torch.no_grad():
                model(**batch)
        results = [probe.measure(p) for p in points if p["iter_idx"] in (0, 3)]

    The forward may be `no_grad`; the probe rebuilds its own graph from the detached
    operating point.
    """

    def __init__(self, model: nn.Module, n_iter: int = 12, seed: int = 0,
                 per_block: bool = True):
        self.root = getattr(model, "_orig_mod", model)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.per_block = bool(per_block)

    # ── capture ────────────────────────────────────────────────────────────────────
    def capture(self):
        """Context manager attaching a capture list to the model. Returns the list."""
        probe = self

        class _Ctx:
            def __enter__(self):
                self.points: list[dict] = []
                probe.root._jac_capture = self.points
                return self.points

            def __exit__(self, *exc):
                probe.root._jac_capture = None
                return False

        return _Ctx()

    # ── measurement ────────────────────────────────────────────────────────────────
    def measure(self, point: dict) -> JacobianResult:
        """sigma_max of the core map's Jacobian at one captured operating point."""
        root = self.root
        h0 = point["h"]
        e = point["e"].float()
        inj = point["inj"].float()
        ret = point["ret_state"]
        t = int(point["iter_idx"])
        # active is [B, L]; h is [B, L, n, C] (hyper-connection) or [B, L, C].
        mask = point["active"].view(*point["active"].shape,
                                    *([1] * (h0.dim() - 2))).to(torch.float32)

        def step(h: Tensor) -> Tensor:
            out, _ = root._apply_core_step(h, e, None, None, None, ret_state=ret,
                                           iter_idx=t, inj_terms=inj)
            return out

        with torch.autocast("cuda", enabled=False):
            sigma, rel, rms = _jacobian_stats(step, h0, mask, self.n_iter, self.seed)
            blocks: list[float] = []
            rms_blocks: list[float] = []
            if self.per_block:
                blocks, rms_blocks = self._block_sigmas(h0, e, inj, ret, t, mask)
        return JacobianResult(iter_idx=t, sigma_step=sigma, rms_step=rms,
                              sigma_blocks=blocks, rms_blocks=rms_blocks,
                              rel_change=rel, n_iter=self.n_iter)

    def _block_sigmas(self, h0, e, inj, ret, t, mask) -> tuple[list[float], list[float]]:
        """sigma_max of each core block on its own, each at the input it really sees.

        The blocks are walked in forward order and the live input is carried forward, so
        block i is probed at the state block i-1 handed it — not at the step's input. The
        product of these is the submultiplicative bound on the whole step's sigma.
        """
        root = self.root
        np_ = root.cfg.n_prelude
        mlp_kw = {"iter_idx": t}
        out: list[float] = []
        rms_out: list[float] = []
        with torch.no_grad():
            h = root.injection(h0.float(), e)
        for i, layer in enumerate(root.core):
            is_ret = root._core_has_retention and (i in root._retention_layers)
            term = inj[i]

            def one_block(hh, _layer=layer, _term=term, _is_ret=is_ret):
                hh = root._apply_injection(hh, _term)
                cap = {} if _is_ret else None
                return _layer(hh, mlp_kwargs=mlp_kw,
                              ret_state=ret if _is_ret else None, ret_capture=cap)

            s, _, r = _jacobian_stats(one_block, h, mask, self.n_iter, self.seed + i)
            out.append(s)
            rms_out.append(r)
            with torch.no_grad():
                h = one_block(h.float()).detach()
        return out, rms_out


# ──────────────────────────────────────────────────────────────────────────────────────
def _gate() -> None:
    """Validate the power iteration against maps whose spectral norm is known exactly."""
    torch.manual_seed(0)
    print("=== CoreJacobianProbe gate ===")
    ok = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Gate 1: a LINEAR map h -> W h. sigma_max(J) = sigma_max(W), known from svdvals.
    n = 48
    W = torch.randn(n, n, device=dev, dtype=torch.float64)
    W = W.float()
    true = float(torch.linalg.svdvals(W)[0])
    h0 = torch.randn(1, 4, n, device=dev)
    mask = torch.ones(1, 4, 1, device=dev)
    est, rel, _ = _jacobian_stats(lambda h: h @ W.t(), h0, mask, 60, seed=0)
    e1 = abs(est - true) / true
    g1 = e1 < 1e-4
    print(f"  [Gate1] linear map: est={est:.6f} true={true:.6f} relerr={e1:.2e} "
          f"conv={rel:.1e} -> {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # Gate 2: a NONLINEAR residual map h -> h + tanh(h W)/4 at h0 = 0, where tanh'(0) = 1
    # so J = I + W/4 exactly, and sigma_max is again known. A residual Jacobian has its
    # singular values CLUSTERED around 1 (here sigma_1/sigma_2 ~ 1.01), so power iteration
    # needs many passes — that is a property of the operator, not of the estimator, and it
    # is why `rel_change` is reported with every real measurement.
    W2 = torch.randn(n, n, device=dev) / n ** 0.5
    J = torch.eye(n, device=dev) + W2.t() / 4.0
    true2 = float(torch.linalg.svdvals(J)[0])
    h0b = torch.zeros(1, 4, n, device=dev)
    est2, rel2, _ = _jacobian_stats(lambda h: h + torch.tanh(h @ W2.t()) / 4.0, h0b, mask, 600, seed=1)
    e2 = abs(est2 - true2) / true2
    g2 = e2 < 1e-4
    print(f"  [Gate2] nonlinear residual at 0: est={est2:.6f} true={true2:.6f} "
          f"relerr={e2:.2e} -> {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # Gate 3: the mask really restricts the operator. Build a map that amplifies 10x on
    # position 1 and 0.5x on position 0; masking to position 0 must report 0.5, not 10.
    scale = torch.tensor([0.5, 10.0, 1.0, 1.0], device=dev).view(1, 4, 1)
    m0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).view(1, 4, 1)
    est3, _, _ = _jacobian_stats(lambda h: h * scale, torch.randn(1, 4, n, device=dev), m0, 40, seed=2)
    g3 = abs(est3 - 0.5) < 1e-5
    print(f"  [Gate3] masked to a 0.5x position: est={est3:.6f} expect 0.500000 "
          f"-> {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    # Gate 4: the typical-gain estimator. For a linear map the exact value is
    # ||W||_F / sqrt(n), so the Hutchinson estimate must land on it within its own
    # sampling error (~1/sqrt(2*n_probe*n) relative, tiny here because n = 4*48).
    Wf = torch.randn(n, n, device=dev) / 3.0
    true4 = float(Wf.norm() / n ** 0.5)
    _, _, rms4 = _jacobian_stats(lambda h: h @ Wf.t(), torch.randn(1, 4, n, device=dev),
                                 mask, 3, seed=3, n_probe=32)
    e4 = abs(rms4 - true4) / true4
    g4 = e4 < 5e-3
    print(f"  [Gate4] typical gain: est={rms4:.6f} true={true4:.6f} relerr={e4:.2e} "
          f"-> {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    print("CORE_JACOBIAN_GATE_PASS" if ok else "CORE_JACOBIAN_GATE_FAIL")


if __name__ == "__main__":
    _gate()
