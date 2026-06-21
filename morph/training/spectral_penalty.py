"""Core-map spectral-norm penalty — soft hinge on σ_max of core MLP linears.

The core-map's worst-case one-step amplification (Jacobian spectral norm) can grow
without bound under certain optimizer/α configurations, destabilizing the looped core.
The healthy regime runs at σ_max~22 (AdamW stable); the target is not to force σ_max≤1
(that lobotomizes the working model) but to prevent runaway above the healthy operating point.

This module adds a soft per-core-linear spectral-norm penalty:
    L_sn = λ · Σ_i  relu(σ_max(W_i) − cap)²
over the core-block MLP linears (gate_up, down). The penalty is zero while every linear
sits below `cap` (healthy training bit-exact at λ=0) and only activates to pull a linear
back when it tries to run away. Loss-side regularizer → optimizer-agnostic.

σ_max(W_i) is estimated by power iteration THROUGH the linear's own forward() — W·v = lin(v), Wᵀ·u via
autograd — so it sees the EFFECTIVE weight (ternary STE + mask + dense/carve mode) with zero dependence
on the internal layout. Only first-order in W (σ = ‖W·v_top‖ with v_top fixed) → NO double-backward.
Top singular vectors are cached + warm-started across steps (1 iter/step converges).

Validation gate: `python -m morph.training.spectral_penalty` → SPECTRAL_PENALTY_GATE_PASS.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _power_iter_sigma(lin: nn.Module, v: torch.Tensor, n_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """σ_max of the linear map implemented by `lin` (bias-free), via forward+autograd power iteration.

    lin: a module computing y = W x on the last dim (effective weight — ternary/mask/mode applied in
         forward). v: cached unit top-right-singular-vector estimate [in_features] (detached).
    Returns (sigma [scalar, DIFFERENTIABLE wrt lin's params], v_new [in_features, detached]).
    """
    # Power iteration on WᵀW (no_grad-ish: each step uses an isolated autograd graph on v only).
    v = v.detach()
    for _ in range(max(1, n_iter)):
        v = v / (v.norm() + 1e-12)
        v = v.requires_grad_(True)
        wv = lin(v.unsqueeze(0)).squeeze(0)                 # W v   [out]
        # Wᵀ(Wv) = ∇_v ½‖Wv‖²  → gives the next WᵀW v iterate (grad wrt v only; W is a leaf).
        wtwv = torch.autograd.grad(0.5 * (wv * wv).sum(), v)[0]   # [in]
        v = wtwv.detach()
    v = (v / (v.norm() + 1e-12)).detach()                   # converged top right singular vector
    # σ = ‖W v_top‖ with v_top FIXED → differentiable wrt W (first-order, no double-backward).
    wv = lin(v.unsqueeze(0)).squeeze(0)
    sigma = (wv * wv).sum().clamp_min(1e-24).sqrt()
    return sigma, v


class CoreSpectralPenalty:
    """Soft spectral-norm penalty over the core-block MLP linears. Stateless wrt the optimizer."""

    def __init__(self, model: nn.Module, cap: float, lam: float, n_iter: int = 1,
                 include_attn: bool = False):
        from morph.model.sparsity import MortarLinear
        root = getattr(model, "_orig_mod", model)
        self.cap = float(cap)
        self.lam = float(lam)
        self.n_iter = int(n_iter)
        self._linears: list[tuple[str, nn.Module, int]] = []   # (name, module, in_features)
        # Collect the core-block MLP linears (gate_up, down) — they are the ONLY MortarLinear under
        # each core block (attention's CCA projections are separate types). The MLP is nested in a
        # _KwargSequential, so enumerate by TYPE via named_modules rather than a hardcoded path.
        for li, blk in enumerate(root.core):
            for sub_name, sub in blk.named_modules():
                if isinstance(sub, MortarLinear) and getattr(sub, "in_features", None):
                    self._linears.append((f"core.{li}.{sub_name}", sub, sub.in_features))
        if not self._linears:
            raise RuntimeError("CoreSpectralPenalty found 0 core MLP linears — enumeration broke; "
                               "refusing to silently run a no-op penalty.")
        # NOTE: attention (CCA) projections deliberately excluded — M3 located the runaway in the
        # core MLP (gate_up); attn is multi-projection/quantized and far harder to spectral-bound.
        # include_attn left as a future hook (currently unused) — flagged so it isn't silently missing.
        self._v: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _ensure_v(self, name: str, in_features: int, ref: torch.Tensor):
        if name not in self._v:
            g = torch.Generator(device="cpu").manual_seed(hash(name) & 0x7fffffff)
            v = torch.randn(in_features, generator=g).to(device=ref.device, dtype=ref.dtype)
            self._v[name] = v / (v.norm() + 1e-12)

    def sigmas(self) -> dict[str, float]:
        """Diagnostic: current σ_max per core linear (no grad)."""
        out = {}
        for name, lin, inf in self._linears:
            ref = next(lin.parameters())
            self._ensure_v(name, inf, ref)
            with torch.enable_grad():
                sig, vnew = _power_iter_sigma(lin, self._v[name].to(ref.dtype), max(self.n_iter, 10))
            self._v[name] = vnew
            out[name] = float(sig.detach())
        return out

    def penalty(self) -> torch.Tensor:
        """L_sn = λ·Σ relu(σ_i − cap)² — differentiable wrt the core MLP weights. λ=0 → exact 0."""
        ref = next(self._linears[0][1].parameters())
        total = torch.zeros((), device=ref.device, dtype=torch.float32)
        if self.lam == 0.0:
            return total
        for name, lin, inf in self._linears:
            p = next(lin.parameters())
            self._ensure_v(name, inf, p)
            sig, vnew = _power_iter_sigma(lin, self._v[name].to(p.dtype), self.n_iter)
            self._v[name] = vnew
            over = (sig.float() - self.cap).clamp_min(0.0)
            total = total + over * over
        return self.lam * total


# ──────────────────────────────────────────────────────────────────────────────────────────────
def _gate():
    torch.manual_seed(0)
    print("=== CoreSpectralPenalty gate ===")
    ok = True

    # Gate 1: power-iter σ_max matches svdvals on a plain bias-free Linear.
    lin = nn.Linear(64, 96, bias=False).double()
    v0 = torch.randn(64, dtype=torch.float64)
    with torch.enable_grad():
        sig, _ = _power_iter_sigma(lin, v0, n_iter=200)
    true = torch.linalg.svdvals(lin.weight)[0].item()
    e = abs(sig.item() - true) / true
    g1 = e < 1e-3
    print(f"  [Gate1] σ_max power-iter vs svdvals: est={sig.item():.5f} true={true:.5f} relerr={e:.2e} "
          f"→ {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # Gate 2: σ is differentiable wrt W and the gradient reduces σ_max (penalty actually bites).
    # Use a NON-degenerate spectrum (one dominant singular value 5, rest ~1) so reducing σ_max is
    # well-posed (an all-equal orthogonal×5 spectrum would just expose a new direction at 5 each step).
    U, _ = torch.linalg.qr(torch.randn(48, 48))
    V, _ = torch.linalg.qr(torch.randn(48, 48))
    svals = torch.ones(48); svals[0] = 5.0
    lin2 = nn.Linear(48, 48, bias=False)
    with torch.no_grad():
        lin2.weight.copy_(U @ torch.diag(svals) @ V.t())   # σ_max = 5, next = 1
    cap = 2.0
    v = torch.randn(48) / 48 ** 0.5
    with torch.enable_grad():
        s0, v = _power_iter_sigma(lin2, v, 50)
    sig_before = float(s0.detach())
    opt = torch.optim.SGD(lin2.parameters(), lr=0.2)
    for _ in range(80):
        opt.zero_grad()
        s, v = _power_iter_sigma(lin2, v.detach(), 3)
        pen = (s - cap).clamp_min(0.0) ** 2
        pen.backward()
        opt.step()
    with torch.enable_grad():
        s_after, _ = _power_iter_sigma(lin2, v.detach(), 50)
    s_after = float(s_after.detach())
    g2 = s_after < sig_before - 1.5 and s_after < cap + 0.5     # pulled from 5 toward cap=2
    print(f"  [Gate2] penalty reduces σ_max: before={sig_before:.3f} after={s_after:.3f} "
          f"(cap={cap}) → {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # Gate 3: below cap → penalty exactly 0 (healthy training untouched).
    lin3 = nn.Linear(32, 32, bias=False)
    nn.init.orthogonal_(lin3.weight)     # σ_max ≈ 1
    v = torch.randn(32) / 32 ** 0.5
    s, _ = _power_iter_sigma(lin3, v, 50)
    pen0 = (s.float() - 10.0).clamp_min(0.0) ** 2     # cap=10 ≫ σ≈1
    g3 = pen0.item() == 0.0
    print(f"  [Gate3] σ≈{float(s):.3f} < cap=10 → penalty={pen0.item():.3e} → {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    print("SPECTRAL_PENALTY_GATE_PASS" if ok else "SPECTRAL_PENALTY_GATE_FAIL")


if __name__ == "__main__":
    _gate()
