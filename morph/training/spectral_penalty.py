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


def collect_core_linears(model: nn.Module, include_attn: bool,
                         who: str) -> list[tuple[str, nn.Module, int]]:
    """(name, module, in_features) for the core linears a spectral control acts on.

    ONE enumeration, shared by the soft penalty and the hard projection, so the two cannot
    drift apart about what "the core linears" means.

    The MLP's gate_up and down are the ONLY MortarLinear under a core block (attention's CCA
    projections are separate types), and the MLP is nested in a _KwargSequential, so this
    enumerates by TYPE via named_modules rather than by a hardcoded path.

    Attention is opt-in. Only nn.Linear is eligible there: the CCA convolutions (conv_q_dw
    and friends) are not rank-2 maps on the last dim, so the power iteration's
    [1, in_features] probe does not apply to them, and skipping them in silence would be the
    same class of defect as the unused hook this replaced.
    """
    from morph.model.sparsity import MortarLinear
    root = getattr(model, "_orig_mod", model)
    out: list[tuple[str, nn.Module, int]] = []
    for li, blk in enumerate(root.core):
        for sub_name, sub in blk.named_modules():
            if isinstance(sub, MortarLinear) and getattr(sub, "in_features", None):
                out.append((f"core.{li}.{sub_name}", sub, sub.in_features))
    n_mlp = len(out)
    if not out:
        raise RuntimeError(f"{who} found 0 core MLP linears — enumeration broke; refusing to "
                           f"silently run a no-op.")
    if include_attn:
        for li, blk in enumerate(root.core):
            attn = getattr(blk, "attention", None)
            if attn is None:
                continue
            for sub_name, sub in attn.named_modules():
                if type(sub) is nn.Linear and sub.weight.dim() == 2:
                    out.append((f"core.{li}.attention.{sub_name}", sub, sub.in_features))
        if len(out) == n_mlp:
            raise RuntimeError(
                f"{who}(include_attn=True) found 0 attention linears — enumeration broke; "
                f"refusing to run a control that silently covers less than it claims.")
    return out, n_mlp


def raw_weight(mod: nn.Module) -> torch.Tensor:
    """The tensor the optimizer owns for `mod`, through the wrappers between them.

    Two layers to get past, and a projection that skipped either would be silently wrong:

    * `MortarLinear` HOLDS NO WEIGHT. It delegates to an inner `CMSBlockLinear` at `._cms`;
      asking a MortarLinear for `.weight` raises.
    * MORPH ternarises the core MLP with a `TernarySTE` weight PARAMETRIZATION, so the
      inner module's `.weight` is a computed property and the trainable leaf is
      `parametrizations.weight.original`. Writing to `.weight` would be discarded on the
      next forward.
    """
    inner = getattr(mod, "_cms", None)
    if inner is not None:
        mod = inner
    par = getattr(mod, "parametrizations", None)
    if par is not None and "weight" in par:
        return par["weight"].original
    return mod.weight


class CoreSpectralPenalty:
    """Soft spectral-norm penalty over the core-block linears. Stateless wrt the optimizer.

    MEASURED LIMIT, read this before choosing it over the projection below: a soft
    loss-side hinge is a TUG OF WAR, and it can lose. At `ademamix_alpha_cap` 3.5, batch 12,
    where the control's sigma_max reaches 5.69 by step 1600, `cap 1.5, lambda 10` failed to
    hold sigma at all — 1.49 at step 300, 2.86 at 1200, 4.26 at 1800 — and the arm took over
    at step 1225, EARLIER than its own control's 1700, with validation CE 2.74 nats above its
    minimum against the control's 1.19. Once the excess is large the quadratic hinge's
    gradient dominates the loss and the model optimises the penalty instead of the data.
    Use `CoreSpectralProjection` when the drive is strong.
    """

    def __init__(self, model: nn.Module, cap: float, lam: float, n_iter: int = 1,
                 include_attn: bool = False):
        self.cap = float(cap)
        self.lam = float(lam)
        self.n_iter = int(n_iter)
        self.include_attn = bool(include_attn)
        self._linears, self._n_mlp = collect_core_linears(
            model, self.include_attn, "CoreSpectralPenalty")
        self._v: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _ensure_v(self, name: str, in_features: int, ref: torch.Tensor):
        if name not in self._v:
            g = torch.Generator(device="cpu").manual_seed(hash(name) & 0x7fffffff)
            v = torch.randn(in_features, generator=g).to(device=ref.device, dtype=ref.dtype)
            self._v[name] = v / (v.norm() + 1e-12)

    def sigmas(self) -> dict[str, float]:
        """Diagnostic: current sigma_max per core linear (no grad).

        50 iterations, not 10. Power iteration on a matrix whose top singular values are
        close converges slowly, and 10 from a cold cache UNDER-reads: measured on the tiny
        fixture, 1.891 against a converged 1.947, and on the real model 1.2674 against
        1.4293. The cache warm-starts across calls so a mid-run reading is closer than a
        step-0 one, but the number goes into wandb as `spec/sigma_max` and is read as if it
        were the spectral norm, so it should be one.
        """
        out = {}
        for name, lin, inf in self._linears:
            ref = next(lin.parameters())
            self._ensure_v(name, inf, ref)
            with torch.enable_grad():
                sig, vnew = _power_iter_sigma(lin, self._v[name].to(ref.dtype), max(self.n_iter, 50))
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



class CoreSpectralProjection:
    """HARD projection of the core linears onto the spectral ball, after the optimizer step.

    `W <- W * min(1, cap / sigma_max(W_eff))`, applied in place once per step. Why this and
    not the soft hinge above:

    * **It cannot lose.** The hinge adds a term to the loss and then argues with the data
      gradient about it; the projection is applied afterwards and the constraint holds by
      construction. Measured: at `alpha_cap` 3.5 the hinge at `cap 1.5, lambda 10` let
      sigma_max reach 4.26 by step 1800 and made the arm WORSE than its control.
    * **It does not compete with the objective.** No term is added to the loss, so the
      gradient the model follows is the data's, and `train/loss` stays comparable across
      arms.
    * **It is cheaper.** No autograd through the power iteration, no second backward.

    Scaling the raw weight scales the EFFECTIVE map exactly on MORPH's path: the ternary
    STE is a weight parametrization whose per-tensor scale is `mean(|W|)`, recomputed from
    `W` every forward, so `W -> cW` gives an unchanged code pattern and `W_eff -> c W_eff`.
    `verify=True` re-measures after each projection and RAISES if it did not land on the
    cap, so that assumption cannot rot silently.

    Interaction with the optimizer: this is projected gradient descent. The optimizer's
    moments are left alone — they describe the unprojected step, which is what the next
    step's momentum should be built from.
    """

    def __init__(self, model: nn.Module, cap: float, n_iter: int = 2,
                 include_attn: bool = False, verify: bool = False,
                 warmup_iters: int = 60):
        self.cap = float(cap)
        self.n_iter = max(1, int(n_iter))
        self.include_attn = bool(include_attn)
        self.verify = bool(verify)
        self._linears, self._n_mlp = collect_core_linears(
            model, self.include_attn, "CoreSpectralProjection")
        self._v: dict[str, torch.Tensor] = {}
        # CONVERGE the top singular vectors once, before the first projection. Two
        # iterations from a RANDOM start under-estimate sigma badly — measured on the real
        # model at step 0: 1.2674 against a converged 1.4293, an 11 % under-read, which made
        # the first projection land 13 % above the cap it was asked for. After this warmup
        # the weights move slowly enough that `n_iter` per step tracks.
        self.warmup(warmup_iters)

    @torch.no_grad()
    def warmup(self, n_iter: int = 60) -> None:
        for name, lin, inf in self._linears:
            self._sigma(name, lin, inf, n_iter)

    def _sigma(self, name: str, lin: nn.Module, in_features: int, n_iter: int) -> float:
        w = raw_weight(lin)
        if name not in self._v:
            g = torch.Generator(device="cpu").manual_seed(hash(name) & 0x7fffffff)
            v = torch.randn(in_features, generator=g).to(device=w.device, dtype=w.dtype)
            self._v[name] = v / (v.norm() + 1e-12)
        with torch.enable_grad():
            sig, vnew = _power_iter_sigma(lin, self._v[name].to(w.dtype), n_iter)
        self._v[name] = vnew
        return float(sig.detach())

    def step(self) -> dict[str, float]:
        """Project every over-cap linear. Returns telemetry; call AFTER optimizer.step()."""
        if self.cap <= 0.0:
            return {}
        n_hit, worst = 0, 0.0
        with torch.no_grad():
            for name, lin, inf in self._linears:
                s = self._sigma(name, lin, inf, self.n_iter)
                worst = max(worst, s)
                if s > self.cap:
                    n_hit += 1
                    raw_weight(lin).mul_(self.cap / s)
                    if self.verify:
                        s2 = self._sigma(name, lin, inf, max(self.n_iter, 20))
                        if abs(s2 - self.cap) / self.cap > 0.05:
                            raise RuntimeError(
                                f"CoreSpectralProjection: {name} did not land on the cap — "
                                f"sigma {s:.4f} -> {s2:.4f}, wanted {self.cap:.4f}. Either "
                                f"the power iteration under-read sigma (raise n_iter or "
                                f"warmup_iters) or the effective map is not homogeneous in "
                                f"the raw weight on this path (CMSBlockLinear.enable_ternary "
                                f"freezes its scale in a buffer, and is not). Either way the "
                                f"projection is not enforcing what it claims.")
        return {"specproj/n_projected": float(n_hit),
                "specproj/sigma_max_preproj": worst,
                "specproj/frac_projected": n_hit / len(self._linears)}



class CoreIsometryPenalty:
    """Flatten the core linears' SPECTRUM instead of shrinking it.

    Why a second instrument at all. The measurements say the core map barely changes size
    across the onset (+2.5 % isotropic per-block gain) while its blocks' amplifying
    directions ALIGN (x2.9). A spectral cap acts on size, and a uniform rescale `W -> cW`
    leaves every singular VECTOR and every RATIO `sigma_i / sigma_j` untouched — so it
    cannot slow the alignment at all, only lower the gain once aligned. Measured, and it
    was not enough: `a35-proj15` held `sigma_max` at exactly 1.50 for its whole life and
    still took over, with a realized per-block gain of 1.659 against its uncapped control's
    1.402.

    What governs the alignment RATE is the spectral GAP: power iteration converges onto the
    top direction like `(sigma_1 / sigma_2)^k`, and it cannot converge at all on a flat
    spectrum. So penalise the spread:

        g_k = ||W v_k||^2 / ||v_k||^2   for m random directions v_k
        L   = mu * sum_i  Var_k(g_k) / mean_k(g_k)^2

    Zero exactly when every direction is amplified equally, i.e. when `W` is a scaled
    isometry — the classical dynamical-isometry condition. Dividing by the (detached) mean
    squared makes the term SCALE-FREE: it constrains the shape of the spectrum and says
    nothing about its size, which is the point, since size was measured not to be the lever.

    Both MLP shapes can satisfy it. `gate_up` is tall (5632 x 1024) so `W^T W = c^2 I` is
    reachable; `down` is wide (1024 x 2816) so `W W^T = c^2 I` is, and for a wide `W` with
    orthogonal rows `||W v||^2 / ||v||^2` concentrates as the dimensions grow, which is what
    the estimator sees.

    Cost is m matrix-vector products per linear per step, through the module's own forward
    so the ternary STE is applied live — a few MFLOP against the model's ~10 TFLOP step.
    """

    def __init__(self, model: nn.Module, mu: float, n_probe: int = 8,
                 include_attn: bool = False, seed: int = 0):
        self.mu = float(mu)
        self.n_probe = max(2, int(n_probe))
        self.include_attn = bool(include_attn)
        self._linears, self._n_mlp = collect_core_linears(
            model, self.include_attn, "CoreIsometryPenalty")
        self._step = 0
        self.seed = int(seed)

    def _gains(self, lin: nn.Module, in_features: int, ref: torch.Tensor) -> torch.Tensor:
        """[m] amplification of m random unit directions, differentiable wrt the weights."""
        g = torch.Generator(device="cpu").manual_seed(
            (self.seed * 1_000_003 + self._step) & 0x7FFFFFFF)
        v = torch.randn(self.n_probe, in_features, generator=g).to(
            device=ref.device, dtype=ref.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        wv = lin(v)                                   # [m, out]
        return (wv.float() ** 2).sum(dim=1)

    def spread(self) -> dict[str, float]:
        """Diagnostic: per-linear coefficient of variation of the probed gains (no grad)."""
        out = {}
        with torch.no_grad():
            for name, lin, inf in self._linears:
                gk = self._gains(lin, inf, next(lin.parameters()))
                out[name] = float(gk.std(unbiased=False) / gk.mean().clamp_min(1e-12))
        return out

    def penalty(self) -> torch.Tensor:
        """mu * sum_i Var(g)/mean(g)^2 — differentiable. mu=0 returns an exact zero."""
        ref = next(self._linears[0][1].parameters())
        total = torch.zeros((), device=ref.device, dtype=torch.float32)
        if self.mu == 0.0:
            return total
        for _name, lin, inf in self._linears:
            gk = self._gains(lin, inf, next(lin.parameters()))
            m = gk.mean()
            total = total + gk.var(unbiased=False) / m.detach().clamp_min(1e-12) ** 2
        self._step += 1
        return self.mu * total


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
