"""AdEMAMix with β1=0 and blockwise-8bit state — memory parity with AdamW8bit.

WHY THIS EXISTS
---------------
bnb's AdEMAMix8bit allocates 3 state buffers (m1, m2, ν) regardless of β1 — its β1=0
"memory trick" is a no-op (measured: 3.05 B/param vs AdamW8bit 2.03). With β1=0 the fast
EMA m1 collapses to the raw gradient and needs NO buffer, so a faithful β1=0 implementation
keeps only 2 blockwise-8bit buffers (m2, ν) → ~2 B/param, matching AdamW8bit.

We reuse bnb's OWN blockwise dynamic quantization (bnb.functional.quantize_blockwise /
dequantize_blockwise, blocksize=256, dynamic qmap) so the quantization is numerically
identical to bnb's optimizers — only the buffer COUNT and the update math differ.

UPDATE (β1=0 AdEMAMix, arXiv:2409.03137):
    m2 ← β3·m2 + (1−β3)·g                  (slow EMA)
    ν  ← β2·ν  + (1−β2)·g²                  (second moment)
    bc2 = 1 − β2^t
    update = (g + α·m2) / √(ν/bc2 + ε) + λ·p       # m1 = g exactly (β1=0, bc1=1); ε INSIDE sqrt
    p ← p − lr·update

STABILITY SCHEDULERS (the load-bearing part for instability-sensitive models):
    α and β3 warm up over t_alpha / t_beta3 steps (NOT LR warmup). A large β3 active from
    step 0 diverges. The paper's β3 schedule warms the half-life from β1's — but β1=0 makes
    that log(β1)=−∞ degenerate, so we decouple the warmup START via `beta3_warmup_start`
    (≈0.9): the slow EMA still ramps gently even with no m1 buffer.

This is the DE-FUSED reference (dequant→update→requant per param). It is correct + hits the
memory target; if its de-fused launch overhead is material vs AdamW8bit we fuse into one
Triton kernel (the reference is then the bit-faithful ground truth for that kernel's gate).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import bitsandbytes.functional as bnbF

__all__ = ["AdEMAMixB1Zero"]


class AdEMAMixB1Zero(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float, float] = (0.0, 0.999, 0.9999),
        alpha: float = 8.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        beta3_warmup_start: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bits: int = 8,
        min_8bit_size: int = 4096,
        blocksize: int = 256,
        fused: bool = True,
    ):
        if betas[0] != 0.0:
            raise ValueError(f"AdEMAMixB1Zero requires β1=0, got betas={betas}")
        if bits not in (8, 32):
            raise ValueError(f"bits must be 8 or 32, got {bits}")
        defaults = dict(
            lr=lr, betas=betas, alpha=alpha, t_alpha=t_alpha, t_beta3=t_beta3,
            beta3_warmup_start=beta3_warmup_start, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.bits = bits
        self.min_8bit_size = min_8bit_size
        self.blocksize = blocksize
        self.fused = fused
        # dynamic qmaps (lazily moved to each param's device); signed for m2, unsigned for ν.
        self._code_signed_cpu = bnbF.create_dynamic_map(signed=True)
        self._code_unsigned_cpu = bnbF.create_dynamic_map(signed=False)
        self._code_cache: dict = {}

    def _code(self, device, signed: bool):
        key = (device, signed)
        c = self._code_cache.get(key)
        if c is None:
            src = self._code_signed_cpu if signed else self._code_unsigned_cpu
            c = src.to(device)
            self._code_cache[key] = c
        return c

    @staticmethod
    def _sched(step, group):
        """Return (alpha_t, beta2, beta3_t) with decoupled α / β3 warmup."""
        b1, b2, b3 = group["betas"]
        alpha = group["alpha"]
        ta, tb = group["t_alpha"], group["t_beta3"]
        if ta:
            alpha = min(step * alpha / ta, alpha)
        if tb:
            bs = group["beta3_warmup_start"]
            ln_s, ln_3 = math.log(bs), math.log(b3)
            s = step / tb
            denom = (1.0 - s) * ln_3 + s * ln_s
            b3 = min(math.exp((ln_s * ln_3) / denom), b3) if denom != 0 else b3
        return alpha, b2, b3

    def _deq(self, q_flat, absmax, signed, p):
        code = self._code(p.device, signed)
        a = bnbF.dequantize_blockwise(q_flat, absmax=absmax, code=code,
                                      blocksize=self.blocksize)
        return a.view_as(p).float()

    def _q(self, t, signed, p):
        code = self._code(p.device, signed)
        q, qs = bnbF.quantize_blockwise(t.reshape(-1).contiguous(), code=code,
                                        blocksize=self.blocksize)
        return q, qs.absmax

    @staticmethod
    def _migrate_linear_nu_to_sqrt(st: dict, p: torch.Tensor, blocksize: int) -> None:
        """Convert legacy fused ν state from linear ν codes to sqrt(ν) codes in-place."""
        if "nu_sqrt_code" in st or "nu_code" not in st:
            return
        n = p.numel()
        nblocks = (n + blocksize - 1) // blocksize
        old_code = st.pop("nu_code").reshape(-1).float()
        old_amax = st.pop("nu_amax").float()
        scale = (old_amax / 127.0).repeat_interleave(blocksize)[:n]
        nu = torch.clamp(old_code[:n] * scale, min=0.0)
        nu_sqrt = torch.sqrt(nu)
        pad = nblocks * blocksize - n
        if pad:
            nu_sqrt = torch.cat([nu_sqrt, torch.zeros(pad, device=p.device)])
        blocks = nu_sqrt.view(nblocks, blocksize)
        amax = blocks.max(dim=1).values
        q = torch.round(blocks / (amax[:, None] / 127.0 + 1e-20)).clamp(0, 127)
        q = torch.where((blocks > 0) & (amax[:, None] > 0) & (q == 0), torch.ones_like(q), q)
        st["nu_sqrt_code"] = q.reshape(-1)[:n].to(torch.int8).contiguous()
        st["nu_sqrt_amax"] = amax.contiguous()

    def _fused_step(self, params, lr, beta2, beta3, alpha, eps, bc2, wd):
        """Fused Triton path for 8-bit large params (linear-int8 blockwise state).

        State per param: int8 code tensors (m2_code, nu_sqrt_code) + fp32 per-block
        absmax (m2_amax, nu_sqrt_amax). On first touch the state is allocated and
        is_init=True is passed so the kernel treats m2/nu as zero. Legacy checkpoints
        with linear ν `nu_code` are migrated once before the kernel launch.
        """
        from morph.training.ademamix_b1zero_kernel import (
            fused_ademamix_b1zero_step, BLOCK,
        )

        for p in params:
            st = self.state[p]
            g = p.grad
            # grads must be fp32 contiguous flat for the kernel
            g_flat = g.reshape(-1).float().contiguous()
            # p_flat must be a VIEW into p's storage so in-place kernel writes propagate.
            # view(-1) raises if p is non-contiguous → make it contiguous in place first.
            if not p.is_contiguous():
                p.data = p.data.contiguous()
            p_flat = p.view(-1)
            n = p_flat.numel()
            nblocks = (n + BLOCK - 1) // BLOCK

            is_init = len(st) == 0 or st.get("init", False)
            if is_init:
                st.pop("init", None)
                st["m2_code"] = torch.zeros(n, dtype=torch.int8, device=p.device)
                st["nu_sqrt_code"] = torch.zeros(n, dtype=torch.int8, device=p.device)
                st["m2_amax"] = torch.zeros(nblocks, dtype=torch.float32, device=p.device)
                st["nu_sqrt_amax"] = torch.zeros(nblocks, dtype=torch.float32, device=p.device)
            else:
                self._migrate_linear_nu_to_sqrt(st, p, BLOCK)

            fused_ademamix_b1zero_step(
                p_flat, g_flat,
                st["m2_code"], st["m2_amax"], st["nu_sqrt_code"], st["nu_sqrt_amax"],
                lr, beta2, beta3, alpha, eps, bc2, wd, is_init,
            )

    def _mask_dead_state(self, p) -> None:
        """Zero the slow-momentum (m₂) and second-moment (ν) state at PRUNED positions.

        WHY (the prune-divergence fix, 2026-06-15): MORPH prunes by masking a tile's
        weight + grad to 0 every step (CMSBlockLinear.apply_prune_mask), which tags the
        param with `_dead_mask` (1=keep, 0=pruned). For AdamW that's enough — grad=0 ⇒
        m,ν both decay ⇒ update→0. For AdEMAMix it is NOT: the α·m₂ term is driven by the
        SLOW EMA, which retains pre-prune gradient mass and decays at β3 ≫ β2 while ν
        collapses → (α·m₂)/(√(ν/bc2)+ε) EXPLODES for the dead param (measured: stock
        diverged @14.2k, faster-β2 @9.4k — bigger β3/β2 gap = earlier blow-up). Zeroing
        m₂/ν here, BEFORE the update, makes the update exactly 0 at dead positions
        ((0+α·0)/(√0+ε)+λ·0 = 0), so pruned params stay dead. grad stays masked every
        step ⇒ the zeroed state stays zero (m₂ ← β3·0 + (1-β3)·0).
        """
        keep = getattr(p, "_dead_mask", None)
        if keep is None:
            return
        st = self.state.get(p)
        if not st or st.get("init"):
            return  # no state yet (first step) → m₂/ν already treated as 0
        dead = (keep.reshape(-1).to(p.device) == 0)
        if not bool(dead.any()):
            return
        if "m2_code" in st:            # fused linear-int8: code 0 → value 0 (exact)
            st["m2_code"].view(-1)[dead] = 0
            if "nu_sqrt_code" in st:
                st["nu_sqrt_code"].view(-1)[dead] = 0
            elif "nu_code" in st:      # legacy checkpoint before sqrt-ν migration
                st["nu_code"].view(-1)[dead] = 0
        elif "m2_q" in st:             # de-fused dynamic-map: code 0 ≠ 0, so dequant→0→requant
            m2 = self._deq(st["m2_q"], st["m2_amax"], True, p); m2.view(-1)[dead] = 0
            nu = self._deq(st["nu_q"], st["nu_amax"], False, p); nu.view(-1)[dead] = 0
            st["m2_q"], st["m2_amax"] = self._q(m2, True, p)
            st["nu_q"], st["nu_amax"] = self._q(nu, False, p)
        if "m2" in st:                 # fp32 fallback
            st["m2"].view(-1)[dead] = 0
            st["nu"].view(-1)[dead] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            alpha_t, beta2, beta3_t = self._sched(step, group)
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            bc2 = 1.0 - beta2 ** step
            grp_bits = group.get("optim_bits", self.bits)

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # ── Prune compatibility: zero slow state at pruned positions BEFORE update ──
            # (see _mask_dead_state). Without this AdEMAMix's α·m₂ term drives grad-masked
            # pruned params to divergence as ν collapses underneath the charged m₂.
            for p in params:
                self._mask_dead_state(p)

            # ── Split: fused-kernel path (8bit + large enough) vs fp32 fallback ──
            # The fused Triton kernel handles its OWN dequant→update→requant in int8.
            # The fp32 fallback (bits=32 group OR numel < min_8bit_size) uses _foreach.
            if self.fused:
                fused_params, fallback_params = [], []
                for p in params:
                    if grp_bits == 8 and p.numel() >= self.min_8bit_size:
                        fused_params.append(p)
                    else:
                        fallback_params.append(p)
                if fused_params:
                    self._fused_step(fused_params, lr, beta2, beta3_t, alpha_t,
                                     eps, bc2, wd)
                params = fallback_params
                if not params:
                    continue

            # ── Dequant / init (per-param: bnb blockwise quant is per-tensor) ──
            # Init holds zero state as fp32 transiently — we never QUANTIZE an all-zero
            # tensor (absmax=0 → 0/0); the first requant is AFTER the first update.
            m2s, nus, gs, use8s = [], [], [], []
            for p in params:
                st = self.state[p]
                use8 = grp_bits == 8 and p.numel() >= self.min_8bit_size
                use8s.append(use8)
                gs.append(p.grad.float())
                if len(st) == 0:
                    st["init"] = True
                if st.get("init"):
                    m2s.append(torch.zeros_like(p, dtype=torch.float32))
                    nus.append(torch.zeros_like(p, dtype=torch.float32))
                elif use8:
                    m2s.append(self._deq(st["m2_q"], st["m2_amax"], True, p))
                    nus.append(self._deq(st["nu_q"], st["nu_amax"], False, p))
                else:
                    m2s.append(st["m2"])
                    nus.append(st["nu"])

            # ── Batched elementwise update (torch._foreach → ~10 launches, not ~8·N) ──
            torch._foreach_mul_(m2s, beta3_t)
            torch._foreach_add_(m2s, gs, alpha=1.0 - beta3_t)
            torch._foreach_mul_(nus, beta2)
            torch._foreach_addcmul_(nus, gs, gs, value=1.0 - beta2)
            # denom = √(ν/bc2 + ε) — eps INSIDE the sqrt (matches the fused kernel; see
            # ademamix_b1zero_kernel.py for why: floors underflowed-ν denom to √ε instead
            # of ε, preventing the g/ε explosion on linear-int8 underflow).
            denoms = torch._foreach_div(nus, bc2)   # ν/bc2
            torch._foreach_add_(denoms, eps)        # ν/bc2 + ε
            torch._foreach_sqrt_(denoms)            # √(ν/bc2 + ε)
            upd = torch._foreach_mul(m2s, alpha_t)          # α·m2
            torch._foreach_add_(upd, gs)                    # + g   (m1 = g, β1=0)
            torch._foreach_div_(upd, denoms)                # /denom
            if wd != 0.0:
                torch._foreach_add_(upd, [p.float() for p in params], alpha=wd)  # + λ·p
            torch._foreach_add_(params, [u.to(p.dtype) for u, p in zip(upd, params)],
                                alpha=-lr)                  # p -= lr·update

            # ── Requant / store (per-param) ──
            for p, m2, nu, use8 in zip(params, m2s, nus, use8s):
                st = self.state[p]
                st.pop("init", None)
                if use8:
                    st["m2_q"], st["m2_amax"] = self._q(m2, True, p)
                    st["nu_q"], st["nu_amax"] = self._q(nu, False, p)
                else:
                    st["m2"], st["nu"] = m2, nu
        return loss
