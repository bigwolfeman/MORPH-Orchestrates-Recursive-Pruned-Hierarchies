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
        eps_inside: bool = True,
        update_clip: float = 0.0,
        update_rms_clip: float = 0.0,
        amsgrad: bool = False,
        trust_ratio: float = 0.0,
        g_coef: float = 1.0,
        g_snr_gate_kappa: float = 0.0,
        g_snr_gate_floor: float = 0.1,
        num_beta1: float = 0.0,
        flip_clamp_kappa: float = 0.0,
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
        # eps_inside: √(ν/bc2 + ε) (denom floored at √ε≈1e-4). REQUIRED for the fused
        # linear-int8 path (ν can underflow to EXACTLY 0 → denom=ε=1e-8 → g/1e-8 explosion).
        # HARMFUL for the de-fused/fp32 path: ν never underflows to 0 there (bnb dynamic qmap)
        # and the prune α·m₂ blowup is handled by _mask_dead_state, so the floor only throttles
        # live small-ν params → slow convergence (measured 2026-06-16). De-fused → eps_inside=False
        # for √(ν/bc2)+ε (true Adam normalization). This flag only affects the de-fused path;
        # the fused kernel always applies eps-inside (it needs it).
        self.eps_inside = eps_inside
        # When create_optimizer tags individual params with `p._eps_inside` (pattern-based,
        # e.g. eps-inside ONLY on the fragile prelude.0 boundary while the looped core stays
        # eps-outside for full advantage), this flag switches the de-fused denom to a per-param
        # branch. Off → the fast batched foreach path (global self.eps_inside), unchanged.
        self._has_eps_overrides = False
        # update_clip: per-coordinate cap on the ADAPTIVE update term (g+α·m₂)/denom, in Adam
        # units (a healthy step is O(1); measured worst healthy ~29). 0 = off. This is the
        # fast+stable fix for the prune-shock detonation: at a prune topology-shock the gradient
        # jumps while ν lags → (g+α·m₂)/√(lagging ν) spikes to ~1e4 on a few coords → single-step
        # blow-up (observed: eps-out b2=.999 @9400, b2=.95 @19400). Clamping the update bounds the
        # per-step param change (lr·clip) on JUST those anomalous coords — unlike eps-inside's √ε
        # floor it does NOT raise the denom for all small-ν params, so no convergence throttle.
        # Weight decay (λ·p, bounded) is added AFTER the clamp so it is never capped away.
        self.update_clip = float(update_clip)
        self._clip_events = 0  # diagnostic: total coords clamped across all steps (no-theater check)
        # update_rms_clip: per-TENSOR cap on the RMS of the adaptive update (g+α·m₂)/denom.
        # The per-COORD clip (above) is the wrong granularity for the COLLECTIVE oscillation that
        # detonates eps-outside on the looped/ternary first layer (measured 2026-06-16, no-prune
        # isolation): ~500k prelude.0 coords spike COHERENTLY in one step → capping each to `c`
        # still moves the whole layer ~Nc → divergence. The RMS clip bounds the layer's herd move:
        # when a tensor's update-RMS exceeds this (healthy ≈0.5, collective spike ≈9), scale the
        # whole tensor down to the cap → kills the coherent overshoot without touching healthy steps
        # (unlike eps-inside which floors every small-ν coord every step = throttle). 0 = off.
        self.update_rms_clip = float(update_rms_clip)
        self._rms_clip_events = 0  # diagnostic: total tensors RMS-scaled across all steps
        # amsgrad: AMSGrad-style denominator MEMORY — use ν_max=max(ν_max,ν) in the denom instead
        # of ν. Reddi et al. (arXiv:1904.09237) "On the Convergence of Adam and Beyond": Adam's EMA
        # 2nd moment lets the denom FORGET past large gradients → effective LR (lr/√ν) rises on a
        # coord that was loud then went quiet → oscillatory g/√(decayed ν) blowup. ν_max ratchets:
        # once a coord shows large variance the denom never shrinks below it → bounds that coord's
        # step, while genuinely-always-quiet coords (ν_max=ν) keep full eps-outside advantage. This
        # targets the EXACT measured failure (genuine small ν≈6e-9, oscillatory, no quant-zero, no
        # stale m2). COST: a 3rd state buffer (forfeits the β1=0 2-buffer memory win) — fp32 here for
        # a clean diagnostic; a blockwise/coarse ν_max would recover memory if this proves the fix.
        self.amsgrad = bool(amsgrad)
        # trust_ratio (τ): downward-only LARS/LAMB-style per-tensor governor. After the adaptive
        # update u is formed, scale so the weight moves at most τ·‖w‖ per step: u *= min(1, τ‖w‖/(lr‖u‖)).
        # Bounds the RELATIVE per-layer move (dimension/scale-aware, self-tuning) — the principled
        # form of the absolute RMS clip. Never amplifies (min with 1). 0 = off.
        self.trust_ratio = float(trust_ratio)
        self._trust_events = 0
        # ── β1=0 ROOT-CAUSE FIX: scale the raw-gradient (m1=g) numerator term ──────────────
        # WHY (the open question the AdEMAMix paper never answered, diagnosed 2026-06-17):
        # β1=0 puts the RAW gradient in the numerator — update = (g + α·m₂)/√E[g²]. On a
        # noise-dominated coordinate (running mean μ ≪ std σ) that is g/√E[g²] ≈ a UNIT-VARIANCE
        # RANDOM WALK every step, injected into every weak/noisy weight. Standard Adam/AdEMAMix
        # (β1=0.9) suppress this: the fast EMA m1 averages noise toward its mean, shrinking the
        # noise-coord magnitude ~(1−β1)=10× — that is Adam's implicit SNR/noise-GATING. β1=0
        # throws the gate away (no m1 buffer = the memory win, but also no gating). Damage
        # accumulates (slow ppl creep) → cascade; the √(1/(1−β2))≈31.6 saturation spikes are the
        # tail of the same phenomenon. Scale-invariant in g, so grad-clipping is useless. We
        # restore the gating WITHOUT an m1 buffer (β1=0 memory parity preserved), two stateless modes:
        #
        # g_coef (γ, constant): numerator = γ·g + α·m₂. Crudest mimic of (1−β1) — damps EVERY
        #   coord's g-term uniformly. Cheap CONTROL (throttles signal coords too). 1.0 = off.
        self.g_coef = float(g_coef)
        # g_snr_gate_kappa (κ, soft per-coord SNR gate): snr = |m₂|/denom ≈ |running-mean|/rms
        #   ∈[0,~1] (m₂ is the slow β3-EMA = the persistent/signal component; denom ≈ √E[g²] = rms
        #   incl. noise). gate = floor + (1−floor)·clamp(snr/κ, 0, 1); numerator = gate·g + α·m₂.
        #   Noise coord (m₂→0 ⇒ snr→0) → gate=floor (≈(1−β1)=0.1, ~10× variance cut, matches Adam).
        #   Signal coord (snr≈1) → gate=1.0 (full speed, AdEMA advantage kept). Fresh spike auto-gated:
        #   ν (β2=.999) responds ~10× faster than m₂ (β3=.9999) so at a fresh spike m₂ is tiny vs the
        #   lagging denom → snr→0 → gate=floor → the 31.6 random-walk is killed at its source. ZERO new
        #   state (reuses m₂ and the already-computed denom). 0.0 = off. floor = g_snr_gate_floor.
        self.g_snr_gate_kappa = float(g_snr_gate_kappa)
        self.g_snr_gate_floor = float(g_snr_gate_floor)
        self._gate_sum = 0.0    # diag: Σ per-tensor mean-gate (÷ _gate_n = mean gate applied)
        self._gate_n = 0
        self._gate_low = 0      # diag: total coords with gate < 0.5 (heavily noise-gated)
        # ── STE-CUSP-VAULT FIXES (2026-06-18, mechanism CONFIRMED frame-by-frame) ──
        # The deploy detonation = β1=0's RAW-gradient step vaults a ternary-STE quantization cusp →
        # ~100k core ternary codes flip in ONE step → discrete realized-weight jump → looped core
        # amplifies T× → residual-stream explosion (RMSNorm-masked → recoverable). Two independent
        # mitigations, tested SEPARATELY (Wolfe 2026-06-18). Both DE-FUSED-path only; off by default.
        # ARM D — num_beta1: a SMALL momentum on the raw-g NUMERATOR term (m1 = β1·m1 + (1−β1)·g,
        #   bias-corrected). Smooths the step direction so it APPROACHES cusps gently instead of
        #   vaulting (the direct β1 analog AdamW has). Slow-EMA m₂/ν still see the TRUE g (unchanged).
        #   Costs ONE extra 8-bit blockwise buffer (~+1 B/param) — bends the β1=0 memory win; this is
        #   the test of whether momentum-smoothing cures the vault (8-bit keeps it ≪ a full fp32 m1).
        #   0.0 = off (numerator = raw g, exactly the β1=0 behaviour).
        self.num_beta1 = float(num_beta1)
        # ARM A — flip_clamp_kappa: per-tensor cap on the realized ternary FLIP-RATE per step. The
        #   vault is a ~20× flip-count spike on an otherwise-healthy trajectory; cap it. For each
        #   weight, estimate the symmetric per-tensor STE scale s=mean|w| (matches deploy
        #   scale_group='tensor'), threshold thr=0.5·s, count codes that WOULD flip under the move
        #   p−lr·u; if flip-fraction > κ, scale the tensor's update by κ/frac (a monotone governor —
        #   reduces the realized flip rate toward the cap). STATELESS (β1=0 memory win intact). Only
        #   bites high-flip tensors (the cusp-vault); healthy steps (flip-frac ≪ κ) pass untouched.
        #   0.0 = off. NOTE applied to all dim≥2 weights via the mean|w| proxy (exact for symmetric
        #   per-tensor ternary; on non-ternary weights flips are rare so it ~never triggers).
        self.flip_clamp_kappa = float(flip_clamp_kappa)
        self._flip_clamp_events = 0  # diag: total tensors flip-rate-scaled across all steps
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

    def load_state_dict(self, state_dict):
        """Restore blockwise-quant CODE dtypes after torch's float-casting load.

        FOOTGUN (real resume bug, 2026-06-16): torch.optim.Optimizer.load_state_dict runs
        _process_value_according_to_param_policy, which casts EVERY non-`step` state value to
        the param's dtype WHEN THE PARAM IS FLOATING POINT. Our quant codes are INTEGER
        (uint8 de-fused m2_q/nu_q; int8 fused m2_code/nu_sqrt_code) — they get silently
        upcast to the fp32 param dtype, so the dequant kernel rejects them ("A must be
        uint8"). bnb's own 8-bit optimizers dodge this via custom loading; our default-loaded
        fork did not (the resumable-ckpt gate used AdamW8bit, so this was never exercised).
        The cast is LOSSLESS for small-int codes (0-255 / -127..127 exact in fp32), so we
        round-and-recast back to the integer dtype. amax tensors are fp32 and only lose
        precision if a bf16 param downcast them — coerce them back to fp32 best-effort
        (no-op for the fp32 master weights used here).
        """
        super().load_state_dict(state_dict)
        for st in self.state.values():
            if not isinstance(st, dict):
                continue
            for k in ("m2_q", "nu_q"):                        # de-fused dynamic-map (uint8)
                v = st.get(k)
                if torch.is_tensor(v) and v.dtype != torch.uint8:
                    st[k] = v.round().clamp_(0, 255).to(torch.uint8)
            for k in ("m2_code", "nu_sqrt_code", "nu_code"):  # fused linear-int8 (int8)
                v = st.get(k)
                if torch.is_tensor(v) and v.dtype != torch.int8:
                    st[k] = v.round().clamp_(-127, 127).to(torch.int8)
            for k in ("m2_amax", "nu_amax", "nu_sqrt_amax"):  # per-block scales (fp32)
                v = st.get(k)
                if torch.is_tensor(v) and v.dtype != torch.float32:
                    st[k] = v.float()

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
            m2s, nus, nu_maxs, gs, use8s = [], [], [], [], []
            m1s = []   # ARM D: raw-g numerator momentum (only populated when num_beta1>0)
            use_m1 = self.num_beta1 > 0.0
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
                    if use_m1:
                        m1s.append(torch.zeros_like(p, dtype=torch.float32))
                elif use8:
                    m2s.append(self._deq(st["m2_q"], st["m2_amax"], True, p))
                    nus.append(self._deq(st["nu_q"], st["nu_amax"], False, p))
                    if use_m1:
                        m1s.append(self._deq(st["m1_q"], st["m1_amax"], True, p)
                                   if "m1_q" in st else torch.zeros_like(p, dtype=torch.float32))
                else:
                    m2s.append(st["m2"])
                    nus.append(st["nu"])
                    if use_m1:
                        m1s.append(st.get("m1", torch.zeros_like(p, dtype=torch.float32)))
                if self.amsgrad:   # fp32 ν_max buffer (resume-safe default for older ckpts)
                    nu_maxs.append(st.get("nu_max", torch.zeros_like(p, dtype=torch.float32)))

            # ── Batched elementwise update (torch._foreach → ~10 launches, not ~8·N) ──
            torch._foreach_mul_(m2s, beta3_t)
            torch._foreach_add_(m2s, gs, alpha=1.0 - beta3_t)
            torch._foreach_mul_(nus, beta2)
            torch._foreach_addcmul_(nus, gs, gs, value=1.0 - beta2)
            # ── ARM D: numerator momentum (m1 = β1·m1 + (1−β1)·g, bias-corrected) ──
            # m2/ν above already consumed the TRUE g; here we smooth ONLY the raw-g numerator term so
            # the step approaches ternary cusps gently (AdamW's β1 analog). gs is rebound to m1_hat →
            # the SNR gate + numerator-add below operate on the smoothed value. Off when num_beta1=0.
            if use_m1:
                b1 = self.num_beta1
                torch._foreach_mul_(m1s, b1)
                torch._foreach_add_(m1s, gs, alpha=1.0 - b1)
                bc1 = 1.0 - b1 ** step
                gs = list(torch._foreach_div(m1s, bc1))   # m1_hat (bias-corrected) replaces raw g
            # AMSGrad ratchet: ν_max = max(ν_max, ν) → denom uses ν_max (never forgets past variance;
            # see __init__). Genuinely-always-quiet coords keep ν_max=ν → full advantage preserved.
            if self.amsgrad:
                for i in range(len(nus)):
                    torch.maximum(nu_maxs[i], nus[i], out=nu_maxs[i])
                denom_src = nu_maxs
            else:
                denom_src = nus
            # denom: eps_inside → √(ν/bc2 + ε) (floored at √ε; needed only for linear-int8
            # underflow, which the de-fused dynamic-qmap path does NOT have); eps_outside →
            # √(ν/bc2) + ε (true Adam normalization — does not throttle live small-ν params).
            denoms = list(torch._foreach_div(denom_src, bc2))  # (ν or ν_max)/bc2 (list: per-param reassign)
            if self._has_eps_overrides:
                # per-param eps placement (pattern-targeted): eps-inside on tagged params
                # (√(ν/bc2+ε), √ε floor → stabilize the fragile boundary), eps-outside elsewhere.
                for i, p in enumerate(params):
                    if getattr(p, "_eps_inside", self.eps_inside):
                        denoms[i] = (denoms[i] + eps).sqrt()
                    else:
                        denoms[i] = denoms[i].sqrt().add_(eps)
            elif self.eps_inside:
                torch._foreach_add_(denoms, eps)        # ν/bc2 + ε
                torch._foreach_sqrt_(denoms)            # √(ν/bc2 + ε)
            else:
                torch._foreach_sqrt_(denoms)            # √(ν/bc2)
                torch._foreach_add_(denoms, eps)        # √(ν/bc2) + ε
            # ── raw-gradient (m1=g) term scaling: soft SNR gate and/or constant γ (see __init__) ──
            # The β1=0 root-cause fix. Restores Adam's lost noise-GATING by damping g on low-SNR
            # coords BEFORE it enters the numerator. State EMAs above already saw the TRUE g; gs is
            # not reused after the add below, so the in-place scale is safe. denoms here = √(ν/bc2)(+ε).
            if self.g_snr_gate_kappa > 0.0:
                floor = self.g_snr_gate_floor
                gate = torch._foreach_abs(m2s)              # |m₂|
                torch._foreach_div_(gate, denoms)           # snr = |m₂| / denom  (≈ |mean|/rms)
                torch._foreach_div_(gate, self.g_snr_gate_kappa)  # snr / κ
                torch._foreach_clamp_min_(gate, 0.0)
                torch._foreach_clamp_max_(gate, 1.0)        # clamp(snr/κ, 0, 1)
                torch._foreach_mul_(gate, 1.0 - floor)
                torch._foreach_add_(gate, floor)            # floor + (1-floor)·clamp(snr/κ,0,1)
                if self.g_coef != 1.0:
                    torch._foreach_mul_(gate, self.g_coef)  # compose with constant γ
                for gt in gate:                             # cheap diagnostics (one reduce/tensor)
                    if gt.numel():
                        self._gate_sum += float(gt.mean()); self._gate_n += 1
                        self._gate_low += int((gt < 0.5).sum())
                torch._foreach_mul_(gs, gate)               # g ← gate·g
            elif self.g_coef != 1.0:
                torch._foreach_mul_(gs, self.g_coef)        # g ← γ·g (constant control)
            upd = torch._foreach_mul(m2s, alpha_t)          # α·m2
            torch._foreach_add_(upd, gs)                    # + (gated) g   (m1 = g, β1=0)
            torch._foreach_div_(upd, denoms)                # /denom = (g·gate + α·m2)/denom
            # ── per-coordinate update clamp (prune-shock stability; see __init__) ──
            if self.update_clip > 0.0:
                c = self.update_clip
                for u in upd:
                    if u.numel():
                        self._clip_events += int((u.abs() > c).sum())
                        u.clamp_(-c, c)
            # ── per-tensor update-RMS clip (collective-oscillation stability; see __init__) ──
            if self.update_rms_clip > 0.0:
                rc = self.update_rms_clip
                for u in upd:
                    if u.numel():
                        rms = u.pow(2).mean().sqrt()
                        if float(rms) > rc:
                            self._rms_clip_events += 1
                            u.mul_(rc / (rms + 1e-12))
            # ── downward-only trust-ratio (LARS/LAMB-style; see __init__): cap the RELATIVE
            #    per-tensor move so lr·‖u‖ ≤ τ·‖w‖. Never amplifies. Scale-aware governor. ──
            if self.trust_ratio > 0.0:
                tau = self.trust_ratio
                for u, p in zip(upd, params):
                    if u.numel():
                        un = u.norm()
                        wn = p.float().norm()
                        cap = tau * wn / (lr * un + 1e-12)
                        if float(cap) < 1.0:
                            self._trust_events += 1
                            u.mul_(cap)
            # ── ARM A: flip-aware update clamp (cap realized ternary flip-rate per tensor) ──
            # The cusp-vault is a ~20× per-step ternary-flip spike on an otherwise-healthy trajectory.
            # For each weight: s=mean|w| (symmetric per-tensor STE scale), thr=0.5·s; count codes that
            # would flip under the move p−lr·u; if flip-fraction > κ, scale that tensor's update by
            # κ/frac (monotone governor → realized flip-rate toward the cap). Healthy steps pass
            # untouched (frac ≪ κ). Done BEFORE weight-decay/apply so the cap governs the real move.
            if self.flip_clamp_kappa > 0.0:
                k = self.flip_clamp_kappa
                for j, (u, p) in enumerate(zip(upd, params)):
                    if u.dim() < 2 or u.numel() == 0:
                        continue
                    pf = p.detach().float()
                    thr = 0.5 * pf.abs().mean().clamp_min(1e-6)
                    c_old = torch.sign(pf) * (pf.abs() > thr)
                    pn = pf - lr * u
                    c_new = torch.sign(pn) * (pn.abs() > thr)
                    frac = float((c_new != c_old).sum()) / pf.numel()
                    if frac > k:
                        self._flip_clamp_events += 1
                        u.mul_(k / (frac + 1e-12))
            if wd != 0.0:
                torch._foreach_add_(upd, [p.float() for p in params], alpha=wd)  # + λ·p
            torch._foreach_add_(params, [u.to(p.dtype) for u, p in zip(upd, params)],
                                alpha=-lr)                  # p -= lr·update

            # ── Requant / store (per-param) ──
            for i, (p, m2, nu, use8) in enumerate(zip(params, m2s, nus, use8s)):
                st = self.state[p]
                st.pop("init", None)
                if use8:
                    st["m2_q"], st["m2_amax"] = self._q(m2, True, p)
                    st["nu_q"], st["nu_amax"] = self._q(nu, False, p)
                    if use_m1:                  # ARM D: 8-bit blockwise m1 (mirrors m2)
                        st["m1_q"], st["m1_amax"] = self._q(m1s[i], True, p)
                else:
                    st["m2"], st["nu"] = m2, nu
                    if use_m1:
                        st["m1"] = m1s[i]
                if self.amsgrad:
                    st["nu_max"] = nu_maxs[i]   # fp32, kept full-precision (diagnostic)
        return loss
