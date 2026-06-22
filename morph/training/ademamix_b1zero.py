"""AdEMAMix with β1=0 and blockwise-8bit state — memory parity with AdamW8bit.

WHY THIS EXISTS
---------------
bnb's AdEMAMix8bit allocates 3 state buffers (m1, m2, ν) regardless of β1 — its β1=0
"memory trick" is a no-op (bnb β1=0 fork still allocates 3 buffers). With β1=0 the fast
EMA m1 collapses to the raw gradient and needs NO buffer, so a faithful β1=0 implementation
keeps only 2 blockwise-8bit buffers (m2, ν) → ~2 B/param, matching AdamW8bit.

We reuse bnb's OWN blockwise dynamic quantization (bnb.functional.quantize_blockwise /
dequantize_blockwise, blocksize=256, dynamic qmap) so the quantization is numerically
identical to bnb's optimizers — only the buffer COUNT and the update math differ.

UPDATE (β1=0 AdEMAMix, arXiv:2409.03137):
    m2 ← β3·m2 + (1−β3)·g                  (slow EMA)
    ν  ← β2·ν  + (1−β2)·g²                  (second moment)
    bc2 = 1 − β2^t
    update = (g + α·m2) / √(ν/bc2 + ε) + λ·p       # m1 = g exactly (β1=0, bc1=1); ε inside √
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
        alpha_cap: float = 0.0,
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
        stale_push_cap_coord: float = 0.0,
        g_coef: float = 1.0,
        g_snr_gate_kappa: float = 0.0,
        g_snr_gate_floor: float = 0.1,
        track_diag: bool = False,
        fused_dynamic_qmap: bool = False,
        fused_nu_floor: bool = True,
    ):
        if betas[0] != 0.0:
            raise ValueError(f"AdEMAMixB1Zero requires β1=0, got betas={betas}")
        # All remaining update-shaping knobs (eps_inside, g_coef, g_snr_gate_kappa,
        # stale_push_cap_coord, update_clip) are elementwise and supported on BOTH the
        # fused kernel (ademamix_b1zero_kernel.py) and the de-fused path. The de-fused-only
        # experimental levers (per-tensor stale_push_cap, align_gate, num_beta1, amsgrad,
        # trust_ratio, flip_clamp, update_rms_clip, eps_inside_patterns, m2_prune_decay) were
        # removed after the per-coord-cap cure won the deploy gauntlet (#4 cleanup).
        if bits not in (8, 32):
            raise ValueError(f"bits must be 8 or 32, got {bits}")
        defaults = dict(
            lr=lr, betas=betas, alpha=alpha, alpha_cap=alpha_cap, t_alpha=t_alpha, t_beta3=t_beta3,
            beta3_warmup_start=beta3_warmup_start, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.bits = bits
        self.min_8bit_size = min_8bit_size
        self.blocksize = blocksize
        self.fused = fused
        # eps_inside: True = √(ν/bc2 + ε) (denom floored at √ε≈1e-4); False = √(ν/bc2) + ε
        # (true-Adam normalization). Honored by BOTH paths — the de-fused step and the fused
        # kernel (EPS_INSIDE constexpr). The floor exists because the LINEAR-int8 fused path can
        # underflow ν to exactly 0 → denom→ε → explosion. The de-fused and dynamic-qmap paths
        # store ν faithfully (no underflow) and _mask_dead_state handles pruned positions, so the
        # floor there only throttles live small-ν params. DEFAULT is False (eps-outside): the
        # deploy path is fused + dynamic-qmap, where eps-outside is safe and is the convergence-
        # preserving denom; update_clip bounds any residual spike.
        self.eps_inside = eps_inside
        # update_clip: per-coordinate cap on the adaptive update (g+α·m₂)/denom, in Adam
        # units (healthy step ≈O(1)). 0 = off. At a prune topology-shock the gradient
        # jumps while ν lags → the step spikes on a few coords → single-step blow-up.
        # Clamping bounds those anomalous coords without raising the denom for all small-ν
        # params (unlike eps-inside's blanket floor). Weight decay (λ·p) is added AFTER
        # the clamp so it is never capped away.
        self.update_clip = float(update_clip)
        # track_diag: when False (deploy default), all per-tensor diagnostic loops (snr-gate
        # mean/low, clip event counts, coord-cap masked counts) are skipped. These loops cost
        # ~67ms/step (per-tensor .item() syncs) and only feed log-cadence telemetry; the param
        # update is bit-identical either way. True → accumulate GPU-resident (sync-free) for
        # diagnostic runs.
        self.track_diag = bool(track_diag)
        self._clip_events = 0  # diagnostic: total coords clamped across all steps (no-theater check)
        # Per-coordinate stale-push cap: |α·m₂_i| ≤ c·|g_i| each coord independently. The
        # failure mode is per-coord magnitude domination (stale α·m₂_i large where live g_i
        # collapsed); the per-tensor cap misses the few collapsed coords hidden by the overall
        # tensor norm. This is the mechanism-matched fix. De-fused path only. 0.0 = off.
        # _stale_cap_coord_masked: GPU-resident diagnostic counter.
        self.stale_push_cap_coord = float(stale_push_cap_coord)
        self._stale_cap_coord_masked = 0.0
        # β1=0 noise-gating fix: restore Adam's implicit SNR filter without an m1 buffer.
        # β1=0 puts the raw gradient directly in the numerator; on noise-dominated coords
        # (signal ≪ noise) this is a unit-variance random walk every step. Standard Adam
        # suppresses this via the fast EMA m1 (β1≈0.9 shrinks noise ~10×). β1=0 removes
        # that filter (the memory win), so two stateless modes restore it:
        #
        # g_coef (γ): numerator = γ·g + α·m₂. Uniform downscale; damps all coords including
        #   signal (coarse control). 1.0 = off.
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
        # g-vs-numerator geometry capture: when enabled via set_diag_capture(), the de-fused
        # step appends per tracked tensor: cos(g, m₂), cos(g, α·m₂+gated_g),
        # ‖α·m₂‖/‖gated_g‖, and component norms → _diag_rows. The train loop drains it each
        # step. Values are the exact working copies used to build the update (cannot silently
        # disagree with the real update).
        self._diag_capture = False
        self._diag_names: dict[int, str] = {}   # id(p) -> name (only tracked params)
        self._diag_rows: list = []              # drained by train.py each step
        self._diag_step = -1                    # set by step() so drained rows carry the step
        # dynamic qmaps (lazily moved to each param's device); signed for m2, unsigned for ν.
        self._code_signed_cpu = bnbF.create_dynamic_map(signed=True)
        self._code_unsigned_cpu = bnbF.create_dynamic_map(signed=False)
        self._code_cache: dict = {}
        # Fused dynamic-qmap: replaces the fused kernel's uniform linear-int8 quantizer (which
        # underrepresents heavy-tailed optimizer state) with bnb's own non-linear dynamic map —
        # the same quantizer as the de-fused reference. ν is stored directly (no sqrt/floor),
        # eliminating the systematic denom bias of the linear path. Memory unchanged (~2.03
        # B/param). When False, the fused kernel uses linear-int8 with sqrt-ν storage.
        self.fused_dynamic_qmap = bool(fused_dynamic_qmap)
        # fused_nu_floor: linear-path-only legacy code-1 floor on sqrt(ν) (prevents int8 underflow→0).
        # Exposed so we can A/B-attribute the tax (floor-bias vs resolution). Ignored when dynamic.
        self.fused_nu_floor = bool(fused_nu_floor)
        # Index in the signed dynamic map whose decoded value == 0 (the map's midpoint).
        # Required because code 0 ≠ value 0 for the signed map (unlike linear int8).
        # Used by _mask_dead_state to zero m2 at pruned positions.
        self._signed_zero_idx = int(self._code_signed_cpu.abs().argmin().item())

    def _code(self, device, signed: bool):
        key = (device, signed)
        c = self._code_cache.get(key)
        if c is None:
            src = self._code_signed_cpu if signed else self._code_unsigned_cpu
            c = src.to(device)
            self._code_cache[key] = c
        return c

    def set_diag_capture(self, model, name_filter=None, enable: bool = True) -> int:
        """Enable per-tensor g-vs-numerator geometry capture. Returns count of tracked params.

        name_filter(name)->bool selects which params to track (default: all requires_grad).
        Tracked params get cos(g,m₂), cos(g,α·m₂+gated_g), ‖α·m₂‖/‖gated_g‖ appended to
        _diag_rows by step(); drain it from the train loop. Cheap (a few norm reductions per
        tracked tensor per step) but only enable for diagnostic runs.
        """
        self._diag_capture = bool(enable)
        self._diag_names = {}
        if enable:
            root = getattr(model, "_orig_mod", model)
            for nm, p in root.named_parameters():
                if p.requires_grad and (name_filter is None or name_filter(nm)):
                    self._diag_names[id(p)] = nm
        return len(self._diag_names)

    @staticmethod
    def _sched(step, group):
        """Return (alpha_t, beta2, beta3_t) with decoupled α / β3 warmup."""
        b1, b2, b3 = group["betas"]
        alpha = group["alpha"]
        ta, tb = group["t_alpha"], group["t_beta3"]
        if ta:
            alpha = min(step * alpha / ta, alpha)
        # α_t hard cap: bounds the scheduled α_t to prevent per-block gain runaway in the looped
        # core. Applied after warmup ramp. 0 = off.
        cap = group.get("alpha_cap", 0.0)
        if cap and cap > 0.0:
            alpha = min(alpha, cap)
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

        FOOTGUN: torch.optim.Optimizer.load_state_dict runs
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

        # Precision-upgrade resume: if this optimizer is fp32 (bits=32) but the checkpoint
        # holds 8-bit de-fused state (m2_q/nu_q), dequantize to fp32 m2/nu so the fp32 path
        # can resume cleanly. Without this the fp32 step branch reads st["m2"] → KeyError.
        # Only the de-fused dynamic-qmap path is converted; fused linear-int8 is unchanged.
        # m2 is signed, ν is unsigned.
        if self.bits == 32:
            n_conv = 0
            for p, st in self.state.items():
                if not isinstance(st, dict) or "m2_q" not in st or "m2" in st:
                    continue
                st["m2"] = self._deq(st.pop("m2_q"), st.pop("m2_amax"), True, p)
                st["nu"] = self._deq(st.pop("nu_q"), st.pop("nu_amax"), False, p)
                n_conv += 1
            if n_conv:
                print(f"  [ademamix] dequantized {n_conv} param states 8-bit→fp32 on resume "
                      f"(bits=32 optimizer; clean precision swap)")

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
            if self.fused_dynamic_qmap:
                # ── dynamic-qmap fused path (#278): uint8 codes into bnb's non-linear map,
                # ν stored DIRECTLY (no sqrt, no floor) → matches the de-fused reference quantizer.
                if is_init:
                    st.pop("init", None)
                    st["m2_dcode"] = torch.zeros(n, dtype=torch.uint8, device=p.device)
                    st["nu_dcode"] = torch.zeros(n, dtype=torch.uint8, device=p.device)
                    st["m2_damax"] = torch.zeros(nblocks, dtype=torch.float32, device=p.device)
                    st["nu_damax"] = torch.zeros(nblocks, dtype=torch.float32, device=p.device)
                fused_ademamix_b1zero_step(
                    p_flat, g_flat,
                    st["m2_dcode"], st["m2_damax"], st["nu_dcode"], st["nu_damax"],
                    lr, beta2, beta3, alpha, eps, bc2, wd, is_init,
                    eps_inside=self.eps_inside,
                    g_coef=self.g_coef,
                    snr_kappa=self.g_snr_gate_kappa,
                    snr_floor=self.g_snr_gate_floor,
                    coord_cap=self.stale_push_cap_coord,
                    upd_clip=self.update_clip,
                    dynamic_qmap=True,
                    code_signed=self._code(p.device, True),
                    code_unsigned=self._code(p.device, False),
                )
            else:
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
                    eps_inside=self.eps_inside,
                    g_coef=self.g_coef,
                    snr_kappa=self.g_snr_gate_kappa,
                    snr_floor=self.g_snr_gate_floor,
                    coord_cap=self.stale_push_cap_coord,
                    upd_clip=self.update_clip,
                    nu_floor=self.fused_nu_floor,
                )

    def _mask_dead_state(self, p) -> None:
        """Zero the slow-momentum (m₂) and second-moment (ν) state at PRUNED positions.

        WHY: MORPH prunes by masking a tile's weight + grad to 0 (CMSBlockLinear tags the
        param with _dead_mask; 1=keep, 0=pruned). For AdamW this is sufficient (grad=0 →
        m,ν both decay → update→0). For AdEMAMix it is not: the α·m₂ slow EMA retains
        pre-prune gradient mass and decays at β3 ≫ β2, while ν collapses. The result is
        (α·m₂)/(√(ν/bc2)+ε) → ∞ for dead params. Zeroing m₂/ν here BEFORE the update
        makes the dead-param update exactly 0, and grad=0 every step keeps the state at 0.
        See Ai-notes/06-21-2026/AdEMAMix-b1zero-Divergence-Cure/TROUBLESHOOTING.md.
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
        if "m2_dcode" in st:           # dynamic-qmap fused: signed-map code 0 ≠ value 0 → use zero-idx
            st["m2_dcode"].view(-1)[dead] = self._signed_zero_idx  # unsigned map[0]=0 → ν code 0
            st["nu_dcode"].view(-1)[dead] = 0
        elif "m2_code" in st:          # fused linear-int8: code 0 → value 0 (exact)
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

            # Zero slow state at pruned positions before update (see _mask_dead_state).
            # Without this, α·m₂ drives pruned params as ν collapses.
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
            # denom: eps_inside → √(ν/bc2 + ε) (floored at √ε; needed only for linear-int8
            # underflow, which the de-fused dynamic-qmap path does NOT have); eps_outside →
            # √(ν/bc2) + ε (true Adam normalization — does not throttle live small-ν params).
            denoms = list(torch._foreach_div(nus, bc2))     # ν/bc2 (list: per-param reassign)
            if self.eps_inside:
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
                if self.track_diag:                         # diagnostics (per-tensor sync — gated)
                    for gt in gate:
                        if gt.numel():
                            self._gate_sum += float(gt.mean()); self._gate_n += 1
                            self._gate_low += int((gt < 0.5).sum())
                torch._foreach_mul_(gs, gate)               # g ← gate·g
            elif self.g_coef != 1.0:
                torch._foreach_mul_(gs, self.g_coef)        # g ← γ·g (constant control)
            upd = torch._foreach_mul(m2s, alpha_t)          # α·m2
            # g-vs-numerator geometry capture: gs = gated_g, m2s = post-EMA m₂, upd = α·m₂,
            # p.grad = raw g. Values are exact working copies used for the update.
            if self._diag_capture and self._diag_names:
                for _i, _p in enumerate(params):
                    _nm = self._diag_names.get(id(_p))
                    if _nm is None or _p.grad is None:
                        continue
                    _rg = _p.grad.detach().float().reshape(-1)
                    _m2 = m2s[_i].detach().float().reshape(-1)
                    _gg = gs[_i].detach().float().reshape(-1)      # gated g
                    _am = upd[_i].detach().float().reshape(-1)     # α·m2
                    _num = _am + _gg                                # numerator = gated_g + α·m₂
                    _rgn = float(_rg.norm()); _m2n = float(_m2.norm())
                    _ggn = float(_gg.norm()); _numn = float(_num.norm()); _amn = float(_am.norm())
                    self._diag_rows.append((
                        _nm,
                        float((_rg @ _m2) / (_rgn * _m2n + 1e-12)),    # cos(g, m₂)
                        float((_rg @ _num) / (_rgn * _numn + 1e-12)),  # cos(g, α·m₂+gated_g)
                        _amn / (_ggn + 1e-12),                          # ‖α·m₂‖ / ‖gated_g‖
                        _m2n, _rgn, _amn, _ggn,
                    ))
            # Per-coordinate stale-push cap: |α·m₂_i| ≤ c·|g_i| for each coord.
            # Bounds the stale slow-EMA contribution on coords where the live gradient has
            # collapsed. Sign preserved. _foreach-vectorized + sync-free.
            # gs = gated g (proportional to |raw g|).
            if self.stale_push_cap_coord > 0.0:
                c = self.stale_push_cap_coord
                caps = torch._foreach_abs(gs)               # |g|
                torch._foreach_mul_(caps, c)                # c·|g|
                au = torch._foreach_abs(upd)                # |α·m₂|
                over = torch._foreach_sub(au, caps)         # |α·m₂| − c·|g|
                torch._foreach_clamp_min_(over, 0.0)        # excess = relu(...)
                torch._foreach_sub_(au, over)               # min(|α·m₂|, c·|g|)
                signs = torch._foreach_sign(upd)
                upd = torch._foreach_mul(signs, au)         # sign(α·m₂)·min(|α·m₂|, c·|g|)
                if self.track_diag:                         # masked-coord count (454-iter loop — gated)
                    cd = self._stale_cap_coord_masked
                    if not torch.is_tensor(cd):
                        cd = upd[0].new_zeros(())           # GPU-resident, synced only on read
                    for o in over:
                        cd = cd + (o > 0).sum()
                    self._stale_cap_coord_masked = cd
            torch._foreach_add_(upd, gs)                    # + (gated) g   (m1 = g, β1=0)
            torch._foreach_div_(upd, denoms)                # /denom = (g·gate + α·m2)/denom
            # ── per-coordinate update clamp (prune-shock stability; see __init__) ──
            if self.update_clip > 0.0:
                c = self.update_clip
                if self.track_diag:                         # over-clip count (per-tensor sync — gated)
                    for u in upd:
                        if u.numel():
                            self._clip_events += int((u.abs() > c).sum())
                # batched clamp (numerics-identical to per-tensor clamp_ — elementwise, order-free)
                torch._foreach_clamp_min_(upd, -c)
                torch._foreach_clamp_max_(upd, c)
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
