"""Gated Linear Attention (GLA) — retention branch for the MORPH loop.

Design (retention ablation #230): a global-context / cross-iteration memory branch added
in PARALLEL to the windowed attention in the 2nd layer of prelude/core/coda. The recurrence
is a per-key-channel gated linear attention (Yang et al. 2023, arXiv:2312.06635):

    S_t = diag(alpha_t) · S_{t-1} + k_t^T v_t          (state  S_t ∈ R^{dk×dv} per head)
    o_t = q_t · S_t                                     (output o_t ∈ R^{dv})

with a data-dependent forget gate alpha_t = sigmoid(x_t W_g) ∈ (0,1)^{dk}. The state can be
SEEDED with an initial S_0 (the cross-iteration carry through the MORPH loop) and the
final state S_T is returned for the next iteration. Output is gated + GroupNorm'd (GLA paper)
then projected.

Two forwards, gated against each other for correctness:
  - `recurrent` : explicit O(T) scan. Obviously correct. The oracle.
  - `chunked`   : O(T·C + (T/C)·dk·dv) chunk-parallel form (the training path). Must match.

Both accept `initial_state` and return `final_state`, so the loop-carry composition
(run-in-two-halves == run-whole) is exact — this is the property the ablation depends on.

Pure PyTorch (bf16-friendly; fp32 state accumulation). A Triton kernel is DEFERRED until the
quality ablation earns it (same discipline as the HC kernel).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─── Fused input-projection batching (perf: launch-count + GEMM SOL) ──────────
#
# q/k/v/g/r are five bias-free K=d_model nn.Linears applied to the SAME input x
# → batched into ONE concat-N GEMM, split into last-dim views (o_proj is
# dep-chained on the GLA output and stays separate). Each output element is the
# identical dot product over the identical K=d_model reduction, so the forward
# is mathematically bit-exact (concat along N never touches the K
# accumulation); weight grads accumulate as exact slices of dW_cat; dX becomes
# one reduction over 5·d_model instead of an autograd sum of five — pure
# reassociation, measured (not assumed) in scratchpad/parity_gla_fused_proj.py.
# Same proven mechanism as the landed attention input-projection superblock
# (morph/model/attention.py::_fused_x_proj). GPU bitwise-ness of the forward is
# shape-dependent (cuBLAS algo selection) → gate with the loss-trace
# noise-floor test before landing.
_FUSED_GLA_PROJ = os.environ.get("MORPH_FUSED_GLA_PROJ", "1").lower() not in ("0", "false")


def set_fused_gla_proj(value: bool) -> None:
    """In-process override of the fused-projection toggle (A/B parity tests)."""
    global _FUSED_GLA_PROJ
    _FUSED_GLA_PROJ = bool(value)




class GatedLinearAttention(nn.Module):
    """Multi-head gated linear attention with seedable / returnable state.

    forward(x, initial_state=None) -> (out [B,S,d], final_state [B,H,dk,dv]).

    Args:
        d_model:  model width.
        n_heads:  number of heads (dk = dv = d_model // n_heads).
        mode:     "chunked" (training) or "recurrent" (reference). Numerically equal.
        chunk:    chunk length for the chunked form.
        gate_logit_bias: init bias on the forget-gate logits. Positive → alpha near 1
                  (slow forgetting / long memory) at init.
    """

    def __init__(self, d_model: int, n_heads: int, mode: str = "chunked",
                 chunk: int = 256, gate_logit_bias: float = 2.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads          # dk == dv == dh
        self.mode = mode
        self.chunk = chunk

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)   # forget-gate logits (per head·dk)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)   # output (swish) gate
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_bias = nn.Parameter(torch.full((d_model,), float(gate_logit_bias)))
        self.gn = nn.GroupNorm(n_heads, d_model)

        # All projections normal-init. Identity-at-init for the ablation comes from the OUTER
        # branch-gate (sigmoid init → 0 in the block), NOT from zeroing o_proj — zero-init o_proj
        # would block gradient to q/k/v/g on step 0 (backward chains through the zero matrix).
        for p in (self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.r_proj, self.o_proj):
            nn.init.normal_(p.weight, std=0.02)

    # ── shared projections ───────────────────────────────────────────────
    def _project(self, x: Tensor):
        B, S, _ = x.shape
        H, dh = self.n_heads, self.dh
        if _FUSED_GLA_PROJ:
            # ONE GEMM for q/k/v/g/r (see module-level note). The weight cat is one
            # cheap kernel per forward (cat backward = narrow views, no copy).
            # w.to(x.dtype) targets the GEMM-input dtype, exactly like the attention
            # superblock template; under autocast F.linear casts both operands to the
            # autocast dtype either way (identical to the eager per-Linear path).
            w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                           self.g_proj.weight, self.r_proj.weight], dim=0)
            y = F.linear(x, w.to(x.dtype))
            q_p, k_p, v_p, g_p, r_pre = y.split(self.d_model, dim=-1)
        else:
            q_p, k_p, v_p = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            g_p, r_pre = self.g_proj(x), None
        q = q_p.view(B, S, H, dh)
        k = k_p.view(B, S, H, dh)
        v = v_p.view(B, S, H, dh)
        # per-key-channel forget gate in (0,1); log for stable cumulative products
        glog = g_p + self.gate_bias
        log_alpha = F.logsigmoid(glog).view(B, S, H, dh)        # ≤ 0
        return q, k, v, log_alpha, r_pre

    def _readout(self, x: Tensor, o: Tensor, r_pre: Tensor | None = None) -> Tensor:
        # o: [B,S,H,dh] → [B,S,d]; GLA gated output + GroupNorm + proj.
        # r_pre: optional r_proj(x) from the fused input GEMM (same bits as the
        # eager Linear — see _project); None → recompute (fusion off).
        B, S, H, dh = o.shape
        o = o.reshape(B, S, H * dh)
        # Per-TOKEN GroupNorm over channels (fold S into batch). The previous
        # transpose form — gn(o.transpose(1,2)) on [B,C,S] — pooled normalization
        # statistics across the WHOLE sequence axis, making every position depend
        # on FUTURE tokens: a causality leak. Trained models learned to read the
        # answer through it (teacher-forced 100% vs generation 60% on identical
        # inputs; corrupting future tokens moved past-position logits by up to
        # 17.9). Diagnosed 2026-07-03, Olympiad stage-2.1 A/B; any retention-ON
        # checkpoint trained before this fix is leak-reliant — retrain, don't load.
        o = self.gn(o.reshape(B * S, H * dh)).reshape(B, S, H * dh)
        r = self.r_proj(x) if r_pre is None else r_pre
        o = o * F.silu(r)                                       # swish output gate
        return self.o_proj(o)

    @staticmethod
    def _acc_dtype(dt):
        # Accumulate state in fp32 for low-precision inputs; keep high precision (fp32/fp64) as-is
        # so the fp64 parity oracle stays tight.
        return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt

    # ── reference recurrence (the oracle) ─────────────────────────────────
    def _recurrent(self, q, k, v, log_alpha, S0, reset_mask=None):
        B, S, H, dh = q.shape
        acc = self._acc_dtype(q.dtype)
        state = S0 if S0 is not None else q.new_zeros(B, H, dh, dh)
        state = state.to(acc)
        outs = []
        for t in range(S):
            a = log_alpha[:, t].exp().to(acc)                   # [B,H,dh]
            kt = k[:, t].to(acc)
            vt = v[:, t].to(acc)
            qt = q[:, t].to(acc)
            if reset_mask is not None:
                # Segment reset (docs/tul-tg-spec.md §4): the state ENTERING a reset
                # position is zeroed EXACTLY — a true multiply-by-zero, not a decay
                # floor, so the gate value alpha_t at the reset position is irrelevant.
                state = state * (~reset_mask[:, t]).to(acc).view(B, 1, 1, 1)
            state = a.unsqueeze(-1) * state + kt.unsqueeze(-1) * vt.unsqueeze(-2)  # [B,H,dk,dv]
            outs.append(torch.einsum("bhk,bhkv->bhv", qt, state))
        o = torch.stack(outs, dim=1)                            # [B,S,H,dv]
        # output → input dtype (residual); STATE stays in acc dtype (fp32) — it is a recurrent
        # accumulator carried across loop iterations, kept high-precision (tiny: ~0.79 MB).
        return o.to(q.dtype), state

    # ── chunk-parallel form (the training path) ───────────────────────────
    def _chunked(self, q, k, v, log_alpha, S0, reset_mask=None):
        B, S, H, dh = q.shape
        C = self.chunk
        acc = self._acc_dtype(q.dtype)
        q, k, v, log_alpha = (t.to(acc) for t in (q, k, v, log_alpha))
        state = (S0.to(acc) if S0 is not None else q.new_zeros(B, H, dh, dh))
        n_chunks = (S + C - 1) // C
        outs = []
        for c in range(n_chunks):
            s, e = c * C, min((c + 1) * C, S)
            L = e - s
            qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]        # [B,L,H,dh]
            la = log_alpha[:, s:e]                              # [B,L,H,dh] (≤0)
            b = la.cumsum(dim=1)                                # cumulative log-gate within chunk
            if reset_mask is not None:
                # Segment reset (docs/tul-tg-spec.md §4), EXACT within fp32: recompute the
                # cumulative log-gate RELATIVE TO EACH RESET SEGMENT instead of one
                # chunk-global cumsum. The chunk-global form composes with the -30
                # overflow clamp below only when the chunk holds ONE segment; with
                # several resets per chunk the global cumsum dives past -30 and the
                # clamp pins every later position to the same value, destroying the
                # relative decay (measured up to ~780% rel err at span-density resets —
                # the tg_restrict regime). Per-segment offsets make each segment exactly
                # the single-segment case the clamp was designed for. Three deltas vs
                # the no-reset path: (a) b is offset per segment, (b) intra-chunk pairs
                # crossing a reset are masked, (c) the carried state feeds only the
                # pre-first-reset positions and only the LAST segment leaves the chunk.
                rm_c = reset_mask[:, s:e]                       # [B, L] bool
                seg = rm_c.to(torch.long).cumsum(dim=1)         # [B, L]; 0 = carry segment
                idx = torch.arange(L, device=q.device)
                # index of each position's segment start (-1 = carry segment: measure
                # from the chunk start, i.e. keep the raw cumsum, matching no-reset math)
                start = torch.where(rm_c, idx.unsqueeze(0).expand_as(rm_c),
                                    torch.full_like(seg, -1)).cummax(dim=1).values
                b_start = torch.gather(
                    b, 1, start.clamp(min=0)[:, :, None, None].expand_as(b))
                b = torch.where((start >= 0)[:, :, None, None], b - b_start, b)
                carry_pos = (seg == 0)                          # [B, L]
                same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)  # [B, L, L]
            else:
                carry_pos = same_seg = None
            # Clamp the cumulative log-gate floor: the intra-chunk form factors the per-channel
            # decay as exp(b_t)·exp(-b_j); the product exp(b_t-b_j)≤1 is bounded but exp(-b_j)
            # alone OVERFLOWS fp32 once b_j is very negative (aggressive forget over a long chunk).
            # Floor at -30 → exp(30)≈1e13 (safe; squared still ≪ fp32 max). Channels decayed past
            # e^-30 contribute ~0, so this is numerically faithful, not an approximation that bites.
            b = b.clamp(min=-30.0)
            b_exp = b.exp()                                     # B_t = prod_{j<=t} alpha_j
            # ---- inter-chunk: contribution of the carried state to each position ----
            # o_inter_t = (q_t ⊙ B_t) · state
            q_dec = qc * b_exp                                  # [B,L,H,dh]
            if carry_pos is not None:
                # (c): the state carried into the chunk is visible ONLY before the
                # chunk's first reset. For carry positions b is the raw cumsum (their
                # segment start is the chunk start), so q_dec there matches no-reset.
                o_inter = torch.einsum(
                    "blhk,bhkv->blhv", q_dec * carry_pos[:, :, None, None], state)
            else:
                o_inter = torch.einsum("blhk,bhkv->blhv", q_dec, state)
            # ---- intra-chunk: decay-masked linear attention within the chunk ----
            # score_{t,j} = (q_t ⊙ B_t) · (k_j / B_j)   for j <= t
            k_dec = kc * (-b).exp()                             # k_j / B_j
            scores = torch.einsum("blhk,bmhk->bhlm", q_dec, k_dec)   # [B,H,L,L]
            causal = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            if same_seg is None:
                scores = scores.masked_fill(~causal, 0.0)
            else:  # (b): additionally mask every pair that crosses a reset
                scores = scores.masked_fill(~(causal & same_seg).unsqueeze(1), 0.0)
            o_intra = torch.einsum("bhlm,bmhv->blhv", scores, vc)
            outs.append((o_inter + o_intra).to(q.dtype))
            # ---- state update to end of chunk ----
            # state <- diag(B_L) state + sum_j (k_j ⊙ B_L/B_j)^T v_j
            B_L = b_exp[:, -1]                                  # [B,H,dh]  (prod over whole chunk)
            decay_to_end = (b[:, -1:].exp() * (-b).exp())       # B_L / B_j, [B,L,H,dh]
            if carry_pos is not None:
                # (c): only the LAST segment survives the chunk. Rows with any reset in
                # the chunk zero the old-state carry; earlier segments' kv terms zero out.
                B_L = B_L * carry_pos[:, -1].to(B_L.dtype).view(-1, 1, 1)
                in_last = (seg == seg[:, -1:])                  # [B, L]
                decay_to_end = decay_to_end * in_last[:, :, None, None]
            k_end = kc * decay_to_end
            kv = torch.einsum("blhk,blhv->bhkv", k_end, vc)
            state = B_L.unsqueeze(-1) * state + kv
        o = torch.cat(outs, dim=1)                              # [B,S,H,dv]
        return o.to(q.dtype), state            # state stays acc dtype (fp32 carry), o → input dtype

    def forward(self, x: Tensor, initial_state: Tensor | None = None,
                return_state: bool = True, reset_mask: Tensor | None = None):
        q, k, v, log_alpha, r_pre = self._project(x)
        # reset_mask ([B, S] bool | None): GLA segment reset (docs/tul-tg-spec.md §4).
        # Implemented STRUCTURALLY inside _recurrent (state zeroed entering a reset
        # position) and _chunked (per-segment cumulative gate + cross-reset pair
        # masking) — NOT as a log_alpha floor: a -30 floor collides with _chunked's
        # pre-existing -30 overflow clamp once a chunk holds several resets (measured
        # up to ~780% rel err at span-density resets before this formulation).
        if self.mode == "recurrent":
            o, final_state = self._recurrent(q, k, v, log_alpha, initial_state,
                                             reset_mask=reset_mask)
        elif self.mode == "kernel":
            if reset_mask is not None:
                raise NotImplementedError(
                    "GLA reset_mask has no fused-kernel implementation; tg_restrict "
                    "requires use_kernels=false, which builds mode='chunked'.")
            # Fused Triton chunked-GLA (sm_120): 2.9× the eager fwd+bwd, grads cos 1.0 vs the
            # recurrent oracle, final_state kept fp32. Gate: ignore/verify_fused_gla.py.
            #
            # SMEM GUARD: the fwd kernel holds ~[DH,DH] fp32 tiles + chunk buffers in shared
            # memory — measured 114688 B at DH=128, i.e. ≈ 7·DH². A GPU whose per-block opt-in
            # smem is smaller than that OOMs the launch (triton OutOfResources). The RTX 5090
            # (sm_120) opt-in limit is 101376 B, so retention_heads=6 at d=768 → DH=128 →
            # 7·128²=114688 > 101376 fails. The eager _chunked path is numerically identical
            # (parity-gated), so fall back for head_dims the fused kernel can't fit on this GPU.
            DH = q.shape[-1]
            try:
                _optin = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
            except Exception:
                _optin = 101376  # sm_120 default
            if 7 * DH * DH <= _optin:
                from morph.kernels.triton.fused_gla import fused_gla
                o, final_state = fused_gla(q, k, v, log_alpha, initial_state)
            else:
                o, final_state = self._chunked(q, k, v, log_alpha, initial_state)
        else:  # "chunked" eager reference
            o, final_state = self._chunked(q, k, v, log_alpha, initial_state,
                                           reset_mask=reset_mask)
        out = self._readout(x, o, r_pre)
        return (out, final_state) if return_state else out
