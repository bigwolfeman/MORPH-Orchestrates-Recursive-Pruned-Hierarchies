"""Fused, CUDA-graphed, batched decoder for the STRICT LXTUL slot-loop model.

Design and op mapping: ``.agents/notes/proposed/feature/2026-10-10-lxtul-fused-decode.md``.

The function is the one ``tul_generate_graphed.TulGraphDecoder`` computes (the same
refusals, ``tul_generate_cached._check_supported``); what changes is HOW. Every block
application runs on the plain decode engine's kernels (``morph/inference/engine.py``,
``morph/kernels/triton/fused_decode_step.py``): the Hyper-Connection pre-map / post with
the sublayer norm folded in, ``decode_front`` (projection, conv, prologue, K/V store), the
``W_up`` GEMV, the ternary MLP GEMVs, ``rmsnorm_rows``, the LM-head GEMV. The strict
geometry's two attention branches are ONE kernel, ``lx_attn`` (spec K1 in the note). The
rest (seed, register, router, reset, winner write, pointer mixture) are the model's own
modules on batched tensors inside the graphs.

Two kinds of event, each one CUDA graph:

* the TOKEN step: every one of the B rows feeds one token (prelude, coda, head, pointer);
* the SLOT step, one graph per bucket size: the rows whose token closed a span this step,
  gathered into the smallest bucket that holds them and padded with the dummy cache row
  ``B`` (nothing reads it). Seed, the M cells' prelude, the loop (6 passes x 6 layers on
  M cells), the winner write, the coda of the Kp cells.

Why the strict geometry lets per-row caches be exact (``tul_generate_cached`` docstring):
the prelude and the coda token read their own span's tokens (and, the coda, every
earlier cell); the loop reads cells of slots ``<= s``; a coda cell reads itself. Every
read is therefore a key RANGE of the row's own cache, and the segment reset of the conv
and the value shift is a ZERO row in the conv history (``decode_front`` recomputes the
latents of its 7 history rows from the stored normed inputs; the projections and the
convs have no bias, so a zero row gives exactly the zero latent the segment cut gives).

Layout (the contract with the kernels, note section "Buffer and batching layout"):

* cache rows ``R = B + 1``; every K/V cache is ``[R, H, N_site, D]`` fp32;
* query rows: the token step ``B`` (query row b = cache row b); a slot bucket of ``nb``
  rows ``nb * M`` (cell i of bucket row q at ``q*M + i``, cache row ``r_q``);
* conv history: token sites ``[B, P + 2, C]`` (span rows, a zero row ``P``, the staging
  row ``P + 1``); cell sites one flat ``[Bmax * cmax + 1, C]`` table passed expanded with
  stride 0 and absolute ``x_off`` rows; loop sites a flat table per site: a ring of
  ``RW`` cells per cache row, a staging block, a zero row.

``use_graphs=False`` runs the same step functions eagerly (debugging). There is no CPU
path: the engine's kernels are Triton on CUDA.
"""

from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parametrize

from morph.inference.engine import hc_apply
from morph.inference.sampling import sample_next
from morph.inference.tul_generate import TulRowBuilder
from morph.inference.tul_generate_cached import _check_supported, loop_cell_count
from morph.kernels.triton.fused_decode_step import (bf16_gemv, decode_front, loop_ssm_term,
                                                    ring_commit, rmsnorm_rows, small_gemv,
                                                    swiglu_rows, ternary_gemv,
                                                    ternary_gemv_rs, ternary_pack)
from morph.model.tul_fan_route import reset_to_winner
from morph.model.tul_layout import BoundaryRule, SlotLayout, TulLayoutSpec
from morph.model.tul_pointer import _rms as _ptr_rms

__all__ = ["LxFusedDecoder", "generate_lx_fused_batch", "lx_attn_reference"]


# ─────────────────────────────────────────────────────────────────────────────
# K1 (lx_attn) and K2 (decode_front kv_row): the kernels or their PyTorch stand-ins
# ─────────────────────────────────────────────────────────────────────────────


def lx_attn_reference(q: Tensor, k_cache: Tensor, v_cache: Tensor, meta: Tensor,
                      kpos: Tensor | None, gateh: Tensor, w_gate2: Tensor, xres: Tensor,
                      sink: Tensor, alpha: Tensor, window: int, scale: float) -> Tensor:
    """The K1 contract in PyTorch (note, "Kernel specs"): ``attn="reference"`` runs it in
    place of ``morph/kernels/triton/lx_decode_attn.py`` (the default since K1 landed is
    the kernel); tests/test_lx_fused_decode.py runs the decoder with both.

    ``meta`` ``[B', 8]`` int32 = (row, lo1, hi1, lo2, hi2, qidx1, qidx2, qpos). Range 1
    ``[lo1, hi1)`` = slot columns: all of them in the slot branch (with the per-head sink
    logit, zero value); in the window iff ``0 < |d1| < window``, ``d1 = qidx1 - k`` or,
    with ``kpos``, ``qpos - kpos[row, k]``. Range 2 ``[lo2, hi2)`` = window only,
    ``d2 = qidx2 - k``. An empty window set gives 0. Then the gate combine of
    ``_CCABase._gate_combine_up`` (SiLU, gate2, sigmoid, ``alpha * x_res``), before
    ``W_up``. Returns ``[B', H*D]`` fp32."""
    Bq, H, D = q.shape
    N = k_cache.shape[2]
    m = meta.long()
    row = m[:, 0]
    K = k_cache.index_select(0, row)                                   # [B', H, N, D]
    V = v_cache.index_select(0, row)
    s = torch.einsum("bhd,bhnd->bhn", q.float(), K.float()) * scale
    k = torch.arange(N, device=q.device).view(1, N)
    lo1, hi1, lo2, hi2 = (m[:, i:i + 1] for i in range(1, 5))
    qi1, qi2, qp = m[:, 5:6], m[:, 6:7], m[:, 7:8]
    in1 = (k >= lo1) & (k < hi1)
    in2 = (k >= lo2) & (k < hi2)
    d1 = (qp - kpos.index_select(0, row).long()) if kpos is not None else (qi1 - k)
    d2 = qi2 - k
    win = (in1 & (d1.abs() < window) & (d1 != 0)) | (in2 & (d2.abs() < window) & (d2 != 0))
    neg = float("-inf")
    sc = s.masked_fill(~in1.unsqueeze(1), neg)
    sc = torch.cat([sc, sink.float().view(1, H, 1).expand(Bq, H, 1)], dim=-1)
    pc = torch.softmax(sc, dim=-1)[..., :N]
    oc = torch.einsum("bhn,bhnd->bhd", pc, V.float())
    has = win.any(dim=-1).view(Bq, 1, 1)
    sw = s.masked_fill(~win.unsqueeze(1), neg)
    sw = torch.where(has, sw, torch.zeros_like(sw))
    ow = torch.einsum("bhn,bhnd->bhd", torch.softmax(sw, dim=-1), V.float())
    ow = torch.where(has, ow, torch.zeros_like(ow))
    g = torch.sigmoid(F.linear(F.silu(gateh.float()), w_gate2.float())).view(Bq, H, 2)
    out = (g[..., 0:1] * oc + g[..., 1:2] * ow
           + alpha.float().view(1, H, 1) * xres.float().view(Bq, H, D))
    return out.reshape(Bq, H * D)


def _resolve_kernels(attn: str, kv_row: str):
    """``(lx_attn callable, use decode_front's kv_row)``. ``attn``: "kernel" = K1 from
    ``morph/kernels/triton/lx_decode_attn.py``, "reference" = :func:`lx_attn_reference`.
    ``kv_row``: "kernel" = K2 (``decode_front(..., kv_row=)``), "reference" = a
    one-entry scratch scattered to the cache rows. Explicit, never auto-detected: a kernel
    file that exists is not yet a kernel that passed its tests."""
    if attn == "kernel":
        from morph.kernels.triton.lx_decode_attn import lx_attn as k1
    elif attn == "reference":
        k1 = lx_attn_reference
    else:
        raise ValueError(f"attn must be 'kernel' or 'reference', got {attn!r}")
    if kv_row == "kernel":
        import inspect
        if "kv_row" not in inspect.signature(decode_front).parameters:
            raise RuntimeError("decode_front has no kv_row argument (K2 not landed)")
    elif kv_row != "reference":
        raise ValueError(f"kv_row must be 'kernel' or 'reference', got {kv_row!r}")
    return k1, kv_row == "kernel"


# ─────────────────────────────────────────────────────────────────────────────
# Per-block weight copies (the engine's packing, read once)
# ─────────────────────────────────────────────────────────────────────────────


class _Layer:
    """One MORPHBlock's decode-time weights: the stacked CCA projection, the conv pack,
    the gate, ``W_up``, the HC mapping copies, the packed ternary MLP."""

    def __init__(self, block, wdt: torch.dtype):
        impl = block.attention._impl
        cca = impl.cca
        self.block, self.cca = block, cca
        mods = [cca.W_down_q, cca.W_down_k, cca.W_v_prev, cca.W_v_curr, cca.gate[0]]
        self.Wqkv = torch.cat([m.weight.detach() for m in mods], dim=0).contiguous().to(wdt)
        self.conv_pack = (
            cca.conv_q_dw.weight.detach().reshape(cca.latent_q_dim, -1).contiguous(),
            cca.conv_q_gp.weight.detach().contiguous(),
            cca.conv_k_dw.weight.detach().reshape(cca.latent_k_dim, -1).contiguous(),
            cca.conv_k_gp.weight.detach().contiguous(),
            cca.q_norm.weight.detach().contiguous(),
            cca.k_norm.weight.detach().contiguous())
        self.qk_eps = float(cca.q_norm.eps)
        self.exp_temp = torch.exp(cca.temp.detach()).float().view(-1).contiguous()
        self.alpha = cca.alpha.detach().float().view(-1).contiguous()
        self.sink = cca.sink_logits.detach().float().contiguous()
        self.Wgate2 = cca.gate[2].weight.detach().contiguous().to(wdt)
        self.Wup = cca.W_up.weight.detach().contiguous().to(wdt)
        self.na_w, self.na_eps = block.norm_attn.weight.detach(), float(block.norm_attn.eps)
        self.nm_w, self.nm_eps = block.norm_mlp.weight.detach(), float(block.norm_mlp.eps)
        self.hc = {}
        for mod in (block.mrr_attn, block.mrr_mlp):
            self.hc[id(mod)] = (mod.proj.weight.detach().contiguous().to(wdt),
                                mod.proj.bias.detach().float().contiguous())
        mlp = block.mlp[0] if hasattr(block.mlp, "__getitem__") else block.mlp
        gu, dn = mlp.gate_up._cms, mlp.down._cms
        if getattr(gu, "_mortar", False) or not gu._dense_mode:
            raise NotImplementedError("lx_fused_decode: a carved or routed MLP (dense MLPs only)")
        if getattr(mlp, "router", None) is not None:
            raise NotImplementedError("lx_fused_decode: a ReMoE-routed MLP")
        wgu, wdn = gu.weight.detach().float(), dn.weight.detach().float()
        self.Tgu, self.Tdown = ternary_pack(wgu), ternary_pack(wdn)
        if self.Tgu is None or self.Tdown is None:
            self.Tgu = self.Tdown = None
            self.Wgu, self.Wdown = wgu.contiguous(), wdn.contiguous()


class _Site:
    """Where one attention application reads and writes (see the module docstring)."""

    __slots__ = ("x_hist", "x_off", "kc", "vc", "widx", "kv_row", "cos", "sin", "meta",
                 "kpos", "nout", "snout")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


# Host step-input table fields (``LxFusedDecoder._host_token`` / ``_host_slot``).
_T_PREV, _T_TOK, _T_POS, _T_J, _T_TI, _T_CJ, _T_JP1, _T_TIM1 = range(8)
_TOK_F = 8
(_S_R, _S_S, _S_N, _S_LAST, _S_RM, _S_PPOS, _S_PW, _S_C, _S_DST, _S_RK, _S_CK, _S_PK,
 _S_BAG) = range(13)
_SLOT_F = 13
_HIST6 = np.arange(-6, 0, dtype=np.int64)      # a conv window's 6 earlier positions


# ─────────────────────────────────────────────────────────────────────────────
# The decoder
# ─────────────────────────────────────────────────────────────────────────────


class LxFusedDecoder(contextlib.AbstractContextManager):
    """Static buffers, the two step functions and their CUDA graphs for B rows.

    A context manager: it holds ``parametrize.cached()`` open for its life (the graphs read
    the cached QAT weights of the modules it calls at fixed addresses). Reusable across
    generations (``reset``)."""

    def __init__(self, model, spec: TulLayoutSpec, rule: BoundaryRule, batch_size: int = 1,
                 emit_source: str = "token", exact_weights: bool = True,
                 use_graphs: bool = True, max_tokens: int | None = None,
                 cells: int | None = None, attn: str = "kernel", kv_row: str = "kernel"):
        if emit_source not in ("slot", "token"):
            raise ValueError(f"emit_source must be 'slot' or 'token', got {emit_source!r}")
        _check_supported(model)
        m, cfg, tc = model, model.cfg, model.cfg.tul
        if m.tul_code_enum is not None:
            raise NotImplementedError("lx_fused_decode: tul.code_enum_k > 1 (the K-rollout "
                                      "model; use tul_generate_graphed)")
        if type(m.core_init).__name__ != "_CloneInit":
            raise NotImplementedError("lx_fused_decode: a loop entry other than h_0 = e "
                                      f"({type(m.core_init).__name__})")
        if getattr(m.injection, "B", None) is not None:
            raise NotImplementedError("lx_fused_decode: DiagonalInjection with B")
        if spec.prefix_k != tc.prefix_k:
            raise ValueError(f"spec prefix_k {spec.prefix_k} != model {tc.prefix_k}")
        self.dev = next(m.parameters()).device
        if self.dev.type != "cuda":
            raise ValueError("lx_fused_decode runs Triton kernels: the model must be on CUDA")
        self.m, self.spec, self.rule, self.emit_source = m, spec, rule, emit_source
        self.B = int(batch_size)
        self.R = self.B + 1                                  # + the dummy cache row
        self.use_graphs = bool(use_graphs)
        self.exact = bool(exact_weights)
        self.M = loop_cell_count(m, cells)
        self.Kp = int(tc.prefix_k)
        self.lsel = m._lsel_mode != "off"
        self.ptr = m.tul_pointer is not None
        self.depth = (int(tc.slot_depth_fixed) if int(tc.slot_depth_fixed) > 0
                      else int(tc.slot_mean_depth or cfg.mean_depth))
        self.P = int(rule.span_cap)
        self.S = int(spec.max_slots)
        self.Cc = self.Kp * self.S
        self.slot_id = int(tc.slot_id)
        self.max_pos = int(m.prelude[0].attention._impl.cca.rope.cos_cached.shape[2])
        self.max_tokens = int(max_tokens or self.max_pos)
        nb = 1
        self.buckets = []
        while True:
            self.buckets.append(nb)
            if nb >= self.B:
                break
            nb *= 2
        self.Bmax = self.buckets[-1]
        self.lx_attn, self.k2 = _resolve_kernels(attn, kv_row)
        self.attn_impl, self.kv_row_impl = attn, kv_row
        self._ctx = contextlib.ExitStack()
        self._ctx.enter_context(parametrize.cached())
        try:
            with torch.no_grad():
                self._build()
        except BaseException:
            self._ctx.close()
            raise

    # -- construction -------------------------------------------------------------
    def _build(self) -> None:
        m, cfg, dev = self.m, self.m.cfg, self.dev
        f32 = torch.float32
        wdt = f32 if self.exact else torch.bfloat16
        self.C = C = int(cfg.d_model)
        self.N = int(m._n_streams)
        cca0 = m.prelude[0].attention._impl.cca
        self.H, self.D, self.Hkv = cca0.n_heads, cca0.d_head, cca0.n_kv_heads
        self.LQ, self.LK = cca0.latent_q_dim, cca0.latent_k_dim
        self.Vh = cca0.W_v_curr.out_features
        self.GH = cca0.gate[0].out_features
        self.window = int(cca0.window_size)
        self.k_conv = int(cca0.conv_q_dw.kernel_size[0])
        if self.k_conv != 4:
            raise NotImplementedError(f"decode_front unrolls a k=4 conv, model has {self.k_conv}")
        if 2 * self.Vh != self.Hkv * self.D or self.GH % self.D:
            raise NotImplementedError("decode_front value / gate layout (2 Vh = Hkv D, GH % D)")
        for blk in list(m.prelude) + list(m.core) + list(m.coda):
            c = blk.attention._impl.cca
            if (c.window_size, c.n_heads, c.d_head) != (self.window, self.H, self.D):
                raise NotImplementedError("lx_fused_decode: blocks with different attention widths")
        self.np_, self.nc, self.nd = cfg.n_prelude, cfg.n_core, cfg.n_coda
        self.pre = [_Layer(b, wdt) for b in m.prelude]
        self.core = [_Layer(b, wdt) for b in m.core]
        self.coda = [_Layer(b, wdt) for b in m.coda]
        self.hcb_zero = torch.zeros(3 * self.N * self.N, device=dev, dtype=f32)
        rope = cca0.rope
        self.cos_flat = rope.cos_cached.reshape(-1, self.D).contiguous()
        self.sin_flat = rope.sin_cached.reshape(-1, self.D).contiguous()
        # injection terms of every layer: one stacked x0 GEMV (the engine's packing)
        n_tot = self.np_ + self.nc + self.nd
        self.n_tot = n_tot
        self.cs, self.ce = m._ctx_start, m._ctx_end
        x0_w = [m.x0_injects[i].proj.weight.detach().float() for i in range(n_tot)]
        packs = [ternary_pack(w) for w in x0_w]
        if all(p is not None for p in packs):
            self.Tx0 = torch.cat([p[0] for p in packs]).contiguous()
            self.Tx0_rs = torch.cat([p[1].expand(w.shape[0])
                                     for p, w in zip(packs, x0_w)]).contiguous()
            self.W_x0 = None
        else:
            self.Tx0 = self.Tx0_rs = None
            self.W_x0 = torch.cat(x0_w, dim=0).contiguous().to(wdt)
        self.x0_scales = torch.stack([m.x0_injects[i].log_scale.detach()
                                      for i in range(n_tot)]).float().view(n_tot, 1, 1)
        self.lambdas = m.embed.bigram.lambdas.detach().float().view(n_tot, 1, 1)
        # value embeddings (prelude layers 0..n_ve-1): the per-token signal
        # ``scale_k * W_k table_k(tok)``; a cell gets its span's mean (``ve_bagged``)
        self.n_ve = len(m._ve_layer_map)
        if m._ve_layer_map != list(range(self.n_ve)) or self.n_ve > self.np_:
            raise NotImplementedError("lx_fused_decode: value embeddings outside prelude 0..n-1")
        self.ve_proj = []
        for k in range(self.n_ve):
            w = m.value_embeds[k].proj.weight.detach().float()
            pk = ternary_pack(w)
            self.ve_proj.append((pk[0], pk[1].expand(w.shape[0]).contiguous(), None)
                                if pk is not None else (None, None, w.contiguous().to(wdt)))
        self.ve_scales = [m.value_embeds[k].log_scale.detach().float() for k in range(self.n_ve)]
        # readout: lm_mixer (frozen scales) + final norm; the tied head
        mix = m.lm_mixer
        sc = F.softplus(mix.channel_scales.detach())
        self.lm_scale = torch.cat([sc[i].expand(w)
                                   for i, w in enumerate(mix.channel_dims)]).contiguous()
        w_mix = mix.mix.weight.detach().float()
        p_mix = ternary_pack(w_mix)
        if p_mix is not None:
            self.Tmix, self.Tmix_rs = p_mix[0], p_mix[1].expand(w_mix.shape[0]).contiguous()
            self.W_mix = None
        else:
            self.Tmix = self.Tmix_rs = None
            self.W_mix = w_mix.contiguous().to(wdt)
        self.W_head = m.embed.lm_weight().detach().contiguous().to(wdt)
        self.V = int(self.W_head.shape[0])
        inj = m.injection
        self.loop_A = inj.log_A.detach().exp().clamp(max=0.9999).contiguous()
        self.loop_dt = inj.log_dt.detach().exp().contiguous()
        self.loop_lo, self.loop_hi = int(inj.start), int(inj.end)
        if m.tul_cell_norm is not None:
            self.cn_w = m.tul_cell_norm.weight.detach()
            self.cn_eps = float(m.tul_cell_norm.eps)
            self.cn_rms = m._cell_norm_mode == "rms"
        # ── state buffers ─────────────────────────────────────────────────────────
        B, R, P, M, Kp, S, H, D = self.B, self.R, self.P, self.M, self.Kp, self.S, self.H, self.D
        def z(*sh):
            return torch.zeros(*sh, device=dev, dtype=f32)

        self.X_tok = z(self.np_ + self.nd, B, P + 2, C)          # zero row P, staging P+1
        self.cmax = max(M, Kp)
        self.CX = z(self.Bmax * self.cmax + 1, C)                # cell conv history table
        self.CX_zero = self.Bmax * self.cmax
        self.RW = max(8, M)                                      # loop history cells / row
        self.L_stage = R * self.RW
        self.L_zero = self.L_stage + self.Bmax * M
        self.n_ls = self.depth * self.nc
        self.XL = z(self.n_ls, self.L_zero + 1, C)
        self.Kpre = z(self.np_, R, H, P + M, D)
        self.Vpre = z(self.np_, R, H, P + M, D)
        self.Kcoda = z(self.nd, R, H, self.Cc + P, D)
        self.Vcoda = z(self.nd, R, H, self.Cc + P, D)
        self.Kloop = z(self.n_ls, R, H, S * M, D)
        self.Vloop = z(self.n_ls, R, H, S * M, D)
        self.kpos = torch.zeros(R, self.Cc + P, device=dev, dtype=torch.int32)
        self.span_bg = z(R, P, C)
        self.span_ve = z(max(self.n_ve, 1), R, P, self.ce - self.cs)
        self.span_emb = z(R, P, C)
        self.span_xn = z(R, P, C) if m.tul_register is not None else None
        if self.ptr:
            p = m.tul_pointer
            self.ptr_keys = z(B, p.h, self.max_tokens, p.dh)
            self.ptr_cand = torch.zeros(B, self.max_tokens, dtype=torch.long, device=dev)
        self.logp_tok = z(B, self.V)
        self.next_tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.logp_cell = z(R, self.V)
        self.next_cell = torch.zeros(R, dtype=torch.long, device=dev)
        self.arB = torch.arange(B, device=dev)
        self.slot_col = torch.tensor([self.slot_id], device=dev)
        self.arM = torch.arange(M, device=dev)
        if self.ptr:
            self.arNk = torch.arange(self.max_tokens, device=dev).view(1, -1)
        nbm = self.Bmax * self.cmax
        self.q_stage_all = self.L_stage + torch.arange(self.Bmax * M, device=dev)
        self.ones_q = torch.ones(self.Bmax, 1, dtype=torch.bool, device=dev)
        self.reg_mask = torch.zeros(self.Bmax, P, dtype=torch.bool, device=dev)
        self.reg_index = torch.zeros(self.Bmax, 1, dtype=torch.long, device=dev)
        # Host -> device step inputs: every index vector and every kernel table (x_off,
        # lx_attn meta) of an event is built on the host (numpy, vectorised over rows) into
        # pinned tables and sent in two copies, instead of ~130 (token) / ~280 (slot)
        # small index kernels per event inside the graphs (0.17 / 0.41 ms at B = 1,
        # measured). Field-major layouts: every field is a CONTIGUOUS row (the kernels
        # read index vectors as ``POSP + b``).
        self.h_tok = torch.zeros(_TOK_F, B, dtype=torch.long, pin_memory=True)
        self.h_tok32 = torch.zeros(3, B, 8, dtype=torch.int32, pin_memory=True)
        self.d_tok = torch.zeros(_TOK_F, B, dtype=torch.long, device=dev)
        self.d_tok32 = torch.zeros(3, B, 8, dtype=torch.int32, device=dev)
        self.W_slot = max(nbm, self.Bmax * P)
        self.h_slot = torch.zeros(_SLOT_F, self.W_slot, dtype=torch.long, pin_memory=True)
        self.h_slot32 = torch.zeros(7, nbm, 8, dtype=torch.int32, pin_memory=True)
        self.d_slot = torch.zeros(_SLOT_F, self.W_slot, dtype=torch.long, device=dev)
        self.d_slot32 = torch.zeros(7, nbm, 8, dtype=torch.int32, device=dev)
        self.h_tok_np, self.h_tok32_np = self.h_tok.numpy(), self.h_tok32.numpy()
        self.h_slot_np, self.h_slot32_np = self.h_slot.numpy(), self.h_slot32.numpy()
        self._ev_tok, self._ev_slot = torch.cuda.Event(), torch.cuda.Event()
        self.g_tok = None
        self.g_slot: dict[int, torch.cuda.CUDAGraph] = {}
        if self.use_graphs:
            self._capture()
        self.reset()

    def _capture(self) -> None:
        self._fill_warmup_inputs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._token_step()
                for nb in self.buckets:
                    self._slot_step(nb)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.g_tok = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_tok):
            self._token_step()
        for nb in self.buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.g_tok.pool()):
                self._slot_step(nb)
            self.g_slot[nb] = g
        torch.cuda.synchronize()

    def _fill_warmup_inputs(self) -> None:
        # token: every row at position 0 of an empty span; slot: every bucket row the
        # dummy row with a one-token span (no division by zero, no read outside row B)
        zero = np.zeros(self.B, dtype=np.int64)
        self._host_token(np.full(self.B, 5, dtype=np.int64), zero, zero, zero, zero, zero)
        self.d_tok.copy_(self.h_tok)
        self.d_tok32.copy_(self.h_tok32)
        self._host_slot([], self.Bmax)
        self.d_slot.copy_(self.h_slot)
        self.d_slot32.copy_(self.h_slot32)

    # -- host-built step inputs ---------------------------------------------------------
    def _host_token(self, tok, prev, pos, j, ti, ncell) -> None:
        """The token step's inputs for all B rows (numpy int64 ``[B]`` each) into the
        pinned tables: ids and positions, the conv-history rows ``x_off`` (span rows
        j-6..j-1, the zero row P before the span, the staging row P+1) and the two
        ``lx_attn`` meta tables (note, "meta table")."""
        P, Cc = self.P, self.Cc
        h = self.h_tok_np
        h[_T_PREV], h[_T_TOK], h[_T_POS], h[_T_J], h[_T_TI] = prev, tok, pos, j, ti
        h[_T_CJ], h[_T_JP1], h[_T_TIM1] = j + Cc, j + 1, np.maximum(ti - 1, 0)
        hist = j[:, None] + _HIST6[None, :]
        x = self.h_tok32_np
        x[0, :, :6] = np.where(hist >= 0, hist, P)
        x[0, :, 6], x[0, :, 7] = P + 1, 0
        b = np.arange(self.B)
        x[1] = 0
        x[1, :, 0], x[1, :, 4], x[1, :, 6] = b, j, j               # prelude: [0, j) window
        x[2, :, 0], x[2, :, 1], x[2, :, 2] = b, 0, ncell            # coda: every cell,
        x[2, :, 3], x[2, :, 4], x[2, :, 5] = Cc, Cc + j, 0          # the span's tokens,
        x[2, :, 6], x[2, :, 7] = Cc + j, pos                        # cells by position

    def _cell_xoff(self, nb: int, cnt: int) -> np.ndarray:
        """``x_off`` ``[nb*cnt, 8]`` of the cells of a fresh segment: cell i of bucket row
        q reads cells 0..i of its own block ``q*cnt`` of the cell table, the zero row
        before them."""
        i = np.tile(np.arange(cnt), nb)
        base = np.repeat(np.arange(nb), cnt) * cnt
        hist = i[:, None] + _HIST6[None, :]
        out = np.empty((nb * cnt, 8), dtype=np.int32)
        out[:, :6] = np.where(hist >= 0, base[:, None] + hist, self.CX_zero)
        out[:, 6], out[:, 7] = base + i, 0
        return out

    def _host_slot(self, entries: list[tuple[int, int, int, int, int]], nb: int) -> None:
        """The slot step's inputs for a bucket of ``nb`` rows: ``entries`` = (cache row r,
        slot index s, first row position fp, span length n, last token) per cutting row;
        the rest of the bucket is the dummy row B with a one-token span."""
        M, Kp, P, RW = self.M, self.Kp, self.P, self.RW
        pad = [(self.B, 0, 0, 1, 0)] * (nb - len(entries))
        r, s, fp, n, last = (np.array(v, dtype=np.int64) for v in zip(*(list(entries) + pad)))
        h = self.h_slot_np
        h[_S_R, :nb], h[_S_S, :nb], h[_S_N, :nb], h[_S_LAST, :nb] = r, s, n, last
        im, rm = np.tile(np.arange(M), nb), np.repeat(r, M)
        sm, fpm, nm = np.repeat(s, M), np.repeat(fp, M), np.repeat(n, M)
        c = sm * M + im
        q = nb * M
        h[_S_RM, :q], h[_S_PPOS, :q], h[_S_PW, :q], h[_S_C, :q] = rm, fpm + im, P + im, c
        h[_S_DST, :q] = rm * RW + c % RW
        ik, rk = np.tile(np.arange(Kp), nb), np.repeat(r, Kp)
        ck, pk = np.repeat(s, Kp) * Kp + ik, np.repeat(fp, Kp) + ik
        k = nb * Kp
        h[_S_RK, :k], h[_S_CK, :k], h[_S_PK, :k] = rk, ck, pk
        h[_S_BAG, :nb * P] = (np.arange(P)[None, :] >= n[:, None]).reshape(-1)
        x = self.h_slot32_np
        # the M cells' prelude: the span's tokens [0, n) and cells P..P+i (by index)
        x[0, :q] = self._cell_xoff(nb, M)
        x[1, :q] = 0
        x[1, :q, 0], x[1, :q, 1], x[1, :q, 2] = rm, P, P + im + 1
        x[1, :q, 4], x[1, :q, 5], x[1, :q, 6] = nm, P + im, nm + im
        # the loop on the compact cell axis: history = earlier cells (ring of RW per row),
        # this slot's earlier cells (staging rows), the zero row before cell 0
        hist = c[:, None] + _HIST6[None, :]
        cur = (sm * M)[:, None]
        stage = self.L_stage + np.arange(q)
        rows = np.where(hist < 0, self.L_zero,
                        np.where(hist >= cur, (stage - im)[:, None] + (hist - cur),
                                 rm[:, None] * RW + hist % RW))
        x[2, :q, :6], x[2, :q, 6], x[2, :q, 7] = rows, stage, 0
        x[3, :q] = 0
        x[3, :q, 0], x[3, :q, 2], x[3, :q, 5] = rm, (sm + 1) * M, c
        # the Kp coda cells: each reads itself alone
        x[4, :k] = self._cell_xoff(nb, Kp)
        x[5, :k] = 0
        x[5, :k, 0], x[5, :k, 1], x[5, :k, 2], x[5, :k, 5] = rk, ck, ck + 1, ck
        x[6, :k, 0] = pk

    def reset(self) -> None:
        """A new batch of rows: every count is host state, and every buffer entry is read
        only below its count, so nothing on the device needs clearing."""
        z = np.zeros(self.B, dtype=np.int64)
        self.n_tok, self.span_len, self.n_slots = z.copy(), z.copy(), z.copy()
        self.prev_id, self.last_tok = z.copy(), z.copy()
        # host counters of this generation: token steps, and slot steps per bucket size
        # (the bench reports them: a slot step costs ~4x a token step at B = 1)
        self.token_steps = 0
        self.slot_events: dict[int, int] = {}

    def __exit__(self, *exc) -> None:
        self.g_tok = None
        self.g_slot = {}
        self._ctx.close()

    # -- shared pieces ----------------------------------------------------------------
    def _ve_signal(self, tok: Tensor) -> Tensor:
        """``[n_ve, rows, ctx_w]``: ``value_embeds[k].precompute(value_embed_tables[k](tok))``."""
        out = []
        for k in range(self.n_ve):
            sig = self.m.value_embed_tables[k](tok).float().contiguous()       # [rows, C]
            T, rs, W = self.ve_proj[k]
            pr = ternary_gemv_rs(sig, T, rs) if T is not None else small_gemv(sig, W)
            out.append(self.ve_scales[k] * pr)
        return torch.stack(out)

    def _terms(self, x0: Tensor, bg: Tensor, ve: Tensor | None) -> Tensor:
        """``[L, rows, C]``: ``_build_injection_term`` for every layer at once, in its
        order: ``lam * bigram``, then the ctx slice ``+= (scale * W x0 [+ ve])``.
        ``ve`` ``[n_ve, rows, ctx_w]`` (the token's signal or the cell's span mean)."""
        rows = x0.shape[0]
        if self.Tx0 is not None:
            x0f = ternary_gemv_rs(x0.contiguous(), self.Tx0, self.Tx0_rs)
        else:
            x0f = small_gemv(x0.contiguous(), self.W_x0)
        x0t = x0f.view(rows, self.n_tot, -1).permute(1, 0, 2) * self.x0_scales
        if self.n_ve:
            x0t = torch.cat([x0t[:self.n_ve] + ve, x0t[self.n_ve:]])
        terms = self.lambdas * bg.unsqueeze(0)
        terms[..., self.cs:self.ce] += x0t
        return terms

    def _readout(self, x: Tensor) -> Tensor:
        """``[rows, C]`` from an HC carrier ``[rows, N, C]``: stream mean, ``lm_mixer``,
        ``final_norm`` (``MORPHTransformer._readout``)."""
        xs = (x.mean(dim=1) * self.lm_scale).contiguous()
        if self.Tmix is not None:
            y = ternary_gemv_rs(xs, self.Tmix, self.Tmix_rs)
        else:
            y = small_gemv(xs, self.W_mix)
        fn = self.m.final_norm
        return rmsnorm_rows(y.contiguous(), fn.weight, fn.eps)

    def _gemv(self, x: Tensor, w: Tensor, rows_gemm: int) -> Tensor:
        """``x @ w.T`` fp32 for ``rows`` = ``x.shape[0]`` rows. The engine's GEMV kernels
        read the weight once PER ROW (measured on the 5090, the [49169, 1024] fp32 head:
        121 / 472 / 944 us at 1 / 4 / 8 rows); cuBLAS reads it once (121 / 128 / 127 us).
        So: fp32 copies and at least ``rows_gemm`` rows -> ``F.linear`` (an fp32 SGEMM,
        the same products in another order; it would run TF32 if a caller turned
        ``torch.backends.cuda.matmul.allow_tf32`` on, as every cuBLAS fp32 op in this
        decoder and in its references would); else the engine GEMV, which also takes the
        bf16 copies with fp32 activations. ``rows`` is fixed per graph: no runtime branch."""
        if w.dtype == torch.float32 and x.shape[0] >= rows_gemm:
            return F.linear(x, w)
        return bf16_gemv(x, w) if w is self.W_head else small_gemv(x, w)

    def _logp(self, xh: Tensor) -> Tensor:
        logits = self._gemv(xh, self.W_head, 2)
        logits = logits.index_fill(-1, self.slot_col, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def _front(self, L: _Layer, st: _Site, rows: int) -> tuple:
        """``decode_front`` for ``rows`` query rows; K/V into ``st.kc`` / ``st.vc`` at cache
        row ``st.kv_row[b]`` (or b) and entry ``st.widx[b]``. Without K2 the stand-in
        writes into a one-entry scratch and scatters it to the cache rows."""
        dev = self.dev
        q = torch.empty(rows, self.H, self.D, device=dev, dtype=torch.float32)
        xres = torch.empty(rows, self.LQ, device=dev, dtype=torch.float32)
        gateh = torch.empty(rows, self.GH, device=dev, dtype=torch.float32)
        part = torch.empty(rows, 4, 7, L.Wqkv.shape[0], device=dev, dtype=torch.float32)
        # The kernel reads ``pos_dev[b]`` as ``POSP + b``: a strided view (a column of the
        # step-input table, the token prelude's ``j``) would read its neighbours' columns.
        # (measured 2026-10-10: with ``j = d_tok[:, 3]`` row b read ``d_tok[0, 3 + b]``, so
        # row 2 stored its prelude K/V at the ncell column's entry). A no-op for the other
        # sites, whose ``widx`` is a fresh tensor.
        if not st.widx.is_contiguous():
            st.widx = st.widx.contiguous()
        args = (self.H, self.Hkv, self.D, self.LQ, self.LK, self.Vh, self.GH, self.k_conv,
                L.qk_eps)
        if st.kv_row is None or self.k2:
            kw = {} if st.kv_row is None else {"kv_row": st.kv_row}
            decode_front(st.x_hist, st.x_off, st.widx, L.Wqkv, st.cos, st.sin, *L.conv_pack,
                         L.exp_temp, st.kc, st.vc, q, xres, gateh, None, part, *args, **kw)
        else:
            ks = torch.empty(rows, self.H, 1, self.D, device=dev, dtype=torch.float32)
            vs = torch.empty_like(ks)
            zero = torch.zeros(rows, dtype=torch.long, device=dev)
            decode_front(st.x_hist, st.x_off, zero, L.Wqkv, st.cos, st.sin, *L.conv_pack,
                         L.exp_temp, ks, vs, q, xres, gateh, None, part, *args)
            st.kc[st.kv_row, :, st.widx] = ks[:, :, 0]
            st.vc[st.kv_row, :, st.widx] = vs[:, :, 0]
        return q, xres, gateh

    def _block(self, L: _Layer, h: Tensor, st: _Site, term_next: Tensor | None,
               front_only: bool = False) -> Tensor:
        """``MORPHBlock.forward`` on the engine's kernels. ``h`` ``[Bh, Sh, N, C]``;
        ``front_only``: the attention pre-map and ``decode_front`` (the K/V store) and
        nothing after it; the caller discards the returned carrier."""
        blk = L.block
        Bh, Sh = h.shape[0], h.shape[1]
        rows = Bh * Sh
        C = self.C

        def _attn_fn(xbar: Tensor) -> Tensor:
            q, xres, gateh = self._front(L, st, rows)
            if front_only:
                return torch.zeros(Bh, Sh, C, device=self.dev, dtype=torch.float32)
            o = self.lx_attn(q, st.kc, st.vc, st.meta, st.kpos, gateh, L.Wgate2, xres,
                             L.sink, L.alpha, self.window, self.D ** -0.5)
            # W_up [C, H*D]: small_gemv 10 / 18 / 34 us at 16 / 32 / 64 rows, F.linear
            # 6.2 / 6.6 / 6.2 us (measured, the model's [1024, 512])
            return self._gemv(o, L.Wup, 16).view(Bh, Sh, C)

        def _mlp_fn(xbar: Tensor) -> Tensor:
            x = xbar.reshape(rows, 1, C).contiguous()
            if L.Tgu is not None:
                gu = ternary_gemv(x, *L.Tgu, rms_weight=L.nm_w, rms_eps=L.nm_eps)
                y = ternary_gemv(gu, *L.Tdown, swiglu_x=True)
            else:
                xn = rmsnorm_rows(x, L.nm_w, L.nm_eps)
                y = F.linear(swiglu_rows(F.linear(xn, L.Wgu)), L.Wdown)
            return y.view(Bh, Sh, C)

        wa, ba = L.hc[id(blk.mrr_attn)]
        h = hc_apply(blk.mrr_attn, h, _attn_fn, wa, ba, self.hcb_zero, norm_w=L.na_w,
                     norm_eps=L.na_eps, norm_out=st.nout, norm_stride=st.snout)
        if front_only:
            return h
        wm, bm = L.hc[id(blk.mrr_mlp)]
        return hc_apply(blk.mrr_mlp, h, _mlp_fn, wm, bm, self.hcb_zero, term=term_next)

    def _rope(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos_flat.index_select(0, pos), self.sin_flat.index_select(0, pos)

    # -- the token step ----------------------------------------------------------------
    def _token_step(self) -> None:
        m = self.m
        B, P, C, N = self.B, self.P, self.C, self.N
        d = self.d_tok
        tok, pos, j, ti, cj = d[_T_TOK], d[_T_POS], d[_T_J], d[_T_TI], d[_T_CJ]
        x_off, meta_pre, meta_coda = self.d_tok32[0], self.d_tok32[1], self.d_tok32[2]
        emb = m.embed(tok.view(B, 1))                                  # [B, 1, C]
        x = m.embed_drop(emb)
        bg = m.embed.get_bigram(d[_T_PREV:_T_TOK + 1].t())[:, 1:]       # [B, 1, C]
        x2, bg2 = x[:, 0].float(), bg[:, 0].float()
        ve = self._ve_signal(tok) if self.n_ve else None               # [n_ve, B, cw]
        terms = self._terms(x2, bg2, ve)                               # [L, B, C]
        self.span_bg[self.arB, j] = bg2
        if self.n_ve:
            self.span_ve[:, self.arB, j] = ve
        self.span_emb[self.arB, j] = emb[:, 0].float()
        cos, sin = self._rope(pos)
        # ── prelude: the span's earlier tokens; no slot column ─────────────────────
        meta = meta_pre
        h = (x2.view(B, 1, 1, C) + terms[0].view(B, 1, 1, C)).expand(B, 1, N, C).contiguous()
        for i, L in enumerate(self.pre):
            X = self.X_tok[i]
            st = _Site(x_hist=X, x_off=x_off, kc=self.Kpre[i], vc=self.Vpre[i], widx=j,
                       kv_row=None, cos=cos, sin=sin, meta=meta, kpos=None,
                       nout=X[:, P + 1], snout=X.stride(0))
            nt = terms[i + 1].view(B, 1, C) if i + 1 < self.np_ else None
            h = self._block(L, h, st, nt)
        inn = m.input_norm
        e = rmsnorm_rows(h, inn.weight, inn.eps)                       # [B, 1, N, C]
        if self.span_xn is not None:
            self.span_xn[self.arB, j] = e.mean(dim=2)[:, 0]
        # ── coda: every written cell (window by absolute position) + the span's tokens
        base = self.np_ + self.nc
        meta = meta_coda
        xc = (e + terms[base].view(B, 1, 1, C)).contiguous()
        for i, L in enumerate(self.coda):
            X = self.X_tok[self.np_ + i]
            st = _Site(x_hist=X, x_off=x_off, kc=self.Kcoda[i], vc=self.Vcoda[i], widx=cj,
                       kv_row=None, cos=cos, sin=sin, meta=meta, kpos=self.kpos,
                       nout=X[:, P + 1], snout=X.stride(0))
            nt = terms[base + i + 1].view(B, 1, C) if i + 1 < self.nd else None
            xc = self._block(L, xc, st, nt)
        xh = self._readout(xc[:, 0])                                   # [B, C]
        logp = self._logp(xh)
        if self.ptr:
            logp = self._pointer(xh, logp, tok, ti, d[_T_TIM1])
        self.logp_tok.copy_(logp)
        self.next_tok.copy_(logp.argmax(dim=-1))
        ring_commit(self.X_tok, d[_T_JP1])                 # staging row -> row j

    def _pointer(self, xh: Tensor, lm: Tensor, tok: Tensor, ti: Tensor,
                 ti_m1: Tensor) -> Tensor:
        """``TULPointer.mixed_logprobs`` at the token just fed, batched over rows
        (``tul_generate_cached.pointer_keys`` / ``pointer_mix``): the previous entry gets
        its follower (this token); this token's key joins the cache after it is read."""
        p = self.m.tul_pointer
        B = xh.shape[0]
        self.ptr_cand[self.arB, ti_m1] = tok
        q = _ptr_rms(p.q(xh).view(B, p.h, p.dh)).float()
        k = _ptr_rms(p.k(xh).view(B, p.h, p.dh)).float()
        scale = (p.log_scale_bias.exp() / math.sqrt(p.dh)).float()               # [H]
        valid = self.arNk < ti.view(B, 1)                                       # [B, N]
        s = torch.einsum("bhd,bhnd->bhn", q, self.ptr_keys) * scale.view(1, -1, 1)
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        s_null = (q * _ptr_rms(p.null_k_bias).float()).sum(-1) * scale.view(1, -1)
        pr = torch.softmax(torch.cat([s, s_null.unsqueeze(-1)], -1), -1)
        alpha, null = pr[..., :-1], pr[..., -1]
        a = torch.softmax(p.gate_norm_head(xh).float(), -1)                     # [B, 1+H]
        w0 = a[:, 0] + (a[:, 1:] * null).sum(-1)                               # [B]
        beta = (alpha * a[:, 1:, None]).sum(1) * valid.to(alpha.dtype)          # [B, N]
        p_ptr = torch.zeros_like(lm).scatter_add_(1, self.ptr_cand.clamp_min(0), beta)
        self.ptr_keys[self.arB, :, ti] = k
        return torch.log(w0.view(B, 1) * lm.exp() + p_ptr)

    # -- the slot step -------------------------------------------------------------------
    def _slot_step(self, nb: int) -> None:
        m, tc = self.m, self.m.cfg.tul
        M, Kp, P, C, N = self.M, self.Kp, self.P, self.C, self.N
        d, d32 = self.d_slot, self.d_slot32
        q, k = nb * M, nb * Kp
        r, s, n, last = d[_S_R, :nb], d[_S_S, :nb], d[_S_N, :nb], d[_S_LAST, :nb]
        rm, c = d[_S_RM, :q], d[_S_C, :q]                 # cache row, loop index per cell
        bag = d[_S_BAG, :nb * P].view(nb, P)              # 1 past the span's end
        # ── the seed (``TULSlots.slot_input``) and the span's bigram mean ─────────────
        rows_mask = (1 - bag).float().view(nb, 1, P)
        nf = n.float().view(nb, 1, 1)
        e_slot = m.tul._e_slot_term(s.view(nb, 1), torch.float32)
        if tc.slot_seed == "boundary":
            seed = m.tul.W_sent(m.embed(last.view(nb, 1)).float()) + e_slot
        else:
            seed = torch.bmm(rows_mask, self.span_emb.index_select(0, r)) / nf + e_slot
        x_cell = m.embed_drop(seed)[:, 0].float()                              # [nb, C]
        bg_cell = (torch.bmm(rows_mask, self.span_bg.index_select(0, r)) / nf)[:, 0]
        ve_cell = None
        if self.n_ve:
            sv = self.span_ve.index_select(1, r)                              # [n_ve, nb, P, cw]
            ve_cell = torch.einsum("vqpc,qp->vqc", sv, rows_mask[:, 0]) / n.float().view(1, nb, 1)
        terms = self._terms(x_cell, bg_cell, ve_cell)                         # [L, nb, C]
        terms_m = terms.unsqueeze(2).expand(self.n_tot, nb, M, C).contiguous()
        # ── the M cells' prelude: the span's tokens + cells 0..i ───────────────────
        cos, sin = self._rope(d[_S_PPOS, :q])
        x_off, meta = d32[0, :q], d32[1, :q]
        x_hist = self.CX.unsqueeze(0).expand(nb * M, *self.CX.shape)
        h = (x_cell.view(nb, 1, 1, C) + terms_m[0].view(nb, M, 1, C)).expand(
            nb, M, N, C).contiguous()
        for i, L in enumerate(self.pre):
            st = _Site(x_hist=x_hist, x_off=x_off, kc=self.Kpre[i], vc=self.Vpre[i],
                       widx=d[_S_PW, :q], kv_row=rm, cos=cos, sin=sin, meta=meta, kpos=None,
                       nout=self.CX, snout=C)
            nt = terms_m[i + 1] if i + 1 < self.np_ else None
            h = self._block(L, h, st, nt)
        inn = m.input_norm
        e = rmsnorm_rows(h, inn.weight, inn.eps)                               # [nb, M, N, C]
        ctx = e[:, :1].mean(dim=2) if self.lsel else None                      # [nb, 1, C]
        if m.tul_register is not None:
            e = (e + self._register(r, bag, nb).unsqueeze(2).to(e.dtype)).contiguous()
        # ── the loop: depth passes x n_core layers, M cells at s*M + i ────────────────
        cos, sin = self._rope(c)
        x_off, meta = d32[2, :q], d32[3, :q]
        q_stage = self.q_stage_all[:q]
        hs = e
        npc = self.np_
        pick = None
        for t in range(self.depth):
            hs = loop_ssm_term(hs.view(nb * M, 1, N, C), e.view(nb * M, 1, N, C), self.loop_A,
                               self.loop_dt, terms_m[npc].view(nb * M, 1, C), self.loop_lo,
                               self.loop_hi).view(nb, M, N, C)
            for i, L in enumerate(self.core):
                si = t * self.nc + i
                XLs = self.XL[si]
                st = _Site(x_hist=XLs.unsqueeze(0).expand(nb * M, *XLs.shape), x_off=x_off,
                           kc=self.Kloop[si], vc=self.Vloop[si], widx=c, kv_row=rm, cos=cos,
                           sin=sin, meta=meta, kpos=None, nout=XLs[self.L_stage:], snout=C)
                nt = terms_m[npc + i + 1] if i + 1 < self.nc else None
                hs = self._block(L, hs, st, nt)
            if m.tul_cell_norm is not None:
                hs = (rmsnorm_rows(hs, self.cn_w, self.cn_eps) if self.cn_rms
                      else m._cell_norm_apply(hs))
            if self.lsel:
                cr = hs.mean(dim=2).unsqueeze(1)                               # [nb, 1, M, C]
                rt = m.tul_fan_lsel_router
                pick = rt.select(rt.scores(cr, ctx))[0]                        # [nb, 1]
                if t + 1 < self.depth:
                    hs = reset_to_winner(hs, pick, self.ones_q[:nb], M).contiguous()
        # ── the prefix write (``tul_generate_cached.prefix_write``, batched) ────────────
        w = m.tul.W_prefix.to(hs.dtype)
        if M == 1:
            vals = torch.stack([torch.matmul(hs[:, 0], w[k]) for k in range(Kp)], dim=1)
        else:
            hw = hs
            if pick is not None:
                oh = (self.arM.view(1, M) == pick.view(nb, 1)).to(hs.dtype)
                hw = hs * oh.view(nb, M, 1, 1)
            vals = torch.stack([torch.matmul(hw[:, k], w[k]) for k in range(Kp)], dim=1)
        # ── the coda of the Kp cells (each reads itself alone; no injection) ───────────
        rk, ck = d[_S_RK, :k], d[_S_CK, :k]
        self.kpos[rk, ck] = d32[6, :k, 0]
        cos, sin = self._rope(d[_S_PK, :k])
        x_off, meta = d32[4, :k], d32[5, :k]
        x_hist = self.CX.unsqueeze(0).expand(nb * Kp, *self.CX.shape)
        xc = vals.contiguous()
        emit_cell = self.emit_source == "slot"
        for i, L in enumerate(self.coda):
            st = _Site(x_hist=x_hist, x_off=x_off, kc=self.Kcoda[i], vc=self.Vcoda[i],
                       widx=ck, kv_row=rk, cos=cos, sin=sin, meta=meta, kpos=None,
                       nout=self.CX, snout=C)   # self only: d1 = qidx1 - k (XSA drops it)
            last_layer = i + 1 == self.nd
            # Nothing reads the cells' output after the last layer unless they emit.
            out = self._block(L, xc, st, None, front_only=last_layer and not emit_cell)
            if not (last_layer and not emit_cell):
                xc = out
        if emit_cell:
            lp = self._logp(self._readout(xc[:, Kp - 1]))
            self.logp_cell.index_copy_(0, r, lp)
            self.next_cell.index_copy_(0, r, lp.argmax(dim=-1))
        # ── the loop history: this slot's cells into each loop site's ring ─────────────
        self._commit_loop_history(d[_S_DST, :q], q_stage)

    def _commit_loop_history(self, dst: Tensor, src: Tensor) -> None:
        """Every loop site's staging rows ``src`` (this slot's M cells) into its history
        ring at ``dst`` (cache row r, cell ``c % RW``): the next slot's conv reads them."""
        self.XL[:, dst] = self.XL[:, src]

    def _register(self, r: Tensor, bag: Tensor, nb: int) -> Tensor:
        """``[nb, M, C]``: ``TULSlotRegister`` over each bucket row's span rows
        (``tul_generate_cached.register_term`` for ``nb`` rows at once: a dead row past
        the span's end joins bag 1)."""
        xs = self.span_xn.index_select(0, r)                                   # [nb, P, C]
        lay = SlotLayout(slot_mask=self.reg_mask[:nb], bag_id=bag,
                         slot_index=self.reg_index[:nb], slot_valid=self.ones_q[:nb],
                         prefix_k=self.Kp)
        return self.m.tul_register(xs, lay)

    # -- the host side ---------------------------------------------------------------------
    @torch.no_grad()
    def feed(self, toks: list[int], builders: list[TulRowBuilder]) -> list[int]:
        """Feed one token per row (``toks[b]`` into ``builders[b]``), run the token step and,
        for the rows whose token closed a span, the slot step. Returns those rows."""
        B = self.B
        assert len(toks) == B == len(builders)
        tok = np.asarray(toks, dtype=np.int64)
        if (self.span_len >= self.P).any():
            raise RuntimeError(f"span longer than rule.span_cap={self.P}")
        if (self.n_tok >= self.max_tokens).any():
            raise RuntimeError(f"a row exceeds max_tokens={self.max_tokens}")
        pos = np.fromiter((len(bd.ids) for bd in builders), dtype=np.int64, count=B)
        if (pos >= self.max_pos).any():
            raise ValueError(f"row position {int(pos.max())} exceeds the RoPE cache "
                             f"{self.max_pos}")
        cut = [b for b in range(B) if builders[b].append(int(tok[b]))]
        self._ev_tok.synchronize()
        self._host_token(tok, self.prev_id, pos, self.span_len, self.n_tok,
                         self.Kp * self.n_slots)
        self.d_tok.copy_(self.h_tok, non_blocking=True)
        self.d_tok32.copy_(self.h_tok32, non_blocking=True)
        self._ev_tok.record()
        if self.g_tok is not None:
            self.g_tok.replay()
        else:
            self._token_step()
        self.span_len += 1
        self.n_tok += 1
        self.prev_id[:] = tok
        self.last_tok[:] = tok
        self.token_steps += 1
        if cut:
            nb = next(x for x in self.buckets if x >= len(cut))
            self.slot_events[nb] = self.slot_events.get(nb, 0) + 1
            entries = []
            for b in cut:
                if self.n_slots[b] >= self.S:
                    raise RuntimeError(f"more than spec.max_slots={self.S} slots")
                fp = builders[b].slot_first[-1]
                if fp + self.Kp - 1 >= self.max_pos:
                    raise ValueError(f"row position {fp + self.Kp - 1} exceeds the RoPE cache")
                entries.append((b, int(self.n_slots[b]), fp, int(self.span_len[b]),
                                int(self.last_tok[b])))
            self._ev_slot.synchronize()
            self._host_slot(entries, nb)
            self.d_slot.copy_(self.h_slot, non_blocking=True)
            self.d_slot32.copy_(self.h_slot32, non_blocking=True)
            self._ev_slot.record()
            if nb in self.g_slot:
                self.g_slot[nb].replay()
            else:
                self._slot_step(nb)
            for b in cut:
                self.n_slots[b] += 1
                self.span_len[b] = 0
                self.prev_id[b] = self.slot_id
        return cut


@torch.no_grad()
def generate_lx_fused_batch(
    model,
    prompts: list[list[int]],
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int | None = None,
    emit_source: str = "token",
    decoder: LxFusedDecoder | None = None,
    top_p: float = 0.0,
    return_logits: bool = False,
    exact_weights: bool = True,
    attn: str = "kernel",
    kv_row: str = "kernel",
):
    """``generate_tul_graphed`` for a batch of rows: per row the same tokens, the same
    builder and (``return_logits``) the same log-prob rows, up to the kernels' reduction
    order. Returns ``(token_ids per row, builder per row[, rows per row])``.

    Every row feeds one token per step. A row whose prompt is shorter starts generating
    earlier; after its ``max_new_tokens`` it keeps decoding greedily until the longest
    row is done (its outputs are discarded and its returned builder is the one at its
    own end)."""
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("generate_lx_fused_batch needs at least one token per prompt")
    B = len(prompts)
    was_training = model.training
    model.eval()
    own = decoder is None
    if own:
        steps = max(len(p) for p in prompts) + max_new_tokens
        decoder = LxFusedDecoder(model, spec, rule, B, emit_source, exact_weights,
                                 max_tokens=steps, attn=attn, kv_row=kv_row)
    dec = decoder
    if dec.m is not model or dec.spec != spec or dec.rule is not rule or dec.B != B \
            or dec.emit_source != emit_source:
        raise ValueError("decoder was built for a different model / spec / rule / batch / emit")
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(dec.dev)).manual_seed(seed)
    builders = [TulRowBuilder(rule=rule, spec=spec) for _ in range(B)]
    final_b: list[TulRowBuilder | None] = [None] * B
    emitted: list[list[int]] = [[] for _ in range(B)]
    rows: list[list[Tensor]] = [[] for _ in range(B)]
    fed = [0] * B
    total = max(len(p) + max_new_tokens - 1 for p in prompts)
    nxt = [int(p[0]) for p in prompts]
    try:
        dec.reset()
        for step in range(total):
            dec.feed(nxt, builders)
            need = [b for b in range(B) if fed[b] + 1 >= len(prompts[b])]
            for b in range(B):
                fed[b] += 1
            if not need:
                nxt = [int(prompts[b][fed[b]]) for b in range(B)]
                continue
            use_cell = [builders[b].slot_mask[-1] and emit_source == "slot" for b in range(B)]
            if temperature <= 0.0:
                nt = dec.next_tok.tolist()
                nc = dec.next_cell.tolist() if emit_source == "slot" else None
            for b in range(B):
                if fed[b] < len(prompts[b]):
                    nxt[b] = int(prompts[b][fed[b]])
                    continue
                lrow = dec.logp_cell[b] if use_cell[b] else dec.logp_tok[b]
                if temperature <= 0.0:
                    t_b = nc[b] if use_cell[b] else nt[b]
                else:
                    t_b = sample_next(lrow, temperature, top_k, gen, top_p)
                live = len(emitted[b]) < max_new_tokens
                if live:
                    emitted[b].append(int(t_b))
                    if return_logits:
                        rows[b].append(lrow.detach().clone())
                    if len(emitted[b]) == max_new_tokens:
                        fb = copy.deepcopy(builders[b])
                        fb.append(int(t_b))          # the layout of the last token, no forward
                        final_b[b] = fb
                nxt[b] = int(t_b)
        for b in range(B):
            if final_b[b] is None:       # max_new_tokens == 0 or the row ended exactly here
                final_b[b] = builders[b]
    finally:
        if own:
            dec.__exit__(None, None, None)
        if was_training:
            model.train()
    if return_logits:
        return emitted, final_b, rows
    return emitted, final_b
