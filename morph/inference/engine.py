"""Static-buffer + CUDA-graph fast B=1 decode engine for MORPH (fast-decode-engine, #245).

The eager incremental decoder (``morph/model/kv_cache.py`` — the GOLDEN reference, untouched)
is LAUNCH-BOUND at B=1: ~13.5k kernel launches/token, CPU launch time >> GPU compute. Its cache
grows via ``torch.cat`` (dynamic shapes) so it cannot be CUDA-graph captured. This module is the
fix, built in parity-gated phases (PLAN: Ai-notes/06-11-2026/MORPH-Fast-Decode-Engine/):

  Phase 1 — static preallocated buffers, device-tensor position, fixed-shape masked reads.
  Phase 2 — three ``torch.cuda.CUDAGraph``s (no-emit / CSA-emit / CSA+HCA-emit; CSA emits every
            csa_m=4 tokens, HCA every hca_m=128; hca-emit ⇒ csa-emit), replayed per token.
            Ternary/int6 parametrizations are MATERIALIZED once (``leave_parametrized=True``
            bakes exactly the tensor the parametrization computes → bit-identical reads).
  Phase 3a — kernel-count strangling (measured 6452 kernels/tok → the graph replay was
            overhead-bound at ~1.5 µs/kernel):
              * ALL sites' rolling windows live in 4 STACKED tensors (win_k/win_v per [site],
                x-history per CSA/HCA kind) rolled ONCE per step (start-of-step roll + staging
                slot) instead of ~400 per-site roll/copy kernels.
              * x-history buffers UNIFY conv history (last 7 incl. current), v_prev (P-1) and
                the compressed-block input window — one buffer per site instead of three.
              * per-layer stacked projection weights: ONE GEMM gives q_lat/k_lat over the conv
                window + v_prev; ONE GEMV gives v_curr + gate-hidden + (CSA) indexer query.
              * retention: one stacked [5·d, d] GEMV for q/k/v/gate/r projections.
              * per-step shared precompute: window/CSA/HCA validity masks, RoPE cos/sin row,
                ALL 12 layers' injection terms (batched x0 projection + bigram scale).
            All reductions keep the golden's operand ordering where shapes allow; deviations
            are fp reduction-tree order only (gated empirically, see below).

Faithfulness contract (no-theater):
  * GATE: ignore/verify_static_decode.py — greedy generation token_match must be 1.0 vs the
    eager golden decoder (which is itself proven vs O(T²) recompute) before any tok/s number
    from this engine is trusted. A speedup that fails the gate is a bug, not a win.
  * The CSA top-k pool is shape-identical to the golden (it already pads to csa_pool_len//m)
    → identical top-k tie-breaking. Window/HCA padded reads contribute exact zeros.

Usage:
    cache = MORPHKVCache(); cache.csa_pool_len = model.cfg.context_len
    for t in range(prompt_len):                      # prefill via the PROVEN eager path
        logits = decode_step(model, prompt[:, t], cache)
    eng = StaticDecodeEngine(model, batch_size=B)
    eng.load_from_eager(cache)                       # convert state → static buffers
    eng.capture()                                    # optional: CUDA-graph replay
    logits = eng.decode_step(next_ids)               # O(1) static per-token step

Constraints (asserted): kv_quant off; eval mode; conversion position >= 2*csa_m (the first CSA
block must already have been emitted by the eager prefill); position < context_len.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

from morph.kernels.triton.fused_decode_step import (
    bf16_gemv, csa_emit, csa_select, decode_attn, decode_front, gla_step,
    hc_post_wide, hc_premap_wide, int8_gemv, loop_ssm_term, mortar_gemv,
    mortar_pack_strips, pack_nibbles, ring_commit, ring_meta, rmsnorm_rows,
    route_flags, small_gemv, swiglu_rows, ternary_pack, ternary_gemv,
    ternary_gemv_rs)
from morph.kernels.triton.fused_router import fused_router

# Fuse the eager ReMoE router pile (proj+LN+subkeys+topk+relu+normalize) into one Triton
# launch. ON by default; set MORPH_FUSED_ROUTER=0 to fall back to the eager pile (for A/B).
import os as _os
_USE_FUSED_ROUTER = _os.environ.get("MORPH_FUSED_ROUTER", "1") != "0"
from morph.kernels.triton.fused_hyper_connection import (
    _LAUNCH as _HC_LAUNCH, _hc_post_fwd_kernel, _hc_premap_fwd_kernel, _next_pow2)
from morph.inference.kv_cache import MORPHKVCache

_BIGRAM_A: int = 279470273
_BIGRAM_B: int = 4294967291


def _mortar_dense(cms) -> torch.Tensor:
    """Dense fp32 reconstruction of a pack_mortar_ternary'd CMSBlockLinear's effective
    weight (pruned blocks = exact zeros). Transient — used only by the engine's
    mlp_dense_expand A/B arm, immediately ternary_pack'd by the caller."""
    eff = cms._mortar_effective_data().float()             # [nnz, blk, blk]
    nnz, blk, _ = eff.shape
    O, I = cms.out_features, cms.in_features
    W = torch.zeros(O, I, dtype=torch.float32, device=eff.device)
    W4 = W.view(O // blk, blk, I // blk, blk)
    W4[cms.mortar_row_indices.long(), :, cms.mortar_column_indices.long()] = eff
    return W


def materialize_quant(model) -> int:
    """Bake every weight parametrization (ternary QAT / int6 embed) into a plain tensor.

    ``leave_parametrized=True`` stores exactly the tensor the parametrization forward returns,
    so every subsequent read is bit-identical to the parametrized read — but costs zero
    recompute and leaves no per-access module call. Returns #parametrizations removed.
    """
    n = 0
    for mod in list(model.modules()):
        if parametrize.is_parametrized(mod):
            for pname in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, pname, leave_parametrized=True)
                n += 1
    return n


class _Site:
    """Per-call-site state: views into the stacked engine buffers + layer weight stacks."""

    __slots__ = ("key", "block", "impl", "is_csa", "m", "max_blk", "iter_idx",
                 "route",
                 "X", "win_k", "win_v", "C_comp", "K_I", "ret_state",
                 "Wqkv", "Wmisc", "Wret", "Wgate2", "Wgu", "Wdown", "Wup",
                 "Tgu", "Tdown", "exp_temp", "alpha_flat", "ret_gate_sig", "conv_pack",
                 "Woproj_s", "Wemit", "Ba", "Bb", "Bia", "CnW",
                 "Wqkv_sc", "Wup_sc", "Mgu", "Mdown")

    def __init__(self, key, block, iter_idx):
        self.key = key
        self.block = block
        self.impl = block.attention._impl
        self.is_csa = hasattr(self.impl, "indexer")
        self.m = int(self.impl.compress_ratio)
        self.iter_idx = iter_idx


class StaticDecodeEngine:
    """Fixed-shape O(1)-per-token MORPH decoder with optional CUDA-graph replay."""

    def __init__(self, model, batch_size: int = 1, materialize: bool = True,
                 mlp_dense_expand: bool = False, attn_pack4: bool = False):
        """mlp_dense_expand (MORTAR-carved models only): expand the carved 2-bit BCSR
        MLP into a DENSE strip-packed ternary tensor (pruned blocks = code 0) and run
        it through the proven ternary_gemv — the dense arm of the carved-vs-dense
        bandwidth A/B (dense reads 4× the codes; same math, zeros contribute 0).
        attn_pack4: the model was built with int4 (hi=7) attention row-quant —
        nibble-pack the Wqkv/W_up code stacks (lossless; halves attn weight traffic)."""
        assert not model.training, "engine is inference-only — call model.eval() first"
        self.model = model
        cfg = model.cfg
        dev = next(model.parameters()).device
        # The kernel suite's activation contract is fp32 (the 276M eval stack IS fp32,
        # so this is a no-op there). The quantized 30B deploy model carries bf16
        # params/buffers — its engine state/activations still run fp32 (weights stay
        # in their quantized storage; tolerance-gated vs the bf16 eager golden).
        dtype = torch.float32
        self.B = B = batch_size
        self._pack4 = bool(attn_pack4)
        self.n_materialized = materialize_quant(model) if materialize else 0
        # deploy-format detection (packed_ternary_infer): int8/row attention Linears
        # + MORTAR-carved 2-bit MLPs. False on the dense fp32 276M stack.
        from morph.inference.deploy_quant import Int8RowLinear
        self._i8_cls = Int8RowLinear

        with torch.no_grad():
            # LM head: the golden recomputes embed.lm_weight() (lorentz log-map over the full
            # table) EVERY token; weights are frozen → one precompute is bit-identical.
            self.W_lm = model.embed.lm_weight().detach().contiguous()
            # bf16 copy for the LOGITS GEMV only (halves the 151MB/tok head read; x and
            # accumulation stay fp32 — deviation = weight rounding only, re-gated).
            # The fp32 table is still used for the exact input-embedding row reads.
            self.W_lm_bf16 = self.W_lm.to(torch.bfloat16).contiguous()

        self.csa_m = int(cfg.csa_compress_ratio)
        self.hca_m = int(cfg.hca_compress_ratio)
        self.max_pos = int(cfg.context_len)
        self.max_blk_csa = self.max_pos // self.csa_m      # == golden's CSA top-k pool size
        self.max_blk_hca = self.max_pos // self.hca_m
        d = int(cfg.d_model)
        self.rh = int(cfg.retention_heads or cfg.n_heads)
        self.rdh = d // self.rh

        # ── site list in execution order ──────────────────────────────────────
        self.sites: list[_Site] = []
        self.site_by_key: dict[str, _Site] = {}

        def _add(key, block, iter_idx):
            s = _Site(key, block, iter_idx)
            self.sites.append(s)
            self.site_by_key[key] = s

        for i, layer in enumerate(model.prelude):
            _add(f"prelude.{i}", layer, 0)
        for t in range(cfg.mean_depth):
            for i, layer in enumerate(model.core):
                _add(f"core.{i}.{t}", layer, t)
        for i, layer in enumerate(model.coda):
            _add(f"coda.{i}", layer, 0)

        s0 = self.sites[0]
        cca0 = s0.impl.cca
        self.H, self.D = cca0.n_heads, cca0.d_head
        self.Hkv, self.n_rep = cca0.n_kv_heads, cca0.n_rep
        self.LQ = cca0.latent_q_dim                        # H*D
        self.LK = cca0.latent_k_dim                        # Hkv*D
        self.Vh = cca0.W_v_curr.out_features               # value half width
        self.k_conv = cca0.conv_q_dw.kernel_size[0]
        self.hist = 2 * (self.k_conv - 1) + 1              # conv window incl. current (7)
        self.w1 = cca0.window_size - 1                     # window keys excl. self (127)
        self.W_win = self.w1 + 1                           # ring width incl. staging slot

        # ── stacked rolling buffers (rolled ONCE per step) ────────────────────
        # State dtype: fp32 on the 276M eval stack (unchanged); bf16 on the quantized
        # deploy stack — matches the bf16 eager-golden cache values AND halves the
        # per-step roll + attention-read traffic of the (much larger) 30B state.
        # Compute scratch stays fp32 everywhere (kernel partials contract).
        deploy_quant = any(isinstance(m, self._i8_cls) for m in model.modules())
        sdtype = torch.bfloat16 if deploy_quant else dtype
        self.deploy_quant = deploy_quant
        S = len(self.sites)
        z = lambda *sh: torch.zeros(*sh, device=dev, dtype=dtype)
        zs = lambda *sh: torch.zeros(*sh, device=dev, dtype=sdtype)
        self.win_k_all = zs(S, B, self.H, self.W_win, self.D)
        self.win_v_all = zs(S, B, self.H, self.W_win, self.D)
        csa_sites = [s for s in self.sites if s.is_csa]
        hca_sites = [s for s in self.sites if not s.is_csa]
        self.W_xc = 2 * self.csa_m + 1                     # CSA x-history width (2m + staging)
        self.W_xh = self.hca_m + 1                         # HCA x-history width (m + staging)
        assert self.W_xc >= self.hist and self.W_xh >= self.hist
        self.X_csa = zs(len(csa_sites), B, self.W_xc, d)
        self.X_hca = zs(len(hca_sites), B, self.W_xh, d)

        for si, s in enumerate(self.sites):
            s.win_k = self.win_k_all[si]
            s.win_v = self.win_v_all[si]
        for j, s in enumerate(csa_sites):
            s.X = self.X_csa[j]
            s.max_blk = self.max_blk_csa
            s.C_comp = zs(B, self.max_blk_csa, self.D)
            s.K_I = zs(B, self.max_blk_csa, s.impl.indexer.W_IQ.out_features)
        for j, s in enumerate(hca_sites):
            s.X = self.X_hca[j]
            s.max_blk = self.max_blk_hca
            s.C_comp = zs(B, self.max_blk_hca, self.D)
            s.K_I = None
        for s in self.sites:
            s.ret_state = (torch.zeros(B, self.rh, self.rdh, self.rdh, device=dev,
                                       dtype=torch.float32)
                           if s.block.retention is not None else None)

        # ── per-layer stacked projection weights + frozen constants ──────────
        with torch.no_grad():
            stacks: dict[int, tuple] = {}
            expand_packed: dict[int, tuple] = {}           # mlp_dense_expand A/B arm
            for s in self.sites:
                lid = id(s.impl)
                if lid not in stacks:
                    cca = s.impl.cca
                    # ONE stacked projection for the whole site: q/k (conv window rows) +
                    # v_prev + v_curr + gate-hidden (+ CSA indexer q). The misc rows are
                    # only needed at the LAST row, but folding them into the window GEMM
                    # trades negligible extra FLOPs (M=7) for one fewer launch per site.
                    proj_mods = [cca.W_down_q, cca.W_down_k, cca.W_v_prev,
                                 cca.W_v_curr, cca.gate[0]]
                    if s.is_csa:
                        proj_mods.append(s.impl.indexer.W_IQ)
                    q8 = isinstance(cca.W_down_q, self._i8_cls)
                    if q8:
                        # int8 deploy stack: codes + per-row scales, dequant IN-KERNEL
                        # (decode_front HAS_SC) — keeps the dominant attention weight
                        # stream at 1 byte/param instead of bf16's 2. attn_pack4:
                        # nibble-pack (0.5 byte/param, int4-built models only).
                        assert all(isinstance(m, self._i8_cls) for m in proj_mods)
                        wqkv = torch.cat([m.codes for m in proj_mods], dim=0).contiguous()
                        if attn_pack4:
                            wqkv = pack_nibbles(wqkv)
                        wqkv_sc = torch.cat([m.scale.view(-1) for m in proj_mods]) \
                            .float().contiguous()
                    else:
                        wqkv = torch.cat([m.weight for m in proj_mods], dim=0).contiguous()
                        wqkv_sc = None
                    wmisc = None
                    wret, woproj_s = None, None
                    if s.block.retention is not None:
                        r = s.block.retention
                        wret = torch.cat([r.q_proj.weight, r.k_proj.weight, r.v_proj.weight,
                                          r.g_proj.weight, r.r_proj.weight], dim=0).contiguous()
                        # fold the branch gate sigmoid(ret_gate) into the output projection
                        # (linear ⇒ exact up to fp mul order; gated).
                        woproj_s = (torch.sigmoid(s.block.ret_gate)
                                    * r.o_proj.weight).contiguous()
                    # MLP weights: MortarLinear dense mode (276M: plain post-materialize
                    # tensors) OR MORTAR-carved 2-bit BCSR (30B deploy: strip-packed for
                    # mortar_gemv; the codes/scale are exactly pack_mortar_ternary's).
                    mlp_mod = s.block.mlp[0] if hasattr(s.block.mlp, "__getitem__") \
                        else s.block.mlp
                    gu_cms, dn_cms = mlp_mod.gate_up._cms, mlp_mod.down._cms
                    is_mortar = bool(getattr(gu_cms, "_mortar", False))
                    if is_mortar and mlp_dense_expand:
                        # dense A/B arm: scatter the carved blocks into a dense fp32
                        # tensor (zeros elsewhere) and ternary_pack it immediately
                        # (per-layer transient; never holds two dense layers at once).
                        expand_packed[lid] = (
                            ternary_pack(_mortar_dense(gu_cms)),
                            ternary_pack(_mortar_dense(dn_cms)))
                        assert expand_packed[lid][0] is not None, "expand pack failed"
                        mgu = mdown = None
                        wgu = wdown = None
                    elif is_mortar:
                        mgu = mortar_pack_strips(gu_cms)
                        mdown = mortar_pack_strips(dn_cms)
                        wgu = wdown = None
                    else:
                        assert gu_cms._dense_mode, \
                            "engine targets dense or MORTAR-carved MLPs"
                        mgu = mdown = None
                        wgu, wdown = gu_cms.weight, dn_cms.weight
                    conv_pack = (
                        cca.conv_q_dw.weight.reshape(self.LQ, -1).contiguous(),
                        cca.conv_q_gp.weight.contiguous(),
                        cca.conv_k_dw.weight.reshape(self.LK, -1).contiguous(),
                        cca.conv_k_gp.weight.contiguous(),
                        cca.q_norm.weight.contiguous(),
                        cca.k_norm.weight.contiguous(),
                    )
                    bf16 = torch.bfloat16
                    if q8:
                        wup = cca.W_up.codes.contiguous()
                        if attn_pack4:
                            wup = pack_nibbles(wup)
                        wup_sc = cca.W_up.scale.view(-1).float().contiguous()
                    else:
                        wup = cca.W_up.weight.contiguous().to(bf16)
                        wup_sc = None
                    stacks[lid] = (
                        wqkv if q8 else wqkv.to(bf16), wqkv_sc, wmisc,
                        wret.to(bf16) if wret is not None else None,
                        cca.gate[2].weight.contiguous().to(bf16),
                        wgu, wdown, mgu, mdown,
                        torch.exp(cca.temp).detach().float().view(-1).contiguous(),
                        cca.alpha.detach().float().view(-1).contiguous(),
                        (torch.sigmoid(s.block.ret_gate).detach()
                         if s.block.retention is not None else None),
                        conv_pack,
                        woproj_s.to(bf16) if woproj_s is not None else None,
                        wup, wup_sc,
                    )
                (s.Wqkv, s.Wqkv_sc, s.Wmisc, s.Wret, s.Wgate2, s.Wgu, s.Wdown,
                 s.Mgu, s.Mdown, s.exp_temp, s.alpha_flat, s.ret_gate_sig,
                 s.conv_pack, s.Woproj_s, s.Wup, s.Wup_sc) = stacks[lid]

                # ── ReMoE route-then-GATHER (mechanism arm) ───────────────────────
                # A TileRouter attached to a MORTAR-carved MLP routes 16 contiguous
                # d_ff neuron-clusters (8 active / token). The engine evaluates the
                # router (fp32, mirrors TileRouter.forward op-for-op) and feeds the
                # mortar kernels per-token gather flags: gate_up skips inactive
                # OUTPUT block-rows, down skips inactive INPUT block-columns — both
                # EXACT (those h neurons are gated to 0 by the combine epilogue).
                s.route = None
                mlp_mod_r = s.block.mlp[0] if hasattr(s.block.mlp, "__getitem__")                 else s.block.mlp
                rt = getattr(mlp_mod_r, "router", None)
                if rt is not None and s.Mgu is not None:
                    # B>1: each stream routes independently — gates [B, n_clusters],
                    # flags [B, n_blocks] (batch-major), strided per-item in the kernels.
                    rid = (id(rt), s.iter_idx)
                    if not hasattr(self, "_route_cache"):
                        self._route_cache = {}
                    if rid not in self._route_cache:
                        d_ff = mlp_mod_r.d_ff
                        ncls = int(mlp_mod_r.n_clusters)
                        cls = d_ff // ncls
                        assert d_ff % ncls == 0, "engine routed path needs equal clusters"
                        blk = 128
                        # cluster spans of gate_up output block-rows (gate half then
                        # up half — same neuron mapping) and down input block-columns.
                        rh = torch.arange(d_ff // blk, device=dev)
                        lo = (rh * blk) // cls
                        hi = (rh * blk + blk - 1) // cls
                        row_lo = torch.cat([lo, lo]).to(torch.int32).contiguous()
                        row_hi = torch.cat([hi, hi]).to(torch.int32).contiguous()
                        it = min(int(s.iter_idx), rt.n_iters - 1)
                        # keep the ROUTER's dtype (bf16 on the 30B deploy arm — the
                        # module casts its input to proj_dtype, so the engine mirrors
                        # the exact same dtype path → identical gates).
                        self._route_cache[rid] = dict(
                            dtype=rt.query_proj.weight.dtype,
                            wq=rt.query_proj.weight.detach().contiguous(),
                            iter_vec=rt.iter_embed[it].detach().view(1, -1),
                            ln_w=rt.query_norm.weight.detach(),
                            ln_b=rt.query_norm.bias.detach(),
                            ln_eps=float(rt.query_norm.eps),
                            ska_t=rt.sub_keys_a.detach().t().contiguous(),
                            skb_t=rt.sub_keys_b.detach().t().contiguous(),
                            gbias=rt.group_bias.detach().view(1, -1),
                            k=int(rt.activation_k), cls=cls, ncls=ncls,
                            row_lo=row_lo, row_hi=row_hi,
                            col_lo=lo.to(torch.int32).contiguous(),
                            col_hi=hi.to(torch.int32).contiguous(),
                            # fused-router launch config (Z3/microbench-tuned for sm_120).
                            num_warps=int(_os.environ.get("MORPH_ROUTER_WARPS", "1")),
                        )
                    s.route = self._route_cache[rid]
                    if not hasattr(self, "_ract_scr"):
                        # batch-major flag scratch [B, 512]; per-layer NR/NC ≤ 512
                        # (NR = 2·d_ff/128, NC = d_ff/128). Sliced [:, :NR] / [:, :NC]
                        # at the call site keeps stride(0)=512 = the per-stream stride
                        # both route_flags and mortar_gemv index by. Shared across
                        # routed layers (strictly sequential consumption per step).
                        self._ract_scr = torch.zeros(B, 512, device=dev, dtype=torch.int32)
                        self._cact_scr = torch.zeros(B, 512, device=dev, dtype=torch.int32)
                        # fused-router scratch (one-launch ReMoE router): per-stream
                        # gates [B, n_clusters] + a [B, d_model] qv work buffer.
                        self._router_gates_scr = torch.empty(
                            B, int(mlp_mod_r.n_clusters), device=dev, dtype=torch.float32)
                        self._router_qv_scr = torch.empty(
                            B, int(mlp_mod_r.d_model), device=dev, dtype=torch.float32)
        self.G_hidden = cca0.gate[0].out_features          # gate MLP hidden width
        # decode_front layout contract (one launch computes the whole site front-end)
        assert cca0.W_v_prev.out_features == self.Vh, "Wqkv stack: v_prev width != v_curr"
        assert self.G_hidden % self.D == 0, "gate hidden must tile by d_head"
        for s in self.sites:
            if s.is_csa:
                assert s.impl.indexer.W_IQ.out_features == self.D, "q_I width != d_head"
        # per-step scratch (shared across sites — strictly sequential consumption)
        self._q_scr = z(B, self.H, self.D)
        self._xres_scr = z(B, self.LQ)
        self._gateh_scr = z(B, self.G_hidden)
        self._qi_scr = z(B, self.D)
        # K-split lat partials for decode_front kernel A (max site width = CSA 864).
        # KS=4 is the validated 276M fp32 launch (byte-identical); the int8 deploy
        # stack wants KS=8 (lab7c: 5.93 vs 6.73 ms/tok at d=8192, tolerance-gated).
        omax = max(s.Wqkv.shape[0] for s in self.sites)
        self._part_scr = z(B, 8 if deploy_quant else 4, 7, omax)
        self._emit_scr = z(B, 2, 6, 4, 32)                 # CSA-emit GEMV partials

        # fused CSA-emit weight stacks (per layer): cat[aKV,aZ,bKV,bZ,iKV,iZ] + biases
        with torch.no_grad():
            emit_stacks: dict[int, tuple] = {}
            for s in self.sites:
                if not s.is_csa:
                    s.Wemit = s.Ba = s.Bb = s.Bia = s.CnW = None
                    continue
                lid = id(s.impl)
                if lid not in emit_stacks:
                    comp = s.impl.compressor
                    idxc = s.impl.indexer.compressor
                    assert comp.c == 32 and comp.m == self.csa_m and comp.two_stream
                    assert idxc.c == 32 and not idxc.two_stream
                    emit_stacks[lid] = (
                        torch.cat([comp.W_aKV.weight, comp.W_aZ.weight,
                                   comp.W_bKV.weight, comp.W_bZ.weight,
                                   idxc.W_aKV.weight, idxc.W_aZ.weight],
                                  dim=0).float().contiguous(),     # fp32 (no-op at 276M;
                        comp.B_a.detach().float().contiguous(),    # dequants the int8
                        comp.B_b.detach().float().contiguous(),    # deploy weights once)
                        idxc.B_a.detach().float().contiguous(),
                        s.impl.comp_norm.weight.detach().float().contiguous(),
                    )
                (s.Wemit, s.Ba, s.Bb, s.Bia, s.CnW) = emit_stacks[lid]

        # ternary int8 packing for the backbone MLP GEMVs (4× weight traffic at M=1).
        # Exactness-checked per tensor (ternary_pack returns None on any non-ternary weight
        # → that layer falls back to cuBLAS fp32). Packs shared per layer (stacks cache).
        # MORTAR-carved layers (s.Mgu set) use mortar_gemv instead — no dense pack.
        with torch.no_grad():
            packed: dict[int, tuple] = {}
            self.n_ternary_packed = 0
            for s in self.sites:
                lid = id(s.impl)
                if lid not in packed:
                    if lid in expand_packed:
                        packed[lid] = expand_packed[lid]
                    elif s.Wgu is None:
                        packed[lid] = (None, None)
                    else:
                        packed[lid] = (ternary_pack(s.Wgu), ternary_pack(s.Wdown))
                s.Tgu, s.Tdown = packed[lid]
                if s.Tgu is not None:
                    self.n_ternary_packed += 1

        # frozen loop-SSM constants (baseline DiagonalInjection only — asserted).
        # Official reverted the SSM ablation arms (#232): the clean DiagonalInjection
        # has no W_a/B/gate attrs at all, so getattr-default keeps the "no active arm"
        # guard intact without an AttributeError on the baseline injection.
        inj = model.injection
        assert getattr(inj, "W_a", None) is None and getattr(inj, "B", None) is None \
            and getattr(inj, "gate", None) is None, \
            "engine supports the baseline DiagonalInjection/LoopSSM only"
        with torch.no_grad():
            # context-channel slice bounds: ablation LoopSSM exposed lo/hi; official
            # DiagonalInjection (post SSM-revert) exposes start/end. Same slice.
            self._loop_lo = getattr(inj, "lo", getattr(inj, "start", None))
            self._loop_hi = getattr(inj, "hi", getattr(inj, "end", None))
            self._loop_A = inj.log_A.exp().clamp(max=0.9999).detach().contiguous()
            self._loop_dt = inj.log_dt.exp().detach().contiguous()
            # LM-head mixer: softplus channel scales expanded to a [d] vector (frozen).
            mix = model.lm_mixer
            sc = F.softplus(mix.channel_scales)
            self._lm_scale_vec = torch.cat([
                sc[i].expand(w) for i, w in enumerate(mix.channel_dims)]).contiguous()
            w_mix = mix.mix.weight.detach().float().contiguous()
            p_mix = ternary_pack(w_mix)         # deploy stack: PackedTernaryLinear
            if p_mix is not None:
                self._Tmix = p_mix[0]
                self._Tmix_rs = p_mix[1].expand(w_mix.shape[0]).contiguous()
                self._W_mix = None
            else:
                self._Tmix = self._Tmix_rs = None
                self._W_mix = w_mix.to(torch.bfloat16)
            del w_mix

        # injection constants (12 layers): stacked x0 projections + scales, bigram lambdas,
        # value-embed projections (prelude layers 0..n_ve-1).
        # Deploy stacks carry PackedTernaryLinear x0/ve projs (exact γ·{-1,0,+1}
        # per layer): re-pack the dequantized weights to 2-bit codes + per-row γ
        # (ternary_gemv_rs) — 8× less traffic than the bf16 stack (W_x0 at d=8192:
        # 1.92 → 0.24 GB/token). Non-ternary stacks (276M fp32) keep the bf16 path.
        with torch.no_grad():
            n_tot = cfg.n_prelude + cfg.n_core + cfg.n_coda
            self.ctx_s, self.ctx_e = model._ctx_start, model._ctx_end
            ctx_w = self.ctx_e - self.ctx_s
            x0_w = [model.x0_injects[i].proj.weight.detach().float().contiguous()
                    for i in range(n_tot)]
            x0_packs = [ternary_pack(w) for w in x0_w]
            if all(p is not None for p in x0_packs):
                self.Tx0 = torch.cat([p[0] for p in x0_packs]).contiguous()
                self.Tx0_rs = torch.cat([p[1].expand(w.shape[0])
                                         for p, w in zip(x0_packs, x0_w)]).contiguous()
                self.W_x0 = None
            else:
                self.Tx0 = self.Tx0_rs = None
                self.W_x0 = torch.cat(x0_w, dim=0).contiguous().to(torch.bfloat16)
            del x0_w, x0_packs
            self.x0_scales = torch.stack([model.x0_injects[i].log_scale
                                          for i in range(n_tot)]).view(n_tot, 1, 1, 1)
            self.lambdas = model.embed.bigram.lambdas.detach().view(n_tot, 1, 1, 1)
            self.n_ve = len(model._ve_layer_map)
            self.ve_proj_w = []
            for k in range(self.n_ve):
                w = model.value_embeds[k].proj.weight.detach().float().contiguous()
                p = ternary_pack(w)
                if p is not None:
                    self.ve_proj_w.append(("tern", p[0],
                                           p[1].expand(w.shape[0]).contiguous()))
                else:
                    self.ve_proj_w.append(("bf16", w.to(torch.bfloat16), None))
            self.ve_scales = [model.value_embeds[k].log_scale for k in range(self.n_ve)]
            self.ctx_width = ctx_w
            self.n_layers_tot = n_tot

        # ── global state ──────────────────────────────────────────────────────
        # PER-STREAM positions (mixed-length): pos_dev [B] long = each stream's absolute
        # position; self.pos = host int mirror of the MIN over streams (for the host-side
        # capacity guard) — the true per-stream host mirror is self.pos_host [B]. When all
        # streams share a length (B=1 / equal-length) every entry is equal and the engine
        # is byte-identical to the former scalar-pos path.
        self.pos = 0                                       # host mirror (min over streams)
        self.pos_host = [0] * B                            # per-stream host mirror
        self.pos_dev = torch.zeros(B, dtype=torch.long, device=dev)
        self.prev_id = torch.zeros(B, dtype=torch.long, device=dev)
        self.in_ids = torch.zeros(B, dtype=torch.long, device=dev)
        self.next_ids = torch.zeros(B, dtype=torch.long, device=dev)
        self.logits = torch.zeros(B, int(cfg.vocab_size), device=dev, dtype=torch.float32)

        self._ar_csa = torch.arange(self.max_blk_csa, device=dev)
        self._ar_hca = torch.arange(self.max_blk_hca, device=dev)
        # ring-step metadata (filled by ring_meta once per token) — PER-STREAM rows.
        self._win_mask_i8 = torch.zeros(B, self.W_win, device=dev, dtype=torch.int8)
        self._xoff_csa = torch.zeros(B, 8, device=dev, dtype=torch.int32)
        self._xoff_hca = torch.zeros(B, 8, device=dev, dtype=torch.int32)
        self._xoff_emit = torch.zeros(B, 8, device=dev, dtype=torch.int32)
        # per-stream emit masks (1 = stream completes a CSA/HCA block this token).
        # In the collapsed single-graph step EVERY emit-capable site runs every token;
        # these masks gate the per-stream writes (no-op for streams that don't complete).
        self._csa_emit_mask = torch.zeros(B, device=dev, dtype=torch.int32)
        self._hca_emit_mask = torch.zeros(B, device=dev, dtype=torch.int32)
        self._ar127 = torch.arange(self.hca_m - 1, device=dev)
        self._stage_row_hca = torch.full((1,), self.W_xh - 1, device=dev,
                                         dtype=torch.long)
        self._bidx = torch.arange(B, device=dev)[:, None, None]
        self._csa_scratch = torch.empty(B, self.max_blk_csa, device=dev,
                                        dtype=torch.float32)

        # RoPE: all layers share identical cos/sin caches (same constructor constants) —
        # verified at build; one flattened [max_seq, D] pair indexed per step.
        rope0 = cca0.rope
        for s in self.sites:
            r = s.impl.cca.rope
            assert torch.equal(r.cos_cached, rope0.cos_cached), "rope caches differ"
        self._cos_flat = rope0.cos_cached.reshape(-1, self.D).contiguous()
        self._sin_flat = rope0.sin_cached.reshape(-1, self.D).contiguous()

        # bf16 copies of the HC mapping projections (loaded bf16→fp32 in the GEMV)
        with torch.no_grad():
            self._hcw: dict[int, Tensor] = {}
            for s in self.sites:
                for mod in (s.block.mrr_attn, s.block.mrr_mlp):
                    if id(mod) not in self._hcw:
                        self._hcw[id(mod)] = mod.proj.weight.detach().contiguous() \
                            .to(torch.bfloat16)
        # wide-HC kernels load a bias unconditionally — dedicated zeros when absent
        # (model untouched; the 276M small_gemv path keeps its bias=None branch).
        self._hcb_zero = torch.zeros(3 * 4 * 4, device=dev, dtype=torch.float32)

        # per-step shared tensors (filled by _precompute_step)
        self._cos = None
        self._sin = None
        self._win_mask = None
        self._csa_vis = None
        self._hca_vis = None
        self._csa_cnt = None
        self._csa_cnt_m1 = None
        self._hca_cnt = None
        self._terms = None
        # Decode mode. False = equal-length / B=1 fast path (3 emit-variant graphs keyed
        # on the shared host pos — unchanged, zero regression). True = MIXED-LENGTH: a
        # SINGLE collapsed graph that runs every emit site every token and gates writes
        # by per-stream emit masks. Set by load_from_eager from the prefill lengths.
        self._mixed = False
        self.graphs: dict[tuple[bool, bool], torch.cuda.CUDAGraph] | None = None
        self.graph_mixed: torch.cuda.CUDAGraph | None = None

    # ── state conversion from the proven eager prefill ────────────────────────

    @torch.no_grad()
    def load_from_eager(self, cache: MORPHKVCache) -> None:
        """Copy an eager MORPHKVCache into the static RING buffers.

        Ring invariants between steps (engine at position p, about to decode token p):
          * window rings: slot q % W_win holds k/v of position q for the last
            min(p, W_win-1) positions (the slot p % W_win is stale — the validity
            mask excludes it and the step overwrites it with position p).
          * x-history: ring row q % (W-1) holds x_q for q in [p-1-(W-2) .. p-2];
            the fixed staging row W-1 holds x_{p-1} (committed into the ring by
            ring_commit at the start of the step).
        """
        assert cache.kv_quant == "off", "static engine targets kv_quant=off"
        P = int(cache.pos)
        assert P >= 2 * self.csa_m, (
            f"convert at pos>={2 * self.csa_m} (first CSA block must be eager-emitted), got {P}")
        assert P < self.max_pos, f"pos {P} exceeds engine capacity {self.max_pos}"
        self._mixed = False
        self.pos = P
        self.pos_host = [P] * self.B
        self.pos_dev.fill_(P)                       # all B streams at the same P
        self.prev_id.copy_(cache.prev_id)
        dev = self.pos_dev.device
        for s in self.sites:
            sc = cache.sites[s.key]
            # x-history: eager comp_x = last min(P, W-1) tokens (positions P-Lc..P-1).
            W = s.X.shape[1]
            WR = W - 1
            s.X.zero_()
            Lc = sc.comp_x.shape[1]
            assert Lc == min(P, WR), f"{s.key}: comp_x len {Lc} != min(pos,{WR})"
            # consistency: conv history must equal the comp_x suffix
            assert torch.equal(sc.x_recent[:, -min(P, self.hist - 1):],
                               sc.comp_x[:, -min(P, self.hist - 1):]), s.key
            s.X[:, WR:].copy_(sc.comp_x[:, -1:])           # staging ← x_{P-1}
            if Lc > 1:                                     # ring ← x_{P-Lc..P-2}
                q = torch.arange(P - Lc, P - 1, device=dev)
                s.X[:, q % WR] = sc.comp_x[:, :-1].to(s.X.dtype)
            L = sc.win_k.shape[2]
            assert L == min(P, self.w1), f"{s.key}: win len {L} != min(pos,{self.w1})"
            s.win_k.zero_(); s.win_v.zero_()
            qw = torch.arange(P - L, P, device=dev) % self.W_win
            s.win_k[:, :, qw] = sc.win_k.to(s.win_k.dtype)
            s.win_v[:, :, qw] = sc.win_v.to(s.win_v.dtype)
            n = 0 if sc.C_comp is None else sc.C_comp.shape[1]
            assert n == P // s.m, f"{s.key}: blocks {n} != pos//m {P // s.m}"
            s.C_comp.zero_()
            if n:
                s.C_comp[:, :n].copy_(sc.C_comp)
            if s.is_csa:
                s.K_I.zero_()
                if n:
                    s.K_I[:, :n].copy_(sc.K_I)
            if s.ret_state is not None:
                assert sc.ret_state is not None, f"{s.key}: missing eager ret_state"
                s.ret_state.copy_(sc.ret_state)

    @torch.no_grad()
    def load_from_eager_mixed(self, caches: list[MORPHKVCache]) -> None:
        """MIXED-LENGTH conversion: one SOLO eager cache per stream, each prefilled to
        ITS OWN length P_b. Writes each stream into batch row b of the static ring
        buffers at its own absolute position, and switches the engine to the collapsed
        single-graph mixed mode. The eager decode path uses a scalar `cache.pos`, so a
        true mixed-length batch can only be PREFILLED stream-by-stream (B=1 each) — this
        is exactly the per-stream solo prefill, the correct mixed-length ground truth.

        If every P_b is equal this is just the batched equal-length case; we still keep
        mixed mode (one graph) for uniformity, but load_from_eager (the 3-graph fast
        path) remains available and is what the equal-length gate exercises."""
        assert len(caches) == self.B, f"need {self.B} per-stream caches, got {len(caches)}"
        dev = self.pos_dev.device
        Ps = []
        for c in caches:
            assert c.kv_quant == "off", "static engine targets kv_quant=off"
            P = int(c.pos)
            assert P >= 2 * self.csa_m, (
                f"convert at pos>={2 * self.csa_m} (first CSA block must be eager-emitted), "
                f"got {P}")
            assert P < self.max_pos, f"pos {P} exceeds engine capacity {self.max_pos}"
            Ps.append(P)
        self._mixed = True
        self.pos_host = list(Ps)
        self.pos = min(Ps)                         # min over streams (capacity guard base)
        self.pos_dev.copy_(torch.tensor(Ps, dtype=torch.long, device=dev))

        # zero every per-stream buffer ONCE (batched), then fill row b per stream.
        for s in self.sites:
            s.X.zero_(); s.win_k.zero_(); s.win_v.zero_(); s.C_comp.zero_()
            if s.is_csa:
                s.K_I.zero_()

        for b, cache in enumerate(caches):
            P = Ps[b]
            self.prev_id[b].copy_(cache.prev_id.view(-1)[0])
            for s in self.sites:
                sc = cache.sites[s.key]
                W = s.X.shape[1]
                WR = W - 1
                Lc = sc.comp_x.shape[1]
                assert Lc == min(P, WR), f"{s.key}[b={b}]: comp_x len {Lc} != min(P,{WR})"
                assert torch.equal(sc.x_recent[:, -min(P, self.hist - 1):],
                                   sc.comp_x[:, -min(P, self.hist - 1):]), f"{s.key}[b={b}]"
                # staging row ← x_{P-1} (solo cache is B=1 → row 0).
                s.X[b, WR].copy_(sc.comp_x[0, -1])
                if Lc > 1:                          # ring ← x_{P-Lc..P-2}
                    q = torch.arange(P - Lc, P - 1, device=dev)
                    s.X[b, q % WR] = sc.comp_x[0, :-1].to(s.X.dtype)
                L = sc.win_k.shape[2]
                assert L == min(P, self.w1), f"{s.key}[b={b}]: win len {L} != min(P,{self.w1})"
                qw = torch.arange(P - L, P, device=dev) % self.W_win
                s.win_k[b, :, qw] = sc.win_k[0].to(s.win_k.dtype)
                s.win_v[b, :, qw] = sc.win_v[0].to(s.win_v.dtype)
                n = 0 if sc.C_comp is None else sc.C_comp.shape[1]
                assert n == P // s.m, f"{s.key}[b={b}]: blocks {n} != P//m {P // s.m}"
                if n:
                    s.C_comp[b, :n].copy_(sc.C_comp[0])
                if s.is_csa and n:
                    s.K_I[b, :n].copy_(sc.K_I[0])
                if s.ret_state is not None:
                    assert sc.ret_state is not None, f"{s.key}[b={b}]: missing ret_state"
                    s.ret_state[b].copy_(sc.ret_state[0])

    # ── per-step shared precompute ─────────────────────────────────────────────

    def _precompute_step(self, ids: Tensor, x0: Tensor, bigram_emb: Tensor) -> None:
        """Everything shared across sites, ONCE per token: ring metadata + staging
        commits (NO rolls — windows are rings keyed pos%W, x-histories are rings +
        one fixed staging row), validity masks, RoPE row, injection terms."""
        # ring metadata: window validity mask (excludes the slot position p will
        # overwrite) + x-history ring-row indices for positions p-6..p / p-7..p.
        ring_meta(self.pos_dev, self._win_mask_i8, self._xoff_csa, self._xoff_hca,
                  self._xoff_emit, self.W_win, self.W_xc - 1, self.W_xh - 1)
        # commit the previous token's staging row into the x-history rings.
        ring_commit(self.X_csa, self.pos_dev)
        ring_commit(self.X_hca, self.pos_dev)

        self._csa_cnt = self.pos_dev // self.csa_m            # [B]
        self._hca_cnt = self.pos_dev // self.hca_m            # [B]
        # per-stream emit masks: a block completes at this token iff (pos+1)%m == 0.
        # These drive BOTH the collapsed single graph (always-run emit, masked write)
        # and the HCA torch emit path (skip non-completing streams).
        self._csa_emit_mask = ((self.pos_dev + 1) % self.csa_m == 0).to(torch.int32)
        self._hca_emit_mask = ((self.pos_dev + 1) % self.hca_m == 0).to(torch.int32)

        self._cos = self._cos_flat.index_select(0, self.pos_dev)           # [B,D]
        self._sin = self._sin_flat.index_select(0, self.pos_dev)           # [B,D]

        # injection terms for ALL layers: [n_tot, B, 1, d] = lam_i·bigram, ctx slice +=
        # x0 term (+ value-embed terms on the prelude layers). Bit-equal math to the golden
        # _build_injection_term (mul/add of identical operands; batched GEMV rows).
        B = ids.shape[0]
        n_tot, cw = self.n_layers_tot, self.ctx_width
        if self.Tx0 is not None:               # deploy: 2-bit ternary stack (8× less)
            x0_flat = ternary_gemv_rs(x0.view(B, -1).contiguous(), self.Tx0,
                                      self.Tx0_rs)
        else:
            x0_flat = small_gemv(x0.view(B, -1), self.W_x0)
        x0_terms = x0_flat.reshape(B, 1, n_tot, cw).permute(2, 0, 1, 3)
        x0_terms = x0_terms * self.x0_scales.to(x0.dtype)            # [n_tot,B,1,cw]
        terms = self.lambdas.to(x0.dtype) * bigram_emb.unsqueeze(0)  # [n_tot,B,1,d]
        ctx = terms[..., self.ctx_s:self.ctx_e]
        ctx += x0_terms
        for k in range(self.n_ve):
            sig = self.model.value_embed_tables[k](ids)              # [B,1,d]
            kind, w_k, rs_k = self.ve_proj_w[k]
            if kind == "tern":
                pr = ternary_gemv_rs(sig.view(B, -1).contiguous(), w_k, rs_k)
            else:
                pr = small_gemv(sig.view(B, -1), w_k)
            ctx[k] += self.ve_scales[k].to(x0.dtype) * pr.view(B, 1, -1)
        self._terms = terms

    # ── incremental attention (static, fixed-shape) ───────────────────────────

    def _emit(self, impl, s: _Site, mixed: bool) -> None:
        """Emit the compressed block completing at this position (staging slot = current x).

        mixed (per-stream positions): the emit runs UNCONDITIONALLY in the collapsed
        single graph, but only streams that complete a block this token may write.
        CSA gates inside the kernel via _csa_emit_mask; HCA gates the index_copy here.
        """
        m = s.m
        if s.is_csa:
            # fused 2-kernel emit (was ~20 eager kernels: 6 cuBLAS + softmax/pool/norm).
            # X is a ring — per-stream rows of tokens p[b]-7..p[b] resolved in-kernel via
            # _xoff_emit[b]; _csa_emit_mask[b] gates the per-stream C_comp/K_I write.
            csa_emit(s.X, self._xoff_emit, s.Wemit, s.Ba, s.Bb, s.Bia, s.CnW,
                     impl.comp_norm.eps, s.C_comp, s.K_I, self._csa_cnt,
                     self._emit_scr, emit_mask=self._csa_emit_mask if mixed else None)
            return
        # HCA: per-stream ordered gather of the m tokens completing block j[b]=pos[b]//m.
        # Each stream's window is positions p[b]-(m-1)..p[b]; the first m-1 live in the
        # ring (row q%(W-1)), the current token in the fixed staging row W-1. With per-
        # stream positions these ring rows DIFFER across b → a per-stream [B, m] gather.
        WR = self.W_xh - 1
        B = s.X.shape[0]
        # ring rows for positions p[b]-(m-1)..p[b]-1  → [B, m-1]
        ring_rows = (self.pos_dev[:, None] - (m - 1) + self._ar127[None, :]) % WR
        stage = self._stage_row_hca.expand(B, 1)                      # [B,1] = W-1
        rows = torch.cat([ring_rows, stage], dim=1)                   # [B, m]
        d = s.X.shape[-1]
        gathered = torch.gather(
            s.X, 1, rows[:, :, None].expand(B, m, d))                 # [B, m, d]
        blk = impl.comp_norm(impl.compressor(gathered))[:, -1:, :]    # [B, 1, D]
        idx = self._hca_cnt                                           # [B] == pos//m == block j
        if mixed:
            # GRAPH-SAFE masked per-stream scatter (FIXED shape — no nonzero/dynamic):
            # for each stream read its current C_comp[b, idx[b]], blend with the freshly
            # computed block by the per-stream emit mask, and write the blend straight
            # back. Non-completing streams (mask==0) write back the value they already
            # held → an exact no-op; completing streams overwrite with the new block.
            bidx = torch.arange(B, device=s.X.device)
            cur = s.C_comp[bidx, idx]                                 # [B, D]
            mfl = self._hca_emit_mask.view(B, 1).to(s.C_comp.dtype)
            blend = mfl * blk[:, 0].to(s.C_comp.dtype) + (1 - mfl) * cur
            s.C_comp[bidx, idx] = blend
        else:
            # equal-length: every stream completes at the same step → single block idx.
            s.C_comp.index_copy_(1, idx[:1], blk.to(s.C_comp.dtype))

    def _attn_site(self, s: _Site, emit: bool, mixed: bool) -> Tensor:
        """Site attention. The norm_attn output (x_t) is ALREADY in the X staging slot.

        mixed: collapsed single-graph path (per-stream positions). Then the emit runs
        EVERY token at every emit-capable site and the per-stream emit masks decide who
        writes — `emit` is ignored (always emit-then-mask). When mixed=False the legacy
        per-graph `emit` bool selects whether this token's site emits at all (the 3-graph
        equal-length path, byte-identical to the pre-mixed engine)."""
        impl = s.impl
        cca = impl.cca
        B = s.X.shape[0]
        H, D = self.H, self.D
        scale = D ** -0.5

        # ONE-launch front-end: the stacked Wqkv GEMM is computed in-register together
        # with conv+qkmean+rms+temp+rope+v-assembly + window staging writes + gate-hidden
        # (+ CSA indexer query). Replaces cuBLAS GEMM + decode_prologue + gate2 GEMV.
        # X is a ring (+ staging row); rows of tokens P-6..P resolved via _xoff_*.
        decode_front(s.X, self._xoff_csa if s.is_csa else self._xoff_hca,
                     self.pos_dev, s.Wqkv, self._cos, self._sin, *s.conv_pack,
                     s.exp_temp,
                     s.win_k, s.win_v, self._q_scr, self._xres_scr, self._gateh_scr,
                     self._qi_scr if s.is_csa else None, self._part_scr,
                     H, self.Hkv, D, self.LQ, self.LK, self.Vh, self.G_hidden,
                     self.k_conv, cca.q_norm.eps, wqkv_scale=s.Wqkv_sc,
                     wqkv_pack4=self._pack4)

        if s.is_csa:
            # fused score+select: exact torch.topk (value desc, index asc) order over the
            # full padded pool → identical tie-breaking to the golden.
            tk = min(int(impl.top_k), s.max_blk)
            top_idx = csa_select(self._qi_scr, s.K_I, self._csa_cnt,
                                 self._csa_scratch, tk)
            cnt = self._csa_cnt
        else:
            top_idx = None
            cnt = self._hca_cnt

        # silu + gate2 logits are folded INTO decode_attn (w_gate2 path).
        out = decode_attn(self._q_scr, s.win_k, s.win_v, s.C_comp, top_idx,
                          self._gateh_scr, self._xres_scr, cca.sink_logits,
                          s.alpha_flat, cnt, self._win_mask_i8, scale,
                          w_gate2=s.Wgate2, pos_dev=self.pos_dev)
        if mixed:
            self._emit(impl, s, mixed=True)        # always run; masks gate per-stream
        elif emit:
            self._emit(impl, s, mixed=False)

        if s.Wup_sc is not None:
            return int8_gemv(out, s.Wup, s.Wup_sc, pack4=self._pack4).view(B, 1, -1)
        return small_gemv(out, s.Wup).view(B, 1, -1)

    def _retention(self, block, s: _Site, x: Tensor) -> Tensor:
        """GLA branch: stacked qkvgr GEMV → single gla_step kernel (recurrence + GroupNorm
        + swish gate, state updated in place) → gate-folded output projection."""
        r = block.retention
        B = x.shape[0]
        d = self.rh * self.rdh
        xn = rmsnorm_rows(x, block.norm_ret.weight, block.norm_ret.eps)
        p = small_gemv(xn.view(B, d), s.Wret)               # [B,5d]
        o = gla_step(p, s.ret_state, r.gate_bias, r.gn.weight, r.gn.bias, r.gn.eps)
        return small_gemv(o, s.Woproj_s).view(B, 1, d)      # sigmoid(ret_gate) pre-folded

    def _block(self, s: _Site, h: Tensor, emit_csa: bool, emit_hca: bool,
               term_next: Tensor | None = None, mixed: bool = False) -> Tensor:
        block = s.block
        emit = emit_csa if s.is_csa else emit_hca

        def _attn_fn(x: Tensor) -> Tensor:
            # norm_attn(x_bar) is computed INSIDE the premap kernel epilogue and
            # written straight into the X staging slot (HAS_NOUT fold).
            a = self._attn_site(s, emit, mixed)
            if block.retention is not None:
                a = a + self._retention(block, s, x)        # branch gate folded into o_proj
            return a

        def _mlp_fn(x: Tensor) -> Tensor:
            if s.Mgu is not None:
                # MORTAR-carved 2-bit BCSR MLP (30B deploy): 3 kernels — rmsnorm,
                # sparse gate_up (silu(g)·u fused into its combine), sparse down
                # on the plain h vector. Reads only the kept blocks' codes (4× less
                # MLP traffic than dense ternary); swiglu computed ONCE instead of
                # per block visit (bit-identical relocation).
                B_ = x.shape[0]
                xn = rmsnorm_rows(x, block.norm_mlp.weight, block.norm_mlp.eps)
                x2 = xn.reshape(B_, -1)
                if s.route is not None:
                    # ReMoE router (mirrors TileRouter.forward, fp32) → cluster
                    # gates → gather flags → routed mortar GEMVs.
                    R = s.route
                    if _USE_FUSED_ROUTER and B_ == 1:
                        # ONE-launch fused router (proj+LN+subkeys+product-logits+topk+
                        # relu+normalize), Z3-proven for sm_120. Bit-matches the eager pile
                        # in fp32 (deploy router params are fp32). Replaces ~8 tiny launches.
                        # B==1 ONLY: the tail kernel is a single-token CTA. B>1 streams take
                        # the eager mirror below (op-for-op batched; no kernel change).
                        gates = fused_router(
                            x2, R["wq"], R["iter_vec"], R["ln_w"], R["ln_b"], R["ln_eps"],
                            R["ska_t"], R["skb_t"], R["gbias"], R["k"],
                            out=self._router_gates_scr[:1],
                            qv_scr=self._router_qv_scr[0], num_warps=R["num_warps"])
                    else:
                        qv = F.linear(x2.to(R["dtype"]), R["wq"]) + R["iter_vec"]
                        qv = F.layer_norm(qv, (qv.shape[-1],), R["ln_w"], R["ln_b"],
                                          R["ln_eps"])
                        d2 = qv.shape[-1] // 2
                        sa = qv[:, :d2] @ R["ska_t"]
                        sb = qv[:, d2:] @ R["skb_t"]
                        logits = (sa.unsqueeze(2) + sb.unsqueeze(1))                             .reshape(B_, -1) + R["gbias"]
                        kth = logits.topk(R["k"], dim=-1).values[:, -1:]
                        gates = torch.relu(logits - kth)
                        gates = (gates * (R["k"] / gates.sum(-1, keepdim=True)
                                          .clamp(min=1e-6))).float().contiguous()
                    # per-stream gather flags [B, NR] / [B, NC] (batch-major scratch
                    # slices keep stride(0)=512 = the per-item stride the kernels use).
                    NR, NC = R["row_lo"].numel(), R["col_lo"].numel()
                    ract = self._ract_scr[:, :NR]
                    cact = self._cact_scr[:, :NC]
                    route_flags(gates, R["row_lo"], R["row_hi"],
                                R["col_lo"], R["col_hi"], ract, cact)
                    h_ = mortar_gemv(x2, s.Mgu, swiglu_out=True,
                                     row_act=ract, gates=gates,
                                     cluster_size=R["cls"])
                    return mortar_gemv(h_, s.Mdown,
                                       col_act=cact).view(B_, 1, -1)
                h_ = mortar_gemv(x2, s.Mgu, swiglu_out=True)
                return mortar_gemv(h_, s.Mdown).view(B_, 1, -1)
            if s.Tgu is not None and s.Tdown is not None:
                # 2-kernel MLP: norm folded into gate_up, swiglu folded into down.
                gu = ternary_gemv(x, *s.Tgu, rms_weight=block.norm_mlp.weight,
                                  rms_eps=block.norm_mlp.eps)
                return ternary_gemv(gu, *s.Tdown, swiglu_x=True)
            xn = rmsnorm_rows(x, block.norm_mlp.weight, block.norm_mlp.eps)
            gu = F.linear(xn, s.Wgu)
            return F.linear(swiglu_rows(gu), s.Wdown)

        h = self._hc(block.mrr_attn, h, _attn_fn,
                     norm_w=block.norm_attn.weight, norm_eps=block.norm_attn.eps,
                     norm_out=s.X[:, -1], norm_stride=s.X.stride(0))
        # the NEXT block's broadcast inject term folds into this POST write (the post
        # output is fp32, so add-before-store ≡ store-then-add bit-exactly).
        h = self._hc(block.mrr_mlp, h, _mlp_fn, term=term_next)
        return h

    def _hc(self, mod, h: Tensor, fn, term: Tensor | None = None,
            norm_w: Tensor | None = None, norm_eps: float = 1e-6,
            norm_out: Tensor | None = None, norm_stride: int = 0) -> Tensor:
        """Inline HyperConnectionResidual.forward (cayley fused path) with a single-launch
        GEMV for the mapping projection (cuBLAS used a dot+reduce+splitK triple at M=1).
        Same premap/post Triton kernels, same math; forward-only (inference engine).

        Wide carriers (C > 2048, the 30B): the single-CTA premap/post + 48-CTA
        serial-K proj GEMV are pure latency (~21 µs/call at d=8192) — dispatch to
        the multi-CTA hc_premap_wide/hc_post_wide suite instead (same math, fp
        reduction-tree order only; tolerance-gated). 276M path byte-identical."""
        h = h.contiguous()
        B, S, N, C = h.shape
        if C > 2048:
            hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
            hpostrow = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
            hprecm = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
            xbar = hc_premap_wide(
                h, self._hcw[id(mod)],
                mod.proj.bias if mod.proj.bias is not None else self._hcb_zero,
                mod.tau, mod.cayley_alpha, mod.cayley_iters, mod.eps,
                hres, hpostrow, hprecm,
                norm_w=norm_w, norm_eps=norm_eps,
                norm_out=norm_out, norm_stride=norm_stride)
            y = fn(xbar).contiguous()
            return hc_post_wide(hres, hpostrow, h, y, term)
        # mapping projection: row-parallel triton GEMV (cuBLAS gemv2T = 10.8 µs at [48,3072]).
        raw = small_gemv(h.view(B * S, N * C), self._hcw[id(mod)],
                         mod.proj.bias).view(B, S, 3 * N * N)
        xbar = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        hpostrow = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        hprecm = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        rms = torch.empty(B, S, 1, device=h.device, dtype=torch.float32)
        has_nout = norm_out is not None
        # w1/w8 measured-best at C=768 (276M launch unchanged).
        _hc_premap_fwd_kernel[(B * S,)](
            h, raw, xbar, hres, hpostrow, hprecm, rms,
            norm_out if has_nout else xbar, norm_w if has_nout else xbar,
            TAU=float(mod.tau), ALPHA=float(mod.cayley_alpha),
            ITERS=int(mod.cayley_iters), EPS=float(mod.eps),
            N=N, C=C, BLOCK_C=_next_pow2(C),
            HAS_NOUT=has_nout, neps=norm_eps, snout=norm_stride,
            num_stages=1, num_warps=1,
        )
        y = fn(xbar)
        # direct POST kernel launch (same kernel the training wrapper uses; w1 — these
        # single-token kernels are latency-bound and cross-warp sync only hurts).
        out = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        y = y.contiguous()
        _hc_post_fwd_kernel[(B * S,)](
            hres, hpostrow, h, y.view(B, S, C), out,
            term if term is not None else y,
            N=N, C=C, BLOCK_C=_next_pow2(C), HAS_TERM=term is not None,
            num_stages=1, num_warps=8,
        )
        return out

    @torch.no_grad()
    def _step(self, emit_csa: bool, emit_hca: bool, mixed: bool = False) -> None:
        """One full decode step on the static buffers. Mirrors kv_cache.decode_step.

        mixed: collapsed single-graph mixed-length path. emit_csa/emit_hca are ignored
        (every emit-capable site runs every token; the per-stream emit masks computed in
        _precompute_step gate the writes). mixed=False keeps the legacy 3-graph behavior
        where (emit_csa, emit_hca) select which sites emit this token."""
        model = self.model
        cfg = model.cfg
        ids = self.in_ids.view(-1, 1)                       # [B,1]
        B = ids.shape[0]

        # bigram for the (prev, id) pair — identical integer hash to BigramEmbedding.compute.
        hash_ids = (_BIGRAM_A * ids ^ _BIGRAM_B * self.prev_id.view(-1, 1)) \
            % model.embed.bigram.hash_vocab
        bigram_emb = model.embed.bigram.embed(hash_ids)     # [B,1,d]

        # input embedding == W_lm rows (HybridEmbedding.forward ≡ lm_weight()[ids]; the
        # lorentz log-map is elementwise per row — same math the golden computes per call).
        # .float() is a no-op on the fp32 276M head; the 30B head is bf16 storage and
        # the engine activations run fp32.
        x = self.W_lm.index_select(0, ids.view(-1)).view(B, 1, -1).float()
        x0 = x
        self._precompute_step(ids, x0, bigram_emb)
        terms = self._terms

        if model._is_hc:
            x = x.unsqueeze(2).expand(B, 1, model._n_streams, x.shape[-1]).contiguous()

        si = 0
        sites = self.sites
        np_, nc, T = cfg.n_prelude, cfg.n_core, cfg.mean_depth
        # inject terms: terms[0] is the only EAGER add; every other term folds into the
        # preceding MLP hc_post epilogue (bit-exact: fp32 add-before-store) or, for the
        # first core block of each iteration, into the fused LoopSSM kernel.
        x = x + terms[0].unsqueeze(-2)
        for i in range(cfg.n_prelude):
            nt = terms[i + 1] if i + 1 < np_ else None      # input_norm boundary: no fold
            x = self._block(sites[si], x, emit_csa, emit_hca, term_next=nt, mixed=mixed)
            si += 1

        e = rmsnorm_rows(x, model.input_norm.weight, model.input_norm.eps)
        h = e
        lo, hi = self._loop_lo, self._loop_hi
        for t in range(T):
            # LoopSSM (baseline arms; A/dt frozen): h_ctx ← A·h_ctx + dt·e_ctx,
            # fused with the first core block's inject term (1 kernel, was ~5).
            h = loop_ssm_term(h, e, self._loop_A, self._loop_dt, terms[np_], lo, hi)
            for i in range(nc):
                if i + 1 < nc:
                    nt = terms[np_ + i + 1]
                elif t == T - 1:
                    nt = terms[np_ + nc]                    # coda block 0's term
                else:
                    nt = None                               # next op is the LoopSSM fold
                h = self._block(sites[si], h, emit_csa, emit_hca, term_next=nt, mixed=mixed)
                si += 1
        x = h

        for i in range(cfg.n_coda):
            # term for coda block 0 was folded into the last core block's post.
            nt = terms[np_ + nc + i + 1] if i + 1 < cfg.n_coda else None
            x = self._block(sites[si], x, emit_csa, emit_hca, term_next=nt, mixed=mixed)
            si += 1

        if model._is_hc:
            x = x.mean(dim=2)
        xs_ = (x * self._lm_scale_vec)[:, 0]                 # lm_mixer (frozen scales)
        if self._Tmix is not None:
            x = ternary_gemv_rs(xs_, self._Tmix, self._Tmix_rs).view(-1, 1, x.shape[-1])
        else:
            x = small_gemv(xs_, self._W_mix).view(-1, 1, x.shape[-1])
        x = rmsnorm_rows(x, model.final_norm.weight, model.final_norm.eps)
        self.logits.copy_(bf16_gemv(x[:, 0], self.W_lm_bf16))

        # advance global state (device side; host mirror advances in decode_step)
        self.prev_id.copy_(ids[:, 0])
        self.next_ids.copy_(self.logits.argmax(-1))
        self.in_ids.copy_(self.next_ids)                    # greedy self-feed for graph replay
        self.pos_dev.add_(1)

    # ── public API ────────────────────────────────────────────────────────────

    def _emit_flags(self) -> tuple[bool, bool]:
        return ((self.pos + 1) % self.csa_m == 0, (self.pos + 1) % self.hca_m == 0)

    @torch.no_grad()
    def decode_step(self, token_ids: Tensor | None = None) -> Tensor:
        """Decode one token. token_ids [B] (None → self-feed previous greedy argmax).

        Returns the [B, vocab] logits STATIC buffer — consume/clone before the next step.

        Equal-length mode replays one of the 3 emit-variant graphs keyed on the shared
        host pos. Mixed-length mode replays the SINGLE collapsed graph (or eager-steps
        with mixed=True); the per-stream emit masks (built inside _step's precompute from
        pos_dev) decide which streams write their CSA/HCA blocks this token.
        """
        # capacity guard against the FARTHEST-ahead stream (per-stream host mirror).
        assert max(self.pos_host) < self.max_pos - 1, \
            "engine capacity exceeded (context_len)"
        if token_ids is not None:
            self.in_ids.copy_(token_ids.view(-1))
        if self._mixed:
            if self.graph_mixed is not None:
                self.graph_mixed.replay()
            else:
                self._step(False, False, mixed=True)
        else:
            flags = self._emit_flags()
            if self.graphs is not None:
                self.graphs[flags].replay()
            else:
                self._step(*flags)
        self.pos += 1
        self.pos_host = [p + 1 for p in self.pos_host]
        return self.logits

    # ── Phase 2: CUDA-graph capture ───────────────────────────────────────────

    def _state_tensors(self) -> list[Tensor]:
        ts = [self.pos_dev, self.prev_id, self.in_ids, self.next_ids, self.logits,
              self.win_k_all, self.win_v_all, self.X_csa, self.X_hca]
        for s in self.sites:
            ts.append(s.C_comp)
            if s.K_I is not None:
                ts.append(s.K_I)
            if s.ret_state is not None:
                ts.append(s.ret_state)
        return ts

    @torch.no_grad()
    def capture(self, warmup: int = 3) -> None:
        """Capture the decode graph(s). State is snapshotted around the warmup runs
        (warmup EXECUTES and mutates buffers; capture itself records without executing).

        Equal-length mode: the three emit-variant graphs (no-emit / CSA-emit / CSA+HCA).
        Mixed-length mode: a SINGLE collapsed graph (_step(...,mixed=True)) — the emit
        path runs every token and per-stream masks gate the writes, so one graph is valid
        regardless of which streams complete a block on any given token."""
        dev_state = self._state_tensors()
        snap = [t.clone() for t in dev_state]
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if self._mixed:
                    for _ in range(warmup):
                        self._step(False, False, mixed=True)
                else:
                    for _ in range(warmup):
                        self._step(False, False)
                        self._step(True, False)
                        self._step(True, True)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        finally:
            for t, sv in zip(dev_state, snap):
                t.copy_(sv)
        torch.cuda.synchronize()

        pool = torch.cuda.graph_pool_handle()
        if self._mixed:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._step(False, False, mixed=True)
            self.graph_mixed = g
        else:
            graphs: dict[tuple[bool, bool], torch.cuda.CUDAGraph] = {}
            for flags in ((False, False), (True, False), (True, True)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self._step(*flags)
                graphs[flags] = g
            self.graphs = graphs
        # capture records without executing, but restore defensively anyway.
        for t, sv in zip(dev_state, snap):
            t.copy_(sv)
        torch.cuda.synchronize()
