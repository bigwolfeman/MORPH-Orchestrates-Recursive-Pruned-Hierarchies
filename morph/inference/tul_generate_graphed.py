"""CUDA-graphed KV-cached generation for the STRICT TUL slot-loop model.

``tul_generate_cached.generate_tul_cached`` is exact but launch-bound: at batch 1 a
generated token costs ~900 small kernel launches and the GPU sits idle for ~75 % of the
step (profile, 2026-09-26, fp01@10k). This module runs the SAME math from fixed-capacity
buffers so that each of the two decode events is one CUDA graph replay:

* the TOKEN step: embed + bigram, the span-local prelude, the K-rollout coda, the head and
  the per-span Bayes mixture — one graph;
* the SLOT step: the seed, the first cell's prelude, every pass and layer of the slot loop
  for the new slot, the prefix write and the coda of its cells, then the reset of every
  span-local cache — one graph.

What changes against the growing cache is only WHERE the state lives: every call-site
holds K/V for its full capacity plus a position and a validity row, and the masks read
those rows on the device, so shapes never change and nothing syncs with the host inside a
step. Capacities: a span holds at most ``rule.span_cap`` tokens (the boundary rule forces a
cut there, the loader's own cap), a row at most ``spec.max_slots`` slots (the builder
raises past it). A key outside its row's relation gets ``-inf`` / weight 0, and every
buffer is zero-initialised, so a stale entry contributes exactly 0.

The relation, the per-position modules and the read are :mod:`tul_generate_cached`'s,
shared through its helpers (``_check_supported``, ``_cca_rows``, ``_branches``,
``_block``); see that module's docstring for why the strict geometry makes it exact.

``use_graphs=False`` runs the same step functions eagerly (the CPU test path).
"""

from __future__ import annotations

import contextlib
from functools import partial

import torch
from torch import Tensor
from torch.nn.utils import parametrize

from morph.inference.sampling import sample_next
from morph.inference.tul_generate import TulRowBuilder
from morph.inference.tul_generate_cached import (_block, _branches, _cca_rows,
                                                 _check_supported, loop_cell_count,
                                                 loop_relation, lsel_step, pointer_keys,
                                                 pointer_mix, prefix_write, register_term)
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec

__all__ = ["TulGraphDecoder", "generate_tul_graphed"]


class _Site:
    """Fixed-capacity state of ONE attention call-site.

    ``k``/``v`` ``[B, H, N, D]`` post-prologue keys/values, ``pos`` ``[N]`` the position of
    each entry, ``valid`` ``[N]`` which entries are live; ``lat`` ``[B, 2(k-1), dq+dk]`` and
    ``vprev`` ``[B, 1, vh]`` the conv / value-shift history of the current segment (zeros
    at a segment start, which is what the segment cut contributes)."""

    def __init__(self, impl, B: int, N: int, dev, dtype):
        cca = impl.cca
        self.impl = impl
        H, D = cca.n_heads, cca.d_head
        self.need = 2 * (cca.conv_q_dw.kernel_size[0] - 1)
        dim = cca.latent_q_dim + cca.latent_k_dim
        vh = cca.W_v_prev.weight.shape[0]
        self.k = torch.zeros(B, H, N, D, device=dev, dtype=dtype)
        self.v = torch.zeros(B, H, N, D, device=dev, dtype=dtype)
        self.pos = torch.zeros(N, dtype=torch.long, device=dev)
        self.valid = torch.zeros(N, dtype=torch.bool, device=dev)
        self.lat = torch.zeros(B, self.need, dim, device=dev, dtype=dtype)
        self.vprev = torch.zeros(B, 1, vh, device=dev, dtype=dtype)
        # A segment start that must NOT touch the running history (a cell's own segment).
        self.zlat = torch.zeros_like(self.lat)
        self.zvprev = torch.zeros_like(self.vprev)

    def write(self, idx: Tensor, k: Tensor, v: Tensor, pos: Tensor) -> None:
        self.k.index_copy_(2, idx, k)
        self.v.index_copy_(2, idx, v)
        self.pos.index_copy_(0, idx, pos)
        self.valid.index_fill_(0, idx, True)

    def clear(self, start: int = 0) -> None:
        self.valid[start:].zero_()
        self.lat.zero_()
        self.vprev.zero_()


def _site_rows(xa: Tensor, *, site: _Site, pos: Tensor, widx: Tensor, seg_mode: bool,
               slot_range: tuple[int, int], fresh: bool = False, cells: int = 0) -> Tensor:
    """One attention application of the new row(s) ``xa`` at ``pos``: project against the
    site's history (or a fresh segment), write K/V at ``widx``, attend over every live
    entry at or before ``pos`` (window: within ``window_size``, never itself; slot branch:
    the static column range ``slot_range``), gate + up. ``cells > 0``: the slot loop's
    cell relation (``loop_relation``) in place of ``at or before``."""
    impl = site.impl
    lat, vp = (site.zlat, site.zvprev) if fresh else (site.lat, site.vprev)
    q, k, v, q_lat, gate_pre, win, v_prev_raw = _cca_rows(impl, xa, pos, lat, vp, seg_mode)
    if not fresh:
        site.lat.copy_(win[:, win.shape[1] - site.need:])
        site.vprev.copy_(v_prev_raw[:, -1:])
    site.write(widx, k, v, pos)
    dist = pos.view(-1, 1) - site.pos.view(1, -1)
    rel = loop_relation(pos, site.pos, cells) if cells else dist >= 0
    allow = site.valid.view(1, -1) & rel
    win_mask = allow & (dist.abs() < impl.cca.window_size) & (dist != 0)
    a, b = slot_range
    out_win, out_comp = _branches(impl, q, site.k, site.v, win_mask, site.k[:, :, a:b],
                                  site.v[:, :, a:b], allow[:, a:b])
    return impl.cca._gate_combine_up(xa, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)


def _cell_rows(xa: Tensor, *, site: _Site, pos: Tensor, widx: Tensor) -> Tensor:
    """The coda of a slot's prefix cells: a fresh segment (the cells' own conv / value
    shift), each cell reading ITSELF alone (strict ``self_only``), then their K/V join the
    site's permanent cell entries at ``widx``."""
    impl = site.impl
    q, k, v, q_lat, gate_pre, _win, _vp = _cca_rows(impl, xa, pos, site.zlat, site.zvprev,
                                                    True)
    dist = pos.view(-1, 1) - pos.view(1, -1)
    allow = dist == 0
    win_mask = allow & (dist != 0)
    out_win, out_comp = _branches(impl, q, k, v, win_mask, k, v, allow)
    site.write(widx, k, v, pos)
    return impl.cca._gate_combine_up(xa, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)


class TulGraphDecoder(contextlib.AbstractContextManager):
    """Static buffers + the two step functions (+ their CUDA graphs) for ONE model.

    A context manager: it holds ``parametrize.cached()`` open for its whole life, because
    the captured graphs read the cached QAT weights (ternary MLPs, int6 embedding) at
    fixed addresses. Reusable across generations (``reset`` between rows)."""

    def __init__(self, model, spec: TulLayoutSpec, rule: BoundaryRule, emit_source: str,
                 use_graphs: bool = True, cells: int | None = None):
        if emit_source not in ("slot", "token"):
            raise ValueError(f"emit_source must be 'slot' or 'token', got {emit_source!r}")
        _check_supported(model)
        m, cfg, tc = model, model.cfg, model.cfg.tul
        if spec.prefix_k != tc.prefix_k:
            raise ValueError(f"spec prefix_k {spec.prefix_k} != model {tc.prefix_k}")
        self.m, self.spec, self.rule = m, spec, rule
        self.emit_source = emit_source
        self.dev = next(m.parameters()).device
        self.dtype = next(m.parameters()).dtype
        self.use_graphs = bool(use_graphs)
        if self.use_graphs and self.dev.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA model; pass use_graphs=False on CPU")
        self.K = int(m._code_enum_k) if m.tul_code_enum is not None else 1
        self.M = loop_cell_count(m, cells)            # looped cells per slot
        self.lsel = m._lsel_mode != "off"
        self.ptr = m.tul_pointer is not None
        self.depth = (int(tc.slot_depth_fixed) if int(tc.slot_depth_fixed) > 0
                      else int(tc.slot_mean_depth or cfg.mean_depth))
        self.Kp = int(tc.prefix_k)
        self.P = int(rule.span_cap)                   # tokens of one span, at most
        self.S = int(spec.max_slots)                  # slots of one row, at most
        self.Cc = self.Kp * self.S                    # coda cell entries
        self.slot_id = int(tc.slot_id)
        self.max_pos = int(m.prelude[0].attention._impl.cca.rope.cos_cached.shape[2])
        self.n_ve = len(m._ve_layer_map)
        self._ctx = contextlib.ExitStack()
        self._ctx.enter_context(parametrize.cached())
        try:
            self._build()
        except BaseException:
            self._ctx.close()
            raise

    # -- construction -------------------------------------------------------------
    def _build(self) -> None:
        m, dev, dt = self.m, self.dev, self.dtype
        C = m.cfg.d_model
        self.w_head = m.embed.lm_weight()
        self.slot_col = torch.tensor([self.slot_id], device=dev)
        Mc = self.M
        # Prelude: P token entries + the M looped cells (indices P ..) for their own read.
        self.pre = [_Site(b.attention._impl, 1, self.P + Mc, dev, dt) for b in m.prelude]
        # Loop: one site per (pass, layer), every cell of every slot of the row (s*M + i).
        self.core = [[_Site(b.attention._impl, self.K, self.S * Mc, dev, dt) for b in m.core]
                     for _ in range(self.depth)]
        # Coda: Cc cell entries (permanent) then P token entries (span-local).
        self.coda = [_Site(b.attention._impl, self.K, self.Cc + self.P, dev, dt)
                     for b in m.coda]
        # Span-local rows the seed reads (embedding, bigram, value embeds).
        self.span_emb = torch.zeros(1, self.P, C, device=dev, dtype=dt)
        self.span_bg = torch.zeros(1, self.P, C, device=dev, dtype=dt)
        self.span_ve = [torch.zeros(1, self.P, m.value_embeds[k].precompute(
            m.value_embed_tables[k](torch.zeros(1, 1, dtype=torch.long, device=dev))
        ).shape[-1], device=dev, dtype=dt) for k in range(self.n_ve)]
        # The register pools the span's token rows (prelude output, stream mean).
        self.span_xn = (torch.zeros(1, self.P, C, device=dev, dtype=dt)
                        if m.tul_register is not None else None)
        V = self.w_head.shape[0]
        # The pointer: every token of the row (capacity: the RoPE cache), its key and the
        # token that followed it; an entry turns valid when its follower is fed.
        if self.ptr:
            ph, pd = m.tul_pointer.h, m.tul_pointer.dh
            self.ptr_keys = torch.zeros(ph, self.max_pos, pd, device=dev, dtype=torch.float32)
            self.ptr_cand = torch.zeros(self.max_pos, dtype=torch.long, device=dev)
            self.ptr_valid = torch.zeros(self.max_pos, dtype=torch.bool, device=dev)
        self.C = torch.zeros(self.K, dtype=torch.float64, device=dev)
        self.last_logp = torch.zeros(self.K, V, dtype=torch.float32, device=dev)
        self.logits_out = torch.zeros(V, dtype=torch.float32, device=dev)
        self.cell_logits_out = torch.zeros(V, dtype=torch.float32, device=dev)
        self.arP = torch.arange(self.P, device=dev)
        self.arKp = torch.arange(self.Kp, device=dev)
        self.arM = torch.arange(Mc, device=dev)
        self.idxP = self.P + self.arM
        # Host -> device step inputs, one copy per step.
        pin = self.dev.type == "cuda"
        self.h_tok = torch.zeros(6, dtype=torch.long, pin_memory=pin)
        self.d_tok = torch.zeros(6, dtype=torch.long, device=dev)
        self.h_slot = torch.zeros(3, dtype=torch.long, pin_memory=pin)
        self.d_slot = torch.zeros(3, dtype=torch.long, device=dev)
        # The async copy out of a pinned buffer reads it when the stream gets there, so a
        # buffer is rewritten only after its previous copy has run (prompt tokens are fed
        # back to back with no host sync in between).
        self._ev_tok = torch.cuda.Event() if pin else None
        self._ev_slot = torch.cuda.Event() if pin else None
        self.g_tok = self.g_slot = None
        if self.use_graphs:
            self._capture()
        self.reset()

    def _capture(self) -> None:
        # Warm up on a side stream (compiles / autotunes every Triton kernel at these
        # shapes and settles the allocator), then capture each step once.
        self.h_tok.copy_(torch.tensor([5, 0, 0, 0, 0, 0]))
        self.d_tok.copy_(self.h_tok)
        self.h_slot.copy_(torch.tensor([0, 1, 1]))
        self.d_slot.copy_(self.h_slot)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._token_step()
                self._slot_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.g_tok = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_tok):
            self._token_step()
        self.g_slot = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_slot, pool=self.g_tok.pool()):
            self._slot_step()
        torch.cuda.synchronize()

    def reset(self) -> None:
        """Empty every cache: a new row."""
        for site in self.pre + self.coda + [s for row in self.core for s in row]:
            site.clear()
        if self.ptr:
            self.ptr_valid.zero_()
        self.C.zero_()
        self.last_logp.zero_()
        self.n_tok = 0
        self.span_len = 0
        self.n_slots = 0
        self.prev_id = 0

    def __exit__(self, *exc) -> None:
        self.g_tok = self.g_slot = None
        self._ctx.close()

    # -- the two steps (device-only: no host sync, fixed shapes) -------------------
    def _rep(self, t: Tensor) -> Tensor:
        return t.repeat(self.K, *([1] * (t.dim() - 1))) if self.K > 1 else t

    def _logp(self, xh: Tensor) -> Tensor:
        logits = (xh[:, -1] @ self.w_head.T).index_fill(-1, self.slot_col, float("-inf"))
        return torch.log_softmax(logits.float(), dim=-1)

    def _mix(self, logp: Tensor, C: Tensor) -> Tensor:
        logw = torch.log_softmax(C, dim=0).float()
        acc = None
        for k in range(self.K):
            term = logp[k].float() + logw[k]
            acc = term if acc is None else torch.logaddexp(acc, term)
        return acc

    def _token_step(self) -> None:
        m = self.m
        d = self.d_tok
        tok, prev = d[0:1].view(1, 1), d[1:2].view(1, 1)
        p, sidx, hasp, tidx = d[2:3], d[3:4], d[4:5] != 0, d[5:6]
        emb = m.embed(tok)
        x = m.embed_drop(emb)
        bg = m.embed.get_bigram(torch.cat([prev, tok], dim=1))[:, 1:]
        ve = [m.value_embeds[k].precompute(m.value_embed_tables[k](tok))
              for k in range(self.n_ve)]
        self.span_emb.index_copy_(1, sidx, emb)
        self.span_bg.index_copy_(1, sidx, bg)
        for k in range(self.n_ve):
            self.span_ve[k].index_copy_(1, sidx, ve[k])
        h = x.unsqueeze(2).expand(1, 1, m._n_streams, x.shape[-1]).contiguous()
        for i, blk in enumerate(m.prelude):
            term = m._build_injection_term(i, m.x0_injects[i].precompute(x), None, bg,
                                           h.dtype, ve_bagged=ve if self.n_ve else None)
            h = m._apply_injection(h, term)
            h = _block(blk, h, partial(_site_rows, site=self.pre[i], pos=p, widx=sidx,
                                       seg_mode=True, slot_range=(self.P, self.P + 1)), 0)
        xn = m.input_norm(h)
        if self.span_xn is not None:
            self.span_xn.index_copy_(1, sidx, xn.mean(dim=2))
        xc = self._rep(xn)
        x0K, bgK = self._rep(x), self._rep(bg)
        cidx = sidx + self.Cc
        base_i = m.cfg.n_prelude + m.cfg.n_core
        for i, blk in enumerate(m.coda):
            term = m._build_injection_term(base_i + i, m.x0_injects[base_i + i].precompute(
                x0K), None, bgK, xc.dtype)
            term = term * torch.ones((1, 1, 1), dtype=term.dtype, device=term.device)
            xc = m._apply_injection(xc, term)
            xc = _block(blk, xc, partial(_site_rows, site=self.coda[i], pos=p, widx=cidx,
                                         seg_mode=True, slot_range=(0, self.Cc)), 0)
        xh = m._readout(xc)
        logp = self._logp(xh)
        if self.K > 1:
            ev = self.last_logp.gather(1, tok.expand(self.K, 1)).squeeze(1).double()
            Cn = self.C + torch.where(hasp, ev, torch.zeros_like(ev))
            self.logits_out.copy_(self._mix(logp, Cn))
            self.C.copy_(Cn)
            self.last_logp.copy_(logp)
        elif self.ptr:
            # The previous token's entry gets its follower (this token) and turns valid;
            # this token's key is written at its own index, valid from the next token on.
            j = (tidx - 1).clamp_min(0)
            self.ptr_cand.index_copy_(0, j, tok.view(1))
            self.ptr_valid.index_copy_(0, j, (tidx > 0).view(1))
            q, k = pointer_keys(m, xh[:, -1])
            self.logits_out.copy_(pointer_mix(m, xh[:, -1], logp[0], q, self.ptr_keys,
                                              self.ptr_cand, self.ptr_valid))
            self.ptr_keys.index_copy_(1, tidx, k.unsqueeze(1))
        else:
            self.logits_out.copy_(logp[0])

    def _slot_step(self) -> None:
        m, tc = self.m, self.m.cfg.tul
        d = self.d_slot
        s, fp, n = d[0:1], d[1:2], d[2:3]
        dt = self.span_emb.dtype
        rows = (self.arP < n).to(dt).view(1, 1, self.P)
        nf = n.to(dt)
        if tc.slot_seed == "boundary":
            last = self.span_emb.index_select(1, n - 1)
            seed = m.tul.W_sent(last) + m.tul._e_slot_term(s.view(1, 1), dt)
        else:
            seed = torch.bmm(rows, self.span_emb) / nf + m.tul._e_slot_term(s.view(1, 1), dt)
        x_cell = m.embed_drop(seed)
        bg_cell = torch.bmm(rows, self.span_bg) / nf
        ve_cell = [torch.bmm(rows, self.span_ve[k]) / nf for k in range(self.n_ve)]
        # ── the looped cells' prelude: each reads its span's tokens, its earlier
        #    siblings and itself ─────────────────────────────────────────────────────
        Mc = self.M
        x_m, bg_m = x_cell.expand(1, Mc, -1), bg_cell.expand(1, Mc, -1)
        ve_m = [v.expand(1, Mc, -1) for v in ve_cell]
        h = x_m.unsqueeze(2).expand(1, Mc, m._n_streams, x_m.shape[-1]).contiguous()
        for i, blk in enumerate(m.prelude):
            term = m._build_injection_term(i, m.x0_injects[i].precompute(x_m), None,
                                           bg_m, h.dtype,
                                           ve_bagged=ve_m if self.n_ve else None)
            h = m._apply_injection(h, term)
            h = _block(blk, h, partial(_site_rows, site=self.pre[i], pos=fp + self.arM,
                                       widx=self.idxP, seg_mode=True,
                                       slot_range=(self.P, self.P + Mc), fresh=True), 0)
        e = m.input_norm(h)                                              # [1, M, n, C]
        ctx = e[:, :1].mean(dim=2) if self.lsel else None
        if self.span_xn is not None:
            e = e + register_term(m, self.span_xn, self.arP < n).unsqueeze(2).to(e.dtype)
        eK = self._rep(e)
        # ── the slot loop for slot s (K rollouts, M cells at s*M + i) ──────────────
        x0K, bgK = self._rep(x_m), self._rep(bg_m)
        hs = m.core_init(eK)
        np_ = m.cfg.n_prelude
        inj = torch.stack(
            [m._build_injection_term(np_ + i, m.x0_injects[np_ + i].precompute(x0K), None,
                                     bgK, hs.dtype) for i in range(m.cfg.n_core)], dim=0)
        valid = torch.ones(self.K, 1, dtype=torch.bool, device=self.dev)
        cpos = s * Mc + self.arM
        pick = None
        for t in range(self.depth):
            hi = m.injection(hs, eK)
            for i, blk in enumerate(m.core):
                hi = m._apply_injection(hi, inj[i])
                hi = _block(blk, hi, partial(_site_rows, site=self.core[t][i], pos=cpos,
                                             widx=cpos, seg_mode=False,
                                             slot_range=(0, self.S * Mc), cells=Mc), t)
            hs = hi
            if self.K > 1:
                hs = m._apply_injection(hi, m.tul_code_enum.term(hi, valid, self.K))
            if m.tul_cell_norm is not None:
                hs = m._cell_norm_apply(hs)                     # as tul_generate_cached
            if self.lsel:
                hs, pick = lsel_step(m, hs, ctx, Mc, reset=t + 1 < self.depth)
        # ── the prefix write and the coda of the cells ─────────────────────────────
        xc = prefix_write(m, hs, pick, self.Kp).to(eK.dtype)
        pc = fp + self.arKp
        widx = s * self.Kp + self.arKp
        x0c = self._rep(x_cell.expand(1, self.Kp, -1))
        bgc = self._rep(bg_cell.expand(1, self.Kp, -1))
        base_i = m.cfg.n_prelude + m.cfg.n_core
        for i, blk in enumerate(m.coda):
            term = m._build_injection_term(base_i + i, m.x0_injects[base_i + i].precompute(
                x0c), None, bgc, xc.dtype)
            term = term * torch.zeros((1, 1, 1), dtype=term.dtype, device=term.device)
            xc = m._apply_injection(xc, term)
            xc = _block(blk, xc, partial(_cell_rows, site=self.coda[i], pos=pc, widx=widx), 0)
        if self.emit_source == "slot":
            lp = self._logp(m._readout(xc))
            self.cell_logits_out.copy_(self._mix(lp, self.C) if self.K > 1 else lp[0])
        # ── the span is closed: every span-local cache empties ─────────────────────
        for site in self.pre:
            site.clear()
        for site in self.coda:
            site.clear(self.Cc)
        self.C.zero_()

    # -- the host side -------------------------------------------------------------
    def feed_token(self, tok: int, pos: int) -> None:
        if self.span_len >= self.P:
            raise RuntimeError(f"span longer than rule.span_cap={self.P}")
        if pos >= self.max_pos:
            raise ValueError(f"row position {pos} exceeds the RoPE cache {self.max_pos}")
        if self._ev_tok is not None:
            self._ev_tok.synchronize()
        self.h_tok[0], self.h_tok[1], self.h_tok[2] = tok, self.prev_id, pos
        self.h_tok[3], self.h_tok[4] = self.span_len, int(self.span_len > 0)
        self.h_tok[5] = self.n_tok
        self.d_tok.copy_(self.h_tok, non_blocking=True)
        if self._ev_tok is not None:
            self._ev_tok.record()
        if self.g_tok is not None:
            self.g_tok.replay()
        else:
            self._token_step()
        self.span_len += 1
        self.n_tok += 1
        self.prev_id = tok

    def feed_slot(self, first_pos: int) -> None:
        if self.n_slots >= self.S:
            raise RuntimeError(f"more than spec.max_slots={self.S} slots")
        if first_pos + self.Kp - 1 >= self.max_pos:
            raise ValueError(f"row position {first_pos + self.Kp - 1} exceeds the RoPE cache")
        if self._ev_slot is not None:
            self._ev_slot.synchronize()
        self.h_slot[0], self.h_slot[1], self.h_slot[2] = self.n_slots, first_pos, self.span_len
        self.d_slot.copy_(self.h_slot, non_blocking=True)
        if self._ev_slot is not None:
            self._ev_slot.record()
        if self.g_slot is not None:
            self.g_slot.replay()
        else:
            self._slot_step()
        self.n_slots += 1
        self.span_len = 0
        self.prev_id = self.slot_id


@torch.no_grad()
def generate_tul_graphed(
    model,
    prompt_ids: list[int] | Tensor,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device=None,
    emit_source: str = "slot",
    use_graphs: bool = True,
    decoder: TulGraphDecoder | None = None,
    top_p: float = 0.0,
    cells: int | None = None,
):
    """``generate_tul`` / ``generate_tul_cached`` with every decode event one CUDA graph
    replay. Same arguments and returns ``(token_ids, builder)``. Pass a live ``decoder``
    (built for this model, spec, rule and emit source) to reuse its captured graphs."""
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(device)).manual_seed(seed)
    if isinstance(prompt_ids, Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if not prompt_ids:
        raise ValueError("generate_tul_graphed needs at least one prompt token")
    own = decoder is None
    dec = decoder or TulGraphDecoder(model, spec, rule, emit_source, use_graphs, cells)
    if dec.m is not model or dec.spec != spec or dec.rule is not rule \
            or dec.emit_source != emit_source:
        raise ValueError("decoder was built for a different model / spec / rule / emit_source")
    builder = TulRowBuilder(rule=rule, spec=spec)
    emitted: list[int] = []
    try:
        dec.reset()

        def _feed(tok: int) -> None:
            pos = len(builder.ids)
            cut = builder.append(int(tok))
            dec.feed_token(int(tok), pos)
            if cut:
                dec.feed_slot(builder.slot_first[-1])

        for t in prompt_ids:
            _feed(int(t))
        for step in range(max_new_tokens):
            use_cell = builder.slot_mask[-1] and emit_source == "slot"
            logits = dec.cell_logits_out if use_cell else dec.logits_out
            nxt = sample_next(logits, temperature, top_k, gen, top_p)
            emitted.append(nxt)
            if step + 1 < max_new_tokens:
                _feed(nxt)
            else:
                builder.append(int(nxt))
    finally:
        if own:
            dec.__exit__(None, None, None)
        if was_training:
            model.train()
    return emitted, builder
