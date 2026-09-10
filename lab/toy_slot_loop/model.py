"""Toy slot-TUL model: prelude -> shared core loop on slot cells -> coda.

The design mirrors MORPH's slot loop (see the TUL section of the repo CLAUDE.md and
docs/tul-spec.md sec 3), stripped to the parts that decide how updates flow:

  row layout, one span:   [tok x L] [SLOT] [PRE x P]
  prelude   : n_pre causal blocks over ALL cells        -> entry state e per slot
  core loop : ONE shared block applied T times to slot cells only (causal over slots),
              with the entry state re-injected on a channel slice every pass
  write     : z (the exit state) -> W_prefix_p -> the P prefix cells of that span
  coda      : n_coda causal blocks over tokens and prefix cells (slot cells masked out
              as keys, and token->token attention restricted to the own span, so the
              ONLY route from an earlier span to a token is z)

Every switch is a constructor argument. The hot loop branches only on flags fixed at
build time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

TOKEN, SLOT, PREFIX = 0, 1, 2


# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------


@dataclass
class Layout:
    """Fixed-shape packing of a row: n_spans copies of [L tokens][1 slot][P prefix]."""

    n_spans: int
    span_len: int
    prefix_k: int

    @property
    def width(self) -> int:
        return self.span_len + 1 + self.prefix_k

    @property
    def n_cells(self) -> int:
        return self.n_spans * self.width

    def cell_kind(self) -> torch.Tensor:
        k = torch.empty(self.n_cells, dtype=torch.long)
        for i in range(self.n_spans):
            b = i * self.width
            k[b : b + self.span_len] = TOKEN
            k[b + self.span_len] = SLOT
            k[b + self.span_len + 1 : b + self.width] = PREFIX
        return k

    def span_of_cell(self) -> torch.Tensor:
        return torch.arange(self.n_cells) // self.width

    def token_positions(self) -> torch.Tensor:
        return (self.cell_kind() == TOKEN).nonzero(as_tuple=True)[0]

    def slot_positions(self) -> torch.Tensor:
        return (self.cell_kind() == SLOT).nonzero(as_tuple=True)[0]

    def prefix_positions(self) -> torch.Tensor:
        """[n_spans, prefix_k] global positions of each span's prefix cells."""
        p = (self.cell_kind() == PREFIX).nonzero(as_tuple=True)[0]
        return p.view(self.n_spans, self.prefix_k)


def build_masks(layout: Layout, coda_reads_z: bool = True) -> dict[str, torch.Tensor]:
    """Boolean attention masks, True = the query may attend to the key."""
    n = layout.n_cells
    kind = layout.cell_kind()
    span = layout.span_of_cell()
    idx = torch.arange(n)
    causal = idx[:, None] >= idx[None, :]

    prelude = causal.clone()

    same_span = span[:, None] == span[None, :]
    is_tok_k = (kind == TOKEN)[None, :].expand(n, n)
    is_pre_k = (kind == PREFIX)[None, :].expand(n, n)
    is_slot_k = (kind == SLOT)[None, :].expand(n, n)

    # tokens see their own span's earlier tokens, plus every earlier prefix cell.
    coda = causal & ((is_tok_k & same_span) | (is_pre_k if coda_reads_z else torch.zeros_like(causal)))
    coda = coda & ~is_slot_k
    coda = coda | torch.eye(n, dtype=torch.bool)  # every query keeps itself: no empty row

    s = layout.n_spans
    si = torch.arange(s)
    core = si[:, None] >= si[None, :]
    return {"prelude": prelude, "coda": coda, "core": core}


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class Attn(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.nh = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.nh, self.dh).transpose(1, 2)
        k = k.view(B, N, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, N, self.nh, self.dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attn(d, n_heads)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, d_ff)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class CoreBlock(nn.Module):
    """One shared loop block. `pass_lora_rank > 0` adds a per-pass rank-r additive delta."""

    def __init__(self, d: int, n_heads: int, d_ff: int, max_depth: int, pass_lora_rank: int = 0):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attn(d, n_heads)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, d_ff)
        self.rank = pass_lora_rank
        if pass_lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(max_depth, pass_lora_rank, d) / math.sqrt(d))
            self.lora_B = nn.Parameter(torch.zeros(max_depth, d, pass_lora_rank))

    def forward(self, x, mask, pass_idx: int):
        x = x + self.attn(self.n1(x), mask)
        h = self.n2(x)
        y = self.mlp(h)
        if self.rank > 0:
            pass_idx = min(pass_idx, self.lora_A.shape[0] - 1)
            a = torch.einsum("bsd,rd->bsr", h, self.lora_A[pass_idx])
            y = y + torch.einsum("bsr,dr->bsd", a, self.lora_B[pass_idx])
        return x + y


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


@dataclass
class ToyConfig:
    vocab: int = 12
    d_model: int = 96
    n_heads: int = 4
    d_ff: int = 256
    n_prelude: int = 2
    n_coda: int = 2
    layout: Layout = field(default_factory=lambda: Layout(8, 3, 2))

    # loop
    mean_depth: float = 6.0
    max_depth: int = 8
    fixed_depth: int = 0  # 0 = Poisson draw per slot

    # entry and carry
    entry: str = "prelude"  # "prelude" | "noise"
    noise_scale: float = 0.5
    inject_decay: float = 0.45  # < 0 disables injection
    inject_frac: float = 0.25  # fraction of channels the injection writes

    # loss attachment: exit | mux_all | mux_all_detach | staged | deep_coda | progressive
    attach: str = "exit"
    mux_weight: float = 1.0
    progressive_p: float = 0.5
    bptt_last: int = 0  # 0 = full BPTT; k>0 keeps grad on the last k passes only

    # terms and capacity
    fixed_point_lambda: float = 0.0
    pass_lora_rank: int = 0

    # reader
    coda_reads_z: bool = True
    coda_token_input: str = "embed"  # "embed" | "prelude"

    def __post_init__(self):
        assert self.attach in {
            "exit",
            "mux_all",
            "mux_all_detach",
            "staged",
            "deep_coda",
            "progressive",
        }
        assert self.entry in {"prelude", "noise"}
        assert self.coda_token_input in {"embed", "prelude"}


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------


class ToySlotLoop(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        lay = cfg.layout
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab, d)
        self.pos = nn.Parameter(torch.randn(lay.n_cells, d) * 0.02)
        self.E_slot = nn.Parameter(torch.randn(d) * 0.02)
        self.E_pre = nn.Parameter(torch.randn(d) * 0.02)

        self.prelude = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_prelude)])
        self.core = CoreBlock(d, cfg.n_heads, cfg.d_ff, cfg.max_depth, cfg.pass_lora_rank)
        self.coda = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_coda)])

        self.core_in = nn.Linear(d, d, bias=False)
        self.W_inj = nn.Linear(d, d, bias=False)
        self.W_prefix = nn.Linear(d, lay.prefix_k * d, bias=False)
        self.norm_out = RMSNorm(d)
        self.norm_mux = RMSNorm(d)

        masks = build_masks(lay, cfg.coda_reads_z)
        for k, v in masks.items():
            self.register_buffer(f"mask_{k}", v, persistent=False)
        self.register_buffer("cell_kind", lay.cell_kind(), persistent=False)
        self.register_buffer("slot_pos", lay.slot_positions(), persistent=False)
        self.register_buffer("tok_pos", lay.token_positions(), persistent=False)
        self.register_buffer("pre_pos", lay.prefix_positions(), persistent=False)
        self.n_inj = max(1, int(cfg.inject_frac * d))

        # instrument state, filled by a forward when the taps are on
        self.tap_params: list[dict[str, torch.Tensor]] = []
        self.cot_hooks: list[torch.Tensor] = []
        self._tap = False
        self._cot = False

    # -- heads (tied to the input embedding, as in the real model) ----------------------

    def head(self, x):
        return F.linear(self.norm_out(x), self.embed.weight)

    def mux_head(self, h):
        return F.linear(self.norm_mux(h), self.embed.weight)

    # -- stages -------------------------------------------------------------------------

    def _base(self, tokens: torch.Tensor) -> torch.Tensor:
        """Cell embeddings before any block: tokens, E_slot, E_pre, plus positions."""
        B = tokens.shape[0]
        d = self.cfg.d_model
        x = self.E_pre.expand(B, self.cfg.layout.n_cells, d).clone()
        x[:, self.tok_pos] = self.embed(tokens)
        x[:, self.slot_pos] = self.E_slot.expand(B, self.slot_pos.numel(), d)
        return x + self.pos

    def _front(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, n_tokens] -> x [B, n_cells, d] after the prelude."""
        x = self._base(tokens)
        for blk in self.prelude:
            x = blk(x, self.mask_prelude)
        return x

    def sample_depths(self, B: int, device, generator=None) -> torch.Tensor:
        c = self.cfg
        S = c.layout.n_spans
        if c.fixed_depth > 0:
            return torch.full((B, S), c.fixed_depth, dtype=torch.long, device=device)
        rate = torch.full((B, S), c.mean_depth, device=device)
        d = torch.poisson(rate, generator=generator).long()
        return d.clamp(1, c.max_depth)

    def _core_params(self, t: int):
        """Parameters for pass t. With the tap on, each pass gets its own graph node."""
        if not self._tap:
            return None
        p = {k: v * 1.0 for k, v in self.core.named_parameters()}
        b = dict(self.core.named_buffers())
        for v in p.values():
            v.retain_grad()
        self.tap_params.append(p)
        return {**p, **b}

    def _core_apply(self, h, t: int):
        params = self._core_params(t)
        if params is None:
            return self.core(h, self.mask_core, t)
        return functional_call(self.core, params, (h, self.mask_core, t))

    def loop(self, e: torch.Tensor, depths: torch.Tensor, generator=None):
        """Run the shared core on the slot states.

        e       [B, S, d] entry state from the prelude
        depths  [B, S]    realised depth per slot
        returns (z, h0, states) where states is the list of post-pass states (live graph).
        """
        c = self.cfg
        if c.entry == "prelude":
            h = self.core_in(e)
        else:
            h = torch.randn(e.shape, device=e.device, dtype=e.dtype, generator=generator) * c.noise_scale
        h0 = h
        Tmax = int(depths.max().item())

        prefix_cut = None
        if c.attach == "progressive" and self.training:
            take = torch.rand(depths.shape, device=e.device, generator=generator) < c.progressive_p
            hi = (depths - 1).clamp(min=1)
            k = (torch.rand(depths.shape, device=e.device, generator=generator) * hi).long() + 1
            prefix_cut = torch.where(take & (depths >= 2), k, torch.zeros_like(k))
        if c.bptt_last > 0:
            cut = (depths - c.bptt_last).clamp(min=0)
            prefix_cut = cut if prefix_cut is None else torch.maximum(prefix_cut, cut)

        inj = self.W_inj(e) if c.inject_decay >= 0 else None
        states: list[torch.Tensor] = []
        h_prev_last = h
        for t in range(Tmax):
            active = (t < depths).unsqueeze(-1)
            h_in = h.detach() if c.attach == "mux_all_detach" else h
            if inj is not None:
                gamma = c.inject_decay**t
                pad = torch.zeros_like(h_in)
                pad[..., : self.n_inj] = inj[..., : self.n_inj]
                h_in = h_in + gamma * pad
            if self._cot:
                if h_in.requires_grad:
                    h_in.retain_grad()
                self.cot_hooks.append(h_in)
            h_new = self._core_apply(h_in, t)
            if prefix_cut is not None:
                cut = (t < prefix_cut).unsqueeze(-1)
                h_new = torch.where(cut, h_new.detach(), h_new)
            # a slot that has reached its depth stops moving; remember the state before
            # its last pass for the fixed-point term
            h_prev_last = torch.where(active & (t + 1 == depths).unsqueeze(-1), h, h_prev_last)
            h = torch.where(active, h_new, h)
            states.append(h)
        return h, h0, states, h_prev_last

    def _write_and_coda(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        d = self.cfg.d_model
        vals = self.W_prefix(z).view(B, self.cfg.layout.n_spans, self.cfg.layout.prefix_k, d)
        x = x.clone()
        x[:, self.pre_pos.reshape(-1)] = vals.reshape(B, -1, d)
        for blk in self.coda:
            x = blk(x, self.mask_coda)
        return x

    # -- losses --------------------------------------------------------------------------

    def token_ce(self, xh, labels, reduce=True):
        logits = self.head(xh[:, self.tok_pos])
        lo = logits.reshape(-1, logits.shape[-1]).float()
        la = labels.reshape(-1)
        if reduce:
            return F.cross_entropy(lo, la, ignore_index=-100)
        return F.cross_entropy(lo, la, ignore_index=-100, reduction="none").view(labels.shape)

    def mux_ce(self, h, target, mask=None):
        logits = self.mux_head(h).reshape(-1, self.cfg.vocab).float()
        tgt = target.reshape(-1).clone()
        if mask is not None:
            tgt = torch.where(mask.reshape(-1), tgt, torch.full_like(tgt, -100))
        return F.cross_entropy(logits, tgt, ignore_index=-100)

    def forward(self, batch, generator=None, force_depth: int = 0, z_override: str = ""):
        """batch: dict with tokens [B,Nt], labels [B,Nt], mux_next [B,S], mux_own [B,S]."""
        c = self.cfg
        tokens = batch["tokens"]
        x = self._front(tokens)
        e = x[:, self.slot_pos]
        # The coda reads RAW cell embeddings for token cells (the real model's
        # `coda_token_input: embed`). Feeding it the prelude output would give every
        # token a causal path to earlier spans and make z redundant by construction.
        cbase = x if c.coda_token_input == "prelude" else self._base(tokens)
        if force_depth > 0:
            depths = torch.full(
                (tokens.shape[0], c.layout.n_spans), force_depth, dtype=torch.long, device=tokens.device
            )
        else:
            depths = self.sample_depths(tokens.shape[0], tokens.device, generator)
        z, h0, states, h_prev_last = self.loop(e, depths, generator)

        if z_override == "entry":
            z_use = h0
        elif z_override == "zero":
            z_use = torch.zeros_like(z)
        else:
            z_use = z

        out: dict[str, torch.Tensor] = {}
        if c.attach == "deep_coda":
            ces = []
            for t, h_t in enumerate(states):
                use = torch.where((t < depths).unsqueeze(-1), h_t, z_use)
                ces.append(self.token_ce(self._write_and_coda(cbase, use), batch["labels"]))
            token_ce = torch.stack(ces).mean()
            xh = self._write_and_coda(cbase, z_use)
            out["token_ce_exit"] = self.token_ce(xh, batch["labels"]).detach()
        else:
            xh = self._write_and_coda(cbase, z_use)
            token_ce = self.token_ce(xh, batch["labels"])
            out["token_ce_exit"] = token_ce.detach()

        mux = torch.zeros((), device=tokens.device)
        valid = torch.ones_like(depths, dtype=torch.bool)
        if c.attach in {"exit", "progressive"}:
            mux = self.mux_ce(z, batch["mux_next"], valid)
        elif c.attach in {"mux_all", "mux_all_detach"}:
            terms = [
                self.mux_ce(h_t, batch["mux_next"], (t < depths)) for t, h_t in enumerate(states)
            ]
            mux = torch.stack(terms).mean()
        elif c.attach == "staged":
            terms = []
            for t, h_t in enumerate(states):
                last = (t + 1) == depths
                mid = (t < depths) & ~last
                if mid.any():
                    terms.append(self.mux_ce(h_t, batch["mux_own"], mid))
                if last.any():
                    terms.append(self.mux_ce(h_t, batch["mux_next"], last))
            mux = torch.stack(terms).mean() if terms else mux
        elif c.attach == "deep_coda":
            mux = self.mux_ce(z, batch["mux_next"], valid)

        loss = token_ce + c.mux_weight * mux
        if c.fixed_point_lambda > 0:
            diff = (z - h_prev_last).pow(2).mean()
            scale = z.detach().pow(2).mean() + 1e-6
            fp = diff / scale
            loss = loss + c.fixed_point_lambda * fp
            out["fixed_point"] = fp.detach()

        out.update(
            {
                "loss": loss,
                "token_ce": token_ce.detach(),
                "mux": mux.detach(),
                "z": z,
                "h0": h0,
                "depths": depths,
                "xh": xh,
            }
        )
        return out
