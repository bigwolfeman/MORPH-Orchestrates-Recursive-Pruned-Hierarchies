"""LXTUL on a Parcae backbone: the strict slot loop with the latent-selected 4-cell fan.

Port of MORPH's `morph/configs/lxtul.yaml` forward and losses. The exact source spec,
with MORPH file:line citations, is `lxtul/SPEC.md`; section numbers below refer to it.

What is ported (SPEC section 0.1): the packed row from MORPH's own packer, the strict
masks, the slot seed `E_slot + W_sent embed(t_last)`, the 4-query register, the loop on
the compact cell axis with per-slot Poisson depth and masked freeze, the per-pass cell
RMSNorm with one shared gain, the router-followed latent-selected reset, the EMA-prelude
latent target with the rank-only head, epivol on passes 1-2, the winner-only coda write
through `W_prefix`, and every loss term with its MORPH weight.

What is Parcae's instead of MORPH's (backbone, SPEC section 0.2): one residual stream
(no Hyper-Connections), Parcae's pre-norm blocks (ReLU2 MLP, QK-norm, RoPE, value
embeddings), bf16 weights (no ternary), Parcae's diagonal injection over ALL channels
(decay init sqrt(1/5) = 0.447, B identity; MORPH injects only its 320 ctx channels, which
is where MORPH's 0.865 gain floor comes from; here a do-nothing pass has gain 0.447), no
per-layer x0 / bigram re-injection, no dropout, and Parcae's own blocks in the span
decoder (RoPE in place of MORPH's learned position table).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from parcae_lm.models.parcae.parcae import Parcae
from parcae_lm.modules.mixer import has_ve

from lxtul.attention import (coda_mask, compact_layer, core_mask, gather_core_mask, prelude_mask,
                             slot_pack_mask, span_pack_mask, use_flex)
from lxtul.cache import copy_features, mix_logp, packed_copy_probs
from lxtul.copy_heads import IdentityReach, PointerHead, mix_pointer
from lxtul.slot_compact import (SLOT_COMPACT_MODES, compact_order, put_rows as put_cells,
                                slot_depth_capacity, stratified_slot_depths, take_rows as take_cells)
from lxtul.ce import (linear_ce, linear_ce_rows, linear_ce_tokens, put_rows, static_pack,
                      take_rows)


@dataclass(frozen=True)
class LXTULConfig:
    """Every LXTUL knob, defaults = MORPH lxtul.yaml (SPEC section numbers in comments)."""
    slot_id: int                       # the <fim_pad> id at every slot position (1.1)
    max_slots: int = 64
    cells: int = 4                     # fan_k (3.3)
    span_cap: int = 32                 # register window and span-decoder length (2.4, 5.3)
    mean_depth: int = 6                # Poisson mean, per slot (3.1)
    max_depth: int = 8
    eval_depth: int = 6
    ckpt_passes: int = 4               # passes 0..ckpt_passes-1 under checkpoint (3.8)
    token_state_dropout: float = 0.15  # (5.1)
    spandec_layers: int = 2
    spandec_weight: float = 1.0
    fixed_point_lambda: float = 0.1    # (5.5)
    gain_lambda: float = 100.0         # (5.6)
    gain_target: float = 0.9
    gain_eps: float = 0.02
    cot_clip: float = 4.0              # (5.7)
    router_rank: int = 64              # (3.5)
    lsel_lambda: float = 1.0           # (3.6)
    lsel_eps: float = 0.05
    lsel_enc_lambda: float = 0.2
    lsel_enc_gamma: float = 0.1
    lsel_router_lambda: float = 1.0
    target_ema: float = 0.996
    repel_lambda: float = 0.1          # epivol (3.7)
    repel_passes: int = 2
    epi_features: int = 64
    epi_hidden: int = 256
    epi_ridge: float = 3.0
    epi_eta: float = 30.0
    geometry: str = "strict"           # "strict" (MORPH's, SPEC 2) | "open": tokens also read every
                                       #   earlier token in prelude and coda (a control arm)
    open_at: int = -1                  # >= 0: train strict, switch geometry to open at this step
                                       #   (lxtul/train.py does the switch; -1 = never)
    copy_cache: bool = False           # exact-copy cache mixed into the output (lxtul/cache.py)
    pointer_heads: int = 0             # > 0: learned pointer/copy head, output-only (copy_heads.py)
    pointer_cell_key: bool = False     # pointer keys in earlier spans also read that span's cells
    identity_reach_heads: int = 0      # > 0: tokens attend earlier tokens' RAW embeddings before
                                       #   the coda (copy_heads.IdentityReach)
    select: str = "latent"             # "latent": router-followed reset, winner-only write (3.5, 4);
                                       # "none": the ungraded fan, no reset, every cell written
    static_index: bool = False         # static-shape epivol and encoder floor: weights over all B*S
                                       #   rows instead of boolean-mask indexing (no host sync, no
                                       #   data-dependent shapes), and a ridge solve of the
                                       #   F x F normal equations instead of a float64 QR. The
                                       #   same maths; ROUNDING-ONLY (different float summation order)
    mask_cache: bool = False           # reuse the prelude/coda BlockMasks while the layout tensors
                                       #   are the same objects (the K-sweep scores one layout at
                                       #   every depth). EXACT. A training batch has a new layout
                                       #   every step, so it never hits there.
    injection: str = "full"            # "full": Parcae's diagonal injection on every channel;
    inj_lo: int = 512                  # "ctx": MORPH's, A*h + dt*e on [inj_lo, inj_hi) only,
    inj_hi: int = 832                  #   A init 0.447, dt 1, identity elsewhere (3.2)
    # --- speed keys (2026-10-14). Default off = the code path before the key, byte for byte. ---
    # compact_slots: the loop runs on the VALID slots of the whole batch packed into one sequence
    #   (one host sync per step for their count, rounded up to `compact_bucket_slots`), not on
    #   all B * max_slots slots. Same function: pad slots sit last and no valid cell reads them;
    #   a FINISHED slot still runs every pass, because later slots read its K/V and those depend on
    #   earlier slots that are still active (SPEC 2.2, 3.5), so finished slots cannot be dropped.
    #   Rounding-only (other GEMM shapes, other softmax order).
    compact_slots: bool = False
    compact_bucket_slots: int = 32     # packed length is a multiple of this many slots (128 cells at M 4:
                                       #   flex's block size); fewer distinct shapes for compile
    # static_depth: the loop trip count is max_depth in training (eval: eval_depth), so there is no
    #   `int(dslot.max())` host sync. Exact whenever some slot draws max_depth, which at 300+ slots
    #   is every step; otherwise the passes past the real max are no-ops (every slot frozen) and the
    #   hinge's random pass is drawn over max_depth passes.
    static_depth: bool = False
    # cot_clip_fn: the cotangent clip (SPEC 5.7) as two autograd Functions that share a device
    #   buffer, in place of two `register_hook` closures that mutate a Python dict. Exact.
    cot_clip_fn: bool = False
    # ce_rows: the token CE / LM head and the span decoder's CE run on the labelled rows packed to a
    #   fixed buffer proven from the shapes (lxtul/ce.py), with no `nonzero()` sync and no chunk
    #   padding to a multiple of 2048; the copy/pointer arms' per-token CE also skips the slot rows.
    #   A batch over the bound gives a NaN loss. Rounding-only (other GEMM row chunks).
    ce_rows: bool = False
    # spandec_packed: the span decoder's blocks run on the real span positions of each row packed in
    #   slot order (block-diagonal causal mask, positions = offset in the span), not on
    #   B * max_slots * span_cap positions of which about half lie past their span's end. Rounding-only.
    spandec_packed: bool = False
    # slot_depth_stratified: in training, the per-slot depths of a row are a systematic sample of
    #   clamp(Poisson(mean_depth), 1, max_depth), randomly permuted over the row's slots (device-side, no
    #   sync). Every slot keeps the SAME marginal law; the joint law changes (a row cannot draw many deep
    #   slots at once), which bounds the active slots of pass t by C_t (lxtul/slot_compact.py). NOT the
    #   same function as off: a different random draw, same distribution per slot.
    slot_depth_stratified: bool = False
    # slot_compact: "off" | "full" | "gather" (training forward only; needs slot_depth_stratified).
    #   A FINISHED cell serves every core layer's attention with the keys and values its last ACTIVE
    #   pass computed, instead of keys and values recomputed from its frozen state and the CURRENT
    #   states of the slots before it. That is a different function from off (the MORPH fast recipe's,
    #   lab/experiments/successes/2026-10-08-slot-compaction-paired.md). "full": every cell is still
    #   computed (the reference form). "gather": the same function computed on the C_t * M cells of
    #   pass t only, static shapes, no host sync; full and gather agree to rounding. Eval is unchanged.
    slot_compact: str = "off"
    slot_compact_round: int = 1        # "gather": cells per pass rounded up to a multiple of this (128 if the
                                       #   flex kernel balks at query lengths off its block size); more filler, same function
    # --- lean recipe keys (2026-10-14). Default off = the code path before the key, byte for byte. ---
    # cell_count: "fixed" (every batch has `cells` cells per slot) | "poisson": each TRAIN batch draws
    #   M = clamp(Poisson(cell_count_mean), cell_count_min, cell_count_max or cells) and its rows are
    #   re-laid out to M prefix positions per slot (lxtul/data.py, MORPH's `relayout_batch`, counter-seeded
    #   on the loader's CPU thread: no GPU sync). The M active cells are the PREFIX of the `cells` role
    #   table (reg_q_embed[:M], reg_p_cell_embed[:M], W_prefix[:M]), so the loop runs on S*M cells and the
    #   coda row is shorter: a smaller M shrinks the computation, it does not mask it. The forward reads
    #   M from `layout.prefix_k`. Not the same function as fixed: a different training distribution.
    #   Needs select="none" (the router and latent head are sized for the full fan). M = 1 has no epivol term.
    cell_count: str = "fixed"
    cell_count_mean: float = 3.5       # poisson only
    cell_count_min: int = 1            # poisson only
    cell_count_max: int = 0            # poisson only; 0 = cells
    # cell_count_eval: the M of the held-out rows (train.py's val pass and depth sweep). 0 = cells.
    #   Only with cell_count != "fixed" (a fixed model has no role prefix to run at fewer cells).
    cell_count_eval: int = 0
    # spandec_reads_cells: the span decoder cross-attends to the slot's M final cells as an UNORDERED set
    #   (no cell-position term; the output is invariant to permuting the cells) after each decoder block,
    #   in addition to its z input (the mean of the cells). MORPH's `tul.spandec_reads_cells`. The cross
    #   block's output projection is zero-initialised, so at step 0 the loss equals the mean-only decoder's.
    #   Training only (the decoder is). A different function from off by design; exact between the
    #   packed and the unpacked decoder.
    spandec_reads_cells: bool = False

    @property
    def lsel_on(self) -> bool:
        return self.cells > 1 and self.lsel_lambda > 0 and self.select == "latent"

    @property
    def repel_on(self) -> bool:
        return self.cells > 1 and self.repel_lambda > 0

    @property
    def spandec_on(self) -> bool:
        return self.spandec_weight > 0


def _ln(x: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],))


class FanRouter(nn.Module):
    """score_i = v . ReLU(W_c LN(c_i) + W_x LN(ctx) + b_x), fp32, inputs detached (3.5)."""
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.W_c = nn.Linear(d, rank, bias=False)
        self.W_x = nn.Linear(d, rank, bias=True)
        self.v = nn.Parameter(torch.randn(rank) / 8)
        for lin in (self.W_c, self.W_x):
            nn.init.normal_(lin.weight, std=0.02)
        nn.init.zeros_(self.W_x.bias)

    def forward(self, c: Tensor, ctx: Tensor) -> Tensor:  # c [B,S,M,d], ctx [B,S,d]
        with torch.autocast("cuda", enabled=False):
            hid = self.W_c(_ln(c.float())) + self.W_x(_ln(ctx.float()))[:, :, None]
            return F.relu(hid) @ self.v


class LatentHead(nn.Module):
    """g: LN -> Linear -> GELU -> Linear, fp32, one head shared by the cells (3.6)."""
    def __init__(self, d: int):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, d), nn.Linear(d, d)
        for lin in (self.l1, self.l2):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, c: Tensor) -> Tensor:
        with torch.autocast("cuda", enabled=False):
            return self.l2(F.gelu(self.l1(_ln(c.float()))))


class SliceInjection(nn.Module):
    """MORPH's DiagonalInjection with `injection_channels: ctx` (SPEC 3.2): only the slice
    [lo, hi) is refilled, h = A h + dt e with A = exp(A_log) <= 0.9999 (init 0.447) and
    dt = exp(dt_bias) (init 1); every other channel passes through. A pass whose blocks
    do nothing then has gain sqrt((d - n + n A^2) / d) = 0.865 at d 1024, n 320."""
    def __init__(self, d: int, lo: int, hi: int):
        super().__init__()
        if not 0 <= lo < hi <= d:
            raise ValueError(f"bad injection slice [{lo}, {hi}) for d={d}")
        self.lo, self.hi = lo, hi
        self.A_log = nn.Parameter(torch.full((hi - lo,), math.log(0.447)))
        self.dt_bias = nn.Parameter(torch.zeros(hi - lo))
        self.A_log._no_weight_decay = self.dt_bias._no_weight_decay = True

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        A = self.A_log.exp().clamp(max=0.9999)
        mid = x[..., self.lo:self.hi] * A + e[..., self.lo:self.hi] * self.dt_bias.exp()
        return torch.cat([x[..., :self.lo], mid.to(x.dtype), x[..., self.hi:]], -1)


class _CotAnchor(torch.autograd.Function):
    """Identity. Its backward writes the per-row norm of the gradient that reaches the loop's final
    state into `r_buf` (SPEC 5.7's reference norm). `cot_clip_fn`'s form of the hook on the final h."""

    @staticmethod
    def forward(ctx, h: Tensor, r_buf: Tensor):
        ctx.r_buf = r_buf          # a plain attribute: save_for_backward would flag the in-place write
        return h.view_as(h)

    @staticmethod
    def backward(ctx, g: Tensor):
        ctx.r_buf.copy_(g.float().flatten(1).norm(dim=1))
        return g, None


class _CotClip(torch.autograd.Function):
    """Identity. Its backward rescales each row's gradient to at most `clip * r_buf`, r_buf being the
    norm `_CotAnchor` stored (+inf until then: no clip). `cot_clip_fn`'s form of the per-pass hook."""

    @staticmethod
    def forward(ctx, h: Tensor, r_buf: Tensor, clip: float):
        ctx.r_buf, ctx.clip = r_buf, clip
        return h.view_as(h)

    @staticmethod
    def backward(ctx, g: Tensor):
        gn = g.float().flatten(1).norm(dim=1)
        s = (ctx.clip * ctx.r_buf / (gn + 1e-12)).clamp(max=1.0)
        return g * s[:, None, None].to(g.dtype), None, None


class SetCross(nn.Module):
    """`spandec_reads_cells`: cross-attention from the span decoder's token states onto a slot's M cells,
    read as an unordered set. Queries are the decoder's states, keys and values are the cells; there is no
    mask (every key belongs to the one slot the position decodes) and NO position or role term, so the
    softmax over the M cells makes the output invariant to the order of the cells. `proj` is zero-init:
    a decoder with the block starts as the mean-only decoder. MORPH `_SpanDecCross`, on Parcae's norm."""

    def __init__(self, config, d: int):
        super().__init__()
        if d % config.num_attention_heads:
            raise ValueError(f"span decoder width {d} not divisible by {config.num_attention_heads} heads")
        self.h, self.dh = config.num_attention_heads, d // config.num_attention_heads
        self.norm_q, self.norm_kv = config.Norm(d, eps=config.norm_eps), config.Norm(d, eps=config.norm_eps)
        self.q, self.kv, self.proj = nn.Linear(d, d, bias=False), nn.Linear(d, 2 * d, bias=False), nn.Linear(d, d, bias=False)
        nn.init.normal_(self.q.weight, std=0.02)
        nn.init.normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        """x [N, J, d] decoder states, mem [N, M, d] the slot's cells -> [N, J, d] residual term."""
        N, J, d = x.shape
        M = mem.shape[1]
        q = self.q(self.norm_q(x)).view(N, J, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.norm_kv(mem)).view(N, M, 2, self.h, self.dh).unbind(2)
        a = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(a.transpose(1, 2).reshape(N, J, d))

    def forward_packed(self, x: Tensor, mem: Tensor, slot: Tensor) -> Tensor:
        """The packed decoder's form. x [B, P, d] states, mem [B, S, M, d] the cells of every slot, slot
        [B, P] the slot each position decodes (in range at unused positions too: their rows are masked
        out of the loss). K and V are projected once per slot and gathered to the positions."""
        B, P, d = x.shape
        S, M = mem.shape[1], mem.shape[2]
        q = self.q(self.norm_q(x)).view(B, P, self.h, self.dh)
        kv = self.kv(self.norm_kv(mem)).view(B, S, M * 2 * d)
        kv = kv.gather(1, slot[..., None].expand(-1, -1, M * 2 * d)).view(B, P, M, 2, self.h, self.dh)
        k, v = kv.unbind(3)                                              # [B, P, M, H, dh]
        w = torch.einsum("bphd,bpmhd->bphm", q, k).float() * (self.dh ** -0.5)
        a = torch.einsum("bphm,bpmhd->bphd", w.softmax(-1).to(v.dtype), v)
        return self.proj(a.reshape(B, P, d))


def tok_row_cap(L: int, K: int, J: int, S: int) -> int:
    """Upper bound, from shapes alone, on the labelled TOKEN rows of one packed row (`ce_rows`).
    A row with n valid slots holds L - K n token positions (K cells per slot) and at most
    J (n + 1) tokens: n closed spans of at most J, plus an open tail shorter than a span."""
    return min(max(min(L - K * n, J * (n + 1)) for n in range(S + 1)), L)


def span_row_cap(L: int, K: int, J: int, S: int) -> int:
    """Upper bound on the span decoder's labelled positions of one packed row (`spandec_packed`).
    Slot s decodes span s + 1, so only slots 0..n-2 decode, each at most J tokens, and the decoded
    tokens are token positions: min(L - K n, J (n - 1))."""
    return min(max(min(L - K * n, J * max(n - 1, 0)) for n in range(S + 1)), S * J)


@dataclass
class _CPlan:
    """One pass of the compacted slot loop (`slot_compact`). `active` [B, N]: the cells whose pass
    it is. "gather": `idx` [B, n] the cells computed (active first, then inactive filler; n = C_t * M)
    and `mask` the strict relation of those query rows against all N cells. "full": idx None, mask the
    dense core mask. `freqs`, `ves`: RoPE table and value embeddings of the cells computed."""
    active: Tensor
    idx: Tensor | None
    mask: object
    freqs: Tensor
    ves: list


@dataclass
class _SlotPack:
    """The valid slots of a batch as one sequence (`compact_slots`). `sel` [Ns]: flat slot ids
    (b * S + s), valid ones first in row-major order; the rest of the Ns are pad slots that fill
    the bucket. `mask`, `freqs`, `ves` are the loop's attention mask, RoPE table and value
    embeddings for the N = Ns * M cells in that order. `bad`: 0-dim bool, a valid slot follows a pad
    slot in some row (the packed mask would drop a cell some valid slot reads); the forward turns that
    into a NaN loss, never a silently different function."""
    sel: Tensor
    mask: object
    freqs: Tensor
    ves: list
    bad: Tensor
    cells: int = 0               # M of this pack; 0 = the config's `cells`


class LXTULParcae(Parcae):
    def __init__(self, config, tul: LXTULConfig):
        super().__init__(config, gradient_checkpointing=False)
        if config.n_embd != config.recurrent_embedding_dimension:
            raise ValueError("LXTUL writes the loop state into the coda: state dim must be n_embd")
        if config.injection_type != "diagonal":
            raise ValueError("LXTUL port expects Parcae's diagonal injection")
        if config.block_size < tul.max_slots * tul.cells + 1:
            raise ValueError("block_size must cover the packed row and the cell axis")
        if tul.geometry not in ("strict", "open"):
            raise ValueError(f"unknown geometry {tul.geometry!r}")
        if tul.copy_cache and tul.pointer_heads > 0:
            raise ValueError("copy_cache and pointer_heads are separate arms; pick one output mixture")
        if tul.open_at >= 0 and tul.geometry != "strict":
            raise ValueError("open_at switches strict -> open; start from geometry='strict'")
        if tul.select not in ("latent", "none"):
            raise ValueError(f"unknown select {tul.select!r}")
        if tul.slot_compact not in SLOT_COMPACT_MODES:
            raise ValueError(f"slot_compact must be one of {SLOT_COMPACT_MODES}, got {tul.slot_compact!r}")
        if tul.slot_compact != "off":
            if not tul.slot_depth_stratified:
                raise ValueError("slot_compact needs slot_depth_stratified: the capacity C_t of a "
                                 "compacted pass is a bound on the stratified draw")
            if tul.compact_slots:
                raise ValueError("slot_compact and compact_slots are two ways to shrink the loop and are "
                                 "not combined: compact_slots packs rows into one sequence, slot_compact "
                                 "keeps the batch axis and gathers per row")
        if tul.compact_slots and tul.compact_bucket_slots < 1:
            raise ValueError("compact_bucket_slots must be >= 1")
        if tul.spandec_packed and not tul.spandec_on:
            raise ValueError("spandec_packed needs the span decoder (spandec_weight > 0)")
        if tul.cell_count not in ("fixed", "poisson"):
            raise ValueError(f"cell_count must be 'fixed' or 'poisson', got {tul.cell_count!r}")
        if tul.cell_count == "fixed":
            stray = [k for k, v in (("cell_count_mean", tul.cell_count_mean != 3.5),
                                    ("cell_count_min", tul.cell_count_min != 1),
                                    ("cell_count_max", tul.cell_count_max != 0),
                                    ("cell_count_eval", tul.cell_count_eval != 0)) if v]
            if stray:
                raise ValueError(f"{stray} set with cell_count='fixed': the knobs would be silently ignored")
        else:
            hi = tul.cell_count_max or tul.cells
            if tul.cells < 2:
                raise ValueError("cell_count='poisson' needs cells >= 2")
            if tul.select != "none":
                raise ValueError("cell_count='poisson' needs select='none': the router and the latent head "
                                 "are sized for the full fan")
            if not (tul.cell_count_mean > 0 and 1 <= tul.cell_count_min <= hi <= tul.cells):
                raise ValueError(f"cell_count law needs mean > 0 and 1 <= min <= max <= cells, got mean "
                                 f"{tul.cell_count_mean}, min {tul.cell_count_min}, max {hi}, cells {tul.cells}")
            if not 0 <= tul.cell_count_eval <= tul.cells:
                raise ValueError(f"cell_count_eval {tul.cell_count_eval} outside [0, cells={tul.cells}]")
        if tul.spandec_reads_cells and not (tul.spandec_on and tul.cells > 1):
            raise ValueError("spandec_reads_cells needs the span decoder and cells > 1 (one cell is the "
                             "mean already)")
        if tul.cells > 1 and tul.select == "latent" and not tul.lsel_on:
            raise ValueError("select=latent with cells > 1 needs lsel_lambda > 0 to pick a winner")
        if tul.cells == 1:
            if tul.select == "latent":
                raise ValueError("select='latent' with cells=1: there is no fan to select from; use "
                                 "select='none'")
            if tul.repel_lambda > 0:
                raise ValueError("repel_lambda > 0 with cells=1: epivol repels cells from each other and "
                                 "one cell has nothing to repel from; set repel_lambda=0")
        del self.transformer["C"]          # W_prefix writes the loop state into the coda (4)
        self.tul = tul
        d, M = config.n_embd, tul.cells
        if tul.injection == "ctx":
            self.transformer["adapter"] = SliceInjection(d, tul.inj_lo, tul.inj_hi)
        elif tul.injection != "full":
            raise ValueError(f"unknown injection {tul.injection!r}")
        use_flex(self.transformer.prelude)
        use_flex(self.transformer.core_block)
        use_flex(self.transformer.coda)

        def lin(i, o, zero=False):
            m = nn.Linear(i, o, bias=False)
            nn.init.zeros_(m.weight) if zero else nn.init.normal_(m.weight, std=0.02)
            return m
        # slot seed (1.4); E_slot is set to the embedding-table mean in init_slot_seed()
        self.E_slot = nn.Parameter(torch.zeros(d))
        self.W_sent = lin(d, d)
        self.E_mask = nn.Parameter(torch.zeros(d))             # token-state dropout (5.1)
        # register (3.3): distinct queries, W_o zero, P_cell zero => reg = 0 at step 0
        self.reg_q_embed = nn.Parameter(torch.randn(M, d) * 0.02)
        self.reg_k, self.reg_v, self.reg_o = lin(d, d), lin(d, d), lin(d, d, zero=True)
        self.reg_p_cell_embed = nn.Parameter(torch.zeros(M, d))
        self.cell_norm = config.Norm(d, eps=1e-6)                # (3.4)
        self.W_prefix = nn.ParameterList(nn.Parameter(torch.eye(d)) for _ in range(M))  # (4)
        if tul.copy_cache:
            # gate over (model, bigram cache, unigram cache) from the token's own hidden state and
            # the match counts. "norm" in the name routes it to recpre's AdamW scale group: Muon
            # on a 3-row matrix would normalise its update. Init: a ~ (0.96, 0.02, 0.02).
            self.copy_gate_norm_head = nn.Linear(d + 4, 3)
            with torch.no_grad():
                self.copy_gate_norm_head.weight.zero_()
                self.copy_gate_norm_head.bias.copy_(torch.tensor([0.0, -4.0, -4.0]))
        if tul.pointer_heads > 0:
            self.pointer = PointerHead(d, tul.pointer_heads, cell_key=tul.pointer_cell_key)
        if tul.identity_reach_heads > 0:
            self.ident_reach = IdentityReach(d, tul.identity_reach_heads)
        if tul.lsel_on:
            self.router = FanRouter(d, tul.router_rank)
            self.latent_head = LatentHead(d)
            # EMA twin of the prelude (3.6): frozen, stepped by ema_update(); lookups read live
            self.twin_prelude = copy.deepcopy(self.transformer.prelude)
            for p in self.twin_prelude.parameters():
                p.requires_grad_(False)
        if tul.spandec_on:
            # span decoder (5.3): Parcae blocks at layer ids with no value embedding
            n_layer = config.n_layer
            sd_ids = [i for i in range(n_layer, n_layer + 4 * tul.spandec_layers)
                      if not has_ve(i, n_layer)][: tul.spandec_layers]
            self.sd_blocks = nn.ModuleList(config.Block(config, layer_id=i) for i in sd_ids)
            self.sd_z_in, self.sd_tok_in = lin(d, d), lin(d, d)
            self.sd_out_norm = config.Norm(d, eps=config.norm_eps)
            if tul.spandec_packed:
                use_flex(self.sd_blocks)         # the packed decoder runs under a BlockMask
            if tul.spandec_reads_cells:
                self.sd_cross = nn.ModuleList(SetCross(config, d) for _ in sd_ids)   # one per block
        if tul.repel_on:
            # epivol reservoir (3.7): frozen, seed 0
            g = torch.Generator().manual_seed(0)
            self.register_buffer("epi_W1", torch.randn(tul.epi_hidden, d, generator=g) / math.sqrt(d))
            self.register_buffer("epi_W2", torch.randn(tul.epi_features, tul.epi_hidden, generator=g)
                                 / math.sqrt(tul.epi_hidden))
        self._core_mask = None
        self._mask_memo = None                   # mask_cache: last (prelude, coda) masks and their key
        self._core_masks: dict[int, object] = {}  # M -> core mask (cell_count: one per cell count)
        self._compact_layer = compact_layer      # compile_blocks() swaps in the compiled one
        self.metrics: dict[str, Tensor] = {}

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def init_slot_seed(self) -> None:
        """E_slot = mean of the (tied) embedding table at step 0 (1.4)."""
        self.E_slot.copy_(self.transformer.wte.weight.float().mean(0) * self.emb_scale)

    @torch.no_grad()
    def ema_update(self) -> None:
        if not self.tul.lsel_on:
            return
        live = [p for p in self.transformer.prelude.parameters()]
        twin = [p for p in self.twin_prelude.parameters()]
        torch._foreach_lerp_(twin, live, 1.0 - self.tul.target_ema)

    def compile_blocks(self) -> None:
        """Per-block inductor compile, the Parcae author's selective scheme."""
        c = lambda b: torch.compile(b, backend="inductor", mode="default", dynamic=False)
        t = self.transformer
        t.prelude = nn.ModuleList(c(b) for b in t.prelude)
        t.core_block = nn.ModuleList(c(b) for b in t.core_block)
        t.coda = nn.ModuleList(c(b) for b in t.coda)
        if self.tul.lsel_on:
            self.twin_prelude = nn.ModuleList(c(b) for b in self.twin_prelude)
        if self.tul.spandec_on:
            self.sd_blocks = nn.ModuleList(c(b) for b in self.sd_blocks)
        if self.tul.slot_compact != "off":
            self._compact_layer = c(compact_layer)

    # ------------------------------------------------------------------ pieces
    def _cells_of(self, layout) -> int:
        """Cells per slot in this forward. cell_count 'fixed': `cells`, as it always was. Otherwise the
        batch's own count, `layout.prefix_k` (a host int: no sync), which must lie in [1, cells]."""
        if self.tul.cell_count == "fixed":
            return self.tul.cells
        m = int(layout.prefix_k)
        if not 1 <= m <= self.tul.cells:
            raise ValueError(f"layout has {m} cells per slot; cell_count={self.tul.cell_count!r} runs "
                             f"1..{self.tul.cells} (the role table has {self.tul.cells} rows)")
        return m

    @staticmethod
    def _roles(table: Tensor, M: int) -> Tensor:
        """The first M rows of a per-role table (cells are roles 0 .. M-1); the table itself at full M."""
        return table if M == table.shape[0] else table[:M]

    def _ve(self, idx: int, ids: Tensor):
        key = str(idx)
        return self.value_embeds[key](ids) if key in self.value_embeds else None

    def _prelude(self, blocks, x, ids, freqs, mask):
        for i, block in enumerate(blocks):
            x = block(x, freqs, mask, ve=self._ve(i, ids))
        return x

    def _core_step(self, h: Tensor, e: Tensor, freqs: Tensor, ves: list) -> Tensor:
        """One pass: diagonal injection of the cell's own seed, then the shared blocks (3.2)."""
        x = self.transformer.adapter(h, e)
        for block, ve in zip(self.transformer.core_block, ves):
            x = block(x, freqs, self._core_mask, ve=ve)
        return x

    def _compact_plans(self, dcell: Tensor, T: int, ves: list, cfreqs: Tensor, M: int) -> list:
        """One `_CPlan` per pass. "gather" pass t computes C_t * M cells per row, C_t the stratified
        draw's capacity; a device assert (no host read) stops the run if the draw ever disagrees."""
        tul, S = self.tul, self.tul.max_slots
        B, N = dcell.shape
        if tul.slot_compact == "full":
            return [_CPlan(dcell > t, None, self._core_mask, cfreqs, ves) for t in range(T)]
        caps = slot_depth_capacity(float(tul.mean_depth), int(tul.max_depth), S)
        off = self.config.n_layers_in_prelude
        plans = []
        for t in range(T):
            act = dcell > t
            n = min(-(-caps[t] * M // tul.slot_compact_round) * tul.slot_compact_round, N)
            torch._assert_async(act.sum(dim=1).max() <= n,
                                "slot_compact: more active cells than the capacity C_t")
            idx = compact_order(act)[:, :n]
            ids = torch.full((B, n), tul.slot_id, device=dcell.device, dtype=torch.long)
            pves = [self._ve(off + i, ids) for i in range(len(self.transformer.core_block))]
            plans.append(_CPlan(act, idx, gather_core_mask(idx, N, M), self.freqs_cis[0][idx], pves))
        return plans

    def _core_step_compact(self, h: Tensor, e: Tensor, plan: _CPlan, caches):
        """`_core_step` of the compacted loop: dense [B, N, d] in, dense out (a cell that is not
        active comes back as it went in), plus the per-layer (K, V) caches the next pass reads."""
        if plan.idx is None:
            hc, ec = h, e
        else:
            hc, ec = take_cells(h, plan.idx), take_cells(e, plan.idx)
        x = self.transformer.adapter(hc, ec)
        new = []
        for l, (block, ve) in enumerate(zip(self.transformer.core_block, plan.ves)):
            kc, vc = (None, None) if caches is None else caches[l]
            x, K, V = self._compact_layer(block, x, plan.freqs, plan.mask, ve, kc, vc, plan.idx, plan.active)
            new.append((K, V))
        if plan.idx is None:
            out = torch.where(plan.active[..., None], x.to(h.dtype), h)
        else:
            out = put_cells(x, h, plan.idx, plan.active)
        return out, tuple(new)

    def _slot_pack(self, valid: Tensor, M: int = 0) -> _SlotPack:
        """Gather map, mask, RoPE table and value embeddings of the packed loop (`compact_slots`)."""
        tul = self.tul
        M = M or tul.cells
        B, S = valid.shape
        dev = valid.device
        bucket = tul.compact_bucket_slots
        n_valid = int(valid.sum())            # the key's one host sync: it fixes this step's shape
        ns = min(B * S, max(1, -(-n_valid // bucket)) * bucket)
        sel = torch.argsort((~valid).flatten().long(), stable=True)[:ns]
        row, slot = sel // S, sel % S
        cell = torch.arange(M, device=dev)
        cell_row = row[:, None].expand(ns, M).reshape(-1)
        cell_slot = slot[:, None].expand(ns, M).reshape(-1)
        pos = (slot[:, None] * M + cell).reshape(-1)             # the cell's index on the dense axis
        off = self.config.n_layers_in_prelude
        ids = torch.full((1, ns * M), self.tul.slot_id, device=dev, dtype=torch.long)
        ves = [self._ve(off + i, ids) for i in range(len(self.transformer.core_block))]
        bad = (valid[:, 1:] & ~valid[:, :-1]).any()
        return _SlotPack(sel, slot_pack_mask(cell_row, cell_slot), self.freqs_cis[:, pos], ves, bad, M)

    def _core_step_packed(self, h: Tensor, e: Tensor, pack: _SlotPack) -> Tensor:
        """`_core_step` on the packed cells: dense [B, S*M, d] in, dense out. Slots that are not in
        the pack come back as zeros; nothing reads them (invalid slots are never read by a valid
        one, and every loss term masks them)."""
        B, SM, d = h.shape
        M = pack.cells or self.tul.cells
        sel = pack.sel
        pk = lambda t: t.reshape(-1, M, d).index_select(0, sel).reshape(1, -1, d)
        x = self.transformer.adapter(pk(h), pk(e))
        for block, ve in zip(self.transformer.core_block, pack.ves):
            x = block(x, pack.freqs, pack.mask, ve=ve)
        out = x.new_zeros(B * (SM // M), M, d).index_copy(0, sel, x.reshape(-1, M, d))
        return out.reshape(B, SM, d)

    def _gain_penalty(self, h: Tensor, e: Tensor, freqs, ves, m: Tensor, pack=None, compact=None) -> Tensor:
        """100 * mean_b relu(gain_b - 0.9)^2 on the RAW map (no cell norm), sources detached (5.6)."""
        tul = self.tul
        hp, e = h.detach(), e.detach()
        mf = m[..., None].to(hp.dtype)
        v = torch.randn_like(hp) * mf
        scale = tul.gain_eps * hp.float().norm(dim=-1) / v.float().norm(dim=-1).clamp_min(1e-12)
        dlt = v * scale[..., None].to(hp.dtype)
        if compact is not None:       # this pass's map, frozen cells served from the caches it read
            plan, cs = compact
            cs = None if cs is None else tuple((k.detach(), v.detach()) for k, v in cs)
            step = lambda x: self._core_step_compact(x, e, plan, cs)[0]
        elif pack is None:
            step = lambda x: self._core_step(x, e, freqs, ves)
        else:
            step = lambda x: self._core_step_packed(x, e, pack)
        f0 = checkpoint(step, hp, use_reentrant=False)
        f1 = checkpoint(step, hp + dlt, use_reentrant=False)
        num = ((f1 - f0).float() * mf).flatten(1).norm(dim=1)
        gain = num / (dlt.float().flatten(1).norm(dim=1) + 1e-6)
        self.metrics["gain"] = gain.detach().mean()
        return tul.gain_lambda * F.relu(gain - tul.gain_target).pow(2).mean()

    def _epivol(self, traj: list, valid: Tensor, M: int) -> Tensor:
        """-(mean_t epi_t) - (mean_t vol_t) over passes 1..repel_passes (3.7). Needs M >= 2."""
        tul = self.tul
        if tul.static_index:
            return self._epivol_static(traj, valid, M)
        B, SM, d = traj[0].shape
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            seed = traj[0].view(B, -1, M, d).float().mean(2)[valid]               # [N, d]
            H = F.elu(_ln(seed) @ self.epi_W1.T) @ self.epi_W2.T                  # [N, F]
            H = (H - H.mean(0)) / H.std(0).clamp_min(1e-6) / math.sqrt(H.shape[1])
            eye = torch.eye(H.shape[1], device=H.device, dtype=torch.float64)
            Q, R = torch.linalg.qr(torch.cat([H.double(), math.sqrt(tul.epi_ridge) * eye]))
            a = torch.linalg.solve_triangular(R, Q[: H.shape[0]].T, upper=True).float()  # [F, N]
        epis, vols = [], []
        with torch.autocast("cuda", enabled=False):
            for t in range(1, min(tul.repel_passes, len(traj) - 1) + 1):
                zc = traj[t].view(B, -1, M, d).float()[valid]                     # [N, M, d]
                dev = (zc - zc.mean(1, keepdim=True)) / zc.norm(dim=-1).mean(1).clamp_min(1e-6)[:, None, None]
                W = torch.einsum("fn,nmd->mfd", a, dev - dev.mean(0, keepdim=True))
                eyeF = torch.eye(W.shape[1], device=W.device)
                epi = 0.5 * torch.logdet(eyeF + tul.epi_eta * W @ W.transpose(1, 2)) / math.log(2)
                epis.append(epi.mean() / W.shape[1])
                G = dev @ dev.transpose(1, 2)
                eyeM = torch.eye(M, device=G.device)
                vol = 0.5 * torch.logdet(eyeM + tul.epi_eta * G) / math.log(2) / (M - 1)
                vols.append(vol.mean())
        epi, vol = torch.stack(epis).mean(), torch.stack(vols).mean()
        self.metrics["epi"], self.metrics["vol"] = epi.detach(), vol.detach()
        return -epi - vol

    def _epivol_static(self, traj: list, valid: Tensor, M: int) -> Tensor:
        """`_epivol` over all B*S rows with the invalid ones weighted out (`static_index`).

        The ridge map `a = R^-1 Q1^T` of the original (QR of [H; sqrt(rho) I]) equals
        `(H^T H + rho I)^-1 H^T`, so it comes from one F x F Cholesky solve in float64. Rows with
        `valid` False get an all-zero column in `a`, so they add nothing to W; their means, std
        and the vol average are weighted by `valid`. Inputs are activations, so nothing here can
        be cached across steps."""
        tul = self.tul
        B, SM, d = traj[0].shape
        w = valid.reshape(-1)                                                      # [B*S] bool
        wf = w.float()
        n = wf.sum()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            seed = traj[0].view(B, -1, M, d).float().mean(2).reshape(-1, d)        # [B*S, d]
            H = F.elu(_ln(seed) @ self.epi_W1.T) @ self.epi_W2.T                   # [B*S, F]
            mu = (H * wf[:, None]).sum(0) / n
            sd = ((((H - mu) ** 2) * wf[:, None]).sum(0) / (n - 1).clamp_min(1)).sqrt()
            H = (H - mu) / sd.clamp_min(1e-6) / math.sqrt(H.shape[1]) * wf[:, None]
            Hd = H.double()
            eye = torch.eye(H.shape[1], device=H.device, dtype=torch.float64)
            L = torch.linalg.cholesky(Hd.T @ Hd + tul.epi_ridge * eye)
            a = torch.cholesky_solve(Hd.T, L).float()                              # [F, B*S]
        epis, vols = [], []
        with torch.autocast("cuda", enabled=False):
            for t in range(1, min(tul.repel_passes, len(traj) - 1) + 1):
                zc = traj[t].view(B, -1, M, d).float().reshape(-1, M, d)           # [B*S, M, d]
                dev = (zc - zc.mean(1, keepdim=True)) / zc.norm(dim=-1).mean(1).clamp_min(1e-6)[:, None, None]
                dev = torch.where(w[:, None, None], dev, torch.zeros_like(dev))
                W = torch.einsum("fn,nmd->mfd", a, dev - dev.sum(0, keepdim=True) / n)
                eyeF = torch.eye(W.shape[1], device=W.device)
                epi = 0.5 * torch.logdet(eyeF + tul.epi_eta * W @ W.transpose(1, 2)) / math.log(2)
                epis.append(epi.mean() / W.shape[1])
                G = dev @ dev.transpose(1, 2)
                eyeM = torch.eye(M, device=G.device)
                vol = 0.5 * torch.logdet(eyeM + tul.epi_eta * G) / math.log(2) / (M - 1)
                vols.append((vol * wf).sum() / n)
        epi, vol = torch.stack(epis).mean(), torch.stack(vols).mean()
        self.metrics["epi"], self.metrics["vol"] = epi.detach(), vol.detach()
        return -epi - vol

    def _masks(self, bag: Tensor, is_slot: Tensor, op: bool, M: int):
        """-> (prelude, coda) BlockMasks. With `mask_cache` the last pair is reused while `bag` and
        `is_slot` are the same unmodified tensors and the cell count M is the same (the key holds the
        tensors, so ids cannot be reused; M is in the key so a varying cell count cannot serve a stale pair)."""
        if not self.tul.mask_cache:
            return prelude_mask(bag, is_slot, op), coda_mask(bag, is_slot, op)
        memo = self._mask_memo
        if memo is not None:
            m_bag, m_slot, m_vb, m_vs, m_op, m_M, masks = memo
            if m_bag is bag and m_slot is is_slot and m_vb == bag._version \
                    and m_vs == is_slot._version and m_op == op and m_M == M:
                return masks
        masks = prelude_mask(bag, is_slot, op), coda_mask(bag, is_slot, op)
        self._mask_memo = (bag, is_slot, bag._version, is_slot._version, op, M, masks)
        return masks

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids: Tensor, labels: Tensor | None, layout, depth: int | None = None,
                **_ignored) -> dict:
        """`layout` is MORPH's SlotLayout. `depth` forces every slot to that many passes
        (the K-sweep); None draws Poisson depth in training and uses eval_depth in eval."""
        tul, S = self.tul, self.tul.max_slots
        M = self._cells_of(layout)
        B, L = input_ids.shape
        d = self.config.n_embd
        dev = input_ids.device
        train = self.training
        is_slot, bag = layout.slot_mask, layout.bag_id
        sidx, valid = layout.slot_index, layout.slot_valid
        if sidx.shape[1] != S:
            raise ValueError(f"layout has {sidx.shape[1]} slots, model was built for {S}")
        if M not in self._core_masks:
            self._core_masks[M] = core_mask(S * M, M, dev)
        self._core_mask = self._core_masks[M]
        op = tul.geometry == "open"
        pmask, cmask = self._masks(bag, is_slot, op, M)
        freqs, cfreqs = self.freqs_cis[:, :L], self.freqs_cis[:, : S * M]
        ar_m, ar_j = torch.arange(M, device=dev), torch.arange(tul.span_cap, device=dev)
        cpos = sidx[:, :, None] + ar_m                                   # [B,S,M] cell positions
        vmask = valid[..., None].float()

        # --- input: tokens, and the slot seed at every cell (1.4) ---
        emb = self.transformer.wte(input_ids) * self.emb_scale
        tlast = emb.gather(1, (sidx - 1).clamp_min(0)[..., None].expand(-1, -1, d))
        seed_add = (self.W_sent(tlast) * vmask)[:, :, None].expand(-1, -1, M, -1).reshape(B, S * M, d)
        x = torch.where(is_slot[..., None], self.E_slot.to(emb.dtype), emb)
        x = x.scatter_add(1, cpos.reshape(B, -1, 1).expand(-1, -1, d), seed_add.to(x.dtype))

        # --- prelude, live and EMA twin (2.1, 3.6) ---
        xp = self._prelude(self.transformer.prelude, x, input_ids, freqs, pmask)
        xn = self.transformer.ln_prelude(xp)

        # --- next-span windows: span s+1 sits at [sidx[s]+M, sidx[s+1]) (1.1) ---
        nvalid = F.pad(valid[:, 1:], (0, 1), value=False)
        nstart = sidx + M
        nend = torch.where(nvalid, F.pad(sidx[:, 1:], (0, 1)), nstart)
        npos = (nstart[..., None] + ar_j).clamp_max(L - 1)                 # [B,S,J]
        nok = (nstart[..., None] + ar_j < nend[..., None]) & nvalid[..., None]
        if labels is not None and tul.lsel_on:
            lab_at = labels.gather(1, npos.flatten(1)).view_as(npos)
            score_ok = nok & (lab_at >= 0)
            ok = valid & score_ok.any(-1)

            def pooled(src):
                g = src.gather(1, npos.flatten(1)[..., None].expand(-1, -1, d)).view(B, S, -1, d)
                w = score_ok[..., None].float()
                return _ln((g.float() * w).sum(2) / w.sum(2).clamp_min(1.0))
            with torch.no_grad():
                xt = self._prelude(self.twin_prelude, x.detach(), input_ids, freqs, pmask)
                z = pooled(xt)                                              # [B,S,d] target
            zo = pooled(xp)                                                 # live, for L_enc
        else:
            z = ok = zo = None

        # --- register (3.3): 4 queries pool the span's own prelude token states ---
        spos = (sidx[..., None] - tul.span_cap + ar_j).clamp_min(0)        # window before slot
        own = (bag.gather(1, spos.flatten(1)).view_as(spos) == torch.arange(S, device=dev)[:, None]) \
            & ~is_slot.gather(1, spos.flatten(1)).view_as(spos) & (sidx[..., None] - tul.span_cap + ar_j >= 0) \
            & valid[..., None]
        win = xn.gather(1, spos.flatten(1)[..., None].expand(-1, -1, d)).view(B, S, -1, d)
        k, v = self.reg_k(win), self.reg_v(win)                            # [B,S,J,d]
        att = torch.einsum("md,bsjd->bsmj", self._roles(self.reg_q_embed, M).to(k.dtype), k).float() / math.sqrt(d)
        att = att.masked_fill(~own[:, :, None], float("-inf"))
        att = torch.where(own.any(-1)[:, :, None, None], att, torch.zeros_like(att)).softmax(-1)
        reg = self.reg_o(torch.einsum("bsmj,bsjd->bsmd", att.to(v.dtype), v)) + self._roles(self.reg_p_cell_embed, M)
        reg = reg * vmask[..., None]

        # --- loop entry (1.5) ---
        e = xn.gather(1, cpos.reshape(B, -1, 1).expand(-1, -1, d)).view(B, S, M, d) * vmask[..., None]
        e = (e + reg.to(e.dtype)).view(B, S * M, d)
        h = e
        # the router's context (the slot's first prelude state); only latent selection reads it
        ctx = xn.gather(1, sidx[..., None].expand(-1, -1, d)).detach() if tul.lsel_on else None   # [B,S,d]
        pack = self._slot_pack(valid, M) if tul.compact_slots else None
        if pack is None:
            slot_ids = torch.full((B, S * M), tul.slot_id, device=dev, dtype=torch.long)
            off = self.config.n_layers_in_prelude
            ves = [self._ve(off + i, slot_ids) for i in range(len(self.transformer.core_block))]
        else:
            ves = None                    # the pack carries its own, at the packed length

        # --- depth (3.1) ---
        if depth is not None:
            dslot = torch.full((B, S), int(depth), device=dev)
        elif train and tul.slot_depth_stratified:
            dslot = stratified_slot_depths(B, S, float(tul.mean_depth), int(tul.max_depth), dev)
        elif train:
            dslot = torch.poisson(torch.full((B, S), float(tul.mean_depth), device=dev))
            dslot = dslot.clamp(1, tul.max_depth).long()
        else:
            dslot = torch.full((B, S), tul.eval_depth, device=dev)
        dslot = torch.where(valid, dslot, torch.ones_like(dslot))
        dcell = dslot.repeat_interleave(M, dim=1)
        vcell = valid.repeat_interleave(M, dim=1)
        if tul.static_depth:
            T = int(depth) if depth is not None else (tul.max_depth if train else tul.eval_depth)
        else:
            T = int(dslot.max())
        t_gain = int(torch.randint(T, (1,))) if train and tul.gain_lambda > 0 else -1

        # --- the loop (3.9) ---
        ref: dict = {}

        def clip_hook(g):
            if "r" not in ref:
                return g
            gn = g.float().flatten(1).norm(dim=1)
            s = (tul.cot_clip * ref["r"] / (gn + 1e-12)).clamp(max=1.0)
            return g * s[:, None, None].to(g.dtype)

        traj = [h]
        final = torch.zeros(B, S, dtype=torch.long, device=dev)
        fp_sum = h.new_zeros((), dtype=torch.float32)
        fp_n = h.new_zeros((), dtype=torch.float32)
        rce_sum, rce_n = fp_sum.clone(), fp_n.clone()
        gain_pen = fp_sum.clone()
        cot_fn = tul.cot_clip_fn and train and tul.cot_clip > 0
        r_buf = h.new_full((B,), float("inf"), dtype=torch.float32) if cot_fn else None
        if pack is None:
            step = lambda hh, ee: self._core_step(hh, ee, cfreqs, ves)
        else:
            step = lambda hh, ee: self._core_step_packed(hh, ee, pack)
        # slot_compact: one plan per pass; the (K, V) caches ride the loop, graph and all
        comp = tul.slot_compact != "off" and train and depth is None
        plans = self._compact_plans(dcell, T, ves, cfreqs, M) if comp else None
        caches = None
        for t in range(T):
            active = dcell > t
            caches_in = caches
            if comp:
                cstep = lambda hh, ee, cc, plan=plans[t]: self._core_step_compact(hh, ee, plan, cc)
                if t < tul.ckpt_passes:
                    h_new, caches = checkpoint(cstep, h, e, caches_in, use_reentrant=False)
                else:
                    h_new, caches = cstep(h, e, caches_in)
            elif train and t < tul.ckpt_passes:
                h_new = checkpoint(step, h, e, use_reentrant=False)
            else:
                h_new = step(h, e)
            if t == t_gain:
                gain_pen = self._gain_penalty(h, e, cfreqs, ves, active & vcell, pack,
                                              (plans[t], caches_in) if comp else None)
            h_new = self.cell_norm(h_new)
            if train and h_new.requires_grad and tul.cot_clip > 0:
                if cot_fn:
                    h_new = _CotClip.apply(h_new, r_buf, tul.cot_clip)
                else:
                    h_new.register_hook(clip_hook)
            if train and tul.fixed_point_lambda > 0:
                fin = (active & vcell & (dcell == t + 1)).float()
                r = (h_new - h).float().pow(2).sum(-1) / (h_new.float().pow(2).sum(-1) + 1e-6)
                fp_sum, fp_n = fp_sum + (r * fin).sum(), fp_n + fin.sum()
            h = torch.where(active[..., None], h_new, h)
            traj.append(h)
            if not tul.lsel_on:          # one cell, or the ungraded fan: no selection, no reset
                continue
            # latent-selected reset (3.5)
            c = h.view(B, S, M, d).detach()
            scores = self.router(c, ctx)
            rpick = scores.argmax(-1)
            act_s = valid & (dslot > t)
            if z is not None:
                with torch.no_grad():
                    tpick = (self.latent_head(c) - z[:, :, None]).pow(2).mean(-1).argmin(-1)
                g_s = (act_s & ok).float()
                rce = F.cross_entropy(scores.flatten(0, 1), tpick.flatten(), reduction="none")
                rce_sum, rce_n = rce_sum + (rce * g_s.flatten()).sum(), rce_n + g_s.sum()
            final = torch.where(act_s, rpick, final)
            reset = valid & (dslot > t + 1)
            hv = h.view(B, S, M, d)
            win_state = hv.gather(2, rpick[:, :, None, None].expand(-1, -1, 1, d))
            h = torch.where(reset[:, :, None, None], win_state.expand(-1, -1, M, -1), hv).view(B, S * M, d)
        if pack is not None:
            h = h + torch.where(pack.bad, torch.full((), float("nan"), device=dev), torch.zeros((), device=dev)).to(h.dtype)
        if train and h.requires_grad and tul.cot_clip > 0:
            if cot_fn:
                h = _CotAnchor.apply(h, r_buf)
            else:
                h.register_hook(lambda g: ref.__setitem__("r", g.float().flatten(1).norm(dim=1)))

        # --- write: the final winner alone, through W_prefix[winner] (4) ---
        hv = h.view(B, S, M, d)
        if tul.select == "none":      # ungraded fan: every cell into its own prefix position
            onehot = torch.ones(B, S, M, device=dev, dtype=h.dtype) * vmask
        else:
            onehot = F.one_hot(final, M).to(h.dtype) * vmask
        Wp = torch.stack(list(self.W_prefix)[:M]).to(h.dtype)             # [M,d,d], the first M roles
        vals = torch.einsum("bsmd,mde->bsme", hv * onehot[..., None], Wp)
        safe = torch.where(valid[..., None], cpos, torch.full_like(cpos, L))  # pads -> dump row
        xc = torch.cat([xn, xn.new_zeros(B, 1, d)], 1)
        xc = xc.scatter(1, safe.reshape(B, -1, 1).expand(-1, -1, d), vals.reshape(B, -1, d).to(xc.dtype))[:, :L]
        if train and tul.token_state_dropout > 0:
            drop = (torch.rand(B, L, device=dev) < tul.token_state_dropout) & ~is_slot
            xc = torch.where(drop[..., None], self.E_mask.to(xc.dtype), xc)
        if tul.identity_reach_heads > 0:
            # earlier tokens' raw embeddings, never their computation; slot cells untouched
            xc = xc + self.ident_reach(xc, emb, is_slot).to(xc.dtype)

        # --- coda (2.3) and the token CE (5.1) ---
        off = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i, block in enumerate(self.transformer.coda):
            xc = block(xc, freqs, cmask, ve=self._ve(off + i, input_ids))
        xc = self.transformer.ln_f(xc)
        out = {"metrics": self.metrics, "final": final}
        if labels is None:
            out["hidden"] = xc
            return out
        W = self.lm_head.weight
        tok_labels = torch.where(is_slot, torch.full_like(labels, -100), labels)
        if tul.ce_rows:
            # labelled token rows only, packed to a buffer fixed by the shapes (no sync, no padding)
            K = int(getattr(layout, "prefix_k", M))
            cap = B * tok_row_cap(L, K, tul.span_cap, S)
            src, used, back, over = static_pack((tok_labels >= 0).reshape(1, -1), cap)
            xr = take_rows(xc.reshape(1, B * L, d), src, back)[0]
            lab_r = torch.where(used[0], tok_labels.reshape(-1).gather(0, src[0]),
                                torch.full_like(src[0], -100))
            nan = torch.full((), float("nan"), device=dev)
        if tul.copy_cache or tul.pointer_heads > 0:
            if tul.ce_rows:
                ce_r = linear_ce_tokens(xr, W, lab_r, self.config.init.logit_scale, tul.slot_id)
                ce_t = put_rows(ce_r[None], src, used, back).view(B, L)
                ce_t = ce_t + torch.where(over, nan, torch.zeros_like(nan))
            else:
                ce_t = linear_ce_tokens(xc.flatten(0, 1), W, tok_labels.flatten(),
                                        self.config.init.logit_scale, tul.slot_id).view(B, L)
            lp = self._out_mix(xc, -ce_t, input_ids, labels, is_slot, layout)
            tok_ok = tok_labels >= 0                     # not `ok`: the latent-selection loss reads that
            n_ok = tok_ok.sum().clamp_min(1)
            ce = -(lp * tok_ok).sum() / n_ok
            out["ce_model"] = ((ce_t * tok_ok).sum() / n_ok).detach()
        elif tul.ce_rows:
            ce = linear_ce_rows(xr, W, lab_r, self.config.init.logit_scale, tul.slot_id)
            ce = torch.where(over, nan, ce)
        else:
            ce = linear_ce(xc.flatten(0, 1), W, tok_labels.flatten(), self.config.init.logit_scale,
                           tul.slot_id)
        out["ce"] = ce
        loss = ce
        if tul.spandec_on:
            sd = self._spandec(hv, input_ids, npos, nok, W, layout)
            out["spandec"] = sd
            loss = loss + tul.spandec_weight * sd
        if tul.lsel_on:
            l_lat, l_enc, l_rce = self._lsel_exit(hv, z, ok, zo, rce_sum, rce_n)
            loss = loss + tul.lsel_lambda * l_lat + tul.lsel_enc_lambda * l_enc + tul.lsel_router_lambda * l_rce
            self.metrics.update(l_lat=l_lat.detach(), l_enc=l_enc.detach(), l_rce=l_rce.detach())
        if train:
            if tul.gain_lambda > 0:          # off: no hinge passes were run, there is no term to add
                loss = loss + gain_pen
            if tul.fixed_point_lambda > 0:
                loss = loss + tul.fixed_point_lambda * fp_sum / fp_n.clamp_min(1)
            if tul.repel_on and len(traj) > 1 and M > 1:     # one cell has nothing to repel from
                loss = loss + tul.repel_lambda * self._epivol(traj, valid, M)
        out["loss"] = loss
        self.metrics.update(passes=torch.tensor(T))
        if tul.fixed_point_lambda > 0:       # off: the residual was never computed
            self.metrics["fp"] = (fp_sum / fp_n.clamp_min(1)).detach()
        if tul.cell_count != "fixed":
            self.metrics["cells"] = torch.tensor(M)
        return out

    def _out_mix(self, h, logp_model, input_ids, labels, is_slot, layout=None):
        """log p of the target under the gated output mixture (copy cache or pointer), [B, L]."""
        if self.tul.pointer_heads > 0:
            g, p, null = self.pointer(h, labels, is_slot, layout)
            return mix_pointer(logp_model, g, p, null)
        c = packed_copy_probs(input_ids, labels, is_slot)
        g = self.copy_gate_norm_head(torch.cat([h, copy_features(c).to(h.dtype)], -1))
        return mix_logp(logp_model, g, c)

    @torch.no_grad()
    def token_nll(self, h, input_ids, labels, is_slot, layout=None):
        """Per-position NLL of `labels` from coda output `h` (slots and -100 read 0), with the
        copy cache mixed in when the arm has one. The eval path's one scoring rule."""
        tok_labels = torch.where(is_slot, torch.full_like(labels, -100), labels)
        lg = (h @ self.lm_head.weight.T).float() * self.config.init.logit_scale
        lg[..., self.tul.slot_id] = float("-inf")
        nll = torch.nn.functional.cross_entropy(lg.flatten(0, 1), tok_labels.flatten().clamp_min(0),
                                                reduction="none").view(tok_labels.shape)
        if self.tul.copy_cache or self.tul.pointer_heads > 0:
            nll = -self._out_mix(h, -nll, input_ids, labels, is_slot, layout)
        return torch.where(tok_labels >= 0, nll, 0.0)

    def _spandec(self, hv, input_ids, npos, nok, W, layout=None):
        """Teacher-forced decoder of span s+1 from the mean of slot s's 4 final cells (5.3)."""
        tul = self.tul
        B, S, M, d = hv.shape
        # --- span decoder (5.3): mean of the 4 final raw cells ---
        zsd = self.transformer.ln_f(hv.mean(2))                             # [B,S,d]
        mem = self.transformer.ln_f(hv) if tul.spandec_reads_cells else None   # [B,S,M,d] the set
        tgt = input_ids.gather(1, npos.flatten(1)).view_as(npos)           # [B,S,J]
        K = int(getattr(layout, "prefix_k", M))
        if tul.spandec_packed:
            return self._spandec_packed(zsd, tgt, nok, W, input_ids.shape[1], K, mem)
        tin = self.transformer.wte(tgt).detach() * self.emb_scale
        tin = self.sd_tok_in(tin) * nok[..., None]
        sx = torch.cat([self.sd_z_in(zsd)[:, :, None], tin[:, :, :-1]], 2).flatten(0, 1)
        sfreq = self.freqs_cis[:, : tul.span_cap]
        mem_n = None if mem is None else mem.flatten(0, 1)                  # [B*S,M,d]
        for i, block in enumerate(self.sd_blocks):
            sx = block(sx, sfreq, None)
            if mem_n is not None:
                sx = sx + self.sd_cross[i](sx, mem_n)
        sx = self.sd_out_norm(sx)
        sd_lab = torch.where(nok, tgt, torch.full_like(tgt, -100)).flatten()
        if tul.ce_rows:
            cap = B * span_row_cap(input_ids.shape[1], K, tul.span_cap, S)
            src, used, back, over = static_pack((sd_lab >= 0).reshape(1, -1), cap)
            xr = take_rows(sx.reshape(1, -1, d), src, back)[0]
            lab_r = torch.where(used[0], sd_lab.gather(0, src[0]), torch.full_like(src[0], -100))
            sd = linear_ce_rows(xr, W.detach(), lab_r, self.config.init.logit_scale, tul.slot_id)
            return torch.where(over, torch.full_like(sd, float("nan")), sd)
        sd = linear_ce(sx.flatten(0, 1), W.detach(), sd_lab, self.config.init.logit_scale, tul.slot_id)
        return sd

    def _spandec_packed(self, zsd: Tensor, tgt: Tensor, nok: Tensor, W: Tensor, L: int, K: int,
                        mem: Tensor | None = None) -> Tensor:
        """The span decoder on the real span positions only (`spandec_packed`). Per row, the positions
        with a label, in (slot, offset) order, fill a buffer of `span_row_cap` positions (rounded up
        to flex's 128). The input at offset 0 is the slot's z, at offset j > 0 the embedding of the
        span's token j - 1; RoPE position = offset; attention is causal inside one span. Per
        position that is what the unpacked decoder computes: its causal attention never lets a
        position read one past its span's end."""
        tul = self.tul
        B, S, J = tgt.shape
        d = zsd.shape[-1]
        cap = -(-span_row_cap(L, K, J, S) // 128) * 128
        src, used, back, over = static_pack(nok.reshape(B, S * J), cap)          # [B, cap]
        slot_k, off_k = src // J, src % J
        zpart = self.sd_z_in(zsd).gather(1, slot_k[..., None].expand(-1, -1, d))
        flat_tgt = tgt.reshape(B, S * J)
        prev = flat_tgt.gather(1, (src - 1).clamp_min(0))
        tpart = self.sd_tok_in(self.transformer.wte(prev).detach() * self.emb_scale)
        sx = torch.where((off_k == 0)[..., None], zpart, tpart)
        mask = span_pack_mask(torch.where(used, slot_k, torch.full_like(slot_k, S)))
        freqs = self.freqs_cis[0][off_k]                                          # [B, cap, 1, hd/2, 2]
        for i, block in enumerate(self.sd_blocks):
            sx = block(sx, freqs, mask)
            if mem is not None:           # spandec_reads_cells: each position reads its own slot's cells
                sx = sx + self.sd_cross[i].forward_packed(sx, mem, slot_k)
        sx = self.sd_out_norm(sx)
        lab = torch.where(used, flat_tgt.gather(1, src), torch.full_like(src, -100))
        sd = linear_ce_rows(sx.reshape(-1, d), W.detach(), lab.reshape(-1),
                            self.config.init.logit_scale, tul.slot_id)
        return torch.where(over, torch.full_like(sd, float("nan")), sd)

    def _lsel_exit(self, hv, z, ok, zo, rce_sum, rce_n):
        """Exit latent loss (rank-only), the online variance floor, the router CE (3.6)."""
        tul = self.tul
        M, d = hv.shape[2], hv.shape[3]
        c = hv.detach()
        dist = (self.latent_head(c) - z[:, :, None]).pow(2).mean(-1)        # [B,S,M]
        tex = dist.detach().argmin(-1, keepdim=True)
        d_win = dist.gather(-1, tex).squeeze(-1)
        d_oth = (dist.sum(-1) - d_win) / (M - 1)
        okf = ok.float()
        l_lat = (((1 - tul.lsel_eps) * d_win + tul.lsel_eps * d_oth) * okf).sum() / okf.sum().clamp_min(1)
        if tul.static_index:
            # the same floor over all B*S rows weighted by ok: no boolean index, no shape sync
            w = ok.reshape(-1, 1).float()
            n = w.sum()
            zf = zo.reshape(-1, d)
            mu = (zf * w).sum(0) / n.clamp_min(1)
            var = (((zf - mu) ** 2) * w).sum(0) / (n - 1).clamp_min(1)
            std = torch.where(n > 1, var, torch.ones_like(var)).sqrt()      # keeps sqrt's grad finite
            l_enc = torch.where(n > 1, F.relu(tul.lsel_enc_gamma - std).mean(), zo.new_zeros(()))
        else:
            zv = zo[ok]
            l_enc = F.relu(tul.lsel_enc_gamma - zv.std(0) if zv.shape[0] > 1 else zv.new_zeros(d)).mean()
        l_rce = rce_sum / rce_n.clamp_min(1)

        return l_lat, l_enc, l_rce
