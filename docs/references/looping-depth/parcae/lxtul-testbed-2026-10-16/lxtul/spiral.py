"""SpiralFormer-style multi-resolution recursion on the plain Parcae arm (arXiv 2602.11698).

The plain arm loops its 6-layer core about 6 times over ALL token positions. A spiral arm runs the
SAME core (same weights, same Parcae injection, same state init) on SHORTER sequences: the token
states are pooled into groups, the core loops over the group sequence, and the result is written
back into the token states, coarse to fine. Loss is plain next-token CE (through the same
`PlainPointer` head as plain + pointer); there is no auxiliary term.

Forward (Anchor topology, paper eq. 5):

    h0 = ln_prelude(prelude(tokens))                    full resolution, causal
    u  = 0
    for level in levels (coarse to fine):
        h  = h0 + u                                     Anchor: the latest update only
        z  = Down_level(h)                              learned softmax pool per group (eq. 17-18)
        zh = core loop over z                           causal attention over the group sequence,
                                                        RoPE position = group index
        u  = Up_level(RMSNorm_level(C(zh)))             gain 1/sqrt(g), broadcast to tokens, causal shift
    logits = lm_head(ln_f(coda(h0 + u)))

Down: one scorer vector per level (`pool_w`, zero-init so the pool starts as a mean; a Linear(d, 1)
bias is a softmax no-op and is not built). The value embeddings the core layers read are pooled with
the SAME weights as the state, so a group carries one state and one value-embedding bundle.
Up: `C`, Parcae's own loop read-out projection, then a per-level RMSNorm (`up_norm`, learned weight
init 1), then every token receives the state of ONE group times lambda / g = 1 / sqrt(group size)
(uniform broadcast, eq. 20 with lambda = sqrt(g); the 1/g is the uniform allocation beta = 1/g).
Why the norm (2026-10-15, measured): plain Parcae's coda reads C(x) INSTEAD of h0, so its scale is
learned end to end; here C(x) is ADDED to h0 and Parcae's loop state is not normed. The first 5k runs
(no norm, gain sqrt(g) without the 1/g) read update RMS 536, 1.9e5, 4.6e7, 6.9e9 over the four
published levels against RMS(h0) 1.04 (Anchor compounds it: each level pools h0 + u), CE 6.95.
Levels:
  chunk  fixed groups of `size` tokens, half-chunk offset (eq. 12), shift `size - 1` (eq. 22). The
         last incomplete chunk is dropped (eq. 14); it has no receiver, so dropping and keeping it
         are the same function.
  span   groups are the spans of the TUL boundary rule (MORPH's `BoundaryRule`, the rule the
         minimal arm's packer cuts with). Span j's state is written ONLY to the tokens of span j + 1
         (no-overlap regime, paper appendix E), so it reaches no token at or before span j's last
         token. The span axis is padded to `max_spans` so shapes are static; a padded span has no
         member token and is never read. Spans past `max_spans` (a row with more) are not pooled:
         their tokens form one tail that receives the last pooled span's state (causal, counted in
         `SpanLayout.n_bound`). The span level loops with Parcae's own depth sampler (n no-grad
         then k grad passes per row, truncated at `max_depth`); chunk levels take `passes`.

Depth argument: `depth` sets the number of core passes of the ONE sampled-depth level: the span level, or
a chunk level with `depth: poisson` (chunks of any size >= 1; size 1 is plain Parcae's loop, one group per
token, with the coda reading h0 + u). Both draw Parcae's law (n no-grad, k grad, truncated at `max_depth`)
and both run `_loop`. A model has at most one such level; every other chunk level runs `passes` fixed core
passes. A model with no sampled-depth level (the published spiral) has a fixed depth and raises on any `depth`.

Cycles (`spiral.cycle`, default absent = everything above, byte-identical). The hierarchy above is descended ONCE. With a
`cycle` block the whole coarse-to-fine descent repeats as an outer loop (a multigrid V-cycle): under the Anchor topology
the level chain just continues, so the next cycle's first level pools h0 + the latest update, which already holds the
fine levels' last inference. The cycle COUNT follows Parcae's law (n no-grad then k grad cycles, truncated at `max`,
drawn per ROW like a poisson level's depth); each cycle STARTS at a sampled level and descends to the finest, skipping
each later level with probability `skip_p` (training only). Eval: no skips, start at the coarsest level, `round(mean)`
cycles (or `depth=`). `credit: end` = one CE after the last cycle; `deep` = HRM-style deep supervision, a CE after EVERY
cycle on the carried state DETACHED between cycles (training only; eval scores the last cycle). Per-level parameters
and the core are shared across cycles; no parameter is added.

Causality is by construction (shift >= size - 1 for chunks, "next span only" for spans); the tests in
tests/test_spiral.py perturb tokens and break each guard once to show they fail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from parcae_lm.models.parcae.parcae import Parcae

from lxtul.attention import block_memory_mask, use_flex
from lxtul.plain_deep import head_logits


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class LevelSpec:
    kind: str                      # "chunk" | "span"
    size: int = 0                  # chunk: tokens per chunk
    offset: int = -1               # chunk: omega, -1 -> size // 2 (paper default)
    shift: int = -1                # chunk: s, -1 -> size - 1 (paper default); must be >= size - 1
    passes: int = 1                # chunk: core passes at this level (fixed depth only; poisson takes passes == 1)
    depth: str = "fixed"           # "fixed": `passes` core passes. "poisson": Parcae's sampled depth (chunk only; the
                                   #   span level always samples). Default "fixed" is byte-identical to before.

    def __post_init__(self):
        if self.kind == "chunk":
            if self.size < 1:
                raise ValueError(f"chunk level needs size >= 1, got {self.size}")
            off = self.size // 2 if self.offset < 0 else self.offset
            sh = self.size - 1 if self.shift < 0 else self.shift
            if not 0 <= off < self.size:
                raise ValueError(f"chunk offset must be in [0, size), got {off} for size {self.size}")
            if sh < self.size - 1:
                raise ValueError(f"chunk shift {sh} < size - 1 = {self.size - 1}: the update of a chunk "
                                 f"would reach tokens before the chunk's last token (a causality leak)")
            if self.passes < 1:
                raise ValueError(f"passes must be >= 1, got {self.passes}")
            if self.depth not in ("fixed", "poisson"):
                raise ValueError(f"unknown depth {self.depth!r} (fixed | poisson)")
            if self.depth == "poisson" and self.passes != 1:
                raise ValueError(f"a poisson chunk level samples its depth: passes must stay 1, got {self.passes}")
            object.__setattr__(self, "offset", off)
            object.__setattr__(self, "shift", sh)
        elif self.kind == "span":
            if self.size or self.offset >= 0 or self.shift >= 0 or self.passes != 1:
                raise ValueError("a span level takes no size, offset, shift or passes: its groups are the "
                                 "boundary rule's spans and its depth is Parcae's sampler")
            if self.depth != "fixed":
                raise ValueError("a span level always samples its depth; depth must be left at its default "
                                 f"(got {self.depth!r})")
        else:
            raise ValueError(f"unknown level kind {self.kind!r} (chunk | span)")


@dataclass(frozen=True)
class CycleSpec:
    """The outer cycle of a spiral (`spiral.cycle`). See the module docstring."""
    mean: float = 3.0              # Poisson mean of the cycle count (drawn per row, floor 1, truncated at `max`)
    max: int = 6                   # cap on the cycle count (static upper bound of the schedule)
    start: str = "uniform"         # "uniform": each cycle starts at a level drawn uniformly; "coarsest": always level 0
    skip_p: float = 0.0            # training only: each level after the start level is skipped with this probability
    credit: str = "end"            # "end": one CE after the last cycle. "deep": a CE after every cycle, state detached between
    bptt: int = -1                 # grad cycles per row (the rest run no-grad first); -1 -> the model's mean_backprop_depth

    def __post_init__(self):
        if not self.mean > 0 or self.max < 1:
            raise ValueError(f"cycle needs mean > 0 and max >= 1, got mean {self.mean}, max {self.max}")
        if self.start not in ("uniform", "coarsest"):
            raise ValueError(f"unknown cycle start {self.start!r} (uniform | coarsest)")
        if not 0.0 <= self.skip_p < 1.0:
            raise ValueError(f"cycle skip_p must be in [0, 1), got {self.skip_p}")
        if self.credit not in ("end", "deep"):
            raise ValueError(f"unknown cycle credit {self.credit!r} (end | deep)")
        if self.bptt == 0 or self.bptt < -1:
            raise ValueError(f"cycle bptt must be >= 1 (or -1 for the model's mean_backprop_depth), got {self.bptt}")


@dataclass(frozen=True)
class SpiralConfig:
    levels: tuple[LevelSpec, ...]
    max_spans: int = 96            # static span axis: the OWT plain rows hold 51 spans on average, 88 at most
                                   #   in 120 rows (morph-scratch/abq/spiral/span_stats.py)
    max_depth: int = 8             # the span loop's depth cap (Parcae's sampler has none; MORPH's max_depth)
    topology: str = "anchor"       # "anchor": h = h0 + the latest update (SpiralFormer eq. 5)
                                   # "mesh": a buffer of n_levels + 3 token-shaped slots with per-level
                                   #   softmax write/read routers (MeSH, arXiv 2510.07739; SpiralFormer app. B)
                                   # "block": Block-Parcae, the SERIES hierarchy (MegaByte / Block Transformer
                                   #   shape). One chunk level, offset 0, shift = size: chunk c's tokens receive
                                   #   the loop state of chunk c - 1. The coda reads the token EMBEDDINGS + that
                                   #   update (never h0) and attends only inside its own chunk, so the loop output
                                   #   is the coda's only route to anything before the chunk.
    block_memory: bool = False     # block topology only. False: today's block path, byte-identical. True: the coda
                                   #   ALSO attends over the loop states of all strictly earlier chunks as a memory
                                   #   (m_j = RMSNorm(C(x_j)), one position per chunk), so the loop's channel to the
                                   #   past is a set of vectors, not one. Still a series hierarchy: the coda sees
                                   #   no earlier TOKEN, only loop states. Not the same function as off.
    cycle: CycleSpec | None = None  # None: the hierarchy is descended once (today's code). A CycleSpec repeats it as an outer
                                   #   cycle; anchor topology only, and no sampled-depth level (the cycle count is the one
                                   #   sampled-depth source, so `depth=` sets it)

    def __post_init__(self):
        if not self.levels:
            raise ValueError("a spiral needs at least one level")
        if sum(l.kind == "span" for l in self.levels) > 1:
            raise ValueError("at most one span level (the span layout is one grouping of the row)")
        if sum(l.kind == "span" or l.depth == "poisson" for l in self.levels) > 1:
            raise ValueError("at most one sampled-depth level (a span level or a poisson chunk level): "
                             "`depth=` sets the passes of exactly one level")
        if self.max_spans < 2 or self.max_depth < 1:
            raise ValueError("max_spans must be >= 2 and max_depth >= 1")
        if self.topology not in ("anchor", "mesh", "block"):
            raise ValueError(f"unknown topology {self.topology!r} (anchor | mesh | block)")
        if self.block_memory and self.topology != "block":
            raise ValueError(f"block_memory needs topology 'block' (got {self.topology!r})")
        if self.cycle is not None:
            if self.topology != "anchor":
                raise ValueError(f"a cycle needs the anchor topology (got {self.topology!r}): the mesh buffer and the block "
                                 f"coda have no defined carried state across cycles")
            if any(l.kind == "span" or l.depth == "poisson" for l in self.levels):
                raise ValueError("a cycle block cannot hold a span level or a poisson level: the cycle count is the one "
                                 "sampled-depth source")
        if self.topology == "block":
            if len(self.levels) != 1 or self.levels[0].kind != "chunk":
                raise ValueError("a block topology takes exactly one chunk level")
            lv = self.levels[0]
            if lv.offset != 0 or lv.shift != lv.size:
                raise ValueError(f"a block topology needs offset 0 and shift == size (chunk c reads the state of "
                                 f"chunk c - 1); got offset {lv.offset}, shift {lv.shift}, size {lv.size}")

    @property
    def n_slots(self) -> int:
        """MeSH's rule B = N_loop + 3, with the levels as its iterations."""
        return len(self.levels) + 3

    @staticmethod
    def from_cfg(c) -> "SpiralConfig":
        """From the `arm.spiral` node (a DictConfig or dict)."""
        from omegaconf import OmegaConf
        d = OmegaConf.to_container(c, resolve=True) if not isinstance(c, dict) else dict(c)
        levels = tuple(LevelSpec(**l) for l in d.pop("levels"))
        cycle = d.pop("cycle", None)
        return SpiralConfig(levels=levels, cycle=None if cycle is None else CycleSpec(**cycle), **d)


# ---------------------------------------------------------------- the span layout (host side)
@dataclass
class SpanLayout:
    """Span grouping of a batch of plain rows, built on the HOST from the token ids (the boundary rule
    is a numpy state machine). `group[b, i]`: the span whose pooled state token i feeds (-1 if none);
    `recv[b, i]`: the span whose pooled state token i receives (-1 if none), always the span BEFORE
    the token's own."""
    group: Tensor                  # [B, L] long
    recv: Tensor                   # [B, L] long
    n_src: Tensor                  # [B] long, spans pooled (<= max_spans)
    n_bound: Tensor                # [B] long, boundaries the rule found in the row (> n_src: overflow)

    def to(self, device, non_blocking: bool = False) -> "SpanLayout":
        return SpanLayout(*(t.to(device, non_blocking=non_blocking) for t in
                            (self.group, self.recv, self.n_src, self.n_bound)))


def span_layout(ids: Tensor, rule, max_spans: int) -> SpanLayout:
    """`ids` [B, L] (any device; a CUDA tensor costs one sync). Each row is cut independently from
    span_len 0, the convention of MORPH's `span_ids_from_ids` and packer. The rule is causal, so
    perturbing token t changes no boundary before t."""
    arr = ids.detach().cpu().numpy()
    B, L = arr.shape
    pos = np.arange(L)
    group = np.empty((B, L), dtype=np.int64)
    recv = np.empty((B, L), dtype=np.int64)
    n_src = np.empty(B, dtype=np.int64)
    n_bound = np.empty(B, dtype=np.int64)
    for b in range(B):
        bpos, _ = rule.cut(np.ascontiguousarray(arr[b]), 0)
        nb = int(bpos.shape[0])
        ns = min(nb, max_spans)
        s = np.searchsorted(bpos, pos, side="left")           # boundaries strictly before token i
        group[b] = np.where(s < ns, s, -1)                    # i <= p_s <=> s < nb
        recv[b] = np.minimum(s, ns) - 1                       # the previous span; the tail takes the last
        n_src[b], n_bound[b] = ns, nb
    return SpanLayout(torch.from_numpy(group), torch.from_numpy(recv),
                      torch.from_numpy(n_src), torch.from_numpy(n_bound))


def chunk_geometry(L: int, spec: LevelSpec, device) -> tuple[Tensor, Tensor, int]:
    """-> (group [L], recv [L], G) for a fixed-chunk level. Token i is in chunk (i + omega) // size
    (chunks past G = L // size are dropped, eq. 14); token i receives the chunk of token i - shift."""
    G = L // spec.size
    if G < 1:
        raise ValueError(f"row length {L} is shorter than one chunk of {spec.size}")
    idx = torch.arange(L, device=device)
    group = (idx + spec.offset) // spec.size
    group = torch.where(group < G, group, torch.full_like(group, -1))
    recv = torch.full_like(group, -1)
    if spec.shift < L:
        recv[spec.shift:] = group[: L - spec.shift]           # u~[i] = u[i - s], zero for i < s
    return group, recv, G


# ---------------------------------------------------------------- pool and broadcast
def pool_groups(feat: Tensor, score: Tensor, flat: Tensor, n_flat: int) -> tuple[Tensor, Tensor]:
    """Softmax-weighted sum of `feat` [B, L, C] within each group (eq. 17-18). `flat` [B, L] is the
    flattened group index b * G + g, or n_flat for a token in no group. -> (pooled [n_flat, C],
    member count [n_flat]); an empty group pools to zero. fp32 throughout."""
    idx = flat.reshape(-1)
    s = score.float().reshape(-1)
    with torch.no_grad():                                     # softmax is shift invariant
        mx = torch.full((n_flat + 1,), float("-inf"), device=s.device).scatter_reduce(
            0, idx, s, reduce="amax", include_self=True)
    e = torch.exp(s - mx[idx])
    den = torch.zeros(n_flat + 1, device=s.device).index_add(0, idx, e)
    alpha = e / den[idx]
    C = feat.shape[-1]
    z = torch.zeros(n_flat + 1, C, device=s.device).index_add(0, idx, alpha[:, None] * feat.reshape(-1, C).float())
    cnt = torch.zeros(n_flat + 1, device=s.device).index_add(0, idx, torch.ones_like(s))
    return z[:n_flat], cnt[:n_flat]


def flat_index(group: Tensor, B: int, G: int) -> Tensor:
    """[B, L] group ids (-1 = none) -> flattened b * G + g, with B * G as the dump bin."""
    boff = torch.arange(B, device=group.device)[:, None] * G
    return torch.where(group >= 0, boff + group, torch.full_like(group, B * G))


# ---------------------------------------------------------------- MeSH topology
class Router(nn.Module):
    """Token-wise softmax over the B buffer slots of a state: Linear(d, B) then softmax (MeSH eq. 3).
    Stored as B 1-D vectors plus a bias so that MuonAdamW routes every tensor to AdamW (a [B, d]
    matrix would go to Muon). Zero init: every slot gets weight 1/B."""

    def __init__(self, d: int, n_slots: int):
        super().__init__()
        self.w = nn.ParameterList(nn.Parameter(torch.zeros(d)) for _ in range(n_slots))
        self.b = nn.Parameter(torch.zeros(n_slots))

    def forward(self, h: Tensor) -> Tensor:                   # [B, L, d] -> [S, B, L, 1]
        logits = h.float() @ torch.stack(list(self.w)).float().T + self.b.float()
        return torch.softmax(logits, dim=-1).permute(2, 0, 1).unsqueeze(-1)


class MeshBuffer(nn.Module):
    """MeSH write/read routers: one pair for the transitional cycle (index 0) and one per level
    (index 1 + level). The buffer is a tensor [S, B, L, d]; slot 0 starts as the token embeddings."""

    def __init__(self, d: int, n_slots: int, n_levels: int):
        super().__init__()
        self.n_slots = n_slots
        self.write = nn.ModuleList(Router(d, n_slots) for _ in range(n_levels + 1))
        self.read = nn.ModuleList(Router(d, n_slots) for _ in range(n_levels + 1))

    def write_read(self, idx: int, m: Tensor, router_in: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """m_b += value * w_write_b(router_in); h = sum_b m_b * w_read_b(router_in)."""
        m = m + value.float().unsqueeze(0) * self.write[idx](router_in)
        return m, (m * self.read[idx](router_in)).sum(0).to(value.dtype)

    def start(self, emb: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-loop buffer (m0 = embeddings, rest zero) and the transitional write-read of the prelude
        output v. MeSH Algorithm 2 routes from the prelude output (SpiralFormer appendix B writes x);
        this follows Algorithm 2. -> (buffer, first loop state h)."""
        m = torch.zeros(self.n_slots, *emb.shape, device=emb.device, dtype=torch.float32)
        m = torch.cat([emb.float().unsqueeze(0), m[1:]])
        return self.write_read(0, m, v, v)

    def step(self, level: int, m: Tensor, h: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Level `level`: routers read the level's input state h; the shifted update u is written."""
        return self.write_read(1 + level, m, h, u)


# ---------------------------------------------------------------- the model
class SpiralParcae(Parcae):
    """Plain Parcae (same blocks, init, value embeddings, tied head) with the loop replaced by the
    spiral. Built exactly like the plain arm, so the shared parameters have the plain arm's init;
    the per-level pool scorers are created last. Wrap in `PlainPointer` for the pointer head."""

    def __init__(self, config, spiral: SpiralConfig, rule, **kw):
        super().__init__(config, **kw)
        if config.recurrent_iteration_method != "per-sequence":
            raise ValueError("the sampled-depth level's law is the per-sequence sampler; got "
                             f"{config.recurrent_iteration_method!r}")
        if config.row_compact or self.gradient_checkpointing:
            raise ValueError("spiral arms run their own loop: row_compact and gradient checkpointing are off")
        if self.has_span_level_cfg(spiral) and spiral.max_spans > config.block_size:
            raise ValueError(f"max_spans {spiral.max_spans} exceeds block_size {config.block_size} (RoPE table)")
        self.spiral = spiral
        self.rule = rule                                      # a numpy BoundaryRule, not a module
        self.pool_w = nn.ParameterList(                       # 1-D: MuonAdamW routes it to AdamW
            nn.Parameter(torch.zeros(config.n_embd)) for _ in spiral.levels)
        self.up_norm = nn.ModuleList(                         # per-level RMSNorm on C(zh), see Up above
            config.Norm(config.n_embd, eps=config.norm_eps) for _ in spiral.levels)
        if spiral.block_memory:
            use_flex(self.transformer.coda)                   # same parameters; the memory mask needs the flex kernel
        self.mesh = (MeshBuffer(config.n_embd, spiral.n_slots, len(spiral.levels))
                     if spiral.topology == "mesh" else None)

    @staticmethod
    def has_span_level_cfg(spiral: SpiralConfig) -> bool:
        return any(l.kind == "span" for l in spiral.levels)

    @property
    def has_span_level(self) -> bool:
        """The model needs a span layout (the host-side boundary cut)."""
        return any(l.kind == "span" for l in self.spiral.levels)

    @property
    def has_sampled_depth(self) -> bool:
        """The model has a K-curve: a span level or a poisson chunk level, whose passes `depth=` sets."""
        return self.spiral.cycle is not None or any(l.kind == "span" or l.depth == "poisson" for l in self.spiral.levels)

    @property
    def cycle_deep_train(self) -> bool:
        """Deep supervision is live: a cycle block with credit deep, in training mode (eval scores the last cycle)."""
        return self.spiral.cycle is not None and self.spiral.cycle.credit == "deep" and self.training

    def span_layout(self, ids: Tensor) -> SpanLayout:
        return span_layout(ids, self.rule, self.spiral.max_spans)

    # -- pieces -----------------------------------------------------------------------------------
    def _core_pass(self, x: Tensor, z: Tensor, ves: list, freqs: Tensor) -> Tensor:
        x = self.transformer.adapter(x, z)
        for li, block in enumerate(self.transformer.core_block):
            x = block(x, freqs, None, ve=ves[li])
        return x

    def _depths(self, B: int, depth: int | None) -> tuple[Tensor, Tensor]:
        """Per-row (no-grad, grad) pass counts of the sampled-depth level, on the CPU. A given `depth` runs that
        many passes in every row. None: training samples Parcae's law (n + k = Poisson total, k <= BPTT
        depth) truncated at `max_depth`; eval uses the mean recurrence."""
        if depth is not None:
            return torch.zeros(B, dtype=torch.long), torch.full((B,), int(depth), dtype=torch.long)
        if not self.training:
            return torch.zeros(B, dtype=torch.long), torch.full((B,), int(self.config.mean_recurrence), dtype=torch.long)
        n, k = self._sample_batch_depths(B, None)
        total = (n + k).clamp(max=self.spiral.max_depth)
        k = total.clamp(max=self.config.mean_backprop_depth)
        return total - k, k

    def _loop(self, z: Tensor, ves: list, freqs: Tensor, n: Tensor, k: Tensor) -> Tensor:
        """Parcae's per-sequence loop on the group sequence: n no-grad passes, then k with grad, each row
        stopping at its own count (finished rows keep their state)."""
        x = self.initialize_state(z)
        dev = z.device
        for steps, grad in ((n, False), (k, True)):
            top = int(steps.max())
            uniform = bool((steps == top).all())
            steps_dev = steps.to(dev)
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                for step in range(top):
                    x_new = self._core_pass(x, z, ves, freqs)
                    x = x_new if uniform else torch.where((step < steps_dev)[:, None, None], x_new, x)
        return x

    def _broadcast(self, out: Tensor, gain: Tensor, recv_flat: Tensor, B: int, L: int) -> Tensor:
        """[B, G, d] group states -> [B, L, d] token updates: token i takes the row of group recv(i)
        (zero if none) times that group's gain."""
        G, d = out.shape[1], out.shape[2]
        table = torch.cat([(out * gain[..., None].to(out.dtype)).reshape(B * G, d), out.new_zeros(1, d)])
        return table[recv_flat.reshape(-1)].view(B, L, d)

    def _block_coda(self, x: Tensor, input_ids: Tensor, coda_off: int) -> Tensor:
        """The block topology's local decoder: the coda blocks run on each chunk as its own sequence
        ([B, L, d] -> [B * L / g, g, d], causal inside the chunk, RoPE positions 0..g-1), then fold back."""
        B, L, d = x.shape
        g = self.spiral.levels[0].size
        if L % g:
            raise ValueError(f"block topology needs the row length {L} to be a multiple of the chunk size {g}")
        x = x.reshape(B * L // g, g, d)
        freqs = self.freqs_cis[:, :g]
        for i, block in enumerate(self.transformer.coda):
            ve = None
            if str(coda_off + i) in self.value_embeds:
                ve = self.value_embeds[str(coda_off + i)](input_ids)
                ve = ve.reshape(B * L // g, g, ve.shape[-1])
            x = block(x, freqs, None, ve=ve)
        return x.reshape(B, L, d)

    def _block_memory_coda(self, x: Tensor, mem: Tensor, input_ids: Tensor, coda_off: int) -> Tensor:
        """Block topology with `block_memory`: the coda runs once over [memory (G positions) ; tokens (L)] under
        `block_memory_mask`. x [B, L, d] is embedding + previous chunk's update, mem [B, G, d] the per-chunk loop
        states. RoPE: token i at i, memory m_j at (j + 1) * g, the first position of the chunk that may read it
        (the last memory is read by no token; its position is clamped to the table, which changes nothing).
        Memory positions get a ZERO value embedding (a value embedding is a function of a token id and a memory
        position has none; zero leaves v unchanged). Returns the token positions only, [B, L, d]."""
        B, L, d = x.shape
        G = mem.shape[1]
        g = self.spiral.levels[0].size
        if L % g:
            raise ValueError(f"block topology needs the row length {L} to be a multiple of the chunk size {g}")
        mpos = ((torch.arange(G, device=x.device) + 1) * g).clamp(max=self.freqs_cis.shape[1] - 1)
        freqs = torch.cat([self.freqs_cis[:, mpos], self.freqs_cis[:, :L]], dim=1)
        mask = block_memory_mask(G, L, g, x.device)
        h = torch.cat([mem.to(x.dtype), x], dim=1)
        for i, block in enumerate(self.transformer.coda):
            ve = None
            if str(coda_off + i) in self.value_embeds:
                ve = self.value_embeds[str(coda_off + i)](input_ids)
                ve = torch.cat([ve.new_zeros(B, G, ve.shape[-1]), ve], dim=1)
            h = block(h, freqs, mask, ve=ve)
        return h[:, G:]

    # -- forward ----------------------------------------------------------------------------------
    def _level_update(self, lvl: int, spec: LevelSpec, h: Tensor, tok_ve: list, ve_layers: list, B: int, L: int,
                      depth: int | None, layout: SpanLayout | None) -> tuple[Tensor, Tensor]:
        """One level: pool the state h [B, L, d] into groups, run the level's core passes on the group sequence,
        broadcast the normed read-out back to tokens. -> (u [B, L, d] token update, upd [B, G, d] group states)."""
        cfg, tr = self.config, self.transformer
        d, kv = cfg.n_embd, tok_ve[0].shape[-1] if tok_ve else 0
        if spec.kind == "chunk":
            grp, rcv, G = chunk_geometry(L, spec, h.device)
            grp, rcv = grp.expand(B, L), rcv.expand(B, L)
        else:
            grp, rcv, G = layout.group, layout.recv, self.spiral.max_spans
        flat = flat_index(grp, B, G)
        feat = torch.cat([h, *tok_ve], dim=-1)
        score = (h.float() * self.pool_w[lvl].float()).sum(-1)
        pooled, cnt = pool_groups(feat, score, flat, B * G)
        pooled = pooled.view(B, G, -1)
        z = pooled[..., :d].to(h.dtype)
        ves = [None] * cfg.n_layers_in_recurrent_block
        for j, li in enumerate(ve_layers):
            ves[li] = pooled[..., d + j * kv: d + (j + 1) * kv].to(h.dtype)
        gfreqs = self.freqs_cis[:, :G]

        if spec.kind == "chunk" and spec.depth == "poisson":
            n, k = self._depths(B, depth)
            x = self._loop(z, ves, gfreqs, n, k)
            gain = torch.full((B, G), 1.0 / math.sqrt(spec.size), device=h.device)
        elif spec.kind == "chunk":
            x = self.initialize_state(z)
            for _ in range(spec.passes):
                x = self._core_pass(x, z, ves, gfreqs)
            gain = torch.full((B, G), 1.0 / math.sqrt(spec.size), device=h.device)
        else:
            n, k = self._depths(B, depth)
            x = self._loop(z, ves, gfreqs, n, k)
            c = cnt.view(B, G)
            gain = torch.where(c > 0, c.clamp_min(1).rsqrt(), torch.zeros_like(c))   # 1/sqrt(span length); 0 if padded
        rflat = flat_index(rcv, B, G)
        upd = self.up_norm[lvl](tr.C(x))
        return self._broadcast(upd, gain, rflat, B, L), upd

    def _coda_blocks(self, x: Tensor, input_ids: Tensor, freqs: Tensor) -> Tensor:
        """The coda blocks over the full row (anchor and mesh topologies)."""
        coda_off = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i, block in enumerate(self.transformer.coda):
            ve = self.value_embeds[str(coda_off + i)](input_ids) if str(coda_off + i) in self.value_embeds else None
            x = block(x, freqs, None, ve=ve)
        return x

    def logits_from(self, x: Tensor) -> Tensor:
        """The tied head on the final-norm output x (see `plain_deep.head_logits`)."""
        return head_logits(self, x)

    def forward(self, input_ids: Tensor, attention_mask=None, position_ids=None, labels=None,
                return_logits: bool = False, num_steps_pair=None, depth: int | None = None,
                layout: SpanLayout | None = None, head_fn=None, backward_scale: Tensor | None = None) -> dict:
        if labels is not None:
            raise ValueError("SpiralParcae returns logits only: the loss lives in PlainPointer")
        if attention_mask is not None or position_ids is not None or num_steps_pair is not None:
            raise ValueError("spiral arms take no attention_mask, position_ids or num_steps_pair "
                             "(depth= sets the sampled-depth level's passes)")
        if depth is not None and not self.has_sampled_depth:
            raise ValueError("this spiral has no sampled-depth level (a span level or a poisson chunk level): "
                             "its depth is fixed (chunk passes), there is no K-curve; call without depth")
        if head_fn is not None and not self.cycle_deep_train:
            raise ValueError("head_fn is the per-cycle head of deep supervision: it needs spiral.cycle.credit 'deep' "
                             "in training mode")
        if backward_scale is not None and (head_fn is None or not torch.is_grad_enabled()):
            raise ValueError("backward_scale (per-cycle backward) needs head_fn and grad enabled")
        if self.cycle_deep_train and head_fn is None:
            raise ValueError("spiral.cycle.credit 'deep' in training needs a per-cycle head (head_fn): wrap the model "
                             "in PlainPointer, whose loss supplies it")
        cfg = self.config
        B, L = input_ids.shape
        freqs = self.freqs_cis[:, :L]
        if self.has_span_level and layout is None:
            layout = self.span_layout(input_ids)              # host sync; the trainer passes it in
        tr = self.transformer

        e = tr.wte(input_ids)
        if self.emb_scale != 1:
            e = e * self.emb_scale
        emb0 = e                                              # MeSH slot 0: the embedding the prelude reads
        for i, block in enumerate(tr.prelude):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            e = block(e, freqs, None, ve=ve)
        h0 = tr.ln_prelude(e) if cfg.prelude_norm else e

        core_off = cfg.n_layers_in_prelude
        ve_layers = [li for li in range(cfg.n_layers_in_recurrent_block)
                     if str(core_off + li) in self.value_embeds]
        tok_ve = [self.value_embeds[str(core_off + li)](input_ids) for li in ve_layers]

        if self.spiral.cycle is not None:
            u, out = self._cycles(input_ids, h0, tok_ve, ve_layers, freqs, depth, head_fn, backward_scale)
            if out is not None:                               # deep supervision: the per-cycle NLLs
                return out
        else:
            u = None
            mesh = self.mesh is not None
            h = h0
            if mesh:
                m, h = self.mesh.start(emb0, h0)              # transitional write-read -> first state
            for lvl, spec in enumerate(self.spiral.levels):
                if not mesh and u is not None:
                    h = h0 + u                                # Anchor: the latest update only
                u, upd = self._level_update(lvl, spec, h, tok_ve, ve_layers, B, L, depth, layout)
                if mesh:
                    m, h = self.mesh.step(lvl, m, h, u)       # h was this level's input state

        coda_off = cfg.n_layers_in_prelude + cfg.n_layers_in_recurrent_block
        if self.spiral.block_memory:
            x = self._block_memory_coda(emb0 + u, upd, input_ids, coda_off)
        elif self.spiral.topology == "block":
            x = self._block_coda(emb0 + u, input_ids, coda_off)
        else:
            x = self._coda_blocks(h if self.mesh is not None else h0 + u, input_ids, freqs)
        x = tr.ln_f(x)

        logits = self.logits_from(x) if return_logits else None
        return {"loss": torch.as_tensor(0.0), "logits": logits, "log_ppl": torch.as_tensor(0.0)}

    # -- cycles -----------------------------------------------------------------------------------
    def _cycle_plan(self, B: int, depth: int | None) -> "CyclePlan":
        """The cycle schedule of a batch, on the CPU (no device sync). Count per ROW: given `depth` -> that many in
        every row (all with grad); eval -> round(mean); training -> Parcae's law with the Poisson mean `mean`
        (floor 1, truncated at `max`; the last `bptt` cycles carry gradient, the first n run no-grad; deep
        supervision puts every cycle on the grad side). Start level and skips are drawn only in training."""
        cy, nL = self.spiral.cycle, len(self.spiral.levels)
        gen = torch.Generator(device="cpu")                   # per-step seed like Parcae's sampler, own stream
        gen.manual_seed((514229 + 7919 + int(getattr(self, "step", 0))) % (2 ** 31 - 1))
        if depth is not None:
            if depth < 1:
                raise ValueError(f"depth is a cycle count >= 1, got {depth}")
            total = torch.full((B,), int(depth), dtype=torch.long)
        elif not self.training:
            total = torch.full((B,), max(1, round(cy.mean)), dtype=torch.long)
        else:
            total = torch.poisson(torch.full((B,), float(cy.mean)), generator=gen).to(torch.long).clamp(1, cy.max)
        if depth is None and self.training and cy.credit == "end":
            k = total.clamp(max=cy.bptt if cy.bptt > 0 else self.config.mean_backprop_depth)
        else:
            k = total                                         # depth=, eval and deep supervision: every cycle on the grad side
        n = total - k
        Cn, Ck = int(n.max()), int(k.max())
        C = Cn + Ck
        g = torch.arange(C)[:, None]
        active = torch.where(g < Cn, g < n[None], (g - Cn) < k[None])             # [C, B]
        if self.training and cy.start == "uniform":
            start = torch.randint(0, nL, (C, B), generator=gen)
        else:
            start = torch.zeros(C, B, dtype=torch.long)
        if self.training and cy.skip_p > 0:
            skip = torch.rand((C, B, nL), generator=gen) < cy.skip_p
        else:
            skip = torch.zeros(C, B, nL, dtype=torch.bool)
        lv = torch.arange(nL)
        run = active[..., None] & (lv >= start[..., None]) & ((lv == start[..., None]) | ~skip)
        return CyclePlan(n=n, k=k, active=active, start=start, skip=skip, run=run)

    def _cycles(self, input_ids: Tensor, h0: Tensor, tok_ve: list, ve_layers: list, freqs: Tensor,
                depth: int | None, head_fn, backward_scale: Tensor | None = None) -> tuple[Tensor, dict | None]:
        """The cycle loop -> (the final token update u, None), or, with deep supervision, (u, the output dict).
        With `backward_scale` each cycle's weighted CE, sum(weight * nll) / backward_scale, is backpropagated right after
        the cycle, so a cycle's graph is freed before the next one (memory of one cycle). h0 and the core value
        embeddings are leaves during the loop; their summed gradient flows into the prelude and the embeddings after the
        last cycle. The total gradient equals one backward through the sum of the cycle losses.
        Which levels run in a cycle is known on the host (the plan is CPU data); rows that are finished, or skip a
        level, keep their update through a `where` on a device copy of the plan, so every level keeps its static
        shape and the compiled blocks see no new shapes. The schedule itself runs in eager Python."""
        B, L = input_ids.shape
        levels = self.spiral.levels
        plan = self._cycle_plan(B, depth)
        run_dev = plan.run.to(h0.device, non_blocking=True)
        any_run, all_run = plan.run.any(1).tolist(), plan.run.all(1).tolist()         # [C][nL] host bools
        deep = head_fn is not None
        total = (plan.n + plan.k).to(torch.float32)
        weight = (plan.active.float() / total[None]).to(h0.device, non_blocking=True)  # row b: 1 / (its cycle count)
        leaves = None
        if backward_scale is not None:
            leaves = [h0, *tok_ve]
            h0 = h0.detach().requires_grad_()
            tok_ve = [t.detach().requires_grad_() for t in tok_ve]
        u, nlls = None, []
        Cn = int(plan.n.max())
        for steps, grad, off in ((int(plan.n.max()), False, 0), (int(plan.k.max()), True, Cn)):
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                for c in range(steps):
                    g = off + c
                    for lvl, spec in enumerate(levels):
                        if not any_run[g][lvl]:
                            continue
                        h = h0 if u is None else h0 + u
                        u_new, _ = self._level_update(lvl, spec, h, tok_ve, ve_layers, B, L, None, None)
                        if all_run[g][lvl]:
                            u = u_new
                        else:
                            keep = u_new.new_zeros(()) if u is None else u.to(u_new.dtype)
                            u = torch.where(run_dev[g, :, lvl][:, None, None], u_new, keep)
                    if deep:
                        x = self.transformer.ln_f(self._coda_blocks(h0 + u, input_ids, freqs))
                        nll = head_fn(x)
                        if backward_scale is not None:
                            (weight[g][:, None] * nll).sum().div(backward_scale).backward()
                            nll = nll.detach()
                        nlls.append(nll)
                        u = self._carry(u)
        if not deep:
            return u, None
        if backward_scale is not None:
            new = [h0, *tok_ve]
            pairs = [(a, b.grad) for a, b in zip(leaves, new) if b.grad is not None]
            torch.autograd.backward([a for a, _ in pairs], [g for _, g in pairs])
        return u, {"loss": torch.as_tensor(0.0), "logits": None, "log_ppl": torch.as_tensor(0.0),
                   "cycle_nll": nlls, "cycle_weight": weight, "cycle_active": plan.active.to(h0.device, non_blocking=True)}

    @staticmethod
    def _carry(u: Tensor) -> Tensor:
        """Deep supervision: the state carried into the next cycle is DETACHED, so each cycle's CE trains that cycle
        (and the prelude through h0) only."""
        return u.detach()


@dataclass
class CyclePlan:
    """CPU schedule of one batch. n, k [B]: no-grad and grad cycles per row. active [C, B]: row b runs cycle g (the
    first n_b slots of the no-grad phase, the first k_b of the grad phase). start [C, B], skip [C, B, nL]: the draws.
    run [C, B, nL]: row b runs level l in cycle g (active, at or after its start, start never skipped)."""
    n: Tensor
    k: Tensor
    active: Tensor
    start: Tensor
    skip: Tensor
    run: Tensor
