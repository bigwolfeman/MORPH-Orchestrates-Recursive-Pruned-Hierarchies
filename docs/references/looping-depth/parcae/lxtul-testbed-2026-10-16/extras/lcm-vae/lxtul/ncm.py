"""Span-level Next-Concept Modeling (NCM, arXiv 2610.14162) on a Parcae backbone.

The units are the spans the TUL packer cuts (same boundary rule, same one-cell rows as the minimal arm). A frozen
multiscale residual VQ tokenizer turns each span into a coarse-to-fine code plan; a generator predicts the plan
level by level from the EARLIER text; a token model reads the plan of the span it is predicting.

  Stage 1  `NCMTokenizer` (arm kind `ncm_tok`). E: token embedding + `enc_layers` Parcae blocks, span-local
           BIDIRECTIONAL attention, norm, Linear(d, dz): one latent z per token (RMS-normalised, see `z_norm`).
           Levels `levels` (1, 2, 4, 8, 16, 32): a span of n tokens has l' = min(l_k, n) segments at level k,
           token j sits in segment floor(j * l' / n). Down_k = segment mean of the residual z - zhat^(k-1);
           the nearest entry of the level's codebook is the code; the reconstruction is the NEAREST UPSAMPLE of
           the code vectors (each token takes its segment's vector); zhat^(k) = zhat^(k-1) + up(e_k). No learned
           map sits inside the residual chain. Codebooks: EMA updates (decay `ema_decay`), dead-code restart (a
           code unused for `restart_after` steps becomes a random current residual vector of its level),
           commitment `beta`, straight-through estimator. D: Linear(dz, d) + `dec_layers` span-local bidirectional
           blocks + norm + a tied output embedding. Nested dropout (paper eq. 9): with probability
           `nested_dropout` the cutoff is uniform in {1..K-1}, else K; recon CE and the commitment term are taken
           at that cutoff. All of it is vectorised over spans (a Python loop runs over the K levels only).
           `ncm.tok_objective: repr` (RePR, arXiv 2610.20533) swaps E, D and the reconstruction loss for a prediction
           trunk (see "Stage 1, repr" below); the default `recon` is everything described above, unchanged.
  Stage 2  `NCMModel` (arm kind `ncm`), tokenizer frozen (`ncm.tok_ckpt`, a stage-1 run dir).
           Token path: Parcae blocks as the gpt8 shape (prelude, coda), the SAME PointerHead as gpt8_pointer when `pointer_heads` > 0.
           The plan enters at the CODA INPUT only: the position whose label is token j of complete span u >= 1
           adds P(zhat_u^(m_u)[j]) + E_m[m_u] (P zero-initialised; E_m zero-initialised too, so at init every m
           scores the same as m = 0). The prelude never sees a plan.
           Generator: per complete span, level blocks of min(l_k, n_u) positions for k <= `gen_levels`, input
           Down_k(zhat_u^(k-1)) built from the TRUE codes (each code replaced by a uniform random code of its
           level at the rate `corrupt[k-1]` in training, clean at eval) -> Linear(dz, d) + level + position
           embedding. Level 1 is embeddings only. The positions share the token prelude (one sequence, one
           mask), then Parcae's own core block loops over the generator positions (Parcae's injection, state
           init and per-row Poisson depth sampler; `depth` is the K-sweep argument). Each core layer's attention
           reads q = generator states and kv = [token prelude states projected by THAT layer's K/V weights,
           once per forward; current generator states]. Heads: one Linear(d, codebook_size) per level.
           A level block reads itself (bidirectional), the generator positions of EARLIER spans (all levels),
           and iff `gen_reads_tokens` the plan-free prelude states of the tokens of EARLIER spans. It never
           reads a token of its own span. Without token memory (`gen_reads_tokens` false) every complete span
           also gets ONE summary position, input = Linear(dz, d) of the mean of zhat_u^(gen_levels) over the
           span's tokens (Down_1), readable only by LATER spans; span 0 has that summary position and no
           level positions (its codes are given context, never charged, as the LCM reference's first concept).
           Value embeddings and the output head are GPT's (gpt.py): embeddings on token layers i with
           i % 2 == (n_layer - 1) % 2, n_layer = prelude + coda, read at token positions only (zero at generator
           positions; in the pure arm they follow the shifted input and are zero at a span-start BOS); logits =
           tied wte, no logit scale, plain cross entropy.
           `coda_span_local` makes every token-path attention span-local causal and, so that a span's first
           token is predicted INSIDE the span, shifts the token input right by one with a learned BOS at each
           span start (position t of a span predicts token t of the span); the pointer head is refused there.
  Loss     (sum of token NLL + sum of code NLL over all gen levels) / number of scored token labels: the bound
           -log p(x) <= -log p(codes) - log p(x | codes), minimised directly (NCM appendix A).
  Eval     `depth_sweep` returns per-row ELBO sums per generator depth and per m in {0..gen_levels}: token NLL
           given zhat^(m) (the coda reads the TRUE codes) + code NLL of levels <= m. `sums` is m = gen_levels.
           Plan arm (`coda_span_local` false): the scored set is every token label of the packed row (span 0 and
           the open tail run at m = 0 with no code charge; the bound holds), the same tokens `evaluate.plain_view`
           scores. Pure arm: the target tokens of complete spans with index >= 1, the LCM reference's set.

Stage 1, repr (`ncm.tok_objective: repr`, arm `ncm_repr_tok`). The span code holds what a span shares with its CONTEXT, not its
surface form. P = `enc_layers` causal Parcae blocks over the compact row, run twice with shared weights: the FULL view (causal,
same document only; a document ends at `ncm.eos_id`) and the LOCAL view (causal inside the input token's own span). Both end in
the LM head tied to the token embedding, slot logit masked. For a label position p (input token x_p in span s):
  PMI_p = stopgrad(nll_local(p) - nll_full(p)), clamped to +-`pmi_clip`.
Heads on the trunk states (no stopgrad): R_loc(p) = Linear(h_local(p)), R_ctx(s) = Linear(h_full at the last input position
BEFORE span s), width `repr_dim` (default dz). R_loc at the position before a span start is 0, so the increments
R_loc(p) - R_loc(p-1) telescope to R_loc(last position of the span). Loss = mean(CE_full, CE_local)
+ `repr_weight` * mean((<R_ctx(s), R_loc(p) - R_loc(p-1)> - PMI_p)^2) (RePR eq. 15) over the scored positions: complete spans
>= 1 whose first token has an earlier token of the same document. The span code z(s) = R_loc at the span's last input position
depends on the tokens of span s ONLY (local mask). It is detached, broadcast to the span's tokens and quantised by the same
residual VQ with `levels` [1, 1, 1, 1] (four residual stages on one vector per span). No reconstruction decoder, no commitment term
(the VQ fits features that move, it must not pull R), no z_norm (the dot product's scale carries the PMI). Stage 2 reads codes
from the LOCAL pass only (`encode`). Readouts: CE full, CE local, mean PMI per scored token, R^2 of the repr prediction
(1 = perfect, 0 = the mean), per-level code use and perplexity.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import lxtul.attention as A
from lxtul import ce
from lxtul.copy_heads import PointerHead, mix_pointer
from lxtul.lcm import Compact, _compile, _freeze_ve, compact, rows_to
from parcae_lm.modules.mixer import has_ve
from parcae_lm.modules.utils import precompute_freqs_cis

NCM_KINDS = ("ncm_tok", "ncm")
TOK_ARCH_KEYS = ("tok_objective", "cells", "enc_layers", "dec_layers", "dz", "levels", "codebook_size", "z_norm", "repr_dim")
TOK_OBJECTIVES = ("recon", "repr")


@dataclass(frozen=True)
class NCMConfig:
    """Every NCM knob (`arm.ncm` in the Hydra arm files). Tokenizer keys first, stage-2 keys after."""
    slot_id: int                     # the <fim_pad> id of the packed row; its logit is masked in every CE
    max_slots: int = 64              # spans per row (the packer's budget)
    cells: int = 1                   # prefix_k of the rows: 1 = the minimal-TUL rows; the models never read a slot position
    # --- tokenizer (stage 1; stage 2 must carry the same values, load_tokenizer refuses otherwise) ---
    enc_layers: int = 2
    dec_layers: int = 2
    dz: int = 256                    # per-token latent width
    levels: tuple = (1, 2, 4, 8, 16, 32)
    codebook_size: int = 4096
    ema_decay: float = 0.99
    ema_eps: float = 1e-5            # Laplace smoothing of the EMA counts
    restart_after: int = 256         # a code unused this many tokenizer steps is restarted
    beta: float = 0.25               # commitment weight
    nested_dropout: float = 0.5
    z_norm: bool = True              # RMS-normalise z per token: the encoder cannot shrink z to cheat the commitment term
    tok_objective: str = "recon"     # "recon": the reconstruction VQ-VAE; "repr": the RePR PMI-factorising span code (module docstring)
    repr_dim: int | None = None      # repr: width of R_ctx, R_loc and of the codebooks; None = dz
    pmi_clip: float = 8.0            # repr: |PMI_p| is clamped to this many nats
    repr_weight: float = 0.01        # repr: weight of the PMI regression next to the two LM losses
    eos_id: int = -1                 # repr: the document separator of the rows (the packer's rule.eos_id); set by ncm_config
    # --- stage 2 ---
    tok_ckpt: str = ""               # the stage-1 run dir (holds final.pt)
    gen_levels: int = 3
    gen_reads_tokens: bool = True
    coda_span_local: bool = False
    m_train: str = "uniform"         # "uniform": m_u ~ U{0..gen_levels} per span; "fixed": gen_levels
    corrupt: tuple = (0.5, 0.25, 0.125)   # per-level conditioning-code replacement rate (training)
    pointer_heads: int = 4
    min_span: int = 4                # the packer's minimum span. EOS is exempt from it (tul_layout.BoundaryRule), so a complete
                                     # span can be shorter: the layout caps every level at the span length and the number of
                                     # such spans is reported (`short_spans`), never raised on
    eval_seed: int = 0               # seeds the generator state init in every sweep

    def __post_init__(self):
        object.__setattr__(self, "levels", tuple(int(v) for v in self.levels))
        object.__setattr__(self, "corrupt", tuple(float(v) for v in self.corrupt))
        lv = self.levels
        if self.tok_objective not in TOK_OBJECTIVES:
            raise ValueError(f"ncm.tok_objective must be one of {TOK_OBJECTIVES}, got {self.tok_objective!r}")
        if self.tok_objective == "repr":
            # the span code is ONE vector per span: repeated 1s are residual stages on it; any other level has no meaning
            if not lv or any(v != 1 for v in lv):
                raise ValueError(f"ncm.tok_objective repr needs ncm.levels of repeated 1s (residual stages on one span vector), got {lv}")
            if self.eos_id < 0:
                raise ValueError("ncm.tok_objective repr needs ncm.eos_id >= 0 (the packer's document separator)")
            if self.z_norm or self.beta != 0.0 or self.nested_dropout != 0.0:
                raise ValueError("ncm.tok_objective repr has no commitment term, no nested dropout and no z_norm: "
                                 "set beta 0, nested_dropout 0, z_norm false")
            if self.coda_span_local:
                raise ValueError("ncm.tok_objective repr is built for the plan arm: refused with coda_span_local")
            if self.pmi_clip <= 0 or self.repr_weight < 0 or (self.repr_dim is not None and self.repr_dim <= 0):
                raise ValueError("ncm.pmi_clip must be > 0, ncm.repr_weight >= 0, ncm.repr_dim > 0")
        else:
            if self.repr_dim is not None:
                raise ValueError("ncm.repr_dim only exists for ncm.tok_objective repr")
            if not lv or lv[0] != 1 or any(b <= a for a, b in zip(lv, lv[1:])):
                raise ValueError(f"ncm.levels must start at 1 and strictly increase, got {lv}")
        if not 1 <= self.gen_levels <= len(lv):
            raise ValueError(f"ncm.gen_levels must be in 1..{len(lv)}, got {self.gen_levels}")
        if len(self.corrupt) != self.gen_levels or any(not 0.0 <= r <= 1.0 for r in self.corrupt):
            raise ValueError(f"ncm.corrupt needs one rate in [0, 1] per generator level ({self.gen_levels}), got {self.corrupt}")
        if self.m_train not in ("uniform", "fixed"):
            raise ValueError(f"ncm.m_train must be 'uniform' or 'fixed', got {self.m_train!r}")
        if not 0.0 <= self.nested_dropout <= 1.0 or not 0.0 <= self.ema_decay < 1.0 or self.beta < 0:
            raise ValueError("ncm.nested_dropout in [0, 1], ncm.ema_decay in [0, 1), ncm.beta >= 0")
        if self.pointer_heads < 0:
            raise ValueError("ncm.pointer_heads must be >= 0")
        if self.pointer_heads > 0 and self.coda_span_local:
            raise ValueError("ncm.pointer_heads > 0 reads tokens of earlier spans: it is refused with coda_span_local")

    @property
    def zdim(self) -> int:
        """Width of a code vector: the codebooks, the plan and the generator input. dz unless repr sets repr_dim."""
        return int(self.repr_dim) if self.tok_objective == "repr" and self.repr_dim else self.dz


# --------------------------------------------------------------------------------------------- row layout
@dataclass
class Rows:
    """A packed batch with the slot positions removed (tokens first, order kept) and what the models need on it."""
    c: Compact
    y: Tensor        # [B, T] next-token labels in compact order, -100 past the row's tokens
    n_all: Tensor    # [B] tokens in the row (complete spans + open tail)
    start: Tensor    # [B, S] compact index of each span's first token
    pos: Tensor      # [B, T] index of a token inside its span (0 for the open tail and the padding)


def make_rows(x: Tensor, y: Tensor, lay, S: int) -> Rows:
    c = compact(x, lay, S)
    B, T = x.shape
    order = torch.sort(lay.slot_mask.to(torch.uint8), dim=1, stable=True).indices   # the order `compact` used
    n_all = (~lay.slot_mask).sum(1)
    ar = torch.arange(T, device=x.device)
    yc = torch.where(ar[None] < n_all[:, None], y.gather(1, order), torch.full_like(y, -100))
    start = c.n_tok.cumsum(1) - c.n_tok
    pos = torch.where(c.bag < S, ar[None] - F.pad(start, (0, 1)).gather(1, c.bag), torch.zeros_like(c.bag))
    return Rows(c, yc, n_all, start, pos)


@dataclass
class SegInfo:
    """Segment structure of a batch at every level of the tokenizer."""
    levels: tuple
    S: int
    tokok: Tensor        # [B, T] bool: the token belongs to a complete span
    bin: list            # per level [B, T] long: bag * l_k + segment (the dump bag S maps to bin S * l_k)
    nseg: list           # per level [B, S] long: min(l_k, n_u)
    seg_ok: list         # per level [B, S, l_k] bool: the segment exists (complete span, i < nseg)
    cnt: list            # per level [B, (S + 1) * l_k] tokens per bin


def build_segments(c: Compact, levels: tuple, S: int) -> SegInfo:
    B, T = c.x.shape
    n, bag = c.n_tok, c.bag
    start = n.cumsum(1) - n
    ar = torch.arange(T, device=c.x.device)
    p = torch.where(bag < S, ar[None] - F.pad(start, (0, 1)).gather(1, bag), torch.zeros_like(bag))
    nt = F.pad(n, (0, 1), value=1).gather(1, bag).clamp_min(1)
    tokok = bag < S
    bins, nsegs, oks, cnts = [], [], [], []
    for lk in levels:
        lp = nt.clamp(max=lk)                           # l' = min(l_k, n)
        seg = (p * lp) // nt                            # floor(j * l' / n)
        b = bag * lk + seg
        nseg = n.clamp(max=lk)
        ok = c.valid[..., None] & (torch.arange(lk, device=c.x.device) < nseg[..., None])
        cnt = torch.zeros(B, (S + 1) * lk, device=c.x.device).scatter_add_(1, b, tokok.to(torch.get_default_dtype()))
        bins.append(b), nsegs.append(nseg), oks.append(ok), cnts.append(cnt)
    return SegInfo(tuple(levels), S, tokok, bins, nsegs, oks, cnts)


def seg_mean(v: Tensor, seg: SegInfo, k: int) -> Tensor:
    """Down_k: segment means of a per-token [B, T, D] -> [B, S, l_k, D] (0 for a segment that does not exist)."""
    B, T, D = v.shape
    lk, S = seg.levels[k], seg.S
    sums = v.new_zeros(B, (S + 1) * lk, D).scatter_add_(1, seg.bin[k][..., None].expand(-1, -1, D),
                                                         v * seg.tokok[..., None].to(v.dtype))
    return (sums / seg.cnt[k][..., None].to(v.dtype).clamp_min(1)).view(B, S + 1, lk, D)[:, :S]


def upsample(e: Tensor, seg: SegInfo, k: int) -> Tensor:
    """Nearest upsample: [B, S, l_k, D] segment vectors -> [B, T, D], every token takes its segment's vector
    (0 for tokens of no complete span)."""
    B, S, lk, D = e.shape
    ext = torch.cat([e.reshape(B, S * lk, D), e.new_zeros(B, lk, D)], 1)
    out = ext.gather(1, seg.bin[k][..., None].expand(-1, -1, D))
    return out * seg.tokok[..., None].to(out.dtype)


def _doc_mask(doc: Tensor, causal: bool):
    """A position reads the positions of its own span (`doc` [B, T]); `causal`: only the earlier ones."""
    if causal:
        return A.span_pack_mask(doc)
    B, T = doc.shape

    def mod(b, h, q, kv):
        return doc[b, q] == doc[b, kv]
    return A._mask(mod, B, None, T, T, device=doc.device)


def _nll_acc(h: Tensor, W: Tensor, labels: Tensor, scale: float, mask_id: int, chunk: int = 2048):
    """Per-row CE and top-1 correctness of tied-output logits, never holding all [N, V] logits."""
    N = h.shape[0]
    nll, hit = h.new_zeros(N, dtype=torch.float64), h.new_zeros(N, dtype=torch.float64)
    for i in range(0, N, chunk):
        lg = (h[i:i + chunk] @ W.T).float() * scale
        lg[:, mask_id] = float("-inf")
        lab = labels[i:i + chunk]
        pk = lg.gather(1, lab.clamp_min(0)[:, None]).squeeze(1)
        ok = lab >= 0
        nll[i:i + chunk] = torch.where(ok, torch.logsumexp(lg, -1) - pk, torch.zeros_like(pk)).double()
        hit[i:i + chunk] = ((lg.argmax(-1) == lab) & ok).double()
    return nll, hit


def r_squared(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """1 - SSE / SST over the masked entries: 1 for a perfect predictor, 0 for predicting the mean, negative below it.
    NaN when the target has no variance (nothing to explain)."""
    m = mask.to(pred.dtype)
    n = m.sum().clamp_min(1)
    mean = (target * m).sum() / n
    sse = ((pred - target).square() * m).sum()
    sst = ((target - mean).square() * m).sum()
    return torch.where(sst > 0, 1 - sse / sst.clamp_min(1e-30), torch.full_like(sst, float("nan")))


def _code_use_scalars(use: Tensor) -> dict:
    """Per-level codebook use fraction and perplexity from [K, V] usage counts."""
    sc = {}
    for k in range(use.shape[0]):
        p = use[k] / use[k].sum().clamp_min(1)
        sc[f"tok/use_frac_l{k + 1}"] = float((use[k] > 0).double().mean())
        sc[f"tok/perplexity_l{k + 1}"] = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
    return sc


# --------------------------------------------------------------------------------------------- stage 1
class NCMTokenizer(nn.Module):
    """Multiscale residual VQ-VAE over the spans of a packed row. `forward(x, y, lay)` is the stage-1 loss.
    `decoder=False` builds the encoder and the codebooks only (stage 2 never decodes). With
    `ncm.tok_objective == "repr"` the same class is the RePR span-code model (module docstring): `decoder=True` then builds
    the training-only R_ctx head, `decoder=False` keeps what `encode` needs (trunk, R_loc, codebooks)."""

    def __init__(self, config, ncm: NCMConfig, decoder: bool = True):
        super().__init__()
        d, dz, K, V = config.n_embd, ncm.dz, len(ncm.levels), ncm.codebook_size
        self.config, self.ncm, self.K, self.S, self.has_decoder = config, ncm, K, ncm.max_slots, decoder
        self.emb_scale, self.logit_scale = float(config.init.embedding_scale), float(config.init.logit_scale)
        self.wte_e = nn.Embedding(config.padded_vocab_size, d)
        self.enc_blocks = nn.ModuleList(config.Block(config, layer_id=i) for i in range(ncm.enc_layers))
        self.ln_e = config.Norm(d, eps=config.norm_eps)
        self.is_repr = ncm.tok_objective == "repr"
        if not self.is_repr:
            self.z_proj = nn.Linear(d, dz, bias=False)
            nn.init.normal_(self.z_proj.weight, std=d ** -0.5)
        config.init.apply(self.wte_e, "embedding")
        config.init.apply(self.ln_e, "normalization")
        A.use_flex(self.enc_blocks)
        _freeze_ve(self.enc_blocks)
        if self.is_repr:
            self.r_loc = nn.Linear(d, ncm.zdim, bias=False)               # R_loc(p) on the LOCAL view's state
            nn.init.normal_(self.r_loc.weight, std=d ** -0.5)
            if decoder:
                self.r_ctx = nn.Linear(d, ncm.zdim, bias=False)           # R_ctx(s) on the FULL view's state before span s
                nn.init.normal_(self.r_ctx.weight, std=d ** -0.5)
        elif decoder:
            self.wte_d = nn.Embedding(config.padded_vocab_size, d)                # the output head is tied to it
            self.dec_in = nn.Linear(dz, d, bias=False)
            nn.init.normal_(self.dec_in.weight, std=float(config.init.get_std("embedding")) / math.sqrt(dz))
            self.dec_blocks = nn.ModuleList(config.Block(config, layer_id=i) for i in range(ncm.dec_layers))
            self.ln_d = config.Norm(d, eps=config.norm_eps)
            config.init.apply(self.wte_d, "embedding")
            config.init.apply(self.ln_d, "normalization")
            A.use_flex(self.dec_blocks)
            _freeze_ve(self.dec_blocks)
        cb = torch.randn(K, V, ncm.zdim)
        self.register_buffer("codebook", cb)
        self.register_buffer("ema_count", torch.ones(K, V))
        self.register_buffer("ema_sum", cb.clone())
        self.register_buffer("last_used", torch.zeros(K, V, dtype=torch.long))
        self.register_buffer("inited", torch.zeros(K, dtype=torch.bool))
        self.register_buffer("qstep", torch.zeros((), dtype=torch.long))
        head_dim = d // config.num_attention_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            head_dim, config.block_size, config.rope_settings.rope_base, config.rope_settings.rope_condense_ratio),
            persistent=False)
        self.metrics: dict[str, Tensor] = {}

    def compile_blocks(self) -> None:
        _compile(self.enc_blocks)
        if self.has_decoder and not self.is_repr:
            _compile(self.dec_blocks)

    # ---- E
    def encode(self, c: Compact) -> Tensor:
        """Per-token latents [B, T, dz] (float): a token reads the tokens of its own span, both directions.
        repr: the LOCAL pass only, and every token of a span holds the span's code z(s)."""
        if self.is_repr:
            rl = self.r_loc(self._trunk(c, A.span_pack_mask(c.bag))).float()
            return self._span_code(rl, c)
        B, T = c.x.shape
        h = self.wte_e(c.x) * self.emb_scale
        mask, freqs = _doc_mask(c.bag, causal=False), self.freqs_cis[:, :T]
        for blk in self.enc_blocks:
            h = blk(h, freqs, mask)
        z = self.z_proj(self.ln_e(h)).float()
        if self.ncm.z_norm:
            z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
        return z

    # ---- repr
    def _trunk(self, c: Compact, mask) -> Tensor:
        """The prediction trunk P under one mask: [B, T, d] after the final norm (the LM head and R read this)."""
        T = c.x.shape[1]
        h = self.wte_e(c.x) * self.emb_scale
        freqs = self.freqs_cis[:, :T]
        for blk in self.enc_blocks:
            h = blk(h, freqs, mask)
        return self.ln_e(h)

    def _doc_ids(self, x: Tensor) -> Tensor:
        """[B, T] document index of each compact token: it grows after every EOS (the EOS closes its own document)."""
        e = (x == self.ncm.eos_id).long()
        return e.cumsum(1) - e

    @staticmethod
    def _span_code(rl: Tensor, c: Compact) -> Tensor:
        """z(s) = R_loc at the last input position of span s, broadcast to the span's tokens: [B, T, zdim]
        (0 for the tokens of no complete span)."""
        B, T, D = rl.shape
        S = c.valid.shape[1]
        last = (c.n_tok.cumsum(1) - 1).clamp(0, T - 1)
        zs = rl.gather(1, last[..., None].expand(-1, -1, D)) * c.valid[..., None].to(rl.dtype)
        z = zs.gather(1, c.bag.clamp(max=S - 1)[..., None].expand(-1, -1, D))
        return z * (c.bag < S)[..., None].to(z.dtype)

    def _repr_pass(self, x: Tensor, y: Tensor, lay) -> dict:
        """Both views, both LM losses, the PMI targets and the repr prediction on one packed batch."""
        nc, S = self.ncm, self.S
        rows = make_rows(x, y, lay, S)
        c = rows.c
        B, T = c.x.shape
        doc = self._doc_ids(c.x)
        hf = self._trunk(c, A.span_pack_mask(doc))                          # FULL: earlier tokens of the same document
        hl = self._trunk(c, A.span_pack_mask(c.bag))                        # LOCAL: earlier tokens of the input token's span
        lab = rows.y.flatten()
        W = self.wte_e.weight
        nll_f = ce.linear_ce_tokens(hf.flatten(0, 1), W, lab, self.logit_scale, nc.slot_id).view(B, T)
        nll_l = ce.linear_ce_tokens(hl.flatten(0, 1), W, lab, self.logit_scale, nc.slot_id).view(B, T)
        with torch.no_grad():                                               # the PMI target is a number, never a path
            pmi = (nll_l - nll_f).float().clamp(-nc.pmi_clip, nc.pmi_clip)
        rl = self.r_loc(hl).float()
        out = {"rows": rows, "nll_f": nll_f, "nll_l": nll_l, "pmi": pmi, "rl": rl}
        if self.has_decoder:
            rc = self.r_ctx(hf).float()
            sidx = c.bag.clamp(max=S - 1)
            start = rows.start.gather(1, sidx)                              # compact index of the span's first token
            ctx_idx = (start - 1).clamp_min(0)                              # the last input position BEFORE the span
            prev = F.pad(rl, (0, 0, 1, 0))[:, :T] * (rows.pos > 0)[..., None].to(rl.dtype)   # R_loc(p-1), 0 at a span start
            rc_p = rc.gather(1, ctx_idx[..., None].expand(-1, -1, rc.shape[-1]))
            out["inc"] = rl - prev                                          # R_loc(p) - R_loc(p-1): telescopes inside a span
            out["pred"] = (rc_p * out["inc"]).sum(-1)
            has_ctx = (start >= 1) & (doc.gather(1, ctx_idx) == doc.gather(1, start.clamp(max=T - 1)))
            out["scored"] = (c.bag >= 1) & (c.bag < S) & (rows.y >= 0) & has_ctx
        return out

    def _forward_repr(self, x: Tensor, y: Tensor, lay) -> dict:
        nc = self.ncm
        o = self._repr_pass(x, y, lay)
        rows, c = o["rows"], o["rows"].c
        n = (rows.y >= 0).sum().clamp_min(1)
        ce_f, ce_l = o["nll_f"].sum() / n, o["nll_l"].sum() / n
        sc = o["scored"]
        n_sc = sc.sum().clamp_min(1)
        repr_loss = (((o["pred"] - o["pmi"]).square()) * sc).sum() / n_sc
        loss = 0.5 * (ce_f + ce_l) + nc.repr_weight * repr_loss
        seg = build_segments(c, nc.levels, self.S)
        self.rvq(self._span_code(o["rl"].detach(), c), seg, update=self.training)   # the VQ fits the code, it never moves R
        if self.training:
            self.qstep += 1
        with torch.no_grad():
            self.metrics.update({"ce_full": ce_f.detach(), "ce_local": ce_l.detach(), "repr_loss": repr_loss.detach(),
                                 "pmi_mean": (o["pmi"] * sc).sum() / n_sc, "scored": sc.sum().float(),
                                 "repr_r2": r_squared(o["pred"], o["pmi"], sc),   # NaN while pmi_var is 0 (e.g. step 1)
                                 "pmi_var": ((o["pmi"] - (o["pmi"] * sc).sum() / n_sc).square() * sc).sum() / n_sc})
        return {"loss": loss, "ce": ce_f.detach()}

    # ---- residual VQ
    @staticmethod
    def _nearest(u: Tensor, cb: Tensor) -> Tensor:
        uf, cf = u.reshape(-1, u.shape[-1]), cb.to(u.dtype)
        d = (uf * uf).sum(-1, keepdim=True) - 2 * (uf @ cf.T) + (cf * cf).sum(-1)[None]
        return d.argmin(-1).view(u.shape[:-1])

    def _draw(self, u: Tensor, ok: Tensor) -> Tensor:
        """[V, dz] random current residual vectors of the level (valid segments only)."""
        w = ok.reshape(-1).to(u.dtype) + 1e-9
        pick = torch.multinomial(w, self.ncm.codebook_size, replacement=True)
        return u.reshape(-1, u.shape[-1])[pick]

    def _init_level(self, k: int, samples: Tensor) -> None:
        """First training call: the level's codebook becomes data (a branch-free `where`, no host sync)."""
        need = ~self.inited[k]
        self.codebook[k].copy_(torch.where(need, samples, self.codebook[k]))
        self.ema_sum[k].copy_(torch.where(need, samples, self.ema_sum[k]))
        self.ema_count[k].copy_(torch.where(need, torch.ones_like(self.ema_count[k]), self.ema_count[k]))
        self.last_used[k].copy_(torch.where(need, self.qstep, self.last_used[k]))
        self.inited[k] = True

    def _ema(self, k: int, u: Tensor, idx: Tensor, ok: Tensor, samples: Tensor) -> Tensor:
        """EMA codebook update, then dead-code restart. Returns this batch's per-code counts."""
        nc, dz = self.ncm, u.shape[-1]
        V = nc.codebook_size
        w = ok.reshape(-1).to(u.dtype)
        fi = idx.reshape(-1)
        cnt = u.new_zeros(V).index_add_(0, fi, w)
        sm = u.new_zeros(V, dz).index_add_(0, fi, u.reshape(-1, dz) * w[:, None])
        dec = nc.ema_decay
        self.ema_count[k].mul_(dec).add_(cnt.to(self.ema_count.dtype), alpha=1 - dec)
        self.ema_sum[k].mul_(dec).add_(sm.to(self.ema_sum.dtype), alpha=1 - dec)
        n = self.ema_count[k].sum()
        smoothed = (self.ema_count[k] + nc.ema_eps) / (n + V * nc.ema_eps) * n
        self.codebook[k].copy_(self.ema_sum[k] / smoothed[:, None])
        self.last_used[k].copy_(torch.where(cnt > 0, self.qstep, self.last_used[k]))
        dead = (self.qstep - self.last_used[k]) > nc.restart_after
        dm = dead[:, None]
        smp = samples.to(self.codebook.dtype)
        self.codebook[k].copy_(torch.where(dm, smp, self.codebook[k]))
        self.ema_sum[k].copy_(torch.where(dm, smp, self.ema_sum[k]))
        self.ema_count[k].copy_(torch.where(dead, torch.ones_like(self.ema_count[k]), self.ema_count[k]))
        self.last_used[k].copy_(torch.where(dead, self.qstep, self.last_used[k]))
        return cnt

    @torch.no_grad()
    def rvq(self, z: Tensor, seg: SegInfo, upto: int | None = None, update: bool = False):
        """-> (codes, cum): per level the code indices [B, S, l_k] (0 where the segment does not exist) and the
        cumulative reconstruction zhat^(k) [B, T, dz]. `update`: EMA codebooks, data init and dead-code restart
        (training only)."""
        K = self.K if upto is None else upto
        codes, cum = [], []
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            zhat = torch.zeros_like(z)
            used = []
            for k in range(K):
                u = seg_mean(z - zhat, seg, k)
                ok = seg.seg_ok[k]
                if update:
                    smp = self._draw(u, ok)
                    self._init_level(k, smp)
                idx = torch.where(ok, self._nearest(u, self.codebook[k]), torch.zeros_like(ok, dtype=torch.long))
                e = self.codebook[k][idx]
                if update:
                    cnt = self._ema(k, u, idx, ok, smp)
                    used.append((cnt > 0).to(u.dtype).mean())
                zhat = zhat + upsample(e.to(z.dtype), seg, k)
                codes.append(idx)
                cum.append(zhat)
        if update:
            self.metrics.update({f"used_l{k + 1}": v for k, v in enumerate(used)})
        return codes, cum

    @torch.no_grad()
    def codes(self, c: Compact, upto: int | None = None, seg: SegInfo | None = None) -> list:
        """Per-level code index tensors [B, S, l_k] of a batch, frozen and no-grad."""
        seg = seg or build_segments(c, self.ncm.levels, self.S)
        return self.rvq(self.encode(c), seg, upto)[0]

    @torch.no_grad()
    def zhat_levels(self, codes: list, seg: SegInfo) -> list:
        """Cumulative per-token reconstructions [zhat^(1), ..., zhat^(len(codes))], each [B, T, dz]."""
        out, zh = [], None
        for k, ck in enumerate(codes):
            up = upsample(self.codebook[k][ck], seg, k)
            zh = up if zh is None else zh + up
            out.append(zh)
        return out

    @torch.no_grad()
    def zhat(self, codes: list, c: Compact, m: int, seg: SegInfo | None = None) -> Tensor:
        """Per-token reconstruction from the first m levels [B, T, dz] (zeros for m = 0)."""
        seg = seg or build_segments(c, self.ncm.levels, self.S)
        if m == 0:
            return torch.zeros(*c.x.shape, self.ncm.zdim, device=c.x.device, dtype=self.codebook.dtype)
        return self.zhat_levels(codes[:m], seg)[-1]

    # ---- D
    def decode_hidden(self, zd: Tensor, c: Compact) -> Tensor:
        """[B, T, d]: the decoder state at every token position from a (possibly partial) reconstruction."""
        B, T = c.x.shape
        h = self.dec_in(zd.to(self.dec_in.weight.dtype)).float()          # fp32 residual stream, as Parcae's
        mask, freqs = _doc_mask(c.bag, causal=False), self.freqs_cis[:, :T]
        for blk in self.dec_blocks:
            h = blk(h, freqs, mask)
        return self.ln_d(h)

    def _cutoff(self) -> int:
        nc = self.ncm
        if self.K > 1 and float(torch.rand(())) < nc.nested_dropout:
            return int(torch.randint(1, self.K, ()))
        return self.K

    def forward(self, x: Tensor, y: Tensor | None, lay, depth: int | None = None) -> dict:
        if self.is_repr:
            return self._forward_repr(x, y, lay)
        c = compact(x, lay, self.S)
        seg = build_segments(c, self.ncm.levels, self.S)
        z = self.encode(c)
        _, cum = self.rvq(z.detach(), seg, update=self.training)
        if self.training:
            self.qstep += 1
        m = self._cutoff() if self.training else self.K
        zm = cum[m - 1]
        zd = z + (zm - z).detach()                                             # straight-through
        h = self.decode_hidden(zd, c)
        labels = torch.where(seg.tokok, c.x, torch.full_like(c.x, -100))
        rec = ce.linear_ce(h.flatten(0, 1), self.wte_d.weight, labels.flatten(), self.logit_scale, self.ncm.slot_id)
        w = seg.tokok.to(z.dtype)
        commit = ((z - zm.detach()).square().mean(-1) * w).sum() / w.sum().clamp_min(1)
        loss = rec + self.ncm.beta * commit
        self.metrics.update({"recon_ce": rec.detach(), "commit": commit.detach(),
                             "cutoff": torch.tensor(float(m))})
        return {"loss": loss, "ce": rec.detach()}

    @torch.no_grad()
    def depth_sweep(self, rows, depths, device="cuda") -> dict:
        """Reconstruction CE per cutoff over every token of the complete spans. A requested `depth` d reads
        cutoff min(d, K) (the standard sweep keys: ref 6 = all six levels). `scalars` holds the readouts:
        recon CE and exact token accuracy at each cutoff 1..K, per-level codebook use fraction and perplexity."""
        if self.is_repr:
            return self._depth_sweep_repr(rows, depths, device)
        was = self.training
        self.eval()
        K, V, S = self.K, self.ncm.codebook_size, self.S
        sums = {m: [] for m in range(1, K + 1)}
        hits = {m: 0.0 for m in range(1, K + 1)}
        counts = []
        use = torch.zeros(K, V, dtype=torch.float64)
        dev = torch.device(device)
        for batch in rows:
            x, _, lay = rows_to(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                c = compact(x, lay, S)
                seg = build_segments(c, self.ncm.levels, S)
                z = self.encode(c)
                codes, cum = self.rvq(z, seg)
                labels = torch.where(seg.tokok, c.x, torch.full_like(c.x, -100)).flatten()
                for m in range(1, K + 1):
                    h = self.decode_hidden(cum[m - 1], c)
                    nll, hit = _nll_acc(h.flatten(0, 1), self.wte_d.weight, labels, self.logit_scale, self.ncm.slot_id)
                    sums[m].append(nll.view(x.shape).sum(1).cpu())
                    hits[m] += float(hit.sum())
                for k in range(K):
                    use[k] += torch.zeros(V, dtype=torch.float64, device=x.device).index_add_(
                        0, codes[k].reshape(-1), seg.seg_ok[k].reshape(-1).double()).cpu()
            counts.append(seg.tokok.sum(1).double().cpu())
        self.train(was)
        cnt = torch.cat(counts)
        total = float(cnt.sum())
        out = {"depths": list(depths), "count": cnt.tolist(),
               "sums": {str(d): torch.cat(sums[min(d, K)]).tolist() for d in depths}}
        sc = {}
        for m in range(1, K + 1):
            sc[f"tok/recon_ce_m{m}"] = float(torch.cat(sums[m]).sum()) / total
            sc[f"tok/acc_m{m}"] = hits[m] / total
        sc.update(_code_use_scalars(use))
        out["scalars"] = sc
        return out

    @torch.no_grad()
    def _depth_sweep_repr(self, rows, depths, device="cuda") -> dict:
        """repr readouts. `sums`/`count` (every depth key, the tokenizer has no depth): the FULL view's NLL over every label
        token of the row, so `val/ce_d6` is the trunk's LM loss. `scalars`: CE full and local (all labels, and on the scored
        set), mean PMI per scored token, R^2 of the repr prediction against PMI, per-level code use and perplexity."""
        was = self.training
        self.eval()
        K, V, S = self.K, self.ncm.codebook_size, self.S
        dev = torch.device(device)
        sums, counts = [], []
        a = {k: 0.0 for k in ("nll_f", "nll_l", "n", "nll_f_sc", "nll_l_sc", "n_sc", "se", "p", "p2")}
        use = torch.zeros(K, V, dtype=torch.float64)
        for batch in rows:
            x, y, lay = rows_to(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = self._repr_pass(x, y, lay)
                c = o["rows"].c
                seg = build_segments(c, self.ncm.levels, S)
                codes, _ = self.rvq(self._span_code(o["rl"], c), seg)
            ok, sc, pmi = o["rows"].y >= 0, o["scored"], o["pmi"].double()
            sums.append(o["nll_f"].double().sum(1).cpu())
            counts.append(ok.sum(1).double().cpu())
            a["nll_f"] += float(o["nll_f"].double().sum()); a["nll_l"] += float(o["nll_l"].double().sum()); a["n"] += float(ok.sum())
            a["nll_f_sc"] += float((o["nll_f"].double() * sc).sum()); a["nll_l_sc"] += float((o["nll_l"].double() * sc).sum())
            a["n_sc"] += float(sc.sum())
            a["se"] += float(((o["pred"].double() - pmi).square() * sc).sum())
            a["p"] += float((pmi * sc).sum()); a["p2"] += float((pmi.square() * sc).sum())
            for k in range(K):
                use[k] += torch.zeros(V, dtype=torch.float64, device=x.device).index_add_(
                    0, codes[k].reshape(-1), seg.seg_ok[k].reshape(-1).double()).cpu()
        self.train(was)
        cnt, n_sc = torch.cat(counts), max(a["n_sc"], 1.0)
        sst = a["p2"] - a["p"] ** 2 / n_sc
        out = {"depths": list(depths), "count": cnt.tolist(), "sums": {str(d): torch.cat(sums).tolist() for d in depths}}
        sc_out = {"tok/ce_full": a["nll_f"] / max(a["n"], 1.0), "tok/ce_local": a["nll_l"] / max(a["n"], 1.0),
                  "tok/ce_full_scored": a["nll_f_sc"] / n_sc, "tok/ce_local_scored": a["nll_l_sc"] / n_sc,
                  "tok/pmi_mean": a["p"] / n_sc, "tok/pmi_scored_tokens": a["n_sc"],
                  "tok/repr_r2": 1.0 - a["se"] / sst if sst > 0 else float("nan")}
        sc_out.update(_code_use_scalars(use))
        out["scalars"] = sc_out
        return out


# --------------------------------------------------------------------------------------------- stage 2
def _ncm_mask(span: Tensor, lvl: Tensor, ok: Tensor, T: int, G: int, local: bool, reads: bool,
              q_off: int, kv_off: int, q_len: int, kv_len: int):
    """The one mask of the prelude and the core over the sequence [T tokens ; Q generator positions]
    (combined index = BlockMask index + offset). `span` [B, T+Q] (tokens: bag, S for the open tail and the
    padding; generator positions: the span they belong to), `lvl` (tokens -1, generator positions their level,
    G for the summary position), `ok` (the position exists)."""

    def mod(b, h, q, kv):
        q = q + q_off
        kv = kv + kv_off
        sq, sk = span[b, q], span[b, kv]
        lq, lk = lvl[b, q], lvl[b, kv]
        qt, kt = q < T, kv < T
        tok = qt & kt & (kv <= q)
        if local:
            tok = tok & (sk == sq)
        code = (~qt) & ok[b, q] & (~kt) & ok[b, kv] & ((sk < sq) | ((sk == sq) & (lk == lq) & (lq < G)))
        if reads:
            code = code | ((~qt) & ok[b, q] & kt & (sk < sq))
        return tok | code | (q == kv)
    return A._mask(mod, span.shape[0], None, q_len, kv_len, device=span.device)


def _core_layer(block, x, kt, vt, mask, freqs):
    """One core layer with generator queries: `TransformerPreNormBlock.forward` where the keys and values are
    [token prelude keys/values (fixed for the forward; None without token memory) ; the generator states']."""
    attn = block.attn
    q, k, v = attn.project_qkv(block.norm_1(x), freqs, None)
    if kt is not None:
        k, v = torch.cat([kt, k], 1), torch.cat([vt, v], 1)
    y = A._flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask,
                enable_gqa=attn.n_rep > 1)
    B, Q, C = x.shape
    x = x + attn.c_proj(y.transpose(1, 2).reshape(B, Q, C))
    return x + block.mlp(block.norm_2(x))


@dataclass
class _Ctx:
    rows: Rows
    seg: SegInfo
    htok: Tensor             # [B, T, d] plan-free token prelude states
    e: Tensor                # [B, Q, d] generator prelude states
    ktv: list | None         # per core layer (k, v) of the token states, or None without token memory
    cmask: object            # core BlockMask
    freqs_q: Tensor
    freqs_tok: Tensor
    coda_mask: object
    Zall: Tensor             # [B, T, G + 1, dz] true cumulative reconstructions (level 0 = zeros)
    lab: Tensor              # [B, T] token labels (-100 = not scored)
    lab_idx: Tensor          # [B, T] compact index of the label token
    lab_u: Tensor            # [B, T] span of the label token (S: none)
    plan_ok: Tensor          # [B, T] the label belongs to a complete span u >= 1 (the plan applies)
    tgt: Tensor              # [B, S, qmax] true code targets (-100: not charged)
    ve_ids: Tensor           # [B, T] ids the token value embeddings read
    ve_keep: Tensor | None   # [B, T, 1] 0 where the shifted input is a span-start BOS (pure arm), else None
    short: Tensor            # complete spans shorter than ncm.min_span (capped levels, still scored)


class NCMModel(nn.Module):
    """Frozen tokenizer + token model with a plan + looped generator (stage 2)."""

    def __init__(self, tok_config, bb_config, ncm: NCMConfig):
        super().__init__()
        if tok_config.n_embd != bb_config.n_embd:
            raise ValueError("the backbone width must equal the tokenizer width (n_embd)")
        d, G, S, dz, V = bb_config.n_embd, ncm.gen_levels, ncm.max_slots, ncm.zdim, ncm.codebook_size
        self.ncm, self.G, self.S, self.d = ncm, G, S, d
        self.summary = not ncm.gen_reads_tokens
        self.tok = NCMTokenizer(tok_config, ncm, decoder=False).requires_grad_(False).eval()
        core = bb_config.construct_model(gradient_checkpointing=False)
        del core.lm_head
        core.value_embeds = nn.ModuleDict()           # generator positions carry no token id: no value embeddings
        core._current_input_ids = None
        t = core.transformer
        for blocks in (t.prelude, t.core_block, t.coda):
            A.use_flex(blocks)
            _freeze_ve(blocks)
        # value embeddings exactly as GPT (gpt.py: layers i with i % 2 == (n_layer - 1) % 2 of the n_layer = prelude + coda
        # token layers, one Embedding(vocab, kv_dim) each, gate Linear(32, n_kv_head) in that layer's attention). They are
        # read at TOKEN positions only; generator positions get a zero value embedding; the core blocks have none.
        n_pre, n_tl = len(t.prelude), len(t.prelude) + len(t.coda)
        self.ve_layers = {g: blk for g, blk in enumerate([*t.prelude, *t.coda]) if has_ve(g, n_tl)}
        for g, blk in self.ve_layers.items():
            blk.attn.ve_gate = nn.Linear(blk.attn.ve_gate_channels, blk.attn.n_kv_head, bias=False)
        kv_dim = t.prelude[0].attn.n_kv_head * t.prelude[0].attn.head_dim
        self.value_embeds = nn.ModuleDict({str(g): nn.Embedding(bb_config.padded_vocab_size, kv_dim) for g in self.ve_layers})
        self.n_pre = n_pre
        self.core = core
        self.emb_scale = float(bb_config.init.embedding_scale)
        std = float(bb_config.init.get_std("embedding")) * self.emb_scale
        # generator layout: per span the level blocks (l_1 .. l_G positions), then the summary position
        ls = ncm.levels[:G]
        self.l_gen, self.qmax = ls, sum(ls)
        self.P = self.qmax + (1 if self.summary else 0)
        self.Q = S * self.P
        self.off = [sum(ls[:k]) for k in range(G)]
        lvl = [k for k, lk in enumerate(ls) for _ in range(lk)] + ([G] if self.summary else [])
        sgi = [i for lk in ls for i in range(lk)] + ([0] if self.summary else [])
        self.register_buffer("gen_lvl", torch.tensor(lvl), persistent=False)
        self.register_buffer("gen_seg", torch.tensor(sgi), persistent=False)
        self.gen_in_proj = nn.Linear(dz, d, bias=False)
        nn.init.normal_(self.gen_in_proj.weight, std=std / math.sqrt(dz))
        self.lvl_embedding = nn.Parameter(torch.randn(G + 1, d) * std)
        self.pos_embedding = nn.Parameter(torch.randn(max(ls), d) * std)
        self.ln_gen = bb_config.Norm(d, eps=bb_config.norm_eps)
        bb_config.init.apply(self.ln_gen, "normalization")
        self.lm_head_code = nn.ModuleList(nn.Linear(d, V, bias=False) for _ in range(G))
        for h in self.lm_head_code:
            nn.init.normal_(h.weight, std=std)
        self.plan_proj = nn.Linear(dz, d, bias=False)
        nn.init.zeros_(self.plan_proj.weight)           # step 0: the token model without a plan
        self.m_embedding = nn.Parameter(torch.zeros(G + 1, d))
        if ncm.coda_span_local:
            self.bos_embedding = nn.Parameter(torch.randn(d) * std)
        if ncm.pointer_heads > 0:
            self.pointer = PointerHead(d, ncm.pointer_heads)
        self.step = 0
        self.metrics: dict[str, Tensor] = {}
        self._core_layer = _core_layer                  # compile_blocks() swaps in the compiled one

    def train(self, mode: bool = True):
        super().train(mode)
        self.tok.eval()                                  # the tokenizer never trains and never updates a codebook
        return self

    def compile_blocks(self) -> None:
        self.tok.compile_blocks()
        t = self.core.transformer
        for blocks in (t.prelude, t.coda):
            _compile(blocks)
        self._core_layer = torch.compile(_core_layer, backend="inductor", mode="default", dynamic=False)

    @property
    def wte_weight(self) -> Tensor:
        return self.core.transformer.wte.weight

    # ---------------------------------------------------------------- inputs
    def _corrupt(self, codes: list) -> list:
        """Conditioning corruption: each code of level k replaced by a uniform random code of its level with
        probability corrupt[k] (training only; the labels keep the true codes)."""
        out = []
        for k, ck in enumerate(codes):
            flip = torch.rand(ck.shape, device=ck.device) < self.ncm.corrupt[k]
            out.append(torch.where(flip, torch.randint(0, self.ncm.codebook_size, ck.shape, device=ck.device), ck))
        return out

    def _prepare(self, x: Tensor, y: Tensor, lay, train: bool, codes: list | None = None) -> _Ctx:
        """Everything that does not depend on the generator depth or on m. `codes` (default: the tokenizer's
        codes of `x`) are the conditioning codes and the generator's labels."""
        ncm, S, G, d, P = self.ncm, self.S, self.G, self.d, self.P
        rows = make_rows(x, y, lay, S)
        c = rows.c
        B, T = c.x.shape
        dev = x.device
        short = (c.valid & (c.n_tok < ncm.min_span)).sum()
        seg = build_segments(c, ncm.levels, S)
        with torch.no_grad():
            if codes is None:
                codes = self.tok.codes(c, upto=G, seg=seg)
            zs = self.tok.zhat_levels(codes, seg)
            zs_in = self.tok.zhat_levels(self._corrupt(codes), seg) if train and any(ncm.corrupt) else zs
            Zall = torch.stack([torch.zeros_like(zs[0])] + zs, 2)
            ar_s = torch.arange(S, device=dev)
            has_codes = c.valid & (ar_s[None] >= 1)
            nseg_cat = torch.stack([seg.nseg[k] for k in range(G)], 2)[:, :, self.gen_lvl[: self.qmax]]
            vp_code = has_codes[..., None] & (self.gen_seg[: self.qmax] < nseg_cat)          # [B, S, qmax]
            tgt = torch.cat([codes[k] for k in range(G)], 2)
            tgt = torch.where(vp_code, tgt, torch.full_like(tgt, -100))
            parts = [torch.zeros(B, S, ncm.levels[0], ncm.zdim, device=dev, dtype=Zall.dtype)]
            parts += [seg_mean(zs_in[k - 1], seg, k) for k in range(1, G)]
            valid_pos = vp_code
            if self.summary:
                parts.append(seg_mean(zs_in[G - 1], seg, 0))            # Down_1: the mean over the span's tokens
                valid_pos = torch.cat([vp_code, c.valid[..., None]], 2)
            gen_x = torch.cat(parts, 2)                                 # [B, S, P, dz]
        gen = self.gen_in_proj(gen_x) + self.lvl_embedding[self.gen_lvl] + self.pos_embedding[self.gen_seg]
        gen = gen.reshape(B, self.Q, d)
        # ---- token inputs
        emb = self.core.transformer.wte(c.x) * self.emb_scale
        local = ncm.coda_span_local
        if local:       # predict token t of a span at position t of the span: input shifted right, BOS at the start
            first = (rows.pos == 0) | (torch.arange(T, device=dev)[None] == 0)
            emb = torch.where(first[..., None], self.bos_embedding.to(emb.dtype), F.pad(emb, (0, 0, 1, 0))[:, :T])
        # ---- one sequence, one mask for the prelude
        span = torch.cat([c.bag, ar_s.repeat_interleave(P)[None].expand(B, -1)], 1)
        lvl = torch.cat([torch.full_like(c.bag, -1), self.gen_lvl.repeat(S)[None].expand(B, -1)], 1)
        ok = torch.cat([torch.ones_like(c.bag, dtype=torch.bool), valid_pos.reshape(B, self.Q)], 1)
        T_all = T + self.Q
        pmask = _ncm_mask(span, lvl, ok, T, G, local, ncm.gen_reads_tokens, 0, 0, T_all, T_all)
        reads = ncm.gen_reads_tokens
        cmask = _ncm_mask(span, lvl, ok, T, G, local, reads, T, 0 if reads else T, self.Q, T_all if reads else self.Q)
        freqs_tok = self.core.freqs_cis[:, :T]
        fc = self.core.freqs_cis[0, :, 0]
        ids = rows.start.clamp(max=T - 1).repeat_interleave(P, dim=1)                    # a span's positions sit at its start
        freqs_q = fc[ids].unsqueeze(2)
        freqs_all = torch.cat([freqs_tok.expand(B, -1, -1, -1, -1), freqs_q], 1)
        t = self.core.transformer
        h = torch.cat([emb, gen.to(emb.dtype)], 1)
        ve_ids = c.x
        if local:       # shifted input: the value embedding follows the input token, none at a span start
            ve_ids = F.pad(c.x, (1, 0))[:, :T]
        ve_keep = None if not local else (~first)[..., None]
        for i, blk in enumerate(t.prelude):
            ve = self._ve(i, ve_ids, ve_keep)
            if ve is not None:
                ve = torch.cat([ve, ve.new_zeros(B, self.Q, ve.shape[-1])], 1)
            h = blk(h, freqs_all, pmask, ve=ve)
        htok, e = h[:, :T], h[:, T:]          # htok: the raw prelude output, as GPT's layer 4 reads it (no ln_prelude)
        htok_n = htok
        if self.core.config.prelude_norm:    # Parcae's loop entry norm: generator side only (states, injection, token K/V)
            e, htok_n = t.ln_prelude(e), t.ln_prelude(htok)
        ktv = None
        if reads:
            ktv = []
            for blk in t.core_block:
                _, k, v = blk.attn.project_qkv(blk.norm_1(htok_n), freqs_tok, None)
                ktv.append((k, v))
        # ---- what each coda position predicts
        ar = torch.arange(T, device=dev)[None].expand(B, -1)
        if local:
            lab_idx = ar
            lab = torch.where((c.bag >= 1) & (c.bag < S), c.x, torch.full_like(c.x, -100))
            in_range = torch.ones_like(c.bag, dtype=torch.bool)
        else:
            lab_idx = (ar + 1).clamp(max=T - 1)
            lab = rows.y
            in_range = (ar + 1) < rows.n_all[:, None]
        lab_u = c.bag.gather(1, lab_idx)
        plan_ok = in_range & (lab_u >= 1) & (lab_u < S)
        coda_mask = A.span_pack_mask(c.bag) if local else None
        return _Ctx(rows, seg, htok, e, ktv, cmask, freqs_q, freqs_tok, coda_mask, Zall, lab, lab_idx, lab_u,
                    plan_ok, tgt, ve_ids, ve_keep, short)

    # ---------------------------------------------------------------- generator
    def _pass(self, x: Tensor, ctx: _Ctx) -> Tensor:
        x = self.core.transformer.adapter(x, ctx.e)
        for i, blk in enumerate(self.core.transformer.core_block):
            kt, vt = ctx.ktv[i] if ctx.ktv is not None else (None, None)
            x = self._core_layer(blk, x, kt, vt, ctx.cmask, ctx.freqs_q)
        return x

    def _gen_states(self, ctx: _Ctx, depth: int | None) -> Tensor:
        """[B, S, P, d] generator states after the loop. `depth` None: Parcae's per-row draw (the first n passes
        without grad, then k with grad); an int: that many passes, no grad."""
        core = self.core
        core.step = self.step
        B = ctx.e.shape[0]
        x = core.initialize_state(ctx.e)
        if depth is not None:
            with torch.no_grad():
                for _ in range(depth):
                    x = self._pass(x, ctx)
        else:
            n_ps, k_ps = core._sample_batch_depths(B)
            n_max, k_max = int(n_ps.max()), int(k_ps.max())
            n_ps, k_ps = n_ps.to(x.device), k_ps.to(x.device)
            with torch.no_grad():
                for s in range(n_max):
                    x = torch.where((s < n_ps)[:, None, None], self._pass(x, ctx), x)
            for s in range(k_max):
                x = torch.where((s < k_ps)[:, None, None], self._pass(x, ctx), x)
        x = self.ln_gen(core.transformer.C(x))
        return x.reshape(B, self.S, self.P, self.d)

    def _gen_logits(self, ctx: _Ctx, depth: int | None) -> list:
        st = self._gen_states(ctx, depth)
        return [self.lm_head_code[k](st[:, :, self.off[k]: self.off[k] + self.l_gen[k]]).float() for k in range(self.G)]

    def _code_nll(self, ctx: _Ctx, depth: int | None) -> list:
        """Per level the code NLL [B, S, l_k] (0 where nothing is charged)."""
        out = []
        for k, lg in enumerate(self._gen_logits(ctx, depth)):
            t = ctx.tgt[:, :, self.off[k]: self.off[k] + self.l_gen[k]]
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="none")
            out.append(nll.view(t.shape))
        return out

    def _ve(self, layer: int, ids: Tensor, keep: Tensor | None):
        """Value embedding of token layer `layer` (global GPT index) at token positions, or None."""
        e = self.value_embeds[str(layer)](ids) if str(layer) in self.value_embeds else None
        if e is not None and keep is not None:
            e = e * keep.to(e.dtype)
        return e

    # ---------------------------------------------------------------- token model
    def _token_nll(self, ctx: _Ctx, m_span: Tensor) -> Tensor:
        """Per-position token NLL [B, T] (0 where the label is not scored) with the plan of m_span[b, u] levels
        for span u (spans without a plan run at m = 0)."""
        G, S, dz = self.G, self.S, self.ncm.zdim
        m_t = torch.where(ctx.plan_ok, m_span.gather(1, ctx.lab_u.clamp(max=S - 1)), torch.zeros_like(ctx.lab_u))
        zl = ctx.Zall.gather(1, ctx.lab_idx[..., None, None].expand(-1, -1, G + 1, dz))
        zsel = zl.gather(2, m_t[..., None, None].expand(-1, -1, 1, dz)).squeeze(2)
        plan = self.plan_proj(zsel) + self.m_embedding[m_t]
        t = self.core.transformer
        h = ctx.htok + plan.to(ctx.htok.dtype)
        for j, blk in enumerate(t.coda):
            h = blk(h, ctx.freqs_tok, ctx.coda_mask, ve=self._ve(self.n_pre + j, ctx.ve_ids, ctx.ve_keep))
        hn = t.ln_f(h)
        # GPT's own head: tied wte, no logit scale, no masked logit (gpt.py return_logits path), then cross entropy
        logits = torch.matmul(hn, self.wte_weight.T).float()
        nll = F.cross_entropy(logits.flatten(0, 1), ctx.lab.flatten().clamp_min(0), reduction="none").view(ctx.lab.shape)
        if self.ncm.pointer_heads > 0:
            g, p, null = self.pointer(hn, ctx.lab, torch.zeros_like(ctx.lab, dtype=torch.bool))
            nll = -mix_pointer(-nll, g, p, null)
        return torch.where(ctx.lab >= 0, nll, torch.zeros_like(nll))

    # ---------------------------------------------------------------- training
    def forward(self, x: Tensor, y: Tensor | None, lay, depth: int | None = None) -> dict:
        ctx = self._prepare(x, y, lay, self.training)
        B = x.shape[0]
        if self.training and self.ncm.m_train == "uniform":
            m_span = torch.randint(0, self.G + 1, (B, self.S), device=x.device)
        else:
            m_span = torch.full((B, self.S), self.G, device=x.device, dtype=torch.long)
        tok = self._token_nll(ctx, m_span)
        code = self._code_nll(ctx, depth)
        n_lab = (ctx.lab >= 0).sum().clamp_min(1)
        loss = (tok.sum() + sum(cn.sum() for cn in code)) / n_lab
        tok_ce = tok.sum().detach() / n_lab
        self.metrics = {"token_ce": tok_ce, "bound": loss.detach(), "m_mean": m_span.float().mean(),
                        "short_spans": ctx.short.detach()}
        for k, cn in enumerate(code):
            n_k = (ctx.tgt[:, :, self.off[k]: self.off[k] + self.l_gen[k]] >= 0).sum().clamp_min(1)
            self.metrics[f"code_ce_l{k + 1}"] = cn.sum().detach() / n_k
        return {"loss": loss, "ce": tok_ce}

    # ---------------------------------------------------------------- evaluation
    @torch.no_grad()
    def depth_sweep(self, rows, depths, device="cuda") -> dict:
        """Per-row ELBO sums, per generator depth d and per m in {0..G}: token NLL given zhat^(m) (the coda reads
        the TRUE codes) + the code NLL of levels <= m. Keys: `count` (scored tokens per row), `sums` (m = G),
        `sums_m{m}[d]`, `token_m{m}` (per row, independent of d), `code_level{k}[d]`, `code_count_level{k}`,
        `scope`, `scalars` (the d = 6 readouts)."""
        was = self.training
        self.eval()
        G, S, ncm = self.G, self.S, self.ncm
        dev = torch.device(device)
        cuda = dev.type == "cuda"
        acc = {"count": [], "ref": [], "short": 0, "tok": {m: [] for m in range(G + 1)},
               "code": {d: [[] for _ in range(G)] for d in depths}, "ncode": [[] for _ in range(G)]}
        with torch.random.fork_rng(devices=[dev.index if dev.index is not None else torch.cuda.current_device()]
                                   if cuda else []):
            for bi, batch in enumerate(rows):
                x, y, lay = rows_to(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    ctx = self._prepare(x, y, lay, train=False)
                    acc["count"].append((ctx.lab >= 0).sum(1).double().cpu())
                    acc["short"] += int(ctx.short)
                    if not ncm.coda_span_local:       # the tokens evaluate.plain_view scores
                        acc["ref"].append(((~lay.slot_mask) & (y >= 0)).sum(1).double().cpu())
                    else:                             # the LCM scored set: complete spans with index >= 1
                        acc["ref"].append((ctx.rows.c.n_tok * ctx.rows.c.valid)[:, 1:].sum(1).double().cpu())
                    for m in range(G + 1):
                        mm = torch.full((x.shape[0], S), m, device=x.device, dtype=torch.long)
                        acc["tok"][m].append(self._token_nll(ctx, mm).sum(1).double().cpu())
                    for k in range(G):
                        acc["ncode"][k].append((ctx.tgt[:, :, self.off[k]: self.off[k] + self.l_gen[k]] >= 0)
                                               .sum((1, 2)).double().cpu())
                    for d in depths:
                        torch.manual_seed(ncm.eval_seed * 1_000_003 + bi)         # same state init at every depth
                        for k, cn in enumerate(self._code_nll(ctx, d)):
                            acc["code"][d][k].append(cn.sum((1, 2)).double().cpu())
        self.train(was)
        cat = lambda v: torch.cat(v)
        count, ref = cat(acc["count"]), cat(acc["ref"])
        if not torch.equal(count, ref):
            raise RuntimeError("scored token count differs from the independent recount of the scored set")
        tok = {m: cat(acc["tok"][m]) for m in range(G + 1)}
        code = {d: [cat(v) for v in acc["code"][d]] for d in depths}
        elbo = lambda m, d: tok[m] + sum((code[d][k] for k in range(m)), torch.zeros_like(count))
        out = {"depths": list(depths), "count": count.tolist(),
               "sums": {str(d): elbo(G, d).tolist() for d in depths}}
        for m in range(G + 1):
            out[f"sums_m{m}"] = {str(d): elbo(m, d).tolist() for d in depths}
            out[f"token_m{m}"] = tok[m].tolist()
        for k in range(G):
            out[f"code_level{k + 1}"] = {str(d): code[d][k].tolist() for d in depths}
            out[f"code_count_level{k + 1}"] = cat(acc["ncode"][k]).tolist()
        out["scope"] = {"rows": int(count.numel()), "scored": int(count.sum()), "spans_below_min_span": acc["short"],
                        "set": "tokens of complete spans with index >= 1" if ncm.coda_span_local
                        else "every token label of the packed row (evaluate.plain_view)"}
        total = float(count.sum())
        sc = {}
        if 6 in depths:
            for m in range(G + 1):
                sc[f"ncm/ce_m{m}_d6"] = float(elbo(m, 6).sum()) / total
                sc[f"ncm/token_ce_m{m}"] = float(tok[m].sum()) / total
            for k in range(G):
                sc[f"ncm/code_ce_level{k + 1}_d6"] = float(code[6][k].sum()) / max(float(cat(acc["ncode"][k]).sum()), 1.0)
        if 1 in depths and 6 in depths:
            for m in range(G + 1):
                sc[f"ncm/K1_K6_m{m}"] = float(elbo(m, 1).sum() - elbo(m, 6).sum()) / total
        out["scalars"] = sc
        return out


# --------------------------------------------------------------------------------------------- build
def ncm_config(cfg, mcfg, rt) -> NCMConfig:
    """`arm.ncm` (+ the packer's slot id and slot budget; `rt` is MORPH's TUL runtime). An unknown key raises."""
    from omegaconf import OmegaConf
    kw = OmegaConf.to_container(cfg.arm.ncm, resolve=True)
    return NCMConfig(slot_id=int(rt.data_cfg.slot_id), max_slots=int(mcfg.tul.max_slots),
                     eos_id=int(rt.data_cfg.rule.eos_id), **kw)


_FIELD_DEFAULTS = {f.name: f.default for f in dataclasses.fields(NCMConfig)}


def load_tokenizer(tok: NCMTokenizer, run_dir: str, nc: NCMConfig) -> None:
    """Load the stage-1 weights of `run_dir/final.pt` into the (decoder-free) tokenizer; refuse another architecture."""
    path = Path(run_dir) / "final.pt"
    if not path.is_file():
        raise FileNotFoundError(f"ncm.tok_ckpt: {path} does not exist (run the stage-1 arm first)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arm = ck["config"]["arm"]
    if arm["kind"] != "ncm_tok":
        raise ValueError(f"{path} is a {arm['kind']!r} run, not an ncm_tok run")
    for k in TOK_ARCH_KEYS:
        have, want = arm["ncm"].get(k, _FIELD_DEFAULTS[k]), getattr(nc, k)     # a pre-repr checkpoint has no tok_objective: recon
        if k == "levels":
            have = tuple(have)
        if have != want:
            raise ValueError(f"ncm.{k}: stage 2 has {want}, the tokenizer was trained with {have}")
    sd = {k.replace("._orig_mod", ""): v for k, v in ck["model"].items()}
    own = tok.state_dict()
    missing = [k for k in own if k not in sd]
    if missing:
        raise ValueError(f"{path} lacks tokenizer entries {missing[:4]}")
    tok.load_state_dict({k: sd[k] for k in own}, strict=True)


def build_ncm(cfg, mcfg, arm: str, compile: bool, load: bool = True):
    """-> (model on cpu, tul runtime). `arm` in NCM_KINDS. Both stages are sized for the packed one-cell row.
    `load` False skips reading `ncm.tok_ckpt` (the eval path loads the whole stage-2 state dict)."""
    from lxtul.build import parcae_config
    from lxtul.data import layout_for_cells, packed_length, tul_runtime
    rt = tul_runtime(mcfg)
    nc = ncm_config(cfg, mcfg, rt)
    rt = layout_for_cells(rt, nc.cells)
    vocab, T = int(mcfg.model.vocab_size), packed_length(mcfg, rt)
    tok_cfg = parcae_config(cfg, vocab, plain=False, block_size=T)
    if arm == "ncm_tok":
        model = NCMTokenizer(tok_cfg, nc, decoder=True)
    elif arm == "ncm":
        model = NCMModel(tok_cfg, parcae_config(cfg, vocab, plain=True, block_size=T), nc)
        if load:
            load_tokenizer(model.tok, nc.tok_ckpt, nc)
    else:
        raise ValueError(f"unknown ncm arm {arm!r}")
    if compile:
        model.compile_blocks()
    return model, rt
