"""LCM-Parcae: a pure Large Concept Model (Meta FAIR, "Large Concept Models", Dec 2024) on a Parcae backbone.

A reference model, not a TUL arm: the units are the spans the TUL packer cuts (same boundary rule, same
rows), and the model sees the text of earlier spans only as ONE vector per span, the concept.

  Stage 1  `LCMAutoencoder` (arm kind `lcm_ae`), standing in for the frozen SONAR encoder/decoder.
           E: token embedding + `enc_layers` Parcae blocks with span-local causal attention, then one learned
           query pools each span to a vector, RMS-normalised to `z_scale`. D: `dec_layers` Parcae blocks,
           span-local causal; each span reads its concept as a PREFIX position and its own tokens, and is
           trained with teacher-forced CE. Fixed Gaussian noise on z (`ae_noise` x `z_scale`) while training.
  Stage 2  `LCMModel` (arm kind `lcm`): E and D frozen (loaded from `lcm.ae_ckpt`, a stage-1 run dir). Parcae's
           own looped model (prelude / recurrent block / coda, Poisson depth as the plain arm samples it) runs
           over the sequence of concepts through a linear input projection; its output at span s predicts
           z_{s+1}. `lcm.loss: mse` is Base-LCM (MSE); `lcm.loss: diffusion` is the One-Tower diffusion LCM
           (a small denoiser conditioned on the backbone output, x0-prediction, cosine schedule, DDIM sampler).

No pointer head and no token path between spans: the backbone never reads a token. Evaluation is token CE of
the next span under the frozen D given a concept (`depth_sweep`, `lxtul/lcm_eval.py`).

All span structure is vectorised: the tokens of a row are compacted (slot positions removed, order kept) and
every attention is a flex `span_pack_mask` over the span ids, so there is no Python loop over spans. A slot
position of the packed row carries no information here; any `prefix_k` of the rows gives the same concepts.
Scored set of the stage-2 sweep: the tokens of the COMPLETE spans (closed by a boundary inside the row) with
index >= 1. The first span has no earlier concept to predict it; the open tail after the last boundary is not
a span and is cut by the row end (its last label is the next row's token, which the encoder cannot see).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import lxtul.attention as A
from lxtul import ce
from parcae_lm.modules.utils import precompute_freqs_cis

LCM_KINDS = ("lcm_ae", "lcm")


@dataclass(frozen=True)
class LCMConfig:
    """Every LCM knob (`arm.lcm` in the Hydra arm files)."""
    slot_id: int                     # the <fim_pad> id of the packed row; its logit is masked in every CE
    max_slots: int = 64              # spans per row (the packer's budget)
    cells: int = 1                   # prefix_k of the TRAINING rows (1 = the minimal-TUL rows); not read by the model
    enc_layers: int = 4
    dec_layers: int = 4
    pool_heads: int = 8              # heads of the one-query attention pooling
    z_scale: float = 1.0             # RMS of a concept vector
    ae_noise: float = 0.1            # stage 1: noise std on z, as a fraction of z_scale
    ae_ckpt: str = ""                # stage 2: the stage-1 run dir (holds final.pt)
    loss: str = "mse"                # "mse" | "diffusion"
    den_layers: int = 2              # diffusion denoiser blocks
    train_timesteps: int = 100       # diffusion noise levels in training
    sample_steps: int = 40           # DDIM steps of the sampler
    cosine_s: float = 0.008          # cosine schedule offset
    eval_samples: int = 8            # K concepts sampled per span for the diffusion marginal CE
    eval_seed: int = 0               # seeds the backbone state init and the sampler in every sweep

    def __post_init__(self):
        if self.loss not in ("mse", "diffusion"):
            raise ValueError(f"lcm.loss must be 'mse' or 'diffusion', got {self.loss!r}")
        if self.sample_steps > self.train_timesteps:
            raise ValueError("lcm.sample_steps cannot exceed lcm.train_timesteps")
        if self.ae_noise < 0 or self.z_scale <= 0:
            raise ValueError("lcm.ae_noise must be >= 0 and lcm.z_scale > 0")


# --------------------------------------------------------------------------------------------- row layout
@dataclass
class Compact:
    """The tokens of a packed row, slot positions removed. `bag` is the span index of a token of a COMPLETE
    span and `max_slots` (a dump bin) for the open tail and the padding."""
    x: Tensor        # [B, T] token ids, row order, tokens first
    bag: Tensor      # [B, T] long
    n_tok: Tensor    # [B, S] tokens per span
    valid: Tensor    # [B, S] bool, the span is complete


def compact(x: Tensor, lay, S: int) -> Compact:
    order = torch.sort(lay.slot_mask.to(torch.uint8), dim=1, stable=True).indices   # tokens (0) first, order kept
    xt = x.gather(1, order)
    bag = torch.where(lay.slot_mask.gather(1, order), S, lay.bag_id.gather(1, order))
    n_tok = torch.zeros(x.shape[0], S + 1, dtype=torch.long, device=x.device).scatter_add_(1, bag, torch.ones_like(bag))
    return Compact(xt, bag, n_tok[:, :S], lay.slot_valid)


def span_sums(v: Tensor, bag: Tensor, S: int) -> Tensor:
    """Per-span sum of a per-token quantity: [B, T] -> [B, S]."""
    return torch.zeros(v.shape[0], S + 1, dtype=v.dtype, device=v.device).scatter_add_(1, bag, v)[:, :S]


def tul_target_mask(lay, S: int) -> Tensor:
    """[B, L] bool: the token positions of a packed row whose LABEL is a token of a complete span with index
    >= 1. This is the set the stage-2 sweep scores, expressed on the positions a TUL arm scores, so a TUL
    run's per-position CE can be summed over exactly the same target tokens. A token's label is the next
    token: it belongs to the token's own span, except at the span's last (boundary) token, whose label is
    the first token of the next span. A boundary token is the one followed by a slot position."""
    tok = ~lay.slot_mask
    boundary = tok & torch.cat([lay.slot_mask[:, 1:], torch.zeros_like(tok[:, :1])], 1)
    target_span = lay.bag_id + boundary.long()
    n_spans = lay.slot_valid.sum(1, keepdim=True)
    return tok & (lay.bag_id < S) & (target_span >= 1) & (target_span <= n_spans - 1)


def rows_to(batch, device):
    x, y, lay = batch
    for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
        setattr(lay, f, getattr(lay, f).to(device))
    return x.to(device), y.to(device), lay


def _freeze_ve(blocks) -> None:
    """The backbone and the autoencoder have no value embeddings: drop the gates that would never get a gradient."""
    for b in blocks:
        b.attn.ve_gate = None


def _compile(blocks) -> None:
    for b in blocks:
        b.compile(backend="inductor", mode="default", dynamic=False)


# --------------------------------------------------------------------------------------------- stage 1
class LCMAutoencoder(nn.Module):
    """Concept encoder E and decoder D (stage 1). `forward(x, y, lay)` is the reconstruction loss."""

    def __init__(self, config, lcm: LCMConfig):
        super().__init__()
        d, H = config.n_embd, lcm.pool_heads
        if d % H:
            raise ValueError(f"n_embd {d} not divisible by lcm.pool_heads {H}")
        self.config, self.lcm = config, lcm
        self.emb_scale, self.logit_scale = float(config.init.embedding_scale), float(config.init.logit_scale)
        self.wte_e = nn.Embedding(config.padded_vocab_size, d)
        self.enc_blocks = nn.ModuleList(config.Block(config, layer_id=i) for i in range(lcm.enc_layers))
        self.ln_e = config.Norm(d, eps=config.norm_eps)
        self.pool_k, self.pool_v, self.pool_o = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.pool_query = nn.Parameter(torch.randn(d) / math.sqrt(d // H))        # 1-D: AdamW
        self.wte_d = nn.Embedding(config.padded_vocab_size, d)                    # the output head is tied to it
        self.z_in = nn.Linear(d, d, bias=False)
        self.dec_blocks = nn.ModuleList(config.Block(config, layer_id=i) for i in range(lcm.dec_layers))
        self.ln_d = config.Norm(d, eps=config.norm_eps)
        for emb in (self.wte_e, self.wte_d):
            config.init.apply(emb, "embedding")
        config.init.apply(self.ln_e, "normalization")
        config.init.apply(self.ln_d, "normalization")
        A.use_flex(self.enc_blocks)
        A.use_flex(self.dec_blocks)
        _freeze_ve(self.enc_blocks)
        _freeze_ve(self.dec_blocks)
        head_dim = d // config.num_attention_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            head_dim, config.block_size, config.rope_settings.rope_base, config.rope_settings.rope_condense_ratio),
            persistent=False)
        self.metrics: dict[str, Tensor] = {}

    def compile_blocks(self) -> None:
        _compile(self.enc_blocks)
        _compile(self.dec_blocks)

    # ---- E
    def encode(self, c: Compact) -> Tensor:
        """Concepts [B, S, d] (0 at spans that do not exist). A token reads earlier tokens of its own span only."""
        B, T = c.x.shape
        S, H = self.lcm.max_slots, self.lcm.pool_heads
        h = self.wte_e(c.x) * self.emb_scale
        mask, freqs = A.span_pack_mask(c.bag), self.freqs_cis[:, :T]
        for blk in self.enc_blocks:
            h = blk(h, freqs, mask)
        h = self.ln_e(h)
        dh = h.shape[-1] // H
        f32 = torch.float32
        k = self.pool_k(h).view(B * T, H, dh)
        v = self.pool_v(h).view(B * T, H, dh).to(f32)
        s = (k * self.pool_query.view(H, dh)).sum(-1).to(f32) / math.sqrt(dh)                  # [BT, H]
        flat = (torch.arange(B, device=h.device)[:, None] * (S + 1) + c.bag).reshape(-1)         # bin of each token
        with torch.no_grad():                                                                    # softmax stabiliser
            m = torch.full((B * (S + 1), H), float("-inf"), dtype=f32, device=h.device)
            m = m.scatter_reduce(0, flat[:, None].expand(-1, H), s, "amax", include_self=True)
        e = (s - m[flat]).exp()
        den = torch.zeros(B * (S + 1), H, dtype=f32, device=h.device).index_add_(0, flat, e)
        w = e / den[flat]
        pooled = torch.zeros(B * (S + 1), H, dh, dtype=f32, device=h.device).index_add_(0, flat, w[..., None] * v)
        z = self.pool_o(pooled.view(B, S + 1, H * dh)[:, :S].to(h.dtype)).to(f32)
        return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-12) * self.lcm.z_scale

    # ---- D
    def _dest(self, c: Compact):
        """Where each compact token (`dest_tok`, the sink T + S for tokens of no complete span) and each span's
        concept position (`dest_z`) land in the sequence [z_0, tokens of span 0, z_1, tokens of span 1, ...]."""
        B, T = c.x.shape
        S = self.lcm.max_slots
        start = c.n_tok.cumsum(1) - c.n_tok
        ar = torch.arange(T, device=c.x.device)
        dest_tok = torch.where(c.bag < S, ar + c.bag + 1, T + S)
        dest_z = start + torch.arange(S, device=c.x.device)
        return dest_tok, dest_z

    def token_hidden(self, c: Compact, z: Tensor) -> Tensor:
        """[B, T, d]: the decoder state that predicts each compact token. D sees the span's concept and the
        span's earlier tokens only (a flex mask over the span id). `z` [B, S, d] is any concept, true or predicted."""
        B, T = c.x.shape
        S, d = self.lcm.max_slots, z.shape[-1]
        Tp = T + S
        dest_tok, dest_z = self._dest(c)
        ar_t = torch.arange(T, device=c.x.device).expand(B, T)
        ar_s = torch.arange(S, device=c.x.device).expand(B, S)
        src = torch.full((B, Tp + 1), Tp, dtype=torch.long, device=c.x.device)
        src = src.scatter(1, dest_tok, ar_t).scatter(1, dest_z, T + ar_s)
        emb = self.wte_d(c.x) * self.emb_scale
        seq = torch.cat([emb, self.z_in(z).to(emb.dtype), emb.new_zeros(B, 1, d)], 1)
        h = seq.gather(1, src[:, :Tp, None].expand(-1, -1, d))
        doc = torch.full((B, Tp + 1), S, dtype=torch.long, device=c.x.device)
        doc = doc.scatter(1, dest_tok, c.bag).scatter(1, dest_z, torch.where(c.valid, ar_s, S))
        mask, freqs = A.span_pack_mask(doc[:, :Tp]), self.freqs_cis[:, :Tp]
        for blk in self.dec_blocks:
            h = blk(h, freqs, mask)
        h = self.ln_d(h)
        return h.gather(1, (dest_tok - 1).clamp_min(0)[..., None].expand(-1, -1, d))

    def _labels(self, c: Compact) -> Tensor:
        return torch.where(c.bag < self.lcm.max_slots, c.x, -100)

    def token_nll(self, c: Compact, z: Tensor) -> Tensor:
        """Per-token CE [B, T] of the tokens of complete spans given their concepts (0 elsewhere)."""
        h = self.token_hidden(c, z)
        nll = ce.linear_ce_tokens(h.flatten(0, 1), self.wte_d.weight, self._labels(c).flatten(),
                                  self.logit_scale, self.lcm.slot_id)
        return nll.view(c.x.shape)

    def forward(self, x: Tensor, y: Tensor | None, lay, depth: int | None = None) -> dict:
        c = compact(x, lay, self.lcm.max_slots)
        z = self.encode(c)
        if self.training and self.lcm.ae_noise > 0:
            z = z + torch.randn_like(z) * (self.lcm.ae_noise * self.lcm.z_scale)
        h = self.token_hidden(c, z)
        loss = ce.linear_ce(h.flatten(0, 1), self.wte_d.weight, self._labels(c).flatten(),
                            self.logit_scale, self.lcm.slot_id)
        self.metrics = {"recon_ce": loss.detach()}
        return {"loss": loss, "ce": loss.detach()}

    @torch.no_grad()
    def depth_sweep(self, rows, depths, device="cuda") -> dict:
        """Autoencoder ceiling on every complete span (the first too), no noise. There is no depth: the
        same sums are returned for each requested depth, as the flat gpt arm does."""
        was = self.training
        self.eval()
        S = self.lcm.max_slots
        sums, counts = [], []
        for batch in rows:
            x, _, lay = rows_to(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                c = compact(x, lay, S)
                nll = span_sums(self.token_nll(c, self.encode(c)), c.bag, S)
            w = c.valid.to(nll.dtype)
            sums.append((nll * w).sum(1).double().cpu())
            counts.append((c.n_tok * c.valid).sum(1).double().cpu())
        self.train(was)
        s = torch.cat(sums).tolist()
        return {"depths": list(depths), "count": torch.cat(counts).tolist(), "sums": {str(k): s for k in depths}}


# --------------------------------------------------------------------------------------------- backbone
class ConceptBackbone(nn.Module):
    """Parcae's looped model with the token embedding, the LM head and the value embeddings removed. It is
    the stock `Parcae` object (its blocks, injection, C, init and depth sampler); only the front door
    differs: `forward` takes input embeddings. The plain arm's code is untouched."""

    def __init__(self, config):
        super().__init__()
        core = config.construct_model(gradient_checkpointing=False)
        del core.transformer["wte"], core.lm_head
        core.value_embeds = nn.ModuleDict()
        core._current_input_ids = None            # read by core_block_forward; there are no ids and no ve
        t = core.transformer
        _freeze_ve([*t.prelude, *t.core_block, *t.coda])
        self.core = core

    def compile_blocks(self) -> None:
        t = self.core.transformer
        for blocks in (t.prelude, t.core_block, t.coda):
            _compile(blocks)

    def forward(self, h: Tensor, depth: int | None = None) -> Tensor:
        """[B, S, d] input embeddings -> [B, S, d] states after the final norm. Causal over S. `depth` None:
        Parcae's training draw; an int: that many core passes (the sweep)."""
        core = self.core
        t = core.transformer
        freqs = core.freqs_cis[:, : h.shape[1]]
        for blk in t.prelude:
            h = blk(h, freqs, None)
        if core.config.prelude_norm:
            h = t.ln_prelude(h)
        pair = None if depth is None else torch.tensor([depth, 0], device=h.device)
        x, _, _, _ = core.iterate_forward(h, freqs, None, pair)
        x = t.C(x)
        for blk in t.coda:
            x = blk(x, freqs, None)
        return t.ln_f(x)


# --------------------------------------------------------------------------------------------- diffusion head
def _timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.to(torch.float32)[:, None] * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


class _AdaMLP(nn.Module):
    """One denoiser layer: adaLN-zero modulated ReLU^2 MLP (the context enters through the modulation)."""

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.ada = nn.Linear(d, 3 * d, bias=False)
        self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
        nn.init.zeros_(self.ada.weight)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        shift, scale, gate = self.ada(F.silu(c)).chunk(3, -1)
        y = F.rms_norm(h, (h.shape[-1],)) * (1 + scale) + shift
        return h + gate * self.down(F.relu(self.up(y)).square())


class Denoiser(nn.Module):
    """The One-Tower LCM's denoiser, position-wise: x0-prediction of the next concept from its noised copy,
    the noise level and the backbone state at the previous span. Cosine schedule over `train_timesteps`."""

    def __init__(self, d: int, ff: int, lcm: LCMConfig):
        super().__init__()
        self.d, self.T, self.steps = d, lcm.train_timesteps, lcm.sample_steps
        self.cond_in, self.z_in = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.tproj1, self.tproj2 = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList(_AdaMLP(d, ff) for _ in range(lcm.den_layers))
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)
        i = torch.arange(lcm.train_timesteps + 1, dtype=torch.float64) / lcm.train_timesteps
        ab = torch.cos((i + lcm.cosine_s) / (1 + lcm.cosine_s) * math.pi / 2).square()
        self.register_buffer("alpha_bar", (ab / ab[0]).clamp_min(1e-5).float(), persistent=False)

    def forward(self, cond: Tensor, zt: Tensor, t: Tensor) -> Tensor:
        """x0 prediction. cond, zt [N, d]; t [N] integer levels in [1, T]."""
        c = self.cond_in(cond) + self.tproj2(F.silu(self.tproj1(_timestep_embedding(t, self.d).to(cond.dtype))))
        h = self.z_in(zt.to(cond.dtype))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(F.rms_norm(h, (self.d,)))

    def loss(self, cond: Tensor, x0: Tensor) -> Tensor:
        """Per-row MSE of the x0 prediction at a uniformly drawn level: [N]."""
        t = torch.randint(1, self.T + 1, (x0.shape[0],), device=x0.device)
        ab = self.alpha_bar[t][:, None]
        zt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        return (self(cond, zt, t).float() - x0).square().mean(-1)

    @torch.no_grad()
    def sample(self, cond: Tensor, generator: torch.Generator) -> Tensor:
        """Deterministic DDIM sampler (eta 0), `sample_steps` steps from pure noise: [N, d]."""
        ts = torch.linspace(self.T, 0, self.steps + 1).round().long().tolist()
        zt = torch.randn(cond.shape, device=cond.device, dtype=torch.float32, generator=generator)
        for t, t_next in zip(ts[:-1], ts[1:]):
            x0 = self(cond, zt, torch.full((cond.shape[0],), t, device=cond.device)).float()
            ab, ab_n = self.alpha_bar[t], self.alpha_bar[t_next]
            eps = (zt - ab.sqrt() * x0) / (1 - ab).sqrt()
            zt = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
        return zt


# --------------------------------------------------------------------------------------------- stage 2
class LCMModel(nn.Module):
    """The concept backbone over a frozen autoencoder (stage 2)."""

    def __init__(self, ae_config, bb_config, lcm: LCMConfig):
        super().__init__()
        if ae_config.n_embd != bb_config.n_embd:
            raise ValueError("the backbone width must equal the concept width (n_embd)")
        self.lcm = lcm
        d = bb_config.n_embd
        self.ae = LCMAutoencoder(ae_config, lcm).requires_grad_(False).eval()
        self.backbone = ConceptBackbone(bb_config)
        self.in_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.in_proj.weight, std=bb_config.init.get_std("embedding") / math.sqrt(d))   # embedding-sized input
        if lcm.loss == "mse":
            self.out_proj = nn.Linear(d, d, bias=False)
        else:
            self.denoiser = Denoiser(d, bb_config.intermediate_size, lcm)
        self.step = 0
        self.metrics: dict[str, Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.ae.eval()                       # E and D never train and never add noise
        return self

    def compile_blocks(self) -> None:
        self.ae.compile_blocks()
        self.backbone.compile_blocks()

    def concepts(self, x: Tensor, lay):
        with torch.no_grad():
            c = compact(x, lay, self.lcm.max_slots)
            return c, self.ae.encode(c)

    def states(self, z: Tensor, depth: int | None) -> Tensor:
        """Backbone state at span s (reads concepts 0..s): [B, S, d]."""
        self.backbone.core.step = self.step
        return self.backbone(self.in_proj(z).to(torch.float32), depth)       # fp32 residual stream, as plain Parcae's

    def forward(self, x: Tensor, y: Tensor | None, lay, depth: int | None = None) -> dict:
        _, z = self.concepts(x, lay)
        cond, tgt = self.states(z, depth)[:, :-1], z[:, 1:]               # state at s predicts concept s+1
        w = lay.slot_valid[:, 1:].to(torch.float32)
        n = w.sum().clamp_min(1)
        if self.lcm.loss == "mse":
            per = (self.out_proj(cond).float() - tgt).square().mean(-1)
        else:
            per = self.denoiser.loss(cond.flatten(0, 1), tgt.flatten(0, 1)).view(w.shape)
        loss = (per * w).sum() / n
        self.metrics = {"loss": loss.detach(), "null_mse": ((tgt.square().mean(-1) * w).sum() / n).detach()}
        return {"loss": loss, "ce": loss.detach()}

    @torch.no_grad()
    def predict(self, cond: Tensor, gen: torch.Generator | None) -> Tensor:
        """Concept for the NEXT span from the state at each span: [B, S, d]."""
        if self.lcm.loss == "mse":
            return self.out_proj(cond).float()
        return self.denoiser.sample(cond.flatten(0, 1), gen).view(cond.shape)

    @torch.no_grad()
    def depth_sweep(self, rows, depths, device="cuda") -> dict:
        """Token CE of span s+1 under the frozen D, for every row, per backbone depth. Scored set: the tokens
        of the complete spans with index >= 1. Keys: `sums[k]` (mse: CE given the predicted concept; diffusion:
        the marginal -log mean_K p(span | z_k)), `sums_sample_mean[k]` (diffusion: mean of the per-sample CE),
        `oracle` (D on the true concept), `null` (D on a zero concept), `count` (scored tokens per row), `scope`."""
        was = self.training
        self.eval()
        lc, S = self.lcm, self.lcm.max_slots
        diff = lc.loss == "diffusion"
        acc = {"count": [], "oracle": [], "null": [], "sums": {k: [] for k in depths},
               "sums_sample_mean": {k: [] for k in depths}, "in_x": [], "tail": [], "first": [], "complete": []}
        dev = torch.device(device)
        cuda = dev.type == "cuda"
        with torch.random.fork_rng(devices=[dev.index if dev.index is not None else torch.cuda.current_device()]
                                   if cuda else []):
            for bi, batch in enumerate(rows):
                x, _, lay = rows_to(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    c, z = self.concepts(x, lay)
                    w = c.valid.clone()
                    w[:, 0] = False                                               # span 0 has no earlier concept
                    wf = w.to(torch.float32)
                    nll = lambda zz: span_sums(self.ae.token_nll(c, zz), c.bag, S)
                    acc["oracle"].append((nll(z) * wf).sum(1).double().cpu())
                    acc["null"].append((nll(torch.zeros_like(z)) * wf).sum(1).double().cpu())
                    acc["count"].append((c.n_tok * w).sum(1).double().cpu())
                    acc["in_x"].append((~lay.slot_mask).sum(1).cpu())
                    acc["tail"].append(((lay.bag_id == S) & ~lay.slot_mask).sum(1).cpu())
                    acc["first"].append((c.n_tok[:, 0] * c.valid[:, 0]).cpu())
                    acc["complete"].append((c.n_tok * c.valid).sum(1).cpu())
                    for k in depths:
                        torch.manual_seed(lc.eval_seed * 1_000_003 + bi)         # same state init at every depth
                        cond = self.states(z, k)
                        gen = torch.Generator(device=dev).manual_seed(lc.eval_seed * 7919 + bi) if diff else None
                        if diff:
                            nlls = []
                            for _ in range(lc.eval_samples):
                                zhat = self.predict(cond, gen)
                                nlls.append(nll(torch.cat([torch.zeros_like(zhat[:, :1]), zhat[:, :-1]], 1)))
                            nlls = torch.stack(nlls)                              # [K, B, S]
                            marg = -(torch.logsumexp(-nlls, 0) - math.log(lc.eval_samples))
                            acc["sums"][k].append((marg * wf).sum(1).double().cpu())
                            acc["sums_sample_mean"][k].append((nlls.mean(0) * wf).sum(1).double().cpu())
                        else:
                            zhat = self.predict(cond, gen)
                            nl = nll(torch.cat([torch.zeros_like(zhat[:, :1]), zhat[:, :-1]], 1))
                            acc["sums"][k].append((nl * wf).sum(1).double().cpu())
        self.train(was)
        cat = lambda v: torch.cat(v).tolist()
        in_x, tail, first, count = (torch.cat(acc[k]) for k in ("in_x", "tail", "first", "count"))
        if not torch.equal(in_x, count.long() + first + tail):
            raise RuntimeError("tokens of the rows != scored + first span + open tail")
        out = {"depths": list(depths), "count": count.tolist(), "oracle": cat(acc["oracle"]), "null": cat(acc["null"]),
               "sums": {str(k): cat(v) for k, v in acc["sums"].items()},
               "scope": {"rows": int(in_x.numel()), "tokens_in_x": int(in_x.sum()), "scored": int(count.sum()),
                         "dropped_first_span": int(first.sum()), "dropped_open_tail": int(tail.sum())}}
        if diff:
            out["sums_sample_mean"] = {str(k): cat(v) for k, v in acc["sums_sample_mean"].items()}
        return out


# --------------------------------------------------------------------------------------------- build
def lcm_config(cfg, mcfg, rt) -> LCMConfig:
    """`arm.lcm` (+ the packer's slot id and slot budget; `rt` is MORPH's TUL runtime). An unknown key raises."""
    from omegaconf import OmegaConf
    kw = OmegaConf.to_container(cfg.arm.lcm, resolve=True)
    return LCMConfig(slot_id=int(rt.data_cfg.slot_id), max_slots=int(mcfg.tul.max_slots), **kw)


def load_ae(ae: LCMAutoencoder, run_dir: str, lc: LCMConfig) -> None:
    """Load the stage-1 weights of `run_dir/final.pt`; refuse a checkpoint of another architecture."""
    path = Path(run_dir) / "final.pt"
    if not path.is_file():
        raise FileNotFoundError(f"lcm.ae_ckpt: {path} does not exist (run the stage-1 arm first)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arm = ck["config"]["arm"]
    if arm["kind"] != "lcm_ae":
        raise ValueError(f"{path} is a {arm['kind']!r} run, not an lcm_ae run")
    for k in ("enc_layers", "dec_layers", "pool_heads", "z_scale"):
        if arm["lcm"][k] != getattr(lc, k):
            raise ValueError(f"lcm.{k}: stage 2 has {getattr(lc, k)}, the autoencoder was trained with {arm['lcm'][k]}")
    ae.load_state_dict({k.replace("._orig_mod", ""): v for k, v in ck["model"].items()}, strict=True)


def build_lcm(cfg, mcfg, arm: str, compile: bool, load: bool = True):
    """-> (model on cpu, tul runtime). `arm` in LCM_KINDS. The autoencoder is sized for the packed row
    (RoPE table of tokens + spans), the backbone for a sequence of `max_slots` concepts. `load` False skips
    reading `lcm.ae_ckpt` (the eval path loads the whole stage-2 state dict instead)."""
    from lxtul.build import parcae_config
    from lxtul.data import layout_for_cells, packed_length, tul_runtime
    rt = tul_runtime(mcfg)
    lc = lcm_config(cfg, mcfg, rt)
    rt = layout_for_cells(rt, lc.cells)
    vocab = int(mcfg.model.vocab_size)
    ae_cfg = parcae_config(cfg, vocab, plain=False, block_size=packed_length(mcfg, rt) + lc.max_slots)
    if arm == "lcm_ae":
        model = LCMAutoencoder(ae_cfg, lc)
    elif arm == "lcm":
        model = LCMModel(ae_cfg, parcae_config(cfg, 1, plain=True, block_size=lc.max_slots), lc)
        if load:
            load_ae(model.ae, lc.ae_ckpt, lc)
    else:
        raise ValueError(f"unknown lcm arm {arm!r}")
    if compile:
        model.compile_blocks()
    return model, rt
