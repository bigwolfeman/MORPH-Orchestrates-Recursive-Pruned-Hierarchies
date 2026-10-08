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

from lxtul.attention import coda_mask, core_mask, prelude_mask, use_flex
from lxtul.cache import copy_features, mix_logp, packed_copy_probs
from lxtul.copy_heads import IdentityReach, PointerHead, mix_pointer
from lxtul.ce import linear_ce, linear_ce_tokens


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
    injection: str = "full"            # "full": Parcae's diagonal injection on every channel;
    inj_lo: int = 512                  # "ctx": MORPH's, A*h + dt*e on [inj_lo, inj_hi) only,
    inj_hi: int = 832                  #   A init 0.447, dt 1, identity elsewhere (3.2)

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
        if tul.cells > 1 and tul.select == "latent" and not tul.lsel_on:
            raise ValueError("select=latent with cells > 1 needs lsel_lambda > 0 to pick a winner")
        del self.transformer["C"]          # W_prefix writes the loop state into the coda (4)
        self.tul = tul
        d, M, S = config.n_embd, tul.cells, tul.max_slots
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
        if tul.repel_on:
            # epivol reservoir (3.7): frozen, seed 0
            g = torch.Generator().manual_seed(0)
            self.register_buffer("epi_W1", torch.randn(tul.epi_hidden, d, generator=g) / math.sqrt(d))
            self.register_buffer("epi_W2", torch.randn(tul.epi_features, tul.epi_hidden, generator=g)
                                 / math.sqrt(tul.epi_hidden))
        self._core_mask = None
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

    # ------------------------------------------------------------------ pieces
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

    def _gain_penalty(self, h: Tensor, e: Tensor, freqs, ves, m: Tensor) -> Tensor:
        """100 * mean_b relu(gain_b - 0.9)^2 on the RAW map (no cell norm), sources detached (5.6)."""
        tul = self.tul
        hp, e = h.detach(), e.detach()
        mf = m[..., None].to(hp.dtype)
        v = torch.randn_like(hp) * mf
        scale = tul.gain_eps * hp.float().norm(dim=-1) / v.float().norm(dim=-1).clamp_min(1e-12)
        dlt = v * scale[..., None].to(hp.dtype)
        step = lambda x: self._core_step(x, e, freqs, ves)
        f0 = checkpoint(step, hp, use_reentrant=False)
        f1 = checkpoint(step, hp + dlt, use_reentrant=False)
        num = ((f1 - f0).float() * mf).flatten(1).norm(dim=1)
        gain = num / (dlt.float().flatten(1).norm(dim=1) + 1e-6)
        self.metrics["gain"] = gain.detach().mean()
        return tul.gain_lambda * F.relu(gain - tul.gain_target).pow(2).mean()

    def _epivol(self, traj: list, valid: Tensor) -> Tensor:
        """-(mean_t epi_t) - (mean_t vol_t) over passes 1..repel_passes (3.7)."""
        tul, M = self.tul, self.tul.cells
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

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids: Tensor, labels: Tensor | None, layout, depth: int | None = None,
                **_ignored) -> dict:
        """`layout` is MORPH's SlotLayout. `depth` forces every slot to that many passes
        (the K-sweep); None draws Poisson depth in training and uses eval_depth in eval."""
        tul, M, S = self.tul, self.tul.cells, self.tul.max_slots
        B, L = input_ids.shape
        d = self.config.n_embd
        dev = input_ids.device
        train = self.training
        is_slot, bag = layout.slot_mask, layout.bag_id
        sidx, valid = layout.slot_index, layout.slot_valid
        if sidx.shape[1] != S:
            raise ValueError(f"layout has {sidx.shape[1]} slots, model was built for {S}")
        if self._core_mask is None:
            self._core_mask = core_mask(S * M, M, dev)
        op = tul.geometry == "open"
        pmask, cmask = prelude_mask(bag, is_slot, op), coda_mask(bag, is_slot, op)
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
        att = torch.einsum("md,bsjd->bsmj", self.reg_q_embed.to(k.dtype), k).float() / math.sqrt(d)
        att = att.masked_fill(~own[:, :, None], float("-inf"))
        att = torch.where(own.any(-1)[:, :, None, None], att, torch.zeros_like(att)).softmax(-1)
        reg = self.reg_o(torch.einsum("bsmj,bsjd->bsmd", att.to(v.dtype), v)) + self.reg_p_cell_embed
        reg = reg * vmask[..., None]

        # --- loop entry (1.5) ---
        e = xn.gather(1, cpos.reshape(B, -1, 1).expand(-1, -1, d)).view(B, S, M, d) * vmask[..., None]
        e = (e + reg.to(e.dtype)).view(B, S * M, d)
        h = e
        ctx = xn.gather(1, sidx[..., None].expand(-1, -1, d)).detach()       # [B,S,d]
        slot_ids = torch.full((B, S * M), tul.slot_id, device=dev, dtype=torch.long)
        off = self.config.n_layers_in_prelude
        ves = [self._ve(off + i, slot_ids) for i in range(len(self.transformer.core_block))]

        # --- depth (3.1) ---
        if depth is not None:
            dslot = torch.full((B, S), int(depth), device=dev)
        elif train:
            dslot = torch.poisson(torch.full((B, S), float(tul.mean_depth), device=dev))
            dslot = dslot.clamp(1, tul.max_depth).long()
        else:
            dslot = torch.full((B, S), tul.eval_depth, device=dev)
        dslot = torch.where(valid, dslot, torch.ones_like(dslot))
        dcell = dslot.repeat_interleave(M, dim=1)
        vcell = valid.repeat_interleave(M, dim=1)
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
        step = lambda hh, ee: self._core_step(hh, ee, cfreqs, ves)
        for t in range(T):
            active = dcell > t
            if train and t < tul.ckpt_passes:
                h_new = checkpoint(step, h, e, use_reentrant=False)
            else:
                h_new = step(h, e)
            if t == t_gain:
                gain_pen = self._gain_penalty(h, e, cfreqs, ves, active & vcell)
            h_new = self.cell_norm(h_new)
            if train and h_new.requires_grad and tul.cot_clip > 0:
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
        if train and h.requires_grad and tul.cot_clip > 0:
            h.register_hook(lambda g: ref.__setitem__("r", g.float().flatten(1).norm(dim=1)))

        # --- write: the final winner alone, through W_prefix[winner] (4) ---
        hv = h.view(B, S, M, d)
        if tul.select == "none":      # ungraded fan: every cell into its own prefix position
            onehot = torch.ones(B, S, M, device=dev, dtype=h.dtype) * vmask
        else:
            onehot = F.one_hot(final, M).to(h.dtype) * vmask
        Wp = torch.stack(list(self.W_prefix)).to(h.dtype)                  # [M,d,d]
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
        if tul.copy_cache or tul.pointer_heads > 0:
            ce_t = linear_ce_tokens(xc.flatten(0, 1), W, tok_labels.flatten(),
                                    self.config.init.logit_scale, tul.slot_id).view(B, L)
            lp = self._out_mix(xc, -ce_t, input_ids, labels, is_slot, layout)
            tok_ok = tok_labels >= 0                     # not `ok`: the latent-selection loss reads that
            n_ok = tok_ok.sum().clamp_min(1)
            ce = -(lp * tok_ok).sum() / n_ok
            out["ce_model"] = ((ce_t * tok_ok).sum() / n_ok).detach()
        else:
            ce = linear_ce(xc.flatten(0, 1), W, tok_labels.flatten(), self.config.init.logit_scale,
                           tul.slot_id)
        out["ce"] = ce
        loss = ce
        if tul.spandec_on:
            sd = self._spandec(hv, input_ids, npos, nok, W)
            out["spandec"] = sd
            loss = loss + tul.spandec_weight * sd
        if tul.lsel_on:
            l_lat, l_enc, l_rce = self._lsel_exit(hv, z, ok, zo, rce_sum, rce_n)
            loss = loss + tul.lsel_lambda * l_lat + tul.lsel_enc_lambda * l_enc + tul.lsel_router_lambda * l_rce
            self.metrics.update(l_lat=l_lat.detach(), l_enc=l_enc.detach(), l_rce=l_rce.detach())
        if train:
            loss = loss + gain_pen
            if tul.fixed_point_lambda > 0:
                loss = loss + tul.fixed_point_lambda * fp_sum / fp_n.clamp_min(1)
            if tul.repel_on and len(traj) > 1:
                loss = loss + tul.repel_lambda * self._epivol(traj, valid)
        out["loss"] = loss
        self.metrics.update(fp=(fp_sum / fp_n.clamp_min(1)).detach(), passes=torch.tensor(T))
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

    def _spandec(self, hv, input_ids, npos, nok, W):
        """Teacher-forced decoder of span s+1 from the mean of slot s's 4 final cells (5.3)."""
        tul = self.tul
        B, S, M, d = hv.shape
        # --- span decoder (5.3): mean of the 4 final raw cells ---
        zsd = self.transformer.ln_f(hv.mean(2))                             # [B,S,d]
        tgt = input_ids.gather(1, npos.flatten(1)).view_as(npos)           # [B,S,J]
        tin = self.transformer.wte(tgt).detach() * self.emb_scale
        tin = self.sd_tok_in(tin) * nok[..., None]
        sx = torch.cat([self.sd_z_in(zsd)[:, :, None], tin[:, :, :-1]], 2).flatten(0, 1)
        sfreq = self.freqs_cis[:, : tul.span_cap]
        for block in self.sd_blocks:
            sx = block(sx, sfreq, None)
        sx = self.sd_out_norm(sx)
        sd_lab = torch.where(nok, tgt, torch.full_like(tgt, -100)).flatten()
        sd = linear_ce(sx.flatten(0, 1), W.detach(), sd_lab, self.config.init.logit_scale, tul.slot_id)
        return sd

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
        zv = zo[ok]
        l_enc = F.relu(tul.lsel_enc_gamma - zv.std(0) if zv.shape[0] > 1 else zv.new_zeros(d)).mean()
        l_rce = rce_sum / rce_n.clamp_min(1)

        return l_lat, l_enc, l_rce
