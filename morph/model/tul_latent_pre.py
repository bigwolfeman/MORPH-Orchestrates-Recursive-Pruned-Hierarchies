"""Stage 1 of the staged latent pretraining plan (``tul.latent_pre_target``, 2026-10-02).

The slot loop and the prelude train on a LATENT objective alone: no coda, no token CE, at
train AND at val. A predictor head ``g`` (:class:`~morph.model.tul_fan_route.FanLatentHead`,
the latent-selected loop's head, reused) reads the loop's exit cells and is scored against
a target computed from the NEXT span (slot ``s`` -> span ``s + 1``, the bins of
:func:`~morph.model.transformer.span_ce_index`). Note:
``.agents/notes/proposed/architecture/2026-10-02-staged-latent-pretraining.md``.

THE TARGETS (``tul.latent_pre_target``):

``plain_prelude``  a FROZEN plain model (the 5k plain checkpoint, built by
    ``morph/training/latent_pre_ref.py``, attached with
    ``MORPHTransformer.tul_latent_pre_attach_ref``). Its own front (embeddings + its
    prelude blocks, ``_front_region``) runs on EACH next span's tokens as ITS OWN
    sequence, the HC streams are averaged, the token states mean-pooled and LayerNormed
    without affine — the pooling of :func:`~morph.model.tul_fan_route.pooled_span_states`.
    SPAN-LOCAL BY CONSTRUCTION: slot ``s``'s target is a function of span ``s + 1``'s
    tokens and nothing else (pinned by perturbation in ``tests/test_tul_latent_pre.py``).
    Why not the plain model's causal prelude over the whole row: that state attends to
    every earlier span, so the target of slot ``s`` would carry the prefix the loop
    already reads, and same-row retrieval could be won by matching the shared prefix
    instead of predicting span ``s + 1`` (the note's S1-D risk, applied to S1-A).
``plain_final``  the SAME frozen plain model's FINAL hidden state (``_back_region``'s
    output, the state its LM head reads) at the LAST token of span ``s + 1``, run on the
    PLAIN token row (the packed row with every slot position removed, tokens in order,
    right-padded to the packed length), LayerNormed without affine. Causal over the whole
    row: it reads the past (the note: compare it on same-row negatives).
``ema_prelude``  the latent-selected loop's target (``MORPHTransformer._tul_fan_target``):
    the LIVE model's EMA prelude twin under the strict prelude, the collapse control.

THE LOSSES (``tul.latent_pre_loss``): ``l2`` per-coordinate MSE to the target (the
conditional mean); ``infonce`` cosine logits / ``tau`` against EVERY target slot in the
batch (same-row and cross-row candidates in one softmax), own span the class; ``wta``
M = ``fan_k`` cells, the relaxed winner-takes-all of :func:`lsel_exit_loss` over the
per-cell L2 (eps ``tul.latent_pre_eps``).

THE NORMALISATION (``tul.latent_pre_target_norm``): ``ln`` is the LayerNormed target
above. ``standard`` (the default for the two FROZEN targets) is that target minus a FIXED
per-coordinate mean ``mu``, over a FIXED per-coordinate std ``sigma`` (:func:`standardise`).
``mu`` and ``sigma`` come from :class:`TargetMoments` over a calibration set, ONCE, at build
(``morph/training/latent_pre_ref.py::calibrate_latent_pre_ref``), and live on the frozen
model as non-persistent buffers. Why: the LayerNormed plain prelude is ~98 % one common
direction (four channels at |mean| ~11; mean pairwise cosine 0.98, centred participation
ratio 3.5-5.3, per-coordinate variance ~0.02), so L2 on it mostly rewards predicting a
constant. Standardised, the mean predictor's L2 is ~1.0 and L2 ~ 1 - R^2.

All math here runs in fp32 with autocast off. Nothing here has parameters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .tul_fan_route import (batch_var, cell_spread, lsel_distance, lsel_exit_loss,
                            participation_ratio)

LATENT_PRE_TARGETS = ("off", "plain_prelude", "plain_final", "ema_prelude")
LATENT_PRE_TARGET_NORMS = ("standard", "ln")
LATENT_PRE_LOSSES = ("l2", "infonce", "wta")


def _ln(x: Tensor) -> Tensor:
    """LayerNorm over the last axis, no affine, fp32 (the target space)."""
    return F.layer_norm(x.float(), (x.shape[-1],))


# ── where the next span is ───────────────────────────────────────────────────────────


def next_span_grid(input_ids: Tensor, labels: Tensor, layout) -> dict:
    """Slot ``s``'s NEXT span (span ``s + 1``) laid out per slot.

    Returns ``{"ids" [B, S, W] int64, "valid" [B, S, W] bool, "ok" [B, S] bool,
    "last" [B, S] int64}``: the input ids of span ``s + 1`` left-aligned (pad 0 past its
    length), which entries are tokens, the slots with a target (``slot_valid`` and at
    least one scored token in bin ``s + 1`` — :meth:`_tul_fan_target`'s ``ok``), and the
    PACKED row position of span ``s + 1``'s last token (0 where ``ok`` is False).

    The tokens are exactly :func:`span_ce_index`'s bin ``s + 1`` (a real TOKEN with a
    label), so this grid, :func:`pooled_span_states` and the coda's span-CE table name
    the same positions. Span ``s + 1`` starts right after slot ``s``'s ``prefix_k``
    positions (the packer's layout, the ``fan_head_wta_targets`` arithmetic); a position
    whose offset falls outside ``[0, W)`` would mean a non-contiguous span and RAISES.
    ``W`` is the longest next span in the batch, rounded up to a multiple of 8 (one host
    sync per call; at most ``span_cap / 8`` distinct shapes).
    """
    from .transformer import span_ce_index, span_token_counts   # circular at module load
    gid, keep_tok, _lab, G = span_ce_index(labels, layout)
    B, L = labels.shape
    S = G - 1
    bag = layout.bag_id.clamp(0, S)
    slot = (bag - 1).clamp(min=0)
    scored = keep_tok & (bag >= 1) & torch.gather(layout.slot_valid, 1, slot)
    start = torch.gather(layout.slot_index, 1, slot) + int(layout.prefix_k)
    j = torch.arange(L, device=labels.device).view(1, L) - start
    n_tok = span_token_counts(gid, keep_tok, G)[:, 1:].round().long()          # [B, S]
    ok = layout.slot_valid & (n_tok > 0)
    jmax, jmin = [int(v) for v in torch.stack([
        torch.where(scored, j + 1, torch.zeros_like(j)).max(),
        torch.where(scored, j, torch.zeros_like(j)).min()]).tolist()]
    if jmin < 0:
        raise RuntimeError(
            "next_span_grid: a token of span s+1 sits BEFORE the first position after "
            "slot s's prefix cells. The packed layout is not the packer's (spans must be "
            "contiguous and follow their slot).")
    W = max(8, -(-jmax // 8) * 8)
    lin = torch.arange(B, device=labels.device).view(B, 1) * (S * W) + slot * W + j
    lin = torch.where(scored, lin.clamp(0, B * S * W - 1),
                      torch.full_like(lin, B * S * W))
    ids = torch.zeros(B * S * W + 1, dtype=torch.long, device=labels.device)
    ids.scatter_(0, lin.reshape(-1), torch.where(scored, input_ids, torch.zeros_like(
        input_ids)).reshape(-1))
    valid = torch.zeros(B * S * W + 1, dtype=torch.bool, device=labels.device)
    valid.scatter_(0, lin.reshape(-1), scored.reshape(-1))
    span_start = layout.slot_index + int(layout.prefix_k)                      # [B, S]
    last = torch.where(ok, span_start + n_tok - 1, torch.zeros_like(n_tok))
    return {"ids": ids[:-1].view(B, S, W), "valid": valid[:-1].view(B, S, W), "ok": ok,
            "last": last, "n_tok": n_tok}


def plain_token_row(input_ids: Tensor, slot_mask: Tensor) -> tuple[Tensor, Tensor]:
    """``(plain_ids [B, L], tok_rank [B, L])``: the packed row with every slot position
    (tail pads included) moved behind the tokens, tokens in their original order, and the
    PLAIN index of every packed position (meaningful at token positions only).

    The moved slot ids sit after the row's last real token, so a CAUSAL plain model reads
    every real token exactly as it would on the unpadded token row; the fixed width ``L``
    (the packed length) keeps one shape per run."""
    order = torch.sort(slot_mask.to(torch.int8), dim=1, stable=True).indices
    plain_ids = torch.gather(input_ids, 1, order)
    tok_rank = torch.cumsum((~slot_mask).to(torch.long), dim=1) - 1
    return plain_ids, tok_rank


# ── the frozen plain model's targets ─────────────────────────────────────────────────


def assert_padding_invariant(ref, length: int) -> None:
    """Both plain targets RIGHT-PAD (the span grid to ``W``, the plain row to the packed
    length). Causal attention makes that harmless EXCEPT for CSA's top-k block selection:
    ``tk = min(top_k, n_blocks)`` and ``torch.topk``'s tie-breaking among equal-scored
    visible blocks both depend on ``n_blocks`` (attention.py says so at the selection), so
    pad blocks can change which PAST blocks an earlier query reads (measured on the tiny
    fixture at top_k 8: positions from 36 on moved by 0.44 when 8 pad tokens were
    appended). With ``top_k >= n_blocks`` every block is selected and the selection cannot
    depend on the padding. The run's shapes satisfy it (top_k 256, ratio 8, packed length
    1152 -> 144 blocks); anything else RAISES instead of computing a padding-dependent
    target."""
    n_blocks = int(length) // int(ref.cfg.csa_compress_ratio)
    if int(ref.cfg.top_k) < n_blocks:
        raise RuntimeError(
            f"stage-1 target model: top_k {ref.cfg.top_k} < {n_blocks} CSA blocks at length "
            f"{length}; its block selection would depend on the right-padding, so the "
            f"target would not be the plain model's function of the real tokens.")



@torch.no_grad()
def plain_prelude_targets(ref, input_ids: Tensor, labels: Tensor, layout) -> tuple[Tensor,
                                                                                   Tensor]:
    """``(z [B, S, C] fp32, ok [B, S])`` — ``plain_prelude`` (module docstring)."""
    g = next_span_grid(input_ids, labels, layout)
    B, S, W = g["ids"].shape
    assert_padding_invariant(ref, W)
    idx = g["ok"].reshape(-1).nonzero().squeeze(1)
    C = int(ref.cfg.d_model)
    z = torch.zeros(B * S, C, device=input_ids.device, dtype=torch.float32)
    if idx.numel() == 0:
        return z.view(B, S, C), g["ok"]
    x, _x0, _bg = ref._front_region(g["ids"].view(B * S, W).index_select(0, idx))
    xs = x.float()
    while xs.dim() > 3:
        xs = xs.mean(dim=-2)                                              # HC stream mean
    v = g["valid"].view(B * S, W).index_select(0, idx).unsqueeze(-1).float()
    pooled = (xs * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
    z.index_copy_(0, idx, _ln(pooled))
    return z.view(B, S, C), g["ok"]


@torch.no_grad()
def plain_final_targets(ref, input_ids: Tensor, labels: Tensor, layout) -> tuple[Tensor,
                                                                                 Tensor]:
    """``(z [B, S, C] fp32, ok [B, S])`` — ``plain_final`` (module docstring)."""
    g = next_span_grid(input_ids, labels, layout)
    assert_padding_invariant(ref, input_ids.shape[1])
    plain_ids, tok_rank = plain_token_row(input_ids, layout.slot_mask)
    x, x0, bg = ref._front_region(plain_ids)
    x = ref._core_region(x, x0, bg, plain_ids)
    h = ref._back_region(x, x0, bg, plain_ids)                             # [B, L, C]
    pos = torch.gather(tok_rank, 1, g["last"])                             # [B, S]
    B, S = pos.shape
    hs = torch.gather(h, 1, pos.unsqueeze(-1).expand(B, S, h.shape[-1]))
    z = _ln(hs) * g["ok"].unsqueeze(-1).float()
    return z, g["ok"]


PLAIN_TARGET_FNS = {"plain_prelude": plain_prelude_targets, "plain_final": plain_final_targets}


# ── the fixed standardisation ────────────────────────────────────────────────────────


class TargetMoments:
    """Running per-coordinate moments of the ``ok`` target rows, fp64.

    ``update(z [B, S, C], ok [B, S])`` adds every ``ok`` row: the count, the coordinate sum
    and the ``C x C`` second-moment matrix (the covariance gives both ``sigma`` and the
    participation ratios; at C = 1024 it is 8 MB in fp64). Order-fixed sums, so the same
    rows in the same order give bit-identical statistics."""

    def __init__(self, dim: int, device) -> None:
        self.n = 0
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, z: Tensor, ok: Tensor) -> None:
        rows = z.reshape(-1, z.shape[-1])[ok.reshape(-1)].to(torch.float64)
        self.n += int(rows.shape[0])
        self.s1 += rows.sum(dim=0)
        self.s2 += rows.t() @ rows

    def finalize(self, floor_frac: float = 1e-3) -> dict:
        """``{"mu" [C] fp32, "sigma" [C] fp32, "mu_norm", "sigma_median", "sigma_min",
        "n_floored", "pr_ln", "pr_standard", "n"}``. ``sigma`` is the population std,
        floored at ``floor_frac * median(sigma)`` so a dead coordinate does not explode.
        ``pr_*``: ``tr(K)^2 / ||K||_F^2`` of the covariance ``K`` of the LayerNormed rows and
        of the standardised rows (the correlation matrix, under the floored sigma)."""
        if self.n < 2:
            raise RuntimeError(f"target calibration saw {self.n} target rows; need >= 2.")
        mu = self.s1 / self.n
        cov = self.s2 / self.n - torch.outer(mu, mu)
        sig = cov.diagonal().clamp_min(0.0).sqrt()
        floor = floor_frac * sig.median()
        if not bool(floor > 0):
            raise RuntimeError("target calibration: the median per-coordinate std is 0; the "
                               "target is constant on the calibration set.")
        n_floored = int((sig < floor).sum())
        sig = sig.clamp_min(floor)
        corr = cov / torch.outer(sig, sig)

        def _pr(k: Tensor) -> float:
            return float(k.diagonal().sum().square() / k.square().sum().clamp_min(1e-30))
        return {"mu": mu.float(), "sigma": sig.float(), "mu_norm": float(mu.norm()),
                "sigma_median": float(sig.median()), "sigma_min": float(sig.min()),
                "n_floored": n_floored, "pr_ln": _pr(cov), "pr_standard": _pr(corr),
                "n": self.n}


def standardise(z: Tensor, ok: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """``(z - mu) / sigma`` per coordinate on the ``ok`` slots, 0 elsewhere (fp32)."""
    return (z.float() - mu) / sigma * ok.unsqueeze(-1).float()


# ── the losses ────────────────────────────────────────────────────────────────────────


def latent_l2(pred: Tensor, z: Tensor, ok: Tensor) -> Tensor:
    """Per-coordinate MSE ``mean_c (p - z)^2`` averaged over the ``ok`` slots.
    ``pred`` ``[B, S, C]``."""
    d = lsel_distance(pred.unsqueeze(2), z).squeeze(2)                     # [B, S]
    okf = ok.float()
    return (d * okf).sum() / okf.sum().clamp_min(1.0)


def latent_infonce(pred: Tensor, z: Tensor, ok: Tensor, tau: float) -> Tensor:
    """InfoNCE over the batch: slot ``i``'s prediction against EVERY ``ok`` slot's target
    (its own row and every other row in one softmax), cosine logits / ``tau``, its own
    target the class; averaged over the ``ok`` slots. ``pred`` ``[B, S, C]``."""
    B, S, C = z.shape
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = F.normalize(pred.float().reshape(B * S, C), dim=-1)
        q = F.normalize(z.float().reshape(B * S, C), dim=-1)
        okf = ok.reshape(-1)
        logits = (p @ q.t()) / float(tau)
        logits = logits.masked_fill(~okf.view(1, -1), float("-inf"))
        logits = torch.where(okf.view(-1, 1), logits, torch.zeros_like(logits))
        ce = F.cross_entropy(logits, torch.arange(B * S, device=z.device), reduction="none")
        w = okf.float()
        return (ce * w).sum() / w.sum().clamp_min(1.0)


def latent_wta(pred_cells: Tensor, z: Tensor, ok: Tensor, eps: float) -> tuple[Tensor,
                                                                              Tensor]:
    """``(loss, winner [B, S])``: the relaxed WTA over the M cells' per-coordinate L2
    (:func:`lsel_exit_loss`, winner = the detached argmin). ``pred_cells`` ``[B, S, M, C]``."""
    dist = lsel_distance(pred_cells, z)                                    # [B, S, M]
    winner = dist.detach().argmin(dim=-1)
    return lsel_exit_loss(dist, winner, ok, float(eps)), winner


# ── the readings ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def retrieval_top1(pred: Tensor, z: Tensor, ok: Tensor) -> dict[str, Tensor]:
    """Top-1 retrieval of each ``ok`` slot's OWN target by cosine similarity.

    ``pred`` ``[B, S, C]`` (one cell) or ``[B, S, M, C]``. Candidates: every ``ok`` slot of
    the same row (``same``) or of the whole batch (``all``). ``retr_*`` scores ONE
    prediction per slot: the cell itself, or at M > 1 the MEAN of the M cells'
    predictions (the read that needs no oracle). At M > 1 ``retr_best_*`` is the hit rate
    of the best of the M cells (an ORACLE over cells). Chance comes from the ACTUAL
    candidate counts: ``chance_*`` = mean over queries of ``1 / n_candidates(query)``;
    ``chance_best_*`` = mean of ``1 - (1 - 1/n)^M`` (M independent uniform picks).
    Returns 0-dim fp32 tensors (no host sync)."""
    B, S, C = z.shape
    cells = pred if pred.dim() == 4 else pred.unsqueeze(2)
    M = int(cells.shape[2])
    N = B * S
    with torch.autocast(device_type=z.device.type, enabled=False):
        q = F.normalize(z.float().reshape(N, C), dim=-1)
        pc = F.normalize(cells.float().reshape(N, M, C), dim=-1)
        p1 = pc[:, 0] if M == 1 else F.normalize(cells.float().mean(dim=2).reshape(N, C),
                                                 dim=-1)
        okf = ok.reshape(N)
        w = okf.float()
        n_q = w.sum().clamp_min(1.0)
        row = torch.arange(B, device=z.device).repeat_interleave(S)
        own = torch.arange(N, device=z.device)
        out: dict[str, Tensor] = {}
        for name, cand in (("same", (row.view(N, 1) == row.view(1, N)) & okf.view(1, N)),
                           ("all", okf.view(1, N).expand(N, N))):
            n_c = cand.float().sum(dim=-1).clamp_min(1.0)                  # [N]
            s1 = (p1 @ q.t()).masked_fill(~cand, float("-inf"))
            out[f"retr_{name}"] = ((s1.argmax(dim=-1) == own).float() * w).sum() / n_q
            out[f"chance_{name}"] = ((1.0 / n_c) * w).sum() / n_q
            if M > 1:
                sm = torch.einsum("nmc,kc->nmk", pc, q).masked_fill(~cand.view(N, 1, N),
                                                                   float("-inf"))
                hit = (sm.argmax(dim=-1) == own.view(N, 1)).float().amax(dim=-1)
                out[f"retr_best_{name}"] = (hit * w).sum() / n_q
                out[f"chance_best_{name}"] = ((1.0 - (1.0 - 1.0 / n_c) ** M) * w).sum() / n_q
        return out


@torch.no_grad()
def r2_score(pred: Tensor, z: Tensor, ok: Tensor) -> Tensor:
    """``1 - mean_ok mean_c (p - z)^2 / mean_c Var_ok(z)`` (the latent-selected loop's
    ``lsel_r2`` form). ``pred`` ``[B, S, C]``."""
    B, S, C = z.shape
    okf = ok.reshape(B * S).float()
    mse = (lsel_distance(pred.unsqueeze(2), z).squeeze(2).reshape(-1) * okf).sum() \
        / okf.sum().clamp_min(1.0)
    var = batch_var(z.reshape(B * S, C), okf).mean()
    return 1.0 - mse / var.clamp_min(1e-12)


@torch.no_grad()
def latent_readings(pred_cells: Tensor, z: Tensor, ok: Tensor, tag: str) -> dict[str, Tensor]:
    """Every reading of one pass, keyed ``{name}_{tag}`` (``tag`` = ``exit`` or ``t{t}``).

    ``pred_cells`` ``[B, S, M, C]``. ``retr_same`` / ``retr_all``, ``r2`` and ``pred_rank``
    read ONE prediction per slot: the cell at M = 1, the MEAN of the M cells' predictions
    at M > 1 (the WTA arm; no oracle). At M > 1 also the oracle read of the best cell
    (``retr_best_*``; ``r2_best`` on each slot's closest cell) and ``cell_spread``. Chance
    keys carry no tag (they depend on the candidate counts only)."""
    B, S, M, C = pred_cells.shape
    okf = ok.reshape(B * S).float()
    out: dict[str, Tensor] = {}
    r = retrieval_top1(pred_cells, z, ok)
    for k, v in r.items():
        out[k if k.startswith("chance") else f"{k}_{tag}"] = v
    single = pred_cells.mean(dim=2)
    out[f"r2_{tag}"] = r2_score(single, z, ok)
    out[f"pred_rank_{tag}"] = participation_ratio(single.reshape(B * S, C), okf)
    if M > 1:
        win = lsel_distance(pred_cells, z).argmin(dim=-1)                  # [B, S]
        best = torch.gather(pred_cells, 2, win.view(B, S, 1, 1).expand(B, S, 1, C)).squeeze(2)
        out[f"r2_best_{tag}"] = r2_score(best, z, ok)
        out[f"cell_spread_{tag}"] = cell_spread(pred_cells, ok)
    return out
