"""TUL-Code (``tul.code``): the slot holds the CODE of the span it precedes, and the core
body samples it. Spec: ``docs/tul-code-spec.md``. Note:
``.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md``.

Four small pieces live here, each with one job, none of them touching the model's graph
when ``tul.code`` is false (the model builds none of them then):

* :class:`TULCodeEncoder` — E. ``M`` learned queries pool ``M`` vectors out of the NEXT
  span's token prelude states (the :class:`~morph.model.tul.TULSlotRegister` mechanism with
  the bag index shifted by one), then a per-cell map and a per-cell embedding. Training-
  time only; the code of span ``s+1`` is made from span ``s+1``.
* :class:`TULCodeTime` — the time embedding ``t -> [C]`` that enters the thinker's
  injection term at the noisy copies (sinusoidal basis + MLP, the ``SigmaConditioning``
  basis of ``morph/model/diffusion_blocks.py``).
* :class:`TULCodeHead` — the velocity head: a PRIVATE RMSNorm over the stream-mean of the
  core's output at a noisy copy, then a zero-init linear. Never ``_readout`` (whose
  ``lm_mixer`` / ``final_norm`` belong to the LM head).
* the pure functions: :func:`code_rmsnorm`, :func:`code_target_valid`,
  :func:`code_thinker_relation`, :func:`cfm_pair`, :func:`cfm_null_floor`,
  :func:`euler_sample` — the arithmetic ``fm_planner._cfm_loss`` / ``generate_plans`` do,
  copied rather than called because those take an ``FMPlanner``.

THE DOUBLED SLOT SEQUENCE, and why its order is what it is. The thinker runs ONE pass over
``2·S·M`` positions laid out slot-major and, within a slot, ``[noisy cells (M), clean
cells (M)]``:

    [ n(0,0) .. n(0,M-1)  c(0,0) .. c(0,M-1) | n(1,0) .. n(1,M-1)  c(1,0) .. c(1,M-1) | ...

A clean copy holds the (detached) code ``z_s`` and is context only. A noisy copy holds
``z_t(s)``. The attention relation (:func:`code_thinker_relation`) lets a noisy cell read
the clean cells of slots ``< s`` and its own slot's noisy cells, and a clean cell read the
clean cells of slots ``<= s``. That relation WIDENS flattened causal within a slot (a noisy
cell sees its later siblings), so it is delivered as ``tg_relation`` (the one attention
kwarg that replaces the causal term; ``slot_cell_relation``'s docstring in
``transformer.py`` records why ``tg_allow`` cannot carry it).

The ORDER matters for the operators the mask does not govern. The CCA causal conv and its
``W_v_prev`` value shift read backwards along the flattened axis. With the noisy copies
FIRST within a slot, everything to the left of ``n(s, ·)`` belongs to slots ``< s`` — past
codes, past noisy states — so neither operator can hand a noisy cell its own target
``c(s, ·)`` (which sits to its RIGHT) or a future code. The other order, clean first, would
put ``z_s`` one position to the left of ``z_t(s)`` and the conv would leak the answer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from morph.model.tul_layout import SlotLayout

__all__ = [
    "TULCodeEncoder", "TULCodeTime", "TULCodeHead", "TULCodeSymHead",
    "code_rmsnorm", "code_target_valid", "code_thinker_relation",
    "cfm_pair", "cfm_null_floor", "euler_sample", "draw_flow_t",
    "code_contrastive_accuracy",
    "mdm_mask", "mdm_loss", "mdm_unmask_counts", "maskgit_sample",
    "code_grade_distinct2", "code_grade_pref_loss",
]

_SEED_ENC = 0xC0DE
_SEED_TIME = 0xC0DE + 1
_SEED_SYM = 0xC0DE + 2
_EPS = 1e-8
_MDM_T_MIN = 1e-3
_FLOW_T_EPS = 1e-4


def code_rmsnorm(z: Tensor, eps: float = _EPS) -> Tensor:
    """Unit per-component scale on the last axis, no affine: ``z / sqrt(mean(z²) + eps)``.

    ``tul.code_norm: rms``, the only mode in v0.1. With this norm ``E‖z‖² = C`` per cell,
    which is what :func:`cfm_null_floor` assumes.
    """
    return z * torch.rsqrt(z.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(z.dtype)


def code_target_valid(layout: SlotLayout) -> Tensor:
    """``[B, S]`` bool: slot ``s`` is valid AND slot ``s+1`` is valid, i.e. span ``s+1`` is
    a complete span with a slot of its own. The row's LAST valid slot precedes the row's
    tail (the dump bin, a partial span cut by the sequence end) and gets no code — the same
    rule :func:`~morph.model.tul.next_span_pool` applies to its retrieval target.
    """
    B, S = layout.slot_valid.shape
    ok = torch.zeros_like(layout.slot_valid)
    ok[:, :S - 1] = layout.slot_valid[:, :S - 1] & layout.slot_valid[:, 1:S]
    return ok


class TULCodeEncoder(nn.Module):
    """E: ``M`` cells per slot pooled from the NEXT span's token prelude states.

    ``A_i = W_o( sum_j softmax_j(<Q_i, W_k x_j>/sqrt(d)) W_v x_j ) + P_cell[i]`` over
    ``j`` in span ``s+1``'s TOKEN positions, then :func:`code_rmsnorm`. Single head. Every
    weight comes from a PRIVATE generator so a code model's base weights are byte-identical
    to the strict ruler's (the ``W_sent`` / register precedent). ``W_o`` is NOT zero-init
    (the register zeroes it so step 0 equals its ruler; here nothing has to equal anything
    — a zero code would be all-``P_cell`` and the RMS norm of a zero vector is meaningless).
    """

    def __init__(self, d_model: int, m_cells: int):
        super().__init__()
        if m_cells < 1:
            raise ValueError(f"TULCodeEncoder needs m_cells >= 1, got {m_cells}")
        self.m = int(m_cells)
        self.scale = d_model ** -0.5
        _rng0 = torch.random.get_rng_state()
        g = torch.Generator(device="cpu").manual_seed(_SEED_ENC)
        self.Q = nn.Parameter(torch.empty(self.m, d_model).normal_(
            mean=0.0, std=0.02, generator=g))
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        for lin in (self.W_k, self.W_v, self.W_o):
            with torch.no_grad():
                lin.weight.copy_(torch.empty(lin.weight.shape, device="cpu").normal_(
                    mean=0.0, std=0.02, generator=g))
            lin._ternary_exclude = True
        self.P_cell = nn.Parameter(torch.zeros(self.m, d_model))
        torch.random.set_rng_state(_rng0)

    def forward(self, xs: Tensor, layout: SlotLayout) -> tuple[Tensor, Tensor]:
        """``xs`` ``[B, L, C]`` single-stream prelude states over the packed row ->
        ``(z [B, S, M, C], ok [B, S])``. ``z`` is unit-RMS per cell; slots with ``ok``
        false hold exactly 0.
        """
        B, L, C = xs.shape
        S = layout.slot_index.shape[1]
        M = self.m
        k = self.W_k(xs)
        v = self.W_v(xs)
        sc = torch.einsum("mc,blc->bml", self.Q.to(xs.dtype), k) * self.scale   # [B, M, L]
        tok = ~layout.slot_mask                                                  # [B, L]
        nxt = (layout.bag_id.unsqueeze(1) == (torch.arange(
            S, device=xs.device) + 1).view(1, S, 1)) & tok.unsqueeze(1)         # [B, S, L]
        # A slot whose next span has no token (a pad, the row's last slot) would give an
        # all-(-inf) softmax row and a NaN that propagates through the BACKWARD (the
        # register's scar). Open those rows to everything and zero the result below.
        _has = nxt.any(dim=-1, keepdim=True)
        nxt = torch.where(_has, nxt, torch.ones_like(nxt))
        logits = sc.unsqueeze(1) + torch.where(
            nxt.unsqueeze(2), sc.new_zeros(()), sc.new_full((), float("-inf")))
        a = torch.softmax(logits, dim=-1)                                        # [B, S, M, L]
        pooled = torch.einsum("bsml,blc->bsmc", a, v)
        out = self.W_o(pooled) + self.P_cell.to(xs.dtype).view(1, 1, M, C)
        ok = code_target_valid(layout) & _has.squeeze(-1)
        z = code_rmsnorm(out) * ok.view(B, S, 1, 1).to(out.dtype)
        return z, ok


class TULCodeProj(nn.Module):
    """``tul.code_target`` (spec §17): the slot loop's exit state -> ``M`` unit-RMS cells.

    ``cell_i = code_rmsnorm(W_code[i] · r + b_code[i])`` with ``r`` the loop's readout of
    the slot's exit state ``[B, S, C]``. ``W_code`` is IDENTITY at init and ``b_code`` zero
    (no RNG draw: a code-target model's base weights equal its ruler's), so at step 0 the
    coda reads the rms-normed exit state itself. Excluded from ternary QAT: it is the ONE
    channel from the loop into the frozen coda (the ``W_prefix`` precedent), and the
    frozen encoder it is regressed onto is dense too.
    """

    def __init__(self, d_model: int, m_cells: int):
        super().__init__()
        if m_cells < 1:
            raise ValueError(f"TULCodeProj needs m_cells >= 1, got {m_cells}")
        self.m = int(m_cells)
        self.W_code = nn.Parameter(torch.eye(d_model).unsqueeze(0).repeat(self.m, 1, 1))
        self.b_code = nn.Parameter(torch.zeros(self.m, d_model))
        self._ternary_exclude = True

    def forward(self, r: Tensor, valid: Tensor) -> Tensor:
        """``r`` ``[B, S, C]`` -> ``[B, S, M, C]`` unit-RMS per cell; a slot with ``valid``
        false holds exactly 0 (the encoder's contract, so a pad cell and a pad target agree).
        """
        B, S, C = r.shape
        w = self.W_code.to(r.dtype)
        out = torch.einsum("bsc,mdc->bsmd", r, w) + self.b_code.to(r.dtype).view(1, 1, self.m, C)
        return code_rmsnorm(out) * valid.view(B, S, 1, 1).to(out.dtype)


def code_target_regression(pred: Tensor, z: Tensor, ok: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor]:
    """The ``tul.code_target`` term on unit-RMS cells: ``mean over valid cells of
    ||pred - z||² / C`` = ``2 (1 - cos)`` per cell. Returns ``(loss, cos_mean, n_cells)``;
    with no valid cell the loss is an exact 0 that still carries ``pred``'s graph.
    ``z`` is the frozen encoder's code and is detached here, so the term can never move it.
    """
    B, S, M, C = pred.shape
    z = z.detach().float()
    p = pred.float()
    okm = ok.view(B, S, 1).expand(B, S, M)
    n = okm.sum()
    d2 = (p - z).pow(2).sum(-1) / float(C)                                  # [B, S, M]
    cos = (p * z).sum(-1) / float(C)                                         # unit RMS: <p,z>/C
    if int(n) == 0:
        zero = (p * 0.0).sum()
        return zero, zero.detach(), n
    loss = d2[okm].mean()
    return loss, cos[okm].mean().detach(), n


def code_target_infonce(pred: Tensor, z: Tensor, ok: Tensor, tau: float
                        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The ``tul.code_target_loss="infonce"`` term: per cell ``m``, the predicted cells of
    the batch's valid slots score against EVERY valid slot's code (``<p_i, z_j> / C / tau``,
    a cosine on unit-RMS cells) and the own code is the class. Returns ``(loss, cos_mean,
    n_cells, acc)``: ``loss`` the mean over cells of the cross-entropy, ``cos_mean`` the
    own-code cosine (the same reading the L2 term reports), ``acc`` the top-1 rate (chance
    is ``1 / n_valid``). ``z`` is detached: the frozen encoder's code never moves.

    Why it exists: the L2 term's optimum is the conditional MEAN of the code, which hedges
    toward the corpus-mean direction (arm A, 2026-09-17: cos to the mean 0.33-0.53 against
    the codes' 0.06-0.09, rank 16 of 75). This term asks the loop to pick its span OUT OF
    the batch instead, own-minus-shuffled directly.
    """
    B, S, M, C = pred.shape
    z = z.detach().float()
    p = pred.float()
    n = ok.sum() * M
    if int(ok.sum()) == 0:
        zero = (p * 0.0).sum()
        return zero, zero.detach(), n, zero.detach()
    losses, coss, accs = [], [], []
    for m in range(M):
        P = p[:, :, m][ok]                                                   # [n, C]
        Z = z[:, :, m][ok]
        sim = P @ Z.t() / float(C)                                           # [n, n]
        tgt = torch.arange(P.shape[0], device=P.device)
        losses.append(F.cross_entropy(sim / float(tau), tgt))
        coss.append(sim.diagonal().mean().detach())
        accs.append((sim.argmax(dim=1) == tgt).float().mean())
    return (torch.stack(losses).mean(), torch.stack(coss).mean(), n,
            torch.stack(accs).mean().detach())


@torch.no_grad()
def code_target_shuffled_cos(pred: Tensor, z: Tensor, ok: Tensor) -> Tensor:
    """The predicted cells against a FOREIGN code: the valid slots' codes rolled by half
    the valid count (a deterministic pairing that crosses rows and consumes no RNG). The
    generic floor beside ``code_target_cos``; own minus this is what the cell knows about
    its own span (``lab/divergence/code_target_mean_probe.py`` is the offline twin).
    """
    B, S, M, C = pred.shape
    if int(ok.sum()) < 2:
        return pred.new_zeros(()).float()
    P = pred.float()[ok]                                                     # [n, M, C]
    Z = z.float()[ok]
    Zr = torch.roll(Z, shifts=Z.shape[0] // 2, dims=0)
    return ((P * Zr).sum(-1) / float(C)).mean()


def code_grade_distinct2(tokens: Tensor, lengths: Tensor) -> Tensor:
    """Distinct-2 of each candidate: unique bigrams / bigrams, ``[..., J] -> [...]``.

    ``tokens`` ``[..., J]`` long (only the first ``lengths`` entries of each row are the
    candidate; the rest are ignored) and ``lengths`` ``[...]`` long. A candidate of length
    ``n`` has ``n - 1`` bigrams; a length below 2 reads 1.0 (nothing repeats).

    The diversity guard of ``tul.code_grade``: a repetition loop is FLUENT, so a
    log-probability grade ranks it first (``genppl-needs-a-diversity-guard``: a repetition
    loop scored gen-PPL 1.46 against real text's 32.44). A candidate under
    ``code_grade_min_distinct2`` is given the worst grade instead of the best.
    """
    J = tokens.shape[-1]
    if J < 2:
        raise ValueError(f"code_grade_distinct2 needs J >= 2, got {J}")
    big = tokens[..., :-1] * (int(tokens.max()) + 2) + tokens[..., 1:]       # [..., J-1]
    n_big = (lengths - 1).clamp_min(0)                                       # [...]
    idx = torch.arange(J - 1, device=tokens.device)
    live = idx.view(*([1] * (big.dim() - 1)), J - 1) < n_big.unsqueeze(-1)   # [..., J-1]
    eq = big.unsqueeze(-1) == big.unsqueeze(-2)                              # [..., J-1, J-1]
    earlier = idx.view(-1, 1) > idx.view(1, -1)                              # [J-1, J-1] i>j
    dup = (eq & earlier & live.unsqueeze(-2)).any(dim=-1)                    # seen before
    uniq = (live & ~dup).sum(dim=-1).float()
    return torch.where(n_big > 0, uniq / n_big.clamp_min(1).float(),
                       torch.ones_like(uniq))


def code_grade_pref_loss(pred: Tensor, z_cand: Tensor, best: Tensor, ok: Tensor,
                         tau: float) -> tuple[Tensor, Tensor]:
    """``tul.code_grade_loss="pref"``: per cell, the predicted cell scores against the K
    CANDIDATE codes of its own slot and the WINNER is the class.

    ``pred`` ``[R, S, M, C]`` (live, unit-RMS), ``z_cand`` ``[R, K, S, M, C]`` (detached,
    unit-RMS), ``best`` ``[R, S]`` long, ``ok`` ``[R, S]`` bool. Logits are
    ``<pred, z_k> / C / tau`` — a cosine on unit-RMS cells, the ``code_target_infonce``
    scale. Returns ``(loss, margin)`` with ``margin`` the detached mean of
    ``cos(pred, z_best) - mean_k!=best cos(pred, z_k)``: the term pushes toward the winner
    AND away from every loser, and ``margin`` is how far it has got.

    With no valid slot the loss is an exact 0 that still carries ``pred``'s graph.
    """
    R, K, S, M, C = z_cand.shape
    if pred.shape != (R, S, M, C):
        raise ValueError(f"code_grade_pref_loss: pred {tuple(pred.shape)} != {(R, S, M, C)}")
    p = pred.float()
    z = z_cand.detach().float()
    sim = torch.einsum("rsmc,rksmc->rskm", p, z) / float(C)                  # [R, S, K, M]
    if int(ok.sum()) == 0:
        zero = (p * 0.0).sum()
        return zero, zero.detach()
    sel = sim[ok]                                                            # [n, K, M]
    tgt = best[ok]                                                           # [n]
    logits = (sel / float(tau)).permute(0, 2, 1).reshape(-1, K)              # [n·M, K]
    loss = F.cross_entropy(logits, tgt.repeat_interleave(M))
    own = sel.gather(1, tgt.view(-1, 1, 1).expand(-1, 1, M)).squeeze(1)      # [n, M]
    other = (sel.sum(dim=1) - own) / max(K - 1, 1)
    return loss, (own - other).mean().detach()


class TULCodeTime(nn.Module):
    """``t`` in ``[0, 1]`` (per slot) -> ``[C]`` added to the noisy copies' injection.

    Sinusoidal basis (``n_freq`` frequencies, the ``SigmaConditioning`` construction)
    then a two-layer MLP to ``d_model``. ``t_embed_scale`` multiplies ``t`` before the
    basis, with the meaning ``fm_planner.FMPlannerConfig.t_embed_scale`` records: at 1.0
    the top frequency is 1 rad/unit and the basis is low order; ~16 gives a real Fourier
    basis over the unit interval. Last layer zero-init so step 0 adds nothing.
    """

    def __init__(self, d_model: int, n_freq: int = 64, t_embed_scale: float = 1.0):
        super().__init__()
        if n_freq < 1:
            raise ValueError(f"TULCodeTime needs n_freq >= 1, got {n_freq}")
        self.n_freq = int(n_freq)
        self.t_embed_scale = float(t_embed_scale)
        _rng0 = torch.random.get_rng_state()
        g = torch.Generator(device="cpu").manual_seed(_SEED_TIME)
        self.l1 = nn.Linear(2 * self.n_freq, d_model)
        self.l2 = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.l1.weight.copy_(torch.empty(self.l1.weight.shape).normal_(
                mean=0.0, std=0.02, generator=g))
            self.l1.bias.zero_()
            self.l2.weight.zero_()
            self.l2.bias.zero_()
        self.l1._ternary_exclude = True
        self.l2._ternary_exclude = True
        torch.random.set_rng_state(_rng0)

    def forward(self, t: Tensor) -> Tensor:
        """``t`` ``[*]`` -> ``[*, C]``."""
        half = self.n_freq
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                          * (-math.log(10000.0) / max(half - 1, 1)))
        ang = (t.float() * self.t_embed_scale).unsqueeze(-1) * freqs
        basis = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.l2(F.silu(self.l1(basis)))


class TULCodeHead(nn.Module):
    """The velocity head: private RMSNorm over the stream-mean carrier, then a zero-init
    linear, so the first velocity estimate is exactly 0 (``fm_planner.FMPlanner.out``).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))          # the private norm's gain
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            self.W_v.weight.zero_()
            self.W_v.bias.zero_()
        self.W_v._ternary_exclude = True

    def forward(self, h: Tensor) -> Tensor:
        """``h`` ``[..., C]`` (already reduced over streams) -> velocity ``[..., C]``."""
        hn = code_rmsnorm(h) * self.weight.to(h.dtype)
        return self.W_v(hn)


def code_thinker_relation(n_slots: int, m_cells: int, device) -> Tensor:
    """``[1, 1, 2·S·M, 2·S·M]`` bool, True = may attend, for the doubled slot sequence.

    Position ``p = s·2M + c·M + i`` with copy ``c`` (0 noisy, 1 clean) and cell ``i``:

    * noisy ``(s, i)`` reads clean cells of slots ``< s`` and every noisy cell of slot ``s``;
    * clean ``(s, i)`` reads clean cells of slots ``<= s`` (itself included).

    Nothing reads a noisy cell of another slot, nothing reads its own slot's clean cells
    from the noisy side (that is the target), nothing reads a later slot.
    """
    n = n_slots * 2 * m_cells
    p = torch.arange(n, device=device)
    slot = p // (2 * m_cells)
    copy = (p // m_cells) % 2                       # 0 noisy, 1 clean
    si, sj = slot.unsqueeze(1), slot.unsqueeze(0)
    ci, cj = copy.unsqueeze(1), copy.unsqueeze(0)
    q_noisy = ci == 0
    k_clean = cj == 1
    k_noisy = cj == 0
    allow_noisy = q_noisy & ((k_clean & (sj < si)) | (k_noisy & (sj == si)))
    allow_clean = (~q_noisy) & k_clean & (sj <= si)
    return (allow_noisy | allow_clean).view(1, 1, n, n)


def cfm_pair(z: Tensor, source_std: float, t: Tensor, generator=None,
             z0: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Straight-line CFM pair for target ``z`` ``[B, S, M, C]`` at per-slot ``t`` ``[B, S]``.

    ``z_0 ~ N(0, source_std²·I)``, ``z_t = (1−t)·z_0 + t·z``, ``v_target = z − z_0``. The
    arithmetic of ``fm_planner._cfm_loss`` lines 928-932, with ``t`` shared by a slot's
    ``M`` cells (one code per span, one time per span). A given ``z0`` (Explorative
    Modeling: the noise whose sample came nearest the data) replaces the fresh draw.
    """
    if z0 is None:
        z0 = torch.randn(z.shape, device=z.device, dtype=torch.float32,
                         generator=generator) * float(source_std)
    else:
        if z0.shape != z.shape:
            raise ValueError(f"cfm_pair: z0 {tuple(z0.shape)} must match z {tuple(z.shape)}")
        z0 = z0.float()
    tt = t.float().view(*t.shape, 1, 1)
    zf = z.float()
    z_t = (1.0 - tt) * z0 + tt * zf
    return z0, z_t, zf - z0


def draw_flow_t(B: int, S: int, tc, device, generator=None) -> Tensor:
    """The flow thinker's per-slot time draw ``[B, S]`` fp32 — the ONE home of the
    training-time NOISE SCHEDULE (``tul.code_t_logit_mean`` / ``code_t_logit_std``).

    ``tc.code_t_logit_mean is None`` (the default) draws ``t ~ U(0, 1)``, the tree's
    behaviour before the knob, and consumes exactly one ``torch.rand`` of shape
    ``[B, S]`` — the same call the flow site made when the line was written inline, so a
    default arm is bit-identical to a pre-knob run, RNG stream included.

    Set, the draw is ``t = sigmoid(mu + sigma * eps)`` with ``eps ~ N(0, 1)`` per slot,
    clamped to ``[1e-4, 1 - 1e-4]``. ``eps`` comes from the SAME uniform by the normal
    quantile ``sqrt(2) * erfinv(2u - 1)`` rather than a second ``torch.randn``, so both
    branches consume one identically shaped draw and an arm and its control stay aligned
    on every LATER random draw of the step.

    WHY THIS KNOB EXISTS — the log-SNR, and what uniform ``t`` spends its training on.
    The interpolant is ``z_t = (1-t) z_0 + t z`` with ``z_0 ~ N(0, s^2 I)`` at
    ``s = code_source_std = 1`` and unit-RMS codes, so signal and noise stand at ``t`` and
    ``1 - t`` and the log signal-to-noise ratio is

        lambda(t) = log( t^2 / (1-t)^2 ) = 2 * logit(t)          (at s = 1)

    Uniform ``t`` therefore puts HALF of training above ``lambda = 0`` (SNR 1), where the
    noisy input already carries the answer and the context is worth nothing to the
    velocity. That is the regime LCM's Table 5 measures: a schedule concentrated at low
    noise gets the best l2 and the WORST contrastive accuracy — "akin to a Base-LCM" —
    while a schedule spread to high noise "learns to contrast". Our own thinker read 1 %
    of its loss from the past (`lctul-thinker-is-context-blind`), which is that failure.

    A logit-normal draw moves the mass. ``mu = -1, sigma = 1`` puts the MEDIAN ``t`` at
    ``sigmoid(-1) = 0.2689`` (``lambda = -2``, SNR ``e^-2 = 0.135``), its MEAN at
    ``0.30327`` (Gaussian quadrature) and ``P(t < 0.5) = Phi(1) = 0.8413``.

    This reweights the training LOSS only. The SAMPLER's Euler grid stays uniform
    (:func:`euler_sample` walks ``t_j = j/k``): LCM changes the training schedule, and
    their sampler's step selection is a separate knob this tree does not add.

    NOTE the discrete path (``tul.code_discrete``) draws its own ``t`` and is NOT this
    function: there ``t`` is the Bernoulli MASK RATE of the ELBO and the ``1/t`` weight in
    :func:`mdm_loss` is the estimator, not a schedule to tune.
    """
    u = torch.rand(B, S, device=device, dtype=torch.float32, generator=generator)
    mu = tc.code_t_logit_mean
    if mu is None:
        return u
    eps = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
    t = torch.sigmoid(float(mu) + float(tc.code_t_logit_std) * eps)
    return t.clamp(_FLOW_T_EPS, 1.0 - _FLOW_T_EPS)


@torch.no_grad()
def code_contrastive_accuracy(pred: Tensor, truth: Tensor, valid: Tensor,
                              row_id: Tensor, slot_id: Tensor) -> tuple[float, float]:
    """LCM's CA (contrastive accuracy) on codes: does the SAMPLED code retrieve its own
    span's TRUE code out of the batch?

    ``pred`` / ``truth`` ``[..., M, C]`` (the sampled cells and E's cells), ``valid`` /
    ``row_id`` / ``slot_id`` ``[...]`` over the same leading axes. Each valid slot's cells
    are FLATTENED to one ``M*C`` vector and l2-normalised; the score against candidate
    ``j`` is the cosine to ``truth_j``. A hit is ``argmax_j = i``.

    LCM's exclusion rule (§2.4.1): the slot's two TEMPORAL NEIGHBOURS in its own row are
    dropped from the pool, because consecutive spans are near-paraphrases and would be
    scored as wrong answers. The slot's own truth is always a candidate; a slot in another
    row never is excluded. ``chance`` is the mean over slots of ``1 / n_candidates``, and
    it is NOT ``1/n``: an interior slot has two fewer candidates than an end slot, so the
    reading has to be compared against its own floor and never against a nominal one.

    Returns ``(acc, chance)`` as Python floats. With no valid slot both are 0.0.

    Why CA and not l2: Base-LCM won l2 on all four corpora and lost CA on all four
    (Table 3). A regression to the conditional mean minimises distance and retrieves
    nothing; CA is the instrument that separates the two, and it is scale-free where l2
    is not.
    """
    if pred.shape != truth.shape:
        raise ValueError(f"code_contrastive_accuracy: pred {tuple(pred.shape)} != "
                         f"truth {tuple(truth.shape)}")
    if pred.dim() < 3:
        raise ValueError(f"code_contrastive_accuracy needs [..., M, C], got "
                         f"{tuple(pred.shape)}")
    if valid.shape != pred.shape[:-2] or row_id.shape != valid.shape \
            or slot_id.shape != valid.shape:
        raise ValueError(
            f"code_contrastive_accuracy: valid {tuple(valid.shape)} / row_id "
            f"{tuple(row_id.shape)} / slot_id {tuple(slot_id.shape)} must all be "
            f"{tuple(pred.shape[:-2])}")
    w = int(pred.shape[-2]) * int(pred.shape[-1])
    v = valid.reshape(-1)
    n = int(v.sum())
    if n == 0:
        return 0.0, 0.0
    P = pred.reshape(-1, w).float()[v]
    Z = truth.reshape(-1, w).float()[v]
    r = row_id.reshape(-1)[v]
    s = slot_id.reshape(-1)[v]
    P = P / P.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    Z = Z / Z.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    sim = P @ Z.t()                                                          # [n, n]
    drop = (r.view(-1, 1) == r.view(1, -1)) & ((s.view(-1, 1) - s.view(1, -1)).abs() == 1)
    n_cand = (~drop).sum(dim=1).clamp_min(1).float()
    hit = sim.masked_fill(drop, float("-inf")).argmax(dim=1) == torch.arange(
        n, device=sim.device)
    return float(hit.float().mean()), float((1.0 / n_cand).mean())


def cfm_null_floor(d_model: int, m_cells: int, source_std: float) -> float:
    """``E‖v_target‖²`` per slot for ``v̂ ≡ 0``: ``M·(C + C·source_std²)`` with unit-RMS
    codes (``E‖z‖² = C`` per cell) and an independent source. This is the number
    ``loss_scale: auto`` divides by — NOT ``fm_planner``'s ``1 + d·s²`` (its targets were
    unit-L2, ``E‖y‖² = 1``). Spec §3.5.
    """
    return float(m_cells) * float(d_model) * (1.0 + float(source_std) ** 2)


@torch.no_grad()
def euler_sample(velocity, z0: Tensor, n_steps: int) -> Tensor:
    """``z_{j+1} = z_j + (1/k)·velocity(z_j, t_j)``, ``t_j = j/k`` — ``fm_planner.
    generate_plans`` lines 985-989. ``velocity(z, t)`` takes ``z`` ``[B, S, M, C]`` fp32 and
    ``t`` ``[B, S]`` fp32 and returns fp32 ``[B, S, M, C]``. Returns the endpoint, fp32,
    NOT normalised: the caller applies :func:`code_rmsnorm` where the coda reads it.
    """
    if n_steps < 1:
        raise ValueError(f"euler_sample needs n_steps >= 1, got {n_steps}")
    z = z0.float()
    dt = 1.0 / float(n_steps)
    B, S = z.shape[:2]
    for j in range(n_steps):
        t = torch.full((B, S), j * dt, device=z.device, dtype=torch.float32)
        z = z + dt * velocity(z, t).float()
    return z


# ── LCTUL-D: the discrete code and the masked denoiser (spec §16) ─────────────────────
#
# On a ``tul.code_discrete`` model the slot's code is N = K·G SYMBOLS from a shared
# codebook of C rows (``TULThoughtVQ`` on E's pooled vector, ``morph/model/tul_vq.py``),
# the thinker is a MASKED DENOISER over those symbols (one categorical per masked
# position, MDLM / LLaDA's objective) and the sampler is MaskGIT's k-round unmasking.
# The doubled slot sequence, the relation mask and the time embedding are the flow
# thinker's, with M := N symbol positions per copy: a noisy copy holds the symbol
# embedding or the MASK embedding, a clean copy the (context) symbol embedding.


class TULCodeSymHead(nn.Module):
    """The denoiser head: private RMSNorm over the stream-mean carrier, then a zero-init
    linear to ``C`` logits, so the first prediction is exactly uniform (``TULCodeHead``'s
    contract, one alphabet wide)."""

    def __init__(self, d_model: int, codebook: int):
        super().__init__()
        if codebook < 2:
            raise ValueError(f"TULCodeSymHead needs codebook >= 2, got {codebook}")
        self.c = int(codebook)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.W_o = nn.Linear(d_model, self.c, bias=True)
        with torch.no_grad():
            self.W_o.weight.zero_()
            self.W_o.bias.zero_()
        self.W_o._ternary_exclude = True

    def forward(self, h: Tensor) -> Tensor:
        """``h`` ``[..., C]`` -> logits ``[..., codebook]`` fp32."""
        hn = code_rmsnorm(h) * self.weight.to(h.dtype)
        return self.W_o(hn).float()


def mdm_mask(t: Tensor, n_sym: int, generator=None) -> Tensor:
    """Bernoulli(``t``) mask per symbol: ``t`` ``[B, S]`` -> ``[B, S, N]`` bool (True =
    masked). No symbol is forced: a slot that draws no mask contributes 0 to the ELBO,
    which keeps the ``1/t`` estimator unbiased (a forced mask at small ``t`` would weigh
    one symbol by ``1/t`` and read several times the uniform floor at the zero-init
    head). The trainer draws ``t`` in ``[_MDM_T_MIN, 1]``."""
    B, S = t.shape
    u = torch.rand((B, S, n_sym), device=t.device, dtype=torch.float32, generator=generator)
    return u < t.unsqueeze(-1)


def mdm_loss(logits: Tensor, target: Tensor, mask: Tensor, t: Tensor, ok: Tensor
             ) -> tuple[Tensor, Tensor]:
    """The masked-diffusion ELBO, per slot: ``(1/t) · Σ_masked CE(logits, target)`` in
    nats per SPAN (an upper bound on ``-log p(code | past)``), and the per-slot term
    summed over the valid slots divided by their count.

    ``logits`` ``[B, S, N, C]`` fp32, ``target`` ``[B, S, N]`` long, ``mask`` / ``ok``
    bool. Returns ``(loss_mean, per_slot)``; ``per_slot`` is zero on invalid slots.
    """
    B, S, N, C = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, C), target.clamp_min(0).reshape(-1),
                         reduction="none").reshape(B, S, N)
    w = mask.to(ce.dtype) / t.clamp_min(_MDM_T_MIN).unsqueeze(-1)
    per_slot = (ce * w).sum(dim=-1) * ok.to(ce.dtype)                     # [B, S]
    return per_slot.sum() / ok.to(ce.dtype).sum().clamp_min(1.0), per_slot


def mdm_unmask_counts(n_sym: int, k: int, schedule: str = "linear") -> list[int]:
    """How many symbols are COMMITTED after each of ``k`` rounds (cumulative, ends at
    ``n_sym``). ``linear``: ``ceil(N·(j+1)/k)``; ``cosine`` (MaskGIT): the fraction still
    masked after round ``j`` is ``cos(π/2 · (j+1)/k)``. Every round commits >= 1 symbol
    while any is left, so ``k <= N`` rounds are all real (``k > N`` is refused by the
    config)."""
    if k < 1 or n_sym < 1:
        raise ValueError(f"mdm_unmask_counts needs k >= 1 and n_sym >= 1, got {k}, {n_sym}")
    out, prev = [], 0
    for j in range(k):
        f = (j + 1) / k
        if schedule == "linear":
            done = math.ceil(n_sym * f - 1e-9)
        elif schedule == "cosine":
            done = n_sym - math.floor(n_sym * math.cos(math.pi / 2.0 * f) + 1e-9)
        else:
            raise ValueError(f"unknown mask schedule {schedule!r} (linear|cosine)")
        done = max(done, prev + 1) if prev < n_sym else n_sym
        done = min(done, n_sym)
        out.append(done)
        prev = done
    out[-1] = n_sym
    return out


@torch.no_grad()
def maskgit_sample(logits_fn, ok: Tensor, n_sym: int, mask_id: int, k: int,
                   schedule: str = "linear", generator=None) -> Tensor:
    """``k`` rounds of parallel unmasking (MaskGIT, Chang et al. 2022, without the
    Gumbel confidence noise): start all-MASK, each round predict every still-masked
    symbol, sample it, and COMMIT the ``n_j`` most confident of the newly sampled ones
    (confidence = the sampled symbol's log-probability), where ``n_j`` follows
    :func:`mdm_unmask_counts`. ``k = 1`` is one-shot sampling (every symbol from its own
    marginal given the past); ``k = N`` commits one symbol per round.

    ``logits_fn(idx [B, S, N] long, t [B, S] fp32 = fraction still masked) -> [B, S, N, C]``
    fp32. Returns ``[B, S, N]`` long with no MASK left on ``ok`` slots and ``0`` elsewhere.
    """
    B, S = ok.shape
    idx = torch.full((B, S, n_sym), int(mask_id), dtype=torch.long, device=ok.device)
    counts = mdm_unmask_counts(n_sym, int(k), schedule)
    done = torch.zeros(B, S, dtype=torch.long, device=ok.device)
    for j in range(int(k)):
        still = idx == mask_id                                              # [B, S, N]
        t = still.float().mean(dim=-1)                                      # [B, S]
        logits = logits_fn(idx, t)                                          # [B, S, N, C]
        logp = torch.log_softmax(logits.float(), dim=-1)
        probs = logp.exp().reshape(-1, logp.shape[-1])
        draw = torch.multinomial(probs, 1, generator=generator).reshape(B, S, n_sym)
        conf = logp.gather(-1, draw.unsqueeze(-1)).squeeze(-1)              # [B, S, N]
        conf = torch.where(still, conf, torch.full_like(conf, float("-inf")))
        n_new = (counts[j] - done).clamp_min(0)                             # [B, S]
        order = conf.argsort(dim=-1, descending=True)                       # [B, S, N]
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(n_sym, device=ok.device).view(1, 1, -1)
                      .expand(B, S, -1))
        commit = still & (rank < n_new.unsqueeze(-1))
        idx = torch.where(commit, draw, idx)
        done = (idx != mask_id).sum(dim=-1)
    idx = torch.where(idx == mask_id, torch.zeros_like(idx), idx)   # k rounds always finish
    return idx * ok.unsqueeze(-1).long()
