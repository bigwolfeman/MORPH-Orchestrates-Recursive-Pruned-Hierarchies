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
    "TULCodeEncoder", "TULCodeTime", "TULCodeHead",
    "code_rmsnorm", "code_target_valid", "code_thinker_relation",
    "cfm_pair", "cfm_null_floor", "euler_sample",
]

_SEED_ENC = 0xC0DE
_SEED_TIME = 0xC0DE + 1
_EPS = 1e-8


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
