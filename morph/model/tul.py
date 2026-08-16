"""TUL model pieces — slot parameters and the gather/scatter that the core loops on.

Spec: ``docs/tul-spec.md`` §3.2 (slot input embedding), §3.3 (core on slots only),
§3.4 (coda, token-state dropout, prefix projections), §5 (losses), §7.2 (metrics).
The forward that uses these lives in :mod:`morph.model.transformer`; this module
holds the parameters and the pure tensor plumbing so ``transformer.py`` stays
readable and every piece is unit-testable on its own.

Nothing here branches on a runtime flag: :class:`TULConfig` is resolved at
construction (spec §8), and ``slot_layout=None`` never reaches this module at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .tul_layout import SlotLayout

__all__ = ["TULConfig", "TULSlots", "bag_mean", "compact_index", "gather_positions",
           "scatter_positions"]


@dataclass
class TULConfig:
    """Construction-time TUL settings (spec §8). Mirrors the Hydra ``tul:`` block.

    Only the keys that change PARAMETERS or SHAPES live here; the schedule keys
    (``activate_at``) and the segmentation keys (``min_span``, ``span_cap``,
    ``fixed_stride``, ``boundary_chars``) belong to the loader and never reach the
    model. ``coda_sees_slots`` and ``tokens_through_core`` are construction-time
    by spec §8 — they change masks and gathers, not an ``if`` in the hot loop.
    """

    prefix_k: int = 2                    # coda positions per slot [W] (§3.1)
    slot_id: int = 4                     # "<fim_pad>"; its LM-head logit is −inf (§3.1)
    token_state_dropout: float = 0.15    # Bowman word dropout on the coda input (§3.4)
    slot_mean_depth: int = 0             # 0 → cfg.mean_depth
    slot_max_depth: int = 0              # 0 → cfg.max_depth
    coda_sees_slots: bool = True         # A4 sets False (§7.1)
    tokens_through_core: bool = False    # A2 sets True (§7.1)
    stp_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    set_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    carry: bool = False                  # arm (§3.5)
    xattn: bool = False                  # arm (§3.5)
    bcast: bool = False                  # arm (§3.5)

    def __post_init__(self) -> None:
        if self.prefix_k < 1:
            raise ValueError(f"tul.prefix_k must be ≥ 1, got {self.prefix_k}")
        # 1.0 is legal and meaningful: it is Bowman 2015's INPUTLESS decoder control
        # (every token state replaced by E_mask), the extreme end of the §3.4 arm sweep.
        if not 0.0 <= self.token_state_dropout <= 1.0:
            raise ValueError(
                f"tul.token_state_dropout must be in [0,1], got {self.token_state_dropout}")
        # Spec §3.5 lists these as arms that are NOT in v1. A config key that is silently
        # ignored is worse than a missing one — fail loudly instead (no-theater).
        for name in ("stp_lambda", "set_lambda"):
            if float(getattr(self, name)) != 0.0:
                raise NotImplementedError(
                    f"tul.{name}={getattr(self, name)} — the {name.split('_')[0]} arm "
                    f"(spec §3.5/§5) is specified but NOT implemented in v1. Set it to 0.0; "
                    f"do not run the arm until the loss term exists."
                )
        for name in ("carry", "xattn", "bcast"):
            if bool(getattr(self, name)):
                raise NotImplementedError(
                    f"tul.{name}=true — arm '{name}' (spec §3.5) is specified but NOT "
                    f"implemented in v1. Leave it false."
                )


# ── pure tensor plumbing ─────────────────────────────────────────────────────

def bag_mean(signal: Tensor, bag_id: Tensor, token_sel: Tensor, n_bags: int) -> Tensor:
    """Mean of ``signal`` over the TOKEN positions of each span (spec §3.2).

    This is the TST ``ve_bagged`` operation with a data-dependent bag map instead of
    a fixed stride: the slot's input is the mean of its span's token embeddings
    (Dynamic Token Pooling: mean-pool beats take-last; BLT Eq. 5 uses the mean as the
    pooling query init). Gradient flows to the embedding table through the scatter.

    Args:
        signal:    ``[B, L, C]`` per-position signal (token embedding, bigram, value-embed).
        bag_id:    ``[B, L]`` int64 bag index; ``n_bags`` is the dump bin.
        token_sel: ``[B, L]`` float 1.0 at token positions, 0.0 at slot positions —
                   slot positions must not pollute their own bag.
        n_bags:    number of real bags (``max_slots``).

    Returns:
        ``[B, n_bags + 1, C]``; row ``n_bags`` is the dump bin and is exactly 0, so a
        gather at the dump bin contributes nothing (tail pads get ``E_slot`` alone).
    """
    B, L, C = signal.shape
    n_out = n_bags + 1
    sel = token_sel.unsqueeze(-1).to(signal.dtype)
    # Fold the batch into the bag index so one flat index_add_ does the whole batch —
    # much cheaper than a scatter_add_ against a [B, L, C]-expanded index view.
    flat_bag = (bag_id + torch.arange(B, device=bag_id.device).unsqueeze(1) * n_out).reshape(-1)
    acc = signal.new_zeros(B * n_out, C)
    acc.index_add_(0, flat_bag, (signal * sel).reshape(-1, C))
    cnt = signal.new_zeros(B * n_out)
    cnt.index_add_(0, flat_bag, sel.reshape(-1))
    out = (acc / cnt.clamp(min=1.0).unsqueeze(-1)).view(B, n_out, C)
    # The dump bin aggregates trailing tokens that have no slot; zero it so the gather
    # at slot positions of tail pads reads 0 rather than a stray span mean.
    out = torch.cat([out[:, :n_bags], out.new_zeros(B, 1, C)], dim=1)
    return out


def gather_positions(x: Tensor, index: Tensor) -> Tensor:
    """Gather along the sequence axis. ``x``: ``[B, L, …]``, ``index``: ``[B, N]`` → ``[B, N, …]``.

    ``index`` may address row ``L`` — the caller is expected to have appended a zero
    dump row, which is how a variable-length compaction keeps a static shape.
    """
    idx = index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])
    return torch.gather(x, 1, idx)


def gather_valid(x: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """Gather ``[B, N]`` positions, zeroing the rows whose ``valid`` is False.

    Equivalent to appending a zero dump row and pointing invalid entries at it, but
    WITHOUT materialising that copy — the carrier is ``[B, L, n, C]`` fp32 after
    ``input_norm`` (335 MB at the 1024×16 arm shape), so the pad copy is the single
    largest avoidable allocation on the TUL path.
    """
    safe = torch.where(valid, index, torch.zeros_like(index))
    out = gather_positions(x, safe)
    return out * valid.reshape(*valid.shape, *([1] * (x.dim() - 2))).to(out.dtype)


def scatter_positions(x: Tensor, index: Tensor, values: Tensor) -> Tensor:
    """Out-of-place scatter along the sequence axis with a dump row.

    ``x``: ``[B, L, …]``, ``index``: ``[B, N]`` (entries in ``[0, L]``; ``L`` = discard),
    ``values``: ``[B, N, …]``. Returns ``[B, L, …]``. One extra row makes invalid slots
    free of a per-row mask and keeps the shape static.

    The scatter is IN-PLACE on the freshly concatenated buffer: ``cat``'s backward needs
    only to slice the incoming gradient, never its own output, so mutating it is
    autograd-safe and saves a second full-carrier copy.
    """
    B, L = x.shape[0], x.shape[1]
    pad = torch.cat([x, x.new_zeros(B, 1, *x.shape[2:])], dim=1)
    idx = index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])
    # Cast at the boundary, as every other injection site in the model does: under
    # autocast RMSNorm returns fp32 (its fp32 weight promotes the product) while the
    # prefix projection comes out of a bf16 matmul, and scatter demands one dtype.
    pad.scatter_(1, idx, values.to(pad.dtype))
    return pad[:, :L]


def compact_index(slot_mask: Tensor) -> Tensor:
    """``[B, L]`` gather map that moves TOKEN positions to the front, slots to a dump row.

    Spec §7.2 / arm A4: dropping slots from the coda is done by a gather, "exact, and it
    needs no per-position attention mask the fused kernels may not have". Token order is
    preserved (stable sort), so the compacted sequence is the plain token stream; the
    tail addresses row ``L`` and is discarded by :func:`gather_positions`' dump row.
    """
    B, L = slot_mask.shape
    order = torch.argsort(slot_mask.to(torch.uint8), dim=1, stable=True)
    n_tok = (~slot_mask).sum(dim=1, keepdim=True)
    pos = torch.arange(L, device=slot_mask.device).unsqueeze(0)
    return torch.where(pos < n_tok, order, torch.full_like(order, L))


# ── parameters ───────────────────────────────────────────────────────────────

class TULSlots(nn.Module):
    """The three TUL parameter groups (spec §3.1/§3.2/§3.4/§5).

    * ``E_slot`` ``[d]`` — the slot token's own embedding, added to the span bag-mean.
      Initialised to the MEAN of the embedding table at the activation step, following
      Block Transformer §3.7's uptraining recipe ("init block embedding = mean of token
      embeddings"), which recovers near-full performance from a vanilla checkpoint with
      ~10 % of the tokens. See :meth:`init_at_activation`.
    * ``E_mask`` ``[d]`` — the learned vector that replaces a dropped token state in the
      coda (Bowman word dropout / He 2019 §3.1 / Optimus "tax the cheap channel"). Init 0.
    * ``W_prefix`` ``[prefix_k, d, d]`` — one looped state ``h_i`` projected into the
      slot's ``prefix_k`` coda positions (spec §3.1; Block Transformer App. F.2 / Fig 3f
      picks prefix length 2 over 1). Init identity, so at the activation step both coda
      positions see ``h_i`` unchanged and the extra position costs nothing.

    Constructed only when TUL is configured, and LAST in ``MORPHTransformer.__init__``
    so a non-TUL model is byte-identical to the baseline (the ``attach_retention``
    convention). All three inits are deterministic — zero RNG draws — so even the TUL
    model's base weights match a baseline built with the same seed.
    """

    def __init__(self, d_model: int, tul: TULConfig):
        super().__init__()
        self.tul = tul
        self.E_slot = nn.Parameter(torch.zeros(d_model))
        self.E_mask = nn.Parameter(torch.zeros(d_model))
        eye = torch.eye(d_model).unsqueeze(0).repeat(tul.prefix_k, 1, 1)
        self.W_prefix = nn.Parameter(eye)

    @torch.no_grad()
    def init_at_activation(self, lm_weight: Tensor) -> None:
        """Set ``E_slot`` to the mean of the (live, trained) embedding table — spec §5.

        Called at the activation step, not at construction: the point of the Block
        Transformer init is that the new position starts as the average TRAINED token,
        which a randomly-initialised table cannot provide. Idempotent-unsafe by design —
        the training loop calls it exactly once and records it in the checkpoint.
        """
        self.E_slot.copy_(lm_weight.mean(dim=0).to(self.E_slot.dtype))

    # -- forward helpers ---------------------------------------------------
    def slot_input(self, signal: Tensor, layout: SlotLayout, add_e_slot: bool) -> Tensor:
        """Replace slot positions of ``signal`` ``[B, L, C]`` with the span bag-mean.

        ``add_e_slot`` is True for the token embedding (the slot's input embedding is
        ``E_slot + mean_j embed(t_j)``, spec §3.2) and False for the bigram / value-embed
        signals, whose slot value is the plain bag-mean of the span ("bigram/value-embed
        signals for the slot are the bag-mean, exactly the TST ``ve_bagged`` path").
        """
        token_sel = (~layout.slot_mask).to(signal.dtype)
        bags = bag_mean(signal, layout.bag_id, token_sel, layout.max_slots)
        at_pos = torch.gather(
            bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, signal.shape[-1]))
        if add_e_slot:
            at_pos = at_pos + self.E_slot.to(signal.dtype)
        return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

    def prefix_project(self, h_slots: Tensor, layout: SlotLayout, l_total: int) -> Tensor:
        """``[B, S, …, C]`` looped states → ``[B, S·prefix_k]`` values and their positions.

        Returns ``(values, index)`` ready for :func:`scatter_positions`: value ``k`` of
        slot ``s`` is ``h_s W_k`` and lands at ``slot_index[s] + k``. Invalid slots address
        the dump row. Spec §3.1: the first ``prefix_k − 1`` positions carry the plan with
        NO label, the last one predicts the first token of the next span.
        """
        K = self.tul.prefix_k
        B, S = layout.slot_index.shape
        C = h_slots.shape[-1]
        mid = h_slots.shape[2:-1]                      # () plain carrier, (n,) HC carrier
        w = self.W_prefix.to(h_slots.dtype)
        # [B,S,M,C] ⊗ [K,C,C] → [B,S,K,M,C] by broadcast matmul (batch dims (B,S,1)×(1,1,K)).
        hm = h_slots.reshape(B, S, -1, C)
        proj = torch.matmul(hm.unsqueeze(2), w.view(1, 1, K, C, C))
        values = proj.reshape(B, S * K, *mid, C)       # slot-major: index s·K + k
        offs = torch.arange(K, device=layout.slot_index.device)
        pos = layout.slot_index.unsqueeze(-1) + offs                      # [B, S, K]
        pos = torch.where(layout.slot_valid.unsqueeze(-1), pos, l_total)
        return values, pos.reshape(B, S * K)

    def apply_token_dropout(self, x: Tensor, layout: SlotLayout, training: bool
                            ) -> tuple[Tensor, Tensor | None]:
        """Replace a fraction ``p`` of TOKEN coda inputs with ``E_mask`` (spec §3.4).

        Returns ``(x, keep)`` where ``keep`` is ``[B, L, 1]`` (1.0 kept, 0.0 dropped) or
        None when nothing was dropped.

        RESOLVED SPEC AMBIGUITY: the drop also zeroes the CODA's x0 / bigram injection at
        the dropped positions. §3.4 only says "the coda input is replaced", but x0 is
        ``proj(embed(t))`` and the bigram term is a hash of ``(t, t−1)`` — both are injected
        into every coda layer, so leaving them would hand the token's own identity straight
        back and make Bowman's word dropout a no-op after the injection scales train up.
        The stated purpose ("the position must then be decoded from the plan slots and its
        neighbours through attention") requires the token to be genuinely absent.
        """
        p = self.tul.token_state_dropout
        if not training or p <= 0.0:
            return x, None
        drop = (torch.rand(layout.slot_mask.shape, device=x.device) < p) & (~layout.slot_mask)
        keep = (~drop).to(x.dtype).unsqueeze(-1)                       # [B, L, 1]
        mask_vec = self.E_mask.to(x.dtype)
        if x.dim() == 4:                       # HC carrier [B, L, n, C]
            x = torch.where(drop[:, :, None, None], mask_vec, x)
        else:
            x = torch.where(drop[:, :, None], mask_vec, x)
        return x, keep
