"""Energies for the gradient-conditioned slot passes (``tul.grad_pass_energy``).

What this module is for
-----------------------
``tul.grad_pass`` (``morph/model/tul.py::TULGradPass``) hands every pass of the slot loop
the gradient of a local objective with respect to the state it is refining, mapped through
a zero-initialised ``W_g``. The gradient is a CONDITIONING INPUT, not the update — the
distinction is load-bearing: a scalar-energy gradient used AS the update is measured
weaker than a learned vector transition conditioned on it, which is exactly the form
``TULGradPass`` already has.

Until now there was ONE energy: ``_tul_mux_loss(target="own")``, the slot's own span read
through the tied head against an ORDER-FREE geometric bag. The arm that ran it
(``slot-mnext-gradpass``, 2026-09-10) descended that energy 0.366 nats in the FIRST pass
and then sat flat for the remaining seven. The reading in its filing: *the own span is
already in the entry state, so a bag target is reachable in one step*. This module
supplies two energies that are not reachable in one step for that reason.

``recon`` — the reconstruction energy
    A SECOND :class:`~morph.model.tul_spandec.SpanDecoder`, with its own parameters, that
    reconstructs the slot's OWN span from ``z``, teacher-forced on that span's tokens.
    Conditional, ordered and per token, where the MUX energy is one marginal per slot.
    Optionally with SOFT targets (``tul.egrad_soft_labels``): the target at decoder
    position ``j`` is ``(1 - m) * onehot(t_j) + m * bag(span)``, the Label-Forcing mixture.
    Because cross-entropy is linear in the target, that mixture is computed EXACTLY as

        ``(1 - m) * FLCE(hard labels)  +  m * FLMCE(the span's token multiset)``

    — two calls to the tree's own chunked cross-entropy kernels
    (``morph/model/fused_ce.py``), so no ``[B, S, J, V]`` logit tensor is ever
    materialised and there is no second CE implementation to keep in step.

``critic`` — the within-context improvement critic
    ``E = -c_phi(z, ctx)``, the same SHAPE as ``disc`` and a different QUESTION. ``disc``
    asks "is this slot's next span easier than the batch median?", which mostly reads how
    predictable the next span is and barely reads ``z``. ``critic`` asks "does the state
    after pass ``t`` beat the state after pass ``t-1`` ON THIS ROW, through the REAL
    coda?" — a within-context comparison, so everything the two candidates share (the
    context, the span, the row's difficulty) cancels. Its label comes from replaying the
    shipped coda with each candidate written into the slot's prefix cells and measuring
    the next span's token CE; the replays are no-grad, so no label ever reaches the loop.
    Trained by a PAIRWISE logistic loss weighted by the CE gap.

``disc`` — the discriminative energy
    ``E = -s_phi(z, ctx)``, ``s_phi`` a 2-layer MLP on the mean-stream ``z`` and the slot's
    prelude-entry state. It is trained with BCE against the OUTCOME the probe measures —
    "is the coda's mean CE over this slot's next span below the batch median?" — on a
    stop-gradient copy of ``z``, with shuffled-context negatives so the head cannot score
    from the context alone. Never a regression onto ``z``: the latent is an input to the
    scorer, never a target (the spec's own rule, and the LCM / CoCoMix failure it records).

Contracts shared by both
------------------------
* **The energy never shapes ``z`` directly.** The energy module's own training loss is
  computed on ``z.detach()``, so its gradient reaches its own parameters and stops. The
  ONLY route from an energy into the loop is ``W_g``, and that route carries a DETACHED
  feature (``create_graph=False``), so no second-order term exists.
* **Off is nothing.** ``tul.grad_pass_energy: "own_mux"`` (the default) builds neither
  class, draws no RNG and leaves the forward exactly as it was.
* **Never quantised, never pruned.** Every ``nn.Linear`` here carries
  ``_ternary_exclude = True`` and none is a ``MortarLinear``, so the ternary QAT scope,
  the CMS prune, the MORTAR carve, the ReMoE router and the deploy packer all walk past.
  These are TRAINING-ONLY scorers and are not in the deployed forward.
* **RNG-neutral.** Private generators only (the ``TULSlots.W_sent`` rule), so an arm that
  turns an energy on holds the same weights as its ruler everywhere else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .fused_ce import fused_linear_cross_entropy, fused_linear_cross_entropy_mce
from .tul_layout import SlotLayout
from .tul_spandec import SpanDecoder, own_span_slots

__all__ = ["ReconEnergy", "DiscEnergy", "CriticEnergy", "per_token_ce",
           "slot_outcome_labels"]

# Private init streams for the scalar heads. Never the global RNG.
_SEED_DISC = 0x5D15C
_SEED_CRITIC = 0xC1717C


def _init_linear(m: nn.Linear, gen: torch.Generator) -> None:
    with torch.no_grad():
        m.weight.copy_(torch.empty(m.weight.shape, device="cpu").normal_(
            mean=0.0, std=0.02, generator=gen))
        if m.bias is not None:
            m.bias.zero_()


# ── the reconstruction energy ─────────────────────────────────────────────────


class ReconEnergy(nn.Module):
    """``E_recon(z) = CE of the slot's OWN span, teacher-forced, decoded from ``z``.

    A second :class:`~morph.model.tul_spandec.SpanDecoder`. It is a SEPARATE instance from
    ``tul.spandec``'s target decoder even when both are built: the target decoder grades
    the NEXT span and is the loss the loop is judged on, this one reconstructs the OWN
    span and is the potential the loop is handed the gradient of. Sharing them would make
    the conditioning feature the gradient of the training target, which is the oracle the
    ``slot_z_optimize`` probe is, not a mechanism a generator could run.

    ``seed_offset`` shifts the decoder's private init stream so the two decoders of a
    ``spandec`` + ``recon`` arm do not start from identical weights.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 max_tokens: int, soft_labels: bool, soft_mix: float):
        super().__init__()
        if not 0.0 <= soft_mix <= 1.0:
            raise ValueError(f"tul.egrad_soft_mix must be in [0, 1], got {soft_mix}")
        self.dec = SpanDecoder(d_model, n_heads, d_ff, n_layers, max_tokens,
                               seed_offset=0x11)
        self.soft_labels = bool(soft_labels)
        self.soft_mix = float(soft_mix)

    @property
    def max_tokens(self) -> int:
        return self.dec.max_tokens

    def loss(self, z: Tensor, input_ids: Tensor, layout: SlotLayout, w_head: Tensor,
             emb: Tensor, chunk: int, mask_token_id: int,
             slot_keep: Tensor | None = None) -> Tensor:
        """``z [B, S, C]`` -> scalar. ``emb`` is the DETACHED input table, ``w_head`` the
        output head (the caller applies ``tul.mux_detach_head`` to it, exactly as
        ``_tul_spandec_loss`` does).

        ``slot_keep`` ``[B, S]`` restricts the supervised slots — inside the loop it is the
        pass's active-and-valid set, so a frozen or pad slot contributes nothing to the
        mean and reads a zero feature, the same contract ``_own_span_grad`` has.
        """
        ids, valid = own_span_slots(input_ids, layout, self.dec.max_tokens)
        if slot_keep is not None:
            valid = valid & slot_keep.unsqueeze(-1)
        st = self.dec.decode(z, ids, valid, emb)                      # [B, S, J, C]
        C = st.shape[-1]
        flat = st.reshape(-1, C)
        hard = torch.where(valid, ids, torch.full_like(ids, -100)).reshape(-1)
        ce_hard = fused_linear_cross_entropy(
            flat, w_head, hard, ignore_index=-100, chunk_size=chunk,
            mask_token_id=mask_token_id)
        if not self.soft_labels or self.soft_mix == 0.0:
            return ce_hard
        # The bag half. `labels` is [N, K]: EVERY token of the row's own span, for every
        # decoder position of that span. `fused_linear_cross_entropy_mce` scores
        # `lse - mean_i logit[t_i]`, i.e. the CE against the UNIFORM distribution over that
        # multiset — repeated tokens carry more weight because they appear more than once,
        # which is what "the bag distribution over the span's tokens" means. A row whose own
        # position is invalid gets an all-ignore label row, so BOTH kernels reduce over the
        # same set of rows (tests/test_tul_egrad.py pins the two counts together).
        B, S, J = ids.shape
        # The span's own tokens, with the past-the-end offsets removed: `ids` carries the
        # scatter's fill (0) there, and leaving it in would put the vocab-0 row into every
        # bag with multiplicity J - len(span).
        row = torch.where(valid, ids, torch.full_like(ids, -100))     # [B, S, K=J]
        bag = row.unsqueeze(2).expand(B, S, J, J)                     # [B, S, J, K]
        # A decoder position that is itself invalid scores nothing, so both kernels reduce
        # over exactly the same rows.
        bag = torch.where(valid.unsqueeze(-1), bag, torch.full_like(bag, -100))
        ce_bag = fused_linear_cross_entropy_mce(
            flat, w_head, bag.reshape(-1, J), ignore_index=-100, chunk_size=chunk,
            mask_token_id=mask_token_id)
        m = self.soft_mix
        return (1.0 - m) * ce_hard + m * ce_bag


# ── the discriminative energy ─────────────────────────────────────────────────


class DiscEnergy(nn.Module):
    """``E_disc(z) = -mean s_phi(z, ctx)``: a critic on the slot state.

    ``s_phi`` is a 2-layer MLP on ``[z, ctx]``, ``ctx`` the slot's prelude-entry state
    (the constant the loop started from, so the head can score "how far has this state
    moved from where it began, and did that help"). Its output is a LOGIT; the energy the
    loop descends is the negated mean logit over the pass's active slots, so descending it
    pushes the state toward what the critic calls a good slot.

    The critic's own loss (:meth:`bce`) is BCE against the measured outcome — the coda's
    mean CE over the slot's next span, thresholded at the batch median — on a DETACHED
    ``z``. It is a classifier of the latent, never a regressor onto it.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        h = int(hidden or d_model)
        gen = torch.Generator(device="cpu").manual_seed(_SEED_DISC)
        self.norm = RMSNorm(2 * d_model)
        self.fc1 = nn.Linear(2 * d_model, h, bias=True)
        self.fc2 = nn.Linear(h, 1, bias=True)
        for m in (self.fc1, self.fc2):
            _init_linear(m, gen)
            m._ternary_exclude = True

    def score(self, z: Tensor, ctx: Tensor) -> Tensor:
        """``[B, S, C]``, ``[B, S, C]`` -> ``[B, S]`` logits."""
        x = torch.cat([z, ctx], dim=-1)
        x = self.norm(x).to(z.dtype)
        return self.fc2(F.silu(self.fc1(x))).squeeze(-1)

    def energy(self, z: Tensor, ctx: Tensor, mask: Tensor) -> Tensor:
        """Scalar energy over the masked slots: ``-mean s_phi``."""
        s = self.score(z, ctx).float()
        w = mask.to(s.dtype)
        return -(s * w).sum() / w.sum().clamp(min=1.0)

    def bce(self, z: Tensor, ctx: Tensor, labels: Tensor, valid: Tensor,
            shuffle_negatives: bool = True) -> Tensor:
        """BCE on the detached latent, plus shuffled-context negatives.

        ``labels`` ``[B, S]`` float in {0, 1}; ``valid`` ``[B, S]`` bool. The caller passes
        ``z`` and ``ctx`` ALREADY DETACHED — this method never opens a route from the
        critic's loss into the loop.

        The negatives pair each slot's ``z`` with ANOTHER slot's context (a roll along the
        slot axis, within the row) and target 0. Without them the critic can reach a good
        AUC from ``ctx`` alone — the span's own length and content predict the next span's
        CE — and the energy would then carry no information about ``z`` at all, which is
        exactly the failure mode this arm exists to avoid.
        """
        s = self.score(z, ctx).float()
        w = valid.to(s.dtype)
        pos = F.binary_cross_entropy_with_logits(s, labels.to(s.dtype), reduction="none")
        loss = (pos * w).sum() / w.sum().clamp(min=1.0)
        if shuffle_negatives and z.shape[1] > 1:
            ctx_sh = torch.roll(ctx, shifts=1, dims=1)
            w_sh = (valid & torch.roll(valid, shifts=1, dims=1)).to(s.dtype)
            s_sh = self.score(z, ctx_sh).float()
            neg = F.binary_cross_entropy_with_logits(
                s_sh, torch.zeros_like(s_sh), reduction="none")
            loss = loss + (neg * w_sh).sum() / w_sh.sum().clamp(min=1.0)
        return loss

    @staticmethod
    def train_auc(z: Tensor, ctx: Tensor, labels: Tensor, valid: Tensor,
                  scorer) -> Tensor:
        """Rank-based ROC-AUC of the critic on THIS batch, as a 0-dim tensor.

        Mann-Whitney U on the ranks, so it needs no sort-free approximation and no host
        sync inside the forward. Returns 0.5 when either class is empty.
        """
        with torch.no_grad():
            s = scorer(z, ctx).float()[valid]
            y = labels[valid] > 0.5
            n_pos, n_neg = y.sum(), (~y).sum()
            if int(n_pos) == 0 or int(n_neg) == 0:
                return s.new_tensor(0.5)
            r = torch.argsort(torch.argsort(s)).float() + 1.0
            return (r[y].sum() - n_pos.float() * (n_pos.float() + 1) / 2) / \
                   (n_pos.float() * n_neg.float())


# ── the within-context improvement critic ─────────────────────────────────────


class CriticEnergy(nn.Module):
    """``E_critic(z) = -mean c_phi(z, ctx)``: a critic on WITHIN-CONTEXT improvement.

    THE SHAPE is :class:`DiscEnergy`'s — a 2-layer MLP on ``[z, ctx]``, ``ctx`` the slot's
    prelude-entry state, whose gradient with respect to ``z`` is what conditions each pass.
    THE QUESTION is different, and that is the whole arm.

    ``disc``'s label is "is this slot's next span below the BATCH MEDIAN CE?". Most of that
    signal is how predictable the next span happens to be — a property of the text, not of
    ``z`` — which is why it needs shuffled-context negatives to stop the head scoring from
    ``ctx`` alone. Wolfe, 2026-09-12: "a within-context critic that scores whether the
    state after pass t beats the state after pass t-1 on the same coda loss."

    ``critic``'s label is a PAIR on ONE row: two candidate slot states are each written into
    that slot's prefix cells, the REAL coda is replayed, and the next span's mean token CE
    is measured for each. Everything the two candidates share — the context, the span, the
    row's difficulty, the decoder — cancels in the comparison, so the label is about the
    state and nothing else. The pairs are (h_{t-1}, h_t) for a sampled pass ``t`` and
    (h_t, h_t + eps*rms(h_t)*n), a local perturbation, so the critic sees both "did this
    pass help" and "which way is up from here".

    THE LOSS is pairwise logistic on the SCORE DIFFERENCE, weighted by the measured CE gap:

        ``w * BCE(c_phi(a) - c_phi(b), 1{CE_a < CE_b})``,  ``w = |CE_a - CE_b|``

    A gap of zero contributes zero, so a pair the coda cannot tell apart teaches nothing
    instead of teaching a coin flip. Lower CE is better, so the target is 1 when ``a`` is
    the better state, and the energy the loop descends is ``-c_phi`` — descending it pushes
    the state toward what the critic calls better.

    THE CONTRACTS are the module's, shared with ``recon`` and ``disc``: the training loss
    runs on a DETACHED ``z`` (the caller detaches, as it does for ``disc``), so the critic's
    own gradient never enters the loop; the only route in is ``W_g``; every leaf is
    ``_ternary_exclude``; the init stream is private, so an arm with the critic on holds
    byte-identical weights to its ruler everywhere else.

    NOT A REGRESSION ONTO ``z``. The standing rule (LCM T3/4, CoCoMix §6b, BT §4.2) is that
    the latent is an INPUT to a scorer, never a target. ``z`` is an input here and the label
    is a measured outcome, which is the same direction ``disc`` takes.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        h = int(hidden or d_model)
        gen = torch.Generator(device="cpu").manual_seed(_SEED_CRITIC)
        self.norm = RMSNorm(2 * d_model)
        self.fc1 = nn.Linear(2 * d_model, h, bias=True)
        self.fc2 = nn.Linear(h, 1, bias=True)
        for m in (self.fc1, self.fc2):
            _init_linear(m, gen)
            m._ternary_exclude = True

    def score(self, z: Tensor, ctx: Tensor) -> Tensor:
        """``[B, S, C]``, ``[B, S, C]`` -> ``[B, S]`` scores. Higher = a better state.

        NO LABEL TOKEN REACHES THIS. Its two inputs are the current slot state and the
        slot's own prelude-entry state, both causal for that slot, so the conditioning
        feature a generator would compute at inference is the one trained here
        (``tests/test_tul_critic.py`` asserts the forward is called with these two alone).
        """
        x = torch.cat([z, ctx], dim=-1)
        x = self.norm(x).to(z.dtype)
        return self.fc2(F.silu(self.fc1(x))).squeeze(-1)

    def energy(self, z: Tensor, ctx: Tensor, mask: Tensor) -> Tensor:
        """Scalar energy over the masked slots: ``-mean c_phi``.

        The SAME signature ``DiscEnergy.energy`` has, so ``_egrad_feature`` dispatches to
        it unchanged and the conditioning path is byte-for-byte the ``disc`` arm's.
        """
        s = self.score(z, ctx).float()
        w = mask.to(s.dtype)
        return -(s * w).sum() / w.sum().clamp(min=1.0)

    def pairwise(self, z_a: Tensor, z_b: Tensor, ctx: Tensor, ce_a: Tensor, ce_b: Tensor,
                 valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(loss, n_pairs, agreement)`` for one candidate pair.

        ``z_a`` / ``z_b`` ``[B, S, C]`` are the two candidates ALREADY DETACHED by the
        caller; ``ctx`` is the shared context (the same for both — a pair that did not
        share its context would be comparing two different questions, which is the
        sabotage ``tests/test_tul_critic.py`` plants); ``ce_a`` / ``ce_b`` ``[B, S]`` are
        the measured next-span CEs; ``valid`` ``[B, S]`` selects the scored slots.

        ``agreement`` is the fraction of scored pairs the critic already ranks correctly —
        the arm's honesty instrument, the twin of ``egrad_auc``. A critic stuck at 0.5
        means the energy carries nothing and the arm is its ruler with an extra injection
        channel.
        """
        s_a = self.score(z_a, ctx).float()
        s_b = self.score(z_b, ctx).float()
        gap = (ce_b - ce_a).float()                                   # > 0 <=> a is better
        w = valid.to(gap.dtype) * gap.abs()
        y = (gap > 0).to(gap.dtype)
        d = s_a - s_b
        per = F.binary_cross_entropy_with_logits(d, y, reduction="none")
        denom = w.sum().clamp(min=1e-6)
        loss = (per * w).sum() / denom
        n = valid.to(gap.dtype).sum()
        agree = (((d > 0).to(gap.dtype) == y).to(gap.dtype) * w).sum() / denom
        return loss, n, agree


# ── the outcome label the critic is trained against (and the Step-0 probe reads) ──


@torch.no_grad()
def per_token_ce(x: Tensor, w: Tensor, labels: Tensor, chunk: int,
                 ignore_index: int = -100) -> Tensor:
    """``[N, d]``, ``[V, d]``, ``[N]`` -> ``[N]`` per-row CE, NO autograd graph.

    The chunked kernels in ``fused_ce.py`` reduce to a scalar because that is what a loss
    is. This is a METRIC — the outcome each slot is graded by — so it needs the per-row
    values and no backward at all, which is why it is a plain chunked loop here rather
    than a reduction knob on a hot-path kernel.
    """
    out = x.new_empty(x.shape[0], dtype=torch.float32)
    safe = labels.clamp(min=0)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        lg = (x[s:e] @ w.to(x.dtype).t()).float()
        out[s:e] = F.cross_entropy(lg, safe[s:e], reduction="none")
        del lg
    return torch.where(labels != ignore_index, out, torch.zeros_like(out))


@torch.no_grad()
def slot_outcome_labels(xh: Tensor, labels: Tensor, layout: SlotLayout, w_head: Tensor,
                        chunk: int) -> tuple[Tensor, Tensor, Tensor]:
    """``(y [B, S] float, scored [B, S] bool, ce [B, S] float)``.

    ``y[b, s] = 1`` when the coda's MEAN token CE over slot ``s``'s NEXT span is below the
    median over every scored slot of the batch, else 0. That is the label the Step-0 linear
    probe fits (``lab/divergence/slot_state_linear_probe.py``) and the label the ``disc``
    critic is trained against, so the two read the same quantity by construction.

    "Slot ``s``'s next span" is the token positions with ``bag_id == s + 1``, which is
    exactly the token set :func:`morph.model.tul_spandec.next_span_slots` gathers. A slot
    is scored only when that span holds at least one token with a real label.
    """
    B, L, C = xh.shape
    S = layout.slot_index.shape[1]
    ce = per_token_ce(xh.reshape(-1, C), w_head, labels.reshape(-1), chunk).reshape(B, L)
    tok = (~layout.slot_mask) & (labels != -100)
    k = layout.bag_id
    # token of span k grades slot k-1
    tgt = (k - 1).clamp(min=0)
    keep = tok & (k >= 1) & (k <= S)
    idx = torch.where(keep, tgt, torch.full_like(tgt, S))          # dump column
    num = ce.new_zeros(B, S + 1).scatter_add_(1, idx, ce * keep.to(ce.dtype))
    den = ce.new_zeros(B, S + 1).scatter_add_(1, idx, keep.to(ce.dtype))
    num, den = num[:, :S], den[:, :S]
    scored = (den > 0) & layout.slot_valid
    mean_ce = num / den.clamp(min=1.0)
    vals = mean_ce[scored]
    med = vals.median() if vals.numel() else mean_ce.new_tensor(0.0)
    return (mean_ce < med).to(ce.dtype) * scored.to(ce.dtype), scored, mean_ce
