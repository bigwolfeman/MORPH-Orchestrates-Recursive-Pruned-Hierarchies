"""The parallel span head: a COMMITTED product reader of the next span (LXTUL-E).

TRAINING-ONLY TARGET AND SCORER. IT IS NEVER A DEPLOYED DECODER.
--------------------------------------------------------------------
The root ``CLAUDE.md`` says: never decode a span from one vector plus an offset with no
token path (Huginn 2026-08-16, MegaByte T7, Bowman T2, Hourglass T6). This head IS that
reader, on purpose. Wolfe approved it on 2026-09-23 ("lets try it") as a training target
and a scorer only, under the design note
``.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md``
(LXTUL-E). The coda keeps the token path and stays the only decoder that generates text.
Nothing in ``morph/inference/`` reads this module, and nothing may.

Why this module exists
----------------------
The Lean development ``lab/theory/tul_exploration/`` proves three facts the design rests on:

* ``multisample_le_bayes`` / ``multisample_bayes_attained``: a reader that can HEDGE (the
  teacher-forced coda, the teacher-forced :class:`~morph.model.tul_spandec.SpanDecoder`)
  gains nothing from width. One deterministic latent already attains its optimum.
* ``product_reader_le``: a COMMITTED product reader, one that decodes every token of the
  span in parallel from ONE latent with no token input, is capped at the product of the
  span's marginals. On a two-token span whose tokens are always equal the cap is
  ``-2 log 2``.
* ``enumerated_pair_gt``: ``K`` enumerated codes with committed product readers, scored
  under the EXACT mixture likelihood, beat that cap.

So width can only be seen by a committed reader, and this is that reader. Stage 0 of the
note measures the width budget on the FROZEN ruler before any loop is trained::

    B0 = CE_par(K = 1) - CE_par(K = 4, enumerated-code mixture)    per span token

and stops the design if ``B0 < 0.005``. Phase B hands the same head ``K`` per-rollout
exit states (the code already inside the state) and reuses the same mixture math.

What it is
----------
For each supervised slot ``s`` the head runs ``J`` INPUT-FREE queries::

    q_j  = z_in(z_s) + pos_j          (K = 1)
    q_kj = z_in(z_s + rms(z_s) u_k) + pos_j     (K > 1, code k, u mean-free over k)

through ``n_layers`` :class:`~morph.model.tul_spandec._SpanDecBlock` blocks, the SAME block
the teacher-forced decoder uses, at the ruler decoder's width, heads and depth. Position
``j`` predicts token ``j`` of span ``s + target_offset``. No token of that span, or of
any span, enters: the only inputs are ``z``, the code and the position.

Self-attention among the ``J`` queries is allowed and keeps the product-reader property.
Every query is a function of ``(z, u_k, j)`` alone, so every block output at position
``j`` is a function of ``(z, u_k, j)`` alone, and ``p(t_0 .. t_{J-1} | z, u_k) =
prod_j p(t_j | z, u_k, j)`` is still a product of per-position marginals: the reader class
``product_reader_le`` is about. The blocks are causal because ``_SpanDecBlock`` is; with
input-free queries causal and bidirectional attention see the same information.

The mixture
-----------
Per supervised slot, ``log p_k = sum_j valid log p(t_j | z, u_k, j)``. The loss is::

    L = - sum_slots [ logsumexp_k(log p_k) - log K ] / sum(valid tokens)

It is the exact mixture likelihood over the ``K`` enumerated codes (note W3: enumerate, do
not sample). At ``K = 1`` it IS the plain parallel CE, mean over valid span tokens, and
``logsumexp`` of one value is that value, so the two agree to summation order.
:func:`mixture_span_nll` is the one home for this math; Phase B calls it on rollouts.

Where the code enters (a choice the note did not fix)
-----------------------------------------------------
In z space, scaled by the slot's own RMS, detached: ``z_k = z + rms(z).detach() * u_k``.
This mirrors Phase B's per-pass injection ``h <- f(h) + r * rms(h).detach() * u_k`` so
Stage 0 and Phase B put the code in the same place relative to the state the head reads.
``u`` is initialised at per-coordinate std ``code_init`` (0.1, the note's ``r``) from a
PRIVATE generator and is learned, scale included. ``u`` is made mean-free across ``k`` at
use (:meth:`ParallelSpanHead.code_offsets`), so a code is a DIRECTION of choice and never
a shared shift the ``K = 1`` twin could not also learn. At ``K = 1`` there is no code
table at all (``codes is None``) and no code op is traced: the ``K = 1`` head is the strict
twin of the ``K = 4`` head, and its every other weight is byte-identical to the ``K = 4``
head's (the codes come from their own generator).

Contracts (copied from ``tul_spandec.py``)
------------------------------------------
* **Off is nothing.** ``tul.spandec_parallel: false`` builds no parameter, draws no RNG
  and adds no term.
* **RNG-neutral.** Every real draw comes from a PRIVATE generator, and the construction
  forks the global CPU (and live CUDA) RNG and restores it: ``nn.Linear`` kaiming-draws on
  the global stream before its weight is overwritten (the ``TULSlotRegister`` lesson). A
  model with the head has base weights byte-identical to its ruler's and the same global
  RNG state after construction.
* **Never quantised, never pruned.** Every ``nn.Linear`` carries ``_ternary_exclude`` and
  none is a ``MortarLinear`` / ``CMSBlockLinear``.
* **The tied head is detached at both ends.** There is no input embedding read at all,
  and the OUTPUT head is ``embed.lm_weight().detach()`` always (this head never trains the
  table the coda speaks through, whatever ``tul.mux_detach_head`` says).
* **Never materialises ``[B, S, J, V]``.** The per-token log-prob of the label comes from
  :func:`morph.model.fused_ce.fused_linear_label_logprob` over the VALID rows only; pad
  slots and past-the-end positions never reach the vocabulary GEMM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .attention import RMSNorm
from .fused_ce import fused_linear_label_logprob
from .tul_layout import SlotLayout
from .tul_spandec import _init_linear, _SpanDecBlock, span_slots

__all__ = ["ParallelSpanHead", "mixture_span_nll", "mixture_token_nll", "code_usage_stats"]

# Private init streams, distinct from `tul_spandec`'s (0x5DEC*): the ruler's decoder and
# this head must not start from the same weights, and the CODE table has its own stream so
# the K = 1 and K = 4 heads share every other tensor bit for bit.
_SEED_BLOCKS = 0x9A5D
_SEED_PROJ = 0x9A5D1
_SEED_CODES = 0x9A5D2


def mixture_span_nll(logp_slot: Tensor, n_tokens: Tensor | float) -> Tensor:
    """``-sum_m [logsumexp_r logp_slot[r, m] - log R] / n_tokens``.

    ``logp_slot [R, M]``: the log-likelihood of supervised slot ``m``'s whole span under
    reader ``r`` (a code at Stage 0, a rollout in Phase B). ``n_tokens``: the number of
    valid span tokens those ``M`` slots hold. The exact mixture likelihood over ``R``
    enumerated readers, per span token; ``R = 1`` is the plain per-token CE.
    """
    if logp_slot.dim() != 2:
        raise ValueError(f"logp_slot must be [R, M], got {tuple(logp_slot.shape)}")
    R = int(logp_slot.shape[0])
    lse = torch.logsumexp(logp_slot, dim=0) - math.log(R)          # [M]
    return -lse.sum() / n_tokens


def mixture_token_nll(lp: Tensor, val_s: Tensor) -> Tensor:
    """``[Nv]`` the mixture NLL split over a span's tokens by the chain rule.

    ``lp [R, Nv]`` / ``val_s [M, J]`` as :meth:`ParallelSpanHead.token_logp` returns them.
    Token ``j`` of a slot gets ``-log sum_r w_r(<j) p_r(t_j)`` with ``w_r(<j)`` the
    posterior over readers after the span's first ``j`` tokens, i.e.
    ``-(logsumexp_r c_r(<=j) - logsumexp_r c_r(<j))`` with ``c`` the running per-reader
    log-likelihood. The terms telescope: a slot's tokens sum to exactly its
    :func:`mixture_span_nll` term, so a per-token mean of this over all tokens IS the
    mixture CE, and a per-offset mean of it is the sequential Bayes read at that offset.
    ``R = 1`` returns ``-lp[0]`` to rounding. Scorer use (position-in-span bins); not a loss."""
    R, nv = int(lp.shape[0]), int(lp.shape[1])
    M, J = int(val_s.shape[0]), int(val_s.shape[1])
    flat = torch.nonzero(val_s.reshape(-1), as_tuple=True)[0]
    dense = lp.new_zeros(R, M * J)
    dense[:, flat] = lp
    dense = dense.view(R, M, J).double()
    c_le = dense.cumsum(-1)
    c_lt = c_le - dense
    tok = -(torch.logsumexp(c_le, 0) - torch.logsumexp(c_lt, 0))    # [M, J]
    out = tok.reshape(-1)[flat]
    if out.shape[0] != nv:
        raise RuntimeError(f"mixture_token_nll: {out.shape[0]} tokens against {nv} log-probs")
    return out.to(lp.dtype)


@torch.no_grad()
def code_usage_stats(logp_slot: Tensor, n_tokens: Tensor | float, mix_ce: Tensor
                     ) -> dict[str, Tensor]:
    """Detached 0-dim tensors (no host sync) describing how the ``R`` readers are used.

    * ``par_resp_entropy`` — mean over slots of the entropy (nats) of the responsibilities
      ``softmax_r(log p_r)``. ``log R`` is uniform credit (the codes are copies); 0 is one
      code owning every span.
    * ``par_code_win_{r}`` — fraction of slots whose best reader is ``r``.
    * ``par_ce_code_best`` — the per-token CE of the best SINGLE reader of this head, read
      alone (``min_r -sum_m log p_r / n``).
    * ``par_width_gain`` — ``par_ce_code_best - mix_ce``: what the mixture buys over the
      head's own best code. It is NOT ``B0``: ``B0`` pairs against the separately trained
      ``K = 1`` twin, on held-out rows (``lab/divergence/lxtul_e_stage0_score.py``). It
      can be NEGATIVE, by at most ``M log R / n_tokens``: when the codes are copies the
      mixture still pays ``log R`` per slot for the choice it never uses.
    """
    R = int(logp_slot.shape[0])
    lp = logp_slot.float()
    resp = torch.softmax(lp, dim=0)                                 # [R, M]
    ent = -(resp * torch.log(resp.clamp_min(1e-30))).sum(0).mean()
    win = torch.bincount(lp.argmax(0), minlength=R).float() / max(int(lp.shape[1]), 1)
    ce_single = -lp.sum(1) / n_tokens                               # [R]
    best = ce_single.min()
    out = {"par_resp_entropy": ent, "par_ce_code_best": best,
           "par_width_gain": best - mix_ce.detach().float()}
    for r in range(R):
        out[f"par_code_win_{r}"] = win[r]
    return out


class ParallelSpanHead(nn.Module):
    """``z [B, S, C]`` -> per-slot span log-likelihood under ``K`` enumerated codes.

    Built by :class:`~morph.model.transformer.MORPHTransformer` as ``tul_spandec_par`` when
    ``tul.spandec_parallel`` is on. The model reads it through
    ``MORPHTransformer._tul_spandec_par_loss``; an offline scorer reads it through
    :meth:`slot_logp` on the same exit state. See the module docstring for what it is and
    why it exists.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 max_tokens: int, n_codes: int = 1, target_offset: int = 1,
                 code_init: float = 0.1):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"parallel span head n_layers must be >= 1, got {n_layers}")
        if max_tokens < 1:
            raise ValueError(f"parallel span head max_tokens must be >= 1, got {max_tokens}")
        if n_codes < 1:
            raise ValueError(f"tul.spandec_parallel_k must be >= 1, got {n_codes}")
        if target_offset < 1:
            raise ValueError(f"target_offset must be >= 1 (1 = the next span), got "
                             f"{target_offset}")
        if not code_init > 0.0:
            raise ValueError(f"code_init must be > 0, got {code_init}")
        self.max_tokens = int(max_tokens)
        self.n_codes = int(n_codes)
        self.target_offset = int(target_offset)
        cuda_devs = ([torch.cuda.current_device()] if torch.cuda.is_available()
                     and torch.cuda.is_initialized() else [])
        # RNG-NEUTRAL: `nn.Linear` draws on the global stream before `_init_linear`
        # overwrites it. Fork and restore, so nothing built after this head, and no
        # forward-time draw (the Poisson depth, dropout), moves.
        with torch.random.fork_rng(devices=cuda_devs):
            gp = torch.Generator(device="cpu").manual_seed(_SEED_PROJ)
            self.z_in = nn.Linear(d_model, d_model, bias=False)
            _init_linear(self.z_in, gp)
            self.z_in._ternary_exclude = True
            # Learned position of the query inside the span, ZERO-init (the SpanDecoder
            # precedent): deterministic, no draw. The per-position targets differ, so
            # the first gradient step already separates the J rows.
            self.pos = nn.Parameter(torch.zeros(self.max_tokens, d_model))
            gb = torch.Generator(device="cpu").manual_seed(_SEED_BLOCKS)
            self.blocks = nn.ModuleList(
                [_SpanDecBlock(d_model, n_heads, d_ff, gb) for _ in range(n_layers)])
            self.out_norm = RMSNorm(d_model)
            if self.n_codes > 1:
                gc = torch.Generator(device="cpu").manual_seed(_SEED_CODES)
                self.codes = nn.Parameter(torch.empty(self.n_codes, d_model).normal_(
                    mean=0.0, std=float(code_init), generator=gc))
            else:
                # The K = 1 twin: no code table, and `code_offsets` refuses to run.
                self.register_parameter("codes", None)

    # ── the target: exactly the ruler decoder's ─────────────────────────────────────

    def targets(self, input_ids: Tensor, layout: SlotLayout) -> tuple[Tensor, Tensor]:
        """``(ids, valid) [B, S, J]``: span ``s + target_offset``'s tokens per slot.

        :func:`~morph.model.tul_spandec.span_slots` at the ruler decoder's shift and J cap,
        the one home of the target mapping (validity: the span is complete and the graded
        slot exists; pad slots, the dump bin and the trailing unterminated text are all
        invalid)."""
        return span_slots(input_ids, layout, self.max_tokens, shift=self.target_offset)

    # ── the codes ───────────────────────────────────────────────────────────────────

    def code_offsets(self) -> Tensor:
        """``[K, C]`` the code table made mean-free across ``k``. Refused at ``K = 1``."""
        if self.codes is None:
            raise RuntimeError("the K = 1 parallel head has no code table")
        return self.codes - self.codes.mean(dim=0, keepdim=True)

    def with_codes(self, z: Tensor) -> Tensor:
        """``z [N, C]`` -> ``[K, N, C]``: ``z + rms(z).detach() * u_k`` per code, or
        ``z.unsqueeze(0)`` at ``K = 1`` (no code op at all)."""
        if self.codes is None:
            return z.unsqueeze(0)
        rms = z.float().pow(2).mean(dim=-1, keepdim=True).sqrt().detach()     # [N, 1]
        off = self.code_offsets()                                              # [K, C]
        return (z.float().unsqueeze(0) + rms.unsqueeze(0) * off.unsqueeze(1)).to(z.dtype)

    # ── the reader ──────────────────────────────────────────────────────────────────

    def states(self, zr: Tensor, J: int | None = None) -> Tensor:
        """``zr [R, N, C]`` -> ``[R, N, J, C]`` pre-head states. INPUT-FREE: the only inputs
        are ``zr`` and the position table."""
        J = self.max_tokens if J is None else int(J)
        if J > self.max_tokens:
            raise ValueError(f"asked for J={J} positions; the head was built for "
                             f"{self.max_tokens}")
        R, N, C = zr.shape
        x = self.z_in(zr).unsqueeze(2) + self.pos[:J].to(zr.dtype).view(1, 1, J, C)
        x = x.reshape(R * N, J, C)
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x).reshape(R, N, J, C)

    def token_logp(self, zr: Tensor, ids: Tensor, valid: Tensor, w: Tensor,
                   chunk_size: int = 1024, mask_token_id: int = -1
                   ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-token log p(label) for readers already inside the state.

        ``zr [R, B, S, C]`` is ``R`` exit states per slot with any choice ALREADY in them
        (Phase B's per-rollout z; at Stage 0 :meth:`slot_logp` builds it from the codes).
        ``ids``/``valid [B, S, J]`` from :meth:`targets`; ``w [V, C]`` the tied table,
        detached by the caller.

        Returns ``(lp [R, Nv] fp32, sup [B, S] bool, val_s [M, J] bool)``: the log-prob of
        each of the ``Nv`` valid target tokens under each reader, in (supervised slot
        row-major, offset) order; which slots were supervised (at least one valid token);
        and the valid mask of the ``M`` supervised slots. Pad slots never enter the
        blocks and invalid positions never enter the vocabulary GEMM.
        """
        if zr.dim() != 4:
            raise ValueError(f"token_logp wants zr [R, B, S, C], got {tuple(zr.shape)}")
        R = int(zr.shape[0])
        J = int(ids.shape[-1])
        sup = valid.any(dim=-1)                                            # [B, S]
        zs = zr[:, sup]                                                    # [R, M, C]
        ids_s, val_s = ids[sup], valid[sup]                                # [M, J]
        st = self.states(zs, J)                                            # [R, M, J, C]
        rows = st[:, val_s]                                                # [R, Nv, C]
        nv = int(rows.shape[1])
        lab = ids_s[val_s].repeat(R)                                       # [R * Nv]
        lp = fused_linear_label_logprob(rows.reshape(R * nv, -1), w, lab,
                                        chunk_size=chunk_size,
                                        mask_token_id=mask_token_id).view(R, nv)
        return lp, sup, val_s

    def rollout_logp(self, zr: Tensor, ids: Tensor, valid: Tensor, w: Tensor,
                     chunk_size: int = 1024, mask_token_id: int = -1
                     ) -> tuple[Tensor, Tensor, Tensor]:
        """THE SHARED SCORING PATH: per-slot span log-likelihoods for ``R`` readers already
        inside the state (see :meth:`token_logp` for the arguments).

        Returns ``(logp_slot [R, M] fp32, n_tokens 0-dim, sup [B, S] bool)``: the span
        log-likelihood of each of the ``M`` supervised slots under each reader, the
        valid-token count and which slots were supervised. :func:`mixture_span_nll` turns
        ``logp_slot`` into the loss.
        """
        lp, sup, val_s = self.token_logp(zr, ids, valid, w, chunk_size, mask_token_id)
        R = int(lp.shape[0])
        M, J = int(val_s.shape[0]), int(val_s.shape[1])
        # Per-slot sums WITHOUT atomics: every (slot, offset) cell is written once
        # (`index_put` at unique indices), then a dense sum over J. An `index_add_` would
        # reduce through CUDA atomics and make the loss nondeterministic.
        flat = torch.nonzero(val_s.reshape(-1), as_tuple=True)[0]          # [Nv]
        dense = lp.new_zeros(R, M * J)
        dense = dense.index_put((torch.arange(R, device=lp.device).view(R, 1),
                                 flat.view(1, -1)), lp)
        logp_slot = dense.view(R, M, J).sum(-1)                            # [R, M]
        n_tok = val_s.sum().to(torch.float32)
        return logp_slot, n_tok, sup

    def slot_logp(self, z: Tensor, ids: Tensor, valid: Tensor, w: Tensor,
                  chunk_size: int = 1024, mask_token_id: int = -1
                  ) -> tuple[Tensor, Tensor, Tensor]:
        """Stage 0: ``z [B, S, C]`` (one exit state per slot) through the ``K`` codes.

        ``K = 1`` reads ``z`` itself; ``K > 1`` reads ``z + rms(z) u_k`` for every ``k``.
        Same return as :meth:`rollout_logp`."""
        B, S, C = z.shape
        zr = self.with_codes(z.reshape(B * S, C)).view(-1, B, S, C)
        return self.rollout_logp(zr, ids, valid, w, chunk_size, mask_token_id)
