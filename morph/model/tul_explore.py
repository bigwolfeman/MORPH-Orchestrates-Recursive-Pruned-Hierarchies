"""LX efficient-exploration knobs (2026-09-29): five default-OFF research levers on top
of the enumerated loop code (``tul.code_enum_k``). Scaffolding for LATER experiments —
nothing here is queued or trained by this change. Every knob is bit-identical to the
tree before it at its default value (see each knob's own docstring for the exact
default and the off-path proof).

Pure tensor / small-module logic lives HERE, not in ``morph/model/transformer.py``,
specifically so this file can be developed and reviewed independently of the WTA
winner-pick work three other agents are doing inside ``_tul_fan_all`` in that file at
the same time (2026-09-29). ``transformer.py`` gets ONLY the minimal additive hook
needed to call into this module from ``_forward_tul``'s existing
``if self._code_enum_k:`` branch — see that hook's own comment for the exact lines.

Design note carried over from the task brief, repeated here because it governs every
function below: an item derived from the TRUE next span may only SCORE or CHOOSE among
existing hypotheses; it must never be written into what the coda or the loop reads.
Every function that touches a "target" latent below takes it pre-detached from the
caller and never returns a value meant to be fed back into the loop's own forward.

── Knob-by-knob map ─────────────────────────────────────────────────────────────────
  tul.hyp_score_head                         -- KNOB 3, ``TULHypScoreHead`` + read gaps
  tul.latent_set_loss / tul.latent_set_weight -- KNOB 4, ``latent_energy_score`` /
                                                 ``latent_infonce``
  tul.enum_decode_k                          -- KNOB 5, ``swor_uniform`` /
                                                 ``gather_rows``
  tul.hyp_merge / tul.hyp_merge_weight       -- KNOB 6, see the warning block below
  (KNOB 2, ``tul.coda_fuse_layer``, late fusion in the coda) is NOT in this file: it was
  investigated and NOT implemented — see the "Knob 2" section of
  ``.agents/notes/proposed/architecture/2026-09-29-lx-efficient-exploration-knobs.md``
  for why (the token path's rollout-invariance under a no-cell-reach coda mask is sound,
  but delivering the claimed compute saving needs either gathering the coda's slot-cell
  positions into their own compact batch — unverified against the coda's GLA/retention
  branch's cross-position state, which this session did not have time to audit safely
  — or running the shared pass on the full sequence shape, which saves nothing on the
  cell side and would ship a knob whose claimed benefit does not exist).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["TULHypScoreHead", "gather_rows", "hyp_merge_mean", "hyp_merge_probe_stats",
           "latent_energy_score", "latent_infonce", "score_head_kl_loss",
           "score_head_read_gaps", "score_head_topk_mask", "swor_uniform"]


# ══════════════════════════════════════════════════════════════════════════════════
# KNOB 3 — SCORE HEAD (rMCL arXiv 2311.01052, LatentRM arXiv 2510.07745,
# LTO arXiv 2509.26314): a tiny head that predicts, from a rollout's OWN hypothesis
# latent alone, the mixture posterior the LX loss already computes over the K
# rollouts — so at eval a K-way Bayes read can be replaced by a 1- or 2-rollout read
# picked by the head instead of by running every rollout's coda pass.
# ══════════════════════════════════════════════════════════════════════════════════


class TULHypScoreHead(nn.Module):
    """``tul.hyp_score_head``: one ``d_model -> 1`` linear scorer, no bias.

    INPUT MUST BE DETACHED BY THE CALLER (``score_head_kl_loss``'s docstring repeats
    this): the head reads each rollout's slot-loop exit latent ``z = readout(h_slots)``
    and is trained to predict the (also detached) mixture posterior direction the LX
    loss's ``_enum_mix_losses`` already computes (``s_out["S"]``), via
    :func:`score_head_kl_loss`. Because both the input and the target are detached,
    the head's own gradient never reaches the loop, the codes, the front or the tied
    table — "training cost ~0" in the task brief means exactly this: one linear layer,
    no second coda pass, no gradient into anything expensive.

    RNG-neutral construction (the ``TULRowContrast`` / ``TULSlotRegister`` convention
    this tree already uses for every small auxiliary head): draws from a PRIVATE
    generator, snapshots and restores the global RNG state, so building this head
    moves no other parameter's init draw.
    """

    def __init__(self, d_model: int, seed: int = 0x5C08) -> None:
        super().__init__()
        _rng0 = torch.random.get_rng_state()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.w = nn.Linear(int(d_model), 1, bias=False)
        with torch.no_grad():
            self.w.weight.copy_(torch.empty(self.w.weight.shape, device="cpu").normal_(
                mean=0.0, std=0.02, generator=g))
        torch.random.set_rng_state(_rng0)

    def forward(self, z: Tensor) -> Tensor:
        """``[..., C] -> [...]`` one pre-softmax score per hypothesis."""
        return self.w(z).squeeze(-1)


def score_head_kl_loss(scores: Tensor, s_target: Tensor, valid: Tensor) -> Tensor:
    """KL(target posterior ‖ head's own softmax), reduced over the valid items.

    ``scores`` ``[R, N]`` -- :class:`TULHypScoreHead` applied to a DETACHED hypothesis
    latent (``R`` rollouts, ``N`` scored (row, slot) items flattened). ``s_target``
    ``[R, N]`` -- the LX mixture's own DETACHED per-rollout direction
    (``_enum_mix_losses``'s ``s_out["S"]``, restricted to the same ``N`` scored slots);
    the target posterior is ``softmax_r(s_target)``, exactly the distribution
    ``enum_code_win{k}`` in ``_enum_mix_losses`` already reads with ``.argmax``. ``valid``
    ``[N]`` bool -- which items actually have a target (a scored, valid slot).

    Both ``scores`` and ``s_target`` must already be detached from anything expensive by
    the caller; this function does not detach on its own so a caller that forgets is not
    silently corrected -- see the module docstring's foot-gun note. The reduction is the
    mean KL over the valid columns, forward KL (target first) so a target with a sharp
    posterior (one rollout clearly wins a span) costs the head more for missing it than
    a soft target does, matching a maximum-likelihood fit of the head to the posterior.
    """
    if scores.shape != s_target.shape:
        raise ValueError(f"score_head_kl_loss: scores {tuple(scores.shape)} != "
                         f"s_target {tuple(s_target.shape)}")
    p = torch.softmax(s_target, dim=0)                              # [R, N] target posterior
    logq = torch.log_softmax(scores, dim=0)                         # [R, N] head's posterior
    kl = (p * (torch.log(p.clamp_min(1e-30)) - logq)).sum(dim=0)    # [N]
    vf = valid.to(kl.dtype)
    n = vf.sum().clamp_min(1.0)
    return (kl * vf).sum() / n


def score_head_topk_mask(scores: Tensor, k: int) -> Tensor:
    """``[R, N]`` scores -> ``[R, N]`` bool, True at each item's top-``k`` rollouts.

    Used by :func:`score_head_read_gaps` to price a read restricted to the head's own
    top-``k`` rollouts per item. ``k`` must be ``>= 1`` and ``<= R``; ``k == R`` is the
    identity mask (every rollout kept, so the read is unchanged). Ties broken by
    a stable sort's order (lowest index among equal scores), matching the
    project's existing "ties: lowest k" convention (``code_enum_credit="hard"``).
    """
    R, N = scores.shape
    if not (1 <= k <= R):
        raise ValueError(f"score_head_topk_mask: k={k} must be in [1, {R}]")
    # A STABLE descending sort, not `torch.topk`: topk does not promise which of two
    # equal scores wins (measured: constant scores did not pick rollout 0).
    idx = torch.sort(scores, dim=0, descending=True, stable=True).indices[:k]   # [k, N]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(0, idx, True)
    return mask



@torch.no_grad()
def score_head_read_gaps(scores: Tensor, s_target: Tensor, scored: Tensor,
                         n_w: Tensor) -> dict[str, Tensor]:
    """What a CHEAP read picked by the head would cost, against the exact K-way read.

    The question knob 3 exists to answer: can a head that reads only the latent replace
    the K coda reads the LX mixture needs? This prices that swap without building it
    (an earlier draft accepted a ``hyp_score_read: top1|top2`` config value that nothing
    read; this instrument replaced it, 2026-09-29).

    ``scores`` ``[R, N]`` the head's scores, ``s_target`` ``[R, N]`` the exact per-span
    log-likelihood ``S_k`` of each rollout (``_enum_mix_losses``'s ``s_out["S"]``, SUMMED
    weighted token log-probs of the span, re-indexed to the same ``N`` slot items),
    ``scored`` ``[N]`` bool (the span has at least one scored token), ``n_w`` the loss's
    own token-weight total. Every return value is ``CE(read) - CE(exact mixture)`` in
    nats per token over ALL of the loss's tokens, so it adds straight onto
    ``enum_ce_mix``:

    * ``read_top1_gap`` / ``read_top2_gap``: the mixture restricted to the head's top 1 /
      top 2 rollouts per span. A deployable read (the head never sees the span).
    * ``read_rand1_gap``: a uniformly random single rollout, in expectation. The chance
      floor; ``>= 0`` by Jensen.
    * ``read_best1_gap``: the hindsight-best single rollout (reads the answer, NOT
      deployable). The ceiling; ``<= 0``.
    * ``score_agree``: fraction of scored spans where the head's argmax is the posterior's.

    A head worth keeping puts ``read_top1_gap`` well below ``read_rand1_gap``. At or near
    it, the head is guessing (the latent-WTA probe's chance-level agreement,
    2026-09-29, is the precedent).
    """
    if scores.shape != s_target.shape:
        raise ValueError(f"score_head_read_gaps: scores {tuple(scores.shape)} != "
                         f"s_target {tuple(s_target.shape)}")
    R = scores.shape[0]
    S = s_target.float()
    sf = scored.to(S.dtype)
    n = n_w.to(S.dtype).clamp_min(1e-6)
    lme_all = torch.logsumexp(S, dim=0) - math.log(R)                        # [N]

    def _gap(read: Tensor) -> Tensor:
        return ((lme_all - read) * sf).sum() / n

    out: dict[str, Tensor] = {}
    for k in (1, 2):
        if k > R:
            continue
        keep = score_head_topk_mask(scores.float(), k)
        read_k = torch.logsumexp(S.masked_fill(~keep, float("-inf")), dim=0) - math.log(k)
        out[f"read_top{k}_gap"] = _gap(read_k)
    out["read_rand1_gap"] = _gap(S.mean(dim=0))
    out["read_best1_gap"] = _gap(S.max(dim=0).values)
    agree = (scores.argmax(dim=0) == S.argmax(dim=0)).to(S.dtype)
    out["score_agree"] = (agree * sf).sum() / sf.sum().clamp_min(1.0)
    return out

# ══════════════════════════════════════════════════════════════════════════════════
# KNOB 4 — SET LOSS in latent space (CALM arXiv 2510.27688, CPC arXiv 1807.03748).
#
# TARGET AND WHY. The target is the detached mean-pool of the NEXT span's own PRELUDE
# token states (``next_span_pool(readout(x.detach()), layout)`` -- the exact
# construction ``TULRowContrast`` already uses in ``morph/model/tul.py``), NOT the
# loop's own exit state of the next slot. That second choice is what
# ``tul.nextlat_weight`` already tried (``morph/model/tul_nextlat.py``) and it
# COLLAPSED (project memory: "Latent prediction on the slot exit collapses" -- the K1-K6
# depth-earning signal shrank 0.0126 -> 0.0010 once the target moved onto the loop's own
# future state). A prelude-pooled target is read-only content the loop did not write,
# so there is nothing for the loop to game by making the target trivial to hit.
#
# COLLAPSE INSTRUMENT (the JEPA Paradox, arXiv 2607.23531, and MORPH's own NextLat
# finding both apply here): an MSE-style target invites the hypotheses to collapse onto
# one point (the conditional mean) rather than spreading over the true predictive
# spread. `latent_energy_score` and `latent_infonce` are both used specifically because
# neither is minimized by collapse to a point (see each function's own docstring for the
# proof); `latent_hyp_spread` below is the instrument a caller should log alongside the
# loss to catch a collapse this design did not intend.
# ══════════════════════════════════════════════════════════════════════════════════


def latent_energy_score(hyp: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """The (strictly proper) energy score of the ``R`` hypotheses against ``target``.

    ``hyp`` ``[R, N, C]``, ``target`` ``[N, C]`` (DETACHED by the caller -- see the
    module docstring), ``valid`` ``[N]`` bool. Per item ``i``:

        e_i = mean_r ||hyp[r,i] - target[i]||_2  -  0.5 * mean_{r,r'} ||hyp[r,i] - hyp[r',i]||_2

    Gneiting & Raftery's energy score (2007): a STRICTLY PROPER scoring rule for a
    forecast represented as a finite sample (here: the ``R`` rollout hypotheses) against
    an observation, minimized IN EXPECTATION only when the ensemble's law matches the
    true predictive distribution of the target. This is what rules out collapse: if
    every ``hyp[r,i]`` collapses to the SAME point (mean_{r,r'} term -> 0), the score
    reduces to the plain mean distance to the target, which is minimized by matching the
    CONDITIONAL MEAN, not the marginal distribution's spread -- and the strictly-proper
    property means that (for a genuinely multi-modal or uncertain target) a collapsed,
    zero-spread ensemble scores STRICTLY WORSE than a spread ensemble whose mean is the
    same, so gradient descent on this loss has a standing incentive against collapse
    that a bare MSE-to-mean loss does not carry. Reduced as the mean over valid items
    (``valid.sum()`` clamped to >= 1 so an all-invalid batch returns exactly 0 with a
    live graph, the project's own zero-anchor convention).
    """
    if hyp.dim() != 3 or target.dim() != 2 or valid.dim() != 1:
        raise ValueError("latent_energy_score: expected hyp [R,N,C], target [N,C], "
                         f"valid [N]; got {tuple(hyp.shape)}, {tuple(target.shape)}, "
                         f"{tuple(valid.shape)}")
    R, N, C = hyp.shape
    hyp = hyp.float()
    target = target.float()
    d_target = (hyp - target.unsqueeze(0)).norm(dim=-1)               # [R, N]
    mean_d_target = d_target.mean(dim=0)                              # [N]
    diff = hyp.unsqueeze(1) - hyp.unsqueeze(0)                        # [R, R, N, C]
    d_pair = diff.norm(dim=-1)                                        # [R, R, N]
    mean_d_pair = d_pair.mean(dim=(0, 1))                             # [N], includes r==r' (=0)
    e = mean_d_target - 0.5 * mean_d_pair
    vf = valid.to(e.dtype)
    n = vf.sum().clamp_min(1.0)
    return (e * vf).sum() / n


def latent_infonce(hyp: Tensor, target: Tensor, valid: Tensor, tau: float) -> Tensor:
    """InfoNCE of each rollout's hypothesis against the true target, IN-BATCH negatives.

    ``hyp`` ``[R, N, C]``, ``target`` ``[N, C]`` (DETACHED by the caller), ``valid``
    ``[N]`` bool, ``tau > 0``. Every ROLLOUT independently plays the retrieval game (the
    ``TULRowContrast`` template, extended from one anchor per item to ``R``): rollout
    ``r``'s hypothesis at item ``i`` is the query, every VALID item's target in the
    WHOLE FLATTENED BATCH is a key (not just the same row's other slots, as
    ``TULRowContrast`` restricts to -- "in-batch negatives" per the task brief), and the
    loss is the mean cross-entropy of picking key ``i`` back out. Averaged over the ``R``
    rollouts and the valid items.

    NOT MSE, so not minimized by every hypothesis collapsing onto the target's mean: a
    collapsed rollout that always predicts the SAME point for every item cannot tell
    items apart (every query gives the same similarity row up to the query's own
    identity, which does not vary across items either), so its retrieval accuracy falls
    toward chance (``1/n_valid``) and the cross-entropy rises toward ``log n_valid`` --
    exactly the chance-level reference :class:`~morph.model.tul.TULRowContrast` already
    documents for its own single-anchor version of this game.
    """
    if tau <= 0.0:
        raise ValueError(f"latent_infonce needs tau > 0, got {tau}")
    R, N, C = hyp.shape
    hyp_n = F.normalize(hyp.float(), dim=-1)                          # [R, N, C]
    tgt_n = F.normalize(target.float(), dim=-1)                       # [N, C]
    sim = torch.einsum("rnc,mc->rnm", hyp_n, tgt_n) / tau             # [R, N, N]
    neg = -1.0e30
    sim = sim.masked_fill(~valid.view(1, 1, N), neg)
    tgt_idx = torch.arange(N, device=hyp.device).view(1, N).expand(R, N)
    ce = F.cross_entropy(sim.reshape(R * N, N), tgt_idx.reshape(R * N), reduction="none")
    ce = ce.view(R, N)
    vf = valid.to(ce.dtype)
    n = vf.sum().clamp_min(1.0)
    per_rollout = (ce * vf.unsqueeze(0)).sum(dim=1) / n                # [R]
    return per_rollout.mean()


def latent_hyp_spread(hyp: Tensor, valid: Tensor) -> Tensor:
    """Diagnostic (no gradient, ``.detach()``'s the input): mean pairwise distance among
    the ``R`` hypotheses of each valid item, averaged over items. Log this beside
    ``latent_energy_score`` / ``latent_infonce``'s own value; a spread collapsing toward
    0 over training is the collapse the energy score / InfoNCE choice is meant to resist,
    caught here rather than inferred from the loss curve alone."""
    hyp = hyp.detach().float()
    diff = hyp.unsqueeze(1) - hyp.unsqueeze(0)                        # [R, R, N, C]
    d = diff.norm(dim=-1).mean(dim=(0, 1))                            # [N]
    vf = valid.to(d.dtype)
    n = vf.sum().clamp_min(1.0)
    return (d * vf).sum() / n


# ══════════════════════════════════════════════════════════════════════════════════
# KNOB 5 — SAMPLING WITHOUT REPLACEMENT (Kool et al. 2020, arXiv 2002.06043; LTC
# arXiv 2608.01593): run the expensive coda pass on only ``k`` of the ``K`` rollouts,
# chosen without replacement, and correct the resulting ``k``-way mixture estimate.
#
# THE ESTIMATOR SHIPPED HERE, STATED EXACTLY (read before trusting a number from it).
# For a UNIFORM (score-blind) draw of size k without replacement from K items, simple
# random sampling theory gives an EXACT result with no correction term needed: the
# sample mean of `exp(S_i)` over the k drawn items is an UNBIASED estimator of the
# population mean over all K items (this is the textbook unbiasedness of the SRSWOR
# sample mean, not an approximation). Since `L = -log(mean_i exp(S_i))`, running the
# ORDINARY k-way mixture computation on the k sampled rollouts (``_enum_mix_losses``
# with ``K=k``) already computes exactly the needed sample-mean-domain quantity, with NO
# extra reweighting term. The forward therefore calls ``_enum_mix_losses`` on the k
# gathered rollouts and nothing else; this header is where that reasoning lives.
#
# JENSEN CAVEAT AND THE OBJECTIVE FOOT GUN (say both every time a number from this path
# is reported). ``E[log X] <= log E[X]``: the log of the unbiased sample mean is biased
# LOW as an estimate of the log of the true mean, so the LOSS (its negative) reads HIGH;
# the bias shrinks as k grows and vanishes at k = K. This is the IWAE-k bound, and like
# IWAE it changes what is optimised, not only how noisily: the credit for rollouts that
# DIFFER falls with k, and at k = 1 the loss is the mean single-rollout CE, which pays
# nothing for a rollout that explains a span the others miss.
#
# A SCORE-WEIGHTED draw (the knob-3 head choosing the k rollouts) is NOT built and
# ``TULConfig`` refuses it. The k-way mean over a head-biased sample has no importance
# correction, and the correct one needs the Gumbel-top-k inclusion probabilities (Kool
# et al. 2020's Rao-Blackwellized estimator), which were not derived and checked here.
# ══════════════════════════════════════════════════════════════════════════════════


def swor_uniform(K: int, k: int, generator: torch.Generator | None = None) -> Tensor:
    """``k`` of ``K`` indices, drawn uniformly WITHOUT replacement. ``[k]`` int64,
    ascending order (so downstream row-block gathers stay in rollout order, which
    matters only for readability of any per-rollout logging, never for correctness).
    Exact inclusion probability ``k / K`` for every index -- see the module-level
    estimator note."""
    if not (1 <= k <= K):
        raise ValueError(f"swor_uniform: k={k} must be in [1, K={K}]")
    perm = torch.randperm(K, generator=generator)
    return perm[:k].sort().values


def gather_rows(x, keep_blocks: Tensor, block_size: int):
    """Gather ``k`` of ``K`` ROW-BLOCKS of size ``block_size`` from the leading axis.

    ``x``: a ``Tensor`` whose ``dim(0) == K * block_size`` (gathered), a ``dict`` (every
    value recursively gathered the same way, values whose ``dim(0) != K * block_size``
    left UNCHANGED -- covers a per-forward ``attn_kwargs`` dict that may mix
    rollout-tiled tensors with scalars or already-base-sized tensors), or ``None``
    (returned as ``None``). ``keep_blocks`` ``[k]`` int64, the rollout indices to keep
    (e.g. :func:`swor_uniform`'s output).

    SAFETY NOTE this function exists to make explicit: selecting whole row-blocks never
    touches what happens WITHIN a row (attention, RoPE/CoPE, any recurrent state a coda
    layer carries) -- a kept row's own sequence is byte-identical to the one it was
    before the gather, only OTHER rows are removed. This is the reason KNOB 5 was judged
    safe to implement where KNOB 2's within-row cell/token split was not (see this
    module's header and the Agent Note): row-block selection is architecturally inert to
    every position-dependent mechanism in the coda, by construction.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: gather_rows(v, keep_blocks, block_size) for k, v in x.items()}
    if not torch.is_tensor(x):
        return x
    K_times_block = x.shape[0]
    if K_times_block % block_size != 0:
        return x
    K = K_times_block // block_size
    if K <= 1:
        return x
    idx = (keep_blocks.to(x.device).view(-1, 1) * block_size
          + torch.arange(block_size, device=x.device).view(1, -1)).reshape(-1)
    return x.index_select(0, idx)


# ══════════════════════════════════════════════════════════════════════════════════
# KNOB 6 — SUPERPOSITION PROBE + NONLINEAR MERGE (Superposed Decoding arXiv 2405.18400,
# Learning to Plan Long-Term arXiv 2409.00070, ParScale arXiv 2505.10475).
#
# READ THIS BEFORE TURNING `hyp_merge` ON.
#
# MIXTURE NON-CLOSURE (Latent-GRPO arXiv 2604.27998): the K rollout hypotheses are K
# points in a latent space the coda was trained to read ONE OF, not an average of. There
# is no guarantee that any convex combination, or any other merge, of K valid
# hypotheses is ITSELF a state the coda (or anything downstream) can read correctly --
# the space these hypotheses live in is not closed under mixture the way, say, a
# probability simplex is. A merge that looks reasonable in isolation can hand the coda
# an off-manifold vector with no meaning at all.
#
# MORPH'S OWN FAILED PRECEDENTS: the soft (mean-mixed) LX fans (0.030 nats behind the WTA fan, 3 of 4 draws detonated,
# lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md) and the four-copies fan
# (arm a1 of docs/slot-cells-distinct-vs-blurred.md, which did not close the gap that
# distinct cells closed). A blurred mean of hypotheses has lost every
# time this tree measured it. `hyp_merge_mean` below IS that naive mean -- kept so
# `hyp_merge_probe_stats` can report, on THIS arm's own geometry, whether it is still as
# bad, rather than assume the answer transfers.
#
# WHY THIS IS "none" BY DEFAULT AND STAYS A PROBE FIRST. `hyp_merge="probe"` computes
# and LOGS how a merge would score against the target, with NO GRADIENT PATH to
# anything (`hyp_merge_probe_stats` detaches its inputs) -- a read-only diagnostic.
# `hyp_merge="learned"` is a REAL trainable merge (`TULHypMergeGate`) that a caller MAY
# fold into a loss via `tul.hyp_merge_weight > 0`, but nothing in this module decides
# that it SHOULD be folded in by default, and the config wiring keeps the weight at 0.0
# unless a future experiment explicitly turns it on.
# ══════════════════════════════════════════════════════════════════════════════════


def hyp_merge_mean(hyp: Tensor) -> Tensor:
    """``[R, N, C] -> [N, C]``, the naive mean over rollouts -- the blurred-mean
    mechanism the soft fans measured as a loss (this module's KNOB 6 header). Read that
    header before reading anything into a good score from this."""
    return hyp.mean(dim=0)


class TULHypMergeGate(nn.Module):
    """``tul.hyp_merge="learned"``: a small gated pool over the ``R`` hypotheses.

    A softmax attention pool with ONE learned query per channel group (``d_model ->
    d_model``, no bias, RNG-neutral construction like every other small head in this
    tree): ``w_r = softmax_r(hyp[r] . q)``, output ``sum_r w_r * hyp[r]``. Strictly more
    expressive than :func:`hyp_merge_mean` (a uniform-weight special case of this pool
    at ``q = 0``), and still a CONVEX combination of the ``R`` hypotheses -- it does NOT
    escape the mixture non-closure warning above; it only lets the weights be learned
    rather than fixed uniform. A genuinely nonlinear (non-convex) merge, e.g. an MLP over
    the concatenated hypotheses, was considered and left OUT of this v0: it would break
    permutation-INVARIANCE over the R rollouts (a real property the mixture posterior
    the LX loss already trains toward has, and a concatenation-based merge would not),
    and testing a merge that breaks it would need a rollout-order sensitivity test this
    session did not have room to write and verify carefully.
    """

    def __init__(self, d_model: int, seed: int = 0x5C09) -> None:
        super().__init__()
        _rng0 = torch.random.get_rng_state()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.q = nn.Linear(int(d_model), int(d_model), bias=False)
        with torch.no_grad():
            self.q.weight.copy_(torch.empty(self.q.weight.shape, device="cpu").normal_(
                mean=0.0, std=0.02, generator=g))
        torch.random.set_rng_state(_rng0)

    def forward(self, hyp: Tensor) -> Tensor:
        """``[R, N, C] -> [N, C]``."""
        qh = self.q(hyp)                                              # [R, N, C]
        score = (qh * hyp).sum(dim=-1)                                # [R, N]
        w = torch.softmax(score, dim=0)                               # [R, N]
        return (w.unsqueeze(-1) * hyp).sum(dim=0)


def hyp_merge_probe_stats(hyp: Tensor, target: Tensor, valid: Tensor,
                          merged: Tensor) -> dict:
    """Diagnostic ONLY -- every input is detached inside; nothing here carries a
    gradient anywhere, matching ``tul.hyp_merge="probe"``'s contract.

    Returns (all scalars, detached): ``merge_dist`` (mean distance merged->target over
    valid items), ``best_single_dist`` (mean over items of the BEST single hypothesis's
    distance to target -- the oracle an honest merge should be compared against, not
    the mean-of-hypotheses distance, which the mixture non-closure warning says is not
    the right reference either), ``mean_single_dist`` (the naive "just look at rollout
    0" baseline), ``merge_beats_best_single`` (fraction of valid items where the merge is
    CLOSER to target than the best single hypothesis -- if this is near 0, the merge is
    not adding anything the best rollout did not already have alone, which is exactly
    the blurred-mean findings restated on this arm's own geometry)."""
    hyp = hyp.detach().float()
    target = target.detach().float()
    merged = merged.detach().float()
    valid = valid.detach()
    d_all = (hyp - target.unsqueeze(0)).norm(dim=-1)                  # [R, N]
    best_single = d_all.min(dim=0).values                             # [N]
    mean_single = d_all.mean(dim=0)                                   # [N]
    d_merge = (merged - target).norm(dim=-1)                          # [N]
    vf = valid.to(torch.float32)
    n = vf.sum().clamp_min(1.0)
    beats = ((d_merge < best_single).to(torch.float32) * vf).sum() / n
    return {
        "merge_dist": float((d_merge * vf).sum() / n),
        "best_single_dist": float((best_single * vf).sum() / n),
        "mean_single_dist": float((mean_single * vf).sum() / n),
        "merge_beats_best_single": float(beats),
    }
