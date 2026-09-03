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
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .tul_layout import SlotLayout

__all__ = ["TULConfig", "TULGate", "TULGateConfig", "TULSlots", "bag_mean", "bound_seed",
           "build_bound_rotations", "mux_span_targets",
           "compact_index", "cw2_retain_mask", "gather_positions", "scatter_positions",
           "window_drop_mask"]


@dataclass
class TULGateConfig:
    """Construction-time settings of the span-length gate (docs/tul-gate-spec.md).

    Present ⇒ :class:`TULGate` is built and the layout must carry ``span_len`` /
    ``len_supervised``. Absent (``TULConfig.gate is None``) ⇒ nothing is built, no
    parameter exists, and the arm is arm A1 (spec §9 invariant 1).

    Args:
        k_max:        the regression DENOMINATOR: the head predicts ``span_len / k_max``
                      and decodes ``k = round(g · k_max)``. It is deliberately allowed to
                      exceed ``span_cap``, and the arms set it to ``1.25 × span_cap``.
                      **Why:** measured on OpenWebText at ``span_cap`` 32, **24.5 %** of
                      labels are a span of exactly 32, so with ``k_max = span_cap`` a
                      quarter of the training signal sits on the target ``g = 1.0`` —
                      an asymptote a sigmoid reaches only in the limit, with a gradient
                      that vanishes as it approaches. Headroom moves the largest target
                      to 0.8 (logit 1.39) and collapses the q10…q90 logit spread the
                      audit must cover from 10.90 to 3.33.
        k_decode_max: the largest ``k`` the model may ask for, ``= span_cap``. Without it
                      the head could pick a budget above ``span_cap``, index a budget row
                      no training example ever reaches, and silently condition the coda
                      on a zero vector.
        lam:          ``gate_lambda``, the weight of the length term in the total loss.
                      1.0 is the predecessor's ``lambda_g`` (``coconut/tul/config.py:81``,
                      the setting under which its gate reached p50 9 against gold p50 9).
                      0.0 ⇒ the term is not added and the arm is bit-identical to A1.
        budget_cond:  §5 — add ``budget_embed(span_len)`` to the slot state before the
                      coda. False ⇒ the head's output changes nothing downstream, which
                      is exactly the predecessor's configuration and why its length
                      decision was never a trade-off.
        huber_beta:   the Huber knee. 1.0 = the predecessor's ``delta`` default.
        train_zeros: the ``k = 0`` ("keep thinking") half of the encoding: supervise ``g``
                      toward 0 on every iteration before a slot's last. **Default False,
                      and the reason is arithmetic, not taste.** The Poisson depth is
                      independent of the input, so no head can know which iteration is the
                      last one; the Bayes-optimal output at iteration ``t`` is then the
                      HAZARD times the mean target, not the length. At ``mean_depth`` 6
                      that is ``0.29 × 0.45 = 0.13`` at ``t = 5`` — the iteration a
                      fixed-depth generation reads — i.e. ``k = 5`` against a true span of
                      19. Measured on the 5090 at step 40 and step 120: ``k = 5.00`` and
                      ``5.68`` against gold ``18.98`` / ``19.58``, matching the predicted
                      table row for row. With the zeros off, the length is regressed at
                      every iteration and the prediction is unbiased at any depth. The
                      stop decision belongs in the separate head of §12, not multiplexed
                      onto the same scalar.
        drives_depth: §7 — the gate chooses the loop depth AT GENERATION/EVAL (arm
                      ``TUL-halt``). Never affects training: §4 teacher-forces the depth,
                      so ``TUL-gate`` and ``TUL-halt`` are ONE training run scored twice.
    """

    k_max: int = 32
    k_decode_max: int = 0                # 0 → k_max
    train_zeros: bool = False            # see the docstring; the measurement says False
    lam: float = 0.0
    budget_cond: bool = True
    huber_beta: float = 1.0
    drives_depth: bool = False
    # Specified in §12 and NOT built. A silently-ignored key is worse than a missing one.
    scheduled_sampling: float = 0.0
    stop_head: bool = False
    ponder_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError(f"tul.gate_k_max must be ≥ 1, got {self.k_max}")
        if self.k_decode_max == 0:
            self.k_decode_max = self.k_max
        if not 1 <= self.k_decode_max <= self.k_max:
            raise ValueError(
                f"tul.gate_k_decode_max must be in [1, gate_k_max={self.k_max}], "
                f"got {self.k_decode_max}")
        if self.lam < 0.0:
            raise ValueError(f"tul.gate_lambda must be ≥ 0, got {self.lam}")
        if self.huber_beta <= 0.0:
            raise ValueError(f"tul.gate_huber_beta must be > 0, got {self.huber_beta}")
        for name, key in (("scheduled_sampling", "gate_scheduled_sampling"),
                          ("ponder_lambda", "gate_ponder_lambda")):
            if float(getattr(self, name)) != 0.0:
                raise NotImplementedError(
                    f"tul.{key}={getattr(self, name)} — specified in "
                    f"docs/tul-gate-spec.md §12 and NOT implemented. Set it to 0.0.")
        if self.stop_head:
            raise NotImplementedError(
                "tul.gate_stop_head — the split stop/length encoding of "
                "docs/tul-gate-spec.md §7 is specified and NOT implemented. Leave it false.")


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
    gate: "TULGateConfig | None" = None  # docs/tul-gate-spec.md; None = arm A1 (nothing built)
    # Per-slot-INDEX input embedding instead of one shared E_slot. 0 = off (one shared
    # vector, the shipped behaviour); >0 = that many rows, and the slot at index s gets row
    # s. Motivated by measurement, not taste: the 50 valid slot states of a row have an
    # effective rank of 1.7 to 4.8 in a 1024-dimensional space with a mean pairwise cosine
    # of +0.39 to +0.71, at EVERY checkpoint including the healthy ones. They are built from
    # one shared E_slot plus a span bag-mean, and a mean over many token embeddings
    # concentrates, so the slots are near-parallel by construction. See
    # lab/experiments/failures/2026-08-24-tul-takeover-cure.md.
    per_slot_embed: int = 0
    per_slot_embed_std: float = 0.0      # jitter added to each row at seating; 0 = rows equal
    coda_token_cut: int = 0              # arm CW (.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md) — drop
                                          # TOKEN positions with row index < C from the coda's
                                          # sequence; every slot stays regardless of its index.
                                          # 0 = off, bit-identical (no new tensors, no new ops).
    # ── §5 double-label weights (arm v1a makes them knobs) ────────────────────
    # The spec weights the twice-predicted first token 0.5/0.5. Arm v1a
    # (lab/experiments/planned/2026-08-25-mux-head-arm-v1a.md) retires the slot's
    # private one-token race by setting emit_weight=0.0 / plast_weight=1.0: the
    # emit position stays a METRIC (ce_emit) but carries no training gradient.
    # Defaults keep every existing arm bit-identical.
    emit_weight: float = 0.5             # training weight of the slot's emit position
    plast_weight: float = 0.5            # training weight of the t_last token position
    # ── MUX local head (arXiv 2607.18264; arm v1a) ────────────────────────────
    # mux_beta > 0 builds nothing (the head reuses _readout + the unembedding —
    # zero new parameters) but adds beta * KL(mux_geo(next span) || softmax(W z / tau))
    # to the loss, where z is the slot's post-core state read out through the
    # model's own LM-head path. beta = 0.0 → the branch is a construction-time
    # constant and the arm is bit-identical to A1.
    mux_beta: float = 0.0                # weight of the local loss (paper: 1.0)
    mux_rho: float = 0.9                 # geometric decay of the span weighting (Prop 5i)
    # WHICH span a slot is supervised toward (arm GL1b).
    #   "next" — slot i is supervised toward span i+1. The PLAN framing: the slot is a
    #            forecast, which is what every FM/TUL arm before GL1 assumed. Default,
    #            so every existing arm is unchanged.
    #   "own"  — slot i is supervised toward span i, the span it terminates. The GIST
    #            framing: under `tg_restrict` the slot is the ONLY route by which a
    #            later token can reach that span, so "be a lossless record of what you
    #            replaced" is the job the architecture actually assigns it. MUX's own
    #            latent replaces a CoT step it must encode, which is this role and not
    #            the forecasting one — the paper's Eq. 2 target is the span the latent
    #            STANDS FOR.
    mux_target: str = "next"
    mux_tau: float = 1.0                 # softmax temperature of the head (paper: 1.0)
    # ── Think-once panel knobs (branch tul/think-once, arms R7/R8;
    #    .agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md)
    # cond_layers: that many NON-SHARED MORPHBlocks run ONCE over the compact slot
    # sequence after the core loop; the mux loss, the gate, the ablations and
    # prefix_project all read the stack's output. 0 = nothing built, bit-identical.
    cond_layers: int = 0
    # detach_z: the coda reads the slot state with stop-gradient ("frozen z"), so the
    # loop and the conditioning stack learn from the mux local loss alone. False =
    # bit-identical (the token CE trains through z, the MUX paper's own setting).
    detach_z: bool = False
    # ── DB-shaped loop (arm L3, lab/experiments/planned/2026-08-29-tul-loop-ladder.md) ──
    # The core loop runs in the FORWARD but the carry is DETACHED between iterations
    # (retention state too), so no gradient ever crosses an iteration boundary — the
    # DiffusionBlocks training shape transplanted to the slot loop. Each supervised
    # iteration's state gets its own LOCAL mux loss (same target, weights summing to
    # mux_beta); the seed's injection stays live, so every local loss shapes the write
    # through exactly ONE core application, never an unrolled iterate.
    db_loop: bool = False
    # How many iterations get the local mux loss (evenly spaced, always including the
    # seed t=0 and the final state). Caps the [B,S,V] fp32 logit cost per step.
    db_mux_iters: int = 4
    # ── faithful DiffusionBlocks (arXiv 2506.14202 App. B "recurrent-depth
    # architectures", §3.3, App. C) — morph/model/iter_cond.py ─────────────────
    # `db_loop` above kept the T-iteration UNROLLED LOOP and only detached the carry
    # between iterations; it built NO σ/timestep conditioning, so every iteration got
    # the identical job and specialised at nothing (measured depth-inertness). This is
    # the paper's ACTUAL recipe: "iter" gives every core-layer application an AdaLN-Zero
    # signal for WHICH loop iteration it is (works inside today's T-iteration loop —
    # arms tul_l2cap_cond / tul_db_cond). "sigma" builds the SAME AdaLN machinery keyed
    # on an EDM noise level instead, which is what unlocks the one-pass training step
    # (`tul_step_mode="db1"`, morph/model/transformer.py::_tul_core_db1) and the
    # deterministic Euler-ladder eval (`_tul_core_db1_ladder`) — see CLAUDE.md for the
    # dispatch rule. "none" (default) builds nothing: zero new parameters, zero RNG
    # draws, forward untouched.
    core_stage_cond: str = "none"
    # GRT recurrence gate (morph/model/recur_gate.py, arXiv 2608.15062 Eqs. 4-5;
    # program note .agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md).
    # "grt" wraps every core-loop iteration in the elementwise convex blend
    # h <- g*h_prev + (1-g)*o with g keyed on STATE + PRELUDE only (the cond-zero
    # constraint: no iteration index may enter the training graph). "none" (default)
    # builds nothing: zero parameters, zero RNG draws, forward untouched.
    recur_gate: str = "none"
    recur_gate_bias: float = 4.0    # fc2 bias init: g ~ 0.98 at init (their App. A)
    recur_gate_noise: float = 0.1   # sigma_g, per-scalar logit noise, training only
    recur_gate_tau: float = 1.0     # gate temperature (their B.4: 1.0 is optimal)
    # Width of the σ/iteration embedding fed to each core layer's AdaLN gate. Same
    # role and same default as diffusion_blocks.DBConfig.cond_dim; kept as its own key
    # because the TUL core's d_model can differ from the whole-model DB arm's.
    db1_cond_dim: int = 256
    # EDM / DiffusionBlocks σ schedule (App. C, App. E defaults — the paper's own
    # numbers, NOT re-derived): log σ ~ N(p_mean, p_std²) truncated to
    # [sigma_min, sigma_max], sampled by equal probability MASS (§3.3). Local to TUL —
    # see morph/model/iter_cond.py's module docstring for why these are NOT the
    # diffusion_blocks.py module globals.
    db1_sigma_min: float = 0.002
    db1_sigma_max: float = 80.0
    db1_p_mean: float = -1.2
    db1_p_std: float = 1.2
    db1_sigma_data: float = 0.5
    # Loss weighting hook (mission spec): EDM's w(σ) = (σ²+σ_d²)/(σ·σ_d)² (App. C),
    # multiplied into the per-step loss when a caller reads it (train.py). False (the
    # default) means w(σ) == 1.0 everywhere — the diffusion_blocks.py finding (2026-08-19,
    # TULConfig docstring above) is that this weighting, derived for an L2 regression
    # loss, badly over-weights the near-trivial low-σ region of a CROSS-ENTROPY loss.
    # Exposed as a knob rather than baked to True/False permanently because it has not
    # been re-measured against the CE-supervised db1 step specifically.
    db1_w_sigma: bool = False
    # Euler-ladder eval step count. 0 -> model.mean_depth (mission spec: "K = the
    # model's mean_depth by default"), so a db1 arm's inference cost tracks the SAME
    # loop depth its bptt sibling would have paid, with no separate knob to forget.
    db1_ladder_steps: int = 0
    # Detach the readout matrix inside the MUX head. TRUE is the corrected default and
    # the setting the paper's own protocol implies: MUX LoRA-finetunes a PRETRAINED
    # model and uses W as a FIXED readout for supervision. MORPH trains from scratch,
    # and worse, `embed.lm_weight()` is WEIGHT-TIED to the INPUT embeddings — so an
    # undetached head sends the auxiliary gradient into the embedding table that (a)
    # every token's representation depends on and (b) the slot input itself is a
    # bag-mean OF (`E_slot + mean(embed(span))`), a feedback loop. Measured with
    # detach OFF: arm v1a diverged and aborted at step 2800 while its control ran
    # healthy past 3250 (lab/experiments/failures/2026-08-25-mux-head-arm-v1a.md).
    # False is kept ONLY so that failure stays reproducible as an ablation.
    mux_detach_head: bool = True
    # Subtract the batch's mean TOKEN signal from every span bag-mean.
    # Measured 2026-08-27 on tul-v1a2b step_3500: the embedding table has a
    # common mean of norm 0.423 against a mean per-token deviation of 1.049.
    # A bag-mean over a span shrinks the DEVIATIONS by 1/sqrt(span) but preserves
    # that common mean EXACTLY, so every slot inherits the same vector. Predicted
    # pairwise cosine from this geometry alone: 0.394 (span 4) to 0.839 (span 32),
    # against a MEASURED slot cosine of +0.39..+0.71 — the collapse is arithmetic,
    # not a training pathology.
    # The subtraction is DETACHED: it must not put a dense gradient on the
    # embedding table (the mistake that made arm v1a diverge).
    # Honest caveat: `E_slot` is added to the same bag-mean, so a CONSTANT shift is
    # already within the model's reach. The value here is that the batch mean
    # TRACKS the drifting embedding mean, which one learned vector cannot.
    center_bag_mean: bool = False
    # Fraction of TOTAL steps before the MUX head switches on. Wolfe's point:
    # MUX starts from a PRETRAINED model, so its latents predict spans using
    # representations that already exist; we asked a random-init model to do it
    # and it learned only the corpus marginal (7.03 vs unigram 7.32). 0.0 = on
    # from step 0 (v1a behaviour). Same schedule shape as `tul.activate_at`.
    mux_activate_at: float = 0.0
    # ── SIGReg on the slot states (LeJEPA arXiv 2511.08544; morph/model/sigreg.py)
    # Attacks a MEASURED pathology: slot states have effective rank 1.7-4.8 in
    # 1024 dims with mean pairwise cosine +0.39..+0.71 at every checkpoint. 0.0
    # builds nothing and adds no term.
    sigreg_lambda: float = 0.0
    sigreg_slices: int = 256             # M directions (paper default)
    sigreg_activate_at: float = 0.0      # same schedule shape as mux_activate_at

    # ── TG restriction (docs/tul-tg-spec.md) ──────────────────────────────────
    # False builds nothing new and adds no mask (bit-identical to master, spec T4).
    # True closes the token shortcut: within-span attention only in the window
    # branch, direct slot attention in the compressed branch (spec §§1-3). The
    # model constructor RAISES if this is set with `use_kernels=true` — the TG
    # arms are eager-only (spec §2/§6).
    tg_restrict: bool = False
    # E-SAC (span-aligned compression, .agents/notes/proposed/architecture/
    # 2026-09-01-slot-channel-recovery.md + the E1 mask-surgery result): the
    # prelude/coda COMPRESSED branch attends per-SPAN mean-pooled K/V of the
    # span's token positions (computed from the LIVE post-projection k/v at each
    # layer) instead of the slot positions. Slots stay readable through the
    # window branch's tg_allow ("or j is any slot position"). Causality is at
    # span granularity: summary j is visible to position i iff span j's LAST
    # token position < i, so a token never sees its own span's summary (which
    # would leak the span's future tokens). Zero new parameters, no RNG draws —
    # false is bit-identical. Requires tg_restrict.
    tg_span_comp: bool = False
    # E-SAC-G (the frozen binding of lab/experiments/failures/
    # 2026-09-01-span-aligned-compression.md, P-S1 FALSE): replace the mean pool
    # with a LEARNED per-head gated softmax pool over each span's token
    # positions — gate logit = <k_pos, W_g[h]>, softmax within the span, the
    # same weights pool k and v. W_g is one [n_heads, d_head] zero-init
    # parameter per attention layer, so at init the pool is EXACTLY the mean
    # (uniform softmax) and the arm starts as tul-sac. Requires tg_span_comp.
    tg_span_gate: bool = False
    # TG3 (spec §6): soften the restriction with one extra allow term — the
    # PREVIOUS span, not just the current one and the slots. Meaningless without
    # `tg_restrict` (there would be nothing to soften).
    tg_soft_prev_span: bool = False

    # ── slot seed (arms TG4a/TG4b; lab/divergence/TG-WORKLIST.md A1) ──────────
    # `pooling_probe` on tg2-s1@3500 confirms the plain-mean pooling law (slope
    # -0.470, r2 0.922): slot-seed signal falls from 0.516 (span 4-5) to 0.210
    # (span 24-32) against a shared constant ||E_slot||=0.238. Under `tg_restrict`
    # the slot already attends its whole span through the prelude, so the bag-mean
    # is redundant AND diluting. Construction-time dispatch — the mode is fixed at
    # init, never branched on per call.
    #   "bag_mean" : E_slot + mean_j embed(t_j) over the span (today's behaviour,
    #                the default, bit-identical to master).
    #   "e_slot"   : E_slot alone. No bag-mean term is computed at all (arm TG4a).
    #   "boundary" : E_slot + W_sent . embed(t_last), t_last the LAST token of the
    #                span (arm TG4b). This is a SEED-LEVEL approximation of Thought
    #                Gestalt's mid-layer tap (arXiv 2512.25026's m_t = W_sent .
    #                H^(l_s)_{i_EOS}) — it reads the raw token embedding, not a
    #                mid-prelude hidden state, so it is NOT the faithful TG tap.
    #                Builds a new bias-free `nn.Linear(d, d)` (`TULSlots.W_sent`)
    #                ONLY in this mode — an unused Linear still draws weight decay
    #                and perturbs the RNG stream, so the other two modes build
    #                nothing.
    #   "content"  : E2's `bag` column (lab/experiments/failures/2026-09-01-bound-
    #                seed-rank.md): the plain span bag-mean, exactly "bag_mean"
    #                minus the E_slot additive term. Arm W1 of the write-side
    #                ladder (lab/experiments/planned/2026-09-01-write-side-ladder.md)
    #                — E2 measured that the shared E_slot constant collapses every
    #                seed to ~rank-1 (unit rank 3.07 -> 38.30 for "bound" alone with
    #                E_slot removed), so this mode tests whether dropping ONLY the
    #                constant is enough to restore write-side rank. Builds nothing new.
    #   "bound"    : HRR-style binding — a frozen per-offset orthogonal rotation
    #                applied to each token of the span before summing, no E_slot term
    #                (arm W2 of the same ladder; exactly the E2 probe's "bound_noeslot"
    #                column, lab/divergence/bound_seed_rank.py). seed = (1/sqrt(n)) *
    #                sum_j R[offset_j] @ embed(t_j), offset_j the token's 0-based
    #                position within its span, R frozen (:func:`build_bound_rotations`,
    #                seed 17 — the exact rotation the probe used). A token whose
    #                offset falls at or past `bound_span_cap` is DROPPED from the sum
    #                (see :func:`bound_seed`). Builds one new persistent=False buffer
    #                (`TULSlots.bound_R`, `[bound_span_cap, d, d]`) ONLY in this mode.
    slot_seed: str = "bag_mean"
    # Rotation-table size for slot_seed="bound" — must be >= the data's `tul.span_cap`
    # (the loader forces a boundary at that length, so no span exceeds it). Duplicated
    # here rather than read from the loader's BoundaryRule because TULConfig is a
    # construction-time, data-independent object (module docstring: "the segmentation
    # keys ... belong to the loader and never reach the model") and `bound_R` must be
    # sized before any batch is seen. `morph/training/tul_setup.py` sets this from
    # `rule.span_cap` so it can never silently disagree with the data. Ignored (and
    # nothing built) unless `slot_seed == "bound"`.
    bound_span_cap: int = 32
    # Eval-only instrument switch (arm GL1). false = every existing arm's eval is
    # unchanged in COST as well as in value. true adds, at each eval batch: the
    # zero / shuffle / wrong-seed plan ablations and the slot-state geometry probe —
    # three extra coda passes and one extra prelude pass, which is why it is a knob
    # and not a default.
    eval_ablations: bool = False

    def __post_init__(self) -> None:
        if self.prefix_k < 1:
            raise ValueError(f"tul.prefix_k must be ≥ 1, got {self.prefix_k}")
        # 1.0 is legal and meaningful: it is Bowman 2015's INPUTLESS decoder control
        # (every token state replaced by E_mask), the extreme end of the §3.4 arm sweep.
        if not 0.0 <= self.token_state_dropout <= 1.0:
            raise ValueError(
                f"tul.token_state_dropout must be in [0,1], got {self.token_state_dropout}")
        if self.emit_weight < 0.0 or self.plast_weight < 0.0:
            raise ValueError("tul.emit_weight / tul.plast_weight must be >= 0")
        if self.mux_beta < 0.0:
            raise ValueError(f"tul.mux_beta must be >= 0, got {self.mux_beta}")
        if not 0.0 < self.mux_rho < 1.0:
            raise ValueError(f"tul.mux_rho must be in (0,1), got {self.mux_rho}")
        if self.mux_tau <= 0.0:
            raise ValueError(f"tul.mux_tau must be > 0, got {self.mux_tau}")
        if self.mux_target not in ("own", "next"):
            raise ValueError(
                f"tul.mux_target must be 'own' or 'next', got {self.mux_target!r}")
        if self.cond_layers < 0:
            raise ValueError(f"tul.cond_layers must be >= 0, got {self.cond_layers}")
        if self.detach_z and self.tokens_through_core:
            raise ValueError(
                "tul.detach_z with tul.tokens_through_core (A2) is not defined: A2 has "
                "no slot state for the coda to read detached.")
        if self.sigreg_lambda < 0.0:
            raise ValueError(f"tul.sigreg_lambda must be >= 0, got {self.sigreg_lambda}")
        if self.sigreg_slices < 1:
            raise ValueError(f"tul.sigreg_slices must be >= 1, got {self.sigreg_slices}")
        for _n in ("mux_activate_at", "sigreg_activate_at"):
            _v = getattr(self, _n)
            if not 0.0 <= _v < 1.0:
                raise ValueError(f"tul.{_n} must be in [0,1), got {_v}")
        if self.coda_token_cut < 0:
            raise ValueError(
                f"tul.coda_token_cut must be ≥ 0, got {self.coda_token_cut}")
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
        if self.tg_span_comp and not self.tg_restrict:
            raise ValueError(
                "tul.tg_span_comp=true requires tul.tg_restrict=true (E-SAC replaces "
                "the tg compressed branch; there is no such branch without the "
                "restriction).")
        if self.tg_span_gate and not self.tg_span_comp:
            raise ValueError(
                "tul.tg_span_gate=true requires tul.tg_span_comp=true (the gate "
                "parameterizes the span pool; there is no span pool without "
                "tg_span_comp).")
        if self.tg_soft_prev_span and not self.tg_restrict:
            raise ValueError(
                "tul.tg_soft_prev_span=true requires tul.tg_restrict=true "
                "(docs/tul-tg-spec.md §6: TG3 SOFTENS the restriction — there is "
                "nothing to soften when the restriction itself is off).")
        _legal_slot_seed = ("bag_mean", "e_slot", "boundary", "content", "bound")
        if self.slot_seed not in _legal_slot_seed:
            raise ValueError(
                f"tul.slot_seed must be one of {_legal_slot_seed}, got {self.slot_seed!r}")
        if self.bound_span_cap < 1:
            raise ValueError(
                f"tul.bound_span_cap must be >= 1, got {self.bound_span_cap}")
        _legal_recur_gate = ("none", "grt")
        if self.recur_gate not in _legal_recur_gate:
            raise ValueError(
                f"tul.recur_gate must be one of {_legal_recur_gate}, got {self.recur_gate!r}")
        if self.recur_gate != "none":
            if self.db_loop:
                raise ValueError(
                    "tul.recur_gate with tul.db_loop is not defined: the db carry is "
                    "detached per iteration and a convex blend with a detached branch "
                    "silently changes what the local losses supervise. Build it when an "
                    "arm needs it.")
            if self.core_stage_cond != "none":
                raise ValueError(
                    "tul.recur_gate with tul.core_stage_cond is banned outright: "
                    "iteration conditioning poisons depth-earning during formation "
                    "(lab/experiments/successes/2026-08-30-tul-condzero-probe.md).")
            if self.tokens_through_core:
                raise ValueError(
                    "tul.recur_gate is wired into the SLOT loop (_tul_core) only; "
                    "tokens_through_core runs the token core region, which has no gate. "
                    "Raises rather than silently running an ungated token loop.")
        _legal_stage_cond = ("none", "iter", "sigma")
        if self.core_stage_cond not in _legal_stage_cond:
            raise ValueError(
                f"tul.core_stage_cond must be one of {_legal_stage_cond}, "
                f"got {self.core_stage_cond!r}")
        if self.db1_cond_dim < 1:
            raise ValueError(f"tul.db1_cond_dim must be >= 1, got {self.db1_cond_dim}")
        if not 0.0 < self.db1_sigma_min < self.db1_sigma_max:
            raise ValueError(
                f"tul.db1_sigma_min/db1_sigma_max must satisfy 0 < min < max, got "
                f"{self.db1_sigma_min}, {self.db1_sigma_max}")
        if self.db1_p_std <= 0.0:
            raise ValueError(f"tul.db1_p_std must be > 0, got {self.db1_p_std}")
        if self.db1_sigma_data <= 0.0:
            raise ValueError(f"tul.db1_sigma_data must be > 0, got {self.db1_sigma_data}")
        if self.db1_ladder_steps < 0:
            raise ValueError(
                f"tul.db1_ladder_steps must be >= 0 (0 -> model.mean_depth), "
                f"got {self.db1_ladder_steps}")
        if self.core_stage_cond != "sigma" and self.db1_w_sigma:
            raise ValueError(
                "tul.db1_w_sigma=true has no defined meaning without "
                "tul.core_stage_cond='sigma' (there is no sampled sigma to weight by).")
        if self.center_bag_mean and self.slot_seed != "bag_mean":
            # Judgment call beyond the letter of the brief (which named only "e_slot"):
            # "boundary" computes no bag-mean at all (E_slot + W_sent . embed(t_last),
            # not a mean over the span), so `center_bag_mean` would silently do nothing
            # there — the exact "config key silently ignored" failure this file already
            # bans loudly for stp_lambda/set_lambda above. "content" and "bound" DO
            # compute a span aggregate, but centering was written and measured against
            # the "bag_mean" path only (write-side ladder note,
            # lab/experiments/planned/2026-09-01-write-side-ladder.md); kept scoped to
            # that one mode rather than silently reused against an aggregate it was
            # never validated on. Raise for every non-"bag_mean" mode.
            raise ValueError(
                f"tul.center_bag_mean=true with tul.slot_seed={self.slot_seed!r} is not "
                f"supported: centering is scoped to slot_seed='bag_mean' only.")


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
    # A one-hot [B, n_out, L] bag map times the signal. The obvious index_add_ form uses
    # float atomics, so its summation ORDER varies run to run and the result is not
    # bit-reproducible forward OR backward (measured: 20/20 repeats differ, 30.7 % of
    # backward elements, max 3.9e-3 in bf16). A GEMM has a fixed reduction order, agrees
    # with index_add_ to bf16 epsilon, and is ~10 % FASTER here. See
    # .agents/notes/proposed/process/2026-08-23-divergence-root-cause-plan.md task 0.1.
    # scatter_ WRITES (one bag per position) rather than accumulating, so it is exact.
    oh = signal.new_zeros(B, n_out, L)
    oh.scatter_(1, bag_id.unsqueeze(1), 1.0)
    oh = oh * sel.squeeze(-1).unsqueeze(1)
    cnt = oh.sum(dim=2, keepdim=True)
    out = torch.bmm(oh, signal) / cnt.clamp(min=1.0)
    # The dump bin aggregates trailing tokens that have no slot; zero it so the gather
    # at slot positions of tail pads reads 0 rather than a stray span mean.
    out = torch.cat([out[:, :n_bags], out.new_zeros(B, 1, C)], dim=1)
    return out


def boundary_token_index(bag_id: Tensor, token_sel: Tensor, n_bags: int) -> Tensor:
    """Position of the LAST token of each bag — the "boundary token" (arm TG4b).

    Companion to :func:`bag_mean`: same inputs, but instead of averaging the span's
    token signal it locates the single position that terminates it. Vectorized with
    ``scatter_reduce_(reduce="amax")`` over the position index — no Python loop over
    slots (a per-row Python loop over up to ``max_slots`` spans would be the actual
    hot-path cost here; this is one kernel launch regardless of span count).

    Args:
        bag_id:    ``[B, L]`` int64 — see :func:`bag_mean`.
        token_sel: ``[B, L]`` bool/float, 1/True at token positions — see :func:`bag_mean`.
        n_bags:    number of real bags (``max_slots``).

    Returns:
        ``[B, n_bags + 1]`` int64. Row ``s`` is the largest token position ``p`` with
        ``bag_id[p] == s``, or ``-1`` when bag ``s`` owns no token position — a real
        slot index the row never reached, OR the dump bin. The ``-1`` sentinel is the
        pad-slot / dump-bin invariant callers must check before gathering.
    """
    B, L = bag_id.shape
    n_out = n_bags + 1
    pos = torch.arange(L, device=bag_id.device).unsqueeze(0).expand(B, L)
    sel = token_sel.to(torch.bool)
    cand = torch.where(sel, pos, pos.new_full((), -1))
    out = bag_id.new_full((B, n_out), -1)
    out.scatter_reduce_(1, bag_id, cand, reduce="amax", include_self=True)
    # The dump bin (index n_bags) aggregates TOKEN positions past the row's last
    # boundary (bag_mean's tail-pad case) — force it to -1 so the gather at a
    # tail-pad SLOT position (bag_mean's documented invariant: tail pads get
    # E_slot alone) never picks up a stray "boundary" from those leftover tokens.
    out[:, n_bags] = -1
    return out


def build_bound_rotations(d_model: int, span_cap: int, seed: int = 17) -> Tensor:
    """``[span_cap, d_model, d_model]`` frozen per-offset orthogonal rotations.

    One QR-orthogonalised matrix per within-span token offset (arm "bound",
    ``TULConfig.slot_seed``; the exact construction of the E2 probe,
    ``lab/divergence/bound_seed_rank.py``'s ``R``). Drawn from a PRIVATE generator —
    never ``torch.default_generator`` — so calling this at model construction never
    perturbs the global RNG stream: a model built with ``slot_seed="boundary"`` (or
    any other mode) draws byte-identical everything-else whether or not a "bound"
    model was built earlier in the same process. Same neutrality convention
    :class:`TULSlots` already documents for ``W_sent``.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.linalg.qr(torch.randn(d_model, d_model, generator=g))[0]
                        for _ in range(int(span_cap))])


def bound_seed(signal: Tensor, bag_id: Tensor, token_sel: Tensor, n_bags: int,
               R: Tensor) -> Tensor:
    """HRR-bound span seed (arm "bound"): ``sum_j R[offset_j] @ embed(t_j) / sqrt(n)``.

    Companion to :func:`bag_mean` — same inputs, same ``[B, n_bags+1, C]`` output
    shape and dump-bin-is-zero contract — but instead of the plain mean it binds
    each token to a frozen rotation keyed on its 0-based OFFSET within the span
    (order of appearance) before summing, and divides by ``sqrt(n)`` rather than
    ``n``. Exactly the "bound_noeslot" column of
    ``lab/divergence/bound_seed_rank.py``'s E2 probe, vectorized for the batched
    training path (no python loop over batch elements or slots).

    A token whose offset falls at or past ``span_cap = R.shape[0]`` — a span longer
    than the rotation table, which cannot happen under a :class:`BoundaryRule` that
    forces a boundary at ``span_cap`` but CAN happen if a layout is built with a
    different ``span_cap`` than ``TULConfig.bound_span_cap`` — is DROPPED from both
    the sum and the ``n`` used for the ``sqrt`` normalisation, rather than clamped
    into the table's last row (a silent wrong-rotation collision is worse than a
    silently smaller sum).

    Method (offset, GEMM-only, no per-position ``[..., d, d]`` tensor is ever
    materialised — that would be ``B · L · d²`` floats, ~65 GB at a training batch's
    shape):

      1. ``offset[b, l]`` = (# token positions with the same ``bag_id`` at or before
         ``l``) − 1, via one cumulative sum along ``L`` of the same one-hot bag map
         :func:`bag_mean` builds (deterministic — a prefix sum has one fixed
         reduction order, unlike ``index_add_``'s atomics).
      2. For each offset ``k`` in ``range(span_cap)`` (a loop over a small FIXED
         constant, not over batch or slots): mask ``signal`` to the positions with
         that offset (each bag has at most one), reduce into bags with the same
         one-hot GEMM :func:`bag_mean` uses (``[B, n_out, L] @ [B, L, C]`` — cheap,
         since the mask has already zeroed everything but one position per bag),
         THEN apply ``R[k]`` to the resulting ``[B, n_out, C]`` bag vectors — matrix
         composition lets the rotation move to after the reduction
         (``oh @ (x_k @ Rk^T) == (oh @ x_k) @ Rk^T``, associativity), so ``R[k]`` is
         ever applied at bag width (``n_out``), never at sequence width (``L``).

    Args:
        signal:    ``[B, L, C]``.
        bag_id:    ``[B, L]`` int64 — see :func:`bag_mean`.
        token_sel: ``[B, L]`` — see :func:`bag_mean`.
        n_bags:    number of real bags (``max_slots``).
        R:         ``[span_cap, C, C]`` frozen orthogonal rotations
                   (:func:`build_bound_rotations`; ``TULSlots.bound_R``).

    Returns:
        ``[B, n_bags + 1, C]``; row ``n_bags`` (the dump bin) is exactly 0.
    """
    B, L, C = signal.shape
    span_cap = int(R.shape[0])
    n_out = n_bags + 1
    sel = token_sel.to(torch.bool)

    oh = signal.new_zeros(B, n_out, L)
    oh.scatter_(1, bag_id.unsqueeze(1), 1.0)
    oh = oh * sel.unsqueeze(1).to(signal.dtype)

    # Offset of each token within its span: cumulative count of same-bag token
    # positions up to and including this one, minus one. `cumsum` is a fixed-order
    # prefix reduction (no atomics) — the same reproducibility bar bag_mean's GEMM
    # meets, just via a different deterministic primitive.
    count_at = torch.gather(oh.cumsum(dim=2), 1, bag_id.unsqueeze(1)).squeeze(1)  # [B, L]
    offset = (count_at - 1).to(torch.int64)
    keep = sel & (offset >= 0) & (offset < span_cap)
    offset_safe = offset.clamp(min=0, max=span_cap - 1)

    out = signal.new_zeros(B, n_out, C)
    cnt = signal.new_zeros(B, n_out)
    keep_f = keep.to(signal.dtype)
    for k in range(span_cap):
        mask_k = (keep_f * (offset_safe == k).to(signal.dtype)).unsqueeze(-1)  # [B, L, 1]
        x_k = signal * mask_k                                    # zero outside bag/offset
        bagged_k = torch.bmm(oh, x_k)                             # [B, n_out, C]
        out = out + bagged_k @ R[k].to(signal.dtype).transpose(0, 1)
        cnt = cnt + torch.bmm(oh, mask_k).squeeze(-1)             # kept-token count per bag

    out = out / cnt.clamp(min=1.0).sqrt().unsqueeze(-1)
    # Zero the dump bin exactly, matching bag_mean's contract (tail pads get E_slot
    # alone, or nothing, in every mode).
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

    Despite the name, the argument is generic: any ``[B, L]`` bool tensor that is True at
    the positions to push to the dump row works (arm CW reuses this with a token-cut mask
    instead of the slot mask, .agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md §"the change").
    """
    B, L = slot_mask.shape
    order = torch.argsort(slot_mask.to(torch.uint8), dim=1, stable=True)
    n_tok = (~slot_mask).sum(dim=1, keepdim=True)
    pos = torch.arange(L, device=slot_mask.device).unsqueeze(0)
    return torch.where(pos < n_tok, order, torch.full_like(order, L))


def window_drop_mask(slot_mask: Tensor, cut: int) -> Tensor:
    """``[B, L]`` bool, True at TOKEN positions with row index ``< cut`` (arm CW's mirror
    of :func:`compact_index`'s slot drop — .agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md).

    Every slot position is False here regardless of its row index — "KEEP every slot
    position, at every index" is the spec's line, and this is a GLOBAL cut (the same
    ``cut`` for every row), not a per-row window: "every query in the coda sees the same
    reduced sequence" is deliberate (spec §"the change" — a per-query window needs a mask,
    which spec §7.2 already rules out for the fused kernels).
    """
    B, L = slot_mask.shape
    pos = torch.arange(L, device=slot_mask.device).unsqueeze(0).expand(B, L)
    return (pos < cut) & (~slot_mask)


def cw2_retain_mask(candidates: Tensor, budget: Tensor, seed: int) -> Tensor:
    """``[B, L]`` bool: a per-row SEEDED random ``budget[row]``-sized subset of ``candidates``.

    Arm CW2's control (.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md): drop every slot and every
    early token EXCEPT an equal-KV-budget random subset. Vectorised with the same
    argsort + row-count-threshold trick as :func:`compact_index` — draw one uniform score
    per position, push non-candidates off the front with a score that always sorts last,
    then keep the ``budget[row]`` lowest-scored candidates in each row. Reproducible: the
    RNG state used is seeded here and nowhere else, so the same ``seed`` always retains
    the same positions.

    Args:
        candidates: ``[B, L]`` bool — the pool a position may be drawn from (e.g. token
                    positions with row index ``< C``). Never a slot position.
        budget:     ``[B]`` int — how many of each row's candidates to retain. Every
                    candidate sorts strictly before every non-candidate (score ``< 1``
                    vs. exactly ``2.0``), so a candidate's RANK never exceeds its row's
                    candidate count regardless of ``budget`` — a ``budget`` that exceeds
                    the pool is a no-op (retains the whole pool), not an error, with no
                    separate clamp needed: the final ``candidates &`` intersection is
                    what actually enforces it.
        seed:       int, logged by the caller so the eval screen is reproducible.
    """
    B, L = candidates.shape
    gen = torch.Generator(device=candidates.device)
    gen.manual_seed(int(seed))
    scores = torch.rand(B, L, generator=gen, device=candidates.device)
    scores = torch.where(candidates, scores, torch.full_like(scores, 2.0))  # non-candidates sort last
    order = torch.argsort(scores, dim=1, stable=True)
    rank = torch.empty_like(order)
    ar = torch.arange(L, device=candidates.device).unsqueeze(0).expand(B, L)
    rank.scatter_(1, order, ar)
    return candidates & (rank < budget.unsqueeze(1))


# ── parameters ───────────────────────────────────────────────────────────────

class TULSlots(nn.Module):
    """The TUL parameter groups (spec §3.1/§3.2/§3.4/§5).

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
    * ``W_sent`` ``[d, d]``, bias-free — ONLY built when ``tul.slot_seed == "boundary"``
      (arm TG4b, lab/divergence/TG-WORKLIST.md A1). Projects the span's boundary token
      embedding into the slot input. An unused Linear still draws weight decay and
      perturbs the optimizer state, so the other two ``slot_seed`` modes build nothing.
      Init ``std=0.02``, matching the rest of the model's Linear/Embedding inits
      (``embeddings.py``, ``gla.py``, ``mhc.py``) rather than the zero/identity
      convention below — see the RNG-neutrality note there for why that convention
      does NOT extend to this parameter.
    * ``bound_R`` ``[bound_span_cap, d, d]`` buffer, ``persistent=False`` — ONLY built
      when ``tul.slot_seed == "bound"`` (write-side ladder arm W2). Frozen per-offset
      orthogonal rotations from :func:`build_bound_rotations` (private generator, seed
      17 — the exact one ``lab/divergence/bound_seed_rank.py``'s E2 probe used).
      ``persistent=False`` is REQUIRED, not a style choice: at ``bound_span_cap=32``,
      ``d=768`` fp32 this is ``32·768·768·4 ≈ 75 MB`` — reproducible deterministically
      from the seed at construction, so paying that in every checkpoint would be pure
      waste. It still moves with the module's ``.to(device)`` (buffers always do); it
      is simply excluded from ``state_dict()``.

    Constructed only when TUL is configured, and LAST in ``MORPHTransformer.__init__``
    so a non-TUL model is byte-identical to the baseline (the ``attach_retention``
    convention). Every init here is RNG-NEUTRAL: ``E_slot`` / ``E_mask`` / ``W_prefix``
    are deterministic (zero draws) and ``W_sent`` / ``bound_R`` take their one real draw
    from a PRIVATE fixed generator each, so the global RNG stream is untouched and a
    TUL model's base weights match a baseline built with the same seed. Verified
    2026-08-28: models built from ``tul_tg2`` / ``tul_tg4a`` / ``tul_tg4b`` at seed 1
    share all 494 parameters byte-identically, with ``tul.W_sent.weight`` the sole
    addition in TG4b (``bound_R`` is a buffer, not a parameter, and is verified the
    same way in ``tests/test_slot_seed_modes.py``).
    """

    def __init__(self, d_model: int, tul: TULConfig):
        super().__init__()
        self.tul = tul
        # [d] shared, or [per_slot_embed, d] one row per slot INDEX. The shared shape stays
        # the default so every existing checkpoint loads and every existing run is unchanged.
        self.E_slot = nn.Parameter(
            torch.zeros(tul.per_slot_embed, d_model) if tul.per_slot_embed > 0
            else torch.zeros(d_model))
        self.E_mask = nn.Parameter(torch.zeros(d_model))
        eye = torch.eye(d_model).unsqueeze(0).repeat(tul.prefix_k, 1, 1)
        self.W_prefix = nn.Parameter(eye)
        self.W_sent: nn.Linear | None = None
        if tul.slot_seed == "boundary":
            self.W_sent = nn.Linear(d_model, d_model, bias=False)
            # RNG-NEUTRAL init from a FIXED generator, the `_seat` precedent below.
            # W_sent is the only TUL parameter with no meaningful zero/identity init
            # (unlike E_slot there is no activation-step re-init to rescue a zero
            # start), so it needs a real draw — but taking that draw from the GLOBAL
            # stream would shift every parameter built AFTER TULSlots, and
            # `MORPHTransformer.__init__` builds `core_init` after it (`_SCSEInit`
            # draws a Linear init whenever `core_init_scale > 0`). A private generator
            # keeps arm TG4b's base weights byte-identical to TG4a's under EVERY
            # config, not merely the ones where core_init happens to be RNG-free.
            g = torch.Generator(device="cpu").manual_seed(0x5E17)
            with torch.no_grad():
                self.W_sent.weight.copy_(
                    torch.empty(self.W_sent.weight.shape, device="cpu").normal_(
                        mean=0.0, std=0.02, generator=g))
        # Frozen rotation table — ONLY in "bound" mode; None in every other mode (no
        # buffer entry with a live tensor, nothing to move to device, nothing to save).
        # PRIVATE generator (seed 17, matching the E2 probe exactly): building this must
        # not perturb the global RNG stream, the same neutrality W_sent needs above.
        bound_R = (build_bound_rotations(d_model, tul.bound_span_cap)
                  if tul.slot_seed == "bound" else None)
        self.register_buffer("bound_R", bound_R, persistent=False)

    @torch.no_grad()
    def init_at_activation(self, lm_weight: Tensor) -> None:
        """Set ``E_slot`` to the mean of the (live, trained) embedding table — spec §5.

        Called at the activation step, not at construction: the point of the Block
        Transformer init is that the new position starts as the average TRAINED token,
        which a randomly-initialised table cannot provide. Idempotent-unsafe by design —
        the training loop calls it exactly once and records it in the checkpoint.
        """
        mean = lm_weight.mean(dim=0).to(self.E_slot.dtype)
        if self.E_slot.dim() == 1:
            self.E_slot.copy_(mean)
            return
        # Per-slot rows: every row starts at the same mean, so the forward at the activation
        # step is IDENTICAL to the shared version, plus optional jitter that breaks the
        # degeneracy from step 0. Deterministic — a fixed generator, no draw from the global
        # stream — so the TUL model's base weights still match a baseline at the same seed.
        self.E_slot.copy_(mean.unsqueeze(0).expand_as(self.E_slot))
        if self.tul.per_slot_embed_std > 0.0:
            g = torch.Generator(device="cpu").manual_seed(0x5107)
            j = torch.randn(self.E_slot.shape, generator=g).to(self.E_slot.device,
                                                               self.E_slot.dtype)
            self.E_slot.add_(j * (self.tul.per_slot_embed_std * float(mean.std())))

    # -- forward helpers ---------------------------------------------------
    def _e_slot_term(self, bag_id: Tensor, dtype: torch.dtype) -> Tensor:
        """``E_slot`` broadcast over every position, indexed by its OWN slot index.

        ``[d]`` (shared) broadcasts over every position; ``[S, d]`` (``per_slot_embed``)
        is indexed by the position's own slot index, which ``bag_id`` already carries
        for both token and slot positions. Shared by all three ``slot_seed`` modes.
        """
        e = self.E_slot.to(dtype)
        return e[bag_id.clamp(max=e.shape[0] - 1)] if e.dim() == 2 else e

    def slot_input(self, signal: Tensor, layout: SlotLayout, add_e_slot: bool) -> Tensor:
        """Replace slot positions of ``signal`` ``[B, L, C]`` with the slot's input.

        ``add_e_slot`` is True for the token embedding and False for the bigram /
        value-embed signals. ``tul.slot_seed`` (construction-time; TG-WORKLIST A1)
        changes ONLY the ``add_e_slot=True`` path — bigram / value-embed signals stay
        the plain bag-mean of the span in EVERY mode ("bigram/value-embed signals for
        the slot are the bag-mean, exactly the TST ``ve_bagged`` path"), so a caller
        with ``add_e_slot=False`` always falls through to the code below unchanged:

            "bag_mean" (default): ``E_slot + mean_j embed(t_j)`` over the span
                        (spec §3.2) — bit-identical to pre-``slot_seed`` master.
            "e_slot"   (``add_e_slot=True`` only): ``E_slot`` alone. No bag-mean is
                        computed — the slot value does not depend on the span's
                        token embeddings at all.
            "boundary" (``add_e_slot=True`` only): ``E_slot + W_sent . embed(t_last)``,
                        ``t_last`` the span's LAST token position. A slot with no span
                        (a tail-pad position, bag_id at the dump bin) gets ``E_slot``
                        alone — see :func:`boundary_token_index`'s dump-bin handling.
            "content"  (``add_e_slot=True`` only; arm W1): ``mean_j embed(t_j)`` over
                        the span — exactly the "bag_mean" formula with the ``E_slot``
                        term dropped. A slot with no span (dump bin) resolves to
                        exactly 0, not ``E_slot`` — there is nothing else to add here.
            "bound"    (``add_e_slot=True`` only; arm W2): the HRR-bound span sum, no
                        ``E_slot`` term — see :func:`bound_seed`. Same dump-bin-is-zero
                        contract as "content".
        """
        token_sel = (~layout.slot_mask).to(signal.dtype)

        if add_e_slot and self.tul.slot_seed == "e_slot":
            at_pos = signal.new_zeros(signal.shape) + self._e_slot_term(layout.bag_id,
                                                                        signal.dtype)
            return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

        if add_e_slot and self.tul.slot_seed == "boundary":
            assert self.W_sent is not None    # built iff slot_seed == "boundary" (__init__)
            b_idx = boundary_token_index(layout.bag_id, token_sel, layout.max_slots)
            b_idx_at_pos = torch.gather(b_idx, 1, layout.bag_id)              # [B, L]
            valid = (b_idx_at_pos >= 0).unsqueeze(-1)                         # False: no span
            safe_idx = b_idx_at_pos.clamp(min=0)
            boundary_sig = torch.gather(
                signal, 1, safe_idx.unsqueeze(-1).expand(*safe_idx.shape, signal.shape[-1]))
            proj = self.W_sent(boundary_sig.to(signal.dtype))
            at_pos = torch.where(valid, proj, torch.zeros_like(proj))
            at_pos = at_pos + self._e_slot_term(layout.bag_id, signal.dtype)
            return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

        if add_e_slot and self.tul.slot_seed == "content":
            # "bag_mean" minus the E_slot term — the plain span content mean, and
            # NOTHING else added, so a no-span dump-bin position is exactly 0 (unlike
            # every other mode, which has an E_slot term to fall back on there).
            bags = bag_mean(signal, layout.bag_id, token_sel, layout.max_slots)
            at_pos = torch.gather(
                bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape,
                                                            signal.shape[-1]))
            return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

        if add_e_slot and self.tul.slot_seed == "bound":
            assert self.bound_R is not None    # built iff slot_seed == "bound" (__init__)
            bags = bound_seed(signal, layout.bag_id, token_sel, layout.max_slots,
                              self.bound_R.to(signal.dtype))
            at_pos = torch.gather(
                bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape,
                                                            signal.shape[-1]))
            return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

        # "bag_mean" (default), and every add_e_slot=False caller in EVERY mode: the
        # plain bag-mean, unchanged from master.
        bags = bag_mean(signal, layout.bag_id, token_sel, layout.max_slots)
        at_pos = torch.gather(
            bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, signal.shape[-1]))
        if self.tul.center_bag_mean:
            # Remove the common mean the bag-mean would otherwise preserve exactly.
            # Applied AFTER the gather and only at REAL slots, so the dump bin stays
            # exactly zero (bag_mean's documented invariant: tail pads get E_slot
            # alone) and empty pad slots are not handed a spurious -mu.
            sel = token_sel.unsqueeze(-1)
            mu = ((signal * sel).sum(dim=(0, 1)) / sel.sum().clamp(min=1.0)).detach()
            n_slots = layout.slot_index.shape[1]
            valid_at = torch.gather(layout.slot_valid, 1,
                                    layout.bag_id.clamp(max=n_slots - 1))
            real = ((layout.bag_id < n_slots) & valid_at).unsqueeze(-1)
            at_pos = torch.where(real, at_pos - mu, at_pos)
        if add_e_slot:
            at_pos = at_pos + self._e_slot_term(layout.bag_id, signal.dtype)
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


class TULGate(nn.Module):
    """The span-length gate: one scalar read off each slot's core state, and the
    budget embedding that tells the coda how many tokens the plan covers.

    docs/tul-gate-spec.md §4 (forward), §5 (why the coda must be told), §6 (loss),
    §9 (invariants), §10 (instruments).

    **Zero RNG draws at construction.** ``nn.Linear`` and ``nn.Embedding`` both draw from
    the global generator in ``reset_parameters``; a model that drew them would advance the
    RNG stream and change every later Poisson depth and dropout mask, so
    ``gate_lambda = 0`` would NOT be bit-identical to arm A1 (§9 invariant 1). The head is
    therefore a rank-1 linear written out as two zero parameters, and the budget table is a
    plain zero ``Parameter`` addressed with ``F.embedding`` — mathematically identical to
    ``nn.Linear(d, 1)`` and ``nn.Embedding(k_max+1, d)``, and deterministic. Constructed
    LAST (after :class:`TULSlots`), the same convention that keeps the TUL parameters from
    perturbing the baseline.

    **What gradient reaches the core.** The readout runs on the core's output OUTSIDE the
    checkpoint / ``no_grad`` block, so it shapes the core state exactly on the iterations
    inside the truncated-BPTT window and is a pure readout on the frozen ones — the SAME
    window the token loss uses. Every iteration still supervises the head itself (``w``,
    ``b``, ``norm.scale``), so no slot's label is silently dead (a depth ≤ ``n_nograd``
    slot would otherwise contribute nothing; that is ~28 % of slots at mean_depth 6).
    """

    def __init__(self, d_model: int, gate: TULGateConfig):
        super().__init__()
        self.gate = gate
        self.norm = RMSNorm(d_model)                       # scale init = ones, no RNG
        self.w = nn.Parameter(torch.zeros(d_model))        # ≡ nn.Linear(d,1).weight
        self.b = nn.Parameter(torch.zeros(1))              # ≡ nn.Linear(d,1).bias
        # Index 0 is the pad slot's budget and stays at zero-init unless a real span of
        # length 0 exists, which the packer forbids (span_len is clamped to ≥ 1).
        self.budget = nn.Parameter(torch.zeros(gate.k_decode_max + 1, d_model))

    # -- readout -----------------------------------------------------------
    def readout(self, h: Tensor) -> Tensor:
        """``[B, S, (n,) C]`` slot state → ``[B, S]`` gate output ``g ∈ (0, 1)``.

        The Hyper-Connection streams are collapsed by the MEAN, the same reduction the
        LM head uses (:meth:`MORPHTransformer._readout`), so the gate reads the same
        representation the rest of the model reads out. The norm is what makes the scalar
        scale-free: the core state is pre-``final_norm`` and its magnitude drifts over
        training, which would otherwise move ``g`` with no change in the length it means.
        """
        z = h.mean(dim=2) if h.dim() == 4 else h
        z = self.norm(z.float())
        return torch.sigmoid((z * self.w).sum(-1) + self.b)

    def choose_k(self, g: Tensor) -> Tensor:
        """``g`` → the integer budget ``round(g · k_max)``, clamped to ``[0, k_decode_max]``.

        ``k = 0`` means "keep thinking" (§1). Callers that need a length — the generator —
        clamp the low end to 1 themselves (§8); this does not, so the halting policy can
        see the zero.
        """
        return (g * self.gate.k_max).round().clamp_(0, self.gate.k_decode_max).long()

    # -- budget conditioning (§5) ------------------------------------------
    def budget_term(self, span_len: Tensor) -> Tensor:
        """``[B, S]`` int64 lengths → ``[B, S, C]`` additive term for the slot state."""
        return F.embedding(span_len.clamp(0, self.gate.k_decode_max), self.budget)

    def apply_budget(self, h_slots: Tensor, span_len: Tensor) -> Tensor:
        """Add the budget embedding to the looped slot states before the coda (§4).

        Broadcast over the Hyper-Connection stream axis, exactly as
        :meth:`MORPHTransformer._apply_injection` broadcasts every other additive signal.
        Zero-initialised, so at step 0 this is an exact no-op and the arm starts as A1.
        """
        if not self.gate.budget_cond:
            return h_slots
        term = self.budget_term(span_len).to(h_slots.dtype)
        if h_slots.dim() == 4:
            term = term.unsqueeze(2)
        return h_slots + term

    # -- bias seating (§10) ------------------------------------------------
    @torch.no_grad()
    def seat_bias(self, span_len: Tensor, valid: Tensor) -> float:
        """Set ``b`` to ``logit(mean span_len / k_max)`` — the corpus base rate (§10).

        The predecessor's gate had to TRAVEL to the base rate and never got there (bias
        −2.00000 → −2.00071 against a required 1.88). Starting there costs one batch of
        arithmetic and removes the failure mode. ``w`` is zero at init, so immediately
        after this call the gate emits the base rate for every slot — the correct
        constant predictor, and the floor that ``gate_separation`` is measured against.

        Returns the seated bias, for the log line and the wandb config.
        """
        sel = valid & (span_len > 0)
        n = sel.sum()
        if n == 0:
            raise ValueError("seat_gate_bias: the batch has no valid slot to seat from")
        mean_len = (span_len * sel).sum().float() / n.float()
        q = (mean_len / self.gate.k_max).clamp(1e-4, 1 - 1e-4)
        self.b.fill_(float(torch.log(q / (1 - q))))
        return float(self.b.item())

    # -- loss + instruments (§6, §10) --------------------------------------
    def loss(self, g_traj: Tensor, depths: Tensor, layout: SlotLayout,
             want_metrics: bool = True) -> dict:
        """``g_traj`` ``[B, S, T]`` + the realised depths → the §6 term and §10 numbers.

        Default target (``train_zeros=False``): ``span_len / k_max`` on EVERY iteration a
        slot is still looping. The head then predicts the length of the span it plans,
        unbiased at whatever depth generation happens to read it.

        ``train_zeros=True`` restores §6's original two-part target — zeros before the
        slot's last iteration, the length on it. Kept as a switch because it is the
        predecessor's shape, but it is not the default: the depth is a Poisson draw the
        head cannot observe, so that target's optimum is the HAZARD, and the length is
        multiplied away (see :class:`TULGateConfig`). Rows whose length came from OUR
        truncation RNG are excluded from the length term either way (§6, §9).
        """
        if layout.span_len is None:
            raise RuntimeError(
                "the TUL gate is built but the layout carries no span_len; the loader was "
                "built without a TulGateSpec (docs/tul-gate-spec.md §3.3).")
        B, S, T = g_traj.shape
        k_max = self.gate.k_max
        t_idx = torch.arange(T, device=g_traj.device).view(1, 1, T)
        last = (depths - 1).unsqueeze(-1)                        # [B, S, 1]
        at_final = t_idx == last
        before = t_idx < last
        valid = layout.slot_valid.unsqueeze(-1)
        sup = layout.len_supervised.unsqueeze(-1) & valid

        alive = t_idx <= last                                    # still looping at t
        tgt_len = (layout.span_len.float() / k_max).unsqueeze(-1)
        if self.gate.train_zeros:
            target = torch.where(at_final, tgt_len, torch.zeros((), device=g_traj.device))
            mask = (sup & at_final) | (valid & before)
        else:
            target = tgt_len.expand_as(g_traj)
            mask = sup & alive
        per = F.smooth_l1_loss(g_traj, target, reduction="none", beta=self.gate.huber_beta)
        denom = mask.sum().clamp(min=1)
        out = {"loss_gate": (per * mask).sum() / denom, "n_gate": denom.float()}
        if not want_metrics:
            return out

        # §10: a gate can sit at a tiny loss and still be dead. These are the numbers
        # that tell the difference, and every one of them exists because the predecessor
        # lost a ladder without it.
        fin_m = (valid & at_final).float()
        bef_m = (valid & before).float()
        g_fin = (g_traj * fin_m).sum() / fin_m.sum().clamp(min=1)
        g_bef = (g_traj * bef_m).sum() / bef_m.sum().clamp(min=1)
        out["gate_g_final"] = g_fin
        out["gate_g_before"] = g_bef
        out["gate_separation"] = g_fin - g_bef            # a dead gate reads ~0 here
        # per-iteration mean g over the slots still looping. Under train_zeros it IS the
        # hazard curve (§7's table is the prediction, this the observation); with the
        # zeros off it should be FLAT at E[span_len]/k_max, and a slope means the slot
        # state's readable length drifts with depth.
        al_v = alive & valid
        out["gate_hazard"] = ((g_traj * al_v).sum(dim=(0, 1))
                              / al_v.sum(dim=(0, 1)).clamp(min=1))        # [T]
        # chosen k vs gold, over the supervised final-iteration slots. Masked with NaN
        # and reduced with the nan-aware ops rather than boolean-indexed: `x[bool_mask]`
        # has a data-dependent output shape, which forces a device→host sync EVERY step.
        pick = sup & (at_final if self.gate.train_zeros else alive)
        nan = torch.tensor(float("nan"), device=g_traj.device)
        k_hat = torch.where(pick, self.choose_k(g_traj).float(), nan)
        gold = torch.where(pick, layout.span_len.unsqueeze(-1).expand_as(g_traj).float(), nan)
        out["gate_k_mean"] = k_hat.nanmean()
        out["gate_k_p50"] = k_hat.nanmedian()
        out["gate_gold_mean"] = gold.nanmean()
        out["gate_gold_p50"] = gold.nanmedian()
        out["gate_k_abs_err"] = (k_hat - gold).abs().nanmean()
        out["gate_k_zero_frac"] = torch.where(pick, (k_hat == 0).float(), nan).nanmean()
        # THE dead-gate number once the zeros are off: a constant predictor scores corr 0
        # however low its loss is, which is precisely the state the predecessor shipped.
        # gate_separation cannot serve that role here — with one target at every iteration
        # it is ~0 BY DESIGN, not by failure.
        kf = torch.where(pick, k_hat, nan)
        gf = torch.where(pick, gold, nan)
        km, gm = kf.nanmean(), gf.nanmean()
        cov = ((kf - km) * (gf - gm)).nanmean()
        sd = (kf - km).pow(2).nanmean().sqrt() * (gf - gm).pow(2).nanmean().sqrt()
        out["gate_k_corr"] = cov / sd.clamp(min=1e-6)
        # …and the SKILL: how much the gate beats the best CONSTANT predictor by, in
        # tokens. The median minimises mean-absolute-error, so `mae_const` is the floor
        # any constant can reach on this batch. Without it `gate_k_abs_err` is unreadable
        # — 8.62 tokens sounds bad and 9.03 is what predicting one number forever gets
        # you, so the whole claim lives in the 0.41 between them. Positive = real.
        out["gate_k_mae_const"] = (gf - gf.nanmedian()).abs().nanmean()
        out["gate_k_skill"] = out["gate_k_mae_const"] - out["gate_k_abs_err"]
        out["gate_bias"] = self.b.detach().squeeze()
        out["gate_w_norm"] = self.w.detach().norm()
        return out


def mux_span_targets(input_ids: Tensor, layout: SlotLayout, rho: float,
                     target: str = "next"
                     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Geometric MUX weights of the span a slot is supervised toward (Eq. 2).

    ``target="own"`` (arm GL1b) supervises slot ``i`` toward span ``i`` — the span it
    TERMINATES and, under ``tg_restrict``, the span it is the only route to. Span ``i``
    runs from ``slot_index[i-1] + prefix_k`` (or 0 for span 0) through the token before
    ``slot_index[i]``; the position index ``j`` restarts at 0 at the span's first token,
    so ``w_j = rho^j`` decays from the start of the span exactly as Eq. 2 specifies.
    Every valid slot is supervised, span 0 included — it has a terminating slot, which
    is the only thing "own" requires.

    ``target="next"`` (the default, unchanged) supervises slot ``i`` toward span
    ``i+1`` — the plan framing every arm before GL1 used. Its docstring follows.

    NEXT: slot ``i`` sits AFTER span ``i`` and its plan is decoded into span ``i+1``,
    so slot ``i``'s local target is the position-weighted superposition of span
    ``i+1``'s tokens: ``alpha_j ∝ rho^j`` (j = 0-based position inside the span),
    normalised within the span. The dense ``|V|``-vector is never built — the KL
    reduces to a weighted CE over the span's own token ids, so this returns
    per-POSITION weights and the slot each position supervises.

    A span ``k`` supervises slot ``k−1``, and only when BOTH slots exist: slot
    ``k−1`` (the plan being supervised) and slot ``k`` (which terminates the span,
    proving it complete). The trailing unterminated text (``bag_id`` dump bin) and
    span 0 (no preceding slot) supervise nothing.

    Returns ``(pos_valid [B,L] bool, alpha [B,L] fp32, tgt_slot [B,L] int64,
    slot_supervised [B,S] bool)``. ``tgt_slot`` is clamped to 0 at invalid
    positions — mask with ``pos_valid`` before use.
    """
    import math

    if target not in ("own", "next"):
        raise ValueError(f"mux target must be 'own' or 'next', got {target!r}")
    B, L = input_ids.shape
    S = layout.slot_index.shape[1]
    dev = input_ids.device
    k = layout.bag_id                                        # [B, L]
    kc = k.clamp(0, S - 1)

    if target == "own":
        # Span k supervises slot k. Valid when slot k is real and the position is a
        # token of that span. Span 0 starts at position 0; span k>0 starts right after
        # slot k-1's prefix block.
        own_ok = torch.gather(layout.slot_valid, 1, kc)
        pos_valid = (~layout.slot_mask) & (k < S) & own_ok
        prev_end = torch.gather(layout.slot_index, 1, (kc - 1).clamp(min=0))
        start = torch.where(kc >= 1, prev_end + layout.prefix_k,
                            torch.zeros_like(prev_end))
        j = (torch.arange(L, device=dev).unsqueeze(0) - start).clamp(min=0)
        w = torch.exp(j.to(torch.float32) * math.log(rho))
        w = torch.where(pos_valid, w, torch.zeros_like(w))
        # Normalise per (row, span); invalid positions scatter into a dump column.
        idx = torch.where(pos_valid, kc, torch.full_like(kc, S))
        denom = torch.zeros(B, S + 1, device=dev, dtype=w.dtype)
        denom.scatter_add_(1, idx, w)
        alpha = w / torch.gather(denom, 1, idx).clamp(min=1e-20)
        return pos_valid, alpha, kc * pos_valid.long(), denom[:, :S] > 0
    span_done = torch.gather(layout.slot_valid, 1, kc)       # slot k exists
    prev_ok = torch.gather(layout.slot_valid, 1, (kc - 1).clamp(min=0))
    pos_valid = (~layout.slot_mask) & (k >= 1) & (k < S) & span_done & prev_ok

    # position inside the span: p − (slot_index[k−1] + prefix_k)
    start = torch.gather(layout.slot_index, 1, (kc - 1).clamp(min=0)) + layout.prefix_k
    j = (torch.arange(L, device=dev).unsqueeze(0) - start).clamp(min=0)
    w = torch.exp(j.to(torch.float32) * math.log(rho))
    w = torch.where(pos_valid, w, torch.zeros_like(w))

    # normalise per (row, span). Invalid positions scatter into a DUMP column at
    # S+1 — NOT S, which slot_supervised below reads as "span S": pos_valid already
    # forbids k ≥ S, so column S must stay empty for slot S−1 to read as
    # unsupervised. (Column S collides with the dump only if the dump sits there.)
    idx = torch.where(pos_valid, kc, torch.full_like(kc, S + 1))
    denom = torch.zeros(B, S + 2, device=dev, dtype=w.dtype)
    denom.scatter_add_(1, idx, w)
    alpha = w / torch.gather(denom, 1, idx).clamp(min=1e-20)

    tgt_slot = (kc - 1).clamp(min=0)
    slot_supervised = denom[:, 1:S + 1] > 0                  # span k ≥ 1 → slot k−1
    return pos_valid, alpha, tgt_slot, slot_supervised
