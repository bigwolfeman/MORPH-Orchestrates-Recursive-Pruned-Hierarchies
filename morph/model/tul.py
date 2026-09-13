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

__all__ = ["TULConfig", "TULGate", "TULGateConfig", "TULGradPass", "TULSlotChain", "TULSlots", "bag_mean",
           "bound_seed", "build_bound_rotations", "mux_span_targets",
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
    # ── WHAT the prefix cells carry (arms `slot-spandec-strict-traj` / `-trajrep`,
    #    2026-09-13). "exit" is the shipped write and is BIT-IDENTICAL to the tree before
    #    this key existed: every one of a slot's `prefix_k` cells holds the loop's EXIT
    #    state through its own `W_prefix[k]`.
    #
    #    THE MEASURED PROBLEM. Twelve slot arms read a token K1-K6 inside
    #    [-0.0001, +0.0033], and the strict panel found out why: "depth use comes from
    #    reachability, not from the target" — the only arms whose passes carried anything
    #    were the forced relays (`prev-reach1` +0.0163). The coda has never been able to
    #    SEE a pass. It reads one vector, written once, after the loop is over, so pass 4
    #    can only matter to it by changing that one vector, and the direct measurement
    #    says one pass changes the coda's CE by 0.001-0.003 nats
    #    (`critic_gap_traj`, lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md).
    #
    #    "trajectory": cell k of a slot (k < prefix_k - 1) carries the state AFTER PASS
    #    k+1, and the LAST cell always carries the EXIT state, whatever depth the slot
    #    realised. So every written pass gets its OWN reader and its own direct gradient
    #    edge, instead of sharing one. A cell whose pass the slot never reached is a PAD:
    #    its carrier value is zero, it is removed from the coda's key set, and the packer
    #    already gives it no label (only the LAST cell carries the emit label).
    #    The exact rule, which is the thing to read before scoring any K-curve here:
    #      cell k, 0 <= k < K-1 : h_{k+1}, written iff the slot's realised depth >= k + 2
    #      cell K-1             : h_depth (the exit), always written
    #    so a slot of depth d writes exactly min(d, K) non-pad cells, the exit is present
    #    at EVERY forced depth and always in the same cell, and no cell ever holds a
    #    duplicate of the exit.
    #
    #    "entry_exit": the INFORMATION control, and the BINDING one. The Lean result in
    #    `.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md`
    #    says `I((z_1..z_T); Y) = I(z_1; Y)` and `I(z_T; Y) <= I(z_1; Y)`: the whole
    #    trajectory carries exactly the ENTRY's information and the exit can only have
    #    lost some of it. So a trajectory prefix that beats an exit-repeat prefix may be
    #    recovering what the exit THREW AWAY, which is not a loop gain and would read as
    #    one. This mode hands the coda the entry and the exit and nothing else:
    #      cell 0            : the ENTRY state z_1 = core_init(e), before any pass
    #      cells 1 .. K-2    : the EXIT (copies, so the live-cell COUNT is K, as in
    #                          exit_repeat — no pads, nothing to mask)
    #      cell K-1          : the EXIT
    #    That makes two clean one-factor pairs: `entry_exit` minus `exit_repeat` is ONE
    #    cell's content (the entry instead of the exit), and `trajectory` minus
    #    `entry_exit` is "cells 1..K-2 carry passes 2..K-1 instead of exit copies". If
    #    trajectory does not beat entry_exit, the trajectory bought nothing the entry did
    #    not already have.
    #
    #    "exit_repeat": the CONTENT control. Every cell holds the exit, exactly as "exit"
    #    does, but the per-cell pass-index embedding `E_pass` is built and added the same
    #    way "trajectory" adds it. So `trajectory` minus `exit_repeat` isolates WHAT the
    #    cells carry with the cell COUNT, the parameter count and the embedding held
    #    fixed. At init (`E_pass` is zeros) `exit_repeat` is bit-identical to `exit` at
    #    the same `prefix_k`; after a step it is not, which is why it is its own value
    #    and not a comment telling somebody to reuse "exit".
    #
    #    A trajectory model needs a coda ALLOW relation to cut its pad cells out of, so
    #    it is refused without `tg_geometry: strict` or `tg_restrict`.
    #    Record: lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md
    prefix_source: str = "exit"          # "exit" | "trajectory" | "exit_repeat"
    slot_id: int = 4                     # "<fim_pad>"; its LM-head logit is −inf (§3.1)
    token_state_dropout: float = 0.15    # Bowman word dropout on the coda input (§3.4)
    slot_mean_depth: int = 0             # 0 → cfg.mean_depth
    slot_max_depth: int = 0              # 0 → cfg.max_depth
    slot_depth_fixed: int = 0            # >0: every valid slot loops EXACTLY this many
                                         # iterations, train and eval (the k=12 panel,
                                         # 2026-09-07); 0 → the per-slot Poisson draw
    coda_sees_slots: bool = True         # A4 sets False (§7.1)
    tokens_through_core: bool = False    # A2 sets True (§7.1)
    # ── THE LOOP READS TOKENS (arm `slot-spandec-strict-tokloop`, 2026-09-13) ──────
    # False is the shipped slot loop and is BIT-IDENTICAL to the tree before this key.
    #
    # True runs the SHIPPED core stage over EVERY position — the row's tokens and its
    # slot cells in ONE sequence, the per-SAMPLE Poisson depth `_core_region` already
    # draws — under the SPAN-RESTRICTED relation `causal AND (same span OR j is a slot
    # cell)` (`MORPHTransformer._core_token_aux_kwargs`, the ONE home; the same relation
    # `tul.core_token_aux` trains its aux core under). So a token reads its own span's
    # tokens and reaches every EARLIER span only through a slot cell: the slot cells are
    # still the only thing that crosses a span boundary, at inference as in training.
    #
    # This is NOT the paid loop. The paid loop (`tokens_through_core`) runs the core
    # UNRESTRICTED over the packed row, so every token reads every earlier token directly
    # and the slot cells carry nothing anyone needs. Here the restriction is the point.
    #
    # WHY. `tul.core_token_aux` gave the shared core the token objective in TRAINING only
    # and the passes still read 0.0005 nats: the core became a 1.10-nat-better token map
    # and the slot loop through those same six blocks was unmoved
    # (lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md). The verdict there
    # was "the core is not under-trained, it is under-USED". This arm is the other half
    # of Wolfe's 2026-09-12 sentence, promoted from an auxiliary to the forward: the loop
    # READS the tokens on every pass, so a pass has something new to look at, which the
    # reread arm (`tul.reread`, a frozen K/V read of the prelude) approximated and read
    # flat at K1-K6 +0.0003.
    #
    # THE CELLS GO STRAIGHT INTO THE CODA. There is no `prefix_project` write: a cell's
    # looped state is already AT its own position when the core returns, so writing it
    # through `W_prefix` would be a second copy of a tensor that is already there. `z`
    # (the state the span decoder and the MUX grade) is `gather_valid` at the slot's
    # FIRST cell, the same seam every other arm reads.
    #
    # There is no per-slot depth and no per-pass trajectory here, so every knob that
    # needs one is REFUSED at construction rather than silently ignored: `prefix_source`,
    # the staged / per-pass / oracle / critic / grad-pass family, `slot_depth_fixed`,
    # `slot_mean_depth`, `plan_mode` at eval. Forced-depth eval goes through
    # `model.cfg.mean_depth`, exactly as it does for the plain model and the paid loop.
    # Record: lab/experiments/planned/2026-09-13-arc-loop-reads-tokens.md
    loop_reads_tokens: bool = False
    stp_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    set_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    carry: bool = False                  # arm (§3.5)
    xattn: bool = False                  # arm (§3.5)
    # ── The coda reads the thought (Wolfe 2026-09-09; the spec's §3.4 let the coda skip
    #    the loop: its token inputs were the tokens' own GLOBAL prelude states, so z was
    #    optional and every slot arm read a flat K-curve). Three construction-time knobs,
    #    each bit-identical at its default:
    # bcast: spec §3.5's own row — z_i, the previous slot's looped state (stream mean),
    #    is projected through one of `bound_span_cap` OFFSET-indexed linears (init 0) and
    #    ADDED to the coda input of every token of span i+1, offset = the token's index
    #    inside its span. The "unpack" of Thought Unpack Loop. `TULSlots.W_bcast`.
    bcast: bool = False
    # reread (2026-09-10, the slot-map levers panel's structural arm): inside `_tul_core`
    #    the compact sequence holds only slot cells, so a pass has nothing new to read —
    #    the paid loop, the one arm that earns depth, re-reads every token state every
    #    pass. With `reread` the looping slot cross-attends the FROZEN prelude token
    #    states (K/V built once before the loop) at the start of every pass; the read is
    #    part of the map the hinge measures. `reread_scope`: "span" = its own span's
    #    tokens, "causal" = its own span and every earlier one. `TULReread`, W_o zero-init
    #    so step 0 is the no-reread forward bit for bit.
    reread: bool = False
    reread_heads: int = 8
    reread_scope: str = "causal"
    # ── progressive loss (Bansal, Schwarzschild et al. 2022, "End-to-end algorithm
    #    synthesis with recurrent networks: logical extrapolation without overthinking",
    #    the Deep Thinking recipe) — arm `slot-mnext-progressive`, 2026-09-10.
    #    Their objective sums a full-trajectory loss and a PROGRESSIVE loss: a random
    #    number of passes runs with NO gradient, the rest with gradient, so the map is
    #    trained to improve ANY state it is handed and cannot settle at the identity or
    #    lean on the pass index. MORPH's slot loop already has a no-grad prefix, but it is
    #    GLOBAL (`model.bptt_depth`, the same cut for every slot in the batch, and at the
    #    shipped `bptt_depth 8` it is empty).
    #
    #    `progressive_p` is the per-SLOT probability of drawing a private prefix: a slot of
    #    realised depth T_i >= 2 draws k_i uniform in [1, T_i - 1] and runs its first k_i
    #    passes DETACHED (state in and state out), the rest with gradient. Slots that do not
    #    draw run full BPTT exactly as today. The loss is unchanged (token CE + MUX).
    #    0.0 = off: the whole mechanism is a Python-level constant that traces out and the
    #    forward is the one from before this existed (tests/test_tul_progressive.py).
    progressive_p: float = 0.0
    # ── per-pass low-rank deltas (Bae et al. 2024, "Relaxed Recursive Transformers:
    #    Effective Parameter Sharing with Layer-wise LoRA") — arm `slot-mnext-per-pass-lora`,
    #    2026-09-10. The shared core keeps its weights; pass t adds its own rank-r delta
    #    `y_t = sublayer(x) + B_t (A_t x)`, B zero-init, one pair per (core block, pass,
    #    targeted sublayer). Motivated by the per-pass gradient probe: the six passes'
    #    gradients on the shared weights are near-orthogonal (|sum_t dW_t| / sum_t |dW_t|
    #    0.52-0.60), i.e. they are asking one map to be six different things.
    #    `pass_lora_rank` 0 = OFF: nothing is built, no RNG is drawn, the forward is the one
    #    from before this existed. `pass_lora_targets` picks the sublayers — see
    #    `morph/model/mhc.py::PassLoRA` for exactly what a "target" covers and what it does
    #    not. The deltas are plain nn.Parameters on a non-Linear module, so ternary QAT,
    #    the CMS prune, the MORTAR carve and the deploy packer all walk past them.
    pass_lora_rank: int = 0
    pass_lora_targets: tuple[str, ...] = ("attn", "mlp")
    # coda_token_input: what the coda's carrier holds at TOKEN positions.
    #    "prelude" — input_norm(prelude output), the shipped §3.4 path (global context
    #                inside every token state; z redundant).
    #    "embed"   — input_norm(the prelude's own INPUT carrier: the token embedding after
    #                embed dropout, `x0` expanded to the HC streams). A token then carries
    #                only itself into the coda; context comes through attention.
    coda_token_input: str = "prelude"
    # tg_restrict_scope: where the same-span-or-slot mask (tg_allow) applies.
    #    "all"  — prelude, and coda (the shipped TG arms; the prelude is starved too).
    #    "coda" — the coda only. The prelude stays global (e_z, the slot's seed, sees the
    #             whole past); the coda reaches earlier spans ONLY through the slot cells.
    #             With coda_token_input="embed" this is Block Transformer's contract: the
    #             local decoder gets the block embedding plus the block's own tokens.
    tg_restrict_scope: str = "all"
    # ── THE STRICT GEOMETRY (arms `slot-spandec-strict*`, 2026-09-12) ─────────────
    # tg_geometry: which allow relation `tg_restrict` means.
    #    "restrict" — the shipped one: causal AND (same span OR j is ANY slot cell), in
    #                 the prelude and in the coda. BIT-IDENTICAL to before this key
    #                 existed, and the default.
    #    "strict"   — the loop is the ONLY cross-span channel. Prelude: same span, full
    #                 stop (`tg_strict_allow`). Coda: a token reads its own span plus the
    #                 PREFIX CELLS of earlier slots; a prefix cell reads itself alone. The
    #                 conv / value shift are reset at every segment (`tg_segment_ids`), the
    #                 retention carry with them, and the coda's per-layer injections at the
    #                 slot cells are zeroed so a cell carries z and nothing else.
    #
    # Why: under "restrict" the slot CELLS carry the cross-span information the loop was
    # supposed to carry. Measured on `slot-spandec-mask` at 5,000 steps, the whole slot
    # channel is worth 0.182 nats and the loop's own prefix write 0.078 — the prelude lets
    # every token and every cell read every earlier cell, so the seed reaches later spans
    # with no pass of the loop in between. Record:
    # lab/experiments/planned/2026-09-12-arc-strict-geometry.md.
    tg_geometry: str = "restrict"
    # Which prefix cells a CODA token may read under "strict".
    #    "all"  — every earlier slot's cells (the primary arm).
    #    "prev" — only the cells of the slot terminating the PREVIOUS span, so everything
    #             older must flow through the chain of loop states. A tail dump-bin token
    #             reads no cell at all (the dump bin is not a span; the same conservative
    #             gating `tg_soft_prev_span` takes).
    # Meaningless at tg_geometry="restrict" and refused there rather than ignored.
    tg_coda_prefix_reach: str = "all"
    # ── DEPTH AS REACH (arm `slot-spandec-strict-reach1`, 2026-09-12) ─────────────
    # loop_reach: how many slots BACK a slot may attend inside the loop, per pass.
    #    0 — unlimited (today, and bit-identical: no mask is built at all).
    #    w — slot k attends slots k-w .. k on every pass. The compact sequence updates all
    #        cells in parallel per pass (Jacobi), so a slot m spans back needs ceil(m/w)
    #        passes to reach k: long-range context REQUIRES depth BY CONSTRUCTION.
    # Every cross-cell route inside the core is cut to that budget, not just the window
    # branch: the compressed branch takes the same relation (`tg_comp_allow`; under
    # tg_restrict it is the DENSE slot-column form at the compact shape, so it is a real
    # route), and the CCA causal conv plus its W_v_prev value shift are reset PER CELL
    # (kernel 4 over two stages is a 6-cell route per block, ~36 per pass across six core
    # blocks — no window can express that, so it is cut rather than budgeted). Naming the
    # second change here because it cannot be separated from the first.
    # Requires tg_geometry="strict": outside it the cross-span routes OUTSIDE the loop are
    # open, so a reach limit inside the loop bounds nothing.
    # The K-curve of a reach arm is FORCED by construction — it reads depth DEPENDENCE, not
    # depth VALUE. The value reading is its depth-6 CE against the strict arm, paired.
    loop_reach: int = 0
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
    # Staged targets (arc E3, lab/experiments/planned/2026-09-04-loop-contribution-arc.md).
    # > 0 gives the loop TWO jobs in sequence: the state after iteration k
    # (= this value) is supervised toward the span it TERMINATES (target "own", the
    # memory the loop earned 0.037 nats on) for the slots whose depth reaches k, and the
    # FINAL state toward `mux_target` (the forecast) as before. The local loss is the
    # mean of the two. The carry stays LIVE (full BPTT through the loop; this is not
    # db_loop, which detaches it). 0 = off: the forward is bit-identical.
    mux_stage_own_iters: int = 0
    # ── what the MUX head reads off the Hyper-Connection carrier (F2, 2026-09-10) ──
    #    "mean" — `_readout`: the n streams are collapsed by an unweighted mean BEFORE
    #             `lm_mixer` and `final_norm`. The shipped path, and what `TULSlots.unpack`
    #             does too.
    #    "full" — `lm_mixer` and `final_norm` run PER STREAM and the n readouts are averaged
    #             afterwards, which (both the mixer and the tied head being linear) is
    #             exactly "the head applied to each stream and the logits averaged".
    #
    #    Why the knob exists: the slot-geometry audit's finding F2
    #    (`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`). On
    #    `slot-unpack-free` the loop's UPDATE survives `h.mean(dim=2)` at 0.139 of its
    #    per-stream norm while the ENTRY survives at 0.972 — the stream mean throws away 86 %
    #    of what the loop did — and `TULSlots.prefix_project`, the reader that feeds the
    #    coda, does NOT take that mean: it projects every stream separately.
    #
    #    Only defined on a Hyper-Connection carrier; a model without a stream axis refuses at
    #    construction. "mean" is bit-identical to the tree before the knob
    #    (tests/test_tul_mux_readout.py).
    mux_readout: str = "mean"
    # ── the staged own-span target at EVERY non-final pass (arm `slot-mnext-staged-all`,
    #    2026-09-10) ──
    #    True changes what `mux_stage_own_iters` MEANS: instead of one intermediate pass k,
    #    the own-span target is applied at every pass j from `mux_stage_own_iters` (which is
    #    then the FIRST supervised pass) up to T-1, for the slots whose realised depth
    #    reaches j, and the own terms are averaged. The next-span target still supervises the
    #    FINAL state alone and the loss is still `0.5 * (own + next)`, so `mux_beta` keeps
    #    its meaning and the reported `mux_local` / `mux_rel` / `mux_kl` still come from the
    #    final (forecast) term.
    #
    #    This is the toy study's winning attachment (`lab/toy_slot_loop/WRITEUP.md`,
    #    2026-09-10): on a task that NEEDS iteration, in the geometry where the slot loop is
    #    the only cross-span path, `staged` — own-span at every non-final pass, next-span at
    #    the exit — solved the chain on 5 of 5 seeds against 2/5 for exit-only and 1/5 for
    #    the same target at every pass (`mux_every_pass`, which ran on MORPH and read flat).
    #    `mux_stage_own_iters` alone expresses the narrower k=3 version of it.
    #
    #    Reuses the SAME live-carry trajectory and the SAME per-pass keep masks
    #    `mux_every_pass` collects in `_tul_core` — one trajectory, not two. TRAINING ONLY:
    #    an eval forward falls back to the single pass-k own term, which is what keeps a
    #    stage arm's eval columns (`mux_local_own_final`, `mux_local_next_final`, both read
    #    from the FINAL state) identical in shape to every other stage arm's.
    #    False = off: the forward is `mux_stage_own_iters`'s, bit-identical
    #    (tests/test_tul_mux_stage_all.py).
    mux_stage_all: bool = False
    # ── the M-next MUX on EVERY pass (arm `slot-mnext-mux-every-pass`, 2026-09-10) ──
    #    The slot loop gets gradient ONLY at its exit state: the token CE through the coda
    #    and the two prefix cells, and ONE MUX term computed on the exit state. No
    #    intermediate pass carries a loss, and the per-pass gradient probe reads a FLAT
    #    cotangent (share 0.168/0.168/0.166/0.160/0.154/0.183) with near-orthogonal per-pass
    #    updates on the shared weights (|sum_t dW_t| / sum_t |dW_t| = 0.520).
    #
    #    True puts the CONFIGURED MUX target (`mux_target`, "next" on the arm) on the state
    #    after EVERY pass of a LIVE carry — no detach anywhere — for the slots whose realised
    #    depth reaches that pass, plus the final state for every valid slot. The terms are
    #    averaged (uniform weights summing to 1), so `mux_beta` keeps its meaning, and the
    #    reported stats (`mux_local`, `mux_rel`, `mux_kl`, ...) come from the FINAL term so
    #    the sweep columns stay comparable with every earlier arm.
    #
    #    NOT `db_loop` (which detaches the carry and so trains ONE core application per
    #    term) and NOT `mux_stage_own_iters` (two fixed stages, two different targets);
    #    both are refused below. TRAINING ONLY: an eval forward computes the single
    #    final-state term exactly as the ruler does, which is what keeps
    #    `core_depth_sweep.py`'s forced-depth `mux_local` column unchanged.
    #    False = off: no trajectory is kept, no term is built, the forward is the one from
    #    before this existed (tests/test_tul_mux_every_pass.py).
    mux_every_pass: bool = False
    # ── the SPAN DECODER target (arm `slot-spandec-mask`, 2026-09-11) ──────────────
    #    The MUX head grades z against an ORDER-FREE bag: one softmax per slot against the
    #    geometric superposition of the next span's tokens (`mux_span_targets`, rho 0.9).
    #    The best z under that target is the span's weighted unigram marginal, which is why
    #    Wolfe read the arm as "we are essentially getting like 2 tokens out of it"
    #    (2026-09-11). The measured cross-span budget the slot has to carry is 0.40 nats,
    #    0.31 of it FLAT at every offset eight or more tokens into the span
    #    (`lab/experiments/failures/2026-09-11-arc-span-budget.md`); a marginal cannot carry
    #    a flat long-range component.
    #
    #    `spandec: true` builds `morph/model/tul_spandec.py::SpanDecoder` and adds
    #    `spandec_weight * mean_j CE(t_j | z, t_{<j})` over the NEXT span's tokens — a small
    #    teacher-forced causal decoder over [z, the span's tokens so far], so the gradient
    #    reaches z from EVERY token of the span instead of once per span. The token prefix
    #    is the token path the spec demands (never decode a span from one vector plus an
    #    offset). Output head = the tied LM head under `mux_detach_head`, exactly as the MUX
    #    head reads it; the decoder's input embeddings come from the same detached table.
    #    False = off: nothing is built, no RNG is drawn, the forward is the one from before
    #    this existed (tests/test_tul_spandec.py).
    spandec: bool = False
    spandec_layers: int = 2              # decoder blocks; cost is linear in this
    spandec_heads: int = 0               # 0 -> the model's n_heads
    spandec_weight: float = 1.0          # weight of the term in the total loss
    spandec_max_tokens: int = 0          # 0 -> bound_span_cap (= the data's span_cap), PER SPAN
    # ── THE DOWNSTREAM TARGET (arm `slot-spandec-strict-h3`, 2026-09-12) ──────────
    # spandec_horizon H: the decoder's target is the concatenated tokens of spans
    # s+1 .. s+H, teacher-forced, ONE causal run over H * spandec_max_tokens positions from
    # z. 1 is the shipped target and is bit-identical. Slot cells are never in the target;
    # a slot is supervised at block h only when span s+h exists AND is complete, so a slot
    # near the end of a row is masked on the blocks it does not have rather than dropped.
    # No h-dependent weight: every supervised TOKEN counts once, so a longer span carries
    # more of the mean than a shorter one — the same convention the H=1 term already has.
    # Why: the measured cross-span budget is not front-loaded (a FLAT 0.31 nats at every
    # offset eight or more tokens into a span, lab/experiments/failures/
    # 2026-09-11-arc-span-budget.md), and the H=1 decoder's own worth profile still decays
    # with offset. A target that ends at the next boundary cannot ask z for anything past
    # it.
    # Cost is linear in H: the decoder goes from 4.0 to 12.0 block-passes per token at
    # H = 3 (2 layers x 64 slots x 32 tokens x H over ~1024 real tokens per row).
    spandec_horizon: int = 1
    # ── THE PER-PASS PLANNING TARGET (arm `slot-spandec-strict-perpass`, 2026-09-12) ──
    #
    # WHAT IT IS. Every pass of the loop gets its own decoder target, and the target grows
    # by ONE span per pass: the state after pass `t` is graded on spans `s+1 .. s+t`
    # (capped at `spandec_pass_horizon_max`), through the SAME `SpanDecoder` the exit state
    # is graded by. The exit term is untouched — it stays the shipped H = 1 "next thought"
    # loss — so this adds a target, it does not replace one.
    #
    # WHY. Amendment 3 of the strict panel (2026-09-12): forcing reachability
    # (`tg_coda_prefix_reach: prev` + `loop_reach: 1`) is the first thing that made the
    # token K-curve move (K1-K6 +0.0163, K3-K6 +0.0042 at CE parity), and it did it by
    # making z hold HISTORY. Wolfe: "z has to hold the history when it should hold the
    # present next thought that needs decoding. Our objectives are still poor." This knob
    # asks for depth from the OBJECTIVE instead of from blindness: pass t can only meet its
    # target by extending the plan one span further than pass t-1 could, and nothing is
    # hidden from the coda.
    #
    # THE REDUCTION, stated because it is a choice. One CE per pass (a mean over that
    # pass's graded target tokens), then a plain mean over the passes. Passes are therefore
    # EQUALLY weighted; a token-weighted mean would make the term mostly about the deepest
    # pass, which has `pass_horizon_max` times the tokens of pass 1.
    #
    # THE MASK. A slot is graded at pass `t` only when its REALISED depth reaches `t` —
    # `tul.oracle_z`'s rule. A frozen slot's state is its final one, and grading it again
    # at every later pass would supervise the `torch.where` carry and over-weight shallow
    # slots.
    #
    # COST, arithmetic and not a guess. At `spandec_pass_tokens` 8 and
    # `spandec_pass_horizon_max` 6 the per-pass term decodes sum_{t=1..6} t*8 = 168
    # positions per slot per step, against the H = 3 exit target's 96 and the shipped H = 1
    # target's 32. In decoder block-passes per real token (2 layers x 64 slots / 1024
    # tokens): 21.0 for the per-pass term plus 4.0 for the exit term, so the arm is 35.7
    # block-passes per token against `slot-spandec-strict`'s 14.7 and `-h3`'s 22.7. A
    # depth-8 draw adds two more capped blocks (264 positions, 33.0 + 4.0). This is the
    # most expensive arm of the family and the smoke's tok/s decides whether it runs.
    spandec_per_pass: bool = False
    spandec_pass_horizon_max: int = 6    # cap on the pass-t horizon; 6 = the max useful depth
    spandec_pass_weight: float = 1.0     # weight of the per-pass term in the total loss
    # Tokens graded per span INSIDE the per-pass term — a PREFIX of each span, not the
    # whole one. 8 because the measured cross-span budget stops being front-loaded there
    # (flat 0.315 nats at offset 8+, lab/experiments/failures/2026-09-11-arc-span-budget.md)
    # and because the cost is linear in it; `tul.oracle_z_max_tokens` and
    # `tul.egrad_max_tokens` are the same precedent. The EXIT target keeps the full
    # `spandec_max_tokens` span.
    spandec_pass_tokens: int = 8
    # ── PARALLEL SPAN DECODING FROM THE CODA (arm `slot-spandec-strict-codaspan`) ──────
    #
    # WHAT IT IS. `coda_span_heads: J` builds J parallel offset heads (the `_MTPHead`
    # construction: RMSNorm + a [d, d] linear at IDENTITY init, deterministic, no RNG) that
    # read the CODA's final state at each slot's emitting position and predict the next
    # span's tokens 1..J AT ONCE, non-autoregressively, with no teacher forcing. Loss = a
    # mean CE over the valid (slot, offset) pairs through the tied head.
    #
    # WHY. Wolfe, 2026-09-12: "try parallel token decoding from the coda. Perhaps the coda
    # needing to spit out a lot of the span or all the span at once changes the behavior."
    # Every span-decoder arm to date grades z through a SEPARATE reader that the token CE
    # never uses. This one grades the coda itself — the reader the model actually ships —
    # and the only way that state can carry the next span is through what the loop wrote
    # into the prefix cell it sits on.
    #
    # WHERE IT READS (`coda_span_source`):
    #   "cell"  (default, the arm) — the slot's LAST prefix cell, `slot_index[s] +
    #           prefix_k - 1`. Under the strict geometry that cell carries the looped state
    #           and nothing else, so the heads' gradient reaches the loop's write directly.
    #           It is also the position whose own emit label is the next span's first token
    #           and which carries NO loss at `emit_weight: 0.0`, so head 1 is the first term
    #           that has ever trained it.
    #   "token" — the boundary TOKEN position instead (`boundary_token_index`), the position
    #           `emit_source="token"` generates from. STATE THE CONSEQUENCE: that token sits
    #           BEFORE its own slot's cells, so a causal coda state there has NEVER seen its
    #           own slot's z. The heads then reach the loop only through EARLIER slots'
    #           writes. It is a control, not the arm.
    #
    # COST. J readout rows per slot: J x 64 per row against the token CE's ~1024, so J = 8
    # is ~0.5x the main CE's readout and J = 32 is ~2x. The heads themselves are J x [d, d]
    # matmuls on [B, S, d] — 3.2 GFLOP at B 6, S 64, d 1024, J 8, which is noise. The
    # readout goes through `fused_linear_cross_entropy` as ONE call over [B*S*J, d], so the
    # [B*S*J, V] logits are never materialised (they would be 604 MB fp32 at B 6, S 64,
    # J 8, V 49169) and the [V, d] fp32 grad_w accumulator is paid ONCE, not J times.
    coda_span_heads: int = 0             # 0 = off: nothing is built and the forward is unchanged
    coda_span_weight: float = 1.0        # weight of the term in the total loss
    coda_span_source: str = "cell"       # "cell" (the arm) | "token" (the control above)
    # ── THE CORE-TOKEN GRADIENT AUXILIARY (arm `slot-spandec-strict-coretok`, 2026-09-12) ─
    #
    # THE FACT IT ANSWERS. In the slot loop the six shared core blocks are trained by the
    # SLOT losses alone — ~51 valid cells per row, one target each — while the token CE
    # reaches the core at about 1 % of the prelude's gradient
    # (`slot-loop-gradient-probe-readings`, 2026-09-10). The PLAIN looped model trains the
    # same six blocks on 1,024 next-token targets per row and earns 0.185 nats of depth
    # (`prelude-entry-flattens-the-loop`, 2026-09-10). Twelve slot arms read a per-pass
    # K-curve inside [-0.0001, +0.0033]. Wolfe, 2026-09-12: "train the core on the token CE
    # as well (tokens through the core for gradient only, slots still the only cross-span
    # channel at inference)".
    #
    # WHAT RUNS. TRAINING ONLY. After the shipped forward has finished, a SECOND pass over
    # the same prelude output sends EVERY position — tokens and slot cells together, the
    # paid loop's shape — through `_core_region`, the per-sample Poisson-depth core the
    # plain model and the paid loop already run, then through the coda, and charges the
    # ordinary weighted token CE on it. That CE is added to the loss as
    # `core_token_aux_weighted` and is exposed so `train.py` subtracts it and keeps
    # train/loss on the SHIPPED path's CE. Nothing about the shipped forward changes: eval,
    # the sweeps, `worth_profile`, `slot_z_optimize` and inference all run the slot loop
    # with tokens OUTSIDE the core, exactly as before.
    #
    # THE GEOMETRY OF THE AUX PATH, and it is load-bearing. Tokens in the core must NOT
    # become a cross-span channel the core can lean on, or the arm would buy its token CE
    # by re-opening the bypass `tg_geometry="strict"` exists to cut. So the aux core runs
    # under `tg_allow_mask` — causal AND (same span OR j is a slot cell) — on the window
    # branch, the same slot-column restriction on the compressed branch, and the
    # `tg_segment_ids` reset on the CCA conv and its value shift. A token therefore reads
    # its own span's tokens and reaches every EARLIER span only through a slot cell, which
    # is exactly the reachability the slot cells themselves have inside the loop. The aux
    # coda takes the shipped strict coda relation and the same zeroed slot-cell injections.
    #
    # THE DEPTH IS ITS OWN DRAW, and that is a decision, not an oversight. The slot loop
    # draws a per-SLOT Poisson depth [B, S]; `_core_region` draws a per-SAMPLE one [B].
    # There is no shared draw, and every reduction of the per-slot table to a scalar
    # distorts the distribution: the max over ~51 Poisson(6) draws capped at 8 is 8 almost
    # surely, which would train the core at a depth the plain model never sees. The point
    # of the arm is to train the core the way the plain model trains it, so the aux takes
    # the plain model's own per-sample Poisson draw. The whole aux is wrapped in an RNG
    # save/restore (the `_slot_gain_penalty` precedent), so it consumes nothing from the
    # run's stream and `loss - core_token_aux_weighted` is the off-model's loss bit for bit.
    #
    # COST, arithmetic and not a guess (row = 1,024 real tokens, L_total 1152, 64 cells,
    # ~51 valid, mean depth 6), block-passes per real token:
    #     shipped slot arm  prelude 4x1152 + core 6x51x6 + coda 4x1152   = 10.8
    #     exit span decoder 2 x 64 x 32                                  =  4.0
    #     THE AUX         core 6x1152x6 + coda 4x1152                    = 45.0
    #                                                                      -----
    #                                                                      59.8
    # against `slot-spandec-strict`'s 14.8 and the PLAIN panel's 44.0. This is the most
    # expensive arm of the family by a wide margin — roughly one extra paid-loop
    # forward+backward per step — and both its rate and its memory are named risks in
    # lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md.
    core_token_aux: bool = False         # False = off: no aux forward, graph unchanged
    core_token_aux_weight: float = 1.0   # weight of the aux CE in the total loss
    # ── THE ORACLE-Z PER-PASS TEACHER (arm `slot-spandec-strict-oracle`, 2026-09-12) ──
    #
    # THIS BREAKS A STANDING RULE AND SAYS SO. The root CLAUDE.md and the spec forbid
    # regressing onto the slot state (LCM T3/4, CoCoMix §6b, BT §4.2): a target that says
    # "be this vector" collapses the state instead of making it useful. This knob does
    # exactly that, deliberately and as a TEST: eleven arms have read a per-pass K-curve of
    # zero, and the open question is whether the passes CANNOT descend a useful objective
    # or whether nothing has ever told them what each pass is FOR. An oracle trajectory
    # that a one-step optimiser could match is the cheapest way to ask. Wolfe decides
    # whether it ever ships; nothing composes it by default.
    #
    # WHAT IT IS. Train time only, once per step, with the SPAN DECODER as the reader:
    #   z*_0 = the loop's entry state (`core_init(e)`, exactly what pass 1 receives), detached
    #   z*_t = z*_{t-1} - step_t,  step_t = oracle_z_lr * ||z*_{t-1}|| * g / ||g||   per slot
    #   g    = d/dz  CE_spandec(next span | z)   at z*_{t-1}, fp32, no outer graph
    # and the added term is
    #   oracle_z_weight * mean_t ||h_t - z*_t||^2 / d   over the slots whose depth reaches t,
    # with h_t the loop's own state after pass t. The trajectory is FULLY detached, so the
    # decoder is NOT trained by it — the decoder trains only through the shipped spandec
    # loss — and the token CE and the spandec loss are unchanged.
    oracle_z: bool = False
    oracle_z_steps: int = 6              # T; clamped to the batch's realised depth
    oracle_z_lr: float = 0.1             # one step moves a slot's state by lr * ||z||
    oracle_z_weight: float = 1.0         # weight of the MSE term in the total loss
    # Tokens of the next span the ORACLE's own readout is graded on. NOT `spandec_max_tokens`,
    # and the reason is cost: the oracle runs T forward+backward passes of a [B, S, J, V]
    # readout per training step, and one readout is 38.7 GFLOP per token of J at B 6, S 64,
    # C 1024, V 49169. At J = 32 and T = 6 that is ~22 TFLOP against a ~50 TFLOP step — the
    # arm would miss the queue's 8,086 tok/s rate floor and never run. 8 is the
    # `egrad_max_tokens` precedent and the offset at which the measured cross-span budget
    # stops being front-loaded (lab/experiments/failures/2026-09-11-arc-span-budget.md).
    # The oracle is a TEACHER for the trajectory, not the shipped target: the target the
    # decoder is trained on stays the full `spandec_max_tokens` span.
    oracle_z_max_tokens: int = 8
    # ── the slot chain (arm `slot-spandec-chain-mask`, 2026-09-11) ─────────────────
    #    A DIRECT, learned path along the slot axis, on top of the core's own causal
    #    attention over the compact slot sequence. At every pass `t` of the loop, slot `k`
    #    receives `W_chain(h_t[k-1])` — the previous slot's state at that pass, which IS its
    #    EXIT state for every neighbour whose realised depth is already spent (a frozen slot
    #    carries its final state through the where-carry). Slot 0 receives one learned
    #    vector; both `W_chain` and that vector are ZERO-init, so step 0 is the ruler's
    #    forward bit for bit.
    #
    #    WHY NOT the literal "slot k's seed gets slot k-1's exit". That needs slot k-1's
    #    loop to finish before slot k's starts, i.e. 64 sequential loops over the compact
    #    sequence instead of one masked loop — the `runtime-invariants` §6b masked-update
    #    contract exists precisely because the slots loop TOGETHER. The wavefront is the
    #    causal, one-loop form of the same recurrence and its reach is T slots per row of
    #    direct path (attention supplies the rest).
    #
    #    `slot_chain_detach` cuts the gradient on the chain input. Default FALSE: the
    #    recurrence depth is bounded by the pass count (<= max_depth), not by the 64 slots,
    #    so there is no deep-recurrence argument for detaching, and detaching would make the
    #    chain a feature rather than a path an earlier span's loss can shape.
    slot_chain: bool = False
    slot_chain_detach: bool = False
    # ── gradient-conditioned passes (arm `slot-mnext-gradpass`, 2026-09-10) ────────
    #    Marino, Yue & Mandt 2018 "Iterative Amortized Inference" and Greff et al. 2019
    #    "IODINE": an iterative inference network is handed, at every step, the GRADIENT of
    #    its own objective with respect to the state it is refining, so the loop can learn
    #    to be an optimiser instead of re-running one fixed update rule.
    #
    #    True computes, before each pass t of the slot loop, a LOCAL and CAUSAL target loss
    #    on the current slot state — the slot's OWN span through the tied head,
    #    `_tul_mux_loss(h_t, target="own")`, which reads only tokens the slot already sits
    #    after — takes `g_t = dL_own/dh_t` with `torch.autograd.grad(create_graph=False)`,
    #    normalises it per slot and adds `W_g(g_t)` to the state the core step receives.
    #    The gradient is a FEATURE: it is DETACHED from the outer graph (IODINE's own
    #    choice, their Eq. 8 / §3.1), so the token CE never differentiates through the
    #    inner backward. `create_graph=True` (a second-order, learned-optimiser objective)
    #    is the alternative and is NOT built here — it would retain one inner graph per
    #    pass and put a Hessian-vector product in every training step.
    #
    #    `W_g` is ZERO-INITIALISED (`TULGradPass`), so pass 0 of step 0 is the ruler's
    #    forward bit for bit and the feature only starts acting once the token CE has moved
    #    W_g off zero. It draws no RNG at construction, so an arm with the knob on holds the
    #    same weights as the ruler everywhere else.
    #
    #    The local target must DIFFER from the exit target (the toy study,
    #    lab/toy_slot_loop/WRITEUP.md): the own span is reachable in about one pass, the
    #    exit keeps the ordinary M-next forecast MUX, and that loss is UNCHANGED — this
    #    knob adds an input, never a loss term.
    #    False = off: nothing is built, no gradient is taken, the forward is the one from
    #    before this existed (tests/test_tul_grad_pass.py).
    grad_pass: bool = False
    # Constant multiplier on the NORMALISED gradient before `W_g`. It sets the scale W_g
    # sees, so it acts like a per-feature learning-rate on an otherwise scale-free input.
    grad_pass_scale: float = 0.1
    # "rms": divide each slot's gradient by its own RMS over the carrier channel, so the
    # feature carries the gradient's DIRECTION and the loop cannot read the raw magnitude
    # (which falls as training proceeds). "none": feed the raw gradient.
    grad_pass_norm: str = "rms"
    # ── WHICH energy the pass is handed the gradient OF (arms `slot-spandec-egrad-recon`
    #    and `slot-spandec-egrad-disc`, 2026-09-12; morph/model/tul_egrad.py) ────────
    #    "own_mux" — `_tul_mux_loss(target="own")`, the slot's own span against an
    #                ORDER-FREE geometric bag through the tied head. The shipped energy and
    #                the only one before this: `slot-mnext-gradpass` descended it 0.366 nats
    #                in the FIRST pass and then sat flat for seven more, which is what a
    #                target already present in the entry state looks like.
    #    "recon"   — a SECOND SpanDecoder (its own parameters) that reconstructs the slot's
    #                OWN span from z, teacher-forced on that span's tokens. Conditional,
    #                ordered and per token, so it is not a marginal one pass can reach.
    #    "disc"    — `-s_phi(z, ctx)`, a 2-layer scalar critic trained with BCE against the
    #                MEASURED outcome (the coda's mean CE over the slot's next span, below
    #                the batch median => 1) on a detached z, with shuffled-context
    #                negatives. Never a regression onto z (spec: never regress the latent).
    #
    #    Both new energies train ONLY their own parameters: their loss is computed on a
    #    stop-gradient copy of z, and the ONLY route from an energy into the loop is the
    #    detached feature through `W_g`. "own_mux" builds nothing and is bit-identical to
    #    the tree before this existed (tests/test_tul_egrad.py).
    grad_pass_energy: str = "own_mux"
    egrad_weight: float = 1.0            # weight of the energy module's OWN training loss
    egrad_layers: int = 2                # "recon": decoder blocks
    egrad_heads: int = 0                 # "recon": 0 -> the model's n_heads
    # "recon": tokens of the own span the energy decoder reconstructs. NOT bound_span_cap by
    # default, unlike `spandec_max_tokens`, and the reason is cost: the energy is read at
    # EVERY pass, so its [B, S, J, V] head readout is paid T times per step while the
    # spandec target's is paid once. At B 6, S 64, C 1024, V 49169 one readout is
    # 38.7 GFLOP per token of J; the feature plus the critic's own training come to roughly
    # 36 readout-equivalents per step, i.e. ~1.4 TFLOP x J against a ~50 TFLOP step. J = 8
    # is ~22 % and J = 32 would be ~90 %. 8 is also where the measured cross-span budget
    # stops being front-loaded (lab/experiments/failures/2026-09-11-arc-span-budget.md).
    egrad_max_tokens: int = 8
    # "recon": A*-Thought-V2 Label Forcing. The target at decoder position j becomes
    # (1 - mix) * onehot(t_j) + mix * bag(the span's tokens). CE is linear in the target, so
    # this is computed EXACTLY as a weighted sum of the tree's two chunked CE kernels — no
    # [B, S, J, V] logits, no second CE implementation.
    egrad_soft_labels: bool = False
    egrad_soft_mix: float = 0.5
    egrad_disc_hidden: int = 0           # "disc": 0 -> d_model
    # ── "critic": THE WITHIN-CONTEXT IMPROVEMENT CRITIC (arm
    #    `slot-spandec-strict-critic`, 2026-09-12; morph/model/tul_egrad.py::CriticEnergy)
    #
    #    WHY A THIRD SCALAR ENERGY. `disc`'s label is "is this slot's next span below the
    #    BATCH MEDIAN CE?". Most of that is how predictable the next span happens to be — a
    #    property of the text — which is why it needs shuffled-context negatives at all.
    #    Wolfe, 2026-09-12: "a within-context critic that scores whether the state after
    #    pass t beats the state after pass t-1 on the same coda loss."
    #
    #    THE LABEL, and it is the only new machinery. Two candidate slot states are each
    #    written into that slot's prefix cells through the SAME `prefix_project`, the REAL
    #    coda is replayed, and the next span's mean token CE is measured per slot. The
    #    critic is trained by a pairwise logistic loss on the sign of the CE difference,
    #    WEIGHTED by its magnitude, so a pair the coda cannot tell apart teaches nothing.
    #    Everything the two candidates share cancels, which is what "within context" buys.
    #    Candidates: the trajectory pair (h_{t-1}, h_t) at a per-slot sampled t <= the
    #    realised depth, and the perturbation pair (h_t, h_t + critic_eps * rms(h_t) * n).
    #
    #    EVERY REPLAY IS `no_grad`. The label is a measurement, not a loss: no gradient
    #    reaches the loop, the coda, `W_prefix` or the decoder from it, and the whole label
    #    computation sits inside an RNG save/restore so it consumes nothing from the run's
    #    stream. `tests/test_tul_critic.py` proves both by autograd, not by reading this.
    #
    #    THE CONFOUND, stated where the knob is. Under `tg_coda_prefix_reach: all` a token
    #    reads EVERY earlier slot's cells, so a replay that substitutes every slot at once
    #    attributes span s+1's CE change to slot s while every earlier slot's substitution
    #    also moved it. `critic_replay_groups: G` splits the substitution into G replays,
    #    each touching slots s = g (mod G), so the nearest confounder sits G spans back —
    #    at G times the replay cost. G = 1 is the cheap default and ACCEPTS the noise.
    #
    #    COST, arithmetic. THREE candidate states (h_{t-1}, h_t, h_t + noise) -> 3G coda
    #    forwards per step, each 4 x 1152 block-passes, no backward: 4.5G block-passes per
    #    real token against the strict ruler's 14.8, plus one no-grad per-token readout of
    #    the coda per replay (~0.7 TFLOP each against a ~50 TFLOP step). `critic_every: k`
    #    computes the label every k-th step instead — the critic then trains on k times
    #    fewer batches, and the energy is read at every pass regardless.
    critic_weight: float = 1.0           # weight of the critic's OWN training loss
    critic_every: int = 1                # compute the label every k-th step (>= 1)
    critic_eps: float = 0.1              # perturbation size, as a fraction of rms(h_t)
    critic_replay_groups: int = 1        # G above: 1 = perturb every slot in one replay
    # ── a BOUNDED per-pass residual (LRT: no residual penalty and the state drifts;
    #    lambda 0.01 best, lambda 1.0 collapses the loop to its entry) ───────────────
    #    lambda * mean_t ||h_{t+1} - h_t||^2 / ||h_t||^2 over the loop's GRADIENT passes and
    #    the slots active at each, added to the loss as `pass_res_weighted`. DISTINCT from
    #    `model.core_fixed_point_lambda`, which charges the SAME ratio at each slot's LAST
    #    pass only: this one charges every pass, so it bounds the trajectory rather than
    #    pinning its end. Both may be on; the ruler's fixed-point value is unchanged.
    #    0.0 = off: no term is built and the graph is the one from before this existed.
    pass_residual_lambda: float = 0.0
    # ── REFUSED, and kept as a knob so the refusal is discoverable ─────────────────
    #    LRT reports -3.2 points for a seed injected at init only, so "re-inject the seed
    #    every pass" is a real lever elsewhere. It is ALREADY WHAT MORPH DOES. `_tul_core`
    #    binds `_e_arg = e` — the prelude's output at the slot position, which is the slot
    #    seed (`E_slot + W_sent . embed(t_last)`) after the prelude has run over it — and
    #    hands it to EVERY pass: `_apply_core_step` opens with
    #    `self.injection(h_in, e_in)`, a DiagonalInjection of that same `e`, and then adds
    #    the per-core-layer x0/bigram terms, which are gathered from the slot positions too.
    #    Building a second additive copy of the seed would be a duplicate path with no
    #    measurement able to separate it from a change in the injection's gain. So this
    #    RAISES at construction rather than silently shipping the duplicate.
    reinject_seed_every_pass: bool = False
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
        if self.prefix_source not in ("exit", "trajectory", "exit_repeat", "entry_exit"):
            raise ValueError(
                f"tul.prefix_source must be 'exit', 'trajectory', 'exit_repeat' or "
                f"'entry_exit', got {self.prefix_source!r}")
        if self.prefix_source != "exit":
            if self.tokens_through_core:
                raise NotImplementedError(
                    f"tul.prefix_source={self.prefix_source!r} with tul.tokens_through_core: "
                    "the paid loop writes no prefix cell at all (TULSlots has no W_prefix "
                    "there), so there is nothing for a per-cell source to select. Raises "
                    "rather than silently writing the exit everywhere.")
            if self.loop_reads_tokens:
                raise NotImplementedError(
                    f"tul.prefix_source={self.prefix_source!r} with tul.loop_reads_tokens: "
                    "that mode has no prefix write either — a cell's looped state is "
                    "already AT its own position when the core returns — and no per-slot "
                    "trajectory (`_core_region` draws ONE depth per sample). Raises rather "
                    "than picking a behaviour neither arm asked for.")
            if self.db_loop:
                raise NotImplementedError(
                    f"tul.prefix_source={self.prefix_source!r} with tul.db_loop is not "
                    "defined: db_loop DETACHES the carry, so 'the state after pass k' is "
                    "not on one graph with the exit and the per-cell gradient edges this "
                    "arm exists to create would not exist.")
        if self.prefix_source in ("trajectory", "entry_exit") and self.prefix_k < 2:
            raise ValueError(
                f"tul.prefix_source={self.prefix_source!r} needs tul.prefix_k >= 2: at "
                f"k = 1 the only cell is the exit cell and the mode is exactly 'exit'.")
        if self.prefix_source == "trajectory":
            if self.tg_geometry != "strict" and not self.tg_restrict:
                raise NotImplementedError(
                    "tul.prefix_source='trajectory' needs a coda ALLOW relation "
                    "(tul.tg_geometry='strict' or tul.tg_restrict=true): a slot whose "
                    "realised depth does not reach cell k leaves that cell a PAD, and the "
                    "pad is cut out of the coda's key set by narrowing that relation. With "
                    "no relation to narrow, every later token would attend a zero cell and "
                    "read the pad pattern as content. Raises rather than building an "
                    "unmasked pad.")
        if self.loop_reads_tokens:
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.loop_reads_tokens with tul.tokens_through_core is two names for "
                    "one forward with two different attention relations. The paid loop "
                    "runs the core UNRESTRICTED over the packed row; this mode runs it "
                    "under `causal AND (same span OR a slot cell)`, which is the whole "
                    "arm. Turn the paid loop off.")
            if self.core_token_aux:
                raise NotImplementedError(
                    "tul.loop_reads_tokens with tul.core_token_aux is the SAME forward "
                    "twice: the aux exists to send the tokens through the core in TRAINING "
                    "only, and this mode already sends them through in the shipped path.")
            if not (self.coda_sees_slots and self.coda_token_cut == 0):
                raise NotImplementedError(
                    "tul.loop_reads_tokens needs the FULL-AXIS coda (coda_sees_slots=true, "
                    "coda_token_cut=0): the cells continue into the coda AS POSITIONS, and "
                    "arm A4 / arm CW run the coda on a gathered subset those positions are "
                    "not in.")
            if self.detach_z:
                raise NotImplementedError(
                    "tul.loop_reads_tokens with tul.detach_z is not defined: detach_z cuts "
                    "the token CE's edge into the loop at the prefix WRITE, and this mode "
                    "has no write — the coda reads the looped carrier itself.")
            if self.bcast:
                raise NotImplementedError(
                    "tul.loop_reads_tokens with tul.bcast is not defined: the unpack adds "
                    "z to the coda input of the NEXT span's tokens, and those token "
                    "positions have already been through the core beside the cell it would "
                    "read. Not specified, so this raises.")
            _lrt_refuse = [
                n for n in ("mux_every_pass", "mux_stage_all", "oracle_z",
                            "spandec_per_pass", "grad_pass", "slot_chain", "reread",
                            "progressive_p", "mux_stage_own_iters", "slot_depth_fixed",
                            "slot_mean_depth", "pass_lora_rank", "loop_reach",
                            "pass_residual_lambda", "db_loop")
                if getattr(self, n)]
            if _lrt_refuse:
                raise NotImplementedError(
                    f"tul.loop_reads_tokens with {sorted(_lrt_refuse)}: every one of those "
                    "needs the SLOT LOOP's per-slot depth table or its per-pass trajectory, "
                    "and this mode runs `_core_region`'s per-SAMPLE Poisson draw with no "
                    "trajectory returned. Raises rather than running the arm with the knob "
                    "silently inert.")
            if self.gate is not None:
                raise NotImplementedError(
                    "tul.gate with tul.loop_reads_tokens: §4 reads a span length off the "
                    "core's PER-SLOT per-iteration trajectory and this mode returns none "
                    "(the tokens_through_core precedent).")
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
        if self.mux_stage_own_iters < 0:
            raise ValueError(
                f"tul.mux_stage_own_iters must be >= 0, got {self.mux_stage_own_iters}")
        if self.mux_stage_own_iters > 0:
            if self.mux_beta <= 0.0:
                raise ValueError("tul.mux_stage_own_iters needs tul.mux_beta > 0 "
                                 "(the staged targets ARE the local loss)")
            if self.db_loop:
                raise ValueError("tul.mux_stage_own_iters with tul.db_loop is not defined: "
                                 "the stage keeps the carry live, db_loop detaches it")
            if self.tokens_through_core:
                raise ValueError("tul.mux_stage_own_iters is a slot-loop lever "
                                 "(tokens_through_core must be false)")
        if self.mux_stage_all:
            # `mux_stage_all` is a MODIFIER of `mux_stage_own_iters`, so every refusal that
            # knob already carries (db_loop, tokens_through_core, mux_beta <= 0) is refused
            # for this one by the block above — with k > 0 required here, that block always
            # runs first. Only the two conditions it cannot see are stated again.
            if self.mux_stage_own_iters <= 0:
                raise ValueError(
                    "tul.mux_stage_all needs tul.mux_stage_own_iters > 0: with the knob on "
                    "that value is the FIRST pass the own-span target is applied at, and 0 "
                    "would name no pass at all.")
            if self.mux_every_pass:
                raise ValueError(
                    "tul.mux_stage_all with tul.mux_every_pass is not defined: both put a "
                    "term on every pass of the same trajectory, one toward the own span and "
                    "one toward `mux_target`, and averaging them would make the reported "
                    "mux_local a mixture of two objectives.")
        if self.mux_readout not in ("mean", "full"):
            raise ValueError(
                f"tul.mux_readout must be 'mean' or 'full', got {self.mux_readout!r}. "
                "'full' needs a Hyper-Connection carrier and is refused at construction on "
                "a model without a stream axis.")
        if self.mux_target not in ("own", "next"):
            raise ValueError(
                f"tul.mux_target must be 'own' or 'next', got {self.mux_target!r}")
        if self.mux_every_pass:
            if self.mux_beta <= 0.0:
                raise ValueError("tul.mux_every_pass needs tul.mux_beta > 0 (the per-pass "
                                 "terms ARE the local loss)")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.mux_every_pass is a SLOT-LOOP lever (_tul_core): it supervises the "
                    "state after each pass of the per-slot depth loop, and the paid loop "
                    "(tokens_through_core) has no per-slot trajectory. Raises rather than "
                    "silently running the token core with the knob ignored.")
            if self.db_loop:
                raise ValueError(
                    "tul.mux_every_pass with tul.db_loop is not defined: db_loop already "
                    "supervises picked iterations and DETACHES the carry, which is the "
                    "opposite of this knob's live-carry contract.")
            if self.mux_stage_own_iters > 0:
                raise ValueError(
                    "tul.mux_every_pass with tul.mux_stage_own_iters is not defined: the "
                    "stage supervises ONE intermediate state toward a DIFFERENT target "
                    "('own'), and averaging that into a per-pass ladder would make the "
                    "reported mux_local a mixture of two objectives.")
            if self.cond_layers > 0:
                raise NotImplementedError(
                    "tul.mux_every_pass with tul.cond_layers is not defined: the think-once "
                    "stack runs ONCE over the FINAL slot state, so the final term would be "
                    "read through it and every per-pass term would not — a silent mixture "
                    "of two readouts.")
        if self.spandec:
            if self.spandec_layers < 1:
                raise ValueError(
                    f"tul.spandec_layers must be >= 1, got {self.spandec_layers}")
            if self.spandec_weight <= 0.0:
                raise ValueError(
                    "tul.spandec needs tul.spandec_weight > 0: at 0 the decoder is built, "
                    "trained by nothing and read by nothing, and the arm is its ruler under "
                    f"another name (got {self.spandec_weight})")
            if self.spandec_horizon < 1:
                raise ValueError(
                    f"tul.spandec_horizon must be >= 1 (1 = the next span), got "
                    f"{self.spandec_horizon}")
            if self.spandec_heads < 0:
                raise ValueError(
                    f"tul.spandec_heads must be >= 0 (0 = the model's), got "
                    f"{self.spandec_heads}")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.spandec is a SLOT-LOOP lever: it grades the per-slot exit state z "
                    "by decoding the next span from it, and the paid loop "
                    "(tokens_through_core) has no per-slot state. Raises rather than "
                    "silently running the token core with the knob ignored.")
            if self.detach_z:
                raise ValueError(
                    "tul.spandec with tul.detach_z is not defined: detach_z exists so the "
                    "loop learns from the local loss ALONE, and the span decoder IS a local "
                    "loss — the combination would say nothing about either.")
            if self.spandec_per_pass:
                if self.spandec_pass_horizon_max < 1:
                    raise ValueError(
                        f"tul.spandec_pass_horizon_max must be >= 1, got "
                        f"{self.spandec_pass_horizon_max}")
                if self.spandec_pass_tokens < 2:
                    raise ValueError(
                        f"tul.spandec_pass_tokens must be >= 2 (position 0 carries z, so a "
                        f"budget of 1 grades one token and no conditional), got "
                        f"{self.spandec_pass_tokens}")
                if self.spandec_pass_weight <= 0.0:
                    raise ValueError(
                        "tul.spandec_per_pass needs tul.spandec_pass_weight > 0: at 0 the "
                        "per-pass targets are built, cost their full readout and train "
                        f"nothing (got {self.spandec_pass_weight})")
                if self.spandec_horizon > 1:
                    raise NotImplementedError(
                        "tul.spandec_per_pass with tul.spandec_horizon > 1 is a two-factor "
                        "arm and is not defined: the per-pass term already grades the "
                        "downstream spans (pass t is graded on s+1 .. s+t), and the EXIT "
                        "term is supposed to stay the shipped 'next thought' at H = 1 so "
                        "the exit's meaning does not change with it. Pick one.")
                if self.oracle_z:
                    raise NotImplementedError(
                        "tul.spandec_per_pass with tul.oracle_z is not defined. Both put a "
                        "target on the SAME per-pass trajectory — the oracle regresses h_t "
                        "onto a detached descent path in STATE space, the per-pass term "
                        "grades h_t through the decoder in TOKEN space — and the oracle's "
                        "teacher is computed FROM the decoder that the per-pass term is "
                        "simultaneously training, so the teacher moves under the student in "
                        "a way neither term's design accounts for. There is no reading of "
                        "the conjunction, so it raises instead of running.")
                if self.db_loop:
                    raise NotImplementedError(
                        "tul.spandec_per_pass with tul.db_loop: the db carry is DETACHED at "
                        "every iteration, so a per-pass target would train pass t's single "
                        "application and never the chain of passes — which is the whole "
                        "claim of this arm.")
        elif self.spandec_per_pass:
            raise ValueError(
                "tul.spandec_per_pass requires tul.spandec: the per-pass targets are graded "
                "by the SAME SpanDecoder the exit state is graded by, and without "
                "tul.spandec there is no decoder to grade them with.")
        elif (self.spandec_layers != 2 or self.spandec_weight != 1.0
                or self.spandec_heads != 0 or self.spandec_max_tokens != 0
                or self.spandec_horizon != 1 or self.spandec_pass_horizon_max != 6
                or self.spandec_pass_weight != 1.0 or self.spandec_pass_tokens != 8):
            raise ValueError(
                "tul.spandec_* set with tul.spandec=false: the decoder is not built, so the "
                "knobs would be silently ignored. Set tul.spandec: true or drop them.")
        if self.coda_span_source not in ("cell", "token"):
            raise ValueError(
                f"tul.coda_span_source must be 'cell' or 'token', got "
                f"{self.coda_span_source!r}")
        if self.coda_span_heads < 0:
            raise ValueError(
                f"tul.coda_span_heads must be >= 0 (0 = off), got {self.coda_span_heads}")
        if self.coda_span_heads > 0:
            if self.coda_span_weight <= 0.0:
                raise ValueError(
                    "tul.coda_span_heads needs tul.coda_span_weight > 0: at 0 the heads are "
                    "built, read the coda every step and train nothing (got "
                    f"{self.coda_span_weight})")
            if self.coda_span_heads > self.bound_span_cap:
                raise ValueError(
                    f"tul.coda_span_heads {self.coda_span_heads} exceeds tul.bound_span_cap "
                    f"{self.bound_span_cap}: the packer caps a span at that many tokens, so "
                    f"every head past it would have a target that is NEVER valid and would "
                    f"train on nothing while costing its readout.")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.coda_span_heads is a SLOT-LOOP lever: it reads the coda at a slot's "
                    "prefix cell, and the paid loop (tokens_through_core) writes no prefix "
                    "cell (TULSlots has no W_prefix there). Raises rather than silently "
                    "grading a position nothing wrote.")
            if not (self.coda_sees_slots and self.coda_token_cut == 0):
                raise NotImplementedError(
                    "tul.coda_span_heads needs the FULL-AXIS coda (coda_sees_slots=true, "
                    "coda_token_cut=0): the heads index the coda readout by "
                    "`slot_index + prefix_k - 1` on the packed axis, and arm A4 / arm CW run "
                    "the coda on a GATHERED subset whose index space that position does not "
                    "live in.")
            if self.detach_z:
                raise ValueError(
                    "tul.coda_span_heads with tul.detach_z is not defined: the arm's claim "
                    "is that the heads' gradient reaches the loop's write through the prefix "
                    "cell, and detach_z cuts exactly that edge.")
        elif self.coda_span_weight != 1.0 or self.coda_span_source != "cell":
            raise ValueError(
                "tul.coda_span_* set with tul.coda_span_heads=0: no head is built, so the "
                "knobs would be silently ignored. Set tul.coda_span_heads > 0 or drop them.")
        if self.core_token_aux:
            if self.core_token_aux_weight <= 0.0:
                raise ValueError(
                    "tul.core_token_aux needs tul.core_token_aux_weight > 0: at 0 the arm "
                    "pays a whole extra core+coda forward and backward every step and "
                    f"trains nothing with it (got {self.core_token_aux_weight})")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.core_token_aux with tul.tokens_through_core is the SAME forward "
                    "twice: the paid loop ALREADY sends every token position through the "
                    "per-sample core, so the aux would add a duplicate copy of the term it "
                    "exists to supply. Turn the paid loop off — the arm's whole claim is "
                    "that the tokens reach the core in TRAINING only.")
            if not (self.coda_sees_slots and self.coda_token_cut == 0):
                raise NotImplementedError(
                    "tul.core_token_aux needs the FULL-AXIS coda (coda_sees_slots=true, "
                    "coda_token_cut=0): the aux replays `_back_region` over the packed axis "
                    "and scores it with `_tul_group_losses` against the row's own labels, "
                    "and arm A4 / arm CW run the coda on a GATHERED subset with a different "
                    "index space and a different label vector.")
            if self.detach_z:
                raise ValueError(
                    "tul.core_token_aux with tul.detach_z is not defined: detach_z's claim "
                    "is that the loop learns from its local loss alone, and the aux puts "
                    "the token CE back onto the shared core weights the loop uses. The two "
                    "answer the same question in opposite directions.")
        elif self.core_token_aux_weight != 1.0:
            raise ValueError(
                "tul.core_token_aux_weight set with tul.core_token_aux=false: no aux "
                "forward runs, so the knob would be silently ignored. Set "
                "tul.core_token_aux: true or drop it.")
        if self.slot_chain:
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.slot_chain is a SLOT-LOOP lever (_tul_core): it carries the "
                    "previous SLOT's looped state into the next one, and the paid loop has "
                    "no compact slot sequence to chain along.")
            if self.db_loop:
                raise ValueError(
                    "tul.slot_chain with tul.db_loop is not defined: db_loop detaches the "
                    "carry, so the state the chain would forward is not part of any "
                    "trajectory the loss can shape.")
        elif self.slot_chain_detach:
            raise ValueError(
                "tul.slot_chain_detach set with tul.slot_chain=false: there is no chain to "
                "detach.")
        if self.grad_pass_norm not in ("rms", "none"):
            raise ValueError(
                f"tul.grad_pass_norm must be 'rms' or 'none', got {self.grad_pass_norm!r}")
        if self.grad_pass_scale < 0.0:
            raise ValueError(
                f"tul.grad_pass_scale must be >= 0, got {self.grad_pass_scale}")
        if self.grad_pass:
            if self.grad_pass_scale <= 0.0:
                raise ValueError(
                    "tul.grad_pass needs tul.grad_pass_scale > 0: at 0 the feature is "
                    "identically zero and the arm is the ruler under another name.")
            if self.grad_pass_energy == "coda_exact":
                # REFUSED, and kept as a NAMED value so the refusal is discoverable — the
                # `reinject_seed_every_pass` precedent. The idea is right and this tree
                # cannot pay for it, so the missing pieces are listed rather than the
                # option quietly omitted.
                raise NotImplementedError(
                    "tul.grad_pass_energy='coda_exact' (the energy IS the next-span coda "
                    "CE, replayed at every pass) is REFUSED. Four things are missing, and "
                    "none of them is a line of plumbing:\n"
                    "  1. THE CONTEXT IS NOT THERE YET. The energy is read inside "
                    "`_tul_core`; the coda's inputs (`base`, the token-state dropout's "
                    "`keep` mask, the coda allow relation, `prefix_project`'s write) are "
                    "built AFTER the loop returns. The critic's replay reuses them because "
                    "it runs at the end of the forward; an energy cannot.\n"
                    "  2. THE REPLAYS HAVE OPPOSITE GRADIENT RULES. `critic`'s replay is "
                    "no_grad BY CONTRACT — a label that reached the loop would make the "
                    "critic a teacher. `coda_exact` needs the same replay differentiable "
                    "with respect to the candidate state. One helper cannot hold both "
                    "contracts without a mode flag whose two branches share no test.\n"
                    "  3. IT MOVES AN RNG DRAW. The coda's token-state dropout is drawn "
                    "after the core; an energy read at every pass needs it drawn before, "
                    "which changes the SHIPPED forward on every arm.\n"
                    "  4. THE COST. T coda forward+backward passes per step is "
                    "8 x 4 x 1152 = 36,864 block-passes against the model's own 11,052 — "
                    "3.3x the model, before the slot loop's own cost. The arm would miss "
                    "any rate floor before it measured anything.\n"
                    "Use 'critic', which asks the same question through a learned scorer "
                    "at 3 no-grad coda forwards per step "
                    "(lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md).")
            if self.grad_pass_energy not in ("own_mux", "recon", "disc", "critic"):
                raise ValueError(
                    "tul.grad_pass_energy must be 'own_mux', 'recon', 'disc' or 'critic', "
                    f"got {self.grad_pass_energy!r}")
            if self.grad_pass_energy == "own_mux" and self.mux_beta <= 0.0:
                # Scoped to the MUX energy ON PURPOSE. That energy IS the MUX head's loss,
                # so at beta 0 the head is never trained and the gradient fed to the loop is
                # the gradient of an objective nothing else optimises. The `recon` and
                # `disc` energies own their own parameters and their own training loss
                # (morph/model/tul_egrad.py), so they are well defined at mux_beta 0 — which
                # is exactly the setting `tul_slot_spandec_mask` runs, because the span
                # decoder REPLACES the MUX rather than supplementing it.
                raise ValueError(
                    "tul.grad_pass with grad_pass_energy='own_mux' needs tul.mux_beta > 0: "
                    "the own-span loss it differentiates is the MUX head's loss, and at "
                    "beta 0 the head is never trained, so the gradient fed to the loop is "
                    "the gradient of an objective nothing else optimises. Set "
                    "grad_pass_energy to 'recon' or 'disc' (which own their own scorer) or "
                    "raise mux_beta.")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.grad_pass is a SLOT-LOOP lever (_tul_core): it takes the gradient "
                    "of a SLOT's own-span loss with respect to that slot's state before "
                    "each pass, and the paid loop (tokens_through_core) has no per-slot "
                    "state to differentiate. Raises rather than silently running the token "
                    "core with the knob ignored.")
            if self.db_loop:
                raise ValueError(
                    "tul.grad_pass with tul.db_loop is not defined: db_loop detaches the "
                    "carry, so 'the state the loop is refining' is a different object at "
                    "every iteration and the feature would condition on a gradient of a "
                    "trajectory that no longer exists.")
            if self.grad_pass_energy == "critic":
                if self.critic_weight <= 0.0:
                    raise ValueError(
                        "tul.grad_pass_energy='critic' needs tul.critic_weight > 0: at 0 "
                        "the critic pays its replays every step and trains nothing, so the "
                        f"energy it hands the loop stays at its init (got {self.critic_weight})")
                if self.critic_every < 1:
                    raise ValueError(
                        f"tul.critic_every must be >= 1 (1 = every step), got "
                        f"{self.critic_every}")
                if self.critic_replay_groups < 1:
                    raise ValueError(
                        f"tul.critic_replay_groups must be >= 1 (1 = one replay per "
                        f"candidate), got {self.critic_replay_groups}")
                if self.critic_eps <= 0.0:
                    raise ValueError(
                        "tul.critic_eps must be > 0: at 0 the perturbation pair is two "
                        f"copies of the same state and teaches nothing (got {self.critic_eps})")
                if self.oracle_z:
                    raise NotImplementedError(
                        "tul.grad_pass_energy='critic' with tul.oracle_z is not defined. "
                        "The oracle REGRESSES every pass onto a descent trajectory of the "
                        "span decoder's loss; the critic CONDITIONS every pass on the "
                        "gradient of a scorer fitted to the coda's own CE. Both write a "
                        "per-pass target onto the same trajectory from two different "
                        "teachers, and no reading could say which one moved the K-curve.")
                if not (self.coda_sees_slots and self.coda_token_cut == 0):
                    raise NotImplementedError(
                        "tul.grad_pass_energy='critic' needs the FULL-AXIS coda "
                        "(coda_sees_slots=true, coda_token_cut=0): its label replays "
                        "`_back_region` over the packed axis and indexes the result by "
                        "`layout.bag_id`, and arm A4 / arm CW run the coda on a GATHERED "
                        "subset whose index space `slot_outcome_labels` does not re-derive.")
                if self.detach_z:
                    raise ValueError(
                        "tul.grad_pass_energy='critic' with tul.detach_z is not defined: "
                        "the label is measured by writing a candidate through "
                        "`prefix_project` and replaying the coda, and detach_z makes the "
                        "coda's reading of z carry no training signal at all — the critic "
                        "would be scoring a channel the run has stopped using.")
        elif (self.critic_weight != 1.0 or self.critic_every != 1
              or self.critic_eps != 0.1 or self.critic_replay_groups != 1):
            raise ValueError(
                "tul.critic_* set without tul.grad_pass + grad_pass_energy='critic': no "
                "critic is built, so the knobs would be silently ignored.")
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
        for name in ("carry", "xattn"):
            if bool(getattr(self, name)):
                raise NotImplementedError(
                    f"tul.{name}=true — arm '{name}' (spec §3.5) is specified but NOT "
                    f"implemented in v1. Leave it false."
                )
        if self.coda_token_input not in ("prelude", "embed"):
            raise ValueError(
                f"tul.coda_token_input must be 'prelude' or 'embed', got {self.coda_token_input!r}")
        if self.tg_restrict_scope not in ("all", "coda"):
            raise ValueError(
                f"tul.tg_restrict_scope must be 'all' or 'coda', got {self.tg_restrict_scope!r}")
        if self.tg_restrict_scope == "coda" and not self.tg_restrict:
            raise ValueError(
                "tul.tg_restrict_scope='coda' requires tul.tg_restrict=true (there is no "
                "restriction to scope).")
        if self.tg_geometry not in ("restrict", "strict"):
            raise ValueError(
                f"tul.tg_geometry must be 'restrict' or 'strict', got {self.tg_geometry!r}")
        if self.tg_coda_prefix_reach not in ("all", "prev"):
            raise ValueError(
                f"tul.tg_coda_prefix_reach must be 'all' or 'prev', got "
                f"{self.tg_coda_prefix_reach!r}")
        if self.tg_geometry == "strict":
            if not self.tg_restrict:
                raise ValueError(
                    "tul.tg_geometry='strict' requires tul.tg_restrict=true: strict is a "
                    "narrowing of the TG relation, and without tg_restrict the attention "
                    "takes the pooled/top-k branch that carries no allow relation at all.")
            if self.tg_restrict_scope != "all":
                raise ValueError(
                    "tul.tg_geometry='strict' requires tul.tg_restrict_scope='all': strict "
                    "defines a PRELUDE relation and a CODA relation, and scope 'coda' "
                    "leaves the prelude global — which is the bypass strict exists to cut.")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' has no meaning on the paid loop "
                    "(tokens_through_core): a slot IS a looped position there, so 'the "
                    "loop is the only cross-span channel' is not a restriction one can "
                    "impose on the prelude and coda.")
            if self.tg_soft_prev_span:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' with tul.tg_soft_prev_span: the soft term "
                    "opens the previous span's TOKENS to a query, a second cross-span "
                    "channel beside the loop. Use tg_coda_prefix_reach='prev' if what you "
                    "want is a one-slot reach through the CELLS.")
            if self.tg_span_comp:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' with tul.tg_span_comp (E-SAC): the compressed "
                    "branch would attend per-SPAN pooled token K/V, which is a cross-span "
                    "route strict does not define a relation for.")
            if not self.coda_sees_slots:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' with coda_sees_slots=false: the coda then runs "
                    "on a GATHERED subset of positions and the strict relation is not "
                    "re-derived for that index space (the tg_restrict precedent).")
            if self.reread:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' with tul.reread: the reread builds K/V from "
                    "the frozen PRELUDE token states and lets a slot query them every pass "
                    "(reread_scope 'causal' over every earlier token), which is a cross-span "
                    "route beside the loop — the one thing strict exists to remove.")
            if self.gate is not None:
                raise NotImplementedError(
                    "tul.tg_geometry='strict' with tul.gate: the gate's budget conditioning "
                    "rewrites z after the loop and the length label is the NEXT span's, a "
                    "lookahead the strict coda relation is not defined against.")
        elif self.tg_coda_prefix_reach != "all":
            raise ValueError(
                "tul.tg_coda_prefix_reach is a tul.tg_geometry='strict' knob; at "
                f"'restrict' it would be silently ignored (got "
                f"{self.tg_coda_prefix_reach!r}).")
        if self.oracle_z:
            if not self.spandec:
                raise ValueError(
                    "tul.oracle_z requires tul.spandec: the oracle descends the SPAN "
                    "DECODER's loss, and without the decoder there is no reader to "
                    "descend.")
            if self.oracle_z_steps < 1:
                raise ValueError(
                    f"tul.oracle_z_steps must be >= 1, got {self.oracle_z_steps}")
            if self.oracle_z_lr <= 0.0:
                raise ValueError(
                    f"tul.oracle_z_lr must be > 0, got {self.oracle_z_lr}")
            if self.oracle_z_weight <= 0.0:
                raise ValueError(
                    "tul.oracle_z needs tul.oracle_z_weight > 0: at 0 the trajectory is "
                    "built and thrown away, which is an expensive way to run the ruler "
                    f"under another name (got {self.oracle_z_weight})")
            if self.oracle_z_max_tokens < 2:
                raise ValueError(
                    f"tul.oracle_z_max_tokens must be >= 2, got {self.oracle_z_max_tokens}")
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.oracle_z has no meaning on the paid loop (tokens_through_core): "
                    "there is no per-slot looped trajectory to supervise.")
            if self.detach_z:
                raise NotImplementedError(
                    "tul.oracle_z with tul.detach_z: the oracle's whole claim is that the "
                    "coda's reader and the per-pass teacher grade the SAME state, and "
                    "detach_z cuts the coda off from it.")
            if self.spandec_horizon > 1:
                raise NotImplementedError(
                    "tul.oracle_z with tul.spandec_horizon > 1: the oracle already runs T "
                    "forward+backward passes of a [B, S, J, V] readout per step, and H "
                    "multiplies that by H. No arm needs both, so the combination raises "
                    "rather than quietly costing H times the arithmetic in the header.")
            if self.db_loop:
                raise NotImplementedError(
                    "tul.oracle_z with tul.db_loop: the db carry is detached per iteration, "
                    "so h_t is not a trajectory of one map and matching it to z*_t "
                    "supervises T independent one-step readouts.")
        elif (self.oracle_z_steps != 6 or self.oracle_z_lr != 0.1
                or self.oracle_z_weight != 1.0 or self.oracle_z_max_tokens != 8):
            raise ValueError(
                "tul.oracle_z_* set with tul.oracle_z=false: nothing is built, so the "
                "knobs would be silently ignored. Set tul.oracle_z: true or drop them.")
        if self.loop_reach < 0:
            raise ValueError(f"tul.loop_reach must be >= 0 (0 = unlimited), got "
                             f"{self.loop_reach}")
        if self.loop_reach > 0:
            if self.tg_geometry != "strict":
                raise ValueError(
                    "tul.loop_reach > 0 requires tul.tg_geometry='strict': outside strict "
                    "the prelude and the coda still carry cross-span routes, so bounding "
                    "the loop's own reach bounds nothing about how far context travels.")
            if self.slot_chain:
                raise NotImplementedError(
                    "tul.loop_reach with tul.slot_chain: the chain adds W(z_{k-1}) at every "
                    "pass, a second cross-cell route inside the loop. It happens to sit "
                    "inside a reach-1 budget by arithmetic, but it is not expressed by the "
                    "mask, so the combination would be a claim nothing here has tested.")
        if self.reread_scope not in ("span", "causal"):
            raise ValueError(
                f"tul.reread_scope must be 'span' or 'causal', got {self.reread_scope!r}")
        if self.reread and self.tokens_through_core:
            raise NotImplementedError(
                "tul.reread=true has no defined meaning under the paid loop "
                "(tokens_through_core): every token already re-reads every token each pass.")
        if self.reread and self.reread_heads < 1:
            raise ValueError("tul.reread_heads must be >= 1")
        if not 0.0 <= self.progressive_p <= 1.0:
            raise ValueError(
                f"tul.progressive_p must be in [0,1], got {self.progressive_p}")
        if self.progressive_p > 0.0:
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.progressive_p is a SLOT-LOOP lever (_tul_core): it draws a "
                    "per-slot no-grad prefix, and the paid loop (tokens_through_core) has "
                    "no per-slot depth to cut. Raises rather than silently running the "
                    "token core with the knob ignored.")
            if self.db_loop:
                raise ValueError(
                    "tul.progressive_p with tul.db_loop is not defined: db_loop already "
                    "detaches the carry at EVERY iteration, so there is no gradient path "
                    "left for a random prefix to cut.")
            if self.mux_stage_own_iters > 0:
                raise NotImplementedError(
                    "tul.progressive_p with tul.mux_stage_own_iters is not defined: the "
                    "staged target supervises the state after iteration k, which a drawn "
                    "prefix may have detached — the stage loss would then train nothing "
                    "on those slots, silently.")
        if self.pass_lora_rank < 0:
            raise ValueError(
                f"tul.pass_lora_rank must be >= 0 (0 = off), got {self.pass_lora_rank}")
        if self.pass_lora_rank > 0:
            _legal_lora = ("attn", "mlp")
            _t = tuple(self.pass_lora_targets)
            if not _t or any(x not in _legal_lora for x in _t) or len(set(_t)) != len(_t):
                raise ValueError(
                    f"tul.pass_lora_targets must be a non-empty subset of {_legal_lora} "
                    f"without repeats, got {_t}")
            self.pass_lora_targets = _t
            if self.tokens_through_core:
                raise NotImplementedError(
                    "tul.pass_lora_rank is a SLOT-LOOP lever: the arm's question is what a "
                    "per-pass delta does to the slot loop's shared map. The paid loop runs "
                    "_core_region over every position and was not measured with it. Raises "
                    "rather than quietly running a different experiment.")
            if self.core_stage_cond != "none":
                raise ValueError(
                    "tul.pass_lora_rank with tul.core_stage_cond is two per-iteration "
                    "conditioning mechanisms at once (an AdaLN-Zero signal for WHICH pass, "
                    "and a per-pass weight delta). Neither the interaction nor the "
                    "attribution has been reasoned about; run one at a time.")
        if self.tokens_through_core and (self.bcast or self.coda_token_input != "prelude"):
            raise NotImplementedError(
                "tul.bcast / tul.coda_token_input='embed' have no defined interaction with "
                "the paid loop (tokens_through_core): there is no per-slot looped state to "
                "unpack and no separate coda input for the tokens. Raises rather than "
                "silently picking a behaviour.")
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


def unpack_index(layout: "SlotLayout") -> tuple[Tensor, Tensor, Tensor]:
    """Per position: the slot whose thought this token decodes, and its offset in the span.

    Token ``p`` of span ``k ≥ 1`` decodes slot ``k−1`` (the slot that terminates the span
    before it). Returns ``(src_slot [B,L] int64, offset [B,L] int64, valid [B,L] bool)``:
    ``offset`` = ``p − (slot_index[k−1] + prefix_k)``, 0 at the span's first token;
    ``valid`` = a token position, ``k ≥ 1``, and slot ``k−1`` exists. Unlike
    :func:`mux_span_targets` the span need NOT be terminated: at generation time the span
    being decoded is always the unterminated one, and the tail dump bin (``bag_id ==
    max_slots``) decodes the last slot like any other span. Span 0 has no slot before it
    and gets nothing. ``src_slot`` / ``offset`` are clamped to 0 where ``valid`` is False.
    """
    k = layout.bag_id                                                   # [B, L]
    S = layout.slot_index.shape[1]
    prev = (k - 1).clamp(0, S - 1)
    prev_ok = torch.gather(layout.slot_valid, 1, prev)
    valid = (~layout.slot_mask) & (k >= 1) & prev_ok
    start = torch.gather(layout.slot_index, 1, prev) + layout.prefix_k
    pos = torch.arange(k.shape[1], device=k.device).unsqueeze(0)
    off = (pos - start).clamp(min=0)
    z = torch.zeros_like(k)
    return (torch.where(valid, prev, z), torch.where(valid, off, z), valid)


def reread_allow(layout: "SlotLayout", scope: str) -> Tensor:
    """``[B, S, L]`` bool: the TOKEN positions slot ``s`` may read each pass (``tul.reread``).

    ``"span"``: the tokens of its own span (``bag_id == s``); ``"causal"``: its own span and
    every earlier one (``bag_id <= s``). Slot cells (prefix and pad positions) are never
    read. A row with nothing to read (an invalid slot) allows position 0 alone so the
    softmax has a key; ``TULReread.read`` masks that row's output off.
    """
    B, L = layout.bag_id.shape
    S = layout.slot_index.shape[1]
    s_idx = torch.arange(S, device=layout.bag_id.device).view(1, S, 1)
    bag = layout.bag_id.unsqueeze(1)                                   # [B, 1, L]
    tok = (~layout.slot_mask).unsqueeze(1)                             # [B, 1, L]
    if scope == "span":
        allow = (bag == s_idx) & tok
    elif scope == "causal":
        allow = (bag <= s_idx) & tok
    else:
        raise ValueError(f"reread scope must be 'span' or 'causal', got {scope!r}")
    allow = allow & layout.slot_valid.unsqueeze(-1)
    empty = ~allow.any(dim=-1, keepdim=True)                           # [B, S, 1]
    first = torch.zeros(1, 1, L, dtype=torch.bool, device=allow.device)
    first[..., 0] = True
    return allow | (empty & first)


class TULReread(nn.Module):
    """The looping slot re-reads the frozen prelude token states each pass (``tul.reread``).

    ``prepare`` builds K/V ONCE per forward from ``input_norm(prelude)`` at every position
    (the loop's own ``xn``, stream-meaned); ``read`` turns the current slot state into a
    query, attends under ``reread_allow`` and returns a ``[B, S, C]`` term the caller adds
    to the carrier before the core blocks run. Plain ``nn.Parameter`` weights, never
    ternarised (the ternary scope quantises ``nn.Linear``; the TUL apparatus stays bf16
    like ``W_prefix`` and ``W_bcast``). Init from a private generator, so building the
    module leaves the model's RNG stream and every other weight byte-identical.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"tul.reread_heads={n_heads} must divide d_model={d_model}")
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        g = torch.Generator(device="cpu").manual_seed(0x5EED)

        def _w() -> nn.Parameter:
            return nn.Parameter(torch.empty(d_model, d_model).normal_(0.0, 0.02, generator=g))

        self.W_q, self.W_k, self.W_v = _w(), _w(), _w()
        self.W_o = nn.Parameter(torch.zeros(d_model, d_model))       # zero: step 0 is a no-op
        self.q_scale = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-6

    def prepare(self, xn_tokens: Tensor, layout: "SlotLayout", scope: str
                ) -> tuple[Tensor, Tensor, Tensor]:
        """K/V ``[B, H, L, D]`` from the frozen prelude states and the allow mask ``[B, 1, S, L]``."""
        B, L, C = xn_tokens.shape
        H, D = self.n_heads, self.d_head
        k = F.linear(xn_tokens, self.W_k.to(xn_tokens.dtype)).view(B, L, H, D).transpose(1, 2)
        v = F.linear(xn_tokens, self.W_v.to(xn_tokens.dtype)).view(B, L, H, D).transpose(1, 2)
        return k, v, reread_allow(layout, scope).unsqueeze(1)

    def read(self, h: Tensor, k: Tensor, v: Tensor, allow: Tensor, slot_valid: Tensor) -> Tensor:
        """The ``[B, S, C]`` read term for the current slot state ``h`` (``[B, S, n, C]`` or ``[B, S, C]``)."""
        z = h.mean(dim=2) if h.dim() == 4 else h
        z = F.rms_norm(z.float(), (z.shape[-1],), self.q_scale.float(), self.eps).to(k.dtype)
        B, S, C = z.shape
        H, D = self.n_heads, self.d_head
        q = F.linear(z, self.W_q.to(z.dtype)).view(B, S, H, D).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)            # [B, H, S, D]
        o = o.transpose(1, 2).reshape(B, S, C)
        term = F.linear(o, self.W_o.to(o.dtype))
        return term * slot_valid.unsqueeze(-1).to(term.dtype)


class TULSlotChain(nn.Module):
    """The slot chain (``tul.slot_chain``): a direct learned path along the slot axis.

    At every pass of :meth:`MORPHTransformer._tul_core`, slot ``k`` receives
    ``W(z_{k-1})`` where ``z_{k-1}`` is the previous slot's CURRENT looped state (its exit
    state whenever that slot's realised depth is already spent — a frozen slot carries its
    final state through the where-carry). Slot 0 receives one learned vector.

    Why a direct path when the core already attends the compact slot sequence causally:
    attention is content-addressed, softmax-normalised and spread over up to 64 keys, so a
    running document state has to win a competition to survive one pass and then win it
    again at the next. A residual edge from the immediate predecessor does not. The
    measured cross-span budget has a FLAT 0.31-nat component at every offset eight or more
    tokens into a span (``lab/experiments/failures/2026-09-11-arc-span-budget.md``), which
    is what a running state carries and a per-span summary does not.

    ``W`` and the first-slot vector are ZERO-init, so the arm's step 0 is its ruler's
    forward bit for bit and no RNG is drawn at construction.
    """

    def __init__(self, d_model: int, detach: bool):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.W.weight.zero_()
        self.first = nn.Parameter(torch.zeros(d_model))
        self.detach = bool(detach)

    def forward(self, h: Tensor, slot_valid: Tensor) -> Tensor:
        """``h [B, S, (n,) C]`` -> the chain term ``[B, S, C]``, zero at invalid slots.

        The Hyper-Connection streams are reduced by the same unweighted mean every other
        single-stream read of this carrier takes (``_readout``, ``TULSlots.unpack``); the
        caller broadcasts the result back over the streams through ``_apply_injection``.
        """
        z = h.mean(dim=2) if h.dim() == 4 else h                       # [B, S, C]
        if self.detach:
            z = z.detach()
        B, S, C = z.shape
        prev = torch.cat([self.first.to(z.dtype).view(1, 1, C).expand(B, 1, C),
                          z[:, :-1]], dim=1)                           # [B, S, C]
        term = self.W(prev)
        return torch.where(slot_valid.unsqueeze(-1), term, torch.zeros_like(term))


class TULGradPass(nn.Module):
    """``W_g``: the normalised own-span gradient becomes an extra input to the next pass.

    The caller (``MORPHTransformer._own_span_grad``) computes the raw ``[B, S, C]``
    gradient ``dL_own/dz`` at the current slot state and hands it here DETACHED. This
    module owns only the two decisions the knob names — how the gradient is normalised
    (``tul.grad_pass_norm``) and at what scale ``W_g`` sees it (``tul.grad_pass_scale``) —
    plus the map itself.

    ``W_g`` is a plain ``nn.Parameter`` at ZERO, never ternarised (the ternary scope
    quantises ``nn.Linear``; the TUL apparatus stays bf16 like ``W_prefix`` and
    ``W_bcast``). Zero init means two things at once: the arm's step 0 is the ruler's
    forward bit for bit, and building the module draws NO random numbers, so every other
    weight of an arm with the knob on equals the ruler's at the same seed. ``W_g`` still
    escapes zero on the first backward — ``dL/dW_g = (dL/dh_in) (x) g`` does not vanish at
    ``W_g = 0``.
    """

    def __init__(self, d_model: int, scale: float, norm: str):
        super().__init__()
        if norm not in ("rms", "none"):
            raise ValueError(f"tul.grad_pass_norm must be 'rms' or 'none', got {norm!r}")
        self.W_g = nn.Parameter(torch.zeros(d_model, d_model))
        self.scale = float(scale)
        self.norm = str(norm)
        self.eps = 1e-6

    def forward(self, g: Tensor, slot_valid: Tensor) -> Tensor:
        """``[B, S, C]`` term to add to the carrier, from the raw gradient ``g``.

        ``slot_valid`` zeroes pad slots. Their gradient is already exactly zero (the own
        target never supervises a pad), so this is a statement of the invariant at the
        place it is relied on, not a correction.
        """
        if self.norm == "rms":
            gf = g.float()
            rms = gf.pow(2).mean(dim=-1, keepdim=True).sqrt()
            g = (gf / (rms + self.eps)).to(g.dtype)
        g = g * slot_valid.unsqueeze(-1).to(g.dtype)
        return F.linear(self.scale * g, self.W_g.to(g.dtype))


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

    def __init__(self, d_model: int, tul: TULConfig, with_prefix: bool = True):
        super().__init__()
        self.tul = tul
        # [d] shared, or [per_slot_embed, d] one row per slot INDEX. The shared shape stays
        # the default so every existing checkpoint loads and every existing run is unchanged.
        self.E_slot = nn.Parameter(
            torch.zeros(tul.per_slot_embed, d_model) if tul.per_slot_embed > 0
            else torch.zeros(d_model))
        self.E_mask = nn.Parameter(torch.zeros(d_model))
        # W_prefix exists only where something WRITES through it: the slot loop's
        # prefix_project and the FM planner. The paid loop (tokens_through_core) writes
        # nothing through a projection, so MORPHTransformer builds it with_prefix=False
        # and its checkpoints carry no such tensor (train.RETIRED_TUL_KEYS).
        self.W_prefix: nn.Parameter | None = None
        if with_prefix:
            eye = torch.eye(d_model).unsqueeze(0).repeat(tul.prefix_k, 1, 1)
            self.W_prefix = nn.Parameter(eye)
        # E_pass [prefix_k, d] — the per-cell PASS-INDEX embedding, built only when
        # `tul.prefix_source` is not "exit". Init ZERO and RNG-neutral (no draw), so an
        # `exit_repeat` model is bit-identical to an `exit` model at the same `prefix_k`
        # at step 0 and differs only once the optimiser has moved it. Its job: under
        # "trajectory" cell k holds the state after pass k+1 and cell K-1 holds the exit,
        # and `W_prefix[k]` alone starts identical for every k, so without this the coda
        # has no way to tell "pass 2" from "the exit" until the projections separate.
        self.E_pass: nn.Parameter | None = None
        if tul.prefix_source != "exit":
            self.E_pass = nn.Parameter(torch.zeros(tul.prefix_k, d_model))
        # W_bcast [bound_span_cap, d, d] — the unpack (spec §3.5 `bcast`; TULConfig.bcast).
        # Init ZERO, so at step 0 the coda input is exactly the no-bcast one and the arm
        # differs from its control by trainable parameters alone. RNG-neutral (no draw).
        self.W_bcast: nn.Parameter | None = None
        if tul.bcast:
            self.W_bcast = nn.Parameter(torch.zeros(tul.bound_span_cap, d_model, d_model))
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

    def prefix_project(self, h_slots: Tensor, layout: SlotLayout, l_total: int,
                       cells: Tensor | None = None) -> Tensor:
        """``[B, S, …, C]`` looped states → ``[B, S·prefix_k]`` values and their positions.

        Returns ``(values, index)`` ready for :func:`scatter_positions`: value ``k`` of
        slot ``s`` is ``h_s W_k`` and lands at ``slot_index[s] + k``. Invalid slots address
        the dump row. Spec §3.1: the first ``prefix_k − 1`` positions carry the plan with
        NO label, the last one predicts the first token of the next span.

        ``cells`` ``[B, S, K, …, C]`` (``tul.prefix_source`` != "exit") gives each cell its
        OWN source state — under "trajectory" cell ``k`` carries the state after pass
        ``k+1`` and the last cell carries the exit; under "exit_repeat" every entry is the
        exit. Each still goes through its own ``W_prefix[k]``, so the projection is the
        shipped one and the arm differs by WHAT is projected. ``None`` — every model
        before this parameter existed — broadcasts ``h_slots`` to every cell, which is
        the identical arithmetic to the old body and is asserted bit-exact in
        ``tests/test_tul_prefix_source.py``.
        """
        K = self.tul.prefix_k
        B, S = layout.slot_index.shape
        C = h_slots.shape[-1]
        mid = h_slots.shape[2:-1]                      # () plain carrier, (n,) HC carrier
        if self.W_prefix is None:
            raise RuntimeError(
                "TULSlots.prefix_project needs W_prefix, which is built only for a slot-loop "
                "model or an FM planner (TULSlots(with_prefix=True)); the paid loop "
                "(tul.tokens_through_core) has no projection to write through.")
        w = self.W_prefix.to(h_slots.dtype)
        if cells is None:
            # [B,S,M,C] ⊗ [K,C,C] → [B,S,K,M,C] by broadcast matmul (batch dims
            # (B,S,1)×(1,1,K)).
            hm = h_slots.reshape(B, S, -1, C)
            proj = torch.matmul(hm.unsqueeze(2), w.view(1, 1, K, C, C))
        else:
            if cells.shape[:3] != (B, S, K) or cells.shape[3:] != h_slots.shape[2:]:
                raise ValueError(
                    f"prefix_project cells {tuple(cells.shape)} must be [B, S, K, *carrier] "
                    f"= {(B, S, K, *h_slots.shape[2:])}")
            # The same per-cell projection, with a per-cell SOURCE: [B,S,K,M,C] ⊗ [K,C,C].
            cm = cells.reshape(B, S, K, -1, C)
            proj = torch.matmul(cm, w.view(1, 1, K, C, C))
        if self.E_pass is not None:
            # Broadcast over (B, S) and over the carrier's stream axis: one vector per
            # CELL INDEX, zero-init, so this line is an exact no-op at step 0.
            proj = proj + self.E_pass.to(proj.dtype).view(1, 1, K, *([1] * (proj.dim() - 4)), C)
        values = proj.reshape(B, S * K, *mid, C)       # slot-major: index s·K + k
        offs = torch.arange(K, device=layout.slot_index.device)
        pos = layout.slot_index.unsqueeze(-1) + offs                      # [B, S, K]
        pos = torch.where(layout.slot_valid.unsqueeze(-1), pos, l_total)
        return values, pos.reshape(B, S * K)

    def unpack(self, h_slots: Tensor, layout: SlotLayout) -> Tensor:
        """The bcast term ``[B, L, C]``: ``W_bcast[offset(p)] · z_{prev slot of p}`` at every
        token position ``p`` of a span whose preceding slot exists, zero elsewhere.

        ``z`` is the looped state's stream mean (the same reduction ``_readout`` and the
        mux head use). Computed slot-major, ``[B, S, K, C]`` = every slot through every
        offset's linear (``B·S·K·C²`` FLOPs, ~15 GFLOP at the panel shape), then gathered
        per token position — 18x cheaper than a per-token masked loop over the ``K``
        linears and static-shaped for the compiler. Offsets at or past ``bound_span_cap``
        are clamped to the last linear (the packer caps spans at ``span_cap`` =
        ``bound_span_cap``, so only the unterminated tail can exceed it).
        """
        if self.W_bcast is None:
            raise RuntimeError("TULSlots.unpack needs tul.bcast=true (W_bcast is not built)")
        z = h_slots.mean(dim=2) if h_slots.dim() == 4 else h_slots      # [B, S, C]
        w = self.W_bcast.to(z.dtype)                                    # [K, C, C]
        B, S, C = z.shape
        K = w.shape[0]
        u = torch.einsum("bsc,kdc->bskd", z, w)                          # [B, S, K, C]
        src, off, valid = unpack_index(layout)
        off = off.clamp(max=K - 1)
        flat = (src * K + off).reshape(B, -1, 1).expand(-1, -1, C)        # [B, L, C]
        term = torch.gather(u.reshape(B, S * K, C), 1, flat)             # [B, L, C]
        return torch.where(valid.unsqueeze(-1), term, torch.zeros_like(term))

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
