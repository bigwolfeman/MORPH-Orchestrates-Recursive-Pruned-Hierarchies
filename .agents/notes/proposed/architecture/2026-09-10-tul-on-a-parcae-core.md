# Agent Note: TUL on a plain Parcae core

Status: proposed

Date: 2026-09-10. Knob: `model.core_impl` (`morph` | `parcae`). Code:
[`morph/model/parcae_core.py`](../../../../morph/model/parcae_core.py), the build in
`MORPHTransformer.__init__` and the stream boundary in `_apply_core_step`
([`morph/model/transformer.py`](../../../../morph/model/transformer.py)). Config:
[`morph/configs/tul_slot_mnext_parcae_core.yaml`](../../../../morph/configs/tul_slot_mnext_parcae_core.yaml).
Tests: [`tests/test_parcae_core.py`](../../../../tests/test_parcae_core.py). Prereg:
[`2026-09-10-arc-slot-mnext-parcae-core.md`](../../../../lab/experiments/planned/2026-09-10-arc-slot-mnext-parcae-core.md).
Sits beside the loss-attachment note
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](2026-09-10-credit-assignment-in-the-slot-loop.md),
which covers the four arms that varied the LOSS; this one varies the CORE.

## Problem

The slot loop's contribution is flat on every arm ever measured. Passes 2 through 6 of a
268 M-parameter core are worth 0.0004 to 0.0018 nats to the coda, on three checkpoints with
three different objectives and three different coda wirings
([slot-geometry-audit](../../../../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md)).
Four one-factor arms on the loss attachment — progressive, per-pass LoRA, MUX at every pass,
staged targets — did not move it either.

That audit closed the obvious forward story: the core is NOT muted on the compact sequence.
One pass moves a slot state as far as the same weights move a token state (0.65 / 2.94 / 247x
against 1.04 / 2.78 / 161x, paired), and the prefix write hands the coda a cell that is 1.07x,
6.8x and 802x the loop's own work. What no instrument in the tree can ask is the next
question: would a DIFFERENT core, on the SAME mechanism, earn depth?

The question has teeth because the core MORPH loops at the slot shape was not designed for a
64-cell weight-shared recurrence, and one of its features is already measured to be a limiter
on the PLAIN loop. The ternary scale rule moved the plain loop's K1−K6 from 0.033 to 0.185
and the core MLP branch's out/in from 0.2 to 0.9
([per-pass strength](../../../../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md));
E20's `depthcand-dense-core` (bf16 core, ternary elsewhere) read 0.168. Neither has ever run
on a SLOT core. On top of that the slot core carries three defects or degeneracies that the
token path does not: the HCA compressed branch is exactly dead at 64 cells while its gate
still spends 0.42-0.48 of the mixture on the zero tensor (audit finding F1), CSA's block
selection never fires (`tk` 8 of 8), and XSA leaves query 0 with an all `-inf` softmax row.

So two explanations survive and cannot be separated: **H-core**, MORPH's core at the slot
shape is the fault; **H-mech**, the TUL mechanism or its target is, and any core reads flat.

## Proposal

Add `model.core_impl`. `"morph"` is today, bit for bit. `"parcae"` builds the looped core from
`ParcaeCoreBlock` instead of `MORPHBlock`:

* dense causal softmax attention over the whole compact sequence, `n_heads` heads of the
  tree's own `d_head`, rotary from the tree's `CoPEEmbedding`, no compression, no block
  selection, no gate, no self-exclusion, no QK-Norm, no conv, no indexer;
* a plain additive pre-norm residual, single stream;
* the tree's dense `_SwiGLU` of `nn.Linear` at the ruler's `d_ff`;
* no ternary QAT: every linear carries `_ternary_exclude`, which `ternary_qat._categorize`
  honours under every scope.

The prelude and the coda stay MORPH blocks with their 4-stream Cayley carrier — they run once
and are not the question. The stream boundary is one collapse and one broadcast per pass,
inside `_apply_core_step`: the carrier enters the core as its stream MEAN (the reduction
`_readout` and `TULSlots.unpack` already use to read it) and the pass output is broadcast back
over the n streams. Everything outside the core sees the shape it always saw.

`_core_is_parcae` is a Python bool fixed at construction, so the choice is a build-time
branch and the compiled graph stays branch-free, per the subtree's first design rule.

Parcae's recurrence is `h_t = A h_{t-1} + B e + R(h_{t-1}, e)` (arXiv 2604.12946 §3, App. C).
`A`/`B` is MORPH's `DiagonalInjection` and `R` is the block stack, so the arm's config also
carries the faithful `A`/`B` — `injection_channels: all` with a learned identity-init `B`,
the same four values `slot-mnext-noise-entry` carries. Without it 768 of 1024 carrier dims
would receive no source term at all. The loop ENTRY is left to `model.core_state_init`, the
knob that already owns it.

The swap is close to parameter-matched, measured at the real shape: core 64,499,712 against
the ruler's 68,038,200 (−5.2 %), whole model 265.73 M against 268.22 M (−0.9 %), with an
identical MLP. A depth gain therefore cannot be read as extra capacity.

Things the Parcae block cannot carry — TG attention kwargs, a GLA retention branch, the HC
carrier engine's `next_inject_term`, SCSE, `core_hca_compress_ratio` — RAISE, at build where
the knob is a config and in the forward where it is an argument. A silently dropped
attention restriction is exactly the defect class F1 was.

## Alternatives considered

* **Port MORPH's core features off one at a time instead** — bf16 core first, then the HC
  carrier, then the attention. This is strictly better science per arm and it is what the
  prereg's Binding queues if H-core fires. It is the wrong FIRST arm because it costs one
  5,000-step draw per feature (three to five arms, 4-5 h each) to answer a question that may
  have the answer "none of them": if the plainest possible core also reads flat, every one of
  those arms is known flat in advance and the whole branch closes on one draw. The order is
  deliberate — one cheap disproof first, then the expensive attribution only if it is needed.
* **Run TUL in the sibling Parcae repository** (`/home/wolfe/parcae`, which already has a
  measured healthy loop: K1−K8 0.294, K3−K8 0.040 on OWT at 144 M). Rejected: TUL is the slot
  layout, the causal boundary rule, the resumable packer, `W_prefix`, the MUX loss, the
  weighted CE and the coda's token-state dropout — porting all of it is more code than
  swapping one block class, it would be a second implementation of the mechanism to keep in
  step, and any result would be confounded by every other difference between the two repos
  (data, tokenizer, optimiser, embeddings, LR). The point of a one-factor arm is that
  everything else is the ruler's, and only this design gives that.
* **A dense-weight MORPH core only** (`ternary_scope: backbone_no_core`, which already
  exists). Rejected as the first arm because it tests ONE of the four candidate features and
  the other three — the dead HCA branch, XSA, the 4-stream carrier — stay in. It is the
  first follow-up in the Binding, where it is the right instrument.
* **Swap the whole model to single-stream when `core_impl: parcae`.** Rejected: the prelude
  and the coda would stop being the ruler's, the arm would be a different model rather than a
  core swap, and every cross-arm comparison in the panel would be void.
* **Collapse the carrier once at the loop entry in `_tul_core` and expand at the exit**, rather
  than per pass in `_apply_core_step`. Rejected: `_tul_core`, `_core_region`, the Jacobian
  probe, the gain penalty and the fixed-point term all call `_apply_core_step`, so one boundary
  there covers every caller and no probe needs a second code path. Per-pass collapse also
  costs nothing extra — the core's state between passes is single-stream either way.
* **Exclude the Parcae core from ternary QAT by a path prefix**, the way `backbone_no_core`
  does with `"core."`. Rejected in favour of a module attribute the block sets on its own
  leaves: a path list has to be kept in step with the tree by hand, and the attribute travels
  with the module wherever it is placed. Same argument `PassLoRA` records for its parameters.

## Acceptance criteria

Met by this change (`pytest tests/test_parcae_core.py tests/test_tul_forward.py
tests/test_tul_setup_keys.py tests/test_checkpoint_compat.py -q` → 76 passed; full
`pytest tests/ -q` → 1055 passed, 1 skipped, 1 xfailed):

1. `core_impl: morph` is the tree as it was — a pinned gradient hash over every parameter of
   the tiny plain model, plus bit-equality of loss, parameters and gradients between an
   unset knob and an explicit `"morph"`.
2. `core_impl: parcae` builds, runs the SLOT loop and the PLAIN forward, and the backward
   reaches every Parcae weight (no parameter left with `grad is None`).
3. `_apply_core_step` returns the carrier shape every caller expects, with all n streams
   equal.
4. `apply_ternary_qat(scope="full")` — the widest scope there is — parametrizes NO module
   under `core.` on a Parcae model and still parametrizes prelude and coda; the negative
   control proves a MORPH core still ternarises.
5. Neither prune walk (`_find_cms_layers`, `_find_mortar_layers`) reaches the Parcae core;
   both reach a MORPH core.
6. Fixture sensitivity on the carry: neutralising `DiagonalInjection` (A → 1, dt → 0) moves
   the slot-loop loss.
7. Every argument the block cannot honour raises.

Not met by this change, and deliberately: nothing here says the arm EARNS anything. That is
the prereg's job, and its predictions are frozen.

## Risks

* **The two components of the swap are confounded.** The block stack and the widened
  learned-B carry change together, because a Parcae core with a ctx-only carry is not
  Parcae's recurrence. The carry alone is measured inert twice on this family
  (`slot-unpack-noise-entry` K1−K6 0.002, `slot-mnext-noise-entry` −0.0001), which is why the
  arm is drawn this way; a POSITIVE result still needs the ctx-only control the prereg's
  Binding queues first.
* **The gain hinge's operating point on a dense bf16 core is unknown.** The ruler's hinge
  binds on under 1 % of steps against a ternary map reading 0.87-0.91. A map with no ternary
  snap has no measured typical gain. If it sits above the 0.9 target the arm becomes a hinge
  fight rather than a core swap. Watch `loss/gain_est` at step 200; the re-draw is
  `slot_gain_lambda: 0`.
* **Stability.** The widened carry detonated 2 of 2 draws under `warmup: 0` in E9, and the
  scale mode (`loop/core_gain_t0`) is the failure the fixed-point term is blind to. The
  1,000-step ramp and the fixed-point term are both on, and the sustained tripwire is the
  abort rule, but this map has never been drawn.
* **A second core implementation is a maintenance cost.** Any future change to
  `_apply_core_step`'s contract now has two block classes to satisfy. The mitigation is that
  `ParcaeCoreBlock` copies `MORPHBlock.forward`'s signature exactly and raises on everything
  it cannot do, so a new mechanism cannot pass through it unnoticed. If the arm reads flat
  (H-mech) this knob should be deleted rather than left as dead surface — a follow-up note,
  in the same spirit as the standing decision to strip the paid loop off main once the slot
  TUL is final.
* **`use_kernels` / `tg_scoped_kernels` stop reaching the core**, because the Parcae block
  calls no fused Triton kernel. Throughput and any kernels-on/off comparison on this arm are
  about the prelude and the coda.
* **`slot_anatomy`'s "every stream" columns degenerate** to the single stream on this arm.
  Not a defect; do not compare that column across arms.

## Outcome (2026-09-11)

The arm ran (`lab/experiments/successes/2026-09-10-arc-slot-mnext-parcae-core.md`, every
prediction held). The slot loop is flat on the Parcae core exactly as on MORPH's: tokens K1−K6
+0.0001, forecast K1−K6 +0.0019, worth zero +0.103, entry-vs-exit +0.0060. The map itself is
very different (typical gain 0.22 against 0.87, a fixed point by pass 6, the passes' gradients
agreeing at cancellation 0.915 against 0.52), the CE is 0.035 better and the wall clock 0.63x
(23,122 tok/s). The core is exonerated; the mechanism (the slot's target and input) is the
fault. Consequence for this note: no core feature gets ported back onto the Parcae core; the
Parcae core becomes the cheap testbed for target and input work on the slot loop.
