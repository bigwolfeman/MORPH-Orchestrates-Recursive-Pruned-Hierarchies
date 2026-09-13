# Planned: train the core on the token CE, and grade a pass against the pass before it

Status: planned

Date: 2026-09-12 (frozen before any GPU step of either arm; the 5090 is running the arc
queue and no smoke of either arm exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Runs on the
STRICT geometry with coda reach `all`
([`2026-09-12-arc-strict-geometry.md`](2026-09-12-arc-strict-geometry.md)), so the
one-factor partner of both arms is `slot-spandec-strict`. Design note:
[`2026-09-12-core-token-gradient-and-within-context-critic.md`](../../../.agents/notes/proposed/architecture/2026-09-12-core-token-gradient-and-within-context-critic.md).

## Two measured facts, and the two arms they produce

**Fact 1 — nothing trains the core.** In the slot loop the six shared core blocks are
trained by the SLOT losses and by nothing else: ~51 valid cells a row, one exit target
each. The token CE reaches the core at about **1 %** of the prelude's gradient (the
per-pass cotangent probe, 2026-09-10 —
[`slot-loop-gradient-probe-readings`]). The PLAIN looped model trains the SAME six blocks
on 1,024 next-token targets a row and earns **0.185 nats** of depth under the noise entry
(2026-09-10). Twelve slot arms since 2026-09-04 read a token K1−K6 inside
**[−0.0001, +0.0033]**, and the strict ruler reads **0.0016**.

The 2026-09-11 overnight batch closed the core as a SHAPE question: no core, stability,
seed, width, HCA or geometry lever moves the slot loop, and the slot loop is flat on a
plain Parcae core too. **But none of those changed what TRAINS the core.** That is the one
input to the loop that has never been varied, and it is the only structural difference
between the flat slot core and the 0.185-nat plain core.

**Fact 2 — every energy the passes have been handed is reachable in one step.** The
`own_mux` energy fell 0.366 nats in the first pass and sat flat for seven; `recon` and
`disc` are the 2026-09-12 twins. `disc`'s label is "is this slot's next span below the
BATCH MEDIAN CE?", which is mostly a property of the text and not of `z` — it needs
shuffled-context negatives precisely because the head can score from the context alone.

Wolfe, 2026-09-12, verbatim: *"train the core on the token CE as well (tokens through the
core for gradient only, slots still the only cross-span channel at inference), and a
within-context critic that scores whether the state after pass t beats the state after
pass t−1 on the same coda loss."*

## Question

Does the slot loop's flat K-curve survive (a) giving the shared core the token objective
the plain model gives it, and (b) conditioning each pass on the gradient of a scorer
fitted to whether that pass ACTUALLY lowered the real coda's next-span CE?

## Hypothesis

The passes read zero because the map they iterate has never been trained on a dense
objective, and because no energy handed to them has ever been measured against what the
coda does with the state. Arm C attacks the first: the same six blocks, the same
per-sample Poisson depth, the same 1,024 targets a row that earn 0.185 nats on the plain
model — in TRAINING only, so the shipped forward, the sweeps and inference are untouched.
Arm D attacks the second: the critic's label is a WITHIN-CONTEXT comparison through the
real coda, so everything the two candidate states share cancels and what is left is the
worth of one pass.

## Method

Two arms, each ONE factor against `slot-spandec-strict`, 5,000 steps, seq 1024, batch 6,
seed 1, ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route off,
retention off, `tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`,
`mux_beta: 0`, `spandec` on at J 32.

| arm | config | one factor against `slot-spandec-strict` |
| --- | --- | --- |
| `slot-spandec-strict-critic` | `tul_slot_spandec_strict_critic.yaml` | `tul.grad_pass` + `grad_pass_energy: critic` |
| `slot-spandec-strict-coretok` | `tul_slot_spandec_strict_coretok.yaml` | `tul.core_token_aux` |

The critic arm carries `grad_pass_scale 0.1`, `grad_pass_norm rms` and
`pass_residual_lambda 0.01`, exactly as `tul_slot_spandec_egrad_disc.yaml` does, so its
one-factor partner within the energy family is the `disc` arm and within the panel is
`slot-spandec-strict`. That is TWO comparisons on purpose, and both are reported.

### Arm C — the core-token gradient auxiliary (`tul.core_token_aux`)

TRAINING ONLY. After the shipped forward finishes, a SECOND pass over the same prelude
output sends EVERY position — tokens and slot cells together, the paid loop's shape —
through `_core_region` (the per-sample Poisson-depth core the plain model and the paid loop
already run; no second core is written) and then through `_back_region`, and charges
`_tul_group_losses`' ordinary weighted CE. The term is exposed as `core_token_aux` /
`core_token_aux_weighted`, which `train.py` subtracts, so `train/loss` and the val loss stay
the SHIPPED path's CE.

**The shipped forward does not change.** Eval, `core_depth_sweep.py`, `worth_profile.py`,
`slot_z_optimize.py` and inference all run the slot loop with the tokens OUTSIDE the core;
the aux is guarded on `self.training`. So the K-curve and the worth profile measure the
same forward every earlier slot arm was measured on, and the slot cells are still the only
channel that crosses a span boundary at inference.

**The aux core is RESTRICTED**, and it is the part most likely to be got wrong. If a token
could read another span's tokens inside the core, the core would learn a cross-span route
the shipped forward does not have, and the arm would buy its gradient by re-opening the
bypass `strict` exists to cut. The aux core therefore runs under
`causal AND (same span OR j is a slot cell)` on the window branch, the same slot-column
restriction on the compressed branch, and the `tg_segment_ids` reset on the CCA conv and
its `W_v_prev` value shift. A token reads its own span and reaches every earlier span only
through a slot cell — the reachability the cells have inside the loop. The aux coda takes
the shipped strict coda relation and the same zeroed slot-cell injections.

**The depth is its own draw**, and that is a decision. The slot loop draws a per-SLOT
depth `[B, S]`; `_core_region` draws a per-SAMPLE one `[B]`. Every reduction of the first
to the second distorts the distribution — the max over ~51 Poisson(6) draws capped at 8 is
8 almost surely — so the aux takes the PLAIN model's own per-sample Poisson draw. Training
the core the way the plain model trains it is the arm's entire claim.

**The aux backward reaches the prelude and the coda too**, not the core alone. Detaching
the aux's entry would make it core-only, and was rejected: the aux coda would then be read
on an input distribution it never learned, and the term would mostly measure that mismatch.
Stated here because it is a real confound on the arm's CE reading, and the instrument for it
is `tul/core_token_aux_ce` against `val/ce_main`.

**Cost, arithmetic** (row = 1,024 real tokens, `L_total` 1152, 64 cells, ~51 valid, mean
depth 6), block-passes per real token:

| part | passes/token |
| --- | --- |
| shipped slot arm (prelude 4×1152 + core 6×51×6 + coda 4×1152) | 10.8 |
| exit span decoder, H = 1 (2 × 64 × 32) | 4.0 |
| **the aux** (core 6×1152×6 + coda 4×1152) | **45.0** |
| **total** | **59.8** |

against `slot-spandec-strict`'s 14.8, `-perpass`'s 35.7 and the PLAIN panel's 44.0. This is
the most expensive arm of the family: roughly one extra paid-loop forward and backward per
step. By simple proportion from the span-decoder smokes (10,829–10,888 tok/s at 14.8) it
lands near **2,700 tok/s**. **The runner these are queued on has the rate floor OFF**, so
the 8,086 figure is reported and does not gate.

### Arm D — the within-context improvement critic (`tul.grad_pass_energy: critic`)

The conditioning path is byte-for-byte the `disc` arm's: `_egrad_feature` takes
`dE/dh` of `E = -c_phi(z, ctx)` at a DETACHED copy of the state, `create_graph=False`,
RMS-normalised, scaled by `grad_pass_scale` and added through the zero-init `W_g`. What is
new is the LABEL.

For each valid slot, three candidate states and two pairs:

* the trajectory pair `(h_{t-1}, h_t)` for a per-slot `t` drawn uniformly in
  `[1, realised depth]` — "did this pass help?";
* the perturbation pair `(h_t, h_t + eps·rms(h_t)·n)`, `n ~ N(0, I)`, `eps =
  tul.critic_eps` (0.1) — "which way is up from here?". Without it the critic only ever
  sees states the loop already produces and has no reason to be smooth anywhere else,
  which is exactly where its gradient is read.

Each candidate is written into that slot's prefix cells through the SAME `prefix_project`,
the REAL coda is replayed with the SAME dropout `keep`, allow relation and retention reset,
and the next span's mean token CE is measured per slot (`slot_outcome_labels`' map: the
token positions with `bag_id == s + 1`). **Every replay is `no_grad`** — there is no graph
at all, so no gradient reaches the loop, `W_prefix`, the coda, the tied table or the span
decoder from the label; and the whole label computation sits inside an RNG save/restore, so
the perturbation draw consumes nothing from the run's stream.

The critic is trained by a PAIRWISE logistic loss on the score difference, weighted by the
measured CE gap (`w · BCE(c_a − c_b, 1{CE_a < CE_b})`, `w = |CE_a − CE_b|`), averaged over
the two pairs, on DETACHED `z`. A weighted form was chosen over a direct regression of the
CE difference because the CE gap's SCALE is not the quantity the energy needs — only its
sign and its confidence are — and a regression would spend the head's capacity fitting how
hard the row is, which is exactly `disc`'s failure mode.

**The confound, stated rather than hidden.** Under `tg_coda_prefix_reach: all` a token of
span `s+1` reads EVERY earlier slot's cells, so a replay that substitutes every slot at
once attributes span `s+1`'s CE change to slot `s` while every earlier slot's substitution
also moved it. `tul.critic_replay_groups: G` splits the substitution into G replays, each
touching slots `s ≡ g (mod G)` and leaving every other slot at its exit state, so the
nearest confounder sits G spans back — at G times the replay cost. **G = 1 is the arm's
setting and it ACCEPTS that noise.**

**Cost, arithmetic.** Three candidate states × G = **3 coda forwards per step**, no
backward through any of them: 3 × 4 × 1152 = 13,824 forward-only block-passes, 13.5 per
real token, against the ruler's 14.8 forward+backward. Plus three no-grad per-token coda
readouts (`per_token_ce`: 6 × 1152 × 1024 × 49169 × 2 = 0.70 TFLOP each, 2.1 TFLOP against
a ~50 TFLOP step, ~4 %). The energy itself has no vocabulary axis and is free. Expect
roughly 1.3–1.5× the ruler's wall clock. Arithmetic, not a measurement.

### `coda_exact` is REFUSED, and the refusal is in the tree

`tul.grad_pass_energy: "coda_exact"` — the energy IS the next-span coda CE, replayed at
every pass, differentiable with respect to the state — was asked for as a follow-up if it
fell out of the critic's replay code. **It does not**, and `TULConfig.__post_init__` raises
with the four reasons rather than omitting the option:

1. **The context is not there yet.** The energy is read inside `_tul_core`; the coda's
   inputs (`base`, the dropout `keep`, the coda allow relation, `prefix_project`'s write)
   are built AFTER the loop returns. The critic's replay reuses them because it runs at the
   END of the forward; an energy cannot.
2. **The replays have opposite gradient rules.** The critic's replay is `no_grad` BY
   CONTRACT — a label that reached the loop would make the critic a teacher. `coda_exact`
   needs the same replay differentiable with respect to the candidate. One helper cannot
   hold both contracts without a mode flag whose two branches share no test.
3. **It moves an RNG draw.** The coda's token-state dropout is drawn after the core; an
   energy read at every pass needs it drawn before, which changes the SHIPPED forward on
   every arm.
4. **The cost.** T coda forward+backward passes per step is 8 × 4 × 1152 = 36,864
   block-passes against the model's own 11,052 — 3.3× the model, before the slot loop.

So this panel is TWO arms, and the queue file carries two lines.

### The gate, built in this change

`tests/test_tul_core_token_aux.py` (22), `tests/test_tul_critic.py` and
`tests/test_tul_core_token_and_critic_arms.py` (compose, build and RUN each config at the
panel's real budgets). Both arms are proved PURELY ADDITIVE rather than argued to be:
`loss − <term>_weighted` equals an off-model's loss BIT FOR BIT, and both wrap their extra
work in an RNG save/restore so that equality does not depend on what runs after them.

Arm C's geometry is proved by a LEAK TEST rather than asserted. One core application
(`n_core: 1`, depth 1) with the slot cells' carrier and injection sources blanked must move
NOTHING outside the edited span, bit-exact on CPU fp32; two two-sided controls keep it
honest (leave the cells in and the edit crosses; widen the relation to plain causal and it
crosses with the cells still blanked). A fourth test pins the design's other half — over
TWO passes the edit DOES cross, through a cell — so nobody reads the leak test as "nothing
crosses".

Arm D's label is proved DETACHED by autograd: every loop, coda, `W_prefix` and decoder
gradient is bit-identical with and without the critic's label term, while the critic's own
gradient reaches the passes.

Ten source-level sabotages, run 2026-09-12 at the tip of this change, each editing the
shipped source, running ONE guard test and reverting, with `__pycache__` cleared and 1 s
between cases. Each patch anchor was asserted to occur exactly once before the edit, so a
patch that failed to apply could not be reported as a catch.

| sabotage | guard test | result |
| --- | --- | --- |
| C1: widen the aux core's relation to plain causal | `test_the_aux_core_lets_no_token_read_another_spans_token` | CAUGHT (1 failed) |
| C2: drop the RNG save/restore around the aux | `test_the_aux_consumes_nothing_from_the_rng_stream` | CAUGHT (1 failed) |
| C3: let the aux core's fixed-point term replace the slot loop's | `test_the_slot_loops_fixed_point_term_survives_the_aux` | CAUGHT (1 failed) |
| C4: run the aux at EVAL too | `test_eval_never_runs_the_aux` | CAUGHT (1 failed) |
| C5: let the aux's Jacobian capture into the slot map's probe | `test_a_jacobian_capture_never_sees_the_aux_core` | CAUGHT (1 failed) |
| D1: un-detach the critic's candidate states | `test_the_label_trains_nothing_but_the_critic` | CAUGHT (1 failed) |
| D2: drop the CE-gap weighting from the pairwise loss | `test_the_pairwise_loss_weights_by_the_measured_gap` | CAUGHT (1 failed) |
| D3: drop the RNG save/restore around the critic's label | `test_the_label_consumes_nothing_from_the_rng_stream` | CAUGHT (1 failed) |
| D4: make the perturbation an ABSOLUTE step | `test_the_perturbation_is_at_the_requested_rms_scale` | CAUGHT (2 failed) |
| D5: make every replay group substitute EVERY slot | `test_replay_groups_substitute_only_their_own_slots` | CAUGHT (1 failed) |

**C1 MISSED on its first run, and the fix is in the tree.** The first draft of the leak
test built its own attention kwargs instead of reading the shipped ones, so widening the
REAL relation to plain causal changed nothing the probe could see. `_core_token_aux_kwargs`
is now a named method with ONE home, the test calls it, and the sabotage is caught. Recorded
because "the guard passed" and "the guard is watching the shipped path" are different claims
and this file has the second one only because the first one failed.

### Readout, both arms

Runner `arc/run_slotloop3.sh`, KIND `slot`, SWEEP_CKS 2500,5000, OUTDIR
`/home/wolfe/morph-scratch/arc/results/2026-09-12-strict`.

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`. **This runs on the SHIPPED forward**, which arm C does not change.
2. `worth_profile.py --rows 192` (`--modes auto`) → zero / all_slots by offset bin.
   `all_slots == zero` is the strict geometry's CHECK, not a result.
3. `slot_z_optimize.py` → `ce_entry − ce_loop` against the fitted-z ceiling.
4. Paired CE against `slot-spandec-strict`'s sweep npz with
   `lab/divergence/span_budget_profile.py --full A.npz --span B.npz`, 480 rows. Arm D also
   paired against `slot-spandec-strict-egrad-disc`.
5. **The depth-isolation probe** (`lab/divergence/slot_depth_isolation.py`): what ONE
   slot's passes are worth, which is the reading a K-curve averages away.
6. Arm C only: `tul/core_token_aux_ce` against `val/ce_main` — **read this first**. If the
   aux CE is far below the model's own token CE, the core has learned a map the shipped
   forward never runs and no K-curve reading means anything.
7. Arm D only: `tul/critic_agree` (the honesty instrument — a critic at 0.5 means the energy
   carries nothing, the `egrad_auc` twin), `tul/critic_gap_traj` (the MEASURED worth of one
   pass through the real coda), `tul/gp_rel_t{t}` (how far the feature moves the state) and
   `tul/own_pass_t{t}`.
8. `lab/divergence/critic_direction_probe.py` (built in this change, see Predictions).
9. Wall clock, `loop/core_gain_t0`, the sustained tripwire.

## Predictions (frozen)

Written before any GPU step of either arm. Probabilities are the builder's.

- **P-1 (coretok gives the passes a non-zero contribution).** `slot-spandec-strict-coretok`
  token **K1−K6 above 0.005**. **35 %.** Reasoning: this is the ONE input to the loop that
  has never been varied, and it is the only structural difference between the flat slot
  core and the plain core that earns 0.185 nats. Against it: the K-curve is measured on the
  SHIPPED forward, where the tokens never enter the core, so the arm has to work by
  transfer — the core becomes a better iterated map in general and the slot passes inherit
  it — and transfer is a weaker claim than "the objective reaches the passes". The
  2026-09-11 batch also closed nine core-side levers. Residual: 50 % it lands in
  [0.000, 0.005], 15 % negative.
- **P-2 (coretok K3−K6).** `slot-spandec-strict-coretok` **K3−K6 above 0.002**. **25 %.**
  Reasoning: every arm that has moved K1−K6 at all moved the FIRST pass; the only reading
  that ever moved K3−K6 was `prev-reach1`'s +0.0042, and it got there by blinding the coda.
- **P-3 (coretok CE against strict).** `slot-spandec-strict-coretok` depth-6 CE, token
  paired on 480 rows, is **BETTER than `slot-spandec-strict` by 0.00 to 0.10** nats.
  **50 %.** Reasoning: the aux is a second dense objective on the prelude, the core and the
  coda, so the shared weights get roughly twice the token gradient per step. Against it:
  the aux coda reads a different input distribution, and pulling the shared coda toward two
  distributions can cost the shipped one. Residual: 30 % within ±0.00 (no move beyond
  noise, i.e. worse or better by under 0.00... read as: 30 % WORSE than strict), 20 %
  better by more than 0.10.
- **P-4 (coretok has not silently become the paid loop).** `tul/core_token_aux_ce` at 5,000
  is **within 0.15 nats** of `val/ce_main` on the same run. **60 %.** Reasoning: the aux
  path sees the same tokens through a strictly richer computation (the core), so it should
  be a little BETTER, not dramatically so — the restriction means it gains no context the
  shipped path lacks. A gap much larger than that says the two paths have diverged into two
  models and the arm is unreadable.
- **P-5 (coretok rate).** `slot-spandec-strict-coretok` clears 8,086 tok/s at step 200:
  **5 %.** Reasoning: 59.8 block-passes per token against strict's 14.8, and the
  span-decoder smokes read 10,829–10,888 at 14.8. This arm is expected to be the slowest of
  the whole family, and the floor is OFF on this runner, so that is a reported number and
  not a gate.
- **P-6 (the critic trains at all).** `tul/critic_agree` at 5,000 is **above 0.60**.
  **55 %.** Reasoning: the pairwise label is within-context, so the easy shortcut `disc`
  had (score the row's difficulty from `ctx`) is arithmetically removed — the two
  candidates share `ctx` exactly. Against it: if one pass genuinely changes the coda's CE
  by almost nothing, the labels are near-ties and the CE weighting shrinks the whole term
  toward zero, which is a null the instrument will show honestly.
- **P-7 (the measured worth of one pass).** `tul/critic_gap_traj` — mean
  `CE(h_{t−1}) − CE(h_t)` through the real coda — is **below 0.01 nats** at 5,000.
  **75 %.** Reasoning: this is the same quantity every K-curve has read near zero, measured
  directly for the first time rather than through a depth sweep. **If it is large and the
  K-curve is still flat, the two instruments disagree and the K-curve is the one to
  distrust** — that alone is worth the arm.
- **P-8 (critic gives the passes a non-zero contribution).** `slot-spandec-strict-critic`
  token **K1−K6 above 0.005**. **25 %.** Reasoning: `gp_rel` 0.086 on the 2026-09-11
  gradpass arm says the feature ACTS, and the critic is the first energy whose target is
  what the coda actually does with the state. Against it: three energies have now acted and
  none moved the K-curve, and a scalar critic's gradient is a rank-1 direction in a
  1024-dim state. Residual: 65 % in [0.000, 0.005], 10 % negative.
- **P-9 (the critic's direction beats a random one).** On held-out rows at 5,000, moving
  the exit state along the critic's gradient by `eps·rms` lowers the REAL coda's next-span
  CE more than an rms-matched random direction does, by **more than 0.01 nats**, on at
  least 60 % of scored slots. **45 %.** This is the arm's decisive probe and it is written
  in this change: `lab/divergence/critic_direction_probe.py`. It is the reading that
  separates "the critic learned something about the state" from "the critic learned the
  batch".
- **P-10 (critic CE against strict).** `slot-spandec-strict-critic` depth-6 CE is within
  **0.02** nats of `slot-spandec-strict` (either sign). **60 %.** Reasoning: `W_g` is
  zero-init, the critic trains only its own parameters, and the only route into the model
  is one added vector per pass. The `disc` and `recon` twins are the comparison.
- **P-11 (survival).** Both arms reach 5,000 with no sustained tripwire
  (`preclip/total > 1e4` at step ≥ 200). **80 %.** Reasoning: the 1,000-step ramp plus
  `core_fixed_point_lambda` 1.0 has held every arm in this family. Arm C puts a second
  dense objective on the core weights, which is the one place I would expect a surprise.
- **P-12 (critic rate).** `slot-spandec-strict-critic` clears 8,086 tok/s at step 200:
  **55 %.** Reasoning: 13.5 forward-only block-passes per token added to a 14.8
  forward+backward arm. The `disc` arm's measured rate is the right prior and it is not in
  front of me.

## Binding

If **P-1 holds** — the token objective on the core moves the slot loop's K-curve without
changing the shipped forward — then the flat curve was never about the map's shape or its
target; it was about what trained it, and the next arm is the aux at the DEPLOY recipe with
the aux weight swept. That also reopens every lever the 2026-09-11 batch closed, because
all nine were measured on a core trained by the slot losses alone.

If **P-1 fails and P-4 holds** — the core learns the token map, the aux CE tracks the
model's own, and the passes still read zero — then the core is exonerated a second time and
more strongly: it is not under-trained, it is under-USED. The lane left is the READ and the
WRITE (what the coda does with `z`, what `prefix_project` can carry), not the map.

If **P-6 holds and P-7 holds** — the critic trains and the measured per-pass worth is still
under 0.01 nats — that is the first DIRECT measurement of what a pass is worth to the real
coda, and it agrees with twelve K-curves. The honest next step is then to price a ONE-pass
slot model against the looped one at matched compute rather than to keep buying passes.

If **P-6 holds and P-7 fails** — the critic trains and one pass IS worth something the
K-curve cannot see — the depth sweep is the wrong instrument and every "flat" reading in
this arc needs re-reading through the replay probe.

If **P-9 holds and P-8 fails** — the critic's direction genuinely improves the state and
conditioning the passes on it still moves nothing — the bottleneck is the pass's ability to
USE a direction, not the availability of one, and that points at `W_g`'s rank and at the
pass transition, not at the objective.

## Not verified before launch

* **No GPU step of either arm.** The 5090 ran the arc queue for the whole build window.
  Every cost number above is arithmetic; the smokes are handed back unrun.
* **Nothing says either arm TRAINS.** The gates prove the terms are correct, additive,
  RNG-neutral and geometrically restricted on a tiny CPU model, and that each config
  composes, builds and runs one forward and backward at the panel's budgets. The 5,000-step
  conjunction is unmeasured.
* **Arm C's memory is unpriced on the real shapes.** The aux core is gradient-checkpointed
  exactly as the shipped one is (`model.ckpt_grad_iters`), but the aux coda's activations
  are live at the same time as the shipped path's graph. `-perpass` died of exactly this
  class of surprise on 2026-09-12 and the priced cause was off by 10×. **This is the most
  likely way arm C dies, and it will show up as an OOM in the smoke, not as a slow step.**
* **Arm C's aux backward reaches the prelude and the coda**, not only the core. That is a
  confound on its CE reading and is named in the Method rather than removed.
* **`core_token_aux_weight: 1.0` is untuned.** The aux CE is the same quantity and the same
  magnitude as the model's own token CE, so weight 1.0 doubles the token gradient on the
  shared weights. No weight has been swept.
* **`critic_eps: 0.1` and the one sampled `t` per slot per step are choices, not
  measurements.** A larger `eps` makes the perturbation pair easier to label and less
  informative about the local geometry; a smaller one buries it in the coda's own noise.
  Nothing has swept either, and no arm has compared "one sampled `t`" against "every `t`".
* **`critic_replay_groups: 1` accepts the attribution confound** described in the Method.
  G > 1 is implemented and tested for WHAT IT SUBSTITUTES and has never been run.
* **The pairwise loss's weighting is a choice.** A direct regression of the CE difference
  was considered and not built; nothing has compared them.
* **`critic_every` is counted in TRAINING FORWARDS, not optimiser steps.** Under gradient
  accumulation those differ. The panel runs accumulation 1, so on this panel they are the
  same number.
* **`lab/divergence/critic_direction_probe.py` has never been run against a trained
  checkpoint**, only against the CPU fixture its test uses.

## Results

Both arms ran on the 5090 at commit `6b2e9ed` (runner `run_recon.sh`, rate floor off), seed 1,
5,000 steps, seq 1024, batch 6. Readouts in
[`../results/2026-09-12-strict/`](../results/2026-09-12-strict/): `sweep_<arm>_{2500,5000}.json`,
`worth_<arm>_5000.json`, `paired_gaps_5000.txt` (sections "critic minus strict", "critic minus
egrad-disc", "coretok minus strict"), `critic_series_slot-spandec-strict-critic.txt`,
`core_token_aux_series_slot-spandec-strict-coretok.txt`, `core_token_aux_probe_5000.json`.
Every CE gap below is token-paired on 501,106 tokens over the same 480 validation rows;
"K" is the forced-depth sweep's token CE difference with its paired bootstrap CI.

### Arm C — `slot-spandec-strict-coretok`

| reading | value |
| --- | --- |
| rate at step 200 | 3,831 tok/s (strict ruler family ~10,850; the arm is 0.35× the ruler) |
| survival | reached 4,999; `preclip/total` max 33.2 at step 1,310; no tripwire |
| tokens K1−K6 @5000 | **+0.0005** [+0.0002, +0.0007]; @2500 +0.0001 |
| tokens K3−K6 @5000 | −0.0003 [−0.0004, −0.0001] |
| tokens K1−K16 @5000 | −0.0017 [−0.0021, −0.0013] |
| `spandec_ce` K1−K6 / K3−K6 @5000 | +0.0047 / +0.0006 |
| CE vs strict, depth 6 | **−0.0684** [−0.0710, −0.0657] (4.2790 vs 4.3474); every offset bin negative, 0: −0.033, 8+: −0.074 |
| worth zero / shuffle | 0.2035 / 0.1737 (ruler 0.1927 / 0.1719) |
| `tul/core_token_aux_ce` − `train/loss`, same batch, tail 4800–4999 | −0.314 (4.336 vs 4.650) — **confounded**: the shipped coda input carries the 0.15 token-state dropout, the aux coda input does not |
| aux path − shipped path, eval, identical rows, no dropout (`core_token_aux_probe.py`) | **−0.0164** [−0.0176, −0.0151] (4.2626 vs 4.2790) |
| the same probe on the strict ruler (a core the tokens never trained through) | **+1.1046** [+1.084, +1.125] (5.4520 vs 4.3474) |

The aux objective did what it was built to do: on the ruler the token-core path is 1.10 nats
worse than the shipped path, on the arm it is 0.016 nats better. The six shared blocks became a
competent per-token map. The slot passes through the same six blocks still read
0.0005 nats. The shipped-path CE moved 0.068 nats at every offset, which is the dense
objective on the shared prelude and coda, not the loop (the K-curve and the worth profile
say the loop's share did not change).

`core_token_aux_probe.py` was written after the run to remove the dropout confound the prereg
did not foresee; its reading, not the run-level series, scores P-4.

### Arm D — `slot-spandec-strict-critic`

| reading | value |
| --- | --- |
| rate at step 200 | 8,450 tok/s |
| survival | reached 4,999; `preclip/total` max 33.4 at step 224 |
| tokens K1−K6 @5000 | **+0.0012** [+0.0009, +0.0015]; K3−K6 −0.0000 |
| `spandec_ce` K1−K6 | +0.0032 |
| CE vs strict, depth 6 | **−0.0025** [−0.0047, −0.0002] |
| CE vs egrad-disc, depth 6 | −0.0106 [−0.0126, −0.0084] |
| worth zero / shuffle | 0.1927 / 0.1719 |
| `critic_agree` (steps 4500–5000) | **0.521** (traj 0.530, pert 0.512); 0.506 at steps 200–600 |
| `critic` loss | 0.689 (ln 2 = 0.693) throughout |
| `critic_gap_traj` = mean CE(h_{t−1}) − CE(h_t) through the real coda | 0.0012 → 0.0024 nats |
| `critic_gap_pert` | ≤ 0.0005 |
| `gp_rel_t0` | 0.011 → 0.039 |

The critic sits at chance for 5,000 steps on both pair types. The measured worth of one pass
through the real coda is 0.001–0.003 nats and the coda's loss is flat to 0.0005 under a
10 % rms perturbation of the state, so the label is a near-tie by measurement and the
CE-gap weighting shrinks the term toward zero, as the P-6 reasoning allowed. The feature
acts (`gp_rel` 0.04) and moves nothing.

### Predictions scored

| P | claim | result |
| --- | --- | --- |
| P-1 | coretok tokens K1−K6 > 0.005 | **FALSE** (0.0005) |
| P-2 | coretok K3−K6 > 0.002 | **FALSE** (−0.0003) |
| P-3 | coretok better than strict by 0.00–0.10 | **TRUE** (−0.068) |
| P-4 | aux CE within 0.15 of the model's own | **TRUE** on the clean instrument (0.016); the run-level series reads 0.31 and is dropout-confounded |
| P-5 | coretok clears 8,086 tok/s | **FALSE** (3,831), as predicted at 5 % |
| P-6 | critic_agree > 0.60 | **FALSE** (0.52) |
| P-7 | critic_gap_traj < 0.01 | **TRUE** (0.0024) |
| P-8 | critic tokens K1−K6 > 0.005 | **FALSE** (0.0012) |
| P-9 | critic direction beats random by > 0.01 on ≥ 60 % of slots | **PENDING** — `critic_direction_probe.py` chained after the math panel and the z-opt probes on the 5090 |
| P-10 | critic CE within 0.02 of strict | **TRUE** (−0.0025) |
| P-11 | both survive | **TRUE** |
| P-12 | critic clears 8,086 tok/s | **TRUE** (8,450) |

## Verdict

**Failure** on the hypothesis (P-1, P-2, P-6, P-8 false). The binding case that fired is
"P-1 fails and P-4 holds": the core is not under-trained, it is under-used. A core trained
to a 1.10-nat token map, running the identical six blocks at the identical depth over the
slot cells, adds 0.0005 nats over one pass. The critic case that fired is "P-6 fails": the
within-context label is a measured near-tie (0.001–0.003 nats per pass) and a critic cannot
learn from ties.

## Updated hypothesis

What trains the core is closed as a lever (this arm), beside its shape (2026-09-11 batch),
its stability, its target (staged, oracle, gradpass, per-pass horizon, critic) and its geometry
(strict panel). Every reading points at the same place: the READ and the WRITE. The coda's
loss is flat to 0.0005 nats under a 10 % perturbation of the exit state and moves 0.001–0.003
nats per pass, so nothing downstream of `prefix_project` asks the state for more than one
pass produces. The lane left is what the coda can take from `z` (the planning-cells design
in the drawing-board note, held for Wolfe's word) and a matched-compute one-pass slot model
priced against the looped one.

Not verified: P-9 (queued); the aux weight (1.0) was never swept; one seed.
