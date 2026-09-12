# Planned: the latent-z gradient — give the loop an energy it cannot descend in one pass

Status: planned

Date: 2026-09-12 (frozen before launch; **GPU smokes pending** — the 5090 was running
`slot-mask-mux-quarter` for the whole build window, so no arm here has run a single GPU
step. The exact commands are in `egrad_smoke_cmds.txt` beside this session's scratch and
are repeated in §Method.) Arc: `2026-09-04-loop-contribution-arc.md`. Design note:
[`2026-09-12-latent-z-gradient-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-12-latent-z-gradient-loop.md).

Evidence it stands on:
[`slot-mnext-gradpass`](../failures/2026-09-10-arc-slot-mnext-gradpass.md) (the mechanism
works and its energy was too easy),
[the span decoder, part 1](2026-09-11-arc-span-decoder.md) (the exit TARGET, −0.072 nats
against the mask arm, with the gain routed through the slot cells rather than the loop's
write),
[`slot_z_optimize`](../results/2026-09-10-slot-z-optimize/README.md) (a gradient-fitted z
is worth 0.9–2.6 nats to the coda and the loop's z is worth +0.002 over its own entry),
[the cross-span budget](../failures/2026-09-11-arc-span-budget.md) (0.399 nats, flat 0.315
at offset 8+).

## Question

`slot-mnext-gradpass` answered "can a pass be conditioned on the gradient of its own
objective" with a clean yes: the feature acted (`gp_rel_t0` 0.086), the loop descended the
energy, and the exit beat the ruler's forecast by 0.017 nats — the largest exit move any
credit-assignment arm has produced. It also answered a question nobody asked: **the energy
was reachable in one pass.** The ladder fell 0.385 nats from t0 to t7, of which 0.366 was
the FIRST pass, and passes 3–7 sat within 0.03 of each other.

The reason is structural, not a tuning failure. That energy is
`_tul_mux_loss(target="own")`: the slot's own span against an ORDER-FREE geometric bag
through the tied head. Its minimiser is the span's weighted unigram marginal, and the span
is already in the slot's entry state because the prelude ran over every one of its tokens.
One pass is enough because one pass is all the target asks for.

So: **does an energy that a single pass CANNOT saturate make the later passes do work, and
does the coda read the result?** And before either arm is worth running —

**does the slot state carry scoreable structure at all?** The literature splits on this by
the training signal, not the architecture: on Huginn-style recurrent trajectories a
correctness classifier on the latent reaches ROC-AUC near 1.0 (Latent Thinking
Optimization, arXiv 2509.26314); on Coconut-style latents supervised only through
downstream tokens it sits at chance (arXiv 2510.12167, Cohen's d 0.17). MORPH's slot is
the second kind and has never been measured on that axis.

## Hypothesis

**H-e-1 (for).** The gradpass result decomposes as *mechanism good, energy trivial*. A
conditional, ordered, per-token reconstruction of the own span is a potential with
curvature in directions a marginal has none, so pass 2 has somewhere to go that pass 1 did
not already reach. The signatures: a ladder that keeps falling past t1, a rising or at
least non-falling probe AUC along the pass index, and an exit the coda uses more
(`ce_entry − ce_loop` up).

**H-e-0 (against).** The binding constraint is not the energy's difficulty, it is that
MORPH's geometry does not NEED a slot thought: the prelude is causal over every cell and
the coda reads a causal chain of prefix cells, so both compose across spans by themselves
(the toy study's Q4, `lab/toy_slot_loop/WRITEUP.md`). On this reading a harder energy makes
the loop a better optimiser of something the model does not want, and the arms read like
the ruler with an extra injection channel and an extra 20 % of wall clock.

The two readings disagree about the `egrad_t` ladder past t1 and about the probe's slope,
which is why those are the two predictions this file is really built on.

## Method

### Step 0, the gate (eval only, no training)

`lab/divergence/slot_state_linear_probe.py`. New file — the name
`lab/divergence/slot_state_probe.py` was already taken by the live per-depth STATE probe
the arc runner invokes as `STATEPROBE`, and overwriting a script the queue is executing
was not an option.

For each slot it fits a linear probe on the per-pass state `z_t` (t = 0..6; `z_0` is the
loop's ENTRY `core_init(e)`, `z_t` for t ≥ 1 is the tensor `prefix_project` would receive
if the loop stopped at t, captured by forcing `tul.slot_depth_fixed`) against a binary
label: `y = 1` when the coda's MEAN token CE over that slot's NEXT span is below the
median over every scored slot of the run. 5-fold cross-validated ROC-AUC on out-of-fold
scores, 200-draw bootstrap CI over slots, two feature reductions (the FLATTENED
Hyper-Connection streams, what `prefix_project` sees; and the stream MEAN, what `_readout`
and the MUX head see — finding F2 says they disagree). Also the concatenated trajectory
`z_{0..t}`, the per-pass relative step, the cosine between successive steps, entropy
effective rank over the dimension, and Hoyer sparsity, each split by label.

The label comes from `morph.model.tul_egrad.slot_outcome_labels` — the SAME function the
`disc` arm trains its critic against, so probe and arm cannot drift apart.

Three self-checks, printed with every result:

* **label shuffle** — the whole pipeline on permuted labels, repeated `--shuffles` times,
  reported as a null BAND. A pass AUC is signal only if it clears that band's upper end.
  The script prints `!! SELF-CHECK FAILED` and marks the JSON when the shuffle mean leaves
  0.50 ± 0.03.
* **entry vs exit** — `z_0` alone against `z_6` alone, in the same table.
* **the split point** — the recorded z replayed through `slot_z_optimize.ZSplit` must
  reproduce the trained forward's token CE EXACTLY, and `prefix_project` must receive the
  substituted object. Both read BIT-EXACT on both CPU smokes below.

**CPU smoke, 6 rows, 306 scored slots, depth 6 — smoke numbers, not a result.** Ran
2026-09-12 02:1x–02:31, `CUDA_VISIBLE_DEVICES=""`, torch LBFGS backend (sklearn is in the
MORPH-TUL venv and NOT in the training venv; the script uses whichever is importable).

| checkpoint | AUC t0 flat / mean | AUC t6 flat / mean | shuffle null mean (p95) | step_rel t0→t5 | step_cos t1→t5 |
| --- | --- | --- | --- | --- | --- |
| `slot-spandec-mask@5000` | 0.598 / 0.602 | 0.566 / 0.599 | 0.504 (0.577) / 0.508 (0.560) | 0.320 → 0.016 | 0.82 → 0.94 |
| `slot-mux-norm-match@5000` (ruler) | 0.677 / 0.655 | 0.682 / 0.667 | 0.527 (0.568) / 0.511 (0.551) | 0.675 → 0.026 | 0.84 → 0.91 |

At 306 slots the CIs are ±0.06, so only the ruler's ~0.68 is clearly outside its null band;
the spandec arm's 0.57–0.60 is not. Two things are already visible and both cut against the
orchestrator's framing: the signal does **not** rise with the pass index on either
checkpoint (it is flat on the ruler and falls slightly on spandec-mask), and the
successive-step cosine settles at **0.82–0.94**, far above the 0.5–0.65 band the prediction
list expected. The 96-row GPU run is the reading; these two shapes are stated here so that
the GPU run cannot be read as confirming a prediction it has already contradicted at n=306.

One defect found and fixed by the smoke, recorded because it would have produced a fake
result: pooling raw per-fold decision scores makes the AUC read each fold's intercept as
signal — the first smoke's shuffled-label control read **0.576** instead of 0.5. The fix
(standardise the scores WITHIN each fold before pooling) brought it to 0.504/0.508.

GPU commands, to run when the card frees (repeated in `egrad_step0_cmds.txt`):

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  $PY lab/divergence/slot_state_linear_probe.py \
  --ckpt spandec-mask=tul_slot_spandec_mask=$CK/slot-spandec-mask/step_5000.pt \
  --rows 96 --batch 3 --depth 6 --shuffles 20 --out $OUT/step0-spandec-mask.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  $PY lab/divergence/slot_state_linear_probe.py \
  --ckpt ruler=tul_slot_mux_norm_match=$CK/slot-mux-norm-match/step_5000.pt \
  --rows 96 --batch 3 --depth 6 --shuffles 20 \
  --override model.core_hca_compress_ratio=256 --out $OUT/step0-ruler.json
```

The ruler override is not optional and is not this arc's doing: `slot-mux-norm-match`
predates the `core_hca_compress_ratio: 16` default shipped in `tul_short.yaml` on
2026-09-11, so its checkpoint carries `[256, 64]` compressor weights and the current
config builds `[16, 64]`. Without the override the load raises a size mismatch.

### Amendment 1 (2026-09-12, Builder 2): the control changed, no prediction touched

The three arms below now compose `tul_slot_spandec_strict` (`slot-spandec-strict`) instead
of `tul_slot_spandec_mask`, and their wandb names gain a `strict` segment:
`slot-spandec-strict-egrad-recon`, `slot-spandec-strict-egrad-disc`,
`slot-spandec-strict-norecur`. Nothing else about them changed — same energies, same
`pass_residual_lambda`, same `slot_depth_fixed: 1` control, same recipe.

Why, in one sentence: on the mask geometry the slot CELLS carry the cross-span information
the loop was supposed to carry, so an energy that conditions the loop's PASSES was being
scored on a forward where the loop is bypassed. The number behind it is already in this
arc's record — `slot-spandec-mask` at 5,000 steps, 480 rows: the whole slot channel
(`all_slots`) is worth **0.182** nats and the loop's own prefix write (`zero`) **0.078**,
so more than half of what the channel carries never passes through an iteration. Under
`tul.tg_geometry: strict` (`.agents/notes/proposed/architecture/2026-09-12-strict-slot-geometry.md`,
`lab/experiments/planned/2026-09-12-arc-strict-geometry.md`) the prelude is same-span only,
a coda prefix cell reads itself alone, and the loop is the only route from one span to the
next.

Consequences for how this file is read, stated so nothing is quietly re-scored:

* The one-factor partner of every arm below is now **`slot-spandec-strict`**, not
  `slot-spandec-mask`. Every paired CE, worth profile and K-curve comparison in
  **Readout, every arm** takes its control from that arm's artifacts.
* **No prediction in "Predictions (frozen)" is edited.** Two of them were written against a
  `slot-spandec-mask` baseline and their thresholds now sit against a different control:
  P-e (`ce_entry - ce_loop` above 0.02) and P-h (val CE within 0.05 of the parent). They
  stay exactly as written and are scored against the NEW parent, which is the honest cost
  of changing a control after freezing predictions — noted, not repaired.
* Step 0 (the linear probe gate) was run on `slot-spandec-mask@5000` and
  `slot-mux-norm-match@5000`. Those readings stand as they are; no strict checkpoint
  exists yet, so the gate has NOT been run on the geometry the arms will train under.

### The arms

Three configs, each ONE factor from `tul_slot_spandec_strict` (`slot-spandec-strict`), so
all three inherit the STRICT geometry (amendment 1 above), the span-decoder exit target,
`norm_match`, the Parcae-entry ruler recipe and the gain constraint.

**(1a) `tul_slot_spandec_egrad_recon.yaml`, arm `slot-spandec-strict-egrad-recon`.**
`tul.grad_pass: true` with `tul.grad_pass_energy: recon`. The energy is a SECOND
`SpanDecoder` with its own parameters (`morph/model/tul_egrad.py::ReconEnergy`,
`egrad_layers: 2`) that reconstructs the slot's OWN span from z, teacher-forced on that
span's tokens, with SOFT targets: the target at decoder position j is
`0.5 * onehot(t_j) + 0.5 * bag(the span's tokens)` — A\*-Thought-V2's Label Forcing
(arXiv 2609.07821). Cross-entropy is linear in the target, so that mixture is computed
EXACTLY as `(1−m)·FLCE(hard) + m·FLMCE(the span's token multiset)`, two calls to the tree's
own chunked kernels, so no `[B, S, J, V]` logit tensor is ever built.

The energy's gradient w.r.t. the slot state is detached, RMS-normalised, scaled by
`grad_pass_scale` 0.1 and mapped through the ZERO-INIT `W_g` — `TULGradPass` unchanged,
because the measured lesson is that the energy gradient is a CONDITIONING INPUT and not the
update (a scalar-energy gradient used AS the update measures weaker than a learned vector
transition conditioned on it). The energy decoder trains on a STOP-GRADIENT copy of z, so
its loss reaches its own parameters and stops; `W_g` is the only route into the loop.

New wandb keys: `loop/egrad_t{0..7}` (the energy's value at each pass — the ladder),
`loop/gp_rel_t{0..7}` (the injected term over the state's norm, unchanged from gradpass),
`tul/egrad`, `tul/egrad_train`, `tul/pass_residual`.

**(1b) `tul_slot_spandec_egrad_disc.yaml`, arm `slot-spandec-strict-egrad-disc`.** Identical
except `grad_pass_energy: disc`: `E = −s_phi(z, ctx)`, `s_phi` a 2-layer MLP on the
mean-stream z and the slot's prelude-entry state, trained with BCE against the Step-0 label
computed ONLINE, on a detached z, with shuffled-context negatives so the critic cannot
score from the context alone. Never a regression onto z. `tul/egrad_auc` is the critic's
own ROC-AUC on each training batch and is the arm's honesty instrument: a critic stuck at
0.5 means the energy carries nothing.

**(1c) `tul_slot_spandec_norecur.yaml`, arm `slot-spandec-strict-norecur`.** The LRT control:
`tul.slot_depth_fixed: 1`. Same parameters, same target, same reader, ONE core application
per slot instead of a Poisson draw. It separates "the loop iterates usefully" from "the
slot cell and its extra map are useful". No new code — `slot_depth_fixed` already exists
and is what every forced-depth probe drives.

**Both energy arms also get** `tul.pass_residual_lambda: 0.01` — a per-pass bound
`λ · mean_t ‖h_{t+1} − h_t‖² / ‖h_t‖²` over the loop's gradient passes, distinct from
`model.core_fixed_point_lambda` (which stays at the ruler's 1.0 and charges the SAME ratio
at each slot's LAST pass only). LRT's sweep is the source: no residual penalty and the
state drifts, 0.01 is best, 1.0 collapses the loop to its entry.

**`tul.reinject_seed_every_pass` is REFUSED, not built.** LRT reports −3.2 points for a
seed injected at init only, so re-injecting it every pass is a real lever elsewhere — and
MORPH already does it. `_tul_core` binds `_e_arg = e`, the prelude's output at the slot
position (i.e. `E_slot + W_sent·embed(t_last)` after the prelude has run over it), ONCE and
hands it to every pass; `_apply_core_step` opens with `self.injection(h_in, e_in)`, a
`DiagonalInjection` of that same `e` at every pass, and then adds the per-core-layer
x0/bigram terms, themselves gathered at the slot positions. A second additive copy would be
a duplicate path no measurement could separate from a change in the injection's gain. The
knob therefore raises at construction with that explanation, and
`tests/test_tul_egrad.py::test_the_reason_it_is_refused_is_true_the_seed_reaches_every_pass`
checks the claim rather than restating it.

**Second-order gradpass (`create_graph=True`) is NOT in this batch.** It is the next twin:
one Hessian-vector product per pass, the learned-optimiser objective of Andrychowicz et al.
2016, and the first follow-up if `egrad_t` keeps falling past t1 and the coda still reads
nothing.

### Cost, arithmetic and not a guess

One `[B, S, J, V]` head readout is 38.7 GFLOP per token of J at B 6, S 64, C 1024,
V 49169. The `recon` feature is read at EVERY pass and the soft target costs two readouts,
so the arm pays roughly 1.4 TFLOP × J against a ~50 TFLOP step. `egrad_max_tokens: 8` is
therefore ~22 %, and J = 32 would have been ~90 % — which is why the knob's default is 8
and not `bound_span_cap`, unlike `spandec_max_tokens`. 8 is also where the measured
cross-span budget stops being front-loaded. Expect 1.2–1.4× the ruler's wall clock for
`recon`. The `disc` energy has no vocabulary axis, so it is free; its LABEL costs one extra
no-grad per-token readout of the coda per step (~0.7 TFLOP, ~1.4 %), so expect within 1.1×.
**Every number in this paragraph is arithmetic. The smoke's tok/s decides.**

### Run

5,000 steps each, the arc's file-driven runner, a 21-step smoke first, the sustained
tripwire, rate floor 8,086 tok/s at step 200. Queue lines are written (NOT appended) to
`egrad_queue_lines.txt`; the runner is NOT started.

### Readout, every arm

The energy per pass (`loop/egrad_t`); the coda CE decoded after every pass (the K-curve at
every forced depth 0..6, `core_depth_sweep.py` at 2,500 and 5,000); `ce_entry − ce_loop`
(`slot_z_optimize.py`); the worth profile with `all_slots` (`worth_profile.py`); the
gradient probe (`slot_gradient_probe.py`, cancellation ratio and per-pass cotangent); the
per-pass step norm and angle (`slot_state_linear_probe.py`); and depth extrapolation at
9 / 12 / 16.

## Predictions (frozen)

Two probabilities per row: the orchestrator's, frozen before the build, and mine, written
after the CPU smoke of Step 0 and before any GPU step of any arm. Where they differ the
reason is the smoke, and it is named.

- **P-a (Step 0 above chance).** `slot-spandec-mask` exit AUC above 0.6 at 96 rows, on at
  least one reduction, clearing its own shuffle band. **Orchestrator 65 %. Mine 55 %.**
  Reasoning: the 6-row smoke reads 0.599 (mean) and 0.566 (flat) against a null p95 of
  0.560/0.577 — right on the line, and 306 slots is 16× fewer than the GPU run will have,
  so the point estimate will move. The ruler reading (0.68, clearly outside its band)
  says the quantity is measurable on this tree, which is why I do not go below 50 %.
- **P-b (the signal RISES with pass count).** Exit AUC above entry AUC by more than the
  shuffle band's half-width, on at least one reduction. **Orchestrator: asserted as part
  of P-a. Mine 20 %.** Reasoning: the smoke says the opposite on both checkpoints — flat
  on the ruler (0.677 → 0.682 flat, 0.655 → 0.667 mean) and falling on spandec-mask
  (0.598 → 0.566 flat). This is the prediction I most expect to come in FALSE, and it is
  the one that matters: a signal already present at the ENTRY and not improved by six
  passes is the same "one pass does all the work" finding as every other arm, measured on
  a new axis.
- **P-c (the reconstruction energy is descended in ONE pass).** `loop/egrad_t0 −
  egrad_t1` above 0.02 nats AND `egrad_t1 − egrad_t_last` below 0.02 — i.e. the gradpass
  ladder shape repeats on the harder energy. **Orchestrator 75 %. Mine 55 %.** Reasoning:
  I am lower because the whole point of the conditional decoder is that its minimiser is
  not in the entry state, and the arm would be uninformative if it behaved identically. I
  am still above half because the *mechanism* that produced the one-pass shape — `W_g`
  injecting one well-scaled direction, then the map settling — is unchanged, and because
  the exit target (`spandec`) already trains z toward a conditional decode, so the
  own-span conditional may be close to what z already carries.
- **P-d (the discriminative twin moves K3−K6).** `slot-spandec-egrad-disc` token K3−K6 at
  5,000 strictly above 0 and below 0.01. **Orchestrator 35 %. Mine 20 %.** Reasoning: no
  arm on this tree has ever moved K3−K6 past 0.002, including the three that moved the
  exit; and a critic's gradient is rank-1 in z per slot by construction (`∇_z (−s_phi)` is
  one direction), which is the LEAST curvature of any energy here. If the critic's AUC
  reads near 0.5 the arm is uninformative rather than negative, which is why `egrad_auc` is
  logged every step.
- **P-e (the write contribution).** `ce_entry − ce_loop` above 0.02 on at least one twin
  (ruler +0.0015, gradpass +0.0075). **Orchestrator 40 %. Mine 30 %.** Reasoning: gradpass
  moved this 5× and landed at 0.0075; another 2.7× is a bigger jump than any single factor
  has produced, and the quantity is bounded by what the coda can use, which
  `slot_z_optimize` puts at 0.9–2.6 nats for a FITTED z — so the ceiling is not the
  problem, the causal reachability is.
- **P-f (survival).** Both energy arms HEALTHY to 5,000, no sustained tripwire. **Both
  75 %.** Reasoning: `W_g` starts at zero, the loss is the parent's plus two terms that are
  bounded by construction, and the gain hinge and the fixed-point term are the parent's.
  The 25 % is for the one thing the apparatus does not cover and gradpass already named:
  the hinge treats the feature as EXOGENOUS, so a `W_g` that grows large adds gain the
  constraint cannot see — and this batch adds a second unhinged input on top of it.
- **P-g (cost).** `recon` within 1.5× the ruler's wall clock: **85 %**. Within 1.3×:
  **60 %**. `disc` within 1.15×: **80 %**. Reasoning: the FLOP arithmetic above, with no
  measurement behind it. The most likely way it breaks is launch overhead — T inner
  backwards per step through a 2-block decoder on the eager TG-scoped path — which the
  gradpass anchor (1.07× for ONE `[B, S, V]` readout per pass) does not cover.
- **P-h (the control earns its place).** `slot-spandec-norecur` val CE at 5,000 within
  0.05 of `slot-spandec-mask`'s. **Mine 60 %** (orchestrator gave none). Reasoning: the
  forced-depth sweeps of every slot arm on this tree read flat to 1e-3 from depth 1 to
  depth 16, so a model TRAINED at depth 1 should land in the same place. If it does not —
  if depth-1 training is clearly worse — that is the first evidence on this tree that the
  iteration is doing something the K-curve cannot see, and it reframes every flat K-curve
  already filed.
- **P-i (geometry, readings not bars).** The per-pass step norm decays and the successive
  step cosine settles. **Orchestrator's band for the cosine: 0.5–0.65.** The 6-row smoke
  already reads 0.82–0.94 on BOTH checkpoints, so I record the band as CONTRADICTED before
  the run and predict the GPU run reads 0.80–0.95 at **80 %**. Step norm decays
  monotonically: **90 %** (0.32 → 0.016 and 0.675 → 0.026 in the smoke).

## Not verified before launch

* **Nothing has trained under the strict geometry** (amendment 1). The arms' new parent
  `slot-spandec-strict` has itself never run a GPU step, so every prediction below is now
  a prediction about a baseline that does not yet exist as a number.

* **No arm here has run one GPU step.** The card was busy for the whole build window, and
  when it freed the arc queue still had `plain-coda-matched` pending — a 21-step smoke
  would have overlapped that arm's own rate check at step 200 and a contention dip below
  the 8,086 floor SKIPS it. The build is proved by CPU tests only: `pytest tests/ -q` reads
  **1151 passed, 9 skipped, 1 xfailed**, including 18 new tests in
  `tests/test_tul_egrad.py`, and six deliberate sabotages of the new mechanisms were each
  caught by exactly the test that claims to cover them.
* **One blocker was found by composing the configs and NOT by the test suite.**
  `TULConfig.__post_init__` refused `tul.grad_pass` at `mux_beta <= 0` unconditionally,
  which would have stopped all three arms at build: every one of them descends from
  `tul_slot_spandec_mask`, where `mux_beta: 0` because the span decoder REPLACES the MUX.
  The guard is now scoped to `grad_pass_energy == "own_mux"` — the only energy that IS the
  MUX head's loss — and the shipped `slot-mnext-gradpass` config still trips it as before.
  Recorded here because it is the shape of failure a CPU suite does not catch: every unit
  test built its own tiny `TULConfig` and none of them held the arms' real combination.
* **Memory at the real shape is arithmetic, not a measurement.** The `recon` arm's transient
  is one chunked `[chunk, V]` tile per CE call plus the kernel's `[V, d]` fp32 `grad_w`
  accumulator (201 MB), built and discarded ~36 times per step. On a 31.4 GB card that a
  desktop already uses ~6 GB of, with the parent arm's training peak near 12 GB, that
  should fit. Should is not measured.
* **`torch.compile` behaviour of an inner `autograd.grad` through a 2-block decoder inside
  the traced forward is untested at the real shape.** Gradpass proved the shape works for a
  single tied-head readout; a decoder with attention inside the same inner backward is a
  step further and the 21-step smoke is the first real evidence.
* **The energy is fitted at the loop's EXIT state and read at EVERY pass**, so the feature
  at pass 0 is an extrapolation. Fitting at every pass would roughly double the arm's extra
  FLOPs; the trade was taken deliberately and is stated in `_egrad_train_loss`'s docstring,
  not hidden. Whether it matters is unmeasured.
* **The `disc` negatives are an interpretation.** "Negatives also from z paired with a
  shuffled slot's label" admits more than one reading; what is built is `(z_s, ctx_{s−1})`
  at target 0 — a correspondence negative that stops the critic scoring from the context
  alone. A different reading would be a different arm.
* **The gain hinge does not bound the new feature**, for the same reason it did not bound
  gradpass's: its finite difference treats the injected term as a constant. Two unhinged
  inputs now compose (`recon`'s feature and, on these arms, nothing else — but the
  interaction of `pass_residual_lambda` with the hinge is also unmeasured).
* **`pass_residual_lambda` × `core_fixed_point_lambda` has never been run.** Both are on in
  these arms, at 0.01 and 1.0, and CPU tests prove only that they are distinct terms.
* **Step 0's CPU smoke is 6 rows.** 306 slots, CIs of ±0.06. Every number in the smoke table
  is a smoke number. The ruler's ~0.68 is the only reading in it that clears its own null
  band with room.
* **The `slot-mux-norm-match` checkpoint needs `model.core_hca_compress_ratio=256` to
  load**, which means the ruler's Step-0 reading is taken on a model whose HCA compressed
  branch was DEAD on three core blocks. That is what the ruler was trained with, so it is
  the right comparison for arms of its generation and the WRONG one for anything built on
  today's `tul_short.yaml`. Named because it is easy to quote across that boundary.

## Results

(pending)

### Step 0, the gate (2026-09-12 03:32 CDT; 96 rows, batch 3, depth 6, 20 shuffles; JSON in `lab/experiments/results/2026-09-12-latent-z-gradient/`)

Split point bit-exact on all three checkpoints (|d| 0.0). 4,906 scored slots each, 0.500
positive. AUC as flat / mean reduction, then the shuffled-label null band p95:

| checkpoint | entry (t0) | pass 1 | exit (t6) | null p95 |
|---|---|---|---|---|
| slot-spandec-mask@5000 | 0.6045 / 0.6109 | 0.6103 / 0.6384 | 0.6082 / 0.6368 | 0.513 / 0.514 |
| slot-mux-norm-match@5000 (ruler, `core_hca_compress_ratio=256`) | 0.6128 / 0.6331 | 0.6028 / 0.6346 | 0.6041 / 0.6285 | 0.513 / 0.514 |
| slot-mux-mask-norm-match@5000 | 0.6047 / 0.6419 | 0.6105 / 0.6565 | 0.6130 / 0.6482 | 0.514 / 0.519 |

Geometry on spandec-mask (stream mean): |z| 26.3 → 35.7, step_rel 0.311 / 0.090 / 0.046 /
0.029 / 0.022 / 0.017 (monotone), step_cos 0.83 / 0.92 / 0.93 / 0.94 / 0.94, eff_rank
17-19, isotropy 0.017-0.018, hoyer 0.45-0.49. The ruler: step_rel 0.65 → 0.030, step_cos
0.83 → 0.71, eff_rank 22-23.

**P-a TRUE** (exit AUC above 0.6 on both reductions, clear of the band by 0.09-0.12).
**P-b**: on the mean reduction the AUC rises 0.611 → 0.638 (+0.027, above the band's
half-width ~0.017) and the whole rise happens at pass 1; passes 2-6 add nothing (0.638 →
0.637). On the flat reduction +0.004, inside the band. TRUE on one reduction by the letter,
and the shape is the one-pass shape every other instrument reads. The ruler FALLS
0.633 → 0.629 and the mask ruler rises 0.642 → 0.648, both inside the band.
**P-i**: the builder's band (0.80-0.95) TRUE, the orchestrator's (0.5-0.65) FALSE; step norm
decays monotonically on all three (TRUE).

### The 21-step cost smokes (2026-09-12 03:34 CDT, frozen worktree at `b2b9198`, bypass geometry)

`recon`: exit 0, 7,584 tok/s at step 20 (the ruler's smoke reads ~10.8k: **1.43x** wall),
peak 15.45 GB, loss/total 33.48 = token CE + spandec 11.16 + egrad 11.18, preclip/total 43,
`gp_rel_t0` 1e-4 (zero-init `W_g`). `disc`: exit 0, 10,827 tok/s (**1.0x**), peak 14.48 GB.
P-g: recon inside 1.5x (TRUE at the smoke rate, not inside 1.3x); disc inside 1.15x (TRUE).
A 21-step rate is not a 5,000-step rate.

**Arms.** `slot-spandec-norecur` on the ORIGINAL (bypass) geometry started 03:35 at
`b2b9198` to use an idle card (control: `slot-spandec-mask`). The strict-geometry versions
of all three arms are queued behind it at `a0b72c4` (amendment above).

## Verdict

(pending)
