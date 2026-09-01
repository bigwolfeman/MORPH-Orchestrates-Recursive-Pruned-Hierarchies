# TUL v1 divergence — root cause analysis (2026-08-17)

Both TUL arms of the first short-schedule pass diverged. `A0` (dense baseline) and
`A3` (`n_core 0`, prelude+coda only) completed 20 000 steps. This note records what
is **measured**, what is **inferred**, and the experiments that separate the two.

Everything below is CPU-only forensics on the saved checkpoints and the wandb
history. No GPU was available while writing it, so no hypothesis here has been
tested by a training run.

## 1. What happened

| arm | config | result | final val loss |
|---|---|---|---|
| A0 | `tul_a0` | completed 20 000 | 3.2736 (ppl 26.41) |
| A3 | `tul_a3` | completed 20 000 | 3.2407 (ppl 25.55) |
| A1 | `tul_a1` | **ABORT step 4540** | — (val 6.409 @4500) |
| A1r | `tul_a1r` (seed 1) | **ABORT step 3240** | — (val 6.674 @3000) |

The divergence guard fired on both TUL arms: two consecutive eval points with train
ppl over 1000. Forensic checkpoints `DIVERGED_step_4540.pt` / `DIVERGED_step_3240.pt`
hold the failed state.

A1's val curve is *not* a slow drift. It improves through step 1500 (4.847, better
than A0 at 500 and 1000) and then climbs monotonically to 6.409.

Two seeds fail the same way. This is structural, not seed luck.

## 2. The decisive number

`val/ppl_tokens` — token positions only, directly comparable to the baseline:

* A3, prelude+coda with **no core at all**: **25.55**
* A1, prelude+coda **plus** the slot core: **754.67** at step 4500 and rising

A1's token path is the same 8 layers as A3 plus slot positions in the sequence. It
ends 30× worse than deleting the core entirely. The slot core is not failing to
help — it is actively destroying the token path it feeds.

## 3. Where the damage is — measured

`_cms.block_score_ema` is an EMA of per-tile ‖W ⊙ ∇W‖_F
(`morph/model/layers/block_sparse.py:480-483`), recorded per module inside every
checkpoint. It is a gradient magnitude, stored by name. At (near) iso-step:

| module | A1 @4540 | A1r @3240 | A0 @5000 | A3 @5000 |
|---|---|---|---|---|
| prelude.0 mlp.down | 2.2e7 | 1.3e8 | 2.89 | 6.71 |
| prelude.3 mlp.down | 2.8e6 | 8.2e7 | 1.68 | 3.14 |
| **core.0 mlp.down** | **1.1e8** | **3.5e8** | **0.976** | — |
| core.1 mlp.down | 3.1e7 | 1.7e7 | 0.998 | — |
| core.2 mlp.down | 3.1e6 | 1.2e7 | 0.913 | — |
| core.3 mlp.down | 1.8e6 | 4.1e6 | 1.10 | — |
| core.4 mlp.down | 2.5e5 | 1.9e6 | 1.17 | — |
| core.5 mlp.down | 3.8e4 | 2.4e5 | 1.06 | — |
| coda.0 mlp.down | 15.9 | 9.71 | 2.58 | 3.35 |
| coda.3 mlp.down | 12.4 | 7.05 | 3.85 | 4.63 |

Read it backward, the direction gradient flows: **coda ≈ 5× the baseline, then the
core amplifies ≈ 5× per layer** (core.5 → core.0 is 3.8e4 → 1.1e8, a factor of 3000
across six layers), and the flood continues into the prelude at 1e7.

A0's core profile over the same six layers is **flat**: 1.06, 1.17, 1.10, 0.91,
1.00, 0.98. Same weights, same depth schedule, same optimiser. The only thing that
changed is what the loop runs on.

**Ruled out — quantiser artefact.** The Taylor score is `|W ⊙ ∇W|` with `W` the
*effective ternary* weight, so an inflated quantiser scale would fake this. It did
not: the core.0 shadow weights are `0.0163` (A1) against `0.0219` (A0) — smaller,
not larger. The 1e8 is gradient.

## 4. What the explosion did to the rest of the model — measured

`training.grad_clip = 1.0` was active for every arm. With a global norm of order
1e8, the clip factor is of order 1e-8 on every step of both TUL arms.

Comparing A1 @4540 against A0 @5000, every parameter region matches within 1.12×
**except** the injection-scale parameters:

| parameter | A1 | A1r | A0 | A3 |
|---|---|---|---|---|
| `embed.hybrid.euc_embed` (token table) | 0.0379 | 0.0378 | 0.0403 | 0.0399 |
| `embed.bigram.lambdas` | **0.107** | **0.149** | 0.771 | 0.672 |
| `x0_injects.1.log_scale` | **0.100** | **0.124** | 0.575 | 0.701 |
| `x0_injects.2.log_scale` | **0.149** | **0.146** | 0.579 | 0.695 |
| `tul.W_prefix` ‖W−I‖_F | 7.36 | 6.66 | — | — |
| `tul.E_mask` rms | 0.0024 | — | — | — |

The token embedding table is **fine** (within 6 %). What stalled is the family of
small, slow-growing scale parameters that control how strongly `x0` and the bigram
signal are injected into each block. A0 and A3 both grow them to ≈0.6–0.7; both TUL
arms sit at ≈0.10–0.15.

`W_prefix` barely moved from its identity init (diagonal 0.995, off-diagonal rms
0.007), so the prefix projection is not the amplifier. `E_mask` never left zero.

**Inferred, not measured:** that the sustained ~1e-8 clip is what stalled those
scales. Adam-family updates are invariant to a *uniform* rescale, so the mechanism
has to be a floor effect — the optimiser epsilon, or `ademamix_g_snr_gate_kappa`
— and neither was instrumented. `optim/snr_gate_*` never appeared in either run's
history, which argues against the SNR gate but does not settle it.

## 5. Why the core backward explodes — NOT established

This is the open question. The core is a weight-shared recurrence, 6 layers × 8
iterations with BPTT over the last 4. In A1 it runs, for the first time, on a
**64-position compact slot sequence** instead of 1024 token positions, and about
22 % of those positions (`tul/spans` ≈ 50 of `max_slots` 64) are **exact zeros**
from `gather_valid`.

Ranked candidates and the experiment that kills each:

1. **The loop is unstable on a short sequence.** Attention over 64 positions is far
   less contractive than over 1024, so the per-iteration Jacobian is larger and 4
   BPTT steps compound it. *Test:* run **A0 unchanged at `data.seq_len 64`** for 500
   steps and read `train/grad_norm`. This uses no TUL code at all. If A0 at 64 also
   explodes, TUL is exonerated and the finding is about the looped core itself.
2. **Zero pad slots.** RMSNorm's Jacobian at an exactly-zero input is 1/√eps — an
   amplifier of ~1e3 per traversal. Pads sit at the tail of the compact sequence so
   causal attention should keep them off the loss path, but `retention_carry=true`
   carries GLA state from the END of iteration *t* into position 0 of iteration
   *t+1*, which is a path from pads back to real slots. *Test:* `retention_carry
   false`, and separately seed pads with a nonzero constant.
3. **Depth schedule interaction.** `_sample_slot_depths` draws 14×64 = 896 samples,
   so `total_iters` is 8 essentially every step, against A0's max over 14 samples.
   *Test:* `bptt_depth 1`.

`core_gain_clip` — the L1 governor written for exactly this failure mode ("the β1=0
gain runaway mode", `transformer.py:227-232`) — was **0.0, i.e. off**, on every arm.
Turning it on (τ ≈ 1.5) is the cheapest mitigation to try, but it treats the
symptom; run experiment 1 first, because it tells us whether this is a TUL result
or a MORPH result.

## 6. Defects fixed in this commit

1. **An aborted run exited 0.** `train.py` returned normally after the divergence
   guard, so the queue read "exit=0" and started the next arm on a dead result. Now
   `wandb.finish(exit_code=1)` and `raise SystemExit(4)`.
2. **The queue trusted the exit code alone.** `run_tul_arms.sh` now also requires the
   arm to have printed `Final val_loss`, and stops with exit 5 if it did not.
3. **The pre-clip gradient norm was computed and thrown away.**
   `clip_grad_norm_` returns it; it is now logged every 20 steps as
   `train/grad_norm`, with `train/clip_factor` beside it.
4. **No per-region gradient visibility.** Added `gradnorm/<region>` every 100 steps
   (post-clip, so ratios between regions are exact and absolutes are recovered by
   multiplying by `train/grad_norm / grad_clip`).

Without (3) this diagnosis needed a checkpoint autopsy of a saliency buffer. With
it, the failure is one glance at a chart.

## 7. What this does NOT say

* Nothing here tests the TUL *hypothesis*. Whether a plan slot helps is untouched —
  the arms never reached a state where `val/plan_nats` means anything. It sat inside
  ±0.04 for the whole run, which is what "the plan channel carried nothing" looks
  like, but under a 1e-8 clip that is not evidence about the design.
* The chain in §4 (clip → floor → stalled scales) is inferred. The gradient
  explosion in §3 is measured.
* Section 5 is ranked speculation. No experiment has been run.
* The A1/A1r comparison is not a noise floor. Both arms failed, so the spread
  between them measures two failures, not run-to-run variance.

## 8. Verification

* `pytest tests/ -q` on CPU: **114 passed, 2 skipped** (the 2 skips are CUDA-only;
  the same suite is 116 passed on GPU).
* `bash -n ignore/tul_logs/run_tul_arms.sh`: clean.
* **Not verified:** none of the four fixes in §6 has run inside a live training
  step. `train/grad_norm`, `train/clip_factor`, `gradnorm/*` and the non-zero exit
  path all need one GPU run to confirm. The `gradnorm/*` block in particular walks
  `named_parameters()` under `torch.compile`, and the name-stripping is reasoned,
  not observed.

---

# Part 2 — the trigger, measured (2026-08-17, GPU)

Part 1 was a checkpoint autopsy with the GPU unavailable. §5 ranked three candidate
causes. **All three are wrong.** The trigger is `tul.token_state_dropout`.

## 9. The prior art was ours

`00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies/Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/`
(Task #276, June 2026) already characterised this failure for the looped core:

> optimizer coherence (β1=0 + heavy slow-EMA α) → the six core blocks' top singular
> subspaces rotate into **alignment** → composition σ_max(J_core) runs away → detonation.

Order parameter = alignment ratio (composition σ_max ÷ worst single block): **0.90
healthy, 0.91 AdamW-stable, 9.5 at the cliff.** Non-aligned blocks cancel; aligned
blocks chain multiplicatively. That campaign ruled out, with runs: `core_gain_clip`
(masks), per-linear spectral penalty (detonated, idle 0/12), loop-STP (harmful),
per-coordinate `update_clip` ("the disease is directional; a per-coord magnitude cap
is the wrong instrument"), `stale_push_cap` (delays ~300 steps).

The shipped mitigation is `ademamix_alpha_cap = 3.5`, sized from an α sweep where α≤4
survived and α=6 detonated. **Every TUL arm ran with the full cure stack** —
`alpha_cap 3.5`, `update_clip 5.0`, `g_snr_gate_kappa 0.3`,
`stale_push_cap_coord 0.5` — and detonated anyway.

## 10. The measured trigger

Batch 6, 800 steps, one flag apart. `core share` = `gradnorm/core` ÷ the norm over all
regions, at steps 100 / 200 / 300 / 500 / 700:

| arm | flag | core share | grad_norm | val @400 |
|---|---|---|---|---|
| D8 | `token_state_dropout=0.15` | 0.009 → 0.056 → **0.866** → 0.994 → **0.999** | → 6.4e5 | 6.62, ABORT @2040 |
| E9 | `token_state_dropout=0.0` | 0.008 → 0.037 → 0.016 → 0.014 → **0.008** | **2.1–2.8 flat** | **6.26** |
| E0 | A0 baseline, no TUL | 0.19 → 0.25 → 0.23 → 0.26 → 0.19 | 1.8–2.8 flat | 6.14 |

With dropout off the TUL arm is stable and tracks the baseline. With dropout at 0.15
the core owns 87 % of the gradient by step 300 and 99.9 % by step 600, `grad_clip=1.0`
then divides everything by 1e5–1e9, and every other region stops learning (Part 1 §4).

**Mechanism.** §3.4 dropout replaces 15 % of token coda inputs with `E_mask` AND zeroes
their x0/bigram injection, so those positions must be reconstructed from context — and
the slots are the most informative thing left. That aims a large COHERENT gradient at
the slots, which are the only path into the looped core. A coherent directional push on
the core is exactly what Task #276 measured as the thing that rotates the blocks into
alignment, and exactly what the per-coordinate cure stack cannot catch.

**The uncomfortable part:** the mechanism designed to force the model to USE the plan is
the mechanism that detonates the loop. At 400 steps E9's `plan_nats` is +0.0018 — with
the pressure removed the plan carries nothing. 400 steps is far too early to conclude
that, but it is the tension the design has to resolve.

## 11. Refuted

* **Short sequence in the loop (§5 candidate 1).** A0 at seq 64 with tokens/step held at
  14336 has a HIGHER core share (0.232) than at 1024 (0.145) and a LOWER peak norm.
  The looped core is fine on a short sequence, and MORPH's machinery is exonerated.
* **Quantisation.** `ternary=false embed_quant=off` (D3/E3) and `adam8bit=false`
  (D7/E7) change nothing about the takeover.
* **§5 candidates 2 and 3** — `retention_carry=false`, `bptt_depth=1` — no effect.
* **A0 is not on the same trajectory.** Its core gradient grows 1.0 → 15 → 42 → 165
  over 5k → 20k steps, but FLAT across core.0–core.5 (117–221 at 20k). Uniform growth
  from growing weights, not the graded backward amplification A1 shows (1.1e8 at core.0
  vs 3.8e4 at core.5).

## 12. Not verified

* One run per condition, no seed replicate. The separation is three orders of magnitude,
  which is not the same as replicated.
* The alignment ratio itself has NOT been measured on any TUL checkpoint. The
  fingerprints match Task #276 but the order parameter has not been read.
  `E9_diag_sigmas.py` / `E10_localize_sigma.py` in the b1zero repo are the instrument.
* Whether TUL trains to a useful `plan_nats` with dropout off, or with a smaller p, is
  untested. 800 steps says nothing about quality.

---

# Part 3 — WITHDRAWN: Part 2's trigger claim (2026-08-18)

**Part 2 §10 is wrong. Read this before acting on it.** It concluded from a single
A/B (D8 with dropout vs E9 without) that `tul.token_state_dropout=0.15` is the
trigger. The rest of sweep 2 refutes that reading.

## 13. What the full sweep showed

Batch 6, 800 steps, one override each. `core share` = `gradnorm/core` ÷ the norm over
all regions, at step 700:

| arm | override | core share | grad_norm |
|---|---|---|---|
| **D8** | **none (unmodified)** | **0.999** | **3.6e5** |
| E9 | `token_state_dropout=0.0` | 0.008 | 2.70 |
| E5 | `core_gain_clip=1.5` | 0.021 | 2.63 |
| E4 | `retention_carry=false` | 0.021 | 2.55 |
| E6 | `bptt_depth=1` | 0.008 | 2.93 |
| E3 | `ternary=false embed_quant=off` | 0.040 | 1.68 |
| E7 | `adam8bit=false` | 0.008 | 2.68 |
| E10 | `prefix_k=1` | 0.024 | 1.87 |
| E0 | A0 baseline, no TUL | 0.190 | 2.61 |

Every arm is stable except the ONE that changed nothing. E9 is not special.

D8's and E5's frozen wandb configs differ only in `core_gain_clip`, `steps` and
`eval_every`. `steps` is inert for the trajectory here (`warmup=0` and
`min_lr == lr`, so LR is flat 1e-4; TST is off; prune/route never fire), and the
takeover occurs at step 300, before either run's first eval. So at the moment it
happens, each E arm differs from D8 by its override alone.

## 14. Why "every intervention worked" is not eight cures

`ternary=false` and `adam8bit=false` are not plausible cures for a subspace-alignment
instability, but both change the numerical trajectory. The consistent reading is that
the takeover is a **stochastic trajectory event that any perturbation avoids in a
single run** — which is exactly how Task #276 described it: a single-iteration
transient spike at a VARYING loop index, appearing at a rate that climbs over
training. One run per arm cannot separate "this override cured it" from "this
override moved the dice".

**Methodological error, named:** an intervention sweep with n=1 per arm and no
replicated control measures trajectory sensitivity, not causation. The control
should have been replicated FIRST, to get the base rate, before any arm was read.
Compare `[[measure-the-noise-floor-before-the-conditions]]` — the same mistake,
in a new place.

## 15. What survives Part 3

* The mechanism (Part 1 §3, Part 2 §10 first paragraph): the core takes ~99.9 % of
  the gradient norm, `grad_clip` then divides everything by 1e5–1e9, and every other
  region stops learning. Measured live in D8 and independently in the checkpoints.
* The per-core-layer amplification profile (~5×/layer, core.5 → core.0).
* **MORPH is exonerated** (Part 2 §11): A0 at seq 64 has a HIGHER core share than at
  1024; A0's core profile is flat across 20k steps; E0 is stable at batch 6.
* The original failure is real and seed-replicated at batch 14: A1 died at 4540 and
  A1r (seed 1) at 3240.

## 16. Running now

`ignore/tul_logs/run_tul_seeds.sh` — 4 seeds × {control, dropout-off}, batch 6, 1200
steps, **eval disabled** so an eval pass cannot perturb the RNG stream and act as a
hidden variable. It measures a RATE (how many of 4 seeds take over) against a rate.
No causal claim about any intervention should be made until that base rate exists.

---

# Part 4 — the long run, prepared 2026-08-18 (NOT started)

## 17. Why the 2026-08-17 diagnostics could not settle anything

30 runs, 26 040 steps, and the failure was reproduced ONCE. Counts:

| sweep | runs | steps each | batch |
|---|---|---|---|
| 1 (D0–D7) | 8 | 500 | 14 (D1 224) |
| D8 trace | 1 | 2040 of 6000, ABORT | 6 |
| 2 (E0–E10) | 9 | 800 | 6 |
| 3 (F0–F3) | 4 | 800 | 6 |
| seeds (G/H) | 8 | 1200 | 6 |

**A1 dies at step 4540 and A1r at 3240.** Every diagnostic was shorter than the
time-to-failure. Sweep 1 probed before the event; sweeps 2 and 3 probed a rare event
at a different operating point; the seed sweep was the only sound design and returned
**0/8** (0/4 control, 0/4 dropout-off) at 1200 steps, batch 6.

`G-a1-b6-seed0` is the same config and seed as D8 and did NOT take over, while D8 hit
0.866 core share at step 300 — before either run's first eval, where they should be
identical. **The run is not reproducible at fixed seed.** Likely `bag_mean`'s
`index_add_` (CUDA atomics, nondeterministic for floats); NOT verified.

Two of four control seeds showed real mid-run `grad_norm` spikes (7.7e4, 4.9e4) that
did NOT run away — Task #276's phenotype exactly: transient excursions whose rate
climbs, occasionally catching.

## 18. The prepared queue

`ignore/tul_logs/run_tul_long.sh` (gitignored; definition duplicated here so it
survives). Read it with `ignore/tul_logs/summarize_long.sh`.

* **batch 14, 5000 steps** — the point where the failure is 2/2 (4540, 3240), and past
  both. ~45 min per run.
* **2 seeds per arm, always.** The outcome is bimodal, so n=1 measures trajectory
  sensitivity. Part 3 is what happens otherwise.
* **Control runs FIRST, both seeds.** If the queue is stopped early, what survives is
  "does this still reproduce at HEAD" — the result every other arm is read against.
* Peaks ~24.1 GB allocated / ~26.7 GB reserved: with a desktop it sits near 31 of
  32 GB. **Away-from-keyboard only** [W].

| arm | override | question |
|---|---|---|
| L0 | *none* | does the failure still reproduce at HEAD? |
| L1 | `tul.token_state_dropout=0.0` | the coherent gradient aimed at the slots (§3.4) |
| L2 | `model.core_gain_clip=1.5` | the governor; #276 called it symptom-masking on the TOKEN core, untested on the slot core |
| L3 | `training.ademamix_alpha_cap=1.0` | #276 mapped a sharp α horizon; TUL may sit below 3.5 |

Verified by composition, one variable each: L0/L2/L3 keep `drop=0.15`, only L1 sets
0.0; only L2 sets `gclip=1.5`; only L3 sets `acap=1.0`; all are seq 1024, batch 14,
`activate_at=0.0`.

Decision rule, pre-registered: **an arm counts as a cure only if BOTH seeds survive
5000 steps while BOTH control seeds take over.** Anything else is one more dice roll.

# Part 5 — the long run, RESULT (2026-08-18)

Ran 04:38–10:39 UTC, 8 runs, 6 h 1 min, at HEAD `2d6ec91`. Full table and the raw
per-step trajectories: `Ai-notes/08-18-2026/TUL-Long-Run/RESULTS.md`.

## 19. The failure reproduces, twice, and it is not the abort

Both control seeds took over. Onset step 2500 (seed 0) and 2100 (seed 1). They end at
`grad_norm` 1.08e7 and 2.83e7 and at loss 5.51 and 5.66. **Both ran to step 4800 and
exited 0** — the divergence guard never fired. So the guard's aborts at steps 4540 (A1)
and 3240 (A1r) were this same ratchet caught further along, not a separate event. The
takeover is the disease; the abort is a late symptom of it.

Part 2 §10's measurement is confirmed at the arms' own operating point: the core takes
essentially the whole gradient norm (share_end 0.9995 and 0.9918), and `grad_clip=1.0`
then starves every other region.

## 20. All three interventions survived, which is why this is not a mechanism yet

`token_state_dropout=0.0`, `core_gain_clip=1.5` and `ademamix_alpha_cap=1.0` each
survived 0/2 with end `grad_norm` between 0.85 and 1.58 and end loss between 3.69 and
4.17. Seven orders of magnitude of separation from the controls, on every seed, needing
no threshold to see.

They cannot all be the mechanism — one changes the forward pass, one clips the core's
per-iteration gain, one halves the slow-EMA weight. This is the shape Part 3 withdrew a
claim over. The live alternative is that the takeover is a knife-edge and any
perturbation steps off it. Part 4's decision rule is therefore **met by all three arms
and is not sufficient**; §21 is the missing control.

## 21. The threshold, read honestly

Part 4 pre-registered "takeover = core share > 0.5" without saying *ever* or *sustained*.
Under `any()` the arms are 1/2, 1/2, 0/2; under *sustained* they are 0/2, 0/2, 0/2. The
two flipped points are single samples: `L1-seed0` is 0.655 at step 2100 with 0.001 and
0.0014 on either side, `L2-seed0` is 0.989 at step 100 with 0.000 and 0.0003 on either
side. A control, for contrast, holds 24–29 consecutive points with a monotone onset and
never returns.

The rule is now "**≥ 3 consecutive logged points over 0.5**", and it is implemented in
`summarize_long.sh` rather than applied by hand. This choice was made **after** seeing
the data — that is a real cost and it is recorded, not buried. What limits the damage:
the run-length counts are 0, 0, 0, 0, 1, 1, 25, 29, so nothing sits near the boundary,
and both threshold-free facts in §19 point the same way.

## 22. Next: the placebo queue, prepared, NOT started

`ignore/tul_logs/run_tul_placebo.sh`, 3 arms, 6 runs, ~4.5 h, cheapest kill first.

* **P2 (first): a byte-identical replicate of the control at the same seeds 0 and 1.**
  Does the config decide the outcome, or the run? In doubt since 2026-08-17, when
  `G-a1-b6-seed0` shared `D8`'s config *and* seed and behaved differently. If P2 is not
  2/2, nothing in §19–§20 is readable and determinism gets fixed first.
* **P0: control at fresh seeds 2 and 3.** Is 2/2 takeover seed-robust?
* **P1: placebo, `token_state_dropout` 0.15 → 0.145.** Same mechanism at 96.7 % strength,
  so it cannot be a cure. If it survives, survival costs nothing but a nudge.

Pre-registered before the runs: P2 2/2 **and** P0 2/2 **and** P1 2/2 → the three cures
are mechanisms. P1 0/2 → trajectory sensitivity and the long run is void as a mechanism
test. P2 under 2/2 → the config does not determine the outcome and no arm is readable.

## 23. Still not verified

* That the control replicates. Everything above rests on it. P2 asks it.
* That any arm prevents rather than delays. Onset moved 2100 → 2500 across two control
  seeds; 4800 steps does not prove "never".
* The loss gap is measured against a diverged control, so it shows the arms are
  healthier, not that any of them is good.

# Part 6 — finishing the original comparison (2026-08-18)

The first arm pass got A0 = 3.2736 and A3 = 3.2407 at 20k steps and lost A1 twice to
the takeover. Part 5 found `ademamix_alpha_cap=1.0` survives 0/2 with the cleanest trace
of anything tried — core share peaked at 0.168 and 0.040, never crossing 0.5 at either
seed. Second pass: `ignore/tul_logs/run_tul_arms2.sh`, A0c then A1c, both at cap 1.0.

## 24. Why both arms re-run, and why not the dropout fix

The cap is an **optimiser** setting, not a TUL knob. A1 at 1.0 against the stored A0 at
3.5 would differ by two things and a win could be the optimiser. Both arms therefore run
at 1.0 and differ by `tul.activate_at` alone (`never` vs `0.0`) — checked by composing
both configs. The stored A0 at cap 3.5 stays as the reference for what the cap cost the
baseline, and is **not** the comparison.

`token_state_dropout=0.0` was the tempting fix: it is TUL-internal, so A0 would not have
needed a re-run, and it gave the lowest loss of the three surviving arms. It was
rejected. Spec line 534 gives the dropout's whole purpose as "tax the cheap channel or
the latent is ignored". Removing it is the one intervention that could let A1 post a good
`val_loss` with the plan doing no work — the exact failure the knob exists to prevent.
Dropout stays at the spec's 0.15 in both arms.

## 25. How to read it, fixed in advance

Per spec §7.1, on `val/plan_nats` and not on `val_loss` alone: **A1c must clear the A3
floor**, and it must beat **A0c** — not the stored A0 — for the plan to have earned its
compute.

## 26. The risk this run carries

The cap is shown to hold for 4800 steps at two seeds. These runs are 20000 steps. If the
takeover was only delayed, A1c dies later, the trainer exits non-zero with `[ABORT]`, and
the queue stops rather than spending five more hours on a dead comparison. A late abort
would be a result — it would say the cap postpones and does not prevent, and it would
retire the cap as a cure.

---

# Part 7 — the placebo ran (2026-08-22)

§22 prepared P1 and marked it NOT started. It ran on 2026-08-22, at both seeds, as
`ignore/perf/div_placebo.sh` (the `ignore/tul_logs/` script from §22 is not in this
worktree; the arm is the same: `tul_a1` with `tul.token_state_dropout=0.145`).

## P1 diverges 2/2

| seed | sigma crosses 3.0 | grad_norm detonates | val CE minimum | end |
|---|---|---|---|---|
| 0 | ~step 1950 | ~2900 (2.4e6 @3200) | 4.173 @2500 | sigma 6.79 @6000, wall clock |
| 1 | ~step 750 | by ~800 | 5.470 @500 | sigma 5.92 @2000 |

Seed 1's `first_tok` goes 4.47 -> 7.84 and `cf` -0.16 -> -3.71 between steps 500 and
1000 — the same signature every control shows.

**The §22 decision rule is met: P1 2/2 means survival is not free.** An intervention at
96.7 % of the original strength, which cannot be a cure, moves the onset — later at seed
0, EARLIER at seed 1 — and never prevents the failure. The four surviving arms are
therefore not merely perturbations that stepped off a knife-edge, and §20's worry, while
correct to raise, is not what is happening.

## Standing of the interventions

| arm | n | outcome |
|---|---|---|
| control | 5 | diverged 5/5 (abort steps 2080, 3240, 4540, 5900, 6200) |
| **placebo, dropout 0.145** | **2** | **diverged 2/2** |
| `token_state_dropout=0` | 2 | survived |
| `core_gain_clip=1.5` | 2 | survived |
| `ademamix_alpha_cap=1.0` | 2 + `tul-a0-acap1` / `tul-a1-acap1` at 20k | survived |
| `spectral_penalty cap=2.0 lambda=10` | 1 | survived 6600 steps, end grad_norm 0.77 |

## sigma is a correlate, not the trigger

New this pass: `spec/sigma_max` is now logged live on every run (see the entry in
`morph/training/train.py` next to the per-region gradnorm block; with `lambda=0` the
penalty early-returns an exact zero, so unpenalised arms stay bit-exact). Calibration:
1.41 at init against `base.yaml`'s documented healthy ~1.5, and 11.1 on
`tul-a1/DIVERGED_step_5900.pt` measured on the EFFECTIVE (ternarised) weight.

On the control, grad_norm went 0.95 at sigma 2.73 to 3.4e5 at sigma 3.89 within 100
steps, which looked like a threshold at sigma ~= 3.0. **The placebo refutes that
reading**: seed 0 crossed 3.0 at ~1950 and did not detonate until ~2900, seed 1 crossed
at ~750 and detonated within ~50. The lag is not constant, so crossing 3.0 does not gate
the failure. sigma tracks the takeover; it does not cause it. Consistent with §9 — the
quantity that runs away is the COMPOSITION `sigma_max(J_core)` under subspace alignment,
which per-linear sigma bounds only loosely.

## What is still open, and the instrument it needs

The four cures change four different things — a forward pass, a per-iteration gain clip,
an optimiser momentum weight, a weight-norm bound. They still cannot all be the
mechanism, and more surviving arms will not separate them. The discriminator is the
Task #276 order parameter — composition `sigma_max(J_core)` / worst single block, 0.90
healthy and 9.5 at the cliff — measured LIVE rather than from checkpoints. That is a
direct extension of the sigma logging now in place: the same power iteration applied to
the composition instead of to each linear.
