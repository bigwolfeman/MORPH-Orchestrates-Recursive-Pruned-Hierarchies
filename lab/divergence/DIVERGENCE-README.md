# READ ME IN CASE OF STRANGE DIVERGENCE

**If a MORPH run is blowing up, stalling, or turning around, read this before touching a
knob.** Two campaigns have already burned weeks on this. Almost every idea you are about to
have is in one of the two indexes below with a number next to it.

Last updated 2026-09-04 10:00 (section D: the clip-through-time result and the phase-2 arms).

## Triage in 60 seconds

Open the run's `probe.jsonl` (written by `training.grad_probe_every=1`, key
`preclip/total`) and answer three questions:

| question | if yes | if no |
|---|---|---|
| Did `preclip/total` exceed **1e4** at any step ≥ 200? | **Paid-axis detonation.** Section A. The run is dead; the loss may look fine for another 1500 steps. | keep reading |
| Is val CE at a MINIMUM and then RISING by nats, on a slot-loop TUL arm (A1/A3, ≤64 looped positions)? | **Core takeover.** [takeover-campaign.md](takeover-campaign.md), 15 hypotheses, 11 refuted. Do not re-derive them. | keep reading |
| Is the run a slot-loop arm with a FORECAST loss (MUX-next, cond4, coda8), under the 1000-step ramp, and does `preclip/total` show single-step spikes that snap back and escalate once the ramp ends? | **Forecast spike train.** Section D. Ternary off removes it; the loop's typical gain has drifted to 1. | keep reading |
| Is val CE flat near 7.3 from the first eval? | The codebook is pinned (a frozen ternary γ, or an equivalent). Section C. | New. Write a prereg before you run anything. |

The step-0 spike (`preclip/total` ~3.6e4 at step 0) is init and is NORMAL. Every verdict
here starts at step 200.

## A. The paid-axis detonation (2026-09, the winner recipe)

### The one rule that pays for this file: ABORT AT 1e4, NOT AT PPL 1000

Measured over every probe file on disk on 2026-09-02, 61 runs
(`lab/experiments/results/detonation_onset_scan.csv`):

| population | n | `preclip/total` after step 200 |
|---|---|---|
| detonated | 17 | first crossing of 1e3 at step **200–775**; 1e4 follows within ≤146 steps |
| healthy | 44 | never above **830**; five of them ran ≥ 20k steps (one to 30k) |
| ambiguous | 0 | |

So: **`preclip/total > 1e4` at any step ≥ 200 is a detonation, with zero false positives in
44 healthy runs and a 12x margin over the healthiest maximum.** The shipped div-guard in
`morph/training/train.py` fires on ppl > 1000 and only from step 2000, which is why every
detonated run in the table died at step 2040 with a `DIVERGED_step_2040.pt`, about 1,700
steps after the probe had already called it. A retry costs one draw; a late abort costs
one draw plus 20 minutes of GPU on a corpse.

Retry is legitimate. Runs decorrelate in ~11 steps at a fixed seed
(`morph-n1-run-comparisons-unreadable`), so a restart is a fresh draw at the same recipe.
The per-draw detonation rate of the winner recipe is ~70%, so expect 2–4 draws per
survivor. **Nothing rolls back or retries automatically yet** (as of 2026-09-02); the
runner scripts under `/home/wolfe/morph-scratch/tulfm/` do it by hand.

### It is an EARLY transient, as far as the data goes

All 17 onsets sit in steps 200–775. Every run that reached step 1000 without crossing
finished (A2 draw 2 to 5000, R1 retry to 5000, ema3 to 2500, notul-20k to 20000). This is
what makes the abort-and-retry rule cheap.

**What this does NOT show.** It does not show that a late detonation is impossible. The
core-Jacobian ladder (`lab/experiments/results/a2_jac_ladder_a2.json`) says the HEALTHY
paid map is expansive at the first loop iteration and moves further out while the loss
falls: worst-direction gain 55 → 97 and typical gain 1.05 → 1.13 between steps 2500 and
5000. Only ONE ≥20k survivor (notul-20k) is on the winner recipe; the other four long runs
carry the old GLA + spectral-cap stabilizers. The 100k schedule with prune/carve/route
has never run on this recipe. Treat "survives past 1000 ⇒ survives" as a bet with n=4 at
5k and n=1 at 20k, not as a law.

### What it is (measured)

- **Recipe, not arm.** notul-20k (survived), R1 notul (1/2), A2 (1/2), A2s (0/2), the
  γ-EMA draws (1/3) and the γ-freeze draws (0/3) all share it: retention off, spectral cap
  0, ternary QAT on, AdEMAMix β1=0 with `alpha_cap` 3.5, flat LR 1e-4 with warmup 0. It is
  the recipe whose loop earns depth (`morph-winner-recipe-gla-cap-out`); the stabilizers
  it removed (GLA, the cap) were also what killed the earning.
- **Ternary is the trigger surface.** Ternary-off draws: 3/3 healthy
  (`tul-a2-nt1..3`, max `preclip/total` 10–21) against a ~70% base rate.
- **Not a leak.** `lab/experiments/successes/2026-09-02-a2-future-leak-probe.md`: A2 at
  2500 and 5000 moves exactly 0 nats under future corruption in every cell. The fast
  descent and the K1−K6 earning (0.12 → 0.16, growing) are causal.
- **Not stale-m2.** The Task #276 cure is active in the fused AdEMAMix kernel
  (verified 2026-09-02, M2G onset capture).
- **The sick operator.** At the diverged checkpoints the whole-core-step typical gain is
  9–159 while each block's typical gain is 1.0–1.3: a low-rank blowup (ten to twenty
  directions at ~1e5) whose directions are ALIGNED across the six weight-shared blocks.
  Blocks 0–1 also carry a magnitude jump (worst-direction 13 → 284). Both moved; do not
  read "alignment, not magnitude" off a post-blowup checkpoint.

### What has been REFUTED as a cure (do not re-run these)

| lever | result | where |
|---|---|---|
| spectral cap / projection on core weights (4 variants, incl. hard σ≤1.5) | did not stop the takeover; two made it worse; the cap KILLS depth-earning | `lab/experiments/failures/2026-08-24-tul-takeover-cure.md`, `l2cap-depth-earning-was-the-leak` |
| `core_gain_clip` (clamp the realized magnitude) | masks the symptom, not ρ(J) | CLAUDE.md nested-dynamics section |
| ternary γ slow-EMA, β=0.99 | 2/3 detonated, same onset window; healthy draw +0.30 nats | `failures/2026-09-02-gamma-ema-paid-validation.md` |
| ternary γ hard freeze, β=1.0 | 3/3 detonated AND never learned (val flat 7.2–7.7) | `failures/2026-09-02-gamma-freeze-discriminator.md` |
| dense warmup, then ternary | **not a candidate**: ternary weights organize differently (Wolfe, 2026-09-02) | memory `no-dense-warmup-before-ternary` |
| GLA / retention as a stabilizer | kills depth-earning; `retention_carry` was a learned acausal leak | `morph-not-causal-retention-carry`, loop-killer bisect |

### THE CURE, MEASURED 2026-09-02/03: a 1000-step linear LR ramp (`training.warmup=1000`)

0 detonations in 9 of 9 draws on the ramp (three 2500-step A2 draws, wu5k, three GLA
draws, and the two 20k arms of the matched pair), against ~70% per draw without it. The
ramp is also 0.14 nats better at 2500 and 0.09 nats better at 20k for the plain model
(`lab/experiments/failures/2026-09-02-warmup-20k-pair.md`). Mechanism
(`successes/2026-09-03-warmup-core-map.md`): under the ramp the core map stays near
identity (typical gain 0.994 at 2500) instead of organizing expansive in the first ~300
steps; the loop's earning and the loop's instability were the same quantity. Cost: the
plain model's loop earning falls from 0.207 to 0.04 nats and stays there; A2's falls to
0.041 at 2500 and grows back to 0.100 by 20k. **`morph/configs/base.yaml` carries
`warmup: 1000` since 2026-09-03; a run that overrides it to 0 reopens this window.**

### What is OPEN (in priority order, 2026-09-03)

1. **Abort-and-retry in the trainer.** The 1e4 rule above, with a checkpoint rollback
   and a reseed. A belt for the ramp's braces.
2. **Code-assignment hysteresis** on the ternary cusp (0.5γ). The mechanism test that the
   γ experiments could not run, because both of them removed γ's adaptivity.
3. **Freeze-after-warmup** of γ on a healthy run at ~step 1000, if the γ-contagion
   question still matters after 1–2.

### Instruments (all exist; do not rebuild)

- `training.grad_probe_every=1` → `probe.jsonl` with `preclip/*` per module family.
- `lab/divergence/jac_ladder.py` — σ_max, typical gain, per-block gains, alignment of
  the core step at a live operating point. Cookbook:
  `docs/cookbook/measuring-the-core-map.md`. On A2 pass `training.batch_size=2`.
- `lab/divergence/future_leak_probe.py` — corrupt the future, score the past. Arm-general
  since 2026-09-02 (`_build.DepthLever`).
- `lab/divergence/a2_depth_sweep.py` — K1..K8 on A2 (its depth knob is
  `model.cfg.mean_depth`; the slot knobs are inert on A2).
- M2G onset capture (`failures/2026-09-02-m2g-onset-capture.md`) — optimizer-state
  forensics around an onset.

### Pre-registered records for this campaign

`lab/experiments/{successes,failures}/2026-09-02-*` — a2s-restricted-paid-loop,
m2g-onset-capture, gamma-ema-paid-validation, gamma-freeze-discriminator,
a2-future-leak-probe, a2-core-jacobian-ladder. Read the Verdict sections; the
predictions were frozen before each run.

## B. The slot-loop core takeover (2026-08)

A different failure with a different shape: a slow TURNAROUND of val CE on arms that loop
the core over ≤64 slot positions. Everything is in
[takeover-campaign.md](takeover-campaign.md). Short form: forward slot-state rank collapse,
not a weight spectrum and not an attention sink; every spectral lever failed;
`tul.per_slot_embed` is the best lever found and not a cure. The A2 arm (tokens through
the core) does not have this failure.

## C. A pinned codebook (flat CE near 7.3)

If val CE never leaves ~7.3 (the unigram floor for this tokenizer), the ternary codebook
cannot grow: a γ frozen at the step-0 mean|W| bounds every effective weight at ±γ₀. Seen
on all three `tul-a2-frz*` draws. The live per-forward γ is load-bearing for learning; a
near-frozen γ (β=0.99) already costs 0.3 nats at 2500 steps.

## D. The forecast spike train under the ramp (2026-09-03/04, the think-once panel)

Measured in `lab/experiments/successes/2026-09-03-tul-onset-capture.md` (C0/C1/C2, with a
bit-exact deterministic replay of the onset) after five of five forecast arms of the
think-once panel tripped the 1e4 rule between steps 1009 and 3618
(`failures/2026-09-03-tul-think-once-panel.md`). The ramp does not touch this face; the
memory target (M-own) on the same recipe ran clean.

**Shape.** `preclip/total` sits at 2–4 through the ramp, then single steps spike 10x to
1000x and snap back, the spikes escalating over ~60 steps to a detonation. The coda's
gradient and the conditioning stack's gradient stay flat; core, prelude, slot modules and
embedding spike together. The forward does NOT move: the core's forward norm stays at
0.8–1.15x its calm median and the loss within 1 nat on 21 of 22 spike steps. The slot
states' effective rank rises on spike steps (56–61 against 38–52). It is a backward
event.

**Mechanism, measured.**

| reading | calm (steps 200–1140) | onset window (1150–1200) |
|---|---|---|
| typical one-step gain of the slot loop, `jac/rms_t3` | 0.872 at step 0 → 0.912 at 1000 → 0.951 at 1150 (Spearman 0.98 against step) | 0.93–1.04; above 1 on 5 of 16 spike steps, 2 of 33 others; Spearman 0.65 against log `preclip/total` |
| worst-direction gain, `jac/sigma_t3` | 3.2 → 19 (1000) → 30 (1150) | 46–266 |
| per-block typical gain | 1.00–1.02 on every block, all along | 1.015–1.022 |
| product of the six blocks' worst gains | 73 → 47k (1000) → 242k (1150) | 0.2M–3M |
| cotangent growth back through the loop, `cot_t0/cot_t7` | 1.6–3 | 39–2436 on spike steps, median 18.5 on the others |
| ternary flips in the core per step | median 73.5k (rising 5k → 78k over the run) | 0.1x–2.3x the calm median on spike steps; NEVER a burst (Spearman flips vs log preclip = −0.49) |
| forward step size per iteration, `delta_ratio` | 0.68–0.78 after iteration 2 | 0.75–0.87 on spike steps |

Read together: each block is a near-isometry on average and the six weight-shared blocks
build a map that contracts a generic direction by ~0.9 per loop step while expanding a few
ALIGNED directions by 100x (the README's "aligned low-rank blow-up", now seen live and
before the blow-up). Training on the forecast target moves the typical gain toward 1 at a
steady rate; the spikes begin when it is within ~0.05 of 1, the spike steps carry the
higher gain (median 0.979 against 0.954), and the crossings cluster on them (5 of 16
against 2 of 33). The spike is the backward product of eight iterations through that map (the
growth sits in the last two backward iterations, where the operating point is most
expansive). **Ternary QAT is necessary** (C0: ternary off, same arm, 5000 steps clean, max
`preclip/total` 43 against 33,625) but the impulse is NOT a code-flip burst: the cusp-vault
mechanism of section A is refuted on this face. The offline sweep over the saved checkpoints
(`jac_sweep.jsonl`, capture record part 2) says the drift needs ternary AND the forecast
target: ternary off holds the typical gain at 0.89 at 2500 and 5000; M-own with ternary
moves 0.90 → 0.91 over 2500 steps; both run clean and both converge along their
trajectory by iteration 4 (step ratio 0.14–0.24), while the ternary forecast arm keeps
stepping at 0.5–0.7 per iteration. Caveat measured on A1 (bptt 4, token CE): a typical
gain of 0.975 and a worst gain of 176 at step 2500 with NO cotangent growth (1.7) and no
spike train. The discriminator is the backward product through the loop, not the gain
alone.

**Not a kernel bug.** C1 reproduced the face on the eager kernels in deterministic mode
(onset 1170, detonation 1208), and the replay from the rolling checkpoint at 1150 matched
C1 to the bit on all 49 steps (`preclip/total` and the loss, relative error 0.0). The
"Schrödinger bug" is retired: the run IS reproducible from a checkpoint. What was not
reproducible before 2026-09-03 was any run with `training.jac_probe_every > 0`, because
the Jacobian probe's measurement consumed the CUDA generator (fixed in
`train._jacobian_probe`; `tests/test_core_jacobian.py::test_probe_measurement_is_rng_neutral_with_dropout`).

**Clip-through-time alone does NOT cure it (measured 2026-09-04, `to-mnext-ctt`,
`lab/experiments/planned/2026-09-04-tul-clip-through-time.md`).** With the cotangent
arriving at every iteration bounded per row to 4x the exit cotangent
(`model.slot_cot_clip`), the clip bound on every row of every step from 1736, the exit
cotangent stayed flat (33 → 27) and the run still tripped at 2764. What grew instead: the
core weight gradient 0.3 → 250 (800x), the prelude's 1.1 → 294, and the FORWARD —
iteration 0's realised gain 1.5 → 2.0 → 3.8 → 5.8 → 9.6 across training, the exit state
norm 1017 → 5392, and the successive-step ratio along the trajectory crossing 1.0 at step
1800, the step the gradient began to climb. So the backward product is the first
symptom, not the whole disease: bound it and the map keeps moving past the edge and the
blow-up rides the weight path on an inflating forward. The clip stays in the tree as an
instrument and a belt (`loop/cot_post_*`, `loop/cot_bind_*`); the levers below are the
next arms (phase 2, `2026-09-04-tul-forward-levers.md`).

**What this says about the levers** (the capture's decision rule had a gap here: ternary
is necessary AND the flip burst is absent AND the amplifier predictions hold; the record
says so rather than patching the rule):

1. Bound the backward product between iterations (clip-through-time), forward untouched.
   DONE and insufficient (above): the forward inflates once the backward is bounded.
2. Keep the loop's typical gain away from 1 on the MAP, not by a weight-spectrum cap
   (section A's four failed caps bound a factor of one block; the map's gain is a product
   of six blocks whose per-block gains never left 1.00–1.02 while the map's worst gain
   went 3 → 500). Two arms in flight 2026-09-04: a per-slot state renorm between
   iterations (`model.slot_state_renorm`) and a hinge penalty on the map's typical gain
   measured by a finite difference every step (`model.slot_gain_lambda`).
3. Code-assignment hysteresis stays a candidate for section A's face only; it has no
   evidence on this one.

**Instruments added for this** (`morph/training/train.py`, on since the capture):
`training.loop_cot_probe=true` (cotangent norm per loop iteration, `loop/cot_norm_t{k}`),
`training.loop_rank_every=N` (`loop/eff_rank_t{k}`, `loop/delta_mean_t{k}`),
`training.batch_dump_every=N` + `batch_dump_dir` (token ids, labels and the slot layout of
each step), `jac/rms_*` (typical gain, whole map and per block), rolling checkpoints
(`ckpt_rolling_every/keep`), `lab/divergence/onset_locate.py`,
`lab/divergence/jac_sweep.py`. Runner: `/home/wolfe/morph-scratch/cap/run_onset_capture.sh`.

## Cross-links

- `CLAUDE.md` → "Core mental model" (nested dynamical system, why ρ(J_core) is the thing
  the optimizer cannot see) and the 2026-08-24 correction (a uniform rescale cannot slow an
  alignment).
- `lab/runtime-invariants.md` §6b/§6c — the invariants each arm must hold.
- vlt thread `tul-span-jepa`, entries 170–190 — the day-by-day record of this campaign.
