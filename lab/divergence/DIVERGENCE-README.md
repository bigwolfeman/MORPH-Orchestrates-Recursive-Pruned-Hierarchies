# READ ME IN CASE OF STRANGE DIVERGENCE

**If a MORPH run is blowing up, stalling, or turning around, read this before touching a
knob.** Two campaigns have already burned weeks on this. Almost every idea you are about to
have is in one of the two indexes below with a number next to it.

Last updated 2026-09-02 21:00 (Wolfe: "we need a big loud note saved about this").

## Triage in 60 seconds

Open the run's `probe.jsonl` (written by `training.grad_probe_every=1`, key
`preclip/total`) and answer three questions:

| question | if yes | if no |
|---|---|---|
| Did `preclip/total` exceed **1e4** at any step ≥ 200? | **Paid-axis detonation.** Section A. The run is dead; the loss may look fine for another 1500 steps. | keep reading |
| Is val CE at a MINIMUM and then RISING by nats, on a slot-loop TUL arm (A1/A3, ≤64 looped positions)? | **Core takeover.** [takeover-campaign.md](takeover-campaign.md), 15 hypotheses, 11 refuted. Do not re-derive them. | keep reading |
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

## Cross-links

- `CLAUDE.md` → "Core mental model" (nested dynamical system, why ρ(J_core) is the thing
  the optimizer cannot see) and the 2026-08-24 correction (a uniform rescale cannot slow an
  alignment).
- `lab/runtime-invariants.md` §6b/§6c — the invariants each arm must hold.
- vlt thread `tul-span-jepa`, entries 170–190 — the day-by-day record of this campaign.
