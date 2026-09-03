# Planned: the matched 20k pair on warmup — A2 (TUL, tokens through the core) vs notul, same schedule

Status: planned
Date: 2026-09-02 (frozen 23:45, before launch; trigger: Wolfe — "for our 20k
tul run we will have to run the no tul baseline again with the same lr" and
"Queue after tonight's draws")

## Question

Does the paid TUL loop beat the plain looped model at 20k when both run the
recipe that survives: retention per the GLA arm's rule, cap 0, ternary on,
AdEMAMix alpha_cap 3.5, flat 1e-4 after a 1000-step ramp? The old 20k pair
(tul-20k 3.8255 vs notul-20k 3.5600 final; last-5 3.8461 vs 3.4894) was the
free-ride TUL on the flat schedule; it is no longer the comparison.

Honest prior from the 5k numbers on the flat schedule: the notul twin
(notul-bg0c0, 4500 steps) finished at 4.1812 with K1-K6 0.220; A2 (5000
steps) finished at 4.2315 with 0.1685. The notul twin was ahead on both axes
at 5k. "TUL beats non-TUL in every way" is the hypothesis this pair tests,
not a prior.

## Method

Two runs, back to back, same seed 1, same panel flags, `training.warmup=1000`,
`training.steps=20000 training.ckpt_every=2500` (keep_last default 8 =
every one), `training.gen_every=0` (the eager generator x tokens_through_core
is still unverified; gen samples are a later, explicit run), tripwire watcher
throughout (a late detonation past step 776 would be the first ever seen and
stops the chain).

- **A2 arm**: `--config-name tul_a2 [+ model.retention=true per the GLA rule]`,
  wandb `tul-a2-20k-wu`. Readout: `a2_depth_sweep.py --depths 1..8 --rows 48`
  on step_20000, and 1,6 on step_5000 / step_10000 for the earning-over-time
  curve.
- **notul arm**: `--config-name notul_bg0c0 [+ model.retention=true per the
  same rule]`, wandb `notul-20k-wu`. Readout: `token_depth_sweep.py --depths
  1..8 --rows 48` on step_20000.
- Throughput reference: A2 1.44 sps, notul 1.65 sps at these flags => ~3.9 h +
  ~3.4 h, plus sweeps.

GLA selection rule (frozen in `2026-09-02-a2-gla-under-warmup.md` and in the
runner): retention=true iff >= 2 healthy GLA draws with mean val@2500 <=
4.4915 and mean K1-K6 >= 0.026; else retention=false. The rule applies to BOTH
arms so the pair stays matched.

## Predictions (frozen)

- **P-P1.** Both arms reach 20000 with the tripwire silent: **60%**.
- **P-P2 (the headline).** A2 final val at 20k < notul final val at 20k:
  **40%** (notul was 0.05 ahead at 5k; TUL's slots cost positions; the
  schedule helps both equally).
- **P-P3.** A2 K1-K6 at 20k >= 0.10: **50%** (0.046 at 2500 under warmup;
  clean A2 grew 0.12 -> 0.17 from 2500 to 5000 on the flat schedule).
- **P-P4.** A2 final val at 20k < old flat notul-20k's 3.5600: **65%**.
- **P-P5.** notul-wu final val at 20k < old flat notul-20k's 3.5600: **75%**
  (warmup helped every arm it touched tonight).

## Binding

No automatic action. The readout table (final val, last-5 val, K1..K8 on
both, earning curve on A2) goes to Wolfe; "we update the config to that
winner while we are in this branch" (Wolfe, 23:35) is a human step after the
table. P-P1 FALSE => the README's transient claim gets its first
counterexample and the onset step is the first thing to read.

## Not verified before run

A 20k run on the warmup schedule (longest so far: wu5k, in flight at
freeze time); disk: 8 checkpoints x 2.25 GB x 2 arms = 36 GB on 1.2 TB free;
the fused-kernel GLA path on A2 if the rule selects it (the GLA arm's draw 1
is that smoke).
