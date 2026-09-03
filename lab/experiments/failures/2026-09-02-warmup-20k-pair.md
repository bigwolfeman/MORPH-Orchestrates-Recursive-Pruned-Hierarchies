# Failure: the matched 20k pair on warmup — A2 (TUL, tokens through the core) vs notul, same schedule

Status: failure
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

## Interim results — A2 arm (2026-09-03 07:04; notul arm running; predictions untouched)

tul-a2-20k-wu: HEALTHY to 20000, tripwire silent (max preclip/total 274 at
step 222). Final val 3.4762; last-5 mean 3.5039; best eval 3.2332 (79 evals).
Old flat references: notul-20k final 3.5600 / last-5 3.4894 / best 3.2736;
tul-20k 3.8255 / 3.8461 / 3.6068. 8 checkpoints kept (2500..20000).

Earning over training (`a2_depth_sweep.py --rows 48`, identical rows):

| step | K1 | K6 | K1−K6 |
|---|---|---|---|
| 2500 (wu draws, mean of 4) | | | 0.044 |
| 5000 | 4.1380 | 4.0812 | 0.057 |
| 10000 | 3.7979 | 3.7258 | 0.072 |
| 20000 | 3.5592 | 3.4590 | **0.100** |

Full curve at 20000: K1 3.5592, K2 3.4872, K3 3.4682, K4 3.4608, K5 3.4590,
K6 3.4590, K7 3.4599, K8 3.4615. Saturates by depth 5; the tail is flat.
The remaining checkpoints (7500, 12500, 15000, 17500) get the 1,6 sweep
after the notul arm frees the card (a concurrent sweep next to a trainer at
500 W is the UPS failure condition).

- **P-P1 (60%): A2 half TRUE** (20000 steps, no detonation); notul pending.
- **P-P3 (50%): TRUE**, on the bar (0.1002 >= 0.10).
- **P-P4 (65%): TRUE.** 3.4762 < 3.5600.
- P-P2, P-P5: pending the notul arm.

The loop's share grows monotonically under the ramp (0.044 -> 0.100 over
2500 -> 20000) where the flat notul's did not (0.220 at 4500, 0.207 at
20000). Wolfe's "the loop contribution may change over time" (00:35) is
measured: it does, upward.

## Results (2026-09-03 10:36; notul arm done 10:31, curves and wide readout after)

Both arms HEALTHY to 20000, tripwire silent (A2 max preclip/total 274 at step
222; notul 37.9 at 612). Wall clock for 20000 steps at the panel flags: A2
16514 s (02:28:14 to 07:03:28), notul 12421 s (07:04:18 to 10:31:19); A2 is
1.33x slower per step. Params 268.2M vs 265.1M.

**Every eval is one different validation batch with a per-eval spread of
~0.085 nats (sd of the last 10 evals: A2 0.086, notul 0.085). The prereg's
"final val" is ONE such eval.** The low-noise instrument is the fixed-row
sweep: identical first N validation rows for both arms, every depth on the
same rows (`a2_depth_sweep.py` / `token_depth_sweep.py`).

| readout at 20k | A2 (tul-a2-20k-wu) | notul (notul-20k-wu) | A2 − notul |
|---|---|---|---|
| final val (1 eval) | 3.4762 | 3.4890 | −0.013 |
| last-5 eval mean | 3.5039 | 3.4118 | +0.092 |
| last-20 eval mean (sd) | 3.4669 (0.105) | 3.4245 (0.091) | +0.042 |
| best eval (both at 16250) | 3.2332 | 3.1748 | +0.058 |
| K6, 48 fixed rows | 3.4590 | 3.4470 | +0.012 |
| **K6, 480 fixed rows** | **3.4701** | **3.4486** | **+0.022** |
| K1, 480 fixed rows | 3.5738 | 3.4900 | +0.084 |
| K1−K6, 480 rows | 0.104 | 0.041 | |
| wall clock, 20k steps | 16514 s | 12421 s | 1.33x |

Full depth curve at 20000 (48 rows). A2: K1 3.5592, K2 3.4872, K3 3.4682,
K4 3.4608, K5 3.4590, K6 3.4590, K7 3.4599, K8 3.4615. notul: K1 3.4905, K2
3.4594, K3 3.4497, K4 3.4474, K5 3.4462, K6 3.4470, K7 3.4469, K8 3.4483. Both
saturate by depth 5.

Earning over training, both arms, same 48 rows (`a2_sweep_tul-a2-20k-wu_*.json`,
`token_sweep_notul-20k-wu_*.json`):

| step | A2 K1 | A2 K6 | A2 K1−K6 | notul K1 | notul K6 | notul K1−K6 | K6 gap A2−notul |
|---|---|---|---|---|---|---|---|
| 2500 | 4.5545 | 4.5133 | 0.041 | 4.4659 | 4.4253 | 0.041 | +0.088 |
| 5000 | 4.1380 | 4.0812 | 0.057 | 3.9879 | 3.9488 | 0.039 | +0.132 |
| 7500 | 3.9234 | 3.8573 | 0.066 | 3.8064 | 3.7676 | 0.039 | +0.090 |
| 10000 | 3.7979 | 3.7258 | 0.072 | 3.7011 | 3.6601 | 0.041 | +0.066 |
| 12500 | 3.7291 | 3.6435 | 0.086 | 3.6249 | 3.5863 | 0.039 | +0.057 |
| 15000 | 3.6582 | 3.5683 | 0.090 | 3.5719 | 3.5307 | 0.041 | +0.038 |
| 17500 | 3.6100 | 3.5122 | 0.098 | 3.5252 | 3.4831 | 0.042 | +0.029 |
| 20000 | 3.5592 | 3.4590 | 0.100 | 3.4905 | 3.4470 | 0.044 | +0.012 |

Old flat-schedule references at 20000: notul-20k final 3.5600, last-20 3.5167,
best 3.2736, K1−K6 0.207; tul-20k (free ride) final 3.8255, last-20 3.8139.

### Predictions scored

- **P-P1 (60%): TRUE.** Both arms reached 20000 with the tripwire silent.
- **P-P2 (40%, the headline): TRUE by the letter, FALSE by every lower-noise
  estimator.** 3.4762 < 3.4890 on the single final eval (spread 0.085). The
  last-20 mean (+0.042), the best eval (+0.058), the 48-row K6 (+0.012) and the
  480-row K6 (+0.022) all put notul ahead. The prereg chose an estimator that
  cannot resolve a 0.02-nat gap. That is the protocol failure that files this
  under `failures/`.
- **P-P3 (50%): TRUE**, on the bar: K1−K6 0.1002 at 48 rows, 0.104 at 480.
- **P-P4 (65%): TRUE.** A2 final 3.4762 < 3.5600; last-20 3.4669 < 3.5167 too.
- **P-P5 (75%): TRUE.** notul-wu final 3.4890 < 3.5600; last-20 3.4245 < 3.5167.

## Verdict

**A2 does not beat notul at 20k.** At matched steps the two are within eval
noise, with a 0.022-nat edge to notul on the 480-row same-rows readout. At
matched wall clock notul is clearly ahead: A2 at step 15000 (~12400 s, K6
3.5683) against notul at 20000 (12421 s, K6 3.4470) is 0.12 nats.

What the pair did measure:

1. **The ramp beats the flat schedule for BOTH arms.** notul-wu is 0.092 nats
   better than flat notul-20k on the last-20 mean (3.4245 vs 3.5167) and 0.099
   on the best eval. Zero detonations in 2 of 2 twenty-thousand-step runs. This
   is the config change the campaign licenses.
2. **The ramp removes the loop's earning from the plain model.** notul-wu's
   K1−K6 is 0.04 at every checkpoint from 2500 to 20000 (flat notul-20k: 0.207
   at 20000). The plain model on the ramp is close to a no-loop model, and it
   is still the best CE in the campaign.
3. **A2's loop keeps taking share (0.041 → 0.100) but it repairs a worse
   depth-1 model.** A2's K1 is 0.084 nats behind notul's at 20k; its extra
   0.06 of earning closes most, not all, of that.
4. **The matched-step K6 gap closes monotonically from 5k on**: 0.132, 0.090,
   0.066, 0.057, 0.038, 0.029, 0.012. A straight-line read puts the crossing
   near 22k to 25k steps. That is a matched-step crossing only; the 1.33x
   wall-clock cost does not close.

Wolfe's hypothesis "TUL beats non-TUL in every way" (2026-09-02) is refuted at
20k on this recipe. The old ledger gap (0.357 nats behind at 20k) is gone; A2
is a paid loop with a growing contribution and a slot-path tax.

## Updated hypothesis

Under the ramp, the loop's value to the plain model is a fixed ~0.04 nats.
A2 converts training into loop earning at ~0.004 nats per 1000 steps and its
matched-step deficit falls ~0.03 nats per 5000 steps, so it passes notul at
matched steps near 22k to 25k if both trends hold, and never at matched wall
clock while tokens pay the slot path AND the core. The next experiment is
the continuation to 40k by full resume (`training.resume`), preregistered in
`planned/2026-09-03-warmup-pair-continue-40k.md` and NOT launched (GPU time
is Wolfe's call).

Method note for every later prereg on this branch: score the headline on the
480-row same-rows K6 sweep or the last-20 eval mean, never on one eval.

## What the method could not distinguish

A 0.02-nat difference at 20k from zero: the 480-row readout is one number per
arm with no bootstrap over rows, and the two arms score slightly different
token sets (A2 rows carry slot positions; 501106 scored tokens vs ~491k). A
per-row paired bootstrap on the same sweep is the cheap upgrade.

Not verified: the eager generator on A2 (gen_every=0 throughout); any seed
other than 1; the ramp on the 100k production schedule with prune/carve/route
on.
