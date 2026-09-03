# Planned: continue the matched warmup pair from 20k to 40k (matched-step crossing)

Status: planned
Date: 2026-09-03 (frozen 10:50, NOT launched; GPU time is Wolfe's call)

## Question

Does A2's matched-step K6 deficit to notul keep closing past 20k, and does it
cross? `failures/2026-09-02-warmup-20k-pair.md` measured the K6 gap (same 48
rows) at 0.132, 0.090, 0.066, 0.057, 0.038, 0.029, 0.012 from 5k to 20k, with
A2's earning rising 0.041 → 0.100 and notul's flat at 0.04.

## Hypothesis

A2 keeps converting training into loop earning at roughly the 20k rate and
its deficit falls ~0.03 nats per 5000 steps; notul's earning stays ~0.04. The
crossing at matched steps lands between 22k and 30k. Wall clock does not
cross (A2 is 1.33x slower per step).

## Method

Full resume of both arms from their `step_20000.pt` (`training.resume=<path>
training.steps=40000 training.ckpt_every=2500`, same panel flags,
`training.warmup=1000`, tripwire on). The flat LR after the ramp makes the
schedule a function of step only, so a resume continues the same run. Readout
every 2500 steps: `a2_depth_sweep.py --depths 1,6 --rows 48` on A2 and
`token_depth_sweep.py --depths 1,6 --rows 48` on notul (same rows as the 20k
tables), and the 480-row depths 1,6 readout on both step_40000 checkpoints.
The headline is scored on the 480-row K6 at 40000, never on one eval.

Before launch: verify that `training.resume` restores optimizer state and the
RNG stream (train.py `load_checkpoint`, "like nothing happened") by resuming
notul for 50 steps and comparing its loss trace to the original run's steps
20001–20050 in the wandb history. If they do not match, this file is rejected
and a fresh 40k pair replaces it.

## Predictions (frozen)

- **P-C1.** Both arms reach 40000 with the tripwire silent: **80%** (a late
  detonation has never been seen; the map drifts outward, so not 95%).
- **P-C2 (headline).** A2 480-row K6 at 40000 < notul 480-row K6 at 40000:
  **55%**.
- **P-C3.** The 48-row K6 gap at 30000 is below 0.012 (still closing): **65%**.
- **P-C4.** notul K1−K6 at 40000 stays within 0.03 to 0.06: **70%**.
- **P-C5.** A2 K1−K6 at 40000 ≥ 0.12: **55%**.

## Binding

No automatic action. A P-C2 TRUE plus a wall-clock loss is "TUL wins at matched
steps only" and the slot-path cost becomes the next target. A P-C2 FALSE
closes the paid-loop line at this scale.
