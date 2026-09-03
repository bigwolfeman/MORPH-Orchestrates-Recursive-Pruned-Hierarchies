# Planned: one warmup draw to 5000 — is the loop's earning killed or delayed by the ramp?

Status: planned
Date: 2026-09-02 (frozen 23:10, before launch)

## Question

With `training.warmup=1000`, A2 is stable (0/3) and 0.14 nats better at
2500, but its K1−K6 earning at 2500 is 0.046-0.049 against clean A2's
0.121. Clean A2's earning grew 0.121 → 0.169 between 2500 and 5000. Does
the warmup model's earning catch up by 5000 (delayed), or stay low (killed)?

## Method

One draw, `tul_a2` + panel flags + `training.warmup=1000`,
`training.steps=5000 training.ckpt_every=2500`, tripwire watcher as before,
`wandb.name=tul-a2-wu5k`. Then `a2_depth_sweep.py --depths 1,2,3,4,5,6,7,8
--rows 48` on step_5000 (and 1,6 on step_2500 as an in-run replication of
the 0.046-0.049). Reference: clean A2 step_5000, K1−K6 0.1685 on the same
instrument and rows; final val 4.2315.

## Predictions (frozen)

- **P-X1 (binding).** wu5k K1−K6 at 5000 >= 0.10: **50%**. The three 2500
  numbers are tightly clustered at 40% of clean A2's, which reads more like
  a fixed cost than a delay; but the ramp is short relative to 5000 steps.
- **P-X2.** wu5k final val at 5000 <= clean A2's 4.2315: **70%**. The 0.14
  lead at 2500 should mostly persist.
- **P-X3.** No detonation (tripwire silent to 5000): **85%**. A late
  detonation here would be the first ever seen past step 776.

## Binding

P-X1 TRUE ⇒ delayed: warmup=1000 is the 20k recipe amendment as it stands.
P-X1 FALSE with P-X2 TRUE ⇒ the warmup model is better AND leans less on the
loop; the 20k question becomes "does the loop earn at all under the better
schedule", and the depth-8 tail of the sweep is the number to read before
the pair (a2 + notul, both on warmup) is queued. P-X3 FALSE ⇒ stop and read
the onset; a late detonation changes the README's transient claim.

## Not verified before run

Nothing new in the path; every piece ran today.
