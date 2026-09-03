# Planned: one warmup draw to 5000 — is the loop's earning killed or delayed by the ramp?

Status: failure
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

## Results (2026-09-03 00:16, run tul-a2-wu5k)

Tripwire silent; max preclip/total after step 200 = 159 at step 222 (the
2500-step warmup draws peaked at 27-37; still 6x under the 1e3 line). Final
val 4.1807 (clean A2 5000: 4.2315). Val trajectory vs clean A2: 2500 4.538 /
4.678, 3000 4.420 / 4.520, 4000 4.292 / 4.367, 4500 4.173 / 4.231.

`a2_depth_sweep.py --rows 48`, identical rows to every A2 sweep:

| ckpt | K1 | K2 | K3 | K4 | K5 | K6 | K7 | K8 | K1−K6 |
|---|---|---|---|---|---|---|---|---|---|
| wu5k @2500 | 4.5529 | | | | | 4.5117 | | | 0.041 |
| wu5k @5000 | 4.1517 | 4.1112 | 4.0990 | 4.0952 | 4.0940 | 4.0941 | 4.0948 | 4.0977 | **0.058** |
| clean A2 @5000 | ~4.34 | | | | | 4.1711 | | | 0.1685 |

- **P-X1 (50%): FALSE.** 0.058 < 0.10. The in-run 2500 number (0.041)
  replicates the three earlier draws (0.046-0.049).
- **P-X2 (70%): TRUE.** 4.1807 < 4.2315 (the 0.14-nat lead at 2500 narrows
  to 0.05 at 5000).
- **P-X3 (85%): TRUE.** No detonation to 5000.

## Verdict

**FAILURE on the binding prediction: the loop's earning is REDUCED by the
ramp, not delayed.** From 2500 to 5000 the warmup model's earning moves 0.041
→ 0.058 while clean A2's moved 0.121 → 0.169; the ratio stays near one third.
The curve saturates by depth 5 and the depth-8 tail is flat (4.0977 vs
4.0941). Yet the warmup model is the better model at every depth: its K6 is
0.077 nats under clean A2's, and its K1 (4.152) beats clean A2's K6 (4.171).

Reading: the ramp lets the prelude and coda take work the loop was doing
under the flat schedule, and the loop keeps a smaller, saturating share. This
is the "P-X1 FALSE with P-X2 TRUE" branch of the binding: the 20k question
is whether the loop earns at all under the better schedule. The pair
(`2026-09-02-warmup-20k-pair.md`) was already queued by Wolfe's decision
("Queue after tonight's draws") and runs as planned; its A2 sweeps at 5k, 10k
and 20k give the earning-over-time curve this run could not.

## Updated hypothesis

Under the warmup recipe the loop is a small, saturating contributor (~0.05
nats at 5k, saturated by depth 5). The CE win over the flat schedule (0.05 at
5k) is larger than the earning the flat schedule's loop had over it, so the
recipe is better even with a weaker loop. Whether the loop's share grows at
20k is the pair's readout; if it does not, the production mean depth can come
down (depth 4-5 saturates) for ~30% less compute per token.
