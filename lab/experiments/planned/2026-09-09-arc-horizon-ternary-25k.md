# Planned: ARC horizon run — does the ternary loop's contribution change over 25,000 steps?

Status: planned
Date: 2026-09-09 (frozen before launch; Wolfe: "25k steps in ternary on open web data to see
if this changes over time"). Arc: `2026-09-04-loop-contribution-arc.md`, horizon row.

## Question

Every loop-contribution number on this arc is a 5,000-step reading. On the ternary
Parcae-entry recipe the loop's passes past the first carry 0.033 nats (K1−K6) and the passes
past the third carry 0.0009 (K3−K6); with bf16 core weights the same recipe reads 0.168 and
0.012. A deep model converges slower, so the ternary loop's contribution may grow with
training, shrink (the paid TUL loop's K-difference fell from 0.168 at 5k to 0.100 at 20k), or
stay flat. This run trains the ternary recipe for 25,000 steps and reads the K-curve, the
branch ratios and the state movement at every 5,000 steps.

## Hypothesis

H-hor: the contribution is flat over training. K1−K6 stays within 0.02 of 0.033 and K3−K6
under 0.01 at every checkpoint. H-hor′: it grows; K1−K6 exceeds 0.06 or K3−K6 exceeds 0.02
by 25k. H-hor″: it shrinks; K1−K6 falls under 0.02 by 25k.

## Method

One run, `horizon-ternary-25k` (`notul_horizon_ternary_25k`): the Parcae-entry recipe
(seq 1024, batch 6, ramp 1000 then flat 1e-4, ternary backbone, AdEMAMix β1=0 with its
horizons pinned at t_alpha 1600 / t_beta3 3500, noise state init, all-dim carry with learned
B, no fixed-point term, web text) for 25,000 steps, checkpoints every 5,000 (5 kept). Runner
`arc/run_horizon.sh`: 12-step smoke, the draw with the sustained tripwire, and a concurrent
watcher that sweeps each checkpoint as it lands (`core_depth_sweep.py`, 480 rows, batch 1 to
leave the trainer its memory, depths 0,1,2,3,6,9,12,16, per-token files) so the K-curve is
read during the run; `core_anatomy.py --rows 3 --depth 8` on every checkpoint and
`core_init_probe.py --rows 96` on the last, after the run. Commit pinned in
`arc/HORIZON_COMMIT`. Results to `lab/experiments/results/2026-09-09-horizon-ternary-25k/`.
Loop CONTRIBUTION is the reading; CE over training is reported as a curve, not a ranking.

If a long-horizon ternary checkpoint from an earlier run exists on disk (a subagent is
searching), it is swept with the same tool BEFORE this run starts and reported beside it as
a second horizon point on the old loop entry.

## Predictions (frozen)

- **P-hor-a (survival).** HEALTHY to 25,000 with the tripwire silent: **75 %** (the longest
  run of this entry; the 20k plain runs survived under the ramp).
- **P-hor-b (K1−K6 over training).** At 25k within 0.02 of 0.033: **50 %**; above 0.06:
  **25 %**; under 0.02: **25 %**. Monotone in one direction across the five checkpoints:
  **40 %**.
- **P-hor-c (K3−K6).** Above 0.02 with the CI above 0 at any checkpoint: **15 %**; at 25k
  above 0.005: **35 %**.
- **P-hor-d (mechanism).** Core MLP branch out/in at iteration 6 at 25k above 0.4 (from 0.24
  at 5k): **25 %**; consecutive state movement at iteration 6 at 25k above 5 %: **35 %**.
- **P-hor-e (readings).** Held-out loss at 20k under 3.60 (the 20k plain control read 3.55
  on its own stream at seq 1024): **60 %**. Wall ≤ 4.8 h: **70 %**; peak under 16 GB with
  the concurrent sweep: **80 %**.

## Binding

- H-hor′ ⇒ contribution is a horizon question and every 5k verdict on this arc is
  re-read at 25k; the under-ternary lever is tested at 25k, not 5k.
- H-hor or H-hor″ ⇒ training length does not rescue the ternary map; the under-ternary
  scale work proceeds at 5k with the precision axis as its reference.
- P-hor-a FALSE ⇒ the trip step and probe go to the divergence README; no automatic re-run.
- NO run beyond 25k from this experiment.

## Not verified before launch

The concurrent sweep's memory beside the trainer (batch 1 was never timed; ~12 min per
checkpoint expected). `ckpt_keep_last 8` with five checkpoints at 5,000 apart. Whether the
trainer's held-out curve at 20k matches the earlier 20k plain run's (different entry).

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
