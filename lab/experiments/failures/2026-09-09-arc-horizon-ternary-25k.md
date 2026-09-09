# Planned: ARC horizon run — does the ternary loop's contribution change over 25,000 steps?

Status: failure
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

**Stopped at step 10,200 on Wolfe's decision (2026-09-09 13:11, "agreed"):** the surviving 20k
plain control (`notul-20k-wu`, ternary, the OLD loop entry, full BPTT, no fixed-point term,
retention off; swept at every kept checkpoint before this run started) answered the horizon
question, and the per-pass-strength panel needed the GPU. The record files as a failure per
protocol (the 25k predictions were not reached); the finding is decisive. Artifacts:
`../results/2026-09-09-horizon-ternary-25k/` (`old_entry_20k/` holds the old control's
eight sweeps and three anatomies; the new run's 5k and 10k sweeps and anatomies beside them;
token files and probes under `ignored/experiment-artifacts/2026-09-09-horizon-ternary-25k/`).
Everything CE-related is a horizon reading, not a ranking.

**The old entry over 20,000 steps** (`old-entry-notul-20k`, 480 rows, the sweep's paired
bootstraps; anatomy on 3 rows at depth 8):

| step | depth-6 CE | K1−K6 | K3−K6 | core MLP branch out/in @6 | carrier rank t8 |
| --- | --- | --- | --- | --- | --- |
| 2,500 | 4.452 | +0.040 [+0.039, +0.042] | +0.0010 | | |
| 5,000 | 3.973 | +0.037 [+0.036, +0.038] | +0.0016 | 0.15–0.20 | 30.2 |
| 7,500 | 3.788 | +0.036 | +0.0018 | | |
| 10,000 | 3.671 | +0.038 | +0.0020 | 0.17–0.25 | 35.3 |
| 12,500 | 3.595 | +0.038 | +0.0022 | | |
| 15,000 | 3.534 | +0.040 | +0.0025 | | |
| 17,500 | 3.492 | +0.042 | +0.0022 | | |
| 20,000 | 3.452 | +0.041 [+0.040, +0.042] | +0.0022 [+0.0018, +0.0026] | 0.20–0.36 | 9.3 |

The loss falls a full nat; the loop's share past pass 1 stays at 0.04 and past pass 3 at
0.002. The K-curve rises past depth 6 at every step (+0.011 at 16). The ternary MLP branches
strengthen slowly (mean 0.17 → 0.26 of their input per pass), a third of the bf16 core's
0.75–0.96 at 5,000. The state on this entry keeps moving 20–28 % per pass at every step and
the readout gains nothing from it past pass 3: drift the coda ignores.

**The Parcae-entry recipe (this run), 5k and 10k:** K1−K6 +0.035 and +0.038, K3−K6 +0.001
and +0.002, flat to depth 16; depth-6 CE 3.968 → 3.654; movement 44/23/9/6/4/3/2 % at both;
tripwire silent (P-hor-a holds to 10k). The 5k point replicates the Parcae-entry panel's
0.033 / 0.0009 to within 0.002.

**Predictions:** P-hor-b's "within 0.02 of 0.033" holds at 10k on the new entry and at every
one of eight checkpoints to 20k on the old (H-hor). P-hor-c: no checkpoint above 0.005 on
K3−K6 on either entry. P-hor-d: the old entry's branch ratio at 20k reads 0.26 mean, under
the 0.4 mark. P-hor-e: the old entry's held-out at 20k on its own stream was 3.55 (record);
the new run reached 3.65 at 10k on the sweep rows. Wall: 1.63 steps/s with the concurrent
sweep at 18.2 GB of 32.6 (P-hor-e's 16 GB clause FALSE by the sweep's share, the trainer
alone 10.2).

## Verdict

**failure** (the 25k predictions were not reached; the question is answered). Training
length does not change the ternary loop's contribution: flat at 0.04 past pass 1 and 0.002
past pass 3 from 2,500 to 20,000 steps on the old entry, and the same at 5k and 10k on the
Parcae entry. The under-ternary lever is tested at 5k with a clear conscience; H-hor holds.

## Updated hypothesis

The ternary core's per-pass strength (branch out/in ~0.2) is set by the quantizer, not by
training length; it creeps up over 20k steps and never approaches the bf16 core's. The
per-pass-strength panel (`2026-09-09-arc-per-pass-strength.md`) tests the scale rule and
the dead zone directly.
