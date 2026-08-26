# Experiment: H24 — does reviving the core's HCA branch stop the TUL takeover?

Status: failure

Ledger: `lab/divergence/takeover-campaign.md` H24.
Supersedes: [the 4-seed design](../failures/2026-08-25-h24-hca-branch-arm-4seed.md) and
[the 1-seed design](../failures/2026-08-25-h24-hca-branch-arm-1seed.md), both rejected
before producing a scored panel; each names its own reason.

## Question

Arm A1 loops the weight-shared core over slot positions and takes over. Arm A0 loops over
tokens and does not. On the slot path three of the six core blocks output **exactly 0.0000**
from their HCA compressed branch, because `hca_compress_ratio: 256` does not divide into 64
slots; on the token path the same weights give 4 blocks and `|out_comp| ~ 1030`.

Does giving the slot core its compressed branch back stop the takeover?

## The signal is binary

The divergence guard fires or it does not. That is the measurement. This experiment does not
compare validation CE against a noise floor, because CE deltas are not the thing being
solved and their spread at n=1 is 6.5 %; the guard firing is not a matter of degree.

Everything below is chosen so the CONTROL reliably fires. A control that survives answers
nothing, and two earlier designs died on exactly that.

## Method

Two configs, one difference:

- control `--config-name tul_a1`
- arm     `--config-name tul_a1_hca16` (`model.core_hca_compress_ratio: 16`)

`core_hca_compress_ratio` is a construction-time field, default `null`, bit-identical to not
having it. Verified on the real model with `attn_sink_probe.py --geometry`: `tul_a1` gives
`n_blocks = 0` and `|out_comp| = 0.0000` at core blocks 1/3/5; `tul_a1_hca16` gives
`n_blocks = 4` and `|out_comp|` 228 to 244. Scoped to the CORE — a global
`model.hca_compress_ratio` would also re-block two prelude and two coda layers.

**The regime is the one where the failure is documented**, `docs/tul-divergence-rca.md` §1:
batch 12, `alpha_cap` 3.5, production kernels, the full 20k-step optimizer schedule. There
A1 aborted at step **4540** and A1r (seed 1) at **3240**, recorded as *"Two seeds fail the
same way. This is structural, not seed luck."*

    training.steps=6000 training.ademamix_t_beta3=20000 training.batch_size=12
    training.ademamix_alpha_cap=3.5 training.eval_every=250 training.gen_every=0
    training.ckpt_every=1000

**`ademamix_t_beta3` is PINNED, not inherited.** `morph/training/optimizer.py:152` falls
back to `training.steps` when the key is null, which `base.yaml` ships, so a shorter run
silently shortens the optimizer's beta3 warmup — the slow EMA this entire failure mode runs
on. Pinning 20000 reproduces the RCA schedule exactly while stopping at 6000, which is past
both documented abort steps. The first launch of this arm changed the budget instead of
pinning the horizon and its control never took over at all; that run is kept at
`morph-scratch/h24arm6000/` as the evidence.

Seeds 0 and 1. Four runs, sequential, interleaved ctrl/arm per seed so machine drift hits
both members of a pair. Runner `lab/divergence/h24_arm.sh`, scorer
`lab/divergence/score_h24_arm.py`.

**Diverged** = the run's own guard wrote an `[ABORT] ... step_N` line. Nothing else. No CE
threshold, no rise, no judgement.

## Predictions

**Validity gate. Runs first, refuses the panel.**

- V1 **BOTH control seeds abort by step 6000.** The RCA regime is 2 of 2 at 3240 and 4540.
  If the control does not reproduce that, the regime is not reproduced, and the answer is
  "the control did not abort" — not a hedged arm comparison.
- V2 every run either reaches 6000 steps or stops on its own `[ABORT]`. A run lost to OOM, a
  crash, or an interrupt voids its pair and is SAID SO, not silently replaced.
- V3 the arm's core HCA branch is alive and the control's is not — already checked with
  `attn_sink_probe.py --geometry` on both configs before launch.

**P1 — the whole experiment.** **Neither arm seed aborts.** The arm reaches step 6000 on
both seeds with no `[ABORT]` line.

**P2 — the fallback if P1 fails.** On every seed where the arm DOES abort, it aborts at
least 50 % later than its own control. A lever that only delays is still a lever, but it is
a different claim and it gets a different number.

**P3 — the arm is actually training, not merely surviving.** On both seeds the arm's final
validation CE is below 5.0 nats. This exists to catch the degenerate pass: a run that avoids
the guard by learning nothing would satisfy P1 and mean nothing. The control arms in this
regime sit near 6.4 to 6.7 at their abort; a healthy A0 finishes near 3.27.

**REFUTER.** If the arm aborts on both seeds within 20 % of its own control's abort step,
reviving the branch does not touch the takeover at `m = 16` and H24 is dead as a cure and as
a lever. The defect is still a defect and the fix still ships on its own terms.

## What each outcome licenses

- **P1 holds, 2 of 2, against a control that failed 2 of 2** — the strongest result this
  campaign has produced. It still needs a third and fourth seed before "solved", and the
  writeup must say so, but it would be the first intervention to hold where every prior one
  failed.
- **P1 fails, P2 holds** — a lever of the H5 `per_slot_embed` class. Report it as that.
- **REFUTER fires** — H24 is closed. Two seeds are enough to close it, because "the arm blew
  up the same way at the same step, twice" needs no larger sample.

## Confounds, declared before the run

- **Not iso-parameter.** `B_a` is `[m, c]`, so the arm has 46 080 FEWER parameters (0.016 %
  of 286.1 M). The direction makes an improvement harder to explain as capacity, not easier.
- **Capacity, not mechanism.** The no-training screen showed the state-geometry lift is
  uniform across healthy and sick rungs. An improvement here is consistent with "the core
  got half its attention back" and supports no mechanism claim about the takeover.
- **One knob, one value.** `m = 16` only. A null result does not clear `m = 8` or `m = 32`.
- **Two seeds.** Enough to close H24 on a refutation; not enough to call a positive
  "solved". Fixed here so the writeup cannot drift.

## Declared not verified

- `seq_len 1024` short schedule only; the deploy recipe at 4096 does not have the defect
- `alpha_cap 3.5`, the regime where the takeover is reliable, not the shipped 1.0
- no generation samples (`gen_every=0`), so nothing here reads the teacher-forcing leak

---

# Results

Status: failure

Ran 2026-08-25, 15:29:36 to 18:02:51, four runs sequential on the 5090.
Artifacts: [`../results/2026-08-25-h24-hca-branch-arm-binary/`](../results/2026-08-25-h24-hca-branch-arm-binary/)
(four training logs, the runner log, and the scorecard).
Scored with the thresholds committed in `lab/divergence/score_h24_arm.py` before the runs.

| run | last step | min CE | final CE | ABORT |
|---|---:|---:|---:|---|
| ctrl-s0 | 2880 | 5.1713 | 6.4192 | **2880** |
| ctrl-s1 | 6000 | 5.0040 | 5.5881 | none |
| hca16-s0 | 6000 | 4.7523 | 5.5666 | none |
| hca16-s1 | 2940 | 5.5720 | 6.6818 | **2940** |

## Validity gate: FAILED on V1

**V1 required both control seeds to abort by step 6000. Control seed 1 completed all
6000 steps.** The RCA regime is 2 of 2, at 3240 and 4540, so one control surviving is a
real miss and the pre-registered panel is refused as a whole.

This is the caveat stated before the run, and it was the right one: the RCA aborts were
measured at **batch 14**, and this arm ran at **batch 12**. Batch 12 does not reproduce
the 2-of-2 divergence. The regime is still not one where the control reliably fails.

V2 passed. V3 was checked with `attn_sink_probe.py --geometry` before the runs.

*Scorer correction, made after the runs and unable to change any verdict:* V2 originally
read the last step from the periodic progress line, whose cadence is 200 and which the
loop's final step does not emit, so a run that finished all 6000 steps reported 5800 and
was called "a crash or an interrupt". `read_run` now also reads the `Final checkpoint:
.../step_<N>.pt` marker. Both ctrl-s1 and hca16-s0 wrote it and exited 0. V1 failed
independently, so the panel was refused before and after the fix.

## What the numbers say anyway

**P1 FAILS, and its failure does not depend on V1.** P1 was "neither arm seed aborts".
`hca16-s1` aborted at 2940. V1 exists to make an arm SURVIVING meaningful — it is not
needed to make an arm DIVERGING meaningful, because an arm that diverges cannot be a
cure whatever the control did. **H24 is refuted as a cure for the divergence.**

**P2 is unevaluable, not failed.** P2 compared each arm abort against its own control's
abort. The aborting seed flipped: the control aborted on seed 0 only, the arm on seed 1
only. There is no matched-seed pair to compare, so the lever reading is not available
from this experiment.

**The REFUTER does not fire literally.** It required the arm to abort on BOTH seeds
within 20 % of its own control. The arm aborted on one.

Three observations, offered as observations and not as anything the panel licenses:

- Divergence rate is **1 of 2 in the control and 1 of 2 in the arm**. Reviving the dead
  HCA branch does not move the rate.
- The two aborts land at **2880 and 2940, 2.1 % apart**, on opposite seeds. That is the
  signature of a coin flip near a 50 % base rate, not of a mechanism responding to the
  intervention.
- `hca16-s0` has the lowest min CE of the four runs, 4.7523 against the control's 5.0040.
  One seed each way is far inside the 6.5 % run-to-run spread measured in
  [`2026-08-23-tul-run-replication.md`](2026-08-23-tul-run-replication.md), so this is
  not evidence of anything.

## Verdict

**Failure.** H24 is refuted as a cure. The dead-HCA-branch defect is real and is fixed by
a shipped knob, but reviving the branch does not prevent the takeover.

The campaign still has no regime at 5090 scale where the control diverges 2 of 2. Every
binary arm is unreadable until it does. That, not H24, is the blocking problem.

## Updated hypothesis

The takeover does not live in the core's compressed attention branch. Combined with H18
(no positional sink) and the H24 screen (the rank-ratio lift was uniform across healthy
and sick rungs, +0.0939 against +0.0947), the core's ATTENTION geometry is not where the
disease is. The next lead should target the state collapse itself, not the paths feeding
it.

Before any further binary arm: find the batch size and `alpha_cap` at which the control
aborts on 2 of 2 seeds locally, and pre-register THAT as a standalone calibration run.
Running an arm against a control that fails half the time has now cost three arm designs.
