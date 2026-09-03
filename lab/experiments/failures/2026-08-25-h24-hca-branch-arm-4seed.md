> **REJECTED 2026-08-25 14:30, 8 minutes into run 1 of 8.** Wolfe scoped the campaign to a
> single seed. Predictions P1 and P3 are written in terms of "how many of the four seeds",
> so they cannot be re-scored at n=1, and the rules forbid editing Predictions during a run.
> This file is therefore rejected rather than amended, and
> [`2026-08-25-h24-hca-branch-arm-1seed.md`](../planned/2026-08-25-h24-hca-branch-arm-1seed.md)
> replaces it. Kept because its Method, failure threshold, and declared confounds carry over
> verbatim, and because a rejected design is part of the audit trail.
>
> **What had been observed when the replacement's predictions were written:** the control
> seed-0 run had reached step ~400 with one validation point, `[VAL 250] loss=6.5285`. No
> ARM number existed — the arm had not been launched. The seed sweep of 2026-08-24 is prior
> data and is cited in both files.
>
> **Not wasted:** the control seed-0 run was left alive and becomes the control of the
> replacement experiment. Only the runner loop was stopped.

# Experiment: H24 arm, 4-seed design — REJECTED MID-RUN, superseded

Status: failure

Ledger: `lab/divergence/takeover-campaign.md` H24.
Agent Note: [`.agents/notes/rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`](../../../.agents/notes/rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md)
Prior: [Phase 0 of H18](../failures/2026-08-25-h18-positional-attention-sink.md) found the defect;
[the no-training screen](../failures/2026-08-25-h24-hca-branch-screen.md) said to expect a lever.

## Question

Does training the TUL slot core with its HCA compressed branch ALIVE change how often, how
soon, or how badly arm A1 takes over?

## What is already known, and what it predicts

Measured: on the slot path (S = 64) `hca_compress_ratio: 256` gives `n_blocks = 0`, so core
blocks 1, 3 and 5 output exactly 0.0000 from their compressed branch while the gate still
spends `g_comp ~ 0.50` on that zero tensor. The token path (A0, healthy) has 4 blocks and
`|out_comp| ~ 1030`. A1 diverges, A0 does not.

The no-training screen showed reviving the branch lifts the loop's slot-state rank ratio by
**+0.0939 on healthy rungs and +0.0947 on sick ones** — a UNIFORM effect. So the expectation
going in is a LEVER of the H5 `per_slot_embed` class (doubles time-to-failure, better CE,
does not cure), not a cure. That expectation is written into P4 below so it can be wrong.

## Method

Two configs, one difference:

- control `--config-name tul_a1`
- arm     `--config-name tul_a1_hca16` (`model.core_hca_compress_ratio: 16`)

`core_hca_compress_ratio` is a new construction-time field, default `null`, which is
bit-identical to not having it (verified: the geometry probe on `tul_a1` still reports
`n_blocks = 0` and `|out_comp| = 0.0000` at core blocks 1/3/5). Under the arm the same probe
reports `n_blocks = 4` and `|out_comp|` 228 to 244. It is scoped to the CORE: setting
`model.hca_compress_ratio` globally would also re-block the two prelude and two coda HCA
layers, making the arm differ by seven modules instead of three.

16 gives the core 4 compressed blocks, the same number the TOKEN path gets at
`seq_len 1024` (1152 // 256 = 4). 8 and 32 are untried.

Shared settings — the protocol that produced the seed sweep and the `onset-capture` ladder,
so the control is comparable to everything already in the campaign:

    training.steps=6000 training.batch_size=6 training.ademamix_alpha_cap=3.5
    model.use_kernels=false training.eval_every=250 training.gen_every=0
    training.ckpt_every=1000

Seeds 0, 1, 2, 3, both arms, **paired and interleaved** (ctrl-s0, arm-s0, ctrl-s1, ...), one
trainer at a time. 8 runs. The control is RE-RUN rather than read off the 2026-08-24 seed
sweep: the tree has moved since, and a control from a different code state is not a control.

6000 steps, not the sweep's 3500. At 3500 only seed 0 of the four diverges (val CE 6.31 ->
7.48, guard aborted at ~2000) while seeds 1 to 3 finish with rises of 0.129, 0.090 and 0.127
nats — inside the measured healthy noise floor of 0.168. The known abort steps at
`alpha_cap` 3.5 are 2080, 3240, 4540, 5900 and 6200, so 6000 steps covers four of five and
gives a base rate a comparison can read.

Runner: `lab/divergence/h24_arm.sh`. Scorer: `lab/divergence/score_h24_arm.py`, thresholds
committed with this file.

**Failure definition, fixed here.** A run DIVERGED if its abort guard fired (non-zero exit)
OR its final validation CE is at least **0.35 nats** above its own minimum. 0.35 is chosen
above the measured healthy within-run noise floor of 0.168 nats and far below seed 0's
observed 1.17 — the campaign has already burned one experiment on a 0.1-nat threshold that
sat below its own metric's noise.

## Predictions

**Validity gate. Runs first, refuses the panel.**

- V1 the CONTROL reproduces the known behaviour: seed 0 diverges, and at least two of seeds
  1, 2, 3 reach 3250 steps with a rise under 0.35 nats. A control that does not behave like
  every prior control makes the arm unreadable.
- V2 every run either completes 6000 steps or stops with a recorded divergence-guard abort.
  A seed pair lost to OOM, a crash, or an interrupt is dropped from the panel and SAID SO,
  not silently replaced.
- V3 the arm's core HCA branch is alive and the control's is not, checked with
  `attn_sink_probe.py --geometry` on both configs before the runs start.

**P1 fewer failures.** The arm diverges on strictly fewer of the four seeds than the control.

**P2 later failure.** On every seed where BOTH arms diverge, the arm's abort step (or first
eval at or past a 0.35-nat rise) is at least 20 % later than the control's.

**P3 better CE.** The arm's final validation CE is lower than the control's on at least 3 of
the 4 seeds, and the mean over surviving seeds is at least 0.10 nats lower.

**P4 not a cure.** The arm still diverges on at least one seed. This is a prediction the
screen forces, and it is the one whose FAILURE would be good news: if the arm holds all four
seeds, P4 fails and H24 is a stronger result than the screen justified. The writeup must say
that rather than counting it as a miss.

**REFUTER.** If the arm and the control diverge on exactly the same set of seeds AND their
mean final validation CE differs by less than 0.05 nats, then reviving the branch changes
nothing that matters and H24 is refuted as a lever. The defect is still a defect and the fix
still ships on its own terms; it just is not a divergence lever.

## Confounds, declared before the run

- **Not iso-parameter.** `B_a` is `[m, c]`, so the arm has 15 360 FEWER parameters per HCA
  core block, 46 080 fewer in total against 286.1 M (0.016 %). The direction makes an
  improvement HARDER to explain away as capacity, but it is not zero.
- **Capacity, not mechanism.** The screen showed the state-geometry lift is uniform across
  healthy and sick rungs. An improvement here is consistent with "the core got half its
  attention back" and does not by itself support any mechanism claim about the takeover.
- **One knob, one value.** `m = 16` only. A null result does not clear `m = 8` or `m = 32`.
- **Four seeds.** MORPH runs decorrelate in 11 steps at a fixed seed with a 6.5 % median
  spread. Four paired seeds is the minimum this campaign treats as readable and it is not
  a lot.

## What would make this inconclusive, and why that is a failure

A validity-gate failure is filed under `failures/` with the gate named, and the next planned
experiment fixes the gate. If the control does not reproduce, the answer is "the control did
not reproduce", not a hedged arm comparison.

## Declared not verified

- `seq_len 1024` short schedule only; the deploy recipe at 4096 does not have the defect
- `alpha_cap 3.5`, the regime where the takeover is reliable, not the shipped 1.0
- no generation samples (`gen_every=0`), so nothing here reads the teacher-forcing leak
