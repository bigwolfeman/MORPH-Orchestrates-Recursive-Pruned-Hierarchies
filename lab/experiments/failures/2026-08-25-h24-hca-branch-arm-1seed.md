> **REJECTED 2026-08-25 15:35, before it produced a scored result.** Two things were wrong
> with it, and only one was the seed count.
>
> 1. **The regime could not deliver the signal.** It ran `tul_a1` at batch 6 for 3500 steps,
>    where the control diverges on 1 of 4 seeds. The question is BINARY — the guard fires or
>    it does not — so the experiment must run where the control reliably fires.
>    `docs/tul-divergence-rca.md` §1 has that regime: batch 12, `alpha_cap` 3.5, production
>    kernels, full 20k schedule, where A1 aborted at 4540 and A1r at 3240, recorded as
>    "Two seeds fail the same way. This is structural, not seed luck."
> 2. **Shortening the run was the wrong fix for the budget confound.** `ademamix_t_beta3` is
>    an explicit config key. Pinning it holds the optimizer schedule while the run stays long
>    enough to see the failure. Amendment 1 below shortened the RUN instead, which kept the
>    confound out but also truncated the window the failure lives in.
>
> Replaced by [`2026-08-25-h24-hca-branch-arm-binary.md`](../planned/2026-08-25-h24-hca-branch-arm-binary.md),
> which pins `ademamix_t_beta3=20000`, runs the RCA regime, and predicts the binary outcome
> directly instead of CE deltas against a noise floor.
>
> **What ran under this file:** one control at 6000 steps / batch 6, which completed healthy
> (min 4.0929, final 4.2152, rise 0.122, core gradnorm ~0.006) — kept at
> `morph-scratch/h24arm6000/`; its arm was killed at step 200. A second control at 3500
> steps / batch 6 was killed at step 600. Neither produced a scored panel. No ARM number
> exists from either, so the replacement's predictions are still written blind to the arm.

# Experiment: H24 arm at one seed — REJECTED, superseded by the binary design

Status: failure

Ledger: `lab/divergence/takeover-campaign.md` H24.
Supersedes: [the 4-seed design](../failures/2026-08-25-h24-hca-branch-arm-4seed.md), rejected
mid-run when the campaign was scoped to a single seed.
Prior: [Phase 0 of H18](../failures/2026-08-25-h18-positional-attention-sink.md) found the
defect; [the no-training screen](../failures/2026-08-25-h24-hca-branch-screen.md) said to
expect a lever.

## What had been seen when these predictions were written

The control seed-0 run was already 8 minutes in, at step ~400, with one validation point:
`[VAL 250] loss=6.5285`. **No ARM number existed** — the arm had not been launched. The
2026-08-24 seed sweep is prior data and is used below to choose thresholds. Stated here
because predictions written after a run has started are worth less than predictions written
before one, and the reader should know exactly how much less.

## Question

Does training the TUL slot core with its HCA compressed branch ALIVE change whether, or when,
arm A1 takes over at seed 0?

## What one seed can and cannot say

Seed 0 at `alpha_cap 3.5` is the strongest single signal in this campaign: in the seed sweep
it aborted at step **2040** with validation CE going 6.31 -> 7.48, a **+1.17 nat** rise,
while seeds 1 to 3 finished with rises of 0.129, 0.090 and 0.127 — inside the measured
0.168-nat healthy noise floor. So a single paired comparison at seed 0 is not symmetric in
what it supports:

- **It can REFUTE strongly.** If the arm blows up at the same step in the same way, the
  branch is not a lever at `m = 16`, and no seed count would rescue that.
- **It can only SUPPORT weakly.** If the arm survives, that is one draw. MORPH runs
  decorrelate in 11 steps at a fixed seed with a 6.5 % median spread, and this campaign has
  already been burned once by one seed carrying a whole panel (H21: removing the extreme
  member took P3 from 5.193x to 1.002x).

**Therefore, fixed here before the run: a HELD panel is a SCREEN result. It licenses a
multi-seed arm. It does not license a claim that the branch is a lever.** The writeup must
say so in the Verdict, not in a footnote.

## Method

Two runs, one difference:

- control `--config-name tul_a1` (ALREADY RUNNING, launched 14:22, PID 3331484)
- arm     `--config-name tul_a1_hca16` (`model.core_hca_compress_ratio: 16`)

`core_hca_compress_ratio` is a construction-time field, default `null`, bit-identical to not
having it. Verified on the real model with `attn_sink_probe.py --geometry`: `tul_a1` reports
`n_blocks = 0` and `|out_comp| = 0.0000` at core blocks 1/3/5; `tul_a1_hca16` reports
`n_blocks = 4` and `|out_comp|` 228 to 244. Scoped to the CORE — a global
`model.hca_compress_ratio` would also re-block two prelude and two coda layers, making the
arm differ by seven modules instead of three. `m = 16` gives the core 4 compressed blocks,
the same number the TOKEN path gets at `seq_len 1024`.

Shared settings, the protocol that produced the seed sweep and the `onset-capture` ladder:

    training.steps=3500 training.batch_size=6 training.ademamix_alpha_cap=3.5
    model.use_kernels=false training.eval_every=250 training.gen_every=0
    training.ckpt_every=500 training.seed=0

Sequential, one trainer at a time. Runner `lab/divergence/h24_arm.sh`, scorer
`lab/divergence/score_h24_arm.py`.

### Method Amendment 1, 2026-08-25 15:20 — the step budget was NOT a free knob

This section originally said `training.steps=6000`, chosen to raise the divergence base
rate. That was an error and it cost the first control. `morph/training/optimizer.py:152`:

    t_beta3 = int(_tb) if _tb is not None else int(tr.steps)

and `base.yaml` ships `ademamix_t_beta3: null`. So a 6000-step run gives the optimizer a
beta3 warmup horizon **71 % longer** than the 3500-step seed sweep it was meant to be
compared against — and beta3 governs the slow EMA that this campaign's whole takeover
story runs on.

Measured consequence. At seed 0, identical in every other setting:

| run | budget | t_beta3 | VAL 250 | min | final | outcome |
|---|---:|---:|---:|---:|---:|---|
| `seedsweep-s0`, 2026-08-24 | 3500 | 3500 | 6.6820 | 6.3056 | 7.4790 | **ABORT at step 2040**, +1.17 nats |
| `h24-ctrl-s0`, 2026-08-25 | 6000 | 6000 | 6.5285 | 4.0929 | 4.2152 | completed 6000, rise 0.122, core gradnorm ~0.006 — **healthy** |

The budget is now the runner's second argument, echoed at start, defaulting to 3500. The
6000-step control is kept at `morph-scratch/h24arm6000/` as a data point in its own right;
its arm was killed 200 steps in rather than finished, because a pair whose control cannot
take over cannot answer the question the pair was built for.

`lab/experiments/failures/2026-08-23-tul-run-replication.md` names this confound in
writing — "the step-budget confound (`ademamix_t_beta3` follows `training.steps`) means the
5000-step Phase 1 run and this 4000-step pair do not share an optimizer schedule". It was
read the same day and the budget was changed anyway.

**Predictions are UNCHANGED.** None of them names a step count except V1's "by step 3000",
which the seed sweep's abort at 2040 already satisfies with room. A second, separate
caveat now stands on its own: the same replication document measures that two IDENTICAL
runs decorrelate to 10 % in eleven steps, so even at the right budget the control may not
reproduce its abort. If it does not, V1 fails and the answer is "the control did not
reproduce" — which is what V1 is for.

**Failure definition, unchanged from the rejected design.** A run DIVERGED if its own abort
guard fired (an `[ABORT] ... at step N` line in its log) OR its final validation CE is at
least **0.35 nats** above its own minimum. 0.35 sits above the measured 0.168-nat healthy
noise floor and far below seed 0's observed 1.17. The campaign already burned one experiment
on a 0.1-nat threshold that sat below its own metric's noise.

## Predictions

**Validity gate. Runs first, refuses the panel.**

- V1 the CONTROL reproduces seed 0's known failure — an `[ABORT]` line, or a rise of at least
  0.35 nats, **by step 3000**. The seed sweep aborted at 2040. A control that does not fail
  leaves nothing to compare against and the panel is void.
- V2 both runs either complete 6000 steps or stop on a recorded `[ABORT]`. A run lost to OOM,
  a crash, or an interrupt voids the panel and is SAID SO, not silently replaced.
- V3 the arm's core HCA branch is alive and the control's is not — already checked with
  `attn_sink_probe.py --geometry` on both configs before launch.

**P1 the arm survives the control's failure point.** The arm is still training, with no
`[ABORT]` and a rise under 0.35 nats, at the step the control failed at.

**P2 the arm is better at the control's failure step.** At the last common evaluation step at
or before the control's failure, the arm's validation CE is at least **0.20 nats** below the
control's. 0.20 is above the 0.168 noise floor, so it cannot be met by run-to-run variance
alone.

**P3 the arm reaches a lower minimum.** The arm's minimum validation CE over the whole run is
at least **0.10 nats** below the control's minimum.

**REFUTER.** If the arm aborts within 20 % of the control's abort step AND its minimum
validation CE is within 0.05 nats of the control's, then reviving the branch changes nothing
at `m = 16` and H24 is refuted as a lever. The defect is still a defect and the fix still
ships on its own terms; it just is not a divergence lever.

## Confounds, declared before the run

- **n = 1.** See "What one seed can and cannot say". This is the dominating limitation and it
  is a scope decision, not an oversight.
- **Not iso-parameter.** `B_a` is `[m, c]`, so the arm has 15 360 FEWER parameters per HCA
  core block, 46 080 fewer in total against 286.1 M (0.016 %). The direction makes an
  improvement harder to explain away as capacity, but it is not zero.
- **Capacity, not mechanism.** The screen showed the state-geometry lift is uniform across
  healthy and sick rungs. An improvement here is consistent with "the core got half its
  attention back" and supports no mechanism claim about the takeover.
- **One knob, one value.** `m = 16` only. A null result does not clear `m = 8` or `m = 32`.

## What would make this inconclusive, and why that is a failure

If the control does not reproduce seed 0's failure, the answer is "the control did not
reproduce", filed under `failures/`, with the next planned experiment fixing the gate. It is
not a hedged arm comparison.

## Declared not verified

- `seq_len 1024` short schedule only; the deploy recipe at 4096 does not have the defect
- `alpha_cap 3.5`, the regime where the takeover is reliable, not the shipped 1.0
- no generation samples (`gen_every=0`), so nothing here reads the teacher-forcing leak
- seeds 1, 2, 3 are NOT run, so nothing here says anything about them
