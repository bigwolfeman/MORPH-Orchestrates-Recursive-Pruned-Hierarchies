# Planned: A2 with a 1000-step LR warmup, and A2 at seq_len 512 — does the danger window close?

Status: failure
Date: 2026-09-02 (frozen ~21:20, before launch; trigger: Wolfe — "do the run.
also maybe try a 512 seq length too, see if that blows up")

## Question

The paid-axis detonation is an early transient: 17 of 17 onsets sit in steps
200-775 and the winner recipe runs flat LR 1e-4 with `training.warmup: 0`
(lab/divergence/DIVERGENCE-README.md). Two one-delta arms on `tul_a2`:

- **W** (the candidate): `training.warmup=1000`, a linear LR ramp
  (`optimizer.py`: `lr_max * step / warmup`) across the whole danger window.
  Does the detonation rate fall, and does the loop still earn?
- **S** (Wolfe's question): `data.seq_len=512` (L_total 640, max_slots stays
  64, so pad_frac rises). Does a shorter row blow up the same way?

## Hypothesis

W: the window is where a normal recipe has its warmup and this one has none;
with the LR at 3e-5 through step 300 the shadow weights move less per step
across the ternary cusp and the aligned low-rank direction has less energy to
grow. Counter: alpha already ramps over 1600 steps and the onset still lands
at 200-330, so the optimizer's early step size may not be the lever at all.

S: no mechanism I can name makes row length the lever (the sick operator is a
low-rank aligned blowup of the core step, which every position feeds). Half
the tokens per step gives a noisier gradient, which if anything raises the
rate. Counter: fewer positions per row changes the attention geometry and the
cotangent's effective rank, which was the takeover's lever on the slot axis.

## Method

3 draws per arm, `tul_a2` + panel flags (batch 6, seed 1, alpha_cap 3.5,
t_beta3 3500, eval 250, gen 0, probe every step), 2500 steps,
`ckpt_every=0` (a completed run still writes its final step_2500.pt). W first,
then S. Runner `/home/wolfe/morph-scratch/tulfm/run_a2_wu_s512.sh`.

**Tripwire (new, from the README's measured rule).** A watcher reads
probe.jsonl every 30 s and kills the trainer the first time `preclip/total`
exceeds 1e4 at a step >= 200 (0 false positives in 44 healthy runs; every
detonation crosses by step 776). A killed draw is DETONATED at that step; no
checkpoint, no retry. A draw that completes is scored by the same rule post
hoc. This is the abort half of abort-and-retry, in bash, before it goes into
the trainer.

Earning: `a2_depth_sweep.py --depths 1,6 --rows 48` on every healthy draw's
step_2500.pt (S draws with `data.seq_len=512` in the ckpt triple so the packer
matches). Clean A2 reference at 2500: val 4.6776; K1-K6 on the sweep's rows
0.119 (from the leak probe's clean column at k=900, positions < 900; the
full-row number is what the sweep reports and will be a little different).

## Predictions (frozen)

- **P-W1 (binding).** W detonations <= 1/3: **55%**.
- **P-W2.** Healthy W draws' final val CE at 2500 within 0.25 of clean A2's
  4.6776 (the ramp costs ~500 effective steps; A2 moved 5.18 -> 4.68 over
  its last 500): **50%**.
- **P-W3 (earning survives).** Healthy W draws' K1-K6 at 2500 >= 0.08:
  **65%**.
- **P-S1.** S detonations <= 1/3: **40%**.
- **P-S2.** Healthy S draws' final val CE at 2500 is worse than clean A2 by
  more than 0.15 (half the tokens per step AND a shorter context at eval):
  **80%**.

## Binding

P-W1 and P-W3 TRUE => warmup=1000 is the recipe amendment for the 20k, and
the 20k is Wolfe's call (not auto-launched). P-W1 TRUE and P-W3 FALSE =>
warmup buys stability by killing the earning, same trap as the cap; next is
a shorter ramp (300) or alpha_cap through the window instead. P-W1 FALSE =>
the early step size is not the lever; abort-and-retry in the trainer becomes
the production answer and cusp hysteresis the next mechanism test.
S is informational: S clean 3/3 while W is not => row length IS a lever and
the next test is batch-matched (batch 12 at 512) to separate tokens-per-step
from geometry.

## Not verified before run

The bash tripwire's kill path against a torch.compile'd trainer (SIGTERM then
SIGKILL after 20 s; wandb will mark the run crashed, which is correct); the
seq-512 packer at real scale (CPU dry build gave L_total 640, max_slots 64);
whether `training.warmup` interacts with the AdEMAMix alpha ramp beyond the
LR (read: it should not, the ramp is on lr only).

### Interpretation amendment — 2026-09-02 21:58 (after wu1, before wu2 finished; predictions untouched)

Wolfe: "we expect warm up to be better. we have kept our schedule for
reproducibility between experiments. but if it causes instability we can
change it. this is why I said to do lower seq length. warming up longer seq
length has a similar effect." So W is the REFERENCE for what a warmup buys
(wu1: healthy, 4.5391 at 2500, 0.14 nats ahead of clean A2), and S is the
CANDIDATE: a short-context start is a warmup that leaves the LR schedule and
every cross-experiment comparison untouched. Reading, if S detonates <= 1/3:
the production recipe is a context-length curriculum (512 through the danger
window, then 1024) using the existing loader
(`morph/training/curriculum_data.py`, `curriculum.py`,
`--config-name pretrain_curriculum`), and the follow-up is that curriculum
arm at 2500 steps vs wu1's 4.5391 at matched steps and matched tokens. If S
detonates >= 2/3 while W does not, length is not the lever and the LR ramp
is the only cheap early-phase fix found.

### Method amendment — 2026-09-02 22:40 (S arm CANCELLED before it started; predictions untouched)

Wolfe: "if lowering seq and lr works, we only add cosine lr warmup and leave
seq Len alone ... no point doing only a single probe about stability." With
wu1 and wu2 both healthy (P-W1 already TRUE), the recipe amendment is the LR
warmup only; the seq-length question is deferred to Wolfe's later warmup-on-
seq-length work. The runner was stopped at step 2400 of wu3 (no S draw was
launched; wu3 ran on under its own process and is scored post hoc by
`finish_wu.sh`, which also runs the K1/K6 sweeps). P-S1 and P-S2 are NOT RUN
and will not be scored. The 20k on this schedule needs a matched notul-20k on
the same schedule (Wolfe, 22:33); both are his call.

## Results (2026-09-02 21:18-23:05, tul-a2-wu1/2/3; S arm not run)

Tripwire never fired. Post-hoc verdicts (max preclip/total at step >= 200):
wu1 27.1, wu2 34.6, wu3 36.7 — all HEALTHY. 0/3 detonations against a ~70%
per-draw base rate (P(0/3) ≈ 0.027 under the base rate).

Val CE (same evaluate(), 20 batches):

| step | 500 | 1000 | 1500 | 2000 | 2250 | final 2500 |
|---|---|---|---|---|---|---|
| clean A2 | 6.369 | 5.391 | 5.227 | 5.175 | 4.689 | 4.6776 |
| wu1 | 6.364 | 5.204 | 5.067 | 5.015 | 4.539 | 4.5391 |
| wu2 | — | 5.214 | — | 5.033 | 4.544 | 4.5440 |
| wu3 | — | — | — | — | — | 4.5413 |

Earning, `a2_depth_sweep.py --depths 1,6 --rows 48`, identical rows for all
four (clean A2 step_2500 swept 23:05 on the same instrument;
lab/experiments/results/a2_sweep_tul-a2-{2500,wu1,wu2,wu3}.json):

| ckpt @2500 | K1 | K6 | K1−K6 |
|---|---|---|---|
| clean A2 | 4.7743 | 4.6534 | 0.1209 |
| wu1 | 4.5591 | 4.5128 | 0.0463 |
| wu2 | 4.5656 | 4.5167 | 0.0489 |
| wu3 | 4.5655 | 4.5199 | 0.0456 |

- **P-W1 (55%): TRUE.** 0/3.
- **P-W2 (50%): TRUE.** All three within 0.25 of 4.6776; in fact 0.13-0.14
  BETTER, and a 0.005 spread across draws.
- **P-W3 (65%): FALSE.** 0.046-0.049, under the 0.08 bar; 38-40% of clean
  A2's earning at the same step. The warmup model's K1 (4.56) beats the flat
  model's K6 (4.65).
- **P-S1, P-S2: NOT RUN** (cancelled 22:40).

## Verdict

**FAILURE by protocol (P-W3), and the recipe question is answered in the
direction that matters.** A 1000-step LR warmup closes the danger window
(0/3), costs nothing in CE (it is 0.14 nats ahead at 2500 with a 0.005
spread), and leaves the loop earning less at 2500 than the flat schedule
did. Two readings the method cannot separate at 2500:

- (killed) the ramp lets the coda/prelude solve what the loop was solving,
  the cap trap in a gentler form;
- (delayed) the earning is a late-forming feature (clean A2: 0.12 at 2500,
  0.17 at 5000) and the ramp shifts its formation by ~the ramp length.

Distinguishing them needs the SAME sweep at 5000 on a warmup draw:
`2026-09-02-a2-warmup-5k-earning.md`.

Recipe bookkeeping (Wolfe, 22:33 and 22:40): the schedule change is
`training.warmup=1000` on the existing cosine-flat 1e-4, seq_len untouched,
and any 20k on it needs a matched notul-20k on the same schedule. Old
flat-LR ledger numbers stay valid for flat-LR runs.

## Updated hypothesis

The paid-axis detonation is an early-phase step-size problem that a standard
LR ramp fixes; ternary is the surface it shows on, not the cause. The open
question is now the earning's TIMING, not stability.
