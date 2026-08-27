# Overnight worklist — 2026-08-27 (Wolfe asleep; finish this)

Read this FIRST after any compaction. It is the live worklist. The campaign
context is [takeover-campaign.md](takeover-campaign.md); the pivot there still
stands (the takeover is credit assignment, not stability).

## Standing constraints

- **Sequential runs only** on the 5090 (UPS). One trainer at a time, ~19 GB peak.
- **n=1 is unreadable here.** Measured control base rates over 4 seeds:
  seed 0 aborts, seeds 1 and 2 take over (steps 2800 / 3000), seed 3 never does.
  So "one clean run" is p≈0.25 evidence. Screen every arm at **seeds 0 and 1**
  (0 is sharpest — control aborts there; 1 is where control takes over at 2800).
- **`ademamix_t_beta3` is null** and falls back to `training.steps`. Matched by
  construction at 3500 steps for fresh runs; PIN it on any resume-based arm.
- Control reference (`seedsweep-s*`, log `/home/wolfe/morph-scratch/seedsweep/`):
  ppl_tok @3250 = 105.54 (s1), 91.77 (s2), 90.28 (s3); median 91.77.
- **Do not `git commit` while a subagent is writing the tree.** Already burned
  once (`d3a86da` swept an agent's files).

## The queue is RUNNING — check it before starting anything

`ignore/overnight_queue_v2.sh`, launched 2026-08-27 02:11 with `setsid nohup`.
Progress log: `/home/wolfe/morph-scratch/queue2/queue.log`. Per-run logs and
probe mirrors: `/home/wolfe/morph-scratch/queue2/<name>-s<seed>/{run.log,probe.jsonl}`.

**It replaced three earlier scripts, and the reason matters.**
`overnight_queue{,2}.sh` launched with the shipped default
`training.grad_probe_every: 0`, so **no `preclip/core_share` was logged at all**.
The only share series such a run has is `gradnorm/*`, sampled every 100 steps,
and `score_arms.py::fires` refuses a window with under 20 samples — its 50-step
rule gets none at that cadence. Every arm would have finished unscorable against
S2, C2 and N1, the predictions the experiment exists to test. v2 runs
`grad_probe_every=1` (~0.5 % throughput, read-only, no RNG draw) and mirrors the
probe to JSONL. The abort guards stay at `0.0`: this is measurement, not
intervention. Method amendment recorded in the pre-registration.
`overnight_queue3.sh` was the repair queue for a run a concurrent launch
OOM-killed; v2 absorbs it and serialises every launch under `flock`, which
closes the check-then-act race that caused the kill.

Order (~33 min each, ~6 h total): warmup s0/s1 → **center s0/s1** → ntpdrop
smoke then s0/s1 → sigreg s0/s1 at λ=0.001 → v1a2b s2/s3. CENTER runs second
because C2 is the sharpest question in the plan.

**To check:** `tail /home/wolfe/morph-scratch/queue2/queue.log`, then
`python lab/divergence/score_0827_arms.py <label>=<run.log> ...`.
**To stop:** `kill <pid>` found with `ps -eo pid,args | grep overnight_queue`,
NOT `pgrep -f`, which self-matches.

## State at handoff

| arm | status |
| --- | --- |
| v1a (mux, no detach) | FAILED, filed in `../experiments/failures/` |
| v1a-2a (detach, β=1.0) | done: no abort, +0.65 nats tax |
| v1a-2b (detach, β=0.1) | done seeds 0 and 1: no takeover on the coarse series, ppl_tok@3250 94.87 / 92.80, both inside the control spread 90.28-105.54 |
| **2b replication seeds 2/3** | queued last in v2. Rule in the v1a-2 doc: real if ≥3 of 4 seeds never take over. Seeds 0/1 keep only the coarse series and are NOT re-run — seed 0's max there is 0.026, twenty times under threshold. |
| warmup / center / SIGReg / NTP-dropout | implemented, tested, pre-registered, RUNNING in v2 |

## What the logs already say, before any new arm finishes

Measured on CPU from logs that already existed, so it costs nothing to know:
`val/plan_nats` at step 3000 is **0.0049 / 0.0037 / 0.0027** on control seeds
1/2/3 against **0.0096 / 0.0092** on v1a-2b seeds 0/1 — the MUX arm's plan is
worth about twice the control's best seed. Two seeds against three is not a
result, and `val/plan_nats` gathers the slots OUT while the criterion below
zeroes `prefix_project`'s values, so **this is not P**. It is the first sign the
head did something to the plan and not only to the loss.

First fine-probed arm, `tul-warmup-s0`: core share **0.8855 at step 411**, and
0.001-0.02 on the other 515 probed steps. The coarse series never sees it
(max 0.0110) and the shipped 30 %-of-50 rule correctly does not fire. Read the
shipped rule, not the literal first crossing — see the plan's confounds.

## The three arms Wolfe asked for, in build order

### 1. MUX warmup — `tul.mux_activate_at` (his idea: build representations first)

Model-side only, easy. The head is off until step `activate_at × steps`, then on.
Rationale: MUX starts from a PRETRAINED model; we asked a random-init model to
predict next-span bags with no representations, and it learned only the marginal
(7.03 vs unigram 7.32). Precedent for the schedule: `tul.activate_at` already
does exactly this shape for the layout.

### 2. SIGReg on the slot states (LeJEPA, arXiv 2511.08544 Algorithm 1)

Directly attacks a MEASURED pathology: the 50 valid slot states of a row have
effective rank 1.7–4.8 in 1024 dims with mean pairwise cosine +0.39 to +0.71 at
every checkpoint, healthy ones included
(`../experiments/failures/2026-08-24-tul-takeover-cure.md`). SIGReg enforces
E[Z]=0, Cov(Z)=I on those states via a sketched Epps-Pulley test.

Algorithm 1, transcribed exactly: 256 random unit directions (column-normalised
`randn(d, M)`, seeded by global step); project; 17 trapezoid knots on
t ∈ [−5, 5]; Gaussian window `exp(−t²/2)`; empirical CF `mean_n exp(i·t·⟨a,z⟩)`;
loss `trapz(|ecf − exp_f|² · exp_f, t) · N`, then MEAN over directions (def. 2
uses the average, not the max, to avoid sparse gradients). One hyperparameter λ.

Caution worth stating: our slot states have norm 200–450 and the blocks are
pre-norm, so forcing N(0, I) is a large change in absolute scale. Start λ small.

### 3. NTP dropout — ~10 % of steps with `slot_layout=None`

Design note with the full rationale, kernel answer, and the risk:
[`.agents/notes/proposed/feature/2026-08-27-ntp-dropout.md`](../../.agents/notes/proposed/feature/2026-08-27-ntp-dropout.md).
Gated on arm 1 succeeding (Wolfe's sequencing). Hardest of the three: a TUL batch
has slot tokens inserted, so it is NOT a valid NTP batch — the loader must also
emit the pre-insertion token sequence (the packer already has it).

**Wolfe's mechanism, which differs from the risk in the note and is worth keeping
both:** he expects an attractor pulling the coda toward solving the whole
prediction itself and ignoring the loop; NTP steps make the CORE do legitimate
token work, which should weaken that attractor. The note's risk is about the
CODA learning to cope without the plan. Both act, on different modules.

## The criterion that decides all three

`plan-off worth` (`lab/divergence/slot_path_worth.py`), not perplexity. Baseline
0.0191 nats. **An arm that improves CE and takeover while leaving plan worth flat
is stabilisation, not the goal** — the goal is a plan the coda relies on. This
has never been measured on any 2a/2b checkpoint; measure it on every arm.
