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

`ignore/overnight_queue.sh`, launched 2026-08-27 ~01:46 with `setsid nohup`.
Progress log: `/home/wolfe/morph-scratch/queue/queue.log`. Per-run logs:
`/home/wolfe/morph-scratch/queue/<name>-s<seed>/run.log`.

It waits for a clear GPU between every job (`pgrep` on the trainer), so it is
safe alongside the replication that was already running. Order:

1. SIGReg magnitude probe (100 steps, λ=1e-8 so the term cannot move the model),
   then λ chosen mechanically by the pre-registered rule — weighted term ≈ 5 %
   of `train/loss`, one significant figure — and written to
   `/home/wolfe/morph-scratch/queue/sigreg_lambda.txt`.
2. `tul_warmup` seeds 0, 1.
3. `tul_sigreg` seeds 0, 1 at the probed λ.
4. `tul_ntpdrop`: a 200-step SMOKE first (this path has no unit test — it needs
   a real loader), then seeds 0, 1 only if the smoke exits 0.

Roughly 4 hours after the replication finishes. **To check:**
`tail /home/wolfe/morph-scratch/queue/queue.log`. **To stop:**
`kill <pid of bash ignore/overnight_queue.sh>` — check with
`ps -eo pid,args | grep overnight_queue`, NOT `pgrep -f`, which self-matches.

## State at handoff

| arm | status |
| --- | --- |
| v1a (mux, no detach) | FAILED, filed in `../experiments/failures/` |
| v1a-2a (detach, β=1.0) | done: no abort, +0.65 nats tax |
| v1a-2b (detach, β=0.1) | done seed 1: no takeover, CE inside control spread |
| **2b replication seeds 0/2/3** | **RUNNING** — decision rule in the v1a-2 doc. Seed 0 passed step 1400 with no abort; the CONTROL aborts at seed 0, so this is already the sharp test going the right way. |
| warmup / SIGReg / NTP-dropout | implemented, tested, pre-registered, QUEUED |

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
