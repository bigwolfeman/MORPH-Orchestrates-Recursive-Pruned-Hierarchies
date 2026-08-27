# Experiment v1a: does the MUX head make the plan load-bearing, without candidates?

Status: **failure.** Run 2026-08-25 (`tul-v1a-s1`, wandb 1q4geafo). ALL FIVE
predictions failed. The arm ACCELERATED the takeover by ~1200 steps rather than
causing one: the control takes over too, ~1200 steps later, and simply does not
reach the abort threshold inside 3500 steps. Results, verdict and the updated
hypothesis are at the bottom. Predictions below are frozen exactly as written
before the run. Split out of
[the soft-min arm](2026-08-25-gradient-flow-soft-min-arm.md) (now v1b) so each
lever family is tested alone — MUX-alone costs ≈ nothing, so it goes first.

## Question

The plan is empty because its only direct supervision is a one-token race it
loses (the 2026-08-25 pivot; core Shapley 0.0007 nats, plan-off 0.0191, `cf` < 0
everywhere). Two minimal changes: retire that race, and give z direct span-level
content gradient through the MUX local head (arXiv 2607.18264). No candidate
latents, no soft-min — z stays deterministic. Does that alone make the plan
load-bearing and keep the takeover from firing?

## Hypothesis

The MUX head fills z with next-span content (its gradient does not route through
the coda's suppressed readout), Prop 16 preserves the coda's routing to z as the
local loss falls, and retiring `ce_emit` removes the fuel of the 90%-norm/
zero-value gradient war. Deterministic-z ceiling accepted: the loss minimizer is
the expected bag given context (proper, CE-shaped); the K-candidate mode-committing
upgrade is v1b's question.

## Arm

Config: `morph/configs/tul_v1a.yaml` — `tul_a1` plus `emit_weight 0.0 /
plast_weight 1.0 / mux_beta 1.0 / mux_rho 0.9 / mux_tau 1.0` (the paper's values).
Fresh run, batch 6, 3500 steps, seed 1, `eval_every 250`, `ckpt_every 500` —
exactly the seedsweep protocol, so the CONTROL is the existing
`seedsweep-s1` run (its log and checkpoints at /home/wolfe/morph-scratch/seedsweep/).
`ademamix_t_beta3` stays null on BOTH: for fresh same-length runs the
`training.steps` fallback is matched by construction (the 5000 pin in v1b's notes
applies to resume-based arms only).

Implementation (shipped with this doc, all defaults bit-identical to A1 —
`tests/test_tul_mux.py`, 6 tests):
- `morph/model/tul.py` — `TULConfig.{emit,plast}_weight`, `mux_{beta,rho,tau}`,
  `mux_span_targets()` (sparse geometric targets; the dense |V| vector is never built).
- `morph/model/transformer.py` — `_tul_mux_loss` (reads h_slots through the
  model's OWN `_readout` → unembedding; zero new parameters), loss fold-in with
  `mux_weighted` exposed so train/loss and val loss stay the MODEL's CE
  (the spectral-penalty precedent).
- `morph/training/train.py` — subtracts `mux_weighted` from reported CE; logs
  `tul/mux_local` per step and `val/mux_local` at eval.

## Predictions (frozen 2026-08-25, before any run)

Baselines: plan-off worth 0.0191 nats and raw slot/token readout ratio 0.055
(ROLL_step_1750, same config family at batch 6); A1 aborts at 1800–2940 by seed
(seedsweep, batch 6, 3500 steps); control curves = `seedsweep-s1`.

- **P1 (survival):** v1a reaches step 3500 with no takeover abort and median
  core pre-clip gradient share < 0.5.
- **P2 (value):** plan-off ablation cost (`slot_path_worth.py`) at the step-3000
  checkpoint ≥ **0.04 nats** (>2x baseline).
- **P3 (readout):** raw slot/token Jacobian ratio (`readout_jacobian.py`) at
  step 3000 HIGHER than at the arm's own step 1500 — the falling trend reverses.
- **P4 (no tax):** `val/ce_tokens` at step 3000 within **+0.05 nats** of
  seedsweep-s1 at the same step.
- **P5 (head bites, refuter):** at step 3000 the mean mux local CE beats 0.8x the
  corpus **unigram prior's** CE against the same targets (best span-independent
  predictor, same batches). If it cannot, z carries nothing span-specific and
  the head is decorative regardless of P1–P4.

Failure reading: P1 holds but P2/P3/P5 fail → the war was `ce_emit` alone and
deterministic z stays empty → v1b (K candidates) is the next question, already
pre-registered.

## Method notes

- Sequential runs only on the 5090 (UPS). wandb on, full config (the tul manifest
  now carries the mux fields).
- Seed-matched single-run comparison is unreadable for CE deltas (6.5% spread
  memory) — P4's +0.05 tolerance is deliberately loose; P2/P3/P5 are within-run.


---

# Results (filled 2026-08-26, after the run)

## What the run did

Aborted at **step 2800** on the trainer's sustained-divergence guard
(`DIVERGED_step_2800.pt`). The control `seedsweep-s1` — identical protocol, no
MUX head, same seed — did **not** abort and was still healthy past step 3250.

**But the control takes over as well.** Core pre-clip gradient share crosses in
BOTH runs: the arm at step **1500** (0.099 → 0.533 → 0.964), the control at step
**2700** (0.143 → 0.969); run medians 0.135 (arm) vs 0.001 (control). So the MUX
head did not create the takeover — it moved the onset ~1200 steps earlier, which
is what pulled the abort inside the 3500-step window. Any reading of this arm as
"induced a divergence the control did not have" is wrong, and an interim report
of mine said exactly that before the gradient-share series was read.

| step | v1a val ppl_tok | control val ppl_tok |
| --- | --- | --- |
| 1250 | 161 (arm's minimum) | — |
| 2500 | 516 | 100 |
| 2750 | 973 | 117 |
| 2800 | **ABORT** | healthy to 3250 (105) |

## Verdicts

| Prediction | Threshold | Measured | Verdict |
| --- | --- | --- | --- |
| P1 survival | reach 3500, no abort | aborted 2800 | **FAILED** |
| P2 plan worth | ≥ 0.04 nats | +0.0088 / +0.0189 / +0.0036 at 500 / 2000 / 2500 | **FAILED** |
| P3 readout | ratio rises | 0.0325 → 0.0088 → 0.0045 (fell ~7x) | **FAILED** |
| P4 no tax | within +0.05 nats of control | ~+1.6 nats at matched steps | **FAILED** |
| P5 head bites | < 0.8 × unigram (5.858) | best 7.027 vs unigram 7.323 | **FAILED** |

P1's SECOND clause (median core share < 0.5) actually HELD — 0.135 against the
control's 0.001 — so P1 fails on the abort clause alone. P2's step-3000
checkpoint does not exist (aborted at 2800); 500 / 2000 / 2500 were measured
instead.

**A hypothesis this arm refutes on the way past.** `emit_weight` was 0.0 for the
whole run, so the slot's private one-token race carried no gradient at all — and
the takeover still happened, EARLIER. The pre-registration's failure reading
("P1 holds but P2/P3/P5 fail → the war was `ce_emit` alone") is therefore dead:
retiring `ce_emit` does not prevent the takeover. Artifacts and commands:
[`../results/2026-08-25-mux-head-v1a/`](../results/2026-08-25-mux-head-v1a/).

**Reporting correction.** An interim report of this run claimed the head learned
"11.20 → 6.36". 6.36 was the MINIMUM of the per-training-batch `tul/mux_local`
series over 140 logged points — a minimum over noise, optimistically biased. On
the fixed 8-batch eval set the best is **7.027** (step 1000). The honest
statement is "4 % better than the corpus unigram prior", not "learned well".

## Why it failed — two findings, one mechanical and one about the idea

**1. The gradient path was wrong (mechanical, mine).** `_tul_mux_loss` read
`embed.lm_weight()` with no `detach`, and that matrix is **weight-tied to the
input embeddings**. The auxiliary therefore trained the embedding table itself —
densely, since a softmax puts gradient on every vocabulary row every step —
which is both a corruption of every token representation and a feedback loop,
because the slot input is `E_slot + mean(embed(span))`, a bag-mean of the same
table. MUX never hits this: they LoRA-finetune a PRETRAINED model and use W as a
fixed readout (their §11.1 / Table 9). Fixed in `d3a86da`
(`TULConfig.mux_detach_head`, default true) and put under test with an exact
discriminator — vocabulary rows absent from the batch, which the dense readout
term reaches and the legitimate slot-input path cannot. Whether this caused the
DIVERGENCE is [arm v1a-2a](../planned/2026-08-26-mux-detached-head-v1a2.md).

**2. The target's minimizer is the blur (about the idea, and the bigger
finding).** The head converged to essentially the corpus marginal: 7.027 against
a unigram baseline of 7.323, then above it. That is what theory says must
happen. A DETERMINISTIC z trained with CE/KL against a MULTIMODAL target has the
conditional mean for a minimizer (XM arXiv 2607.27372 §F.2), and for "bag of the
next span in OpenWebText" the conditional mean sits close to the corpus
marginal. The plan did not fail to learn its target; it learned it correctly,
and the correct answer for a deterministic plan IS the blur.

MUX works on GSM8K because the reasoning span is nearly determined by the
question — low conditional entropy. Our next span is not. Their Limitations
(§12) already flag that their evaluation is mathematical reasoning and parallel
search, and that other domains are future work; this is the transfer boundary.

## Updated hypothesis

The empty plan is not a supervision GAP that a richer target closes. It is a
BLUR: any deterministic plan trained by a mean-seeking loss against a
high-entropy future converges to the marginal, whatever the target's shape.
Giving the plan a span-level target does not change the minimizer — only
breaking determinism does.

That makes best-of-K the mechanism, not an upgrade:
[arm v1b](../planned/2026-08-25-gradient-flow-soft-min-arm.md) is now the main
line, and this arm's own head has a second life inside it — scoring K candidates
by the MUX local loss (slot readout only) rather than by span CE through the
coda, which prices K candidates at K × core-on-slots plus K × a small head
instead of K × coda. The 1.81x measured for coda-folding at K=4
([cost benchmark](../results/2026-08-25-coda-k-cost/)) is likely a large
overestimate for that design; it is unbenchmarked.

## Scoring note for the follow-up arm

Because the control also takes over, "reaches 3500 without aborting"
([v1a-2a](../planned/2026-08-26-mux-detached-head-v1a2.md) Q1) is a weaker
criterion than it looked when it was written — the control clears it while still
taking over. The frozen prediction stands as written and will be scored as
written, but the richer signal is the takeover ONSET step: arm 1500, control
2700. A detached head that pushes onset back toward 2700 has done something even
if Q1 would have passed anyway.

## Process failure worth recording

The orchestrator ran `git commit` (`d3a86da`) while the forensics agent was still
writing files, sweeping that agent's outputs and its new probe
(`lab/divergence/mux_unigram_baseline.py`) into an unrelated commit and leaving
the code state during P3's measurement ambiguous. The ambiguity is harmless HERE
— `readout_jacobian.py` backwards `ce_main` only, and the commit's sole
`transformer.py` hunk is inside `_tul_mux_loss`, which no `ce_main` backward can
reach — but the rule it breaks is real: do not commit a shared tree while an
agent is writing into it.

## Not verified

The definitive P3 re-run under the post-commit code OOM'd (the v1a-2a training
run holds 22 GB); the argument above that the change cannot affect a `ce_main`
backward is reasoning, not a re-measurement. The takeover-onset contrast is n=1
per run. Finding 1 is proven to EXIST (the contract test fails when the detach is
reverted) but is not proven to have CAUSED the divergence — v1a-2a is that
ablation. The blur reading (finding 2) rests on two agreeing measurements, not
on an intervention; the thing that would prove it is K candidates producing
differentiated span losses.
