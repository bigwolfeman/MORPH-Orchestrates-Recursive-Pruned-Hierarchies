# Planned: ARC E15 — the Olympiad panel: three TUL variants and the plain control on synthetic math

Status: failure (rejected run: method replaced before the first checkpoint)
Date: 2026-09-07 (frozen before launch; Wolfe: "Olympiad AI ... its all synth math, it should
show better loop contribution on something that needs it ... do 15k steps and see how they
progress through the curriculum too. and a larger batch than we've used.")
Arc: `2026-09-04-loop-contribution-arc.md`.

## Question

Every loop-contribution number in this arc is on web text, where the arc found that no
lever moves the loop's depth and every stable loop finishes by iteration 3. Wolfe's read:
looped models earn on THINKING problems. Olympiad-AI's synthetic math has a question, a
scratchpad of numbered reasoning steps one per line, and an answer block; MORPH's boundary
rule cuts ONE SPAN PER STEP (mean 12 tokens, 10–14 spans per doc), so a slot is a step.
Does the slot loop contribute more here, does its depth matter for the ANSWER, and how do
the arms progress through the curriculum's stages over 15k steps?

## Method

Four arms, one seed each, 15,000 steps, seq 512 (every doc ≤ 512 tokens; 20 of 2.18M
carry-split), effective batch 24 (micro 12 × accum 2 — micro 16 peaked 26.53 GB on the
eager mask arm, too close to the card), data 50/50 by tokens from the two Olympiad shards
`olympiad_stage11_13` (367M tok, stages 11–13) and `olympiad_math` (291M tok, stages 2–10)
through the curriculum loader (`oly_data.yaml`; ~184M tokens seen, under one epoch).
Ramp 1000, flat 1e-4, hinge 0.9 / clip 4.0, `core_fixed_point_lambda 1.0`, Poisson slot
draw mean 12 / max 16, full BPTT; checkpoints every 2500 (six per arm), sustained tripwire.

| arm | config | what it is |
|---|---|---|
| mask | `tul_oly_mask.yaml` | M-next-mask: forecast MUX, tokens reach earlier steps only through the slot (eager) |
| mnext | `tul_oly_mnext.yaml` | M-next: forecast MUX, tokens keep the direct route |
| a1 | `tul_oly_a1.yaml` | A1 classic: bag-mean seed, emit/plast 0.5/0.5, no MUX, no mask |
| notul | `notul_oly.yaml` | the plain model (TUL never activates; core mean 6 / max 8), the control |

Order: mask, notul, mnext, a1 (the informative arm and its control first). Runner
`arc/run_e15.sh`. The smoke passes `curriculum.stages=[{…steps: 12}]` as well as
`training.steps=12` because the curriculum's stage sum overrides the step count.

Readout: `lab/divergence/olympiad_sweep.py` on 600 held-out documents (seeded interleave of
the two shards' `eval_holdout.jsonl`, never in a shard) at every checkpoint (2500 … 15000),
forced depths 1, 2, 3, 6, 9, 12, 16 (slot depth on the TUL arms; `model.cfg.mean_depth` on
notul). Per depth: token CE, ANSWER-region CE and top-1 accuracy (the tokens after `<|A|>`
up to `<|/A|>`), span-first CE, forecast `mux_local` where built, and per-stage-band answer
CE / accuracy (bands 2–13). Units are DOCUMENTS, so arms pair with each other offline.
K-diffs: K1−K6, K3−K6, K6−K12, K1−K16, paired over documents. Rulers: E13 on web text
(mask tokens K1−K6 +0.049, K3−K6 +0.0018; mnext +0.0011; a1 0.0000). Wall clock from the
queue-log epochs. A 2500 checkpoint of an arm that later trips is pre-onset and not cited.

## Predictions (frozen)

- **P15a (survival).** All four arms reach 15,000 with the tripwire silent: **40 %**
  (mask 70 %, mnext 80 %, a1 85 %, notul 90 %; 3x the steps of any slot-loop run so far).
- **P15b (contribution, tokens).** Mask arm tokens K1−K6 at 15,000 > 0.049 (its web-text
  value) with the CI above it: **65 %**. mnext tokens K1−K6 > 0.005: **35 %** (the direct
  route wins on text; a step's tokens may need the slot here). a1 tokens K1−K6 > 0.002:
  **25 %**.
- **P15c (the answer needs the loop).** Mask arm answer-region K1−K6 (CE) > 2x its token
  K1−K6 at 15,000: **60 %**. Answer accuracy K1−K6 > 0.02 (2 points): **55 %**.
- **P15d (depth, the point).** Mask arm answer-region K3−K6 > 0.01 with the CI above 0 at
  15,000: **40 %** (never seen on text: 0.0018 on tokens). K6−K12 > 0.005: **25 %**.
- **P15e (the control).** notul's answer accuracy at its trained depth 6 is within 3 points
  of the mask arm's at depth 12 at 15,000: **55 %** (E13: the mask costs tokens 0.11 nats).
  notul token K1−K6 (core depth) > 0.037 (its web-text value): **60 %**.
- **P15f (progression).** At 2500 the mask arm's answer accuracy on bands 2–5 exceeds 80 %
  while bands 11–13 sit under 40 %; by 15,000 bands 11–13 exceed 70 %: **50 %** each.
- **P15g (the mask's price).** Mask token CE at depth 12 minus mnext's at depth 12 at
  15,000, paired over docs, is under 0.05 (web text: 0.11): **45 %**.
- **P15h (cost).** Mask wall clock ≤ 6 h for 15,000 steps (smoke: 0.82 steps/s at micro 16;
  accum 2 at micro 12 ~0.5 steps/s ⇒ ~8 h): **35 %**.

## Binding

- P15d TRUE ⇒ depth is real on math and text was the wrong ruler: the next run is the
  mask arm back on web text with a math replay fraction (Wolfe's call), and E3 (staged
  targets) on math.
- P15d FALSE and P15c TRUE ⇒ the loop matters for the answer but finishes by 3–6 here too:
  contribution without depth is the general law of this loop; the lever is the target
  (E3) or the write-back, not the draw.
- P15c FALSE ⇒ the slot loop does not help solve math either; the masked arm's contribution
  on text was forecast, not thought. Stop the slot-loop lane; the paid loop and the
  Spiral write-back are what is left.
- P15a FALSE on any arm ⇒ its trip step, `loop/core_gain_t0` and `loss/gain_est_max` are
  filed; the divergence README's second-hold table gets the row.
- NO 20k run from this experiment; Wolfe's call.

## Not verified before launch

The curriculum loader's TUL layout on Olympiad docs at scale (the 12-step smoke is the
first), micro 12 × accum 2 on every arm (only the mask arm at micro 16 was smoked), the
sweep script's plain path (notul) and its per-band tables on a real checkpoint (exercised
on an E13 web-text checkpoint before launch), the packer's slot cap (64 at seq 512) against
~43 spans per row.

## Results

Launched 22:25 (commit ea57f69); the mask arm's smoke passed (peak 20.67 GB, 0.43 steps/s at
micro 12 × accum 2) and its draw ran to about step 200 before Wolfe killed the plan at
22:30: 15k steps at that rate is 9.7 h for one arm ("way too slow"); the run should be 6k
steps, should walk the Olympiad curriculum in order from stage 2.1 instead of a uniform mix,
and a throughput audit comes first. No checkpoint, no sweep, no scores.

## Verdict

Rejected: the method changed before any data existed. The predictions are unscored and
stand as the record of what I expected of a uniform 15k mix; the replacement is E16
(`planned/2026-09-07-arc-e16-olympiad-curriculum-panel.md`), written after the audit.

## Updated hypothesis

None from data. Two facts for the next prereg: the curriculum's stage sum overrides
`training.steps` (a smoke must set both), and the eager mask arm's memory is set by the
attention over 640 positions (micro 16: 26.5 GB; micro 24: OOM), not by the slot loop.
