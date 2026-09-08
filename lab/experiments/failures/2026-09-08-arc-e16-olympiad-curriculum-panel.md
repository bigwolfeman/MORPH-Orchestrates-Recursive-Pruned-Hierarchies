# Planned: ARC E16 — the Olympiad curriculum panel: three TUL variants and the plain control on synthetic math, 6k steps, the data's own curriculum

Status: failure (the predictions did not hold: P16c, P16d, P16i and the replay clause of P16f are FALSE)
Date: 2026-09-08 (frozen before launch; replaces E15, rejected before data. Wolfe 2026-09-07:
"do 6k steps ... this has a curriculum system start on 2.1 ... take a quick pit stop and
optimize this though").
Arc: `2026-09-04-loop-contribution-arc.md` (amendments 6 and 7).

## Question

E15's question, unchanged: every loop-contribution number in this arc is on web text, where
no lever moves the loop's depth and every stable loop finishes by iteration 3. Wolfe's read:
looped models earn on THINKING problems. Olympiad-AI's synthetic math has a question, a
scratchpad of numbered reasoning steps one per line, and an answer block; MORPH's boundary
rule cuts ONE SPAN PER STEP (mean 12 tokens, 10–14 spans per doc), so a slot is a step.
Does the slot loop contribute more here, does its depth matter for the ANSWER, and how do
the arms move through the curriculum's bands?

## Method

Four arms, one seed each, 6,000 steps, seq 512, effective batch 24 as micro 12 × accum 2
(the audit's frontier: `lab/perf/2026-09-08-oly-throughput-audit/`), through the
curriculum loader in FOUR stages of 1,500 steps at one length that walk the Olympiad bands
in order with ~15 % replay of the band just left (`oly_data.yaml`; per-band shard views
`b2_3` … `b11_13` from `scripts/olympiad_band_views.py`, per-stage blends):

| stage | steps | blend (token fractions) |
|---|---|---|
| 0 | 0–1500 | b2_3 0.70, b4_5 0.30 |
| 1 | 1500–3000 | b2_3 0.15, b4_5 0.45, b6_7 0.40 |
| 2 | 3000–4500 | b4_5 0.15, b6_7 0.35, b8_10 0.50 |
| 3 | 4500–6000 | b6_7 0.15, b8_10 0.35, b11_13 0.50 |

~74M tokens seen per arm. Ramp 1000, flat 1e-4, hinge 0.9 / clip 4.0,
`core_fixed_point_lambda 1.0`, Poisson slot draw mean 12 / max 16, full BPTT, sustained
tripwire. From the audit: `model.ckpt_grad_iters 4` on every arm (exact) and
`model.tg_scoped_kernels true` on the mask arm (1.63×; NOT bit-identical; it makes the
mask arm's gain-hinge reading the fused one, 0.87 at init instead of the eager 0.94 — the
noise-bias note). Validation is the corpus's own held-out shard (`curriculum.val_source
holdout`: 3,230 docs, bands 2–13, rewound before every eval so each eval and each arm
scores the same docs; `training.eval_every 250`, 20 batches × 12 rows).

| arm | config | what it is |
|---|---|---|
| mask | `tul_oly_mask.yaml` | M-next-mask: forecast MUX, tokens reach earlier steps only through the slot (TG-scoped kernels) |
| notul | `notul_oly.yaml` | the plain model (TUL never activates; core mean 6 / max 8), the control |
| mnext | `tul_oly_mnext.yaml` | M-next: forecast MUX, tokens keep the direct route |
| a1 | `tul_oly_a1.yaml` | A1 classic: bag-mean seed, emit/plast 0.5/0.5, no MUX, no mask |

Order: mask, notul, mnext, a1. Runner `arc/run_e16.sh` (smoke of 12 steps with
`curriculum.stages=[{… steps: 12}]` AND `training.steps=12`, an eval at step 6; then the
draw; then the readout on every checkpoint). Checkpoints at 1500, 3000, 4500, 6000 (the
end of every stage).

Readout: `lab/divergence/olympiad_sweep.py` on 1,000 held-out documents (seeded interleave
of the two shards' `eval_holdout.jsonl`; the same docs the in-run val draws from) at every
checkpoint, forced depths 1, 2, 3, 6, 9, 12, 16 (slot depth on the TUL arms;
`model.cfg.mean_depth` on notul). Per depth: token CE, ANSWER-region CE and top-1 accuracy
(the tokens after `<|A|>` up to `<|/A|>`), span-first CE, forecast `mux_local` where built,
per-band answer CE / accuracy. Units are DOCUMENTS, so arms pair offline. K-diffs: K1−K6,
K3−K6, K6−K12, K1−K16, paired over documents. Rulers: E13 on web text (mask tokens K1−K6
+0.049, K3−K6 +0.0018; mnext +0.0011; a1 0.0000). Wall clock from the queue-log epochs. A
checkpoint of an arm that later trips is pre-onset and not cited.

### Method amendment 8 (2026-09-08 08:24; after the data, readout only — the run's method is untouched)

The mask arm's held-out `val/loss` reached 0.48 at step 5000 with a 0.7–1.2 nat cliff at
every stage switch. A hash pass over every doc (2026-09-08 02:30) found the eval holdout
CONTAMINATED: 1,324 of its 3,230 docs sit verbatim in a training band view, and the bands
themselves are duplicate-heavy (unique docs: b2_3 95.9 %, b4_5 94.3 %, b6_7 46.6 %, b8_10
29.2 %, b11_13 23.2 %; one band-6/7 doc appears 3,442 times). Bands 7–13 keep 14–35 clean
docs each. So `val/loss` and the runner's 1000-doc sweeps read memorization at bands ≥ 6.
Added AFTER the panel ended, without touching the runs: a re-sweep of all 16 checkpoints
on `holdout_clean/` (the 1,906 holdout docs absent from every band and unique within the
holdout; 85 % bands 2–6), `arc/run_e16_clean.sh`, results in `results/.../sweeps_clean/`.
The predictions were written against the original files; both sets are scored below and the
CLEAN set decides. Decision record:
`.agents/notes/proposed/bug-fix/2026-09-08-olympiad-holdout-contamination.md`.

## Predictions (frozen)

- **P16a (survival).** All four arms reach 6,000 with the tripwire silent: **55 %** (mask
  70 %, mnext 85 %, a1 85 %, notul 95 %).
- **P16b (contribution, tokens).** Mask arm tokens K1−K6 at 6,000 > 0.049 (its web-text
  value) with the CI above it: **60 %**. mnext tokens K1−K6 > 0.005: **35 %**. a1 tokens
  K1−K6 > 0.002: **25 %**.
- **P16c (the answer needs the loop).** Mask arm answer-region K1−K6 (CE) > 2× its token
  K1−K6 at 6,000: **55 %**. Answer accuracy K1−K6 > 0.02 (2 points): **50 %**.
- **P16d (depth, the point).** Mask arm answer-region K3−K6 > 0.01 with the CI above 0 at
  6,000: **35 %** (never seen on text: 0.0018 on tokens). K6−K12 > 0.005: **20 %**.
- **P16e (the control).** notul's answer accuracy at its trained depth 6 is within 3 points
  of the mask arm's at depth 12 at 6,000: **55 %**. notul token K1−K6 (core depth) > 0.037
  (its web-text value): **55 %**.
- **P16f (progression).** At 1,500 (only bands 2–5 seen) the mask arm's answer accuracy on
  bands 2–3 exceeds 80 % while bands 11–13 sit under 40 %: **50 %**. At 6,000 bands 11–13
  exceed 70 %: **40 %**. Replay holds: bands 2–3 accuracy at 6,000 ≥ its 1,500 value − 2
  points on every surviving arm: **60 %**. The held-out `val/loss` never rises by more than
  0.05 across a stage switch (the two evals around 1500/3000/4500): **60 %**.
- **P16g (the mask's price).** Mask token CE at depth 12 minus mnext's at depth 12 at 6,000,
  paired over docs, is under 0.05 (web text: 0.11): **45 %**.
- **P16h (cost).** Mask arm training wall clock ≤ 2.0 h for 6,000 steps (1055 ms/step +
  24 evals × 28 s = 1.95 h): **60 %**. The four-arm panel, sweeps included, ≤ 10 h: **60 %**.
  Mask peak allocated stays under 24.0 GB for the whole run (audit: 22.72 at 60 steps):
  **70 %**.
- **P16i (the hinge on the fused path).** Mask arm `loss/gain_est` mean over steps 1,000–
  6,000 under 0.9 and the hinge penalty nonzero on fewer than 5 % of steps: **65 %**.
- **P16j (the held-out val).** `val/loss` at 6,000 orders a1 < mnext < mask (E13's token-CE
  order on text): **50 %**; notul < a1: **50 %**.

## Binding

- P16d TRUE ⇒ depth is real on math and text was the wrong ruler: the next run is the mask
  arm back on web text with a math replay fraction (Wolfe's call), and E3 (staged targets)
  on math.
- P16d FALSE and P16c TRUE ⇒ the loop matters for the answer but finishes by 3–6 here too:
  contribution without depth is the general law of this loop; the lever is the target (E3)
  or the write-back, not the draw.
- P16c FALSE ⇒ the slot loop does not help solve math either; the masked arm's contribution
  on text was forecast, not thought. Stop the slot-loop lane; the paid loop and the Spiral
  write-back are what is left.
- P16a FALSE on any arm ⇒ its trip step, `loop/core_gain_t0` and `loss/gain_est_max` are
  filed; the divergence README's second-hold table gets the row. A mask trip in the spike
  mode with `loss/gain_est` under 0.9 for the 500 steps before it is the hinge-noise note's
  risk realized and is said so.
- P16f's replay clause FALSE ⇒ the 15 % replay is too thin; the next curriculum run keeps
  30 %.
- NO 20k run from this experiment; Wolfe's call.

## Not verified before launch

The full six-hour path: only a 12-step smoke of the mask arm ran on this tree (two stages of
6 steps, an eval at 6 and 12 on the held-out shard, `kernels=EAGER+TGSCOPED`, proxy 12.14,
peak 21.6 GB). Peak memory under `ckpt_grad_iters 4` past 60 steps. The per-band tables of
the sweep on a checkpoint trained on math (exercised on web-text checkpoints only). The
other three arms' smokes run inside the runner before their draws. Whether 1,000 sweep
docs at 7 depths fit the between-arm gap (the E13 sweep on 480 rows was minutes).

## Results

Launched 00:19 on 2026-09-08 at `effa7b0` (`arc/run_e16.sh`); the panel ended 08:23 (8 h 04 min);
the clean re-sweep ran 08:24–08:50. Artifacts: `lab/experiments/results/2026-09-08-arc-e16/`
(`sweeps/` the runner's 1000-doc original draw, `sweeps_clean/` all 1,906 clean docs at
depths 1,2,3,6,9,12,16, `run_<arm>.log`, `score_e16.py` → `score_e16.txt`; the four probe
JSONL files (15–20 MB) under `ignored/experiment-artifacts/2026-09-08-arc-e16/`, `PROBES.md`).

**Runs.** All four arms reached 5,999 with the tripwire silent. Mask: two ONE-step excursions
(`preclip/total` 1.1e4 at 3116, 1.9e4 at 3550, back to 2–46 the next step; `loss/gain_est`
0.95/0.96 on the spike steps, `loop/core_gain_t0` 3.7 and 8.2), peak 22.85 GB, 2 h 00 min 31 s.
Notul: max pre-clip 25, peak 23.15 GB, 1 h 49 min 37 s. Mnext: max 31.5, peak 22.16 GB,
2 h 01 min 57 s. A1: max 19.8, peak 21.86 GB, 1 h 56 min 50 s. Fused-path hinge over steps
1000–6000: mask `gain_est` mean 0.8997, max 1.0027, penalty nonzero on 80.5 % of steps;
mnext 0.882 / 0.889 / 0.04 %; a1 0.884 / 0.895 / 0.3 %.

**Clean sweeps, token CE (depth 1 / 3 / 6 / 12) and K1−K6 with its 95 % CI:**

| arm | ck | ce@1 | ce@3 | ce@6 | ce@12 | tok K1−K6 | ans K1−K6 | ans K3−K6 | acc@6 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| mask | 1500 | 1.478 | 1.434 | 1.433 | 1.436 | +0.044 [0.040, 0.049] | +0.023 | −0.003 | 0.860 |
| mask | 3000 | 1.153 | 0.900 | 0.889 | 0.885 | +0.264 [0.258, 0.270] | +0.073 | −0.000 | 0.959 |
| mask | 4500 | 1.183 | 0.839 | 0.823 | 0.824 | +0.360 [0.349, 0.373] | +0.240 | +0.000 | 0.966 |
| mask | 6000 | 1.313 | 0.903 | 0.907 | 0.917 | +0.406 [0.395, 0.418] | +0.159 | −0.030 [−0.040, −0.021] | 0.948 |
| notul | 6000 | 0.758 | 0.729 | 0.734 | 0.733 | +0.024 [0.021, 0.027] | −0.016 | −0.006 | 0.971 |
| mnext | 6000 | 0.870 | 0.840 | 0.871 | 0.890 | −0.002 [−0.007, +0.004] | −0.022 | −0.006 | 0.966 |
| a1 | 6000 | 0.742 | 0.739 | 0.737 | 0.733 | +0.005 [0.003, 0.007] | −0.002 | +0.001 | 0.970 |

(the full 16-row table for both holdouts: `score_e16.txt`). Paired over the same clean docs
at depth 12, step 6000: mask − mnext +0.028 [0.017, 0.038]; mask − notul +0.185 [0.170,
0.199]; mnext − notul +0.157 [0.144, 0.169]; a1 − notul +0.000 [−0.010, +0.011]. Answer
accuracy at depth 12 by band (clean; bands 8–13 carry 14–35 docs each): every arm reaches
0.97–0.99 on bands 11–13 by 6000, and every arm LOSES 3–7 points on bands 2–3 between 1500
and 6000 (mask 0.978 → 0.906, notul 0.987 → 0.957, mnext 0.981 → 0.948, a1 0.980 → 0.948).
On the clean set every arm is WORSE at 6000 than at 4500 (notul 0.731 → 0.734, a1 0.724 →
0.737, mask 0.823 → 0.907, mnext 0.766 → 0.871 at depth 6): the last stage (bands 8–13 at
85 %, replay 15 %) forgets the bands the clean set is made of, and the two loop arms with a
loop target forget most.

**Original (contaminated) sweeps** tell the same ordering at 6000 (depth 12: notul 0.493,
a1 0.497, mnext 0.584, mask 0.611; mask − mnext paired +0.026 [0.015, 0.038]) with the
memorized bands pulling every CE down. Contaminated `val/loss` at 5750: notul 0.491, a1
0.501, mnext 0.594, mask 0.614; it DROPS 0.56–1.22 nats at every stage switch.

**Scores (clean set decides).** P16a TRUE (4/4; the mask excursions were one-step and
recovered). P16b: mask tokens K1−K6 = 0.406 ≫ 0.049 TRUE; mnext −0.002 FALSE; a1 0.005 TRUE
(by 0.003). P16c FALSE: the mask arm's answer K1−K6 (0.159) is 0.39× its token K1−K6, not
2×; answer accuracy does rise 6.1 points from depth 1 to 6, but from a depth-1 hole the arm
dug for itself (1.31 nats at depth 1) to a floor 2.4 points BELOW the plain model's. P16d
FALSE twice: answer K3−K6 is −0.030 with the CI below 0 (deeper is worse past 3 at 6000) and
K6−K12 is −0.004. P16e: notul acc@6 0.971 vs mask acc@12 0.947, within 3 points TRUE (the
control is ahead); notul tokens K1−K6 0.024 < 0.037 FALSE. P16f: bands 2–3 > 80 % at 1500
TRUE (0.978), bands 11–13 < 40 % FALSE (0.654 on 93 unseen docs), bands 11–13 > 70 % at 6000
TRUE (0.972), replay FALSE on all four arms (−3.0 to −7.2 points), the val clause TRUE by
the letter and void (the val is contaminated and falls at switches). P16g TRUE (+0.028 <
0.05). P16h: mask 2 h 00 min 31 s FALSE by 31 s; panel 8 h 04 min TRUE; peak 22.85 < 24.0
TRUE. P16i FALSE: gain mean 0.8997 sits on the target and the hinge fires on 80.5 % of
steps. P16j TRUE twice: a1 < mnext < mask and notul < a1 (0.518 < 0.521 contaminated; tied
at 0.733 clean).

## Verdict

Failure: the predictions that carried the question (P16c, P16d) are FALSE and the loop's
depth reading on math is the text reading. The masked slot loop is the only arm that
depends on depth, and it depends on it because depth 1 is broken for it (1.31 nats vs the
plain model's 0.76); it finishes by iteration 3 and is worse past 6; at its trained depth
it sits 0.18 nats BEHIND the plain model on clean math, paired, and 2.4 accuracy points
behind on the answer tokens. The two loop arms with a loop target (mask, mnext) are the
two that forget the early bands most in the last stage. Mnext and a1 are depth-flat, and
a1 ties the plain model to 0.000 ± 0.010. Binding, applied: P16c FALSE ⇒ the slot loop does
not help solve math either; the masked arm's contribution on text was forecast, not
thought; stop the slot-loop lane, the paid loop and the Spiral write-back are what is left
(Wolfe confirms before any config is retired). P16f replay FALSE ⇒ the next curriculum run
keeps 30 % replay. P16i FALSE ⇒ `.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md`
is amended: the fused hinge at 0.9 does NOT go quiet on the mask arm; the map drifts up to
the target and sits on it. The data finding stands on its own: the Olympiad holdout is
contaminated and bands 6–13 are 23–47 % unique
(`.agents/notes/proposed/bug-fix/2026-09-08-olympiad-holdout-contamination.md`); no
further Olympiad run before the band views are deduped and the held-out shard rebuilt.

## Updated hypothesis

The loop's depth dependence is a property of the TARGET, not the data: a masked slot target
makes the slot loop need 2–3 iterations on math exactly as on text, and never more; no
target so far makes a stable loop earn past iteration 3, and the arms whose loop earns the
most are the arms furthest behind the plain model. What is left to test is the write-back
(the Spiral schedule, the paid loop) and staged targets (E3), on deduped data with a
matched-compute shallow control and 30 % replay.
