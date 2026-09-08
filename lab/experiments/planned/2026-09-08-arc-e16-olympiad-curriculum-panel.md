# Planned: ARC E16 — the Olympiad curriculum panel: three TUL variants and the plain control on synthetic math, 6k steps, the data's own curriculum

Status: planned
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

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
