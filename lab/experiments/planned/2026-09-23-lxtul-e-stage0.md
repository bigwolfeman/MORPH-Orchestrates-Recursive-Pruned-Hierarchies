# Planned: LXTUL-E Stage 0, the width budget of the ruler's cells under a committed reader

Status: planned

Date: 2026-09-23 (frozen before any GPU step). Design:
[`2026-09-23-provable-loop-contribution.md`](../../../.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md)
(its P-S0 is carried here unchanged). Build: ddbf1eb (merge 5900296). Wolfe's go:
2026-09-23 ("lets try it. Build it with an opus agent.").

## Question

Given the ruler's frozen exit cell z, how much does a committed parallel reader of the next
span gain from four enumerated codes over one? The Lean result `product_reader_le` caps a
parallel reader of one fixed latent at the product of the span's marginals;
`enumerated_pair_gt` says enumerated codes under the exact mixture lift that cap wherever the
span has joint structure. B0 measures how much joint structure web text's next span has,
given the ruler's cell, that four codes can reach.

## Hypothesis

Web text's next span, given the cell, has correlated tokens (a named entity, a phrase, a
code identifier), so a mixture of four product reads beats one. The effect is modest,
because four modes cover little of a span's joint law.

## Arms

Both resume the kept `slot-spandec-strict/step_5000.pt` weights only
(`training.init_from`, `init_from_new_modules: [tul_spandec_par.]`), freeze everything but
the head (`train_only`), run the frozen model in eval mode (`frozen_eval`), and train the
head 2000 steps (AdamW 3e-4, warmup 100, cosine to 3e-5) on batches the ruler never saw
(`data_skip_batches: 5000`).

| arm | config | one factor |
|---|---|---|
| `e0k1` | `tul_slot_spandec_strict_e0k1.yaml` | one reader, no code |
| `e0k4` | `tul_slot_spandec_strict_e0k4.yaml` | four enumerated codes, exact mixture |

## Instrument

`lab/divergence/lxtul_e_stage0_score.py` on the 480 validation rows the sweeps use: the SAME
z for both heads (the frozen tensors are identical); `B0 = CE_par(e0k1) − CE_par(e0k4)` per
span token, paired 95 % block bootstrap over 1,024-token stream blocks; B0 by
position-in-span bin; the e0k4 code usage (responsibility entropy, share of spans each code
wins); and the frozen ruler's own teacher-forced span-decoder CE on the same tokens.

## Predictions (frozen)

- **P-S0 (from the note).** B0 >= 0.02 nats per span token: **55 %.** B0 < 0.005 stops the
  design (no Stage 1): **20 %.**
- **P-S1 (the product cap is large).** e0k1's parallel CE minus the ruler's teacher-forced
  span-decoder CE on the same tokens is >= 0.5 nats per token: **80 %.**
- **P-S2 (the codes are used).** e0k4's mean responsibility entropy at the end is above
  0.5 · log 4 nats, and no code wins more than 60 % of spans: **55 %.**
- **P-S3 (where the width goes).** B0 on span offsets 8 and above is larger than B0 on
  offsets 0 to 3: **45 %.**

## Verdict rules

B0 < 0.005: stop LXTUL-E; the committed reader has nothing for width to buy on these cells.
0.005 <= B0 < 0.02: Stage 1 runs with its width priors halved (the note's rule). B0 >= 0.02:
Stage 1 runs as written. P-S2 failing with B0 >= 0.005 means the gain comes from one or two
codes, and Stage 1 should read its code usage first. No clause passes on a cosine.

## Method

One GPU smoke of e0k4 (~50 steps) first, for memory, tok/s and a falling `par_ce`; then
e0k1 and e0k4 in sequence on the 5090 (one trainer at a time), then the scorer. Artifacts:
JSON and run logs in `../results/2026-09-23-lxtul-e-stage0/`.

## Method amendment (2026-09-23 23:44, after a void first launch, before any scored reading)

Predictions unchanged. The first launch (3f03adf, 23:29) is VOID. Its smoke passed (par_ce
11.46 → 7.28 in 50 steps, 7.83 GB, 13.7k tok/s), but a diff of e0k1's step-1000 checkpoint
against the ruler found exactly one of 478 frozen tensors changed: `tul.E_slot` (max |d|
0.10). `init_from` resets the step to 0, and the trainer's TUL activation branch
overwrote the seed's trained slot embedding with the embedding-table mean, so both heads
were training on a cell the ruler never wrote. Fixed in 7f73f0a (`train.e_slot_seeded`;
test `tests/test_init_from_e_slot.py`). A 60-step e0k1 run on 7f73f0a then diffs at 0 of
478 frozen tensors changed. The void run was stopped at step ~1100 of e0k1, its checkpoints
and its wandb run id were deleted, and nothing from it is scored.

Also measured on the void run: the val loss moved between evals (4.4765, 4.5431, 4.4382,
4.3248) because the val loader rewinds only on curriculum runs; each eval reads different
documents. So the trainer's val loss is not a frozen-model check; the tensor diff is.

The chain reruns on 7f73f0a, same configs.
