# Planned: LXTUL-E Stage 0, the width budget of the ruler's cells under a committed reader

Status: success

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

## Results (filed 2026-09-24 00:27)

Artifacts: [`../results/2026-09-23-lxtul-e-stage0/`](../results/2026-09-23-lxtul-e-stage0/)
(`stage0_score.json`, run logs of the chain, the smoke, both arms and the scorer). Rerun
on 7f73f0a. Frozen-tensor diff of each arm's step-2000 checkpoint against the ruler: 0 of
478 changed (e0k1 and e0k4). Scorer self-check against the model's own forward: max |dev|
8.9e-7 (parallel term), 6.5e-7 (ruler decoder). 480 rows, 24,339 spans, 486,031 span
tokens, 490 blocks. Both arms read val 4.4912 at step 1750: the same frozen model on the
same val stream.

| reading | value, 95 % CI |
|---|---|
| CE_par, K = 1 | 6.8943 [6.8703, 6.9189] |
| CE_par, 4-code mixture | 6.7912 [6.7662, 6.8173] |
| CE, the ruler's teacher-forced span decoder, same tokens | 4.4839 [4.4406, 4.5295] |
| **B0 = K1 − mixture** | **+0.1031 [+0.1001, +0.1063]** |
| mixture gain over its best single code | +0.2545 [+0.2468, +0.2622] |

B0 by offset in the span: 0: −0.0099 [−0.0161, −0.0039]; 1: +0.3072; 2: +0.3867; 3:
+0.1944; 4–7: +0.1263; 8–15: +0.0718; 16+: +0.0485 [+0.0442, +0.0528]. Code usage: mean
responsibility entropy 0.333 of log 4 = 1.386; share of spans each code wins 0.223, 0.227,
0.253, 0.296; each code read alone 7.09, 7.28, 7.17, 7.05.

| clause | reading | verdict |
|---|---|---|
| P-S0 B0 >= 0.02 (and not < 0.005) | +0.1031 [+0.1001, +0.1063] | held |
| P-S1 K1 parallel CE − teacher-forced CE >= 0.5 | 2.41 | held |
| P-S2 entropy > 0.5 log 4 AND no code > 60 % | entropy 0.333; top share 0.296 | failed (entropy clause) |
| P-S3 B0 at offsets 8+ larger than at offsets 0–3 | 0.07 / 0.05 vs 0.31 / 0.39 at 1–2 | failed |

## Verdict

Success on the deciding clause. Four enumerated codes under the exact mixture buy 0.103
nats per span token over one committed reader on the ruler's frozen cells, five times the
note's go threshold. By the verdict rule, Stage 1 runs as written.

What the numbers say about the mechanism:

- **The product cap is large.** A parallel reader of one cell sits 2.41 nats per token
  behind the teacher-forced decoder on the same cell. Width closes 0.10 of it (4 %).
  `product_reader_le` was the prediction; the size was not.
- **The codes act as a span type chosen by the span's first token.** B0 is slightly
  negative at offset 0, where the mixture's weights are still uniform, and peaks at
  offsets 1 and 2 (0.31, 0.39), right after the first tokens have picked a code. It decays
  but stays positive to offsets 16+ (0.049).
- **P-S2 failed on a clause I set wrong.** Low responsibility entropy (0.33 of 1.39) means
  each span is assigned DECISIVELY to one code, and the wins are balanced (22 % to 30 %).
  All four codes are used. The prediction read low entropy as collapse; it is the
  opposite. The clause stays failed as written.

Unverified: one seed per arm; B0 includes the difference between two separately trained
heads (the note's named risk); 2000 steps may not have converged either head.

## Updated hypothesis

The ruler's cell has joint next-span structure that a committed reader can only reach
through enumerated codes, and the amount is not small (0.10 nats per token). Stage 1 asks
whether a loop that carries a re-injected code integrates it with depth (the note's P-1
to P-5), with the width priors unchanged.
