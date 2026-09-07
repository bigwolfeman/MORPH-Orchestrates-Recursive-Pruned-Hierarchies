# Failure: loop contribution on Huginn-3.5B (a recurrent-depth model trained to use depth)

Status: failure
Date: 2026-09-04 (frozen before the sweep; eval only; Wolfe's call: "we should take huginn
3.5b and test loop contribution on it")

## Question

Every stable MORPH loop, token or slot, earns nothing past iteration 3 on OpenWebText
(arc branch (a): `failures/2026-09-04-arc-e1-*`, `-e2-*`), and its earning is a flat
refinement of the predictable part of the loss (`failures/2026-09-04-arc-e0-*`). Is that
a property of web text, or of how MORPH is trained (mean depth 6, max 8, 328M positions,
ternary, the gain hinge)? Huginn-0125 (Geiping et al. 2025, arXiv 2502.05171) is the
same shape at scale — 2 prelude + 4 recurrent + 2 coda layers, 3.5B parameters, trained
on 800B tokens with the recurrence sampled r ~ U(4, 64) (mean ~32, truncated backprop
over the last 8) — and it reports per-step gains on reasoning benchmarks up to 64 steps
while the paper's language-modeling loss curve saturates after a handful of steps. This
measures, on the SAME rows and with the SAME instruments as our arms, how much a
recurrent-depth model that was trained to use depth earns from extra iterations on
web text, and where.

## Method

`lab/huginn/huginn_depth_sweep.py` (new): load `tomg-group-umd/huginn-0125` (bf16,
`trust_remote_code`) on the 5090; 480 rows of OpenWebText validation text, 1024 tokens
each under Huginn's own tokenizer (the same source documents the MORPH sweeps use, drawn
in the same order; the tokenization differs, so rows are matched by document, not by
token); forward at `num_steps` ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64} on identical
batches; per-token CE from the logits; per-row and per-offset-in-span sums with the arc's
`_earning.EarningProfile` (spans cut by MORPH's boundary rule rebuilt on Huginn's
tokenizer with `boundary_lut_from_tokenizer`); paired-bootstrap CIs over rows for
K1−K3, K3−K6, K6−K16, K16−K32, K32−K64; `score_arc_e0.py` for the offset profile and
the row-difficulty correlation at (1, 6) and (3, 32). Output
`lab/experiments/results/2026-09-04-huginn-loop-contribution/`. Eval only; no training.

Comparison rulers (ours, same instruments, 480 rows): notul-20k-wu K1−K6 0.041, K3−K6
0.002 (20k), a2 K1−K6 0.104; every slot arm K3−K6 ≤ 0.0014.

### Method amendment, 2026-09-05

The earlier external attempt stopped with exit 137 after depths 1 and 2. Its logged
means were 4.6123 and 3.7694. No completed paired sweep is available from that attempt.
This amendment precedes the corrected attempt and preserves H1-H6 below.

- Pin model and tokenizer to local snapshot
  `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Patch only the in-process Transformers 5
  tied-weight declaration. Assert the loaded embedding and prediction weights alias.
- The source is the sorted local OpenWebText training-shard stream, with skip zero.
  It is not a held-out split. Concatenation followed by Huginn tokenization and row
  packing gives 480 identical Huginn rows across depths, not document-matched MORPH
  rows. Save each row's SHA256 and the combined row hash.
- Reset the RNG to `seed + batch_index` before each forward. This pairs Huginn's
  random initial latent across depths. Use next-token labels and classify CE by the
  predicted token's span offset, including the final target outside the input row.
- Compute every adjacent depth CI as well as the preregistered coarse pairs. H1 uses
  adjacent pairs up to depth 32. Report thresholds separately from broader causal
  interpretations. Failure of H1 and H2 does not prove web text is universally flat.
- The snapshot config says `poisson-lognormal-filling`; the uniform sampler claim
  above is not established by this evaluation. Cross-tokenizer CE values are not
  common units. H6 remains a within-Huginn numerical prediction.
- Hydra config lives in `lab/huginn/configs/depth_sweep.yaml`. Online W&B records the
  complete experiment config, source model config, versions, source hashes, data
  hashes, and each depth. Use bf16 evaluation and one CPU thread. Atomically save
  completed depths for resume. Resume rejects changed experiment, source, or rows.
- Correctness predictions before the smoke: repeated same-seed forwards give
  identical CE; independently computed shifted-label CE agrees with Huginn's loss
  within 2e-5 absolute and relative tolerance; K64 at batch 3 and length 1024 fits
  the available GPU without competing compute jobs. The smoke uses three prefix
  rows and depths 1 and 64. It is a runtime gate, not evidence for H1-H6.
- The corrected output is `ignore/huginn/2026-09-05-corrected/`, with the resolved
  Hydra config, online W&B, atomic result JSON, stdout log, PID and true exit status.
  Preserve all earlier artifacts. Do not call a partial result complete.

### Resume logging repair, 2026-09-05

After the UPS interruption, the capped attempt loaded the model but failed during
W&B metadata comparison. Persisted JSON used string dictionary keys, while the
fresh model configuration used integer label keys. The evaluator now canonicalizes
that logging value through JSON. A one-off migration verifies the exact logging-only
source patch and every unchanged dependency hash. It preserves the original
checkpoint bytes and records old/new source hashes. All measured fields remain
unchanged. The audit is logged on resume. No evaluation math or H1-H6 changes.
See [the repair check](../successes/2026-09-05-huginn-wandb-resume-serialization.md).

## Predictions (frozen)

- **H1.** Huginn's token CE on web text falls monotonically with `num_steps` from 1 to
  32 (each successive pair's paired CI above 0): **80%**.
- **H2.** Huginn's K3−K6 > 0.05 nats: **70%** (3 steps is far below its trained
  regime; MORPH's is 0.002).
- **H3.** Huginn's K16−K32 > 0.01 nats: **55%**; K32−K64 > 0.005: **35%** (saturation
  near the training mean; the paper's LM curve flattens early).
- **H4.** Huginn's earning profile over offset-in-span is flat to ±15 % of its mean at
  (1, 6): **55%**; the span's first token earns the least, as on every MORPH model:
  **50%**.
- **H5.** Spearman(row CE₁, row earning K1−K6) ≤ +0.10: **50%** (earning does not
  concentrate on hard rows).
- **H6.** At its trained depth, Huginn's CE on these rows is below 3.0 nats: **60%**
  (a 3.5B model at 800B tokens on web text; MORPH's plain 20k model reads 3.49 at
  depth 1, 3.45 at 6).

## Decision rule (binding)

- H2 TRUE ⇒ depth earning past iteration 3 on web text EXISTS for a model trained to use
  it: the arc's closing option (i), "data where depth pays", is downgraded and the
  training regime (deep recurrence at train time, long training) becomes the candidate
  cause of MORPH's flat loops. The next MORPH prereg trains the plain loop at a deep
  recurrence draw (mean 16+) with truncated backprop, under the abort rule.
- H2 FALSE and H1 FALSE ⇒ web text is depth-flat for a 3.5B model trained on it too;
  the closing rule's option (i) stands and the loop question moves off web text.
- H4/H5 read the SHAPE either way: a flat profile on Huginn says the flat shape is a
  property of next-token prediction on web text, not of MORPH.

## Not verified before launch

Huginn's remote code under transformers 5.15 (written for 4.x; the load is the first
check); its `num_steps` forward on a labelled batch; the boundary rule rebuilt on a
65k-vocab tokenizer; memory at 64 steps and batch 3 × 1024 (the recurrent block's
activations are not checkpointed at eval).

## Results (scored 2026-09-07; run `ignore/huginn/2026-09-05-corrected/`, exit 0, status complete, 480 rows, 12 depths, batch 3, seq 1024, bf16, peak 8.75 GB, 674 s at depth 64; wandb `morph-huginn-loop-contribution/d21387a3`; files in `results/2026-09-04-huginn-loop-contribution/`)

Per-token CE over all 480 rows (491,520 targets) at each `num_steps`:

| steps | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CE | 4.614 | 3.768 | 3.274 | 2.989 | 2.708 | 2.592 | 2.518 | 2.502 | 2.499 | 2.499 | 2.499 | 2.500 |

Paired row-bootstrap differences (2000 draws, 95 % CI), earlier depth minus later:

| pair | point | CI |
|---|---|---|
| K1−K2 | +0.846 | [+0.833, +0.860] |
| K2−K3 | +0.494 | [+0.487, +0.500] |
| K3−K4 | +0.285 | [+0.281, +0.289] |
| K4−K6 | +0.281 | [+0.277, +0.285] |
| K6−K8 | +0.116 | [+0.114, +0.119] |
| K8−K12 | +0.074 | [+0.072, +0.076] |
| K12−K16 | +0.0151 | [+0.0143, +0.0159] |
| K16−K24 | +0.0034 | [+0.0030, +0.0039] |
| K24−K32 | −0.0001 | [−0.0003, +0.0002] |
| K32−K48 | −0.0004 | [−0.0006, −0.0002] |
| K32−K64 | −0.0005 | [−0.0007, −0.0003] |
| K3−K6 | +0.566 | [+0.559, +0.573] |
| K6−K16 | +0.206 | [+0.201, +0.210] |
| K16−K32 | +0.0034 | [+0.0028, +0.0040] |
| K1−K6 | +1.906 | [+1.884, +1.928] |
| K3−K32 | +0.775 | [+0.764, +0.786] |

`score_arc_e0.py` (`score_1_6.txt`, `score_3_32.txt`, `score_6_16.txt`):

| pair | Spearman(row CE_lo, row earning) | offset-0 / mean(4..31) | earning per token by offset 0, 1, 2, 3, 4-7, 8-15, 16-31 | top-decile loss share / earning share |
|---|---|---|---|---|
| (1, 6) | +0.301 [+0.215, +0.389] | 0.63x | 1.248, 1.448, 1.805, 1.854, 1.948, 1.993, 2.022 | 0.114 / 0.108 |
| (3, 32) | +0.319 [+0.233, +0.398] | 0.59x | 0.485, 0.579, 0.660, 0.711, 0.783, 0.827, 0.840 | 0.118 / 0.110 |
| (6, 16) | +0.245 [+0.151, +0.329] | 0.60x | 0.131, 0.153, 0.174, 0.185, 0.209, 0.219, 0.224 | 0.121 / 0.115 |

Scored against the frozen predictions:

- **H1 FALSE.** CE falls with every adjacent pair up to 16→24; the 24→32 pair's CI
  covers zero (−0.0001 [−0.0003, +0.0002]). Monotone to 24, flat from 24 on, and
  32→48→64 is slightly NEGATIVE (−0.0005 to 64, CI below zero).
- **H2 TRUE.** K3−K6 = +0.566, eleven times the 0.05 bar, 280x MORPH's 0.002.
- **H3 FALSE, both halves.** K16−K32 = +0.0034 < 0.01; K32−K64 = −0.0005 < 0.005.
- **H4 FALSE (flatness), TRUE (first token earns least).** At (1, 6) the bin mean is
  1.760 and the ±15 % band is [1.50, 2.02]; offsets 0 and 1 (1.25, 1.45) fall below it.
  The profile RISES with offset-in-span: the last bin earns 1.62x the first token. On
  every MORPH model the ratio is 0.74–0.89x and the rest of the profile is flat.
- **H5 FALSE.** Spearman +0.301 [+0.215, +0.389] at (1, 6): Huginn's earning
  concentrates on the rows it finds hardest. MORPH's a2 reads −0.18. (The top-decile
  earning share still sits just below the loss share, 0.108 vs 0.114, so the
  concentration is a rank effect over the whole distribution, not a tail effect.)
- **H6 TRUE.** 2.499 at 32 steps.

Three of six held (H2, H4's second half, H6). H1 was the 80 % prediction and it missed on
the last pair; H3, H4's first half and H5 missed on the shape.

## Verdict

The binding rule fires on H2: a recurrent-depth model trained to use depth earns
0.57 nats between iterations 3 and 6 and 0.21 more between 6 and 16 on the SAME web text
where every stable MORPH loop earns nothing past iteration 3. Web text is not depth-flat.
The arc's closing option (i), "data where depth pays", is downgraded as written; the
training regime is now the candidate cause of MORPH's flat loops.

Two readings, kept apart as the amendment asked:

1. **Threshold reading.** Huginn saturates between 16 and 24 iterations against a
   training draw centred on 32; MORPH saturates at 3 against a mean of 6. Both stop near
   HALF their training mean. That is one observation per model, not a law.
2. **Shape reading.** Huginn's earning is not the uniform refinement E0 measured on
   MORPH. It grows with offset-in-span (1.25 → 2.02 nats per token from the span's
   first token to offsets 16–31) and with row difficulty (Spearman +0.30). A loop that
   uses depth uses it MORE where the context is long and the row is hard. E0's flat
   profile is MORPH's, not web text's.

Not verified, named: the rows are the sorted OpenWebText TRAIN-shard prefix, not a
held-out split, and Huginn's 800B-token mix may contain them, so absolute CE (H6) and
possibly the row-difficulty correlation carry a contamination risk this eval cannot
bound; depths 1–3 sit outside Huginn's training draw, so K1−K6 mixes "never trained at
this depth" with earning, and the within-regime numbers are K6−K16 and K12−K16;
cross-tokenizer CE is not a common unit with MORPH (the comparison is of DIFFERENCES,
per row, under the same instruments); the boundary rule rebuilt on Huginn's tokenizer
was not audited span by span.

## Updated hypothesis

A weight-shared loop earns depth up to roughly half the depth it was TRAINED at, and
MORPH's mean-6 draw is why its loops stop at 3. The test is the next prereg,
`planned/2026-09-07-arc-e6-deep-recurrence-draw.md`: the plain loop at a Poisson draw
of mean 16 (max 24) with truncated backprop over the last 8 iterations, the ramp, 5000
steps, scored on forced-depth K3−K6 and K6−K12 on 480 rows. If it earns past 3, the
loop question reopens as a regime question (deep draw × long training) and the arc's
E5 moves to the deep draw. If it does not, the training-depth explanation fails on
MORPH and the remaining candidates are training length (800B against 0.3B tokens) and
width.
