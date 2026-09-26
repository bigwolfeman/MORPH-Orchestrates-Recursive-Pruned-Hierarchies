# Planned: can fp01's slot state predict the next span's CONCEPT? (the LX-Concept gate)

Status: planned

Date: 2026-09-26 16:37 (frozen before any GPU step of this probe; the unit tests ran on tiny CPU
models, and step 1(a), the alignment audit, ran on CPU before this file was written; its
result is quoted under Method because it is a precondition, not a graded prediction).
Parent readings: [`../failures/2026-09-26-lx-selection-ceiling.md`](../failures/2026-09-26-lx-selection-ceiling.md)
(span-level selection over fixed codes has ~0.0006 nats to buy) and
[`../failures/2026-09-26-lx-carry-stage0.md`](../failures/2026-09-26-lx-carry-stage0.md)
(the rollouts carry no row identity). Together: LX's four rollouts act as an ensemble, not
as span-level hypotheses.

## Question

LX-Concept would give each LX rollout a discrete per-span CONCEPT hypothesis, predicted from
context, CoCoMix style (arXiv 2502.08524: a discrete concept CE beats regression; labels come
from a teacher and are chosen by attribution). The teacher cannot be the plain LM (it is
undertrained); the candidate is the LCTUL VAE-stage span autoencoder
(`tul_code_vae`, `tul-code-vae/step_10000.pt`): E pools a span's code from that span's own
prelude states, and its frozen coda decodes the span from the code
(`docs/tul-code-spec.md` §3.3, §17; `val/ce_tf` 1.34 at step 9750). Before any arm is built:
does fp01's slot state at slot s carry enough about span s+1 to predict that span's concept
label better than the context-free prior? If it does not, rollouts conditioned on a
predicted concept have nothing to condition on at this scale.

## Hypothesis

The next span's coarse identity (topic, register, form: what a k-means cluster or a handful
of SAE latents of a 2,048-d autoencoder code captures) is mostly set by the text before it,
so a linear or one-layer head on fp01's exit state predicts it well above the marginal. The
entry state (the prelude at the slot, which under the strict geometry has seen span s only)
predicts most of it too, because the current span is the best single predictor of the next;
the loop's exit adds the earlier spans and should read a little better than the entry.

## Predictions

These are MY predictions (the builder agent's, Claude), written before any GPU number of
this probe exists. Each is graded on the point estimate unless it names a CI.

- **C-1 (oracle).** The teacher's clean-code oracle coda CE on the 480 validation rows is in
  [1.2, 1.7] nats/token (the training log read 1.34 at 9750 on other rows); the same code at
  its training statistic (`rmsnorm(z + 3.0 eps)`) reads in [2.0, 3.6]. fp01's coda CE at
  depth 6 is 4.07 at 10k (Stage 2), so teacher_clean minus fp01 is in [-2.9, -2.3].
- **C-2 (dictionaries).** k-means labels use most of their classes: label entropy >= 0.85
  log M for M = 64 and 256. The TopK SAE (4,096 latents, k = 32) reads FVU <= 0.20 on the
  val region (the VAE code has rank ~57, spec §17), with dead fraction in [0.1, 0.7].
- **C-3 (the gate, fp01 10k, exit, best head, val rows).** Gain over the marginal:
  type64 +0.25 (range [+0.12, +0.45]); type256 +0.30 ([+0.15, +0.55]); sae_top4 +0.25
  ([+0.10, +0.50]). The overall verdict is ALIVE. My probability that it reads ALIVE: 0.75.
- **C-4 (entry vs exit).** The entry head's gain is 60 to 90 % of the exit head's on every
  graded target; exit minus entry CE is in [-0.10, -0.02] on type256 (exit better, CI
  excluding 0).
- **C-5 (5k vs 10k).** The 10k exit gain exceeds the 5k exit gain on every graded target,
  by 0.01 to 0.06.
- **C-6 (attribution).** At least 1,000 distinct SAE latents appear in the val region's
  top-4 attribution sets, and fewer than half of the top-4 are among the span's 32 active
  latents (the attribution is taken over all 4,096 pre-activations, as in the paper).

## Reading rule

- **Gate.** On fp01 10k's exit (best of `exit` / `exit_raw` and of the linear / MLP
  heads), primary set = the 480 val rows: if the gain over the marginal is below **0.1 nats**
  on type64, type256 AND sae_top4 (point estimates), **LX-Concept is dead at this scale** and
  no LX-Concept arm is built. At least one target at or above 0.1 nats: ALIVE, and that
  target family is the one the arm uses.
- The secondary set (the train region's held-out test rows) must give the same verdict. If
  the two sets disagree, the gate is inconclusive: file under failures and plan the next
  measurement (more rows) rather than build.
- A negative gain on a feature is the head's overfitting floor, not information; the
  synthetic smoke of the script read about -0.06 nats on pure-noise features.
- Nothing passes or fails on a cosine, an FVU or an entropy; C-1, C-2 and C-6 are
  instrument readings graded as predictions only.

## Method

- Code (uncommitted at the time of writing; the chain checks out the SHA it is given):
  [`../../divergence/_concept.py`](../../divergence/_concept.py) (terms, joins, teacher and
  student reads, dictionaries, attribution, head math), `concept_audit.py`,
  `concept_dump.py`, `concept_dict.py`, `concept_precheck.py`; tests
  `tests/test_lx_concept.py`. GPU chain `/home/wolfe/morph-scratch/concept/concept_chain.sh`,
  after the LX probe chain ends, one job on the GPU, a smoke of every step first.
- Checkpoints: teacher `tul-code-vae/step_10000.pt` (config `tul_code_vae`); student
  `lxtul-e4probe-fp01` steps 5000 and 10000 (config `tul_slot_spandec_strict_e4probe_fp01`).
- **Step 1(a), alignment (ran on CPU at write time, not graded).** Composed configs agree on
  tokenizer (`bigcode/starcoder2-7b`), dataset, and the boundary rule (1,101 boundary ids,
  lookup-table sha1 92209b12..., min_span 4, span_cap 32, eos 0) and layout (seq 1024,
  prefix_k 2, max_slots 64, L_total 1152); the two runs' wandb configs agree on the same
  keys. Packing the 480 validation rows with each runtime: 0 row mismatches in any field,
  24,339 coded spans each, 24,339 joined on the span key, 0 span-token mismatches.
- **Step 1(b), oracle CE.** The teacher's coda reads E's code of every span (the `encoder`
  eval mode, E on the teacher's OWN prelude, never a live model's: memory "frozen encoder on
  a live front is not a fixed target"), per scored position, clean and at the training
  statistic (seeded noise). fp01's coda CE at forced depth 6 via
  `lxtul_e_stage1_score.score_arm` on the same rows; token-paired by stream index, paired
  block bootstrap over 1,024-token stream blocks, 2,000 resamples. Self-check: offline CE
  equals the forward's `ce_tokens` within 2e-3.
- **Step 2, dump.** Regions: val = the 480 rows above; train = 2,000 rows of the OWT stream
  after 200,000 documents (past what either model trained on; ~100k spans). For every coded
  slot s (slots s and s+1 valid): E's clean code of span s+1 (2 x 1,024, fp16), the
  teacher's span NLL, and fp01's entry, exit and exit_raw states at slot s (forced depth 6,
  rollout mean). Teacher and student rows are packed separately and joined on the span key
  (first and last stream index of span s+1); any unjoined span raises. Note, not a
  confound we can remove here: the 480 "validation" rows are the start of the OWT train
  shards, which every run reads at steps 0-80.
- **Step 3, dictionaries (fitted on the train region only).** k-means, M in {64, 256},
  k-means++ init, Lloyd <= 60 iterations, seed 0. TopK SAE, 4,096 latents, k = 32, AuxK
  (k_aux 256, coefficient 1/32, dead after 2M samples), Adam 3e-4, batch 1,024, 20,000 steps,
  unit decoder rows, on 95 % of the train codes (5 % held out). **CoCoMix attribution:** the
  teacher's coda reads the SAE reconstruction D(c) at slot s (every other slot E's clean
  code); the exact gradient of span s+1's NLL alone w.r.t. those cells (one row copy per
  span, so later spans that also read the cell add nothing); a = c_pre * (W_dec g) over all
  4,096 latents; label = the 4 highest a (the paper's rule, K_attr = 4). The 4 lowest are
  written as a second label set, reported and not graded. All val spans; the first 60,000
  train spans (compute). Finite-difference check on 4 spans.
- **Step 4, the gate.** Heads: linear softmax (bias initialised at the prior) and one
  hidden layer of 512 GELU units; AdamW lr 1e-3, wd 1e-2, z-scored features, early stop on
  dev rows (patience 3, <= 40 epochs). Train region rows split 80/10/10 into fit / dev / test
  by row, seed 0. Loss for a set target: the mean over its 4 indices of the softmax CE
  (CoCoMix's L_concept). Prior: additive-0.5 smoothed index frequencies on the fit rows.
  Readings: held-out CE, prior CE, gain with a 95 % bootstrap over rows, top-1 / top-5 hits,
  and exit - entry paired.
- Estimated GPU time (from Stage 2's 25-30 s per depth-pass over 480 rows): audit ~3 min,
  dumps ~12 min, dictionaries ~2 min, attribution ~20 min (val) + ~50 min (60k train spans),
  gate ~15 min. Unmeasured until the smokes run.
