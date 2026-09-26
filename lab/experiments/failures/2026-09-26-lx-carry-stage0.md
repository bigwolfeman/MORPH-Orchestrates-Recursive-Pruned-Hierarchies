# Planned: does carrying the rollout posterior across spans help the deployed LX read (LX-Carry Stage 0)

Status: failure

Date: 2026-09-26 14:37 (frozen before the re-scoring ran; no carried number had been computed. The unit
tests ran on the tiny CPU model only.)
Design: `/home/wolfe/morph-scratch/tulv2/opus.md` D2 "LX-Carry", Stage 0. The same idea is
gpt.md B1 and fable.md V2-B.
Parent readings: [`../successes/2026-09-25-lxtul-fp01-10k.md`](../successes/2026-09-25-lxtul-fp01-10k.md)
(the stored arrays this re-scores).

## Question

LXTUL-E's deployed read restarts the posterior over the K = 4 code rollouts at uniform in
every span. Rollout k is one coherent world across the row: every slot's loop in rollout k
attends to earlier slots built with u_k. If the rollouts carry row-level identity, a read
that starts span s at the posterior the earlier spans earned should predict better,
most of all at the span's first positions, where the read today has no evidence at all.
Does it, on fp01 at 5k and 10k, with no training?

## Method

- Instrument: [`../../divergence/lx_carry_stage0.py`](../../divergence/lx_carry_stage0.py). The
  math is `morph/model/rollout_mixture.py` (`fixed_share_log_prior`,
  `carried_position_nll`); tests `tests/test_lx_probes.py`.
- Data: the stored per-rollout per-token coda log-probs of the fp01 Stage 2 scorer run,
  `/home/wolfe/morph-scratch/lxtul-10k/score.tokens.npz`, labels `fp01_5k` and
  `fp01_10k` (keys `*_coda_code_{d}` [4, 501,106], `*_coda_idx`), 480 packed
  validation rows (the `core_depth_sweep.py` rows), forced depths 1 and 6. CPU only, no
  forward pass. The rows are rebuilt on CPU by the same packer to get the runs, the
  evidence positions, the offsets and the EOS positions.
- The read: `alpha_s(k) ∝ pi_s(k) exp S_k(s)`, `pi_{s+1} = (1 - eps) alpha_s + eps / 4`,
  `S_k(s)` summed over every scored position of the run (evidence and the span's last
  token). Inside the run, the per-token Bayes read starts at `pi_s`. A run holding an EOS
  input token ends a document; the next run restarts at uniform. Each row starts at
  uniform. `eps` in {1, 0.5, 0.2, 0.05, 0.01}; eps = 1 is today's read.
- Split: eps* is the eps with the lowest mean coda CE on rows 0-239 at depth 6; every
  reported delta is on rows 240-479. Paired block bootstrap over stream blocks of 1,024,
  2,000 resamples, seed 0.
- Readings, per checkpoint: coda CE(eps*) - CE(1) overall and in the offset-0 bin
  (`_earning.BINS`, offset 0 = a span's first input token), every bin, and every eps on
  both halves; K1-K6 under the carried read with the SAME eps at both depths, and its
  paired change from today's K1-K6; the carried prior's normalised entropy at run starts;
  the plug-in mutual information between consecutive in-document runs' best rollout
  (argmax_k S_k) with a 200-shuffle null.
- Self-checks (each raises): the rebuilt scored positions equal the stored `coda_idx`;
  today's read recomputed on CPU matches the stored `coda_mix` within 1e-4 per token;
  eps = 1 is bit-identical to `_enum_position_nll`; each row's carried NLL sums to the
  switching model's likelihood from an independent per-run loop.

## Predictions

From the Opus analyst file (opus.md D2, Stage 0), for fp01, stated for 5k and 10k alike:

- **O-1 (overall).** Coda CE(eps*) - CE(1) = **-0.002 [-0.006, 0]**. Holds if the measured
  point lies in [-0.006, 0].
- **O-2 (offset 0).** The offset-0 bin delta = **-0.010 [-0.03, 0]**. Holds if the point
  lies in [-0.03, 0].

Opus's caveat, kept: this is a LOWER bound on what a carry can do, because the model was
trained with the per-span restart.

Secondary, from the Fable analyst file (fable.md V2-B; its alpha is 1 - eps, so its
alpha = 0.5 is eps = 0.5 here), on the 10k checkpoint:

- **F-1.** Run-best MI >= 0.05 nats.
- **F-2.** Offset-0 CE improves by >= 0.02 at eps = 0.5.

## Reading rules

- From opus.md D2: the Stage 0 eps is the one the 5k arm would train with. The arm's
  falsifier is a within-model paired read: if the arm's carried read beats its own eps = 1
  read by < 0.005 nats, the rollouts have no row-level identity worth carrying. Stage 0
  does not run the arm; a Stage 0 delta far below 0.005 in magnitude is a warning for it,
  not its verdict.
- From fable.md V2-B: MI < 0.02 nats says the codes are per-span types with no row
  meaning, and the carry is not a lever.
- If eps* = 1 (no carry helps on the fit rows), O-1 and O-2 are read at eps* and the
  best eps < 1 is reported beside them.
- The carry is a cross-span route of at most log 4 = 1.39 nats per span outside the cells
  (opus.md D2 named cost). A K1-K6 reading under the carry is reported with the eps held
  identical across depths, never fitted per depth.

## Results (filed 2026-09-26 14:59)

CPU rescoring of the stored Stage 2 per-rollout log-probs (`lab/divergence/lx_carry_stage0.py`,
commit 2194b08). Self-checks: the rebuilt rows match the stored coda indices exactly, and
the eps = 1 read reproduces the stored mixture (max |dev| 1.9e-6). 381 EOS resets in the 480
rows. eps fitted on rows 0-239 at depth 6; every number below is on rows 240-479 (250,376
tokens, 12,265 at offset 0). Artifacts: [`../results/2026-09-26-lx-carry-stage0/`](../results/2026-09-26-lx-carry-stage0/).

| eps | 5k overall | 5k offset 0 | 10k overall | 10k offset 0 |
|---|---|---|---|---|
| 1 (today) | 0 | 0 | 0 | 0 |
| 0.5 | +0.0006 [+0.0003, +0.0010] | +0.0027 [+0.0003, +0.0051] | +0.0010 [+0.0007, +0.0014] | +0.0013 [−0.0008, +0.0034] |
| 0.2 | +0.0037 | +0.0092 | +0.0046 | +0.0077 |
| 0.05 | +0.0083 | +0.0160 | +0.0099 | +0.0152 |
| 0.01 | +0.0122 | +0.0192 | +0.0145 | +0.0204 |

- Every carry makes coda CE WORSE, most of all at offset 0, the bin it was meant to help.
  eps* = 1 at both checkpoints, so K1−K6 under the carry equals today's (+0.0177 at 10k).
- Run-best mutual information between consecutive spans: 0.0018 nats (5k) and 0.0009 (10k),
  against a shuffle null q95 of 0.0003 / 0.0004. Above the null, 20-50x below Fable's
  0.02 cut.

## Clause by clause

- **O-1 and O-2 hold only trivially.** By the reading rule they are read at eps* = 1, where
  the delta is exactly 0. The best eps below 1 (0.5) lies OUTSIDE both predicted ranges, on
  the wrong side. Read as a miss of the substance of the prediction.
- **F-1 FAILS** (0.0009 vs >= 0.05). **F-2 FAILS** (offset 0 gets worse at eps 0.5).

## Verdict

**Failure.** The four LX rollouts carry almost no row-level identity: which rollout explained
one span says next to nothing about which explains the next. The per-span restart is the
right read for this model. Caveat, from the prereg: the model was trained with the restart,
so this is a lower bound on a carry trained end to end; with MI at 0.001 nats the ceiling of
such an arm is small.

## Updated hypothesis

LX's rollouts behave as per-span alternatives, not as coherent row-level "worlds". That fits
the three analysts' reading that the fixed row-level codes are not hypotheses about content.
Designs that make the hypotheses per span from context (LX-Concept, a context-oriented code)
are the direction; a learned prior over the fixed codes (the gate) inherits this weak
identity and drops in priority. Step 0 of the amplitude probe ran with this one: the codes did
not move off the injected slice (share 0.314 at init, 0.324 at 5k, 0.331 at 10k, all inside
the null band); it is filed with the amplitude-matched K-curve.
