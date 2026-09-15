# Planned: TUL-Code — the slot holds a ground-truth code of the span it precedes, and the core body samples it

Status: failure

Date: 2026-09-14 (frozen before any code exists; this file is committed before the contracts
are written and before any GPU step). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Spec:
[`docs/tul-code-spec.md`](../../../docs/tul-code-spec.md) (the contract; this file owns the
NUMBERS). Design note:
[`2026-09-14-tul-span-code.md`](../../../.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md).
Paper readings: `docs/references/tul-latent-emission/{ladir,ltm,diffusion-forcing}/`.

## Question

Every slot-loop arm since 2026-09-04 trained the slot state only through the next span's
token loss, and every one read a flat loop, a rank-6 state and a probe that scored the
state at entry as well as at exit. Wolfe, 2026-09-14: "if I just had ground truth latent
values I could do anything." TUL-Code gives every slot a ground-truth code: an encoder E
reads the span the slot precedes and writes its code into the cells; the strict coda speaks
the span reading the code plus the span's own tokens; the core body is trained, one pass
per slot, as a flow-matching velocity field from noise toward that code, and samples the
code in k passes at inference. **Three questions, in order: (1) is the code a code the
mouth reads, (2) does the sampler improve with k, (3) does a sampled code beat the
deterministic loop write on the same geometry, at a compute the d1 rung cannot match.**

## What is already known

- The strict geometry makes the loop's write the whole cross-span channel: 0.1865 nats at
  CE parity with the bypass arm (`failures/2026-09-12-arc-strict-geometry.md`, verdict 1).
  The reader problem that killed FM1 / FM1-CW / FM2 (worth 0.0000 on an additive detached
  prefix with tokens in reach) is not present on this geometry.
- There is 0.3994 nats [0.3838, 0.4162] of cross-span information at 5k, 0.3149 of it flat
  at offsets 8+ (`failures/2026-09-11-arc-span-budget.md`). That is the ceiling on what the
  code channel can be worth here.
- The slot family's state reads effective rank 5.7–7.3 of 1024, pairwise cos 0.72–0.77; the
  Thought Register 1.24 of 4 (`failures/2026-09-13-arc-thought-register.md`).
- The linear probe scores the slot state at ENTRY at AUC 0.60–0.64 (null 0.51), exit no
  higher (`failures/2026-09-12-arc-latent-z-gradient.md`, Step 0).
- The d1 rung (12 block passes per token) sits +0.0674 behind the Poisson-6 plain model at
  20k on paired rows; the plain model's own K1−K6 is 0.170 (`2026-09-13-arc-depth-ladder-ship.md`).
- The FM planner line's terminal reading (P1 on l2cap: trained ≈ untrained ≈ shuffled) and
  its binding rule; Wolfe waived the rule for this design on 2026-09-14 (note, "The binding
  rule").
- Rates: `slot-spandec-strict` on the queue runs at the panel shape (seq 1024, batch 12);
  its rate and resident memory are the comparison for the smoke's numbers.

## Method

Arms, all composing `tul_code.yaml` (spec §9) unless named otherwise, at the strict
ruler's SHAPE (seq 1024, batch 6, seed 1 — `tul_to_panel.yaml` in the lineage), 20,000
steps with `training.steps=20000 training.ademamix_t_beta3=20000 training.ckpt_every=5000`
as queue overrides, wandb `morph-tul`, one seed each in this panel; a second seed of
`tul-code` and of the ruler before any claim.

Method amendment 2026-09-14 (before launch; reason: the lineage's batch is 6, not the 12 I
wrote from `tul_short.yaml`, and the strict ruler exists only as a 5k run): the ruler gets
a 20k twin `slot-spandec-strict-20k` with the same overrides, so G3/G4 pair at 20k on the
same shape. The 5k readings pair against the existing `slot-spandec-strict` checkpoint.

Method amendment 2026-09-14, second (before launch; Wolfe: "I think we can skip the 20k
run"): the three TUL-Code arms run the lineage AS IT STANDS — 5,000 steps, batch 6,
`ademamix_t_beta3` 3500, checkpoints and sweeps at 2500 and 5000 — with NO overrides, so
each is a clean pair against the existing `slot-spandec-strict` checkpoint at the same
schedule. The 20k ruler twin is cut. Phases therefore switch at step 500 (phase 2) and
2,500 (phase 3) (spec defaults × 5k). P-4, P-5 and P-6 are 20k readings and are DEFERRED,
not edited: they are scored only if a 20k follow-on (a fresh draw of the winner plus the
ruler twin, never a resume) is run after P-1 to P-3 read. P-8 is read at 5k.

| arm | config | one factor |
| --- | --- | --- |
| `tul-code` | `tul_code.yaml` | the design, spec defaults (`code_noise 0.5`, phase2 0.10, phase3 0.50, rollout p 0.5 at 8 steps) |
| `tul-code-seeddetach` | `tul_code_seeddetach.yaml` | the FM gradient may not reach the seed path (Wolfe: an ablation) |
| `tul-code-nophase3` | `tul_code_nophase3.yaml` | `code_phase3_at: 1.0` (rollout never starts) — LaDiR's stage-2 ablation on our data |
| `slot-spandec-strict` | exists (5k) | the deterministic loop write on the same geometry: the MEAN control at 5k |
| `slot-spandec-strict-20k` | `tul_slot_spandec_strict.yaml` + the 20k overrides | the same control at 20k |
| `notul-d1` (`norm-match-20k-d1`) | exists, done | the compute bar |
| `norm-match-20k` | exists, done | the plain looped ceiling on CE |

Before any queue entry: contracts C1–C10 of the spec green on CPU
(`OMP_NUM_THREADS=2 nice -n 19 pytest tests/test_tul_code.py -q`, count reported); every
config composed through Hydra + `tul_setup` at the queued commit; a 12-step GPU smoke of
`tul_code_smoke.yaml` on the Spark with exit 0, `fm/rel` finite, `val/code_eff_rank`
printed, resident memory and tok/s recorded here under "Not verified before launch".

Instruments (spec §8), all on the 480 paired rows at 5k, 10k, 15k, 20k: `ce_tf`,
`ce_k{1,2,4,8,16}` (teacher-forced tape), `ce_k8_rolled`, `code_gap`, `code_eff_rank`,
`code_pairwise_cos`, `fm/rel` per band, the cells' shuffle cost, `ce_k8` by span-length
bucket; generation at k ∈ {1, 8} with rep4 / distinct-3 at 5k and 20k. Paired gaps by
`span_budget_profile.py` against the ruler and the two plain rungs; never the runner's
Final val_loss.

Phases: phase 2 starts at step 2,000, phase 3 at step 10,000 (spec defaults × 20k).
(Second amendment: at 5k they are steps 500 and 2,500.)

## Predictions (frozen)

Numbers are nats on the 480 paired rows unless stated. "The ruler" is `slot-spandec-strict`
at the same step.

- **P-1 (G1, sanity: the code is a code and it is read).** At 5k on `tul-code`: `ce_tf` at
  least 0.15 below the ruler; the cells' shuffle cost in the `ce_tf` regime at least 0.15;
  `code_eff_rank ≥ 16` of 1024. Prior 80 %. Failing P-1 means the mouth does not read the
  cells or E writes a constant; nothing below is readable and the panel stops after the
  first arm.
- **P-2 (G2: the sampler earns with k).** At 5k: `ce_k1 − ce_k8 ≥ 0.02`, `ce_k8 − ce_k16`
  within [−0.005, +0.02] (diminishing, not reversing), `fm/rel` below 0.9 in every t-band.
  Prior 60 %. My expectation is a curve that flattens between k = 4 and k = 8.
- **P-3 (the code is coarse, not the span).** `code_gap = ce_k16 − ce_tf` at 5k in
  [0.05, 0.35]. Below 0.05 says the code carries nothing the past cannot guess (the noise
  is too strong, or E collapsed onto the seed); above 0.35 says the code carries the span
  verbatim and the sampler cannot follow. Prior 55 %.
- **P-4 (G3: a sample beats a mean).** At 20k: `ce_k8` below the ruler, CI excluding 0.
  Prior 45 %. The honest alternative is that a mean suffices on web text at this horizon,
  which is LCM's result with a token path added, and it is a real possibility.
- **P-5 (the rolled tape holds).** At 20k: `ce_k8_rolled − ce_k8` within [0, 0.05] on
  `tul-code`, and larger by at least 0.02 on `tul-code-nophase3`. Prior 55 %. This is the
  LaDiR stage-2 claim on our data.
- **P-6 (G4: the ship bar).** At 20k: `ce_k8` within 0.02 of `notul-d1` (12 passes per
  token) at 6 + 8 = 14 block passes per token counted per position, i.e. the code model is
  at least as good as the d1 rung at near its compute. Prior 35 %. The plain looped model at
  42 stays 0.05–0.07 ahead; matching IT is not predicted.
- **P-7 (seed-path gradient).** `tul-code-seeddetach` vs `tul-code` at 20k: within
  ±0.01 on `ce_k8`. Prior 60 %. A larger gap in either direction decides the default.
- **P-8 (rate and memory).** `tul-code` runs at ≥ 1.3× the ruler's tok/s (one thinker pass
  per slot instead of a Poisson-6 loop on the slot positions) at resident memory ≤ the
  ruler's. Prior 70 %. The smoke measures it before launch; a miss here amends Method, not
  Predictions.
- **P-9 (no collapse over training).** `code_eff_rank` at 20k ≥ its 5k value − 4, and
  `code_pairwise_cos` ≤ 0.6 at every eval. Prior 60 %. The slot family's 0.72–0.77 is the
  number to beat.

## Binding

- P-1 fails ⇒ stop the panel, file under `failures/`, the next step is the reader
  (geometry or E), not the sampler.
- P-1 holds and P-2 fails on two seeds ⇒ the thinker is not integrating; file under
  `failures/`; the design note moves to `rejected/`.
- P-1, P-2 hold and P-4 fails on two seeds ⇒ multimodality does not pay on web text at
  this horizon; file under `failures/` with the K-curve as the finding; the next arm is the
  unrolled-trajectory phase 3 (the spec's follow-on), preregistered separately.
- P-1, P-2, P-4 hold ⇒ `successes/`; the note moves to `implemented/` once `tul_code.yaml`
  is on master with C1–C10 green; P-6 decides whether it is the ship candidate.
- No run beyond 20k without Wolfe's word. No PonderNet / ACT. k is chosen from the K-curve
  and stated; never tuned on the paired rows it is then reported on.

## Not verified before launch

Filled 2026-09-14 before queueing. Code at `44b6ac4`; the runner checks out the commit
that carries this section.

Verified:
- `pytest tests/test_tul_code.py tests/test_generation_sampling.py -q` → 34 passed, exit 0
  (17 contracts C1-C10 + eval modes + generation, 17 generation-sampling) at `aa7aaee`;
  the regression subset (208) passed at `361504f`, the build commit.
- The four configs (`tul_code`, `tul_code_seeddetach`, `tul_code_nophase3`,
  `tul_code_smoke`) compose through Hydra + `tul_setup` and build (test C10).
- DGX Spark GPU smoke `tul_code_smoke` with `training.gen_every=6`: exit 0, phases
  switched 2→1 at step 0, 1→2 at step 1, 2→3 at step 6; two generation dumps written and
  the generator crossed span boundaries (3 slots on the first prompt); `val/ce_tf 11.2093`,
  `val/code_gap 0.0104` at the earlier gen-off smoke; peak 1.22 GB at the tiny shape.
- On that step-12 checkpoint: `core_depth_sweep.py` (k = 0/1/2/4, exit 0, `ce_0` = the
  encoder ceiling 11.2267, sampled 11.2263), `worth_profile.py` (exit 0 after the
  wrong_seed skip, zero/shuffle/all_slots reported), `slot_state_probe.py` (exit 0, clean
  skip with a note).

Not verified:
- The panel shape (seq 1024, batch 6, 266M params) has never run on the 5090: resident
  memory and tok/s (P-8) come only from the runner's own 12-step smoke at launch. The
  tiny-shape smoke says nothing about them.
- The trainer's compiled path (`training.compile: true` in the lineage): the smoke ran
  `compile: false`. The two phase recompiles are a design expectation, not a measurement.
- The runner's slot sweep list is k ∈ {1, 2, 3, 6, 9, 12, 16}; the `ce_k4` and `ce_k8`
  readings in the Predictions need a hand sweep on the same 480 rows after each arm, and
  `val/loss` (k = `code_infer_steps` 8) on the trainer's val rows is the interim reading.
- No real-data reading of the encoder's code rank, the flow loss or the shuffle cost
  beyond 12 steps. P-1 is the first thing the 5k checkpoint answers.

## Results

Filed 2026-09-14 (all three arms at 5k, one seed each; ruler = `slot-spandec-strict` 5k;
paired on 501,106 shared tokens, `paired_vs_strict_ruler_5000.json`; artifacts in
`lab/experiments/results/2026-09-14-arc-tul-code/`).

| arm | rate tok/s | `ce_tf` (encoder code) | `ce_k1` | `ce_k16` | code gap | eff. rank | paired `ce_k1` − ruler | zero / shuffle cost |
|---|---|---|---|---|---|---|---|---|
| tul-code | 33.1k | 0.57 | 5.05 | 5.07 | 4.5 | 42 | +0.705 [+0.702, +0.709] | -0.0109 / -0.003 |
| tul-code-seeddetach | 28.8k | 0.53 | 5.02 | 5.02 | 4.5 | 40 | +0.675 [+0.671, +0.678] | 0.0057 / -0.0014 |
| tul-code-nophase3 | 29.6k | 0.40 | 11.78 | 12.36 | 11.7 | 63 | +7.435 [+7.418, +7.451] | -7.2415 / -0.1951 |

The ruler runs at 10.4k tok/s. Sampled-code CE with the ENCODER code on every past slot
(`ce_k1`, the sampled mode) is 0.68–0.71 nats behind the ruler on the shared tokens; with
the encoder code on the open slot too (`ce_tf`) it is 3.8 nats below it — that number is
the reconstruction ceiling, not an LM reading.

Flow ratio (nophase3, the only arm with the logging): 1.00 at the phase-2 switch, 0.33 at
5k (bands t<.25 / .25–.5 / .5–.75 / >.75 = 0.49 / 0.32 / 0.22 / 0.27; a content-blind
linear field on an isotropic code reaches 0.64 / 0.93 / 0.93 / 0.65). Still falling at 5k.

`code_subspace_probe` (Spark, 4900 slots, both checkpoints): the code has no tail —
cumulative variance 0.99 at rank 128 of 1024 per cell, 1.00 at 256 — and `ce_tf` from the
top 128 components alone equals the full value. The thinker's sample sits at 1.7–1.9 times
the code's own variance from the true code in EVERY subspace (a conditional-mean predictor
reads ≤ 1.0, an uncorrelated draw 2.0): the sample is close to an unconditional draw.

`code_span_samples` (10 cuts, `code_span_samples_*_5000.txt`): with the true code in the
cells the coda writes the span back near-verbatim (5/10 exact); the thinker's samples are
newswire-register sentences of the right shape with no topic thread (0/10), where the
ruler's slot loop keeps a thread in about 5/10 on the same cuts and seeds.

Trainer defects found during the panel, fixed at 19535fe / 2c61b89 / 182c6a7: the flow
statistics were never forwarded to wandb (train whitelist), the flow term sat inside
train/loss on the train side, the `tul/` loop lacked the keys. `tul-code` and
`tul-code-seeddetach` ran before the fix and have no flow-ratio trace.

## Verdict

- P-1: PARTIAL. `ce_tf` 3.8 below the ruler and rank 42 pass; the shuffle cost is ≈ 0,
  because shuffling a SAMPLE against another sample costs nothing — the prediction assumed
  the sampled code would carry span identity. Failed as written.
- P-2: FAILED. `ce_k1 − ce_k8` ≈ −0.01 (wrong sign); the K-curve is flat. The flow ratio
  half (< 0.9) holds on nophase3 (0.33).
- P-3: FAILED. Gap 4.5, predicted [0.05, 0.35].
- P-4, P-5, P-6: deferred (no 20k run) and moot at 5k: every sampled reading is behind
  the ruler.
- P-7: HELD. seeddetach within 0.04 of tul-code (sampled), sweeps identical.
- P-8: HELD. 2.8–3.2× the ruler's rate.
- P-9: HELD. No collapse (rank 40–63, pairwise cos ≤ 0.10).

Verdict: the 5k PREDICTIONS failed (P-2, P-3, and P-1's shuffle half); the DESIGN is not
judged by this panel. The design does what the spec says — E defines a code by
reconstruction and the coda reads it back — and one sample of that code costs the coda
0.7 nats against the deterministic slot loop at 5k. Whether that is the target (a verbatim
copy of an uncertain sentence) or the horizon (a flow model learns the marginal before the
conditioning, and this thinker's loss was still falling at 5k, with band-0 showing
conditioning under way) the 5k readings cannot separate. Wolfe, 2026-09-14: "much better
than it looks, it is just undertrained" — the token path continues cleanly through the
cut ("conf" → "ident", "sh" → "ots") on every arm.

## Updated hypothesis

The code's content is entirely in the L2-visible head, so the flow loss is not the
problem. About a quarter of the code's variance is predictable from context (band-0 ratio
0.49 vs the 0.64 blind floor); the sample shows less than that and drops the topic the
ruler keeps. Two things follow. (1) The coda must be trained on the EVAL distribution:
`code_rollout_p: 1.0` in phase 3 freezes E (its only gradient is the teacher-forced CE)
and lets the coda learn how much of a guess to trust and to lean on the past codes — the
LaDiR stage-2 shape; expected to return the arm to the ruler, not below it. (2) Getting
below the ruler needs a code defined by what the past determines about the span, not a
copy of the span — a different object from the LaDiR latent, to be specified before any
further arm. Generation quality, not one-sample CE, is where a sampled code can win, and
it was not scored beyond ten cuts. (3) One-sample CE is the wrong likelihood for a
latent-variable LM: the next instrument is the K-sample marginal, log of the mean over K
sampled codes of the span's likelihood under the coda — a lower bound on the true
log-likelihood — read against the ruler's CE. (4) The 20k pair from the original Method
(tul-code and the ruler twin) plus a `code_rollout_p: 1.0` twin is the next panel, once
(3) exists.

### Addendum 2026-09-15: the K-sample marginal at 5k

Instrument (3) ran on the tul-code 5k checkpoint on the Spark
(`lab/divergence/code_marginal_sweep.py` at f64886b; 96 rows, K = 8 draws per span,
`code_marginal_tul-code_5000.{json,txt}` in the results directory):

| k (Euler steps) | ce_marginal (8 draws) | ce_single_mean |
| --- | --- | --- |
| 0 (encoder code) | 0.536 | |
| 1 | 4.971 | 4.973 |
| 2 | 4.972 | 4.974 |
| 4 | 4.966 | 4.975 |
| 8 | 4.966 | 4.984 |
| 16 | 4.967 | 4.988 |

marginal(k=16) − marginal(k=1) = −0.004 [−0.006, −0.003], paired bootstrap over 32 batches.

Reading. The marginal is at most log 8 = 2.08 nats better than the BEST of the eight
draws, and better than the MEAN single draw by an amount set only by the spread of the
draws' likelihoods (a Jensen gap, unbounded). Here it beats the mean single draw by 0.002
at k = 1 and 0.022 at k = 16: the eight draws are near-interchangeable to the coda, which
is the subspace result (sample ≈ unconditional draw) seen through the likelihood.
(Corrected 2026-09-15 02:20: an earlier wording called log 8 the cap on that gap; it is
the cap against the best draw, not the mean. The 20k arm's phase-2 trace shows gaps of
2.5–3 nats, which a log-8 cap would forbid.) More Euler steps spread the draws (the single-draw mean
worsens by 0.015 from k = 1 to 16 while the best of them improve), so the sampler does
make different codes at higher k; the coda just does not find a better one among them.
The inference-depth signal under the right metric is −0.004 nats at 5k: present, and
negligible. Wolfe, 2026-09-15: the full set of phases (the 20k arms, phase 3 from 10000
to 20000) is needed before this probe reads anything about the design; the 20k arms log
`val/ce_marginal` at every val, and the k-sweep reruns on the Spark at each checkpoint.
