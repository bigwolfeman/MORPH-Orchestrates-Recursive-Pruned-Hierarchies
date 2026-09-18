# Experiment: the code target — the TUL slot loop regressed onto the frozen VAE code

Status: planned

Date: 2026-09-17 (frozen before any GPU step beyond a 12-step resume smoke on the Spark).
Note: `.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md`.
Spec: `docs/tul-code-spec.md` §17. Tests: `tests/test_tul_code_target.py` (12, CPU).

## Question

When the slot loop is trained to MATCH the frozen VAE code of the next span (LaDiR with
TUL as the latent generator, no sampler), does the loop's cell carry the predictable part
of the span into the frozen coda's generation, and do the loop's passes move toward the
code?

## Hypothesis

H1: the loop can regress. The exit cosine to the code rises well above the entry state's,
the predicted cell helps the frozen coda write the span (OWN > ZERO, OWN > SHUF), and the
passes climb toward the code (per-pass cosine rises with t; forced depth lowers CE).
H0: the conditional mean is a confident vector at the mean direction; the frozen coda reads
it as a wrong code and OWN ≤ ZERO; the passes meet the target in one step (per-pass cosine
flat after l1), as every per-pass target before it (`per-pass-targets-met-in-one-step`).

## Predictions (frozen)

Arms `tul-code-target` (A, `code_target_detach: true`) and `tul-code-target-ce` (B,
`false`), both `tul_code_target*.yaml`: resume `tul-code-vae/step_10000.pt`, 20k steps,
batch 6, seq 1024, one seed each, E + prelude + embeddings + coda frozen. Controls on the
same 120 cuts: the strict ruler (`slot-spandec-strict-20k@20000`, OWN cosine 0.187), the
plain model (0.230), the VAE oracle (0.64 on the LaDiR chain's coda; this arm's ORACLE
condition re-reads it on the same coda).

- P-T1 (the loop regresses): `val/code_target_cos` ≥ 0.20 at 20k on arm A. 55 %.
- P-T2 (the passes move): at 20k on arm A, `tul/code_target_cos_l{T}` − `tul/code_target_cos_l0`
  ≥ 0.05 averaged over the last 500 logged steps, T the deepest logged pass. 45 %.
- P-T3 (own beats none): semantic probe at 20k, arm A, OWN − ZERO on cos_true > 0 with the
  95 % interval clear of 0. 40 %.
- P-T4 (own beats foreign): OWN − SHUF on cos_true > 0, interval clear of 0. 45 %.
- P-T5 (depth pays on the frozen coda): forced-depth sweep at 20k, arm A, `ce_tokens` at
  depth 6 minus depth 1 ≤ −0.02, paired interval clear of 0. 30 %.
- P-T6 (the bar): OWN within 0.03 of the strict ruler's OWN on cos_true, paired over the
  120 cuts (`code_semantic_pair.py`). 35 %.
- P-T7 (arm B reads the same): arm B's OWN − ZERO within 0.02 of arm A's (the CE route
  adds nothing the regression did not). 50 %.
- P-T8 (rate): both arms ≥ 0.9× the strict ruler's tok/s at batch 6 (the frozen token
  path stores no activations). 70 %.

## Binding

- P-T1 and P-T3 hold → the loop carries the predictable part into generation: next is the
  same arm with the coda UNFROZEN after 10k (does a reader that adapts to the predicted
  cell widen the gap) and the seed lane.
- P-T1 holds, P-T3 fails → the loop matches the code's mean direction and the frozen coda
  cannot use a mean: the refiner probe (draft → encode → regenerate) becomes worth its
  cost, and the note goes to rejected with the reading.
- P-T1 fails → the loop cannot regress onto this code from the past at 20k: the target is
  not reachable and the note goes to rejected.
- P-T2 fails with P-T1 holding → the regression is met in one pass like every per-pass
  target before it; the arm says nothing new about depth.

## Method

Queue lines at the head of `recon_arms.txt` (kind `slot`, so the runner sweeps forced
depths 1..16 and runs `worth_profile` at 20k), EXTRA
`training.resume=/home/wolfe/morph-to/checkpoints/morph/tul-code-vae/step_10000.pt`
plus the panel extras. The runner's 12-step smoke does NOT carry EXTRA, so the resume path
is smoked once on the Spark before queueing (12 steps, the same overrides). Spark watcher
`spark_code_probes_target.sh`: the semantic probe (`--kind target`) at 10k and 20k for
both arms; results under `lab/experiments/results/2026-09-17-lctul-target/`. Pairing
against the ruler and plain controls already saved under
`results/2026-09-16-lctul-dplan/controls/` with `code_semantic_pair.py`.

Amended 2026-09-17 10:22 (reason: the smoke ran and the lines were inserted): Spark
12-step resume smoke of `tul_code_target` at 9e831d5 exit 0 — 71.6M trainable / 198.3M
frozen, 9 code-thinker tensors dropped loudly, only `tul_code_proj.*` fresh,
`val/ce_tf` 1.61 (2 batches), `val/code_eff_rank` 57.5, `val/code_target_cos` 0.001 at
step 12. Queue lines inserted at the head of `recon_arms.txt` at 10:21 (both arms, kind
slot, commit 9e831d5, EXTRA with the resume path); watcher `spark_code_probes_target.sh`
started 10:21. Predictions unchanged.

Amended 2026-09-17 14:35 (reason: the first draw of arm A aborted at step 10020, exit 4):
the trainer's divergence guard (`MORPH_DIV_PPL` 1000, two strikes, live past step 2000)
fired at the resume step on a train CE of 9.4 nats (ppl 12,228 at 10000, 19,050 at 10020)
while the oracle read 1.46 nats — the frozen strict coda reading the identity-initialised
projection's cell, not a divergence; the LaDiR stages passed the same guard because their
coda read the noised TRUTH code at train (3.5 nats). Three changes to the OPENING, none to
the model or the predictions: (1) `training.init_from` replaces `training.resume`, so the
step axis restarts at 0 and the loop's core — at init in the VAE checkpoint — gets the
measured 1000-step LR ramp and a fresh optimizer (the first draw ran flat 1e-4 from its
first step and the CE rose 7.4 → 9.9 nats in 20 steps); (2) `training.data_skip_batches:
10000` moves the stream past the VAE stage's batches; (3) the guard's ceiling becomes a
config key (`training.div_ppl_ceiling`, 1e5 on these arms) because the coda's CE on the
predicted cell is the arm's instrument (1.3 nats = the true code; the "7.4 = nothing, 4.4 =
ruler" figures written here were a transient from the aborted draw and are withdrawn - see
the worth-profile correction below) and
the default ceiling sits inside that range. Arm B (`resume`, no ramp) was killed and both
arms re-queued at the head on the new commit after a second Spark smoke with `init_from`.
Second smoke (14:33, commit d8e93d2, which also declares `init_from` / `data_skip_batches` /
`div_ppl_ceiling` / `div_strikes` in base.yaml): exit 0, 467/469 tensors loaded, 0
unexpected, stream skip applied, guard ceiling 100000. Queue lines re-inserted at the head at
14:35 on d8e93d2; arm B's confounded draw killed at 14:35 (exit 143, last step 12010); both
aborted checkpoint dirs moved aside (`*.aborted-1417`). Watcher restarted 14:35.

Amended 2026-09-17 15:30 (reason: guard risk observed on the running arm A): at step 5360
the guard printed one strike (train CE 12.04 nats, ppl 169,276 > 100,000); the next step
fell back and the strike count reset. In arm A this CE is the frozen coda reading a
DETACHED cell, so it trains nothing; the regression loss it sits beside fell 2.0 → 1.72
(exit cosine 0.004 → 0.14). The guard counts CONSECUTIVE per-step exceedances, checked
every step; the logged (every 20 steps) CE 2000-5480 has mean 9.3 nats, max 12.04, and one
strike in 3480 steps. Arm A keeps running under its 1e5 ceiling (a running process cannot
be changed); if it aborts it is resumed from the DIVERGED checkpoint with the raised ceiling
and that is recorded here. The queued detach arm (progressive) gets the raised ceiling 1e8
through `tul_code_target.yaml`. Predictions unchanged.

Amended 2026-09-17 19:55 (reason: the forced-depth sweep was never scored on this family,
and the reading it gives needs its own instrument). Two corrections of method, no change to
the predictions:

1. THE SWEEP NEVER RAN. `core_depth_sweep.py` treated any model carrying `tul_code_enc` as
   a SAMPLER model and passed `code_steps`, which a code-target model refuses; every runner
   sweep on `tul-code-target@{5000,10000,15000,20000}` exited 1 with that NotImplementedError
   (queue.log 14:35-17:34). Fixed at 2236ac4 (a code-target model is the slot-loop path: it
   has `tul_code_proj`), and the 20k sweep was re-run on the Spark.
2. THE OPEN SLOT HAD NO CELL. `_tul_code_target_write` masked the projection by
   `code_target_valid`, which is False for a row's LAST valid slot; at generation the open
   span's slot IS that slot, so the coda wrote every generated span from a ZERO cell. The
   10k semantic probe therefore read OWN = SHUF = ZERO byte-identical (cos_true 0.1291,
   paired +0.0000 [+0.0000, +0.0000]) while ORACLE, injected on the same route, read 0.5590.
   Fixed at 14f004d (eval masks by `slot_valid`, train keeps `ok`). Every OWN/SHUF/ZERO
   reading on a code-target checkpoint before 14f004d is void; the probes are re-run.

## The depth reading, and why it is not depth earning

The re-run sweep at 20k (480 rows, paired): frozen-coda CE 9.1420 at depth 1, 9.1099 at 6,
9.0461 at 16; K1-K6 +0.0320 [+0.0235, +0.0412], K1-K16 +0.0959 [+0.0841, +0.1080]. P-T5's
letter holds. The mechanism does not survive its own instrument
(`code_target_mean_probe.py --depths`, the same 4900 slots at each forced depth):

| depth | 1 | 3 | 6 | 9 | 16 |
| --- | --- | --- | --- | --- | --- |
| cos(cell, own code) cell 0 | 0.1501 | 0.1539 | 0.1544 | 0.1542 | 0.1524 |
| cos(cell, shuffled code) | 0.0194 | 0.0194 | 0.0196 | 0.0197 | 0.0199 |
| cos(cell, corpus mean) | 0.3081 | 0.3121 | 0.3164 | 0.3180 | 0.3225 |
| effective rank | 17.5 | 17.0 | 17.0 | 17.1 | 17.5 |

and the passes DO move the cell, in a direction the code knows nothing about: against the
depth-1 cell, depth 6 sits at cosine 0.980 with |delta|/|p| 0.188 and depth 16 at 0.954
with 0.29, while that delta's cosine to the slot's OWN code is +0.017 / +0.014 (depth 6)
and +0.006 / +0.004 (depth 16) — at or below the 0.031 a random direction in 1024 dimensions
scores — and its cosine to the corpus mean is +0.044 / -0.084 and +0.051 / -0.037, opposite
in sign between the two cells. So fifteen extra passes move the cell by a third of its norm
and change its relation to the target by four thousandths. The CE gain is a reader at 9.1
nats (a ZERO cell reads 7.4) finding a perturbed harmful cell slightly less harmful, not the
loop finding a better code. Scored honestly: P-T5 holds and means nothing on its own, and
the arm's depth question is answered by P-T2, which fails (l6 - l1 = +0.003 over the last
500 steps).

## The cell is worth −4.6 nats to the frozen coda (correction, 2026-09-17 20:55)

The runner's own worth profile ran on arm A at 20k and was not read until now
(`results/2026-09-17-lctul-target/worth_tul-code-target_20000.json`, 192 rows, paired CE
deltas of `ablated − intact`, stratified by offset):

| ablation | arm A @20k | strict ruler @5k |
| --- | --- | --- |
| zero the cells | **−4.608** | +0.186 |
| shuffle the cells across slots | +0.050 | +0.174 |

Read plainly: on the ruler, taking the cell away costs 0.186 nats and giving a FOREIGN cell
recovers only 0.012 of that, so the write is worth 0.19 nats and 94 % of it is
slot-specific. On arm A, taking the cell away GAINS 4.6 nats, and the slot's own cell beats
a foreign one by 0.050. The regressed cell therefore carries about a quarter of the ruler's
slot-specific signal and arrives wrapped in 4.6 nats of poison for this reader. Every CE
number in this arm — the val loss, the forced-depth curve, P-T5 — is measured through a
reader that would rather have nothing, which puts the depth curve's 0.096 nats in scale:
it recovers 2 % of the self-inflicted wound. The "7.4 nats = the cell carries nothing"
figure used in the Method above, in `tul_code_target.yaml` and in the spec was a transient
from the aborted first draw, never a measurement; it is withdrawn everywhere.

## Not verified before launch

- The 12 CPU contracts and a CPU generation smoke; the GPU resume smoke is run before the
  queue lines are written and its exit code recorded here.
- The frozen coda was trained on cells with cosine ~0.32 to the truth (noise 3.0 renormed);
  how it reads a cell at cosine 0.2 to 0.5 is not known and is what P-T3 measures.
- One seed per arm.

## Results

### Arm A `tul-code-target` (detached cell), 20k steps, wandb `edgwqcr1`, commit d8e93d2

Ran 14:37 → 17:33 on the 5090, 14,393 tok/s, two isolated divergence-guard strikes (steps
5360 and 12300, neither consecutive), no abort. Artifacts in
`../results/2026-09-17-lctul-target/`. Every OWN / SHUF / ZERO reading below is from the
RE-RUN probe on the open-slot fix (14f004d); the pre-fix probe is void.

| Prediction | Reading | Holds |
| --- | --- | --- |
| P-T1 `val/code_target_cos` ≥ 0.20 at 20k | 0.147 at the 20k val, 0.160 on the last 500 logged train steps (from 0.004 at init, 0.144 by step 1500) | no |
| P-T2 `cos_l{T}` − `cos_l0` ≥ 0.05 | +0.1018 (l0 0.0604, l6 0.1622) | yes |
| P-T3 OWN − ZERO > 0, CI clear | +0.0054 [−0.0167, +0.0283] | no |
| P-T4 OWN − SHUF > 0, CI clear | +0.0230 [+0.0051, +0.0421] | yes |
| P-T5 CE at depth 6 − depth 1 ≤ −0.02, CI clear | −0.0320 [−0.0412, −0.0235]; depth 16 −0.0959 | yes |
| P-T6 OWN within 0.03 of the strict ruler, paired | −0.0570 [−0.0864, −0.0288] (arm A 0.1303, ruler 0.1873; the plain model is +0.0425 above the ruler) | no |
| P-T7 arm B's OWN − ZERO within 0.02 of arm A's | arm B stopped at step 15000 (checkpoint kept) to free the GPU; not scored | pending |
| P-T8 rate ≥ 0.9× the strict ruler | 14,393 vs 11,657 tok/s = 1.23× | yes |

Conditions at 20k, 120 cuts: OWN 0.1303, SHUF 0.1073, ZERO 0.1249, ORACLE 0.5645.

### What the four instruments say together

1. **The loop regresses, in one pass.** Exit cosine 0.004 → 0.144 by step 1500 and 0.160 at
   20k. P-T2 holds on its letter and the prediction was badly specified: `l0` is the ENTRY
   state, so `l6 − l0` spans the first pass and measures whether the loop does anything at
   all. The question P-T2 meant to ask is `l6 − l1`, which reads **+0.0052**.
2. **The cell carries a real, small, slot-specific signal.** OWN − SHUF is +0.023 with the
   interval clear of zero, and the worth profile reads +0.050 nats for the same contrast.
   This is the FIRST positive own-versus-foreign reading in the LCTUL family: the 24-bit plan
   arm read +0.008 [−0.006, +0.023], the LaDiR chain −0.006 [−0.024, +0.011], the flow arm the
   same. A regressed latent is not a sampled one, exactly as Wolfe said on 2026-09-17.
3. **It is not worth having.** OWN − ZERO is +0.005 with the interval across zero at
   generation, and the worth profile says zeroing the cell makes the frozen coda's token CE
   **4.608 nats better**. Against the ruler the arm is 0.057 worse, paired, and the plain
   model is 0.043 better than the ruler. Order: plain > ruler > arm A ≈ no cell.
4. **The depth curve is not depth earning.** P-T5 holds and the mechanism does not survive
   the depth-resolved probe: the cell's cosine to its own code is 0.150 / 0.154 / 0.152 at
   depths 1 / 6 / 16 and its rank is 17.5 / 17.0 / 17.5, while the passes move it 0.19 (depth
   6) to 0.29 (depth 16) of its norm in directions whose cosine to the target code is +0.004
   to +0.017, at or below the 0.031 a random direction in 1024 dimensions scores. Fifteen
   passes move a third of the vector and change its relation to the target by four
   thousandths.

### The reader is the open confound

Both CE instruments run through a coda frozen at the VAE stage that would rather have no
cell at all (−4.608 nats). The generation instrument agrees: OWN beats SHUF but not ZERO.
So the measured "the cell is harmful" is a statement about THIS reader, and the depth
curve's 0.096 nats is 2 % of a self-inflicted wound. Wolfe, 2026-09-17: "have we tried
using a fresh random coda after the core has learned?" — no arm ever has. The queued
unfreeze continuations ask the same question from a strong prior; a coda re-initialised and
trained alone on the frozen loop's cells would measure what the cell is worth to a reader
with no prior against it, against the ruler's +0.186 nats and the true code's 1.3.

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
