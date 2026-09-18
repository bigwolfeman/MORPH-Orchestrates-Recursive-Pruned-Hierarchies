# Experiment: the code target, continued 10k steps with the reader unfrozen

Status: success

Date: 2026-09-17 (frozen before either continuation starts; arm A's first draw is at
step ~1k of 20k as this is written). Parent: `2026-09-17-lctul-target-slot-loop.md`.
Note: `.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md`.

## Question

Wolfe: "we just take the checkpoints and take them out 10k more steps unfrozen. 1
checkpoint at half way and see if it is improving." With the coda (and its injections,
readout norm and mixer) trainable from the 20k checkpoint — encoder, prelude and
embeddings still frozen — does the reader adapt to the loop's predicted cells and does
generation improve over the arm's own 20k reading?

## Hypothesis

H1: the frozen coda was the bottleneck (it was trained to read noised TRUTH codes, never a
predicted cell); let it adapt and the coda's CE on the predicted cell falls and OWN rises
above the arm's 20k OWN. H0: the coda learns to ignore the cell (the flow arm's measured
failure: a coda trained on samples reads OWN − SHUF +0.009, CI spanning 0) and OWN does
not move while CE improves through the coda's own token path.

## Predictions (frozen)

Arms `tul-code-target-uf` (from arm A's 20k) and `tul-code-target-ce-uf` (from arm B's
20k), `tul_code_target(_ce)_uf.yaml`, 10k steps each, checkpoints at 25k and 30k, one seed.

- P-U1 (the reader adapts): the coda's CE on the predicted cell (`val/loss`, the sampled
  reading) at 30k is ≥ 0.30 nats below the same arm's at 20k, and at 25k already ≥ 0.15
  below. 60 %.
- P-U2 (generation improves): semantic probe OWN cos_true at 30k minus the same arm's OWN
  at 20k > 0 with the paired 95 % interval clear of 0. 40 %.
- P-U3 (the cell is still read, not ignored): at 30k, OWN − ZERO on cos_true > 0 with the
  interval clear of 0, AND OWN − SHUF > 0. 35 %.
- P-U4 (half way is on the path): the 25k OWN sits between the 20k and 30k readings
  (monotone) on cos_true. 50 %.
- P-U5 (the loop keeps its target): `val/code_target_cos` at 30k within 0.03 of its 20k
  value or higher (the CE route does not pull the cell off the code). 55 %.

## Binding

- P-U1 and P-U2 and P-U3 → the reader was the limit; the next arm trains the reader from
  the start (arm B with the coda in `train_only` from step 0).
- P-U1 holds, P-U2 or P-U3 fails → the coda improves by ignoring the cell (H0): the note
  records that a reader adapting to a mean cell drops it, and the refiner probe is next.
- P-U1 fails → 10k steps of the reader change nothing; report and stop.

## Method

Queue lines behind arm B (kind slot, commit recorded at insertion, EXTRA with the resume
path), so each continuation starts after its parent's 20k checkpoint exists. The runner's
smoke does not carry EXTRA; the `_uf` config is smoked once on the Spark with the VAE
checkpoint as the resume stand-in (same drop path) before the lines go in. Watcher: a
second `spark_code_probes_target.sh` instance with `ARMS`/`STEPS` for the two continuations
at 25000 and 30000. Pairing: `code_semantic_pair.py`, each continuation against its own
parent's 20k probe and the 2026-09-16 controls.

Amended 2026-09-17 14:55 (reason: the smoke ran and the lines were inserted): Spark smoke
of `tul_code_target_uf` at 0f1d1b7 with the VAE checkpoint as the resume stand-in, 12 steps
past the resume: exit 0, 118.3M trainable / 151.6M frozen, fresh optimizer, guard ceiling
100000. Queue lines inserted at 14:55 behind arm B (commit 0f1d1b7); the continuation
watcher started 14:46 for steps 25000 and 30000. Predictions unchanged.

## Not verified before launch

- The parent arms have not produced their 20k checkpoints; the lines are inserted on the
  assumption that both finish (if a parent aborts, its continuation is pulled).
- One seed each; the coda trains at flat 1e-4 from a fresh optimizer with no ramp (the
  loop and the coda are both trained weights at 20k; the LaDiR stages used the same
  opening).

## Results

Run on the 3070 and the DGX Spark (identical OWT shard, byte-identical, so both hosts score
the same cuts; `code_semantic_pair.py` accepted a 3070 probe against a Spark reference, which
verifies cut agreement line for line).

**Only one of the two arms ran.** `tul-code-target-ce-uf` was launched 09:15 at commit
0f1d1b7, whose queue line asked to resume `tul-code-target-ce/step_20000.pt`. That file has
never existed (arm B stops at step_15000), 0f1d1b7 predates the resume guard, and the arm
silently started from step 0 with `train_only` freezing 151.6M randomly-initialised
parameters. It was killed at 09:42 (exit 143, last=2750) and produced no checkpoint, so both
its readouts are `READOUT SKIP`. The guard is now on master (5b791ca): a re-queued line at
master or later fails loudly instead. Everything below is `tul-code-target-uf` only.

| | 20k (arm A) | 25k | 30k |
|---|---|---|---|
| `val/loss` (sampled) | 8.71 | 4.27 | 4.25 |
| semantic OWN cos_true | 0.1303 | 0.1553 | 0.1592 |
| `val/code_target_cos` | 0.1499 | | 0.1784 |

Semantic probe at 30k, 120 cuts: OWN 0.1592, SHUF 0.1368, ZERO 0.1414, ORACLE 0.1463.
OWN-SHUF +0.0225 [+0.0021, +0.0422]; OWN-ZERO +0.0179 [+0.0025, +0.0329];
ORACLE-OWN -0.0129 [-0.0364, +0.0104].
Paired against arm A's 20k over the same 120 cuts: 30k +0.0290 [+0.0040, +0.0533];
25k +0.0250 [-0.0010, +0.0516] (marginal: an earlier bootstrap draw of the same pair gave
[+0.0007, +0.0510], so the 25k gap straddles zero across draws and carries no weight).

- P-U1 HOLDS. 8.71 -> 4.25 at 30k (>= 0.30 required) and 8.71 -> 4.27 at 25k (>= 0.15).
- P-U2 HOLDS. +0.0290 [+0.0040, +0.0533], paired, clear of 0.
- P-U3 HOLDS. OWN-ZERO +0.0179 with the interval clear of 0, and OWN-SHUF +0.0225 > 0.
- P-U4 HOLDS. 0.1303 -> 0.1553 -> 0.1592 is monotone.
- P-U5 HOLDS. `val/code_target_cos` 0.1499 -> 0.1784, higher, not merely within 0.03.

Costs not in the predictions, both measured:

- ORACLE collapsed 0.5645 -> 0.1463. Arm A's frozen coda decoded the TRUE code at cos 0.5645;
  after 10k unfrozen steps the same coda manages 0.1463. `val/ce_tf` moved 1.43 -> 4.17. The
  reader stopped being a decoder of E's code space, giving up far more there (-0.42) than it
  gained on the loop's cells (+0.029).
- Depth-dependence FELL. Forced-depth sweep, 480 rows: arm A K1-K6 +0.0323 [+0.0236, +0.0415]
  (K3-K6 +0.0305); uf K1-K6 +0.0047 [+0.0038, +0.0058] (K3-K6 +0.0002). Unfreezing the reader
  cut the K-curve about sixfold. Arm A's larger number is NOT better: its cell drifts toward
  the corpus mean with depth (`pred.zbar` 0.3081 -> 0.3225) and its frozen coda rewards the
  blander input. uf's CE minimum (depth 6) coincides with its cosine maximum (depth 6), so
  only the ADAPTED reader's CE tracks the code match. See the 2026-09-17 progressive
  experiment for the three-arm table.

## Verdict

success (on the one arm that ran). All five predictions held. The reader was the limit on
whether the cell is USABLE: the same weights read -4.64 nats frozen and +0.177 adapted, and
generation improves by +0.029 cos_true paired.

The binding clause fires: P-U1 and P-U2 and P-U3 -> the next arm trains the reader from the
start. Two qualifications go with it. First, `tul-code-target-ce-uf` must be re-run, which
needs arm B carried from step_15000 to step_20000 first. Second, expect a usable cell and a
FLAT K-curve: unfreezing reduced depth-dependence, and a reader trained on loop cells from
step 0 will never have the oracle capability whose loss is invisible when it was never there.

## Updated hypothesis

H1 is confirmed for usability and refuted for depth. A frozen reader trained on noised truth
codes is not a neutral measuring device: it prefers no cell to a confidently-wrong one, so it
reports a genuine latent as worse than nothing, and its depth-CE curve tracks how generic the
cell becomes rather than how well it matches the code. Adapting the reader fixes the first
problem and shrinks the second signal.

What the unfreeze does NOT do is make the loop use depth. Across arm A, uf and the
progressive arm the centred cosine to the own code peaks at depth 6, the training mean, on
every arm. Depth-dependence is a property of the training draw (E6), not of the reader.

Next: score generation on an arm whose reader trains from step 0, and keep an oracle readout
so the code-space capability can still be measured. The open question this run raises is
whether the reader must give up E's code space to read the loop's cells, or whether that
trade is an artefact of starting from a coda already specialised on truth codes.
