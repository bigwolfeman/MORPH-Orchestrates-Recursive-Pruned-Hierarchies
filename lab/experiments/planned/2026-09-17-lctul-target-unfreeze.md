# Experiment: the code target, continued 10k steps with the reader unfrozen

Status: planned

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

## Not verified before launch

- The parent arms have not produced their 20k checkpoints; the lines are inserted on the
  assumption that both finish (if a parent aborts, its continuation is pulled).
- One seed each; the coda trains at flat 1e-4 from a fresh optimizer with no ramp (the
  loop and the coda are both trained weights at 20k; the LaDiR stages used the same
  opening).

## Results

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
