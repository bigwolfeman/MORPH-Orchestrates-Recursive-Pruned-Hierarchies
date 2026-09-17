# Agent Note: The code target — LaDiR with the TUL slot loop as the latent generator

Status: proposed

Date: 2026-09-17. Spec: [`docs/tul-code-spec.md`](../../../../docs/tul-code-spec.md) §17.
Prereg: [`lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md`](../../../../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md).
Follows [`2026-09-16-lctul-ladir-recipe.md`](2026-09-16-lctul-ladir-recipe.md) (the VAE stage
this arm resumes; its thinker stages are filed as failures) and
[`2026-09-14-tul-span-code.md`](2026-09-14-tul-span-code.md) (LCTUL: the slot holds the code of
the span it precedes). Supersession check: no earlier note regresses the slot loop onto a
frozen span code. [`2026-09-16-lctul-d-discrete-code-masked-denoiser.md`](2026-09-16-lctul-d-discrete-code-masked-denoiser.md)
stays as the record of the discrete sampler, both of whose arms are filed.

## Problem

Three thinkers were built to SAMPLE the next span's code from the past, and on generation
every sample reads as a foreign one: own minus foreign is +0.008 [−0.006, +0.023] on the
24-bit plan, +0.011 [−0.003, +0.024] on the flow code, −0.006 [−0.024, +0.011] on the
LaDiR chain, all on the same 120 validation cuts
(`lab/experiments/failures/2026-09-16-lctul-*`). The decoder side was never the limit:
the VAE stage's frozen coda decodes the TRUE code at 1.34 nats and its oracle continuation
scores 0.64 cosine to the true span, the best of any arm.

I read that as "the sample carries nothing about the span" and Wolfe corrected it: a
latent TRAINED to match the oracle code cannot carry nothing. It carries the conditional
mean of the code given the past, which is at least the 0.40 nats/token cross-span budget.
The earlier future-prediction targets that read flat
(`horizon-targets-degenerate-on-web-text`) were span-mean embeddings within 0.01 cosine of
each other; there was nothing to predict. The VAE code has effective rank 57 and a reader
that is known to decode it. That confound is gone, and the LaDiR shape with the TUL slot
loop as the reasoning model has never been run.

## Proposal

`tul.code_target: true` (`morph/model/tul.py`, `_check_code_target`; the model in
`morph/model/transformer.py::_tul_code_target_write`):

1. Resume the VAE stage (`tul-code-vae/step_10000.pt`) with E, the prelude, the embeddings
   and the CODA frozen (`training.train_only` lists the slot loop's core, its injections,
   the seed path and the projection; `train.py::drop_code_thinker_keys` drops the
   checkpoint's flow thinker loudly).
2. Run the ordinary strict slot loop. Its exit state, through `TULCodeProj` (identity init,
   unit-RMS cells, ternary-excluded), IS the M = `prefix_k` cells the coda reads at the
   prefix positions. `W_prefix` is built and inert.
3. Regress those cells onto E's code of the true next span: `2 (1 − cos)` per valid cell,
   `code_target_weight`. Arm A (`tul_code_target.yaml`) reads the cells with stop-gradient
   (the loop learns from the regression alone, LaDiR's rule); arm B
   (`tul_code_target_ce.yaml`) lets the token CE through the frozen coda train the loop too.
4. Read the per-pass cosine to the code (`code_target_cos_l{t}`, entry state at `l0`): the
   arm's depth instrument, beside the forced-depth K-curve and the semantic probe
   (`--kind target`: OWN / SHUF / ZERO / ORACLE).

THIS BREAKS THE STANDING RULE "never regress onto the slot state" (root `CLAUDE.md`; LCM
T3/4, CoCoMix §6b, BT §4.2). `oracle_z` carved the one earlier exception with "Wolfe
decides whether it ever ships". Wolfe directed this arm on 2026-09-17: "we can train TUL
latent to match oracle." The rule stays in `CLAUDE.md`; this note is the record of the
second exception and of who made it.

## Alternatives considered

- **A sampler on a richer or coarser code** (the three filed arms). Rejected by
  measurement: no code size tried is both carried by the coda and predictable from the
  past through a sampler.
- **The frozen VAE as an inference-time refiner** (draft, encode, regenerate). Not built:
  the coda decodes a code almost verbatim, so the fixed point is the draft. Kept as a
  cheap probe if this arm's OWN reads above ZERO.
- **Token CE through the frozen coda alone** (arm B with weight 0). Included as arm B's
  natural ablation but not queued first: it is the ordinary TUL signal with a fixed
  reader, and the question Wolfe asked is whether matching the oracle code is a target the
  loop can descend.
- **Regressing the raw exit state without a projection.** Rejected: the readout's
  statistic is not the code's, and a per-cell projection with identity init costs one
  `[M, d, d]` tensor and lets step 0 equal the rms-normed exit state.

## Acceptance criteria

The prereg's predictions, read at 20k on one seed: the exit cosine to the code rises above
its entry cosine and above 0.2; OWN beats ZERO and SHUF on the 120-cut semantic probe
with intervals clear of 0; the forced-depth K-curve on the frozen coda's CE is negative;
OWN lands within 0.03 of the strict ruler paired. Any one of the first two failing files
this note as rejected with the reading.

## Risks

- The conditional mean of a unit-RMS code, rms-normed, is a confident vector pointing at
  the mean direction; the frozen coda may read it as a wrong-but-confident code and do
  worse than ZERO. Arm B and the ZERO condition are the controls.
- One seed, 20k steps, a coda frozen at 10k steps of its own training: the bar against a
  jointly trained ruler is not level, and P-T6 says so.
- The regression is the rule-breaking term; if it wins, the win must be re-read against
  LCM's and CoCoMix's reasons for the rule before anything ships.
