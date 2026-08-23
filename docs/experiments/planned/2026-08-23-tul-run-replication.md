# Experiment: does a fixed-seed TUL run replicate?

Status: planned

## Question

`docs/tul-divergence-rca.md` §17: `G-a1-b6-seed0` shared `D8`'s config **and** seed and
behaved differently — D8 hit 0.866 core share at step 300, G did not take over at all,
and both were before either run's first eval. The note concludes "the run is not
reproducible at fixed seed", suspects `bag_mean`'s `index_add_` float atomics, and marks
the suspicion NOT VERIFIED. §22 prepared this replicate as arm **P2** and it was never run.

Everything in the divergence programme rests on it. Two checkpoints at the same step from
two runs are not comparable if the runs are different trajectories, so no bisect, no
onset study and no mediation analysis is readable until this is settled.

The suspicion is now verified at the operator level (2026-08-23,
`ignore/perf/determinism_probe.py`): `bag_mean` is non-bit-identical on **20/20** repeats
in forward and backward, with 30.7 % of backward elements differing, and
`torch.use_deterministic_algorithms(True)` does not flag it. One full TUL training step
repeats non-identically **4/4** with a worst-tensor **relative gradient error of 3.92e-2**.
A deterministic one-hot `bmm` replacement (`ignore/perf/bagmean_deterministic.py`) is
bit-identical 0/20, agrees within bf16 epsilon, and is 10 % faster; with it in place the
full-step error falls to 3.17e-3 — 12× better, **not zero**. The residue is `tl.atomic_add`
in `fused_csa_attention.py:279` and `fused_hca_attention.py:288`.

## Hypothesis

`bag_mean` is the dominant source of run-to-run divergence in the TUL path. With it made
deterministic, two byte-identical runs at one seed will track each other closely enough to
bisect, even though the attention-backward atomics remain.

## Predictions

Two runs of `tul_a1`, same seed, same config, eval DISABLED, to step 2600.

- **P1.** With the current `index_add_` `bag_mean`, the two runs differ in `train/loss` at
  step 1000 by more than 1e-3, and their onset steps differ by more than 25.
- **P2.** With the deterministic `bag_mean`, `train/loss` at step 100 agrees to 4 decimal
  places.
- **P3.** With the deterministic `bag_mean`, the two runs' onset steps agree within ±25,
  where onset is the first of ≥3 consecutive logged points with `gradnorm/core` > 0.5
  (the rule fixed in RCA §21, implemented rather than applied by hand).
- **P4.** With the deterministic `bag_mean`, `train/loss` at step 1000 still differs by
  more than 1e-6 — the attention atomics remain, so exact bit-replication is NOT expected
  and its absence must not be read as a failure of P3.

**Falsification and the consequence:** if P3 fails, `bag_mean` was not the dominant source,
the attention-backward atomics move to the top of the queue, and Phase 1 of
[`the plan`](../../../.agents/notes/proposed/process/2026-08-23-divergence-root-cause-plan.md)
does not start. Do not proceed on an irreproducible run.

## Method

```
# arm R0a / R0b — current bag_mean, seed 0, twice
PYTHONPATH=$PWD python -m morph.training.train --config-name tul_a1 \
    training.steps=2600 training.eval_every=999999 training.seed=0 \
    wandb.name=repl-idxadd-a     # and -b

# arm R1a / R1b — deterministic bag_mean, seed 0, twice
#   (same command, after task 0.1 lands)
```

Eval is disabled because an eval pass consumes RNG and would act as a hidden variable —
the same reason the RCA's seed sweep disabled it. Batch 12, seq 1024,
`ademamix_alpha_cap` 1.0 as `tul_short.yaml` ships. ~25 min per run at 0.54 s/step, so
four runs is under 2 h.

Read with the onset rule above, not by eye.

## Known confound, stated before the run

The control diverges 5/5 but at onset steps 2080–6200, so a run to 2600 may not reach
onset at all. If neither replicate pair takes over by 2600, P3 is untestable as written
and the comparison falls back to the P2/P4 loss-agreement predictions plus a trajectory
distance (`|loss_a − loss_b|` against step). That fallback is named here, in advance, so
it is not invented after seeing the data.
