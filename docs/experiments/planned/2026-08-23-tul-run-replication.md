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

### Amendment 2026-08-23 — the arm as written cannot diverge

The Method above inherits `tul_short.yaml`'s `ademamix_alpha_cap: 1.0`. That value **is one
of the four surviving cures** — it held both arms for a full 20k run — so P3's onset does
not occur in this arm at ANY step, not merely by 2600. P3 was untestable as written, and
the "Known confound" below understates the problem: it is not that 2600 steps may be too
few, it is that the arm never takes over.

Every run of this experiment therefore adds `training.ademamix_alpha_cap=3.5` to restore
the divergent control (the value `base.yaml` ships and the one the 5/5 control divergence
was measured at). Predictions P1–P4 are unchanged; only the arm is corrected.

Seed 0's control onset is step ~4520, so the step budget rises from 2600 to 5000. At the
measured 2.08 steps/s that is ~40 min per run.

### Amendment 2026-08-23 — scope actually run in this session

Under a 1.5 h ceiling, one run of the corrected arm costs ~43 min including startup, so
the replicate PAIR does not fit. What ran is **one** run, seed 0, 5000 steps, probe every
step (`phase1-onset-s0`). Consequences, stated rather than discovered later:

- **P2, P3 and P4 are NOT tested by this session's run.** They need two runs. The gate
  they form is still open, and no cross-run checkpoint comparison may be made.
- **P1 is not re-tested either.** The prior evidence for it stands (RCA §17: `G-a1-b6-seed0`
  and `D8` shared config and seed and behaved differently) and the operator-level
  determinism result is now direct, but neither is this experiment's own run.
- What the single run DOES support is a **within-run temporal ordering** of the probed
  quantities across the onset. That ordering is measured along one trajectory and never
  compares two, so it does not depend on the replication gate. It is the Phase 1.3
  deliverable and it is the only claim this run licenses.

Incidental evidence collected on the way, worth recording because it bears on P4: two
60-step runs at seed 0 differing ONLY in `grad_probe_every` (0 vs 1) reached
`train/loss` 8.4143 and 8.4520 at step 40. The probe is `no_grad` and `detach`-only and
cannot change the math, so the gap is the residual attention-backward atomics reacting to
changed launch timing. A 4.5e-2 loss gap by step 40 is consistent with P4 and is a warning
that the replicate pair must be run with byte-identical commands, probe included.

Read with the onset rule above, not by eye.

## Known confound, stated before the run

The control diverges 5/5 but at onset steps 2080–6200, so a run to 2600 may not reach
onset at all. If neither replicate pair takes over by 2600, P3 is untestable as written
and the comparison falls back to the P2/P4 loss-agreement predictions plus a trajectory
distance (`|loss_a − loss_b|` against step). That fallback is named here, in advance, so
it is not invented after seeing the data.
