# Experiment: three arms against the takeover — warmup, SIGReg, NTP dropout

Status: **planned. Warmup and SIGReg implemented and unit-tested; NTP dropout
not yet built.** Predictions frozen 2026-08-27 before any of the three runs.
Requested by Wolfe; worklist at [OVERNIGHT-WORKLIST.md](../../divergence/OVERNIGHT-WORKLIST.md).

## Question

The campaign's finding is that the takeover is credit assignment: the plan is
worth ~0.02 nats while the core captures >90 % of the gradient. The MUX head
([v1a](../failures/2026-08-25-mux-head-arm-v1a.md)) gave the plan a span-level
target and it learned only the corpus marginal. Three different attacks:

1. **Warmup** — the head could not learn because there were no representations
   yet (MUX starts from a pretrained model; we started from noise).
2. **SIGReg** — the slot states are geometrically COLLAPSED, so no target can be
   carried by them: effective rank 1.7-4.8 in 1024 dims, mean pairwise cosine
   +0.39..+0.71, at every checkpoint including the healthy ones.
3. **NTP dropout** — an attractor pulls the coda toward solving the prediction
   alone and ignoring the loop (Wolfe's framing); steps with no slots at all
   make the CORE do legitimate token work and should weaken it.

## Arms and protocol

All: 3500 steps, batch 6, `ademamix_alpha_cap=3.5`, `use_kernels=false`,
`eval_every=250`, `ckpt_every=500`. **Screened at seeds 0 AND 1** — n=1 is
unreadable here, because the control's own base rate is 1 seed in 4 with no
takeover (s0 aborts, s1 takes over at 2800, s2 at 3000, s3 never).

- **`tul_warmup`** — `tul_v1a2b` + `mux_activate_at: 0.4` (head on at step 1400).
- **`tul_sigreg`** — `tul_a1` + `sigreg_lambda`, no MUX head, so the geometry
  fix is tested alone.
- **`tul_ntpdrop`** — not built; see
  [the design note](../../.agents/notes/proposed/feature/2026-08-27-ntp-dropout.md).

### Method amended 2026-08-27, after the first launch, before any arm finished

**Change:** every arm now runs with `training.grad_probe_every=1` and
`training.grad_probe_path=<run>/probe.jsonl`. The abort guards stay at `0.0`.

**Reason:** the first launch used the shipped default `grad_probe_every: 0`, so
`preclip/core_share` was never logged. The only share series such a run has is
`gradnorm/*`, sampled every 100 steps, and
[`score_arms.py`](../../divergence/score_arms.py)`::fires` refuses a window with
fewer than 20 samples — its 50-step rule gets none at that cadence. Every arm
would have finished unscorable against S2, C2 and N1, which are the predictions
this experiment exists to test. The probe is read-only (one `_foreach_norm`, one
host sync, no RNG draw), so it costs ~0.5 % throughput and changes no result.
The guards stay off because the predictions ask whether the share CROSSES 0.5,
not whether a guard fires; aborting would replace the measurement with an
intervention.

**Not amended, and it limits the record:** `tul_v1a2b` seeds 0 and 1 already ran
under the coarse series and are NOT re-run. Seed 0's maximum share there is
0.026, twenty times under the threshold, so a finer probe cannot move its
verdict. Seeds 2 and 3 carry the fine probe. The replication is therefore scored
on the coarse series, which all four seeds have.

**Confound recorded here rather than discovered later:** `_preclip_probe` reads
`p.grad` after the backward of the FULL objective, so an auxiliary term that is
not uniform over the parameter tree lands inside the region it touches. Both the
MUX head and SIGReg push gradient into the core through the slot states, so
`preclip/core_share` on `tul_warmup`, `tul_sigreg` and every v1a arm counts the
auxiliary's own gradient as the core's. Measured precedent: a spectral-penalty
arm reached `preclip/total` 1.6e5 against a control's 1.35 and a share of 0.998,
all of it the penalty. This is a further reason **P** decides — the plan-off
ablation is auxiliary-free — and it means an arm that "avoids the takeover" by
share alone has proved less than it looks. `tul_center` and `tul_ntpdrop` add no
auxiliary loss and are clean on this point.

### SIGReg weight: probe before weighting

`tul_sigreg_probe` runs ~100 steps at `sigreg_lambda: 1e-8` — the term is
computed and logged but cannot move the model. The statistic scales with the
number of valid slots AND with the distance from N(0, I), and our slot states
have norm 200-450, so its magnitude is not guessable. **Rule fixed here:** set
λ so that `sigreg_weighted` is ≈ 5 % of `train/loss` at step 100, rounded to one
significant figure. Guessing blind is how v1a failed — its auxiliary outweighed
the LM objective.

## Predictions (frozen 2026-08-27, before any run)

Reference: control ppl_tok @3250 = 105.54 / 91.77 / 90.28 (s1/s2/s3), median
91.77; takeover in 3 of 4 seeds; plan-off worth baseline 0.0191 nats; MUX head
best 7.027 against a unigram baseline of 7.323.

- **W1 (warmup helps the head):** with the head activated at step 1400,
  `val/mux_local` at step 3000 is **below 7.023** — i.e. it beats the best the
  from-scratch head ever managed, on the same eval measurement. This is the
  direct test of Wolfe's hypothesis.
- **W2 (warmup does not cost CE):** `tul_warmup` ppl_tok @3250 seed median ≤ the
  control median 91.77.
- **S1 (SIGReg decollapses):** slot-state effective rank at the step-3000
  checkpoint is **> 10** (baseline 1.7-4.8), measured on the same probe batches.
- **S2 (SIGReg and the takeover):** SIGReg avoids the takeover (core share never
  crosses 0.5) in **both** screened seeds. Seed 0 is the sharp one — the control
  aborts there.
- **N1 (NTP dropout, when built):** avoids the takeover in both screened seeds.
- **P (the criterion that decides all three):** plan-off ablation worth at the
  step-3000 checkpoint is **> 0.0191 nats**, the measured baseline.

**P is the one that matters.** An arm that fixes the takeover and leaves plan
worth flat has stabilised training without making the core useful, which is not
the goal. Any arm passing its own letter but failing P is filed as a partial
result, never as a success.

## Risks and confounds recorded up front

- **SIGReg fights pre-norm.** MORPH's blocks are pre-norm, so the slot states'
  absolute scale is partly free; forcing N(0, I) is a large change in scale, not
  only in shape. If the arm dies immediately, suspect λ before suspecting the
  idea.
- **The `sigreg` term uses the global RNG** to draw its directions (single GPU
  needs no DDP sync). That consumes RNG, so a SIGReg run is not step-comparable
  to a non-SIGReg run under a fixed seed. It does not affect any arm's own
  validity, and every comparison here is at the level of curves, not bit
  equality.
- **NTP dropout cuts both ways** — see the design note's Risks section. The
  criterion is P, not perplexity.

---

# Added arm: CENTER — the root-cause version of SIGReg (added 2026-08-27, before the run)

Wolfe asked whether SIGReg could act per token. It cannot in the form he
remembered — SIGReg's population is SAMPLES, and the design choice is only what
counts as a sample — but the question pointed at the right level, and chasing it
produced a measurement that reframes the SIGReg arm.

**Measured on `tul-v1a2b/step_3500.pt` (CPU, embedding table only):**

```
||mean embedding||            = 0.4230
mean ||token - mean||         = 1.0491
```

A span bag-mean averages the per-token DEVIATIONS down by 1/sqrt(span) but
preserves that common mean EXACTLY, so every slot inherits the same vector.
Predicted pairwise cosine between two spans' bag-means,
`cos ≈ ||mu||² / (||mu||² + dev²/span)`:

| span | 4 | 10 | 20 | 32 |
| --- | --- | --- | --- | --- |
| predicted cos | 0.394 | 0.619 | 0.765 | 0.839 |

against a **measured** slot-state pairwise cosine of **+0.39 to +0.71** (spans
are min 4, cap 32, ~20 mean on OpenWebText). Both ends agree. The slot collapse
that motivated the SIGReg arm is therefore ARITHMETIC — a property of the
bag-mean construction plus embedding anisotropy — not a training pathology.

**Arm `tul_center`:** `tul_a1` + `center_bag_mean: true`, which subtracts the
batch's mean token signal (DETACHED — no dense gradient on the embedding table,
the mistake that killed v1a) from every real slot's bag-mean, leaving the dump
bin untouched so tail pads still receive `E_slot` alone.

Honest caveat, recorded because it limits the claim: `E_slot` is added to the
same bag-mean, so a CONSTANT shift is already within the model's reach. What
centering adds is that the batch mean TRACKS the drifting embedding mean, which
one learned vector cannot. This is an optimization/geometry intervention, not an
expressivity one — the same class as a normalization layer.

## Predictions for CENTER (frozen before the run)

- **C1 (it decollapses):** slot-state mean pairwise cosine at the step-3000
  checkpoint **< 0.20**, from a measured baseline of +0.39..+0.71.
- **C2 (does the geometry matter?):** core gradient share never crosses 0.5 in
  **both** screened seeds. This is the interesting one: if centering alone
  prevents the takeover, the entire campaign's disease reduces to a construction
  artefact and SIGReg is treating a symptom.
- **C3 (the criterion):** plan-off worth at step 3000 **> 0.0191 nats**.
- **C4 (no CE cost):** ppl_tok seed median at 3250 ≤ the control median 91.77.

If C1 holds and C2 fails, the collapse is real but not the cause of the
takeover, and that is worth knowing before any more geometry work.
