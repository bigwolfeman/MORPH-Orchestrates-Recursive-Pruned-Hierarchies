# Experiment: is the TUL divergence a forward–backward asymmetry in the looped core?

Status: failure

## Question

Wolfe, 2026-08-23: *"A seed is not a divergence root cause."* Correct. The 2026-08-22
note [`failures/2026-08-22-tul-divergence-cause.md`](../failures/2026-08-22-tul-divergence-cause.md)
named two candidates (contractivity vs the optimizer's slow-EMA push) and did not
separate them. Four new facts, measured from the batch-12 campaign logs today, narrow it:

1. `train/loss` sits at 4.55–4.80 through the whole onset window. At step 2040 the loss
   is 4.5496 and `train/grad_norm` is 105.4, up from 1.36 one step earlier. The loss only
   degrades after ~step 2200, once the gradient has already exploded.
2. The surviving and diverging A1 arms eat the **same batches in the same order**: the
   paired residual correlation of `train/loss` over steps 0–1900 is **0.938 at lag 0** and
   **0.233 at lag 1**. The seed changes the weight initialisation, not the data stream.
3. `gradnorm/core` — the looped weights' share of the total gradient norm — ratchets
   0.0092 → 0.0426 → 0.1083 → 0.8996 over steps 1800–2100, about 140 steps *before* the
   norm explodes, and never returns. It is a ratchet, not a level: the gate arm touches
   0.3462 at step 700 and falls back to 0.0783.
4. The runaway is concentrated in **one** core block. At the abort, block 0's one-pass
   gain σ(gate_up)·σ(down) is 22.95 while blocks 3–5 sit at 4.5–5.3 and are *declining*.

Those four rule out a bad batch and rule out "seed" as an explanation, but they do not
name a mechanism. This experiment tests one.

## Hypothesis

In a weight-shared loop the forward pass is renormalised (HyperConnection carrier plus
`input_norm`) and the backward pass is not. The core map's amplification is therefore
**invisible to the loss and fully visible to the gradient**: the loss cannot see a gain
that the carrier renormalises away, while the gradient accumulates the full product over
all T iterations. Divergence is that product crossing 1 and compounding as `ρ^T`, and the
single global gradient clip then converts the core's explosion into a ~1e-7 multiplier on
every other parameter group, which is why prelude, coda, embeddings and the TUL parameters
stop learning while the loss still looks fine.

## Predictions

Let `f` be one core step with context held fixed (the `_apply_core_step` map the real
loop applies), `σ_T = σ_max(J of f^T)` and `ρ_eff = σ_T^(1/T)`. Measured with the
validated estimator in `ignore/perf/order_param.py` (Gate A: non-normal ρ=0.5 < 1 < σ=6;
Gate B: core-step parity against the real forward).

- **P1.** On `tul-a1r/DIVERGED_step_2080.pt` (just past onset), `σ_T` grows with T faster
  than linearly, and `ρ_eff > 1.0` at T = 6.
- **P2.** On the same checkpoint the forward carrier gain `‖f^T(h)‖ / ‖h‖` stays inside
  [0.5, 2.0] for every T in 1…8 — flat while `σ_T` climbs. This is the asymmetry itself.
- **P3.** On the survivors (`tul-a1/step_5000.pt`, `tul-gate/step_5000.pt`) `ρ_eff ≤ 1.0`
  at T = 6, i.e. the survivors are on the other side of the same quantity.
- **P4.** `σ_T` at T = 6 separates the diverged checkpoint from both survivors by more
  than 5×. (The per-linear σ_max does NOT: the gate arm survives to 20k with a core
  linear at 5.618, higher than the diverged arm's 5.508 at its abort. If the composition
  fails to separate them either, the contractivity story is wrong at the block level and
  the alignment/order-parameter story is what is left.)
- **P5.** `tul-a1r/DIVERGED_step_4160.pt` (deep into the blow-up) scores a larger `σ_T`
  at T = 6 than `DIVERGED_step_2080.pt` — the quantity keeps moving in one direction.

Falsification: if `ρ_eff ≤ 1` on the diverged checkpoint, or if the forward gain climbs
with T alongside `σ_T`, the asymmetry hypothesis is dead and the loss-blind story is wrong.

## Method

```
PYTHONPATH=$PWD python ignore/perf/depth_gain.py --config morph/configs/tul_a1.yaml \
  --ckpt a1r_div_2080=tul_a1r=checkpoints/morph/tul-a1r/DIVERGED_step_2080.pt \
  --ckpt a1r_div_4160=tul_a1r=checkpoints/morph/tul-a1r/DIVERGED_step_4160.pt \
  --ckpt a1_lived_5k=tul_a1=checkpoints/morph/tul-a1/step_5000.pt \
  --ckpt gate_lived_5k=tul_gate=checkpoints/morph/tul-gate/step_5000.pt \
  --depths 1,2,3,4,6,8 --kpow 60 --restarts 3 --real-text --seq 256 --batch 2 \
  --csv lab/experiments/results/tul_depth_gain.csv
```

One batch of real validation text is drawn ONCE and reused for every checkpoint, fp32
throughout, both estimator gates must pass or the script aborts.

## Known confound, stated before the run

The survivor checkpoints are at step 5000 and the diverged ones at 2080/4160. There is no
survivor checkpoint at step 2080, so the comparison is not step-matched and a difference
could in principle be a training-time effect rather than a health effect. P5 (the same
arm at two of its own steps) is the within-arm control that does not need matching.
Settling it properly needs a re-run of A1r with dense checkpoints through the onset, which
is a separate ~20-minute experiment and is not attempted here.

---

# Results

Run 2026-08-23. `ignore/perf/depth_gain.py`, both estimator gates PASS on every
checkpoint (Gate A: est 6.0414 vs true 6.0414 on the non-normal ρ=0.5<1<σ=6 case;
Gate B: core-step parity max|probe − forward| = 0.000e+00). Data:
[`../results/tul_depth_gain.csv`](../results/tul_depth_gain.csv),
[`../results/tul_lin_ratio.csv`](../results/tul_lin_ratio.csv),
[`../results/tul_lin_ratio_local.csv`](../results/tul_lin_ratio_local.csv).

## Scoring the predictions: 0 of 5 confirmed

| | prediction | outcome |
|---|---|---|
| P1 | `σ_T` grows super-linearly and `ρ_eff > 1` at T=6 on the diverged checkpoint | **unresolvable** — `σ_T` is not a defined quantity there (below) |
| P2 | forward carrier gain stays in [0.5, 2.0] on the diverged checkpoint | **falsified** — 33.9 at T=1 rising to 76.2 at T=8 |
| P3 | survivors have `ρ_eff ≤ 1` at T=6 | **falsified** — every survivor is 2.29–3.23 |
| P4 | `σ_T` at T=6 separates diverged from survivors by >5× | **unresolvable** — same reason as P1 |
| P5 | step 4160 scores a larger `σ_T` than step 2080 | **falsified** on the T=4 proxy: 34604 < 75508 |

The hypothesis is dead. The forward is **not** blind to the amplification: on the diverged
checkpoint the carrier itself grows 66× over six iterations. And `ρ ≤ 1` is not the line
between healthy and sick — **every** checkpoint measured here, including two that trained
to 20k, has `ρ_eff` between 1.9 and 3.2 at depth 6–8.

## What the run found instead: the linearisation is gone

`depth_gain.py` returned exactly `σ = 0.0000` at T ≥ 6 on both diverged checkpoints and
finite values on every survivor. That asymmetry sent us to check the estimator rather than
to publish the zero. `ignore/perf/saturation_check.py` perturbs the carrier by `eps·‖h‖`
and measures how far the output moves:

| eps (relative) | 1e-3 | 1e-2 | 1e-1 | 1.0 |
|---|---|---|---|---|
| ‖Δ output‖, A1r DIVERGED@2080, T=6 | 9.29e4 | 9.31e4 | 9.52e4 | 9.31e4 |
| ‖Δ output‖, A1 lived@5000, T=6 | 3.10 | 30.3 | 292 | 1797 |

The survivor responds in proportion to the step, across three decades. The diverged
checkpoint returns the **same size of response to a 0.1 % perturbation as to a 100 % one**,
with 100 % of output elements moving. There is no local linearisation, so there is no
Jacobian, so `σ_max` has no referent — and a finite-difference estimator will still print a
confident number for it (75507.74 at T=4). `depth_gain.py` now measures
`lin_ratio = ‖Δ(eps_hi)‖ / ‖Δ(eps_lo)‖`, which a locally linear map returns as
`eps_hi/eps_lo`, and marks a row INVALID when it is off by more than 10×. Every diverged
row is now marked; every survivor row at step 5000 passes.

That is the mechanical statement of what divergence IS here: **the optimizer keeps stepping
on a gradient that describes the loss surface at no step size it might take.** It is not a
spectral radius crossing 1.

## Direction robustness, and a ranking that does NOT survive it

8 independent probe directions per cell, mean ± sd, at loop depth 6 (the model's own
`mean_depth`):

| checkpoint | lin_ratio, eps 1e-3→1.0 (linear = 1000) | lin_ratio, eps 1e-4→1e-3 (linear = 10) |
|---|---|---|
| A1r DIVERGED @2080 | **1.00 ± 0.02** | **0.99 ± 0.02** |
| A1 lived @5000 | 633 ± 106 | 4.59 ± 0.62 |
| A1 lived @20000 | 29.2 ± 8.7 | 6.70 ± 1.36 |
| gate lived @20000 | 409 ± 40 | 1.54 ± 0.10 |

The diverged checkpoint sits at ~1.0 under **both** eps pairs, four decades apart, with a
standard deviation of 0.02. That is unambiguous.

The survivors are not. The two eps pairs rank them in **opposite orders** — A1@20k is the
worst survivor on the wide pair and the best on the local pair; the gate arm is the reverse.
So `lin_ratio` identifies a map that has already lost its linearisation and does **not**
grade the health of maps that still have one. A first draft of this note read the wide-pair
column as "the cap delays rather than cures, A1 at 20k is drifting toward the dead value";
the local column kills that reading and it is withdrawn. Neither column is trusted for
ranking survivors, and no cure-versus-delay verdict is claimed here.

## The one quantity that does separate, and its limit

Forward carrier gain `‖f^T(h)‖ / ‖h‖` at T=6, one batch of real validation text, all six
checkpoints:

| A1r@2080 | A1r@4160 | A0@20k | A1@5k | A1@20k | gate@5k | gate@20k |
|---|---|---|---|---|---|---|
| 66.49 | 34.06 | 2.93 | 2.42 | 2.75 | 1.53 | 5.71 |

Diverged 34–66, survivors 1.5–5.7: a 6× floor and a 43× ceiling, from a plain norm ratio
with no estimator in it. **But every survivor value is a healthy value**, so this too tells
you a run has died, not that one is about to.

## What this does NOT answer

Nothing here is a *predictor*. Every diverged checkpoint measured is post-onset (steps 2080
and 4160; onset is ~1900–2040), because no checkpoint exists inside the onset window. So
these quantities describe what divergence IS, not what precedes it. The pre-onset signal
comes from the training logs instead, and it exists today: `gradnorm/core` ratchets
0.0092 → 0.0426 → 0.1083 → 0.8996 over steps 1800–2100, roughly 140 steps before
`train/grad_norm` moves, and it never returns. It is a ratchet and not a level — the gate
arm touches 0.3462 at step 700 and falls back to 0.0783 without dying.

The pre-registered step-mismatch confound stands: survivors are at 5000 and 20000, diverged
at 2080 and 4160. Settling it needs A1r re-run with checkpoints every 100 steps through the
onset, which is ~20 GPU-minutes and is not done.

## Updated hypothesis

Divergence in the TUL core is a **loss of local linearisation of the looped map**, not a
contractivity failure. Contractivity is ruled out as the discriminator: `ρ_eff` is 1.9–3.2
on runs that finish. The next experiment should checkpoint A1r every 100 steps from 1700 to
2200 and ask which of `gradnorm/core`, `lin_ratio` and the forward carrier gain moves
first — that is the only way to turn any of this into an abort criterion.
