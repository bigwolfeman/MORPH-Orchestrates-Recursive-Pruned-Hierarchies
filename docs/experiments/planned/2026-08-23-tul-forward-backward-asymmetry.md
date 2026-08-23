# Experiment: is the TUL divergence a forward–backward asymmetry in the looped core?

Status: planned

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
  --csv docs/experiments/results/tul_depth_gain.csv
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
