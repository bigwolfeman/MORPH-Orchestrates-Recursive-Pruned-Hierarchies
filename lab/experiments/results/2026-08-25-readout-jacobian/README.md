# Readout Jacobian: does the coda's main-token CE gradient reach the slot states?

**NOT pre-registered — diagnostic probe.** No planned/ file exists for this run; it is
a follow-up measurement on the "cold start / dead readout" hypothesis raised by the
Shapley result (`region_shapley.py`: plan worth 0.0007–0.02 nats on ce_main).

## Question

Has the coda learned to ignore the slot positions? Concretely: is the gradient of the
plain main-token CE (`ce_main` ONLY — not the total loss, not `ce_emit`, not
`ce_plast`) with respect to the slot states at the coda input near zero, and much
smaller than the same gradient into ordinary token states? If yes, any plan-side
objective is gradient-starved regardless of its design.

## Method

`lab/divergence/readout_jacobian.py`. Builds the TUL A1 model via the shared
`lab/divergence/_build.py` path (`model.use_kernels=false`, batch_size 6), loads each
checkpoint with the trainer's own `load_checkpoint`, and patches
`morph.model.transformer.scatter_positions` to `retain_grad()` on `x_coda` — the
tensor produced at `_forward_tul`'s scatter (transformer.py ~line 2059), i.e. the coda
input with the projected plan written into the slot positions. Forward runs in eval
mode (deterministic mean slot depth, dropout off) under bf16 autocast with grads
enabled; `out["ce_main"]` is backwarded; per-position L2 norms of `x_coda.grad` (g)
and `x_coda` (h) are split into SLOT (valid scatter targets from `prefix_project`'s
`pos`), TOKEN (`~layout.slot_mask`), and PAD (tail-pad slots, negative control).
Model grads are zeroed between batches; the optimizer is never stepped. The same 8
eval batches (validation split, skip_samples=60000, same loader args as
`region_shapley.py`) are materialised once and reused for all three checkpoints.

## Command

```
cd /home/wolfe/morph-perf && PYTHONPATH=/home/wolfe/morph-perf \
  /home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python \
  lab/divergence/readout_jacobian.py \
  --ckpts checkpoints/morph/onset-capture/ROLL_step_1650.pt \
          checkpoints/morph/onset-capture/ROLL_step_1750.pt \
          checkpoints/morph/onset-capture/ROLL_step_1850.pt \
  --out lab/experiments/results/2026-08-25-readout-jacobian/results.json
```

Exit code 0. Raw numbers: `results.json` (this directory).

## Results

g = per-position L2 of d(ce_main)/d(x_coda); h = per-position L2 of x_coda.
n per checkpoint: 4624 slot positions, 50668 token positions, 4 tail pads.

### ROLL_step_1650 (early) — ce_main 4.8532

| group | stat | mean | median | p90 | max |
|---|---|---|---|---|---|
| slot | g | 3.1382e-06 | 2.6012e-06 | 5.3403e-06 | 2.6656e-05 |
| slot | h | 2.1757e+02 | 2.0861e+02 | 2.8455e+02 | 4.6500e+02 |
| slot | g·h | 6.4268e-04 | 5.5477e-04 | 1.0259e-03 | 4.6782e-03 |
| token | g | 4.6128e-05 | 4.1637e-05 | 7.8271e-05 | 2.5345e-04 |
| token | h | 6.4489e+01 | 6.4490e+01 | 6.4526e+01 | 6.4588e+01 |
| token | g·h | 2.9750e-03 | 2.6850e-03 | 5.0481e-03 | 1.6347e-02 |
| pad | g | 0 | 0 | 0 | 0 |

Headline: median_g slot/token = **0.0625**; median_g·h slot/token = **0.2066**.

### ROLL_step_1750 (primary, healthy, pre-takeover) — ce_main 4.8093

| group | stat | mean | median | p90 | max |
|---|---|---|---|---|---|
| slot | g | 2.9215e-06 | 2.3089e-06 | 5.3623e-06 | 2.7293e-05 |
| slot | h | 2.3727e+02 | 2.2955e+02 | 3.0749e+02 | 5.8795e+02 |
| slot | g·h | 6.3945e-04 | 5.4168e-04 | 1.0718e-03 | 5.5929e-03 |
| token | g | 4.6390e-05 | 4.1955e-05 | 7.8292e-05 | 2.8841e-04 |
| token | h | 6.4507e+01 | 6.4508e+01 | 6.4547e+01 | 6.4604e+01 |
| token | g·h | 2.9926e-03 | 2.7062e-03 | 5.0514e-03 | 1.8611e-02 |
| pad | g | 0 | 0 | 0 | 0 |

Headline: median_g slot/token = **0.0550**; median_g·h slot/token = **0.2002**.

### ROLL_step_1850 (at takeover) — ce_main 4.7620

| group | stat | mean | median | p90 | max |
|---|---|---|---|---|---|
| slot | g | 2.2604e-06 | 1.7760e-06 | 4.2956e-06 | 2.2300e-05 |
| slot | h | 4.7376e+02 | 4.5505e+02 | 6.3708e+02 | 1.1491e+03 |
| slot | g·h | 9.7189e-04 | 8.0360e-04 | 1.7147e-03 | 9.7834e-03 |
| token | g | 4.6580e-05 | 4.1958e-05 | 7.8774e-05 | 2.8818e-04 |
| token | h | 6.4524e+01 | 6.4525e+01 | 6.4563e+01 | 6.4628e+01 |
| token | g·h | 3.0057e-03 | 2.7071e-03 | 5.0843e-03 | 1.8601e-02 |
| pad | g | 0 | 0 | 0 | 0 |

Headline: median_g slot/token = **0.0423**; median_g·h slot/token = **0.2969**.

## Sanity checks (all passed, all three checkpoints)

- Capture fired exactly once per loss forward: `captures_per_forward` was `[1]*8`
  for every checkpoint; grad was not None.
- ce_main finite and plausible: 4.76–4.85 nats.
- Slot input norms h nonzero (median 209–455 — far from zero).
- Token grad norms nonzero (median ~4.2e-05).
- Bonus negative control: the 4 tail-pad positions (end of row, label −100, nothing
  after them to attend back) have exactly zero gradient, as causality requires.

## Reading

The readout is weak but not dead. The raw gradient of ce_main into a slot position is
16–24× smaller than into a token position (median ratio 0.063 → 0.055 → 0.042 across
1650/1750/1850), so the hypothesis's direction is confirmed: the coda routes most of
its sensitivity to token states, and the imbalance worsens toward the takeover.
However, it is a ~20× suppression, not a numerically-zero channel — a plan-side
objective is heavily disadvantaged, not literally starved. The scale-adjusted
sensitivity g·h tells a second story: slot input norms are 3–7× LARGER than token
norms (209 → 230 → 455, doubling at the takeover — the known state-norm blowup),
which mechanically props the g·h ratio up to 0.20–0.30. Per unit of parameter-induced
relative change, the slot channel carries roughly 3–5× less main-CE signal than a
token, and the raw per-unit-input gradient (what an upstream plan objective actually
inherits through the scatter) is falling while the slot norm inflates — consistent
with the coda progressively down-weighting a channel whose magnitude is exploding.
Caveats: single fixed eval set (8×6 seqs), eval-mode forward (no token-state dropout,
so the coda's need for the plan is at its weakest — training-mode dropout would raise
slot gradients), bf16 autocast numerics, and gradient NORMS say nothing about whether
the surviving slot gradient direction is useful.
