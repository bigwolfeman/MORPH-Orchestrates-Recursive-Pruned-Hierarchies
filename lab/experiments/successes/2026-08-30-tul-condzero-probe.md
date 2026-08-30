# Planned: cond-zero probe — is the AdaLN shortcut load-bearing at eval, or did training reorganize the weights?

Status: success
Date: 2026-08-30, frozen before the run. Follows the ILV pair filing
(successes/2026-08-30-tul-ilv50-l2capcond.md), which queued this probe in its
Updated hypothesis. No training — one 48-row forced-depth sweep on a modified
copy of `checkpoints/morph/tul-l2cap-cond/step_4500.pt` with every
`tul_stage_cond.gates.*.to_mod.{weight,bias}` zeroed (AdaLN-Zero: zero to_mod ⇒
zero modulation ⇒ bit-identical to an unconditioned forward, proven by
tests/test_tul_dbfix.py::test_conditioning_zero_init_bit_identical).

## Question

l2cap-cond (exact l2cap recipe + iter-AdaLN) earned 0.0127 nats of depth vs
l2cap's 0.233. Two stories fit: (a) the conditioning shortcut is load-bearing at
EVAL — the composition was learned underneath and stripping the module reveals
it; (b) training WITH the shortcut reorganized the loop weights into a
non-composing solution — the damage is in the weights and stripping the crutch
restores nothing.

## Predictions (frozen)

- **P1 (binding).** Depth-earned CE (K=1 − K=6, forced-depth sweep, same 48-row
  instrument) on the cond-zeroed model stays < 0.05 nats: 75%. (Revival ≥ 0.10
  would flip to story (a); the 0.05–0.10 band is partial revival, read toward
  (a) with a caveat.)
- **P2.** Absolute CE at K=6 on the cond-zeroed model degrades vs the intact
  model's 4.4165 by ≥ 0.05 nats (the model leans on the modulation it trained
  with): 60%.

## Method

1. Copy the checkpoint; zero the 12 `to_mod` tensors (6 gates × weight+bias);
   leave `embed.mlp.*` untouched (its output multiplies into zeros).
2. Run core_depth_sweep (auto mode = forced-depth for an iter model, 48 rows)
   on the modified copy with the unchanged `tul_l2cap_cond` config.
3. Compare against the intact sweep already filed
   (results/2026-08-30-tul-ilv50-l2capcond/depth_sweep_tul-l2cap-cond_auto.json).

## Not verified before launch

That zeroing to_mod on THIS trained checkpoint reproduces the unconditioned
forward exactly (the bit-identity test proves it at init; the module keeps the
same functional form, so it should hold, but the sweep itself is the check).

## Results

Same 48-row instrument, same eval rows as the intact sweep.

| K | intact CE | cond-zero CE |
|---|---|---|
| 1 | 4.4292 | 4.4303 |
| 6 | 4.4165 | 4.4175 |
| earned (K1−K6) | 0.0127 | 0.0128 |

- **P1 PASS** (bar < 0.05): earned 0.0128 — the curve is unchanged. No hidden
  composition; story (b) holds.
- **P2 FAIL** (bar ≥ 0.05 degradation at K=6): degradation is ~0.001 nats. The
  trained model barely reads the conditioning at eval at all.
- Artifact: `lab/experiments/results/2026-08-30-tul-ilv50-l2capcond/depth_sweep_condzero.json`
  (filed with the parent pair's results).

## Verdict

The binding prediction held (P1, 75%); P2 (60%) missed informatively. The
conditioning module is nearly functionally inert in the trained model — zeroing
it costs ~0.001 nats — yet its presence during training collapsed the depth
curve from 0.233 to 0.013. The damage is a TRAINING-DYNAMICS effect: the
shortcut steered optimization into a non-composing basin and was then largely
abandoned by the network. There is no "train with cond, strip at inference"
trick to salvage.

## Updated hypothesis

Iteration-differentiating side-channels do their harm during formation, not at
inference. For the gate-vs-cap ladder this sharpens the constraint: it is not
enough that a gate is cheap to remove later — it must never give the optimizer
an index-keyed way to differentiate iterations DURING training. State-keyed
gates remain admissible; anything keyed on the iteration counter (embeddings,
schedules, per-iteration learned scalars) is presumptively poisonous to
depth-earning.
