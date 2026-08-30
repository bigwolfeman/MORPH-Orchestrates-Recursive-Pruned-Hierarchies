# Planned: cond-zero probe — is the AdaLN shortcut load-bearing at eval, or did training reorganize the weights?

Status: planned
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
