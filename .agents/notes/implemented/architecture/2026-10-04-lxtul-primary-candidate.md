# Agent Note: LXTUL primary candidate, the rank-only latent head with a per-pass cell norm

Status: implemented

## Problem

Since 2026-08-16 the slot loop has had one recurring failure: its passes do not add to the
model ("loop contribution"), or they add only through a channel that is not content. The
ungraded fan, the cheap base of the LX family, earns K1-K6 +0.0042 on the 480 sweep rows.
Every attempt to force more out of it either cost CE or was gamed
([campaign history](../../../../docs/tul-loop-contribution-history.md)). Wolfe's goal is a
recipe whose loop earns depth reliably, at a speed it can be iterated on, so that CE and
tok/s can then be refined.

## Decision

The primary LXTUL candidate is `morph/configs/lxtul.yaml`. It composes byte-identically to
`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm_10k` (only `wandb.name`
differs). In words:

- the strict slot loop (the loop is the only cross-span channel; a coda token reads its
  own span plus every earlier slot's prefix cells);
- a 4-cell fan (`fan_mix: all`), epivol diversity on passes 1-2, no winner-take-all. Under
  the latent-selected loop (`fan_lsel_read: winner`, the default) the coda reads the FINAL
  WINNER alone: the winner goes into its own prefix position through `W_prefix[winner]` and
  the three losers' positions are exactly zero (`_fan_route_cells`). Corrected 2026-10-05 01:02 CDT: this
  record first said every cell is written to the coda;
- the fixed-point term at 0.1;
- the latent-selected loop: after every pass a router picks one of the 4 cells and the
  slot's cells reset to it (`fan_loop_select: joint`, `fan_lsel_train_follow: router`); the
  router is trained toward the cell nearest the next span's EMA-prelude latent;
- the latent head is rank-only (`fan_lsel_head_input: detached`, weight 1): it labels the
  router and sends no latent gradient into the loop;
- a per-pass RMSNorm on every slot cell, one learned gain shared by every pass
  (`tul.slot_cell_pass_norm: rms`);
- the training-speed recipe that is byte-identical to master by gate:
  `training.ademamix_fused_fp32: true`, `model.ckpt_grad_iters: 4`.

Evidence, 480 rows, paired CIs:

| run | K1-K6 | ledger CE | gap to plain at the same step | tok/s |
| --- | --- | --- | --- | --- |
| 5k seed 1 | +0.0215 [+0.0205, +0.0225] | 4.353 | +0.272 | 9800 |
| 5k seed 2 | +0.0172 [+0.0163, +0.0181] | 4.362 | +0.282 | 8876 |
| 10k seed 1 (speed recipe) | +0.0236 [+0.0225, +0.0246] | 4.0975 | +0.356 | 12402 |
| ungraded fan, 5k (reference) | +0.0042 | 4.334 | +0.254 | 9797 |

On all three runs the depth earning is directional: setting the cells' shared-direction
amplitude to its mean moves K6-K1 by 1 % to 7 %. Without the norm the same arm's K1-K6
(+0.0052) was one magnitude along that direction (mean-ablation erased it). Filings:
[pair](../../../../lab/experiments/failures/2026-10-03-slot-cell-pass-norm-pair.md),
[seed twin](../../../../lab/experiments/successes/2026-10-03-rank-only-cell-norm-seed-twin.md),
[10k](../../../../lab/experiments/successes/2026-10-04-rank-only-cell-norm-10k.md).
The model-side reasons: Huginn and HRM-Text both normalise their carried state every step
([Huginn](../../../../lab/experiments/successes/2026-10-02-huginn-loop-geometry.md),
[HRM-Text](../../../../lab/experiments/failures/2026-10-03-hrm-text-loop-geometry.md)), and
a latent pull on the loop collapses the cells onto one direction under any norm
([read norm](../../../../lab/experiments/failures/2026-10-03-slot-cell-read-norm.md),
[LayerNorm note](../../proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)).

## Alternatives considered

- **Detached latent pull at weight 1** (no norm: K1-K6 +0.0333, CE 4.447). The highest
  K1-K6 measured, but its cells grow 4x along one shared bias at pass 2, and under any norm
  they rotate onto it (cos 0.96); with the per-stream norm its K1-K6 fell to +0.0119.
- **Ungraded fan** (K1-K6 +0.0042, CE 4.334). Better CE by 0.02-0.03 at 5k, but its loop
  contributes about a fifth as much.
- **Coda-graded fan** (CE 4.298 at 5k). Best CE of the family, at 0.78x plain speed.
- **Four-rollout ensemble.** 6151 tok/s; its width is an ensemble gain, not loop depth.
- **The non-bit-identical speed keys** (`compile_blocks`, `compile_core_dynamic: false`,
  `ce_softmax_kernel`, `ce_compact_rows`: 388 ms/step, 1.60x). Left off: only 300-step
  overlays support their neutrality.

## Consequences

- New work on CE and tok/s starts from `lxtul.yaml`, and new arms compose it.
- `base.yaml` is NOT changed: it still ships the paid loop. Removing the paid loop from main
  is Wolfe's earlier stated plan once the slot TUL is final; that is a separate change.
- Open problems, measured: the gap to a plain model at the same step widens with training
  (+0.272 at 5k, +0.356 at 10k), and repetition is unchanged (greedy rep-4 overlaps every
  arm). The fast decode path (`tul_generate_cached`, `_graphed`) refuses `fan_k > 0`, so
  TUL's decode speed is not yet measurable on this arm.
- Training speed: [2026-10-04 speed note](../../proposed/architecture/2026-10-04-slot-loop-training-speed.md).
