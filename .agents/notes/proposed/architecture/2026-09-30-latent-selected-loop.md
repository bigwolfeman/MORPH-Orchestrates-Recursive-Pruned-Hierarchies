# Agent Note: the latent-selected loop (selection inside the slot loop, trained by a latent target)

Status: proposed

## Problem

The latent-teacher router ([2026-09-30-fan-opf-and-routers.md](2026-09-30-fan-opf-and-routers.md),
`tul.fan_route: latent`) failed at 5k on seed 1: its teacher picked the coda's best cell at
chance. The cause is in its wiring, not in the idea of a latent teacher. Its map `g` read the
cells DETACHED, and only the coda's token CE trained the cells. So nothing ever pulled any cell
toward the latent target, and "the cell closest to the target" carried no information about
which cell the coda could use. The selection also happened once, after the loop, on cells that
had looped independently.

Wolfe's design (2026-09-30): move the selection INTO the loop and let the latent objective
train the loop.

## Proposal

One key, `tul.fan_loop_select: off | joint | detached` (default `off`), on the ungraded fan
(`tul_slot_spandec_strict_fan4_all_fp01_nowta.yaml`: write-all fan, M = 4 cells, no WTA term,
epivol 0.1, fixed-point term 0.1). Code: `morph/model/tul_fan_route.py` (`FanLatentHead`,
`lsel_distance`, `lsel_exit_loss`, `reset_to_winner`, `cell_spread`) and
`MORPHTransformer._lsel_begin / _lsel_pass / _lsel_finish`, called from `_tul_core`.

1. **Target.** The factor fan's: the EMA prelude twin (`FanTargetFront`, momentum
   `tul.fan_target_ema` 0.996) pooled over span s+1's tokens, LayerNorm without affine
   (`_tul_fan_target`). Under strict geometry the target of slot s depends on span s+1 alone
   (pinned). The twin's whole lifecycle (build after quantisation, sync after load, EMA after
   each step, checkpoint key `fan_target`, lab loader) is reused unchanged:
   `_fan_target_needed` now includes this key.
2. **Latent head g.** LayerNorm (no affine), `Linear(d, d)`, GELU, `Linear(d, d)`, one head
   shared by the M cells (`tul.fan_lsel_hidden`, 0 = d). WITH gradient into the cells at the
   exit: this head is how the latent target trains the loop.
3. **Selection every pass.** After pass t (inside `_tul_core`'s per-slot depth loop, after the
   trajectory append) the distances `d_i = mean_c (g(c_i) - z)^2` are computed under
   `no_grad`. At train the winner is the teacher's argmin on the slots with a target; slots
   with no next span, every eval forward and deploy follow the router. Then every cell of a
   slot whose depth continues (`depth > t + 1`) is set to the winner's state
   (`reset_to_winner`, a gather on the cell axis, so BPTT flows through the winner's path).
   A slot that finishes at pass t is not reset; a slot that stopped earlier and every pad are
   untouched.
4. **How the cells stay distinct after a reset.** All M cells of a slot enter pass t + 1 with
   the same carrier, but the pass is not the same function for each cell: (a) each cell's
   per-pass injection source `e_i` is the prelude output at its OWN prefix position plus its
   own register term (`TULSlotRegister`, `W_o(pooled_i) + P_cell[i]`), and
   `DiagonalInjection` re-adds `dt * e_i` on the context channels at every pass; (b) each cell
   sits at its own position of the compact core sequence, so attention and CoPE see M
   different positions. No learned offset is added after the reset. The test
   `test_the_cells_stay_distinct_after_a_reset` reads the cells one pass after a reset at the
   fresh init (register `W_o` and `P_cell` both zero, the weakest case) and finds every pair
   distinct; the run logs `fan/lsel_cell_spread` (exit cells of slots that were reset at
   least once; 0 = M copies).
5. **Loss only at the exit.** The relaxed WTA on the FINAL pass's cells:
   `(1 - eps) d_win + eps / (M - 1) sum_{i != win} d_i`, eps `tul.fan_lsel_eps` 0.05, the
   teacher's exit argmin as the winner, averaged over slots with a target, weight
   `tul.fan_lsel_lambda` 10. No per-pass latent term (this repo measured per-pass targets
   being met in one step with passes 2+ idle). Plus the online floor `L_enc`: the hinge
   `max(0, 0.1 - std)` per coordinate of the ONLINE prelude's pooled span states, weight
   0.2 (the factor fan's ratio at 10x). Folded as `fan_lsel_weighted`; train.py subtracts it
   at train and at val.
6. **Router (the deploy path).** `FanRouter` (reused; detached cells, detached `ctx`), trained
   by cross-entropy onto the teacher's pick at every (pass, slot) where a teacher exists,
   weight `tul.fan_lsel_router_lambda` 1, inside `fan_lsel_weighted`. It sends no gradient into
   the cells or the loop. No balance bias is stepped (the bias buffer stays 0); add it only if
   the router collapses.
7. **Coda read.** The final winner alone, hard, through `_fan_route_cells` (losers exactly zero,
   no p). The final pass's losers stay readable by LATER slots' loops. Earlier passes' losers
   are not kept: the reset overwrites them, and they reach nothing after their pass except the
   diversity term (which reads the trajectory, where the pre-reset candidates are recorded).
8. **Two variants.** `joint`: the coda's token CE reaches the loop through the winner cell.
   `detached`: ONE cut, `h_slots.detach()` right after `_tul_core` returns, upstream of every
   reader of the loop's output (the coda's write, the span decoder's mean of the cells and its
   `cells=` memory, the val oracle). The loop then trains only from the exit latent loss, the
   fixed-point term, the epivol diversity term and the gain hinges, all built inside
   `_tul_core` on the live states. Every per-pass token target that reads the trajectory is
   refused under `detached` (MUX, oracle_z, spandec_per_pass, horizon, grad_pass,
   code_target). Pinned: with every loop term at weight 0, the token CE leaves every core,
   injection, register and core-x0 gradient exactly zero under `detached` and nonzero under
   `joint`.
9. **Refused:** `fan_opf`, `fan_route != none`, `fan_all_wta_lambda > 0`, `code_enum_k > 1`,
   `fan_mix != all`, `bcast`, the halting gate, `loop_denoise`, `prefix_source != exit`,
   `pass_readout != last`, `fan_history_streams`, `fan_lineage`, `loop_carry`, `n_core = 0`,
   and the DiffusionBlocks one-pass / ladder paths.
10. **Instruments** (`fan/lsel_*` at train, `val/fan_lsel_*` at val): `lat`, `enc`,
    `router_ce`, `r2` (R^2 of g at the followed winner against z, 0 = batch mean, 1 =
    perfect), `teacher_router_agree` and `_t{t}` (chance 1/M), `exit_teacher_router_agree`,
    `switch_rate` (the followed winner changes between consecutive passes), `share_k{i}`,
    `cell_spread`, `enc_std_min`, `target_rank`, `win_dist`, `mean_dist`. At val only:
    `fan/lsel_teacher_pick_ce` (one extra no-grad forward with the loop following the teacher,
    `tul_forward_ablated(lsel_follow="teacher")`) and `fan/lsel_teacher_pick_gap` = router-path
    CE minus teacher-path CE, what a perfect router would buy. The val oracle prices the final
    winner (`fan/router_coda_agree`, `fan/router_pick_regret`, `fan/teacher_coda_agree`).

Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint.yaml` (latent-selected loop,
joint) and `tul_slot_spandec_strict_fan4_all_fp01_lsel_det.yaml` (latent-selected loop,
detached), each one key from the ungraded fan.

## Alternatives considered

* **Keep the latent-teacher router and un-detach g.** Its g would then pull the exit cells
  toward z, but selection would still happen once, after M independent loops. The loop would
  never be asked to refine a chosen hypothesis, which is the point of moving selection in.
* **Per-pass latent losses.** Rejected from measurement: seven per-pass targets in this repo
  were all met by pass 1, and passes 2+ went idle (`per-pass-targets-met-in-one-step`).
* **Soft selection (a softmax mixture of the cells as the next pass's input).** Keeps
  gradient to every cell, but the M cells then converge to the mixture and the fan collapses
  (the Thought Register read rank 1.24 of 4 without pressure). A hard reset keeps one
  committed hypothesis per pass.
* **A learned per-cell offset after the reset.** Not added: the cells already differ through
  their own injection source, register term and core position (item 4), and an extra offset
  is a second factor. It stays the fallback if `fan/lsel_cell_spread` reads near 0.
* **Router balance bias from step 0.** Not added: the router is trained by CE onto the
  teacher, so its load follows the teacher's. If the teacher collapses to one cell, a bias on
  the router would hide that instead of fixing it.

## Acceptance criteria

* Every key at its default is the tree: the ungraded fan's pins (measured before the route
  keys existed) and the factor fan / both router arms' pins measured on the unmodified tree at
  213b585 hold (`tests/test_tul_fan_lsel.py`).
* Both arms run 5000 steps with finite losses and `train/loss_total - train/loss` equal to the
  weighted aux terms.
* The arm is read against the ungraded fan on val token CE (the ROUTER path), on
  `fan/lsel_teacher_pick_gap`, on the K-curve (does the loop use depth), and on
  `fan/teacher_coda_agree` above chance (is the latent pick the coda's best cell, which the
  latent-teacher router failed).
* A prereg in `docs/experiments/planned/` with predictions before either run.

## Risks

* **The latent target collapses.** The online floor guards the online prelude only; read
  `fan/lsel_target_rank` and `fan/lsel_enc_std_min` (LCTUL-J lost rank 77 to 40).
* **Train/deploy gap.** Train follows the teacher, eval follows the router. If the router
  learns the teacher poorly, the val CE pays for it; `fan/lsel_teacher_pick_gap` measures
  exactly that price.
* **Weak per-cell variation.** If the four post-reset variations differ only in the 320
  context channels' injection, the fan explores little. `fan/lsel_cell_spread` and
  `fan/lsel_switch_rate` (0 = the first pass's pick is never revised) are the readings.
* **The latent loss at weight 10 dominates the joint arm's loop gradient.** The token CE is
  about 4.4 nats; the exit term starts near 1 per coordinate times 10.
* **The detached arm's loop has no token signal at all.** If the latent target carries little
  of what the coda needs, the detached arm will trail, and that is a finding, not a bug.
* **Cost.** Per pass: one router score and one no-grad g over B x S x M cells; at the exit one
  graded g; at val one extra forward for the teacher-pick CE. The 41-step smoke measures the
  tok/s against the ungraded fan.
