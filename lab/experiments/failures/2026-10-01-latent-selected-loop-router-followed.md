# Planned: the latent-selected loop with the router driving the forward at train

Status: failure

Date: 2026-10-01 10:17 CDT (frozen before either run trained).
Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf` and
`tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf`. Each is its parent
(`..._lsel_joint` / `..._lsel_det`) plus one key, `tul.fan_lsel_train_follow: router`.
Parent prereg: [`2026-09-30-latent-selected-loop.md`](2026-09-30-latent-selected-loop.md).
Wolfe (2026-10-01): "Yes fix it and run it."

## Question

In the parents, the loop followed the latent teacher at train. The teacher picks with the
next span's latent, which is built from the tokens the coda must predict, so the coda
trained on cells chosen with hindsight and deployed on the router's picks. On the same 480
rows the joint parent read 4.372 nats when it followed the teacher and 4.795 when it
followed the router. Zeroing its cell improved the router path by 0.17, and its
loop contribution was -0.229. If the forward follows the router at train too, with the
teacher only as the router's label and the exit loss's winner, does the cell become useful
to the coda and does the loop earn depth?

## Hypothesis

The train and deploy forwards are now the same, so the collapse goes: the cell helps
again and the CE comes back near the ungraded fan, near the reader-trained router's cost
of reading one cell (+0.011). Whether the per-pass search adds depth on top is open.

## References (seed 1, 5000 steps; CE on the ledger's 480 rows)

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth |
| --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | |
| reader-trained router | 9718 | 4.345 | +0.265 | +0.0068 | |
| latent-selected, joint (teacher-followed) | 9566 | 4.795 | +0.715 | -0.229 | -0.173 |
| latent-selected, detached (teacher-followed) | 9687 | 4.747 | +0.666 | -0.170 | -0.133 |

## Predictions (mine, orchestrator, before the runs)

Joint, router-followed:

- **R-1**: no detonation. 85 %.
- **R-2**: ledger shipped CE vs the ungraded fan's 4.334: within 0.03: 50 %; worse by more
  than 0.03: 40 %; better by more than 0.03: 10 %.
- **R-3**: val `fan/lsel_teacher_pick_gap` below 0.10 (the hindsight gap closes). 70 %.
- **R-4**: worth-profile zero TOTAL > 0 (the cell helps the coda). 75 %.
- **R-5**: K1-K6 > 0: 50 %. K1-K6 > +0.0092: 25 %.
- **R-6**: val `fan_lsel_r2` > 0.1. 40 %.
- **R-7**: val `fan_lsel_teacher_router_agree` > 0.40. 40 %.
- **R-8**: tok/s within 5 % of the ungraded fan (9797). 80 %.
- **R-9**: ledger `random_exit` > 0 with its CI above 0: 60 %. Ledger `no_reset` > 0 with its
  CI above 0: 50 %.

Detached, router-followed:

- **RD-1**: no detonation. 85 %.
- **RD-2**: ledger shipped CE worse than the joint router-followed arm by more than 0.0137. 55 %.
- **RD-3**: worth-profile zero TOTAL > 0. 60 %.

Pass, joint: R-4 AND K1-K6 > 0 AND ledger CE within 0.03 of the ungraded fan.
Pass, detached: RD-3 AND ledger CE within 0.03 of the ungraded fan.

## Method

- Seed 1, 5000 steps each, `runner_steps.sh` at the commit of this prereg, joint first.
  Readouts as for every fan arm (depth sweep, gap to plain, worth profile). The stage-2
  score and its pair step fail on every fan arm (no parallel head) and are ignored.
- Then the exploration ledger (wt-xled, 1ab4c2e) on both arms, and the repetition eval
  (N=32, 128 tokens, router path) on both arms with the same prompts as the baselines.
- Smoke before queueing: 45 steps of each config at this commit on the 5090.

## Results

Both arms ran 5000 steps, seed 1, joint first. No detonation (`chain.log` TRIPWIRE HEALTHY,
max preclip/total 125 joint, 50.7 detached). The naming in this section: joint
router-followed and detached router-followed are the two arms of this prereg; the
teacher-followed arms from the parent prereg (plain "joint"/"detached" below) are the
reference row.

| reading | joint router-followed | detached router-followed |
| --- | --- | --- |
| tok/s | 9569 | 9566 |
| K1-K6 | +0.0067 [+0.0062, +0.0072] | +0.0095 [+0.0086, +0.0104] |
| gap to plain | +0.4102 [+0.3939, +0.4286] | +0.4091 [+0.3930, +0.4278] |
| ledger shipped CE | 4.4908 | 4.4894 |
| ledger worth, zero TOTAL | +0.1058 | +0.1016 |
| `val/fan_lsel_teacher_pick_gap` | -0.0057 | -0.0034 |
| `val/fan_lsel_r2` | -0.0273 | -0.0039 |
| `val/fan_lsel_teacher_router_agree` | 0.3837 | 0.3956 |
| `val/fan_lsel_cell_spread` | 0.2225 | 0.2128 |
| `fan/lsel_switch_rate` | 0.1936 | 0.3141 |
| ledger `random_exit` | +0.0096 [+0.0090, +0.0101] | +0.0113 [+0.0106, +0.0120] |
| ledger `random_search` | +0.0135 [+0.0128, +0.0143] | +0.0163 [+0.0156, +0.0171] |
| ledger `fixed_lineage` | +0.0576 [+0.0558, +0.0594] | +0.0308 [+0.0294, +0.0322] |
| ledger `no_reset` | +0.0005 [+0.0003, +0.0007] | +0.0071 [+0.0066, +0.0076] |
| ledger `teacher` | +0.0048 [+0.0043, +0.0053] | +0.0038 [+0.0033, +0.0043] |
| ledger `oracle` | -0.0231 [-0.0237, -0.0226] | -0.0224 [-0.0229, -0.0219] |
| ledger `loser_t0` | +0.0035 [+0.0032, +0.0039] | +0.0072 [+0.0067, +0.0077] |

Reference: ungraded fan tok/s 9797, ledger CE 4.3343, K1-K6 +0.0042. Reader-trained router
tok/s 9718, ledger CE 4.3452, K1-K6 +0.0068.

| prediction | reading | held |
| --- | --- | --- |
| R-1 no detonation | HEALTHY | yes |
| R-2 joint ledger CE vs 4.334: worse by > 0.03 | 4.4908, +0.1565 | yes |
| R-3 joint `teacher_pick_gap` below 0.10 | -0.0057 | yes |
| R-4 joint worth zero TOTAL > 0 | +0.1058 | yes |
| R-5 joint K1-K6 > 0 | +0.0067 | yes |
| R-5 joint K1-K6 > +0.0092 | +0.0067 | no |
| R-6 joint `fan_lsel_r2` > 0.1 | -0.0273 | no |
| R-7 joint `teacher_router_agree` > 0.40 | 0.3837 | no |
| R-8 joint tok/s within 5% of 9797 | 9569, 2.3% off | yes |
| R-9 joint ledger `random_exit` > 0, CI above 0 | +0.0096 [+0.0090, +0.0101] | yes |
| R-9 joint ledger `no_reset` > 0, CI above 0 | +0.0005 [+0.0003, +0.0007] | yes |
| RD-1 no detonation | HEALTHY | yes |
| RD-2 detached ledger CE worse than joint by > 0.0137 | 4.4894 vs 4.4908, detached is BETTER | no |
| RD-3 detached worth zero TOTAL > 0 | +0.1016 | yes |

## Verdict

Failure for both arms. Pass for joint router-followed needs the worth-profile TOTAL above 0,
K1-K6 above 0, and ledger CE within 0.03 of the ungraded fan's 4.334; it clears the first two
and misses the third by more than 5x (+0.1565 against the 0.03 bar). Pass for detached
router-followed needs the same worth condition and the same CE bar; it clears the worth
condition and misses the CE bar by the same margin (+0.1551). So the hypothesis in this
prereg's own title is half right: making train follow the router instead of the teacher DOES
fix the collapse the parent prereg found. The cell helps again (worth +0.106 joint, +0.102
detached, versus -0.173 and -0.133 when the forward followed the teacher), the loop earns
real depth (K1-K6 +0.0067 to +0.0095, in the same range as the reader-trained router's
+0.0068, not the deeply negative numbers from the teacher-followed arms), and the ledger's
`no_reset` and `random_exit` ablations both read small and positive, meaning the per-pass
search and the reset-to-winner mechanics are doing something, not nothing. But CE did not
come back to the ungraded fan's level the way the hypothesis predicted: both arms sit 0.41-
0.42 nats above plain, about 0.155 nats worse than the ungraded fan's own 0.254 gap, roughly
14x the reader-trained router's extra cost of reading one cell instead of four (+0.011).
Router agreement with the teacher (0.384 joint, 0.396 detached, chance 0.25) misses this
prereg's 0.40 bar, but it does not explain the CE gap by itself: following the teacher at
eval is now WORSE for the coda than following the router (ledger `teacher` +0.0048 / +0.0038),
so a router that matched the teacher would not recover CE. The CE gap has two open causes
that this run cannot separate (orchestrator's interpretation, 2026-10-02): the weight-10
latent loss, which nothing learns (R^2 at or below 0), and the single-cell read.

## Updated hypothesis

The train/deploy mismatch from the parent prereg is fixed (the cell is useful again, the
loop earns depth), but the latent target itself is not learned: the latent head's
`fan_lsel_r2` sits at or below zero in all four latent-selected arms (joint, detached, and
both router-followed). The router learns from the teacher's picks, not from that R^2, and
the ledger shows its picks have value (+0.010 to +0.011 over a random pick). The 10x factor fan filing found the same pattern independently:
raising a latent term's weight to compete with the token CE does not make it learn faster,
it crowds out the tokens. The open causes worth separating are the latent loss weight itself
(currently 10x the token CE here too) and the single-cell read (the ungraded fan's own
all-cell read is worth +0.17 on its ledger, so reading only the router's winner may be
costing more than the selection mechanism gains). Each gets its own one-key arm in
[`../planned/2026-10-01-latent-selected-loop-weight1-and-read-all.md`](../planned/2026-10-01-latent-selected-loop-weight1-and-read-all.md),
running as of this writing.
