# Planned: the latent-selected loop with the router driving the forward at train

Status: planned

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
