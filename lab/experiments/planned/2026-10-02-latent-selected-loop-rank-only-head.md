# Planned: a latent head that ranks the cells but does not pull them

Status: planned

Date: 2026-10-02 10:47 CDT (frozen before either run trained).
Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1` (the latent loss at weight
1 pulls the cells, joint) and `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank`
(the same, the latent head reads DETACHED cells). The pair differs in one key,
`tul.fan_lsel_head_input`.
Parent: [`2026-10-01-latent-selected-loop-weight1-and-read-all.md`](../failures/2026-10-01-latent-selected-loop-weight1-and-read-all.md).
Wolfe (2026-10-02): "I want to try that arm."

## Question

In all six latent-selected arms the latent head never learned its target (R^2 at or below
0.01), yet the selection worked: the router's pick, the search and the lineage all have
value on the ledger. Cutting the latent weight from 10 to 1 improved CE by 0.042 and
tripled the loop's K1-K6 (+0.0095 -> +0.0333, detached). So the loss that PULLS the cells
toward the target looks like a tax. If the head only learns to RANK the cells (it reads
them detached), the teacher still works and the cells pay nothing. Does CE improve, and
does the loop keep earning? The pair runs JOINT: a detached loop with a rank-only head
would get no task objective (TULConfig refuses it).

## Hypothesis

The rank-only head keeps the selection (the teacher ranks as well as before, since it
never learned the target anyway) and removes the tax, so CE improves over the pull twin.
The loop's K1-K6 stays positive: in the joint arm the token CE trains the loop through the
winner.

## References (seed 1, 5000 steps, ledger 480 rows)

| arm | tok/s | ledger CE | gap to plain | K1-K6 |
| --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 |
| router-followed joint, weight 10 | 9569 | 4.491 | +0.410 | +0.0067 |
| router-followed detached, weight 1 | 9818 | 4.447 | +0.366 | +0.0333 |

## Predictions (mine, orchestrator, before the runs)

Pull twin (joint, weight 1):

- **P-1**: no detonation. 85 %.
- **P-2**: ledger CE better than the weight-10 joint arm's 4.491 by more than 0.03. 60 %.
- **P-3**: K1-K6 above +0.0095. 55 %.

Rank-only head:

- **K-1**: no detonation. 85 %.
- **K-2**: ledger CE vs the pull twin: better by more than 0.0137: 50 %; within 0.0137: 40 %;
  worse by more than 0.0137: 10 %.
- **K-3**: ledger CE within 0.03 of the ungraded fan's 4.334. 25 %.
- **K-4**: K1-K6 > 0: 70 %. Above the pull twin's: 35 %.
- **K-5**: ledger random exit pick > 0 with its CI above 0 (selection works with no pull). 70 %.
- **K-6**: val teacher-router agreement > 0.35. 60 %.
- **K-7**: tok/s within 5 % of 9797. 85 %.
- **K-8**: worth-profile zero TOTAL > 0. 85 %.

Pass, rank-only head: ledger CE better than the pull twin by more than 0.0137 AND K1-K6 > 0
AND K-5.

## Method

- Seed 1, 5000 steps each, `runner_steps.sh` at the commit of this prereg, pull twin
  first. Readouts as for every fan arm. Then the exploration ledger and the repetition eval
  (N=32, 128 tokens) on both.
- Smokes: 45 steps of each config at this commit on the 5090 before queueing.
