# Planned: a latent head that ranks the cells but does not pull them

Status: failure

Date: 2026-10-02 10:47 CDT (frozen before either run trained).
Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1` (the latent loss at weight
1 pulls the cells, joint) and `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank`
(the same, the latent head reads DETACHED cells). The pair differs in one key,
`tul.fan_lsel_head_input`.
Parent: [`2026-10-01-latent-selected-loop-weight1-and-read-all.md`](2026-10-01-latent-selected-loop-weight1-and-read-all.md).
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

## Results

Both runs finished 5000 steps at f3893cc, seed 1, pull twin first. Both tripwires read
HEALTHY. Artifacts: [`../results/2026-10-02-latent-selected-loop-rank-only-head/`](../results/2026-10-02-latent-selected-loop-rank-only-head/).
CE is the ledger's shipped CE on the 480 sweep rows; tok/s is the mean of the logged steps
after the first two.

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth | greedy rep-4 |
| --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | | 0.523 [0.450, 0.597] |
| router-followed joint, weight 10 | 9569 | 4.491 | +0.410 | +0.0067 | +0.106 | 0.595 |
| pull twin (joint, weight 1) | 9621 | 4.387 | +0.307 [+0.291, +0.324] | +0.0096 [+0.0089, +0.0103] | +0.159 | 0.522 [0.462, 0.588] |
| rank-only head | 9635 | 4.380 | +0.299 [+0.284, +0.316] | +0.0052 [+0.0047, +0.0058] | +0.166 | 0.559 [0.493, 0.622] |

CE at each forced depth: pull twin 4.3964 / 4.3916 / 4.3877 / 4.3869 / 4.3867 / 4.3868;
rank-only 4.3851 / 4.3825 / 4.3801 / 4.3796 / 4.3797 / 4.3799 (K1..K6).

Ledger deltas (nats per token, 95 % CI): pull twin / rank-only. Random exit pick +0.0058 /
+0.0039; random walk +0.0127 / +0.0100; teacher +0.0045 / +0.0039; no reset +0.0015 /
+0.0030; fixed lineage +0.0074 / +0.0225; loser at pass 0 +0.0028 / +0.0043; single cell
+0.0116 / +0.0056; best-of-4 oracle -0.0243 / -0.0207. Every CI excludes 0 except the
losers at passes 2 and 3 (and 4 for the rank-only head).

Val at step 5000, pull twin / rank-only: latent R^2 +0.0004 / -0.0092, teacher-router
agreement 0.363 / 0.423, cell spread 0.192 / 0.170, switch rate 0.206 / 0.310, teacher pick
gap -0.0025 / -0.0029.

| prediction | reading | held |
| --- | --- | --- |
| P-1 no detonation | HEALTHY | yes |
| P-2 better than 4.491 by > 0.03 | better by 0.104 | yes |
| P-3 K1-K6 > +0.0095 | +0.0096 | yes |
| K-1 no detonation | HEALTHY | yes |
| K-2 vs the pull twin | better by 0.0069 (the "within" bin, 40 %) | within |
| K-3 within 0.03 of 4.334 | +0.046 | no |
| K-4 K1-K6 > 0 / above the pull twin | +0.0052 / below | yes / no |
| K-5 random exit pick CI above 0 | +0.0039 [+0.0034, +0.0043] | yes |
| K-6 agreement > 0.35 | 0.423 | yes |
| K-7 tok/s within 5 % of 9797 | 9635 | yes |
| K-8 zero worth > 0 | +0.166 | yes |

## Verdict

Failure: the pass rule needs the rank-only head better than the pull twin by more than
0.0137, and it is better by 0.0069.

The large result is the pull twin, not the head. Joint with the latent weight at 1 instead
of 10 improved CE by 0.104 (4.491 -> 4.387) and kept the loop earning (+0.0096). The
rank-only head adds a little more CE (0.007, inside the bar), the best router agreement of
any arm (0.423), and the best gap to plain of any latent-selected arm (+0.299), at about
half the pull twin's K1-K6. So at weight 1 the pull is a small tax; at weight 10 it was the
main one. The selection still has value with no pull at all (random pick +0.0039, random
walk +0.0100, fixed lineage +0.0225).

## Updated hypothesis

The latent loss's weight was the tax; the pull itself costs little at weight 1. Both arms
still trail the ungraded fan by about 0.05 on CE. A measured cause found during this run
(stage-1 build, 2026-10-02): the LayerNormed target is about 98 % one common direction
(mean cosine 0.98, per-coordinate variance about 0.02), so the teacher's argmin mostly
ranks how well each cell matches that constant, not span content. Next: the same rank-only
arm with a standardised target (per-coordinate (z - mu) / sigma, running stats for the EMA
twin). If the common mode was blinding the teacher, router agreement and the pick's ledger
value should rise.
