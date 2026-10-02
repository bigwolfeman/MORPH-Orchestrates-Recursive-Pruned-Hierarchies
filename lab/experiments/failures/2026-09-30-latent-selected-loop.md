# Planned: does selecting inside the loop on the next span's latent make the fan's loop useful?

Status: failure

Date: 2026-09-30 23:12 CDT (frozen before either run trained).
Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint` (latent-selected loop, joint) and
`tul_slot_spandec_strict_fan4_all_fp01_lsel_det` (latent-selected loop, detached). Each is the
ungraded fan plus one key, `tul.fan_loop_select`.
Design note: [`2026-09-30-latent-selected-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-30-latent-selected-loop.md).
Wolfe (2026-09-30): "OR we do as I said and we do it at the loop not the coda output?" and
"token gradients pass through both coda and loop vs only coda and loop gets its signal from
its objective."

## Question

The latent-teacher router failed: it ranked the exit cells by a latent the cells were never
trained toward, and it was judged against the coda's best cell. Wolfe's design moves the
selection INTO the loop. After every pass, a teacher picks the cell whose latent head output
is nearest the next span's EMA-prelude latent. All four cells restart from that winner, so
the next pass explores four variations around it. One latent loss at the exit trains the
cells toward the target. A router learns the teacher's pick and drives selection at eval and
in generation. The coda reads only the final winner.

Two variants. In the joint variant, the coda's token CE also reaches the loop through the
winner. In the detached variant, the coda and span decoder read the loop's output detached,
so the loop learns only from the latent loss (plus the fixed-point, diversity and gain terms).

Does a loop that searches toward a latent target give the coda a better cell than the
ungraded fan's four unguided cells? Is the latent objective alone enough to train the loop?

## Hypothesis

The selection works mechanically: the cells learn the latent, the router learns the teacher,
and the cells stay distinct after each reset. The coda pays for reading one cell instead of
four (both routers paid 0.011 to 0.013 for zeroing three cells). The joint variant recovers
part of that through a better-aimed cell. The detached variant learns the latent best but
gives the coda less than the joint variant.

## References (seed 1, 5000 steps)

| arm | tok/s | K1-K6 | gap to plain | paired vs ungraded fan |
| --- | --- | --- | --- | --- |
| ungraded fan | 9797 | +0.0042 | +0.254 | 0 |
| coda-graded fan | 7784 | - | +0.218 | -0.036 |
| reader-trained router | 9718 | +0.0068 | +0.2649 | +0.0109 [+0.0081, +0.0137] |
| latent-teacher router | 9355 | +0.0063 | +0.2677 | +0.0134 [+0.0107, +0.0161] |
| factor fan (weight 1) | 9197 | +0.0073 | +0.262 | +0.0078 [+0.0056, +0.0098] |

Positive "paired vs ungraded fan" means worse than the ungraded fan.

## Predictions (mine, orchestrator, before the runs)

Joint variant:

- **J-1**: no detonation. 80 %.
- **J-2**: `val/fan_lsel_r2` at step 5000 > 0.1. 55 %.
- **J-3**: paired CE vs the ungraded fan: worse by > 0.0137: 45 %; within 0.0137: 45 %;
  better by > 0.0137: 10 %.
- **J-4**: `val/fan_lsel_teacher_router_agree` > 0.40 (chance 0.25). 55 %.
- **J-5**: `val/fan_lsel_cell_spread` > 0.05 at step 5000 (the cells do not collapse onto
  the winner). 65 %.
- **J-6**: `fan/lsel_switch_rate` > 0.10 (the winner changes between passes). 60 %.
- **J-7**: `fan/lsel_teacher_pick_gap` > +0.005 nats (following the teacher beats following
  the router). 40 %.
- **J-8**: K1-K6 > +0.0092 (the ungraded fan plus 0.005). 35 %.
- **J-9**: tok/s within 5 % of the ungraded fan (9797). 75 %.

Detached variant:

- **D-1**: no detonation. 80 %.
- **D-2**: `val/fan_lsel_r2` at step 5000 > 0.1. 70 %. Higher than the joint variant's: 65 %.
- **D-3**: paired CE vs the ungraded fan worse by > 0.0137. 75 %.
- **D-4**: paired CE vs the joint variant worse by > 0.0137. 60 %.
- **D-5**: tok/s within 5 % of the ungraded fan. 75 %.

Repetition (greedy seq-rep-4, N=32, 128 tokens): the joint variant below the ungraded fan's
0.523 with disjoint intervals. 15 %.

Pass, joint: J-2 AND J-4 AND not worse than the ungraded fan by more than 0.0137 (the loop
learns the latent, the router learns the pick, and the tokens do not pay).
Pass, detached: D-2 AND within 0.0137 of the joint variant (the latent objective alone
trains the loop as well as the token CE does).

## Method

- Seed 1, 5000 steps each, `runner_steps.sh` at the commit of this prereg, joint first,
  after the 10x factor fan. Readouts: depth sweep (K1-K6), gap to plain, stage-2 score,
  worth profile, as for every fan arm. Paired CE against the ungraded fan's 5k sweep with
  `paired_vs_ruler.py`; the detached variant also paired against the joint variant.
- Repetition eval (N=32, 128 tokens, eager generator, router path) after the queue, beside
  the factor fan, both routers, the 10x factor fan and the four-rollout ensemble.
- Known eval flaw, kept for comparability: the 32 prompts are consecutive chunks of a few
  val articles, so the repetition intervals are too narrow.
- Smoke before queueing: 41+ steps of each config at this commit on the 5090.

## Results

Both arms ran 5000 steps, seed 1, joint first. No detonation (`chain.log` TRIPWIRE HEALTHY,
max preclip/total 49.3 joint, 48.3 detached). The teacher it describes is the next span's
EMA-prelude latent; "teacher-followed" below means the train forward reset the loop to the
teacher's pick after every pass, the same design this prereg specifies. The router-followed
variant (train forward follows the router instead) is a separate filing,
[`2026-10-01-latent-selected-loop-router-followed.md`](2026-10-01-latent-selected-loop-router-followed.md).

| reading | joint | detached |
| --- | --- | --- |
| tok/s | 9566 | 9687 |
| K1-K6 | -0.2294 [-0.2346, -0.2243] | -0.1701 [-0.1739, -0.1664] |
| gap to plain | +0.7146 [+0.6969, +0.7327] | +0.6663 [+0.6480, +0.6858] |
| ledger shipped CE | 4.7953 | 4.7474 |
| ledger zero-cell worth (TOTAL) | -0.1732 | -0.1333 |
| ledger teacher reading | -0.4235 [-0.4315, -0.4161] | -0.3592 [-0.3656, -0.3532] |
| `val/fan_lsel_r2` | -0.2586 | -0.2258 |
| `val/fan_lsel_teacher_router_agree` | 0.3066 | 0.3136 |
| `val/fan_lsel_cell_spread` | 0.0997 | 0.0808 |
| `fan/lsel_switch_rate` | 0.7536 | 0.8159 |
| `fan/lsel_teacher_pick_gap` | 0.4818 | 0.4099 |
| greedy seq-rep-4 | 0.301 [0.227, 0.382] | 0.214 [0.176, 0.260] |

Ungraded fan reference: tok/s 9797, K1-K6 +0.0042, ledger shipped CE 4.3343, greedy seq-rep-4
0.523 [0.450, 0.597]. Plain's greedy seq-rep-4: 0.794 [0.760, 0.828].

| prediction | reading | held |
| --- | --- | --- |
| J-1 no detonation | HEALTHY | yes |
| J-2 joint `fan_lsel_r2` > 0.1 | -0.2586 | no |
| J-3 joint paired CE vs ungraded fan: worse by > 0.0137 | +0.4610 (ledger CE) | yes |
| J-4 joint `teacher_router_agree` > 0.40 | 0.3066 | no |
| J-5 joint `cell_spread` > 0.05 | 0.0997 | yes |
| J-6 joint `switch_rate` > 0.10 | 0.7536 | yes |
| J-7 joint `teacher_pick_gap` > +0.005 | 0.4818 | yes |
| J-8 joint K1-K6 > +0.0092 | -0.2294 | no |
| J-9 joint tok/s within 5% of 9797 | 9566, 2.4% off | yes |
| D-1 no detonation | HEALTHY | yes |
| D-2 detached `fan_lsel_r2` > 0.1 | -0.2258 | no |
| D-2 detached r2 higher than joint's | -0.2258 > -0.2586 | yes |
| D-3 detached paired CE vs ungraded fan: worse by > 0.0137 | +0.4131 (ledger CE) | yes |
| D-4 detached paired CE vs joint: worse by > 0.0137 | -0.0479 (detached is BETTER) | no |
| D-5 detached tok/s within 5% of 9797 | 9687, 1.1% off | yes |
| Repetition: joint rep4 below ungraded fan's 0.523, disjoint | 0.301 vs 0.523, disjoint | yes |

## Verdict

Failure for both variants. Pass for the joint variant needs `fan_lsel_r2` > 0.1 and
`teacher_router_agree` > 0.40 and CE within 0.0137 of the ungraded fan; it clears none of
the three. Pass for the detached variant needs `fan_lsel_r2` > 0.1 and CE within 0.0137 of
the joint variant; it clears neither (it is 0.0479 BETTER than the joint variant, which is
itself outside the 0.0137 equivalence bar in the other direction). Both variants are far
worse than the ungraded fan on CE: +0.46 nats joint, +0.41 nats detached, 30-35x the noise
bar, and the ledger's own zero-cell worth is negative (-0.17 joint, -0.13 detached) meaning
the loop's output actively hurts the coda relative to not reading it at all. The cause was
found after the run, and the prereg did not anticipate it (the orchestrator wrote the
spec): the teacher picks with the next span's latent, built from tokens the coda has not
yet seen, so train sees teacher-chosen cell lineages and deploy sees the router's. The
ledger's own "teacher" reading, replaying the hindsight pick at eval, scores 4.372 nats
(joint) and 4.388 nats (detached), close to the ungraded fan's 4.334 and far better than
either arm's shipped CE. That gap between the teacher path (good) and the router path
(shipped, bad) is a train/deploy mismatch, not a capacity problem with the cells themselves.
The one genuinely surprising result is the repetition number: the joint variant's greedy
seq-rep-4, 0.301, is far BELOW the ungraded fan's 0.523, disjoint CIs, the 15% long shot in
this prereg's own predictions. I do not read this as a win. A model whose shipped CE is 0.46
nats worse is choosing less-repetitive text because its token distribution is damaged, not
because it is more diverse in a useful sense; "repeats less" and "writes well" came apart
here.

## Updated hypothesis

Selecting inside the loop on a hindsight latent teaches the cells something real (the
teacher reading recovers near the ungraded fan's quality), but deploying on the router,
which only reaches 0.31 agreement with that teacher, throws almost all of it away. The fix
is not a better router trained after the fact; it is making the train-time forward match
the deploy-time forward, i.e. following the router during training too, so the coda never
sees cell lineages it will not see at inference. See
[`2026-10-01-latent-selected-loop-router-followed.md`](2026-10-01-latent-selected-loop-router-followed.md)
for that attempt.
