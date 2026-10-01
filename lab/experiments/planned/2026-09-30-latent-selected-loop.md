# Planned: does selecting inside the loop on the next span's latent make the fan's loop useful?

Status: planned

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
