# Planned: can the write-all fan keep a2's quality without the coda-graded WTA passes?

Status: planned

Date: 2026-09-29 23:53 CDT (frozen before either run trained).
Code: 7a1c181 (`tul.fan_all_wta_grader`, arm A; note
[`2026-09-29-fan-head-graded-wta.md`](../../../.agents/notes/proposed/architecture/2026-09-29-fan-head-graded-wta.md)).
Wolfe (2026-09-29): "What if we just explored everything at once and let the coda learn how to
pick a winner? This is basically A"; "ideally LXTUL would be faster than plain ... Exploring the
latents shouldnt be more expensive than the central loop."

## Question

a2 (`tul_slot_spandec_strict_fan4_all_fp01`) writes M=4 loop cells into the coda and lets the
coda's per-token attention select. Its WTA term costs M no-grad coda passes plus one grad pass
per step. Two arms remove that cost:

- **nowta**: a2 at `fan_all_wta_lambda: 0`. No grader at all: the coda reads all four cells in
  its one ordinary pass and learns to pick. This is Wolfe's "explore everything at once" arm
  and the cost floor of the write-all fan.
- **hwta**: a2 with the parallel span head as the grader. The head reads each cell alone and
  scores it on the tokens a2's coda table scores; a relaxed WTA over the cells trains the head
  and the cells. No extra coda pass.

Does either keep a2's CE, and at what tok/s against a2 and against plain?

## Hypothesis

H: the coda passes are most of a2's extra cost, so nowta runs at about plain's tok/s or faster
(the FLOP count says a TUL row with one coda pass is about half of plain's). The WTA term is
what makes the cells specialise, so nowta loses some of a2's gain. The head grader is not
cheap at this shape: its vocabulary GEMM runs M x tokens x 49k, about what the coda passes
cost (41-step smoke: hwta 7093, a2 6876, nowta 9503 tok/s). A 2-block committed reader ranks
the cells on what IT can see; that agrees with the coda only weakly.

## References (seed 1, 5000 steps, 480 sweep rows)

| arm | K1-K6 | gap to plain 5k | val loss | tok/s (mean, logged steps >= 200) |
| --- | --- | --- | --- | --- |
| a2 (fan4-all-fp01) | +0.0057 [+0.0052, +0.0063] | +0.218 [+0.203, +0.235] | 4.4365 | 7784 |
| plain panel (nm ctrl s1r) | - | 0 | - | 9988 |

The paired CE bar on this tree is 0.0137 (3.4x seed spread measured on the a/b/a2 panel). The
K1-K6 seed spread in this family is about 0.007.

## Predictions (mine, orchestrator, before either run)

- **P-1**: neither run detonates. 85 %.
- **P-2**: nowta mean tok/s (logged steps >= 200) >= plain's 9988. 50 %.
- **P-3**: hwta mean tok/s < 9000 (the head grader recovers little of a2's WTA cost). 85 %.
  hwta >= 1.15x a2 (8952): 15 %.
- **P-4**: nowta is worse than a2 by more than 0.0137 nats/token (paired on the same 480 rows,
  CE at depth 6). 55 %.
- **P-5**: hwta is within 0.0137 of a2 (paired). 45 %. Worse by more than 0.0137: 40 %.
  Better by more than 0.0137: 15 %.
- **P-6**: hwta `fan/head_coda_agree` at step 5000 > 0.35 (chance 0.25). 40 %.
- **P-7**: hwta `fan/head_pick_regret` < 0.5 x `fan/rand_pick_regret` at step 5000. 30 %.
- **P-8**: hwta's largest `fan/head_wta_share_k{i}` at step 5000 < 0.7 (no collapse onto one
  cell). 60 %.
- **P-9**: each arm's K1-K6 is within 0.007 of a2's +0.0057. 70 %.

Pass for Wolfe's arm (nowta): P-2 AND not P-4 (plain's speed at a2's quality). Pass for hwta:
not worse than a2 by more than 0.0137 AND tok/s >= 1.1x a2 (8562).

## Method

- Configs `tul_slot_spandec_strict_fan4_all_fp01_hwta` and `..._nowta` (both compose a2).
  Seed 1, 5000 steps, runner `runner_steps.sh` at the commit of this prereg, queued after
  arm B's pair (`2026-09-29-code-policy-one-rollout.md`). Order: nowta, then hwta.
- Readouts as for every slot arm: depth sweep (K1-K6, 480 rows), gap to plain 5k, worth.
  The stage-2 scorer cannot pair fan arms (it exits 1 on a2), so the a2 comparison is
  `paired_vs_ruler.py` with a2's 5k sweep as the ruler, run after both arms finish.
- tok/s: mean of the trainer's logged points at step >= 200, beside a2's and plain's means
  computed the same way from their run logs.
- No CPU test suites run on this machine while either arm trains. The merge suite for this
  code ran on the DGX Spark (CPU, 8 shards, master vs 7a1c181): the ONLY failures A adds are its
  two a2 value pins, which miss on the Spark (ARM) and the 3070 exactly as master's own a2 pins in
  `test_tul_lx_credit.py` do; both pass on this machine (1.5 s on one niced core, logged in
  chain.log). Configs compose and reach MORPHConfig (grader, lambda, spandec_parallel asserted).
- Guard: a copy of `policy_guard.sh` with this pair's tags (sustained tripwire, kill by PID).
