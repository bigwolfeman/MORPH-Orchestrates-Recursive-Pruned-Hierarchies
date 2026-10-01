# Planned: does the factor fan learn its latent task at 10x weight, and what does it cost the tokens?

Status: planned

Date: 2026-09-30 22:26 CDT (frozen before the run trained).
Config: `tul_slot_spandec_strict_fan4_all_fp01_opf10` (the factor fan with the whole latent
block x10). Parent prereg: [`2026-09-30-fan-opf-and-routers.md`](2026-09-30-fan-opf-and-routers.md).
Wolfe (2026-09-30): "I think the multi task learning is what previous tuls needed." "start it".

## Question

The factor fan (four fan cells, each predicting one orthogonal slice of the next span's
latent, jointly with the token CE) barely learned its latent task at weight 1: val R^2 mean
0.026 at step 5000, three of four factors below 0. Its CE sat 0.0078 behind the ungraded fan
(inside the 0.0137 bar). So the multi-task idea was not tested: the second task never
became a task. At 10x the latent term is about the size of the token CE. Does the latent
task get learned, and is the token CE hurt, helped or unchanged when it does?

## Hypothesis

At 10x the cells learn their factors (R^2 well above 0). Whether that helps the tokens is the
open question; my lean is a small cost on CE (the two tasks compete for the same four cells)
and slightly more depth use, as at weight 1.

## References (seed 1, 5000 steps)

| arm | tok/s | K1-K6 | gap to plain | paired vs ungraded fan | latent R^2 mean |
| --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | +0.0042 | +0.254 | 0 | - |
| factor fan (weight 1) | 9197 | +0.0073 | +0.262 | +0.0078 [+0.0056, +0.0098] | 0.026 |
| plain | 9988 | - | 0 | - | - |

## Predictions (mine, orchestrator, before the run)

- **P-1**: no detonation. 80 %.
- **P-2**: `fan/opf_r2_mean` at step 5000 > 0.1. 60 %.
- **P-3**: paired CE vs the ungraded fan: worse by > 0.0137: 45 %; within 0.0137: 45 %; better
  by > 0.0137: 10 %.
- **P-4**: `fan/opf_enc_std_min` >= 0.1 (the floor holds): 75 %. Target effective rank >= 12
  (no lower than at weight 1): 55 %.
- **P-5**: K1-K6 above the weight-1 factor fan's +0.0073. 40 %.
- **P-6**: tok/s within 3 % of the weight-1 factor fan (9197). 85 %.
- **P-7**: greedy repeated-4-gram rate below the ungraded fan's 0.523 with disjoint intervals. 20 %.

Pass: P-2 AND not worse than the ungraded fan by more than 0.0137 (the second task is learned
without costing the tokens).

## Method

- Seed 1, 5000 steps, `runner_steps.sh` at the commit of this prereg, after the
  latent-teacher router. Same readouts as the parent prereg; paired CE against the ungraded
  fan's 5k sweep with `paired_vs_ruler.py`.
- Repetition eval (N=32, 128 tokens, same prompts as the baselines) after the queue, for this
  arm, the factor fan, both routers and the four-rollout ensemble.
- Known eval flaw, kept for comparability: the 32 prompts are consecutive chunks of a few
  val articles, so the repetition intervals are too narrow.
