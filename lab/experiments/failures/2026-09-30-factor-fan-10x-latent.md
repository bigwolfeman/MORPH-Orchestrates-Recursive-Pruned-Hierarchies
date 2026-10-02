# Planned: does the factor fan learn its latent task at 10x weight, and what does it cost the tokens?

Status: failure

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

## Results

Ran 5000 steps, seed 1, config `tul_slot_spandec_strict_fan4_all_fp01_opf10`. No detonation
(`chain.log` TRIPWIRE HEALTHY, max preclip/total 36.1, step 331). tok/s is the mean of the
24 logged points at step >= 200 in `run.log` (the same convention used for every other arm
in this campaign): 8982. A looser read that drops only the first two logged lines (step 0
and step 200, instead of keeping step 200) gives 9007; I use 8982 because it is the
convention that reproduces the already-filed tok/s numbers for the factor fan, the
reader-trained router and the latent-teacher router to within 2 tok/s.

| arm | tok/s | K1-K6 | gap to plain | ledger shipped CE | `fan_opf_r2_mean` | `fan_opf_enc_std_min` | `fan_opf_target_rank` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | +0.0042 | +0.254 | 4.3343 | - | - | - |
| factor fan (weight 1) | 9197 | +0.0073 | +0.262 | 4.3421 | 0.0264 | 0.4332 | 12.39 |
| 10x factor fan | 8982 | +0.0047 [+0.0043, +0.0052] | +0.3845 [+0.3688, +0.4016] | 4.4655 | 0.0420 | 0.3692 | 6.45 |
| plain | 9988 | - | 0 | - | - | - | - |

Paired CE, 10x factor fan minus ungraded fan, on the ledger's 480 shared rows: 4.4655 -
4.3343 = +0.1312. This is a ledger-to-ledger comparison, not a `paired_vs_ruler.py` CI, but
both ledgers score the same 511089 tokens, so the comparison is apples to apples.

| prediction | reading | held |
| --- | --- | --- |
| P-1 no detonation | HEALTHY | yes |
| P-2 `fan_opf_r2_mean` > 0.1 | 0.0420 | no |
| P-3 paired CE vs ungraded fan: worse by > 0.0137 | +0.1312 | yes |
| P-4 `fan_opf_enc_std_min` >= 0.1 | 0.3692 | yes |
| P-4 target rank >= 12 | 6.45 | no |
| P-5 K1-K6 above the weight-1 factor fan's +0.0073 | +0.0047 | no |
| P-6 tok/s within 3% of the weight-1 factor fan's 9197 | 8982, 2.3% off | yes |
| P-7 greedy seq-rep-4 below the ungraded fan's 0.523, disjoint | 0.593 [0.523, 0.655], HIGHER not lower | no |

## Verdict

Failure. The pass rule is `fan_opf_r2_mean` > 0.1 AND not worse than the ungraded fan by
more than 0.0137. Neither half holds. At 10x weight the latent task still does not train:
R^2 mean moved from 0.026 to 0.042, still far under the 0.1 bar that would call it learned,
and the target effective rank fell from 12.39 to 6.45, meaning the ten-fold weight pushed
the four cells toward a narrower, not a wider, span of the target space. The token CE paid
for the attempt regardless: the gap to plain widened from +0.262 to +0.3845, and paired
against the ungraded fan on the same 511089 tokens the 10x arm is 0.1312 nats worse, about
17x the 0.0137 noise bar and 17x the weight-1 arm's own 0.0078 gap. The repetition eval
moved the wrong way too: greedy seq-rep-4 rose to 0.593, the highest of any fan arm measured
so far, not the lowest. Raising the weight did not turn "the second task never became a
task" into a working multi-task objective; it mostly cost tokens.

## Updated hypothesis

The factor fan's latent task is not weight-limited. Ten times the loss moved R^2 by 0.016
and tightened the target rank, the opposite of what a genuinely multi-task objective should
do under more pressure. The four cells do not have the capacity, or the architecture does not
give them the right gradient path, to carry both an orthogonal-factor prediction and the
token CE at once. Reweighting this term further is not worth trying without first checking
whether the cells can predict their factors at all when the token CE is removed; if they
cannot, the factor task itself, not its weight, is the dead end.
