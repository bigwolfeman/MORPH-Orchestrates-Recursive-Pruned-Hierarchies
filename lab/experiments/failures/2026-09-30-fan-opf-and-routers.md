# Planned: orthogonal factors or a trained router on the write-all fan (arms F, R, T)

Status: failure

Date: 2026-09-30 12:24 CDT (frozen before any of the three runs trained; nowta's and e1probe's
repetition eval was still running, so no nowta repetition number is known here).
Code: cbdd501 (arms), d9fa146 (repetition eval); note
[`2026-09-30-fan-opf-and-routers.md`](../../../.agents/notes/proposed/architecture/2026-09-30-fan-opf-and-routers.md).
Wolfe (2026-09-30): "every exploration should sit in the context window for late use";
a router picks the winner "based on ground truth latents. No decoding"; "multi task
learning is probably what we need" (JEPA-Anything, arXiv 2609.20800); "Main thing we can
measure at this early training is repetition and diversity." "Let's try all of the above."

## Question

nowta (write-all fan, M=4 cells in the coda's context, no winner test) runs at plain's speed
but sits 0.036 nats behind a2, whose coda-graded winner test makes the cells specialise at
0.78x plain's speed. Can a cheap mechanism give the cells distinct jobs at nowta's cost?

- **F (`tul.fan_opf`)**: the cells are orthogonal FACTORS of the next span's latent (an EMA
  prelude twin's pooled next-span state, split by a QR-orthogonal P; cell k predicts slice k),
  jointly with the token CE. No selection.
- **R (`tul.fan_route: reader`)**: a top-1 router; the coda reads only the winner, scaled by
  p_win, so the coda's CE trains the router (MoE style); DeepSeek-V3 bias balance.
- **T (`tul.fan_route: latent`)**: the same read; the router copies a latent teacher (the cell
  whose detached map lands closest to the true next-span latent). No gradient into the cells.

All three are nowta plus ONE TULConfig key (compose diff verified).

## Hypothesis

H-F: a per-cell latent job is the multi-task anchor LCTUL-J never had; with the token CE on the
same cells and the floor on the online prelude, the target does not collapse and the cells
split into distinct content, which helps the coda a little and repeats less.
H-R: reading one cell of four throws away most of what nowta's coda reads; the router learns
little in 5k steps, so R trails nowta.
H-T: latent closeness does not match what the coda needs (the latent-WTA probe read chance);
T trails nowta and its teacher agrees with the coda's best at about chance.

## References (seed 1, 5000 steps, 480 sweep rows; repetition at N=32 prompts, 128 tokens)

| arm | tok/s | K1-K6 | gap to plain | val loss | greedy seq-rep-4 |
| --- | --- | --- | --- | --- | --- |
| plain | 9988 | - | 0 | - | 0.794 +- 0.034 |
| a2 | 7784 | +0.0057 | +0.218 | 4.4365 | 0.510 +- 0.067 |
| nowta | 9797 | +0.0042 | +0.2543 | 4.4658 | (running) |

Paired CE bar 0.0137. tok/s = mean of the trainer's logged points at step >= 200.

## Predictions (mine, orchestrator, before any run)

- **P-1**: none of the three detonates. 80 %.
- **P-2**: F tok/s >= 0.9x nowta (8817). 75 %. R and T each >= 0.95x nowta (9307). 80 %.
- **P-3**: F vs nowta, paired CE at depth 6: better by > 0.0137: 25 %; within: 55 %; worse: 20 %.
- **P-4**: R worse than nowta by > 0.0137: 55 %. Better by > 0.0137: 10 %.
- **P-5**: T worse than nowta by > 0.0137: 60 %.
- **P-6**: F at step 5000: `fan/opf_enc_std_min` >= 0.1 (the floor holds): 70 %;
  `fan/opf_r2_mean` > 0.1 (the cells predict their factors above the batch mean): 60 %.
- **P-7**: R `fan/router_coda_agree` > 0.35 (chance 0.25): 45 %. R's largest load share < 0.4: 70 %.
- **P-8**: T `fan/teacher_router_agree` > 0.5: 55 %. `fan/teacher_coda_agree` > 0.35: 30 %.
- **P-9**: each arm's greedy seq-rep-4 is below plain's 0.794 with disjoint CIs: 80 %.
  F's greedy seq-rep-4 is below nowta's with disjoint CIs: 30 %.
- **P-10**: each arm's K1-K6 within 0.007 of nowta's +0.0042. 75 %.

Pass for an arm: paired CE not worse than nowta by more than 0.0137, AND greedy seq-rep-4 at or
below nowta's (overlapping or lower CI), AND tok/s >= 0.9x nowta.

## Method

- Configs `tul_slot_spandec_strict_fan4_all_fp01_{opf,rmoe,rlat}`, seed 1, 5000 steps,
  `runner_steps.sh` at the commit of this prereg, order F, R, T. Guard: a copy of
  `hwta_guard.sh` with these tags.
- Readouts as for every slot arm (sweep, gap to plain, worth). Paired CE vs nowta with
  `paired_vs_ruler.py`, nowta's 5k sweep as the ruler.
- Repetition: `gen_diversity_eval.py` via `run_eval.sh`, N=32 val prompts (trainer offset
  50k), 128-token prompts, 128-token continuations, greedy and t=1.0/top-p 0.95, gen-PPL scored
  by the plain checkpoint, beside the real continuations. Run after the three trainings.
  N and length are below the builder's N=64 x 256 because the eager generator costs ~12 s per
  row at 128 tokens on a fan arm (measured: per-row time flat from N=8 to N=64).
- The full test suite ran at d9fa146 on this machine before the queue started (2 one-core
  shards: 3137 passed, 28 failed, all 28 in `test_tul_generate_cached.py` from a stale
  `sample_next` spy after top_p; fixed in 81baee1, that file then 78 passed with the generation tests).
- Hazard: another Claude session (Olympiad) runs GPU jobs of ~23 GB outside
  `gpu.lock`; one OOM-killed an eval on 2026-09-30. A run killed that way is re-queued, and a
  tok/s series with a foreign job in it is marked.

## Results

All three runs finished 5000 steps at commit cbdd501, seed 1, in order factor fan, then
reader-trained router, then latent-teacher router. In this section F is the factor fan, R
is the reader-trained router, and T is the latent-teacher router (the letters match the
Predictions section above, which is frozen). No run detonated: `chain.log` reads TRIPWIRE
HEALTHY for all three, with the worst preclip/total ratio at 243 (factor fan), far under the
1e4 abort bar.

| arm | tok/s | paired CE vs ungraded fan | K1-K6 | gap to plain | greedy seq-rep-4 |
| --- | --- | --- | --- | --- | --- |
| plain | 9988 | - | - | 0 | 0.794 [0.760, 0.828] |
| ungraded fan (nowta) | 9797 | 0 | +0.0042 | +0.2543 | 0.523 [0.450, 0.597] |
| coda-graded fan (a2, reference) | 7784 | -0.036 | +0.0057 | +0.218 | 0.510 [0.444, 0.579] |
| factor fan (F) | 9197 | +0.0078 [+0.0056, +0.0098] | +0.0073 | +0.262 | 0.561 [0.487, 0.639] |
| reader-trained router (R) | 9718 | +0.0109 [+0.0081, +0.0137] | +0.0068 | +0.2649 | 0.5535 [0.482, 0.623] |
| latent-teacher router (T) | 9355 | +0.0134 [+0.0107, +0.0161] | +0.0063 | +0.2677 | 0.5628 [0.489, 0.635] |

Per-arm val readings at step 5000: factor fan `fan_opf_enc_std_min` 0.4332, `fan_opf_r2_mean`
0.0264, `fan_opf_target_rank` 12.39, `fan_opf_orth_err` 0.0039. Reader-trained router
`router_coda_agree` 0.4574, `router_pick_regret` 0.0300 against `rand_pick_regret` 0.0637,
load shares 0.36 / 0.25 / 0.08 / 0.32, `route_p_win` 0.8387. Latent-teacher router
`router_coda_agree` 0.2894, `teacher_coda_agree` 0.2766, `teacher_router_agree` 0.2938.

| prediction | reading | held |
| --- | --- | --- |
| P-1 none detonates | all three HEALTHY | yes |
| P-2 F tok/s >= 8817 | 9197 | yes |
| P-2 R, T tok/s >= 9307 | 9718, 9355 | yes |
| P-3 F vs ungraded fan within 0.0137 | +0.0078 | yes (within) |
| P-4 R worse than ungraded fan by > 0.0137 | +0.0109, under the bar | no |
| P-5 T worse than ungraded fan by > 0.0137 | +0.0134, under the bar by 0.0003 | no |
| P-6 F `opf_enc_std_min` >= 0.1 | 0.4332 | yes |
| P-6 F `opf_r2_mean` > 0.1 | 0.0264 | no |
| P-7 R `router_coda_agree` > 0.35 | 0.4574 | yes |
| P-7 R largest load share < 0.4 | 0.36 | yes |
| P-8 T `teacher_router_agree` > 0.5 | 0.2938 | no |
| P-8 T `teacher_coda_agree` > 0.35 | 0.2766 | no |
| P-9 each arm's rep4 below plain's 0.794, disjoint | 0.561 / 0.5535 / 0.5628, all disjoint | yes |
| P-9 F's rep4 below the ungraded fan's, disjoint | 0.561 vs 0.523, F is HIGHER | no |
| P-10 each arm's K1-K6 within 0.007 of +0.0042 | +0.0073, +0.0068, +0.0063 (diffs 0.0031, 0.0026, 0.0021) | yes |

## Verdict

Filed as a failure because the predictions that test each mechanism missed, not because
of the pass rule. (Corrected 2026-10-02 by the orchestrator: an earlier draft of this
section misread the pass rule.)

The pass rule is a non-inferiority bar, and all three arms MEET it. Each arm's paired CE
sits within 0.0137 of the ungraded fan's (+0.0078, +0.0109, +0.0134). Each arm's greedy
seq-rep-4 interval overlaps the ungraded fan's [0.450, 0.597], which the rule accepts. Each
arm's tok/s is above 0.9x the ungraded fan's 8817. The latent-teacher router clears the CE
bar by only 0.0003.

The mechanism predictions are what fail. The factor fan did not learn its latent task
(P-6: R^2 mean 0.026 against the 0.1 bar). The latent-teacher router's teacher did not
track the router or the coda (P-8: 0.294 and 0.277, chance 0.25). The factor fan did not
repeat less than the ungraded fan (P-9: 0.561 against 0.523). The two routers cost less CE
than I predicted (P-4, P-5 missed in the good direction). Only the reader-trained router's
mechanism held: it picks the coda's own best cell at 0.457 (chance 0.25), with pick regret
0.030 against a random pick's 0.064. Its exploration ledger, run later on the same
checkpoint, prices the learned pick at +0.033 nats over a random cell.

So: none of the three arms hurts the ungraded fan beyond the bar, and none of them buys
anything the ungraded fan lacks, at 5000 steps on seed 1.

## Updated hypothesis

A cheap auxiliary task does not reliably make a write-all fan's cells distinct enough to help
generation diversity at this scale, in 5000 steps. The factor fan's task undertrains, and
both routers read the coda's preference at or only slightly above chance. Reading one cell of
four instead of all four (reader router, latent router) costs nothing on CE beyond the noise
bar, but does not buy anything either: the question "can we get the ungraded fan's speed and
the coda-graded fan's quality cheaply" is open after six mechanisms now (ungraded fan itself,
head-graded WTA, factor fan, reader router, latent router). The next lever worth trying is
sized correctly rather than auxiliary: see
[`2026-09-30-factor-fan-10x-latent.md`](2026-09-30-factor-fan-10x-latent.md) for
what happens when the latent term is reweighted to compete evenly with the token CE.
