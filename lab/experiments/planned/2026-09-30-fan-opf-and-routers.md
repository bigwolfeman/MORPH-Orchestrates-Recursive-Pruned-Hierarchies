# Planned: orthogonal factors or a trained router on the write-all fan (arms F, R, T)

Status: planned

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
