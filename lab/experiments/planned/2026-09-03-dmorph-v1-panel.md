# dmorph v1 panel: two no-loop flow-matching arms against the coreless bar

Status: planned
Date written: 2026-09-03 (before any run; predictions frozen at commit time)

## Question

Does a DiffusionBlocks-routed flow-matching objective on a 12-layer flat MORPH (no core
loop, packed TUL row) buy token CE that the coreless bar cannot, at matched wall-clock
and at matched tokens — and does the B-step ladder refine anything at inference?

## Hypothesis

Wolfe's: the loop's value can be bought back without BPTT by (a) an explicit
"predict the next thought / next token in latent space" objective on shared weights
and (b) iterative refinement at inference that costs one forward of the flat stack.
The record's counter-hypothesis: denoising objectives on text at this scale trade
likelihood for the wrong metric (testbed: −0.67 nats at 4x tokens), and latent
prediction on additive channels scores zero worth.

## Arms (seq 1024, batch 6, seed 1 and seed 2, 20k steps, the panel flags of the
warmup pair: `training.warmup=1000 training.ademamix_alpha_cap=3.5
training.ademamix_t_beta3=3500 training.eval_every=250 training.gen_every=0`)

- `dmorph_ctl`: flat 6:0:6, packed row, plain CE (the bar on this depth).
- `dmorph_tok`: + noisy stream at every position, `y` = next-token embedding.
- `dmorph_hs`: + noisy stream at slot positions, `y` = detached post-stack slot state.
- Reference, not re-run: paid loop A2 at 20k on the ramp (`tul-a2-20k-wu`, 3.4701 on
  480 rows at depth 6) and notul (3.4486).

Scoring: clean-head `val/ce_tokens` last-20 mean and the 480-row same-rows sweep at 20k;
matched-wall-clock read at the step where each arm's cumulative wall clock equals the
control's 20k; ladder metrics; generation with the diversity guard (rep4, distinct-3)
against the real-text anchor.

## Predictions (majority priors; a prediction is scored on its side)

- P1 (55 %): at matched TOKENS (20k steps) `dmorph_tok`'s clean head is within
  ±0.02 nats of `dmorph_ctl` — the auxiliary stream neither helps nor hurts the
  likelihood head.
- P2 (65 %): at matched WALL-CLOCK `dmorph_ctl` beats both dmorph arms on the clean
  head, because the noisy stream costs 1.25x per token analytically and more in
  practice (eager attention).
- P3 (70 %): `dmorph_tok`'s ladder head after B=4 steps with the hard bridge has a
  greedy token accuracy BELOW the clean head's greedy accuracy on the same rows.
- P4 (60 %): `dmorph_hs`'s ladder output has cosine > 0.9 to the clean target, and the
  plan-worth cost of replacing the clean slot state with the ladder output is
  ≤ 0.01 nats (i.e. the refinement is a faithful copy, not an improvement) — the
  refinement does not beat the one-pass state it was trained to reach.
- P5 (60 %): both dmorph arms beat the paid loop A2's 480-row number (3.4701) at
  matched wall-clock, because the flat stack sees ≥2.5x the tokens in the same time.
- P6 (75 %): the fraction of `dmorph_tok` training mass above the decodability
  threshold (the σ* test in `t` coordinates) exceeds 40 % with uniform `t`, and the
  filing has to reshape `t` — the same trap, new clothes.

## Method

Build per the design note; CPU suite green; `dmorph_smoke.yaml` for all three arms;
then the six runs (three arms × two seeds) sequentially on the 5090 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, wandb `adew-me/morph-tul`,
`WANDB_DIR=/home/wolfe/morph-scratch`, one trainer at a time, abort at
`preclip/total > 1e4` after step 200. Artifacts to
`lab/experiments/results/2026-09-03-dmorph-v1-panel/`. File under `successes/` iff
every prediction resolves on its majority side, else `failures/`.

### Method amendment, 2026-09-03 (before launch; Predictions untouched)

1. **5,000 steps per run, not 20,000.** Wolfe, 2026-09-03: "let's just send them for
   5k". Every read that said "at 20k" is read at 5k: the last-20 eval mean over
   evals every 250 steps, and the same-rows sweep at step 5000. The matched-wall-clock
   read is taken at the step where each dmorph arm's cumulative wall clock equals the
   control's wall clock at 5,000. `training.warmup` stays 1000 and
   `training.ademamix_t_beta3` stays 3500 (pinned; the note's rule 5).
2. **`dmorph.source_std = 1.0` on both dmorph arms**, not the note's matched `1/√d`.
   Reason: the builder's decodability read at init (`lab/dmorph/decodability.py`,
   implementation record in the design note) put 88.6 % of the tok arm's training mass
   where `x_t` is already nearest-neighbour decodable against the table; the design
   note's Risks section set ~40 % as the reshape trigger. `source_std 0.5` gives 31 %,
   `1.0` gives 16 %. P6 (the trap exceeds 40 % with the matched source) is therefore
   resolved BEFORE the run, on the tok arm at init, on its majority side (TRUE), and the
   panel runs the reshaped source. The hs arm gets the same value so the two arms differ
   only in the target.
3. The run names carry the step count: `dmorph-{ctl,tok,hs}-s{1,2}-5k`.

