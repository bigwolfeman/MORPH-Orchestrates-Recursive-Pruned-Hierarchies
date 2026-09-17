# Experiment: the code target with the progressive loss — do later passes move toward the code?

Status: planned

Date: 2026-09-17 (frozen before the arm starts; arm A is at step ~3k as this is written,
reading pass 1 = pass 6 on the per-pass cosine). Parent: `2026-09-17-lctul-target-slot-loop.md`.

## Question

On the code target, arm A's first 1,500 steps read the entry state at cosine 0.03 to the
code, pass 1 at the exit value (0.14 to 0.17) and passes 2 to 6 at the same value: the map
meets the target in one pass, as every per-pass target before it did. Wolfe: "Didn't we
have a design to address loop regression" — the Deep Thinking progressive loss
(`tul.progressive_p`, `2026-09-10-credit-assignment-in-the-slot-loop.md`) is the one built
for a fixed external target, and it has only been read against a bag-of-words target.
With the first k passes' outputs detached at random, does the map learn to improve a state
it did not produce, so that pass 6 sits closer to the code than pass 1?

## Hypothesis

H1: the one-pass fixed point is a co-adaptation artefact of full BPTT on the map's own
trajectory; cut it and later passes move the state toward the code. H0: the target is
reachable in one application of a 6-block map from the entry state, so no training regime
makes the later passes matter (the 2026-09-10 reading: the state kept moving past depth 6
and the loss did not follow).

## Predictions (frozen)

Arm `tul-code-target-prog` (`tul_code_target_prog.yaml`: arm A + `progressive_p: 0.5`,
20k steps, one seed), against arm A at matched steps.

- P-G1 (later passes move): at 20k, `tul/code_target_cos_l6` − `tul/code_target_cos_l1`
  ≥ 0.03 averaged over the last 500 logged steps (arm A's reading at 1.5k: 0.00). 30 %.
- P-G2 (no cost on the exit): `val/code_target_cos` at 20k within 0.03 of arm A's, or
  higher. 55 %.
- P-G3 (depth pays on the frozen coda): forced-depth sweep at 20k, `ce_tokens` at depth
  6 minus depth 1 ≤ −0.02 with the paired interval clear of 0, where arm A's is not. 25 %.
- P-G4 (generation): semantic probe OWN at 20k within 0.02 of arm A's OWN paired, or
  higher. 50 %.
- P-G5 (rate): ≥ 0.9× arm A's tok/s (the cut adds no compute). 80 %.

## Binding

- P-G1 and P-G3 → the fixed point was a training artefact: the progressive cut goes into
  the continuation recipe and the depth question reopens on this target.
- P-G1 holds, P-G3 fails → the passes move toward the code without the coda gaining; read
  with the unfreeze continuation before calling it.
- P-G1 fails → H0: one application reaches the target; the loop's depth is not a
  training-regime question on this target either. File and stop this knob for good.

## Method

Queue line behind the two continuations (kind slot, commit recorded at insertion, EXTRA
with `training.init_from` and the panel extras, exactly arm A's line). Spark smoke of the
config with `init_from` (12 steps) before insertion; a third watcher instance for the
semantic probe at 10k and 20k. Pairing against arm A with `code_semantic_pair.py`; the
per-pass series from wandb (`tul/code_target_cos_l{t}`).

Amended 2026-09-17 15:20 (reason: the smoke ran and the line was inserted): Spark smoke of
`tul_code_target_prog` at 9c40efe with `init_from`, 12 steps: exit 0, 71.6M trainable /
198.3M frozen, guard ceiling 100000. `tests/test_tul_progressive.py` + `test_tul_code_target.py`:
28 passed. Queue line inserted at 15:20 behind the two continuations (commit 9c40efe);
watcher instance started 15:10 for 10000 and 20000. Predictions unchanged.

Amended 2026-09-17 15:30 (reason: arm A took one divergence-guard strike at step 5360 on
the frozen coda's CE, which trains nothing on a detach arm): `tul_code_target.yaml` raises
`div_ppl_ceiling` 1e5 → 1e8 and this arm's queue line moves to the commit that carries
it. Nothing else in the config changes. Predictions unchanged.

## Not verified before launch

- CPU: a tiny code-target model with `progressive_p: 0.5` runs forward/backward and emits
  the per-pass keys; the knob's own contracts are `tests/test_tul_progressive.py` (2026-09-10).
- One seed; the same frozen coda as arm A.

## Results

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
