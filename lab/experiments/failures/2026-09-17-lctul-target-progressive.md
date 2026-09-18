# Experiment: the code target with the progressive loss — do later passes move toward the code?

Status: failure

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

Arm ran to 20k, one seed. Depth sweeps and the semantic probe were re-run on the 3070 from
master: the runner's own sweeps exited 1 at all four checkpoints because the pinned commit
predates the sweep fix (2236ac4), so every in-queue depth readout for this arm is missing.
Per-pass series from wandb `b4wev1jb` (prog) and `edgwqcr1` (arm A).

- P-G1 FAILS. `code_target_cos_l6 - l1` over the last 500 logged steps: prog +0.0073, arm A
  +0.0051 (needed >= 0.03). Stable across windows: prog +0.0073 / +0.0065 / +0.0060 over the
  last 500 / 2000 / 5000, arm A +0.0051 / +0.0050 / +0.0033; sd about 0.010 on n=26, so
  roughly +0.0073 +/- 0.0020. The knob DOES move later passes, by about 1.4x arm A, and
  misses the bar by a factor of four. Corroborated independently by the forced-depth mean
  probe (centred cosine d1->d6: prog +0.0071/+0.0074 per cell vs arm A +0.0036/+0.0033, about
  2x).
- P-G2 HOLDS. `val/code_target_cos` 0.1652 vs arm A 0.1676, inside 0.03.
- P-G3 FAILS. `ce_tokens` depth 6 minus depth 1 = +0.1173 (needed <= -0.02): the WRONG
  direction. prog K1-K6 -0.1173 [-0.1304, -0.1040]; arm A's is +0.0323 [+0.0236, +0.0415].
- P-G4 HOLDS. Semantic OWN paired against arm A over the same 120 cuts: +0.0028
  [-0.0127, +0.0182] (prog 0.1331, arm A 0.1303). ORACLE is 0.5645 on both arms, as it must
  be: same frozen coda, same VAE stage.
- P-G5 HOLDS. 14495 tok/s vs arm A's 14102, 1.03x.

**P-G3's failure does not mean what the prediction assumed.** The frozen coda prefers NO cell
to a confidently-wrong one (`zero - own` -4.64 nats), so its depth-CE curve tracks how GENERIC
the cell becomes, not how well it matches the code. Measured here in both directions:

| arm | `pred.zbar` d1 -> d16 | frozen-coda CE d1 -> d16 | K1-K6 |
| --- | --- | --- | --- |
| arm A | 0.3081 -> 0.3225 (blander) | 9.1713 -> 9.0759 | +0.0323 |
| prog | 0.3554 -> 0.3051 (sharper) | 9.0482 -> 9.1685 | -0.1173 |
| uf (reader trained) | 0.3002 -> 0.3125 | 4.1490 -> 4.1460 | +0.0047 |

prog's cell becomes MORE span-specific with depth and the frozen reader punishes it for
exactly that. Arm A's cell collapses toward the corpus mean and is rewarded. Generation agrees
with the cell and not with the CE: prog's OWN-ZERO is +0.0222 [+0.0045, +0.0407], clear of 0,
where arm A's is +0.0054 [-0.0167, +0.0283] and spans it. So on the one instrument with a
reader that is not hostile, prog's cell is MORE read, not less.

Note also `pred.zbar` at depth 1: prog 0.3554 vs arm A 0.3081. prog starts blander and ends
sharper. The knob trades pass-1 quality for per-pass refinement, which is what detaching the
first k passes is supposed to do, and an instrument evaluated at sampled depth integrates over
both halves of that trade and sees nothing: the worth profiles match band for band within
0.05 (arm A shuffle total +0.0502, prog +0.0456).

Instrument change shipped with this writeup: `core_depth_sweep.py` now warns when
`tul.code_target` is set and `coda.` is absent from `training.train_only`
(`tests/test_depth_sweep_frozen_reader_warning.py`, note
`.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md`).

## Verdict

failure. P-G1 and P-G3 both fail. The binding clause on P-G1 fires: one application of the
map reaches the target, the loop's depth is not a training-regime question on this target,
and this knob stops here as a DEPTH lever.

The stop is on the depth claim only, and two measured facts are logged against any future
revival. The knob is not inert: it roughly doubles per-pass movement (confirmed on two
independent instruments) and pays for it at pass 1, netting below baseline at every depth
measured. And P-G3, the prediction that was supposed to decide whether depth pays, was scored
on an instrument that cannot answer it; its failure is evidence the cell got SHARPER, which is
the wanted direction.

## Updated hypothesis

H0 is confirmed on magnitude and H1 on direction. Detaching the first k passes does make the
map improve a state it did not produce, and the effect is real, reproducible across two
instruments, and about fifty times too small to matter. Neither arm extends its useful range:
centred cosine to the own code peaks at depth 6, the training mean, on arm A, prog and uf
alike, consistent with E6's finding that saturation tracks the training draw.

The open question this run creates is not about `progressive_p` and depth. It is that prog
produced a MORE span-specific cell that only the non-hostile instrument could see, while both
CE-based instruments reported it as worse or unchanged. The test is a 10k-step unfreeze
continuation on prog rather than on arm A, scored on generation and on `pred.zbar`, with the
depth sweep read only as a genericity readout. That needs its own prereg; it is not this one.
