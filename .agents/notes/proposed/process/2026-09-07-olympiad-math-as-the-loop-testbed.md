# Agent Note: Olympiad math as the loop-contribution testbed, then back to NLP

Status: proposed

Date: 2026-09-07. Wolfe's direction after the September 4–7 loop-contribution arc.

## Problem

Fourteen experiments on web text (`lab/experiments/planned/2026-09-04-loop-contribution-arc.md`)
found that every lever on the slot loop (the gain target, the draw's mean, per-iteration
operators, the state renorm) moves how much of the tokens' work the loop CARRIES and never
how DEEP it works: every stable loop finishes by iteration 3, the tokens read the loop only
when the TG mask removes their direct route, and the end point never moves. Web text at
this scale may simply have nothing to iterate on. Wolfe: "looped models generally work
better with thinking problems"; Olympiad-AI's synthetic math is the testbed that needs it.

## Proposal

1. **Testbed.** Olympiad-AI (`/mnt/sda1/Projects/00NN/Olympiad-AI`, `docs/olympiad-interop.md`):
   every document is a question, numbered reasoning steps one per line, and an answer
   block. MORPH's boundary rule cuts one span per step (mean 12 tokens), so a slot IS a
   step, and the answer block gives a region whose CE and accuracy say whether the loop
   helped solve the problem. Shards `olympiad_math` (stages 2–10, 291M tokens) and
   `olympiad_stage11_13` (367M) through the curriculum loader (`oly_data.yaml`); held-out
   rows from the shards' `eval_holdout.jsonl`; readout `lab/divergence/olympiad_sweep.py`
   (token CE, answer-region CE and accuracy, per-stage-band curves, paired over documents).
2. **Runs are short.** 6,000 steps, not 15,000 (the first launch at 15k projected 9.7 h for
   the mask arm; Wolfe: "way too slow"). Every sample is ≤ 512 tokens; seq 512.
3. **The data's own curriculum is used.** Olympiad stages start at 2.1 and climb; the run
   should progress through them in order instead of a uniform mix, and the sweep reports
   per-band accuracy at every checkpoint so the progression is visible. How the loader
   expresses the order (per-stage blends over per-band shards, or stage-ordered reading)
   is decided after the audit below.
4. **Optimize first.** A pit stop before the relaunch: a throughput audit of the training
   step on this configuration (the eager TG-mask attention at 640 positions, the slot
   loop's 16 full-BPTT iterations and its hinge's two extra core applications, gradient
   accumulation, the per-step gradient probe, the eval cadence, the loader). Measured
   options, ranked by steps/s per GB, then the relaunch.
5. **Then NLP again.** Whatever earns depth on math goes back to web text (with a math
   replay fraction, Wolfe's call) to confirm it was the data and not the arm.

## Alternatives considered

- Keep sweeping levers on web text. Rejected: the arc closed that lane (E1, E2, E6, E7,
  E12–E14 all move share, not depth).
- A synthetic algorithmic task written for the purpose (parity, graph reachability).
  Rejected for now: Olympiad-AI already exists, is MORPH-compatible (vocab, shard format,
  loader), has graded difficulty and held-out sets, and is the data the seed model will
  train on anyway.
- 15k steps at the first launch's rate. Rejected: 9.7 h per arm, ~20 h for the panel;
  the first checkpoint of the killed run showed loss 4.5 by step 400, so 6k is enough to
  read a progression.

## Acceptance criteria

- The relaunched panel (prereg to be written after the audit) runs the mask arm in ≤ 3 h
  for 6k steps with the same per-step tokens or more.
- The sweep reports per-band answer accuracy at every checkpoint for every arm.
- The verdict on depth is the mask arm's answer-region K3−K6 with a paired CI over
  documents, against the plain control.

## Risks

- The curriculum order interacts with the ramp and the tripwire (an easy first band may
  hold the loop at a trivial fixed point; a band switch may spike).
- The mask arm is eager-only; if the audit finds no kernel path for the TG mask, its cost
  stays ~2x the other arms.
- Olympiad-AI's drive is 95 % full; new per-band shards need space (97 GB free).
