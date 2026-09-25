# Agent Note: span-level NextLat on the slot loop's exit state

Status: proposed

## Problem

Stage 3 ([lab/experiments/successes/2026-09-24-lxtul-stage3-map.md](../../../../lab/experiments/successes/2026-09-24-lxtul-stage3-map.md))
gave the first slot-loop arm whose loop moves with the fixed-point term off (`nofp`: coda
K1-K6 +0.0123, rollout separation 3.47 at depth 6). The loop now moves, but nothing asks the
exit state `z_s` to be a state the NEXT slot follows from. Every reader grades `z_s` against
span s+1's tokens alone. No term ties `z_s` to `z_{s+1}`, so the loop's exit can be a
per-span summary with no transition structure. Wolfe's TUL intent ("think once, decode
cheap") needs that structure: a decoder that can draft the next slot state from the current
one and the span it just emitted does not need to run the loop at every span.

NextLat (Teoh et al., arXiv 2511.05963) adds one auxiliary to next-token training. A small
network `p_psi(h_t, x_{t+1})` predicts `stop_grad(h_{t+1})` under SmoothL1. If the token
prediction and the transition are both exact, `h_t` is a belief state (their Theorem 3.2).
Their perplexity gains are mixed. Their drafting gain (3.3x self-speculative at 1.3B) is not.

## Proposal

Built 2026-09-25, not yet run. `tul.nextlat_weight > 0` builds `SpanTransition`
(`morph/model/tul_nextlat.py`): a one-layer GRU started at `z_s` steps through span s+1's
tokens and predicts `z_{s+1}` through an identity-init linear. Loss:
`SmoothL1(z_hat_{s+1}, stop_grad(z_{s+1}))` over valid (s, s+1) pairs, weighted and folded as
`nextlat_weighted`, and removed from the val loss. The token rows come from the detached tied
table. The gradient reaches `z_s`. Seam: `MORPHTransformer._tul_nextlat_loss`. Brief:
[morph/model/CLAUDE.md](../../../../morph/model/CLAUDE.md).

The arm composes on whichever of `nofp` and `fp01` the 2026-09-25 follow-up
([lab/experiments/planned/2026-09-25-lxtul-nofp-followup.md](../../../../lab/experiments/planned/2026-09-25-lxtul-nofp-followup.md))
holds up, and only if the nofp positive replicates. The prereg comes before the run.

## Alternatives considered

- **The paper's per-token form, on the coda's token states.** This is the literal port. It
  shapes the token path, not the slot loop, and the slot loop is the object under test. Not
  built.
- **The paper's KL term through the frozen head.** It needs full-vocabulary logits for both
  states at every slot and span position (about 1.5 k slots x 32 x 49 k per step). The
  tree's readers never build `[.., V]`. Left out. The draft instrument reads the same
  token-space agreement without training on it.
- **Predict `z_{s+1}` from `z_s` alone, with no token input.** That asks the state to know
  the next span before it is emitted. The target would then be the loop's job, not a
  transition's. Rejected: the transition must see the tokens it steps through.
- **Use the transition as the deployed drafter now.** No evidence yet that a drafted state
  reads as well as the loop's. The draft instrument measures that first.

## Acceptance criteria

- The term falls below `tul/nextlat_copy_l1` (the no-change guess) on the val stream.
- `tul/nextlat_draft_gap` is small against the parallel head's CE of span s+2 (the number
  goes in the prereg).
- Coda K1-K6 and paired CE against the base arm are read with the Stage 3 scorer. No verdict
  on a cosine.
- The build is bit-identical to the parent at weight 0 and RNG-neutral at weight > 0
  (`tests/test_tul_nextlat.py`, 15 tests).

## Risks

- The term can pull `z_s` toward a state that is easy to transition from and worse to read.
  Paired CE against the base arm catches it.
- The GRU runs in fp32 over about 1 k sequences of up to 32 steps per step. Its memory and
  wall-clock cost are unmeasured. Trace memory before the queue (e4probe peaks at 16.65 GB).
- A moving target: `z_{s+1}` is live. The stop-grad is the paper's only guard against
  collapse to a constant state. Watch `nextlat_copy_l1` and the state RMS together.

## Outcome log

- 2026-09-25: the arm as built FAILED
  ([lab/experiments/failures/2026-09-25-lxtul-fp01-nextlat.md](../../../../lab/experiments/failures/2026-09-25-lxtul-fp01-nextlat.md)).
  Coda K1−K6 fell from +0.0126 (fp01) to +0.0010. The term collapsed the exit's per-slot
  part 7x (diff_rel 0.230 → 0.032) inside the LR ramp, and the no-change guess met it. The
  stop-grad on a LIVE target is not a collapse guard when the source state has no strong
  reader. Not built, open for Wolfe's call: (a) an EMA copy of the loop as the target
  (BYOL/JEPA), (b) a variance floor on the per-slot part (VICReg), (c) the term with the
  parallel head NOT detached, so a strong reader holds `z` up, (d) close the lane. Keep
  the module: weight 0 builds nothing.
