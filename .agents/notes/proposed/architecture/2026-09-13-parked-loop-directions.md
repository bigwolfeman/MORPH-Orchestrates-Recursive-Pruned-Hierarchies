# Agent Note: parked loop directions from the 2026-09-13 literature mining

Status: proposed

## Problem

The 2026-09-13 mining of 26 loop papers
([`docs/references/looping-depth/2026-09-13-lit-mining/README.md`](../../../../docs/references/looping-depth/2026-09-13-lit-mining/README.md))
produced a menu of arms. Wolfe read LoopMTP (arXiv 2608.03624) and decided it "directly
addresses seemingly everything we need it to": the panel narrows to LoopMTP on the token
loop (`2026-09-14-arc-loopmtp-token-loop.md`, in build), its amortized slot-loop port
(`2026-09-14-arc-horizon-passes.md`, in build), and four zero-training diagnostics on the
Spark (`2026-09-14-arc-loop-diagnostics.md`, in build). Three builders were stopped mid-work
on 2026-09-13 23:05. This note records those and the rest of the menu so nothing has to be
re-derived if LoopMTP does not close the question.

## Proposal (the parked directions, with the design each builder was given)

**Stopped mid-build, no commits:**

1. **Loop-time attention over the pass trajectory** (RecurTrace, 2609.03379 §3, Eq 4-5).
   Before pass k >= 2, each slot attends along LOOP TIME ONLY over its own states from
   passes 1..k-1: single head, own projections, relative-pass-distance bias, sigmoid gate
   zero-initialised so it is an exact no-op at init. Knob `tul.traj_attn`; `_ternary_exclude`
   on every leaf. Instrument: whether the consecutive-update cosine leaves the ruler's
   −0.2 to −0.6 band. Prior 25 %. Reuse the pass-state collection `db_traj` /
   `prefix_source: trajectory` already keep. Gate: off == bit-identical, no cross-slot or
   token leak under perturbation, strict leak cut holds, memory unchanged when off.
2. **Decode-then-encode realignment** (DiscoLoop, 2607.00341, Eq 3-6) at pass boundaries
   and at the prefix write: z through the DETACHED tied head to a softmax over the
   vocabulary (temperature knob), re-embedded as the softmax-weighted sum of tied rows, mixed
   back by a zero-init per-slot gate (d+1 params per boundary). Knob `tul.realign:
   none|passes|write|both`. Run its probe FIRST: `lab/divergence/slot_alignment_probe.py`,
   per slot P(target | z) and logit-lens rank vs cos(z, target embedding); DiscoLoop's
   signature is high P with cos ~0.27-0.33 (Table 1b). The arm is worth building only if the
   probe fires on the strict ruler checkpoint. Prior 40 % conditional on the probe.
3. **Full-bandwidth latent feedback on the plain model** (2608.08888, §3.1-3.3): position
   t's stack input becomes W_U h^L_{t-1} · sigmoid(W_G e_t) (Eq 4), the previous position's
   final state on the value pathway, zero-init so pass 1 == the plain model; trained by
   temporal parallelism (pass k re-runs the stack with pass k-1's shifted top states; loss
   CE(pass 1) + 1/(K-1) Σ CE(pass k), not detached, Eq 12) with the paper's 75/22/3 mix of
   1/2/3-pass batches from a start step; eval by a forced-K sweep; < 1 % per-token decode
   cost. This is the one COMPUTE-REDUCING design besides the depth ladder: 2x data
   efficiency at 1B. The builder's partial module (untracked, ~1 file) is kept at
   `ignore/parked/latent_feedback_partial_2026-09-13.py` (private scratch); the design above
   is the record. Prior 55 % that it beats the ladder's depth-1 rung by > 0.02 at 20k.

**On the menu, never started:**

4. Per-pass routing: the tile router conditioned on the pass index, active from step 0 on
   the slot core (Sparse Layers, 2605.09165, Fig. 5: 25-53 % of tokens get disjoint experts
   between passes). Instrument: tile-overlap histogram pass k vs pass 1. Build risk: the
   router runs post-carve in this tree.
5. Register with per-cell horizons: cell c owns span i+c (the k > 1 demand the register
   never had). Cheap once the horizon arm exists.
6. Energy + diversity objective with a randomized entry (Bifurcation Models, 2605.07277,
   Alg. 2 + Table 2): K parallel rollouts in training, one at inference. Prior low on web text.
7. Corruption of the loop's INPUT at low pass index (Denoising Recursion Models,
   2604.18839): our staged arm corrupted the target, theirs corrupts what the loop reads.
8. Anchored bounded-residual two-timescale refiner (Latent Recurrent Thoughts, 2609.01117).
9. Deflation targets: pass k graded on the residual after passes 1..k-1 (from the
   power-method proof, 2606.00605).
10. State-prediction separation (Free Pause Tokens, 2609.03807): a query-only second stream
    over the backbone's KV, 0.028 nats at 1B, still ahead at iso-compute.
11. Stochastic variational loop (GRAM, 2605.19376). Not amortized; prior low.

**Free instruments (in build under the diagnostics prereg):** asymptotic-alignment score,
per-row σ1/σ2 vs K1−K6, attention-sink mass per pass, basin map.

## Alternatives considered

- Let the three builders finish anyway: rejected by Wolfe ("not worth doing anything else");
  every extra knob is one more thing to keep bit-identical when off.
- Keep the partial relay module in the tree: rejected, an untested module in `morph/model/`
  is the kind of junk the finalize rule removes; the design is recorded here and the file
  is in private scratch.
- Record only LoopMTP: rejected, this note exists precisely so the menu survives a null.

## Acceptance criteria

Any of items 1-11 gets its own prereg before a GPU step; this note is superseded by that
prereg for that item. If the LoopMTP token-loop arm shows depth 3 matching plain depth 6 at
20k, items 3 and 10 are the next compute-side reads; if it does not, item 3 is the next
build.

## Risks

The designs above are from reader summaries of the papers, PDF-checked only for the
Full-bandwidth and LRT items; each builder must re-read its paper before building.
