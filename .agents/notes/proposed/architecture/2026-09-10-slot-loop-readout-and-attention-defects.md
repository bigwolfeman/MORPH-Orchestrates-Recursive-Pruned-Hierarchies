# Agent Note: Two measured defects on the slot loop's readout and attention

Status: proposed

Date: 2026-09-10. Source:
[`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`](../../../../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md),
findings F1 and F2, measured live on three step-5000 checkpoints. Arms:
`slot-mnext-staged-fullread` (`morph/configs/tul_slot_mnext_staged_fullread.yaml`) and
`slot-mux-hca-fix` (`morph/configs/tul_slot_mux_hca_fix.yaml`). Sibling note on the loss
side: [`2026-09-10-credit-assignment-in-the-slot-loop.md`](2026-09-10-credit-assignment-in-the-slot-loop.md).

## Problem

The slot-geometry audit set out to answer why ONE pass of a six-block core changes nothing
the coda reads. Its verdict is that the core is not muted: one pass moves a slot state as
far as it moves a token state, and the prefix write hands the coda a cell that is 1.07x
(mux arm), 6.8x (unpack arm) and 802x (noise arm) the entry's own contribution. The
flatness is direction, not magnitude, and that is a credit-assignment result.

Along the way it confirmed two REAL defects, neither of which is the cause of the flat
K-curve and both of which are wrong on their own terms. They are written down here because
"not the cause" is not "fine", and because each has a one-line fix that nobody has run.

**F1 — the HCA compressed branch is exactly dead at the slot budget, and the learned gate
keeps paying for it.** `GatedPoolCompressor.forward` (`morph/model/attention.py:310`)
computes `n_blocks = S // m` and returns an empty stream when `S < m`. The slot loop runs
the core on 64 cells with `hca_compress_ratio` 256, so `n_blocks = 0`, `fused_hca_attention`
attends nothing, and `_CCABase._gate_combine_up` (`attention.py:840`) blends the gate weight
into a zero tensor. Measured at step 5000 on `slot-mux-norm-match` and
`slot-mnext-noise-entry`: `|out_comp|` is exactly `0.000` on core blocks 1, 3 and 5 with
`g_comp` 0.42-0.52 there, while the SAME weights on a 1,152-position token sequence output
680-830 at the same gate value. Three of six core blocks deliver about half the attention
output they were built for, silently, for a whole run. The cost shows up as an attention
branch on those blocks at 0.32-0.43x the token figure against 0.56-0.94x on the CSA blocks;
`slot-unpack-free`, which routes the compressed branch elsewhere and never calls the pooled
compressor, reads 1.01-1.12x on the same blocks. `model.core_hca_compress_ratio` exists
for exactly this and has never been used outside `tul_a1_hca16.yaml`, which never ran.

**F2 — the MUX head and `TULSlots.unpack` read the stream MEAN; the reader that feeds the
coda does not.** `TULSlots.unpack` does `z = h_slots.mean(dim=2)` (`morph/model/tul.py`),
and the MUX head reaches the same reduction through `_tul_mux_loss` → `_readout`, which
does `x = x.mean(dim=2)` before `lm_mixer` and `final_norm`. `TULSlots.prefix_project` does
NOT: it projects every stream separately and the coda sees all four. Measured on
`slot-unpack-free` at step 5000: the ENTRY state survives the stream mean at 0.972 of its
per-stream norm and the loop's UPDATE at 0.139. On that arm the two readers therefore see
very different objects — the prefix cell carries 6.8x the entry's contribution while the
MUX target sees a seventh of that. On the other two arms the mean treats entry and update
alike (0.581 vs 0.577; 0.500 vs 0.818), so the size of the loss is arm-specific; the
asymmetry between the two readers is not.

## Proposal

Two one-factor arms, each fixing one defect and nothing else, so that "F1/F2 are not the
cause" stops resting on an inference from a third arm and starts resting on a repaired one.

**Arm A, `slot-mux-hca-fix` (`model.core_hca_compress_ratio: 16`), on the ruler
`slot-mux-norm-match`.** 64 // 16 = 4 blocks, which is what the token path gets at
`seq_len` 1024 (1152 // 256 = 4), so the core's compressed branch sees the same number of
blocks on both shapes. Scoped to the CORE: setting `model.hca_compress_ratio` globally
would re-block the prelude and coda HCA layers too, which run on all 1,152 positions and do
not have the problem, and the arm would differ by seven modules instead of three. Declared
confound, unchanged from `tul_a1_hca16.yaml`: `B_a` is `[m, c]`, so the arm has 15,360
FEWER parameters per HCA core block (46,080 in total against 268 M, 0.017 %) — not
iso-parameter, in the direction that makes an improvement harder to explain as capacity.

**Arm B, `slot-mnext-staged-fullread` (`tul.mux_readout: full`), on the staged arm
`slot-mnext-staged`.** The MUX head applies `lm_mixer` and `final_norm` per stream and
averages the results, which is the tied head applied to each stream with the logits
averaged. Everything after the streams is linear, so `mean_n (z_n W') == (mean_n z_n) W'`
and the implementation costs one `[B, S, V]` matmul, not `n` of them. The ONLY difference
from the shipped readout is where the RMS normalisation sits: one per stream, or one after
the mean. `mux_readout: "mean"` is bit-identical to the tree before the knob (loss
`9.359314918518066`, sha256 `189911591a8f6…461aefc0` over all 208 gradient tensors of the
tiny CPU model at master `cffe3bb` and after the change), and three sabotage runs fail the
suite.

Bound, stated before the run: on an UNTRAINED model the two readouts differ by about 3e-6
nats, because the HC streams start almost equal. The knob acts only once the streams
diverge, which they have by step 5000.

## Alternatives considered

- **Leave the MUX readout alone and change `prefix_project` to take the mean instead**, so
  the two readers agree the other way. Rejected: the mean is the lossy side. The audit
  measured the prefix write delivering the loop's update to the coda at gain >= 1 on every
  arm, and the mean discarding 86 % of it on one; making the good reader match the bad one
  removes the asymmetry by removing the information. It would also change the tensor the
  coda consumes, which is a much larger change than a loss-side readout.
- **A learned `[n·C -> C]` projection on the flattened carrier**, the audit's own "smallest
  fix if it is ever load-bearing". Rejected as the FIRST arm: it adds parameters (4·1024·1024
  = 4.2 M at the panel width) and a new init, so a CE or K-curve difference has a capacity
  explanation as well as a geometry one. The per-stream readout adds nothing and answers the
  narrower question — does the head see more when the streams are normalised separately —
  which is the one F2 actually poses. If `full` moves anything, the learned projection is
  the follow-up.
- **A per-stream HEAD (n separate unembeddings)**. Rejected: the head is weight-tied to the
  input embedding table (`lm_weight()` IS the embedding), so "per-stream head" means either
  n embedding tables or n untied heads at 49,169 × 1024 each. Both are large, and neither is
  needed to test whether per-stream normalisation changes what the head sees.
- **Changing `unpack` too.** Not done, and not a gap: `unpack`'s only operation on `z` is
  the linear `W_bcast`, and a linear map commutes with the stream mean, so a per-stream
  variant there is algebraically the same tensor. Making `unpack` behave differently
  requires the learned projection above. The fullread arm does not run `bcast` in any case.
- **Fixing the HCA ratio GLOBALLY (`model.hca_compress_ratio: 16`) rather than for the
  core.** Rejected: the prelude and coda run on 1,152 positions where 256 gives 4 blocks and
  the branch is healthy (`|out_comp|` 680-830). Re-blocking them changes seven modules to
  fix three and confounds the arm.
- **Fixing F1 inside `GatedPoolCompressor` (pad to one block, or fall back to a dense
  branch when `S < m`).** Rejected: the empty return is deliberate and load-bearing — the
  `two_stream=False` path reaches the same shape naturally, and a `F.pad` on the empty block
  dim invents a block that the gate tensor does not have, which is the 2026-08-18 generation
  crash. The defect is a CONFIG mismatch between a ratio and a sequence length, and the
  config knob for it already exists.
- **Doing neither, on the grounds that the audit already shows F1 is not the cause.** That
  argument rests on `slot-unpack-free`, an arm that differs from the ruler in its coda, its
  objective and its stability terms as well as in its compressed branch. One repaired ruler
  is worth more than that inference, and three of six core blocks running at half strength
  is worth fixing whatever the K-curve does.

## Acceptance criteria

- `mux_readout: "mean"` is bit-identical to the pre-change tree: loss, base weights and
  every base gradient. Verified, cited above.
- The `full` readout is proven to ACT and proven to be the RIGHT object: it equals the hand
  computed per-stream readout to 1e-6, it changes the MUX loss when the streams differ, and
  it collapses onto `mean` when all four streams are identical. Three tests, and three
  sabotage runs that fail them.
- The stream-SUM definition is recorded as identical to the mean by a test, so the choice
  between the two candidate definitions is a measured fact and not a preference.
- The HCA fix is proven to reach the CORE and only the core: at S = 64 the core's HCA
  compressor gives >= 2 blocks and a non-zero compressed stream at the new ratio and zero
  blocks at the old one, while the prelude and coda compressors keep the old ratio.
  `tests/test_tul_hca_fix.py`.
- Each arm reports its K-curves against its ruler and says flat when flat. Neither arm is
  scored on CE at 5,000 steps (Wolfe 2026-09-09).

## Risks

- **Both arms read flat.** The modal outcome, and the audit already predicts it for F1: the
  one arm with no dead branch reads K1−K6 0.0018. The value of running them is that "not the
  cause" becomes a measurement on a repaired arm instead of an inference across arms.
- **The HCA arm's parameter change gets read as the effect.** It has 0.017 % FEWER
  parameters, so an improvement cannot be capacity; a REGRESSION could be, and would have to
  be reported as ambiguous.
- **The fullread arm's mechanism is weaker than F2's headline number suggests.** The 0.139
  survival figure is from `slot-unpack-free`, an arm with no stability terms whose loop grows
  7.3x; the arm this knob runs on reads 0.577 against 0.581, which is nearly symmetric. Per
  stream normalisation also cannot undo an exact antipodal cancellation. The honest framing
  is that this tests the READER asymmetry, not that it recovers 86 % of anything.
- **A per-stream `final_norm` changes the scale the head sees**, which interacts with
  `mux_tau` (1.0 on these arms) and with the tied head's own scale. Untested beyond the
  tiny CPU model; the smoke's first `mux_local` value is the first real reading.
- **Neither arm has run on a GPU.** CPU builds, CPU tests, sabotage runs and config compose
  checks only.
