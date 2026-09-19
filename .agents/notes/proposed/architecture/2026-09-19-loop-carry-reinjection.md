# Agent Note: re-inject what a cell read from its neighbour, every pass, like the own-span term

Status: proposed

## Problem

The hop-distance probe's second pass (`lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md`)
measured why the strict slot loop earns so little depth even where the geometry forces a hop.
On `slot-spandec-strict-prev-reach1` @ 5000 content h spans back arrives at pass h − 1, but
what arrives decays under the cell's own later passes: with the reach cut so that nothing
arrives after pass 0, a planted copy two spans back falls from 0.148 to 0.035 nats of benefit
between depth 1 and depth 6, and the tokens that depend on that span lose 0.083 nats. The
cell's OWN span does the opposite under the same cut: its planted benefit grows from 0.181 to
0.291. The difference between the two is where the content sits. Own-span content is
re-supplied every pass (the per-layer x0 / bigram injection `inj_terms`, and `tul.reread`
where it is on); neighbour content enters once, through the layer-0 reach attention, and
then lives only in the recurrent carrier, which the core re-processes every pass. The loop
therefore earns +0.064 at h = 3, +0.054 at h = 4, +0.029 at h = 5 and nothing at h = 6: the
farther the source, the more passes its content spends decaying before the coda reads it.
The toy `eliminate6` shows the same decay on an untrained six-hop chain
(`lab/experiments/successes/2026-09-19-toy-eliminate6-hop-distance.md`).

## Proposal

Give neighbour content the footing own-span content already has: a per-cell carry that is
written when the cross-cell read arrives and re-injected at the entry of every later pass.

- `tul.loop_carry: none | sum | gate` (default `none`, bit-identical to today; unknown values
  raise in `tul_setup`, like every `tul.*` key).
- At pass t the existing layer-0 reach read of cell k is `r_k(t)` (the window-branch output
  of core layer 0 restricted to cells k − w … k − 1; it is computed today, it just is not
  kept). The carry is `c_k(t) = c_k(t − 1) + u_k(t)` with `u_k(t) = r_k(t)` under `sum` and
  `u_k(t) = σ(W_g [r_k(t); c_k(t − 1)]) ⊙ r_k(t)` under `gate` (`W_g` zero-init so the gate
  starts at ½ and `gate` equals half of `sum` at step 0). `c_k(0) = 0`.
- The carry is added to the carrier at the entry of pass t + 1 through `_apply_injection`,
  exactly where `TULReread.read` adds its term (`_tul_core`, the `_rr` block), and it is
  RMS-normalised to the carrier's scale before the add so accumulation cannot grow the norm
  (`norm_match`'s rule for tensors, applied to a state).
- The carry is part of the per-pass state, so the forced-depth probes, the fixed-point term
  (applied to the carrier as today) and the per-slot depth sampling see nothing new. It is
  reset per forward. No new parameter under `sum`; one `d × 2d` gate under `gate`.
- Instruments: the per-hop K-curve and the planted pair's decay row are the score
  (`lab/divergence/hop_distance_probe.py`), not a whole-arm K1−K6, since a whole-arm mean of
  +0.015 hid +0.064 and −0.020 in the first pass.

## Alternatives considered

- **Width** (d192 vs d96 solved 8/15 vs 2/15 in the toy). Cheap to queue, but it raises how
  many hops survive without changing why they decay; it is a second arm, not the first.
- **A per-pass write gate on the cross-cell read** (gate what enters, keep the carrier as
  the only store). Rejected as the first arm: it still leaves the carried content in the
  re-processed carrier, which is the measured failure.
- **`tul.reread` extended to neighbour cells** (re-read the neighbour's prelude state each
  pass). Rejected: it re-reads the neighbour's ENTRY, not what the neighbour's loop has
  computed, so nothing beyond one hop can accumulate; the chain would flatten to h ≤ 2.
- **A fixed-point term on the carry.** Not needed for the first arm; the carry is additive
  and normalised, and the carrier's term stays on.

## Acceptance criteria

All on `slot-spandec-strict-prev-reach1` geometry (coda reach `prev`, loop reach 1) at 5000
steps against the ruler's own probe readings, 480 rows, same instrument, prereg first:

1. Planted g = 2 under `--cut-after 1`: benefit at depth 6 within 0.03 of its depth-1 value
   (ruler: 0.148 → 0.035). This is the mechanism clause; if it fails the carry is not doing
   what it is for.
2. Per-hop K1−K6: h = 5 ≥ +0.045 and h = 6 ≥ +0.010 (ruler +0.029 and −0.020); h = 3 and
   h = 4 not below the ruler's minus 0.010.
3. Own-span content not paid for it: h = 1 within 0.005 of the ruler and the g = 0 pair
   benefit within 0.05.
4. Bit-identity at `loop_carry: none` (the `tests/test_tul_prefix_source.py` constants) and
   a contract test that under `sum` at forced depth T the carry at cell k equals the sum of
   its T reach reads.
5. Stability: no detonation and `loop/core_gain_t0` under the ruler's band over the 5000
   steps; the smoke prints the same peak memory within 5 %.

## Risks

- Accumulation can act as a second recurrence with its own gain; the RMS normalisation is the
  guard, and criterion 5 is the check. If it trips, the `gate` variant with a decay
  (`c ← λc + u`) is the fallback, named here so it is not invented after a detonation.
- Re-injecting neighbour content every pass could displace own-span content the way later
  arrivals do today (g = 1 fell 0.181 → 0.137 uncut). Criterion 3 reads it.
- The reach read `r_k(t)` is the window branch's output at layer 0 under `tg_allow`; on the
  register arm (`slot_cells > 1`) the relation differs and the carry is refused there until
  measured.
