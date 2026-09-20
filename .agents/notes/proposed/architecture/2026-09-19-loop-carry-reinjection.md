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

## Build notes

Written 2026-09-19 with the code, and it records two places where the proposal above was
not precise enough to implement and one place where it was wrong about what is free.

**Where the read is captured.** `r_k(t)` is core layer 0's WINDOW-branch contribution to
that layer's attention output, in `d_model` space: `W_up(g_win * out_win)`, computed in
[`morph/model/attention.py`](../../../../morph/model/attention.py)
`_CCABase._gate_combine_up` under a new `win_capture` argument and requested through a
new `tg_win_capture` attention kwarg. `W_up` is bias-free, so that tensor is EXACTLY what
the layer's output would lose if `out_win` were zero — the window branch's whole
contribution, none of the compressed branch's and none of the residual alpha's. Under
`loop_reach w >= 1` the window branch's XSA excludes the self token, so at layer 0 its
rows hold cells `k-w .. k-1` and nothing else, which is what makes that tensor the
cross-cell read and not "layer 0's output". Layers 1..n-1 run at reach 0, so capturing
layer 0 captures every cross-cell route of the pass. No block forward changed:
`MORPHBlock` already forwards `attn_kwargs` verbatim, and the capture dict travels in the
reach kwargs `_tul_core` already builds for layer 0. The second forward of the window
attention that the brief allowed as a fallback was NOT needed.

The capture is RETURNED from `_core_step` (`want_carry=True`), not read off the dict: a
pass may run inside `torch.utils.checkpoint`, where a side-channel tensor is not safe —
the same reason `ret_state` is returned. The carry likewise enters as an ARGUMENT, never
as a closure variable, because a nonlocal read during the checkpoint's backward recompute
would hold `c(T-1)` instead of `c(t-1)`.

**The normalisation rule, exactly.** With `h` the carrier at the injection site
(`[B, S, n, C]`) and `c` the carry (`[B, S, C]`):

    rms_h[b,k] = sqrt( mean over (n, C) of h[b,k]^2 )
    ms_c[b,k]  = mean over C of c[b,k]^2
    rms_c[b,k] = sqrt( ms_c[b,k] + eps^2 ),            eps = 1e-6
    term[b,k]  = c[b,k] * rms_h[b,k] / rms_c[b,k]      where ms_c > eps^2
    term[b,k]  = 0                                      otherwise

fp32 throughout, cast to the carrier's dtype at the end, then broadcast over the `n`
streams by `_apply_injection`. Both guards are load-bearing and were found by running the
thing: `sqrt(0)` has an infinite derivative and the total model gradient read `nan` on the
tiny fixture without the `+ eps^2`, because cells with an exactly zero carry are not an
edge case — every pad slot for the whole forward, and cell `0` of every row forever (its
window row under a reach budget is empty, so its read is identically zero).

**Named, because it contradicts this tree's habit: the arm is NOT a no-op at
initialisation.** The RMS match cancels any constant scale in front of `c`, so a zero-init
`W_g` cannot make step 0 the ruler's forward — `gate` at `W_g = 0` injects exactly what
`sum` injects at pass 1, and the two modes give the SAME loss on an untrained model
(`tests/test_tul_loop_carry.py::test_sum_and_gate_agree_at_init_because_the_rms_match_cancels_the_half`).
They differ only from pass 2 on. The bit-identity claim of this key is
`loop_carry: "none"`, pinned against `tests/test_tul_prefix_source.py`'s `strict_k2` row.
A learned scale in front of the term would restore a zero-init no-op and is deliberately
NOT added: it is a second mechanism and it would be measured as one.

**A second confound, named:** `gate` adds 2 097 152 parameters (`d x 2d` at `d = 1024`,
+0.7 % of 292.5 M). `sum` adds none. The confound-free reading against the ruler is
therefore `sum`; `gate` is read against `sum`, not against the ruler.

**Instruments.** `carry/rms_t{t}` (the carry's per-cell RMS after pass `t`, mean over
valid slots), `carry/gate_mean_t{t}` on `gate`, and `carry/inject_ratio_t{t}` — which is
1.0 BY CONSTRUCTION and is a check that the match is live, not a reading. It is computed
from the DIFFERENCE of the two carriers `_core_step` goes on to use, not from the term,
because the first sabotage pass showed a version that read the term and happily reported
1.0 while the injection was computed and dropped.

## Risks

- Accumulation can act as a second recurrence with its own gain; the RMS normalisation is the
  guard, and criterion 5 is the check. If it trips, the `gate` variant with a decay
  (`c ← λc + u`) is the fallback, named here so it is not invented after a detonation.
- Re-injecting neighbour content every pass could displace own-span content the way later
  arrivals do today (g = 1 fell 0.181 → 0.137 uncut). Criterion 3 reads it.
- The reach read `r_k(t)` is the window branch's output at layer 0 under `tg_allow`; on the
  register arm (`slot_cells > 1`) the relation differs and the carry is refused there until
  measured.

## Outcome (2026-09-20, measured)

Filed as a failure: `lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md`
(P1, P2, P3 and P5 fail on `sum`; `gate` 0.047 nats better than `sum` at depth 6 and
0.050 behind the ruler). Two things this note got wrong, in the order they matter:

1. The carry REPLACED the pass-1 read instead of adding to it. At depth 1 a carry cell
   holds no neighbour content (planted g = 1 and g = 2 both +0.000 at d1; the ruler reads
   +0.181 and +0.148). Everything arrives at pass 2. The arm was never one factor from
   the ruler.
2. The first risk above happened, and the RMS normalisation did not guard it: the state is
   an unbounded sum (RMS 0.5 → 75 over training, 5 → 75 across six passes at 5k) and the
   first-iteration gain ran 1.24 → 2.8e5 (`gate` 573). The match at the injection site
   bounds the injected vector, not the state; with the read uncut the injected vector at
   pass T is one part in T of each earlier read, so the far bins lose to dilution and get
   worse with every pass after 2 (h = 5 K1−K6 −0.088, h = 6 −0.096; the ruler +0.029 and
   −0.020).

What held: under `--cut-after 1` a re-supplied read stops decaying (g = 2 +0.108 at d2 →
+0.210 at d6, the ruler +0.148 → +0.036). The Problem section's mechanism (decay under
re-processing) is real; the Proposal's vehicle is not. A next carry keeps the pass-1 read
and is bounded by construction (a running mean or an EMA with a fixed target RMS). None is
queued; the lifecycle of this note (rejected, or superseded by a bounded carry) is Wolfe's
call after the `gate` hop table lands.

