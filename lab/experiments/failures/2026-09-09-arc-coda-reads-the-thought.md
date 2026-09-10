# Planned: the coda reads the thought (the unpack arm)

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "we have to test this, add it as an
arm"). Arc: `2026-09-04-loop-contribution-arc.md`. Follows the slot-loop panel
(`2026-09-09-arc-slot-loop-norm-match.md`), whose first two arms read a flat token
K-curve under norm_match (slot loop 0.0000, M-next 0.0000; forecast K1−K6 0.0067).

## Question

Wolfe's reading of today's flat curves: "the design should be that the prelude evolves the
loop state, and the final z + MUX tokens are what conditions the coda. This skipping will
cause the coda to want to ignore the loop." The spec's §3.4 feeds the coda every token's
own GLOBAL prelude state (`x_coda = input_norm(prelude)`), so a token of span i+1 arrives
carrying all of span i inside its own state and `z_i` is two optional cells among ~1,100.
Block Transformer's local decoder gets the block embedding plus the block's own raw
tokens, nothing else, which is what makes its global model load-bearing. Does the slot
loop earn depth when the coda is wired that way?

## Hypothesis

H-unpack-1: with the coda's only route to earlier spans being `z`, the token CE depends on
slot depth (token K1−K6 leaves zero) and the forecast K-curve grows, because the token
loss now trains the loop directly through a reader that cannot bypass it. H-unpack-1′:
the reader problem was necessary but not sufficient: the token curve leaves zero but the
map still converges by pass 3 (K3−K6 stays at 0), so the next lever is the map or the
target, not the reader. H-unpack-2 (the price): CE at 5k sits well behind the unmasked
M-next arm, the mask's known price (0.13–0.20 nats in the gist family), and the arm is
judged on contribution, not on that reading.

## Method

ONE arm, `slot-unpack-norm-match` (`tul_slot_unpack_norm_match.yaml`), on top of the
panel's M-next arm (`tul_slot_mux_norm_match`: think-once panel recipe, seq 1024, batch 6,
seed 1, 5,000 steps, norm_match, MUX M-next β 1.0, gain hinge 0.9, clip-through-time
4.0). Three construction-time changes, all in `morph/model/tul.py` / `transformer.py` /
`attention.py`, each bit-identical at its default (`tests/test_tul_unpack.py`, 9 tests):

| knob | what it does |
| --- | --- |
| `coda_token_input: embed` | the coda's token carrier is `input_norm` of the token's own embedding (the prelude's input `x0`), not the prelude's output; the coda's per-layer injections at the slot cells are zeroed so a cell carries `z` alone |
| `tg_restrict: true`, `tg_restrict_scope: coda` | same-span-or-slot attention in the coda ONLY; the prelude stays global (so `e_z` sees the whole past); in the coda a slot cell attends slot cells only; the CCA Q/K conv taps and the value shift are reset at span/slot boundaries (`segment_causal_conv`), so a cell's key and value carry `z` and not the boundary token's state |
| `bcast: true` | spec §3.5's own unpack row: `z_i` (stream mean) through `span_cap` offset-indexed `d×d` linears, zero-init, added to every token of span i+1's coda input, after the token-state dropout |

The unpack is 33.5 M new parameters (`[32, 1024, 1024]`, d_model 1,024 on this recipe; the
arm has 298.2 M parameters against the M-next arm's 268.2 M), all zero at step 0. The
arm is read on loop CONTRIBUTION, where the extra parameters carry nothing at step 0 and
are one route for the thought at every step after.

The contract, tested at model scale on CPU: with `z` zeroed (`plan_mode zero`) the logits
of span ≥ 2 tokens are invariant to span 0's content; with `z` present they are not; the
shipped path leaks 0.09 nats of logit movement with `z` zeroed. Eager attention (the
restriction is eager-only at construction). Runner `arc/run_slotloop2.sh`, commit pinned
in `arc/SLOTLOOP2_COMMIT`; 12-step smoke, the draw under the sustained tripwire and the
rate stop, then the readouts of the slot-loop panel (sweeps at forced slot depths 1, 2, 3,
6, 9, 12, 16 at 2,500 and 5,000 with the token and forecast curves; `worth_profile.py` at
5,000; the gain traces). Results to `lab/experiments/results/2026-09-09-slot-loop-norm-match/`.
The arm runs FIRST in the new queue; the panel's remaining arms (`slot-mux-absmean`,
`plain-panel-norm-match`) and the two recipe-reads arms follow.

**Amendment 2026-09-09 23:55 (the rate rule).** The first draw ran fully eager
(`model.use_kernels: false` alone) and read 8,076 tok/s at step 200, under the paid loop's
8,086; the runner stopped it (Wolfe's rule) at step 215, healthy, no checkpoint. The
arm re-runs with `model.tg_scoped_kernels: true` (the kernel path of every mask arm since
E16: the prelude and the core fused, the three coda blocks eager because they carry the
mask and the segment reset). Measured before the relaunch on one packed batch at a fresh
init with `W_bcast` randomised (scratch `unpack_kernel_parity.py`): the scoped path is
x1.46 faster per training step (497 vs 727 ms) at 17.8 vs 23.3 GB; its loss differs from
the eager path by 0.055 nats at init (logits mean |diff| 0.010, max 0.25) against 0.0002
nats (mean 0.009, max 0.10) for the MUX arm's kernels-on vs kernels-off. The mean logit
noise is the same; the loss gap is consistent with bf16 rounding on the init identity
logit (about 20 nats on the current token under `coda_token_input: embed`, ulp 0.125),
which this arm alone has. Not proven in fp32. Predictions untouched.

## Predictions (frozen)

- **P-unpack-a (survival).** HEALTHY to 5,000: **70 %** (the mask arms were healthy; the
  new parameters start at zero).
- **P-unpack-b (the reader now reads).** Plan worth at offset 0 (zero the slot) above
  0.5 nats: **80 %** (the slot is the coda's only context; the mask arms read 0.7+).
- **P-unpack-c (the point).** Token K1−K6 over forced slot depth at 5,000 above 0.01
  with the CI above 0: **45 %**; above 0.03: **25 %**. Token K3−K6 above 0.005: **20 %**.
- **P-unpack-d (the forecast).** Forecast (`mux_local`) K1−K6 above 0.02 (M-next under
  norm_match: 0.0067): **50 %**; forecast K3−K6 above 0.005: **25 %**.
- **P-unpack-e (the price, a horizon reading).** 480-row token CE at the trained depth
  behind `slot-mux-norm-match` (4.3290) by more than 0.10: **70 %**; by more than 0.30:
  **30 %**.
- **P-unpack-f (cost).** tok/s at step 200 above 8,086: **80 %** (the mask arm ran at
  9,100 on the eager path; the unpack adds ~15 GFLOP per forward); peak under 20 GB:
  **80 %**.

## Binding

- P-unpack-c TRUE ⇒ the reader was the limiter; the next arm is the same wiring under
  absmean (is the rule needed once the reader reads) and then 20k on Wolfe's word.
- P-unpack-b TRUE and P-unpack-c FALSE ⇒ H-unpack-1′: the coda depends on `z` but not on
  the loop's depth; the map or the target is next (the slot core's own anatomy first).
- P-unpack-b FALSE ⇒ the contract test missed a route; find it before any other arm.
- `RATE STOP` ⇒ the queue stops; Wolfe decides.
- NO run beyond 5,000 steps from this arm.

## Not verified before launch

The real-scale forward ran on CPU (batch 2, seq 256, one thread) for one step; the GPU
path runs first in the queue's 12-step smoke. The unpack's compute cost is estimated, not
measured. The v1 eager generator is unchanged and recomputes the whole row per step; the
arm's cost model at generation is not exercised here.

## Results

Filed 2026-09-10 07:10. `slot-unpack-norm-match`, 5,000 steps under `tg_scoped_kernels`
(the fully eager first draw read 8,076 tok/s and the rate rule stopped it; Method amendment
23:55): HEALTHY, 12,638 tok/s at step 200, peak 15.0 GB, val 4.5791 (`Final val_loss`).
Sweeps (480 rows, forced slot depths 1..16): tokens K1−K6 +0.0004 [+0.0003, +0.0006] at
2,500 and +0.0006 [+0.0004, +0.0008] at 5,000; K3−K6 +0.0000 at both; forecast K1−K6
+0.0024 → +0.0038 [+0.0031, +0.0046], K3−K6 +0.0015 → +0.0011 [+0.0007, +0.0015]. Worth
profile at 5,000: zero z +0.811 [+0.771, +0.851] at offset 0 (then 0.43, 0.32, 0.27, 0.20),
shuffle +1.651 (the shipped path read 0.051 / 0.09). CE at the trained depth 4.4971 against
`slot-mux-norm-match`'s 4.3290 on the same rows: +0.168. Init handicap: the raw-embedding
coda with the weight-tied head starts at 28 nats on the span tokens (the identity path),
the ramp absorbs it. Slot anatomy (`results/2026-09-10-slot-map-levers/`): full-carrier
movement 8.7 % then 3.8/2.2/1.6/1.4/1.2/1.1/1.1 % per pass; core MLP out/in 1.4–2.4 %.

## Verdict

- P-unpack-a TRUE. P-unpack-b TRUE (0.811, above 0.5): the reader reads; the contract holds
  on the trained model.
- P-unpack-c FALSE on every clause (0.0006 / 0.0000). P-unpack-d FALSE (0.0038 / 0.0011).
- P-unpack-e TRUE on the first clause (+0.168, more than 0.10), FALSE on the second.
- P-unpack-f TRUE under scoped kernels (12,638; 15.0 GB); the eager path fails the rate bar.
- Binding: H-unpack-1′. The coda depends on z and not on the loop's depth; the slot core's
  anatomy was the next read and became the levers panel
  (`2026-09-10-arc-slot-map-levers.md`).

## Updated hypothesis

The coda's incentive was a real design error (the spec's §3.4 let the coda skip the loop)
and closing it does not move the depth curve: the coda uses z (0.8 nats) and z is the same
at every depth because the slot map is the injection's geometric pull plus near-inert
blocks. The construction is kept (the knobs are bit-identical at their defaults) for the
finalize decision; it is the honest wiring for a coda that must read the thought, at a
0.17-nat price against the M-next arm at 5k.
