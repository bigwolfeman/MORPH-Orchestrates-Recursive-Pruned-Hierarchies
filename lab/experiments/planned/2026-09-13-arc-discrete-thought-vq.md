# Planned: the DISCRETE thought — K vector-quantized codes per span instead of one vector

Status: planned

Date: 2026-09-13 (frozen before any GPU step of either arm; no smoke of either exists at
filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). One-factor
partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).
Sibling build, the SAME defect from the other side:
[`2026-09-13-arc-thought-register.md`](2026-09-13-arc-thought-register.md).
Design note:
[`2026-09-13-discrete-thought-vq.md`](../../../.agents/notes/proposed/architecture/2026-09-13-discrete-thought-vq.md).

## The measured defect

A row's written slot states are near copies of each other.

| reading | value | source |
| --- | --- | --- |
| `val/slot_eff_rank` on the one-factor partner | **5.7598** in 1024 dims | `run_slot-spandec-strict.log`, last val |
| `val/slot_pairwise_cos` on the same run | **0.7104** | same |
| the same pair across the slot family | 5.7 – 7.3 / 0.72 – 0.77 | `2026-09-10-arc-slot-mux-prefix4-norm-match.md` (ruler 6.3389 / 0.7388, prefix-4 7.12) |
| the geometry audit | 1.7 – 4.8 | `2026-09-10` audit |
| across 4,906 slots at step 0 | 18 – 23 | `results/2026-09-12-latent-z-gradient/step0-*.json` |

Sixty-four states, each 1024-dimensional, spanning about six directions. Twelve arms have
read token K1−K6 inside [−0.0001, +0.0033] — the partner **+0.0016 [+0.0013, +0.0019]**,
K3−K6 **+0.0002** — across every lever the 2026-09-11 and 2026-09-12 batches tried: core
shape, stability terms, entry, seed, width, HCA, geometry, and ten targets. **None of them
changed what KIND of object a span hands forward.**

## Question

The coda reads TOKENS well and reads the slot channel badly, and a token is a DISCRETE
symbol out of a large alphabet with its own embedding row. If a span hands forward K
symbols instead of one point in R^1024 — read by the coda as K token-like cells — does the
channel carry more, do the passes become worth anything, and does the codebook get used at
all?

## Hypothesis

The rank collapse is a property of the CONTINUOUS write: a shared seed plus a bag mean plus
a loop that contracts gives a cloud of near-parallel vectors, and no continuous lever has
moved it. A quantizer does not have to be argued into giving the thought rank — it gives it
BY CONSTRUCTION. Two spans that pick different codes are exactly as far apart as those codes
are, whatever the loop's continuous states did.

That is the part I am confident about, and it is also the part that is nearly trivial. The
real question is the one the information account
([`2026-09-13-information-view-of-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md))
already answers for the depth half: a deterministic loop adds no information, so the passes
can only pay through the READER, and with the coda at `tg_coda_prefix_reach: all` there is
no relay to create. A bottleneck cannot manufacture a relay either. So I expect the K-curve
to stay flat and I say so below, at high confidence; the interesting predictions are
whether the codes get USED and whether a discrete channel beats a continuous one on CE.

## Method

Two arms, each 5,000 steps, seq 1024, batch 6, seed 1, ramp 1,000, `norm_match`,
`core_fixed_point_lambda` 1.0, prune/carve/route off, retention off,
`tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`, `mux_beta: 0`,
`spandec` on at J 32, `spandec_horizon` 1.

| arm | config | one factor |
| --- | --- | --- |
| `slot-spandec-strict-vq8` | `tul_slot_spandec_strict_vq8.yaml` | `vq_codes: 8` (+ `prefix_k: 8`, forced) |
| `slot-spandec-strict-vq4` | `tul_slot_spandec_strict_vq4.yaml` | `vq_codes: 4` against vq8 |

### What the mechanism is

The loop's exit state `z` — the SAME seam the span decoder, the MUX and SIGReg already read
— goes through `W_vq` to K sub-vectors of width `d_c = 1024/K`; each is snapped to the
nearest entry of ONE shared codebook `vq_E` of C = 512 rows; the K quantized sub-vectors are
lifted back to K full-width cells through `W_vq_out[k]`; cell k is written 1:1 into prefix
cell k through the shared `W_prefix[k]`. The coda is UNCHANGED. The span decoder grades the
DEQUANTIZED thought, the mean of the K lifted codes, so the write is trained by the same
target it is trained by today.

Five decisions, each written down as a decision in
[`morph/model/tul_vq.py`](../../../morph/model/tul_vq.py) with its reason:

1. **Cosine (l2-normalized) codes.** Both sides normalised before the match, so the
   encoder's scale — which nobody has swept — cannot decide the outcome. ViT-VQGAN §3.2.
2. **The VQ-VAE loss, not an EMA codebook.** An EMA book is updated inside the forward, and
   this tree runs forwards with no optimiser step behind them (7 forced depths in
   `core_depth_sweep.py`, 4 modes in `worth_profile.py`, 20 eval batches). Every one would
   step an EMA book and every instrument in this arc would stop meaning what it says.
3. **The dequantized thought is the MEAN of the K lifted codes**, not the sum: a sum scales
   with K and the vq8/vq4 pair would then differ in magnitude as well as in content.
4. **ONE shared codebook** across the K positions and the 4 Hyper-Connection streams, so a
   symbol useful at position 0 can be reused at position 3. `vq_groups: 1` here.
5. **`vq_reset_after: 0`** — the dead-code reset is BUILT and OFF, so a collapsed codebook
   is a result about the mechanism rather than a number the reset manufactured.

Plus one numerical decision that is not a design choice so much as a hazard closed: under
autocast the carrier is bf16, and an `argmax` over cosine similarities in bf16 can flip
between two close codes for rounding reasons alone — a run's code assignments would then be
a property of the arithmetic and not of the state. Everything downstream of the `W_vq`
matmul is pinned to fp32; the lift follows the carrier, as every other Linear does. Guarded
(a bf16 autocast forward gives the same index tensor as its fp32 twin and the two loss terms
come back fp32) and sabotaged (S13).

### What a step-0 model is, relative to the ruler

**Not the ruler, and this arm is the only one in the family that cannot be.** Every other
TUL arm zero-inits its new path so step 0 is its ruler exactly. A quantizer cannot: a zero
`W_vq` gives the encoder no direction to normalise, and a zero `W_vq_out` writes an all-zero
cell so the coda reads nothing at all. At step 0 the coda therefore reads
`W_vq_out[k] @ e_k`, a near-arbitrary unit code picked by a random projection of `z`, at a
per-component rms of 1.0 (the lift's init std is `G^-0.5`, chosen so a lifted cell matches
the scale of the `input_norm`'d field it is scattered into). The ruler at step 0 writes `z`
through an IDENTITY `W_prefix`. **This arm starts behind its ruler by construction and its
early CE says nothing.** Measured on a CPU build at the shipped `d_model` 1024: step-0 loss
13.5196 (ruler), 13.5499 (vq4), 13.7079 (vq8) on the same fixture — the ordering is the
bottleneck, not a bug.

### The packer confound, stated once

`L_total = 1024 + prefix_k · 64`: **1152** at the k=2 partner, **1280** at vq4, **1536** at
vq8 (`TulLayoutSpec.l_total`, `tul_layout.py`; printed by the compose check). The packer
turns the unused slot budget into tokens, so these rows see a different number of real
tokens than the partner's. Against `slot-spandec-strict` the CE comparison is paired on
`tok_index`, which removes the row cut but not the context length. **Neither comparison in
this panel is clean on one factor**: vq8 against the partner differs by the quantizer AND
the prefix width, and vq8 against vq4 differs by the code count AND the code width AND the
prefix width (a 1:1 write forces `prefix_k == vq_codes`). The arm that would make it clean
is a strict ruler at `prefix_k` 8 with no quantizer, and it is not built. Named in every
row, not buried.

### Parameters, measured

CPU build at the shipped `d_model` 1024 (vocab cut to 512 so the build fits, so the DELTA is
the number): vq4 **+4,325,376** against the strict ruler, of which `W_prefix` is +2,097,152
and the quantizer 2,228,224; vq8 **+8,454,144**, of which `W_prefix` is +6,291,456 and the
quantizer 2,162,688.

### The gate, built in this change

`tests/test_tul_vq_thought.py`, **59 passed**. OFF-state bit-identity was proved by RUNNING
the pre-work tree (`f89256d`) and this one on four fixtures and comparing loss, logit sum,
grad sum, `state_dict` key count, parameter count and the whole slot probe — all match to
the last printed digit, and three of the four are pinned inside the gate. Twelve
Thirteen source-level sabotages, each anchored to exactly one occurrence in the shipped
source: **13/13 CAUGHT**, after three MISSED on the first pass and one of them corrected a
claim in the source (see Not verified).

### Readout

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`, paired bootstrap over rows. Partner: **+0.0016 [+0.0013, +0.0019]** /
   **+0.0002**.
2. `worth_profile.py --rows 192 --modes auto` → zero / shuffle / wrong_seed / all_slots by
   offset bin, and the offset-0 bin on its own. **On a strict arm `all_slots` equals `zero`
   by construction** (asserted in the gate at K=4); the partner's `all_slots` total is
   **0.1865**. On math the cells HURT offset 0 by about 2 nats, so the offset-0 bin is
   reported separately and not folded into the total.
3. `tul/vq_perplexity` and `tul/vq_used` from the run's own log — **the first number to
   read on this panel.**
4. `val/slot_eff_rank` / `val/slot_pairwise_cos` over all 64·K LIFTED CODES of a row, and
   `val/slot_cell_eff_rank` / `val/slot_cell_pairwise_cos` WITHIN a slot across its K
   codes. **The probe reads the lifted cells**, which is what the coda holds; reading the
   continuous exit state would report the rank of a tensor no reader on this arm sees.
5. Depth-6 CE paired on 480 rows against `slot-spandec-strict` AND against the other VQ arm.
6. `slot_depth_isolation.py` — per-pass contribution, per slot.
7. tok/s and peak memory at step 200. Partner: 3,337 s, 11,759 tok/s.
8. The matched-compute row: `plain-depth1` at 14 block-passes/token is 0.25 nats AHEAD of
   spandec-mask and 0.33 ahead of the mask ruler at 5k. Every slot filing carries it.


### Method amendment, 2026-09-13 (baseline re-read; predictions unchanged)

The strict ruler's `val/slot_eff_rank` 5.7598 / `val/slot_pairwise_cos` 0.7104 quoted
above comes from the trainer's last `[VAL]` line and did NOT reproduce on the saved
checkpoint: the model's own `tul_slot_state_probe`, run by
`lab/divergence/slot_rank_anatomy.py` on the DGX Spark, reads 11.9597 / 0.5635 on the
480-row probe panel and 13.8466 / 0.5201 on the trainer's own val recipe
(`lab/experiments/results/2026-09-13-rank-anatomy/README.md`). The cause is now measured
(`lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`, commit ab7c353): the run
logged its rank from the PRE-7a24adf probe, which rebuilt the front UNRESTRICTED on a
strict model; the checkpoint reproduces 5.7598 / 0.7104 / CE 4.4249 to four decimals under
that bare front and reads 13.85 / 0.520 under the shipped probe on the trainer's val recipe.
So the "rank 5.76" headline of this lane was an inert-instrument number; the real per-row
rank of the ruler is about 13 of ~50 slots (cosine 0.52), and every `val/slot_eff_rank`
logged before 7a24adf by a front-restricted arm (strict, or tg_restrict at scope all) is a
bare-front number. Scoring rule for every rank prediction in this file: the
partner's baseline is the SAME instrument on the SAME rows as the arm (the sweep's 480
rows, per-row median), never the trainer's logged figure. Rank predictions stated as
"vs 5.7598" are scored against that re-read baseline; the predicted DIRECTION and the
thresholds are unchanged. Also recorded from the same instrument: the prelude makes the
rank (seed 2.24 to entry 12.75 per row), six passes leave it at 13.17, and the prefix
write cuts it to 7.16 while doubling the cell count.

## Predictions (frozen)

Written before any GPU step of either arm. Probabilities are the builder's.

- **P-1 (the codes get used).** `slot-spandec-strict-vq8`'s `tul/vq_perplexity` at 5,000 is
  **above 32** (of a codebook of 512). **60 %.** Reasoning: the cosine match removes the one
  failure mode that kills VQ perplexity for reasons unrelated to the data (an encoder cloud
  that misses the codebook's shell), and at init the CPU build already reads 113.98. Against
  it: the thing being quantized is the SAME near-degenerate state the rank probe measures at
  5.76 effective dimensions, so a codebook fitted to a six-dimensional cloud has very little
  to distinguish; and nothing stops the decoder from learning to ignore the cells, after
  which the commitment term alone decides the assignment. Residual: 25 % in [4, 32],
  15 % at or below 4 (a collapsed book, in which case the arm's other numbers mean nothing
  and the named follow-up is `vq_reset_after`).
- **P-2 (the rank moves, and it is not the interesting part).** vq8's `val/slot_eff_rank`
  over all 64·8 lifted cells is **above 20** at 5,000, against the partner's 5.7598.
  **80 %.** Reasoning: this is close to true by construction — distinct codes are distinct
  vectors — and the only way it fails is P-1 failing with it. It is listed so the panel
  cannot claim a win on a number that was never in doubt.
- **P-3 (the within-slot rank).** vq8's `val/slot_cell_eff_rank` at 5,000 is **above 4.0**
  (of a maximum of 8). **55 %.** Reasoning: nothing forces a slot's 8 code positions to pick
  8 different symbols, and `W_vq_out[k]` differs per position so even a repeated symbol
  lands in a different direction — which pushes this UP for a reason that is about the lift
  and not about the thought. Against it: at 5.76 effective input dimensions there may be
  little for eight positions to disagree about. Residual: 30 % in [2, 4], 15 % below 2.
- **P-4 (the passes still read nothing).** vq8's token **K3−K6 stays below 0.002**, the band
  every strict arm sits in. **85 %.** Reasoning: the information account's `mi_relay_redundant`
  — with the coda at reach `all` the reader already sees every cell, so moving content among
  coordinates it can see is worth exactly zero — and twelve arms in that band. A bottleneck
  changes WHAT the reader sees, not WHETHER it needs the passes to see it. This is the
  theory prediction and its failure would be the interesting result, not its success.
- **P-5 (the first pass).** vq8's token **K1−K6 stays below 0.005**, against the partner's
  +0.0016. **70 %.** Reasoning: the same account, weakened by the one thing that is genuinely
  new — the quantizer makes the write a hard non-linear function of the exit state, so a
  pass that moves `z` across a code boundary changes the write discretely instead of
  slightly. That is a real mechanism for a first-pass effect and it is why this is 70 % and
  not 85 %. Residual: 25 % in [0.005, 0.02], 5 % above 0.02.
- **P-6 (CE against the partner).** vq8's depth-6 CE, token-paired on 480 rows, is **WORSE
  than `slot-spandec-strict` by 0.02 to 0.30 nats**. **55 %.** Reasoning: the strict channel
  carries 0.1865 nats and a K-symbol bottleneck at K=8, C=512 can carry at most 8·log(512) =
  50 nats of raw capacity per span, which is not the binding constraint — but the arm starts
  behind its ruler by construction (step 0 above), pays 8.45 M new parameters in 5,000 steps,
  and quantization is a hard optimisation problem the arm gets one seed and one horizon to
  solve. Residual: 25 % within ±0.02, 20 % BETTER than the partner (which, with P-4 holding,
  would be the most interesting cell in the panel: a discrete write beating a continuous one
  with no change in the loop's depth use).
- **P-7 (the code count is not the lever).** vq8 does NOT beat vq4 on depth-6 CE by more than
  0.02 nats. **60 %.** Reasoning: if the channel's limit is what the loop puts INTO it, 8
  symbols and 4 symbols carry the same nothing and this holds; if the limit is the write's
  capacity, 8 wins and this fails. That is the split this pair exists to make — with the
  caveat that a win could also be the wider prefix, which this panel cannot separate.
- **P-8 (the commitment stays small).** vq8's `tul/vq_commit` at 5,000 is **below 0.30**
  (the term is `1 − cos` averaged over assignments, so it lives in [0, 2]). **70 %.**
  Reasoning: with 512 codes on a sphere in 128 dimensions the nearest-code angle is small
  for any encoder that has settled; at init the CPU build reads 0.0115. A term that RISES
  over training means the encoder is running away from its own codebook, which is the
  failure this instrument exists to name.
- **P-9 (it fits, and it costs).** vq8 runs 5,000 steps without an OOM on the 5090 at batch
  6, and clears **9,000 tok/s** at step 200 against the partner's 11,759. **50 %.**
  Reasoning: `L_total` 1536 against 1152 is a 33 % longer prelude+coda sequence, which is the
  dominant cost; the core stage is unchanged at 64 cells and the quantizer is one matmul
  against a [512, 128] book. The arithmetic says roughly 0.75x. Against it: `slot-mux-prefix4`
  at L_total 1280 cost +2.93 GB, so vq8 at 1536 is the most likely arm in this panel to die
  on memory. Neither figure has been measured.

## Binding

If **P-1 holds and P-4 holds** — the codes are used and the passes still read nothing — the
discrete write is a real channel and the loop still does not need its own depth to fill it.
That is the cleanest statement the arc can make against the capacity reading: the thought's
rank was never the thing stopping the passes. The lane then moves to the READER (cut the
coda's reach and see whether a discrete relay beats a continuous one) and not to the write.

If **P-1 holds and P-4 fails** — K3−K6 above 0.01 with a paired CI clear of zero — that is
the observation the information account says is impossible with the coda at reach `all`, and
the first response is to run the strict leak gate (`tests/test_tul_strict_geometry.py`)
rather than to believe the number. If the gate passes and the number stands, the bridge from
the theorems to the measurement is broken and the quantizer is the thing that broke it, which
would make this the most important arm in the arc.

If **P-1 fails** — the codebook collapses — nothing else in the panel is readable. The named
follow-ups, in order: `vq_reset_after` (built, off), a smaller `vq_codebook`, and
`vq_groups > 1` so a cell carries several cheap symbols instead of one expensive one.

If **P-6 lands in its 20 % tail** — vq8 BEATS the partner on CE while P-4 holds — the
discrete write is worth something the continuous one is not, and the next arm is the one
this panel could not build: a strict ruler at `prefix_k` 8 with no quantizer, to say how
much of it was the width.

If **P-7 fails** — vq8 clearly beats vq4 — the write's capacity is a live lever and the
sweep continues upward, with the `prefix_k` confound resolved first.

## Not verified before launch

* **No GPU step of either arm.** No smoke, no wall clock, no memory figure. Every cost
  number above is arithmetic or a CPU build.
* **Nothing has run in bf16 on a GPU.** The fp32 pin is guarded through CPU autocast, which
  exercises the dtype path but not the CUDA kernels, the fused attention or the real
  magnitudes. The quantizer's own arithmetic on the card is unverified.
* **Checkpoint compatibility was not exercised.** A VQ checkpoint carries three keys no other
  model has, and loading one into a non-VQ model raises on the homeless keys (the tree's
  contract). Both arms train from scratch, so no loader path was tested.
* **The forward has run on CPU at the shipped `d_model` 1024**, forward and backward, with
  no non-finite gradient — but at seq 192, batch 2, vocab 512 and `max_slots` 12, which is
  not the arm's shape. The quantizer's distance matrix at the real shape is
  `B·S·n·K·G × C` = 6·64·4·8 × 512 = 12.6 M entries in fp32 (50 MB) and has never been
  allocated on the card.
* **A claim in the source was WRONG and is corrected, not hidden.** The first draft said the
  straight-through estimator is "the only edge by which the loss reaches the loop". It is
  not: the COMMITMENT term also reaches the encoder, so at `vq_weight > 0` the loop learns
  from two edges. Sabotage S2 (detach the STE) MISSED because of it — the core still read a
  nonzero gradient through the commitment term — and the test now runs at `vq_weight: 0`,
  where the STE really is the only route. Two other sabotages MISSED on the first pass (S4,
  the commitment's stop-gradient, whose VALUE is identical either way; S5, pad slots in the
  usage histogram) and each added a test. 13/13 after the fixes.
* **`vq_reset_after` is built and OFF on both arms.** It has a test but has never run in
  training, and its "re-seed from the worst-served encoder vector" rule has never been
  measured against the usual random pick.
* **The codebook size 512, the commitment 0.25, the loss weight 1.0 and the lift's init std
  `G^-0.5` are choices, not measurements.** Nothing has swept any of them. The init std is
  arithmetic (it targets a lifted-cell rms of 1.0 to match the `input_norm`'d field) and the
  arithmetic has not been checked against the ruler's actual written-cell rms at 5,000 steps.
* **`vq_groups > 1` is built and tested and neither arm uses it.**
* **The `vq8`/`vq4` pair confounds three factors** (code count, code width, prefix width) and
  the `vq8`/partner pair confounds two. Stated in the Method; no arm in this panel resolves
  them.
* **The instruments have not been run on a trained VQ checkpoint.** `core_depth_sweep.py`,
  `worth_profile.py` and `slot_depth_isolation.py` are asserted to RUN on a VQ model by the
  gate (worth-profile modes, the depth lever, the probe) at the 64-dim CPU fixture only.
* **One seed, 5,000 steps, one scale.** Short-horizon CE does not rank arms, and this file
  does not use it to except where a prediction names a paired within-run gap.

## Results

### vq8 (run 2026-09-20)

Run: 5,000 steps at `53c0497` on the 5090 through `run_recon.sh`, started 21:51 local
2026-09-20, DONE 23:00, tripwire HEALTHY (max 58 at step 214), final val_loss 4.4738, RATE
OK 9,450 tok/s at step 200, peak 24.83 GB. An earlier attempt at the same commit exited 1
at step 36 (queue log, 14:50 local); the re-queued run passed the runner's smoke. Artifacts
in `../results/2026-09-13-register/`: `sweep_slot-spandec-strict-vq8_{2500,5000}.json`,
`worth_slot-spandec-strict-vq8_5000.json`, `slot_state_slot-spandec-strict-vq8_5000.json`,
`run_slot-spandec-strict-vq8.txt`, `paired_vq8_5000.{txt,json}`.

**Quantizer at 5,000.** `tul/vq_perplexity` 105.7 (288 at step 0, 65.8 at step 2480, mean
89.0 over the last 1,000 steps), `tul/vq_used` 295 of 512, `tul/vq_commit` 0.0089 (0.0114 at
step 0; flat, not rising). `val/slot_eff_rank` 35.09 over the 64·8 lifted cells,
`val/slot_pairwise_cos` 0.109; `val/slot_cell_eff_rank` 6.72 of 8, `val/slot_cell_pairwise_cos`
0.135.

**Depth sweep (480 rows).** 5,000: d1 4.3751, d2 4.3741, d3 4.3736, d6 4.3736, d9 4.3741,
d12 4.3748, d16 4.3764; tokens K1−K6 **+0.0015 [+0.0011, +0.0019]**, K3−K6 **−0.0000
[−0.0002, +0.0002]**; the span decoder's own CE K1−K6 +0.0038, K3−K6 +0.0005. 2,500: d1
4.6721, d6 4.6702, K1−K6 +0.0018.

**Paired (`paired_vq8_5000.txt`, 501,106 tokens, 490 blocks).** Depth 6: vq8 −
`slot-spandec-strict` **+0.0250 [+0.0217, +0.0283]**; depth 1: +0.0264.

**Worth profile (token-weighted total / offset-0 bin).** `all_slots` 0.1397 / 0.5717 against
the partner's 0.1865 / 0.7567; `shuffle` 0.1263 / 1.2815 (partner 0.1739 / 1.7391);
`wrong_seed` 0.0362 / 0.4132 (partner 0.0426 / 0.6180). The discrete channel carries LESS
than the continuous one at every bin (no clause predicted this; recorded).

**Scoring, vq8 clauses.**

- **P-1: HOLDS.** Perplexity 105.7 > 32; 295 codes in use. The codebook did not collapse.
- **P-2: HOLDS.** Rank 35.1 > 20 (the partner's 5.7598 in the clause is the bare-front
  probe number; the corrected ruler reads 13.85. Either way the clause holds).
- **P-3: HOLDS.** Within-slot rank 6.72 > 4.0.
- **P-4: HOLDS.** K3−K6 −0.0000 < 0.002.
- **P-5: HOLDS.** K1−K6 +0.0015 < 0.005 (partner +0.0016). No first-pass effect from the
  hard code boundary.
- **P-6: HOLDS.** +0.0250 worse, inside [0.02, 0.30]; the interval's lower end (+0.0217)
  clears the 0.02 floor.
- **P-8: HOLDS.** Commitment 0.0089 < 0.30, flat.
- **P-9: HOLDS.** No OOM at batch 6 (24.83 GB); 9,450 ≥ 9,000 tok/s.
- **P-7:** waits on `slot-spandec-strict-vq4` (queued 2026-09-20 at the same commit,
  started 23:15 local).
