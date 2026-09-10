# How good can z be? A gradient-optimised slot state on three slot-loop checkpoints (2026-09-10)

Instrument: [`lab/divergence/slot_z_optimize.py`](../../../lab/divergence/slot_z_optimize.py).
Test: `tests/test_slot_z_optimize.py`.
JSON artifacts: `ignored/experiment-artifacts/2026-09-10-slot-z-optimize/<label>.json`
(one per arm, with the console log beside it).

## The question

The slot loop writes one state `z` per span through `tul.W_prefix` into that span's two
coda cells, and the coda reads it. This morning's per-pass gradient probe
([`2026-09-10-slot-gradient-probe`](../2026-09-10-slot-gradient-probe/README.md)) says the
token CE puts about 1 % of the prelude's gradient into the loop. Before asking HOW the
loop should build z, this asks what a GOOD z is worth: freeze the trained model, make z a
free variable, and minimise the trainer's token CE over z alone.

Two possible readings, decided by one number:

* a gradient-optimised z beats the loop's z by a lot ⇒ the coda CAN use z, and the loop is
  a poor inferencer;
* the gain is small ⇒ the coda cannot use z, and no loop will fix it.

This is Latent Thought Language Models (Kong et al. 2025) used as an INSTRUMENT — their
per-sample latent inference at test time, run here on a frozen model to read a ceiling —
and not as a training method.

## Method

**Split point.** For a slot-loop model `_forward_tul` runs

```
x, x0, bigram          = _tul_front(...)
xn, h_slots, ...       = _tul_core(...)          # <- z
[mux loss] [_tul_plan_ablate]
values, pos            = tul.prefix_project(h_slots, layout, L)   # <- the prefix write
x_coda                 = scatter_positions(base, pos, values)
[tul.unpack(h_slots)]                             # the bcast reader, unpack arm only
xh                     = _back_region(x_coda, ...)
groups                 = _tul_group_losses(xh, labels, layout)    # <- the token CE
```

z is `h_slots` as `prefix_project` receives it, shape **`[B, S, n, C]` = `[3, 64, 4, 1024]`**
— the FULL Hyper-Connection carrier, all four Cayley streams, nothing collapsed.
(batch 3 and `max_slots` 64 from the packer's own startup line, `hc_streams` 4 and
`d_model` 1024 from `base.yaml`.) On these
three arms nothing sits between the `_tul_core` return and the write (no think-once
conditioning stack, no gate, `detach_z` false), and the probe refuses to run when
something does. The identity is CHECKED, not read off the source: every replay asserts
that `prefix_project` receives the substituted object itself.

The probe caches the front and the core and re-runs EVERYTHING downstream through
`_forward_single`, so the prefix write, the bcast unpack, the token-state dropout seam,
the TG masks, the coda and the weighted CE are the model's own code, not a copy.

**Objective mask.** `_tul_group_losses(...)["loss"]` — the trainer's ONE weighted CE,
taken BEFORE the sigreg / gain / mux terms `_forward_tul` adds to its copy of the dict.
That is the §5 double label: `emit_weight` 0.0 on the slot's emitting cell (so the slot
carries no loss of its own), `plast_weight` 1.0 on the span's last token, 1.0 everywhere
else, −100 pads ignored, the `slot_id` logit masked out of the head. The whole objective
therefore lives at TOKEN positions, and it is exactly the term the arm trained on.

**Mode.** `model.eval()`: dropout off, token-state dropout off, the eval depth branch.
This is an instrument on a FIXED function — with dropout on, every optimisation step
would descend a different objective. Every slot is forced to `--depth 6`
(`tul.slot_depth_fixed`) — a PIN, not a change: at eval `_sample_slot_depths` already
returns the deterministic `mean_depth`, which is 6 on these arms. `model.slot_gain_lambda` is set to 0 because the hinge
applies the core step twice more; it is a probe and does not change `h`. bf16 autocast, as
the trainer. All parameters `requires_grad_(False)`; z is optimised as an fp32 master and
cast to the core's dtype on every step, so the model sees exactly what the recorded
forward produced while Adam's moments stay fp32. Pad / invalid slots are frozen by
zeroing their gradient.

**What "optimised" means, and what it does NOT mean.** The optimiser sees the CE of the
tokens the coda predicts AFTER slot i, and z_i is fitted on exactly those tokens. A causal
inferencer cannot see them — the slot's own span is all the loop is given. So `ce_zopt` is
an ORACLE upper bound: it bounds the CODA's capacity to use z through this frozen write.
It says nothing about what the loop could reach. The M-next MUX target asks the loop to
forecast the next span; this hands it the answer.

**Causality gate (measured, not argued).** Causal attention puts every slot cell after the
tokens of span 0, so no z can reach them. `before_slot0` is that bucket, and its CE must
not move under the optimisation. It is reported per arm below; it is exactly 0 everywhere.

**Controls in the same run.** `ce_zero` and `ce_shuffle` go through the shipped
`_tul_plan_ablate` (`plan_mode` zero / shuffle), so the zero and the within-row permutation
are the model's own ablations. `ce_entry` substitutes `h_0 = core_init(e)`, the loop's
entry state — what the coda would read after zero passes. `ce_zopt rand` starts from a
random z with the same per-slot norm as the loop's and optimises for the same number of
steps: it says whether the loop's z is a useful starting point or the coda can be driven
from anywhere.

**Bucket split.** `head` = the first `--head 8` tokens of a span that HAS a preceding slot;
`tail` = the rest of such a span; `before_slot0` = the tokens of span 0; `dump` = tokens
past the last slot. The four are a partition of the token positions (checked in the test).
Each bucket's CE is computed by masking the other positions to −100 and calling the SAME
weighted-CE code.

## Command

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python lab/divergence/slot_z_optimize.py \
  --ckpt <label>=<config>=/home/wolfe/morph-to/checkpoints/morph/<label>/step_5000.pt \
  --rows 12 --batch 3 --depth 6 --steps 200 --lrs 1e-2,3e-3 \
  --out ignored/experiment-artifacts/2026-09-10-slot-z-optimize/<label>.json
```

Run one arm at a time beside a live trainer. 12 packed validation rows, batch 3
(4 batches), every slot at depth 6.

The three runs were made on a working tree that also carried another agent's UNCOMMITTED
per-pass-LoRA work in `morph/model/{transformer,mhc}.py` and `morph/model/tul.py`. It is inert
on these arms — `tul.pass_lora_rank` is 0, no `TUL PASS-LORA ON` line appears in any of the
three logs, and the only forward change is a `pass_idx` argument a block with no LoRA
attached ignores — but it is named here rather than hidden.

## The reproduction gate

Before any number is read, the recorded z is substituted back UNCHANGED and must reproduce
the trained forward's token CE bit for bit. It does, on all three arms and on the CPU test
model: `|Δ| = 0.000e+00`. The deliberate wrong split — substituting `h_0`, the loop's ENTRY
state, where its OUTPUT belongs — does NOT reproduce it (tiny CPU model: 5.271486 →
5.275347, `+0.003861`), so the gate has teeth.

`pytest tests/test_slot_z_optimize.py -q` → `6 passed in 1.69s`.

## Results

### slot-mux-norm-match (`tul_slot_mux_norm_match`, step 5000)

12 rows, batch 3, depth 6, 200 Adam steps, mux_local 6.7959. Reproduction |Δ| 0.000e+00 (BIT-EXACT).

| arm | token CE | Δ vs loop | before_slot0 | head | tail | dump | \|z\| | eff-rank | cos(z, z_loop) |
|---|---|---|---|---|---|---|---|---|---|
| loop z (the trained forward) | 4.1173 | +0.0000 | 4.6431 | 4.1461 | 4.0858 | 3.9836 | 143.1 | 12.0 | — |
| z = 0 | 4.1299 | +0.0125 | — | — | — | — | — | — | — |
| z shuffled within a row | 4.1336 | +0.0162 | — | — | — | — | — | — | — |
| z = h_0 (loop entry, 0 passes) | 4.1188 | +0.0015 | 4.6431 | 4.1480 | 4.0875 | 3.9798 | 72.1 | 12.4 | — |
| z optimised, lr0.01 | 3.1732 | -0.9442 | 4.6431 | 2.6082 | 3.4919 | 2.7796 | 148.4 | 21.1 | +0.866 |
| z optimised, lr0.003 | 3.8741 | -0.2432 | 4.6431 | 3.7257 | 3.9477 | 3.7037 | 143.2 | 14.4 | +0.971 |
| z optimised, rand_lr0.01 | 3.2576 | -0.8597 | 4.6431 | 2.6910 | 3.5791 | 2.9081 | 145.1 | 130.0 | -0.002 |

- curve `lr0.01` (mean over 4 batches): step 0 4.1180, step 20 4.0757, step 50 3.9889, step 100 3.7284, step 150 3.4317, step 190 3.2206, final 3.1732
- curve `lr0.003` (mean over 4 batches): step 0 4.1180, step 20 4.1053, step 50 4.0856, step 100 4.0418, step 150 3.9736, step 190 3.8968, final 3.8741
- curve `rand_lr0.01` (mean over 4 batches): step 0 4.1382, step 20 4.0306, step 50 3.8872, step 100 3.6557, step 150 3.4476, step 190 3.2949, final 3.2576
- causality, |Δ| on `before_slot0`: {"lr0.01": 0.0, "lr0.003": 0.0, "rand_lr0.01": 0.0}
- z_loop: |z| 143.1, eff-rank 12.0, 159 valid slots per batch of 3
- h_0 (entry): |z| 72.1, eff-rank 12.4

### slot-unpack-norm-match (`tul_slot_unpack_norm_match`, step 5000)

12 rows, batch 3, depth 6, 200 Adam steps, mux_local 6.8119. Reproduction |Δ| 0.000e+00 (BIT-EXACT).

| arm | token CE | Δ vs loop | before_slot0 | head | tail | dump | \|z\| | eff-rank | cos(z, z_loop) |
|---|---|---|---|---|---|---|---|---|---|
| loop z (the trained forward) | 4.2931 | +0.0000 | 4.7140 | 4.3214 | 4.2656 | 4.1417 | 84.0 | 12.2 | — |
| z = 0 | 4.4967 | +0.2037 | — | — | — | — | — | — | — |
| z shuffled within a row | 4.4531 | +0.1601 | — | — | — | — | — | — | — |
| z = h_0 (loop entry, 0 passes) | 4.2964 | +0.0033 | 4.7140 | 4.3277 | 4.2671 | 4.1476 | 78.3 | 12.8 | — |
| z optimised, lr0.01 | 1.6673 | -2.6258 | 4.7140 | 1.0712 | 1.9579 | 1.2391 | 92.2 | 82.2 | +0.873 |
| z optimised, lr0.003 | 2.8425 | -1.4506 | 4.7140 | 2.2522 | 3.1625 | 2.5486 | 85.9 | 66.8 | +0.959 |
| z optimised, rand_lr0.01 | 2.1984 | -2.0947 | 4.7140 | 1.5587 | 2.5318 | 1.8215 | 90.3 | 125.2 | -0.001 |

- curve `lr0.01` (mean over 4 batches): step 0 4.2938, step 20 3.9046, step 50 3.4218, step 100 2.6230, step 150 2.0136, step 190 1.7239, final 1.6673
- curve `lr0.003` (mean over 4 batches): step 0 4.2938, step 20 4.1565, step 50 3.9676, step 100 3.6069, step 150 3.2286, step 190 2.9188, final 2.8425
- curve `rand_lr0.01` (mean over 4 batches): step 0 4.5703, step 20 4.0792, step 50 3.7178, step 100 3.1678, step 150 2.6412, step 190 2.2780, final 2.1984
- causality, |Δ| on `before_slot0`: {"lr0.01": 0.0, "lr0.003": 0.0, "rand_lr0.01": 0.0}
- z_loop: |z| 84.0, eff-rank 12.2, 159 valid slots per batch of 3
- h_0 (entry): |z| 78.3, eff-rank 12.8

### slot-loop-norm-match (`tul_slot_loop_norm_match`, step 5000)

12 rows, batch 3, depth 6, 200 Adam steps, no MUX. Reproduction |Δ| 0.000e+00 (BIT-EXACT).

| arm | token CE | Δ vs loop | before_slot0 | head | tail | dump | \|z\| | eff-rank | cos(z, z_loop) |
|---|---|---|---|---|---|---|---|---|---|
| loop z (the trained forward) | 4.0222 | +0.0000 | 4.6154 | 4.0459 | 3.9921 | 3.8884 | 68.8 | 12.5 | — |
| z = 0 | 4.0308 | +0.0086 | — | — | — | — | — | — | — |
| z shuffled within a row | 4.0414 | +0.0192 | — | — | — | — | — | — | — |
| z = h_0 (loop entry, 0 passes) | 4.0223 | +0.0002 | 4.6154 | 4.0465 | 3.9919 | 3.8900 | 67.1 | 11.3 | — |
| z optimised, lr0.01 | 2.8372 | -1.1849 | 4.6154 | 2.4032 | 3.0667 | 2.3483 | 78.0 | 44.6 | +0.695 |
| z optimised, lr0.003 | 3.5618 | -0.4603 | 4.6154 | 3.3226 | 3.6874 | 3.2205 | 70.3 | 21.9 | +0.901 |
| z optimised, rand_lr0.01 | 2.7424 | -1.2798 | 4.6154 | 2.3024 | 2.9745 | 2.1720 | 73.1 | 108.9 | -0.002 |

- curve `lr0.01` (mean over 4 batches): step 0 4.0227, step 20 3.9777, step 50 3.8303, step 100 3.4306, step 150 3.0942, step 190 2.8843, final 2.8372
- curve `lr0.003` (mean over 4 batches): step 0 4.0227, step 20 4.0101, step 50 3.9877, step 100 3.9031, step 150 3.7477, step 190 3.5994, final 3.5618
- curve `rand_lr0.01` (mean over 4 batches): step 0 4.0361, step 20 3.8598, step 50 3.6120, step 100 3.2608, step 150 2.9673, step 190 2.7822, final 2.7424
- causality, |Δ| on `before_slot0`: {"lr0.01": 0.0, "lr0.003": 0.0, "rand_lr0.01": 0.0}
- z_loop: |z| 68.8, eff-rank 12.5, 159 valid slots per batch of 3
- h_0 (entry): |z| 67.1, eff-rank 11.3

## Reading

**1. The split point holds.** The recorded z substituted back unchanged reproduces the
trained forward's token CE with `|Δ| = 0.000e+00` on all three arms. Every number below
is read on the shipped downstream code.

**2. The loop's z is worth almost nothing to the coda as it stands.** Zeroing it costs
0.0086 (loop), 0.0125 (mux) and 0.2037 (unpack) nats. Shuffling whole slots within a row
costs 0.0192 / 0.0162 / 0.1601. Replacing z with `h_0`, the loop's ENTRY state — what the
coda would read after ZERO passes — costs 0.0002 / 0.0015 / 0.0033 nats. Six passes of a
268 M-parameter core buy the coda between two ten-thousandths and three thousandths of a
nat.

**3. A gradient-optimised z is worth 100x to 300x that.** At lr 1e-2 for 200 Adam steps:

| arm | loop's z is worth (zeroing cost) | an optimised z is worth | the loop delivers |
|---|---|---|---|
| slot-mux-norm-match | 0.0125 | ≥ 0.944 | 1.3 % |
| slot-unpack-norm-match | 0.2037 | ≥ 2.626 | 7.8 % |
| slot-loop-norm-match | 0.0086 | ≥ 1.185 | 0.7 % |

The `≥` is not decoration. Every curve is still falling at step 200 (mux 3.2206 → 3.1732
over the last 10 steps; unpack 1.7239 → 1.6673), and lr 3e-3 lands far above lr 1e-2 at
the same step count on all three arms. The optimised numbers are a LOWER bound on the
oracle ceiling, set by the step budget, not a converged value.

**So the answer to the question this probe was built for is the first branch: the coda
CAN use z, by a wide margin, on all three arms — including the two where it also has a
token path.** The reader is not the bottleneck.

**4. The loop's z is not even a useful STARTING point.** Optimising from a random z with
the same per-slot norm reaches 3.2576 / 2.1984 / 2.7424 — within 0.09 nats of the
loop-start run on the mux arm, and BETTER than it on the no-MUX arm (2.7424 vs 2.8372).
A random z's own CE before any optimisation is 4.1382 on the mux arm against the loop's
4.1173: 0.02 nats. The loop's six passes put the state somewhere the coda treats as
almost interchangeable with noise of the same length.

**5. The usable z-space is far bigger than the one the loop occupies.** The loop's z sits
at participation rank 12.0-12.5 across 159 valid slots (of a 4096-dimensional carrier,
`n·C` = 4 × 1024). Optimisation raises the rank to 21 (mux), 45 (loop) and 82 (unpack);
starting from random it ends at 109-130 with cosine −0.002 to the loop's z. Two different
z's, orthogonal to each other, both drive the coda well. Norms barely move (143 → 148,
84 → 92, 69 → 78), so this is a DIRECTION problem, not a scale one.

**6. Where z helps: right after the write, and then less.** Head is the first 8 tokens of
a span that has a preceding slot; tail is the rest of that span.

| arm | head, loop → zopt | tail, loop → zopt |
|---|---|---|
| slot-mux-norm-match | 4.1461 → 2.6082 (−1.538) | 4.0858 → 3.4919 (−0.594) |
| slot-unpack-norm-match | 4.3214 → 1.0712 (−3.250) | 4.2656 → 1.9579 (−2.308) |
| slot-loop-norm-match | 4.0459 → 2.4032 (−1.643) | 3.9921 → 3.0667 (−0.925) |

The channel is strongest at the head and decays across the span without vanishing. It is
strongest of all on the unpack arm, where the coda's only route to earlier spans is z.

**7. The causality gate is exactly satisfied.** `before_slot0` — the tokens of span 0,
which causal attention puts before every slot cell — moves by 0.0 on every arm and every
optimisation run. Nothing about the optimisation leaks backwards.

## What this does NOT say

* **`ce_zopt` is an ORACLE, not a target.** z_i is fitted on the tokens it is asked to
  predict. A causal loop sees only the span it terminates. The ceiling bounds the READER
  (the coda plus `W_prefix` plus, on the unpack arm, `W_bcast`); it says nothing about
  what any WRITER could reach. The obvious follow-up — refit z per span using ONLY that
  span's own tokens, so the ceiling becomes a causal one — is not run here.
* **The three arms are not comparable to each other.** They differ in the training
  objective (MUX or not) and in the coda's wiring (`coda_token_input`, `tg_restrict`), so
  `ce_loop` 4.0222 vs 4.1173 vs 4.2931 ranks nothing.
* **One seed, 12 rows, 4 batches, depth 6, step 5000.** A reading, not a ranking.
* **`ce_zero` on the unpack arm reads 0.2037 here against the 0.811 in the earlier worth
  profile.** Different instruments — this zeroes `h_slots` before BOTH the prefix write
  and the bcast unpack, through `_tul_plan_ablate`; `slot_path_worth.py` zeroes
  `prefix_project`'s VALUES — and different row draws. The larger ablation reading the
  SMALLER cost is not explained, and one paired run on the same rows would settle it. It
  does not touch conclusions 3-6, which are all within-run comparisons.
* **Not measured:** whether the optimised z's directions are learnable, whether the
  ceiling moves with more optimisation steps or a schedule, whether it holds at other
  checkpoint steps, and whether an optimised z survives being decoded (generation quality
  was not scored).
