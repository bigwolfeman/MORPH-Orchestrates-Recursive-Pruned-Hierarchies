# Is the core muted on the slot loop's compact sequence? (2026-09-10)

Instrument: [`lab/divergence/slot_geometry_audit.py`](../../../lab/divergence/slot_geometry_audit.py).
Test: `tests/test_slot_geometry_audit.py`.
JSON artifacts: `ignored/experiment-artifacts/2026-09-10-slot-geometry-audit/<label>.json`
(one per arm, with the console log beside it).

## The question

Not "why does the loop not earn depth". The narrower one: **why does even ONE pass of a
six-block core change nothing the coda reads.** Two earlier probes set it up.
[`2026-09-10-slot-z-optimize`](../2026-09-10-slot-z-optimize/README.md) says the coda's
CE with the loop's exit state equals its CE with the loop's ENTRY state to 0.003 nats,
while a gradient-fitted z is worth 0.9-2.6 nats to the same frozen coda — so the READER
is not the bottleneck. The forward instruments (`slot_anatomy.py`, `slot_state_probe.py`)
say the core blocks' attention and MLP branches output 2-20 % of the state norm on slot
states against 40-110 % on tokens. Wolfe: "the standard TUL is principled. there has to
be a bug or something."

So: is the shared core structurally MUTED at S = 64, and by what?

**The answer is no.** The core does as much work per pass on a slot state as on a token
state, the readout hands almost all of that work to the coda, and the coda's CE does not
respond. One real geometry defect is confirmed live and quantified (three of six core
blocks lose about 44 % of their attention mixture to an empty tensor), and it is NOT the
cause: the one arm that avoids it by construction is just as flat.

## Method

Three slot-loop checkpoints at step 5000, seq 1024, `norm_match` ternary, 12 packed
validation rows in 4 batches of 3, every slot forced to depth 6, `model.eval()`, bf16
autocast, `model.use_kernels=false`, `model.slot_gain_lambda=0`.

| label | config | what is different |
|---|---|---|
| `slot-mux-norm-match` | `tul_slot_mux_norm_match` | the M-next ruler: prelude entry, MUX loss, hinge + fixed-point term on, pooled compressor in the core |
| `slot-unpack-free` | `tul_slot_unpack_free` | unpack coda (`bcast`), `tg_restrict` on (so the core's compressed branch is dense causal slot attention, NOT the pooled compressor), both stability terms off |
| `slot-mnext-noise-entry` | `tul_slot_mnext_noise_entry` | Parcae noise entry (`core_state_init: noise`, std 0.02), all-dim injection with a learned B, both stability terms off |

**Where the numbers come from.** The audit hooks every core block's attention branch, MLP
branch and both HC residual writes, and patches three methods whose quantities are frame
locals and otherwise unreadable: `_gate_combine_up` (the compressed / window branch
outputs and the gate that blends them), `_window_attn` (the attention weights, which SDPA
never materialises), and `model.injection` / `model._apply_injection` (the SSM ctx rewrite
and the per-layer x0 term, plus the pass counter the injection counterfactual needs).

**The token twin.** Every "same weights, S = 1152" column is ONE application of the SAME
checkpoint's core to `input_norm(prelude)` over ALL positions of the SAME rows,
constructed exactly as `_tul_core` constructs its own call (`core_init`, the per-core-layer
injection stack, `_apply_core_step` at `iter_idx=0`). Tokens never loop in training on
these arms; the twin says what the weights DO at the long shape, and is not a trained
control.

**Norms are PER POSITION.** `|h|` is the mean over the selected positions of that
position's L2 norm over the whole 4-stream carrier. S differs by 18x between the two
shapes and a whole-tensor norm would carry that difference into every comparison. Slot
rows are restricted to `layout.slot_valid` (158 valid slots in the first batch of 3,
159 averaged over the four); token rows are
all 1152 positions.

**Two gates, both exact on all three arms.**

1. The state captured after the last core block IS the tensor `prefix_project` receives:
   `max |Δ|` over valid slots = **0.000e+00**.
2. Substituting that captured state reproduces the trained forward's token CE bit for
   bit: `|Δ|` = **0.000e+00**.

The deliberate wrong split (the ENTRY state where the exit belongs) does NOT reproduce it
— that is `K0` in section 5, and it differs from `ce_loop` on every arm — so gate 2 has
teeth.

`pytest tests/test_slot_geometry_audit.py -q` → **5 passed in 2.39s**.

## Command

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python lab/divergence/slot_geometry_audit.py \
  --ckpt <label>=<config>=/home/wolfe/morph-to/checkpoints/morph/<label>/step_5000.pt \
  --rows 12 --batch 3 --depth 6 \
  --out ignored/experiment-artifacts/2026-09-10-slot-geometry-audit/<label>.json
```

One arm at a time beside the live `horizon-ternary-25k` trainer (12.2 GB); the audit peaks
around 3 GB.

---

## Lead 1 — attention geometry at S = 64

Config numbers as the code reads them at build (not the yaml): `d_model` 1024, 8 heads,
`compression` 2 → `d_head` 64, `window_size` 256, `csa_compress_ratio` 8,
`hca_compress_ratio` 256, `top_k` 256, `core_hca_compress_ratio` unset. Core blocks 0/2/4
are CSA (global layer index 4/6/8, even), 1/3/5 are HCA.

`|comp|` and `|win|` are the raw branch outputs before the gate; `g_comp` is the mean gate
weight the compressed branch gets; `H` is the window branch's attention entropy in nats;
`prev` is the share of window mass on the immediately preceding cell; `keys` is the mean
number of keys a query actually has.

### `slot-mux-norm-match` (pooled compressor)

| blk | kind | m | n_blocks | tk | \|comp\| | \|win\| | g_comp | H | keys | part. | top-1 | prev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | CSA | 8 | 8 | 8 of 8 | 59.7 | 300.3 | 0.037 | 2.14 | 26 | 8.0 | 33.4 % | 9.9 % |
| 1 | HCA | 256 | **0** | — | **0.000** | 223.5 | **0.479** | 2.23 | 26 | 9.8 | 33.2 % | 24.9 % |
| 2 | CSA | 8 | 8 | 8 of 8 | 98.9 | 227.8 | 0.110 | 2.37 | 26 | 8.9 | 27.3 % | 24.9 % |
| 3 | HCA | 256 | **0** | — | **0.000** | 217.8 | **0.438** | 2.31 | 26 | 10.1 | 28.0 % | 19.4 % |
| 4 | CSA | 8 | 8 | 8 of 8 | 49.2 | 205.5 | 0.022 | 2.75 | 26 | 15.1 | 17.2 % | 16.3 % |
| 5 | HCA | 256 | **0** | — | **0.000** | 236.9 | **0.420** | 2.01 | 26 | 9.9 | 40.7 % | 38.2 % |

Same weights, token sequence S = 1152:

| blk | kind | n_blocks | tk | \|comp\| | \|win\| | g_comp | H | keys | part. | top-1 | prev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | CSA | 144 | 144 of 144 | 1137.2 | 867.4 | 0.082 | 4.32 | 227 | 47.5 | 10.8 % | 4.5 % |
| 1 | HCA | 4 | — | 830.2 | 807.1 | 0.484 | 4.53 | 227 | 62.5 | 10.6 % | 8.7 % |
| 2 | CSA | 144 | 144 of 144 | 1113.0 | 975.4 | 0.068 | 4.31 | 227 | 41.9 | 13.5 % | 12.3 % |
| 3 | HCA | 4 | — | 828.4 | 959.5 | 0.426 | 4.67 | 227 | 72.4 | 7.2 % | 6.0 % |
| 4 | CSA | 144 | 144 of 144 | 1122.8 | 819.3 | 0.015 | 4.59 | 227 | 59.0 | 7.7 % | 7.2 % |
| 5 | HCA | 4 | — | 679.7 | 990.9 | 0.424 | 4.20 | 227 | 56.3 | 15.0 % | 13.8 % |

Readings:

* **The HCA compressed branch is exactly dead at S = 64, and the gate does not know.**
  `|out_comp|` is `0.000` — not small, zero — on blocks 1, 3 and 5, while `g_comp` puts
  0.42-0.48 of the head mixture on it. On the same weights at S = 1152 the same branch
  outputs 680-830 with the same gate value. This is the defect `morph/model/CLAUDE.md`
  predicted from an untrained probe in August, confirmed here on a trained checkpoint at
  step 5000. Full write-up as finding **F1** below.
* **CSA's sparse selection never fires.** `tk = min(top_k, n_blocks) = min(256, 8) = 8` of
  8 blocks on slots and 144 of 144 on tokens. CSA is dense pooled attention on both
  shapes; nothing is selected away. The gate has learned to spend 1.5-11 % on it either
  way, so on the CSA layers the compressed branch is mostly ignored by the model itself.
* **Query 0 has no key at all, on BOTH shapes.** `keys` for row 0 is 0 everywhere: the
  window is causal and XSA excludes the self token, so cell 0's softmax row is all `-inf`
  and SDPA returns 0. This is not a slot-path property — the token twin does the same
  thing — so it is not a candidate cause.
* **The window branch is diffuse on both shapes.** Participation ratio 8-15 of 26 visible
  keys on slots (31-58 %) and 42-72 of 227 on tokens (19-32 %); top-1 mass 17-41 % vs
  7-15 %. A slot puts more of its mass on its immediate predecessor (9.9-38.2 % vs
  4.5-13.8 %), which is what a 64-cell causal sequence with a 256-wide window looks like.
  Neither shape is a sink.

### `slot-unpack-free` — the arm with no compressor in the core

`tg_restrict` routes the core's compressed branch to `_tg_slot_attention` with
`slot_mask=None`, i.e. dense causal attention over all 64 cells with a per-head sink. The
pooled compressor is never called, so `n_blocks` does not apply.

| blk | kind | branch | \|comp\| | \|win\| | g_comp | H | part. | top-1 |
|---|---|---|---|---|---|---|---|---|
| 0 | CSA | tg dense causal | 153.3 | 152.6 | 0.066 | 2.50 | 12.5 | 27.8 % |
| 1 | HCA | tg dense causal | 167.1 | 165.7 | 0.021 | 2.37 | 9.6 | 29.0 % |
| 2 | CSA | tg dense causal | 179.7 | 178.4 | 0.034 | 2.55 | 11.9 | 20.3 % |
| 3 | HCA | tg dense causal | 200.9 | 199.5 | 0.038 | 2.09 | 6.9 | 32.2 % |
| 4 | CSA | tg dense causal | 132.2 | 131.9 | 0.051 | 2.56 | 13.4 | 27.3 % |
| 5 | HCA | tg dense causal | 225.7 | 224.2 | 0.020 | 2.53 | 12.6 | 22.6 % |

No branch is empty here. **This arm is the control that decides F1's role in the
flatness**, and section 5 shows it is just as flat past pass 1.

### `slot-mnext-noise-entry`

Same shape as the mux arm: HCA `|out_comp|` = `0.000` on blocks 1, 3, 5 with `g_comp`
0.44-0.52; CSA `tk` 8 of 8. Its window branch is the most concentrated of the three
(top-1 40-49 % on blocks 2/3/4, participation 4.0-8.8 of 26).

---

## Lead 2 — branch outputs, absolute, same weights, both shapes

`|h|` is the 4-stream carrier entering the block; `attn` / `mlp` are the single-stream
sublayer outputs — the tensors the HC POST write scatters into the carrier. Pass 1.

### `slot-mux-norm-match`

| blk | kind | \|h\|_slot | attn_slot | mlp_slot | \|h\|_tok | attn_tok | mlp_tok | attn slot/tok | mlp slot/tok |
|---|---|---|---|---|---|---|---|---|---|
| 0 | CSA | 99.00 | 3.80 | 1.84 | 83.45 | 4.04 | 3.23 | **0.94** | 0.57 |
| 1 | HCA | 101.97 | 2.55 | 1.56 | 85.90 | 6.95 | 3.11 | **0.37** | 0.50 |
| 2 | CSA | 104.29 | 2.48 | 1.54 | 90.29 | 4.44 | 2.79 | **0.56** | 0.55 |
| 3 | HCA | 107.08 | 2.40 | 1.58 | 94.12 | 5.54 | 3.10 | **0.43** | 0.51 |
| 4 | CSA | 108.18 | 0.50 | 1.42 | 95.76 | 0.60 | 2.71 | **0.84** | 0.53 |
| 5 | HCA | 108.32 | 1.49 | 7.00 | 96.03 | 4.65 | 7.30 | **0.32** | 0.96 |

### `slot-unpack-free` (no dead branch)

| blk | \|h\|_slot | attn_slot | mlp_slot | \|h\|_tok | attn_tok | mlp_tok | attn slot/tok | mlp slot/tok |
|---|---|---|---|---|---|---|---|---|
| 0 | 68.32 | 4.79 | 5.19 | 68.22 | 4.61 | 4.18 | 1.04 | 1.24 |
| 1 | 80.13 | 4.63 | 4.90 | 78.47 | 4.15 | 4.44 | 1.12 | 1.10 |
| 2 | 95.08 | 13.97 | 2.46 | 91.67 | 13.17 | 2.39 | 1.06 | 1.03 |
| 3 | 145.61 | 8.16 | 12.74 | 139.00 | 8.08 | 10.04 | 1.01 | 1.27 |
| 4 | 180.82 | 2.94 | 3.71 | 170.44 | 1.97 | 3.31 | 1.49 | 1.12 |
| 5 | 186.71 | 0.76 | 0.81 | 175.31 | 1.06 | 0.75 | 0.72 | 1.08 |

### `slot-mnext-noise-entry`

attn slot/token 1.53, **0.25**, 1.24, **0.44**, 1.31, 1.33; mlp slot/token 1.31, 1.50,
1.35, 1.15, 1.17, 1.68.

**Readings.**

1. **The blocks are not silent on slot states.** On the unpack arm every branch output is
   within 4-49 % of the same weights' output on tokens, and the MLP is uniformly LARGER on
   slots. On the noise arm the MLP is 1.15-1.68x the token figure.
2. **The one systematic deficit tracks the dead branch exactly.** On the mux arm the
   attention branch on the three HCA blocks is 0.32-0.43x the token figure, against
   0.56-0.94x on the three CSA blocks; on the noise arm 0.25 / 0.44 on blocks 1 and 3
   against 1.24-1.53 on the CSA blocks. The arm with no pooled compressor (unpack-free)
   shows no such split — 1.01-1.12 on its HCA blocks. That is a clean attribution of the
   deficit to F1 and nothing else.
3. **The "2-20 % vs 40-110 %" reading is partly a norm artefact.** The slot carrier here
   is 1.00-1.07x the token carrier on the unpack arm, 1.13-1.19x on the mux arm and
   1.31-1.45x on the noise arm, so an `out/in` ratio is 0-31 % smaller on slots for that
   reason alone. The rest of the gap is F1, on three blocks of six, on two arms of three —
   and on the third arm there is no gap to explain.

---

## Lead 3 — the injection

`ssm_delta` is `|injection(h, e) − h|`, the `DiagonalInjection` ctx rewrite. `layer_term`
is the per-core-layer x0/bigram additive term. `|dh|` is the whole pass's movement,
`|h_after_pass − h_before_pass|`. All per position.

| arm | pass | \|h_in\| | ssm_delta | layer_term | \|dh\| | \|dh\|/\|h_in\| |
|---|---|---|---|---|---|---|
| mux | 1 | 72.09 | 31.25 | 0.145 | 46.87 | 0.650 |
| mux | 2 | 112.60 | 14.45 | 0.145 | 16.54 | 0.147 |
| mux | 3 | 127.67 | 10.40 | 0.145 | 10.97 | 0.090 |
| mux | 4 | 135.12 | 9.57 | 0.145 | 8.31 | 0.066 |
| mux | 5 | 139.03 | 9.66 | 0.145 | 7.00 | 0.054 |
| mux | 6 | 141.45 | 9.93 | 0.145 | 6.24 | 0.048 |
| mux | token, 1 pass | 68.34 | 21.31 | 0.062 | 70.82 | 1.036 |
| unpack | 1 | 67.00 | 5.47 | 0.059 | 196.96 | 2.940 |
| unpack | 6 | 434.23 | 15.28 | 0.059 | 53.10 | 0.120 |
| unpack | token, 1 pass | 67.03 | 5.26 | 0.191 | 186.42 | 2.781 |
| noise | 1 | 1.28 | 122.33 | 0.139 | 315.75 | 246.7 |
| noise | 6 | 405.20 | 191.21 | 0.139 | 222.80 | 0.391 |
| noise | token, 1 pass | 1.28 | 93.36 | 0.030 | 206.36 | 161.2 |

**Readings.**

1. **One pass moves a slot state as far as it moves a token state.** 0.650 against 1.036
   (mux), 2.940 against 2.781 (unpack), 246.7 against 161.2 (noise). The first pass's
   ratio is inflated on all three because its input is `input_norm(prelude)` or noise
   rather than a previous core output — `_tul_core`'s own probe comment names that — but
   the slot and token columns share the inflation, so the comparison stands.
2. **The map contracts, fast.** Relative movement per pass on the mux arm: 0.650, 0.147,
   0.090, 0.066, 0.054, 0.048. On the unpack arm 2.94, 0.46, 0.26, 0.18, 0.14, 0.12. The
   loop reaches a neighbourhood in one pass and creeps after that.
3. **The injection does not swamp the blocks, and cutting it changes nothing that matters.**
   On the mux arm the SSM rewrite (31.2) is two thirds of the whole first pass's movement
   (46.9) and from pass 3 it EXCEEDS the net movement (10.4 against 11.0, then 9.6 against
   8.3) — so from pass 3 on, most of what the core does to a slot state is re-imposing `e`.
   But cutting the injection after pass 1 moves the exit state by 26.5 % (mux) / 13.7 %
   (unpack) / 106 % (noise) of its norm and moves the token CE by **+0.0000 / +0.0043 /
   +0.0019** nats. The per-layer x0/bigram term is negligible everywhere (0.06-0.15
   against carriers of 67-434).

---

## Lead 4 — the readout

`scatter_positions` REPLACES the coda cell, so the cell IS `W_prefix[k] · z` and
`|W_k·(exit − entry)| / |W_k·entry|` is exactly the loop's share of what the coda gets.
`stream_mean_ratio` is `|mean_n u| / mean_n |u_n|` — the MUX head and `tul.unpack` read
`h.mean(dim=2)`, and this says how much of a quantity survives that mean.

| quantity | mux | unpack-free | noise-entry |
|---|---|---|---|
| \|entry\| per slot | 72.09 | 67.00 | 1.28 |
| \|exit\| per slot | 143.08 | 483.49 | 407.21 |
| \|exit − entry\| / \|entry\| | 1.136 | 7.292 | 318.3 |
| `W_0·entry` | 153.62 | 101.48 | 1.42 |
| `W_0·(exit − entry)` | 164.22 | 690.31 | 1136.04 |
| **loop's share of coda cell 0** | **1.076** | **6.803** | **802.3** |
| loop's share of coda cell 1 | 1.067 | 7.065 | 765.3 |
| `W_0` σ_max / σ_min | 4.64 / 0.054 (86.6x) | 2.84 / 0.336 (8.5x) | 5.00 / 0.124 (40.4x) |
| \|cos\| of the update with `W_0`'s top-8 left singular directions | 0.06-0.39 | 0.01-0.06 | 0.09-0.41 |
| stream-mean survival, ENTRY | 0.581 | 0.972 | 0.500 |
| stream-mean survival, the loop's UPDATE | 0.577 | **0.139** | 0.818 |

**Readings.**

1. **The prefix write does not discard the loop's work — it is dominated by it.** The
   loop's contribution to the coda cell is 1.07x the entry's on the mux arm, 6.8x on the
   unpack arm and 802x on the noise arm. On the noise arm the coda cell is, to three
   figures, ONLY the loop's output: the entry is 0.02-std noise.
2. **`W_prefix` is anisotropic but is not filtering the loop out.** σ_max/σ_min is 8.5-87
   and the update's alignment with the top singular directions is modest (max |cos| 0.45),
   yet the share above is ≥ 1 on every arm — the update survives the projection with gain
   ≥ 1.
3. **The stream mean is a real loss on the unpack arm, and only there.** 86 % of the
   loop's update cancels in `h.mean(dim=2)` (survival 0.139) while only 3 % of the entry's
   does (0.972). The MUX head and `tul.unpack` read that mean; the prefix write does not.
   Finding **F2** below.

---

## Lead 5 — K0 / K1 / K6 at the coda

The trainer's ONE weighted token CE with `z` replaced by the state after exactly `t`
passes, through the SHIPPED downstream code (prefix write, unpack, TG masks, coda,
weighted CE). `K0` is the entry state — what the coda would read after ZERO passes.

| arm | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K0−K6 | K1−K6 |
|---|---|---|---|---|---|---|---|---|---|
| slot-mux-norm-match | 4.1195 | 4.1184 | 4.1183 | 4.1181 | 4.1181 | 4.1182 | 4.1180 | **0.0015** | 0.0004 |
| slot-unpack-free | 4.4327 | 4.3272 | 4.3252 | 4.3251 | 4.3252 | 4.3252 | 4.3254 | **0.1074** | 0.0018 |
| slot-mnext-noise-entry | 4.1062 | 4.0876 | 4.0874 | 4.0876 | 4.0876 | 4.0878 | 4.0877 | **0.0185** | −0.0001 |

Injection cut after pass 1, at depth 6: 4.1180 (`+0.0000`), 4.3297 (`+0.0043`), 4.0895
(`+0.0019`).

**Readings.**

1. **Pass 1 is the whole loop.** K1−K6 is 0.0004 / 0.0018 / −0.0001 nats. Passes 2 through
   6 of a 268 M-parameter core are worth less than two thousandths of a nat on every arm,
   while the state keeps moving 5-46 % of its norm per pass.
2. **Even pass 1 buys almost nothing, and the amount is not set by how far the state
   moves.** The noise arm moves z by 318x its entry norm and delivers a coda cell that is
   802x the entry's contribution — the coda reads essentially nothing BUT the loop's work
   — and it is worth 0.0185 nats. The mux arm moves z by 1.14x and gets 0.0015. The unpack
   arm's 0.1074 is the largest and is still 4 % of what a gradient-fitted z buys the SAME
   frozen coda on the sibling `slot-unpack-norm-match` arm (≥ 2.626 nats,
   [z-optimize](../2026-09-10-slot-z-optimize/README.md)).
3. **The counterfactual confirms the direction of the problem.** Cutting the injection
   after pass 1 changes z by 13-106 % of its norm and costs 0.000-0.004 nats. Large,
   structured changes to z are free. That is the same fact as (2), by a different lever.

---

## Lead 6 — norm growth

| arm | entry | p2 | p3 | p4 | p5 | p6 | exit | token entry → after 1 pass |
|---|---|---|---|---|---|---|---|---|
| mux | 72.09 | 112.60 | 127.67 | 135.12 | 139.03 | 141.45 | 143.08 | 68.33 → 100.45 |
| unpack-free | 67.00 | 186.95 | 263.61 | 326.18 | 382.11 | 434.23 | 483.49 | 67.03 → 175.69 |
| noise-entry | 1.28 | 315.74 | 351.23 | 388.24 | 399.01 | 405.20 | 407.21 | 1.28 → 206.36 |

The slot carrier ends at 2.0x (mux), 7.2x (unpack) and 318x (noise) its entry norm. The
mux arm's growth slows to about 2 % per pass by pass 4; the unpack arm's does not (no stability terms, an
expansive map). A pre-norm reader inside the block sees `x_bar` after an RMSNorm, so the
growth does not itself shrink what the branches see — and it does not: the branch outputs
in lead 2 are flat or rising across blocks, and `slot/token` is near 1 on the two arms
without F1.

---

## Verdict

**The core is NOT muted on the compact sequence.** Measured on three arms, with both gates
exact:

* one pass moves a slot state 0.65 / 2.94 / 247x its own norm, against 1.04 / 2.78 / 161x
  for the same weights on a token sequence — mux / unpack / noise, paired;
* on `slot-unpack-free`, the one arm with no dead branch, every branch output on slots is
  0.72-1.49x (attention) and 1.03-1.27x (MLP) of the same weights' output on tokens. The
  two arms with the dead HCA branch are lower on the blocks that carry it (0.25-0.43) and
  ordinary elsewhere (0.53-1.53);
* the prefix write hands the coda a cell that is 1.07x, 6.8x and 802x the loop's own work.

**The flatness is neither the writer's magnitude nor the reader's plumbing. It is
direction.** The loop moves z a very long way, the coda receives that movement almost
undiminished, and the CE does not respond: K0−K6 is 0.0015-0.107 nats where a
gradient-fitted z on the same frozen coda is worth 0.9-2.6. Three independent levers make
the same point — six passes instead of one (0.0004-0.0018), the injection cut
(0.000-0.004), and the noise entry that hands the coda a state built ENTIRELY by the loop
(0.0185). The loop is a contraction to a neighbourhood the coda's loss is flat along, and
it reaches it in one pass.

That is a credit-assignment result, not a geometry one, and it matches the backward
reading: the per-pass gradient probe found the cotangent share flat across passes
(0.155-0.21) with per-pass weight updates that cancel (`|Σ dW_t| / Σ|dW_t|` 0.20-0.60), and
the MUX paying 7.3x what the token CE pays. Nothing in this audit points at a broken
forward that a shape fix would repair.

### Findings

**F1 — the HCA compressed branch is dead at the slot budget, and the gate keeps paying it.**

* Where: `morph/model/attention.py:310` — `GatedPoolCompressor.forward`'s `n_blocks == 0`
  early return, reached whenever `S < m`. `n_blocks = S // m = 64 // 256 = 0`, so
  `fused_hca_attention` attends an empty stream and `_CCABase._gate_combine_up`
  (`morph/model/attention.py:840`) blends `g_comp` into a zero tensor.
* Measured, live, at step 5000: `|out_comp|` exactly `0.000` on core blocks 1, 3 and 5 of
  `slot-mux-norm-match` and `slot-mnext-noise-entry`, with `g_comp` 0.42-0.52 there. The
  same weights on S = 1152 give 680-830.
* Cost: on `slot-mux-norm-match` the attention branch of blocks 1, 3 and 5 outputs
  0.37 / 0.43 / 0.32x what the same weights output on tokens, against 0.94 / 0.56 / 0.84
  on the CSA blocks. On `slot-mnext-noise-entry` blocks 1 and 3 read 0.25 / 0.44 against
  1.24-1.53 on its CSA blocks — but its block 5 reads 1.33, so the deficit is not uniform
  across the three dead branches and this is an association on 5 of 6 measured HCA blocks,
  not a controlled effect. The arm with no pooled compressor in the core
  (`slot-unpack-free`, `tg_restrict`) shows 1.12 / 1.01 / 0.72 on the same blocks.
* Smallest fix: set `model.core_hca_compress_ratio: 16` in the slot-loop config root
  (`tul_short.yaml`). The knob exists for exactly this; 16 gives the slot core 4 blocks,
  the same number the token path gets. **Not applied, not run.**
* This is NOT the cause of the flat K-curve. `slot-unpack-free` avoids it entirely and its
  K1−K6 is 0.0018.

**F2 — the MUX / unpack reader loses 86 % of the loop's update to the stream mean.**

* Where (line numbers at `bf87aeb`): `morph/model/tul.py:1367` — `TULSlots.unpack` does
  `z = h_slots.mean(dim=2)`. The MUX head reaches the same reduction by another route:
  `_tul_mux_loss` calls `self._readout(h_slots)` (`morph/model/transformer.py:3483`) and
  `_readout` collapses the streams with `x = x.mean(dim=2)`
  (`morph/model/transformer.py:1940`). `TULSlots.prefix_project` does NOT — it projects
  each stream separately.
* Measured on `slot-unpack-free`: the entry state survives the stream mean at 0.972 of its
  per-stream norm, the loop's UPDATE at 0.139. So the two readers on that arm see very
  different objects — the prefix cell carries 6.8x the entry's contribution, the bcast term
  and the MUX target carry a seventh of that.
* On the other two arms the mean treats entry and update alike (0.581 vs 0.577; 0.500 vs
  0.818), so this is specific to the arm whose loop grows 7.3x with no stability term.
* Smallest fix if it is ever load-bearing: read the flattened carrier through a learned
  `[n·C → C]` projection instead of the unweighted mean, the way `prefix_project` reads it
  per stream. **Not applied, not run, and not shown to matter** — on this arm the prefix
  cell already delivers the update and the CE is still flat.

**Not a defect, recorded so it is not rediscovered:** CSA selects every block on both
shapes (`tk` 8 of 8 and 144 of 144), and query 0's window row is empty under XSA on BOTH
shapes. Neither distinguishes the slot path.

## What this does NOT say

* **One checkpoint step (5000), one seed, 12 rows, 4 batches, depth 6.** A reading, not a
  ranking. The three arms differ in objective and coda wiring, so their absolute CEs do
  not compare with each other.
* **The token twin is not a trained control.** It is the same weights applied once to
  token states that never went through the core during training. It says what the core
  DOES at S = 1152; it does not say what a token loop would have learned.
* **CE means here are unweighted over the 4 batches**, where `slot_z_optimize` weights by
  `n_targets`. Absolute CEs therefore differ from that README by up to 0.002 nats (its
  `ce_loop` 4.1173 against this 4.1180 on the mux arm); every per-batch CE is bit-identical
  (3.7964353, 3.9412927, 4.4130 …) and every K-difference here is within-run.
* **`|h_out − h_in|` on an HC residual was NOT used as "the branch's update"** and is not
  reported: `hc_post` mixes the four streams through a Cayley rotation in the same write, so
  that difference is large even with a zero branch. Lead 2 reports the branch outputs
  themselves and lead 3 the whole pass's movement.
* **The window-attention statistics come from weights this probe rebuilds**, because SDPA
  never materialises them. `tests/test_slot_geometry_audit.py` checks that `w @ v`
  reproduces `_window_fallback`'s own output to 2e-6 in fp32, which is the dtype the runs
  use (`use_kernels=false`). The fused Triton window kernel was NOT checked, and is not on
  this path.
* **Attention geometry is read on batch 0 only** (3 rows, 158 valid slots); the branch
  norms, injection, readout and K-curve use all 4 batches.
* **The proposed fixes were not run.** No arm was trained with
  `core_hca_compress_ratio: 16`, so the claim that F1 is not the cause rests on the
  `slot-unpack-free` control, not on a repaired mux arm.
* **The runs were made on a tree carrying another agent's then-uncommitted work** in
  `morph/model/{transformer,tul}.py` and `morph/training/{train,tul_setup}.py`
  (`tul.mux_every_pass`, since landed as `bf87aeb`). It is off by default and
  training-only, and the per-batch token CEs here reproduce the `slot_z_optimize` run's
  exactly, but it is named rather than hidden. `slot_z_optimize.ZSplit` did not run on that tree when this audit was written —
  it unpacked `_tul_core`'s return into six names and the return now has seven — which is
  why this audit carries its own arity-tolerant `Split` subclass. Another agent made the
  parent tolerant the same afternoon, mid-audit; `slot-mux-norm-match` was re-run against
  the changed module and every entry of `results`, `geometry`, `static` and `gates` came
  back identical, so the tables here are unaffected. The subclass is now duplicated logic
  and its docstring says to delete it when that edit lands.
* **Not measured:** whether the coda's insensitivity is a property of this checkpoint's
  training or of the architecture; whether the flat directions are the same on every slot;
  whether the MUX loss (rather than the token CE) responds to the loop's update; and
  whether any of this changes past 5000 steps.
