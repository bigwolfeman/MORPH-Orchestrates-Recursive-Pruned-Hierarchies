# Reading note: Perceiver vs MORPH's TUL slot loop

> Reading filed 2026-09-21. Source: Jaegle, Gimeno, Brock, Zisserman, Vinyals,
> Carreira, "Perceiver: General Perception with Iterative Attention," ICML
> 2021 / PMLR v139, arXiv:2103.03206. Produced by reading the pdftotext of
> arXiv 2103.03206v1 in full (`ignore/papers/2103.03206v1-perceiver.txt`,
> 870 lines); unverified items are marked. Section 6 (Perceiver IO) is from
> the abstract/architecture description only, not a full read.

## 1. The latent array: N×D per experiment, initialization, sharing across inputs

| Experiment | N (latents) | D (channels) | Source |
|---|---|---|---|
| ImageNet (main, best model) | 1024 | 512 | §3.1/§4.1 (stated directly) |
| Ablation "base" model (Appendix A) | 512 | 512 | Appendix A (stated directly) |
| AudioSet (audio / video / audio+video) | **Not stated** | **Not stated** | §4.2 only says the model is "a faster version of the ImageNet model" with 2 cross-attends and 8 blocks/attend; N and D for this variant are never given a number in the main text or appendix |
| ModelNet40 (point clouds) | **Not stated** | **Not stated** | §4.3 gives the positional-encoding max frequency (256) but never states N/D for this run |

**Initialization.** The latent array is a single **learned parameter
tensor** — the paper states "The latent array uses a learned position
embedding (Gehring et al., 2017)" (§3.1). It is explicitly *input-independent
at initialization*: the array's initial values are the same fixed learned
parameters regardless of what is fed into the model. The cross-attention
module is what lets the latents become input-conditioned, over the course of
the forward pass — the latent's *starting point* carries no information
about the specific input.

**Shared across all inputs?** Yes. There is one learned `N×D` parameter
tensor total, used as the starting latent state for every training and
inference example. This is not qualified anywhere in the ablation or
architecture sections — it's a plain nn.Parameter-style array, no
per-example latent initialization scheme is described or ablated.

## 2. The iterative cross-attention: counts, structure, and exactly what is shared

**ImageNet (main, best model), §4.1 + Table 5:**
- **8 cross-attends** total per forward pass, each one reading the *entire*
  50,176-pixel input array (not a subset).
- **6 latent self-attention transformer blocks** between each cross-attend
  (i.e. cross-attend → 6-block latent transformer → cross-attend → ... ×8).
- Total depth: 48 latent-transformer blocks + 8 cross-attends.
- **Weight sharing, exact statement (Table 5 caption):** *"shares weights
  between cross-attention modules 2-8 and between the corresponding blocks
  of latent transformers 2-8. The first cross-attention module and
  transformer use their own, unshared weights."* So: **the first
  cross-attend, and the first latent-transformer's 6 blocks, are excluded
  from sharing**; cross-attends 2 through 8 share one set of weights with
  each other, and latent-transformer blocks 2 through 8 share their own
  (separate) set of weights, block-position-for-block-position (block 1 of
  iteration 2 shares with block 1 of iteration 3, etc., not with block 2 of
  iteration 2 — this per-position sharing scheme is implied by "the
  corresponding blocks" but the paper does not spell out the indexing any
  more explicitly than that phrase).

**AudioSet, §4.2:**
- **2 cross-attends** (byte-attends), not 8.
- **8 latent transformer blocks per attend** (more than ImageNet's 6, to
  compensate for fewer attends).
- **No weight sharing at all** ("also no weight sharing to compensate for
  smaller size" — read as: the 2-attend model doesn't overfit the way the
  8-attend ImageNet model did, so sharing wasn't needed as a regularizer).
- A brief, separate experiment tried temporal unrolling (one iteration per
  video frame instead of 2 fixed attends): helped or was neutral for video,
  hurt audio. Not the shipped configuration; reported as a side note.

**Ablation "base" model, Appendix A:**
- **2 cross-attends**, 4 latent-transformer blocks per attend, 8 heads per
  block, **no weight sharing** anywhere (used specifically to isolate each
  hyperparameter's effect without sharing as a confound).

## 3. Ablations: number of cross-attends, and weight sharing (every row)

**Weight sharing — Table 5, ImageNet, full table:**

| Config | Valid top-1 (%) | Train top-1 (%) | Params |
|---|---|---|---|
| No weight sharing | 72.9 | 87.7 | 331.3M |
| W/ weight sharing | 76.4 | 79.7 | 43.9M |

Architecture for both rows: 8 cross-attends, 6 blocks per latent
transformer (the main-paper best model, not the Appendix A ablation-base
model). Weight sharing trades 7.9 points of train accuracy for +3.5 points
of *validation* accuracy, at 7.5x fewer parameters (331.3M → 43.9M). Read
straightforwardly, weight sharing is not primarily a compute/memory trick
here — it's the thing that stops this specific ImageNet-scale model from
overfitting.

**Number of cross-attends — Figure 5, NOT a table.** This is the important
caveat for anyone quoting this ablation: **the paper does not tabulate the
cross-attend sweep.** Figure 5 is a 4-panel line plot (latent dimensions,
latent count, number of attends, transformer-blocks-per-attend), each swept
independently around the Appendix A base config (2 attends, 512×512
latents, 4 blocks/attend, no sharing, 5M training steps, batch 64 on 32
TPUs). The "attends" panel's x-axis runs 1→4 (not 1→8); the paper's only
written claim about it is the qualitative sentence: *"Increasing the number
of latents, attends and transformers per attends always seems to help."*
No per-point top-1 numbers are given in prose or a table for 1, 2, 3, or 4
attends. The OCR'd plot axis ticks put the y-range at roughly 62-74% top-1
across all four panels combined, but reading exact per-x-value numbers off
a scanned line plot is not reliable and **is not attempted here** — do not
treat any specific number pulled from that plot as paper-stated. The one
*exception* to "more always helps" the text does call out: latent
dimensionality specifically became unstable to optimize at 1024 channels in
this ablation setting (with 512 latents, unshared, 2 attends) — the paper
does not reconcile this with the main ImageNet model successfully using 512
channels (not 1024) at N=1024, since that's a different, larger, sharing-on
configuration.

**Net read for MORPH:** the one clean quantitative number the paper gives
for "does re-reading the input help" is the *weight-sharing* comparison
(Table 5), not a clean isolated cross-attend-count number — the closest
thing to the latter is a monotonic-and-otherwise-unquantified trend line.

## 4. Any reading of what the latents do across iterations? Collapse / distinctness?

**Not stated as a quantitative claim.** The paper never measures latent-latent
similarity, an effective-rank number for the latent bank, or any explicit
"do the latents collapse onto each other" statement. The closest thing is a
*qualitative* read of cross-attention maps (Fig. 3, §4.1, Appendix D), which
is a different object — it's the attention **from latents to input bytes**,
not latent-to-latent similarity:

- Early cross-attention modules (layer 1) show clear traces of input
  structure ("the input dog is clearly visible in several of the first
  module's attention maps").
- Later modules (layers 2-8, weight-shared in this model) show high-frequency
  "tartan"-like plaid patterns, attributed to the spatial-frequency
  structure of the Fourier positional encoding itself rather than to image
  content.
- Modules 2 and 7 (both drawn from the shared, later-layer weights) show
  "similar structure" to each other, but the paper notes "the specific
  details of corresponding maps do vary, which suggests the network attends
  to different sets of pixels at subsequent stages" — read by the authors as
  evidence against redundant/collapsed iterations, but this is an
  interpretive claim about *where the model looks*, not a measured
  similarity or rank statistic over the latent states themselves.
- Video attention maps (Appendix D, Fig. 6) show a split: the *final*
  timestep of a given attention map looks like the static ImageNet-style
  maps, while earlier timesteps look more like spatiotemporal filters —
  again a qualitative attention-map read, not a latent-state metric.

**Not stated:** any latent-collapse warning, any effective-rank number for
the latent bank, any per-iteration cosine-similarity or drift measurement.
This paper simply does not run the instrument MORPH's TUL work runs
(`loop/core_gain_t0`, the K-curve, the fan stream-rank probe
`lab/divergence/fan_stream_probe.py`). It's a gap the reading
note flags rather than papers over.

## 5. Positional encoding

- **Input side:** Fourier features, one bank of log-linearly-spaced
  frequencies per spatial/temporal dimension, `[sin(f_k·π·x_d),
  cos(f_k·π·x_d)]` plus the raw coordinate `x_d ∈ [-1,1]`, **concatenated**
  (not added) to the raw input feature (RGB, audio amplitude, point
  coordinate, etc.) before the first cross-attention layer. Band count and
  max frequency are set per modality (64 bands / max res 224 for ImageNet;
  audio and video use their sample rate to pick the max frequency; ModelNet40
  sweeps max frequency directly since it has no natural sample rate,
  settling on 256).
- **Latent side:** Yes — the latent array itself "uses a learned position
  embedding (Gehring et al., 2017)" (§3.1). This is folded into the same
  single learned `N×D` parameter tensor discussed in §1 above, not a
  separate additive positional term. The paper does not describe the latents
  as position-*free*; they carry position from initialization, just not
  Fourier-feature position, and not per-input position — a learned, static,
  shared-across-inputs positional identity per latent slot.

## 6. Perceiver IO (arXiv 2107.14795) — from the abstract/architecture description only, not a full read

Fetched only the abstract and architecture-overview text of Perceiver IO
(arXiv:2107.14795), per instruction, not archived locally and not read in
full. From that limited read: **Perceiver IO moved away from the original
Perceiver's repeated cross-attend structure.** Instead of alternating
cross-attention and latent self-attention multiple times through the
forward pass, Perceiver IO uses **one encoding cross-attention** (input →
latent, once), then a **deep latent self-attention transformer** (all
further computation happens purely in latent space, no further reads of the
input), then **one decoding cross-attention** that queries the final latent
state with output queries to produce arbitrarily-shaped outputs. The
querying mechanism is the paper's headline addition over the original
Perceiver (its abstract: it "augments the Perceiver with a flexible
querying mechanism that enables outputs of various sizes and semantics").

This is a genuine architectural fork from the paper archived in this
directory, not a strict superset: Perceiver IO trades the original's
repeated-read iterative attention for a single-read/deep-latent-processing
design plus a general output decoder. Caveat: this section is built from an
abstract-and-overview-level fetch, not the full Perceiver IO PDF/text — if
a future note needs exact per-layer counts or ablations from Perceiver IO,
that needs its own full read and its own archive file.

## 7. Where MORPH's TUL slot loop is the same as, and where it differs from, Perceiver

1. **Same: an asymmetric bottleneck read, not full attention.** Perceiver's
   cross-attention reads a large byte array (M≈50k) into a small latent set
   (N=1024) at `O(MN)` cost (§3.1). TUL's slot re-reads a span's token
   states through attention into one slot cell each pass — the same
   asymmetric-attention-into-a-bottleneck shape, just with the bottleneck
   sized per-span (one slot, or K=4 under the fan arm) instead of
   per-sequence.
2. **Same: a near-input-independent learned seed.** Perceiver's latent array
   is one learned `N×D` tensor, identical at init for every input (§3.1,
   §1 above). TUL's slot seed is `E_slot` (one learned embedding) plus the
   span's last token embedding — almost as input-independent as Perceiver's
   pure learned array, differing only by that one token's worth of
   conditioning.
3. **Same: weight-shared iteration read as an unrolled RNN.** Perceiver
   explicitly frames its weight-shared repeats as "an RNN... unrolled in
   depth to the same input" (§3.1). MORPH's core loop is exactly this: one
   weight-shared core run for T≈6 Poisson-sampled passes over the slot
   states.
4. **Same shape, different frequency: repeated cross-attends vs. repeated
   token-state reads.** Perceiver's *shipped* ImageNet model re-reads the
   full input 8 times (§4.1). TUL's slot loop re-reads its span's token
   states via attention on every one of its T passes — structurally the
   Perceiver's repeated-cross-attend design, not Perceiver IO's
   single-encode design (§6).
5. **Different: Perceiver never measures whether the repeats are pulling
   their weight; TUL does, and the answer is "barely."** §3-4 above show
   Perceiver has no per-iteration ablation isolating cross-attend count with
   real numbers, and no latent-collapse/rank measurement at all. MORPH's own
   instrument (K-curve, `loop/core_gain_t0`) shows pass 1 does essentially
   all the work and passes 2-6 add ~0.002 nats — the exact question
   Perceiver's ablation section gestures at (Fig. 5's "more attends always
   seems to help," unquantified) but never actually answers with a number.
6. **Different: Perceiver's bottleneck is genuinely mixing modalities/scale;
   TUL's bottleneck is genuinely mixing time.** Perceiver's `N≪M` gap exists
   to make an otherwise-intractable 50k-126k-element input tractable at all.
   TUL's one-slot-per-span bottleneck exists to compress a variable-length
   span (≤32 tokens) into a fixed-size planning unit for the next span's
   decode — a sequence-modeling motivation, not a compute-tractability one.
7. **Different, and the one Perceiver IO's fork is directly relevant to:**
   TUL's coda reads the slot's *exit* state through attention over
   `prefix_k` cells to decode the next span — structurally Perceiver IO's
   **output-query decoding cross-attention** (§6), not anything in the
   original Perceiver (which has no decode-side cross-attention at all —
   ImageNet/AudioSet/ModelNet40 all pool/classify directly off the final
   latent state). TUL is closer to a hybrid: Perceiver's *repeated-encode*
   front half, bolted to Perceiver IO's *query-decode* back half.
8. **Different: the fan arm is a literal N-latents-per-unit instantiation of
   §1's `N` axis, and it hits the same flat-ish wall.** Perceiver's own
   ablation says more latents "always seems to help" on ImageNet
   classification (Fig. 5, unquantified). MORPH's fan arm gives each span
   K=4 latent streams instead of 1 — the same axis Perceiver's `N` sweep
   is on, just moved from "1024 latents shared across one whole image" to
   "K=4 latents for one ~32-token span" — and per the project's fan-arm
   notes (`fan-repulsion-finds-a-constant-axis.md`,
   `fan-diversity-terms-get-gamed.md`) the four streams only stay apart
   under a within-slot volume term (pass-1 centered rank 2.8 of a ceiling
   of 3), and the weight-shared loop then contracts them (pass 6: 2.0).
   That is exactly the failure mode Perceiver's paper never checks for
   (§4 above) because it never measures latent distinctness at all.
