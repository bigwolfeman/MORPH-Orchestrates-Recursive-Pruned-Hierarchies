# Perceiver: General Perception with Iterative Attention

- **Title:** Perceiver: General Perception with Iterative Attention
- **Authors:** Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira (DeepMind)
- **URL Source:** https://arxiv.org/abs/2103.03206
- **Published:** 2021 (arXiv v1: 4 Mar 2021; ICML 2021 / PMLR v139)
- **PMLR:** https://proceedings.mlr.press/v139/jaegle21a.html (fetched 2026-09-21; title, authors and abstract match)
- **Local source:** `ignore/papers/2103.03206v1-perceiver.pdf`, text extracted with `pdftotext -layout` to `ignore/papers/2103.03206v1-perceiver.txt` (870 lines).

> **Note on this archive.** Unlike some other files in this directory, this is
> **not** a verbatim transcription of the paper's prose. It is a close,
> section-by-section paraphrase built from the full `pdftotext` extraction,
> written to preserve every number, hyperparameter, and architectural claim
> the paper makes. All five result/ablation tables (Tables 1-5) are reproduced
> with their numbers intact, since those are the facts this note exists to
> capture. Read the PDF directly (`ignore/papers/2103.03206v1-perceiver.pdf`)
> for the authors' exact wording.

## Abstract (paraphrase)

Biological systems process high-dimensional input from many modalities
(vision, audition, touch, proprioception) with one perceptual system, while
deep learning models are built per-modality and lean on domain-specific
structure (e.g. the 2D grid locality that every vision model exploits). The
paper introduces the **Perceiver**, a Transformer-derived architecture that
makes few assumptions about the relationship between its inputs — like a
Transformer — but also scales to hundreds of thousands of input elements —
like a ConvNet. It does this with an asymmetric attention mechanism that
iteratively distills the input into a small latent bottleneck. The Perceiver
matches or beats specialized models on classification across images, point
clouds, audio, video, and audio+video: performance comparable to ResNet-50 on
ImageNet while directly attending to all 50,176 pixels with no convolutions,
and state-of-the-art results on AudioSet across all three input
configurations (audio, video, audio+video).

## 1. Introduction (paraphrase)

Strong architectural priors (e.g. spatial locality) are efficient but lock a
model to one modality: swapping from a single image to a stereo pair, to raw
audio, to Lidar point clouds each forces a different architectural family.
The Perceiver instead builds on the Transformer, which makes few assumptions
about input structure but pays for that generality with quadratic
time/memory in the number of input elements. The paper's core idea is a
small set of latent units that forms an attention bottleneck: the input
array is read into these latents through cross-attention, avoiding the
quadratic all-to-all cost of a plain Transformer. Because the model reads the
input repeatedly, informed by what it extracted on previous reads, the
authors describe it as performing something like an end-to-end soft
clustering of the input, with the latents as cluster centers. Because there
is no 2D grid assumption, spatial/temporal structure is supplied instead via
Fourier positional feature encodings concatenated onto every input element.

## 2. Related work (paraphrase, condensed)

- **ConvNets** dominate perception via local, weight-shared, hierarchical
  computation but don't generalize across modalities.
- **Transformers** are flexible but scale quadratically; prior work applying
  them to vision avoids full quadratic attention by patchifying (ViT),
  factorizing into rows/columns, or heavy subsampling. The Perceiver instead
  changes the attention operation itself (asymmetric cross-attention into a
  small latent set) rather than restructuring the input.
- **Multimodal architectures** typically use separate per-modality feature
  extractors and then choose a fusion layer; the Perceiver aims to accept any
  modality combination through one input interface.
- **Top-down / bottom-up processing.** The Perceiver's latent-conditioned
  re-reading of the input is framed as a soft form of top-down attention,
  connected to older ideas in vision (Gestalt grouping, feedback attention).

## 3. Methods

### 3.1 The Perceiver architecture

**Two modules, alternated.** (i) A **cross-attention module** maps a byte
array (e.g. the pixel array) and a latent array to a new latent array. (ii) A
**latent transformer** (a GPT-2-style decoder stack, Radford et al. 2019)
maps a latent array to a latent array. The byte array's size is set by the
data (e.g. 50,176 pixels for a 224×224 ImageNet image); the latent array's
size is a hyperparameter and is much smaller (e.g. 1024 latents on
ImageNet). The model alternates cross-attention and latent-transformer
application — repeatedly projecting the large byte array through a small
attentional bottleneck before processing it with a deep transformer in that
low-dimensional space. **Because weights are shared between instances of the
latent transformer (and between some instances of the cross-attention
module), the model can be read as a weight-shared RNN unrolled in depth
against the same fixed input** (rather than unrolled in time against
different inputs). All attention in the Perceiver is non-causal (no masks).

**Why cross-attention tames the quadratic cost.** For query `Q ∈ R^(M×D)`,
key `K ∈ R^(M×C)`, value `V ∈ R^(M×C)`, plain QKV attention
`softmax(QK^T)V` costs `O(M^2)` in the index dimensionality `M`. The
Perceiver introduces asymmetry: `K` and `V` come from the input byte array
(size `M`), but `Q` comes from a **learned latent array** of size `N`, with
`N ≪ M`. The resulting cross-attention costs `O(MN)` instead of `O(M^2)`.

**Latent transformer cost.** The cross-attention output depends only on the
`Q` (latent) side — the cross-attention layer *is* the bottleneck. Because of
that bottleneck, the following latent self-attention transformer only costs
`O(N^2)` per layer regardless of how large the byte array `M` is, which lets
the authors build much deeper transformers than would otherwise be
affordable at large `M`. As a function of depth `d`: a plain transformer over
bytes costs `O(d·M^2)`; the Perceiver's latent transformer costs `O(d·N^2)`.
The best ImageNet model uses a latent transformer 48 blocks deep in total
(8 cross-attends × 6 latent-transformer blocks each).

**Iterative attention.** Because the bottleneck could otherwise lose detail
from the input, the Perceiver structures itself with multiple byte-attend
(cross-attention) layers, so the latent array can keep extracting new
information from the input as needed — described as analogous to a skip
connection. This lets the design trade expensive-but-informative
byte-attends against cheaper-but-potentially-redundant latent self-attends.

**Weight sharing.** Because of the iterative/RNN-like structure, weights can
be shared between corresponding latent-transformer blocks and/or between
cross-attention modules across iterations. On ImageNet this weight sharing
gives roughly a 10x reduction in parameter count while *also* reducing
overfitting and improving validation accuracy (see Table 5 below). The
resulting architecture is functionally an RNN with a cross-attentional input
projection, a bottlenecked latent dimensionality, and a latent-transformer
recurrent core.

### 3.2 Positional encodings

**Permutation invariance is a problem to solve, not free.** Attention is
permutation-invariant, so a purely attentional model returns the same output
regardless of input ordering — useful for not baking in the wrong spatial
prior, bad because spatial relationships (unlike a ConvNet, which bakes in
locality, weight sharing across space, and small-filter scale invariance)
are not otherwise available to the model at all. The paper injects them via
positional encodings appended to the input features, the standard
Transformer strategy.

**Fourier features, chosen for three properties:** (i) they directly encode
1D/2D/3D positional structure (temporal for audio, spatial for images,
spatiotemporal for video); (ii) the number of frequency bands can be set
independently of the cutoff frequency; (iii) all frequencies up to a target
resolution are sampled log-uniformly. Concretely, each dimension `d`'s
position `x_d ∈ [-1, 1]` is expanded to `[sin(f_k·π·x_d), cos(f_k·π·x_d)]`
for a log-linearly spaced bank of frequencies `f_k` between 1 and `µ/2`
(`µ` = target max resolvable frequency, a Nyquist-style choice), and the raw
`x_d` value is also concatenated in. Final positional encoding size per
input element is `d(2K+1)` for `d` spatial dimensions and `K` bands. This
differs from the NeRF parameterization (powers-of-two bands), which the
authors found numerically unstable past ~15 bands; their log-uniform
parameterization densifies the sampled spectrum instead of exploding the max
frequency as more bands are added.

**Concatenation, not addition.** Unlike standard language-model practice
(add positional encoding to token encoding), the Perceiver *concatenates*
positional and input features before the cross-attention layer — attributed
to language input features typically being larger-dimensional than the
modalities considered here.

**Positional encodings don't reintroduce a domain-specific architecture,**
the authors argue, because: the network can learn to use or ignore the
positional features (rather than the prior being hard-wired into
convolution/pooling structure); the same Fourier-feature scheme adapts to
new domains just by changing dimensionality; and multimodal inputs can each
carry their own positional coding plus a categorical code to disambiguate
domains.

**The latent array itself uses a learned position embedding** (citing
Gehring et al. 2017's ConvS2S-style learned positional embedding) — i.e. the
`N`-latent, `D`-channel array is a single learned parameter tensor that is
the same for every input (it is not derived from the input at all; the
cross-attention module is what lets it become input-conditioned over the
course of the forward pass).

## 4. Experiments

Baselines throughout: ResNet-50 (He et al. 2016), ViT-B (Dosovitskiy et al.
2021), and a plain stack of Transformers fed a downsampled input.

### 4.1 Images — ImageNet

Trained on ILSVRC-2012 with Inception-style preprocessing, 224×224 crops,
RandAugment. Optimizer: LAMB (not SGD — found easier to optimize than SGD
for the Perceiver), 120 epochs, initial LR 0.004, decayed 10x at epochs 84,
102, 114.

**Best ImageNet model:** 8 cross-attends, each reading the full 50,176-pixel
input, each followed by a 6-block latent transformer; single-head
cross-attention; no channel bottleneck inside the transformer's dense
sublayer (same channel count throughout); latent array of **1024 indices ×
512 channels**; Fourier positional encoding with 64 bands, max resolution
224 pixels; weights shared across cross-attends 2-8 and the corresponding
latent-transformer blocks 2-8 (the *first* cross-attend and *first* latent
transformer keep their own, unshared weights — needed because an unshared
44M-ish-parameter-per-block model of this size overfit ImageNet without
sharing). Total ≈44M parameters, comparable to a conv model of similar size.

**Table 1 — top-1 validation accuracy (%) on ImageNet.** Red-labeled rows
(in the original) exploit grid structure; blue-labeled rows do not. Rows 1-2
are literature numbers; rows 3-5 are the same baseline architectures
re-trained by the authors on RGB + Fourier-feature (FF) input, matching what
the Perceiver receives.

| Model | Top-1 (%) |
|---|---|
| ResNet-50 (He et al., 2016) | 76.9 |
| ViT-B-16 (Dosovitskiy et al., 2021) | 77.9 |
| ResNet-50 (RGB+FF) | 73.5 |
| ViT-B-16 (RGB+FF) | 76.7 |
| Transformer (64×64 downsampled input) | 57.0 |
| **Perceiver** | **76.4** |

**Table 2 — top-1 validation accuracy (%) on *permuted* ImageNet**, testing
how much each model relies on grid structure. "Fixed" = one shared
permutation applied to every image; "Random" = a fresh per-image
permutation. All models get identical RGB+FF input features. The rightmost
column gives each model's first-layer receptive field size in pixels.

| Model | Fixed (%) | Random (%) | Receptive field (px) |
|---|---|---|---|
| ResNet-50 (RGB+FF) | 39.4 | 14.3 | 49 |
| ViT-B-16 (RGB+FF) | 61.7 | 16.1 | 256 |
| Transformer (64×64) | 57.0 | 57.0 | 4,096 |
| **Perceiver** | **76.4** | **76.4** | **50,176** |

Both the plain Transformer and the Perceiver are permutation-agnostic by
construction, so their scores don't move between Fixed/Random; ResNet-50 and
ViT-B-16 both collapse hard, especially under Random, because they rely on
grid-local structure that a random permutation destroys. Perceiver and the
Transformer see the *entire* input in their very first layer (receptive
field = full input); ResNet-50's first 7×7 conv sees 49 px, ViT-B-16's 16×16
patch sees 256 px.

**Attention maps (Fig. 3).** For a model whose first cross-attend/latent
transformer are unshared but every later layer is shared, the first-layer
cross-attention maps clearly show input structure (the input dog is visible
in the raw attention map), while later-layer maps look like high-frequency
"tartan"/plaid lattices whose banded structure is attributed to the spatial
frequency structure of the Fourier positional encoding itself. Attention
maps from modules 2 and 7 share overall structure but differ in which
specific pixels are attended, which the authors read as evidence the network
attends to different input subsets at different stages (a cross-attention
map read, not a latent-latent similarity/rank measurement — see reading
note §4).

### 4.2 Sound and video — AudioSet

10s, 1.7M-video, 527-class multi-label dataset; sigmoid cross-entropy loss;
evaluated by mAP. Evaluated on raw audio alone, video alone, and audio+video
jointly. 32-frame (1.28s @ 25fps) training clips; eval splits each video
into eight 32-frame clips and averages scores; augmentation is random flip
and 256×256 crop for video, and time-consistent sampling for audio.

Given AudioSet's scale, the model used is a faster variant of the ImageNet
model: **2 cross-attends** (byte-attends) instead of 8, but **8 latent
transformer blocks per attend** instead of 6, and **no weight sharing**
(sharing wasn't needed to compensate for the smaller iteration count). A
brief experiment with temporal unrolling (one iteration per frame) was
tried; it seemed fine for video but hurt audio, which the authors suggest
needs longer attentional context.

- **Raw audio:** sampled at 48 kHz → 61,440 input elements over the 1.28s
  clip; 2 byte-attend iterations over the full input. Adding Fourier
  features on the audio *magnitude* (not just the time axis, both scaled to
  [-1, 1]) gave up to a 1.0-point mAP boost.
- **Video:** a full 32-frame 256×256 clip is >2M pixels; instead the model
  reads 2×4×4 (time×height×width) space-time patches, giving 65,536 input
  elements, with Fourier features on horizontal, vertical, and time
  coordinates concatenated to RGB. Same architecture as audio, 2 attention
  iterations.
- **Audio+video:** both streams concatenated at the input (65,536 + 61,440 =
  126,976 elements); since fusion happens at the input, both streams are
  padded to the same feature dimensionality using a modality-specific
  learned embedding (fixed size 8 for video, sized as needed for audio) —
  this outperformed passing audio through a linear layer to match video's
  dimensionality.

**Table 3 — AudioSet mAP** (higher is better; best result per column in
bold in the original).

| Model / Inputs | Audio | Video | A+V |
|---|---|---|---|
| Benchmark (Gemmeke et al., 2017) | 31.4 | – | – |
| Attention (Kong et al., 2018) | 32.7 | – | – |
| Multi-level Attention (Yu et al., 2018) | 36.0 | – | – |
| ResNet-50 (Ford et al., 2019) | 38.0 | – | – |
| CNN-14 (Kong et al., 2020) | 43.1 | – | – |
| CNN-14, no balancing & no aug (Kong et al., 2020) | 37.5 | – | – |
| G-blend (Wang et al., 2020b) | 32.4 | 18.8 | 40.2 |
| Attention AV-fusion (Fayek & Kumar, 2020) | 38.4 | 25.7 | 46.2 |
| **Perceiver** | **44.9** | **38.0** | **47.3** |

Perceiver beats CNN-14 (43.1 mAP) on audio despite CNN-14 using SpecAugment
and AugMix and class-balancing, none of which the Perceiver run used; against
the CNN-14 variant without those tricks (37.5 mAP), the margin is 7.4 mAP.

### 4.3 Point clouds — ModelNet40

40-category dataset of 3D point clouds from triangular meshes; 9,843 train /
2,468 test examples, 2048 3D points per example. Preprocessing:
zero-centering, per-point scaling (0.99-1.01) and translation (±0.02),
zero-mean/unit-cube normalization, random rotation as augmentation. Points
are arranged into a 2D grid at random before being fed to every model
(there is no natural grid on this data). The Perceiver's positional encoding
here used a higher max frequency (256) than for images — swept directly
(unlike audio/video, this dataset has no natural sample rate to derive the
frequency from); values above 256 tended to overfit more.

**Table 4 — top-1 test accuracy (%) on ModelNet40** (best result per model
class, selected by test-set score; ViT variants differ in assumed input
patch size).

| Model | Top-1 (%) |
|---|---|
| PointNet++ (Qi et al., 2017) | 91.9 |
| ResNet-50 (FF) | 66.3 |
| ViT-B-2 (FF) | 78.9 |
| ViT-B-4 (FF) | 73.4 |
| ViT-B-8 (FF) | 65.3 |
| ViT-B-16 (FF) | 59.6 |
| Transformer (44×44) | 82.1 |
| **Perceiver** | **85.7** |

PointNet++ uses extra geometric features (e.g. fitted surfaces, face
normals) and more advanced augmentation not used by any of the generic
baselines (including the Perceiver) in this comparison, so it is not a like-
for-like number.

## 5. Discussion (paraphrase)

The Perceiver scales to over a hundred thousand inputs without
domain-specific architecture, opening a path to general perception
architectures that can handle arbitrary sensor configurations and fuse
information both bottom-up and top-down. Flexibility comes with a higher
overfitting risk, which shaped several design choices (crop-relative
positional encoding, weight sharing, RandAugment). Results were strongest on
the largest dataset (AudioSet, 1.7M examples), where the Perceiver beat
strong recent baselines on audio, video, and audio+video; on ImageNet it
lands essentially on par with ResNet-50. The authors note the model still
relies on modality-specific augmentation and positional embeddings despite
reducing architectural priors, and flag fully end-to-end modality-agnostic
learning as open work. They also mention wanting to pretrain the image model
on much larger-scale data in future work.

## Appendix A. Ablations

A separate, smaller "base" Perceiver was used for ablations (distinct from
the main ImageNet model): **no weight sharing** between any transformer or
cross-attention instances, 8 heads per transformer block, 4 transformer
blocks per byte-attend, **2 byte-attends per image**, **512 latents × 512
channels each**. Every module (thanks to the iterative-attention structure)
sees the *entire* input, so hyperparameters (depth, capacity, etc.) can be
swept without shrinking the model's effective receptive field. Batch size
64 across 32 TPUs (kept small so every swept configuration fits in memory);
5M training steps per run, same optimization recipe as the main ImageNet
experiments.

**Figure 5 — line-plot ablations, not a table** (the paper does not give a
tabulated per-point readout, only the plot and the summary sentence; exact
inter-point values below are read off the OCR'd axis ticks and are
approximate, not paper-stated numbers). Four panels, each sweeping one
hyperparameter around the base config: number of latent dimensions (400 →
1000+, channel width), number of latents (400 → 1000+), **number of
cross-attends (1 → 4)**, and number of transformer blocks per attend (2 →
8). Top-1 accuracy on the vertical axis spans roughly 62-74% across the four
panels. The paper's only quantitative claim from this figure is qualitative:
*"Increasing the number of latents, attends and transformers per attends
always seems to help."* The one exception noted in the text: latent
*dimensionality* stopped helping at 1024 channels because optimization
became unstable at that width in this ablation setting. **No exact per-step
top-1 numbers for the cross-attend sweep (1, 2, 3, 4 attends) are given in
the paper text** — see the reading note (§3) for what can and can't be
claimed from this.

**Table 5 — weight sharing ablation**, on the *best-performing* ImageNet
architecture from Tables 1-2 (8 cross-attends, 6 blocks per latent
transformer — this is the main-paper model, not the ablation-base model
above). "W/ weight sharing" shares cross-attention modules 2-8 and the
corresponding latent-transformer blocks 2-8; the first cross-attention
module and first latent transformer keep unshared weights in both rows.

| Config | Valid top-1 (%) | Train top-1 (%) | Params |
|---|---|---|---|
| No weight sharing | 72.9 | 87.7 | 331.3M |
| **W/ weight sharing** | **76.4** | **79.7** | **43.9M** |

Weight sharing costs 7.9 points of train accuracy but *gains* 3.5 points of
validation accuracy (87.7→79.7 train, 72.9→76.4 valid) while cutting
parameter count by 7.5x (331.3M → 43.9M) — read by the authors as weight
sharing acting as a strong regularizer against overfitting at this model
scale, not merely a parameter-efficiency trick.

## Appendix B. Architectural details (paraphrase)

Cross-attention module: input → LayerNorm → linear projections to Q/K/V →
QKV cross-attention → output linear layer (dropout applied here). Latent
self-attention block: same pattern (LayerNorm → Q/K/V → self-attention →
output linear + dropout). Every attention block (cross- or self-) is
followed by a dense (MLP) block: LayerNorm → linear → GELU → linear
(dropout applied on the final linear). All linear layers, including Q/K/V
and the dense-block layers, preserve dimensionality and are applied
identically at every input-index position (equivalent to a 1×1 convolution
along the index axis). The whole architecture is fully residual, Transformer
-style: for both the latent self-attention transformer and the
cross-attention module, the latent-side input is added back to that
module's output.

## Appendix C. Positional encodings and Fourier features (paraphrase)

Positional coordinates for ImageNet are computed from the crop, not the raw
image (Fig. 4) — using raw-image coordinates caused overfitting, plausibly
because the model could latch onto a small set of pixels always paired with
the same (RGB, position) feature; cropping augments both position and aspect
ratio and breaks that shortcut. The chosen Fourier parameterization lets the
max resolvable frequency be set directly and independently of band count
(unlike NeRF's powers-of-two band scheme, where the 64th band would already
be at frequency 2^64 ≈ 1.8e19 — numerically unusable); adding bands instead
densifies sampling of the same target spectrum. For irregularly/finely
sampled signals like ModelNet40 point clouds, the max frequency is treated
as a plain hyperparameter rather than derived from a sample rate.

On AudioSet, computing Fourier features on the audio *magnitude* (not just
time) is related to, but conceptually different from, a spectrogram: a
spectrogram's k-th band uses a term shaped `x(t)·sin(f_k·t)`, while the
Fourier-feature encoding here uses `sin(f_k·x(t))` — i.e. Fourier features of
the audio signal's *value*, not a Fourier transform of the signal in time.
Per Tancik et al. (2020), training on Fourier-feature inputs approximates
kernel regression; here that means the encoding corresponds to the Fourier
transform of a kernel applied to the audio signal, not the transform of the
signal itself.

## Appendix D. Audiovisual attention maps (paraphrase)

Figures 6-7 visualize first-cross-attention-module attention maps for the
best AudioSet model. **Video:** attention maps are sensitive to both static
and dynamic content; static structure consistently concentrates on the
*last* timestep of each attention map, with earlier timesteps showing
spatiotemporal-filter-like structure — read as possible evidence the network
reasons about change over time starting from the clip's end (noted as
possibly specific to these short, 32-frame clips). **Audio:** attention maps
share a low-frequency oscillatory structure, with additional
higher-frequency structure that varies per-map and over time; described as
loosely analogous to but less interpretable than the content modulation seen
in image attention maps.
