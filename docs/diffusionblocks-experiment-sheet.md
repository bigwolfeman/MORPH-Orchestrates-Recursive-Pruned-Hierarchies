# DiffusionBlocks — experiment tracking sheet (PRE-REGISTERED)

**Paper:** [arXiv:2506.14202](https://arxiv.org/abs/2506.14202) (ICLR 2026). Archive:
[`references/training-objectives/2506.14202.md`](references/training-objectives/2506.14202.md).
Design analysis: [`diffusionblocks-morph-assessment.md`](diffusionblocks-morph-assessment.md).

**Rule for this file: every expected number is written BEFORE the run.** Result columns stay
empty until a run fills them. If a result lands outside its pre-registered band, that is a finding
to write down, not a band to quietly widen. Do not edit an Expected cell after its arm has started.

**Nothing has run. No GPU work has started. The 5090 is busy with another project.**

---

## 1. Metric contract

### 1.1 The three performance metrics (Wolfe's ask)

| Metric | wandb key | Status | Definition |
| --- | --- | --- | --- |
| Throughput | `perf/tokens_per_sec` | **exists** (`train.py:2048`) | `steps_per_sec × batch × seq_len × (bag_size or 1)` |
| VRAM | `perf/peak_mem_alloc_mib`, `perf/peak_mem_reserved_mib` | **exists** (`train.py:2007`) | `max_memory_allocated` / `max_memory_reserved`. Report **both** — the alloc/reserved gap is the fragmentation tax. |
| FLOP efficiency | `perf/mfu`, `perf/model_tflops` | **MISSING — must be built (G1)** | see §1.2 |

### 1.2 FLOP efficiency needs two new keys and one measured constant

We currently log a FLOP *proxy* only when TUL is on (`tul/layer_passes_per_token`). That is not
enough here, because DiffusionBlocks changes the **position count** as well as the pass count.

Add:

- `perf/positions_per_token` = `L_total / seq_len`. 1.0 for A0. 1.125 for TUL at `prefix_k=2`,
  `max_slots=64`, `seq_len=1024`. **2.0×** those under the clean|noisy concatenation (App. E.4).
- `perf/layer_passes_per_token` — promote the TUL-only key so **every** arm reports it. This key
  is REALIZED (measured passes): A0 reports ≈ 42.0, not 44.0, because depth is
  `clamp(Poisson(6),1,8)` with mean 5.67 — see the G1 fix. The nominal 44.0
  (4 prelude + 6 core × T̄=6 + 4 coda) lives in `perf/flop_proxy`.
- `perf/flop_proxy` = `layer_passes_per_token × positions_per_token`. One comparable number
  across every arm in this sheet. A0 = 44.0.
- `perf/model_tflops` = analytic step FLOPs ÷ step time.
- `perf/mfu` = `model_tflops / ceiling_tflops`.

`ceiling_tflops` is **measured, not quoted from a spec sheet** (G2): a dense bf16 GEMM sweep at
MORPH's actual shapes on this 5090. Quoting a marketing TFLOPS number would make every MFU in this
sheet wrong in the same direction, and the ternary/MORTAR path does not reach dense peak anyway.

**How the numerator gets measured.** No single tool does it (all four verified present on this box,
torch 2.12.1):

| Path | Gives | Catch |
| --- | --- | --- |
| hand-written analytic model | exact, reproducible from the Hydra config alone | we write it; **must** cover the Triton kernels |
| `torch.utils.flop_counter.FlopCounterMode` | analytic, per-module, free | **blind to Triton** — fused attention, HC, GLA, decode and CE are all custom, so it undercounts. Cross-check for the aten parts only |
| `ncu` (installed at `/usr/bin/ncu`) | real hardware counters | serialises kernels, needs profiling perms; one-off validation, never per-step |
| `nvidia-smi` | utilisation % | **no FLOP counter at all** — do not use it for this |

Logged number = the analytic model. `FlopCounterMode` validates its aten half; one `ncu` run
validates the whole thing once. Log the analytic model's version alongside `perf/mfu`, or an MFU
from run A is not comparable to run B.

### 1.3 Quality: two metric families and ONE bridge

This is the trap in the whole programme. Per App. E.4, *"computing traditional perplexity is
non-trivial"* under DiffusionBlocks — the CE it reports is conditioned on a σ-noised target, so it
is a reconstruction number, not a likelihood.

| Family | Members | Metric | Comparable to |
| --- | --- | --- | --- |
| **NTP** | A0, A1, A1r, A3 | `val/ppl_tokens` (val CE) | each other **only** |
| **Diffusion** | every DB arm | σ-conditioned CE at `σ_min` | each other **only** |
| **Bridge** | all arms | **gen-PPL(GPT2-XL)** + **MAUVE** + **rep4@512** | everything |

**Never put a DB arm's CE next to A0's CE in a table.** The bridge metrics are the only legal
cross-family comparison, and they are the paper's own protocol (App. E.4: 5 continuations of 50
tokens from 1K prompts, MAUVE scaling factor 2, teacher gen-PPL).

Bridge metrics are computed **post-hoc from saved checkpoints in a separate process**, never
in-training — GPT2-XL is ~3 GB bf16 and we are not putting a teacher in the training job's VRAM.

---

## 2. Shared baseline anchors (MEASURED, not predicted)

From [`ablation-ledger.md`](ablation-ledger.md), 5090, `tul_short.yaml` (seq 1024 × batch 14 ×
20k steps = 287 M tokens, TST off, prune/carve/route off, dense),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

| Arm | peak alloc | s/step | tok/step | passes/token | flop_proxy | **tok/s** | 20k steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 baseline | 20.32 GB | 0.947 | 14336 | 44 | 44.0 | **15,138** | 5.3 h |
| A1 TUL | 24.06 GB | 0.544 | ~14462 | 10.68 | 12.0 | **26,584** | 3.0 h |
| A3 shallow (`n_core 0`) | ~17.7 GB | ~0.30 | 14336 | 8 | 8.0 | **47,787** | ~1.7 h |
| A0 @ batch 16 | 22.92 GB | 0.99 | 16384 | 44 | 44.0 | 16,549 | 5.5 h |
| A1 @ batch 16 | **OOM** | — | — | — | — | — | — |

### 2.1 The launch-bound cost model these anchors imply

Fit `s/step ≈ a + b · flop_proxy` through A0 (44, 0.947) and A3 (8, 0.30):

```
b = (0.947 − 0.300) / 36 = 0.01797 s per unit proxy
a = 0.300 − 8 × 0.01797 = 0.156 s   ← fixed, launch/overhead floor
```

**A0's step is 16 % fixed overhead; A3's is 52 %.** Every DB arm lands in the low-proxy regime
where `a` dominates. This is why the FLOP win will NOT convert 1:1 into tok/s, and why the
**memory** win matters more than the FLOP win: it buys batch size, and batch size is the only lever
that amortises `a`. Say this out loud now so nobody is surprised later.

`a` is itself a prediction that DB will make **worse**: AdaLN σ-conditioning adds kernels on every
norm, and target-noising adds work per step. Pre-registered: `a` rises to **0.17–0.21 s**.

**Reviewer note (2026-08-19) — the fit fails its only held-out point.** The model predicts A1
(proxy 12.0) at `0.156 + 12×0.018 = 0.37 s`; A1 measured **0.544 s** (+46 %). Cause: the proxy
counts positions, but launch overhead counts layer APPLICATIONS — A1 still launches the full T×6
core stack on tiny 64-slot tensors, plus TUL pack/gather work, none of which the proxy sees. DB
training is single-pass, so DB-1/DB-2 sit in the regime the fit was built in; DB-3/DB-4 inherit the
TUL residual and their §4.2 bands are widened for it. Two smaller corrections: (a) A0's REALIZED
passes/token is ≈ 42.0, not 44.0 — depth is `clamp(Poisson(6),1,8)`, mean 5.67 — so the fit's
x-values are nominal; refitting on realized A0 gives `a ≈ 0.148, b ≈ 0.019` (conclusions
unchanged). (b) `a` was measured WITH the front/back static CUDA graphs live; if DB arms cannot
keep graph capture (plan A5), their `a` rises for that reason too. The bands stay a hypothesis
until the first ~200 measured steps of each arm check them.

---

## 3. What is being tested — the algorithms, stated precisely

### 3.1 The shared conversion (all DB arms)

1. **VE noising in embedding space** (App. B): `z_σ = y + σ·ε`, `y` = clean target token embedding.
   The target's **scale must be pinned** — see R3. The paper says "L2-normalise against embedding
   collapse" citing Diffusion-LM (Li et al. 2022), but the binding constraint is scale, not
   diversity: EDM's preconditioning *and* its weighting `w(σ) = (σ²+σ_data²)/(σ·σ_data)²` are both
   parameterised by `σ_data`, fixed at 0.5 in every one of their experiments. If the embedding scale
   is free and learned, `σ_data = 0.5` is a fiction and `w(σ)` is mis-scaled at every σ — and the
   model can escape the task outright by inflating `‖y‖` until σ stops corrupting anything. So the
   requirement is "make `σ_data` a number we actually know", which L2 normalisation is one way to
   achieve and not the only way.
2. **log-normal `p_noise`**, `log σ ~ N(P_mean, P_std²)`. Their actual values, App. E.1 —
   `P_mean = −1.2`, `P_std = 1.2`, `[σ_min, σ_max] = [0.002, 80]`, Euler 50 steps at inference
   unless stated. Pin these; "EDM defaults" is not reproducible, and the equi-probability
   boundaries `σ_b = exp(P_mean + P_std·Φ⁻¹(q_b))` are computed FROM them. Note their AR text
   run used only **4** steps, not 50 (App. E.4) — see DB-13.
3. **Equi-probability σ partitioning** (§3.3) — `σ_b = exp(P_mean + P_std·Φ⁻¹(q_b))`. Table 7 says
   this matters far more than the layer split (FID 38.03 vs 43.53), so it is not optional.
4. **EDM weighting** `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, `σ_data = 0.5`, plus EDM preconditioning.
   App. C calls the weighting *crucial* for equi-probability to work.
5. **Overlap** `γ = 0.1` (their text-generation setting, not the 0.05 default).
6. **Euler seam, EDM sign** — `z_b = α z_{b-1} + (1−α) D_θ`, `α = σ_b/σ_{b-1} ∈ (0,1)`.
   **NOT** the sign as rendered in v4, which makes noise increase down the schedule
   (assessment §4.5). Gate G3 settles this against the authors' code before any arm runs.
7. **Causal consistency** (App. E.4) — clean|noisy concatenation with a modified causal mask so
   noisy targets attend clean past. Doubles positions. New Triton mask path (risk R1).
8. **HC stays inside a block; the σ-blend is the seam between blocks** (assessment §4.4). The n=4
   Cayley stream carrier is untouched.
9. **Loss is cross-entropy, not L2** (App. B, AR case).

### 3.2 Arm DB-B1 — recurrent-depth mode, B = 1 (the paper's literal Huginn setting)

The whole 4:6:4 net is **one** denoiser. No partition. Sample σ, build `z_σ`, one forward pass,
predict `y`. Core carrier is initialised from `z_σ` instead of `input_norm(prelude_out)`. Inference
keeps the T-iteration loop, each iteration one Euler step down the schedule.

- **This is arm 1, not B=3.** It is the setting the paper validated on MORPH's closest relative, and
  it isolates *"does the diffusion objective work on MORPH at all"* without also testing block
  independence and without inheriting any of the §5.4 schedule-stretch hazards.
- Training passes/token = 4 + 6 + 4 = **14**, positions ×2 → `flop_proxy` **28.0**.
- BPTT is gone. `bptt_depth` is irrelevant.

### 3.3 Arm DB-B3 — prelude / core / coda as 3 blocks (Wolfe's structure)

Yes: one block for prelude, one for the core, one for the coda. It splits on seams MORPH already
has, so **the core stays weight-tied and no layers are added.** It also matches the paper's best
cell (equi-probability σ + an even layer split, Table 7).

**Poisson depth survives, and training never sees T.** An earlier draft of this sheet cut the σ
range into 8 discrete Euler steps and picked the block from the sub-interval. That was wrong — it
silently fixed `T = 6` and killed Poisson depth sampling. The correct split:

- **Training samples σ CONTINUOUSLY** from the log-normal `p_noise`. It takes no Euler step, so it
  never needs T. The denoiser conditions on **σ, not on Δσ**.
- **Inference discretises**: `1 + T + 1` Euler steps with `T ~ Poisson(6)` capped 8, `Δσ` set by the
  realised T. One trained model serves any step count — the paper does 50 by default, 1000 for the
  NoProp comparison, and 4 for AR text.

So Poisson-T is free, and TUL's **per-slot** depth survives too: each slot walks its own number of
Euler steps ("per-idea refinement count"). Better than free — today Poisson-T forces the map to
generalise across depths by brute force (`references.md` §1); under σ-conditioning the map is told
where it is on the trajectory, so depth generalisation is explicit instead of emergent.

**Two knobs, not one.** The paper does not distinguish "what σ range does block b own" from "how
often is block b trained" — but not because of weight tying: its partition is equal-mass, and it
samples blocks **uniformly at random per iteration** (App. E.1), so the two rules coincide there.
They separate here only because our mass split is unequal. MORPH's core is tied and applied T
times, so we must pick each knob explicitly:

| Knob | Setting | Why |
| --- | --- | --- |
| σ interval boundaries | probability mass **1/8 : 6/8 : 1/8** (prelude : core : coda), T̄ = 6 | keeps the `1 + T + 1` Euler steps evenly spaced in probability mass — the equi-probability rule of §3.3 applied per *step* |
| block visit frequency | **uniform, 1/3 each** (default; a sweepable knob) | mass-proportional visits would give the prelude and coda 1/8 of steps each — 2500 effective updates over a 20k run, against 20k today. That starves them. Decoupling costs nothing. |

Visit the core → sample σ inside its wide interval. Visit the prelude → sample inside its narrow
one. Geometry stays right; every block gets a third of the updates.

```
E[passes/token] = (1/3)(4) + (1/3)(6) + (1/3)(4)  = 4.67
flop_proxy      = 4.67 × 2.0 (clean|noisy)        = 9.34      ← 4.7× under A0's 44.0
```

**DB-12 (new arm):** block-visit distribution `uniform` vs `mass-proportional` (1/8 : 6/8 : 1/8).
Isolates whether starving the prelude and coda actually hurts. Cheap. **Reviewer correction:**
uniform visits ARE the paper's stated rule (App. E.1: "blocks are sampled uniformly at random for
each iteration"); the place this design departs from the paper is the **unequal mass split**
(1/8 : 6/8 : 1/8 against the paper's 1/B per block) — DB-9 covers that axis.

**Conditioning sub-variants (the 320-channel question, risk R2):**

| Arm | Core block conditioned on | Independence | Extra cost |
| --- | --- | --- | --- |
| DB-B3 | `x0` only (post-embed, pre-prelude — `transformer.py:924`) | **true** — no prelude forward | none |
| DB-B3p | a no-grad prelude forward | partial | +4 no-grad passes |

`x0` reaches the core through `ChannelInject` into the ctx slice — **320 of 1024 dims**, additive,
`log_scale` init 0. Today the prelude builds the representation and x0 tops it up. In DB-B3 that
320-dim slice is the core's *only* view of the input. DB-B3p is the fallback if that pipe is too
thin, and the DB-B3 → DB-B3p delta **is the measurement of how thin it is.**

### 3.4 The TUL cross

Under TUL the core runs on slot positions only (64 slots vs 1024 tokens), so the core term collapses
even though it is already a single pass:

```
core term = 6 passes × (64 / 1024)                              = 0.375 passes/token
E[passes/token] = (1/3)(4) + (1/3)(0.375) + (1/3)(4)            = 2.79
positions_per_token = 2 × (1024 + 2×64)/1024                    = 2.25
flop_proxy                                                       = 6.28   ← 7.0× under A0
```

Note the prelude and coda now dominate: at 1/3 visits each they contribute 2.67 of the 2.79 passes.
Under TUL the core is nearly free, so **the prelude and coda become the whole cost**, and the
`n_coda: 8` reinvestment cell (`TUL-A1+` in the ledger) becomes the interesting knob rather than
core depth.

**Correction to my earlier read.** Last turn I said TUL's win "largely evaporates" under the
recurrent-depth mode. That was wrong. TUL's saving is a **position** saving (core on 64 instead of
1024 positions), not a **loop** saving, so it survives the loop's removal. What shrinks is the
wall-clock multiplier, because at `flop_proxy` 6.28 the step is ~55–60 % fixed overhead `a`.
*(Reviewer fix: an earlier draft said 2.88 here — a leftover from the superseded mass-proportional
visit arithmetic.)*

**The genuinely tricky part — what is a slot's denoising target?** TUL slots carry no token label
(spec: slot label = first token of the next span; the slot's core state has no loss of its own). A
core block that only sees slots therefore has no `y` to denoise toward. Three options, and the
choice is a real fork:

**Wolfe, 2026-08-19:** *"a tul hidden should decode to the first punctuation (not commas)
or 32 tokens."* That is the SPAN, not a single token — and it matches the implemented
segmentation exactly (`base.yaml`: `boundary_chars: ".;!?"` with no comma,
`boundary_substrings: ["\n", "—", "–", "--"]`, `min_span: 4`, `span_cap: 32`). It changes
the target below: a slot's job is to carry its whole span, so a single-token target
under-specifies it.

| Option | Slot target `y` | Verdict |
| --- | --- | --- |
| T-a | next span's first token embedding (TUL's existing slot label) | **demoted to fallback.** Still a legal prediction target, but per Wolfe's clarification it under-specifies what a slot must carry: one token cannot represent a span of up to 32. |
| **T-d** | **the span's token embeddings, denoised at the `prefix_k` coda positions the slot already projects into** | **NEW pre-registered choice.** The slot hidden must decode to its span (first non-comma punctuation, or 32 tokens). Supervise the span through the prefix positions that already exist (`tul.prefix_k`, Block Transformer App. F.2 / Fig 3f) rather than inventing a second head. Keeps it a *prediction* target, not autoencoding. |
| T-b | the span bag-mean (which is already the slot *input*) | **rejected.** Span autoencoding — exactly what the spec warns against (LCM / CoCoMix / BT §4.2). |
| T-c | no slot objective; core gets gradient through the coda | fallback. Kills core independence, collapsing B=3 → B=2 (prelude \| core+coda) for TUL arms. |

**Consequence for the arms.** T-d needs the span→prefix supervision wired before DB-3/DB-4
can run, which is more work than T-a was. DB-3/DB-4 therefore move behind DB-1/DB-2 rather
than beside them, and `db_step + slot_layout` keeps raising until T-d exists. `prefix_k` (2
by default) also becomes a live variable for these arms: 2 coda positions carrying a span of
up to 32 tokens is a compression ratio worth sweeping, and arm `TUL-prefix1` in the ledger
already exists for the non-DB case.

---

## 4. Pre-registered expectations

Bands are wide on purpose. A band this wide that still misses is real information.

### 4.1 Phase 0 — instrumentation gates, NO GPU TRAINING

| ID | Gate | Expected | Result |
| --- | --- | --- | --- |
| G1 | `perf/{layer_passes_per_token, positions_per_token, flop_proxy, model_tflops, mfu}` land for every arm | Nominal (analytic from config, T̄=6): `flop_proxy` A0 = **44.0**, A1 = **12.0**, positions 1.0 / 1.125 — exactly. Realized (`layer_passes_per_token`, measured): A0 = **42.0 ± 0.5** (depth is `clamp(Poisson(6),1,8)`, mean 5.67 — NOT 6), A1 ≈ 10.68. **Reviewer fix:** the original gate demanded a realized counter report exactly 44.0; it would have failed on the clamp. Anything outside these = counter bug, fix before proceeding | |
| G2 | measured bf16 dense GEMM ceiling at MORPH shapes on this 5090 | a number, plus the shape sweep it came from. Expect well under any marketing figure | |
| G3 | EDM sign resolved against the authors' released code ([github.com/SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks); README suggests ViT-only — the sampler still settles the sign) | v4's rendered Eq (3)–(5) sign is a typo (it starts at Eq (3): a correct Euler step of Eq (1) reads `z + Δσ·σ∇log p`); code uses `α = σ_b/σ_{b-1}`. Fallback if the sampler is absent: the derivation + a landing unit test (`y + σ_b·ε`) in our code | |
| G4 | A0 shape facts reproduce from the ledger on today's master | 0.947 s/step ± 5 %, 20.32 GB ± 0.5 GB. Drift here invalidates every anchor in §2 | |
| G5 | `slot_layout=None` still bit-identical; DB off is bit-identical to today | step-0 loss **11.2379** (the A0 marker). **Reviewer note:** this needs a GPU forward with the Triton kernels — it is confirmed at the first GPU window, at Phase-B step 0. The Phase-A form of this gate is a CPU forward-parity test in the `tests/test_tul_forward.py` pattern; step-0 loss alone is a one-point forward check, the CPU test covers the path | |

### 4.2 Phase 1 — the decisive 2×2, plus DB-B1

All at `tul_short.yaml` shape (seq 1024, batch 14, 20k steps) so every cell is paired with §2.

Recomputed after the §3.3 fix (uniform 1/3 block visits, not mass-proportional).

| ID | Arm | passes/tok | pos/tok | flop_proxy | Expected s/step | Expected tok/s (vs A0) | Expected peak alloc | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DB-1 | DB-B1 (B=1, whole net) | 14.0 | 2.0 | 28.0 | 0.62 – 0.80 | 18k – 23k (**1.2 – 1.5×**) | 14 – 19 GB | |
| DB-2 | DB-B3 (`x0` conditioning) | 4.67 | 2.0 | 9.34 | 0.33 – 0.47 | 30k – 43k (**2.0 – 2.9×**) | 11 – 16 GB | |
| DB-3 | DB-B3 + TUL (T-a target) | 2.79 | 2.25 | 6.28 | 0.27 – 0.47 | 31k – 53k (**2.0 – 3.5×**) | 13 – 19 GB | |
| DB-4 | DB-B1 + TUL | 8.375 | 2.25 | 18.8 | 0.47 – 0.68 | 21k – 30k (**1.4 – 2.0×**) | 15 – 21 GB | |

If the shifted-`x0` conditioning replaces the clean\|noisy concatenation (assessment §4.3 / open
question in conversation), `pos/tok` halves for every row and each flop_proxy halves with it —
DB-2 → 4.67, DB-3 → 3.14. Those are NOT pre-registered yet; the concat numbers above are the
committed prediction until that design call is made.

**Reviewer widening (2026-08-19, before any run).** DB-3/DB-4 upper s/step edges widened
(0.40→0.47, 0.62→0.68): the cost model misses its held-out point A1 by +46 % (§2.1 note), and the
missing term — TUL pack/gather overhead, invisible to `flop_proxy` — is carried by exactly these
two arms. DB-1/DB-2 are untouched (single-pass, full positions: the fit's own regime). Widened
while nothing has run; the no-edits-after-start rule is intact.

**Reasoning behind the tok/s bands.** Cost model `a + b·proxy` with `a` raised to 0.17–0.21 for the
AdaLN + noising overhead, `b = 0.018`. DB-2: `0.19 + 11×0.018 = 0.39 s`, band widened for the 2×
sequence hitting attention superlinearly. All bands are **below** what the raw FLOP ratio would
predict, deliberately — that is the launch-bound floor in §2.1.

**Reasoning behind the VRAM bands.** Two forces fight. Down: no BPTT (A0 checkpoints
`bptt_depth=4 × 6 = 24` core layer activations; DB stores 6 unrolled once), and in B=3 only one
block holds grads + optimizer state. Up: 2× positions on the active block, AdaLN params, and the
noised-target tensor. Net expected win is **real but not 3×** — this is the App. G claim meeting our
actual bottleneck, which is activations at sequence length, not parameters.

### 4.3 The three predictions I most want to be wrong about

| # | Prediction | Why it matters | Falsified by |
| --- | --- | --- | --- |
| P1 | **The memory win is worth more than the FLOP win**, because it lifts batch 14 → 24+, and batch is the only lever on the `a=0.156 s` floor. A1 currently **OOMs at batch 16**. | If true, the next run after Phase 1 is a batch sweep, not another arm | DB arms landing ≥ 19 GB, i.e. no batch headroom bought |
| P2 | **DB-B3 ≈ DB-B1 on bridge quality** despite 2.5× less compute, because Table 8 shows moderate `B` *beating* `B=1` (FID 9.90 vs 12.09). *Reviewer caveat: Table 8 is ImageNet FID on untied plain-layer blocks; for language the paper's best was B=4 (App. F). Directional support only — not the same experiment, and not a weight-tied core.* | Decides whether block independence is free or paid for | DB-B3 gen-PPL more than 10 % worse than DB-B1 |
| P3 | **The 320-dim `x0` slice is wide enough** — DB-B3 ≈ DB-B3p within noise, so true block independence holds. *Pre-registration note: I predicted the opposite (too thin); Wolfe overruled on architecture knowledge. Recording both so the outcome scores somebody.* | Decides whether "true block independence" survives contact with MORPH's channel layout | DB-B3p beating DB-B3 by a visible margin |

### 4.4 Bridge quality, pre-registered

Post-hoc, from checkpoints. A0/A1 numbers do not exist yet either — they must be measured on the
**same** protocol for the bridge to mean anything.

| Arm | gen-PPL(GPT2-XL) ↓ | MAUVE ↑ | rep4@512 ↓ | Result |
| --- | --- | --- | --- | --- |
| A0 (reference) | measure first | measure first | measure first | |
| DB-1 (B=1) | within ±15 % of A0 | ≥ 0.85 × A0 | ≤ A0 | |
| DB-2 (B=3) | within ±15 % of DB-1 (P2) | ≥ 0.9 × DB-1 | ≤ DB-1 | |
| DB-3 (B=3+TUL) | within ±20 % of DB-2 | ≥ 0.85 × DB-2 | ≤ DB-2 | |

Their Huginn gain (MAUVE 0.49 → 0.70) came at **3× the epochs**, so we do **not** pre-register a
quality *improvement*. Parity at 2–4× the throughput is the win being chased.

### 4.5 Kill criteria — stop the arm, write it down, do not tune around it

1. Non-finite loss → `train.py`'s existing self-abort fires. Do not restart with a lower LR without
   recording the σ that was sampled at the abort step.
2. `flop_proxy` measured more than 25 % off its pre-registered value → **counter or design bug**,
   not a result. Stop and fix G1.
3. `clip_factor` sustained ≪ 1 → the loss curve is being driven by a gradient the clip discards.
   The ledger already warns to read this before believing any cross-arm loss comparison. Under
   per-σ EDM weighting, gradient scale varies a lot by sampled σ, so expect this to be noisy and
   **log the σ→grad_norm relation**.
4. Embedding collapse (all embeddings → one vector, App. C). Monitor mean pairwise embedding cosine
   every 20 steps. Rising toward 1.0 → L2 normalisation is not doing its job. Blocks everything.
5. A0 parity marker (step-0 loss 11.2379) broken with DB off → the conversion leaked into the
   baseline path. Highest-severity failure in the sheet.

### 4.6 Hazards that must be neutralised BEFORE any B=3 arm runs (assessment §5.4)

Per-block sampling means a block gets gradient on `1/B` of steps, so every step-counted schedule we
own silently stretches. `tul_short.yaml` runs dense with prune/carve/route **off**, which hides two
of these — they detonate the moment a DB arm runs on `base.yaml`.

| Knob | Today | Under DB-B3 | Action |
| --- | --- | --- | --- |
| `ademamix_t_alpha` | 8000 steps | ~1/B the effective updates per block | divide by B, or state that it was not and why |
| `ademamix_beta3_warmup_start` | 0.9 | same stretch, in the divergence-prone direction | re-derive per block |
| MORTAR `prune_start` / `prune_interval` | 3000 / 167 | Taylor saliency accumulates B× slower | scale cadence by B **and** assert on logged `[prune] density=…` before carve |
| `compact_step` | 29000 | would carve a still-dense model → **K/C = 1.0** | the CLAUDE.md gotcha. Hard assert, not a comment |
| `route_start` | 30000 | inherits the same stretch | scale by B |

---

## 5. Phase 2 — only if Phase 1 clears

| ID | Arm | Isolates | Gate to reach it |
| --- | --- | --- | --- |
| DB-5 | DB-B3p (no-grad prelude conditioning) | P3, the 320-channel pipe — confirmatory only (R2 downgraded) | DB-2 quality below band |
| DB-6 | batch sweep 14 → 20 → 24 → 28 on the best Phase-1 arm | P1 — turn the memory win into tok/s | any DB arm ≤ 16 GB |
| DB-7 | `γ ∈ {0.0, 0.05, 0.1}` | overlap; their text default is 0.1 and untested for us | Phase 1 clears |
| DB-8 | B=2 (prelude+core \| coda) | Table 8's best cell was B=2 | DB-2 ≥ DB-1 quality |
| DB-9 | equi-probability vs uniform σ partition | Table 7's 38.03 vs 43.53 — confirm it holds for text | Phase 1 clears |
| DB-10 | T-c fallback (core not independent, B=2 for TUL) | rescues DB-3 if the slot target fails | DB-3 fails on T-a |
| DB-11 | σ-blend contraction on the core carrier, **DB objective OFF** | assessment §5.3 — the `ρ ≤ 1` handle on its own, as a Task #276 cure | independent of Phase 1; cheap; keeps CE and PPL |
| DB-12 | block-visit distribution: uniform vs mass-proportional (1/8 : 6/8 : 1/8) | whether starving the prelude and coda to 1/8 of updates hurts; the one place this design departs from the paper's stated rule (§3.3) | DB-2 runs |
| DB-13 | inference step count: `T ~ Poisson(6)` vs fixed 4, 8, 16 | the trained denoiser is step-count agnostic (§3.3), so test-time depth becomes a free dial. Their AR setting used only **4** steps | DB-1 runs; no retraining needed |

**DB-11 is worth flagging separately.** It takes only the prescribed contraction
`h_k ← α h_{k-1} + (1−α) f(h_{k-1})` with a scheduled `α < 1`, keeps ordinary next-token CE, keeps
val PPL comparable to A0, and tests the one mechanism this paper shares with
`Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md`. It is not a DiffusionBlocks arm;
it is the idea extracted from it, and it lands on the existing ledger.

---

## 6. Open risks, ranked

| ID | Risk | Blocks | Status |
| --- | --- | --- | --- |
| R1 | clean\|noisy causal mask through CCA + CSA + HCA + XSA needs a new Triton mask path | every DB arm | unscoped — assume this is the largest single work item |
| R2 | 320-dim `x0` ctx slice too thin to be the core's only conditioning | DB-B3 | **downgraded** — Wolfe's call is that 320 of 1024 is wide enough. DB-5 becomes confirmatory, not a required fallback. See P3 |
| R3a | **Target scale must be pinned** so `σ_data` is real (§3.1 item 1). **Per-slice, not global** (reviewer): the euclidean slice inits near unit norm, the Lorentz TANGENT slice inits at std 0.005 (`embeddings.py`: `space_embed` std 0.005, log-map ≈ identity near origin) — roughly two orders of magnitude apart, so one global `σ_data = 0.5` drowns the Lorentz slice at every σ. Rule must be measured on the QUANTISED embedding output (int6 `embed_quant` is ON in the anchor configs) | every DB arm | design known, untested — the live blocker; plan A2 |
| R3b | `y + σ·ε` on a hyperboloid point would be off-manifold — but MORPH never exposes one. **Reviewer resolution (2026-08-19), from `morph/model/embeddings.py::LorentzEmbedding`:** the module log-maps the hyperboloid point to the origin tangent space INSIDE the embedding (`log_map_origin(project_to_hyperboloid(·))`), so the network — and the target `y` — lives in ambient Euclidean coordinates and the paper's recipe never touches hyperboloid coordinates. The "ambient Euclidean, Lorentz at readout" branch of the original question is the status quo, not a change | nothing, once R3a is settled | **downgraded from BLOCKING** — resolved by code reading; Wolfe signs off on keeping the status quo (O1) |
| R3c | Embedding *collapse* in the paper's own sense (all embeddings → one vector, Diffusion-LM) may still bite independently of scale | every DB arm | monitored by kill criterion 4 |
| R4 | slot denoising target (§3.4) | DB-3, DB-4 | pre-registered T-a, fallback T-c |
| R5 | step-counted schedule stretch by B | any B=3 arm on `base.yaml` | mitigations in §4.6, none implemented |
| R6 | bridge metrics need a resident teacher; must run post-hoc, not in-training | all quality rows | design decided, not built |
| R7 | 2× positions may push TUL arms back into OOM at batch 14 (A1 already needs 24.06 GB) | DB-3, DB-4 | first thing DB-3 will tell us |
| R8 | The authors set `‖y‖ = 1` (whole-vector L2) **and** `σ_data = 0.5`. Those are only consistent at `d = 4`; at `d = 1024` per-component std is 0.031, ~16× under `σ_data`. If `σ_data` overstates the scale, `c_skip` stays near 1 until σ≈0.5 while the signal dies by σ≈0.03 | fidelity of any port that copies them | our O1 rule scales to `σ_data·√dim` so `σ_data` is literally true — but we have NOT tested whether the discrepancy matters. CE tolerates it better than L2 would |
| R9 | **Sampler scale drift.** The denoised estimate is `softmax(logits) @ E`, a convex combination of unit-norm embeddings, so its norm shrinks toward 0 as the model is less certain — while training always sees a full-scale `y`. A train/inference mismatch inherent to the method | every DB arm's generation quality | inherited from the method, not ours to fix. Monitor `‖denoised‖` per Euler step during sampling |
## 7. Run log

Append one row per actual run. Empty until the GPU frees up.

| Date | Arm | Config | wandb run | steps | Outcome | Sheet cell filled |
| --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | nothing has run | — |
