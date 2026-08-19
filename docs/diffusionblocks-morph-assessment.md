# DiffusionBlocks × MORPH — applicability assessment

**Paper:** DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation —
Shing, Koyama, Akiba (Sakana AI / U. Tokyo), ICLR 2026,
[arXiv:2506.14202](https://arxiv.org/abs/2506.14202) (v4, 2026-06-12).
Local archive: [`references/training-objectives/2506.14202.md`](references/training-objectives/2506.14202.md)
(+ PDF beside it). Curated entry: `references.md` §9.

**Status: BRAINSTORM. Nothing is built. No arm is registered in `ablation-ledger.md`, no config
key exists, no code path is touched.** This file records what the paper says, what transfers to
MORPH, what does not, and the cheapest experiment that would settle it. Read it before writing any
DiffusionBlocks code.

---

## 1. The paper has two separable modes. Only one of them is about looped models.

The abstract sells memory. That is the **block-wise mode**, and it is not the mode that matters
most to MORPH.

| | Block-wise mode (§3, main claim) | Recurrent-depth mode (§5.5, App. E.5) |
|---|---|---|
| What is partitioned | the `L` layers, into `B` blocks | **nothing** |
| Weight tying | none (blocks have separate params) | preserved (one map, applied `K` times) |
| What you win | `B`× memory on params **+** grads **+** optimizer state (App. G) | `K`× **training compute**; BPTT deleted |
| Training compute | **unchanged** — `L·K` layer evals either way (App. H) | ~10× less total (their Huginn number) |
| Applies to MORPH as-is | no (core is weight-tied) | closest fit, but needs a core rewrite |

App. H is explicit and easy to miss: for the block-wise mode, *"DiffusionBlocks requires the same
total amount of computation as standard training, while reducing memory usage by a factor of `B`."*
Measured on ViT: 0.0507 s/iter end-to-end vs 0.0181 × 3 = 0.0543 s/iter aggregated. It is a memory
method, plus embarrassing parallelism across blocks if you have spare GPUs. We have one 5090.

## 2. What the paper says about Huginn — and it is not "split the loop"

Wolfe's prior was to split the core loop in two for DiffusionBlocks. The paper does the opposite.
App. E.5, verbatim:

> *"Unlike other architectures, recurrent-depth models do not require block partitioning since the
> entire network is applied recurrently. Instead, we train the full network as a denoiser by
> sampling different noise levels σ at each training step."*

So for the architecture that is MORPH's closest published relative, **B = 1**. The `B`-way split and
the recurrence are alternative ways to cover the σ range: layers-in-series, or one map re-applied.
You do not need both, and using both means un-tying the core.

Their Huginn config is near-identical in shape to MORPH: 2 prelude / 4-layer recurrent block /
2 coda, d=512, 8 heads, Pythia-70M-like. MORPH local is 4 / 6 / 4 at d=1024. Same skeleton, ~2×
the width and ~1.5× the depth.

Result (Table 5, LM1B): MAUVE 0.49 → 0.70, gen-PPL(Llama-2) 17.04 → 16.08, gen-PPL(GPT2-XL)
46.73 → 42.43. Better on all three at ~1/10 the training compute.

**Read that number carefully.** It is not equal-token and not equal-metric: they trained 15 epochs
against Huginn's 5, they changed the objective, and they report MAUVE + teacher-model generative
perplexity because true perplexity is no longer computable (§4 below). The honest statement is
"at roughly a tenth the compute, and three times the epochs, the diffusion-trained model wins on
generation-quality proxies." It is a strong result. It is not "strictly better".

### Why the mapping to Huginn works, and where MORPH breaks it

The paper's §5.5 leans on one property: Huginn *"applies the same network multiple times, starting
from noise."* Huginn's recurrent state is initialised random, so its recurrence already reads as a
reverse-diffusion trajectory from `σ_max`.

MORPH's core does not start from noise. `transformer.py::_core_region`:

```python
e = self.input_norm(x)
h = e.clone()          # deterministic function of the prelude — not noise
```

To take the recurrent-depth mode literally, MORPH's core carrier would have to be initialised as a
**noised target embedding**, with the prelude output demoted to conditioning `x` (the role
`x0_core_terms` / `ChannelInject` already plays every iteration — that part maps cleanly). The core
would then carry a running estimate of the *clean target embedding*, not a running token
representation. That is a change to what the loop is for, not a training flag.

## 3. Answering "split the loop in 2" directly

Three readings, in increasing order of how much they break:

1. **Split the σ schedule, not the weights (what the paper does).** The loop's `T` iterations are
   already `T` Euler steps. One σ-conditioned, weight-tied core serves every sub-interval. `B = 1`.
   No new parameters. This is Huginn + DiffusionBlocks.
2. **Three blocks along MORPH's own seams: prelude / core / coda.** MORPH is *not* a pure loop —
   prelude(4) and coda(4) are untied; only core(6) is. So `B = 3` where the middle block is itself
   the recurrent denoiser applied `T` times is structurally honest and needs no un-tying. It also
   matches the paper's finding that a uniform layer split is best (Table 7: equi-probability +
   [4,4,4] = FID 38.03, the best cell). MORPH local is literally 4:6:4. This is the variant I would
   write down first.
3. **Two separately-parameterised cores, coarse-σ then fine-σ (Wolfe's original idea).** Not in the
   paper. Coherent — it is the equi-probability idea applied to a loop — and Table 8's `B = 2`
   beating `B = 1` on FID (9.90 vs 12.09) is weak support. But Table 8's blocks were plain layer
   partitions, *not* weight-tied loops, so that support does not transfer. Cost: 2× core parameters
   and half the weight tying, which is where MORPH's depth-scaling story lives (`references.md` §1:
   depth-via-looping beats params at equal FLOPs).

### On "larger layer count to accommodate diff blocks" — the arithmetic supports it, for mode (1)/(3) only

Table 8 (ImageNet, `L` = 24) degrades as layers-per-block falls:

| `B` | FID ↓ | `L/B` |
|---|---|---|
| 1 | 12.09 | 24 |
| **2** | **9.90** | 12 |
| **3** | **11.11** | 8 |
| 4 | 11.90 | 6 |
| 6 | 14.43 | 4 |

Quality holds down to ~8 layers/block and falls off below that. MORPH local `L` = 14 gives 7/block
at `B` = 2 — right at the edge. Cloud target 4:8:4 = 16 gives 5.3/block at `B` = 3, already past it.
To run `B` = 3 with 8 layers per block you need `L` ≈ 24. So yes, more layers is the right call —
but only if we un-tie the core. Under reading (2) the split follows MORPH's existing 4:6:4 seams and
no extra layers are needed.

Also from Table 7: **do not hand-tune the layer split.** Equi-probability σ partitioning beat
uniform σ partitioning at every layer distribution (38.03 vs 43.53 at [4,4,4]), and uniform *layers*
won inside equi-probability. Tune the noise partition; leave the layers even.

## 4. The costs, in the order they would actually hurt us

### 4.1 Perplexity stops being computable — this is the big one

App. E.4: *"Since DiffusionBlocks is not derived from ELBO-based objectives, computing traditional
perplexity is non-trivial."* They report MAUVE plus generative perplexity scored by Llama-2-7B and
GPT2-XL.

The AR adaptation does still use cross-entropy (App. B: *"We minimize cross-entropy loss instead of
L2 loss"*), but it is CE against a target reached from a **σ-noised embedding**. It is a
σ-conditioned reconstruction number, not a language-model likelihood. Every row in
`ablation-ledger.md` is val-CE / PPL. Adopting this on the AR path makes the entire historical
ledger incomparable and forces a new eval protocol (MAUVE needs a teacher model resident in VRAM
alongside training). On a single 5090 with a research programme that lives on CE deltas, losing the
comparison metric is the most expensive line item here, and it is not recoverable by cleverness.

### 4.2 Causal consistency doubles the sequence, and collides with TUL

App. E.4: to keep the AR property, noisy tokens must attend **clean** past, not noisy past. They
follow Block Diffusion and concatenate the clean and noisy sequences under a modified causal mask —
*"This approach doubles sequence memory."* The stated alternative (separate K/V for clean and noisy)
keeps memory but needs two forward passes.

For MORPH that means seq 4096 → 8192 with a **new custom mask** threaded through CCA + CSA + HCA +
XSA and their Triton kernels. That is real kernel work, not a config change.

It collides head-on with TUL, which already spends sequence budget on slot positions
(`L_total = seq_len + prefix_k · max_slots`, `tul_layout.py::pack_tul_row`). Stacking the clean|noisy
doubling on a slot-augmented layout is `2 · (seq + slots)`. Sequence length is the resource both
features want.

### 4.3 TUL's measured win partly evaporates under the recurrent-depth mode

`docs/tul-arms-result.md`: TUL is a **conditional-compute** win — 1.6× wall clock at slightly better
CE — because the core loops over slot positions only instead of all positions. The latent-memory
claim was falsified.

**Correction.** My first read of this was that TUL's win "largely evaporates" once the training
loop is a single forward. That was wrong, and the layer-pass accounting says why: **TUL's saving is
a position saving, not a loop saving.** The core runs on 64 slots instead of 1024 tokens, and that
holds whether the core is applied once or `T` times. Under DB-B3 the core term drops from
`6` passes/token to `6 × 64/1024 = 0.375`, taking `flop_proxy` from 11.0 to 2.88. TUL survives.

What does shrink is the **wall-clock multiplier**. TUL buys 1.76× on top of A0 (0.947 → 0.544 s/step)
because A0's step is 84 % real compute. At `flop_proxy` 2.88 the step is ~60 % fixed launch overhead,
so the same FLOP ratio converts into much less time. Pre-registered band: 1.3–1.8× for TUL on top of
DB, against 1.76× on top of A0. Sheet §3.4 and §4.2.

**The genuinely tricky part is the slot target, not the compute.** A TUL slot has no token label
(spec: slot label = first token of the next span; the slot's core state has no loss of its own), and
the DiffusionBlocks-AR target `y` is a token embedding. A core block that only sees slots therefore
has no `y` to denoise toward. Pre-registered choice: `y` = the next span's first token embedding, so
it stays a *prediction* target and does not become span autoencoding — which the TUL spec explicitly
forbids (LCM / CoCoMix / Block Transformer §4.2). If that fails, the fallback drops core
independence and collapses `B` = 3 → 2 for the TUL arms. Sheet §3.4, options T-a / T-b / T-c.

### 4.4 MORPH's residual is not `z + f(z)` — resolvable, but check it

The derivation (§2.2) needs the update to be an Euler step on one carrier. MORPH's residual is
`HyperConnectionResidual`: carrier `[B,S,n,C]` with `n=4`, and
`x_out = Hres·x_streams + Hpost·(y⊗1)` where `Hres` is Cayley-orthogonal. That is not `z + f(z)`.

This looks like a blocker and is not. Eq (5) is explicitly an **inter-block** connection, and App. C
says *"By modifying the inter-block connections to match the discretization scheme of other solvers,
any diffusion sampling method can be employed."* So HC stays **inside** a block and the σ-blend is
the **seam between** blocks. Nothing about the n=4 stream carrier has to change.

Worth noting the tension it creates, though: JPmHC was chosen precisely because Cayley gives exact
dynamical isometry — all singular values 1, i.e. **ρ = 1**, sitting exactly on the bifurcation
manifold that `Ai-notes/06-19-2026/.../MENTAL-MODEL.md` identifies as the cliff. Isometry inside a
block plus a prescribed `α < 1` contraction at the seam is a coherent and arguably better-conditioned
combination than isometry alone. See §6.

### 4.5 Smaller, but each one is a day

- **AdaLN σ-conditioning on every block** (Step 3). New params on every norm and a new forward
  input. Compatible with the "no runtime flags, bake it in at init" rule, but it touches the fused
  norm kernels.
- **Embedding L2-normalisation is mandatory**, not optional — App. C cites embedding collapse in
  continuous relaxations of discrete targets. MORPH's embeddings are hybrid
  Euclidean + Lorentz (`lorentz_fraction = 0.25`). "L2-normalise the embedding" is not well defined
  on a Lorentz component. Unresolved.
- **EDM preconditioning + weighting** `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, `σ_data = 0.5`. App. C:
  the weighting is *crucial* for equi-probability partitioning to work. Not optional.
- **Overlap ratio** `γ`: 0.05 default, **0.1 for text**. Cheap, but do not leave it at 0.
- **Sign check before implementing.** As rendered in v4, Eq (3)–(5) give
  `z_b = z_{b-1} + (Δσ_b/σ_{b-1})(z_{b-1} − D)` with `Δσ_b := σ_{b-1} − σ_b > 0`. That is
  `α = 1 + Δσ/σ_{b-1} > 1`, `β < 0` — an *extrapolation away* from the denoiser, and substituting
  `z_{b-1} = y + σ_{b-1}ε`, `D = y` lands at noise level `2σ_{b-1} − σ_b`, i.e. noise **increases**
  as you walk down the schedule. EDM (Karras et al. 2022, Alg. 2) has the opposite sign:
  `x_{i+1} = x_i + (σ_{i+1} − σ_i)(x_i − D)/σ_i`, which with `σ_{i+1} < σ_i` gives
  `z_b = α z_{b-1} + (1−α) D`, `α = σ_b/σ_{b-1} ∈ (0,1)`, landing correctly at `y + σ_b ε`.
  Take the EDM form; treat the paper's rendered sign as a typo and confirm against the authors'
  released code before writing a line.

## 5. The optimizer question — does this depend on AdamW?

**No.** The paper uses AdamW everywhere (lr 5e-4 ViT, 1e-4 CIFAR DiT, 5e-5 ImageNet, 3e-4 text;
cosine + warmup) but nothing in the derivation touches the update rule. DiffusionBlocks changes the
**objective and the gradient graph**, not the optimizer. App. G invokes Adam only to put a number on
the memory ratio (2P of state → 4P per layer for params+grads+state); swap the optimizer and only
that constant moves.

So AdEMAMix does not conflict with it. But there are four real interactions, and two of them are
bugs waiting to happen.

### 5.1 Synergy: the block-wise mode refunds the AdEMAMix memory tax

AdEMAMix's cost is an extra slow-EMA buffer. Gradient checkpointing does nothing for optimizer
state; DiffusionBlocks cuts it by `B` (App. G is explicit that this is the distinction). `B` = 2–3
would pay back the slow-EMA buffer outright. With β1 = 0 (the b1zero arm drops the fast EMA) the
state is already ≈ Adam-sized, so this compounds rather than merely breaking even.

### 5.2 The `ρ^T` detonation mechanism disappears from training — and reappears at inference

`MENTAL-MODEL.md` argues the β1 = 0 AdEMAMix detonations (Task #276) come from the optimizer being
blind to `ρ(J_core)`: it carries stale α-amplified momentum across the `ρ = 1` manifold and the loop
turns one bad step into `ρ^T`.

Under the recurrent-depth mode the training step is a **single forward pass**. The gradient no
longer integrates over an inner trajectory, so there is no `ρ^T` factor in it at all. The optimizer's
blind spot stops being a *training-stability* problem. That is a direct hit on the root cause, and
it is the most interesting thing in this paper for MORPH.

The catch: the loop still runs `T` times **at inference**. A non-contractive learned map now
diverges silently, in generation, with nothing in the loss to show it. We would trade a loud
detonation for a quiet one. If we go this way, the `ρ(J_core)` probe becomes an **inference gate**,
not an optional diagnostic.

### 5.3 The paper hands us a `ρ` control knob for free

With the corrected sign (§4.5) the inter-block update is a convex blend:

```
z_b = α z_{b-1} + (1 − α) D_θ(...),     α = σ_b / σ_{b-1} ∈ (0, 1)
J   = α I + (1 − α) ∂D/∂z    ⇒   ρ(J) ≤ 1  whenever ‖∂D/∂z‖ ≤ 1
```

The contraction is **prescribed by the noise schedule** instead of being left to the optimizer to
discover. That is precisely the prescription `MENTAL-MODEL.md` arrives at from the other direction —
*"target contractivity (`ρ ≤ 1`: spectral/Lipschitz control, direction-preserving carrier renorm),
not symptom-clamping"* — and it is a stronger version of what Parcae's negative-diagonal injection
buys us today (`references.md` §1). Two independent lines of reasoning landing on the same
mechanism is the best evidence in this document that the idea is not just fashionable.

Note this knob is available **without** adopting the objective: a σ-like scheduled convex blend on
the core carrier is a candidate contractivity fix on its own.

### 5.4 Bug trap: every step-counted schedule in MORPH silently stretches by `B`

This applies to the block-wise mode only, and it will bite.

If a block is sampled once per `B` iterations, then per-block **everything step-counted runs `B`×
slower in effective time**:

- `ademamix_t_alpha = 8000` and `ademamix_t_beta3` are α / β3 warmup horizons in **steps**. Per
  block they become 8000/`B` effective updates. The base.yaml comment already warns that a large β3
  active from step 0 diverges even with LR warmup — this makes the warmup shorter, in the dangerous
  direction. Same for `ademamix_beta3_warmup_start = 0.9`. **Divide these by `B`, or the slow EMA is
  effectively frozen relative to the schedule you think you set.**
- **MORTAR is worse.** `accumulate_scores()` runs between `backward()` and `zero_grad()`. A
  `MortarLinear` in an unsampled block gets no gradient, so Taylor saliency accumulates `B`× slower
  and the density schedule (`prune_start = 3000`, `prune_interval = 167`, target 0.25 by ~27050)
  stretches accordingly. `carve()` at `compact_step = 29000` would then run on a still-dense model —
  exactly the `K/C = 1.0` gotcha in `CLAUDE.md`. Any DiffusionBlocks arm **must** either scale the
  prune cadence by `B` or assert on the logged `[prune] density=…` before carve.
- Whole-body ReMoE at `route_start = 30000` inherits the same problem.

Nobody in the paper hits this because they train from scratch with no sparsity schedule and no
long-horizon optimizer warmup. It is entirely ours.

## 6. Implementation order

We are building this. The order below is chosen so each arm answers one question, and so the
cheapest arm is also the one the paper already validated on a looped model.

Full arm definitions, pre-registered expected numbers, metric contract, kill criteria and the
schedule-stretch mitigations live in
**[`diffusionblocks-experiment-sheet.md`](diffusionblocks-experiment-sheet.md)**. That sheet is the
tracking document; this file is the design rationale behind it.

1. **Phase 0 — instrumentation, no GPU.** We cannot judge any of this without FLOP efficiency, and
   `perf/mfu` does not exist yet. Add `layer_passes_per_token` for every arm (not TUL only),
   `positions_per_token`, `flop_proxy`, `model_tflops`, `mfu`. Measure the bf16 GEMM ceiling on this
   5090 rather than quoting a spec sheet. Settle the Eq (3)–(5) sign (§4.5) against the authors'
   code. Re-confirm the A0 shape facts and the step-0 parity marker.
2. **DB-B1 first, not B=3.** `B = 1`, the whole 4:6:4 net as one denoiser — the paper's literal
   Huginn setting. It isolates *"does the diffusion objective work on MORPH at all"* without also
   testing block independence, and it inherits none of the §5.4 schedule-stretch hazards.
3. **DB-B3 second** — prelude / core / coda, splitting on seams MORPH already has, so the core stays
   weight-tied and no layers are added. The σ range is cut into `T + 2 = 8` equi-probability Euler
   steps: step 1 = prelude, steps 2–7 = the core applied once each, step 8 = coda. Expected training
   cost `E[passes/token] = 5.5` against A0's 44.
4. **The TUL cross on both**, per Wolfe. See §4.3 below for the correction to what I said about it,
   and the sheet §3.4 for the slot-target fork, which is the genuinely tricky part.
5. **Then the batch sweep.** The memory win is probably worth more than the FLOP win — see the
   launch-bound cost model in the sheet §2.1. A1 already OOMs at batch 16, and batch is the only
   lever on a 0.156 s fixed step floor.

Separately, and cheaply: **the σ-blend contraction is extractable from the objective.** A scheduled
`h_k ← α h_{k-1} + (1−α) f(h_{k-1})` with `α < 1` keeps ordinary next-token CE, keeps val PPL
comparable to A0, and tests the one mechanism this paper shares with our own nested-dynamics note
(§5.3). Sheet arm DB-11. It is not a DiffusionBlocks arm; it is the idea taken out of it.

## 7. Open questions, named

1. Does MORPH's core carrier actually contract? Unmeasured — the `ρ(J_core)` probe from
   `Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md` §5, plus
   `‖h_k − h_{k-1}‖` across the loop. Worth running alongside sheet arm DB-11.
2. Can a TUL slot carry a denoising target at all, given it has no token label?
3. What does "L2-normalise the embeddings to prevent collapse" mean for the Lorentz component?
4. Is the Eq (3)–(5) sign a rendering typo or in the released code too? (Check the authors' repo.)
5. Does the clean|noisy concatenated mask survive CCA + CSA + HCA + XSA, or does it need a new
   Triton mask path? (Assume the latter.)
6. Their Huginn gain came with 3× the epochs. What is it at equal tokens?
