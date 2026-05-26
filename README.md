# MORPH

**MORPH Orchestrates Recursive Pruned Hierarchies**

A memory-augmented looped transformer that optimzer per parameter capability with depth, then pruning what's left. MORPH reuses a small set of layers many times (Parcae looping), stabilizes that reuse with multi-channel residual dynamics, augments the backbone with a gradient-based neural memory, and prunes the looped layers to extreme sparsity using learned topology — all in a single training run.

The result: a model that matches dense transformers with far fewer parameters and FLOPs, while maintaining a long-term memory that persists across the sequence.

---

## Architecture Overview

<p align="center">
  <img src="docs/figures/morph_overview.png" width="700" alt="MORPH Architecture Overview"/>
</p>

Data flows top-to-bottom through five stages:

1. **Hybrid Embedding** — Input tokens are embedded via a combination of Euclidean, Lorentz, and bigram embeddings. The Lorentz component provides hyperbolic geometry for hierarchical token relationships.

2. **MAC Token Append** — N memory-augmented context (MAC) tokens are appended as a suffix to the sequence. These are persistent tokens that participate in attention at every layer, giving the neural memory a direct channel into the attention mechanism. They sit at the end of the sequence so they never invalidate the KV cache during inference. This also helps stabilize neural memory dynamics. Prepending memory tokens causes heavy weighting in the attention heads that is less prevelant in appending.

The Neural memory is updated during forward pass based on surprisal and its z latent.

3. **Prelude** (N layers) — Standard transformer blocks that establish initial representations. The neural memory injects retrieved context here via MAG (memory-augmented gating), a learned residual that blends memory content into the hidden states.

4. **Core Loop** (2N layers, iterated T times) — The heart of the architecture. The same set of layers is reused T times with diagonal injection at the loop boundary (Parcae-style, spectral radius < 1 guaranteed). Depth T is sampled from a Poisson distribution during training and truncated BPTT limits the backward pass to the last 4 iterations. The core MLP layers use Block-ELL sparse format for extreme pruning.

5. **Coda** (N layers) — Post-loop layers that refine the representation. After the coda, MAC tokens are stripped and the output goes through the LM head.

The memory system sits alongside the backbone as a parallel component — it reads from the backbone (observing hidden states) and writes back into it (injecting retrieved context). The STP loss enforces smooth token predictions, and the z-latent objective trains the memory to predict the future. This is a form of SSM, maintaining a long running state trained from the current context window and its latent predictions about the future.

---

## The MORPH Block

<p align="center">
  <img src="docs/figures/morph_block.png" width="600" alt="MORPH Block"/>
</p>

Every block in MORPH uses a **Multi-Rate Residual (MRR)** — a 3-channel residual stream where each channel updates at a different rate:

| Channel | Width | &gamma; | Role |
|---------|-------|---------|------|
| Compute | 3N | ~1.0 | Full-strength updates. Does the heavy lifting. |
| Context | 2N | ~0.5 | Half-strength updates. Carries positional and contextual state. |
| Memory | N | ~0.1 | Slow updates. Preserves information across many layers and loop iterations. |

**The key insight**: sublayers (attention, MLP) see the full d<sub>model</sub>-dimensional input — all channels concatenated. The channel separation happens *only* on the residual update side, via learned per-channel &gamma; gains. This means the compute doesn't lose any information from the slow channels; it just updates them gently.

**Why this matters for looping**: When the same layers execute T times, a standard residual connection would amplify signals by T&times;. The MRR memory channel (&gamma;&approx;0.1) acts as a built-in damper — information in this channel persists across loop iterations without exploding. The compute channel (&gamma;&approx;1.0) does fresh work each iteration. This is what makes deep looping stable.

### Block internals

Each block has two sublayers with identical structure:

1. **RMSNorm** &rarr; **Attention** (CCA + CSA/HCA) &rarr; **MRR**
2. **RMSNorm** &rarr; **SwiGLU MLP** (Block-ELL in core layers) &rarr; **MRR**

The attention mechanism combines:
- **CCA** (Cross-Chunk Attention): The primary attention path with channel compression and CoPE-RoPE positional encoding
- **CSA** (Chunked Sparse Attention): Used on even-numbered layers, sparse windowed attention with compression
- **HCA** (Hyper-Connection Aware attention): Used on odd-numbered layers, dense attention that respects the MRR channel structure
- A **sigmoid gate** blends the CSA/HCA outputs

---

## Neural Memory & Z-Latent

<p align="center">
  <img src="docs/figures/morph_memory.png" width="700" alt="Neural Memory & Z-Latent"/>
</p>

MORPH's neural memory is an MLP whose **weights are the memory state**. Unlike a key-value store or embedding table, the memory is a function approximator that learns associations through gradient descent — the same mechanism the outer training loop uses, but applied as an inner loop within the forward pass.

### The read/write cycle

**Retrieve & Inject** (memory &rarr; backbone):
The memory receives queries from the backbone and returns context vectors. These are injected via two paths:
- **MAG** (Memory-Augmented Gate): A learned gated residual that blends retrieved memory into the prelude hidden states
- **MAC** (Memory-Augmented Context): Suffix tokens appended to the sequence that attend to (and are attended by) all other tokens

**Observe & Store** (backbone &rarr; memory):
After the backbone processes a chunk of tokens, the hidden states are projected into key-value pairs:
1. k = W<sub>k</sub>(h) — the "address" (what to index by)
2. v = W<sub>v</sub>(h) — the "content" (what to store)
3. The memory queries itself: M(k) = "what do I already know at this address?"
4. **Surprise** = ||M(k) - v||&sup2; — high if v is new information, low if already stored
5. Gradient descent on M's weights: surprised &rarr; large update, unsurprised &rarr; small update

The update uses momentum with learned gates:
- **&alpha;** (decay): How much of the old memory to retain
- **&eta;** (momentum): How much of the previous gradient direction to keep
- **&theta;** (learning rate): How large each update step is

All three gates are `sigmoid(Linear(chunk_pooled))` — learned per-chunk from pooled hidden states. This means the memory dynamically adjusts its learning behavior based on what it's seeing.

### Z-Latent: Teaching memory what to remember

Without the z-latent objective, the memory has no direct gradient signal for *what* to store — it would just memorize everything equally. Z-latent provides a targeted learning signal: **predict the future**.

- **Backbone head**: Given coda representations from window t, predict what window t+1's coda will look like
- **Memory head**: Given memory state at window t, predict what window t+1's prelude state will be

This is computed in a **single forward pass** — all windows are processed at once and predictions are sliced from the representation at segment boundaries. No repeated computation.

The effect: memory learns to prioritize storing information that will be *useful later*, not just information that is *surprising now*.

---

## How the Pieces Synergize

None of MORPH's components work in isolation. The architecture is designed so each piece reinforces the others:

### Looping &times; MRR = Stable depth without parameter growth

The core loop reuses 2N layers T times, giving effective depth of 2N&middot;T with only 2N layers of parameters. But naive looping is unstable — residual magnitudes grow with each iteration.

MRR solves this: the memory channel (&gamma;&approx;0.1) provides a slow-moving state that persists across iterations without amplification, while the compute channel (&gamma;&approx;1.0) does fresh work each time. The diagonal injection at loop boundaries (spectral radius < 1) provides additional stability guarantees. Together, these allow T=6-8 loop iterations without divergence.

### Looping &times; Block-ELL = Multiplicative FLOP savings

Because the same weights are reused T times per forward pass, pruning the core block to 25% density doesn't just save 75% of MLP FLOPs — it saves 75% &times; T. For T=6, that's 4.5&times; the absolute FLOP reduction compared to pruning a non-looped layer.

The pruning uses CMS (Continuum Memory System) topology scoring: gradient-based importance scores accumulated over training, with periodic topology decisions that prune low-scoring blocks and optionally regrow high-potential ones.

### Memory &times; Z-Latent = Learns what matters

The neural memory can store anything, but storage capacity is finite (the MLP has limited parameters). Z-latent provides the optimization pressure: store information that helps predict the future. Without z-latent, the memory would allocate capacity uniformly across all inputs. With it, the memory learns to prioritize information that has predictive value — rare events, contextual shifts, and dependencies that span beyond the attention window.

### Memory &times; MAC Append = KV-cache-friendly injection

MAC tokens are appended as a suffix (not prepended). This is critical for inference: when the memory updates and produces new MAC token values, only the last N positions in the KV cache need updating. Prepending would shift all position indices, invalidating the entire cache.

The suffix mask ensures MAC tokens are always-visible: every query can attend to MAC tokens, and MAC tokens can attend to everything. This gives the memory bidirectional attention context without the cache penalty.

### Block-ELL &times; Triton = Actual speedup, not just parameter reduction

Structured sparsity on paper doesn't help if the runtime can't exploit it. MORPH includes custom Triton kernels for Block-ELL forward and backward passes that operate directly on the sparse tensor format. After compaction, the MLP layers execute only the surviving blocks — no wasted computation on zero blocks, no gather/scatter overhead.

---

## Training Pipeline

MORPH trains in a multi-phase pipeline within a single run:

```
  Dense         Prune              Compact   Settle    Route
  warmup        to 25% density     ┃         ┃         (ReMoE)
  ┃             ┃                  ┃         ┃         ┃
  0 ──── 6k ──── 6k ──────── 66k ── 66k ── ~70k ── 70k ── end
```

### Phase 1: Dense Warmup (~6k steps)
All blocks active. The model learns basic language representations with full-density MLPs. CMS gradient scores are accumulated every step to build importance estimates.

### Phase 2: Prune (~60k steps)
Starting at step ~6k, topology decisions happen every `prune_interval` steps: low-scoring blocks are removed, reducing density toward the 25% target. The model continues training through each pruning step, adapting its weights to compensate for removed capacity. This is gradual — removing 5% of remaining blocks per round over many rounds.

### Phase 3: Compact
At the target density, the sparse mask is converted into actual smaller Block-ELL tensors. Memory footprint drops immediately. From this point, forward/backward passes use the compact Triton kernels.

### Phase 4: Settle
Post-compact recovery training. The model adjusts to the now-permanent sparse structure. Learning rate may be reduced. This phase typically runs for a few thousand steps.

### Phase 5: Route (ReMoE) *[pending integration]*
After the model has settled into its pruned structure, per-token routing is activated over the surviving tile-groups. Different tokens activate different subsets of the remaining 25% of blocks, effectively giving each token a specialized sparse MLP.

The routing module (`morph/model/routing.py`) implements `TileRouter` with PEER-style product keys and `RoutedBlockELLLinear` for the post-compact forward pass. The Triton kernel (`morph/kernels/triton/block_ell_routed_forward.py`) is ready. **Integration into the training loop is pending.**

---

## Project Structure

```
morph/
  model/
    transformer.py      # MORPHTransformer — the full model
    attention.py         # CCA + CSA + HCA attention implementations
    memory.py            # Neural memory: update, retrieve, MAG/MAC inject
    mhc.py               # Multi-Rate Residual (MRR) channel dynamics
    embeddings.py        # Hybrid Euclidean + Lorentz + bigram embeddings
    prediction.py        # Z-latent and STP prediction heads
    sparsity.py          # BlockELLLinear — dense pre-compact, sparse post-compact
    routing.py           # TileRouter + RoutedBlockELLLinear (post-compact)
    titans_core/         # Vendored CMS block-sparse and topology scoring
  kernels/
    triton/
      fused_window_attention.py    # Fused windowed attention with MAC suffix mask
      block_ell_forward.py         # Block-ELL sparse forward kernel
      block_ell_backward.py        # Block-ELL sparse backward kernel
      block_ell_routed_forward.py  # Routed sparse forward (per-token tile selection)
      memory_mlp.py                # Fused memory MLP kernel
      fused_gate_combine.py        # Fused gate+combine for MAG
      fused_ops.py                 # Fused auxiliary operations
  training/
    train.py             # Training loop with Hydra config
    data.py              # OpenWebText data loading
    optimizer.py         # DeepNestedOptimizer integration
    pruning.py           # CMS 3-phase pruning schedule
  configs/
    base.yaml            # Default config (d=768, 3:6:3 architecture)
    cloud.yaml           # Cloud/multi-GPU config
docs/
  figures/               # TikZ source + rendered PNGs for all diagrams
  references.md          # Paper citations
```

---

## Quick Start

```bash
# Install
pip install -e ".[train]"

# Train (local, single GPU)
python -m morph.training.train

# Train with overrides
python -m morph.training.train \
  model.d_model=512 \
  training.steps=30000 \
  training.batch_size=8

# Train on cloud (multi-GPU)
python -m morph.training.train --config-name cloud
```

Training logs to [Weights & Biases](https://wandb.ai). The full config is logged to every run for reproducibility.

---

## Default Configuration

From `morph/configs/base.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 768 | 6N where N=128 (3 MRR channels: 384+256+128) |
| Layers | 3 + 6 + 3 | Prelude + Core (looped) + Coda |
| Mean loop depth | 6 | Poisson(&lambda;=6), max 8 |
| Attention | CCA+CSA/HCA | 12 heads, 4 KV heads, window=128 |
| Vocab | 49,152 | StarCoder2 tokenizer |
| Max seq len | 4,096 | Windowed attention keeps memory O(T&middot;w) |

---

## Key Results

| Model | Params | Density | Val PPL | Step/s | Notes |
|-------|--------|---------|---------|--------|-------|
| Dense baseline (34L) | 531M | 100% | 38.4 | 2.0 | Standard transformer |
| Looped (3+6+3, T=6) | 81M | 100% | 35.2 | 3.1 | 6.5&times; fewer params |
| Looped + pruned | 81M&rarr;~20M active | 25% | 37.1 | 4.8 | 56% faster post-compact |
| Looped + pruned + routed | — | — | — | — | *Pending* |

The looped architecture at 81M parameters beats the 531M dense baseline. Pruning to 25% density adds ~5% PPL cost while nearly doubling throughput.

---

## References

- **Titans** (arXiv:2501.00663) — Neural memory with gradient-based surprise updates
- **Parcae** — Stable looped transformers with diagonal injection
- **Multi-Rate Residual** (MRR) — Per-channel residual scaling for loop stability (inspired by but distinct from Hyper-Connections)
- **CMS** — Continuum Memory System for block-sparse topology
- **ReMoE** — Dynamic per-token expert routing
- **PEER** — Product-key retrieval for efficient routing

See [`docs/references.md`](docs/references.md) for full citations.

---

## License

Research code. See repository for terms.
