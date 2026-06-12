# CLAUDE.md — MORPH

## Overview

**MORPH** — Orchestrates Recursive Pruned Hierarchies

Production model combining: Parcae-style looped transformer, MORTAR structured sparsity,
CCA+CSA+HCA attention, Cayley HyperConnection residual (n=4 streams), STP (Semantic Tube
Predictor) geometric regularization, hybrid embeddings, STE ternary shadow weights.
Dual PyTorch (GPU) + JAX (TPU).

> Neural memory (Titans) and LeJEPA split_nsm z-latent prediction were **removed**
> (neural memory deferred — stability at small context lengths still unresolved; JEPA
> dropped). MRR (multi-rate residual) is **dead under HyperConnections** (`mhc.py`
> `MultiRateResidual` is never instantiated when `residual_mode: hc_cayley`, the default) —
> the residual stream is the Cayley HC n=4 mixer, not MRR γ-channels.

> **Current state (2026-06-11):** this repo is the unified hotpath. The ablation winners
> (MORTAR-only sparse + always-on whole-body ReMoE + full quant stack: ternary backbone +
> int6 embeddings + 8-bit AdamW; HC Cayley n=4; GLA retention default-on) were migrated in
> as the default+only path, and the legacy 16×16 Block-ELL backend was ripped end to end.
> `configs/base.yaml` reproduces the validated mortar winner recipe (flat LR, no warmup,
> prune 3000/100/0.005 → carve@20000 → whole-body route@21000, taylor saliency).

## Architecture

Loop hierarchy:
1. **Inner**: Parcae core loop — local **3 prelude + 6 core × T + 3 coda** (cloud target 4:8:4,
   d=2048). T = per-sequence Poisson depth (mean 6, max 8), truncated BPTT depth 4, gradient
   checkpointed. d_model=768, d_ff=2048, seq 4096 locally.
2. **Outer**: Zyphra RSA harness (inference-time, requires RL — deferred)

Attention: CCA channel compression → CSA sparse global + HCA dense compressed (alternating layers),
with XSA, Residual Attention, CoPE Clipped RoPE, QK-Norm baked in.

Embeddings: Hybrid (euclidean + Lorentz hyperbolic) + bigram (30-60k hash vocab).

STP (Semantic Tube Predictor): full-sequence multi-scale geodesic smoothness on hidden
states. Geometric regularizer applied during pretraining (see Critical Patterns).

## Design Principles

- **No runtime feature flags.** Features are baked in at init. No `if use_feature:` in forward pass.
  torch.compile sees a clean graph with no branching.
- **Dual framework.** Every module has PyTorch + JAX implementations. Checkpoints convert between them.
- **Hydra configs.** All hyperparams in YAML. Every run reproducible from its wandb config.
- **Custom kernels.** Triton (GPU), Pallas (TPU). SM120 tuned for 5090.

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"

# Train (PyTorch, GPU)
python morph/training/train.py --config configs/base.yaml

# Train (JAX, TPU)
python morph/jax/train.py --config configs/tpu.yaml

# Tests
pytest tests/

# Format
black morph/ --line-length 100
ruff check morph/
```

## Project Structure

```
morph/
  model/
    transformer.py   # Core looped transformer (Parcae loop; _SwiGLUMortar hosts the ReMoE router)
    attention.py     # CCA+CSA+HCA+XSA+ResAttn+CoPE (one module, no flags)
    embeddings.py    # Hybrid (eucl+Lorentz) + bigram
    mhc.py           # HyperConnections (Cayley n=4, fused kernel). MRR class is DEAD under HC.
    sparsity.py      # MortarLinear (dense pre-carve → MORTAR 128×128 BCSR post-carve)
    routing.py       # TileRouter — whole-body ReMoE over the d_ff hidden-neuron bank
    prediction.py    # STP (Semantic Tube Predictor) — zero-param geodesic regularizer
  kernels/
    triton/          # GPU kernels (fused ops, attention, decode)
    pallas/          # TPU kernels
  training/
    train.py         # Single entry point
    optimizer.py     # AdamW + STE ternary shadow weights
    data.py          # OpenWebText + StarCoder2 tokenizer
    pruning.py       # CMS schedule (dense → prune → carve → route); density helper logs skips
  jax/               # JAX/Flax mirror
    model/
    kernels/
  interop/
    checkpoint.py    # PT ↔ JAX converter (name-driven)
configs/             # Hydra YAML
tests/
docs/
```

## Critical Patterns

### MORTAR Sparsity Schedule
Dense → prune (prune_step_blocks, 128×128-aligned) → carve() to MORTAR BCSR at
compact_step → ReMoE route at route_start. MORTAR is the ONLY sparse backend
(legacy 16×16 Block-ELL removed 2026-06-11; there is no sparse_backend knob).
accumulate_scores() MUST be called between loss.backward() and optimizer.zero_grad().
**GOTCHA: if `prune_start` is disabled (e.g. 9999999), no pruning happens and carve()
then runs on a still-dense model → K/C=1.0. Always confirm `[prune] density=…` actually
fell before trusting a "0.25 sparse" claim.** The validated winner cadence
(`base.yaml`) reaches target_density 0.25 by ~step 18000, before carve@20000.

### STP During Pretraining
Paper tested fine-tuning only. We use STP during pretraining as a geometric regularizer.
Doesn't improve PPL (teacher-forced metric can't see it). Improves generation quality.
Full-sequence multi-scale geodesic (strides 1,2,4,...,tau), tau=64. Zero parameters.

### torch.compile
mode="default" (NOT reduce-overhead — CUDA graphs cause eval OOM).
No fullgraph=True (the looped core uses gradient checkpointing, use_reentrant=False).
torch._functorch.config.donated_buffer = False (import the submodule explicitly first).

### Project Cleanliness
Do not litter scripts around the directory. Keep a ignored/ folder for temporary scripts (keep it organized), this folder is set to gitignore.
