# CLAUDE.md — MORPH

## Overview

**MORPH** — Orchestrates Recursive Pruned Hierarchies

Production model combining: Parcae-style looped transformer, MORTAR structured sparsity,
CCA+CSA+HCA attention, multi-rate residual (MRR) channels, STP (Semantic Tube Predictor)
geometric regularization, hybrid embeddings, STE ternary shadow weights.
Dual PyTorch (GPU) + JAX (TPU).

> Neural memory and LeJEPA split_nsm z-latent prediction were **removed** (neural
> memory deferred — stability at small context lengths still unresolved; JEPA dropped).
> The MRR "slow" channel (channel 2) that neural memory fed is retained as a reserved
> slow-rate residual channel.

## Architecture

Loop hierarchy:
1. **Inner**: Parcae core loop (4 prelude + 8 core × T + 4 coda)
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
    transformer.py   # Core looped transformer (Parcae + MRR)
    attention.py     # CCA+CSA+HCA+XSA+ResAttn+CoPE (one module, no flags)
    embeddings.py    # Hybrid (eucl+Lorentz) + bigram
    mhc.py           # Multi-Rate Residual (MRR) — per-channel γ scaling
    sparsity.py      # MortarLinear (dense pre-carve → MORTAR BCSR post-carve)
    routing.py       # ReMoE over macro tiles
    prediction.py    # STP (Semantic Tube Predictor) — zero-param geodesic regularizer
  kernels/
    triton/          # GPU kernels (fused ops, attention, decode)
    pallas/          # TPU kernels
  training/
    train.py         # Single entry point
    optimizer.py     # AdamW + STE ternary shadow weights
    data.py          # OpenWebText + StarCoder2 tokenizer
    pruning.py       # CMS schedule (dense → prune → compact → route)
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
