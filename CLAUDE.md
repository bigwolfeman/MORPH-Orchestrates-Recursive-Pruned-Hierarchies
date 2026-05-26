# CLAUDE.md — MORPH

## Overview

**MORPH** — Orchestrates Recursive Pruned Hierarchies

Production model combining: Parcae-style looped transformer, Block-ELL structured sparsity,
CCA+CSA+HCA attention, neural memory with SSM outer loop, multi-rate residual (MRR) channels,
STP geodesic regularization, LeJEPA split_nsm z-latent prediction, hybrid embeddings,
STE ternary shadow weights. Dual PyTorch (GPU) + JAX (TPU).

## Architecture

Three-loop hierarchy:
1. **Inner**: Parcae core loop (4 prelude + 8 core × T + 4 coda)
2. **Middle**: Neural memory SSM (gradient-based surprise update on forward pass)
3. **Outer**: Zyphra RSA harness (inference-time, requires RL — deferred)

Attention: CCA channel compression → CSA sparse global + HCA dense compressed (alternating layers),
with XSA, Residual Attention, CoPE Clipped RoPE, QK-Norm baked in.

Embeddings: Hybrid (euclidean + Lorentz hyperbolic) + bigram (30-60k hash vocab).

Memory prediction: split_nsm — backbone predicts mean(next segment z_coda),
memory predicts next segment prelude state. STP enforces geodesic smoothness.

## Design Principles

- **No runtime feature flags.** Features are baked in at init. No `if use_feature:` in forward pass.
  torch.compile sees a clean graph with no branching.
- **Dual framework.** Every module has PyTorch + JAX implementations. Checkpoints convert between them.
- **Hydra configs.** All hyperparams in YAML. Every run reproducible from its wandb config.
- **Custom kernels.** Triton (GPU), Pallas (TPU), CUDA (neural memory). SM120 tuned for 5090.

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
    memory.py        # Neural memory v3 + SSM inject + z-latent (split_nsm)
    mhc.py           # Multi-Rate Residual (MRR) — per-channel γ scaling
    sparsity.py      # Block-ELL + CMS + compaction
    routing.py       # ReMoE over macro tiles
    prediction.py    # STP + LeJEPA split_nsm
  kernels/
    triton/          # GPU kernels (Block-ELL, fused ops, attention, memory)
    cuda/            # CUDA kernels (neural memory backward)
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
    checkpoint.py    # PT ↔ JAX converter
configs/             # Hydra YAML
tests/
docs/
```

## Critical Patterns

### Memory Forward-Pass Update
Neural memory updates weights on forward via gradient-based surprise.
Z-latent targets must be forward-looking (memory can't trivially solve backward-looking targets).
split_nsm: backbone→mean(next_seg z_coda), memory→next_seg prelude state.

### Block-ELL Sparsity Schedule
Dense 0-50k → prune 50-60k (5%/3k steps) → compact → route 60-75k.
accumulate_scores() MUST be called between loss.backward() and optimizer.zero_grad().

### STP During Pretraining
Paper tested fine-tuning only. We use STP during pretraining as a geometric regularizer.
Doesn't improve PPL (teacher-forced metric can't see it). Improves generation quality.
Full-sequence multi-scale geodesic (strides 1,2,4,...,tau), tau=64.

### torch.compile
mode="default" (NOT reduce-overhead — CUDA graphs cause eval OOM).
No fullgraph=True (neural memory uses autograd.grad with retain_graph).
torch._functorch.config.donated_buffer = False.

### Project Cleanliness
Do not litter scripts around the directory. Keep a ignored/ folder for temporary scripts (keep it organized), this folder is set to gitignore.