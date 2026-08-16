# CLAUDE.md — MORPH

## Overview

**MORPH** — Orchestrates Recursive Pruned Hierarchies

Production model combining: Parcae-style looped transformer, MORTAR structured sparsity,
CCA+CSA+HCA attention, Cayley HyperConnection residual (n=4 streams), GLA retention branch,
hybrid embeddings, STE ternary shadow weights. PyTorch-first; a JAX/Flax model mirror
exists under `morph/jax/` but lags the PyTorch path (see gotcha below).

> **Naming gotcha:** the residual attributes on `MORPHBlock` are called `mrr_attn` /
> `mrr_mlp` for checkpoint compatibility, but they hold `HyperConnectionResidual`
> (Cayley n=4) modules — there is no `MultiRateResidual` class in the PyTorch tree.
> The JAX mirror (`morph/jax/model/`) still implements the old MRR residual and has
> not been ported to HC-Cayley; do not assume PT/JAX parity.

> **Source of truth for the training recipe is `morph/configs/base.yaml`** (heavily
> commented). Current schedule: flat LR 1e-4 (warmup=0, min_lr==lr), taylor saliency,
> prune_start=3000 / prune_interval=167 (density hits 0.25 by ~step 27050) →
> carve at compact_step=29000 → whole-body ReMoE at route_start=30000, all inside a
> 100k-step run with TST superposition for the first 30k steps. Do not restate these
> numbers elsewhere — read the YAML.

## ⭐ Core mental model — MORPH is a NESTED dynamical system (read before optimizing)

The looped core makes MORPH **two** dynamical systems, not one: the **outer** (optimization,
`θ_{t+1}=θ_t−η·u_t` — what every optimizer models) and the **inner** (the forward itself,
`h_{k+1}=f_θ(h_k)` over T loop iterations — exists only because of the loop). The optimizer sees only
`∇_θ L`, which integrates over the inner trajectory and discards its structure → it is **blind to the
inner map's contractivity `ρ(J_core)`**. Consequence: the loss landscape **inherits the bifurcation
structure of `f_θ`** — a `ρ=1` manifold separates a smooth contractive region from an expansive region,
and the cliff's steepness grows like `ρ^T` (razor-sharp at depth ~6). This is the working explanation for
the β1=0 AdEMAMix detonations (Task #276): **clamping the realized magnitude (`core_gain_clip`) masks the
symptom but not the disease — the disease is `ρ(J_core)` crossing 1, which the optimizer can't see and a
magnitude clamp doesn't touch.** Implication for *any* fix here (and for looped/recursive/weight-shared
nets generally): target **contractivity** (`ρ≤1`: spectral/Lipschitz control, direction-preserving carrier
renorm), not symptom-clamping — that also preserves the β1=0 memory win + α·m_slow gains.
**Full writeup + evidence chain + the decisive `ρ(J_core)` probe:**
`Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md`.

## ⭐ TUL — Thought Unpack Loop (this copy, branch `experiments/tul`) — IMPLEMENTED, NOT RUN

`docs/tul-spec.md` is the source of truth; `docs/figures/tul_overview.png` is the diagram;
`docs/references.md` §13 lists every paper a decision comes from. Read all three before
touching the model for TUL. Short mental model:

- ONE shared sequence of token positions and **slot positions** (one slot after each span;
  boundary rule `.;!?` + newline + dashes, NO comma; a `.`+`\n` run is ONE boundary;
  min span 4; cap 32). Slot input = `E_slot` + mean of the span's token embeddings
  (TST-native). Prelude runs on all positions.
- **Core loops on SLOT positions only** (gather → loop → scatter), per-SLOT Poisson depth as a
  masked update (never a per-position gather — frozen slots still serve K/V). Tokens skip
  the core: coda input for tokens = `input_norm(prelude)` (the existing `n_core == 0` path).
- Coda runs on all positions; tokens attend slots as ordinary positions (the plan is a
  PREFIX the coda refines — Block Transformer Fig 3f, Coconut). Slot label = first token of
  the next span; the slot's CORE state has no loss of its own. `slot_id` logit is masked.
- Token-state dropout `p` on the coda input (Bowman word dropout) is the collapse tax.
- `slot_layout` is a per-forward ARGUMENT like `bag_size`; `None` ⇒ bit-identical to today.
  The 5090 arms (`tul_short.yaml`) run it from step 0 with TST off and prune/carve/route off;
  `tul.activate_at = ${training.tst_ratio}` is the full-schedule variant (switch at the TST
  boundary). One config per arm: `tul_a0` / `tul_a1` / `tul_a1r` / `tul_a3`, all at batch 14.
- NEVER decode a span from one vector + offset with no token path (Huginn 2026-08-16 collapse;
  MegaByte T7; Bowman T2; Hourglass T6). Never regress onto the slot state (LCM, CoCoMix, BT §4.2).
- Arms and gates: `docs/ablation-ledger.md` "Planned — TUL"; invariants: `runtime-invariants.md` §6b
  (LIVE, each row names the test that fails when it breaks).
- Lineage: successor of coconut's `tul/` + `ltd/` (fine-tunes of frozen models; left behind).

**Where the code is** (v1, `pytest tests/` → 116 passed; no arm has been TRAINED yet):

| File | What |
| --- | --- |
| `morph/model/tul_layout.py` | The ONE causal boundary rule (`BoundaryRule.cut`, a resumable state machine used by the loader AND the generator), the fixed-shape row packer (`pack_tul_row` / `pack_tul_batch`), `SlotLayout`. |
| `morph/model/tul.py` | `TULConfig` (construction-time switches), `TULSlots` (`E_slot`, `E_mask`, `W_prefix`), the span bag-mean, gather/scatter, token-state dropout. |
| `morph/model/transformer.py` | `_core_region` (the old inline core loop, extracted verbatim), `_tul_front` / `_tul_core` / `_tul_group_losses` / `_forward_tul`. `slot_layout` is a forward ARGUMENT; `None` is the untouched baseline. |
| `morph/training/tul_setup.py` | Resolves the `tul:` Hydra block once → ids, rule, configs, wandb manifest. |
| `morph/inference/tul_generate.py` | Eager recompute-per-step generator (spec §6 v1 — no KV cache by design). |

Two v1 deviations from the spec text, both recorded in `tul-spec.md` and §6b: run
collapse is CAUSAL (boundary after the FIRST token of a run — the spec's "after the
LAST" needs a lookahead the generator cannot have), and the packer pads a row's last
≤ `prefix_k` positions rather than dropping a boundary. Arms `stp_lambda`,
`set_lambda`, `carry`, `xattn`, `bcast` are NOT built and RAISE if configured.

## Architecture

Loop hierarchy:
1. **Inner**: Parcae core loop — local **3 prelude + 6 core × T + 3 coda** (cloud target 4:8:4,
   d=2048). T = per-sequence Poisson depth (mean 6, max 8), truncated BPTT depth 4, gradient
   checkpointed. d_model=768, d_ff=2048, seq 4096 locally.
2. **Outer**: Zyphra RSA harness (inference-time, requires RL — deferred)

Attention: CCA channel compression → CSA sparse global + HCA dense compressed (alternating layers),
with XSA, Residual Attention, CoPE Clipped RoPE, QK-Norm baked in.

Embeddings: Hybrid (euclidean + Lorentz hyperbolic, `lorentz_fraction=0.25`) + hash-bigram
(`bigram_hash_vocab=49152`). Loss is plain cross-entropy (fused/chunked when
`model.use_kernels=true`).

## Design Principles

- **No runtime feature flags.** Features are baked in at init. No `if use_feature:` in forward pass.
  torch.compile sees a clean graph with no branching.
- **PyTorch-first.** The JAX/Flax mirror (`morph/jax/`) and the PT↔JAX converter
  (`morph/interop/checkpoint.py`) exist but lag the PyTorch model; verify parity before relying on them.
- **Hydra configs.** All hyperparams in YAML (`morph/configs/`). Every run reproducible from its wandb config.
- **Custom kernels.** Triton (GPU), SM120 tuned for 5090. (`morph/kernels/pallas/` is currently empty.)

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"

# Train (PyTorch, GPU) — Hydra entry point, defaults to morph/configs/base.yaml
python -m morph.training.train
python -m morph.training.train training.steps=50000 training.batch_size=4   # overrides
python -m morph.training.train --config-name pretrain_curriculum            # other configs

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
    transformer.py       # Core looped transformer (Parcae loop, DiagonalInjection; _SwiGLUMortar hosts the ReMoE router)
    attention.py         # CCA+CSA+HCA+XSA+ResAttn+CoPE (one module, no flags)
    embeddings.py        # Hybrid (eucl+Lorentz) + bigram
    hyper_connections.py # HyperConnectionResidual (Cayley n=4, fused kernel)
    mhc.py               # MORPHBlock wiring + ChannelInject (mrr_* attrs = HC modules, legacy names)
    gla.py               # GLA retention branch
    sparsity.py          # MortarLinear (dense pre-carve → MORTAR 128×128 BCSR post-carve)
    routing.py           # TileRouter — whole-body ReMoE over the d_ff hidden-neuron bank
    ternary_qat.py       # ternary forward-STE QAT
    embed_quant.py       # int8/int6 embedding QAT
    attn_proj_quant.py   # attention-projection QAT (opt-in)
    fp8_scope.py         # FP8 scoping (off by default)
    fused_ce.py          # chunked/fused cross-entropy host
    kv_quant.py          # inference KV cache quantization
    layers/              # CMSBlockLinear (block-sparse scoring), topology scorer, norms
  kernels/
    triton/              # GPU kernels (fused attention, HC, GLA, decode, router, CE)
    l2_persist.py        # L2 cache persistence helper
  sparse/stk/            # vendored BCSR sparse execution backend
  training/
    train.py             # Single Hydra entry point
    optimizer.py         # AdamW + STE ternary shadow weights
    ademamix_b1zero.py   # β1=0 AdEMAMix optimizer (+ fused kernel)
    spectral_penalty.py  # core-map spectral-norm penalty
    data.py              # OpenWebText + StarCoder2 tokenizer
    curriculum_data.py / curriculum.py  # context-length curriculum loader + schedule
    sft.py / sft_data.py # SFT fine-tuning
    pruning.py           # CMS schedule (dense → prune → carve → route); density helper logs skips
  inference/             # generation engine, KV cache, deploy quant
  posttrain/             # deploy artifacts, masks, validation
  jax/                   # JAX/Flax model mirror (lags PT: still MRR residual; kernels/ empty)
  interop/
    checkpoint.py        # PT ↔ JAX converter (name-driven)
  configs/               # Hydra YAML (base, cloud, pretrain_curriculum[_smoke], scale30b, sft)
tests/
docs/
```

## Critical Patterns

### MORTAR Sparsity Schedule
Dense → prune (prune_step_blocks, 128×128-aligned) → carve() to MORTAR BCSR at
compact_step → ReMoE route at route_start. MORTAR is the ONLY sparse backend
(there is no sparse_backend knob). Saliency is scored at tile_size=16; pruning and
execution blocks are 128×128.
accumulate_scores() MUST be called between loss.backward() and optimizer.zero_grad().
**GOTCHA: if `prune_start` is disabled (e.g. 9999999), no pruning happens and carve()
then runs on a still-dense model → K/C=1.0. Always confirm `[prune] density=…` actually
fell before trusting a "0.25 sparse" claim.** The `base.yaml` cadence (prune_start=3000,
prune_interval=167) reaches target_density 0.25 by ~step 27050, before carve at
compact_step=29000 and routing at route_start=30000.

### torch.compile
mode="default" (NOT reduce-overhead — CUDA graphs cause eval OOM).
No fullgraph=True (the looped core uses gradient checkpointing, use_reentrant=False).
torch._functorch.config.donated_buffer = False (import the submodule explicitly first).

### Project Cleanliness
Do not litter scripts around the directory. Temporary scripts, agent notes, Hydra
`outputs/`, and local `wandb/` live under `ignore/`, which is a **private** git repo
(`morph-scratch`) and is gitignored by the public tree. Root paths `Ai-notes`,
`ai-notes` (same target — agents forget the capital), `Incomplete`, `outputs`, and
`wandb` are symlinks into `ignore/`. Write notes to either spelling; both resolve
to `ignore/Ai-notes/`. Do not commit weights, `*.m2g`, `*.optlog`, or perf traces
into the private repo either — see `ignore/.gitignore`.
