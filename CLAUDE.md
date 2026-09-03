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
> commented). Current schedule: 1000-step linear LR ramp then flat 1e-4 (min_lr==lr), taylor saliency,
> prune_start=3000 / prune_interval=167 (density hits 0.25 by ~step 27050) →
> carve at compact_step=29000 → whole-body ReMoE at route_start=30000, all inside a
> 100k-step run with TST superposition for the first 30k steps. Do not restate these
> numbers elsewhere — read the YAML.

## 🚨 STRANGE DIVERGENCE? READ [lab/divergence/DIVERGENCE-README.md](lab/divergence/DIVERGENCE-README.md) FIRST

Two campaigns (2026-08 slot-loop takeover, 2026-09 paid-axis detonation) already refuted
spectral caps, `core_gain_clip`, ternary-γ EMA and freeze, GLA-as-stabilizer, and dense
warmup. The README has the 60-second triage, the measured abort rule (`preclip/total >
1e4` at any step ≥ 200: 17/17 detonations caught by step 775, 0 false positives in 44
healthy runs), the open levers in priority order, and every instrument that already
exists. Do not re-derive any of it. **The cure is measured (2026-09-03): a 1000-step LR
ramp, now `training.warmup: 1000` in `base.yaml`, 0 detonations in 9 of 9 draws. Do not
override it to 0 on the ternary+AdEMAMix recipe.**

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
[`.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md`](.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md).
**The 2026-08 campaign settled the cure: the l2cap recipe (full BPTT + hard σ≤1.5
post-step projection, `morph/configs/tul_l2.yaml`) is the ONE recipe whose loop earns
depth (0.233 nats), and the identity-escape law says why every alternative failed —**
[`.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md`](.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md).

> **2026-08-24 correction, measured.** "Target contractivity" is right about WHERE to look
> and wrong about WHAT to bound. `ρ(J_core)` is now measurable
> ([`morph/training/core_jacobian.py`](morph/training/core_jacobian.py), procedure in
> [docs/cookbook/measuring-the-core-map.md](docs/cookbook/measuring-the-core-map.md)), and on
> the TUL takeover it says the map's per-block gain moves +2.5 % while its blocks' amplifying
> directions ALIGN x2.9 and the backward cotangent collapses from 13 effective slot positions
> to 2.5. Four interventions that bound the core weights’ spectrum — including a hard
> projection that pinned `σ_max` at 1.50 for a whole run — all failed, and two made it worse
> than doing nothing. A uniform rescale leaves every singular vector and every ratio
> `σ_1/σ_2` untouched, so it cannot slow an alignment. Read
> [lab/experiments/failures/2026-08-24-tul-takeover-cure.md](lab/experiments/failures/2026-08-24-tul-takeover-cure.md)
> before reaching for a spectral cap in this tree. What DID hold is upstream of the map: the
> 50 slot states of a row sit at effective rank 1.7–4.8 in 1024 dimensions because a slot's
> input is one shared `E_slot` plus a bag-mean, and the loop's effect on that rank flips sign
> at the onset. `tul.per_slot_embed` (removed from the tree 2026-09-03 with the slot path) was the best lever found and NOT a cure:
> it holds one seed of two, doubles the time to failure (step 1150 -> 2225) and reaches 0.78
> and 0.46 nats better val CE on the two seeds.

## ⭐ TUL — Thought Unpack Loop — THE PAID LOOP IS THE SHIPPED FORWARD (2026-09-03)

Status: implemented, run, measured, and ON in `base.yaml` (`tul.activate_at:
${training.tst_ratio}` — TUL switches on when the TST phase ends). The shipped forward is
the **paid loop** (`docs/tul-paid-loop-recipe.md`): tokens and slots are ONE sequence and
the ordinary per-sample Poisson-depth core runs over ALL of it. The slot-only arms
(A0/A1/A3/A4, the span-length gate, the TG restriction, the gist mux, the DB interleave,
per-slot embeddings, the fixed-stride rule) were **cut from the tree on 2026-09-03**;
their specs and results stay as records (`docs/tul-spec.md` §3.3, `docs/tul-gate-spec.md`,
`docs/tul-tg-spec.md`, `docs/gist-mux-recipe.md`, `docs/ablation-ledger.md`) and the last
commit that runs them is `d9e04e6`. Decision record with the alternatives weighed:
[`.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md`](.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md).

Why the paid loop and not the slot loop: on the slot-only arms the loop was a free ride
(K1−K6 ≤ 0.011 nats however long it ran); paying the loop on every position is the ONE
arm whose loop earns depth (0.168 nats at 5k, 0.100 at 20k —
`lab/experiments/successes/2026-09-02-*`, `failures/2026-09-02-warmup-20k-pair.md`).
The measured caveat, not to be hidden: at 20k on seq 1024 the paid arm sits 0.022 nats
BEHIND the plain model on 480 identical rows at 1.33x the wall clock; its matched-step
deficit closed 0.132 → 0.012 from 5k to 20k. Wolfe's 2026-09-03 call: ship it as the
recipe — the wall-clock gap is optimization work, not architecture. The `base.yaml`
conjunction (TUL × TST × prune/carve/route × seq 4096 × batch 4) is UNMEASURED beyond a
startup smoke.

Short mental model:

- ONE shared sequence of token positions and **slot positions** (one slot after each span;
  boundary rule `.;!?` + newline + dashes, NO comma; a `.`+`\n` run is ONE boundary;
  min span 4; cap 32; `prefix_k` = 2 positions per slot). Slot input (`slot_seed:
  boundary`) = `E_slot + W_sent · embed(t_last)`, `t_last` the span's LAST token;
  `bag_mean` (spec §3.2, `E_slot` + span mean) is the kept control.
- Prelude, the per-sample Poisson-depth core (`_core_region` — the SAME code the plain
  path runs) and the coda all run on every position. Nothing is gathered, projected or
  scattered. `W_prefix` exists only on the FM planner (`cfg.fm`, `n_core: 0`).
- Loss: ONE weighted CE. `emit_weight: 0.0` (the slot's own emit label carries no loss),
  `plast_weight: 1.0` (the boundary token keeps its ordinary label); the `slot_id` logit
  is masked; pad slots are −100. Token-state dropout 0.15 on the coda input (Bowman).
- `slot_layout` is a per-forward ARGUMENT like `bag_size`; `None` ⇒ bit-identical to the
  plain model (invariants §6b). Generation reads a span's first token from the boundary
  TOKEN position (`emit_source="token"`, the only trained head under `emit_weight: 0`).
- Configs: `base.yaml` (production: seq 4096, TST on, prune/carve/route on, batch 4,
  warmup 1000, retention off, `bptt_depth: 8` = full BPTT, spectral cap 0);
  `tul_a2.yaml` (the 5090 panel: seq 1024, batch 6, 20k steps, TUL from step 0,
  TST/prune/route off); `notul.yaml` (the same with `activate_at: never` — the matched
  control); `tul_smoke.yaml` (12-step small smoke); `tul_fm1.yaml` (FM planner arc).
- The detonation (`preclip/total > 1e4` at step ≥ 200) and its cure, the 1000-step LR
  ramp: `lab/divergence/DIVERGENCE-README.md`. The old slot-loop takeover campaign:
  `lab/divergence/takeover-campaign.md` (historical — the slot path is gone).
- NEVER decode a span from one vector + offset with no token path (Huginn 2026-08-16
  collapse; MegaByte T7; Bowman T2; Hourglass T6). Never regress onto the slot state
  (LCM, CoCoMix, BT §4.2).
- Old checkpoints carry `tul.W_prefix`; the loaders drop that ONE key, loudly, for a model
  without an FM planner (`morph/training/train.py::drop_retired_tul_keys`,
  `tests/test_checkpoint_compat.py`). Every other homeless key still raises.

**Where the code is**:

| File | What |
| --- | --- |
| `morph/model/tul_layout.py` | The ONE causal boundary rule (`BoundaryRule.cut`, a resumable state machine used by the loader AND the generator), the fixed-shape row packer (`pack_tul_row` / `pack_tul_batch`), `SlotLayout`. |
| `morph/model/tul.py` | `TULConfig` (six fields: `prefix_k`, `slot_id`, `token_state_dropout`, `emit_weight`, `plast_weight`, `slot_seed`), `TULSlots` (`E_slot`, `E_mask`, `W_sent`; `W_prefix` iff `with_prefix`), `bag_mean`, `boundary_token_index`, gather/scatter (FM planner + probes), token-state dropout. |
| `morph/model/transformer.py` | `_tul_front` → `_core_region` → `_back_region` → `_tul_group_losses`, wired in `_forward_tul`. `_tul_fm_core` / `tul_forward_ablated` are FM-planner only. `slot_layout` is a forward ARGUMENT; `None` is the untouched baseline. |
| `morph/training/tul_setup.py` | Resolves the `tul:` Hydra block once → ids, rule, configs, wandb manifest. Unknown `tul.*` keys RAISE. |
| `morph/inference/tul_generate.py` | Eager recompute-per-step generator (spec §6 v1 — no KV cache by design). |
| `tests/test_tul_forward.py`, `test_slot_seed.py`, `test_tul_layout.py`, `test_checkpoint_compat.py` | The contracts, one test per invariant row. |

Two v1 deviations from the spec text, both recorded in `tul-spec.md` and §6b: run
collapse is CAUSAL (boundary after the FIRST token of a run — the spec's "after the
LAST" needs a lookahead the generator cannot have), and the packer pads a row's last
≤ `prefix_k` positions rather than dropping a boundary.

## Documentation map (where to put / find writing)

Do not dump new markdown at repo root or into `Ai-notes/`. That path is gitignored private scratch.

| Kind of work | Where it goes |
|---|---|
| This file | Model/runtime brief and gotchas. Recipe **numbers** stay in `morph/configs/base.yaml`. |
| Standing orders | [AGENTS.md](AGENTS.md) |
| Specs, invariants, paper notes, figures, ablation table | `docs/` — **do not restructure** that tree; add a file or a section, or link. Start at [docs/MANIFEST.md](docs/MANIFEST.md). |
| Why we chose X, what we gave up | `.agents/notes/{proposed\|implemented\|rejected\|archived}/{class}/yyyy-mm-dd-slug.md` — read [.agents/notes/README.md](.agents/notes/README.md) before writing one |
| Hot-path subtree brief | [morph/model/CLAUDE.md](morph/model/CLAUDE.md) |
| Spikes, Experimental runs | `lab/` |
| wandb, Hydra `outputs/`, throwaway scripts | `ignore/` (private) |

After adding or moving Agent Notes, run `python scripts/verify_template.py`.



## Architecture

Loop hierarchy:
1. **Inner**: Parcae core loop — local **3 prelude + 6 core × T + 3 coda** (cloud target 4:8:4,
   d=2048). T = per-sequence Poisson depth (mean 6, max 8), truncated BPTT depth 4, gradient
   checkpointed. d_model=768, d_ff=2048, seq 4096 locally.
2. **Outer**: TUL

Attention: CCA channel compression → CSA sparse global + HCA dense compressed (alternating layers),
with XSA, Residual Attention, CoPE Clipped RoPE, QK-Norm baked in.

Embeddings: Hybrid (euclidean + Lorentz hyperbolic, `lorentz_fraction=0.25`) + hash-bigram
(`bigram_hash_vocab=49152`). Loss is plain cross-entropy (fused/chunked when
`model.use_kernels=true`).

## Design Principles

- **No runtime feature flags.** Features are baked in at init. No `if use_feature:` in forward pass.
  **torch.compile sees a clean graph with no branching.**
- **PyTorch-first.** The JAX/Flax mirror (`morph/jax/`) and the PT↔JAX converter
  (`morph/interop/checkpoint.py`) exist but lag the PyTorch model; verify parity through bit exactness before relying on them.
- **Hydra configs.** All hyperparams in YAML (`morph/configs/`). Every run reproducible from its wandb config.
- **Custom kernels.** Triton (GPU), SM120 tuned for 5090. (`morph/kernels/pallas/` is currently empty.)

## Commands

### ⚠ ALWAYS export this before any GPU run

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Not optional on the 5090.** Without it the TUL path OOMs on the FIRST backward with
8.17 GB reserved-but-unallocated (`docs/ablation-ledger.md`). With it, the TUL arms still
sit at **~26.2 GB resident** on a 31.4 GB card that a 3-monitor desktop already uses ~6 GB
of — roughly 1 GB of slack. That is how the 2026-08-21 gate bake-off died at step 1050
with 157 MiB free. Measured 2026-08-22: the training step only ALLOCATES 22.2–23.6 GB;
the remaining 2.7–4.0 GB is caching-allocator slack driven by the looped core's varying
active-set sizes (1…14 distinct shapes). Neither a bf16 carrier (−1.42 GB allocated) nor
an `empty_cache()` after the compile warmup moves the resident figure — both were tried
and measured. Every script under `ignore/perf/` exports it; a bare `python -m
morph.training.train` does NOT, so export it in your shell.

```bash
# Install
pip install -e .
pip install -e ".[dev]"

# Train (PyTorch, GPU) — Hydra entry point, defaults to morph/configs/base.yaml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # see above — mandatory
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
.agents/notes/           # Public decision records (see AGENTS.md)
lab/                     # Spikes, Test various algorithms, optimizations, and changes on a branch and document FINAL results here. Implementation and intermediate results go in .agents/notes/
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
Format readout (CMS prune + MORTAR BCSR): [docs/mortar-bcsr.md](docs/mortar-bcsr.md).
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
Do not litter scripts around the directory. Temporary scripts, Hydra
`outputs/`, and local `wandb/` live under `ignore/`, which is a **private** git repo
(`morph-scratch`) and is gitignored by the public tree. Root paths `Ai-notes`,
`ai-notes`, `Incomplete`, `outputs`, and `wandb` are leftover symlinks into `ignore/`.
**New decision notes go in `.agents/notes/` (public).** Do not write new markdown
to `Ai-notes/`. Do not commit weights, `*.m2g`, `*.optlog`, or perf traces
into the private repo either — see `ignore/.gitignore`.
