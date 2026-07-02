# MORPH

**MORPH** is a PyTorch research model for looped transformer training and sparse deployment. The model reuses a small Parcae-style core for variable depth, stabilizes the repeated core with Cayley Hyper-Connections, and trains the MLP stack while pruning low impact weights to a 25% total density before carving it into the MORTAR BCSR runtime. Enabling less than 1% ppl regression and improved memory footprint and training throughput. All while natively quantized trained.

To further improve per bit intelligence and memory foot print for research, it utilizes extensive linear attention methods to provide a lower ppl at long contexts with less memory.

Extensive ablations have forced ever component to earn their keep with in the architecture. It is the goal of the MORPH project to provide a true open source architecture that stays at the bleeding edge of research.

The PyTorch path is the implementation target. The JAX/Flax mirror under `morph/jax/` is maintained as a converter target and currently lags the PyTorch architecture.

## Current Architecture

The default local model is defined in `morph/configs/base.yaml`: `3 + 6xT + 3` blocks, `d_model=768`, `d_ff=2048`, sequence length 4096, Poisson loop depth with mean 6 and max 8, and truncated BPTT over the last four core iterations. The cloud target is `4 + 8xT + 4` at `d_model=2048`.

The active stack is:

- **Looped transformer body:** prelude blocks, a shared core loop, and coda blocks.
- **Cayley Hyper-Connections:** four residual carrier streams across the network, reduced before the output head.
- **CCA + CSA/HCA attention:** channel-compressed attention with local window attention plus alternating sparse and dense compressed global context.
- **GLA retention:** a gated branch beside attention on configured section-local layers, with optional carry across core-loop iterations.
- **Hybrid embeddings:** Euclidean token embeddings, a Lorentz channel, and a learned hash-bigram signal injected through the body.
- **MORTAR sparse MLP path:** `MortarLinear` is used throughout the MLP stack, with CMS pruning and 128x128 BCSR carve.
- **ReMoE routing:** whole-body hidden-neuron routing after carve.
- **Deploy QAT:** ternary backbone weights, int6 Euclidean/bigram embeddings, and 8-bit AdamW optimizer state by default.

The residual modules in `MORPHBlock` are still named `mrr_attn` and `mrr_mlp` for checkpoint compatibility. In the PyTorch model they are `HyperConnectionResidual` modules, not Multi-Rate Residual modules.

## Training Recipe

`morph/configs/base.yaml` is the source of truth for the current training recipe. The default run is a 100k-step local training schedule with flat `1e-4` learning rate, CMS pruning, MORTAR carve, ReMoE routing, Token Superposition Training, ternary backbone QAT, int6 embedding QAT, and 8-bit AdamW.

High-level schedule:

| Phase | Config keys |
| --- | --- |
| Dense masked training | `training.prune_start`, `training.prune_interval` |
| CMS pruning | `training.target_density`, `training.cms_score_mode` |
| MORTAR carve | `training.compact_step` |
| ReMoE routing | `routing.route_start`, `routing.route_scope` |
| Token Superposition Training | `training.tst_bag_size`, `training.tst_ratio` |

Evaluation and generation use normal next-token prediction. TST is a training-only data-efficiency phase.

For dense curriculum work, use `pretrain_curriculum.yaml`. It deliberately disables sparse carve/routing, TST, ternary QAT, int6 embedding QAT, and 8-bit AdamW so curriculum behavior can be isolated.

## Quick Start

```bash
pip install -e .
pip install -e ".[train]"

python -m morph.training.train
python -m morph.training.train training.steps=50000 training.batch_size=4
python -m morph.training.train --config-name pretrain_curriculum
```

Training logs the resolved Hydra config to Weights & Biases when W&B is enabled.

## Repository Map

```text
morph/
  model/
    transformer.py          # MORPHTransformer, looped core, DiagonalInjection, _SwiGLUMortar host
    attention.py            # CCA, local window, CSA/HCA attention
    embeddings.py           # Euclidean + Lorentz + hash-bigram embeddings
    hyper_connections.py    # HyperConnectionResidual
    mhc.py                  # MORPHBlock wiring and ChannelInject
    gla.py                  # GLA retention branch
    sparsity.py             # MortarLinear wrapper
    routing.py              # TileRouter
    ternary_qat.py          # Ternary forward-STE QAT
    embed_quant.py          # int8/int6 embedding QAT
    attn_proj_quant.py      # attention-projection QAT experiments
    fused_ce.py             # chunked/fused cross-entropy
    kv_quant.py             # inference KV cache quantization
  kernels/
    triton/                 # fused attention, HC, GLA, decode, router, CE/support kernels
    l2_persist.py           # L2 cache persistence helper
  sparse/stk/               # vendored BCSR sparse execution backend
  training/
    train.py                # Hydra training entry point
    pruning.py              # prune -> carve -> route coordinator
    optimizer.py            # AdamW, 8-bit AdamW, ternary shadow optimizer support
    ademamix_b1zero.py      # beta1=0 AdEMAMix optimizer
    spectral_penalty.py     # core-map spectral-norm penalty
    data.py                 # OpenWebText + StarCoder2 streaming loader
    curriculum_data.py      # pretokenized multi-source curriculum loader
    curriculum.py           # context-length curriculum schedule
  inference/                # generation engine, KV cache, deploy quantization
  posttrain/                # deploy artifacts, masks, validation
  jax/                      # JAX/Flax mirror; not feature-parity with PyTorch
  interop/                  # PyTorch/JAX checkpoint conversion
  configs/                  # Hydra configs
docs/
  figures/                  # TikZ source diagrams
  references.md             # paper map and implementation notes
```

## Figures And References

Architecture diagrams are maintained as TikZ sources in `docs/figures/*.tex`. Regenerate them from `docs/figures/` with `pdflatex` when diagram content changes.

The paper map lives in `docs/references.md`, with local notes under `docs/references/`.

## License

Research code. See repository for terms.
