<p align="center">
  <img src="docs/figures/hydra.png" alt="MORPH hydra" width="320" />
</p>
<p align="center">
<h1 style="text-align: center;">
MORPH 0.1
</h1>
</p>

**MORPH** is a PyTorch research model for **looped transformer** training and sparse deployment. The model reuses a small Parcae-style core for variable depth, stabilizes the repeated core with [Cayley Hyper-Connection](docs/references/residual-streams/jpmhc/jpmhc.md), and trains the MLP stack while pruning low impact weights down to as little as 25% total density before carving it into the MORTAR BCSR runtime. Enabling less than 1% ppl regression and improved memory footprint and training throughput over full density. All while natively quantized trained.

To further improve per bit intelligence and memory foot print for research, it utilizes extensive linear attention methods to enable a lower memory foot print at long contexts. Both [GLA](docs/references/memory/gla/gla.md) and [DeepSeek CSA/HCA](docs/references/attention/deepseek-v4/deepseek-v4.md) are used, with a convolutional based compression of the kv ([CCA](docs/references/attention/cca/cca.md)).

Extensive ablations have ran through dozens of [papers](docs/references.md) and techniques to carve out the MORPH Architecture. It is the goal of the MORPH project to provide a true open source architecture that stays at the bleeding edge of research.

---

The PyTorch path is the implementation target. The JAX/Flax mirror under `morph/jax/` is maintained as a converter target and currently lags the PyTorch architecture.

<p align="center">
  <img src="docs/figures/morph_overview.png" alt="MORPH architecture overview: hybrid embeddings into an HC carrier, prelude / looped core / coda, then LM head" width="720" />
</p>
<p align="center"><em>Architecture overview: Parcae-style prelude / core loop / coda on a <a href="docs/references/residual-streams/jpmhc/jpmhc.md">Cayley Hyper-Connection</a> carrier, with a gated <a href="docs/references/memory/gla/gla.md">GLA</a> retention branch on layer 1.</em></p>

## TUL: Thought Unpack Loop (merged, OFF by default)


<p align="center">
  <img src="docs/figures/tul_mechanism.png" alt="TUL: shared token/slot sequence into prelude; think (core × T on slots) saves z; freeze z in sequence; decode next span; cut on punctuation" width="720" />
</p>
<p align="center"><em>TUL: loop once per thought, freeze <code>z</code> in the sequence, amortize decode over the next span.</em></p>


TUL loops the Parcae core over one **thought slot per span** (punctuation-bounded) instead
of over every token, and decodes tokens with the slot's looped state visible as an attended
prefix position. Tokens run prelude → coda only. Specification, provenance and planned
ablations: [docs/tul-spec.md](docs/tul-spec.md); paper map: [docs/references.md](docs/references.md) §13;
measured arms: [lab/tul/arms-result.md](lab/tul/arms-result.md).

`base.yaml` ships `tul.activate_at: never`, which constructs no TUL parameters, so the
default recipe is bit-identical to plain MORPH and the main line's behaviour is unchanged.

**Results of TUL:** Over the same 20k steps the TUL arm beat the dense
baseline by 0.056 nats of `val/ce_tokens` (slightly better than noise) while running 177 minutes against 278. A 1.6x
wall-clock win at slightly better loss.

**FLOPs (same runs: `no-tul-a0-acap1` / `tul-a1-acap1`):** ~4× fewer FLOPs per token
(1793 → ~452 MFLOP forward pass; ~5.84 → ~1.47 GFLOP executed total) and ~3.9× fewer total
FLOPs over the run, at nearly equal token count. Wall clock only moved 1.6×
because the step is still launch / fixed-overhead bound. Details:
[lab/tul/arms-result.md](lab/tul/arms-result.md). Needs further optimization.

**TUL For Dummies:** 
- After the core loop save the hidden state, lets call z1.
- Use a frozen z1 to decode a span. A span goes to the next punctuation mark, or 32 tokens max. This is a loop through the coda.
- z1 kept in the sequence

So the core loop is forced to contain the full semantic thought and amortize the loop cost over many tokens. As opposed to looping many times per token.

This is based on a series of experiments run on
[Coconut](docs/references/tul-latent-emission/coconut/coconut.md),
[AGCLR](docs/references/tul-latent-emission/agclr/agclr.md), and
[Quiet-STaR](docs/references/tul-latent-emission/quiet-star/quiet-star.md)
(paper map: [docs/references.md](docs/references.md) §13).

---

From experimentation with looping and halting, halting mechanisms don't give a real win over no halting.
HRM dropped [ACT](docs/references/tul-latent-emission/act/act.md) for their LLM for this reason. The likely cause for this is what inspired TUL.

[Future Lens](docs/references/tul-latent-emission/future-lens/future-lens.md) shows us that the middle layers contain whole semantic thoughts, lets call ST. We can decode many tokens from it successfully.
Anthropic's [J-lens / J-space](docs/references/tul-latent-emission/j-space/j-space.md) is the same picture from the other direction: mid-depth verbalizable concepts held for future report, not just the next token.
Measurements I made while pretraining showed me that over a span of tokens (a ST or sentence) this latent barely moves.
The final hidden state moves a lot more, as it is decoding the actual token from the ST based on the position in the sequence.
The text gives state for what comes next in the sentence, and it mostly effects lower layers.

If looping more deeply gives a deeper thought. Every token in the span needs that thought to decode accurately.
Because previous methods are still doing autoregressive next token prediction, this looping must match per token in the span.
[PonderNet](docs/references/tul-latent-emission/pondernet/pondernet.md) and [ACT](docs/references/tul-latent-emission/act/act.md) ignore this and try to vary per token.
Difficulty that needs deeper thought lives at the span level and not the token level.

TUL gives a method of exploiting this, while genuinely reducing compute costs.

Testing is happening for a gated version. Initial results are good. It is based on Quiet-STaR.
The gate can produce a variable k, and k=0 is to loop. The magnitude of k determines how many tokens to decode for the next span.
Because we are delimitting the spans at easily detected values (punctuation), it is self supervised training for halting.


## Current Architecture

The default local model is defined in `morph/configs/base.yaml`: `3 + 6xT + 3` blocks, `d_model=768`, `d_ff=2048`, sequence length 4096, Poisson loop depth with mean 6 and max 8, and truncated BPTT over the last four core iterations. This is used for small scale testing and ablation. This fits comfortably on a 5090 at batch 4, and should fit on a 4090 if allocations do not fragment too much. Smaller sequence lengths can increase batch for these scales. 4k is selected to stress test during A/B ablation.

The cloud target is to prune from 3b to 1b params.

The active stack is:

- **[Looped transformer body](docs/references.md#parcae--stable-looped-transformer):** prelude blocks, a shared core loop, and coda blocks. Parcae style.
- **[Cayley Hyper-Connections](docs/references.md#jpmhc--jacobian-preserving-manifold-hyper-connections-cayley):** four residual carrier streams across the network, reduced before the output head.
- **[CCA](docs/references.md#cca--compressed-convolutional-attention) + [CSA/HCA](docs/references.md#csa--hca--compressed-sparse--heavily-compressed-attention) attention:** Compressed Convolutional Attention with local window attention plus alternating sparse and dense compressed global context. Providing sub-quadratic attention a la Deepseek, with further compression on the KV cache using CCA.
- **[GLA retention](docs/references.md#gla--gated-linear-attention-retention-branch):** a gated branch beside attention on configured section-local layers, with optional carry across core-loop iterations. Chosen over interleaving full attention blocks. TODO: [RAVEN](docs/references.md#raven--sparse-memory-routing-planned-on-gla) attention applied to GLA.
- **[Hybrid embeddings](docs/references.md#hybrid-mixed-curvature-embeddings):** Euclidean token embeddings, a [hyperbolic Lorentz](docs/references.md#lorentz--hyperbolic-embeddings) channel, and a learned hash-bigram signal injected through the body.
- **[MORTAR sparse MLP path](docs/mortar-bcsr.md)** ([MegaBlocks STK](docs/references.md#megablocks--block-sparse-gpu-kernels-stk)): MORTAR provides control over 16x16 groups of perceptrons to make tracking importance tractable as an EMA for pruning (don't need a matrix of equal size as the weights). It utilizes the MegaBlocks kernel to realize the performance benefits post carving. The 16x16 sizing is GPU tile friendly for the MegaBlocks kernel to compact into something that realizes the computational savings.
- **[ReMoE routing](docs/references.md#remoe--differentiable-moe-routing):** whole-body hidden-neuron routing after carve. Enables per token routing selection of 16x16 MORTAR tiles.
- **Deploy QAT:** [ternary](docs/references.md#ste-ternary--straight-through-estimator--bitnet-b158) backbone weights, int6 Euclidean/bigram embeddings, and 8-bit [AdEMAMix](docs/references.md#ademamix--dual-ema-adam-variant) optimizer state by default. Lorentz embeddings must stay in bf16.
- **Triton Kernels:** Extensive Triton kernels are provided (see the fused-kernel notes under [JPmHC](docs/references.md#jpmhc--jacobian-preserving-manifold-hyper-connections-cayley), [GLA](docs/references.md#gla--gated-linear-attention-retention-branch), and [MegaBlocks STK](docs/references.md#megablocks--block-sparse-gpu-kernels-stk)).

<p align="center">
  <img src="docs/figures/morph_memory.png" alt="MORPH GLA retention: gated linear-attention branch parallel to attention, with sequence-axis SSM state and optional core-loop carry" width="720" />
</p>
<p align="center"><em>Retention: GLA branch on layer 1 of prelude / core / coda; sequence-axis SSM state always on, cross-iteration carry core-only and optional.</em></p>

<p align="center">
  <img src="docs/figures/morph_attention.png" alt="MORPH attention triple-axis compression: CCA prologue into local window and alternating CSA/HCA global-compressed branches, gated blend to attn out" width="720" />
</p>
<p align="center"><em>Attention: CCA channel compress, then local window plus alternating CSA (even) / HCA (odd) global-compressed branches, gated blend into the block residual.</em></p>

Paper attributions: [docs/references.md](docs/references.md).



## Training Recipe

`morph/configs/base.yaml` is the source of truth for the current local training recipe. The default run is a 100k-step local training schedule with flat `1e-4` learning rate, CMS pruning, MORTAR carve, ReMoE routing, [Token Superposition Training](docs/references.md#token-superposition-training-tst), ternary backbone QAT, int6 embedding QAT, and 8-bit AdEMAMix optimizer based on bits and bytes implementation. This is a clean ablation surface for A/B testing.

<p align="center">
  <img src="docs/figures/morph_cms_lifecycle.png" alt="MORPH MLP lifecycle: dense train, CMS block prune, MORTAR BCSR carve, then ReMoE routing" width="720" />
</p>
<p align="center"><em>MLP lifecycle: dense train → CMS prune to ~25% density → carve to MORTAR BCSR → whole-body ReMoE routing.</em></p>

High-level schedule:

| Phase | Config keys |
| --- | --- |
| Dense masked training | `training.prune_start`, `training.prune_interval` |
| [CMS pruning](docs/mortar-bcsr.md#cms-how-blocks-die) | `training.target_density`, `training.cms_score_mode` |
| [MORTAR carve](docs/mortar-bcsr.md#carve-what-actually-happens) | `training.compact_step` |
| [ReMoE routing](docs/references.md#remoe--differentiable-moe-routing) | `routing.route_start`, `routing.route_scope` |
| [Token Superposition Training](docs/references.md#token-superposition-training-tst) | `training.tst_bag_size`, `training.tst_ratio` |

Evaluation and generation use normal next-token prediction. TST is a training-only data-efficiency phase.

For dense curriculum work, use `pretrain_curriculum.yaml`. It deliberately disables sparse carve/routing, TST, ternary QAT, and int6 embedding QAT, so curriculum behavior can be isolated. To see if these optimizations are causing issues with an A/B.

AdEMAMix was partially based on Bits and Bytes implementation. AdEMA can have its memory foot print dramatically reduced by keeping beta-1 at 0, and not holding it in a full tensor. BnB maintains it as a full tensor even at 0.

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
.agents/notes/              # Public decision records (see AGENTS.md)
lab/                        # Spikes + campaign finals (TUL arms, runtime-invariants)
tests/
scripts/                    # verify_template, pretok, probes
morph/
  model/
    transformer.py          # MORPHTransformer, looped core, TUL forward paths
    tul.py / tul_layout.py  # Thought Unpack Loop (slots, boundary packer)
    attention.py            # CCA + CSA/HCA + XSA + ResAttn + CoPE
    embeddings.py           # Euclidean + Lorentz + hash-bigram
    hyper_connections.py    # HyperConnectionResidual (Cayley n=4)
    mhc.py                  # MORPHBlock wiring (mrr_* attrs = HC, legacy names)
    gla.py                  # GLA retention branch
    sparsity.py             # MortarLinear (dense → MORTAR BCSR)
    layers/                 # CMSBlockLinear, topology scorer, norms
    routing.py              # TileRouter (whole-body ReMoE)
    ternary_qat.py          # Ternary forward-STE QAT
    embed_quant.py          # int8/int6 embedding QAT
    attn_proj_quant.py      # attention-projection QAT (opt-in)
    fused_ce.py / kv_quant.py / fp8_scope.py
  kernels/triton/           # fused attention, HC, GLA, decode, router, CE
  sparse/stk/               # vendored MegaBlocks STK (MORTAR BCSR exec)
  training/
    train.py                # Hydra entry point
    pruning.py              # dense → prune → carve → route
    tul_setup.py            # resolve tul: Hydra block
    optimizer.py / ademamix_b1zero.py
    data.py / curriculum*.py / sft*.py
  inference/                # generation, KV cache, TUL generate
  posttrain/                # deploy artifacts, masks, validation
  jax/                      # JAX/Flax mirror (lags PyTorch)
  interop/                  # PT ↔ JAX checkpoint conversion
  configs/                  # Hydra YAML (base.yaml is recipe SoT)
docs/
  MANIFEST.md               # docs navigator
  mortar-bcsr.md            # CMS prune + MORTAR BCSR readout
  ablation-ledger.md        # accepted / rejected / deferred
  tul-spec.md               # Thought Unpack Loop contract
  olympiad-interop.md       # PT ↔ JAX / Olympiad notes
  figures/                  # PNG previews + topic-grouped TikZ sources
  references.md             # paper map + MORPH usage notes
  references/               # local paper archive (see references/MANIFEST.md)
ignore/                     # private scratch (wandb, Hydra outputs), not public
tile-prover/                # Lean/z3 confirmations of kernel correctness to original algorithm
```

## Figures And References

The figures above are the README highlights. The full set lives under `docs/figures/`
(PNG previews at the top; TikZ/PDF sources topic-grouped underneath). Start with
[`docs/MANIFEST.md`](docs/MANIFEST.md). Regeneration: [`docs/figures/MANIFEST.md`](docs/figures/MANIFEST.md).

Paper map: [`docs/references.md`](docs/references.md); local copies indexed by
[`docs/references/MANIFEST.md`](docs/references/MANIFEST.md). MORTAR format:
[`docs/mortar-bcsr.md`](docs/mortar-bcsr.md). Ablations: [`docs/ablation-ledger.md`](docs/ablation-ledger.md).
Runtime invariants: [`lab/runtime-invariants.md`](lab/runtime-invariants.md).
Known-good runs: [`.agents/notes/implemented/process/2026-07-03-known-good-runs.md`](.agents/notes/implemented/process/2026-07-03-known-good-runs.md).
Campaign logs and gate scripts stay under `ignore/` / `lab/`.

## Contributing

MORPH uses a stable snapshot plus research integration model. See `CONTRIBUTING.md` for branch policy, evidence expectations, and release rules. 

The biggest contributions towards MORPH would be benefiting the FOSS ML
ecosystem in other ways beyond this project. Such as building and testing RL environments.
Or thoroughly tested implementations of closed research papers.
Or hard forks to test variants of MORPH that can help steer design decisions across major versions.

The number of times open implementations dramatically sped up development of MORPH was huge. 
Often times I would drop a research direction for a few days and come back to 
it to find someone had released on github a related implementation in that time frame.
A lot of these are not referenced in docs because the due dilligence wasn't there
in recreating *at least* a facsimile of the original paper. Or there were claims/behaviors in the project
that I was not seeing in my testing. Overall a lot of these projects were sloppy, and tended to poison the LLM working on MORPH.

These projects sometimes were a giant waste of time, often times they still helped catch oversights
on implementation details even when other aspects were quite poor. The dilligence needs to be higher on these projects. Try to break the mechanism.
Show where it is weak. Test it on different architectures and data. And PLEASE stop training
things that are 95% embedding matrix and trying to use that as proof of concept or verification.
Aceing CIFAR is basically noise at this point. At least try to torch.compile.

These issues are often times upstream of the implementation. If you are recreating an implementation
of a closed paper you are likely walking into a minefield of things left unsaid and under specified.
You resolve this ambiguity by trying to break it. Found an attention mechanism that sounds really good?
It shows a 110% NIAH retrieval at 20m tok seqlen? Does it work on semantic retrieval too and not just word matching? Did you perturb the mechanism to prove its actually contributing?

## License

Apache License 2.0. See `LICENSE`.

Vendored third-party components keep their own license notices, including `morph/sparse/stk/LICENSE`.


## Thanks

- **DeepSeek** CSA / HCA
- **DeepSeek** tile-prover methodology
- **Zyphra** CCA
- **UCSD / Together AI** Parcae
- **MIT / IBM Research** GLA
- **Moonshot AI (Kimi)** Residual Attention
- **Apple ML Research** XSA
- **JPMorgan Chase** JPmHC (Cayley Hyper-Connections)
- **DeepSeek** mHC (related)
- **ByteDance** Hyper-Connections (related)
- **FAIR** Lorentz embeddings
- **Stanford** hybrid / mixed-curvature embeddings
- **MIT CSAIL** Lottery Ticket Hypothesis
- **Stanford / MSR / Google** MegaBlocks STK
- **Tsinghua** ReMoE
- **FAIR** PEER
- **Microsoft Research** BitNet / STE ternary
- **Nous Research** Token Superposition Training
- **EPFL / Apple** AdEMAMix
- **BigCode / Hugging Face** StarCoder2 tokenizer
- **Google** SwiGLU
- **Bits n Bytes** AdEMAMix reference implementation

## TODO
- [RAVEN](docs/references.md#raven--sparse-memory-routing-planned-on-gla) attention on GLA.