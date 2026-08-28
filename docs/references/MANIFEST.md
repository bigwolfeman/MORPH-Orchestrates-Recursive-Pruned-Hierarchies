# MORPH Reference Archive

Local markdown and PDF archive of papers and technical references used by
`../references.md`. The archive is grouped by MORPH component so related papers
stay together.

## Topic Index

| Topic | Local references |
|---|---|
| Looping & depth | [Parcae / Poisson depth sampling](looping-depth/parcae/parcae.md) |
| Attention | [CCA / value shift](attention/cca/cca.md), [DeepSeek-V4 CSA/HCA](attention/deepseek-v4/deepseek-v4.md), [XSA](attention/xsa/xsa.md), [Residual Attention](attention/residual-attention/residual-attention.md), [CoPE](attention/cope/cope.md), [Attention Sinks](attention/attention-sinks/attention-sinks.md) |
| Memory | [GLA](memory/gla/gla.md), [RAVEN](memory/raven/raven.md), [Titans](memory/titans/titans.md), [Nested Learning](memory/nested-learning/nested-learning.md) |
| Residual streams | [mHC](residual-streams/mhc/mhc.md), [JPmHC Cayley HC](residual-streams/jpmhc/jpmhc.md), [Hyper-Connections](residual-streams/hyper-connections/hyper-connections.md) |
| Embeddings | [Lorentz embeddings](embeddings/lorentz-embeddings/lorentz-embeddings.md), [Hybrid embeddings](embeddings/hybrid-embeddings/hybrid-embeddings.md) |
| Sparsity & routing | [Lottery Ticket Hypothesis](sparsity-routing/lottery-ticket-hypothesis/lottery-ticket-hypothesis.md), [Block-ELL](sparsity-routing/block-ell-nvidia/block-ell-nvidia.md), [MegaBlocks / STK](sparsity-routing/megablocks/megablocks.md), [ReMoE](sparsity-routing/remoe/remoe.md), [PEER](sparsity-routing/peer/peer.md) |
| Regularization & latent objectives | [STP](regularization-objectives/stp/stp.md), [LeJEPA / SIGReg](regularization-objectives/lejepa/lejepa.md), [LLM-JEPA](regularization-objectives/llm-jepa/llm-jepa.md), [Semantic Step Prediction](regularization-objectives/semantic-step-prediction/semantic-step-prediction.md) |
| Feed-forward networks | [SwiGLU](feed-forward/swiglu/swiglu.md) |
| Training objectives & QAT | [MTP](training-objectives/mtp/mtp.md), [STE ternary / BitNet b1.58](training-objectives/bitnet-b1.58/bitnet-b1.58.md), [Token Superposition Training](training-objectives/token-superposition-training/token-superposition-training.md), [DiffusionBlocks](training-objectives/diffusionblocks/diffusionblocks.md) |
| Optimizer | [AdEMAMix](optimizer/ademamix/ademamix.md) |
| Tokenization & data | [StarCoder2](tokenization-data/starcoder2/starcoder2.md) |
| Inference scaling | [Zyphra RSA / ZAYA1-8B](inference-scaling/zyphra-rsa-zaya1-8b/zyphra-rsa-zaya1-8b.md) |
| TUL — latent emission & hierarchy (spec) | [BLT](tul-latent-emission/blt/blt.md), [MegaByte](tul-latent-emission/megabyte/megabyte.md), [H-Net](tul-latent-emission/h-net/h-net.md), [Block Transformer](tul-latent-emission/block-transformer/block-transformer.md), [Dynamic Token Pooling](tul-latent-emission/dynamic-token-pooling/dynamic-token-pooling.md), [Hourglass](tul-latent-emission/hourglass/hourglass.md), [Patch-Level Training](tul-latent-emission/patch-level-training/patch-level-training.md), [DeepSeek-V3 MTP](tul-latent-emission/deepseek-v3-mtp/deepseek-v3-mtp.md), [Future Lens](tul-latent-emission/future-lens/future-lens.md), [ACT](tul-latent-emission/act/act.md), [PonderNet](tul-latent-emission/pondernet/pondernet.md), [J-lens / J-space](tul-latent-emission/j-space/j-space.md), [Blockwise Parallel Decoding](tul-latent-emission/blockwise-parallel-decoding/blockwise-parallel-decoding.md), [Medusa](tul-latent-emission/medusa/medusa.md), [Coconut](tul-latent-emission/coconut/coconut.md), [Quiet-STaR](tul-latent-emission/quiet-star/quiet-star.md), [AGCLR](tul-latent-emission/agclr/agclr.md), [CODI](tul-latent-emission/codi/codi.md), [CCoT](tul-latent-emission/ccot/ccot.md), [Looped Transformers](tul-latent-emission/looped-transformers/looped-transformers.md), [Sentence Embedding Prediction](tul-latent-emission/sentence-embedding-prediction/sentence-embedding-prediction.md), [LCM](tul-latent-emission/lcm/lcm.md), [SONAR](tul-latent-emission/sonar/sonar.md), [CoCoMix](tul-latent-emission/cocomix/cocomix.md), [Sentence VAE (PDF)](tul-latent-emission/sentence-vae/sentence-vae.pdf), [Lagging Inference (PDF)](tul-latent-emission/lagging-inference/lagging-inference.pdf), [Optimus (PDF)](tul-latent-emission/optimus/optimus.pdf), [Latent Transformer (PDF)](tul-latent-emission/latent-transformer/latent-transformer.pdf), [NAT](tul-latent-emission/nat/nat.md), [LLaDA](tul-latent-emission/llada/llada.md), [BD3-LM](tul-latent-emission/bd3-lm/bd3-lm.md), [LD4LG](tul-latent-emission/ld4lg/ld4lg.md), [Explorative Modeling](tul-latent-emission/explorative-modeling/explorative-modeling.md), [SpaceByte](tul-latent-emission/spacebyte/spacebyte.md), [AU-Net (PDF)](tul-latent-emission/au-net/au-net.pdf), [Hierarchical AT (PDF)](tul-latent-emission/hierarchical-at/hierarchical-at.pdf) |

## Source Of Truth

`../references.md` remains the curated architecture paper map with MORPH-specific usage notes.
This folder is the local source archive for deeper reading.

## Skipped By Design

| Reference | Reason |
|---|---|
| CMS Topology | Original MORPH work; implemented originially for Titan's Neural Memory, EMA portion borrowed to track tile contribution. |
| MRR residual streams | mHC prototype, MRR var names kept for checkpoint compatibility. |
| Value Shift | Covered by the CCA paper. |
| Poisson Depth Sampling | Covered by the Parcae paper. |
| SIGReg | Covered by the LeJEPA paper. |
| Gloeckle MTP, STP (TUL section) | Already archived under training-objectives / regularization-objectives; TUL cites the same files. |
