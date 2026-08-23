# MORPH Reference Archive

Local markdown and PDF archive of papers and technical references used by
`../references.md`. The archive is grouped by MORPH component so related papers
stay together.

## Topic Index

| Topic | Local references |
|---|---|
| Looping & depth | [Parcae / Poisson depth sampling](looping-depth/2604.12946.md) |
| Attention | [CCA / value shift](attention/2510.04476.md), [DeepSeek-V4 CSA/HCA](attention/deepseek-v4.md), [XSA](attention/2603.09078.md), [Residual Attention](attention/2603.15031.md), [CoPE](attention/2602.05258.md), [Attention Sinks](attention/2309.17453.md) |
| Memory | [GLA](memory/2312.06635.md), [RAVEN](memory/raven.md), [Titans](memory/2501.00663.md), [Nested Learning](memory/2512.24695.md) |
| Residual streams | [mHC](residual-streams/2512.24880.md), [JPmHC Cayley HC](residual-streams/2602.18308.md), [Hyper-Connections](residual-streams/2409.19606.md) |
| Embeddings | [Lorentz embeddings](embeddings/1806.03417.md), [Hybrid embeddings](embeddings/hybrid-embeddings-gu2019.md) |
| Sparsity & routing | [Lottery Ticket Hypothesis](sparsity-routing/1803.03635.md), [Block-ELL](sparsity-routing/block-ell-nvidia.md), [MegaBlocks / STK](sparsity-routing/2211.15841.md), [ReMoE](sparsity-routing/2412.14711.md), [PEER](sparsity-routing/1907.05242.md) |
| Regularization & latent objectives | [STP](regularization-objectives/2602.22617.md), [LeJEPA / SIGReg](regularization-objectives/2511.08544.md), [LLM-JEPA](regularization-objectives/2509.14252.md), [Semantic Step Prediction](regularization-objectives/2604.18464.md) |
| Feed-forward networks | [SwiGLU](feed-forward/2002.05202.md) |
| Training objectives & QAT | [MTP](training-objectives/2404.19737.md), [STE ternary / BitNet b1.58](training-objectives/2402.17764.md), [Token Superposition Training](training-objectives/2605.06546.md), [DiffusionBlocks](training-objectives/2506.14202.md) |
| Optimizer | [AdEMAMix](optimizer/2409.03137.md) |
| Tokenization & data | [StarCoder2](tokenization-data/2402.19173.md) |
| Inference scaling | [Zyphra RSA / ZAYA1-8B](inference-scaling/2605.05365.md) |
| TUL — latent emission & hierarchy (spec) | [BLT](tul-latent-emission/2412.09871.md), [MegaByte](tul-latent-emission/2305.07185.md), [H-Net](tul-latent-emission/2507.07955.md), [Block Transformer](tul-latent-emission/2406.02657.md), [Dynamic Token Pooling](tul-latent-emission/2211.09761.md), [Hourglass](tul-latent-emission/2110.13711.md), [Patch-Level Training](tul-latent-emission/2407.12665.md), [DeepSeek-V3 MTP](tul-latent-emission/2412.19437.md), [Future Lens](tul-latent-emission/2311.04897.md), [Blockwise Parallel Decoding](tul-latent-emission/1811.03115.md), [Medusa](tul-latent-emission/2401.10774.md), [Coconut](tul-latent-emission/2412.06769.md), [Quiet-STaR](tul-latent-emission/2403.09629.md), [AGCLR](tul-latent-emission/2606.07720.md), [CODI](tul-latent-emission/2502.21074.md), [CCoT](tul-latent-emission/2412.13171.md), [Looped Transformers](tul-latent-emission/2502.17416.md), [Sentence Embedding Prediction](tul-latent-emission/2505.22202.md), [LCM](tul-latent-emission/2412.08821.md), [SONAR](tul-latent-emission/2308.11466.md), [CoCoMix](tul-latent-emission/2502.08524.md), [Sentence VAE (PDF)](tul-latent-emission/1511.06349.pdf), [Lagging Inference (PDF)](tul-latent-emission/1901.05534.pdf), [Optimus (PDF)](tul-latent-emission/2004.04092.pdf), [Latent Transformer (PDF)](tul-latent-emission/1803.03382.pdf), [NAT](tul-latent-emission/1711.02281.md), [LLaDA](tul-latent-emission/2502.09992.md), [BD3-LM](tul-latent-emission/2503.09573.md), [LD4LG](tul-latent-emission/2212.09462.md), [Explorative Modeling](tul-latent-emission/2607.27372.md), [SpaceByte](tul-latent-emission/2404.14408.md), [AU-Net (PDF)](tul-latent-emission/2506.14761.pdf), [Hierarchical AT (PDF)](tul-latent-emission/2501.10322.pdf) |

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
