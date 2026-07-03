# MORPH Reference Archive

Local markdown and PDF archive of papers and technical references used by
`../references.md`. The archive is grouped by MORPH component so related papers
stay together.

## Topic Index

| Topic | Local references |
|---|---|
| Looping & depth | [Parcae / Poisson depth sampling](looping-depth/2604.12946.md) |
| Attention | [CCA / value shift](attention/2510.04476.md), [DeepSeek-V4 CSA/HCA](attention/deepseek-v4.md), [XSA](attention/2603.09078.md), [Residual Attention](attention/2603.15031.md), [CoPE](attention/2602.05258.md), [Attention Sinks](attention/2309.17453.md) |
| Memory | [Titans](memory/2501.00663.md), [Nested Learning](memory/2512.24695.md) |
| Residual streams | [mHC](residual-streams/2512.24880.md), [JPmHC Cayley HC](residual-streams/2602.18308.md), [Hyper-Connections](residual-streams/2409.19606.md) |
| Embeddings | [Lorentz embeddings](embeddings/1806.03417.md), [Hybrid embeddings](embeddings/hybrid-embeddings-gu2019.md) |
| Sparsity & routing | [Lottery Ticket Hypothesis](sparsity-routing/1803.03635.md), [Block-ELL](sparsity-routing/block-ell-nvidia.md), [MegaBlocks / STK](sparsity-routing/2211.15841.md), [ReMoE](sparsity-routing/2412.14711.md), [PEER](sparsity-routing/1907.05242.md) |
| Regularization & latent objectives | [STP](regularization-objectives/2602.22617.md), [LeJEPA / SIGReg](regularization-objectives/2511.08544.md), [LLM-JEPA](regularization-objectives/2509.14252.md), [Semantic Step Prediction](regularization-objectives/2604.18464.md) |
| Feed-forward networks | [SwiGLU](feed-forward/2002.05202.md) |
| Training objectives & QAT | [MTP](training-objectives/2404.19737.md), [STE ternary / BitNet b1.58](training-objectives/2402.17764.md), [Token Superposition Training](training-objectives/2605.06546.md) |
| Optimizer | [AdEMAMix](optimizer/2409.03137.md) |
| Tokenization & data | [StarCoder2](tokenization-data/2402.19173.md) |
| Inference scaling | [Zyphra RSA / ZAYA1-8B](inference-scaling/2605.05365.md) |

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
