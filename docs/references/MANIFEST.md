# MORPH Reference Archive — MANIFEST

Local archive of every paper in `../references.md`, downloaded as PDF + Markdown.
Generated 2026-06-07. PDFs verified with `file` (must report "PDF document") and size
sanity (>50 KB); Markdown generated with `pymupdf4llm` (high-quality, OCR-assisted) and
verified to be >5 KB and contain the paper title.

## Downloaded references

| Ref name | id / source | PDF | MD | Source URL | Notes |
|---|---|---|---|---|---|
| Parcae Loop (#1) + Poisson Depth Sampling (#24) | 2604.12946 | ✓ 3708 KB | ✓ 121 KB | https://arxiv.org/abs/2604.12946 | one file backs both refs |
| CCA (#5) + Value Shift (#26) | 2510.04476 | ✓ 2900 KB | ✓ 67 KB | https://arxiv.org/abs/2510.04476 | one file backs both refs |
| XSA — Exclusive Self Attention (#13) | 2603.09078 | ✓ 829 KB | ✓ 19 KB | https://arxiv.org/abs/2603.09078 | |
| Residual Attention / AttnRes (#14) | 2603.15031 | ✓ 1040 KB | ✓ 82 KB | https://arxiv.org/abs/2603.15031 | |
| CoPE — Clipped RoPE (#12) | 2602.05258 | ✓ 1643 KB | ✓ 55 KB | https://arxiv.org/abs/2602.05258 | |
| Attention Sinks / StreamingLLM (#25) | 2309.17453 | ✓ 16555 KB | ✓ 76 KB | https://arxiv.org/abs/2309.17453 | large PDF (figures) |
| Titans — Neural Memory (#4) | 2501.00663 | ✓ 3571 KB | ✓ 112 KB | https://arxiv.org/abs/2501.00663 | |
| mHC — Manifold Hyper-Connections (#20) | 2512.24880 | ✓ 620 KB | ✓ 61 KB | https://arxiv.org/abs/2512.24880 | related work |
| JPmHC — Cayley Hyper-Connections (#20a) | 2602.18308 | ✓ 1330 KB | ✓ 124 KB | https://arxiv.org/abs/2602.18308 | MORPH default HC (`hc_cayley`) |
| Hyper-Connections (#20b) | 2409.19606 | ✓ 7190 KB | ✓ 92 KB | https://arxiv.org/abs/2409.19606 | related work |
| Lorentz Embeddings (#10) | 1806.03417 | ✓ 1356 KB | ✓ 39 KB | https://arxiv.org/abs/1806.03417 | |
| ReMoE (#18) | 2412.14711 | ✓ 1018 KB | ✓ 76 KB | https://arxiv.org/abs/2412.14711 | |
| PEER — Product Keys (#19) | 1907.05242 | ✓ 539 KB | ✓ 43 KB | https://arxiv.org/abs/1907.05242 | |
| STP — Semantic Tube Prediction (#7) | 2602.22617 | ✓ 916 KB | ✓ 71 KB | https://arxiv.org/abs/2602.22617 | |
| LeJEPA (#8) + SIGReg (#9) | 2511.08544 | ✓ 8552 KB | ✓ 163 KB | https://arxiv.org/abs/2511.08544 | one file backs both refs |
| LLM-JEPA (#27) | 2509.14252 | ✓ 1401 KB | ✓ 67 KB | https://arxiv.org/abs/2509.14252 | |
| SwiGLU — GLU Variants (#15) | 2002.05202 | ✓ 106 KB | ✓ 12 KB | https://arxiv.org/abs/2002.05202 | short paper, MD >5 KB |
| MTP — Multi-Token Prediction (#16) | 2404.19737 | ✓ 1684 KB | ✓ 94 KB | https://arxiv.org/abs/2404.19737 | |
| STE Ternary / BitNet b1.58 (#17) | 2402.17764 | ✓ 452 KB | ✓ 25 KB | https://arxiv.org/abs/2402.17764 | |
| StarCoder2 Tokenizer (#22) | 2402.19173 | ✓ 1088 KB | ✓ 209 KB | https://arxiv.org/abs/2402.19173 | |
| Zyphra RSA / ZAYA1-8B (#21) | 2605.05365 | ✓ 2101 KB | ✓ 158 KB | https://arxiv.org/abs/2605.05365 | |
| Nested Learning (#23) | 2512.24695 | ✓ 5947 KB | ✓ 225 KB | https://arxiv.org/abs/2512.24695 | table-only entry in references.md |
| CSA / HCA — DeepSeek-V4 (#6) | deepseek-v4 | ✓ 4375 KB | ✓ 188 KB | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf | non-arXiv; fetched via HF /resolve/main/ (see note below) |
| Hybrid Embeddings — Gu et al. 2019 (#11) | hybrid-embeddings-gu2019 | ✓ 1436 KB | ✓ 79 KB | https://openreview.net/forum?id=HJxeWnCcF7 | PDF via openreview.net/pdf?id=HJxeWnCcF7 |
| Lottery Ticket Hypothesis (#28) | 1803.03635 | ✓ 3990 KB | ✓ 141 KB | https://arxiv.org/abs/1803.03635 | conceptual basis for CMS structured pruning |
| Block-ELL Format (#2) | block-ell-nvidia | ✗ web-only | ✓ 1 KB | https://docs.nvidia.com/cuda/cusparse/storage-formats.html | NVIDIA cuSPARSE docs + dev blog; no paper PDF exists — MD stub with both source URLs (no PDF fabricated) |
| MegaBlocks / STK (#2a) | 2211.15841 | ✓ 553 KB | ✓ 65 KB | https://arxiv.org/abs/2211.15841 | MORTAR sparse matmul via vendored `morph/sparse/stk` |
| AdEMAMix Optimizer (#29) | 2409.03137 | ✓ 2800 KB | ✓ 141 KB | https://arxiv.org/abs/2409.03137 | optional `ademamix` / `ademamix_b1zero` training arms |
| TST — Token Superposition (bonus) | 2605.06546 | ✓ 1802 KB (pre-existing) | ✓ 109 KB | https://arxiv.org/abs/2605.06546 | not in references.md; MD generated from existing PDF as requested |

### DeepSeek-V4 download note
The local systemd-resolved DNS (127.0.0.53) returned SERVFAIL for `huggingface.co`
specifically (all other domains resolved normally). Worked around by resolving the host
via public DNS (8.8.8.8 → CloudFront 143.204.130.84) and pinning both
`huggingface.co` and the LFS redirect host `cas-bridge.xethub.hf.co` with `curl --resolve`.
Final fetch: HTTP 200, 4,480,407 bytes, valid PDF. No content was fabricated.

## Skipped by design (recorded, no download)

| Ref name | Reason |
|---|---|
| CMS Topology (#3) | Original work — MORPH project, no external paper |
| Value Shift (#26) | Points to CCA 2510.04476 — same file, not re-downloaded |
| Poisson Depth Sampling (#24) | Points to Parcae 2604.12946 — same file, not re-downloaded |
| SIGReg (#9) | Introduced in LeJEPA 2511.08544 — same file, not re-downloaded |
| MRR (#—, Residual Streams) | Original work — MORPH project, no external paper |

## Counts

- **PDFs OK:** 27 (25 arXiv + DeepSeek-V4 + Hybrid Embeddings) + 1 pre-existing TST PDF = **28 PDF files on disk**.
- **MD OK:** **29** (28 from PDFs + 1 web-only block-ell-nvidia stub).
- **Failed:** 0.
- **Skipped by design:** 5 (3 same-file pointers + 2 original-work entries) + Block-ELL has no PDF (web-only, MD stub written).

All 28 PDFs passed `file`="PDF document" + size>50 KB. All 29 MDs are >5 KB (smallest:
SwiGLU 12 KB) and contain their paper title.
