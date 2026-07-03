# Gated Linear Attention Transformers with Hardware-Efficient Training

- **Authors:** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim
- **Year:** 2023 (ICML 2024)
- **Source:** https://arxiv.org/abs/2312.06635
- **PDF:** [2312.06635.pdf](2312.06635.pdf)
- **MORPH uses:** GLA retention branch (`morph/model/gla.py`) as the cross-iteration memory
  path, on by default (`retention: true` in `base.yaml`). Per-key-channel gated linear attention
  with optional carry of the final state `S_T` across core-loop iterations; output is gated and
  GroupNorm'd following the paper.

This is a short archive stub (full-text markdown extraction not available in this environment).
Read the PDF for the complete paper.
