# Blocked-ELL (Blocked-Ellpack) Sparse Storage Format

- **Source:** NVIDIA cuSPARSE Library + NVIDIA Developer Blog (web-only documentation; no paper PDF exists)
- **Year:** 2021+ (cuSPARSE), blog 2023
- **Source URLs:**
  - cuSPARSE Blocked-ELL storage format documentation: https://docs.nvidia.com/cuda/cusparse/storage-formats.html
  - NVIDIA Developer Blog — "Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores": https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/
- **MORPH uses:** The Blocked-Ellpack (Blocked-ELL) storage format where nonzero weight sub-matrices are stored in fixed-size dense tiles (32x32 on SM120/5090) with a companion column-index array. MORPH's CMS pruning selects which tiles survive; surviving tiles are stored in Block-ELL for hardware-efficient SpMM using Tensor Cores, achieving near-linear speedup proportional to sparsity.

---

This reference is web-only. NVIDIA does not publish a downloadable paper PDF for the
Blocked-ELL format — the canonical documentation lives in the cuSPARSE storage-formats
page and the accompanying NVIDIA Developer Blog post (URLs above). No PDF was fabricated.
