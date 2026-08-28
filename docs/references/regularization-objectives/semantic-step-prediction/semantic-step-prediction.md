# Semantic Step Prediction: Multi-Step Latent Forecasting in LLM Reasoning Trajectories via Step Sampling

- **arXiv:** 2604.18464
- **Author:** Yidi Yuan
- **Code:** https://github.com/YYDreamzure/SSP/
- **Source:** Extracted from the arXiv HTML version (https://arxiv.org/html/2604.18464). This is a clean
  markdown summary/extraction, not a verbatim full-text copy; equation and table numbers are preserved
  as cited in the HTML.

> NOTE on lineage: This paper is the **step-boundary follow-up** to the original Semantic Tube Prediction (STP)
> work (Huang, LeCun, Balestriero 2026, arXiv:2602.22617 — the paper MORPH's `prediction.py` docstring cites).
> The original STP defines the geodesic-smoothness regularizer on random token triplets; THIS paper's central
> claim is that **where** you sample the STP loss (semantic step boundaries vs random token positions)
> dominates the geometric outcome.

---

## Abstract

STP regularizes LLM hidden-state trajectories toward locally-linear geodesics during fine-tuning. The key
contribution is applying STP at **semantic reasoning step boundaries** rather than random token positions,
yielding a **168x** improvement in latent prediction accuracy over a frozen baseline on ProcessBench
(3,400 samples) versus only **4x** for random-token sampling. Trajectories form smooth *curves* rather than
straight lines: a 3-layer MLP forecaster reduces prediction error 3–12x over the zero-parameter linear
extrapolation. Removing the language-modeling (NTP) loss makes trajectories ~2x more predictable but costs
generation accuracy — a generation-quality vs geometric-purity trade-off. Sampling position is identified as
the critical lever, and a **multi-step latent prediction MSE** is proposed as a new evaluation metric.

---

## (1) Multi-Step Latent Prediction MSE Metric

Zero-parameter linear extrapolation forecast over the step-boundary hidden-state trajectory
`z = (z_0, z_1, ..., z_K)` (each `z_k` = hidden state at a `<|step|>` delimiter position).

**Eq. 4 — linear predictor (m steps ahead):**

```
ẑ_{k+m} = z_k + m · (z_k − z_{k−1})
```

i.e. velocity `v = z_k − z_{k−1}`, and the m-step forecast is `z_k + m·v`.

**Eq. 5 — normalized MSE metric:**

```
MSE_m = (1/N) · Σ  ‖ẑ_{k+m} − z_{k+m}‖²  /  ‖z_{k+m}‖²
```

- Per-sample squared Euclidean error normalized by the squared norm of the **actual** target embedding,
  then averaged over all `(k, sample)` pairs in the eval set.
- `MSE ≈ 0` → perfect local linearity; `MSE ≈ 1` → prediction error equals signal magnitude (no better than
  predicting nothing).

---

## (2) Headline Results

### Table 2 — Multi-step latent MSE (Qwen2.5-Math-1.5B, ProcessBench, 3,400 samples)

| Model | Description | MSE₁ | Factor vs frozen (B1) |
|-------|-------------|------|-----------------------|
| B1 | Frozen baseline | 0.955 | 1× |
| C  | Random-token STP | 0.226 | **4×** |
| A  | Consecutive step-boundary STP | 0.006 | **168×** |

The ~40× gap between C and A "arises from **where** the loss is sampled, not **what** the loss computes."

### Table 5 — Decoding fidelity (m=1, 200 ProcessBench samples)

| Model | Top-1 token agreement | KL divergence | Step retrieval acc |
|-------|----------------------|---------------|--------------------|
| A (step-boundary STP) | **93.4%** | **0.019 nats** | 92.2% |
| B1 (frozen) | 19.1% | 4.581 nats | 70.2% |

Decoding the linearly-extrapolated latent through the LM head recovers the true next-step token 93.4% of the
time for model A — the geometric smoothness is *decodable*, not just a numeric artifact.

---

## (3) Training Recipe

- **Step delimiter token:** `<|step|>` inserted between reasoning steps (Eq. 2). The trajectory is read off
  the hidden states at these delimiter positions.
- **STP step-level loss (Eq. 3):**
  ```
  L_STP^step = 1/(K−1) · Σ_k [ 1 − cos(z_k − z_{k−1},  z_{k+1} − z_k) ]
  ```
  (geodesic colinearity of consecutive step-to-step increments).
- **Combined objective:** `L = L_NTP + β · L_STP^step`, with **β = 1**.
- **Sampling — "consecutive boundaries" vs "random boundary triplets":** Model A samples **consecutive**
  step-boundary triplets `(z_{k−1}, z_k, z_{k+1})`. Model C samples random TOKEN positions. The 168x vs 4x
  result is exactly this comparison.
- **Embed/head sharing across the boundary:** The HTML does NOT explicitly state a "share + never reinit
  embed/head" recipe for this paper. (That guidance came from the Nous TST paper, arXiv:2605.06546 — do not
  attribute it to 2604.18464.) This paper fine-tunes via LoRA on a fixed pretrained base, so the base
  embedding/head are inherently shared and unmodified.
- **Adapter:** LoRA rank 16 on q,k,v,o projections (~4.4M trainable params on the 1.5B base).
- **Equal-FLOPs:** All compared models (A/B1/B2/C) are trained for the same optimizer budget
  (3 epochs, ~1,150 optimizer steps each) on the same data, so differences are attributable to sampling
  position, not compute.
- **Data:** 6,132 MATH competition problems, split at paragraph boundaries with `<|step|>` delimiters.

### Table 4 — GSM8K accuracy (downstream task preservation)

| Model | GSM8K accuracy |
|-------|----------------|
| B1 (frozen) | 68.8% |
| A (step-boundary STP) | **73.0%** |
| B2 (vanilla LM fine-tune) | **73.0%** |

Model A matches vanilla fine-tuning (73.0%) — the geometric regularizer does NOT degrade task accuracy.

---

## (4) Explicit Negative Results

- **Smoothness ≠ correctness (Table 7, Appendix A.2):** Using trajectory smoothness to detect *incorrect*
  reasoning steps gives binary AUC ≈ 0.5 (indistinguishable from random). "Geometric smoothness does not
  encode step correctness" — it captures organized/predictable reasoning *flow*, independent of whether the
  answer is right.
- **Removing L_NTP trade-off:** Dropping the language-modeling loss yields ~2× better MLP latent prediction
  but costs **−3.7 pp GSM8K accuracy** — generation quality vs geometric purity are competing objectives.
- **Random-token STP harms accuracy:** Model C (random-token STP) is the only variant showing an accuracy
  *decrease* vs the frozen baseline.
- **Trajectories are curves, not lines:** A 3-layer MLP forecaster beats the zero-parameter linear
  extrapolation by 3–12×, i.e. the latent path is smooth but nonlinear.
- **r>0.5 recovery:** The HTML did NOT surface an explicit "r>0.5 masking-ratio recovery fails" result for
  THIS paper. That r-range/recovery guidance traces to the TST line (arXiv:2605.06546), not 2604.18464.
  (Flagged as unverified for this paper — see caveats below.)

---

## Verification caveats

- Numbers above were extracted by a summarization model over the arXiv HTML, not transcribed line-by-line
  from the source LaTeX. The MSE formula, the 0.955 / 0.226 / 0.006 (168x / 4x) headline, the 93.4% / 0.019-nat
  decoding fidelity, β=1, and 73.0% GSM8K were each returned consistently across two independent fetches.
- Two items the task asked about were NOT found in this paper's HTML and appear to belong to the TST paper
  (2605.06546) instead: (a) the explicit "share + never reinit embed/head across the boundary" rule, and
  (b) the "r>0.5 recovery fails" masking-ratio result. Do not attribute these to 2604.18464.
