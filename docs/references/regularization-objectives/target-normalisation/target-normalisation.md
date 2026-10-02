# How latent-prediction methods normalise their targets: reading for MORPH

Read 2026-10-02. How seven self-supervised methods put their regression target into a
space where a distance means something, and what each one says the normalisation is for.
MORPH reason for the reading: the latent-selected loop regresses onto a LayerNormed
(no affine) pool of EMA prelude states, and the stage-1 arms choose between that and a
fixed per-coordinate standardisation (MORPH-side analysis:
[`.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md`](../../../../.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)).

## Cache and reading coverage

No PDFs saved. A research subagent read each source on 2026-10-02 (arXiv HTML, ar5iv, and
for I-JEPA the official code) and returned a quote and a location per claim. I did not
re-read these sources myself. Claims the subagent could NOT find in the source are marked
"not found".

## Summary table

| method | target | normalisation | stated reason |
| --- | --- | --- | --- |
| data2vec (2022) | mean of the top-K blocks of an EMA teacher | each block's output normalised BEFORE the average: parameter-less LayerNorm for NLP and vision, instance norm for speech | "prevent the model from collapsing into a constant representation" and "prevents layers with high norm to dominate the target features" |
| I-JEPA (2023) | EMA target encoder's patch features | `F.layer_norm` over the feature axis, no affine (code, `src/train.py`) | not stated in the paper text (not found) |
| BYOL (2020) | EMA target projection | L2-normalise prediction and target; loss `2 - 2 cos` | the loss is a cosine |
| VICReg (2022) | none (joint embedding) | hinge on the per-dimension std, `max(0, gamma - sqrt(Var + eps))`, gamma = 1, plus a covariance term | "explicitly avoids the collapse problem"; std, not variance, or the gradient vanishes near collapse |
| MAE (2022) | pixels | per-patch normalised pixels | "enhances the contrast locally"; 84.9 -> 85.4 fine-tune, 73.5 -> 73.9 linear probe (Table 1, read on ar5iv) |
| Timkey and van Schijndel (2021) | (similarity of LM states) | per-dimension standardisation with corpus statistics | rogue dimensions dominate cosine; see the massive-activations reading |
| Colton (2025), IJEPA feature normalisation | I-JEPA's target | argues the LayerNorm is harmful; DynTanh instead | "LN forces all features to have identical L2 norms ... preventing the model from prioritizing semantically rich regions" |

## Sources

- Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
  *data2vec*. ICML 2022. [arXiv 2202.03555](https://arxiv.org/abs/2202.03555), Sec. 3.3.
  A quantitative ablation of target normalisation on vs off: not found in the text read.
- Mahmoud Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture* (I-JEPA). CVPR 2023.
  [arXiv 2301.08243](https://arxiv.org/abs/2301.08243). Code:
  `facebookresearch/ijepa/src/train.py`, `forward_target()`:
  `h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim`.
- Adam Colton. *Elucidating the Role of Feature Normalization in IJEPA*. 2025.
  [arXiv 2508.02829](https://arxiv.org/abs/2508.02829). Replacing the target LayerNorm with
  DynTanh: ImageNet linear probe 38 % -> 42.7 % (ViT-S), NYU-Depth-V2 RMSE -0.08.
- Jean-Bastien Grill et al. *Bootstrap Your Own Latent* (BYOL). NeurIPS 2020.
  [arXiv 2006.07733](https://arxiv.org/abs/2006.07733), Sec. 3.1, Eq. 2.
- Adrien Bardes, Jean Ponce, Yann LeCun. *VICReg*. ICLR 2022.
  [arXiv 2105.04906](https://arxiv.org/abs/2105.04906), Sec. 4.1, Eq. 1.
- Kaiming He et al. *Masked Autoencoders Are Scalable Vision Learners* (MAE). CVPR 2022.
  [arXiv 2111.06377](https://arxiv.org/abs/2111.06377), Table 1 (the sub-table letter was
  read from ar5iv; check the camera-ready before you cite the letter).
- V-JEPA (Bardes et al. 2024, [arXiv 2404.08471](https://arxiv.org/abs/2404.08471)): no
  target normalisation stated in the paper text; the code was not checked. Do not cite it
  for a LayerNorm-on-targets claim.

## What this means for MORPH

1. Per-layer, parameter-less LayerNorm on the target is standard (data2vec, I-JEPA code).
   Its stated job is to stop one high-norm source from dominating and to stop collapse to
   a constant. It is a per-VECTOR normalisation: it does not remove a direction that every
   vector shares. A target whose every row carries the same few large channels stays
   dominated by them after the LayerNorm.
2. A per-COORDINATE normalisation with fixed statistics (Timkey and van Schijndel;
   all-but-the-top in the massive-activations reading) is the one that removes a shared
   direction and equalises the coordinates. MORPH's stage-1 build uses it
   (`tul.latent_pre_target_norm: standard`).
3. VICReg's lesson for MORPH's online floor (`fan_lsel_enc_gamma` on the online prelude's
   pooled span states): the floor is a hinge on the per-coordinate std at 0.1, so it is
   inactive on a coordinate whose std is already above 0.1, whatever the shared mode does.
