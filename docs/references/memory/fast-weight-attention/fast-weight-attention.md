# Fast Weight Attention for Continual Learning

- **Authors:** Yifan Zhang, Steve Ta, Jasper Zhang, Jichen Feng, Shuzhen Li, Yongxin Zhang, Yifeng Liu, Huizhuo Yuan, Mengdi Wang, Quanquan Gu, Andrew Chi-Chih Yao
- **Year:** 2026
- **arXiv:** [2608.27763](https://arxiv.org/abs/2608.27763)
- **Hugging Face:** https://huggingface.co/papers/2608.27763
- **PDF:** [fast-weight-attention.pdf](fast-weight-attention.pdf)
- **MORPH relevance:** Deep analysis of recurrent fast-weight memories and selective SSMs (such as GLA, DeltaNet, and Titans) as online learning rules under autoregressive semantics. Proposes normalized first-order update families (Falcon-1 scalar NLMS, Falcon-2 per-column, Falcon-3 mini-batch) and numerically stable positive-decay renormalization to prevent memory divergence and improve length extrapolation.

---

### Abstract

Recurrent fast-weight memories and selective state-space models compress an expanding context into a fixed-size recurrent state, making the state transition an online learning rule. We study this rule under read-after-write autoregressive semantics. For the prefix-prediction objective considered here, the local fast-memory example revealed at step $t$ is the prefix-aligned pair $(x_t, y_t) = (\phi(k_{t-1}), v_t)$. The common same-step association $(\phi(k_t), v_t)$ remains causal, but optimizes a different internal objective. We derive normalized first-order updates for squared-error regression and negative inner-product objectives. The regression family comprises Falcon-1 (a scalar NLMS update), Falcon-2 (its per-column extension), and Falcon-3 (a sliding-window mini-batch update); Falcon-1A/Falcon-2A/Falcon-3A are the corresponding inner-product variants. We provide recurrent, masked-parallel, and chunk-parallel forms, together with numerically stable positive-decay renormalization. Representative variants remain competitive in language modeling and improve length extrapolation on variable-digit addition. This framework separates temporal alignment, plasticity, forgetting, and bounded rehearsal in recurrent sequence models.
