> Reading filed 2026-09-14 for docs/tul-code-spec.md. Produced by a fetch-and-summarise pass over the
> arXiv HTML/PDF, not a hand transcription; items the reader flagged as unverified are marked in the text.

# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

Chen, Marti Monso, Du, Simchowitz, Tedrake, Sitzmann. NeurIPS 2024. arXiv **2407.01392** (verified, v4). Read from the arXiv PDF, full text plus appendices.

## 1. Core idea and training objective

Sec 3.2: "Diffusion Forcing (DF) is a framework for training and sampling arbitrary sequence lengths of noisy tokens $(\mathbf{x}_t^{k_t})_{1\le t\le T}$, where critically, the noise level $k_t$ of each token can vary by time step." Causal instantiation = **Causal Diffusion Forcing (CDF)**, backbone an RNN: $\mathbf{z}_t \sim p_\theta(\mathbf{z}_t \mid \mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)$. At $k_t=0$ this is the Bayes-filter posterior update; at $k_t=K$ it is the Bayes-filter *prior* $p_\theta(\mathbf{z}_t\mid\mathbf{z}_{t-1})$ — no new information enters.

**Loss (Eq. 3.1):**
$$\mathbb{E}_{k_t,\mathbf{x}_t,\epsilon_t,\ \mathbf{z}_t\sim p_\theta(\mathbf{z}_t\mid\mathbf{z}_{t-1},\mathbf{x}_t^{k_t},k_t)} \sum_{t=1}^{T}\Big[\lVert \epsilon_t - \epsilon_\theta(\mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)\rVert^2\Big]$$
$k_{1:T}$ sampled **uniformly from $[K]^T$**, every token drawing an i.i.d. uniform level independent of the others (Algorithm 1 line 4: "Sample independent noise level $k_t \in \{0,1,\dots,K\}$"). $\epsilon_t\sim\mathcal N(0,\sigma_{k_t}^2 I)$ from the forward process (Eq. 2.1). The dynamics model plus observation model $p_\theta(\mathbf{x}_t^0\mid\mathbf{z}_t)$ form an ordinary conditional-diffusion unit, so training is the usual $\epsilon$-prediction MSE, per-token-noised.

## 2. Noise as partial masking

Sec 3.1: teacher forcing masks *along the time axis*; full-sequence diffusion masks *along the noise axis* (after $K$ steps $\mathbf{x}_{1:T}^K$ is "approximately pure white noise without information about the original data"). $(\mathbf{x}_t^{k_t})$ gives "the degree of partial masking applied to each token through noising." Fig. 2: **teacher forcing** = past at $k=0$, future at $k=K$; **full-seq diffusion** = every token at the same $k$; **Diffusion Forcing** = arbitrary independent $k_t$, the strict superset. "We remark that a special case of 'all sequences of noise levels' are those for which either $k_t=0$ or $k_t=K$; thus, one can mask out *any* prior token and DF will learn to sample from the correct conditional distribution" (p.6).

## 3. Sampling

**Algorithm 2** (DF Sampling with Guidance): initialize all tokens to white noise; walk a scheduling matrix $\mathcal K\in[K]^{M\times T}$ row by row ($m=M-1,\dots,0$), column by column ($t=1,\dots,T$), updating $\mathbf{z}_t$ then denoising $\mathbf{x}_t$ one step toward $\mathcal K_{m,t}$; after each row, add guidance $\nabla_\mathbf{x}\log c(\mathbf{x}_{1:H})$. Columns index time, rows the pyramid step; the pyramid schedule (Eq. D.1) holds far-future tokens at higher noise than near-future ones — "the tokens in the far future are kept at higher noise level than near future."

**Stabilization** (Sec 3.3, App B.4): instead of feeding the previous step's clean latent forward, DF keeps a latent tied to a slightly noisy version of the just-generated token, $0<k_{\text{small}}\ll K$ — since training samples exactly this input, it stays in-distribution (DART/DAgger-style noise injection). **Zig-zag scheduling** (Sec 3.3): from $[\mathbf{x}_1^K,\mathbf{x}_2^K,\mathbf{x}_3^K]^\top$, fully denoise token 1 while leaving later ones partly noised, e.g. $[\mathbf{x}_1^0,\mathbf{x}_2^{K/2},\mathbf{x}_3^K]^\top$ — "encodes the immediate future as more certain than the far future."

**Long-horizon guidance** (Sec 3.3): a gradient on a still-noisy future can propagate backward through the causal recursion to tokens being denoised, respecting causality — full-sequence diffusion cannot do this cleanly (App B.6: "conditioning by replacement" is inconsistent, wastes compute).

**Monte Carlo Guidance (MCG)** (Sec 3.4, App B.5): "Instead of drawing a single trajectory sample to calculate this guidance gradient, we can draw multiple samples of the future and average their guidance gradients," estimating cost-to-go $c(\mathbf{x}_t^k)=\mathbb E[\sum_{t'>t}\mathbf r_{t'}(\mathbf x_{t'}^{k_{t'}})\mid\mathbf x_t^k]$ (Eq. B.1). Full-sequence diffusion has no randomness source to roll futures from at fixed noise, so MCG "is not feasible" there.

## 4. Stabilization

Sec 4.1: "auto-regressive architectures are known to diverge, especially when sampling past the training horizon. In contrast, Diffusion Forcing can stably roll out long sequences even beyond the training sequence length by updating the latents using the previous latent associated with slightly 'noisy tokens.'" Baselines "diverge quickly," with "frame-to-frame discontinuity," while DF stays a "consistent 3D environment" to 1000 frames. Measured extent (App E.3): trained on 72 (Minecraft) / 36 (DMLab) frames, DF rolls out to 180 ("2x-5x times longer than it's trained on") and was tested to 2000 "without seeing the model blowing up." No isolated metric is tied to the trick alone — the evidence is rollout length vs. baselines diverging within the training horizon.

## 5. Theory

**Theorem 3.1 (informal):** "The Diffusion Forcing training procedure (Algorithm 1) optimizes a reweighting of an Evidence Lower Bound (ELBO) on the expected log-likelihoods $\ln p_\theta((\mathbf x_t^{k_t})_{1\le t\le T})$, where the expectation is averaged over noise levels $k_{1:T}\sim[K]^T$... Moreover, under appropriate conditions, optimizing (3.1) also maximizes a lower bound on the likelihood for *all sequences of noise levels, simultaneously*." Appendix A's derivation ends at a per-token bound with weight $\frac{j+1}{K+1}$ on the KL term at intermediate level $j$ — one term per token, weighted by how much of the noise ladder it still has to climb.

"Decaying memory" is not the paper's language. The theorem licenses one model simultaneously valid as an ELBO for *every* assignment of noise to positions, past or future. Combined with App B.7's Bayes-filter reading — "by varying $k$ between $K$ and $0$, the same neural network can parameterize everything between prior and posterior" — a noised past token contributes less determinate information than a clean one. A monotone recency-decay law over many past tokens is not stated or quantified; that is an extrapolation, not a proven claim.

## 6. Architecture

**RNN (main experiments, Sec 3.2, App D.2):** "For simplicity, we focus on a minimal implementation with a vanilla Recurrent Neural Network (RNN)." A U-Net (video) or ResMLP (vectors) feeds a GRU; observation head is a 1-layer ResNet+conv. $k_t$ conditions every step, like a diffusion timestep embedding. Sizes: Minecraft 36M params, DMLab 24M, maze planning 4.33M.

**Transformer (App B.1, described not benchmarked):** a transformer trained with independent per-token noise, optionally *without* a causal mask: "By labeling the future as full white noise, there is no information leaked into the past tokens... By labeling the future tokens as noisy, a slight amount of information about the future is provided to the prediction of past tokens." Called "already verified" effective but "beyond the scope of this paper" — no sizes or numbers given.

## 7. Results relevant to language

**Text was not tested.** Nearest domains, with numbers:

- **Video** (Minecraft/DMLab): qualitative only (Fig. 3) — consistent past 1000 frames where baselines diverge; no FVD reported.
- **Planning** (D4RL Maze2D, Table 1, reward): single-task average Diffuser\* 119.5, Ours w/o MCG 129.67, **Ours 141.7**; multi-task average Diffuser\* 129.4, **Ours 146.2**.
- **Imitation learning** (Franka fruit-swap, needs memory): "DF achieves 80% success rate while diffusion policy [10]... fails." Under occluded-camera corruption ($k>0$), DF drops "by 4% to 76%" vs a baseline at 48%.
- **Time series** (CRPS$_\text{sum}$, Table 2, lower better): DF beats TimeGrad/Transformer-MAF on 5 of 6 sets, e.g. Exchange **0.003±0.001** vs TimeGrad 0.006±0.001; loses only to ScoreGrad, on Wikipedia. "Outperforms all prior methods except for [ScoreGrad]... time series is not the core application."

**Compute/steps-per-token as a quality dial:** not ablated as a curve. App D.6: $K=1000$ training steps, DDIM "100 for video prediction and 50 for non-video domains" — a speed choice, not a quality sweep.

## 8. Limitations and follow-up

**Limitations (Sec 5, full text, verbatim):** "Our current causal implementation is based on an RNN. Applications to higher-resolution video or more complex distributions likely require large transformer models following instructions in Appendix B.1. We do not investigate the scaling behavior of Diffusion Forcing to internet-scale datasets and tasks."

**Follow-ups applying per-token/per-chunk noise at scale:**
- **History-Guided Video Diffusion / DFoT** (arXiv 2502.06764, ICML 2025) — realizes the App B.1 transformer extension with per-frame noise, adds "History Guidance" over variable-length noisy history.
- **MAGI-1** (arXiv 2505.13211) — 24B-parameter autoregressive video DiT denoising 24-frame chunks with per-chunk noise rising monotonically over time.
- **CausVid** (arXiv 2412.07772) — distills a bidirectional video teacher into a causal student; one leg trains it "under diffusion forcing on real data"; Self Forcing later swaps that for self-rollout.

## How this gives a decaying memory tape, in three sentences

Per-token noise $k_t$ *is* the token's masking degree, so giving older span-latents progressively higher $k$ moves them from their exact posterior toward the content-free prior transition (App B.7) — graded fuzziness, not a binary keep/drop. Theorem 3.1 bounds the likelihood simultaneously for every assignment of noise to positions, the property needed for one model that works whether history is clean or only the newest span is, no separate head per schedule. The paper gives the mechanism and the guarantee that any noise pattern is valid to train on; it does not give the decay law — how fast $k$ should grow with age is a design choice left to the builder.

## Open questions the paper does not answer

- No discrete-token variant: Eq. 3.1 is Gaussian MSE only.
- No rule mapping a span's age to $k_t$ over a long past tape — shown schedules govern the current window's future only.
- No ablation of $k_{\text{small}}$ against rollout length, and no transformer-vs-RNN benchmark — "already verified" is asserted, not shown.
- No test of how many simultaneously-noisy old positions one causal step can condition on before quality degrades — all experiments use short sequences (≤72 frames, small mazes).
