# Efficient latent exploration: literature to seed an LXTUL redesign

Author: LitScout (Claude Opus 5.5 research subagent), 2026-09-29.
vlt thread: `tul-span-jepa` (progress notes 680 onward, author `LitScout`).

## How to read this report

**Verification.** Every entry below is VERIFIED. For each ID I pulled the arXiv export API record
(`export.arxiv.org/api/query?id_list=...`), which is the same record the abs page shows, and
checked title, first author, author count and submission date against what I cite. I did not
invent or guess any ID. Papers I could not confirm are not in the report.

**Depth of reading.** Each entry says one of these:

- `method`: I fetched the full-text HTML (or PDF) and read the method section through a
  summarizing fetch tool. The numbers quoted come from that pass. A summarizer can drop or bend
  details, so treat exact constants as "check before you build".
- `abstract`: I read the abstract only. Any mechanism detail beyond the abstract comes from my
  prior knowledge of the paper and is marked "(prior knowledge)".

**Mapping terms.** I use these words with one meaning each:

- *hypothesis*: one of the K x M loop outputs for one slot (K code rollouts, M fan cells).
- *coda read*: one pass of the 4 coda layers plus the vocab cross-entropy over every token of
  the row. This is the unit that the 2026-09-29 profile says dominates LX-Fan+WTA cost.
- *scorer*: any function that ranks hypotheses for a slot without a coda read.
- *assignment*: the choice of which hypothesis gets the training signal (the WTA winner).

## The cost frame (read this before the buckets)

From the orchestrator's profile note (vlt `tul-span-jepa` #678 and #679, ev01, steps 8-12):
LX-Fan+WTA takes 2.28 s GPU per step against 0.69 s for the plain looped model. Forward is
0.95 s: coda 6 calls x 78 ms (0.47 s), fan non-coda vocab CE and writes 0.225 s, slot loop
0.19 s, rest 0.08 s. Backward plus optimizer is 1.33 s. There are 24 coda units per step
(4 mixture, 16 no-grad WTA picks, 4 grad).

An inference I made from those numbers, NOT measured: if the coda's share of the backward is
roughly its share of the grad-carrying reads, the non-coda part of the step (slot loop over
K x M streams, the fan's per-cell vocab CE, writes, prelude) is on the order of 1.3-1.5 s, which
alone is about 2x the plain step. The note does not split the 1.33 s backward by component, so
this is a guess. If it holds, a "read tokens once" coda is necessary but NOT sufficient. The
redesign must also (a) keep the scorer out of vocab space and (b) not run the full K x M slot
loop with gradient through every stream. Every shortlist item below is costed against this.
A cheap check: re-run `scratchpad/prof/prof_driver.py` with `record_function` labels inside the
backward (autograd profiler already attributes backward kernels to forward scopes when
`record_shapes=True` and `with_stack=True`).

The papers repeat two lessons that matter more than any single mechanism:

1. **Cheap assignment exists.** A WTA winner does not need a decode per hypothesis if the
   winner can be read off a cheap target derived from the true output (Cho et al. 2019), or
   predicted by a score head (rMCL), or ranked contrastively in latent space (LatentRM). The
   risk is misprice, and nobody in this literature measured misprice for a looped LM.
2. **Point regression in latent space collapses on text; distributional or multi-hypothesis
   objectives do not.** JEPA Paradox (2607.23531), ATF (2609.33271), and WTA-as-quantization
   (2406.04706) agree. This fits MORPH's own NextLat failure. It argues FOR keeping multiple
   hypotheses, and for scoring them with a proper scoring rule or a contrastive rule, never MSE
   to a single target.

---

## Bucket 1. Efficient multiple-hypothesis training

### 1.1 Mixture Content Selection for Diverse Sequence Generation
Cho et al., 2019. arXiv:1909.01953. VERIFIED. Read: method.
- Mechanism: a Selector with K expert embeddings predicts K "focus" masks over the source in one
  pass. Stochastic hard-EM picks the best expert per example by the Selector's OWN likelihood of
  an oracle focus mask (derived from source-target token overlap), not by decoding. The
  generator is trained once per example, on the oracle focus.
- Compute: Selector cost is one pass for all K; generator cost is 1 decode. The paper reports
  3.7x faster training than a K-decoder mixture, whose cost grows linearly in K.
- For LXTUL: this is the template for "span-level assignment, one token read". Try: the WTA
  winner among the K x M hypotheses is the one nearest (in d-dim latent space) to a cheap
  encoding of the TRUE next span; decode only that hypothesis, with gradient. Replaces: the 16
  no-grad pick reads and 3 of the 4 grad reads. Risk: Cho feeds the ORACLE focus to the
  generator; in LXTUL that is the teacher-forcing bypass that already failed. So feed the
  model's own winning hypothesis, and use the true span only to choose among hypotheses.

### 1.2 Resilient Multiple Choice Learning (rMCL)
Letzelter et al., 2023. arXiv:2311.01052. VERIFIED. Read: method.
- Mechanism: K hypothesis heads plus K score heads on a shared trunk. Score heads are trained by
  binary cross-entropy: winners are positives, non-winners negatives. At test time the scores
  give each hypothesis a probability mass, so the set becomes a density estimate.
- Compute: one pass for all K hypotheses and scores; score heads are about 4k params on a 1.5M
  model. Memory trick: update only a subset of negative scores per step.
- For LXTUL: a per-hypothesis score head on the slot's loop output (no future access) that
  predicts "this hypothesis wins". Replaces: the per-token Bayes read at eval (which needs all K
  coda reads) with the top-scored hypothesis or a score-weighted mix of the top 2. Risk: the
  winner labels themselves are what the cheap assignment (1.1) produces, so a biased assignment
  trains a biased score head. Low constraint risk.

### 1.3 Winner-takes-all learners are geometry-aware conditional density estimators
Letzelter et al., 2024. arXiv:2406.04706. VERIFIED. Read: abstract.
- Mechanism: trained WTA hypotheses are a centroidal Voronoi tessellation of the conditional
  distribution. Voronoi cell volume plus score gives a density estimate without changing
  training.
- Compute: no training change; density read is geometric.
- For LXTUL: theory support for doing WTA in the LATENT space (distance to a span encoding)
  rather than in token CE space: the hypotheses then quantize the distribution of next-span
  encodings, which is the multi-modal object JEPA regression collapses (see 4.3). Risk: the
  quantization is only as good as the span encoding; a live-front encoder can drift (MORPH
  already saw a frozen encoder on a live front collapse).

### 1.4 Annealed Multiple Choice Learning (aMCL)
Perera et al., 2024. arXiv:2407.15580. VERIFIED. Read: abstract (Boltzmann softmin assignment
with a temperature schedule is prior knowledge, confirmed by LoRA-MCL's description of it).
- Mechanism: soft assignment weights `q_k ∝ exp(-loss_k / T)`, T annealed from high (uniform,
  every hypothesis learns) to 0 (hard WTA). Avoids WTA's greedy bad local minima and dead heads.
- Compute: same as the underlying loss per hypothesis; the win is in optimization, not cost.
- For LXTUL: LX's exact mixture IS the T=1 member of this family (posterior weights
  `softmax(-CE_k)`), and LX hard credit is T≈0 with eps. aMCL says anneal between them. Cheap
  only if the per-hypothesis loss is cheap, so pair it with a latent-space loss (1.1, 4.x), not
  coda CE. Risk: a temperature schedule is a curriculum over assignment, not over depth, so it
  does not hit the no-depth-curriculum rule; say so explicitly if proposed.

### 1.5 Multiple Choice Learning of Low-Rank Adapters for Language Modeling (LoRA-MCL)
Letzelter et al., 2025. arXiv:2507.10419. VERIFIED. Read: method.
- Mechanism: K LoRA adapters on a frozen LM, per-SEQUENCE WTA with relaxed (eps) or annealed
  weights. K hypotheses run by duplicating the batch K times with grouped convolutions for the
  adapters.
- Compute: about K x the forward cost. They match budgets by rank (r for MCL vs K*r for MLE),
  not by FLOPs.
- For LXTUL: evidence that per-sequence (per-span) WTA works for text and that eps relaxation
  and annealing are the standard anti-collapse knobs (LX hard credit already uses eps 0.05).
  It does NOT solve cost: it pays K decodes, exactly LXTUL's problem. Cite as the thing to beat.

### 1.6 Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses
Rupprecht et al., 2016. arXiv:1612.00197. VERIFIED. Read: abstract (relaxed WTA with eps is
prior knowledge).
- Mechanism: relaxed WTA, winner weight 1-eps, others eps/(K-1). The origin of LX hard credit.
- For LXTUL: already in the tree. Listed so the lineage is clear.

### 1.7 Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles (sMCL)
Lee et al., 2016. arXiv:1606.07839. VERIFIED. Read: abstract.
- Mechanism: WTA over an ensemble, SGD on the winner's loss only. Parameter-free.
- For LXTUL: background only.

### 1.8 Mixture Models for Diverse Machine Translation: Tricks of the Trade
Shen et al., 2019. arXiv:1902.07816. VERIFIED. Read: abstract (full-text fetch failed on the
PDF encoding). Shared decoder with a per-component start token is prior knowledge.
- Mechanism: K-component mixture for MT trained by EM. Degeneracies: only one component trains,
  or the latent is ignored. Finding stated in the abstract: disabling dropout noise in the
  responsibility computation is critical. Hard vs soft EM, online vs offline assignment all
  change results.
- Compute: responsibilities need K decoder passes (the same cost LXTUL pays).
- For LXTUL: two checks worth doing now. (a) Are the 16 no-grad WTA pick reads run with dropout
  or token-state dropout (0.15 on the coda input) active? Shen says that noise in the
  responsibility step breaks mixture training. (b) The "rollouts are an ensemble" result
  (selection ceiling ~0.0006 nats) is the "latent is ignored" degeneracy by another name.

### 1.9 Importance Weighted Autoencoders (IWAE)
Burda et al., 2015. arXiv:1509.00519. VERIFIED. Read: abstract.
- Mechanism: K-sample importance-weighted ELBO; tighter bound as K grows.
- For LXTUL: LX's exact mixture with uniform prior is an IWAE-style bound with the prior as
  proposal (`iw_span_bound` in the tree). The IWAE lesson that matters here: the gradient of the
  inference side gets WORSE as K grows (signal-to-noise falls). Proposing from a posterior
  (conditioned on the true span, cheap) instead of the prior is the classical fix, and it is the
  same idea as 1.1.

### 1.10 Variational inference for Monte Carlo objectives (VIMCO)
Mnih and Rezende, 2016. arXiv:1602.06725. VERIFIED. Read: abstract.
- Mechanism: unbiased multi-sample gradient for discrete latents with a leave-one-out baseline
  per sample.
- For LXTUL: if the redesign samples which hypothesis to decode (one discrete choice per slot),
  the selector gradient needs a baseline. VIMCO's leave-one-out baseline needs every sample's
  score, which is fine if the scores are latent-space (cheap) and only the decode is sampled.

### 1.11 Estimating Gradients for Discrete Random Variables by Sampling without Replacement
Kool et al., 2020. arXiv:2002.06043. VERIFIED. Read: abstract.
- Mechanism: unbiased estimator from k samples drawn without replacement, with a built-in
  control variate at no extra model evaluations. Strong in both high- and low-entropy regimes.
- Compute: k evaluations of the downstream loss.
- For LXTUL: with k=2 decoded hypotheses per slot (sampled without replacement from a score
  head), this gives an unbiased, low-variance gradient for the selector at 2 coda reads instead
  of K x M. Risk: 2 grad reads, so cost is about 2/8 of today's grad reads, not 1.

### 1.12 Reweighted Wake-Sleep
Bornschein and Bengio, 2014. arXiv:1406.2751. VERIFIED. Read: abstract.
- Mechanism: train an inference network q(z|x) by importance-weighted wake and sleep phases.
- For LXTUL: the "posterior" scorer in 1.1 is an inference network q(k | context, true span).
  RWS is the principled way to train it from occasional exact responsibilities (e.g. one full
  K-read step in N), so the cheap assignment is calibrated against the expensive one.

### 1.13 Categorical Reparameterization with Gumbel-Softmax
Jang et al., 2016. arXiv:1611.01144. VERIFIED. Read: abstract.
- For LXTUL: the relaxed choice of hypothesis. Warning from 2.9 (E2E planner): straight-through
  over a hard choice did not work there; the soft probability-weighted mix did.

---

## Bucket 2. Latent reasoning with exploration, search and credit

### 2.1 Parallel Test-Time Scaling for Latent Reasoning Models (LatentTTS, LatentRM)
You et al., 2025. arXiv:2510.07745. VERIFIED. Read: method.
- Mechanism: stochastic latent rollouts by MC dropout or additive Gaussian noise; a Latent
  Reward Model (backbone plus a scalar head reading the latent trajectory, no token decode)
  trained with a STEP-WISE contrastive objective: at each step, softmax over the N candidates'
  scores, NLL on the candidate with the best Monte Carlo correctness. Labels come from M
  rollouts per thought. Best-of-N and beam search use cumulative scores.
- Compute: scoring is one head per latent; label making is expensive (N x M rollouts), paid once.
- Findings: MC dropout gave higher coverage than Gaussian noise at N = 4..32; contrastive
  supervision beat BCE by about 2 points.
- For LXTUL: a latent-only scorer trained contrastively across the K x M hypotheses of the same
  slot, labelled by occasional exact coda reads. Replaces the per-step WTA pick reads. Also: MC
  dropout held fixed across the loop (see 2.4) is a candidate exploration source instead of
  fixed simplex codes. Risk: labels need decodes; amortize them (1 step in N).

### 2.2 Latent Thought Credit: Multi-Answer Credit Assignment for Latent Reasoning (LTC)
Zhao et al., 2026. arXiv:2608.01593. VERIFIED. Read: method.
- Mechanism: sample K latent thoughts, and for each FIXED thought sample M answers; thought
  advantage is the mean answer reward, normalized across thoughts. An advantage-weighted
  thought-matching term pulls the policy toward high-credit thoughts.
- Compute: K x M answer decodes. At a fixed budget of 8, (K, M) = (2, 4) beat (4, 2) by 1.97
  points. Multi-answer averaging cut pairwise credit-ordering error from 0.384 (M=1) to 0.262
  (M=8).
- For LXTUL: the transferable fact is "at a fixed decode budget, FEWER hypotheses with less noisy
  credit beat more hypotheses with noisier credit". For LXTUL the next span is deterministic
  given data, so the "answer noise" is dropout and token-state dropout noise in the coda read.
  Supports K=2 decoded hypotheses (1.11) over K=4. Risk: thought-matching is a regression onto
  selected thoughts; in LXTUL that is a regression onto a slot state, which MORPH's rules warn
  against. Keep only the credit part.

### 2.3 Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning
Deng et al., 2026. arXiv:2604.27998. VERIFIED. Read: method.
- Mechanism: three failure modes of RL in latent space: no latent manifold (rollouts go off
  manifold), exploration-optimization misalignment (two-sided Gumbel noise flips update signs),
  and **latent mixture non-closure**: reinforcing several correct latent paths at once averages
  them into a state that supports no correct continuation. Fixes: zero advantage for invalid
  samples, one-sided noise with a conditional straight-through estimator, and updating only the
  first latent of the single best correct path.
- For LXTUL: the non-closure result is the strongest warning against "superpose the K hypotheses
  and read once" (3.x, 6.x). It also explains why a soft mixture credit over nearly equal
  rollouts gives an ensemble, not specialists. Risk flag for any averaging design.

### 2.4 Dropout-GRPO: Variational Stochasticity for Continuous Latent Reasoning
Jung, 2026. arXiv:2606.10184. VERIFIED. Read: abstract.
- Mechanism: one Bernoulli dropout mask per rollout, held constant across all latent recurrence
  steps and replayed in the update. Interpreted as a posterior sample over weights.
- For LXTUL: a parameter-free alternative to the fixed simplex codes: rollout k = loop with mask
  k held over all passes. It is still exploration inside the loop, so it does not by itself cut
  coda reads. Constraint: no learned sigma involved (good). Evidence is small (Coconut GSM8K
  27.3 to 29.0).

### 2.5 GTS: Inference-Time Scaling of Latent Reasoning with a Learnable Gaussian Thought Sampler
Wang et al., 2026. arXiv:2602.14077. VERIFIED. Read: abstract.
- Mechanism: a small module outputs a context-conditioned Gaussian over latent perturbations,
  trained with GRPO on a frozen backbone.
- For LXTUL: flagged. This is a LEARNED noise scale, which MORPH measured to die under the
  likelihood. GTS trains it with RL reward, not likelihood, on a frozen backbone, which may be
  why it survives. Do not port as-is.

### 2.6 Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge
Tang et al., 2026. arXiv:2601.08808. VERIFIED. Read: method.
- Mechanism: at each thinking step sample K tokens, average their one-hot vectors, embed as one
  "multiplex" token. Log-prob is the sum of the K samples' log-probs, so GRPO applies directly.
- Compute: one forward per step regardless of K. K=3 is usually enough.
- For LXTUL: the "branch then merge before the expensive part" pattern. The merge is over
  discrete vocabulary samples, which keeps the merged input inside the embedding manifold
  (unlike an average of free latents, see 2.3). Analog: hypotheses as discrete concept codes
  (5.3), merged into one cell before the coda. Risk: the merge is an average, so the non-closure
  risk is present but reduced.

### 2.7 MUX: Continuous Reasoning via Multiplexed Tokens
Suleymanzade et al., 2026. arXiv:2607.18264. VERIFIED. Read: method.
- Mechanism: each latent token is supervised to match a position-weighted mixture of the one-hot
  vectors of a whole reasoning span (`sum_j alpha_j onehot(r_j)`, geometric or rotary weights),
  by KL through a linear-softmax probe. Weights are chosen so the span is exactly recoverable
  ("lossless multiplexing").
- Compute: one vocab projection per latent token, only as supervision.
- For LXTUL: MORPH already had an order-free MUX bag target. MUX's positional weights make the
  span target injective. As a scorer it is cheap in coda terms (one vocab projection per
  hypothesis per slot, about KM/20 per token, i.e. 0.8 at KM=16) but that is roughly the same
  order as the fan's current per-cell vocab CE (0.225 s forward). So it is NOT cheap enough to
  score all 16 hypotheses; it could score a top-2 shortlist.

### 2.8 Soft Tokens, Hard Truths
Butt et al., 2025. arXiv:2509.19170. VERIFIED. Read: abstract.
- Mechanism: RL of continuous CoT with soft tokens (mixtures) plus noise on the input embedding
  for exploration; minimal overhead; train soft, deploy hard is best.
- For LXTUL: background for noise-as-exploration; not a cost lever.

### 2.9 End-to-end Planner Training for Language Modeling
Cornille et al., 2024. arXiv:2410.12492. VERIFIED. Read: method.
- Mechanism: a planner predicts a distribution over 1024 k-means "writing actions" (clusters of
  next-sentence embeddings). The LM is conditioned on the probability-weighted average of action
  embeddings, added inside every layer after attention. Straight-through over the argmax action
  was ineffective; the soft mix gave exact gradients and worked.
- Compute: one LM pass; action mixing is free.
- Findings: small perplexity gains (0.3 GPT-2 small, 0.08 OLMo-1B); oracle actions at train
  hurt perplexity by exposure bias.
- For LXTUL: direct precedent for "sentence-level discrete hypotheses, one token read, soft
  mixture". The small gains are a realistic prior for what span-level planning buys on web text.
  Oracle-action exposure bias is MORPH's teacher-forcing bypass under another name.

### 2.10 Learning to Plan Long-Term for Language Modeling
Mai et al., 2024. arXiv:2409.00070. VERIFIED. Read: method.
- Mechanism: sample K action sequences (T steps ahead) from a MuZero-style planner; a
  PathTransformer embeds each path and a SampleTransformer aggregates the K paths nonlinearly
  into one conditioning vector for the LM.
- Compute: planner cost linear in K; the LM still reads once. Gains grow with K up to 50.
- For LXTUL: the one-read aggregation over K sampled plans exists and helps next-token accuracy.
  Analog: aggregate the K x M hypothesis cells with a small set transformer before the coda,
  then one coda read. Risk: non-closure (2.3) and MORPH's "a1: four copies of one state fail".
  The SampleTransformer is nonlinear, which is not a plain average.

### 2.11 Latent Thinking Optimization (LTO)
Du et al., 2025. arXiv:2509.26314. VERIFIED. Read: abstract.
- Mechanism: on Huginn-3.5B (a recurrent-depth model), a classifier on the latent thoughts
  predicts answer correctness; used as a latent reward model to reweight latent thinking.
- For LXTUL: evidence that a looped model's latent states carry a readable quality signal that
  a cheap classifier can extract. Supports the score-head idea (1.2) on a looped architecture.

### 2.12 Seek in the Dark (LatentSeek)
Li et al., 2025. arXiv:2505.13308. VERIFIED. Read: abstract.
### 2.13 GradCuit
Yu et al., 2026. arXiv:2608.02585. VERIFIED. Read: abstract.
- Mechanism (both): test-time optimization of instance-specific latent states by
  reward-weighted gradients through the frozen model.
- For LXTUL: already covered by MORPH's gradpass and fitted-z work (fitted z used the answer).
  Not new. Listed only to mark it as explored.

### 2.14 Latent Thought Flow (LTF)
Zou et al., 2026. arXiv:2606.16222. VERIFIED. Read: abstract.
- Mechanism: continuous GFlowNet over variable-length latent trajectories, entropy-weighted
  subtrajectory balance for intermediate rewards, reference-prior regularizer.
- For LXTUL: see wildcard W2. Subtrajectory balance assigns credit to partial trajectories
  without per-pass targets.

### 2.15 Amortizing intractable inference in large language models
Hu et al., 2023. arXiv:2310.04363. VERIFIED. Read: abstract.
- Mechanism: GFlowNet fine-tuning to sample CoT as a latent variable proportional to reward.
- For LXTUL: background for W2.

### 2.16 Training Chain-of-Thought via Latent-Variable Inference (TRICE)
Phan et al., 2023. arXiv:2312.02179. VERIFIED. Read: abstract.
- Mechanism: MCMC-EM over rationales with a control variate whose variance goes to zero as the
  model improves.
- For LXTUL: the persistent-chain idea (keep the last winning hypothesis per example and propose
  around it) does not fit a streaming pretraining corpus. Low priority.

### 2.17 Other latent-reasoning entries (abstract only, context)
- Soft Thinking, Zhang et al., 2025, arXiv:2505.15778, VERIFIED.
- Reasoning by Superposition, Zhu et al., 2025, arXiv:2505.12514, VERIFIED. Continuous thought
  can hold a superposition of search frontiers (theory, graph reachability).
- Hybrid Latent Reasoning via RL (HRPO), Yue et al., 2025, arXiv:2505.18454, VERIFIED.
- SoftCoT++, Xu et al., 2025, arXiv:2505.11484, VERIFIED. Diverse soft thoughts via distinct
  initial tokens plus a contrastive term.
- Language Models are Hidden Reasoners (LaTRO), Chen et al., 2024, arXiv:2411.04282, VERIFIED.
- Latent-Space Contrastive RL (DLR), Shan et al., 2026, arXiv:2601.17275, VERIFIED. A small
  assistant samples K latent chain encodings, a reward filters them, and ONLY the kept
  trajectories go to a frozen main model for a single decode. Same "score cheap, decode few"
  pattern as 1.1.
- Latent Chain-of-Thought as Planning (PLaT), Wang et al., 2026, arXiv:2601.21358, VERIFIED.
  Planner and decoder split; lower pass@1 but higher pass@128 (coverage).
- Latent Reasoning via Sentence Embedding Prediction, Hwang et al., 2025, arXiv:2505.22202,
  VERIFIED.

---

## Bucket 3. Diffusion and flow matching in latent space

LCTUL showed a flow thinker over a span code can be context-blind (the past was 1% of its loss)
and Euler depth can be flat. Each entry says how the paper handles context and bypass.

### 3.1 Next Thoughts Are Distributions: Autoregressive Thought Flow (ATF)
Li et al., 2026. arXiv:2609.33271. VERIFIED. Read: method.
- Mechanism: the causal backbone computes the context state; a lightweight rectified-flow head
  (24 steps) samples the next continuous thought from it; the sample is fed back. Targets come
  from a frozen causal-register VAE over text rationales. Optional RL with separate latent and
  text advantages (N trajectories x M answers).
- Context: the head conditions on the backbone's hidden state, which carries the context. The
  paper does not describe an explicit anti-bypass mechanism.
- Findings: direct ablation vs MSE regression on the same targets: MSE collapses to one mode and
  gives repetitive thoughts; the flow head keeps modes. MATH-500 pass@100 83.2 vs LaDiR 63.7.
- For LXTUL: the flow head is the cheap way to get K hypotheses: K samples from a small MLP head
  on ONE loop state instead of K loop rollouts. That removes the K x slot-loop cost. Risks: LCTUL
  context-blindness (the head's loss can be dominated by the unconditional shape of the code),
  and the frozen-VAE target on a live front.

### 3.2 Stop-Think-AutoRegress (STAR-LDM)
Lovelace et al., 2026 (COLM 2025 per search results). arXiv:2602.20528. VERIFIED. Read: method.
- Mechanism: a 768-d Sentence-T5 embedding of the continuation is the plan. The noisy plan is
  projected to 8 soft-prompt vectors; the AR decoder reads prefix plus soft prompt. The decoder
  is trained on NOISED ground-truth plan embeddings with the noise level as input, so it learns
  to rely on the plan at low noise and fall back to the prefix at high noise.
- Context: the diffusion denoiser runs through the AR model, so it sees the prefix.
- Compute: 15-20 denoising steps before diminishing returns.
- For LXTUL: the noise-level-conditioned decoder is a specific answer to "teacher forcing of the
  latent is a bypass": at high noise the plan carries nothing, so the decoder must use context.
  But at low noise it IS a bypass; STAR-LDM accepts that because at inference the plan is
  generated. In strict geometry there is no context fallback for cross-span content, so this
  trick changes meaning. Medium constraint risk.

### 3.3 Diffusion Guided Language Modeling (DGLM)
Lovelace et al., 2024. arXiv:2408.04220. VERIFIED. Read: abstract.
- Precursor of 3.2: diffusion proposes a semantic embedding, a prompt generator turns it into a
  soft prompt for an AR decoder. Background.

### 3.4 Diffusion Forcing
Chen et al., 2024. arXiv:2407.01392. VERIFIED. Read: abstract.
- Mechanism: causal model denoises tokens with independent per-token noise levels.
- For LXTUL: per-slot independent noise levels on the slot state is a way to train the coda to
  read hypotheses of every quality level (a cheap robustness trick, cf. 3.2). Not a cost lever.

### 3.5 Block Diffusion
Arriola et al., 2025. arXiv:2503.09573. VERIFIED. Read: abstract.
- Background: AR over blocks, diffusion inside blocks. Relevant only as a hierarchy precedent.

### 3.6 Continuous Latent Diffusion Language Model (Cola DLM)
Guo et al., 2026. arXiv:2605.06548. VERIFIED. Read: abstract.
- Mechanism: text VAE, block-causal DiT flow-matching prior over latents, conditional decoder;
  matched 2B AR and LLaDA baselines, scaling to about 2000 EFLOPs.
- For LXTUL: the strongest 2026 evidence that a hierarchical continuous latent prior can scale
  on text. Its decoder reads the latent once. Context handling is block-causal in the DiT.

### 3.7 Latent Reasoning with Normalizing Flows (NF-CoT)
Tu et al., 2026. arXiv:2606.06447. VERIFIED. Read: method.
- Mechanism: a TARFlow-style normalizing flow gives EXACT likelihoods for continuous thoughts.
  The NF head inside the backbone outputs a diagonal Gaussian per thought position; a shallow
  flow maps VAE codes of CoT into that space. One causal stream holds thoughts and text, so the KV
  cache is shared. Policy gradient uses the exact thought likelihood.
- Compute: 2.48x cheaper per sample than LaDiR (30 steps).
- For LXTUL: exact likelihood for a sampled slot hypothesis lets REINFORCE-style credit work
  with one decode per sampled hypothesis. Risk: the head outputs sigma, i.e. a learned noise
  scale, the thing that died in MORPH. Also frozen-VAE targets. High constraint risk.

### 3.8 Autoregressive Image Generation without Vector Quantization (MAR)
Li et al., 2024. arXiv:2406.11838. VERIFIED. Read: abstract.
- Mechanism: small per-token diffusion MLP head conditioned on the transformer output. The origin
  of "cheap many samples from one backbone state" (ATF, CALM heads follow it).

### 3.9 Multimodal Latent Language Modeling with Next-Token Diffusion (LatentLM)
Sun et al., 2024. arXiv:2412.08635. VERIFIED. Read: abstract.
- Mechanism: VAE latents plus next-token diffusion; sigma-VAE fixes the latent variance to
  avoid variance collapse.
- For LXTUL: the fixed-sigma choice agrees with MORPH's learned-sigma failure. Useful if a span
  VAE is ever built as the scorer's encoder.

### 3.10 Consistency Models; Mean Flows
Song et al., 2023, arXiv:2303.01469, VERIFIED. Geng et al., 2025, arXiv:2505.13447, VERIFIED.
Read: abstract.
- Mechanism: one-step (or few-step) sampling from a flow.
- For LXTUL: if a flow head (3.1) is used for hypotheses, one-step sampling keeps it cheap. Does
  nothing for context-blindness.

---

## Bucket 4. Energy-based, verifier-based and contrastive scoring

### 4.1 Continuous Autoregressive Language Models (CALM)
Shao et al., 2025. arXiv:2510.27688. VERIFIED. Read: method.
- Mechanism: an autoencoder compresses K=4 tokens to one vector (variational, KL clipping,
  15% latent dropout, 15% input masking, so a sigma≈0.3 perturbation still decodes >99.9%).
  The LM predicts the next vector with an energy-score head: N=8 samples from an MLP head fed
  noise, loss `2/(NM) sum ||z_target - z_sample|| - 1/(N(N-1)) sum ||z_sample - z_sample'||`.
  The energy score is strictly proper and needs samples only, no likelihood. Feeding back the
  DECODED tokens beat feeding the continuous vector (BrierLM 4.70 vs 3.25).
- Compute: the head is about 10% of params; 8 samples cost 8 MLP passes. CALM-L (K=4) beats a
  Transformer-M at 35% less train FLOPs on BrierLM. K=1 is worse than a baseline; K=8 degrades.
- For LXTUL: the energy score is a principled span-level loss for a SET of hypotheses: the first
  term pulls each hypothesis toward the true span encoding, the second pushes hypotheses apart.
  It is strictly proper, so unlike the gamed cos/epi fan diversity terms its optimum is the true
  conditional distribution. Cost O((KM)^2 d) per slot, negligible. Replaces: the mixture loss as
  the exploration signal (the coda still reads one hypothesis for token CE). Risks: needs a span
  encoder target (live front drift, frozen-encoder collapse); the energy score can still be
  context-blind if the target distribution is mostly context-free. CALM's "discrete feedback
  beats continuous" is a warning for writing a raw latent into prefix cells.

### 4.2 Energy-Based Transformers are Scalable Learners and Thinkers (EBT)
Gladstone et al., 2025. arXiv:2507.02092. VERIFIED. Read: method.
- Mechanism: learn a scalar energy E(context, candidate); predict by gradient descent on the
  candidate; train by backprop through the descent (Hessian-vector products); replay buffer,
  Langevin noise, randomized step size and depth. Self-verification: generate M candidates,
  keep the minimum energy (+4 to 18%).
- Compute: more expensive per step than a feed-forward transformer (second order). Claims
  slightly better FLOP scaling than Transformer++ and better data scaling.
- For LXTUL: see wildcard W1. An energy over (context, span hypothesis) is exactly a span-level
  scorer; min-energy selection is WTA at inference. Risk: the descent converges to a minimum,
  i.e. settling, the failure mode of the fixed-point term at strength 1.0.

### 4.3 The JEPA Paradox in Language: The Geometry of Linguistic Alternatives
Dinh and Vo, 2026. arXiv:2607.23531. VERIFIED. Read: method.
- Mechanism (argument): squared-error latent prediction learns the conditional mean. Text has
  several valid continuations per context, so the mean is a centroid that is none of them;
  either distinctions are kept (prediction is hard) or dropped (collapse). Experiment: T-JEPA
  validation effective rank fell from 4.66 to 1.57; downstream at chance.
- For LXTUL: an outside explanation of MORPH's NextLat-on-slot-exit failure and of the frozen
  encoder collapse. It argues for multi-hypothesis or distributional latent targets (WTA, energy
  score, flow), never a single MSE target. It does not propose a fix.

### 4.4 LLM-JEPA
Huang et al., 2025. arXiv:2509.14252. VERIFIED. Read: abstract.
- Mechanism: JEPA loss added to the LM loss, predictor tied to the LM via a [PRED] token.
- For LXTUL: works because the generative loss is kept. Background.

### 4.5 LeJEPA
Balestriero and LeCun, 2025. arXiv:2511.08544. VERIFIED. Read: abstract. Already in the tree
(`2026-09-16-tul-code-lejepa.md`). SIGReg is the known guard if any span encoder is trained by a
latent loss.

### 4.6 Representation Learning with Contrastive Predictive Coding (CPC)
van den Oord et al., 2018. arXiv:1807.03748. VERIFIED. Read: abstract.
- Mechanism: InfoNCE: score the true future encoding against negatives from other sequences.
  The optimal critic is the density ratio p(y|x)/p(y).
- For LXTUL: this is the direct answer to LCTUL-style context-blindness. The ratio divides out
  the marginal p(y), so the context-free part of the next span earns nothing; the scorer is paid
  only for what the context (through the loop) predicts. Use InfoNCE across in-batch spans to
  score hypotheses: score(h_k, e(true next span)) vs score(h_k, e(other spans)). Cost:
  B*S*KM*d dot products. Risk: a live span encoder; negatives from the same document can be
  near-duplicates.

### 4.7 Residual Energy-Based Models for Text Generation
Deng et al., 2020. arXiv:2004.11714. VERIFIED. Read: abstract.
- Mechanism: a sequence-level EBM on top of an AR LM, trained by noise-contrastive estimation
  with AR samples as negatives; generation by importance sampling from the AR proposal.
- For LXTUL: the "cheap proposal, energy reranks" pattern. The negatives are free here: the
  losing hypotheses of the same slot.

### 4.8 NCP (ConceptLM) and NCP-ArchPreview: see 5.3.

---

## Bucket 5. Hierarchical and span-level models with cheap token decoding

### 5.1 Block Transformer: Global-to-Local Language Modeling for Fast Inference
Ho et al., 2024. arXiv:2406.02657. VERIFIED. Read: method.
- Mechanism: lower layers run over block embeddings (L_B tokens each); a token decoder decodes
  each block with LOCAL attention only, reading global context as 2 prefix vectors. Prefix
  conditioning beat cross-attention and summation in their ablation.
- Compute: 10-20x inference throughput; about 2-3x params to match vanilla perplexity.
- For LXTUL: MORPH's strict-geometry coda already has this shape (span-local tokens plus prefix
  cells). The paper's ablation supports prefix cells as the read. The price paid for locality is
  parameters, not FLOPs: a warning that the strict channel may need a wider coda, not more reads.

### 5.2 Latent Thought Models with Variational Bayes Inference-Time Computation (LTM)
Kong et al., 2025. arXiv:2502.01567. VERIFIED. Read: method.
- Mechanism: 24-96 latent vectors per layer, read by cross-attention in every decoder layer;
  Gaussian posterior per sequence, fitted by 16-64 fast gradient steps; slow step on the decoder.
- Compute: the fast inner loop dominates (T_fast decoder forward+backward per step).
- Caution: the reported OpenWebText perplexity (3.05) is an ELBO with z inferred FROM the
  sequence being scored, i.e. the posterior saw the answer. That is MORPH's "fitted z used the
  answer" trap. Do not compare it to an AR perplexity.
- For LXTUL: per-layer cross-attention to latents is the late-fusion interface for 6.1. The
  inference-by-descent part is already explored in MORPH (gradpass, fitted z).

### 5.3 Next Concept Prediction in Discrete Latent Space (ConceptLM) and NCP-ArchPreview
Liu et al., 2026. arXiv:2602.08984. VERIFIED. Read: method.
Intern-NCP Team, 2026. arXiv:2609.10715. VERIFIED. Read: abstract.
- Mechanism (ConceptLM): mean-pool hidden states over k=4 tokens, product-quantize (one codebook
  of 64 per head segment, SimVQ MLP against codebook collapse, near 100% use). A concept head
  predicts the next concept (MSE between the soft codebook mix and the true concept). The
  PREDICTED concept (not the true one) is broadcast and added to token states, shifted k-1
  positions to avoid leaks. NTP, NCP and VQ losses train together.
- Results: GPT-2 1.5B scale matches baseline with 76% of tokens; NCP-ArchPreview (8.9B, 5.73T
  tokens) reaches OLMo-3-7B's final loss with 51.3% of the tokens (abstract claim).
- For LXTUL: the largest-scale evidence that a span-level latent target helps web-text
  pretraining, and it uses the predicted concept in the token path, which is the anti-bypass
  choice. Mapping: quantize the true next span's pooled state; the concept id of the true span
  is a zero-decode WTA assignment over concept-valued hypotheses. Risks: MSE on the soft mix is
  a regression (JEPA paradox applies, though PQ over 64^S codes is very multi-modal); MORPH's
  vq8/vq4 arms found rank was not the depth limit; no loop in ConceptLM, so depth use is untested.

### 5.4 Hierarchical Reasoning Model (HRM); Less is More: Tiny Recursive Model (TRM)
Wang et al., 2025, arXiv:2506.21734, VERIFIED. Jolicoeur-Martineau, 2025, arXiv:2510.04871,
VERIFIED. Read: abstract (training details are prior knowledge).
- Mechanism: recursive latent refinement; deep supervision over many outer steps; gradient only
  through the last recursion (one-step or truncated gradient); ACT halting. Independent ARC
  analysis credits deep supervision more than the recursion.
- For LXTUL: the compute lesson is "backprop through the last pass, detach earlier passes, and
  supervise every outer step". MORPH runs full BPTT on purpose (l2cap recipe), so truncation is a
  recipe change with depth-use risk. Deep supervision per outer step is close to per-pass
  targets, which the constraints forbid. Low fit.

### 5.5 Scaling Latent Reasoning via Looped Language Models (Ouro)
Zhu et al., 2025. arXiv:2510.25741. VERIFIED. Read: abstract.
- Looped LM through the full pipeline with an entropy-regularized exit gate. Background for the
  plain-loop baseline the PI compares against.

### 5.6 Byte Latent Transformer; H-Net (Dynamic Chunking); MEGABYTE lineage
Pagnoni et al., 2024, arXiv:2412.09871, VERIFIED. Hwang et al., 2025, arXiv:2507.07955,
VERIFIED. Read: abstract.
- Mechanism: a large latent model over patches or learned chunks, small local encoders and
  decoders over bytes. Compute sits in the latent model; local decoders are cheap and run once.
- For LXTUL: the cost shape to copy: expensive thinking at span rate, one cheap local read at
  token rate.

### 5.7 Semformer; Language modeling via stochastic processes; Learning to Plan (Cornille 2024)
Yin et al., 2024, arXiv:2409.11143, VERIFIED. Wang et al., 2022, arXiv:2203.11370, VERIFIED.
Cornille et al., 2024, arXiv:2404.00614, VERIFIED. Read: abstract.
- Semformer: planning tokens regress onto an autoencoder's latent of the response (a regression
  target, JEPA-paradox risk). Time Control: Brownian-bridge sentence latents via contrastive
  learning. Cornille: planner predicts k-means writing actions of future sentences, LM
  conditioned on them. Background for span-level planning on text.

### 5.8 Better and Faster LLMs via Multi-token Prediction
Gloeckle et al., 2024. arXiv:2404.19737. VERIFIED. Read: abstract. Background: extra heads on a
shared trunk, cheap multi-future supervision.

---

## Bucket 6. Sharing one token decoder across hypotheses

### 6.1 Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward
Liu et al., 2025. arXiv:2506.05433. VERIFIED. Read: abstract (attention split is prior
knowledge).
### 6.2 Hydragen: High-Throughput LLM Inference with Shared Prefixes
Juravsky et al., 2024. arXiv:2402.05099. VERIFIED. Read: abstract.
- Mechanism (both): split attention into a shared-prefix part and a per-branch part, combine by
  log-sum-exp, so the shared prefix is encoded once per group, with exact gradients.
- For LXTUL: exact sharing works only for positions whose states do NOT depend on the
  hypothesis. In the coda, every token attends to the prefix cells at every layer, so nothing is
  shared today. Late fusion makes it exact: if coda layers 1..j do not see the cells, token
  states up to layer j are shared across all K x M hypotheses, and only layers j+1..4 plus the
  vocab CE run per hypothesis. Combine with the log-sum-exp attention split so the per-hypothesis
  part attends to shared token keys without recomputing them.

### 6.3 Parallel Scaling Law for Language Models (ParScale)
Chen et al., 2025. arXiv:2505.10475. VERIFIED. Read: method.
- Mechanism: P streams made distinct by learnable prefixes (distinct KV per stream); the P final
  representations are aggregated by an MLP that outputs softmax weights. Loss scales like
  multiplying params by O(log P).
- Compute: the whole network runs P times, so training FLOPs scale with P (they apply it only in
  a short second stage).
- For LXTUL: LX's fixed simplex codes are ParScale's input transforms, confined to the cheap slot
  loop. ParScale then aggregates BEFORE the loss with a learned soft weight, not by a mixture of
  K separate decodes. Analog: aggregate the K code rollouts per cell with a learned softmax
  weight (from the loop state), write one cell, coda reads once. Risk: this is an average (2.3
  non-closure), and ParScale's gain is O(log P) capacity, i.e. ensemble width, which MORPH already
  measured as what LX rollouts are ("width = ensemble gain").

### 6.4 Superposed Decoding
Shen et al., 2024. arXiv:2405.18400. VERIFIED. Read: method.
- Mechanism: feed the probability-weighted superposition of k drafts' last-token embeddings as one
  input; split the output back into k drafts with n-gram interpolation. Works because layers are
  roughly linear in the superposed input (cosine >= 0.6 to components over 10 steps).
- Compute: k drafts at the cost of one pass; 2.44x lower latency at k>=3.
- For LXTUL: a test of whether the coda is linear enough in its prefix cells that one read of a
  superposed cell approximates the K separate reads. Cheap offline probe. Risk: non-closure again;
  and the demux step needs a cheap scorer (their n-grams).

### 6.5 Learning Tractable Distributions of Language Model Continuations (LTLA)
Yidou-Weng et al., 2025. arXiv:2511.16054. VERIFIED. Read: abstract.
- Mechanism: a neural head predicts only a prefix-dependent latent prior; one shared HMM answers
  all continuation queries exactly in one batched pass (14% decode overhead).
- For LXTUL: the "tractable surrogate scorer" idea (wildcard W3): score all hypotheses with a
  small span-local model whose cost does not scale with a coda read.

### 6.6 Breaking the Softmax Bottleneck (Mixture of Softmaxes)
Yang et al., 2017. arXiv:1711.03953. VERIFIED. Read: abstract.
- Mechanism: K softmaxes from K projections of one hidden state, mixed by learned priors.
- For LXTUL: the cheapest exact "mixture over hypotheses" is at the output layer: shared token
  states, per-hypothesis logits. But at V≈49k and d=1024 a vocab projection costs about as much as
  4 transformer layers per token, so K softmaxes are NOT cheap here. Use only on a top-2 shortlist.

### 6.7 Parallel Decoder Transformer; K-Forcing
Robbins, 2025, arXiv:2512.10054, VERIFIED. Tang et al., 2026, arXiv:2606.10820, VERIFIED.
Read: abstract. Background only (planner-conditioned parallel lanes; one-pass joint k-token
sampling). Not directly usable.

---

## Cross-cutting warnings (what the literature says will break)

- **Averaging hypotheses before the decoder** (6.3, 6.4, 2.10, 2.6) risks Latent-GRPO's mixture
  non-closure and MORPH's "four copies of one state" failure. Only merge discrete or nonlinear
  (set-transformer) forms, and measure.
- **Latent regression targets** (Semformer, ConceptLM's MSE, NextLat) risk JEPA-paradox centroid
  collapse. Prefer WTA, energy score, or InfoNCE.
- **Oracle latents at train time** (Cho 2019, E2E planner oracle actions, STAR-LDM low noise) are
  the teacher-forcing bypass. Use the true span only to CHOOSE among the model's hypotheses.
- **Responsibility noise** (Shen 2019): compute assignments without dropout.
- **Learned sigma** (NF-CoT, GTS, any Gaussian head) collides with MORPH's measured failure.
- **Settling** (EBT descent, DEQ-like scorers) collides with the fixed-point finding.

---

## (a) Ranked shortlist: cheap first experiments

Cost is stated in coda reads per step. Today: 24 coda units per step (8 carry gradient), 3.4x
the plain looped step. "One grad read" is the floor for any design that trains the token decoder.
The first two experiments are OFFLINE probes on an existing LX-Fan+WTA checkpoint: zero training
cost, and they decide whether items 1, 2 and 6 are worth building.

**1. Latent-space WTA assignment, one decode (Cho 2019 + WTA geometry + CPC).**
Experiment A (offline, no training): on the b-5k / ev01-3k checkpoints, for each slot compute
the coda-CE winner among the K x M hypotheses (already done in the winner-agreement probe), and
three cheap candidate assignments: (i) nearest hypothesis to the NEXT slot's own pooled
pre-loop cell (the fan already pools span g+1 with learned queries in the same forward, so this
encoding is free), (ii) nearest to the prelude mean-pool of span g+1, (iii) InfoNCE argmax with
in-batch negatives. Report agreement with the coda winner and REGRET (coda CE of the cheap winner
minus coda CE of the true winner, in nats per span). Kill rule: if regret exceeds the WTA gain
itself, stop. Experiment B (training): train with assignment (best of i-iii) and one grad coda
read of the winner. Cost: 1 grad read (vs 8 grad + 16 no-grad) plus O(KMd) scoring, so the coda
part drops by about 8x. The slot loop still runs K x M streams; see item 5 for that cost.

**2. Late fusion in the coda (Block Transformer prefix read + Prefix Grouper split + LTM
per-layer read).** Keep LX's exact mixture, but let the prefix cells enter only in coda layers
3-4 (or only 4). Token states for layers 1-2 are computed once and shared by every hypothesis;
the per-hypothesis part attends to shared keys via the log-sum-exp split. Experiment: first an
offline probe (retrain nothing; mask cell attention in coda layers 1-2 on the existing
checkpoint and measure CE loss of the thought's value), then a 3k arm. Cost: per-hypothesis coda
cost falls to about half (2 of 4 layers) but the vocab CE is still per hypothesis, so expect the
coda part to fall 1.5-2x, not 8x. Method-preserving. Risk: "the reader was the limit" (though
cond4 found reader depth was not).

**3. Score head for eval and sampling (rMCL + LatentRM + LTO).** Add a scalar head on each
hypothesis's loop output, trained with a softmax-contrastive loss over the slot's hypotheses on
the item-1 winners. At eval, read the top-1 (or a score-weighted top-2) instead of the K-way
per-token Bayes read. Cost: training about free; eval 1-2 reads instead of K. Experiment:
measure val CE of top-1-by-score vs the full Bayes read on the same rows. Low constraint risk.

**4. Energy-score or InfoNCE span loss over the hypothesis set (CALM + CPC).** Replace the
mixture/WTA token loss as the exploration signal with a strictly proper set loss in latent space:
energy score (attraction to the true span encoding, repulsion among hypotheses) or InfoNCE
against in-batch spans (divides out the context-free marginal, the direct answer to LCTUL's
context-blindness). The coda reads one hypothesis for the token CE. Experiment: a 3k arm with
target = stop-grad next-slot pooled cell (item 1's encoder), with SIGReg on that encoder if its
rank falls. Report K1-K6, CE, and hypothesis spread. Cost: 1 grad read plus O((KM)^2 d). Risks:
live-encoder drift, and whether a latent loss moves depth use at all (per-pass targets were met in
one pass; this is a per-slot target on the final pass, not per pass, so it does not break the
rule, but watch for the same flatness).

**5. Fewer, noisier-free decodes: K=2 without replacement (Kool 2020 + LTC).** If item 1's regret
is too high, decode the top-2 hypotheses by score (item 3), train the coda on both, and use
Kool's unordered-set estimator for the scorer. LTC's (2,4) > (4,2) result says two well-credited
hypotheses beat four noisy ones at equal budget. Cost: 2 grad reads (about 1/4 of today's coda
cost). Also cut the slot loop: run only the K code rollouts whose cells are needed (top-2 by a
pass-1 score), which attacks the non-coda cost the profile inference points at.

**6. Superposition probe, then a nonlinear merge (Superposed Decoding + 2.10 SampleTransformer
+ ParScale aggregation).** Experiment A (offline): on the checkpoint, compare the coda CE of one
read with the posterior-weighted superposition of the K cells against the exact K-read mixture.
If the gap is small, the coda is linear enough and a merged read is viable. Experiment B: a tiny
set transformer merges K x M cells per slot into M cells, one coda read. Cost: 1 grad read.
Risk: non-closure (2.3) and the a1 copies failure; this is the most likely of the six to reduce
depth use, which is why it ranks last.

## (b) Wildcards

**W1. The loop as energy descent (EBT).** Make the slot core compute a gradient step on a
learned span-level energy E(context, z), with K initializations as the hypotheses and min-energy
as the selection. Depth use would then mean "more descent steps lower the energy", which is
emergent, not forced. Risk: descent settles into a minimum, which is the fixed-point freeze MORPH
already measured; EBT needs Langevin noise and randomized step size to avoid a sharp landscape,
and second-order backprop costs more per step.

**W2. Continuous GFlowNet credit over loop passes (LTF + NF-CoT).** Treat each pass as a
stochastic transition and train with subtrajectory balance, reward = -CE from ONE coda read of
the final hypothesis. This assigns credit to intermediate passes without per-pass targets. Risks:
it needs a transition density (a Gaussian head, i.e. sigma, which died under the likelihood;
here it is trained by a balance loss, not the likelihood, which is untested in MORPH), and
GFlowNet variance at pretraining scale is unknown.

**W3. A draft coda as the scorer (speculative decoding at train time + LTLA).** Distill a 1-layer
draft coda (shared embeddings) that reads only the first few tokens of the next span, and use it
to rank all K x M hypotheses; the real coda reads only the winner. Cost per hypothesis is 1 layer
on a few tokens plus a small vocab slice. Risk: the draft misprices exactly where the thought
matters. Cheap to measure on the checkpoint (agreement and regret as in item 1).

## (c) What I could not find or verify

- No paper trains a LOOPED latent reasoner with multi-hypothesis exploration on web-text
  pretraining and reports cost against a plain looped baseline. Everything here is either
  task-level RL (math), inference-time search, or a non-looped pretraining model.
- No paper measures agreement between a cheap latent-space assignment and the decoder's own
  winner for language. Cho 2019 uses an oracle mask in seq2seq; LatentTTS measures scorer
  accuracy against rollouts, not against a decoder winner. Item 1's offline probe would be new.
- No paper tests whether late fusion of a latent (item 2) keeps a looped model's depth use.
- Shen et al. 2019: the PDF full text did not parse; I used only the abstract.
- The STAR-LDM venue (COLM 2025) comes from search snippets, not from a page I opened.
- The component split of the 1.33 s backward in the profile is not in the note I read; the
  "non-coda part is about 2x plain" figure above is my inference and is unverified.
- Numbers quoted from `method` reads came through a summarizing fetch tool. I did not check
  every constant against the PDF.

## Counts

- Distinct arXiv IDs cited: 76 (checked by grep against the API-verified list). VERIFIED: 76 (ID, title, first author, date confirmed from
  the arXiv API record). UNVERIFIED: 0. Dropped for lack of confirmation: 0 (every candidate I
  searched for resolved).
- Method sections read (via full-text fetch): 21 (Cho 2019, rMCL, LoRA-MCL, LatentTTS, LTC,
  Latent-GRPO, Multiplex Thinking, MUX, E2E planner, Learning to Plan Long-Term, ATF, STAR-LDM,
  NF-CoT, CALM, EBT, JEPA Paradox, Block Transformer, LTM, ConceptLM, ParScale, Superposed
  Decoding). The rest: abstract only.
