# Massive activations, outlier dimensions and rogue dimensions: reading for MORPH

Read 2026-10-02. Six papers on one effect: a few fixed channels of a transformer's hidden
state carry very large, nearly input-independent values. These channels dominate any
cosine or L2 statistic taken on the raw state. MORPH reason for the reading: the latent
target of the latent-selected loop is a LayerNormed pool of prelude states, and the plain
model's prelude has four such channels (MORPH-side analysis:
[`.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md`](../../../../.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)).

## Cache and reading coverage

No PDFs saved. A research subagent read each paper from its arXiv HTML or ar5iv page on
2026-10-02 and returned a quote and a section for each claim. I re-read two claims myself
from the arXiv HTML (Sun et al. definition, zero-vs-mean ablation, "fixed but important
biases", layer of emergence; Timkey and van Schijndel's 99 % single-dimension claim and
their standardisation result). The other quotes come from the subagent's reading. Where
the source did not state a claim, it is not here.

## 1. Massive Activations in Large Language Models

Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. COLM 2024.
[arXiv 2402.17762](https://arxiv.org/abs/2402.17762).

- **Definition** (Sec. 2.2): "an activation qualifies as a massive activation if its
  magnitude surpasses 100 and is at least or around 1,000 times larger than the median
  magnitude." LLaMA2-7B's largest is about 2,000, against a median of about 0.2.
- **Where.** A very few FIXED feature dimensions (LLaMA2-7B: 1415 and 2533). In the
  sequence: the start token, the first delimiter ("." or "\n"), and in some models weak
  words ("and", "of").
- **When.** They appear in the early layers and stay nearly constant. LLaMA2-7B: they
  "first appear in layer 2 and remain nearly constant values until layer 30", and they
  emerge within one layer of computation.
- **What they do.** Setting them to zero sends LLaMA2-7B WikiText perplexity to infinity
  and mean zero-shot accuracy from 68.95 % to 36.75 %. Setting them to their MEAN gives
  WikiText 5.47 -> 5.47 and accuracy 68.95 % -> 68.94 % (Table 3). "Massive activations
  act as fixed but important biases in LLMs."
- **Attention.** Attention concentrates on the tokens that carry them (Sec. 4.1), and the
  value updates from those tokens are nearly identical across query tokens: additive
  biases. A learned explicit attention bias (a key and value per head, Eq. 3) makes them
  disappear in GPT-2. ViTs (CLIP, DINOv2) have them too; register tokens play the same
  fixed-bias role.

## 2. BERT Busters: Outlier Dimensions that Disrupt Transformers

Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, Anna Rumshisky. Findings of ACL 2021.
[arXiv 2105.06990](https://arxiv.org/abs/2105.06990).

- The outliers are in the LayerNorm scaling factors and biases ("the affected component is
  the scaling factors and biases in the LayerNorm").
- Disabling two outlier dimensions of fine-tuned BERT-base drops STS-B by 44.1 points and
  MNLI by 27.9 (Table 3). Removing 24 parameters raises RoBERTa's loss by almost 4x.
- They emerge early in pre-training (about 50k steps in their Fig. 7) and appear in BERT,
  BART, XLNet, ELECTRA and GPT-2.

## 3. Outlier Dimensions that Disrupt Transformers Are Driven by Frequency

Giovanni Puccetti, Anna Rogers, Aleksandr Drozd, Felice Dell'Orletta. Findings of EMNLP
2022. [arXiv 2205.11380](https://arxiv.org/abs/2205.11380).

- "the magnitude of hidden state coefficients corresponding to outlier dimensions
  correlates with the frequency of encoded tokens in pre-training data." The outliers also
  drive "vertical" attention onto special tokens.

## 4. Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. NeurIPS 2023.
[arXiv 2306.12929](https://arxiv.org/abs/2306.12929).

- Cause (Sec. 3): heads that want to make NO update put their attention on fixed,
  low-information tokens, and the residual grows outliers to make that no-op possible.
- Fixes: clipped softmax `clip((zeta - gamma) softmax(x) + gamma, 0, 1)` (Sec. 4.1) and
  gated attention `sigmoid(G(x)) * softmax(QK^T / sqrt(d)) V` (Sec. 4.2).

## 5. All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality

William Timkey, Marten van Schijndel. EMNLP 2021.
[arXiv 2109.04404](https://arxiv.org/abs/2109.04404).

THE paper for our instruments (H3 in the MORPH note).

- "Perhaps the most striking case is layers 10 and 11 of XLNet, where a single dimension
  contributes more than 99% of the expected cosine similarity between randomly sampled
  tokens."
- The fix is per-dimension standardisation over a corpus: subtract the mean vector and
  divide each dimension by its std ("the z-score in each dimension"). "We found that
  standardization was the most successful postprocessing method, showing consistent
  improvement over the original embeddings in all but the early layers of BERT."

## 6. Anisotropy and the common vector (two older readings)

- Kawin Ethayarajh, *How Contextual are Contextualized Word Representations?*, EMNLP 2019,
  [arXiv 1909.00512](https://arxiv.org/abs/1909.00512). Anisotropy = the expected cosine of
  random word pairs; "The closer this average is to 1, the more anisotropic the
  representations." In GPT-2's last layer any two words are almost perfectly similar.
- Jiaqi Mu, Pramod Viswanath, *All-but-the-Top*, ICLR 2018,
  [arXiv 1702.01417](https://arxiv.org/abs/1702.01417). Word vectors "share a large common
  vector (with norm up to a half of the average norm of word vector)". Removing the mean
  and the top few PCA directions improves similarity, analogy and classification; the top
  directions encode word frequency.

## What this means for MORPH

1. A raw cosine, a mean-squared distance or an L2 regression on a LayerNormed vector that
   carries massive channels measures mostly those channels. Sun et al.'s mean-ablation
   result says those channels are a fixed bias: their per-input jitter is not the
   content.
2. Timkey and van Schijndel's standardisation (fixed per-coordinate mu and sigma from a
   corpus) is the published fix for similarity. The stage-1 build adopted it as
   `tul.latent_pre_target_norm: standard`.
3. The effect is a property of the MODEL that produced the state. On 2026-10-02 the plain
   5k prelude has it (channels 194, 899, 1018, 905 at |mean| about 100 per token), and the
   strict slot-loop arms' prelude and EMA twin do not (top channel means about 1 to 4,
   the top four channels 1 % to 2 % of the energy, measured on nine arms). Read the MORPH
   note before you assume it for any other target.
4. Sun et al.'s mean-vs-zero ablation is a cheap, decisive test of whether a shared
   direction is a bias or content. Applied to the latent-selected loop's written slot
   cells on 2026-10-02, it found that the pulled loops write their own bias direction
   (mean-ablation 0.0005 nats or less, zero-ablation 0.006 to 0.014), while the arm
   without a pull carries its depth signal in the amplitude along its shared direction.
