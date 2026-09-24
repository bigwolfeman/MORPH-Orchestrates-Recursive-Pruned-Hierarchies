# State of Thought Enables Endogenous Reasoning: reading note

Read 2026-09-23, for the LXTUL-GK prereg
([`2026-09-23-lxtul-gk-multisample.md`](../../../../../lab/experiments/planned/2026-09-23-lxtul-gk-multisample.md))
and the LXTUL-G failure
([`2026-09-23-lxtul-g-panel.md`](../../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md)).
Wolfe pointed at it with Reasoning by Superposition while asking which paper solves the
noise collapse in the stochastic slot loop.

Short answer for MORPH: this paper does not address the problem. It has no latent
variable, no sampling, no loop and no superposition. It keeps a frozen LLM and its text
chain of thought, and trains a 582-parameter controller that picks which earlier
sentences stay in the context and when to stop. The one idea that maps onto the slot loop
is "let a small state decide which earlier spans the next step reads". That is a reach or
gather policy, and it is listed at the end as a weak lead.

## Citation (verified)

Zhiren Gong, Yikun Hou, Zihao Zeng, Ming Xiao, Chau Yuen, Wei Yang Bryan Lim (Nanyang
Technological University; KTH). *State of Thought Enables Endogenous Reasoning*.
[arXiv 2609.16055v1](https://arxiv.org/abs/2609.16055) [cs.CL], submitted 2026-09-13.
49 pages. No venue on the PDF or the arXiv page.

Read: Sections 1 to 3 and Appendices A to C in full, from `pdftotext`. The results tables
were read for the rows quoted below. The later appendices (VLM transfer, API judging,
sensitivity) were not read line by line.

## Local cache

- [PDF](state-of-thought.pdf), fetched from `https://arxiv.org/pdf/2609.16055v1` on
  2026-09-23. 10,442,109 bytes, 49 pages.
  SHA256 `2f0ef6a041f8b00571687bb07e0b83f0cd3bee94a1164323126866fc2a6c91c3`.

## What it does

Reasoning is framed as choosing the evidence context for the next step (Section 2.1,
Eq. 1). A frozen backbone writes one sentence `y_t` per step. After each sentence the
method reads a four-number state from the backbone's hidden states and uses it for two
decisions: which earlier sentences stay in the prompt, and whether to stop.

### The state (Appendix B.1, Eq. 12 to 19)

    u_t = mean over tokens i of h_{t,i}             (final-layer sentence centre, Eq. 12)
    delta_t = mean ||h_{t,i}||^2 - ||u_t||^2        (within-sentence spread, Eq. 14)
    v_t = ||u_t - u_{t-1}||                         (step size, Eq. 15)
    c_t = cos(u_t - u_{t-1}, u_{t-1} - u_{t-2})     (direction consistency, Eq. 16)
    H_t = mean next-token entropy over the sentence (Eq. 17)
    m_t = z-score of [delta_t, v_t, c_t, H_t]       (Eq. 18, 19)

### The controller (Section 2.3, Appendix A, Eq. 5 to 9)

For every earlier sentence `i < t` it builds `d_{t,i} = [m_t; m_i; m_t - m_i; m_t * m_i]`
(16 numbers, Eq. 7). A small head scores it, `g_{t,i} = phi_sel(d_{t,i})` (Eq. 8). The
top-scoring sentences under a context budget B are kept in time order and form the
context for the next sentence (Eq. 3, Eq. 9). A linear 4 to 1 head on `m_t` decides to
stop (Eq. 4). Total: a 577-parameter selector and a 5-parameter stop head (Appendix B.2).

### Training (Section 2.3, Appendix C, Eq. 20, 21)

The backbone is frozen. They roll it out on training problems, segment the transcripts
into sentences, and label offline, by replay, whether each earlier sentence is "both
state-compatible and incrementally useful" (a soft target in [0, 1]), and when stopping
is safe. The selector is fit by weighted binary cross-entropy with a sparsity penalty on
gate mass (Eq. 21). No RL, no gradient into the backbone. Selector AUC 0.779 +/- 0.005
over 5 problem-held-out refits (Section 2.3).

## Results (verified from the tables)

Llama-3.1-8B, Table 1, accuracy in %:

| method | GSM | MATH | DROP | FOL | PW | BBH | HE |
|---|---|---|---|---|---|---|---|
| Vanilla | 79.0 | 58.2 | 6.2 | 29.6 | 26.0 | 26.0 | 25.0 |
| CoT | 80.8 | 64.2 | 1.7 | 4.9 | 24.5 | 70.0 | 25.0 |
| Coconut (COCO) | 81.4 | 60.0 | 3.7 | 45.3 | 45.2 | 51.0 | 20.0 |
| SoT | 81.8 | 64.8 | 40.5 | 49.8 | 64.8 | 78.0 | 59.1 |

The abstract reports 62.6 % fewer generated tokens and 44.6 % lower latency, averaged over
3 LLMs and 16 datasets. Several baselines collapse on some tasks (CoT 1.7 on DROP, GRPO-SP
near 0 everywhere), so the relative gains in the abstract (1.34x to 2.51x) are ratios over
a mean baseline that includes these failures. I would not quote the ratios as a measure
of the method. The per-task numbers above are the honest reading.

## What carries over to the slot loop

Mostly nothing, and the reasons are structural.

- No latent. The paper says so: the state controls "which explicit evidence is carried
  forward, without replacing language-space reasoning by an opaque latent chain"
  (Section 2.1). There is no continuous thought and nothing to superpose or sample.
- No loop. Each step is one sentence of ordinary decoding by a frozen model.
- A reward-free supervision signal exists only because the tasks have checkable answers:
  the replay labels use the final outcome (Appendix C, "final outcome (e.g.,
  correctness)"). Web text has no such label.

The one lead, inferred and weak: the selector is a learned GATHER policy. It decides,
from a tiny state, which earlier units the next step reads. MORPH's slot loop already
reads earlier cells through `tul.loop_reach` and attention, so a learned policy would be a
restriction of that read unless it only reweights it. Wolfe closed restriction
geometries on 2026-09-22, and the LXTUL-R relay was measured as a partial refund of the
tax a restriction imposes. So this is not a proposal. It is recorded so that nobody
re-reads this paper expecting a latent-exploration method.

## Verified against inferred

- Verified from the PDF text: the citation, the state definitions (Eq. 12 to 19), the
  controller (Eq. 5 to 9), the training objective (Eq. 20, 21), the parameter count, the
  AUC and the Table 1 rows above.
- Inferred: the reading of the abstract's ratios, and the gather-policy mapping onto
  `tul.loop_reach`.
