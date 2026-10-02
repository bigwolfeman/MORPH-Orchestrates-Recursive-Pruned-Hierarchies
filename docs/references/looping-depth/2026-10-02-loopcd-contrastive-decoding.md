# LoopCD: Decoding Looped Transformers Better for (Almost) Free, reading and local cache

Read 2026-10-02. Weihao Liu, Huangjie Zheng, Tianrong Chen, Rohit Dilip, Richard He Bai,
Yizhu Jiao, Yuyang Wang, Ruixiang Zhang. *Decoding Looped Transformers Better for
(Almost) Free*. [arXiv 2610.02185](https://arxiv.org/abs/2610.02185), submitted
2026-10-01 (v1).

## Cache and reading coverage

No PDF saved to the local archive for this note (out of scope for this edit). Read
from the abstract page (https://arxiv.org/abs/2610.02185) and the full HTML text
(https://arxiv.org/html/2610.02185). The HTML page loaded in full; the PDF fallback
was not needed.

## What it does

The paper treats a looped transformer's own recurrence as a free source of weak and
strong predictions. Each loop pass through the shared block gives an intermediate
state that the model's own output head can decode to a next-token distribution. An
early pass has done less computation than the final pass, so the early and final
passes form an aligned weak/strong pair with no extra model and no extra training.
The method, LoopCD, re-ranks the final loop's token distribution against an earlier
loop's token distribution at decode time only. Nothing in the base model changes.

## The method

Let `h_1` be the reference (early) state and `h_r` be the final loop's state.

- **LoopCD-Logits.** Run the output head on both states to get logits `z_1` and
  `z_r`, then combine: `z' = z_r + omega * (z_r - z_1)`. This costs one extra pass
  through the output layers (the vocabulary projection), not through the recurrent
  core.
- **LoopCD-Hidden.** Combine the hidden states before the output head:
  `h' = h_r + omega * (h_r - h_1)`, then run the output head once on `h'`. Same
  single output-head cost as plain decoding.

**Choosing the reference pass.** For most model families the first iteration
(`h_1`) is the best reference for LoopCD-Logits. Huginn is the exception: Huginn's
recurrent state starts from Gaussian noise, so its early passes have not yet
"burned in." On Huginn, reference passes 1 through 5 give a loss of up to -0.58
points (hurts), and gain turns positive only once the reference is the 6th
iteration (+0.83 points). The gain tracks how often the reference pass disagrees with the final
prediction: on ARC-Challenge, Huginn's first step disagrees on 51 % of questions
and gives 3.07 points at omega 0.5, while its sixteenth step disagrees on 7 % and
gives 0.17 (quoted from the paper's text, checked 2026-10-02). The two variants do not pick the same reference
pass on the same model.

**Contrast strength (`omega`).** Two schedules:

- Fixed `omega`: about 0.5 for multiple-choice scoring, 0.2 to 0.3 for free-form
  generation. Generation tolerates about half the strength that scoring does; past
  roughly `omega = 0.6` on generation, early token shifts compound across the
  generated prefix and accuracy drops sharply.
- Adaptive `omega`: `omega = omega_max * [1 - (p_r,(1) - p_r,(2))]`, scaled by the
  margin between the final pass's top-1 and top-2 token probabilities. A confident
  final pass (large margin) gets little contrast; an uncertain final pass gets more.
  The adaptive schedule holds a positive gain across a wider range of `omega_max`
  (0.5 to 1.0) than any single fixed value does.

**No plausibility mask.** The paper does not restrict the vocabulary or add a
degeneration penalty of the kind contrastive-decoding methods for open-ended text
generation often use. Their stated reason: the adaptive margin gate already
concentrates the re-ranking on tokens the model is unsure about, so "a re-ranking
can only change a decision it can reach."

**Decomposition.** The logit contrast splits into a component parallel to the
final logits (acts like a temperature change) and a component orthogonal to them
(a re-ranking of candidates). Using only the orthogonal re-ranking component
matches or beats the full update, so re-ranking, not temperature, is where the gain
comes from.

## Models

- **Ouro-1.4B and Ouro-2.6B** (including the -Thinking variants). Ouro loops its
  full decoder stack, R = 4.
- **Huginn-0125**, evaluated at R = 32 and R = 16. Huginn starts its recurrent state
  from Gaussian noise (no learned prelude-to-core hand-off the way Ouro and Parcae
  have one).
- **Parcae-370M and Parcae-1.3B**, R = 8. The paper notes Parcae injects a learned
  linear term into the recurrent block each pass.
- **Looped-Qwen3**, built on Qwen3-4B, R = 8 substeps, applying damped updates to a
  frozen middle-layer window (not a from-scratch looped model).

## Benchmarks and headline numbers

Math reasoning, Ouro-2.6B-Thinking:

| Benchmark | Baseline pass@1 | LoopCD-Logits (adaptive) | Delta |
|---|---|---|---|
| AIME 2024 | 61.88% | 73.33% | +11.45 |
| AIME 2025 | 49.58% | 56.88% | +7.30 |
| OlympiadBench | 64.05% | 67.29% | +3.24 |

AIME 2024 pass@10 on the same model: 82.65% to 88.33% (+5.68).

Code generation, Huginn, HumanEval pass@1:

| Setting | Baseline | LoopCD-Hidden | LoopCD-Logits (adaptive) |
|---|---|---|---|
| R = 32 | 22.56% | 31.71% (+9.15) | 28.66% (+5.49) |
| R = 16 | 21.95% | 28.05% (+6.10) | not reported |

Multiple-choice, seven-benchmark mean, smaller but consistent gains across every
model family tested: Ouro-1.4B +0.29 to +0.58 points, Huginn-0125 (R = 32) +0.74 to
+0.86 points, Parcae-1.3B +1.38 to +1.59 points.

**Halving loops while matching full depth.** At half the recurrent iterations,
LoopCD matches or beats the unguided full-depth baseline across every model family
tested. Two examples: Huginn at R = 16 with LoopCD-Logits beats unguided R = 32 by
1.02 points on the seven-benchmark mean; Looped-Qwen3 at half depth matches full
depth exactly (0.00 point difference) under LoopCD-Hidden.

**FLOP savings.** Running at half the recurrent iterations with LoopCD applied
needs 0.52 to 0.78 of the original forward FLOPs while matching or beating the
full-depth unguided baseline, a reduction of 22.5% to 48.2%. This saving is on the
recurrent core's forward pass; LoopCD-Logits still costs one extra output-head
pass, and LoopCD-Hidden costs none.

## Repetition and degeneration

The paper does not run a repetition or degeneration study and does not cite one.
Two related observations appear. First, pushing the fixed contrast strength past
about `omega = 0.6` on generation tasks causes "early token shifts [to] compound
across generated prefixes," which reads as a stability warning on long free-form
output, not a repetition measurement. Second, on pass@k reasoning metrics, LoopCD
can lower pass@1 while raising pass@10 (Looped-Qwen3 on AIME 2024: pass@1 61.88% to
61.04%, pass@10 82.91% to 86.10%), which the paper reads as guidance changing
trajectory diversity rather than uniformly improving every sample. Neither
observation is a direct claim about repeated n-grams or degenerate loops in
generated text.

## Relation to MORPH

**Verdict (Wolfe, 2026-10-02): LoopCD is a deployment-only, training-free decoding
strategy. It gives no training-time savings, and as a drop-in it is not useful to
us now. It is kept for insights.**

Possible insights, each untested on MORPH:

(a) The hidden-state variant (LoopCD-Hidden) maps onto our slot cells as writing
`h_K + alpha * (h_K - h_1)` into the coda's prefix instead of the raw final-pass
cell `h_K`, at zero extra coda passes (the coda already reads one cell per slot;
this only changes what is written there). Untested.

(b) Contrastive decoding is known to reduce degenerate repetition in open-ended
generation. Repetition and generation diversity are already one of our early
metrics. Whether LoopCD's specific weak/strong recurrence pairing carries that
same effect on our slot loop is untested.

(c) Our router-followed latent-selected loop at latent weight 1 is flat after pass
3: forced-depth CE reads 4.480 at pass 1, 4.447 at pass 3, 4.447 at pass 6. Read on
its own terms, deployment at depth 3 is already free for that arm, independent of
LoopCD. This is our own measured number, not LoopCD's.

(d) The paper's evidence is on reasoning benchmarks with 2.6B-class models. Our
slot loop's depth signal is about 0.03 nats at 300M. An amplified small signal may
be noise rather than a real depth-refinement direction worth contrasting against.
Untested.
