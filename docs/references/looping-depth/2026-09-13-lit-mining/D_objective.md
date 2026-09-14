# Lit review D — objectives that give each pass a job

Read against the MORPH/TUL slot-loop failure in `loop_lit_brief.md`: every per-pass target we
tried is met in pass 1 (K3-K6 ~ 0, passes 2-6 add <= 0.002 nats).

---

## 1. LoopMTP (arXiv 2608.03624, Shomali et al. 2026)

**What they measured, and how.** GPT-2-style decoder-only transformer, L=12, d=1024, 32 heads,
nominal FFN 4096; **260M params** for LoopMTP, 266M for the non-looped baseline, ~280M for the
LoopFormer baseline (Sec 4.1). Trained from scratch on the Nemotron-CC-v2 + Nemotron-CC-Math-v1
subsets, 6.8B tokens, Muon+AdamW, peak LR 1.9e-3, cosine decay to 1/10 peak (Sec 4.1). Evaluated
with perplexity on FineWeb-Edu/OpenWebText, OLMES general tasks (ARC-C/E, HellaSwag, WinoGrande,
SIQA, PIQA, LAMBADA), bits-per-byte on math/code/QA suites, and GSM8K 8-shot for a domain-expert
variant (Sec 4.1, Sec 5). **Their loop-contribution instrument is not a forced-depth K-curve** —
they never truncate the loop. Instead: (a) a ground-truth-rank-at-each-iteration probe, reading
the LM head off each iteration's *raw* hidden state (Fig 3); (b) cosine similarity between
consecutive iterations' outputs (Fig 2, bottom); (c) the learned per-iteration gate weight in
their aggregator (Fig 4, right). There is no matched-compute (same total block-passes, no loop)
control.

**Numbers.** Table 1: non-looped 266M gets PPL 21.08/23.56 (FineWeb-Edu/OpenWebText), general-task
avg accuracy 46.28, QA/Math/Code avg BPB 0.8562. LoopMTP at loops=9 gets PPL 18.86/20.92
(10.5%/11.2% better, Sec 4.2) and general-task avg 50.02 (+8.08% relative vs non-looped, Sec 4.2);
at loops=7 it gets avg BPB 0.7922 (-7.5% vs non-looped 0.8562, -57% vs LoopFormer's 1.8470, Sec
4.2) and beats LoopFormer in 27 of 28 matched loop-count x benchmark comparisons (Sec 4.2). GSM8K
domain-expert (Sec 5, Fig 5): LoopMTP 19.03% vs param-matched non-looped 7.05% (+11.98 p.p., ~170%
relative, Conclusion), trained on ~6.8B tokens of Nemotron-CC-Math-v1 with loop count swept over
T in {3,7,9,11,13,15}. Fig 3: per-iteration ground-truth token rank is up to 35.6x better with the
MTP signal than without it, at various iterations (annotated 2.9x-35.6x per iteration).

**Mechanism.** Each loop iteration t>=2 is trained against a *different* target — the embedding
of the token t steps ahead, not the next token repeated (Sec 3.2, Eq 13) — via a cheap cosine
similarity to a stop-gradient token embedding ("soft MTP"), not a full-vocabulary CE head. So the
T loop passes carry T distinct, horizon-indexed supervision signals. Iteration 1 is left
unconstrained on purpose (no target) so it can serve as a rich substrate the horizon-specific
later iterations read from (Sec 3.2). All T iterations are then combined by a learned
content-conditional gate (Eq 9-11) rather than reading off only the final iterate — nothing is
discarded, unlike a standard loop (Fig 1a, "overwritten & discarded"). The paper frames the
alternative (same target every iteration) explicitly as the failure mode it targets: "latent
overthinking" and "undifferentiated computation" (Sec 1).

**Bearing on Q1-Q4.**
- Q1 (is NTP geometry too disjoint from iteration): evidence AGAINST. Ordinary NTP-adjacent
  targets (future-token embeddings) DO differentiate loop passes once the label is horizon-indexed
  rather than repeated (Fig 2 bottom shows the with-MTP model's iteration-2 output already maps to
  a markedly distinct subspace; the without-MTP model's iterations stay similar through the first
  three — this is close to our own "consecutive pass updates cancel" finding, and MTP appears to
  be what breaks that cancellation).
- Q2 (3B scale threshold): CONTRADICTS a hard floor for this specific mechanism. Gains hold at
  260-280M, essentially MORPH's own scale. Caveat, stated by the authors themselves (Sec 4.3):
  this is NOT the hard-CE full-vocabulary-projection MTP of Gloeckle et al. 2024, which they say
  "hurt performance at a small scale" — their soft, cosine, embedding-space variant is cheap and
  works small; the classic vocab-projection MTP might still need scale. Wolfe's belief is probably
  right about the expensive variant and wrong about the cheap one.
- Q3 (amortized design, later-pass job): no direct bearing — LoopMTP pays full per-token compute
  on every loop pass, there is no amortization. The horizon-indexed-label IDEA is transferable
  even though the architecture isn't.
- Q4 (math principle for iteration): no bearing — no fixed-point or attractor argument is given.

**Concrete arm for us.** Give slot-loop pass t (of T, e.g. T=6) a target that is genuinely
DIFFERENT from pass t-1's: predict the span t spans ahead (not always "next span"), via a cheap
cosine loss against a frozen span-decoder embedding of the t-ahead span, and read the coda from a
gated combination of ALL iterations' z, not just the final one. Measure: K-curve on tokens (does
K3-K6 move off ~0), a Fig-3-style per-iteration ground-truth-rank probe (does iteration-k's z
predict the k-ahead span better than iteration-1's z does), and cosine similarity between
consecutive z's (should drop if they differentiate, mirroring Fig 2 bottom). **Prior: medium.**
This is the most mechanistically apt idea here — genuine content decomposition, not a restaged
copy of one label, which is exactly what every one of our failed arms shared — but our spans are
sparser and longer-range (~50 slots/1024 tokens) than LoopMTP's every-token loop, so a t-ahead span
may run out of usable future context within a 1024-token row for large t.

**Contradicts the brief?** No measured MORPH number is contradicted. It does sharpen how we should
state the Q2 finding: "MTP needs 3B+" is not a settled fact, it's true only for one (expensive)
formulation of MTP.

---

## 2. One Step Forward and K Steps Back / DRM (arXiv 2604.18839, Cameron et al. 2026)

**What they measured, and how.** TRM-style models, **7M and 14M params** (7M: 512 hidden, 8
heads, 4 gradient loops; 14M: 768 hidden, 12 heads, 6 gradient loops — Table 3). Trained/evaluated
on the ARC-AGI grid-transformation puzzle family: ARC-Easy (1,000 tasks), reARC (~400 tasks x
1,000 examples), NVARC-Train (~47,000 tasks x 24 examples), tested on ARC2-Eval (Table 1). This is
**not autoregressive LM on natural text** — grids up to 30x30, 10 colors per cell (Appendix A.1).
No per-step loss/accuracy breakdown by recursion depth is reported; the paper compares only
aggregate multi-step (DRM) vs single-step (standard discrete diffusion) performance.

**Numbers (Table 1).** Standard (single-step) diffusion baseline: 0.0% (7M) / 0.7% (70M) on
ARC2-Eval with no pretraining. DRM 7M with multi-step denoising: 9.6%. With pretraining: TRM(7M)
ARC-Easy 45.7% / ARC2-Eval 6.3%; DRM(7M) 50.5% / 9.6%; TRM(14M) 53.2% / 10.6%; DRM(14M) 55.0% /
13.3%. Largest pretraining set (NVARC): TRM(14M) 21.8%, DRM(14M) 24.9%.

**Corruption schedule.** Cosine schedule: alpha-bar_tau = cos^2(pi*tau/2), masking fraction
r(tau) = 1 - alpha-bar_tau (Eq 8). tau sampled uniformly in [0,1] during training; at inference, 16
denoising steps are sampled and sorted descending. The corruption level is tied to the continuous
diffusion timestep tau, not a raw pass index — the architecture's own recursion ("4/6 gradient
loops," Table 3) sits INSIDE one denoising step's masked-token reconstruction; the OUTER denoising
loop (up to 16 inference steps) is what actually varies corruption pass to pass.

**Could pass 1 solve it alone?** Explicitly no, by construction: an early denoising step's target
is a mostly-MASKED interpolant, which cannot equal the fully-resolved answer by definition — the
"one-step denoiser" (standard discrete diffusion) baseline fails almost completely (0.0%/0.7% vs
DRM's 9.6%, Table 1 + Discussion).

**Mechanism.** "Training through a recursive window gives the model the capacity for implicit
planning in latent space before committing to a final answer... training a denoiser to reverse
that corruption in a single step... does not incentivize learning to produce intermediate states
that remain useful under further self-application" (Sec 1). Discussion (Sec 6): "this provides
multi-step credit assignment: intermediate states are trained to be useful because they enable
progress several steps later, which discourages greedy updates... weight tying across depth can
act as a strong regularizer for algorithmic reasoning."

**Bearing on Q1-Q4.**
- Q1: no bearing directly (puzzle grids, not NTP-adjacent).
- Q2: no bearing — 7-14M models only, no scale sweep.
- Q3: strong bearing. This is a clean instance of "give each pass a job pass 1 cannot do" by
  construction: the LABEL each pass is scored against carries strictly less information at low
  pass index (more masked), so pass 1 is not merely under-trained on the same target, it is being
  asked a different, easier question with a mostly-hidden answer. Non-collapse is structural, not
  emergent.
- Q4: the closest thing to a formal principle in this batch that is stated plainly: iteration is
  required whenever the per-pass loss is evaluated against targets/inputs whose available
  information content strictly increases across passes. No saddle/attractor language, but a clean,
  checkable design rule.

**Concrete arm for us.** Put a corruption/masking schedule on the slot loop's OWN INPUT at low
iteration index (not just the loss target): mask part of the span content the slot sees at low k,
widening to the full span by k=T, and grade every iteration's z against the clean target. Measure
the K-curve on tokens, and specifically whether K1 (near-total corruption) is now much worse than
K6 — if it isn't, the "job" wasn't actually made harder. **Prior: medium-low.** The puzzle domain's
trick works because the corrupted grid literally IS the loop's input; porting this means corrupting
the loop's own input, a bigger architectural change than a loss change, and it must stay causal
(no peeking at the clean span) to remain a fair test.

**Contradicts the brief?** No. It reinforces the brief's own framing: the paper's non-collapse has
nothing to do with NTP/loop geometry compatibility, it's a target-information argument — consistent
with suspecting our failures are about SAME-TARGET-EVERY-PASS design, not an inherent geometry
mismatch.

---

## 3. Thinking with Looped Flows (arXiv 2609.11801, Suleymanzade et al. 2026)

**What they measured, and how.** A stateful denoiser D_t with recurrent state z, trained on a
rollout of k+1 sampled timesteps 0<=t_0<...<t_k<=1 (Eq 9), each a LOCAL cross-entropy denoising
loss on the interpolant x_{t_i} = (1-t_i)x_0 + t_i*x_1 (Eq 3), stop-gradient between steps — no
BPTT. **Models are 5-7M params** (an MLP-Mixer for Sudoku, a 7M transformer elsewhere, Sec 5.1).
Evaluated on Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2 (Table 1), N-Queens and Graph Coloring (Table
2). **Not autoregressive LM on natural text** — same puzzle-grid family as paper 2. Loop
instruments: (a) inference-step scaling, exact accuracy vs number of integration steps on Sudoku,
74.5% at 8 steps to 97.9% at 128 (Fig 3); (b) a training-component ablation (Table 3) removing
time-conditioning / interpolant training / decreasing-noise schedule / noise-sharing, each of which
drops ARC-AGI-1 from 58.8+-1.8 to 43.6-56.4; (c) a comparison against FLM (flow, no recurrence) and
self-conditioning variants (Fig 5), where recurrence-across-decreasing-noise wins.

**Numbers (Table 1, single trajectory).** Looped flows: Sudoku-Extreme 97.9+-0.4, Maze-Hard
86.7+-1.1, ARC-AGI-1 58.8+-1.8, ARC-AGI-2 12.2+-1.9 — vs TRM 87.4/85.3/44.6/7.8, FPRM 94.2/87.0/
47.5/6.2, HRM 55.0/74.5/40.3/5.0. With 5-trajectory best-Q ensembling: 99.3+-0.2/86.9+-1.5/
59.5+-1.9/12.2+-0.9. Table 3 (training ablation, ARC-AGI-1): removing the decreasing-noise
schedule -> 51.6+-1.9; removing noise sharing -> 56.4; removing BOTH time-conditioning and
interpolant training -> 43.6 (near the TRM floor).

**Could pass 1 solve it alone?** No, by construction: an early rollout step's target is a
near-pure-noise interpolant. The paper's Sec 4 explicitly contrasts this with the plain
stop-gradient loop (Eq 2), which gives no learning signal to past states and produces spurious
attractors — TRM fails on 12.6% of Sudoku-Extreme cases this way (88.3% non-convergence, 11.7%
spurious attractor); looped flows recover 89.9% of the non-convergence failures and 98.0% of the
spurious-attractor failures (Sec 5.1).

**Mechanism.** Temporal alignment of per-step local denoising losses via (i) a progressively
DECREASING noise level across the sampled rollout and (ii) SHARING the same noise draw x_0 across
all t_i in one rollout (Sec 3.1, Eq 9), which "encourages hidden states to remain useful across
updates... even when gradients cover only one or a few updates" (Abstract). This is architecturally
close to paper 2's cosine schedule, formulated as continuous flow-matching instead of discrete
masking.

**Bearing on Q1-Q4.**
- Q1, Q2: no bearing (puzzle domain, 5-27M models, no scale study).
- Q3: bearing on WIRING, not amortization. Their ACT head (Eq 10), trained with BCE to flag when
  "further loop steps give no useful signal," is a concrete instrument for detecting an exhausted
  per-pass job. The paper's own text: "Since noise levels decrease over the recurrence, the
  denoising loss overall decreases monotonically and may saturate. Subsequent steps then provide
  no meaningful training signal and could cause overfitting" (Sec 3.1). This is outside
  confirmation that later-pass saturation is EXPECTED even in a design that provably cannot be
  solved in one pass — saturation and non-collapse are separate facts, not opposites.
- Q4: the closest of the four to a formal principle. Sec 4 frames looped models as fixed-point
  iterations related to energy minimization, and Sec 5.1/Fig 4 names two distinct dynamical
  failure modes plain recurrence falls into — spurious attractors (wrong fixed point) and failure
  to converge (oscillation) — neither of which is "solved in 1 pass." Iteration is needed when the
  per-step map is a poorly-conditioned fixed-point solver, a different condition from "the target
  has a closed-form 1-step answer" (which our own results, per the brief, suggest may be our case).

**Concrete arm for us.** Add an ACT-style saturation head (Eq 10: a per-position BCE classifier
predicting "further slot-loop iteration adds nothing") as a DIAGNOSTIC, not a fix — train it
alongside the slot loop and read whether it predicts "halt after iteration 1" almost everywhere by
step ~2500-5000. Measure: fraction of slot positions with predicted halt=1 at iteration 1 vs our
forced-depth K1-K6. **Prior: high that it confirms (not contradicts) our existing K-curve result** —
it's an independently-trained instrument measuring the same saturation phenomenon, not a cure.

**Contradicts the brief?** No MORPH number is contradicted. Worth flagging to Wolfe: this paper
shows "some early passes are required" and "later passes saturate" are not mutually exclusive —
so our K3-K6 ~ 0 finding does not by itself prove passes 1-2 aren't doing real iteration-requiring
work; it could mean 3-6 are past an earned saturation point. That's a nuance for the next
experiment design, not a reversal of the K3-K6 measurement itself.

---

## 4. DiscoLoop (arXiv 2607.00341, Fu, Guo, Wang, Zhu, Lee, Jiao, Russell, Mei 2026)

**What they measured, and how.** A symbolic two-hop reasoning task (Sec 2): entities/relations as
dedicated tokens, atomic facts `<a><r1><b>`, two-hop chains `<a><r1><r2><c>`, with an entity-disjoint
ID/OOD graph split (500 entities/graph, |R|=50, Sec 3.1). Vanilla looped transformer with **K=2**
loop applications, matched to reasoning depth (2 hops, Eq 2, Sec 3.1). Also a synthetic
natural-language verbalization of the same task, direct and reverse phrasing (Sec 5.2, Fig 4). Also
**real pretraining: 440M params, 24 layers, d=1024, loop step 4 (4 total backbone applications),
tied input/output embeddings** (same backbone as Zhu et al. 2025c), 20B tokens on a 6:4
FineWeb-Edu:FineMath mixture (Sec 5.3). Loop-contribution instrument: NOT a K-curve over many
depths, but a genuine **Stage-1 vs Stage-2 forced-partial-depth readout** — reading the LM head off
the hidden state after only the FIRST loop (K=1 at inference) vs after both loops (Table 1a) —
plus a logit-lens probe of bridge-entity decodability and embedding alignment at the intermediate
position (Table 1b), plus zero-shot benchmark averages for the 440M run (Table 2).

**Numbers.** Table 1a, vanilla 2-loop transformer: Stage-1 (after loop 1) accuracy is
train_atom=100%, train_id=8.8%, test_id=0.9%, test_ood=0.0%; Stage-2 (after both loops, the actual
prediction) is train_atom=100%, train_id=100%, test_id=71.1%, test_ood=8.3%. Table 1b, logit lens
at the bridge position post-loop-1: P(correct bridge | H1) = 1.000 on BOTH test_id and test_ood
(the bridge is already essentially perfectly decodable after loop 1), but cosine(H1, embedding of
that bridge) is only 0.327 (test_id) / 0.266 (test_ood) — the answer is present in the readable
logit direction, but the raw state is far from the embedding the NEXT loop needs to consume. A
training-free intervention (Eq 3) mixing in the decoded bridge embedding at the loop boundary
lifts OOD accuracy from 8.3% to 25.9% at alpha=0.1 and to ~100% at alpha~0.5 (Sec 3.2, Fig 2
right). DiscoLoop, the learned version of this fix (Eq 4-6), reaches near-100% ID/OOD on the
symbolic task by end of training (Fig 3 left) vs vanilla loop plateauing ~70% ID / <10% OOD vs a
FLOPs-matched non-looped baseline at <20% ID / ~0% OOD. On the synthetic-language task (Fig 4):
DiscoLoop ~100% ID / ~95% OOD both phrasings vs vanilla loop ~90% ID (fails OOD, worse reversed) vs
non-looped ~40%/~20% ID, ~0% OOD. Real pretraining (Table 2, 440M/20B tokens): vanilla loop avg
49.3, PonderLM avg 49.8, DiscoLoop avg 50.5, best-or-tied on 6 of 7 benchmarks (ARC-C 32.6, ARC-E
57.6, HellaSwag 44.2, LAMBADA 37.6, PIQA 68.4, RACE 31.3, SciQ 81.5); DiscoLoop's training loss
overtakes vanilla loop's after ~13B of the 20B tokens (Sec 5.3).

**Why the discrete channel helps generalization.** The bottleneck is not capacity — the bridge
entity is already near-perfectly decodable via the LM head after loop 1 (P=1.000, Table 1b) — it
is **representational misalignment**: loop 1 receives clean discrete token embeddings as input,
but loop 2 must consume loop 1's raw continuous hidden state, a distribution the shared weights
were never trained to read as "the same entity" ("the base model f_theta is asked to consume two
qualitatively different input distributions across the two loops," Sec 3.2). DiscoLoop fixes this
by softmax-decoding the hidden state through the (tied) LM head into a vocabulary distribution,
re-embedding it as a weighted sum of token embeddings Phi(h) (Eq 5), and injecting it into the next
loop via a learned per-position, per-loop gate alpha (Eq 4, 6) — d+1 extra parameters (Sec 4).

**Bearing on Q1-Q4.**
- Q1: bearing, and it cuts against a pure "geometry too disjoint" story in an actionable way:
  DiscoLoop shows a loop can fail even when the information genuinely IS present after one pass
  (Stage-1 P(bridge)=1.0) — not because the task is unsolvable in one pass, but because the
  STATE'S FORMAT is wrong for the next consumer. This reframes Q1 toward "is z in a format the next
  reader can use" rather than "does the task require more information than one pass provides" —
  close to the brief's own EXTRACTABILITY framing, with a concrete, measured, fixable diagnosis.
- Q2: not directly about the MTP-scale question, but the 440M pretraining result (Sec 5.3) is
  directly germane at a scale close to MORPH's own (~330M) and shows a loop-representation fix
  transfers from puzzle tasks to real LM pretraining.
- Q3: the clearest hit on Q3 among all four papers. The decode-then-encode channel is cheap (d+1
  params, Sec 4) and touches the CHANNEL FORMAT at a loop boundary, not the amount of compute per
  token — structurally compatible with "think once per span, decode cheaply."
- Q4: no formal saddle/attractor argument, but Table 1a's Stage-1/Stage-2 split is a K1-vs-Kfinal
  probe exactly like ours, and here loop 2 clearly does NEW work (8.8%->100% train_id, 0.9%->71.1%
  test_id) even though the constituent fact is already retrievable after loop 1 — because the
  SECOND retrieval (use bridge b + relation r2 to look up c) is a literally different query against
  the weights than the first. The task is a genuine 2-step composition, not "refine my guess."

**Concrete arm for us.** Apply DiscoLoop's decode-then-encode realignment where z crosses a
boundary — between slot-loop iterations, and/or where z is written into the prefix cells the coda
reads: decode z through the (tied) LM head / span-decoder into a soft distribution over span-level
targets, re-embed (Eq 5), gate per-slot (Eq 6), instead of feeding the raw continuous z forward.
Measure: (a) a Table-1b-style logit-lens probe — is the coda's target already near-perfectly
decodable from raw z (likely yes, per our fitted-z finding) while cosine(z, target embedding) is
LOW, mirroring DiscoLoop's 0.327/0.266; if so, this is the SAME diagnosis and the realignment fix
directly applies; (b) K-curve on tokens after adding the channel; (c) rank of z (does realignment
raise the ~13-effective-dim rank, the way DiscoLoop's fix closes the OOD gap). **Prior:
medium-high.** Best-matched diagnosis to our own measured symptom — "answer present, barely read"
is close kin to our slot-rank-anatomy finding (high rank pre-loop, cut by the prefix write) — and
the logit-lens probe (a) is a training-free, sub-hour first check before touching architecture.

**Contradicts the brief?** Table 1a is a genuine counterexample to reading "pass-1-already-good"
as proof a second pass is useless IN GENERAL — Stage-1 train_id accuracy is only 8.8% vs Stage-2's
100%, even though the bridge fact itself is retrievable after loop 1 by a DIFFERENT readout (the
raw LM-head probability). This nuances our own fitted-z/K-curve diagnosis rather than contradicting
its numbers: it depends on which readout is checked (does the DOWNSTREAM reader use z, vs does the
LM head alone decode from z), and DiscoLoop's alignment-vs-information distinction is exactly the
next question to ask of our own z.

---

## Synthesis for Q1-Q4 (300 words max)

**Q1 (is NTP geometry too disjoint from iteration):** No support for "too disjoint" as a hard rule.
LoopMTP differentiates ordinary near-NTP passes once labels are horizon-indexed (Fig 2/3).
DiscoLoop shows pass 2 does real work (8.8%->100% train_id, Table 1a) even when pass 1's answer is
already retrievable by the LM head (P=1.0) — the failure there was representational (cos~0.27-0.33
misalignment), not informational. Our own "coda barely reads z" result (0.0005 nats per 10% rms
perturbation, per brief) looks like the same diagnosis: check alignment before concluding the
geometry itself is wrong.

**Q2 (3B threshold):** Contradicted for cheap mechanisms. LoopMTP (260-280M) and DiscoLoop's
pretraining arm (440M) both show gains near MORPH's own scale. Wolfe's belief likely applies to
expensive hard-CE, full-vocabulary MTP (Gloeckle et al., which hurt at small scale) and not to
cheap cosine-alignment or decode-then-encode variants.

**Q3 (amortized design, later-pass job):** DiscoLoop's decode-then-encode realignment (d+1 params,
applies once per loop boundary) is the best fit — it fixes a format problem, not a compute problem,
so it composes with think-once-per-span amortization. LoopMTP's horizon-indexed per-pass targets
are the cleanest "genuinely distinct label per pass" mechanism, but pay full per-token compute as
published.

**Q4 (when must a map iterate):** DRM and Looped Flows give the sharpest testable principle:
iteration is required when the per-pass loss is scored against inputs/targets whose available
information strictly differs across passes (a corruption schedule) — not when the same label is
restaged, which is what every one of our arms did. Looped Flows also separates "needs more passes"
from "later passes saturate" — they are not opposites; our K3-K6~0 doesn't prove passes 1-2 are
idle.

**Recommended next steps, in order:** (1) DiscoLoop's logit-lens alignment probe on our own z
(training-free, sub-hour). (2) If misalignment confirmed, its decode-then-encode channel. (3) A
LoopMTP-style horizon-indexed slot target as the next real intervention.
