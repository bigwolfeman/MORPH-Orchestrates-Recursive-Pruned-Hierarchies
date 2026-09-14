# Lit review C: mechanisms that make later passes different from pass 1

Read against the MORPH/TUL slot-loop failure in `loop_lit_brief.md`: eleven-plus slot-loop
arms earn 0.0003-0.002 nats from passes 2-6 (K3-K6 ~0 or negative); every per-pass target we
tried is met in one pass; the loop is a power iteration (rank 181 -> 9, consecutive updates
cancel, cosine -0.2 to -0.6).

---

## 1. RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory (arXiv 2609.03379, Wang et al. 2026)

**What they measured and how.** Qwen3 decoder-only backbones at 0.6B, 1.7B, 44B, 88B params
(Table 1: 28-36 layers, hidden 1024-4096; a 3-layer block at layers 12-14 or 15-17 is
weight-tied and re-executed T times). Loop-Memory Attention (Section 3, Eq. 4-5) adds, before
each block application, an attention op along LOOP TIME ONLY: query is the current-iteration
state at a token position, keys/values are that SAME token position's states from up to W
prior loop iterations (`M_ℓ^(t) = {x_ℓ^(t') | max(1,t-W) <= t' <= t-1}`), with a learned
relative-loop-distance bias and a scalar+token gate. "The attention runs along loop time only
and never mixes token positions" (Section 3). Matched-compute control (Table 2): fixed loop
counts T=1/2/8/16 on the SAME backbone, plus ACT, PonderNet, CALM, LoopUS-Conf, TaH-Mismatch,
all under "matched budget of steps, tokens, data, and hardware" (Section 4.1) - though the
paper flags that RecurTrace adds ~2.2% trainable params on top of a frozen backbone while the
baselines fine-tune the full model, "any budget mismatch favors the baseline."
Numbers (Table 2, MathQA, 1.7B, 2400 items): fixed T=1 49.96%, fixed T=2 54.71% (the peak
fixed depth), fixed T=8 53.62% (WORSE than T=2 - reasoning peaks at shallow depth then
degrades), RecurTrace adaptive 56.92% at mean depth 2.04. Per-iteration gain is read off this
curve, not a monotone table: +4.75 pts from T=1->T=2, then NEGATIVE from T=2->T=8. GSM8K
1.7B (Table 3): adaptive 10.4% vs best-fixed 9.2% at mean depth 1.72.
**Plain LM loss:** yes, but narrowly. Figure 4 reports teacher-forced NLL (nats) deltas vs
baseline: 0.6B -0.15 nats, 1.7B -0.03, 44B -0.03, 88B -0.02 nats - i.e. the NLL gain from
adding loop memory SHRINKS with scale. Table 5.1 ablation: plain looping (no memory) at 1.7B
COSTS +0.07 nats NLL vs baseline; adding loop-time memory recovers to -0.03 nats. These NLL
numbers are teacher-forced over benchmark answer sequences (MathQA/GSM8K), not perplexity on
a held-out pretraining text corpus - so this is close to but not identical to "plain language
modeling loss" in the brief's sense.
Halting (Eq. 6, Algorithm 1): a learned linear classifier over pooled block state, trained by
distillation from an oracle label `y_t = 1[min_{t'>t} L_{t'} < L_t - delta]` (continue only if
a DEEPER loop would have reduced loss); requires a T_min=2 floor or it collapses to 1 loop.

**Mechanism, one paragraph.** Loop-Memory Attention gives later passes something pass 1
structurally cannot have: a look-back over that SAME position's own trajectory across
iterations, gated per-token and per-loop-distance. It is not new information about the input;
it is a summary of "how my own state has been moving," used to decide both what update to
apply next and (via the halting head) whether to apply one at all. The claimed value is
therefore less "iteration 5 discovers something iteration 1 could not" and more "iteration 5
can see iterations 1-4 and can choose to stop drifting or correct an overshoot" - a
self-correction/damping mechanism, not a relay of external information.

**Q1-Q4.**
- Q1 (geometry disjoint from NTP): no direct bearing - RecurTrace is trained on reasoning
  answer sequences with an oracle halting label, not plain NTP; the NLL result (Table 5.1)
  shows PLAIN looping (no memory, no oracle) actively HURTS NLL (+0.07 nats) versus baseline,
  i.e. their own dense loop without loop-memory failed on the LM-loss axis first, consistent
  with our finding that ordinary looping doesn't easily use extra passes on token prediction.
- Q2 (scale threshold): weak evidence AGAINST a "needs 3B+" threshold in this direction - the
  NLL gain from the mechanism is LARGEST at 0.6B (-0.15 nats) and shrinks toward 88B (-0.02
  nats). If anything this mechanism is more useful small than large, opposite the parameter
  threshold Wolfe hypothesized (though on accuracy, not nats-per-token, gains "range 0.6-3.4
  points across sizes" per abstract with no clean monotone trend reported).
  bears on us; MathQA T=2 beats T=8 (Table 2) — a small compute-matched control ALREADY had a
  fixed T=2 baseline that beat T=8, i.e. more raw loop depth without memory or halting was
  a net loss even at fixed compute, mirroring our "plain model earns 0.033-0.185, slot loop
  earns ~0" split — extra passes need a JOB, not just more depth.
- Q3 (amortized design with a real job for later passes): direct hit. Loop-Memory Attention IS
  a candidate "job" for passes 2-6 that a one-pass map cannot do: it is stateful across
  iterations by construction, so pass k's output is a function of a trajectory a static map
  can't see in one shot. It still amortizes per-TOKEN compute the way ours does (looping over
  the same token positions), not per-span, so it isn't a template for TUL's span-amortization,
  but it is a template for "loop time is itself a queryable axis."
- Q4 (mathematical principle): no formal treatment - the halting oracle (Eq. 6) implicitly
  assumes iteration is a MONOTONE-ISH loss-reduction search with a small number of useful
  steps (mean depth 1.7-2.0 even at 88B) and diminishing/negative returns past ~2 (T=8 losing
  to T=2), which is closer to a bounded local-refinement process than an attractor/saddle
  story - no bearing on the dynamical-systems question in Q4.

**Concrete arm to try.** Give the slot loop a Loop-Memory-Attention-style term: at each
iteration k, let the slot state attend over its OWN states from iterations 1..k-1 (a small
per-slot loop-time KV cache, gated), instead of / in addition to the current per-pass update.
Measure: K-curve (K1-K6) on tokens, and specifically whether K3-K6 (currently ~0/negative)
turns positive, plus the per-pass instrument (does pass 5's update stop looking like -cos(pass
4's update) once it can see pass 1-3 directly?). Prior: LOW-MEDIUM. RecurTrace's own ablation
shows plain looping without this mechanism already loses to baseline on NLL (+0.07 nats,
Table 5.1) even in their reasoning-token setting - loop-memory rescued a broken loop rather
than adding value to an already-working one, which is a different starting point than our
already-not-clearly-broken plain-model loop (0.033-0.185 nats). It could plausibly fix the
"passes 2-6 do nothing" symptom by giving them literal access to their own history (directly
addressing the "power iteration with no memory of earlier iterates" diagnosis in our brief),
but it is unproven on token-level LM loss on general text and adds a nontrivial new module.

**Contradictions.** None outright, but a soft tension: our brief states "the loop is a power
iteration" that reaches a fixed point with no per-pass job. RecurTrace's Table 2 result (T=8
WORSE than T=2 on MathQA) is consistent with "excess depth is not merely inert but can be
actively harmful" without a stopping/memory mechanism - stronger than our own finding, since
we only measured near-zero-or-negative contribution, not outright degradation from depth
alone on plain LM loss.

---

## 2. Sparse Layers are Critical to Scaling Looped Language Models (arXiv 2605.09165, Lee, Biloki, Hu, May 2026)

**What they measured and how.** Small dense-transformer-scale models (Table 5: d_model
128/256/512/1024, d_ff 384-2752, 2-16 heads; active params 16M-305M, stored Looped-MoE
params 18M-407M), trained on FineWeb-Edu (10B tokens, GPT-2 tokenizer, V=50,257) - PLAIN
LANGUAGE MODELING (perplexity), not reasoning benchmarks for the core scaling law, with the
AI2 OLMES suite (ARC, BoolQ, CSQA etc. - general knowledge/reasoning, not math-CoT) as a
downstream check (Table 4/6). Loop config for the routing-divergence analysis: 8 unique
layers x 2 passes = depth 16 (Section 6.1); MoE uses k=2 of E=8 experts (Section 2.2).
Routing divergence is measured NOT as a KL/JS metric but as categorical top-k expert-set
overlap between pass 1 and pass 2 per token (Figure 5): "25-53% of tokens receive entirely
non-overlapping expert assignments between passes, while only 4-14% receive identical
assignments." This is measured at a SINGLE scale point (the 10^18-FLOP compute-optimal
Looped-MoE model) - no scaling trend for routing divergence itself is reported, and only two
passes (not 3+) are compared; Section 6.3's 4x4/2x8 early-exit variants don't repeat the
divergence analysis.
Dense-vs-sparse looped gap (Table 4, 10^18 FLOPs, downstream OLMES avg): Looped (dense) 37.4
at 168M params vs Looped-MoE 39.6 at 216M params - a compute-matched comparison, not a
parameter-matched one. Test-loss power-law exponents (Figure 10): Base alpha=0.076,
Looped-MoE alpha=0.077 - CLOSE scaling slopes, "but Looped-MoE offset favorably" (i.e. same
scaling rate, better constant). Early-exit / loop-boundary finding (Section 6.2, Eq. 1):
Jensen-Shannon divergence between an intermediate loop's output distribution and the final
layer's; Figure 6 - "by the end of the first loop iteration, the majority of tokens have
already reached near-final output distributions" (JSD < 0.5 threshold). Table 7 perplexity at
10% FLOPs saved: Looped 50.2, Looped-MoE 51.0, Base (no loop) 55.4.

**Mechanism, one paragraph.** The claim is narrow and specific: a SHARED weight-tied layer
computes the SAME linear projection every pass, so without routing variation, pass 2's
computation is a near-repeat of pass 1's (same weights, similar activation regime) and adds
little. Giving the layer a MoE router lets it pick a DIFFERENT expert subnetwork on pass 2
than on pass 1 for the same token - the shared "skeleton" (attention + router) stays tied,
but the active computational path diverges per pass, which the paper frames as "recovering
expressivity without additional parameters" (abstract). This is a per-pass ROUTING mechanism,
not a per-pass attention-over-history mechanism (contrast with paper 1) - later passes differ
because they take a different COMPUTATIONAL PATH through the same parameter bank, not because
they see new information.

**Q1-Q4.**
- Q1 (NTP-vs-loop geometry disjoint): partial bearing, and encouraging for us: this is a PLAIN
  LM-loss study (FineWeb-Edu perplexity) and the loop DOES gain from routing divergence in
  that setting (Table 7: Looped/Looped-MoE both beat Base by ~5 ppl at matched FLOPs-saved).
  So NTP does not categorically reject loop depth - but the paper's own mechanism story is
  about giving PASS 1 vs PASS 2 different WEIGHTS effectively (via routing), not about later
  passes doing iterative refinement on a fixed map, which is closer to our "plain model earns
  0.033-0.185" case than to the zero-gain slot loop.
- Q2 (scale threshold): no clean bearing - all models tested are small (16M-305M active), no
  scale sweep of routing divergence is shown (single 10^18-FLOP point), and the paper does not
  test whether divergence needs a parameter floor. Cannot confirm or refute a 3B threshold.
- Q3 (amortized design, later-pass job): direct hit on WIRING, not on amortization. Per-pass
  routing gives literally each pass a different SUBNETWORK, which is a job a single fixed map
  cannot do (a static one-pass map has one routing decision; a looped MoE core has up to T
  independent ones). It does not amortize per-span (MORPH's asked-for goal in Q3) since every
  token still goes through every pass - this is a token-loop idea, directly portable to our
  PAID-loop lane (already rejected as the shipped design) more than to the slot loop, unless
  the slot's own core pass is given per-pass routing.
- Q4 (mathematical principle): no formal treatment beyond the empirical overlap histogram; no
  bearing on saddle/attractor/power-iteration questions.

**Concrete arm to try.** MORPH already HAS a ReMoE tile router (`morph/model/routing.py`,
whole-body, route_start=30000 in base.yaml) sitting unused as a per-pass differentiator. Wire
the router to condition on loop-iteration index k (or let its natural routing vary freely
across the shared core's T passes, then MEASURE whether the tile-selection set actually
diverges pass-to-pass the way Figure 5 shows, rather than collapsing to the same tiles every
pass). Measure: (a) our own routing-overlap histogram (fraction of tiles/neurons active on
pass k vs pass 1, per token) as a direct replication of Figure 5 on our architecture; (b) the
K-curve on tokens before/after; (c) whether K3-K6 turns positive once passes are routing to
different sub-networks. Prior: MEDIUM. This targets a documented, specific cause (weight-tied
layers repeat their computation with no forced diversity) that maps cleanly onto our "pass 1
sets the scale, the rest rotate" / rank-collapse observation - a router literally forces
different active parameters per pass, which a power-iteration-style shared linear map cannot
do on its own. Caveat: their gains are on PLAIN dense tokens through a token-level loop, not a
sparse-slot-only loop; our slot loop only has ~50 slots per row touching the core, so router
load-balancing at that scale is untested and may not have enough slot-tokens per batch to
learn stable per-pass specialization.

**Contradictions.** Table 2 (RecurTrace, T=8 worse than T=2) and here Table 7 (looped models
DO beat base at matched compute) are not contradictory since methodologies differ, but this
paper does show that DENSE looping alone still helps LM perplexity at their scale (Table 7:
Looped 50.2 vs Base 55.4, a raw win with NO extra mechanism beyond weight-tying) - this
contradicts a strong reading of our brief's implicit worry that "web text never benefits from
naive looping"; it does, at their FLOPs-matched, small-model, plain-loop setting. This is a
useful counter-example to keep in mind: dense looping-without-a-trick is NOT inherently loss
here, it is our SLOT-restricted loop (only ~50/1024 positions loop) plus the per-pass targets
we tried that produced the near-zero result, not looping in general.

---

## 3. Two-Scale Latent Dynamics for Recurrent-Depth Transformers (arXiv 2509.23314, Pappone, Crisostomi, Rodolà 2025)

**What they measured and how.** NOT Huginn - a from-scratch GPT-2-style decoder-only
transformer, 12 layers, 12 heads, hidden 768, block size 512, trained on FineWeb (Section 2,
Appendix B). Evaluated ONLY on plain language modeling: perplexity (PPL) and cross-entropy
(CE) on held-out text (Appendix D); no reasoning benchmarks anywhere in the paper.
Update-norm curve (Figure 3a, Section 3): step norms `||Delta_g^(k)||_2` across recurrent
groups (their groups 4, 5-6, 7) - reported ONLY qualitatively: "loop-step sizes shrink
rapidly...typically within 5-10 steps." No exact iteration-1-vs-iteration-N norm values are
given in the extracted text (figure-only).
Update-angle curve (Figure 3b, Figure 5/Appendix C.1): cosine of the angle between consecutive
updates `cos∠(Delta^(k), Delta^(k-1))`. Reported finding: "consecutive updates become more
orthogonal: ... rises from noisy/low values and settles at lower levels (~0.5-0.65 in later
checkpoints)" - note this SETTLES AT 0.5-0.65, i.e. positive and correlated, NOT approaching 0
and NOT negative. The paper's own gloss: "non-collinear rather than repeated pushes," which
the extraction explicitly flags does not mean "becoming orthogonal" (cos->0) in the strict
sense, despite the abstract's "increasingly orthogonal" language.
Two-scale framework: within a block, updates are small, "increasingly orthogonal" refinements;
across blocks, a Drift-to-Loop Ratio (Eq. 2) `DLR = ||x_{g+1}^(0) - x_g^(K_g)||_2 / mean(||
Delta_g^(k)||_2)` is >> 1 at block hand-offs (Figure 2), i.e. the jump between recurrent
blocks dwarfs any single within-loop step.
Early-exit (Eq. 5, Algorithm 1, Appendix D): exit when second-order acceleration
`a^(k) = ||Delta^(k) - Delta^(k-1)||_2 < tau` for two consecutive steps. Figure 4: this hits
~580->360 ms/token as threshold sweeps 1e-5->1e-2, with PPL/CE "essentially flat," while a
first-order step-norm criterion degrades PPL beyond a 1e-3 threshold, and KL-based exit stays
"comparatively slow." No ablation isolating what LATER iterations specifically contribute
(paper states its own evidence is "observational and iteration-based rather than mechanistic,"
Section 5).

**Mechanism, one paragraph.** No mechanism is proposed for WHY later passes should differ from
pass 1 in a way that helps the task - the paper is purely descriptive of geometry. Its
positive claim is narrower than our brief assumed: within-loop updates get SMALL and
NON-COLLINEAR (settling around cos 0.5-0.65, not anticancelling), which the authors read as
evidence of local convex-ish refinement toward a nearby point, distinct from the coarse
between-block "drift" that does the heavy lifting of representation change. If anything, this
paper's model treats the loop as a fine-grained damping/settling process riding on top of
depth that does the real work (DLR >> 1 at hand-offs) - closer to "the loop polishes, depth
moves" than to "the loop discovers something new per pass."

**Q1-Q4.**
- Q1 (geometry disjoint from NTP): weak bearing, and possibly informative by DIFFERENCE from
  us - on their model+data, consecutive updates settle around cos 0.5-0.65 (correlated,
  shrinking, convergent), whereas WE measure cos -0.2 to -0.6 (anticorrelated, cancelling).
  If both measurements are sound, the SIGN of the pass-to-pass update correlation may be a
  free variable of training/architecture (their loop stabilizes toward small aligned nudges;
  ours flips sign every pass, i.e. genuinely oscillates/overshoots) rather than an inherent
  property of NTP objectives. Worth checking whether our anticancellation is itself a symptom
  of something fixable (e.g. overshoot from a fixed per-pass step size) rather than evidence
  the geometry is fundamentally hostile to NTP.
- Q2 (scale threshold): no bearing - only a single ~124M-scale model is tested, on plain LM,
  and gains are about early-exit LATENCY/PPL tradeoffs at fixed quality, not about whether
  looping earns net nats over a non-looped control. No matched-compute-vs-nats comparison is
  given (see "Contradictions" below), so this paper cannot speak to whether earning appears
  only above a size threshold.
- Q3 (amortized job for later passes): no bearing - the paper's own early-exit result argues
  that MOST tokens are done after very few steps (small tau already gives flat PPL down to
  360ms/token), i.e. it supports STOPPING early rather than finding a job for passes 2-6; this
  is consistent with, not contradictory to, our "pass 1 does everything" finding, but offers
  no design for making later passes useful - only for skipping them cheaply.
- Q4 (mathematical principle): no bearing - explicitly observational, no attractor/saddle
  formalism, though the DLR construct (a local-vs-global-motion ratio) is a reusable
  INSTRUMENT (not a principle) we could apply to our own within-pass vs prelude->coda drift.

**Concrete arm to try.** Reuse their DLR instrument directly: compute
`||coda_entry - slot_exit_after_pass_T||_2 / mean_k(||slot_state_update_k||_2)` for our slot
loop, i.e. is the write-out from the loop to the coda ("the hand-off") large relative to a
typical within-loop step, or is the loop itself already the dominant motion (in which case
"drift vs refinement" doesn't even apply cleanly to a system with no depth before the loop)?
Measure: the DLR number itself, and re-examine our existing cos(-0.2,-0.6) result through
their lens - is the cancellation happening between SMALL steps (matching their "increasingly
orthogonal/small" story, just with a sign flip) or LARGE ones (a genuinely different,
oscillatory regime)? Prior: LOW as a fix, MEDIUM-HIGH as a diagnostic - this costs one cheap
instrument and could tell us whether our loop's cancellation is "healthy convergence with
sign noise" or "actual overshoot," which changes whether a per-pass step-size schedule
(explicitly ruled in-scope by their first-order early-exit ablation) is worth trying.

**Contradictions.** The abstract's "increasingly orthogonal" is over-strong relative to the
body text's own number (cos settles at 0.5-0.65, i.e. correlated, not orthogonal at cos~0);
flag this discrepancy explicitly rather than repeat the abstract's framing. Also worth noting
for us: their loop SHRINKS and becomes small/correlated (a convergent, stable process) while
ours reaches a fixed point via CANCELLING (anticorrelated) updates - these are different
dynamical signatures, so "loop reaches a fixed point" is not by itself informative; the PATH
to the fixed point differs and may matter more than the fact of convergence.

---

## 4. A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791, Blayney et al. 2026)

**What they measured and how.** Three models (Section 4.1, Table 1/Appendix B): Ouro 1.4B
(Zhu et al. 2025), Huginn-0125 (Geiping et al. 2025), and a Llama retrofitted with recurrence
(McLeish et al. 2025), plus randomly-initialized 12-layer models for controls. Data: mostly
reasoning - a 256-example subset of the GSM8K test set (Section 4.1) for the main cyclic
fixed-point measurements, plus HellaSwag as a "non-reasoning" control (Section E.4); Section
5.1 also trains small models from scratch on an unspecified "simplified loss function." This
paper does NOT run a general held-out plain-LM-perplexity evaluation as its primary
instrument - GSM8K-256 dominates, so it sits mostly on the reasoning-benchmark side despite
"mechanistic" framing.
Cyclic fixed point (Proposition 4.1; Figures 3-4): measured as the norm of the difference
between the residual stream after successive re-applications of the SAME shared layer across
loop iterations, compared against an "approximate fixed point" computed at iteration 128.
Convergence is shown only visually (norm curves declining over ~8-128 iterations); NO explicit
numeric threshold (e.g. "converged when norm < X") or exact convergence-iteration count is
given in the extracted text - "Huginn-0125 and retrofitted Llama quickly reach a fixed point,
Ouro does not" (Fig. 4 caption) is qualitative.
"Stages" (Section 2.2, Figs. 7-8): identified via a metric called ColSum Concentration,
`C = 1 - H_col` (normalized entropy of attention-column sums, i.e. an attention-sink /
concentration measure), NOT via probing or logit-lens. Figure 7: "individual layers quickly
converge towards constant behavior" while "cyclic action results in cyclic stages of
inference" - i.e. within one loop iteration, the shared block's internal layers traverse the
SAME sequence of concentration values every iteration, and that sequence resembles the
depth-wise sequence a real feedforward stack would show (Fig. 8: Ouro, trained from scratch
with a loop, "mirrors Llama mixing stages" despite never seeing Llama's depth-wise weights).
Non-convergent cases (Section 4.2, Fig. 6, Appendix C): Ouro 1.4B is the standout
non-convergent model; the paper ties this to the ABSENCE of input injection - "input injection
results in stable fixed point behavior for all norm types other than Ouro," and "pre-norm
without input injection reaches a degenerate fixed point where each layer converges to the
SAME point" (collapse, not a useful cyclic fixed point). No per-token/per-layer percentages
are given (qualitative assessment across GSM8K and a "Long Persona / Short Math" dataset pair
in Appendix C.1).
Performance vs iteration count (Section 5.2): the paper explicitly does NOT provide its own
accuracy-vs-iteration curves; it states models with stable cyclic fixed points "avoid
performance deterioration" at test-time iteration counts beyond training, CITING Geiping et
al. and Zhu et al.'s own papers for that claim rather than measuring it here.

**Mechanism, one paragraph.** The core claim inverts the naive picture: later passes are NOT
where new computation happens - by iteration ~8-128 the shared block has settled into a
CYCLIC fixed point (a fixed TRAJECTORY of K distinct states that the block maps onto itself,
one per position IN the cycle, not one single point), and further iterations just repeat that
cycle. The useful work, per this paper, is done by the FIRST few iterations establishing which
cyclic orbit the state lands on (gated by input injection, which reinjects the original input
at every iteration and prevents collapse to a degenerate single-point fixed point); once the
orbit is set, iteration count beyond that is redundant by construction, matching a
feedforward network's fixed sequence of layer-stages mapped onto loop-time instead of
depth-time. This directly predicts "pass 1 (or the first few) does everything, later passes
do nothing new" as the EXPECTED and ARCHITECTURALLY CORRECT behavior of a well-formed loop,
not a failure mode.

**Q1-Q4.**
- Q1 (geometry disjoint from NTP): no direct bearing - evaluated on GSM8K/HellaSwag, not on
  plain NTP loss; but the mechanism itself (cyclic fixed point = a stable finite orbit) has no
  stated dependence on the training objective, so it likely applies to NTP-trained loops too
  and is consistent with (arguably PREDICTS) our finding.
- Q2 (scale threshold): no bearing - Ouro is 1.4B and IS the non-convergent, degenerate case;
  Huginn (3.5B) converges cleanly; no controlled scale sweep isolates size as the variable
  (the three models differ in architecture/training, not just scale), so this cannot support
  or refute a parameter threshold for depth-earning.
- Q3 (amortized job for later passes): the strongest and most uncomfortable hit in this batch.
  This paper's evidence, taken at face value, says our observed result (pass 1 does
  everything, passes 2-6 add ~0) is not a symptom to fix but the SIGNATURE of a properly
  functioning weight-tied loop that has found its cyclic fixed point - meaning "give later
  passes a job a one-pass map cannot do" may be structurally impossible for ANY shared-weight
  loop once it converges, UNLESS the per-iteration input changes (their "input injection")
  in a way that keeps moving the orbit itself, which is a different mechanism from ours (we
  do not re-inject a changing signal into the slot loop each pass beyond what the state already
  carries forward).
- Q4 (mathematical principle): direct hit. This is the one paper that gives Q4 a concrete
  answer with a name: a weight-tied iterated map with a fixed input does not need to keep
  iterating once it reaches a CYCLIC attractor (a periodic orbit of the map, of which a true
  fixed point is the period-1 special case); iteration is only "required" (in their sense of
  producing new per-iteration state) while the orbit is still forming, i.e. during the
  transient, not after. Our own "rank collapses 181->9, power iteration, cosine cancellation"
  read exactly as symptoms of settling into such an orbit (or a degenerate period-1/period-2
  collapse, given our cosine goes NEGATIVE, i.e. we may be closer to their "degenerate fixed
  point without input injection" failure than to a healthy cyclic orbit).

**Concrete arm to try.** Test their INPUT-INJECTION prescription directly on the slot loop:
re-inject a signal into the slot state at EVERY iteration (not just at entry) that is not
already derivable from the current state - e.g. re-read the span's bag/E_slot embedding fresh
each pass (already partly true via `bag_mean`/`E_slot`, but confirm it is re-added each pass
rather than only seeding iteration 1), or inject a per-iteration positional/iteration-index
signal, and check whether this breaks the cosine-cancellation pattern and moves K3-K6 off
zero. Measure: (a) whether removing/adding input injection changes the sign/magnitude of
consecutive-pass cosine similarity (does it become positive/small like paper 3's healthy case,
or does it stay negative like our current "degenerate" case); (b) K1-K6 and specifically
K3-K6 on tokens; (c) rank of the per-pass update (does 181->9 collapse loosen). Prior: MEDIUM.
This paper's own within-domain evidence (Fig. 6) is that input injection is the SPECIFIC fix
that separates Ouro's degenerate collapse from Huginn/retrofitted-Llama's healthy cyclic
fixed point, and our own diagnostics (negative cosine, extreme rank collapse) pattern-match
closer to their described degenerate case than their healthy one - this is the single most
directly actionable, evidence-backed lever in this batch of four papers.

**Contradictions.** This paper's Section 5.2 claim - that reaching a stable cyclic fixed point
is GOOD and "avoids performance deterioration" - is in tension with our brief's framing of
fixed-point convergence as evidence of a WASTED loop. Both can be true simultaneously (a
converged loop is stable/safe AND has nothing left to contribute past the transient), but the
paper does not treat "no further gain" as a problem to solve, only "instability" as one. We
should not borrow their optimism about convergence as evidence our loop is healthy - our
degenerate-looking signature (negative cosine, rank->9) more closely resembles their FAILURE
case (Ouro, no input injection) than their success case.

---

## Synthesis for Q1-Q4

Q1: no paper tests loop-vs-NTP disjointness directly. Paper 2 shows plain looping DOES help
FineWeb perplexity at matched FLOPs with no trick (Table 7), so NTP does not categorically
reject depth - the zero result is closer to our specific slot-restricted design than to NTP
itself. Q2: no paper isolates scale as the sole variable for depth-earning; paper 1 shows
loop-memory NLL gains SHRINK with scale (0.6B: -0.15 nats to 88B: -0.02), the opposite of a
"needs 3B+" story, though on a different metric. Q3, the productive lane: paper 1's loop-time
self-attention and paper 2's per-pass routing both give later passes a structurally distinct
COMPUTATION (a different history-read or a different subnetwork) rather than relying on the
per-pass TARGET to create difference, which is what all our failed arms tried. Q4 gets its
sharpest answer from paper 4: a weight-tied map with a static input needs to iterate only
during its TRANSIENT into a cyclic fixed point (period 1 = ordinary fixed point); once there,
further passes are correctly inert by construction, and their good-vs-degenerate split
(input-injection present vs absent) maps disturbingly well onto our own signature (negative
consecutive-pass cosine, rank collapse 181->9) resembling their DEGENERATE case, not their
healthy one. Two concrete, differently-flavored arms to run next: (a) per-pass ReMoE routing
on the slot core (paper 2, medium prior, direct instrument = routing-overlap histogram); (b)
continuous input re-injection into the slot loop at every iteration, not just entry (paper 4,
medium prior, direct instrument = does cosine flip sign / does rank loosen). Both are cheap,
both attack "give pass k something pass 1 structurally lacks" rather than another loss target,
and (b) is backed by the most literal match to our failure signature of any paper read.
