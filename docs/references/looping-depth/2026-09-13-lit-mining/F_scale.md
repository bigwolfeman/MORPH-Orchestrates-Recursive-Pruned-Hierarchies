# Theme F: Scale — does loop depth-earning need a parameter threshold?

Read against the MORPH/TUL brief (`loop_lit_brief.md`, 2026-09-13). Numbers below come from
WebFetch summaries of arXiv HTML/abstract pages (a small extraction model), not from reading
the raw PDF myself. I flag anywhere a claim looked internally inconsistent or worth
double-checking against the source before it drives a decision.

---

## 1. Schwethelm, Rueckert, Kaissis — "How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models" (arXiv 2604.21106)

**What they measured.** A joint scaling law over width (7 scales, s ∈ {6,8,...,18}), recurrence
count r ∈ {1,2,4,8}, and training tokens D, at 6 iso-FLOP budgets from 4.64e17 to 2.15e19 FLOPs
(0.45B–47B tokens). Unique-parameter counts range from ~10.9M (s=6, r=8) to ~403M (s=18, r=1)
(Table 6, Appendix E). No downstream tasks; the instrument is pretraining loss itself, fit as

    L(N_once, N_rec, D, r) = E + A(N_once + r^φ · N_rec)^(-α) + B·D^(-β)   (Eq. 3, Sec 3.3)

φ is the recurrence-equivalence exponent: the effective parameter count of a looped model is
N_once + r^φ·N_rec, so φ<1 means each extra recurrence is worth less than a truly-separate
block, and φ=0 would mean a recurrence is worth nothing at all beyond a one-time parameter cost.
Baseline fit: **φ = 0.459, 95% CI [0.41, 0.53]** (Table 2, Sec 4.3). Worked example: a 410M
looped model (r=4) matches a 580M non-looped model in loss, at the training-compute cost of a
1B non-looped model (Abstract; Appendix F.4).

**Mechanism claimed.** None, by design. This is a curve-fit, not a mechanistic study — it treats
recurrence as a capacity knob and measures its exchange rate against real parameters. It does not
ask what a later pass computes.

**Smallest scale where a recurrence is worth anything measurable.** The CI on φ never includes 0
anywhere in their grid: "No resample reaching φ=0 or φ=1" (Sec 4.3). Splitting the grid in half,
low-budget half (C ≤ 2.15e18) gives φ=0.44, high-budget half (C ≥ 4.64e18) gives φ=0.49, both
inside the global CI (Sec 4.3, "Robustness"). An extrapolation to s=34 (~4e20 FLOPs, ~20x the top
grid point) still tracks the r=4 band within [0.05, 0.08] nats (Sec 4.2). So the smallest scale
they tested (~11M unique params at r=8) already shows a statistically nonzero, non-vanishing φ,
and it stays flat out to the largest scale they extrapolate to. Two named interventions move φ
directly: truncated BPTT drops it to 0.38 (a training-side tax), hyperconnections raise it to
0.65 (Sec 4.3) — offered as a validation that the method separates architecture wins from
training-side costs.

**Bearing on Q1–Q4.**
- Q1 (geometry too disjoint from NTP): no bearing — no representation-geometry analysis.
- Q2 (parameter threshold): **direct bearing.** φ is nonzero and roughly scale-invariant from
  ~11M unique params to a ~4e20-FLOP extrapolation, with no threshold effect and no φ→0 anywhere
  tested. For a Parcae-style full-sequence loop, "worth something" is present at scales an order
  of magnitude below MORPH's 330M. This is evidence against a hard parameter floor — but it is
  evidence about a loop where EVERY position recurs; it says nothing about a loop restricted to
  ~50/1024 slot positions, which is TUL's actual knob.
- Q3 (amortized design): no bearing — no partial/sparse recurrence variant tested.
- Q4 (when must an iterated map iterate): no bearing — no dynamical-systems argument, pure
  capacity accounting.

**Concrete arm for us.** Fit our own iso-FLOP φ curve, but on the instrument we already use
(K1-K6 CE, not raw val loss), holding total training FLOPs fixed and sweeping (width, slot-loop
recurrence count) at 2-3 scales below and at 330M. If φ for the SLOT-only loop comes out
indistinguishable from 0 while their whole-sequence loop sits at 0.46 at similar unique-param
counts, that isolates the failure to "only slots recur" rather than to scale. Measure: fit
L = E + A(N_once + r^φ·N_slot)^-α + B·D^-β against CE at a small width/recurrence grid; report φ
and its CI. Prior: I expect φ to come out much lower than 0.46, plausibly CI-including-0, because
N_slot here is exercised on ~5% of positions per row, not all of them — this paper's whole
argument assumes the recurrent block sees the same work an extra transformer block would, which
is not true of TUL's slot loop.

**Contradictions with the brief.** None directly (different loop shape), but it undercuts a
scale-only explanation for OUR result: if scale alone gated depth-earning, a Parcae-style full
loop this small (~11-35M unique params) should show nothing either, and it doesn't.

---

## 2. Zhu et al. — "Scaling Latent Reasoning via Looped Language Models" (Ouro, arXiv 2510.25741)

**What they measured.** 1.4B and 2.6B LoopLMs, 7.7T total training tokens (Stage 1a/1b 6T,
Stage 2 CT annealing 1.4T, Stage 3 LongCT 20B, Stage 4 mid-training 300B; Table 1, Sec 4.2).
Entropy-regularized adaptive-depth training objective (Eq. 4, Sec 3.3) that weights a per-step
loss by exit probability, with a uniform-prior KL term to stop the router from collapsing to max
depth. **No pretraining-loss-vs-loop-count curve is reported anywhere in the paper or appendix**
— I searched explicitly and found none. What IS reported is downstream accuracy at fixed
recurrent step counts T=1..8 (Tables 10, 11):

| Model | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| 1.4B MMLU | 41.21 | 60.43 | 66.71 | 67.45 |
| 2.6B MMLU | 51.55 | 67.63 | 73.57 | 74.60 |

Gains: T1→T2 is huge (+19.2 / +16.1 points), T2→T3 smaller (+6.3 / +5.9), T3→T4 nearly flat
(+0.7 / +1.0). Same diminishing-returns shape as our own pass-1-does-everything finding, but on
downstream accuracy, not on loss, and at 1000x our parameter count.

A smaller-scale synthetic battery exists (Section 6, "Physics of LoopLMs") using **1M–40M
parameter** GPT-2-style models on two constructed tasks: Capo (knowledge storage — capacity
unaffected by looping, ~2 bits/param regardless of loop count, Fig. 6 left) and Mano (knowledge
manipulation — looped models beat iso-parameter non-looped baselines across difficulty levels
L∈{10,16,24}, Fig. 6 right; multi-hop QA needs fewer samples with more loops, Fig. 7 right). This
is NOT a pretraining-loss ablation and not web text.

**Mechanism claimed.** Looping does not add storage capacity; it improves "flexible usage of
knowledge already stored" (Sec 6, capacity vs. manipulation split) — later passes recombine
information the model already has, rather than accessing new information. The entropy-regularized
depth objective is a training-stability device, not itself a mechanism for why a later pass helps.

**Bearing on Q1-Q4.**
- Q1: weak, indirect bearing. The Capo/Mano split is the closest thing in this batch to
  evidence that a plain LM objective (predict-the-answer, here multi-hop QA, not raw web-text
  NTP) CAN produce an iteration-useful map for "manipulation" tasks specifically, while it never
  helps "storage" tasks. It's a different objective from web-text NTP, so it bears only by
  analogy.
- Q2: bearing but confounded — no controlled small-scale pretraining-loss test exists at their
  main (1.4B/2.6B) recipe, so no threshold claim is testable from this paper alone. The 1M-40M
  synthetic manipulation gains argue loosely against a large threshold, but on synthetic tasks,
  not web text.
- Q3: no bearing — every token loops in Ouro, no amortization scheme.
- Q4: no bearing — no dynamical-systems account.

**Concrete arm for us.** Build a Capo/Mano-style synthetic multi-hop probe (an explicit 2-span
fact-chaining task) mixed into our web-text data, and read the SLOT loop's K1-K6 on the probe
specifically, not just on ordinary tokens. Measure: same K-curve instrument, restricted to
probe positions. Prior: low-moderate (I'd guess ~25% this moves), because Ouro's own split never
isolates PASS-LEVEL contribution the way our K-curve does — their manipulation gain could be
carried entirely by pass 1 too, and this paper gives us no reason to expect otherwise.

**Contradictions with the brief.** None. The shape of their downstream-accuracy loop-count curve
(all the gain by T=2-3, flat after) is independent, cross-domain, 1000x-larger-scale support for
the brief's "pass 1 does everything" pattern — an agreement, not a contradiction.

---

## 3. Rauba, Fanconi, van der Schaar — "Tiny Autoregressive Recursive Models" (arXiv 2603.08082)

**What they measured.** Character-level algorithmic tasks (Copy, Reverse, Addition), tiny models
(paper never states exact parameter counts; it references the original TRM's 7M-param scale on
ARC-AGI but not its own model sizes), evaluated under a matched block-pass compute budget across
a 7-point architecture ladder from Dense Transformer to full Autoregressive TRM (Table 1):

| Variant | Compute placement |
|---|---|
| Dense Transformer | distinct params per layer, no tying |
| Iterative Transformer | tied params across recurrent applications |
| Iterative Step Transformer | tied + step embeddings |
| Universal Transformer (UT) | tied recurrence + ACT halting, weighted readout |
| Dual UT ("flat two-stream") | two streams (solution Y, auxiliary Z), flat updates |
| Dual Nested UT | multiple Z refinements per Y update (nested) |
| Autoregressive TRM | nested + binary halting + **terminal-iterate-only readout** |

**Matched-compute results** (Figure 3, Sec 5.1), character accuracy at length 10:

| Task | Dense Transformer | UT | Autoregressive TRM |
|---|---|---|---|
| Copy | 100% | 100% | 12% |
| Reverse | 100% | 100% | 11% |
| Addition | 80% | 66% | 10% |

The full ATRM is "close to chance-level" on every task (Sec 5.1). Untied depth (=Dense) and flat
two-stream recurrence (=Dual UT) are named explicitly as the strongest generalizers per block
under matched compute (Sec 5.3 quote: "untied depth and a flat two-stream recurrent baseline
yield the strongest generalization per block evaluation").

**Mechanism claimed.** Not compute AMOUNT but compute PLACEMENT decides whether gradient can
reach a task's real dependency structure. Addition (and, catastrophically, everything for the
ATRM) requires carrying state across the whole sequence — a "final-dependency barrier." Dense and
Dual UT expose every intermediate computation to the loss; Dual Nested UT and full ATRM collapse
multiple refinement steps down to reading only the TERMINAL iterate, and that collapse is what
breaks learning on tasks needing cross-position carry (Sec 5.3, "Takeaway").

**Bearing on Q1-Q4.**
- Q1: real bearing, off web text. It shows a target CAN be learnable by a flat/untied
  architecture and simultaneously UNLEARNABLE by a nested/terminal-readout one at the identical
  compute budget and identical task — i.e. the failure the brief describes (pass 1 does
  everything, gradient can't reach later passes) can be a WIRING artifact rather than a
  fundamental mismatch between iterative maps and a training objective.
- Q2: no bearing — one tiny scale, no scaling-law claim.
- Q3: bearing — "flat two-stream" and "untied depth" are concrete alternatives to nesting that
  still separate compute without collapsing it; a candidate wiring family distinct from TUL's
  design.
- Q4: light bearing — an empirical (not formal) instance of the brief's own claim: a
  deterministic map whose output is read ONLY at the terminal step, with intermediate steps
  invisible to the loss, starves gradient on tasks needing cross-position information; this
  matches the "RELAY vs EXTRACTABILITY" framing but from the opposite failure mode (relay
  blocked by the readout, not by the map itself).

**Concrete arm for us.** TUL's z is read ONLY at its terminal pass, through 2 prefix cells — this
is structurally the ATRM's terminal-iterate-only readout, the worst performer in this paper's
ladder. Try a flat readout instead: write EVERY pass's slot state (or a running sum/concat) into
the prefix cells, not only the final one, mirroring Dual UT's flat two-stream exposure. Measure:
K1-K6 on tokens, and whether "one pass does everything" changes when all 6 passes are individually
visible to the coda rather than collapsed into one terminal z. Prior: the brief already reports a
"4-cell register" lever that did not help, which is the closest existing test of this idea and it
was null — so I'd put this at ~20%, worth a cheap run, not a high-confidence bet.

**Contradictions with the brief.** None; it reinforces the brief's fitted-z / info-theoretic
argument with an independent, cross-domain instance of nested terminal readout starving gradient.

---

## 4. Prairie, Novack, Berg-Kirkpatrick, Fu — "Parcae: Scaling Laws For Stable Looped Language Models" (arXiv 2604.12946)

Our core is explicitly Parcae-style, so this is the most load-bearing paper in the set.

**What they measured.** Transformers at 100M (one comparison table only), 140M, 370M, 770M, 1.3B
params. Main comparison (Table 5) at optimal training-time recurrence T=8:

| Scale | Val PPL (Transformer, T=1) | Val PPL (Parcae, T=8) | Core (T=1) | Core (T=8) |
|---|---|---|---|---|
| 140M | 21.48 | 19.06 | 13.00 ± 0.15 | 14.04 ± 0.20 |
| 370M | 15.79 | 14.49 | 17.46 ± 0.03 | 20.00 ± 0.06 |
| 770M | 13.08 | 12.49 | 22.42 ± 0.20 | 25.07 ± 0.33 |
| 1.3B | 11.95 | 11.42 | 25.45 ± 0.08 | 28.44 ± 0.28 |

**IsoFLOP grid at fixed compute** (Table 6) is noisier than the headline table suggests. At 140M:

| FLOPs (×1e18) | optimal μ_rec | Fixed-depth Core | Looping Core |
|---|---|---|---|
| 11 | 2 | 7.6 | 7.9 |
| 22 | 2 | 9.0 | 10.5 |
| 44 | 4 | 11.2 | **10.7 (looping loses)** |
| 88 | 6 | 10.5 | 11.8 |
| 128 | 8 | 14.6 | **13.0 (looping loses)** |
| 64 | 10 | 16.2 | 15.0 |

At 370M: 32e18 (μ_rec=4) looping wins 16.8 vs 15.2; 64e18 (μ_rec=6) it's a tie, 18.1 vs 18.1;
128e18 (μ_rec=6) looping loses, 18.1 vs 20.1. So even at Parcae's own smallest scale, looping
does NOT beat fixed-depth at every FLOPs/recurrence-count pairing — the "optimal μ_rec" fit
(power laws γ_μ≈0.40, γ_D≈0.78, Fig. 5) is what recovers a net win, not every raw grid point.
Test-time extrapolation goes to T=24 (Sec 5.3).

Parameter/token threshold: none identified. Training tokens up to 100B at 1.3B scale (Table 5).
Section 6 (limitations, verbatim): "our observations are limited to small architectures. It
remains to be seen if Parcae compares favorably when scaling these observations to large FLOP
budgets and parameterizations."

**Mechanism claimed.** Purely a stability fix, not a "why does a later pass help" argument. Prior
looped architectures suffer residual-state explosion because their injection-parameter spectral
radius ρ(Ā) is marginally stable or unconstrained (Sec 3). Parcae reparameterizes the recurrent
update as a discretized negative-diagonal linear system, Ā = exp(Δ⊙A) with A = -diag(exp(log_A)),
guaranteeing ρ(Ā)<1 (Sec 4.1), plus per-sequence (not per-batch) depth sampling to reduce loss
spikes (Sec 4.2, Appendix G). This makes the loop trainable/well-posed; it does not argue what
work a 4th or 8th pass computes that a 2nd cannot.

**Bearing on Q1-Q4.**
- Q1: no bearing — no representation-geometry claim.
- Q2: **direct and important.** Parcae's own smallest scale, 140M — below MORPH's 330M — already
  shows a measurable, positive (if operating-point-dependent) loop gain at its optimal μ_rec: an
  11% relative val-PPL drop and a +1.04 Core-score gain (Table 5). This is an existence proof of
  depth-earning well under 1B params on ordinary pretraining loss, with a whole-sequence
  Parcae-style loop — the same core MORPH implements. It weighs against a hard "needs ~3B"
  threshold. But the isoFLOP grid (Table 6) shows the SAME 140M model losing to fixed-depth at 2
  of the 6 tested budget/μ_rec pairs, so scale is not the only gate — the (compute, recurrence
  count) OPERATING POINT matters and can flip the sign even at fixed scale. This matches the
  brief's own observation that our plain (non-slot) core earns 0.033-0.185 nats DEPENDING ON
  ENTRY (prelude vs noise) — an operating-point sensitivity, not evidence of a scale floor.
- Q3: no bearing — full-sequence loop, no amortization axis tested.
- Q4: light bearing, negative-direction only — ρ(Ā)<1 is a NECESSARY condition for the loop to
  be trainable as a map at all (else it diverges); it says nothing about whether a stable,
  well-posed loop then has anything left to do past pass 1, which is exactly our failure mode.
  Stability and earning are separate axes, matching the brief's own "Branch (a) closed" finding.

**Concrete arm for us.** Run Table 6's methodology on our PLAIN (non-slot) core specifically: an
iso-FLOP grid over training-time recurrence mean (not just our fixed mean-6) at a smaller
scale (~100-140M unique params, matching Parcae's own floor), and check whether an optimal μ_rec
beats fixed-depth-1 at matched FLOPs the way Table 6 does. Measure: val CE at matched FLOPs
across a μ_rec grid, not just K1-K6 at a single fixed μ_rec. Prior: moderate-high (~60%) that
this recovers a Parcae-shaped win at SOME μ_rec, because the brief already shows our plain core
earns a nonzero, entry-dependent amount — consistent with Parcae's qualitative story. This is
orthogonal to the slot loop's own failure: it would confirm our BASE architecture is fine and
narrow the fault further onto the slot loop's specific target/wiring.

**Contradictions with the brief.** None directly, but the isoFLOP grid's own internal noise
(looping losing at 2/6 budget points even at 140M) is a caveat worth carrying forward: "isoFLOPs
favor looping" is a property of the FITTED optimum, not a guarantee at an arbitrary
compute/recurrence-count pair, even in the paper our core is modeled on.

---

## 5. Wang et al. — "SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers" (arXiv 2609.01343)

**What they measured.** MoE looped transformers, active-parameter grid from 100M to 1.6B
(Fig. 8), each SMELT model repeating the MIDDLE HALF of its layers a second time, matched against
a dense-FLOP-equivalent MoE baseline on per-token FLOPs, total params, and memory. Compute-matched
scaling law (Eq. 2, Sec 4.2):

    L(F,S,D) = E + A(1-S)^b / F^a + K/D^c

with compute-optimal frontier L*(C,S) = E + f(...)·C^(-γ), γ = ac/(a+c). Fitted: baseline
a=0.3703, c=0.6594, γ=0.237; SMELT a=0.3892, c=0.7011, γ=0.250 (Sec 4.3). Loss/FLOPs-savings gain
at equal loss ranges **6.8%-18.0%**, and it GROWS with compute budget rather than shrinking:

| Compute budget | S≈97% | S≈95% | S≈85% |
|---|---|---|---|
| 1e20 FLOPs | 6.8% | 7.8% | 10.0% |
| 1e21 FLOPs | 14.7% | 15.8% | 18.0% |
| 1e22 FLOPs (extrapolated) | 19.6% | 20.9% | 23.5% |

(Table in Sec 4.3.) Recipe ablations at **200M active params** (Sec 3.3-3.5) already show SMELT
beating baseline on validation loss (1.9257 vs 1.9445 at S≈85%) — below MORPH's 330M — and the
full grid spans 100M-1.6B with the advantage holding at every point on the cosine-decay frontier
(Fig. 8, Sec 4.1).

**Mechanism claimed.** The second visit through the repeated layers reduces attention-sink mass
(concentration on early/BOS-like tokens) and redirects it to content-relevant tokens (Sec 6.2,
6.4). A Dyck-language case study: "the attention sink nearly vanishes on the second visit" (Sec
6.4). Q and K representations stay similar across visits and the two visits attend to nearly the
same tokens, but the attention VALUES/weights diverge — a re-weighting account, not a
re-representation account.

**Bearing on Q1-Q4.**
- Q1: partial, useful bearing. Sink-reduction is a concrete, well-specified job a second pass can
  do that a single pass architecturally cannot (you can't remove a sink you haven't already
  formed once). This is real evidence AGAINST "NTP never produces an iteration-useful map" — at
  least for attention allocation, ordinary next-token loss on ordinary text does produce a
  second-pass-usable structure.
- Q2: bearing, and against a threshold story specifically. Gain is present and measurable at
  200M active params and then GROWS smoothly with compute rather than switching on above some
  floor (Sec 4.3 table above). This argues scale sets the MAGNITUDE of the gain, not its
  existence — there is no evidence of a hard floor near 330M or above.
- Q3: bearing, but on a different axis than TUL. SMELT repeats a subset of LAYERS for every
  token (an amortization-adjacent design, but along depth, not position). TUL repeats ALL layers
  for a subset of POSITIONS. Both are "compute placement" ideas, but SMELT's result does not
  transfer directly to "can restricting the loop to specific token positions still earn."
- Q4: no formal bearing, but the sink-reduction mechanism is a clean real-world instance of the
  brief's own "EXTRACTABILITY" disjunct: the second pass adds no new information, it puts
  information already present into a form the reader (later attention) can use more of, by
  removing competing mass. That is exactly the brief's Lean-checked framing, independently
  observed in a different model family.

**Concrete arm for us.** Borrow SMELT's instrument directly: measure attention-mass-on-first-slot
(or first-token, sink-analog) per slot-loop pass, comparing pass 1 to pass 6. If mass on a sink
position drops loop over loop even though CE (K1-K6) is flat, that would mean the loop is doing a
real extractability edit our CE-based instrument can't see. Measure: per-pass attention-mass
histogram on the earliest slot/token vs. content positions. Prior: low-moderate (~30%) — the
brief's rank-collapse and cosine-cancellation findings already suggest passes 2-6 do close to
nothing by several independent instruments, so a null result here would not be surprising, but
this is a genuinely different lens and cheap to run.

**Contradictions with the brief.** None directly. Its "gain grows with, rather than switches on
above, scale" finding complements Parcae's 140M existence-proof: two independent papers, two
different loop designs, both show measurable gain well under 1B params with no floor effect.

---

## Synthesis for Q1-Q4 (answer to Wolfe's question)

**Direct answer: no, published evidence does not show a parameter threshold for loop
depth-earning on web-text next-token loss, and 330M sits inside the range where three
independent papers already report a nonzero effect.** Parcae (paper 4) beats fixed-depth at
140M params (val PPL 21.48→19.06, +1.04 Core; Table 5) with a whole-sequence loop of the same
family MORPH implements. SMELT (paper 5) shows a compute-matched gain at 200M active params
(1.9257 vs 1.9445 val loss; Sec 3.3-3.5) that then GROWS with scale (6.8%→18.0% FLOPs-saved,
1e20→1e21 FLOPs; Sec 4.3) rather than switching on above a floor. Schwethelm et al. (paper 1)
fit a nonzero recurrence-equivalence exponent (φ=0.46, CI excludes 0) from ~11M unique params
up through a ~4e20-FLOP extrapolation, with no scale dependence in the exponent itself (Sec
4.2-4.3). None of these three papers found or looked for a lower bound; where they tested small,
the effect was already there.

What none of them share with TUL is the WIRING: all three loop every position through every
pass. The paper that DOES vary compute placement, TARM (paper 3), found the failure mode that
best matches ours: a nested, terminal-iterate-only readout is the one design on its ladder that
cannot learn cross-position dependencies at ANY scale, while flat/untied designs at the identical
tiny scale succeed (Fig. 3, Sec 5.3). TUL's z-into-2-prefix-cells design is structurally that
terminal-only readout.

So the evidence points away from Q2 (scale) and toward Q3/wiring: try exposing every pass's slot
state to the coda (a flat readout, TARM's "Dual UT" analog) rather than only the terminal z, and
separately probe for an SMELT-style extractability effect (attention-sink shift per pass) that a
CE-only K-curve would miss. Neither guarantees a fix; both are cheap, targeted, and matched to a
real mechanism found in this literature rather than to scale.

*(299 words)*
