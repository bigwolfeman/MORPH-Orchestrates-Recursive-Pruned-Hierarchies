# Lit review A: theory of when an iterated map must iterate

Read against the brief in `loop_lit_brief.md`. All four papers fetched in full (arXiv HTML,
`arxiv.org/html/<id>`) on 2026-09-13; no truncated fetch, no section skipped for length.

---

## 1. Zhang, "Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation" (2605.30757)

### What they measured and how

This is mostly a complexity-theory paper, not a training study, plus two small controlled
sweeps. No model-size or parameter count is reported anywhere; the sweeps use toy
transformers on synthetic tasks.

Three idealized reasoners, each with a persistent-memory budget M (Def 3.1-3.3, Sec 3):
- Compressed latent loop `CLL[s,d,p,T]`: s persistent slots, M = s·d·p bits. The input
  positions are read-only and never materialize a hidden sequence.
- Sequence-state loop `SSL[d,p,T]`: a hidden vector at every one of n input positions,
  M = n·d·p bits.
- CoT scratchpad `CoT[d,p,l]`: M = Θ(l log|Σ|) bits, since generated tokens are appended
  to context.

Theorem 4.2: `CLL[s,d,p,T] ⊆ DSPACE(O(s·d·p + log n + log T))`. Running the loop longer
(T) adds only a `log T`-bit counter to the space bound, not more memory (Lemma 4.1,
proof intuition p.6). Proposition 4.3: `SSL[d,p,T] ⊆ DSPACE(O(n·d·p + log n + log T))`,
i.e. Θ(n) space — "memory-rich." Theorem 4.4 (conditional on the standard, unproven
assumption `P ⊄ DSPACE(polylog n)`): no compressed-loop family with `s·d·p =
polylog(n)` and `log T = polylog(n)` decides any P-complete language, while
polynomial-length CoT can. This is an **expressivity** bound — what a family of such
machines can compute in principle, for any T — not a claim about what gradient descent
finds (the paper says this explicitly in §6: "success is not automatic... the remaining
failures are mostly about whether training finds a usable routing scheme").

Empirical sweeps (§5, seed tables in Appendix F, no run excluded):
- Pointer chasing (§5.1, Fig 1): n=16 cells, k∈{1,2,4,8} concurrent chains to track,
  compressed-loop slot count s∈{1,2,4,8,16}, T=16 iterations fixed, 10 seeds. Subdiagonal
  regime s<k: mean cell accuracy 0.041, max cell-mean 0.129, std 0.00-0.02 for
  k∈{2,4,8} — a robust, low-variance collapse. Sequence-state control (SSL, M=Θ(n)):
  ≈1.00 for k=2,4,8; 0.936 for k=1 (9/10 seeds ≈1.00, one optimization failure). Above
  the diagonal, some cells are seed-dependent (e.g. k=8,s=16 ranges from near-failure to
  near-perfect) — described as a training/routing issue, not a capacity issue.
- Associative recall (§5.2, Fig 2, Appendix F): N∈{8,16,24} key-value pairs, gated
  linear attention with state dim m swept, 5 seeds. `m_crit` (smallest m reaching 0.9
  accuracy): N=8 median 8 (range 4-16), N=16 median 32 (range 16-64), N=24 median 64
  (range 32-128). Full softmax attention saturates (≥0.999) everywhere; vanilla linear
  attention stays low for N=16,24 even at the best m in the sweep.

There is no matched-compute control in our sense (FLOPs-per-token vs depth); the closest
analogue is T held fixed at 16 while s varies — the opposite axis from our "same
block-passes, no loop" control.

### Mechanism, in one paragraph

A compressed loop can only ever carry s·d·p bits across an iteration boundary, no matter
how many iterations T it runs, because the state that persists between loop steps is
exactly those s slots — everything else is recomputed fresh from the read-only input each
time. Iteration therefore buys time (more chances to refine the same s·d·p-bit state) but
not memory. When a task needs to hold k concurrent, mutually-independent pieces of state
(k pointer chains, N key-value pairs) and s (or the recurrent state dimension) is below
that demand, no amount of looping repairs the deficit — the model is architecturally
incapable of holding more than it can hold, and accuracy collapses to near-chance,
uniformly across seeds. Above the threshold, whether training actually finds a working
routing solution is a separate, seed-dependent question.

### Bearing on Q1-Q4

- **Q1** (is NTP geometry disjoint from what the loop needs): partial bearing. The paper
  gives a structural (not merely "web text is boring") reason a compressed loop could
  look flat: our TUL slot loop carries exactly s=1 slot per span. Their s=1 column is the
  worst-case column in Fig 1's grid — and our own numbers (13.2 effective dims across
  ~50 slots per row, i.e. well under 1 dim per slot) put us in the same regime as their
  s=1 wall. This reframes the question from "is NTP's geometry compatible with
  iteration" to "does our 1-slot-per-span budget structurally forbid it regardless of
  target," independent of the CE objective.
- **Q2** (parameter-scale threshold): no bearing. The bound is asymptotic in n (sequence
  length) and the persistent-state size s·d·p, not in total parameter count; no scaling
  sweep is run.
- **Q3** (amortized design that still gives later passes a job): direct bearing. Their
  own stated "concrete experimental target" (§6) is to expose the persistent-state budget
  as a controllable knob (slot count, state dimension, cache rank) and measure how the
  critical budget scales with task load — precisely the missing axis in our ablations,
  which have swept targets and wiring but never slot count/dimension against an
  engineered concurrent-demand task.
- **Q4** (when must an iterated map iterate): this paper's answer is a pure
  **capacity** condition, orthogonal to contraction/saddles/power-iteration: iterating
  helps only once the persistent state's size already matches or exceeds the task's
  concurrent-item demand; below that it is a hard combinatorial wall, not a stability
  problem.

### One concrete arm for our setting

Build a synthetic span-structured task with a controllable "concurrent demand per span"
k (e.g. k independent facts/entities introduced within one span that the coda must
later retrieve, mirroring their pointer-chasing k), and sweep our slot budget per span
(s=1 today; try s∈{1,2,4}, or a higher-dimensional single slot) against k, holding the
target design fixed. Measure the K1-K6 / K3-K6 depth-earning curve and the per-pass
rank/cosine instruments already in place, as a function of (s, k).
**Prior: moderate-high** that K-curve gains appear only once s crosses the task's real
k, and that a null result at s=1, k=1 (ordinary web text, where there may genuinely be
only one dominant "next span" signal) is itself informative — it would rule out slot
capacity as our bottleneck and point back toward target design, consistent with
Wolfe's existing framing (`tul-intent-think-once-decode-cheap`,
`web-text-does-loop-target-is-the-lever`).

### Contradictions

None to the brief's numbers directly. One nuance worth flagging: their above-threshold
failures are *seed-dependent* (some seeds succeed, some fail) — a training/routing
signature. Our slot-loop flatness is instead *uniform across eleven-plus arms* with very
small variance (K3-K6 ≈ 0 or negative everywhere). That pattern looks more like their
low-variance *subdiagonal collapse* (s<k) than their high-variance *above-threshold*
regime, which is evidence (not proof) that we are capacity-bound rather than merely
routing-unlucky.

---

## 2. Lai, Bao, Quinn, Gilpin, "Fractal basins trap latent reasoning" (2609.04963)

### What they measured and how

Models (Appendix B, Supplementary Methods): Equilibrium Reasoners (EqR) on Sudoku
(hidden dim 512, 97-token sequence, 24 fixed loop iterations, injected noise disabled)
and Maze (hidden dim 128, 916 tokens); Fixed-Point Reasoning Models (FPRM), iterate to a
residual threshold τ=0.02 (Sudoku, cap 1000 iters) or τ=0.1 (maze, cap 100 iters); Parcae,
a 140M-parameter looped language model fine-tuned on the Countdown arithmetic task
(loop timestep Δt fine-tuned from 1.0 to 0.3); Tiny Recursive Model on ARC-AGI-1. A
fifth, purpose-trained model (an encoder-only looped transformer solving `Ax=b` over
`F_3^8`, i.e. integer linear systems mod 3) is used only for the training-dynamics
bifurcation study (Fig 5).

Instrument: for a fixed model, prompt, and loop schedule, vary only the initial latent
state along a random 2D slice of the full latent space (QR-orthonormalized directions,
200×200 grid), decode at every loop, and record the number of loops until the decoded
output stops changing — the "settling time." This settling-time field over the 2D slice
is the basin map (Fig 1B, Appendix B). Fractality is quantified by basin entropy S_b,
boundary entropy S_bb (Appendix D), and the uncertainty exponent α (basin-boundary
dimension = 2-α, Appendix E); saddle boundaries are localized with the fast Lyapunov
indicator λ_F (Appendix B). At least 300 valid slices per model/task pair, after
excluding slices where <90% of initial conditions converge or >1% hit the iteration cap.
Basin entropy correlates strongly with mean settling time (task difficulty) across all
four architecture/task pairs (Fig 2C) and replicates on two more architecture/task pairs
(Appendix C, Fig S1).

Task-difficulty axis: Sudoku-Extreme's annotated backtracking-based difficulty rating;
maze shortest-path length or added-wall degeneracy; the F_3^8 linear-system study varies
training step, not task difficulty, to watch a bifurcation emerge (Fig 5).

**Critically: every task in the paper and every appendix (including the Appendix C
replicate) is a discrete combinatorial puzzle** — Sudoku, maze, Countdown arithmetic,
ARC-AGI visual riddles, and integer linear systems mod a prime. There is no
next-token / perplexity / language-modeling task anywhere in the main paper or the
supplement. Parcae, the one looped *language* model used, is fine-tuned to solve an
arithmetic puzzle (Countdown), not evaluated as an LM.

### Mechanism, in one paragraph

Hard constraint-satisfaction problems have many "nearly correct" candidate solutions.
Early in training these are separate stable local minima (Fig 5A, pre-bifurcation: all
finite-time Lyapunov exponents negative, no chaos, trajectories converge smoothly to
whichever nearby wrong answer is closest). As training crosses a bifurcation where the
model gains multi-step solving capability, those wrong answers **lose stability and
become saddle points** rather than disappearing — the model becomes globally
monostable (one true fixed point) but the space fills with weakly-unstable saddles that
still weakly attract trajectories. A trajectory that starts near one of these saddles
gets scattered ("Plinko," Fig 3A) before eventually escaping toward the true solution,
and two initially-adjacent trajectories that straddle a saddle diverge by orders of
magnitude in convergence time — this is transient chaos, and it produces the observed
self-similar basins. Decisively (Fig 5D): only the "core" variables that require
genuine multi-step reasoning (Gaussian-elimination-style, not one-shot substitution)
show positive Lyapunov exponents; the easy, directly-substitutable variables never show
chaos. So iteration is required/useful specifically for escaping near-miss saddles on a
problem's irreducible combinatorial core — not for reasoning in general.

### Bearing on Q1-Q4

- **Q1**: no direct evidence (no NTP task tested anywhere), but real indirect bearing.
  Their mechanism is specific to tasks with discrete, checkable near-miss wrong answers
  that a trained model must actively escape (a Sudoku grid with a repeated digit, a
  maze dead end). Ordinary next-span text continuation does not obviously present this
  structure — there is no single discrete "wrong grid" a language model gets stuck near.
  This is consistent with, but does not prove, a real geometric gap between what NTP
  produces and what their saddle-escape mechanism requires.
- **Q2**: no bearing. No parameter-scale axis (Parcae 140M is the only sized model
  mentioned; the rest are unspecified small nets).
- **Q3**: no bearing. The paper is purely diagnostic on existing reasoning loops; it
  says nothing about chunked/amortized designs.
- **Q4**: strong, direct bearing, and it is a genuinely different mechanism from
  Wu/Zhang/Cao's power method (paper 3) or Anil et al.'s path independence (paper 4):
  iteration is required specifically near a **multistable-to-monostable bifurcation**
  where escaping saddle-shaped near-miss solutions takes variable, sometimes very long,
  time. A model whose core map is *everywhere* contractive with no saddle structure
  (never near such a bifurcation) predicts, by their own logic, that depth cannot help —
  which is exactly the "pre-bifurcation" picture in their Fig 5A (all FTLE negative, no
  chaos, smooth monotone convergence), and matches our own measured slot-loop gain
  (0.2-0.9, monotone convergence to a fixed point by pass 6, no reported instability).

### One concrete arm for our setting

Run their basin-map diagnostic directly on the MORPH slot loop: fix a row and a target
span, perturb the slot's entry state along a random 2D slice (as in their Appendix B),
decode the coda's next-span prediction at each loop iteration, and check for (a) any
settling-time variation at all across the slice, and (b) fractal structure (basin
entropy, uncertainty exponent) if variation exists.
**Prior: low** that we find any fractal structure on ordinary web text — our own gain
numbers already look like their pre-bifurcation, purely-contractive regime, so I expect
a flat, low-entropy basin map. That would be a useful **negative/confirmatory** result:
it would show, via an independent instrument, that our map has never been in the
saddle-escape regime their mechanism requires, which is a second, mechanistically
distinct reason (alongside the power-method reading below) that later passes have
nothing to do. It's a cheap add-on to instrumentation we already have (we already record
per-pass decoded state and per-pass Jacobians).

### Contradictions

None to the brief's numbers. Worth flagging as a distinction, not a contradiction: their
mechanism (saddle-escape, transient chaos) is not the same phenomenon as "the loop is a
power iteration" (our brief, and paper 3) — a power iteration is a purely linear,
globally contractive/expanding rotation toward a dominant eigenvector, with no saddles
and no chaos at all. The two mechanisms are not mutually exclusive in general, but this
paper gives no evidence that saddle-trapping occurs outside genuinely combinatorial
puzzle tasks, so it neither supports nor undermines the power-method reading of our own
loop — they are compatible, disjoint explanations for two different regimes.

---

## 3. Wu, Zhang, Cao, "Looped Transformers with Layer Normalization Provably Learn the Power Method" (2606.00605)

### What they measured and how

Fully theoretical (population-loss gradient descent, exact convergence proofs) plus a
small numerical illustration; no LM task, no perplexity, no real text.

Setting (§3.2): a one-layer linear (no softmax) self-attention block with RMSNorm-style
LN and a residual connection, `TF(E;θ) = LN(E + V·E·Eᵀ·W·E)`, looped L times with fully
shared parameters θ=(V,W). Task (§3.1, "principal component prediction"): given a data
matrix X (columns unit-norm, eigen-decomposition of `XXᵀ` with top eigenvalue λ_1 of
multiplicity r) and a random query vector `a` on the unit sphere, predict a vector in
`X`'s leading eigenspace. Population loss = squared distance from the model's L-loop
output to the leading eigenspace (Def 3.1, eq. before Thm 4.1). GD initialization is a
specific block-structured `V^(0), W^(0)` (§4.1) chosen to guarantee nonzero gradients.

**Theorem 4.1** (population GD on the looped, LN'd model): for all t≥0 the weight
matrices stay exactly on a 2-parameter manifold `W^(t) = [[0, w_t·I],[0,0]]`,
`V^(t) = [[0,0],[v_t·I,0]]`, with `w_t, v_t = Θ((η·Υ1·t/d)^(1/4))` diverging to
infinity, and the loss converges to a strictly positive limit
`L_L^(∞) = E[Dist²((XXᵀ)^L·a / ‖(XXᵀ)^L·a‖₂, V(X))]` at rate `O(√(dΥ1/(ηt)))`. As
`ρ_t = w_t·v_t → ∞`, the model's exact output is
`y_L = (I+ρ_t·XXᵀ)^L·a / ‖·‖₂ → (XXᵀ)^L·a / ‖(XXᵀ)^L·a‖₂` — **exactly the L-step power
method** — even though the model was trained only to predict a principal component,
never supervised toward the power-iteration algorithm itself (§4.1, "algorithmic
implicit bias").

**Theorem 4.2/4.3/4.6** (LN vs no-LN): even under *layerwise* supervision by the power
iteration's own targets, an unnormalized transformer layer converges only to a fixed,
suboptimal contraction rate `R* = (1+γ*λ_{r+1})/(1+γ*λ_1) > λ_{r+1}/λ_1`, not improvable
by further training, whereas the LN'd model's contraction rate converges (as γ_t→∞) to
exactly `λ_{r+1}/λ_1`, the optimal power-method rate (Parlett, 1998).

**Theorem 4.6/4.7** (convergence rate = the exact "when do you need more iterations"
answer): the angle `φ_L` between the L-loop output and the true top eigenspace obeys
`|c_{r+1}|·(λ_{r+1}/λ_1)^L ≤ lim_t sin φ_L ≤ (‖a⊥‖/‖a_Z‖)·(λ_{r+1}/λ_1)^L` for the LN
model. **The number of loop iterations needed to reach a target error is set entirely by
the eigenvalue ratio λ_{r+1}/λ_1 (the spectral gap)**: a large gap converges in one or
two iterations and extra iterations do nothing but rotate an already-converged vector; a
small gap needs many.

Numerical experiments (§5, Fig 1-2): d=16, n=32, L=10 (parameter-structure check) and
L∈{1,...,25} (iterative-inference check), SGD lr=0.1, batch 128, 10000 train / 2000 test
examples, 2000 epochs. Confirms the `Θ(t^(1/4))` weight growth and the LN-vs-no-LN
separation in angular error decay qualitatively.

### Mechanism, in one paragraph

Under gradient descent with this initialization, the parameters are provably confined
(by a Schur's-lemma argument, Appendix A.3) to a 2-scalar family whose product ρ=wv acts
as a single "gain" on `XXᵀ`. LN is what lets that gain diverge to infinity during
training rather than settling at a finite value — without LN the population loss has no
finite minimizer either, but the model gets pinned at a strictly worse contraction rate.
Once ρ→∞, the per-layer update is exactly a scaled step of matrix power iteration on
`XXᵀ`, so L shared layers implement exactly L steps of the power method, and the whole
"looping helps or doesn't" question reduces to: how close is the top eigenvalue to the
rest of the spectrum? If the gap is wide, the model's own linearized per-pass map
already has a completely dominant direction, one pass captures almost all of it, and
every subsequent pass can only rotate the (already tiny) remaining error — this is
mechanistically identical to "one pass sets scale, rest rotate; rank collapses" in our
own brief.

### Bearing on Q1-Q4

- **Q1**: partial, diagnostic bearing. This gives an exact, checkable criterion — not an
  answer about NTP specifically, but a test: measure the effective spectral gap
  (σ_1/σ_2, or the top-two-eigenvalue ratio) of whatever linear(ized) operator our
  per-pass Jacobian approximates on real web-text spans. If that gap is large, this
  theory predicts flat K-curves regardless of target design, matching our "rank of the
  update collapses 181→9" finding to the letter. It does not itself say WHY NTP would
  produce a wide-gap operator — that remains an open empirical question about the
  geometry of "predict the next span."
- **Q2**: no bearing (toy d=16 setting, no parameter-count axis).
- **Q3**: indirect but concrete bearing. Nothing here is about amortized/chunked
  computation, but the mechanism suggests a specific fix: if each pass is trained to
  extract a *different* residual direction (deflation — pass k targets what's left after
  removing directions 1..k-1), a one-pass map genuinely cannot do the job of k passes,
  because deflation is inherently sequential; a fixed, shared per-pass eigen-extraction
  step cannot be collapsed into one iteration the way a single-target power iteration
  can.
- **Q4**: direct, exact bearing — the clearest quantitative answer in this batch. An
  iterated linear(ish) map benefits from more iterations exactly when the ratio of its
  top two (or top-r vs (r+1)) singular/eigenvalues is close to 1; convergence is
  geometric in that ratio, full stop. This is a precise, falsifiable version of "when
  must an iterated map iterate" for the power-iteration regime specifically (distinct
  from paper 2's saddle-escape regime and paper 4's path-independence regime).

### One concrete arm for our setting

(1) Cheap, reuses existing instrumentation: on the per-pass core Jacobian already
computed in prior audits, measure σ_1/σ_2 per row and correlate against that row's
K1-K6 (or per-row depth-earning). **Prior: moderate-high** this correlation is visible
and explains our aggregate ~0 K3-K6 as "the gap is uniformly wide on typical web text."
(2) A training-rule arm: replace the current same-target-every-pass slot loss with a
deflation-style target (pass k's loss is on the residual after projecting out what pass
1..k-1 already predicted), and measure whether successive-pass cosine similarity moves
from the current cancellation regime (-0.2 to -0.6) toward near-orthogonality (~0,
consistent with each pass extracting a genuinely new direction) and whether K3-K6 rises
above 0. **Prior: moderate** that this measurably changes the per-pass geometry (it
directly targets the mechanism), **lower** that it fully closes the K3-K6 gap, since a
too-small slot budget (paper 1) would still cap how much genuinely new information any
single slot can carry regardless of what each pass targets.

### Contradictions

None to the brief's numbers, but one important caution: Theorem 4.1 shows the LOOP LOSS
ITSELF never reaches zero for finite L, even after infinite training (`L_L^(∞) > 0`),
purely because a finite-L power iterate is not exactly the eigenvector. A nonzero,
plateaued loop loss is not by itself evidence that the loop "failed" — it can be exactly
what an optimal L-step algorithm looks like. This is a caution for reading any of our
own loss-plateau numbers as verdicts without a matched-depth-optimal baseline.

---

## 4. Anil, Pokle, Liang, Treutlein, Wu, Bai, Kolter, Grosse, "Path Independent Equilibrium Models Can Better Exploit Test-Time Computation" (2211.09961)

### What they measured and how

Small algorithmic/toy models throughout (no parameter counts stated; DEQ-style
weight-tied, input-injected networks). Tasks (§3): prefix-sum (32-bit training strings,
OOD tested on other lengths, e.g. 64-bit), mazes (9×9 train, OOD on 13×13/25×25),
plus blurry MNIST, 10×10 matrix inversion, and graph edge-copy in supplementary
material. Compared against non-input-injected fixed-depth baselines and "progressive
nets" (Bansal et al. 2022).

Instrument: the **Asymptotic Alignment (AA) score** (Algorithm 1, §4.2) — batch two
inputs, run the fixed-point solver from a zero initialization to convergence, then swap
each example's converged state into the OTHER example's initialization and re-run to
convergence; the AA score is the mean cosine similarity between the twice-converged and
once-converged state for the same input. AA=1 means the model reaches the same fixed
point no matter what it started from (fully path independent); lower AA means the
reached state still depends on where the iteration started.

Numbers: Table 1 (adversarial stress test) — PI networks: AA 1.00 (maze) / 0.99
(prefix-sum), 100%/100% accuracy, and AA/accuracy are UNCHANGED (1.00/100%, 0.99/100%)
even under an L-BFGS adversarial search for initializations designed to break alignment.
Non-PI networks: AA 0.32 (maze) / 0.62 (prefix-sum), 87.12%/66.66% accuracy, collapsing
to AA 0.09/0.18 and 0%/0% accuracy under the same attack. Table 2 (training-time
convergence ≠ path independence): a DEQ trained with phantom gradients has a WORSE
in-distribution residual (11.83) than one trained with implicit-function-theorem
gradients (1.4), yet both reach high AA (0.96/0.99 in/OOD vs 0.99/0.99) and near-ceiling
accuracy (99.96%/99.88% vs 99.99%/100%) — convergence quality and path independence are
empirically dissociated. Fig 4: AA score correlates strongly with upward-generalization
accuracy specifically "when the inference depth is large enough" at test time.

Two causal interventions (not just correlational), both on the same 12 prefix-sum
networks, §6: (a) **promoting** PI — train with half the batch initialized from zero and
half from Gaussian noise, plus randomized forward-pass iteration budgets during
training — raises AA scores and raises accuracy together, even for very shallow (5-6
layer) unrolled networks that are far from converged at train time (Fig 3); (b)
**penalizing** PI — an auxiliary loss term that directly penalizes the alignment
(dot product) between fixed points reached from different Gaussian-noise
initializations of the same input — lowers AA scores and correspondingly degrades
accuracy, on the same accuracy-vs-AA trend line (Fig 3, §6.2).

### Mechanism, in one paragraph

A weight-tied, input-injected recurrent module that converges to the SAME limiting
state regardless of where the iteration is initialized has, in effect, "learned where to
stop" rather than learned one fixed finite computation graph. Because the limit doesn't
care about the starting point, running it for more iterations than seen in training
still drives it toward the same input-dependent target, so extra test-time depth
translates directly into more accuracy on harder instances that need more steps to
settle. A path-DEPENDENT model's output is a function of the specific trajectory (start
point and iteration count) as well as the input, so pushing iteration count past the
training regime moves it off-distribution rather than toward a better answer. Crucially,
this property is dissociated from mere numerical convergence: a model can have a large
train-time residual and still be highly path independent (it just hasn't finished
converging yet, but is heading toward the same place from anywhere), and a model can
converge cleanly to different limit cycles under different solvers and still be path
independent in the relevant sense (§7, Table 2, Fig 5).

### Bearing on Q1-Q4

- **Q1**: no direct evidence (all tasks are discrete algorithmic puzzles with a
  well-defined fixed point that benefits from open-ended refinement; no NTP/LM task is
  tested). Indirect bearing only: path independence is a property of the trained MAP,
  not of the "topic" of the objective, so nothing here rules out NTP inducing it — but
  nothing shows it does either.
- **Q2**: no bearing (no parameter-scale sweep; all toy-scale networks).
- **Q3**: strong, directly actionable bearing — the clearest instrument in this batch for
  testing wiring/training-rule choices cheaply. The AA score is directly computable on
  the MORPH slot loop today: it is exactly the kind of "does this map converge to the
  same place from a different start" probe we have not run.
- **Q4**: a genuinely distinct axis from papers 2 and 3. Path independence is about
  whether MULTIPLE initializations agree on a limit, not about the SPEED of convergence
  from one (already-good) start (paper 3's eigengap) or about SADDLE structure near a
  bifurcation (paper 2). A map can be fast-converging and saddle-free (paper 3's regime)
  while still being path-DEPENDENT if its limit depends on where it started — and
  conversely can be path-independent while converging to a limit that is *useless*
  (encodes nothing new). This paper's own framing implicitly assumes the shared fixed
  point is informative; it does not test what happens when path independence holds but
  the fixed point is uninformative — exactly our own regime, per the brief's
  RELAY/EXTRACTABILITY framing.

### One concrete arm for our setting

Implement Algorithm 1 directly on the slot loop: for a batch of rows, run the slot
loop's iteration from its real (prelude-conditioned) entry to iteration 6, then
re-initialize from a DIFFERENT row's converged slot state (or from Gaussian noise) and
run 6 more iterations, and measure the cosine similarity between the twice- and
once-converged z. Run this on the existing prelude-entry arm (K1-K6 0.033, near-flat)
and the noise-entry arm (K1-K6 0.185 on the plain model) side by side.
**Prior: moderate.** I expect the prelude-entry arm to show a HIGH AA score (it already
converges to nearly the same place regardless of entry — consistent with a fixed point
that adds no information, matching the brief's "coda barely reads the channel" finding)
and the noise-entry arm to show a LOWER AA score (its extra iterations are doing real,
entry-dependent work, consistent with its larger measured K1-K6). If instead the
noise-entry arm ALSO shows high AA, that would falsify "path independence explains our
flatness" and point the open question elsewhere — worth running specifically because it
is a cheap, previously-untested axis, distinct from every instrument already in our
ledger (rank, cosine of successive updates, Jacobian gain, causal fitted-z).

### Contradictions

A real tension with paper 1, worth carrying forward rather than resolving here: this
paper's headline claim is "more test-time iteration should help whenever the model is
path independent," while paper 1's claim is that a compressed loop's persistent-state
BUDGET caps what more iteration can ever store, regardless of convergence properties. A
model can be perfectly path independent (AA=1, converges to the same z from any start)
while that z still holds only s·d·p bits — upward generalization would then be capped by
capacity even though path independence holds. The two papers are not about the same
axis: path independence asks whether iterating converges usefully; the memory-budget
paper asks whether the converged state has room for the answer at all. Both are
necessary, neither is sufficient, and nothing in either paper's own results contradicts
the other.

---

## Synthesis for Q1-Q4 (≤300 words)

Four independent axes, none contradicting our numbers, each giving a different
necessary (not sufficient) condition for "iterating helps":

**Q1** (NTP vs iteration, disjoint geometry?): no paper tests NTP directly. But three
converging diagnostics are now available and cheap: (a) per-span slot capacity vs.
concurrent demand (paper 1: are we at s=1 against a k>1 task?); (b) the per-pass
Jacobian's top-two-eigenvalue ratio (paper 3: is our map's spectral gap simply wide on
web text?); (c) saddle/basin structure of the per-pass map (paper 2: are we ever near a
multistable bifurcation, or purely contractive throughout?). Our own numbers (rank
collapse 181→9, gain 0.2-0.9, monotone convergence by pass 6) already look like the
"wide gap, no saddle" regime in both papers 2 and 3 — consistent with, not proof of, a
real geometric mismatch.

**Q2** (parameter-scale threshold): no paper varies parameter count as an axis; genuinely
open, unaddressed by this batch.

**Q3** (what design still amortizes but gives later passes a job): two concrete,
previously untested arms fall out directly — (a) sweep slot count/dimension against an
engineered concurrent-demand task (paper 1); (b) a deflation-style per-pass target so
each pass targets a residual direction the previous pass didn't cover, which a one-pass
map provably cannot collapse into itself (paper 3).

**Q4** (mathematical principles): three genuinely distinct, complementary "must iterate"
conditions, none of which is "iteration helps in general": a capacity wall (state
budget below task demand, paper 1), a spectral-gap condition (power-method rate, paper
3), and a path-independence condition dissociated from both (paper 4's AA score — cheap,
directly runnable on our slot loop, untested by us so far). Saddle/basin escape (paper
2) is a fourth but needs a discrete near-miss structure our targets don't have.
