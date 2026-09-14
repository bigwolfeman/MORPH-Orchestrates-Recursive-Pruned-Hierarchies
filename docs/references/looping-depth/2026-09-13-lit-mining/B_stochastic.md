# Literature read B: stochastic, set-valued and attractor loops

Read against the brief at `loop_lit_brief.md`. All numbers cite section/table/figure of
the source paper (fetched via arxiv.org/html).

---

## 1. GRAM — Generative Recursive Reasoning (arXiv 2605.19376, Baek/Jo/Kim/Ren/Bengio/Ahn)

**What they measured.** GRAM is a hierarchical recurrent reasoner: high-level state
updated stochastically over T=3 steps, each wrapping K=4-6 deterministic low-level
steps, deep supervision Nsup=16 (Section 2.1, Appendix B.1-B.2). Model: D=512, 2-layer
transformer+SwiGLU backbone (Table 4). Sizes: GRAM 10M; baselines Looped-TF 7M, HRM
27M, TRM 7M, Direct-Pred 27M/100M (Table 1). Tasks: Sudoku-Extreme, ARC-AGI-2, N-Queens
8x8/10x10, graph coloring (7,002/13,465 train, Appendix C.2), binarized MNIST, and
unconditional Sudoku generation (50K solutions, Appendix D.5). Loop-contribution
instrument: Figure 4 (left), inference-time scaling by iterations x samples — GRAM at
16 iterations x 20 samples beats TRM at 320 deterministic iterations (97.0% vs 90.5%).
No language modeling task anywhere in the paper (checked explicitly: no perplexity or
bits-per-byte reported). No FLOPs- or depth-matched plain-transformer control is
reported for the reasoning tasks; Table 2 (Direct-Pred at 27M/100M) is the closest
parameter-matched comparison and GRAM still wins with 10M.

**Mechanism.** GRAM turns the loop into a latent-variable generative model: at each
high-level step, noise is drawn from a learned Gaussian, eps_t ~ N(mu_theta(u_t),
sigma_theta^2(u_t) I), and added to the deterministic update, z_t = u_t + eps_t
(Eq. 4-5, 8-9; "stochasticity is introduced only at the high level," Section 2.1).
Training is amortized variational inference against a trajectory ELBO (Eq. 11, 13,
truncated to Eq. 14 in practice). Because each iteration injects genuinely new
randomness rather than a deterministic function of the entry state, a later pass is not
redundant with pass 1 by construction: it can land on a different high-level latent and
therefore a different completion, which is what lets test-time sampling of many
trajectories beat a single deterministic rollout at any iteration budget (Fig. 4).

**Q1-Q4.**
- Q1 (NTP geometry vs loop-needing geometry): no bearing — never tested on text.
- Q2 (scale threshold): no bearing — all models 7-100M, no scale sweep.
- Q3 (amortized design giving later passes a job): weak/negative bearing. The mechanism
  gives later passes a job (explore a new trajectory), but the design is NOT amortized
  in the TUL sense — GRAM's wins come from spending MORE test-time compute (more
  samples x more iterations, Fig. 4), the opposite of "think once, decode cheaply."
- Q4 (when must an iterated map iterate): bears directly. This is the cleanest instance
  of the brief's own information-theoretic RELAY criterion satisfied by construction —
  injecting fresh randomness at every step means each z_t genuinely carries information
  the entry state did not (new entropy), so a deterministic-map-adds-no-information
  argument does not apply to a stochastic map. The open question is whether a
  single-answer-per-row LM objective can make USE of that extra entropy.

**Concrete arm for our setting.** Make the slot update stochastic and variational:
z_{t+1} = f(z_t) + eps_t, eps_t ~ N(mu_theta(z_t), sigma_theta(z_t)), trained with a KL
term against a learned prior (Eq. 11-14 form), scored with an importance-weighted
multi-sample ELBO over z at the span decoder. Measure: (a) the existing K-curve on
tokens (K1-K6) under this objective; (b) a per-pass instrument specific to stochasticity
— cosine similarity between z_T rollouts from different noise seeds at the same entry,
to see if the coda is actually sensitive to which sample it gets (contrast with the
brief's causal-fitted-z result, which used a deterministic point estimate). Prior: LOW.
Web text has one ground-truth continuation per row under teacher forcing; GRAM's wins
come from tasks with structurally MANY correct answers (many N-Queens solutions, many
valid Sudoku completions under partial constraints). A KL-regularized latent on a single
teacher-forced target usually degenerates into free-bits collapse or acts as dropout,
which is a known unhelpful place we've already been (per-pass targets all met in one
pass, per the brief). Worth a cheap 2.5k-step probe given it is the one genuinely
untried "inject real entropy per step" lever, but do not expect it to survive contact
with a single-answer target.

**Contradictions.** None direct. Indirectly supportive of the brief: GRAM's gains
require a task with many valid outputs for one input, which is exactly the kind of
target MORPH's slot loop does NOT have (next-span prediction under teacher forcing is
graded against one path). The softmax head already gives NTP its local multimodality;
GRAM's stochastic loop is solving a different problem (global multimodality over
discrete combinatorial answers) that a token-level distribution does not.

---

## 2. Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning (arXiv 2605.21488, Huang/Geng/Kolter, ICML 2026)

**What they measured.** Weight-tied iterative models: 2 blocks x 21 iterations
(5.03M params) on Sudoku-Extreme vs a 42-block non-recurrent feedforward baseline
(105.6M params); 1 block x 15 iterations on Maze-Unique (Appendix D.1: MLP-mixer for
Sudoku, self-attention for Maze). Loop-contribution instrument, Table 4, same trained
model, test-time depth only: D=16->64 iterations lifts Sudoku 86.4%->93.0% and Maze
82.2%->88.9% (B=1, no extra samples); Figure 1 shows a model trained at 16 iterations
still improving out to 1,024+ test-time iterations ("~40,000 effective layers"). Full
recipe (depth x breadth scaling, D=64,B=128) reaches 99.8%/93.0% (Table 4). Baselines
(Table 1): HRM 55.0%/0.3%, TRM 84.8%/44.9%, URM 77.6%/51.4%, EqR 99.8%/93.0%. No
language modeling task. No FLOPs-matched control beyond the construction path in
Table 2, which itself shows plain depth without weight-tying or training tricks fails
badly (64-layer feedforward: 2.6% accuracy) — i.e. their own ablation argues the effect
is a training-regime effect, not a scale effect ("suggesting the mechanism, not raw
scale, drives performance").

**Mechanism.** Two training-time interventions shape the attractor landscape:
randomized initialization (RI, z_0 ~ mu_0(.|x) drawn independently per trajectory
instead of fixed, Section 5.1) and noise injection (NI) during rollout,
z_{k+1} = z_k + (1-lambda) r_theta(z_k, x) + beta*eps_k (Section 5.2). Together these
widen the basin explored during training and push the operator toward being a genuine
contraction near the CORRECT fixed point rather than a spurious one. The formal
argument (Section A.1, Eq. 5-7) bounds distance to the fixed point by the residual under
an L-Lipschitz map: ||z - z*|| <= ||R_theta(z,x)|| / (1-L), with a sufficient-residual
correctness condition. Later passes matter here because the target itself IS the
attractor of a genuine constraint-satisfaction process (Sudoku/Maze solving decomposes
naturally into repeated constraint propagation); a single step cannot jump the whole
basin from an arbitrary start, so the ~2x accuracy gain from D=16->1024 (Fig. 1) is real
depth-of-correction, not depth-of-parameterization.

**Q1-Q4.**
- Q1: bears, and is the strongest indirect evidence for "yes, disjoint" on our
  side without contradicting the reasoning-task literature. EqR's own theory (Section
  A.1) requires the ground truth to be an attractor reachable by contraction from a
  possibly-distant start. Sudoku/Maze targets satisfy that by construction (CSP solving
  is iterative). Next-token prediction over web text is not obviously the fixed point of
  any iterative process over the entry representation — there is no analogous
  "constraint propagation toward the true next span" structure the loop's target gives
  it to converge to. This matches the brief's own finding that the slot state hits its
  fixed point by pass 6 with no task-relevant work being done past pass 1.
- Q2: no bearing — no scale sweep; paper argues explicitly against a scale explanation
  for its own results.
- Q3: weak positive bearing. RI+NI are TRAINING-time costs; beta can be annealed to 0 at
  test time so inference stays a single rollout of the trained iteration count — this is
  compatible with "think once per span" if the extra test-time iterations (D up to 1024)
  are not required for the gain. But Table 4 shows the accuracy gain is mostly a TEST-TIME
  depth-scaling effect (86.4%->93.0% from D alone), which is not free at inference and is
  the opposite of amortized decoding.
- Q4: bears directly, and gives one of the two clean positive answers in this batch: an
  iterated map has to iterate when its target is the attractor of a contraction and the
  entry point starts outside the basin's "already converged" region; the residual bound
  ties correctness explicitly to iteration count. This is a genuine "attractor" answer to
  Q4's list.

**Concrete arm for our setting.** Randomize the slot's entry seed (currently the
deterministic `E_slot + W_sent . embed(t_last)`) and add small per-pass Gaussian noise
during TRAINING only (annealed to 0 at inference), i.e. z_{t+1} = z_t +
(1-lambda) f(z_t) + beta_t * eps_t with beta_t -> 0 over training, to see whether
forcing robustness to noisy starts creates a landscape with a real attractor for the
NEXT-SPAN target. Measure: the existing K1-K6 and K3-K6 forced-depth sweep on tokens,
plus a same-trained-model TEST-TIME depth sweep beyond the trained mean (their Table 4
protocol: does accuracy/CE keep improving past T=6 on the SAME weights, the way EqR's
does past its trained 16?). Prior: LOW. Our failure mode already shows the state reaches
a fixed point by pass 6 with the CURRENT deterministic entry; landscape-shaping tricks
change how INITIALIZATION-sensitive the map is, but the brief's causal-fitted-z result
(0.25 nats WORSE than the loop's own z, no causal headroom) suggests there isn't a
better fixed point sitting nearby to steer toward for THIS target — this is a fix for a
landscape-shaping problem, not for a target-design problem, and the brief's own note
says the mechanism problem is the target.

**Contradictions.** None direct to a specific number in the brief. It does contradict
an overreach the brief should avoid: "no reasoning task benefits from depth past a
handful of passes" — EqR's same-trained-model D=16->1024 curve (Fig. 1, Table 4) is real
evidence that a weight-tied loop CAN keep earning past its training depth when the task
target is genuinely iterative. The relevant test for MORPH is whether OUR target (next
span from teacher forcing) has that property at all, which per Q1 above looks unlikely,
but the test itself (same-weights depth sweep beyond training depth, not just to 6) is
worth running before concluding no.

---

## 3. Solve the Loop: Attractor Models for Language and Reasoning (arXiv 2605.12466, Fein-Ashley/Rashidinejad)

**What they measured.** A backbone transformer proposes an initial output-embedding
sequence, y~0 = T_theta_b(E(x)) (Eq. 1); a SEPARATE attractor module refines it,
y~_{t+1} = T_theta_a(y~t, y~0) (Eq. 2), solved to A_theta_a(y~*, y~0) := T_theta_a(y~*,
y~0) - y~* = 0 (Eq. 3), trained through implicit differentiation:
dL/dtheta = u^T dT_theta_a/dtheta, u = (I - J_y~^T)^-1 v (Eq. 4, Section 3.2). Language
modeling on FineWeb-Edu (nanochat pretraining recipe, Section 4.1), 140M/370M/770M,
compared to a parameter-matched plain Transformer and to Parcae (a looped LM). Table 1
validation PPL: 140M Transformer 21.48 vs Attractor 18.30 (-14.8%); 370M 15.79 vs 14.03
(-11.1%); 770M 13.08 vs 12.09 (-7.6%) — the gain SHRINKS with scale. Lambada PPL, 140M:
Transformer 127.39 vs Attractor 68.02 (-46.6%). Backbone (T_theta_b) and attractor
(T_theta_a) are SEPARATE parameter sets, theta := (theta_a, theta_b, E) — confirmed via
follow-up query, not weight-tied. "All models matched in parameter count," but there is
NO FLOPs-matched deeper-Transformer control anywhere in the paper (confirmed by direct
query) — an Attractor-model forward at T=1 pays a full backbone forward PLUS a full
attractor-module forward, strictly more FLOPs per token than a plain Transformer of
equal TOTAL parameter count run once. Iterations are counted via adaptive halting,
exiting when ||A(y~t, y~0)||_2 / ||y~t||_2 < epsilon or at Tmax (Section 3.1); exact
Tmax values are not stated numerically in the text extracted.

**The load-bearing number in this paper is Figure 7**, not Table 1: perplexity vs
test-time iteration count T shows T=0 (decode the backbone's proposal directly, no
attractor solve at all) is "already at or near the converged value" at every scale; T=1
reaches peak performance; beyond T=1 the model is "essentially converged." The authors'
own words: "our method reaches peak performance at T=1 at every scale." By contrast
their Parcae baseline's PPL "improves monotonically from T=1 until it plateaus near
T=8" — i.e. Parcae's loop earns depth up to 8 iterations but the Attractor model's loop
does not earn past 1.

**Mechanism.** The authors attribute the PPL win to the BACKBONE learning to propose an
output already close to the fixed point ("the backbone learns to produce a proposal
that already lies close to the fixed point that the solver would otherwise compute
through iteration") plus a stability regularization effect of implicit differentiation:
"the implicit gradient contains the inverse factor (I - J^T)^-1, which becomes
ill-conditioned near non-contractive regimes. This creates a barrier against unstable
fixed-point dynamics." In plain terms: training through the implicit fixed-point
gradient forces the attractor module's Jacobian to stay contractive, which is a
regularizer on that ONE extra module's weights, not a claim that iterating helps beyond
one corrective step.

**Q1-Q4.**
- Q1: this is the strongest and most direct evidence in the whole batch, because it is
  the only paper testing real language modeling. Their own Figure 7 shows the SAME
  collapse-to-one-pass pattern the brief measures on MORPH (T=0 near-converged, T=1
  peak, flat after), on an architecturally DIFFERENT loop (a separate attractor module
  with implicit differentiation, not a weight-shared 6-block core). That two unrelated
  architectures both find next-token web-text prediction saturates a loop at one usable
  step is independent, reproducing evidence that Q1's "too disjoint" hypothesis is
  likely TRUE for plain NTP CE targets on this kind of corpus.
- Q2: bears, and argues AGAINST a scale threshold in the 140M-770M range: the PPL
  improvement percentage falls from 14.8% to 11.1% to 7.6% as scale rises (Table 1),
  the opposite of "needs more scale to show a benefit." It does not test past 770M so it
  cannot rule out a much larger threshold (e.g. 3B), but the direction of the trend
  inside the tested range does not support one.
- Q3: no positive bearing. The design pays a full second module's FLOPs on every token
  (dense, not per-span/per-chunk), so it is not amortized in the TUL sense at all; it
  is closer to our own "paid loop," which the brief already scores as "earns 0.010-0.17
  nats but is not the design."
- Q4: weak bearing. The implicit-function-theorem stability argument matches the
  brief's own ρ(J_core) framing (a Jacobian regularized toward contractivity) but
  explains why the ONE step that does happen doesn't diverge, not why more steps would
  be needed — consistent with the brief's "power iteration" reading (one pass sets
  scale, further passes rotate/cancel).

**Concrete arm for our setting.** None strongly recommended as new — the paper's real
contribution to our question is a NEGATIVE result to file, not a lever to try: even a
fully separate second module, trained through an implicit fixed-point objective, on
real FineWeb-Edu NTP, saturates its own loop at T=1 (Figure 7). If we wanted to probe
their actual mechanism, the closest arm is giving the slot loop's pass 1 a genuinely
UN-shared parameter set from passes 2-6 (their backbone/attractor split) to see if a
one-time capacity increase (not iteration) explains any residual MORPH gap — but this
is the same "add more depth" experiment the brief's matched-compute control already
covers (depth-1 plain beats every slot arm by 0.25-0.33 nats), so prior is LOW this
reveals anything new.

**Contradictions — flag explicitly.** Table 1's headline PPL gains (7-15%) could be
misread as "looping helps LM perplexity," which would look like it contradicts this
project's findings. It does not: per the paper's own Figure 7, those gains are captured
entirely at T=1 (one extra module, one pass) with ITERATION contributing nothing
further. Read correctly, this is a second confirmation of the brief's central claim
(pass 1 does everything), not a counter-example, and should be cited as such rather than
as "a paper that got looped LM to work."

---

## 4. Bifurcation Models: Learning Set-Valued Solution Maps with Weight-Tied Dynamics (arXiv 2605.07277, Jore/Liu)

**What they measured.** Weight-tied recurrent maps on non-language tasks: a
message-passing GNN on frustrated antiferromagnetic Ising graphs up to ~15,000 nodes
(Appendix E.3), and a Fourier Neural Operator on the Allen-Cahn PDE on a 64x64 grid with
random forcing (Section 4). No language modeling. Loop-contribution evidence is
implicit (Figures 1 and 4 show progressive convergence over unrolled steps y1, y2, y4,
y8) rather than an explicit accuracy-vs-iteration ablation table. The headline number is
DIVERSITY OF SOLUTIONS, not accuracy: Table 2, Ising — energy-trained recurrent GNN
finds 13.635 +/- 5.760 distinct low-energy configurations out of 20 trajectories (each
started from a different y_0), while a LABEL-trained recurrent GNN or a non-recurrent
vanilla GNN both collapse to 1.000 +/- 0.000 solutions regardless of y_0. Allen-Cahn
with the diversity-regularized objective (lambda_init=1.2): 16.11 distinct steady states
found vs 1.00 for label-trained or energy-only (no diversity term) training. Energy
values: label+vanilla GNN -0.059 +/- 0.358, energy+vanilla GNN -0.750 +/- 0.190,
energy+recurrent GNN -0.983 +/- 0.101 (best, and only recurrent+energy finds multiple
solutions).

**Mechanism — the one genuinely different mechanism in this batch.** There is NO
injected noise in the recurrence itself. Diversity comes from DETERMINISTIC bifurcation
across DIFFERENT initial conditions: "for a fixed input x, different initializations y_0
may converge to different stable equilibria. Thus the model does not need to output a
single label." The attractor SET is A_theta(x) := {lim_{t->infinity} g_theta^t(y_0, x) :
y_0 in Y}. Training needs either full branch labels (Algorithm 1, direct regression per
branch) or, when branch identity is unknown, an unsupervised ENERGY objective
(Algorithm 2, minimize E(y_T, x) rather than a one-hot label) — critically, Table 2
shows that LABEL training on the SAME recurrent architecture collapses to one solution
even though the architecture is capable of multiplicity; only energy-style training lets
different y_0 actually land in different basins. On top of that, a diversity term across
M parallel rollouts (pairwise squared distance, Section 4) is needed to stop energy-only
training from having all M rollouts converge to the SAME good basin. Later passes
matter here because a single feedforward step cannot decide, from one starting point,
which of several genuinely distinct stable basins that point belongs to — only running
the recurrence to convergence reveals it.

**Q1-Q4.**
- Q1: bears, and is the second clean positive answer. The multiplicity mechanism
  requires a fixed input WITH a genuinely multi-valued/multi-basin target (many
  degenerate Ising ground states, many Allen-Cahn steady states) AND training that does
  not force collapse to one label. Web-text next-span prediction as MORPH currently
  grades it (teacher-forced CE to one bag/next-span target, per the brief) is exactly
  the "label training" regime this paper's own Table 2 shows COLLAPSES a capable
  recurrent architecture to one solution. This suggests the disjointness in Q1 is not
  inherent to weight-tied loops or even to text, but to the TRAINING OBJECTIVE: a
  single-label CE target actively suppresses whatever multi-basin structure a loop
  might otherwise use.
- Q2: no bearing — tiny GNN/FNO models, no scale sweep.
- Q3: bears constructively — this is the one design in the batch that could keep
  amortization intact. It adds no extra parameters (still one weight-tied map,
  no second module unlike Paper 3) and no extra test-time iterations (same loop
  depth as now) — the only change is the ENTRY (randomize y_0 instead of the current
  deterministic seed) and the TRAINING LOSS (energy/diversity instead of one-hot CE).
- Q4: bears directly, and is the cleanest of the two positive Q4 answers: a weight-tied
  map has to iterate when the target is SET-VALUED and training uses an
  energy/diversity objective rather than a single teacher-forced label — the recurrence
  is the only thing that can resolve WHICH member of the solution set a given entry
  belongs to, and Table 2's label-vs-energy comparison is a controlled demonstration
  that the SAME architecture only does this under the right loss.

**Concrete arm for our setting.** Three changes, testable independently: (a) randomize
the slot's entry seed instead of deterministic `E_slot + W_sent . embed(t_last)`;
(b) replace the teacher-forced span-decoder/M-next CE with an energy-style
compatibility loss between the converged z and the true next span's embedding (minimize
E(z_T, span_embed), not CE to one target — Algorithm 2's form); (c) roll out K parallel
z's per slot from K different random seeds during TRAINING and add a diversity penalty
across them (dropped at inference — one seed, one rollout, unchanged amortized cost).
Measure: the existing K1-K6/K3-K6 token curve under this objective, AND a bifurcation-
specific instrument borrowed directly from their Table 2 — cluster the K converged z's
per slot and count distinct clusters as a function of loop depth (does the count of
distinct clusters fall with more passes, the way their label-trained baseline collapses
to 1?). Prior: MEDIUM-LOW. This is the one lever in the whole four-paper batch that is
mechanistically distinct from everything the brief's ledger has already tried (every
existing MORPH arm uses a deterministic seed and a single teacher-forced target). The
risk: natural-language spans may not have well-separated, comparably-likely completion
basins at the one-span granularity the way Ising ground states or PDE steady states do —
the softmax head already absorbs token-level multimodality, and a bag/next-span target
may be closer to "one correct answer with soft uncertainty" than "several discrete,
equally valid branches." Worth a cheap SYNTHETIC probe first (a toy task with an
explicit small number of discrete branches per span, in the spirit of the ledger's toy
slot-loop successes) before spending GPU hours confirming or refuting basin structure on
real web text.

**Contradictions.** This paper is evidence AGAINST the broadest possible reading of the
brief's findings — "weight-tied loops just don't do useful iterative work" is false in
general (Table 2's 13.6 vs 1.0 distinct solutions is a controlled existence proof on a
real recurrent architecture). It sharpens rather than contradicts the brief's own
information-theoretic note: the failure is specifically that MORPH's target is
single-valued under teacher forcing, which (per this paper's own label-vs-energy
ablation) is sufficient by itself to collapse a capable recurrent architecture to
one-pass behavior, independent of whether the underlying task has latent multi-basin
structure.

---

## Synthesis for Q1-Q4 (word count under 300)

All four papers point the same way: a weight-tied loop only earns iterations when its
TARGET has genuine iterative structure — an attractor a single step cannot reach
(Equilibrium Reasoners), or a set of distinct valid answers a single step cannot select
among (Bifurcation Models) — or when the loop is allowed to inject real per-step
randomness that a deterministic map cannot (GRAM). None of these hold for MORPH's
current slot target (one teacher-forced next-span label from a deterministic seed).
Paper 3 (Solve the Loop) is the load-bearing result: on real web-text NTP, an
independently-designed loop (separate attractor module, implicit-diff training) ALSO
saturates at one pass (Fig. 7), and its PPL gain over scale actually SHRINKS (14.8% ->
7.6%, Table 1), arguing against Q2's scale-threshold hypothesis in the tested range.
This is a second, architecture-independent confirmation of the brief's central finding,
not a counter-example — it should be cited as evidence FOR the collapse, not against.

For Q1: yes, likely disjoint, but Bifurcation Models locates the disjointness precisely
— it's the training objective (single-label CE), not text or weight-tying per se; their
own architecture collapses to one solution under label training and multiplies under
energy training on the SAME weights (Table 2).

For Q3: only Bifurcation Models offers a design that stays amortized (no extra params,
no extra test-time iterations) — randomize the slot's entry, replace CE with an energy/
diversity objective across parallel rollouts (dropped at inference).

For Q4: two concrete mathematical answers — attractors reachable only by contraction
from a distant start (Equilibrium Reasoners), and set-valued targets resolved only by
running to convergence under non-collapsing training (Bifurcation Models). GRAM's
stochastic-entropy answer is real but not amortized.

Recommended next arm: the Bifurcation Models energy/diversity design, tested first on a
cheap synthetic multi-branch span task before real web text.
