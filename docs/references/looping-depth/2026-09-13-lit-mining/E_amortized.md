# Lit review E: designs that amortize compute across tokens, and loop memory/state design

Read against the brief at `loop_lit_brief.md`. All five sources fetched and read in full
(papers 1-4 via arXiv PDF, page-by-page; paper 5 via WebFetch on the blog HTML — flagged
lower-confidence sourcing where relevant).

---

## 1. "Almost Free State Prediction Separation" — Langford, Godey, Monea, Artzi, Dong, Fan,
de Rosa, Zhan (arXiv 2609.03807, 2026). NOTE: the brief calls this "Free Pause Tokens"; the
paper's actual title is "Almost Free State Prediction Separation" and the mechanism is the
"free pause token," a sub-idea within it.

### What they measured, and how
1B-parameter decoder (24 layers, hidden 1536, GQA 16q/8kv, sliding window 2048 with a full
layer every 6, seq 8192), Phi-4-derived pretraining data, Muon optimizer, global batch
524,288 tokens, 8xB200 (§4). Single scale only — the Conclusion states explicitly:
"Our experiments here are limited to a single scale (1B) with a single primary seed" (§7).
No smaller-scale run exists, so the brief's request for a cross-scale number cannot be
answered from this paper.

- Iso-token CE at 100B tokens: control 2.8957, full pause (w=0) 2.8673, Δ = −0.0284 nats
  (§6, Table in §B). Abstract rounds this to "2-3 centinats."
- Phased variants (pay the second pass only on the run's tail): 42.5%→pause reaches 2.8691
  (recovers ~94% of the full-pause gain) at 1.33x wall-clock; 75%→pause reaches 2.8756
  (recovers ~71%) at 1.14x wall-clock (Table 2).
- Shared-gated-FFN variant (halves the second pass's FFN cost): full pause 2.8825 vs its
  own control 2.9064 (−0.0239), at 1.35x wall-clock, cheapest point 1.09x at 75% phasing
  (Table 3, §6.2).
- Iso-compute (matched node-hours against the control's own compute-for-loss curve): every
  free-pause variant stays ahead — full pause Δ −0.005, 42.5% split Δ −0.012, 75% split
  Δ −0.013 (Table 4). At true iso-FLOP (~1.9x) the full pause turns slightly POSITIVE
  (i.e. worse), +0.006, and phased deltas roughly halve (§6.3 text).
- Downstream: iso-token, full pause raises DCLM CORE 0.327→0.347 and lowers climbmix BPB
  0.777→0.769, both at 100B tokens (Table 5, §6.5).

### Exact wiring
Two weight-shared streams over the SAME backbone (Fig. 2, §2). The **state stream** `a` is
the ordinary causal pass: it embeds tokens and writes the per-layer keys/values, exactly
like a normal transformer. The **prediction stream** `p` carries no token embedding at all:
at EVERY position it is initialized from one shared learned vector `predict_embedding`
(the only new parameter), forms a QUERY ONLY against `a`'s keys/values at each layer
(dashed arrows in Fig. 2 — causally, `p_i` reads `a_{≤i}`), and writes no key/value of its
own. `p`'s output at every position feeds the LM head and the CE loss falls only on `p`.
Because `p` adds no key/value and no sequence position, it is "free" at inference: no added
context length, no KV cache entry, no decode step (§1, Fig. 2 caption).

Training: two ordered FlashAttention-friendly passes — state stream first (produces cached
K/V), then prediction stream as plain cross-attention over those K/V (§2, "A FlashAttention-
friendly two-pass split"). Four cost-reduction mechanisms, each in Table 1: (i) the two-pass
split itself (4x over a naive flexible-mask single pass); (ii) `w=0` — the prediction takes
no self-window over its own past predictions (a stricter reading than the original SPS
paper's `w>0`, costs only ≤0.009 nats at 100B, Table 1); (iii) a shared gated FFN evaluating
the FFN once per position instead of once per stream (halves the dominant second-pass cost,
§2 equations, costs ~0.005-0.010 nats); (iv) phasing — running plain single-stream training
for a fraction `f` of the schedule then switching on the split for the rest, paying the
second pass on only `1−f` tokens.

Decode-cost accounting (§6.4, Appendix A): at prefill only `x_i` is forwarded (no `p_i`
needed). At decode, `(x_i, p_i)` forward together as a two-token step, sharing the KV read
across both queries via a fused `flash_attn_with_kvcache` call; this stays within ~1% of a
standard decode step in a B200 microbench. The extra compute is FLOPs, not added serial
depth — decode latency is normally set by the sequential depth (L layers x one step per
token), which the free pause leaves unchanged; the extra parallel compute is hidden because
decode is typically memory-bandwidth-bound, not compute-bound (§6.4).

### The mechanism claimed (one paragraph)
State-prediction separation (Monea et al. 2607.01218, cited [25]) holds that one hidden
state cannot simultaneously *summarize* the context (what later positions attend to) and
*be a good predictor* of the next token — the two goals pull the vector in different
directions. Splitting them into two streams over a shared backbone lets the state stream
specialize purely in summarizing while the prediction stream, initialized from a single
learned "pause" vector and given a free query over the state's own KV, specializes purely
in predicting. This is a SECOND non-iterated full-depth READOUT of the current position's
already-computed context, not a repeated pass over the same computation — there is no loop
here, no T>1 iteration count, just one extra parallel pass through all L layers per
position.

### Bearing on Q1-Q4
- **Q1** (is loop geometry disjoint from NTP's geometry?): weak bearing. The gain (0.028
  nats) is real but small and comes from role-separation within a SINGLE pass, not from
  iterating a map. It shows NTP text has some headroom for "looking again" at existing
  context with a differently-purposed readout, but it doesn't test whether REPEATED
  iteration (T>1) buys anything beyond this one extra look.
- **Q2** (scale threshold): no bearing — single scale (1B) only, explicitly scoped that way.
- **Q3** (amortized design giving later passes real work): no bearing on "later passes" per
  se — this design is not iterative at all, and it does not amortize across tokens (its
  extra compute is spent identically at EVERY position, not once per span). It is closer to
  the paid-loop-at-token-granularity idea we already rejected for lacking amortization, made
  cheap only because decode is memory-bound rather than compute-bound.
- **Q4** (math principles for iteration): no bearing — no dynamical-systems framing.

### Concrete arm for our setting
Give the slot's coda reader a single extra non-iterated cross-attention query stream that
reads the CORE's own per-layer keys/values as they are produced during pass 1 of the slot
loop (a "prediction stream" over the loop's first pass), rather than only reading the loop's
final exit state z. Measure: does a single extra READOUT (zero extra iterations) of pass-1
internals recover a meaningful fraction of the 0.18-nat cross-span budget the slot channel
already carries? Prior: MEDIUM. The free-pause paper shows role-separation without iteration
buys ~0.03 nats on generic NTP at 1B/100B; if a similar single extra readout captures a
comparable slice of our (much larger, cross-span-restricted) 0.18-nat budget, that would be
further confirmation that the value is in role-separation/extractability, not in iterating
T>1 times — consistent with "pass 1 does everything."

### Contradicts the brief?
Not directly, but it sharpens the framing: this paper's ENTIRE gain comes from a
non-iterated second look. It does not test or claim that iterating further would help, so it
should not be read as evidence FOR the value of loop iteration — if anything it's a data
point that "one extra look, no iteration" is where the money already is, matching our
"pass 1 does everything, passes 2-6 do <=0.002" finding rather than contradicting it.

---

## 2. "Latent Recurrent Thoughts" — Chen, Fu (arXiv 2609.01117, 2026)

### What they measured, and how
Frozen Qwen3-8B decoder (d=4096, never updated), task-dedicated proposer (bidirectional
Transformer encoder, 4.2M trainable params) + TRM recurrent refiner (7M trainable params,
working dim d'=256) trained per task family on: symbolic (Countdown-4 "CD4", Sudoku, answer
supervision only, no reasoning traces) and natural-language (HumanEval, MBPP, StrategyQA)
(§3.1, §4.1). Table 1 (main results, mean±std over 3 seeds):

| Method | CD4 | Sudoku | HumanEval | MBPP | StrategyQA | Avg |
|---|---|---|---|---|---|---|
| Direct (no CoT) | 27.8±1.6 | 17.3±0.3 | 13.4±1.1 | 28.7±1.8 | 67.4±6.1 | 30.9 |
| Zero-Shot CoT | 30.0±0.6 | 23.9±0.2 | 15.9±0.5 | 35.4±1.6 | 69.9±1.4 | 35.0 |
| SoftCoT | 5.9±0.2 | 10.4±0.1 | 20.7±0.9 | 40.2±2.1 | 70.2±6.5 | 29.5 |
| EBM-CoT | 8.4±0.1 | 17.2±0.4 | 25.0±1.6 | 46.1±3.6 | 71.0±5.7 | 33.5 |
| **LRT (ours)** | **56.7±1.9** | **49.2±1.5** | **37.8±3.3** | **51.5±1.7** | **75.1±2.3** | **54.1** |

Compute: thinking-mode Qwen3-8B (not compute-matched) reaches 85.3/0.5/92.1/49.3/76.4 on
the same five tasks at 8.9-453.9 TFLOP/example vs LRT's ~1 TFLOP/example — "roughly 9x
(StrategyQA) to 450x (Sudoku) more inference compute" for thinking mode (Table 2, §4.3).
Notably thinking mode SCORES 0.5 on Sudoku (worse than Direct) despite spending the most
compute of any method on that task — "token-space search is not a substitute for iterative
computation on a constraint-satisfaction substrate" (§4.3).

Model sizes: proposer 4.2M trainable (P_down 1.0M + P_up 1.0M + 8 query vectors), refiner
7M trainable, frozen decoder 8B (0.2% of parameters reasoned with are trainable, §3.3,
Conclusion).

### What the refiner iterates on, and how many iterations help
Two persistent states in the 256-dim working space: a fast scratch state `z_L` and a slow
integrating state `z_H` (§3.3). The proposer's base latents enter as an external signal
`u = P'_down(L^(0))` that is RE-INJECTED AT EVERY FAST UPDATE, not only at initialization —
"so refinement stays anchored to the problem while z_H integrates the scratch state across
cycles" (§3.3). One high-level cycle: T=4 fast updates `z_L <- f(z_L, z_H+u)` then one slow
update `z_H <- f(z_H, z_L)`. Full unroll: S=3 outer iterations of H=3 high-level cycles
each, 3x3x5 = 45 total passes through the transition block (§3.3, "Refiner unrolling").
Truncated-gradient unrolling (TRM-style, Algorithm 1): only the FINAL cycle (the last 5 of
45 passes) is differentiated; the preceding 40 run under stop-gradient. The refined latents
are a BOUNDED RESIDUAL correction to the proposed base: `Delta = P'_up(z_H)`,
`L* = L^(0) + Delta`, with a small penalty `lambda||Delta||^2`, lambda=0.01 (§3.3, §3.5).

Iteration-count evidence: §4.4 scales refiner PARAMETER count (7M→14M→28M), not iteration
count directly — CD4 rises 56.7→66.3→69.1, HumanEval 37.8→44.0→46.1, monotonic, largest gain
at the first doubling (Fig. 2). But §4.5 ("Depth versus capacity") directly addresses
iteration count at FIXED capacity: "a non-recurrent refiner with the same 7M budget but a
single feed-forward pass reaches only 47.5/33.0 on CD4/HumanEval against LRT's 56.7/37.8;
and a refiner trained at S=H=3 but unrolled deeper AT TEST TIME ONLY keeps improving,
indicating a convergent iterative process rather than a fixed-length circuit." And:
"Decoding the answer after each high-level cycle shows accuracy climbing monotonically
across the unroll rather than appearing only at the final step, while the residual
||L^(t)-L^(t-1)|| contracts toward a fixed point" (§4.5, "Refinement is incremental",
Appendix G). This is a genuine, directly-measured per-cycle accuracy climb — unlike our
result, later cycles here DO something.

### Does the decoder read only the final latent?
Yes. `M` (the frozen decoder) receives `[I; x; L*]` — the task instruction, question, and
the K refined latents as soft input-embedding tokens (§3.4, Fig. 1). The decoder never sees
intermediate `z_L`/`z_H` states; only the refiner iterates over them, and only the FINAL
`L*` is injected.

### The mechanism claimed (one paragraph)
The refiner is a two-timescale contraction: a fast state anchored every step to the
original proposal (so it cannot drift arbitrarily far from the instance) and a slow state
that integrates corrections across cycles, together predicting a BOUNDED RESIDUAL rather
than regenerating the latent from scratch. Re-injecting the anchor `u` at every fast update
— not just at init — is what the paper argues keeps refinement well-posed as an iterative
correction process (closer to a proximal/Newton-style optimization step toward a target)
rather than an unconstrained autonomous dynamical system that can wander or collapse to an
uninformative fixed point.

### Bearing on Q1-Q4
- **Q1**: indirect bearing. This is symbolic/constraint reasoning (Countdown-4, Sudoku)
  with explicit per-instance answer supervision, not generic web-text NTP. It suggests TASK
  STRUCTURE (an actual sequential/combinatorial problem with a checkable answer), not the
  NTP objective per se, may be what makes iteration pay off — Countdown-4's TRM-alone score
  is only 1.2 (the recurrent module lacks "some sequential prior" for a from-scratch
  solver, §4.3/§4.6), showing even a good iterator needs the RIGHT task shape.
- **Q2**: evidence AGAINST a hard scale threshold, with a big confound. The refiner is tiny
  (7-28M params) and shows real incremental depth benefit — so "iteration only helps at 3B+"
  is not universally true. But this is symbolic reasoning with answer supervision, not
  generic LM loss on web text, so it does not directly refute a scale threshold specific to
  NTP.
- **Q3**: the closest design match to "think once, decode cheaply" in this batch — the
  refiner iterates ONCE PER PROBLEM INSTANCE (not per token), producing K fixed latents,
  and the frozen decoder then autoregressively generates the WHOLE answer cheaply,
  conditioned on those latents with no further iteration. This is architecturally the same
  shape as "think once per span, decode cheaply per token." The mechanism differences from
  our slot loop: (a) explicit re-injected anchor every step, (b) predicts a BOUNDED RESIDUAL
  against that anchor rather than an unconstrained new state, (c) two-timescale state
  (fast+slow) rather than one.
- **Q4**: directly relevant. The paper's "contracts toward a fixed point" + monotonic
  per-cycle accuracy is a literal example of a converging iterative map that keeps doing
  useful work because each step is constrained to be a bounded correction relative to a
  fixed external anchor — different from an autonomous map whose fixed point need not
  correlate with task quality (which is closer to what our loop's fixed point looks like).

### Concrete arm for our setting
Add a persistent anchor `u` = a small projection of the slot's PRELUDE-ONLY entry state,
re-injected at every core-loop iteration (not just entry), plus a two-timescale register
(fast state updated per iteration, slow state updated once per cycle), and change the
loop's target from "produce a new state" to "produce a bounded residual correction" (with an
explicit `lambda||Delta||^2` penalty) relative to that anchor. Measure: the K-curve
(K3-K6), and — following LRT's own instrument — an auxiliary decode probe at EACH iteration
to see if accuracy climbs monotonically across iterations rather than saturating at
iteration 1. Prior: MEDIUM-LOW. LRT's task has verified per-instance answer supervision
that our web-text spans lack, and the brief already lists "gradient-conditioned passes" and
"oracle gradient trajectory" as tried-and-met-in-one-pass; but the specific bundle
(anchor re-injection + bounded residual + two-timescale) has not been tried together, and
LRT's own ablation (§4.5, non-recurrent 7M-budget refiner: 47.5/33.0 vs LRT's 56.7/37.8)
shows the RECURRENCE itself (not just the anchor) contributes ~9 points — worth one
controlled run before concluding this family is closed.

### Contradicts the brief?
Yes, usefully: LRT is direct counter-evidence to a strong (unstated) worry that "later
passes NEVER do real work in any recurrent-latent design." They show an architecture where
later cycles measurably climb accuracy under a training regime that differentiates only
the LAST cycle (identical truncated-gradient bottleneck to ours) — meaning our null result
is not an inevitable property of truncated-gradient recurrent training, but plausibly a
property of OUR specific target/anchor/injection design.

---

## 3. "Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive
Reasoning" — Sapunov (arXiv 2604.21999, 2026)

### What they measured, and how
Single-block Universal Transformer with memory tokens (pre-norm attention + SwiGLU FFN,
hidden=512, heads=8, head_dim=64, 3.2M params total), applied iteratively up to K=18 steps,
with Adaptive Computation Time (ACT) halting, on Sudoku-Extreme (81-cell, 17-24 givens,
3.83M train / 423K test instances, §2.1, §4.1). `T` in their notation = NUMBER OF MEMORY
TOKENS (0, 4, 8, 16, 32, 64), not ponder depth — ponder depth is `K` (up to 18, extended to
64 at test only).

- Sharp threshold: T=0 always fails (2.5-2.7% EM, 3 seeds); T=4 is seed-sensitive/borderline
  (5.4%-55.8%); T=8 always succeeds (57.4±0.7% EM); stable plateau T=8-32
  (57.4/56.9/56.4 ± 0.7-1.7%); dilution boundary at T=64 (56.7/54.9/4.5% — bimodal across
  seeds) (Table 3, Fig. 3, §4.2).
- Router initialization trap (§3): default bias=0 gives halting prob p≈0.5 → halts after
  ~2 steps; Graves' recommended bias=1 gives p≈0.73 → even shallower. Across 13 completed
  runs at bias=0, only 4/13 escaped a shallow-halt equilibrium (>20% EM) (Table 1, §3.1).
  Fix: bias=-3 ("deep start", p≈0.05) rescues the worst seed at every memory size — e.g.
  seed 123, T=8: 4.1%→57.9% EM (Table 2, §3.2).
- Depth-state substitution (§4.3): at lambda=0 (no halt-side pressure) minimum halt steps
  fall 17.7 (T=8) → 16.4 (T=16) → 15.5 (T=32); under lambda-warmup (0.001, 20k-step warmup)
  mean halt drops monotonically 11.60 (T=8) → 11.50 (T=16) → 10.29 (T=32) → 8.25 (T=64)
  while EM stays ~57% across the plateau and falls 2pp at the dilution boundary
  (57.09→58.00→56.87→54.91%, Table 4). Total token-steps (memory+sequence positions x mean
  halt) grows only mildly with T (1032→1116→1163→1196, ~13% over T=8→32), so compute is
  "approximately conserved, not strictly fungible" (§4.3 text).
- Lambda warmup: 57.0±1.1% EM (3 seeds) using 34% fewer ponder steps than lambda=0's
  56.9±0.7% (Table 5, §5.1).
- ACT vs fixed-depth: ACT-enabled 56.9±0.7% EM (order-of-magnitude lower seed variance)
  vs fixed-depth-18 53.4±9.3% (range 44.9-63.4%) — Table 5, §4.4.
- Extended inference beyond trained depth (trained K=18, tested to 64): the lambda-warmup
  model PEAKS at step 36 (66% EM, +14pp over step 17's ~52%) then degrades gracefully (64%
  at step 48/63, never crashes); sweet spot ≈2x trained depth (Fig. 7, §5.2).
- Attention head specialization (Table 6, step 17, T=16, best model): three functional
  groups emerge without supervision — memory readers (H4: s→m 0.34, "mixed read +
  constraint"), memory writers (H1, H3: high m→s, 0.81/0.80), constraint propagators
  (H2, H6: s→m≈0.01, structured diagonal/vertical S→S patterns, one head at m→s≈1.0 and
  s→s≈1.0 simultaneously) (§4.5, Fig. 6). In the trapped (0% EM) model, no head develops
  s→m > 0.16 and the constraint patterns of H2/H6 are absent — attention specialization
  itself is diagnostic of whether the model escaped the trap.

### The memory-token design (what the brief asks for concretely)
Memory tokens are N learned vectors, each with its own type embedding (memory vs sequence)
and per-step positional embedding, concatenated with the sequence tokens
`[mem_1..mem_N, seq_1..seq_L]` and processed by ONE shared weight block with bidirectional
attention, iterated K times (Fig. 1, §2.1). Crucially, memory tokens are READ AND WRITTEN
AT EVERY STEP — they participate in the same self-attention and FFN as sequence tokens at
every iteration, not just at entry/exit — so they behave as a genuine, actively-updated
scratchpad rather than a passively-carried register.

### Depth-vs-state result
"Memory tokens and ponder depth function as substitutable computational resources at
consistent accuracy" (abstract, §4.3): more memory buys the SAME accuracy at fewer ponder
steps, and vice versa, within the T=8-32 plateau — but this substitution only exists once
memory is present at all; with T=0 (no memory), no amount of ponder depth reaches
non-trivial performance ("no configuration without them reaches non-trivial performance",
abstract).

### The mechanism claimed (one paragraph)
A single weight-shared recursive block iterated over the raw token sequence has no way to
accumulate PARTIAL solutions or propagate constraint violations across steps except through
the sequence representations themselves, which are also being consumed by the next-step
computation — there's no dedicated buffer. Adding N memory tokens that are actively read
AND written by attention at every recursive step gives the loop an explicit, growing scratch
state that can hold intermediate constraint information (row/column/box consistency in
Sudoku) separate from the token positions; the attention heads then spontaneously specialize
into read/write/propagate roles that implement iterative constraint satisfaction. Without
that dedicated, actively-updated buffer, the paper's data says the recursive block collapses
to doing (almost) all its useful work in the first couple of steps.

### Bearing on Q1-Q4
- **Q1**: bears on task structure, not NTP objective directly. Sudoku is an explicit
  constraint-satisfaction task; the paper doesn't test NTP loss on natural text. It suggests
  iteration earns when the target decomposes into propagatable sub-constraints — unclear
  whether generic web-text NTP ever presents that shape per token, but see paper 5's
  task-dependent topology result for a related point.
- **Q2**: evidence against a strict scale threshold for THIS task (3.2M params, tested up to
  K=64 ponder) — but confounded: this is combinatorial puzzle-solving with answer
  supervision, not general LM loss.
- **Q3**: the single most concrete, testable design element in this batch: memory that is
  READ AND WRITTEN EVERY ITERATION, not just seeded at entry and extracted at exit. Our
  slot design writes the slot input once (a fixed seed) and only extracts z at the end
  (into 2 prefix cells) — the loop never gets an explicit, actively-updated scratch buffer
  distinct from its own hidden state.
- **Q4**: the clearest stated principle in this batch: an iterated map needs a persistent,
  actively read/written state to carry information forward across steps; without it,
  recursion collapses to doing its useful work in step 1 (T=0 fails outright) — this is
  essentially a memory-budget argument (Q4's third category) rather than a fixed-point or
  attractor argument.

### Concrete arm for our setting
Give the slot loop an explicit N-cell memory bank (separate parameters from the slot's own
state) that is attended to (read) AND updated (written) by the core blocks at EVERY loop
iteration — not seeded once and read once at the end — so the memory can accumulate
information ACROSS iterations the way UTM's memory tokens do. Measure: K-curve (K1-K6,
K3-K6 especially), and per-iteration decodability of an auxiliary probe (à la LRT's
per-cycle decode, or UTM's per-step attention-head role emergence). Prior: MEDIUM. This
differs meaningfully from the brief's already-tried "4-cell register" lever (item under
"Rank of z"), which — per the brief — LOWERED rank or left it unchanged; that register may
have been read-mostly / written-once rather than read-write EVERY iteration. UTM's finding
is specifically about active per-step read+write, which is a different, untried mechanism.

### Contradicts the brief?
Partial tension, worth flagging explicitly: the brief records that "a reader with MORE
capacity (4 non-shared blocks after the loop) makes the passes matter LESS and costs 0.006
nats" — i.e. adding capacity hurt in our setting. UTM's memory tokens also add capacity, yet
help a lot. The difference is WHERE the capacity sits: our tried lever added capacity
AFTER the loop (a bigger reader consuming the loop's output); UTM's capacity sits INSIDE
the loop, read and written by every iteration. These are not the same experiment, and UTM's
result does not straightforwardly generalize to "any added capacity helps" — but it does
mean "more capacity" isn't fully closed as a lever; only "more capacity downstream of the
loop" is.

---

## 4. "Full-bandwidth transformer" — Wang, Cai, Zhan, Dong, Fan, de Rosa, Pearce, Langford
(arXiv 2608.08888, 2026)

### What they measured, and how
1B-parameter decoder-only transformer, trained up to 400B tokens on a Phi-4-style mixture
(§4), NorMuon for matrix params (lr 1e-2) + Adam for the rest (lr 5e-4), WSD schedule,
context length 8192 for pretraining, seq extension to 32K + 6B-token instruction tuning for
downstream evals. Runs (page-8 table): 10B tokens/100% three-pass/40B token-equivalent
compute; 100B tokens/75% one-pass+25% three-pass/150B compute; 200B tokens/75%+22%+3% one/
two/three-pass mix/256B compute; 400B tokens/same mix/512B compute.

- Fig. 4 (prefill feedback-pass sweep, 10-task 5-shot LM Eval + validation loss): "most of
  the improvement appears after the FIRST fused prefill pass"; with 2 feedback passes the
  100B-token model reaches the 200B-token standard baseline, and the 200B-token model
  reaches the 400B-token standard baseline — "roughly 2x pretraining data efficiency"
  (§4.1).
- Table 1 (instruction-tuned + long-context, no few-shot; STANDARD/SOFT/FUSED decoding
  regimes defined in §4.2 — STANDARD = no feedback used at inference, SOFT = feedback used
  from a single-pass prefill, FUSED = feedback used after one extra fused prefill pass):

  | Task | FB-200B Std/Soft/Fused | FB-400B Std/Soft/Fused | Standard 200B/400B/1T |
  |---|---|---|---|
  | GSM8K (Pass@1) | 64.52 / 67.93 / 67.55 | 67.90 / 71.00 / 71.80 | 62.93 / 68.39 / 70.13 |
  | MATH-500 (Pass@1) | 43.80 / 45.60 / 45.60 | 46.00 / 45.40 / 48.40 | 42.40 / 46.40 / 47.40 |
  | HumanEval (Pass@3) | 42.54 / 45.06 / 45.92 | 46.50 / 47.20 / 47.60 | 37.16 / 44.85 / 50.01 |
  | MBPP (Pass@3) | 38.39 / 39.80 / 41.22 | 40.50 / 40.60 / 41.70 | 38.61 / 40.28 / 41.93 |

  (Table 1, §4.2; SOFT typically wins on math, FUSED typically wins on coding — "carrying
  hidden state through generation helps [math] reasoning" vs "refining the prompt
  representation before generation is especially useful" for coding.)
- Decode-time overhead: "The added inference cost is independent of context-length and
  model-depth and UNDER 1% PER TOKEN" (§3.1, "Latent feedback is free to serve") — the
  fusion is exactly two D×D matrix multiplications per generated token, negligible against a
  full L-layer forward pass; KV cache and attention are untouched. No added asymptotic
  decode depth: "So T tokens cost O(TL)", same as standard decoding (§3.2).

### Exact wiring, and is it a cross-token or within-token loop?
It is EXPLICITLY a cross-position (token-to-token) recurrence, not a within-token loop
(§3.1, Eq. 3-4). At decode step t: `h_t^L = f_theta(e_t (x) h_{t-1}^L; C)`, where the fusion
`(x)` is a gated linear unit — `e_t (x) h_{t-1} = W^U h_{t-1} . sigmoid(W^G e_t)` (Eq. 4) —
with the hidden state on the VALUE pathway and the sampled token acting only as a
multiplicative GATE (asymmetric by design, to prevent the model from learning a shortcut
that suppresses the state pathway and falls back to plain decoding, §3.1). The paper states
this directly: "No added asymptotic depth at decoding time... What changes is the bandwidth
of the path" and formalizes it with a reachability-set argument — standard decoding lets
layer l at position t read only `{(t', l') : t' < t, l' < l}` (a triangular, depth-limited
region, `|R_std| = Theta(Tl)`, Eq. 2); latent feedback expands this to EVERY layer at every
past position, `|R_lf| = Theta(TL)` (Eq. 7) — "including the shallowest, [reads] the past as
processed by the full stack." This is precisely "a thought carried forward" across
positions, not a loop that revisits the same position.

### Training (scheduled multi-pass objective)
Trained via "temporal parallelism" (§3.3): pass k re-runs the WHOLE stack in parallel over
all positions using pass-(k-1)'s shifted, gated hidden states as input (Eqs. 9-11), so
sequentiality is paid across a handful of PASSES rather than across the sequence. Loss
(Eq. 12): standard NTP on pass 1 PLUS `lambda/(K-1) * sum` of NTP losses on passes 2..K
(lambda=1, no tuning); gradients are NOT detached, so later-pass loss backpropagates into
earlier passes' states, "acting as an auxiliary objective." Scheduling (§3.3, "Feedback-pass
scheduling"): the bulk of training uses the ordinary single-pass objective; feedback passes
are introduced PROGRESSIVELY mid-training — first as two-pass batches, later a SMALL
fraction of three-pass batches. Fig. 3 (1B model, 200B tokens): training with ONLY
single-/two-pass batches (75%/25% mix) performs well at the trained depth but "fails to
extrapolate": beyond that depth, validation loss rises sharply and the hidden-state update
size oscillates rather than decays. Adding just 3% three-pass batches (75%/22%/3% mix)
makes the learned feedback map "behave like a CONTRACTION toward a fixed point," stable to
extrapolation far beyond trained depth — verified out to k=1000 feedback passes at
inference (Fig. 10, appendix) with no breakdown.

### The mechanism claimed (one paragraph)
Standard autoregressive decoding compresses the entire top-layer state — a D-dimensional
vector — down to a single sampled token (at most log2|V| bits) before it can influence the
next step; all deeper, non-verbalized computation is "depth-frozen," readable only by layers
above where it was produced, and can never route back to the bottom of the stack. Latent
feedback fuses that FULL top-layer state (not just the token) into the next position's
input via a gated linear unit, so non-verbalized computation gets a renewed depth budget
and every layer — including the shallowest — can read the past as the full stack already
processed it, rather than only a partially-processed prefix view. The paper is explicit
that the resulting gain is "computational, not informational": since the fed-back state is
a deterministic function of the context already available, no new INFORMATION enters; the
value is purely in making already-present information more directly ACCESSIBLE/usable by
the network (§3.2, "Latent feedback improves computational accessibility").

### Bearing on Q1-Q4
- **Q1**: bears directly and is the single most important finding in this batch for Q1.
  Validation loss, 5-shot LM Eval, math/code generation, and instruction-tuned performance
  ALL improve on ordinary web-scale NTP text (Fig. 4, 5, Table 1) from extra per-step
  compute — refuting the strongest form of "NTP text can never reward iteration-shaped
  compute." But the winning SHAPE here is cross-position full-state relay, not
  same-position revisiting. This reframes Q1: the question isn't "does NTP reward extra
  compute" (yes, clearly) but "does it reward the SAME-POSITION loop shape we built,"
  which this paper does not test.
- **Q2**: no scale-threshold evidence — only 1B params tested, varying TOKEN budget
  (10B-400B), not parameter count.
- **Q3**: directly and strongly relevant, arguably the best-fit design in the whole batch.
  This amortizes at TOKEN granularity for near-zero extra cost (<1% per token, no added
  serial depth): every future token gets FREE, gated access to the full hidden state the
  CURRENT token already computed at full depth, instead of only the compressed token id.
  The "job later positions do that a one-pass map at that position cannot" is: relay the
  UN-VERBALIZED full-depth computation forward, past the log2|V|-bit bottleneck of the
  sampled token.
- **Q4**: directly relevant. The reachability-set formalization (Eq. 2 vs Eq. 7) is a
  precise mathematical answer to "when must an iterated/recurrent channel do work": when the
  standard architecture's reachable-information set at a given layer is asymptotically
  smaller than the full history that layer could in principle use, widening the channel
  captures value from a SINGLE step of relay — no repeated iteration over the same input is
  required. And the explicit "computational, not informational" framing (§3.2) is an
  independent confirmation, from a different group, of our own Lean-checked note that a
  deterministic map adds no new information beyond what the entry carried, and that value
  must come from relay or extractability.

### Concrete arm for our setting
Replace (or augment) the slot's compressed 2-cell prefix write with a GATED fusion, in the
style of Eq. 4, that relays the FULL top-layer hidden state from a span's last core-block
computation directly into the entry of the READER'S first core-block computation for the
NEXT span — state on the value pathway, a cheap learned signal as the gate — rather than a
concatenated/prefix-injected compressed summary. Measure: the existing rank-of-z and
cross-span-budget instruments (currently: prelude ~13 effective dims, loop keeps 13.2,
prefix write cuts to 7 — per the brief) to see whether a wider, gated relay channel raises
the ceiling of what ONE pass can carry without needing more loop iterations, plus the
K-curve as a check that this doesn't accidentally start requiring iteration. Prior:
MEDIUM-HIGH. This paper's own ablation (Fig. 4) shows "most of the gain arrives at the FIRST
recurrence step" — exactly consonant with our "pass 1 does everything" finding — and it
reframes the fix as a CHANNEL-WIDTH problem (the prefix write's rank-7 bottleneck) rather
than an iteration-count problem, which is a testable, low-risk change orthogonal to
everything in the "levers that lower or leave rank unchanged" list in the brief (that list
was about separating z's post-hoc, not about widening the channel that carries z in the
first place).

### Contradicts the brief?
Yes, meaningfully. It undercuts a strong reading of Q1's premise ("is NTP text
iteration-averse"): NTP DOES reward extra per-step compute on generic web-ish text at 1B
scale, measurably, in validation loss and downstream evals — just not via same-position
iteration. Any conclusion of the form "web text does not want extra per-position compute"
would be directly contradicted by this paper's Fig. 4/5/Table 1; the correct, narrower
claim is that OUR specific loop SHAPE (same-position, same-block, T-iteration revisiting)
may not be the shape NTP rewards, while a cross-position full-state relay clearly is.

---

## 5. "Towards Looped Models Done Right — Part I: Topology, Input Injection, Recurrent-State
Design" — Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing,
Xuezhe Ma (IFM Research blog, 2026-07-31)

**Sourcing caveat**: this is an HTML blog post, not a paper with a stable rendered PDF; the
numbers below were extracted via an automated web-fetch/summarize tool (not a direct
page-by-page visual read, which was possible for papers 1-4 via arXiv PDF). Cross-checked
across two independent fetch calls with consistent results (MATH500 +12.00, DROP +2.61
reproduced both times), which raises confidence they are transcribed from the real page
rather than fabricated, but treat these numbers as MODERATE rather than HIGH confidence
relative to the PDF-sourced papers above.

### What they measured, and how
730M-parameter dense model, TxT360 data, seq len 8192, global batch 512 sequences (4.19M
tokens/step), AdamW (lr in {4e-4, 6e-4, 8e-4}), 200-step warmup + cosine decay, wd 0.1,
token budgets 42B/84B/168B/336B (336B = ~460 tokens-per-parameter). Compute-matching across
topology variants is by equal LOGICAL DEPTH (total unrolled block applications) and equal
token budget, not explicit FLOPs or wall-clock — e.g. full-stack recurrence
`R_28^4` (28 tied blocks x 4 iterations) and the sandwich `P_8 R_12^8 C_8` (8 untied prelude
+ 12 tied core x 8 iterations + 8 untied coda) both unroll to 112 logical layers.

- Topology (Table 2, 336B tokens / 460 TPP): sandwich beats full-stack recurrence on
  instance-conditioned multi-step derivation tasks — MATH500 +12.00 points, DROP +2.61
  points, "substantial gains" on BBH-CoT (exact delta not captured by the fetch) — but
  "mixed or negative" on knowledge tasks (ARC, MMLU, TriviaQA).
- Input injection (Table 3, 460 TPP): re-injecting the prelude's encoded state at every
  core iteration helps context-dependent/spec-constrained tasks (MMLU +2.53, BBH-CoT +6.63,
  DROP +1.39, HumanEval+ +5.49, MBPP+ +4.23) but HURTS quantitative reasoning (MATH500
  −3.60, GSM8K −2.51). Stated reasoning: "repeatedly reintroducing a fixed contextualized
  representation may interfere with the iterative state transformations required for
  quantitative reasoning."
- Recurrent-state design (Table 4, 460 TPP): random-vs-direct state initialization is mixed
  (+3.34 ARC-Challenge, +1.22 GSM8K; −1pt each on MMLU/MATH500/HumanEval+/MBPP+); an
  HRM-style shared high/low hierarchy gives "no consistent additional benefit" (three gains
  and three losses >1pt, largest loss on MATH500). Neither ingredient is load-bearing.
- MoE transfer: the sandwich topology, ported to an 8B-resident/0.8B-active MoE trained on
  500B tokens, beats "Ouro" on 8 of 10 benchmarks, notably MATH500 +3.60 and GSM8K +4.70.

### The mechanism claimed (one paragraph)
The post's own stated position is that localizing recurrence to the MIDDLE of the network —
untied prelude and coda doing input encoding and output decoding exactly once, with only the
middle block reapplied — lets the tied recurrent core devote its full capacity to iterative
STATE refinement, rather than splitting that capacity across encoding/decoding duties that
don't benefit from repetition. This is offered as a plausible explanation, not a proven
mechanism; the authors explicitly hedge: "The present experiments establish the behavioral
trade-off but do not isolate its mechanism" (per the fetch). No claim is made about WHY
later iterations specifically do more work — the ablations are architectural (topology,
injection, init), not per-iteration instrumentation.

### Bearing on Q1-Q4
- **Q1**: moderate bearing, and a useful methodological warning. The sandwich topology
  (prelude-core-coda) is exactly MORPH's existing shape, and this post independently finds
  it wins in a compute-matched comparison — but the gain is TASK-DEPENDENT: it shows up on
  multi-step derivation tasks (MATH500, DROP) and is flat-to-negative on knowledge recall
  (ARC, MMLU, TriviaQA). This means an aggregate, generic-web-text CE number (our own
  primary instrument) could hide a real, task-dependent gain the loop provides — the flat
  K-curve we observe may reflect that our eval mixes task types where the loop helps and
  hurts, averaging to near zero, rather than proving the loop is useless everywhere.
- **Q2**: no scale-threshold evidence — the topology ablation is fixed at 730M; the MoE
  transfer to 8B-resident changes architecture (dense→MoE) at the same time as scale, so it
  is not a controlled scale ablation of the topology finding itself.
- **Q3**: no new amortization design — this paper doesn't propose a mechanism for later
  passes to do a one-pass-map's job; it's a topology/injection/init ablation, not a
  per-iteration mechanism study.
- **Q4**: no bearing — purely empirical ablation, no dynamical-systems framing offered.

### Concrete arm for our setting
Since MORPH's topology already matches the winning sandwich shape, the actionable takeaway
is not architectural but EVALUATIVE: re-score the slot loop's contribution on a multi-step,
"provided-context extraction" task shaped like DROP (multi-hop question answering over a
given passage) — distinct from our already-closed generative-math lane (Olympiad/Sudoku) —
before concluding the loop earns nothing on any downstream-shaped task. Separately: given
this paper's finding that repeated re-injection of a FIXED encoded representation can hurt
quantitative-reasoning-style tasks, and paper 2 (LRT)'s finding that repeated re-injection
of an anchor HELPS its refiner — these directly conflict on the surface. Read together they
argue the effect is task/target-dependent (bounded-residual-against-answer vs.
open-ended-generation-against-context), not a universal rule either way; any re-injection
experiment for our slot loop should be paired with both an extraction-style and a
generation-style probe rather than assumed to generalize from one paper.
Prior: LOW-MEDIUM (the math/generative lane is closed per our own notes; a DROP-style
extractive task on non-symbolic text is untried and is precisely the category where this
post found the largest topology gains, +12.00 MATH500 / +2.61 DROP, but our own math lane
closure suggests caution about assuming transfer).

### Contradicts the brief?
No direct numeric contradiction, but a methodological one: it suggests our current
loop-contribution instrument (K-curve on generic held-out web-text CE) may not be the right
lens to detect a real, task-dependent gain from the sandwich topology we already ship — the
same topology that, in this controlled ablation, shows LARGE gains on multi-step-derivation
tasks and near-zero-to-negative gains on knowledge tasks, which would average close to flat
on an unstratified generic corpus.

---

## Synthesis for Q1-Q4 (word count check: under 300)

**Q1** (is loop geometry disjoint from NTP's target geometry?): No paper supports the strong
form. Full-bandwidth transformer (#4) directly refutes it: ordinary NTP on generic web-ish
text rewards extra per-step compute, measurably, at 1B scale — the winning shape is
cross-position relay of the full hidden state, not same-position revisiting. Reframe Q1 as:
NTP wants MORE ACCESSIBLE information carried forward, not necessarily an iterated map over
one position. Paper 5 adds a methodological warning: a sandwich topology's real gain may be
task-dependent (helps multi-step derivation, hurts knowledge recall) and could look flat on
an unstratified aggregate CE — worth checking before ruling the loop out entirely.

**Q2** (scale threshold): No cross-scale evidence exists in any of the five sources for
generic LM loss (#1 tested one scale only; #4 varied tokens, not params). Symbolic-reasoning
papers (#2, #3) show real depth-earning at 3-30M params, arguing against a hard threshold —
but both have explicit answer supervision on constraint tasks, unlike our web-text NTP.

**Q3** (an amortized design giving later passes real work): two credible, DIFFERENT-from-
ours candidates. Full-bandwidth's (#4) gated cross-position relay: cheap (<1% per token),
amortized per-TOKEN, carries full state past the sampled-token bottleneck. LRT's (#2)
anchor-re-injected, bounded-residual, two-timescale refiner: amortized per-INSTANCE, shows
genuine monotonic per-cycle accuracy climb under the same truncated-gradient bottleneck we
use. Neither is "loop the same block over the same position with no anchor," which is what
we built.

**Q4** (math principles): UTM (#3) gives a memory-budget answer — iteration needs an
actively read/written persistent buffer or it collapses to one useful step (T=0 fails
outright, matching our "pass 1 does everything"). Full-bandwidth (#4) gives a
reachability-set answer, independently matching our own Lean-checked relay/extractability
framing: value comes from widening what's ACCESSIBLE, not from adding information.
