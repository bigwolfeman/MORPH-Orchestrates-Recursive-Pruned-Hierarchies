# Lessons learned: looped transformers, slot loops and span models (MORPH + Parcae, 2026-06 to 2026-10)

Status: written 2026-10-29 as the hand-off for a fresh start. Sources: the MORPH tree (`lab/`, `docs/`, `.agents/notes/`), the Parcae
testbed filings (branches `lxtul/*`, `docs/experiments/`), the paper-note ledger and the assistant's memory files. Harvested by ten
read-only agents, synthesised and spot-checked against the sources (the Lean theorems at the cited lines, the carrier-constant note, the
per-token depth probe, the divergence tripwire).

## How to read this

Wolfe's frame for the fresh start: **the core objective of almost everything here was wrong** (next-token CE with span targets that were
too weak, slot targets that one pass could meet). So every performance verdict below ("arm A beat arm B by N nats", "supported") is
inconclusive. What this doc keeps is what survives that:

- **theorems** (Lean, linear or Lipschitz models) and loop **dynamics** we measured repeatedly;
- **mechanisms** and **interpretability readings** (what a state, cell, code or reader actually held or used);
- **loop-contribution behaviour** as a phenomenon (what moves depth use, what only looks like it);
- **measurement pitfalls** that produced false numbers;
- **systems facts** that cost real time.

Tags: **[theorem]** proved in Lean (`lab/theory/`), **[robust]** measured on several arms or seeds, **[one-shot]** one run or one seed,
**[speculation]** labelled so in the source. "Lit" says whether the published literature states it, per our paper notes.

Terms. *K1-K6*: CE at 1 loop pass minus CE at 6 passes, on the same rows. *Plain loop*: Parcae-style prelude, core looped T times
(Poisson depth per sequence), coda, every token through the loop. *Slot loop*: only one state per span (a sentence-like segment) loops.
*Injection*: the per-pass re-supply of the prelude output into the core (Parcae `DiagonalInjection`).

---

## 1. Loop map dynamics and theory

**1.1 A quiet core sits on its injection floor, and the learned decay never moves.** [robust] The slot map's typical gain of about 0.87
is the floor that `DiagonalInjection` gives when the blocks do not respond: `sqrt((704 + sum A^2)/1024)` = 0.864 to 0.867 at the init
`A ≈ 0.447`. Slot maps read 0.870 to 0.883 (blocks add 0.005 to 0.018); the plain loop that earns depth reads 0.92 to 0.97. `A` did
not move from init in any model read, so any gain target is met most cheaply through `A`, not through the blocks.
`lab/theory/tul_exploration/READ-BEFORE-TUNING-THE-LOOP-MAP-0.87-IS-THE-INJECTION-FLOOR.md`. Lit: not found.

**1.2 Washout: an input that enters once is forgotten at rate L^T; re-supplying it each pass keeps different inputs apart.** [theorem]
`washout` (`TulExploration/Washout.lean:44`): dist(F^T x, F^T y) <= L^T dist(x, y). `reinjected_rollouts_stay_apart` (`:76`): with
per-pass re-injection, separation >= ||c - c'||/(1 + L) minus a term that decays like L^T. Linear case: the injection's weight sums to
(1 - lam^T)/(1 - lam), about 4.6 at six passes for lam 0.89. Lit: not found.

**1.3 Two distinct fixed points force the pass to expand; a contraction collapses every particle to one point.** [theorem]
`parcae_needs_expansion` (`Particles/Collapse.lean:178`): two fixed points of x -> F(Dx + b) with ||Dv|| <= d||v|| need 1 <= L·d.
`stable_pair_forces_expansion` (`:228`): gain above 1 somewhere between two stable 1-D fixed points. So "a contractive loop that keeps
several hypotheses" is impossible by construction; keeping K particles needs expansion or a persistent per-particle input.
Measured echo: particles without a persistent identity collapsed; with an identity offset they became one particle plus a small
correction (cosine 0.8). Lit: consistent with the saddle/bifurcation picture; Bifurcation Models (2605.07277) gets many solutions only
under an energy objective, one under labels.

**1.4 A depth-independent target read at a random depth is only met by a one-step map or an entry already at the fixed point.**
[theorem, linear] `depth_invariant_iff` (`MapCause.lean:186`): equal exits at two depths need lam = 0 or (lam - 1) x0 + b = 0. Relay
jobs ask for lam = 0, accumulation jobs for lam = 1 (`relay_zero_iff`, `newEv_zero_iff`). On a settled trajectory the recurrence's
gradient equals the gradient of a per-pass write (`settled_block_is_write`, `:463`): once settled, "looping" and "writing a constant" are
the same thing to the optimizer. This is the formal side of "the loop has no job": with a Poisson depth draw and one target, the cheapest
solution is to finish in pass 1 and hold still. Lit: not found.

**1.5 The single-linear spectrum misses the composed loop map; sigma_max caps do not cure loop failures.** [robust] Composed
sigma_max(J_core) rose 54.6 (healthy) to 800.1 (cliff) while every per-linear sigma_max stayed near 1.5 (0 of 12 over cap 3).
A hard projection pinned sigma_max at 1.50 for a whole run and its realized per-block gain (1.659) was ABOVE the uncapped control
(1.402). Four weight-spectrum interventions failed, two worse than nothing. A uniform rescale leaves singular vectors and ratios
sigma_1/sigma_2 untouched, so it cannot slow an alignment of amplifying directions (which rose x2.9 at the takeover while per-block gain
moved +2.5 %). `lab/experiments/failures/2026-08-24-tul-takeover-cure.md`, `.agents/notes/implemented/architecture/2026-06-19-e8-e10-*`.
Lit: Think Shallow, Solve Deep (2608.18222) independently: sigma_max is a transient amplification, successful operators are non-normal
with sigma_max/rho of 4 to 5.4, and forcing gain above 1 killed training.

**1.6 An unbounded loop state is a power iteration.** [robust] E7 block loop: one block had sigma_max 9.4e4 at iteration 0; the state
norm went 485 -> 1.5e7 in one iteration, then each iteration rotated the state about 2 %; effective rank fell 181 -> 9 in about 3
iterations. A random-direction gain hinge read 0.85 to 1.04 while the top direction sat at 95. The first-iteration scale reading
(`loop/core_gain_t0`) led the divergence tripwire by 500 to 600 steps. `lab/experiments/failures/2026-09-07-arc-e7-block-loop.md`.

**1.7 Web-text loops converge in about three passes; puzzle loops settle in 20 to 35.** [robust] Every stable MORPH/Parcae loop on web
text finishes its change by iteration 3. SFPRM (2610.25646) halts at 33 passes on Sudoku. Lit: not reported for language models.

**1.8 Contraction is not universal across looped nets.** [robust, external models] Huginn's steps are 99.6 to 99.9 % perpendicular to
a shared direction and shrink about 0.8x per iteration; HRM-Text (fixed depth 2) has steps that do not shrink, and its CE worsens past
the trained depth (3.10, 3.53, 4.83). In both, zero-ablating the shared bias direction costs 0.4 to 1.3 nats and a random direction
costs nothing. `lab/experiments/successes/2026-10-02-huginn-loop-geometry.md`, `failures/2026-10-03-hrm-text-loop-geometry.md`.

**1.9 The typical gain averages over directions; the cotangent gain along the loss direction separates regimes.** [robust, toy] Toy
accumulation with the fixed-point term: typical gain 0.81, cotangent gain 0.92 to 1.09 on every pass. Read the map along the gradient,
not on random directions.

**1.10 Divergence ("detonation") in a full-BPTT loop is a backward-product spike near gain 1; the forward stays flat.** [robust]
Spikes start within 0.05 of typical gain 1. Spike steps carry cotangent growth 39 to 2436x while the forward norm stays 0.8 to 1.15x its
median. The tripwire `preclip/total > 1e4` at any step >= 200 caught 17 of 17 detonations with 0 false positives in 44 healthy runs.
Ternary QAT drove the rate (ternary off: 3 of 3 healthy, max preclip/total ~10 to 21; on: 6 of 8 detonated). Clipping the backward
product (cotangent capped at 4x) did not stop a takeover: the weight-path gradient still grew ~800x. `lab/divergence/DIVERGENCE-README.md`.

**1.11 Stability and depth contribution are separate axes, except when they are the same event.** [robust] A gain constraint holds the
slot map at 0.89 with zero spikes to 5000 steps and adds no depth. A terminal fixed-point term (lambda 1.0) held the warmup-0 plain
detonation (0 of 6 vs 4 of 7) at no CE cost, but it is blind to scale growth and to single-sample spikes, and its DOSE decides whether
the loop moves at all (slot loop: lambda 1.0 gives K1-K6 0.0009, lambda 0.1 gives +0.0126). The 1000-step LR ramp stopped
detonations (0 of 9) and also cut plain-loop earning (0.207 -> about 0.04): the doc reading is that earning and instability were one
quantity. `.agents/notes/implemented/architecture/2026-09-07-fixed-point-objective-*`.

**1.12 Rounding inside a loop costs far more than outside it.** [robust] FP8 forward (about 2.7 % relative error per MLP input) cost
55 % of K1-K6 and +0.007 CE@6 for a 1.7 % step saving; a bf16 carrier store (about 0.4 % rounding) cost nothing.
`lab/experiments/failures/2026-10-12-fan-speed-lever-split.md`, `.agents/notes/rejected/architecture/2026-10-12-fp8-ternary-forward.md`.

**1.13 The ternary scale rule sets per-pass strength.** [robust] Absmean ternary gives a layer 0.668 of its latent's Frobenius norm;
the core MLP branch reads out/in 0.19 to 0.24 and K1-K6 0.033. A norm-matched scale (`||W||_F / sqrt(nnz)`) gives 0.86 to 1.10 and
K1-K6 0.185. A learnable scale shrank to 0.68 to 0.79 of mean|W|. A frozen live scale never learned. Lit: TWN (1605.04711) is the
nearest prior. `.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md`.

**1.14 Feeding the core only a small deviation amplifies it.** [one-shot x2] First-iteration amplification 13.28 vs 1.16 to 1.68 for
full-state input; the deviation arm took over at step 1795. `lab/experiments/failures/2026-08-25-scse-*`.

## 2. Loop contribution: what moves depth use

**2.1 Depth use follows the share of the loss that no shallower route can take (the bypass law).** [robust] Every cheaper path around
the loop takes the loop's share:
- the same six core blocks earn K1-K6 +0.0102 on token states and +0.0005 on slot states;
- plain loop with a noise entry 0.185, with a prelude entry 0.033, slot side channel 0.002;
- a span stack as a side channel is worth 0.002 to 0.011 nats; as the only cross-span path (series cut) 0.12 to 0.14;
- a multi-span read gives K1-K6 +0.0014 to +0.0018, a one-span read +0.0029;
- under teacher forcing, passes 2 to 6 denoised the entry they were given and pass 1 never learned (cosine 0.02 for 20k steps).
Sources: `failures/2026-09-13-arc-loop-reads-tokens.md`, `failures/2026-09-21-lxtul-loop-denoise.md`, `docs/span-only-model.md` laws 1, 2.

**2.2 Load is not depth: a job one pass can finish earns its whole value at depth 1.** [robust] The span scan plan carries 0.078 nats
and its value is complete at pass 1; series cells cost +0.0453 at depth 6 and +0.0451 at depth 1; draft cells, pass chains and K/V
memories all move K1-K6 by under +0.001. Every per-pass target tried (staged corruption, oracle trajectory, gradient-conditioned
passes, critic, LoopMTP horizons, graded targets) was met by pass 1; six horizon targets sat within 0.01 cosine of each other.
Per-pass (deep) CE supervision on plain Parcae REMOVED its depth use (K1-K6 +0.0376 -> +0.0015) and cost 0.018 CE@6. Looping only the
last coda blocks per token gave K1-K6 +0.006 to +0.010 and lost CE at 1.2 to 1.3x the step time.
Lit: contradicts the principle in Denoising Recursion Models (2604.18839) and LoopMTP (2608.03624) at our budget; agrees with LOTUS
(2606.31779) that a post-loop readout beats per-iteration readouts.

**2.3 A K-curve measures depth DEPENDENCE, not depth VALUE.** [robust] A mean-16 draw reads K3-K6 +0.277 yet ends 0.104 nats worse than
mean-6 at its own depth. K1-K6 follows the training depth draw (0.002 / 0.039 / 0.048 / 0.071 / 0.170 at mean 2 / 3 / 3 / 4 / 6) while
final CE is flat inside the seed spread. At 160k the parameter-matched flat twin ties the span loop at depth 6 and beats it at depth 1:
most of a span loop's K1-K6 is a worse depth 1. A K-curve needs a depth-1-TRAINED twin at the same config, and a broken depth-1 control
inflates it. Huginn's "earns to half its training mean" did not reproduce: depth earning saturates at the training mean.
Topology alone moves the metric: an Anchor skip (the coda reads h0 + u) takes plain Parcae's K1-K6 from +0.0377 to +0.0008 (47x) at a
slightly BETTER CE@6 (3.7112 vs 3.7256). Because K1-K6 is measured against the arm's own depth 1, it ranks arms in CE@1 order: in one
round every added part raised CE@1 more than it won back, and a diversity term (epivol) that "drove" a fan's K1-K6 to +0.0462 did it
through a worse depth 1 (CE@1 3.802 vs 3.745) and a worse depth 6. Seed spread in K1-K6 was about 0.02 while CE@6 agreed to 0.001.
(Parcae `failures/2026-10-14-parcae-chunk-loop-5k.md`, `...-lxtul-round2-5k.md`, `...-epivol-vs-stabilisers-5k.md`.)

**2.4 The plain loop's value over its depth-1 twin grows with training; the slot loop's does not.** [robust] Plain: about +0.004 at 5k,
+0.0674 [0.0647, 0.0701] at 20k. Slot loop: a constant 0.0075 from 5k to 20k. A span loop's K1-K6 grew 3.7x from 5k to 160k
(+0.0030 -> +0.0110), fastest in the LR cooldown, but its flat twin tied (2.3). Lit: DiscoLoop (2607.00341) has the vanilla loop leading
until about 13B tokens, 400x our 5k budget.

**2.5 Token-loop depth gain is private to its own token.** [one-shot, eval-only] Giving depth 6 only to boundary tokens: they keep 0.898
of their own gain with all neighbours frozen at pass 1; later tokens get 0.004 of it (random 5 % positions: 0.044). The loop refines each
token; it does not compute something other tokens read. Parcae `failures/2026-10-14-plain-loop-boundary-depth-probe.md`.

**2.6 A loop earns only at positions whose loss is measured through it.** [robust] Slot-only TUL: 0.015 nats at 20k. Paying the loop on
every position: 0.168 at 5k, honest under future corruption. `.agents/notes/rejected/architecture/2026-09-02-tul-paid-loop-recipe.md`.

**2.7 Reach: content h spans back arrives at pass h-1 and decays about a third per pass; depth buys reach only behind a window.**
[robust] Hop staircase on a reach-1 chain: K1-K6 by hop +0.011, -0.017, +0.064, +0.054, +0.029, -0.020; arrival at pass h-1 replicated.
Unbounded carry grew state RMS to 5 to 75. A window-4 coarse level earns K1-K6 +0.0082 vs +0.0031 global, at a CE cost depth only
partly recovers. Bounding each core layer's attention to an 8-token window raised plain-style K1-K6 (+0.0113 / +0.0118 vs +0.0083 /
+0.0058) at the same CE: when a pass cannot see everything, depth buys reach. Giving every pass a query into far content made the loop
FLATTER (+0.0004 vs +0.0012). A register's reach dial needs a
segment reset on every position-local op, or it is only a per-layer relay.

**2.8 Width, rank and diversity move instruments, not the loop.** [robust] Four cells, VQ codes, eight prefix cells, three diversity
terms, noise entry and lineages each moved a rank, cosine or oracle gap; twelve slot arms kept K1-K6 in [-0.0001, +0.0033]. Eight
proposals per pass halved depth earning. Width buys reader nats (six cells beat two by about 0.030 "whatever they hold").

**2.9 A loop can be depth-flat on a task it solves.** [one-shot] The core reached 0.938 per-cell answer accuracy on Sudoku, flat from
T = 2 to T = 16 in every bucket; whole-board solves stayed near 1 %. Flatness is not a verdict on the task.
`lab/experiments/successes/2026-09-08-arc-e17-sudoku-depth-grid.md`.

**2.10 Earning depends on the corpus.** [one-shot per corpus] Olympiad math +0.0111 and Sudoku text +0.0049 are the largest clean
slot-loop K-curves; web-text slot arms stay at the floor.

**2.11 Index-keyed conditioning and gates are identity escapes.** [one-shot] An input-blended gate learned to front-load the work and
then copy. AdaLN iteration conditioning cut earned depth about 18x; zeroing those weights at eval cost 0.001 nats, so the damage is
done in training. (Baseline was leak-confounded; ratios are indicative only.)

## 3. Representations and readers (interpretability)

**3.1 The carrier constant: after pass 1, 99.6 to 99.8 % of a slot carrier's energy is one vector shared by every slot.** [robust]
It cancels in the Hyper-Connection stream mean, so probes reading the mean missed it (and reported "98 % one direction" at a LayerNorm
target, a stream-mean artefact). Removing it at eval costs +0.11 to +0.13 nats; a uniform HC rebuilt it by another path; batch-mean
centering detonated. Slot-specific share falls from 54 to 57 % of energy at entry to 0.2 to 0.4 % after pass 1.
`.agents/notes/rejected/architecture/2026-10-05-slot-loop-carrier-constant.md`, `successes/2026-10-05-snap-dataflow-probe.md`.

**3.2 A shared slot embedding dominates slot norm and collapses rank; pooling and the loop remove more.** [robust] Content term alone:
rank 38.3; with a shared `E_slot`: population rank 1.13; trained slot states sit at effective rank 1.7 to 4.8 (later corrected ruler:
13.85 per row on the shipped probe). Slot diversity falls with span length at slope about -0.5 (three caps); the loop destroys about 10x
more diversity than pooling does.

**3.3 Earning plain loops are attractors; slot loops are not.** [robust] Attractor score 0.9998 for the earning plain loop against 0.82
to 0.86 for strict slot loops. The spectral gap ratio (about 1.05 on both) does not separate them.

**3.3b Under a skip connection, each hierarchy level learns to ignore its incoming update.** [one-shot, CPU probe] In spiral
hierarchies with an Anchor skip, removing a level's update changed its output by 2.9 to 5.8 % although RMS(u)/RMS(h0) was 0.29 to 0.70;
a second cycle changed the final token update by 0.096 %. All 8 cycled-hierarchy runs read K1-K6 0.0000. Coarse levels of 16 to 64
tokens were worth -0.0002 to +0.004 when zeroed once the coda could read tokens. When a coarse level is the ONLY far path (block-strict
spiral) it carries about 0.13 nats, and still trails the local-fine control.

**3.4 Writes and reads fail separately.** [robust]
- An additive prefix write delivers no decodable content, even a perfect one: planting the true target at the slot prefixes moved CE by
  -0.0000. A gradient-carrying write under a mask did give a reader (shuffle 0.06, zero 0.15 to 0.18).
- The coda read the loop's exit and its entry equally well (to 0.003 nats) while a gradient-fitted z was worth 0.9 to 2.6 nats to the
  same coda: the reader has capacity; the writer put nothing into the directions it reads.
- Plan worth splits into write, dynamics and read: span specificity of a write was 65 % in one cell and 0.4 to 3 % in others, and a
  takeover erased it to about 1 % within a few hundred steps.

**3.5 A frozen reader prefers no state to a confidently wrong one.** [robust] The same cell reads -4.64 nats to a frozen coda and +0.177
(80 % span-specific) once only the reader adapts. A frozen reader's K-curve tracks genericity (cells drifting to the corpus mean), not
content. Lit: DiscoLoop's misaligned-but-decodable loop token is a possible match [speculation].

**3.6 Deterministic heads on a high-entropy future converge to the marginal; sampled codes are unconditional draws.** [robust]
- A mean-seeking plan head reached 4 % better than unigram.
- A sampled next-span code was 0.27 nats worse than no code (the deterministic code was worth 0.36): the code is a copy of text the past
  does not determine, so p(z | past) is broad. Best-of-4 selection ratio 0.91 = the order statistic of interchangeable draws.
- A KL-priced span code sits on whatever free-bits floor it is given (1.0 to 1.6 nats per span) when the decoder has a token path inside
  the span. A fixed-rate FSQ code (11 nats per span) is unpredictable: the prior learns 0.5 to 0.7 nats of it, and generation feeds
  near-random codes.
- A flow "thinker" over span latents put 1 % of its loss on the past (0.3125 with the past vs 0.3161 with a null).
- A PREDICTED continuous concept (LCM-style) decoded worse than no concept (CE 14.91 vs 14.63, uniform 10.80; the true concept gives
  0.233). A variational concept collapsed (KL 0.0185 nats per token, prior sample = the bound). Hiding decoder input tokens buys concept
  bits and costs likelihood (KL 0.019 / 0.042 / 0.109 at 0 / 25 / 50 % dropout; bound +0.108 / +0.275).
- A discrete plan (span NCM) cost the bound what it saved (level 1: costs 0.226 nats per token, saves 0.118; three levels: token CE
  -0.64, bound +1.04). Lit: NCM (2610.14162) reports no likelihood.
- Under teacher forcing a committed PRIOR draw carries no target information (proper scoring makes the optimal leaf ignore it); CE
  through samples rewards hedged draws. Commitment needs a posterior at train.
- A learned noise scale in a loop switches itself off (sigma/r 0.1 -> 0.00026) under a smooth multi-sample bound.

**3.7 Cells blur unless one cell must work alone; diversity terms get gamed.** [robust] A shared write, a mean read or a reader trained on
a mixture gave cell cosines 0.94 to 0.997. Winner-take-all with a write-all read kept cells distinct (rank 2.82 of 3); a set reader kept
cosine at -0.28. Pairwise repulsion found a rank-1 sign split on a fixed axis (`++--` in 2536 of 2573 slots, mean cosine at -1/(K-1));
a scalar epiplexity term was paid by gating one live stream; volume plus epiplexity was the one term not gamed. A uniform mixture over K
streams is bounded by log K / mean span length. Stream identity is not kept by supplying it (four entry draws at cosine 0.99 by pass 0,
entry norm grew 11x).

**3.8 Copying is most of the cross-span gap, and a pooled state does not hold token identity.** [robust] Far-bigram tokens are 17.9 % of
tokens and 85.7 % of the strict-slot model's +0.294 gap to plain; bigram repeats 33 to 256 tokens back are 15 % of tokens and 65 % of the
gap, which falls off at 256 = the plain model's attention window. A normalised pointer head closes about two thirds; the pointer is worth
0.31 to 0.405 nats on span models. In a fan, cell 0 became a copy cell (wins 59 % of spans, 0.126 of 0.170 copy worth).

**3.9 A positional gain can look like a copy channel.** [one-shot] A pseudo-token gate stayed at 0.001 to 0.009; the "channel" was
RMSNorm eps plus q/k norm turning a 9e-5 input into full-size keys, and its picks recurred no more than random tokens.

**3.10 The cross-span budget on web text is small.** [robust] With no slots, about 0.40 nats per token of cross-span information at 5k;
zeroing slot cells costs 0.19 to 0.20 nats; loop depth is about 1 % of the slot channel's worth. A span model's CE gap to plain is a
CONTENT gap, not wording: a free-running meaning grade (frozen encoder on 1 to 4 decoded spans) gives the same ordering, flat over
horizon (Parcae `successes/2026-10-28-parcae-span-grade.md`).

**3.11 Token throughput is not CE per second.** [one-shot, equal wall clock] A span-only model trained 3.8x the tokens per second and
needed about 6x more tokens for the same CE; at an equal 3 h 25 min it trailed plain token Parcae by 0.314 nats CE@6. The per-token loop
it skipped was the part that earns (K1-K6 +0.14 vs +0.011). Lit: hierarchies are reported to lose at low data and win near 60B tokens;
untested here.

## 4. Measurement pitfalls (each produced a false number here)

| pitfall | what happened | guard |
|---|---|---|
| causal leak through a non-attention channel | a GLA retention carry across loop iterations let positions see the future; the "0.233 nats depth earned" was the leak (carry off: K1 4.622, K6 5.742) | corrupt-future and carry-off probes on every depth claim; list every op that mixes positions |
| frozen component in the reader | frozen-coda K-curves measured genericity (-4.64 vs +0.177) | adapt the reader before grading a latent |
| instrument runs a different path | a bare front probe gave slot rank 5.76; the trainer path gives 13.85; a whole target grid was void | instruments call the model's own forward with its kwargs |
| hindsight fits | fitted-z "headroom" of 0.9 to 2.6 nats used the answer; a causal fit was 0.25 nats WORSE than the loop's own z | no oracle that sees the target |
| padding and width | a trajectory prefix K-curve (+0.034) was pad cells; same-width control +0.002 | width-matched controls |
| unshared randomness | per-rollout dropout masks gave fake width (0.0065 vs 0.00006) | share noise across compared branches |
| oracle over samples | min-over-noise floor: four near-copy streams (cosine 0.99) still read oracle gap 0.112 | subtract a near-copy floor |
| seed noise | same-code reruns differ 0.006 K1-K6 / 0.009 CE@6 vs a 480-row CI of ±0.0006; two seeds differ 44 % of mean K1-K6; rounding-only changes decorrelate a pair in about 11 steps; a 5 % test needs about 30 seed pairs | seeds, not rows, set the error bar |
| non-determinism | `index_add_` in a bag mean non-identical 20 of 20 repeats, not flagged by `use_deterministic_algorithms`; identical runs split on the first backward | bit-reproducible mode costs 2.28x |
| CE above uniform | absolute CE 12.64 > ln(49152) = 10.80 made all contrasts nonsense; a row bootstrap cannot see it | check CE vs ln V first |
| bf16 readings | the gain hinge read 0.97 fused vs 0.884 in fp32; a "bit-faithful" HC reference ran einsum in bf16 (+10.9 % loss) | read maps in fp32 |
| Jacobian probes | a pad slot at h = 0 gives an RMSNorm Jacobian ~1/eps (sigma_max 1.5e6); `load_state_dict(strict=False)` silently dropped every MLP tensor | validity masks; load through the trainer's loader |
| dead branches | a shape guard returned zeros (`S // m == 0`): three of six core blocks ran at half attention, while the gate still spent 0.5 on it | check per-block output sizes |
| depth curves before a detonation | readings at step 2500 (+0.043, +0.060) belonged to runs that detonated later | read only stable runs |
| fixed vs sampled depth | a clamped Poisson (mean = max) still runs 1/3 of rows at depths 1 to 5; a fixed-depth model is out of distribution at other depths | state the depth law |
| truncated BPTT | a global no-grad prefix left 27 to 57 % of samples with no loop gradient | front-pad so every sample's last passes carry grad |
| generation readouts | sampling a position training never supervised made boundary artefacts (31 to 45 % lost spaces); gen-PPL rewards repetition ("the the the" 1.46 vs real text 32); diffusion readouts depend on steps (6 steps: gen-PPL 10k, 16 steps: 380 to 775) | diversity guard; sample what was trained |
| label wiring | a one-token SFT label shift trained copying (ppl ~1.0); a multi-hot CE flattened into a single-hot kernel read ~log(V)/s | loss at init near ln V |
| validation overlap | offline probe rows were the first ~420 docs, trained at steps 0 to 80; an olympiad holdout had 1,324 docs verbatim in training | dedupe eval against train |
| copy head that cannot abstain | a pointer head with no null mass gave pessimistic CE; re-scoring old checkpoints changed CE < 0.0001, retraining with the fix moved it 0.044 | fix in training, not in scoring |
| retractions do not propagate | the retracted 0.233 still stood in CLAUDE.md and four notes | one home per number |
| gradient share vs loss weight | the slot label carried 2.8 % of loss weight and about half the gradient reaching h_i | measure gradient routes |

## 5. Systems facts

- **Looped steps are launch-bound.** MORPH LXTUL ran 516 ms/step vs Parcae's 277 at equal FLOPs, with 31.3k launches vs 8.8k. The plain
  Parcae step's single host thread sits at about 96 % of its core.
- **Logged tok/s was a host rate** before 2026-10-12 (139 ms host vs 214 ms wall); the old 23.5k to 44k figures are about 1.5x high.
  Profiler GPU-busy time is not wall clock (a "1.47x" was an objective change; the real lever was 1.18x).
- **Compiled flex attention and `create_block_mask` recompile on every new shape and Dynamo keeps every graph**: a decode tool hit 47 GB
  host RSS and was OOM-killed. Decode with `torch.compiler.set_stance("force_eager")` (flat RSS, and faster with no per-shape compile).
- **A byte gate proves only the path it runs**: programmatic dependent launch passed two bit-exact gates (45 steps at M = 4) and then broke
  training at other M, later steps and after the LR ramp.
- **Memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory on the 5090 for looped arms (8.17 GB reserved-but-free OOM
  without it); resident memory runs 2.7 to 4 GB above allocated because the looped core's active-set shapes vary; gate on reserved.
- **Library tricks can be no-ops**: bnb's AdEMAMix8bit allocates m1 and m2 regardless of beta1 (3.05 vs 2.03 B/param); per-tensor
  Python diagnostics cost 71.8 ms/step through about 900 GPU-to-CPU syncs.
- **Loops are expensive per step at short horizons.** At 5k on the Parcae testbed: gpt8 + pointer 80 ms/step (CE@6 3.6996), flat14 +
  pointer 123 ms (3.6731), plain looped + pointer 240 to 283 ms (3.72). Horizon-dependent (deep models converge slower), but the step
  cost is a fact.
- **Shared-weight gradient accumulation launches a lot**: a full cross-span BPTT slot loop profiled 11,172 addmm copies from Inductor's
  shared-weight gradient adds; batching and larger graphs gave 1.5x.
- **Reduction order matters to Muon-style updates**: CUDA streams changed the largest parameter update by 61.8 % of the reference norm,
  far beyond float tolerance; the code was removed.
- **A fix that lives in a gitignored script is not in the run** (an alpha cap; the control diverged 5 of 5).
- **`init_from` silently reset a trained embedding** (1 of 478 frozen tensors changed); diff checkpoints against the ruler.

## 6. Retracted numbers (do not cite)

- l2cap "0.233 nats depth earned" and its identity-escape table: retention-carry leak.
- Takeover "cured": the second seed took over. "THE TUL DIVERGENCE FIX" (alpha cap 1.0): holds one seed.
- Slot rank 5.76 / cosine 0.71: bare-front probe (13.85 / 0.52 on the shipped path).
- Absmean mask K-curves 0.020 to 0.060: starved core. Plain 0.185 and paid 0.168 as slot yardsticks: different entry and recipe (slot
  yardstick is about +0.002).
- Trajectory prefix +0.034; arm A +0.032 and denoiser +0.56: pad cells and frozen-coda genericity.
- Fitted-z headroom (-0.94, -2.63, -1.55): hindsight.
- Winner K1-K6 +0.0155: a high draw (reruns +0.0095). "WTA holds pass-0 write near 3x": seed 2 grew 23x.
- "The K-stream branch closes": retracted by its own amendment (untrained reader).
- Olympiad E16 gaps: contaminated holdout. Hinge gain_est 0.94 to 1.06: bf16 reference.
- Pre-2026-10-12 tok/s figures: host rates. Span-CALM CE 8.92/4.55: an 8-sample estimate of an implicit sampler, not a likelihood.
- "Set loss destroys superposition" (our paper-note reading): the paper's Appendix F.1 shows a training-budget effect.
- `val/ppl_tokens` on the TUL path: a mean of exp(CE).

## 7. Open contradictions (unresolved in the docs)

- Healthy sigma_max: about 1 (iterative-map note P1) vs 22 to 115 (its own E2).
- The l2cap recipe is filed `implemented` and names a sigma 1.5 projection the winner; its correction says the cap kills depth earning.
- Contractivity as the general target (MORPH CLAUDE.md core mental model) vs HRM-Text, which is not contractive and works at its depth.
- Slot core "blocks are quiet" (READ-BEFORE, 5000-step models) vs "every block gain 1.00 to 1.02, the six blocks build the map"
  (DIVERGENCE-README, post-ramp onset): likely different phases.
- Ramp cures detonation (0 of 3, 0 of 2) vs A1 seed 2 detonating at 2041 under the ramp.
- Ternary cusp-vault trigger named in two filings vs absent in the onset capture.
- Coda reads the exit better than the entry (+0.0235, staged mask) vs exit and entry read alike (noise-entry, progressive).
- Span decoder on the LXTUL fast base: +0.07 nats; on the ungraded fan: -0.022. The bases disagree on sign.
- The paid loop listed as a standing positive (9-26 ledger) vs rejected as "not TUL".
- Span loop K1-K6 +0.011 at 160k as a contribution (span-only doc law 7) vs mostly a depth-1 handicap (flat-twin filings).
