# Awesome-Loop-Models mining against the slot-loop failure (2026-09-13)

Six themed reads of 26 papers from https://github.com/huskydoge/Awesome-Loop-Models (210
entries indexed in `awesome-loop-models-index-2026-09-13.txt`), each read against the brief
in `00-brief.md` (the measured failure: the TUL slot loop's passes 2-6 earn <= 0.002 nats
on tokens across eleven-plus arms). Reports A-F are the readers' own; this file is the
synthesis. Sourcing: E read papers 1-4 from the arXiv PDFs page by page; A-D and F worked
from arXiv HTML through a summarising fetch, cross-checked on the load-bearing numbers.
Treat exact decimals as "as extracted".

## Wolfe's four questions, answered

**Q1. Is the geometry the loop needs disjoint from what next-token prediction needs?**
Not in general, and the literature locates the disjointness in OUR design. Four results:

- Two loops trained on real web text saturate at one pass, like ours: Solve the Loop (Attractor
  Models, 2605.12466) reaches its peak at T=1 at 140M/370M/770M on FineWeb-Edu (Fig. 7), and
  the Full-bandwidth transformer (2608.08888) gets "most of the improvement after the FIRST
  fused prefill pass" (Fig. 4). Pass 1 doing everything is the published norm for NTP, not a
  MORPH defect.
- Plain looping of EVERY position does earn on web text at matched FLOPs: Sparse Layers
  (2605.09165, Table 7: looped 50.2 vs base 55.4 ppl at 10 % FLOPs saved), Parcae (2604.12946,
  Table 5, 140M: 21.48 -> 19.06 ppl), SMELT (2609.01343, 200M active). What earns is the
  whole-sequence loop, the shape our plain model has and our slot loop does not.
- Bifurcation Models (2605.07277, Table 2): the SAME weight-tied architecture collapses to one
  solution under single-label training and finds 13.6 distinct solutions under an energy
  objective. The single teacher-forced next-span label is sufficient by itself to collapse a
  capable loop to one pass.
- The memory-budget separation (2605.30757, Thm 4.2): a loop over s persistent slots is
  bounded by s.d.p bits however long it runs; only the full sequence-state loop is in the
  memory-rich regime. Our slot loop is the s = 1 column of their Fig. 1, the worst column,
  and our low-variance flatness across arms matches their subdiagonal collapse signature
  rather than their seed-dependent above-threshold regime.

**Q2. Is 3B the scale where this starts working?** No published evidence of a threshold for
loop depth-earning on web-text loss, and three independent papers see the effect at or
below 330M: Parcae at 140M (Table 5), SMELT at 200M active with the gain GROWING with
compute (6.8 % -> 18.0 % FLOPs saved from 1e20 to 1e21), and the iso-depth law
(2604.21106) with a recurrence exponent phi = 0.46, CI [0.41, 0.53], flat from 11M unique
params upward. Two mechanisms SHRINK with scale: RecurTrace's loop-memory NLL gain (0.6B
-0.15 nats to 88B -0.02) and Attractor Models' ppl gain (14.8 % -> 7.6 %). The 3B belief is
true of full-vocabulary hard multi-token prediction (Gloeckle 2024) and false of cheap
cosine/embedding variants: LoopMTP (2608.03624) gains at 260M with a soft t-ahead target.
Caveat: Parcae's own isoFLOP grid (Table 6) has looping LOSING to fixed depth at 2 of 6
tested 140M budget points; the win is a property of the fitted optimum, not every point.

**Q3. What amortises compute and still gives later passes a job?** Three shapes in print,
none of which is "loop the same block over one position with a fixed target":

1. Cross-position relay of the FULL top-layer state (Full-bandwidth transformer): the last
   token's whole hidden state, gated, enters the next token's bottom layer; < 1 % per-token
   cost, 2x data efficiency at 1B, most of it from ONE relay step. Value is reachability,
   "computational, not informational", which is our Lean note's relay/extractability
   claim stated by another group.
2. Per-pass routing (Sparse Layers, Fig. 5: 25-53 % of tokens get disjoint expert sets
   between passes) and loop-time attention over the pass trajectory (RecurTrace): later
   passes get a different COMPUTATION, not a different target. MORPH has a tile router.
3. Think-once-per-instance refiners with an anchor re-injected every step and a BOUNDED
   residual target (Latent Recurrent Thoughts, 2609.01117): per-cycle accuracy climbs
   monotonically under the same truncated-gradient training we use, on tasks with a
   checkable answer.

Also in print and directly about our wiring: Tiny Autoregressive Recursive Models
(2603.08082, Fig. 3) find the terminal-iterate-only readout is the ONE design on their
ladder that cannot learn cross-position dependencies at any scale; TUL's z-into-prefix is
that readout. Our queued `prefix_source: trajectory` arm is the flat-readout test.

**Q4. When must an iterated map iterate?** Five necessary conditions, none sufficient:

- Capacity: the persistent state must hold the task's concurrent items (memory-budget
  theorem; Universal Transformers Need Memory, 2604.21999: T = 0 memory tokens fails
  outright, memory and depth are substitutable above 8 tokens).
- Spectral gap: a looped map with layer norm learns exactly the power method (2606.00605),
  and iterations needed scale with lambda_{r+1}/lambda_1. A wide gap converges in one pass
  and the rest rotate: our measured rank collapse 181 -> 9.
- Target structure: an attractor a single step cannot reach (Equilibrium Reasoners), a
  SET-valued answer (Bifurcation Models), or per-pass targets whose available information
  strictly differs by pass (Denoising Recursion Models; Looped Flows).
- Path independence (2211.09961): converging to the same limit from any start is what
  makes extra depth safe; it does not make the limit informative.
- Saddles (Fractal basins, 2609.04963): variable settling time arises only near a
  multistable-to-monostable bifurcation on discrete near-miss solutions. Every task in that
  paper is a puzzle; no language task shows it. Our monotone convergence is their
  pre-bifurcation picture.

## Claims in our ledger the reads correct or sharpen

- "Web text does not want depth" is refuted twice over (Sparse Layers Table 7; Full-bandwidth
  Fig. 4). The flat thing is the slot loop's shape and target, as the ledger already says.
- Reader D says every arm of ours restaged one label per pass. Not so: the staged corruption
  target and the oracle trajectory both ran and were met in one pass. Their principle
  (information must differ by pass) held in those arms and still did not open passes 2-6,
  so on web text the corruption route is weaker than the puzzle papers suggest.
- Reader C's input-injection lead is closed: `_tul_core` injects the seed every pass
  (`reinject_seed_every_pass` is refused as a no-op).
- Two-Scale Latent Dynamics (2509.23314) reports consecutive-update cosine settling at
  +0.5 to +0.65 on a healthy loop; ours is -0.2 to -0.6. The sign differs, so "reaches a
  fixed point" is not one signature; the path to it is the reading.
- Looped Flows: saturation of passes 3-6 does not prove passes 1-2 are idle. Our K3-K6 is
  the right instrument for the first claim only.

## Cheapest new instruments, none run

1. DiscoLoop's logit-lens alignment probe (2607.00341, Table 1b): is the coda's target
   decodable from z while cos(z, target embedding) is low? Training-free.
2. Asymptotic-alignment score (2211.09961, Alg. 1) on the prelude-entry and noise-entry
   arms side by side.
3. Per-row sigma_1/sigma_2 of the pass Jacobian against per-row K1-K6 (power-method test).
4. Attention-sink mass per pass (SMELT Sec 6.4): an extractability edit CE cannot see.

## What this changes for the ship path

Nothing in the 26 papers rescues a one-slot-per-span loop on a single-label target. The
compute-reducing designs that do earn in print are whole-sequence loops at an operating
point (Parcae Table 6), or a one-step gated relay of the full state across tokens
(Full-bandwidth). The depth ladder (`lab/experiments/planned/2026-09-13-arc-depth-ladder-ship.md`)
reads our own operating point; the Full-bandwidth relay is the one new design worth a
prereg if the ladder says depth 1 ships.
