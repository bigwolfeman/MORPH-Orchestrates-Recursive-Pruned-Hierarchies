# The carry decays with hop distance, and training is what repairs it

Date: 2026-09-19. 36 runs, 0 failures, RTX 3070 (`ssh 3070`) sharing the GPU with the
hop-distance probes, 2 h 48 m wall. Pre-registration:
[`../../successes/2026-09-19-toy-eliminate6-hop-distance.md`](../../successes/2026-09-19-toy-eliminate6-hop-distance.md).
Code: [`lab/toy_slot_loop/`](../../../toy_slot_loop/), commits `2f66d40`, `60d26f4`.

- [`tables.md`](tables.md) — every aggregated table.
- Raw JSON, the grid log and the runner: `ignored/experiment-artifacts/2026-09-19-toy-eliminate6/`
  (36 grid cells, 2 smoke cells, 1.5 MB). Regenerate with
  `python lab/toy_slot_loop/aggregate.py ignored/experiment-artifacts/2026-09-19-toy-eliminate6`.

Per-pass tap self-check over all 36 runs: worst relative error **1.13e-07**.

## The headline

Iteration 1 found that a linear probe reads the alive set at 1.000 on a RANDOM-INIT loop
and concluded that carrying a set costs this architecture nothing. That was measured on a
two-hop chain. On six hops it is false.

`dead` — "is symbol c already eliminated" — read at the ANSWER slot after t passes, where
pass t reaches exactly hop t. Balanced accuracy, held-out rows, model frozen:

| cell | hop 0 | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 | hop 6 |
|---|---|---|---|---|---|---|---|
| **random init, d96** | 1.000 | 0.994 | 0.881 | 0.774 | **0.668** | 0.676 | 0.682 |
| **random init, d192** | 1.000 | 1.000 | 0.953 | 0.819 | **0.713** | 0.727 | 0.739 |
| `exit` d96 (0/5 solved) | 1.000 | 1.000 | 0.979 | 0.999 | **0.999** | 1.000 | 1.000 |
| `exit`+fp d192 (4/5 solved) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| `staged` d96 (0/5) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |

An untrained loop carries a fact perfectly for one hop, loses a third of it by hop 4 and
never recovers; every trained cell carries it perfectly to hop 6 and beyond. **Carrying
across depth is learned. "Carrying costs nothing" was a two-hop artefact**, and the honest
version of iteration 1's sentence is: carrying costs nothing for one or two hops.

The twin says the same thing mechanically. A one-token change at span 2 moves the answer
slot's state by a cosine of 0.0002 at hop 4 in an untrained loop (drop pass None in 6 of 6
random-init cells) and by 0.009 to 0.04 in the trained ones (drop pass 4 in 28 of 30).

## What stops the stuck seeds: the last two hops, not the readout

The answer needs the candidate set, which sits at hops 5 and 6. `in_set` at the answer
slot:

| cell | hop 3 | hop 4 | hop 5 | hop 6 | hop 8 | escaped |
|---|---|---|---|---|---|---|
| random init d96 | 0.550 | 0.556 | 0.562 | 0.566 | 0.572 | 0/3 |
| `exit` d96 | 0.661 | 0.897 | 0.891 | 0.888 | 0.894 | 0/5 |
| `staged` d96 | 0.669 | 0.902 | 0.882 | 0.873 | 0.879 | 0/5 |
| `exit` d192 | 0.662 | 0.888 | 0.886 | 0.896 | 0.930 | 3/5 |
| `exit`+fp d96 | 0.668 | 0.899 | 0.877 | 0.913 | 0.934 | 2/5 |
| **`exit`+fp d192** | 0.664 | 0.882 | 0.898 | **0.952** | **0.979** | **4/5** |

0.917 is the ceiling reachable from the eliminations ALONE: knowing the five eliminated
symbols identifies five of the six candidates, and the best a state with no candidate-span
information can do for the sixth is guess. Every stuck cell sits at that ceiling and does
not pass it. The cells that pass it are the ones that solve the task, in the order of their
escape rate. **The chain fails at its last two hops, and the failure is visible in the
state, not in the head.**

The plateau agrees: the modal stuck seed sits at **1.0986 = ln 3**, the depth-4
reachability floor, which is exactly "all five eliminations, no candidate set". 8 of 20
stuck seeds are within 0.02 of it.

## The fitted reader: the head is not the bottleneck

The instrument built to separate "the head cannot read it" from "the state does not hold
it" says the head is innocent. On stuck seeds, coda value CE minus fitted-reader value CE
at depth 6: **median +0.030**, above 0.15 on 1 of 20 seeds. A ridge-selected linear reader
fitted on the exit state recovers essentially nothing the coda is not already reporting.

On a random-init loop the reader reads 2.087 / 2.026 / 2.013 / 2.002 nats at forced depths
1 / 4 / 6 / 8 against a chance of ln 8 = 2.0794. **An untrained loop hands a linear reader
about 0.08 nats after eight passes**, even though every fact is formally inside the window
from pass 6. (That instrument had to be fixed mid-flight; see the prereg's amendment 1.)

## Escape rates: the basin is back, and width is the biggest lever measured

| cell | escaped | escape step | value CE @6 | K1−K6 | s/run |
|---|---|---|---|---|---|
| `exit`+fp d192 | **4/5** | 2625 | 0.1267 [0.0026, 0.4966] | +1.8144 | 293 |
| `exit` d192 | 3/5 | 3167 | 0.3057 [0.0338, 1.0424] | +1.6108 | 290 |
| `exit`+fp d96 | 2/5 | 2625 | 0.6556 [0.0040, 1.1185] | +1.2182 | 156 |
| `staged` d192 | 1/5 | 2500 | 0.8294 [0.0822, 1.1185] | +1.2781 | 477 |
| `exit` d96 | 0/5 | — | 0.9819 [0.6268, 1.1146] | +0.9440 | 154 |
| `staged` d96 | 0/5 | — | 0.9196 [0.6431, 1.1284] | +0.9643 | 345 |

- **Width is worth more than any attachment**: d192 solves 8 of 15 cells, d96 solves 2 of
  15. Iteration 1 could not read this because every cell escaped. This is the first width
  effect in the study, and it lands where the literature put it — the six-hop carry is what
  needs the dimensions.
- The fixed-point term beats plain `exit` at both widths (2/5 against 0/5, 4/5 against 3/5),
  a third independent replication of the 2026-09-10 result.
- `staged` is the worst attachment here, at 1 of 10 against `exit`'s 3 of 10, and it costs
  2.2x the wall clock. Its own-span target is the span's own elimination, and its states
  say so: at the middle slot it puts per-symbol mass 0.333 on the dead candidates and
  **0.000** on the survivor, entropy 0.000.
- Every stuck seed is depth-starved rather than committed: value CE falls from forced depth
  1 to 6 by a median of **0.986** nats, on 20 of 20.

## A partial superposition, where iteration 1 found none

Read per SYMBOL (each class mass divided by the number of symbols it covers), at the middle
slot after 6 passes, where the alive set is 3 of 6:

| cell | survivor | alive non-survivor | dead | entropy (ln 3 = 1.099 is a perfect 3-way spread) |
|---|---|---|---|---|
| `exit` d96 stuck | 0.310 | 0.308 | 0.025 | 0.783 |
| `exit`+fp d96 | 0.245 | 0.249 | 0.085 | 0.674 |
| `exit` d192 solved | 0.253 | 0.249 | 0.083 | 0.202 |
| `staged` d96 | **0.000** | **0.000** | **0.333** | 0.000 |

On `exit` and `exit`+fp the alive symbols share the mass evenly while the dead ones are
suppressed by a factor of 3 to 12 — the superposition signature that `eliminate` failed in
20 of 20 cells. Two things changed: the chain is six hops instead of two, and the classes
are no longer single symbols, so the masses must be read per symbol. Iteration 1's finding
should be read as "the tied head's mass sat on the slot's own elimination token", which is
what `staged` still does here by construction.

The `candidate_mass` instrument is uninformative at the ANSWER slot on this task, for the
reason iteration 1 already named: restricting the softmax to the instance's six candidates,
five of which the state can prove dead, hands it the sixth. It reads 0.996+ on the survivor
on stuck seeds whose value CE is 1.1.

## Scorecard

| id | claim | outcome |
|---|---|---|
| P1 | a basin returns; `exit` <= 2/5 at d96; fp strictly better; `staged` >= 3/5; the axis separates | **MOSTLY TRUE**: 0/5, fp better at both widths, the axis separates 0/5 against 4/5. The `staged` clause is **FALSE** (0/5 and 1/5) |
| P2 | the median stuck seed lies in [1.0986, 1.7918] | **FALSE**, narrowly: median 1.0130, with 8 of 20 within 0.02 of the ln 3 floor. The second clause (some seed below 0.6931) is TRUE |
| P3 | stuck seeds are depth-starved, CE falls >= 0.10 from depth 1 to 6 | **TRUE**, 20 of 20, median 0.986 |
| P4 | **the hop ladder decays at random init and training lifts it** | **TRUE**: monotone decay to 0.668 at hop 4 (d96), >= 0.95 at hops 0-1, `in_set` 0.566 at hop 6, and the trained cells sit above random init by 0.23 to 0.33 at every hop >= 3. One clause **FALSE**: `dead` at hop 6 is 0.682, not below 0.65 |
| P5 | the exclusion encoding replicates; superposition unlikely | **REFRAMED**: written in iteration 1's three-symbol framing, where the classes are single symbols. Read per symbol, `exit` and `exit`+fp DO spread over the alive set and suppress the dead; `staged` collapses onto the dead exactly as predicted |
| P6 | the fixed-point term raises the readout entropy at BOTH widths | **FALSE**: +0.259 at d96, −0.108 at d192 |
| P7 | the random-init reader: >= 1.79 at depth 1; below 1.88 from depth 4; above 0.8 at depth 6; depth 6 within 0.25 of depth 4 | three of four **TRUE** (2.087, 2.013, +0.013). The "below 1.88 from depth 4" clause is **FALSE**: 2.026, still above chance |
| P8 | the fitted reader beats the coda by >= 0.15 on stuck seeds | **FALSE**: median +0.030, above 0.15 on 1 of 20 |
| P9 | twin drop None / 2 / 4 by geometry | **TRUE**: None 30/30, 2 30/30, 4 on 28/30 (two seeds read 5) |
| P10 | d192 escapes on at least two more cells than d96 | **TRUE**, and by six: 8 of 15 against 2 of 15 |

## Verdict, against the decision rule filed in advance

The rule was: if the random-init hop ladder is flat and high and the random-init reader
reaches below 0.5 nats at depth 6, the carry is free at six hops and the loop's problem is
the readout; if the ladder decays and training lifts it, carrying across depth is learned.

**The ladder decays and training lifts it.** 0.668 against 0.999 at hop 4. The reader reads
2.002 at depth 8 on an untrained loop, 0.08 nats better than chance. Both clauses of the
second branch hold, and the first branch is refuted on both of its terms.

So iteration 1's conclusion needs its qualifier, and the statement that survives is:

1. A shared-core slot loop carries a fact for one or two hops for free, in an untrained
   network. Beyond that the carry decays, and closing the gap is one of the things training
   buys.
2. On a six-hop chain the thing that fails is the carry of the OLDEST facts, not the
   readout: the fitted reader recovers +0.030 nats over the coda on stuck seeds, while the
   `in_set` probe shows every stuck cell pinned at the ceiling reachable from the recent
   evidence alone.
3. Width buys hops. 96 dimensions solve 2 of 15 cells and 192 solve 8 of 15, with the
   `in_set` probe at hop 6 separating the two (0.888 against 0.952 on the best arm).

## What this says for the real model

The sibling hop-distance probe on real checkpoints asks whether MORPH's slot loop earns
depth only where the content is hops away. This toy says what a healthy answer looks like:
the carry of a fact h hops back should be measurable, it should decay with h in an
untrained or under-trained model, and the arms that earn depth should be the ones whose
decay is flattest at the largest h. It also says which instrument to trust. A tied-head
readout misreads the state twice over — once because the head exposes the local token
(iteration 1) and once because restricting it to the right candidates leaks the answer
(here) — while the fitted reader, on this task, adds almost nothing to the coda. The
informative instrument in both iterations was the per-fact probe scored against the
reachable window.

## Caveats, named

- Nothing here is about a language model. 22 ids, 8 spans, 4,000 steps, one answer position
  per row.
- The hop ladder's decay at random init is measured on ONE architecture at two widths. It
  is a property of a 1-block shared core with an entry injection, not a theorem.
- `staged` is confounded with wall clock: it is 2.2x slower per step, so at a fixed 4,000
  steps it has seen the same data but paid more compute.
- The two `answer`-slot twin cells that read a drop pass of 5 instead of 4 were not chased
  down; the cosine at pass 4 in those seeds sits just inside the 1e-3 tolerance.
- The membership probe's fit order changed between iterations, so iteration 1's probe
  numbers are not bit-comparable with a re-run (they were saturated; no conclusion moves).
