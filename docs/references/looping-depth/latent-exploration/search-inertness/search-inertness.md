# Anatomy of a Sound Neural Reasoner (first-pass poisoning, search inertness): reading note

Read 2026-09-23, for the controls of the LXTUL-G build
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../../.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md)).
The brief asked for the instruments only; the architecture (CoLT) is summarised just
enough to read them.

## Citation (verified)

Aleksey Komissarov. *Anatomy of a Sound Neural Reasoner: One-Shot Amortization, First-Pass
Poisoning, and Search Inertness in Clue-Rich Completion*.
[arXiv 2607.19635v1](https://arxiv.org/abs/2607.19635), 2026-07-22. Neapolis University
Pafos. No venue. Code, frozen protocol and raw results:
<https://github.com/ad3002/colt/tree/r14-2026-07-15>. The paper states that the code,
protocol, analyses and manuscript were produced by Claude agents under the author's
direction, and lists what the human checked.

Read in full: all 24 pages, including the claim audit (Appendix A) and hyperparameters
(Appendix B).

## Local cache

Not cached into `ignore/papers/` in this pass (the brief allowed only the note files).
Fetched from `https://arxiv.org/pdf/2607.19635` on 2026-09-23: 457,473 bytes, 24 pages,
SHA256 `e7f13b495a3203c67a90677636e3b12c390aa1c913c790c29b83d357bdecd55b`. Suggested cache
name: `ignore/papers/2607.19635v1-sound-neural-reasoner-anatomy.pdf`.

## What it actually does

The system under test is a lattice solver for constraint problems (the Lattice Deduction
Transformer, LDT, and the author's extension CoLT). The state is a multi-hot set of
candidate values per cell. Each round runs a recurrent network once, deletes candidates
below a threshold, and, if nothing changed, commits one cell to one value. An exact
checker gates every emitted answer, so the system is sound: it answers correctly or
abstains. CoLT adds a learned branch-choice head, depth-first backtracking with value
exclusion, and a shared ban set of verified-wrong grids. Every search component can be
switched off at inference, which gives a {restart, dfs} x {random, MRV, learned} grid from
one checkpoint.

The question: does the search layer set accuracy? The answer, in clue-rich Sudoku: no.
The first forward pass decides everything.

## The instruments, one by one

These are the parts worth porting. Each is stated with its definition and its reading.

**I-1. One-shot commit rate (H1).** Run one forward pass on the clues-only state, apply the
elimination threshold, and count cells left with exactly one candidate (singleton rate) and
puzzles fully committed. Also the commit precision: the fraction of those committed values
that are correct.

    6x6 standard split     singleton rate 1.000, full-commit 1.000
    6x6 hard slice         0.998 and 0.956; 5.9 % of committed singletons wrong
    9x9, no augmentation   singleton 0.97-1.00, commit precision 0.584-0.589
    9x9, augmentation      singleton 0.94-0.96, commit precision 0.998-0.999
    from-scratch coloring  singleton 0.014 (c = 3.0)

Reading: when one pass commits everything, the loop is cosmetic at test time. On
from-scratch coloring the rate is near zero and the loop genuinely iterates.

**I-2. First-pass poisoning and the contingency table.** A puzzle is "poisoned" when its
first pass deletes a value the true solution needs. Monotone search can never restore it,
so poisoning is sufficient for failure. The empirical question is whether it is also
necessary:

                          solved   failed
    first pass clean        137       0
    first pass poisoned       0      43

The zero in the clean-failed cell is the finding. It replicates on two more seeds
(49/49 vs 0/131, 38/38 vs 0/142) and on other hardware (42/42 vs 0/138). No failure
happens late; every failure is decided in pass 1.

**I-3. Threshold invariance.** Lowering the elimination threshold from 0.10 to 0.02
changes nothing (43 poisoned at every value). The wrong eliminations are confident
(`sigmoid(b) < 0.02`), not borderline. Reading: the failure is miscalibration, not a
decision rule.

**I-4. Flat ablation grid plus budget sweep.** The six search arms solve the identical
puzzle set (exact set equality, not equal percentages). On the hard slice every arm
scores 76.1 %, and a round-budget sweep (5, 15, 60 rounds) leaves it at 76.1 %. The
population bifurcates: a puzzle is solved within 5 rounds or never. What search DOES change
is waste: backtracking plus the ban set cuts repeated wrong derivations from 74,864 to 50
(1,497x) at identical accuracy.

**I-5. Symmetry-frame union, with an identity-frame null (H2).** Run each pass in K random
digit-permutation frames (permute values, forward, invert) and keep any candidate any
frame keeps.

    first-pass poisoning   K=1 23.9 %   K=4 union 5.0 %    K=8 union 1.7 %
    accuracy               K=1 76.1 %   K=4 union 100 %    K=8 union 100 %

Controls: mean aggregation at K = 8 reaches 97.8 % (the union is the active part). Identity
frames (the same pass repeated K times) leave all 43 poisonings. Geometry-only frames
reduce them to 27; digit frames to 3. The wrapper does nothing without a symmetry inside
it.

**I-6. The one-axis crossing table.** At the first density where the solver fails, move one
resource at a time (graph 3-coloring, c = 4.0):

    baseline (d=64, 4k steps, 12 sampled solutions)   0.0 / 0.0   (restart / dfs)
    12x inference budget                               1.9 / 0.0
    ~10x training compute                              9.5 / 8.6
    4x denser solution sampling (48)                  15.2 / 14.3
    no learning, exact search only                     0.0 / 0.0

Search-side resources do almost nothing; supervision coverage moves it most.

**I-7. Base-rate control for a transfer claim.** On zero-shot 9x9 states, one pass
eliminates with precision 0.977 at recall 0.597. Plain constraint propagation from the
decided cells already accounts for 98.3 % of those eliminations; restricted to candidates
propagation cannot remove, precision falls to 0.33 at recall 0.45. Reading: always score
the learned component against the trivial mechanism that produces most of the signal.

**I-8. Why one-shot is the optimum.** For a unique-solution puzzle the depth-0 target is
the full solution, so a head that fits its target perfectly is a one-shot solver by
construction. Proposition 1: for a value-symmetric problem solved from scratch the exact
depth-0 target is all-ones and carries no elimination signal; with K sampled solutions,
every missed (cell, value) pair becomes a false elimination label. Multi-solution targets
change the optimum; that is the regime (from-scratch coloring) where DFS beats restart
(+2.9 pp pooled over three seeds, up to +11.4 pp single-seed) and the learned branch
policy still never beats the hand-coded MRV heuristic.

The frozen-protocol practice is worth noting too: success criteria were committed before
training, failed criteria are reported as failures, equalities are reported as exact set
equalities, and every table regenerates from committed JSON.

## What LXTUL-G should take

The paper's central claim maps onto MORPH's central problem almost word for word: "one
pass does all the work" is the slot loop's measured state ("the strict slot loop's passes
2-6 do essentially nothing",
[`lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md`](../../../../../lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md)).
The instruments give a way to tell whether LXTUL-G changes that or only adds noise around
it.

- **I-1 as the per-pass commit rate.** Decode the coda from each pass's slot state (the
  slot loop is trained at sampled depths, so every pass is decoder-ready). Per slot,
  report whether the pass-1 decode already has the final decode's top-1 first token, and
  the fraction of the final pass's CE gain over depth 0 already present at pass 1. Report
  it on PRIOR samples, split by sample.
- **I-2 as the first-pass contingency.** Define "pass-1 poisoned" for a slot as: the
  pass-1 decode ranks the true first token of the span below top-k (k = 10) with high
  confidence, or the pass-1 span CE is in the worst quintile. Define "failed" as the final
  span CE in the worst quintile. Tabulate. A near-empty clean-failed cell and a near-empty
  poisoned-recovered cell means the loop commits on pass 1. Recoveries (poisoned at pass
  1, fine at pass T) are the direct evidence that later passes search.
- **I-4 as a depth x width factorial on prior samples.** Depths {1, 2, 3, 6, 9, 16} crossed
  with N = {1, 4, 16} samples. Flat along depth at every N means the passes are inert;
  gain along N but not depth means parallel guessing, not iterative search.
- **I-5's identity-frame null as the near-copy control.** N copies of the SAME prior sample
  (identical noise) must give zero gain from any aggregation. MORPH already measured the
  analogous floor: the oracle over four near-copies sits about 0.04 nats below the
  deployed read (`lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md`). Every
  N-sample reading needs this floor beside it.
- **I-6's discipline.** Move one axis at a time, and include the no-learning control.
- **I-7's base-rate control.** Score the loop's per-pass gain against the depth-1-trained
  twin, the trivial mechanism that produces most of MORPH's signal.
- **I-8's lesson for the objective.** A target with one reachable conditional mean has a
  one-pass optimum. LXTUL-G's prior must cover a SPREAD of posterior steps, which makes
  the prior's target multi-modal; that is the change of optimum the paper says is needed
  for iteration to matter.

## What does not transfer

- **Monotone deletion.** Poisoning is irreversible only because the lattice only removes
  candidates. A slot state can move back, so "poisoned at pass 1" in LXTUL-G is a
  prediction, not a lemma, and recoveries are possible and measurable.
- **Exact verifier and abstention.** There is no checker for a text span and no abstain
  option. Accuracy becomes CE.
- **The symmetry cure.** Digit relabeling is an exact symmetry of Sudoku. Text has no
  exact value symmetry, so the H2 cure has no direct analogue; only its identity-frame
  null does.
- **Scale and regime.** 64-dim, 4-layer models on 4x4 to 9x9 grids with minutes to hours of
  training. Most boundary cells are single-seed.
- **The GRAM row.** GRAM appears only as a budget-starved soundness baseline; its 98.9 to
  100 % wrong-emission rate says nothing about GRAM's ceiling, as the paper states.
