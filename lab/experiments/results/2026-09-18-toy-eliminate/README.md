# `eliminate`: the toy slot loop carries the set, and reads out the exclusions

Date: 2026-09-18. 32 runs, 0 failures, RTX 3070 (`ssh 3070`), 55 minutes wall.
Pre-registration: [`../../planned/…` moved to
`../../failures/2026-09-18-toy-eliminate-deferred-commitment.md`](../../failures/2026-09-18-toy-eliminate-deferred-commitment.md).
Code: [`lab/toy_slot_loop/`](../../../toy_slot_loop/), commit `2f66d40`.

- [`tables.md`](tables.md) — every aggregated table, `python aggregate.py <dir>` output.
- Raw JSON, the grid log and the runner:
  `ignored/experiment-artifacts/2026-09-18-toy-eliminate/` (32 grid cells, 2 smoke cells,
  848 KB). Regenerate the tables with
  `python lab/toy_slot_loop/aggregate.py ignored/experiment-artifacts/2026-09-18-toy-eliminate`.

Per-pass tap self-check over all 32 runs: worst relative error **1.34e-07**.

## The four answers

**(a) Does it solve the task? Yes, every time.** 30 of 30 trained cells reach value CE
**0.0000** and accuracy **1.000**, at both widths and all three attachments. `exit` and
`exit`+fixed-point escape at step 250, `staged` at 350 (d96) and 500 (d192). The escape-rate
axis that carried the 2026-09-10 `compose` study is dead here: a 2-pass chain has no basin.

**(b) Is the alive set carried as a superposition? No. It is carried as the EXCLUSIONS.**
At the pruning slot (`elim1`, pass 1, where the state holds the candidate set and the first
elimination, alive set 2) the tied answer head's restricted softmax reads:

| cell | survivor | dead | alive non-survivor | entropy (max ln 2 = 0.693) | top mass |
|---|---|---|---|---|---|
| `exit` d96 | 0.277 | 0.446 | 0.277 | 0.256 | 0.894 |
| `exit` d192 | 0.101 | 0.809 | 0.090 | 0.105 | 0.956 |
| `exit`+fp d96 | 0.009 | 0.982 | 0.009 | 0.080 | 0.982 |
| `exit`+fp d192 | 0.099 | 0.805 | 0.096 | 0.424 | 0.838 |
| `staged` d96/d192 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |

The mass sits on the candidate the state can already prove DEAD, not spread over the two
alive ones, in 20 of 20 solved cells. That is the prediction the study was built to test
(P5) and it is falsified everywhere.

**The membership probe says the set is there anyway, and that carrying it is free.** The
linear probe is a step function exactly at the reachability boundary, in every cell:

| role | pass 0 | pass 1 | pass 2 | pass 6 |
|---|---|---|---|---|
| `cand` | 1.000 | 1.000 | 1.000 | 1.000 |
| `elim1` | **0.50** (the null: the candidate span is not yet reachable) | 1.000 | 1.000 | 1.000 |
| `answer` | **0.50** | **0.50** | 1.000 | 1.000 |

and the RANDOM-INIT cells read 0.998/0.950 (`elim1`) and 0.835–0.957 (`answer`, pass 2) on
the same probe. So the alive set becomes linearly decodable the moment an attention hop can
reach it, in an untrained network. **Carrying a set costs this architecture nothing and
needs no training. What training buys is the readout, not the carry.** That is the
2026-09-10 z-optimize result from the other side: there the reader was fine and the writer
was empty; here the writer carries everything and the head reads the local token.

**(c) Is commitment deferred? It tracks reachability exactly, which is not the same thing.**
At the answer slot the survivor's mass goes 0.25 → 0.97 → 1.000 over passes 0, 1, 2, and the
value CE at forced depth 1 is 1.45–1.48 against the enumerated depth-1 floor of 1.3863. The
state does not commit early because the information to commit is not there. The pass-1
reading of 0.97 is a leak of the instrument, not early knowledge: restricting the softmax to
the instance's own three candidates, two of which the state can prove dead, hands it the
third. `twin_divergence` confirms the geometry in 30 of 30 trained cells — the drop pass is
None at `cand`, 0 at `elim1`, 1 at `answer` — but the drop is shallow on solved seeds
(cosine 0.992–0.998 at (`answer`, pass 1)), against the prediction of below 0.90.

**(d) Does it depend on width? Not measurably, because the escape axis saturated.** d96 and
d192 both read 5 of 5 at every attachment. The only width signal is in the readout spread,
and it points the other way: the wider model is MORE peaked at the candidate slot (`exit`
entropy 0.26–0.35 at d192 against 0.35–0.46 at d96). n = 5, and no claim is made on it.

## The one lever that spreads the readout

The terminal fixed-point term is, again, the only knob that changes the state's character.
At the candidate slot (`cand`, alive set 3, maximum entropy ln 3 = 1.0986):

| cell | entropy p0 | p1 | p2 | p6 | top mass p6 |
|---|---|---|---|---|---|
| `exit`+fp d96 | 0.923 | 0.956 | 0.983 | **0.989** | 0.507 |
| `exit`+fp d192 | 0.855 | 0.772 | 0.796 | 0.799 | 0.630 |
| `exit` d96 | 0.446 | 0.395 | 0.455 | 0.352 | 0.858 |
| `exit` d192 | 0.305 | 0.258 | 0.276 | 0.349 | 0.856 |
| `staged` d96 | 0.243 | 0.337 | 0.448 | 0.548 | 0.771 |
| random init d96 | 0.097 | 0.135 | 0.165 | 0.096 | 0.962 |

Under the fixed-point term the candidate slot reads 0.989 of a possible 1.0986 with a top
mass of 0.507 — as close to a genuine 3-way superposition as anything measured in this tree.
Every mean mass in that table is 1/3 per candidate, including the random network's, so only
the entropy separates a superposition from a per-row commitment. That column is the reason
the instrument reports it.

The gradient tables repeat the 2026-09-10 sign: the fixed-point arm reads the lowest
agreement between passes (mean pairwise cosine +0.192 and cancellation 0.470 at d96, against
+0.636 and 0.805 for plain `exit`), and `staged` again develops a NEGATIVE cosine at the exit
pass (−0.38 d96, −0.15 d192).

## Two things that cost something

**`staged` destroys depth generalisation.** Value CE at forced depth 12: `staged` 11.490
(d96) and 3.513 (d192), against 0.000 for `exit` and `exit`+fp. Its own-span target is the
span's own elimination, so every non-final pass is trained to hold the dead candidate, and
past the trained depth the state has nothing else.

**The write is worth a lot here.** Value CE with z = the entry state (zero passes) is
1.72–3.47 nats against 0.000 with z = the exit, and 7.9–14.3 with z = 0. The real model's
entry-minus-exit gap is +0.0015.

## Instrument caveats, named

1. `twin_divergence` expects cosine 1.0 at role `cand` forever. That is only true INSIDE an
   instance. A row holds two instances and the core chain crosses their boundary, so slot 4
   reaches span 1 at pass 3. The two random-init cells read a drop at pass 4–5 (cosine
   0.9963/0.9971) for exactly that reason; the trained cells suppress it. The stated
   expectation in `instruments.py` is within-instance and should be read that way.
2. `candidate_mass` at role `answer` leaks after both eliminations are reachable, because
   the softmax is restricted to the instance's own three candidates. Its honest reading is
   at roles `cand` and `elim1`.
3. The membership probe cannot separate a trained loop from a random one on this task. It is
   a destruction test: a low accuracy is evidence, a high accuracy is not. The random-init
   cells are in the grid for that reason.
4. No stuck seed was produced, so the 0.9242 and 0.6931 point-carry plateaus were never
   exercised. They remain derived and enumerated, never observed.
