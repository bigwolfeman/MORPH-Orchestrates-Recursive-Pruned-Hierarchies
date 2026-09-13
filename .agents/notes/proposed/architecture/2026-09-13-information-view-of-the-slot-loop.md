# Agent Note: An information view of the slot loop

Status: proposed

## Problem

The slot loop does not earn depth. Forty-five arms say so. The arc ledger
([`lab/experiments/planned/2026-09-04-loop-contribution-arc.md`](../../../../lab/experiments/planned/2026-09-04-loop-contribution-arc.md),
results table, lanes 3, 4 and 5) records token K1−K6 between −0.0001 and +0.0070 on every
slot arm whose coda reads all cells, and between −0.0002 and +0.0019 on the twelve
strict-geometry arms. Their K3−K6 sits between −0.0008 and +0.0003. That holds whatever the
target (M-next, span decoder, staged, oracle trajectory, gradient-conditioned pass, per-pass
horizon, critic, two energies), whatever the core (MORPH ternary at three scale rules, a
dense Parcae block stack), and whatever the stability term.

The same tree also shows a loop that does earn. The plain model, every token position through
the core, reads K1−K6 **0.1849** on web text under `norm_match`
([`successes/2026-09-09-arc-per-pass-strength.md`](../../../../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md))
and **0.064** on Olympiad math, 0.865 at depth 1 down to 0.801 at depth 6 (runner
`queue.log` 05:11 on 2026-09-13, filing pending). So the corpus is not the reason. The
question is what separates the two loops.

Twelve lanes of levers are closed. None of them named a mechanism that predicts both
readings at once. This note proposes one, and proves the part of it that is a theorem.

## Proposal

### The claim in one paragraph

A deterministic loop cannot add information. Both MORPH loops are deterministic maps of
their own entry state, so by the data-processing inequality the state after any number of
passes carries at most what the entry carried about any target. Loop value can therefore come
from exactly one place: the READER'S OBSERVATION. The state does not gain, but what the coda
gets to look at can. That happens two ways. A RELAY moves content into a narrow observation
window that could not see it before. EXTRACTABILITY re-arranges content that the reader can
already see into a form its fixed, bounded computation can use. The plain loop has a large
extractability gap and the slot loop has almost none, because the coda is already flat around
the slot state. The slot loop has no relay either, because the coda reads every cell
directly. Cut the coda's reach and a relay appears, on cue, and only then.

### The forward pass, as the code actually runs it

Read from `morph/model/transformer.py::_tul_core` and `morph/model/tul.py` on 2026-09-13.

```
h_0[k]     = core_init(input_norm(prelude(x))[slot_index[k]])
h_{t+1}[k] = f_theta({h_t[j] : k-w <= j <= k}, e[k], inj[k], ret_state_t, t)   if depth[k] > t
           = h_t[k]                                                            otherwise
z[k]       = h_T[k]
row[slot_index[k] + m] = z[k] W_m,   m = 0 .. prefix_k-1
```

Four facts about that code are load bearing and two of them corrected my own brief.

1. **No token position is read at any pass.** `e[k]` and `inj[k]` are computed once from the
   prelude output at cell `k` and re-fed unchanged at every pass
   (`transformer.py:3400-3408`). The loop runs on a compact sequence of `max_slots` cells,
   one per slot.
2. **Cells do read each other, at the same pass.** At `loop_reach 0` cell `k` attends cells
   `0..k`, and K/V are recomputed from the live carrier every pass
   (`transformer.py:3216-3219`). So the pass map is a map of the WHOLE cell array, not of one
   cell. The iteration to reason about is `S_{t+1} = f(S_t)` on the array.
3. **`tg_restrict`, `tg_geometry` and `tg_coda_prefix_reach` do not touch the loop.**
   `_tul_core` is never handed `tg_attn_kwargs` (`transformer.py:5519-5521`). They are
   relations on the long row and act in the prelude and the coda only. `tul.loop_reach` is
   the one key that changes the in-loop query-key relation, and only under
   `tg_geometry: strict` (`transformer.py:3534-3551`).
4. **Nothing is detached.** `bptt_depth 8` at `max_depth 8` gives `n_nograd = 0`
   (`transformer.py:3434`), so every pass carries gradient.

Because of 1 and 2, the whole array after pass `t` is a deterministic function of one object,
the loop's entry data `E = (h_0, e, inj, ret_state_0, depths)`. The passes are iterations of
one map on `E`.

### The theorems

Lean 4, Mathlib v4.31.0, in
[`lab/theory/tul_information/`](../../../../lab/theory/tul_information/). `lake build` exits
0 with no warnings, and `Axioms.lean` prints thirteen lines all reading
`[propext, Classical.choice, Quot.sound]`. There is no `sorry` in the source. The model is a
joint probability mass function on two finite alphabets; `I` is mutual information in nats.

| name | where | what it says here |
|---|---|---|
| `Joint.mi_map_le` | `TulInformation/Basic.lean:233` | `I(f(Z);Y) <= I(Z;Y)`. One pass adds nothing. |
| `Joint.condEntropy_map_ge` | `Basic.lean:363` | the same in log-loss form. |
| `Joint.mi_iterate_le` | `SlotLoop.lean:56` | `I(f^n(Z);Y) <= I(Z;Y)`. No number of passes adds anything. |
| `Joint.mi_traj` | `SlotLoop.lean:80` | `I((Z, fZ, ..., f^n Z); Y) = I(Z;Y)`. The trajectory carries exactly the entry. |
| `Joint.mi_factor_le` | `SlotLoop.lean:102` | a read-out that factors through a projection carries at most what the projection carries. This is the reach bound. |
| `Joint.mi_relay_le` | `SlotLoop.lean:115` | a pass carries at most what its inputs carry. |
| `Joint.mi_relay_redundant` | `SlotLoop.lean:128` | `I((F(Z,N), Z, N); Y) = I((Z,N); Y)`. A reader that already sees the pass's inputs gains exactly zero from the relayed state. |

`Joint.mi_nonneg`, `Joint.map_id`, `Joint.map_comp` and Gibbs' inequality are in the same
build. The README lists what is assumed and not proved: finite alphabets, conditional entropy
DEFINED as `H(Y) - I`, and the independence of the depth draw from the target.

### Correction to the brief I was given

The brief said a trajectory arm can add no information over the EXIT state. That is wrong.
`mi_traj` (`SlotLoop.lean:80`) says the trajectory equals the ENTRY, and `mi_exit_le_traj`
(`SlotLoop.lean:90`) says the exit is at most the trajectory. A map can throw information
away, so the trajectory can carry strictly more than the exit. The built control for the
trajectory arm repeats the exit (`prefix_source: exit_repeat`), which is the right control
for the question "six different states or six cells", and the wrong one for the question
"information the exit lost or a reader that finds six copies easier". Separating those needs
one more arm that writes the ENTRY and the exit. That is prediction P1c below; it changes no
other criterion.

### Why the plain loop earns and the slot loop does not

Both loops are information-flat. The difference is entirely in the reader's observation.

**The plain loop.** The state is 1,024 token positions. The coda is three blocks and a tied
head. At pass 1 a position has seen nine blocks of computation; at pass 6 it has seen
thirty-nine. The reader's suboptimality at pass 1 is the whole prize, and it is large:
0.1849 nats. The measured shape agrees with an extractability reading and not with a relay
reading. Cutting every cross-span route changes the plain arm's K1−K6 barely at all, 0.0279
with the full context against 0.0301 with nothing crossing a boundary
([`failures/2026-09-11-arc-span-budget.md`](../../../../lab/experiments/failures/2026-09-11-arc-span-budget.md)),
so the depth the plain loop uses is span-local computation, not long-range carriage. The
per-pass map's strength moves it directly: absmean 0.033, `norm_match` 0.1849, bf16 core
0.168 (lane 2). A stronger map per pass does more computation per pass. That is what
extractability predicts and what a relay account cannot explain.

**The slot loop, relay side.** The coda reads every slot's cells directly under
`tg_coda_prefix_reach: all`. So the reader's observation is the whole cell array whether or
not cell `k` also holds cell `m`'s content. `mi_relay_redundant` is that statement: moving
content among coordinates the reader already sees changes the mutual information by exactly
zero. The prediction is that no reach change can force depth while the coda reads all cells,
and the strict panel tested exactly this without meaning to.
`slot-spandec-strict-reach1` cuts the loop's reach to one cell per pass and, with the coda
still reading all cells, reads K1−K6 **+0.0014**. Cut the coda to the previous slot only and
add the same reach limit, and the first forced K-curve in the whole arc appears: **+0.0163**,
K3−K6 +0.0042
([`failures/2026-09-12-arc-strict-geometry.md`](../../../../lab/experiments/failures/2026-09-12-arc-strict-geometry.md)).
Widen the hop to two cells and the curve SHRINKS to +0.0127 with better CE, which is the
reach bound again: the curve counts how many passes the window needs, not how much value the
depth makes.

**The slot loop, extractability side.** There is almost no reader suboptimality to harvest.
Three measurements, none of them a proxy.

* The coda's loss moves at most 0.0005 nats under a 10 % rms perturbation of the exit state,
  and one pass moves it 0.001 to 0.003 nats
  ([`planned/2026-09-12-arc-core-token-and-critic.md`](../../../../lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md),
  arm D). A critic asked to tell pass `t` from pass `t-1` through the real coda sat at 0.521
  agreement for 5,000 steps. It is a tie by measurement.
* A state fitted by gradient descent from history alone is 0.252 nats WORSE than the loop's
  one-pass state (`causal_fit_slot-spandec-strict_g1`, Outcome section of
  [`.agents/notes/proposed/testing/2026-09-12-causal-instruments-for-the-slot-loop.md`](../testing/2026-09-12-causal-instruments-for-the-slot-loop.md)).
  The 1.479-nat "headroom" that the earlier fit reported used the answer. On the causal
  estimator there is no headroom above one pass at all.
* The linear signal about the next span is already in the entry state, AUC 0.605 to 0.642
  against a null p95 of 0.513, and the only rise is at pass 1
  ([`planned/2026-09-12-arc-latent-z-gradient.md`](../../../../lab/experiments/planned/2026-09-12-arc-latent-z-gradient.md),
  step 0).

The account makes one more prediction that is already scored. If the passes add nothing, a
slot model trained at depth 1 should tie the depth-6 model. It does: `strict@6 − norecur@1`
is **+0.0020** [−0.0006, +0.0044], interval crossing zero, at 0.78x the wall clock (same
file). And the twelve strict arms reach that tie from twelve different objectives, which is
what a ceiling looks like and not what a tuning problem looks like.

### Where the loop's value would have to come from

The channel is small and mostly already spent. The whole cross-span budget on web text is
**0.3994** nats [0.3838, 0.4162] at 5,000 steps, 0.3149 of it flat at offsets 8 and beyond
(span-budget file). Under strict geometry the slot channel carries **0.1865** of that, and on
a strict arm the loop's write is the entire channel because every other route is cut. So the
headroom for any better write is at most 0.214 nats, and the loop's own share of whatever a
better write buys is bounded by the reader's per-pass headroom, which measures 0.001 to
0.003 nats. Against that, matched-compute plain is 0.2536 nats ahead of `slot-spandec-mask`
at the same block-passes per token
([`successes/2026-09-11-arc-span-decoder.md`](../../../../lab/experiments/successes/2026-09-11-arc-span-decoder.md)).

### What this predicts for the three builds under construction

Stated as inequalities in Acceptance criteria below.

## Alternatives considered

Other explanations for the flat curve, and why the ledger rules them in or out.

**The map is unstable, so the passes never settle into useful work.** Ruled out. Under the
gain constraint the map holds typical gain 0.887 to 0.897 for 5,000 steps with zero spikes
and still reads K3−K6 0.000
([`successes/2026-09-04-tul-forward-levers.md`](../../../../lab/experiments/successes/2026-09-04-tul-forward-levers.md)).
The gain dial at 0.95 and 0.98 is a stability dial, not an earning dial (E1). Turning the
fixed-point term off makes the state expand 2.2x and moves K1−K6 by +0.0005 (lane 3).

**The per-pass map is too weak.** Ruled out for the slot loop, and it was TRUE for the plain
loop. The `norm_match` rule took plain K1−K6 from 0.026 to 0.1849 and left both slot arms at
0.0000 on the same day
([`failures/2026-09-09-arc-slot-loop-norm-match.md`](../../../../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md)).
The rule reaches the core the tokens run through and does not reach the slot map, because the
slot map's limit is not its strength.

**The core is the wrong shape, or a branch is dead.** Ruled out. The HCA compress fix, the
width sweep at `prefix_k` 2, 4 and 8, a dense Parcae block stack as the core, and a rank-32
per-pass LoRA all land at K1−K6 <= 0.0070 (lane 3). A core trained to a competent per-token
map by an auxiliary objective, the same six blocks at the same depth over the same cells,
adds 0.0005 (`coretok`, lane 5).

**The target is wrong.** Ruled out, after ten targets. The decisive instrument is the
corrected identical-target grid: held to ONE target, forced depth moves the next-span column
by **+0.0000** and the six-span column by **+0.0015**
([`planned/2026-09-12-arc-objective-arms.md`](../../../../lab/experiments/planned/2026-09-12-arc-objective-arms.md)).
The rising per-pass ladder that looked like evidence was the changing target.

**The geometry bypasses the loop.** Ruled out as a cause, though it was a real bypass.
Closing it costs exactly 0.0000 nats and leaves K1−K6 at 0.0016 (strict panel). The channel
was never the bottleneck.

**Training is too short.** Ruled out. Contribution is flat at 0.036 to 0.042 from 2,500 to
20,000 steps while CE falls 4.45 to 3.45
([`failures/2026-09-09-arc-horizon-ternary-25k.md`](../../../../lab/experiments/failures/2026-09-09-arc-horizon-ternary-25k.md)).
The staged arm's gap to plain is flat at 0.16 from 10k to 20k.

**Gradient credit assignment: the passes cancel, so they never learn to differ.** Kept, as a
complement and not a rival. The per-pass cotangents cancel at 0.52 to 0.60 and the MUX pays
7.3x the token CE
(`lab/experiments/results/2026-09-10-slot-gradient-probe/README.md`). This explains why an
available extractability gain would not be harvested. It does not explain why the plain loop,
trained by the same optimiser on the same graph, harvests 0.1849. The information account
supplies the missing half: on the slot channel there is very little to harvest, so the
cancellation has nothing to spoil.

**Web text does not need depth, or one attention hop is enough.** Not offered. Fact (c)
refutes both: the same corpus, the same tokenizer, the same core, and the plain loop reads
0.1849. Wolfe has corrected this three times and the ledger agrees with him.

## Acceptance criteria

Falsifiable predictions for the three builds now under construction. Every number below is a
number a 5,000-step arm on the strict recipe can produce, scored token-paired on the same 480
validation rows the arc uses.

**Build 1, trajectory as prefix** (`morph/configs/tul_slot_spandec_strict_traj.yaml`,
`prefix_k` 6, `prefix_source: trajectory`; control `..._trajrep.yaml`, the same shape with
`prefix_source: exit_repeat`).

* P1a. Against the built control, `CE(traj) − CE(trajrep)` >= **−0.020** nats. The
  trajectory carries the entry and no more (`mi_traj`), and the entry is measured 0.0185
  [0.0140, 0.0216] WORSE than the exit as a reader input (causal-fit table). The gap between
  a state that lost nothing and the exit the control repeats is the whole prize, and it is
  that small.
* P1b. Its token K1−K6, read against `trajrep` and not against `strict`, stays <= **0.005**
  with the coda at reach all. A trajectory write is not a relay: the reader sees pass 1 and
  pass T either way, so `mi_relay_redundant` applies.
* P1c. The pair as built cannot separate the two mechanisms, and this is worth one more
  config. `trajrep` repeats the EXIT, so a win could be information the exit threw away or
  it could be a reader that finds six copies easier to read. The control that separates them
  writes the ENTRY in cell 0 and the exit in the last cell, at the same `prefix_k` 6 and the
  same `E_pass` table. Predict `CE(traj) − CE(entry+exit)` >= **−0.005** nats. If the
  trajectory beats THAT control by more than 0.005 with a paired CI clear of zero, the
  intermediate states carry something neither end holds, and the extractability reading of
  the slot channel is incomplete.

**Build 2, the loop reads the tokens**
(`morph/configs/tul_slot_spandec_strict_tokloop.yaml`, `tul.loop_reads_tokens`, the whole row
through `_core_region` under `causal AND (same bag OR j is a slot cell)`).

* P2a. Its TOKEN K1−K6 is >= **0.5x** the matched plain control's on the SAME recipe and the
  same loop entry. The anchors, and the entry matters more than anything else here: a plain
  model reads 0.1849 under the noise entry and 0.033 under the prelude entry
  ([`failures/2026-09-09-arc-slot-loop-norm-match.md`](../../../../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md)),
  so the bar is 0.09 or 0.017 depending on which entry the arm runs. Do not score it against
  the absolute 0.1849. The reason to expect most of the plain arm's curve to survive the
  span restriction: cutting every cross-span route left the plain arm's own K1−K6 at 0.0301
  against 0.0279 with the full context (span-budget file), so the depth the plain loop uses
  is span-local computation.
* P2b. Its SLOT cells still read K3−K6 <= **0.002** while the coda keeps reach all. Tokens
  gaining is not cells gaining, and nothing in this arm narrows the reader's window.
* P2c. Its `spandec_ce` K1−K6 stays <= **0.005**, the band every strict arm sits in.
* Refuted if P2a lands under 0.2x the matched plain control. That would mean the plain loop's
  earning is not span-local computation, and the extractability reading of fact (c) is wrong.

**Build 3, TUL as memory.**

* P3a. Any memory arm's CE gain over a control with no cross-span route at all is <=
  **0.3994** nats [0.3838, 0.4162]. That is the whole budget and it is measured.
* P3b. The strict channel already carries 0.1865, so any write improvement is worth at most
  **0.214** nats. A memory arm that beats `slot-spandec-strict` by more than that has found a
  route the budget experiment did not cut, and the first move is to re-run the leak gate.
* P3c. The LOOP's share of any such gain is <= **0.018** nats at depth 6, six passes times
  the measured 0.003-nat per-pass reader headroom, while the coda reads all cells. Cut the
  coda to `prev` and this bound does not apply; the relay does, and it is worth +0.0163 with
  reach 1 at a CE cost of +0.0104 against strict.
* Refuted if a memory arm with coda reach all shows K3−K6 above 0.01 with a paired CI clear
  of zero.

**The observation that refutes the whole account.** A slot arm with coda reach all, no
per-pass token read, and no per-pass randomness that reads token K3−K6 above **0.01** with a
paired bootstrap CI clear of zero. Under the theorems that is impossible unless a new input
enters each pass, so the first response is to run the strict leak gate
(`tests/test_tul_strict_geometry.py`) rather than to believe the number. If the gate passes
and the number stands, the bridge from the theorems to the measurement is broken, and the
place to look is the assumption list in the project README.

## Risks

**The coda is not a fixed bounded reader.** It is trained jointly with the loop. So "reader
headroom" is measured at the convergence of one arm and may be a symptom rather than a cause:
a coda trained against a useless write learns to ignore it, and then reads flat. The
perturbation flatness (0.0005 nats at 10 % rms) cannot tell those two apart. The instrument
that could is a coda trained from scratch on a FROZEN loop of depth 1 against one of depth 6,
which no arm has run.

**Extractability gains can be large.** The plain loop's 0.1849 nats proves it. Nothing here
says a loop cannot work. The account says the slot channel's gain is small because its
reader's headroom is small, and that is an empirical claim about this channel at this scale,
not a theorem.

**The relay costs CE.** `prev-reach1` is +0.0104 nats worse than `strict` at depth 6 and
`prev` is +0.0082 worse. Forcing the reader to be narrow buys a K-curve and pays for it.
"Make the coda blind so the loop matters" is not a recipe, and the ledger already says the
reach arms make `z` carry history rather than the next thought, which was Wolfe's objection
in amendment 3 of the strict panel.

**The bridge from mutual information to cross-entropy is assumed.** The Lean development
defines conditional entropy as `H(Y) − I` and does not prove it equals the Bayes log loss.
Every arm measures the CE of a trained reader, which is above the Bayes log loss by an
unmeasured amount. The theorems bound the information; the numbers bound a reader.

**The causal fitted z is one estimator on six rows.** It does not bound the true causal
optimum. A better teacher, more samples, or an amortised fit could find headroom that this
one missed, and then the extractability side of the account has a target again.

**Randomness is an input.** The Poisson depth draw and the 0.15 token-state dropout are not
functions of the entry. `mi_relay_le` covers them (a pass carries at most what its inputs
carry), but the stronger claim, that randomness independent of the target carries nothing, is
assumed and not proved.

**One seed, 5,000 steps, one scale.** Every number cited is from a 5,000-step arm at seq
1024, batch 6, seed 1, except the 20k rows named as such. Short-horizon CE does not rank
arms, and this note does not use it to; it uses K-curves, worth profiles and paired gaps,
which are within-run measurements.
