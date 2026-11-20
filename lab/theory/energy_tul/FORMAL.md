# Energy TUL: formal statement

Date: 2026-11-19. Status: theory draft. No run was made for this file.

Each claim has a tag: **[definition]**, **[proved here]**, **[proved, Lean]** (earlier work, file named),
**[measured]** (file named), **[conjecture]**.

Sources, by short name:

- NARROW, ANSWER, STATEMENT, START: the four hop filings in `/home/wolfe/clones/parcae/parcae-wt-hops/docs/experiments/`
  (`successes/2026-11-18-parcae-narrow-core-hops.md`, `failures/2026-11-18-parcae-answer-routed-loop-hops.md`,
  `failures/2026-11-19-parcae-statement-routed-loop-hops.md`, `failures/2026-11-19-parcae-statement-routed-loop-trained-start.md`).
- CHAIN: `lxtul/hops/chain.py` in that worktree. CLOCK: `planned/2026-11-19-parcae-depth-clock-hops.md` there.
- HOPS, WASHOUT: `lab/theory/tul_exploration/TulExploration/Hops.lean`, `Washout.lean`. DEPTHLAW:
  `lab/theory/parallel_loop/ParallelLoop/DepthLaw.lean`.
- LESSONS: `docs/lessons-learned-loops-and-spans.md`.

## 1. Definitions

All items in this section are **[definition]**.

- **Token stream.** `x_1 .. x_n` over a vocabulary. The main loss is next-token CE.
- **Readable set** `R`. The stored states that a read can return. It holds one state per consumed token and one
  state per finished thought.
- **Thought state** `z`. One vector at one position.
- **Think step.** `z <- f(z; R)`: one pass of the weight-tied core (`L` blocks) at one position. No token moves.
- **Emit step.** One pass of the token path. It produces the next token from `z`, `R` and the earlier tokens, and it
  adds that token's state to `R`. In training the token is given.
- **Thought.** A run of `T >= 1` think steps, then `N >= 0` emit steps. At its end `z` joins `R`. `T` and `N` are
  set separately.
- **Control.** At each step a policy picks THINK or EMIT from `(z, R)`.
- **Thought graph.** Thought `k` has an edge from thought `j` when a read of `k` returns `z_j`. This graph is the
  hierarchy. Nothing fixes its shape.
- **Cost.** The unit is one block at one position. `C = L_tok * (token passes at full width) + L * w * (think steps)`,
  where `w <= 1` is the price of a narrow core. The count leaves out the price of a read over `|R|` entries. That is
  a known gap: a loop on 1.2 % of tokens cost 27 % of the full loop's extra step time **[measured, ANSWER]**.
- **Objective.** `J = E[CE] + lambda * E[C]`.
- **Lookup model.** A fact is a pair `(key, value)`. One attention read at a position returns the stored content of
  one position that its query selects. A pass gives a position `L` reads in sequence. This is the adaptive lookup
  program of HOPS, with one addition: a stored state may hold a composite fact that spans `s` hops.

## 2. The six requirements as properties

| Requirement | Property of the object above |
|---|---|
| Halt | `T` is a function of the input, set by the control |
| Cheap decode | `N` emit steps cost `N` token passes, and each has a token path |
| Preserve | `z` joins `R` at the end of a thought; storage is one state per thought |
| Read | A think step reads the token states since the last thought and all earlier `z_j` |
| Payment | The loss changes when `z` is removed (need), and the control has a gradient or a target (signal) |
| Updated states | A read of another thought returns its newest state, not its pass-1 state |

Dependence between them:

1. **Preserve is Read plus a storage limit [proved here].** A thought with `N = 0` changes the loss only through
   later reads of its `z`. So legal `N = 0` thoughts make Preserve necessary.
2. **Updated states follows from Preserve and Read when thoughts run one after another [proved here, Proposition 2].**
   Every earlier thought is then finished, and its newest state is its final state. The requirement is separate
   only when thoughts run at the same time.
3. **Cheap decode and need pull against each other.** The token path is the bypass. By `passes_needed` (HOPS), the
   loss needs the thought only on tokens whose serial depth is above the depth of the token path **[proved, Lean]**.
   Need is a property of the data and the decoder depth together. No architecture gives it alone.
4. Halt is a capability. Signal is its training. Proposition 6 shows they cannot be designed apart.

## 3. Propositions

### Proposition 1. Walk against double

Setting: `F` facts, `n` tokens, a chain of `h` hops, `q` questions per document.

1. **Walk bound [proved here].** Let every stored fact state be static and span at most `s` hops. A thought that
   makes `r` reads knows the answer for at most `r * s` hops. Proof: induction on reads. After `r` reads the state
   is a function of table entries within `r * s` chain steps of the start. A table that differs only past that
   point gives the same state and a different answer. For `s = 1` this is `hop_lower_bound` (HOPS). So `T` passes
   reach at most `s * L * T` hops. A readable trail of the thought's own states does not change the bound.
2. **Double [proved, Lean: `double_iterate`].** If all fact states update and read each other, the span of a
   composite can double per read. Reach is `2^t` after `t` reads.
3. **Direction under the causal mask [proved here].** In CHAIN the head statement comes first and the chains are
   merged at random. No statement sees the fact that resolves its right side. A fact state can only compose
   backward: it can learn who points at it. Doubling stays valid in that direction, because the state it reads is
   earlier in the text.
4. **Break-even [proved here, arithmetic].** Costs in blocks:

| Scheme | Cost | At `n = 1024`, `L = 6`, `h = 16` |
|---|---|---|
| One thought walks per question, one hop per block | `q * h` | `16 q` |
| All tokens double, one doubling per block | `n * ceil(log2 h)` | 4,096 |
| All tokens loop at the measured rate `c = 3.5` hops per pass | `n * L * h / c` | 28,087 |

   The walk is cheaper than ideal doubling while `q < n * log2(h) / h` (256 questions). Against the measured loop
   the break-even is `q < n * L / c` (1,755 questions), at every `h`.

5. **Regime of the trained loop [measured, NARROW].** Solved hops at eval depth 1 / 2 / 3 / 4 / 6 are
   3 / 6 / 8 / 12 / 16 at full width and 2 / 4 / 6 / 8 / 12 at half width. The growth is additive. Doubling predicts
   a reach that at least doubles per pass. So the trained loop pays the parallel price and gets a serial rate: about
   2 blocks per hop at full width, 3 at half width.
6. **Relay account [conjecture].** Each fact state copies a label from the one earlier fact whose value equals its
   key. The match is static. The content it copies is updated. One copy moves a label one hop, so growth is
   additive, frozen facts cannot do it, and the head of a chain is a natural stop (it has no earlier fact). This
   fits four of the five readings. It does not explain STATEMENT.

Experiment. Eval only, checkpoint `hops2-tokenloop-s1`, CHAIN's `nodes_all_but_one` questions on long chains, read
by chain length and by asked node. Control: the same checkpoint's head questions. If the needed depth follows the
asked node and not the chain length, the question position walks and item 6 is refuted. If it follows the chain
length, the facts relay.

### Proposition 2. Finished-state reads turn loop depth into arrival order

Let `Phi` be one pass over all positions. It is causal: position `i` depends on positions `<= i`.

1. **[proved here]** Suppose the all-token loop converges to `H*`, and each position has a unique fixed point given
   the earlier states. Then a sequential process gives the same `H*`: for `i = 1 .. n`, run position `i` to its
   fixed point while it reads the finished states of the earlier positions. Proof: `H*` satisfies
   `h_i = Phi_i(h_<i, h_i)` for every `i`. Induction on `i` with uniqueness.
2. **Cost [proved here].** The all-token loop costs `n * L * T`, where `T` is at least the longest dependence chain
   divided by the reads per pass. The sequential process costs `L * sum_i tau_i`, where `tau_i` is what position
   `i` needs when its inputs are final. Under the relay account `tau_i` is constant. The chain task then costs
   about `n * L`, with no `h` in it. A question at any node needs two reads (its fact's label, then the end fact
   with that label).
3. The serial depth `h` does not go away (`hop_lower_bound`). It moves to the sequence axis. Generation is serial
   on that axis already. Training is not: finished states are not teacher-forced.

This is the exact content of requirement 6. It also says the chain task cannot test halting in such a design.

Experiment. Eval only, `hops2-tokenloop-s1`: decode each row position by position, and let every pass read the
final-pass keys and values of the earlier positions. Control: the same checkpoint in the normal loop at the same
passes per position. Refuted if the solved hop count at 2 passes per position is not above the control's at eval
depth 2. Risk: pass 1 never read finished neighbours in training.

### Proposition 3. A random unseen stop pays for a decodable state at every depth

Map `f`, reader `g`, states `h_t = f(h_(t-1), x)`, loss `l_t = loss(g(h_t), y)`. The stop `T` has law `pi`,
independent of `(x, y)`, and `f` does not see it.

1. **[proved here]** `J = sum_t pi(t) * E[l_t]`. One orbit and one reader serve every depth. Let `l*_t` be the best
   loss any state can reach after `t` passes. Then `J >= sum_t pi(t) * l*_t`, with equality only if the model is
   optimal at every depth in the support at once.
2. **[proved here]** A plan that leaves passes `1 .. k-1` undecodable pays `sum_(t<k) pi(t) * (l_t - l*_t)`. It is
   better only if its gain at depth `>= k` is larger. For Poisson(6) clamped to `[1, 8]`, `pi(1) = 0.017`,
   `pi(2) = 0.045`, `pi(3) = 0.089`. So the price of a scratch pass 1 is small. The objective prices early
   decodability. It does not force it.
3. **Zeno steps are permitted, not implied [proved here].** In the bounded-step model of DEPTHLAW the optimum under
   this objective takes full steps and then stops (`greedy_infDist`). Also, the mean CE by depth cannot separate
   "each token takes shrinking steps" from "each token takes full steps and fewer tokens are unfinished"
   (`tokens_kcurve_pos_iff` sums a step function over tokens, and a mix of job lengths gives any falling curve).
   The measured web gains per pass are 0.0368, 0.0076, 0.0023, 0.0010, 0.0001.
4. **A second source of the decay [conjecture].** The injection adds a geometric series in the pass count, and a skip
   around the loop removed most of the web K-curve **[measured, LESSONS section C]**. Part of the Zeno shape may
   be the injection decay and not a learned plan.
5. **The map sees the passes remaining [proved here].** The orbit may now differ per budget. The loss touches only
   the state at zero remaining. Every clockless solution is still a solution, so the optimum is no worse. Whether
   it is strictly better on a given task is open.
6. **The stop depends on the input [proved here].** With an oracle stop at completion, no state before completion
   and no state after it is in the loss. With a learned stop, the loss touches the states where the stop has mass.
7. **On chains the pressure costs nothing [conjecture].** No early answer exists, the walk's current node is the
   best guess, and the end of the chain absorbs. So a count-down should not raise hops per pass on head questions.
   This is against the hypothesis in CLOCK.

Experiments. (i) CLOCK as planned: item 7 is refuted if the count-down loop is above `hops2-tokenloop-s1` by its
written rule. (ii) Item 2: train the token loop from `hops-web-tokenloop-s1` with the stop law moved off depths 1
to 3. Control: `hops2-tokenloop-s1`. Chains that the control solves at eval depth 1 should fail at eval depth 1 in
the new arm. If they do not, the stop law is not what keeps early states decodable. (iii) Item 3: per-token loss by
depth on the web checkpoint, eval only. Shrinking gains inside one token support Zeno. One step at a
token-specific pass supports full steps.

### Proposition 4. A clockless map and budgets

1. **[proved here]** Without a clock the answers at budgets `1, 2, ..` are `g(f^T(h_0))`: cuts of one orbit. The
   path for budget `T` is a prefix of the path for `T + 1`. The map cannot run one plan for a short budget and
   another for a long one. A count-up clock keeps this limit. A count-down removes it.
2. **[proved here]** "Correct at every budget `>= k(x)`" holds iff the orbit's tail stays in `g^-1(y)`. So the done
   states must contain a set that `f` maps into itself. On chains the data gives it: the end has no next fact
   (`fixed_after`, HOPS). Elsewhere `f` must compute "done" on every pass.
3. **[proved, Lean: `washout`, `reinjected_rollouts_stay_apart`]** A budget given once at entry decays in a
   contractive loop. A clock added at the injection on each pass does not.
4. **[conjecture]** A count-down loop loses accuracy at budgets past the trained range, where the clockless loop
   holds (16 hops at depth 16 **[measured, NARROW]**).

Experiment. CLOCK's readings at budgets 12 and 16 against `hops2-tokenloop-s1` test item 4. The Proposition 1 eval
tests item 2: if accuracy on a mid-chain question falls when depth runs past the needed depth, the loop has no
learned done state, only the one the data gives.

### Proposition 5. Depth and a run of zero-length thoughts

1. **[proved here]** Let thought `t` enter with the exit state of thought `t - 1` and read `R` only. Then `T`
   thoughts with `N = 0` are a depth-`T` loop at one position. The loop is the trail with the reads of its own
   earlier states masked.
2. **[proved here]** Without that exit-to-entry feed, extra positions add no serial depth. Every dependence path
   still has at most the model's layer count in reads.
3. **[proved here]** A readable trail adds three things. Memory: `T` states of working memory where the loop has one.
   Reads by others: later thoughts see the middle results. A count-up clock: the trail's length. It adds no
   speed over static facts (Proposition 1, item 1), and it costs `T` stored states where Preserve asks for one.

Experiment. The depth-1 arm with `k` filler positions before each answer, no feed. Control: `hops2-plaind1-s1`.
Item 2 predicts the same solved hop count at every `k`. A staircase in `k` refutes the lookup model.

### Proposition 6. A choice at each step against a budget up front

1. **Power [proved here].** A choice at each step realises any stop that is a function of the state so far. A
   budget up front must name the stop before the work. On chains the hop count is not known in fewer than `h`
   reads (`hop_lower_bound` applied to the distance to the end). So an exact budget needs a thought that does the
   work first. Any other budget is an upper limit and needs the done set of Proposition 4.
2. **Signal [proved here].** Without a clock, the paths for all stops share a prefix. One run of `T_max` passes
   gives every `l_t`, and `sum_t p_t * l_t + lambda * sum_t t * p_t` has an exact gradient. This is Proposition 3
   with a learned `pi`, so the every-depth pressure comes back where the stop has mass. With a count-down each
   budget has its own path. The exact sum costs `T_max^2` passes, so the budget must be sampled. That is where
   score-function (REINFORCE) variance enters. It is large on web text, where per-token CE noise is far above a
   0.001 to 0.04 gain per pass. It is small on chains, where the gain is a jump from chance.
3. **The price [proved here].** With `lambda = 0` in training and a loss that does not rise with depth, the stop gets
   no signal to stop early. A stop rule needs a price or a target.
4. **Keeping the rule [proved here for the stopping part; the rule's reading is Wolfe's call].** Train the map with
   the main loss only. Fit a detached head that predicts the future drop of the main loss. Use `lambda` at eval
   only. A one-step head is optimal only when gains shrink (the monotone case of optimal stopping). Web text is
   that case, and there the rule equals "halt when the state stops moving". On chains the gains are
   0, 0, .., jump, so a one-step head halts at pass 1. The head must predict the best future drop net of cost.

Experiment. Offline, on `hops2-tokenloop-s1` states: fit both heads on per-row loss by depth. Control: the oracle
stop (first depth with the right answer). The one-step rule should halt at pass 1 on chains above the depth-1
reach, and the value rule near the oracle. No training of the map.

## 4. What this does not explain

**The failed walker.** A position that walks static facts is expressible (Proposition 1, item 1), and three runs
did not learn it (ANSWER, STATEMENT, START). Candidate causes, cheapest test first:

1. The full loop never walks at the question. Its facts relay. Test: the Proposition 1 eval.
2. The walker's pass-2 query does not fit keys that pass 1 built for another job. Test: attention maps of the
   routed positions in `hops2-answerrouted-s1` at pass 2 and up, eval only.
3. Frozen reads are the cause, not the 1 % to 9 % share of trained tokens. Test: the all-token shared-cache arm on
   the hop stream, one run. Every token loops and reads pass-1 keys.
4. It exists and is hard to find. Test: start the statement-routed arm from `hops2-tokenloop-s1`, one run.
5. The relay state is on the name tokens. STATEMENT gave current keys to sentence ends only. Test: route every
   token of hop rows and none of web rows, one run.
6. 20,000 steps is short for 1 % of tokens. Test: a longer run. Most costly.

Also open: why both token loops answer graph questions worse than their depth-1 models (NARROW), why width sets
hops per pass, and why the narrow loop falls from 24 to 16 solved hops between depth 12 and 16, which is against a
clean done set.

## 5. Questions for Wolfe, with his answers of 2026-11-20

1. Does a compute price in the training loss break "only the main loss buys depth"? It sets no depth target.
   **Answer: allowed.** Without a price nothing tells the model to stop.
2. Is a detached head that predicts future main-loss drop allowed as the halter?
   **Answer: allowed.** A detached head cannot shape the map. Measured caution (signal-picked depth probe,
   2026-11-19): on web text an oracle pick of a quarter of the sentences gets 57 % of the gain for 41 % of the
   extra passes, so the most a halter can win there is about 1.4 times a random pick.
3. Finished-state reads make training serial along thoughts. Is the plan synchronous sweeps in training and
   finished reads at inference, or serial training?
   **Answer: synchronous sweeps in training, finished reads at inference.** First test of the cheap end of this:
   the shallow token loop plan of 2026-11-20 (`parcae-wt-hops`, `2026-11-20-parcae-shallow-loop-finished-reads.md`).
4. Does your interpretability evidence for Zeno steps hold per token, and with the injection decay removed?
   **Open.** Measured half (Zeno per token, 2026-11-19): web tokens close about 64 % of the remaining loss per
   pass; chain answers jump in one late pass. The decay-removed half is not tested.
5. Must an old thought change after new tokens arrive? If not, requirement 6 becomes "read finished states".
   **Answer: either can be built, but a thought that changes after later text has read it would probably break
   the model (exposure bias: the readers trained on a state that no longer exists).** Working rule: a finished
   thought is fixed, and requirement 6 is "read finished states".
6. Which task tests halting? Chains amortise across arrival order (Proposition 2). The work must depend on the
   question.
   **Open.**
