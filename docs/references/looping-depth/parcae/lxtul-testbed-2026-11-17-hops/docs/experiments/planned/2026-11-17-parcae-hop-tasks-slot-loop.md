# Parcae: does the strict slot loop answer more hops than its depth-1 twin on hop-demanding documents?

Status: planned

Date proposed: 2026-11-17. Branch `lxtul/hops` (worktree `parcae-wt-hops`), cut from `lxtul-testbed` 89b6e4d.

## Question

When hop-demanding documents are mixed into the web-text stream, does the strict slot loop answer more hops than
the same model trained and read at depth 1?

## Why now

- Three codebases agree that a slot loop adds about nothing to next-token CE on web text. On this testbed the
  minimal 1-cell strict slot loop reads K1-K6 -0.0001 (`failures/2026-10-14-parcae-minimal-slot-loop-5k.md`).
- The published record says loop depth past 2 passes buys reasoning, not next-token loss: looped models gain on
  multi-hop, no-CoT and in-context tasks while NLL stays at or above a dense model (arXiv 2602.16490, 2610.31421,
  2512.08819; a LessWrong study of a 4B looped model reads bits per byte flat after 2 loops while a reasoning index
  rises to 8 loops). A 592-run scaling study (arXiv 2610.26030) finds a loop multiplies effective parameters
  (1.91x at 2 loops, ceiling near 3.0x) and does not save FLOPs. Notes: MORPH `docs/references/paper-notes/`.
- On a graph-search task with no text, a Parcae token loop solved about one more hop per pass and reached 16 hops
  with passes it never trained on, while a flat model stopped near 12 (hnet repo,
  `successes/2026-11-16-energy-loop-graph-search.md`). The same question for a COARSE loop was planned there
  (`planned/2026-11-16-hop-chain-coarse-loop.md`) and never run. This experiment runs it on the strict slot loop.
- Wolfe, 2026-11-17: "I like this, we could also test graphqa maybe? Or something similar."

So web-text CE may be the wrong instrument for a slot loop. This test reads the loop on content that needs serial
lookups, with a fixed chance anchor.

## Task

Two families of generated documents, mixed into the OpenWebText stream by rows. The loss is plain next-token CE on
every token. No extra target.

- **Hop chains** (ported from hnet `chaintask.py`). A document holds K chains. Chain k is
  `Y_l is Y_(l-1). ... Y_1 is V_k.` with the head first, and the chains are merged at random. Then half of the
  chains are asked: `Find Y_l now. It is V_k.` The answer needs l lookups. Names and values are fresh single-token
  words per document. Each statement is a 4-token sentence, so it is one span. Chance for a model that picks a
  value of the document is 1 / K.
- **Friendship graphs** (GraphQA style: the friendship encoder and the question strings of the public GraphQA
  code, our own generator). A random forest of two trees, one `<A> and <B> are friends.` sentence per edge in
  random order, then questions: reachability (Yes / No balanced, chance 0.5; hop label = path length for Yes),
  shortest path length (one-token answers, distance 1 to 9), and edge existence (the 0-hop control, chance 0.5).

Two facts that shape the reading:

- The pointer head copies the token that followed an attended position, from any earlier span. It can do the LAST
  lookup of a chain by itself. So an answer with hop label h needs h - 1 lookups in hidden states and one copy.
  This holds for every arm.
- One loop pass is six core blocks, and the coda has four more layers that read the slot states. A depth-1 model
  can chain several lookups in one pass. That is why the hop counts run to 32.

## Arms (2 seeds each)

| arm name | config | what loops |
|---|---|---|
| slot loop | `arm=minimal` | six core blocks over one state per span, Poisson(6) passes in [1, 8]; tokens attend inside their span |
| slot twin | `arm=minimal` with `lxtul.mean_depth=1 lxtul.max_depth=1 lxtul.eval_depth=1` | the same model, one pass |
| token loop (reference) | `arm=plain_pointer` | six core blocks over every token, Poisson(6) |
| flat (reference) | `arm=gpt8_pointer` | nothing: eight layers per token |

The slot arms hold up to 256 slots per row (stock 64), because a chain statement is 4 tokens. Web-text rows keep
the stock packing (64 slots, the same tokens as every filed run) and are widened with tail pads to the 256-slot
row shape. Raising the slot budget for the whole stream was tried first and rejected in the smoke: the packer
fills a row to its position budget, so web-text rows grew from about 1037 to about 1219 tokens for the slot arm
and overflowed the flat arms' eval rows.

## Hypothesis

I think the slot loop answers more hops than its twin. One lookup is one attention step from a slot to the earlier
slot whose statement names the wanted word, and a weight-tied loop repeats that step once per pass. The strict
layout makes the slot states the only path between sentences, so the lookups must happen there. A chain statement
holds two words, so one state per statement is a fair unit.

Evidence against, read before writing the predictions:

- The twin is not shallow: six core blocks in its one pass, four coda layers and the pointer. It may reach the
  whole training range of hops, and then only the hop counts above the training range can separate the arms.
- Later slots can absorb earlier ones, so reach can double per layer. Untied layers reached 8 to 12 hops with six
  blocks on the graph-search task.
- The first graph-search run sat at chance for 20,000 steps until it had mixed graph sizes and many targets per
  row. Here the only signal is next-token CE at the answer tokens, and half of the rows are web text.
- Every earlier slot-loop positive on this testbed came with a worse depth 1, not a better depth 6.

## Predictions

Terms. Hard-hop accuracy of a run on the chain family = its mean exact-match accuracy over the eval hop counts
6, 8, 12, 16, 24 and 32, at the eval depth (of 1, 2, 3, 4, 6, 8, 12, 16) where that mean is highest. A depth-1
twin and the flat model have one depth. Hard-distance accuracy on the graph family = the balanced reachability
accuracy at distances 6, 8, 12, 16 and 24: the mean of the Yes accuracy at those distances and the No accuracy
on the same documents, so a model that always answers Yes reads 0.5. Fixed anchors: 0.25 (chain chance at 4
chains), 0.5 (reachability chance), 1.0.

Main prediction (only this one answers the Question):

- The slot loop's chain hard-hop accuracy is above its twin's in both seeds: both slot loop seeds read above both
  twin seeds. If so: supported. If both twin seeds read at or above both loop seeds, or the seeds interleave: not
  supported. If neither arm reads chain accuracy at 1 hop above the gap between its two seeds over chance: can't
  tell (nothing learned the chains).

Other predictions:

- The same direction on graphs: the slot loop's reachability hard-distance accuracy is above its twin's in both
  seeds.
- Ordering on chains: the token loop's hard-hop accuracy is at or above the slot loop's in both seeds. The token
  loop has four positions per statement and full attention; the slot loop has one state per statement.

If supported, we do: the slot loop contributes where serial lookups are needed. Next: the FLOP ledger for the
loop on chain, graph and web rows; the same test on the 4-cell fan and on the series span arm; then a web-text
arm that keeps the hop gain without the strict layout's CE cost.

If not supported, we do: read the token loop against the flat model. If the token loop extends with eval depth
and the slot loop does not, one state per statement is the limit, and the next arm is the 4-cell fan on the same
task. If no loop extends past the flat model, untied depth covers this hop range, and the test moves to longer
chains before any design change.

If can't tell, we do: raise the task share (or run a task-only warm start) on the flat arm until one hop is
learned, then rerun.

Gates (a failed gate means a broken run; gates are not predictions):

- The new CPU tests pass (chain documents against a reference reader, graph documents against a BFS solver, row
  builder, mixing, label alignment through the packer, the accuracy function). `test_geometry.py` passes at 256
  slots.
- A 60-step smoke of each arm ends with a finite loss and logs hop accuracy for every eval bucket.
- No answer leaks inside a row: "repeat the previous answer" and "majority answer so far" baselines read chance
  on the eval rows and on training rows, for every family (a CPU test). The first eval rows failed this: each
  graph bucket's documents shared one answer, and a 60-step model read 0.87 to 0.94 on them.
- The first full run (flat, seed 1) reads chain accuracy above 0.9 at 1 hop at its last step. If it does not, no
  other arm runs: the stream does not teach the format, and Method is amended first (dated).
- Every run: finite loss to the last step, peak memory under the card.

Readings, not predictions: the full accuracy table by family, hop count and eval depth for every run; web-text
CE by depth (K1-K6) for the loops; shortest-path and edge-existence accuracy; ms/step and peak GiB; the share of
positions the loop runs on per row type (1 in 4 on chain rows, about 1 in 21 on web rows).

## Method

```
code:    branch lxtul/hops at the commit that adds this file
stream:  rows by a seeded per-row draw: web text 0.50 / hop chains 0.25 / friendship graphs 0.25
         chain train documents: K ~ U(2, 5) chains, lengths ~ U(1, 16)
         graph train documents: n ~ U(8, 40) nodes, 8 questions (reachability 0.5 / shortest path 0.3 / edge 0.2)
         task rows hold whole documents from the row start; the same documents reach every arm for a seed
budget:  20,000 steps, batch 6 x 1024, the testbed's schedule and optimizer as filed for each arm
loss:    next-token CE on every token
eval:    fixed rows from a seed the training stream never uses, the same documents for every arm
         chain: 4 chains per document, hop counts 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
         graph: 40 nodes; reachability distances 1, 2, 3, 4, 6, 8, 12, 16, 24; shortest path 1 to 9
         metric: exact match of the argmax of the model's full output (model + pointer mixture) at each answer
         looped arms at eval depths 1, 2, 3, 4, 6, 8, 12, 16; web-text depth sweep as in every testbed run
order:   seed 1: flat (the gate), slot loop, slot twin, token loop; then seed 2 in the same order
grade:   chain hard-hop accuracy per run at the last step, slot loop seeds against slot twin seeds
```

Exact commands (script `/home/wolfe/morph-scratch/abq/hops/run.sh`, one run at a time under the GPU lock):

```
python -m lxtul.train task=hops training.steps=20000 training.val_every=1000 task.eval.rows_per_bucket=24 \
       run.seed=<1|2> run.name=hops-<arm>-s<seed> <arm overrides>
flat        arm=gpt8_pointer
slot loop   arm=minimal arm.lxtul.compact_bucket_slots=512
slot twin   arm=minimal arm.lxtul.compact_bucket_slots=512 +lxtul.mean_depth=1 +lxtul.max_depth=1 +lxtul.eval_depth=1
token loop  arm=plain_pointer
```

`task=hops` is `lxtul/configs/task/hops.yaml`. The eval set is 240 chain rows (24 per hop count) and 120 graph rows
of mixed documents (6 reachability, 4 shortest-path and 4 edge questions each, a fair Yes / No coin per question,
a 24-edge path planted in each 40-node graph). Buckets are per answer, never per document.

Smoke readings (2026-11-17, 60 and 300 steps, the desktop GPU in use by a game, so the times are upper bounds):

| arm | ms/step | peak GiB |
|---|---|---|
| flat | 102 | 9.40 |
| slot loop | 234 (median of 9 windows after step 120) | 14.78 |
| slot twin | 186 (median of 9 windows after step 120) | 11.23 |
| token loop | 364 | 21.10 |

At these times one seed of the four arms is about 4.9 hours of training. After 300 steps the slot loop reads chance
on every family (chain 0.234 against 0.25, always-Yes on both Yes / No tasks, shortest path 0.101 against 0.111),
so the instrument reads its anchors on a model that has not learned the task.

## Method amendments

**2026-11-17, training documents (after a failed gate, before any graded run).** The first full run (flat, seed 1,
`hops-flat-s1`) read chance on every bucket at step 20,000: chain accuracy at 1 hop 0.237 (chance 0.25), balanced
edge existence 0.47, balanced reachability 0.47. Its web-text CE fell as usual (4.84 to 3.62). That is a failed
gate, so the run holds no reading for any prediction, and no other arm ran. A flat run on hop rows alone (chains
0.5 / graphs 0.5, 5,000 steps, `hops-diag-flat-taskonly`) also stayed at chance, so the web-text share is not the
cause. At the first ranges a chain row held 13.6 answers and 6 % of them were 1-hop answers. The hnet graph-search
run failed its gate the same way and was cured by small instances in the training mix plus dense targets
(`successes/2026-11-16-energy-loop-graph-search.md` there).

The change, the same for every arm, TRAINING documents only (the eval rows are byte-identical; a test pins their hash):

```
chain:  per document a length cap ~ uniform {1, 2, 4, 8, 16}, chain lengths ~ U(1, cap)        (task.sources.chain.len_caps)
        K - 1 chains are asked, each once, at a uniformly random node of the chain; the hop label is that node's
        distance to the value                                                                  (ask: nodes_all_but_one)
        measured on 2,000 rows: 33.6 answers per row, 58 % at 1 hop (before: 13.6 and 6 %)
graph:  n ~ uniform {4, 6, 8, 12, 16, 24, 32, 40}; clip(n // 3, 2, 12) questions, no (task, pair) asked twice
        measured: 16.7 answers per row, 16 % of reachability-Yes answers at distance 1 (before: 17.7 and 10 %)
```

The Predictions, the arms, the budget and the eval are unchanged. The gate run is repeated first.
