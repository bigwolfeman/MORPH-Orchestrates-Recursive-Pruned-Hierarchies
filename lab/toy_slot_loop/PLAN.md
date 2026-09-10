# Toy slot-loop study: plan and frozen predictions

Date: 2026-09-10. Written and committed BEFORE the grid ran. Predictions are frozen: they
are not edited after any result is seen. If the method has to change, the change is
appended below with its date and reason, and the predictions stay as written.

Motivating readings on the real model, all from today:

- [`lab/experiments/results/2026-09-10-slot-gradient-probe/README.md`](../experiments/results/2026-09-10-slot-gradient-probe/README.md)
  — the per-pass cotangent is FLAT under the prelude entry (share 0.168 / 0.168 / 0.166 /
  0.160 / 0.154 / 0.183), the MUX pays for the loop 7.3x over the token CE, and the six
  passes' updates to the shared core cancel: `|sum_t dW_t| / sum_t |dW_t|` = 0.520.
- [`lab/experiments/results/2026-09-10-slot-z-optimize/README.md`](../experiments/results/2026-09-10-slot-z-optimize/README.md)
  — the coda's CE with the loop's exit equals its CE with the loop's ENTRY to within
  0.0002-0.0033 nats, while a gradient-fitted z is worth 0.9-2.6 nats. The reader is fine;
  the writer is empty.
- [`.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md`](../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md)
  — Wolfe's framing: the loop fails on credit assignment, and the fix under test at full
  scale is supervision at every pass.

The real model cannot separate "the attachment is wrong" from "the target needs no
iteration", because on web text nobody knows how much iteration the next span needs. This
toy makes the iteration requirement a property of the DATA, so the two can be separated.

## The model (exact sizes)

`lab/toy_slot_loop/model.py`. One row, one span: `[3 token cells][1 slot cell][2 prefix
cells]`, 8 spans, 48 cells, 24 token positions.

| part | value |
|---|---|
| d_model / heads / d_ff | 96 / 4 / 256 |
| prelude | 2 pre-norm blocks, causal over ALL cells |
| core | **1** shared pre-norm block, applied T times to the 8 slot cells only, causal over slots |
| coda | 2 pre-norm blocks; token cells read RAW embeddings, token->token attention is restricted to the own span, slot cells are never keys, prefix cells are causal keys |
| depth | T per slot ~ clamp(Poisson(6), 1, 8); a slot freezes once it reaches its depth |
| entry | `h_0 = W_in(e)` with `e` the prelude state at the slot cell; injection `h += 0.45^t · W_inj(e)` on the first 24 channels every pass |
| write | `z -> W_prefix -> 2 prefix cells` |
| heads | token head and MUX head both tied to the input embedding |
| optimiser | AdamW, lr 2e-3, betas (0.9, 0.95), wd 0.01, 200-step warmup then cosine to 0.1x, clip 1.0 |
| budget | 3,000 steps, batch 128, 3 seeds per cell |

The coda's token input is the raw embedding, not the prelude output. That is the real
model's `coda_token_input: embed`. With the prelude output there, causal attention would
already put every earlier span at every token position and z could not matter by
construction — `selfcheck.py::t_prelude_leak` measures both and shows the difference.

## The tasks

`lab/toy_slot_loop/tasks.py`. Input alphabet: the 6 elements of the permutation group S_3.
Answer alphabet: 6 output-only ids. **The answer never appears as an input token.** It
appears as the LABEL at the first token of the NEXT span (the double-label shape of the
real model's spec sec 5). Every other token position carries the ordinary next-symbol
label, which is uniform by construction (floor ln 6 = 1.7918), exactly as most of the real
token CE is local statistics.

1. **compose** (iterative state tracking). Span i's product `P_i` is the ordered product of
   its 3 symbols in S_3; the register is `R_i = R_{i-1} · P_i`. Label at the head of span
   i+1 is `R_i`. S_3 is not commutative, so no bag or average computes `R_i`: composing 8
   spans in order needs at least ceil(log2 8) = 3 rounds of prefix combination even with a
   perfect parallel scan, and the prelude's 2 layers are largely spent turning 3 raw
   symbols into `P_i`.
2. **summary** (one-pass control). Label at the head of span i+1 is the most frequent
   symbol of span i, ties to the smallest id. Slot i's own prelude already sees every
   symbol of span i, so this needs no iteration.

MUX target for slot i: `R_i` (compose) or the mode (summary). The staged attachment's
own-span target is `P_i` (compose) or the same mode (summary).

## Precondition: the capacity ladder (runs before the grid)

Train at fixed depth 1, 2, 3 and 6 on `compose`, at both `exit` and `mux_all`, and evaluate
at the trained depth. The task is admissible only if depth 1 lands near chance (1.79 nats,
accuracy about 0.17) and depth 6 lands well below it. `summary` at fixed depth 1 must
already solve its task, and `compose` at depth 6 with the coda blind to z must sit at
chance. If depth 1 solves `compose`, the METHOD changes (more spans and/or longer spans)
and the change is recorded here; the predictions below do not change.

## The grid

**A. attachment x task** (default entry, decay 0.45, no fixed-point term, shared weights,
coda reads z), 6 x 2 x 3 seeds = 36 cells.

| id | attachment |
|---|---|
| a | `exit` — MUX at the exit state only. What the real model does today. |
| b | `mux_all` — MUX at EVERY pass, live carry. |
| c | `mux_all_detach` — MUX at every pass, carry detached between passes (the DB style). |
| d | `staged` — own-span target at every non-final pass, next-span target at the exit. |
| e | `deep_coda` — the coda is run on the state after EVERY pass and the token CE averaged (TRM/HRM style). |
| f | `progressive` — exit only, plus Bansal's random no-grad prefix (p = 0.5). |

**B. one-factor extensions**, all on `compose` at attachment `exit`, 3 seeds each, 7 cells:
noise entry; injection decay 0.9; injection off; fixed-point term at lambda 1.0; per-pass
LoRA rank 8; coda blind to z; truncated BPTT with grad on the last 2 passes only.

57 cells at roughly 80 s each on the RTX 3070. **Cut on purpose:** the extensions are NOT
crossed with the other five attachments, there is no learning-rate or depth-mean sweep, and
there is no longer-horizon (20k-step) draw. Three seeds cannot separate differences under
about 0.02 nats and nothing below that bar will be called a result.

## Instruments (per cell)

`lab/toy_slot_loop/instruments.py`, ported in spirit from `lab/divergence/`:

- **K-curve.** Value-position CE, value accuracy, total token CE and MUX CE at forced depth
  1, 2, 3, 6, 8, 12 on a fixed 2,048-row eval draw shared by every cell.
- **Write contribution.** The same CE with z = exit, z = entry (zero passes) and z = 0.
- **Per-pass cotangent** at the loop state, and **per-pass share and cosine** of the shared
  core weight gradient, for three sources (total, token CE alone, MUX alone), at forced
  depth 6 on 4 x 64 rows. The mandatory self-check is `sum_t dW_t == leaf.grad` in the same
  backward. The cancellation ratio's reference values are 1/sqrt(6) = 0.408 for six
  orthogonal equal-norm updates and 1.0 for six aligned ones.
- **Participation rank** of the exit states across slots, and of the entry states.

## Predictions (frozen 2026-09-10, before the grid ran)

Bars are on `compose` unless stated. "Earns depth" means value-CE K1-K6 > 0.30 nats.

- **P1. Exit-only earns depth on `compose`.** 65 %. The exit MUX target is computable and
  needs iteration, the cotangent is flat under the prelude entry (measured on the real
  model), and full BPTT gives every pass a path. If this holds, the real model's flat
  K-curve is NOT caused by the attachment, and the study's headline is about the target.
- **P2. Exit-only earns nothing on `summary`.** 90 % that value-CE K1-K6 < 0.05 while the
  value accuracy is above 0.9. A task solvable in one pass cannot make a loop earn depth
  under ANY attachment; this is the control that says K-curves measure the task, not the
  wiring.
- **P3. Dense per-pass supervision (`mux_all`) is WORSE than exit-only on `compose`.**
  60 % that its value CE at depth 6 is higher than exit-only's. Asking pass 1 to already
  emit `R_i` asks for something no map can do in one pass; averaging that impossible term
  over all passes biases the map toward the best one-pass answer (the last span's product)
  and against building a carry. This is my main disagreement with the "supervise every
  pass" intuition.
- **P4. Detached carry (`mux_all_detach`) is the worst attachment on `compose`.** 75 %.
  Detaching the carry deletes the very product the task needs: the map can no longer be
  trained to prepare a state for a LATER pass.
- **P5. Staged (`d`) beats dense (`b`) on `compose`.** 55 %. Its intermediate target is
  reachable in about one pass, so it structures the state without asking for the
  impossible, and the exit still carries the composed target.
- **P6. Deep coda (`e`) lands within 0.05 nats of `mux_all` (`b`) on `compose`** at about
  6x the cost. 50 %. Both ask for the final answer at every pass; the only difference is
  which head reads it.
- **P7. Progressive (`f`) is within 0.05 nats of exit-only at depth 6, and better than it
  at depth 12.** 55 % on the first clause, 70 % on the second. A random no-grad prefix
  trains a map that must improve arbitrary states, which is what a scan step is; the
  measurable consequence is depth generalisation, not depth-6 CE.
- **P8. Cancellation tracks earning, not attachment.** 60 % that across all 36 cells of
  grid A the Pearson correlation between the cancellation ratio and value-CE K1-K6 is
  positive, and 70 % that every `summary` cell reads a lower cancellation ratio than the
  matched `compose` cell. If a shared map has ONE consistent job at every pass, its passes
  agree; the real model's 0.520 is then a symptom of a map with nothing to do, not a cause.
- **P9. The noise entry hurts `compose`.** 70 % that it is at least 0.10 nats worse than
  the prelude entry at depth 6, and 80 % that its per-pass cotangent share is a ramp with
  pass 6 at least 5x pass 1 (the real model read a 33x ramp).
- **P10. The fixed-point term hurts `compose`.** 65 % that it costs at least 0.05 nats. A
  scan must keep changing the state; a terminal fixed-point term prices that motion.
- **P11. Per-pass LoRA changes nothing on `compose`.** 70 % that it is within 0.05 nats of
  exit-only. A scan step is the SAME map at every pass, so per-pass capacity has nothing to
  buy. (On the real model this is the sibling arm now running at full scale.)
- **P12. Truncated BPTT (last 2 passes) destroys `compose`.** 85 % that value CE at depth 6
  is at least 0.30 nats worse than full BPTT. This is the sharpest prediction here and the
  one that says the real model's full-BPTT choice is load-bearing whenever the task needs
  iteration.
- **P13. Coda blind to z sits at chance on both tasks.** 95 %. It is the channel gate.
- **P14. The write contribution separates the tasks.** 75 % that on `compose` at
  attachment `exit` the CE with z = entry is at least 0.30 nats worse than with z = exit,
  against the real model's 0.0015. If the toy reproduced the real model's near-zero write
  contribution on a task that NEEDS the loop, the toy would be broken, not informative.

## Reading rules, fixed in advance

- Nothing under 0.02 nats is a result at 3 seeds.
- A CE at 3,000 steps ranks nothing across cells that differ in parameter count
  (`pass_lora_rank`) or in cost per step (`deep_coda`). Only the K-curve, the write
  contribution and the gradient tables speak to depth.
- Every table reports mean and min-max over the 3 seeds.

---

## Method amendments (2026-09-10, after the gate, before the grid)

The predictions above are unchanged. The gate ran three times and forced three changes to
the METHOD. Every number below comes from a run whose JSON is in
`ignored/experiment-artifacts/2026-09-10-toy-slot-loop/`.

**Amendment 1 — the span's operator is its FIRST symbol** (`ladder/`, 9 cells). With the
span product defined over all 3 symbols, nothing learned the scan at any depth: value CE
plateaued at 1.23-1.34 against a chance of 1.79 at fixed depth 1, 2 and 3, and the MUX CE
plateaued at 1.31-1.40 with the training loss flat from step 1,500. Folding the within-span
product into the prelude consumed the whole budget. The scan is the object of study, so the
within-span work was removed: `P_i` is now the span's first symbol and the other two are
distractors that keep the ordinary next-symbol loss non-trivial.

**Amendment 2 — an explicit `geometry` switch, and the grid runs `strict`** (`ladder2/`,
6 cells). With the operator fix and MORPH's own geometry, a model trained at fixed depth 1
reached value CE 0.283 and accuracy 0.66, i.e. depth 1 nearly solved an 8-span scan. The
reason is architectural, and it is the study's first result: under MORPH's geometry the
prelude is causal over every cell, the core attends every earlier slot, and the coda reads
every earlier prefix cell, so the prelude and the coda can compose across spans on their
own and the loop is not the only cross-span path. `geometry: strict` closes all three
routes (prelude within-span, core attends the previous slot and itself, a token reads only
the previous span's prefix cells, and prefix cells do not chain to each other). Under
`strict` the depth price is exact and PROVEN by measurement, not by argument:
`selfcheck.py::t_depth_requirement` differentiates the value logit at the head of span 5
with respect to span 0's operator cell and reads 0.000e+00 at depths 1, 2 and 3, and
9.7e-03 at depth 4. Both geometries stay in the tree and the permissive one is reported.

**Amendment 3 — 5 seeds and 4,000 steps, because the outcome is bimodal** (`ladder3/`,
`diag/`). Under `strict`, `compose` has exactly two attractors. The one-pass shortcut sits
at the depth-1 reachability ceiling: (7-2)/7 · ln 6 = 1.2798 nats, and the measured plateau
is 1.281 on every stuck run. The full chain sits at 0.000 with accuracy 1.000. Trained at
fixed depth 6 the escape happened on 1 of 3 seeds; at fixed depth 8 on 2 of 3. Stuck runs
were flat for 2,000 steps, so this is a basin, not a budget. The primary metric is
therefore the ESCAPE RATE over seeds, with the escape step from a 256-row probe run every
250 steps, and the seed count goes from 3 to 5 and the budget from 3,000 to 4,000 steps.
Grid B gains one cell, `fixed_depth 6`, because ragged per-slot depths break a chain that
needs slot i-1 to have run at least as many passes as slot i.

The reachability ceiling at forced depth d, strict geometry, 8 spans:

| d | 1 | 2 | 3 | 6 | 8 |
|---|---|---|---|---|---|
| best possible value CE | 1.2798 | 1.0239 | 0.7679 | 0.0000 | 0.0000 |
