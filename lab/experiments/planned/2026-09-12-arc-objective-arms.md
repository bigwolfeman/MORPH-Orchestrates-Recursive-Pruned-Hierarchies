# Planned: the objective arms — ask the passes for depth, and ask the coda for the span

Status: planned

Date: 2026-09-12 12:09 CDT (frozen before any arm ran; no GPU step of either arm below
exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Extends
[`2026-09-12-arc-strict-geometry.md`](2026-09-12-arc-strict-geometry.md) and runs on ITS
geometry with coda reach `all`. Design note:
[`2026-09-12-objective-arms-per-pass-plan-and-parallel-coda.md`](../../../.agents/notes/proposed/architecture/2026-09-12-objective-arms-per-pass-plan-and-parallel-coda.md).

## What the strict panel already measured, and why that produces these two arms

* **The loop's write is now the whole cross-span channel.** On `slot-spandec-strict`
  (`tul.tg_geometry: strict`, coda reach `all`) the slot channel is worth **0.187** nats
  and `worth_profile`'s `all_slots` equals its `zero` by construction, so nothing crosses a
  span boundary except through a pass. The bypass the 2026-09-11 span-decoder record named
  is closed.
* **The passes still read zero.** Token **K1−K6 0.0016** on that arm. Twelve arms since
  2026-09-04 now sit in [−0.0001, +0.0033] — one geometry change, one target change, one
  entry change, one depth draw, one width, one HCA fix, one seed and the core itself.
* **Forcing reachability DOES move the curve, and it moves the wrong thing.** Amendment 3:
  `slot-spandec-strict-prev-reach1` (coda reach `prev`, `loop_reach` 1) reads tokens
  **K1−K6 +0.0163** [+0.0153, +0.0173], **K3−K6 +0.0042**, at depth-6 CE parity with
  `slot-spandec-strict-prev` (**+0.0023** [−0.0000, +0.0046]). It gets there by BLINDING
  the coda: a token may read only the previous slot's cells, so everything older has to
  walk the chain of loop states. Wolfe: "I am suspect of this method. It proves that TUL
  can work for sure. But the amount of blindness here is concerning. z has to hold the
  history when it should hold the present next thought that needs decoding. Our objectives
  are still poor."
* **A longer target is worth something.** `slot-spandec-strict-h3`
  (`tul.spandec_horizon` 3) is **0.0093** nats better than the H = 1 strict arm, and the
  gain is entirely at offsets 3+.

So: the one lever that made passes matter did it by hiding information, and the one target
change that helped reached further ahead. Both arms below keep the coda's full reach — no
blindness — and change only what the model is ASKED for.

## Question

Does an objective that needs more than one pass, or a reader that must emit a whole span
at once, make the slot loop's passes carry something the coda uses — with nothing hidden
from the coda?

## Hypothesis

The per-pass K-curve has read zero on every unblinded arm because no objective has ever
distinguished pass `t` from pass `t−1`. `reach1` distinguished them by amputation. Arm A
distinguishes them by the TARGET: pass `t` is graded on spans `s+1 .. s+min(t, 6)`, so a
deeper pass can only win by planning one span further. Arm B attacks the same flat reading
from the reader's side: a coda state asked for eight (or thirty-two) tokens of the next
span AT ONCE, with no token path to lean on, has to get them from the one thing written
into the cell it sits on — the looped state.

## Method

Two arms, each ONE factor against `slot-spandec-strict`, 5,000 steps, seq 1024, batch 6,
seed 1, ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route off,
retention off, `tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`.

| arm | config | one factor against `slot-spandec-strict` |
| --- | --- | --- |
| `slot-spandec-strict-perpass` | `tul_slot_spandec_strict_perpass.yaml` | `tul.spandec_per_pass` |
| `slot-spandec-strict-codaspan` | `tul_slot_spandec_strict_codaspan.yaml` | `tul.coda_span_heads` 0 → 8 |
| `slot-spandec-strict-codaspan32` | `tul_slot_spandec_strict_codaspan32.yaml` | `tul.coda_span_heads` 0 → 32 |

### Arm A — the per-pass planning target (`tul.spandec_per_pass`)

The state after pass `t` is decoded into spans `s+1 .. s+H_t`, `H_t = min(t,
tul.spandec_pass_horizon_max = 6)`, at `tul.spandec_pass_tokens = 8` tokens per span, through
the SAME `SpanDecoder` the exit state is graded by (shared parameters; its own zero-init
position table, because `H_t` blocks of 8 is a different block geometry from the exit
target's one block of 32 and row 8 cannot mean two offsets at once). The EXIT term is
unchanged: `tul.spandec_horizon` stays 1, so z is still graded as the next thought.

A slot is graded at pass `t` only when its REALISED depth reaches `t` — `tul.oracle_z`'s
mask. Reduction: one CE per pass (a mean over that pass's graded tokens), then a plain mean
over passes, so passes weigh equally; a token-weighted mean would make the term mostly
about the deepest pass, which carries six times pass 1's tokens.

Nothing is detached. The term reaches pass `t`'s core application through the trajectory
and, through the live carry, every earlier pass. The decoder trains on it too — one reader,
one notion of a decodable plan. `tul.oracle_z` is REFUSED with it (both write a per-pass
target onto the same trajectory, and the oracle's teacher is computed from the decoder the
per-pass term is simultaneously training), and so is `spandec_horizon > 1` and `db_loop`.

**Cost, arithmetic** (row = 1,024 real tokens, `L_total` 1152, 64 slot cells, ~51 valid,
mean depth 6, 2 decoder layers), block-passes per real token:

| part | passes/token |
| --- | --- |
| model (prelude 4×1152 + core 6×51×6 + coda 4×1152) | 10.7 |
| exit span decoder, H = 1 (2 × 64 × 32) | 4.0 |
| per-pass target (2 × 64 × Σ_{t=1..6} min(t,6)·8 = 2 × 64 × 168) | 21.0 |
| **total** | **35.7** |

against `slot-spandec-strict`'s **14.7**, `-h3`'s **22.7** and the plain panel's **44.0**.
A depth-8 draw adds two more capped blocks (264 positions → 47.7). This is the most
expensive arm of the family.

### Arm B — parallel span decoding from the coda (`tul.coda_span_heads`)

At each slot's LAST prefix cell (`slot_index + prefix_k − 1`, the emitting position; under
strict that cell carries the looped state and nothing else) the coda's FINAL readout goes
through J parallel offset heads — the `_MTPHead` construction, RMSNorm plus a `[d, d]`
linear at IDENTITY init, so at step 0 every head reproduces that position's own next-token
head — and each head is scored through the tied table against token `j` of the next span.
Non-autoregressive: no teacher forcing, no token path. Loss = one mean CE over the valid
(slot, offset) pairs; the token CE is unchanged; the tied table is read detached
(`tul.mux_detach_head`, inherited).

`tul.coda_span_source: token` moves the read to the boundary TOKEN position and is a
CONTROL, not an arm: that token sits before its own slot's cells and the coda is causal, so
its state has never seen its own slot's z.

**Cost.** No block pass. J readout rows per slot: J × 64 per row against the token CE's
~1,024, so J = 8 is ~0.5× the main CE's readout and J = 32 is ~2×. The heads are J `[d, d]`
matmuls on `[B, S, d]` — 3.2 GFLOP at B 6, S 64, d 1024, J 8. ONE chunked
`fused_linear_cross_entropy` call over `[B·S·J, d]`, so the `[B·S·J, V]` logits (604 MB fp32
at J 8, 2.4 GB at J 32) are never materialised and the `[V, d]` fp32 `grad_w` accumulator is
paid once, not J times.

### The gate, built in this change

`tests/test_tul_spandec_per_pass.py` (22 tests), `tests/test_tul_coda_span.py` (20) and
`tests/test_tul_objective_arms.py` (3 — compose, build and RUN each config at the panel's
real budgets). Both arms are proved PURELY ADDITIVE rather than argued to be: every shared
parameter is byte-identical to an off-model from the same seed, and `loss − <term>_weighted`
equals the off-model's loss BIT FOR BIT (`torch.equal`, CPU fp32; measured 9.706100463867188
on both arms' fixture). Targets are checked against a Python oracle that walks `bag_id`
itself, never against a second call of the function under test.

Six source-level sabotages, run 2026-09-12 at the tip of this change, each editing the
shipped source, running ONE guard test and reverting, with `__pycache__` cleared and 1 s
between cases. Each patch anchor was asserted to occur exactly once before the edit, so a
patch that failed to apply could not be reported as a catch:

| sabotage | guard test | result |
| --- | --- | --- |
| A: grade the wrong span set (shift every block one span downstream) | `test_every_pass_grades_the_right_spans` | CAUGHT (1 failed) |
| A: drop the realised-depth mask | `test_a_slot_is_graded_at_exactly_the_passes_its_depth_reaches` | CAUGHT (1 failed) |
| A: detach the trajectory | `test_the_gradient_reaches_the_core_through_that_pass` | CAUGHT (2 failed) |
| B: shift the head offsets by one | `test_head_j_is_scored_against_the_next_spans_token_j` | CAUGHT (1 failed) |
| B: read a token of the slot's OWN span instead of the cell | `test_the_heads_read_the_slots_last_prefix_cell` | CAUGHT (1 failed) |
| B: drop the validity mask | `test_the_masking_is_exact` | CAUGHT (1 failed) |

### Readout, both arms

Runner `arc/run_slotloop3.sh`, KIND `slot`, SWEEP_CKS 2500,5000, OUTDIR
`/home/wolfe/morph-scratch/arc/results/2026-09-12-strict`.

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → K1−K6, K3−K6, `spandec_ce`.
2. `worth_profile.py --rows 192` (`--modes auto`) → zero / all_slots by offset bin.
   `all_slots == zero` is the strict geometry's CHECK, not a result.
3. `slot_z_optimize.py` → `ce_entry − ce_loop` against the fitted-z ceiling.
4. Paired CE against `slot-spandec-strict`'s sweep npz with
   `lab/divergence/span_budget_profile.py --full A.npz --span B.npz`, 480 rows.
5. Arm A only: `tul/spandec_pass_t{t}`, one series per pass — **read this first**. If it
   does not FALL with `t`, a deeper pass is not making a better plan and no K-curve reading
   means anything.
6. Arm B only: `tul/coda_span_ce` and `tul/coda_span_n_tokens` (the far heads of J 32 are
   supervised on a minority of slots; the packed mean span is ~20 tokens).
7. Wall clock, `loop/core_gain_t0`, the sustained tripwire.

## Predictions (frozen)

Written before any GPU step of either arm. Probabilities are the builder's.

- **P-1 (per-pass gives the passes a non-zero contribution).** `slot-spandec-strict-perpass`
  token **K1−K6 above 0.005**. **45 %.** Reasoning: this is the first unblinded objective
  that makes pass `t`'s job differ from pass `t−1`'s, and `prev-reach1` proved the loop CAN
  carry state across passes when something forces it to. Against it: E7 says one pass sets
  the scale and the rest rotate, and a target the loop cannot reach is met by the DECODER
  getting better at the near tokens of every block rather than by the state changing.
  Residual: 40 % it lands in [0.000, 0.005] like every unblinded arm before it, 15 %
  negative.
- **P-2 (the per-pass CE falls with the pass index).** `tul/spandec_pass_t6` <
  `tul/spandec_pass_t1` by more than **0.05** nats at 5,000. **75 %.** Reasoning: pass 6's
  target is strictly harder (six spans against one), so a FALL is a real claim, not a
  tautology — a flat or rising curve says the extra passes add nothing the decoder can use.
  This is the honesty instrument; if it fails, P-1 is unreadable.
- **P-3 (per-pass CE against strict).** `slot-spandec-strict-perpass` depth-6 CE, token
  paired on 480 rows, is **WORSE than `slot-spandec-strict` by 0.00 to 0.05** nats. **55 %.**
  Reasoning: h3 (a 3× decoder target) was 0.0093 BETTER, so a bigger target is not
  automatically a tax; but this one also puts gradient into every pass, and the recipe has
  never optimised that. Residual: 25 % it is BETTER than strict, 20 % worse by more than
  0.05.
- **P-4 (per-pass rate).** `slot-spandec-strict-perpass` clears the 8,086 tok/s floor at
  step 200: **35 %.** Reasoning: 35.7 block-passes per token against h3's 22.7 and strict's
  14.7, and the span-decoder smokes read 10,829–10,888 at 14.7. Simple proportion puts this
  arm near 4,500. **This is the arm most likely to be skipped by the rate rule, and that is
  a design outcome, not a failure.**
- **P-5 (codaspan gives the passes a non-zero contribution).** `slot-spandec-strict-codaspan`
  token **K1−K6 above 0.005**. **20 %.** Reasoning: the heads change what the CODA must do,
  not what a PASS must do. Every pass still faces the same exit target. The mechanism by
  which it could move the loop is indirect — a harder read makes the cell's content more
  valuable, so the write gets more gradient — and indirect pressure is exactly what the
  eleven flat arms already applied. Residual: 70 % in [0.000, 0.005], 10 % negative.
- **P-6 (codaspan moves the WRITE, which is the outcome it is really for).**
  `slot_z_optimize`'s `ce_entry − ce_loop` on `slot-spandec-strict-codaspan` exceeds
  `slot-spandec-strict`'s by more than **0.01** nats. **50 %.** Reasoning: the heads read
  the prefix cell directly and a parallel span target is the largest demand ever put on
  that cell; on strict the only source is the write. Against it: the loop's write has moved
  by less than 0.006 over its entry state on every arm to date.
- **P-7 (codaspan CE against strict).** `slot-spandec-strict-codaspan` depth-6 CE is within
  **0.02** nats of `slot-spandec-strict` (either sign). **60 %.** Reasoning: the heads are
  identity-init on the coda readout and the token CE is untouched, so the only cost is
  gradient competition on the coda. Residual: 30 % it is worse by more than 0.02, 10 %
  better by more than 0.02.
- **P-8 (J 32 is not better than J 8).** `slot-spandec-strict-codaspan32`'s depth-6 CE is
  NOT better than `slot-spandec-strict-codaspan`'s by more than 0.01. **70 %.** Reasoning:
  the packed mean span is ~20 tokens, so heads 21..32 are supervised on a minority of slots
  and the far offsets of a span are the part a parallel readout can say least about
  (the budget's flat 0.315 is what CONTEXT buys, not what one state predicts).
- **P-9 (survival).** All three arms reach 5,000 with no sustained tripwire
  (`preclip/total > 1e4` at step ≥ 200). **80 %.** Reasoning: the 1,000-step ramp plus
  `core_fixed_point_lambda` 1.0 has held every arm in this family, and neither arm changes
  the map. Arm A adds gradient at every pass, which is the one place I would expect a
  surprise.
- **P-10 (codaspan rate).** `-codaspan` clears the floor: **85 %** (no block pass, half the
  main CE's readout). `-codaspan32`: **60 %** (twice the main CE's readout).

## Binding

If P-1 holds and P-2 holds — the per-pass CE falls AND the token K-curve moves without
blindness — then the objective, not the geometry, is the lever, and the next arm is the
per-pass target at a deeper draw with the reach limit OFF.

If P-2 holds and P-1 fails — the passes descend their own targets and the coda is unmoved —
the reading is the one `oracle_z` was built to give and gives it for a target the coda
actually shares: the passes CAN be driven, and where they go is not what the coda needs.
That closes the "no objective has asked" lane and points at the READ (what the coda does
with z), which is Arm B's half of this panel.

If P-6 holds and P-5 fails — the write gets more valuable and the passes still read zero —
then the loop's value is entirely in its first pass, and the honest next step is to price a
ONE-pass slot model against the looped one at matched compute rather than to keep buying
passes.

## Not verified before launch

* **No GPU step of either arm.** The 5090 ran the arc queue for the whole build window.
  Every cost number above is arithmetic; the 12-step smokes are handed back unrun.
* **Nothing says either arm TRAINS.** The gates prove the terms are correct, additive and
  RNG-neutral at init on a tiny CPU model, and that each config composes, builds and runs
  one forward and backward. The 5,000-step conjunction is unmeasured.
* **The per-pass reduction (equal weight per pass) is a CHOICE, not a measurement.** No arm
  has compared it against a token-weighted mean.
* **`spandec_pass_tokens: 8` is inherited reasoning**, not tuned: it is the offset at which
  the measured cross-span budget stops being front-loaded, and the `oracle_z_max_tokens` /
  `egrad_max_tokens` precedent.
* **`spandec_pass_weight` and `coda_span_weight` are both 1.0 and untuned.** Arm A's term
  is a mean over six CEs and therefore comparable in magnitude with the exit term; arm B's
  is one CE. Neither weight has been swept.
* **`coda_span_source: "token"` is implemented and tested for WHERE it reads, and has never
  been run.** Its causality consequence (the boundary token's state has not seen its own
  slot's z) is reasoned from the coda's causal mask, not measured.
* **The per-pass term's memory is unpriced on the real shapes.** It runs up to `T` chunked
  `[B·S·H_t·J1, d]` readouts per step at B 6, S 64; each `fused_linear_cross_entropy` call
  allocates and saves a `[V, d]` fp32 `grad_w` (201 MB at V 49169, d 1024) for the backward,
  and six live calls is ~1.2 GB on a card with ~1 GB of slack. **This is the most likely way
  arm A dies, and it will show up as an OOM in the smoke, not as a slow step.**
* **`_tul_layer_passes`' per-pass count** uses `depths.max()` for the batch, which is the
  right number for the cost actually paid but has not been checked against a profiler.

## Amendment 1 (2026-09-12, before any GPU step of either arm): how P-2 may be READ

An outside review of the record (pasted by Wolfe, 2026-09-12 ~14:40 CDT) found two holes
in the per-pass arm's reading. Both are accepted. The predictions above stay exactly as
written; this amendment changes what they may be taken to MEAN and adds the instrument
that gives the honest reading.

1. **P-2 is not a progress reading.** `spandec_pass_t6` and `spandec_pass_t1` are scored on
   DIFFERENT token sets (six spans against one). A fall can come from an easier mixture of
   the added tokens and a rise can hide better predictions on every shared token. P-2 is
   scored as written, but it no longer gates P-1's readability. The decisive reading is the
   depth-by-horizon grid: the state after forced depth t in {1, 2, 3, 6}, scored by the shared
   decoder against IDENTICAL targets per column — (a) the same H=1 8-token target on
   `pos_pass`, (b) the same H=6 48-token target on `pos_pass`, (c) the exit target — with the
   eligibility mask fixed across depths within a column
   (`lab/divergence/spandec_horizon_grid.py`, being built; the strict ruler runs column (c)
   as the control). **The progressive-planning interpretation holds only if a deeper state
   scores better on the SAME tokens; an improvement that appears only when the scored target
   changes with t fails it.** No numeric prediction is added after the fact.
2. **Teacher forcing across spans weakens the target.** At pass t the decoder reads the TRUE
   tokens of spans s+1..s+t-1 before decoding span s+t, so pass 2 is not required to build on
   pass 1's forecast of span s+1; it may read it. The arm still asks z_t to carry what those
   true tokens do not give about span s+t, which is a real demand, but it is weaker than
   "extend the plan". Recorded so the binding is not over-read: a positive P-1 under this
   target says the passes can be driven by a growing target, not that they plan.

Neither point changes the queue line or the config.

## Amendment 2 (2026-09-12, after the smoke, before any 5,000-step draw): arm A OOMed, and the term's memory is the DECODER, not the CE

"Not verified before launch" called this: *"The per-pass term's memory is unpriced on the
real shapes ... six live calls is ~1.2 GB on a card with ~1 GB of slack. This is the most
likely way arm A dies, and it will show up as an OOM in the smoke, not as a slow step."*
It died exactly that way, and the priced cause was the wrong one — off by 10x.

**What happened.** The 12-step smoke of `slot-spandec-strict-perpass` reached step 0 at
`peak=25.50GB` (every other strict arm reads 12.9-13.1 GB there) and then raised
`torch.OutOfMemoryError` inside the MAIN token CE on step 1, with 26.99 GiB in use on a
31.4 GB card that a three-monitor desktop already holds ~6 GB of
(`/home/wolfe/morph-scratch/arc/smoke-slot-spandec-strict-perpass/run.log`). No 5,000-step
draw of arm A was started. Arms B (`-codaspan`, 13.13 GB) and B32 are unaffected.

**Where the memory went, measured.** The 5090 was running the arc queue, so the arm was
re-priced on the DGX Spark (GB10, 118 GB unified) at the panel budget, seq 1024, batch 6,
64 cells, `pass_horizon_max` 6, `pass_tokens` 8, 3 steps each:

| run | peak alloc, step 0 | peak alloc, step 20 |
| --- | --- | --- |
| `tul_slot_spandec_strict` (the ruler) | 12.95 GB | 14.88 GB |
| `..._perpass`, as filed | 25.55 GB | 27.48 GB |
| `..._perpass`, `spandec_pass_horizon_max=1` | 17.20 GB | — |
| `..._perpass`, `spandec_pass_tokens=4` | 20.04 GB | — |

The Spark reproduces the 5090's step-0 figures to 0.05 GB, and its step-20 27.48 GB is
what the 5090 had no room for. The three per-pass rows fit `delta = a·P + b·T` exactly
(`P` = decoded positions, `T` = 8 passes): **a = 111.3 KB per decoded position**, and the
fixed part 1.58 GB is the `T` fused-CE `[V, d]` fp32 `grad_w` accumulators (8 x 0.202 =
1.62 GB). So the accumulators the config comment priced are **1.6 of the 12.6 GB**; the
other 11.0 GB is the SPAN DECODER's own saved activations over 101,376 decoded positions.

A `torch.cuda.memory._record_memory_history` replay of the term alone at those shapes
(12.87 GB, within 0.3 GB of the in-trainer delta) attributes the peak live set as:

| site | GB |
| --- | --- |
| `attention.py:140-141` `RMSNorm.forward` — the fp32 upcast, 4 norms per position | 5.64 |
| `tul_spandec.py:231-232` the SwiGLU (`gate_up`, `silu(g)*u`, `down`) | 3.09 |
| `tul_spandec.py:223` `qkv` | 1.16 |
| `fused_ce.py:95` the 8 `grad_w` accumulators | 1.50 |
| `tul_spandec.py:228` `proj` | 0.39 |
| `tul_spandec.py:339` the embedding / `z_in` / `tok_in` / concat chain | 0.22 |
| `fused_ce.py:94` the 8 `grad_x` buffers | 0.19 |

**The fix, and it is not an objective change.** `SpanDecoder.decode` is now called through
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` inside
`_tul_spandec_per_pass_loss` (`MORPHTransformer._spandec_pass_decode`), so the decoder's
activations are recomputed in backward one pass at a time. The fused CE is deliberately
LEFT OUT of the recompute region: checkpointing the decode alone takes the isolated term
from 12.87 to 3.78 GB for one extra 2-block decoder forward (~4.4 TFLOP/step), while
pulling the CE in as well saves a further 0.76 GB and costs a second pass over the vocab
(~33 TFLOP/step, more than the model). Nothing about the target moved: the same spans, the
same `pass_tokens`, the same realised-depth mask, the same one-CE-per-pass reduction with
equal weight per pass, the same live trajectory, the same `spandec_pass_t{t}` series.
`tests/test_tul_spandec_per_pass.py` grew from 22 tests to 24 — one asserts the decode is
actually checkpointed (one region per pass, so the equivalence test cannot pass vacuously),
one asserts the loss and EVERY parameter gradient (core and span decoder named explicitly)
match the un-checkpointed forward to 1e-6 on the CPU fp32 fixture. Two-sided sabotage,
2026-09-12: a dropout planted inside the recompute region passes with the default
`preserve_rng_state=True` and FAILS with `preserve_rng_state=False`, so the test does see
a forward/recompute mismatch.

**The new cost, measured on the same GB10, same budget, 21 steps:**

| run | peak, step 0 | peak, step 20 | tok/s, step 20 |
| --- | --- | --- | --- |
| `strict` (ruler) | 12.95 GB | 14.88 GB | 625 |
| `perpass`, as filed | 25.55 GB | 27.48 GB | 304 |
| `perpass`, checkpointed | 15.11 GB | **17.03 GB** | 278 |

17.03 GB against the ruler's 14.88 leaves the whole term at 2.15 GB, and 17.03 GB fits
beside a 6 GB desktop on the 31.4 GB card. The GB10 was sharing its GPU with a
depth-isolation probe throughout, so **those tok/s are not the 5090's and rank nothing**;
the fix's own throughput cost reads ~8 % there (278 against 304) and is unmeasured on the
5090.

**No prediction changed.** P-1 through P-10 stand exactly as frozen, including P-4's
8,086 tok/s floor, which is still what decides whether the arm runs. The recompute makes
the arm ~8 % slower on the one GPU that could price it, so if anything P-4 got harder, and
it is scored on the 5090 smoke as written. The only thing this amendment retires is the
config comment's claim that the T separate CE calls are what the arm's memory buys: they
are an eighth of it, and the reading they preserve is now paid for at 2.15 GB.

## Results

Three arms on the 5090 at `d48ad3a` (per-pass, decode checkpointed) and `18f7b2d`/`397bdf3`
(coda heads), seed 1, 5,000 steps, seq 1024, batch 6, runner `run_slotloop3.sh`. Readouts in
[`../results/2026-09-12-strict/`](../results/2026-09-12-strict/): `sweep_<arm>_{2500,5000}.json`,
`worth_<arm>_5000.json`, `paired_gaps_5000.txt`, `spandec_pass_series_slot-spandec-strict-perpass.txt`,
`coda_span_series_slot-spandec-strict-codaspan.txt`. Every CE gap is token-paired on 501,106
tokens over the same 480 validation rows; K is the forced-depth sweep's token CE difference
with its paired bootstrap CI over rows.

| reading | `-perpass` | `-codaspan` (J 8) | `-codaspan32` | ruler `slot-spandec-strict` |
| --- | --- | --- | --- | --- |
| rate at step 200 (tok/s) | 4,736 | 11,135 | 10,403 | ~10,850 |
| wall clock, 5,000 steps | 129 min (2.31×) | 58 min (1.04×) | 68 min (1.24×) | 55 min |
| survival, `preclip/total` max | 33.8 @224 | 118 @338 | 56.3 @230 | 22.1 @1310 |
| tokens K1−K6 @5000 | **+0.0008** [+0.0006, +0.0010] | **+0.0012** [+0.0008, +0.0015] | **+0.0017** [+0.0015, +0.0020] | +0.0016 [+0.0013, +0.0019] |
| tokens K3−K6 | −0.0000 | −0.0002 | +0.0001 | +0.0002 |
| `spandec_ce` K1−K6 | +0.0013 | +0.0040 | +0.0041 | +0.0031 |
| depth-6 CE | 4.3646 | 4.3889 | 4.3729 | 4.3474 |
| CE vs ruler, depth 6 | **+0.0172** [+0.0150, +0.0195] | **+0.0415** [+0.0387, +0.0439] | **+0.0255** [+0.0228, +0.0280] | — |
| worth zero / shuffle | 0.1826 / 0.1671 | 0.1721 / 0.1688 | 0.1765 / 0.1679 | 0.1865 / 0.1739 |

`-codaspan32` vs `-codaspan`: **−0.0160** [−0.0183, −0.0138], every offset.

**Arm A's own ladder** (`tul/spandec_pass_t{t}`, steps 4500–5000): t1 4.513, t2 4.614,
t3 4.646, … t6 4.706 — the per-pass CE RISES with the pass index at every checkpoint of the
run (t1 6.573 → t3 6.650 at steps 200–600). Pass t's target is t spans of 8 tokens each,
so a rise is what a state that carries nothing new past pass 1 produces; amendment 1
already said this column cannot be read as progress. The identical-target reading
(`spandec_horizon_grid.py`, forced depths 1,2,3,6 × columns h1/h6/exit on the same targets)
is queued on the Spark (chain C) and is the binding reading of P-2.

**The identical-target grid, FIRST RUN — INVALID** (`spandec_horizon_grid.py` at `0c1f042`,
DGX Spark, 480 rows, forced depths 1/2/3/6;
`../results/2026-09-12-instruments/horizon_grid_slot-spandec-strict-perpass.{json,txt}`).
**Do not read this table.** The instrument rebuilt the front with a bare
`_tul_front(inp, layout)`, so on the strict geometry the prelude ran UNRESTRICTED and
every state below is off-distribution; the strict ruler's exit column read 4.65 with
depth making it worse (−0.022) where the shipped forward reads 4.48 and +0.003. Fixed at
`7a24adf` (one home, `_tul_tg_kwargs`; the grid's test gains a strict twin that the bare
front fails). The corrected grids (chain D) replace this table when they land.

| target column | d1 | d2 | d3 | d6 | K1−K6 | K3−K6 |
| --- | --- | --- | --- | --- | --- | --- |
| pass_h1 (next span, 8 tokens) | 4.7543 | 4.7670 | 4.7708 | 4.7742 | **−0.0200** [−0.0220, −0.0181] | −0.0034 |
| pass_h6 (next six spans, 48 tokens) | 4.7564 | 4.7607 | 4.7618 | 4.7640 | **−0.0075** [−0.0082, −0.0068] | −0.0022 |
| exit (the span decoder) | 4.7029 | 4.7094 | 4.7110 | 4.7131 | **−0.0102** [−0.0112, −0.0093] | −0.0021 |

(Reading withdrawn with the table above; see the corrected grid below when filed.)

**Arm B's heads** read 6.62 nats/token at the tail (7.46 at steps 200–600), against the
coda's 4.39 on the same tokens: a parallel readout of a whole span from one cell is 2.2
nats behind the causal read, at every J.

### Predictions scored

| P | claim | result |
| --- | --- | --- |
| P-1 | per-pass tokens K1−K6 > 0.005 | **FALSE** (0.0008) |
| P-2 | `spandec_pass_t6` < `t1` by > 0.05 | **FALSE** by the letter (rises 0.19); the binding identical-target reading is being RE-RUN after the grid's bare-front bug (chain D) |
| P-3 | per-pass worse than ruler by 0.00–0.05 | **TRUE** (+0.0172) |
| P-4 | per-pass clears 8,086 tok/s | **FALSE** (4,736), the 65 % case; the floor was OFF on this runner so the arm ran |
| P-5 | codaspan tokens K1−K6 > 0.005 | **FALSE** (0.0012) |
| P-6 | codaspan `ce_entry − ce_loop` exceeds the ruler's by > 0.01 | **PENDING** (`slot_z_optimize` on codaspan, strict, perpass chained after the math panel and the energy probes) |
| P-7 | codaspan within 0.02 of ruler | **FALSE** (+0.0415 worse) |
| P-8 | J 32 not better than J 8 by > 0.01 | **FALSE** (J 32 better by 0.016) |
| P-9 | all three survive | **TRUE** |
| P-10 | codaspan / codaspan32 clear the floor | **TRUE / TRUE** (11,135 / 10,403) |

## Verdict

**Failure** on the hypothesis: P-1 and P-5 false, so neither a per-pass planning target nor
a parallel read from the coda moves the passes; the per-pass arm's own ladder rises with the
pass index. The binding case that fired is "P-2 fails and P-1 fails" in its weak form: the
passes were handed a target that differs by pass and the state after pass 6 is no better
at pass 1's job than the state after pass 1 (the exit column, K1−K6 0.0008). Both arms cost
CE (+0.017, +0.042) for their gradient competition on the shared coda and decoder. P-8's
failure says the heads at far offsets do get supervision worth 0.016 nats, which is a
statement about the coda's readout, not the loop. P-6 and the identical-target grid remain
open and are filed when they land.

## Updated hypothesis

The objective lane is closed alongside geometry (strict panel) and what trains the core
(core-token panel): staged, oracle, gradpass, per-pass horizon, critic and parallel-decode
targets all leave the passes past the first at ≤ 0.002 nats. The only reading on the whole
tree where passes 2..6 carry anything is `prev-reach1` (0.0127), a relay forced by cutting
the coda's reach, and it costs 0.011 nats of CE. Next: the READ (what the coda takes from a
cell; planning-cells design, held for Wolfe's word) and a matched-compute one-pass slot
model.
