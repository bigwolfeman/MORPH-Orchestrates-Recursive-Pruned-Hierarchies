# Planned: the Thought Register — M mutable cells per span instead of one

Status: planned

Date: 2026-09-13 (frozen before any GPU step of any arm; the 5090 is running the math
panel and no smoke of any arm here exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). One-factor
partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).
Design note:
[`2026-09-13-the-thought-register.md`](../../../.agents/notes/proposed/architecture/2026-09-13-the-thought-register.md).

## The measured defect

A row's written slot states are near copies of each other.

| reading | value | source |
| --- | --- | --- |
| `val/slot_eff_rank` on the one-factor partner | **5.7598** in 1024 dims | `run_slot-spandec-strict.log`, last val |
| `val/slot_pairwise_cos` on the same run | **0.7104** | same |
| the same pair across the slot family | 5.7 – 7.3 / 0.72 – 0.77 | `2026-09-10-arc-slot-mux-prefix4-norm-match.md` (ruler 6.3389 / 0.7388, prefix-4 7.12) |
| the geometry audit | 1.7 – 4.8 | `2026-09-10` audit |
| across 4,906 slots at step 0 | 18 – 23 | `results/2026-09-12-latent-z-gradient/step0-*.json`, `geometry.eff_rank` |

Fifty to sixty-four slot states, each 1024-dimensional, spanning about **six** directions.
The coda therefore sees six dimensions of variation however many slots a row has, and the
loop iterates a state that is nearly the same vector at every position. That is the
condition under which "pass 2 relates pass 1's result to something" has no referent:
there is nothing to relate it TO.

Twelve arms have now read token K1−K6 inside [−0.0001, +0.0033] (the partner: **+0.0016
[+0.0013, +0.0019]**, K3−K6 **+0.0002**), across every lever the 2026-09-11 batch tried —
core shape, stability terms, entry, seed, width, HCA, geometry. **None of them changed the
number of states a span holds.** That is the one structural fact every flat arm shares.

The 2026-09-13 second-opinion review reaches the same place from theory: a single vector
cannot support iterative relational computation, and the fix is M mutable cells per span.
Wolfe endorses that one point, and this is the arm.

## Question

If a span holds M mutable cells instead of one — seeded apart, looped together, and read
individually by the coda — do the passes become worth anything, and does the slot channel
carry more?

## Hypothesis

The flat K-curve is a CAPACITY-AND-GEOMETRY fact, not an objective fact. One vector per
span is a bottleneck so tight that the state has nowhere to move that the coda can read,
so the map degenerates into "write the exit, ignore the passes". Give the span four cells
that start out looking at different parts of the span and that can read each other inside
the loop, and the loop has an internal relation to compute. If that is right, the
within-slot rank rises off 1, the row rank rises off ~6, and the K-curve moves for the
first time in twelve arms.

The control that has to run beside it is `sameinit`: M cells all pooled from the SAME
query, differing only by `P_cell`. If `m4` beats `sameinit`, the win is the pull-apart. If
`m4` and `sameinit` land together and both beat `strict`, the win is capacity alone, and
the honest name for the arm is "wider prefix", not "register".

## Method

Three arms, each ONE factor against `slot-spandec-strict` at the matched `prefix_k`,
5,000 steps, seq 1024, batch 6, seed 1, ramp 1,000, `norm_match`,
`core_fixed_point_lambda` 1.0, prune/carve/route off, retention off,
`tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`, `mux_beta: 0`,
`spandec` on at J 32.

| arm | config | one factor |
| --- | --- | --- |
| `slot-register-m4` | `tul_slot_register_m4.yaml` | `slot_cells: 4` (+ `prefix_k: 4`, forced) |
| `slot-register-m4-sameinit` | `tul_slot_register_m4_sameinit.yaml` | `slot_cell_init: same` against `m4` |
| `slot-register-m8` | `tul_slot_register_m8.yaml` | `slot_cells: 8` against `m4` |

**`prefix_k` is not a free factor here.** A register writes cell i into prefix cell i, so
`prefix_k` must equal `slot_cells` and the config raises otherwise. That means `m4`
differs from the k=2 partner by TWO things — the register and the prefix width — and the
prefix width alone is already measured (`slot-mux-prefix4`, 2026-09-10: `slot_eff_rank`
6.3389 → 7.12). The clean one-factor comparison inside this panel is **`m4` against
`m4-sameinit`**, which share `prefix_k` exactly. Both comparisons are reported and the
prefix-width confound is named in every row, not buried.

**The packer confound, stated once.** `L_total` is 1024 + `prefix_k`·64: 1152 at k=2,
1280 at k=4, 1536 at k=8. The packer converts the unused slot budget into tokens, so the
k=4 and k=8 arms see a different number of real tokens per row than the k=2 partner. The
three register arms pair cleanly with EACH OTHER. Against `slot-spandec-strict` the CE
comparison is paired on `tok_index` (`memory_cost_table.py`'s pairing, and the sweep's),
which removes the row-cut difference but not the context-length difference.

### What the mechanism is

Per span, M mutable cells:

* **seed** — M learned queries cross-attend (single head, softmax over the span's OWN
  prelude token states, causal to the boundary) and pool M different vectors; each gets
  `E_slot` and a per-cell embedding `P_cell[i]`. `W_o` is zero-init and `P_cell` is zeros,
  so **the register's term is exactly 0 at step 0** and every cell's seed is today's
  `slot_seed: boundary` value. The arm starts AT its ruler, which is what makes the
  `m4`/`sameinit` pair one factor.
* **loop** — the compact sequence in `_tul_core` is S·M cells at ONE shared per-slot
  depth. Cell i of slot k reads every cell of slots < k (per `loop_reach`, as today) AND
  every cell of its own slot k, LATER siblings included: full within the slot, causal
  across slots. Plain causal on the flattened axis would leave cell 0 permanently blind to
  its siblings, which is the opposite of a register.
* **exit** — cell i is written 1:1 into prefix cell i through the shared `W_prefix[i]`.
  The coda is UNCHANGED: a cell reads itself alone, a token reads the cells of earlier
  slots per `tg_coda_prefix_reach`.
* **the span decoder grades the MEAN of the M cells.** That is a cost decision and it is
  written down as one. Every reader between the core and the write (MUX, spandec, SIGReg,
  the energy) takes one state per slot; handing them the mean keeps each of those
  mechanisms the shipped one, so the arm differs from its ruler by the register alone. A
  decoder that cross-attends to the M cells is a SECOND mechanism and is the follow-up,
  not this arm. The mean's gradient still reaches every cell.

### Method amendment, 2026-09-13 (before any GPU step of any arm; predictions unchanged)

**The documented in-loop relation was not the one that executed, and it was fixed before
launch.** The mask `slot_cell_relation` builds was right. The DELIVERY was not:
`blk[p][q] = slot(p) >= slot(q)` is a SUPERSET of flattened causal, and the attention
kwargs it travelled on — `tg_allow` / `tg_comp_allow` — are ANDed into an already-causal
relation and can only NARROW. So at `loop_reach 0`, which is every arm in this panel, the
register executed plain FLATTENED CAUSAL: cell 0 of a slot never read its later siblings,
in the loop or in the think-once stack. On the 64-dim CPU fixture the three losses were
identical (`blk` 9.8935718536, all-true 9.8935718536, flattened causal 9.8935718536).

The fix adds ONE explicit mechanism, `tg_relation`: a `[*,1,S,S]` bool that REPLACES a
branch's causal term instead of narrowing it, honoured by both halves of a TG-restricted
layer, wired only where the register passes its relation (the loop's core stage on the
cell axis and `_tul_cond_apply`). Replacing is safe across slots because `blk` is
block-causal there; within a slot it widens to all M cells. Measured after the fix on the
same fixture: `blk` 9.8953599930, all-true-within-slot 9.8953599930, flattened causal
9.8935718536, and a nudge to cell 3 of a slot moves cell 0 of that slot by 0.1048612595
where the causal build moves it by exactly 0. `slot_cells: 1`, the token path, the
prelude, the coda and every `tg_*` relation are bit-identical (six fixtures; loss, logit
sum, grad sum and key count to the last printed digit).

**Nothing else changed.** The arms, the config values, the readout and every prediction
below are untouched: they were written against the relation the arm was always documented
to run, and the arm now runs it. The CCA conv, the `W_v_prev` value shift and a GLA
retention carry stay CAUSAL on the flattened cell axis by decision — position-wise or
recurrent operators with no mask to widen, and every position they reach the relation
already allows, so cell 0 gets its sibling context through attention alone.

### The gate, built in this change

`tests/test_tul_slot_register.py`, **46 passed**. It covers: OFF builds no module and
moves no base weight; the register's term is exactly 0 at init; `distinct` and `same` are
identical at init and differ once `W_o` moves; the compact sequence is S·M; a slot's M
cells loop at one depth; cell i lands in prefix cell i; the in-loop relation both ways
(own slot's cells including later ones, earlier slots, never a later slot) with plain
flattened causal rejected as a two-sided control; a pad slot's term is exactly 0 and no
parameter reads a NaN gradient; the strict leak test at M=4; the worth-profile modes and
the depth lever on a register model; the within-slot rank instrument; and every refusal.
The 2026-09-13 amendment above extended it to grade the relation on the FORWARD and not
only on the mask: the loss against a flattened-causal build and against an independently
written all-true-within-slot build, each attention branch separately load-bearing, both
branches handed the relation on every core-stage call, the cell-level perturbation probe
with its causal control, and a pad cell never a key of a valid one. **54 passed.**

OFF-state bit-identity was proved by RUNNING the pre-work tree (`d778845`) and this one on
four fixtures and comparing loss, logit sum, grad sum and `state_dict` key count. All four
match to the last printed digit.

Ten source-level sabotages, each anchored to exactly one occurrence, applied to the
shipped source: **10/10 CAUGHT**, after D7 (the register drawing from the global RNG
stream) MISSED on the first pass and exposed a real leak — `nn.Linear` kaiming-draws on
the global stream before the weight is overwritten, three Linears, three draws. Fixed by
snapshotting and restoring `torch.random` state around `__init__`.

### Readout

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`, paired bootstrap over rows.
2. `worth_profile.py --rows 192 --modes auto` → zero / shuffle / wrong_seed / all_slots by
   offset bin. **On a strict arm `all_slots` equals `zero` by construction**; the partner
   reads **0.1865**. The profile must operate on M cells per slot — covered by the gate.
3. `val/slot_eff_rank` and `val/slot_pairwise_cos` over ALL 64·M cells of a row (the
   existing keys, now over more cells) **and** `val/slot_cell_eff_rank` /
   `val/slot_cell_pairwise_cos` WITHIN a slot, across its M cells (new in this change).
   The within-slot pair is the number the arm exists to move; it reads 1.2383 / 0.8443 on
   the CPU fixture at init, which is the floor a working register has to leave.
4. `slot_depth_isolation.py` — per-pass contribution, per slot.
5. Depth-6 CE paired on 480 rows against `slot-spandec-strict` AND against
   `m4-sameinit`; wall clock and tok/s at step 200 (partner: 3,337 s, 11,759 tok/s).
6. The matched-compute row: `plain-depth1` at 14 block-passes/token is 0.25 nats AHEAD of
   spandec-mask and 0.33 ahead of the mask ruler at 5k. Every slot filing carries it.


### Method amendment, 2026-09-13 (baseline re-read; predictions unchanged)

The strict ruler's `val/slot_eff_rank` 5.7598 / `val/slot_pairwise_cos` 0.7104 quoted
above comes from the trainer's last `[VAL]` line and did NOT reproduce on the saved
checkpoint: the model's own `tul_slot_state_probe`, run by
`lab/divergence/slot_rank_anatomy.py` on the DGX Spark, reads 11.9597 / 0.5635 on the
480-row probe panel and 13.8466 / 0.5201 on the trainer's own val recipe
(`lab/experiments/results/2026-09-13-rank-anatomy/README.md`). The cause is now measured
(`lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`, commit ab7c353): the run
logged its rank from the PRE-7a24adf probe, which rebuilt the front UNRESTRICTED on a
strict model; the checkpoint reproduces 5.7598 / 0.7104 / CE 4.4249 to four decimals under
that bare front and reads 13.85 / 0.520 under the shipped probe on the trainer's val recipe.
So the "rank 5.76" headline of this lane was an inert-instrument number; the real per-row
rank of the ruler is about 13 of ~50 slots (cosine 0.52), and every `val/slot_eff_rank`
logged before 7a24adf by a front-restricted arm (strict, or tg_restrict at scope all) is a
bare-front number. Scoring rule for every rank prediction in this file: the
partner's baseline is the SAME instrument on the SAME rows as the arm (the sweep's 480
rows, per-row median), never the trainer's logged figure. Rank predictions stated as
"vs 5.7598" are scored against that re-read baseline; the predicted DIRECTION and the
thresholds are unchanged. Also recorded from the same instrument: the prelude makes the
rank (seed 2.24 to entry 12.75 per row), six passes leave it at 13.17, and the prefix
write cuts it to 7.16 while doubling the cell count.

## Predictions (frozen)

Written before any GPU step of any arm. Probabilities are the builder's.

- **P-1 (the rank moves, within a slot).** `slot-register-m4`'s `val/slot_cell_eff_rank`
  at 5,000 is **above 2.0** (of a maximum of 4). **75 %.** Reasoning: the M queries are
  independently initialised and `P_cell` is free, so the only force pushing the cells back
  together is the shared loop and the shared write. Against it: the loop is exactly the
  force that collapsed rank in the first place, and it now runs on cells that can read
  each other, which is a collapse mechanism as much as a separation one. Residual: 20 %
  in [1.2, 2.0], 5 % it stays at the init floor.
- **P-2 (the rank moves, across the row).** `slot-register-m4`'s `val/slot_eff_rank` over
  all 64·4 cells is **above 12** at 5,000, against the partner's 5.7598. **50 %.**
  Reasoning: four near-independent directions per slot times a row rank of ~6 is an
  optimistic 24, and the measured prefix-4 arm moved 6.34 → 7.12 for a wider write alone.
  12 is deliberately between those. Residual: 35 % in [7, 12] (the prefix-width effect and
  nothing more), 15 % below 7.
- **P-3 (the passes finally read something).** `slot-register-m4` token **K1−K6 above
  0.005**, against the partner's +0.0016. **35 %.** Reasoning: this is the first arm in
  the whole arc that changes how many states a span holds, and every flat arm shares that
  one fact. Against it: capacity is necessary, not sufficient — the objective still grades
  one state per slot (the mean), so the gradient that reaches the cells is still one
  target per span. Residual: 45 % in [0.000, 0.005], 20 % negative (the register costs the
  loop by giving the write an easier job).
- **P-4 (the per-pass curve).** `slot-register-m4` **K3−K6 above 0.002**. **20 %.**
  Reasoning: every arm that moved K1−K6 at all moved the FIRST pass. A register is the one
  mechanism with a story for the LATER passes (cells relating to each other takes more
  than one step), which is why this is 20 % and not 5 %.
- **P-5 (the pull-apart matters).** `slot-register-m4` beats `slot-register-m4-sameinit`
  on depth-6 CE, token-paired, by **more than 0.01 nats**. **40 %.** Reasoning: `P_cell`
  alone can separate cells that pool the same vector, so `sameinit` is not a null arm —
  it is a slower route to the same place. If the pull-apart is worth nothing, the arm is a
  wider prefix and should be called one. Residual: 45 % within ±0.01, 15 % `sameinit`
  wins.
- **P-6 (CE against the partner).** `slot-register-m4` depth-6 CE, token-paired on 480
  rows, is **better than `slot-spandec-strict` by 0.00 to 0.10** nats. **45 %.**
  Reasoning: it writes twice the prefix width and sees more real tokens per row, and the
  prefix-4 arm already moved CE. Against it: more cells is more parameters to train in
  5,000 steps, and every capacity arm in this arc has paid an early price. Residual: 30 %
  worse than the partner, 25 % better by more than 0.10.
- **P-7 (M = 8 is not simply better than M = 4).** `slot-register-m8` does NOT beat
  `slot-register-m4` on depth-6 CE by more than 0.02 nats. **65 %.** Reasoning: if the
  defect is a hard capacity wall, 8 should beat 4 and this prediction fails; if it is a
  geometry problem, 8 cells collapse the same way 4 do and cost more. This is the
  prediction that separates the two readings.
- **P-8 (rate).** `slot-register-m4` clears **9,000 tok/s** at step 200 against the
  partner's 11,759. **55 %.** Reasoning: the core runs over 256 cells instead of 64, still
  far below the 1152-token coda, plus one cross-attention pool over the row per forward.
  The arithmetic says the core stage roughly quadruples and the core stage is not the
  dominant cost. Against it: the S·M relation tensors are eager-only, which the partner's
  `tg_scoped_kernels` path avoids.
- **P-9 (it fits).** `slot-register-m8` runs 5,000 steps without an OOM on the 5090 at
  batch 6. **60 %.** Reasoning: `L_total` 1536 against 1152 is a 33 % longer coda sequence
  on top of 512 compact cells. This is the most likely way M = 8 dies.

## Binding

If **P-1 and P-3 hold** — the cells stay apart AND the passes read something — the flat
curve was a capacity-and-geometry fact all along, twelve arms were measuring a bottleneck
rather than an objective, and the next arm is the span decoder cross-attending to the M
cells so the gradient stops being one target per span.

If **P-1 holds and P-3 fails** — the cells stay apart and the passes still read zero —
that is the strongest result available against the capacity reading: a span can hold four
distinguishable states and the loop STILL does not use its own depth. The lane left is the
target (what the cells are graded on), and the mean-graded decoder is the first thing to
remove.

If **P-1 fails** — the cells collapse onto each other anyway — the collapse is caused by
the loop and the shared write, not by the seed, and the next arm is a write that keeps the
cells apart (per-cell losses, or an orthogonality term), not more cells.

If **P-5 fails and P-6 holds** — `sameinit` matches `m4` and both beat the partner — the
arm is a wider prefix with extra steps. Say so, rename it, and compare against
`slot-mux-prefix4` rather than claiming a register.

If **P-7 fails** — M = 8 beats M = 4 clearly — the defect is a capacity wall and the
sweep continues upward until it stops paying.

## Not verified before launch

* **No GPU step of any arm.** The 5090 ran the math panel for the whole build window.
  Every cost number above is arithmetic. The smokes are handed back unrun.
* **`slot_cells` has never run at the real `d_model`.** Every measurement in this change
  is on a 64-dimensional CPU fixture. The cross-attention pool is `[B, S, M, L]` logits —
  at B=6, S=64, M=4, L=1280 that is 2.0 M entries per forward in fp32, which is small, but
  it has never been allocated on the card.
* **The wall-clock estimate is arithmetic, not a measurement.** The partner's 3,337 s is
  measured; the register's is not.
* **The span decoder grades the MEAN.** That is a deliberate simplification and it is the
  most likely reason a working register still reads a flat K-curve. It is named in the
  Method, not removed.
* **`P_cell` and the query init std (0.02) are choices, not measurements.** Nothing has
  swept either.
* **The `m4` vs `strict` comparison carries the `prefix_k` confound** described in the
  Method. Only `m4` vs `sameinit` is clean.
* **The register's in-loop relation is eager-only.** The S·M relation tensors force the
  eager attention path on the core stage — `tg_relation` routes the window branch to the
  reference path unconditionally, and the fused window kernel bakes causality in so it
  could not run this relation at all — which the partner avoids under `tg_scoped_kernels`.
  That is a second difference from the partner and the smoke's tok/s decides how big it is.
* **The widened relation has never run at `d_model` 1024 either.** The two-sided losses
  and the perturbation probe are on the 64-dim CPU fixture. The argument (an AND into
  causality cannot widen, a replacement can) is shape-independent; the numbers are not,
  and no GPU step of any arm has run.
* **`test_scse.py::test_real_model_loop_is_source_free_and_anchored` was not re-run on a
  free card.** It fails in the full suite with a CUDA OOM caused by the concurrent
  trainer, in a code branch this change does not touch.

## Results

(to be filled after the runs; predictions above are frozen)
