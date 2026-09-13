# Agent Note: the Thought Register — M mutable cells per span

Status: proposed

## Problem

A span's looped state is ONE vector, and a row's slot states are near copies of each
other. Measured:

| reading | value | source |
| --- | --- | --- |
| `val/slot_eff_rank`, strict ruler | **5.7598** in 1024 dims | `run_slot-spandec-strict.log` |
| `val/slot_pairwise_cos`, same run | **0.7104** | same |
| the same pair, slot family | 5.7 – 7.3 / 0.72 – 0.77 | `lab/experiments/successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md` |
| geometry audit | 1.7 – 4.8 | 2026-09-10 audit |
| 4,906 slots at step 0 | 18 – 23 | `results/2026-09-12-latent-z-gradient/step0-*.json` |

Fifty to sixty-four states, each 1024-dimensional, spanning about six directions. The coda
sees six dimensions of variation whatever the row holds, and the loop iterates a state
that is nearly the same vector everywhere. Under that condition "pass 2 relates pass 1's
result to something" has no referent.

Twelve arms read token K1−K6 inside [−0.0001, +0.0033] (the strict ruler +0.0016, K3−K6
+0.0002) across every lever tried — core shape, stability terms, entry, seed, width, HCA,
geometry, target. **None of them changed how many states a span holds.** That is the one
structural fact every flat arm shares, and it has never been varied.

## Proposal

`tul.slot_cells: M` (default 1 = today's forward, bit-identical). Per span, M mutable
cells.

* **Seed.** M learned queries cross-attend (single head, softmax over the span's OWN
  prelude token states, causal to the boundary) and pool M different vectors; each gets
  `E_slot` and a per-cell embedding. `W_o` is zero-init and `P_cell` is zeros, so the
  register's term is **exactly 0 at step 0** and every cell's seed reduces to today's
  `slot_seed: boundary` value. The arm starts AT its ruler.
* **Loop.** The compact sequence in `_tul_core` is S·M cells at ONE shared per-slot depth.
  Cell i of slot k reads every cell of slots < k (per `loop_reach`, as today) AND every
  cell of its own slot k, LATER siblings included — full within the slot, causal across
  slots. That relation is a SUPERSET of flattened causal, so it cannot be delivered as
  `tg_allow` / `tg_comp_allow`, which are ANDed into an already-causal relation and only
  ever narrow. It travels as `tg_relation`, one explicit attention kwarg that REPLACES a
  branch's causal term, honoured by both halves of a TG-restricted layer and wired only
  here and in the think-once stack (see Risks).
* **Exit.** Cell i is written 1:1 into prefix cell i through the shared `W_prefix[i]`, so
  `prefix_k` must equal `slot_cells` (it raises otherwise). The coda is UNCHANGED.
* **Target.** The span decoder grades the MEAN of the M cells.

Arms: `slot-register-m4`, `slot-register-m4-sameinit` (one shared query — the control that
separates capacity from pull-apart), `slot-register-m8`.

Instruments: `val/slot_eff_rank` / `val/slot_pairwise_cos` over all 64·M cells of a row,
and NEW `val/slot_cell_eff_rank` / `val/slot_cell_pairwise_cos` WITHIN a slot across its M
cells. The within-slot pair is the number the arm exists to move; it reads 1.2383 / 0.8443
on the CPU fixture at init.

## Alternatives considered

* **A wider prefix, `prefix_k: 4`, with one looped state.** Already run
  (`slot-mux-prefix4`, 2026-09-10): `val/slot_eff_rank` 6.3389 → 7.12, and the loop did not
  move. It widens the WRITE, not the state the loop iterates. Rejected as insufficient,
  but it is the reason `m4` vs `m4-sameinit` — not `m4` vs the k=2 ruler — is the clean
  pair in the panel.
* **An orthogonality or decorrelation penalty on the slot states.** Cheaper and it targets
  the measured symptom directly. Rejected for now because it fights the loop for the same
  state rather than giving the loop more state, and because the 2026-08 campaign already
  showed that penalising a geometry the map produces does not change what the map does
  (four spectral interventions, two worse than nothing). It stays available if the
  register's cells collapse anyway — which is exactly what prediction P-1 failing would
  mean.
* **Per-cell span-decoder targets (cell i decodes token bucket i of the next span).** A
  stronger separating force than the mean. Rejected for THIS arm because it changes the
  target at the same time as the state, making the result unattributable. It is the named
  follow-up if the cells stay apart and the K-curve still reads flat.
* **A span decoder that cross-attends to the M cells as a memory.** Considered and
  written down as the alternative to grading the mean. Rejected for this arm on the same
  one-factor grounds: it is a second mechanism. The mean's gradient still reaches every
  cell. **BUILT as its own arm, 2026-09-13** (`tul.spandec_reads_cells`, arm
  `slot-register-m4-reader`, one factor against `slot-register-m4`) —
  [`2026-09-13-register-reader-and-downstream-target.md`](2026-09-13-register-reader-and-downstream-target.md).
  It is zero-init, so the two arms are bit-identical at step 0 and the pair is clean; it
  is UNRUN, and it must be read beside `slot-register-m4`, which is also unrun.
* **More `prefix_k` with the trajectory write** (`prefix_source: trajectory`, the sibling
  build). It gives the coda more cells to read WITHOUT giving the loop more state, and the
  information identity says those cells carry no more than the entry does. Complementary,
  not a substitute.

## Acceptance criteria

1. `slot_cells: 1` builds no module, draws no RNG, and is byte-identical to the pre-work
   tree. **Met**: proved by running `d778845` and this tree on four fixtures and comparing
   loss, logit sum, grad sum and `state_dict` key count.
2. The register's term is exactly 0 at init, and `distinct`/`same` are identical until
   `W_o` moves. **Met**, guarded.
3. Cell i lands in prefix cell i; a slot's M cells loop at one depth; the in-loop relation
   is own-slot-full and cross-slot-causal, two-sided — on the MASK and on the FORWARD.
   **Met**, guarded. The forward half is numerical: the loss differs from a
   flattened-causal build (9.8953599930 vs 9.8935718536 on the CPU fixture) and equals an
   independently written all-true-within-slot build (9.8953599930), each branch of the
   layer is separately load-bearing, and perturbing cell 3 of a slot moves cell 0 of that
   slot while moving no earlier slot at all.
4. No parameter reads a NaN gradient, and a pad slot's term is exactly 0. **Met**, guarded
   (this was a real bug — see Risks).
5. `val/slot_cell_eff_rank` above 2.0 (of 4) at 5,000 steps on `slot-register-m4`. **Not
   met — unrun.** This is the arm's reason to exist.
6. Token K1−K6 above 0.005 against the ruler's +0.0016. **Not met — unrun.**

Prereg: `lab/experiments/planned/2026-09-13-arc-thought-register.md`.

## Risks

* **The cells may collapse onto each other anyway.** The loop and the shared write are
  exactly the forces that produced the rank collapse, and they now act on cells that can
  read each other. If P-1 fails, the mechanism is refuted and the lever moves to a write
  that keeps cells apart.
* **The objective still grades one state per slot.** The mean is one target per span, so
  the gradient reaching the cells is no richer than today's. Capacity may be necessary and
  not sufficient.
* **Two factors against the k=2 ruler.** `prefix_k` must equal `slot_cells`, so `m4`
  differs from `slot-spandec-strict` by the register AND the prefix width, and the packer
  turns the unused slot budget into tokens (`L_total` 1280 vs 1152). Only `m4` vs
  `sameinit` is clean. Named in every row, not removed.
* **The documented relation did not execute, and was fixed before launch (2026-09-13).**
  The mask was right; the DELIVERY threw half of it away. `blk[p][q] = slot(p) >=
  slot(q)` is a superset of flattened causal and `tg_allow` / `tg_comp_allow` can only
  narrow, so at `loop_reach 0` the arm executed plain flattened causal — cell 0 blind to
  its siblings, which is the opposite of a register — in the loop AND in the think-once
  stack (`blk` 9.8935718536 = all-true 9.8935718536 = flattened causal 9.8935718536 on
  the CPU fixture). Every mask-level test passed the whole time, which is why the gate now
  grades the FORWARD two-sided and per branch. The fix is `tg_relation`
  (`morph/model/attention.py::_tg_relation_guard`), wired only on the register's cell
  axis; `slot_cells: 1`, the token path, the prelude and the coda are bit-identical on six
  fixtures. Not fixed, by decision: the CCA conv, the `W_v_prev` value shift and a GLA
  retention carry stay causal on the flattened cell axis — position-wise or recurrent
  operators with no mask to widen, and everything they reach the relation already allows.
* **Two real defects were found while building it, both now guarded.** (a) A tail-pad slot
  has no own-span token, so its softmax row was all `-inf`; the NaN is created INSIDE the
  softmax and propagates through the BACKWARD, and `nan_to_num` on the output does not
  stop it — `Q` and `W_k` read `grad = nan`. (b) `nn.Linear` kaiming-draws on the GLOBAL
  RNG stream before the weight is overwritten, three Linears, three draws, so anything
  constructed after the register would sit at a different point in the stream than on the
  ruler. That one was found only because sabotage D7 MISSED.
* **Eager-only core attention.** The S·M `tg_allow` tensors force the eager path on the
  core stage, which the ruler avoids under `tg_scoped_kernels`. A second difference from
  the ruler, sized only by the smoke.
* **Nothing has run on a GPU.** No smoke, no wall clock, no memory figure. `slot_cells`
  has never run at the real `d_model`.
