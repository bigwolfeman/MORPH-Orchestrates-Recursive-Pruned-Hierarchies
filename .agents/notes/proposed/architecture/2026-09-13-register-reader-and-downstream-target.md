# Agent Note: the register's reader, and a target that is not the next span

Status: proposed

## Problem

Two separate things in the slot-loop stack have never been varied, and both are about the
OBJECTIVE rather than the state.

**The register grades the weakest reader of its own cells.** `tul.slot_cells: M` gives a
span M mutable looped cells and then hands every reader between the loop and the coda's
write `_reg_cells.mean(dim=2)`: one vector, one target per span. The register note
([`2026-09-13-the-thought-register.md`](2026-09-13-the-thought-register.md)) makes that a
deliberate cost decision, lists the cross-attending decoder in its own Alternatives, and
names the mean in its Risks as the most likely reason a working register still reads a flat
K-curve: "The objective still grades one state per slot ... Capacity may be necessary and
not sufficient." Its prereg's binding is explicit — if the cells separate and the passes
still read zero, "the lane left is the target, and the mean-graded decoder is the first
thing to remove".

**Nothing has ever asked z for a span that is not the next one.** The 2026-08-14 JEPA
screen said the informative target sits two to three spans downstream. Ten targets later,
every span-decoder arm still grades z on span s+1 — which is also the span the coda
predicts best unaided. The measured cross-span budget
([`lab/experiments/failures/2026-09-11-arc-span-budget.md`](../../../../lab/experiments/failures/2026-09-11-arc-span-budget.md),
491,520 paired tokens) is a **0.958**-nat spike at a span's FIRST position on top of a
**flat 0.315** at every offset eight or more tokens in. A next-span target is dominated by
the spike.

**The obvious objection, answered: `spandec_horizon` is not this.** `-h3` set the horizon
to 3 and decoded spans s+1, s+2 AND s+3 as one concatenated run at 3x the decoded
positions. The next span stayed in the target. It read K1−K6 +0.0016, K3−K6 −0.0000,
`all_slots` 0.1887, CE −0.0093 [−0.0116, −0.0073] against strict, and its worth profile's
far bins did not move at all (P-f FALSE,
[`failures/2026-09-12-arc-strict-geometry.md`](../../../../lab/experiments/failures/2026-09-12-arc-strict-geometry.md)).
Widening is not moving. A target that decodes span s+k and ONLY span s+k has never run.

## Proposal

Two knobs, two independent factors, three arms.

**`tul.spandec_reads_cells: true`** (default false = today, bit-identical). Each
`_SpanDecBlock` gains a cross-attention between its causal self-attention and its MLP
(`morph/model/tul_spandec.py::_SpanDecCross`). Queries are the decoder's own token states;
keys and values are the slot's M cells, each through the same `_readout` that makes `z`.
The memory is the object `TULSlots.prefix_project` writes into the coda's prefix positions
— the think-once stack's output where `tul.cond_layers > 0`, since `_tul_cond_apply` runs
before the cells are formed — so the decoder and the coda read ONE thing. The `z`
conditioning stays the MEAN, so the arm ADDS a path and replaces none. The cross output
projection is zero-init and the cross weights come from a third private generator with the
global RNG snapshotted and restored around their construction, so step 0 is
bit-for-bit `slot-register-m4`. Refused at `slot_cells == 1` and with `spandec: false`.

**`tul.spandec_target_offset: k`** (default 1 = today, bit-identical). Slot s's decoder
decodes span s+k and only span s+k, through the existing `span_slots(shift=k)` primitive
that `spandec_horizon` already uses for its blocks. The last k−1 slots of a row have no
span s+k and are masked to −100 as pad slots are. Teacher forcing, cost, and everything
else are unchanged. Offset and horizon compose (spans s+k .. s+k+H−1) because that falls
out of the same loop; no arm runs the combination. Refused below 1, at or past the derived
slot budget (config time in `build_tul_runtime` AND at the layout in `span_slots`), with
`spandec: false`, and with `spandec_per_pass` / `oracle_z` / `coda_span_heads`, each of
which defines "the next span" for itself and would silently disagree with the decoder.

Arms: `slot-register-m4-reader` (against `slot-register-m4`), `slot-spandec-strict-off2`
and `slot-spandec-strict-off3` (against `slot-spandec-strict`).

**What the theory says, stated because it is not a prediction of success.** The information
view ([`2026-09-13-information-view-of-the-slot-loop.md`](2026-09-13-information-view-of-the-slot-loop.md))
says a deterministic loop cannot add information, so loop value comes only from the
reader's observation. All three arms keep `tg_coda_prefix_reach: all`, so the coda already
sees every cell and `mi_relay_redundant` applies: **no relay is available to any of them.**
The span decoder is an AUXILIARY scorer, not the coda, so a better decoder changes the
objective's shape and not the coda's window. A deterministic reader cannot make passes
matter, and the frozen prediction is that K3−K6 stays under 0.002 on all three arms
(P-5, 80 %). That makes the reader arm a direct test of the theory's central claim rather
than a bet on it.

## Alternatives considered

* **The mean of the cells — today's build.** One target per span, every downstream reader
  (MUX, spandec, SIGReg, the energy) left as the shipped mechanism, and the register arm
  therefore one factor against its ruler. That is exactly why it shipped first, and it is
  the arm this one is paired with. Kept as the partner, not superseded: `m4-reader` is
  unreadable without `m4`'s numbers beside it.
* **Concatenate the M cells into one wide z** (`[M·d]` into a widened `z_in`). Cheaper
  than attention and it does give the decoder every cell. Rejected: the decoder would read
  the cells through ONE fixed linear map at ONE position of its sequence, so a cell's
  contribution could not depend on which token is being decoded — which is the whole
  mechanism the cross-attention is for. It also fixes M in the decoder's parameter shapes,
  so `m4` and `m8` could not share a reader design.
* **A decoder that reads the stack's MEAN instead of the loop's cells** (put the
  cross-attention after `_tul_cond_apply` but over the stack output averaged). Rejected as
  a distinction without a difference: `_tul_cond_apply` already runs before the cells are
  formed, so the cells ARE the stack's output on a cond arm, and averaging them is the
  build this arm replaces.
* **Per-cell targets — cell i decodes token bucket i of the next span.** A stronger
  separating force than a shared memory, and the register note's own named follow-up.
  Rejected for now because it changes the TARGET's factorisation and the reader at once:
  if it won, nothing would say whether the cells needed their own losses or just their own
  read. It stays queued behind this arm and becomes the next move if P-1 and P-3 both hold.
* **Offset via a longer horizon** (`spandec_horizon: 3`, already run as `-h3`). It is the
  cheapest way to reach span s+3 and it is already in the ledger, so it needed answering
  rather than proposing. Rejected as a substitute: it keeps the next span in the target,
  which is the thing the JEPA screen says to remove, and it measured its far worth bins
  UNCHANGED at 3x the decoder cost. The two knobs are kept separate and the difference is
  written into both configs and into `TULConfig`.
* **A horizon that skips its first blocks** (decode s+1..s+3 but weight block 1 at zero).
  Same effect as the offset, at three times the decoded positions and the readout. Rejected
  on cost: `-h3` already took the decoder from 4.0 to 12.0 block-passes per token and the
  arm nearly missed the queue's rate floor.
* **Doing nothing about the offset and running more register arms.** Rejected because the
  register's own prereg says the target is the lane if its P-3 fails, and the two builds
  are independent: the offset arms do not touch the register and cost no rate.

## Acceptance criteria

1. `spandec_reads_cells: false` and `spandec_target_offset: 1` build nothing, draw no RNG
   and leave the forward bit-identical. **Met, measured**:
   `lab/divergence/spandec_off_pin.py` ran on the parent commit `f89256d` (this change's
   four source files stashed) and on this tree — four fixtures at M = 1 and M = 4; loss,
   logit sum, `spandec_ce`, grad sum and `state_dict` key count agree to the last printed
   digit. The rows are pinned as literals in `tests/test_tul_spandec_reads_cells.py`.
2. `spandec_reads_cells` at step 0 is bit-for-bit its partner (zero-init projection), every
   shared gradient included, and diverges after one optimizer step. **Met**, guarded.
3. The memory the decoder reads equals, cell by cell, the object `prefix_project` writes.
   **Met**, guarded by a spy on both, not by reading the call site.
4. Every memory cell is load-bearing: blanking cell j with `z` HELD FIXED moves the
   decoder's states, for every j — and does NOT at the zero init, which is the two-sided
   control. **Met**, guarded.
5. `spandec_target_offset: k` decodes exactly span s+k, masks the last k−1 slots, and its
   supervised token count matches an independent Python oracle. **Met**, guarded on a
   RAGGED-span fixture (a uniform one silently overwrites a mis-targeted slot).
6. Every refusal fires. **Met**, guarded, including the two in `tul_setup.py` (reached
   through a stubbed tokenizer, so the suite stays offline).
7. `val/slot_cell_eff_rank` on `slot-register-m4-reader` above `slot-register-m4`'s by more
   than 0.3 at 5,000 steps. **Not met — unrun.**
8. Token K1−K6 above 0.005 on `slot-register-m4-reader`. **Not met — unrun.**
9. `slot-spandec-strict-off2`'s `worth_profile` far bin (offset 16+) at least 0.02 nats
   above `slot-spandec-strict`'s. **Not met — unrun.** This is the offset arms' reason to
   exist.

Prereg:
[`lab/experiments/planned/2026-09-13-arc-register-reader-and-downstream-target.md`](../../../../lab/experiments/planned/2026-09-13-arc-register-reader-and-downstream-target.md).

## Risks

* **Nothing has run on a GPU.** No smoke, no wall clock, no memory figure for any of the
  three arms. Every cost number is arithmetic. The forward and backward were exercised at
  the real `d_model` 1024 on a 256-token CPU batch for all three composed configs; every
  test is at `d_model` 64.
* **`slot-register-m4` itself has never run either.** `m4-reader` is one factor against an
  arm with no measurements. The pair has to be queued and read together or the reader arm
  says nothing.
* **The reader may improve the decoder and change nothing else.** That is P-2 holding and
  P-3 failing, and it is the most likely single outcome (the information view predicts it).
  It is a real result — the first direct measurement that a better AUXILIARY reader cannot
  move the coda's K-curve — but it will look like a null arm to anyone reading only the CE
  column, so the prereg's binding says so in advance.
* **The offset arms should cost CE at 5k** (P-6, 60 %). The next span's first tokens lose
  their direct z target and that is where the 0.958-nat spike lives. Do not read that as a
  failure; the reading is the worth profile by offset bin.
* **The far target can go inert.** Three boundaries out, p(span s+3 | z, its own prefix)
  may be close to the unconditional, the decoder learns to ignore z, and `-off3` becomes a
  slower ruler with a noise term bolted on. P-9 is the non-vacuity check and its zeroed-z
  control is NOT built — it is one extra eval pass with the existing `plan_mode: zero`.
* **`spandec_ce` is not comparable across offsets.** At k > 1 it grades a harder span.
  `spandec_target_offset` now travels with the column through the trainer's val stats and
  `core_depth_sweep.py`'s `MUX_KEYS`, so a scorer cannot read the two as the same number —
  but no scorer REFUSES to compare them either, and the ledger has no baseline for a
  span-s+2 CE.
* **The cross-attention's cost is not in `_tul_layer_passes`.** That metric counts BLOCK
  passes and the block count is unchanged, so the reader arm's flop proxy under-reports it.
  Named rather than fudged.
* **The register's eager core is inherited.** `slot_cells > 1` forces the eager attention
  path on the core stage (`tg_relation`), which `slot-spandec-strict` avoids under
  `tg_scoped_kernels`. The reader arm inherits that difference from its partner and adds
  nothing to it.
* **One defect found and one sabotage withdrawn, both recorded.** Three of fifteen
  sabotages MISSED on the first pass and each exposed a real gap: a one-layer decoder
  fixture that could not see a private-stream leak, a uniform-span fixture that overwrote a
  mis-targeted slot, and no test at all reaching `tul_setup.py`'s wiring. All three gaps
  are now closed and the second pass is 15/15. A sixteenth patch — weakening the graded
  slot's existence check from `slot_valid[k-shift]` to `slot_valid[k-1]` — turned out to be
  an INERT rewrite (measured on three slot budgets x six seeds x four shifts: identical
  everywhere, because `slot_valid` is a prefix and `span_done` already covers it), so it
  was withdrawn rather than recorded as a catch.
* **The sabotage harness itself had to be fixed.** Its first run reported 15/15 CAUGHT with
  a RED clean gate — a new test failed before any patch was applied, so every patch
  "caught" the same unrelated failure. The harness now refuses to report until the clean run
  is green.
