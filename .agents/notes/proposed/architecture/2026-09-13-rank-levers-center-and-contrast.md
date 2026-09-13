# Agent Note: two levers against the slot states' rank collapse — row centering and a within-row contrastive target

Status: proposed

## Problem

A row's written slot states are near copies of each other. On the strict ruler
`slot-spandec-strict`, 64 states of 1024 dimensions read `val/slot_eff_rank` **5.7598** and
`val/slot_pairwise_cos` **0.7104**; the slot family sits at 5.7–7.3 / 0.72–0.77, the
2026-09-10 geometry audit at 1.7–4.8, and 4,906 slots at step 0 at 18–23. Twelve arms have
read token K1−K6 inside [−0.0001, +0.0033] (the ruler +0.0016, K3−K6 +0.0002) across every
lever tried, and none of them changed how DISTINCT the states are.

[The Thought Register](2026-09-13-the-thought-register.md) attacks this by giving a span
more state. Two other directions have never been tried on the exit: **removing what the
states share**, and **demanding that they differ**.

### One fact that reshapes the problem, and it came out of reading the instrument

`effective_rank` ([`morph/model/fm_planner.py`](../../../../morph/model/fm_planner.py))
is the participation ratio of the **CENTERED** covariance — it subtracts the mean of the
pooled cloud before the eigendecomposition. `mean_pairwise_cos` does not.

So **the ruler's 5.76 is already offset-free**: the states span about six directions
*after* the common mean is removed. A shared offset is therefore NOT an explanation of the
rank number. It is a complete explanation of the cosine number, and those are two different
claims that the lane has been treating as one. Any lever aimed at a shared offset should be
scored on `val/slot_pairwise_cos`, and a flat `val/slot_eff_rank` is not evidence it failed.

## Proposal

Two independent knobs, both default-off and both bit-identical when off.

### C1 — `tul.center_exit` (the row-centered exit)

After the loop and after the think-once stack, every VALID slot's exit state has the ROW's
mean over its valid slots subtracted and ONE learned bias `b_center` (zero-init) added
back. Pads are untouched. At `slot_cells > 1` each CELL INDEX is centered separately
across the row.

Four decisions, each with a reason that is in the code:

* **On the CARRIER, at the one seam upstream of every reader.** `h_slots` at that point is
  the tensor the MUX, the span decoder, SIGReg, the energy and `prefix_project` all take:
  the first four call `_readout` on it themselves and `prefix_project` projects it with its
  four Hyper-Connection streams intact. One edit centers what all five see; centering "a
  readout" would leave the coda's whole channel projecting the raw state.
* **Pads untouched**, the `center_bag_mean` dump-bin rule. Not cosmetic: a pad enters the
  loop at h = 0 through `gather_valid` but the first core step moves it off zero, so its
  state is live garbage that must not enter a row statistic.
* **Per CELL INDEX at M > 1.** Pooling all S·M cells into one mean would subtract the
  register's within-slot structure — the axis `val/slot_cell_eff_rank` measures — out of
  the row statistic.
* **The mean is NOT detached.** A differentiable reparameterisation, like a LayerNorm's:
  the readers see only the centered state, so the gradient must too. Detaching would leave
  a live gradient path pushing a shared offset no reader can read.

`b_center` is zero at init and drawn by `torch.zeros`, so it costs no RNG draw and a center
arm's base weights are byte-identical to its partner's. +4,096 parameters at d_model 1024.

### C2 — `tul.row_contrast_lambda` (the within-row contrastive objective)

For every valid slot `i` whose next span exists, span `i+1`'s pooled PRELUDE token states
must pick slot `i`'s exit state out of the row's other valid slots: a softmax at
temperature `row_contrast_tau` (0.1), charged `-log p(i | span_{i+1})`, weighted by λ.

* **The negatives are the row's OTHER valid slots and nothing else.** A pad is never a key;
  two rows never mix; slot `i`'s target is span `i+1`, so no two anchors share a target and
  there are no false negatives. The LAST valid slot is not an anchor — the tokens after it
  go to the dump bin, whose `bag_mean` row is defined to be exactly 0.
* **ONE direction, span → z.** With every state identical this softmax is EXACTLY uniform,
  so the term reads `log n_valid` and `tul/row_contrast_acc` reads `1/n_valid` at chance
  with no calibration run. The z → span half has no analytic chance value (its keys are the
  span pools, which stay distinct however far the states collapse), so including it would
  make the reported number unreadable on its own. Both directions push the states apart;
  only this one says by how much.
* **The pooled targets are DETACHED before the readout.** `_readout` is `lm_mixer` +
  `final_norm`, both trained and both shared with the LM head, so detaching after it would
  let the target path reshape them — `mux_detach_head`'s subject, and what made arm v1a
  diverge at step 2800. The target path trains `W_contrast` and nothing else.
* **It is causal.** Slot `i` sits after its own span and before span `i+1`'s tokens, so
  under every TG geometry here `z_i` has not read the span it is asked to identify. This is
  the span decoder's target scored as a within-row retrieval instead of a token decode.
* At `slot_cells > 1` it reads the CELLS' MEAN — the state every other reader at that seam
  takes. Per-cell targets are a SECOND mechanism and the named follow-up.

`W_contrast` (d_model × d_model, +1,048,576 parameters) is drawn from a private generator
with the global RNG stream snapshotted and restored, the `TULSlotRegister` rule, so a
contrast arm's base weights are byte-identical to its partner's.

### The arms

`tul_slot_spandec_strict_center.yaml`, `tul_slot_register_m4_center.yaml`,
`tul_slot_spandec_strict_contrast.yaml`, `tul_slot_register_m4_contrast.yaml` — each ONE
factor against `slot-spandec-strict` or `slot-register-m4`. Prereg:
[`2026-09-13-arc-rank-levers-center-and-contrast.md`](../../../../lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md).

## Alternatives considered

* **SIGReg on the slot states (`tul.sigreg_lambda`), the tree's existing isotropy
  regulariser.** The obvious first move, and it is still available: the machinery ships
  (`morph/model/sigreg.py`, `_tul_sigreg_loss`), it acts on exactly the right tensor (the
  readout of the valid slot states), and the 2026-08-27 panel's own verdict is "**SIGReg is
  untested, not refuted**" — both seeds aborted at step 2040 with the loss stuck at
  ppl ~1900 and core share 1.0 by step 16, on a λ picked from a probe under a contaminated
  head. **Checked and recorded here because the lane keeps assuming otherwise: it is OFF on
  every arm in this lineage.** `tul_gl1.yaml` sets `tul.sigreg_lambda: 0.02` and its own
  child `tul_gl1b.yaml` sets it back to `0.0`; the `sigreg_lambda: 0.02` in `base.yaml`
  lives in the **`fm:`** block, regularises the FM planner's pooled targets, and
  `fm.enabled: false` builds nothing. Measured through the composed config:
  `tul_slot_spandec_strict` builds `sigreg_lambda = 0.0`. Not chosen for THIS panel because
  a distributional penalty says nothing about whether a state is USEFULLY distinct, and C2
  answers that with `row_contrast_acc` while SIGReg cannot. It is the named next arm if C1
  moves the cosine and C2 does not train.
* **Per-row whitening (ZCA on the row's states) instead of centering.** Strictly stronger:
  it removes the shared offset AND equalises the directions, so it attacks `slot_eff_rank`
  head-on where centering cannot. Rejected for this panel on two grounds. It needs a
  64 × 64 inverse square root per row in the training graph, differentiated, at every step —
  a real cost and a real stability surface on a loop whose divergence history fills
  `lab/divergence/`. And it is not one factor: it changes the scale of every direction as
  well as the mean, so a result could not be attributed. Centering is the cheap, isolable
  half; whitening is what to reach for if the cheap half moves the cosine and the rank
  stays put.
* **A VICReg variance term (hinge on the per-dimension standard deviation across the row).**
  Cheaper than whitening, attacks the rank directly, and needs no inverse. Rejected as the
  FIRST objective to try because it is the same CLASS as SIGReg — a distributional penalty
  on the states' marginals, with no reference to what the state is for. The 2026-08-27
  record's most useful sentence is that geometry (center, SIGReg), schedule (warmup) and
  shortcut removal (ntpdrop) all failed while a NEW OBJECTIVE moved the loop. C2 is the new
  objective; VICReg would be a fourth geometry lever. It is the fallback if C2 trains
  (`row_contrast_acc` well above chance) and the geometry still does not move — that
  combination would mean the retrieval job is satisfiable without spreading the states.
* **The register alone (`tul.slot_cells`), and nothing here.** The strongest single
  argument: the register is already built, already queued, and gives a span four states
  where these two levers give it one better-shaped state. Rejected as sufficient because
  the register's OWN note names the risk that "the cells may collapse onto each other
  anyway — the loop and the shared write are exactly the forces that produced the rank
  collapse", and nothing in the register pushes back on that force. C1 and C2 are the push,
  and `slot-register-m4-center` / `-contrast` are the arms that test the register under it.
  The register is not an alternative to these; it is what they compose with.
* **Symmetric InfoNCE (both directions) for C2.** The standard CLIP form and the default
  choice. Rejected for the readout: the z → span half has no analytic chance value, so the
  reported `tul/row_contrast` would need a calibration run to be read at all, and the
  lane's whole problem is that it has no absolute distinctness reference. One direction
  gives `log n_valid` exactly. The rejected half pushes the states apart in the same
  direction, so nothing is lost but a constant.
* **Per-cell contrastive targets on the register (cell `i` against span `i+1`).** A stronger
  separating force than the cells' mean. Rejected for the same one-factor reason the
  register's note rejected per-cell span-decoder targets: it changes the target AND the
  state at once. Named as the follow-up if `slot-register-m4-contrast` moves the row
  reading and leaves `val/slot_cell_eff_rank` where it was.

## Acceptance criteria

1. OFF is nothing: neither knob builds a parameter, draws RNG, adds an output key or
   changes a number. **Met** — `state_dict` key diffs are exactly `{tul_center.b_center}`
   and `{tul_contrast.W_contrast.weight}`, every other tensor is byte-equal, and the
   contrast arm's `spandec` (which reads the same seam) is bit-identical to its ruler's.
2. The state really is centered where the CODA reads it: the per-row mean over valid slots
   of the tensor handed to `prefix_project` equals `b_center` to 1e-6, per cell index at
   M = 4, with pads bit-identical to an OFF twin. **Met**, guarded.
3. The mean is live: a common shift of every valid slot carries EXACTLY zero gradient.
   **Met**, guarded (a detached mean passes the forward half and fails this).
4. `tul_slot_state_probe` applies the centering, so `val/slot_eff_rank` is not blind to the
   arm. **Met**, guarded.
5. The contrastive term reads `log n_valid` and accuracy `1/n` when the states are
   identical, at n = 3, 6 and 9. **Met**, guarded.
6. A pad is never a key and two rows never mix. **Met**, guarded two-sided (the vacuous
   version of each test is rejected by a companion assert).
7. The pooled targets are detached BEFORE the readout. **Met**, guarded by recording the
   `requires_grad` of each `_readout` call in order: `[True, False]`.
8. All four configs compose and run one forward+backward at d_model 1024. **Met** —
   `lab/divergence/rank_levers_compose.py`, `COMPOSE GATE: PASS`.
9. `val/slot_pairwise_cos` below 0.40 on `slot-spandec-strict-center`. **Not met — unrun.**
10. `tul/row_contrast_acc` above 0.35 on `slot-spandec-strict-contrast`. **Not met —
    unrun.**
11. Token K1−K6 above 0.005 on either arm. **Not met — unrun**, and the prediction says it
    will not be met.

## Risks

* **The precedent says this class of lever does not pay.** `tul.center_bag_mean` centered
  the SEED (arm `tul_center`, 2026-08-27): its frozen prediction was pairwise cosine
  < 0.20, it read 0.337–0.578 against a control's 0.485–0.589, and it bought **+0.0015
  nats** of loop worth. It DID reverse the trend across the loop's passes (control +0.334 →
  +0.729 with rank 4.21 → 1.46; centered +0.578 → +0.337 with rank 2.63 → 4.41). C1 is that
  lever moved to the EXIT, under `norm_match`, a span decoder and the strict geometry, none
  of which existed then. It is still a geometry intervention against a record that says
  geometry alone has not moved this loop.
* **C1's escape hatch, and it is arithmetic.** `mean_pairwise_cos` does not center, so a
  `b_center` that grows large puts the pooled cosine straight back while the states stay
  exactly as collapsed. `val/slot_eff_rank` cannot be gamed that way, and `||b_center||` is
  the number to read if the cosine rebounds. Written into the prereg as a separate outcome
  (P-6), not folded into "the lever failed".
* **C1 cannot raise `slot_eff_rank` much, by construction.** The instrument already
  centers. What C1 removes that the instrument does not is the PER-ROW mean. This is stated
  in the prereg, in the config header and in the startup banner, because "the rank did not
  move" is the reading most likely to be misinterpreted here.
* **λ is a guess.** 0.1 against a term near `log 60 ≈ 4.1` at chance is roughly 10 % of the
  token loss, sized by arithmetic and not by a probe. The 2026-08-27 SIGReg arm died at step
  2040 on a λ picked badly, and its own record says "if the arm dies immediately, suspect λ
  before suspecting the idea." Same rule here.
* **The contrastive term can be satisfied without helping.** A retrieval job over ~50
  in-row candidates is easy to win with a low-dimensional identity code — a span index, in
  effect — which would raise `row_contrast_acc` and the rank while carrying nothing the
  coda wants. `worth_profile`'s `all_slots` total (P-14) and the paired CE are what catch
  that, and a high `row_contrast_acc` with a flat worth profile is a specific, nameable
  outcome rather than a null.
* **Both register arms pair with an UNMEASURED partner.** `slot-register-m4` has not run a
  GPU step either. The pairs are still one-factor and still readable as pairs; what they
  cannot say until the register runs is whether register + lever beats the k=2 ruler.
* **Nothing has run on a GPU.** No smoke, no wall clock, no memory figure. Every cost line
  in the configs is arithmetic.

## Sabotage

Sixteen plausible defects, applied to the source one at a time and reverted. **16 caught,
0 missed** — but only after the pass found two real problems, and both are recorded because
the first-round result (13/13) was wrong about its own coverage.

| id | defect | verdict |
| --- | --- | --- |
| D1 | center the PADS too (drop the valid-slot `where`) | CAUGHT |
| D2 | pool all M cells into ONE mean (drop per-cell centering) | CAUGHT |
| D3 | DETACH the row mean (forward identical, backward wrong) | CAUGHT |
| D4 | divide by S instead of the VALID count | CAUGHT |
| D5 | skip the centering in `tul_slot_state_probe` | CAUGHT |
| D6 | drop the centering from the forward entirely | CAUGHT |
| D7 | read the cell axis CELL-major instead of slot-major | CAUGHT (after a new test) |
| E1 | detach AFTER the readout (target path trains `lm_mixer`) | CAUGHT |
| E2 | mask ROWS instead of COLUMNS (pads become keys) | CAUGHT |
| E3 | make the LAST valid slot an anchor (target is the zero dump bin) | CAUGHT |
| E4 | score against the WHOLE batch (rows leak into each other) | CAUGHT |
| E5 | use `-inf` (an all-masked row NaNs in the backward) | CAUGHT |
| E6 | score single-anchor rows (an unearned exact 0) | CAUGHT |
| E7 | build the head but never add the term | CAUGHT |
| E8 | drop the term on the `loop_reads_tokens` branch | CAUGHT (after a new test) |
| E9 | pool slot `i`'s OWN span instead of its next one | CAUGHT (after a stronger test) |

**D7 was a real hole and it is the instructive one.** Reading the compact cell axis
CELL-major instead of slot-major PERMUTES which slot's state lands where — and it still
satisfies "the per-cell row mean equals `b_center`", because the permutation maps groups
onto groups. A mean identity cannot detect a permutation that preserves the groups. The
fix is a test that pins VALUES: slot `s` and cell `m` carry `(s+1)(1 + 0.01m)`, so the two
axes stay separable after centering and every output position is predicted exactly.

**E9's first form was a bad PATCH, not a hole, and that is worth writing down too.**
"Anchor on slot `i+1`'s validity alone" changes no behaviour: `SlotLayout` guarantees pads
are LAST, so `slot_valid[i+1]` implies `slot_valid[i]` and the AND in `next_span_pool` is
redundant under the invariant. A sabotage patch that is semantically equivalent proves
nothing either way. Replaced with the off-by-one that actually matters — pooling slot `i`'s
OWN span instead of its next — which the original trend-shaped test also missed, so that
test now pins the pool by VALUE against `layout.bag_id`.

## A repair made in the same change

`morph/configs/tul_center.yaml` was a BUILD ERROR and nothing had noticed, because no arm
has composed it since 2026-08. `tul.center_bag_mean` is legal only with
`slot_seed: bag_mean` and `TULConfig.__post_init__` raises otherwise — that guard is
correct and already tested, so the knob is **not** silently inert. What changed under the
file is the tree's default seed: `tul_tg4b.yaml` moved it to `boundary`, so the composed
config became illegal. Measured:

```
build_tul_runtime(compose("tul_center"))
-> ValueError: tul.center_bag_mean=true with tul.slot_seed='boundary' is not supported:
   centering is scoped to slot_seed='bag_mean' only.
```

Fixed by pinning `slot_seed: bag_mean` in that file, which restores the arm to the thing it
was measured as. It does not resurrect the arm (C1 is on record as FAILED) and it does not
change what the knob means.
