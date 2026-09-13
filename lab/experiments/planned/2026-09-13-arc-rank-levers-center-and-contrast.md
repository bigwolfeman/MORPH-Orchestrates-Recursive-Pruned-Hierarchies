# Planned: two levers against the slot states' rank collapse — row centering and a within-row contrastive target

Status: planned

Date: 2026-09-13 (frozen before any GPU step of any of the four arms; no smoke of any arm
here exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). One-factor
partners: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)) and
`slot-register-m4`
([`2026-09-13-arc-thought-register.md`](2026-09-13-arc-thought-register.md), unrun).
Design note:
[`2026-09-13-rank-levers-center-and-contrast.md`](../../../.agents/notes/proposed/architecture/2026-09-13-rank-levers-center-and-contrast.md).

## The measured defect

A row's written slot states are near copies of each other.

| reading | value | source |
| --- | --- | --- |
| `val/slot_eff_rank` on `slot-spandec-strict` | **5.7598** in 1024 dims | `run_slot-spandec-strict.log`, last val |
| `val/slot_pairwise_cos` on the same run | **0.7104** | same |
| the same pair across the slot family | 5.7 – 7.3 / 0.72 – 0.77 | [`2026-09-10-arc-slot-mux-prefix4-norm-match.md`](../successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md) |
| the 2026-09-10 geometry audit | 1.7 – 4.8 | audit |
| across 4,906 slots at step 0 | 18 – 23 | `results/2026-09-12-latent-z-gradient/step0-*.json` |

Twelve arms read token K1−K6 inside [−0.0001, +0.0033] — the strict partner **+0.0016
[+0.0013, +0.0019]**, K3−K6 **+0.0002** — across every lever the 2026-09-11 batch tried.
The Thought Register attacks this by giving a span more state. These two arms attack the
same number from the other two directions: **remove what the states share** (C1) and
**demand that they differ** (C2).

## Question

Does the collapse yield to a geometry edit or to an objective, and does either one move
the loop's contribution?

## Hypothesis

**Low probability of moving the K-curve, and the arms are worth running anyway.** The
honest prior is the 2026-08-27 record: `tul.center_bag_mean` centered the SEED, changed
the geometry measurably and bought **+0.0015 nats** of loop worth. Geometry alone has been
tried and it did not pay. What is different now: `norm_match`, the span decoder, the
strict geometry, and — for C2 — an OBJECTIVE rather than a geometry edit, which is the one
class of intervention that has ever moved this loop (arm `tul_v1a2b`, the same 2026-08-27
panel: "what moved the loop was a NEW OBJECTIVE with its own headroom").

The value of the panel is not the K-curve. It is that `tul/row_contrast_acc` gives the
lane its **first absolute distinctness reading** — 1/n_valid is chance with no calibration
run — and that the two arms separate "the states share an offset" from "the states carry
nothing to tell apart", which no instrument in the tree has done.

### A correction to the theory's own prediction, before the run

The brief for this build said "rank up, K-curve flat, high probability". The first half is
wrong for C1 and it matters, because a flat `val/slot_eff_rank` would otherwise be read as
"the lever did not act". **`effective_rank` already centers**
([`morph/model/fm_planner.py`](../../../morph/model/fm_planner.py): `rows = rows -
rows.mean(0, keepdim=True)` before the covariance), so a globally shared offset contributes
**nothing** to `val/slot_eff_rank` and C1 cannot remove it from there. Two consequences:

* **The ruler's 5.76 is ALREADY offset-free.** The states span about six directions *after*
  the common mean is gone. A pure shared offset is therefore NOT the explanation of the
  rank number — only of the cosine number (`mean_pairwise_cos` does not center).
* **C1's head-on reading is `val/slot_pairwise_cos`.** What it can still do to the rank is
  remove the PER-ROW mean, which the instrument's global centering leaves in. That is a
  real but second-order effect and the prediction below is sized for it.

The rank-anatomy instrument
([`lab/divergence/slot_rank_anatomy.py`](../../divergence/slot_rank_anatomy.py), run
separately) is what decides between "a shared offset" and "a true collapse"; this panel
does not need its answer to run, because C1 is built for the first and C2 for the second.

## Method

Four arms, each ONE factor against its partner, 5,000 steps, seq 1024, batch 6, seed 1,
ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route off, retention
off, `tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`,
`mux_beta: 0`, `spandec` on at J 32.

| arm | config | partner | one factor |
| --- | --- | --- | --- |
| `slot-spandec-strict-center` | `tul_slot_spandec_strict_center.yaml` | `slot-spandec-strict` | `tul.center_exit: true` |
| `slot-register-m4-center` | `tul_slot_register_m4_center.yaml` | `slot-register-m4` | `tul.center_exit: true` |
| `slot-spandec-strict-contrast` | `tul_slot_spandec_strict_contrast.yaml` | `slot-spandec-strict` | `tul.row_contrast_lambda: 0.1` |
| `slot-register-m4-contrast` | `tul_slot_register_m4_contrast.yaml` | `slot-register-m4` | `tul.row_contrast_lambda: 0.1` |

**`slot-register-m4` has not run either.** The two register arms therefore pair with an
UNMEASURED partner. They are still one-factor pairs and they are still readable as pairs;
what they cannot do until the register runs is say whether a register + lever result is
better than the k=2 ruler. Named here, not buried.

### C1, the row-centered exit — what runs

After the loop and after the think-once stack, every VALID slot's exit state has the ROW's
mean over its valid slots subtracted and ONE learned bias `b_center` added back. Pads are
untouched. At `slot_cells > 1` each CELL INDEX is centered separately across the row.

* **On the CARRIER, at the one seam upstream of every reader.** `h_slots` there is the
  tensor the MUX, the span decoder, SIGReg, the energy and `prefix_project` all take —
  the first four call `_readout` on it themselves, and `prefix_project` projects it with
  its four Hyper-Connection streams intact. One edit centers what all five see. Centering
  "a readout" would leave the coda's whole channel reading the raw state.
* **The mean is NOT detached.** A differentiable reparameterisation, like a LayerNorm's:
  the readers see only the centered state, so the gradient must too. Guarded by a test
  that a common shift carries EXACTLY zero gradient — a detached mean passes the forward
  half of that and fails the backward half.
* `b_center` is zero at init and draws no RNG, so a center arm's base weights are
  byte-identical to its partner's (+4,096 parameters at d_model 1024, n=4 streams).
* `tul_slot_state_probe` applies it too, or `val/slot_eff_rank` would report the loop's
  raw exit and the arm's headline instrument would be blind to the arm.

### C2, the within-row contrastive objective — what runs

For every valid slot `i` whose next span exists, span `i+1`'s pooled PRELUDE token states
must pick slot `i`'s exit state out of the row's other valid slots: a softmax at
temperature 0.1, charged `-log p(i | span_{i+1})`, added at weight 0.1.

* **Negatives are the row's OTHER valid slots and nothing else.** A pad is never a key;
  two rows never mix; slot `i`'s target is span `i+1`, so no two anchors share a target
  and there are no false negatives. The LAST valid slot is not an anchor — the tokens
  after it go to the dump bin, whose `bag_mean` row is defined to be exactly 0.
* **ONE direction, span → z.** With every state identical this softmax is EXACTLY uniform,
  so the term reads `log n_valid` and the accuracy reads `1/n_valid` with no calibration
  run. The z → span half has no analytic chance value (its keys stay distinct when the
  states collapse), so including it would make the reported number unreadable alone.
* **The pooled targets are DETACHED before the readout**, so the target path trains
  `W_contrast` and nothing else — not the prelude, not `lm_mixer`, not the embedding
  table (`mux_detach_head`'s subject: the tied table IS the input embedding table, and arm
  v1a diverged at step 2800 over it). The anchor path is live and reaches the loop, the
  seed and the core.
* **Causal.** Slot `i` sits after its own span and before span `i+1`'s tokens, so under
  every TG geometry here `z_i` has not read the span it is asked to identify. It is the
  span decoder's target scored as a within-row retrieval instead of a token decode.
* At `slot_cells > 1` it reads the CELLS' MEAN — the state every other reader at that seam
  takes. Per-cell targets are a SECOND mechanism and the named follow-up.
* +1,048,576 parameters (`W_contrast`, d_model × d_model), drawn from a private generator
  with the global stream snapshotted and restored, so base weights are byte-identical to
  the partner's.

### λ = 0.1, and why not more

The term is a CE over roughly 60 classes, so chance is near `log 60 ≈ 4.1` and λ = 1 would
make it comparable to the model's own token CE. 0.1 puts it near 10 % of the token loss.
The 2026-08-27 SIGReg arm is the warning: λ = 0.001 came from a magnitude probe, both
seeds ABORTED at step 2040 with the loss stuck at ppl ~1900, and the record's own verdict
is "if the arm dies immediately, suspect λ before suspecting the idea." **The same rule
applies here.**

### The gate, built in this change

* `tests/test_tul_center_exit.py` — **25 passed** (`pytest -q`, exit 0).
* `tests/test_tul_row_contrast.py` — **25 passed** (`pytest -q`, exit 0).
* `lab/divergence/rank_levers_compose.py` — all four configs compose through Hydra and
  `build_tul_runtime`, build at **d_model 1024**, and run one forward+backward with every
  gradient finite. `COMPOSE GATE: PASS`.
* Sabotage: **13 defects, 13 caught, 0 missed** (table in the note).

### Two audit findings recorded here because the record needs them

**1. `tul.center_bag_mean` is NOT silently inert under `slot_seed: boundary` — it RAISES**
(`morph/model/tul.py`, `TULConfig.__post_init__`: *"centering is scoped to
slot_seed='bag_mean' only"*), and `tests/test_slot_seed.py` already pins it. What WAS
broken: `morph/configs/tul_center.yaml` composed to `slot_seed='boundary'` (the default
moved under it in `tul_tg4b.yaml`) and was therefore a BUILD ERROR nothing had noticed,
because no arm has composed it since 2026-08. Measured:

```
build_tul_runtime(compose("tul_center"))
-> ValueError: tul.center_bag_mean=true with tul.slot_seed='boundary' is not supported
```

Fixed in this change by pinning `slot_seed: bag_mean` in that file, which restores the arm
to the thing it was measured as. It does not resurrect the arm and does not change what
the knob means.

**2. SIGReg is OFF on the strict ruler, on the register, and on all four arms here.**
Measured through the composed config, not read off a YAML:

```
tul_slot_spandec_strict :  tul.sigreg_lambda (BUILT) = 0.0   slices = 1024
tul_slot_register_m4    :  tul.sigreg_lambda (BUILT) = 0.0   slices = 1024
..._center / ..._contrast: tul.sigreg_lambda (BUILT) = 0.0   slices = 1024
every one of them       :  fm.enabled = False,  fm.sigreg_lambda = 0.02
```

`tul_gl1.yaml` sets `tul.sigreg_lambda: 0.02` and `tul_gl1b.yaml` — its own child, and an
ancestor of every arm in this lineage — sets it back to **0.0**. The `sigreg_lambda: 0.02`
in `base.yaml` is in the **`fm:`** block: it is the FM planner's regulariser on the POOLED
TARGETS, and `fm.enabled: false` builds nothing. So the 2026-08-27 verdict stands
unchanged: **SIGReg on the slot states is untested, not refuted.** It is the named
alternative to C1 in the design note and it is not in this panel.

## Predictions (frozen, one line each)

**C1, `slot-spandec-strict-center` against `slot-spandec-strict`:**

* **P-1.** `val/slot_pairwise_cos` at 5,000 steps **below 0.40**, from the partner's
  0.7104 — a fall of at least 44 %. This is the reading the lever acts on directly.
* **P-2.** `val/slot_eff_rank` moves by **less than ±20 %** of the partner's 5.7598 (i.e.
  stays inside 4.61 – 6.91). The instrument already centers globally, so a large move
  either way would mean the lever did something other than remove a common component.
* **P-3.** Token K1−K6 stays inside **[−0.001, +0.005]** against the partner's +0.0016
  [+0.0013, +0.0019], and K3−K6 stays below +0.002. **High probability**, and it is the
  2026-08-27 precedent talking: the seed-centering arm changed the geometry and bought
  +0.0015 nats.
* **P-4.** `worth_profile` `all_slots` total within **±0.02 nats** of the partner's
  **0.1865**, and the offset-0 bin within ±0.05.
* **P-5.** Paired depth-6 val CE within **±0.03 nats** of the partner (paired on
  `tok_index`), and tok/s within **±3 %**. The lever is one mean and one add.
* **P-6, the escape hatch.** If P-1 fails AND `||b_center||` at 5,000 steps exceeds the
  mean slot-state norm, the model put the shared component back into the bias and the
  cosine number was never the lever's to move. That is a distinct outcome from "the lever
  acted and the cosine did not fall", and it is reported as one.

**C1, `slot-register-m4-center` against `slot-register-m4`:**

* **P-7.** `val/slot_cell_eff_rank` (rank WITHIN a slot, max 3 of 4 because the instrument
  centers) moves by **less than ±15 %** against `slot-register-m4`. The centering is
  per-cell-index across the ROW, so a slot's four cells all move by four different vectors
  and their relative geometry is not a fixed shift — any movement here is a training
  effect, and a large one would be a surprise worth its own line.
* **P-8.** Same direction as P-1 on the row reading: `val/slot_pairwise_cos` over the
  row's 256 cells **below 0.45**.

**C2, `slot-spandec-strict-contrast` against `slot-spandec-strict`:**

* **P-9.** `tul/row_contrast_acc` at 5,000 steps **above 0.35**, against a chance of
  roughly **1/50 = 0.02** at the panel's ~50 spans per row. Anything at or below 0.05 means
  the term did not train and λ or the head is the fault, not the idea.
* **P-10.** `val/slot_pairwise_cos` **below 0.30** and `val/slot_eff_rank` **above 12.0**
  (roughly double the partner's 5.7598). Unlike C1 this lever CAN move the rank: the
  instrument centers, and a retrieval objective demands directions that survive centering.
* **P-11.** Token K1−K6 stays inside **[−0.001, +0.005]** and K3−K6 below +0.002 — the same
  flat prediction as C1, at the same **high probability**. If this one is wrong it is the
  first lever in thirteen arms to move the K-curve, and the panel's whole reason to exist.
* **P-12.** Paired depth-6 val CE within **±0.05 nats** of the partner. The term is an
  auxiliary at λ 0.1 and it is subtracted out of the reported loss, so a CE cost larger
  than that means the term is fighting the token objective, not shaping the write.
* **P-13.** tok/s within **±5 %** (one [B, 64, 64] similarity and two d_model projections
  against a coda that already runs 1152 positions).
* **P-14.** `worth_profile` `all_slots` total **above 0.20** (partner 0.1865) — the slot
  channel should carry more if its states are more distinguishable, even if the passes do
  not.

**C2, `slot-register-m4-contrast` against `slot-register-m4`:**

* **P-15.** `tul/row_contrast_acc` **above 0.35**, matching P-9: the register does not make
  the row job harder, it only changes what is averaged into z.
* **P-16.** `val/slot_cell_eff_rank` **above 2.0** (of a maximum 3). The term acts on the
  row axis and reaches the cells only through the mean, so this is the INDIRECT reading and
  the one that says whether an objective can pull a register's cells apart. A value at or
  below the register's own is the result that sends the lane to per-cell targets.

**Panel-level, decided before the data:**

* If C1 moves `slot_pairwise_cos` and C2 moves `slot_eff_rank` while BOTH K-curves stay
  flat, the rank collapse is real, both levers work as advertised, and **the collapse is
  not what is keeping the loop flat**. That closes the geometry lane and the honest next
  move is the objective, not another geometry edit.
* If C2 moves the K-curve and C1 does not, the lever is the OBJECTIVE, matching the
  2026-08-27 finding, and the follow-up is per-cell targets on the register.
* If C2's `row_contrast_acc` sits near chance, nothing about the collapse was tested and
  the arm is a λ failure, not a result.

## Not verified at filing time

* **No GPU step of any of the four arms, and none of the register partner either.** No
  smoke, no wall clock, no memory figure. Every cost line is arithmetic.
* `center_exit` and `row_contrast_lambda` have never run at the real `d_model` for more
  than one forward+backward.
* The λ 0.1 sizing is an arithmetic estimate against an assumed ~60-way softmax. The
  panel's real span count per row is not measured here.
* The two levers COMPOSE (a test pins that the term scores the centered state), but no arm
  in this panel runs both. That is a deliberate one-factor choice, not an oversight.
* `tests/test_tul_prefix_source.py::test_exit_is_pinned_to_the_pre_knob_values` fails 4/4
  on this machine at the tenth significant digit of `grad_sum`. It fails **byte-identically
  at `f89256d` without this change** (968.7720451522796 both ways), so it is a
  pre-existing float-summation pin, not a regression — but it is not fixed here either.
