# Planned: does a span-level NextLat term make the slot loop's exit a state the next slot follows from

Status: failure

Date: 2026-09-25 12:52 (frozen before the arm's first 5000-step GPU step; a 400-step
smoke for memory and speed was running, its NextLat readings not yet read).
Parent: [`../successes/2026-09-25-lxtul-nofp-followup.md`](../successes/2026-09-25-lxtul-nofp-followup.md).
Wolfe's go: 2026-09-25 ("Yes I agree with that", on a span-level NextLat arm after the
follow-up, if the positive replicated; it did).
Note: [`../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md`](../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md).

## Question

fp01 (the strict e4probe slot loop, fixed-point term 0.1) earns coda K1−K6 +0.0126. No term
ties the exit state `z_s` to the next slot's `z_{s+1}`. If a transition that reads `z_s` and
span s+1's tokens is trained to predict `z_{s+1}` (NextLat, arXiv 2511.05963, at the span
level), (1) does the loop use depth more, and (2) can the transition draft the next slot
state well enough for the parallel head to read it?

## Hypothesis

The term's gradient reaches `z_s` only. It asks the exit to carry what the NEXT exit will
hold, beyond span s+1's own content: a belief state. Building a state that summarises the
past for a future two spans away is work a looped map can do over passes, so depth use
grows. The transition is small (one GRU) and the target is stop-graded, so it learns a
draft better than the no-change guess but does not reach the loop's own state.

## Arm

| arm | config | differs from fp01 |
|---|---|---|
| lxtul-e4probe-fp01-nextlat | `tul_slot_spandec_strict_e4probe_fp01_nextlat` | `tul.nextlat_weight 1.0` (and the run name) |

Seed 1, 5000 steps, the fp01 recipe otherwise. The reference is fp01 itself (same seed,
same data order), read from its filed step-5000 checkpoint.

## Instruments

- `lxtul_e_stage2_score.py --ref fp01` (480 rows): coda K1−K6 and par K1−K6 under the
  4-rollout mixture, exit separation at depth 6, paired coda mix @6.
- `core_depth_sweep.py` (480 rows): K1−K6 read a second way.
- Val log at step 5000 (the val pass scans `nextlat*` since `c05269e`):
  `val/nextlat_l1`, `val/nextlat_copy_l1`, `val/nextlat_draft_gap`,
  `val/nextlat_draft_ce`, `val/nextlat_true_ce`.
- `tripwire_sustained.py` on the per-step probe (abort rule 1e4).
- `core_map_fd.py` fp32 on the final checkpoint (a diagnostic, not scored).

## Predictions

- **N-1 (still earns).** Coda K1−K6 >= 0.005, CI clear of zero: **65 %.**
- **N-2 (earns more, past seed noise).** Coda K1−K6 >= 0.0172 (fp01's +0.0126 plus the
  0.0046 spread between the two nofp seeds): **20 %.**
- **N-3 (the transition learns).** `val/nextlat_l1` <= 0.8 x `val/nextlat_copy_l1` at step
  5000: **70 %.**
- **N-4 (the draft reads).** `val/nextlat_draft_gap` <= 0.10 nats per head token: **35 %.**
- **N-5 (stable).** Tripwire HEALTHY: **65 %.**

Not scored: paired coda CE against fp01. The follow-up measured a 0.0137 spread between two
seeds of one arm on that statistic, so a one-seed cross-arm CE is unreadable. It is
recorded.

## Verdict rules

Success if N-2 and N-5 hold (the question is whether the term moves depth use). If N-2
fails but N-3 and N-4 hold, file a failure on depth use and record the drafter as a separate
positive. No clause passes on a cosine (`nextlat_cos` is a diagnostic).

## Method

Chain in `/home/wolfe/morph-wt-map` at the commit that files this prereg; checkpoints to
`/home/wolfe/morph-to/checkpoints/morph/lxtul-e4probe-fp01-nextlat/`. One GPU job under
`gpu.lock`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A 400-step smoke at
`1748dd9` measured the memory and speed before this file was frozen (numbers go into
Results, not into the predictions). Readouts: the map, the scorer with `--arm fp01=...`
and `--arm nextlat=...` and `--ref fp01`, the sweep, the tripwire.

## Results (filed 2026-09-25 15:20)

Chain at `444a574` in `/home/wolfe/morph-wt-map`, 5000 steps, exit 0, 5373 tok/s (fp01
6178, so 0.87x), peak 18.5 GB (fp01 16.65). Artifacts:
[`../results/2026-09-25-lxtul-fp01-nextlat/`](../results/2026-09-25-lxtul-fp01-nextlat/).
Scorer `--ref fp01`, 480 rows, self-check max |dev| 1.1e-6; the sweep reads the same K1−K6.

| reading | fp01 | fp01 + NextLat |
|---|---|---|
| coda K1−K6 [95 % CI] | +0.0126 [+0.0119, +0.0134] | **+0.0010 [+0.0007, +0.0013]** |
| par K1−K6 | +0.1190 | +0.0201 |
| coda mix @1 / @6 | 4.3534 / 4.3408 | 4.3451 / 4.3441 |
| paired coda mix @6 (arm − ref) | | +0.0033 [+0.0008, +0.0060] (seed noise, not scored) |
| exit separation @6 | 5.61 | 5.23 |
| eval map fp32, passes 1-5 | 0.883–0.897 | 0.882–0.887 |
| tripwire | HEALTHY, max 475 | HEALTHY, max 23 @1486 |
| `val/nextlat_l1` / `val/nextlat_copy_l1` | | 0.0010 / 0.0009 |
| `val/nextlat_draft_gap` | | 0.0092 (draft CE 7.0744, true 7.0652) |

The train history (wandb `morph-tul/a9mjfvck`): `tul/nextlat_copy_l1` 0.388 at step 0,
0.0050 at step 200, 0.0013–0.0023 from step 800 to the end. The no-change guess became
near perfect inside the LR ramp. The transition beat it by about 15-20 % on train batches
and lost to it by 11 % on held-out rows.

### What the term did: a collapse probe

`lab/divergence/nextlat_copy_probe.py` (new) reads the exit readout `z = _readout(h_slots)`
that the parallel head grades, at forced depths 1 and 6, on the same 96 held-out rows for
four arms (`copy_probe.json`). `diff_rel` is RMS(z_{s+1} − z_s) / RMS(z); `cos_xrow` is the
cosine between the same slot index in two DIFFERENT rows.

| arm | depth | copy_l1 | cos_next | cos_far (s, s+4) | cos_xrow | diff_rel | rank_all |
|---|---|---|---|---|---|---|---|
| fp01 | 1 | 0.0016 | 0.9988 | 0.9987 | 0.9988 | 0.047 | 14.6 |
| fp01 | 6 | 0.0377 | 0.9710 | 0.9705 | 0.9682 | 0.230 | 25.2 |
| **nextlat** | 1 | 0.0003 | 0.9998 | 0.9997 | 0.9997 | 0.021 | 20.6 |
| **nextlat** | 6 | 0.0007 | 0.9995 | 0.9994 | 0.9994 | **0.032** | 39.9 |
| map (term 1.0) | 6 | 0.0979 | 0.9249 | 0.9208 | 0.9147 | 0.377 | 33.7 |
| nofp | 6 | 0.0515 | 0.9600 | 0.9590 | 0.9535 | 0.266 | 19.7 |

The probe's depth-6 copy_l1 on the NextLat checkpoint (0.0007) agrees with the val log's
sampled-depth 0.0009.

Three facts, measured:

1. **In every arm the exit readout is one shared vector plus a small per-slot part.**
   cos_xrow is 0.91–0.97 at depth 6: a slot of one row points almost the same way as the
   same slot of an unrelated row.
2. **Neighbours are no more alike than strangers.** cos_next ≈ cos_far ≈ cos_xrow in every
   arm. The exit states of fp01 carry no sequence structure a transition could exploit
   beyond the common mode.
3. **NextLat shrank the per-slot part 7x and removed the loop's motion across slots.**
   diff_rel at depth 6 is 0.032 against fp01's 0.230. From depth 1 to 6 the per-slot part
   grows 0.047 → 0.230 on fp01 and 0.021 → 0.032 under NextLat. The map is unchanged
   (0.88), so the earning died with the map in place, as in every earlier arm.

## Clause by clause

- **N-1 FAILS.** Coda K1−K6 +0.0010, below 0.005.
- **N-2 FAILS.** Below 0.0172.
- **N-3 FAILS.** `val/nextlat_l1` 0.0010 is above the no-change guess 0.0009, not 20 %
  below it.
- **N-4 HOLDS, vacuously.** The draft gap is 0.0092 nats, but the states it drafts
  between differ by 3 % of their RMS, so the drafter had almost nothing to predict.
- **N-5 HOLDS.** Tripwire HEALTHY, max 23.

## Verdict

**Failure.** The term did not make the exit a belief state. It made the exit a constant.
The gradient of `SmoothL1(T(z_s, span), sg(z_{s+1}))` reaches `z_s`, and every `z` is a
source for its own successor, so the cheapest descent is to shrink every slot's deviation
from the common mode until the identity-init transition's copy is exact. The stop-grad on
the target is the paper's only guard, and it is not a guard here. NextLat's `h_t` also
carries the full next-token CE from the same state; this exit state's readers are the
coda (weak: the loop is worth about 0.01 nats) and a DETACHED probe head (no pull at all).
Nothing held the per-slot content up against the term.

## Updated hypothesis

A latent-prediction term on the loop's exit collapses it unless something holds the
per-slot content up: a strong reader on the same state, an EMA or frozen target, or a
variance floor. The term then acts like the fixed-point term across slots: it charges the
loop's motion. Two facts outlive this arm and bear on any slot-state objective: the exit
readout is 91–97 % one shared direction in every arm, and consecutive exit states are no
more alike than states from unrelated rows. The next step is Wolfe's call; the options are
in the note [`../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md`](../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md).
