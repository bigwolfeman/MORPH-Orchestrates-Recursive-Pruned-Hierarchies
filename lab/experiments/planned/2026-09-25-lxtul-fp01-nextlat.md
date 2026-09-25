# Planned: does a span-level NextLat term make the slot loop's exit a state the next slot follows from

Status: planned

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
