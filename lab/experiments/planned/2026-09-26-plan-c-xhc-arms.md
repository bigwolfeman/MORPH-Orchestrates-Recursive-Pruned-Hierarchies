# Planned: do 16 routed streams in the slot loop (plan C) close the gap to plain

Status: planned

Date: 2026-09-26 09:48 (frozen before either arm's first 5000-step GPU step; 60-step smokes
ran during the build and read nothing about CE at 5k).
Parent: [`2026-09-26-slot-channel-arms-ab.md`](2026-09-26-slot-channel-arms-ab.md) (a1 is the
control here).
Note: [`../../../.agents/notes/proposed/architecture/2026-09-26-plan-c-xhc-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-26-plan-c-xhc-slot-loop.md).
Wolfe's go: 2026-09-26 ("I also have a C based around this https://arxiv.org/abs/2607.14530";
"Make sure this all runs end to end fot a b and c").

## Question

fp01 trails the plain model by 0.265 nats at 5k. a1 widens the slot channel to 4 cells, but
every cell is a projection of ONE exit state; at 5k a1 reads +0.2729, no better. Plan C gives
the slot loop's core a 16-stream carrier with a routed write (xHC, Zhang et al.), and the exit
maps the 16 streams onto the 4 cells, so each cell carries different content. Does distinct
content in the cells close the gap, and does the per-pass routing make the loop use depth?

## Arms (all 5000 steps, seed 1, built at 375c836, run at the master SHA queued)

| arm | config | differs from a1 (`..._fp01_pk4`) |
|---|---|---|
| C1 | `tul_slot_spandec_strict_e4probe_fp01_xhc` | `tul.xhc_streams: 16`, `xhc_active: 4`, `xhc_fixed: 2` |
| C2 | `tul_slot_spandec_strict_e4probe_fp01_xhc_ta` | C1 plus `tul.xhc_temporal_kernels: [2, 4, 8]` (causal depthwise conv over the slot axis, Gram-Schmidt) |

Deviation recorded in the note: on xHC models the gain hinge's two core applications run
under activation checkpointing (the first C1 smoke OOMed at 24 GB without it).

## Instruments

- Gap to plain: `paired_vs_ruler.py` of each arm's `core_depth_sweep` (480 rows) against the
  plain panel's 5k sweep at depth 6, token-paired by stream index. References at 5k: fp01
  +0.2650, a1 +0.2729, b +0.2501.
- Channel worth: `worth_profile.py` (192 rows), zero and shuffle totals. a1: 0.210 / 0.174.
- Coda K1−K6 from the sweep. a1: +0.0112 [+0.0103, +0.0120]; fp01: +0.0125.
- Router: `lab/divergence/xhc_router_hist.py` (one val batch, depth 6): the share of (slot, residual) cases at
  pass 5 whose routed pair differs from pass 0's. The 60-step smoke read 52 % (C1), 50 % (C2).
- The tripwire on each run's per-step probe.
- NOT available: `lxtul_e_stage2_score.py` builds its batches from the first arm's layout
  (prefix_k 2) and fails on prefix_k 4 models, as it did on a1; the notul pairing needs its
  output. `core_map_fd.py` does not replay the router and reads the stream-choice jump, so no
  fp32 map reading is planned.
- Amended 2026-09-26 09:53, before either arm's first 5000-step GPU step, instruments only
  (no prediction changed): per-cell worth by `worth_profile.py --modes
  zero,cellall,cell0,cell1,cell2,cell3` (192 rows) on C1, C2 and a1, built in 649f7e6
  (`cellall` equals `zero` exactly on tiny models; the check that it does on the 5k
  checkpoints is the TOTAL lines); and C1/C2 paired against fp01's and a1's 5k sweeps with
  `paired_vs_ruler.py`, which replaces the stage-2 scorer's fp01 pairing.

## Predictions

"Closes" means by at least 0.02 (the seed spread of cross-arm CE on this family is 0.0137).

- **C1-1.** C1's gap <= 0.245: **25 %.**
- **C1-2 (distinct content).** C1's gap <= a1's gap − 0.02 (<= 0.2529): **30 %.**
- **C1-3.** C1's channel worth (zero) >= a1's + 0.02 (>= 0.230): **40 %.**
- **C1-4.** C1's coda K1−K6 above a1's CI (> +0.0120): **40 %.**
- **C1-5 (router alive).** At 5k, >= 20 % of C1's pass-5 routed pairs differ from pass 0's:
  **60 %.**
- **C2-1.** C2's gap <= C1's gap − 0.02: **20 %.**
- **H.** Each arm's tripwire HEALTHY: **60 %** each.

## Verdict rules

Per arm: success if its gap clause holds (C1-1 for C1; C2-1 or a gap <= 0.245 for C2). C1-2
is the plan's own test (does distinct content beat copies of one state); it is reported for
C1 whether or not C1-1 holds. A router that fails C1-5 makes C1 an a1 with extra
parameters and is named as such. An arm that succeeds at 5k is resumed to 10k and paired
against plain 10k. No clause passes on a cosine.
