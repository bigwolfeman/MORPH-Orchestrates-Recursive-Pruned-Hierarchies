# Planned: a token-like read of the slot state, and the slot loop's horizon pair

Status: failure

Date: 2026-09-22 20:37 (frozen before any GPU step of any arm below; the 12-step smokes on the
builder branch are build checks, not runs). Note:
[`2026-09-22-slot-token-like-read.md`](../../../.agents/notes/proposed/architecture/2026-09-22-slot-token-like-read.md).
Ledger: [docs/9-26-TUL-run-history-IMPORTANT.md](../../../docs/9-26-TUL-run-history-IMPORTANT.md).
Wolfe's direction (2026-09-22): build on the positives; test a token-like read; the
horizon result is item 1. Runs start when Wolfe says so, not before.

## Question

Two questions from the ledger's "never built on" list.

1. Every loop that earns has the token CE read the loop's output at the token's own
   position, re-supplied at every block. Does the slot loop start to earn when its exit
   state is added to every token of the next span at every coda layer (`tul.bcast` plus
   `tul.bcast_layers: all`), on the strict spandec ruler?
2. The plain loop's value over its depth-1-trained twin grew from 0.004 (5k) to 0.067
   (20k). The strict slot loop's twin gap is +0.0020 [-0.0006, +0.0044] at 5k and was never
   read past 5k. Does the slot loop's value over its twin grow with horizon?

## Hypothesis

H1. The direct read is used (the gates leave zero) and the token CE then has a gradient
into the loop's exit at every token, so the passes get a token-like job. The prior is low:
the one-shot unpack read +0.0006 on 2026-09-10 (M-next family, coda-only restriction), and
the reader was never shown to be the limit on the strict ruler (`cond4`, 2026-09-13).
H2. The slot loop's twin gap at 20k is larger than at 5k, but by less than the plain
loop's (the plain loop's K-curve at 5k is 0.136 while the slot loop's is 0.0016: the
dependence that later becomes value is not there at 5k).

## Arms

All on `tul_slot_spandec_strict` (seq 1024, batch 6, seed 1, norm_match, the 1000-step
ramp, fixed-point 1.0), through the recon runner, at the commit that carries the builder
branch merged. Sweeps by `sweep_score.py` on 480 rows; pairing on the shared tokens
with 500 blocks, the 5k procedure.

| arm | config | steps | reading |
|---|---|---|---|
| `slot-spandec-strict-norecur-20k` | `tul_slot_spandec_norecur.yaml` | 20k | pairs with `slot-spandec-strict-20k` step_20000 (kept) at 5k, 10k, 15k, 20k |
| `slot-spandec-strict-bcast` | `tul_slot_spandec_strict_bcast.yaml` | 5k | K-curve, paired vs the ruler at 5k |
| `slot-spandec-strict-bcast-all` | `tul_slot_spandec_strict_bcast_all.yaml` | 5k first; 20k if queued | K-curve, paired vs the ruler and vs bcast; gates |
| `slot-spandec-strict-bcast-all-d1` | `tul_slot_spandec_strict_bcast_all_d1.yaml` | 20k, only with bcast-all-20k | the twin pair at 5k, 10k, 15k, 20k |

Queue order: norecur-20k (no build needed), bcast-all 5k, bcast 5k, then the 20k pair if
Wolfe queues it. 20k queue overrides as on the code-20k panel:
`training.steps=20000 training.ademamix_t_beta3=20000 training.ckpt_every=5000`.

## Predictions (frozen)

Slot-loop floor: token K1-K6 in [-0.0001, +0.0033] on twelve arms; the ruler reads
+0.0016 at 5k, CE@6 4.3474.

- **P-1 (the read is used).** `slot-spandec-strict-bcast-all` at 5k: mean |gate| over
  the coda layers above 0.05. **70 %.**
- **P-2 (bcast alone).** `slot-spandec-strict-bcast` token K1-K6 at 5k above +0.005
  with the CI clear of it. **20 %.**
- **P-3 (per-layer read earns).** `slot-spandec-strict-bcast-all` token K1-K6 at 5k
  above +0.005 with the CI clear of it: **30 %**; above +0.020: **10 %**.
- **P-4 (no CE price).** bcast-all paired depth-6 CE vs the ruler at 5k within +0.010
  (not worse by more): **80 %**; better by more than 0.010: **35 %**.
- **P-5 (horizon on the slot loop).** `slot-spandec-strict-20k`@6 minus
  `slot-spandec-strict-norecur-20k`@1, paired at 20k, above +0.020: **25 %**; above the 5k
  reading (+0.0020) with the CI clear of it: **45 %**.
- **P-6 (horizon with the read).** If the 20k pair runs: bcast-all-20k@6 minus
  bcast-all-d1-20k@1 above +0.020: **30 %**; larger than P-5's gap: **50 %**.
- **P-7 (cost).** bcast-all tok/s at step 200 at or above 0.90x the ruler's on the same
  runner: **80 %.**

## Verdict rules

The verdict rests on the token K-curve (sweep_score, CI), the paired CE, the worth
profile and the gates. No clause passes or fails on a cosine. A flat K-curve with used
gates (P-1 holds, P-3 fails) is FILED as: the read was not the limit; the slot loop's job
is. P-5 holding with P-3 failing is filed as: the slot loop's value is a horizon effect
that the 5k K-curve cannot see, and every 5k slot-arm verdict is qualified by it.

## Method

Builder branch `bcast-build` (opus agent, 2026-09-22): `tul.bcast_layers`, the gates,
the `_back_region` term, four configs, tests in `tests/test_tul_unpack.py`, Hydra
compose of all four, 12-step smokes of bcast and bcast-all. Merged by diff/apply and
committed with this file before any arm starts. Readouts: the runner's sweep at each
checkpoint, `worth_profile`, `slot_state`, and the pairing scripts under
`lab/divergence/`; artifacts under `../results/2026-09-22-slot-token-like-read/`
(json and runlog only).

## Results

Filed 2026-09-23 02:25. Three arms through the recon runner at ed5c5a5, all HEALTHY, no
NaN. Sweeps on 480 rows; pairing on 501,106 identical tokens in 490 blocks
(`paired_vs_ruler.py`). Artifacts: `../results/2026-09-22-slot-token-like-read/` (json and
runlog only). The bcast-all 20k pair (P-6) was not queued.

| reading at 5k | bcast (entry) | bcast-all | strict ruler |
|---|---|---|---|
| token K1-K6 | +0.0016 [+0.0014, +0.0019] | +0.0013 [+0.0011, +0.0015] | +0.0016 [+0.0013, +0.0019] |
| token K3-K6 | +0.0001 | +0.0002 | +0.0002 |
| paired depth-6 CE minus ruler@6 | -0.0166 [-0.0191, -0.0142] | -0.0123 [-0.0143, -0.0102] | 0 |
| bcast-all@6 minus bcast@6 | | +0.0044 [+0.0020, +0.0067] | |
| gates at 5k (coda 0..3) | | -0.40, -0.14, +0.03, +0.08 (mean abs 0.165) | |
| worth, zero / wrong_seed total | | 0.1925 / 0.0731 | 0.1865 / 0.0426 |
| wrong_seed worth at offset 0 | | 1.25 | 0.62 |
| tok/s at step 200 | | 11,087 | ~11.7k |

The horizon pair. `slot-spandec-strict-20k` (the looped ruler, 20k schedule) against
`slot-spandec-strict-norecur-20k` (the same config trained at slot depth 1, same schedule),
paired at every checkpoint:

| step | looped@6 minus twin@1 | looped@1 minus twin@1 | looped K1-K6 |
|---|---|---|---|
| 5000 | -0.0076 [-0.0103, -0.0047] | -0.0074 [-0.0101, -0.0045] | +0.0001 |
| 10000 | -0.0075 [-0.0103, -0.0049] | -0.0064 [-0.0091, -0.0038] | +0.0011 |
| 15000 | -0.0083 [-0.0107, -0.0057] | -0.0071 [-0.0095, -0.0045] | +0.0012 |
| 20000 | -0.0074 [-0.0104, -0.0045] | -0.0060 [-0.0090, -0.0031] | +0.0014 |

(The twin's own forced-depth curve is negative, -0.0196 at 5k and -0.0851 at 20k: a
depth-1-trained map applied more times gets worse. It is not a contribution reading.)

Scoring. P-1 HOLDS (0.165). P-2 FAILS. P-3 FAILS (both clauses). P-4 HOLDS on both
clauses, the second at its CI edge (-0.0123 [-0.0143, -0.0102]). P-5 first clause FAILS
(0.0074, bar 0.020); second clause HOLDS on its letter (+0.0074 [+0.0045, +0.0104] against
the 2026-09-12 5k reading +0.0020), but that 5k reading came from a different, 5k-schedule
twin pair; the matched 20k-schedule pair reads the same +0.0076 at 5k, so the letter holds
and the thing it was written to detect, growth with horizon, is absent. P-6 not run. P-7
HOLDS.

## Verdict

Failure on the question. A direct read of the slot loop's exit state is used and worth
about 0.012-0.017 nats to the coda at 5k, almost all of it at the span's first token, and
the loop's passes still add nothing: the K-curve is the ruler's to the fourth decimal
under either read, and re-adding the term at every coda layer is 0.004 WORSE than adding
it once. The read was not the limit.

The horizon pair is the new reading. Training with the slot loop is worth a constant
0.0075 nats over the depth-1 twin from 5k to 20k, with no growth, and most of it survives
when the looped model is itself evaluated at depth 1 (0.006-0.007). The plain loop's twin
gap grew 0.004 to 0.067 over the same span. So the slot loop's small value is a
training-time effect on the shared weights, present from 5k, and not a horizon effect the
5k K-curve was hiding. Every 5k slot-arm verdict on the ledger stands.

## Updated hypothesis

The slot loop's passes have no job that the token loss pays for, on web text, at any
horizon we have run and with any reader we have built. What the plain loop has and the
slot loop does not is not the read (this file), not the horizon (this file), not the
target family (the synthesis note), but the fact that its passes act on the token states
themselves, the positions the loss is charged on. The next design question is whether a
think-once slot loop can have passes that change something per token without running the
tokens through the core; the one tested form (bcast) cannot, because it adds one fixed
function of the exit state.
