# Fan speed round 2: hinge rows, bf16 carrier, FP8 forward at 5k

Status: planned

## Question

Three more levers change the trained function: the gain hinge reads 2 of the 6 batch rows
(`model.slot_gain_rows: 2`), the HC carrier is stored in bf16 with fp32 math and an fp32 carrier
for the hinge (`model.hc_carrier_bf16`), and the dense ternary MLP forward GEMMs run in FP8 e4m3
with rowwise activation scales (`model.fp8_ternary_fwd`). Do quality, loop contribution and the
loop's stability hold with all three on, and what is the wall-clock gain?

## Hypothesis

Each lever adds noise of a size the run already tolerates: a 3x noisier hinge estimate that still
holds the gain near 0.9, one bf16 rounding per carrier store, and about 2.7 % relative rounding on
the MLP inputs. The 5k result sits near the speed base.

## Predictions

Rulers (480 rows, seed 1): locked recipe CE@6 3.8954, K1-K6 +0.0051; speed base CE@6 3.8985,
K1-K6 +0.0053. Wall clock (`lab/perf/bench_step.py`, 300 steps, skip 100): locked 228.7 ms/step,
speed base 213.5. Single-lever wall benches: bf16 carrier not yet benched on the wall clock
(about -10 ms on the host rate), hinge rows -6.0 ms, FP8 -3.6 ms.

| Quantity | Best guess | Pass if |
|---|---|---|
| CE@6 | 3.8990 | <= 3.9005 (locked + 0.005) |
| K1-K6 | +0.0050 | >= +0.0045 |
| wall ms/step, package vs speed base (same bench, interleaved) | 194 | <= 200 |
| `train/grad_norm` max / median over steps >= 200 (ruler: speed base 7.9, 8-token arm 7.7) | 8 | <= 30 |
| `tul/gain_est` over steps 1000-5000 (ruler: speed base 0.884-0.902) | 0.88-0.91 | inside 0.85-0.95 |

`preclip/total` (the divergence README's tripwire) is not logged on this recipe, so the
grad-norm ratio stands in for it; both are read by `abq/r2bench/stab.py` from the offline run.

Cell cos is a diagnostic only. If the package fails CE or K1-K6, the next step is one 5k run per
lever, starting with the bf16 carrier (the largest change to the loop's numerics).

## Method

Integration branch `perf/speed-r2` (master 77e7393c base 7b792ccb + perf/hinge-rows ec26b7ac +
perf/fp8-fwd 61b4fa45 + perf/bf16-carrier-v2 1a294303 + the trainer wall-rate fix dea1911d),
frozen as a snapshot in `/home/wolfe/morph-scratch/abq/tree-l`. Step 1: integration byte gate
(45 deterministic graph-step steps of `nowta_pointer_fast_spdcells`, every new key off), SHA must
equal `4a73ebf5...bd70d7`, or nothing else runs. Step 2: the three new GPU test files
(`test_hc_carrier_bf16.py`, `test_slot_gain_rows.py`, `test_fp8_ternary_fwd.py`). Step 3: wall
bench, speed base vs package, 2 repeats interleaved. Step 4: `nowta_pointer_fast_spdcells_speed2`,
5000 steps, seed 1, WANDB offline, then the 480-row depth sweep (`core_depth_sweep.py --depths 1,6
--rows 480 --batch 3`).
