# Parcae lean arm at 5k: speed with little CE cost and the loop still earning

Status: failure (written 2026-10-14, before any 5k run; WITHDRAWN the same day before any run)

## Question

Wolfe (2026-10-14) wants the LXTUL mechanism on Parcae published with three things shown against plain
Parcae: a sizable speed gain, little CE regression or an outright CE gain, and loop contribution; flat
GPT as a reference. Today strict LXTUL + pointer is 1.05x plain + pointer's step speed (270.7 vs 283.5 ms,
`morph-scratch/abq/parcae-speed/REPORT.md`), trails it +0.045 CE@6, and earns K1-K6 +0.047 vs +0.042
(`failures/2026-10-06-parcae-pointer-cellkey.md`).

The lean arm stacks the levers that held on MORPH at 5k (one-block span decoder + per-batch Poisson cell
count mean 2: `MORPH lab/experiments/successes/2026-10-13-flop-diet-5k.md`; ungraded fan + set reader:
MORPH note `2026-10-11-ungraded-fan-set-reader.md`) with the exact speed keys of `lxtul/compact`. Does the
stack keep CE and loop contribution on Parcae?

## Arm

`arm=lean` (`lxtul/configs/arm/lean.yaml`) with the frozen-cache gather and stratified draw OFF (on Parcae the
gather is slower, 276.3 vs 269.7 ms, and is a function change) and two exact keys ON:

| part | key | kind |
| --- | --- | --- |
| ungraded fan | `select: none` | function change |
| normalised pointer head, 4 heads | `pointer_heads: 4` | as LXTUL + pointer |
| cell count M = clamp(Poisson(2), 1, 4) per batch, eval at M = 4 | `cell_count: poisson` | function change |
| set-reader span decoder, one block | `spandec_reads_cells`, `spandec_layers: 1` | function change |
| pad-slot compaction, static depth, clip as Functions, CE on loss rows, packed span decoder | `compact_slots`, `static_depth`, `cot_clip_fn`, `ce_rows`, `spandec_packed` | exact or rounding-only |
| no per-pass checkpoints | `ckpt_passes: 0` | exact (memory only) |

## Runs (5k, 480 paired rows, `lxtul.train` defaults)

| run | seeds |
| --- | --- |
| lean (above) | 1, 2 |
| strict LXTUL + pointer, normalised | 2 (seed 1 is `lxtul-pointer-norm-5k`, 3.7608) |
| plain + pointer, normalised | 2 (seed 1 is `plain-pointer-norm-5k`, 3.7155) |

The seed-1 references ran at 505ba1d. The code since then adds keys that default off; the eval-doc exclusion
(default on) drops docs 50000-50405, which a 5k run never reaches (it ends near doc 24k). Then the head-off
leak check (`lxtul.mixture_off_eval`) on every new checkpoint.

## Predictions

1. Lean CE@6 (mean of 2 seeds) in [3.725, 3.765]: inside LXTUL + pointer's +0.005 and up to 0.036 better
   (the ungraded fan read 0.014 better than LXTUL without the head; the set reader was the MORPH winner) (65 %).
2. Lean gap to plain + pointer (pooled 2 seeds each) at most +0.040 (60 %).
3. Lean K1-K6 at least +0.025 (the ungraded fan without the head read +0.035, below LXTUL's +0.059) (65 %).
4. Lean trains at least 1.20x LXTUL + pointer's tok/s in the run log (the 300-step benches read 1.17-1.22x
   before the compaction key) (70 %).
5. Seed-2 references within 0.005 CE@6 of their seed-1 values (85 %).
6. Leak tripwire, not a prediction: a lean CE@6 more than 0.30 below LXTUL + pointer, or a head-off CE
   below the head-on CE, stops the panel and is filed as a leak (on 16 cycled bench batches the lean arm
   fit 0.27 vs 2.27 at step 300; the GPU causality tests passed with the lean keys).
7. No run raises or reports a non-finite loss.

## Method

Gate first (`morph-scratch/abq/pz-lean2/gate.sh`): geometry tests with the arm's keys, 300-step bench, FLOPs.
Then one queue script, one job at a time under the GPU lock. Amendments get a date and a reason here.

## Results

Not run. Withdrawn 2026-10-14 before any 5k run: Wolfe set the paper on the strict slot loop and the
work moved to the minimal arm (`planned/2026-10-14-parcae-minimal-slot-loop-5k.md`), which removes the MORPH
machinery instead of slimming it.

## Verdict

Withdrawn, no data. The 300-step benches stand as measured: lean 230.8 ms, 13.83 TFLOP/step vs LXTUL + pointer
269.7 ms, 21.49 TFLOP; plain + pointer with row compaction 243.7 ms.

## Updated hypothesis

The lean arm only slims machinery that Parcae does not need. The minimal arm tests removing it.
