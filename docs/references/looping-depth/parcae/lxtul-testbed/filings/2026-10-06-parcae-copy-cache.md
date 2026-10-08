# Parcae copy cache and the strict-then-open switch

Status: failure (written 2026-10-06 16:12 CDT, before the probe or any run; unit and
geometry tests only)

## Question

The gap decomposition (`failures/2026-10-06-parcae-gap-decomposition.md`) put 93 % of strict
LXTUL's +0.294 gap on the strict mask, and the gap probe put 86 % of it on tokens whose bigram
occurred in an earlier span (median 95 tokens / 5 spans back, so a small window cannot reach
them). Open geometry closes the gap but flattens the loop (K1-K6 +0.0042). Wolfe (2026-10-06):
"try option 3", the exact-copy cache, and "start it without open geometry so it is forced to
learn to handle closed geometry then move to open geo and see what happens".

1. Does an exact-copy cache, which carries token identity into the OUTPUT only and never into a
   hidden state, recover the copy loss while the strict loop keeps its job?
2. Does a loop learned under the strict mask survive when the mask opens?

## Mechanism

`lxtul/cache.py`: for token i, every earlier token j < i with the same bigram key
(x_{i-1}, x_i), or the same unigram key x_i, proposes y_j. p = a0 p_model + a1 p_bi + a2 p_uni,
a = softmax of a linear gate on the token's coda output and the match counts (sources with no
match get weight 0). Tests: `lxtul/tests/test_cache.py` (brute force, causality, mixture,
per-token CE gradients; a j <= i sabotage is caught) and `test_geometry.py` (end-to-end
causality with the cache; a planted repeat scores cheaper).

## Runs

| run | config | what |
| --- | --- | --- |
| cache probe | `python -m lxtul.cache_probe` | the cache with a CONSTANT gate (grid) on trained checkpoints, no training |
| strict + cache | `arm=lxtul_cache` | trained gate, strict geometry, 5k |
| strict then open | `arm=lxtul_switch` | strict to step 2500, open to 5000 |
| cache, strict then open | `arm=lxtul_cache_switch` | both |

References (480 paired rows, 5k): plain 3.8303; strict LXTUL 4.1242 / 4.1252, K1-K6 +0.0585 /
+0.0794; open LXTUL 3.8508, K1-K6 +0.0042; gpt8 3.8086.

## Predictions

1. Probe: the untrained cache recovers at least 0.10 nats on strict LXTUL (70 %).
2. Probe: on plain it recovers under 0.03 (plain already copies with induction heads) (75 %).
3. Strict + cache: gap to plain at most +0.15, half the strict gap (55 %).
4. Strict + cache: K1-K6 at least +0.03, the loop still used (60 %).
5. Strict then open: final CE (open eval) within 0.03 of open LXTUL's 3.8508 (65 %).
6. Strict then open: final K1-K6 at least +0.014, above open LXTUL's +0.0042 by 0.01: the
   strict start leaves loop use in place (40 %).
7. Strict then open, re-scored under the STRICT mask at the end: CE at most 4.30 (it keeps
   working strict after 2500 open steps) (50 %).
8. Cache, strict then open: the cache adds under 0.02 over strict-then-open (60 %).
9. No run raises.

## Method

One job at a time under the GPU lock: the probe on plain-5k, lxtul-5k, lxtul-open-5k, gpt8-5k;
then 30-step smokes of the three arms; then the three 5k runs (`panel2.sh`), seed 1, the
round-1 recipe. The switched arms also write `sweep_strict.json` (scored with the strict mask).

## Method amendment (2026-10-06 16:15 CDT, before any 5k run)

The first strict + cache smoke failed: the cache branch named its token mask `ok` and shadowed
the forward's `ok`, which the latent-selection loss reads. Fixed in e25088c (test_geometry now runs
a training forward with labels); every 5k run below is at e25088c. The probe ran at e0fad1f;
the cache path it uses did not change.

## Results

480 paired rows, seed 1, 5000 steps. Readout `results/2026-10-06-parcae-copy-cache/readout_vs_plain5k.json`.

Cache probe (constant gate, no training; `cache_probe.json`):

| model | own CE | + cache | gain |
| --- | --- | --- | --- |
| plain | 3.8303 | 3.7999 | +0.0304 |
| strict LXTUL | 4.1243 | 3.9269 | +0.1974 |
| open LXTUL | 3.8508 | 3.8179 | +0.0330 |
| gpt8 | 3.8086 | 3.7803 | +0.0283 |

Trained arms:

| run | CE@1 | CE@6 | K1-K6 | gap to plain 5k | strict re-score CE@6 | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 3.876 | 3.8303 | +0.0462 | - | - | 21.8k |
| strict LXTUL | 4.183 | 4.1242 | +0.0585 | +0.2939 [+0.2756, +0.3151] | - | 24.4k |
| open LXTUL | 3.855 | 3.8508 | +0.0042 | +0.0205 [+0.0176, +0.0236] | - | 24.3k |
| strict + cache | 3.901 | 3.8518 | +0.0491 [+0.0475, +0.0508] | +0.0215 [+0.0110, +0.0306] | (strict) | 22.4k |
| strict then open | 4.090 | 4.0600 | +0.0304 [+0.0290, +0.0317] | +0.2296 [+0.2139, +0.2477] | 4.2224 | 24.0k |
| cache, strict then open | 3.858 | 3.8287 | +0.0290 [+0.0278, +0.0302] | -0.0017 [-0.0115, +0.0067] | 3.9212 | 22.2k |

Leak check (`cache_off_eval.log`): the strict + cache checkpoint scored with the cache OFF reads
CE@1 4.1750, CE@6 4.1182, K1-K6 +0.057, where strict LXTUL reads 4.183 / 4.1242 / +0.0585. The
model learned no copy path of its own; the 0.266 nats come through the output mixture, and the
loop's own depth use is unchanged.

Validation curves (60 rows, every 500 steps): strict then open sat 0.21 behind open LXTUL at the
switch (4.585 vs 4.372) and gained the same 0.48 nats in the second half; the switch gave no jump.

| # | prediction | result |
| --- | --- | --- |
| 1 | probe recovers >= 0.10 on strict LXTUL | holds (+0.197) |
| 2 | probe recovers < 0.03 on plain | fails narrowly (+0.0304) |
| 3 | strict + cache gap <= +0.15 | holds (+0.0215) |
| 4 | strict + cache K1-K6 >= +0.03 | holds (+0.0491) |
| 5 | strict then open within 0.03 of open LXTUL | fails (4.0600, +0.209) |
| 6 | strict then open K1-K6 >= +0.014 | holds (+0.0304) |
| 7 | strict then open, strict re-score <= 4.30 | holds (4.2224) |
| 8 | cache adds < 0.02 over strict then open | fails (adds 0.231) |
| 9 | no run raises | holds |

## Verdict

Failure on 2, 5 and 8; the main hypothesis holds. The CE gap of strict LXTUL is the copy channel:
an output-only exact-copy cache takes it from +0.294 to +0.022 at 5k while the loop keeps its
depth use (K1-K6 +0.049; +0.057 with the cache off). Strict + cache reaches open geometry's CE
(3.8518 vs 3.8508) with 12 times its loop use. With the cache, strict then open ties plain
(-0.002, CI across 0) and keeps K1-K6 +0.029, and still scores 3.921 under the strict mask.
Without the cache, the strict start costs 0.21 nats at 5k and the open half does not recover it.

Prediction 8 failed for an informative reason: the cache was worth as much after the opening as
in the strict arm, so open-geometry tokens trained from a strict start do not copy well by
themselves within 2500 steps; the cache does that job for them.

## Updated hypothesis

The copy cache is the gap lever and leaves the loop alone. Open questions: (1) a seed twin of
strict + cache; (2) plain + trained cache, for the gap with the cache on both sides (the probe
suggests about +0.05); (3) 10k, to see whether the gap stays closed; (4) the MORPH port, where the
loop earns less and the same probe should read first.
