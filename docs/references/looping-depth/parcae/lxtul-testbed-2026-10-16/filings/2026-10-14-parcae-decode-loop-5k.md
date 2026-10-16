# TUL decode loop: a per-token loop in the coda next to the slot channel

Status: failure (prereg written 2026-10-14 before any GPU run; filed 2026-10-15)

## Question

The plain loop's depth gain at 5k is per token and private (`failures/2026-10-14-plain-loop-boundary-depth-probe.md`:
a deep boundary token gives later tokens R 0.004; 90 % of a token's own gain survives with every neighbour
frozen at pass 1). A once-per-span slot loop cannot collect it, and every slot arm is flat. The decode
loop (Wolfe approved 2026-10-14) loops the last k coda blocks per token with Parcae's own recurrence and
keeps the slot loop as the cross-span channel. Does it collect the per-token gain, at what cost, and where
does it land against plain + pointer and the flat references on the same tokens?

## Arms (5k, seed 1, all on `minimal_fan`, keys in `lxtul:`)

| arm | looped coda blocks | mean / max / bptt | K/V |
| --- | --- | --- | --- |
| dec2 | last 2 of 4 | 4 / 8 / 4 | recomputed each pass |
| dec4 | all 4 | 3 / 6 / 3 | recomputed each pass |
| dec2, frozen K/V | last 2 of 4 | 4 / 8 / 4 | passes 2+ reuse pass-1 K/V |

References on the same 510,003 tokens: minimal fan 3.7442 (2 seeds, K1-K6 0), plain + pointer 3.7206 (2
seeds, K1-K6 +0.040), gpt8 + pointer 3.6996 (2 seeds), flat14 + pointer 3.6731 (1 seed). Bench (bench4,
batch 6): minimal 153.4 ms, plain + pointer row-compact 240.5 ms, flat14 + pointer 123.3 ms.

K1-K6 here is the DECODE loop's (the sweep varies the decode depth; the slot loop runs its trained depth).
CE@6 is at decode depth 6.

## Predictions

1. GPU tests pass and the key-off 300-step trace matches the unchanged tree as closely as the unchanged tree
   matches itself (80 %).
2. dec2 CE@6 at least 0.015 better than the minimal fan's 3.7442 (55 %).
3. dec2 decode K1-K6 at least +0.02 (60 %).
4. The best decode arm's CE@6 beats plain + pointer's pooled 3.7206 (35 %).
5. The best decode arm's CE@6 beats flat14 + pointer's 3.6731 (10 %).
6. dec2 bench step at most 0.85x plain + pointer row-compact (60 %).
7. Frozen K/V: CE@6 within +-0.010 of dec2 and a bench step at least 5 % faster (45 %).

## What each outcome changes

- 2 and 3 hold: the per-token gain is collectable inside TUL; next is its price against flat14 and the
  per-token loop's size (k, mean).
- 3 holds and 2 fails: the decode loop earns depth but pays it back at depth 1 (the tax refund); the
  design does not buy CE.
- Both fail: the per-token gain needs the core's width/depth, not two coda blocks.
- 7 holds: frozen K/V becomes the default (cheaper training, and a KV cache of one pass at decode).

## Method

`morph-scratch/abq/decode-loop/gpu_todo.sh` from the lxtul/decode-loop worktree: GPU tests, key-off trace
with a noise control, three smokes, a synced bench of six arms (`lxtul/bench_all.py`, the bench4 harness),
three 5k runs with head-off checks, `lxtul.paired_eval` over all runs on dec2's rows. CPU tests:
`test_decode_loop.py` 36 passed 2 skipped (CUDA), `test_bench_all.py` 20 passed, `test_minimal.py` 11
passed. Amendments get a date and a reason here.

## Results

Source: `morph-scratch/abq/decode-loop/gpu_todo.log` and `paired_dec.json` (filed 2026-10-15). Seed 1, 5k.

| arm | CE@1 | CE@6 | K1-K6 | head-off CE@6 | bench ms | TFLOP/step |
| --- | --- | --- | --- | --- | --- | --- |
| minimal fan (reference) | | 3.7442 | 0.0000 | | 180.4 | 13.35 |
| dec2 | 3.7632 | 3.7551 | +0.0081 [+0.0076, +0.0086] | 4.1612 | 224.4 | 18.27 |
| dec4 | 3.7692 | 3.7594 | +0.0097 [+0.0090, +0.0104] | 4.1650 | 240.0 | 20.07 |
| dec2, frozen K/V | 3.7658 | 3.7602 | +0.0056 [+0.0051, +0.0060] | 4.1662 | 218.1 | 17.64 |
| plain + pointer, row compact | | | | | 240.0 | 22.41 |
| gpt8 + pointer | | | | | 79.7 | 5.96 |

Paired at depth 6 against plain + pointer s2 (`plain-pointer-norm-5k-s2`, 3.7256) on dec2's rows:
gpt8 + pointer -0.0223 [-0.0244, -0.0201]; minimal fan +0.0190; dec2 +0.0295 [+0.0271, +0.0318];
dec4 +0.0338; dec2 frozen K/V +0.0346.

Prediction check:
1. FAILS. GPU geometry tests with the decode key on: 2 failed (live K/V, `test_coda_reads_the_winner_alone`,
   a loser cell is nonzero) and 8 failed (frozen K/V, KeyError in the coda-input hook). Both are the
   winner-select configuration, which no decode arm trained (the arms are on the ungraded fan), so the
   5k runs are not affected; the build should have raised on that combination. The key-off trace is not
   identical to the unchanged tree: first divergence at step 3 (the tree against itself: step 132), max
   abs difference 0.006 / 0.003 / 0.013 at steps 100 / 200 / 300 against a noise control of 0 / 0.005 / 0.004.
2. FAILS. dec2 CE@6 3.7551 is 0.011 WORSE than the minimal fan, not 0.015 better.
3. FAILS. Decode K1-K6 +0.0081, under +0.02.
4. FAILS. Best decode arm 3.7551 against plain + pointer 3.7206.
5. FAILS. Against flat14 + pointer 3.6731.
6. FAILS. dec2 bench 224.4 ms is 0.935x plain + pointer row compact (240.0), not at most 0.85x.
7. FAILS. Frozen K/V CE@6 +0.0051 against dec2 holds the CE half; its bench is 2.8 % faster, not 5 %.

## Verdict

Failure, 0 of 7. The decode loop earns a little depth (K1-K6 +0.006 to +0.010) and pays it back at
depth 1 and more: every decode arm ends 0.011 to 0.016 nats behind the minimal fan it is built on, at
1.21x to 1.33x its step time. Outcome row "both fail": the per-token gain needs the core's width and
depth, not two or four coda blocks looped.

## Updated hypothesis

Looping the coda does not collect the per-token gain the plain loop shows. The redesign moves to arms
where the loop output is part of the token state at a coarse resolution (SpiralFormer on Parcae,
`2026-10-15-parcae-spiral-5k.md`) and to the NCM code plan (`2026-10-14-parcae-span-ncm-5k.md`).
