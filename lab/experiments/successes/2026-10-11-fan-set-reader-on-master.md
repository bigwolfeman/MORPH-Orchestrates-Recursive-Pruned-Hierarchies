# Fan with the set-reader decoder on the master tree

Status: success

## Question

Does the fan + set-reader recipe (two runs on tree-h: CE@6 3.8951, K1-K6 +0.0056, cell cos
-0.28, `failures/2026-10-11-fan-set-reader-decoder.md`) reproduce on master, with the 2026-10-11
byte-identical speed levers (C3, E, Q, C4) and the graph-pool fix (e651c95b), and at what speed?

## Hypothesis

The levers are byte-identical on the training step, so the result is a third draw from the same
distribution, only faster.

## Predictions

- CE@6 in [3.885, 3.905], best guess 3.895.
- K1-K6 +0.003 to +0.008.
- Cell cos below 0.
- Trainer rate (third tok/s reading) at least 28,000 (tree-h read 23,950; the 260-step check on
  master read 34.0k at step 200).
- No out-of-memory at any validation.

## Method

`nowta_pointer_fast_spdcells`, 5000 steps, seed 1, from a frozen snapshot of the master commit
adding this file (`/home/wolfe/morph-scratch/abq/tree-j`), the A/B runner and its 480-row sweep.

## Results

Run 2026-10-11 in tree-j (master 58ef105e), one run, 480 rows. Artifacts:
`../results/2026-10-11-fan-set-reader-on-master/`.

| Run | Tree | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|---|
| master | tree-j | 3.9005 | 3.8954 | +0.0051 [+0.0047, +0.0055] | 0.181 | -0.28 | 29,209 |
| run 1 | tree-h | 3.9001 | 3.8949 | +0.0052 | 0.177 | -0.28 | 23,962 |
| run 2 | tree-h | 3.9013 | 3.8953 | +0.0060 | 0.180 | -0.28 | 23,933 |

Predictions:
- CE@6 in [3.885, 3.905]: 3.8954. Held.
- K1-K6 +0.003 to +0.008: +0.0051. Held.
- Cell cos below 0: -0.28. Held.
- Rate at least 28,000: 29,209. Held (22 % above tree-h).
- No out-of-memory at any validation: none in the log. Held.

## Verdict

Success. Three runs on two trees agree to 0.0005 CE@6 (3.8949, 3.8953, 3.8954). On master the
recipe trains at 29.2k tok/s, 22 % faster than on tree-h, and 3 % faster than the fan without
the decoder on tree-h (28.2k).

## Updated hypothesis

The byte-identical speed levers change speed only, as their gate says. The fan + set-reader
recipe is reproducible at 5k: CE@6 3.895, K1-K6 +0.005, cells distinct. It is the proposal in
`.agents/notes/proposed/architecture/2026-10-11-ungraded-fan-set-reader.md`; Wolfe decides.
