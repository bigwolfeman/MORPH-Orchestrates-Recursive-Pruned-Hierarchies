# LXTUL + the output-only pointer head on MORPH (with and without cell keys)

Status: planned (written 2026-10-06 23:11 CDT, before any run; tests/test_tul_pointer.py
11 passed, 3 sabotages caught; 60-step smoke of lxtul_pointer passed at c7424802)

## Question

On the Parcae testbed, ~90 % of strict LXTUL's CE gap to plain is copying across spans. An
output-only pointer head (pointer-generator / pointer sentinel; nothing enters a hidden state,
so only the loop crosses spans inside the model) took strict LXTUL from 4.1242 to 3.7608: from
+0.294 behind plain to 0.070 ahead of a plain model with no head, +0.045 behind a plain model
that has one too, with K1-K6 kept (parcae lxtul-testbed,
docs/experiments/failures/2026-10-06-parcae-pointer-cellkey.md). Cell keys (keys in earlier
spans also read that span's loop cells) added -0.0074 there. Wolfe (2026-10-06): "Its time to
port it to MORPH." Does the head close MORPH LXTUL's gap the same way?

Reference: `lxtul.yaml` 5k, gap to the plain ruler (plain-panel-norm-match, 480 rows) +0.272 /
+0.282 over two seeds, K1-K6 +0.0215 / +0.0172, 9.8k-12.4k tok/s
(.agents/notes/implemented/architecture/2026-10-04-lxtul-primary-candidate.md). MORPH's recall
probe (lxtul-hcsingle, 2026-10-06) puts far-bigram repeats at 19 % of tokens and +1.19 nats of
gap each: about 0.23 of a ~0.30 gap, so less of MORPH's gap is copying than Parcae's.

## Runs (5k, seed 1, the runner_steps readout chain, 480 rows)

| run | config |
| --- | --- |
| LXTUL + pointer | `lxtul_pointer` (tul.pointer_heads 4) |
| LXTUL + pointer + cell keys | `lxtul_pointer_cellkey` (+ tul.pointer_cell_key true) |

Then for each, the head-off leak check: `core_depth_sweep.py --pointer-off --depths 1,6`.

## Predictions

1. LXTUL + pointer: gap to the plain ruler at most +0.13 (a drop of at least 0.14 from +0.27)
   (70 %).
2. LXTUL + pointer: gap to the plain ruler at most 0 (30 %).
3. LXTUL + pointer: K1-K6 at least +0.015 (70 %).
4. Head off: CE@6 at least 0.10 worse than head on (the model hands copying to the head, as on
   Parcae) and no better than LXTUL's own CE (80 %).
5. Cell keys: CE@6 at least 0.003 lower than LXTUL + pointer (50 %).
6. LXTUL + pointer: tok/s at least 90 % of lxtul.yaml's at the same code (smoke: 11.6k) (65 %).
7. No run raises or trips the detonation tripwire.

## Method

`/home/wolfe/morph-scratch/abc/smoke_queue.sh <sha> lxtul-pointer:lxtul_pointer
lxtul-pointer-cellkey:lxtul_pointer_cellkey`, then runner_steps.sh; the leak checks run after
both DONE lines (`/home/wolfe/morph-scratch/abc/pointer_off_chain.sh`).

## Results

## Verdict

## Updated hypothesis
