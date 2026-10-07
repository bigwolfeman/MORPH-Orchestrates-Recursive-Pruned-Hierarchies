# LXTUL + the output-only pointer head on MORPH (with and without cell keys)

Status: mixed (filed 2026-10-07 02:12 CDT; written 2026-10-06 23:11 CDT, before any run; tests/test_tul_pointer.py
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

Artifacts: `lab/experiments/results/2026-10-06-lxtul-pointer/{pointer,pointer-cellkey}/`
(sweep, head-off sweep, gap, write worth, train curve). 480 rows, depth 6 unless named.

| run | CE@6 | gap to plain ruler | K1-K6 | head off CE@6 | head off K1-K6 | write worth (zero / shuffle) | tok/s | tripwire |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LXTUL + pointer | 3.9280 | -0.1497 [-0.1690, -0.1320] | +0.0155 [+0.0147, +0.0164] | 4.4097 | +0.0234 | +0.147 / +0.149 | 11817 | AMBIGUOUS, max 8.76e3 @ 2862 |
| LXTUL + pointer + cell keys | 3.9208 | -0.1569 [-0.1768, -0.1389] | +0.0098 [+0.0089, +0.0107] | 4.4078 | +0.0156 | +0.156 / +0.146 | 11368 | DETONATED, 4.18e8 @ 2297 |
| reference: `lxtul.yaml`, no head (2 seeds) | 4.3529 / 4.3621 | +0.272 / +0.282 | +0.0215 / +0.0172 | | | | 9.8k-12.4k | |
| reference: plain ruler | 4.0806 | 0 | | | | | | |

Paired cell keys minus no cell keys, the same 480 rows, bootstrap over 480 contiguous row blocks
of the per-token sweep files (both runs seed 1): depth 6 -0.0073 [-0.0091, -0.0053], depth 1
-0.0130 [-0.0151, -0.0109]. The interval covers row sampling only. The two `lxtul.yaml` seeds
differ by 0.009 at depth 6, so the cell-key difference sits inside training-seed noise.

The cell-key detonation is one spike train: pre-clip prelude grad norm ~1.5 jumps to 523-759 in
bursts over steps 2297-2310 and is back at 1.6 by step 2500. Val CE kept falling through it
(4.2802 at 2250, 4.2363 at 2500, final 3.9132 against 3.9202 for the arm without cell keys).

The SCORE / PAIR readouts exit 1 on both runs: they need `tul.spandec_parallel`, which no LXTUL
arm has had since 2026-10-06 02:10. This is a readout gap, not a pointer fault.

Per prediction:

1. Gap at most +0.13: HELD (-0.150).
2. Gap at most 0: HELD (-0.150, upper bound -0.132).
3. K1-K6 at least +0.015: HELD by a hair (+0.0155, lower bound +0.0147). The cell-key arm,
   which this prediction did not name, reads +0.0098.
4. Head off at least 0.10 worse and no better than LXTUL's own CE: HELD on both arms (head off
   costs 0.48 / 0.49; head-off CE 4.41 is 0.05 worse than LXTUL trained without a head).
5. Cell keys at least 0.003 lower CE@6: HELD on the point (-0.0073 paired), but inside seed noise.
6. tok/s at least 90 % of `lxtul.yaml`: HELD (11.8k / 11.4k against 12.4k max, 90 % = 11.2k).
7. No raise, no tripwire: FALSIFIED. The cell-key run detonated once and recovered; the run
   without cell keys read AMBIGUOUS (8.76e3, under the 1e4 rule).

## Verdict

Mixed: six of seven held, the stability prediction failed on the cell-key arm. The headline is a
strong positive. The pointer takes MORPH LXTUL from 0.27 behind the plain ruler to 0.150 ahead,
a 0.42-nat move, larger than on Parcae (0.36). The leak check says the gain lives in the head
(head off costs 0.48) and the body did not cheat (head off lands 0.05 worse than a model that
never had the head, the expected price of handing copying to the head).

Two caveats bound the reading. First, the ruler is plain WITHOUT a pointer. On Parcae plain + a
pointer was 0.045 ahead of strict + a pointer; MORPH has no plain + pointer run, so "LXTUL beats
plain" here means "LXTUL + a copy channel beats plain with none". Second, cell keys move pass 1
more than pass 6 (-0.013 vs -0.007): they make K1-K6 smaller by improving the shallow model, and
the run carrying them is the one that detonated. One seed each; the cell-key CE win is inside
seed noise.

## Updated hypothesis

The copy channel, not the slot loop, was the bulk of MORPH LXTUL's CE gap, as on Parcae. The
pointer without cell keys is the arm to build on: same CE within noise, a larger K1-K6, and no
detonation. Next: the coverage final phase (prereg 2026-10-07-lxtul-pointer-coverage, running),
then a plain + pointer MORPH run to put the ruler on equal terms, then a seed twin before any
10k run.
