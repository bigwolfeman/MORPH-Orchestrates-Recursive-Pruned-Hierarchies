# Paper note: Flow Reasoning Models (FRM) and Fixed-Point Forcing

- **Title:** Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners (Helbling et al.)
- **arXiv or URL:** 2606.29150 (v3, 2026-09-01). Read: the PDF text via pdftotext (main text, Table 3, appendix headings). The HTML page returned 404.
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it
- 2026-11-17: Its Fixed-Point Forcing (FPF) fixes the train/inference gap for a recurrent carry. This is the same exposure gap our notes call teacher forcing as a bypass.

## 2. What the paper shows
- 2026-11-17 (Sec 2): discrete flow matching over the solution y, clue c held fixed. A categorical clean-prediction denoiser; the previous clean prediction s is fed back as a carry (self-conditioning). Every update gets direct flow-matching CE, no BPTT.
- 2026-11-17 (Sec 3.2, Fig 3-4): FPF builds the training carry by a detached rollout of the denoiser itself (start at a random earlier flow time, run the inference integrator to the supervised time), instead of one pass from ground truth. Loss and path unchanged. Table 3: rollout probability 0.5, rollout depth 16, 4 supervised states. So training pays up to 16 extra no-grad forwards in half the steps.
- 2026-11-17 (Sec 4.2-4.3, Table 2): Sudoku-Extreme 13.1 % plain flow, 32.6 % self-conditioning, 99.2 % FPF (3 seeds). Maze-Unique 96.2 to 98.0. Zebra self-conditioning 36.9 then FPF 99.9. Under plain self-conditioning gold CE RISES with depth while the residual shrinks (confident wrong fixed points, residual AUROC 0.50); under FPF it falls and residual AUROC is 1.00 (Fig 6).
- 2026-11-17 (Table 3): 7M to 25M parameters, 100K+100K to 300K+100K iterations. "44x fewer inference FLOPs" is at matched 98.7 % Sudoku solve rate vs a named next-best method (Fig 1).
- 2026-11-17: no language-model result. Language claims are cited from other flow-LM papers.

## 3. Our speculation
- 2026-11-17 Speculation: FPF is the recurrent-carry version of the fix our Poisson-depth draw gives for tied loops: train on the states inference makes. It matters for us only if a loop carry is fed back across passes or spans.

## 4. Corrections
- None recorded yet.

## 5. Where our findings disagree
- None recorded yet.

## 6. Triage
Score 1. Puzzle only. The exposure-bias lesson (self-conditioning can make confident wrong fixed points) is the one transferable point.
