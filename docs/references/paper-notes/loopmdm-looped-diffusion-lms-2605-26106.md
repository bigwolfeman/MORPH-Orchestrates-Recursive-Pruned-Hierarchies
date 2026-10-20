# Paper note: Looped Diffusion Language Models (LoopMDM)

- **Title:** Looped Diffusion Language Models (Sanghyun Lee, Chunsan Hong, Seungryong Kim, Jonghyun Lee, Jongho Park, Dongmin Park; KAIST, KRAFTON, Berkeley)
- **arXiv or URL:** 2605.26106 (v1, 2026-05-25)
- **Local paper text:** docs/references/looping-depth/loopmdm-looped-diffusion-lms-2605-26106/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It loops a SMALL block (2 layers) with a per-step uniform loop draw and compares at matched TRAINING FLOPs, on OpenWebText at 170M. Those are our levers: a partial core, a loop-count draw, and a FLOP-matched ledger. The model is a masked diffusion LM, not an AR LM.

## 2. What the paper shows

Read: main text, all tables in Sec 4, limitations. Appendices A to D were not read.

- 2026-10-20 (Sec 3, Eq 4 to 6): head / looped mid-block / tail. Only the mid-block repeats, S ~ U{1..Smax} each training step, loss on the final loop output. Per-step cost F = F_base + (E[S]-1) n_m F_layer; the baseline is matched on total training FLOPs by giving LoopMDM fewer steps.
- 2026-10-20 (Sec 4.1, Fig 2): 170M MDM, 12 layers, loop layers 1 and 2 (zero-based), Smax=12, baseline 1M steps batch 512. For S >= 6 LoopMDM matches the baseline's final test NLL with 3.34x, 2.95x, 2.34x fewer training FLOPs on LM1B, OpenWebText, FineWeb-Edu. At S=1 it is WORSE than the baseline. S=24 (beyond Smax) still improves.
- 2026-10-20 (Table 4, LM1B, 0.1M steps, baseline NLL 3.798): looping layer position 1 or 2 gives 3.729; position 0 gives 3.744; looping 1 or 2 layers gives 3.743 or 3.729; looping more than 6 layers is WORSE than the baseline (7 layers 3.857, 12 layers 3.933); Smax 12 gives 3.729, Smax 24 gives 3.800.
- 2026-10-20 (Tables 1 to 3, FineWeb-Edu checkpoints): downstream average 40.0 (MDM) vs 39.4 (S=1), 40.7 (S=6), 41.4 (S=12), 5 seeds; gen-PPL falls with S at every step budget.
- 2026-10-20 (Sec 4.2, Fig 3, TinyGSM 125M): +8.5 points GSM8K at S=16 over the 14-layer baseline; it beats a 21-layer baseline of equal per-step FLOPs by +2.7 (S=8) and +3.4 (S=16).
- 2026-10-20 (Sec 4.3, Table 5): a Sudoku probe with a forced left-to-right order: 1-layer MDM 10.9 %, looped model at S=1 10.6 %; errors fall 24 to 7 to 0 over S=1,2,3 on one shown puzzle. Gains concentrate at intermediate noise levels. Adaptive stop on relative hidden change (eps 0.1) cuts the mean loop from 12 to about 5 with downstream 41.4 to 41.3.
- Failure modes and limits stated: too many looped layers, too large Smax, loop near input or output all lose to the baseline; medium scale only; the stopping rule is hand set. The theory example (k-Clique with masked positions as workspace) is a sketch.

## 3. Our speculation

- 2026-10-20 Speculation: the closest match to our work is "loop few layers, draw the count, pay in FLOPs". Here two layers beat twelve. Our core loops six layers at a Poisson depth. A core-width ablation (loop 2 of the 6 core layers, rest once per pass) at a matched FLOP ledger would test whether the small block is the better looped unit, and it is a speed lever as well.
- 2026-10-20 Speculation: S=1 worse than the baseline while S >= 6 wins mirrors our finding that a loop earns only when drawn deep. It is evidence for the draw, not for a curriculum (rule: no depth curricula).
- 2026-10-20 Speculation: the paper's gains sit in a model where unresolved (masked) positions are a workspace that other positions attend to. That is the mechanism our span states lack ("necessary is not deep"): a diffusion LM has many partly unknown positions at every step, an AR LM at 21-token spans has few.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Their gain is large (3x FLOP saving) at matched FLOPs, ours at 5k steps is K1-K6 +0.0377 on plain. Different objective (NELBO over all masked positions, bidirectional) so a loop has more to refine per step. Not a contradiction, and no AR result is offered.
