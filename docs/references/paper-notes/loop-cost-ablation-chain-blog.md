# Blog note: Loop-Model FLOPs and Memory in an Ablation Chain

- **Title:** Loop-Model FLOPs and Memory in an Ablation Chain (Benhao Huang, Husky'Log). Type: blog, 2026-04-19, 8470 words.
- **arXiv or URL:** https://huskydoge.github.io/husky-blog/posts/recursive_models/loop-cost/
- **Local paper text:** docs/references/looping-depth/loop-cost-ablation-chain-blog/ (paper.txt, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: A cost model of BPTT variants for loop models (tied weights, per-step losses, detach, instant update, truncation, checkpointing). We use truncated and full BPTT and chase ms/step. Read in full (text extract; the math is in raw TeX and the charts are flattened to number lists).

## 2. What the post shows

- 2026-10-20: a seven-step chain, each step changing one thing: untied + final loss; tied; per-step losses; detach between steps; instant update (optimizer step per outer step); internal truncation (gradient only through the last inner refinement); checkpointing. It writes FLOP and activation-memory formulas for each (symbols c_l, c_J, c_h, c_B, c_theta, c_u).
- 2026-10-20 (toy benchmark, CPU, tanh RNN cell, batch 32, 32 outer steps x 6 inner, d=256, Adam): profiler FLOPs (M) 4038 untied, 4038 tied, 4428 per-step losses, 4298 detach. Tracked peak memory 451.85 MB to 69.10 MB from tying (activations unchanged at 66.09 MB). Detach + streaming backward: activations 69.00 to 2.16 MB at 4302 M FLOPs. Instant update 134.45 M per step, plus truncation 71.40 M, checkpointing raises it to 121.83 M.
- Key claim: detach saves little compute because the local loss and parameter-gradient work dominate; streaming the backward after each detached step saves memory, not FLOPs.
- Limits, stated by the author: it measures cost, not accuracy; FLOPs are profiler estimates of operator FLOPs; a toy RNN, not a transformer; one run. No LM result, no looped-LM loss comparison.

## 3. Our speculation

- 2026-10-20 Speculation: nothing to run from it. The post supports a bookkeeping point we already hold: truncating BPTT (our `bptt_depth`) saves activation memory more than FLOPs, so a speed gain must come from fewer forward passes, kernels, or launch count, not from the backward alone. It is a sanity reference for a cost ledger, nothing more. Triage 1.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: None. Our rule bans gradient checkpointing unless asked, and the post's checkpoint row (FLOPs 4428 to 6042 M, about 1.4x; peak memory 79.76 to 16.01 MB, about 5x) is in line with that cost trade.
