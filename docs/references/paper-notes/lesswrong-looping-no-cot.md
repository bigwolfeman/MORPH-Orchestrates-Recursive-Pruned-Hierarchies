# Paper note: LessWrong, looping raises no-CoT capability more than general capability

- **Title:** Looping a transformer increases no-CoT capabilities more than general capabilities (Westover, Kuhn; LessWrong, 2026-11-12)
- **arXiv or URL:** https://www.lesswrong.com/posts/uxsDuWWHNTpKrsF6P/looping-a-transformer-increases-no-cot-capabilities-more
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: It states which capability a loop buys in a trained looped model (no-CoT reasoning) against generic loss (bpb).

## 2. What the paper shows

Read 2026-11-17 (WebFetch summary of the post; not a paper, one-seed blog-level evidence).

- 2026-11-17 (model): NVIDIA Dory 4B, trained at up to 4 loops, tested to 8.
- 2026-11-17 (Result 1): 1 to 8 loops raises NCRI (no-CoT reasoning index) by about 18 points (best prompt) or 13 (plain prompt). Bits per byte change slightly from 1 to 2 loops, then not at all. Mini-ECI (27 benchmarks, with CoT) rises at most about 1.5 points. Against parameters per token Dory is about 10 NCRI points above trend at 4 loops; gain levels off after about 6 loops.
- 2026-11-17 (caveat): a commenter says most of the NCRI jump is two of 19 domains. There is no identically trained non-looped control; Qwen3.5-4B is the outside comparison.
- 2026-11-17 (Result 4): recurrent KV sharing (all loops attend to the final loop's KV) gives most of the NCRI gain with no training: 60 vs 49 NCRI at 2 loops. Compression unchanged.
- 2026-11-17 (Result 3): under monitor pressure, recurrent-KV Dory hides reasoning in filler tokens faster. Single seeds.
- 2026-11-17 (filler): repeating the question helps only when looped, and more with more loops.

## 3. Our speculation

- 2026-11-17 Speculation: matches our view that generic loss gains little from the loop. The recurrent KV result is a free inference lever for a token loop; a span loop has no KV per pass to share.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 7 differs in kind: here the loop benefit exists in a model trained with it, but no timing is given.
