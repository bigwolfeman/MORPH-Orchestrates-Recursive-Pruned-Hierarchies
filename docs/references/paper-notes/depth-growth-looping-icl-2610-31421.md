# Paper note: How Depth Growth and Looping Shape In-Context Learning

- **Title:** How Depth Growth and Looping Shape In-Context Learning in Language Models (Kapl, Maile, Bauer, von Oswald; 2026)
- **arXiv or URL:** 2610.31421
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: It is the only paper in this group with training-time curves for a loop effect (ICL emergence), and it gives a LIDAS training-FLOP saving at 3B.

## 2. What the paper shows

Read 2026-11-17 via arXiv HTML (first 100k of 162k characters).

- 2026-11-17 (Table 1, App A): dense 360M/32L (209.7B tokens), 1.7B/24L (419.4B), 3B/36L (100B). 3B compare: dense, LIDAS-grown, Loop 10-8x2-10 (about 2.45B params). One LM seed per arm.
- 2026-11-17 (cost, Sec 3.1): growing saves an estimated 26 % of training FLOPs at 3B and 22 % at 360M and 1.7B, at matched final inference FLOPs.
- 2026-11-17 (loss, Fig 1a,b): grown and looped models both worse on tokens at index 8 or less, with a larger early-to-late loss drop than dense. LIDAS beats dense after about position 1000. The looped model stays above dense NLL to the end.
- 2026-11-17 (GSM8K, Fig 1c): LIDAS about +7 % over dense at 8 shots. Loop keeps an edge that shrinks at 8 shots (text is inconsistent on this).
- 2026-11-17 (Table 2, 5-shot 9-task MC / MMLU): 3B dense 61.33 / 39.15, LIDAS 62.89 / 39.28, Loop 60.53 / 38.32. So the loop is not ahead of dense on generic benchmarks at 3B.
- 2026-11-17 (WHEN, Fig 3b, Sec 3.4): on Omniglot ICL-only, ICL emerges earlier for looped, grown and 4-layer dense than for 12-layer dense; all near 100 % by 1M steps. On fixed labels, looped and grown models keep ICL and show lower in-weights learning.
- 2026-11-17 (induction heads, Sec 4): LIDAS gains induction heads that trace to four ancestors in the 4-layer start model; the loop ends near dense. Ablating induction heads hurts grown and looped models more.

## 3. Our speculation

- 2026-11-17 Speculation: early ICL emergence is a toy-task result (Omniglot). It is the one piece of evidence that a loop can earn EARLY, but only on a task where shallow compute is cheap to learn. It is weak evidence against our item 7 and not a test of it.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 7: partial. Here the loop helps early on a synthetic ICL task; on token loss the loop stays worse than dense.
