# Paper note: Do Depth-Grown Models Overcome the Curse of Depth?

- **Title:** Do Depth-Grown Models Overcome the Curse of Depth? An In-Depth Analysis (Kapl, Angelis, Hoeppe, Maile, von Oswald, Scherrer, Bauer; 2025-12 id)
- **arXiv or URL:** 2512.08819
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: It gives the MIDAS training-speed numbers and the LIDAS variant, and explains the gain as better use of depth.

## 2. What the paper shows

Read 2026-11-17 via arXiv HTML (first 100k of 133k characters). The abstract said MIDAS; the page text says LIDAS is the "lightweight modification".

- 2026-11-17 (speed): the MIDAS reproduction gives 1.29x training speed (intro), 29 % speedup (Sec 3.2, Table 7), and LIDAS 23 % lower training cost (Sec 5). FLOPs not reported. Compute: baseline 1.7B on 128 A100 for 4.5 days.
- 2026-11-17 (setup): SmolLM-v1 360M (32L) and 1.7B (24L), 200B and 400B tokens, block size 4, PROP-1 schedule, 170k growth steps of 200k. Baselines: plain, and LayerNorm Scaling. A second 360M seed is in App D.2.
- 2026-11-17 (Table 1): 1.7B NLL baseline 1.96, MIDAS 1.97, LIDAS 1.96. Gains concentrate in math word problems and reasoning primitives. Last column (labels unclear): 1.7B baseline 42.77, LN-Scaling 50.58, MIDAS 55.46, LIDAS 53.00; 360M baseline 33.12, MIDAS 35.14, LIDAS 50.36.
- 2026-11-17 (mechanism, Sec 4): the dense baseline stops improving its early-exit accuracy by layer 18; grown models keep improving to the last layer. Grown models tolerate swapping blocks of up to 4 layers and middle 8-layer blocks; dense drops to random.
- 2026-11-17: no analysis of when in training the effect appears.

## 3. Our speculation

- 2026-11-17 Speculation: "late layers do little" is the curse of depth. A loop that re-applies the same layers may cure it in another way. We have no layer-contribution probe on the span loop.

## 4. Corrections

- 2026-11-17: Cost figures differ by place in the paper (1.29x, 29 %, 23 %). Cite the section with the figure. 

## 5. Where our findings disagree

- None.
