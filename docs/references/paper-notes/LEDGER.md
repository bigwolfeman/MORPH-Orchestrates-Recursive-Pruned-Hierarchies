# Paper notes ledger

Append only. One line per dated entry. Format:

    yyyy-mm-dd | <paper-slug> | <section touched> | one-line gist

2026-10-16 | ncm-next-concept-modeling-2610-14162 | Why we reference it, What the paper shows, Where our findings disagree | Seed note from memory: block-diagonal state read, no likelihood reported, Parcae span NCM costs more than it saves
2026-10-16 | repr-paper-2610-20533 | Why we reference it, What the paper shows, Where our findings disagree | Seed note from memory: PMI representation target; text run ncm-repr-tok-5k did not fit it (R squared -0.02)
2026-10-16 | hybridagent-paper-2610-18861 | Why we reference it, What the paper shows | Seed note from memory: latent steps near tokens, SFT parity, compute cut 51.2 to 13.9 TFLOP per question
2026-10-16 | spiralformer-vs-tul-mask | Why we reference it, What the paper shows, Corrections | Seed note from memory: spiral reproduces on Parcae; 2026-09-07 summary of its FLOPs claim corrected
2026-10-16 | explorative-modeling | Why we reference it, What the paper shows | Seed note from abstract and section 7: best-of-K training; LLM case modest and hardest
2026-10-16 | spiralformer-vs-tul-mask | Where our findings disagree | Added FLOP ratio (0.31x plain) and the MeSH-vs-Anchor disagreement (ours 0.004 worse)
2026-10-16 | qrr-quotient-recurrent-reasoner-2610-18174 | Why we reference it, What the paper shows, Our speculation, Where our findings disagree | New note, paper read in full: slow-mode rate inheritance, gauge-equivariant QRR, quotient halting; agrees with our fixed-point-term finding
2026-10-16 | explorative-modeling | Where our findings disagree | XM span plan 5k: plan used (gap 0.13-0.17), bound worse 0.083, K-curve +0.0012; agrees with section 7
2026-10-16 | conceptlm-next-concept-prediction-2602-08984 | Why we reference it, What the paper shows, Corrections, Where our findings disagree | Read in full: C1, C2, C4-C10 match printed numbers; C3 stop-grad NOT FOUND; C11 FALSE for GPT-2 (before first layer); Table 5 and 15 printed averages differ from their rows
2026-10-16 | ncm-next-concept-modeling-2610-14162 | header | Public checkpoints found on HF (hazyresearch/ncm-checkpoints): LM1B x3, Sudoku x3, TinyStories, XSum, WMT14; OWT not released (TODO.md only)
2026-10-16 | ncm-next-concept-modeling-2610-14162 | What the paper shows | Paper text archived; training processes sum of level lengths per example (about 6x AR positions at K=12 linear); budgets are token exposure, not FLOPs; ELBO exists for scoring
2026-10-16 | ncm-next-concept-modeling-2610-14162 | What the paper shows, Where our findings disagree | Ran released LM1B x3 + TinyStories: Gen PPL reproduces (TinyStories 16.68 vs 15.41); NCM 1.7x (TinyStories) and 2.3x (OWT) worse Gen PPL than AR sampling per Table 8; 642 positions per 128-token example
