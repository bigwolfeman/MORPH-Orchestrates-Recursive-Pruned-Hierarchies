# Paper note: Truncate or Not? scaling of looped transformers

- **Title:** Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation (arXiv 2610.26030v2, Xiong et al.)
- **arXiv or URL:** arXiv 2610.26030
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: The one scaling study that fits loop gain against model size, tokens and BPTT depth. Gives token-per-parameter crossovers.

## 2. What the paper shows

Read 2026-11-17 via summarising fetches of arxiv.org/html/2610.26030 (about 288k characters; Appendix G was NOT in the text we got).

- 2026-11-17 (setup): five sizes, 57.8M to 918.7M non-embedding params; FineWeb-Edu; 1.1 to 1291 tokens per parameter; 226 grid configs, 592 runs, about 2.7e22 FLOPs (App B/C). Setting (T, B): T forward loops, backprop through last B. (1,1) is a standard model.
- 2026-11-17 (law, Eq 4.1): N_eff = N_once + g(T,B) N_rec, g = exp[c(1 - T^-p B^-q)]. Ceiling e^c about 3.04 (extrapolated). g(2,2) 1.910, g(4,4) 2.504, g(8,8) 2.803, g(16,16) 2.939 (Table 10). The law does NOT let g depend on N or D. Per-config empirical estimates do vary (Fig 6, only IQRs visible). So "how gain changes with size and tokens" is not answered in the text we saw.
- 2026-11-17 (compute accounting, Sec 5.2): forward loop 1 unit, backward 2, so recurrent cost T + 2B. N_comp = N_once + (T+2B)/3 N_rec.
- 2026-11-17 (truncation cost): at equal compute (8,2) is worse than (4,4) by 0.0087 to 0.0278 nats, gap shrinking with size: 2.78, 1.47, 1.36, 1.07, 0.87 e-2 (Table 6). Truncation helps only if the forward loop count is fixed. When both are free, full BPTT is optimal (Sec 7). They say this suggests truncation may be suboptimal in recipes like Parcae. No wall clock reported.
- 2026-11-17 (crossovers, App D.3, Table 11, 12): two loops beat no looping at about 30 tokens per parameter; at 50 and 200 tokens per parameter the (2,2) lead is about 0.01 and 0.025 nats. Switch points 1 to 2 loops about 14 to 17 tokens per parameter; 2 to 4 about 312 to 413; 4 to 8 about 7,000 to 9,600; 8 to 16 about 147k to 210k. Observed (4,4) overtakes (2,2) near 200 at 60M to 240M.
- 2026-11-17 (equal compute, Sec 6.3, Fig 14): at about 2.1e19 FLOPs, 344M (2,2) with 8 percent more tokens reaches 2.3590 vs 2.3663 for 471M flat, (-7.3 +- 1.8) e-3 nats, with 27 percent fewer params. At strictly matched compute (2,2) is worse by 7.9, 3.8, 1.4 e-3 nats at three budgets (Table 12). So looping does not win at equal FLOPs; it saves parameters.
- 2026-11-17 (optimal loops, Sec 6.1): best T moves 1, 2, 4 as compute grows; T* ~ C^0.24, D* ~ C^0.76. Free-size: flat (T=B=1) is optimal at every compute (Eq 6.6).

## 3. Our speculation

- 2026-11-17 Speculation: the law says a loop's worth is worth more tokens-per-parameter. A loop's lead over a flat model of equal FLOPs may stay near zero at our scale and token count. We should expect a 5k to 20k step lead over a depth-1 twin to be small and late.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 7 agrees. Gain per loop grows with tokens per parameter (switch points rise by more than 20x per doubling of loops (about 14, 350, 8000, 170k)). Value forms late.
- Equal-FLOPs lead is about zero or negative at matched compute. This supports item 4 (no free hierarchy win) and our long-standing equal-wall-clock results.
- Item 2: law assumes full-depth gain is generic, not job-specific. We do not test that here.
