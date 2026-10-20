# Paper note: Gated Recurrent Transformers (expressive depth through recurrent modulation)

- **Title:** Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation (Hegazy, Alanwar, Elhoushi; GUC, TUM, Cerebras)
- **arXiv or URL:** 2608.15062 (v4, 2026-08-26; v1 2026-08-15). Type: paper. Code: github.com/Amr-Hegazy1/gated-recurrent-transformer (not read).
- **Local paper text:** docs/references/looping-depth/gated-recurrent-transformers-2608-15062/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is a Parcae-layout looped model (prelude, shared core, coda, depth sampling) with a measured component ablation. It adds three pieces next to our diagonal injection: a prelude anchor fed to every pass, fresh noise every pass, and an elementwise gate that blends the old state with the core output. Triage score 2.
- 2026-10-20: Scale is far from ours (nanoGPT, GPT-2 sizes, 20k steps, ~9.8B tokens, 491k tokens per step). Nothing here is a 5k-step number.

## 2. What the paper shows

Read in full: abstract, Sections 1 to 5, Table 1, ablation text, early-exit text, limitations, and the gate-trajectory appendix (the Gate=0 and Gate=1 bracket). Not read: Appendix B.3 to B.6 tables line by line, Appendix on KV-cache sharing, the code repo.

- 2026-10-20 (Sec 3.3, Eqs 2 to 5): per pass r, input = Wproj[h(r-1)+noise, h_pre] (Wproj is d x 2d); core output o(r); new state h(r) = g*h(r-1) + (1-g)*o(r), g = sigmoid(MLP([LN(h), LN(h_pre)])/tau + gate noise), second-layer bias +4 so g is about 0.98 at init (the loop starts as identity). Noise sigma 0.1 on state and gate. Depth r ~ Uniform{1..R}, no auxiliary loss.
- 2026-10-20 (Sec 5, Figure 5, small model, 20k steps, one run each): recurrence alone costs +0.107 nats vs the dense model. Prelude+coda recovers 0.035, per-step state noise 0.018, prelude re-injection every pass 0.022, the elementwise gate 0.048 (largest). Sum is about 0.123, so the full model ends about 0.016 under the dense baseline.
- 2026-10-20 (Table 1, isoFLOP, val loss): small 3.14 vs dense 3.15 (3 seeds: 3.145 +/- 0.004 vs 3.188 +/- 0.056); medium 2.89 vs 2.84 dense; large 2.77 vs 2.71 dense. Their "heavy-tail Poisson" Parcae arm: 3.23 / 2.97 / 2.93; the gate arm leads it by 0.09 / 0.08 / 0.16.
- 2026-10-20 (Sec B.5, B.6): removing state noise costs 0.031 (medium, 5k steps) and 0.018 (small, full run); doubling it costs 0.019. Gate bias sweep {-2, 0, 2, 4} spreads 0.019 nats at the full horizon.
- 2026-10-20 (appendix gate analysis): forcing the gate to 0 everywhere gives loss 12.03; forcing it to 1 (never update) gives 5.26 = prelude-only; trained model 2.68. Output KL to the final pass is 2.61 at the prelude, 0.46 at pass 2, 0.014 at pass 5. Early exit at half the passes keeps about 92 % (their metric).
- 2026-10-20 Failure modes reported: R is fixed at inference (no halting); gate bias and noise may need retuning off the GPT-2 family; at medium and large scale dense still leads at the standard budget (by 0.05 and 0.06); Ouro is poor at medium/large (2.93, 2.82).

## 3. Our speculation

- 2026-10-20 Speculation: the gate bias +4 is an identity start. Our measured 0.87 floor in `DiagonalInjection` is the same idea in another form (a quiet pass is a near-identity map). Their result says the blend gate is worth 0.048 against a plain loop, which is the largest single piece. We have no per-element learned blend in the core; our injection is a diagonal decay `A` that has not moved from init.
- 2026-10-20 Speculation: per-pass fresh noise on the state is cheap and was worth 0.018 to 0.031. Our noise-seeded particle dead end used noise to make hypotheses differ, not to regularise the map. Do not conflate the two.
- 2026-10-20 Speculation: their loss gap between full model and dense closes with tokens (medium and large), which matches Wolfe's reading that K1-K6 grows with tokens. Their model never has a K1-K6 reading, so the paper cannot say whether the gate raises loop contribution or only raises CE at a fixed R.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Their Poisson-depth Parcae control trails dense by 0.09 to 0.16 and the gate arm beats it. Our plain Parcae loop is the baseline we trust; their control uses 1+1x10+1 and 2+5x4+2 layouts, not our 4+6xT+4, and only one run per cell outside the small model. Treat the gap as untested on our stack.

## Test ideas (score 2)

- 2026-10-20 Test: control = plain Parcae with `DiagonalInjection`. Arm = add the elementwise blend gate (bias +4, prelude-conditioned) only, no new noise. Prediction (behaviour): K1-K6 and CE@6 move in the same direction as their 0.048 gain; if CE improves while K1-K6 does not, the gate bought per-pass quality, not depth. Run 2 seeds, since K1-K6 spread between seeds is 44 % of its mean.
- 2026-10-20 Test: control = plain loop. Arm = fresh state noise sigma 0.1 per pass at train only. Prediction: CE@6 improves slightly and K1-K6 holds; cheapest of their three pieces (no new parameters).
