# Paper note: LiFT (Loop Flow Transformers), looped DiT

- **Title:** LiFT: Loop Flow Transformers. Not the same as LIFT 2609.38149.
- **arXiv or URL:** 2610.05538 (v1). Read: HTML text characters 0 to 100k of 165k.
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it
- 2026-11-17: A looped core trained with few passes (K_train = 2) that is run deeper at test time. Closest to our Poisson-depth idea in a generative model.

## 2. What the paper shows
- 2026-11-17 (Sec 3.1, 4.1): class-conditional ImageNet 256 (SD-VAE latents), no language. 4-block prelude, shared core, 0-block coda. Each loop is trained toward a point on a straight path from the detached first estimate to the flow target; K_train - 1 uniform depth coordinates per sample, loss averaged over readouts, prelude aux loss 0.01. Core input RMSNorm(h) + h0 plus a depth embedding. Gradients flow through the whole rollout. 500k updates, batch 256, D = 3.28e10 latent tokens.
- 2026-11-17 (Tables 2-5): extrapolation with no retraining: XL/2-R12 FID 17.07 at 2 loops to 9.31 at 16. L/2-R10 20.30 to 10.95 at 8. Single-block cores plateau.
- 2026-11-17 (Sec 4.3): at equal training depth LiFT trails dense (L/2-R10 K_train 2: 20.30 vs dense L/2 16.94). At equal inference FLOPs it beats a larger dense: L/2-R10 at 8 loops 10.95 at 28.24 TFLOPs vs dense XL/2 14.30 at 59.31. Training cost 61.98 vs 91.10 EFLOPs (-32 %). Dense B/2 stays stronger at every cost.
- 2026-11-17: single run per config; analytic FLOPs only; no K_train = 1 LiFT number.

## 3. Our speculation
- 2026-11-17 Speculation: each loop has a flow-time coordinate and a depth embedding, so each pass has a different job by construction (step along a path). That is a built-in answer to "load is not depth". Text has no such path unless noise is added, and token-level diffusion cost us 16 to 64x (item 5).

## 4. Corrections
- None recorded yet.

## 5. Where our findings disagree
- 2026-11-17: Paper: loops trained at 2 extrapolate to 16. Ours (item 7): token-loop value forms late and slot-loop value stayed flat. Image flow has an explicit per-pass target; we found per-pass targets are met by pass 1 (item 2). Not the same setting.

## 6. Triage
Score 1.
