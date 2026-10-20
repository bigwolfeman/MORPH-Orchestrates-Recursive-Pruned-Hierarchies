# Paper note: STARS, Jacobian spectral-radius regularization for looped LMs

- **Title:** Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models (Yang, Han, Zhang, Wei, Shao, Guo, Li; Nanjing University; ICML 2026)
- **arXiv or URL:** 2605.26733 (cs.LG, submitted 2026-05-26). Type: paper.
- **Local paper text:** docs/references/looping-depth/stars-stable-recurrent-scaling-2605-26733/ (paper.txt, paper.pdf 5.4 MB, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is the "yang et al 2026" that Think Shallow cites for rho-control. It regularizes the Jacobian spectral radius of the looped map directly, not the weight spectrum. Our dead-end list holds spectral caps and gain clamps, which bound weights or magnitudes. This is a different object.
- 2026-10-20: Read main text through Table 2. Appendices and Sec. 6 ablations only skimmed.

## 2. What the paper shows

- 2026-10-20 (Sec. 4, Findings 1 to 5, toy: 4-digit addition, one GPT block looped, d=512, trained at T=4): the placement of the norm, not its type, sets the dynamics. Pre/Pre-Sandwich states grow without bound and collapse past the train horizon; Post/Post-Sandwich stay bounded but settle into poor attractors. Prelude/coda layers do not change this. L2 penalty between adjacent iterates helps little and underperforms plain random loop sampling alone. Wide sampling ranges can make Post-Sandwich training collapse.
- 2026-10-20 (Sec. 5, Eq. 3): JSRR loss = mean over the batch of ||J v||^2 with one power-iteration step and a JVP, evaluated at the current state at a sampled iteration t (a proxy for the unknown fixed point). Combined with random loop sampling to give STARS. One step is chosen because multi-step power iteration gave abnormal gradients.
- 2026-10-20 (Table 2, fine-tuning pretrained Ouro-1.4B, avg of 5 math sets): Ouro-1.4B-SFT 65.25 / 70.46 / 52.97 at 2 / 4 / 8 recurrences; STARS 66.90 / 74.18 / 65.55. GSM8K: Ouro drops 20.47 % from its peak at 8 steps, STARS 8.26 %; peak GSM8K +4.01 %.
- Failure modes reported: instability beyond the training horizon (Ouro peaks and falls); residual drop remains at 8 steps (65.55 vs peak 74.18).

## 3. Our speculation

- 2026-10-20 Speculation: JSRR cost is one JVP per step, which on our core (6 layers, 1024 d, seq 1024) is a second forward-mode pass. For a TRAINING-speed priority that is likely a 1.3x to 2x cost on the core. We have the fixed-point term at no cost, which targets the same asymptote.
- 2026-10-20 Speculation: it penalizes ||Jv|| at a random direction, so it bounds typical gain, the quantity we already hold at the 0.87 floor. Think Shallow shows forcing typical gain up breaks training and holding it down is harmless; STARS likely lands on the same floor. Nothing here asks the loop to EARN depth.
- 2026-10-20 Speculation: no CE or K-curve at pretraining scale. The result is robustness of an already trained loop after SFT, not loop contribution.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Their claim that random loop sampling retains performance beyond the training range matches our Poisson draw. Their claim that L2 on adjacent iterates is useless is consistent with our "clamps mask the symptom" finding.
