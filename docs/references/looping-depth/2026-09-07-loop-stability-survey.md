# Loop-model stability survey (Awesome-Loop-Models, 2026-09-07)

Source: an Opus scout pass over https://github.com/huskydoge/Awesome-Loop-Models, filtered
for training stability, contractivity, depth sampling and truncation. All 26 candidate
arXiv ids were checked against the arXiv API; 26/26 resolved with matching titles. Four
papers were read in full HTML (2605.26733, 2604.12946, 2605.18797, 2502.05171); the rest
at abstract level, so their numbers beyond the abstract are unchecked. None tests ternary
QAT or β1=0 AdEMAMix.

| # | arXiv | Title | Mechanism | Evidence | Applicability to MORPH |
|---|---|---|---|---|---|
| 1 | 2604.12946 | Parcae: Scaling Laws For Stable Looped Language Models | ρ(A)<1 diagonal carry by construction + norm on e; per-sequence depth; truncation hurts extrapolation | measured, 100M–1.3B | see `parcae/2026-09-07-parcae-vs-morph-mechanisms.md` |
| 2 | 2607.13491 | DeepLoop: Depth Scaling for Looped Transformers | visit-alignment coefficient κ_R; residual scaling exponent 1/4 → 1/2 when visits align (Post-LN DeepNorm α=(2N)^{1/2}, β=(8N)^{-1/2}) | measured, GPT-2 S/M looped | highest: κ_R IS the measured whole-core 9–159 vs per-block 1.0–1.3 |
| 3 | 2606.18524 | On the Residual Scaling of Looped Transformers | ε = 1/N (not 1/√L) for L unique layers looped N times; optimal LR depends on L only | measured | a loop-count-independent LR; two constants |
| 4 | 2605.26733 | STARS: Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning | L_JSRR = (1/N) Σ‖Jv‖² via single-step power iteration JVP, λ 0.1, log-normal loop sampling | measured ablation (Ouro-1.4B: 4→8 recurrences drop 20.47 → 7.51 points) | the directional upgrade of MORPH's scalar finite-difference gain hinge |
| 5 | 2607.10681 | LayerNorm as Implicit Gain Control in Looped Transformers | LN makes the recurrence Jacobian non-normal; contractive at fixed points where operator norm > 1; the budget is the spectral MARGIN | analytic + CPU scale | explains why 4 operator-norm caps did nothing |
| 6 | 2608.18222 | Think Shallow, Solve Deep | settling / marginal / drifting regimes; one terminal fixed-point objective gives depth-safe extrapolation | measured causal ablation (Sudoku 0.19 → 0.34); Huginn measured non-settling | one loss term against both drift-to-1 and "nothing past 3" |
| 7 | 2604.21106 | How Much Is One Recurrence Worth? Iso-Depth Scaling Laws | L = E + A(N_once + r^φ N_rec)^{-α} + B D^{-β}, φ 0.46; truncated BPTT lowers φ to 0.38; hyperconnections raise it to 0.65 | measured, ~50x compute span | φ is the missing metric for depth value; MORPH ships HC |
| 8 | 2605.18797 | Simply Stabilizing the Loop via Fully Looped Transformer | Fully-Looped Architecture (h_L^{t−1} into every layer) + Attention Injection (softmax-gated write) | measured; baseline collapses at 9 loops at 318M | the 12-loop baseline signature is E6/E7's |
| 9 | 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning (Huginn) | log-normal-Poisson depth, last k=8 iterations get gradient; sandwich norm, embedding scale, adapter, LR 4e-4 → 4e-5 after two failed runs | documented anecdote | the truncation recipe is the published one; its failures were collapse/no-op |
| 10 | 2202.05826 | End-to-end Algorithm Synthesis with Recurrent Networks | recall (re-inject input) + progressive training (random start/segment) | measured, thousands of iterations | progressive training is the untried half vs gain drift |
| 11 | 2604.15259 | Stability and Generalization in Looped Transformers | without recall: countable fixed points, no input dependence at ANY spectral regime; recall + outer normalization fixes it | proofs + small tasks | spectral tuning alone cannot fix input dependence |
| 12 | 2211.09961 | Path Independent Equilibrium Models | path independence predicts upward generalization | measured, small | a read-only probe: same input from two inits |

Test first: (1) DeepLoop / 1/N residual scaling on the shared block (attacks alignment,
which a uniform rescale cannot); (2) STARS' JVP Jacobian penalty (directional; MORPH's
hinge measured rms gain 0.85 while σ_max sat at 95 on the E7 checkpoint); (3) the terminal
fixed-point objective, with Fully-Looped attention injection as the fallback.
Contradictions with what MORPH tried: weight-spectrum caps (papers 5 and 11 agree they
bound the wrong quantity); the gain hinge is not contradicted (STARS is the same family)
but both STARS and Parcae bound a state-dependent Jacobian or a parameterised carry, never
a raw weight spectrum; the LR ramp is unopposed and unexplained (Huginn cut LR 10x; Parcae's
baselines fail above 4e-4). Sharpest: papers 1 and 7 both measure that truncated BPTT trains
the loop worst, which is E6/E7's regime.
