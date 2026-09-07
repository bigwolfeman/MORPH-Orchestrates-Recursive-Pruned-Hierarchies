# Parcae (arXiv 2604.12946) against MORPH, mechanism by mechanism

Source: an Opus research-reader pass on 2026-09-07 over the repo's extracted paper text
(`parcae.md`), `docs/references.md`, and the sibling reimplementation's result
(`/home/wolfe/parcae/docs/experiments/successes/2026-09-05-parcae-owt-loop-contribution.md`).
Every paper claim carries a section/table; "not in paper" is explicit. The nats figures in
§4 are ln(PPL) conversions of the paper's tables, not numbers the paper prints. Verified
against MORPH's code the same night: the `DiagonalInjection` channel range and the global
`n_nograd` (see `lab/experiments/failures/2026-09-07-arc-e7-block-loop.md`).

## 1. What the paper specifies

Parcae frames a middle-looped model as `h_t = A·h_{t-1} + B·e + R(h_{t-1}, e)` (§3, App. C)
and blames instability on ρ(A).

- **The A constraint (§4.1, Table 1).** `A := ZOH(Diag(−exp(log A)))`, `B := ΔB`, learned
  Δ. Table 1: additive injection is ρ(A)=1 *marginally stable*; concatenation unconstrained
  *unstable*; Parcae ρ(A)<1 *stable*. Fig. 3: divergent runs learn ρ(A) ≥ 1. B unconstrained.
- **Prelude norm (§4.1, App. J).** `e ← LN(P(s))`. App. J: at 1.3B, stable to 150k steps,
  then loss spikes with residual norm ~7e7 (Fig. 17); σ(A), σ(B) flat; the explosion
  localised to "the initial injection of prelude output e" (Fig. 21). Fix: one norm on P.
- **Loop entry state.** `h_0 ∼ N(0, σ²I)` (§2.1, Alg. 1 line 2). Never ablated.
- **Depth distribution (§4.2, App. H/I).** Poisson(µ_rec), per sequence inside the
  micro-batch (Alg. 2), no max clamp. Table 9: at fixed µ_bwd, µ_rec ∈ {4,8,14,20,26,32} →
  **µ_rec = 8 wins at every test depth** incl. T=16 and 64. Table 8: per-sequence beats
  per-micro-batch 300.32 → 70.47 PPL at T=1 (100M).
- **Backprop truncation.** `µ_bwd = ⌈µ_rec/2⌉`. Alg. 2 pads at the FRONT: `τ_i = T_max −
  T_i`, no update for t < τ_i, no-grad for τ_i ≤ t < T_max − µ_bwd, grad for the last µ_bwd.
  Every sequence ends at T_max with min(T_i, µ_bwd) grad iterations. Fig. 16/Table 10: at
  µ_rec=20, raising µ_bwd 4→12 monotonically improves T=1/16/64. Full BPTT is only a
  reference curve (Fig. 14).
- **Optimizer / LR (App. Q).** RDM setup: Adam β=(0.9,0.95), update clipping, ε removed,
  warmup+cooldown 4096 steps, constant η 4e-3 (100M) / 2e-3 (350M), clip 1.0. Transformer
  setup: AdamW β=(0.8,0.95) wd 0 on 1-D params + A,Δ,B,C; Muon (lr 8e-3, wd 0.2) on
  matrices; 0 % warmup, 50 % cooldown; clip 1.0; bf16. Pre-Norm RMSNorm, scaled init, tied
  embeddings, RoPE θ=50k.
- **Scale.** 100M/350M (µ_rec 16 and 8); 140M–1.3B (µ_rec 8, µ_bwd 4), ctx 2048, batch 256.
- **Not in paper:** any spectral cap on core weights; state renorm in the loop; a loss on
  the loop; quantisation; a gain hinge; AdEMAMix; a max-depth clamp.

## 2. Mechanism table

| Mechanism | Parcae | MORPH | Verdict |
|---|---|---|---|
| A (carry) | ZOH diag, ρ<1, all d_h | `DiagonalInjection`: `A=exp(logA).clamp(0.9999)` on the ctx channel ONLY (256 of 768); the other 512 dims ride the HC-Cayley residual (norm-preserving) | Partial: 2/3 of the carrier is Table 1's ρ(A)=1 row |
| B (injection) | ΔB, unconstrained, on LN(P(s)) | `dt=exp(log_dt)`, ctx channel; plus `ChannelInject` terms | Narrower |
| Prelude norm | LN(P(s)) | `input_norm` RMSNorm before the loop | Has it (unclamped learnable gain) |
| h_0 | N(0, σ²I) | `_CloneInit`: h_0 = e | Changed (paper never ablates) |
| Norm placement | Pre-Norm only | Pre-Norm, final_norm | Matches |
| Depth dist. | Poisson per sequence, no clamp | Poisson per sample, clamp at max_depth | Matches + clamp |
| µ_rec | 8 best (Table 9) | 6 base; 16 in E6/E7 | 16 is measured worse |
| µ_bwd | ⌈µ/2⌉, front-padded per sequence | `bptt_depth` as a GLOBAL no-grad prefix; samples ≤ prefix get no loop gradient | Changed; silent at full BPTT |
| LR | 2e-3 … 8e-3, warmup/cooldown or 0 %/50 % | 1e-4 flat after a 1000-step linear ramp | MORPH detonates 20–80x below Parcae's unstable baselines |
| Optimizer | Adam/AdamW+Muon | AdEMAMix β1=0, α 8 cap 3.5 | Not in paper |
| Weight decay | 0 / 0.2 | 0.1 | Changed |
| Grad clip | 1.0 | 1.0 | Matches |
| Precision | bf16 | bf16 + ternary QAT | Ternary not in paper |
| Spectral cap on core | none | off (4 variants refuted) | Neither |
| State renorm | none | `slot_state_renorm` off | Neither |
| Loss on the loop | none | gain hinge λ100@0.9, cot clip 4 | MORPH-only |
| Scale | 100M–1.3B, d 768–1536 | 284M, d 768 | Comparable |

## 3. Divergence in the paper

(a) Hyperparameter-driven at 100M (Table 2, App. F): the pre-norm baseline converges only
at 2e-4; residual-norm RDM fails at 6e-4+; Parcae survives all five LRs; diverging runs
reach ‖h_T‖ ~1e17–1e19 (Figs. 2, 9). (b) Late instability at 1.3B only (App. J): stable to
150k, then spikes with residual norm 7e7 (Fig. 17). Per-micro-batch depth sampling causes
spikes at 350M (App. G), attributed to large residual jumps at the final recurrence.

## 4. Depth curves (ln PPL conversions)

100M, µ_rec 16 (Table 8): T=1 70.47 → T=4 17.15 → T=8 14.08 → T=16 13.59 PPL: K1−K16 ≈
1.65 nats, K4−K16 ≈ 0.23, K8−K16 ≈ 0.036. 350M, µ_rec 8: 17.92 → 10.49 → 10.09 → 10.11:
K1−K8 ≈ 0.58, K4−K8 ≈ 0.039, K8−K16 ≈ −0.002. §5.3: gains plateau near µ_rec; the fit
`L(T) = L_∞ + Z e^{−zT}` with `L_∞ ≈ L(µ_rec)` (App. L). The sibling reimplementation
(144M, µ_rec 8, µ_bwd 4, OWT, 10k updates): K1−K8 0.294 [0.286, 0.303], K3−K8 0.040,
K4−K8 0.017, K8−K16 0.0002. A healthy Parcae loop earns 0.02–0.04 nats past iteration 3.
MORPH's paid loop (0.168 at 5k) and E6 (K3−K6 0.277) are LARGER: the likelier reading is
an anomalously bad K3 (unconverged shallow readout), not unusually productive deep
iterations.

## 5. Ranked mechanisms and tests

1. Constrain A on the whole residual (§4.1, Table 1, Fig. 3). Test: widen
   `DiagonalInjection` to all d_model dims; rerun the 9 `warmup: 0` detonation draws under
   the abort rule (verdict by step 775).
2. Per-sample gradient window (Alg. 2). Test: fraction of samples with nonzero loop
   gradient 0.27 → 1.0 on `notul_deep16`.
3. Do not raise µ_rec past 8 at fixed µ_bwd (Table 9); spend on µ_bwd (Fig. 16).
4. Instrument as App. J (per-checkpoint state norm over T, per-block norms within
   iteration 1, per-layer norms through the prelude) on the detonation checkpoints.
5. h_0 = e vs Gaussian: inference from Figs. 20–21 (jump at T=1), not a paper result.
