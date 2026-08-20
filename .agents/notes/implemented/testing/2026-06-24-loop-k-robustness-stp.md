# Agent Note: Loop K Robustness Stp

Status: implemented

Origin: Ai-notes/06-24-2026/Loop-K-Robustness-STP/RESULTS.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Loop-K Robustness & Latent-Forecastability (STP) — Eval-First Results

**Date:** 2026-06-24
**Paper:** arXiv:2604.18464 "Semantic Step Prediction" (step-boundary STP + multi-step latent
prediction MSE). Local extract: `docs/references/2604.18464.md`. Predecessor STP: 2602.22617.
**Reframe (Wolfe):** the paper is, for MORPH, a **loop-k robustness** paper. The looped weight-tied
core *is* the latent rollout; loop count `k` *is* the rollout length. The deployable payoff is
"**use fewer loop steps at inference**" if the loop trajectory is forecastable/convergent.

## Harnesses built (all eval-only, ride the real forward)
- `ignore/fixed_eval_phasec.py` → refactored: extracted `load_phasec_model(...)` (conform CMS
  sparse shapes + enable routing exactly as training + ABORT on material-missing). Single source
  of truth for the Phase-C faithful load.
- `ignore/loop_k_robustness.py` — sweeps `cfg.mean_depth=k` (eval forward pins depth, transformer.py
  ~691, no monkeypatch) over a FIXED val-batch set; per k reports ppl, KL(p_k‖p_canon), top-1
  agreement vs canonical k=6. Logits via `labels=None` branch (fused-CE returns logits=None).
  Bug caught pre-run: `nll_loss` on log-probs, not `cross_entropy` (would double-softmax).
- `ignore/loop_latent_mse.py` — the paper's metric on the LOOP axis. Captures per-iteration carrier
  z_0..z_T (via gated `model._capture_traj`, bit-exact when off — transformer.py edit), computes
  Eq5 normalized latent-MSE `‖z_k+m·(z_k−z_{k−1}) − z_{k+m}‖²/‖z_{k+m}‖²` and decode fidelity
  (predicted vs actual carrier through real head tail lm_mixer→final_norm→head; top-1 agreement).
- All loads verified: **0 material missing / 0 unexpected** on every ckpt (no-theater).

## Checkpoints (all full-stack: looped + ternary-QAT + AdEMAMix; tst@50k also routed, masked-dense)
- `tst_stp_off_50k`, `tst_stp_on_50k` — token-axis STP A/B, 50k, full deploy stack.
- `loopstp_off`, `loopstp_on` — loop-axis STP A/B, **only 5k, DENSE (pre-prune/carve/route)** ⇒
  degenerate/undertrained regime, CONFOUNDED. Secondary evidence only.

## Loop-K robustness (30 batches B=2, Δppl% vs canonical k=6)
```
                 k=1     k=2    k=3    k=4   | k=8   k=16  | canon ppl
tst_stp_off    +30.5  +10.9  +4.7   +2.1   | -0.9  -2.4  | 21.93
tst_stp_on     +24.7   +8.9  +4.1   +2.0   | -1.2  -3.7  | 23.07
loopstp_off_5k +18.0   +5.7  +1.9   +0.7   | +0.0  +0.8  | 48.26
loopstp_on_5k  +31.9   +8.9  +3.0   +1.0   | -0.1  +1.0  | 51.20
```

## Latent forecastability (20 batches B=2, depth=8)
```
                m1-MSE  decode-top1  | per-step MSE_1: k1    k4     k7
tst_stp_off     0.251     0.750      |               1.372  0.034  0.021
tst_stp_on      0.947     0.725      |               5.751  0.090  0.067
loopstp_off_5k  0.208     0.770      |               1.076  0.024  0.002
loopstp_on_5k   0.020     0.822      |               0.129  0.001  0.000
```

## Findings
1. **No cliff in the 50k regime.** Both deployed models degrade *gracefully*; k=16 (2× trained
   max=8) still *improves* ppl; k=4 costs only +2.1%. Poisson(6)∈[1,8] training already bought a
   wide usable band — MORPH *already* supports ~33% loop-compute savings (k=4) nearly free. Cost
   concentrates at k≤3. (Wolfe's remembered "cliff to 0" is not present here in [1,16] — likely a
   different/earlier model or k beyond range.)
2. **loop-STP works exactly as the paper predicts, on our loop axis: 10× more forecastable**
   (latent-MSE 0.208→0.020), late steps→0.000 (loop reaches a FIXED POINT by ~k4), decode
   0.77→0.82. = the paper's 0.955→0.006 effect reproduced (milder; not LoRA-from-scratch β=1).
   **Mechanism confirmed: loop-STP is the correct knob for loop-trajectory geometry.**
3. **token-axis STP is the WRONG axis for the loop**: latent-MSE 4× WORSE (0.25→0.95). Smoothing
   the token trajectory *de*-smooths the iteration trajectory. token-STP ≠ loop-STP (near-opposing
   on this axis). The small token-STP band-widening (finding 4) is some OTHER mechanism.
4. **token-STP widens the low-k band only in RELATIVE terms, and loses on ABSOLUTE quality.**
   k=1: off +30.5% vs on +24.7% (on degrades less), but on's base ppl is worse (23.07 vs 21.93),
   so off@k wins absolutely at every k. STP trades base quality for relative truncation-robustness
   ⇒ NOT a deployment win for "fewer steps." Consistent with prior "STP-in-pretraining hurts."
5. **⚠️ Forecastability ≠ truncation-robustness (in the 5k regime).** loopstp_on is MAXIMALLY
   forecastable (MSE 0.02) yet its low-k ppl curve is WORSE (k1 +31.9 vs off +18.0). Reconciliation:
   loop-STP makes the loop *converge to a fixed point fast* → late steps redundant (good for "stop
   once converged") but you must still run enough steps to GET there (k1-2 stay bad). And the 5k
   dense regime is degenerate. Whether loop-STP's confirmed forecastability buys deployable
   fewer-steps on a PROPERLY-TRAINED model is the open question.

## Open question → overnight Arm B (full-stack loop-STP)
Resume `tst_stp_off_50k` (flat 1e-4, prune/carve/route all past → clean NTP+routing continue),
fork: control (`loop_stp_lambda=0`) vs loop-STP (`loop_stp_lambda>0`), ~4k steps each. Re-run both
harnesses. **Q: does full-stack loop-STP lower latent-MSE AND flatten the low-k ppl curve (= the
deployable fewer-steps win)?** Resume needs a STAGED seed dir (tst_stp_off has a wandb_id.txt that
would otherwise resume its wandb run + write into its dir) and full-resume reconstructs routing.

## Caveats (no-theater)
- loop-STP evidence so far is 5k/dense/undertrained — CONFOUNDED. Arm B de-confounds on full stack.
- Eval is B=2, 20-30 batches → relative curves stable; absolute ppl ~±1%.
- latent-MSE multi-step (m=2,3) blows up (0.84→1.92 off) — loop is locally-affine but globally
  CURVED (paper: 3-layer MLP beats linear 3-12×). Early-exit policy should predict 1 step + re-anchor.
- decode-fidelity decodes carriers through the real head tail (lm_mixer→final_norm→head); the carrier
  is pre-coda so this is predicted-vs-actual self-consistency (the paper's metric), not the model's
  literal output at that depth (that = the k-sweep truncation).
- Known bug (audit): token-axis STP regularizes `final_norm(x)` but head reads
  `final_norm(lm_mixer(x))` (transformer.py 900 vs 907) — fix before leaning on token-STP.
```
