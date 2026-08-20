# Agent Note: Stp Prediction 1

Status: implemented

Origin: Ai-notes/06-23-2026/STP-AB-Results/PREDICTION1.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# STP on/off A/B (pretraining) — Prediction 1 Results

**Date:** 2026-06-23 ~01:40 CDT
**Question:** Does the STP geodesic-smoothness regularizer (`model.stp_lambda=0.02`),
applied during **pretraining** (TST + prune→carve→route deploy pipeline, 50k steps),
help or hurt generation vs no STP?
**Wolfe's prediction 1:** "STP will be more repetitive than without." → **CONFIRMED.**

> Context correction (Wolfe, 2026-06-22): the STP paper applies STP **only in SFT**, never
> pretraining. So this A/B tests an *off-label* use. Result below ⇒ STP does not belong in
> pretraining (consistent with the paper confining it to SFT). The paper-faithful test
> (STP-during-SFT) is the gated next step.

## Setup (paired, single-variable)
- 2 arms, same seed/data/pipeline, only `stp_lambda` differs: OFF=0.0, ON=0.02.
- 50k steps, TST bag=6 ratio=0.3 (superposition→NTP@15k), prune 3k→0.25, carve@29k, route@30k.
- β1=0 AdEMAMix coord-cap cure (de-fused), both arms ran clean to 50k, no divergence.
- Eval: `ignore/stp_bench.py` (batched B=50 generation, 97 OWT-val prompts ≥32 tok, 128 new
  tokens, greedy+sampled) + `ignore/stp_judge.py` (paired blind 3-judge panel, 40-pair subset,
  fluency & correctness scored separately).

## Results — all axes agree: STP-on is WORSE

### Matched-step VAL ppl (teacher-forced)
- STP-off **21.96** | STP-on **23.12**  → STP-on **+5.3%** worse.

### Automatic degeneracy (97 prompts; +rep / −distinct / −entropy = STP-on worse) — 16/16 consistent
| metric | OFF | ON | Δ(on−off) |
|---|---|---|---|
| sampled rep-1 | 0.386 | 0.458 | +0.072 |
| sampled rep-2 | 0.138 | 0.214 | **+0.077 (+56% rel)** |
| sampled rep-3 | 0.070 | 0.138 | +0.069 |
| sampled distinct-2 | 0.862 | 0.786 | −0.077 |
| sampled entropy | 2.072 | 1.946 | −0.125 |
| greedy rep-2 | 0.724 | 0.776 | +0.052 |
| greedy loop_frac | 0.116 | 0.132 | +0.016 |
| greedy entropy | 0.612 | 0.505 | −0.107 |

Every repetition metric up, every diversity metric down, entropy down — greedy AND sampled.

### Blind judge panel (3 judges, 40 paired blind tasks, majority vote, un-blinded)
- **FLUENCY:** STP-off **22** | STP-on **17** | tie 1  (STP-off wins 56% of decided)
- **CORRECTNESS:** tie **37** | off 2 | on 1  (wash — both near-contentless, as designed)

### ARC-Easy (off-axis footnote)
- OFF 39% | ON 34% | chance 25%. Both modestly above chance; ON nominally lower but inside
  the ±9% noise band at n=100. STP doesn't touch knowledge — not a decision axis.

## Calibration (no-theater)
- The **judge fluency margin alone** (22 vs 17, n=40 ≈ 56%) is directionally clear but NOT
  individually significant (CI includes 50%). The verdict is solid because of **convergence**:
  judge + 16/16 auto metrics + ppl all agree, and the auto metrics carry larger relative
  effects (rep-2 +56%).
- The **37/40 correctness ties vindicate the separate-axis design** — a fused quality vote
  would have drowned the fluency signal in ties.

## Verdict & action
**STP in pretraining hurts: more repetitive, less fluent, +5.3% ppl, no knowledge gain.**
⇒ Set `base.yaml` pretraining default `stp_lambda` 0.02 → **0.0** (decision supported by
prediction 1 alone). Mechanism: in pretraining (no termination/goal geometry) STP's
low-acceleration prior pulls toward the fixation/looping attractor.

## Next (gated — Wolfe greenlights)
Paper-faithful **STP-during-SFT** test: both bases → STP-SFT (Dolly-400, same seed/steps);
arms = `base_off→STP-SFT` vs `base_on→STP-SFT`. Tests prediction 2 ("totally changes"):
does the STP-SFT finish erase/flip the pretraining-STP penalty? Optional plain-SFT control
on base_off to prove STP-SFT itself helps on our weak 276M.

## Artifacts
`ignore/bench_off_50k.json`, `bench_on_50k.json`, `judge_tasks.json`, `judge_key.json`,
`judge_verdicts.json`, `judge_report.json`. Ckpts `checkpoints/morph/tst_stp_{off,on}_50k/step_50000.pt`.
wandb: off `u57sqdx7`, on `emg3y2qy`.
