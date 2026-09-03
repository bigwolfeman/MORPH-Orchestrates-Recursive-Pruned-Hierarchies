# Planned: E-SAC-G — learned gated span pooling, the SAC binding's final arm

Status: success
Date: 2026-09-01 (frozen before launch; executes the pre-committed binding of
failures/2026-09-01-span-aligned-compression.md — "P-S1 FALSE ⇒ run ONE
learned-gated-pooling variant before abandoning the lane")

## Question

Does replacing E-SAC's mean pool with a learned per-head gated softmax pool
recover any of the TUL token-axis deficit, or is the E-SAC lane dead?

## Hypothesis

Weak. E-SAC showed the span-pooled compressed branch is heavily load-bearing
(0.187 nats at eval) yet recovers nothing vs slot-comp — the branch
substitutes for capacity rather than adding to it. The one untested pooling
degree of freedom is WHICH tokens represent the span: mean pooling dilutes
salient tokens exactly the way the slot seed's bag-mean does
(pooling_probe's -0.470 slope law). A learned gate can concentrate the
summary on informative positions. The counter-hypothesis (favoured): the
model already routes around pooling quality, and gating moves nothing.

## Method

One construction-time flag `tul.tg_span_gate` on top of the EXACT tul-sac
recipe: config `tul_sacg` = `tul_sac` + the flag. ONE change vs the measured
tul-sac arm. Per attention layer, one zero-init parameter W_g [n_heads,
d_head]; gate logit for a token position = <k_pos, W_g[h]> (its own
post-projection k at that layer); softmax WITHIN the span; the same
normalized weights pool k and v. Zero-init ⇒ uniform softmax ⇒ exactly the
mean pool, so the arm is functionally tul-sac at step 0 and learns to
deviate (test: zero-init logits match the mean-pool model to 1e-4).

Run: 20000 steps, panel flags identical to tul-sac/tul-20k (batch 6, seed 1,
alpha_cap 3.5, t_beta3 3500, eval_every 250, gen_every 0, grad_probe_every
1), eager, ckpt_every 2000 + prune to step_20000, core depth sweep 1..8, gen
samples. References: tul-sac last-5 3.8623 / sweep K6 3.7853; tul-20k last-5
3.8461 / K6 3.7568; notul-20k last-5 3.4894.

Verification before launch: tests/test_tg_span_gate.py (6 tests: validation,
zero-init equivalence, gates exist+train, gate changes function, hand-value
gated pool, causality) — 6 passed; full suite 745 passed 1 xfailed; smoke
gate requires the "TUL TG SPAN-GATE ON" print, no acausal warning, retention
keys 0, loss < 14.

## Predictions (frozen)

- **P-G1.** tul-sacg last-5 val CE ≤ 3.746 (the original recovery
  threshold): 15%.
- **P-G2.** Matched-sweep K6 ce beats tul-sac's 3.7853 by ≥ 0.05 (clears the
  n=1 noise band — tul-sac vs tul-20k differed 0.028 on this harness): 30%.
- **P-G3.** Clean 20k, no div-guard/takeover: 80%.
- **P-G4.** The gates actually learn nonuniform pooling: final W_g RMS > 0.05
  in at least half the restricted attention layers: 70%.
- **Binding.** P-G1 or P-G2 TRUE ⇒ the lane stays open; next arm stacks the
  E3 seed rebalance. BOTH FALSE ⇒ the E-SAC lane CLOSES per the parent
  binding, regardless of P-G4; the gap hunt moves to the window-branch
  visibility restriction (e1b's 0.487 > e1c's 0.231) or the training recipe
  (horizon rebaseline). P-G4 FALSE with P-G2 FALSE ⇒ additionally record
  that the branch is insensitive to pooling quality — evidence the gap never
  lived in the compressed branch's content.

## Not verified before run

Training dynamics of the gate (only a 30-step smoke); whether k is the right
gate input (the layer input x is the GatedPoolCompressor's choice — v2 if
this moves); P-G4's W_g-RMS probe is post-hoc.

## Results (2026-09-01, run tul-sacg, wandb adew-me/morph-tul)

Clean 20000-step run, exit 0, 4.71 h wall (15:24:32 → 20:06:51). Sweep:
`$Q/core_depth_sweep_tul-sacg.json`; gate probe: step_20000 checkpoint,
per-layer `tg_span_gate_w` RMS on CPU.

| Metric | tul-sacg | tul-sac | tul-20k | threshold |
|---|---|---|---|---|
| last-5 val CE (19000..final) | **3.8233** | 3.8540 | 3.8461 | ≤ 3.746 |
| sweep K6 ce_tokens | **3.7517** | 3.7853 | 3.7568 | < 3.7353 |
| sweep K1 − K6 (loop contribution) | 0.0013 | 0.0022 | — | — |
| gate layers RMS > 0.05 | 8 / 14 | — | — | ≥ 7 |

- **P-G1 (15%): FALSE.** Last-5 = (3.7620, 3.8219, 3.8861, 3.8359, 3.8108)
  → mean 3.8233 > 3.746.
- **P-G2 (30%): FALSE.** K6 3.7517 beats tul-sac's 3.7853 by 0.0336 < 0.05.
  Real movement, inside the n=1 noise band's stated margin.
- **P-G3 (80%): TRUE.** No div-guard, no takeover, sigma_max drifted
  normally.
- **P-G4 (70%): TRUE.** 8/14 layers over threshold — and the split is the
  finding: ALL 8 prelude+coda gates learned structure (RMS 0.060–0.149,
  absmax up to 0.62); ALL 6 core gates are BIT-EXACT ZERO after 20k steps.
  The core runs on slot positions only, so its compressed branch never sees
  token K/V to pool — the core gates never received one gradient.
  Optimizer-trace confirmation that the core's attention machinery is
  structurally inert under TUL, independent of the E2 rank probes.

## Verdict

**SUCCESS as calibration; the lane CLOSES.** Every prediction resolved on
the side the frozen probabilities favored. Gated pooling is a real but
sub-noise-band improvement (0.031 nats on last-5, 0.034 on K6): the branch
is not fully insensitive to pooling quality, but pooling quality was never
where the 0.36-nat TUL gap lives. Per the parent binding
(failures/2026-09-01-span-aligned-compression.md) with P-G1 AND P-G2 both
FALSE, the E-SAC lane is CLOSED.

## Updated hypothesis

The compressed branch's CONTENT is a second-order lever. The deficit sits
upstream: the core is blind (rank-collapsed seeds) and unpaid (tokens skip
it). The gap hunt proceeds on the write side —
planned/2026-09-01-write-side-ladder.md (R0/R1/W1/W2, frozen 591dd08) tests
whether full-rank slot seeds (content / HRR-bound) restore core input rank
and buy loop contribution. The zero core gates strengthen the ladder's
premise: whatever the core does, it is not using span-content attention to
do it.
