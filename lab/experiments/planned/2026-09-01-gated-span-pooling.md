# Planned: E-SAC-G — learned gated span pooling, the SAC binding's final arm

Status: planned
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
