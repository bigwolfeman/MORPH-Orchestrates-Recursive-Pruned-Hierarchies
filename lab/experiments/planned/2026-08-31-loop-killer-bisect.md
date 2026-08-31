# Planned: the loop-killer bisect — what MORPH added that TitanMAC didn't have

Status: planned
Date: 2026-08-31 (frozen before any cell runs; written pre-compaction).

## Question

TitanMAC `looped_b1_gelfix` (causally clean: no GLA, plain residual, no core
weight cap, diagonal injection decay<1, truncated BPTT) earned real depth:
T=1 +43.8% PPL vs T=8, T=4 within 2.1% (verified in
`111TitanMAC-Standalone/experiments/looped_b1_gelfix.log:6203-6210`, 3 runs,
6 snapshots). MORPH's causal loop earns 0.006-0.015 nats in both geometries.
Which MORPH addition killed it? Wolfe's ordering (structure VETOED — 4:8:4
was tested working, Parcae runs 4:4:4): **GLA prime suspect; HC Cayley
residual and σ-cap plausible.** Supporting evidence: the 30k leak model
opened core ret_gate 8× (0.024 vs init 0.0025) — the optimizer demonstrably
uses this door; at 4500 the gates are near-init (0.003-0.005) in all causal
arms, so GLA-off is cheap to test and the gate values alone do not clear it
(branch NORMS unmeasured — the audit-module-geometry rule).

## Common base (every cell)

`notul_l2` (causal carry "none", kernels where legal), panel flags
(steps=4500 batch=6 seed=1 alpha_cap=3.5 t_beta3=3500 eval_every=250
ckpt_every=500 gen_every=0 grad_probe_every=1), smoke-gated, checkpoints
pruned to step_4500, `token_depth_sweep.py` depths 1..8 (48 rows) as the
readout. Primary metric per cell: depth-earned CE K1−K6 and trained-support
K3−K6. Reference flat baseline: notul-l2nc (K1−K6 = 0.120, K3−K6 = 0.015).

## Arms (in run order)

- **BG0 — GLA off**: `model.retention=false` (never constructed; the Opus
  touchpoint map `lab/gla_touchpoint_map.md` is the verification basis).
  Param count drops — depth-curve SHAPE is the readout, not absolute CE.
- **BC0 — cap off**: `training.spectral_project_cap=0`. Uncapped core at
  4500 causal steps; S1 guard live (divergence risk accepted, short horizon).
- **BG0C0 — both off**: runs only if BG0 and BC0 are both flat (tests the
  conjunction).
- **BHC — plain residual**: NOT config-reachable today (HyperConnectionResidual
  is the residual implementation). Two stages: (probe, free) read the trained
  Cayley mixing parameters of notul-l2nc — if the core layers' HC learned to
  weight the branch (core output) near zero vs the identity streams, HC is
  bypassing and the code work is justified; (arm, code) implement a
  `residual: standard` construction path and run the cell. Stage 2 only on
  Wolfe's go after the probe + the first cells.

## Predictions (frozen)

- **P-G0.** BG0 depth-earned K1−K6 ≥ 0.10 nats (GLA was the killer): 35%.
  Wolfe's prior is higher; mine is tempered by the near-init gates at 4500 —
  recorded disagreement, the cell settles it.
- **P-C0.** BC0 K1−K6 ≥ 0.10: 25%. Sub-prediction: S1-clean without the cap:
  60% (the cap was introduced as the takeover cure).
- **P-G0C0** (if run): ≥ 0.10: 40% — killers can be conjunctive.
- **P-HCprobe.** notul-l2nc's core HC mixing puts < 25% aggregate weight on
  the branch (bypass signature): 55%.
- **Binding.** Any cell ≥ 0.10 ⇒ that component is implicated; re-run the
  winning cell's config at mean_depth 8 / max 14 (TitanMAC's regime) to
  confirm magnitude, then design the fix (GLA: causal carry or off-by-default;
  cap: replace with decay-parameterized contraction like TitanMAC's; HC:
  standard-residual option). ALL cells flat ⇒ the killer is in the remaining
  diffs (CCA/CSA/HCA attention, QAT, d1024, mean 6 vs 8) — next bisect round,
  new prereg.

## Not verified before launch

`retention: false` composition never trained (smoke gates it; the Opus map
verifies construction); uncapped causal token-path stability unknown; the
depth sweep's sensitivity floor (~±0.01 nats at 48 rows) is well below the
0.10 threshold.

### Method amendment — 2026-08-31 16:00 (after BG0/BC0, before round 2)

Round-1 outcomes: **BC0 flat** (K1−K6 0.142, K3−K6 0.013 vs baseline
0.120/0.015; trained clean to val 4.32 — S1-clean TRUE, P-C0 resolved on the
75% no-effect side). **BG0 catastrophic** — an unpredicted third outcome: flat
at unigram CE ~7.4 from step 250, upward drift, div-guard abort at 2080
(`DIVERGED_step_2080.pt`). Branch-norm probe (d562088) on the trained baseline
shows attention alive in all 14 blocks and gated GLA at only 5–6% of attn RMS
(42% coda.1) with ret_gates at init — GLA is near-inert at convergence yet
training collapses without it.

BG0C0's original trigger ("both singles flat") did not materialize, but the
cell is now decisive for a NEW question: BG0's projection was BINDING
(σ mean 1.41 / cap 1.5), so the stall may be a GLA-off × cap interaction
(attention forced to carry everything while spectrally capped) rather than
GLA being load-bearing per se. Round 2, frozen before launch:

- **BG0C0** (retention=false, cap=0) runs UNCONDITIONALLY, full 4500 + sweep.
  **P-G0C0r (new, frozen): 40%** that it escapes the unigram basin
  (val < 6.0 by step 1500). Escape ⇒ interaction story; GLA = workaround for
  over-constriction; the loop-killer hunt turns to the cap+attention geometry.
  Stall ⇒ GLA is required for training in this architecture; next arms must
  keep GLA and manipulate it (bootstrap-then-close; core-site-only removal).
- **BG0-seed2** (notul_bg0, training.seed=2, 1500 steps, no sweep) closes the
  RNG-shift confound (module removal changes every other tensor's init under
  seed=1). **P-G0s2 (frozen): 75%** it stalls the same way (val > 6.5 at 1500).

Original predictions untouched. BG0C0's original P-G0C0 (40% depth-earning
restored) still applies IF it trains to 4500.
