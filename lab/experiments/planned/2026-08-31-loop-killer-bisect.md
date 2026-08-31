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

### Method amendment — 2026-08-31 17:10 (BG0C0 scored; criterion defect recorded)

**Criterion defect, recorded rather than papered over:** the frozen "≥ 0.10
nats" threshold is satisfied by the flat REFERENCE itself (0.120) — it cannot
discriminate. Written intent was a contrast against the flat baseline; scoring
below uses Δ(K1−K6) vs notul-l2nc's 0.120 with the ±0.01 sweep floor. BC0's
0.142 (Δ +0.022) stays scored FLAT under either reading of intent; it is
nowhere near a regime change.

**BG0C0 result (4500, clean, exit=0):** K1 4.4261, K2 4.2621, K3 4.2228,
K4 4.2091, K6 4.2059, K8 4.2089. K1−K6 = **0.220** (Δ +0.100 vs baseline,
PPL +24.6% at K1 vs baseline's +12.7%); K3−K6 = 0.017 (unchanged). Earning
concentrated in K1→K3; saturated by K4. Absolute: val 4.13@4250, K6 4.206 —
BEATS baseline (4.344) and BC0 (4.401) by ~0.14 nats with 18.9M fewer params.

**Scored: the conjunction GLA×cap is implicated** (P-G0C0 40% side hit on the
Δ contrast; singles both null ⇒ conjunctive, as the 40% prior anticipated).
Not yet TitanMAC regime (T1 +35–44%, gains through T4): halfway there on the
K1 axis, trained-support K3−K6 still flat.

**Binding executed:** re-run the winning cell at TitanMAC's depth regime —
notul_bg0c0 + `model.mean_depth=8 model.max_depth=14`, 4500 steps, sweep
depths 1,2,3,4,6,8,10,12,14 (arm: notul-bg0c0-d8). Frozen before launch:
**P-D8: 45%** that d8 K1−K8 ≥ 0.30 nats (regime shift toward TitanMAC's
curve); sub-prediction 70% it trains clean (uncapped, deeper loop, TBPTT 4
unchanged). Runs after BG0-seed2 completes.

### Method amendment — 2026-08-31 17:30 (round 4: loop-detached GLA, frozen pre-launch)

Wolfe: "Taming GLA is complicated. Is there an arm testing it detached from the
loop but on the rest of the layers?" There wasn't; `retention_layers` applied
one index to all three sections. Added `model.retention_sections` (default
`[prelude, core, coda]`, byte-identity of base weights across choices proven
in `tests/test_retention_sections.py`, 4 passed). Correction on the record:
the seed-2 RNG-shift confound I named for BG0 was already dead by construction
(retention RNG draws are a tail; transformer.py comment at the attach site) —
BG0's stall was never an init artifact.

- **BGpc** — `notul_bgpc.yaml`: GLA at prelude.1 + coda.1 only, core clean,
  cap off. Reference points, same recipe: BC0 (GLA all, cap off): K1−K6 0.142,
  K6 4.401. BG0C0 (GLA none, cap off): 0.220, K6 4.206.
  **P-PC1 (frozen): 55%** that BGpc K1−K6 ≥ 0.19 (in-loop GLA specifically is
  the depth-killer; out-of-loop GLA harmless to the loop).
  **P-PC2 (frozen): 50%** that BGpc K6 ≤ 4.25 (out-of-loop GLA adds absolute
  value once detached; the alternative is GLA is net-harmful anywhere in this
  regime).
  Binding: P-PC1 TRUE + P-PC2 TRUE ⇒ memory branch stays, outside the loop —
  and a Raven-style RSM (goombalab; sparse routed slot writes) becomes the
  upgrade candidate FOR THOSE SITES, own prereg. P-PC1 TRUE + P-PC2 FALSE ⇒
  ship no-GLA no-cap; memory branch only returns as a causal decode-time
  mechanism. P-PC1 FALSE ⇒ GLA steals gradient from anywhere; remove, and do
  NOT port Raven into this position without a new hypothesis.

### Scoring — 2026-08-31 18:30 (P-G0s2, P-D8)

- **P-G0s2 TRUE** (75% side): seed-2 BG0 stalls identically (val 7.85→7.40→
  7.45 over 1250 steps). The GLA-off+cap-on stall is deterministic.
- **P-D8 FALSE** (55% side): notul-bg0c0-d8 K1 4.5119, K4 4.3284, K8 4.3191,
  K14 4.3253. K1−K8 = 0.193 < 0.30; sub-prediction (70% clean) TRUE — trained
  clean, and degrades gracefully past trained depth. Deeper training regime
  did NOT move the curve shape (still saturates by K4) and cost 0.11 nats
  absolute vs the mean-6 cell at equal steps. The conjunction removal buys
  shallow depth utility only; the remaining gap to TitanMAC's regime lives in
  the remaining diffs (attention stack / HC residual / QAT / injection /
  d1024 / data), or in the training dynamics of depth itself. Next-round axis
  decision deferred to Wolfe with tonight's BGpc + BWS results.
