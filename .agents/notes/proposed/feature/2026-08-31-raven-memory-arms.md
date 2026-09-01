# Agent Note: Raven memory arms (R1/R2/R3) for post-campaign MORPH

Status: proposed

## Problem

The 2026-08-31 loop-killer bisect removed GLA entirely: it was a
training-dynamics crutch for the σ-cap (cap-on without it stalls at unigram)
and a gradient parasite everywhere once the cap is off (monotone harm per
site: earning 0.142→0.186→0.220, absolute CE 4.401→4.303→4.206 as sites are
removed; `lab/experiments/planned/2026-08-31-loop-killer-bisect.md`). That
verdict BINDS: no memory mechanism returns as a parallel residual branch
without a new hypothesis. But two capabilities left with GLA:
long-range recall beyond the attention window, and the decode-time
persistent-summary conditioning that made the (leaky) carry model's
generation less repetitive (greedy rep4 0.614 vs 0.902 honest;
`lab/experiments/successes/2026-08-31-carry-leak-audit.md`).

Raven (goombalab, "High-Recall Sequence Modeling with Sparse Memory
Routing"; PDF read 2026-08-31) supplies the missing design family: M slot
memories, sparse input-dependent top-K write routing, decay applied only to
written slots (unselected slots frozen exactly), SWA-like K/V slot states
with softmax readout. GLA is the dense-router row of their Table 1; Raven is
the sparse-router corner. Near-perfect NIAH recall at 16× training length
where GLA collapses to ~2%.

## Proposal

Three arms, cheapest-first, each its own prereg. All satisfy the binding
because none is a parallel residual branch:

- **R1 — replacement mixer, out of loop.** Raven block REPLACES attention in
  prelude/coda layers (run-once sites; their LM results transfer most
  directly). Core keeps CCA/CSA/HCA untouched. A replacement mixer does not
  compete with a main path for gradient — it IS the layer (new hypothesis,
  to be measured). Readout: CE parity + depth sweep + synthetic NIAH probe.
- **R2 — intra-layer hybrid.** Swap the HCA dense-compressed half of the
  attention module for a Raven RSM; keep CSA sparse-global. Their hybrid
  results say this is where recall-per-state-byte wins. Higher wiring risk
  (touches `attention.py`'s fused module).
- **R3 — causal span-slot decode memory.** The MORPH-native design. When a
  TUL span ends, a router writes that span's summary into top-K of M
  persistent slots (decay only on written slots). Tokens in span j read the
  memory as of the END of span j−1 — causal by construction in training AND
  decode; no position can read its own label. Slots surface as prefix
  positions the coda attends to (TUL slot machinery reused). This recovers
  the carry's generation benefit with zero leak surface, and the frozen-slot
  property gives protected long-range recall. Sequential recurrence over
  ≤~64 spans/row; Raven's chunk-parallel form applies with spans as chunks.
  Readout: gen rep4/distinct-3 toward 0.614 at no CE regression + planted
  long-range recall probe + memory-off ablation + early-training
  gradient-competition probe (parasitism check, not assumed away).

Sequencing gate: the σ-cap replacement (decay-parameterized contraction,
TitanMAC-style) lands FIRST — the shipping no-cap config detonated under a
small in-loop perturbation (BWS, `failures/2026-08-31-gla-write-alignment.md`)
and stability margin must not depend on luck before long runs.

## Alternatives considered

- **Port Raven into GLA's parallel-branch position** — rejected: BGpc's
  monotone-harm verdict binds against it; a stronger fast memorizer likely
  sharpens the bypass.
- **Falcon/FWA write-alignment retrofit on GLA** — tested and dead
  (failures/2026-08-31-gla-write-alignment.md): no bootstrap value capped,
  detonates uncapped. NLMS-normalized form shelved; only relevant if a
  branch-shaped memory ever returns.
- **Chunk-boundary dense carry** — rejected earlier (2026-08-31 fix agent):
  double-counts the attention-visible prefix; R3's sparse selective routing
  is specifically the repair for that failure mode.
- **Do nothing** — keeps the honest model's repetitive generation
  (rep4 0.902) and window-bounded recall.

## Acceptance criteria

- R1: CE within noise of the no-GLA baseline at 4500 steps AND ≥5× GLA's
  recall on the synthetic probe; no stall, no detonation.
- R3: rep4 ≤ 0.75 at distinct-3 ≥ 0.12 with CE within 0.02 nats of baseline;
  memory-off ablation shows the gain comes from the slots; planted-fact
  recall beats the no-memory model at ≥4k-token distance.
- Any arm that stalls at unigram or trips the div-guard is filed under
  failures/ with its probe forensics, per campaign practice.

## Risks

- Router trains on sparse top-K gradients at our scale/objective (Raven
  reports it works at 400M/800M LM; unverified here).
- R3 span-summary pooling must detach or avoid the weight-tied LM head
  (morph-lm-head-is-weight-tied lesson).
- R2 touches the most entangled module in the tree.
- Double-counting vs attention is mitigated by selective writes but only the
  ablation proves it.
