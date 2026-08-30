# Agent Note: gate ladder — cap vs GRT gate vs both, plus the TUL decode-gate revival

Status: proposed

## Problem

The loop program has one winning recipe (l2cap: full BPTT + hard σ≤1.5 projection,
0.233 nats depth-earned, the only greedy-resistant arm) and two freshly closed
lines (interleave, iteration conditioning — see
`lab/experiments/successes/2026-08-30-tul-ilv50-l2capcond.md` and the cond-zero
probe). Two open questions remain: (1) is contractivity-by-architecture (a learned
convex gate) better than contractivity-by-constraint (the projection)? (2) now
that the loop earns depth, does the ORIGINAL TUL gate — k=0 "keep thinking",
k>0 "decode k tokens from the latent" — finally have something real to steer?

Hard constraint from the cond-zero probe (binding, formation-level): nothing in
the training graph may be keyed on the iteration index — no iteration embeddings,
schedules, or per-iteration scalars. Index-keyed signals poison depth-earning
during formation even when the trained model abandons them functionally. All
gates below are state-keyed and satisfy this by construction.

## Proposal

Five arms on the panel recipe (batch 6, 4500 steps, seed 1, eager, mux on —
l2cap's exact base), preregs per pair before any launch:

- **G0 — cap only.** The l2cap control. Already run (CE@4250 4.3489, 0.233 nats,
  greedy rep4 0.61); reuse, do not retrain.
- **G1 — GRT gate, no cap.** The verified Eqs. 2–5 mechanism (close read
  2026-08-30, PDF pages 4–5): per-element gate g = σ(f_g([LN(h), LN(h_pre)])/τ +
  ε_g), f_g a 2-layer SiLU MLP (hidden d), second-linear bias init +4 (their own
  ablation: bias ∈ {−2..+4} spread only 0.019 nats — the init is NOT the load-
  bearing part); update h ← g⊙h_prev + (1−g)⊙o. MORPH adaptation: our core
  already injects the prelude per iteration (DiagonalInjection/ChannelInject), so
  we do NOT import their W_proj re-injection (Eq. 2) — the gate and its MLP are
  the only additions. VERIFY the injection-equivalence claim against
  transformer.py before building.
- **G2 — GRT gate + cap.** Both controls. If G1 diverges and G2 trains, the gate
  alone is insufficient contractivity control at our depth; if G1≈G2, the cap is
  redundant under a gate.
- **G3 — TUL decode-gate revival (Wolfe's variant).** The original tul-gate-spec
  mechanism on the l2cap recipe: one gate value per slot, k = round(g·k_max);
  k=0 ⇒ loop again; k≥1 ⇒ decode k tokens from the latent (slot as coda prefix +
  budget_embed(k), token path intact). MANDATORY fixes from the spec's own §7
  post-mortem: (a) split the STOP decision from the LENGTH value (two heads — a
  halt logit and a length head — never one scalar whose tiny values mean "stop
  and emit 1"); (b) the halt head must train against gate-driven or state-
  dependent depth, never regress on the input-independent Poisson draw (the
  hazard-function collapse, spec §7 table); (c) the length label stays the data's
  own next-span length from BoundaryRule (self-supervised, no reward). The 2026-
  08-21 bakeoff never measured this gate — arms died of the TUL-short recipe
  divergence (since cured by the GL→l2cap line) and OOM headroom; gate footprint
  was 0.07 GB.
- **G4 — depth-sampling variant.** Best-of-G0..G2 with uniform-{1..R} training
  depth instead of clamped Poisson(6) (which almost never trains depths 1–2;
  GRT's emergent early exit — 92% at half depth — comes from every depth being
  trained). Cheap and orthogonal.

Sequencing: G1+G2 first (one prereg pair — the contractivity question), then G3
(its own prereg — the decode-gate question needs the winner's loop as substrate),
G4 folded wherever a retrain is already happening.

Known ceiling for G3, recorded up front (spec §7): per-slot halting saves no
compute in batched inference (masked update, batch runs max(T) anyway); its case
is quality + single-stream generation wall-clock. Score it on CE, depth sweep,
and the generation panel — not on batched throughput.

## Alternatives considered

- **Index-keyed gates or per-iteration learned scalars**: excluded by the
  cond-zero probe (formation-level poisoning; no strip-at-inference salvage).
- **Importing GRT wholesale (W_proj + noise init + gate)**: rejected for round 1 —
  W_proj duplicates MORPH's existing per-iteration prelude injection and confounds
  the gate question; revisit only if G1 fails in a way that implicates input
  anchoring.
- **Reviving the single-scalar TUL gate unchanged**: rejected — §7's hazard
  arithmetic shows it halts at t=2 emitting 1 token at converged low loss; the
  encoding, not the threshold, is the defect.
- **Scoring G3 on batched-inference speed**: rejected — invariant §6b masked
  update makes it a null measurement.

## Acceptance criteria

- Frozen prereg per launch pair, panel flags, one trainer on the 5090 at a time.
- G1/G2 gate is state+prelude-keyed only; a grep for iteration-index inputs into
  any gate MLP comes back empty, and the smoke prints the gate's input sources.
- G3 has two heads (halt, length); the halt head's training signal is verified
  state-dependent before launch; generation follows spec §8 (boundary token kept
  and fed to the cache before breaking).
- Every arm gets the 48-row depth sweep + the generation panel; depth-earned CE
  and greedy rep4 are the paired readout (mechanism tracks generation, campaign-
  wide).

## Risks

- The GRT gate's copy branch (g≈0.98 at init) may slow early training at our
  4500-step budget — their result is at full-horizon GPT-2 scale; the bias init
  may need the {−2..+4} sweep repeated at our budget.
- G3's gate-driven depth changes the training graph per slot; interaction with
  the σ-cap projection is untested (the cap projects weights, so it should
  compose, but the smoke must confirm gradients cross the gated iterations).
- G4's uniform depth spends half its samples at shallow depths — at a fixed
  4500-step budget this may cost absolute CE even if it buys early-exit
  robustness; that is the measurement, not a bug.
