# Agent Note: the l2cap recipe — the winning loop formation recipe

Status: implemented

## Problem

The 2026-08 campaign had to find a training recipe under which MORPH's TUL core
loop actually EARNS its iterations — measurable as depth-earned CE on the paired
forced-depth sweep — after every earlier loop arm trained to good CE with a loop
that did nothing. Fifteen-plus arms later, exactly one recipe wins.

## Decision

**The winning recipe (config: `morph/configs/tul_l2.yaml`, arm name l2cap):**

1. **Full BPTT through the slot loop** — `bptt_depth ≥ max_depth` (8 ≥ 8), so the
   gradient of every loss crosses every realized iteration. Non-negotiable:
   truncation to 4 retains ~4% of the depth curve (l2trunc).
2. **Hard spectral projection, σ_max ≤ 1.5**, applied to the 12 core MLP linears
   after EVERY optimizer step (`training.spectral_project_cap: 1.5`). The
   projection, not the soft penalty (the penalty lost this fight in the takeover
   campaign). Present from step 0 — contractivity control must be there DURING
   formation; it cannot wake a loop post-hoc (l3wake, l3wake-cap).
3. **TUL slot geometry with tg_restrict** — the loop runs on ≤64 slot positions,
   which is what makes full BPTT affordable, and the slot is the only cross-span
   route, which is what gives the loop something to carry.
4. **Mux on** (`mux_beta 1.0, mux_target own, mux_detach_head false`,
   active from step 0) — a state-keyed per-slot local objective; coexists with
   depth-earning.
5. **Nothing else touching the iterations.** No stage conditioning, no
   recurrence gate, no step-mode mixing, no per-iteration anything. This clause
   is load-bearing — see the law below.

**Measured, at the 4500-step / batch 6 / seed 1 / eager panel:**
depth-earned CE 0.233 nats (4.622@K1 → 4.389@K6, 48-row paired sweep), CE@4250
4.3489, S1-clean, 66 min, and uniquely greedy-degeneration-resistant (rep4 0.61
vs 0.82–0.89 for every dead-loop arm; the plan mechanism is what buys greedy
health). Filings: `lab/experiments/successes/2026-08-29-tul-loop-ladder.md` and
the 2026-08-30 series beside it.

**The identity-escape law (three falsifications deep).** Any mechanism that lets
an iteration cheaply approximate identity is taken by the optimizer and the
composition never forms — and the damage happens during FORMATION, functionally
abandoned modules included:

| falsified add-on | depth earned | filing |
|---|---|---|
| (recipe intact — the control) | **0.233** | loop-ladder |
| iter-keyed AdaLN conditioning | 0.013 | ilv50-l2capcond |
| ...same, conditioning zeroed post-hoc | 0.013 (unchanged, −0.001 CE) | condzero-probe |
| σ+EDM one-pass (faithful DB) | inverted (worse with K) | dbfix-pair |
| 50/50 bptt/db1 step mix | 0.000 | ilv50-l2capcond |
| state-keyed convex gate (GRT), no cap | 0.000 (gate never opened) | gate-pair |
| state-keyed convex gate + cap | 0.000 (gate OPENED, front-loads then copies) | gate-pair + gate-value-probe |
| BPTT truncation to 4 | ~0.010 | l2-trunc |

The σ-cap wins because it bounds the map's expansion WITHOUT offering an
identity escape: every iteration is forced to transform. Corollary for any
future gate: output-bounded gated-delta `h + α(h)·f(h)` with α floored above 0
is the only admissible shape; anything allowing α→0 is presumptively dead.

## Alternatives considered

Each was a real arm, run under a frozen prereg, and lost on the sweep:
contractivity-by-architecture (GRT convex gate, alone and stacked); iteration
conditioning (AdaLN, iter- and σ-keyed); one-pass DB training with Euler-ladder
inference; interleaving one-pass steps for wall-clock; truncated BPTT for
memory; post-hoc wake-up of a DB-formed loop (AdamW, with and without cap); the
soft spectral penalty instead of the projection (earlier takeover campaign).

## Consequences

- G3 (the TUL decode-gate: k=0 keep-thinking / k>0 decode-length, two-head
  encoding per docs/tul-gate-spec.md §7) builds ON this recipe unmodified; so
  does G4 (uniform {1..R} depth sampling).
- Wall-clock improvements must come from kernels/compile (the eager TG attention
  is the big unmeasured lever), never from touching the iteration structure.
- `morph/configs/base.yaml` still carries `bptt_depth: 4` — WRONG for capped
  recipes; flipping it (and porting the cap into base) is an open decision for
  Wolfe, flagged, not yet made.
- Panel caveats, stated: one seed at 4500 steps (replicate CE spread 0.030–0.036;
  the depth-curve gap 0.233-vs-0.000 is far outside it), d_model 1024 perf-branch
  variant, eager kernels, and MORPH's retention_carry non-causality applies to
  absolute CE numbers as everywhere.
