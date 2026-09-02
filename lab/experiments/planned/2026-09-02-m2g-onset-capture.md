# Planned: M2G onset capture — is the paid-axis detonation stale-m2?

Status: planned
Date: 2026-09-02 (frozen ~15:40, before the diagnostic draw)

## Question

Does the paid-axis detonation carry the Task #276 stale-momentum signature
at onset (medRatio_am_gg >> 1 with medCos_g_num -> 0, per the
TROUBLESHOOTING.md §10 playbook), confirming the mechanism before we spend
a 20k run on the mechanism-matched knob?

## Method

ONE diagnostic draw: `tul_a2` unmodified (ternary ON — the failing
configuration), panel flags, 1200 steps (every observed onset is by ~750),
`MORPH_DIAG_M2G=$Q/tul-a2-diag/m2g.jsonl`, grad probe on. The draw is a
lottery ticket: ~50-70% it detonates. If it detonates: read the m2g rows at
the probe's onset step. If it stays healthy: geometry baseline only, no
onset — the capture is re-armed on the next GPU window, not scored FALSE.

## Predictions (frozen)

- **P-D1.** Conditional on the draw detonating (preclip/total > 1e4): the
  m2g log shows median |alpha*m2|/|g| > 10 AND median cos(g, numerator)
  < 0.2 within ±100 steps of onset — the stale-m2 signature: 70%.
- **P-D2.** The onset step's m2g rows localize the worst ratio in core
  gate_up tensors (the cusp-vault precedent, core.1.gate_up): 50%.

## Binding

P-D1 TRUE => mechanism confirmed; the fix arm is t_alpha=8000 (the one
config deviation from the Task #276 cure), coord-cap 0.25 in reserve.
P-D1 FALSE on a real detonation => the paid-axis disease is NOT (only)
stale-m2 — reopen before burning the 20k.
