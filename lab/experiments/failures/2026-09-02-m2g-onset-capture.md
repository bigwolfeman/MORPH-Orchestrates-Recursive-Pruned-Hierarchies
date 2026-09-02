# Planned: M2G onset capture — is the paid-axis detonation stale-m2?

Status: failure
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

## Results (2026-09-02, run tul-a2-diag, 1200 steps, detonated)

The draw detonated (probe max 6.41e12 at step 1078) with full M2G coverage.
Onset is a SPIKE TRAIN: preclip/total 1.6@300 -> 485@330 -> 9.1e3@333 ->
2.6e4@334 (core 2.4e4 of it) -> RECOVERS to 238@335 and 76@340 -> spikes
again 2.3e4@360 -> calm 1.5e3@500 -> the cluster sticks and compounds to
6.4e12@1078. Frame-matched to the Task #276 cusp-vault record (single-step
forward explosions, mostly recovering, divergence when a cluster lands).

- **P-D1 (70%): FALSE.** No stale-m2 signature at onset: M2G |m2|/|g|
  medians ~0.05-0.2 (nothing near 10), medCos_m2g low throughout early
  training rather than dropping AT onset. m2 is ~330 steps old — too young
  to dominate. Instrument caveat: M2G reads post-clip g, so the ratio is
  against clipped gradients; the qualitative absence of domination stands.
- **P-D2 (50%): FALSE.** Worst M2G ratios sit in tiny attention scalars
  (cca.temp, sink_logits, comp_norm), not gate_up — but the probe's
  group-level epicenter IS the core (2.4e4 of the 2.6e4 spike), and M2G's
  per-tensor g is post-clip, so tensor-level localization from M2G is not
  trustworthy for magnitude.

## Verdict

**FAILURE of the predictions; decisive mechanism finding.** Per the frozen
binding (P-D1 FALSE on a real detonation): the paid-axis disease is NOT
(only) stale-m2 — it is the ternary cusp VAULT compounding under clipping,
with no amplifier needed. Consistent with the ternary-off result (3/3
clean) and with why the full Task #276 cure (coord cap + SNR gate, both
verified active in the fused kernel) does not prevent these detonations:
that cure treats the stale-momentum amplifier, and this failure mode's
trigger fires upstream of it.

## Updated hypothesis

Mechanism-matched fix = stop the vault at the source: freeze or slow-EMA
the ternary scale gamma (DIRECTION-REVIEW §6(ii)'s unbuilt proposal), so a
coherent drift of mean|W| cannot re-threshold a whole tensor in one step.
t_alpha=8000 and coord cap 0.25 are demoted to second-order levers.
Validation: 3 paid draws with gamma-EMA at the 2500-step window, then the
walkover 20k.
