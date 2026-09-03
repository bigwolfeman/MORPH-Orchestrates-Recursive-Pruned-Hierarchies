# Planned: mask-surgery decomposition — what does TUL's visibility cost a model that never adapted?

Status: success
Date: 2026-09-01 (frozen before the harness runs; rung E1 of
.agents/notes/proposed/architecture/2026-09-01-slot-channel-recovery.md).

## Question

How much of the 20k head-to-head's 0.357-nat gap is information deleted by
the TG-restrict visibility pattern itself, versus TUL's mechanism failing or
optimization tax?

## Hypothesis

The visibility restriction is the dominant term: imposing TUL's pattern on
the TRAINED noTUL checkpoint at eval costs a large fraction of the gap, and
the cost concentrates on span-first tokens (the positions whose targets
depend on cross-span context).

## Method

Eval-only surgery on `checkpoints/morph/notul-20k/step_20000.pt`, eager path,
48 rows x seq 1024 (same protocol as the token depth sweep), depth = trained
mean 6. Per-attention-module forward replacement mirroring the tg_restrict
branch verbatim (project -> `_tg_slot_attention` over CARRIER positions +
`_window_attn(extra_mask)` -> `_gate_combine_up`), applied to BOTH attention
classes at every layer. Spans from `BoundaryRule.cut` on the raw token
stream (no slot insertion); the CARRIER of span j is its boundary token —
the position a slot would sit after. Variants, all sharing the carrier
compressed branch to avoid the M=0 edge:

- **base** — unpatched forward. Sanity: must reproduce the sweep's K6 CE
  3.5089 within noise (same rows protocol, fresh draw).
- **E1c** — window branch UNRESTRICTED (extra_mask None), compressed branch
  -> carriers. Prices the pooled-compressor/top-k deletion alone.
- **E1b** — window same-span-or-carrier, compressed -> carriers. The full
  TUL-visibility analogue.
- **E1a** — window same-span-only, compressed -> carriers. Harsh floor.

Readouts: CE per variant; Δ vs base; CE stratified by within-span offset
(offset 0 = span-first token) for base and E1b.

## Predictions (frozen)

- **P-M1.** Δ(E1b) ≥ 0.25 nats (the mask is most of the 0.357 gap): 60%.
- **P-M2.** Δ(E1c) ≤ 0.10 nats (the cost is the window restriction, not the
  compressor swap): 65%.
- **P-M3.** Span-first stratum: Δ(E1b, offset 0) ≥ 2x Δ(E1b, all): 70%.
- **Binding.** P-M1 TRUE ⇒ the write channel carries the program (E2/E3
  bound seed next as planned). P-M1 FALSE (Δ ≤ 0.10) ⇒ the gap is mechanism
  or optimization tax ⇒ promote E4 teacher-distill, demote E3. Middle
  (0.10 < Δ < 0.25) ⇒ split verdict; run E2 (free) and decide E3-vs-E4 with
  the stratification.

## Not verified before run

The tg branch has only ever executed on tg_restrict=True builds; the patched
path on a normal build is new code (mitigation gate, run FIRST: patched
forward with extra_mask=all-causal AND compressed-branch UNPATCHED must
byte-match the unpatched forward — if that shim gate cannot be constructed
cleanly, base-vs-E1c carries the compressor-deletion caveat instead).
Carrier-token reads are a proxy for slot reads; E1a/E1b bracket, not point.

## Results — 2026-09-01 09:31 (single eval pass, exit 0)

Gate PASSED bitwise (max|Δ|=0.000e+00). Sanity anchor EXACT: base ce 3.5089
== the token depth sweep's K6 3.5089 (same deterministic rows). 48 rows,
51.5 carriers/row. Artifact: /home/wolfe/morph-scratch/tulfm/mask_surgery.json.

| variant | ce | Δ | span-first ce | Δ |
|---|---|---|---|---|
| base | 3.5089 | — | 3.0650 | — |
| e1c (comp→carriers only) | 3.7399 | +0.2310 | 3.2763 | +0.2113 |
| e1b (full TUL analogue) | 3.9963 | +0.4874 | 3.7299 | +0.6650 |
| e1a (same-span only) | 4.0613 | +0.5524 | 3.8515 | +0.7866 |

- **P-M1 TRUE** (60%): Δ(e1b)=0.487 ≥ 0.25 — the visibility restriction
  costs MORE than the whole 0.357 TUL gap on a never-adapted model.
- **P-M2 FALSE** (65% miss): Δ(e1c)=0.231 — deleting the pooled global
  compressed branch is ~half the damage alone. The noTUL model leans
  heavily on global compressed attention.
- **P-M3 FALSE** (70% miss): span-first Δ 0.665 vs all-token 0.487 = 1.36x,
  not ≥2x. The restriction hurts every position, not mainly span-firsts.

## Verdict

Predictions 1/3; the deciding P-M1 held. The gap decomposition: TUL's
mechanism already recovers ~0.13 nats over naive masking (0.487 masked cost
vs 0.357 realized gap), and roughly half the information loss is the global
compressed branch, half the window restriction. **Binding applied: the write
channel carries the program — E2/E3 proceed.** Design note for E3+: slots
replace a GLOBAL branch, so the read side must eventually serve global
access (Raven/R3 territory), not only span summaries.

## Updated hypothesis

The 0.357 gap is information cost the slot channel fails to compensate; a
TUL variant only closes it if slots both (a) carry span content and (b)
restore some global read capacity. Span-first tokens are the worst-hit
stratum but the cost is broad.
