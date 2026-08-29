# Agent Note: Generator emit_source — span-first tokens must come from a trained position

Status: implemented

## Problem

Wolfe spotted that generation samples drop the space after sentence punctuation
("story.Michigan"). Measured: 31–45% of boundaries in the gl1b/gl1-ctrl samples
vs 12.5% for the no-TUL twin, ~1% at commas, and 0.34/1k chars in real OWT text
(nearly all abbreviations). Root cause is a train/generate mismatch: the GL line
trains with `tul.emit_weight = 0`, so a span's first token is supervised ONLY at
the boundary-token position (`pack_tul_row` token labels skip slots), but
`generate_tul` sampled it from the slot's emit position (spec §6 v1) — a readout
that is untrained (ctrl) or MUX-shaped toward the PREVIOUS span (gl1b). The
probe `lab/divergence/emit_space_probe.py` closed the chain: emit-position space
mass 0.473/0.576/0.797/0.826 (ctrl/gl1b/gl1c/gl1) ranks the sample artifact
4/4, while the token position is flat at 0.80–0.81 across all arms.

## Decision

Add `emit_source ∈ {"slot","token"}` to `generate_tul` and `generate_tul_batch`
(`morph/inference/tul_generate.py`). `"token"` reads the boundary TOKEN position
(`-1 - prefix_k` when the row ends with a fresh slot); everything else about the
procedure is unchanged. Default stays `"slot"` (spec §6 v1 — no silent behavior
change). `scripts/tul_samples.py` and the trainer's gen path select `"token"`
when the arm's `emit_weight == 0`. Spec deviation recorded in
`docs/tul-spec.md` §6. Tests: `tests/test_generation_sampling.py` (position-
coded stub asserts the read index per mode, default, batch parity — 17 pass).

## Alternatives considered

- **Train the slot readout (emit_weight > 0):** rejected — FM2 measured direct
  emit supervision both absorbed by the shortcut and degrading the writer; the
  GL line's emit_weight=0 is deliberate (tul_gl1.yaml comment).
- **Constrained decode (ban no-space tokens after punctuation):** rejected —
  cosmetic band-aid that hides the untrained-readout fact instead of fixing the
  read.
- **Change the default to auto-detect:** rejected — the generator cannot see the
  training config; callers can, and an explicit argument keeps spec v1 exact.

## Consequences

Emit-off arms sample every span-first token from the same trained distribution
as the no-TUL baseline; prediction (registered in vlt tul-span-jepa/123-124
before the rerun) is that all four arms' missing-space rate collapses to the
~12.5% undertrained baseline. When a future arm turns emit supervision back on,
pass `emit_source="slot"` and the original spec path is exercised unchanged.
