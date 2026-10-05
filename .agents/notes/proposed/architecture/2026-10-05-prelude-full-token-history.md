# Agent Note: the prelude's own causal history (`tul.tg_strict_prelude`)

Status: proposed

## Problem

Wolfe, 2026-10-05: "we need to test the prelude and loop seeing full token history."

Under `tul.tg_geometry="strict"` (LXTUL's shipped geometry, `morph/configs/lxtul.yaml`)
the prelude is SAME-SPAN only (`tg_strict_allow(layout, "prelude")` — `causal AND
bag_id[i] == bag_id[j]`), and "the loop is the only cross-span channel" is the arm's
whole claim. That claim has never been tested with the prelude itself carrying the
row's full causal history into the loop's seed — every strict arm to date starves the
prelude of everything but its own span, so the loop's seed is a summary of ONE span,
not of the row read so far. Whether a richer seed changes how much the loop's depth is
worth (K1-K6) is an open question this key makes measurable.

## Proposal

`tul.tg_strict_prelude: str = "span"` (default, bit-identical) | `"causal"`.

At `"causal"`:

- A PRELUDE TOKEN query reads every earlier TOKEN of the whole row (`causal AND NOT
  slot_mask[j]`), not only its own span's — `tg_strict_allow`'s new `prelude_history`
  argument (`morph/model/tul_layout.py`).
- A PRELUDE SLOT-CELL query's relation is UNCHANGED (`bag_id[i] == bag_id[j]`, its own
  span's tokens and its own earlier cells). The loop still reaches the row only through
  the ordinary seed (pooled from the span's own, now-causal, prelude token states), not
  through a second route opened directly in the prelude.
- The prelude's CCA conv, its `W_v_prev` value shift, and the GLA retention carry stop
  resetting at segment boundaries (`_front_seg` / `_front_reset` go to `None` in
  `MORPHTransformer._tul_tg_kwargs`, which both `segment_causal_conv` and the GLA branch
  already treat as "run the ordinary unsegmented causal form" — their own defaults).
  Leaving these resetting while the attention relation widened would make "causal" a
  contradiction: attention sees the whole row, but the local conv keeps forgetting at
  every span edge.
- The CODA's own relation, its conv/value-shift/retention reset, and the zeroed
  per-cell injections are UNCHANGED at every value of this knob — it is a prelude-only
  widening, composed exactly like `tul.tg_coda_token_reach` is a coda-only one.

Refused outside `tg_geometry="strict"` (there is no strict prelude relation to widen at
`"restrict"`).

`tul.coda_token_input` (pre-existing key) composes with either value. `"prelude"`
(default) means a coda TOKEN's own carrier is the prelude's output at its own
position — under "causal" that carrier is now a global-context summary by itself, so
the bypass "strict" exists to cut is reopened at every token (measured, condition 3
below). `"embed"` swaps that carrier for the prelude's INPUT (the token's own embedding,
computed before the prelude runs and therefore unaffected by this knob), closing the
bypass so the slot loop is again the only cross-span route into the coda (condition 4).
`"embed"` was not refused under `tg_geometry="strict"` before this change and needed no
new refusal lifted — only tests, since it was untested there.

### A finding surfaced while testing, not hidden

"Slot positions keep exactly what they read today" is true of the ATTENTION RELATION
(proven structurally: a slot-cell query's allowed-key set is byte-identical between
`"span"` and `"causal"`), but NOT of the resulting VALUE at `n_prelude >= 2`: a slot
cell still attends its own span's tokens, and under "causal" those tokens' OWN prelude
output already absorbed cross-span content one layer earlier through THEIR widened
relation — composition, not a changed relation. Measured both ways in
`tests/test_tg_strict_prelude.py`: at `n_prelude=1` (no earlier layer to compose
through) a slot cell's seed is exactly unchanged by a 2-spans-back edit; at the
production depth it is not. This matches the existing `tg_coda_token_reach` precedent
("REAL RECEPTIVE FIELD: layers compose") rather than contradicting it.

A second finding: the conv/retention-reset half of this change (as opposed to the
attention-relation half) is NOT independently verifiable by any TOKEN-position
perturbation, because under "causal" a token's attention relation is already unmasked
at any distance — it alone carries a far edit regardless of what the conv does. The
sabotage "leave the conv segment-reset ON under causal" is invisible to a bulk
token/logit check for exactly this reason. It IS visible at a SLOT CELL (whose
attention relation never changes): the only way an edit in the previous span can reach
a slot cell is the conv contaminating the next span's own first token's key features
across the boundary. `tests/test_tg_strict_prelude.py`'s dedicated sabotage test reads
out there, at `n_prelude=1` (no composition confound), and does catch it.

## Alternatives considered

- **Widen the slot-cell query too** (so a cell may also read any earlier token
  directly). Rejected: this would open a SECOND cross-span channel beside the loop,
  exactly the bypass `tg_geometry="strict"` exists to cut — the whole point is that the
  loop, not the prelude, carries history forward; widening the cell's own relation would
  make the arm untestable against "the loop is the only channel."
- **Keep the conv/retention segmented even under "causal"** (narrower change, touch
  only the attention relation). Rejected as a contradiction in terms: a conv or a
  retention state that still forgets at every span boundary is not a causal model of
  the row, whatever the attention mask says — and it is exactly the class of bug named
  in `morph/model/CLAUDE.md` ("a module correct at its design shape can be degenerate at
  the shape a new subtree runs it at").
- **A separate per-query-type segment mask** (reset for slot-cell queries, no reset for
  token queries) to keep the slot cell's value, not just its relation, byte-identical.
  Rejected for this change: `tg_seg` is a KEY-side partition shared by every query that
  reads a given position, not a per-query-type one, and building a query-conditional
  version would be a second conv/retention code path beside `segment_causal_conv` and
  the GLA branch's existing `seg`/`reset_mask` contracts. The finding above (relation
  unchanged, composed value is not) is recorded instead of hidden behind a bigger patch
  that was not asked for.

## Acceptance criteria

- `tul.tg_strict_prelude="span"` bit-identical to a model that never set the key
  (state_dict, loss, logits) — `tests/test_tg_strict_prelude.py`.
- `"causal"` perturbation-verified end-to-end through the real
  `MORPHTransformer._tul_tg_kwargs` seam (never a hand-rolled kwargs dict — the
  2026-09-22 lesson): widens the prelude's token relation, carries into the loop's seed,
  and composes with `coda_token_input` exactly as documented (bypass open at
  `"prelude"`, closed at `"embed"`, forced depths 1 and 2).
- Two sabotages (conv-reset left on under causal; coda conv crossing a segment)
  explicitly checked, with the known-blind case (token-level check vs. this sabotage)
  reported rather than hidden.
- `morph/configs/lxtul_hist.yaml` / `lxtul_hist_embed.yaml` compose against `lxtul.yaml`
  and differ only in the stated keys (`tests` verified by hand via Hydra compose; no
  automated diff test was added — see Risks).

## Risks

- **Unmeasured on GPU.** Nothing in this change has run a training step; the configs
  are queued, not executed. The CE and K1-K6 effect of a richer seed is unknown.
- **Unmeasured cost.** The prelude's window branch now attends the full row per token,
  not one span — `O(S)` instead of `O(span)` per prelude layer, worse than the coda's
  already-bounded relation. No wall-clock or memory measurement exists yet; `base.yaml`
  / `lxtul.yaml` run at `seq_len` large enough that this could matter and was not
  profiled.
- **No automated Hydra-compose diff test.** The two new configs were checked once by
  hand (`OmegaConf.to_container` diff against `lxtul.yaml`); a future edit to either
  file could silently add an unrelated key with nothing failing in CI.
