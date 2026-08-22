# Agent Note: CMS→MORTAR mechanism figure (candidate)

Status: rejected — not different enough from morph_cms_lifecycle.png

## Problem

`morph_cms_lifecycle.png` is a four-phase schedule poster (Dense → Prune → Carve →
ReMoE). It does not teach the mechanism gaps operators hit: **16×16 score tiles
nested in 128×128 exec blocks**, the **backward → accumulate → zero_grad**
contract, or the **dense-carve** failure when prune never fires. ReMoE shares
equal visual weight with CMS/MORTAR even though it is an orthogonal activation
axis.

## Proposal

Add a companion mechanism figure for visual keep/drop review:

- Source: `docs/figures/sparsity-routing/morph_cms_mortar.tex`
- Preview: `docs/figures/morph_cms_mortar.png`
- MANIFEST row under Sparsity & routing

Stars SCORE → PRUNE → CARVE, with timing rail, dense-carve gotcha, and ReMoE as
a thin coda. Recipe integers stay in `base.yaml`. Lifecycle poster remains the
schedule view. Do not link from README/CLAUDE until keep is confirmed.

## Alternatives considered

- Rewrite lifecycle in place — rejected while under review; loses the clear
  A/B1/B2/C schedule read.
- Fold only the gotcha into the lifecycle facts panel — rejected; nesting +
  timing deserve ink.
- **Keep the companion** — rejected after visual review: reads as a restatement
  of the lifecycle poster, not unique enough to retain.

## Acceptance criteria

- Side-by-side with `morph_cms_lifecycle.png`, the new figure teaches at least
  one of: tile nesting, accumulate timing, dense-carve gotcha — without reading
  as a restatement of the lifecycle poster.
- Compile recipe in MANIFEST works; no recipe step numbers hardcoded as gospel.
- Keep → move this note to `implemented/` and link from CLAUDE/docs as needed.
  Drop → delete tex/png, revert MANIFEST, reject this note.

## Risks

- Two sparsity figures confuse readers about which is canonical.
- Dropping after merge leaves MANIFEST/link churn — keep the note in `proposed/`
  until the visual call is made.
