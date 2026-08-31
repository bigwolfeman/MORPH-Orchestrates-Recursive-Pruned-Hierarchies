# Agent Note: the two objective lines vs the winning recipe — FM and DB postmortem

Status: implemented

Companion to
[2026-08-30-l2cap-winning-recipe.md](2026-08-30-l2cap-winning-recipe.md) (the
recipe itself). This note records how the two OBJECTIVE lines — flow matching
and DiffusionBlocks — matched up against it, and where each line's tested
boundary lies. Both are closed by binding rules; the pointers below are the
evidence.

## Problem

Two families promised to replace or augment plain CE-through-the-loop:
**FM** (predict the slot plan directly — a planner regressing slot states) and
**DB** (read the loop as reverse diffusion — denoise instead of iterate). Each
consumed a campaign. Neither survived contact with the sweep. A future reader
proposing either again needs the terminal numbers and the exact boundary of
what was and was not tested.

## Decision

Both lines are CLOSED against the l2cap substrate; the loop program continues
on plain CE + mux under the capped recipe.

**FM line (5 filings, terminal: `lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md`).**
The arc (rejected:
[../../rejected/architecture/2026-08-28-tul-fm-arc.md](../../rejected/architecture/2026-08-28-tul-fm-arc.md))
decomposed the slot-path failure into write/dynamics/read and proposed P1: a
planner predicting slot states from context. P1 and its variants (whitening,
objective swaps — the 2026-08-28 tulfm filings) never beat floors on the old
substrate. The decisive gate re-ran P1 on the l2cap checkpoint — the best
substrate that will ever exist for it: trained planner ≈ untrained ≈ shuffled at
every scope (within-row top-1 0.0167 vs floor ~0.020, bar 0.06; MRR 0.0858 vs
0.12), controls clean. **Binding rule: the FM planner line stays dead.**
Boundary: every P1-family design targeted PRE-core prelude features (pairwise
cos 0.50, eff-rank 40/1024 — clustered, hard targets); post-core carriers were
never tested by any FM design. That is the one live opening, parked as a NEW
design, not a revival.

**DB line (terminal: `lab/experiments/successes/2026-08-30-tul-dbfix-pair.md` +
the winning-recipe note's table).**
Whole-model DB was rejected 2026-08-21
([../../rejected/feature/2026-08-21-diffusionblocks-verdict.md](../../rejected/feature/2026-08-21-diffusionblocks-verdict.md):
plain NTP 4.0010 vs best DB arm 5.0801 at matched tokens). The loop-side
resurrection tried three mechanisms — target scheduling (l3), the
paper-faithful σ+EDM one-pass with Euler-ladder inference (dbfix), iter-AdaLN
conditioning (db_cond) — all stable, all decent CE, all depth-dead or
depth-inverted; the 50/50 interleave erased l2cap's curve outright. The paper
audit matters for the record: arXiv 2506.14202 never measured a depth-vs-steps
curve for its Huginn variant and never compared against flat compute — our
K-sweep was the first measurement, and it ran backwards. **Binding rules:
interleave cancelled; conditioning banned; DB does not transfer to TUL slot
geometry at this budget.** Boundary: DB-as-wall-clock on a NO-loop TUL is
untested — that is Wolfe's dmorph program, recorded on `feat/db-objective-l2`
(`.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md`), scored on
token CE at matched wall-clock AND matched tokens, K≤4 inference.

**The common failure shape.** Both lines died the same way the gate did: they
gave the optimization an alternative to building composition through the
iterated map — FM by routing plan formation around the loop, DB by replacing
iteration with conditioning-plus-inference-recurrence. The identity-escape law
in the winning-recipe note is the unifying statement; this note is its
objective-line annex.

## Alternatives considered

- **Fold this into the winning-recipe note**: rejected — that note owns the
  recipe and its ablation table; the objective lines have their own arcs,
  boundaries, and revival conditions, and burying them would lose the
  dmorph/post-core-carrier openings.
- **Leave the record distributed across the seven filings**: rejected — that is
  the state Wolfe's question exposed; filings are per-experiment, and the
  cross-line verdict lived only in chat and vlt.
- **Reopen either line at larger budget first**: rejected — both terminal gates
  ran on the best available substrate (l2cap) with clean controls; the recorded
  openings (post-core carriers, dmorph) are new designs, not budget retries.

## Consequences

- Any future FM-flavored proposal must target post-core carriers or it is
  pre-refuted; any DB-flavored proposal must be loop-free (dmorph) or it is
  pre-refuted at this scale.
- The loop program's live lines are exactly: G3 decode-gate and G4 uniform
  depth, both on unmodified l2cap.
- The FM arc note moves to rejected/ in the same change, its decomposition intact
  and the rejection stamped at the top.
