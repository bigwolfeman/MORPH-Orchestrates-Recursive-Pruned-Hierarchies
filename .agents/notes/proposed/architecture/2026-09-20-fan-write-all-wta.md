# Agent Note: write every stream and let the coda pick — winner-takes-all responsibility on the register's route

Status: proposed

## Problem

The select arm (`2026-09-20-fan-select-then-commit.md`, prereg
`lab/experiments/planned/2026-09-20-lxtul-fan4-select.md`) measured, at 5,000 steps, that
winner-takes-all training makes the four streams individually useful (oracle 0.113 nats
below the best single stream, every stream within 0.076 of the best) and that a gate
choosing one stream BEFORE the span cannot cash it (0.014 of the 0.113; agreement with
the oracle 33 %). The write discards the other three streams: `prefix_project` receives one
state and `cells=None`, so the coda and the span decoder see one stream and nothing else.

The selection problem is causal. A per-slot choice before the span is a guess about the
next span's mode and the target decides the mode. The reader that has evidence is the
coda itself, per token, once the span's first tokens are in.

## Proposal

`tul.fan_mix: "all"`. Every stream is written into its own prefix cell through
`W_prefix[i]` — the Thought Register's 1:1 route (`prefix_project(cells=...)`, so
`prefix_k == fan_k`) — and the coda's own attention over the four cells is the selector.
No gate. Responsibility is winner-takes-all, the one thing the register did not have: at
train, K no-grad coda passes with stream i ALONE in its cell (the other cells blank, so
they carry `E_pass` only) score every span; the per-slot argmin is the winner
(`fan_select_eps` random on valid slots); ONE more pass with grad writes the winner alone
the same way and its token-weighted span CE is charged (`fan_all_wta_lambda`). "Stream i
alone" has one home, `_tul_fan_stream_write`: the deployed geometry minus the losers under
`all`, the single-source broadcast (unchanged) under the mixing modes. Keys: `fan/wta_ce`,
`fan/wta_weighted`, `fan/wta_oracle_ce`, `fan/wta_single_ce`, `fan/wta_pick0`,
`fan/wta_forced`, `fan/wta_share_k{i}`; the eval oracle (`fan/oracle_ce`, `fan/mixed_ce`,
`fan/stream_ce_k{i}`) reads the same one-stream write.

Cost over the select arm: one `_back_region` pass with activations per step.

## Alternatives considered

- **Per-token hard top-1 attention over the K cells** (true per-token winner-takes-all,
  the brainstorm's "states as experts"). Needs a custom attention path under the fused
  kernels; not built. It is the named next lever if the free read collapses the streams
  (prereg P-4).
- **Annealed credit with hard writes** (annealed MCL / LoRA-MCL; the research report's
  experiment 2): K coda passes with grad, softened responsibility cooled to WTA. Rejected
  for now on cost (K backward passes) and because it keeps the one-stream write whose
  selector is the problem.
- **BASE-style balanced commit on the select arm** (the report's experiment 1). Attacks
  reader starvation, which the select arm did not show (P-6 held, min share 0.078). Kept
  for a follow-up, not this arm.
- **A mixture of readers at eval** (K coda passes, gate-weighted token probabilities).
  Sound by Jensen but 4x decode cost against think-once, decode-cheap; the fallback if
  the coda's read does not select either.
- **The register's cells reader for the span decoder** (`spandec_reads_cells`). Left off
  so the arm differs from the register by ONE term; named in the prereg's Method.

## Acceptance criteria

The prereg's P-3, P-4 and P-5: the deployed write recovers the oracle to within 0.05
nats, the streams keep rank above 2.0 with axis cosine below 0.5, and the arm beats its
width partner pk4 on paired depth-6 CE by 0.005.

## Risks

- The register's collapse pressure is intact (shared core, shared write, the span decoder
  on the mean). One CE term may not hold four streams apart; `fan/stream_rank_t1` and the
  3070 probe read it, and the per-token hard read is the named next lever.
- The coda may average the four cells the way the softmax mixture averaged four states.
  P-1 and P-3 read it; a mixture of readers at eval is the named fallback.
- The WTA term is a full CE (about 4.6 nats) beside `ce_main` and the span decoder's
  term, so `fan/wta_weighted` is a third of the total loss; the prereg reads the term's
  GAP to `ce_main`, not its share.
- One more coda pass with activations per step: peak memory and rate are read from the
  Spark smoke before the queue line is written.
