# Agent Note: CSA's sparse selection never fires on the TUL short schedule

Status: proposed

## Problem

`_CCACSAAttention.forward` computes `n_blocks = S // csa_compress_ratio`, scores the blocks
with the Lightning Indexer, and then selects

    tk = min(self.top_k, n_blocks)

`base.yaml` ships `top_k: 256` and `csa_compress_ratio: 8`. Whether that is a SELECTION
depends entirely on the sequence length:

| path | S | `n_blocks` | `tk` | selection |
|---|---:|---:|---:|---|
| deploy, `seq_len 4096` | 4096 | 512 | 256 | **fires** — half the blocks are dropped |
| TUL short schedule, `seq_len 1024` | 1152 | 144 | 144 | **inert** — every block kept |
| TUL slot core, `max_slots 64` | 64 | 8 | 8 | **inert** — every block kept |

Measured 2026-08-25 (`lab/divergence/attn_sink_probe.py --geometry --token-path`, `tul_a1`,
batch 6): `distinct_top_idx == n_blocks` at every recorded call on both paths, 144 of 144
and 8 of 8.

So on the whole TUL short schedule CSA runs as **dense pooled attention**, not sparse
attention. The Lightning Indexer still runs, still costs its projections, and its output is
discarded — `top_k` selects everything it scored.

This is not a defect in the module. `tk = min(top_k, n_blocks)` is the correct guard, and at
the deploy length the mechanism works as designed. It is a CONFIG mismatch that changes what
an experiment measured, and it has been invisible because nothing prints the selection rate.

It is separate from, and was found alongside,
[the dead HCA branch](2026-08-25-hca-compressed-branch-dead-on-slot-path.md). That one is a
silent zero; this one is a silent no-op.

## Why it matters

Every TUL arm run to date — A0, A1, A1r, A3, the gate arms, the seed sweep, the
`onset-capture` ladder, the SCSE campaign, and the H24 arm now running — used a dense CSA.
Two consequences:

1. **Reading those arms.** Any claim of the form "CSA's sparse global selection does X on the
   slot path" is unsupported by them. Nothing in the campaign has made such a claim, which is
   why this is a note and not a postmortem, but the next person to reason about CSA on these
   runs needs to know.
2. **Transfer to deploy.** A recipe tuned on the short schedule was tuned with a mechanism
   switched off that IS on at 4096. Anything the short arms conclude about attention cost,
   memory, or the indexer's usefulness does not transfer without re-measuring.

## Proposal

1. **Log the selection rate once per run.** At build time, print `n_blocks` and
   `tk = min(top_k, n_blocks)` per section for the configured sequence length AND, when TUL
   is on, for the slot budget. One line, no forward cost. Silence is what made this last
   this long.
2. **Derive `top_k` from `n_blocks` for the short schedule**, so the arms exercise the same
   mechanism the deploy recipe does — e.g. `top_k` at the same FRACTION of `n_blocks` that
   256 of 512 gives at 4096, which is 72 of 144 at 1024 and 4 of 8 on the slot core.
3. **Assert it.** A test that a named config's CSA selection is actually selective, in the
   same file as the HCA-budget xfail (`tests/test_compressor_short_seq.py`).

Point 2 changes the model, so it goes through its own arm before it ships, and NOT inside
the H24 arm — two model changes in one comparison make the comparison unreadable.

## Alternatives considered

- **Leave it and only document it.** Rejected for point 1: a config whose meaning silently
  flips with sequence length must announce itself. Point 2 alone would be accepted.
- **Fold the `top_k` change into the H24 arm** so one run covers both. Rejected: the H24 arm
  is already carrying a declared parameter-count confound; adding a second model change would
  make any result unattributable. The campaign's own lesson list has this failure twice.
- **Drop the Lightning Indexer on the short schedule** since its output is discarded.
  Rejected: it removes a mechanism from the arms rather than restoring it, and it would make
  the short arms LESS like the deploy recipe, not more.
- **Raise when `top_k >= n_blocks`.** Rejected: the guard is legitimate at short sequences —
  generation from a prompt shorter than one block is a real case — so a raise would break
  inference to catch a config smell. A printed line does the job.

## Acceptance criteria

- A run's startup log states `n_blocks` and `tk` per section, for the token stream and, when
  TUL is on, for the slot budget.
- A test asserts a named config's CSA selection is selective, and it fails when `top_k` is
  raised above `n_blocks`.
- If point 2 ships, it ships on the result of its OWN arm, and this note moves to
  `implemented/` or `rejected/` on that result — not on the code landing.

## Risks

- Making CSA genuinely sparse on the short schedule will change every short-arm number, so
  the TUL ablation table would need a re-baseline. That cost is the reason to decide it with
  an arm rather than by declaring the current state a bug.
- The indexer's gradients are `None` by design (`_fuse_mods_nograd`); a `top_k` that actually
  selects does not change that, but anyone adding an indexer aux loss must re-read that
  comment in `attention.py` first.
