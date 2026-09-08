# Agent Note: the Olympiad eval holdout is contaminated and the high bands are mostly duplicates

Status: proposed

## Problem

Measured 2026-09-08 after the E16 mask arm's held-out `val/loss` reached 0.48 nats at step
5000 with a cliff at every stage switch
([`lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md`](../../../../lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md)).
A blake2b hash of every doc's token sequence (trailing EOS stripped) across the five band
views in `$OLYMPIAD_REPO/data/olympiad_bands/` and the 3,230-doc held-out shard built by
`scripts/olympiad_holdout_shard.py` from the two `eval_holdout.jsonl` files:

| Band view | Training docs | Unique | Most copies of one doc | Docs seen more than 10 times |
| --- | ---: | ---: | ---: | ---: |
| b2_3 | 664,000 | 95.9 % | 54 | 2,177 |
| b4_5 | 568,000 | 94.3 % | 160 | 6,841 |
| b6_7 | 712,000 | 46.6 % | 3,442 | 286,181 |
| b8_10 | 232,000 | 29.2 % | 241 | 122,771 |
| b11_13 | 2,040,000 | 23.2 % | 2,678 | 1,342,701 |

1,324 of the 3,230 held-out docs (41 %) appear verbatim in a training band; 88 more are
duplicates within the holdout. Bands 7 to 13 keep 25, 26, 14, 17, 34, 24 and 35 clean docs.
The generator's template space is small at the high bands, so the "held-out" set at those
bands is the training set. Two consequences for E16 and every future Olympiad run:

- `val/loss` under the curriculum reads memorization: it fell 3.51 → 0.48 on the mask arm
  and dropped 0.7 to 1.2 nats at each stage switch as the new band's holdout docs entered
  training.
- `lab/divergence/olympiad_sweep.py` scores the same jsonl docs, so its answer CE and
  accuracy at bands 6 and up are memorization too (mask arm at 6000: bands 8 to 10 at
  0.995, bands 11 to 13 at 0.980).

## Proposal

1. **Readout now.** `$OLYMPIAD_REPO/data/olympiad_bands/holdout_clean/*.eval_holdout.jsonl`:
   the holdout docs absent from every band view and unique within the holdout (1,813 +
   93 = 1,906 docs, 85 % bands 2 to 6). E16's checkpoints are re-swept on it
   (`/home/wolfe/morph-scratch/arc/run_e16_clean.sh`, results in
   `lab/experiments/results/2026-09-08-arc-e16/sweeps_clean/`). Built, run.
2. **Training data.** Dedup the band views before the next Olympiad run: a view becomes a
   doc-index list rather than a contiguous range (`scripts/olympiad_band_views.py` writes
   `doc_offsets`/`doc_lens` arrays, which the loader reads per doc, so a non-contiguous view
   needs only the `np.diff(off) == lens` assertion relaxed). Bands 8 to 13 shrink to a
   quarter of their size; the curriculum's token targets need a re-plan. Not built.
3. **Held-out shard.** Rebuild `olympiad_bands/holdout` from the clean jsonl so
   `curriculum.val_source` reads generalization. Not built.
4. **Upstream.** The Olympiad-AI generator should dedup at write time and hold out by
   template, not by draw; a held-out set drawn from the same small template pool cannot
   be clean. Not built; that repo's call.

## Alternatives considered

- **Near-duplicate filtering** (same template, different numbers) on top of exact-match.
  Not measured; exact match alone already removes 41 %, and the answer-token accuracy on
  the clean set is the number that tells whether templates leak. Revisit if the clean
  bands 2 to 6 accuracy sits above 0.95 at 1500 steps.
- **Stop E16 and relaunch on deduped data.** Rejected for E16: the checkpoints survive and
  a re-sweep costs a minute each, so the panel's survival, hinge and depth-dependence
  readings stand and only the generalization readout moves. The next run should not
  inherit the choice.
- **Keep the contaminated `val/loss` as a monitor.** It stays in the logs; it is no longer
  cited as generalization anywhere.

## Acceptance criteria

- The clean re-sweep of all 16 E16 checkpoints lands and the prereg's Results cite it.
- A deduped band-view set exists with per-band unique counts in its `meta.json`, and the
  held-out shard is rebuilt from clean docs, before any further Olympiad run is launched.
- `docs/olympiad-interop.md` records the duplication table.

## Risks

- The clean set is thin at bands 7 to 13 (14 to 35 docs each), so per-band CIs there are
  wide; do not read a high-band verdict from it.
- Exact-match dedup leaves template-level leakage in place; the per-band accuracy at 1500
  on bands never seen in training is the tell (clean bands 8 to 13 on the mask arm at 1500
  should sit far below the bands 2 to 5 that were trained).
