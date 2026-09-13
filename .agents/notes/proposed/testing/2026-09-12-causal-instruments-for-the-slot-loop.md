# Agent Note: causal instruments for the slot loop

Status: proposed

## Problem

The slot loop has been read flat for a month. Across the 2026-09-10 and 2026-09-11 arms
every K-curve sits at or under 0.0023 nats on the token CE, and the standing conclusion
is "no core, stability, seed, width, geometry or target lever moves the loop". An outside
review of that evidence found that three of the readings cannot in principle say what they
are being used to say, and that the gap is in the instruments and not in the arms.

1. **The worth of `z` is only ever measured with hindsight.**
   `lab/divergence/slot_z_optimize.py` fits `z` by gradient descent on the very tokens the
   coda is about to predict and scores it on those same tokens. Its docstring is honest
   about it — an ORACLE upper bound on the coda's capacity — but "a fitted `z` is worth
   0.9-2.6 nats to the coda while the loop's own `z` is worth 0.002" is now quoted as
   evidence that the reader has capacity the loop is failing to fill. It is not: a causal
   producer cannot see the continuation, and nobody has measured what one could deliver.

2. **Depth is only ever forced globally.**
   `lab/divergence/core_depth_sweep.py` sets one depth for every slot in the row. At depth
   1 the current thought and the fifty ARCHIVED slots the coda reads for history are
   shallow at the same time. A flat global K-curve is consistent with "the loop is worth
   nothing" AND with "the loop refines the current thought and the archive's collapse
   hides it". No reading in the tree separates them.

3. **The per-pass planning target grades two things at once.**
   `tul.spandec_per_pass` (397bdf3) supervises pass `t` against spans `s+1 .. s+min(t, cap)`
   and logs `tul/spandec_pass_t{t}`. That series changes the DEPTH and the HORIZON
   together, so a falling curve is equally consistent with "a deeper pass plans further"
   and with "six spans are an easier average than one".

## Proposal

Three scripts under `lab/divergence/`, each independent of the other two, each with
`--help`, JSON output and bootstrap CIs, following `slot_z_optimize.py`'s conventions
(`--ckpt NAME=CONFIG=PATH`, `--rows`, `--batch`, `--depth`, `--out`).

**1. `slot_z_causal_fit.py` — the causal fitted-z diagnostic.**
For each slot: take the history up to and including the span's boundary token, sample K
continuations from a frozen full-context TEACHER (a plain MORPH checkpoint) at temperature
1.0, fit `z` against the coda CE of `--fit-samples` of them placed in the next span's row
positions (inputs and labels), and score that `z` on the held-out sample(s) and on the
untouched real continuation. Six columns on one position set — entry, loop, hindsight,
`hindsight_full` (`slot_z_optimize.optimise_z` verbatim), causal-on-real,
causal-on-held-out — plus the teacher's own CE on the same positions, without which the
causal columns have no scale. `ce_loop - ce_causal_real` is what a better causal
inferencer could still win at this frozen coda and this frozen write;
`ce_causal_real - ce_hindsight` is the hindsight that can never be won.

The fit and the scoring split the forward at the `_tul_core` return through
`slot_z_optimize.ZSplit`, so only the coda is re-run per optimiser step and the prefix
write, the TG masks and the weighted CE are the model's own code.

**2. `slot_depth_isolation.py` — per-slot depth isolation.**
A new EVAL-ONLY forward argument `slot_depths: Tensor[B, max_slots] | None` (the
`slot_layout` pattern: a per-forward DATA argument; `None` is bit-identical to before it
existed) lets one slot run at a different depth from the rest. Three paired arms per
depth, each scored on ONLY that slot's own next span: ISOLATE (slot `s` shallow, archive
intact), COMPLEMENT (slot `s` intact, archive shallow) and UNIFORM (the global column, on
the same positions). If ISOLATE moves and UNIFORM does not, the global K-curve is hiding a
current-thought effect behind the archive.

**3. `spandec_horizon_grid.py` — the depth x horizon grid.**
Hold the span-decoder target fixed and move the forced depth: column `pass_h1` (H=1 at
`spandec_pass_tokens`, `pos_pass`), column `pass_h6` (H=6, same table) and column `exit`
(the shipped H=1 at `spandec_max_tokens` through `pos` — the `tul/spandec_ce` a run logs),
each at depths 1, 2, 3, 6, with K1-K6 and K3-K6 per column. A K-curve on `pass_h6` and not
on `pass_h1` is depth buying HORIZON. It runs on a NON-per-pass checkpoint for the `exit`
column alone, so the strict ruler is a usable control.

`lab/divergence/_next_span.py` is the one home for "which row positions is a slot
answerable for, and at what offset", shared by instruments 1 and 2 and checked against
`morph.model.tul_spandec.next_span_slots` on every call.

## Alternatives considered

**Leave `slot_z_optimize` as the worth instrument and caveat it harder.** Rejected: the
caveat is already in the file's docstring and in the vault memory, and the number is still
being read as "the reader has capacity". A caveat that is ignored twice is not an
instrument; the causal twin is.

**Use a held-out human continuation instead of a teacher's sample.** There is only one
real continuation per span, so "fit on one future and score on another" is impossible
without a generative model of futures. The teacher is the only way to get more than one
draw, and its cost is named: the samples are a MODEL's futures, so a weak teacher makes
the causal fit look easy. Reported beside the teacher's own CE for that reason.

**Fit `z` with the real next span in the decoder context and only the LABELS swapped for
samples.** Cheaper (no re-recording of the front and the core) and wrong: the coda would
still read the real continuation as teacher-forced context, which is exactly the hindsight
the instrument exists to remove. The test asserts the real tokens reach neither the input
ids nor the label tensor.

**Replace one slot's span per counterfactual row (zero contamination).** Correct and
unaffordable: the front and the core would have to be re-recorded per (slot, sample), so
each optimiser step would cost `n_slots` coda replays instead of `--fit-samples`.
`--fit-groups G` is the compromise — G interleaved groups, the nearest counterfactual span
G spans back, G times the cost — with G=1 the default and the contamination named in the
docstring and in the JSON notes.

**Use the KV-cache decode engine for the teacher.** `morph/inference/kv_cache.py`'s
`decode_step` requires every batch element at the same absolute position, so a batch of
prefixes of different lengths cannot be prefilled. The design that would win — walk the
row once with a batch of K and clone the cache at every boundary — needs a cache-clone
helper the module does not have and a parity gate of its own. Named and not built;
`generate_plain_batch` is eager, ragged and already gated against single-row greedy.

**A `tul.slot_depths` config knob instead of a forward argument.** Rejected on the repo's
own law: no runtime feature flags in the forward, and a config knob cannot express "slot 7
at depth 1, the rest at 6" per batch anyway. A per-forward data argument is the
`slot_layout` / `bag_size` precedent.

**Read the per-pass target's own `tul/spandec_pass_t{t}` series harder.** That series is
the confound. Nothing done to it separates depth from horizon; a grid does.

## Acceptance criteria

* `slot_z_causal_fit.py` reports `ce_causal_real` between `ce_loop` and `ce_hindsight` on
  a trained slot-loop arm, with a CI over rows that excludes 0 against at least one of
  them, and with the teacher's CE printed beside it. If `ce_causal_real` is within noise
  of `ce_loop`, the "the reader has capacity" reading is dead and the note is updated to
  say so.
* `slot_depth_isolation.py` runs its identity check at 0 (a table filled with the model's
  own eval depth reproduces `slot_depths=None` bit for bit) and reports ISOLATE,
  COMPLEMENT and UNIFORM with CIs over rows on at least 48 rows.
* `spandec_horizon_grid.py` reproduces the model's own `spandec_ce` in its `exit` column
  at the model's own depth to 1e-5, and reports K1-K6 / K3-K6 for every column it runs.
* `pytest tests/ -q` stays green and `python scripts/verify_template.py` gains no line.
* Every instrument's docstring names what it CANNOT say, and each is paired with a
  depth-1-TRAINED control before any result is read as "the loop earns" (the 2026-09-12
  reading: training depth moves CE, eval depth does not).

## Risks

**The teacher decides the answer.** `slot_z_causal_fit` is bounded by how good the
teacher's futures are. A teacher weaker than the student makes the causal fit look easy; a
stronger one makes it look hard. Mitigation: the teacher's own CE on the same positions is
a reported column, and a second teacher is the obvious control if the first result is
close to the decision boundary. There is no mitigation that makes the number
teacher-free.

**Counterfactual history in the fit rows.** At `--fit-groups 1` a slot's `z` is fitted
against a history whose earlier spans are teacher samples. `--fit-groups 4` bounds it at
4x the cost; the number to check is whether the causal column moves between G=1 and G=4.

**Cost.** The teacher's generation is the dominant term: about 6,500 single-row forwards
per scored row at seq 1024 with K=4. `slot_depth_isolation` is `O(n_slots)` forwards per
row per depth, roughly 40x a `core_depth_sweep` run. Both are written to run at small
`--rows` first and let the CI say whether more are needed. The 5090 is committed to a
training queue, so none of the three has been run.

**An eval-time depth intervention is not a training result.** All three read a trained
checkpoint and change the depth or the state at eval. A flat column says this checkpoint
does not use depth, not that a model trained differently would not — the trap the
2026-09-12 reading names. Every docstring says so; the risk is that a future reader quotes
the number without the control, which is exactly how `slot_z_optimize`'s oracle came to be
quoted as a capacity result.

**Unrun code.** The three scripts have only been exercised by CPU fixtures
(`tests/test_slot_z_causal_fit.py`, `tests/test_slot_depth_isolation.py`,
`tests/test_spandec_horizon_grid.py`) and by `--help`. The teacher's sampling path, the
loaders, the checkpoint loads and every GPU-only code path are unverified, and there is no
`spandec_per_pass` checkpoint on disk yet for the grid's `pass_h*` columns.

## Outcome: the causal fitted-z reading (2026-09-13, DGX Spark, chain B)

`slot_z_causal_fit.py` on `slot-spandec-strict@5000`, teacher `plain-panel-norm-match@5000`,
6 rows, depth 6, K 4 teacher samples (fit on 3, one held out), 200 steps, 1 fit group
(`lab/experiments/results/2026-09-12-instruments/causal_fit_slot-spandec-strict_g1.{json,txt}`,
23,717 s on the GB10). CE on the real next span's tokens, per position:

| z | CE | vs the loop's z | 95 % CI |
| --- | --- | --- | --- |
| entry | 3.8601 | +0.0185 | [+0.0140, +0.0216] |
| loop (the shipped exit) | 3.8416 | 0 | |
| hindsight (fit on the real span) | 2.3624 | **−1.479** | [−1.559, −1.397] |
| hindsight, full | 2.3632 | −1.478 | |
| **causal** (fit on the teacher's samples, scored on the real span) | 4.0933 | **+0.252** | [+0.227, +0.273] |
| causal, scored on the held-out teacher sample | 4.6879 | | 6,081 positions |
| teacher's own CE on the same positions | 3.6928 | | |

The gradient-fitted z that saw the answer is worth 1.48 nats; the one built from history
alone is 0.25 nats WORSE than the loop's state, and it does not even transfer to a fourth
sample of the same teacher (4.69). So the 0.9–2.6-nat "headroom" measured by
`slot_z_optimize` on 2026-09-10 is hindsight in full, and this estimator finds NO causal
headroom above the one-pass state. It does not bound the true causal optimum: a better
causal estimator (more samples, a better teacher, a learned amortised fit) could still find
some. Six rows; the interval is over positions within them.
