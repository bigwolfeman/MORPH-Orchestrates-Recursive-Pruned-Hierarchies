# Agent Note: plain-then-TUL bootstrap, and the loader that allows it

Status: proposed

## Problem

Every slot-loop arm in the 2026-09 arc trains the loop, the reader and the language model
together from step 0. Thirteen of them read token K1−K6 inside [−0.0001, +0.0033]. One
explanation has never been tested: at step 0 the slot state `z` is noise while the coda is
still learning to read tokens, so the cheapest policy the coda can learn is to ignore `z` —
a habit fixed long before the loop has anything to say. If the flat K-curve partly measures
a cold reader, no amount of work on the loop's mechanism will move it.

Wolfe asked for the direct test: "boot strap the model by training no tul for like 5k steps
then use TUL for 5k steps to see what happens to loop contribution."

The trainer could not do it. A TUL model started from a plain checkpoint has homeless
tensors in BOTH directions, and the two loaders disagreed about them:

* `load_checkpoint` (`training.resume`) RAISES on any unexpected tensor — correct, and it
  refuses this load, because a `tg_restrict` model does not build the plain seed's pooled
  compressor or top-k indexer.
* `load_checkpoint` only WARNS on missing tensors. So a plain → TUL resume would have
  "worked" while silently leaving any number of non-TUL tensors at their random init.
* `load_weights_only` (`training.init_from`) reports missing and unexpected and the trainer
  ignores both; its only guard is "at least 50 % of tensors matched". A load that silently
  discarded a third of the seed's attention would pass it.

None of those is a contract. The first refuses the arm, the other two would run it without
saying what was lost.

## Proposal

### The schedule: `init_from`, not `resume`

`create_lr_schedule` (`morph/training/optimizer.py`) is a pure function of the ABSOLUTE
step. Under `training.resume` at step 5,000 with `steps: 10000`, `warmup: 1000` is four
thousand steps in the past, so the TUL phase opens at full flat LR with no ramp. That is the
configuration `lab/divergence/DIVERGENCE-README.md` §A names as detonating, and the
1,000-step ramp is the measured cure. Nothing in the tree re-runs a ramp from a resumed step.

`training.init_from` already means "seed a brand-new schedule from trained weights": model
tensors only, step axis back to 0, optimizer fresh. That gives the arm its own 1,000-step
ramp and its own 0…5,000 axis, at the cost of the plain phase's optimizer state, which is
discarded. Recorded rather than hidden: the AdEMAMix slow EMA restarts.

### `training.data_skip_batches`

`init_from` starts the deterministic token stream at its head, so the TUL phase would
re-read the exact batches the seed already trained on. The new key fast-forwards the stream
the way a faithful resume does, and refuses (rather than guesses) when a resume already
asked for a position or when the curriculum loader owns it.

### `training.resume_plain_to_tul`

One opt-in flag, honoured by BOTH loaders, that allows exactly two classes of homeless
tensor and raises on everything else:

* MISSING — keys under a TUL-owned root child (`tul`, `tul_*`). They keep their fresh init.
* UNEXPECTED — keys under a pooled branch a `tg_restrict` attention did not build
  (`compressor`, `comp_norm`, `indexer`). They are dropped.

Both lists are enumerated from the LIVE model (`tul_owned_prefixes` walks `named_children`;
`tg_dropped_attention_prefixes` walks `named_modules` for a module that declares
`tg_restrict` True and holds `None` in one of those slots), never from a hardcoded name
list, so a module added to the tree tomorrow is classified by what it is. Both lists print
in full. With the flag off, both loaders behave exactly as before — including
`load_checkpoint`'s warn-on-missing, which the flag TIGHTENS into a raise for any non-TUL
key.

Measured end to end on CPU at the real 292.5 M shape against the real checkpoint
(2026-09-13): 20 missing (all TUL-owned), 105 unexpected (all pooled-branch, 3,565,184
parameters), 0 refused in either direction, and `load_weights_only` returned.

### What the arm gives up, said plainly

About 1.2 % of the seed's parameters are thrown away and the compressed branch of every
block restarts under otherwise-trained weights. This is not avoidable: `tg_restrict` is only
reachable with a TUL config, so no plain checkpoint anywhere can carry that attention. It is
a second difference from `slot-spandec-strict` and it belongs in every row that reports the
arm.

## Alternatives considered

* **Train TUL from step 0 — every arm in the arc so far.** This is the thing the bootstrap
  is testing against, not an alternative to it. Kept as the one-factor partner.
* **`tul.activate_at` at a fraction inside ONE run — the `base.yaml` recipe.** TUL switches
  on part-way through a single schedule. It restarts nothing: no ramp, no optimizer, and the
  TUL parameters sit in the optimizer from step 0 with `grad None`. It answers a different
  question ("what happens when TUL joins a run in progress") and it cannot give the loop its
  own 1,000-step ramp, which the divergence work says is load-bearing. Rejected for this
  arm; it remains the shipped production recipe.
* **`training.resume` with `steps: 10000`.** Keeps the optimizer state and the data
  position, which is attractive. Rejected because the LR schedule is absolute-step and the
  ramp cannot be re-run — see above. Rejecting it is what forced `data_skip_batches` to
  exist.
* **Freeze the plain weights and train only the TUL parameters.** Isolates "can a mature
  reader learn to read a z" from "does the loop change the model". Rejected as the FIRST
  arm: a frozen coda cannot learn to read `z` at all, so a flat K-curve under it would be
  uninformative. It is the natural follow-up if the bootstrap moves the curve.
* **Allow the dropped pooled-branch tensors by widening `RETIRED_TUL_KEYS`.** That tuple is
  a fixed list of retired names, checked against `model.tul.W_prefix` being absent. The
  pooled-branch names are per-block and depend on which modules a given config built, so a
  static tuple would either be wrong for some config or would have to be regenerated by
  hand. Rejected in favour of enumerating from the live model.
* **Make missing keys raise by default.** Tempting — the current warn is exactly the silent
  partial load the loader's own comments complain about — and rejected here because it would
  change behaviour for every existing resume in the tree, including the `_prune_mask`
  back-compat case, while another agent is editing the model files. The flag tightens the
  path this arm uses; the default is a separate decision with its own note.

## Acceptance criteria

* `tests/test_resume_allow_missing_tul.py` — 11 tests, two-sided: the allowed classes load
  byte-exact and the same probe refuses a sabotaged non-TUL missing key, a fabricated
  unexpected key, a plain model (which may allow nothing) and a TUL model without
  `tg_restrict` (which may drop nothing).
* With the flag off, `load_weights_only` produces a byte-identical state dict to with it on,
  and `load_checkpoint` still raises on the same seed.
* The real-scale load reports 0 refused in both directions.

## Risks

* The 105 dropped tensors have never been dropped on a RUNNING model. No forward, no
  backward, no step has been taken from those weights.
* `data_skip_batches` has never fast-forwarded a real stream. Its ~10-minute cost is the
  resume path's measured figure.
* The runner's 12-step smoke passes no overrides, so it will also load the 2.2 GB seed and
  fast-forward 5,000 batches.
* A bootstrap arm conflates two changes — mature backbone, and a re-initialised compressed
  branch. The control that separates them (seed a NON-`tg_restrict` slot arm from the same
  checkpoint) is not queued.

Record: [`lab/experiments/planned/2026-09-13-arc-plain-then-tul-bootstrap.md`](../../../../lab/experiments/planned/2026-09-13-arc-plain-then-tul-bootstrap.md).
Config: `morph/configs/tul_slot_strict_bootstrap.yaml`.
