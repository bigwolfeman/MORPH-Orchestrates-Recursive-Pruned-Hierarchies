# Agent Note: periodic checkpoints had no retention and grew without bound

Status: implemented

## Problem

`checkpoints/morph/` reaches **292 GB**. `train.py` has two checkpoint writers. The
rolling pre-onset ring rotates its files. The ordinary `ckpt_every` path does not: it
writes `step_<N>.pt` forever, so a run's disk cost grows with its length with no ceiling.

The cost is not hypothetical. A four-seed 3500-step sweep at `ckpt_every=500` writes 7
files per seed at 2.26 GB each — 63 GB for one experiment. A 100k-step run at the shipped
`ckpt_every=2500` writes 40 files, 90 GB. Nothing in the code or the config says this
will happen, so the growth is invisible until the disk is full.

The second, quieter problem: the two rings are separate copies of the same "append, pop,
remove, log" logic. That is why only one of them ever learned to rotate.

## Decision

One `RetentionRing` class in `morph/training/ckpt_retention.py`, used by **both** rings.

`training.ckpt_keep_last: 8` in `base.yaml`; 0 means unbounded. 8 matches the existing
`ckpt_rolling_keep` and covers every probe ladder this project has actually read — the
drift and forcing-bias sweeps read 7 rungs per run — so the default truncates no
experiment that has been run.

Four details that are the whole value of the class:

- **Startup prints which mode it got.** A bounded run says the bound, an unbounded run
  says it is unbounded. Silent unbounded growth is the original defect.
- **The ring is seeded from disk.** Otherwise a resumed run rotates only its own writes
  and the directory grows past the bound on every restart.
- **Sorting is numeric.** Lexically `step_900` sorts after `step_1000`, so a lexical
  sort rotates away the newest checkpoint — the one a resume needs.
- **Only paths handed to the ring are removed.** `DIVERGED_*.pt` and `TAKEOVER_*.pt` are
  never added, so a failure capture can never be rotated away.

A failed `os.remove` is logged and the path leaves the ring, so one undeletable file
cannot stall rotation forever.

## Alternatives considered

- **Keep the default unbounded and rely on the operator.** Rejected: that is the current
  behaviour, and it produced 292 GB. A default that only works when someone remembers is
  not a default.
- **Default `keep_last` to 2 or 3.** Bounds harder, but silently truncates the 7-rung
  ladders the drift probes read. Losing experiment data to save disk is the wrong trade;
  8 bounds a 100k-step run to 18 GB and truncates nothing that has been run.
- **A cleanup script run between experiments.** Rejected: it only works after the disk
  has already filled, and it cannot know which checkpoints a live experiment still wants.
  Retention belongs where the file is written.
- **Rotate inside `save_checkpoint`.** Rejected: `save_checkpoint` would then need to
  know which of the two rings it is serving and which filenames are guard output. The
  ring is the thing with the policy; the writer should stay a writer.
- **Delete by age or by a disk-usage watermark.** Rejected: both make retention depend on
  wall-clock or on other runs, so the same experiment keeps a different number of
  checkpoints depending on what else is on the machine. Count is reproducible.

## Consequences

A run's checkpoint cost is bounded at `ckpt_keep_last x ~2.3 GB` regardless of length.
An experiment that wants a longer ladder raises the number and that choice is in its
config, which is where a reader looks for it.

`tests/test_ckpt_retention.py` covers the contract — which files survive on disk, not the
return shape — so removing or inverting the rotation fails the suite. The tests were
checked against 8 deliberate sabotages; the first draft caught 6 of 7, and the negative-keep
test was strengthened until it caught the seventh.

The pruning done alongside this, and the keep-rules used, are recorded in
[`lab/divergence/checkpoint-retention.md`](../../../../lab/divergence/checkpoint-retention.md).
