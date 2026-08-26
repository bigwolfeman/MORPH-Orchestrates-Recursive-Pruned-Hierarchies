# Checkpoint retention — MORPH divergence campaign

Written 2026-08-25, when `checkpoints/morph/` had grown to **292 GB** across 138 `.pt`
files. `checkpoints/` is gitignored, so this tracked file is the record of what was
deleted and why.

## The rule

A checkpoint survives only if one of these is true:

1. **It is irreplaceable.** The run cannot be re-made by re-running the same command.
2. **A live experiment reads it.** Live means a file in `lab/experiments/planned/`.
3. **It is the trained product of a long run**, i.e. real GPU time that a future run
   could start from.

Everything else goes. Being *named in a filed writeup* is NOT a reason to keep a
checkpoint — several campaigns cleaned up earlier (`tst_stp_on_50k`, `sft_*`,
`b1zero_*`, `tul-a3`) are still cited by name in `docs/` and their writeups stand on
the probe JSON, the training log and the saved Hydra config. That is the precedent.

## Kept

| dir | size | rule | why |
|---|---:|:--:|---|
| `onset-capture` | 25 G | 1, 2 | The takeover ladder, `ROLL_step_{1625..1850}` + `TAKEOVER_step_1866`, 25-step spacing. **This event cannot be re-made.** Two runs of MORPH at a fixed seed decorrelate to 10 % in eleven steps (`lab/experiments/failures/2026-08-23-tul-run-replication.md`), so re-running `tul_a1` seed 0 does not reproduce this takeover. Both live planned experiments read it, and so does every filed probe. |
| `onset-sub` | 12 K | — | Symlinks into `onset-capture`. Free. |
| ~~`h24-*` step ladders~~ | 0 | — | **Pruned 2026-08-25 18:20**, 44 G, once the arm was filed as a failure. The rule said "delete after that experiment is filed" and it was applied to itself. |
| `h24-ctrl-s0/DIVERGED_step_2880.pt`, `h24-hca16-s1/DIVERGED_step_2940.pt` | 4.5 G | 1 | The two abort captures. They are the only state that exists for either divergence, and the retention ring is built never to rotate a guard file away. |
| `tul-a1/step_20000.pt`, `tul-gate/step_20000.pt` | 4.5 G | 3 | 20k-step trained TUL models, about 2.5 GPU-hours each. |
| `tul-a1/DIVERGED_step_5900.pt` | 2.3 G | 1 | The only snapshot of the original batch-14 production-kernel divergence, which is a different regime from `onset-capture`. |

## Deleted

Every campaign below is **filed** (`lab/experiments/successes/` or `failures/`), and for
every one the probe JSON, the training log and the Hydra config live outside the
checkpoint tree. Verified before deleting, not assumed.

| dirs | size | writeup | artifacts that survive |
|---|---:|---|---|
| `seedsweep-s0..s3` | 55 G | `failures/2026-08-24-tul-forcing-bias-predicts-divergence.md` | `morph-scratch/seedsweep/drift_s{0..3}.json`, `s{0..3}.log`, `hy-s{0..3}/` |
| `scse1-s0..s3` | 57 G | `failures/2026-08-25-scse-stage1-initial-deviation.md` | `morph-scratch/scse1/drift_s{0..3}.json`, `regress_s3.json`, logs, Hydra dirs |
| `scse2-{ctrl,scse}-s{1,2,3}` | 53 G | `failures/2026-08-25-scse-full-method.md` | `lab/experiments/results/2026-08-25-scse-full-method/` (JSON + logs, tracked) |
| `stage0-tul_a0`, `stage0-tul_a1` | 31 G | `successes/2026-08-24-tul-forcing-bias-arm-control.md` | `morph-scratch/drift_a0.json`, `drift_a1.json`, `morph-scratch/stage0/` |
| `tul-a1r` | 4.4 G | `docs/tul-divergence-rca.md` | Two `DIVERGED_*.pt` only, no clean rungs. The abort steps (2080, 4160) are in the RCA text. |
| `s0-slotembed` | 4.4 G | H5 lever, filed | training log |
| `h24-ctrl-s0-3500` | 2.2 G | `failures/2026-08-25-h24-hca-branch-arm-1seed.md` | Abandoned wrong-regime run, rejected before scoring. |
| `scse-smoke-ctrl`, `scse-smoke-scse` | 4.4 G | — | 60-step smoke tests. |
| `tul-a1`, `tul-gate` intermediates | 13.6 G | — | `step_{5000,10000,15000}` from both. The 20k product is kept. |
| 68 empty run dirs | 0 | — | Leftover `hydra.run.dir` shells with no `.pt`. |

Also removed: `lab/scse-lean/.lake` (3.7 G), the Lean/Mathlib build cache for the filed
SCSE proof attempt. The source (`Scse.lean`, `lakefile.toml`, `lean-toolchain`,
`lake-manifest.json`) is kept and tracked, so `lake exe cache get` rebuilds it.

## Standing habit

`training.ckpt_every` was left at 500 for several 3500-step sweeps, which is 7
checkpoints per seed at 2.26 GB each — 63 GB for one four-seed sweep. Set `ckpt_every`
from what the experiment's probe actually reads. A binary divergence arm reads the
`[ABORT]` line in the log and needs **no** checkpoints at all.

## The code fix (the reason this cannot happen again)

Deleting 220 GB is half the job. The defect was that the periodic save path had **no
retention at all**, so its cost grew with run length forever.

`morph/training/ckpt_retention.py` — `RetentionRing`, one class, now used by **both**
checkpoint rings in `train.py`. Before this they were separate code: the rolling ring
rotated, the `ckpt_every` ring did not. Two copies of "append, pop, remove, log" is how
they drifted apart in the first place.

- `training.ckpt_keep_last: 8` in `base.yaml`. 0 means unbounded, and the run prints
  which of the two it got at startup, so an unbounded run is never silent.
- The ring is **seeded from disk** at startup. Without that, a resumed run rotates only
  its own writes and the directory grows past the bound on every restart.
- Sorting is numeric. A lexical sort puts `step_900` after `step_1000` and would rotate
  away the newest checkpoint, which is the one a resume needs.
- Only paths handed to the ring are ever removed, so `DIVERGED_*.pt` and
  `TAKEOVER_*.pt` cannot be rotated away. Losing the state that captured a failure is
  the one unacceptable outcome.
- A failed delete is logged and the path leaves the ring, so one undeletable file cannot
  stall rotation forever.

`tests/test_ckpt_retention.py` — 12 tests, all passing. Checked against 8 deliberate
sabotages of the module (rotation removed, newest popped instead of oldest, lexical
sort, negative-keep clamped by magnitude, `seed()` deleting on adopt, the `except`
swallowing silently, `base.yaml` set to 0, the log tag dropped). The first draft caught
6 of 7: the
negative-keep test only wrote one file, so a ring that quietly became `keep=4` still
passed. That test now writes six.

**Effect on the H24 binary arm running during this change: none.** It writes 6
checkpoints (6000 steps at `ckpt_every=1000`), which is under the bound of 8, so no
rotation occurs. Verified by arithmetic on `h24_arm.sh`, not assumed.

## Wider survey, 2026-08-25 (reported, NOT acted on)

Large `.pt`/`.safetensors` outside this project, for the record:

| tree | size | what it actually is |
|---|---:|---|
| `projects/TitanBoard` | 301 G | **Mostly not checkpoints.** ~240 G is `data/processed/` — preprocessed dataset shards that use the `.pt` extension. Real checkpoints are ~40 G over 40 dirs, 2-3 files each, already modest. Deleting on the filename pattern here would destroy a dataset. |
| `projects/jepa-data` | 91 G | 72 files |
| `projects/00NN` | 83 G | 126 files |
| `projects/llama-megacontext` | 37 G | 5 files |
| `projects/student-forcing` | 27 G | 18 files |

None of these were touched. They are other projects, and pruning them needs their own
results-cited keep-set the way the DB testbed got one.

## Incident during this cleanup: "no checkpoints yet" is not "dead"

While removing the 74 leftover run dirs that held only a `wandb_id.txt`, the rule used
was "no `.pt` file inside". That rule matched **`h24-hca16-s1`, the run that was
executing at that moment** — it was at step 800 with `ckpt_every=1000`, so it had not
written its first checkpoint yet. The directory was deleted.

Consequences and what was done:

- The directory itself. `train.py` calls `os.makedirs(ckpt_dir, exist_ok=True)` once at
  startup, so the missing directory would have raised `FileNotFoundError` inside
  `torch.save` at step 1000 and killed the run. Recreated immediately.
- The `wandb_id.txt` sidecar. Read only at startup for resume, so the live run was
  unaffected, but a later resume would have started a new wandb run. Recovered from the
  run's own log (`runs/w9rgy73e`) and rewritten byte-exact: 8 bytes, no trailing
  newline, verified with `od -c` against a sibling.

The lesson is the rule, not the recovery. **Emptiness is not a liveness test.** A dir
with no `.pt` is either a dead run or a young one, and the two are indistinguishable
from the filesystem alone. Check the process list first, and exclude the run tags a
live runner script is going to use — not just the ones it has already used.
