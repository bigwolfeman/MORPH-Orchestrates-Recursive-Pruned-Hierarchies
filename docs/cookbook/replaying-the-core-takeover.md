# Replaying the TUL core takeover from a checkpoint

The takeover used to be something you waited for: run ~1000–4000 steps, hope it fires, and
autopsy whatever the abort left behind. Every checkpoint in the repo from before
2026-08-23 — `DIVERGED_step_2040`, `2080`, `4160`, `5900`, `6200`, `TAKEOVER_step_1093` —
was written AT the abort, so all of them are post-mortems and none can seed a replay.

It is now reproducible on demand. This is how.

## Why it works

Two measured facts, both verified rather than assumed:

1. **The run is bit-reproducible** in the deterministic configuration. Two 300-step runs
   agree on all 300 steps across all 85 probe series
   ([evidence](../experiments/results/2026-08-23-morph-bit-reproducible.md)).
2. **A resume continues the same trajectory, bit for bit.** Measured: run 0→120 straight
   through, then resume from a step-80 checkpoint and run to 120. Steps 81–119 agree on
   **0 of 39 differing**, across **87** probe series.

So a state saved shortly before the onset replays the onset every time it is loaded.

## Capturing a pre-onset state

The onset is narrow — in the reproducible control the core's share of the pre-clip gradient
was 0.018 at step 1000, 0.301 at 1050 and 0.903 at 1093, with the last sub-0.05 step at
1053. About 40 steps wide. So capture a rolling window and let the abort stop the run
inside it:

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m morph.training.train --config-name tul_a1 \
    training.deterministic=true model.use_kernels=false training.batch_size=6 \
    training.seed=0 training.ademamix_alpha_cap=3.5 \
    training.steps=5000 training.eval_every=999999 training.gen_every=0 \
    training.grad_probe_every=1 \
    training.ckpt_rolling_every=25 training.ckpt_rolling_keep=10 \
    training.abort_core_share=0.5
```

`ckpt_rolling_*` keeps the last 10 checkpoints at 25-step spacing — 250 steps of history at
~2.4 GB each — and deletes the rest. `abort_core_share` stops the run once the core holds
the gradient. What is left on disk is `ROLL_step_*.pt` bracketing the onset plus the
`TAKEOVER_step_*.pt` the guard wrote. Guard checkpoints are never rotated away.

Pick the newest `ROLL_` whose logged core share is still healthy — read
`training.grad_probe_path`'s JSONL, take `preclip/core / preclip/total`.

## Replaying it

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m morph.training.train --config-name tul_a1 \
    training.deterministic=true model.use_kernels=false training.batch_size=6 \
    training.seed=0 training.ademamix_alpha_cap=3.5 \
    training.resume=checkpoints/morph/<run>/ROLL_step_<N>.pt \
    training.steps=<N+300> training.grad_probe_every=1 \
    training.eval_every=999999 training.gen_every=0
```

The resume restores model, optimizer, scaler and CPU+CUDA RNG, and fast-forwards the
unshuffled data stream by the batch count, so the continuation is the original run's
trajectory. A 300-step replay reproduces an onset that originally cost 1100 steps.

## Using it to test an intervention

This is the point. Resume the SAME checkpoint twice, changing exactly one thing:

```
# baseline replay
... training.resume=ROLL_step_<N>.pt
# with an intervention
... training.resume=ROLL_step_<N>.pt model.core_gain_clip=1.5 model.core_gain_clip_iter_lo=0 model.core_gain_clip_iter_hi=0
```

Both start from an identical state and see an identical data stream, so any difference is
the intervention. That is a controlled experiment at n=1, which is what an irreproducible
run could never give — see
[the failed replication gate](../experiments/failures/2026-08-23-tul-run-replication.md)
for what n=1 was worth before.

## Verified end to end (2026-08-23)

Not a procedure on paper — this was run.

**Capture.** `onset-capture`, the reproducible control, aborted at step **1866** with the
ring buffer intact. What it left on disk, read from the probe JSONL:

| checkpoint | core share | block gain | r² | state |
|---|---:|---:|---:|---|
| `ROLL_step_1700` | 0.0124 | 1.057 | 0.306 | healthy |
| `ROLL_step_1750` | 0.0205 | 1.066 | 0.182 | healthy |
| `ROLL_step_1775` | 0.0541 | 1.106 | **0.869** | share still looks healthy, r² has already moved |
| `ROLL_step_1800` | 0.3723 | 1.329 | 0.836 | onset |
| `ROLL_step_1825` | 0.1184 | 1.240 | 0.837 | falls back |
| `ROLL_step_1850` | 0.8903 | 1.434 | 0.924 | taken over |
| `TAKEOVER_step_1866` | — | — | — | written by the guard |

Last step with core share below 0.05: **1822**. Onset width: **44 steps**, matching the
~40 measured on the earlier control.

**Replay.** Resumed from `ROLL_step_1750` and ran to the abort:

- data fast-forward of 1751 batches took **10.2 s**
- steps 1751–1861 versus the original: **0 of 111 differ**, across **87** probe series
- the takeover recreated itself exactly — share 0.0541 → 0.3723 → 0.1184 → 0.8903 at steps
  1775 / 1800 / 1825 / 1850, matching the original to every logged digit

A takeover that cost 1866 steps to reach now reproduces in about 110.

One expected difference: the replay's guard fired at step 1861 against the original's 1866.
The guard's sliding window starts empty on a resume and needs `window` steps to fill, so
that 5-step gap is guard state, not model state — the trajectory itself is bit-identical.

**Recommended seed:** `ROLL_step_1750`. It is 72 steps clear of the last healthy step and
116 clear of the abort, so an intervention has room to act before onset, and the replay is
short.

**Note for the r² readers:** at `ROLL_step_1775` the core share is 0.0541 — which reads as
healthy — while the block-gain fit's r² has already jumped to 0.869. The geometric
structure appears before the share does, which is the same lead the
[block-gain result](../experiments/results/2026-08-23-tul-block-backward-gain.md) measures
on other trajectories.

## Cautions

- **The reproducible configuration is not the production one.** `use_kernels=false` runs
  eager attention at 2.28× lower throughput and needs about half the batch, which changes
  the gradient-noise scale. Conclusions transfer as mechanism, not as step numbers.
- **Bit-reproducibility is per build.** Two different torch builds gave different
  trajectories from the same seed (29 of 30 steps differed). A checkpoint replays on the
  environment that produced it; record which venv that was.
- Rolling checkpoints cost ~2.4 GB each. `keep=10` is 24 GB.
- Verified over a 39-step overlap, not a 1000-step one. Longer replays are expected to hold
  and are not yet measured.
