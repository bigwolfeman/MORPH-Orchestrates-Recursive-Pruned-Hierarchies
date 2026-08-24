# Result: MORPH can be made bit-reproducible, and what it costs

Status: measured 2026-08-23. Verified end to end — two full 300-step training runs are
bit-identical.

This is diagnostic work, not a pre-registered ablation: it started as "which op breaks
determinism" after the [replication gate failed](../failures/2026-08-23-tul-run-replication.md),
and no hypothesis was registered in advance. It is written up as a measurement, and the
one claim that gets a formal test is the end-to-end one, which is stated below with the
command that produced it.

## The problem it solves

The Phase 0.4 gate failed: two runs at one seed with identical code and data parted on the
**first backward pass** and reached a 10 % difference in the pre-clip gradient norm by step
11. Worse, run A finished healthy while run B took over (final pre-clip core share 0.8131
against 0.0152). Byte-identical runs, opposite outcomes. Nothing downstream that compares
two runs is readable in that state.

## Where the nondeterminism is

`ignore/perf/phase1/attn_determinism.py`. Same model, same input, backward six times,
counting how many of the 150 parameters differ from the first pass.

| configuration | backward non-identical | parameters affected | worst relative |
|---|---:|---:|---:|
| fused kernels ON | 6/6 | 119 / 150 | 2.1e-4 |
| fused kernels OFF (eager) | 1/6 | 36 / 150 | 3.6e-7 |
| fused kernels ON + `use_deterministic_algorithms` | 6/6 | 119 / 150 | 2.4e-4 |
| **fused kernels OFF + `use_deterministic_algorithms`** | **0/6** | **0 / 150** | — |

The worst-affected tensors with kernels on are the CCA attention convolutions —
`cca.conv_k_gp`, `cca.conv_q_dw`, `cca.conv_q_gp`, `cca.temp` — in the prelude and the
core. The fused CSA/HCA attention backward is the dominant source and it is **outside
PyTorch's determinism machinery**: turning `use_deterministic_algorithms` on does not
touch it, because Triton kernels are not PyTorch ops.

### A probe bug worth recording, because it produced a wrong answer first

The first version of this probe stopped at the first differing parameter, so the tensor it
named as "worst" was really just the first one registered — the embedding table. That is
how an early reading blamed the embedding backward. Checking every parameter instead moved
the answer to the attention convolutions. **A probe that reports an ordering must not
short-circuit.**

Second, `CUBLAS_WORKSPACE_CONFIG` must be exported **before the process starts**. Setting
it in-process, after CUDA has initialised, is too late, and with `warn_only=True` PyTorch
then produces silently wrong numbers rather than an error — the first deterministic-mode
run reported forward losses differing 6/6 and relative errors of 1.9e+1, which is not a
result, it is a broken measurement. It is gated behind an explicit check now.

## The end-to-end verification

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m morph.training.train --config-name tul_a1 \
    training.steps=300 training.seed=0 training.deterministic=true \
    model.use_kernels=false training.batch_size=6 training.grad_probe_every=1 \
    training.eval_every=999999 training.gen_every=0
```

Run twice (`detrepro-a`, `detrepro-b`, wandb `morph-tul`):

- **0 of 300 probed steps differ, across all 85 probe series.**
- Every console loss matches exactly: 11.0763, 8.7087, 7.5938, 7.2031, 7.2316, 7.0520,
  6.8762, 7.1098, 6.8042, 6.8071, 6.7592, 8.2892, 6.5602, 6.8519, 7.0690.

Compare the same test on the fast configuration, where the pre-clip gradient norm differs
at step 0 and by a factor of 6 at step 50.

## What it costs

Measured on `tul_a1`, 60 steps, same card:

| | tokens/s | steps/s | peak alloc | batch |
|---|---:|---:|---:|---:|
| fused kernels (production) | 24780 | 2.02 | 20.70 GB | 12 |
| reproducible configuration | 10845 | 1.77 | 18.64 GB | **6** |

**2.28× fewer tokens per second, and roughly half the batch.** Eager attention materialises
what the kernels fuse, so batch 12 is a hard OOM at 31 GB. A token-matched reproducible run
therefore costs about 2.3× the wall clock.

The halved batch is not a free variable: it changes the gradient noise scale, so a
reproducible run is **not** the same experiment as a fast-configuration run. Arms that are
to be compared must all use one configuration.

## How to use it

`training.deterministic: true` (new, defaults false) plus `model.use_kernels: false`, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the process. The flag raises if that
environment variable is missing, and prints a warning if `use_kernels` is left on, because
that combination looks reproducible and is not.

Use it for anything that must bisect or compare two runs. Do not use it for production
runs — it is 2.28× slower for no quality benefit.

## Not verified

- That a **long** reproducible run stays bit-identical. 300 steps is verified; 4000 is not.
  Nothing in the mechanism suggests drift, but that is reasoning, not a measurement.
- That the reproducible configuration diverges at the same *rate* as the fast one. Different
  attention implementation and half the batch: the takeover statistics have to be
  re-measured there, not carried over.
- Whether making the Triton attention backward deterministic is practical. That would give
  reproducibility at full speed and is not attempted here.
- Whether `warn_only=True` is masking an op with no deterministic implementation. It reports
  0/6 differing, so nothing is currently escaping, but the flag is permissive by design.
