# Results: TUL arm v1a — the MUX local head

Status: **FAILED.** The arm aborted on sustained divergence at step 2800 of 3500. The
seed-matched control (`seedsweep-s1`, tul_a1, batch 6, seed 1) ran healthy past step
3250. All five pre-registered predictions failed.

**Pre-registration:** the ARM was pre-registered at
[`lab/experiments/planned/2026-08-25-mux-head-arm-v1a.md`](../../planned/2026-08-25-mux-head-arm-v1a.md).
**These forensics were NOT pre-registered** — they are diagnostic measurements taken
after the abort to answer one question the abort left open: did the head ever work?

Raw numbers: [`results.json`](results.json). Per-probe dumps:
`p5_unigram.json`, `p2_slot_path_worth_step{500,2000,2500}.json`, `p3_readout_jacobian.json`.

- Arm run log: `/home/wolfe/morph-scratch/v1a/run.log`; wandb `adew-me/morph-tul/1q4geafo`.
- Control run log: `/home/wolfe/morph-scratch/seedsweep/s1.log`; wandb `adew-me/morph-tul/ne3t9917`.
- Checkpoints: `checkpoints/morph/tul-v1a-s1/` (500, 1000, 1500, 2000, 2500,
  `DIVERGED_step_2800.pt`). **There is no step-3000 checkpoint**, so P2/P3/P4, which
  named step 3000, are reported at the nearest surviving steps and that substitution is
  stated at every table.

## Probe set

One fixed eval set for every probe: `--config tul_v1a`, `training.batch_size=6`,
`model.use_kernels=false`, 8 batches from
`create_dataloader(split="validation", skip_samples=60_000, tul=val_data_cfg)` — the
same loader arguments `region_shapley.py`, `slot_path_worth.py` and
`readout_jacobian.py` already use. `tul_v1a` differs from `tul_a1` only in loss weights
(`emit_weight`, `plast_weight`, `mux_*`); model geometry and the layout are identical,
so these numbers are comparable to the control-family references.

## Verdicts

| # | Prediction | Threshold | Measured | Verdict |
|---|---|---|---|---|
| P1 | reaches step 3500, no takeover abort, median core grad share < 0.5 | — | **ABORT at 2800**; median core share 0.135 (control 0.001) | **FAILED** (the abort clause; the median clause held) |
| P2 | plan-off ablation cost ≥ 0.04 nats at step 3000 | 0.04 nats | 0.0088 / 0.0189 / 0.0036 nats at steps 500 / 2000 / 2500 | **FAILED** |
| P3 | slot/token readout Jacobian ratio at 3000 > the arm's own 1500 | reversal | 0.0325 → 0.0088 → 0.0045 at 500 → 2000 → 2500 (still falling) | **FAILED** |
| P4 | `val/ce_tokens` at 3000 within +0.05 nats of control | +0.05 nats | **+2.118 nats** at step 2750 (ppl_tok 972.59 vs 116.93) | **FAILED** |
| P5 | MUX local CE beats 0.8x the corpus unigram prior | ≤ 5.858 nats | best-ever 6.255 (train, step 2660); best val 6.964 (step 1250) | **FAILED** |

### P5 — the deciding number

The corpus unigram prior, estimated from **2,457,600 independent training tokens**
(23,438 of 49,169 vocab entries observed, add-one smoothing, `slot_id` masked and
renormalised exactly as `_tul_mux_loss` does), scored against the same MUX targets with
the same reduction as `_tul_mux_loss` (`-(alpha * logp[input_ids] * pos_valid).sum() /
slot_supervised.sum().clamp(min=1)`):

```
unigram baseline CE            = 7.3227 nats
0.8 x unigram (the threshold)  = 5.8581 nats
uniform prior ln(V=49169)      = 10.8030 nats
```

The head, measured on the same 8 probe batches:

| checkpoint | mux_local | / unigram | beats 0.8x? |
|---|---|---|---|
| step_500  | 7.1927 | 0.982 | no |
| step_1000 | 7.0267 | 0.960 | no |
| step_1500 | 7.0297 | 0.960 | no |
| step_2000 | 7.4712 | 1.020 | no |
| step_2500 | 7.5785 | 1.035 | no |
| DIVERGED_step_2800 | 7.8995 | 1.079 | no |

The run's own logged series agree. `val/mux_local` best is **6.9640 at step 1250**;
`tul/mux_local` (per-step, noisy) starts at 11.1979, reads 6.3592 at step 1320, and its
whole-run minimum is 6.2546 at step 2660. Every one of those is above 5.8581.

**P5 FAILED, and it failed at every point in the run, not only at the end.** The head
beat the unigram prior by at most 4-5% of a nat-ratio (0.96) at its best, and after
step 2000 it was *worse* than the unigram prior. z never carried span-specific content.

### P2 — plan worth (`slot_path_worth.py`)

Cost in nats of zeroing what `prefix_project` writes into the coda sequence. Reference:
0.0191 nats at the control family's `ROLL_step_1750`. **Step 3000 does not exist.**

| checkpoint | Δ ce_main | Δ loss | Δ ce_plast | Δ ce_emit |
|---|---|---|---|---|
| step_500  | **0.0088** | 0.0088 | 0.0099 | 1.3697 |
| step_2000 | **0.0189** | 0.0185 | 0.0110 | 0.5012 |
| step_2500 | **0.0036** | 0.0037 | 0.0036 | 1.2817 |

Best case 0.0189 nats — indistinguishable from the 0.0191 baseline, less than half the
0.04 threshold, and by step 2500 it has fallen to 0.0036. The MUX head bought no plan
worth at all.

### P3 — readout Jacobian (`readout_jacobian.py`)

Median gradient norm of `ce_main` w.r.t. the coda input at slot positions over the same
at token positions. Control family: 0.063 / 0.055 / 0.042 at steps 1650 / 1750 / 1850.
**Step 3000 does not exist.**

| checkpoint | median g slot/token | median g·h slot/token | ce_main | sanity |
|---|---|---|---|---|
| step_500  | **0.0325** | 0.2058 | 5.9423 | all pass |
| step_2000 | **0.0088** | 0.1730 | 5.1939 | all pass |
| step_2500 | **0.0045** | 0.1786 | 6.2285 | all pass |

The trend does not reverse. It falls by 7.2x between step 500 and step 2500, and every
value is below the control family's range. The coda reads the plan *less* under the MUX
head, not more.

## What the measurements say about WHY

1. **The head never bit (P5).** The MUX objective on a deterministic z converged to
   roughly the corpus unigram — the span-independent answer. The pre-registration named
   this the refuter: if the head cannot beat 0.8x the unigram prior, z carries nothing
   span-specific and the head is decorative regardless of P1-P4. It could not, so P2 and
   P3 failing is the expected consequence, not an independent second failure.
2. **The head brought the takeover forward by ~1200 steps.** Core gradient share
   (clip-invariant, sampled every 100 steps):

   | step | 1300 | 1400 | **1500** | 1600 | 1700 | 2700 | 2800 | 2900 |
   |---|---|---|---|---|---|---|---|---|
   | v1a arm | 0.122 | 0.099 | **0.533** | 0.964 | 0.962 | 0.997 | (abort) | — |
   | control | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.143 | 0.969 | 0.993 |

   Both runs take over. The arm's onset is step ~1500; the control's is step ~2700.
   The takeover is therefore **not** caused by the MUX head — it is the standing TUL
   failure mode — but the head accelerated it decisively. That is consistent with the
   mechanism: `_tul_mux_loss` puts a direct, large gradient on `h_slots` that bypasses
   the coda entirely, so the core gets loss pressure the coda cannot balance.
3. **Retiring `ce_emit` was not the cure.** The pre-registration's hypothesis was that
   `ce_emit` supplied the fuel for the takeover. `emit_weight` was 0.0 for this whole
   run and the takeover still fired — earlier than in the control, which still had
   `ce_emit` on. The `ce_emit` race is not the (only) fuel.
4. **The failure reading the pre-registration wrote down is the one that happened,
   minus P1.** It predicted "P1 holds but P2/P3/P5 fail → the war was `ce_emit` alone
   and deterministic z stays empty". P2/P3/P5 did fail, and deterministic z is empty —
   but P1 also failed, so the second half of that reading ("the war was `ce_emit`
   alone") is refuted too. v1b's K-candidate upgrade is being asked to fix an emptiness
   that this arm shows is not caused by determinism alone, on top of a takeover that
   this arm shows the head makes worse.

## Concurrency during these forensics (read this before reusing the numbers)

A parallel session was working in the same checkout while these probes ran. Two events:

1. **`morph/model/transformer.py` and `morph/model/tul.py` changed at 00:06:01**, and
   commit `d3a86da` ("Fix the MUX head's gradient path: detach the weight-tied readout")
   landed at 00:07:57. That commit also swept this directory's `p2_*.json`, `p3_*.json`,
   `p5_unigram.json` and `lab/divergence/mux_unigram_baseline.py` into itself. **These
   forensics did not run `git commit`;** those files were committed by the other session.
2. **A new training run `tul-v1a2a-s1` started at 00:09:24** (PID 3675202, 22 GB VRAM),
   after every measurement here had finished.

Timing of the outputs against the 00:06:01 code change:

| output | written | code state |
|---|---|---|
| `p5_unigram.json` | 00:04:48 | pre-change (certain) |
| `p2_slot_path_worth_step500.json` | 00:05:14 | pre-change (certain) |
| `p2_slot_path_worth_step2000.json` | 00:05:26 | pre-change (certain) |
| `p2_slot_path_worth_step2500.json` | 00:05:39 | pre-change (certain) |
| `p3_readout_jacobian.json` | 00:06:23 | **ambiguous** — the process started between 00:05:40 and 00:06:23 |

Why the ambiguity cannot move the P3 numbers:

- The commit deletes **zero** lines from every `morph/` file (`git show d3a86da
  --numstat`). Its only `transformer.py` hunk is at line 1975, inside `_tul_mux_loss`
  (lines 1955-1993); its only `tul.py` hunk adds one `TULConfig` field. Nothing outside
  the MUX loss changed.
- `readout_jacobian.py` backwards `ce_main` **only**, never the total loss. `ce_main` has
  no dependence on `mux_loss`, so a `detach()` inside `_tul_mux_loss` cannot reach it.
  `slot_path_worth.py` runs entirely under `torch.no_grad()`. `detach()` never changes a
  forward value in any case.
- Empirical cross-check: `readout_jacobian`'s `ce_main` means (5.9423 / 5.1939 / 6.2285)
  match `slot_path_worth`'s `full` `ce_main` at the same three checkpoints to four
  decimals, and `slot_path_worth` ran wholly pre-change. Two independent processes agree
  on the forward.

**The definitive re-run of P3 under the current code was attempted and OOM'd** (exit 1,
`torch.OutOfMemoryError`, 409 MiB free) because `tul-v1a2a-s1` now holds the GPU. It was
not retried, so the argument above rests on diff scope plus the `ce_main` cross-check,
not on a repeated measurement.

## Not verified

- Nothing was measured at step 3000 for any prediction; that checkpoint does not exist.
  P2/P3/P4 verdicts rest on nearby steps (2500 or 2750), which is a substitution, not
  the pre-registered measurement.
- P1's "median core pre-clip gradient share" was computed from the run's logged
  post-clip per-region gradnorms at 100-step intervals. The clip is a single uniform
  rescale so the share is exact, but the sampling is 100-step, not per-step.
- The unigram prior was estimated from the head of the train stream (the first 400
  batches of the deterministic unshuffled stream), not from a random sample across the
  corpus. A different slice would move the baseline; the arm's margin (0.96-1.08 of the
  baseline, against a 0.80 threshold) is far too large for that to change the verdict.
- No causal test was run for claim 2 above (that the MUX gradient on `h_slots` is what
  advanced the takeover). The evidence is the onset-step contrast between two runs,
  which is n=1 per arm. MORPH run comparisons at n=1 are known to be unreadable for
  small effects; a 1200-step onset shift is large, but it is still n=1.
- `tul/mux_local` and `val/mux_local` were read from the local wandb datastore
  (`/home/wolfe/morph-scratch/wandb/run-20260825_225247-1q4geafo/run-1q4geafo.wandb`),
  not from the wandb web API.

## Commands

```
PYTHONPATH=/home/wolfe/morph-perf $PY lab/divergence/mux_unigram_baseline.py \
  --unigram-batches 400 --batches 8 \
  --ckpts checkpoints/morph/tul-v1a-s1/step_{500,1000,1500,2000,2500}.pt \
          checkpoints/morph/tul-v1a-s1/DIVERGED_step_2800.pt \
  --out lab/experiments/results/2026-08-25-mux-head-v1a/p5_unigram.json          # exit 0

for st in 500 2000 2500; do
  PYTHONPATH=/home/wolfe/morph-perf $PY lab/divergence/slot_path_worth.py \
    --config tul_v1a --ckpt checkpoints/morph/tul-v1a-s1/step_$st.pt --batches 8 \
    --out lab/experiments/results/2026-08-25-mux-head-v1a/p2_slot_path_worth_step$st.json
done                                                                             # exit 0, 0, 0

PYTHONPATH=/home/wolfe/morph-perf $PY lab/divergence/readout_jacobian.py \
  --config tul_v1a --batches 8 \
  --ckpts checkpoints/morph/tul-v1a-s1/step_{500,2000,2500}.pt \
  --out lab/experiments/results/2026-08-25-mux-head-v1a/p3_readout_jacobian.json  # exit 0
```

`$PY = /home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python`, cwd `/home/wolfe/morph-perf`,
branch `perf/throughput-lever-stack`. No trainer was running (`pgrep -fa
"[m]orph.training.train"` exit 1); no new training run was started.
