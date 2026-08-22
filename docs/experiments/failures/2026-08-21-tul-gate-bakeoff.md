# Experiment: the gated-TUL bake-off

Status: failure (no verdict — every arm died before its question could be answered)

Predictions for these arms were pre-registered in `docs/tul-gate-spec.md` §11, committed as
`3bad5eb` before any code was written. There was no `docs/experiments/planned/` file; that is
itself a process gap, and this record exists because §11 says "a results note is written for
these arms whether they win or lose."

## Question

Does a model-chosen span length pay (`TUL-gate` vs `TUL-A1`), and does gate-driven variable
depth pay on top of it (`TUL-halt`)?

**Pre-registered:** `TUL-halt` does not beat `TUL-gate` on val CE; fixed depth wins or ties.
**Falsifier:** `TUL-halt` beats `TUL-gate` by more than the `A1`/`A1r` retrain noise floor
without losing on generation metrics.

## Method as run

`ignore/perf/gate_bakeoff.sh`, sequential on the 5090, 20000 steps per arm, batch 14,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Arms `tul_gate` -> `tul_a1` -> `tul_a1r`.
Started 2026-08-22 00:34.

## Results

| arm | gate | outcome | reached |
|---|---|---|---|
| `tul_gate` | on | CUDA OOM | step ~1050 of 20000 |
| `tul_a1` | **off** | DIV-GUARD abort | step 5900 of 20000 |
| `tul_a1r` | **off** | DIV-GUARD abort | step 2080 of 20000 |

**No arm produced a comparison.** The A1 reference never finished, so there is nothing for the
gate arm to be measured against, and the gate arm did not survive to its own first checkpoint.

### Why each died

- `tul_gate`: 23.58 GB peak against a desktop that was using ~6 GB of the 31.4 GB card. The
  gate's own footprint is 0.07 GB (23.58 vs A1's 23.51), so this is headroom, not a gate cost.
  It was detonating anyway — `train/grad_norm` was 8.0e5 by step 800.
- `tul_a1` and `tul_a1r`: a slow loss detonation. `tul_a1` val CE bottomed at 4.806 (step 1000)
  and climbed to 6.127 by 5500. Both are gate-free (`gate: false` builds no gate parameter and
  leaves the data path untouched, `morph/training/tul_setup.py:117`), so **the divergence is a
  property of the TUL short recipe, not of the gate.** These were the first TUL arms ever
  trained, which is why it had never been seen.

### The one number the run did produce

At the two evals `tul_gate` survived, the halting arm collapsed to `depth = 1.00`:

| | val CE | gate k / gold | halt CE | halt depth |
|---|---|---|---|---|
| step 500 | 5.7539 | 19.01 / 20.07 | 5.7215 (+0.0010 vs fixed) | 1.00 |
| step 1000 | 6.4542 | 17.35 / 20.01 | 6.3969 (-0.0029 vs fixed) | 1.00 |

`k0 = 0.000` at both: the gate never asks for zero tokens, so `choose_k(g) >= 1` fires at the
first iteration for every slot and the loop always stops at `t = 1`. This is structural, not a
tuning accident — it follows from the §7 encoding, where "loop again" is the `k = 0` codepoint
and the length target is bounded away from 0. **`TUL-halt` as specified does not test variable
depth; it tests depth 1.** Any future halting arm needs a different encoding for "continue".

Both halt deltas are far inside any plausible noise floor, and both arms were mid-detonation,
so the pre-registered prediction is neither confirmed nor falsified.

## Verdict

Inconclusive, which under `docs/experiments/AGENTS.md` is a protocol failure and is filed here.

**What the method could not distinguish:** anything about the gate. The design assumed the A1
reference was a stable baseline. It had never been run, and it is not stable. A bake-off cannot
resolve an arm against a reference that diverges.

**Second defect:** three arms were queued back to back with no stability gate between them, so
~10 GPU-hours were committed before the first arm's health was read. A short reference run
would have exposed the divergence in 20 minutes.

## Next

`docs/experiments/planned/2026-08-22-tul-divergence-cause.md` — find and fix the divergence
first. The bake-off is re-run only after a TUL reference arm holds for 6000+ steps.
