# Agent Note: val/ppl_tokens averaged exp(CE) instead of exponentiating mean CE

Status: implemented

## Problem

`morph/training/evaluate()` reports two perplexities. The baseline path returns
`exp(mean CE)`. The TUL path accumulated `exp(CE)` **per batch** and then averaged those,
giving `mean(exp(CE))`. Jensen's inequality makes the second strictly larger whenever the
validation batches differ, and they always differ.

The two numbers are compared arm-to-arm. `docs/ablation-ledger.md` names
`val/ppl_tokens` as the primary per-arm metric, and `docs/tul-arms-result.md` printed it
in the same table as the baseline's `exp(mean CE)`. The `evaluate()` docstring stated the
opposite of the truth:

> `val/ppl_tokens` is over TOKEN positions only, so it stays comparable to the baseline's
> token PPL.

Token positions were never the problem. The aggregation was.

Size of the error on the arm already published: `tul-a1-acap1` logged `val/ppl_tokens`
**25.8907**, while `exp(val/ce_tokens = 3.2243)` is **25.14**. The A1-vs-A0 effect the
table exists to report is 1.27 PPL, so the artifact was **59 % of the result**.

`val/halt_ppl_tokens` had the same form. Arm TUL-halt is one of the two arms in the gate
bake-off, so the defect was about to ship into that comparison too.

## Decision

Delete both per-batch `exp` accumulations. Derive PPL once, after the CE means exist:

    for _ce_k, _ppl_k in (("val/ce_tokens", "val/ppl_tokens"),
                          ("val/halt_ce_tokens", "val/halt_ppl_tokens")):
        if _ce_k in extra:
            extra[_ppl_k] = math.exp(min(extra[_ce_k], 20.0))

Two regression tests in `tests/test_train_phase.py` drive `evaluate()` with a stub model
over batches of deliberately different CE:

- `test_val_ppl_tokens_is_exp_of_the_mean_not_the_mean_of_exps`
- `test_val_ppl_tokens_uses_the_same_aggregation_as_the_baseline_val_ppl`

Both **fail** against the previous code (27.358 against an expected 20.086) and pass
after. `pytest tests/` → 192 passed.

`docs/tul-arms-result.md` is corrected in the same change, with the old figure and the
reason recorded inline rather than silently overwritten.

## Alternatives considered

- **Log both forms and let readers choose.** Rejected: two metrics with one name is how
  this happened. A reader comparing rows in one table cannot be expected to check the
  aggregation of each cell.
- **Change the baseline to `mean(exp)` instead.** Rejected: `mean(exp(CE))` is not a
  perplexity of anything. It has no interpretation as tokens-per-branch, it is not
  comparable to any published number, and it is unstable — one high-CE batch dominates.
- **Leave the historical runs' logged values alone and only fix forward.** Partly kept:
  wandb history cannot be rewritten, so the note and the results doc both record that
  every `val/ppl_tokens` logged before 2026-08-23 is inflated and must be recomputed as
  `exp(val/ce_tokens)`.

## Consequences

- Every `val/ppl_tokens` and `val/halt_ppl_tokens` in wandb before 2026-08-23 is
  **inflated**. Recompute as `exp(val/ce_tokens)`. `val/ce_tokens` itself is a plain mean
  and was always correct, so no CE-based conclusion changes.
- The direction of the A1-vs-A0 result does not change — A1 already won on CE. What
  changes is the size: the PPL gap is 1.45 (26.59 → 25.14) against A0c, not 0.70.
- The gate bake-off will log the corrected metric from step 0, so `TUL-gate` and
  `TUL-halt` are comparable to the baseline arms without post-hoc repair.
- Related: [alpha-cap-belongs-in-a-config](2026-08-22-alpha-cap-belongs-in-a-config.md).
  Same failure class — a number that decides an experiment living somewhere nobody
  re-derives. There it was a config in a gitignored script; here it was an aggregation
  inside a metric name.
