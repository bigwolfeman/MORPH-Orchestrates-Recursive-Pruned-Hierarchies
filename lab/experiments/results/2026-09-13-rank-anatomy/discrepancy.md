# The 5.7598 gap, and where the prefix write loses rank (2026-09-13)

Two questions the rank anatomy left open, both answered on `slot-spandec-strict`
`step_5000.pt` with `lab/divergence/slot_rank_probe_variants.py`, run on the DGX Spark
(`/home/wolfe/morph-instruments/MORPH` at `f89256d`, GB10, eager + `tg_scoped_kernels`).
Raw output: `variants-strict-trainer-rows.json` (the trainer's exact final-eval rows) and
`variants-strict-offsets.json` (two other row sets).

## 1. Why the trainer logged 5.7598 / 0.7104 and the checkpoint reads 13.85 / 0.52

**Cause: a code change, candidate (f).** Commit `7a24adf` (2026-09-13 08:01, "One home
for the TG relation (`_tul_tg_kwargs`)") changed one line inside
`MORPHTransformer.tul_slot_state_probe`:

```
-        x, x0, bigram = self._tul_front(input_ids, layout)
+        _fkw, _freset, _ckw, _creset = self._tul_tg_kwargs(layout)
+        x, x0, bigram = self._tul_front(input_ids, layout, attn_kwargs=_fkw,
+                                        ret_reset_mask=_freset)
```

The run (`/home/wolfe/morph-scratch/arc/slot-spandec-strict/run.log`, started 2026-09-12
04:21, code from the `a93b6a1` / `047bf79` era) logged from the OLD line. On this model
(`tg_geometry: strict`) the old probe ran the PRELUDE UNRESTRICTED while the model was
trained same-span only, so every state it measured is off-distribution. Nothing else in
the probe changed; `_tul_core` already took `input_ids` before that commit.

The same defect is what `7a24adf` fixed in the horizon grid, and the reasoning is the
same (`morph-memory: instruments-must-use-the-models-tg-kwargs`).

### The reproduction

The trainer's val loader is never rebuilt on this run (`_curr_val_batches` is None with no
curriculum), so the 19 periodic evals at `eval_every: 250` consume 380 batches and the
FINAL eval reads batches 380..399. Scoring that exact row set:

| reading | `slot_eff_rank` | `slot_pairwise_cos` | `ce_tokens` |
|---|---|---|---|
| the run's last `[VAL]` / `Final val_loss=` line | **5.7598** | **0.7104** | **4.4249** |
| `bare_fp32` (the trainer's exact context) | **5.7598** | **0.7104** | **4.4249** |
| `bare_bf16` | 5.7590 | 0.7105 | |
| `shipped_fp32` (HEAD's probe) | 13.1183 | 0.5430 | |
| `shipped_bf16` | 13.1211 | 0.5430 | |

`spandec_ce` reproduces too: 4.5565 logged, 4.5565 measured. `slot_norm_mean` 36.3315 and
`slot_component_std` 1.1352 reproduce on the bare reading to every logged digit.

### Every candidate, with its number

| candidate | verdict | evidence |
|---|---|---|
| (a) mode / dtype / compile | NOT the cause | The trainer calls the probe after its autocast block closes, so it is fp32; the instrument used bf16. The two differ by 0.0008 rank (5.7598 vs 5.7590) and 0.0001 cosine. Both `evaluate` and the instrument run `model.eval()`, so token-state dropout is off in both, and both call the probe on `getattr(model, "_orig_mod", model)`, never the compiled wrapper. |
| (b) a different stage | NOT the cause | The probe body is `_tul_front` then `_tul_core` then `_readout`, and `7a24adf` touched only the front's kwargs. The stage is the loop exit in both versions. |
| (c) live vs saved weights | NOT the cause | The checkpoint reproduces the run's val CE exactly: `ce_tokens` 4.4249 and `spandec_ce` 4.5565 against the logged 4.4249 / 4.5565, on the same rows. The same weights give 5.7598 and 13.1183 depending only on the front's relation. |
| (d) per-batch pooling | NOT a difference | `effective_rank` pools all valid slots of a batch and `evaluate` averages the per-batch values; the instrument does the same. Both readings above use batch 6, 20 batches. |
| (e) a different step or row set | NOT the cause, but it moves the number | The log line and the checkpoint are both step 5000. The row set is worth ~1.2 rank units: `shipped_fp32` reads 13.8587 at stream offset 0, 13.1183 at 380 (the trainer's rows), 14.3568 at 400; `bare_fp32` reads 5.8762 / 5.7598 / 6.0361 on the same three. Per-batch spread inside one set is much larger (9.6 to 18.0). |
| (f) a code change | **THE CAUSE** | `7a24adf`, the one line above. Bare 5.7598 / 0.7104 against shipped 13.1183 / 0.5430 on identical rows, identical weights, identical dtype. |

### Which number is right, and what to use

The RIGHT definition is the shipped one. The coda reads a state produced by a prelude
that ran under the model's own relation; a state from an unrestricted prelude is one no
reader ever sees. The old number is not a lower reading of the same quantity, it is a
reading of a different forward.

**Baseline for any 2026-09-13 prereg on the strict ruler: `val/slot_eff_rank` 13.85,
`val/slot_pairwise_cos` 0.520** (shipped probe, trainer val recipe, `skip_samples=50_000`,
batch 6, 20 batches from the STREAM START). Two independent implementations agree there:
13.8587 / 0.5198 through the trainer's own loader here, 13.8466 / 0.5201 through the
anatomy's stream packer (`strict-trainer-matched.json`). Offset 0 is the recipe to pin,
because it does not depend on how many evals a run happened to do. The probe family's own
recipe (`skip_samples=0`, 480 rows, batch 3) reads 11.9597 / 0.5635 and is a different
ruler; do not mix the two.

Two rules that follow:

1. **Score arms on the SAME rows.** The row set is worth ~1.2 rank units and a single
   batch is worth 8. A rank comparison between two arms is only readable when both read
   the same stream positions.
2. **Every `val/slot_eff_rank` logged before `7a24adf` by a model whose FRONT is
   restricted is a bare-front number and is not comparable with one logged after.** The
   front is restricted when `tg_geometry == "strict"`, or `tg_restrict` is true at
   `tg_restrict_scope: "all"` (`_tul_tg_kwargs`). Measured on the configs: the
   spandec-strict, spandec-mask and register arms are affected; `tul_a1` and the
   `slot-mux-*` arms have `tg_restrict: false`, their front kwargs are None either way,
   and their logged numbers (6.34, 7.12, 7.29 and so on) still stand.

## 2. Where the prefix write loses the rank

`W_prefix` is not the narrow thing. Audited on the same checkpoint:

| | `W_prefix[0]` | `W_prefix[1]` |
|---|---|---|
| shape / dtype | `[1024, 1024]` fp32 (parameter is `[2, 1024, 1024]`) | same |
| distinct values / fraction zero | 1,041,355 / 0.000 | 1,041,377 / 0.000 |
| participation ratio of the σ² spectrum | **680.3** of 1024 | **663.8** of 1024 |
| σ max / median / min | 2.8117 / 0.9966 / 0.2792 | 2.9454 / 0.9973 / 0.3083 |
| top-1 / top-8 share of σ² | 0.0068 / 0.0476 | 0.0073 / 0.0481 |
| mean diagonal, ‖offdiag‖/‖diag‖ | 1.0100, 0.344 | 1.0106, 0.375 |

It is dense, fp32, and still close to its identity init after 5000 steps. **It is not
quantized.** The ternary scope walks modules and quantizes `nn.Linear`, `nn.Embedding` and
CMS layers (`morph/model/ternary_qat.py`); `W_prefix` is a bare `nn.Parameter` on
`TULSlots`, so no parametrization is registered on it (the checkpoint holds 46
`parametrizations.weight.original` tensors and none of them is a TUL tensor) and
`prefix_project` multiplies the raw fp32 weight, cast to the carrier's dtype. It is also in
the no-decay group on purpose (`morph/training/optimizer.py::_NO_DECAY_KEYWORDS`).

### The stage table (raw view, per-row median over 120 rows, trainer rows)

| set | vectors/row | rankRaw | rankCen | pc1Cen | cos | ‖·‖ |
|---|---|---|---|---|---|---|
| exit | 54 | 1.336 | **12.440** | 0.196 | 0.863 | 47.43 |
| `h W_0` | 54 | 1.758 | 8.942 | 0.248 | 0.745 | 57.07 |
| `h W_1` | 54 | 2.115 | 9.031 | 0.247 | 0.674 | 52.65 |
| both cells, one pooled set (`sW_write`) | 108 | 2.733 | **7.076** | 0.321 | 0.579 | 54.81 |
| both cells, each centered on its OWN mean | 108 | 12.796 | **12.796** | 0.199 | −0.003 | 28.63 |

The same in the readout view: exit 9.289, `W_0` 8.575, `W_1` 7.493, pooled 8.981,
cell-centered 10.257.

### The reading

**The halving is mostly a pooling artefact, and the rest is the matrix's anisotropy, not
its rank.**

1. **Removing the inter-cell offset recovers the whole drop, and a little more.** The two
   cells of a slot differ by a fixed vector: `‖μW_0 − μW_1‖ = 39.80`, which is **2.29x**
   the per-slot centered residual RMS. Pool the 108 vectors, center them once, and that
   single direction becomes the largest component of the centered spectrum (`pc1Cen` 0.196
   at the exit against 0.321 at the write). Take each cell's own mean out instead and the
   SAME 108 vectors read **12.796**: +5.72 against the pooled 7.076, and +0.36 against the
   exit's 12.440. Two cells carry slightly more directions than one, which is what a
   register is supposed to do. The pooled statistic cannot see that, because it charges
   the set for an offset the coda gets for free: the two cells sit at different positions
   and the coda knows which is which.
2. **One cell read ALONE does lose ~3.5 rank units, and that is anisotropy, not rank.**
   `h W_0` reads 8.942 and `h W_1` 9.031 against the exit's 12.440, while the matrix has
   680 effective singular directions and a condition number near 10. `W_k` amplifies the
   centered residual overall (gain 1.638 on `W_0`, 1.724 on `W_1`, against 1.110 and 0.976
   on the row's mean direction), but it amplifies some residual directions more than
   others, so the centered energy concentrates and the participation ratio falls. The two
   cells do not lose the same directions, which is why their union is back above the exit.
   A deeper rank cut is not available from a matrix this well conditioned.
3. **In the space the coda reads, there is no halving at all.** Through `lm_mixer` +
   `final_norm` the write is 9.289 to 8.981, and 10.257 with the cell offset out. The
   "13.17 to 7.16" line in `README.md` is a raw-view, single-pooled-set statement and
   should be quoted with both qualifiers.

`README.md` reading 2 says "the prefix write then cuts it to 7.16 while DOUBLING the
number of vectors, `W_prefix` is the narrowest thing downstream of the prelude". That is
the measurement, and this section is its cause: the narrow thing is the pooled statistic,
not the matrix. Reading 5 ("the register is aimed at the wrong bottleneck") loses its
`sW_write` half; the seed and the prelude half still stands.

## What this does not say

The variant probe mirrors `tul_slot_state_probe` for the ruler family only and RAISES on a
`cond_layers`, `vq_codes`, `center_exit` or `slot_cells > 1` model, so the cross-campaign
rule in part 1 is derived from the config knobs, not measured on those arms. Only
`slot-spandec-strict` was re-read; `coretok` and `prev-reach1` keep their anatomy numbers
and are not re-derived here. The write-stage table is 120 rows against the anatomy's 480,
so its exit row (12.440) and the anatomy's (13.17) differ by the row set. The
cell-centered readout row applies `lm_mixer` + `final_norm` to an already-centered vector,
which is no stage of the shipped forward, and is a diagnostic only. Nothing here is a CE
or loop-contribution result.
