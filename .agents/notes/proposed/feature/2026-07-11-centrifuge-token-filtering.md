# Agent Note: Centrifuge Token Filtering

Status: proposed

Origin: Ai-notes/07-11-2026/Centrifuge-Token-Filtering-Ablation/PLAN.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Centrifuge / Reference-Guided Token Filtering Ablation

**Date:** 2026-07-11  
**Status:** proposed; no MORPH implementation or training result yet  
**Paper:** [Unlocking Full Efficiency of Token Filtering in Large Language Model Training](https://arxiv.org/html/2502.00340v2)  
**Upstream code:** [Di-Chai/Centrifuge](https://github.com/Di-Chai/Centrifuge)  
**VLT thread:** `default/morph-centrifuge-ablation`

## Decision under test

Centrifuge has earned a bounded MORPH ablation, but it has not earned a place in
the default training stack.

The experiment must separate three claims that the paper partially combines:

1. **Selector utility:** reference-guided excess-loss selection improves the
   training signal relative to supervising every token.
2. **Selective-gradient utility:** deleting additional backward paths preserves
   the benefit of loss-only token selection.
3. **Systems value:** compacting the selected-token backward pass produces a
   material end-to-end speedup on MORPH's real optimized backend.

A failure of an earlier claim stops work on later claims. In particular, there
is no reason to build MORPH-native backward compaction if the selector itself
does not improve the model.

## Load-bearing scope correction: never during TST

Token filtering is **off for the entire Token Superposition Training phase**.
TST's multi-hot target bags define a different supervision unit, so applying a
single-token importance mask there would test an ill-specified hybrid rather
than either method.

If the parent run uses TST, all experimental arms share the exact same TST
trajectory and fork only at the existing TST-to-standard-NTP transition:

```text
shared initialization and data
          |
          v
TST superposition phase (token filtering OFF)
          |
          v
single transition checkpoint at standard-NTP boundary
          |
          +--> R0: all-token NTP recovery
          +--> R1: random-mask NTP recovery
          +--> R2+: reference-guided NTP recovery
```

The transition checkpoint is the common experimental starting artifact. Each
arm must restore the same model weights, optimizer/scaler state, RNG state,
curriculum state, data manifest, and next-batch position. A run is invalid if
the arms do not consume the same raw sequences in the same order.

This design answers the actual question: does reference-guided filtering improve
the standard-NTP recovery phase after the shared MORPH/TST prelude?

## What the paper changes

The selector keeps tokens with high excess loss:

```text
importance_i = target_nll_i - reference_nll_i
```

At a fixed keep rate, tokens with the largest excess loss remain supervised.
The forward pass remains dense, preserving the full causal context.

Loss-only filtering zeros the LM-loss contribution for dropped positions.
Centrifuge goes further: it compacts the backward graph and intentionally
deletes contextual `dK`/`dV` credit into dropped token states while preserving
selected-query `dQ` contributions from the full K/V context. Therefore the
Centrifuge gradient is not mathematically identical to the loss-only objective.

For MORPH this distinction is unusually important. Inter-token or recurrent
credit also flows through:

- CCA causal convolution and compressed representations;
- CSA/HCA pooling, indexing, and compressed attention;
- the GLA recurrence across sequence positions;
- GLA state carried across shared-core iterations;
- the repeated core and its truncated/checkpointed backward window;
- whole-body ReMoE routing after carve.

The paper's standard-attention reasoning cannot be assumed to cover these paths.

## Experimental contract

### Frozen across arms

- Common transition checkpoint and `next_step`.
- Identical raw batches and order.
- Identical optimizer, LR schedule, gradient accumulation, clipping, and bf16
  policy.
- Identical model construction, retention behavior, pruning/carve/routing state,
  and loop-depth sampling.
- Identical evaluation schedule and generation settings.
- Hydra-resolved config logged in full to W&B.
- Reference artifact identity, tokenizer identity, data-manifest hash, selector
  implementation version, keep-rate policy, and every selector constant logged
  in the W&B config.
- Runs execute sequentially on a GPU; do not colocate two training models on the
  same device.

### Three comparison budgets

Every result must be reported against all three budgets:

1. **Equal raw tokens:** same sequences and number of recovery steps.
2. **Equal supervised tokens:** continue filtered arms until their count of kept
   targets matches the baseline's supervised-target count.
3. **Equal wall clock:** compare the best checkpoint reachable in the same elapsed
   recovery-training time.

The equal-raw-token comparison tests data selection. Equal-supervised-token and
equal-wall-clock comparisons prevent a keep-rate accounting trick from being
misreported as sample or systems efficiency.

Reference-model training, reference scoring, storage, and loading costs remain
separate line items. Report both unamortized cost and amortized cost for the
expected number of target runs.

## Phase 0: reference-loss artifact

Start with a paper-faithful same-tokenizer, same-family reference model trained
on the curated math distribution. A smaller reference can be tested later as a
cost ablation; it must not silently replace the faithful arm.

Precompute reference NLL per target position for the recovery corpus. Each record
must carry enough identity to make misalignment impossible:

- dataset/source and immutable sample identifier;
- tokenizer name/version and vocabulary hash;
- token-id sequence hash;
- packing-boundary metadata;
- reference checkpoint hash;
- per-position reference NLL in fp32 or a separately validated storage dtype.

Hard gates:

- Re-tokenizing a sampled record reproduces its stored token ids exactly.
- Stored reference-loss length equals the number of causal target positions.
- Packing boundaries and ignored/padding positions agree with MORPH labels.
- A sampled online reference forward agrees with stored NLL within the declared
  storage tolerance.
- Missing, duplicated, reordered, or hash-mismatched records terminate the run.

## Phase 1: selector-only ablation

Do this before any Centrifuge-style graph rewrite. The forward and ordinary
MORPH backward remain unchanged except for which NTP positions contribute to
the LM loss.

### Arms

| Arm | Recovery supervision | Purpose |
| --- | --- | --- |
| R0 | All valid NTP targets | Baseline |
| R1 | Random targets, matched keep count | Tests whether less supervision or regularization explains the result |
| R2 | Lowest target NLL dropped | Self-loss control; tests whether trivial/easy-token removal is sufficient |
| R3 | Excess loss, 25% dropped | Conservative reference-guided arm |
| R4 | Excess loss, 40% dropped | Midpoint arm near the paper's useful range |
| R5 | Excess loss, 50% dropped | Paper headline arm |

Run a single-seed screen first. Confirm the baseline, random control, and the
best reference-guided arm with at least three seeds before declaring selector
utility.

### Selector definition

For each packed sequence independently:

1. Compute the current target model's per-position NLL without detaching the
   eventual selected loss path.
2. Load and validate aligned reference NLL.
3. Compute `target_nll - reference_nll` under `no_grad` for selection only.
4. Exclude padding, ignored labels, structural positions explicitly protected by
   the experiment, and positions without valid reference scores.
5. Keep the highest-excess-loss positions at the configured rate.
6. Normalize the scalar LM loss by the actual number of retained targets, not by
   the original token count.

Tie breaking must be deterministic. Exact eligible, kept, and dropped counts are
asserted per sample and logged by source and token category.

The current fused CE does not expose per-token NLL. The implementation must add a
real masked/per-token contract to the fused CE path rather than falling back to
materializing full `[B, T, V]` logits. A likely correct structure is a chunked
selection pass followed by a chunked gradient pass using the frozen mask. The
extra head work must be measured; it is part of selector overhead.

### Phase-1 measurements

- Full-token validation CE/PPL on the ordinary NTP distribution.
- Existing MORPH math graduation evaluations and generation-based task accuracy.
- General-domain retention evaluations already used by the active curriculum.
- Long-context and loop-depth robustness checks where available.
- Selected-token loss and all-token diagnostic loss.
- Keep/drop rate by data source, position, token class, and sequence length.
- Overlap between random, self-loss, and excess-loss masks.
- Forward, selector, backward, optimizer, and total step time.
- Raw tokens/s, supervised tokens/s, peak allocated/reserved VRAM, and host I/O.
- Gradient norm and clipping rate, including per-module summaries.

### Phase-1 acceptance gate

Proceed to systems work only if the reference-guided arm:

- beats the matched random and self-loss controls on the preregistered primary
  math metric;
- does not exceed the allowed general-validation or generation regression;
- remains favorable under equal raw-token and equal wall-clock accounting;
- shows no selector/data alignment failures; and
- reproduces directionally across at least three seeds.

If only the 50% arm wins while 25% and 40% fail, inspect mask composition and
variance before accepting the result. If random masking matches it, the
reference model has not earned its cost.

## Phase 2: selective-gradient semantics

Use the exact stored masks from the winning Phase-1 arm so selection cannot
confound backward semantics.

### Arms

| Arm | Backward behavior | Purpose |
| --- | --- | --- |
| G0 | Loss-only mask; ordinary dense backward | Quality reference |
| G1 | Compact only proven intra-token FFN/projection paths | Smallest systems slice |
| G2 | G1 plus a MORPH-native standard-attention split backward | Tests paper mechanism without touching MORPH-specific memory paths |
| G3+ | One additional MORPH-specific inter-token/state branch at a time | Attribution, only after an explicit reference implementation exists |

CCA compression, GLA recurrence/carry, the shared-loop state boundary, routing,
and any auxiliary loss remain unfiltered in G1/G2. They may be added only as
separate named arms; there is no global "filter everything" shortcut.

### Gradient falsification gate

Before a training run, use a tiny deterministic batch and frozen mask. Record by
module and parameter group:

- gradient cosine against G0;
- relative L2 difference;
- maximum absolute difference;
- fraction of exactly zero rows/elements;
- finite-gradient status;
- one-step parameter-update difference.

Differences intentionally prescribed by the selective-gradient algorithm must
be predicted before the run. Any additional deleted or shape-misaligned gradient
is a failure. Break the reference implementation deliberately in a test and
confirm that the gate fails; a test that remains green is not evidence.

Also prove:

- filtered and unfiltered arms take the same number of optimizer steps;
- no exception can skip a microstep or optimizer step;
- gradient accumulation preserves the same mask semantics on every microbatch;
- checkpoint recomputation sees the same mask as the original forward;
- loop-depth and active-set variation cannot reuse an incompatible generated
  graph or compiled operator;
- retained-token normalization is identical between reference and optimized
  paths.

## Phase 3: real systems measurement

Do not vendor the upstream graph-rewrite extension. Its private-autograd mutation,
model-generated C++, FlashAttention/cuDNN assumptions, and silent skipped-step
error path are not an acceptable MORPH production surface. Implement only the
mechanism that survives Phases 1 and 2 against MORPH's actual Triton/custom
autograd paths.

Benchmark the optimized configuration on the intended 5090 path with the same
Hydra config and transition checkpoint:

- warm all Triton and compiled variants before timing;
- measure at least 200 stable recovery steps;
- report median, p10/p90, and total elapsed time rather than a best iteration;
- separately time forward, selector, compaction/rewrite, backward, pruning or
  routing work, optimizer, and data/reference-loss loading;
- record raw and supervised tokens/s, MFU if trustworthy, and peak VRAM;
- include compilation/offline generation cost and the amortization denominator;
- inspect a profiler trace to verify that smaller GEMMs or reduced work occurred
  on the claimed paths.

An HTTP success, completed process, or lower isolated backward time is not an
end-to-end speed result.

### Systems acceptance gate

Promotion requires all of the following:

- at least 10% end-to-end recovery-step speedup on MORPH's optimized live path;
- quality non-inferior to the Phase-1 loss-only winner within a preregistered
  tolerance across seeds;
- no loss of long-context, GLA-memory, loop-depth, routing, or generation behavior;
- no skipped steps, swallowed exceptions, dynamic-graph cache mismatches, or
  selector/reference alignment errors;
- favorable fixed-wall-clock quality after accounting for online selector and
  reference-loss I/O;
- a clear amortization statement for reference training and corpus scoring.

If selector utility wins but systems compaction fails, retain reference-guided
loss-only filtering as a possible quality mechanism and reject Centrifuge-style
acceleration. If systems speed wins but quality fails, reject it. These are
independent decisions in the ablation ledger.

## Hydra and W&B requirements

The eventual implementation must expose every value through Hydra, including:

```yaml
training:
  token_filter:
    enabled: false
    activate_phase: ntp_recovery
    strategy: excess_loss
    drop_rate: 0.0
    reference_artifact: null
    reference_checkpoint_sha256: null
    selector_dtype: fp32
    tie_break: stable_position
    protected_token_ids: []
    compact_backward: false
    compact_scope: []
```

Names are provisional, not authorization to add a superficial wrapper. The
resolved W&B config must also include derived values and implementation details
that affect results: actual transition step, eligible/kept counts, reference
artifact hash, storage dtype/tolerance, selector kernel version, and compacted
operator inventory.

## Promotion outcome

The final ledger decision must be one of:

1. **Reject selector:** reference-guided masking did not beat controls.
2. **Accept selector only:** quality gain exists, but compaction is unsafe or not
   fast enough.
3. **Accept bounded compaction:** specific proven paths accelerate recovery while
   MORPH-specific state paths remain dense.
4. **Accept broader Centrifuge mechanism:** only after every added inter-token or
   recurrent path has its own reference, falsification test, and quality result.

No result from this plan applies to TST. Token filtering remains disabled whenever
TST is active unless a separate future experiment defines and validates a
multi-target selection objective.

## Current verification status

This document is an experimental design, not a result. The paper and upstream
source were inspected, and the relevant MORPH loss, attention, GLA, loop, and
training boundaries were read. No Centrifuge extension was built, no MORPH code
was changed for token filtering, and no training or throughput gate has been run.
