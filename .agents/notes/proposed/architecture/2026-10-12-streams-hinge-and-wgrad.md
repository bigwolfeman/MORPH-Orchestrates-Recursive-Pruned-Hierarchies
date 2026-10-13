# Agent Note: Hinge stream and split weight grads

Status: proposed

Design source: [`2026-10-12-core-pass-megakernel-design.md`](2026-10-12-core-pass-megakernel-design.md), stages S1 and S2.

## Problem

The core pass is host-serial and its GPU work is one stream. Two pieces of work have no data dependence on the rest of the step: the gain hinge's two core applications, and the weight-gradient GEMMs of the linears (the backward of a linear forms `dx` and `dW`; only `dx` is on the critical path). Running them beside the main work can hide their time. Each must give the same bits as today.

## Proposal

Two keys in `base.yaml`, both default off, both bit-exact (code: `morph/model/streams.py`, `morph/model/split_wgrad.py`).

- `model.slot_gain_stream`: the hinge runs on its own CUDA stream inside the graph step. The main stream joins with events before the reduce. Needs `slot_gain_lambda > 0`, CUDA and `slot_gain_no_ckpt`.
- `model.split_wgrad`: the covered linears (ternary MLP linears, the fused attention input projection, `W_up`) run through a custom op pair (`morph::split_linear`, `morph::split_linear_bwd`). The backward forms `dx` on the node stream and `dW = dy^T x` on a side stream. Reads of one weight add into an accumulator in the engine's order, so the fp32 sums keep their order. A final callback joins and binds `p.grad`. The op is armed only inside `train_backward_scope()` (the training forward: eager step, warm-ups, graph_step bodies); every read outside it, such as a probe's `autograd.grad` or a `create_graph=True` call, is the plain path. The ops are opaque to `torch.compile`, so no graph break. Needs `ternary_step_cache` and CUDA.

Held memory: `dy` and `x` of covered reads live until the side stream is done with them. They are freed in groups of 8 reads, two groups behind the newest. Measured reserved memory on the byte-gate config under graph step: off 22.35 GiB, hinge stream 23.05, split 23.38, both 24.03.

## Alternatives considered

- A plain `autograd.Function` with `torch.compiler.disable`. Rejected: the graph break changed which kernels inductor fused and how many dropout seeds the compiled graphs drew, so the step was not the same step (byte gate gave a different hash, 2026-10-12). The custom op pair keeps one graph.
- `record_stream` on every held tensor. Rejected: inside a CUDA graph capture it defers every free to the capture end.
- Hold every `dy` and `x` to the join (first version). It ran out of memory at graph instantiate on the 32 GB card; grouped release fixed that.

## Acceptance criteria

- Byte gate (`nowta_pointer_fast_spdcells`, 45 steps, deterministic, graph step): `FINAL_SHA256 4a73ebf5...bd70d7` with each key on and both on. Met (see report).
- `tests/test_stream_overlap.py` passes with sabotage checks failing the right tests. Met: 23 passed; three sabotages caught.
- Bench: interleaved, 2 repeats, 300 steps. `speed_bf16c`: 215.05 off, 209.42 hinge stream, 200.53 split, 196.80 both (ms/step medians). Not yet a 5 percent K1-K6 check (the outputs are bit-identical, so none is owed for the keys).

## Persistent buffers

Each covered weight owns one fp32 sum buffer and the stream owns one bf16 GEMM scratch per shape, all made at bind, outside any capture. A read does `mm(out=scratch)` and `buf.copy_/add_`, with no allocation. `p.grad` is a view of the buffer (so `model.split_wgrad` needs `grad_accum` 1). Speed config, both keys: graph pool 20.15 GiB against 19.00 off, 0.34 GiB free after the eval (off: 1.54).

## Risks

- Memory (superseded by the persistent buffers below): both keys on did NOT fit the `eval_keep_graphs` recipe on this card. The graph pool grows 19.00 to 20.72 GiB, and the eval then runs out of memory (the key-off eval leaves 1.54 GiB free; hinge stream alone leaves 0.27, split alone 0.15). Smaller release groups moved the pool by 0.17 GiB only.

- A probe that calls `autograd.grad` on the covered parameters themselves misses their grads (they are not in the autograd graph); the scope keeps probes on the plain path, which covers this.
- The compiled-mode exactness of the explicit bf16 cast of `W_up` is shown on one tiny model and the byte gate, not on the 5k recipe.
- Extra memory (up to 1.7 GiB reserved) on a card with about 1 GB slack at the edge of the recipe.
- Idle gap between graph replays (3.8-5 ms per step): not resolved. See report; the host blocks inside the first graph replay of a step, and cond-branch flags, instantiation flags and microbenchmarks did not explain it.
