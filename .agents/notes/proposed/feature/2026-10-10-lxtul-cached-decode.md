# Agent Note: LXTUL cached and graphed decode, and the generation forward

Status: proposed

## Problem

TUL's intent is "think once, decode cheap": the loop runs once per span on the slot cells,
and a token pays only the prelude and the coda. For `lxtul_pointer` the analytic decode cost
is about 2.3x fewer FLOPs per token than plain MORPH (2.6x fewer block applications;
`/home/wolfe/morph-scratch/research/lxtul-decode-savings.md`). Measured on 2026-10-07 it was
inverted: eager LXTUL + pointer decoded at 7.1 tok/s against plain MORPH's 86 tok/s
(`lab/experiments/results/2026-10-07-lxtul-pointer-ditto/gen/gen.log`). Three causes:

1. The KV-cached decoders (`morph/inference/tul_generate_cached.py`,
   `tul_generate_graphed.py`) refused the fan, the register and the pointer, so LXTUL could
   only use the eager recompute-per-step generator.
2. The eager generator's forward ran every loss head on every step: the span decoder (a
   2-layer decoder and a full-vocab CE over every slot; the 2026-10-07 coverage run OOMed
   inside it), the fan's diversity terms and per-pass readings.
3. A latent hole (tracker C1f): `tul_cell_norm` was not refused, so a one-cell rollout
   model with `slot_cell_pass_norm` decoded the un-normed loop without an error.

## Proposal

Built in this change (tracker items C1f, C1e, C1c):

- **C1f.** Both cached decoders apply the per-pass cell norm (`_cell_norm_apply`, after
  the code term, as `_tul_core` does). Test first: `test_cell_norm_matches_eager` failed
  (tokens diverge at index 3) before the fix and passes after.
- **C1e. The generation forward.** `MORPHTransformer.forward(..., generate=True)` (eval,
  label-free, slot-loop models; `tul_generate_supported` is the one predicate) skips the
  span decoder, the parallel span head, NextLat, the MUX, the fan's diversity terms and
  per-pass readings, SIGReg and the row contrast at their call sites. The logits path is
  untouched. It is an explicit argument, not a `labels is None` / `self.training` gate,
  because the val pass and `lab/divergence/core_depth_sweep.py` read those heads from a
  label-free or eval forward. `generate_tul` and `generate_tul_batch` pass it.
- **C1c. LXTUL in the cached and graphed decoders.** One state machine serves the
  K-rollout model and LXTUL (K = 1, M = `slot_cells` looped cells):
  the M cells' prelude at the slot's first M prefix positions; the register's seed term
  (the model's own `TULSlotRegister` over the span's cached token rows); the loop cache on
  the compact cell axis `s*M + i` with the cell relation (own slot all cells, later
  siblings included; `loop_relation`); after each pass the cell norm, the router's pick
  and, while the depth continues, `reset_to_winner`; the winner-only write; the coda of
  all `prefix_k` cells (losers stay zero keys); the one-rollout read; and the pointer with
  a row-wide cache of every earlier token's key and its follower. The ungraded fan (no
  router) and the plain one-cell loop come out of the same code and are tested too.
- **The cell count M is a decoder argument** (`cells=`, `loop_cell_count`). Every buffer,
  mask and loop is sized by M, but a value other than the model's own is REFUSED: the
  register's per-cell queries, `W_prefix[i]` and the router's bias are indexed by cell,
  so a forward at another M is undefined until tracker item B1 builds index-free parts.

Refused by name (`_check_supported`): the fan under `code_enum_k` (LX-Fan), a register
without the fan, `fan_mix` other than `all`, `fan_lsel_read: all`, fan lineage, history
streams, trigger-every-pass, the head grader, the fan router and factor fan, pseudo-snap,
the LX exploration knobs, the pointer's cell key and coverage, and `cells != M`.

## Measured (2026-10-10)

Correctness, `lxtul-pointer-rerun/step_5000.pt`, 5090, fp32, greedy, row 0 (prompt 128,
256 new tokens), `lab/perf/bench_lxtul_decode.py --check`:

| comparison | tokens | worst row difference (log-probs) |
|---|---|---|
| eager, generation forward vs old forward | 256/256 equal | 1.0e-6 (not bit-equal) |
| eager vs eager, same forward, run twice | 256/256 equal | 9.7e-7 (GPU atomics: not bit-reproducible) |
| cached vs eager | 256/256 equal | 4.1e-3 |
| graphed vs eager | 256/256 equal | 4.1e-3 |

The 4.1e-3 is not the fused Triton kernels: with `MORPH_FORCE_EAGER=1` the cached and
graphed differences are the same 4.1e-3. Open suspect: cuDNN TF32, ON by default in
torch, runs the CCA `conv1d` of an fp32 model at TF32 precision on different shapes in the
two paths (whole row against a short window). The `--no_tf32` run that tests it is queued.

On CPU the generation forward's logits are bit-equal to the old forward's
(`tests/test_tul_generate_mode.py`). The tolerance of the tiny-model tests comes from fp64:
with the model and both generators fp64 end to end the worst row difference is 3e-14 on
every fixture (`test_fp64_reference_is_exact`), so the fp32 gaps (3.3e-6 without the
pointer, 2.2e-5 with it) are summation order; TOL is 1e-4. Building that reference found
a PyTorch behaviour: CPU SDPA reads an fp32 additive mask under fp64 queries wrongly once
there are more than 16 keys (torch 2.11); the decoders now cast the bias to the query's
dtype (a no-op in fp32).

Speed (`lab/perf/bench_lxtul_decode.py`; step_5000 and `plain-panel-norm-match/step_5000`,
5090, fp32, greedy, 8 real validation prompts of 128 tokens, 256 new tokens; tok/s counts
generated tokens over a wall that includes the prompt; busy = device kernel time / wall of
a 32-token run of the same arm):

| arm | batch 1 tok/s (busy) | batch 8 tok/s (busy) |
|---|---|---|
| LXTUL eager, old forward (C1e off) | 6.2 (0.93) | not measured (the process died, no traceback) |
| LXTUL eager, generation forward | 12.2 (0.92) | 16.2 (0.99) |
| LXTUL cached | 82.9 (0.24) | 65.1 (0.21), 8 rows one after another |
| LXTUL graphed | 294.8 (0.98) | 262.2 (0.99), 8 rows one after another |
| plain eager (`generate_plain_batch`) | 16.6 (0.59) | 86.4 (0.95) |
| plain cached (`kv_cache.decode_step`) | 11.2 (0.49) | 87.5 (0.59) |

Per event, graphed LXTUL (CUDA-event replay timing; kernel count of one eager execution
of the same step): the token step is 1.94 ms and 835 kernels; the slot step is 11.7 ms and
4056 kernels, 6.0x a token step against an analytic 5.5x (4 prelude + 36 loop + 4 coda
block applications on 4 rows, against 8). Row 0 cut 8 spans in its 256 new tokens (a
greedy loop held at the 32-token cap) and 4 in the prompt: 384 x 1.94 + 12 x 11.7 = 885
ms against 868 ms measured. At the trained mean span (19.7 tokens) the steady state is
1.94 + 11.7 / 19.7 = 2.53 ms per token, about 395 tok/s. The token step reads about
550 MB of fp32 weights (8 blocks + the tied head), a 0.31 ms floor at 1.79 TB/s, so it
runs at about 6x that floor; the gap is the kernel count (835 kernels, about 2.3 us each).
Plain MORPH has no graphed number in this table yet (`StaticDecodeEngine` arm queued), so
the LXTUL-versus-plain ratio at equal decoder quality is not measured.

## Alternatives considered

- **Gate the loss heads on `labels is None`** (the research file's first suggestion).
  Rejected: the val pass and the offline depth sweep call the forward label-free and read
  the span decoder's K1-K6; a labels gate would silently remove their instrument.
- **A separate LXTUL decoder class** beside the K-rollout one. Rejected: the slot step,
  the prelude, the coda and the attention sites are the same; two copies would drift. The
  two families differ in four places (the rollout axis and its code term, the Bayes read,
  the register and router, the pointer), each a branch on a constructor-time constant.
- **Variable M now** (accept any `cells`). Rejected: no forward exists at another M, so
  any number it produced would be invented. The refusal names B1.
- **One-hot via `F.one_hot` for the winner route** (what the model uses). Replaced by an
  equality against `arange(M)`: the same exact 0/1 and no value check that could sync the
  host inside a CUDA graph.

## Acceptance criteria

- CPU tiny-model parity: `tests/test_tul_generate_lxtul.py` (cached and graphed against
  eager on LXTUL + pointer, LXTUL without the pointer, the plain one-cell loop and the
  ungraded fan; both emit sources; greedy and sampled; past the window; fp64 exact) and
  every sabotage there fails the comparison.
- GPU: `test_cuda_graphs_match_eager` passes; step_5000 tokens equal for 256 greedy tokens.
- The bench table filed with tok/s at batch 1 and 8 against the analytic expectation.
- Open: a batched graphed LXTUL decoder (rows insert slots at different steps); the
  graphed decoder's remaining gap to the weight-bandwidth floor.

## Risks

- Greedy near-ties: on the real checkpoint the cached rows differ from eager by up to
  4.1e-3 nats, so a long or sampled decode can flip a token where eager's top two are
  closer than that. Equal tokens over 256 greedy steps on one row is not a guarantee.
- The router's argmax is a discrete decision inside the loop. A near-tie between two
  cells' scores can pick a different winner under rounding; nothing here bounds that.
- The pointer cache grows with the row (capacity: the RoPE cache, 4096 positions).
