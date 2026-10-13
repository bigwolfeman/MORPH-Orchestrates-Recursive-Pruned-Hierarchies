# Agent Note: Phase 2 tracker (hinge pass, variable cell count, speed and decode)

Status: proposed

## Problem

Phase 1 (2026-10-08) made LXTUL training 2.3x faster (`lxtul_pointer_fast`, 26.7k tok/s, quality
inside the eager spread; merge 74a334b6). Wolfe (2026-10-10) set phase 2 to three open items, plus
one he raised: "I feel like we still haven't realized the actual TUL savings from decoding the
latent." His own low-hanging list is out of scope until he brings it. Work spans several notes and
agents, so one checklist tracks it. Each item links to the note that owns its detail; this file
holds only status.

## Proposal

Status keys: [ ] open, [~] in progress, [x] done (with the commit or filing), [-] dropped (with why).

### Targets (Wolfe 2026-10-10)

- **Target restated (Wolfe 2026-10-12):** 9.1 TFLOP/step for everything EXCEPT the training-only span decoder
  (measured today: 15.25 total, decoder 3.69, so the rest is 11.56). Megakernel design pass approved (Opus).
  `eval_diag_every: 4` approved for now.

- **Recipe switch 2026-10-11:** speed work now targets `nowta_pointer_fast_spdcells` (the fan recipe,
  [note](../../implemented/architecture/2026-10-11-ungraded-fan-set-reader.md)). The fan has no EMA twin and no
  router; it keeps the 4-cell loop, the hinge and the span decoder (now with a cross-attention onto the cells).
- **Fan ledger, measured 2026-10-11** (master 95a2c6e9, `nowta_pointer_fast_spdcells`, one eager step with the
  hinge active, `/home/wolfe/morph-scratch/perf/flame2/out/flops.{json,folded}`): **15.25 TFLOP/step**, not
  the 21 I estimated. Span decoder 3.69 (head 1.37 on the labelled rows; blocks 2.3 on all 12288 rows, 45 %
  of them past the span end); coda 2.34; LM head 2.34 (runs on 1280 rows, 1024 are tokens); prelude 2.12;
  loop 3.65 + glue 0.14 (`slot_compact` already shrinks each pass to its live slots: pass 0 0.61 -> pass 7
  0.22; recompute 0); hinge 0.91 on an active step (C4 skips its backward on inactive steps). At 29.2k tok/s
  that is 72.6 TFLOPS, 33 % of the 5090's dense bf16 peak. The deploy-model parts sum to 10.65: the analytic
  9.1 leaves out attention scores, pads and slot rows in the head, so 9.1 is below the measured floor of
  this model; about 9.9 is reachable without a model change.
- **Plan to the target (2026-10-11):** [ ] T1 packed span decoder (rows = real span tokens, fixed capacity
  B x L; exact in math) about -1.0. [ ] T2 C2 head on token rows -0.47 (built, needs the 5k check).
  [ ] T3 `tul.spandec_max_tokens: 8` (behaviour change; the lens says the decoders read about 2 tokens)
  about -1.6, no build. [ ] T4 hinge on a row subsample about -0.35. [ ] T5 pad slots out of prelude, coda,
  head about -0.3. [ ] T6 GPU-time ledger (time, not FLOPs: 33 % MFU is the larger lever). Quality: one
  package run per class against the locked recipe (CE@6 3.8952, K1-K6 +0.0054).
  [x] T6 GPU-time ledger 2026-10-11 (`/home/wolfe/morph-scratch/perf/flame2/LEDGER.md`, flame PNGs in
  `flame2/png/`): graph replay 253 ms busy under the profiler (trainer step about 210 ms), 14750 kernels, 58
  TFLOPS. Ms over ideal: core passes 54.8, coda 22.0, prelude 21.1, span decoder 16.9, core glue 15.9, hinge
  12.5, LM head 8.4. The HC Cayley Triton kernels are the top non-GEMM cost in every block region (about 51 ms
  in all); fp32 elementwise dominates core glue and the decoder; late core passes run GEMMs at 80 TFLOPS
  (small compacted rows). [~] T7 loop glue fusion (sonnet, perf/loop-glue). [~] T8 HC kernel speed (sonnet,
  perf/hc-kernels). [~] T1 (sonnet, perf/spandec-packed).
  **2026-10-12, merged master 0c01b175:** T1 + T2 + T7 + T8 as one package (`nowta_pointer_fast_spdcells_speed`):
  +8.3 % paired trainer rate (30.1k -> 32.6k harmonic mean), K1-K6 +0.0053 (held), CE@6 3.8985 (+0.0010 over
  the prereg line, one run; filed `lab/experiments/failures/2026-10-12-fan-speed-package.md`). Kept under the 5 %
  loop-contribution rule; the `_speed` config is the base for further speed work. [x] T1 [x] T2 [x] T7 [x] T8.
  [x] T3 REJECTED: the 8-token decoder is +17.8 % faster but CE@6 3.9213, worse than no decoder; cap stays 32.
  **Round 2 (2026-10-12, sonnet builders, branches off 7b792ccb):** [~] T4 hinge on k of B rows + the standard
  steady-state bench `lab/perf/bench_step.py` (perf/hinge-rows; hinge is 17.5 ms). [~] T9 bf16-stored HC carrier with
  fp32 math and an fp32 carrier for the hinge's applications (perf/bf16-carrier-v2; HC Triton 56 ms, bandwidth bound).
  [x] T10 FP8 e4m3 forward GEMMs, PARKED (perf/fp8-fwd 61b4fa45): 2x at 6144 rows in isolation, but the ternary MLP
  forward is ~1.5 of 15.3 TFLOP, so the step moves 139.0 -> 136.7 ms (1.8 %). Not in the package. FP8 dgrad would add
  about as much again. Note: steady-state step is ~139 ms (44k tok/s); the 5k trainer mean (32.6k) includes eval every 250.
  Then one 5k package run against the `_speed` config. Not started: T5 pad slots, small-row core passes (megakernel; ask Wolfe).
  **Wall clock (2026-10-12, `lab/perf/bench_step.py`, merged master 3dd77c05):** the trainer's tok/s before dea1911d timed
  the HOST (about 1.5x high on this GPU-bound step); it is now the synced wall rate. Locked recipe 228.7 ms/step (26.9k
  tok/s), speed base 213.5 (28.8k), + FP8 209.9, + hinge rows 207.5, all three round-2 levers 186.9 (32.9k, 1.148x the
  base). Round-2 package FAILED at 5k (`lab/experiments/failures/2026-10-12-fan-speed-round-2.md`): CE@6 3.9117,
  K1-K6 +0.0035 (-34 %), stability held. [~] Lever split running (one 5k per lever, prereg `planned/2026-10-12-fan-speed-lever-split.md`).
  [x] Lever split filed (`failures/2026-10-12-fan-speed-lever-split.md`): bf16 carrier holds (3.8996 / +0.0052), FP8 forward
  costs the loop (3.9057 / +0.0024, rejected), hinge rows 3.8975 / +0.0044 (off; one more run before use). Speed base is now
  `nowta_pointer_fast_spdcells_speed_bf16c`: 198.9 ms/step wall. bf16-carrier models run outside autocast (master 24b5077c;
  decorator costs 0: 198.7 ms). Eval is ~26 % of a 5k run's wall (20 evals x ~19 s: ~13 s eval + 6 s graph re-record
  after GraphStep.released()); [x] eval cost (master 30a246fb): `training.eval_keep_graphs` (exact; ON in the bf16c speed base)
  23.2 -> 16.3 s per eval; `training.eval_diag_every: k` runs the ablation/probe block (75 % of eval) on every k-th eval,
  5.0 s on the skipped ones; k > 1 forks the RNG around eval (traces identical for k 2 vs 3, verified). Left: the
  attention-lift probe reads 0 calls on this recipe (2.8 s of every eval for one constant); base-forward sharing (~5 s).
  Fixed on the way: test_tul_lxfan pins moved by lever E (one embedding grad, 3.9e-8 rel), re-pinned 8d489734.
  **Core-pass plan (Opus design, c85cd646, `2026-10-12-core-pass-megakernel-design.md`):** fresh profile 202.4 ms/step,
  core passes 62.5 ms (floor ~17-23). No megakernel yet; stages S1-S7 ~-25 to -30 ms. [x] S1 hinge stream + S2 wgrad
  stream MERGED (9a4bcb3a, keys default off; bench off 215.05 -> both 196.80 ms, 1.093x; graph pool +1.15 GiB,
  0.34 GiB free after eval). [x] both-keys byte gate after the buffer change (4a73ebf5, match) + quiet-desktop bench
  off 199.26 -> both 177.22 ms (1.124x): ON in speed_bf16c. [ ] idle gap (3.8-5.4 ms, cause not found). [x] S4 HC exit+entry fusion: exact but 0.975x
  (213.8 -> 219.3 ms), parked on perf/hc-exit-entry 1c220f33 for a re-bench at larger width (Wolfe 2026-10-12).
  Bench drift (2026-10-12 evening): the bf16c off arm reads 213-215 ms vs 198.9 this morning. Same code (24b5077c
  tree) reads 212.9 now, so it is the environment (desktop GPU/CPU load, a game), not the code; core pinning (213.1 vs
  214.5) and the eval keys (1.0007) do not move it. Interleaved ratios stand; take absolute ms only on a quiet desktop.
  [x] bf16 carrier IN the speed base: matched controls CE@6 +0.0005, K1-K6 105 % (successes/2026-10-12-bf16-carrier-matched-controls.md).

- **Training: 9.1 TFLOP per step** for `lxtul_pointer` at batch 6 x 1024 (today 21.67 TFLOP,
  2026-10-08 flame ledger). 9.1 is the analytic LXTUL step: three times the decode forward,
  i.e. the loop once per span plus prelude and coda per token, with no train-only extras
  (`/home/wolfe/morph-scratch/research/lxtul-decode-savings.md`). Where the other 12.6 TFLOP
  sits today: span decoder 5.68 (26 %), gain hinge 1.64 (7.6 %), EMA twin 0.72 (3.3 %), the
  4-cell loop on all 8 passes (core 5.87; 1 cell would be about 1.47; up to 2.33 is discarded
  pass work), slot rows in the LM head about 0.34, and 16.9 % of slots are pads.
- **Decode: at least plain MORPH's cached speed x 2.3** per token (the analytic ratio for
  4 cells; 3.3x at 1 cell, 3.9x with zero span cost). Today eager LXTUL + pointer decodes at
  7.1 tok/s against plain MORPH's 86 (eager, same run): 12x slower, not 2.3x faster.

### A. The gain hinge's fixed pass (Wolfe's decision)

The hinge draws its pass from the CPU generator and restores it, so it regularises ONE pass per
run, chosen by the seed (`transformer.py` `_t_gain`; filing
`lab/experiments/mixed/2026-10-08-fast2-package-paired.md`). Every winner number trained that way.

- [x] A1 Read the drawn pass directly (2026-10-10, eager `lxtul_pointer`, 50 steps, a wrapper
      logging every `torch.randint` drawn inside `_tul_core`): 54 of 54 draws are pass 6 on
      seed 1 (the winner's seed) and 54 of 54 are pass 3 on seed 2 (`n_grad_iters` 8). The pass
      is fixed per run and differs between seeds, so two seeds of one recipe regularise
      different passes. Scripts and logs: `/home/wolfe/morph-scratch/perf/hingepass/`.
- [x] A2 Wolfe decided 2026-10-10: a fresh pass per step (the 2026-09-04 intent; 34d94a04 put
      the CPU generator back after the draw, which froze the pass). Rejected: one pass per run
      (the constrained region is a seed accident); every pass per step (8x the hinge, about
      +200 ms per step at 29.5 ms per pass).
- [x] A3 Build it: a per-step draw that keeps `graph_step` capturable (8 captured graphs sharing
      one pool, or a device-side index into the stored per-pass detached states; the hard part
      is replaying pass t's dropout masks). Byte gate: with the draw forced to pass 6 it must
      equal today's step. Then a paired check (3 continuations from the rerun step_2500) against
      the fixed pass on CE@6 and K1-K6; if it holds, it goes into `lxtul_pointer_fast`.
      2026-10-10: [~] built on perf/hinge-draw e19234c2 (rebased on 7b028dab). Private CPU
      generator, one uniform per step; `model.slot_gain_pass_force` (-1 fresh, k pins); one
      graph per (kind, pass), 16 in one pool; kept gradients in one shared buffer set. CPU
      tests pass (gain_pass_draw 14, graph_safe 13, graph_step 8 + 9 CUDA skipped). NOT run:
      byte gates at forced pass 6, CUDA graph-step tests, speed, memory, recording time
      (queued in go_queue.sh). Prereg `lab/experiments/planned/2026-10-10-hinge-fresh-pass-paired.md`
      is in the commit; the paired chain runs after the gates pass. Open: `model.core_gain_lambda`
      (`_core_region`) has the same frozen draw; no decision covers it.
      GPU gates 2026-10-10 on be624816 (e19234c2 + warm-up slot_id remap: the 8 random warm-up
      rows hit slot_id, 4 never had): forced pass 6 byte-identical to 6a33958e on fast AND eager
      (45 steps, loss/grad-norm/7 gain terms in hex, FINAL_SHA256 equal); fresh draw moves (50
      steps, all passes 0-7, seed 2 a different sequence, eager seed 1 = fast seed 1); CUDA
      graph-step tests 36 passed; sabotage (replay pass 0's graph) fails 4/4. 16 graphs record
      in 5.1 s. Bench 420 steps: 233.9 / 235.6 ms/step vs base 230.0 / 230.1 (+2.0 %; likely the
      hinge now reads earlier passes, which carry more rows under slot_compact). Paired check
      SUCCESS (perf/hinge-draw 55ecb33f, `lab/experiments/successes/2026-10-10-hinge-fresh-pass-paired.md`):
      CE@6 4.0482 / 4.0480 / 4.0487 vs fixed mean 4.0473; K1-K6 mean +0.0121 vs +0.0104; every
      pass drawn (chi2 4.38). Not merged to master yet.
- [ ] A4 Interaction with C4 (skipping the inactive hinge): decide A first.

### B. Variable cell count (owner note: [`2026-10-05-lxtul-variable-cell-count.md`](../architecture/2026-10-05-lxtul-variable-cell-count.md))

- [ ] B1 Index-free cell seeding, router bias and write (the note's design blockers), M a
      per-forward argument; test that M = 4 reproduces the fixed-4 forward with matched seeds.
- [ ] B2 Train-time draw of M that keeps `graph_step`: one graph per M value or a masked M_max
      layout; memory at M_max on the 5090.
- [ ] B3 Prereg + two 5k arms (fixed 4, Poisson 4), both on `lxtul_pointer_fast`.
- [ ] B4 Inference sweep M in {1, 2, 4, 8, 16, 32, 64, 128}: K1-K6, CE, router pick vs oracle,
      cell diversity, tok/s. Filed.

### C. Speed and decode (owner note: [`2026-10-08-lxtul-training-speed-plan.md`](../architecture/2026-10-08-lxtul-training-speed-plan.md))

- [~] C1 Realise TUL's decode saving ("think once, decode cheap"): the loop runs once per span,
      so per-token decode cost should be the prelude + coda path, far below a model that loops
      every token. Today the KV-cached generators refuse the fan (`tul_generate_cached.py`
      `_check_supported`: `tul.fan_k > 0`), so LXTUL decodes with the eager recompute-per-step
      generator. Steps:
  - [x] C1a Accounting (2026-10-10, research agent; `/home/wolfe/morph-scratch/research/lxtul-decode-savings.md`).
        Decode, analytic: plain 44 block applications per token (about 1139 MFLOP); LXTUL as
        trained 16.95 (about 496 MFLOP, 2.3x fewer); 1 cell at inference 3.3x fewer; zero span
        cost 3.9x. Measured eager decode is INVERTED: LXTUL + pointer 7.1 tok/s vs plain MORPH
        86 tok/s, same run (`lab/experiments/results/2026-10-07-lxtul-pointer-ditto/gen/gen.log`).
        Training: LXTUL and plain MORPH are at about equal FLOPs per token because the
        train-only terms are 37.5 % (span decoder 26 %, hinge 7.6 %, twin 3.3 %); an ideal LXTUL
        step is about 9.1 TFLOP vs 21.67 today. Parcae evidence: bare strict loop 37.2k tok/s
        vs plain 21.8k; Parcae LXTUL without the span decoder 28.2k vs 24.4k at CE@6 +0.006.
  - [~] C1b..C1f: agent `decode`, branch `perf/lxtul-decode` (worktree /home/wolfe/morph-wt-decode),
        launched 2026-10-10 02:10; vlt thread `lxtul-decode`.
  - [~] C1b Decode bench (2026-10-10, `lab/perf/bench_lxtul_decode.py`, fp32 greedy, prompt 128
        + 256 new, batch 1): LXTUL + pointer graphed 294.8 tok/s, cached 82.9, eager 12.2
        (generation forward) / 6.2 (old forward); plain MORPH eager 16.6, `kv_cache` 11.2.
        Graphed LXTUL token step 1.94 ms (835 kernels, about 6x the 0.31 ms weight-read
        floor), slot step 11.7 ms. 2026-10-10: graphed plain MORPH measured (perf/lxtul-decode
        38675819, `StaticDecodeEngine` fixed for the current geometry): B=1 / B=8 tok/s plain
        graphed exact_weights 302.6 / 1137.5 (bf16 copies 319.7 / 1318.5, flips 2/10 streams on a
        0.0062-nat tie), LXTUL graphed same run 295.3 / 260.6. RATIO LXTUL/plain = 0.98x at B=1,
        0.23x at B=8, against the 2.3x target: the decode saving is NOT realized. Cause: the
        plain engine is fused Triton (617 kernels for 44 layer passes, about 50 us per layer pass,
        GPU-bound); the LXTUL graphed decoder is generic PyTorch (835 kernels for 8 token-path
        layer passes, about 240 us each) and holds one row. Open: an LXTUL decode on the fused
        engine kernels with real batching; Parcae arm.
        2026-10-10 later: `LxFusedDecoder` on perf/lxtul-fused-decode d8921dfd (design f3b32a63,
        kernels lx_attn + decode_front kv_row f1441571, Z3 bounds proofs). Same invocation as
        plain graphed exact: 951.2 / 2684.5 tok/s at B=1 / B=8 = 3.10x / 2.34x (bf16 copies 3.46x
        / 2.18x). Greedy decode loops with no boundary, so few slots: teacher-forced real val
        text gives 2.96x / 2.44x per step. Token step 0.589 ms / 191 kernels, slot step 2.40 ms /
        687 kernels (bucket 1). Tokens equal graphed and eager for 256 tokens on prompts 0 and 1;
        B = 4/8/16 rows equal B=1. TARGET MET at B=1 and B=8 (B=8 margin thin). Open: prompts 2-7
        against the references, sampling, B=16 vs plain, slot-step fusion (~55 us per layer).
        Fusion (perf/lx-decode-megakernel 6952a5e8, merged b792581d; note
        `.agents/notes/proposed/feature/2026-10-10-lx-decode-megakernel.md`): 8 kernels per layer,
        latent cache, glue kernels, int6-codes LM head. Median of 3, same invocation: 1338 / 4357 /
        5251 tok/s vs plain 268 / 999 / 1212 = 4.99x / 4.36x / 4.33x at B = 1 / 8 / 16; repeat on
        final kernels 5.12x / 4.22x / 4.32x (plain 309 / 1149 / 1387). Token step 0.369 ms / 74
        kernels, slot bucket 1 1.56 ms / 438. Open: host gap (busy 0.86-0.91), slot seed/register
        glue (~100 us), logits row stride, persistent megakernel (<= 0.3-0.4 ms per slot step).
        FINDING: the Triton `fused_cca_prologue` (training forward too) is ~3e-3 relative off fp64
        for q/k; `cca_prologue_reference` is 1e-7. Likely source of the eager-vs-cached logit gaps.
        RESOLVED in the kernel (fix/cca-prologue-precision 2e434a60, not merged): the forward
        rounded to bf16 after every op while the backward recomputed in fp32. Fixed to fp32 with
        one rounding at the store (bf16 autocast q/k 3.9e-3 -> 1.66e-3 = the floor; time within
        1 %). It was the whole decode gap: 4.1e-3 -> 2.4e-5. Training fwd+bwd moves the grad 1.5e-2
        relative. Paired check SUCCESS under its rule and MERGED
        (`lab/experiments/successes/2026-10-10-cca-prologue-rounds-once-paired.md`): CE@6 +0.0011,
        K1-K6 -0.0017, val +0.0007, all leaning against the fix at the resolution (run 3 missed,
        its rerun held). Open: from-scratch 5k pair to tell a resume transient from a real cost.
        Side finding: on lxtul-pointer-rerun step_5000 (trained with the hinge fixed on pass 6),
        pinning the hinge on passes 0-2 reads gain 1.8-2.0 (gain_reg_weighted 87-126, loss 95):
        the passes the old hinge never read drifted expansive. Resuming a long fixed-pass
        checkpoint under the fresh draw would start with a large hinge term.
  - [x] C1c Cached + graphed LXTUL decode: `perf/lxtul-decode` 46c2fa0d (not merged). On
        step_5000 row 0, 256 greedy tokens equal eager (row diff 4.1e-3, not explained. 2026-10-10:
        NOT TF32 (`--no_tf32` 4.15e-3) and NOT run noise (deterministic mode: eager repeat bit-equal,
        gap still 4.15e-3): a fixed numeric path difference in the cached path, tokens equal); fp64 agrees to 3e-14 on CPU fixtures. M other
        than the trained 4 is refused until B1. Original item: bit-equal tokens to `generate_tul`. Refused today at
        `tul_generate_cached.py:85-159` (fan_k, slot_cells, pointer_heads, code_enum_k > 1,
        tul_fan / tul_register). Minimal set: 4 cells through the prelude; register seed with a
        per-span token K/V cache; 4-cell loop caches with sibling reads; per-pass cell RMSNorm;
        router pick + reset_to_winner; routed winner write; coda on the cells (all 4 stay as
        keys); one-rollout read; pointer head with a cache over all earlier tokens. Cell count
        M as a decoder argument from the start (item B).
  - [x] C1d FAILED 2026-10-10 (`lab/experiments/failures/2026-10-10-lxtul-fast-no-spandec-5k.md`):
        34.7k tok/s and 26.4 min (speed held), but CE@6 3.9927 vs 3.92, K1-K6 +0.0020 vs +0.0098,
        val 3.9806. The span decoder stays; its cost is a speed target, not a cut.
        Training-side, Wolfe's go 2026-10-10: a no-span-decoder arm on MORPH. Prereg, config
        `lxtul_pointer_fast_nospandec`. The span
        decoder is 5.68 TFLOP (26 %), runs on 2 rows per token, and half of those rows carry no
        label. Parcae evidence (one 5k seed each): Parcae LXTUL 24.4k tok/s with it, 28.2k
        without, CE@6 +0.006. Plan: prereg, then `lxtul_pointer_fast` with the span decoder off
        at 5k, read against the two fast 5k runs (CE@6 3.9248 / 3.9229, K1-K6 +0.0104 /
        +0.0091). Next on the same axis: the 1-cell vs 4-cell loop cost (core 1.47 vs 5.87
        TFLOP) meets item B.
  - [x] C1e (46c2fa0d: `forward(generate=True)`) The eager generator runs the training-only span decoder on every step
        (`transformer.py` ~14526, no gate; OOM trace
        `lab/experiments/results/2026-10-07-lxtul-pointer-coverage/gen/gen_oom.log`). Add an
        explicit generation forward that skips every loss head; do not gate on `labels` or
        `self.training` (the val pass and the offline sweep read the span decoder).
  - [x] C1f (46c2fa0d, test `test_cell_norm_matches_eager`) Hole in the existing cached decoder: `tul_cell_norm` is not refused, so a one-cell
        multi-rollout model with the cell norm would decode a different function silently.
        Refuse or implement first.
- [ ] C2 LM-head slot rows (Lemma 3: needs `pointer_cell_key: false`; the coda keeps K/V at slot
      rows). About 0.34 TF per step. 2026-10-10, branch perf/train-speed-free, committed:
      `model.lmhead_row_cap` (OFF in lxtul_pointer_fast until a 5k quality check). Not bit-exact (reduction order):
      step 0 loss identical, grad norm 6.7e-8 relative, then the 45-step trajectories
      decorrelate (max 0.055 nats). About -0.8 ms per step. Note
      `.agents/notes/proposed/architecture/2026-10-10-exact-training-step-levers.md`.
- [ ] C3 Drop the logits recompute GEMM in the token head (keep bf16 logits, about +0.76 GB).
      2026-10-10, same branch, committed: byte-identical gate, -4.1 ms per step. Same
      note; also E (one int-N quantisation and one tied head per forward, -4.0 ms) and Q
      (one-pass Triton quantiser, -6.3 ms), both byte-identical. All levers: 245.6 -> 233.5
      ms per step in the bench window.
- [ ] C4 Skip the hinge backward on inactive steps (71 % of steps; needs a CUDA conditional node
      under `graph_step`). After A. 2026-10-10, same branch, committed: built
      (graph_cond.py), byte-identical gate; -8.2 ms on an inactive step, +0.9 ms on an
      active one. Note
      `.agents/notes/proposed/architecture/2026-10-10-hinge-branch-conditional-node.md`.
      The A4 decision is still open.
      2026-10-11 RISK, measured: with all levers merged, `nowta_pointer_fast_spdbest` (a span
      decoder over all 4 cells) ran out of memory at the FIRST validation (step 200; 22 GB in
      CUDA-graph private pools, 596 MiB free). The same-size each-cell arm passed on the
      pre-lever tree. Not yet measured: which lever takes the headroom (C4's second body is the
      suspect), and whether the plain fast recipe still has validation headroom on a full desktop.
      RESOLVED 2026-10-11 (branch fix/eval-mem): not a lever's size. `GraphStep.released()` left a
      loop variable bound to the last `_Captured` across its `yield`, which kept that capture's
      graphs and the whole 18.5 GiB private pool alive through eval. Branch closing now runs in a
      helper. Byte gate identical (45 steps); test_graph_step 20 passed; 260-step runs of
      nowta_pointer_fast_spdcells / _spdbest / lxtul_pointer_fast pass validation (~24.9 GiB free).
- [ ] C5 The 4-stream HC carrier: 1 stream was worth 60 ms per step at FAST. An architecture
      change, so it needs a quality arm, not only a speed bench.

### D. Literature: JEPA-Anything (arXiv 2609.20800)

- [x] D1 Read (2026-10-10, research agent; write-up `/home/wolfe/morph-scratch/research/jepa-anything.md`).
      MORPH built from this paper twice: `tul.fan_opf` (factor fan, weight 1 and 10), both filed
      failures (`lab/experiments/failures/2026-09-30-fan-opf-and-routers.md`,
      `2026-09-30-factor-fan-10x-latent.md`): only one factor of four learned (R^2 +0.13 / +0.17,
      others below 0), weight 10 cost +0.131 CE, both floor terms read 0. Our arm differed from
      the paper: each predictor read its own fan cell (not one shared context), no recombination.
      The winner already carries the paper's EMA target and variance floor. The only untested
      placement is a factored version of the rank-only latent head (shared input, ranking on
      the predictable factors); its ceiling is the best-of-4 oracle (-0.0207 nats) and the
      teacher is not better than the router today. Verdict: optional later ablation, as Wolfe
      said; a cheap offline pick-agreement probe gates it.
- [ ] D2 DiscoLoop (arXiv 2607.00341, read 2026-10-10). Between loop passes it adds
      `alpha * RMSNorm(sum_v softmax(W h / tau)_v W[v])` (tied W, top-128 approximation, sigmoid
      gate, last pass ungated). Multi-hop toy: OOD 8.3 % -> about 100 %. 440M / 20B-token
      pretrain: behind the plain loop until about 13B tokens, then about 0.01 loss ahead,
      benchmark mean 50.5 vs 49.3. Not built in MORPH: `tul.pseudo_k` (snap) writes a span-token
      mixture AFTER the last pass, not between passes. Open questions before an arm: the slot
      cell has no trained token readout (`emit_weight: 0`), so its logit lens may give the
      slot-shared unigram mean (the carrier constant again); the soft bridge already failed us
      in DiffusionBlocks (norm 0.27, hard argmax fixed it); cost is about 4 block applications
      per cell per pass (one 1024 x 49169 GEMM), against the 9.1 TFLOP target. Its 5k signal
      would be inside our noise (their gain shows after 13B tokens; our 5k is 31M). A fair test
      is a multi-hop task on the Parcae testbed, not a 5k web-text arm.
  - Wolfe 2026-10-10: test it as an ALTERNATIVE to LXTUL, both ways: DiscoTUL (the strict slot
    loop, one cell, re-embedding between passes) and LX-DiscoTUL (LXTUL with it on every cell).
    Build now; run after the optimization work (A, C).
  - [~] D2a Built on `feat/discotul` cf65bd7a (not merged): `tul.disco` (`tul_disco.py`,
    `_disco_term`), term `alpha * RMSNorm(Phi(h))` at the input of pass t >= 1 through the
    model's own token head, table detached; bridges soft / topk (arm default, k 128) / hard;
    learned gate starts at 0.5. On the latent-selected loop Phi is read once per slot (cells
    are bit-equal at each pass entry). Arms `discotul_fast`, `lxtul_disco_fast`, control
    `tul_onecell_pointer_fast` (+ eager twins). One-cell arms cannot carry
    `tg_fused_attention` / `slot_compact`, so they run slower than LXTUL. 41 CPU tests, 10/10
    sabotages, three 30-step graph_step smokes clean. Bench 2026-10-10 (420 steps, graph_step,
    `perf/r1007/bench.sh`, one run each): `lxtul_pointer_fast` 230.1 ms/step (26.7k tok/s);
    LX-DiscoTUL 274.1 (22.4k, +19 %); one-cell pointer control 201.5 (30.5k); DiscoTUL 244.9
    (25.1k, +22 % over its control). The disco term (top-128 read of the 49k table each pass)
    costs about 44 ms/step on both. Open: graph replay vs eager with disco on, soft / hard on GPU.
  - [x] Four CPU tests failed at master 6a33958e (`test_tul_strict_geometry` sabotage_2, three
    per-slot-depth tests in `test_tul_forward`), plus 11 more in three sibling files. Cause:
    test doubles without the optional kwargs phase 1 added (`lookups`/`twin` on `_tul_front`,
    4284c374; `cells` on `_sample_slot_depths`, f1ac1b9c). Stale doubles, not a code
    regression; the phase-1 merge ran only the CUDA tests. Fixed in the tests (no assertion
    weakened) and in `lab/divergence/bench_coda_k.py`'s depth pin.
  - [ ] D2b Prereg + runs, after A and C.

## Alternatives considered

- **Track in a hosted page with a shared database.** Visible from any device, but agents and
  future sessions read the repo, and a second copy of the status drifts. Kept in the repo.
- **One note per item with no tracker.** The owning notes exist for B and C; without one list,
  the order constraints (A before C4, C1c designed for B) are not written anywhere.

## Acceptance criteria

- Every box is [x] or [-] with a commit, filing or reason.
- Each experiment in it has a prereg in `docs`/`lab` experiments before its run.
- The note moves to `implemented/` (or `archived/`) when phase 2 closes.

## Risks

- B and C1c both change the fan's shapes; building C1c for fixed M first would need rework.
- C5 changes the architecture; its speed gain is measured, its quality cost is not.
