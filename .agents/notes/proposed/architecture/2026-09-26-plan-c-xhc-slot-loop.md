# Agent Note: plan C, expanded hyper-connections (xHC) in the slot loop

Status: proposed

## Problem

The strict slot model trails the plain model by 0.265 nats at 5k and 0.333 at 10k on the
same coda tokens ([failures/2026-09-26-lxtul-fp01-vs-plain-10k.md](../../../../lab/experiments/failures/2026-09-26-lxtul-fp01-vs-plain-10k.md)).
The slot channel does work (zero-ablation worth 0.197 → 0.265) but keeps a roughly constant
share of a cross-span budget that grows with training. Plan (a) widens the channel by cells
and plan (b) adds a direct previous-span read
([2026-09-26-slot-channel-width-and-reach.md](2026-09-26-slot-channel-width-and-reach.md), being built). Plan C is
Wolfe's third route, built around xHC (Zhang et al., "xHC: Expanded Hyper-Connections",
arXiv 2607.14530).

A first-principles limit on what C can do, stated before the design. A span is about 18-21
tokens (the fp01 samples); at 4 nats per token that is under 90 nats of content. A slot
state is `[4, 1024]` in bf16. The channel's RAW capacity exceeds a span's content by orders
of magnitude, so a wider state alone is not expected to help. What can limit the channel is
what the loop WRITES into the state and what the coda can READ from two cells. xHC's own
ablation says the same thing about its residual: going from 4 to 16 streams gains 0.006
nats; the gain comes from giving the extra streams DIFFERENT content (temporal
augmentation, 0.014). C is therefore a test of distinct content in more streams, not of
more width.

## What xHC is (checked against the paper's HTML, 2026-09-26)

- HC update `X_{l+1} = H_res X_l + H_post F(H_pre X_l)` over N streams; mHC puts `H_res` on
  the doubly-stochastic manifold (Sinkhorn).
- Plateau at N=4, two reasons: every stream is written from the ONE sublayer output
  `out ∈ R^C` (extra streams become linear combinations of the same vector), and predicting
  `H_res` costs `O(N^3 C)`.
- Temporal feature augmentation, after the MLP only: `out_aug = [out; DWConv_4(out);
  DWConv_8(out); DWConv_12(out)]` (causal depthwise convolutions over the sequence), made
  orthogonal by modified Gram-Schmidt: K_r = 4 write-back components instead of 1.
- Sparse write, dense read: N = 16 streams; k = 4 written per layer, m = 2 always active
  plus 2 chosen by TopK of `σ(x̃ W_r)`; no load-balancing loss; all 16 streams are READ.
  `H_res` and `H_post` are computed on the k-subset only, so the mixer costs `O(k^3 C)`.
- Table 2 (10B): mHC N=4 2.004 (+0.6 % FLOPs); mHC N=16 1.998 (+18.8 %); + temporal
  augmentation 1.984 (+20.1 %); full xHC 1.983 (+3.3 %); k=2 1.991; k=8 1.982.
- Tested only on MoE transformers of 2.5B-28B. No looped, weight-shared or sub-1B model.
  Code: github.com/aHapBean/xHC (not yet checked for contents).

MORPH today (`morph/model/hyper_connections.py`): n = 4, `H_pre` / `H_post` softmax,
`H_res` ORTHOGONAL by a closed-form 4×4 Cayley map, deliberately chosen over Sinkhorn for
the weight-shared loop; a fused Triton kernel unrolled for n = 4; `_readout` takes the
stream mean. Facts sheet with paths: `/home/wolfe/morph-scratch/xhc/xhc_research.md`.

## Proposal

**Scope: the slot loop only (C-slot), not the whole model.** Under the slot loop the core
blocks run on SLOT positions only (`_tul_core`, 64 slots per row); tokens never pass through
the core. So the core's HC can be widened to N = 16 without touching the token path:
- memory is small: 64 slot positions × 16 streams is about 4.5x fewer elements than the
  token carrier's 1152 positions × 4 streams;
- the plain control is unchanged, so the gap-to-plain reading stays clean.

Design:
1. **Entry.** The prelude's slot seed arrives as `[4, C]`; streams 4-15 start at zero, so
   the first pass's reads see today's state.
2. **Sparse write in the loop's six blocks.** m = 2 fixed streams + 2 routed by TopK of a
   sigmoid score on the RMS-normed full 16-stream state, deterministic (no noise). Because
   the score reads the CURRENT state, the chosen streams can differ from pass to pass with
   one shared set of weights. That is how the weight-shared loop gets per-pass memory
   without per-pass jobs (Wolfe's rule: depth use is emergent, not forced). The k × k
   active mixer keeps MORPH's Cayley map, so the existing closed-form 4×4 math and kernel
   apply unchanged; the 12 idle streams pass through untouched.
3. **Dense read.** `H_pre` spans all 16 streams (one softmax row).
4. **Temporal augmentation, on the SLOT axis.** In the loop the sequence is the slot
   sequence, so `DWConv_κ` over slots gives each slot write-back components built from the
   outputs of the previous κ slots. Kernel sizes in slots, not tokens: {2, 4, 8} is the
   proposal (a span is about 20 tokens, so the paper's {4, 8, 12} tokens is well under one
   slot). Plus the paper's Gram-Schmidt. Built as a separate knob so its effect is
   isolated.
5. **Exit to the coda.** 16 streams map 1:1 onto `prefix_k = 4` cells × 4 streams (cell j
   carries streams 4j..4j+3 through its own `W_prefix[j]`), so the coda's carrier stays
   n = 4 and reads all 16 streams across 4 cells.
6. **Off is bit-identical.** N = 4 builds nothing new.

Arms (fp01 recipe, 5k then 10k, each paired against plain at the same step):
- **C1:** xHC slot loop, N = 16, k = 4, no temporal augmentation.
- **C2:** C1 + slot-axis temporal augmentation.
- **Control:** a1 (fp01 with `prefix_k: 4`, being built). It has the same 4 cells, but
  every cell holds the same state. So C1 − a1 isolates distinct content in the cells.

Readouts: the gap to plain at 5k and 10k (the failed test's instrument); the channel worth
(`worth_profile.py`); coda K1−K6; the router's stream-use histogram per pass (do passes
write different streams?); and the per-cell worth (ablate one cell at a time).

## Alternatives considered

- **C-model: xHC in every block, tokens included.** This is the paper's own setting.
  Rejected as the FIRST step: it widens plain and slot models alike, so it cannot show that
  the channel is what improved, and the researcher's back-of-envelope puts the token
  carrier at +8-9 GB at N = 16 against about 1 GB of headroom (an estimate, not measured).
  It is the right second step if C-slot wins.
- **Sinkhorn instead of Cayley for the active mixer.** Every measured N = 16 number is
  Sinkhorn. It is not adopted: MORPH chose Cayley over Sinkhorn for the weight-shared loop
  on its own evidence, and xHC's ablations do not test that choice. It is an ablation
  candidate if C1 fails.
- **A static stream set for all passes** (routing computed once before the loop). It is
  cheaper, but with one set 12 streams are dead for the whole loop, which is close to
  n = 4 with unused parameters. Rejected.
- **The paper's token-axis kernel sizes {4, 8, 12}.** They span under one slot on the slot
  axis. Rejected for the loop; kept for C-model.

## Acceptance criteria

- The tests hold: N = 4 is bit-identical (loss, params, grads `torch.equal`); the routed
  gather and scatter equal a dense masked reference; the router is deterministic; the
  checkpoint loads; the temporal-augmentation knob off is bit-identical to C1.
- A 30-step GPU smoke, with peak memory within 1 GB of fp01's.
- Preregs with predictions come before any 5k run.

## Risks

- **The loop's gain growth:** 16 streams give the map more directions to amplify. The
  divergence README's measured tripwire applies. Read `core_map_fd.py` fp32 on C1 at 5k.
- **The router can collapse** onto the same 2 routed streams every pass (no balancing loss
  in the paper). The per-pass stream histogram shows it. A collapsed router makes C1 equal
  to a1 with extra parameters.
- **The slot-axis convolution is a cross-slot carry.** Earlier carry arms replaced the
  cross-slot read and diluted it (`tul.loop_carry`, 2026-09-20). C2 must beat C1, not only
  fp01.
- **The capacity argument above says width alone will not help.** If a1 and C1 both gain
  nothing over fp01, the channel's limit is the coda's read, not the write, and plan (b) is
  the lead.

## Build notes (2026-09-26)

Built on branch `tul-xhc` from 21e9705. Module `morph/model/tul_xhc.py`; tests
`tests/test_tul_xhc.py`; configs `tul_slot_spandec_strict_e4probe_fp01_xhc.yaml` (C1) and
`..._xhc_ta.yaml` (C2). What was built, and where the code forced a choice the plan did not make:

- **The core's residuals are replaced, not added to.** On an xHC model the six core
  blocks' `mrr_attn` / `mrr_mlp` hold `XHCResidual` (built last, private generator, so every
  other weight equals the same-seed model without the key). The core then has no n = 4
  residual, so `_core_region` (the token path) raises. The arm needs `tul.activate_at: 0`
  (fp01 has it). An fp01 checkpoint cannot initialise the core residuals of C1.
- **Router gradient.** TopK is not differentiable in its index. The router score multiplies
  the routed streams' WRITE (`H_post y` and the temporal components); the fixed streams have
  gate 1; the Cayley mixer is never scaled, so it stays orthogonal. No straight-through, no
  noise, no balancing loss. Ties go to the lower stream index (stable sort).
- **Write weights.** The main write keeps MORPH's HC form (row sum of a column softmax over
  the k x k block, about 1 per stream at init). The temporal components are written through
  `tanh` weights `[k, R]` whose bias starts at atanh(0.01). Gram-Schmidt removes projections
  and does not renormalise. Conv kernels start as the causal moving average `1/kappa`.
- **Entry.** The per-pass source `e` is zero-padded like the carrier, so `DiagonalInjection`
  refills streams 0-3 only. The per-layer x0/bigram term and the LXTUL-E code are
  single-stream terms and broadcast into all 16 streams (the carrier's standing rule). The
  code's size is `r * rms` over all 16 streams, so it is smaller relative to streams 0-3
  than on fp01 while streams 4-15 are small.
- **Readers of the exit state.** Parallel head, NextLat and the enum stats: `_readout`, the
  16-stream mean. Fixed-point term and gain hinge: the full 16-stream carrier. After the
  write, `h_slots` is the per-stream mean over the four cells. Plan ablations act on the
  four-cell stack together.
- **Deviation: the gain hinge replays the routing.** A finite difference across a TopK flip
  is a jump. Tiny fixture at init: row gain 1.17 and a slot at 4.77 with the router free
  (penalty 24 at lambda 100), 0.87 with the perturbed call reusing the unperturbed call's
  stream choice. Without the replay the hinge charges the router for being decisive.
  `core_map_fd.py` and other external finite-difference probes do not replay and read the
  jump on an xHC model.
- **Deviation: the hinge's two applications are checkpointed on xHC models.** The first
  C1 smoke OOMed. Eager, one fwd+bwd at the panel shape: fp01 14.28 GB, fp01 with
  `prefix_k: 4` 15.76, C1 24.04, C1 hinge off 15.52, C1 after the change 15.70, C2 15.71.
- **Refused** (`TULConfig._check_xhc` and the build): the paid loop, `loop_reads_tokens`,
  `code`, `code_target` / `code_grade`, `loop_denoise`, `slot_cells > 1`, `fan_k > 0`,
  `vq_codes`, `prefix_source != exit`, `core_stage_cond`, `db_loop`, `gram`, `gate`,
  `grad_pass`, `slot_chain`, `reread`, `loop_carry`, `recur_gate`, `cond_layers`,
  `pass_readout != last`, `bcast`, `mux_readout != mean`, the per-pass trajectory readers,
  `core_token_aux`, `center_exit`, a parcae core, SCSE, an FM planner, and
  `xhc_streams != prefix_k * hc_streams`.
- **Smoke (60 steps, compile on, 2026-09-26).** fp01: peak 15.86 GB, 5966 tok/s, val 9.489.
  C1: 17.51 GB, 3765 tok/s, val 9.515. C2: 17.51 GB, 3428 tok/s, val 9.505. C1 runs at
  0.63x fp01's throughput (a1's 5k run logged 5723 tok/s early). The plan asked for a
  30-step smoke with peak within 1 GB of fp01's; C1 is 1.65 GB above fp01 at 60 steps.
- **Router after 60 steps (one val batch, depth 6).** Streams 2-15 are all used; at pass 0
  most residuals send almost every slot to one pair of routed streams; by pass 5 the routed
  pair differs from pass 0's for 52 % (C1) and 50 % (C2) of (slot, residual) cases.

## 2026-09-26 outcome and C1b

**C1 and C2 detonated.** C1 crossed the tripwire at 3190 (`preclip/total` 5.7e4; 9.8e5 at
3200), never recovered, and finished at val 4.923 against ~4.49 for the 4-cell control, with
the slot channel worth 0.020 (`plan_worth_zero`). C2 crossed 1e4 at 2679 and the guard killed
it at 2711. C1 also had one recovered single-step spike at 1831-1832 (`preclip/total` 4.3e4,
back to 1.26 at 1833) with the same signature as the detonating step: a tail-hinge excursion.

**Anatomy** (per-step `probe.jsonl` of both arms and of the N = 4 control a1/pk4; onset =
first step where the 50-step rolling median crosses the threshold; `core_gain_t*` is the
trainer's norm ratio `|h_out| / |h_in|`, max over rows, not a Jacobian):

| reading | C1 | C2 | pk4 (control) |
|---|---|---|---|
| `loss/gain_est` (hinge, routing replayed) > 0.98 | 2698 | 1887 | 1752 |
| `loss/gain_est` > 1.00 | 2799 | 2111 | 1947 |
| `loss/gain_reg_weighted` > 0.1 / > 1 | 2810 / 2895 | 2119 / 2195 | 1975 / never |
| `preclip/core` > 1 / > 10 (baseline ~0.1) | 2815 / 2910 | 2121 / 2278 | 1937 / never |
| `loop/delta_ratio_t1` > 0.8 | 2905 | 2214 | never |
| `loop/core_gain_t1` > 1.3 / > 3 | 2956 / 3043 | (1.3 from step 200) / 2446 | never > 3 |
| `preclip/total` > 1e4 | 3175-3200 | 2679 | never |

- **The hinge's own reading led, and it did not hold.** It crossed its 0.98 target 258 steps
  (C1) and 559 steps (C2) before the carrier grew, and 490 / 792 steps before the tripwire.
  The hinge then fired at penalty 1-5 with the core gradient 10-300x its baseline for
  ~400 steps, while `loss/ce_main` rose (C1 median 11.85 over 2000-2500, 15.19 over
  2900-3100; with the global clip at 1.0 the hinge's direction took most of each update).
  The reading kept climbing (C1 median 1.134 over 2900-3100, C2 1.088 over 2000-2500).
- **The carrier grew in two stages.** At 3000 every pass grew it (per-pass ratios 1.3-1.5,
  row norm 740 -> 1990 over passes 1-6). From 3025 one pass carried it: pass 1 jumped
  x4.2 (3025) -> x10.7 (3100) -> x15.8 (3150), row norm 810 -> 12,600 at 3150, with passes
  2-7 at 1.0-1.3.
- **The detonating gradient was the hinge's.** At 3190 the penalty was 49.6 (tail 31.0,
  per-slot gain max 8.1); at 3200 it was 4042 (tail 2222, per-slot max 73.6). The 1831
  spike: per-slot max 7.2 / 16.5, penalty 24.7 / 445.
- **The control met the same crossing through scale.** pk4's reading crossed 0.98 at 1752
  and 1.00 at 1947 under the same hinge, and fell back to 0.955-0.965 by 2200 for the rest
  of the run (penalty rolling median above 0.1 from 1975, never above 1). Over the same window its PASS-0
  write grew: `core_gain_t0` 4.7 (1500) -> 20.6 (2000) -> 21-22, pass-1 entry norm
  1990 -> 7500-9500. A pre-norm block writes `w(h/|h|)`, whose size the weights set, so the
  pass map's gain grows like `|w| / |h|`; a large carried state makes every later pass
  quiet. The xHC arms did not take this route at pass 0 (`core_gain_t0` 1.1-1.4 while the
  reading crossed and the fight began, ~2 later) and took it late at pass 1, under the fight. So a norm ratio of 10-20 is not by
  itself a detonation signature: pk4 lives at 20 for 3000 steps. WHY xHC did not grow its
  pass-0 write is not measured (the GPU diagnostic below reads it).

**The working diagnosis, scored.**
1. *"The hinge replays the routing, so it is blind to growth the routing adds."* Refuted as
   the cause: the replayed reading was the FIRST instrument to move, in both arms. The
   replay is also the right quantity for the backward: autograd differentiates the chosen
   piece (TopK indices carry no gradient), so the backward product sees the replayed
   Jacobian. The router-free reading is a different map: on the tiny fixture at init it
   reads 1.04-1.32 per row and up to 5.9 per slot against 0.87 replayed (the diagnostic
   below, CPU).
2. *"Streams 4-15 start at 0 and get additive writes with no decay."* Partly wrong on the
   mechanism and not decidable from the probe. `DiagonalInjection`'s `A` multiplies the
   context slice of EVERY stream (the carrier is `[B, S, N, C]`, `A` acts on the last axis);
   only the `dt * e` refill is limited to streams 0-3. The 704 non-context channels have no
   decay in any stream, entry streams included, on the N = 4 carrier too. The probe logs
   whole-carrier norms, so which streams grew is open.

**C1b, the fix** (`morph/configs/tul_slot_spandec_strict_e4probe_fp01_xhc_c1b.yaml`,
`..._c1b_ta.yaml`; wandb `lxtul-e4probe-fp01-xhc-c1b[-ta]`, 5000 steps). Two keys on C1 / C2:

- `model.slot_state_renorm: true` (existing lever, 2026-09-04): after every pass each slot's
  16-stream carrier is rescaled to the norm it entered the loop with, direction kept. It
  already acts on the expanded carrier (`_n0` is taken after `_xhc_expand`), and is now the
  function `MORPHTransformer._renorm_to`, shared with the hinge.
- `model.slot_gain_renorm: true` (new, default false): the gain hinge differences `R(f(h))`,
  the map the loop applies under the renorm, instead of `f(h)`. The router replay is kept.
  The raw reading of the same two applications is logged as `loss/gain_est_raw`. It raises
  unless both the renorm and the hinge are on.

Why this pair. With `f(h) = h + w(h/|h|)` a hinge can be met in two ways: grow `|h|` (pk4's
route, C1's late route) or lower the blocks' angular sensitivity. The renorm removes the
first, so the scale jump cannot happen. The hinge on `R(f)` reads
`(|h| / |f(h)|) P_perp J_f`, which stays bounded by the angular part however large the write
is, so the hinge stops charging the write size and keeps charging directional expansion
(the tail hinge, per slot, target 1.1, is the lever E14 said was missing). Measured on the
tiny test models: with the core writes scaled x30 and the renorm on, today's hinge reads the
raw step at 2.09 and charges a penalty of 863 for a map the loop never applies, while `R(f)`
reads 0.26 and charges 0; on the linear step `f(h) = 3h` the raw reading is 3.0 and `R(f)`
reads 1.00005.

Why this is not the magnitude clamp CLAUDE.md warns about. `core_gain_clip` clamped the
realised ratio only when it passed a threshold, and the spectral caps bounded weights; the
optimizer trained the unclamped map in the normal regime. The renorm is part of the map at
every pass from step 0, so the optimizer trains the loop as a map on the sphere; measured on
Y1 (2026-09-04): healthy to 5000 with 0 spikes and a raw map that drifted less
(`rms_t3` <= 0.950). It does NOT bound directional expansion: E14 (hinge on the RAW map at
target 1.02 plus the renorm) still spiked at 4639 through single-sample directional
excursions. C1b differs from E14 in what the hinge reads and in the tail hinge.

**Alternatives considered.**
- (a) The hinge reads the router-free map. Rejected: the replayed reading led the onset, and
  the free reading charges TopK flips (1.04-1.32 per row at init on the fixture, against a
  0.98 target), i.e. the router for being decisive.
- (b) A decay (A-style) or an RMS bound on streams 4-15. Rejected: the growth was not a
  steady per-pass accumulation until the hinge fight was already on, a decay does not bound
  one pass's write, and a new decay is a new floor lever the hinge can be met through (the
  injection-floor README). A per-stream RMS bound is a conditional clamp, the pattern above.
- The renorm with the hinge on the raw map (the E14 pairing). Rejected: under the renorm
  that hinge charges the write size (863 vs 0 on the grown fixture).
- The renorm with the hinge off (the Y1 pairing). Not chosen: it changes the fp01 recipe's
  constraint as well, one more factor against C1.
- No secondary lever. C1b-ta is C1b with C2's temporal augmentation (the renorm acts after
  the pass, the augmentation inside the MLP write; no new code).

**Risks.**
- The renormed hinge can be met by writes that swamp the carried state (`|w| >> |h|`, small
  angular part): each pass then forgets most of its input (gain 0.26 on the grown fixture).
  Read `gain_est` against `gain_est_raw` and `loop/delta_ratio_t*`.
- `loop/core_gain_t*` reads ~1.00 by construction under the renorm. The LXTUL-E code is
  added after the renorm, so the carried norm moves ~1 % per pass (0.992-1.017 on the
  fixture at k = 4).
- C1b against a1 (pk4) changes two things, xHC and the renorm pair. A C1b gap to a1 is not
  an xHC effect until a1 with the same two keys has run (config not built).
- Nothing here has trained. A prereg with predictions comes before the 5k run.

**Instruments and chain.** `lab/divergence/xhc_carrier_anatomy.py` (one eval forward at a
forced depth): per pass and per stream the carrier's RMS (entry streams vs 4-15, context
channels vs the rest), the common-mode share (the "large shared write" reading), the norm
ratio, the hinge's finite difference replayed / router-free / renormed, and per residual the
router's choice, gate, read mass on the entry streams, `|y|` and the write per stream group.
Run on the tiny fixtures on CPU; NOT yet run on a checkpoint. The chain
`/home/wolfe/morph-scratch/c1b/c1b_chain.sh SHA WT` waits for `PROBES DONE` in
`/home/wolfe/morph-scratch/tulv2/probe.log`, reads the anatomy on C1 step_2500, C2
step_2500, C1 step_5000 and pk4 step_2500, then runs a 300-step C1b smoke (peak memory,
max `core_gain`, `gain_est`, `gain_est_raw`, `delta_ratio`, `preclip/total`); status lines in
`/home/wolfe/morph-scratch/c1b/chain.log`, last line `C1B DONE`.

**Tests** (`tests/test_tul_xhc.py`, the `c1b` block): `slot_gain_renorm` off is
bit-identical to the tree before C1b (`36a9823`) with the renorm off and on; the renorm
holds the 16-stream carrier at its entry norm on a model whose carrier grows ~10x per pass
without it; the hinge reads `R(f)` (linear step, grown model; `gain_est_raw` equals today's
reading bit for bit); the knob refuses where it would do nothing; both arms compose, differ
from C1 / C2 by the two keys, build and train a step. Each was sabotaged once and failed.
