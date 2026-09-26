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
