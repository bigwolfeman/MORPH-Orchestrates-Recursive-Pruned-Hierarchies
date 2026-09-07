# Agent Note: Spiral-TUL — the shared core looped over a span → chunk → token resolution schedule

Status: proposed

**Wolfe, 2026-09-07: "This is not something for us to be touching right now." Parked. Do not build
from this note without his say; it records the reading of the paper and nothing more.**

Date: 2026-09-07. Wolfe: the TG mask helps loop contribution for the TUL shape, but the loop
needs full token granularity; SpiralFormer (arXiv 2602.11698, Yu et al.) "seems to solve that
very literally". Drawing first, build after review.

## Problem

The slot loop's output lives at a SIDE position. A token reaches it by attention, the same way
it reaches any earlier token, and the direct route wins: M-next's loop contributes +0.0003
(mean 6) / +0.0011 (mean 12) nats to the tokens (K1−K6, paired, 480 rows). The TG mask
(`tul.tg_restrict`) removes the direct route and the contribution becomes +0.021 / +0.049,
but the tokens lose direct attention to prior tokens and pay 0.11 nats for it. We want both:
full token attention AND a loop the tokens cannot walk around.

SpiralFormer's answer (from the paper; no code was published — only the lab's predecessor
MeSH repo, github.com/LivingFutureLab/MeSH): the loop never produces a side position. Each
iteration POOLS the token stream to a coarse sequence, runs the shared core on it, BROADCASTS
the output back over the tokens as an additive per-token update, right-shifted for causality,
and the next iteration (or the coda) starts from `anchor + update`. Full-resolution attention
lives in the unshared prelude and coda; the best variant (SpiralFormer-L) has NO full-res loop
iteration at all, and coarse-to-fine beats fine-to-coarse by 0.24 PPL.

## Proposal

One sequence of tokens. No slot positions, no prefix positions, no `W_prefix`, no mask, no MUX
head. One shared core, T iterations, each at its own resolution. Loss: plain next-token CE.

```
 tokens ─────────► PRELUDE (unshared, full res, causal attention over ALL prior tokens)
                        │
                        ▼
                  h(0) ∈ [B, L, n=4, d]      ◄── the ANCHOR, kept for every iteration
                        │
   ┌────────────────────┼──────────────────────────────────────────────────────────────┐
   │  iteration t       │        (ONE shared core; only the grouping changes per t)     │
   │                    ▼                                                               │
   │   h(t) = h(0) + ũ(t−1)          [B, L, ·]           (ũ(−1) = 0)                    │
   │        │                                                                           │
   │        │ POOL  by group π_t(i):   z_j = mean_{i∈group j} h_i(t)      [B, L_t, ·]   │
   │        ▼                                                                           │
   │   ┌─────────────────────────────────────────────┐                                  │
   │   │  CORE  (prelude-less MORPH core, 6 blocks)  │  causal attention over L_t       │
   │   │  z(t) ──► ẑ(t)                              │  positions = group index         │
   │   └─────────────────────────────────────────────┘                                  │
   │        │                                                                           │
   │        │ BROADCAST: u_i = λ_t · β_{π_t(i), ρ_t(i)} · ẑ_{π_t(i)}   (λ_t = √g_t,       │
   │        │            β = softmax over the g_t members of the group, from ẑ; or 1/g_t) │
   │        │ RIGHT-SHIFT: ũ_i = u_{i − s_t}, s_t = g_t − 1  ⇒ token i reads ONLY a       │
   │        │            summary whose members are all ≤ i                                │
   │        ▼                                                                           │
   │   ũ(t) ∈ [B, L, ·]  ──────────────────────────────────────────────────────────────┘
   │                                   next iteration: h(t+1) = h(0) + ũ(t)
   └───────────────────────────────────────────────────────────────────────────────────
                        │  after t = T−1
                        ▼
                  h(T) = h(0) + ũ(T−1)
                        │
                        ▼
                  CODA (unshared, full res, causal attention over ALL prior tokens) ──► logits


 RESOLUTION SCHEDULE, coarse → fine (the direction the ablation says matters):

   t     groups                         L_t          what the core sees
   ───   ─────────────────────────────  ───────────  ─────────────────────────────────
   0     SPANS by the boundary rule     ≈ L / 12     one state per sentence-ish span  ◄─ this is today's slot loop
   1     fixed chunks of 4 (offset 2)   L / 4        one state per 4 tokens
   2     fixed chunks of 2 (offset 1)   L / 2        one state per pair
   3     tokens (g = 1, no shift)       L            the paid loop            ◄─ optional: Spiral-L skips it

 Cost per iteration ∝ L_t: T = 4 costs about (1/12 + 1/4 + 1/2 + 1) ≈ 1.8 core passes over
 L, against T × L = 4 passes for a paid loop of the same T, and 0.83 without the g=1 level.


 WHERE TODAY'S PIECES GO:

   today                                   Spiral-TUL
   ─────────────────────────────────────   ────────────────────────────────────────────
   slot positions in the sequence          gone: a span is a POOL GROUP, not a position
   E_slot + W_sent·embed(t_last) seed      gone: z_j is pooled from the live stream h(t)
   W_prefix / prefix_k                     gone: the write-back is the broadcast-add
   tg_restrict (the mask)                  gone: tokens keep full attention in prelude+coda
   MUX forecast head, emit/plast weights   gone: LM loss only
   slot loop over slots only               = iteration t=0 (spans)
   paid loop over tokens+slots             = iteration t=3 (g=1), optional
   Poisson depth per sample                fixed T (Spiral); Poisson over the SCHEDULE is open
   HC-Cayley n=4 carrier                   the natural MeSH: 4 streams with learned mixing
                                           (MeSH beats the plain anchor by 0.13 PPL)
   gain hinge / fixed-point term           the loop is 4 iterations on short sequences; the
                                           contraction question is reopened, not answered


 DECODE (Spiral Alg. 2 = "think once per span, decode cheaply"):
   per token: prelude + coda always; iteration t's core fires only when the token CLOSES a
   group of level t (once per span, once per 4, once per 2); each level keeps its own KV
   cache of group states. Tokens between closes read the cached ẑ of the last closed group.
```

## Alternatives considered

- **Keep the slot loop and add the broadcast only** (span level, T iterations at spans, then
  add ẑ_n into every token of span n+1, no mask). The smallest change from today; it is the
  t=0 row alone with the write-back fixed. Loses the multi-resolution part that the paper's
  title rests on. Kept as the FIRST arm to run: it isolates the write-back from the schedule.
- **Paid loop with the mask** (tokens in the loop, slots as positions, tg_restrict on). Keeps
  the side position and the mask; the paper says the side position is the problem.
- **Anchor vs the HC carrier as MeSH.** Spiral's anchor discards h(t) each iteration and adds
  ũ(t) to h(0); MeSH accumulates updates into B slots with per-token softmax read/write gates
  and wins 0.13 PPL. MORPH's n=4 HC streams already are a learned multi-slot carrier. Open:
  whether "h(t+1) = HC-mix(h(0), ũ(t))" is closer to MeSH than to the anchor.
- **Spans vs fixed chunks at every level.** Spiral uses fixed chunks with an offset and no
  semantic boundaries. The span level is MORPH's addition (think once per span); levels 1–2
  are fixed chunks because splitting a span "in half" has no clean definition.

## Acceptance criteria

- The forward with `slot_layout=None` is bit-identical to the plain model (invariant §6b) —
  the pooling path is a new argument, not a flag in the forward.
- A CPU test: the right-shift proposition (a token's update depends only on positions ≤ i),
  checked by finite differences on the input embeddings.
- On the arc's 480 rows at 5000 steps, K1−K6 (here: the loop OFF vs ON at each level) on the
  TOKENS, paired CI, against the mean-6 rulers; wall clock against E4 (68 min) and notul.

## Risks

- MORPH's attention module (CCA/CSA/HCA, window 16-ish, compression ratios) on an L_t ≈ 85
  sequence: branches may degenerate at that length (the "audit module geometry first" lesson
  — print branch norms at the real shapes before any run).
- RoPE/CoPE positions at coarse levels: the paper does not say; group index at native
  spacing is the reading of its per-level KV cache.
- Fixed T and no Poisson: the per-sample depth draw that every MORPH result rests on has no
  counterpart in a resolution schedule.
- The loop's contractivity story (E12/E7 scale mode) is untested on 4 short iterations.
- The per-level decode caches are new inference code; the eager recompute generator covers
  training-time checks only.
