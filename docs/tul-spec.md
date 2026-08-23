# TUL — Thought Unpack Loop: specification v0.1

Status: **implemented, run, and measured** — A1 beats dense A0 on the short
schedule (see [lab/tul/arms-result.md](../lab/tul/arms-result.md)); further
testing in progress. Off by default in `base.yaml` (`tul.activate_at: never`).
The v1 mechanism is in the tree and gated by `pytest tests/`: the causal
boundary rule and packer (`morph/model/tul_layout.py`), the slot parameters and
plumbing (`morph/model/tul.py`), the TUL forward inside
`morph/model/transformer.py`, the `tul:` Hydra block resolved by
`morph/training/tul_setup.py`, loader support in `morph/training/data.py` (and
`curriculum_data.py`), and the eager generator (`morph/inference/tul_generate.py`).
NOT implemented and asserted-zero rather than ignored: the `stp_lambda`,
`set_lambda`, `carry`, `xattn` and `bcast` arms (§3.5) all RAISE on a non-default
value. Two v1 deviations from the text below are recorded in §3.1 (run collapse
is causal) and §4 (the packer's tail padding). Written 2026-08-16 from the
prior-art review in `ignore/Ai-notes/08-16-2026/prior-art/` (SYNTHESIS.md,
MORPH-READ.md, 28 per-paper notes) and from a read of
`morph/model/transformer.py`, `morph/training/train.py`,
`morph/training/data.py`, `morph/inference/kv_cache.py`.
Markers **[W]** are project design decisions (not paper defaults).

---

## 0. One paragraph

MORPH keeps its Parcae loop, but the loop runs over **thought slots**, one per
span of text, instead of over every token. Text is cut into spans at
punctuation (`.;!?`, dashes, newlines — no comma **[W]**). A slot position is
inserted after each span; its input is the mean of the span's token embeddings
(TST-native), the prelude reads it in context, the core loops on it for a
per-slot Poisson number of iterations **[W]**, and the coda decodes tokens with
the slot's looped state visible as an ordinary attended position. Token
positions run prelude → coda only; they never enter the core. The slot state
has no loss of its own — it is trained only through the tokens the coda emits.
TUL structure switches on at the TST superposition → recovery boundary **[W]**;
before that the run is plain MORPH on bags. Everything else in MORPH (HC
carrier, GLA, MORTAR/ReMoE, ternary/int6 QAT, AdEMAMix, curriculum) stays on
**[W]**.

## 1. Why this shape, in five measured facts

1. **Decoding a span from one vector plus an offset, with no token feedback,
   collapses.** Measured on Huginn 2026-08-16 (span decoder 7.12 nats, emits
   `.` at offset 1). Predicted by every family: Bowman 2015 inputless decoder
   380 vs 119 PPL; MegaByte one-shot patch 1.263 vs 0.687 bpb (Table 7); NAT
   positional-only ~2 BLEU; Future Lens blind linear 0.292 vs token-fed
   learned prompt 0.484 at t+2 (Table 2, Eq. 9/11); Explorative Modeling:
   one-shot multi-target regression predicts the mean. DeepSeek-V3 rejected
   blind heads and shipped the token-fed chain (§2.2). Per-offset injection is
   the weakest variant in Block Transformer, Hourglass and MegaByte.
2. **The shape that works is a loss-free latent per unit plus a small AR
   decoder that sees the unit's previous tokens** (MegaByte, H-Net, BLT, Block
   Transformer, DeepSeek-MTP, Pred-Sent, LD4LG). Removing the AR path is
   catastrophic (MegaByte 0.687 → 1.263; Hourglass Table 6 1.128 → 1.460).
3. **How the latent reaches the decoder matters, and the regime decides.**
   Decoder that sees only its unit: prefix positions the decoder can refine
   beat per-layer cross-attention by 0.18 nats (Block Transformer Fig 3f;
   Hourglass: attention-upsampling > repeat > per-offset linear). Decoder with
   a byte window across units: cross-attention at all layers beats input-only
   (BLT Table 7). Both agree that input-only summation and per-offset slices
   lose. TUL uses the prefix route because MORPH's coda attention already
   exists (§3.4).
4. **The unit's summary must NOT be regressed onto** (LCM Table 3/4, Pred-Sent
   §3.2, CoCoMix Fig 6b, Block Transformer §4.2 block-level MSE hurts). Token
   CE through the decoder is the objective; auxiliaries are arms.
5. **Deterministic, content-aligned boundaries are close to learned ones and
   fixed stride is worst** (BLT §4; Dynamic Token Pooling Table 2: whitespace
   1.133 ≈ unigram 1.134 > Gumbel 1.136 > entropy 1.138; fixed SF2 1.149;
   H-Net pool-6 0.780 vs space 0.755). Punctuation with a collapse/merge/cap
   rule is v1; learned boundaries are a later ablation.
6. **Deep compute at boundary positions only, with the deep state read back
   through a causal token stream, ties BPE at matched compute — at word
   scale.** SpaceByte (global blocks at the first byte of a space/punct run,
   residual-added there, local layers attend it: 1.009/0.748/0.500 bpb vs
   SentencePiece 0.989/0.768/0.508 at 1e19 FLOPs, Table 1; fixed stride
   +0.10 bpb on PG-19); AU-Net (75 % of layers at the word stage beats 25/50 %,
   Table 5; 1B AU-Net-2 69.9 vs BPE 70.2 HellaSwag, Table 2); Hierarchical
   AT (backbone state as prefix of a 3–4 layer char decoder; word accuracy
   ties BPE at 1B/3B/7B, Table 1; MegaByte's fixed 8-byte split −2.7
   HellaSwag). Two of three use the prefix route; AU-Net's per-offset
   broadcast is ADDED to a causal byte stream and matters only at ≥3 levels
   (Table 4). None of them loops at the boundary positions — that part of
   TUL has no precedent in this family. TUL's spans are 3–10× longer than a
   word; §10 keeps that as the open risk.

Two things the literature says NOT to expect at 5090 scale: beating the
per-token baseline on PPL (Block Transformer needs 2–3× params to match
vanilla; BD3-LM block 4 is +24% PPL; Gloeckle's blind heads help only ≥3B),
and downstream wins. The bar **[W]** is: it works — the slot state measurably
carries the plan and generation does not collapse — so it can be scaled.

## 2. The diagram

The design as first drawn (kept verbatim; the MORPH-integrated version follows):

```
tokens ──► LOCAL ENCODER (prelude, 3-4 layers, per token, causal windowed attn)
              │  token states e_t
              ├──► POOL each span into one vector (1 cross-attn layer, learned query)  [BLT local encoder]
              ▼
         SLOT SEQUENCE  <z_1> <z_2> ... <z_N>     (N = number of spans, ~8-10x shorter)
              │  slot input_i = pool(span_i) + W·h_{i-1}          [BLT input; Coconut feedback]
              ▼
         CORE × T over SLOTS ONLY  (6-8 layers, Poisson depth per SLOT, DiagonalInjection, GLA carry as now)
              │  h_i = plan for span_{i+1}; later slots attend h_≤i at every layer, every iteration  [Coconut channel, free]
              ▼
         LOCAL DECODER (coda, 6 layers — grow it, BLT T9)  per token of span_{i+1}:
              input_j   = drop_p( Emb(tok_{j-1}) ) + offset_embed[j]         [DeepSeek/MegaByte fed-back; Bowman dropout]
              self-attn = causal, windowed, over emitted tokens               [MegaByte local]
              cross-attn= to h_≤i at EVERY layer, per-layer K/V projections  [BLT T7, Optimus F5]
              head      = tied LM head, fused CE
```

Three edits were made after the MORPH read, each traceable to a source:

* **POOL** is not a new module. The slot's input embedding is the mean of the
  span's token embeddings (TST-native, `data.py` bagging; Dynamic Token
  Pooling: mean-pool > take-last), and the prelude's own attention at the slot
  position over the span's tokens is BLT's cross-attention pooling with the
  mean-pool as the query initialiser (BLT Eq. 5). `W·h_{i-1}` is an arm.
* **The decoder input is the prelude state, not a fed-back embedding plus
  offset.** BLT Eq. 9 seeds its local decoder with the local encoder's token
  states (`D_0 = h_lE`); MORPH's `n_core == 0` path already routes
  `prelude → input_norm → coda`. MORPH positions predict the next token from
  the current token's state, so "fed-back" is the ordinary causal alignment.
  The Bowman word-dropout tax becomes dropout of the token state entering the
  coda (§3.4).
* **The latent reaches the decoder as attended slot positions in the shared
  sequence** (Block Transformer prefix; Coconut thought positions; Optimus
  memory), refined by the coda's own layers, not by a separate per-layer
  cross-attention branch (that is arm `xattn`).

MORPH-integrated (one shared sequence; `z` = slot position, `t` = token):

```
sequence      t t t t t . z | t t t t t t t ; z | t t t t . z | ...
                        ↑span i ends   ↑slot i     ↑span i+1
                                                                                     [source]
PRELUDE (4 layers, ALL positions, causal, windowed+CSA/HCA as today)
  token input  = embed(t) (+ bigram, value-embeds, x0 as today)
  slot input   = E_slot + mean(embed(tokens of span i))                              TST; DTP; BLT Eq.5
  slot output  = pooled-in-context span summary (prelude attention over the span)     BLT §3.2.2

CORE × T_i  (6 layers shared, SLOT positions ONLY: gather → loop → scatter)
  T_i ~ Poisson(mean_depth) per SLOT, clamp [1, max_depth]                          Parcae; [W]
  masked update: frozen slots keep h and still serve K/V each iteration               MORPH _forward_single
  h_i attends h_<i at every layer, every iteration                                    Coconut (attended positions)
  DiagonalInjection(e = slot prelude state), GLA carry, TBPTT depth 4 — unchanged     Parcae; MORPH
  h_i has NO loss of its own                                                          MegaByte, H-Net, LD4LG, Pred-Sent

CODA (4 layers, ALL positions)
  token position input = input_norm(prelude state)     — tokens SKIP the core          MORPH seed path; BLT Eq.9
                          dropped (replaced by E_mask) with prob p                     Bowman word dropout
  slot position input  = h_i (looped state)                                            Block Transformer prefix
  attention as today: tokens attend slots as ordinary positions (window + CSA/HCA)     BT Fig 3f; Hourglass; Optimus
  LM head (tied), fused CE
    label(z_i)          = first token of span i+1                                       MegaByte p=0; BT last prefix
    label(last t of i)  = first token of span i+1 (kept: plain-LM counterfactual)       — (TUL, see §7.3)
    label(other t)      = next token                                                    NTP
```

Layer-passes per token, measured span lengths (OWT 19.2 after cap 32; code 9.2):
local 4:6×6:4 → tokens 8 + slots (4+36+4)/19.2 = **10.3 vs 44** (4.3×); code
(4+36+4)/9.2 → 12.8 vs 44 (3.4×). Cloud 4:8×8:4 → 8 + 72/19.2 = 11.8 vs 80
(6.8×). The core's attention runs over 9–19× fewer positions.

## 3. Architecture, MORPH terms

### 3.1 Sequence layout and the boundary rule

* One shared position axis. A **slot** is one looped core state per span,
  inserted AFTER the last token of the span. In the shared layout it owns
  `prefix_k` adjacent positions (**default 2 [W]**, Block Transformer App.
  F.2 / Fig 3f: prefix length 2 chosen over 1, 2–6 all beat 1). Prelude: both
  positions take the slot input (§3.2). Core: only the FIRST position is
  gathered — one loop, one state `h_i` per span, zero extra core compute.
  Coda: `h_i` is projected into `prefix_k` vectors `h_i W_1`, `h_i W_2`
  (`W_k` init identity-scaled) that become the coda inputs of the two
  positions — the first has NO label and only carries the plan (later
  positions attend it), the second predicts the first token of the next span
  (its label is `t_1(i+1)`). This separates "be a good summary" from "be a
  good next-token predictor" at one vector (the LTD think-position failure,
  t+1 0.339 vs t+2 0.108 at one position). Cost: `max_slots` extra coda and
  prelude positions per row (2048 → `L_total` 2560, +10 %). `prefix_k = 1`
  is the arm. The spec is written so
  either is one config key. Its input id is `slot_id`, a StarCoder2
  special token that never occurs in the shards (default `<fim_pad>`, resolved
  at build, absence asserted at data prep). The LM head never predicts
  `slot_id`: its logit is masked to −inf in the fused CE and at generation.
* **Boundary rule (causal, identical at training and generation) [W]:**
  1. `B` = set of token ids whose decoded string, after stripping trailing
     whitespace, ends in one of `.;!?`, or contains `\n`, `—`, `–`, or `--`.
     EOS (id 0) ∈ B. Resolved once from the tokenizer at build (the
     `punct_boundary.resolve_punct_token_ids` approach; the id-membership
     caveat in that file's docstring is accepted for v1 because the rule must
     be causal to be usable at generation).
  2. A **run** of consecutive B tokens (e.g. `.` then `\n`) is ONE boundary,
     placed after the LAST token of the run **[W]**.
     **v1 DEVIATION (implemented 2026-08-16):** "after the LAST token of the run"
     needs to see the NEXT token, so it cannot be decided causally, and §6 plus
     invariant 1 both require the identical rule at generation. The
     implementation therefore places the boundary after the **FIRST** token of
     the run and lets rule 3 absorb the rest into the following span — a `.`+`\n`
     pair still produces exactly ONE boundary, which is what this rule was for.
     The only difference is which span owns the trailing `\n`. Measured on 3.0 M
     OWT tokens the segmentation still matches the numbers below: mean span 19.9
     (spec 19.2), 27 % hit the cap (spec 26 %), 1.0 % one-token spans.
  3. A boundary is **suppressed** while the current span has fewer than
     `min_span = 4` tokens (the span continues; the short piece merges into
     the same span). This is the causal form of "merge spans < 4 into the
     previous"; at generation it reads "do not insert a slot yet".
  4. A boundary is **forced** when the span reaches `span_cap = 32` tokens
     **[W]**.
  Measured on MORPH's OWT shards (starcoder2-7b): raw rule gives 37.9% one-
  token spans (`.` `\n` pairs); after 2–4: mean 19.2, median 19, p90 32,
  2.3% one-token, 26% hit the cap. Code (lines): mean 9.2. See MORPH-READ.md.
* **Fixed shapes.** `L_total = seq_len_tokens + prefix_k · max_slots` is constant per
  stage; the packer (`curriculum_data.py` carry-split) fills a row until
  `tokens + slots == L_total`, so token count varies per row (log
  `tokens_per_batch`; BLT §4.3 says hold it constant across arms — hold it
  in expectation and report it). `max_slots = seq_len // 8` (2048 → 256):
  code at 9 tok/span needs ~228; with `prefix_k 2` that is 512 slot positions
  in `L_total 2560`. If a row would exceed `max_slots` the packer
  ends the row early; it never drops a boundary.
* Positions/RoPE: tokens and slots share the sequence coordinate in prelude
  and coda. Inside the core the gathered slot sequence uses slot-index
  coordinates 0..N−1 (Block Transformer's block decoder does the same).
* Attention masks: causal everywhere, as today. Slots see all earlier
  positions (tokens and slots) in the prelude and coda; tokens see slots.
  The window (128) covers ~6 slots of prose; CSA/HCA cover the rest.
* Documents: EOS is a boundary; a slot follows EOS like any other boundary.
  Nothing crosses a document boundary that does not already in MORPH.

### 3.2 Prelude — unchanged

All positions, all existing injections. The slot's input embedding is
`E_slot + mean_j embed(t_j)` over the span's tokens (bigram/value-embed
signals for the slot are the bag-mean, exactly the TST `ve_bagged` path).

### 3.3 Core — slots only, per-slot depth

* `_forward_single`: after the FRONT region, **gather** slot positions into
  `[B, max_slots, n, C]` (padded; pad slots masked out of the loss and placed
  at the END so causal attention over the compact sequence is unchanged for
  real slots). `e = input_norm(x_slots)`, `h = e.clone()`; the loop runs
  exactly as today on the compact sequence; **scatter** `h` back to the slot
  positions of the full carrier. Token positions of the carrier keep
  `input_norm(prelude state)` (the seed-model path).
* **Per-slot depth [W]:** `depths ~ Poisson(mean_depth)` of shape
  `[B, max_slots]`, clamped `[1, max_depth]`, eval = `mean_depth`. Update is
  `h = where(active[..., None, None], h_new, h)`. Per-sample active-set
  shrinking is replaced by a per-slot mask; frozen slots are still computed
  and still serve K/V (MORPH recomputes K/V from the current carrier each
  iteration; a per-position gather would change what frozen slots' keys are
  and is not exact). The compact sequence is 9–19× shorter than tokens, so
  the lost shrink is affordable. `bptt_depth`, `ckpt_grad_iters`, GLA carry,
  x0/bigram/value-embed injection terms, `core_gain_clip`: unchanged.
* **Learned per-slot exit at inference — specified, deferred.** Poisson depth
  per slot makes the model robust to variable depth; an exit head (Huginn
  KL-exit style: stop when the coda readout of `h_i` at iteration t and t−1
  agree under KL < ε) can be added without retraining the core. Not in v1.
* `n_core == 0` continues to mean "no core at all" (seed models); with TUL on
  it means tokens AND slots skip the core (a degenerate arm; keep it legal).

### 3.4 Coda — all positions, token-state dropout, slots as prefix

* Input: token positions `input_norm(prelude)`, slot positions `h_i`. The
  existing `_back_region` runs over the full carrier; nothing new in the
  block. The coda attends to slots because they are positions.
* **Token-state dropout** `p` (Bowman word dropout, He/Optimus "tax the cheap
  channel"): with probability `p` per token position (training only), the
  coda input is replaced by a learned `E_mask` vector (broadcast over HC
  streams). The position must then be decoded from the plan slots and its
  neighbours through attention. Default `p = 0.15`, arm sweep {0, 0.15, 0.3}.
  Never applied to slot positions.
* Slot positions carry two functions with one state: `h_i` is refined by
  four coda layers into the first-token logits (its label), and its coda K/V
  are what later tokens read. This is MegaByte's `p = 0` position and Block
  Transformer's last prefix. If the first-token loss is found to crowd out
  the plan (the LTD think-position failure, t+1 0.339 vs t+2 0.108 at one
  position), `prefix_k = 2` (default) gives one loss-free prefix position
  before the emitting one, by projection (§3.1), not by a second looped slot.
* Labels: `label(z_i) = t_1(i+1)`; `label(t_last(i)) = t_1(i+1)` as well (a
  second, plan-free prediction of the same token — the counterfactual §7.2
  needs and generation ignores). Both terms are weighted 0.5 so first tokens
  are not counted twice in the loss (or `t_last`'s term is computed under
  `no_grad` as a metric only — implementation choice, state it in the config).
  Other tokens: next token; pad slots and pad prefix positions: −100.
* HC mean → `lm_mixer` → `final_norm` → fused CE with the `slot_id` logit
  masked. Unchanged otherwise.
* Norm balance (H-Net §2.3): the seams are `input_norm` (already RMSNorm) on
  the token path and the HC carrier on the slot path; the HC residual is
  norm-preserving by construction. No new norms in v1; if slot and token
  coda inputs drift apart in scale (log `‖h_i‖ / ‖input_norm(prelude)‖`), add
  an RMSNorm on `h_i` at the scatter (H-Net "network normalization").

### 3.5 What is explicitly NOT in v1 (arms or deferred)

| item | why not default | source | status |
|---|---|---|---|
| per-layer cross-attention branch to slots (`attach_xattn`, like `attach_retention`) | prefix route wins in the only BPE-level ablation (BT Fig 3f); coda attention already reaches slots | BLT T7 for the counter-case | arm `xattn` |
| `W·h_{i-1}` explicit carry into slot i+1's input | attention over earlier slots is the same channel; CCoT: last-layer feedback ≈ nothing, mid-layer works | Coconut §3, CCoT | arm `carry` |
| per-offset embedding as the coda's ONLY token input (replacing the token stream) | weakest injection in BT / Hourglass / MegaByte; caused the Huginn collapse | §1 | never |
| MegaByte per-offset slices of the global vector as the only input | same | MegaByte Eq. 4 | never |
| `bcast`: `h_i` repeated over span i+1's token positions through one of `span_cap` offset-indexed linears, init 0, ADDED to the coda token input (`input_norm(prelude)`), on top of the prefix | AU-Net `up()` (Sec 2.1.1, Table 4): harmless at 2 levels (62.9 vs 63.5), needed at 3 (60.6 vs 66.0). Adds nothing the prefix does not carry at one level; kept as an arm because it is the one injection the "never" row above does not cover | AU-Net; H-Net (residual-path linear init 0) | arm `bcast` |
| fixed-stride slots (`stride = 19`, mean span matched, no punctuation rule) | the control that separates alignment from depth: SpaceByte Table 1 (+0.10 bpb), HAT Table 1 (−2.7 HS), BLT §4, DTP T2 | SpaceByte §3.2 | arm `stride` |
| learned boundaries (H-Net router + ratio loss) | rule-based is close (BLT, DTP), and TUL's boundary must be causal | H-Net §2.2 | later |
| learned per-slot exit | needs the trained model first | Huginn KL-exit | later |
| inference engine port | separate work; eager generation for the test | — | later **[W]** |
| `prefix_k = 1` (one coda position per slot, the plan and the first-token label on one vector) | Block Transformer Fig 3f: length 1 loses to 2–6; LTD think-position conflict | BT Fig 3f | arm `prefix1` |
| slot-set multi-hot warm-up (`set_lambda`) | block-level aux losses hurt in BT §4.2; TST validated MCE only as a phase-1 objective | TST; CODI/CCoT for the need | arm, default 0 |
| punc-STP on the slot trajectory (`stp_lambda`) | zero params; MORPH punc-STP finding (next token ~80% decodable from the boundary state); the STP paper itself has no boundary or pretraining claim | STP; MORPH punc-STP | arm, default 0 |

## 4. Data and segmentation

* Shards: `data/pretok/<src>/tokens.u16.bin` (unchanged). No new shard file:
  the boundary rule is a pure function of ids, so the loader computes it per
  row on CPU (vocab-mask lookup, run collapse, min-span/cap scan) and emits
  `(input_ids [B, L_total], labels [B, L_total], slot_mask [B, L_total],
  slot_index [B, max_slots])`. `slot_index` is the gather map for §3.3 (the
  FIRST of each slot's `prefix_k` positions); `slot_mask` marks all of them.
* Data **[W]**: OpenWebText with the StarCoder2 tokenizer — the `base.yaml`
  arrow path (`data.py::create_dataloader`), NOT the pretok curriculum blend.
  The slot layout is built in that loader; the curriculum loader shares the
  same layout function. Short schedule for the 5090 arms (`tul_short.yaml`,
  2026-08-16 [W]): `seq_len 1024`, `batch_size 16`, 20k steps = 328 M token
  positions (was 2048 × 8 × 100k = 1.64 B, ~33 h/arm; now ~5–6 h/arm).
* Boundary stats logged per batch: spans/row, mean span, one-token fraction,
  fraction hitting the cap, tokens/row, tail-pad fraction.
* **v1 notes (implemented 2026-08-16).** The loader emits a 5th tensor, `bag_id
  [B, L_total]`: a token position carries the index of the slot that closes its
  span, a slot position carries its own index, and everything else carries a dump
  bin. One tensor then drives both halves of the §3.2 bag-mean (summed over token
  positions, read back at slot positions), which needs a span→slot map that
  `slot_mask` and `slot_index` alone do not give. **Tail padding:** when the next
  unit does not fit in the remaining room the row ends and its last ≤ `prefix_k`
  positions become inert pad slots (input `slot_id`, label −100). This keeps
  `L_total` exact WITHOUT dropping a boundary inside the row; measured cost on OWT
  at `max_slots 64` is 1.18 % of positions, and it is logged as `tul/pad_frac`.
  **`max_slots` is sized from the data, not from `seq_len // 8`** (which is sized
  for code at ~9 tokens/span): on OWT at `seq_len 1024`, `max_slots 64` gives
  1033 tokens/row — matching A0's 1024, the tokens-per-batch control BLT §4.3
  asks for — where `seq_len // 8 = 128` gives 1161 and does not fit in 32 GB.
* Val/gen: `bag_size = 0` and TUL layout ON (val PPL is over token positions
  only, so it is comparable to the baseline's token PPL; slot positions'
  first-token CE is reported separately as `val/first_tok_ce`).

## 5. Training schedule and losses

* **5090 arms (`tul_short.yaml`) [W, 2026-08-16]: TST OFF, TUL layout on
  from step 0** (`tul.activate_at 0.0`, `tst_bag_size 0`), and **prune →
  carve → route OFF** (dense backbone; QAT and AdEMAMix on). Reason: the
  question is whether the slot loop works, and a staged topology change is
  one more thing to attribute a loss jump to. `ademamix_t_alpha` scales
  with the run (1600 of 20k = the same 8 %). Measured: A0 on this file runs
  0.99 s/step, 22.96 GB peak → 20k steps ≈ 5.5 h.
* **Full-schedule variant (kept as config, later):** phase 1 = TST
  superposition on bags, no slots (bit-identical to today); at the TST switch
  TUL layout on (`bag_size 0`, `slot_layout` present), coinciding with
  `route_start` (carve 29k, route + TST switch 30k) — three transitions at
  one step. Expect a loss jump larger than the TST switch alone (the coda's
  input changes from looped states to prelude states for token positions).
  Block Transformer §3.7 is the datum FOR a mid-run switch: uptraining a
  vanilla checkpoint into the block/token split recovers near-full
  performance with ~10% of the tokens (init block embedding = mean of token
  embeddings, prefixes = replicated context embedding).
* **CMS ordering (full-schedule variant only):** prune → carve (27–29k)
  completes BEFORE TUL activates (30k) and the mask comes from token-position
  saliency; the short schedule has no prune/carve/route. Recorded so a later
  regression is not misread.
* New parameters: `E_slot` (init: mean of the embedding table), `E_mask`
  (init 0), `W_1..W_prefix_k` (init identity-scaled). No new blocks. In the
  full-schedule variant their AdEMAMix state starts at the switch.
* Loss: `L = CE_tokens_and_slots` (fused, chunked; `slot_id` masked)
  `+ stp_lambda · STP_slots` (arm) `+ set_lambda · MCE(next-span token set | h_i)` (arm).
  STP_slots = `forward_step_boundary` on the slot trajectory (consecutive
  slot states colinear), zero params, applied to the head-input latent at
  slot positions.
* Everything else unchanged: ternary/int6 QAT, AdEMAMix β1=0, flat LR;
  prune → carve → route in the full variant only (CMS saliency then scores
  the core on slot positions — note in the ledger). Per-stage LR (H-Net App. C: outer stages higher,
  by tokens processed and width) is a knob to add if the core under-trains
  on 9–19× fewer positions; log per-group update norms.

## 6. Generation (eager, v1)

```
state: KV for prelude+coda sites over all positions; core (layer,iter) sites over slots
loop:
  emit token from the last position's coda logits (slot_id masked)
  run prelude → coda on the new token (no core)
  span_len += 1
  if (token ∈ B and span_len ≥ min_span) or span_len == span_cap:
      insert slot: input = E_slot + mean(embed of the span's tokens)
      prelude on the slot; core × T (T = mean_depth in v1; exit head later); coda on the slot
      emit the first token of the next span from the slot's logits
      span_len = 0
```
Train/generation parity: the boundary rule, `min_span`, `span_cap` and the
run-collapse are the same function at both ends (one implementation,
`morph/model/tul_layout.py`, called by the loader and by the generator).
Test: teacher-force a generated layout and assert the slot positions match
the loader's layout exactly (the coconut `assert_layout_parity` lesson).

## 7. Evaluation, arms, gates

### 7.1 Arms (all at the same tokens/batch, same steps, same seed policy)

| id | name | slots | tokens through core | coda sees slots | depth on slots | what it isolates |
|---|---|---|---|---|---|---|
| A0 | MORPH baseline | no | yes (Poisson/sample) | — | — | reference |
| A1 | **TUL** | yes | no | yes | Poisson/slot | the method |
| A2 | slots-as-memory | yes | yes | yes | Poisson/slot | C2 alone (plan readable, uniform depth) |
| A4 | depth-only | yes | no | **no** (slots masked from coda attention) | Poisson/slot | C1 alone (depth per idea, plan unreadable) |
| A3 | shallow control | no | no (seed path) | — | — | compute floor: what prelude+coda alone do |
| A1r | TUL repeat | as A1 | | | | **retrain noise floor** |
| A1+ | TUL reinvest | yes | no | yes | Poisson/slot, `mean_depth 12` | the layer-pass savings spent: `n_coda 8`, deeper loop, still ≤ A0's layer-passes/token |
| A5 | fixed stride | yes, every 19 tokens | no | yes | Poisson/slot | alignment vs depth (SpaceByte Table 1, HAT Table 1): A1 − A5 is what the boundary rule buys |

**First pass [W, 2026-08-16]: A0, A1, A1r, A3 only** (~5–6 h each on
`tul_short.yaml`; A3 is cheaper). A2/A4/A1+/A5 and the sweeps are queued
behind the "works" gate; `plan_nats` (slots gathered out at eval) already
measures whether the plan is used without training A4.

C1 = depth per idea beats depth per token at equal layer-passes; C2 = a span
is decodable from the plan. The 2×2 is {A0/A2 × A4/A1}; A3 is the floor. Read
per CoCoMix Fig 6(d): if A1 does not beat both A2 and A4, one knob is
carrying the result. A1 at iso-params runs 8 layer-passes/token against A0's
44, so it will trail A0 on token PPL (Block Transformer needed 2–3× params to
match vanilla); A1+ is the fair-compute cell and A3 is the floor A1 must
clear by `plan_nats`.

### 7.2 Metrics

* `val/ppl_tokens` (token positions), `val/first_tok_ce` (slot positions),
  per-offset-in-span CE curve (Block Transformer Fig: first token hardest).
* **`val/plan_nats`** = CE over span tokens with the slots REMOVED from the
  coda's sequence at eval (a gather that drops slot positions — exact, and
  it needs no per-position attention mask the fused kernels may not have)
  minus the normal CE (the h_z-ablation; the C2 number). Reported with the
  A1r spread. Arm A4 is trained the same way (slots dropped from the coda).
* **`val/first_tok_counterfactual`** = CE(t_1 | t_last, no plan) − CE(t_1 | z_i)
  from the double label — free, per batch.
* Generation: rep4@512, distinct-3, mean span length, fraction of spans
  ending by cap, samples (seeded).
* Boundary stats (§4). Layer-passes/token and tokens/s per arm.
* Later: future-lens probe on slot states with `teacher_acc` ceiling.
* Do NOT size the coda (`n_coda`, A1+) by per-token accuracy inside a span:
  Hierarchical AT Fig 3 shows a bigger char decoder raises byte accuracy
  and leaves word accuracy flat — per-token accuracy rewards span
  COMPLETION, not prediction. Size it by `val/first_tok_ce` and
  `val/plan_nats`; report whole-span exact-match as the word-accuracy analogue.

### 7.3 Pre-registered gates (margins set before the runs)

* **Works:** `plan_nats > A1r spread`, and A1's rep4@512 ≤ A0's, and no
  collapse in samples (span length distribution within ±30% of the data's).
* **C1:** A4 beats A3 by more than the spread at iso layer-passes.
* **C2:** A2 beats A0 by more than the spread at iso layer-passes.
* **Method:** A1 beats both A2 and A4.
Beating A0 on `val/ppl_tokens` is NOT a gate at this scale (§1).

## 8. Config (proposed keys, `morph/configs/base.yaml` block `tul:`)

| key | default | note |
|---|---|---|
| `tul.activate_at` | 0.0 (`tul_short.yaml`) | fraction of steps; `= training.tst_ratio` in the full-schedule variant; `never` = plain MORPH (A0) |
| `tul.slot_token` | `"<fim_pad>"` | resolved to id at build; asserted absent from shards |
| `tul.boundary_chars` | `".;!?"` + newline + dashes | resolves `B` from the tokenizer |
| `tul.min_span` | 4 | suppress boundary below this |
| `tul.span_cap` | 32 | force boundary at this **[W]** |
| `tul.max_slots` | `seq_len // 8` | fixed-shape slot budget |
| `tul.prefix_k` | 2 **[W]** | coda positions per slot, by projection from one looped state (§3.1); 1 is the arm |
| `tul.token_state_dropout` | 0.15 | arm sweep |
| `tul.slot_mean_depth` / `slot_max_depth` | `= mean_depth` / `max_depth` | per-slot Poisson |
| `tul.coda_sees_slots` | true | A4 sets false |
| `tul.tokens_through_core` | false | A2 sets true |
| `tul.stp_lambda` / `tul.set_lambda` | 0.0 / 0.0 | arms |
| `tul.carry` / `tul.xattn` | false / false | arms |
| `tul.bcast` | false | arm: offset-indexed linears, `h_i` added to span i+1's coda token input (§3.5) |
| `tul.fixed_stride` | 0 | arm A5: >0 replaces the boundary rule with a slot every N tokens (§7.1) |

No runtime flags in the forward: `slot_layout` is a per-forward argument like
`bag_size` (None → plain MORPH path); `coda_sees_slots` and
`tokens_through_core` are construction-time (they change masks/gather, not
`if` branches inside the hot loop).

## 9. Invariants added (also in `runtime-invariants.md`)

1. Boundary rule, `min_span`, `span_cap`, run-collapse: ONE function, used by
   the loader and the generator; parity-tested.
2. Per-slot depth is a masked update over the full compact slot sequence;
   never a per-position gather.
3. Slot positions have no loss on their core state; their only labels are
   the first token of the next span (coda output). Pad slots are −100.
4. `slot_id` is masked from the LM head everywhere.
5. `L_total = tokens + prefix_k · slots` is fixed per stage; token count varies per row.
6. Val/gen always run TUL layout on and `bag_size 0`.

## 10. Risks, honestly

* Token-level hierarchy is the least-proven link: BLT/MegaByte/H-Net win at
  bytes; Block Transformer at BPE needs 2–3× params to match vanilla PPL and
  buys decode throughput. Prose spans here are ~19 tokens; BT's L_B=8 already
  costs 0.43 nats at 85M. TUL's coda has cross-span context (BLT regime),
  which BT's local decoder did not; that is the bet.
* Short schedule runs dense (no MORTAR result carries over); the
  full-schedule variant has three transitions at step 30k (carve+1k, route,
  TST switch, TUL).
* The core trains on 9–19× fewer positions than before; may need per-stage LR.
* The causal id-rule mis-cuts abbreviations/decimals (documented in
  `punct_boundary.py`); accepted for v1, measured by boundary stats.
* Fixed-shape padding of the slot sequence wastes compute on prose rows
  (~108 real of 256).

## 11. Provenance — decision → source

| decision | source | evidence |
|---|---|---|
| loop over slots, tokens skip the core | MegaByte §2 (global over patches, local per byte); H-Net §2.1; BLT §3 | hierarchy wins at matched compute (MegaByte T2, H-Net T1) |
| loss-free slot state | MegaByte, H-Net, LD4LG, Pred-Sent, Coconut | six independent instances |
| small AR decoder that sees previous tokens | MegaByte T7 (0.687 vs 1.263), Hourglass T6, DeepSeek-V3 §2.2, Bowman T2 | removing it is catastrophic |
| latent as attended prefix positions, refined by decoder layers | Block Transformer Fig 3f; Hourglass (attention upsampling); Coconut §3; Optimus Fig 5 | prefix > cross-attn (BPE, unit-only decoder); memory > embedding-add |
| decoder input = encoder token states | BLT Eq. 9 (`D_0 = h_lE`); MORPH `n_core == 0` path | — |
| slot input = mean-pooled span embeddings | TST (`data.py`); Patch-level training; DTP (mean > last); BLT Eq. 5 (mean-pool init) | — |
| token-state dropout | Bowman T2 (word dropout), He 2019 §3.1, Optimus free bits | tax the cheap channel or the latent is ignored |
| no regression onto the latent | LCM T3/4, Pred-Sent §3.2, CoCoMix Fig 6b, Block Transformer §4.2 | loses every time |
| punctuation boundary, causal, collapse/merge/cap | BLT §4, DTP T2, H-Net T1, LCM (200-char cap), CCoT (punct states) | rule ≈ learned; fixed stride worst |
| per-slot Poisson depth | Parcae (per-sequence Poisson); [W] | C1 needs depth per idea |
| TUL from step 0, TST off (5090 arms); activate at TST switch (full variant) | [W] 2026-08-16; Patch-level training (patch phase then token phase) for the full variant | — |
| 2×2 arms + repeat | CoCoMix Fig 6(d); coconut noise-floor findings | attribution |
| plan-nats ablation as the C2 metric | Kaiser T4 (oracle code vs predicted), He Fig 5 (MI not KL) | measure usage, not loss |
| STP on slot trajectory (arm) | STP Eq.; MORPH punc-STP note in `references.md` §7 | MORPH finding, not the paper's |
| slot-set MCE warm-up (arm, off) | TST MCE; CODI/CCoT/Coconut (untargeted slot learns nothing) vs BT §4.2 (aux hurt) | contested → arm |
| do not per-offset inject as the ONLY input | Block Transformer, Hourglass, MegaByte, Huginn 2026-08-16 | weakest everywhere |
| deep compute at boundary positions only, read back through the causal token stream | SpaceByte T1/T6, AU-Net T2/T5, Hierarchical AT T1 | ties BPE at matched compute (word scale); depth belongs at the coarse level (AU-Net T5: 75 % > 50 % > 25 %) |
| fixed-stride control arm A5 | SpaceByte T1 (+0.10 bpb), HAT T1 (−2.7 HS), BLT §4, DTP T2 | the only arm that isolates alignment from depth |
| `bcast` broadcast-add arm, off by default | AU-Net T4 (2-stage tie, 3-stage +5.4), H-Net §2.3 (init 0) | matters only with ≥3 levels |
| coda sized by first-token / plan metrics, not per-token accuracy | Hierarchical AT Fig 3, App A.5 | per-token accuracy rewards completion |
| slot-only LOOP has no precedent | SpaceByte §2 (cites ACT, MoD as layer skipping only); AU-Net; HAT | TUL's own claim, unsupported by prior art |

Sources: `docs/references.md` §13 and `docs/references/tul-latent-emission/`.

## 12. Tests to write with the implementation

* layout: rule → slots; run-collapse; min_span suppression; cap force; parity
  loader vs generator on 1,000 rows; `slot_id` absent from shards.
* gather/scatter round-trip on the HC carrier; pad slots never receive loss.
* per-slot depth: a slot with depth d is bit-identical to a per-sample run of
  depth d when all slots share d (reduces to today's path).
* `slot_layout=None` → forward bit-identical to today (the TST phase).
* fused CE with `slot_id` masked: gradient to that row is zero.
* mutation tests: (a) coda cannot see slots → `plan_nats` must read ~0 on a
  trained checkpoint; (b) generator ignoring `min_span` → parity test red.
