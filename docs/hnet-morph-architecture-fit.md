# MORPH architecture vs H-Net: what carries over

Status: written 2026-10-29 for Wolfe's H-Net project.

## How to read this

This doc compares each MORPH part with what an H-Net (Hwang et al., arXiv 2507.07955) needs at the same position. Each part gets one label.

- KEEP. The part works in an H-Net as it is.
- ADAPT. The part works after a named change.
- REPLACE. H-Net has its own version. Use that one.
- DROP. The part only made sense with BPE tokens or slots.

Evidence tags: "measured" means we ran it and filed the result. "code fact" means I read the code. "untested" means nobody ran it on bytes or chunks. Every CE verdict in `docs/lessons-learned-loops-and-spans.md` is inconclusive by Wolfe's rule, so I use only its mechanisms, theorems, pitfalls and systems facts. The H-Net facts come from `docs/references/tul-latent-emission/h-net/h-net.md`. Where that note lacks a fact, I say "not in our note". I did not run code, tests or the GPU.

## Summary table

| part | label | one-line reason |
|---|---|---|
| Hybrid Euclidean + Lorentz embedding | DROP | A byte model has 256 rows, no vocabulary tree to embed. Lorentz math in the router or dechunker would be new code. |
| Hash-bigram embedding | ADAPT | Becomes a byte n-gram hash on the outer encoder input (BLT does this). |
| Weight-tied head, fused chunked CE | DROP | Logits are [N, 256], so the memory problem is gone. Keep the CE-at-init check. |
| StarCoder2 tokenizer, data pipeline | REPLACE | Needs a byte loader and bits-per-byte accounting. |
| TST token superposition | DROP | Fixed-stride bags conflict with learned boundaries. Untested as a byte warmup. |
| TUL boundary rule | REPLACE | The router learns boundaries. Keep the rule as a control and a bootstrap. |
| Row packer, SlotLayout | ADAPT | The fixed-capacity padded layout carries over. Slot insertion does not. |
| Slot positions, slot loop, gain constraint | DROP | H-Net has no slot token. The chunk head vector is the state. |
| Pointer/copy head | ADAPT | Copying drove most of our cross-span gap. A byte decoder may need it. Untested. |
| CCA/CSA/HCA attention, XSA, CoPE, QK-norm | ADAPT | Good for the inner chunk stage. Not for the byte stage. |
| GLA retention | ADAPT | Same family as Mamba-2. Never tested as a main mixer. Causal leak pitfall applies. |
| Cayley Hyper-Connection, n = 4 | KEEP | Already is the "Cayley over 4 streams" the paste asks for. Costs memory at byte length. |
| Parcae prelude / looped core / coda | ADAPT | Maps onto encoder / main network / decoder. The loop belongs on the main network only. |
| DiagonalInjection | KEEP | Re-supplies the encoder chunk vector each pass. Sets a 0.87 gain floor. |
| Poisson depth | ADAPT | Per-sequence depth is fine. Per-chunk depth would be new. |
| Fixed-point term | KEEP | Cheap and held detonation on the plain loop. |
| Slot gain constraint | DROP | Acts only inside the slot loop. |
| Ternary QAT with norm_match | ADAPT | Use on the main network. Keep the router, norms and Mamba/GLA gates in bf16. |
| MORTAR block sparsity, carve, ReMoE routing | ADAPT | Weight tiles do not care about ragged chunks. Two routers will compete for aux losses. |
| Compile and kernel stack | ADAPT | Pad chunk count to a fixed cap so compile and CUDA graphs still work. |
| KV-cached generators | REPLACE | H-Net decode steps three recurrent states. Our generators target the strict slot loop. |
| wandb and Hydra infra | KEEP | No change needed. |
| Instruments (K-curve, core_jacobian, tripwire) | ADAPT | Keep the tripwire and Jacobian. Judge by bits per byte, not by K-curve. |

## The big picture: Parcae already is a 1-stage H-Net with identity chunking

Parcae has three parts. The prelude runs on every token. The core loops on every token. The coda runs on every token. An H-Net with one stage has three parts too. The encoder runs on every byte. The main network runs on chunks. The decoder runs on every byte.

So the map is direct.

```
MORPH / Parcae                      H-Net (1 stage)
prelude (4 layers, every token)  -> encoder (4 Mamba-2 layers, every byte)
looped core (6 layers x T)       -> main network (chunks only)
coda (4 layers, every token)     -> decoder (4 Mamba-2 layers, every byte)
DiagonalInjection (prelude -> core, each pass)   -> no H-Net twin
coda reads h0 + u ("Anchor")     -> encoder-to-decoder residual
```

The difference is the downsampler and the upsampler. In Parcae the chunk length is one token everywhere, so nothing shrinks. Every lesson we learned about the prelude/core/coda split therefore transfers to the encoder/main/decoder split. That is the most useful single fact in this doc. The "decoder residual is a bypass" section below uses it.

Two more H-Net facts matter for the rest of this doc, all from our note.

- The router makes p_t from cosine between q_t = W_q x_t and k_{t-1} = W_k x_{t-1}. A boundary fires at p_t >= 0.5. The downsampler keeps the vector at the boundary byte, so a chunk is represented by its head byte, which the encoder has already seen causally.
- The dechunker smooths with z_bar_t = P_t z_t + (1 - P_t) z_bar_{t-1}, then copies the chunk value to every byte of the chunk, then multiplies by an STE of the router confidence c_t. A ratio loss with weight 0.03 steers the boundary rate. The learning rate is scaled up for outer stages. An RMSNorm closes each of encoder, decoder and main network. The residual from encoder to decoder gets a Linear projection and starts near zero.

## Per-part sections

### Hybrid Euclidean + Lorentz embeddings

- What it does. The token embedding splits d_model into 75% Euclidean and 25% Lorentz channels (`lorentz_fraction: 0.25` in `morph/configs/base.yaml`). The Lorentz part is stored as space coordinates, lifted to the hyperboloid, then log-mapped at the origin back to a tangent vector (`morph/model/embeddings.py:27-84`). After that point everything is ordinary Euclidean math. The same table doubles as the tied LM head.
- H-Net fit: DROP for the byte table.
- Why. An H-Net has a 256-row byte table. A vocabulary tree needs many rows to show a hierarchy. 256 rows have none. The hierarchy in an H-Net is built by the chunking operators, not stored in a lookup table.
- Evidence. The log-map code fact: the Lorentz channel acts as a norm squash, since the map coefficient asinh(r)/r makes embedding norms grow like ln r (`embeddings.py:41-84`). I found no ablation row in `docs/ablation-ledger.md` that measures a gain from the Lorentz channel against an all-Euclidean table. Its only Lorentz line is the int6 rule that keeps it in bf16. So the benefit is untested.
- What carries over to the paste. The paste wants Lorentz math inside the router and dechunker. MORPH has no such code. Only the embedding table is hyperbolic, and only up to the first log map. Anything Lorentz in an H-Net would be new code and untested.

### Hash-bigram embeddings

- What it does. A hash of (token, previous token) indexes a 49152-row table. A per-layer scalar lambda, initialized to zero, scales the table and adds it into the residual at each layer (`embeddings.py:227-301`, `bigram_hash_vocab: 49152`).
- H-Net fit: ADAPT.
- Why. A byte alone carries little. The outer encoder needs short-range context fast. A byte pair has only 65536 values, so a hash is not even needed for n = 2. BLT hashes byte n-grams of size 3 to 8 into fixed tables for this reason (our note `docs/references/tul-latent-emission/blt/blt.md`, section 3.2.1). The H-Net note does not say whether the paper uses n-gram tables; that is "not in our note".
- Change. Inject only at the encoder input, not at every layer. The existing code adds the bigram term at all layers including the coda, and `model/CLAUDE.md` warns that anything which hides identity from the coda must also zero these terms. Hidden decoder input is not a goal here, so inject once.
- Evidence. Code fact only. Untested on bytes.

### Weight-tied LM head and fused chunked CE

- What it does. The LM head is the embedding table (`HybridEmbedding.lm_weight()`). `morph/model/fused_ce.py` computes the loss in row chunks so a [N, 49152] logit tensor never exists (`fused_ce.py:1-40`).
- H-Net fit: DROP the fused CE. KEEP the sanity checks.
- Why. With 256 classes, [N, 256] fp32 is 192 times narrower in vocabulary. Rows grow about 4.6 times, so logit memory falls by a factor near 40. The kernel solves a problem that no longer exists. Plain `F.cross_entropy` is enough. Weight tying a 256-row table saves nothing.
- Evidence. Code fact. The lesson that does carry over is the pitfall "CE above uniform" (lessons section 4): check the loss at init against ln V. For bytes that is ln 256 = 5.545 nats.

### StarCoder2 tokenizer and data pipeline

- What it does. `morph/training/data.py` streams OpenWebText, tokenizes with `bigcode/starcoder2-7b`, and yields (input, label) rows of `seq_len` 4096 tokens. TUL rows are packed by `pack_tul_batch` (`data.py:1-50`).
- H-Net fit: REPLACE.
- Why. An H-Net reads raw UTF-8 bytes. The paper uses 8192 bytes per sequence for byte models against 1792 GPT-2 tokens for the Transformer baseline (our note, Table 1 and "Experimental Setup"), about 4.6 bytes per token. At 4096 StarCoder2 tokens the same text is likely near 19k bytes, but I did not measure the StarCoder2 ratio. Mark untested.
- Change. Write a byte loader. Compare models in bits per byte, not CE per token. Our CE@6 and K1-K6 numbers are in nats per token and cannot be compared across tokenizers.
- Keep. The tokenizer can still build a boundary lookup table for the heuristic control (see boundary rule). The H-Net note also lists a distillation recipe from token models to byte models in its Appendix F. I did not read it in depth.

### TST token superposition

- What it does. For the first part of a run, s tokens are averaged into one input position. The loss is multi-hot over the next s tokens. The net label shift is s (`data.py` docstring, `train.py:2432-2440`, `tst_bag_size: 6`).
- H-Net fit: DROP.
- Why. TST packs a fixed stride of s tokens into one position. That is exactly the fixed-width chunking that the H-Net note calls ineffective: "H-Net (pool) is much worse than all other H-Net variants, validating that fixed-width chunking is not effective" (our note, Large Scale). It also needs a causal shift of s that a learned boundary cannot honor, since a bag must end where a bag ends. Hourglass makes the same point with a right shift of k - 1 before every shortening (`docs/references/tul-latent-emission/hourglass/hourglass.md`, lines 66-96).
- Evidence. Code fact and the H-Net note. A TST-like phase on the outer encoder as a cheap early speedup is untested. It would need a design that does not touch the router.

### TUL boundary rule

- What it does. A deterministic causal state machine cuts spans after a token ending in `.;!?` or containing a newline or a dash. No comma. Minimum span 4, cap 32. One boundary per run of boundary tokens (`morph/model/tul_layout.py:1-80`, `BoundaryRule.cut`).
- H-Net fit: REPLACE with the learned router. KEEP the rule as a control and as a bootstrap target.
- Why. H-Net learns where to cut from adjacent-state similarity and a ratio loss. The rule has the nice property that the same code serves the loader and the generator. A learned router also serves both, because p_t depends only on the current and previous encoder outputs.
- Collision with our own docstring. The `tul_layout.py` header says deterministic boundaries are "within noise of learned ones" and cites "H-Net Table 1". Our H-Net note says the opposite for H-Net itself: "H-Net (1-stage) is stronger than H-Net (space), validating that our dynamic chunking mechanism successfully learns how to segment data in a context-dependent way that improves over strong heuristics" (Large Scale). Do not cite the docstring claim. The learned router beats a space heuristic in the paper, and the gap grows with data.
- Level mismatch. Our spans are sentence-sized. `max_slots` defaults to seq_len / 8, and the cap is 32 tokens. An H-Net first stage targets about 4.5 to 5 bytes per chunk. The TUL rule fits a second or third stage, not the first. That level is untested.
- Alignment mismatch. TUL puts the slot after the span's last token. The H-Net chunk vector is the head byte of the chunk, available at once. They give different delays between seeing text and having a state. Hourglass documents leaks that come from getting this shift wrong.

### Row packer and SlotLayout

- What it does. `pack_tul_row` builds a fixed-shape row: tokens plus `prefix_k` slot cells per span, padded to `L_total`. Pad slots get label -100 (`tul_layout.py`, `base.yaml: max_slots`).
- H-Net fit: ADAPT.
- Why. A learned router yields a different chunk count per sequence. Any batched, compiled model needs a fixed inner length. The packer's approach is the right one: pick a cap, pad, mask the pad, and give pad positions no loss. The cap needs a rule for overflow, since the ratio loss only steers the average. Our note does not say how the paper batches ragged chunk counts. That is "not in our note".
- Change. Replace "insert slots" with "gather head-byte vectors into the first n_chunks cells of a [B, L_inner_max] buffer". Keep the pad mask, the segment ids and the lesson from `tg_segment_ids`: a causal conv or a value shift must reset at segment starts, or it leaks across segments.
- Evidence. Code fact. Segment-reset conv exists as `fused_segment_causal_conv`. Untested on chunks.

### Slot positions, slot loop and gain constraint

- What it does. One extra position follows each span. Only the slot positions loop in `_tul_core`. A gain hinge (`slot_gain_lambda: 100`, `slot_cot_clip: 4`) and `E_slot` stabilize it.
- H-Net fit: DROP.
- Why. An H-Net chunk is not a summary written into a new cell. It is the encoder's own vector at the head byte. There is no slot token to seed, no `E_slot` to collapse, and no prefix write.
- Evidence. Measured: a slot-only loop earned 0.015 nats at 20k steps while paying the loop on every position earned 0.168 at 5k (lessons 2.6). Measured: slot cells sit at rank 1.7 to 4.8 because they share one `E_slot` (lessons 3.2). Measured: the slot-cell carrier is 99.6 to 99.8 percent one shared vector (lessons 3.1). These are reasons not to rebuild slot machinery in an H-Net.
- Warning. The H-Net inner positions are the chunk vectors. Their loss is paid through the decoder, not directly. See the bypass section.

### Pointer / copy head

- What it does. `tul_pointer.py` adds a learned head that scores earlier token positions for copying (`pointer_heads` in `tul.py`).
- H-Net fit: ADAPT. Untested on bytes.
- Why. Measured: far-bigram copying was 85.7 percent of the strict-slot model's gap to plain, and a normalized pointer closed about two thirds (lessons 3.8). The reason was that a pooled state does not hold token identity. An H-Net main network sees chunk vectors, and the byte decoder is recurrent. A pooled chunk vector has the same weakness, and a recurrent decoder has no direct way to copy a rare string from far back.
- Change. Point from decoder byte positions to earlier byte or chunk positions. Add a null option, because a head that cannot abstain gave pessimistic CE (lessons section 4, "copy head that cannot abstain"). Test repetition at low temperature, because a copy head can loop on its own output.
- Evidence. Measured on slot models only. Whether an H-Net needs this is open.

### CCA / CSA / HCA attention, XSA, residual attention, CoPE, QK-norm

- What it does. Channel-compressed attention with a causal conv, QK-mean, value shift, learned temperature and QK RMSNorm, CoPE clipped RoPE. Even layers use CSA (pool m, Lightning Indexer, top-k). Odd layers use HCA (heavily compressed dense). Every layer also has a causal sliding window of 256 with the self token excluded (XSA). A gate blends branches (`morph/model/attention.py:1-25`; config `window_size: 256`, `csa_compress_ratio: 8`, `hca_compress_ratio: 256`, `top_k: 256`).
- H-Net fit: ADAPT for the main network. DROP for the byte encoder and decoder.
- Why, byte stage. The H-Net note ablates encoder layers. A pure Transformer encoder and decoder (T2-T2) was "by far the worst performance despite using more FLOPs", and Mamba layers helped even when the input was BPE tokens (our note, Encoder and Decoder Layer Selection). So the attention stack does not belong at the byte stage. A 256-byte window also covers only about 55 BPE-token equivalents of context.
- Why, chunk stage. The main network is the BPE-token analog: about 4 to 5 bytes per position, so chunk sequences are in the same regime where MORPH already runs (seq 4096, window 256). The sparse parts help only when the sequence is long. Measured geometry: at 1152 positions CSA selects every block and is just dense pooled attention; selection fires only at 4096 (`model/CLAUDE.md`, "What the core's attention ACTUALLY is"). At 8192 bytes and 4.6 bytes per chunk the inner sequence is near 1800 chunks, so CSA selection may not fire at all. Check `n_blocks` against `top_k` before trusting any sparse claim.
- Pitfall. The geometry note records that integer division gave zero compressed blocks silently when the slot axis was 64 and the ratio 256. Chunk counts that vary per row will hit this: HCA with ratio 256 needs at least 256 valid chunk positions. Guard it.
- XSA. Excluding the self token means position 0 attends to nothing and its softmax row is all -inf. In an H-Net the chunk vector already carries the head byte, so self exclusion may be wrong for the inner stage. Untested.
- Also. The attention modules hide a causal conv and a value shift that need segment resets under packing. Those exist (`fused_segment_causal_conv`).

### GLA retention versus Mamba-2

- What it does. `morph/model/gla.py` is Gated Linear Attention: S_t = diag(alpha_t) S_{t-1} + k_t^T v_t with a data-dependent per-channel gate, a chunked parallel form, a recurrent oracle, and a per-token GroupNorm output. In MORPH it is an optional side branch with gate init -6 (`retention: false` in `base.yaml`).
- H-Net fit: ADAPT. It is the nearest thing we own to the H-Net encoder and decoder layer.
- Why. The H-Net paper credits SSMs with a compression bias that suits chunking (our note, Encoder and Decoder Layer Selection). GLA is in the same linear-recurrence family as Mamba-2. I found no Mamba code in `morph/`. GLA's vector gate differs from Mamba-2's scalar decay per head. Whether GLA matches Mamba-2 as an H-Net outer layer is untested.
- Pitfall, causal leak. Two leaks cost us. First, a GroupNorm over the sequence axis pooled statistics across the whole row, so every position read the future. A trained model read the answer through it (teacher-forced 100 percent vs generation 60 percent; `gla.py:140-146`). Second, the retention carry across loop iterations let positions see later tokens, and the "0.233 nats depth earned" was that leak (lessons section 4 and 6). An H-Net adds more places for the same bug: the router EMA, the smoothing scan, the upsample gather and any norm over L. List every op that mixes positions and run a corrupt-future probe.
- Another fact. The smoothing scan z_bar_t = P_t z_t + (1 - P_t) z_bar_{t-1} is a gated linear recurrence with a scalar gate. The chunked GLA code could host it. Untested. The Triton kernel path refuses a reset mask (`gla.py` forward), so the pure PyTorch chunked path is the one that handles segments.

### Cayley Hyper-Connection, n = 4 streams

- What it does. Every block's two residuals are `HyperConnectionResidual` modules (the attribute names `mrr_attn` and `mrr_mlp` are legacy). The carrier is [B, S, 4, C]. A fused projection of the normed stream vector gives three maps: a softmax read, a softmax write and a mixer H_res. H_res is made orthogonal by the exact Cayley closed form for 4 by 4, with no inverse (`morph/model/hyper_connections.py:1-100`, `hc_streams: 4`).
- H-Net fit: KEEP, with a memory warning.
- Why. The paste proposes "Cayley across 4 residual streams" for an orthogonal inner loop. MORPH already has exactly that, in closed form, as a per-sublayer stream mixer. It is not a loop-state transition. The loop's state map is the whole pass (blocks plus injection), and its gain moves little with the mixer (see collisions).
- Byte-stage cost. A 4-stream carrier is 4 times the activation of a single stream. At 4.6 times the length, the outer stages would carry about 18 times the residual memory of a single-stream BPE model. The outer encoder and decoder should use a plain residual (`UniformResidual`, already in the file) or stay single-stream. The bf16 carrier store was free in our test (lessons 1.12).
- Evidence. Code fact for what it is. Measured: a bf16 carrier cost nothing. Untested at the chunk stage.

### Parcae prelude / looped core / coda

- What it does. 4 prelude layers, 6 core layers repeated T times, 4 coda layers (`base.yaml: n_prelude 4, n_core 6, n_coda 4`). `d_model` is 1024 in the config. The shipped paid loop runs all positions through the core.
- H-Net fit: ADAPT. Loop only the main network.
- Why. H-Net compute is concentrated in the main network on short sequences, so looping it is cheap in FLOPs. A byte stage that loops is a waste.
- Measured warnings that apply. (1) Looping earns at positions whose loss is measured through the loop (lessons 2.6). In an H-Net the loss reaches chunk positions only through the decoder, and the encoder residual offers a shorter path. (2) A span-only model, which skips the per-token loop, trained 3.8 times the tokens per second, needed about 6 times more tokens for equal CE, and trailed plain token Parcae by 0.314 nats CE@6 at equal wall clock (lessons 3.11). The lesson for an H-Net is to judge by equal wall clock and bits per byte, not by tokens per second. (3) Hierarchies are reported to lose at low data and win near large data. The H-Net paper crosses over the BPE Transformer only after 30B bytes (2-stage), 100B (1-stage) or 200B (space heuristic) (our note, XL Scale). Our 5k-step budget sits far below that.
- Evidence. Measured (listed above). The launch cost fact from section 5 of the lessons applies: looped steps are launch-bound. A short inner sequence makes kernels small, so launch cost dominates even more.

### DiagonalInjection

- What it does. Each core pass re-supplies the prelude output on a context channel slice: h_ctx = decay * h_ctx + dt * e_ctx. Outside that slice the map is identity (`transformer.py:978-1000`).
- H-Net fit: KEEP.
- Why. A looped main network needs its chunk input re-supplied every pass, or it forgets it. Washout is proved: without re-injection, distance between inputs decays like L^T (lessons 1.2, theorem `washout`). The chunk vector from the encoder is the natural thing to inject.
- Evidence. Theorem plus code fact. The map has a floor: typical gain sqrt((704 + sum A^2) / 1024) = 0.865 at the init A = 0.447, and A did not move in any model read (lessons 1.1). Report "floor plus blocks' part" before you tune any gain.

### Poisson depth

- What it does. Each sequence draws a loop depth from a Poisson with mean 6 and cap 8. The core shrinks the active set row by row as sequences finish (`transformer.py` `_core_region`; `mean_depth: 6`, `max_depth: 8`).
- H-Net fit: ADAPT.
- Why. Depth is per row, so a ragged batch with different chunk counts still works. The active-set shrink changes batch size, not sequence length, and that is the part compile and graph code already tolerate. Per-chunk depth would be new. The H-Net paper notes that inference decides compute per byte, and calls it latent compute allocation (our note, Latent Test-Time Compute), but it is a hypothesis there.
- Evidence. Measured: a clamped Poisson still runs some rows at depths 1 to 5, and a fixed-depth model is out of distribution at other depths (lessons section 4). Measured: depth gain was private to its own token (lessons 2.5), so per-chunk depth would help each chunk alone.

### Fixed-point term

- What it does. A terminal fixed-point loss (`core_fixed_point_lambda: 1.0`) on the last iterations.
- H-Net fit: KEEP.
- Why. Measured: it held the warmup-0 plain detonation, 0 of 6 against 4 of 7 controls, at no CE cost (lessons 1.11). It is blind to scale growth and to single-sample spikes, so the tripwire stays.
- Evidence. Measured, caveats as stated.

### Slot gain constraint

- DROP. It acts only inside `_tul_core`, and a plain model prints "INERT" (`CLAUDE.md`, divergence section). An H-Net looped main network is closer to the plain loop than to the slot loop.

### Ternary QAT and the norm_match rule

- What it does. Ternary forward STE with shadow weights. The `norm_match` rule scales codes by ||W||_F / sqrt(nnz) and one function serves training, the carved path and the packer (`morph/model/ternary_rule.py`).
- H-Net fit: ADAPT. Use on the main network. Keep out of the router and the scan.
- Why. Measured: absmean scale starved the looped core (K1-K6 0.033) and norm_match fixed it (0.185) (lessons 1.13). Measured: ternary raised the detonation rate (6 of 8 detonated with ternary on, 0 of 3 with it off; lessons 1.10). Rounding inside a loop is costly, FP8 forward cost 55 percent of K1-K6 (lessons 1.12). The router's W_q and W_k work on cosine of small vectors, and one wrong boundary shifts every later byte in the chunk. That makes it fragile to rounding. Mamba and GLA gate projections are similar.
- Name clash. H-Net's STE on c_t is a different STE from ternary STE. Name them apart in code.
- Evidence. Measured for the loop. Untested on a router or SSM.

### MORTAR sparsity, carve and ReMoE routing

- What it does. Prune 128 by 128 weight tiles, carve to BCSR at step `compact_step`, then gate d_ff neuron clusters per token with a product-key router (`sparsity.py`, `routing.py`, `base.yaml`).
- H-Net fit: ADAPT.
- Why. Tiles live on weights. Ragged chunk lengths change the number of token rows, not the weight tile grid, so this part is less sensitive than it first looks. The main network is where the parameters sit (the paper's 1.3B model has 24 Transformer layers in the main network and 4 Mamba layers on each side; our note, Table 1), so sparsity helps there. The outer stages stay small and dense. Dimension limits: tile 128 means widths must divide by 128.
- Competition. The H-Net paper compares to MoE and states that H-Net gains are not just sparsity (our note, Comparison to MoE). Two learned routers in one model each add an auxiliary loss: the ratio loss on chunk boundaries (weight 0.03) and the ReMoE load-balance loss. Their interaction is untested. Add routing last.
- Schedule risk. The prune, carve and route schedule runs over about 30k steps in `base.yaml`. A boundary rate that is still settling at those steps changes row counts under the carve. Untested.

### Compile and kernel stack

- What it does. `torch.compile` mode "default", fused Triton kernels for HC, CCA, CSA, HCA, GLA, CE, a step graph (`graph_step`), and a mandatory `expandable_segments` allocator setting.
- H-Net fit: ADAPT.
- Why. Fixed shapes are a hard assumption in places. The TUL path uses masking instead of shrinking for exactly this reason, and `compile_core_dynamic` exists because the token path's active set changes shape. A learned router makes the inner length data dependent. Options in order of cost: pad to a cap and mask (cheap, wastes FLOPs on pad); bucket the cap by a few lengths; use varlen kernels with cumulative sequence lengths (our note says nothing about how the paper batches; "not in our note").
- Measured facts. Compiled flex attention recompiles for every new shape and Dynamo keeps every graph. One decode tool hit 47 GB of host RAM and died (lessons section 5). Use `set_stance("force_eager")` for decode. A byte gate proves only the path it runs (PDL passed two gates and then broke training), so test at other chunk counts. Resident GPU memory runs 2.7 to 4 GB above allocated because shapes vary, so budget for it.
- Evidence. Measured. Look at launch counts: MORPH LXTUL ran 31.3k launches against 8.8k for Parcae at equal FLOPs (lessons section 5). A short inner sequence with small kernels makes this worse, so graphs matter more than before.

### KV-cached generators

- What it does. `tul_generate_cached.py` and `tul_generate_graphed.py` are exact incremental decoders for the strict slot-loop model. They raise on anything else (`_check_supported`).
- H-Net fit: REPLACE.
- Why. H-Net decode per byte steps the encoder, the router, the main network only on boundaries, the dechunker and the decoder. Its states are KV caches for attention, SSM states for Mamba, the last key for the router and the last z_bar for the dechunker (our note, Inference). None of that matches our slot caches.
- Keep. The method: write the cached decoder to produce the same tokens and log-probs as the recompute decoder, and test that equality. `tul_generate.py` recompute generator is the model for a reference oracle.
- Evidence. Code fact.

### wandb and Hydra infra

- KEEP. Nothing here depends on the tokenizer. Log the full config, including the chunk ratio targets, per-stage learning rates and the router parameters.

### Measurement instruments

- What exists. K-curve (CE at 1 pass minus CE at 6), `core_jacobian.py`, the preclip/total tripwire, `core_map_fd.py` in fp32.
- H-Net fit: ADAPT.
- Keep the tripwire. `preclip/total > 1e4` at any step of 200 or more caught 17 of 17 detonations with 0 false positives in 44 healthy runs (lessons 1.10). Test it on the first H-Net run. Keep the Jacobian probe for the looped main network, and read maps in fp32 (bf16 readings gave 0.97 against 0.884; lessons section 4).
- Do not use the K-curve as a verdict. It measures depth dependence, not depth value, and the arm's own depth 1 sets it (lessons 2.3). Add the H-Net quantities: boundary rate F and mean confidence G per stage, and bits per byte.

## The decoder residual is a bypass

H-Net adds the encoder output (through a Linear projection) to the upsampled main-network output before the decoder (our note, Separation of Two Streams). It starts the residual near zero. That is a skip around the whole inner network.

Our lessons say what a skip does to a loop.

- Bypass law (lessons 2.1). Depth use follows the share of loss that no shallower route can take. Every cheaper path around the loop takes the loop's share. The same six core blocks earned +0.0102 K1-K6 on token states and +0.0005 on slot states.
- The Anchor skip (lessons 2.3). Letting the coda read h0 plus the loop update `u` took plain Parcae's K1-K6 from +0.0377 to +0.0008, which is 47 times lower, at a slightly better CE@6 (3.7112 against 3.7256). The skip did not hurt the model. It only emptied the metric.
- Levels learn to ignore their update (lessons 3.3b). Under an Anchor skip, coarse levels of 16 to 64 tokens were worth -0.0002 to +0.004 when zeroed, once the coda could read tokens. The same coarse level was worth about 0.13 nats when it was the only far path.

What this means for an H-Net.

1. The residual is by design the path for byte detail. That is its job and it is good. But a looped main network will earn only the share of loss that the residual cannot carry. Expect a flat K-curve and judge by bits per byte at matched wall clock against an unlooped main network.
2. The H-Net note's own caution fits: the residual is "initialized close to 0" so that the main network is not skipped at the start. Keep that. Our Anchor result says zero init alone does not stop the model from learning to bypass.
3. If you want the inner loop to carry content, the residual should carry detail only. One lever from our lessons is to cut other paths (the series cut raised the worth of the remaining path 12 times in one test; lessons 2.1). I would not copy that into an H-Net before measuring, since the byte decoder needs the residual to predict inside a chunk.
4. A cleaner test is a three-arm run: residual on, residual scaled to a small constant, residual off. Judge by bits per byte and by zero-ablation of the main network output (the probe from lessons 3.3b).
5. Mind the norm balance. The H-Net note says deep main-network outputs dominate residual signals without a closing RMSNorm. Our carrier-constant finding (lessons 3.1) is a related fact: after one pass, 99.6 to 99.8 percent of a slot carrier's energy was one shared vector, which cancels in a stream mean. A probe that reads only a mean would miss it.

## Sequence length at the byte stage

Bytes make the outer sequence about 4 to 5 times longer than the BPE sequence (4.6 for GPT-2 in the paper's Table 1; StarCoder2 untested).

- Helps. The outer stages are the only places with this length, and they should run Mamba-2 or GLA, which are linear in length. The fused GLA path and the chunked form already scale this way. The fused CE kernel and the 49k-row head no longer matter.
- Hurts. Quadratic attention at byte length. The sliding window of 256 bytes is short. The 4-stream carrier multiplies activation memory. Per-token Python diagnostics cost 71.8 ms per step through about 900 GPU-to-CPU syncs (lessons section 5), and a byte run has 4.6 times more rows to diagnose.
- Neutral. MORPH's attention at chunk level runs at lengths it already knows.

## Fixed shapes versus dynamic chunk counts

| MORPH assumption | what breaks | simplest fix | status |
|---|---|---|---|
| `torch.compile` shape stability (`compile_core_dynamic`) | Router changes L_inner | Pad to a cap and mask | untested |
| CUDA graphs (`graph_step`, `graph_safe`) | Graphs need static shapes | Cap by buckets, one graph each | untested |
| TUL packer (`max_slots`) | Slot budget is a hard cap | Reuse the cap idea for chunks, define overflow | pattern exists, untested |
| MORTAR 128 tiles | Tiles are on weights | No change, pad rows to a multiple if the kernel needs it | untested |
| HCA ratio 256, CSA top-k | Compressed block count can be zero | Guard like `core_hca_compress_ratio` | measured pitfall |
| Active-set shrink by Poisson depth | Already ragged by row | Keep | code fact |

## Collisions with the research paste

The paste is LLM output. I treat every number in it as unverified.

1. "RMSNorm at the end of each loop pass keeps 95 percent of Stiefel's gradient stability." No source in our tree. What we measured points elsewhere. Weight-spectrum caps and projections did not cure loop failures: the composed sigma_max rose from 54.6 to 800.1 while every per-linear value stayed near 1.5 (lessons 1.5). A uniform rescale cannot slow an alignment of amplifying directions. RMSNorm on a looped state rescales; it does not fix direction alignment. Also, the carrier constant (lessons 3.1) lives in the shared direction, and a norm keeps it. And a pad slot at h = 0 gives an RMSNorm Jacobian near 1/eps, sigma_max 1.5e6 (lessons section 4). Masks matter for any norm inside a ragged loop.
2. "Orthogonal inner loops" keep isometry. A theorem in our tree says an isometric or contractive loop cannot keep several hypotheses: two distinct fixed points force the pass to expand (`parcae_needs_expansion`, lessons 1.3), and a contraction collapses all particles to one point. A pure rotation has no attractor either. Measured: successful looped operators in the literature are non-normal with gain above 1 in some direction, and forcing gain above 1 killed training in one case (lessons 1.5, citing 2608.18222). Huginn contracts about 0.8 per iteration while HRM-Text does not contract at all (lessons 1.8). Contractivity is not a universal target.
3. "Cayley over 4 residual streams." Already present: `cayley_orthogonal` in `hyper_connections.py`, closed form for n = 4. It acts per sublayer, not on the loop state.
4. "Halt when 1 - cos(z^r, z^{r-1}) < eps." Web-text loops in our tree finish their change by pass 3 (lessons 1.7), so this rule would stop near pass 3. Measured: K1-K6 follows the training depth draw (lessons 2.3), and a model trained at a fixed depth is out of distribution at others (lessons section 4). Halting needs a trained depth law, not an eps.
5. "Looping the inner stage is cheap because L_2 is short." True in FLOPs. In wall clock our loops were launch-bound (lessons section 5), and a shorter sequence makes kernels smaller, so the saving is smaller than FLOP counts suggest.
6. Lorentz router, centroid dechunker, "D_0 halves at equal routing fidelity", "under 0.2 percent FLOP overhead". No code, no measurement. Our Lorentz channel is an embedding table only (see that section). The cosine router uses direction; the paste says norm carries signal. Both are untested.
7. Flow matching, JEPA and neuralese on silent chunks. Several measured results bear on this. A flow thinker over span latents put about 1 percent of its loss on the past (lessons 3.6). A predicted continuous concept decoded worse than no concept (CE 14.91 against 14.63). A sampled next-span code was 0.27 nats worse than no code. Writes and reads fail separately: a perfect additive write moved CE by -0.0000 (lessons 3.4). Under teacher forcing, committed draws carry no target information. None of this says an H-Net cannot do latent thinking. It says a free continuous "thought chunk" needs a measured reader and a trained writer, and that selective dechunking is a new reader.
8. "H-Nets match BPE on compute-to-loss." The H-Net note says byte H-Nets start worse than the token Transformer and cross over only after 30B bytes (2-stage), 100B (1-stage) or 200B (space heuristic). The paste's "Cartesia Tokens to Bytes II" source is not in our tree. Do not rely on it.

## Open questions

Only an experiment can answer these.

1. Does a looped main network earn anything under the decoder residual? Run the three-arm residual test and judge by bits per byte at matched wall clock.
2. Does GLA work as the outer layer, against Mamba-2, on bytes? Same layer count, same width, bits per byte.
3. Does the learned router beat the TUL punctuation rule inside our code? The paper says yes for its setup. We have no byte run.
4. How does the boundary rate behave with our flat 1e-4 learning rate recipe and the 1000-step ramp? The paper scales learning rate per stage and trains with WSD; our recipe has neither.
5. Does a pointer head help a byte decoder? Measured only on slot models.
6. Does ternary hold on an H-Net main network, and does the router survive ternary or bf16 only?
7. Does ReMoE routing on the main network fight the ratio loss?
8. What is the byte count of our StarCoder2 rows, and what bytes per chunk does the router reach on our data?
9. Do padded chunk caps waste enough FLOPs to justify varlen kernels?
10. Does selective dechunking (silent chunks) give a reader, using the adapt-the-reader rule from lessons section 4 before any grade?

## Parts I skipped

- Olympiad/SFT data (`sft.py`, `sft_data.py`): task data, not architecture.
- JAX/Flax mirror (`morph/jax/`): lags PyTorch and has the old residual.
- Deploy packing (`deploy_quant.py`) and KV quantization (`kv_quant.py`): post-training steps with no H-Net-specific design question yet.
- TUL code, VQ, fan, spandec, FM planner and related files: span-latent experiments built on the slot loop, covered only through the slot section.
- AdEMAMix and the optimizer stack: not tied to tokenization. H-Net used AdamW with WSD (our note).
- Embedding int6 QAT: applies to a byte table of 256 rows only trivially.
