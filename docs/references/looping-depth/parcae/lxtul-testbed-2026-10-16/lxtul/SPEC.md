# LXTUL training forward and losses: a port spec

Source of truth: the MORPH repo at master `3f9aae2`, config `morph/configs/lxtul.yaml`.
That file composes byte-identically to
`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm_10k` (only `wandb.name`
differs). I composed it with Hydra and traced only the branches this config takes. Every
claim below has a `file:line` citation. Paths are relative to the repo root.
`T.py` = `morph/model/transformer.py`, `tul.py` = `morph/model/tul.py`,
`lay.py` = `morph/model/tul_layout.py`, `fr.py` = `morph/model/tul_fan_route.py`,
`fan.py` = `morph/model/tul_fan.py`, `sd.py` = `morph/model/tul_spandec.py`.

Status of this document: I read the code. I did NOT run anything (no tests, no GPU), so
every statement is "the code says", not "measured".

---------------------------------------------------------------------------------------

## 0. What must port and what is backbone

### 0.1 The LXTUL mechanism (port all of this)

1. One packed row: token positions plus `prefix_k = 4` slot-cell positions after every
   span (section 1).
2. Strict geometry masks for the prelude, the loop and the coda (section 2).
3. The slot seed: `E_slot + W_sent * embed(t_last)` at every cell position (section 1.4).
4. The Thought Register term: 4 learned queries pool the span's prelude token states, one
   vector per cell (section 3.3).
5. The slot loop over the compact cell axis only (`S*M = 64*4 = 256` cells), per-slot
   Poisson depth, masked freeze of finished slots, full BPTT (section 3).
6. The diagonal injection of the per-cell seed `e_i` at every pass, ctx channels only
   (section 3.2). On Parcae this is the existing `h = A h + B e` injection; read 3.2 for the
   channel restriction.
7. The per-pass cell RMSNorm with one shared learned gain (section 3.4).
8. The latent-selected loop: a router picks one of the 4 cells after every pass, and the
   cells of every continuing slot are reset to the winner (section 3.5).
9. The EMA-prelude latent target, the rank-only latent head `g`, the teacher pick, the
   router CE (section 3.6).
10. epivol diversity on passes 1-2 (section 3.7).
11. Winner-only write into the coda, losers exactly zero (section 4).
12. Losses: weighted CE with slot-id masking, token-state dropout, span decoder, fixed-point
    term, slot-gain hinge, cotangent clip, lsel loss, epivol (section 5).

### 0.2 MORPH backbone specifics (do not port; listed only)

- Hyper-Connection residual, `n = 4` streams, Cayley mixing (`hc_streams 4`). The carrier is
  `[B, L, 4, C]`. Every "stream mean" below is a no-op on a single-stream Parcae carrier.
- Ternary STE shadow weights with `norm_match` scales (`training.ternary: true`).
- CCA + CSA/HCA attention. Under `tg_restrict` every layer has two branches: a window branch
  (window 256, XSA excludes the self token) and a "compressed" branch that becomes a dense
  softmax over the allowed columns plus a learned per-head sink logit
  (`morph/model/attention.py:1131-1162`, `_tg_slot_attention` at `attention.py:580`,
  `_window_fallback` at `attention.py:697`). Plus a CCA causal conv and a `W_v_prev` value
  shift that reset at segment boundaries (`tg_seg`). CoPE clipped RoPE.
- Hybrid euclidean + Lorentz embedding, hash-bigram embedding (`bigram_hash_vocab 49152`)
  with a per-layer lambda, per-layer `x0` re-injection into the ctx channel slice
  (`T.py:3836-3867`). `n_ve 0`: no value embeddings.
- `lm_mixer` before `final_norm` in `_readout` (`T.py:4805-4822`).
- AdEMAMix beta1=0 optimizer, fused fp32 path, int6 embedding QAT, 8-bit Adam states.
- MTP heads: `mtp_heads: 1` builds NO extra head (`T.py:2241-2243`). Nothing to port.
- Retention (GLA) is OFF (`retention: false`).
- Dropout 0.1 (`training.dropout`) on embeddings and MLPs.

Backbone numbers that set shapes: `d_model 1024`, `n_heads 8`, `n_kv_heads 4`,
`d_ff 2816`, `n_prelude 4`, `n_core 6`, `n_coda 4`, vocab 49169, `channel_dims [512, 320, 192]`
(the "ctx" slice is channels 512..831, `T.py:1997-2008`).

---------------------------------------------------------------------------------------

## 1. Sequence layout

### 1.1 One packed sequence, fixed length

ONE sequence per row holds tokens AND slot cells (`lay.py:340-531`, `pack_tul_row`).

- `L_total = seq_len + prefix_k * max_slots = 1024 + 4*64 = 1280` (`lay.py:335-337`;
  `max_slots 64`, `prefix_k 4`). Batch 6.
- The packer consumes tokens until `tokens + 4 * slots == 1280`, so the token count per row
  varies (about 1040 at about 60 slots). If a row has more than 64 boundaries it ends at the
  65th boundary; leftover positions become TAIL PADS (`lay.py:415-437`).
- After every span boundary token `t_last` of span `s`, 4 slot positions follow. Token `i`
  lands at row position `i + 4 * (#boundaries before i)` (`lay.py:441-444`).
- Arrays per row (`lay.py:446-475`):
  - `input_ids [L]`: tokens, and `slot_id` (the `<fim_pad>` id) at every slot and pad
    position.
  - `labels [L]`: at a token position, the NEXT TOKEN in the token stream (slots skipped).
    So `t_last`'s label is the first token of span `s+1`. At the LAST prefix cell of each
    slot (the "emit" position) the label is also the first token of span `s+1`. Every other
    prefix cell and every pad has `-100`.
  - `slot_mask [L]`: True at every slot cell and pad.
  - `bag_id [L]`: for a token, the index of the span it belongs to; for a slot cell, its own
    slot index; `max_slots` (=64, the "dump bin") for tail pads AND for the open tail tokens
    after the last boundary.
  - `slot_index [S]`: row position of the slot's FIRST cell. Cell `i` of slot `s` is at
    `slot_index[s] + i`. `slot_valid [S]`: real slots (pads last).
- Boundary rule (`lay.py:187-240`, LUT at `lay.py:75-101`): token `i` closes a span iff it is
  EOS, or (its decoded string, right-stripped, ends in one of `.;!?`, or contains `\n`, `—`,
  `–`, `--`) and the span length is at least `min_span 4`, or the span length reached
  `span_cap 32`. No comma. Causal: one state machine for loader and generator.

### 1.2 Separate tensors

The prelude and the coda run on the full `[B, 1280]` row. The loop runs on a SEPARATE
compact tensor: the cells gathered by `slot_index + i`, `[B, S*M = 256, (4 streams,) C]`,
slot-major (`index = s*4 + i`) (`T.py:5926-5955`). The loop output is scattered back into the
row for the coda (`T.py:14012, 14043`).

### 1.3 What the prelude sees

The prelude (4 blocks) runs over all 1280 positions under the strict prelude mask (2.1).
Input = `embed_drop(slot_input(embed(input_ids)))` (`T.py:5613-5614`), then expanded to the 4
HC streams, plus a per-layer injection (x0 in the ctx slice + lambda * bigram) at EVERY
position including cells (`T.py:4386-4393`). Backbone detail: the bigram signal at a slot
position is the bag-mean of the span's bigram embeddings (`T.py:5615-5617`,
`tul.py:6754-6773` with `add_e_slot=False`).

### 1.4 The slot seed (`slot_seed: boundary`)

At every slot position (all 4 cells of a slot get the SAME input embedding)
(`TULSlots.slot_input`, `tul.py:6688`; boundary branch `tul.py:6722-6733`):

    seed = E_slot + W_sent @ embed(t_last)

- `t_last` = the span's LAST token (`boundary_token_index`).
- `W_sent`: `[d, d]` bias-free, init N(0, 0.02) from a private generator.
- `E_slot [d]`: set to the MEAN of the (tied) embedding table at step 0
  (`morph/training/train.py:3219`, `TULSlots.init_at_activation`), trained afterwards.
- Tail pads get `E_slot` alone.
- `x0` (the per-layer re-injection signal) is a clone of this embedding (`T.py:4374`), so x0
  is also identical across the 4 cells of a slot.

The 4 cells of one slot differ in the prelude ONLY by (a) their row position (RoPE/CoPE) and
(b) causal order: cell `i` may read cells `0..i-1` of its own slot (2.1).

### 1.5 The loop entry state per cell

    xn = input_norm(prelude_out)                  # RMSNorm(d), per stream   (T.py:5941)
    e[s, i] = xn[slot_index[s] + i] + reg[s, i]   # reg = register term       (T.py:5955, 5978-5980)
    h_0[s, i] = e[s, i]                           # core_state_init prelude   (T.py:6107; _CloneInit T.py:808-819)

Pads are zeroed by `gather_valid` (`tul.py:6414`). `e` is ALSO the per-pass injection
source (3.2).

---------------------------------------------------------------------------------------

## 2. Attention masks (FlexAttention `mask_mod` form)

Notation per row `b`: `bag[q]`, `is_slot[q]` = `slot_mask`, positions in row order.
MORPH builds these as bool `[B, 1, L, L]` "allow" tensors (`lay.py:903-1032`) built ONCE
per forward in `_tul_tg_kwargs` (`T.py:12747-12845`; strict branch `T.py:12771-12817`).

### 2.1 Prelude (`tg_strict_allow(stage="prelude")`, `lay.py:1007-1016, 1031`)

    def prelude_mask(b, h, q, kv):
        return (kv <= q) & (bag[b, q] == bag[b, kv])

- A token reads only its own span's earlier tokens (span-local, causal).
- A slot cell reads its own span's tokens and its own slot's earlier cells (its own cells
  share its `bag_id`; the span's tokens come before them).
- Nothing crosses a span. So the prelude output at a cell is a pure function of span `s`.
- MORPH detail (backbone): the window branch also excludes `kv == q` (XSA) and the
  compressed branch reads slot columns only, so a TOKEN query's attention never reads
  itself. On a plain SDPA backbone include `kv == q` (the standard causal self-attention).
- The conv / value-shift reset ids are `seg = 2*bag + is_slot` (`lay.py:858-871`): a span's
  tokens, its cells and the next span's tokens are three segments. Not needed on Parcae (no
  conv).
- `tg_strict_prelude: span` (default) — the causal-history variant is OFF.

### 2.2 The loop core (`slot_cell_relation`, `T.py:1608-1721`; delivered at `T.py:6510-6540`)

Compact axis, `M = 4`, `q, kv in [0, 256)`:

    def core_mask(b, h, q, kv):
        return (kv // M) <= (q // M)

- A cell reads EVERY cell of its OWN slot (earlier AND later siblings) and every cell of
  every EARLIER slot. No token position is in the loop at all.
- Same mask for all 6 core layers and all passes (`loop_reach 0`, so `_kw0` everywhere,
  `T.py:6539-6540`). It REPLACES the causal term (`tg_relation`), it is not ANDed into it.
- No validity term: pad slots sit last, so a real cell never reads a pad (pads have the
  highest slot index).
- Frozen (finished) slots stay in the sequence and keep serving K/V from their frozen state
  (`T.py:5888-5894`).
- Positions in the core = compact index `s*4 + i` (positions come from the tensor's own
  sequence axis).
- MORPH detail: the window branch also excludes self (`attention.py:741`), the compressed
  branch includes self plus a sink logit. The CCA conv in the core is NOT segmented (no
  `tg_seg` in `_core_akw`), so it runs causally over the flattened cell axis. Port: one SDPA
  with `core_mask`, self included.

### 2.3 Coda (`tg_strict_allow(stage="coda", coda_prefix_reach="all")`, `lay.py:1017-1031`)

    def coda_mask(b, h, q, kv):
        causal = kv <= q
        if is_slot[b, q]:                       # a prefix cell query
            return kv == q                      # itself only
        tok = (bag[b, q] == bag[b, kv]) | (is_slot[b, kv] & (bag[b, kv] < bag[b, q]))
        return causal & tok

- A token reads its own span's earlier tokens plus EVERY prefix cell of every EARLIER slot
  (`bag_j < bag_i`). It does NOT read its own span's slot (those cells come after it), and it
  reads no earlier span's TOKENS (`tg_coda_token_reach 0`).
- Open-tail tokens (bag = 64) read all real cells. Tail pads (bag 64, slot) are never read
  by a token (`64 < 64` is false).
- A prefix cell reads only itself, so it carries only what the loop wrote into it.
- All 4 cells of each earlier slot are keys, including the 3 loser cells, whose input is
  exactly zero (section 4). They are still keys: after the first coda layer they carry a
  function of a zero input plus their position.
- Cut together with the mask (this is part of strict geometry, `T.py:14051-14064`): the
  coda's per-layer x0 / bigram injections are multiplied by 0 at every slot cell
  (`slot_cell_inject_keep`, `lay.py:842-855`), so a cell carries the loop state and nothing
  else.
- MORPH detail: the window branch limits a token to keys within 256 positions; the
  compressed branch reads ALL allowed slot columns with no window. So in effect a token
  reads every earlier cell. The CCA conv/value shift run inside each segment
  (`2*bag + is_slot`), so within one slot's 4 cells the conv mixes cell `k` with cells
  `k-1..k-3`. This is a backbone side channel; a plain SDPA port has no such mixing.

### 2.4 The register pooling (section 3.3)

Cell `(s, i)` pools over the TOKEN positions with `bag == s` (all of span `s`, no causal
order needed because the whole span precedes the cell) (`tul.py` register forward, file line
6202 onward: `own = (bag_id == s) & ~slot_mask`).

### 2.5 The EMA target prelude

Same as 2.1 (it runs the twin prelude with the same strict prelude kwargs, `T.py:9617-9618`).

### 2.6 Span decoder (`sd.py:290-348`)

Its own tensor `[B*S, J=32, C]`, plain causal self-attention per (row, slot), no cross-slot
or cross-row reads: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
(`sd.py:338`).

---------------------------------------------------------------------------------------

## 3. The loop

### 3.1 Depth (`T.py:5631-5659`, `6327-6359`)

- TRAIN: `d ~ Poisson(mean_depth = 6)`, clamped to `[1, max_depth = 8]`, ONE draw PER SLOT
  (the draw is made on the `[B, S*M]` cell axis and cell 0's value is used for all 4 cells of
  the slot, `T.py:6337-6352`). `torch.poisson` on the device (global RNG). Pad slots get 1.
- EVAL: `d = 6` for every valid slot (`int(slot_mean_depth or mean_depth)`).
- `total_iters = max(d)` over the batch (one host sync, `T.py:6357-6359`). Each pass runs the
  WHOLE compact sequence; `active = depth > t` (`T.py:6800`) and finished slots are frozen by
  `h = where(active, h_new, h)` (`T.py:7338`).
- `slot_depth_fixed 0`, `slot_mean_depth 0`, `slot_max_depth 0` (defaults, `tul.py:824-826`).

### 3.2 One pass: the injection and the core (`T.py:4013-4152`, `_apply_core_step`)

    h_inj = DiagonalInjection(h_in, e)             # once per pass  (T.py:4100)
    for l in 0..5:                                  # 6 shared core blocks
        h_inj = h_inj + inj_term[l]                 # x0/bigram at the cell positions (backbone)
        h_inj = core_block[l](h_inj, mask=core_mask)
    h_new = h_inj

`DiagonalInjection` (`T.py:720-773`, built at `T.py:2070-2073` with `injection_channels ctx`):

    A  = exp(log_A).clamp(max=0.9999)    # [320], init 0.447
    dt = exp(log_dt)                     # [320], init 1.0 (log_dt = 0)
    h[..., 512:832] = A * h[..., 512:832] + dt * e[..., 512:832]
    # the other 704 channels pass through unchanged (Jacobian = identity there)

- `injection_B: false`: no `B` matrix.
- So the injection only refills the 320 ctx channels with the cell's own seed `e_i`. The
  other 704 channels are carried, not re-injected. At init the map whose blocks do nothing
  has typical gain `sqrt((704 + 320 * 0.447^2) / 1024) = 0.865` (`T.py:726-736`).
- Port note for Parcae: Parcae's injection is normally full width. To be faithful, apply it
  to a fixed 320/1024 channel slice and leave the rest as identity. If you choose full width,
  say so; it is the documented floor the gain hinge (5.6) is relative to.
- `inj_term[l]` (`T.py:6298-6303`) = `lambda_l * bigram + ctx-slice proj_l(x0)`, gathered at
  the cell positions, identical for the 4 cells of a slot, added before EVERY core layer at
  EVERY pass. Backbone (Parcae has no per-layer re-injection; dropping it is a backbone
  choice, keeping it is optional).
- `slot_source_once false`: the source is injected at every pass.

### 3.3 The 4 cells and `slot_cell_init: distinct` (`TULSlotRegister`, `tul.py:6115-6248`)

`fan_k 4` aliases `slot_cells 4` (`morph/training/tul_setup.py:284-292`). Built at
`T.py:2738-2740`.

    k = W_k xn_tok            # xn single-stream (stream mean of input_norm(prelude)), [B, L, C]
    v = W_v xn_tok
    a[s, i, :] = softmax_j( <Q_i, k_j> / sqrt(d) )  over token positions j with bag==s
    pooled[s, i] = sum_j a[s,i,j] v_j
    reg[s, i] = W_o pooled[s, i] + P_cell[i]          # masked to 0 at invalid slots

- `Q [4, d]` (distinct: one query per cell; "same" would share one), `W_k`, `W_v` `[d, d]`
  bias-free, all init N(0, 0.02) from a private generator (seed 0x5E63). `W_o` ZERO-init,
  `P_cell [4, d]` zeros. So at step 0 `reg = 0` and the cells differ only through the
  prelude outputs at their 4 different positions.
- `slot_pool_heads 1`: single head, scale `d ** -0.5`.
- A slot with no token key (pads) opens the mask to all positions and is zeroed after, to
  avoid an all `-inf` softmax.
- `reg` is added to `e` BEFORE `core_init`, so it is part of `h_0` AND of the per-pass
  injection source (`T.py:5978-5980, 6306`).

HOW CELLS STAY DISTINCT AFTER A RESET (3.5). Nothing explicit is added. After a reset all 4
cells of a slot hold the winner's full carrier. They diverge again through:
1. the injection source: `dt * e_i[ctx]` differs per cell (`e_i` = prelude output at the
   cell's own position + `reg_i`);
2. the core position index `s*4 + i` (RoPE/CoPE);
3. MORPH-only: the unsegmented causal CCA conv over the flattened cell axis.
(`fr.py` header and CLAUDE note `morph/model/CLAUDE.md`, latent-selected loop row: "Cells
stay distinct after a reset through their own injection source e_i ... and core position;
no offset is added.")

### 3.4 The per-pass cell RMSNorm (`slot_cell_pass_norm: rms`)

- Module: `self.tul_cell_norm = RMSNorm(d)` (`T.py:2942`), `attention.RMSNorm`
  (`attention.py:139-147`): `y = x * rsqrt(mean_C(x^2) + 1e-6) * g`, computed in fp32, `g`
  `[1024]` init ones, ONE gain shared by every pass, every cell, every stream. No weight
  decay (name contains "norm"), never ternary.
- Applied per HC stream over C (`T.py:3975-3976`).
- WHERE: after the core step (which includes the injection) and before everything that
  reads the pass output (`T.py:7185-7187`). Order inside a pass:

      h_new = core_step(h_in)                      # T.py:6976-6996
      [gain hinge reads core_step at h_in]         # T.py:7056-7073 (raw map, no norm)
      h_new = RMSNorm(h_new) * g                   # T.py:7185-7187
      [cotangent clip hook on h_new]               # T.py:7246-7247
      [fixed-point term reads h_new vs h_in]       # T.py:7268-7292
      h = where(active, h_new, h)                  # T.py:7338
      traj.append(h)                               # T.py:7339-7340 (pre-reset candidates)
      h = lsel_reset(h)                            # T.py:7341-7349

- The loop ENTRY `h_0 = e` is NOT normed by this module (it is already `input_norm`ed).
- Every cell is normed, pads included.

### 3.5 Selection each pass: the latent-selected loop (`fan_loop_select: joint`)

State built once before pass 0 (`_lsel_begin`, `T.py:9752-9791`, called at `T.py:6365-6367`):

    ctx[s] = stream_mean( xn[slot_index[s]] ).detach()   # normed prelude at the slot's FIRST cell
    z[s], ok[s], zo[s] = EMA target (3.6)                 # None on a label-free forward
    teacher_drives = False                                # train_follow: router (and eval follow None)

After pass `t` (`_lsel_pass`, `T.py:9793-9841`):

    act = valid & (depth > t)                             # slots whose pass t ran
    c = stream_mean(h.view(B, S, 4, ...)).detach()        # [B, S, 4, C], post-norm cells
    scores = router(c, ctx)                               # [B, S, 4], grad to router only
    rpick = argmax(scores + bias)                         # bias buffer stays 0 here
    if z is not None:
        with no_grad: tpick = argmin_i mean_C (g(c_i) - z)^2
        rce_sum += sum_{act & ok} CE(scores, tpick);  rce_n += count
    follow = rpick                                        # router-followed at train AND eval
    final[act] = follow
    reset = valid & (depth > t + 1)                       # slots that run another pass
    h[s, :] = h[s, follow[s]]  for every reset slot       # reset_to_winner (fr.py:510-526)

- Router `FanRouter` (`fr.py:368-438`), rank `fan_lsel_router_rank 64`:
  `score_i = v^T ReLU(W_c LN(c_i) + W_x LN(ctx) + b_x)`; `LN` = LayerNorm without affine;
  `W_c [64, d]` no bias, `W_x [64, d]` with bias, `v [64] ~ randn / 8`; fp32, autocast off.
  Inputs DETACHED (cells and ctx). Its balance bias is a buffer that is never stepped for
  the lsel router (only `tul_fan_router` is stepped, `T.py:10071-10072`, and that one is not
  built), so the pick is plain `argmax(score)`.
- Reset is a gather on the cell axis: the gradient of all 4 copies flows back into the
  winner's path; a loser's pass-`t` output gets no gradient from later passes.
- A slot that finishes at pass `t` is NOT reset: it keeps its 4 final candidates. The coda
  reads its final winner; later passes of OTHER slots read all 4 frozen candidates as K/V.
- The trajectory keeps the PRE-reset candidates (epivol reads them, 3.7).
- `fan_select_eps 0.05` is INERT in this config: it is read only by the WTA picker in
  `_tul_fan_all`, which returns early at `fan_all_wta_lambda 0` (`T.py:8950-8952`). The
  "eps 0.05" that is live is `fan_lsel_eps` (the relaxed WTA in the exit loss, 3.6), default
  0.05 (`tul.py:661`).

### 3.6 The latent target, the rank-only head, the exit loss

TARGET (`_tul_fan_target`, `T.py:9588-9624`, called at `T.py:13087-13090` on every labelled
forward with `plan_mode normal`):

    with no_grad:
        xt = twin_front(input_ids, layout, strict prelude mask)   # FanTargetFront (fr.py:91-194)
        z[s] = LN( mean over scored TOKEN positions p with bag[p] == s+1 of stream_mean(xt[p]) )
    zo[s] = LN( same pooling of the LIVE prelude output x )         # WITH grad
    ok[s] = slot_valid[s] & (count of those positions > 0)

- "Scored token positions" = `labels >= 0` and not a slot (`span_ce_index`,
  `T.py:1287-1313`). Bin `s+1` = span `s+1`, the span the coda decodes after slot `s`.
  Pooling and LN: `pooled_span_states` (`fr.py:197-216`), LN without affine, fp32.
- `xt`, `x` are the prelude OUTPUT before `input_norm`.
- The twin: deep copies of the 4 prelude blocks and the prelude layers' x0 projections (and
  value-embed projections, none here). Lookup tables (token embedding, bigram, `W_sent`,
  `E_slot`) are read LIVE. Eval mode (no dropout), no embedding dropout, no RNG draw.
  Parameters frozen. Built after quantisation, before compile (`train.py:2462`).
- EMA after every optimizer step: `theta_twin <- 0.996 * theta_twin + 0.004 * theta_live`
  (`fan_target_ema 0.996`, `tul.py:620`; `fr.py:175-187`; called from `tul_fan_after_step`,
  `T.py:10053-10068`, at `train.py:3965`).
- Because the prelude is span-local, `z[s]` depends on span `s+1`'s tokens only.
- Edge case (read off the code, not measured): the open tail has `bag = 64`, so it is bin 64
  and is read by slot 63 only. The last REAL slot of a row with fewer than 64 slots reads an
  empty bin and gets `ok = False`.

HEAD `g` (`FanLatentHead`, `fr.py:461-485`): `LN -> Linear(d, d) -> GELU -> Linear(d, d)`
(`fan_lsel_hidden 0` means `d`), fp32, autocast off, ONE head shared by the 4 cells.

DISTANCE: `dist_i = mean_C (g(c_i) - z)^2` (`fr.py:488-492`).

EXIT LOSS (`_lsel_finish`, `T.py:9843-9909`, called at `T.py:7464-7467` on the final carrier):

    c = stream_mean(h_final.view(B, S, 4, ...))           # final candidates
    g_in = c.detach()                                     # fan_lsel_head_input: detached  (rank-only)
    dist = dist(g(g_in), z)                               # [B, S, 4]
    tex = argmin(dist.detach())
    L_lat = sum_{ok} [ (1 - eps) * dist[tex] + eps/(M-1) * sum_{i != tex} dist[i] ] / n_ok
            # eps = fan_lsel_eps = 0.05, M = 4              (fr.py:495-507)
    L_enc = mean_C relu( 0.1 - std_over_ok_slots(zo[:, C]) )  # fan_lsel_enc_gamma 0.1; batch_std eps 1e-6
    L_rce = rce_sum / max(rce_n, 1)                       # router CE over every (pass, slot) graded
    L_lsel = 1.0 * L_lat + 0.2 * L_enc + 1.0 * L_rce      # fan_lsel_lambda 1.0 (config),
                                                          # fan_lsel_enc_lambda 0.2, fan_lsel_router_lambda 1.0
    winner = final                                        # the router's pick at the slot's last pass

What each term trains (read off the detach points):
- `L_lat` trains `g` ONLY (cells detached). "Rank-only": no latent gradient reaches the loop.
- `L_enc` trains the LIVE prelude (and embeddings) through `zo`: a variance floor on the
  pooled next-span prelude states.
- `L_rce` trains the router only.
- The teacher (`tpick`) only labels the router. The loop FOLLOWS the router at train and at
  eval, so train and eval run the same selection rule.

### 3.7 epivol diversity (`fan_repel_mode epivol`, passes 1-2)

Computed in `_forward_tul` from the trajectory `traj` (`T.py:13570-13661`), training only
for the charged term (`T.py:13660-13661`). `traj[0] = h_0` (the seed), `traj[t]` = the
carrier after pass `t` (after the freeze, BEFORE the reset). `valid` = per-slot
`slot_valid`. Penalised passes `t = 1..fan_repel_passes = 2`, WITH grad; if the batch max
depth is 1 only `t = 1` exists.

EPI part (`fan.py:414-539`):

    with no_grad:
        seed = stream_mean(traj[0]) mean over the 4 cells, valid slots   -> [N, C]
        H = reservoir(seed)        # LN -> W1 (256 x C, N(0,1/C)) -> ELU -> W2 (64 x 256, N(0,1/256)), frozen, seed 0
        a = ridge_map(H, lam=3.0)  # standardise columns (center, /std, /sqrt(64)), double;
                                   # QR of [H; sqrt(lam) I], a = R^-1 Q_top^T   -> [64, N]
    for t in 1..2:
        zc = stream_mean(traj[t]) valid slots                             -> [N, 4, C]
        dev = (zc - mean_i zc) / mean_i ||zc_i||                          # scale-free, live
        epi_t = mean_i [ 0.5 * log2 det(I_64 + 30 * W_i W_i^T) / 64 ],  W_i = a @ (dev_i - mean_N dev_i)
    E = - mean_t epi_t

VOL part (`fan.py:559-614`):

    for t in 1..2:
        G = dev dev^T per slot  [N, 4, 4]  (fp32, autocast off)
        vol_t = mean_slots [ 0.5 * log2 det(I_4 + 30 * G) / (4 - 1) ]
    V = - mean_t vol_t

`L_epivol = E + V`; it enters the loss as `fan_repel_lambda * L_epivol = 0.1 * L_epivol`
(`T.py:14638-14650`). Its gradient reaches the cells' pass-1 and pass-2 states (including
losers' candidates), hence the loop, the register and the prelude.

### 3.8 What is detached, BPTT and checkpointing

- `bptt_depth 8 >= max_depth 8`, so `n_nograd = 0`: FULL BPTT through every pass
  (`T.py:6374`).
- `ckpt_grad_iters 4`: passes `t = 0..3` run under `torch.utils.checkpoint(use_reentrant=False)`,
  passes `4..7` eager (`T.py:6415-6416, 6825, 6981-6985`).
- Detached: router inputs (cells, ctx); teacher distance (no_grad); `g`'s exit input; the EMA
  target `z`; the reservoir seed and ridge readout; the gain hinge operating point, `e` and
  `inj` (5.6); the tied embedding table in the span decoder (both sides).
- NOT detached (`joint` mode): the loop output into the coda write and into the span decoder.

### 3.9 Loop pseudocode (training, this config only)

    # x: prelude output [B, L, (n,) C]; layout: per-slot view; cells: slot_index[s] + i
    xn = input_norm(x)
    e  = gather(xn, cell_positions) * valid + register(stream_mean(xn), layout)   # [B, S*4, (n,) C]
    h  = e                                                                         # h_0
    inj = [x0_bigram_term(l, gather(x0, cells), gather(bigram, cells)) for l in core]
    depth = per_slot(clamp(poisson(6), 1, 8)); depth[pad] = 1                      # train
    T = max(depth)
    t_gain = uniform_int(0, T)                     # RNG put back afterwards
    ls = lsel_begin(xn, layout, depth, z, ok, zo)
    traj = [h]; fp_terms = []
    for t in range(T):
        active = depth_cells > t
        step = checkpoint(core_step) if t < 4 else core_step
        h_new = step(h, e, inj)                    # DiagonalInjection(h, e) then 6 blocks
        if t == t_gain:
            gain_pen = slot_gain_penalty(core_step, h.detach(), e.detach(), inj.detach(), t,
                                         mask=active & valid)
        h_new = RMSNorm(h_new) * g_norm            # shared gain, per stream
        h_new.register_hook(cot_clip(t))           # 5.7
        fin = active & valid & (depth_cells == t + 1)
        fp_terms += [ ||h_new - h||^2 / (||h_new||^2 + 1e-6)  for cells in fin ]   # 5.5
        h = where(active, h_new, h)
        traj.append(h)
        h = lsel_pass(ls, h, t)                    # router pick, router CE, reset continuing slots
    h.register_hook(cot_ref)                        # exit cotangent reference
    lsel_out = lsel_finish(ls, h)                  # exit latent loss, final winner
    L_fp = mean(cat(fp_terms))
    return xn, h, depth, traj, gain_pen

---------------------------------------------------------------------------------------

## 4. The write to the coda

- `W_prefix`: `[prefix_k=4, d, d]`, init IDENTITY for every copy (`prefix_per_cell 1`)
  (`TULSlots.__init__`, `tul.py:6582-6602`).
  Applied as a right multiply per stream: `value = cell @ W_prefix[i]`
  (`TULSlots.prefix_project`, `tul.py:6775`, the `cells is not None` branch).
- Winner-only write (`_fan_route_cells`, `T.py:9729-9748`, used at `T.py:13968-13971`):

      routed[s, i] = cells[s, i] * onehot(winner[s])[i]      # losers EXACTLY 0, scale None

- Cell `i` of slot `s` (routed) goes to row position `slot_index[s] + i` through
  `W_prefix[i]`. So the WINNER keeps its own index: if cell 2 won, position
  `slot_index[s] + 2` holds `cell_2 @ W_prefix[2]` and positions `+0, +1, +3` hold exactly
  zero. `E_pass` is not built (`prefix_source: exit`), so there is no per-cell bias
  (`tul.py:6853-6856`).
- `x_coda = scatter(xn, prefix_positions, values)`: token positions keep `xn` (normed prelude
  output); valid slot cells are overwritten; pad slots go to a dump row (`T.py:14012-14043`,
  `scatter_positions` `tul.py:6427`, `prefix_positions` `tul.py:6860`).
- Positions / RoPE ids of the prefix cells in the coda are their ROW positions
  (`slot_index[s] + i`, in `[0, 1280)`).
- The coda's loser cells receive exactly zero gradient through this write; a loser's final
  state still gets gradient from the span decoder (it reads the mean of the 4 raw cells,
  5.3), from epivol if its pass is 1 or 2, and as K/V for other slots' later passes.
- `pseudo_k 0`: the snap carrier that would fill loser positions is OFF.

---------------------------------------------------------------------------------------

## 5. Losses

### 5.1 Coda and the main CE

    x_coda, keep = token_state_dropout(x_coda)      # train only (tul.py:6898, T.py:14049)
    keep = keep * (~slot_mask)                       # strict: zero cell injections (T.py:14063-14064)
    xh = coda(x_coda, x0, bigram, inject_keep=keep, mask=coda_mask)   # 4 blocks + _readout
    L_ce = sum_i w_i * CE_i / sum_i w_i  over positions with label != -100

- Token-state dropout (`token_state_dropout 0.15`, `tul.py:6898` onward): each TOKEN position
  (never a slot) is replaced, with p = 0.15, by `E_mask` (`[d]`, init 0, learned) in every
  stream, AND its coda per-layer x0/bigram injection is zeroed (`keep = 0`,
  `T.py:4521-4522`). Without the second part the token would leak back through x0.
- Weights (`_tul_half_weights`, `T.py:10245-10272`): `w = 1` everywhere; `w = plast_weight
  = 1.0` at `t_last` (the token just before each slot); `w = emit_weight = 0.0` at the LAST
  prefix cell of each slot (the emit position). Other prefix cells and pads have label
  `-100`. Net effect: plain next-token CE over every token position; the emit label carries
  no loss.
- Slot-id masking: the `slot_id` logit is `-inf` in the CE (`mask_token_id=slot_id`,
  `T.py:12479-12482`), and in generation logits (`T.py:14889-14890`).
- Head: weight-tied to the input embedding (`embed.lm_weight()`), fused chunked CE
  (`T.py:12426-12518`).

### 5.2 Which positions carry labels (summary)

All token positions (each predicts the next token of the token stream, crossing span
boundaries at `t_last`). Slot cells: only the emit cell has a label and its weight is 0.

### 5.3 Span decoder (`spandec true`, `_tul_spandec_loss` `T.py:11112-11238`, `SpanDecoder`
`sd.py:358-508`)

- Input state: `z[s] = _readout(h_slots)[s]` where `h_slots` is the MEAN of the 4 FINAL raw
  cells (NOT the routed winner) (`_tul_fan_all` returns the mean, `T.py:8944, 8950-8952`;
  `TULFanMix` mode "all" is the uniform mean, `fan.py:290-301`; read at `T.py:13825`).
  `_readout` = stream mean -> `lm_mixer` -> `final_norm` (shared with the main head path).
- Target: span `s+1`'s INPUT token ids, first `J = 32` tokens (`spandec_max_tokens 0` ->
  `bound_span_cap 32`), valid only if slot `s` AND slot `s+1` exist (span complete)
  (`sd.py:93-171`, `horizon_span_slots` at `sd.py:186` with horizon 1, offset 1). So the
  last real slot of a row (open tail) has no span-decoder target.
- Decoder (`sd.py:445-508`): `x_0 = z_in(z)`, `x_j = tok_in(E[t_{j-1}]) for j >= 1` (invalid
  tokens zeroed), `+ pos[j]` (zero-init learned table), 2 blocks of pre-norm causal MHA (8
  heads) + SwiGLU (`d_ff 2816`), `out_norm` (RMSNorm). Plain `nn.Linear`s, init N(0, 0.02)
  from private generators, never ternary.
- Loss: `L_sd = mean over valid (s, j) of -log softmax(state_{s,j} @ E^T)[t_j]`, slot id
  masked. `E` is the tied table DETACHED on both the input side (always) and the output
  side (`mux_detach_head true`, `T.py:11176`). Weight `spandec_weight 1.0`.
- Gradient: decoder params, `z_in`, `tok_in`, `pos`, `lm_mixer`, `final_norm`, and through
  `z` the loop (all 4 final cells), register, prelude. Not the embedding table.
- Runs at train and eval (it is subtracted from the reported val loss in `train.py`).

### 5.4 MUX, sigreg, MTP

`mux_beta 0.0`: no MUX term (`T.py:13700-13701`). `sigreg_lambda 0.0`: none. `mtp_heads 1`:
none. `mux_detach_head true` only sets the span decoder's head detach.

### 5.5 Fixed-point term (`core_fixed_point_lambda 0.1`, `T.py:7268-7292, 7469-7478`)

Training only, at every pass inside the BPTT window (all passes here):

    fin = active & valid & (depth == t + 1)          # cells that finish at pass t
    r_c = ||u_c - h_c||_F^2 / (||u_c||_F^2 + 1e-6)   # norms over (streams, C), fp32
    L_fp = mean over all finishing CELLS in the batch of r_c     (4 cells per slot)

- `u` = the pass output AFTER the cell norm; `h` = the state ENTERING the pass. For a slot
  with depth `T >= 2`, `h` is the post-reset winner copy from pass `T-2`, so for the 3
  non-winning copies the term measures their distance from that shared start. For depth 1,
  `h` is the un-normed entry `e`.
- Weighted `0.1 * L_fp`, added via `_apply_core_aux` (`T.py:4631-4654`, at `T.py:14868-14872`).

### 5.6 Slot gain hinge (`slot_gain_lambda 100`, `target 0.9`, `eps 0.02`, `T.py:7516-7650`)

Training only, ONE pass per step: `t_gain ~ Uniform{0..n_grad_iters-1}` drawn with the
global CPU RNG and the RNG state put back (`T.py:6408-6413`). At that pass:

    m  = active & valid                                    # per cell
    hp = h_in.detach(); e, inj detached
    v  = randn_like(hp) * m
    d  = v * (0.02 * ||hp_cell|| / ||v_cell||)             # per cell, norms over (streams, C)
    RNG restored before each call (same dropout masks); both calls under checkpoint
    f0 = core_step(hp);  f1 = core_step(hp + d)            # RAW map: injection + 6 blocks, NO cell norm
    gain_b = ||(f1 - f0) * m||_F / (||d||_F + 1e-6)         # per batch ROW b, over all its cells
    L_gain = 100 * mean_b relu(gain_b - 0.9)^2

- The gain includes the DiagonalInjection floor (0.865 at init).
- Gradient reaches the core blocks and the injection's `log_A`/`log_dt` only (sources are
  detached).
- Under the cell norm the hinge bounds the raw map `f`, not the carried map `RMSNorm(f)`
  (`T.py:3913-3930`).
- Tail and floor terms are off (`slot_gain_tail_lambda 0`, `slot_gain_floor_lambda 0`).

### 5.7 Cotangent clip (`slot_cot_clip 4.0`): a gradient hook, not a loss (`T.py:5699-5739`,
registered at `T.py:7246-7247, 7379-7383`)

- On the loop's final carrier `h`: a hook records `ref_b = ||grad_h[b]||_F` per batch row.
  It does not change the gradient.
- On every pass's output `h_new` (post-norm), a hook rescales the incoming gradient per row:

      scale_b = min(1, 4.0 * ref_b / (||g_b||_F + 1e-12));   g <- g * scale_b

- If the exit got no gradient, nothing is clipped. Only when `h_new.requires_grad`.

### 5.8 lsel loss: section 3.6. epivol: section 3.7.

### 5.9 Total loss (training)

    L = L_ce
      + 1.0 * L_sd                                   # span decoder
      + 100 * mean_b relu(gain_b - 0.9)^2            # slot gain hinge (one sampled pass)
      + 0.1 * L_fp                                   # fixed point, finishing cells
      + 0.1 * ( -mean_{t=1,2} epi_t - mean_{t=1,2} vol_t )      # epivol
      + 1.0 * L_lat + 0.2 * L_enc + 1.0 * L_rce      # lsel (L_lat trains g only)

Fold order in code: CE (`T.py:14234`), gain (`T.py:14345-14346`), spandec
(`T.py:14364-14377`), lsel (`T.py:14612-14619`), epivol (`T.py:14638-14650`), fixed point
(`T.py:14868-14872`). `train.py` backprops `out["loss"]` (all of the above); it subtracts
the `*_weighted` terms only for the REPORTED train/val loss (`train.py:153-238`). Grad clip
1.0 on the global norm (`train.py:3944`), then the optimizer step, then the EMA twin update
(`train.py:3965`).

Inert in this config (set, but no effect): `fan_select_eps 0.05`, `fan_all_wta_lambda 0`
(no WTA coda passes), `tg_restrict_scope all` (strict takes the other branch first,
`T.py:12771`), `mux_rho/mux_tau/mux_target` (mux off), `spandec_max_tokens 0` (means 32).

---------------------------------------------------------------------------------------

## 6. Eval-time behaviour

- `model.eval()`: depth = 6 for every valid slot (3.1). No dropout, no token-state dropout.
  No fixed-point term, no gain hinge, no epivol charge (instruments still computed).
- Selector at eval: the SAME router argmax as training (train follows the router too). The
  teacher is computed on labelled val forwards for readings only; `lsel_follow="teacher"`
  is an eval-only instrument that makes the loop follow the teacher (`T.py:12932-12938`,
  `T.py:9784-9786`).
- The coda reads the final winner alone, exactly as at train.
- K sweeps (K1..K6): `lab/divergence/core_depth_sweep.py` sets
  `model.cfg.tul.slot_mean_depth = d` between evals (lines 14-16, 300-307), so every valid
  slot runs exactly `d` passes. Rows are packed once and reused, so CE(d) is paired.
  "K1-K6" = token CE at d = 1 minus token CE at d = 6, with a paired bootstrap CI over rows.
  A `slot_depths [B, S]` table can also force per-slot depths (eval only, range [1, 8],
  `T.py:5661-5697`).
- `eval_ablations: true` makes `train.py` run extra eval forwards (plan ablations); they are
  instruments, not part of the model.

---------------------------------------------------------------------------------------

## 7. Other things on in this config that change the forward

- `tg_restrict: true`: builds the restricted attention variant in EVERY layer (prelude, core,
  coda): no pooled compression, no top-k; the "compressed" branch becomes a dense softmax over
  allowed columns plus a sink (`attention.py:1131-1162`). Backbone-level, but it is why every
  branch honours the masks of section 2.
- `tg_geometry: strict` + `tg_coda_prefix_reach: all`: the masks of 2.1 and 2.3, the segment
  resets, and the zeroed coda injections at slot cells (`T.py:14051-14064`).
- `slot_seed: boundary`: section 1.4.
- `core_hca_compress_ratio 16` (from `tul_short.yaml`): moot under `tg_restrict` (no pooling).
- `model.use_kernels false`, `tg_scoped_kernels true`: speed only.
- `ternary_scale_mode: norm_match`, `ternary_scope: backbone`: backbone quantisation. By the
  selection rule (`morph/model/ternary_qat.py:649-691, 797-806`) every `nn.Linear` outside
  an attention module and WITHOUT `_ternary_exclude` is ternarised. So, read off the code:
  `W_sent` and the register's `W_k`, `W_v`, `W_o` ARE ternary STE weights in MORPH;
  `W_prefix`, `E_slot`, `E_mask`, `Q`, `P_cell`, the cell-norm gain and the injection's
  `log_A`/`log_dt` are raw Parameters and are NOT; the router, `g`, the span decoder and the
  reservoir carry `_ternary_exclude` (`fr.py:405-407, 477-479`; `sd.py:315`) and are NOT.
  For a non-ternary Parcae port this is backbone; ignore it.
- `mux_beta 0`: the MUX is replaced by the span decoder.
- Not on: `pseudo_k`, `loop_attn_center`, `loop_attn_hc` (cayley), `fan_trigger_every_pass`,
  `fan_seed_noise`, `fan_lineage`, `fan_history_streams`, `loop_reach`, `loop_carry`,
  `code_enum_k` (1), `gram`, `spandec_parallel`, `nextlat`, `recon`, `bcast`, `cond_layers`,
  `center_exit`, `vq_codes`, `retention`, `scse`.

---------------------------------------------------------------------------------------

## 8. Shapes at the panel config (B = 6)

| tensor | shape |
| --- | --- |
| packed row | `[6, 1280]` |
| prelude / coda carrier | `[6, 1280, 4, 1024]` (MORPH HC); `[6, 1280, 1024]` on Parcae |
| compact loop carrier | `[6, 256, 4, 1024]` (64 slots x 4 cells) |
| register term | `[6, 256, 1024]` |
| `z`, `zo`, `ctx` | `[6, 64, 1024]` |
| router scores, dist | `[6, 64, 4]` |
| span decoder input | `[6*64, 32, 1024]` |
| `W_prefix` | `[4, 1024, 1024]` |

## 9. Known unverified edges

- I traced code only; no run, no test.
- The ternary status in section 7 comes from the selection rule, not from a dump of the
  parametrised modules of a built model.
- The exact CoPE / window-256 behaviour inside the coda for a token far from an earlier
  cell: the compressed branch has no window, so I conclude every earlier cell is readable,
  but I did not read the compressed-branch code path for coda queries line by line beyond
  `_tg_slot_attention`.
- The open-tail latent-target bin mapping (section 3.6, edge case) is read off
  `span_ce_index` and `pooled_span_states`; I did not confirm it on a real batch.
