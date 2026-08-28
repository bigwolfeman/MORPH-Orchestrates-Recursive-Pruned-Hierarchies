# TG worklist — opened 2026-08-28

Supersedes the TG section of [OVERNIGHT-WORKLIST.md](OVERNIGHT-WORKLIST.md).
Panel result and the paper analysis that motivates all of this:
[../experiments/failures/2026-08-27-tg-restriction.md](../experiments/failures/2026-08-27-tg-restriction.md).

## Where we stand (measured, 2026-08-27/28)

- TG restriction WORKS on the plan: plan worth 0.033–0.155 ce_main vs control band
  0.0124–0.0164. Takeover eliminated in TG2 (0/2, end core share 0.0020/0.0035).
- It COSTS 0.17–0.42 nats ce_main at matched step 3000. TG's own whole prize is
  ~0.03 nats and DECLINES with model size in their sweep. So the deficit is the
  thing to attack, and a longer single-pass run is not the way.
- The restriction's compute win is REAL and entirely UNREALIZED: 10.96× fewer
  attention pairs on real data, but implemented as a dense mask, so TG arms run
  1.73 sps vs the control's 1.82 (a 5% LOSS).

## Padding audit (2026-08-28) — answers "are we fighting padding for gradient?"

Measured on a real OWT validation batch, `tul_tg2`, b=6:

    valid spans/row     : 55 43 42 42 44 40   (max_slots=64 -> 69% of capacity used)
    slot POSITIONS/row  : 110 86 84 84 88 80
    dump-bin positions  : 1.36% of the sequence
    token positions/row : 1042-1072 of 1152
    row0 span lens      : n=55 mean=18.73 median=18 min=1 max=32
                          at_cap(32)=24%   <=16 tokens=47%

**Sequence padding is 1.36% and most of it is REAL tokens past the last slot, not
pad. We are NOT fighting sequence padding for gradient.**

**But the instinct was right in a different place.** `_tul_core` gathers
`slot_index [B, max_slots]` and loops a FIXED 64-slot compact sequence while only
~44 are valid, so ~31% of core-loop compute runs on zeroed pad slots.
`gather_valid` zeroes them, so there is no content pollution — real slots attend
zero-vectors, which dilutes the softmax denominator slightly and buys nothing.
The core is ~20-24% of layer passes, so this is ~6-7% of total compute, free to
recover.

## Arms, in the order I would run them

### A1. TG4a — delete the bag-mean  [1 line, no new params, 35 min/seed]
`slot_input = E_slot` alone; drop `+ mean_j embed(t_j)`.
WHY: `pooling_probe` on tg2-s1@3500 confirms the plain-mean law at slope -0.470,
r2 0.922. Slot-seed signal falls 0.516 (spans 4-5) -> 0.210 (spans 24-32) against
a constant of 0.238. Under the restriction the slot already attends its whole span
through 4 prelude blocks, so the mean is redundant AND diluting.
WATCH: all slots then start identical. Run `slot_rows_probe` / `pooling_probe` on
the first checkpoint — MORPH's takeover was a slot-state rank collapse.

### A2. TG3 — soft restriction  [ALREADY BUILT, config only]
Tokens see own span + previous span + slots (`tul.tg_soft_prev_span: true`).
WHY: the cheapest possible read on "is the restriction simply too tight". The
deficit's size (0.17-0.42) makes this worth running now, not only on TG1 failure.

### A3. Blind-decoder PROBE on FROZEN z  [no training, minutes, zero risk]
Wolfe's idea, reframed from a loss into a measurement. Freeze `tg2-s1/step_3500.pt`,
extract `h_slots` (z), train a SMALL decoder on z alone and ask how many nats of
**span i+1** it recovers vs a unigram baseline.
WHY THIS SHAPE: it answers the question the campaign has circled for weeks — is the
plan EMPTY, or FULL BUT UNREAD? — and it does so without adding an objective to a
training run. It also side-steps the standing ban: `CLAUDE.md` forbids decoding a
span from one vector with no token path (Huginn/MegaByte/Bowman/Hourglass), and that
ban is about the SHIPPED path, not about a frozen-weights probe.
DECIDES: high recovery -> the plan is full and the coda is not reading it; the fix is
the reader. Low recovery -> the plan is empty; the fix is provenance/objective.

### A4. max_slots trim  [config, recovers ~6-7% compute]
64 -> 56, after checking the truncation rate on a real-data sweep (row 0 hit 55).
NOTE: this becomes BINDING, not slack, if span_cap is ever lowered — see below.

### A5. TG6 — restriction + small cross-boundary window  [small mask change]
Allow each token a ~32-token window ACROSS span boundaries, on top of span+slots.
WHY: TG itself names copying as its weak spot ("GPT-2 has direct access to the exact
token sequence, while TG must infer the sequence from a compressed sentence state").
Induction/copying is the most likely single share of our deficit, and OWT is far more
copy-heavy than TG's WikiText-103. Cost: +37K pairs on 121K, still ~8.6× sparse.

### A6. TG5 — asymmetric restriction  [one-line mask change]
Slots attend ALL prior tokens; tokens stay restricted to own span + slots.
WHY: the shortcut stays closed (no token -> past-token path) while the channel gets
strictly richer. Slots are only ~89 positions, so compute barely moves.
CAVEAT: deviates from TG, which computes its sentence vector from its own sentence.

### A7. FlexAttention rewrite  [~half a day, torch 2.11 confirmed available]
Replace the dense-mask implementation with a `mask_mod` closing over
`bag_id`/`slot_mask` + `create_block_mask`.
PAYOFF: 10.96× fewer pairs measured. Attention is 18% of a block at seq 1152 (so
~16% saved) but 44% at seq 4096 (so ~40% saved, ~1.7×). Inference KV cache
compresses by `mean_span / prefix_k` ≈ 9.4× today.
SPIKE FIRST: confirm XSA self-exclusion + CoPE + per-head sink logits survive one
`score_mod`/`mask_mod` pair. Unverified.

### A8. T=1 loop-off under restriction  [config only]
WHY: the core loop is ~24% of layer passes and buys at most 0.036 nats. This answers
P2 from the other direction — if CE is unchanged at T=1, the loop is dead weight and
the TUL thesis needs to be restated.

## Gated / rejected for now

### A9. span_cap sweep  [config only, 2 arms]
Measured 2026-08-28 by re-packing one real OWT stream at each cap (ABSOLUTE means are
inflated vs the true stream — the reconstruction concatenates across removed slot ids —
but the TREND across caps is valid, all caps share one buffer):

    cap    mean  spans/row  slotpos   tokpos  at_cap%   KVcomp
    16    14.04       64.0    242.3    909.7    72.7%    7.02x  << max_slots SATURATED
    24    19.26       53.7    107.3   1044.7    61.5%    9.63x
    32    23.63       44.3     88.7   1063.3    49.2%   11.82x   (current)
    48    29.87       35.3     70.7   1081.3    34.0%   14.93x
    64    33.23       32.0     64.7   1087.3    24.0%   16.61x

**The sequence is FULL: tokens and slots compete for the same 1152 positions.** More
spans means more slot positions means FEWER token positions of LM signal per step.

So span_cap is an allocation knob with a real trade in BOTH directions, and the two
hypotheses are symmetric and untested here:
- **DOWN (Wolfe's 24):** +21% slots = +21% plan channel capacity. Worth testing IF the
  0.17-0.42 nat deficit is the channel being capacity-limited. Costs 18.6 token
  positions/row and drops KV compression 11.8x -> 9.6x.
- **UP (48):** +18 token positions/row, KV compression 11.8x -> 14.9x, and TG's own
  ablation runs this way (sentence length 64 beats 32 by 0.6 PPL).
RUN BOTH, one seed each, AFTER A1 — the pooling law was the only argument that ever
favoured short spans, and A1 deletes it.

- **span_cap 32 -> 16: NOT recommended. MEASURED SATURATION.** cap=16 drives spans/row to
  exactly max_slots=64, so boundaries get DROPPED, and token positions fall 1063 -> 910
  (a 14% loss of LM signal per step). KV compression falls to 7.0x.
  The stated motivation (less padding) does not hold — padding is 1.36%. Measured
  mean span is 18.73 (median 18), not 12. Lowering the cap to 16 truncates ~53% of
  spans, pushes spans/row from ~44 to ~85 which EXCEEDS max_slots=64, and cuts KV
  compression from ~9.4× to ~6×. TG's own ablation runs the other way (sentence
  length 64 beats 32 by 0.6 PPL). Revisit only AFTER A1, since the pooling law is
  the only argument that ever favoured short spans and A1 deletes it.
- **Flow-matching / diffusion span head as a TRAINING loss**: gated on A3. Our
  objective-split O5 found aux objectives orthogonal, not conflicting; MUX failed;
  TG2 (removing objectives) is our best arm; TG uses NO auxiliary loss. Adding one
  needs the probe to say the plan is FULL first, and needs to target span **i+1**
  (prediction), not span **i** (autoencoding — the DB `sigma*` trap, where MORPH's
  SliceScaler put 77% of training into autoencoding).
- **Longer single-pass run**: contraindicated. TG's fitted exponents match GPT-2's
  (α 0.152 vs 0.149); the gain is an intercept shift and does not compound. If the
  undertrained hypothesis is ever tested, use TG's protocol — a small fixed subset,
  multi-epoch, early stopping — not more single-pass steps.
- **`bptt_depth` / loop depth**: under Wolfe's standing veto. FLAGGING ONLY, with new
  evidence: TG's single biggest ablation by 10× is detaching sentence vectors at
  memory write (29.8 -> 35.0 PPL; nothing else in their table moves >0.7). Our core
  runs mean depth 6 with `bptt_depth=4`, so 2 of 6 iterations execute under
  `no_grad`. Wolfe's call, not mine.
