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
**RUN 2026-08-28, seed 1 (seed 2 in flight).** ce_main@3000 4.8094, @3500 4.7146 —
statistically on top of tg1-s1 (4.8104 / 4.7154) and still 0.26–0.35 nats behind the
control band. Takeover HELD, end core share **0.0011**, the campaign's lowest.
**Do NOT quote its 0.0921 loop worth.** `loop_off` falls back to the slot's own INPUT,
which this arm deliberately stripped of span content, so that column measures "loop vs
a CONSTANT" while a bag_mean arm's measures "loop vs a span summary". The fix is the
`no-loop, bag-mean seed` condition added to `slot_path_worth.py` on 2026-08-28
(`seed_bagmean`); tg4a-s1's two checkpoints predate it and need a backfill re-run.
Pre-registration for the rest of round 2, and the two process failures behind it:
../experiments/planned/2026-08-28-tg-round2-seed-and-softness.md

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

### A4. max_slots — MEASURED 2026-08-28, RECOMMENDATION REVERSED. DO NOT TRIM.
Over 360 real rows (60 batches), span_cap=32:

    spans/row: mean 51.84  median 51  p90 64  p99 64  MAX 64
    39 of 360 rows (10.8%) sit at EXACTLY 64 -> max_slots is ALREADY BINDING
    max_slots=56 -> 20.8% of rows truncate;  52 -> 39.7%;  48 -> 67.2%

`pack_tul_row` handles saturation by ENDING THE ROW EARLY (`n_tok = min(n_tok, bpos[S])`,
tul_layout.py:412-414), NOT by dropping boundaries — the spec is explicit and the leftover
tokens return to the buffer for the next row. So this is a documented, bounded cost (a
saturated row carries 1024 tokens instead of up to ~1072), not a silent data defect. I
raised it as an alarm first; it is not one.

TWO CORRECTIONS TO MY OWN EARLIER NUMBERS IN THIS FILE:
- The padding audit's "~44 valid of 64" came from ONE batch. Over 360 rows the mean is
  **51.84**, so core-loop pad is **19%**, not 31% — about **4%** of total compute, not 6-7%.
- Trimming max_slots is therefore WRONG in both directions: it saves ~4% at best and
  truncates 21% of rows at 56. Recovering the 19% honestly needs a ragged/compact gather
  (which breaks the fixed shapes torch.compile wants), not a smaller cap. PARKED, not queued.

### A4c. RAISE max_slots — CENSORING CONFIRMED 2026-08-28, this is the real defect

The top-tail histogram over 360 real rows settles it. Counts fall smoothly, then pile up
at exactly the cap:

    span count : 56  57  58  59  60  61  62  63  64
    rows       : 16   7   6   6   8   2   5   2  39   <-- 10-20x the neighbouring bins

That is textbook right-censoring. **max_slots=64 binds on 10.8% of rows**, and
`pack_tul_row` responds by ENDING THE ROW EARLY, so those rows carry fewer tokens.

How far does the natural tail actually reach? Re-packing with the cap lifted:

    max_slots :  64    80    96   128   192
    observed MAX : 60    64    66    68    74

So the true demand tops out near **74**. `max_slots = 80` captures essentially all of it.

**CAREFUL — `L_total = seq_len + max_slots * prefix_k`.** Raising max_slots alone LENGTHENS
the sequence (64->1152, 80->1184, 96->1216) and hands the row more tokens, so a naive bump
changes three things at once and is not a clean arm.

CLEAN ARM: `max_slots: 80` WITH `data.seq_len: 992`, which holds `L_total = 992 + 160 = 1152`
exactly. Then the ONLY change is the slot budget. Compute impact is confined to the core
loop's compact sequence (64 -> 80 = +25% core = ~+5.5% total); attention and the coda are
unchanged because L_total is unchanged.
This also directly tests the CHANNEL CAPACITY hypothesis that A9-DOWN was reaching for,
without the row-shortening confound that made span_cap 24 unsafe.

### A4b. RAISE span_cap instead  [config only — this is the lever A4 thought it was]
Raising span_cap is strictly better than raising max_slots: it relieves the saturation,
adds token positions, improves KV compression, AND matches TG's ablation direction. It is
the same knob as A9-UP below.

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
- **DOWN (Wolfe's 24): now measured as UNSAFE.** The sweep's re-packed buffer understated
  span counts — real data gives mean 51.84 spans/row at cap=32, with 10.8% of rows ALREADY
  at the max_slots=64 ceiling. Lowering the cap can only raise the span count, so cap=24
  would saturate most rows and shorten them. The "+21% channel capacity" argument does not
  survive: the extra slots cannot be allocated, the ceiling is already binding. If the
  capacity hypothesis is worth testing, test it by RAISING max_slots, not by lowering the
  cap — that isolates capacity from row length.
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

## Incidental finding — validation is the TRAIN stream, offset (2026-08-28)

`create_dataloader` falls back to `split="train"` for OpenWebText because OWT has no
validation split (data.py:103-105). `train.py:1755` separates them with
`skip_samples=50_000` instead. So validation is the SAME stream, 50k documents ahead.

- **Safe at our current rung.** 3500 steps x b6 x 1153 tokens = 24.2M tokens, roughly 24k
  documents — about half way to the validation region. Every number in this campaign is
  clean.
- **HAZARD for the long runs discussed.** Past roughly **7,000 steps at batch 6** training
  walks into its own validation set. Any >=10k-step run MUST raise `skip_samples` first, or
  it will report a validation loss on text it has trained on.
- (My earlier train-vs-validation comparison returned byte-identical statistics. That was my
  test being degenerate — I passed the same `skip_samples` to both — not evidence of a leak.)
