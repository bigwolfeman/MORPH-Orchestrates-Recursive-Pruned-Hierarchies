# Planned: a wider slot write on the ruler, under the shipped ternary rule

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`). The factor is the WRITE WIDTH: how many
coda positions the one looped slot state is projected into. Prior sweep of the same knob:
arc E18 (`2026-09-08-arc-e18-slot-width-sweep.md`, arms `e18-mask-k2/k4/k8`), which ran on
the MASKED mean-12 arm and under the OLD ternary rule. Wolfe's 2026-09-10 reading: the
slot's target and the slot's input/state capacity are the levers left; this arm is the
state-capacity half.

## Question

`tul.prefix_k` has been 2 on every arm of this lineage. The value came from Block
Transformer App. F.2 / Fig 3f, which picks a prefix length of 2 over 1, and nobody on this
tree has re-asked it since. `W_prefix` is `[prefix_k, d, d]` at identity init, so at step 0
the k cells all carry the same looped state `h_i` and the extra cells cost nothing; training
is free to make them differ.

E18 swept the width once, at 5,000 steps, on the masked mean-12 M-next arm under `absmean`.
The token K-curve rose monotonically with the width: k2 +0.0243 [+0.0233, +0.0253], k4
+0.0314 [+0.0300, +0.0328], k8 +0.0486 [+0.0470, +0.0502]. The CE did not follow (4.3395 /
4.3416 / 4.3351 at depth 6; final val_loss 4.4104 / 4.4762 / 4.4345), and the trained
`val/slot_eff_rank` rose with it (5.78 / 6.38 / 7.29).

That reading is confounded twice over for the question this arc is now asking. It ran under
the ternary rule the tree no longer ships, and it ran with the MASK on, which is the one
geometry where the slot is the only cross-span route and where every arm reads a token
K-curve whether or not the width moves. The ruler is unmasked and runs `norm_match`. Does
the width do anything when the loop is not the only road?

## Hypothesis

H-width-1. The coda reads the slot through `prefix_k` positions and that is the entire
channel between the loop and the tokens. A rank-limited state written through two cells is
the bottleneck the geometry audit keeps pointing at: the slot states sit at effective rank
1.7 to 4.8 in 1024 dimensions and the ruler reads `val/slot_eff_rank` 6.3389 with pairwise
cosine 0.7388. Doubling the cells doubles the read-side capacity at a cost of 2.10 M
parameters and gives training somewhere to put a second, different projection of the same
state. E18's monotone token curve is the direct evidence, and E18's rank rising with the
width says the extra cells were used.

H-width-0, and the favoured one. E18's monotone curve is an artefact of the MASK. Under
`tg_restrict` every path from an earlier span to a later token goes through the slot cells,
so the number of cells is the width of the only road, and widening the only road changes the
token loss whatever the loop does. Remove the mask and the prelude and the coda already
compose across spans on their own, which is exactly what the toy study's permissive-geometry
cell found (one core pass reached the task's answer and the loop's contribution fell from
+0.514 to +0.106 with the attachment held fixed). On this reading the ruler's token curve
stays at +0.0001, the forecast curve barely moves, and the arm buys 2.10 M parameters, 11 %
more positions per row and nothing else.

H-width-2. The E18 CE readings say the width is not free even where it helps: k4 was the
WORST of the three on final val_loss (4.4762 against k2's 4.4104) while having the middle
K-curve. If that reproduces, the width trades absolute loss for depth dependence, which is
the same trade E13 and E14 measured for the depth draw and the gain target, and it is not a
trade this arc is looking for.

## Method

`tul_slot_mux_prefix4_norm_match` = `tul_slot_mux_norm_match` (the ruler) plus ONE change,
`tul.prefix_k: 4`. Everything else is the ruler: M-next MUX at beta 1 through the tied head,
exit-only attachment, prelude entry, slot loop (`tokens_through_core: false`), `slot_seed`
boundary, `max_slots` 64, hinge lambda 100 at target 0.9, `core_fixed_point_lambda` 1.0,
`slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, fused kernels, seq 1024, batch 6,
seed 1, 5,000 steps, ramp 1,000, checkpoints at 2,500 and 5,000.

`max_slots` deliberately stays 64. Raising `prefix_k` alone holds the SLOT budget fixed and
widens each slot's channel; raising `max_slots` too would change how the row is segmented and
make this a two-factor arm.

**Legality and the token count, measured rather than argued.** `prefix_k` is legal at any
value at or above 1 (`TulLayoutSpec.__post_init__`, `TULConfig.__post_init__`), and E18 has
already run 4 and 8 on this rule and this tokenizer. `L_total = seq_len + prefix_k *
max_slots` = 1024 + 4*64 = **1280**, against the ruler's 1152. `pack_tul_row` fills a row to
`L_total` and places a boundary only when its `prefix_k` slot cells fit too, so tokens per row
move with both numbers, and the row's last at most `prefix_k` positions become tail pads.
Measured `val/span_tokens` at the same rule, seq and batch: the ruler holds **1014.0** tokens
and 54.8 spans in 1,152 positions at `pad_frac` 0.0247 (E18's k2 arm reads the identical
1014.0 / 54.8 / 0.0247), and E18's k4 arm holds **1034.5** tokens and 56.3 spans in 1,280
positions at `pad_frac` 0.0158. So this arm sees **2.0 % more tokens per row** and pays
**11.1 % more positions per row**: about 9 % more compute per token, before attention's
quadratic term. Parameters: `W_prefix` goes from [2, 1024, 1024] to [4, 1024, 1024], **+2.10
M**, identity init, so it draws no random numbers and every other weight is the ruler's at
seed 1.

**Rate and memory, extrapolated and labelled as such.** E18 measured 9,354 tok/s at step 200
for k2 and 9,098 for k4, a factor 0.973, with peak 14.10 GB against 17.03 GB. Scaling the
ruler's 12,429 tok/s by 0.973 gives about 12,100, well above the 8,086 rate floor; adding
E18's +2.93 GB to the ruler's 10.1 GB smoke peak gives about 13 GB on a 31.4 GB card. Both
numbers come from a DIFFERENT arm (masked, mean-12, eager with scoped kernels) and are
predictions, not measurements of this one.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16, 480 rows, giving
the token K-curve and the forecast column `mux_local`; `worth_profile.py` and
`slot_state_probe.py` at 5,000 (the runner's `slot` kind does both); `slot_gradient_probe.py`
and `slot_z_optimize.py` at 5,000 by hand. `val/slot_eff_rank`, `val/slot_pairwise_cos` and
`val/span_tokens` at 5,000 come from the run log. No new wandb keys.

The numbers this arm is read against, cited once. RULER `slot-mux-norm-match` at 5,000:
480-row CE at depth 6 **4.3290**; token K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000;
`mux_local` K1-K6 +0.0067 [+0.0053, +0.0081], K3-K6 +0.0005; forecast at depth 1 / 6 =
6.7791 / 6.7724; worth zero at offset 0 +0.094; z-optimisation `ce_loop` 4.1173 with the loop
ENTRY worth +0.0015; combined cancellation ratio 0.520; per-pass cotangent share 0.168 /
0.168 / 0.166 / 0.160 / 0.154 / 0.183 and per-pass cosine to the total 0.33 / 0.27 / 0.37 /
0.63 / 0.75 / 0.67; `val/slot_eff_rank` 6.3389, `val/slot_pairwise_cos` 0.7388,
`val/span_tokens` 1014.0, `val/span_pad_frac` 0.0247; 12,429 tok/s at step 200; wall clock
52 min 38 s; tripwire max 78.7; runner final val_loss 4.3775; smoke peak 10.1 GB.
E18 at 5,000 steps, ALL under `absmean`, ALL masked, mean-12 depth draw, so a LINEAGE
reference and not a paired control: token K1-K6 +0.0243 / +0.0314 / +0.0486 for k2 / k4 / k8;
K3-K6 +0.0034 / +0.0020 / +0.0029; `mux_local` K1-K6 +5.2621 / +0.3585 / +7.2894; CE at depth
6 4.3395 / 4.3416 / 4.3351; final val_loss 4.4104 / 4.4762 / 4.4345; tok/s 9,354 / 9,098 /
8,263; peak 14.10 / 17.03 / 22.77 GB; `val/slot_eff_rank` 5.78 / 6.38 / 7.29;
`val/span_tokens` 1014.0 / 1034.5 / 1083.8.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 5,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **82 %**. E18's k4 arm ran HEALTHY with tripwire max 329 at step 220
  and cleared the floor at 9,098 on a slower kernel path; this arm is the fused ruler with the
  same width. The 18 % is the detonation base rate plus the one thing E18 cannot speak to:
  a 2.10 M identity-initialised parameter block on a map held at gain 0.89 by a hinge, under
  a ternary rule that makes every pass stronger.
- **P-b (the arm's whole point: tokens).** Token K1-K6 at 5,000 above 0.010: **25 %**. Above
  0.020, where the masked k4 arm sat: **12 %**. Inside the ruler's CI, at or below 0.0005:
  **50 %**. These are low because the arm is built to test H-width-0, and H-width-0 is the
  one the toy study's permissive-geometry cell and this campaign's six flat unmasked arms both
  point at. A token curve appearing here without the mask would be the first time the reader
  moved on an unmasked arm at all.
- **P-c (forecast).** `mux_local` K1-K6 above 0.02 at 5,000 (the ruler +0.0067): **35 %**.
  Above the ruler's 0.0067 at all: **60 %**. E18's k4 read +0.3585 where its k2 read +5.2621,
  which is a reminder that the forecast column under the mask is dominated by depth 1 being
  catastrophic and is not comparable across arms without its end point. Scored with the end
  point.
- **P-d (the reader, the quantity the width is supposed to move).** The coda's
  exit-minus-entry CE from `slot_z_optimize.py` above 0.02 nats (the ruler +0.0015; the
  largest unmasked reading is the staged arm's +0.0073; the masked staged arm read +0.0235):
  **30 %**. This is the clause that tests the mechanism directly: if a wider channel makes the
  coda able to use the exit state, it shows here first. It is under 50 % because the channel
  was never measured as saturated; the audit's finding was that the loop's update survives the
  readout at 0.139 of its norm, which is a reduction problem, not a width problem.
- **P-e (the state, the instrument for "the extra cells were used").** `val/slot_eff_rank` at
  step 5,000 ABOVE the ruler's 6.3389: **65 %**. E18 read the rank rising monotonically with
  the width (5.78 / 6.38 / 7.29) and that measurement is about the states, which the mask does
  not directly constrain. A FALSE here says the extra cells are duplicates that training never
  differentiated, which would make P-b's and P-d's outcomes uninformative about capacity.
- **P-f (worth).** `worth_profile.py` zero at offset 0 above 0.15 nats (the ruler +0.094):
  **30 %**. Twice the cells is twice the surface the coda can lean on, whether or not the loop
  earns; this is the same clause and the same reasoning as the seed arm's, on the other side
  of the slot.
- **P-g (CE, the price).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **45 %**.
  E18's k4 was 0.002 from its k2 on the sweep and 0.066 WORSE on the trainer's val loss, so
  the two instruments disagreed on the same pair. Recorded as the price, not as a verdict
  (Wolfe 2026-09-09).
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **70 %**. The arithmetic
  says 11 % more positions per row for 2 % more tokens, and E18 measured 0.973x tok/s; but
  tok/s is per TOKEN and the wall clock follows positions, so the honest expectation is about
  1.09x to 1.15x and the 30 % is the attention term at S 1280 against 1152.
- **P-i (memory).** Smoke peak under 15 GB: **75 %**. The ruler smoked at 10.1 GB and E18
  measured +2.93 GB for the same width change on a row of the same shape.

## Binding

- P-b TRUE at the 0.010 bar ⇒ the write width moves the reader WITHOUT the mask, which no
  lever on this arc has done. The next arm is then k8 on the ruler (E18's own top of the
  sweep, one factor, already configured in the tree as a masked arm) and the width becomes a
  design question rather than a control.
- P-b FALSE with P-e TRUE (the rank rose, the tokens did not) ⇒ the extra cells were used and
  the reader still did not care. That closes the read-side capacity lane the way the seed arm
  closes the write-side input lane, and together the two arms say the slot's channel is not
  the binding constraint at either end. The arc's remaining candidates are then the TARGET
  (what the slot is asked for) and the GEOMETRY (whether anything forces the tokens to use
  it), which are the staged family and the mask family.
- P-b FALSE with P-e FALSE ⇒ training never differentiated the extra cells, the arm did not
  test capacity, and the honest filing says so: a wider `W_prefix` at identity init may need
  an init that breaks the symmetry before the width means anything. That is a code question
  and is NOT queued by this file.
- P-d TRUE with P-b FALSE ⇒ the coda can use a wider exit and the token loss still does not
  reward it. Read that against `slot-mux-mask-norm-match`, which is the same reader question
  in the geometry where the tokens have no alternative.
- P-g FALSE in the bad direction with P-b FALSE ⇒ the width is a pure cost on the ruler and no
  further width arm runs in this arc.
- A rate stop ⇒ the arm is skipped and the queue continues.

## Not verified before launch

The arm has not run on a GPU. What was run: a Hydra compose check printing `tul.prefix_k` 4
against the ruler's 2 with every other key in the panel list identical (`slot_seed` boundary,
`max_slots` 64, `tokens_through_core` False, `mux_beta` 1.0 / `mux_target` next,
`core_fixed_point_lambda` 1.0, `slot_gain_lambda` 100 at 0.9, `slot_cot_clip` 4.0,
`ternary_scale_mode` norm_match, `ademamix_t_beta3` 3500, `ademamix_t_alpha` 1600, 5,000 steps
at batch 6 and seq 1024) and the derived `L_total` moving 1152 to 1280; and a tiny CPU build
on the `tests/test_tul_forward.py` fixtures that ran a training forward and a backward at each
width, loss 9.673785 (k2) against 9.830132 (k4), `W_prefix` shape (2, 64, 64) against
(4, 64, 64), fixture `L_total` 42 against 52, and the backward reaching 59 core tensors with
summed absolute gradient 288.3 against 265.6.

Not verified: the rate, the wall clock and the memory at the real shape are all scaled from
E18's masked arms and none of them is a measurement of this arm. `prefix_k` 4 has never run
UNMASKED, never under `norm_match`, never on the mean-6 depth draw, and never with fused
kernels; E18's three arms are the only GPU evidence and all three differ from this arm in the
ternary rule, the mask, the depth draw and the kernel path at once. The claim that the extra
cells can differentiate during training rests on `W_prefix` being a free parameter, not on any
measurement that they did differentiate on E18; `val/slot_eff_rank` measures the slot STATES,
not the per-cell projections, so P-e is indirect evidence for that claim. Tail padding is
0.0158 of positions by E18's measurement rather than the 0.0247 the ruler pays, which means
this arm's rows are slightly BETTER packed and slightly harder to compare row for row; the
token-paired sweep handles that and the raw per-row figures do not. No test in this change
exercises the width: `tests/test_tul_layout.py` owns the packer's `prefix_k` contract and this
arm adds no code.
