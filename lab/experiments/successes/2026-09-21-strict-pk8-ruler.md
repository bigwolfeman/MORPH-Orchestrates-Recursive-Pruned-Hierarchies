# Planned: the strict partner at `prefix_k` 8 with no quantizer (the width ruler the vq panel lacked)

Status: success

Date: 2026-09-21 (frozen before any GPU step). Arc: the discrete-thought panel
([`mixed/2026-09-13-arc-discrete-thought-vq.md`](../mixed/2026-09-13-arc-discrete-thought-vq.md)),
whose Binding says: "If P-7 fails — vq8 clearly beats vq4 — the write's capacity is a live
lever and the sweep continues upward, with the `prefix_k` confound resolved first." P-7
failed (vq8 − vq4 = −0.0328 [−0.0358, −0.0300]). This arm is the confound's resolution.

## Question

vq8 differs from the strict partner (`prefix_k` 2) by the quantizer AND the prefix width
(`L_total` 1536 against 1152). How much of vq8's +0.025 paired deficit against the partner,
and of its 0.033 win over vq4, is the width? A strict ruler at `prefix_k` 8 with ONE
continuous stream projected into 8 cells and no quantizer differs from vq8 by the
quantizer alone.

## Method

`morph/configs/tul_slot_spandec_strict_pk8.yaml`: `tul_slot_spandec_strict` with
`tul.prefix_k: 8` and nothing else (composes: `L_total` 1536, `slot_cells` 1, strict, span
decoder on). Same seed, packer, rows, 5,000 steps, batch 6, the runner into
`/home/wolfe/morph-scratch/arc/results/2026-09-13-register/`. Readouts: SWEEP@2500/5000,
WORTH, STATEPROBE; paired depth-6 reads (`paired_vs_ruler.py`, 1024-token blocks) against the
strict partner's, vq8's and vq4's `tokens.npz`.

## Predictions (frozen)

References: strict partner d6 sweep CE (2026-09-12) as the ruler; vq8 − strict +0.0250
[+0.0217, +0.0283]; vq4 − strict +0.0577; vq8 − vq4 −0.0328; the register's measured width
gain for a 1→4 prefix 0.022; vq8 RATE 9,450 tok/s at `L_total` 1536, peak 24.83 GB.
Probabilities are the builder's.

- **P-1 (healthy).** 5,000 steps, tripwire HEALTHY, no OOM at batch 6. **90 %.**
- **P-2 (width is worth something).** pk8 − strict at depth 6, paired, inside
  **[−0.050, −0.005]** (pk8 BETTER). **60 %.** Residual: 25 % inside ±0.005, 15 % worse.
- **P-3 (THE ARM'S REASON: the quantizer costs beyond the width).** vq8 − pk8 at depth 6,
  paired, at or above **+0.020**. **65 %.** If it holds, the discrete write's deficit is the
  quantizer's own, and vq8's win over vq4 reads as code capacity rather than prefix width.
- **P-4 (the loop stays flat).** Tokens K1−K6 below **+0.010**. **85 %.**
- **P-5 (cost).** RATE at step 200 at or above **9,000** tok/s. **60 %.**

## Binding

If **P-3 holds**: the vq panel's Binding stands with the confound resolved: the write's
capacity is a live lever in CODE COUNT, and the residual-code form (LCM Quant-LCM-c) is the
next discrete arm. If **P-3 fails** (vq8 within 0.02 of pk8): the discrete write costs
nothing beyond its width, and vq8's win over vq4 was the prefix; the discrete lane closes on
capacity and the LCM reconstruction-ceiling caveat is the remaining open item. If **P-2
lands in its 15 % tail** (pk8 worse than strict): width is not free at 8 cells on one stream,
and the fan's `prefix_k 4` remains the reference width.

## Not verified before launch

- No GPU step; the runner's smoke is the first. The compose (Hydra + `build_tul_runtime`,
  CPU) printed `prefix_k=8 L_total=1536 slot_cells 1`.
- Whether `prefix_project` from one stream into 8 cells has ever run past a smoke: `pk4` ran
  (the fan's width partner); 8 has not.

## Results

Run: 5,000 steps at `6d71e6c` on the 5090 through `run_recon.sh`, START 07:47 local
2026-09-21, DONE 08:55, tripwire HEALTHY (`preclip/total` max 38.6 at step 214), final
val_loss 4.4118, RATE 9,426 tok/s at step 200 (9,347 at 4800), peak 24.83 GB (vq8 24.83).
Artifacts in `../results/2026-09-13-register/`:
`sweep_slot-spandec-strict-pk8_{2500,5000}.json`, `worth_..._5000.json`,
`slot_state_..._5000.json`, `run_slot-spandec-strict-pk8.txt`, `paired_pk8_vs_strict_5000.json`
(ruler = the strict partner's 5,000 sweep at depth 6), `paired_vq_vs_pk8_5000.json` (ruler =
pk8 at depth 6, arms vq8 and vq4).

**Depth sweep (480 rows).** 5,000: d1 4.3154, d2 4.3146, d3 4.3145, d6 4.3144, d9 4.3146,
d12 4.3148, d16 4.3151; tokens K1−K6 **+0.0010 [+0.0008, +0.0012]**, K3−K6 +0.0000; span
decoder K1−K6 +0.0043. 2,500: K1−K6 +0.0005. The slot-loop yardstick (+0.002) holds at
eight cells on one stream.

**Paired reads at depth 6 (1024-token blocks).** pk8 − strict **−0.0349 [−0.0378,
−0.0319]** (490 blocks; depth 1 −0.0340). vq8 − pk8 **+0.0592 [+0.0563, +0.0623]** (514
blocks); vq4 − pk8 +0.0923 [+0.0886, +0.0962] (500 blocks). By subtraction vq8 − vq4 is
−0.033, the panel's −0.0328.

**Worth profile at 5,000.** zero 0.1947, shuffle 0.1769, wrong_seed 0.0734 (vq8: 0.1397 /
0.1263 / 0.0362). Slot-state probe: the exit moves 0.215 of its norm from depth 1 to 6, cos
0.984.

**Scoring.**

- **P-1: HOLDS.** 5,000 steps, HEALTHY, no OOM at batch 6 (peak 24.83 GB).
- **P-2: HOLDS.** −0.0349, inside [−0.050, −0.005]. Eight cells on one stream are worth
  0.035 nats to the coda at 5k, more than the register's 0.022 for four.
- **P-3 (the arm's reason): HOLDS.** +0.0592 against +0.020, the interval clear by 0.036.
- **P-4: HOLDS.** K1−K6 +0.0010.
- **P-5: HOLDS.** 9,426 ≥ 9,000.

## Verdict

**Success** (5 of 5). The width confound in the vq panel is resolved: a continuous
one-stream write into eight prefix cells is 0.035 nats BETTER than the two-cell strict
partner, and the eight-code quantizer is 0.059 nats WORSE than that width ruler. The
discrete write's deficit is the quantizer's own, and vq8's 0.033 win over vq4 is code
capacity, not prefix width. Binding branch taken: "P-3 holds: the vq panel's Binding
stands with the confound resolved; the write's capacity is a live lever in CODE COUNT, and
the residual-code form (LCM Quant-LCM-c) is the next discrete arm." The loop is flat at
every width tried (2, 4, 8 cells; K1−K6 ≤ 0.005 on every one-stream arm): width buys the
READER nats and buys the loop nothing, the register's 2026-09-13 reading again.

## Updated hypothesis

Prefix width is a reader lever with a measured value (0.022 at 4 cells, 0.035 at 8) and
no depth value. A discrete write pays for its capacity in nats the reader loses; the next
discrete arm is the residual-code form, scored against pk8 as its width ruler. Nothing
here moves the LXTUL-P ladder, which asks a different question (a job per pass).
