# Planned: the strict partner at `prefix_k` 8 with no quantizer (the width ruler the vq panel lacked)

Status: planned

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
