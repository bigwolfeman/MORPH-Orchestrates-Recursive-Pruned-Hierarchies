# Planned: the ruler with its core's HCA compressed branch alive

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the ruler [`slot-mux-norm-match`](../failures/2026-09-09-arc-slot-loop-norm-match.md)
(`morph/configs/tul_slot_mux_norm_match.yaml`). Source of the mechanism: finding **F1** of
[`results/2026-09-10-slot-geometry-audit/README.md`](../results/2026-09-10-slot-geometry-audit/README.md).
Design note:
[`2026-09-10-slot-loop-readout-and-attention-defects.md`](../../../.agents/notes/proposed/architecture/2026-09-10-slot-loop-readout-and-attention-defects.md).
Prior run of the same knob on a different question:
[`2026-08-25-h24-hca-branch-arm-binary.md`](../failures/2026-08-25-h24-hca-branch-arm-binary.md).

## Question

`GatedPoolCompressor.forward` computes `n_blocks = S // m` and returns an empty stream when
`S < m`. The slot loop runs the core on 64 cells with `hca_compress_ratio` 256, so
`n_blocks = 0`, `fused_hca_attention` attends nothing, and `_gate_combine_up` blends the
learned gate weight into a zero tensor. Measured live at step 5000 on THIS ruler:
`|out_comp|` is exactly `0.000` on core blocks 1, 3 and 5 with `g_comp` 0.42-0.52 there,
while the same weights on a 1,152-position token sequence output 680-830 at the same gate
value. The attention branch on those three blocks reads 0.32-0.43x the token figure against
0.56-0.94x on the three CSA blocks.

Three of six core blocks have been running at about half their attention output, silently,
on every slot-loop arm this campaign has scored. Does giving them the branch back change
what the loop earns?

## Hypothesis

H-hca-1: half the core's blocks are structurally weaker on the slot path than on the token
path, and a weak per-pass map is the one thing this campaign has already shown moves loop
contribution (the `norm_match` scale rule took K1−K6 from 0.033 to 0.185 by making the
per-pass map stronger). Restoring the branch is the same kind of intervention on a different
factor, so the forecast K-curve moves.

H-hca-0: the audit's own control says no. `slot-unpack-free` never builds the pooled
compressor at all (`tg_restrict` sets `self.compressor = None`), so it has no dead branch,
and its K1−K6 is 0.0018. And H24 already trained this exact knob for 6,000 steps in 2026-08
without changing the behaviour it was aimed at. On this reading the fix is correct and inert.

## Method

`tul_slot_mux_hca_fix` = `tul_slot_mux_norm_match` plus ONE change:
`model.core_hca_compress_ratio: 16`.

16, not 8 or 32: 64 // 16 = 4 blocks, exactly what the token path gets at seq_len 1024
(1152 // 256 = 4), so the core's compressed branch sees the same number of blocks on both
shapes. 32 gives 2 and is the floor; anything at or above 64 is the defect again. Scoped to
the CORE: a global `model.hca_compress_ratio` would also re-block the two prelude and two
coda HCA layers, which run on all 1,152 positions and do not have the problem, and the arm
would differ by seven modules instead of three.

Verified on a built model rather than argued from the code
(`tests/test_tul_hca_fix.py`, CPU): at S = 64 the core's HCA compressors produce **0** blocks
and a compressed stream whose absolute sum is exactly 0.0 at the shipped ratio, and **4**
blocks with a non-zero stream at 16, while the prelude and coda compressors keep the shipped
ratio. Two sabotage runs fail the suite: unwiring the knob from the core, and letting it leak
out of the core.

Declared confound, unchanged from `tul_a1_hca16.yaml`: `B_a` is `[m, c]`, so the arm has
15,360 FEWER parameters per HCA core block (46,080 in total against 268 M, 0.017 %). It is
NOT iso-parameter, in the direction that makes an improvement harder to explain as capacity
and a regression ambiguous.

Prior, stated so this does not read as the ratio's first run: `tul_a1_hca16.yaml` ran the
same knob on 2026-08-25 (H24) on the PRE-`norm_match` A1 recipe at batch 12, alpha_cap 3.5,
6,000 steps, and was scored on ONE binary question — does the revived branch stop the
takeover divergence. It does not. That arm carried no MUX, never ran under `norm_match`, and
never measured a K-curve. This arm asks the depth question on the current ruler.

Everything else is the ruler: M-next MUX at beta 1 through the tied head, prelude entry,
hinge lambda 100 at 0.9, `core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0,
`ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, fused
kernels.

Runner `arc/run_slotloop3.sh`, a 12-step smoke first, the draw under the sustained tripwire.
Rate rule: tok/s below 8,086 at step 200 skips the arm.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16 — the token
K-curve and the `mux_local` forecast K-curve on 480 rows; `worth_profile.py` and
`slot_state_probe.py` at 5,000; `slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000
by hand. No new wandb keys. The instrument that speaks DIRECTLY to the mechanism is the
geometry audit itself (`lab/divergence/slot_geometry_audit.py`) rerun on this checkpoint: it
should read `n_blocks` 4 and a non-zero `|out_comp|` on blocks 1, 3, 5, and the attention
branch's slot/token ratio on those blocks should stop being the low outlier. That rerun is
part of the readout, not an optional extra, because "the fix reached the trained model" must
be measured and not assumed.

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at 5,000; token K1−K6
+0.0001 [−0.0000, +0.0002], K3−K6 −0.0000; `mux_local` K1−K6 +0.0067 [+0.0053, +0.0081],
K3−K6 +0.0005; wall clock 52 min 38 s; 12,429 tok/s at step 200; cancellation ratio 0.520;
per-pass cotangent 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183; z-optimisation `ce_loop`
4.1173 with the loop ENTRY worth +0.0015. The audit's branch ratios on that checkpoint:
attention slot/token 0.94 / **0.37** / 0.56 / **0.43** / 0.84 / **0.32** on blocks 0-5, the
bold three being the HCA blocks with the dead branch.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire: **82 %**. The ruler ran
  healthy at a preclip max of 78.7 and this arm changes an attention branch's block count,
  not the loop's training. The 18 % is the base rate plus one real mechanism: reviving a
  branch that has been zero for the whole of training changes the map's gain, and the hinge
  is the only thing holding it at 0.89.
- **P-b (tokens).** Token K1−K6 at 5,000 above 0.03: **12 %**. Above 0.10: **4 %**. The coda's
  indifference is upstream of anything an attention branch inside the core can fix, and the
  audit's control arm has no dead branch and no token curve either.
- **P-c (forecast K-curve, the bar).** `mux_local` K1−K6 above 0.02 at 5,000 (the ruler:
  0.0067): **22 %**. K3−K6 above 0.002 (the ruler: +0.0005): **14 %**. Lowest of the four
  arms in this batch, on purpose: the audit's own control (`slot-unpack-free`, no dead
  branch, K1−K6 0.0018) is direct evidence against, and H24 already showed the revived branch
  not doing what it was expected to do once. What keeps it above 10 % is that stronger
  per-pass maps ARE the one thing that has moved this number before (the `norm_match` scale
  rule, 0.033 → 0.185 on the plain loop).
- **P-d (the mechanism reached the model).** The audit rerun on this checkpoint reads
  `n_blocks` 4 and `|out_comp| > 0` on core blocks 1, 3, 5: **97 %**. This is a build-time
  property already verified on a CPU model and it is here so that a flat result cannot be
  explained away as "the fix did not apply".
- **P-e (the branch deficit closes).** The attention branch's slot/token ratio on the three
  HCA core blocks lands above 0.5 (ruler: 0.37 / 0.43 / 0.32; its CSA blocks: 0.56-0.94):
  **65 %**. This is the quantity F1 predicts, measured on a repaired arm rather than inferred
  across arms. Not 90 %, because the gate is free to re-learn its mixture over 5,000 steps
  and may simply move weight off a branch it never used.
- **P-f (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520: **30 %**. No
  mechanism connects an attention block count to how the passes' updates agree; this is
  recorded because every arm files it.
- **P-g (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **55 %**. Three
  of six core blocks getting their compressed branch back is a real capacity change in the
  arm's favour and 0.017 % fewer parameters is not. Not a verdict either way.
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **92 %**. The compressed
  branch goes from 0 blocks to 4 on 64 cells in three core blocks per pass; the pooled
  projections are already computed. A small addition on the smallest tensor in the model.

## Binding

- P-c TRUE ⇒ the dead branch WAS load-bearing for depth and the audit's cross-arm inference
  was wrong. Every slot-loop number in this campaign was measured on a core running at
  reduced attention capacity, and the ruler has to be re-cut at ratio 16 before any further
  arm is read against it.
- P-c FALSE and P-e TRUE ⇒ the defect is fixed, measurably, and it changes nothing about
  depth. F1 is then closed as a lever on a REPAIRED arm rather than on a control that differs
  in four ways, which is the outcome this arm exists to buy. Whether the ratio ships in
  `tul_short.yaml` anyway is a separate call: it is a correctness fix, not a depth fix.
- P-e FALSE ⇒ the gate moved its mixture off the revived branch and the arm did not test what
  it set out to test. Report that, and the follow-up is the gate's own weight over training
  (`g_comp` on blocks 1/3/5), not another ratio.

## Not verified before launch

The arm has not run on a GPU. The only measurements behind it are the audit's readings on the
UNREPAIRED ruler and a CPU build check on a tiny model at the real sequence length: nothing
here has trained with the branch alive under `norm_match`. The gate's response over 5,000
steps (does `g_comp` stay at 0.42-0.52 once the branch is non-zero?) is unmeasured and is the
mechanism most likely to make P-e false. The parameter difference is arithmetic from the
`B_a` shape, not a build-time count on the real model. The interaction with the ternary QAT
of the compressor's projections is untouched and unexamined. `torch.compile` behaviour at the
new block count is untested until the runner's 12-step smoke.
