# Planned: per-pass low-rank deltas on the M-next slot loop's shared core

Status: planned

Date: 2026-09-10 (frozen before launch; Wolfe's framing after the per-pass gradient probe:
"the loop issue is credit assignment"). Arc: `2026-09-04-loop-contribution-arc.md`.
Follows `results/2026-09-10-slot-gradient-probe/README.md`. Sibling arm on the same
question, queued first:
[`2026-09-10-arc-slot-mnext-progressive.md`](2026-09-10-arc-slot-mnext-progressive.md).
Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

Same reading as the progressive arm, opposite intervention. The gradient probe shows the
six passes of the shared core asking for near-orthogonal updates:
`|sum_t dW_t| / sum_t |dW_t|` is 0.520 combined on `slot-mux-norm-match` (0.601 token CE,
0.550 MUX), against 0.41 for six orthogonal equal-norm vectors, with per-pass cosines to
the total from 0.27 to 0.75 and pass 6 carrying 36 % of the norm sum while passes 2-4
carry 10-12 % each. The progressive arm asks whether that disagreement is a training
artefact that a random no-grad prefix removes. This arm asks the other question: if the
passes genuinely want to be different maps, GIVE them a way to be — and see whether the
loop then earns depth.

Bae et al. 2024 (Relaxed Recursive Transformers) is exactly that relaxation: keep the
shared weights, add a small per-recursion-index low-rank delta. Their result is that a
looped model recovers most of the quality of an unshared one with a few percent of the
parameters. Here the recursion index is the slot loop's pass.

## Hypothesis

H-lora-1 (for): the shared map is over-constrained. Six passes fighting over one weight
matrix is why nothing specialises and why the loop reads flat; a rank-32 escape valve per
pass lets pass 6 do something pass 1 does not, and the K-curve separates.

H-lora-0 (against): the passes are not fighting over capacity, they are fighting over an
input that barely changes. The levers panel measured the core blocks moving the slot state
by 2-20 % per pass against 86-110 % on token states, and every entry, coda, target and
depth lever this tree has tried has read flat. Adding per-pass parameters to a
near-identity map gives every pass its own way of doing nearly nothing, and the arm reads
flat like the rest — possibly with a slightly better CE from the extra 2.35 % of
parameters, which would be a capacity reading, not a depth one.

There is a specific way H-lora-1 could be TRUE and still be bad news: the deltas could
absorb the per-pass disagreement into parameters the loop does not share, leaving the
SHARED map exactly as it was. The cancellation ratio measured on the shared weights alone
is the instrument that separates these, which is why it is a prediction below.

## Method

`tul_slot_mnext_per_pass_lora` = `tul_slot_mux_norm_match` (the ruler `slot-mux-norm-match`)
plus ONE change: `tul.pass_lora_rank: 0 -> 32`, targets `[attn, mlp]`.

The mechanism, as implemented (`morph/model/mhc.py::PassLoRA`,
`morph/model/mhc.py::MORPHBlock.forward`, threaded from
`MORPHTransformer._apply_core_step` as `pass_idx=iter_idx`): each of the six core blocks
carries stacked parameters `A [T, r, C]` and `B [T, C, r]` per targeted SUBLAYER, and pass
`t` adds `B_t (A_t x)` to that sublayer's output. `B` is exactly zero at init, so the arm
is bit-identical to the ruler at step 0 (verified: same loss, same logits, same base
weights, same 208 base gradient tensors on the tiny CPU model at ranks 8, 32-mlp-only and
4-attn-only). The pass index is an INDEX INTO A STACKED PARAMETER, never a Python branch,
so `torch.compile` sees no data-dependent control flow.

GRANULARITY, stated because it is not the paper's. Bae puts a LoRA on each linear of the
shared layer. This puts ONE delta on each targeted sublayer: `attn` is the whole attention
branch (input `norm_attn(x)`, output where the output projection's contribution lands) and
`mlp` is the whole SwiGLU (input `norm_mlp(x)`, output `d_model`). What it cannot express
is a delta acting inside the SwiGLU nonlinearity, or on q/k/v before the score. Reasons:
one mechanism at one code site; the attention projections are not reachable as modules on
this tree (several read `.weight` and matmul it, and the block's attention takes no
iteration argument); and plain `nn.Parameter` tensors on a non-Linear module are invisible
to BOTH weight schedules — ternary QAT walks `nn.Linear` / `nn.Embedding` / CMS modules,
and the CMS prune, the MORTAR carve and the deploy packer walk `MortarLinear` /
`CMSBlockLinear`. The deltas are therefore never ternarised, pruned, carved or packed by
construction rather than by an exclusion list, and both facts have their own tests.

COST, measured by building both models on CPU: 6,291,456 new parameters
(268,223,095 -> 274,514,551, **+2.35 %**), bf16 compute with fp32 master weights like every
other parameter here. FLOPs: 262,144 per token per core block per pass (two targets, two
matmuls each) against 17,301,504 for that block's SwiGLU alone, i.e. **+1.52 %** of the
MLP's cost, and the core runs on the ~64 compact slot positions rather than 1152 — about
604 MFLOPs per row per forward at six passes.

Everything else is the ruler: M-next MUX at beta 1 through the tied head, prelude entry,
hinge lambda 100 at target 0.9, `core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0,
`ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000,
`tg_scoped_kernels`.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips
the arm and the queue continues (the ruler read 12,429).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16,
reporting both the token K-curve and the `mux_local` forecast K-curve on 480 rows;
`worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind does both);
`slot_anatomy.py` and `slot_gradient_probe.py` at 5,000 by hand. Note for whoever reads the
gradient probe on this arm: its tap sits on the CORE WEIGHTS, so the cancellation ratio it
reports is still the SHARED map's, with the LoRA parameters excluded — which is exactly the
number this arm needs, and also the reason the probe's own self-check
(`sum_t tap_t == leaf.grad`) still holds unchanged.

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at step 5,000
(`sweep_5000.log`; the trainer's final val_loss was 4.3775 and the last `[VAL]` line kept
in `run.log` is step 4,750 at 4.4486 — there is no `[VAL 5000]` line in that log); token
K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000; `mux_local` K1-K6 +0.0067
[+0.0053, +0.0081], K3-K6 +0.0005; wall clock 52 min 38 s for 5,000 steps
(21:41:42 -> 22:34:20 in `arc/queue.log`); combined cancellation ratio 0.520.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **70 %**. Reasoning: the ruler ran HEALTHY (max preclip 78.7 at step 245)
  and B starts at zero, so the first few hundred steps are the ruler's own trajectory. But
  this arm ADDS a per-pass degree of freedom to the map's gain, and the two detonation
  modes on this tree are exactly that — a first-pass scale runaway under a passing hinge
  (E7, E12, E13-mtp4) and a backward product (E14). The gain hinge measures ONE random
  grad iteration per step (`slot_gain_all_iters` is off on this recipe), so a map that is
  now genuinely different per pass is sampled at 1 of 6 passes per step. That is the
  known blind spot from arc E2 and it is why this is my lowest survival number. If it
  detonates, the first thing to check is `loop/core_gain_t0`, which led the tripwire by
  ~500 steps on E7/E13.
- **P-b (tokens above the plain ruler).** Token K1-K6 at 5,000 above 0.03: **15 %**.
  Above 0.10: **5 %**. Reasoning: as in the progressive arm — every slot arm under
  norm_match on any lever has read at or under 0.005, and the coda reads the slot state
  once, so per-pass capacity does not create anything for it to read at depth 6 that was
  not there at depth 1. I give this slightly more than the progressive arm only because a
  per-pass delta can, in principle, make the last pass a different map from the first,
  which is the one structural thing none of the earlier levers changed.
- **P-c (forecast K-curve).** `mux_local` K1-K6 above 0.02 at 5,000: **35 %**. Reasoning:
  the MUX is the loss that pays for the loop and the K-curve that is not zero (0.0067 on
  the ruler), and if per-pass specialisation is worth anything it shows on the target the
  loop is directly supervised on. Higher than the progressive arm's 30 % because this arm
  gives the passes capacity rather than only changing how they are trained. K3-K6 above
  0.002: **20 %**.
- **P-d (cancellation on the SHARED map).** Combined cancellation ratio from
  `slot_gradient_probe.py` at 5,000 above 0.60 (the ruler: 0.520): **45 %**. Reasoning:
  this is the mechanism's most direct consequence — if the deltas absorb what the passes
  disagreed about, the shared weights should receive a more consistent update. Nearly even
  money because the alternative (the deltas absorb the disagreement AND the shared map's
  gradient stays exactly as scattered) is just as plausible, and that outcome is the one
  that would say the disagreement is not about capacity at all.
- **P-e (the deltas actually move).** At step 5,000, the mean `|B_t|` over passes and
  blocks is above 1e-3, and the per-pass deltas are NOT all equal (max pairwise relative
  difference of `|B_t|` across t above 20 %): **70 %**. Reasoning: B has a direct gradient
  from step 1 and 5,000 steps at 1e-4 is enough to move it off zero; the passes see
  different states, so their gradients differ. The 30 % is for the arm reading like the
  `reread` lever, whose W_o reached norm 9.3 and still produced a constant 5 % per-pass
  effect — "the parameters moved" and "the mechanism did something" are different claims,
  and this prediction only settles the first.
- **P-f (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **60 %**.
  Below the ruler (better): **45 %**. Reasoning: +2.35 % parameters usually buys a small
  CE gain at a fixed step count, and unlike the progressive arm nothing is being removed
  from the gradient. A CE gain here is a CAPACITY reading and is NOT evidence for
  H-lora-1; only the K-curves and P-d speak to depth (Wolfe 2026-09-09: short-horizon CE
  cannot rank looped against unlooped).
- **P-g (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **85 %**. Reasoning:
  +1.52 % of the MLP's FLOPs on the compact slot sequence is nothing, but four extra
  small matmuls per block per pass is 48 extra kernel launches per forward on a step that
  is launch-bound, and the optimizer now carries 6.29M more parameters.

## Binding

- P-c TRUE and P-d TRUE => capacity was part of the problem: the next step is the two
  mechanisms together (progressive + per-pass deltas) and a rank sweep, then a 20k horizon.
- P-d TRUE, P-c FALSE => the deltas tidied the shared map's gradient and bought no depth.
  That is the sharpest possible refutation of "credit assignment is the bottleneck": the
  gradient got what it asked for and nothing downstream moved.
- P-e TRUE with everything else FALSE => the mechanism ran, the parameters moved, the loop
  did not care. Same reading as the `reread` lever, and it belongs in the same list.
- P-e FALSE => the arm did not run its own mechanism; the result is void and the finding
  is a build defect, not a fact about the loop.
- All flat, on both this arm and the progressive one => both credit-assignment attacks have
  failed, and the honest next question is not another lever on this loop: it is whether
  ~50 slot cells carrying a rank-2-to-5 state can support a depth-6 map at all.

## Not verified before launch

Neither arm has run on the GPU. `torch.compile` behaviour of the stacked-parameter index
is untested until the runner's 12-step smoke: it is an index with a Python int into an
`nn.Parameter`, the same shape the ReMoE router's `iter_embed` already uses, but the core
MLPs are compiled with `dynamic=True` on this tree and no one has traced the new path.
Memory at the real shape is unmeasured — the arm's arms already sit at ~26 GB resident on
a 31.4 GB card, and 6.29M extra parameters plus their AdEMAMix state is roughly 100 MB,
which should fit but has not been observed. The hinge is NOT adapted to a per-pass map
(`slot_gain_all_iters` stays false, so it samples one of six passes per step); arc E2 says
that is the wrong instrument for a per-iteration map, and running it that way is a
deliberate one-factor choice against the ruler, not a claim that the instrument is right.
The gradient probe has not been run against a model carrying pass-LoRA parameters; its tap
should ignore them (it walks core weights), but that is an argument from the code, not a
measurement. Rank 32 is Bae's own order of magnitude, not a value tuned here; no rank
sweep is planned before this draw. Only two sublayer targets are covered, and a delta
inside the SwiGLU nonlinearity is not tested at all.
