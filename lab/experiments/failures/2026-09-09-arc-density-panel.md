# Planned: ARC density panel — does the production prune density change what the loop earns?

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "we need to test density. i am going to bed.
get this done"). Arc: `2026-09-04-loop-contribution-arc.md`, density row. Chained behind
the depth-candidates panel in the queue (`arc/run_density.sh` waits for `DEPTHCAND
COMPLETE`).

## Question

Every loop-depth panel on this arc ran DENSE: prune, carve and routing off. The production
recipe (`morph/configs/base.yaml`) prunes the MLP hidden bank to density 0.25 with MORTAR
128×128 blocks, and the looped core's MLPs are in that bank. So every depth number the arc
has produced was measured on a model the production recipe never ships. Two things can
happen at density: the per-pass map loses capacity and converges faster (the loop earns
less), or the model, denied width, learns to reuse depth (the loop earns more). Neither is
measured. This panel prunes the Parcae-entry recipe to 0.5 and 0.25 within a 5,000-step run,
prune only (masked-dense execution; no carve, no routing), and reads the K-curve, the end
point against the dense Parcae-entry arm, and the price against the depth-1 control.

## Hypothesis

H-density: density is a tax, not a depth lever. Both pruned arms are worse than the dense
arm at every depth by an amount that grows with sparsity, and their K3−K6 stays under 0.01.
H-density′: width denied, depth reused. The quarter arm's K3−K6 rises above 0.02 with the CI
above 0, at whatever end-point cost. H-density″: the loop entry does not survive the prune
under ternary (a detonation or a hinge-free spike train after the first events).

## Method

Base: `notul_parcae_entry` (the Parcae-entry loop arm's recipe: seq 1024, batch 6, 5,000
steps, ramp 1000 then flat 1e-4, ternary backbone, AdEMAMix β1=0, noise state init, all-dim
carry with learned B, no fixed-point term, grad probe every step, web text). Arms:

| arm | config | prune schedule |
| --- | --- | --- |
| `density-half` | `notul_density_half` | `prune_start 1500`, `prune_interval 25`, `prune_rate 0.03`, `target_density 0.5` |
| `density-quarter` | `notul_density_quarter` | the same with `target_density 0.25` |

The dense control is the Parcae-entry arm already on disk (its 5,000 sweep and token file);
the price control is the depth-1 arm of the same panel. `compact_step` and `route_start`
stay never (the carve changes module classes and the routing head is a second factor; the
depth question is about the pruned MAP). The rule drops floor(alive × 0.03) blocks per
event with a one-block floor, so from step 1,500 the half arm reaches its target after about
23 events (step ~2,075) and the quarter arm after about 46 (step ~2,650), leaving over 2,000
steps at the final density before the 5,000 readout. The 2,500 checkpoint of the quarter arm
is MID-PRUNE (density about 0.3) and is read as such. Saliency is `cms_score_mode: taylor`
(the production choice).

Runner `arc/run_density.sh`: per arm a 12-step smoke, the draw, the sustained tripwire, a
`PRUNE <arm> last:` queue note quoting the trainer's last `[prune] step … density=` line (a
run that never pruned is the CLAUDE.md gotcha and is recorded as `NO PRUNE LINES`), then
`core_depth_sweep.py` at 2,500 and 5,000 over 480 rows at depths 0,1,2,3,6,9,12,16 with
per-token files, `core_anatomy.py --rows 3 --depth 8` and `core_init_probe.py --rows 96`.
Commit pinned in `arc/DENSITY_COMMIT`. Results to `lab/experiments/results/2026-09-09-density-panel/`,
token files and probes under `ignored/experiment-artifacts/2026-09-09-density-panel/`.
Scored by `score_density.py` with the shared readers in `lab/divergence/sweep_score.py`.
Every sweep's CE at depth 6 is checked against the trainer's `[VAL]` at the same step before
it is read (the E17 rule), and every arm's final density is read from the trainer's own
`[prune]` line, never assumed from the config.

**Method amendment 1 (2026-09-09 10:20, both arms trained, quarter not yet swept).** Wolfe: at
5,000 steps final losses cannot rank densities or looped against unlooped models (they break
even or win at longer horizons). P-den-c and P-den-e are scored as written and reported as
5,000-step readings; the panel's finding is P-den-d and P-den-f, the loop contribution and its
mechanism at each density. Predictions untouched.

## Predictions (frozen)

- **P-den-a (survival).** Reaches 5,000 with the sustained tripwire silent: half **85 %**,
  quarter **75 %** (the topology shock lands on a ternary map under a flat 1e-4 with the
  noise-start loop; base.yaml's cadence is 7× gentler).
- **P-den-b (density fell).** The trainer's last `[prune]` line reads within 0.02 of the
  target for both arms: **90 %**. The quarter arm's 2,500 checkpoint reads between 0.25 and
  0.40: **80 %**.
- **P-den-c (the tax).** Token-paired CE at depth 6 at 5,000 against the dense Parcae-entry
  arm: half worse by 0.03–0.10: **55 %**; quarter worse by 0.10–0.25: **60 %**; quarter worse
  by more than 0.25: **25 %**; either arm within 0.02 of dense: **10 %**.
- **P-den-d (depth).** K3−K6 > 0.02 with the CI above 0: half **15 %**, quarter **25 %**.
  K1−K6 larger than the dense arm's 0.033 by more than 0.02: quarter **40 %**, half
  **25 %**. Both arms' K6−K12 within ±0.005: **75 %**.
- **P-den-e (the price).** Quarter at depth 6 loses to the dense depth-1 control at depth 1
  (its CI above 0): **55 %**. Half loses: **30 %**.
- **P-den-f (anatomy).** Consecutive state movement at iteration 6 above 10 % on the quarter
  arm: **30 %**. Branch out/in at iteration 6 on the quarter arm lower than the dense arm's
  (0.24 max) by more than 0.05: **55 %**. Init probe spread at T=8 within 0.01 on both:
  **75 %**.
- **P-den-g (cost).** Wall clock within 1.1× of the dense arm (masked-dense execution costs
  the same FLOPs): **85 %**. Peak allocated under 16 GB: **85 %**.

## Binding

- P-den-d TRUE on the quarter arm (K3−K6 > 0.02, CI above 0) ⇒ density is a depth lever:
  every future depth panel runs at the production density, and the next test is whether the
  earning survives the carve.
- P-den-c TRUE and P-den-d FALSE ⇒ H-density: density is a pure tax at this scale and depth
  is unrelated to it; the depth panels stay dense and the record says why.
- P-den-a FALSE ⇒ the trip step, the density at the trip and the probe go into the
  divergence README's second-hold table; the arm re-runs ONCE at `prune_rate 0.015`
  (twice as many events, half the shock).
- P-den-b FALSE ⇒ the run did not prune; nothing is read from it; the schedule is fixed and
  the arm re-runs.
- NO 20k run from this experiment.

## Not verified before launch

No GPU forward of either config before the queue's own 12-step smoke (which cannot reach
`prune_start`). The compressed schedule is exercised only by a CPU test on a tiny model.
The prune mask's persistence through the lab loaders is verified on the CPU by that test
(the dead weights are zero in the state dict either way). The tripwire's behaviour on a
prune-event step (a saliency drop is a topology change; the probe's `preclip/total` may
spike on the event step itself).

## Results

**Scope note (2026-09-09 10:30).** This panel ran on a misreading of Wolfe's "test density": he
meant the weight data type (ternary against bf16), which the depth-candidates panel's
bf16-core arm and the precision-axis arm (`2026-09-09-arc-precision-axis.md`) cover. This
record stays as what it is, a block-prune side result on loop contribution, and no prune
panel follows it. Everything below is a 5,000-step reading; the CE gaps rank nothing.

Run 2026-09-09 08:18–10:28 at `137893f`. Scored by `../results/2026-09-09-density-panel/score_density.py`
(`score_density.txt`); token files and probes under `ignored/experiment-artifacts/2026-09-09-density-panel/`.
The E17 rule passes on all four sweeps (gaps 0.028–0.032).

**P-den-a: TRUE ×2.** Both HEALTHY (probe peaks 215 at 225 and 43 at 2461). **P-den-b: TRUE.**
The trainer's own lines: half reaches 0.5000 at step 2,025 after 22 events; quarter reaches
0.2500 at step 2,525 after 42; the quarter arm's 2,500 checkpoint reads 0.2538 (predicted
0.25–0.40: TRUE).

**P-den-d (loop contribution) at 5,000:**

| arm | density | K1−K6 | K3−K6 | branch out/in @6 (max) | carrier rank t8 |
| --- | --- | --- | --- | --- | --- |
| parcae-entry (dense) | 1.0 | +0.033 | +0.0009 | 0.236 | 73.0 |
| density-half | 0.5 | +0.043 [+0.042, +0.045] | +0.0028 [+0.0025, +0.0032] | 0.114 | 8.5 |
| density-quarter | 0.25 | +0.074 [+0.072, +0.075] | +0.0054 [+0.0050, +0.0058] | 0.058 | 4.4 |

K3−K6 > 0.02: FALSE ×2. Quarter's K1−K6 larger than dense by > 0.02: TRUE (half FALSE).
Both converge by 6 (TRUE). The mechanism reads the other way from the bf16-core arm: the
pruned MLP branches emit LESS per pass (0.058 and 0.114 against 0.236) and the carrier's
rank collapses (4.4 and 8.5 against 73), so the larger K1−K6 is a larger SHARE of a
weaker model's loss carried by the first passes, not more computation in later ones. The
quarter arm's mid-prune 2,500 checkpoint read K1−K6 0.126 and K3−K6 0.010, the largest of
the panel, and both fell by 5,000 as the model settled at its density.

**P-den-c / P-den-e (5,000-step CE readings, not rankings):** half@6 − dense@6 = +0.0101
[+0.0070, +0.0132] (predicted 0.03–0.10: FALSE; within 0.02: TRUE); quarter@6 − dense@6 =
+0.0899 [+0.0870, +0.0929] (predicted 0.10–0.25: FALSE, just under). Against the depth-1
control: half +0.0063, quarter +0.0861. The half arm's gap to dense shrank from 0.043 at
2,500 to 0.010 at 5,000 on the trainer's batches; the quarter arm's from 0.174 to 0.097.

**P-den-f:** movement at iteration 6: 3.3 % and 3.6 % (> 10 %: FALSE ×2); branch out/in
lower than dense by > 0.05: TRUE ×2; init probe spreads 0.0003 and 0.0035 at T = 8 (TRUE
×2; the quarter arm's zero and small-noise starts read 0.003 under its prelude start).
**P-den-g:** 0.89 h each, 10.63 GB (TRUE ×2).

## Verdict

**failure** (H-density's CE bands missed on both arms; the panel answered a question the
arc did not ask). On loop contribution the block prune is not a lever: the quarter arm's
larger K1−K6 comes with a weaker per-pass map (branch out/in 0.058) and a rank-4 carrier,
the signature of dependence without computation, and K3−K6 stays under 0.006. The
production prune at 5,000 steps also reads as a recoverable CE gap (half within 0.02 of
dense by 5,000), which is a horizon reading and ranks nothing.

Binding applied: P-den-d FALSE on K3 ⇒ the depth panels stay dense; the record says why.
The precision axis (ternary against bf16) is the live lever and runs next.

## Updated hypothesis

Removing MLP blocks weakens the per-pass map the same way ternary weights do (smaller branch
output per pass) and adds a rank collapse of the carrier; neither adds computation to later
passes. The loop-contribution lever is the per-pass strength of the core's MLPs, which the
precision axis and the under-ternary scale work address. The production prune's effect on
the loop at a real horizon is untested and is not this arc's question.
