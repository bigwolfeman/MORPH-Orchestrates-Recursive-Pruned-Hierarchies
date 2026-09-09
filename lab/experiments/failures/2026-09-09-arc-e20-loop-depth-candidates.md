# Planned: ARC E20 — three loop-depth candidates on the E19 loop arm: dense core, a carry that can train, Parcae's depth schedule

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "yeah this should be good. have a subagent
prepare the runs. then kick them off. do not name arms with single letters. Use descriptive
names"). Arc: `2026-09-04-loop-contribution-arc.md`, row E20. Chained behind E19's
completion in the queue (`arc/run_e20.sh` waits for `E19 COMPLETE`).

## Question

E19 (`2026-09-09-arc-e19-parcae-loop-entry.md`) rebuilt the loop's entry the way Parcae does
it (state from noise, e re-injected on every dim through a learned B, no fixed-point term)
and got a better model (−0.023 nats token-paired at depth 6 against the E18 plain arm) whose
loop still earns 0.033 nats past pass 1 and 0.0009 past pass 3. The state now moves like
Parcae's (37/23/7/5/3/2 % per pass), the fixed point no longer depends on the start, and the
injection did not train (B diag 1.035, decay 0.444 from 0.447, dt 0.802 from 0.8 after 5,000
steps). The hyper-connection residual is a plain residual at init and its mixing map did
train, so it is off the list. Three candidates remain, each a single factor on the loop
arm's config, ranked by how far MORPH's choice sits from Parcae's:

1. **The depth schedule.** MORPH trains full BPTT over a Poisson draw of mean 6, max 8;
   Parcae truncates backprop to the last 4 of a mean-8 draw. E6 showed the draw sets depth
   dependence (mean 16: K3−K6 0.277, and 0.104 nats worse). Full BPTT with depth-1 and
   depth-2 samples in the draw trains the state to be readable at pass 1, which is what a
   flat K-curve looks like.
2. **The per-pass map's precision.** MORPH's core is ternary under STE; Parcae's blocks are
   dense bf16. A diagnostic, not a recipe (Wolfe's standing rule: no dense-then-ternary).
3. **The carry never trains.** B, decay and dt sit at init after 5,000 steps at lr 1e-4
   under AdEMAMix; Parcae's B diagonal reached 2.4 under Muon at 8e-3.

E20 runs one arm per candidate on `notul_parcae_entry` and scores each on its K-curve, on
token-paired CE against the E19 loop arm, and on the depth-1 price (against E19's
`e19-d1`, the model trained at depth 1) so a manufactured dependence cannot pass as a win.

## Hypothesis

H20: the depth schedule is the lever. `draw8-bptt4` earns depth (K3−K8 > 0.02) and is the
only arm that does; `dense-core` and `carry-lr20x` leave the K-curve where E19 left it
(K3−K6 under 0.01). H20′: the ternary core is the limit (`dense-core` alone moves K3−K6).
H20″ (the null): no arm moves K3−K6 past 0.01 and the loop's flatness lives in the readout,
not in the map, the carry or the schedule.

## Method

Base: `notul_parcae_entry` (E18 plain recipe + the E19 entry: seq 1024, batch 6, 5,000 steps,
ramp 1000 then flat 1e-4, ternary backbone, AdEMAMix β1=0, web text, checkpoints at 2,500
and 5,000, grad probe every step). Arms, one factor each:

| arm | config | the one change |
| --- | --- | --- |
| `depthcand-dense-core` | `notul_depthcand_dense_core` | `model.ternary_scope: backbone_no_core` — ternary everywhere except modules under `core.`; prelude and coda stay ternary |
| `depthcand-carry-lr20x` | `notul_depthcand_carry_lr20x` | `training.injection_lr_mult: 20` — the parameters whose name contains `injection` (B, log_A, log_dt) train at 2e-3 in their own optimizer group; everything else at 1e-4 |
| `depthcand-draw8-bptt4` | `notul_depthcand_draw8_bptt4` | `model.mean_depth: 8`, `max_depth: 12`, `bptt_depth: 4` — Parcae's per-sequence Poisson mean 8 with backprop through the last 4 iterations only |

Order: dense-core, carry-lr20x, draw8-bptt4. Runner `arc/run_e20.sh` (waits for `E19
COMPLETE`, then per arm: 12-step smoke, the draw, the sustained tripwire, then
`lab/divergence/core_depth_sweep.py` at 2,500 and 5,000 over 480 rows at depths
0,1,2,3,6,8,9,12,16 with per-token files, `core_anatomy.py --rows 3 --depth 8` and
`core_init_probe.py --rows 96` at 5,000). Commit pinned in `arc/E20_COMMIT`. Results to
`lab/experiments/results/2026-09-09-arc-e20/`; token files and probes under
`ignored/experiment-artifacts/2026-09-09-arc-e20/`. Scored by `score_e20.py` (written after
launch; reads only the JSONs, logs and constants) with the shared readers in
`lab/divergence/sweep_score.py`.

Trained depth per arm: 6 for dense-core and carry-lr20x, 8 for draw8-bptt4. Its K-differences
are K1−K8 and K3−K8; its end point is read at depth 8. Every sweep's CE at the trained depth
is checked against the trainer's `[VAL]` at the same step before it is read (the E17 rule; the
two cuts differ by about 0.025 on this stream, as E18 and E19 both showed).

The price: CE(arm @ trained depth) − CE(`e19-d1` @ 1), token-paired, from E19's d1 sweep.
The end point: CE(arm @ trained depth) − CE(`e19-loop` @ 6), token-paired (3.9573 on the
480 rows). The draw8 arm's wall clock is longer by construction (mean 8 against 6, less
backward); its matched-wall-clock reading is the E19 loop arm's held-out loss at the step
the draw8 arm reaches at the same wall clock, from the two run logs.

## Predictions (frozen)

- **P20a (survival).** Reaches 5,000 with the sustained tripwire silent: dense-core **85 %**;
  carry-lr20x **60 %** (2e-3 on a 1024×1024 B under AdEMAMix β1=0 is the riskiest thing in
  the panel); draw8-bptt4 **80 %** (truncated backprop under ternary; retention is off, so
  the l2cap leak has no channel).
- **P20b (depth).** At 5,000, K3−K6 > 0.02 with the CI above 0 (K3−K8 for draw8):
  dense-core **25 %**; carry-lr20x **30 %**; draw8-bptt4 **55 %**. K1−K6 > 0.10 (K1−K8 for
  draw8): dense-core **20 %**; carry-lr20x **25 %**; draw8-bptt4 **60 %**. Every arm's
  K6−K12 within ±0.005 (K8−K12 for draw8): **70 %**.
- **P20c (end point vs the E19 loop arm, token-paired at the trained depth).** dense-core
  better by more than 0.02: **55 %**. carry-lr20x within ±0.02: **60 %**. draw8-bptt4 worse
  by more than 0.03 at matched steps: **55 %**; better at matched steps: **15 %**.
- **P20d (the price).** Arm at its trained depth beats `e19-d1` at depth 1 (CI above 0 in
  d1's favour is a FAIL): dense-core **70 %**; carry-lr20x **55 %**; draw8-bptt4 **40 %**.
- **P20e (anatomy).** carry-lr20x: B's diagonal mean moves from 1.0 by more than 0.3:
  **65 %**; decay mean moves from 0.447 by more than 0.05: **60 %**. Consecutive state
  movement at iteration 6 above 10 %: draw8-bptt4 **45 %**, dense-core **25 %**,
  carry-lr20x **25 %**. Init probe: every arm's four starts agree within 0.01 at T=8 (the
  E19 property survives every factor): **80 %**.
- **P20f (cost).** dense-core wall within 1.1× of the E19 loop arm (1.00 h): **70 %**.
  draw8-bptt4 wall between 1.15× and 1.5×: **65 %**. Every peak allocated under 16 GB:
  **85 %**.

## Binding

- draw8-bptt4 earns depth (P20b TRUE) AND beats d1 (P20d TRUE) AND is within 0.03 of the
  E19 loop arm at matched wall clock ⇒ the schedule is the lever: the next panel separates
  the draw from the truncation (mean 8 full BPTT; mean 6 bptt 4), and the arc's standing
  recipe becomes the winner.
- draw8-bptt4 earns depth but is more than 0.03 behind at matched steps and loses the price
  ⇒ E6 again: dependence without value; the schedule lane closes and the record says so.
- dense-core moves K3−K6 > 0.02 ⇒ the ternary map is the limit. Ternary stays (no dense
  warmup, Wolfe's rule); the next work is core precision under ternary (threshold, scale
  group, a 2-bit core) as an Agent Note before any run.
- carry-lr20x moves B (P20e TRUE) and not K ⇒ the carry is not the lever. carry-lr20x
  detonates ⇒ its trip step and probe go into the divergence README's table and it re-runs
  ONCE at 5×.
- Every arm FALSE on P20b ⇒ H20″: the map, the carry and the schedule are all cleared and
  what is left is the readout. The next instrument is a per-iteration readout probe (a
  linear head fit on h_t for each t) before any further arm; and the paid TUL loop gets the
  E19 entry, since it is the one loop that earned depth (0.168 at 5k).
- NO 20k run from this experiment.

## Not verified before launch

No GPU smoke of any arm before the queue's own 12-step smoke (E19 owns the GPU until the
queue reaches E20; a failed smoke skips that arm and is recorded). `bptt_depth 4` under
`max_depth 12` on the plain path with the noise init (the code path exists and is
documented as Parcae's form, never run on this entry). A third optimizer group under
`injection_lr_mult` on a RESUME (no resume is planned). `backbone_no_core` on a compiled
model (the tests build eager CPU models). The draw8 arm's memory at max depth 12.

## Results

**Horizon caveat (Wolfe, 2026-09-09 10:20).** 5,000 steps is too short to rank a looped model against a depth-1 model, or one density against another, on final loss: deeper and sparser models converge slower and the break-even sits at a longer horizon. The CE differences below are reported as what they are, a matched-step reading at 5,000, and are NOT verdicts on looping or on ternary. The finding of this panel is the LOOP CONTRIBUTION: the K-curve, the branch ratios and the state movement.

Run 2026-09-09 05:20–08:17 at `5c6dec1` (the arms and configs carry descriptive names:
`depthcand-dense-core`, `depthcand-carry-lr20x`, `depthcand-draw8-bptt4`; references
`parcae-entry` and `plain-depth1` from the Parcae-entry panel). Scored by
`../results/2026-09-09-arc-e20/score_e20.py` (`score_e20.txt`); token files and probes under
`ignored/experiment-artifacts/2026-09-09-arc-e20/`. The E17 rule passes on all six sweeps
(gaps 0.020–0.035, the two-cut difference).

**P20a (survival): TRUE ×3.** dense-core HEALTHY (probe peak 100 at 612), carry-lr20x
HEALTHY (40 at 226; the 20× rate did not detonate), draw8-bptt4 HEALTHY (32 at 471).

**P20b (depth): dense-core moves, the others do not.** At 5,000:

| arm | K1−Kt | K3−Kt | Kt−K12 |
| --- | --- | --- | --- |
| dense-core (t = 6) | **+0.168** [+0.165, +0.171] | +0.0119 [+0.0113, +0.0125] | −0.002 |
| carry-lr20x (t = 6) | +0.089 | +0.0072 [+0.0066, +0.0078] | −0.000 |
| draw8-bptt4 (t = 8) | +0.060 | +0.0022 [+0.0018, +0.0025] | −0.000 |
| parcae-entry (t = 6) | +0.033 | +0.0009 [+0.0007, +0.0011] | −0.000 |

K3−Kt > 0.02: FALSE on all three (dense-core 0.012 with the CI above 0, 13× the ternary
arm). K1−Kt > 0.10: TRUE on dense-core only (5× the ternary arm). Every arm converges by
its trained depth (Kt−K12 within 0.005: TRUE ×3).

**P20c (end point vs parcae-entry@6, token-paired, 491,520 tokens):** dense-core
**−0.0298** [−0.0326, −0.0271] (better by > 0.02: TRUE); carry-lr20x +0.0371 [+0.0342,
+0.0402] (worse by > 0.03: TRUE; "within ±0.02" FALSE); draw8-bptt4 +0.0921 [+0.0891,
+0.0952] (worse by > 0.03 at matched steps: TRUE; better: FALSE).

**P20d (the price vs plain-depth1@1, token-paired):** dense-core **−0.0336** [−0.0365,
−0.0309] (ahead of the depth-1 model at 5,000: TRUE); carry-lr20x +0.0333 (FALSE); draw8-bptt4
+0.0883 (FALSE); parcae-entry −0.0038 (the panel's base, for reference).

**P20e (anatomy at 5,000, depth 8, 3 rows):** carry-lr20x's B diagonal mean 1.744 (moves
> 0.3: TRUE), dt 0.475 from 0.8, decay 0.421 (> 0.05 from 0.447: FALSE); dense-core B
1.046, draw8 B 1.026 (FALSE). Consecutive state movement at iteration 6: dense-core 6.1 %,
carry 2.1 %, draw8 4.2 %, base 2.1 % (> 10 %: FALSE ×3). Branch out/in at iteration 6,
the mechanism: dense-core MLP **0.75–0.96** and attention 0.14–0.43, against the ternary
base's MLP 0.19–0.24 and attention 0.08–0.21; carry-lr20x MLP 0.33–0.42 (its carrier
scale changed), draw8 max 0.26. Carrier rank at t = 8: dense-core 87.7, base 73.0, draw8
14.8, **carry-lr20x 5.8** (the trained carry collapses the state's rank). Init probe: every
arm's four starts agree within 0.0002 at T = 8 (TRUE ×3; the E19 property survives every
factor). dense-core's start-independent fixed point reads 3.862 on the 96 rows against the
base's 3.896.

**P20f (cost):** dense-core 0.88 h (within 1.1× of the loop arm's 1.00 h: TRUE; the
ternary STE costs the difference), carry-lr20x 0.88 h, draw8-bptt4 0.58 h (the truncated
backward; predicted 1.15–1.5×: FALSE, the other way). Peaks 10.17 / 10.17 / 9.80 GB (< 16:
TRUE). Matched wall clock for draw8: at 0.58 h the parcae-entry arm sits near step 2,900
(held-out 4.30 at 3,000) against draw8's final 4.084, so per hour the truncated schedule is
ahead early; at matched steps it is 0.092 behind the base and 0.088 behind the depth-1 model, at this horizon.

## Verdict

**failure** (H20 refuted; H20′ half-confirmed; the finding is decisive). Parcae's depth
schedule is not the lever: mean 8 with backprop through the last 4 reads 0.09 nats
behind at matched steps (a 5,000-step reading, see the caveat) and, the finding, earns 0.002
past pass 3. A
carry that trains (B diagonal 1.74) is not the lever either: it collapses the carrier to
rank 6, costs 0.037 nats and still earns 0.007 past pass 3. The ternary core limits the LOOP'S CONTRIBUTION at this
horizon: with bf16 core weights and nothing else changed the loop's value past pass 1
goes 0.033 → 0.168, past pass 3 goes 0.0009 → 0.012, and the MLP branches move the state
four times more per pass (the 0.030-nat CE gap to the ternary base is a 5,000-step
reading and says nothing about ternary at a real horizon; ternary is the recipe). The prereg's 0.02 bar on K3−K6 is not cleared, so the
ternary map is a limiter and not the whole answer: even the bf16 core converges by pass 6.

Binding applied: dense-core moved K1−K6 by 5× but K3−K6 only to 0.012 ⇒ the "ternary map
is the limit" clause fires on its mechanism (the MLP branch ratio) if not on its number.
Ternary stays (Wolfe's rule, no dense warmup); the next work is core precision UNDER
ternary, written as an Agent Note before any run
(`.agents/notes/proposed/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md`).
The schedule lane and the carry-rate lane close. H20″ (all three FALSE on K3) is what
happened, so the readout probe stays on the list behind the ternary lever.

## Updated hypothesis

The loop's per-pass map is too weak to refine the state past pass 2, and the ternary
quantization of the core's MLPs is the largest single cause found so far (a 4× smaller
branch output per pass at the same architecture). The remaining gap between the bf16 core
(K3−K6 0.012) and Parcae (0.040 at K3−K8) is unexplained; candidates are the per-pass
block count and stream count (6 HC-Cayley blocks × 4 streams against 2 plain blocks), the
optimizer (Muon at 8e-3 against AdEMAMix at 1e-4; the carry-rate arm says a faster carry
alone is harmful), and tokens (31M against 246M). Under ternary the lever is the ternary
output magnitude of the core MLPs: the symmetric scale γ = mean|W| with threshold 0.5
shrinks each ternary layer's output against its bf16 twin, and a gate-up-down MLP
compounds three of them. The test is a ternary core whose per-pass MLP branch reads
0.7–0.9 of its input like the bf16 one (a learnable or norm-matching scale on the core's
MLPs, `ternary_scale_mode: ttq` being the existing knob), scored on the same three
instruments. The density panel (`2026-09-09-arc-density-panel.md`, running) tells whether
the production prune moves the same map the same way.
