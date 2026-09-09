# Planned: ARC E20 — three loop-depth candidates on the E19 loop arm: dense core, a carry that can train, Parcae's depth schedule

Status: planned
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

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
