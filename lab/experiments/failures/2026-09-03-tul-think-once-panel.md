# Failure: the think-once round-1 panel — does the slot loop earn on a job of its own?

Status: failure
Date: 2026-09-03 (frozen before any arm ran; only 12-step smokes, whose numbers are
not metrics, may precede launch). Branch `tul/think-once` (from `d9e04e6`), worktree
`/home/wolfe/morph-to`. Drawing board and the reading of the record:
`.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md`
(Revision 2026-09-03 evening).

## Question

TUL's purpose is "think once per span, decode cheaply": the core loops on the slot,
tokens run prelude → coda (8 layers) and read the thought. Every slot-loop arm in the
record earned ≤ 0.015 nats of depth, and every one of them ran under `warmup: 0` with at
least one of GLA, the spectral cap or the carry leak. This panel asks, on the winner
recipe (retention off, cap 0, carry none) plus the 1000-step ramp, at one shape (seq
1024, batch 6, seed 1, 5000 steps):

1. Does the ORIGINAL design (A1) survive under the ramp, and is its plan still empty?
   (Wolfe's hypothesis: the warmup may have been the whole problem.)
2. Can the slot loop earn depth on a job of its OWN — the MUX local loss (arXiv
   2607.18264) on a memory (`own`) or a forecast (`next`) target — measured without the
   coda in the way?
3. Where do four extra layers pay: once per span on the thought (R7) or on every token
   (R8)? And does the thought help when the coda reads it with stop-gradient (frozen z)?

## Arms (configs `morph/configs/tul_to_*.yaml`; every arm composes `tul_to_panel.yaml`)

| # | arm | config | one line |
|---|---|---|---|
| R0 | A3-wu | `tul_to_a3` | coreless, no slots: the cheap-decode FLOOR (8 layers/token) |
| R1 | A1-wu ×2 | `tul_to_a1` (seed 1; `training.seed=2 wandb.name=to-a1-s2`) | the original TUL, first time on the ramp |
| R3 | M-own | `tul_to_mown` | boundary seed, aux off, MUX own β 1 through the head, full-BPTT slot loop, no mask, fused kernels |
| R4 | M-next | `tul_to_mnext` | as R3 with the FORECAST target — the arm |
| R5 | M-next-mask | `tul_to_mnext_mask` | R4 + TG mask (eager) |
| R6 | M-own-mask (optional) | `tul_to_mown_mask` | tul-20k's design under the ramp |
| R7f | cond4 | `tul_to_cond4` | R4 + 4 non-shared slot layers after the loop; z = their output |
| R7d | cond4, frozen z | `tul_to_cond4_dz` | R7f with stop-gradient at the coda's read |
| R8f | coda8 | `tul_to_coda8` | R4 with an 8-layer coda (12 layers/token) |
| R8d | coda8, frozen z | `tul_to_coda8_dz` | R8f with stop-gradient at the coda's read |

Rulers on the SAME recipe and shape, already trained (kept under
`/home/wolfe/morph-scratch/checkpoints-keep/`): `notul-20k-wu/step_5000.pt` (looped
tokens, no slots; 48-row K1 3.9879 / K6 3.9488) and `tul-a2-20k-wu/step_5000.pt` (the
paid loop; K1 4.1380 / K6 4.0812). Both get the 480-row readout in this panel.

## Readouts (all on the first 480 validation rows, paired by row)

- `core_depth_sweep.py` depths 1..8, batch 3: `ce_tokens`, `ce_span_first` and — new
  — `mux_local` per forced depth, with paired bootstrap CIs for K1−K6, K3−K6, K1−K8
  (rows for CE, batches for mux). Token arms: `token_depth_sweep.py`.
- `worth_profile.py` (zero / shuffle / wrong_seed by offset-in-span, 192 rows).
- Val CE: the last-20 eval mean; never one eval (the 2026-09-02 pair lesson).
- Stability: the tripwire (`preclip/total > 1e4` at step ≥ 200) on every draw; the
  takeover rule (`score_arms.py::fires`, share > 0.5 on > 30% of the last 50 probed
  steps) on the A1 draws.
- Wall clock per arm from the queue log; `layer_passes_per_token` from the eval line.

## Predictions (frozen)

Floor and rulers:
- **P0a.** R0's 480-row CE at 5000 sits 0.15–0.45 nats ABOVE notul-wu's 480-row K6 at
  5000 (the token loop is worth that much at matched steps): 60%.
- **P0b.** At matched WALL CLOCK, R0 beats notul-wu (R0 finishes 5000 steps in less time
  than notul-wu needed for the steps that reach R0's CE): 70%.

The original design under the ramp (R1, two seeds):
- **P1a.** Both A1 draws finish 5000 steps with the tripwire silent AND the takeover
  rule not firing: 60%.
- **P1b.** Slot K1−K6 on 480 rows ≤ 0.02 on both seeds: 80%.
- **P1c.** Plan worth (zero-ablation, all tokens) ≤ 0.04 on both seeds: 75%.
- **P1d.** A1-wu's 480-row CE beats R0's by more than 0.05: 25%.

MUX targets (R3, R4):
- **P3.** R3's `mux_local(own)` K1−K6 has a CI that includes 0 or a point < 0.02: 70%.
- **P4a (the gate).** R4's `mux_local(next)` K1−K6 > 0.02 with the CI above 0: 45%.
- **P4b.** R4's `mux_local(next)` K3−K6 > 0.01: 35%.
- **P4c.** R4's shuffle profile at offset 0 ≥ 0.05 nats: 40%.
- **P4d.** R4's 480-row CE beats R0's: 60%; beats R3's by > 0.05: 25%.

The mask (R5):
- **P5a.** R5's 480-row CE is 0.08–0.35 nats worse than R4's: 70%.
- **P5b.** R5's shuffle profile at offset 0 ≥ 0.05: 65%.
- **P5c.** R5's `mux_local(next)` K1−K6 exceeds R4's: 50%.

Four layers, and frozen z (R7/R8):
- **P7a.** R8f's 480-row CE beats R7f's by > 0.03: 55%. R7f within 0.03 of R8f: 35%.
  R7f better: 10%.
- **P7b.** R7f's `mux_local(next)` at trained depth is lower than R4's: 65%.
- **P7c.** R7f's wall clock ≤ 1.10× R4's; R8f's ≥ 1.25× R4's: 70%.
- **P7d.** Each frozen-z arm's 480-row CE is > 0.05 worse than its full-gradient twin:
  65%.
- **P7e.** Each frozen-z arm's `mux_local(next)` at trained depth is LOWER than its
  twin's (the loop's whole gradient budget goes to the forecast): 55%.
- **P7f.** Each frozen-z arm's shuffle-profile at offset 0 is below its twin's: 60%.

Stability across the panel:
- **PS.** Tripwire silent on all 10 draws: 65%.

## Decision rule (binding)

An arm THINKS iff its `mux_local` K1−K6 CI sits above 0 AND K3−K6 > 0.01 (P4a/P4b for
R4; the same test applies to R3, R5, R7, R8). An arm PAYS iff its 480-row CE beats R0 at
matched wall clock. Round 2 is conditional: any arm that thinks goes to 20k against
`notul-20k-wu` and a 20k R0 (own prereg). No arm thinks ⇒ the slot loop cannot earn on
any target we have at this depth and scale; the program moves to loop composition itself
(the token loop saturates by K4 on every recipe) or to a deep slot stack without weight
sharing, and TUL-as-loop is parked with this file as the record. P1a TRUE with P1b/P1c
TRUE ⇒ the ramp fixes the takeover and leaves the plan empty — "the original design was
fine" is refuted on the contribution axis and confirmed on the stability axis; both are
reported.

## Method

Runner `/home/wolfe/morph-scratch/to/run_think_once.sh` (sequential, one trainer, GPU
refusal if any compute process > 2 GiB, tripwire watcher per draw): smokes first (12
steps, eval at 5/10, checkpoint at 10 — memory, NaN, the `TUL THINK-ONCE` banner where
expected, the eval instruments); then draws in the order R0, R1s1, R1s2, R3, R4, R7f,
R8f, R7d, R8d, R5, R6; then the readouts (sweeps at 480 rows batch 3, worth profiles,
the two rulers), never concurrent with a trainer (UPS). Checkpoints at 2500 and 5000
under `/home/wolfe/morph-to/checkpoints/morph/<name>`; pruned to step_5000 after the
verdict. Artifacts to `lab/experiments/results/2026-09-03-tul-think-once-panel/`.
Estimated: ~5.5 h of training, ~1.5 h of readouts.

### Method amendment, 2026-09-03 20:30 (during the run; Predictions above untouched)

Reason: `tul_to_a1` inherits `model.bptt_depth: 4` from `base.yaml` (the record's A1
setting), while every MUX arm trains its slot loop at full BPTT (8). Wolfe: "The A1
question is if the loop contributes now even if there are stability issues... bptt should
probably be 8 on A1." A loop trained through 4 of 8 iterations cannot be compared with the
MUX arms on K1−K6. Added after the panel, same recipe, no other change:

| # | arm | config | one line |
|---|---|---|---|
| R1b | A1-wu, full BPTT ×2 | `tul_to_a1_b8` (seed 1; `training.seed=2 wandb.name=to-a1-b8-s2`) | R1 with `model.bptt_depth: 8` |

Runner `/home/wolfe/morph-scratch/to/run_a1_b8.sh` (waits for `PANEL COMPLETE`, then
`run_think_once_v2.sh` = v1 + `SKIP_RULERS`). The R1 predictions P1a–P1d are scored on
R1 (bptt 4) as frozen. Stated here for R1b, with the same priors as R1 and dated as an
amendment, not a frozen prediction: tripwire silent on both seeds 60%; slot K1−K6 ≤ 0.02
on both 80%; plan worth ≤ 0.04 on both 75%; beats R0 by > 0.05 25%. Record also: to-a1-s2
(bptt 4, seed 2) DETONATED at step 2041 (onset ≈ 1850–1900, `preclip/core` 0.026 → 8.6 →
92 → 1.0e4 over steps 1800/1900/2000/2041), so P1a and PS are already FALSE.

## Not verified before launch

The conditioning stack and `detach_z` on GPU at the real shape (CPU tests only: 10 new
+ 183 existing TUL tests pass); torch.compile over a model with `tul_cond`; coda8's
activation memory at batch 6 (coda layers are not checkpointed); fused kernels on the
full-BPTT slot loop without the mask at this shape (tul-a1 ran fused at bptt 4; tul_l1
ran eager); the sweep's mux column on a real checkpoint (unit-tested arithmetic only).
The smokes gate the first four; the first sweep gates the last.

## Results (2026-09-04 02:20; runner stopped by Wolfe 2026-09-03 22:34 for the onset capture)

Ten draws were planned. Eight ran to a verdict, one (R5) was killed by the stop at
step 1639, one (R6) never started. NO contribution readout ran (no depth sweep, no worth
profile, no ruler): P0a–P0b, P1b–P1d, P3, P4a–P4d, P5a–P5c, P7a–P7f are UNMEASURED. The
panel produced ONE axis, stability, and on it the tripwire (`preclip/total > 1e4` at step
≥ 200) fired on six of eight finished draws. Per arm, from each draw's `probe.jsonl`
(`/home/wolfe/morph-scratch/to/<name>/`) and the queue log:

| arm | draw | verdict | first `preclip/total` > 1e3 | peak | last step | val CE (one eval) |
|---|---|---|---|---|---|---|
| R0 A3-wu | to-a3 | healthy | never (max 31 at 612) | 31 | 4999 | 4.0473 |
| R1 A1-wu s1 | to-a1-s1 | healthy | never (max 29 at 244) | 29 | 4999 | 4.1897 |
| R1 A1-wu s2 | to-a1-s2 | DETONATED 2041 | 1989 | 4.2e5 at 2092 | 2098 | — |
| R3 M-own | to-mown | healthy | never (max 91 at 244) | 91 | 4999 | 4.2631 |
| R4 M-next | to-mnext | DETONATED 3618 | 3347 | 3.4e4 at 3618 | 3627 | — |
| R5 M-next-mask | to-mnext-mask | killed by the stop | none by 1639 (max 233 at 337) | 233 | 1639 | — |
| R6 M-own-mask | — | never ran | | | | |
| R7f cond4 | to-cond4 | DETONATED 1009 | 984 | 2.2e4 at 1009 | 1077 | — |
| R7d cond4 frozen z | to-cond4-dz | DETONATED 1019 | 1000 | 3.4e5 at 1033 | 1064 | — |
| R8f coda8 | to-coda8 | DETONATED 1384 | 1350 | 1.0e4 at 1384 | 1443 | — |
| R8d coda8 frozen z | to-coda8-dz | DETONATED 1430 | 1381 | 2.5e6 at 1443 | 1443 | — |

Two shapes, both under the 1000-step ramp:

- **The A1 takeover shape** (to-a1-s2): a monotone core-gradient ramp with onset near
  1850–1900 (`preclip/core` 0.026 → 8.6 → 92 → 1.0e4 over steps 1800/1900/2000/2041),
  the same window as the warmup-0 takeovers of the 2026-08 campaign. One seed of two.
- **The forecast spike train** (every arm with the MUX-next target): the coda's gradient
  flat (0.55–0.75), the conditioning stack's gradient flat (0.1–0.6), and the core,
  prelude, slot modules and embedding spiking together on single steps that snap back and
  escalate once the LR plateau begins. Five of five forecast draws. The memory target
  (R3) on the same recipe ran clean, so the target, not the loop's presence, selects the
  face. Its mechanism was measured the same night: `2026-09-03-tul-onset-capture.md`.

Wall clock, from the queue log (START to DONE): R0 13.0 min, R1 s1 34.5 min, R3 44.6 min
per 5000 steps; R4 3627 steps in 33.0 min (110 steps/min, R3's pace, about 3.5x slower
per step than R0's 385 steps/min); R7f 1077 steps in 10.0 min, R8f 1443 in 14.5 min,
R7d 1064 in 10.0 min, R8d 1443 in 14.0 min (P7c's ratios are not scorable on partial
draws; the two cond4 draws ran at R4's pace, the two coda8 draws at about 0.9x of it).

## Verdict

- **P1a FALSE** (one A1 draw of two detonated). **PS FALSE** (six trips in eight finished
  draws). The ramp does not cure the A1 takeover and does not touch the forecast face.
- Every other prediction: **unmeasured**. The method was not completed: the runner was
  stopped at Wolfe's call ("we should do this before the queue finishes. It is worth
  stopping for this") once the stability axis had shown that five of five forecast draws
  die before any 5000-step readout could exist.
- What the method could not distinguish: whether the slot loop THINKS on the forecast
  target (P4a/P4b), because no forecast arm survived to a checkpoint that a depth sweep
  could read. to-mnext left a step-2500 checkpoint (healthy at that step, tripped at
  3618); the healthy arms left step 2500 and 5000. Those readouts are still runnable and
  are GPU time at Wolfe's call.

## Updated hypothesis

The forecast target is the target that makes the slot loop's operator gain climb; the
stability question has to be answered on R4 before any contribution question about R4
means anything. The capture record names the mechanism (a slow drift of the loop's
typical gain toward 1 and a backward product through eight iterations past it, with
ternary QAT as the necessary driver and no forward event) and the lever order. Round 2
of this panel is conditional on a forecast arm that survives 5000 steps under that lever.
