# Planned: the loop-contribution arc (think once per span, decode cheaply)

Status: planned
Date: 2026-09-04 (frozen; no arm launched; GPU time is Wolfe's call)

This is the master file of an arc. Each experiment below has its own prereg in this
folder (`2026-09-04-arc-e*.md`) with its own frozen predictions. This file holds the
question, the order, the gates between experiments and the arc-level decision rule. It
is the successor to the round-1 panel
(`failures/2026-09-03-tul-think-once-panel.md`) and the forward-lever pair
(`successes/2026-09-04-tul-forward-levers.md`); the owning design note is
`.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md`.

## Question

The slot loop is now STABLE (the gain constraint holds the map's typical gain at 0.887
to 0.897 for 5000 steps with zero spikes) and EMPTY (forecast K3−K6 = 0.000, token loss
flat to 0.001 over slot depth, plan worth 0.07). Under what change does the slot loop
earn on trained support, and do the tokens read it?

## What is measured, stated once (every prereg cites these)

| fact | number | record |
|---|---|---|
| F1 every stable loop finishes by iteration 3 | step ratio at iteration 3 / 7: 0.45 / 0.33 to 0.54 / 0.47 (Y2), 0.27 / 0.24 (Y1); typical gain pinned 0.894 | forward-levers |
| F2 the memory target earns more depth than the forecast | own-loss K1−K6 0.037 (M-own) vs forecast 0.012–0.014 (M-next, Y1, Y2); both K3−K6 ≤ 0.001 | panel part 2, forward-levers |
| F3 the reader ignores the slot | token K1−K6 ≤ 0.0006 on every slot arm; the oracle-planted TRUE target moved token CE by 0.0000 on a detached-z arm | panel part 2, `successes/2026-08-28-oracle-prefix-probe.md` |
| F4 the plain token loop earns little at this scale | 0.037 at 5k and 0.04 at 20k under the ramp; 0.12 (K3−K6 0.015) on the flat schedule | panel part 2, `failures/2026-09-02-warmup-20k-pair.md`, `successes/2026-08-31-loop-killer-bisect.md` |
| F5 the cheap-decode floor | A3 480-row CE 4.0208 at 5k; every slot arm 0.13–0.20 behind it at 2.6–3.4x the wall clock | panel part 2 |
| F6 cost | a constrained M-next draw is 46 min per 5000 steps (110 steps/min); A3 is 13 min | forward-levers, panel |

## The hierarchy this arc walks

```
GOAL  the slot loop earns on trained support AND the tokens read it, at matched wall clock vs A3
│
├── (a) MAP REGIME   does the loop stop because the map converges?
│     F1: the trajectory contracts at 0.3–0.5 per iteration while the typical gain sits at 0.89.
│     E1  gain-target dial 0.95 / 0.98 (0.90 = Y2, on disk)        [2 draws, 92 min]
│     E2  iteration conditioning: core_stage_cond=iter on Y2       [1 draw, 46 min]
│
├── (b) TARGET       is there compute-limited work for the loop?
│     F2: memory (a multiplex of 32 tokens) earns 3x the forecast; both are done by iteration 3.
│     E0  where depth earns, on the kept checkpoints (per-row, per-offset)  [eval only, ~15 min]
│     E3  staged targets: iterations 1–2 supervised as MEMORY (own), 3–T as FORECAST (next)
│         [code first, then its own prereg; 2 draws, ~100 min]
│
├── (c) READER       will the tokens ever read the slot?
│     F3: 0.0000 from a planted true target when z is detached; unmeasured with the write live.
│     E4  the TG restriction under the constraint (R5 rerun)      [1 draw, ~50 min eager]
│
└── (d) SCORE
      E5  20k of any arm that THINKS, against A3-20k and notul-20k-wu on 480 rows,
          at matched steps AND matched wall clock                 [~4 h]
```

Reading the tree: (a) and (b) are the two ways a loop can be empty, and they are
separable by instrument. (a) shows as a step-ratio profile that falls monotonically with
the iteration index whatever the target. (b) shows as a loss that is flat over depth
even when the step ratio is large. The forward-levers record shows BOTH: ratio 0.5 at
iteration 3 (the state still moves) and K3−K6 0.000 (the loss does not). So the state
moves in directions the readout does not use, which is why (c) is the third branch and
not a footnote.

## Order and gates

1. **E0** runs first and has no gate: it is eval-only on the kept checkpoints and it
   shapes E3's target. It cannot block E1/E2.
2. **E1 and E2** run next, in that order, one trainer at a time. Both are one-line config
   changes on Y2 (`tul_to_mnext_y2_g95`, `_g98`, `_iter`).
3. **E3** needs code (intermediate slot states returned from `_tul_core`; a per-iteration
   `mux_local` with a per-iteration target; the depth sweep reporting both targets by
   forced depth). Its prereg is written AFTER the code passes its CPU tests and BEFORE its
   launch; it is not frozen here. E0's reading picks the target split.
4. **E4** runs on the arm with the best forecast K3−K6 among Y2, E1, E2, E3. If none
   THINKS (bar below), E4 still runs once on Y2: it is the only measurement of (c) with
   the write live, and it is cheap.
5. **E5** runs only on an arm that THINKS.

THINK bar (unchanged from the panel): forecast `mux_local` K3−K6 on the same 480 rows
> 0.01 with the paired-bootstrap CI above 0, at step 5000. PAYS bar: token CE beats A3
at matched wall clock (queue-log epochs), on 480 rows.

## Arc-level decision rule (binding)

- Any of E1/E2/E3 THINKS ⇒ E4 on it, then E5. The arc's product is then a 20k
  matched-wall-clock verdict on think-once, not a 5k one (deep models converge slower;
  `failures/2026-09-02-warmup-20k-pair.md`).
- None THINKS ⇒ the conclusion is written as: a weight-shared 6-block core does its work
  in three iterations on every target we can pose at this width on OpenWebText, whether
  or not the map is held near the edge and whether or not the iterations are told apart.
  The next arc is then one of two, Wolfe's call, and neither is a lever on this loop:
  (i) data where depth is known to pay (a code / math mix; the tokenizer is StarCoder2's
  already, the loader is OpenWebText only), or (ii) the deep slot stack without weight
  sharing (`tul_to_cond4` lineage, which detonated under the panel and has never run
  under the constraint).
- E4 answers (c) on its own: if the masked arm's token K1−K6 at the slot stays under
  0.01 even when the slot is the only route, the reader branch is closed at this scale
  and (c) is not the binding condition.

## Not in this arc, and why

- A per-group warmup (flat LR on the core from step 0, ramp elsewhere). The Gemini
  report's third recommendation. The flat schedule on the core is the detonation recipe
  (17/17); the constraint has never been tested against it. Parked until (a) and (b) are
  read; it is a stability experiment, not a contribution one.
- Per-iteration LoRA on the ternary core (Relaxed Recursive Transformers). E2's
  iteration conditioning is the same symmetry break at ~0 parameters and is already in
  the tree; LoRA is the follow-up if E2 moves K3−K6 but not enough.
- A richer slot seed. The boundary seed is already on every arm here; the rank reading
  belongs to E5's instruments, not to a 5k arm.

## GPU budget

E0 ~15 min eval; E1 92 min; E2 46 min; E3 ~100 min after ~1 day of code; E4 ~50 min;
E5 ~4 h. About 9.5 GPU-hours end to end, one trainer at a time, none launched without
Wolfe's call.

## Progress (the task list; status only, predictions untouched)

| # | item | status |
|---|---|---|
| E0 | where depth earns | FILED `failures/2026-09-04-arc-e0-where-depth-earns.md`: earning is FLAT over offsets and rows and smallest where the loss is largest (P0b, P0d false) |
| E1-95 / E1-98 | gain dial | FILED `failures/2026-09-04-arc-e1-gain-target-dial.md`: the hinge is a stability dial, not an earning dial; stays at 0.90 |
| E2 | iteration conditioning | FILED `failures/2026-09-04-arc-e2-iteration-conditioning.md`: held stable it earns LESS than Y2 (K1−K6 0.0087, K3−K6 +0.0001); the +0.0077 at 2500 was the expansive iteration. Branch (a) CLOSED. |
| E3 | staged targets (`tul.mux_stage_own_iters`; configs `tul_to_mnext_y2_stage2/3`; prereg `arc-e3-staged-targets.md`) | code done, 823 tests pass; E3-2 queued behind E2 in the GPU window |
| E4 | `to-mnext-y2-mask` on Y2 (mean 6, the constraint) | FILED `successes/2026-09-04-arc-e4-mask-under-constraint.md` (5 of 5): healthy to 5000; token K1−K6 +0.0209 (the tokens read the slot's depth on a stable map), shuffle worth 0.31, forecast K1−K6 +0.187; K3−K6 still ~0; +0.13 nats vs Y2 at 5000. Binding ⇒ E5 on E4 (Wolfe's call, not queued). |
| E5 | 20k matched wall clock on any THINK arm | not queued: no arm THINKS; Wolfe 2026-09-07: no 20k yet |
| E12 | the k=12 panel at a FIXED slot depth of 12 (`tul.slot_depth_fixed`) | REJECTED (wrong method: Wolfe asked for a Poisson draw of mean 12); the one arm that ran, `k12-mnext-mask`, detonated at 1446 in E7's mode (first-iteration state scale x170 by step 1100, hinge reading 1.0). `failures/2026-09-07-arc-e12-k12-panel.md`. The knob stays as a readout instrument. |
| E13 | the mean-12 panel: four TUL variants at a per-slot Poisson draw of mean 12, max 16, full BPTT, the fixed-point term on the slot loop (configs `tul_m12_*.yaml`) | FILED `failures/2026-09-07-arc-e13-m12-panel.md`: mask/mnext/a1 HEALTHY to 5000 (112/73/69 min), mtp4 DETONATED 2138 in the E7 scale mode (t0 ratio 172 under a hinge reading 0.95). The draw at mean 12 changes what the loop carries, not how deep it works: mask tokens K1−K6 0.021 → 0.049, forecast 0.187 → 0.579, K3−K6 tokens +0.0018 [+0.0016, +0.0020], same end point as E4 (mask@6 − E4@6 -0.0008 [-0.0031, +0.0015]). mnext free ride (K1−K12 +0.0010 [+0.0008, +0.0012]); a1 K1−K6 exactly 0 and the best token CE (4.141). |
| E14 | the expansive dial: E13's mask arm with the gain hinge at 1.02, with and without `slot_state_renorm` (configs `tul_m12_mask_g102*.yaml`) | FILED `failures/2026-09-07-arc-e14-expansive-dial.md`: both DETONATED (3877 single spike; 4639 spike train) in the backward-product mode with the state scale flat (renorm pinned t0 at 1.00 and delayed it 400 steps); the map sat at typical gain 0.99 and STILL settled by its last iteration (delta 0.03); K3−K6 unchanged (0.0007). The effect: tokens K1−K6 at 2500 0.033 → 0.043 → 0.060 (0.90 / 1.02 / 1.02+renorm) at the same end point 4.657. |
| E15 | the Olympiad panel at 15k steps on a uniform mix of the two math shards (configs `tul_oly_*`, `notul_oly`, `oly_data.yaml`; sweep `lab/divergence/olympiad_sweep.py`) | REJECTED before its first checkpoint (Wolfe 2026-09-07 22:30: too slow at 0.43 steps/s; 6k steps, the data's own curriculum from stage 2.1, optimize first). `failures/2026-09-07-arc-e15-olympiad-panel.md`. Direction note: `.agents/notes/proposed/process/2026-09-07-olympiad-math-as-the-loop-testbed.md`. |
| E16 | the Olympiad curriculum panel: mask, notul, mnext, a1 on synthetic math, 6k steps in four 1500-step band stages (2–5 → 11–13, ~15 % replay), effective batch 24, the held-out shard as val, the audit's levers (`ckpt_grad_iters 4`; `tg_scoped_kernels` on the mask arm) (configs `tul_oly_*`, `notul_oly`, `oly_data.yaml`; runner `arc/run_e16.sh`) | FAILURE `failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md` (ran 00:19–08:23 on 2026-09-08 at `effa7b0`, 4/4 survived; clean re-sweep 08:24–08:50). Mask tokens K1−K6 0.406 on clean math and 0.18 nats BEHIND the plain model paired at depth 12; answer K3−K6 −0.030 (worse past 3); mnext and a1 depth-flat, a1 ties plain ±0.010. P16c/P16d FALSE ⇒ Binding: stop the slot-loop lane (Wolfe confirms). Replay 15 % too thin (all arms −3 to −7 points on bands 2–3). Data: the Olympiad holdout is 41 % contaminated and bands 6–13 are 23–47 % unique (`.agents/notes/proposed/bug-fix/2026-09-08-olympiad-holdout-contamination.md`). Pit stop: `lab/perf/2026-09-08-oly-throughput-audit/`. |
| E17 | the Sudoku-Extreme depth grid: does the loop's accuracy on hard-rated boards rise with iterations T? Data = HRM's `sapientinc/sudoku-extreme` (1,000 train puzzles × HRM augmentation, test CSV as the held-out shard + sweep jsonl, the `rating` column as the depth axis; a board is `puzzle rows \n <\|A\|> solution rows <\|/A\|>`, one row per line so the boundary rule gives 9 slots per grid). Arms mask + notul first, mnext/a1 after. Readout = the grid solve-rate[rating bucket][T] from the held-out sweep at every checkpoint, paired over boards. Build: `scripts/sudoku_shards.py`, `morph/configs/sudoku_data.yaml` + `tul_sud_*` / `notul_sud`, tests, a 12-step smoke per arm | **SUCCESS (H17 held, H17′ refuted)** `successes/2026-09-08-arc-e17-sudoku-depth-grid.md` (scored 2026-09-08 19:25). Plain arm learns to 0.938 answer-token accuracy (mask 0.787, 0.155 behind paired) and every bucket is FLAT from T = 2 to 16 on both arms: hard-bucket acc@12 − acc@3 = −0.0000 [−0.0006, +0.0005] (notul), −0.0001 (mask); the diagonal is 0; whole-board solves 1 %. Method amendment 1: the sweep's plain rows advanced by seq_len not seq_len+1 (fixed `53b2472`, re-swept). Built by a sonnet subagent, vlt thread `e17-sudoku-build`. Data shipped: 1,000 puzzles × 1,000 augmentations = 1,000,000 boards, held-out 3,000 test boards (600 per rating bucket, 0 leaked), one 183-token board per packed row at seq_len 182; `tul.max_slots 0` (the inherited 64 OOMed the mask arm). Smokes: all four arms exit 0, peaks 15.6–20.1 GB. Runner `arc/run_e17.sh` (mask + notul first). |
| E18 | the slot-channel WIDTH sweep: the mask arm (tg_restrict, verified by the severed/intact gradient probes 2026-09-08) at `prefix_k` 2 / 4 / 8 vs the plain control, E13's recipe (seq 1024, batch 6, 5k steps, web text), scored on depth contribution AND CE at matched inference cost (block-passes per row) | RUNNING since 2026-09-08 19:25 `planned/2026-09-08-arc-e18-slot-width-sweep.md` (frozen 2026-09-08; launched by `arc/run_e17_resweep_then_e18.sh` after the E17 re-sweep; k2 smoke exit 0 at 11.06 GB). Configs `tul_m12_mask_k{2,4,8}`, `notul_e18`. Wolfe: the "bottleneck" I proposed IS this arm; "we build it fresh and test again ... We could also try increasing the width" → the probes passed, so the width sweep runs on the verified arm. |
| E19 | Parcae's loop entry on the PLAIN model (Wolfe 2026-09-09: "We misimplemented the looping in MORPH. It has never been contributing."): state from noise (`core_state_init: noise`), e re-injected on every dim through a learned B (`injection_channels: all`, `injection_B`), no fixed-point term; arms loop / fp0 (one-factor control) / d1 (trained at depth 1, prices the loop); E18 recipe, 5k steps; sweeps with per-token files + `core_anatomy.py` + `core_init_probe.py` | PLANNED `planned/2026-09-09-arc-e19-parcae-loop-entry.md` (frozen 2026-09-09; smokes exit 0 at 9.07 / 7.95 GB; launch is Wolfe's call). Probes behind it: T=0 costs 0.317, iterations 2-8 earn 0.020, state movement 2 %/iter, branches 2-13 %; Parcae's blocks move 40 %/pass, its state starts at 4 % of scale, its B diag 2.4. |
| E20 | Three loop-depth candidates on the E19 loop arm, one factor each (Wolfe 2026-09-09: "yeah this should be good ... Use descriptive names"): `e20-dense-core` (ternary everywhere but the core, diagnostic), `e20-carry-lr20x` (the injection parameters at 20x LR in their own group), `e20-draw8-bptt4` (Parcae's schedule: Poisson mean 8 / max 12, backprop through the last 4). Scored on the K-curve, token-paired CE vs `e19-loop`@6, and the depth-1 price vs `e19-d1`@1 | PLANNED `planned/2026-09-09-arc-e20-loop-depth-candidates.md` (frozen 2026-09-09; chained behind E19 COMPLETE in the queue). Behind it: E19 cleared the entry (K3-K6 0.0009 with Parcae-shaped state movement, start-independent fixed point, B/decay/dt untrained). |
| H | Huginn-3.5B loop contribution (eval only, the external ruler) | FILED `failures/2026-09-04-huginn-loop-contribution.md`: H2 TRUE (K3−K6 +0.566, K6−K16 +0.206, saturates at 16–24 against a training mean of 32); H1/H3/H4a/H5 false. Decision rule fired: option (i) downgraded, training regime is the candidate cause. |
| E6 | plain loop at a deep recurrence draw (mean 16, max 24, bptt 8; `notul_deep16.yaml`; prereg `arc-e6-deep-recurrence-draw.md`) | FILED `successes/2026-09-07-arc-e6-deep-recurrence-draw.md` (5 of 6): the draw is the lever on K-diffs (K3−K6 0.277, K6−K12 0.040, saturates at the mean 16–24) AND the deep model is 0.104 nats WORSE than mean-6 on the same rows at 1.23x cost: K-diffs measure depth DEPENDENCE, not depth value. E5 at the deep draw is the matched-compute test. |
| E10c | the directional gain penalty with the power iteration inside the step (`core_gain_power_iters 2`, STARS' λ·g², `notul_pgain2_wu0.yaml`), 6 draws under the assay; E10b as run was inert (penalty never fired) and its draws count as warmup-0 controls | DONE 2026-09-07 10:30, filed inside E10: 2 of 6 detonated (240, 315), P10b FALSE; the within-step power reading leads the tripwire by 20–40 steps (1.1–1.3 healthy → 3–8 at the onset) where `core_block_gain` is blind, but λ 1e-3 · g² is 0.008–0.036 at the onset and does not act |
| E11 | the fixed-point term at 5000 ramped steps vs notul (`notul_fp.yaml`; prereg `arc-e11-fixed-point-ramped.md`): the price and the earning of the E10a hold | FILED `successes/2026-09-07-arc-e11-fixed-point-ramped.md` (3 of 5, P11d held): FREE at the trained depth (+0.0009 [−0.0022, +0.0040] vs notul), healthy (max 62), depth-1 CE 0.014 better, K1−K6 0.022 vs 0.037. Proposal note to ship it beside the ramp (Wolfe's call). |
| E10 | two loop loss terms on the plain loop: terminal fixed point (`core_fixed_point_lambda`, `notul_fp_wu0.yaml`) and the directional power-iterated gain hinge (`core_gain_lambda`, `notul_pgain_wu0.yaml`); prereg `arc-e10-loop-loss-terms.md`; 6 + 6 draws under the assay, controls from E9 | FILED `successes/2026-09-07-arc-e10-loop-loss-terms.md` (2026-09-07 10:30): E10a fixed point 0 of 6 detonated vs 4 of 7 controls (Fisher p = 0.049), settles below 0.05 by step 20, survivors 0.41 nats BETTER at 800 → followed up as E11; E10b inert (penalty never fired, its 6 draws are the controls); E10c see its row |
| E9 | widen the diagonal carry to all 768 dims (`model.injection_channels: all`; `notul_carry_all.yaml` + control `notul_carry_ctx_wu0.yaml`; prereg `arc-e9-widen-the-carry.md`) under the warmup-0 detonation assay, 6 + 3 draws | FILED `failures/2026-09-07-arc-e9-widen-the-carry.md`: 2 of 2 widened draws detonated at 419 and 200 (control at 201); cut after P9a was decided. The early detonation is not a carry-spectrum event. |
| E8 | parallel multi-token prediction on the coda (`model.mtp_heads 4`; `notul_mtp4.yaml`, `notul_deep16_mtp4.yaml`; prereg `arc-e8-multi-token-coda.md`) | FILED `failures/2026-09-07-arc-e8-multi-token-coda.md`: E8-6 K3−K6 +0.0021, heads use depth LESS than the next-token head (0.32x at t+4), +0.345 nats worse than notul at matched steps; E8-16 detonated at 3446 (E6 did not), 0.26 behind E6 at 2500. Branch (b) closed. |
| E7 | the block-loop: E4 mask at slot draw mean 16 / max 24, grad through 8 (`tul_to_mnext_y2_mask_d16.yaml`; prereg `arc-e7-block-loop.md`) | FILED `failures/2026-09-07-arc-e7-block-loop.md`: DETONATED 2712; the hinge lost (gain 1.04 from step 1000), state norm 1.5e3 → 1.7e8; the Jacobian shows a 3e4x first-iteration jump, σ_max 95 vs rms gain 0.85, rank 181 → 9; the slot loop saturates at 3 even at mean 16; pre-onset 2500 clears both bars (not citable). Two verified defects: the global no-grad prefix silences 27–57 % of samples (bug-fix note), and the diagonal carry covers 256 of 768 dims. |

### Method amendment 1 (2026-09-04 17:08; order only, no prediction touched)

E2's first draw detonated at 2556, but its pre-onset 2500 checkpoint is the first slot
arm to move the bar: forecast `mux_local` K1−K6 +0.0726 [+0.0688, +0.0765] and K3−K6
+0.0077 [+0.0068, +0.0087] on 480 rows (Y2 at 5000: +0.0135 / +0.0002; g95: +0.0138 /
+0.0014). Token K1−K6 +0.0019 (every other slot arm ≤ 0.0006). The symmetry half of (a)
is the one lever that has moved earning, and its 5000-step reading needs the
every-iteration hinge (E2 Amendment 1). So the window's remaining draw is the E2 rerun
(`to-mnext-y2-iter-all`), and E3-2 moves to the next window. E3's prereg and predictions
are untouched.

### Method amendment 2 (2026-09-04 18:50; branch (a) closed, order for the next window)

E1 (`failures/2026-09-04-arc-e1-gain-target-dial.md`) and E2
(`failures/2026-09-04-arc-e2-iteration-conditioning.md`) are filed: neither the map's
contraction rate nor the weight-sharing symmetry makes a STABLE slot loop earn past
iteration 3; every earning past iteration 3 this campaign has seen came with an expansive
map and left with it. E0 (`failures/2026-09-04-arc-e0-where-depth-earns.md`) says the
loop's earning is a uniform refinement of the predictable part of the loss, never
concentrated where the target is hard. The next window runs E3-2 (staged targets) then
E4 (the mask on Y2), `arc/run_next_window.sh`, with the sustained tripwire
(`lab/divergence/tripwire_sustained.py`) on both. Predictions untouched. If E3 does not
THINK, the arc's closing rule applies as written.

### Method amendment 3 (2026-09-07; the Huginn rule fired, order for the next window)

The Huginn sweep (`failures/2026-09-04-huginn-loop-contribution.md`, run 2026-09-05,
scored 2026-09-07) answers the arc's closing option (i) before E3/E4 ran: web text is NOT
depth-flat. A 3.5B recurrent-depth model trained at a mean of 32 iterations earns
0.566 nats between iterations 3 and 6 and 0.206 between 6 and 16 on the same rows and
instruments where every stable MORPH loop earns ≤ 0.002 past 3, and it stops earning
between 16 and 24, near half its training mean, as MORPH stops at 3, half of its mean 6.
Its earning also has the shape E0 could not find on MORPH: it rises with offset-in-span
(1.62x from the span's first token to offsets 16–31) and with row difficulty (Spearman
+0.30). Per the prereg's binding rule, option (i) "data where depth pays" is downgraded
and the training regime becomes the candidate cause. E6 (`planned/2026-09-07-arc-e6-deep-
recurrence-draw.md`) is the test and takes the next window's first slot; E3-2 and E4 keep
their preregs and follow if the window allows. Predictions untouched everywhere.

### Method amendment 4 (2026-09-07 11:40; Wolfe's redirection, order for this window)

E8–E11 ran on the plain loop and produced the second hold (the fixed-point term, shipped in
`base.yaml`); no slot arm ran with it. Wolfe: the "done by iteration 3 at mean 6" reading is
the normal half-the-mean shape of a looped model and not a problem; the run to do is a
FIXED slot depth of 12, on the actual TUL variants. E12 (`planned/2026-09-07-arc-e12-k12-
panel.md`) takes the window: the fixed-point term ported to `_tul_core`, `tul.slot_depth_
fixed`, and the E8 heads wired onto the TUL coda, four arms. E5 is not queued (no 20k yet).
E3 keeps its prereg and follows E12's binding rule. Predictions untouched everywhere.

### Method amendment 5 (2026-09-07 12:05; E12 rejected, E13 replaces it)

Amendment 4 misread Wolfe's "a fixed run at k of 12" as a fixed DEPTH; he meant one run
per variant at a Poisson slot draw of mean 12. E12 was killed after its first arm (the
mask arm detonated at 1446 in E7's power-iteration mode) and is filed as a rejected run.
E13 (`planned/2026-09-07-arc-e13-m12-panel.md`) runs the same four variants at
`tul.slot_mean_depth 12`, `slot_max_depth 16`, `bptt_depth 16`. E3 follows E13's binding
rule. Predictions untouched everywhere.

### Method amendment 6 (2026-09-07 22:30; the testbed moves to Olympiad math)

Wolfe: looped models earn on thinking problems; every number so far is web text. The arc's
readout moves to Olympiad-AI's synthetic math (one reasoning step per span, an answer
block; `docs/olympiad-interop.md`), scored by `lab/divergence/olympiad_sweep.py` on held-out
documents: token CE, answer-region CE and accuracy, per-stage-band curves, paired over
documents. Runs are 6k steps, walk the data's curriculum from stage 2.1 (per-band shard
views + per-stage blends in the curriculum loader), and follow a throughput audit. E15 was
rejected before data; E16 replaces it. Predictions untouched everywhere.

### Method amendment 7 (2026-09-08 00:30; the pit stop, and what E16 carries from it)

The throughput audit (`lab/perf/2026-09-08-oly-throughput-audit/`) measured the mask arm at
2143 ms/step eager and 1055 ms with `model.tg_scoped_kernels true` (1.63×, −5.4 GB; the
fused kernels everywhere but the TG-restricted branches) plus `model.ckpt_grad_iters 4`
(exact, 1.25×, +6.3 GB). Micro 24 × 1 OOMs on every arm; micro 12 × 2 stays. Prefetch depth
is a null lever (the data wait is the host blocking on the GPU queue; the step is GPU-bound).
Two findings change the record: `perf/flop_proxy` divided by the effective batch under
grad accumulation (fixed in this change; E15's launch line "proxy=16.33" was inflated),
and the gain hinge on the eager path reads the map 0.07 high at `slot_gain_eps 0.02`
(noise bias; both paths converge on 0.870), so E4/E13-mask/E14 paid a hinge on noise. E16
runs the mask arm fused, which makes the panel paired on kernels and on the hinge reading
(P16i is the first reading of the hinge on a fused TG arm). Validation moves from
OpenWebText to the Olympiad held-out shard (`curriculum.val_source`, rewound before every
eval). Predictions untouched everywhere.

### Method amendment 8 (2026-09-08 09:00; E16 scored, the arc's slot-loop lane closes)

E16 ran in full (`failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md`). Its readout was
re-done after the fact on a decontaminated holdout (amendment 8 there); predictions untouched.
The Binding fires on P16c FALSE: the slot loop does not help solve math either. The arc's
remaining lanes are the write-back (Spiral schedule, the paid loop) and staged targets (E3),
on deduped Olympiad data with a matched-compute shallow control and 30 % replay. No Olympiad
run before the band views are deduped.
