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
| E0 | where depth earns (eval only; `--profile` on both depth sweeps, `score_arc_e0.py`) | code done, queued behind arc-a2 (`arc/run_e0.sh`) |
| E1-95 | `to-mnext-y2-g95` | DRAWN 2026-09-04 (healthy 4999; sweep + worth on disk; not a THINK: K3−K6 +0.0014) |
| E1-98 | `to-mnext-y2-g98` | DRAWN (reached 4999, AMBIGUOUS 1e3 excursion, 121 spike steps; not a THINK: K3−K6 +0.0005) |
| E2 | `to-mnext-y2-iter` | drawing (started 16:35, `arc/run_arc_a2.sh`) |
| E3 | staged targets (code: intermediate slot states, per-iteration `mux_local`, sweep columns), then its prereg | not started |
| E4 | `to-mnext-y2-mask`, on the best arm of E1/E2/E3 (Y2 if none THINKS) | waits on E1/E2 |
| E5 | 20k matched wall clock on any THINK arm | waits on E4 |
