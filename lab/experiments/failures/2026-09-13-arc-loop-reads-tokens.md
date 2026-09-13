# Planned: does the core USE its depth when it is reading tokens?

Status: failure

Date: 2026-09-13 (frozen before the probe runs; the 5090 is running the math panel and
the probe is queued behind it). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Parent
prereg:
[`2026-09-12-arc-core-token-and-critic.md`](2026-09-12-arc-core-token-and-critic.md).
Design note:
[`2026-09-13-the-loop-reads-the-tokens.md`](../../../.agents/notes/proposed/architecture/2026-09-13-the-loop-reads-the-tokens.md).

**Scope, 2026-09-13.** This was drafted as a training arm. Wolfe cut the training arm and
kept the probe: `loop_reads_tokens` costs about 44 block-passes per generated token
against the slot loop's 10.59 (`docs/tul-as-memory.md`), which abandons "think once,
decode cheap" — the thing TUL is for. **The knob is built, tested and OFF by default; no
arm using it is queued.** What runs is a cheap EVAL probe on a checkpoint that already
exists.

## The fact

`slot-spandec-strict-coretok` (2026-09-12) gave the shared core the token objective in
TRAINING only: a second pass sends every position through `_core_region` under the strict
relation and charges the token CE. Measured at 5,000 steps on 480 rows
(`results/2026-09-12-strict/core_token_aux_probe_5000.json`):

| reading | coretok | strict |
| --- | --- | --- |
| shipped-path CE @6 | **4.2790** | 4.3474 |
| aux-path CE @6 | **4.2626** | 5.4520 |
| aux − shipped | **−0.0164 [−0.0176, −0.0151]** | +1.1046 [+1.0839, +1.1247] |

So the aux core reads the tokens BETTER than the shipped stack does, by 0.016 nats, at the
same weights — and the slot loop through those same six blocks still read K1−K6 0.0005.
The verdict filed then was "the core is not under-trained, it is under-USED."

**That verdict has a hole, and this probe closes it.** The aux path was only ever scored
at ONE depth. Nobody has asked whether the aux path — the core reading tokens — USES its
loop depth. If it does, the core is a depth-using map when it reads tokens and a flat one
when it reads slots, and the difference is the INPUT, not the map. If it does not, the
core is a flat map on both inputs and every slot-side lever was always beside the point.

## Question

Does the token-reading aux path of `slot-spandec-strict-coretok` earn depth — K1−K6 and
K3−K6 on the AUX CE — on the same 480 rows the shipped path is scored on?

## Method

`lab/divergence/core_token_aux_probe.py`, extended in this change with `--depths`. Eval
only, no dropout, no training step, one checkpoint, CPU-side pairing.

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python lab/divergence/core_token_aux_probe.py \
  --ckpt coretok=tul_slot_spandec_strict_coretok=/home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict-coretok/step_5000.pt \
  --depths 1,2,3,6 --rows 480 --batch 3 --device cuda \
  --out /home/wolfe/morph-scratch/arc/results/2026-09-13-aux-depth/core_token_aux_depth_5000.json
```

Run it FIRST when the GPU frees: it is one checkpoint, 480 rows, four depths, and it
changes which of the queued arms is worth running.

What the extension does, and what it protects:

* `--depths` forces the AUX core's depth through `model.cfg.mean_depth` while PINNING the
  slot loop's depth at `slot_mean_depth or mean_depth` first, so the shipped path is
  scored at its trained depth at every point of the sweep.
* It asserts the shipped CE **did not move** across the depth sweep. If the shipped
  number drifts, the two depths are entangled and the sweep is unreadable; it raises
  instead of reporting.
* It reports `ci_aux_ce` with a paired bootstrap over batches for K1−K6, K3−K6 and
  K1−Kmax, the same estimator the depth sweeps use.
* The old top-level keys are still written, so the 2026-09-12 numbers above remain
  directly comparable.

Gate: `tests/test_tul_loop_reads_tokens.py`, **38 cases**, green in
`tests/test_tul_slot_register.py tests/test_tul_setup_keys.py
tests/test_tul_prefix_source.py tests/test_tul_loop_reads_tokens.py -q` → **145 passed**.
It covers the knob's OFF state being byte-identical (`W_prefix` is the only key
difference; the `_core_token_aux_kwargs` builder is never reached), the span-restriction
leak test with three two-sided controls, a spy proving the SHIPPED forward uses that exact
relation, eval CE equal to the training forward's at a pinned depth, `uses_sample_depth`,
and that `mean_depth` is the lever while `slot_mean_depth` is not.

### Which relation the core stage runs under, stated because it was asked

When `loop_reads_tokens` is on, the core stage runs under the STRICT
`_core_token_aux_kwargs` relation: `causal AND (same span OR j is a slot cell)`. A token
reads its own span's tokens and reaches every EARLIER span ONLY through a slot cell, so
the cells remain the whole cross-span channel at inference as in training. **The coda's
reach is unchanged** — `tg_coda_prefix_reach` still governs it, exactly as on the partner.
This is NOT the paid loop, whose core is unrestricted and in which the cells carry nothing
anyone needs.

Verified fact from the 2026-09-13 audit, recorded here so it is not re-derived:
`tg_restrict`, `tg_geometry` and `tg_coda_prefix_reach` never touch the IN-LOOP relation —
`_tul_core` is handed no `tg_attn_kwargs`, and `tul.loop_reach` is the only in-loop key.

### What an independent review measured on the knob (2026-09-13)

Recorded here so the arm's NAME does not oversell its mechanism, and so a future reader
does not have to re-derive any of it.

* **It loops the TOKEN states.** The coda reads the CORE's output at token positions, not
  the prelude's. So `loop_reads_tokens` is **the paid loop with a span restriction**, not
  the slot loop: the "think once, decode cheap" property is gone and only the cross-span
  restriction survives. That is the real reason the training arm is not queued, and it is
  a stronger reason than the cost figure.
* **The leak test's scope is ONE core block.** It proves "no token reads another span's
  token" on an `n_core = 1` fixture. At `n_core = 2` the token -> cell -> token route is
  open BY DESIGN and reads 0.589 on the same fixture. That is the intended channel, not a
  leak — the cells are supposed to be the cross-span path — but the test does not and
  cannot say "no cross-span information reaches a token".
* **The slot-loop levers were silently inert on this mode.** `tul_slot_spandec_strict_
  tokloop.yaml` inherits `slot_gain_lambda: 100` and `slot_cot_clip: 4.0` from the
  slot-loop config root, and they act only inside `_tul_core`, which this mode never
  enters. The `[slot-levers] ... INERT` predicate named the paid loop and not this one.
  Fixed in the same change, with a three-way test twin (tokloop prints it, the paid loop
  prints it, the slot loop must NOT).

## Predictions (frozen)

Written before the probe runs.

- **Q-1 (the aux path earns depth).** `coretok` aux **K1−K6 above 0.05** nats. **45 %.**
  Reasoning: the plain looped model earns 0.185 nats under the noise entry and 0.033 under
  the prelude entry (2026-09-10), and this is the same core on the same token objective
  under a prelude entry — with a span restriction that removes most of the context, which
  cuts both ways (less to integrate, but more need to integrate it). Residual: 35 % in
  [0.005, 0.05], 20 % below 0.005.
- **Q-2 (the aux path's LATER passes).** `coretok` aux **K3−K6 above 0.01**. **25 %.**
  Reasoning: every depth reading in this tree concentrates in the first pass.
- **Q-3 (the split is the input, not the map).** `coretok` aux K1−K6 is **at least 10x**
  its shipped-path slot K1−K6 (0.0005). **60 %.** This is the prediction the probe exists
  for: it separates "the core cannot use depth" from "the core cannot use depth ON SLOT
  STATES".
- **Q-4 (the theory agent's, transcribed).** If the training arm were run, its token
  K1−K6 would be **at least 0.5x the matched plain control on the same loop entry** (0.017
  under the prelude entry), with slot **K3−K6 ≤ 0.002**. **Unscorable** — the arm is cut.
  Kept verbatim so the record shows what the scope call gave up.
- **Q-5 (the shipped path does not move).** The probe's assertion holds: shipped CE is
  identical to 1e-6 at all four depths. **90 %.** If this fails the sweep is void, not
  interesting.

## Binding

If **Q-1 and Q-3 hold** — the aux core earns depth on tokens while the slot loop through
the same weights reads zero — the map is exonerated for the third time and the fault is
proved to be the INPUT the loop iterates. That points straight at the Thought Register
(the slot state's rank) and away from every core-side lever, and it is the cheapest
evidence for the register that exists.

If **Q-1 fails** — the aux core is flat on tokens too — then this core does not use depth
on ANY input at 5,000 steps, the plain model's 0.185 nats comes from something the aux
path does not have (its own coda, its own unrestricted context, or simply a longer
horizon), and "the core is under-used" was the wrong verdict. The next measurement is then
the plain model's own K-curve under the SAME span restriction, which is a config away.

If **Q-5 fails** — depth-forcing moves the shipped path — the probe is wrong and both the
2026-09-12 aux numbers and this sweep need re-reading.

## Not verified before launch

* **The probe has not been run.** The GPU is held by the math panel. Everything above is
  the design plus the 2026-09-12 single-depth reading.
* **`loop_reads_tokens` has never run a training step.** It is built, tested, refused in
  every combination it does not compose with, and OFF. Its OFF state is proved
  byte-identical; its ON state has never seen a GPU.
* **The cost figure that killed the training arm is arithmetic.** 44 block-passes per
  token comes from the cost model in `docs/tul-as-memory.md`, not from a wall-clock
  measurement.
* **The probe scores the aux path of a checkpoint trained with the aux ON.** It says
  nothing about a core trained without it.
* **480 rows, one seed, one step (5,000).** No horizon reading.

## Results

Run 2026-09-13 on the DGX Spark at commit 9b6427e (the probe and its `--depths` extension
are in d99e9d9; nothing after that commit touches the probe or the model's eval path).
480 rows, batch 3, 160 paired units, 2000 bootstrap draws. Files:
[`results/2026-09-13-aux-depth/core_token_aux_depths_coretok_5000.json`](../results/2026-09-13-aux-depth/core_token_aux_depths_coretok_5000.json)
and the run log beside it (`.txt`).

| forced AUX depth | shipped CE | aux CE | aux − shipped [95 % CI] |
| --- | --- | --- | --- |
| 1 | 4.2790 | 4.2727 | −0.0063 [−0.0073, −0.0054] |
| 2 | 4.2790 | 4.2657 | −0.0134 [−0.0144, −0.0123] |
| 3 | 4.2790 | 4.2638 | −0.0152 [−0.0163, −0.0141] |
| 6 | 4.2790 | 4.2626 | −0.0165 [−0.0177, −0.0152] |

Aux-path K-curve, paired over the same 160 units:

| reading | point | 95 % CI |
| --- | --- | --- |
| aux K1−K6 | **+0.0102** | [+0.0093, +0.0110] |
| aux K3−K6 | **+0.0012** | [+0.0009, +0.0016] |

Where the depth is spent: passes 1→2 give 0.0071 of the 0.0102, 2→3 give 0.0019, 3→6
give 0.0012. The shipped path's CE is 4.279032 at every forced aux depth (the probe's
assertion held; the slot loop was pinned at its trained depth throughout).

## Verdict

- **Q-1 FALSE.** Aux K1−K6 is 0.0102, inside the 35 % residual band [0.005, 0.05], not
  above 0.05.
- **Q-2 FALSE.** Aux K3−K6 is 0.0012, below 0.01.
- **Q-3 TRUE.** 0.0102 is 20x the shipped slot loop's 0.0005 (the 10x bar), with a CI
  clear of the bar.
- **Q-4** unscorable (the training arm is cut), as filed.
- **Q-5 TRUE.** The shipped CE did not move across the sweep.

Status: failure (two of the four scorable predictions failed). The binding clause for
"Q-1 fails" said the core does not use depth on ANY input; that is too strong, because
Q-3 held: the SAME six blocks that read K1−K6 0.0005 on the slot states read 0.0102 on
the span-restricted token stream. The right statement sits between the two clauses.

## Updated hypothesis

The core uses depth on a token input, weakly, and almost all of it in the first two passes:
0.0102 nats, against 0.033 for the plain MORPH model under the same prelude entry
(2026-09-10) and 0.185 under the noise entry. The span restriction is NOT what removes the
rest. The control already exists on the Parcae core
([`failures/2026-09-11-arc-span-budget.md`](2026-09-11-arc-span-budget.md), sweeps at
step 5,000): `budget-web-full` reads token K1−K6 +0.0279 [+0.0269, +0.0290] and
`budget-web-span`, where nothing crosses a span boundary, reads +0.0301 [+0.0288, +0.0313].
Cutting every cross-span read leaves the plain loop's depth reading unchanged. So the
ordering on one set of weights is: token input 0.0102 > slot-state input 0.0005 (20x), and
the aux path's own shortfall against a plain model (0.010 vs 0.028 to 0.033) is not the
restriction. What differs is that this core is shared with the slot loop and trained on
both jobs, with the token CE as a secondary objective. That is consistent with the
information view (a pass earns when there is something to extract from the input it reads)
and with the rank-collapse diagnosis (the slot state has the least to extract), and it
points at the Thought Register lane rather than at any core-side lever. A first version
of this section said the restriction cost two thirds of the reading; the Parcae control
above refutes that, and the correction is recorded here rather than hidden.
