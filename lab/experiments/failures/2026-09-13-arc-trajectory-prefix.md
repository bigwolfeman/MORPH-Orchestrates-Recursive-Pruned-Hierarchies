# Planned: the coda reads the loop's trajectory, not only its exit

Status: failure

Date: 2026-09-13 (frozen before any GPU step of any arm; the 5090 is running the math
panel and no smoke of any arm here exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). One-factor
partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).
Design note:
[`2026-09-13-trajectory-as-the-prefix-content.md`](../../../.agents/notes/rejected/architecture/2026-09-13-trajectory-as-the-prefix-content.md).
Theory:
[`2026-09-13-information-view-of-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md).

## The two facts this arm sits between

**Fact 1 — the coda only ever sees the exit.** `prefix_project` writes ONE state, the
loop's last iteration, into all `prefix_k` cells through `W_prefix[k]`. Whatever the
intermediate passes computed is discarded at the boundary. Twelve arms read token K1−K6
inside [−0.0001, +0.0033] (the partner **+0.0016 [+0.0013, +0.0019]**, K3−K6 **+0.0002**),
and in every one of them the instrument is measuring depth through a one-state window.

**Fact 2 — and the theory says the trajectory carries the ENTRY, not the exit.** The
sorry-free Lean result of 2026-09-13: for a deterministic map,
`I((z1,…,zT); Y) = I(z1; Y)` and `I(zT; Y) ≤ I(z1; Y)`. A trajectory prefix therefore
cannot carry MORE than the entry does, and any win it shows is information the exit LOST
from the entry — a recovery, not a loop gain. **That is why this panel has a binding
control and not just a content control.**

## Question

Does giving the coda the loop's intermediate states, instead of `prefix_k` copies of its
exit, buy anything — and if it does, is the win the TRAJECTORY or just the ENTRY?

## Hypothesis

If the loop computes anything the exit does not keep, a coda that can read the
intermediate states will use it, and `trajectory` beats `exit_repeat` at the same cell
count. The theory predicts the win, if any, is bounded by what `entry_exit` recovers: two
cells holding `z1` and `zT` carry the same information as the whole trajectory. So

* `trajectory` − `exit_repeat` > 0 means the passes hold something the exit dropped;
* `trajectory` − `entry_exit` ≈ 0 means that something is just the ENTRY, and the honest
  fix is a residual entry path, not a trajectory write;
* `trajectory` − `entry_exit` < 0 with both beating `exit_repeat` would contradict the
  data-processing bound and means the measurement is wrong, not the theory.

## Method

Three arms, each ONE factor against each other at `prefix_k: 6`, 5,000 steps, seq 1024,
batch 6, seed 1, ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route
off, retention off, `tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`,
`mux_beta: 0`, `spandec` on at J 32.

| arm | config | `prefix_source` |
| --- | --- | --- |
| `slot-spandec-strict-traj` | `tul_slot_spandec_strict_traj.yaml` | `trajectory` |
| `slot-spandec-strict-entryexit` | `tul_slot_spandec_strict_entryexit.yaml` | `entry_exit` |
| (`slot-spandec-strict-trajrep`) | `tul_slot_spandec_strict_trajrep.yaml` | `exit_repeat` — built and tested, NOT queued (Wolfe, 2026-09-13) |

**`exit_repeat` is the content control and it is out of the queue.** It is built, tested
and composable; the scope call of 2026-09-13 cut it to spend the GPU on the register
instead. That has a cost and the cost is named here: without it, `trajectory` has no
same-cell-count partner, and its comparison against `slot-spandec-strict` mixes the cell
CONTENT with the cell COUNT (k=6 against k=2). **`trajectory` vs `entry_exit` is the one
clean pair in this panel** — same `prefix_k`, same packer, same `L_total` 1408 — and every
reading must be read against that, never against the k=2 partner alone.

### The write rule

Under `trajectory`, cell k of a slot gets the loop state AFTER pass k, through the SAME
`W_prefix[k]`, plus a zero-init per-cell pass embedding `E_pass[k]`. The EXIT state is
always written, always into the LAST cell, and never duplicated: a slot that ran d
iterations fills `min(d, K)` cells and the rest are PAD. A pad's carrier is EXACTLY zero
(`E_pass` included), it is cut out of the coda's key set, and it keeps only its own
self-edge.

**Zeroing the pad's SOURCE state is not enough, and that is a measured defect from this
build, not a hypothetical.** `prefix_project` adds `E_pass` AFTER the projection, so a pad
would otherwise carry `E_pass[k]` — a learned constant announcing how deep the slot went.
Cutting it out of the key set does not cut it out of the CCA causal conv or the `W_v_prev`
value shift, which run inside a slot's own segment and carry cell k into cells k+1…K−1
INCLUDING the exit cell. Before the fix `E_pass.grad.abs().sum()` read **7407** on a
fixture whose cells are mostly pads, against **0.25** on the `exit_repeat` twin. The fix
zeroes `values` at pad cells and the guard edits a pad cell and asserts no logit moves.

Under `entry_exit`, cell 0 holds the ENTRY `z1 = core_init(e)` and the last cell holds the
exit; the cells between are exit copies. `E_pass` applies the same way.

### The gate, built in this change

`tests/test_tul_prefix_source.py`, **53 cases**, run green with the rest of the TUL suite
(`tests/test_tul_slot_register.py tests/test_tul_setup_keys.py
tests/test_tul_prefix_source.py tests/test_tul_loop_reads_tokens.py -q` → **145 passed**).
It covers: `exit` is bit-identical to the pre-knob tree on pinned values; the cells are
actually distinct under `trajectory`; the forced-depth write rule (`min(d, K)` written,
exit always last); pad inertness four ways (logit, key-set structure, self-edge, and a spy
on the SHIPPED `_back_region`'s `attn_kwargs`); gradient reaching every written pass
directly through its own cell; the strict leak test with a `tg_restrict` control; the
worth-profile modes; the depth levers; every refusal; and both configs composing through
Hydra and `build_tul_runtime`.

Eight source-level sabotages, 8/8 CAUGHT — but **A2 MISSED on the first pass** and that
miss is the reason the `_back_region` spy exists: `test_the_pad_cells_are_out_of_the_
codas_key_set` called `_tul_pad_cell_narrow` itself, so deleting the call from
`_forward_tul` left it green. "The guard passed" and "the guard is watching the shipped
path" are different claims, and this is the second time in two days that difference has
bitten (the 2026-09-12 coretok C1 miss was the same class).

### Readout

Same six instruments as the register panel: `core_depth_sweep.py --depths 1,2,3,6,9,12,16
--rows 480`; `worth_profile.py --rows 192 --modes auto` (partner `all_slots` = `zero` =
**0.1865**); the slot-state rank pair; `slot_depth_isolation.py`; depth-6 CE paired on 480
rows; wall clock and tok/s at 200 (partner 3,337 s, 11,759 tok/s). Plus one specific to
this panel: **`E_pass.grad` per cell**, which says whether the model uses the pass index at
all, and the **per-cell attention mass** the coda puts on each prefix cell.

**The packer confound, stated once.** `prefix_k: 6` makes `L_total` 1408 against the k=2
partner's 1152, so these arms see about 1093 real tokens a row against the partner's 1033.
The three `prefix_source` arms pair cleanly with EACH OTHER and NOT with
`slot-spandec-strict`.

## Predictions (frozen)

Written before any GPU step of any arm. Probabilities are the builder's. The three marked
**(theory)** are the Theory agent's, transcribed unchanged.

- **T-1 (theory).** `CE(traj) − CE(entry_exit) ≥ −0.005` nats: the trajectory does not beat
  the entry-plus-exit pair by more than noise. **70 %.** Reasoning: the data-processing
  identity says the whole trajectory carries exactly what the entry carries, so two
  well-chosen cells suffice. Against it: the coda is a small stack that has to EXTRACT the
  information, and an easier representation can beat an equally informative one.
- **T-2 (theory).** `CE(traj) − CE(trajrep) ≥ −0.020` nats: the trajectory does not beat
  the exit-repeat control by more than 0.020. **60 %.** **This prediction is now
  UNSCORABLE as written** — `trajrep` was cut from the queue on 2026-09-13. It is kept
  here, unedited, so the record shows what was predicted and what the scope call cost.
- **T-3 (theory).** `slot-spandec-strict-traj` token **K1−K6 ≤ 0.005**, read against
  `trajrep`. **75 %.** Also degraded by the `trajrep` cut: the reading against
  `entry_exit` is what will be scored, and that substitution is recorded here rather than
  made silently at scoring time.
- **P-1 (the trajectory is used at all).** `E_pass.grad.abs().sum()` at 5,000 on
  `-traj` is **above 1 %** of `W_prefix.grad.abs().sum()`. **70 %.** Reasoning: the pass
  embedding is free capacity in the coda's input; a model that ignores the trajectory
  entirely still has reason to use a per-cell bias. This is a WEAK instrument on purpose —
  it detects use, not value.
- **P-2 (entry_exit beats the k=2 partner).** `slot-spandec-strict-entryexit` depth-6 CE,
  token-paired, is better than `slot-spandec-strict` by **more than 0.01** nats. **55 %.**
  Reasoning: it writes 6 cells against 2 and it hands the coda the entry, which is the
  richest single state the theory allows. Confounded by the packer difference, which is
  why the threshold is small and the pair-with-traj comparison carries the weight.
- **P-3 (no arm here moves the per-pass curve).** Both arms read **K3−K6 below 0.002**.
  **80 %.** Reasoning: twelve arms, and the write is downstream of every pass; changing
  what the coda READS cannot make pass 4 compute more than pass 3 did.
- **P-4 (rate).** `-traj` clears **10,000 tok/s** at step 200 against the partner's 11,759.
  **50 %.** Reasoning: `L_total` 1408 against 1152 is 22 % more coda positions, and the
  trajectory write is a scatter of the same size as the exit write.

## Binding

If **T-1 holds** — trajectory ≈ entry_exit — the loop's intermediate states carry nothing
past the entry, exactly as the identity says, and the cheap fix is a residual ENTRY path
into the coda, not a trajectory write. That result also retires "the coda only sees the
exit" as an explanation for the flat K-curve.

If **T-1 fails clearly** — trajectory beats entry_exit by more than 0.005 — the
information identity is not the operative constraint at this scale, because EXTRACTION
cost is. The coda's ability to read a representation is then the lever, and the next arm
is a wider or deeper read of the same information rather than more information.

If **P-2 holds and T-1 holds** — the entry is worth something and the trajectory adds
nothing on top — the shipped write should carry the entry alongside the exit at
`prefix_k: 2`, which is a cheap change with a measured reason.

If **P-1 fails** — the model does not even use `E_pass` — the write is being ignored and
the CE readings in this panel are about the packer, not about the prefix content. Say so
and stop.

## Not verified before launch

* **No GPU step of any arm.** Cost numbers are arithmetic; the smokes are handed back
  unrun.
* **`exit_repeat` is built and will not run.** T-2 is unscorable and T-3 loses its
  intended reference. That is a scope decision, recorded, not a measurement.
* **The `prefix_k: 6` choice is a guess.** Six cells at mean depth 6 means a typical slot
  fills most of them, but the Poisson draw puts real mass at d ≤ 4, so a large fraction of
  cells are PAD in practice. Nothing has measured the realised fill rate on real rows.
* **`E_pass` is zero-init and unregularised.** No alternative was tried.
* **The pad-zeroing fix is verified on a CPU fixture only.** The 7407-vs-0.25 reading that
  motivated it is from that fixture.
* **The comparison against `slot-spandec-strict` carries the packer confound** described
  above, and no arm in this panel removes it.

## Results

All three arms ran (2026-09-13/14, 5,000 steps each, seed 1, `prefix_k` 6): `traj`
(9,734 tok/s), `trajrep` (10,469; queued back in on 2026-09-14 after the register pair
read out, so T-2 and T-3 are scorable as written) and `entryexit` (10,440). All HEALTHY.
Artifacts: [`results/2026-09-13-trajectory/`](../results/2026-09-13-trajectory/) (sweeps
2500/5000, worth, state probes, the paired gaps, the P-1 probe).

**Own sweeps at 5,000 (480 rows), token CE:**

| arm | depth-6 CE | K1−K6 | K3−K6 | worth `all_slots` (offset 0) |
|---|---|---|---|---|
| ruler `slot-spandec-strict` (k=2) | 4.3474 | +0.0016 | +0.0002 | 0.1865 |
| `traj` | 4.3203 | **+0.0342** | +0.0034 | 0.2648 (1.875) |
| `trajrep` | 4.3208 | +0.0019 | +0.0003 | 0.2681 (1.623) |
| `entryexit` | 4.3196 | +0.0014 | +0.0003 | 0.2850 (2.022) |

**Paired gaps at 5,000 (`span_budget_profile.py`, `budget_web_full`, 480 identical rows;
gap = second − first, negative favours the second):**

| pair | depth 6 | depth 1 |
|---|---|---|
| ruler → traj | −0.0271 [−0.0297, −0.0243] | |
| ruler → trajrep | −0.0302 [−0.0327, −0.0275] | −0.0299 |
| ruler → entryexit | −0.0311 [−0.0338, −0.0282] | −0.0313 |
| traj → trajrep | −0.0030 [−0.0055, −0.0007] | −0.0353 |
| traj → entryexit | −0.0042 [−0.0065, −0.0021] | −0.0370 |
| trajrep → entryexit | −0.0012 [−0.0035, +0.0011] | −0.0017 |

**P-1 probe** (`lab/divergence/epass_grad_probe.py` on the Spark, `traj` step 5000, 16
val rows, eval mode, training loss mean 8.27): `E_pass.grad.abs().sum()` 0.686 against
`W_prefix.grad.abs().sum()` 608.9, ratio **0.0011**; per cell [0.285, 0.124, 0.047,
0.023, 0.021, 0.185]; `E_pass` row norms 0.14–0.28.

**Scores.**

- **T-1 holds.** CE(traj) − CE(entryexit) = +0.0042 ≥ −0.005.
- **T-2 holds** (scorable after all). CE(traj) − CE(trajrep) = +0.0030 ≥ −0.020.
- **T-3 holds against `trajrep`.** `traj`'s own K1−K6 of +0.0342 is the pad-cell
  artefact of forced-depth eval: at forced depth d the trajectory write fills d cells
  and pads the rest, so the depth-1 row loses five prefix cells. The same-width
  `trajrep` reads +0.0019, and K(traj) − K(trajrep) = 0.032 is the whole reading. The
  traj → trajrep gap of −0.0353 at depth 1 against −0.0030 at depth 6 is the same fact.
- **P-1 fails.** 0.11 % < 1 %. Cells 1 and 6 carry the mass; cells 3–5 are near zero.
- **P-2 holds on its face** (entryexit beats the k=2 ruler by 0.031 > 0.01) **and the
  width control takes it away**: entryexit − trajrep = −0.0012 [−0.0035, +0.0011].
- **P-3 holds** for `trajrep` and `entryexit` (K3−K6 0.0003) and **fails** for `traj`
  (0.0034), by the artefact above.
- **P-4 fails.** 9,734 < 10,000.

## Verdict

**Failure.** The one non-flat slot-loop K-curve of the arc was an eval artefact, and
every CE win in the panel is prefix WIDTH. Six cells beat two by 0.030 nats whatever
they hold (the exit six times, the trajectory, or the entry and the exit), and the
three contents are within 0.004 of each other, with the trajectory write the WORST of
the three. The pass embedding takes 0.11 % of the prefix gradient and the coda reads
the first and last cells only. `entryexit` − `trajrep` = −0.0012 ± 0.0023: the entry
adds nothing an exit copy does not already give.

The binding clause for "T-1 holds" fires (the intermediate states carry nothing past
the entry) but its recommended cheap fix, an entry path at `prefix_k` 2, is NOT
supported: with the width control in place the entry cell is worth zero. The binding
clause for "P-1 fails" also fires: the write is ignored and the panel's CE readings are
about the packer.

## Updated hypothesis

The coda cannot use anything the loop's intermediate states or its entry carry beyond
what one exit copy carries, so "the coda only sees the exit" is retired as an
explanation for the flat K-curve. Prefix width is a real, cheap 0.03-nat lever of its
own (six exit copies at `L_total` 1408), to be named as width and not as loop
contribution. What is left on the slot loop is the target and the wiring
([`2026-09-14-arc-loop-diagnostics.md`](../failures/2026-09-14-arc-loop-diagnostics.md)
reads the same night's mechanism instruments the same way).

