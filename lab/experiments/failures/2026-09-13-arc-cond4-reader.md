# Planned: cond4 as the READER — a non-shared stack between the loop and z

Status: failure

Date: 2026-09-13 (frozen before any GPU step of either arm; the 5090 is running other
work and no smoke of either arm exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).
Sibling panel: [`2026-09-13-arc-thought-register.md`](2026-09-13-arc-thought-register.md).
Design note:
[`2026-09-13-cond4-as-the-reader-of-the-register.md`](../../../.agents/notes/proposed/architecture/2026-09-13-cond4-as-the-reader-of-the-register.md).
Theory this scores against:
[`2026-09-13-information-view-of-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md).

## Question

`tul.cond_layers` — Wolfe's 2026-09-03 sketch, prelude → core loop → a few NON-SHARED
blocks → z → coda — has never been measured. Its only two runs, R7f/R7d
([`failures/2026-09-03-tul-think-once-panel.md`](../failures/2026-09-03-tul-think-once-panel.md)),
detonated at step ~1010 under `warmup: 0` and the absmean ternary rule, both of which the
tree has since replaced (the 1,000-step ramp; `norm_match`). The question: does four
blocks of NON-SHARED capacity between the loop's exit and the readers of `z` buy anything
— on the strict ruler, and on the Thought Register where those four blocks are the first
mechanism in the arc that lets a span's M cells relate to each other OUTSIDE the
weight-shared loop?

## Hypothesis

Two claims, and they point in opposite directions on purpose.

**The theory's claim, and it is a theorem.** A deterministic map after the loop carries at
most what the loop's exit carried (`Joint.mi_map_le`, `lab/theory/tul_information/`). The
stack is deterministic and reads nothing the exit did not hold. So NO stack can make the
passes matter: the K-curve must stay flat. If it does not, the first response is the
strict leak gate, not belief.

**The reader's claim.** The information account says the slot loop's value has to come
from the READER'S OBSERVATION, and the slot channel's measured reader headroom is tiny
(one pass is worth 0.001–0.003 nats through the real coda; a 10 % rms perturbation of the
exit moves the coda's loss by 0.0005). Four non-shared blocks are a capacity lever ON that
reader. If the reader is the binding constraint, CE and the worth profile move while the
K-curve does not. That combination is the predicted outcome, and it is a result, not a
null: it says the slot channel is reader-limited and not pass-limited.

On the register the stack is also the first place where a span's four cells can mix
without the shared loop's weights, so if the cells stay apart (`val/slot_cell_eff_rank`
above 2.0) the stack is the arm that says whether four distinguishable cells are worth
anything to a reader that can combine them.

## Method

Two arms, each ONE factor against its partner, 5,000 steps, seq 1024, batch 6, seed 1,
ramp 1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route off, retention
off, `tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`,
`mux_beta: 0`, `spandec` on at J 32, `detach_z: false`.

| arm | config | one factor against |
| --- | --- | --- |
| `slot-spandec-strict-cond4` | `tul_slot_spandec_strict_cond4.yaml` | `slot-spandec-strict` (`tul.cond_layers` 0 → 4) |
| `slot-register-m4-cond4` | `tul_slot_register_m4_cond4.yaml` | `slot-register-m4` (`tul.cond_layers` 0 → 4) |

`slot-register-m4` is itself queued in the register panel and is NOT yet run; the
`register-m4-cond4` row is only readable once its partner lands.

### What the stack is, exactly

`cond_layers` ordinary `MORPHBlock`s, non-shared, built with the CORE's attention kwargs
and drawn AFTER every shared parameter (so a `cond_layers: 0` model from the same seed has
byte-identical weights). They run ONCE — not per pass — over the compact axis straight out
of `_tul_core`, BEFORE the register's mean:

* `slot_cells: 1` — the S slot axis, plain causal, no mask. This is the placement the
  stack has always had.
* `slot_cells: 4` — the S·M CELL axis, under the SAME relation the register loops with,
  from ONE builder (`transformer.slot_cell_relation`) the core stage also calls.

Everything downstream reads the stack's output: the span decoder grades it (its mean on a
register model), SIGReg reads it, and `prefix_project` writes it into the coda's prefix
cells (cell `i` → prefix cell `i` on a register model). `tul_slot_state_probe` runs it too,
so `val/slot_eff_rank` measures what the coda reads.

**The relation, and the defect the build found in how it was delivered.** The mask allows
cell `i` of slot `k` every cell of its OWN slot — LATER siblings included — and every cell
of slots `< k`. See the Method amendment below: through `tg_allow`/`tg_comp_allow` that
executed as plain flattened causal, and it is now delivered as `tg_relation`. The stack is
held to the loop's relation, whatever it is, which is the contract; both stages call the
one builder and both now pass it the same way.

`tul.loop_reach > 0` with a stack RAISES: the reach budget spends every cross-cell route
in core layer 0, and a stack running the unbudgeted relation afterwards would re-open it.

### Method amendment, 2026-09-13 (before any GPU step of either arm; predictions unchanged)

**The register's documented in-loop relation did not execute, in the loop OR in this
stack, and it was fixed before launch.** `blk[p][q] = slot(p) >= slot(q)` is a SUPERSET of
flattened causal, and `tg_allow` / `tg_comp_allow` are ANDed into an already-causal
relation and can only NARROW. At `loop_reach 0` — both arms — a cell therefore never read
a LATER cell of its own slot: the mask was a no-op (CPU fixture, all three identical at
9.8935718536).

The fix adds ONE explicit mechanism, `tg_relation`: a `[*,1,S,S]` bool that REPLACES a
branch's causal term instead of narrowing it, honoured by both halves of a TG-restricted
layer, wired only on the register's cell axis — the loop's core stage and
`_tul_cond_apply`. After the fix, on the same fixture, a `cond_layers: 2` register model's
loss is 9.9034347534, against 9.9073696136 for a build where both stages run flattened
causal and 9.9034347534 for one where both run an independently written
all-true-within-slot mask. The stack is isolated by the perturbation probe: nudging cell 3
of a slot at the loop EXIT moves cell 0 of that slot's STACK output, and moves it by
exactly 0 under a flattened-causal build.
`cond_layers: 0`, `slot_cells: 1`, the token path, the prelude and the coda are
bit-identical on six fixtures.

**Nothing else changed.** Both arms, their configs, the readout and every prediction below
are untouched. `slot-spandec-strict-cond4` runs at `slot_cells: 1`, where the stack takes
no mask at all and this fix cannot reach it; only `slot-register-m4-cond4` is affected,
and it is affected by getting the relation it was always documented to run.

### The gate, built in this change

`tests/test_tul_cond4_strict.py`, **25 passed** after the amendment above. It covers: the
coda's read and the span decoder's grade are the STACK's output (both readers, two-sided,
with a bypass caught); the stack runs on the CELL axis and not the mean; cell `i` lands in
prefix cell `i`; the stack's relation is the loop's own builder AND arrives by the same
delivery, its mask is load-bearing, its EXECUTED relation differs from a flattened-causal
build and equals an all-true-within-slot build, a stack cell reads a later cell of its own
slot under a perturbation (and does not under a causal build), and it is causal across
slots under a perturbation; pad cells never reach a valid cell and a pad slot's
write still goes to the dump row; the strict leak cut and `zero == all_slots` still hold;
the slot-gain constraint still acts and prints no INERT notice; the forced-depth lever
moves the loop exit, the stack output AND the coda's read; the layer-pass count is per
cell; the rank probe reads the stack; and the `loop_reach` refusal.

`cond_layers: 0` bit-identity is owned by `tests/test_tul_think_once.py` and is not
duplicated.

M = 1 bit-identity of the NEW placement was proved by RUNNING the pre-change source
(commit `b1075f2`) and this tree in two processes on the same fixture and seeds:

```
cond=2 loss=9.9817762375 logit_sum=553.297651 n_fin=10584 passes=950.0 gradsum=1812.370773
cond=4 loss=9.9952659607 logit_sum=414.534669 n_fin=10584 passes=982.0 gradsum=1916.135578
```

Every column matched to the last printed digit. Seven source-level sabotages, each
anchored to exactly one occurrence in the shipped source: **7/7 CAUGHT**.

### Readout

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`, paired bootstrap over rows. Partner: **+0.0016 [+0.0013, +0.0019]** /
   **+0.0002**.
2. `worth_profile.py --rows 192 --modes auto` → `all_slots` worth by offset bin. On a
   strict arm `all_slots` equals `zero` by construction (a CHECK, not a result); the
   partner reads **0.1865**.
3. `val/slot_eff_rank` and `val/slot_pairwise_cos` (partner **5.7598** / **0.7104**), and
   on the register arm `val/slot_cell_eff_rank` / `val/slot_cell_pairwise_cos`. On a cond4
   arm these now read the STACK's output, which is what the coda reads.
4. `slot_depth_isolation.py` — per-pass contribution, per slot.
5. Depth-6 CE paired on 480 rows against the one-factor partner; tok/s and peak memory at
   step 200. Partner: 3,337 s, **11,759 tok/s**.
6. The matched-compute row, carried on every slot filing: `plain-depth1` at 14
   block-passes/token is 0.25 nats AHEAD of `spandec-mask` and 0.33 ahead of the mask
   ruler at 5k.


### Method amendment, 2026-09-13 (baseline re-read; predictions unchanged)

The strict ruler's `val/slot_eff_rank` 5.7598 / `val/slot_pairwise_cos` 0.7104 quoted
above comes from the trainer's last `[VAL]` line and did NOT reproduce on the saved
checkpoint: the model's own `tul_slot_state_probe`, run by
`lab/divergence/slot_rank_anatomy.py` on the DGX Spark, reads 11.9597 / 0.5635 on the
480-row probe panel and 13.8466 / 0.5201 on the trainer's own val recipe
(`lab/experiments/results/2026-09-13-rank-anatomy/README.md`). The cause is now measured
(`lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`, commit ab7c353): the run
logged its rank from the PRE-7a24adf probe, which rebuilt the front UNRESTRICTED on a
strict model; the checkpoint reproduces 5.7598 / 0.7104 / CE 4.4249 to four decimals under
that bare front and reads 13.85 / 0.520 under the shipped probe on the trainer's val recipe.
So the "rank 5.76" headline of this lane was an inert-instrument number; the real per-row
rank of the ruler is about 13 of ~50 slots (cosine 0.52), and every `val/slot_eff_rank`
logged before 7a24adf by a front-restricted arm (strict, or tg_restrict at scope all) is a
bare-front number. Scoring rule for every rank prediction in this file: the
partner's baseline is the SAME instrument on the SAME rows as the arm (the sweep's 480
rows, per-row median), never the trainer's logged figure. Rank predictions stated as
"vs 5.7598" are scored against that re-read baseline; the predicted DIRECTION and the
thresholds are unchanged. Also recorded from the same instrument: the prelude makes the
rank (seed 2.24 to entry 12.75 per row), six passes leave it at 13.17, and the prefix
write cuts it to 7.16 while doubling the cell count.

## Predictions (frozen)

Written before any GPU step of either arm. Probabilities are the builder's.

- **P-1 (the theory's prediction, both arms).** Token **K3−K6 below 0.002** on
  `slot-spandec-strict-cond4` AND on `slot-register-m4-cond4`. **90 %.** Reasoning: a
  deterministic stack after the loop carries at most what the exit carried
  (`Joint.mi_map_le`), so it cannot make a later pass matter; and every one of twelve
  strict arms already sits in [−0.0008, +0.0003]. Against it: nothing in the theorem
  bounds a TRAINING effect — a better reader changes what gradient the loop receives, and
  the passes could learn to differ for that reason. Residual: 10 % at or above 0.002.
- **P-2 (the first pass, strict arm).** `slot-spandec-strict-cond4` token **K1−K6 below
  0.005**, the band every strict arm sits in, against the partner's +0.0016. **85 %.**
  Residual: 15 % above, which would be the first strict arm to leave the band.
- **P-3 (CE, strict arm).** `slot-spandec-strict-cond4` depth-6 CE, token-paired on 480
  rows, is **better than `slot-spandec-strict` by 0.02 to 0.20 nats**. **55 %.**
  Reasoning: four non-shared blocks on ~64 positions is real capacity placed exactly where
  the reader is, and the coda is only four blocks deep. Against it: more parameters to
  train in 5,000 steps, and every capacity arm in this arc has paid an early price.
  Residual: 25 % better by less than 0.02 or unchanged, 20 % worse than the partner.
- **P-4 (the worth profile moves).** `slot-spandec-strict-cond4`'s `all_slots` worth is
  **above 0.20**, against the partner's 0.1865. **50 %.** Reasoning: on a strict arm the
  slot channel is the only cross-span route, so a better reader of that channel shows up
  here directly. Residual: 35 % in [0.17, 0.20], 15 % below 0.17 (the stack costs the
  channel).
- **P-5 (the pair is consistent).** If P-3 holds, `slot-register-m4-cond4` also beats
  `slot-register-m4` on depth-6 CE, token-paired, **by more than 0.00 nats**. **60 %.**
  Reasoning: the same capacity lever on the same reader. Against it: on a register the
  stack runs over 4x the positions and mixes cells the register was trying to keep apart.
  Residual: 40 % the register arm does not follow the strict arm's sign.
- **P-6 (the stack does not collapse the register's cells).** `slot-register-m4-cond4`'s
  `val/slot_cell_eff_rank` is **within 0.5 of `slot-register-m4`'s**, and above 2.0 if
  its partner is. **45 %.** Reasoning: the probe now reads the STACK's output, and the
  stack is four blocks of mixing over cells that already read each other in the loop —
  mixing is a collapse mechanism as much as a separation one. Residual: 35 % the stack
  LOWERS the within-slot rank by more than 0.5, 20 % it raises it by more than 0.5.
- **P-7 (rate, strict arm).** `slot-spandec-strict-cond4` clears **10,500 tok/s** at step
  200 against the partner's 11,759. **70 %.** Reasoning: 4 x 64 = 256 extra block-passes
  per row against the prelude+coda's 8 x 1152 = 9,216 and the core's 6 x 64 x 6 = 2,304 —
  arithmetic says roughly 2 % more block work. Against it: the stack's blocks are new
  parameters with their own optimizer state and their own backward.
- **P-8 (rate, register arm).** `slot-register-m4-cond4` clears **7,500 tok/s**. **55 %.**
  Reasoning: 4 x 256 = 1,024 extra block-passes per row on top of the register's own
  eager-only core stage, and the stack's attention is eager too (the S·M relation tensors
  force it). The partner's own rate is unmeasured, which is why this is 55 and not higher.
- **P-9 (it fits).** Both arms run 5,000 steps without an OOM on the 5090 at batch 6.
  **70 %.** Reasoning: `L_total` is unchanged from each partner (1,152 and 1,280); the
  stack adds four blocks' activations over 64 and 256 positions. Against it: the ruler
  already sits near the card's limit at ~26 GB with a 3-monitor desktop on it.

## Binding

If **P-1 holds and P-3 holds** — flat K-curve, better CE — the slot channel is
reader-limited, not pass-limited, and the information account's extractability reading is
confirmed on the one lever that tests it directly. The next arm is more reader, not more
loop: a deeper stack, or the cross-attending span decoder the register note names.

If **P-1 holds and P-3 fails** — flat K-curve, no CE move — capacity between the loop and
the readers buys nothing either, and the slot channel is limited by neither the passes nor
the reader's depth. That closes the reader lane and leaves the WRITE (what the loop is
asked to produce) as the only thing untried.

If **P-1 FAILS** — K3−K6 at or above 0.002 with a paired CI clear of zero — run the strict
leak gate (`tests/test_tul_strict_geometry.py`) before believing it. If the gate passes
and the number stands, a deterministic post-loop map moved the per-pass curve, and the
bridge from the theorems to the measurement is broken; the place to look is the assumption
list in `lab/theory/tul_information/README.md`.

If **P-6 fails downward** — the stack collapses the register's cells — the stack and the
register are fighting over the same state, and the register's follow-up (per-cell targets)
comes before any more reader capacity.

## Not verified before launch

* **No GPU step of either arm.** Every cost number above is arithmetic. The smokes are
  handed back unrun.
* **No wall clock and no memory figure at `d_model` 1024.** The partner's 3,337 s and
  11,759 tok/s are measured; neither arm's is.
* **`slot-register-m4` has not run either.** The `register-m4-cond4` row is unreadable
  until its partner lands, and P-5/P-6 are stated against a number that does not exist yet.
* **The stack's attention is EAGER on a register model** (the S·M relation tensors force
  it), which the strict partner avoids under `tg_scoped_kernels`. A second difference from
  the partner, sized only by the smoke.
* **The widened relation has never run at `d_model` 1024.** The two-sided losses and the
  perturbation probes behind the 2026-09-13 amendment are on the 64-dim CPU fixture. The
  argument (an AND into causality cannot widen, a replacement can) is shape-independent;
  the numbers are not.
* **`cond_layers: 4` is not swept.** 2 and 8 are unbuilt, and nothing here says 4 is the
  right number.
* **The forward+backward check at the composed config ran on CPU with
  `model.use_kernels=false`, `hc_use_kernel=false` and `tg_scoped_kernels=false`**, on a
  144/160-position packed batch, not the 1,152/1,280 the arms train on.

## Results

Both arms ran 5,000 steps on the 5090 (`arc/run_recon.sh`, 2026-09-13; `slot-spandec-strict-cond4`
at commit 368e108 19:35-20:39, `slot-register-m4-cond4` at 368e108 20:52-21:59), both
`DONE ... exit=0 verdict=HEALTHY last=4999`. Artifacts:
[`results/2026-09-13-cond4/`](../results/2026-09-13-cond4/) (sweeps at 2,500 and 5,000,
worth profiles, state probes, `paired_gaps_5000.txt`, run-log excerpts). The GPU-utilisation
watch over each arm's first vals read 13 of 72 and 6 of 108 samples under 50 %, single-sample
dips only; neither arm stalls the card (the register arm's pre-53c0497 probe did).

### The numbers

| | strict-cond4 | partner `slot-spandec-strict` | register-m4-cond4 | partner `slot-register-m4` |
| --- | --- | --- | --- | --- |
| token K1−K6 [95 % CI] | **+0.00026** [+0.00016, +0.00036] | +0.0016 [+0.0013, +0.0019] | **+0.00062** [+0.00047, +0.00078] | +0.0020 [+0.0016, +0.0023] |
| token K3−K6 | **−0.00008** [−0.00013, −0.00003] | +0.0002 | **+0.00022** [+0.00013, +0.00031] | −0.0003 |
| spandec K1−K6 | +0.0005 | +0.0031 | −0.0001 | +0.0072 |
| depth-6 CE, paired vs partner (span − full) | **+0.0057** [+0.0028, +0.0084] | — | **+0.0159** [+0.0131, +0.0184] | — |
| `all_slots` worth (offset-0 zero worth) | 0.1809 (0.744) | 0.1865 (0.757) | 0.1811 (0.824) | 0.1942 (0.796) |
| `wrong_seed` worth | 0.0786 | 0.0426 | 0.0821 | 0.0680 |
| `val/slot_eff_rank` / `slot_pairwise_cos` (trainer probe, same instrument) | 10.87 / 0.479 | 13.85 / 0.520 | 13.52 / 0.429 | 10.57 / 0.656 |
| `val/slot_cell_eff_rank` / `slot_cell_pairwise_cos` | — | — | 1.311 / 0.799 | 1.245 / 0.943 |
| step-200 tok/s | 10,730 | 11,759 | 9,648 | 9,582 |
| peak memory | 16.5 GB | 10.2 GB | 22.0 GB | — |
| `Final val_loss` (trainer) | 4.4348 | — | 4.4857 | 4.4696 |

The strict pairing is exact (501,106 tokens both); the register pairing is exact within the
register family (511,089 both, `prefix_k` 4), so the register row carries no packer confound
against ITS partner. The strict-cond4 CE loss sits at offsets 0-4 (+0.013 to +0.029) and
is within noise from offset 5 on; the register-cond4 loss is flat across offsets (+0.012 to
+0.030 everywhere).

### Scoring

- **P-1 TRUE (both arms).** K3−K6 −0.00008 and +0.00022, both under 0.002 by an order of
  magnitude. A deterministic stack after the loop did not make a later pass matter.
- **P-2 TRUE.** Strict-cond4 K1−K6 +0.00026, under 0.005, and one sixth of the partner's
  +0.0016: the stack made the FIRST pass matter less, not more.
- **P-3 FALSE (20 % residual band).** Strict-cond4 is 0.0057 nats WORSE than the ruler,
  paired, CI clear of zero. Four non-shared blocks on the slot axis cost CE at 5k.
- **P-4 FALSE (35 % residual band).** `all_slots` 0.1809, inside [0.17, 0.20] and below
  the ruler's 0.1865. The only worth that moved is `wrong_seed`, 0.043 → 0.079: the stack
  makes the cell more specific to its own seed without making it worth more to the coda.
- **P-5 condition not met (P-3 failed); the sign is consistent.** Register-m4-cond4 is
  0.0159 nats worse than `slot-register-m4`, paired, so the register arm follows the strict
  arm's sign: the stack costs CE on both partners.
- **P-6 TRUE.** Within-slot rank 1.311 vs the partner's 1.245, within 0.5; neither is above
  2.0. The stack did pull the cells apart in cosine (0.943 → 0.799) and raised the per-row
  rank 10.57 → 13.52 (back to the strict ruler's 13.85), the first cond4 reading that
  moved a rank instrument, and nothing downstream moved with it.
- **P-7 TRUE.** 10,730 tok/s, over the 10,500 bar (9 % under the partner).
- **P-8 TRUE.** 9,648 tok/s, over 7,500 and level with the partner's 9,582.
- **P-9 TRUE.** Both arms ran 5,000 steps without an OOM; peaks 16.5 GB and 22.0 GB.

### Verdict

**Failure.** The binding clause "P-1 holds and P-3 fails" fires: capacity between the loop
and the readers buys nothing, and the slot channel is limited by neither the passes nor
the reader's depth. On the register the stack is the first mechanism in the arc that
separates the four cells (cosine 0.94 → 0.80) and restores the row rank the register had
cut, and the coda does not pay for either: worth 0.181 vs 0.194, CE 0.016 worse. This is
the third arm in a day (register, ultralight, cond4) to move a rank or separation instrument
with no downstream effect; rank of `z` is not the binding constraint.

Read with the literature mining of the same evening
([`docs/references/looping-depth/2026-09-13-lit-mining/`](../../../docs/references/looping-depth/2026-09-13-lit-mining/README.md)):
the memory-budget separation (arXiv 2605.30757) bounds a one-slot-per-span loop by what
one slot holds however many blocks read it afterwards, and Tiny Autoregressive Recursive
Models (2603.08082) finds the terminal-only readout is the wiring that cannot learn
cross-position dependence. Both say the reader is not where the fault is.

### Not verified

- 5,000 steps, one seed each; the 0.006 and 0.016 CE costs are early-training prices and
  the horizon rule applies (capacity arms pay early). Neither arm ran to 20k.
- The rank baseline is the trainer's own probe on its val stream; `slot_rank_anatomy.py`
  was not re-run on these checkpoints.
- `slot_depth_isolation.py` (readout 4) was not produced by the runner; only the depth
  state probe exists.
- `cond_layers` 2 and 8 remain unbuilt; nothing here says 4 was the right number, only that
  4 did not move the reader.

