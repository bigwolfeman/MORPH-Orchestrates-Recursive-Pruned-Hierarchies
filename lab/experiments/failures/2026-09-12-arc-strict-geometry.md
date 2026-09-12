# Planned: the strict slot geometry — make the loop the only cross-span channel, then ask it for depth

Status: failure

Date: 2026-09-12 03:32 CDT (frozen before any arm ran; no GPU step of any arm below exists
at filing time). Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).
Design note: [`2026-09-12-strict-slot-geometry.md`](../../../.agents/notes/proposed/architecture/2026-09-12-strict-slot-geometry.md).

Evidence it stands on, all measured and all in this tree:

* **The loop is bypassed, and the number says by how much.** `slot-spandec-mask` at 5,000
  steps, 480 rows ([`2026-09-11-arc-span-decoder.md`](../successes/2026-09-11-arc-span-decoder.md)):
  the whole slot channel (`worth_profile --plan-mode all_slots`) is worth **0.182** nats and
  the loop's own prefix write (`zero`) **0.078**. More than half of what the channel carries
  never passes through an iteration. Under `tul.tg_restrict` the allow relation is
  `causal AND (same span OR j is ANY slot cell)` in the PRELUDE as well as the coda, so a
  cell's seed — `E_slot` plus a bag-mean of its own span — reaches later spans directly, and
  the coda re-adds that same content at every layer through `x0` and the bigram term.
* **The passes contribute nothing.** Same arm, same rows: token K1−K6 **0.0007**. Eleven
  arms since 2026-09-04 sit in [−0.0001, +0.0033].
* **The missing information is long-range, not local.** With NO slot cells at all, cutting
  every cross-span route costs **0.3994** nats [0.3838, 0.4162] at 5,000 on 491,520 paired
  tokens; **0.3149** [0.3025, 0.3314] of it is FLAT at offsets 8+, with a 0.9580 spike at
  offset 1 and 0.2826 at offset 0 ([`2026-09-11-arc-span-budget.md`](../failures/2026-09-11-arc-span-budget.md)).
* **The reader has capacity.** A gradient-fitted z is worth 0.9–2.6 nats to the coda; the
  loop's own z is worth +0.0062 over its entry state on `slot-spandec-mask`.

## Question

If the slot loop is the ONLY route from one span to the next — no cell-to-cell prelude
attention, no cell that re-summarises its own span in the coda, no conv or bigram crossing a
boundary — do the loop's PASSES start to contribute? And if a pass is further restricted so
that reaching m spans back REQUIRES m passes, does the forced depth buy anything at the same
cost?

## Hypothesis

The per-pass K-curve has read zero on every arm because the information the loop was meant to
carry has always had a cheaper route. Close the cheaper routes and the loop either starts to
earn or is shown to be unable to, and the second answer is worth as much as the first. Two
further levers are tested in the same panel because they attack the same "the passes are not
asked for anything they cannot already do" reading from the other side: a per-pass TARGET
(`oracle_z`) and a target that reaches past the next boundary (`spandec_horizon`).

## Method

Five arms, each ONE factor, 5,000 steps, seq 1024, batch 6, seed 1, ramp 1,000, `norm_match`,
`core_fixed_point_lambda` 1.0, prune/carve/route off, retention off. All five compose
`morph/configs/tul_slot_spandec_strict.yaml`, whose own one factor against
`slot-spandec-mask` is `tul.tg_geometry`.

### The geometry (`tul.tg_geometry: strict`)

| stage | allow relation |
| --- | --- |
| prelude | `causal AND bag_id[i] == bag_id[j]`. A token sees its own span's tokens; a cell sees its own span's tokens and its own earlier cells. |
| loop | unchanged — slots attend earlier slots' states, causally. The only cross-span channel. |
| coda | a TOKEN sees its own span plus the PREFIX CELLS of earlier slots (`tg_coda_prefix_reach`); a PREFIX CELL sees ITSELF alone. |

And, part of the same geometry rather than separate knobs: the CCA causal conv and its
`W_v_prev` value shift reset at every segment (`tg_segment_ids` = a span's tokens | that
span's cells | the next span's tokens), the retention carry resets on the same partition
(retention is off on every arm here; the reset is asserted, not assumed), and the coda's
per-layer injections at the slot CELLS are zeroed. That last one is load-bearing: without it
a cell hands its own span's token identities to every later span with no pass in between.

The hash bigram needs no cut on a TUL row — a span's first token is always preceded by the
previous slot's cells, whose input id is the constant `slot_id` — and that is proved
two-sided in the gate rather than assumed.

### The gate, built in this change

`tests/test_tul_strict_geometry.py`. The leak test: with the loop's prefix write zeroed
(`plan_mode="zero"`), a token id in span 0 must move NOTHING outside span 0, **bit-exact 0.0
on CPU fp32**. Two-sided at every point — the `restrict` model run through the same probe
MUST leak, the write-ON strict model MUST move a later span, and the edit must move its own
span. Five sabotages, each re-opening ONE cut route, ALL CAUGHT: prelude window widened,
prelude compressed branch unmasked, coda cell query widened, conv/value-shift reset dropped,
coda cell injections restored. Each sabotage is self-verifying: with no patch the probe reads
exactly 0.0, so a patch that failed to apply would fail its own test.

### The arms

| arm | config | one factor against |
| --- | --- | --- |
| `slot-spandec-strict` | `tul_slot_spandec_strict.yaml` | `slot-spandec-mask` — the GEOMETRY |
| `slot-spandec-strict-reach1` | `tul_slot_spandec_strict_reach1.yaml` | `slot-spandec-strict` — `tul.loop_reach` 0 → 1 |
| `slot-spandec-strict-prev` | `tul_slot_spandec_strict_prev.yaml` | `slot-spandec-strict` — `tul.tg_coda_prefix_reach` all → prev |
| `slot-spandec-strict-oracle` | `tul_slot_spandec_strict_oracle.yaml` | `slot-spandec-strict` — `tul.oracle_z` |
| `slot-spandec-strict-h3` | `tul_slot_spandec_strict_h3.yaml` | `slot-spandec-strict` — `tul.spandec_horizon` 1 → 3 |

**reach1** makes a pass ONE Jacobi step of `h_{t+1}[k] = f(h_t[k-1 .. k])`, so a slot m spans
back first reaches slot k at pass m. The budget is PER PASS, spent in core layer 0 and closed
in layers 1..n-1, with the conv reset per cell in every layer — applying reach w at every
layer carries `w * n_core` per pass, which was measured on the CPU fixture (reach 2 over two
core layers moved a slot four cells away at pass 1) and is why the first implementation was
wrong. Consequence, named: on this arm the core mixes cells once per pass and is
position-local for the rest of it, and that is inseparable from the reach limit.

**oracle** BREAKS THE STANDING RULE against regressing onto the slot state (LCM T3/T4,
CoCoMix §6b, BT §4.2) deliberately, as a test of whether a per-pass target that a one-step
optimiser could match produces a per-pass K-curve. Wolfe decides whether it ever ships.

**h3** grades z on spans s+1..s+3 as one causal run, because the budget's missing part is
flat at offsets 8+ and a target that ends at the next boundary cannot ask for it.

### Cost, arithmetic and not a guess

Block-passes per token at the panel shape (1,024 real tokens per row, 64 slot cells, mean
depth 6), against the measured table in [the span-decoder record](../successes/2026-09-11-arc-span-decoder.md):
ruler `slot-mux-mask-norm-match` 10.745, `slot-spandec-mask` 14.676, plain at mean depth 6
44.000, `plain-coda-matched` 14.000.

* `slot-spandec-strict` adds NO parameter and NO block pass. It adds two `[B, 1, L, L]` bool
  masks per forward (~8 MB each at B 6, L 1152) and moves the PRELUDE's CCA prologue onto the
  eager segment-conv path, which the mask arm already paid for in the coda alone.
* `reach1` additionally moves the CORE's window branch and CCA prologue off the fused path.
  At 64 cells the tensors are small; the smoke decides.
* `oracle` runs T forward+backward passes of a `[B, S, J, V]` readout. One readout is 38.7
  GFLOP per token of J at B 6, S 64, C 1024, V 49169, so T 6 × J 8 is ~5.6 TFLOP against a
  ~50 TFLOP step, about **11 %**. `oracle_z_max_tokens` is 8 and not the decoder's 32 for
  exactly this reason — at J 32 it is ~45 % and the arm would miss the rate floor.
* `h3` takes the decoder from 4.0 to **12.0** block-passes per token, so the arm is 22.7
  against strict's 14.7. This is the arm most likely to be skipped by the rate rule.

The rate floor is 8,086 tok/s at step 200 (`run_slotloop3.sh`); the span-decoder smokes read
10,829–10,888 against the ruler's 12,662.

### Readout, every arm

Runner `arc/run_slotloop3.sh`, KIND `slot`, SWEEP_CKS 2500,5000, OUTDIR
`/home/wolfe/morph-scratch/arc/results/2026-09-12-strict`.

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → K1−K6, K3−K6, `spandec_ce`
   (the H = 1 part on every arm, including h3).
2. `worth_profile.py --rows 192` (`--modes auto`, so `all_slots` is added automatically on a
   `tg_restrict` arm) → zero / shuffle / wrong_seed / all_slots by offset bin.
   **On a strict arm `all_slots` EQUALS `zero` by construction** — the other two routes are
   cut on every forward — so report it as a CHECK on the geometry, never as a result.
3. `slot_z_optimize.py` → `ce_entry − ce_loop`, the write contribution, against the fitted-z
   ceiling.
4. Paired CE against `slot-spandec-strict` (and, for the strict arm itself, against
   `slot-spandec-mask` and `plain-coda-matched`) with
   `lab/divergence/span_budget_profile.py --full A.npz --span B.npz`, 480 rows.
5. Wall clock, `tul/oracle_z_l{t}` (oracle), `tul/spandec_ce_h` (h3), `loop/core_gain_t0`,
   the sustained tripwire.

## Predictions (frozen)

The first five are the orchestrator's framing; the probabilities and the reasoning are the
builder's, written before any GPU step of any arm.

- **P-a (strict COSTS CE at 5,000).** `slot-spandec-strict` is WORSE than
  `slot-spandec-mask`, token-paired at depth 6 on 480 rows, by **0.05 to 0.20 nats**.
  **70 %.** Reasoning: the channel carried 0.182 and the loop's write 0.078, so ~0.10 nats of
  information now has to flow through the loop or not at all, and no arm has yet shown the
  loop able to take up slack. Residual: 20 % that it costs MORE than 0.20 (the cells were
  carrying more than the `all_slots` lower bound says), 10 % that it costs less than 0.05.
- **P-b (`all_slots` == `zero` on the strict arm).** Equal to within bootstrap noise in every
  offset bin. **Not a prediction — a CHECK.** If they differ, one of the geometry's cuts
  stopped firing and the panel is void.
- **P-c (strict gives the passes their first non-zero contribution).** `slot-spandec-strict`
  token K1−K6 above **0.005**. **35 %.** Reasoning: this is the first arm in which the loop is
  the only cross-span route, so the gradient has nowhere else to go — but E7 and the gradient
  probe say one pass already does all the work, and closing a bypass does not by itself make
  pass 4 different from pass 3. Residual: 50 % it lands in [0.000, 0.005] like every arm
  before it, 15 % it goes NEGATIVE (deeper is worse).
- **P-d (reach1's K-curve is large and its CE is not).** `slot-spandec-strict-reach1` token
  K1−K6 above **0.05** (forced by construction) AND its depth-6 CE within **0.02** nats of
  `slot-spandec-strict`. **K-curve above 0.05: 85 %** — it is built in, and the only way it
  fails is if the model learns to ignore cross-cell context entirely. **CE within 0.02: 25 %.**
  Reasoning for the low second number: the arm also makes five of six core layers
  position-local, which is a real capacity cut, and 0.02 is tight. I expect reach1 to be
  0.02–0.10 WORSE.
- **P-e (oracle moves the per-pass curve).** `slot-spandec-strict-oracle` token K3−K6 above
  **0.005**. **30 %.** Reasoning: the loop can almost certainly MATCH the trajectory — the
  target is a smooth function of the entry state and the map has six passes to hit it — but
  matching a teacher's intermediate states is not the same as the coda finding them useful,
  and the MSE term does not touch what the coda reads. The more likely outcome is
  `oracle_z` falling toward zero with the token CE unmoved, which is itself a clean answer.
  **`tul/oracle_z_l{t}` falls monotonically over the T steps at the panel shapes: 80 %**
  (it does at lr 0.2 / 0.1 / 0.05 / 0.02 on the CPU fixture).
- **P-f (h3 moves the FAR bins).** `slot-spandec-strict-h3`'s `all_slots` worth at offset bin
  **16+** exceeds `slot-spandec-strict`'s by more than it does at bins 0-3, proportionally.
  **45 %.** Reasoning: this is the one prediction the budget's SHAPE directly motivates
  (flat 0.31 at 8+), and the H = 1 arm's profile decays hard (0.708 at offset 0 to 0.068 at
  16+). Against it: the decoder may simply spend its extra capacity on the near tokens of
  each of the three spans, which are also offsets 0-3 of THEIR spans.
- **P-g (prev is the worst of the family on CE).** `slot-spandec-strict-prev` depth-6 CE is
  worse than `slot-spandec-strict`'s by **more than 0.03** nats. **75 %.** Reasoning: it cuts
  every reach past one slot, AND a tail dump-bin token reads no cell at all under "prev".
- **P-h (survival).** All five arms reach 5,000 with no sustained tripwire
  (`preclip/total > 1e4` at step >= 200). **75 %.** Reasoning: the recipe's 1,000-step ramp
  plus `core_fixed_point_lambda` 1.0 has held every arm in this family; strict changes the
  attention relation, which is not a known detonation lever. The oracle arm adds a new loss
  term with an untuned weight, which is where I would expect a failure.
- **P-i (rate).** `slot-spandec-strict`, `-prev` and `-oracle` clear the 8,086 tok/s floor at
  step 200: **85 %, 85 %, 65 %**. `-reach1`: **70 %**. `-h3`: **40 %** — the decoder triples
  and 14.7 → 22.7 block-passes per token against a ruler at 12,662 and a spandec arm at
  ~10,850 puts it near 7,000 by simple proportion.

## Binding

If P-a holds and P-c fails — strict costs CE and the passes still read zero — the lane that
closes is "the loop is being bypassed". The next question is not another geometry: it is
whether the slot's exit state can be USEFUL at all, which `slot_z_optimize`'s 0.9–2.6 nat
fitted-z ceiling says it can and the loop's +0.006 says it is not.

If P-c holds, the strict geometry becomes the baseline for every later slot-loop arm and the
mask lineage is retired.

If P-d's CE half holds — reach1 within 0.02 of strict while its K-curve is large — then forced
depth is free, and the next arm is reach1 at a deeper draw.

### Amendment 1 (2026-09-12 06:45 CDT, orchestrator): reach1 cannot force depth under coda reach "all"; one arm added

`slot-spandec-strict-reach1` read K1−K6 +0.0007 at 2,500 (4.6493 → 4.6486). The design
error is in the brief, not the build: with `tg_coda_prefix_reach: all` a token reads every
earlier slot's prefix cells directly, so a slot's chain of loop states is not the only route
to older spans and limiting the loop's reach limits nothing the coda needs. The combination
that forces depth is coda reach `prev` AND `loop_reach 1` (builder 2's own test
`test_reach_prev_plus_loop_reach_needs_depth_to_carry_three_spans` proves the law at the
fixture). Arm added: `slot-spandec-strict-prev-reach1`
(`morph/configs/tul_slot_spandec_strict_prev_reach1.yaml`), queued after `strict-prev`.
P-d is re-read on THIS arm; the original reach1 row stays as a control (reach limited, coda
unrestricted).

- **P-d′ (prev-reach1's K-curve is forced).** token K1−K6 above **0.05**: **70 %**; its
  depth-6 CE within 0.03 of `slot-spandec-strict-prev`: **55 %** (the arm has to learn to
  route through six passes what prev routes in one; at 5k that is a price). If K1−K6 is
  under 0.02, the coda's `prev` cells plus the span's own tokens already carry what the
  model uses and no reachability in the loop matters to it.

### Amendment 2 (2026-09-12 09:25 CDT, orchestrator): the oracle arm's smoke failed at build; fixed and re-queued

`slot-spandec-strict-oracle` at `a0b72c4` died in the runner's 12-step smoke inside
`SpanDecoder.decode` (`shape '[1, 1, 8, 1024]' is invalid for input of size 32768`): the
decoder added its whole 32-row position table viewed as J columns, which only works when
the oracle's budget equals the decoder's. Every unit test had set both to 8. Fixed in
`36a4cfc` (`self.pos[:J]`, a refusal when J exceeds the table, two tests at the panel's
budgets). The arm is re-queued at `36a4cfc` after `slot-spandec-strict-norecur`; no
prediction changed. P-i's rate prediction for the oracle arm (65 %) still stands.

### Amendment 3 (2026-09-12 11:2x CDT, Wolfe): add prev + reach2

`slot-spandec-strict-prev-reach1` read tokens K1−K6 +0.0163 [+0.0153, +0.0173], K3−K6
+0.0042, depth-6 CE +0.0023 [−0.0000, +0.0046] vs prev. Wolfe: "we should try reach2. I am
suspect of this method. It proves that TUL can work for sure. But the amount of blindness
here is concerning. z has to hold the history when it should hold the present next
thought that needs decoding. Our objectives are still poor." Arm added:
`slot-spandec-strict-prev-reach2` (`tul_slot_spandec_strict_prev_reach2.yaml`), queued
after the oracle arm.

- **P-d″ (reach 2).** token K1−K6 between 0.005 and 0.0163 (a wider hop per pass needs
  fewer passes, so the forced curve SHRINKS): **65 %**. Depth-6 CE better than
  `slot-spandec-strict-prev` by more than 0.003 (reach turns into value): **25 %**. K3−K6
  above 0.002: **50 %**.

## Not verified before launch

* **No GPU step of any arm here.** The card ran the arc probe chain
  (`lab/divergence/slot_z_optimize.py`, 98 % util) for the whole build window. Every cost
  number above is arithmetic; the smokes are handed back unrun.
* **Nothing says a strict model trains.** The gate proves the geometry is leak-free at init
  on a tiny CPU model. A prelude that sees only its own span is a large change to the seed,
  and the 5,000-step conjunction is unmeasured.
* **`reach1`'s position-local core layers** are a capacity change nothing here prices.
* **`oracle_z`'s lr 0.1** is untuned at the panel's shapes; only the CPU fixture says the
  trajectory descends.
* **`spandec_ce` on the h3 arm is computed at EVAL only** (it is a second full `[B, S, J, V]`
  readout), so a training run logs `spandec_ce_h` and no `spandec_ce`. The sweep column is
  unaffected.
* **The `prev` arm's dump-bin tokens** read no prefix cell at all. How many tokens that is at
  seq 1024 with `max_slots` 64 has not been measured on the real loader.
* **`plan_mode="wrong_seed"`** under strict has not been exercised; only `zero`, `shuffle`
  and `all_slots` are covered by the gate.

## Results

All seven arms at 5,000 steps, HEALTHY, every rate above the floor. Numbers: forced-depth
sweeps on 480 rows (`results/2026-09-12-strict/sweep_*_5000.json`), token-paired gaps on
501,106 tokens (`results/2026-09-12-strict/paired_gaps_5000.txt`), worth on 192 rows.

| arm | wall (s) | tok/s @200 | K1−K6 | K3−K6 | worth zero | paired gap (span − full), depth 6 |
|---|---|---|---|---|---|---|
| strict | 3,337 | 11,759 | +0.0016 [+0.0013, +0.0019] | +0.0002 | 0.1865 (= all_slots) | −0.0001 [−0.0024, +0.0024] vs spandec-mask |
| reach1 (coda reach all) | 4,140 | 9,196 | +0.0014 | +0.0003 | 0.1924 | −0.0106 vs strict |
| prev (coda reach prev) | 3,337 | 11,721 | +0.0036 | +0.0006 | 0.1665 | +0.0082 vs strict |
| prev-reach1 | 4,119 | 9,163 | +0.0163 [+0.0153, +0.0173] | +0.0042 [+0.0038, +0.0046] | 0.1477 | +0.0023 [−0.0000, +0.0046] vs prev; +0.0104 vs strict |
| prev-reach2 | 4,320 | 8,801 | +0.0127 [+0.0119, +0.0136] | +0.0028 [+0.0025, +0.0031] | 0.1552 | −0.0086 [−0.0109, −0.0063] vs prev; −0.0109 vs prev-reach1; −0.0004 vs strict |
| h3 | 4,991 | 9,168 | +0.0016 | −0.0000 | 0.1887 | −0.0093 [−0.0116, −0.0073] vs strict |
| oracle | 4,270 | 9,046 | −0.0002 | −0.0008 | 0.1214 | +0.1195 [+0.1157, +0.1237] vs strict |

Offset profiles worth reading (all in `paired_gaps_5000.txt`): prev-reach1 vs its own
depth-1 read is better at EVERY offset (−0.014 to −0.028); prev-reach2 vs strict is worse at
offsets 0-2 (+0.028, +0.027, +0.020) and better at 8+ (−0.0086); h3's gain over strict sits
at offsets 3 and beyond; the oracle's loss is at every offset (0.107 at 8+, 0.181 at 1).

Oracle instrument: the per-pass regression loss fell 0.45 → 0.11 over the run (the loop
tracked the teacher); the teacher's own ladder `oracle_z_l0..l5` is monotone at steps
200-600 and 2300-2700 and overshoots at its first step late in training (4.673 → 4.861 →
… → 4.468 at steps 4500-5000; `results/2026-09-12-strict/oracle_ladder_*.txt`).

Scores:

- P-a FALSE: strict costs 0.0000 nats against spandec-mask (predicted 0.05-0.20).
- P-b TRUE (the check): `all_slots` == `zero` = 0.1865 on strict, in every bin.
- P-c FALSE: K1−K6 0.0016.
- P-d FALSE as written: reach1 (coda reach all) K1−K6 0.0014, not above 0.05; its CE part
  (within 0.02 of strict) held, at −0.0106.
- P-d′ FALSE at the letter: prev-reach1 K1−K6 0.0163, not above 0.05; the CE part (within
  0.03 of prev) held, +0.0023.
- P-d″ TRUE on all three parts: prev-reach2 K1−K6 in [0.005, 0.0163]; CE better than prev by
  0.0086 (> 0.003); K3−K6 0.0028 (> 0.002).
- P-e FALSE: oracle K3−K6 −0.0008; the secondary (monotone ladder) FALSE late in training.
- P-f FALSE: h3's far-bin worth is unchanged; its gain is CE at offsets 3+, not worth.
- P-g FALSE: prev is +0.0082 worse than strict, not > 0.03.
- P-h TRUE: 7 of 7 HEALTHY, no sustained tripwire.
- P-i TRUE: every arm cleared 8,086 (h3 at 9,168 against a 40 % prior).

4 of 11 held.

## Verdict

Status: failure (the predictions did not hold; the panel itself ran clean).

What the panel measured, in order of weight:

1. Closing the slot-cell bypass costs nothing. Under strict geometry the loop's write is the
   whole cross-span channel (0.1865 = all_slots) at CE parity with the bypass arm. The
   channel was never the bottleneck; what is written was.
2. Depth use comes from reachability, not from the target. Every arm with coda reach "all"
   reads K1−K6 ≤ 0.0016 whatever its target (H = 1, H = 3, the oracle). The two arms that
   make old spans reachable only through the loop's chain read 0.0163 (reach 1) and 0.0127
   (reach 2), with K3−K6 0.0042 and 0.0028, at CE parity with unrestricted reach. Reach 2
   beats reach 1 at every offset and beats strict at offsets 8+ while losing at 0-2.
3. Regressing the passes onto a descent trajectory of the decoder loss is closed: the loop
   tracks the teacher and the model pays 0.12 nats at every offset for it.
4. A 3-span decoder target is a small CE win (0.009, offsets 3+) with no depth and no
   far-bin worth change.

What it does NOT say: the reach arms make z carry history, not the next thought (Wolfe's
objection, amendment 3), and nothing here shows a pass improving a prediction of the SAME
span. The 5k horizon ranks nothing between arms within 0.01 nats.

## Updated hypothesis

The passes are used when, and only when, the information the coda needs is unreachable in
one hop, and then they are used as a relay, not as refinement. No target tried (next span,
three spans, a descent trajectory) makes pass t+1 improve on pass t for the same span. Two
open hypotheses replace "the target is the lever": (a) the working state is one compressed
cell, and several mutable cells per span that re-read the prelude evidence each pass could
support refinement (the planning-cells panel, proposed); (b) the causal headroom above the
loop's exit has never been measured — the fitted-z numbers used the answer
(`lab/divergence/slot_z_causal_fit.py`, being run). The objective arms queued under
`2026-09-12-arc-objective-arms.md` (per-pass planning target, parallel coda decode) test the
target side once more with the identical-target grid as the reading.

