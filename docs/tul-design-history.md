# TUL and LXTUL: design history (2026-08 to 2026-10-14)

This is the record of every variant of the TUL idea, built from the dated filings. TUL is the looped slot design: a shared core runs on one state per span, and a coda decodes text from it. LXTUL is the later line with several cells per slot, a latent-selected loop, and a pointer head. The record covers the MORPH tests from 2026-08-16 to 2026-10-14 and the Parcae port that began in September. It exists for a from-the-top redesign. Read it to see what each try measured, which readings were later taken back, and which questions are still open. Numbers are as filed. Retractions are marked.

## How to read this

The record has eight parts, all below in this file, in order. The File column names the source file in the scratch tree. Parts 01 to 07 test the MORPH model. Part 08 tests the Parcae port in a separate repository. Its numbers use a different ruler, so do not compare them directly with MORPH numbers.

| Part | File | Dates | Scope |
| --- | --- | --- | --- |
| 01 | `01-origins-to-0831.md` | 2026-08-16 to 2026-08-31 | First slot-only loop, takeover campaign, plan-is-empty measurement, gist interface, loop ladder, l2cap retraction |
| 02 | `02-0901-to-0909.md` | 2026-09-01 to 2026-09-09 | Paid loop, warmup ramp, shipped and reversed, slot loop with gain constraint, fixed-point term, norm_match rule |
| 03 | `03-0910-to-0913.md` | 2026-09-10 to 2026-09-13 | Slot map levers, credit assignment, staged targets, span decoder, strict geometry, register, VQ, rank retraction |
| 04 | `04-0914-to-0919.md` | 2026-09-14 to 2026-09-19 | Loop diagnostics, TUL-Code and LCTUL sampling, code target, frozen reader, hop staircase, carry arms, fan4 |
| 05 | `05-0920-to-0925.md` | 2026-09-20 to 2026-09-25 | Fan diversity and selection, write-all fan, LXTUL-P ladder, LXTUL-R relay, stochastic loop, LXTUL-E, map 0.87, fixed-point weight |
| 06 | `06-0926-to-1005.md` | 2026-09-26 to 2026-10-05 | Channel width, LX rollouts and WTA, latent-selected loop, rank-only head and cell norm, LXTUL locked, carrier constant, snap |
| 07 | `07-1006-to-1014-morph.md` | 2026-10-06 to 2026-10-14 | Pointer head, DITTO, credit panel, fast recipe, set-reader fan recipe, variable cell count, speed diet |
| 08 | `08-parcae.md` | 2026-09-05 to 2026-10-14 | Parcae engineering, serial TUL panel, LXTUL port, copy cache, normalised pointer, minimal slot loop, minimal fan, speed ladder |

Each entry in a part names its source filing in the entry body. Each filing is a file under `lab/experiments/` or `.agents/notes/`, or a commit or memory note that the part cites. The master index gives the part number for each row.

Conventions used in this record:

- K1-K6 is the CE at depth 1 minus the CE at depth 6 of the same model. A positive value means later passes lower the CE. K3-K6 uses depth 3 in place of depth 1. Part 08 uses K1-K8 and K3-K16 for stock Parcae runs.
- Retractions are marked [R] in the tree and in the index. Each retraction also appears in the Retractions section, with the corrected reading.
- Dead ends are marked [x]. Current or live lines are marked [*].
- Variants are named by what they do. Config and flag names appear in backticks.

## The arc in ten steps

1. The idea started on 2026-08-16. The design said to think once per span and decode cheaply. Tokens would not enter the loop. A slot position would hold one looped state per span, and the coda would decode from it. The first slot-only arm beat its dense baseline by 0.0562 nats on val CE at 20k, but that number sat on a leak in the retention carry, so it does not stand.

2. From 08-21 to 08-28 the slot loop went unstable. The validation CE rose after step 500 to 2000 while the core took the gradient. A takeover campaign tested about twenty hypotheses. Spectral caps, sinks and truncated BPTT all failed. On 08-25 a measurement found that the coda's plan was worth 0.02 to 0.07 nats and the core was worth almost nothing at a healthy checkpoint. On 08-28 a no-slot floor beat the slot model, 4.0061 against 4.8434 at step 4250.

3. On 08-29 a gist mask with a gradient-carrying write made the coda read slot content for the first time. Mux supervision on the write gave the winning arm, 4.4047 against 4.6656 for the unmasked control. This line had no loop. The same day a loop ladder reported a depth win of 0.233 nats for a sigma-capped core. On 08-31 a leak audit showed that win came from the carry. With the carry off, the same weights read -1.1198 nats.

4. From 09-01 to 09-03 the paid loop was the only arm that earned depth. Tokens and slots passed through the core together. It earned 0.1685 nats at 5k, while the free-ride slot arms stayed at or below 0.011. The loop detonated often, so a 1000-step LR ramp was added. On 09-03 Wolfe shipped the paid loop as the recipe, then reversed the call the same evening because it looped every token and broke the think-once purpose.

5. The slot loop returned on 09-04 with a gain constraint (`model.slot_gain_lambda`, `slot_cot_clip`). The constraint held the map's typical gain near 0.89 and ended the spike train. The loop was stable but empty: it trailed a coreless floor by 0.18 nats. On 09-07 a terminal fixed-point term held the early detonation on the plain loop, 0 of 6 draws against 4 of 7 controls. On 09-09 the norm_match ternary rule lifted the plain loop's per-pass strength from 0.033 to 0.185 nats of K1-K6.

6. Between 09-10 and 09-13 every loop-side lever read flat. Credit probes showed the six passes cancelled each other. Per-pass targets were met in one step, and gradient-conditioned passes took one descent step and then held. The span decoder moved the slot channel, but a bypass through the cells let the coda skip the loop. Strict geometry closed that bypass for free. On 09-13 the rank headline of 5.76 was retracted to 13.85 on a strict ruler.

7. From 09-14 to 09-19 the team tried to make the slot carry a sampled code of the next span (TUL-Code and LCTUL). Every sample read as an unconditional draw. The code target then regressed the loop onto a frozen code, and the cell held a small span-specific signal. Unfreezing the coda moved that cell from -4.64 to +0.177 nats, so the frozen reader had been the limit. A hop staircase on the strict slot loop showed carried content decaying with hop distance, and the carry arms did not fix it.

8. From 09-20 to 10-04 the team built a fan of K latent streams per slot. The write-all fan, where each stream gets its own coda cell, was the first fan arm to beat its width partner. The LX rollouts and a coda-graded winner-take-all were the next try. A latent-selected loop with a rank-only head and a per-pass cell RMSNorm then earned depth on two seeds. It was locked on 10-04 as `morph/configs/lxtul.yaml`. In parallel, the stochastic loop, the enumerated-code line and the map 0.87 work were tested and mostly closed. The fixed-point term at weight 0.1 kept the loop earning, and at weight 1.0 it cut the earning to +0.0009.

9. From 10-05 to 10-13 the copy channel was the largest part of the gap to plain. A carrier constant formed in the first loop pass. A pointer head with an output-only copy mechanism closed most of the gap at 5k. Its repetition under sampling was the next cost. DITTO halved that repetition at +0.071 nats of clean val, and it became the winner recipe on 10-07. A set-reader span decoder kept the fan's cells distinct and became the fan recipe on 10-11. Variable cell count v1 showed the model reads well at fewer cells, but no arm gained above four cells.

10. Part 08 ran the same ideas on Parcae. The LXTUL port earned K1-K6 +0.0585 there, 2.7 times the MORPH two-seed mean. A bare one-cell loop earned nothing, so the LX machinery made the loop earn. On 10-14 a minimal one-cell slot loop and a four-cell minimal fan both earned nothing (K1-K6 -0.0001 and 0.0000). Wolfe then set the paper on the slot loop itself. Most depth claims in the record are small and rest on one or two seeds, so a replicate is the bar.

## Lineage

Plain character tree. Marks: [x] dead end, [R] retracted headline, [*] current or live.

```
Dense baseline and first slot-only loop (08-16)                         part 01
|
+-- Plain looped core, norm_match (`notul`)                         [*] the ruler for slot arms
|
+-- Slot-only loop, first TUL (`tul_short`, tokens_through_core false)  part 01
|   |
|   +-- Takeover campaign (spectral caps, sinks, bag_mean)          [x] parts 01
|   |     `-- Per-slot input embeddings (`tul.per_slot_embed`)      [x] removed 09-03
|   +-- Span-length gate (`tul_gate`)                                [x] rejected 09-03
|   +-- MUX head (`tul_v1a`)                                         [x] part 01
|   |     +-- Detached MUX (`tul_v1a2b`)                             [x] part 01
|   |     +-- Warmup, SIGReg, NTP dropout panel                      [x] part 01
|   |     `-- Staged targets (`tul.mux_stage_own_iters`)            [x] part 03
|   |           `-- Staged on every pass (`tul.mux_stage_all`)       [x] never run
|   +-- Thought Gestalt restriction (`tul.tg_restrict`)              [x] part 01
|   |     `-- Strict geometry (`tul.tg_geometry strict`)           [*] parts 03 onward
|   +-- Flow-matching planner (`tulfm_p1`)                           [x] closed 08-30
|   +-- Gist interface (`tul_gl1`)                                   [x] coreless
|         `-- Gist with MUX supervision (`tul_gl1b`)                 [x] coreless, success at filing
|
+-- Loop ladder with sigma cap (`tul_l2`)                       [R] 0.233 retracted 08-31
|     `-- l2cap recipe (`morph/configs/tul_l2.yaml`)             [R][x]
|
+-- Paid loop, tokens through the core (`tul_a2`)                  part 02
|     +-- Warmup ramp (`training.warmup 1000`)                    [*] kept in the paid line
|     `-- Shipped to base.yaml (09-03), reversed same evening      [x] rejected as not TUL (part 05)
|
+-- Slot loop returns with gain constraint (`tul_short`, 09-04)   [*] parts 02 onward
      |
      +-- Forward levers Y1, Y2 (`tul_to_mnext_y2`)                [x] stable and empty
      |     `-- Mask under constraint (`tul.tg_restrict`)           [x] reader yes, depth no
      +-- Fixed-point term (`model.core_fixed_point_lambda`)       [*] switch at weight 0.1
      +-- Norm-matched ternary rule (`training.ternary_scale_mode`)[*] recipe rule
      +-- Span decoder (`tul.spandec`)                             [*]
      |     +-- Objective arms (per-pass, coda span heads)          [x] part 03
      |     +-- Latent-z gradient, critic, core-token aux           [x] parts 03
      |     +-- Register (`tul.slot_cells`), cond4, VQ              [x] VQ mixed, part 03
      |     +-- Reach relay (`tul.loop_reach 1`)                    [R] Step 1 leaky; Step 1b tax, part 05
      |     +-- TUL-Code and LCTUL (`tul.code`)                     [x] parts 04
      |     |     +-- Code target, detached cell (`tul.code_target`) [x] ref twin, part 04
      |     |     |     `-- Unfrozen reader continuation            [*] success on one arm
      |     |     +-- Graded continuation target (`tul.code_grade`) [x] part 04
      |     |     +-- Discrete code, masked denoiser (`code_discrete`) [x] part 04
      |     |     `-- EMA target (LCTUL-J, stage 1)                [x] part 05
      |     `-- Fan of K streams (`tul.fan_k`, `tul.fan_mix`)     [*]
      |           +-- Diversity terms: cos, epi, epivol             [x] part 05
      |           +-- Select-then-commit, gate write                [x] part 05
      |           +-- Write-all fan (`tul.fan_mix all`)            [*] part 05 onward
      |           |     +-- Fixed-point off, noise, lineage, trigger [x] part 05
      |           |     +-- LX rollouts with coda WTA (`tul.code_enum_k`) [x] parts 06
      |           |     +-- Eight cells, wider write, cell packing  [x] part 06
      |           |     `-- Set-reader span decoder (fan recipe)     [*] part 07
      |           `-- Latent-selected loop (`tul.fan_loop_select`)   [*]
      |                 +-- Teacher-followed, joint and detached     [x] part 06
      |                 +-- Router-followed (`tul.fan_lsel_train_follow`) [x] pass rule, part 06
      |                 +-- Rank-only head plus cell norm            [*] LXTUL locked 10-04
      |                 |     +-- LXTUL with pointer head and DITTO  [*] winner 10-07
      |                 |     |     `-- Fast pointer recipe           [*] speed base
      |                 |     `-- Speed diet stack (cell mean 2, one-block decoder) [*] part 07
      |                 `-- Carrier constant twin, centered attention  [x] detonated, part 06
      |
      +-- Stochastic loop (LXTUL-G, LXTUL-GK)                      [x] part 05
      +-- Enumerated codes (LXTUL-E, stages 0 to 2)               [x] width earns, loop adds little
      +-- Map 0.87 and Stage 3 (fixed-point off)                   [*] fixed-point term is the switch
      `-- Hop staircase and carry arms (carry overrides `carry-sum`, `carry-gate`)          [x] part 04

Parcae port (separate repo, `lxtul/` arms, 10-06)                   [*] part 08
|
+-- Strict LXTUL port (`arm=lxtul`)                                 [*] K1-K6 +0.0585
|     +-- Copy cache (`arm=lxtul_cache`)                            [*] gap +0.0215
|     +-- Normalised pointer head (`arm=lxtul_pointer`)             [*] gap +0.0453 like-for-like
|     `-- Lean arm (`arm=lean`)                                     [x] withdrawn
|
+-- Minimal slot loop, one cell (`arm=minimal`)                    [x] K1-K6 -0.0001
      +-- Minimal fan, four cells (`arm=minimal_fan`)              [x] K1-K6 0.0000
            `-- Minimal fan with latent selection                  pending at step 3800
```

## Master index

Columns: date, variant (short name), config or flag, best depth reading (or what the part reports), verdict, part number. Rows follow the date order inside each part. Numbers are as filed.

| Date | Variant | Config or flag | Depth reading or main result | Verdict | Part |
| --- | --- | --- | --- | --- | --- |
| 2026-08-16 | Slot-only first TUL | `tul_short.yaml`, `tul.tokens_through_core: false`, `tul.activate_at: 0.0` | Slot 3.2243 against dense 3.2805 val CE at 20k (0.0562, pre-leak) | Qualified, leak sits under it | 01 |
| 2026-08-17 | Token-state dropout as trigger | `token_state_dropout` | None (claim only) | [R] withdrawn same day, n=1 | 01 |
| 2026-08-18 | Compaction-window screen | `tul.coda_token_cut`, eval arms 0 to 3 | Slots kept beat random subset by 0.009 nats (CI 0.0078 to 0.0102); stratified re-score reversed it | Screen only, archived 09-03 | 01 |
| 2026-08-18 | Merge as compute win | `tul.activate_at: never` default | No-core floor at 2.76 times dense throughput | Merged, default off | 01 |
| 2026-08-21 | Gated TUL, span-length gate | `tul_gate.yaml`, `gate_drives_depth`, `tul_halt` | Gate CE 3.3121 against 3.4175 (-0.1054); halt depth 1.00 on every eval | Rejected 09-03, single seed pair | 01 |
| 2026-08-22 | Alpha cap 1.0 | `training.ademamix_alpha_cap: 1.0` | Seed 1 died at step 4140 | Not a fix | 01 |
| 2026-08-23 | Deterministic bag mean | one-hot `bmm` in place of `index_add_` | Gradient error 3.17e-3 after fix (was 3.92e-2) | Failed reproducibility goal | 01 |
| 2026-08-24 | Per-slot input embeddings | `tul.per_slot_embed` | Time to failure 1150 to 2225 steps; CE 0.46 and 0.78 better on two seeds | Best lever, not a cure; removed 09-03 | 01 |
| 2026-08-24 | Spectral caps and sinks ledger | `spectral_penalty_cap`, `spectral_penalty_lambda`, `sink_logits`, `bptt_depth` | Spectral gap median sigma1/sigma2 1.069 to 1.132 | Most hypotheses refuted | 01 |
| 2026-08-24 | Span pooling law | span length L, `bag_mean` | Slope -0.473, -0.504, -0.527; r squared up to 0.93 | Law confirmed; "10x" claim retracted | 01 |
| 2026-08-25 | Plan-is-empty measurement | `lab/divergence/region_shapley.py`, `slot_path_worth.py` | Core 0.0007 on ce_main at step 1750; plan 0.0191 at p 0 and 0.0699 at dropout 1.0 | Standing, pre-fix | 01 |
| 2026-08-25 | HCA branch dead on slot path | HCA branch arms (H24) | One of two seeds diverged | Defect real, not the cause | 01 |
| 2026-08-25 | SCSE initial deviation | stage 1 `Delta_0` | Final CE 0.815 worse | H23 refuted | 01 |
| 2026-08-25 | MUX local head, first arm | `tul_v1a.yaml`, `mux_activate_at` | Plan worth +0.0088, +0.0189, +0.0036 at 500, 2000, 2500 | Failure, aborted by 2800 | 01 |
| 2026-08-27 | Detached MUX, beta 0.1 | `tul_v1a2b.yaml` | Loop worth +0.0058 to +0.0107 on four seeds at 3500 steps | Post-hoc, disputed status | 01 |
| 2026-08-27 | Warmup, SIGReg, NTP dropout panel | `tul_warmup.yaml`, `tul_sigreg.yaml`, `tul_ntpdrop.yaml`, `tul_w1.yaml`, `tul_w2.yaml` | None stopped the takeover | Failure | 01 |
| 2026-08-27 | Objective split | `_preclip_probe` | O2 failed; O5 matched neither mode | Refuted | 01 |
| 2026-08-27 | Thought Gestalt restriction, round 1 | `tul_tg1.yaml`, `tul.tg_restrict`, GLA `reset_mask` | Loop worth +0.0060 on ce_main against a 0.05 line | Restriction kept, loop did not earn | 01 |
| 2026-08-27 | Thought Gestalt, round 2 | `tul_tg2.yaml` | Loop worth +0.024 to +0.036, below line | Failure on loop line | 01 |
| 2026-08-28 | Specificity 2x2 | `emit_weight`, `plast_weight`, `tg_restrict` | Specificity 65.1%, 0.4%, 3.0%, 0.3% | Auxiliary losses write the plan | 01 |
| 2026-08-28 | Slot apparatus against floor | `tul_a3.yaml`, `n_core: 0` | No-slot 4.0061 against best slot 4.8434 at step 4250 | Slot apparatus negative | 01 |
| 2026-08-28 | Flow-matching planner | `tulfm_p1*`, `tul_fm1.yaml`, `tul_fm1_cw.yaml`, `tul_fm2.yaml` | Plan worth 0.0000; oracle prefix -0.0000 [-0.0001, +0.0001] | Closed (also l2cap revival failed 08-30) | 01 |
| 2026-08-28 | Round two slot seeds and TG panel | `tul.slot_seed` bag_mean, e_slot, boundary; `tul_tg3b`, `tul_tg4a`, `tul_tg4b` | Soft restriction held 4 of 4 predictions | Mixed; auxiliary losses explain most | 01 |
| 2026-08-29 | Gist mask, first arm | `tul_gl1.yaml` | Shuffle cost 0.06, zero 0.15 to 0.18, wrong value 0.33 to 0.43; price 0.34 | Half pass: slots carry content, mask too costly | 01 |
| 2026-08-29 | Gist with MUX supervision | `tul_gl1b.yaml` | Step 4250: 4.4047 against unmasked ruler 4.6656 (0.26 better) | Success at filing | 01 |
| 2026-08-29 | Gist with warmup then mask | `tul_gl1c.yaml` | 4.5974 at step 4250 | Success at filing | 01 |
| 2026-08-29 | Gist composition | `tul_gl1bc.yaml`, `tul_gl1_ctrl.yaml` | Result not separately stated in part 01 | Line 2 replication: "MUX is the CE lever" | 01 |
| 2026-08-29 | Loop ladder, sigma cap 1.5 | `tul_l2.yaml` | K1-K6 0.233 at 4500 steps | [R] retracted 08-31 (carry leak) | 01 |
| 2026-08-29 | Loop ladder, full BPTT and DB shape | `tul_l1.yaml`, `tul_l3.yaml` with `tul.db_loop`, `training.optimizer=adamw` | Not stated as separate numbers in part 01 | Part of the ladder | 01 |
| 2026-08-30 | l2cap winning recipe | `morph/configs/tul_l2.yaml` | Note claimed 0.233 as "the ONE recipe whose loop earns depth" | [R] retracted 08-31 | 01 |
| 2026-08-30 | Gate ladder recurrence gate | `tul_g1.yaml`, `tul_g2.yaml`, `recur_gate: grt` | Gate without cap +0.0002; with cap -0.0002 | Both fail, depth flat | 01 |
| 2026-08-30 | DB fix pair and cond arm | `tul_dbfix.yaml`, `tul_db_cond.yaml`, `tul_l2cap_cond.yaml` | l2cap cond K1-K6 0.0127 (against retracted 0.233) | Both fail; comparator retracted | 01 |
| 2026-08-30 | Interleave | `tul_ilv50.yaml` | Not run to a verdict | Cancelled | 01 |
| 2026-08-30 | Wake arms, DB init then AdamW | `tul_l3wake.yaml`, `tul_l3wake_cap.yaml`, `tul_l2trunc.yaml` | Wake pure CE 4.2044; truncated K1-K6 0.0085 | DB init is an inert basin | 01 |
| 2026-08-31 | Carry leak audit | `model.retention_carry`, carry-off eval | Carry off: K1 4.622, K6 5.742, K1-K6 -1.1198 (claim +0.2328) | [R] Retracted; leak 3.85 nats at 30k | 01 |
| 2026-08-31 | Causal re-train of l2cap | `tul-l2nc` | +0.0171, flat | Honest rebaseline, flat | 01 |
| 2026-08-31 | Loop-killer bisect | `notul_bg0c0.yaml` (GLA off, cap off) | K1-K6 0.220 at 4500 (K3-K6 0.017), saturates by K4 | First honest plain K-curve | 01 |
| 2026-08-31 | Causal no-TUL loop worth | no-TUL causal config, two geometries | K3-K6 0.015 | Flat in both | 01 |
| 2026-08-31 | TUL against no-TUL at 20k | `tul-20k`, `notul-20k` | Core-axis K1-K6 TUL 0.015, no-TUL 0.207; token CE gap 0.357 | TUL stays merged, default off | 01 |
| 2026-08-31 | GLA write alignment | `write_shift`, `notul_bws` | First prediction P-A1 false | Failure | 01 |
| 2026-09-01 | Span-pooled compressed branch | `tul.tg_span_comp`, `tul_sac` | Eval worth 0.187; recovered 0 of 0.357 gap; earning 0.015 to 0.002 | Rejected | 02 |
| 2026-09-01 | Learned gated span pooling | `tul.tg_span_gate`, `tul_sacg` | +0.031 on last five evals, +0.034 on K6 (inside noise) | Lane closed | 02 |
| 2026-09-01 | Mask surgery on trained plain model | eval only on `notul-20k` | Masking cost 0.487 against realized gap 0.357 | Kept as decomposition | 02 |
| 2026-09-01 | Bound superposition seed | `tul.bound_seed`, `tul_w2` | Rank 38.3 without E_slot against 3.5 bag mean; collapses to 1.1 to 3.1 with E_slot | Binding works; E_slot dominates | 02 |
| 2026-09-01 | Write-side ladder, free-ride seeds | `tul_r0`, `tul_w1`, `tul_w2` | K1-K6 0.0113 (R0), 0.0007 (W1), 0.0019 (W2) | No seed moves earning | 02 |
| 2026-09-01 | Paid loop, tokens through core | `tul_a2.yaml`, `tul.tokens_through_core: true` | K1-K6 0.1685 at 5k; paid no-slot twin 0.1937 | Success: the loop earns when paid | 02 |
| 2026-09-02 | Future-leak probe, paid loop | eval only | Leak 0 nats in bf16; earning 0.12 at 2500, 0.17 at 5000 | Paid loop is causal | 02 |
| 2026-09-02 | Core Jacobian ladder | `morph/training/core_jacobian.py` probe | Healthy gain 1.05 (2500), 1.13 (5000); sick gain 9 to 159 | Failure by protocol; reading clear | 02 |
| 2026-09-02 | Ternary off on paid draws | `training.ternary=false` | 3 of 3 healthy against about 70% base detonation | Success at rate level | 02 |
| 2026-09-02 | M2G onset capture | frame-level capture, `tul_a2` | No stale-m2 signature | Failure of predictions; ternary cusp vault | 02 |
| 2026-09-02 | Ternary scale EMA | `training.ternary_scale_ema_beta=0.99` | Detonated 2 of 3; healthy draw cost 0.30 nats | Rejected | 02 |
| 2026-09-02 | Ternary scale freeze | gamma pinned at step 0 | Detonated 3 of 3; val flat near 7.4 | Gamma not exonerated | 02 |
| 2026-09-02 | Restricted paid loop | `tul_a2s`, `tul.tg_restrict: true` | Detonated in both draws before step 2000 | Cannot train at these flags | 02 |
| 2026-09-02 | Capped paid loop (planned) | `tul_a2c`, `training.spectral_project_cap=1.5` | Never run | Contingency did not fire | 02 |
| 2026-09-02 | GLA retention under ramp | `model.retention=true`, `training.warmup=1000` | 0 of 3 detonated; earning 0.049 against 0.047 | Inert under the ramp | 02 |
| 2026-09-02 | Warmup 1000 on paid loop | `training.warmup: 1000`, `tul_a2` | 2500 steps: 0 of 3 detonated; earning 0.046 to 0.049 against 0.121 flat | Ramp adopted; earning drop real | 02 |
| 2026-09-02 | Warmup paid loop, 5k earning | `wandb.name=tul-a2-wu5k` | Earning 0.058 against 0.1685 flat | Failure on earning prediction | 02 |
| 2026-09-02 | 20k pair on the ramp | `tul-a2-20k-wu` against `notul-20k-wu` | Depth 6: paid 3.4701, plain 3.4486; earning 0.100 at 20k (index says 0.104) | Paid loop does not beat plain at 20k | 02 |
| 2026-09-03 | Ship paid loop as base | `tul.activate_at: ${training.tst_ratio}`, `tokens_through_core: true` | 80-step smoke 7.2k tok/s against 49.6k without TUL | Reversed the same evening | 02 |
| 2026-09-03 | Think-once round-1 panel | `tul_to_a3`, `tul_to_a1`, `tul_to_mown`, `tul_to_mnext`, `tul_to_mnext_mask`, `tul_to_cond4`, `tul_to_coda8` | Six of eight draws tripped the tripwire; memory draw clean | Failure | 02 |
| 2026-09-03 | Onset capture, forecast arm | `tul_cap_c0`, `tul_cap_c1`, `tul_to_mnext` | Typical gain drifts 0.87 toward 1.00; forward never explodes | Success on the question | 02 |
| 2026-09-04 | Slot loop back with gain constraint | `tul_short.yaml`, `model.slot_gain_lambda: 100`, `slot_gain_target: 0.9`, `slot_gain_eps: 0.02`, `model.slot_cot_clip: 4.0` | 5000 steps, zero spike steps in one draw; fp32 gain 0.887 to 0.897 | Shipped | 02 |
| 2026-09-04 | Forward levers Y1 and Y2 | `tul_to_mnext_y1` (`slot_state_renorm`), `tul_to_mnext_y2` | Y2 forecast K1-K6 +0.0135; token K1-K6 at or below 0.0006 | Stable and empty | 02 |
| 2026-09-04 | Backward clip through time | `tul_to_mnext_ctt`, `slot_cot_clip: 4.0` | Weight-path gradient still grew 800 times; gain 0.89 to 8.7 | Retired as cure, kept as instrument | 02 |
| 2026-09-04 | Where depth earns | eval on paid and plain loops | Paid loop earns 2.5 times plain; spread even over offsets | Hard-positions idea wrong | 02 |
| 2026-09-04 | Gain-target dial | `tul_to_mnext_y2_g95`, `tul_to_mnext_y2_g98` | Token val CE up 0.05 (at 0.95) and 0.15 (at 0.98) | Failure: map slowed, loss unmoved | 02 |
| 2026-09-04 | Per-iteration operator | `tul_to_mnext_y2_iter`, `tul_to_mnext_y2_iter_all`, `tul.core_stage_cond=iter` | K1-K6 0.0087 against shared 0.0135 | Branch closed | 02 |
| 2026-09-04 | Mask under constraint | `tul_to_mnext_y2_mask`, `tul.tg_restrict: true` | Token K1-K6 +0.0209; mask costs 0.13 nats of token CE | Reader yes, depth no | 02 |
| 2026-09-04 | Huginn external ruler | no MORPH config | Earns 0.57 between iterations 3 and 6; 0.21 between 6 and 16 | Web text is not depth-flat | 02 |
| 2026-09-07 | Terminal fixed-point term, warmup 0 | `model.core_fixed_point_lambda: 1.0`, `notul_fp_wu0.yaml` | 0 of 6 detonated against 4 of 7 controls (Fisher p 0.049) | Success; shipped | 02 |
| 2026-09-07 | Fixed-point term under ramp | `notul_fp.yaml` | CE +0.0009 [-0.0022, +0.0040]; K1-K6 0.022 against 0.037 | Free on plain loop | 02 |
| 2026-09-07 | Deep recurrence draw, mean 16 | `notul.yaml`, `model.cfg.mean_depth: 16` | K3-K6 0.277 against 0.0016 at mean 6; depth-16 CE 0.104 worse | Success: K measures dependence, not value | 02 |
| 2026-09-07 | Block loop at mean 16 | `slot_max_depth`, slot loop | Entry norm 485 to 1.5e7 in one iteration; rank 181 to 9 | Failure; saturation and norm growth one mechanism | 02 |
| 2026-09-07 | Multi-token coda, lookahead heads | `notul_mtp4`, `notul_deep16_mtp4` | Cost 0.34 nats of next-token CE at matched steps | Rejected | 02 |
| 2026-09-07 | Widen the carry, rho below 1 | `notul_carry_all`, `notul_carry_ctx_wu0` | No effect on early detonation | Failure | 02 |
| 2026-09-07 | Fixed depth 12, mask arm | `tul_k12_mnext_mask`, `tul_k12_mnext_mask_mtp4`, `tul_k12_a1`, `slot_depth_fixed` | Entered power-iteration face; detonated at 1446 | Rejected run (wrong method) | 02 |
| 2026-09-07 | Mean-12 draw panel | `tul_m12_a1`, `tul_m12_mnext`, `tul_m12_mnext_mask`, `tul_m12_mnext_mask_mtp4` | Masked token loop share 0.021 to 0.049; end point matched mean 6 (-0.0008) | Failure on main predictions | 02 |
| 2026-09-07 | Expansive dial, gain near 1 | `tul_m12_mask_g102`, `tul_m12_mask_g102_rn`, `model.slot_state_renorm` | K3-K6 0.0007; spikes at 3877 and 4639 | Gain target is a dial on share, not depth | 02 |
| 2026-09-07 | Olympiad uniform mix | `tul_oly_a1`, `tul_oly_mask`, `tul_oly_mnext`, `notul_oly.yaml` | Rejected before any data (stage sum overrode steps) | Rejected run | 02 |
| 2026-09-08 | Olympiad curriculum panel | same arms, curriculum mix | Masked arm 0.18 behind plain on clean math; A1 ties plain to 0.000 +/- 0.010 | Failure; holdout contaminated (41% verbatim) | 02 |
| 2026-09-08 | Sudoku depth grid | `tul_sud_a1`, `tul_sud_mask`, `tul_sud_mnext`, `notul_sud` | Plain 0.938 accuracy, flat from T 2 to 16; masked 0.155 behind at trained depth | Success on question; lane closed | 02 |
| 2026-09-08 | Slot write width sweep | `tul.prefix_k` 2, 4, 8 (absmean) | K1-K6 +0.0243, +0.0314, +0.0486 at 5k | No filed verdict; later called absmean artefact | 02 |
| 2026-09-08 | Eager hinge reads high | `model.slot_gain_eps` 0.02, `tul_oly_mask` | Eager 0.94 (sd 0.014) against fused 0.87 (sd 0.002) | Proposed fix; about 0.07 bias | 02 |
| 2026-09-07 | Truncated BPTT silences shallow samples | `bptt_depth` below max depth | Silent fraction 27% plain, 57% slot at mean 16 | Proposed fix, not built | 02 |
| 2026-09-09 | Coda reads the thought | `tul_slot_unpack_norm_match` | Token K1-K6 flat (0.0006); reader reads z at 0.811 | Failure on depth prediction | 02 |
| 2026-09-09 | Norm-matched ternary rule | `training.ternary_scale_mode: norm_match`, `morph/model/ternary_rule.py` | Plain K1-K6 0.185 [0.182, 0.188] at 5k; absmean 0.033 | Shipped as recipe rule | 02 |
| 2026-09-09 | Norm-match 20k and horizon | `notul_norm_match_20k`, `notul_horizon_ternary_25k` | K1-K6 0.1702 at 20k; final val 3.5037 against absmean 3.4575 | Failure on one prediction; 5k discrepancy open | 02 |
| 2026-09-09 | Paid loop under norm_match | `tul_norm_match`, `tul_absmean`, `tul.activate_at 0.0` | K1-K6 +0.1200 at 5k; paid minus plain at depth 6 +0.1252 | Side result; Wolfe: "a hallucinated TUL" | 02 |
| 2026-09-09 | Slot loop under norm_match | `tul_slot_loop_norm_match`, `tul_slot_mux_norm_match`, `tul_slot_loop_absmean`, `tul_slot_mux_absmean` | Forecast K1-K6 0.0067 against absmean twin 0.0069; gain 0.868 to 0.887 | Failure on loop-earning predictions | 02 |
| 2026-09-09 | Parcae-style loop entry | `notul_parcae_entry`, `notul_e18`, `notul_depth1.yaml` | Earns 0.033 past pass 1 against Parcae 0.29; depth-1 within 0.02 of looped | Entry is not the lobotomy | 02 |
| 2026-09-09 | Depth candidates panel | `notul_depthcand_carry_lr20x`, `notul_depthcand_dense_core`, `notul_depthcand_draw8_bptt4` | Parcae schedule K3-K8 0.002; bf16 core moved past-pass-1 value 0.033 to 0.168 | Ternary core limits contribution; led to norm_match | 02 |
| 2026-09-09 | Block density panel | `notul_density_half`, `notul_density_quarter` | Quarter density K3-K6 under 0.006 | Failure: dependence without computation | 02 |
| 2026-09-10 | Slot map levers | `tul` unpack base, `core_gain_lambda` 0, `core_fixed_point_lambda` 0, `core_state_init: noise`, `slot_depth_fixed` 6 | Fixed depth 6 K1-K6 +0.1168 (CE@6 +0.052 worse); other two +0.0030, +0.0019 | Failure; stability terms not the pin | 03 |
| 2026-09-10 | Reread slot cross-attention | `tul.reread`, `reread_heads` 8, `reread_scope: causal` | Token K1-K6 +0.0003 | Read used, changes nothing coda reads | 03 |
| 2026-09-10 | Noise entry on forecast loss | `tul_slot_mnext_noise_entry` | Token K1-K6 +0.0004; mux_local +0.0091 | Failure: entry coda-independent | 03 |
| 2026-09-10 | Progressive loss | `tul.progressive_p: 0.5` | Token K1-K6 +0.0006; cancellation 0.497 | Failure, flat | 03 |
| 2026-09-10 | Per-pass LoRA | `tul.pass_lora_rank: 32` | Token K1-K6 -0.0000; +6.29M params | Failure; deltas moved, loop did not | 03 |
| 2026-09-10 | MUX on every pass | `tul.mux_every_pass: true` | Depth 1 6.7698 equals ruler depth 6 6.7724 | Failure; trains depth independence | 03 |
| 2026-09-10 | Staged targets, middle pass | `slot-mnext-staged`, `tul.mux_stage_own_iters: 3` | Token K1-K6 +0.0020; mux_local_next +0.0670; CE@6 4.2483 (0.081 better than ruler) | Success by count; K gain is depth 1 worse | 03 |
| 2026-09-10 | Staged targets, every pass | `slot-mnext-staged-all`, `tul.mux_stage_all: true` | No measured result; a smoke died at OOM | Pending, unclear | 03 |
| 2026-09-10 | Staged plus mask | `slot-mnext-staged-mask`, `tul.tg_restrict: true` | Token K1-K6 +0.0033; CE@6 4.4373 (0.108 behind ruler) | Failure; highest-confidence prediction false | 03 |
| 2026-09-10 | Whole-carrier reader | `slot-mnext-staged-fullread`, `tul.mux_readout: full` | Token K1-K6 +0.0014; exit forecast 6.7543 | Failure; F2 real on paper, inert | 03 |
| 2026-09-10 | Mask on the ruler, norm_match | `slot-mux-mask-norm-match`, `tg_restrict: true` | Token K1-K6 +0.0009; CE@6 4.4194 | Failure; retracts absmean mask K-curves | 03 |
| 2026-09-10 | No MUX, masked | `slot-loop-mask-norm-match` | Token CE identical at depths 1 to 6 (4.3792); core gradient 0.121 against prelude 21.9 | Failure; MUX load-bearing | 03 |
| 2026-09-10 | Fixed-point term off | `tul_slot_mux_fixed_point_off`, `core_fixed_point_lambda: 0.0` | Token K1-K6 +0.0005; max preclip ratio 1441 against ruler 78.7 | Failure; term is a state bound | 03 |
| 2026-09-10 | Parcae core swap | `model.core_impl: parcae`, `injection_channels: all`, `injection_B: true` | Token K1-K6 +0.0001; mux_local +0.0019 (ruler +0.0067); 23,122 against 12,429 tok/s | Success: core exonerated | 03 |
| 2026-09-10 | Gradient-conditioned passes | `slot-mnext-gradpass`, `tul.grad_pass: true`, `grad_pass_scale 0.1` | Token K1-K6 +0.0014; mux_local +0.0236; CE@6 4.2772 | Filed failure; loop takes one descent step then holds | 03 |
| 2026-09-10 | Staged at 20k | `tul_slot_mnext_staged_20k`, `training.steps 20000` | Token-paired gap to plain 0.1919 (5k), 0.1596 (10k), 0.1628, 0.1609 (20k) | Success, 9 of 10 predictions | 03 |
| 2026-09-10 | Prefix width 4 | `tul_slot_mux_prefix4_norm_match`, `tul.prefix_k: 4` | Token K1-K6 +0.0001; CE@6 4.2904 (0.039 better than ruler) | Success by count; cells carry seed, not loop | 03 |
| 2026-09-10 | Bag-mean seed | `tul_slot_mux_bagmean_norm_match`, `tul.slot_seed: bag_mean` | Token K1-K6 +0.0005; CE@6 4.3536 (0.025 behind) | Seed axis closed under shipped rule | 03 |
| 2026-09-10 | HCA branch alive | `tul_slot_mux_hca_fix`, `model.core_hca_compress_ratio: 16` | Token K1-K6 +0.0004; CE@6 4.2986 (0.030 better) | Success; shipped 09-11 | 03 |
| 2026-09-10 | Toy slot loop | `lab/toy_slot_loop` (not a MORPH config) | Stuck cancellation 0.946 against solved 0.749; loop +0.514 strict against +0.106 permissive | Proposed; 7 of 14 frozen predictions failed | 03 |
| 2026-09-10 | TUL on a Parcae core knob | `model.core_impl: morph` or `parcae` | Build and tests only (1055 passed, 1 skipped, 1 xfailed) | Unresolved | 03 |
| 2026-09-11 | Cross-span budget | `model.span_mask: span`, `budget_root.yaml`, `core_impl: parcae` | Gap +0.3994 [0.3838, 0.4162] at depth 6; offset 0 0.2826; offset 1 0.9580 | Measurement; the arc's first ceiling | 03 |
| 2026-09-11 | Span decoder | `tul.spandec`, `tul.spandec_layers` 2, `tul.span_cap` 32, `tul.mux_detach_head`, `tul.slot_chain`, `plan_mode="all_slots"` | CE gap to mux-mask -0.0720 [-0.0748, -0.0698]; token K1-K6 0.0007 | Success by protocol, 8 of 10 | 03 |
| 2026-09-12 | Strict slot geometry | `tul.tg_geometry: strict`, `tul.tg_coda_prefix_reach`, `tul.loop_reach`, `tul.oracle_z`, `tul.spandec_horizon` | Strict K1-K6 +0.0016; paired CE vs spandec-mask -0.0001; prev-reach1 +0.0163 | Failure by protocol, 4 of 11 held; bypass closed for free | 03 |
| 2026-09-12 | Latent-z gradient energies | `tul.grad_pass_energy: recon`, `disc`, `tul.pass_residual_lambda 0.01`, `tul.slot_depth_fixed: 1` | Recon K1-K6 +0.0007 (CE +0.0495, 1.78 times wall); disc +0.0019 | Failure; one pass met the target | 03 |
| 2026-09-12 | Math under norm_match | `tul_oly_spandec_strict`, `tul_sud_spandec_strict`, `notul_oly`, `notul_sud` | Slot minus plain at depth 6: Sudoku +0.2502, Olympiad +0.2161; Sudoku solve 0.0% against plain 0.8% | Failure; math lane closed for this design | 03 |
| 2026-09-12 | Objective arms | `tul.spandec_per_pass`, `tul.coda_span_heads` 8 and 32, `tul.coda_span_source` | Per-pass K1-K6 +0.0008; codaspan +0.0012; codaspan32 +0.0017 | Failure; 8 of 12 false by letter; grid invalid, corrected | 03 |
| 2026-09-12 | Core-token auxiliary and critic | `tul.core_token_aux`, `tul.grad_pass_energy: critic`, `tul.critic_eps 0.1` | Coretok CE -0.0684 vs strict; critic K1-K6 +0.0012; critic loss 0.689 | Failure; "under-used, not under-trained" | 03 |
| 2026-09-12 | Causal instruments | `slot_z_causal_fit.py`, `slot_depth_isolation.py`, `spandec_horizon_grid.py`, forward arg `slot_depths` | Causal fit on real span +0.252, worse than loop (CE 4.0933 vs 3.8416) | Retracts headroom reading | 03 |
| 2026-09-13 | Information view | `lab/theory/tul_information`, `tul.loop_reads_tokens` | 10% perturbation of z moves coda loss at most 0.0005 nats; one pass 0.001 to 0.003 | Adopted as theory | 03 |
| 2026-09-13 | Second-opinion rank review | `tul.slot_cells`, `n_coda: 1` | Headline 5.76 rank was a bare-front reading; collapse about 13 directions in about 50 slots | Retraction same day | 03 |
| 2026-09-13 | Rank anatomy | `lab/experiments/results/2026-09-13-rank-anatomy/` | Rank 5.7598 retracted to 13.85; cosine 0.7104 to 0.520 | [R] Corrected baseline | 03 |
| 2026-09-13 | Rank levers | `tul.center_exit`, `tul.row_contrast_lambda 0.1`, `b_center`, `W_contrast` | No run recorded; gate built (53 tests) | Superseded in practice | 03 |
| 2026-09-13 | Thought register | `tul.slot_cells`, `slot_cell_init`, `prefix_k`; `slot-register-m4`, `slot-register-m4-sameinit`, `slot-register-m8` | m4 token K1-K6 +0.00197; paired CE m4 vs strict -0.0217; row rank 10.57 | Failure; 5 of 7 scorable false; cells collapse | 03 |
| 2026-09-13 | cond4 reader | `tul.cond_layers: 4`, `slot_cells`, `tg_relation` | Strict-cond4 K1-K6 +0.00026; register-cond4 +0.00062; coda worth 0.181 vs 0.194 | Failure; reader depth not the limit | 03 |
| 2026-09-13 | Register reader and target offset | `tul.spandec_reads_cells`, `tul.spandec_target_offset`; `slot-spandec-strict-off2`, `-off3` | Off2 K1-K6 +0.0023 (CE -0.0035); off3 +0.0009 (CE -0.0008) | Failure; P-7 bar +0.02 missed | 03 |
| 2026-09-13 | Ultralight coda | `n_prelude: 7`, `n_core: 6`, `n_coda: 1` | K1-K6 +0.00169 against partner +0.00161; CE +0.0073 worse | Failure, 4 of 4 false | 03 |
| 2026-09-13 | Trajectory prefix | `tul.prefix_source`: trajectory, exit_repeat, entry_exit; `prefix_k: 6` | Traj K1-K6 +0.0342 (later retracted); trajrep +0.0019; entryexit +0.0014 | Failure; every CE win is prefix width | 03 |
| 2026-09-13 | Loop reads the tokens | `tul.loop_reads_tokens`, `core_token_aux_probe.py --depths` | Aux K1-K6 +0.0102 at 480 rows; slot loop 0.0005 | Failure, Q-1 false (bar 0.05) | 03 |
| 2026-09-13 | Plain-then-TUL bootstrap | `training.init_from`, `training.data_skip_batches: 5000`, `training.resume_plain_to_tul`, `tul_slot_strict_bootstrap.yaml` | K1-K6 +0.0019 [+0.0016, +0.0022]; paired CE -0.2164 | Failure; seed improves, loop does not | 03 |
| 2026-09-13 | Discrete thought VQ | `tul.vq_codes`, `vq_codebook` 512, `vq_groups`, `vq_beta`; `tul_slot_spandec_strict_vq8`, `_vq4` | vq8 K1-K6 +0.0015, CE vs strict +0.0250; vq4 +0.0017, CE +0.0577 | Mixed; code capacity is a live lever | 03 |
| 2026-09-13 | TUL as memory | `plain_span_local.yaml`, `model.span_mask: span`, `lab/divergence/memory_cost_table.py` | Slot 10.59 block-passes and 64.4 keys per token; plain depth 1 14.00 and 512.5 | Instrument adopted; matched control unbuilt | 03 |
| 2026-09-13 | Depth ladder, first filing | `notul_norm_match_20k_d1`, `_d2`, `_d3`, `_d3fixed`, `norm-match-20k-p3`, `-p4`, `model.depth_fixed` | Scored in part 04 | Filed as failure with ship call | 03 |
| 2026-09-13 | Parked loop directions | `tul.traj_attn`, `tul.realign`, latent feedback, LoopMTP | No measured result; items stopped mid-build | Parked on Wolfe's call | 03 |
| 2026-09-14 | Loop diagnostics panel | `lab/divergence/aa_score.py`, `jacobian_gap_vs_k.py`, `sink_mass_per_pass.py`, `basin_map.py` | AA(swap) strict 0.8644, plain 0.9777; spectral ratio 1.052 against 1.105 | Three of four predictions failed; sink mass the one agreement | 04 |
| 2026-09-14 | Trajectory prefix, width against content | `traj`, `trajrep`, `entryexit`, `prefix_k` 6 | Depth-6 CE 4.3203, 4.3208, 4.3196 against ruler 4.3474 | Failure; K +0.0342 was pad artefact | 04 |
| 2026-09-14 | Horizon-indexed passes (LoopMTP port) | `tul_slot_spandec_strict_horizon`, `tul_slot_spandec_strict_fixed6`, `pass_readout: gated`, `horizon_weight`, `horizon_free_first` | Horizon K1-K6 +0.0040; paired depth-6 CE -0.0001 | Failure; horizon targets degenerate on web text | 04 |
| 2026-09-14 | LoopMTP on token loop | `notul_norm_match_20k_loopmtp.yaml`, `loopmtp_weight 0.05` | Pass cosine 0.95 to 0.94; consecutive 0.949 to 0.996; no K or CE filed | Open, unfiled | 04 |
| 2026-09-14 | TUL-Code first panel | `tul_code.yaml`, `tul_code_seeddetach.yaml`, `tul_code_nophase3.yaml`, `code_rollout_steps`, `code_noise 0.5`, `code_norm: rms` | Sampled ce_k1 +0.705 [+0.702, +0.709] behind ruler; rates 33.1k against 10.4k tok/s | 5k predictions failed; sampled code unconditional | 04 |
| 2026-09-14 | TUL-Code at 20k, frozen-E rollout | `tul-code-20k`, `tul-code-rollout1-20k`, `code_rollout_p` 1.0 | One-draw gap +0.626 (p 0.5), +0.359 (p 1.0); topic thread 0 of 10 cuts (ruler 4 of 10) | Failure; one of eight predictions held | 04 |
| 2026-09-15 | K-sample marginal instrument | `lab/divergence/code_marginal_sweep.py` | Marginal k16 minus k1 -0.004 [-0.006, -0.003] at 5k | Instrument; became standard | 04 |
| 2026-09-15 | Renorm fork | `tul_code_renorm.yaml`, resumed at step 10000 | One-draw gap at 20k +0.659; fork minus parent +0.033 | Success as correctness fix | 04 |
| 2026-09-15 | Conditioned thinker, CFG and predictability pressure | `tul_code_cfg.yaml`, `code_cfg_drop 0.1`, `code_cfg_scale 2.0`; `tul_code_jepa.yaml`, `code_target_lambda 0.1` | One draw +0.648 and +0.654; marginal k16 minus k1 -0.005 and -0.017; ce_tf 0.32 | Failure; both rejected | 04 |
| 2026-09-15 | Thinker only, frozen decoder 50k | `tul_code_thinker.yaml`, batch 12 | Marginal k16 minus k1 -1.88 [-2.00, -1.73]; one-draw CE 13.66 | Failure on thresholds | 04 |
| 2026-09-15 | Explorative modeling, best of K draws | `tul_code_xm.yaml`, K 4; `tul_code_xmc` (queued), `tul_code_xmn.yaml` (planned) | One-draw gap +0.627; marginal k16 minus k1 +0.016 | Failure; cannot help unconditional draws | 04 |
| 2026-09-16 | LCTUL-D, discrete masked denoiser | `tul_code_d.yaml`, `code_discrete: true`, 8 symbols of 512 | ce_tf 3.960 against ruler 4.378 at 5k; k1 minus k8 -0.162; past share 0.02% | Failure; code rate too large to predict | 04 |
| 2026-09-16 | LCTUL-D plan arm, semantic score | `tul_code_dplan.yaml`, 4 symbols of 64 (24 bits) | OWN@4 minus SHUF +0.008 [-0.006, +0.023]; OWN@4 against ruler -0.030 | Failure; 4 of 5 semantic predictions false | 04 |
| 2026-09-16 | LaDiR-style chain | `tul_code_ladir_tf.yaml`, `tul_code_ladir_ro.yaml`, `tul_code_vae.yaml` | Marginal k16 minus k1 +0.319; own minus shuffled -0.006 at 40k | Failure; frozen coda reads code, thinker does not | 04 |
| 2026-09-16 | LeJEPA on the code | `tul_code_lejepa.yaml`, SIGReg guard | Rank minimum 40.4; one-draw gap +0.681 at k 6; marginal -0.0006 at 20k | Failure; target moved to lower loss | 04 |
| 2026-09-16 | Thinker phase 3 and 50k continuations (planned) | `tul_code_thinker_p3.yaml`, `tul_code_50k` | Never run | Open | 04 |
| 2026-09-17 | Code target, detached cell | `tul_code_target.yaml`, `code_target_detach: true` | Own minus shuffled +0.0230 [+0.0051, +0.0421]; exit cosine 0.160 at 20k | First positive own-against-foreign reading; reader is the open confound | 04 |
| 2026-09-17 | Code target, CE companion | `tul_code_target_ce.yaml`, `code_target_detach: false` | Stopped at step 15000 | Not scored | 04 |
| 2026-09-17 | Code target, progressive loss | `tul_code_target_prog.yaml`, `progressive_p` | K1-K6 -0.1173 (wrong direction); cell cosine l6 minus l1 +0.0073 | Failure on depth | 04 |
| 2026-09-17 | Code-only arm, first draw | `tul_code_only.yaml`, `code_target_skip_coda: true`, `training.train_only: []` | Own cosine 0.61 rising to 0.99 by step 2500 to 3000 | [R] Void; target drifted with prelude | 04 |
| 2026-09-17 | Frozen reference twin | `tul.code_target_ref: true`, `tul_code_only_nce.yaml` (planned InfoNCE twin) | Own minus shuffled 0.138 and 0.110; effective rank 17.4 and 17.6 against 77.5 and 70.2 | Failure on idea; K values void | 04 |
| 2026-09-17 | Graded continuation target | `tul_code_grade.yaml`, `tul_code_grade_l2.yaml`, `code_grade_k`, `code_grade_loss: pref`, `code_grade_grader: coda_past` | Clean margin best minus worst -0.004 (after sentinel fix; was +0.0052); truth rank 0.878 | Failure; text ranking is not code ranking | 04 |
| 2026-09-17 | Unfrozen reader continuation | `tul_code_target_uf.yaml` (from 20k to 30k) | Cell worth -4.64 frozen, +0.177 adapted; own minus shuffled +0.0225 [+0.0021, +0.0422] | Success on the one arm; reader was the limit | 04 |
| 2026-09-18 | Hop-distance probe, first pass | `tul_slot_spandec_strict_prev_reach1.yaml`, `hop_distance_probe.py`, `--planted`, `--cut-after 1` | prev-reach1 per hop h3 +0.0640, h4 +0.0535, h5 +0.0288, h6 -0.0203 | Failure by letter; staircase real | 04 |
| 2026-09-18 | Frozen coda K-curve is genericity | `lab/divergence/core_depth_sweep.py`, `tests/test_depth_sweep_frozen_reader_warning.py` | Detached arm cell reads better by +0.032 from blandness | Implemented warning | 04 |
| 2026-09-18 | Sample-oracle gate, frozen reader | `lab/divergence/sample_oracle_probe.py`, N 16, entry sigma 0.3 and 1.0 | Gain(16) depth 1 +0.0117 (sigma 0.3), +0.0251 (sigma 1.0); cos_exit 0.962 and 0.720 | Failure; qualified by Wolfe's 09-19 review | 04 |
| 2026-09-18 | Toy set-carry, deferred commitment | toy `eliminate`, two-pass chain, no YAML | Solved seeds commit early (entropy 0.080 to 0.424 of 0.693); membership probe 1.000 | Falsifier fired; alive set carried | 04 |
| 2026-09-19 | Toy six-hop carry | toy `eliminate6`, four passes, six candidates | Random-init hop decay 0.668 at hop 4 (d96); trained cells 0.999 to 1.000; fitted reader 2.002 against chance 2.079 | Success on contrast; two-hop limit retracted | 04 |
| 2026-09-19 | Hop plateau and dilution, second pass | `tul_slot_spandec_strict_prev_reach1.yaml`, `hop_distance_probe.py` at depths 1 to 8, 12, 16 | h5 gets 79% of fall at pass 4; disjoint rows h3 +0.0665, h4 +0.0545, h5 +0.0302 | Failure by filing rule; carried content decays | 04 |
| 2026-09-19 | Loop carry, sum and gate | `tul_slot_spandec_strict_prev_reach1_carry.yaml`, overrides `carry-sum`, `carry-gate` | Depth-6 CE 4.4554 (sum, behind ruler 0.098), 4.4081 (gate, behind 0.050) | Failure; state unbounded, far hops lose | 04 |
| 2026-09-19 | LXTUL fan4, K streams with repulsion | `tul_slot_spandec_strict_fan4.yaml`, `fan4-norepel`, `fan4-mean`, width partner `pk4` | Depth-6 CE fan4 4.3471 against width partner 4.3398; K1-K6 +0.0023 | Failure; fan 0.0073 nats worse than partner | 04 |
| 2026-09-19 | Sample-oracle gate, adapted reader | `tul-code-target-uf` at 30000, `sample_oracle_probe.py`, exit site `code_proj` | Gain(16) sigma 0.3 +0.0149; sigma 1.0 +0.0388; cos_exit 0.985 and 0.849 | Failure; fan4 queue restored by review | 04 |
| 2026-09-19 | Depth ladder, scored | `notul_norm_match_20k_p3.yaml`, `_p4.yaml`, `_d1.yaml`, `_d2.yaml`, `_d3.yaml`, `_d3fixed.yaml`, `_d6fixed.yaml`, `mean_depth` | Token-paired CE vs top: d1 +0.0674, d3 +0.0095, p3 -0.0013, p4 -0.0051; own K1-K6 top +0.170 | p3 ships by file rule; clamped rungs out | 04 |
| 2026-09-19 | Hop staircase replicates | disjoint 480-row draw | h3 +0.0665, h4 +0.0545, h5 +0.0302 | Stands | 04 |
| 2026-09-20 | Fan diversity, cosine and epiplexity | `tul.fan_repel_mode: cos`, `epi`, `epivol`; `slot-spandec-strict-fan4`, `-epi`, `-epivol` | Epi stream rank 1.0001; epivol oracle gain 0.018; oracle over mixture 0.008 to 0.018 | Both failures; falsifier 0.022 failed | 05 |
| 2026-09-20 | Select-then-commit fan | `tul.fan_mix: select`, `tul.fan_select_eps 0.05`, `tul.fan_select_gate_lambda 1.0`, `slot-spandec-strict-fan4-select` | Deployed one-stream write 0.086 behind width partner `pk4`; gate recovered 0.014 of 0.113 | Failure; train and eval mismatch | 05 |
| 2026-09-20 | Select arm writes the gate's pick | `tul.fan_select_write: gate`, `anneal`, `slot-spandec-strict-fan4-select-gate` | Recovered 0.066 of the 0.086 deficit | Failure on P-5; mismatch carries most of it | 05 |
| 2026-09-20 | Write-all fan, per-token coda selection | `tul.fan_mix: all`, `tul.fan_all_wta_lambda`, `slot-spandec-strict-fan4-all` | Tokens K1-K6 +0.0049 [+0.0044, +0.0055]; paired against pk4 at depth 6 -0.034 | Success under filing rule; loop flat | 05 |
| 2026-09-21 | Mixture probe on write-all read | `lab/divergence/fan_mixture_probe.py`, fan4-all at 5k, 48 rows | Oracle 4.2082, deployed 4.2485, uniform mixture 4.2583, best single 4.2874 | Selector question closed; regret is price of hindsight | 05 |
| 2026-09-21 | Mixed fan with no fixed-point term | `model.core_fixed_point_lambda 0.0`, `slot-spandec-strict-fan4-all-fp0` | Pass-6 rank 2.01 against 2.06; loop/core_gain_t0 9.92 to 2.81 | Failure; term holds exit scale, not stream spread | 05 |
| 2026-09-21 | Streams as K independent samples | `tul.fan_seed_noise`, `slot-spandec-strict-fan4-all-noise` | Four streams read as one (cosine 0.99 at pass 0); entry norm grew 11x | Failure; draw swamped | 05 |
| 2026-09-21 | Streams as lineages | `tul.fan_lineage`, `slot-spandec-strict-fan4-all-lineage` | Paired reader gain +0.004, inside spread; winner persistence +0.044 | Failure; channel costs nothing and buys nothing | 05 |
| 2026-09-21 | Per-pass trigger re-injection | `tul.fan_trigger_every_pass`, `slot-spandec-strict-fan4-all-trig` | Exit rank 1.43 of 3 (write-all 2.03) | Failure; closed | 05 |
| 2026-09-21 | Slot loop as a denoiser | `tul.loop_denoise`, base `tul-code-target`, 20k | Rolled-out sample 0.126 worse than base; pass 1 cosine 0.02, flat | Failure; teacher forcing is a bypass | 05 |
| 2026-09-21 | No prelude on strict slot loop | `model.n_prelude: 0`, `slot-spandec-strict-np0` | K1-K6 +0.0045, K3-K6 -0.0008 | Failure; flat under any entry | 05 |
| 2026-09-21 | Strict partner at prefix_k 8 | `slot-spandec-strict-pk8`, width ruler | 0.035 nats better than two-cell strict; eight-code quantizer 0.059 worse than it | Success; width ruler for discrete panel | 05 |
| 2026-09-21 | Flow noise shifted high | `tul.code_t_logit_mean -1`, `tul-code-cfg-tlow`, 20k | Context dependence 1% to 1.9%; one draw 0.010 better than parent | Failure on arm's reason | 05 |
| 2026-09-21 | Semantic target for the code (planned) | `tul.code_target_source: sonar`, `lab/experiments/planned/2026-09-21-lctul-tlow-sonar.md` | Not run | Open | 05 |
| 2026-09-21 | Far-budget split | `loop_reach`, `tg_coda_prefix_reach`, span-h, reach 1, reach all, coda layer 0 | Depth 6: span-h 4.3975, reach1 4.2181, reachall 4.0922; far budget at least 0.126 | Failure on 6 of 9; Step 1 queued | 05 |
| 2026-09-22 | Token-like read of slot exit | `bcast`, `bcast-all` | K1-K6 +0.0016 (bcast), +0.0013 (bcast-all), +0.0016 (ruler); worth 0.012 to 0.017 | Failure on the question; read was not the limit | 05 |
| 2026-09-22 | Seed twins of single-block reach arm | `budget-web-reach1-coda`, seeds 1 and 2 | Two seeds differ by 0.152 nats at depth 6; plain span seeds differ 0.004 | Failure; learned-trade reading refuted | 05 |
| 2026-09-22 | EMA code target (LCTUL-J stage 1) | wandb `ema`, `ema0`, `ema-l2` | Token K1-K6 +0.0006, +0.0016, +0.0014; cell worth 0.165 to 0.174 | Failure on both draws | 05 |
| 2026-09-22 | LXTUL-R Step 1, leaky chain | `slot-spandec-strict-fan4-all-reach1`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `core_fixed_point_lambda 0`, `slot_state_renorm True` | Token K1-K6 +0.0202 (paired -0.033 vs strict) | [R] Method fault; leaked about 4 slots per pass | 05 |
| 2026-09-22 | LXTUL-R Step 1b, corrected relay | `loop_reach 1`, one slot per pass | Token K1-K6 +0.026; depth-1 cost 0.053 against direct-read control | Failure; a tax, not a gain | 05 |
| 2026-09-22 | LXTUL-R Step 2 panel | `lxtul-r-step2`: persistent-carry arm and history-stream arm | History-stream K1-K6 +0.0204; persistent-carry +0.0234 (bar +0.035) | Failure on both; chain geometry closed as measurement | 05 |
| 2026-09-22 | Slot-loop campaign synthesis | `.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md` | Twelve designs without forced relay: token K1-K6 in [-0.0001, +0.0033] | Filter for later work (two conditions) | 05 |
| 2026-09-23 | Stochastic slot loop, posterior and prior | `LXTUL-G`, `lab/divergence` `lxtul_g_probe.py` | Exposure gap 0.9840 at 5000; shuffled cell costs 1.12 at span first token | Failure; reader trained on cells loop never makes | 05 |
| 2026-09-23 | Stochastic loop on K prior rollouts | `LXTUL-GK`, `gk4-shared`, `gk1` | Learned sigma/r 0.1 to 0.00026; width gain at four samples 0.00005; gk1 final val 4.4238 vs ruler 4.4249 | Failure; objective prefers deterministic loop | 05 |
| 2026-09-23 | Provable loop contribution (Lean) | `lab/theory/tul_exploration/` (Lean 4, Mathlib v4.31.0) | None measured; four-stage plan | Proposed | 05 |
| 2026-09-23 | Enumerated codes, width budget (LXTUL-E stage 0) | `e0k1`, `e0k4` (2000 steps) | Four codes buy 0.103 nats per span token over one committed reader | Success on deciding clause | 05 |
| 2026-09-23 | Superposition probe on slot cells | `lab/divergence/superposition_probe.py` | Ratio 1.5 to 1.8; span-specific worth 0.096 per token | Success; probe cannot separate branches from shared content | 05 |
| 2026-09-24 | Teacher-forced codes read in parallel (LXTUL-E stage 1) | `e1`, `e4` (5k, 490 blocks) | e4 6.788 at depth 6 against frozen-cell head 6.791; separation 1.99 | Failure as written; width ~0.11 nats per span token | 05 |
| 2026-09-24 | Committed read points (LXTUL-E stage 2) | `e4j4`, `e4probe` | Coda K1-K6 0.0011 to 0.0020; pointing read at first four tokens costs coda 0.067 | Failure; next move is the reader | 05 |
| 2026-09-24 | Slot map cause interventions | `lxtul-e4probe-map` reference; forced-work, floor-move and source-removal interventions | Floor-move map 0.781 (floor 0.773); source removal map 0.871, coda lost 0.016 | Failure on causal link; floor finding stands | 05 |
| 2026-09-24 | Fixed-point term holds the loop still (Stage 3) | `core_fixed_point_lambda` 0.0 and 1.0; arms `nofp`, `map`, `e4probe` | `nofp` coda K1-K6 +0.0123 (95% by pass 3); eval map 0.884; `map` +0.0009 | Success by rule; the term is the switch | 05 |
| 2026-09-25 | Fixed-point weight replication | `nofp_s2` (seed 2), `fp01` at 0.1, `map` at 1.0 | `nofp_s2` +0.0077 [+0.0071, +0.0082]; `fp01` +0.0126; `map` +0.0009 | Success; dose is the switch | 05 |
| 2026-09-25 | Span-level NextLat on exit state | `lxtul-fp01-nextlat`, fp01 plus NextLat | Coda K1-K6 +0.0010 against fp01 +0.0126; par K1-K6 falls to +0.0201 | Failure; exit collapses to a constant | 05 |
| 2026-09-25 | fp01 resumed to 10k | `lxtul-fp01-10k` | K1-K6 +0.0125 at 5k, +0.0177 at 10k; final val 4.4199 and 4.1154 | Success; depth use grows | 05 |
| 2026-09-26 | KV-cached generation | `tul_generate_cached.py`, `generate_tul_cached`, `generate_tul_graphed` | 256 tokens: eager 16.6, cached 89.0, graphed 280 tok/s | Kept (implemented); refuses fan | 06 |
| 2026-09-26 | Wider slot channel (pk4 projections) | `tul.prefix_k: 4`, `tul_slot_spandec_strict_e4probe_fp01_pk4` | Gap +0.2729 at 5k (baseline +0.2650); coda K1-K6 +0.0112 | Failed; four projections of one state add no content | 06 |
| 2026-09-26 | Previous-span direct read | `tul.tg_coda_token_reach: 1`, `..._reach1` | Gap +0.2501; coda K1-K6 +0.0085; zero-cell worth 0.197 to 0.082 | Failed; closed 0.015 of gap | 06 |
| 2026-09-26 | Strict baseline gap widens at 10k | `tul_slot_spandec_strict_e4probe_fp01`, 10k resume | Gap +0.3326 at 10k (5k +0.265); plain K1-K6 +0.0528 to +0.0612 | Failed; widening 5 times seed spread | 06 |
| 2026-09-26 | Write-all fan with fixed-point term | `tul.fan_mix: all`, `tul.fan_k: 4`, `core_fixed_point_lambda: 0.1` | Gap +0.2181, 0.0472 better than strict; coda K1-K6 +0.0057 | Success on gap; depth small | 06 |
| 2026-09-26 | Rollout selection ceiling | analysis of LX rollouts | Bayes minus oracle +0.0413 (5k), +0.0418 (10k); KL from uniform about 0.45 | Failed own reading rule; spread small | 06 |
| 2026-09-26 | LX-Carry stage 0 | analysis | K1-K6 change 0.0009 (needed 0.05) | Failed; rollouts an ensemble | 06 |
| 2026-09-26 | Hard credit over rollouts | `tul.code_enum_credit: hard`, `tul.code_enum_hard_eps` 0.05 | Span-coherent part +0.0048 (predicted 0.05); coda K1-K6 +0.0100 | Failed headline; two of four arms detonated | 06 |
| 2026-09-26 | LX-Fan with winner-take-all | `tul_slot_spandec_strict_lxfan4_wta_fp01`, `tul.code_enum_k: 4`, `tul.fan_k: 4`, `tul.fan_mix: all`, `tul.fan_all_wta_lambda: 1.0`, `tul.fan_all_wta_winner: per_rollout` | Seed 1 K1-K6 +0.0162 [+0.0153, +0.0172]; seed 2 +0.0093 [+0.0088, +0.0100] | Depth held on two seeds; gap prediction missed | 06 |
| 2026-09-26 | Soft-credit fan retry | `tul_slot_spandec_strict_lxfan4_fp01`, `..._lxfan6_fp01` | 4-cell K1-K6 +0.0161; 6-cell detonated at 1545 | Failed; soft 4-cell fan unstable at some rate | 06 |
| 2026-09-27 | Epivol lambda sweep | `tul.fan_repel_mode: epivol`, `tul.fan_repel_lambda` 0.01, 0.05, 0.11, 3000 steps | K1-K6 +0.0301 (0.01), +0.0099 (0.05), +0.0167 (0.11) | Failed flatness; not separable at 3000 steps | 06 |
| 2026-09-28 | Epivol grid 2 and seed bands | `tul.fan_repel_lambda` 0, 0.0001, 0.001, 0.005, 0.015 (planned, partly run) | Lambda 0.01 at seed 2: +0.0026 against seed 1 +0.0301 | Not finished; queue paused by Wolfe | 06 |
| 2026-09-29 | One-rollout code policy | `tul.code_policy_k`, `tul.code_policy_lambda`, `tul_slot_spandec_strict_e1probe_fp01_policy4` | Control (one code) K1-K6 +0.0047; policy +0.0038; policy minus control -0.0050 | Failed; one-code control 1.16 times plain wall clock | 06 |
| 2026-09-29 | Head-graded WTA against ungraded fan | `tul_slot_spandec_strict_fan4_all_fp01_nowta`, `_hwta` | Ungraded K1-K6 +0.0042 (0.98 times plain); head-graded +0.0045 (0.70 times); coda-graded +0.0057 (0.78 times) | Failed both; ungraded fan becomes cheap base | 06 |
| 2026-09-29 | Map winner for WTA fan | `tul.fan_all_wta_winner: map`, `tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin` | Mapwin K1-K6 +0.0096 [+0.0087, +0.0105] against band +0.0072 to +0.0150 (3000 steps) | Quality-neutral within band | 06 |
| 2026-09-29 | Latent WTA (proposal) | `tul_slot_spandec_strict_lxfan4_wta_fp01_latwta` (in tree) | No filing in range | Proposal only | 06 |
| 2026-09-29 | Per-depth persistent state (planned) | `lab/experiments/planned/2026-09-29-per-depth-state.md` | Not run | Planned only | 06 |
| 2026-09-30 | Factor fan and learned routers | `tul_slot_spandec_strict_fan4_all_fp01_opf`, `_opf10`; reader-trained router; latent-teacher router | Factor fan K1-K6 +0.0073; opf10 +0.0047 (gap +0.3845, latent R^2 0.042); routers +0.0068 and +0.0063 | Failed; factor task is the dead end | 06 |
| 2026-09-30 | Latent-selected loop, teacher-followed | `tul.fan_loop_select: joint`, `detached`; `tul.fan_route: latent` | Joint K1-K6 -0.2294; detached -0.1701; ungraded reference +0.0042 | Failed hard; teacher path worked, router path did not | 06 |
| 2026-10-01 | Latent-selected loop, router-followed | `tul.fan_lsel_train_follow: router`, joint and detached | Detached K1-K6 +0.0095; ledger CE 4.4894 against ungraded 4.334 | Failed pass rule; loop earns, CE 0.155 behind | 06 |
| 2026-10-01 | Detached latent at weight 1 | `tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1` | K1-K6 +0.0333 [+0.0318, +0.0347]; ledger CE 4.447 | Failed on CE (0.113 behind) | 06 |
| 2026-10-01 | Full cell read at the coda | `tul.fan_lsel_read` (full read) | K1-K6 +0.0170; ledger CE 0.168 behind ungraded | Failed on CE | 06 |
| 2026-10-02 | Rank-only latent head | `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank`, `tul.fan_lsel_head_input: detached` | K1-K6 +0.0052; ledger CE 4.380 (pull twin 4.387, K1-K6 +0.0096) | Failed pass rule; kept selection | 06 |
| 2026-10-02 | Huginn and HRM loop geometry | Huginn-0125 (3.5B) and HRM-Text probes | Huginn participation ratio 215 to 604 from pass 1 to 4; HRM shared direction 6.5% then 1% | Huginn success; HRM rotation prediction failed | 06 |
| 2026-10-02 | Staged latent pretraining (planned) | `tul_latent_pre_prelude_l2`, `_nce`, `_wta4`, `tul_latent_pre_base` | Not run in range | Planned only | 06 |
| 2026-10-02 | Common-mode probe on frozen prelude | instrument | 98% of frozen plain prelude output one direction | Corrected 10-05 (slot carrier constant) | 06 |
| 2026-10-03 | Per-pass cell RMSNorm, rank-only head | `tul.slot_cell_pass_norm: rms`, `..._lsel_joint_rf_lam1_rank_cnorm` | K1-K6 +0.0215 [+0.0205, +0.0225] seed 1; +0.0172 seed 2; ledger CE 4.353 | Success on pass rule, both seeds | 06 |
| 2026-10-03 | Per-stream norm on detached weight-1 arm | same filing | K1-K6 +0.0119 (down from +0.0333) | Rejected | 06 |
| 2026-10-03 | Read-side norm | `tul.slot_cell_pass_norm: rms_read` | Read norm 31.0 to 31.5; ledger CE 4.611 at 3000 | Failed; cells still rotated | 06 |
| 2026-10-04 | Lead arm at 10k | `..._cnorm_10k` | K1-K6 +0.0236 [+0.0225, +0.0246]; CE 4.0975; gap +0.356 at 10k | Success on pass rule | 06 |
| 2026-10-04 | LXTUL locked primary candidate | `morph/configs/lxtul.yaml` | Seed 1 +0.0215, seed 2 +0.0172, 10k +0.0236; ungraded reference +0.0042 | Implemented; base for later loop work | 06 |
| 2026-10-04 | Eight cells per slot | `lxtul_fan8`, `tul.fan_k: 8` | K1-K6 +0.0103; gap +0.2697; far-bigram bucket 75.7% of gap | Failed | 06 |
| 2026-10-04 | Exact-recall gap probe | analysis | Bigram repeats 33 to 256 back: 15% of tokens, 65% of gap at 10k | Success on question; window explanation consistent, not proven | 06 |
| 2026-10-04 | CE test queue and speed notes (proposed) | `2026-10-04-lxtul-ce-test-queue.md`, `2026-10-04-slot-loop-training-speed.md` | Items 9 and 10 ran on 10-05 | Proposed | 06 |
| 2026-10-05 | Carrier constant twin | `lxtul_hcuni`, `tul.loop_attn_hc`; `lxtul_center`, `tul.loop_attn_center` | Uniform K1-K6 +0.0171, gap +0.270; centered detonated at 470 (2.3e6) | Failed both; constant rebuilt by another path | 06 |
| 2026-10-05 | Cell packing panel | `lxtul_pool16`, `tul.slot_pool_heads: 16`; `lxtul_recon`, `tul.recon_weight: 1.0`; `lxtul_pool16_recon` | Pool16 K1-K6 +0.0221; recon +0.0010; both +0.0039 (gap +0.3942) | Failed; reconstruction killed depth | 06 |
| 2026-10-05 | Wider write | `lxtul_fan4x2`, `tul.prefix_per_cell: 2` | K1-K6 +0.0171; gap +0.2835 | Failed on gap | 06 |
| 2026-10-05 | Full token history | `lxtul_hist`, `lxtul_hist_embed`, `tul.coda_token_input` | Open coda K1-K6 +0.0140, gap +0.2308; embeddings coda +0.0134, gap +0.4046 | Failed both; closed 0.04 of gap | 06 |
| 2026-10-05 | Snap pseudo-token carrier | `lxtul_snap`, `lxtul_snap_entry` | Snap K1-K6 +0.0119, gap +0.2426; seed-sourced +0.0106, detonated at 811 | Success on pass rule; gate never opened | 06 |
| 2026-10-05 | Data-flow probe on slot carrier | analysis, entry, pass 1, exit | Entry 54 to 57% differs between slots; after pass 1 0.2 to 0.4%; exit 1.7 to 1.8% | Success; first pass makes one shared vector | 06 |
| 2026-10-05 | Variable cell count (proposed) | `2026-10-05-lxtul-variable-cell-count.md` | Not built | Proposed | 06 |
| 2026-10-06 | Output-only pointer head | `tul.pointer_heads: 4`, `lxtul_pointer` | CE@6 3.9280; K1-K6 +0.0155 [+0.0147, +0.0164]; gap to plain ruler -0.150 (plain 4.0806) | Kept; head off costs 0.48 CE | 07 |
| 2026-10-06 | Pointer cell keys | `tul.pointer_cell_key: true`, `lxtul_pointer_cellkey` | CE@6 3.9208; paired -0.0073 [-0.0091, -0.0053]; K1-K6 +0.0098; detonation at 2297 to 2310 | Rejected; win inside 0.009 seed spread | 07 |
| 2026-10-07 | Coverage repetition phase | `tul.pointer_coverage_lambda: 1.0`, `lxtul_pointer_cov` against `lxtul_pointer_ft` | T 0.7 seq_rep_4 0.374 against control 0.291; greedy 0.925 against 0.909 | Failure; sign wrong; kept default off | 07 |
| 2026-10-07 | DITTO repetition phase | `tul.ditto_rows: 3`, `tul.ditto_lambda: 0.5`, `lxtul_pointer_ditto` | T 0.7 seq_rep_4 0.140 against 0.291 (-52%); trainer val 4.0254 against 3.9548 (+0.071); K1-K6 +0.0294 | Mixed; first working guard; shipped | 07 |
| 2026-10-07 | Winner recipe declared | `lxtul_pointer.yaml` then `lxtul_pointer_ditto.yaml` | Plain 4.0806; LXTUL 4.3529 (gap +0.272); pointer 3.9280 (gap -0.150) | Adopted as loop base by Wolfe | 07 |
| 2026-10-08 | Fast pointer recipe | `morph/configs/lxtul_pointer_fast.yaml`, `lxtul_pointer_fast` | CE@6 3.9248 and 3.9229; K1-K6 +0.0104 and +0.0091; 29.8k tok/s (eager 11.8k) | Success; base for later panels | 07 |
| 2026-10-08 | Production rerun noise floor | `lxtul_pointer` eager, seed 1, rerun | Rerun CE@6 3.9187, K1-K6 +0.0095; winner read +0.0155 | Protocol failure; winner was a high draw | 07 |
| 2026-10-08 | Paired continuation noise | instrument, `step_2500.pt` continuations | Delta K1-K6 0.0005 (first pair), 0.0012 (three draws) | Mixed; resolves change of about 0.0015 | 07 |
| 2026-10-08 | FAST2 package, paired | `model.fan_target_online: true` plus speed package | CE@6 4.0461, 4.0457, 4.0471 against eager mean 4.04645; trainer 19.2k against 11.5k tok/s | Mixed; CE held | 07 |
| 2026-10-08 | Slot compaction | `model.slot_depth_stratified: true`, `model.slot_compact: gather` | CE@6 4.04733 against eager 4.04637; mean K1-K6 +0.01037 against +0.00917; 18.8k against 18.2k tok/s | Success on stated predictions | 07 |
| 2026-10-10 | Credit panel round 1: one-cell loop | `tul_onecell_pointer_fast` | CE@6 3.8997; K1-K6 +0.0022; write worth 0.193 | Failure of hypotheses; one-cell beats LXTUL on CE | 07 |
| 2026-10-10 | Credit panel round 1: random pick | `lxtul_pointer_fast_randpick`, `tul.fan_lsel_pick: random` | CE@6 3.9344; K1-K6 +0.0060 | Inside noise on CE | 07 |
| 2026-10-10 | Credit panel round 1: prelude detach | `lxtul_pointer_fast_preldet`, `tul.coda_prelude_detach: true` | CE@6 4.2237 (about 0.29 worse); K1-K6 +0.0135 | Real loss; prelude needs token gradient | 07 |
| 2026-10-10 | Span decoder reads winner | `tul.spandec_read: winner`, `lxtul_pointer_fast_spdwin` | CE@6 3.9093 and 3.9290; K1-K6 +0.0046 and +0.0124 | Failure; two runs do not separate accounts | 07 |
| 2026-10-10 | Span decoder grades each cell | `tul.spandec_read: each`, `lxtul_pointer_fast_spdeach` | CE@6 3.9159; K1-K6 +0.0166; cell cosine 0.98; 39% rate cost | Failure; cells did not become distinct | 07 |
| 2026-10-10 | LXTUL without span decoder | `tul.spandec: false`, `lxtul_pointer_fast_nospandec` | CE@6 3.9927 against 3.9248 and 3.9229; K1-K6 +0.0020 against +0.0104 and +0.0091 | Failure; decoder stays (corrected overstated K1-K6) | 07 |
| 2026-10-10 | Cell lens probe | `lab/divergence/cell_lens.py` | Fan cells patch CE 4.6305, 4.6704, 4.6685, 4.6787; cell 0 wins 0.59 to 0.60 of spans | Failure against predictions; cell 0 is the copy cell | 07 |
| 2026-10-10 | Credit panel round 2 | `nowta_pointer_fast` (ungraded fan), `tul_onecell_pointer_fast_nospandec` | Fan CE@6 3.9196 (predicted 3.897), K1-K6 +0.0045; one-cell no-decoder 3.8996, K1-K6 +0.0015 | Failure of headline; decoder buys nothing on one cell | 07 |
| 2026-10-10 | Credit panel round 3, fan without decoder | `nowta_pointer_fast_nospandec` | CE@6 3.8994 (predicted 3.920); K1-K6 +0.0024 | Failure of headline; decision rule drops decoder | 07 |
| 2026-10-10 | Hinge reads a fresh pass each step | `model.slot_gain_pass_force: -1` | CE@6 4.0482, 4.0480, 4.0487 (fixed mean 4.0473); K1-K6 mean +0.0121 (fixed +0.0104); cost +2.0% | Success; fresh draw is default | 07 |
| 2026-10-10 | DiscoTUL built | `tul.disco`, `discotul_fast`, `lxtul_disco_fast`; control `tul_onecell_pointer_fast` | Bench only: LXTUL 230.1 ms/step, LX-DiscoTUL 274.1 (+19%), one-cell 201.5, DiscoTUL 244.9 (+22%) | Built on `feat/discotul`, not merged, not trained | 07 |
| 2026-10-11 | Fan decoder: mean, each, best | `tul.spandec_read: mean`, `each`, `best` | No decoder K1-K6 +0.0025, CE 3.8993; mean read 3.9212 (K1-K6 +0.0042); each-cell 3.9032 (cos +0.81); best 3.9066 (cos +0.61) | Failure on mechanism; cells collapse | 07 |
| 2026-10-11 | Fan no-decoder replicate | `nowta_pointer_fast_nospandec`, run 2 | CE@6 3.8993, within 0.0003 of run 1 | Success; proposal to drop decoder | 07 |
| 2026-10-11 | Set-reader span decoder (fan recipe) | `tul.spandec_reads_cells: true`, `nowta_pointer_fast_spdcells` | CE@6 3.8949 and 3.8953; K1-K6 +0.0052 and +0.0060; ~15% throughput cost | Adopted as fan recipe by Wolfe | 07 |
| 2026-10-11 | Set-reader cell lens | lens on set-reader checkpoint | Cell 0 holds about 70% of copied-token worth; cell 1 novel worth 0.052 | Success on predictions | 07 |
| 2026-10-12 | Eval regime check | new eval regime (graphs kept, probe every 4th eval) | Fresh 5k runs CE@6 3.903 to 3.908 (five runs); earlier seed-1 family 3.8949 to 3.8996 | Success; seed-1 family not typical | 07 |
| 2026-10-13 | Variable cell count v1 | `tul.cell_count`, `tul.cell_count_mean`, `..._cellg375`, `..._cellp3` | At M=4: gentle CE@6 3.9033, Poisson 3.9052; Poisson gap M=1 +0.0051, M=2 +0.0002, M=3 -0.0001; M=8 +0.06; M=16 +0.10 to +0.14 | Success on rules; downward knob live; no gain above 4 | 07 |
| 2026-10-13 | FLOP diet stack | `tul.spandec_layers: 1`, `tul.cell_count_mean: 2.0`, `..._cellp2_dec1` | Stack CE@6 3.8999 (-0.0014 vs pooled control 3.9013); K1-K6 mean +0.0064; bench 0.887 times | Success for the stack (seed 1 grad-norm max/median 45.5) | 07 |
| 2026-10-14 | Hinge every four steps | `model.slot_gain_every: 4`, `..._cellp2_dec1_ge4` | CE@6 mean 3.9043 (+0.0030, grey line); paired +0.0094, +0.0089, -0.0049; K1-K6 mean +0.0083; 123.9 against 129.96 ms | GREY; not in speed base; Wolfe's call | 07 |
| 2026-10-14 | Half-slot span decoder | `tul.spandec_slot_keep: 0.5`, `..._cellp2_keep05` | CE@6 mean 3.9075 (+0.0062, fails by letter); K1-K6 +0.0049 (94% of control); 123.1 ms | Failed; decoder needs every slot's target | 07 |
| 2026-10-14 | Speed base v2 rounding keys | `model.hc_bwd_fused`, `model.glue_fold` on the stack | CE@6 +0.0034 (sd 0.0026) over +0.003 line; K1-K6 -0.0001 [-0.0075, +0.0073]; 129.0 to 122.1 ms (1.056x) | FAIL by letter; keys off pending Wolfe | 07 |
| 2026-09-05 | Stock Parcae loop contribution | `parcae-small-140m`, mean 8, BPTT 4 | K1-K8 0.294 at 10k [0.286, 0.303]; K3-K8 0.0397; 5k K1-K8 0.261 | Success; positive control | 08 |
| 2026-09-05 | Selective compile throughput (Parcae 140M) | no key | +16.8% to +49.3% tok/s over eager blocks; best 91.9k tok/s | Success | 08 |
| 2026-09-05 | Slot-only conditioner panel on stock Parcae | `slot16-b4`, `slot16-b8`, `cond4-16-b4`, `cond4-16-b8`, `coda4-16-b4`, `coda4-16-b8` | K3-K16 +0.00011 to +0.00018 in all six arms | Failure of useful-depth hypothesis | 08 |
| 2026-09-05 | Serial boundary-only TUL, 16 arms | `embedding-all_tokens-attached-s1-k8` and 15 siblings, `serial_tul_setup` | None; first arm stopped at 11,965 of 15,000 updates | Pending; no scientific claim | 08 |
| 2026-09-05 | Serial TUL stop and exact resume | `run_control`, `serial_tul_setup` probe configs | Six four-update continuations matched reference exactly; 78 tests, 209 subtests | Success, engineering only | 08 |
| 2026-09-05 | Serial TUL full trainer, batch one fallback | `serial_tul_setup action=train` | Batch-two attention peak 24.13 GiB; worst row 18.26 GiB after fix | Failure of prediction; batch one fallback | 08 |
| 2026-09-05 | Serial throughput repair | `serial_execution` K/V publication, compiled segments | Median warmed update 2.4515 s to 1.6307 s (1.503 times) | Success | 08 |
| 2026-09-05 | Packed coda | `serial_execution` packed coda | Packed 1.01680 s against 1.3241 s (1.302 times); GPU 42.22% | Failure on targets; packed coda kept | 08 |
| 2026-09-05 | Row concurrency on CUDA streams | `serial_execution` row streams (removed) | 2.5248 s to 2.2390 s (1.128 times); largest Muon update differed by 61.8% | Failure; removed | 08 |
| 2026-09-05 | Packed thinking frontier | `serial_execution` frontier, index-map custom op | 2.525 s to 1.637 s (1.542 times); peak 10.70 GiB against 10.78 | Failure on GPU target by 3.6 points; kept | 08 |
| 2026-10-06 | LXTUL port panel on Parcae | `arm=lxtul`, `arm=plain`, `arm=lxtul_ctxinj`, `arm=bare` | LXTUL K1-K6 +0.0585 [0.0564, 0.0607], CE 4.1242; plain +0.0462, CE 3.8303; bare -0.0001 | Failure by frozen rule; hypothesis held (2.7 times MORPH) | 08 |
| 2026-10-06 | Round 2: seed twin, no span decoder, ungraded fan | `arm=lxtul` seed 2, `arm=nospandec`, `arm=ungraded` | Seed twin K1-K6 +0.0794 [0.0771, 0.0817]; no decoder +0.1325 (CE 4.1304); ungraded +0.0346 (CE 4.1100) | Failure of predictions 1 and 2 | 08 |
| 2026-10-06 | Gap decomposition: strict mask versus depth | `arm=gpt8`, `arm=lxtul_open`, `lxtul/gap_probe.py` | Strict minus open +0.2734 (93% of gap); per-token depth -0.022; far-bigram 85.7% of gap | Question answered: the gap is the strict mask, and copying inside it | 08 |
| 2026-10-06 | Exact-copy cache | `arm=lxtul_cache`, `lxtul/cache.py` | CE 3.8518; gap +0.0215 [0.011, 0.031]; K1-K6 +0.0491; cache-off K1-K6 +0.057 | Main hypothesis held | 08 |
| 2026-10-06 | Copy cache seed twin | `arm=lxtul_cache`, `run.seed=2` | CE 3.8533; gap +0.0230 [0.013, 0.032]; K1-K6 +0.0690 | Success; two seeds agree on CE to 0.0015 | 08 |
| 2026-10-06 | Strict then open, with and without cache | `arm=lxtul_switch`, `arm=lxtul_cache_switch` | Strict then open CE 4.0600, K1-K6 +0.0304; cached switch gap -0.0017 [-0.012, +0.007] | Switch alone costs about 0.21; cache ties plain | 08 |
| 2026-10-06 | Learned pointer head, unnormalised | `arm=lxtul_pointer`, `arm=plain_pointer`, `lxtul/copy_heads.py` | Strict plus pointer CE 3.8047, K1-K6 +0.0579; head worth 0.361; plain plus pointer 3.7072 | [R] Superseded by normalised numbers | 08 |
| 2026-10-06 | Identity reach, raw embedding copy | `arm=lxtul_reach`, `IdentityReach` | CE 4.1089; gap +0.2786; K1-K6 +0.0446; far-bigram gap 1.41 to 1.32 | Failure; module-specific | 08 |
| 2026-10-06 | Normalised pointer head and cell keys | `arm=lxtul_pointer` normalised, `arm=lxtul_pointer_cellkey`, `arm=plain_pointer` normalised | Strict plus pointer CE 3.7608, K1-K6 +0.0468 [0.0452, 0.0485], gap to plain plus pointer +0.0453; plain plus pointer 3.7155; cell keys gap +0.0379 | Failure on two predictions; leading Parcae recipe | 08 |
| 2026-10-06 | Pointer generation check | `lxtul/generate.py`, 32 prompts, 128 tokens | Strict plus pointer seq_rep_4 0.0105 against 0.0013; greedy loop 0.79 against plain 0.78 | Failure on predictions 2 and 6 as written | 08 |
| 2026-10-14 | Lean arm (withdrawn) | `arm=lean`, `lxtul/configs/arm/lean.yaml` | Bench 230.8 ms, 13.83 TFLOP against 269.7 ms, 21.49 TFLOP | Withdrawn before 5k run | 08 |
| 2026-10-14 | Minimal slot loop, one cell | `arm=minimal`, `arm=minimal_noptr` | CE@6 3.7647 and 3.7615; K1-K6 -0.0001 and -0.0000; 39.4k and 39.8k tok/s (1.78 times LXTUL) | Failure of predictions 3 and 4; loop does not earn | 08 |
| 2026-10-14 | Minimal fan, four ungraded cells | `arm=minimal_fan`, run dir `minfan-5k` | Seed 1 CE@6 3.7446; K1-K6 +0.0000 [-0.0000, +0.0001]; 32.7k tok/s | Run contradicts premise that fan buys depth | 08 |
| 2026-10-14 | Minimal fan with latent selection | `arm=minimal_fan_lsel`, run dir `minfan-lsel-5k` | Not filed; step 3800 of 5000 at read | Pending; no summary | 08 |
| 2026-10-14 | Flat references with pointer (planned) | `arm=gpt8_pointer`, `arm=flat14_pointer` | Not run; bench 79.8 ms and 123.3 ms at batch 6 | Planned only | 08 |
| 2026-10-14 | Speed ladder | `bench4`, `bench_all.py` | Batch 6 ms: gpt8 75.9; plain plus pointer 281.8 (240.5 with compaction); LXTUL plus pointer 269.5, exact keys 242.1 | Measurement; filing and bench agree within noise | 08 |

## Retractions

Each item gives the number, what it claimed, the corrected reading, and the part where the correction appears.

Part 01:
- Token-state dropout as divergence trigger (08-17). Withdrawn the same day, n=1 per arm. Part 01.
- Slot arm beats dense by 0.0562 nats and gate beats slot by 0.1054. Both qualified: the retention carry leak adds +0.1433 on the baseline. Part 01.
- "Loop destroys about 10x more diversity than pooling" (08-24). Retracted the same day. The law itself stands (slope -0.473 to -0.527). Part 01.
- Per-slot embeddings called "the cure" (08-24). Corrected to "best lever, not a cure." Part 01.
- Loop-ladder sigma-capped arm, K1-K6 0.233 (08-29), called "the first LOAD-BEARING loop." Retracted 08-31: K1-K6 -1.1198 with the carry off. Part 01.
- l2cap "winning recipe" and its 0.233 (08-30). Retracted 08-31 with a banner in the note. Five other docs still carry the claim, per the index. Part 01.
- DB-fix pair, interleave and l2-truncated comparators built on 0.233. Retracted with it. Their own numbers (0.0127, 0.0085) stand. Part 01.
- Carry-leak future attribution at 30k: 3.85 nats. Part 01.

Part 02:
- Paid loop shipped to `base.yaml` (09-03). Reversed the same evening. The 09-03 ship note does not record the reversal. Part 02.
- Eager gain hinge (the eager mask runs of the 09-04 and 09-07 panels) reads about 0.07 high (eager 0.94 against fused 0.87 at eps 0.02). Part 02.
- Warmup earning at 20k: 0.100 in the ship and recipe notes, 0.104 in the history index. The filing is the primary source (0.100). Part 02.
- Norm-match 5k value: 0.136 in one run against 0.185 in the per-pass panel. Not resolved in part 02. Part 02.

Part 03:
- Rank headline 5.7598 (cosine 0.7104) for `slot-spandec-strict` came from a pre-7a24adf probe that ran the prelude unrestricted. Corrected to 13.85 and 0.520 (trainer recipe), or 11.96 and 0.5635 (480-row panel). Part 03.
- Fitted-z headroom of 0.9 to 2.6 nats. Hindsight. Causal fit on real span reads +0.252 worse than the loop. Part 03.
- Mask K-curves from the 09-04, 09-07 and 09-08 panels were absmean artefacts. Under norm_match the mask lost its token dependence (+0.0009). Part 03.
- Span decoder part 1 runner val 4.4192 corrected to 4.4206. Part 03.
- First identical-target depth grid in the objective arms is invalid. Corrected at 7a24adf. Part 03.
- Math file's offset-0 damage claim does not hold on web text (web offset-0 worth +0.757, strict). Part 03.
- Math prereg bars (0.18 and 0.155) misquoted. Recomputed 0.1728 and 0.2136. Part 03.
- Core-token run-level aux CE gap 0.31 is dropout-confounded. Clean probe reads -0.0164. Part 03.
- Perpass memory mispriced. Decoder activations were 11.0 GB, not CE accumulators. Part 03.
- Trajectory K1-K6 +0.0342 is a forced-depth pad-cell artefact. Same-width trajrep reads +0.0019. Part 03 (scored in part 04).
- "Restriction costs two thirds" of the loop gap (loop-reads-tokens). Retracted: the Parcae control shows equal gaps (+0.0301 and +0.0279). Part 03.
- Depth ladder wall-clock arithmetic off by a factor of 2. Parts 03 and 04.
- Gradpass first filing said no state probe ran. One had. Corrected at 428efd5. Part 03.
- Static seed rank probe retracted as a predictor of trained rank. Part 03.
- Fixed-point-off "sign reversed half" reading not supported by its own numbers. Part 03.
- Information note brief said a trajectory adds information over the exit. `mi_traj` shows it equals the entry. Part 03.

Part 04:
- Horizon-fixed6 K1-K6 +0.278: out of distribution (trained only at depth 6), not depth. Part 04.
- Encoder code minus ruler -2.550 for the p 0.5 arm: a norm defect in the truth cell, fixed at bed41d7. Part 04.
- Trajectory K1-K6 +0.0342 (09-13, scored 09-14). Pad-cell artefact. Part 04.
- Detached code target K1-K6 +0.0323 read as depth. Retracted as depth by the genericity note (frozen coda). Part 04.
- All OWN, SHUF and ZERO code-target probes before commit 14f004d: void (open-slot bug gave every valid slot a zero cell). Part 04.
- Code-only collapsed first draw: void. Part 04.
- Code-only frozen-reference K1-K6 values (+0.0698, -0.0014, +0.0414, +0.0291) and shuffle cost +0.297: void, because the coda drifted above uniform. Own minus shuffled 0.138 and 0.110 stands. Part 04.
- Graded arm clean margin +0.0052 corrected to -0.004 after a sentinel bug (commit b3381dd). Part 04.
- "Dilution" as the name for carried-content decay (09-18). Withdrawn 09-19 for carried content. Part 04.
- Toy conclusion that a slot loop cannot carry past two hops (09-18). Qualified 09-19: a shared-core loop carries one or two hops for free. Part 04.
- Sample-oracle "closed" verdict (09-18). Qualified by Wolfe's 09-19 review, which says neither test used a noise- or stream-trained map. Part 04.
- Renorm "log 8 cap" wording (09-15). It caps the gap against the best draw, not the mean. Part 04.
- "7.4 nats = nothing" in the spec's section 17. Never measured. Part 04.

Part 05:
- Select arm's 0.086 deficit read as "a gate before the span cannot predict." The train and eval mismatch carries 0.066 of it. Part 05.
- Write-all selector regret 0.042 read as a lever. The 09-21 mixture probe calls it the price of hindsight. Part 05.
- Coda seed twin "learned single-block trade" (Step 0). Retracted: seeds differ by 0.152 nats. Part 05.
- LXTUL-R Step 1 K1-K6 +0.0202 ("largest slot-loop K-curve") is a leaky-geometry record. Relayed about four slots per pass. Part 05.
- EMA target stage 1: encoder loss sat on the predictor output, and `val/ce_tf` is a reader statement. Part 05.
- fan4-all prereg P-8 first clause miswritten. Part 05.
- Hinge gain logged in bf16 reads 0.97 on the Stage 3 arm against 0.884 in fp32. Use `core_map_fd.py`. Part 05.
- Stage 3 positive (nofp, seed 1, +0.0123): seed-2 replicate is 40% smaller (+0.0077). Part 05.

Part 06:
- Selection ceiling reading rule (KL 0.05 to 0.20) was the wrong test. KL of 0.447 is a real measurement. Part 06.
- Latent router teacher picked the coda's best cell at chance. The 09-30 proposal traced it to wiring: the map read cells detached. Part 06.
- LXTUL primary candidate note first said every cell was written to the coda. Corrected: the coda reads the final winner alone. Part 06.
- Common-mode 98% figure (10-02) read the stream mean. The 10-05 probe shows the constant lives in the stream differences. Part 06.
- Read-side norm guess: wrong. The cells still rotate. Part 06.
- Snap: pass rule holds, but the pseudo-token gate never opened (g 0.001 to 0.009). Part 06.
- Exact-recall prediction R-4 failed; the gap falls off at 256, the window size. Consistent, not proven. Part 06.

Part 07:
- Winner's K1-K6 +0.0155 (10-06) was a high draw. Production rerun read +0.0095 (10-08). Part 07.
- No-decoder LXTUL filing (10-10) overstated K1-K6. Corrected: the drop is 0.0071 to 0.0084, about 1.3 times the independent-run spread. Part 07.
- Coverage sweep rows contaminated by `init_from` restarting the train stream at doc 0. Control gap -0.209 is not comparable with the 5k gap -0.150. Part 07.
- Seed-1 fan family (CE@6 3.8949 to 3.8996) read as "one family of near-copied trajectories" (10-12). Fresh runs read 3.903 to 3.908. Fan numbers from 10-10 and 10-11 need this reading. Part 07.
- Paired noise first pair (|delta K1-K6| 0.0005) was a low draw. Three draws give 0.0012. Part 07.

Part 08:
- Unnormalised pointer numbers (strict 3.8047, plain 3.7072) superseded by normalised numbers (3.7608, 3.7155). Copy-heads headline "0.026 ahead of plain" superseded by like-for-like +0.098 (unnormalised) and +0.045 (normalised). Part 08.
- Minimal slot-loop filing said "the fan is the element that buys depth." The same-day hypothesis note and the minfan-5k run (K1-K6 0.0000) contradict it. Part 08.
- Round-1 K1-K6 headline +0.0585 and seed twin +0.0794 differ by 0.021. Any K1-K6 difference under about 0.02 is inside seed noise. Part 08.
- Speed claim: 1.78 times (trainer logs) against 1.76 times (step bench). Part 08.
- Ledger item 3 (`bare-5k` on a dirty tree) not reconciled with "code bc03f38 for every arm." Part 08.

## Mechanisms that held up across eras

- Depth earning needs payment through the core. The paid loop earned K1-K6 0.1685 at 5k, while free-ride slot arms stayed at or below 0.0113 (R0). Part 02. The paid loop itself was later rejected as not TUL (part 05), so this holds as a mechanism only.
- The gain constraint holds the slot map near 0.89 and stops the spike train. Fp32 gain 0.887 to 0.897 over 5000 steps, zero spike steps in one draw. Part 02. Iteration 0 still drifts from 0.94 to 0.996 (part 02).
- A terminal fixed-point term holds the early detonation. 0 of 6 draws detonated against 4 of 7 controls (Fisher p 0.049). Part 02.
- The fixed-point term sets a dose response on depth. Weight 0.1 keeps earning: `nofp` +0.0123, `fp01` +0.0126, seed-2 replicate `nofp_s2` +0.0077. Weight 1.0 kills it: `map` +0.0009. Parts 05 and 06.
- The fixed-point term is a scale control, not a spread control. Pass-6 rank with and without the term: 2.06 and 2.01. Part 05.
- The ternary norm_match rule lifts plain per-pass strength: plain K1-K6 0.185 against 0.033 under absmean at 5k. Part 02. It also removed the mask K-curves (token K1-K6 +0.0009). Part 03.
- A ternary core limits loop contribution at this horizon. Bf16 core moved the past-pass-1 value from 0.033 to 0.168. Part 02.
- Whole-span targets move the slot channel. Span decoder CE gap to mux-mask -0.0720 [-0.0748, -0.0698]. Part 03.
- Strict geometry closes the bypass for free. Paired CE vs spandec-mask -0.0001 [-0.0024, +0.0024]. Part 03.
- Reachability forces passes to matter, but only as a relay. Prev-reach1 K1-K6 +0.0163; passes two to six carry 0.011 nats at CE cost. Parts 03 and 05.
- Prefix width is a CE lever. Six cells beat two by about 0.030 whatever they hold (part 04). Strict pk8 beats strict by 0.035 (part 05). Width of distinct content beats width of copies: fan with fixed-point term gap +0.2181 (0.0472 better), while pk4 projections gap +0.2729 (part 06).
- Distinct content per cell needs a set-reading decoder. Mean-read and each-cell decoders collapse cells (cos +0.6 to +0.8). The set reader keeps them apart (cos -0.28). Parts 06 and 07.
- Write-all fan with per-token coda selection: gap closes by 0.047 nats on the same tokens. Parts 05 and 06.
- Rank-only latent head plus per-pass cell RMSNorm earns depth on two seeds: K1-K6 +0.0215 and +0.0172 at 5k, +0.0236 at 10k. Part 06.
- Copy channel carries most of the gap to plain. Far-bigram tokens hold 85.7% of the gap (part 08), 86% (part 07 winner note), and 65% for bigram repeats 33 to 256 back at 10k (part 06). The pointer head closes it: gap -0.150 against plain with no head (part 07), +0.045 like-for-like on Parcae (part 08).
- Output-only pointer repetition is a real cost, and DITTO halves it. Seq_rep_4 at T 0.7: 0.140 against 0.291 (part 07), at +0.071 nats of clean val.
- Cell 0 is the copy cell in the mean-read fan. Cell 0 wins 0.59 to 0.60 of spans (part 07), and holds about 70% of copied-token worth on the set reader (part 07).
- The hop staircase is real on the strict slot loop. Earning falls with hop distance: h3 +0.0640, h4 +0.0535, h5 +0.0288, h6 -0.0203 (part 04), replicated on disjoint rows h3 +0.0665, h4 +0.0545, h5 +0.0302 (part 04).
- Carried content decays under later passes, roughly a third per pass over passes 1 to 3 (part 04).
- The frozen coda's depth curve measures how generic the cell is (part 04). A frozen-coda K-curve needs a warning.
- Unfreezing the coda moves the cell from -4.64 to +0.177 nats of worth (part 04). The frozen reader was the limit on whether a cell is usable.
- K-curves measure dependence on depth, not the value of depth. The deep draw (mean 16) earns K3-K6 0.277 but its depth-16 CE is 0.104 worse than mean 6 (part 02). Mean depth moves K1-K6 but not final CE (part 04: own K1-K6 top +0.170 against p3 -0.0013 CE).
- Teacher forcing is a bypass. A pass gets a job only if the loss reads that pass's output with no shallower route (part 05).
- Selector regret is hindsight. Oracle over streams sits below the best deployed read by a price, not a gate defect (part 05, 0.042, and part 06, 0.0413).
- Gain hinge reads high in eager and bf16 paths. Eager 0.94 against fused 0.87 (part 02). Logged bf16 0.97 against fp32 0.884 (part 05). Use the fp32 finite difference.
- Warmup 1000 removes most flat-schedule detonations: 0 of 3 at 2500 steps against about 70% flat (part 02).
- Parcae core swap: the core is not the fault. Five maps in one flat band (part 03). Slot-only conditioner on Parcae reads flat (part 08).
- Variable cell count v1 gives a downward knob. Poisson-trained model reads M=1 within 0.005 of M=4 (part 07).
- A recovered spike is normal for AdEMAMix with ternary weights at this scale (part 05, Wolfe 09-25).

## Mechanisms refuted across eras

- Gradient-side and forward-side loop levers. Per-pass LoRA token K1-K6 -0.0000 (part 03). Progressive loss +0.0006 (part 03). Noise entry +0.0004 (part 03). MUX every pass depth 1 6.7698 equals ruler depth 6 6.7724 (part 03). Reread +0.0003 (part 03). Gradient-conditioned passes +0.0014 (part 03). All met their target in one pass.
- Per-pass targets are met in one step. Staged, latent-z energies (`recon` +0.0007; `disc` +0.0019), and critic (+0.0012) all left passes two to six near zero (part 03). Horizon targets degenerate on web text: LoopMTP slot port paired CE -0.0001 (part 04).
- Gain-target dial. Lowering the hinge to 0.95 or 0.98 moves the map 0.02 to 0.035 above 0.894 and costs token CE 0.05 and 0.15 (part 02). Expansive dial at gain 1.02: K3-K6 0.0007, spikes at 3877 and 4639 (part 02).
- Mask (tg_restrict) arms on the slot loop. The mask forces the reader but not depth (part 02 Y2 mask; part 03 staged mask +0.0033; part 03 mask on ruler +0.0009).
- Span-pooled and gated compressed branch. Eval worth 0.187 recovered 0 of the 0.357-nat gap (part 02). Gated pooling +0.031 inside noise (part 02).
- Coverage loss. Seq_rep_4 at T 0.7 rose to 0.374 against control 0.291 (part 07).
- Multi-token coda heads. Cost 0.34 nats of next-token CE at matched steps (part 02).
- Widen the carry at rho below 1 on every dim: no effect on early detonation (part 02).
- Block density (quarter density) gives dependence without computation: K3-K6 under 0.006 (part 02).
- Token-like read of slot exit: K1-K6 +0.0016 with or without the read (part 05).
- Prelude removal on the slot loop: K1-K6 +0.0045, K3-K6 -0.0008 (part 05).
- Stochastic loops. Posterior trained, prior deployed: exposure gap 0.9840 at 5000 (part 05). K prior rollouts: learned sigma fell to 0.00026 (part 05).
- Code target as sampler. Every sampled code read as an unconditional draw: marginal k16 minus k1 -0.004 at 5k (part 04), +0.016 (XM, part 04), -1.88 at thinker only (part 04). Discrete denoiser context share 0.02% (part 04).
- Thinker phase and classifier-free guidance. CFG guidance sweep moves one-draw CE under 0.01 (part 04).
- LaDiR-style two-stage training. Own minus shuffled -0.006 (part 04).
- LeJEPA on the code. Letting the target move lowers the flow loss by moving the target (part 04).
- Graded continuation target. Clean margin -0.004 after the sentinel fix (part 04). Ranking text does not rank codes.
- EMA target on the code. Target moved (rank 77 to 40) but the reader saw none of it. Token K1-K6 at the floor (part 05).
- Select-then-commit fan and gate write. Deployed 0.086 behind pk4; gate write recovered 0.066 (part 05).
- Fan diversity terms (cos, epi, epivol). Oracle gap inside 0.008 to 0.018 of the mixture (part 05). Epivol lambda flat test failed: 0.0301, 0.0099, 0.0167 (part 06). The 0.01 peak did not hold at seed 2 (part 06).
- Fan streams from noise (`fan_seed_noise`): four streams read as one, cosine 0.99 (part 05). Lineage channel buys nothing (part 05). Per-pass trigger re-injection: rank 1.43 against 2.03 (part 05).
- Factor fan and learned routers. Latent R^2 0.026 and 0.042, below the 0.1 bar (part 06). Router-followed latent loop: CE 0.155 behind ungraded (part 06).
- Teacher-followed latent selection: joint K1-K6 -0.2294, detached -0.1701 (part 06).
- Read-side norm (`rms_read`): cells still rotate onto the shared direction (part 06).
- Centered loop attention: detonated at step 470 (part 06). Uniform loop attention rebuilt the constant by another path (part 06).
- Wider write (two positions per cell): gap +0.2835, no CE gain (part 06). Cell packing with reconstruction: reconstruction killed most of the depth earning (part 06).
- Eight cells per slot: gap +0.2697, bar 0.262, failed (part 06). Fan with the mean-read span decoder: CE 0.022 worse than the fan with no decoder (part 07). The set reader replaced this read.
- Span decoder that grades the winner only (`spandec_read: winner`): two runs, mean K1-K6 +0.0085, not separable (part 07).
- Full token history for the prelude (`lxtul_hist`): closes 0.04 of the gap (part 06).
- Trace-free snap carrier: gate never opened, gain came from extra coda positions (part 06).
- Identity reach (raw embedding copy): far-bigram gap falls only from 1.41 to 1.32 (part 08).
- Learned copy heads, unnormalised: scores dropped the null-key mass, so they read pessimistic (part 08). Superseded by the normalised head.
- Latent prediction on the exit (NextLat, the span-level variant): par K1-K6 falls from +0.1190 to +0.0201 and the exit becomes a constant (part 05).
- Minimal one-cell loop (no stabilisers, no span decoder): K1-K6 -0.0001 (part 08). Minimal four-cell fan: K1-K6 0.0000 (part 08).
- Lean arm withdrawn (part 08). The MORPH-only machinery did not help the Parcae port.
- Serial TUL packed coda and row streams: row streams removed (1.128 times, part 08).
- Enumerated codes plus parallel read (LXTUL-E stage 1): width worth about 0.11 nats per span token, and the loop adds almost nothing (part 05).

## Threads still open

Merged from the eight part lists. Duplicates are removed. Items closed by a later part are in the next section.

Loop and reader questions:
- Is the depth earning on the strict slot loop real at more seeds and longer horizons? The fp01 recipe grew from 0.0125 at 5k to 0.0177 at 10k on one seed (part 05). The 20k reading was not run (parts 05 and 06).
- Gap to plain widens with training. Strict slot loop at 5k +0.265 (part 06), at 10k +0.333 (part 06). The pointer head closes it at 5k only (part 07). 10k pointer runs for plain and LXTUL were deferred and never run (part 08).
- The first loop pass makes the carrier one shared vector. Centered attention detonated. Whether another route removes the constant without detonation is open (part 06).
- Bounded carry that keeps the pass-1 read (`2026-09-19-loop-carry-reinjection.md`, proposed). The success test is the per-hop K-curve (part 04).
- Plateau and dilution at depths 4 and 5, and the near-bin dilution read directly (part 04).
- Two toy questions: whether the last two hops in the six-hop toy can be bought by anything other than width, and whether a fitted reader on real cells separates "head cannot read" from "state does not hold" (part 04).
- Frozen versus adapted reader for any depth claim. No arm in parts 04 and 05 used an adapted reader for depth (part 04).
- Noise-trained or stream-trained maps for the sample oracle. Wolfe asks for fan4 against fan4-norepel on the oracle gap and stream cosine (part 04). The fan4 control was run (part 04) but a noise-trained map was not.
- The 40k continuation of the 20k pair, and the 40k staged read. Gap flat from 10k to 20k (parts 02 and 03).
- The 20k and 40k horizon check on the `p3` depth rung, with a second seed and depths 4 and 5 (parts 03, 04).
- Staged targets on every pass (`slot-mnext-staged-all`): never run (part 03).
- Target the entry state does not already contain, needs more than one pass, and the coda can read (part 03, Wolfe's stated lane).
- Second-order gradient pass (`create_graph=True`) and masked twin of the gradient-conditioned arm: recorded, not queued (part 03).
- Core-token aux weight sweep (one seed, the aux backward confound on prelude and coda) (part 03).
- Coda split (6/6/2, 5/6/3, 4/6/1) to separate coda capacity from block count. Not built (part 03).
- Register m8, register reader, four rank-lever arms, per-cell targets with orthogonality, residual-code VQ against width-matched pk8. Owed or not run (parts 03, 05).
- Plain-span-local matched control for the memory table. Unbuilt (part 03).
- Bootstrap: dropped compressed branch against bootstrap is unseparated (part 03).
- Precision axis planned and not run (part 02).
- Truncated BPTT fix and eager eps fix are proposed, not built (part 02). The eager reading is handled by fp32 finite differences (part 05).
- Slot constraint inert under norm_match. Whether to keep it on under norm_match is open (part 02).
- Gain constraint iteration 0 drift (0.94 to 0.996) (part 02).
- Norm-match 5k discrepancy (0.136 against 0.185) and final val against paired CE disagreement at 20k (part 02).
- Semantic target, the sonar code target (`tul.code_target_source: sonar`), planned, blocked on a cache check (part 05).
- NextLat combined with a frozen or EMA target, or a variance floor, to hold per-slot content (part 05).
- Winner-take-all responsibility under a stochastic loop. The XM hard-min fix (arXiv 2607.27372, App. F.1) is cited and not built (part 05).
- Horizon pair on the slot loop: 20k K-curve not queued (part 05).
- Decode speed of the LXTUL recipe is not measured. The fast decode path refuses the fan (parts 06 and 07).
- Stratified-only arm to split the compaction cause (part 07).
- `model.core_gain_lambda` in `_core_region` still reads a frozen hinge pass (part 07).
- `model.slot_gain_rows` (hinge on k rows, rotating) is proposed, not built (part 07).

Code, sampling and generation questions:
- Code target InfoNCE twin (`tul_code_only_nce.yaml`). Planned, never filed (part 04).
- Code target CE companion and its unfreeze twin. The companion stopped at 15000 and its continuation never ran (part 04).
- Pure graded arm (`tul_code_grade.yaml`, weight 0). Not run (part 04).
- Coda-space XM and noise-search XM. Not filed (part 04).
- Thinker phase 3 and the 50k continuations. Planned, never run (part 04).
- LoopMTP token loop: 20k probe unfiled (part 04).
- Generation side of the semantic probe. No reader trained on samples has been scored (part 04).
- Pointer generation and DITTO redesign: rows on top of the batch, fewer copies per pseudo row, longer or lower-rate DITTO. DITTO on the fan is untested (part 07).
- Plain plus pointer ruler on MORPH (owed since 10-06). A pointer seed twin is owed. Normalised pointer seed twin is single seed (parts 07, 08).
- Plain plus trained cache (the fair gap with the cache on both sides): untested (part 08).
- Strict then open switch at 10k, and a MORPH port of the copy cache (part 08).
- Full token history and the prelude proposal (`2026-10-05-prelude-full-token-history.md`): only the open coda and embeddings pair ran (part 06).

Fan, selection and recipe questions:
- Hard credit and soft credit at 5k: credit question open. One-cell control 1.16 times plain wall clock (part 06). Part 07 credit panel ran; the one-cell loop beat LXTUL on CE (part 07). Credit is partly answered and partly open.
- Epivol lambda range unresolved (part 06).
- Fan comparisons at the 0.004 level rest on seed 1 runs that the eval regime note says are not typical. A multi-seed check of the fan arms is open (part 07).
- Variable cell count v2 (exchangeable cells, pooled coda read) is the only route to "train at 4, infer at 128." Tracker items B1 to B4 open (part 07). v1 answered only the downward question.
- Raven memory arms (part 01): proposed, never run in any part.
- Gist with MUX supervision at matched budget and on a second seed. The unmasked twin with MUX was not run (part 01).
- Span-length gate: one seed pair, never replicated. Rejected 09-03 with the paid loop (parts 01, 02).
- Think-once panel readouts: the healthy arms have checkpoints, but their depth sweeps and worth profiles were never run (parts 02, 03).
- Wolfe's calls pending: hinge every 4 (GREY, +0.00303 against +0.003 line), the speed base v2 keys (+0.0034 paired), and whether the speed base becomes the stack (part 07).
- Two numbers for one gap disagree in scope: the planned 20k filing (strict one-cell led LXTUL by 0.007 at 5k and 0.029 at 10k) and the Parcae plan (0.265 at 5k to 0.333 at 10k). Not reconciled (part 07).
- The 2B-token knee. Wolfe expects four cells to win later. The planned 20k one-cell run is not queued (part 07).
- DiscoTUL and LX-DiscoTUL: built, not merged, not trained. Tracker item D2b waits (part 07). The disco paper's gain appears after about 13B tokens, per the filing (part 07).
- JEPA-style factored fan (tracker item D1): optional ablation, gated by a pick-agreement probe (part 07).
- The `spandec_read: winner` and `each` reads need paired continuation tests (part 07).
- Parked loop directions: RecurTrace attention, DiscoLoop realignment, latent feedback (part 03; DiscoTUL is the DiscoLoop arm, part 07).
- Per-depth persistent state (parts 06 and planned): not run (part 06).
- Latent WTA: proposal only, no filing (part 06).
- Staged latent pretraining (frozen targets, with and without the coda): planned and never filed (part 06).
- Plan C, expanded hyper-connections in the slot loop (xHC): planned, not run (part 06).

Parcae port questions (part 08):
- Minimal fan with latent selection: no summary at read time, step 3800 of 5000 (part 08).
- Minimal fan seed 2 and a two-cell fan: no run dirs (part 08).
- Flat references with the pointer head (`gpt8_pointer`, `flat14_pointer`), two seeds each: planned, not run. The publishing claim needs them (part 08).
- Emit-label direction A (move the first-token label to the slot's own cell): proposed (part 08).
- Small token loop (direction C, mean depth 2 or 3 on tokens): a question for Wolfe (part 08).
- Serial TUL 15k panel: 15 arms pending, first arm stopped at 11,965 of 15,000 updates. GPU use target (65%) still missed (part 08).
- Reach module: a different identity channel was not tried after the raw-embedding reach failed (part 08).
- Variable cell count at train and test on the Parcae port: not run (part 08).

Housekeeping (not a research thread):
- Many 09-10 to 09-13 notes sit in `proposed/` with appended outcomes. Lifecycle calls are Wolfe's (parts 03 and 04).

### Closed by a later part (kept here for the record)

- Paid loop as the shipped forward (part 02): closed as a decision in part 05 (2026-09-22 synthesis). Wolfe rejected it as not TUL. The code state was not re-checked in the parts.
- Staged targets, middle pass (part 03): run in part 03 and closed on its prediction count.
- Width sweep under absmean (part 02): closed by the pk8 width ruler in part 05 (a different test).
- fixed-point lambda 1.0 not swept (part 02): closed in part 05 by the weight 0.1, 1.0 comparison.
- Unread cited files: `2026-09-21-strict-pk8-ruler.md` closed in part 05; `2026-09-14-arc-loop-diagnostics.md` closed in part 04; `2026-09-19-hop-distance-plateau-and-dilution.md` closed in part 04.
- l2cap causal re-baseline (part 01): closed in part 01 (the causal re-train read +0.0171, flat).
- Flow-matching planner on post-core carriers (part 01): not run. The FM line closed on 08-30 (part 01).
- Ultralight coda (part 03): closed as failure in part 03. The coda-split arms are still open.
- Trajectory `trajrep` cut and queued back (part 03 and 04): closed in part 04 (run 09-13 to 09-14).
- Epivol seed bands at 3000 steps (part 06): partly superseded by the grid 2 run (part 06).
- LoopMTP slot port (part 04): failed in part 04. Token loop variant still unfiled (open above).

## Contradictions between sources

- Paid loop status. Part 02 records the 09-03 ship as a decision and then the same-evening reversal. The 09-02 `rejected/` recipe note says "not TUL" (Wolfe, 09-09). Part 05 says Wolfe rejected it as not TUL (09-22). Part 02 says the paid loop stayed in `base.yaml` through 09-09. The parts do not give the current code state.
- Paid loop K1-K6 under norm_match. Reread (part 03) says 0.120. Staged-20k (part 03) says 0.1363 at 5k. Parcae file (part 03) cites 0.185 and 0.033. These are different arms and runs. They are not reconciled.
- Plain K1-K6 at 5k. Part 02 reads 0.185 [0.182, 0.188] under norm_match. Part 02's 20k note reads 0.1363 at 5k in another run. Part 02 flags the 5k discrepancy and does not resolve it.
- 20k earning for the warmup paid loop. Parts 02 and its recipe note say 0.100. The history index says 0.104. The filing is the primary source (0.100).
- Token CE gradient share to the core. About 1 percent (part 03, the core-token file, 09-10). About 0.3 (part 03, the strict probes and latent-z file, 09-12). Different models, but the core-token file still quotes 1 percent as current.
- Best exit. The gradpass file claims the best exit of its family (part 03). The fullread filing records an exit 0.0014 higher. Raw numbers favour fullread.
- Staged arm replication. Original 5k gap +0.2099; the 20k run's own 5k checkpoint gap +0.1919 (part 03).
- Strict geometry. Bypass closed for free (part 03, 0.0000 cost) against the math result where the same write hurts offset 0 and helps web text (part 03).
- Register verdict against width ruler. The register verdict says "the honest name is a wider prefix" (part 03). The VQ file says pk8 beats strict by 0.035 while m4 beats strict by 0.022 (part 03). "m4 beats width" is untested.
- Coretok under-used or too strong. The filing says under-USED with no depth. The loop-reads-tokens clause calls the reading too strong (part 03).
- Trajectory config. Cut from the queue on 09-13, queued back on 09-14. The rejected note still says not queued (parts 03 and 04).
- Rank anatomy. The README reading 2 (prefix halves rank) is mostly a pooling artefact. In readout space there is no halving (part 03). The second-opinion note adopts rank starvation as the main build the same day (part 03).
- Carried content decay or dilution. The plateau file says per-pass decay. The carry file says dilution by accumulation, and that re-supply stops the decay but not the far hop. Both cite the same row set, and neither has a test that separates them (part 04).
- Sample oracle. The 09-18 file says the cell direction is irrelevant to the frozen reader. The 09-19 adapted-reader file says an adapted reader recovers 1.4 to 1.6 times more. Wolfe's review says neither tests a noise- or stream-trained map (part 04).
- Detached-cell cosine. Planned file reads val cosine 0.147. Code-only reference reads train cosine 0.1413 for the same arm. Different metrics, not a run conflict (part 04).
- Spectral gap. The loop diagnostics find no gap (ratio 1.05, part 04). The 08-24 alignment reading (directions align 2.9 times) was taken at another checkpoint. Not reconciled (parts 01 and 04).
- TUL-Code design status. The spec says the FM binding rule was waived for TUL-Code on 09-14. The 09-14 note keeps "proposed" pending Wolfe (part 04).
- Fan4-all. Its filing calls it a success. The same filing says the K-curve is flat (+0.0049). The synthesis note says the arm does not make the loop earn depth. Both are true: the success is on the exit (width and reader), not on the loop (part 05).
- LXTUL-R Step 1. The filing calls the chain "the largest slot-loop K-curve on the ledger." The synthesis calls it "a tax, not a gain." Step 1b cost 0.053 nats at depth 1 (part 05).
- The 0.87 map. The map-cause file keeps the 0.865 floor and refutes the causal link. The READ-BEFORE doc calls 0.87 "an architecture default" and "an effect." Both agree on the floor. They differ on whether the blocks' part is a cause (part 05).
- nofp size. The Stage 3 filing reads +0.0123 on seed 1. The follow-up reads +0.0077 on seed 2. Both cited. No mean is set (part 05).
- Pointer gap. Part 07 gives a gap of -0.150 against plain with no head. Part 08 gives +0.045 against plain with the same normalised head. These use different rulers. Both are cited.
- Decoder effect on LXTUL against the fan. On LXTUL, removing the span decoder costs CE by about 0.07 (part 07). On the ungraded fan, the mean-read decoder costs CE by about 0.022 (part 07). The set-reader result is the fan's current answer.
- Pointer and repetition. The 10-06 filing calls the CE result "a strong positive." The 10-07 coverage filing reads the same pointer model as repeating more under sampling (part 07). Both hold: they measure different things.
- Fan and depth on Parcae. The minimal filing says "the fan is the element that buys depth." The same-day hypothesis note says "the fan was a passenger." The minfan-5k run reads 0.0000 (part 08).
- Parcae round-1 and seed twin. Headline +0.0585 against seed twin +0.0794 (part 08). The difference is inside seed noise for this family.
- Mask gain readings. Part 02's mask note reads the eager hinge as biased high by about 0.07. A later part 02 note calls the mask K-curves an absmean artefact. Part 03 retracts those K-curves.
- Inline MUX detached arm and region Shapley numbers. The 09-26 run history calls them standing. The index calls them pre-fix. Both are in the record (part 01).

## Glossary

- K1-K6. CE at depth 1 minus CE at depth 6 of the same model. Positive means later passes lower the CE. It measures dependence on depth, not the value of depth. K3-K6 uses depth 3 in place of depth 1. Part 08 uses K1-K8 and K3-K16 for stock Parcae runs.
- Own minus shuffled, own minus zero. Paired readings of a cell. "Own" uses the row's own cell. "Shuffled" swaps cells across rows. "Zero" removes the cell. A positive own minus shuffled says the cell is span-specific (parts 04, 06).
- Worth, zero-cell worth. The CE cost of removing the cell from the coda, in nats. Part 03 onward.
- Slot. One position in the sequence per span. It holds a looped state and feeds the coda. Slot input is `E_slot` plus a bag mean or a boundary seed. Part 01 onward.
- Cell. One slot state written to the coda. The slot can hold several cells (fan or register). Part 05 onward.
- Fan. K latent streams per slot, each in its own cell. `tul.fan_k` sets K. Part 05 onward.
- Strict geometry. Tokens see only their own span. The slot loop is the only cross-span channel (`tul.tg_geometry strict`). Part 03.
- Open geometry. Tokens read every earlier token. The cells stay strict (`lxtul_open`). Part 08.
- Prelude. The token layers before the loop. They run on all positions. Part 01 onward.
- Core. The looped block stack. The paid loop runs the core on tokens and slots. The slot loop runs it on slot positions only.
- Coda. The token layers after the loop. They decode tokens from the slot prefix cells. Part 01 onward.
- Paid loop. The core runs on tokens and slots together (`tul.tokens_through_core: true`). Part 02. Rejected as not TUL (part 05).
- Slot loop. The core runs on slot positions only (`tul.tokens_through_core: false`). Part 02 onward.
- Span decoder. A training-only decoder that predicts the whole next span from the slot's exit state (`tul.spandec`). Set-reader variant reads the cells as a set (`tul.spandec_reads_cells`). Parts 03, 07.
- MUX. A local head that predicts the next span's token set from the slot. The bag target is order-free. Part 01 onward. Replaced by the span decoder as the main target (part 03).
- TG, Thought Gestalt. The restriction that lets a token see its own span and the slot states (`tul.tg_restrict`). Built after arXiv 2512.25026. Part 01.
- TST. The training phase that ends before TUL switches on (`training.tst_ratio`). Part 02.
- Ruler. The control arm that every other arm is compared with. The plain loop, the strict baseline, or the width partner is the ruler, depending on the panel. Part 02 onward.
- Width partner. The same model with the cell count changed to match width (for example `pk4` or `pk8`). Used to separate width from content. Part 05 onward.
- LCTUL. Latent Coded TUL. The slot holds a code of the next span, and a thinker samples it. Parts 04 onward.
- LCTUL-D. The discrete version. The code is a set of symbols from a cosine codebook and a masked denoiser samples it (`code_discrete`). Part 04.
- LX, LXTUL. LX is the enumerated-code rollout family (K rollouts per span). LXTUL is the slot loop with the fan, latent-selected loop, and per-pass cell norm, as locked on 10-04 (`morph/configs/lxtul.yaml`). Parts 05 onward.
- WTA, winner-take-all. A responsibility rule. Each rollout or stream is scored, and only the best one gets the loss (`tul.fan_all_wta_lambda`). The winner can be picked by the coda's CE (coda-graded) or a latent target. Parts 05, 06.
- Epivol. A diversity term on the fan. It adds a within-slot volume term to an epiplexity score (`tul.fan_repel_mode: epivol`). Part 05 onward.
- Gain hinge. A penalty on the map's typical gain above a target (`model.slot_gain_lambda`, `slot_gain_target`, `slot_gain_eps`). It reads one pass per step, drawn at random after 10-10 (`slot_gain_pass_force -1`). Part 02 onward.
- Fixed-point term. A penalty on the relative change of the state between the last two core iterations (`model.core_fixed_point_lambda`). Weight 1.0 shipped on 09-07. Weight 0.1 is the later recipe. Part 02 onward.
- Cotangent clip. The backward clip on the loop's carried cotangent at 4 times the exit cotangent (`model.slot_cot_clip: 4.0`). Part 02.
- Pointer head. Output-only copy heads (`tul.pointer_heads`). A head attends over earlier tokens and reads the attention as a distribution over the next token. A null key lets it abstain. Part 07 (MORPH), part 08 (Parcae).
- DITTO. A repetition phase. Some rows are pseudo-repetition rows, and a loss on the copy probability cuts repetition (`tul.ditto_rows`, `tul.ditto_lambda`). Part 07.
- norm_match. The ternary scale rule. It sets the per-tensor scale to the Frobenius norm over the square root of nonzero count (`training.ternary_scale_mode: norm_match`). Part 02.
- absmean. The earlier ternary scale rule (`symmetric`). It starved the looped core. Part 02.
- Warmup ramp. A 1000-step linear LR ramp, then flat (`training.warmup: 1000`). Part 02.
- Detonation, spike. A loss blow-up (the pre-clip to total ratio above 1e4 at step 200 or later). A recovered spike is one step or a few that returns to normal. Part 02.
- Tripwire. The abort rule for a detonation. Part 02.
- Cell norm, rank-only head. The per-pass RMS gain on every cell (`tul.slot_cell_pass_norm: rms`). The rank-only head ranks the cells and sends no latent gradient back (`tul.fan_lsel_head_input: detached`). Part 06.
- Latent-selected loop. After each pass, a teacher or router picks one cell, and the slot's cells reset to it (`tul.fan_loop_select`). Router-followed training (`tul.fan_lsel_train_follow: router`) is the later form. Part 06.
- Write-all fan. Every stream gets its own prefix cell (`tul.fan_mix: all`, `prefix_k == fan_k`). Part 05.
- Set reader. The span decoder cross-attends to the cells as a set, the way the coda reads them. Part 07.
- Snap. A trace-free pseudo-token carrier. It writes loser positions from span-restricted pseudo tokens (`lxtul_snap`). Part 06.
- Span budget. The CE gap between a model with and without cross-span information, at fixed depth (`model.span_mask`). Part 03.
- Reach, loop reach. How many earlier slots a coda token or the loop can read (`tul.loop_reach`, `tul.tg_coda_prefix_reach`). Part 03 onward.
- Relay. Content that crosses slots only by passing through the loop, one slot per pass. Part 05.
- Prefix cells, prefix_k. The number of coda positions a slot writes to (`tul.prefix_k`). Two is the default. Part 03.
- E_slot, W_sent, bag mean. Slot-input parts. `E_slot` is a shared learned vector. `W_sent` projects the span's last token. Bag mean is the mean over the span's embeddings. Parts 01, 02.
- Boundary rule. The rule that cuts text into spans at `.;!?`, newlines, and dashes, with no comma. Min span 4, cap 32. Part 01.
- Think once, decode cheaply. The design intent. The loop runs once per span, and the coda decodes tokens cheaply from that state. Part 01.
- Think-once panel. The slot loop as its own job (`tul_to_*`). Part 02.
- Gist. A gisting mask. Tokens see only their span and the slot states. A gradient-carrying write writes the slot. The line has no loop. Part 01.
- Flow-matching planner. A planner that writes span content into a latent through `W_prefix` (`tulfm_p1`). Closed in part 01.
- Graded target. A computed target. The loop's cells condition the frozen coda, which samples K continuations, and a grader ranks them (`tul.code_grade`). Part 04.
- Frozen reference. A frozen deep copy of the model, used to compute the VAE stage's readings (`tul.code_target_ref`). Part 04.
- Frozen coda, adapted reader. A frozen coda reads the cell with no training. An adapted reader is the coda after training on the cell. Part 04 onward.
- Genericity. A frozen coda's depth CE tracks how generic the cell is, not how much the loop earns. Part 04.
- Marginal k16 minus k1. The log mean likelihood under the coda across K draws, compared between 16 and 1 draws. A lower bound on the true log-likelihood. Part 04.
- Semantic probe. A MiniLM cosine between a generated span and the true span (120 cuts). Part 04.
- Distinct-2, seq_rep_4. Generation metrics as the parts report them. Higher distinct-2 is more varied text. Higher seq_rep_4 is more repetition (for example 0.291 against 0.140 under DITTO at T 0.7). Parts 04, 07, 08.
- Coverage loss. A See et al. loss that penalises attention to already-attended positions (`tul.pointer_coverage_lambda`). Failed in part 07.
- Epiplexity. The score behind the `epi` diversity mode (`tul.fan_repel_mode: epi`). The parts use it as a diversity term and do not define it further. Part 05.
- Oracle gap. The CE gap between the best stream and the mixture. Used as the falsifier for fan arms (part 05).
- Selector regret. The CE gap between the oracle over streams and the deployed read. Part 05.
- Ledger CE. The CE on the reported ledger rows. Used in part 06 for latent-selected loops.
- Parcae core, `core_impl: parcae`. A dense bf16 pre-norm single-stream block stack in place of the MORPH looped core. Part 03 onward.
- Parcae injection, diagonal injection. The recurrence's re-injection of the input through a learned B. Parts 03, 08.
- Hyper-connections, HC. The multi-stream residual used in MORPH blocks (n=4 streams). Parts 06, 07.
- Iteration and passes. A pass is one core iteration. Passes are numbered from 1. Part 01 onward.
- Poisson depth, mean_depth. The per-sample loop depth is Poisson with a mean, clamped to a max. A fixed depth removes the draw (`model.depth_fixed`, `slot_depth_fixed`). Parts 02, 04.
- BPTT depth. How many loop passes take the gradient (`bptt_depth`). Full BPTT means every pass. Part 02.
- Carrier, carry. The state passed from one pass to the next. Part 04, 05.
- Oracle z, fitted z. A z fitted to the real span, used as a hindsight target. Parts 03 and 04. Not a valid headroom reading.
- Span decoder target, cell target. Targets in the code and span families. Parts 04, 07.
- TUL-Code, z. The slot holds z, the code of the next span. Part 04.
- XM. Explorative modeling: best of K full generations per slot, selected by distance to E's code (`tul_code_xm`). Part 04.
- CFG. Classifier-free guidance on the thinker (`code_cfg_drop`, `code_cfg_scale`). Part 04.
- EMA target. An exponential moving average of the online front's pooled states, used as the code target (LCTUL-J). Part 05.
- NextLat. A next-latent auxiliary that predicts the next slot's exit (`lxtul-fp01-nextlat`). Part 05.
- LaDiR. A three-stage recipe for a latent code: VAE code, frozen decoder with a thinker, then rollout. Part 04.
- LeJEPA, SIGReg. A regulariser that keeps latent rank from collapsing. Parts 02, 04.
- DiscoTUL, DiscoLoop. The loop's output token head is applied between passes (`tul.disco`). Built in part 07, not trained.
- LoopMTP. A horizon-indexed target on each pass (arXiv 2608.03624). Part 04.
- HRM, Huginn. Looped LMs used as external rulers. Parts 02, 06.
- Toy slot loop, toy eliminate. Small synthetic tasks used to test loop geometry (parts 03, 04).
- Staged target. An own-span target at a middle pass and a next-span target at the exit (`tul.mux_stage_own_iters`). Part 03.
- Progressive loss. The first k passes are detached with some probability per slot (`tul.progressive_p`). Part 03.
- Cancellation. The fraction by which the six passes' updates cancel each other. Part 03.
- Preclip ratio. The ratio of the pre-clip gradient norm to the total. The tripwire is 1e4. Part 02.
- Map, typical gain, fp32 gain. The loop's per-pass Jacobian gain, read by a power iteration or a finite difference. Part 02 onward.
- Core_gain_t0. The gain at the first loop iteration. Part 05.
- Full BPTT. Gradient through every loop pass. The shipped `base.yaml` uses full BPTT. Part 02.
- Eager, fused, graphed, cached. Execution paths. Eager is the plain PyTorch path. Fused uses fused kernels. Graphed uses CUDA graphs. Cached refers to the KV cache decoder. Parts 02, 06, 07.
- Compaction, slot compaction. Dropping pad slots and finished slots from the compute (`model.slot_compact: gather`). Part 07.
- Stratified depth draw. A depth draw that is stratified per row (`model.slot_depth_stratified`). Part 07.
- Cell mean, cell count. The Poisson mean and range of cells per slot (`tul.cell_count`, `tul.cell_count_mean`). Part 07.
- Speed base, speed stack. The recipe with speed keys (fast pointer, then the diet stack). Part 07.
- Diet, FLOP diet. The config cuts that reduce FLOPs: one-block decoder and cell mean 2. Part 07.
- Hinge every 4. The hinge on every 4th step with 4 times the lambda (`model.slot_gain_every: 4`). Part 07.
- Half-slot decoder. Training on a random half of each row's slots (`tul.spandec_slot_keep: 0.5`). Part 07.
- Minimal loop, minimal fan. The Parcae arms with stabilisers and span decoder removed (`arm=minimal`, `arm=minimal_fan`). Part 08.
- Lean arm. A Parcae arm that stacks MORPH's speed-positive levers. Withdrawn (part 08).
- Serial TUL. The Parcae engineering track with a boundary-only thought per span and full cross-span gradients. Part 08.
- Packed thinking frontier, packed coda. Serial TUL throughput forms. Part 08.
- K-sample marginal. See marginal k16 minus k1 above.

## Status when this record was assembled (2026-10-14, evening)

Two later Parcae results are not in part 08. The minimal fan with latent selection is also flat (CE@6 3.7563,
K1-K6 0.0000), so a one-slot predictive target alone does not make the loop use depth. Every Parcae arm that
earned depth also carried epivol, which rewards cell spread at passes 1 and 2 only. A split run (epivol alone
against hinge, fixed point and clip alone) is queued. The working notes for the redesign are in the private
scratch tree, `morph-scratch/abq/drawing-board/README.md`.


---

## TUL history, part 1: origins to 2026-08-31

Scope: the first TUL commit (2026-08-16) through the takeover campaign, "the plan is empty",
MUX, Thought Gestalt, the FM planner, the l2cap recipe and its retraction (2026-08-31).
Sources are the spec files in `.agents/specs/`, the dated filings in `lab/` and `.agents/notes/`,
the commit log, and the existing index `docs/tul-loop-contribution-history.md` (section 4, the first three era sections).
`docs/tul-as-memory.md` was checked and left out. Its content covers September slot-arm
work, so it belongs to a later part.

### Era summary

- **What TUL was meant to be.** Think once per span, decode cheaply. Tokens never enter the
  loop. Text is cut at punctuation into spans. One slot position per span holds a looped
  state. The core runs on slot positions only, and the coda decodes the next span from that
  state. The spec (`.agents/specs/tul-spec.md` section 0, written 2026-08-16) states this
  directly. Wolfe restated the same intent on 2026-09-03 in a later memory note: "This is
  totally wrong, it should be: think once per span, decode cheaply."
- **Aug 16 to 18: first slot-only arms.** `tul_short.yaml` ran a dense baseline against a slot-loop arm.
  slot arm beat dense baseline by 0.0562 nats in one run. That number sat on the retention-carry leak, found
  on 08-23 and fixed on 08-31, so it is qualified.
- **Aug 21 to 24: the takeover.** A span-length gate was built. The slot-loop validation CE
  turned up after step 500 to 2000, while the core took the gradient. About twenty hypotheses
  followed. Per-slot input embeddings were the best lever found, but not a cure.
- **Aug 25 to 28: "the plan is empty."** On one checkpoint, the core was worth 0.0007 nats on
  ordinary token CE. The slot's plan was worth 0.02 to 0.07 nats. Attacks followed: a MUX head,
  Thought Gestalt restriction, auxiliary losses, warmup, SIGReg, and a flow-matching planner.
  On 08-28 the no-slot floor beat the slot model (4.0061 against 4.8434 at step 4250).
- **Aug 29: the gist interface.** A gisting mask with a gradient-carrying write made the coda
  read slot content for the first time. The winning variant beat the unmasked ruler by 0.26
  nats. The loop ladder then said instability, not iteration, was the disease. The sigma cap
  produced a 0.233 depth "win".
- **Aug 30 to 31: the wins fall.** The flow-matching line closed. DB init and the gate both
  stayed depth-flat. The 08-31 carry-leak audit retracted the l2cap depth win: with the carry
  off, the same weights read -1.12 nats. The honest plain loop earned 0.220 nats at 4500 steps.
  On the TUL recipe at 20k, the slot loop earned 0.015, and the no-TUL model beat TUL by 0.357
  nats on tokens.
- **End of era.** Loop earning on the slot path was unproven. TUL stayed merged and off by
  default. Every CE number from 08-16 to 08-30 is qualified by the leak, unless a later
  causal re-run is named.

---

#### 2026-08-16: Slot-only loop, first TUL (`tul_short.yaml`, `tul.tokens_through_core: false`, `tul.activate_at: 0.0`)
- Idea: Spans are cut at `.;!?`, newlines, and dashes, with no comma. Rule: min span 4,
  cap 32, and a run of boundary tokens counts as one boundary. A slot holds `E_slot` plus the
  bag mean of the span. The prelude runs on all positions. The core runs on slot positions
  only, with Poisson depth per slot. The coda reads slots as prefix positions (`prefix_k` 2).
  The boundary rule and packer are in `morph/model/tul_layout.py`.
  Token-state dropout is 0.15. The spec also lists a full-schedule variant (TST switch and
  carve at 30k), kept as config only and never run.
- Why: A span decoded from one vector plus an offset collapses. The Huginn span decoder scored
  7.12 nats (spec section 1). The slot has to hold a latent, and the coda has to read it.
- Result: The dense baseline reached 3.2805 and the slot arm 3.2243 val CE at 20k, so the slot arm is 0.0562 nats better. The
  slot arm run has 10.86 layer passes per token. The "works" gate could not be scored: the replicate
  aborted at step 3240. The first-token counterfactual was -0.1196, so the plan lost to
  the trivial previous-token channel. Source: `lab/tul/arms-result.md`.
- Verdict: Qualified. The result is n=1 per arm and pre-fix, because the GLA retention carry
  leaked future tokens (`.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md`).
  The spec's "works" gate stayed unevaluated.
- Led to: The takeover campaign (08-22 to 08-25), the gate, and the compaction-window screen.

#### 2026-08-17: Token-state dropout as takeover trigger (withdrawn) (`token_state_dropout`)
- Idea: A claim that token-state dropout triggered the TUL divergence.
- Why: Both TUL arms diverged on 08-17.
- Result: Commit `9586452c` named the trigger. Commit `7deb0ef0` withdrew it, because it was
  n=1 per arm.
- Verdict: Retracted on 08-17, the same day.
- Led to: The 08-22 RCA, which looked at the core Jacobian and the ademamix slow EMA instead.

#### 2026-08-18: Compaction-window screen (`tul.coda_token_cut`, eval arms 0 to 3, named by what they keep)
- Idea: At eval, gather-drop token positions before a cut point and keep every slot. Tests
  whether late-token prediction depends on slots. Eval only, no training.
- Why: With tokens still in the coda, the token CE can be satisfied by a bag of the closed
  span. Dropout taxes the cheap channel but does not delete it.
- Result: On `tul-a1-acap1/step_20000.pt`, CE was 3.2985 (everything kept), 3.3439 (early tokens dropped, slots kept),
  3.3530 (early tokens dropped, a random equal-budget token subset kept, no slots), 3.3732 (early tokens and slots dropped). The slots-kept arm beats the random-subset arm by 0.009 nats (95% CI [0.0078, 0.0102]). Stratified re-scoring
  reversed the win on the tokens that need old text. Source:
  `.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md`.
- Verdict: Screen only, not a trained arm. Archived 09-03 when the slot-only code was cut.
- Led to: The arm-D teacher distillation proposal, which was not built in this era.

#### 2026-08-18: Merge as "compute win, not memory" (`tul.activate_at: never` default)
- Idea: TUL merged to master, off by default. The status was corrected to "a compute win not
  a memory one" (commit `64f70a0d`).
- Why: The 08-18 long run failed 2 of 2 seeds, while the three arms survived 0 of 2 (commit
  `80740c7a`). A merge needed a clear status.
- Result: The no-core floor ran at 2.76x the dense baseline's throughput (`lab/tul/arms-result.md`).
- Verdict: Merged, default-off. The compute claim stayed; the loop-value claim did not.
- Led to: The gate bake-off and the divergence program.

#### 2026-08-21: Gated TUL, a span-length gate (`tul_gate.yaml`, `gate_drives_depth`, `tul_halt`)
- Idea: A length head on the slot state chooses how many tokens the next span covers.
  Budget-conditioned coda. A loader augmentation truncates spans at 15% (`truncate_p`).
  The "halt" arm reuses the same weights, with the gate choosing each slot's depth.
  The gate label is the next span's length, not the current one.
- Why: The predecessor project (`00DeepNet/coconut`) had a gate, but it was built for a
  frozen decoder and needed reward. In pretraining the segmentation is known, so a supervised
  label exists. Source: `.agents/notes/rejected/feature/2026-08-21-gated-tul.md`.
- Result: The first bake-off (08-21 to 22) gave no verdict, because the arms died on the
  alpha-cap defect. The second, at batch 12 (08-23): gate val CE 3.3121 against 3.4175 for the
  plain slot arm, a -0.1054 gap. plan_nats was 0.1900 against 0.0045. Halt depth collapsed to
  1.00 on every eval. The matched replicate diverged at step 4160, so the gate result is a
  single seed pair. Source: `lab/experiments/results/2026-08-23-tul-gate-bakeoff.md`.
- Verdict: Rejected 09-03, because the paid loop has no per-slot depth for a gate to drive. The
  -0.1054 is qualified: the pre-fix leak (+0.1433 on the baseline) is larger than the headline.
- Led to: Nothing directly. The span-length idea returned later as the TG and gist work.

#### 2026-08-22: Alpha cap 1.0 as the "divergence fix" (`training.ademamix_alpha_cap: 1.0`)
- Idea: Cap the AdEMAMix slow-EMA coefficient at 1.0 in `tul_short.yaml`, and call it the TUL
  divergence fix.
- Why: Uncapped runs detonated 5 of 5 times.
- Result: Seed 0 held for 20k steps. Seed 1 died at step 4140. The order parameter test
  (`lab/experiments/failures/2026-08-22-tul-order-parameter.md`) asked whether the cap
  suppressed the mechanism or only slowed it.
- Verdict: Not a fix. The core-takeover note says so directly
  (`.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md`).
- Led to: The forward-state collapse analysis.

#### 2026-08-23: Deterministic `bag_mean` (no flag, a one-hot `bmm` replacement)
- Idea: Replace the `index_add_` bag mean with a deterministic one-hot batched matmul.
- Why: Two identical runs diverged. The bag mean's float atomics looked like the cause.
- Result: The operator was non-bit-identical in 20 of 20 repeats. One full step repeated
  non-identically in 4 of 4, with a worst gradient error of 3.92e-2. The fix cut that to 3.17e-3,
  12x better, but not zero. The residue is `tl.atomic_add` in the fused CSA and HCA attention
  kernels. Source: `lab/experiments/failures/2026-08-23-tul-run-replication.md`.
- Verdict: Failure for the replication goal. The run still is not reproducible. The filing's own rule:
  do not proceed on an irreproducible run.
- Led to: Bit-reproducible training mode later in the project.

#### 2026-08-24: Per-slot input embeddings (`tul.per_slot_embed`)
- Idea: Give each slot its own input embedding row, not a shared `E_slot`.
- Why: Four spectral controls had failed to stop the takeover. The slot states sat at effective
  rank 1.7 to 4.8 in 1024 dimensions.
- Result: The same-day note called it "the cure". Seed 1 held. Seed 0 failed at 2225. The
  time to failure doubled, from 1150 to 2225, and the CE was 0.46 and 0.78 nats better on
  the two seeds. Source: `lab/experiments/failures/2026-08-24-tul-takeover-cure.md`.
- Verdict: The best lever, not a cure. The same-day "cure" wording was corrected in commit
  `219a0aa2`. Removed from the tree on 09-03.
- Led to: The campaign pivot to gradient flow (08-25).

#### 2026-08-24: Takeover hypothesis ledger, spectral caps and sinks (`spectral_penalty_cap`, `spectral_penalty_lambda`, `sink_logits`, `bptt_depth`)
- Idea: The campaign tested spectral caps (soft and hard sigma), a spectral gap, a learnable
  attention sink, a positional sink, finer spans (`span_cap` 8), truncated BPTT, the per-iteration
  injection decay, and levers that stack.
- Why: The takeover was read as a core stability problem.
- Result: Four sigma-cap arms all had worse CE rises than the control. Spectral gap was refuted
  (median sigma1 over sigma2 went from 1.069 to 1.132). The learnable sink never engaged. The
  positional sink was "not supported". Finer spans gave a 1.5x delay, not a cure. The forcing-bias
  hypothesis (H19) was refuted. The arm-linked anchor response (H20) survived its first control.
  Source: `lab/divergence/takeover-campaign.md` (hypothesis ledger).
- Verdict: Most hypotheses refuted. Wolfe vetoed varying depth per slot (`slot_mean_depth`) on
  08-24. The campaign file records the reason: it "removes the mechanism rather than fixing it." The campaign index says
  the `bptt_depth 2` arm "helped" by truncating the loop, the same trap.
- Led to: The gradient-flow pivot (08-25).

#### 2026-08-24: Span pooling law, and its same-day retraction (`bag_mean`, span length L)
- Idea: The span bag mean pools slot diversity out as 1/sqrt(L).
- Why: The collapse in slot diversity needed a mechanism.
- Result: The slope was -0.473, -0.504, -0.527 at three caps, with r squared up to 0.93. The
  same file first claimed that the loop "destroys about 10x more diversity than pooling". That
  claim compared two different measures. Source: `lab/experiments/successes/2026-08-24-tul-span-pooling-law.md`.
- Verdict: Law confirmed. The "10x" claim was retracted the same day (`lab/divergence/takeover-campaign.md`).
- Led to: Pooling as one of the causes of slot collapse, tested further in the strict-geometry
  work of September.

#### 2026-08-25: "The plan is empty" measurement (`lab/divergence/region_shapley.py`, `slot_path_worth.py`, `token_state_dropout` sweep)
- Idea: An exact Shapley split over prelude, core, and coda. Then a separate zeroing of the
  plan content only, which removes the slot's written content but keeps its positions.
- Why: The earlier loop-worth numbers could not say whether the loop or the plan carried value.
- Result: At step 1750 (healthy), the core is worth 0.0007 on `ce_main` and 0.3296 on `ce_emit`.
  At step 1850 (taken over), the core turns negative on every metric. The loop alone costs
  0.0051 nats on `ce_main`. The whole plan costs 0.0191 at p 0, and 0.0699 at token dropout p 1.0.
  Prelude and coda are worth about 3.2 and 3.1 nats each. Source:
  `lab/experiments/results/2026-08-25-region-shapley/README.md`.
- Verdict: Standing measurement, pre-fix (the leak is still present in these checkpoints).
  The "plan is empty, not outcompeted" reading holds at this checkpoint.
- Led to: The MUX head (08-25 to 08-27), the TG restriction, the auxiliary-loss tests, and
  the flow-matching planner.

#### 2026-08-25: HCA compressed branch dead on the slot path (HCA branch arms)
- Idea: Test whether the compressed attention branch (HCA) was dead on slot positions, so
  three of six core blocks ran at half attention capacity.
- Why: The H24 hypothesis was the slot arm versus dense baseline asymmetry.
- Result: The defect was confirmed. As a cure, it was refuted: one of two arm seeds diverged,
  the same as the control. Source: `lab/experiments/failures/2026-08-25-h24-hca-branch-arm-4seed.md`
  and related H24 files. The branch was also noted as dead in
  `.agents/notes/rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`.
- Verdict: Defect real, not the cause.
- Led to: The core attention geometry audit on the slot path (commit `f6f9ac25`).

#### 2026-08-25: Standalone "SCSE" initial deviation (stage 1, `Delta_0`)
- Idea: A non-zero initial displacement of the core state, to break the identity at step 0.
  The paper's own primitive was the forcing bias `b_t`.
- Why: Hypothesis H19 and H23 (zero-deviation forcing bias drives the takeover).
- Result: The displacement was real (4.2%). The final CE was 0.815 nats worse on the stage-1
  test. The forcing bias grew on three of four seeds. Source:
  `lab/experiments/failures/2026-08-25-scse-stage1-initial-deviation.md`.
- Verdict: H23 refuted, on its pre-registered condition.
- Led to: The SCSE spec was closed as a takeover mechanism.

#### 2026-08-25: MUX local head, first arm (`tul_v1a.yaml`, MUX head with `mux_activate_at`)
- Idea: A local head predicts the next span's token set from the slot, with the readout
  supervised. Tests whether the slot's own target can carry plan content.
- Why: `emit_weight` was 0.0, so the slot's one-token race had no gradient at all.
- Result: All five runs aborted by step 2800. Plan worth was +0.0088, +0.0189, and +0.0036 at
  500, 2000, and 2500. The readout ratio fell about 7x. The CE tax was about +1.6 nats against
  control. Source: `lab/experiments/failures/2026-08-25-mux-head-arm-v1a.md`.
- Verdict: Failure. The arm was pre-registered, so the failure stands.
- Led to: The detached-MUX variant (08-27).

#### 2026-08-27: Detached MUX head, beta 0.1 (`tul_v1a2b.yaml`)
- Idea: MUX head with the weight-tied readout detached, at weight beta 0.1.
- Why: The MUX head had to be tested again without the gradient path into the readout.
- Result: Loop worth on `ce_main` was +0.0058, +0.0082, +0.0106, and +0.0107 across four seeds at
  3500 steps. The control band was -0.0002 to +0.0042. Permutation p = 0.0286. Source:
  `lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md` (lines 268 to 276).
- Verdict: Post-hoc, and pre-fix. The file itself says the replication has no results. The
  09-26 run history calls it "standing, small, post-hoc". The index (hist-A) disagrees and
  marks it pre-fix. Both readings are in the record.
- Led to: The loop-worth yardstick used in later work.

#### 2026-08-27: Warmup, SIGReg, and NTP dropout panel (`tul_warmup.yaml`, `tul_sigreg.yaml`, `tul_ntpdrop.yaml`, `tul_w1.yaml`, `tul_w2.yaml`)
- Idea: Three attacks on the takeover. Warmup turns the MUX head on at step 1400. SIGReg adds a
  regularizer on the slot state. NTP dropout runs a slice of steps with no slots.
- Why: The campaign's finding was credit assignment, not stability.
- Result: The warmup, SIGReg, and NTP dropout arms all failed except one. The NTP dropout arm held only by removing the thing it protected.
  The one arm that moved every measure was the `tul_v1a2b` control. Source:
  `lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md`.
- Verdict: Failure. None of the three arms stopped the takeover.
- Led to: The objective split (below) and the Thought Gestalt restriction.

#### 2026-08-27: Objective split, which loss writes the core (`_preclip_probe`)
- Idea: Measure which objective sends gradient into the core, and whether the objectives fight.
- Why: Wolfe asked for it: "our real root is gradient assignment on the core."
- Result: O2 failed. O5 matched neither predicted mode. The coda's objective has "almost no
  achievable gain through the slot." Source: `lab/experiments/failures/2026-08-27-objective-split.md`.
- Verdict: Hypothesis refuted. The measurement worked.
- Led to: The restriction tests, which asked the same question with the mask.

#### 2026-08-27: Thought Gestalt restriction (`tul_tg1.yaml`, `tul_tg2.yaml`, `tul.tg_restrict`, GLA `reset_mask`)
- Idea: Restrict token attention to the current span. Cross-span content must go through the
  slot positions. The compressed branch attends only over slots, with a sink. The GLA carry is
  reset at each span start. The design follows arXiv 2512.25026 (Thought Gestalt).
- Why: The shortcut through tokens let the plan stay empty. Closing the shortcut was meant to
  make the slot load-bearing.
- Result: Round one. Predictions P1, P3, and P4 held. P2 failed: loop worth was +0.0060 on
  `ce_main` against a 0.05 line. Round two (`tul_tg2`): loop worth +0.024 to +0.036, below the
  line. Plan worth "rise" was withdrawn as a fallback confound. Source:
  `lab/experiments/failures/2026-08-27-tg-restriction.md`.
- Verdict: Failure on the loop-worth line. The restriction survived, the loop did not earn.
- Led to: The specificity 2x2 (below) and round two (`tul_tg3b`, `tul_tg4a`, `tul_tg4b`).

#### 2026-08-28: Specificity 2x2, auxiliary losses versus restriction (`emit_weight`, `plast_weight`, `tg_restrict`)
- Idea: A 2x2 of auxiliary losses on and off against restriction on and off, measured by
  plan specificity to the span.
- Why: Is the plan empty because nothing writes to it, or because the restriction degrades it?
- Result: Specificity was 65.1%, 0.4%, 3.0%, and 0.3%. The auxiliary losses write the plan. The
  restriction degrades it. Commit `c203318c` corrected an earlier reading. The takeover erased
  plan content: 65.1% healthy against 1.2% taken over. Source:
  `lab/experiments/planned/2026-08-28-plan-span-specificity.md`.
- Verdict: The auxiliary losses carry the content.
- Led to: The reader and target sweeps (plan worth 0.0148 normal, 0.0867 at full token starvation).

#### 2026-08-28: Slot apparatus versus no-slot floor (`tul_a3.yaml`, `n_core: 0`)
- Idea: A no-core, no-slot model at the same budget as the slot model.
- Why: To test whether the slot apparatus pays at all.
- Result: The no-slot model reached val CE 4.0061 at step 4250. The best slot arm reached 4.8434.
  The no-slot floor was 2.38x faster to train. Source:
  `lab/experiments/planned/2026-08-28-does-the-slot-apparatus-pay.md`.
- Verdict: "The slot apparatus is negative" (commit `a959597f`). The loop is unproven.
- Led to: The flow-matching planner, the gist line, and the loop-ladder revival.

#### 2026-08-28: Flow-matching planner (`tulfm_p1.yaml`, `tulfm_p1c_*.yaml`, `tul_fm1.yaml`, `tul_fm1_cw.yaml`, `tul_fm2.yaml`)
- Idea: A flow-matching planner writes the span-(i+1) content into a latent. Then an
  additive `W_prefix` interface passes it to the coda. Stages: P1 (retrieval gate), whitened planner test (whitened
  targets, true conditional flow matching), planner arm (planner in the live model), planner with compaction window (planner
  with the compaction window), and reader-training arm (emit CE as the reader trainer).
- Why: The decomposition named three parts: write, dynamics, and read.
- Result: P1 wrote real content at 2.1x chance, but missed the 0.06 gate. whitened planner test was capped at
  about 2.5x chance. planner arm: plan top-1 was 0.70 against 0.20 for copy, and plan worth was exactly
  0.0000. planner with compaction window: slot positions were worth 0.81 nats, but plan content was worth 0.0001. The
  oracle prefix probe planted the true target at the prefix, and the CE change was -0.0000
  [-0.0001, +0.0001]. reader-training arm: emit supervision was absorbed by the context shortcut, and the copy
  gap fell from 0.47 to 0.26. Sources: `lab/experiments/failures/2026-08-28-tulfm-p1*.md`,
  `2026-08-28-tul-fm1-cw.md`, `2026-08-28-oracle-prefix-probe.md`, `2026-08-29-tul-fm2.md`.
- Verdict: The additive prefix interface reads nothing. reader-training arm closed the training-pressure family.
  On 08-30 the revival on l2cap states failed with clean controls (`2026-08-30-tulfm-p1-l2cap.md`).
  Wolfe asked, "is l2cap compatible with the flow matching, is that a good idea?" The filing
  closed the FM planner line. The index and the rejected arc note both record the closure.
- Led to: The gist line (below). The decision note is `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md`.

#### 2026-08-28: Round-two slot seed and the Thought Gestalt panel (`tul.slot_seed: bag_mean | e_slot | boundary`, `tul_tg3b.yaml`, `tul_tg4a.yaml`, `tul_tg4b.yaml`)
- Idea: Three slot-input seeds. Round two of the Thought Gestalt arms, with the soft restriction
  and no auxiliary losses (`tg3b`).
- Why: To separate the mask from the auxiliary losses (the confound).
- Result: soft-restriction arm held 4 of 4 predictions, and the deficit halved. slot-seed arm falsified prediction B2 and
  the column it justified was withdrawn (commit `d5eb7a0d`). third restriction round diverged 2 of 2. Source: the
  commit log for 08-28 and `lab/experiments/planned/2026-08-28-tg-round2-seed-and-softness.md`.
- Verdict: Mixed. The auxiliary losses still explain most of the effect.
- Led to: The gist interface (08-29).

#### 2026-08-29: Gist mask with gradient-carrying write, first arm (`tul_gl1.yaml`, gisting mask, SIGReg on writes, no loop)
- Idea: The gisting mask lets tokens see only their span and the slot states. A gradient-carrying
  tap writes the slot. SIGReg decorrelates the states. There is no loop in the graph.
- Why: The fix for the FM line was interface surgery. The write has to carry gradient, and the
  alternative must be closed.
- Result: The content gate fired. Shuffling the slots costs 0.06 nats, zeroing 0.15 to 0.18, and
  a wrong value 0.33 to 0.43. Every additive-prefix arm was at or below 0.0006. The price was
  0.34 nats against the twin, so the price gate failed. Source:
  `lab/experiments/failures/2026-08-29-tul-gl1.md`.
- Verdict: Half pass. Slots now carry content. The mask costs too much.
- Led to: gist with MUX supervision and gist with warmup.

#### 2026-08-29: Gist mux write supervision and warmup-then-mask (`tul_gl1b.yaml`, `tul_gl1c.yaml`, `tul_gl1bc.yaml`, `tul_gl1_ctrl.yaml`)
- Idea: gist with MUX supervision adds the MUX write-target loss to the gist arm. gist with warmup warms up unmasked, then
  switches on the mask. gist composition composes the two.
- Why: To reduce the mask's price. Price was partly write supervision, partly curriculum.
- Result: At step 4250, the unmasked ruler (`tul_gl1_ctrl`) was 4.6656, gist arm was 5.0033, gist with MUX supervision was
  4.4047 (0.26 better than the ruler), and gist with warmup was 4.5974. Both gates passed. The gist arm line 2
  replication said MUX is the CE lever and the inversion claim was dead. Sources:
  `lab/experiments/successes/2026-08-29-tul-gl1b-gl1c.md`, `lab/experiments/failures/2026-08-29-tul-gl1-line2.md`.
- Verdict: Success at the filing. The mask's price was write supervision. The "Gist-Slot
  Recipe" doc (commit `707b80d3`) records the shipped code. The line is coreless.
- Led to: The loop ladder, which asked whether the loop could join this line.

#### 2026-08-29: Loop ladder, four arms (`tul_l1.yaml` full BPTT, `tul_l2.yaml` sigma cap 1.5, `tul_l3.yaml` with `tul.db_loop`, and `tul_l1` with `training.optimizer=adamw`)
- Idea: Four arms with the gist baseline. The full-BPTT arm has no cap. The sigma-capped arm adds a hard
  sigma projection at 1.5. The DiffusionBlocks-shaped arm has a detached carry and local
  losses. The AdamW arm is the full-BPTT arm with AdamW.
- Why: Wolfe challenged the claim that iteration was fatal. He said the instability lived in
  runaway activations and gradients, and that the loop itself was controllable. The earlier restriction runs
  never measured rho or state norms.
- Result: sigma-capped arm gave K1 minus K6 = 0.233 nats at 4500 steps, a depth-earned CE on a span-wide plan.
  Source: `lab/experiments/successes/2026-08-29-tul-loop-ladder.md`, verdict section.
- Verdict: Success at the filing, later retracted (see the 08-31 carry-leak entry). The ladder
  itself was sound. Its headline number was not.
- Led to: The "l2cap winning recipe" note, and the wake and DB arms.

#### 2026-08-30: l2cap "winning recipe" and the identity-escape law (`tul_l2.yaml`, `morph/configs/tul_l2.yaml`)
- Idea: The recipe is full BPTT plus a hard sigma cap of 1.5. The "identity-escape law" says that
  every alternative failed because it escaped to the identity.
- Why: Ablations of the cap and truncated BPTT all flattened the depth curve.
- Result: The note called l2cap "the ONE recipe whose loop earns depth (0.233 nats)". It
  also called l2cap "the best substrate that will ever exist."
- Verdict: Retracted 08-31. The 0.233 was a causal leak. The banner is in
  `.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md`. The index
  says five files still carry the claim (root `CLAUDE.md`, the loop-ladder note, the DB-fix
  archive note, the gate-ladder note, and the objective-lines note).
- Led to: The 08-31 leak audit.

#### 2026-08-30: Gate-ladder recurrence gate (`tul_g1.yaml`, `tul_g2.yaml`, `recur_gate: grt`)
- Idea: A Gated Recurrent Transformer recurrence gate (GRT Eqs. 4 and 5, bias +4). The first arm is
  full BPTT with the gate. The second adds the sigma cap.
- Why: The gate ladder asked whether a gate could open the loop.
- Result: Depth-flat. gate arm without cap +0.0002, gate arm with cap -0.0002. The gate-value probe found the gate near its +4
  identity basin, with g about 0.98. Sources: `lab/experiments/successes/2026-08-30-tul-gate-pair.md`,
  `lab/experiments/failures/2026-08-30-tul-gate-value-probe.md`.
- Verdict: Both-fail branch. The state-keyed copy branch "also poisons depth-earning formation."
- Led to: The DB-fix pair and the wake arms.

#### 2026-08-30: DB fix pair and interleave (`tul_dbfix.yaml`, `tul_db_cond.yaml`, `tul_ilv50.yaml`, `tul_l2cap_cond.yaml`)
- Idea: Faithful DiffusionBlocks depth in the slot geometry. The interleave arm mixes the DB and
  standard steps.
- Why: To test DB-style depth inside TUL.
- Result: The DB and cond arms both failed, so the binding both-fail branch fired. The interleave was cancelled.
  The `tul_l2cap_cond` pair gave K1 minus K6 of 0.0127, against 0.233 for l2cap. Source:
  `lab/experiments/successes/2026-08-30-tul-dbfix-pair.md`, `2026-08-30-tul-ilv50-l2capcond.md`.
- Verdict: Faithful DB does not transfer to the TUL slot geometry at this budget. The
  comparator (0.233) is retracted.
- Led to: The wake and cap arms.

#### 2026-08-30: Wake arms, DB init then AdamW (`tul_l3wake.yaml`, `tul_l3wake_cap.yaml`, `tul_l2trunc.yaml`)
- Idea: Start from DB init, then train with standard AdamW and full BPTT. The capped twin adds
  the sigma cap. The truncated arm tests the cap under truncated BPTT.
- Why: Can DB init plus normal training grow a depth curve after the fact?
- Result: The wake arm was depth-flat, but reached 4.2044 pure CE, the best measured pure-CE recipe at
  that point. The capped twin was also depth-flat. The truncated arm gave K1 minus K6 of 0.0085.
  Sources: `lab/experiments/successes/2026-08-30-tul-l3-wake.md`,
  `lab/experiments/failures/2026-08-30-tul-l3-wake-cap.md`, `2026-08-30-tul-l2-trunc.md`.
- Verdict: "DB init is an inert basin for depth." The cap applied after the fact does not wake
  the loop.
- Led to: The 08-31 leak audit.

#### 2026-08-31: Carry-leak audit and retraction of the l2cap win (`model.retention_carry`, carry-off eval)
- Idea: Re-read the same weights with the GLA carry turned off at eval. Run future-leak
  attribution. Retrain the recipe with a causal carry.
- Why: The depth curve of the 30k run gave an impossible val CE of 1.12.
- Result: With the carry off, the 4500-step l2cap checkpoint reads K1 4.622 and K6 5.742, so
  K1 minus K6 = -1.1198 (the claim was +0.2328). Future-leak attribution: a 105% collapse at 4500
  steps and 269% at 30k. The causal re-train (`tul-l2nc`) read +0.0171, flat. The leak was
  3.85 nats at 30k. Sources: `lab/experiments/successes/2026-08-31-carry-leak-audit.md`,
  `2026-08-31-future-leak-attribution.md`.
- Verdict: Retracted. "No recipe in the campaign has been shown to earn depth honestly"
  (carry-leak audit).
- Led to: The loop-killer bisect, and the rule that every depth claim needs a carry-off control.

#### 2026-08-31: Loop-killer bisect, plain loop earns (`notul_bg0c0.yaml`, GLA off, cap off)
- Idea: Build a plain MORPH loop from each addition (GLA, cap, HC), with the causal carry, and
  bisect which addition kills earning.
- Why: Wolfe's question about loop contribution with no TUL.
- Result: The conjunction of GLA and the cap killed earning. The plain loop with GLA off and cap off
  read K1 minus K6 = 0.220 at 4500 steps (K3 minus K6 = 0.017). It saturated by K4. Source:
  `lab/experiments/successes/2026-08-31-loop-killer-bisect.md`.
- Verdict: The first honest plain-loop K-curve. It is pre-ramp and pre-norm-match.
- Led to: The depth-8 regime follow-up, and the 20k head-to-head.

#### 2026-08-31: Causal no-TUL loop worth (causal no-TUL config named in the filing, 4500 steps)
- Idea: Run the no-TUL model, with a causal carry, in two geometries. Measure its loop worth.
- Why: Wolfe asked for "a MORPH with out tul for loop contribution."
- Result: Honestly flat in both geometries. K3 minus K6 = 0.015. Source:
  `lab/experiments/successes/2026-08-31-notul-causal-loop-worth.md`.
- Verdict: "No demonstrated way to make the core contribute at this scale/task."
- Led to: The 20k head-to-head.

#### 2026-08-31: TUL versus no-TUL at 20k on the winner recipe (`tul-20k`, `notul-20k`, winner recipe)
- Idea: A 20k head-to-head on the same recipe, one with the slot loop and one without. It was
  planned at 30k (`lab/experiments/planned/2026-08-30-tul-vs-notul-30k.md`). Wolfe shortened it:
  "20k run then... Lets get this up and running."
- Why: To settle whether TUL helps once the leak is fixed.
- Result: The no-TUL model beats TUL on token CE by 0.357 nats on the last-5 mean (3.489 against
  3.846). The 4500-step gap was 0.096 and widened 3.7x. Core-axis K1 minus K6: TUL 0.015, no-TUL
  0.207. Both arms survived 20k uncapped. Wall ratio TUL to no-TUL was 1.275. Source:
  `lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md`.
- Verdict: The binding rule fired. TUL stays merged but default-off, with no promotion. The
  wall-clock result is confounded by the eager Thought Gestalt kernel path (a 4.6x kernel-efficiency gap).
- Led to: The Raven memory arms proposal (`.agents/notes/proposed/feature/2026-08-31-raven-memory-arms.md`).

#### 2026-08-31: GLA write alignment (`write_shift`, `notul_bws`, FWA paper)
- Idea: Align the GLA write with the next-latent predictor, from the FWA paper arXiv 2608.27763.
- Why: Wolfe: "we will likely try it either way."
- Result: The first prediction (P-A1) was false. The backward loop amplification was real, with the branch inert. Source:
  `lab/experiments/failures/2026-08-31-gla-write-alignment.md`.
- Verdict: Failure on its one predicted branch.
- Led to: The retention-sections arm and the detached-GLA loop arm (commits `34180dd4`, `8f64fc23`).

---

### Retractions and disagreements in this era

- **2026-08-17.** Token-state dropout as the divergence trigger: withdrawn (n=1 per arm).
- **2026-08-18 to 08-23.** slot arm beats dense baseline by 0.0562 nats, and the gate beats slot arm by 0.1054: both are
  pre-fix. The retention-carry leak (+0.1433 on the baseline) sits under both. The campaign index
  marks them qualified.
- **2026-08-24.** "Loop destroys about 10x more diversity than pooling": retracted the same day.
- **2026-08-24.** Per-slot embeddings called "the cure": corrected to "best lever, not a cure."
- **2026-08-29.** The loop ladder's sigma-capped arm K1 minus K6 of 0.233 was called "the first LOAD-BEARING
  loop." Retracted 08-31 (carry-leak audit, -1.1198 with carry off).
- **2026-08-30.** The l2cap "winning recipe" note and its 0.233 claim: retracted 08-31, with a
  correction banner. The index says five other docs still carry the claim.
- **2026-08-30.** Comparators for the DB-fix pair, interleave, and l2-truncated arms were built on
  the 0.233 number, so they are retracted too. Their own numbers (0.0127, 0.0085) stand.
- **Disagreement on the MUX detached arm and on the region Shapley numbers.** The 09-26 run history
  calls them "standing." The index (hist-A) calls them pre-fix and not standing. Both are in the record.
- **Spec deviation, not a retraction.** The spec places a boundary after the LAST token of a
  run. The v1 code places it after the FIRST, because the causal rule cannot see the next token
  (spec section 3.1). The deviation is recorded in the spec.

### Threads left open in this era

- Whether the loop earns on the slot path at all, measured with a causal carry. The 20k
  head-to-head says no at this budget. The index says the lever may be the task, not the wiring.
- The depth-8 regime follow-up for the plain loop (`notul_bg0c0` with `mean_depth=8`, `max_depth=14`).
  It was pre-registered on 08-31 and its result is outside this part.
- The fused Thought Gestalt kernel path. The wall-clock comparison used eager kernels, so the
  speed result is not clean.
- The post-core carrier question. The FM planner was tested on pre-core features only. The index
  says a planner on post-core carriers would be a new design.
- The raven memory arms, the Gated Recurrent Transformer gate with a cap, and the l2cap recipe's
  causal re-baseline (`2026-08-31-l2-honest-rebaseline.md`, planned, not run in this era).
- Whether the gist line (gist with MUX supervision) keeps its win at matched budget and on a second seed. The gist arm
  line 2 replication is in the record, but its verdict is "MUX is the CE lever," and the
  unmasked twin with MUX was not yet run.
- The span-length gate, with the halt arm collapsed to depth 1. It was rejected when the paid loop
  arrived in September, but its single-seed result was never replicated.

---

## TUL history, 2026-09-01 to 2026-09-09

Scope: the paid loop, the slot loop's return as think-once, the slot gain constraint and hinge, the fixed-point term, the detonation and LR ramp as they touch TUL, the mask (TG restriction) arms, and the norm_match ternary rule and its effect on the loop. Sources are the filed experiments under `lab/experiments/`, the Agent Notes under `.agents/notes/`, and the history index `docs/tul-loop-contribution-history.md`. Paths are relative to the MORPH repo.

### Era summary

- Sep 1 to 2: Seed and write-side fixes failed. The loop earned only when the tokens paid for it. Arm `tul_a2` (tokens through the core) earned 0.1685 nats at 5k, and every free-ride arm stayed at or below 0.011.
- Sep 2: The paid loop had a detonation problem. A 1000-step LR ramp (`training.warmup: 1000`) closed it, but the ramp also cut the loop's earning. At 20k the paid arm sat 0.022 nats behind the plain model on matched steps.
- Sep 3: Wolfe shipped the paid loop as the TUL recipe (`base.yaml`). The same evening he called it "totally wrong" because the forward looped the core on every token, which is the opposite of TUL's purpose (think once per span, decode cheaply). The slot-only tree was kept on branch `tul/think-once`.
- Sep 4: The slot loop came back on master, with a gain constraint (`model.slot_gain_lambda`, `slot_cot_clip`). The constraint stopped the detonation. The stable slot loop was "stable and empty": 0.18 nats behind a coreless floor, with no token-side loop earning.
- Sep 4 to 8: The arc panels (E0 to E18) tested every lever on the loop's dependence and width. Mask arms (`tg_restrict`) were stable but worse than plain. Depth earning was a dependence measure, not a value measure. Olympiad and Sudoku did not change the verdict.
- Sep 7: The terminal fixed-point term (`model.core_fixed_point_lambda: 1.0`) held the early detonation on the plain loop with no ramp (0 of 6 against 4 of 7 controls). It shipped in `base.yaml`.
- Sep 9: The ternary scale rule `training.ternary_scale_mode: norm_match` lifted the plain loop's per-pass strength. The paid loop under norm_match was a side result. Wolfe called it "a hallucinated TUL" and said the real TUL is the slot loop. Under norm_match the slot loop stayed flat.

### Entries

#### 2026-09-01: Span-pooled compressed branch (`tul.tg_span_comp`, `tul_sac`)
- Idea: Add a compressed global branch that attends to per-span mean-pooled K and V inside the TG restriction. Slots stay in the sequence.
- Why: The mask surgery (see the next entry) showed that the deleted pooled branch was worth about 0.23 nats on a trained plain model. The hope was that restoring it would recover the TUL gap.
- Result: Eval worth 0.187 nats for the branch. It recovered 0 of the 0.357-nat gap to plain. The loop's earning fell from 0.015 to 0.002. Source: `lab/experiments/failures/2026-09-01-span-aligned-compression.md`.
- Verdict: Rejected. Post-hoc surgery on a trained model did not transfer to training from scratch. The model trained under either compressor reached the same token CE.
- Led to: Learned gated pooling (next entry), then the seed ladder.

#### 2026-09-01: Learned gated span pooling (`tul.tg_span_gate`, `tul_sacg`)
- Idea: One zero-init gate per head inside each span. The softmax over the span's tokens weights the pooled K and V.
- Why: Mean pooling dilutes salient tokens. A learned gate might concentrate on informative tokens.
- Result: +0.031 nats on the last five evals and +0.034 on K6. Both are inside the noise band. Source: `lab/experiments/successes/2026-09-01-gated-span-pooling.md`.
- Verdict: The lane closed. Pooling quality was not where the gap lived. Recorded as a success because the prediction was a calibration test.
- Led to: The write-side (seed) work.

#### 2026-09-01: Mask surgery on the trained plain model (eval only, no config)
- Idea: Apply TUL's visibility pattern (the TG restriction, `_tg_slot_attention` over carrier positions) to the trained `notul-20k` checkpoint at eval, without retraining. Compare variants that restore the compressed branch.
- Why: To separate information deleted by the visibility rule from TUL's mechanism failing.
- Result: Masking the trained model cost 0.487 nats. The realized gap was 0.357. So TUL's mechanism already recovered about 0.13 nats over naive masking. About half the loss was the global branch and half was the window. Source: `lab/experiments/successes/2026-09-01-mask-surgery-decomposition.md`.
- Verdict: Kept as the decomposition. The file says "the write channel carries the program", which set the next seed experiments.
- Led to: The bound-seed rank check and the write-side ladder.

#### 2026-09-01: Bound superposition seed (`tul.py::bound_seed`, `tul_w2`)
- Idea: Per-offset frozen random orthogonal rotations before the span sum (HRR-style binding), so summands decorrelate.
- Why: The slot seed `E_slot + bag_mean` has effective rank 1.7 to 4.8 and pairwise cosine +0.39 to +0.71.
- Result: Without `E_slot`, the bound seed has rank 38.3 against 3.5 for the bag-mean (an 11x lift). With `E_slot` added, every seed collapses to about rank 1.1 to 3.1. Source: `lab/experiments/failures/2026-09-01-bound-seed-rank.md`.
- Verdict: The binding works, but the shared `E_slot` vector dominates the norm. This explains the old "seed moves CE by 0.003" finding.
- Led to: The write-side ladder.

#### 2026-09-01: Write-side ladder, free-ride seeds (`tul_r0`/`tul-r0-5k`, `tul_w1`, `tul_w2`)
- Idea: Three slot seeds on the same recipe: the boundary seed (R0), a content seed without `E_slot` (W1), and the bound seed (W2). The question was whether restoring rank lets the core earn depth.
- Why: The slot-only core earned about 0.015 nats at 20k while the plain looped tokens earned 0.207.
- Result: Loop earning K1-K6 of 0.0113 (R0), 0.0007 (W1), 0.0019 (W2). The notul twin with paid tokens earned 0.1937 at 5k. Source: `lab/experiments/failures/2026-09-01-write-side-ladder.md`.
- Verdict: No seed mode moves the loop's earning. "The core ignores its input rank." Rank was not the lever. Wolfe's preferred competing hypothesis (the loss does not reward a free-ride loop) held.
- Led to: The paid-axis arm (next entry).

#### 2026-09-01: Paid loop, tokens through the core (`tul_a2`, `tul.tokens_through_core: true`)
- Idea: Run tokens and slots through the same per-sample core. Slots stay in the sequence, and the loop depth is per sample (Poisson). Four deltas from `tul_g0c0`: `tokens_through_core` on, `tg_restrict` off, `mux_beta` 0, and fused kernels on.
- Why: The free-ride arms could not earn. The paid arm tests whether payment is the missing piece.
- Result: K1-K6 0.1685 at 5k against 0.0113 for R0 and 0.1937 for the paid notul twin with no slots. Best absolute CE of the five 5k arms (K6 4.1711, final val 4.2315). Source: `lab/experiments/successes/2026-09-01-a2-paid-loop.md`. Two of four paid 5k draws detonated (per `.agents/notes/rejected/architecture/2026-09-02-tul-paid-loop-recipe.md`).
- Verdict: Success. "The loop earns when paid." Costs about 44 passes per token, which gives up TUL's 1.6x compute win.
- Led to: The future-leak probe, the detonation work, and the 20k pair.

#### 2026-09-02: Future-leak probe on the paid loop (eval, no config)
- Idea: Corrupt every token after a cut and check whether CE before the cut moves.
- Why: The paid loop reads the whole row. A leak would fake the earning.
- Result: CE before the cut moved by exactly 0 nats in bf16, at both checkpoints and three depths. Earning was 0.12 at 2500 and 0.17 at 5000, and it grew with training. Source: `lab/experiments/successes/2026-09-02-a2-future-leak-probe.md`.
- Verdict: Success. The paid loop is causal.
- Led to: The Jacobian ladder (next entry).

#### 2026-09-02: Paid loop Jacobian ladder (`morph/training/core_jacobian.py` probe)
- Idea: Measure the core step's power-iterated gain over checkpoints and track the alignment of the top directions.
- Why: The paid runs detonated. The question was whether the core map was expansive.
- Result: The healthy map is expansive from the start (typical gain 1.05 at 2500 and 1.13 at 5000). Sick checkpoints show a low-rank blow-up (gain 9 to 159 across the step) while every block stays at 1.0 to 1.3. Source: `lab/experiments/failures/2026-09-02-a2-core-jacobian-ladder.md`.
- Verdict: Failure by protocol (the alignment prediction missed its prior), but the reading is clear. Wolfe's phrase "the aggressive minimum is unstable" was measured here.
- Led to: The ramp and the Jacobian-based gain work.

#### 2026-09-02: Ternary off on paid draws (`training.ternary=false`, `tul_a2`)
- Idea: Turn ternary QAT off and rerun the paid arm.
- Why: To test whether ternary QAT is the trigger for the detonation.
- Result: Ternary-off draws were 3 of 3 healthy against a base rate of about 70% detonations. Source: `lab/experiments/successes/2026-09-02-ternary-off-paid-draws.md`.
- Verdict: Success at the rate level. The named confound is that ternary-off changes 48% of the parameters. The frame-level capture is the next entry.
- Led to: The M2G onset capture.

#### 2026-09-02: M2G onset capture (`tul_a2`, frame-level)
- Idea: Capture the optimizer state at a detonation frame.
- Why: To test the stale second-moment explanation for the detonations.
- Result: No stale-m2 signature. The disease is the ternary cusp "vault" compounding under clipping. The existing Task #276 cure was active and did not stop it. Source: `lab/experiments/failures/2026-09-02-m2g-onset-capture.md`.
- Verdict: Failure of the predictions. This is the decisive mechanism finding for the ternary trigger.
- Led to: The gamma EMA and freeze tests (next two entries).

#### 2026-09-02: Ternary scale EMA, beta 0.99 (`training.ternary_scale_ema_beta=0.99`, `tul_a2`)
- Idea: Slow down the ternary scale gamma with an EMA.
- Why: Test whether gamma contagion triggers the detonation.
- Result: Detonated 2 of 3. Cost the healthy draw 0.30 nats. Source: `lab/experiments/failures/2026-09-02-gamma-ema-paid-validation.md`.
- Verdict: Rejected. The 20k run was not launched.
- Led to: The freeze test.

#### 2026-09-02: Ternary scale freeze (gamma pinned at step 0, `tul_a2`)
- Idea: Hard-freeze gamma at the step-0 mean |W|.
- Why: The discriminator for the EMA test.
- Result: Detonated 3 of 3 and never learned. Val stayed flat near 7.4. A frozen gamma caps every effective weight, so the network cannot grow its norm. Source: `lab/experiments/failures/2026-09-02-gamma-freeze-discriminator.md`.
- Verdict: The "gamma exonerated" branch is not earned. The live per-forward gamma is load-bearing.
- Led to: The warmup test.

#### 2026-09-02: Paid loop with restricted core, the A2s arm (`tul_a2s`, `tul.tg_restrict: true`)
- Idea: Thread the TG restriction (same-span-or-slot attention) through the paid core.
- Why: A cheaper paid loop. The earlier plan named it as the efficient hybrid.
- Result: Detonated in both draws before step 2000. No number. Source: `lab/experiments/failures/2026-09-02-a2s-restricted-paid-loop.md`.
- Verdict: Cannot train at these flags. The restriction may aggravate the instability (n=2 each, suggestive only).
- Led to: The mask arms in later panels were run only on the slot loop.

#### 2026-09-02: Capped paid loop, planned (`tul_a2c`, `training.spectral_project_cap=1.5`)
- Idea: The l2cap spectral projection (sigma_max at most 1.5) on the paid loop, as a contingency.
- Why: The contingency for a failing 20k run.
- Result: Never run. The trigger was "tul-a2-20k attempt 2 fails", and the 20k pair ran healthy. Source: `lab/experiments/planned/2026-09-02-a2c-capped-paid-loop.md`.
- Verdict: Not run. The contingency did not fire.
- Led to: Nothing directly.

#### 2026-09-02: GLA retention under the ramp (`model.retention=true`, `tul_a2`, `training.warmup=1000`)
- Idea: Turn the GLA retention branch on under the ramp.
- Why: GLA added +0.18 nats of cost on the flat schedule in the bisect. The question was whether that cost was the instability tax.
- Result: 0 of 3 detonated. Val 4.5379 against 4.5415 without GLA. Earning 0.049 against 0.047. Source: `lab/experiments/failures/2026-09-02-a2-gla-under-warmup.md`.
- Verdict: Inert under the ramp. GLA stays in the tree but off by default. Its capability question (long context) is separate.
- Led to: Raven (a GLA variant) as a later candidate.

#### 2026-09-02: Warmup 1000 on the paid loop (`training.warmup: 1000`, `tul_a2`)
- Idea: Add a 1000-step linear LR ramp, then flat 1e-4.
- Why: Flat LR detonated about 70% of paid draws in steps 200 to 775.
- Result: At 2500, 0 of 3 detonated, val 4.5391 / 4.5440 / 4.5413 against 4.6776 flat. The paid loop's earning at 2500 fell to 0.046 to 0.049 from 0.121 flat. Sources: `lab/experiments/failures/2026-09-02-a2-warmup-and-seq512.md` and `lab/experiments/successes/2026-09-03-warmup-core-map.md`.
- Verdict: The ramp is adopted (Wolfe, 2026-09-02 23:35: "we update the config to that winner while we are in this branch"). The earning drop is a real cost.
- Led to: The 5k earning check and the 20k pair.

#### 2026-09-02: Warmup paid loop, 5k earning check (`tul_a2`, `training.warmup=1000`, `wandb.name=tul-a2-wu5k`)
- Idea: The same ramped arm to 5000 steps.
- Why: To see whether the lower earning at 2500 was delayed or reduced.
- Result: Reduced, not delayed. Earning 0.058 against 0.1685 for flat A2 at 5k. The depth-1 CE (4.152) beat flat A2's depth-6 CE (4.171). Final val 4.1807 against 4.2315. Source: `lab/experiments/failures/2026-09-02-a2-warmup-5k-earning.md`.
- Verdict: Failure on the earning prediction. The loop is a smaller, saturating share under the ramp, and the recipe is better anyway.
- Led to: The 20k pair.

#### 2026-09-02: The 20k pair on the ramp (`tul-a2-20k-wu` against `notul-20k-wu`)
- Idea: Paid loop against its plain twin, both on the ramp, to 20k, on 480 identical rows.
- Why: The ramp's effect on both arms, and whether the paid loop beats plain at matched steps.
- Result: At depth 6, A2 3.4701 against notul 3.4486 (plain ahead 0.022). At depth 1, A2 3.5738 against 3.4900. Wall clock 16514 s against 12421 s (A2 1.33x). The matched-step deficit closed from 0.132 at 5k to 0.012 at 20k. The earning grew from 0.041 to 0.100 (per `.agents/notes/implemented/architecture/2026-09-02-paid-loop-warmup-recipe.md`). The ramp beat the flat schedule for both arms by 0.092. Source: `lab/experiments/failures/2026-09-02-warmup-20k-pair.md`.
- Verdict: A2 does not beat plain at 20k. Failure by protocol on the "final eval" prediction, because one eval has 0.085 nats of spread.
- Note: The earning at 20k is 0.100 in the ship and recipe notes but 0.104 in the history index (`docs/tul-loop-contribution-history.md`). The two files disagree by 0.004. The filing is the primary source.
- Led to: The ship decision (next entry) and the 40k continuation (planned, not run).

#### 2026-09-03: Ship the paid loop as `base.yaml` (`tul.activate_at: ${training.tst_ratio}`, `tul.tokens_through_core: true`)
- Idea: Make the paid loop the production forward. Cut the slot-only arms from master (commit `223cf85`). TUL switches on when TST ends. Seq 4096, batch 4, prune, carve and route on.
- Why: Wolfe's call on the afternoon of 2026-09-03: "TUL is only slower because of optimizations still to do; it is the winning recipe; ship it to master and cut the arms that did not make it." The 20k pair had put the paid arm 0.022 behind plain.
- Result: The shipped conjunction was measured only as a startup smoke. The 80-step compressed smoke ran at seq 1024 (7.2k tok/s with TUL on, against 49.6k tok/s before it). The seq 4096 smoke ran out of memory. Source: `.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md`.
- Verdict: Reversed the same evening. Wolfe, in the memory note `tul-intent-think-once-decode-cheap`: "This is totally wrong, it should be: think once per span, decode cheaply. I hadn't realized you did this." He said it wasted two days of compute. The 2026-09-03 ship note records the decision but not the reversal. The `rejected/` recipe note (2026-09-02) says the paid loop is rejected as "not TUL" (Wolfe, 2026-09-09).
- Led to: The slot loop returns on branch `tul/think-once`. The paid loop stayed in `base.yaml` through 2026-09-09 (see the "open" list).

#### 2026-09-03: Think-once round-1 panel (`tul_to_a3`, `tul_to_a1`, `tul_to_mown`, `tul_to_mnext`, `tul_to_mnext_mask`, `tul_to_cond4`, `tul_to_coda8`, `tul_to_*_dz`)
- Idea: The slot loop on a job of its own. Ten arms on the winner recipe plus the 1000-step ramp. Arms: a coreless floor (`tul_to_a3`), the original slot loop (`tul_to_a1`), a memory target (`tul_to_mown`), a forecast target (`tul_to_mnext`), the same with a TG mask (`tul_to_mnext_mask`), four extra slot layers (`tul_to_cond4`, plus a frozen-z version `_dz`), and an 8-layer coda (`tul_to_coda8`, plus `_dz`).
- Why: TUL's purpose is "think once per span, decode cheaply". The earlier slot-only arms earned at most 0.015 nats.
- Result: The run was stopped at Wolfe's call on 2026-09-03 22:34 ("we should do this before the queue finishes"). Eight draws finished. Six tripped the tripwire. The memory draw (`tul_to_mown`) was clean. Every forecast draw (five of five) spiked. The A1 draw `tul_to_a1` detonated once in two seeds. No contribution readout ran. Source: `lab/experiments/failures/2026-09-03-tul-think-once-panel.md`.
- Verdict: Failure. The ramp did not cure the A1 takeover and did not touch the forecast spike.
- Led to: The onset capture (next entry) and the slot-loop gain constraint.

#### 2026-09-03: Onset capture on the forecast arm (`tul_to_mnext`, `tul_cap_c0`, `tul_cap_c1`)
- Idea: Capture the forward and backward state at the spike onset. Compare the ternary-off control (`tul_cap_c0`, `training.ternary=false`) with the ternary-on draw.
- Why: The forecast draws spiked on single steps.
- Result: Ternary is necessary (P-C0), but the named trigger (a burst of code flips) is absent. The forward never explodes. The spike is the backward product of eight iterations through a map that contracts a generic direction by about 0.95 and amplifies a few aligned directions about 100x. The typical gain drifts from 0.87 toward 1.00. Source: `lab/experiments/successes/2026-09-03-tul-onset-capture.md`.
- Verdict: Success on the question. None of the three rule branches fired as written. The record says so.
- Led to: The gain constraint, the forward levers (next), and the design note `.agents/notes/proposed/architecture/2026-09-04-loop-contractivity-as-design.md`.

#### 2026-09-04: Slot loop back on master with the gain constraint (`tul_short.yaml`, `model.slot_gain_lambda: 100`, `model.slot_gain_target: 0.9`, `model.slot_gain_eps: 0.02`, `model.slot_cot_clip: 4.0`)
- Idea: Merge branch `tul/think-once` into master. The slot loop (`tokens_through_core: false`, `_tul_core`) comes back, with a penalty on the map's typical gain above 0.9 and a backward clip of 4x the exit cotangent. The knobs print `[slot-levers] ... INERT` on a paid-loop model.
- Why: The slot loop detonated after the ramp in 7 of 7 unregularized draws. The map's gain drifts from 0.87 to 1.00.
- Result: The exact configuration ran 5000 steps with zero spike steps in one draw, and held the fp32 gain at 0.887 to 0.897. Source: `.agents/notes/implemented/architecture/2026-09-04-slot-loop-gain-constraint.md`. The paid loop stayed bit-identical to pre-merge master (`lab/divergence/paid_loop_gate.py`).
- Verdict: Shipped. Wolfe chose a full merge over porting the constraint to the paid core ("merge the branch rather than port"). Known gap: iteration 0's gain still drifts (0.94 to 0.996 over 5000 steps).
- Led to: The forward-levers pair (next).

#### 2026-09-04: Forward levers Y1 and Y2 (`tul_to_mnext_y1` with `model.slot_state_renorm: true`; `tul_to_mnext_y2` with the gain penalty)
- Idea: Y1 pins each slot's carried norm to its entry norm after each iteration. Y2 is the typical-gain penalty above. Both sit on top of the backward clip (`tul_to_mnext_ctt`).
- Why: With the clip alone, M-next detonated at 2764. Its forward inflated (iteration 0's gain went from 1.5 to 9.6).
- Result: Y2 reached 5000 steps healthy. Its gain held at 0.887 to 0.894. Forecast K1-K6 +0.0135 against a stable loop. Token K1-K6 at or below 0.0006. The loop sat 0.18 nats behind the coreless floor on 480 rows. Source: `lab/experiments/successes/2026-09-04-tul-forward-levers.md`.
- Verdict: Success on stability. The slot loop is "stable and empty at this depth and scale". The hinge touched 24.65% of steps, which missed the 30% prediction by the letter.
- Led to: The arc (E0 to E18).

#### 2026-09-04: Backward clip through time (`tul_to_mnext_ctt`, `slot_cot_clip: 4.0`)
- Idea: Bound the loop's carried cotangent at 4x the exit cotangent.
- Why: The backward product was the first visible symptom of the detonation.
- Result: The weight-path gradient still grew 800x while the forward inflated. The map's typical gain went from 0.89 to 8.7. Source: `lab/experiments/failures/2026-09-04-tul-clip-through-time.md`.
- Verdict: Retired as a cure. Kept as an instrument. The disease is the map moving past the edge. Bounding one half alone leaves the other to carry it.
- Led to: The forward levers (Y1, Y2).

#### 2026-09-04: Where depth earns (`tul_a2`-era paid loop, plain loop, per-offset read, eval only)
- Idea: Split the loop's earning by offset in the span and by row difficulty.
- Why: "Hard positions need more passes" was the working idea.
- Result: The paid loop earns 2.5x the plain loop with the same flat shape. Earning is spread evenly across offsets. It is slightly smaller on hard rows (Spearman -0.18). Source: `lab/experiments/failures/2026-09-04-arc-e0-where-depth-earns.md`.
- Verdict: The hard-positions picture is wrong for these targets. Extra passes refine the predictable part of the loss uniformly.
- Led to: The gain-target dial (next).

#### 2026-09-04: Gain-target dial (`tul_to_mnext_y2_g95`, `tul_to_mnext_y2_g98`, `hinge 0.95 / 0.98`)
- Idea: Loosen the gain hinge to 0.95 or 0.98 and see whether the map climbs to it.
- Why: Test whether the 0.9 hinge is the reason the loop is empty.
- Result: The map does not climb to the looser hinge. It settles 0.02 to 0.035 above Y2's 0.894. Token val CE rose by 0.05 (at 0.95) and 0.15 (at 0.98). At 0.98 a spike train appeared. The forecast column moved by under a thousandth of a nat. Source: `lab/experiments/failures/2026-09-04-arc-e1-gain-target-dial.md`.
- Verdict: Failure. The map can be slowed, and the loss does not care.
- Led to: Iteration conditioning (next).

#### 2026-09-04: Per-iteration operator (`tul_to_mnext_y2_iter`, `tul_to_mnext_y2_iter_all`, `tul.core_stage_cond=iter`, `model.slot_gain_all_iters`)
- Idea: Give each loop iteration its own gates, so the passes can differ.
- Why: A shared operator may be the reason the loop converges by pass 3.
- Result: The per-iteration operator earned less than the shared one (K1-K6 0.0087 against 0.0135) at a val cost of 0.027. The earlier K3-K6 +0.0077 was the expansive iteration-0 map (fp32 gain 2.97), and it vanished when every iteration was held under 0.90. Source: `lab/experiments/failures/2026-09-04-arc-e2-iteration-conditioning.md`.
- Verdict: Branch (a) of the arc is closed in both halves. Stability and earning are not the same axis.
- Led to: The mask under the constraint (next).

#### 2026-09-04: Mask arm under the constraint (`tul_to_mnext_y2_mask`, `tg_restrict: true`, eager attention)
- Idea: Add the TG restriction (tokens see their own span and the slots) to Y2, so the slot is the only route across spans.
- Why: The earlier slot arms were free. The mask forces the tokens to read the slot.
- Result: With the route forced, tokens lose 0.31 nats at the span's first token when the slot's content is shuffled, and 0.021 nats when the slot loop is cut from 6 iterations to 1. Token K1-K6 +0.0209. Token and forecast K3-K6 were +0.0002 and +0.0016. The mask costs 0.13 nats of token CE against Y2 on the same rows. Source: `lab/experiments/successes/2026-09-04-arc-e4-mask-under-constraint.md`.
- Verdict: Success on the reader question (the tokens do read the slot). The loop's depth does not earn. The E5 follow-up at 20k was not queued (Wolfe's call).
- Note: Later files call the eager mask gain readings "an absmean artefact" and show that eager hinge readings run 0.07 high (see the 2026-09-08 bug note below).
- Led to: The mask family in E7, E13, E14, E16, E17, and E18.

#### 2026-09-04: Huginn as an external ruler (`huginn`, no MORPH config)
- Idea: Run a recurrent-depth model with a published design on the same web text as an external ruler.
- Why: Test whether web text is depth-flat, or whether MORPH's loops are the problem.
- Result: Huginn earns 0.57 nats between iterations 3 and 6, and 0.21 more between 6 and 16. Source: `lab/experiments/failures/2026-09-04-huginn-loop-contribution.md`.
- Verdict: The binding rule moved the candidate cause to the training regime. Web text is not depth-flat.
- Led to: The fixed-point term, the deep draw (E6), and the ternary rule.

#### 2026-09-07: Terminal fixed-point term, warmup 0 (`model.core_fixed_point_lambda: 1.0`, `notul_fp_wu0.yaml`)
- Idea: Penalize the relative change of the state between the last two core iterations, per sample, at training time only.
- Why: Gain-based penalties read the typical gain but not the spike direction. The ramp is a schedule and cannot act on a map that goes expansive later.
- Result: 0 of 6 draws detonated (max preclip ratio 11 to 18) against 4 of 7 controls (Fisher one-sided p = 0.049). The survivors were 0.41 nats better at 800 steps. The directional gain penalty failed twice (E10b read 1.1 to 1.5 and never fired; E10c detonated 2 of 6). Source: `lab/experiments/successes/2026-09-07-arc-e10-loop-loss-terms.md`.
- Verdict: Success on the stability half. Shipped on Wolfe's call the same day ("ship the fix").
- Led to: The ramped check (next) and the slot-loop port (also 2026-09-07).

#### 2026-09-07: Fixed-point term under the ramp (`notul_fp.yaml`)
- Idea: The same term with the shipped 1000-step ramp, 5000 steps.
- Why: Check that it costs nothing on the shipped schedule.
- Result: CE +0.0009 [-0.0022, +0.0040] against notul on 480 paired rows at trained depth. Depth-1 CE 0.014 better. Loop dependence K1-K6 0.022 against 0.037. Source: `lab/experiments/successes/2026-09-07-arc-e11-fixed-point-ramped.md`.
- Verdict: Free on the plain loop. The term lowers the loop's dependence on depth by improving the shallow readout.
- Open: 20k behavior, the slot path, and the deep draw were not run.
- Led to: The slot loop port (`_tul_core` gets the same term, commit `7ff72a0`, per `.agents/notes/implemented/architecture/2026-09-07-fixed-point-objective-as-the-loop-stability-term.md`).

#### 2026-09-07: Deep recurrence draw, mean 16 (`notul.yaml`, `model.cfg.mean_depth: 16`, sample depth Poisson)
- Idea: Train the plain loop with a deeper Poisson draw (mean 16, max 24, BPTT 8).
- Why: The Huginn reading said a loop earns to about half its training mean.
- Result: The deep plain loop earns K3-K6 0.277 (against 0.0016 at mean 6). But its depth-16 CE is 0.104 worse than the mean-6 model at depth 6, and its depth-1 CE is 0.93 worse than notul. Source: `lab/experiments/successes/2026-09-07-arc-e6-deep-recurrence-draw.md`.
- Verdict: Success. The finding that matters is that K-differences measure dependence on depth, not the value of depth.
- Led to: The block loop (E7).

#### 2026-09-07: Block loop at mean 16, the power-iteration face (`tul_to_mnext_y2_mask` family, `slot_max_depth`)
- Idea: Run the slot loop as a block loop at a deep draw, with the gain hinge.
- Why: To see the failure shape at depth.
- Result: The map sends a 485-norm entry state to 1.5e7 in one iteration (block 5 sigma_max 9.4e4 at iteration 0). After that each iteration is a 2% rotation. The state's effective rank falls from 181 to 9. The random-direction finite-difference hinge read 0.85 to 1.04 while the top direction sat at 95. Source: `lab/experiments/failures/2026-09-07-arc-e7-block-loop.md`.
- Verdict: Failure. Saturation at 3, state-norm growth, and the detonation are one mechanism. The hinge could not see it.
- Led to: The fixed-point term and the carry test (next).

#### 2026-09-07: Multi-token coda, lookahead heads (`notul_mtp4`, `notul_deep16_mtp4`)
- Idea: Add lookahead (MTP) heads at weight 1.0 to the coda.
- Why: A multi-token target might make the loop depend on its own depth.
- Result: It cost 0.34 nats of next-token CE at matched steps. On the deep draw it added 0.26 nats over E6 and detonated where E6 did not. Source: `lab/experiments/failures/2026-09-07-arc-e8-multi-token-coda.md`.
- Verdict: Rejected. It is a stability load on the deep loop, not a contribution.
- Led to: Nothing directly.

#### 2026-09-07: Widen the carry, rho below 1 on every dim (`notul_carry_all`, `notul_carry_ctx_wu0`)
- Idea: Apply Parcae's diagonal rho-below-1 carry to the whole residual, with init decay 0.9 and dt 0.1.
- Why: Parcae's constraint might stop the detonation.
- Result: No effect on the early detonation at warmup 0. Source: `lab/experiments/failures/2026-09-07-arc-e9-widen-the-carry.md`.
- Verdict: Failure. The MORPH trigger is not on the state carry, or not only.
- Led to: The fixed-point term.

#### 2026-09-07: Fixed depth 12, mask arm (`tul_k12_mnext_mask`, `tul_k12_mnext_mask_mtp4`, `tul_k12_a1`, `slot_depth_fixed`)
- Idea: Test the deep regime at a fixed depth of 12 instead of a draw.
- Why: The question was about a deeper draw. The fixed depth removed the draw.
- Result: The masked arm entered the power-iteration face (first-iteration scale x170) and detonated at step 1446. Source: `lab/experiments/failures/2026-09-07-arc-e12-k12-panel.md`.
- Verdict: Rejected run (wrong method). The predictions were not scored.
- Led to: The mean-12 draw (next).

#### 2026-09-07: Mean-12 draw panel (`tul_m12_a1`, `tul_m12_mnext`, `tul_m12_mnext_mask`, `tul_m12_mnext_mask_mtp4`)
- Idea: Use a Poisson draw with mean 12 (not a fixed depth), with the constraint and the fixed-point term.
- Why: To keep the draw and still read depth.
- Result: A mean-12 draw is safe for the three loss shapes that were safe at mean 6. The masked arm's token loop share grew 2.3x (K1-K6 0.021 to 0.049). Its end point matched mean 6 (-0.0008). Past iteration 3 nothing moved (+0.0018 on tokens). Without the mask the loop was a free ride. The MUX loss cost the tokens 0.08 nats. Source: `lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md`.
- Verdict: Failure on the main predictions. The draw changes what the loop carries, not how deep it works.
- Led to: The expansive-dial test (next).

#### 2026-09-07: Expansive dial, gain near 1 (`tul_m12_mask_g102`, `tul_m12_mask_g102_rn`, `model.slot_state_renorm`)
- Idea: Let the map sit at typical gain 1 (hinge 1.02), with and without the renorm.
- Why: Test whether a looser map makes the loop work deeper.
- Result: K3-K6 unchanged at 0.0007. Both arms spiked (3877 and 4639). Token K1-K6 rose from 0.033 (no hinge) to 0.043 (hinge only) to 0.060 (hinge plus renorm). The renorm delayed the spike by about 400 steps. Source: `lab/experiments/failures/2026-09-07-arc-e14-expansive-dial.md`.
- Verdict: Failure. The gain target is a dial on the loop's share of the tokens' work, not on its depth.
- Led to: The Olympiad panels (next).

#### 2026-09-07: Olympiad panel, uniform mix (`tul_oly_a1`, `tul_oly_mask`, `tul_oly_mnext`, `notul_oly.yaml`)
- Idea: Run the slot arms on Olympiad math with a uniform 15k mix.
- Why: Math may need depth more than web text.
- Result: Rejected before any data. The curriculum's stage sum overrode `training.steps`, and the eager mask arm's memory was set by attention over 640 positions. Source: `lab/experiments/failures/2026-09-07-arc-e15-olympiad-panel.md`.
- Verdict: Rejected run. Replaced by the curriculum panel (next).

#### 2026-09-08: Olympiad curriculum panel (`tul_oly_mask`, `tul_oly_mnext`, `tul_oly_a1`, `notul_oly.yaml`)
- Idea: A curriculum from easy to hard math, with the same arms.
- Why: The replacement for the rejected uniform mix.
- Result: The masked arm sits 0.18 nats behind the plain model on clean math (paired). Its depth 1 is broken (1.31 nats against plain 0.76). The arm "finishes by iteration 3". A1 ties plain to 0.000 +/- 0.010. Source: `lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md`.
- Verdict: Failure. The slot loop does not help solve math. The predictions P16c and P16d are false.
- Note: The Olympiad eval holdout overlaps training (41% verbatim per the memory note `olympiad-holdout-contaminated`, and `.agents/notes/proposed/bug-fix/2026-09-08-olympiad-holdout-contamination.md`). These math readings carry that caveat.
- Led to: Sudoku (next).

#### 2026-09-08: Sudoku depth grid (`tul_sud_a1`, `tul_sud_mask`, `tul_sud_mnext`, `notul_sud`)
- Idea: A task that needs serial constraint propagation and cannot be memorized, with depth T from 2 to 16.
- Why: A task that really needs iteration.
- Result: The plain model reaches 0.938 answer-token accuracy. Every bucket is flat from T = 2 to T = 16 on both arms. The masked arm is 0.006 behind at depth 1, flat after 2, and 0.155 behind plain at its trained depth. Source: `lab/experiments/successes/2026-09-08-arc-e17-sudoku-depth-grid.md`.
- Verdict: Success on the question. The architecture does not iterate on any data tried. The slot-loop lane stays closed.
- Led to: The width sweep (next).

#### 2026-09-08: Slot write width sweep (`tul.prefix_k` 2, 4, 8; mask arm, absmean)
- Idea: The number of coda cells a looped state is written into (`prefix_k`), on the masked arm.
- Why: The width of the slot's write might limit the loop's share.
- Result: K1-K6 +0.0243, +0.0314, +0.0486 for k2, k4, k8 at 5000 steps (absmean). The depth-12 mask arm sits 0.35 nats behind notul at depth 6. Width changes the scores by under 0.005 nats. Plain notul at 5k reads K1-K6 +0.0203 and K3-K6 0.0000. Source: `lab/experiments/results/2026-09-08-arc-e18/score_e18.txt`.
- Verdict: No filed verdict. The planned file has no Results section, and the results folder has no success or failure file. The later arc plan calls the width curve "an absmean artefact".
- Led to: The norm_match panels (2026-09-09).

#### 2026-09-08: Eager hinge reads high (`tul_oly_mask`, bug note, `model.slot_gain_eps`)
- Idea: Not a design. A measurement bug in the gain hinge's finite difference.
- Why: The eager path and the fused path read different gains at the same weights.
- Result: At `slot_gain_eps` 0.02, eager reads 0.94 (sd 0.014) and fused reads 0.87 (sd 0.002). Both converge near 0.870 at larger eps. Source: `.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md`.
- Verdict: Proposed fix, not built. Every eager TG arm's gain reading (E4, E13, E14) is biased high by about 0.07.
- Led to: Nothing built yet.

#### 2026-09-07: Truncated BPTT silences shallow samples (bug note, `bptt_depth`)
- Idea: Not a design. A gradient-window bug. Both loop paths run a no-grad prefix shared by the whole batch, so a sample of depth at or below that prefix gets no loop gradient.
- Why: Found while reading the E6 and E7 deep-draw runs.
- Result: Simulated silent fractions: 27% of plain samples and 57% of slot samples at mean 16, max 24, BPTT 8. Invisible at full BPTT (the shipped `base.yaml`). Source: `.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md`.
- Verdict: Proposed fix, not built. It affects any run with `bptt_depth` below max depth, including the E6, E8, and E12 draws.
- Led to: Nothing built yet.

#### 2026-09-09: Coda reads the thought (`tul_slot_unpack_norm_match`, `tg_scoped_kernels`)
- Idea: Wire the coda so that its only route to earlier spans is the slot state z. Remove the global prelude state from the coda's input.
- Why: Wolfe on the flat curves: "the design should be that the prelude evolves the loop state, and the final z + MUX tokens are what conditions the coda. This skipping will cause the coda to want to ignore the loop." The spec's global prelude state let a token read all of span i directly.
- Result: The reader reads z (0.811, above 0.5). Token K1-K6 was flat (0.0006). The coda depends on z, not on loop depth. Under scoped kernels the path ran at 12,638 tok/s and 15.0 GB. Source: `lab/experiments/failures/2026-09-09-arc-coda-reads-the-thought.md`.
- Verdict: Failure on the depth prediction. The coda depends on z and not on depth.
- Led to: The slot-core anatomy and the norm_match panels.

#### 2026-09-09: Norm-matched ternary rule (`training.ternary_scale_mode: norm_match`, `morph/model/ternary_rule.py`)
- Idea: Keep the symmetric codes and set the per-tensor scale to ||W||_F / sqrt(nnz), so the ternary weight has the latent's Frobenius norm.
- Why: The absmean rule gives a ternary core about 0.44 of its bf16 twin's emission at the scale alone. The core's branch emits 0.19 to 0.24 of its input at pass 6, so the state sits on a fixed point by pass 3.
- Result: Plain loop at 5k: K1-K6 0.185 [0.182, 0.188], K3-K6 0.0139, MLP out/in 0.86 to 1.10. The bf16 core diagnostic reads 0.168. The absmean base reads 0.033. The dead-zone arm and the learnable-scale arm did not move the loop. Source: `lab/experiments/successes/2026-09-09-arc-per-pass-strength.md`, and the Agent Note `.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md`.
- Verdict: Shipped as the recipe's ternary rule (`base.yaml`, `scale30b.yaml`). Wolfe: "the norm matched seems to be our solution ... flip base.yaml to norm_match."
- Open: The 5k value conflicts with the next entry (see the note there).
- Led to: The norm_match panels and the 20k run.

#### 2026-09-09: Norm-match 20k and the horizon (`notul_norm_match_20k`, `notul_horizon_ternary_25k`, `tul-absmean` resume)
- Idea: Run the norm_match plain loop to 20k, and resume the absmean run from 10k to 20k.
- Why: The norm_match CE at 5k was 0.08 worse than absmean. Does the gap close with training?
- Result: Norm-match K1-K6 at 5k +0.1363 in this run, against +0.185 in the per-pass panel at 5k. At 20k it is +0.1702 [0.1675, 0.1729], K3-K6 +0.0156. Paired CE at depth 6 is -0.0100 against absmean (norm_match is better). But final val_loss is 3.5037 against 3.4575 for absmean (norm_match is worse). Source: `lab/experiments/failures/2026-09-09-arc-norm-match-recipe-reads.md`.
- Verdict: Failure on one prediction. The 5k discrepancy (0.136 against 0.185) and the final-val versus paired-CE disagreement are not resolved in the files. The absmean loop's contribution was flat from 5k to 10k (`failures/2026-09-09-arc-horizon-ternary-25k.md`).
- Led to: The paid loop under norm_match (next).

#### 2026-09-09: Paid loop under norm_match, side result (`tul_norm_match`, `tul_absmean`, `tul.activate_at 0.0`)
- Idea: Run the paid loop under norm_match and absmean, 5000 steps each.
- Why: Wolfe asked for "the paid TUL loop under norm_match". This was the wrong target.
- Result: Paid loop K1-K6 +0.1200 [0.1177, 0.1223] at 5000, K3-K6 +0.0116. Paid minus plain at depth 6: +0.1252 (the paid loop is worse). The absmean control was killed at about step 800. Source: the same file.
- Verdict: Side result only. Wolfe (2026-09-09): "that is wrong. We need to test the real TUL, that is a hallucinated TUL you made in a previous session." The real TUL is the slot loop (memory `tul-means-the-slot-loop`).
- Led to: The slot loop under norm_match (next). Wolfe also said: "No paid TUL once we get slot TUL working and our variant finalized" (memory `tul-finalize-no-paid-loop-on-main`).

#### 2026-09-09: Slot loop under norm_match (`tul_slot_loop_norm_match`, `tul_slot_mux_norm_match`, `tul_slot_loop_absmean`, `tul_slot_mux_absmean`)
- Idea: The real TUL (slot loop, `tokens_through_core: false`) on both rules, with and without the MUX loss.
- Why: The slot loop had never run under norm_match. Wolfe's stated priority: "slot loop norm match and slot loop + MUX norm match".
- Result: All five arms healthy. Forecast K1-K6 0.0067 against the absmean twin's 0.0069. K3-K6 0.0005. The gain estimate read 0.868 to 0.887, and the hinge bound on under 1% of steps. Source: `lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md`.
- Verdict: Failure on the loop-earning predictions. The rule does not lift the slot map's earning. The constraint is inert under norm_match, as under absmean.
- Led to: The 2026-09-10 panels (not in this range).

#### 2026-09-09: Parcae-style loop entry (`notul_parcae_entry`, `notul_e18`, `notul_depth1.yaml`)
- Idea: Build the plain loop's entry the way Parcae does: state from noise, with the input re-injected through a learned B. No fixed-point term.
- Why: Test the entry as the reason the loop converges by pass 3.
- Result: The loop still converges by pass 3 and earns 0.033 past pass 1, against Parcae's 0.29. The model is 0.023 nats better at the same wall clock. The depth-1 twin sits within 0.02 nats of the looped model at depth 6 (5k). Source: `lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md`.
- Verdict: Failure. The entry is not the lobotomy.
- Led to: The depth candidates panel (next).

#### 2026-09-09: Depth candidates panel (`notul_depthcand_carry_lr20x`, `notul_depthcand_dense_core`, `notul_depthcand_draw8_bptt4`)
- Idea: Parcae's depth schedule, a carry at 20x the learning rate, and a bf16 core with the rest unchanged.
- Why: Find the lever that made the loop contribute.
- Result: Parcae's schedule gave K3-K8 0.002. The fast carry collapsed the carrier to rank 6 and cost 0.037 nats. The bf16 core moved the loop's value past pass 1 from 0.033 to 0.168, and past pass 3 from 0.0009 to 0.012. Source: `lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md`.
- Verdict: Failure on the candidates. The ternary core limits the loop's contribution at this horizon. This led to the norm_match rule.
- Led to: The norm_match panels.

#### 2026-09-09: Block density panel (`notul_density_half`, `notul_density_quarter`)
- Idea: Block pruning as a lever on loop contribution.
- Why: The prune schedule was in the recipe.
- Result: The quarter-density arm has a larger K1-K6, but its per-pass map is weaker (branch out/in 0.058) with a rank-4 carrier. K3-K6 stays under 0.006. Source: `lab/experiments/failures/2026-09-09-arc-density-panel.md`.
- Verdict: Failure. Dependence without computation. Block prune is not a loop lever.
- Led to: The precision axis (planned, not run).

### Threads left open in this era

- The paid loop is still the shipped forward in `base.yaml` (`tul.tokens_through_core: true`) as of the 2026-09-09 filings. Wolfe asked for it to come out once the slot TUL variant is final. The records also disagree on whether it was a shipped recipe or a hallucinated variant.
- Think-once panel readouts were never run. Healthy arms (`tul_to_a3`, `tul_to_a1`, `tul_to_mown`) have step 2500 and 5000 checkpoints. Their depth sweeps and worth profiles are still runnable (GPU time at Wolfe's call).
- E3 (staged targets, memory for iterations 1 and 2, forecast for the rest) is planned in `lab/experiments/planned/2026-09-04-arc-e3-staged-targets.md`. It was not run.
- E5 (the 20k matched wall-clock run for the mask arm) is not queued (Wolfe's call).
- The E18 width sweep has results on disk (`lab/experiments/results/2026-09-08-arc-e18/`) and no filed verdict. The planned file has no Results section.
- The 40k continuation of the 20k pair is in `lab/experiments/planned/2026-09-03-warmup-pair-continue-40k.md`. It was not launched.
- The capped paid loop (`tul_a2c`) is planned, not run.
- The paid loop under norm_match at 20k, and the paid loop with the constraint, are unmeasured.
- The norm_match 5k discrepancy: 0.136 in the 20k run's 5k checkpoint against 0.185 in the per-pass panel. The final-val and paired-CE readings at 20k disagree in sign. No file resolves it.
- The gain constraint's iteration 0 still drifts (0.94 to 0.996 over 5000 steps). The fixed-point term at lambda 1.0 has not been swept (no other lambda was run).
- The truncated-BPTT fix and the eager eps fix are proposed, not built. Both change past readings of the slot-loop arms.
- The slot loop's gain constraint (`slot_gain_lambda` 100) has been shown inert under norm_match. Whether it should stay on under norm_match is open.
- The precision axis (`lab/experiments/planned/2026-09-09-arc-precision-axis.md`) is planned, not run.
- Out of TUL scope, noted only: `lab/experiments/failures/2026-09-05-huginn-temporary-power-cap.md` is a hardware validation that was blocked by admin authentication. Its CUDA resume is unverified.

---

## TUL history, 2026-09-10 to 2026-09-13

### Era summary

1. The slot loop read flat on every map lever tried (2026-09-10). The question moved from "what pins the map" to "the core does not compute on slot states". Parcae core swap reads the same flat curve, so the core is cleared.
2. Credit assignment came next. Gradient probes showed the six passes cancel. Four loop-side arms (noise entry, progressive loss, per-pass LoRA, MUX on every pass) read flat. The staged target (own-span at the middle pass) reproduced the toy's signatures, but its exit stayed at one pass of the ruler.
3. Under the shipped `norm_match` rule, the old mask K-curves turned out to be an `absmean` artefact. The HCA dead branch was fixed and shipped (2026-09-11).
4. The span decoder (2026-09-11) moved the slot channel. Wolfe's target (decode the whole next span) worked, but the gain rode cells the coda reads, so it bypassed the loop. Strict geometry closed that bypass for free (2026-09-12). Only reachability forced passes to matter, and that was a blind relay.
5. By 2026-09-12 every energy, planning, parallel-head, critic and core-token lever left passes past the first at 0.002 or less. Fitted-z headroom (0.9 to 2.6 nats) was shown to be hindsight.
6. On 2026-09-13 the rank of the slot state was the new suspect. The rank headline (5.76) was retracted to 13.85 on the strict ruler. Register, VQ, cond4 and trajectory arms moved rank or separation but not the K-curve. Width, not the loop, was the cheap CE lever. Wolfe then said "I am not married to TUL" and the depth ladder became the lane.

---

### Shared baselines for this era

- Recipe: seq 1024, batch 6, seed 1, 5,000 steps, `norm_match`, `core_fixed_point_lambda` 1.0, 1,000-step ramp.
- Ruler at 5k (`slot-mux-norm-match`): token K1-K6 +0.0001, `mux_local` K1-K6 +0.0067, CE@6 4.3290, cancellation 0.520, 52 min 38 s.
- Plain depth-6 Parcae-style reference at 5k: K1-K6 about +0.033. Plain loop on the same corpus: K1-K6 0.1849.
- Strict ruler (`slot-spandec-strict`) after 2026-09-13 correction: row rank 13.85, cosine 0.520.

---

### 2026-09-10 entries

#### 2026-09-10: Slot map levers (`tul` `unpack` base, `core_gain_lambda` 0, `core_fixed_point_lambda` 0, `core_state_init: noise`, `slot_depth_fixed`)
- Idea: three one-factor arms on the unpack arm (coda reads a frozen slot vector z). Stability terms off, Parcae noise entry, and a fixed depth of 6.
- Why: z is used (zeroing it costs 0.811 nats) but the depth curve is flat. Something pins the map near identity.
- Result: terms off K1-K6 +0.0030 (CE@6 +0.030 worse), noise entry K1-K6 +0.0019 (CE@6 -0.007), fixed depth 6 K1-K6 +0.1168 with K3-K6 +0.0067 and worth-zero 0.947 (CE@6 +0.052). Core blocks output 2 to 20 percent on slot states against 86 to 110 percent on token states. Source: `lab/experiments/failures/2026-09-10-arc-slot-map-levers.md`.
- Verdict: failure. Stability terms and entry are not the pin. The fixed-depth gain is depth dependence at a CE price, not useful depth.
- Led to: the reread arm, then credit-assignment probes.

#### 2026-09-10: Reread (`tul.reread`, `reread_heads` 8, `reread_scope: causal`)
- Idea: the looping slot cross-attends to frozen prelude token states at the start of every pass (4.2M params, zero-init `W_o`).
- Why: the paid loop re-reads tokens each pass and earns depth. The slot pass cannot see a token.
- Result: token K1-K6 +0.0003, K3-K6 -0.0000, CE@6 4.4955 against unpack 4.4971. Read adds a constant 5 percent of state each pass. Source: `lab/experiments/failures/2026-09-10-arc-slot-reread.md`.
- Verdict: failure. The read is used and changes nothing the coda reads. Design note filed as rejected: `.agents/notes/rejected/architecture/2026-09-10-the-slot-rereads-its-span.md`. The knob stays default off until TUL finalize.
- Led to: credit-assignment arms.

#### 2026-09-10: Noise entry on M-next (`tul_slot_mnext_noise_entry`: `core_state_init: noise`, `injection_channels: all`, `injection_B: true`)
- Idea: Parcae noise entry with both stability terms off, on the M-next MUX ruler.
- Why: the levers panel showed noise entry does not move the unpack map. Does a forecast-loss reader change that?
- Result: token K1-K6 +0.0004, `mux_local` K1-K6 +0.0091, CE@6 4.2876, 0.83x wall clock. State moves (relative distance 0.48 to 0.57). Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-noise-entry.md`.
- Verdict: failure. The entry is coda-independent. A moving state changes nothing.
- Led to: progressive and per-pass LoRA arms.

#### 2026-09-10: Progressive loss (`tul.progressive_p: 0.5`)
- Idea: Bansal-style. With probability 0.5 a slot's first k passes are detached from the gradient.
- Why: the six passes' updates cancel (0.520). Maybe each pass only sees states its own trajectory built.
- Result: token K1-K6 +0.0006, `mux_local` K1-K6 +0.0088, cancellation 0.497, CE@6 4.2938. Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-progressive.md`.
- Verdict: failure. Flat.
- Led to: per-pass LoRA.

#### 2026-09-10: Per-pass LoRA (`tul.pass_lora_rank: 32`, targets attn and mlp)
- Idea: Relaxed Recursive Transformers. Each pass gets its own rank-32 delta on the shared core (+6.29M params, +2.35 percent).
- Why: if passes want different maps, give them capacity.
- Result: token K1-K6 -0.0000, `mux_local` K1-K6 +0.0070, cancellation 0.463, CE@6 4.3060. Every B_t moved off zero. The sweep crashed past depth 8 (`IndexError` in `PassLoRA.delta`) and was re-run capped at 8. Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-per-pass-lora.md`.
- Verdict: failure. The deltas moved, the loop did not care. With progressive also flat, both forward-side attacks are closed.
- Led to: loss on every pass.

#### 2026-09-10: MUX on every pass (`tul.mux_every_pass: true`)
- Idea: the same next-span target on the live carry after every pass, uniform weights summing to 1.
- Why: the loop was scored only at its exit. Wolfe: "this is likely where our issue actually lives. that one arm needs to be run for sure."
- Result: token K1-K6 -0.0000, `mux_local` K1-K6 +0.0001, CE@6 4.3061. MUX at depth 1 (6.7698) already equals the ruler's depth 6 (6.7724). Cancellation rose to 0.771. Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-mux-every-pass.md`.
- Verdict: failure. Per-pass supervision toward one target trains depth independence.
- Led to: staged targets.

#### 2026-09-10: Staged targets, middle pass (`slot-mnext-staged`, `tul.mux_stage_own_iters: 3`)
- Idea: own-span target at pass 3, next-span target at the exit, loss 0.5 times (own plus next). The knob was built 2026-09-04 and never ran.
- Why: a toy study (`lab/toy_slot_loop/WRITEUP.md`) solved its chain 5/5 seeds with this shape, against 2/5 exit-only, 1/5 every-pass, 0/5 progressive.
- Result: token K1-K6 +0.0020, `mux_local_next` K1-K6 +0.0670, exit forecast 6.7698 (ruler 6.7724), pass-6 cosine -0.22, cancellation 0.442, CE@6 4.2483 (0.081 better than ruler), 1.04x wall clock. Source: `lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md`.
- Verdict: success by prediction count (P-c, P-e, P-f held). All three toy signatures transferred. The K-curve is depth dependence without exit value: the +0.067 is depth 1 being 0.058 worse, not depth 6 being better.
- Led to: staged-all, staged-mask, staged-fullread, and the 20k read.

#### 2026-09-10: Staged targets, every pass (`slot-mnext-staged-all`, `tul.mux_stage_all: true`, `mux_stage_own_iters: 1`)
- Idea: the toy's full winning attachment. Own-span term at every non-final pass, next-span term at the exit.
- Why: the middle-pass arm supervised only one intermediate pass.
- Result: no measured result. Only frozen predictions exist (P-c 32 percent, P-e 30 percent). A smoke died at step 0 with CUDA OOM because a detached probe ran beside it. Source: `lab/experiments/planned/2026-09-10-arc-slot-mnext-staged-all.md`.
- Verdict: pending. No filed result found in `successes/` or `failures/`. Unclear whether it ran or was dropped.
- Led to: nothing recorded.

#### 2026-09-10: Staged plus mask (`slot-mnext-staged-mask`, `tul.tg_restrict: true`, scope `all`)
- Idea: staged arm in strict geometry, where the slot is the only cross-span path.
- Why: the toy's larger finding was that permissive geometry hid the loop's value.
- Result: token K1-K6 +0.0033, `mux_local_next` K1-K6 +0.0604, exit 0.021 worse than ruler, CE@6 4.4373 (0.108 behind ruler), worth-zero +0.336, cancellation 0.385. Probe ordering was luck (a stale guard line started probes early). Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-mask.md`.
- Verdict: failure. The highest-confidence prediction (token K1-K6 above 0.03, 72 percent) was false. Mask K-curves of 0.02 to 0.06 from earlier runs did not reproduce.
- Led to: plain mask control under `norm_match`.

#### 2026-09-10: Whole-carrier reader (`slot-mnext-staged-fullread`, `tul.mux_readout: full`)
- Idea: the MUX head applies `lm_mixer` and `final_norm` per HC stream and averages logits, not streams.
- Why: audit finding F2. The stream mean kept only 0.139 of the loop's update on one arm.
- Result: token K1-K6 +0.0014, `mux_local_next` K1-K6 +0.0612, exit forecast 6.7543 (0.018 better than ruler), CE@6 4.3994, cancellation 0.382. Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-fullread.md`.
- Verdict: failure. F2 was a real defect on paper and inert in practice. The stream-mean survival figure (0.139) came from one arm that reads 0.577 vs 0.581 on the other, so it is nearly symmetric.
- Led to: the mask control and no-MUX arms.

#### 2026-09-10: Mask on the ruler (`slot-mux-mask-norm-match`, `tg_restrict: true`)
- Idea: plain mask, no staged term, under `norm_match`.
- Why: separate the ternary rule from the staged loss as the cause of the lost mask K-curve.
- Result: token K1-K6 +0.0009, `mux_local` K1-K6 +0.0091, CE@6 4.4194, worth-zero +0.554, cancellation 0.556. Source: `lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md`.
- Verdict: failure. The ternary rule removed the mask's token dependence. Every mask K-curve in the record (E4, E13, E16, E18) was measured on the starved `absmean` core. The staged loss is exonerated. This is a retraction of those mask K-curves.
- Led to: no-MUX masked arm.

#### 2026-09-10: No MUX, masked (`slot-loop-mask-norm-match`)
- Idea: MUX off, mask on. The coda's token CE is the only loss touching the slot.
- Why: Wolfe's reading that the slot's target is the lever.
- Result: token CE identical at depths 1 to 6 (4.3792). K1-K6 -0.0001. Core gradient norm 0.121 against prelude 21.9. Source: `lab/experiments/failures/2026-09-10-arc-slot-loop-mask-norm-match.md`.
- Verdict: failure. Token CE gives the core about 0.5 percent of the prelude's gradient, so it cannot train the loop. The MUX was load-bearing for every movement the loop made.
- Led to: the lane "a per-slot supervised quantity that needs more than one pass and the coda can read".

#### 2026-09-10: Fixed-point term off (`tul_slot_mux_fixed_point_off`, `core_fixed_point_lambda: 0.0`)
- Idea: remove the toy's top lever (the fixed-point term) and test it honestly.
- Why: in the toy the term took escape from 2/5 to 5/5.
- Result: runner verdict AMBIGUOUS. Max `preclip/total` 1,441 at step 245 (ruler 78.7). State norm 344 at depth 1 to 767 at depth 6. K1-K6 +0.0005, `mux_local` K1-K6 +0.0145, CE@6 4.3144. Source: `lab/experiments/failures/2026-09-10-arc-slot-mux-fixed-point-off.md`.
- Verdict: failure. The term is a state bound, not a contribution lever. Releasing it changed the trajectory a lot and the readouts not at all. It stays shipped on the plain-model detonation record. The file's "sign reversed half" reading is not supported by its own numbers.
- Led to: Parcae core arm.

#### 2026-09-10: Parcae core swap (`model.core_impl: parcae`, `injection_channels: all`, `injection_B: true`)
- Idea: replace the looped MORPH core with a plain dense bf16 pre-norm single-stream Parcae block stack (64.50M core params). Prelude and coda stay MORPH.
- Why: separate "MORPH's core at slot shape is the fault" from "any core reads flat".
- Result: token K1-K6 +0.0001, `mux_local` K1-K6 +0.0019 (ruler +0.0067), cancellation 0.915, z-optimisation entry +0.0060, CE@6 4.2940 (ruler 4.3290), wall clock 0.63x, 23,122 tok/s against 12,429. Typical gain 0.222 against 0.87. Source: `lab/experiments/successes/2026-09-10-arc-slot-mnext-parcae-core.md`.
- Verdict: success by prediction count. The core is exonerated. Five maps differing in precision, attention, residual, carry width and gain all sit inside the same flat band. Wolfe's "TUL on Parcae directly" question is answered.
- Led to: Parcae core as the cheap testbed for target and input work.

#### 2026-09-10: Gradient-conditioned passes (`slot-mnext-gradpass`, `tul.grad_pass: true`, `grad_pass_scale 0.1`, `grad_pass_norm: rms`)
- Idea: before each pass, inject the detached gradient of the slot's own-span loss with respect to its state, through a zero-init `W_g`.
- Why: the reader works and the writer produces nothing it wants. A fitted z has a gradient the loop's z lacks.
- Result: injected term is 8.6 percent of state at pass 0. Own-span CE ladder 7.105 to 6.740 at pass 1, then flat. Token K1-K6 +0.0014, `mux_local_next` K1-K6 +0.0236, exit forecast 6.7557, cancellation 0.502, CE@6 4.2772, 1.07x wall clock. Source: `lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md`.
- Verdict: filed as failure because three majority-lean predictions came out true, not because the lever failed. The loop takes one descent step on its own target and then holds. Correction at 05:02 (commit 428efd5): first filing said no state probe ran, and one had. Fullread's exit is 0.0014 higher, so "best exit" claims conflict across the two files.
- Led to: second-order form and masked twin, recorded as proposals, not queued.

#### 2026-09-10: Staged at 20k (`tul_slot_mnext_staged_20k`, `training.steps` 20000)
- Idea: horizon read of the middle-pass staged arm against a matched plain 20k control.
- Why: the 0.08 CE lead over the ruler at 5k was unread past 5k. Does the loop share grow with training?
- Result: token-paired gap to plain: 5k +0.1919, 10k +0.1596, 15k +0.1628, 20k +0.1609. Token K1-K6 0.0018 to 0.0025. `mux_local_next` K1-K6 0.0599 to 0.0637. Cancellation 0.453. Source: `lab/experiments/successes/2026-09-10-arc-slot-mnext-staged-20k.md`.
- Verdict: success (9 of 10 predictions held; P-e missed at 0.241 against a 0.30 bar). The gap shrank only between 5k and 10k. The slot apparatus costs about 0.16 nats vs plain from 10k on. Replication note: the 20k run's own 5k checkpoint reads CE 4.2310 (gap +0.1919), while the original 5k reads 4.2483 (gap +0.2099). Forecast K1-K6 is 0.0670 vs 0.0599.
- Led to: the overhead line item, and the 40k question (Wolfe's call, not queued).

#### 2026-09-10: Prefix width 4 (`tul_slot_mux_prefix4_norm_match`, `tul.prefix_k: 4`)
- Idea: `W_prefix` writes 4 cells per slot instead of 2.
- Why: `prefix_k` 2 had not been re-asked since Block Transformer. An earlier width sweep under `absmean` saw K-curves rise with width.
- Result: token K1-K6 +0.0001, `mux_local` K1-K6 +0.0074, CE@6 4.2904 (0.039 better than ruler; runner val 0.033 worse), slot effective rank 7.12 (ruler 6.34). Source: `lab/experiments/successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md`.
- Verdict: success by prediction count. The extra cells were used and the reader did not care. They carry the seed, not the loop.
- Led to: target as the only open lever.

#### 2026-09-10: Bag-mean seed (`tul_slot_mux_bagmean_norm_match`, `tul.slot_seed: bag_mean`)
- Idea: slot input is `E_slot` plus the span's token-embedding mean (no `W_sent`, 1.05M fewer params).
- Why: slot states sit at effective rank 1.7 to 4.8. The seed axis had never been measured under `norm_match`.
- Result: token K1-K6 +0.0005, `mux_local` K1-K6 +0.0057, CE@6 4.3536 (0.025 behind), slot effective rank 6.52, 0.97x wall clock. Source: `lab/experiments/successes/2026-09-10-arc-slot-mux-bagmean-norm-match.md`.
- Verdict: success by prediction count. Seed axis closed under the shipped rule. Static seed probe ranked bag-mean lowest (rank 1.13 vs 1.58), but training put it above, so the static rank instrument stops being cited as a design lever. Small price of 0.025 to 0.040 nats, not the 0.13 discount of the old content-seed arm.
- Led to: closing the slot channel at both ends.

#### 2026-09-10: HCA branch alive (`tul_slot_mux_hca_fix`, `model.core_hca_compress_ratio: 16`)
- Idea: scope the compressed attention ratio to the core, so the 64-cell slot sequence gets 4 compressed blocks, not 0.
- Why: audit finding F1. `GatedPoolCompressor` returned an empty stream when S is less than m, so three of six core blocks ran at about half attention while the gate spent 0.42 to 0.52 on the zero tensor.
- Result: n_blocks 4, slot/token attention ratios 0.73/0.83/1.08 (ruler 0.37/0.43/0.32). Token K1-K6 +0.0004, `mux_local` K1-K6 +0.0078, CE@6 4.2986 (0.030 better), 1.04x wall clock. Source: `lab/experiments/successes/2026-09-10-arc-slot-mux-hca-fix.md`.
- Verdict: success, every prediction held. A correctness fix worth about 0.03 nats at 5k and 4 percent wall clock, with no depth change. Shipped 2026-09-11 as the `tul_short.yaml` default.
- Led to: agent note `.agents/notes/implemented/bug-fix/2026-09-11-hca-compressed-branch-dead-on-the-slot-sequence.md`.

#### 2026-09-10: Toy slot loop (`lab/toy_slot_loop`, not a MORPH config)
- Idea: 0.60M-parameter toy (prelude, one looped core block on slot cells, prefix write, coda) on an S_3 prefix-scan task that provably needs j passes. Five seeds per arm, run on the 3070.
- Why: separate "the attachment is wrong" from "the M-next target does not need iteration".
- Result: stuck runs at cancellation 0.946, solved runs at 0.749. Strict versus permissive geometry moves the loop's contribution from +0.514 to +0.106. Source: `.agents/notes/proposed/architecture/2026-09-10-toy-slot-loop-update-flow.md`.
- Verdict: proposed. Staged target was the only attachment that solved the scan. Transfer risk: 450x smaller than MORPH. Seven of fourteen frozen toy predictions failed.
- Led to: staged, mask and fixed-point-off arms.

#### 2026-09-10: Credit assignment, readout and attention defects (notes)
- Idea: living notes that collect the loop-side arms and the F1/F2 defects.
- Result: no new numbers. Cotangent flat across passes (0.168 to 0.183). MUX pays 7.3x over token CE. Source notes: `.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md` and `2026-09-10-slot-loop-readout-and-attention-defects.md`.
- Verdict: both still `proposed` although every arm they name has a result. The credit-assignment note still describes `slot-mnext-staged-all` as an arm with no result. Its P-d still carries the old cancellation direction, which the toy overturned.
- Led to: staged arms, gradient-conditioned arm, Parcae core.

#### 2026-09-10: TUL on a Parcae core (`model.core_impl: morph | parcae`)
- Idea: build-time knob for the looped core (`ParcaeCoreBlock`, `morph/model/parcae_core.py`). Prelude and coda stay MORPH.
- Result: build and tests only (full suite 1055 passed, 1 skipped, 1 xfailed). Note: `.agents/notes/proposed/architecture/2026-09-10-tul-on-a-parcae-core.md`.
- Verdict: the note says delete the knob if the arm reads flat. It did read flat, but the note keeps it as the testbed. Unresolved.

---

### 2026-09-11 entries

#### 2026-09-11: Cross-span budget (`model.span_mask` row vs span, `budget_root.yaml`, `core_impl: parcae`)
- Idea: no-slot measurement. Two plain Parcae-core models identical except whether information can cross a span boundary. `span_mask: span` cuts with the one `BoundaryRule` and masks attention, CCA conv, value shift and hash bigram. Gate test `tests/test_span_mask_leak.py`.
- Why: nine slot arms read K1-K6 in [-0.0001, +0.0033]. Nobody had measured how much there is to carry. Route 1 predicted 0.10 to 0.18. Route 2 predicted 0.3 to 0.7.
- Result: gap +0.3994 [0.3838, 0.4162] at depth 6 over 491,520 tokens. By offset: 0.2826 at offset 0, 0.9580 at offset 1, falling to 0.3149 [0.3025, 0.3314] at offset 8+. Gap grows with training: 0.1633 at step 2,500. Token K1-K6 0.0279 (full) vs 0.0301 (span). Source: `lab/experiments/failures/2026-09-11-arc-span-budget.md`.
- Verdict: headline prediction missed upward, four shape predictions missed. The measurement is clean and is the arc's first ceiling. The budget is not front-loaded: it has a flat 0.31 long-range part plus a 0.96 spike at offset 1.
- Caveats: 5k number, still rising. Neither arm's absolute CE is comparable to the plain ruler. Code corpus pair not built.
- Led to: all-slots instrument and the span decoder.

#### 2026-09-11: Cross-span budget note (`model.span_mask`, `tg_comp_allow`)
- Idea: same design written as a proposal, listing every cross-position route and its cut. Note: `.agents/notes/proposed/architecture/2026-09-11-cross-span-budget.md`.
- Verdict: kept as the instrument, but status is still `proposed` with an Outcome appended.
- Led to: `span_budget_profile.py` as the paired-gap instrument in later panels.

#### 2026-09-11: Span decoder (`tul.spandec`, `tul.spandec_layers` 2, `tul.span_cap` 32, `tul.mux_detach_head`, `tul.slot_chain`, `plan_mode="all_slots"`)
- Idea: a 2-block causal, teacher-forced decoder over `[z, t_0..t_{J-2}]` predicts the whole next span from the slot's exit state. It replaces the MUX bag target (order-free). Training only, adds 27.83M params. Arms on the mask ruler: `slot-spandec-mask`, `slot-spandec-mnext-mask` (bag target added back), `slot-spandec-chain-mask` (`tul.slot_chain`), `slot-mask-dropout-off`, `slot-mask-mux-quarter`, plus `plain-coda-matched`. New instrument `plan_mode="all_slots"` cuts the prefix write, cell injections and own-span attention.
- Why: Wolfe, 2026-09-11 (quoted in source): "M-next asks for the next span's first token through the tied head; this is probably 99 % of our problem. We need the whole span to decode from z, and we are essentially getting like 2 tokens out of it." The bag target's minimiser is the unigram marginal.
- Result: CE gap spandec-mask minus mux-mask -0.0720 [-0.0748, -0.0698]. Bag target added back is +0.0409 worse, so harmful. Chain +0.0068 CE. Token K1-K6: spandec 0.0007, mnext 0.0013. Prefix-write worth falls (mux-mask 0.096, spandec 0.078) while all-slot worth rises (0.115 to 0.182). Gain at short offsets (bins 1 to 3 channel share 36/28/28 percent to 86/62/53 percent). Matched compute: spandec-mask at depth 6 sits 0.2536 behind plain depth-1 coda. Source: `lab/experiments/successes/2026-09-11-arc-span-decoder.md`.
- Verdict: success by protocol (8 of 10 predictions). The target works on the slot channel, but the verdict calls the gain a negative for the loop. The gain rides slot cells the coda reads directly, a bypass that `tg_restrict` never closed.
- RETRACTION: part-1 runner val 4.4192 was wrong, corrected to 4.4206. Only paired sweeps count.
- Contradiction: the prereg named the chain without decoder as next arm. The verdict moved to strict geometry instead. A deliberate deviation.
- Led to: strict geometry.

#### 2026-09-11: Span decoder design note (`tul.spandec`, `tul.slot_chain`)
- Same design as the filing. Alternatives weighed: wider write (flat), every-pass decoding (flat), sentence-encoder regression (forbidden), private 37.7M embedding, ternarising the decoder (rejected). Note: `.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md`, status still `proposed`.

#### 2026-09-11: HCA defect note
- Recorded under the 2026-09-10 HCA entry above. Note `.agents/notes/implemented/bug-fix/2026-09-11-hca-compressed-branch-dead-on-the-slot-sequence.md`.
- Contradiction: a parent note (`.agents/notes/rejected/bug-fix/2026-08-25-hca-compressed-branch-dead-on-slot-path.md`) rejected the same fix when the slot loop left the tree. It was re-adopted here.
- Caveat: old checkpoints such as `slot-mux-norm-match` need `model.core_hca_compress_ratio=256` to load.

---

### 2026-09-12 entries

#### 2026-09-12: Strict slot geometry (`tul.tg_geometry: strict`, `tul.tg_coda_prefix_reach`, `tul.loop_reach`, `tul.oracle_z`, `tul.spandec_horizon`)
- Idea: the loop becomes the only cross-span channel. Prelude allows only causal AND same bag. Coda lets a token see its own span plus prefix cells of earlier slots. CCA conv and `W_v_prev` reset per segment. Gate `tests/test_tul_strict_geometry.py` (bit-exact zero leak, 5 of 5 sabotages).
- Arms (one factor each against `slot-spandec-strict`): `slot-spandec-strict`, `-reach1` (`loop_reach` 1), `-prev` (coda reach `prev`), `-prev-reach1`, `-prev-reach2`, `-oracle`, `-h3`.
- Why: on the spandec mask arm the whole channel is worth 0.182 but the loop's own write only 0.078. More than half the channel never passes through an iteration.
- Result (5k, 7 of 7 healthy): strict K1-K6 +0.0016, worth-zero 0.1865 (equal to all-slots), paired CE vs spandec-mask -0.0001 [-0.0024, +0.0024]. `reach1` K1-K6 +0.0014. `prev` K1-K6 +0.0036 (CE +0.0082 vs strict). `prev-reach1` K1-K6 +0.0163 [+0.0153, +0.0173], CE +0.0104 vs strict. `prev-reach2` K1-K6 +0.0127. `h3` K1-K6 +0.0016. `oracle` K1-K6 -0.0002, CE +0.1195 vs strict. Source: `lab/experiments/failures/2026-09-12-arc-strict-geometry.md`.
- Verdict: failure by protocol (4 of 11 predictions held; P-a predicted a 0.05 to 0.20 cost, measured 0.0000). Geometry works: closing the bypass is free. Depth use comes from reachability, not target. Only arms where old spans are reachable solely through the loop's chain read K1-K6 above 0.012, and they are a relay, not refinement.
- Wolfe (Amendment 3, quoted in source): "we should try reach2. I am suspect of this method. It proves that TUL can work for sure. But the amount of blindness here is concerning. z has to hold the history when it should hold the present next thought that needs decoding. Our objectives are still poor."
- Amendments: Amendment 1 (reach1 cannot force depth under coda reach `all`, a design error; `prev-reach1` added). Amendment 2 (oracle smoke died at build, fixed in 36a4cfc).
- Led to: objective arms and core-token/critic arms, both on strict.

#### 2026-09-12: Strict geometry note (`.agents/notes/proposed/architecture/2026-09-12-strict-slot-geometry.md`)
- Same design. Outcome: "Strict is kept as the panel's geometry for the objective arms; the reach arms are a relay, not refinement, and are not a design." The first reach implementation spent its budget per block and was caught on a CPU fixture; the budget is now per pass.

#### 2026-09-12: Latent-z gradient (`tul.grad_pass` with `grad_pass_energy: recon` / `disc`, `tul.pass_residual_lambda` 0.01, `tul.slot_depth_fixed: 1`)
- Idea: condition each pass on the gradient of a harder energy than the earlier gradpass. `recon` is a second `SpanDecoder` reconstructing the slot's own span with soft labels. `disc` is a 2-layer critic trained with BCE on "next-span coda CE below median". Control `slot-spandec-strict-norecur` fixes depth at 1.
- Why: gradpass showed the mechanism works and its energy was trivial (one pass took 0.366 of 0.385 nats). Nobody had checked whether the slot state is scoreable.
- Result: step-0 linear probe (96 rows, 4,906 slots): exit AUC 0.608 flat, 0.637 mean. Signal already in entry state. Strict 5k arms: `recon` K1-K6 +0.0007, CE +0.0495 vs strict, 1.78x wall clock. `disc` K1-K6 +0.0019, CE +0.0081 vs strict, 1.07x. `norecur` 0.0020 [-0.0006, +0.0044] behind the depth-6 ruler. Strict parent reads 0.01353 on `ce_entry-ce_loop`; the bar was 0.02 and neither energy beats it. Source: `lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md`.
- Verdict: failure. Passes are paid, credited about equally, and deliver one pass of write. "A latent-z gradient is one more per-pass target met in one step."
- Amendment 1: all arms moved from the mask base to the strict base after predictions were frozen. P-e and P-h were scored against the new parent. P-c unscorable (energy ladder never logged).
- RETRACTION: the `ce_zopt` rows (-1.4 to -1.6 nats) are hindsight. The headroom reading is retracted here.
- Contradiction: earlier records say token CE reaches the core at about 1 percent of prelude gradient. This strict probe reads about 0.3. Different models, but the core-token file still quotes the 1 percent figure as current.
- Note: `.agents/notes/proposed/architecture/2026-09-12-latent-z-gradient-loop.md`, status `proposed`.

#### 2026-09-12: Math under norm_match (`tul_oly_spandec_strict`, `tul_sud_spandec_strict`, `notul_oly`, `notul_sud`, `max_slots: 0` for Sudoku)
- Idea: rerun Olympiad and Sudoku with the shipped recipe (`norm_match`, spandec, strict), 6,000 steps, plain controls first.
- Why: Wolfe's standing rule (2026-09-12): a flat loop on a corpus is not a verdict about that corpus. E16 and E17 ran on a starved absmean core.
- Result (depth 6, doc-paired): Sudoku slot minus plain +0.2502 [+0.2469, +0.2534]. Olympiad +0.2161 [+0.2015, +0.2313]. Slot K1-K6 Olympiad +0.0111 [+0.0085, +0.0136]. Sudoku +0.0049. Worth profile: Olympiad -0.1393, Sudoku -0.1903. Sudoku solve rate: plain 0.8 percent, slot 0.0 percent. Source: `lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md`.
- Verdict: failure. Gaps grew by about 0.04 on both corpora. Math lane closes for the slot loop as defined.
- RETRACTION (internal): the first version claimed the offset-0 damage held on web text. Web offset-0 worth is +0.757 (strict), so the sign flips between corpora. Candidate mechanism untested.
- Also: the prereg's E16 and E17 bars (0.18, 0.155) differ from recomputed values (0.1728, 0.2136).

#### 2026-09-12: Objective arms (`tul.spandec_per_pass`, `tul.coda_span_heads` 8 and 32, `tul.coda_span_source`)
- Idea: keep full reach, change only the objective. `slot-spandec-strict-perpass`: pass t decodes spans s+1 to s+min(t,6), shared decoder with its own position table. `slot-spandec-strict-codaspan` (J=8) and `-codaspan32` (J=32): parallel offset heads on the slot's last prefix cell, non-autoregressive.
- Result (5k): perpass K1-K6 +0.0008, CE +0.0172 vs strict, 129 min. codaspan K1-K6 +0.0012, CE +0.0415, 58 min. codaspan32 K1-K6 +0.0017, CE +0.0255. Identical-target grid (corrected): depth buys nothing at any horizon. Source: `lab/experiments/failures/2026-09-12-arc-objective-arms.md`.
- Verdict: failure. 8 of 12 predictions false by the letter. Objective lane closed alongside geometry. Only prev-reach1 has passes 2 to 6 carrying anything, at 0.011 nats CE cost.
- Amendments: Amendment 1 (P-2 compares different token sets, so it is not a progress reading). Amendment 2 (perpass OOM at 25.5 GB; the cause was span decoder activations at 11.0 GB, not CE accumulators; fixed by checkpointing).
- RETRACTION: the first identical-target grid is INVALID (bare `_tul_front` ran the prelude unrestricted under strict). Corrected at 7a24adf.
- Contradiction: the strict file says passes were ceilinged by reach. This file shows the rate floor was off on this runner.

#### 2026-09-12: Objective arms note (`.agents/notes/proposed/architecture/2026-09-12-objective-arms-per-pass-plan-and-parallel-coda.md`)
- Same design. Alternatives weighed included autoregressive heads (reintroduces token path), 32 tokens per span (84 decoder block-passes per token). No outcome section.

#### 2026-09-12: Core-token auxiliary and critic (`tul.core_token_aux`, `tul.grad_pass_energy: critic`, `tul.critic_eps` 0.1, `tul.critic_replay_groups`, `tul.critic_every`)
- Idea: two arms. `slot-spandec-strict-coretok`: training-only second pass sending every position through `_core_region` with its own Poisson depth, under `causal AND (same span OR slot cell)`. `slot-spandec-strict-critic`: pairwise logistic critic on three candidate states per slot, replayed through the real coda under no_grad.
- Why: nothing trains the core. Wolfe, 2026-09-12 (quoted in source): "train the core on the token CE as well (tokens through the core for gradient only, slots still the only cross-span channel at inference), and a within-context critic that scores whether the state after pass t beats the state after pass t-1 on the same coda loss."
- Result (5k): coretok CE -0.0684 vs strict, every offset negative. Aux path 0.016 better on a clean probe. Slot passes through the core still read token K1-K6 +0.0005. Run-level `tul/core_token_aux_ce` 0.31 gap is dropout-confounded (clean probe reads -0.0164). Critic K1-K6 +0.0012, CE -0.0025 vs strict. Critic loss sits at 0.689 (ln 2 is 0.693) for 5,000 steps. Source: `lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md`.
- Verdict: failure. "The core is not under-trained, it is under-used." Critic labels are measured near-ties (0.001 to 0.003 per pass).
- Open: aux weight (1.0) never swept, one seed, the aux backward also reaches prelude and coda (confound).
- Note: `.agents/notes/proposed/architecture/2026-09-12-core-token-gradient-and-within-context-critic.md`, no outcome section.

#### 2026-09-12: Causal instruments (`lab/divergence/slot_z_causal_fit.py`, `slot_depth_isolation.py`, `spandec_horizon_grid.py`, forward arg `slot_depths`)
- Idea: three eval-only instruments. Causal fit samples continuations from a frozen plain teacher, rather than fitting z on the real span. Per-slot depth isolation. Depth-by-horizon grid.
- Why: `slot_z_optimize` fits z on the tokens it scores, so the 0.9 to 2.6 nat number is hindsight.
- Result (only the causal fit ran; 6 rows, depth 6, DGX Spark): entry +0.0185 vs loop, hindsight -1.479 [-1.559, -1.397], causal-on-real +0.252 [+0.227, +0.273] WORSE than the loop (CE 4.0933 vs 3.8416). Source: `.agents/notes/proposed/testing/2026-09-12-causal-instruments-for-the-slot-loop.md`.
- Verdict: RETRACTION of the "reader has capacity" reading. Headroom was hindsight. Instruments kept. Status still `proposed`. Only 6 rows from one teacher, so weak.

---

### 2026-09-13 entries

#### 2026-09-13: Information view of the slot loop (`lab/theory/tul_information`, `tul.loop_reads_tokens`)
- Idea: Lean 4 account (sorry-free). A deterministic loop adds no information. Value comes only through a relay or extractability. The slot loop has no relay because the coda reads all cells.
- Result: no measured result. Cited readings: 10 percent perturbation of z moves coda loss at most 0.0005 nats. One pass is worth 0.001 to 0.003. Strict channel worth 0.1865 of a 0.3994 budget.
- Verdict: adopted as theory. Self-correction: a trajectory equals the entry in information terms, not the exit (the brief's claim was wrong). Note: `.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md`.

#### 2026-09-13: Second-opinion review, rank collapse (`tul.slot_cells`, `n_coda: 1`)
- Idea: outside model's review checked point by point against filed numbers. Wolfe flagged it as a suspected early RSI model and said not to trust it fully.
- Result: the review's rank-collapse finding matched the ledger. Its Proposal 1 (rank starvation) became the Thought Register build.
- RETRACTION (appended correction, same day): headline 5.76 rank was a bare-front reading. The collapse is real but milder (about 13 directions in about 50 slots; a shared row offset carries 93 percent of a state's length). Note: `.agents/notes/proposed/architecture/2026-09-13-second-opinion-review-rank-collapse.md`.

#### 2026-09-13: Rank anatomy (retraction of the rank headline)
- `val/slot_eff_rank` 5.7598 and `val/slot_pairwise_cos` 0.7104 for `slot-spandec-strict` came from a pre-`7a24adf` probe that ran a strict model's prelude unrestricted. Corrected baseline 13.85 / 0.520 (trainer val recipe), or 11.96 / 0.5635 (480-row probe panel). Source: `lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`.
- Readout partly survives. Centring each cell on its own mean gives 12.80. In readout space there is no halving (9.29 to 8.98). The seed and prelude half of the anatomy reading still stands.
- Three notes still quote 5.7598 uncorrected: `2026-09-13-the-thought-register.md`, `2026-09-13-discrete-thought-vq.md`, `2026-09-13-rank-levers-center-and-contrast.md`.

#### 2026-09-13: Rank levers (`tul.center_exit`, `tul.row_contrast_lambda` 0.1, `row_contrast_tau`, `b_center`, `W_contrast`)
- Idea: four arms. Center exit state per row (per cell index at M>1). Within-row contrastive target: span i+1 picks slot i out of the row. Detached targets, one direction.
- Result: no measured result. Gate built: 53 tests, compose PASS at d_model 1024, 16 of 16 sabotages caught. Found `tul_center.yaml` composed to a build error. Predictions P-1 and P-2 written against the retracted baseline.
- Verdict: superseded in practice. No recorded run. Sources: `lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md` and `.agents/notes/proposed/architecture/2026-09-13-rank-levers-center-and-contrast.md`. Precedent: `center_bag_mean` bought +0.0015 nats in 2026-08.

#### 2026-09-13: Thought register (`tul.slot_cells`, `slot_cell_init`, `prefix_k`; arms `slot-register-m4`, `slot-register-m4-sameinit`, `slot-register-m8`)
- Idea: M mutable cells per span (M queries pool the span, `E_slot` plus `P_cell`), looped together. Cell i reads its own slot's cells and earlier slots. Cell i writes 1:1 into prefix cell i. Decoder grades the mean of cells.
- Why: twelve arms read flat and none changed how many states a span holds.
- Result (5k, 480 rows): m4 token K1-K6 +0.00197, K3-K6 -0.00029. sameinit +0.00247 / +0.00001. Within-slot rank 1.2445 (sameinit 1.2366), cosine 0.9432. Row rank 10.57 against corrected ruler 13.85. Paired depth-6 CE m4 vs strict -0.0217, m4 vs sameinit -0.0070 [-0.0093, -0.0049]. Source: `lab/experiments/failures/2026-09-13-arc-thought-register.md`.
- Verdict: failure, 5 of 7 scorable predictions false. Cells collapse; the cause is the loop and shared write, not the seed. The win is mostly prefix width and extra packed tokens. Capacity reading dead in its strongest form.
- `slot-register-m8` was pulled at 19:45 and never ran (P-7 and P-9 unscored). First m4 run killed at step 600 by a per-slot `eigvalsh` probe (fixed in `53c0497`).
- Note: `.agents/notes/proposed/architecture/2026-09-13-the-thought-register.md`, quotes uncorrected rank 5.76.
- Led to: cond4, reader, rank levers, width ruler.

#### 2026-09-13: cond4 reader (`tul.cond_layers: 4`, `slot_cells`, `tg_relation`; arms `slot-spandec-strict-cond4`, `slot-register-m4-cond4`)
- Idea: four non-shared MORPH blocks between the loop exit and every reader of z, on the S by M cell axis. Both the coda's prefix write and the span decoder read the stack's output.
- Why: Wolfe's 2026-09-03 sketch had never had a clean run. The stack also read the wrong tensor on a register (means instead of cells).
- Result (5k): strict-cond4 K1-K6 +0.00026 (partner +0.0016). register-cond4 K1-K6 +0.00062 (partner +0.0020). Paired CE arm minus partner: strict +0.0057, register +0.0159. Cell rank 1.311 vs 1.245, cell cosine 0.799 vs 0.943, row rank 13.52 vs 10.57. Coda does not pay for either (worth 0.181 vs 0.194). Source: `lab/experiments/failures/2026-09-13-arc-cond4-reader.md`.
- Verdict: failure. Binding clause fired: P-1 holds and P-3 fails. Capacity between loop and readers buys nothing. Reader-side lanes closed for the slot channel.
- Note: `.agents/notes/proposed/architecture/2026-09-13-cond4-as-the-reader-of-the-register.md`, superseded by the filing.

#### 2026-09-13: Register reader and target offset (`tul.spandec_reads_cells`, `tul.spandec_target_offset`; arms `slot-spandec-strict-off2`, `slot-spandec-strict-off3`, `slot-register-m4-reader`)
- Idea: (1) span decoder cross-attends to the slot's M cells. (2) decoder decodes span s+k, k 2 or 3, not the next span.
- Why: the register grades the mean of its cells. The 2026-08-14 JEPA screen points two to three spans out.
- Result: `slot-register-m4-reader` NEVER ran. Offsets: off2 K1-K6 +0.0023, CE -0.0035 vs strict [-0.0057, -0.0012]. off3 K1-K6 +0.0009, CE -0.0008. Far worth bin (16+) +0.0941 and +0.0961 against a strict ruler +0.0905. P-7 bar was +0.02, failed. Source: `lab/experiments/failures/2026-09-13-arc-register-reader-and-downstream-target.md`.
- Verdict: failure. Decoder with zeroed z reads +0.26 to +0.32 (upper bound only). "Where z is graded does not move what the coda gets."
- Note: `.agents/notes/proposed/architecture/2026-09-13-register-reader-and-downstream-target.md`, says "built, unrun" for the reader arm.

#### 2026-09-13: Ultralight coda (`n_prelude: 7`, `n_core: 6`, `n_coda: 1`; arm `slot-ultralight-coda`)
- Idea: coda cut to one block, freed blocks move to the prelude (14 blocks conserved).
- Why: test "the coda has capacity to ignore z".
- Result (5k): K1-K6 +0.00169 vs partner +0.00161. Paired CE +0.0073 worse [+0.0043, +0.0099]. All-slots worth 0.1823 vs 0.1865. Offset-0 worth +0.9493 vs +0.7567. Row rank 12.58. Source: `lab/experiments/failures/2026-09-13-arc-ultralight-coda.md`.
- Verdict: failure, 4 of 4 scorable predictions false. "The coda ignores z because it does not have to read it" is dead as an explanation. Caveat: two changes at once (coda down, prelude up). Note: `.agents/notes/proposed/architecture/2026-09-13-the-ultralight-coda-control.md`.

#### 2026-09-13: Trajectory prefix (`tul.prefix_source`: `trajectory`, `exit_repeat`, `entry_exit`; `prefix_k: 6`)
- Idea: coda reads the loop's per-pass states instead of copies of the exit. Pad cells exactly zero and cut from the key set.
- Why: Lean result says a trajectory carries exactly the entry, so `entry_exit` is the binding control.
- Result (5k): `traj` depth-6 CE 4.3203, K1-K6 +0.0342 (K3-K6 +0.0034). `trajrep` CE 4.3208, K1-K6 +0.0019. `entryexit` CE 4.3196, K1-K6 +0.0014. Ruler (k=2) CE 4.3474. Paired: traj vs entryexit -0.0042. Source: `lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md`.
- RETRACTION: the +0.0342 K-curve is a forced-depth pad-cell artefact. Same-width `trajrep` reads +0.0019. Pad leak (E_pass gradient 7407 vs 0.25 on pad fixture) was found and fixed before the run.
- Verdict: failure. Every CE win is prefix width: six cells beat two by about 0.030 whatever they hold. "The coda only sees the exit" is retired as a reason for the flat curve.
- Note (rejected): `.agents/notes/rejected/architecture/2026-09-13-trajectory-as-the-prefix-content.md`. Its Risks still say `exit_repeat` is not queued, but `trajrep` was queued 2026-09-14.

#### 2026-09-13: Loop reads the tokens (`tul.loop_reads_tokens`, `core_token_aux_probe.py --depths`)
- Idea: training arm cut by Wolfe (about 44 block-passes per token against 10.59 for "think once, decode cheap"). Kept item: eval probe on the existing `slot-spandec-strict-coretok` checkpoint at forced aux depths 1, 2, 3, 6.
- Why: coretok's "under-used" verdict was scored at one depth. Does the token-reading aux core earn depth?
- Result (480 rows): shipped CE 4.2790 at every depth. Aux CE 4.2727, 4.2657, 4.2638, 4.2626 at depths 1, 2, 3, 6. Aux K1-K6 +0.0102 [+0.0093, +0.0110]. Slot loop on same weights 0.0005, so about 20x. Source: `lab/experiments/failures/2026-09-13-arc-loop-reads-tokens.md`.
- Verdict: failure. Q-1 false (bar 0.05). The "restriction costs two thirds" claim was RETRACTED: Parcae control `budget-web-span` (+0.0301) equals `budget-web-full` (+0.0279).
- Contradiction: coretok filing said "under-USED" with no depth; the loop-reads-tokens clause called it too strong.
- Note: `.agents/notes/proposed/architecture/2026-09-13-the-loop-reads-the-tokens.md`, no training arm queued.

#### 2026-09-13: Plain-then-TUL bootstrap (`training.init_from`, `training.data_skip_batches: 5000`, `training.resume_plain_to_tul`; config `tul_slot_strict_bootstrap.yaml`)
- Idea: Wolfe: "boot strap the model by training no tul for like 5k steps then use TUL for 5k steps to see what happens to loop contribution." Strict arm seeded from `plain-panel-norm-match` step 5000.
- Why: untested confounder. At step 0 z is noise, so a cold coda may learn to ignore z.
- Result (run 2026-09-20): K1-K6 +0.0019 [+0.0016, +0.0022]. All-slots worth 0.2287 vs 0.1865. Paired depth-6 CE vs strict -0.2164. Row rank 12.44. Source: `lab/experiments/failures/2026-09-13-arc-plain-then-tul-bootstrap.md`.
- Verdict: failure. P-6 bar set on retracted baseline. P-5 (-0.216) measures the extra 5,000 steps, not the loop. Reader maturity closed as a confounder. Seed improves, loop does not.
- Not isolated: the dropped compressed branch (105 pooled tensors, 1.2 percent of params) versus bootstrap.
- Note: `.agents/notes/proposed/architecture/2026-09-13-plain-then-tul-bootstrap.md`, status says `proposed` while the filing and Wolfe's call say failure.

#### 2026-09-13: Discrete thought VQ (`tul.vq_codes`, `vq_codebook` 512, `vq_groups`, `vq_beta`, `vq_reset_after` off; configs `tul_slot_spandec_strict_vq8`, `_vq4`)
- Idea: z passes through `W_vq` into K sub-vectors, each snapped to one shared 512-row cosine codebook, lifted by `W_vq_out[k]`, written 1:1 into prefix cell k. VQ-VAE loss.
- Why: continuous write is rank-collapsed and no continuous lever moved it. Quantizer gives rank by construction.
- Result (5k): vq8 codes used 295 of 512, perplexity 105.7, row rank 35.09, cell rank 6.72 of 8, K1-K6 +0.0015, paired CE vs strict +0.0250. vq4 perplexity 39.2, row rank 17.13, K1-K6 +0.0017, CE vs strict +0.0577. `all_slots` worth vq8 0.1397, vq4 0.1109. vq8 minus vq4 CE -0.0328 [-0.0358, -0.0300]. Source: `lab/experiments/mixed/2026-09-13-arc-discrete-thought-vq.md`.
- Verdict: mixed. 8 of 9 predictions held; P-7 failed. Filed as failure, moved to `mixed/` on Wolfe's call. Rank was not what stopped the passes. Code capacity is a live lever (the width-matched ruler resolved that vq8 vs vq4 is code capacity, not width). No reconstruction ceiling measured.
- Width ruler (quoted from this file; file `successes/2026-09-21-strict-pk8-ruler.md` was not read): pk8 minus strict -0.0349, vq8 minus pk8 +0.0592, vq4 minus pk8 +0.0923.
- Contradiction: register verdict said "the honest name is a wider prefix"; VQ file says pk8 beats strict by 0.035 while m4 beats strict by only 0.022. Different widths and packing; "m4 beats width" is untested.
- Note: `.agents/notes/proposed/architecture/2026-09-13-discrete-thought-vq.md`, quotes 5.7598 uncorrected.

#### 2026-09-13: TUL as memory (`plain_span_local.yaml`, `model.span_mask: span`, `--allow-core-mismatch`, `lab/divergence/memory_cost_table.py`)
- Idea: price the slot as a memory. Cost model: block-passes, visible keys and KV bytes per generated token, token-paired on `tok_index`.
- Result: slot arms cost 10.59 block-passes and 64.4 visible keys per token, against 14.00 and 512.5 for plain depth 1 (8.0x less context). Slot arms sit 0.385 nats behind it. `budget-web-span` is 0.092 nats worse than `slot-spandec-strict` (4.4389 vs 4.3466) at 4.2x block-passes. Recovered fraction under core mismatch: strict 0.231, coretok 0.402, plain depth-1 1.196 (impossible, so the mismatch shows). Sources: `lab/experiments/planned/2026-09-13-arc-tul-as-memory.md`, `.agents/notes/proposed/architecture/2026-09-13-tul-as-memory.md`.
- Verdict: instrument adopted. The matched control `plain-span-local` is unbuilt. Recovered-fraction column refuses to print without `--allow-core-mismatch`.

#### 2026-09-13: Depth ladder (`notul_norm_match_20k_d1/_d2/_d3/_d3fixed`, `norm-match-20k-p3/p4`, `model.depth_fixed`)
- Idea: Wolfe, 2026-09-13 evening (quoted in source): "At this point I would just be happy to have something that reduces compute enough that I can ship. I am not married to TUL." Plain rungs at depths 1, 2, 3 and sampled-draw rungs (`p3` Poisson mean 3 max 6, `p4` mean 4 max 8), 20k steps, against the top rung (mean 6, max 8).
- Result (20k, 480 rows, scored 2026-09-19; top rung CE 3.4516): d1 +0.0674 CE over top, 0.37x wall, 25,931 tok/s. d2 +0.0233, 0.48x. d3 +0.0095, 0.54x. p3 -0.0013 at depth 3 and -0.0019 at depth 6, 0.69x. p4 -0.0013 at depth 3 and -0.0051 at depth 6, 0.94x (contended). Seed spread at d3 is 0.024 nats. Source: `lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md`.
- Verdict: failure as a record, with a ship call. `p3` ships (level with top at 26 of 44 passes, 1.30x throughput). `p4` the safer choice if the K-curve matters. The loop is not a training-compute win at this horizon. Expected FLOPs is the wrong cost model (d3 and p3 average 26 passes, yet d3 is 1.40x faster).
- RETRACTION: wall-clock arithmetic in the Method was off by 2x (R8). Eval depth 4 named in S-3b was never swept.
- Led to: 20k vs 40k horizon check on `p3`, second seed, depths 4 and 5 (owed). Fixed-depth LoopMTP prereg.

#### 2026-09-13: Parked loop directions (`tul.traj_attn`, `tul.realign`, latent feedback, LoopMTP)
- Idea: a menu from 26-paper mining (`docs/references/looping-depth/2026-09-13-lit-mining/README.md`). Wolfe read LoopMTP (arXiv 2608.03624) and said it "directly addresses seemingly everything we need it to".
- Result: no measured result. Items stopped mid-build: loop-time attention over pass trajectory (RecurTrace), decode-then-encode realignment (DiscoLoop), full-bandwidth latent feedback (arXiv 2608.08888). Items 4 to 11 never started.
- Verdict: parked on Wolfe's call. Partial latent-feedback module at `ignore/parked/latent_feedback_partial_2026-09-13.py`. Note: `.agents/notes/proposed/architecture/2026-09-13-parked-loop-directions.md`.
- Led to: `2026-09-14` LoopMTP, horizon-passes and loop-diagnostics files (not read in this pass).

#### 2026-09-13: Planned arms that never filed
- Register `m8` (P-7, P-9 unscored), register reader (`slot-register-m4-reader`, P-1 to P-4 and P-11 unscored), four rank-lever arms, `plain-span-local`, `loop_reads_tokens` training arm (cut), and per-cell targets with an orthogonality term (named, not built). Owed, not run.

---

### Retractions and corrections (all eras in range)

1. Rank headline 5.7598 / 0.7104 retracted to 13.85 / 0.520 (`lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`). Several notes still quote the old number.
2. Fitted-z headroom (0.9 to 2.6 nats) is hindsight. Causal fit reads +0.252 worse than the loop (2026-09-12 instruments; latent-z file).
3. Mask K-curves in E4, E13, E16 and E18 were `absmean` artefacts. The shipped `norm_match` rule removed the mask's token dependence (2026-09-10 mask filing).
4. Span decoder part 1 runner val 4.4192 corrected to 4.4206.
5. First identical-target depth grid (objective arms) is INVALID. Corrected at 7a24adf.
6. Math file: offset-0 damage does not hold on web text (web offset-0 worth +0.757 strict).
7. Math prereg bars (0.18, 0.155) misquoted; recomputed 0.1728 and 0.2136.
8. Core-token run-level `tul/core_token_aux_ce` gap (0.31) is dropout-confounded (clean probe -0.0164).
9. Perpass memory mispriced (decoder activations 11.0 GB, not CE accumulators).
10. Latent-z file: P-e and P-h scored against a different control than frozen; P-c unscorable (energy ladder not logged).
11. Trajectory `traj` K1-K6 +0.0342 is a forced-depth pad-cell artefact. `trajrep` reads +0.0019.
12. Loop-reads-tokens "restriction costs two thirds" retracted (Parcae control shows equal gaps).
13. Depth ladder wall-clock arithmetic off by 2x.
14. Gradpass first filing said no state probe ran; corrected at 05:02 (428efd5).
15. Static seed rank probe retracted as predictor of trained rank.
16. Fixed-point-off "sign reversed half" reading not supported.
17. Information note's brief wrongly claimed a trajectory adds info over the exit; `mi_traj` says it equals the entry.
18. VQ source claim that STE is the only path from loss to loop was wrong; commitment term also reaches the loop.
19. `plan_mode: zero` ablates what the coda reads, not what the span decoder reads.
20. Register's in-loop relation did not execute until `tg_relation` was added.

### Contradictions between files

- Token CE gradient share to core: about 1 percent (2026-09-10 and core-token file) vs about 0.3 (strict spandec probes, latent-z file).
- Best exit: gradpass claims best of family, fullread filing records 0.0014 higher exit. Raw numbers favour fullread.
- Paid loop K1-K6: reread file says 0.120 under `norm_match`, staged-20k file says 0.1363 at 5k, Parcae file cites 0.185 and 0.033. Different arms, not reconciled.
- Replication spread on staged arm: 5k original gap +0.2099, the 20k run's own 5k checkpoint gap +0.1919.
- Strict geometry: bypass closed for free (0.0000 cost) versus math where the same write hurts offset 0 and web helps it.
- Register: "the honest name is a wider prefix" (register verdict) versus m4 vs pk8 width ruler (m4 beats strict by 0.022, pk8 by 0.035). Untested.
- Coretok: "under-USED" with no depth vs clause "too strong" on the depth-curve reading.
- Trajectory: `trajrep` cut from queue 2026-09-13 then queued back 2026-09-14. Rejected note still says not queued.
- Rank anatomy README reading 2 (prefix halves rank) is mostly a pooling artefact; readout-space shows no halving.
- Second-opinion note adopts rank starvation as main build, the same day the anatomy README says state count is not the limit.
- Folder and status drift: many 09-10 to 09-13 notes sit in `proposed/` with appended outcomes, and some `failures/` files carry "Planned:" in their titles. Several `planned/` paths cited by notes no longer exist (results were filed in place).
- Folder convention: gradpass (the most positive loss-side result) sits in `failures/`. Parcae, hca-fix, bagmean and prefix4 (null results) sit in `successes/`.

---

### Threads left open in this era

- `slot-mnext-staged-all` (own-span target at every non-final pass): never run, no filed result.
- A target the entry state does not already contain, that needs more than one pass and that the coda can read. Wolfe's stated lane, on the Parcae core.
- Second-order gradient pass (`create_graph=True`) and a masked twin of the gradient-conditioned arm. Recorded, not queued.
- 40k-step staged read. Gap flat from 10k to 20k.
- Ctx-only carry control for the Parcae core. Plain-model-with-pointer ruler.
- Core-token aux weight sweep (one seed), the aux backward confound (prelude and coda), critic replay groups above 1.
- Coda split (6/6/2, 5/6/3) and a 4/6/1 arm, to separate coda capacity from block count. Not built.
- `slot-register-m4-reader` and `slot-register-m8` never ran. Four rank-lever arms have no recorded run.
- `plain-span-local` matched control for the memory table: unbuilt.
- Per-cell targets (cell i decodes token bucket i) and an orthogonality term: named, not built.
- Residual-code VQ (one code per pass), to be scored against the width-matched pk8 ruler.
- Bootstrap: dropped compressed branch versus bootstrap is unseparated (no non-`tg_restrict` seeded control).
- Depth ladder: 20k vs 40k horizon on `p3`, second seed per rung, eval depths 4 and 5. Seed spread is 0.024 nats.
- Parked loop directions: LoopMTP (Wolfe reads it as the likely answer), RecurTrace attention, DiscoLoop realignment, latent feedback.
- Unread cited files: `successes/2026-09-21-strict-pk8-ruler.md`, `2026-09-14-arc-loop-diagnostics.md`, `2026-09-19-hop-distance-plateau-and-dilution.md`.
- Lifecycle housekeeping: most 09-10 to 09-13 notes still say `proposed`, and the lifecycle calls are Wolfe's.
- Variable cell count at inference, and the fan work, are outside this range.

---

## TUL history, 2026-09-14 to 2026-09-19

Scope: loop diagnostics, trajectory prefix, horizon targets, TUL-Code and LCTUL (sampled codes, discrete codes, LaDiR staging, LeJEPA, XM, CFG), the code target, the graded target, the frozen reference, the reader finding, the hop staircase, the mean_depth dial, and the sample oracle. Sources: `lab/experiments/{failures,successes,planned}/`, `.agents/notes/{proposed,implemented,rejected}/`, `docs/tul-code-spec.md`, and the index `docs/tul-loop-contribution-history.md`. Filing dates come from the file name. Where a file was scored later, the scoring date is given.

### Era summary

- The strict slot loop earned almost nothing from depth (K1-K6 about +0.002). The plain loop earned about 0.17. Every per-pass target tried so far was met in one pass.
- 09-14 opened two lines. TUL-Code asked the slot to hold the code of the next span, and a thinker to sample that code. Loop diagnostics asked whether any independent instrument could see structure.
- 09-14 to 09-16: every sampled code read as an unconditional draw. The thinker learned almost nothing about the past. Discrete codes, LaDiR staging, LeJEPA, XM and CFG all failed the same way. The filings reached one shared conclusion: a code the past can predict is too small to carry the span, and a code that carries the span is not predictable.
- 09-17 pivoted. The slot loop regressed onto a frozen VAE code. Pass 1 met the target. The cell held a real, small, span-specific signal, but the frozen coda gave it -4.6 nats. A graded target, where a grader ranks text and the encoder must learn the winner's code, closed with a sign-inverted margin.
- 09-18 showed the reader was the limit. Unfreezing the coda for 10k steps moved the same cell from -4.64 to +0.177 nats. A frozen coda's depth curve measures how generic the cell is, not how much depth matters. The hop staircase appeared on the strict slot loop with a reach cut.
- 09-19 replicated the staircase on fresh rows and traced the decay to loop-carried content. Carry arms and a K-stream fan failed. The depth ladder showed that `mean_depth` moves K1-K6 but not final CE. The sample oracle closed after Wolfe's review.

### Entries

#### 2026-09-14: Loop diagnostics panel (`lab/divergence/{aa_score,jacobian_gap_vs_k,sink_mass_per_pass,basin_map}.py`, `tul_slot_spandec_strict`, `notul_norm_match_20k`)
- Idea: four zero-training instruments from the 09-13 literature batch. Anil path independence (AA score), a power-method gap (sigma1/sigma2), attention sink mass per pass, and a basin map of entry perturbations. Two checkpoints at step 5000: the strict slot loop and the plain noise-entry loop.
- Why: the K-curve, rank and cosine instruments all said "one pass does everything". The panel asked whether four independent mechanisms agreed.
- Result: AA(swap) strict 0.8644, plain 0.9777. AA(noise) strict 0.8187, plain 0.9998. Spectral ratio median strict 1.052, plain 1.105. Sink mass flat across passes 2-6 (under 1 percent). Basin map: strict flat at 4 of 6 (row, slot) pairs, structured at 2 of 6 (entropy about 1.6 nats). Source: `lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md`.
- Verdict: three of four predictions failed. The plain loop, which earns, is an attractor. The strict loop, which does not earn, is entry-sensitive. Attractor-ness goes with earning. The spectral gap separates nothing. The filing calls the spectral-gap result a conflict with the 08-24 alignment reading, which it did not resolve. Sink mass is the one agreement.
- Led to: the basin question became an instrument question only, not a depth signal, until a paired CE moves with it.

#### 2026-09-14: Trajectory prefix, width against content (`traj`, `trajrep`, `entryexit`, prefix_k 6)
- Idea: the coda reads a six-cell prefix built from the loop's trajectory (`traj`), the same width with the exit repeated (`trajrep`), or the entry plus exit (`entryexit`).
- Why: test whether the coda needs the loop's path, not only its exit.
- Result: at 5k, depth-6 CE `traj` 4.3203, `trajrep` 4.3208, `entryexit` 4.3196 against the ruler 4.3474. The `traj` K1-K6 of +0.0342 looked like depth. Same-width `trajrep` reads +0.0019. Pass embedding takes 0.11 percent of the prefix gradient. Source: `lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md` (run 09-13 to 09-14).
- Verdict: failure. Retraction: the +0.0342 K1-K6 came from pad cells. At forced depth d the trajectory write fills d cells and pads the rest, so depth 1 loses five cells. Every CE win was prefix width. The three contents sat within 0.004 of each other.
- Led to: the width control became standard. Later "width" arms (`pk4`, `pk8`) were the ruler to beat.

#### 2026-09-14: Horizon-indexed passes, LoopMTP port (`tul_slot_spandec_strict_horizon`, `tul_slot_spandec_strict_fixed6`, `pass_readout: gated`, `horizon_weight`, `horizon_free_first`, `slot_depth_fixed`)
- Idea: pass t of the slot loop gets a horizon-indexed target. Pass t predicts the token t steps ahead, using LoopMTP (arXiv 2608.03624). Pass 1 is free. A gated readout mixes passes.
- Why: every earlier per-pass target asked each pass for the same thing. LoopMTP asks each pass for something different.
- Result: K1-K6 `horizon-arm` +0.0040 [+0.0036, +0.0044]. Paired against its fixed-depth control, depth-6 CE is -0.0001 [-0.0023, +0.0025]. The gate reads 0.83 on pass 1. The horizon matrix M[pass, horizon] has columns within 0.047 of each other, and the six horizon targets sit within 0.01 cosine. Retraction: the control `horizon-fixed6` K1-K6 +0.278 is out of distribution (trained only at depth 6). Source: `lab/experiments/failures/2026-09-14-arc-horizon-passes.md`.
- Verdict: failure. The horizon targets are degenerate on web text. Span-mean tied embeddings barely change with t. Any future per-pass target must show separated columns before an arm is queued.
- Led to: the rejected note `.agents/notes/rejected/architecture/2026-09-14-horizon-indexed-passes.md`. The token-loop variant below.

#### 2026-09-14: LoopMTP on the token loop (`notul_norm_match_20k_loopmtp.yaml`, `loopmtp_weight` 0.05, `loopmtp_free_first`, `loopmtp_proj: linear`, `loopmtp_ponder_weight` 0.05)
- Idea: the same horizon target on the plain token loop, not the slot loop.
- Why: a loop with token positions might separate passes where the slot did not.
- Result: a 20k iteration probe exists (`lab/experiments/results/2026-09-14-loopmtp-token-loop/loopmtp_iteration_probe_loopmtp_20000.json`, 48 rows, T=6). Cosine to the first pass is 0.95 and rises to 0.94 by pass 6. Consecutive-pass cosine goes from 0.949 to 0.996. The gate weight is flat near 0.166 on every pass. No K-curve or CE was filed.
- Verdict: open. The prereg (`lab/experiments/planned/2026-09-14-arc-loopmtp-token-loop.md`) has no Results section. The history index flags the probe as unfiled.
- Led to: nothing filed in this range.

#### 2026-09-14: TUL-Code first panel (`tul_code.yaml`, `tul_code_seeddetach.yaml`, `tul_code_nophase3.yaml`, `code_rollout_steps`, `code_noise` 0.5, `code_norm: rms`)
- Idea: the slot holds z_s, the code of span s+1. An encoder E makes the code from the span's own prelude states. A flow thinker, a conditional flow matching velocity field over codes, samples the code from the past. The mouth (coda) reads the code and speaks the span. Spec: `docs/tul-code-spec.md`.
- Why: the slot's regressed state was a mean (rank 6 of 1024). A code made by reconstruction should hold span identity.
- Result: all arms at 5k vs the strict ruler (ruler K1-K6 +0.0016, worth 0.1865). `tul-code` sampled `ce_k1` +0.705 [+0.702, +0.709] behind the ruler. Encoder code `ce_tf` 0.57. Effective rank 42. Shuffle cost about 0. `tul-code-nophase3` +7.44 behind. Rates 33.1k vs ruler 10.4k tok/s. Flow ratio on nophase3 falls to 0.33 at 5k. Source: `lab/experiments/failures/2026-09-14-arc-tul-code.md`.
- Verdict: the 5k predictions failed (P-2, P-3, and the shuffle half of P-1). The panel said the design "does what the spec says" but one sample costs 0.7 nats. Wolfe, 09-14: "much better than it looks, it is just undertrained". The filing says 5k cannot separate the two readings.
- Led to: the 20k pair and the rollout arm. The FM binding rule (spec §14) was waived for TUL-Code by Wolfe on 09-14. The slot-loop rule against regressing onto an external vector was not waived until 09-17, for the code target.

#### 2026-09-14 to 09-15: TUL-Code at 20k, frozen-E rollout (`tul-code-20k` = `tul_code.yaml` at 20k, `tul-code-rollout1-20k` with `code_rollout_p: 1.0`)
- Idea: phase 3 hands the coda sampled codes. The rollout arm freezes E from step 10k and gives the coda only sampled codes (`code_rollout_p` 1.0). The other arm uses p 0.5.
- Why: the coda must learn to read what the thinker actually makes.
- Result: paired one-draw gap to ruler +0.626 [+0.618, +0.634] (p 0.5) and +0.359 (p 1.0). The rollout arm beats the mixed arm by -0.268. Phase-2 K-curve is steep (-1.25 at 5k, -1.29 at 10k), flat after rollout. Span samples keep a topic thread on 0 of 10 cuts (ruler 4 of 10). The sample is worth the top 32 of 1024 code directions. Retraction: the "encoder code - ruler -2.550" figure for the p 0.5 arm used a bare z that had a norm defect (fixed in bed41d7). Source: `lab/experiments/failures/2026-09-14-arc-tul-code-20k.md`.
- Verdict: failure. One of eight predictions held, and it held because the rollout coda ignores its cells. A code model whose sample is an unconditional draw is worse than no code (+0.27 nats). The ruler's deterministic cell is worth 0.36 nats over no code at this horizon.
- Led to: the renorm fork and the conditioned thinker.

#### 2026-09-15: K-sample marginal instrument (addendum in the 09-14 file; `lab/divergence/code_marginal_sweep.py`)
- Idea: a likelihood-style reading. For K = 8 draws per span, the log of the mean likelihood under the coda. The marginal is a lower bound on the true log-likelihood.
- Why: one-draw CE is the wrong likelihood for a latent-variable LM.
- Result: at 5k, marginal k16 minus k1 = -0.004 [-0.006, -0.003]. The eight draws are near-interchangeable to the coda. Correction filed on 09-15 02:20: an earlier wording called log 8 the cap on the draw gap. It is the cap against the best draw, not the mean.
- Verdict: the inference-depth signal under this metric is present and negligible. This instrument became the standard marginal sweep for the LCTUL family.
- Led to: the k-curve checks on every later arm.

#### 2026-09-15: Renorm fork (`tul_code_renorm.yaml`, resumed from `tul-code-20k` at step 10000)
- Idea: remove the norm flag that let the coda see a truth cell at a different RMS than a sampled cell.
- Why: the norm defect could leak a sample flag into the coda.
- Result: one-draw gap at 20k +0.659 [+0.651, +0.667]. Fork minus parent +0.033 [+0.031, +0.035]. Marginal k16 minus k1 -0.010 [-0.013, -0.007]. Source: `lab/experiments/successes/2026-09-15-tul-code-renorm-r10k.md`.
- Verdict: success under its protocol. The coda's discount of samples is content-based, not norm-based. The renorm is a correctness fix, not a lever.
- Led to: the conditioned thinker arms. Two defects found on the resume were fixed (wandb run id inheritance, fp32 optimizer moments).

#### 2026-09-15: Conditioned thinker, CFG and predictability pressure (`tul_code_cfg.yaml`, `code_cfg_drop` 0.1, `code_cfg_scale` 2.0; `tul_code_jepa.yaml`, `code_target_lambda` 0.1)
- Idea: `tul-code-cfg` uses classifier-free guidance on the thinker. `tul-code-jepa` lets the flow gradient reach E with a rank abort at 4.
- Why: make the sample carry the span, from the thinker side and from the encoder side.
- Result: cfg residual 1.81 to 1.84 (target 1.4). Guidance sweep w 1 to 3 moves one-draw CE by under 0.01. jepa: code rank fell from 70 to a minimum of 16.3, then rebounded to 31. Its phase-2 one-step sample read 2.5 nats better, but rollout erased it (sampled val 4.615 against parent 4.617 at 15k). Both arms: one draw +0.648 and +0.654 vs ruler. Marginals k16 minus k1: cfg -0.005, jepa -0.017. Source: `lab/experiments/failures/2026-09-15-tul-code-conditioned-thinker.md`.
- Verdict: failure. Both levers rejected in `.agents/notes/rejected/architecture/2026-09-15-tul-code-conditioned-thinker.md`. The code never stopped being a copy (ce_tf 0.32).
- Led to: the thinker-only arm and XM.

#### 2026-09-15: Thinker-only, frozen decoder for 50k steps (`tul_code_thinker.yaml`, run `tul-code-thinker`, batch 12)
- Idea: freeze E, the prelude and the coda at the parent's step 10000. Train only the thinker for 40k more steps at twice the batch.
- Why: test whether more thinker steps on a fixed target would move the sample.
- Result: flow share (the thinker's loss over the code's variance) 0.320 at 10k, 0.279 at 50k. Residual at rank 128 is 1.68. One-draw sampled CE 13.66. Marginal k16 minus k1 -1.88 [-2.00, -1.73], at 10 to 14 nats. Source: `lab/experiments/failures/2026-09-15-tul-code-thinker-only.md`.
- Verdict: failure on the thresholds. The inverted-looking k-curve sits on a coda that cannot use the sample at that loss level. The thinker's gains per 10k steps fell to 0.003 to 0.004.
- Led to: the rule that the coda must learn on samples (rollout arm), and the conditioned-thinker panel.

#### 2026-09-15: Explorative modeling, best of K draws (`tul_code_xm.yaml`, run `tul-code-xm`, K = 4; coda variant `tul-code-xmc` queued; noise-search variant `tul_code_xmn.yaml` planned)
- Idea: draw K full generations per slot, keep the endpoint nearest E's code, and train the flow loss on that seed. Based on Explorative Modeling (arXiv 2607.27372).
- Why: Wolfe asked whether XM would "solve training for us".
- Result: one-draw gap +0.627 [+0.619, +0.634]. Flow share 0.304. Residual 1.88 to 1.94. Marginal k16 minus k1 +0.016 (every extra Euler step costs). Selection ratio 0.91 in phase 2, the order statistic of interchangeable draws. Phase-3 rate 15.2k tok/s. Source: `lab/experiments/failures/2026-09-15-tul-code-xm.md`.
- Verdict: failure. Best-of-K over complete generations cannot help when every draw is an unconditional draw from one distribution. The coda-space variant (`tul-code-xmc`) never filed. The noise-search variant (`tul_code_xmn.yaml`, K = 4 thinker passes) was planned on 09-16 and never filed.
- Led to: the masked denoiser and the discrete-code arms.

#### 2026-09-16: LCTUL-D, discrete code with masked denoiser (`tul_code_d.yaml`, `code_discrete: true`, 8 symbols of 512, `code_sub_p`, `code_vq_codebook`)
- Idea: the code is 8 discrete symbols from a cosine codebook. The thinker is a masked denoiser (MDLM/LLaDA ELBO). The sampler unmasks k rounds in parallel.
- Why: the flow thinker put 1 percent of its loss on the past (context-blind probe), so the sample could not depend on context. A categorical objective might.
- Result: killed at about step 9900. At 5k, `ce_tf` 3.960 against ruler 4.378. Context share 0.02 percent (the flow arm had 1.1). k1 minus k8 = -0.162 [-0.176, -0.146], more rounds were worse. Perplexity 234.9. Source: `lab/experiments/failures/2026-09-16-lctul-d-first-arm.md`.
- Verdict: failure. The categorical objective did not bring the past into the gradient. The code rate (72 bits) is too large to be predictable from the past.
- Led to: the plan arm, a smaller code.

#### 2026-09-16: LCTUL-D plan arm, coarse code scored on generation (`tul_code_dplan.yaml`, 4 symbols of 64, 24 bits)
- Idea: a 24-bit plan from the same masked denoiser. Score on generated text with a semantic probe (MiniLM cosine to the true span, 120 cuts). Controls: OWN, SHUF, ZERO, ORACLE.
- Why: test the rate trade-off. A code the past can predict must be small. A code that is small may not matter.
- Result at 20k: OWN@4 minus SHUF +0.008 [-0.006, +0.023]. OWN@4 minus OWN@1 +0.009 [-0.008, +0.028]. OWN@4 vs ruler -0.030. Distinct-2 0.371 against ruler 0.437. Oracle plan 0.189 against ruler 0.187: a perfect thinker could only match the ruler. Context share 3.1 percent. The 8-draw marginal k4 minus k1 was -0.0119 [-0.0155, -0.0082] at 20k (inverted at 5k and 10k). Source: `lab/experiments/failures/2026-09-16-lctul-dplan-semantic.md`.
- Verdict: failure. Four of five semantic predictions failed. Sampled plans do no better than a foreign plan or no plan. The CE depth signal is real but does not reach generated text. The 72-bit and 1024-d codes carry more but are less predictable. No code tried was both.
- Led to: the semantic probe kept as the main generation instrument. The rate-predictability trade-off became a named open problem.

#### 2026-09-16: LCTUL trained the LaDiR way (`tul_code_ladir_tf.yaml` stage 1, `tul_code_ladir_ro.yaml` stage 2 and 3; VAE checkpoint `tul_code_vae.yaml`)
- Idea: three stages. Stage 1 trains the VAE code (noise 3.0). Stage 2 freezes the decoder and trains the thinker on its own tape. Stage 3 rolls out the thinker.
- Why: copy the LaDiR recipe, which decodes a noised code and never trains the decoder on samples.
- Result: stage 1 `ce_tf` 1.34, effective rank 57. Stage 2 sampled `ce_tokens` rose from 7.69 to 9.90. Marginal k16 minus k1 +0.319 [+0.225, +0.428]. Own minus shuffled -0.006 [-0.024, +0.011] at 40k. Stage 3 at 50k: own minus zero -0.022 [-0.044, -0.001]. Distinct-2 of own samples 0.89 (word salad). Source: `lab/experiments/failures/2026-09-16-lctul-ladir-chain.md`.
- Verdict: failure. A frozen coda reads the true code well (oracle continuation cosine 0.64, the best oracle of any arm). The thinker's sample still carries nothing about the span.
- Led to: the claim that a sampled next-span latent is the wrong shape for this data.

#### 2026-09-16: LeJEPA on the code (`tul_code_lejepa.yaml`, SIGReg guard, full flow gradient into E)
- Idea: let the flow gradient reach the encoder in full, guarded by SIGReg (LeJEPA) against rank collapse.
- Why: the jepa arm had collapsed rank. SIGReg should hold it.
- Result at 20k: rank held (minimum 40.4, 73 at the end). Flow share 0.256 (predicted at most 0.22). Residual 1.95 to 1.98, the most independent yet. One-draw gap +0.681 at k 6, +0.773 at k 1. Against parent +0.038 at k 6. One-draw depth curve k1 4.585, k2 4.486, then flat. Marginal k16 minus k1 -0.0006 [-0.004, +0.003]. Source: `lab/experiments/failures/2026-09-16-tul-code-lejepa.md`.
- Verdict: failure. Letting the target move lowers the flow loss by moving the target. The coda loses the parts the field cannot guess. What remains is still unconditional.
- Led to: the regressed-code target, which fixes the target instead of moving it.

#### 2026-09-16: Planned, never run: thinker phase 3 (`tul_code_thinker_p3.yaml`), 50k continuations (plain, jepa, xm), and the semantic probe as an instrument
- Idea: thinker phase 3 would unfreeze the coda on samples only after a long fixed-target phase 2. The 50k continuations would extend three 20k checkpoints. The semantic probe is the instrument that scored dplan and later arms.
- Why: test whether a slow learner closes the gap.
- Result: the 50k continuations and thinker-p3 have planned files only and no results in this range. The thinker-only 50k run (above) covered part of the same question. The semantic probe ran on dplan, cfg, lejepa, ladir and the target arms.
- Verdict: open. The planned files are `lab/experiments/planned/2026-09-16-tul-code-thinker-p3.md` and `2026-09-16-tul-code-50k-continuations.md`.
- Led to: the 09-17 pivot away from sampling.

#### 2026-09-17: Code target, detached cell (`tul_code_target.yaml`, `code_target_detach: true`, `code_target_ref` not yet)
- Idea: the ordinary strict slot loop, regressed onto the frozen VAE code of the next span. The coda reads the predicted cells with stop-gradient. Pass count is the instrument. 
- Why: samplers never carried the span. A regressed latent is not a sample. Wolfe, 09-17: "we can train TUL latent to match oracle. Its impossible for it to carry nothing about the span."
- Result at 20k: exit cosine 0.004 to 0.144 by step 1500, 0.160 at 20k. Own minus shuffled +0.0230 [+0.0051, +0.0421], the first positive own-versus-foreign reading in the LCTUL family. Own minus zero +0.0054 [-0.0167, +0.0283]. Own vs ruler -0.057. Frozen coda's oracle cosine 0.5645. Zeroing the cell makes the frozen coda 4.608 nats better. Rate 1.23 times the ruler. Source: `lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md` (Results section).
- Verdict: the regression works in one pass (exit cosine 0.06 at entry, 0.135 at pass 1, 0.132 at pass 6). The cell is real but not worth having to this reader. The filing names the frozen reader as the open confound. Retraction: the "7.4 nats = nothing" figure in an earlier spec version was never measured (spec §17.1). Retraction: the 20k OWN, SHUF and ZERO probes taken before commit 14f004d are void. An open-slot bug gave every valid slot a zero cell at eval, so the three readings were identical.
- Led to: the CE companion arm, the progressive knob and the code-only arm.

#### 2026-09-17: Code target, CE companion (`tul_code_target_ce.yaml`, `code_target_detach: false`)
- Idea: the token CE through the frozen coda trains the loop too.
- Why: test whether the coda's gradient helps.
- Result: stopped at step 15000 to free the GPU. Not scored. The prediction P-T7 is pending.
- Verdict: open. The checkpoint at 15000 is kept.
- Led to: the 09-17 unfreeze plan, which needed this arm's step 20000 checkpoint. It never existed.

#### 2026-09-17: Code target with a progressive loss (`tul_code_target_prog.yaml`, `progressive_p`)
- Idea: the first k passes are detached with some probability per slot, so later passes must refine a state they did not produce.
- Why: tie depth to refinement.
- Result at 20k: `code_target_cos_l6` minus `l1` +0.0073 (predicted at least +0.03), against +0.0051 for the detached arm. K1-K6 -0.1173 (wrong direction). Own minus zero +0.0222 [+0.0045, +0.0407], against +0.0054 for the detached arm. Its `pred.zbar` moved from 0.355 at depth 1 to 0.305 at depth 16 (the cell got sharper with depth). The frozen reader punished the sharper cell. Source: `lab/experiments/failures/2026-09-17-lctul-target-progressive.md`.
- Verdict: failure on depth. The knob is real and roughly doubles per-pass movement. It is about fifty times too small to matter as depth. The cell was more span-specific, which only the non-hostile instrument saw.
- Led to: the unfreeze continuation, which read the cell fairly.

#### 2026-09-17: Code-only arm, first draw collapsed and void (`tul_code_only.yaml`, `code_target_skip_coda: true`, `training.train_only: []`)
- Idea: no coda at train. The loss is the code term plus the slot-loop constraint only. The prelude, embeddings and loop all train.
- Why: Wolfe: "Poisson loop, the loop guesses the code. We never even run the coda." The earlier frozen prelude was thought to bound the target.
- Result: the first draw (wandb `38naddpq`) was killed at step 3500. The prelude drifted. Target E pooled the live prelude, so the codes collapsed onto one direction (own cosine 0.61, rising to 0.99 by step 2500 to 3000). Frozen coda's oracle CE 13.1 nats against 1.4 for the detached arm.
- Verdict: void, not a result. The fix is `code_target_ref` (below).
- Led to: the frozen reference twin.

#### 2026-09-17: Frozen reference twin (`tul_code_only.yaml` with `tul.code_target_ref: true`; `tul_code_only_nce.yaml` for the InfoNCE twin)
- Idea: a frozen deep copy of the model, snapshotted after weights load, computes every reading of the VAE stage. E and the coda read from the twin. The live model contributes only the loop's predicted cell.
- Why: stop the target from drifting with the prelude.
- Result at 20k (wandb `wy38mb6y`, one seed): codes held. Shuffled cosine ended at 0.038 (it fell during training). Own cosine 0.1674 vs 0.1413 for the detached arm. Own minus shuffled 0.138 and 0.110. Effective rank of predictions 17.4 and 17.6, against 77.5 and 70.2 for the true codes. Remove the corpus mean and own cosine is 0.15 against a shuffled floor of 0.003. The cell is a genuine 17-dimensional shadow of a 73-dimensional code. Source: `lab/experiments/failures/2026-09-17-lctul-code-only-ref.md`.
- Verdict: failure on every prediction about the idea. The prelude was not the bound. The frozen coda drifted to 12.64 nats, worse than uniform (10.80), so the K-curve, worth profile and oracle (0.1694) read void. Retraction: the K1-K6 values +0.0698, -0.0014, +0.0414, +0.0291 and the shuffle cost +0.297 are void for that reason. The InfoNCE twin (`tul_code_only_nce.yaml`, `code_target_loss: infonce`, `code_target_tau` 0.1) was planned and never filed.
- Led to: the unfreeze continuation (the reader must be trained), and a note that offline probes must rebuild the twin. `tul.code_target_ref` was not rebuilt by `scripts/tul_samples.py::load_ckpt` until commit 9263204.

#### 2026-09-17: Graded continuation target (`tul_code_grade.yaml`, `tul_code_grade_l2.yaml`, `code_grade`, `code_grade_k`, `code_grade_tokens`, `code_grade_grader: coda_past`, `code_grade_loss: pref`)
- Idea: the loop's predicted cells condition the frozen coda, which samples K continuations of span s+1. A grader that cannot see span s+1 ranks them. Loss pushes the cell toward the winner's code (InfoNCE over candidates). Based on the 09-17 note `.agents/notes/proposed/architecture/2026-09-17-lctul-graded-continuation-target.md`.
- Why: every target so far was a deterministic function of the past, and every one was met in one pass. A computed target might not be.
- Result: only the L2 twin ran (`tul-code-grade-l2`, wandb `w5cfd32z`, 20k). The grader works: truth rank 0.878. The loop's cell predicts the winner's code at cosine 0.532, but the winner's code is no closer to the truth (0.153) than the loser's (0.147). Per-pass cosines to the winner l0 0.401, l1 0.524, l6 0.531. Correction: a sentinel bug (`worst = gr.argmin`, commit b3381dd) corrupted the worst reading. The clean margin best minus worst is -0.004, not +0.0052. Its CE instruments are void (no coda at train, CE above uniform). The twin read: own minus shuffled +0.092, the least span-specific of the three. Source: `lab/experiments/failures/2026-09-17-lctul-graded-target.md`.
- Verdict: failure. Wolfe's question was "didn't we also have an arm that was grading plausible continuations with a grader?" The grader ranks text well, but encoding the text winner does not teach codes. Ranking text and ranking codes are different orderings. The pure arm (`tul_code_grade.yaml`, weight 0) was never run. Its prediction, if run, is that the cell gets closer to the sampler's style, not the truth.
- Led to: the graded family moved to rejected. The next question became the map, not the target.

#### 2026-09-17: Unfrozen reader continuation (`tul_code_target_uf.yaml`, run `tul-code-target-uf`, 10k steps from the 20k code-target checkpoint, to 30k)
- Idea: keep the code target and unfreeze the reader (the coda). Train it for 10k more steps.
- Why: test whether the frozen reader was the limit. Wolfe, 09-17: "have we tried using a fresh random coda after the core has learned?"
- Result at 30k: sampled `val/loss` 8.71 to 4.25. Semantic OWN cosine 0.1303 to 0.1592. Own minus shuffled +0.0225 [+0.0021, +0.0422]. Own minus zero +0.0179 [+0.0025, +0.0329]. Paired against the 20k point +0.0290 [+0.0040, +0.0533]. Cell worth -4.64 nats frozen and +0.177 adapted. Costs: the oracle fell from 0.5645 to 0.1463 (the reader stopped decoding the code space), `val/ce_tf` 1.43 to 4.17, and the K1-K6 fell from +0.0323 to +0.0047. Source: `lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md`.
- Verdict: success on the one arm that ran. The reader was the limit on whether the cell is usable. Unfreezing did not make the loop use depth. Centred cosine to the own code peaks at depth 6 on every arm.
- Led to: the sample-oracle probe on the adapted reader. Also `tul-code-target-ce-uf`, a queued twin of this with the CE companion, never ran. Its launch silently started from step 0 with 151.6M random parameters, was killed, and wrote no checkpoint. A guard now refuses that.

#### 2026-09-18: Hop-distance probe, first pass (`tul_slot_spandec_strict_prev_reach1.yaml`, `tul_slot_spandec_strict.yaml`, `reach1` override arm, `--planted`, `--cut-after 1`)
- Idea: a per-hop K-curve. Bin tokens by the span they depend on most, using a localiser. Measure K1-K6 per hop. Add planted copies of a rare token at distance g.
- Why: whole-arm K-curves hid structure. A forced hop might only earn where the geometry forces content through the loop.
- Result (5k, 480 rows, run 09-19 on the 3070): `prev-reach1` per-hop K1-K6: h1 +0.0109, h2 -0.0169, h3 +0.0640 [0.0610, 0.0671], h4 +0.0535, h5 +0.0288, h6 -0.0203. Strict and reach1 are flat (spreads 0.0038 and 0.0076). Planted benefit: content h spans back arrives at pass h minus 1. Control g=0 reads 0.107 (strict), 0.125 (reach1) and 0.162 (prev-reach1), so the probe is not inert. Source: `lab/experiments/failures/2026-09-18-hop-distance-earning.md`.
- Verdict: failure by the letter. The staircase is real. The magnitude falls with hop distance and turns negative by h 6. The frozen bars assumed a far-bin gain that does not decay. The 09-18 wording called the fall "dilution". The 09-19 second pass withdrew that wording for carried content.
- Led to: the second pass (plateau and dilution), and the carry arms.

#### 2026-09-18: Frozen-coda K-curve measures genericity (`lab/divergence/core_depth_sweep.py`; note `.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md`)
- Idea: a warning in the sweep tool when the coda is frozen.
- Why: a frozen coda prefers no cell (zero minus own -4.64 nats), so its depth CE tracks how generic the cell is, not how much the loop earns. Retractions of the K-curves in the target family rest on this.
- Result: the shipped check measures genericity. On the frozen coda, the progressive arm's cell got sharper with depth and read as worse. The detached arm's cell got blander and read as better (+0.032).
- Verdict: implemented. A frozen-coda K-curve needs a warning. Test: `tests/test_depth_sweep_frozen_reader_warning.py`.
- Led to: the unfrozen reader as the required reader for any depth claim.

#### 2026-09-18: Sample-oracle gate, frozen reader (`lab/divergence/sample_oracle_probe.py`, `slot-spandec-strict` at 5000, N = 16 samples, entry sigma 0.3 and 1.0)
- Idea: sample the slot entry (and exit) with noise. Measure the best-of-16 oracle gain over the deterministic point. A large gain would justify a selector.
- Why: test whether sampling around the loop finds alternatives the point state loses. The fan and stream arms depended on it.
- Result (run 09-19 on the Spark): gain(16) at depth 1 entry sigma 0.3 +0.0117, sigma 1.0 +0.0251. Exit sigma 1.0 +0.0296. cos_exit 0.962 (sigma 0.3) and 0.720 (sigma 1.0), against a prediction of 0.98 and 0.90. Passes 2 to 6 add nothing (growth -0.0008). Source: `lab/experiments/failures/2026-09-18-sample-oracle-gate.md`.
- Verdict: failure. Closed on this reader. The cell's direction is close to irrelevant to the frozen reader. The predicted collapse did not happen.
- Led to: the adapted-reader repeat. Wolfe's review (below) then qualified this verdict.

#### 2026-09-18: Toy set-carry, deferred commitment (`toy eliminate`, a 2-pass chain, no YAML)
- Idea: a toy with six candidates. Candidates are eliminated one by one. Ask whether a slot loop carries the alive set (superposition) or commits early. Gate on entropy of candidate mass.
- Why: the Thought Register collapse said a loop cannot hold a set.
- Result: solved seeds commit early on candidate mass (entropy 0.080 to 0.424 of 0.693). The membership probe reads 1.000 for the alive set. Random-init cells already hold the set (0.998). Fixed-point term entropy 0.989 of 1.0986 at the candidate slot, against 0.352 for plain exit and 0.096 at random init. Source: `lab/experiments/failures/2026-09-18-toy-eliminate-deferred-commitment.md`.
- Verdict: falsifier fired. The alive set is carried, not as superposition. The head exposes exclusions, not candidates. Carrying a set costs nothing. Training buys the readout. Predictions P3 and P10 were unreadable (2-pass chain has no basin).
- Led to: a deeper toy (`eliminate6`, 6 candidates, 4 eliminations).

#### 2026-09-19: Toy six-hop carry (`toy eliminate6`)
- Idea: a 4-pass chain with 6 candidates and a per-fact fitted reader (linear probe) against an untrained baseline.
- Why: the 09-18 toy had no basin.
- Result: random-init hop decay: 1.000 at hop 0, 0.668 at hop 4 (d96), 0.713 (d192). Trained cells lift it to 0.999 to 1.000 at every hop of 3 or more. Random-init fitted reader 2.002 nats at depth 8 against chance 2.079. Width: 96 dims solve 2 of 15 cells, 192 dims solve 8 of 15. Fitted reader +0.030 over the coda on stuck seeds. Source: `lab/experiments/successes/2026-09-19-toy-eliminate6-hop-distance.md`.
- Verdict: success on its contrast clause. Retraction: iteration 1's conclusion that a loop cannot carry past two hops was a two-hop artefact. A shared-core slot loop carries a fact one or two hops for free. Closing the gap is what training buys.
- Led to: the real-model hop probe with an untrained baseline.

#### 2026-09-19: Hop plateau and dilution, second pass (`tul_slot_spandec_strict_prev_reach1.yaml`, `hop_distance_probe.py` at depths 1 to 8, 12, 16; `--cut-after 1`; `--row-offset 480 --seed 1`)
- Idea: fill the depth curve. Test the plateau (arrival at pass h minus 1), the dilution timing, and the mechanism (does loop-carried content decay under later passes).
- Why: the 09-18 pass could not read the plateau.
- Result: arrival is a step and a half. h5 gets 79 percent of its fall at pass 4, the rest at pass 5. Under the reach cut, planted g=2 falls 0.148 to 0.035. The h2 bin is 0.083 nats worse by depth 6. Own-span planted g=1 rises 0.181 to 0.291 under the cut, and falls 0.181 to 0.137 uncut (later arrivals displace it). A disjoint 480-row draw reproduces every bin with the same sign: h3 +0.0665, h4 +0.0545, h5 +0.0302. Source: `lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md`.
- Verdict: failure by the filing rule (P2 and P3 failed). The settled answer: loop-carried content decays under the cell's own later passes, roughly a third per pass over passes 1 to 3. Own-span content, re-supplied each pass, improves. The 09-18 "dilution" wording is withdrawn for carried content and kept only for displacement by later arrivals.
- Led to: the carry arms.

#### 2026-09-19: Loop carry, sum and gate (`tul_slot_spandec_strict_prev_reach1_carry.yaml`, overrides `carry-sum` and `carry-gate`)
- Idea: a cell keeps what it read from its neighbour. Sum arm: running sum of every pass's read, re-injected from pass 2. Gate arm: the same with a learned gate (mean 0.27).
- Why: stop the decay of carried content. The 09-19 plateau file named the fix.
- Result (5k): depth-6 CE `carry-sum` 4.4554 (behind ruler by 0.098), `carry-gate` 4.4081 (behind by 0.050). Sum arm inverts the staircase (Spearman -1.000). Far hops go negative: h3 -0.021, h4 -0.074, h5 -0.088, h6 -0.096. Under the cut, planted g=2 rises 0.108 to 0.210 with depth (decay stopped). Carry RMS grows to 75. First-iteration gain reaches 2.8e5 on the sum arm (ruler max 1.95). No detonation. Source: `lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md`.
- Verdict: failure. Re-supply stops the decay. Far hops still lose, through dilution. The state is unbounded, because the RMS match bounds only the injection. The carry replaces the pass-1 read rather than adding to it. Note: the carry file frames far-bin loss as dilution, and the plateau file frames the same data as per-pass decay. Both are cited. The carry file calls the two the same mechanism "made worse by accumulating".
- Led to: a bounded carry that keeps the pass-1 read (not queued). Proposed design: `.agents/notes/proposed/architecture/2026-09-19-loop-carry-reinjection.md`.

#### 2026-09-19: LXTUL fan4, K latent streams with repulsion (`tul_slot_spandec_strict_fan4.yaml`, override arms `fan4-norepel`, `fan4-mean`, width partner `pk4`)
- Idea: K = 4 latent streams through the shared core, a cosine repulsion on the streams, a softmax gate and an oracle over streams. Based on the proposed note `.agents/notes/proposed/architecture/2026-09-19-lxtul-fan-streams.md`.
- Why: a single stream could not explore. Streams might.
- Result (5k): `single - oracle` 0.1069 (the prereg's falsifier was at most 0.022, so the letter reads true). But `oracle - mixed` is 0.0078, and `stream_ce_k{i} - mixed` is +0.099 to +0.029 on the repelled arm. The repulsion made a constant two-against-two split on one global axis (sign pattern in 2536 of 2573 slots). Controls: `fan4-norepel` oracle minus mixed 0.0128, `fan4-mean` 0.0162. Depth-6 CE: fan4 4.3471, width partner 4.3398. K1-K6 +0.0023. Source: `lab/experiments/failures/2026-09-19-lxtul-fan4.md`.
- Verdict: failure. The fan is 0.0073 nats worse than the width partner at depth 6. The number that decides a stream arm is `oracle - mixed`, not `single - oracle`. Two follow-up arms, `fan4-epi` (epiplexity, gamed by a one-hot stream) and `fan4-epivol` (volume plus epiplexity, not gamed, oracle gain 0.018), read the same.
- Led to: the width-first ladder. "Width of the prefix is worth more than four streams of it."

#### 2026-09-19: Sample-oracle gate, adapted reader (`tul-code-target-uf` at 30000, `lab/divergence/sample_oracle_probe.py`, `exit site: code_proj`)
- Idea: repeat the 09-18 gate on the reader trained on the cell (the unfrozen code-target checkpoint).
- Why: test whether a reader trained on the cell recovers more of the cell's direction.
- Result: gain(16) at sigma 0.3 depth 6 +0.0149 (band 0.02 to 0.06), sigma 1.0 +0.0388. The adapted reader recovers 1.4 to 1.6 times the frozen reader, not the predicted 2 to 5. Exit gain is below entry gain (0.0130 against 0.0149). cos_exit 0.985 (sigma 0.3) and 0.849 (sigma 1.0). Source: `lab/experiments/failures/2026-09-19-sample-oracle-adapted-reader.md`.
- Verdict: failure. Three of five predictions failed. Wolfe's review (16:50 UTC, same file) withdraws the branch-opening rule as a proxy for fan4. The reason: both checkpoints were trained deterministically with the terminal fixed-point term at weight 1.0, which pulls exits together. A map trained with noise or streams was never tested. The fan4 and pk4 queue lines were restored, and the direct test is fan4 against fan4-norepel.
- Led to: the fan4 result above, which was filed as the direct test.

#### 2026-09-19: Depth ladder, mean_depth as a depth dial (`notul_norm_match_20k.yaml` top rung, `notul_norm_match_20k_p3.yaml`, `_p4.yaml`, `_d1.yaml`, `_d2.yaml`, `_d3.yaml`, `_d3fixed.yaml`, `_d6fixed.yaml`; `mean_depth` on the config)
- Idea: a Poisson mean depth ladder on the plain loop at 20k steps. Rungs differ in mean and in whether depth is clamped or fixed. Prereg `.../failures/2026-09-13-arc-depth-ladder-ship.md` (planned 09-13, scored 09-19).
- Why: find which core depth ships. The plain-loop value over its depth-1 twin was 0.0674 at 20k.
- Result (20k, token-paired against top rung, mean 6, max 8, depth 6 CE 3.4516): `d1` +0.0674 [+0.0647, +0.0701]. `d3` clamped +0.0095. `p3` (Poisson 3, max 6) -0.0013 [-0.0036, +0.0010]. `p4` (Poisson 4, max 8) -0.0051 at depth 6. Own K1-K6 rises monotonically with mean: d1 -0.142, d2 +0.002, d3 +0.039, p3 +0.048, p4 +0.071, top +0.170. Seed spread (`d3fixed` at two seeds) is 0.0237 at training depth, so CE differences under 0.024 are ties. Wall clock: `p3` 69 percent of top rung, `p4` 94 percent. Source: `lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md`.
- Verdict: `p3` ships by the file's own rule. The file says mean_depth is a dial on the depth axis, not on final CE. Correction 09-14: clamped and fixed rungs are out of the ship set. A clamped rung has no depth axis worth the name. Correction to the Method: wall-clock bracket was off by a factor of 2. Caveats: one seed per rung, one horizon (20k), no generation quality.
- Led to: a planned 20k vs 40k horizon check with a second seed. Not run in this range.

#### 2026-09-19: Hop staircase replicates (note)
- Idea: the staircase was measured on a different checkpoint and row set. Test it again.
- Why: a single draw is not a result.
- Result: disjoint rows reproduce every bin. See the plateau entry above.
- Verdict: stands.

### Retractions and corrections in this era

- The horizon-fixed6 K1-K6 of +0.278 (09-14). Out of distribution, not depth.
- The "encoder code - ruler" of -2.550 for the p 0.5 TUL-Code arm (09-14). A norm defect in the truth cell, fixed in commit bed41d7.
- The trajectory-prefix K1-K6 of +0.0342 (09-13 file, scored 09-14). Pad-cell artefact of forced-depth eval.
- The frozen-coda K1-K6 for the detached code target arm, +0.0323 (09-17). Retracted as depth by the genericity note (09-18).
- All OWN, SHUF and ZERO probes on the code target taken before commit 14f004d (09-17). An open-slot bug gave every valid slot a zero cell at eval. Void.
- The code-only collapsed draw (09-17). Void.
- The code-only frozen-reference K1-K6 and shuffle cost (09-18). Void, because the coda had drifted above uniform. The own minus shuffled 0.138 and 0.110 stands.
- The graded arm's clean margin: +0.0052 corrected to -0.004 after a sentinel bug (commit b3381dd). P-G2 still fails.
- "Dilution" as the name for carried-content decay (09-18), withdrawn on 09-19 for carried content.
- "Iteration 1" toy conclusion that a slot loop cannot carry past two hops (09-18), qualified on 09-19. A shared-core loop carries one or two hops for free.
- The "7.4 nats = nothing" figure in the spec's §17 text. Never measured (spec §17.1).
- The 09-18 sample-oracle "closed" verdict. Qualified by Wolfe's 09-19 review. The fan4 and pk4 queue lines were restored at 16:50 UTC.
- The depth ladder's clamped rungs (09-14 correction). Out of the ship set.
- The depth ladder's wall-clock bracket (Method). Off by a factor of 2.
- The renorm's "log 8 cap" wording (09-15 02:20 correction). It caps the gap against the best draw, not the mean.

### Where two files disagree

- Carried content decay. `2026-09-19-hop-distance-plateau-and-dilution.md` says the loss is per-pass decay of carried content. `2026-09-19-loop-carry-prev-reach1.md` says the far-bin loss is dilution by accumulation, and that re-supply stops the decay but does not start the far hop. Both use the same row set and checkpoint family. Neither has a test that separates them.
- Sample oracle. `2026-09-18-sample-oracle-gate.md` says the cell's direction is irrelevant to the reader. `2026-09-19-sample-oracle-adapted-reader.md` says an adapted reader recovers 1.4 to 1.6 times more. Wolfe's 09-19 review says neither result tests a map trained with noise or streams.
- Detached-cell cosine at 20k. The planned file reads val cosine 0.147. The code-only reference file reads train cosine 0.1413 for the same arm. These are different metrics (val against logged train mean), not a conflict in the run.
- Spectral gap. `2026-09-14-arc-loop-diagnostics.md` finds no gap (ratio 1.05). It notes that the 2026-08-24 alignment reading (directions align times 2.9) was taken at another checkpoint. The two were not reconciled.
- TUL-Code design status. The spec (`docs/tul-code-spec.md`) says the FM binding rule was waived for TUL-Code on 09-14. The 09-14 TUL-Code note keeps status "proposed" pending Wolfe's call.

### Threads left open in this era

- Bounded carry that keeps the pass-1 read. Design is in `.agents/notes/proposed/architecture/2026-09-19-loop-carry-reinjection.md`. The success test is the per-hop K-curve, with h5 and h6 not worse with depth.
- Noise-trained or stream-trained maps for the sample oracle. Wolfe's review asks for a direct test: fan4 against fan4-norepel on `fan/oracle_ce - fan/single_ce` and stream cosine.
- Code target InfoNCE twin (`tul_code_only_nce.yaml`). Planned, never filed.
- Code target CE companion (`tul_code_target_ce.yaml`). Stopped at 15000. Its checkpoint at 15000 is kept. The continuation to 20000 never ran, so the `tul-code-target-ce-uf` unfreeze twin never ran.
- Pure graded arm (`tul_code_grade.yaml`, weight 0). Not run. Its predicted failure is a cell closer to the sampler's style.
- Coda-space XM (`tul-code-xmc`) and noise-search XM (`tul_code_xmn.yaml`). Coda arm queued, never filed. Noise-search arm planned, never run.
- Thinker phase 3 (`tul_code_thinker_p3.yaml`) and 50k continuations. Planned, never run.
- LoopMTP token loop. Probe output at 20k is unfiled. No K-curve or CE filed. The prereg has no Results section.
- Depth ladder horizon check at 20k vs 40k, with a second seed per rung and depths 4 and 5 in the sweep. Planned in the ladder file, not run in range.
- Plateau and dilution. Hop probe at depths 4 and 5 on a two-token planted copy, and the near-bin dilution read directly.
- Two toy questions: whether the last two hops in the six-hop toy can be bought by anything other than width, and whether a fitted reader on real slot cells separates "head cannot read" from "state does not hold".
- Frozen versus adapted reader for any depth claim. The 09-18 genericity note says a frozen-coda K-curve needs a warning. No later arm in this range used the adapted reader for depth.
- The generation side of the semantic probe. Distinct-2 on the adapted and frozen readers differs. No one has scored a sample with a reader trained on samples (`tul-code-rollout1-20k` was the nearest, 1 of 10 topic threads).

---

## TUL history, 2026-09-20 to 2026-09-25

Source root: MORPH repo (`/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`).
Paths below are repo-relative. Read-only pass. No GPU, no git writes.

### Era summary

Six days, two campaigns, one closing call. The K-stream fan (the LXTUL fan, `fan_mix`) was the working
design at the start. Its diversity terms (cosine, epiplexity, volume) were gamed or read as no help, so the
team moved to a write-all route where every stream gets its own prefix cell and the coda picks per token.
That write-all arm (`fan4-all`) was the one success on width. The LXTUL-P ladder then tried to give each pass
a job (denoising, lineages, triggers, noise entries, no prelude, a semantic target). Every rung failed on its
own mechanism. Wolfe's 2026-09-22 call closed the restriction geometries, and the relay chain (LXTUL-R) was
measured and closed too. From 2026-09-23 the work shifts to proofs (Lean), stochastic loops (failed), enumerated
codes (LXTUL-E, width earns, loop adds little), and a diagnosis of the slot map's 0.87. The last two days found
the first deployed depth use on the strict slot family: a loop with the fixed-point term off or at 0.1 earns
K1-K6 of +0.012 to +0.018, and it grows from 5k to 10k steps. Every positive is small and rests on one or two seeds, and the
campaign's standing bar stays the same: flat K-curves are the default, and each positive must replicate.

### Entries

#### 2026-09-20: Diversity term on the fan, cosine and epiplexity (`tul.fan_repel_mode: cos | epi | epivol`, `slot-spandec-strict-fan4`, `-epi`, `-epivol`)
- Idea: K = 4 latent streams per span (`fan4`). A cosine repulsion (`fan_repel_lambda 0.1`, passes 1 and 2) was
  the first term. The `epi` mode swaps it for an epiplexity score. `epivol` adds a within-slot volume term.
- Why: the cosine repulsion sat on its floor (mean pairwise cosine -1/3) by step 500. The stream probe found a
  rank-1 split on a fixed axis (`lab/experiments/failures/2026-09-19-lxtul-fan4.md`, note
  `.agents/notes/proposed/architecture/2026-09-20-fan-epiplexity-diversity.md`).
- Result: `fan4-epi` (5k, final val 4.4878): stream rank 1.0001 of 4, one-hot stream per slot. The oracle over
  streams beat the mixture by 0.011. `fan4-epivol` (5k, final val 4.4616): stream rank 2.823 at pass 1 and 2.312
  at pass 6, live and near-orthogonal streams. Still, the oracle beat the mixture by only 0.018, and the mixture
  put 71 % of its weight on stream 0 (`fan/oracle_pick0` 0.711). Streams 1 to 3 read 0.14 to 0.42 nats worse
  than the mixture. Source: `lab/experiments/failures/2026-09-20-lxtul-fan4-epi.md`,
  `lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md`.
- Verdict: both failures. The term acted, and the falsifier (oracle gap above 0.022) failed both times. The
  first arm met the term with a shape that carries no diversity. The second had diverse streams the coda still
  did not use. The filing says three diversity terms (cosine, epi, epivol) all leave the oracle inside
  0.008 to 0.018 of the mixture.
- Led to: the coda reads one stream because it only trains on the mixture. That led to select-then-commit.

#### 2026-09-20: Select-then-commit fan (`tul.fan_mix: select`, `tul.fan_select_eps 0.05`, `tul.fan_select_gate_lambda 1.0`, `slot-spandec-strict-fan4-select`)
- Idea: no mixture. Each slot writes ONE stream into the coda. The table of K no-grad passes (coda CE per stream)
  picks the winner, a gate learns to predict the winner before the span, and an epsilon of random writes keeps
  every stream readable.
- Why: under the mixture, the coda learned one reader and the rest of the streams were never read. Winner-takes-all
  responsibility was meant to make each stream individually useful.
- Result (5k, final val 4.5582): winner-takes-all worked on the training side. The oracle over four streams sat
  0.113 nats below the best single stream, and every stream had a reader (spread 0.076). The gate recovered
  0.014 of the 0.113 (33 % agreement with the oracle, selector regret 0.099). The deployed one-stream write sat
  0.086 nats behind the width partner `slot-spandec-strict-pk4` at depth 6 (paired, 511k tokens). Source:
  `lab/experiments/failures/2026-09-20-lxtul-fan4-select.md`.
- Verdict: failure (P-3 and P-5 failed). Wolfe's reading, 2026-09-20: the arm trains the coda on the table's
  winner and deploys the gate's guess. That is "the fitted-z trap in a new place", so the gap is not a ranking.
  The verdict's "gate cannot predict" reading was therefore not the whole story. Note the correction in the next
  entry.
- Led to: the write-the-gate's-pick follow-up and the write-all arm.

#### 2026-09-20: Select arm writes the gate's pick (`tul.fan_select_write: gate | anneal`, `slot-spandec-strict-fan4-select-gate`)
- Idea: the training pass writes the gate's own argmax, so the coda trains on the input distribution it will see
  at eval. In the `anneal` mode the write moves from the table's winner to the gate's argmax with probability
  step/1500. The table still runs each step, but only as the gate's label.
- Why: the select arm's 0.086 deficit might be the train/eval mismatch and not a gate limit. Wolfe asked for
  this test: "I feel like this cant transfer to inference at all, and may be part of our over all problem ...
  this sounds easily resolved ... go, queue it behind write-all."
- Result (5k, final val 4.4922): the mismatch closed as predicted. The deployed write recovered 0.066 of the
  0.086 deficit. The closed loop of gate, coda and table settled on one stream within 500 steps of the gate write
  turning on. The strict ruler at prefix_k 4 still sat ahead, with the K-pass table paying +0.020 nats over it.
  Source: `lab/experiments/failures/2026-09-20-lxtul-fan4-select-gate.md`, note
  `.agents/notes/proposed/architecture/2026-09-20-fan-select-gate-write.md`.
- Verdict: failure (P-5 failed; P-1 and P-2 held). Correction: this filing and the select note both say the
  select filing's "gate cannot predict" reading was the train/eval mismatch. Both notes keep the select result
  as filed.
- Led to: the write-all arm, which drops the gate altogether.

#### 2026-09-20: Write-all fan with per-token coda selection (`tul.fan_mix: all`, `tul.fan_all_wta_lambda`, `slot-spandec-strict-fan4-all`)
- Idea: every stream goes into its own prefix cell (`prefix_k == fan_k`, the register's 1:1 route). The coda's
  attention over the four cells is the selector. Training uses winner-takes-all responsibility: each stream
  alone in its cell is scored by K no-grad coda passes, and the best one is replayed with grad. No gate.
- Why: the select arm wrote one stream and threw the rest away. The Thought Register wrote four cells the same
  way and collapsed to rank 1.24 of 4 with no responsibility term. The write-all arm adds that term.
- Result (5k, final val 4.4376): oracle 4.3925, deployed mixed read 4.4348, selector regret 0.042. Stream rank
  2.820 at pass 1 and 2.060 at pass 6. Paired against `pk4` at depth 6, the arm sits 0.034 nats BELOW it
  (depth 1: -0.029). Tokens K1-K6 +0.0049 [+0.0044, +0.0055], K3-K6 +0.0004. Source:
  `lab/experiments/successes/2026-09-20-lxtul-fan4-all.md`, note
  `.agents/notes/proposed/architecture/2026-09-20-fan-write-all-wta.md`.
- Verdict: success under the filing rule. The coda recovered 0.056 of the 0.099 nats of oracle value that the
  gate could not reach. Direction from Wolfe, 2026-09-20: "do the code/mux tokens hold the other explorations? ...
  we should at least try that." The filing also says the arm does not make the loop earn depth (K-curve flat),
  and the P-8 first clause was miswritten and recorded with a substitute reading.
- Led to: the LXTUL-P ladder. This is the only fan arm in the range that beat its width partner.

#### 2026-09-21: Mixture probe on the write-all read (`lab/divergence/fan_mixture_probe.py`, fan4-all at 5k, 48 rows, depth 6)
- Idea: measure the fan4-all read against the best possible mixture, without training anything new.
- Why: the write-all filing left a 0.042 "selector regret" that could be read as a gate defect or as a price.
- Result (nats per token, 95 % bootstrap): oracle 4.2082, deployed all-cell read 4.2485, uniform mixture 4.2583,
  best single stream 4.2874. The deployed read beats the prefix-weighted mixture by 0.0098 [-0.0119, -0.0077], in
  every offset bin. The uniform mixture pays 71 % of the log K bound. Winner persistence is 0.304 against 0.268
  chance. Source: `.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`
  (outcome log, probe 1), artifacts under `lab/experiments/results/2026-09-19-lxtul-fan4/`.
- Verdict: the selector question is closed. The 0.042 regret is the price of hindsight, not a gate defect. The
  lever left for the read is better candidates (ladder rungs P1 to P3), not a better selector. Not explained: why
  a learned joint read beats every convex mix of the streams' predictions (untested hypothesis).
- Led to: the LXTUL-P ladder, which tests candidates.

#### 2026-09-21: Mixed fan with no fixed-point term (`model.core_fixed_point_lambda 0.0`, `slot-spandec-strict-fan4-all-fp0`)
- Idea: the write-all arm with the terminal fixed-point term removed (ladder rung P0).
- Why: the fixed-point term was the one loss on the exit state that might be pulling K deviations together.
- Result (5k, final val 4.4368): pass-6 rank 2.01 against 2.06, so the term does not contract the K deviations.
  With the term off, every stream's norm grows 2.3 to 4.2x through the loop. `loop/core_gain_t0` falls from a
  9.92 max to 2.81. The early pre-clip is 18x larger (max 522 at step 239, still under the 1e4 tripwire). The
  coda reads the larger exit 0.006 nats better. Source: `lab/experiments/failures/2026-09-21-lxtul-fan4-all-fp0.md`.
- Verdict: failure (the arm's reason failed). The term held the exit's SCALE, not the streams' spread. The
  `loop/core_gain_t0` drop is a ratio the growing state changes. It is not a stabler loop.
- Led to: the fixed-point term is read as a scale control from here on.

#### 2026-09-21: Streams as K independent samples (`tul.fan_seed_noise`, `slot-spandec-strict-fan4-all-noise`)
- Idea: one unit-RMS random draw per stream per slot added to the entry state (not to the injected `e`), so the
  K streams start as K samples (ladder rung P1).
- Why: if the K streams start apart and the loop keeps them apart, the passes get K candidates to work on.
- Result (5k, final val 4.4352): the entry norm grew 11x over training and the four streams read as one (cosine
  0.99 at pass 0). Source: `lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md`.
- Verdict: failure. Nothing charges the model for making the draw small against the state, so the draw is
  swamped. The volume term was load-bearing for the rank after all.
- Led to: the honest base for the next rungs is write-all with the term.

#### 2026-09-21: Streams as lineages (`tul.fan_lineage`, `slot-spandec-strict-fan4-all-lineage`)
- Idea: each stream is a lineage. Stream i of slot s+1 is linked to stream i of slot s (ladder rung P4).
- Why: winner identity persists a little more than chance (P(winner repeats) 0.304 against 0.268 chance, excess
  +0.036 [+0.017, +0.057]), so a lineage might carry the winner.
- Result (5k, final val 4.4479): the channel held the centred rank at 2.0 (against 1.3 without). Paired reader
  gain +0.004, inside the spread. Winner persistence +0.044 against +0.018 and +0.036 for the controls. Source:
  `lab/experiments/failures/2026-09-21-lxtul-fan4-all-lineage.md`.
- Verdict: failure. The channel costs nothing and buys nothing. The note says lineages need the weighting, and the
  weighting needs a different forward.
- Led to: no further lineage arms.

#### 2026-09-21: Per-pass trigger re-injection (`tul.fan_trigger_every_pass`, `slot-spandec-strict-fan4-all-trig`)
- Idea: re-supply the per-stream trigger at the start of every pass, so each pass sees its stream's identity
  again.
- Why: a stream's identity may decay across passes.
- Result (5k, final val 4.4427): exit rank 1.43 of 3 (write-all 2.03). One stream ends at 4.1x its siblings. The
  `+---` sign family reaches 95 % by pass 3. Source: `lab/experiments/failures/2026-09-21-lxtul-fan4-all-trig.md`.
- Verdict: failure. A shared map sees the offset every pass and can learn to cancel it.
- Led to: nothing new. Closed.

#### 2026-09-21: Slot loop as a denoiser (`tul.loop_denoise`, `tul-code-target` base, 20k steps)
- Idea: each pass learns to denoise the code. The rollout starts from noise at pass 1 and the loop refines it
  (ladder rung P2, single stream form).
- Why: a denoiser gives each pass a job whose output the loss reads.
- Result (20k, HEALTHY, 490 blocks paired): the rolled-out sample reads 0.126 nats WORSE than the regressed mean
  of the base arm at depth 6. The sample is 4.74 nats worse than no cell. Pass 1 never learned: its cosine to
  the target is 0.02, flat for 20k steps. Under teacher forcing the summed loss falls only in passes 2 to 6, and
  the rollout inherits pass 1. Source: `lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md`.
- Verdict: failure on the rung's reason. The note's falsifier (the rung must beat the rung below) fails. Filing
  line: "Teacher forcing is a bypass." A pass gets a job only if the loss reads that pass's output with no
  shallower route. Under teacher forcing pass i's input is the true prefix, so later passes learn and pass 1 does
  not.
- Led to: no schedule sweep queued.

#### 2026-09-21: No prelude on the strict slot loop (`model.n_prelude: 0`, `slot-spandec-strict-np0`)
- Idea: remove the prelude so the loop's entry is the raw embedding (ladder fallback, Wolfe 2026-09-21: "If these
  fail we should test removing the prelude.").
- Why: on the plain norm_match loop, a prelude entry earns 0.033 and a noise entry earns 0.185. The prelude might
  be taking the easy part.
- Result (5k, final val 4.5962): K1-K6 +0.0045, K3-K6 -0.0008. The entry's rank doubles and the loop moves the
  state almost twice as far. Source: `lab/experiments/failures/2026-09-21-strict-np0.md`.
- Verdict: failure. The prelude-entry explanation does not transfer to the slot loop, which is flat under any
  entry.
- Led to: the ladder ended. The next design note (LXTUL-R) started from a reach composition, not an entry.

#### 2026-09-21: Strict partner at prefix_k 8, the width ruler (`slot-spandec-strict-pk8`)
- Idea: one continuous stream written into eight prefix cells with no quantizer. This was the missing width ruler
  for the discrete-thought panel.
- Why: the vq8 and vq4 arms (2026-09-13, `lab/experiments/mixed/2026-09-13-arc-discrete-thought-vq.md`) were
  never compared with a same-width continuous arm.
- Result (5k, final val 4.4118): the continuous eight-cell write is 0.035 nats BETTER than the two-cell strict
  partner. The eight-code quantizer is 0.059 nats WORSE than this width ruler. vq8 beat vq4 by 0.033, which is
  code capacity, not width. Source: `lab/experiments/successes/2026-09-21-strict-pk8-ruler.md`.
- Verdict: success. The discrete write's deficit is the quantizer's own. Its capacity is a lever in code count.
- Led to: the residual-code form was named as the next discrete arm.

#### 2026-09-21: Flow noise shifted to high noise (`tul.code_t_logit_mean -1`, `tul-code-cfg-tlow`, 20k)
- Idea: move the LCTUL flow thinker's noise schedule to high noise (E[t] 0.30 against 0.50), so the thinker
  sees less of the target and more of the context (LXTUL-P rung P2, first half).
- Why: the flow thinker was context-blind. The past was worth 1 % of its loss (`lctul-thinker-is-context-blind`).
- Result (20k, final val 4.4686, HEALTHY): context dependence doubled, 1 % to 1.9 %. One sampled code is retrievable
  in-batch at 2.7 times chance. One draw reads 0.010 nats better than the parent. The ceiling against the strict
  ruler stays at +0.64 nats. Source: `lab/experiments/failures/2026-09-21-lctul-cfg-tlow.md`.
- Verdict: failure on the arm's reason. The parent's diagnosis stands: on a code whose predictable-from-context
  fraction is a tenth, no schedule makes the field read the past.
- Led to: the semantic target (planned, see below).

#### 2026-09-21: Semantic target for the code (`tul.code_target_source: sonar`, planned only)
- Idea: the LCTUL target is a sentence embedding, not the frozen encoder's code (ladder rung P3). Paired at 20k
  against the flow arm above.
- Why: the code target might be verbatim and so easy to memorise. A semantic target would make the code carry
  meaning.
- Result: not run in this range. The prereg is frozen (`lab/experiments/planned/2026-09-21-lctul-tlow-sonar.md`)
  and needs a cache check first.
- Verdict: open.
- Led to: the 2026-09-22 EMA target arm (below) took the target line instead.

#### 2026-09-21: The far-budget split (`loop_reach`, `tg_coda_prefix_reach`, span-reach arms, four runs)
- Idea: measure how much of the 0.40-nat cross-span budget is the PREVIOUS span, and how much is further back.
  Four arms: span-h (no reach), reach 1 (previous slot), reach all, and reach 1 at coda layer 0 only.
- Why: the LXTUL-R design needs to know whether the loop's relay (passing content across slots) has a far
  budget worth chasing.
- Result (5k, depth 6, 481 blocks): span-h 4.3975, reach1 4.2181, reachall 4.0922. The far budget's lower bound
  (reach1 minus reachall) is +0.1259 [+0.1165, +0.1364]. The previous span's value has an upper bound of 0.179.
  The coda-layer-0 arm reads 0.044 worse than span-h. Source: `lab/experiments/failures/2026-09-21-span-reach-split.md`,
  design note `.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md`.
- Verdict: failure on P-4, P-8, P-9 (6 of 9 hold). The headline holds: the far budget is at least 0.126 nats, so
  Step 1 was queued and launched on that bound alone.
- Led to: LXTUL-R Step 1.

#### 2026-09-22: Token-like read of the slot exit (arms `bcast`, `bcast-all`, and the strict ruler)
- Idea: give the coda a direct token-like read of the slot exit state, the way the plain loop's output reaches the
  token loss. The `bcast` arm reads at the entry. `bcast-all` re-adds the term at every coda layer.
- Why: every slot positive stayed at or below 0.026 and did not grow with steps. The paid loop and the plain loop
  both have a direct token read. Reference note: `.agents/notes/proposed/architecture/2026-09-22-slot-token-like-read.md`.
- Result (5k, 480 rows, 490 blocks): the bcast read is used and worth 0.012 to 0.017 nats, almost all at the
  span's first token. The K-curve is the strict ruler's to the fourth decimal under either read (+0.0016 bcast,
  +0.0013 bcast-all, +0.0016 ruler). Re-adding the term at every layer is 0.004 WORSE. The horizon pair shows the
  slot loop worth a constant 0.0075 nats over its depth-1 twin from 5k to 20k. Source:
  `lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md`.
- Verdict: failure on the question. "The read was not the limit." The loop's job is the open question. P-5
  holding with P-3 failing files as a horizon effect that the 5k K-curve cannot see.
- Led to: no 20k pair was queued (P-6 not run).

#### 2026-09-22: Seed twins of the single-block reach arm (`budget-web-reach1-coda`, seed 1 and seed 2)
- Idea: two seeds of the one-block reach config. The seed-1 arm had looked like a learned trade between reach and
  the coda.
- Why: the Step 0 filing read a "learned single-block trade". Two runs were needed to separate a learned trade
  from a run draw.
- Result: two seeds of the reach config differ by 0.152 nats at depth 6 on the same rows. Two seeds of the plain
  span config differ by 0.004. At seed 2 the single non-looped block that reads the previous span is worth 0.112
  nats at every offset. Source: `lab/experiments/failures/2026-09-22-coda-seed-twin.md`.
- Verdict: failure (P-3, P-4 fail). The learned-trade hypothesis is refuted. The seed-1 arm was a bad draw of
  that config.
- Correction: the Step 0 filing's coda paragraph, its "learned single-block trade" reading, and the brackets it
  declined to read are corrected by a dated note in that filing. The design note's outcome log is corrected too.
  The seed spread (0.152) is now the noise floor for reach arms.
- Led to: Step 1 proceeded on the far-budget bound alone.

#### 2026-09-22: EMA code target (wandb runs `ema`, `ema0`, `ema-l2`; Stage 1 of LCTUL-J)
- Idea: the code target is an exponential moving average of the online front's pooled states, with a variance
  floor on the online cells (`.agents/notes/proposed/architecture/2026-09-22-lctul-ema-target-and-factors.md`).
- Why: the frozen-encoder target was not moving with the loop, so the regression target was fixed. An EMA target
  would track the loop without collapsing.
- Result (5k, two draws plus one retry): the EMA target moved on both draws (rank 77 to 40 per cell, cosine to
  the corpus mean 0.06 to 0.53). The reader saw none of it. Val loss matched the frozen twin to three decimals,
  cell worth 0.165 / 0.168 / 0.174 nats, token K1-K6 at the floor (+0.0006 / +0.0016 / +0.0014), paired token CE
  inside +-0.005. Source: `lab/experiments/failures/2026-09-22-lctul-ema-target.md`.
- Verdict: failure on both draws. The note's first risk (a slow drift of the target into a low-rank code) is the
  finding.
- Correction (Stage 1 errors, named in the outcome log): Stage 1 put the encoder loss on the predictor's output.
  The paper floors the ONLINE ENCODER's output, which is the tensor the EMA copies. Also `val/ce_tf` collapses to
  the token CE on the frozen control too, so it is a statement about the reader, not the target.
- Wolfe's call, 2026-09-22: no arm passes or fails on a cosine. The verdict rests on nats and rank.
- Led to: Stage 2 not built on this target.

#### 2026-09-22: LXTUL-R Step 1, the chain on a leaky geometry (`slot-spandec-strict-fan4-all-reach1`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `core_fixed_point_lambda 0`, `slot_state_renorm True`)
- Idea: the fan's K streams on a relay chain. Each slot reads the previous slot's exit through a loop reach of one.
- Why: the reach split said the far budget was at least 0.126 nats.
- Result (5k, 1.21 steps/s): token K1-K6 +0.0202 [+0.0193, +0.0212], the largest slot-loop K-curve on the ledger at
  the time. Paired -0.033 against strict, +0.019 against fan4-all-fp0 at depth 6. Channel worth 0.162 (fan4-all
  0.196). But the planted row read content three and four spans back at depth 1. The register branch passed
  `tg_relation` without `tg_seg`, so the CCA conv and the value shift relayed about one slot per LAYER, four
  slots per pass. Source: `lab/experiments/failures/2026-09-22-lxtul-r-step1.md`.
- Verdict: failure, method fault. The arm did not run the design's geometry. None of the binding clauses applies.
  The numbers stand as the record of the LEAKY chain.
- Correction: the chain's K-curve is a record of a leaky geometry. Fixed at 9b430d3 with a per-slot `tg_seg`
  under `loop_reach > 0`, and a test for exact zeros beyond one slot per pass.
- Led to: Step 1b.

#### 2026-09-22: LXTUL-R Step 1b, the corrected one-slot-per-pass relay (`loop_reach 1`, one slot per pass)
- Idea: the same arm as Step 1 at the corrected geometry.
- Result (5k, 1.03 steps/s, wandb `u8xh91vd`): token K1-K6 +0.026, the largest slot K-curve of the strict family.
  The late passes do twice the work they did under the leak (K3-K6 +0.0057). The cost is 0.053 nats at depth 1
  against the direct-read control, and the depth-6 CE sits 0.0265 behind fp0. Source:
  `lab/experiments/failures/2026-09-22-lxtul-r-step1b.md`.
- Verdict: failure (P-3, P-4 fail). The forced relay makes the loop carry the content at a CE tax. The synthesis
  note reads the chain's K-curve as "a tax, not a gain."
- Led to: the Step 2 panel.

#### 2026-09-22: LXTUL-R Step 2 panel, persistent carry (C1) and history stream (C2) (`lxtul-r-step2` arms)
- Idea: C1 keeps an accumulator that persists the arrived content (`carry/persist_ratio`). C2 adds a history stream
  of three plan streams.
- Why: Step 1b's relayed content decays. Keeping it or separating it from the diversity term might pay.
- Result (5k): C2 hist1 token K1-K6 +0.0204, C1 persist +0.0234 (bar +0.035). Paired depth-6 CE against fp0
  +0.0255 and +0.0248 (bar +0.015 or less). Against Step 1b the gap is -0.0010 and -0.0018, both inside their
  intervals. C1 costs 0.6 % more rate. Source: `lab/experiments/failures/2026-09-22-lxtul-r-step2-panel.md`,
  speed and carry note `.agents/notes/proposed/architecture/2026-09-22-lxtul-r-step2-speed-and-carry.md`.
- Verdict: failure on both arms. The relay itself is the limit. Wolfe's calls: the chain geometry is a measurement,
  not a final design. C3 and C4 are not queued, and the chain geometry is closed as a measurement.

#### 2026-09-22: Slot-loop campaign synthesis (standing filter, implemented note)
- Idea: a record of about forty slot arms from 2026-09-09 to 2026-09-22 and a design filter. Note:
  `.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md`.
- Result: twelve designs without a forced relay read token K1-K6 in [-0.0001, +0.0033]. The paid loop earns
  0.1685 with slots present but is rejected as "not TUL" by Wolfe (tokens run through the core).
- Decision: every slot-loop proposal states two conditions before a run. Condition A: no shallower route can do the
  pass's job. Condition B: the loss pays for the pass's output. Standing call (Wolfe, 2026-09-22): no more
  restriction geometries whose mechanism is to cut the coda's read so the loop must matter. Also: "this is not a
  universal truth" (on per-pass targets met in one step). Also: never argue that web text does not need loop
  depth.
- Verdict: implemented. This note is the filter for everything after 2026-09-22.

#### 2026-09-23: Stochastic slot loop as a latent-variable model (`LXTUL-G`, `lxtul_g_probe.py`)
- Idea: each pass takes a posterior step (training) or a prior step (eval) from a learned Gaussian. The loop is a
  latent-variable model (note `.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md`).
- Why: a stochastic loop might explore and keep alternatives, which the deterministic loop could not.
- Result (5k): the latent does not collapse, and the coda reads it: a shuffled cell costs 1.12 nats at a span's first
  token. The exposure gap (posterior trained, prior deployed) is 0.9840 nats at 5000. Source:
  `lab/experiments/failures/2026-09-23-lxtul-g-panel.md`.
- Verdict: failure. The coda reads the posterior's cell. Training took posterior steps and eval takes prior steps.
  The reader is trained on cells the loop never produces.
- Led to: LXTUL-GK (below).

#### 2026-09-23: Stochastic loop trained on K prior rollouts (`LXTUL-GK`, `gk4-shared`, `gk1`)
- Idea: train on K rollouts drawn from the prior, under a multi-sample log-mean-exp bound. The gap should close.
- Result (5k): the exposure gap closes as designed (P-1 holds). But the learned noise switches itself off: sigma/r
  goes from 0.1 at init to 0.00026. The width gain at four samples is 0.00005. K1-K6 is +0.0036 for `gk4-shared`.
  `gk1` (one sample, same function) reaches final val 4.4238 against the ruler's 4.4249. Source:
  `lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md`.
- Verdict: failure. The objective prefers a deterministic loop. The log-mean-exp aggregate has second-order gain
  from spread and first-order cost on the mean. The filing cites XM (arXiv 2607.27372, App. F.1), which uses a
  hard min to pay for spread at first order.
- Led to: the provable-loop design note (below).

#### 2026-09-23: Provable loop contribution (design note, Lean proofs, LXTUL-E plan)
- Idea: prove what width and depth need, using Lean 4 with Mathlib v4.31.0 (`lab/theory/tul_exploration/`), then
  derive one design. Wolfe, 2026-09-23: "we need to actually prove and design a real solution. Maybe in lean."
- Result: a plan with four stages. Stage 0 measures the width budget of the ruler's cells under a committed reader
  (enumerated codes). Stage 1 and beyond follow. Note: `.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md`.
- Verdict: proposed. The existing Lean development proves a deterministic pass adds no information, so the loop can
  earn only by a relay or by extractability.

#### 2026-09-23: Enumerated codes on a committed reader, width budget (LXTUL-E Stage 0, `e0k1`, `e0k4`)
- Idea: four enumerated codes under an exact mixture, read by one committed reader on the ruler's frozen cells.
- Result (2000 steps, one seed per arm): four codes buy 0.103 nats per span token over one committed reader. The
  product cap is large (a parallel reader of one cell sits 2.41 nats behind the teacher-forced decoder). The codes
  act as a span type chosen by the first token. Source: `lab/experiments/successes/2026-09-23-lxtul-e-stage0.md`.
- Verdict: success on the deciding clause. Stage 1 runs as written. Failed clause: P-S2 read low responsibility
  entropy (0.33 of 1.39) as collapse. It is the opposite: spans are assigned decisively.

#### 2026-09-23: Does a trained cell hold a superposed frontier? (`lab/divergence/superposition_probe.py`, slot cells)
- Idea: is a slot cell a superposition (several branches), a committed point, or a blur?
- Result (480 rows, about 486k scored tokens): the verdict is "superposition" on every arm by the rule. The
  span-specific worth is 0.096 nats per token. The ratio is 1.5 to 1.8 at fixed difficulty. The GK arms match the
  ruler. Source: `lab/experiments/successes/2026-09-23-superposition-probe.md`.
- Verdict: success on the predictions, with a caveat. A frontier in the RbS sense predicts a ratio at or below 1.
  The ratio is above 1, so the cell holds content shared by all branches. The probe cannot tell "several
  branches" from "what every branch shares."

#### 2026-09-24: Teacher-forced codes read in parallel (LXTUL-E Stage 1, `e1`, `e4`)
- Idea: a loop that carries an enumerated code and is read in parallel, so width creates a depth job.
- Result (5k, 490 blocks): three of six clauses held. The one the design exists for (P-3) missed by 0.002, with its
  CI wholly below. Separation ratio 1.99 (the refutation rule needs below 1.3). Width survives training, and the
  loop integrates the code (exit states move apart 0.27 to 0.53 over passes). But the loop adds almost nothing over
  a code bolted onto the frozen ruler (e4 6.788 at depth 6, the frozen-cell head 6.791). Width is worth about 0.11
  nats per span token either way. Source: `lab/experiments/failures/2026-09-24-lxtul-e-stage1.md`.
- Verdict: failure as written. Correction of a premise: the teacher-forced coda does not hedge (P-2's premise was
  wrong). One code alone costs 0.0275 over the four-rollout read.
- Led to: Stage 2.

#### 2026-09-24: Committed read points and who pays for width (LXTUL-E Stage 2, `e4j4`, `e4probe`)
- Idea: move the committed read point and check who pays for width (the parallel head or the coda).
- Result (5k): neither J-1 nor J-2 held. Coda K1-K6 is 0.0011 to 0.0020 on all three arms. Cutting the parallel head's
  gradient improves the coda by 0.012. The probe's 4-rollout coda is 0.0176 better than the ruler. Pointing the
  read at the first four tokens costs the coda 0.067. Source: `lab/experiments/failures/2026-09-24-lxtul-e-stage2.md`.
- Verdict: failure. "The next move is the reader, not the objective."

#### 2026-09-24: The slot map's 0.87, the cause interventions (I-0, I-1, I-2, `lxtul-e4probe-map` reference)
- Idea: find what sets the slot loop's per-pass map (gain near 0.87). Three interventions: I-0 forces work, I-1
  moves the injection floor, I-2 removes the per-pass source so the loop must keep the entry.
- Why: the plain loop's map sits near 1 and the slot loop's sits at 0.87 with no depth value.
- Result (5k each, 480 rows): I-1 moved the injection floor to 0.773 and the map to 0.781, with the blocks' part
  unchanged. So 0.87 is an architecture default. I-2 removed the source after pass 0: `A` stayed at 0.443 and the
  map at 0.871. The coda lost 0.016. I-0 sat +0.019 above the floor. Source:
  `lab/experiments/failures/2026-09-24-lxtul-map-cause.md`.
- Verdict: failure. The causal account is not supported (P2-1 failed) and is refuted on its I-2 link. The floor
  finding (0.865 from the injection decay `A` at init) stands. The reading note is `lab/theory/tul_exploration/READ-BEFORE-TUNING-THE-LOOP-MAP-0.87-IS-THE-INJECTION-FLOOR.md`.
- Led to: Stage 3 found the lever this account ranked fourth.

#### 2026-09-24: The fixed-point term holds the loop still (`core_fixed_point_lambda` 0.0 and 1.0, `lxtul-stage3-map`, arms `nofp`, `map`, `e4probe`)
- Idea: Stage 3 asks what holds the map at 0.87 and whether a map near 1 earns depth. Arm `nofp` turns the
  fixed-point term off. Arm `map` forces a train-time gain of 0.955 with a floor hinge, against the
  `e4probe` reference.
- Result (5k, exit 0): `nofp` gives coda K1-K6 +0.0123 (about 10x the ~0.002 slot baseline, 95 % of it by pass 3).
  Its eval map sits at 0.884, rising with pass, no position above 1. Its depth-6 coda is 0.0046 worse than
  `e4probe`. It detonated once and recovered. Arm `map` got no depth value and a worse coda. Source:
  `lab/experiments/successes/2026-09-24-lxtul-stage3-map.md`.
- Verdict: success by the rule. The map is not what moved. A map near 1 was neither necessary (`nofp`) nor sufficient
  (`map`). The fixed-point term is the switch. Note: the hinge's logged gain reads high (0.97 logged against 0.884
  in fp32 on the same arm, a bf16 finite difference). Read `core_map_fd.py` fp32.
- Led to: the replication in the next entry.

#### 2026-09-25: Fixed-point weight and the replication (`nofp` at seed 2, `nofp_s2`, `fp01` at weight 0.1, `map` at weight 1.0)
- Idea: does the fixed-point-off positive replicate at a second seed, and can a weak term (0.1) keep it stable?
- Result (5k, final val, coda K1-K6 with 95 % CI): `nofp` seed 1 +0.0123, `nofp_s2` seed 2 +0.0077 (CI +0.0071 to
  +0.0082), `fp01` at 0.1 +0.0126. At weight 1.0 (`map`) it is +0.0009. Seed 1 `nofp` detonated at step 4048 and
  recovered. Source: `lab/experiments/successes/2026-09-25-lxtul-nofp-followup.md`.
- Verdict: success. The positive replicates (smaller at seed 2). The recipe is YES: the term at 0.1 keeps the loop
  earning with no detonation on this seed. The dose is the switch (1.0 kills it, 0.1 keeps it). Two predictions
  failed: S-2 (missed the size band by 0.0006) and F-3 (coda gain inside seed noise).
- Note: the Stage 3 positive was one seed. The seed-2 replicate is 40 % smaller. The filing says the Stage 3
  positive is one seed plus one replicate.

#### 2026-09-25: Span-level NextLat on the exit state (`fp01` plus NextLat, `lxtul-fp01-nextlat`, proposed note `.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md`)
- Idea: a NextLat auxiliary (predict the next slot's exit from the current exit and the span it emitted) so the exit
  is a belief state the next slot follows from.
- Result (5k, 5373 tok/s): coda K1-K6 +0.0010 [+0.0007, +0.0013] against fp01's +0.0126. Par K1-K6 falls from
  +0.1190 to +0.0201. The exit became a constant. The gradient of the transition loss shrinks each slot's deviation
  from the common mode, because the stop-grad target is not a guard. Source:
  `lab/experiments/failures/2026-09-25-lxtul-fp01-nextlat.md`.
- Verdict: failure. A latent-prediction term on the exit collapses it unless something holds the per-slot content up
  (a strong reader, a frozen or EMA target, or a variance floor). The EMA target (2026-09-22) failed for another
  reason. The NextLat note is still proposed.
- Led to: the fp01 recipe is still the reference.

#### 2026-09-25: The fp01 recipe keeps earning with training (`fp01` resumed to 10k, `lxtul-fp01-10k`)
- Idea: does fp01's depth use grow with training? Resume the 5k run to 10k on the same rows.
- Result (final val 4.4199 at 5k, 4.1154 at 10k; coda K1-K6 +0.0125 [+0.0117, +0.0132] at 5k, +0.0177 [+0.0166,
  +0.0188] at 10k). One excursion recovered in one step. Source:
  `lab/experiments/successes/2026-09-25-lxtul-fp01-10k.md`.
- Verdict: success. Depth use grows on this recipe. Wolfe, 2026-09-25: "take the current 5k step run and give it
  another 5k steps." Wolfe on spikes: a recovered spike is normal for AdEMAMix with ternary weights at this scale.
- Led to: the 20k reading (resume from 10k), which is not in this range.

### Retractions and corrections in this range

- 2026-09-20, select arm: the first filing read the 0.086 deficit as "a gate before the span cannot predict."
  The write-the-gate's-pick arm showed the train/eval mismatch carries most of it (0.066 of 0.086 recovered).
  Both readings are cited.
- 2026-09-20, write-all arm: the 0.042 "selector regret" was read as a lever. The 2026-09-21 mixture probe calls it
  the price of hindsight (oracle is not reachable before the span).
- 2026-09-22, coda seed twin: the Step 0 "learned single-block trade" reading is retracted. Seeds differ by 0.152
  nats on the same config.
- 2026-09-22, Step 1 (LXTUL-R): the K-curve of +0.0202 is a leaky-geometry record. The arm relayed about four slots
  per pass. Not a valid reading of the design.
- 2026-09-22, EMA target stage 1: two named errors (the encoder loss sat on the predictor output; `val/ce_tf` is a
  reader statement). The filing stands as a failure on both draws.
- 2026-09-22, fan4-all P-8 first clause: miswritten in the prereg and recorded with a substitute reading.
- 2026-09-24, hinge gain: logged bf16 gain reads 0.97 on the Stage 3 arm against 0.884 in fp32. Use `core_map_fd.py`.
- 2026-09-25, Stage 3 positive: not retracted, but the seed-2 replicate is 40 % smaller (+0.0077 against +0.0123). Read the size from the follow-up's table.

### Where two files disagree

- Fan4-all is a "success" in its own filing (`lab/experiments/successes/2026-09-20-lxtul-fan4-all.md`), but the
  same filing says its K-curve is flat (+0.0049), and the synthesis note (`.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md`)
  says the arm "does not make the loop earn depth." Both are true: the success is on the exit (width and reader),
  not on the loop.
- The LXTUL-R Step 1 filing calls the relay chain "the largest slot-loop K-curve on the ledger." The synthesis
  note calls the same chain "a tax, not a gain." Step 1b's cost of 0.053 nats at depth 1 is the reason for the
  second reading.
- The 0.87 map: the map-cause filing (`lab/experiments/failures/2026-09-24-lxtul-map-cause.md`) keeps the floor
  finding and refutes the causal link. The READ-BEFORE doc calls the 0.87 "an architecture default" and "an effect."
  Both agree on the floor. They differ on whether the blocks' part is a cause.
- The `nofp` size: the Stage 3 filing reads coda K1-K6 +0.0123 on seed 1. The follow-up's seed-2 run reads +0.0077.
  Both are cited. The filings do not report a mean, and two seeds are not enough to set one.

### Threads left open in this era

- The semantic target (`tul.code_target_source: sonar`, planned, not run). Blocked on the cache check in its prereg.
- The fp01 recipe at 20k (resume from 10k). Depth use growth is one run from 5k to 10k.
- The NextLat exit collapse. Open question: does a frozen or EMA target, or a variance floor, hold per-slot content
  up under the transition term? The EMA target failed on its own, so this is untested in combination.
- The reader, not the objective. Wolfe's next move after E Stage 2 (the parallel head costs the deployed coda 0.012).
- Winner-takes-all responsibility under a stochastic loop. The GK arms learned sigma to the floor. The XM hard-min
  fix (arXiv 2607.27372, App. F.1) is cited and not built.
- The horizon pair on the slot loop: constant 0.0075 nats over the depth-1 twin from 5k to 20k. A 20k K-curve on
  the same pair was not queued (P-6 not run).
- Variable cell count (Wolfe's reminder) is not in this range. The design note is
  `.agents/notes/proposed/architecture/2026-10-05-lxtul-variable-cell-count.md` (dated later).
- Out of range but relevant: `docs/slot-cells-distinct-vs-blurred.md` is dated 2026-09-27. I read only its header
  and did not use its numbers. It is a later table of the same cell family.

---

## TUL history, 2026-09-26 to 2026-10-05

### Era summary

- The strict slot model (`tul_slot_spandec_strict_e4probe_fp01`) trailed the plain model by 0.265 nats at 5k. The gap grew to 0.333 at 10k. The cross-span channel was the cause suspected first.
- Widening the channel by cells, or adding a direct read of the previous span, did not close the gap. Distinct content per cell did (write-all fan, +0.047 nats).
- LX (K=4 enumerated code rollouts) was tested for ensemble versus selection. Rollouts acted as an ensemble. Hard credit, soft credit and selector ideas gave no gain. Several arms detonated.
- Fan variants (WTA, factor fan, routers, latent-selected loop) mostly lost on CE. Their K1-K6 depth numbers were small. The latent router's teacher picked at chance. The 09-30 proposal traced this to detached cells, which nothing pulled toward the target.
- From 2026-10-02 the latent-selected loop with a rank-only head and a per-pass cell RMSNorm (`slot_cell_pass_norm: rms`) gave a loop that earned depth on two seeds (K1-K6 +0.0215 at 5k on two seeds, +0.0236 at 10k).
- That recipe was locked on 2026-10-04 as `morph/configs/lxtul.yaml`. The 10-05 probes then found that the loop's first pass turns the slot carrier into one shared vector. The cell-content axis closed at 5k. The remaining gap looked like exact copying of tokens 33 to 256 back.

### Entries

#### 2026-09-26: KV-cached generation for the strict slot loop (`tul_generate_cached.py`, `generate_tul_cached`, `generate_tul_graphed`)

- Idea: Decoder that reuses K/V caches per span and per slot, and a CUDA-graph version with fixed-capacity buffers. Same tokens as the eager `generate_tul`. Refuses any mechanism outside its set, including `fan_k > 0` and `slot_cells > 1`.
- Why: The eager generator recomputes the whole row at each token. Decode cost had never been measured. The baseline ran at 11.4 tok/s (256 tokens) on the strict baseline at 10k.
- Result: On the strict baseline at 10k, 256 new tokens: eager 16.6, cached 89.0, graphed 280 tok/s. At 512 tokens: 9.3, 71.6, 235 tok/s. Token parity on 6 of 6 prompts at 256 greedy tokens. Source: `.agents/notes/implemented/feature/2026-09-26-tul-kv-cached-generation.md`.
- Verdict: Kept (implemented). The attention kernel was not the bottleneck. The step was launch-bound, so CUDA graphs came first.
- Led to: The fast path refuses the fan. The LXTUL decode speed stayed unmeasured (`.agents/notes/implemented/architecture/2026-10-04-lxtul-primary-candidate.md`).

#### 2026-09-26: Wider slot channel (`tul.prefix_k: 4`, `tul_slot_spandec_strict_e4probe_fp01_pk4`)

- Idea: Four prefix cells per slot instead of two. All four are projections of one exit state `z` (`W_prefix[k] @ z`). The loop state is not wider. Only the coda reads more keys and values.
- Why: Test whether the channel is too narrow for the coda to read.
- Result: Gap to plain at 5k +0.2729 (strict baseline +0.2650). That is 0.0074 worse. Coda K1-K6 +0.0112 (baseline +0.0125). Zero-cell worth 0.197 to 0.210. Source: `lab/experiments/successes/2026-09-26-slot-channel-arms-ab.md` (filed under successes, but the verdict is a failure for this arm).
- Verdict: Failed. Four projections of one state do not add content.
- Led to: The fan (write-all, distinct cells). The same filing tests the next two arms.

#### 2026-09-26: Previous-span direct read (`tul.tg_coda_token_reach: 1`, `tul_slot_spandec_strict_e4probe_fp01_reach1`)

- Idea: A coda token of span s also reads the tokens of span s-1 directly. The prelude stays same-span only.
- Why: Near context might be cheaper to read directly than through the one slot vector.
- Result: Gap +0.2501 (0.0148 better than the strict baseline). Coda K1-K6 +0.0085. Zero-cell worth fell from 0.197 to 0.082. Source: `lab/experiments/successes/2026-09-26-slot-channel-arms-ab.md`.
- Verdict: Failed. The direct read replaced the slot channel. It closed only 0.015 of the gap.
- Led to: Plan C (expanded hyper-connections) and the fan branch. Both are in the note `.agents/notes/proposed/architecture/2026-09-26-slot-channel-width-and-reach.md`.

#### 2026-09-26: Strict baseline gap widens at 10k (`tul_slot_spandec_strict_e4probe_fp01`, 10k run)

- Idea: Not a new design. The strict baseline resumed to 10k steps, paired against plain at 10k.
- Why: Wolfe asked whether the gap closes with training ("lets run it").
- Result: Gap +0.3326 at 10k, up from +0.265 at 5k. Plain improved 0.339 nats from 5k to 10k. The baseline improved 0.272. Plain's own loop K1-K6 went +0.0528 to +0.0612. Source: `lab/experiments/failures/2026-09-26-lxtul-fp01-vs-plain-10k.md`.
- Verdict: Failed on both predictions. The gap grew. The filing says the widening is five times the measured seed spread (0.0137).
- Led to: The channel width and reach arms (above), built to test whether the cross-span channel is the bottleneck.

#### 2026-09-26: Write-all fan with fixed-point term (`tul_slot_spandec_strict_fan4_all_fp01`, `tul.fan_mix: all`, `tul.fan_k: 4`, `model.core_fixed_point_lambda: 0.1`)

- Idea: Four cells per slot, each pooled from the span by its own learned query, looped separately and written 1:1 to its own prefix cell. Not one factor from the baseline. It also drops the rollout code and parallel head, which both refuse the fan.
- Why: The one arm that moved the gap in the channel tests had distinct content per cell, not more cells.
- Result: Gap +0.2181 at 5k, 0.0472 better than the strict baseline on the same tokens (3.4 times the seed spread). Coda K1-K6 +0.0057. Source: `lab/experiments/successes/2026-09-26-slot-channel-arms-ab.md`.
- Verdict: Success on the gap. Its depth earning was small.
- Led to: LX-Fan (the code rollouts on the fan). The 10k resume was queued (`lab/experiments/planned/2026-09-26-fan4-all-fp01-10k.md`) but no filing was found in this range.

#### 2026-09-26: Rollout selection ceiling (analysis, no config)

- Idea: Measure how much any selector over the LX rollouts could buy. Oracle over the K=4 rollouts, against the Bayes read.
- Why: LX trains a per-span mixture over four fixed codes. Does selection matter?
- Result: Bayes minus oracle +0.0413 at 5k and +0.0418 at 10k. The best fixed rollout is 0.02 to 0.036 nats behind Bayes. Credit KL from uniform is about 0.45 nats. Source: `lab/experiments/failures/2026-09-26-lx-selection-ceiling.md`.
- Verdict: Failed its own reading rule (KL 0.05 to 0.20 predicted). The filing says the reading rule was the wrong test. The useful result is that the spread is small. A selector has at most about 0.04 nats to buy.
- Led to: Carry stage 0 (next entry) and the credit arms.

#### 2026-09-26: Carrying the rollout posterior across spans (LX-Carry, stage 0, analysis)

- Idea: Carry the rollout posterior from span s into span s+1 as a prior.
- Why: If rollouts had row identity, carrying would sharpen the read.
- Result: K1-K6 change 0.0009 under carry (requirement was 0.05). Offset 0 got worse at eps 0.5. Source: `lab/experiments/failures/2026-09-26-lx-carry-stage0.md`.
- Verdict: Failed. "The four LX rollouts carry almost no row-level identity." The filing reads the four rollouts as an ensemble, not as span-level hypotheses.
- Led to: Credit arms (committing credit) and the LX-Concept gate (planned, not run).

#### 2026-09-26: Committing credit over the rollouts (`tul.code_enum_credit: hard`, `tul.code_enum_hard_eps`, default 0.05)

- Idea: Multiple-choice-learning credit. Per span, the rollout with the lowest CE gets weight 1 minus eps. The others share eps. Training only. Eval and deploy read the usual mixture.
- Why: Under soft credit the rollouts start equal and nothing breaks the symmetry. Wolfe asked for three placements of the credit.
- Result: Filed 2026-09-27 11:19. Span-coherent part of the hard-credit arm +0.0048 at depth 6 (predicted at least 0.05). Its coda K1-K6 +0.0100 (strict baseline +0.0125). The soft-credit WTA arm that ran alongside it reached K1-K6 +0.0162 at seed 1. One soft fan arm detonated at step 3311, and the 6-cell fan detonated at 1545. Source: `lab/experiments/failures/2026-09-26-lx-credit-arms.md`. (The filing does not state each arm's config name in the summary table. Read the filing.)
- Verdict: Failed on the headline (P-c1, missed narrowly). The filing says the hard credit makes the rollouts differ per span, but the read does not pay for it. Two of four arms detonated (both soft-credit fans).
- Led to: The WTA fan with per-rollout winners (next entry), the soft-fan retry, and the cost reading of the one-code control (next entry on the cost frontier).

#### 2026-09-26: LX-Fan with winner-take-all (`tul_slot_spandec_strict_lxfan4_wta_fp01`, `tul.code_enum_k: 4`, `tul.fan_k: 4`, `tul.fan_mix: all`, `tul.fan_all_wta_lambda: 1.0`, `tul.fan_all_wta_winner: per_rollout`)

- Idea: LX's code rollouts re-added to each fan cell at every pass. Each rollout picks its own winning cell per span through a coda-graded WTA. The refusal that stopped the code on the fan was lifted for `fan_mix: all` at WTA 0 only. Proposal in `.agents/notes/proposed/architecture/2026-09-26-lx-fan.md`.
- Why: Combine the two mechanisms that each held a result (LX code, write-all fan). Neither had been run together.
- Result, seed 1, 5k: K1-K6 +0.0162 [+0.0153, +0.0172], gap to plain +0.2193. Source: `lab/experiments/successes/2026-09-27-lxfan-wta-seed2.md` (seed 1 numbers are quoted there from the 2026-09-26 credit arms filing).
- Result, seed 2, 5k: K1-K6 +0.0093 [+0.0088, +0.0100], gap +0.2375. Seed 2 is 0.0178 worse than seed 1 on the same rows, which is above the 0.0137 seed spread. The pass-0 write rose from 2.0 at step 1000 to 23.4 at step 5000. Source: same file.
- Verdict: Depth earning held at both seeds, and the lower bound beat the write-all fan without codes (+0.0057). The gap prediction missed. Wolfe's direction on 5k CE: "5k CE is not a verdict" (from the seed 2 filing's own rule).
- Led to: The epivol sweep (diversity term on the fan), the per-rollout winner speed fix ("map" winner), and the latent-WTA proposal.

#### 2026-09-26: Soft-credit fan retry (`tul_slot_spandec_strict_lxfan4_fp01`, `tul_slot_spandec_strict_lxfan6_fp01`)

- Idea: Soft-credit LX-Fan (no WTA) with 4 and 6 cells. Retried after both detonated at seed 1.
- Why: Test whether the detonation was a one-off draw. Runs decorrelate within 11 steps, so a rerun at the same seed is a new draw.
- Result, filed 2026-09-27 16:18: the 4-cell retry survived. Its K1-K6 +0.0161, gap to the write-all fan +0.0317 (outside the seed band). The 6-cell retry detonated. A WTA fan at a second seed grew its pass-0 write almost linearly. Source: `lab/experiments/failures/2026-09-27-lx-soft-fan-retry.md`.
- Verdict: Failed on the headline. The soft 4-cell fan is unstable at some rate (1 of 2 draws). Filing's note: "The codes on the fan buy depth, not CE."
- Led to: The WTA seed 2 check (above) and the decision to read the depth earning from the codes.

#### 2026-09-27: Epivol lambda sweep (`tul.fan_repel_mode: epivol`, `tul.fan_repel_lambda`, 0.01 / 0.05 / 0.11, 3000 steps)

- Idea: The fan repulsion term in its epivol form (both volume and epicov). Sweep its lambda with WTA on.
- Why: Wolfe asked for a small sweep ("0.01 0.05 0.11"). The prediction was that the depth use would be flat in lambda.
- Result: K1-K6 +0.0301 at 0.01 (seed 1), +0.0099 at 0.05, +0.0167 at 0.11. The band is 0.022 wide. The prediction was 0.006. The seed 1 values span 0.020. Source: `lab/experiments/failures/2026-09-27-epivol-lambda-sweep.md`.
- Verdict: Failed the flatness prediction. The filing says 0.01 and 0.05 and 0.11 are not separable beyond seed noise at 3000 steps.
- Led to: The grid 2 sweep and seed check (next two entries).

#### 2026-09-28: Epivol grid 2 and seed bands (`tul.fan_repel_mode: epivol`, `tul.fan_repel_lambda`, planned, partly run)

- Idea: Lambda 0, 0.0001, 0.001, 0.005, 0.015, plus the seed checks. Wolfe wanted a second seed for the 0.01 point, and the grid widened.
- Why: The 3000-step point at 0.01 read +0.0301 on seed 1. One more seed was needed before reading it.
- Result: Lambda 0.01 at seed 2 read K1-K6 +0.0026 against seed 1's +0.0301. The queue was paused by Wolfe after that point. Source: `lab/experiments/planned/2026-09-28-epivol-lambda-grid2.md` (planned file, with an amendment) and `lab/experiments/planned/2026-09-28-epivol-seed-bands.md`.
- Verdict: Not finished. The 0.01 peak did not survive a second seed. The seed spread is larger than the sweep's effect.
- Led to: No further epivol sweep was filed in this range.

#### 2026-09-29: One-rollout code policy (`tul.code_policy_k`, `tul.code_policy_lambda`, `tul_slot_spandec_strict_e1probe_fp01_policy4`)

- Idea: One sampled code per slot, trained with REINFORCE, instead of four enumerated rollouts. A control with K=1 and no policy was run beside it.
- Why: Wolfe said "We need to test both a and b." The question was whether one code is enough.
- Result: Control (one code, no policy) K1-K6 +0.0047, gap +0.2830, 11574 tok/s. The policy arm K1-K6 +0.0038, gap +0.2777, 10919 tok/s. Plain ran at 9988 tok/s. Policy minus control: -0.0050 [-0.0075, -0.0024]. Source: `lab/experiments/failures/2026-09-29-code-policy-one-rollout.md`.
- Verdict: Failed. The policy did not beat the control. The one-code control was 1.16 times plain in wall clock. That is the first cost-frontier reading.
- Led to: The cost frontier. The one-code control is the cheapest slot arm measured in this range.

#### 2026-09-29: Head-graded WTA versus ungraded fan (`tul_slot_spandec_strict_fan4_all_fp01_nowta`, `tul_slot_spandec_strict_fan4_all_fp01_hwta`)

- Idea: Test whether the write-all fan keeps its quality without the coda-graded WTA passes. The "nowta" arm is the ungraded fan. The "hwta" arm grades the winner through the head.
- Why: Wolfe's "explore everything at once and let the coda learn how to pick." Coda-graded WTA cost about 0.78 times plain speed.
- Result, 5k seed 1: Ungraded fan K1-K6 +0.0042, gap +0.2543, speed 0.98 times plain, val loss 4.4658. Head-graded K1-K6 +0.0045, gap +0.2670, 0.70 times plain. Coda-graded (reference) K1-K6 +0.0057, gap +0.218, 0.78 times plain. Source: `lab/experiments/failures/2026-09-29-fan-head-graded-wta.md`.
- Verdict: Failed for both. The ungraded fan became the cheap base for the later LX family (the table in `lab/experiments/failures/2026-09-30-fan-opf-and-routers.md` uses it).
- Led to: The factor fan and routers (30 Sep), and the latent-selected loop.

#### 2026-09-29: Map winner for the WTA fan (`tul.fan_all_wta_winner: map`, `tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin`)

- Idea: One posterior winner shared across all K rollouts, found with a B0-row batch instead of K times the rows. Needed because the per-rollout winner search cost about 16 of 24 coda-units per step. The "mapgrad" arm (MAP-only gradient) was in the queue but was never run.
- Why: Speed. Wolfe said "This is still quite slow but worth testing."
- Result, 3000 steps, three seeds for the control band: mapwin K1-K6 +0.0096 [+0.0087, +0.0105] against a band of +0.0072 to +0.0150. Gap to plain +0.4553 against a band of +0.4497 to +0.4627. Pass-1 stream rank 2.87. Source: `lab/experiments/failures/2026-09-29-wta-map-speed-quality.md`.
- Verdict: Quality-neutral inside the three-seed band. Note: the gaps in this file are at 3000 steps, so they are not comparable with the 5k gaps elsewhere. The filing says the mapgrad arm is held. Wolfe paused the queue after these runs.
- Led to: The latent-selected loop, which uses the map winner as its base.

#### 2026-09-29: Latent WTA (proposal only)

- Idea: Pick each slot's winner by distance to a latent target, instead of the coda's CE. Config `tul_slot_spandec_strict_lxfan4_wta_fp01_latwta` exists in the tree.
- Why: The coda CE credit was not separating the cells.
- Result: No filing found in this range that reads it. The proposal is `.agents/notes/proposed/architecture/2026-09-29-latent-wta.md`.
- Verdict: No filing in range reads it. The next work went to the latent-selected loop (2026-09-30).
- Led to: Latent-selected loop and the router arms.

#### 2026-09-29: Per-depth persistent state (planned only)

- Idea: Each core layer keeps one persistent state per loop depth. Layer l on pass t+1 reads its own state from pass t.
- Why: Wolfe said "Per depth state may be different enough though." The plain looped core might use it.
- Result: Planned in `lab/experiments/planned/2026-09-29-per-depth-state.md`. Not filed in this range.
- Verdict: Not run.
- Led to: Nothing in this range. The idea is also parked in `.agents/notes/proposed/architecture/2026-09-29-per-depth-state-and-resonant-depth.md`.

#### 2026-09-30: Factor fan and learned routers (`tul_slot_spandec_strict_fan4_all_fp01_opf`, `..._opf10`, reader-trained router, latent-teacher router)

- Idea: Factor fan: each cell trained to predict an orthogonal factor of the next span's latent (weight 1 and 10). Reader-trained router: a learned router picks one cell per slot. Latent-teacher router: a router trained toward the next span's EMA latent.
- Why: Wolfe: "every exploration should sit in the context window for late use." Factor fan was expected to make the cells distinct.
- Result, 5k seed 1: Factor fan (weight 1) K1-K6 +0.0073, gap +0.262, latent R^2 0.026. Factor fan (weight 10) K1-K6 +0.0047, gap +0.3845, R^2 0.042 (bar 0.1). Reader-trained router K1-K6 +0.0068, gap +0.2649. Latent-teacher router K1-K6 +0.0063, gap +0.2677, teacher-router agreement 0.29. Source: `lab/experiments/failures/2026-09-30-fan-opf-and-routers.md` and `lab/experiments/failures/2026-09-30-factor-fan-10x-latent.md`.
- Verdict: Failed. The latent task did not train at 10x weight. The filing says "the factor task itself, not its weight, is the dead end."
- Led to: Latent-selected loop (the latent teacher moves into the loop).

#### 2026-09-30: Latent-selected loop, teacher-followed (`tul.fan_loop_select: joint | detached`, `tul.fan_route: latent`)

- Idea: Selection inside the slot loop after every pass. The teacher (next span's EMA-prelude latent) picks a cell and the slot's cells reset to it. The latent head reads the cells with gradient (joint) or detached (detached).
- Why: The earlier latent router picked the coda's best cell at chance. The proposal's diagnosis: its map read the cells detached, so nothing pulled any cell toward the target. Proposal: `.agents/notes/proposed/architecture/2026-09-30-latent-selected-loop.md`.
- Result, 5k seed 1: Joint K1-K6 -0.2294, gap +0.7146, ledger CE 4.7953. Detached K1-K6 -0.1701, gap +0.6663, ledger CE 4.7474. Ungraded fan reference: K1-K6 +0.0042, CE 4.3343. Source: `lab/experiments/failures/2026-09-30-latent-selected-loop.md`.
- Verdict: Failed hard. The teacher path (following the teacher's pick) worked. The router path did not. The gap between the two was the finding.
- Led to: Router-followed training (next entry).

### Entries (continued, 2026-10-01 to 2026-10-05)

#### 2026-10-01: Latent-selected loop, router-followed (`tul.fan_lsel_train_follow: router`, joint and detached)

- Idea: Train forward follows the router, not the teacher. Train and deploy forwards become the same.
- Why: Wolfe: "Yes fix it and run it." The 09-30 result trained on hindsight picks that deploy never sees.
- Result, 5k seed 1: Joint K1-K6 +0.0067, gap +0.4102, ledger CE 4.4908. Detached K1-K6 +0.0095, gap +0.4091, ledger CE 4.4894. Ungraded fan CE 4.334. Source: `lab/experiments/failures/2026-10-01-latent-selected-loop-router-followed.md`.
- Verdict: Failed on the pass rule. The loop earns depth (detached K1-K6 is 2.3 times the ungraded fan), but CE trails by 0.155 nats.
- Led to: Latent weight 1 and full read tests (next two entries).

#### 2026-10-01: Detached latent at weight 1 (`tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1`)

- Idea: The same detached latent head, with the latent loss weight cut from 10 to 1.
- Why: Test whether the CE tax comes from the latent loss.
- Result, 5k seed 1: K1-K6 +0.0333 [+0.0318, +0.0347], ledger CE 4.447, gap +0.366, zero-cell worth +0.102, latent R^2 -0.038. Ledger CE improved by 0.042 against weight 10. Source: `lab/experiments/failures/2026-10-01-latent-selected-loop-weight1-and-read-all.md`.
- Verdict: Failed on CE (0.113 behind the ungraded fan, the bar was 0.03). Its K1-K6 is the highest measured in this range. The later 10-03 note says its cells grow about 4 times along one shared bias at pass 2.
- Led to: Rank-only head (below). This is the "detached weight 1" reference in later notes.

#### 2026-10-01: Full cell read at the coda (`tul.fan_lsel_read`, set to the full read; value in the filing)

- Idea: The coda reads all four cells instead of the final winner alone.
- Why: The router-followed arm trails the ungraded fan on CE. The winner-only read was the suspect, since the ledger priced the cells' worth at about 0.17 nats.
- Result, 5k seed 1: K1-K6 +0.0170 (above the ungraded fan's +0.0042). Ledger CE 0.013 worse than the weight-1 arm. CE 0.168 behind the ungraded fan. Source: same file as above.
- Verdict: Failed on CE. The full read did not close the CE gap. It raised K1-K6.
- Led to: Nothing directly. The winner-only read stayed in the lead recipe.

#### 2026-10-02: Rank-only latent head (`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank`, `tul.fan_lsel_head_input: detached`)

- Idea: The latent head ranks the cells but does not pull them. It reads detached cells and sends no latent gradient into the loop.
- Why: Wolfe: "I want to try that arm." The pull looked like a tax on CE.
- Result, 5k seed 1: Ledger CE 4.380, gap +0.299, K1-K6 +0.0052. The pull twin (joint, weight 1, `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1`) read CE 4.387 and K1-K6 +0.0096. Source: `lab/experiments/failures/2026-10-02-latent-selected-loop-rank-only-head.md`.
- Verdict: Failed the pass rule (CE was 0.007 better, the bar was 0.0137). The filing's table shows the rank-only head keeps the selection. Its K1-K6 is half the pull twin's.
- Led to: The per-pass cell norm (10-03), which doubled the rank-only arm's depth earning.

#### 2026-10-02: Huginn and HRM loop geometry probes (instruments)

- Idea: Measure how a looped LM (Huginn-0125, 3.5B) moves its state. The result changed how the MORPH cells were read.
- Why: Wolfe: "I feel like magnitude isn't true diversity." Huginn earns depth on web text. Does it grow or rotate its state?
- Result, Huginn, 2026-10-02: The participation ratio triples from pass 1 to pass 4 (215 to 604). The state grows where CE falls fastest. Source: `lab/experiments/successes/2026-10-02-huginn-loop-geometry.md`. HRM-Text, 2026-10-03 (run on the 3070): z_H per cycle, step share along the shared direction 6.5 percent at the trained cycle and 1 percent after. Removing the shared bias costs 0.41 nats. Source: `lab/experiments/failures/2026-10-03-hrm-text-loop-geometry.md`.
- Verdict: Huginn success on its hypothesis, with one prediction missed. HRM-Text failed its rotation prediction (R-4, R-5). The filing says HRM-Text is not a contractive loop. Its steps do not shrink.
- Led to: The per-pass cell RMSNorm test.

#### 2026-10-02: Staged latent pretraining (planned only)

- Idea: Train the slot loop alone on a latent target with no coda and no token CE. Test whether the latent is learnable, and whether passes add to it. Frozen target variants (L2, InfoNCE, 4-cell winner-take-all, final hidden).
- Why: Wolfe: "Right, we may get blurred means. We should try those variants." The latent head had val R^2 at or below 0.01 in all six latent arms.
- Result: Planned in `lab/experiments/planned/2026-10-02-stage1-latent-pretraining.md`. Configs `tul_latent_pre_prelude_l2`, `tul_latent_pre_prelude_nce`, `tul_latent_pre_prelude_wta4`, `tul_latent_pre_base`. No filing in this range.
- Verdict: Not run in this range.
- Led to: Nothing filed by 10-05.

#### 2026-10-02: Common-mode probe on the frozen prelude (instrument)

- Idea: Measure how much of a span's pooled, LayerNormed prelude output is one shared direction.
- Why: Wolfe's puzzle: the latent pull makes the loop earn depth, yet the detached target looks like a common mode.
- Result: 98 percent of the frozen plain prelude output is one shared direction (`.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md`).
- Verdict: Instrument result. Its object was the frozen plain prelude output, not the slot carrier. A later probe (10-05) found the same effect in the slot carrier, but only in the difference between the Hyper-Connection streams. The 10-02 probes read the stream mean and found a small shared part. See the 10-05 correction below.
- Led to: The 10-03 per-pass cell norm pair.

#### 2026-10-03: Per-pass cell RMSNorm with the rank-only head (`tul.slot_cell_pass_norm: rms`, `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm`)

- Idea: One learned RMS gain on every slot cell after every pass, shared by all passes. It removes the magnitude channel that the cells used to grow along a shared direction.
- Why: Wolfe: "magnitude isn't true diversity." Without the norm, the rank-only arm's K1-K6 (+0.0052) lived in one amplitude along the shared direction. Setting that amplitude to its mean erased the depth earning.
- Result, 5k seed 1: K1-K6 +0.0215 [+0.0205, +0.0225], ledger CE 4.353, gap +0.272 [+0.257, +0.290]. Zero-cell worth +0.176. Source: `lab/experiments/failures/2026-10-03-slot-cell-pass-norm-pair.md`.
- Result, seed 2 (`lab/experiments/successes/2026-10-03-rank-only-cell-norm-seed-twin.md`): K1-K6 +0.0172, ledger CE 4.362, gap +0.282. Greedy rep-4 0.474 (seed 1: 0.560). Between-slot spread rose from 0.07 to 2.34.
- Verdict: Success. The pass rule held on both seeds. The filing says the cells' depth earning is directional, and the norm changed the mechanism from magnitude to direction. Wolfe's rule (2026-09-22, per the MORPH memory index) is that cosines are diagnostics, not pass or fail gates. The pass rule here used K1-K6 and CE.
- Led to: 10k run and the locked recipe.

#### 2026-10-03: Per-stream norm with the detached weight-1 arm (same filing, second arm)

- Idea: Apply the same per-pass norm to the detached weight-1 arm.
- Why: Test whether the norm helps the arm with the highest K1-K6.
- Result: K1-K6 fell to +0.0119 (from +0.0333 without the norm). Source: `.agents/notes/implemented/architecture/2026-10-04-lxtul-primary-candidate.md`.
- Verdict: Rejected. The highest K1-K6 without the norm was one amplitude along a shared bias.
- Led to: Rank-only head plus norm became the lead arm.

#### 2026-10-03: Read-side norm (`tul.slot_cell_pass_norm: rms_read`)

- Idea: Fix the scale of the state the coda reads, not the scale of the loop state.
- Why: Test whether the detached arm's pass-2 jump is growth or a read-side artefact. Wolfe: "try #2 out to 3k steps."
- Result, 3000 steps: The read state's norm was 31.0 to 31.5 at every pass. The cells still rotated onto the shared direction. The off-direction part fell from 30.9 to 9.2. Ledger shipped CE 4.611. Source: `lab/experiments/failures/2026-10-03-slot-cell-read-norm.md`.
- Verdict: Failed. The agent's prediction was wrong. "The read norm fixes the read state's scale exactly ... and the cells still go onto one shared direction." The pass-2 event is rotation, not growth. Norms decide only whether it shows as growth (no norm) or rotation (read norm).
- Led to: The pass-by-pass norm on the loop state (above).

#### 2026-10-04: Lead arm at 10k steps (`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm_10k`)

- Idea: The lead arm (rank-only head plus per-pass cell norm) run to 10k steps with the speed recipe (`training.ademamix_fused_fp32: true`, `model.ckpt_grad_iters: 4`).
- Why: Wolfe: "do a 10k training on our current best arm."
- Result: K1-K6 +0.0236 [+0.0225, +0.0246] (5k: +0.0215). Ledger CE 4.0975 (5k: 4.353). Gap to plain at the same step +0.356 [+0.336, +0.379] (plain at 10k). Speed 12402 tok/s. Source: `lab/experiments/successes/2026-10-04-rank-only-cell-norm-10k.md`.
- Verdict: Success on the depth pass rule. The gap to plain still widens with training: +0.272 at 5k, +0.356 at 10k.
- Led to: The lock of this recipe as the primary candidate.

#### 2026-10-04: LXTUL locked as primary candidate (`morph/configs/lxtul.yaml`)

- Idea: The recipe as one config. Strict slot loop, 4-cell write-all fan, epivol diversity on passes 1 and 2 (no WTA), fixed-point term 0.1, latent-selected loop (router-followed, rank-only head), per-pass cell RMSNorm. The coda reads the final winner alone. `base.yaml` was not changed.
- Why: Give later work one base. The goal: a recipe whose loop earns depth reliably at a speed suited to iteration.
- Result: Evidence rows are the 5k seed 1 (K1-K6 +0.0215, CE 4.353, gap +0.272, 9800 tok/s), 5k seed 2 (+0.0172, CE 4.362, gap +0.282), 10k (+0.0236, CE 4.0975, gap +0.356, 12402 tok/s). Ungraded fan reference K1-K6 +0.0042. Source: `.agents/notes/implemented/architecture/2026-10-04-lxtul-primary-candidate.md`.
- Verdict: Implemented. Open problems named in the note: the gap widens with training, and repetition is unchanged (greedy rep-4 overlaps every arm). The fast decode path refuses the fan.
- Led to: LXTUL eight cells, the exact-recall probe, the carrier-constant twin. The note says it was superseded as the loop base on 2026-10-11 (outside this range).

#### 2026-10-04: More cells, eight per slot (`lxtul_fan8`, `tul.fan_k: 8`)

- Idea: Eight cells per slot instead of four. One factor from `lxtul.yaml`.
- Why: Wolfe: "How about we try more cells?" The hypothesis: the cross-span budget grows with cells.
- Result, 5k seed 1: K1-K6 +0.0103, gap +0.2697 (above the +0.262 bar), worth +0.158, share of worth over (worth plus gap) 0.37 (bar 0.42). Ledger CE 4.3470. Tripwire had three recovered one-step outliers. Source: `lab/experiments/failures/2026-10-04-lxtul-eight-cells.md`.
- Verdict: Failed. The filing says the far-bigram bucket still holds 75.7 percent of the gap.
- Led to: The exact-recall probe (next entry).

#### 2026-10-04: Exact-recall gap probe (analysis, success)

- Idea: Find where LXTUL's gap to plain sits by token distance and bigram repeat.
- Why: The gap might be exact copying from far back.
- Result: Bigram repeats 33 to 256 tokens back are 15 percent of tokens and 65 percent of the gap at 10k. The gap falls off at 256 tokens, which is plain's exact sliding-window size (`model.window_size: 256`). Past 256, LXTUL's cells nearly match plain (+0.17). Source: `lab/experiments/successes/2026-10-04-exact-recall-gap-probe.md`.
- Verdict: Success on the question asked. The filing says the window explanation is consistent, not proven. A direct test is a plain model with a shorter window.
- Led to: Pseudo-token carrier (snap) and the trace-free carrier proposal.

#### 2026-10-04: CE test queue and training speed notes (proposed)

- Idea: A numbered queue of 11 CE items (e.g. item 9 is the multi-head span pool, item 10 is the own-span reconstruction decoder). A separate note covers slot-loop training speed.
- Why: LXTUL trails plain by the same gap at the same step. The queue sorts the candidates.
- Result: `.agents/notes/proposed/architecture/2026-10-04-lxtul-ce-test-queue.md` and `.agents/notes/proposed/architecture/2026-10-04-slot-loop-training-speed.md`. The speed note was not read in detail for this history.
- Verdict: Proposed. Items 9 and 10 ran on 10-05. The rest were not filed in this range.
- Led to: The cell-packing panel (10-05).

#### 2026-10-05: Carrier constant twin (`lxtul_hcuni`, `tul.loop_attn_hc`; `lxtul_center`, `tul.loop_attn_center`)

- Idea: Remove the slot loop's carrier constant. Two arms: uniform loop attention residual (`loop_attn_hc`: a plain residual) and centered loop attention (`loop_attn_center: ema`).
- Why: The data-flow probe (see the snap entry) found the loop's first pass turns the slot carrier into one shared vector.
- Result, 5k: Uniform loop attention K1-K6 +0.0171, gap +0.270, ledger CE 4.350, HEALTHY. Centered loop attention detonated at step 470 (2.3e6) and again from 1617 (max 2.7e12). It never trained. Source: `lab/experiments/failures/2026-10-05-carrier-constant-twin.md`.
- Verdict: Failed on both arms. The uniform arm rebuilt the constant by another path (the per-slot share after pass 1 was 1.64 percent, under the 2 percent kill line). The centered arm was a broken model. Its trigger is inferred, not logged.
- Led to: The rejection of the constant route (`.agents/notes/rejected/architecture/2026-10-05-slot-loop-carrier-constant.md`).

#### 2026-10-05: Cell packing panel (`lxtul_pool16`, `tul.slot_pool_heads: 16`; `lxtul_recon`, `tul.recon_weight: 1.0`; `lxtul_pool16_recon`)

- Idea: Pack more of the span into each cell. Multi-head span pool (16 heads over the span's prelude states). Own-span reconstruction (a train-only decoder rebuilds the slot's own span from the winner cell). Both.
- Why: Wolfe's "width first." Test whether the cells carry too little span content.
- Result, 5k seed 1: Multi-head pool K1-K6 +0.0221, gap +0.2789, ledger CE 4.3593, one recovered detonation at step 2844. Reconstruction K1-K6 +0.0010, gap +0.3774, CE 4.4580. Both K1-K6 +0.0039, gap +0.3942, CE 4.4749. Source: `lab/experiments/failures/2026-10-05-lxtul-cell-packing-panel.md`.
- Verdict: Failed. No arm closed the gap. Reconstruction killed most of the depth earning.
- Led to: Wider write, and the full-history pair.

#### 2026-10-05: Wider write (`lxtul_fan4x2`, `tul.prefix_per_cell: 2`)

- Idea: Each cell writes into two coda positions instead of one. One factor from `lxtul.yaml`.
- Why: Wolfe asked for width first.
- Result, 5k seed 1: K1-K6 +0.0171, gap +0.2835, worth +0.158, CE 4.3603, tok/s 11156. Source: `lab/experiments/failures/2026-10-05-lxtul-wider-write.md`.
- Verdict: Failed on the gap. The filing says a second, rotated view of the same winner cell adds no CE.
- Led to: The snap pseudo-token carrier.

#### 2026-10-05: Full token history (`lxtul_hist`, `lxtul_hist_embed`, `tul.coda_token_input`)

- Idea: Two arms. Open coda (`lxtul_hist`): the prelude sees the row's full causal token history, not only its own span. Embeddings coda (`lxtul_hist_embed`): the coda token carrier reads the raw embeddings, not the prelude output.
- Why: Wolfe: "we need to test the prelude and loop seeing full token history." The strict geometry starves the prelude of everything but its own span.
- Result, 5k seed 1: Open coda K1-K6 +0.0140, gap +0.2308, ledger CE 4.3109, worth +0.033. Embeddings coda K1-K6 +0.0134, gap +0.4046, CE 4.4850, worth +0.173. Source: `lab/experiments/failures/2026-10-05-lxtul-full-history-pair.md`.
- Verdict: Failed on both pass rules. Full history closed only 0.04 of the gap. The cells became almost worthless, but the loop still earned depth. The filing says depth earning is not the cells' cross-span content.
- Led to: The snap pair and the prelude full-history proposal (`.agents/notes/proposed/architecture/2026-10-05-prelude-full-token-history.md`).

#### 2026-10-05: Trace-free pseudo-token carrier, "snap" (`lxtul_snap`, `lxtul_snap_entry`)

- Idea: Write each loser position's coda input from span-restricted pseudo tokens (a probability-weighted embedding mix over the span's tokens). Snap takes its pseudo tokens from the loop's winner. The seed-sourced arm (`lxtul_snap_entry`) takes them from the slot's entry state, with no loop pass.
- Why: Wolfe's next line: A2 (pseudo-token write), then B (pack more into cells), then A3 and A1. The copy gap needed a carrier that holds exact tokens.
- Result, 5k seed 1 for snap: K1-K6 +0.0119, gap +0.2426, ledger CE 4.3229, tok/s 12228, HEALTHY (max 967). Seed-sourced arm: K1-K6 +0.0106, gap +0.2571, CE 4.3378, detonated at 811 (recovered). Source: `lab/experiments/successes/2026-10-05-lxtul-snap-pair.md`.
- Verdict: Success under the pass rule. The filing says snap "does NOT work the way it was designed." The data-flow probe found the pseudo-token gate never opened (g 0.001 to 0.009). The gain came from extra coda positions used as free keys. The far-bigram gap moved 0.04, not the 0.10 predicted.
- Led to: The data-flow probe (next entry) and the trace-free carrier proposal.

#### 2026-10-05: Data-flow probe on the slot carrier (analysis, success)

- Idea: Measure the energy of the cell carrier at each stage: entry, after pass 1, exit, for snap and LXTUL on the same rows.
- Why: The snap result did not match its design. Where does the information go?
- Result: At entry 54 to 57 percent of the carrier's energy differs between slots. After pass 1 it is 0.2 to 0.4 percent. At exit 1.7 to 1.8 percent. The RMS stays near 1.05. The shared vector sits in the differences between the 4 Hyper-Connection streams, so it cancels in the stream mean. Source: `lab/experiments/successes/2026-10-05-snap-dataflow-probe.md`.
- Verdict: Success under the prereg's rule. The first pass replaces the slot state with one vector shared by every slot of every row. Every earlier probe read the stream mean and saw a small shared direction. That is the correction of the 10-02 common-mode reading (see the corrections list).
- Led to: The carrier constant twin, and the rejected note on the constant route.

#### 2026-10-05: Variable cell count (proposed only)

- Idea: Poisson-sample the number of cells per slot at train. Test at inference with a different count. The idea: train at 4, infer at 128.
- Why: Wolfe, verbatim: "For lxtul cell count, we should try applying poisson sampling to the number of cells to see if it generalizes. This whole thing needs to be tested for its behaviors. For instance can I train at 4 and inference at 128 fine? a variable cell count at inference would be huge."
- Result: Not built. The note says that under the latent-selected loop the cell count is the search breadth per pass, not a channel. The coda reads only the final winner. The losers' positions are exactly zero (`_fan_route_cells`).
- Verdict: Proposed. Nothing filed in this range.
- Led to: The cell-count reminder kept in the tree (`.agents/notes/proposed/architecture/2026-10-05-lxtul-variable-cell-count.md`).

### Corrections and retractions in this era

- Selection ceiling reading rule (2026-09-26): the KL reading rule (0.05 to 0.20) was the wrong test. The KL of 0.447 is a real measurement. The predicted window was wrong. The filing calls it "a failure of the prediction."
- Latent router (2026-09-30): the teacher picked the coda's best cell at chance. The 09-30 proposal says the cause was wiring. The map read the cells detached, so no latent term pulled the cells. The proposal calls the idea of a latent teacher sound and the wiring wrong.
- LXTUL primary candidate (2026-10-04, corrected 2026-10-05 01:02 CDT): the note first said every cell was written to the coda. The note was corrected. The coda reads the final winner alone under the latent-selected loop.
- Common-mode probe (2026-10-02 against 2026-10-05): the 10-02 note says the frozen prelude output is 98 percent one shared direction. The 10-05 data-flow probe says the earlier stream-mean readings missed the slot-carrier constant, which lives in the stream differences. The 10-02 note's claim that every latent arm had such a target was an orchestrator claim, and the note records it as one.
- Read-side norm (2026-10-03): the agent's guess was wrong. The read norm fixes the scale of the read state. The cells still rotate. Filing: "My guess was wrong."
- Snap pass (2026-10-05): the pass rule holds, but the mechanism in the design is not what runs. The gate never opened. The arm is a pass on the gap and a mechanism failure.
- Exact-recall probe (2026-10-04): prediction R-4 failed in an informative way. The gap falls off at 256 tokens, which is the window size. The filing says the window explanation is consistent with the data, not proven.

### Planned and never filed in this range

- LX-Concept gate (`lab/experiments/planned/2026-09-26-lx-concept-precheck.md`). Question: can fp01's slot state predict the next span's concept label above the marginal. Step 1(a) ran on CPU. No GPU filing.
- Plan C, expanded hyper-connections in the slot loop (`lab/experiments/planned/2026-09-26-plan-c-xhc-arms.md`, `.agents/notes/proposed/architecture/2026-09-26-plan-c-xhc-slot-loop.md`). Based on xHC (arXiv 2607.14530). Not run.
- LX amplitude pin and matched K-curve (`lab/experiments/planned/2026-09-26-lx-amplitude-pin.md`, `...-matched-k-curve.md`). Not run.
- Strict write-all fan at 10k, paired against plain 10k (`lab/experiments/planned/2026-09-26-fan4-all-fp01-10k.md`). Queued, not filed in range.
- Epivol seed bands at 3000 steps (`lab/experiments/planned/2026-09-28-epivol-seed-bands.md`). Partly superseded by the grid 2 run.
- Per-depth persistent state (`lab/experiments/planned/2026-09-29-per-depth-state.md`). Not run.
- Staged latent pretraining (`lab/experiments/planned/2026-10-02-stage1-latent-pretraining.md`). Not run.
- Latent WTA (`.agents/notes/proposed/architecture/2026-09-29-latent-wta.md`). No filing in range.
- Trace-free pseudo-token carrier, full set of arms (`.agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md`). Only the snap pair was run.
- Exploration ledger (`.agents/notes/proposed/testing/2026-09-30-exploration-ledger.md`). A proposed method for comparing fan arms with a ledger of per-token readings.

### Threads left open in this era

- Gap to plain widens with training: +0.272 at 5k, +0.356 at 10k on the locked recipe. Repetition is unchanged (greedy rep-4 overlaps every arm).
- Exact copying past 256 tokens is the main source of the gap. The window test (a plain model with a shorter window, or LXTUL with a raw window) was named but not run.
- The first loop pass makes the slot carrier one shared vector. The centered-attention fix detonated. Whether another route removes the constant without a detonation is open.
- Variable cell count (train at 4, infer at more) is not built. Wolfe's question is still open.
- Epivol lambda: the 0.01 peak did not hold at seed 2. The lambda range is unresolved.
- Hard credit and soft credit: the credit question is open at 5k. The one-code control reached 1.16 times plain speed, so the cost frontier is still open too.
- Decode speed of the LXTUL recipe is not measured, because the fast decode path refuses the fan.
- Staged latent pretraining (frozen targets, with and without the coda) was planned and never filed.
- Plan C (expanded hyper-connections in the slot loop) was planned and never filed.
- Per-depth persistent state was planned and never filed.

---

## TUL history, 2026-10-06 to 2026-10-14 (MORPH)

### Era summary

- The pointer head (`tul.pointer_heads`) closed most of the LXTUL gap to plain at 5k. It moved the loop from 0.272 nats behind plain to 0.150 ahead of a plain model with no pointer. The copy channel was the bulk of the gap, not the slot loop.
- Repetition under sampling became the open cost of the pointer. Coverage made it worse. DITTO halved it, at +0.071 nats of clean val. Wolfe declared the pointer plus DITTO the winner on 2026-10-07.
- The credit panel (2026-10-10) asked which credit paths carry the loop. Most single-path cuts landed inside seed noise. Two were large: a prelude that does not train through the coda (CE +0.29) and LXTUL without the span decoder (CE +0.07).
- The span decoder's target sets the cell diversity. A mean, per-cell or best-cell target pulls the cells together (cos +0.6 to +0.8) or costs CE. A set reader, which cross-attends to the cells as the coda does, keeps them distinct (cos -0.28). Wolfe adopted this as the ungraded fan recipe on 2026-10-11.
- Variable cell count v1 (2026-10-13) answered the downward question. Trained at 3 or 4 cells, the model reads well at fewer cells. No arm gained above 4 cells at inference.
- From 2026-10-13 the speed work changed the design: a one-block span decoder and cell mean 2 passed at 3 seeds. Hinge every 4 and a half-slot decoder did not. Rounding-only keys stayed off pending Wolfe.
- Measurement tightened. Same-code 5k runs spread about 0.006 in K1-K6 and 0.01 to 0.02 in CE@6. Several earlier reads were revised (see the retraction list at the end).

### Entries

#### 2026-10-06: Output-only pointer head (`tul.pointer_heads: 4`, `lxtul_pointer`)

- Idea: four output-only pointer heads on the strict LXTUL loop. Each head attends from a token's final hidden state over earlier tokens and reads the attention as a distribution over the next token. A null key lets a head abstain. Nothing enters a hidden state, so the loop stays the only cross-span channel inside the model. Ported from the Parcae testbed, where the head closed most of the gap.
- Why: LXTUL at 5k sat 0.272 to 0.282 nats behind the plain ruler. The filing puts about 0.23 of a 0.30 gap on far-bigram repeats (recall probe). The winner note puts 86 % of the gap on far-bigram tokens. Wolfe, 2026-10-06: "Its time to port it to MORPH."
- Result: CE@6 3.9280. Gap to plain ruler -0.150 [-0.169, -0.132] (plain ruler CE@6 4.0806). K1-K6 +0.0155 [+0.0147, +0.0164]. Head off: CE@6 4.4097, which is 0.48 worse, and 0.05 worse than LXTUL trained with no head. Write worth +0.147 (zeroed) and +0.149 (shuffled). Trainer 11.8k tok/s. Source: `lab/experiments/mixed/2026-10-06-lxtul-pointer.md`.
- Verdict: kept. Six of seven predictions held. Prediction 7 (no raise, no tripwire) failed on the cell-key arm, which detonated once. The no-keys run read AMBIGUOUS (max 8.76e3, under the 1e4 rule). The "beats plain" claim has a caveat. The ruler is plain with no pointer. MORPH never ran plain plus pointer, so the result means "LXTUL plus a copy channel beats plain with none." Parcae plain plus pointer led strict plus pointer by 0.045, so the true ruler gap is unknown.
- Led to: pointer cell keys (rejected), coverage (failed), DITTO (mixed), the winner declaration.

#### 2026-10-06: Pointer cell keys (`tul.pointer_cell_key: true`, `lxtul_pointer_cellkey`)

- Idea: keys in earlier spans also read that span's loop cells, so the pointer can copy from cell states, not only output tokens.
- Why: the Parcae run had shown a small gain from the same change.
- Result: CE@6 3.9208 against 3.9280 without keys. Paired depth-6 difference -0.0073 [-0.0091, -0.0053]. K1-K6 +0.0098 against +0.0155. Detonation: pre-clip prelude grad norm jumped to 523 to 759 in bursts at steps 2297 to 2310, then recovered. Source: `lab/experiments/mixed/2026-10-06-lxtul-pointer.md`.
- Verdict: rejected. The CE win sits inside the 0.009 seed spread. Cell keys moved pass 1 more than pass 6, so K1-K6 fell. The keyed run is the one that detonated.
- Led to: the no-keys pointer became the arm to build on.

#### 2026-10-07: Coverage repetition phase (`tul.pointer_coverage_lambda: 1.0`, `lxtul_pointer_cov` against control `lxtul_pointer_ft`)

- Idea: See et al. (arXiv 1704.04368) coverage loss as a separate 1000-step final phase from the 5k pointer checkpoint. Per head, the coverage of key j is the attention earlier queries paid it. The loss adds lambda times the sum of min(attention, coverage). A learned per-head gain (zero init) enters the score. The score reads coverage-free attention, a forced deviation from the paper.
- Why: the pointer model repeats under sampling. At T 0.7 and top-k 40, seq_rep_4 was 0.307 against plain 0.215 and real text 0.020. Greedy looping was at plain's level for every 5k model.
- Result: the coverage loss fell from 0.304 to 0.030 (started at 0.52). Trainer val 3.9548 (control) against 3.9725 (coverage). T 0.7 seq_rep_4 0.291 (control) against 0.374 (coverage), 29 % higher. Greedy seq_rep_4 0.909 against 0.925. Source: `lab/experiments/failures/2026-10-07-lxtul-pointer-coverage.md`.
- Verdict: failure. Predictions 1, 2, 5, 6 held. Predictions 3 and 4 failed, and the sign is wrong. The filing's mechanism: coverage counts attended source positions. In LM self-copy the source is the output so far, and a copied cycle reads fresh positions with near-zero coverage. The loss trained the heads toward the newest positions, which is the pattern behind the loops. Coverage stays in the tree, default off, as a measured negative.
- Amendment: `training.init_from` resets the train stream to doc 0, so the 480 sweep rows were contaminated for both phases. The control sweep gap read -0.209 against -0.150 at 5k while clean val got worse. Phase comparisons used only matched sweep rows, and the trainer's clean val for the verdict.
- Led to: DITTO. The filing names DITTO (Xu et al. 2022, arXiv 2206.02369) and sequence-level unlikelihood (arXiv 1908.04319) as the candidates.

#### 2026-10-07: DITTO repetition phase (`tul.ditto_rows: 3`, `tul.ditto_lambda: 0.5`, `lxtul_pointer_ditto`)

- Idea: three of six rows per batch become pseudo-repetition rows (real text up to a random span, then that span repeated). On copy n at least 1 the CE is replaced by -log(1 - |p_n - 0.5 sg(p_{n-1})|) on the pointer-mixed probability. Added at weight 1. Same 1000-step phase from the 5k pointer checkpoint, fresh optimizer, 200-step ramp. Wolfe, 2026-10-07: "the foot gun of over repitition will still bite later ... I think we should just do it."
- Why: coverage failed. DITTO targets the self-reinforcement that makes a phrase repeat once it has repeated.
- Result: T 0.7 seq_rep_4 0.140 against control 0.291 (-52 %) and plain 0.215. Greedy seq_rep_4 0.821 against control 0.909 and plain 0.873. Head worth 0.450 against 0.465. Trainer val 4.0254 against 3.9548 (+0.071). Sweep K1-K6 +0.0294 against +0.0180. One spike train at step 477 (grad 1.6e7), recovered by step 483. `tul/ditto_p_ratio` was still falling at step 1000 (0.73). Source: `lab/experiments/mixed/2026-10-07-lxtul-pointer-ditto.md`.
- Verdict: mixed. It is the first guard that works on the pointer's repetition. Predictions 2, 3, 4 held. Predictions 1, 5, 6 and the tripwire part of 7 failed. The CE cost sits in the body, not in a muted head. The filing names two plausible causes. DITTO rows replace half the real rows. Pseudo rows feed the slot loop many identical spans. Its own proposal: (a) add DITTO rows on top of the real batch, (b) cut copies to a few, (c) a longer phase or a low-rate DITTO in the main run. The junk-subword artifact ("studymbok of the worldmbok") was seen and not counted.
- Led to: the winner declaration, which ships DITTO anyway. See the next entry.

#### 2026-10-07: Winner recipe declared (`lxtul_pointer.yaml` then `lxtul_pointer_ditto.yaml`)

- Idea: two stages. Stage 1 is 5k steps of `lxtul.yaml` plus `tul.pointer_heads: 4`. Stage 2 is the 1000-step DITTO phase (`tul.ditto_rows: 3`, `tul.ditto_lambda: 0.5`) from the stage 1 checkpoint.
- Why: LXTUL alone is 0.272 behind plain, and a copy channel that closes the gap makes sampled text repeat more.
- Result: from the two filings above. Plain CE@6 4.0806. LXTUL CE@6 4.3529, gap +0.272, K1-K6 +0.0215. LXTUL plus pointer CE@6 3.9280, gap -0.150, K1-K6 +0.0155. Plus DITTO, K1-K6 +0.0294 and trainer val +0.071 against its control. Source: `.agents/notes/implemented/architecture/2026-10-07-lxtul-pointer-ditto-winner.md`.
- Verdict: adopted as the loop base, by Wolfe's word on 2026-10-07. The note records DITTO's +0.071 cost as an open item, and it is in the recipe. The same-day filing's proposed DITTO redesign (rows on top of the batch, fewer copies) was not run in this range. The note's "Consequences" section lists owed items: a MORPH plain plus pointer ruler, a pointer seed twin, and a DITTO-from-start or lighter-DITTO variant.
- Led to: the fast recipe (next entry), the credit panel, and, on 2026-10-11, the fan recipe that supersedes the LXTUL base.

#### 2026-10-08: Fast pointer recipe (`morph/configs/lxtul_pointer_fast.yaml`, `lxtul_pointer_fast`)

- Idea: the same pointer recipe with the speed package. It adds the graph-captured step, compiled blocks, fused TG attention, fused HC grad, ternary step cache, no checkpoint recompute, compiled fan twin, and a bf16 lsel pick. The function is meant to be unchanged.
- Why: phase 1 of the speed plan. The eager pointer recipe ran at 11.8k tok/s.
- Result: CE@6 inside the eager range (predicted 3.914 to 3.933). K1-K6 at least +0.0075. Final val within 0.010 of the eager rerun. Phase 1 reports 2.3x speed (26.7k tok/s). Panel references later read CE@6 3.9248 and 3.9229 with K1-K6 +0.0104 and +0.0091 at 29.8k tok/s. Source: `lab/experiments/successes/2026-10-08-lxtul-pointer-fast-5k.md`, `.agents/notes/proposed/process/2026-10-10-phase-2-tracker.md`.
- Verdict: success. The three quality predictions held. The resolution is one run per arm.
- Led to: all credit panel arms in this range run on this base.

#### 2026-10-08: Production rerun noise floor (`lxtul_pointer`, eager, seed 1, rerun)

- Idea: measurement. Two same-code 5k runs of the production recipe, to size the run-to-run spread.
- Why: no single-run comparison of the winner could be read without it.
- Result: the winner (2026-10-06) read CE@6 3.9280 and K1-K6 +0.0155. The rerun read 3.9187 and +0.0095. The graph recipe read 3.9316 and +0.0088. The rerun and the winner differ by 0.0060 in K1-K6 and 0.0093 in CE@6. The 480-row CIs are about 0.0006, so the spread is between runs. Source: `lab/experiments/failures/2026-10-08-production-rerun-noise-floor.md`.
- Verdict: filed as a protocol failure, since no prediction was written. The data stand. Consequence: the winner's +0.0155 was a high draw. Numerics-only changes can be tested at fixed weights. Changes to the training function need a paired design or several seeds per arm.
- Led to: the paired continuation design (next entry).

#### 2026-10-08: Paired continuation noise (instrument)

- Idea: resume two continuations from one checkpoint (`step_2500.pt`), same seed and data order, 1000 steps. Measures the noise a paired design leaves.
- Why: one 5k run cannot test the owner's rule "loop contribution within 5 % of baseline." 5 % of K1-K6 is about 0.00045 to 0.0006.
- Result: |delta K1-K6| 0.0005 for the first pair, and 0.0012 across three draws (+0.0086 to +0.0098). |delta CE@6| 0.0007. Pairing removes about 80 to 90 % of the independent-run spread. Source: `lab/experiments/mixed/2026-10-08-paired-continuation-noise.md`.
- Verdict: mixed. The first pair was a low draw, and the amendment says so. Several pairs per cut are needed for 5 %. Pairs resolve a change of about 0.0015 or more in K1-K6.
- Led to: the FAST2 package paired test and the compaction test.

#### 2026-10-08: FAST2 package and online teacher, paired (`fan_target_online`, speed package)

- Idea: the whole speed package (see the fast recipe) plus `model.fan_target_online: true`, the online-prelude router teacher. Tested as paired continuations from one checkpoint.
- Why: the owner's 5 % rule. The online teacher is the only key that changes the training function. The rest are rounding changes.
- Result: CE@6 of the three package runs at 4.0461, 4.0457 and 4.0471, all within 0.001 of the eager baseline mean (4.04645). K1-K6 +0.0003 and -0.0007 against the band for two runs. Run d read K1-K6 +0.0199 (high). Its depth-6 CE matched the others and its depth-1 CE was 0.011 worse. Run e reran the arm and read +0.0082. Trainer speed about 19.2k tok/s against 11.5k for eager on the same rows. Source: `lab/experiments/mixed/2026-10-08-fast2-package-paired.md`.
- Verdict: mixed. CE held in all three runs. One K1-K6 draw broke the high side, and the rerun was inside the band. Side finding: the gain hinge read one pass for a whole run in eager production too (the seed chose it). The fresh-pass change (2026-10-10) addresses that.
- Led to: the paired test of compaction.

#### 2026-10-08: Slot compaction (`model.slot_depth_stratified: true`, `model.slot_compact: gather`)

- Idea: two keys that change the function. The depth draw becomes stratified per row (each slot's marginal depth law unchanged, the joint changed). Frozen cells serve the attention input cached at their last live pass, not a fresh pass (fixed-weight loss rel 2.6e-4, up to 8 of 80 winner picks flipped).
- Why: the active-slot cut in the speed plan. Pad slots and finished slots still computed.
- Result: CE@6 means: compaction 4.04733, eager 4.04637, package 4.04630. Mean K1-K6 +0.01037 against +0.00917 (eager) and +0.01230 (package). Speed about 18.8k tok/s at step 2600 against 18.2k. Trainer val 4.0920 against 4.0904. Source: `lab/experiments/successes/2026-10-08-slot-compaction-paired.md`.
- Verdict: success on the stated predictions. The K1-K6 drop is below about 0.0015. CE may cost 0.001 to 0.0016, at the edge of what three runs resolve. The stratified-only arm would split the depth-draw cause from the cache cause. That arm was not run in this range.
- Led to: compaction is part of the fast base. The stratified-only split is open.

#### 2026-10-10: Credit panel round 1: one-cell loop, random pick, prelude detach

- Idea: three arms, each on `lxtul_pointer_fast` at 5k, each changing one credit path.
  - One-cell loop (`tul_onecell_pointer_fast`): no fan, no selection, no z. Tests what the 4 cells and the selection buy.
  - Random pick (`lxtul_pointer_fast_randpick`, `tul.fan_lsel_pick: random`): same 4 cells and reset, but the followed cell is uniform at train and eval. The router still trains on z as an instrument.
  - Prelude detach (`lxtul_pointer_fast_preldet`, `tul.coda_prelude_detach: true`): the coda reads the prelude, but the token CE does not train the prelude through the coda.
- Why: the credit matrix said the loop's weights learn 0.69 from the span decoder, about a fifth from token CE, and nothing from the next-span latent z. Wolfe was not confident this was the right credit structure.
- Result: references CE@6 3.919 to 3.932 (mean 3.9245), K1-K6 +0.0088 to +0.0104, write worth 0.154 to 0.168. One-cell loop CE@6 3.8997, K1-K6 +0.0022, write worth 0.193. Random pick CE@6 3.9344, K1-K6 +0.0060. Prelude detach CE@6 4.2237, K1-K6 +0.0135. Source: `lab/experiments/failures/2026-10-10-lxtul-credit-panel.md`.
- Verdict: failure of the hypotheses. The one-cell loop beat LXTUL on CE by 0.019 to 0.032 and lost on K1-K6. Random pick is inside noise on CE, so the z selection's worth is undecided. The prelude detach loses 0.29 CE, which is real. The prelude needs the token gradient through the coda. Wolfe, recorded not argued: he expects 4 cells to win after about 2B tokens. The knee costs too many tokens to see at 20k.
- Led to: round 2 and round 3.

#### 2026-10-10: Span decoder reads the winner (`tul.spandec_read: winner`, `lxtul_pointer_fast_spdwin`)

- Idea: the span decoder grades the winner cell only, not the mean of the 4 exit cells. The coda reads only the winner, so this aligns the target with the readout. The rival account: the mean is a selection pool, and dropping the losers' gradient lets them decay.
- Why: the decoder was 0.69 of the loop's gradient, and the losers carried 2.6 times the winner's share.
- Result: two runs, same seed. CE@6 3.9093 and 3.9290 (mean 3.9192). K1-K6 +0.0046 and +0.0124 (mean +0.0085). `fan/oracle_gap` 0.024 and 0.062. Rate 23.9k tok/s, 20 % below the references (not explained). Source: `lab/experiments/failures/2026-10-10-spandec-reads-winner.md`.
- Verdict: failure. The two runs do not separate the accounts. Single-run noise is 0.020 CE@6 and 0.008 K1-K6. The proposed next step is a paired continuation with three draws per arm. Not run in this range. `spandec_read: winner` was not adopted.

#### 2026-10-10: Span decoder grades each cell alone (`tul.spandec_read: each`, `lxtul_pointer_fast_spdeach`)

- Idea: the decoder grades each cell alone against the next span, then averages the four losses. Each candidate gets the gradient of its own decodability.
- Why: test whether a per-candidate text signal helps the loop and the selection.
- Result: CE@6 3.9159 (references 3.9239). K1-K6 +0.0166, the highest K1-K6 in the panel. Span-decoder K1-K6 +0.0445, also the highest. `fan/oracle_gap` 0.043 (wrong side of 0.031 to 0.035). Cell cos 0.98 (LXTUL 0.95). Trainer rate 18.3k tok/s (39 % below). Source: `lab/experiments/failures/2026-10-10-spandec-grades-each-cell.md`.
- Verdict: failure. The CE and K1-K6 "help" readings are inside single-run noise. The cells did not become distinct under the per-cell grade. The 39 % rate cost rules it out for the recipe until more runs exist.
- Led to: the fan decoder arms on 2026-10-11, which repeat this test on the ungraded fan.

#### 2026-10-10: LXTUL without the span decoder (`tul.spandec: false`, `lxtul_pointer_fast_nospandec`)

- Idea: turn off the training-only span decoder on LXTUL and read the loop's own contribution.
- Why: the decoder costs 5.68 of 21.67 TFLOP per step. Tracker item C1d, approved by Wolfe on 2026-10-10.
- Result: CE@6 3.9927 against references 3.9248 and 3.9229 (+0.068 to +0.070). CE@1 also worse by 0.058 to 0.063. K1-K6 +0.0020 against +0.0104 and +0.0091. Trainer rate 34.7k against 29.8k tok/s (+16 %). Source: `lab/experiments/failures/2026-10-10-lxtul-fast-no-spandec-5k.md`.
- Verdict: failure. The decoder stays. Its CE effect is decisive on LXTUL. Its K1-K6 drop is about 1.3 times the independent-run spread on one seed. The filing's first K1-K6 number was overstated and was corrected the same day. The cost becomes a speed target. Note the tension: on the ungraded fan, a mean-read decoder costs CE (see 2026-10-11).

#### 2026-10-10: Cell lens probe (`lab/divergence/cell_lens.py`)

- Idea: measurement of what each of the 4 cells holds, per pass. Patch (frozen coda, coda-only CE of the next span), a linear probe on cell to tied embedding, special-property NMI, and CKA/rank geometry. Run on the ungraded fan plus pointer and on LXTUL plus pointer (winner read).
- Why: Wolfe asked for "a future lens style decoding of each cell." The fan and LXTUL cells were thought to be specialists or copies.
- Result: fan patch CE at depth 6: cells 4.6305, 4.6704, 4.6685, 4.6787; all four 4.5467; per-span best cell 4.5587. Cell 0 wins 0.59 to 0.60 of spans. Copied tokens are 0.5376 of scored tokens. Cell 0 alone carries 0.126 of the 0.170 copied worth. LXTUL cells 4.5971, 4.5670, 4.5594, 4.5685. Between-cell CKA 0.832 (fan) and 0.940 (LXTUL). The probe CE overfit and is unusable. Only top-1 is used. Source: `lab/experiments/failures/2026-10-10-cell-lens.md`.
- Verdict: failure against its predictions. The best-cell gap was 0.012 (predicted within 0.010). Cell 0 won 0.59 (predicted 0.35 to 0.55). Real finding: the ungraded fan's cells split the work. Cell 0 is the copy cell. The novel-token value sits in the combination.
- Led to: the set-reader cell lens on 2026-10-11, and the copy-split arms.

#### 2026-10-10: Ungraded fan with pointer, and one-cell without decoder (round 2)

- Idea: round 2 of the panel. (a) Ungraded fan plus pointer (`nowta_pointer_fast`): 4 independent lineages, all written to the coda, no z, EMA twin, router, teacher or reset. (b) One-cell loop without the decoder (`tul_onecell_pointer_fast_nospandec`).
- Why: distinct lineages were the old fan idea. The fan had the best CE in the older eager filings.
- Result: (a) CE@6 3.9196 (predicted 3.897). K1-K6 +0.0045. Write worth 0.181. Cell cos -0.09. Trainer 24.7k tok/s, 17 % below LXTUL. (b) CE@6 3.8996, K1-K6 +0.0015, write worth 0.186. Trainer 32.7k tok/s. Source: `lab/experiments/failures/2026-10-10-credit-panel-round-2.md`.
- Verdict: failure of the headline prediction. The fan does not beat the one-cell loop at 5k. It matches the low end of LXTUL. Distinct cells cost no CE. The span decoder buys nothing for the one-cell loop at 5k.
- Led to: round 3.

#### 2026-10-10: Fan without decoder, and second runs (round 3, `nowta_pointer_fast_nospandec`)

- Idea: the fan without its span decoder (`tul.spandec: false`), plus second runs of the fan with decoder and of the one-cell loop, to size the noise.
- Why: round 2 showed the decoder buys nothing on the one-cell loop. Does it on the fan?
- Result: fan without decoder CE@6 3.8994, K1-K6 +0.0024, cell cos -0.28, 28.2k tok/s. Fan with decoder run 2 CE@6 3.9227. Two-run mean for fan with decoder is 3.9212 and K1-K6 +0.0042. One-cell run 2 CE@6 3.9029 (two-run mean 3.9013). Source: `lab/experiments/failures/2026-10-10-credit-panel-round-3.md`.
- Verdict: failure of the headline prediction (3.920 predicted, 3.8994 measured). The decision rule says drop the decoder from the fan. Caveat: one run on that side. The two-run spreads were 0.003, and the 0.022 gain sits at the 0.020 same-config noise floor. Direction holds, size does not.
- Led to: the 2026-10-11 replicate, and the set-reader decoder, which Wolfe chose over dropping the decoder.

#### 2026-10-10: Gain hinge reads a fresh pass each step (`model.slot_gain_pass_force: -1`)

- Idea: the slot loop's gain hinge (`model.slot_gain_lambda` 100 at target 0.9) reads one loop pass per step. Before this change the pass was fixed for the run, because the draw restored the global CPU stream. The draw is now a private generator, one uniform per step, one graph per (kind, pass).
- Why: tracker A1 found that all 54 draws were pass 6 on seed 1 and pass 3 on seed 2. Two seeds of one recipe regularised different passes. Wolfe decided a fresh pass per step on 2026-10-10 (tracker A2). Rejected: one pass per run, and every pass per step (about +200 ms).
- Result: paired against three fixed-pass continuations. CE@6 4.0482, 4.0480, 4.0487 (fixed mean 4.0473). K1-K6 mean +0.0121 (fixed +0.0104). Every pass drawn (chi-squared 4.38). Cost +2.0 % per step. Source: `lab/experiments/successes/2026-10-10-hinge-fresh-pass-paired.md`.
- Verdict: success. CE and K1-K6 hold at this test's resolution. The fresh draw is the default. Open: `model.core_gain_lambda` in `_core_region` still reads a frozen pass.
- Led to: the fan recipe and the diet chain both run with the fresh draw.

#### 2026-10-10: DiscoTUL built (`tul.disco`, `discotul_fast`, `lxtul_disco_fast`, `tul_onecell_pointer_fast` as control)

- Idea: between loop passes, add `alpha * RMSNorm(Phi(h))`, where Phi is the model's own token head, applied at the input of pass t at least 1. Bridges soft, top-k (k 128) and hard. Learned gate, starting at 0.5. Two arms: DiscoTUL (strict slot loop, one cell) and LX-DiscoTUL (LXTUL with the term on every cell). Wolfe, 2026-10-10: "test it as an ALTERNATIVE to LXTUL, both ways."
- Why: DiscoLoop (arXiv 2607.00341) reports gains on multi-hop tasks.
- Result: bench only, 420 steps, graph_step. LXTUL pointer fast 230.1 ms/step. LX-DiscoTUL 274.1 ms (+19 %). One-cell control 201.5 ms. DiscoTUL 244.9 ms (+22 % over its control). The disco term costs about 44 ms per step. No training run in this range. Source: `.agents/notes/proposed/process/2026-10-10-phase-2-tracker.md` (item D2a).
- Verdict: built on `feat/discotul` (not merged). 41 CPU tests and 10 of 10 sabotages. Training arm not run (item D2b, after the optimization items).
- Led to: open. The filing notes that a 5k arm would sit inside MORPH's noise, since the paper's gain appears after about 13B tokens.

#### 2026-10-11: Fan decoder grades the mean, each cell, or the best cell

- Idea: three decoder reads for the ungraded fan. Mean read (`tul.spandec_read: mean`, the inherited default). Each-cell read (`tul.spandec_read: each`). Best-cell read (`tul.spandec_read: best`, the winner per span). Each read is compared with no decoder.
- Why: the round 3 result said the decoder costs the fan CE. The question was whether the cost comes from the decoder's gradient or from the target it grades.
- Result (two runs each except best, which is one): no decoder CE@6 3.8993, K1-K6 +0.0025, cos -0.28. Mean read 3.9212, K1-K6 +0.0042, cos -0.10. Each-cell 3.9032, K1-K6 +0.0062, cos +0.81, 18.5k tok/s. Best-cell 3.9066, K1-K6 +0.0063, cos +0.61, win shares near uniform (0.24 to 0.26). Source: `lab/experiments/failures/2026-10-11-fan-each-cell-decoder.md`, `...-fan-best-cell-decoder.md`.
- Verdict: failure on the mechanism. Each-cell removes the mean-read CE cost (0.018 better), but the cells collapse. Any decoder that grades cells on the same span pulls them together, even when each span trains only one cell. The filing's proposal: a decoder that keeps diversity must grade the cells as a set. That is what the next entry tests.

#### 2026-10-11: Fan no-decoder replicate (`nowta_pointer_fast_nospandec`)

- Idea: a second run of the fan without decoder, to test the 0.022 CE gain from round 3.
- Why: round 3's gain rested on one run.
- Result: run 2 CE@6 3.8993, within 0.0003 of run 1 (two-run mean 3.8993). Fan with decoder two-run mean 3.9212. Gain 0.022. K1-K6 +0.0025 against +0.0042. Cell cos -0.28. Rate 14 % higher. Source: `lab/experiments/successes/2026-10-11-fan-no-decoder-replicate.md`.
- Verdict: success. Proposal to Wolfe: drop the decoder from the fan recipe.
- Led to: superseded the same day by the set-reader decoder, which Wolfe chose.

#### 2026-10-11: Set-reader span decoder, the fan recipe (`tul.spandec_reads_cells: true`, `nowta_pointer_fast_spdcells`)

- Idea: the span decoder cross-attends to the four cells as an unordered set, the way the coda reads them. Its z conditioning stays the mean. The decoder is training only. Code in `morph/model/tul_spandec.py` (`_SpanDecCross`).
- Why: the mean-read and per-cell decoders each collapse the cells or cost CE. A decoder that reads the set should keep the coda's division of labour.
- Result: run 1 CE@6 3.8949, K1-K6 +0.0052, cos -0.28, 23.9k tok/s. Run 2 CE@6 3.8953, K1-K6 +0.0060. Master (tree-j) CE@6 3.8954, K1-K6 +0.0051. CE prediction was 3.912 and missed low. Cost about 15 % of throughput against no decoder. Source: `lab/experiments/failures/2026-10-11-fan-set-reader-decoder.md`, `lab/experiments/successes/2026-10-11-fan-set-reader-on-master.md`.
- Verdict: adopted as the fan recipe by Wolfe's decision on 2026-10-11 ("his original design"). The filing is under failures only because the CE prediction missed. Caveats recorded in the note: every run is seed 1; the 0.0004 agreement is run-to-run nondeterminism, not seed variance; the margin over the no-decoder fan has no second seed. The 2026-10-12 eval regime check (below) says the seed 1 family these runs sit in is not the typical value. Cached and graphed decoders refuse the fan (`tul.fan_k > 0`).
- Led to: the speed base, variable cell count, and the diet chain.

#### 2026-10-11: Set-reader cell lens (success)

- Idea: rerun the cell lens on the set-reader checkpoint to see whether the cells keep the copy split.
- Why: the cell lens had shown cell 0 is the copy cell in the mean-read fan.
- Result: cell 0 holds about 70 % of the copied-token worth. Novel-token worth needs the combination. The total worth is 0.006 higher than the plain fan's, all of it on copied tokens. Cell 1 carries more alone (novel worth 0.052 against about 0.02). Source: `lab/experiments/successes/2026-10-11-set-reader-cell-lens.md`.
- Verdict: success on its predictions. One checkpoint, one lens run.

#### 2026-10-12: Eval regime check (correction to the seed 1 family)

- Idea: does the new eval regime (graphs kept, probe every 4th eval, RNG forked around eval) change training?
- Why: six seed 1 runs of the fan family read CE@6 3.8949 to 3.8996. Four runs under the new regime read 3.9029 to 3.9061.
- Result: the same config under the old regime, seed 2, read CE@6 3.9076. The new regime's matched seed 2 read 3.9029. Fresh 5k trajectories now read 3.903 to 3.908 (five runs). Source: `lab/experiments/successes/2026-10-12-eval-regime-check.md`.
- Verdict: success. The new regime costs nothing. The earlier seed 1 family "was one family of near-copied trajectories, not the recipe's typical value." Pass lines for later checks are set against matched fresh runs.
- Impact: the 2026-10-10 and 2026-10-11 fan numbers in the seed 1 range (set reader 3.8949 and 3.8953, master 3.8954, no decoder 3.8993) should be read with this in mind. The fan-arm ranking at the 0.004 level is not settled.

#### 2026-10-13: Variable cell count v1 (`tul.cell_count`, `tul.cell_count_mean`, arms `..._cellg375` and `..._cellp3`)

- Idea: each training batch draws its cell count M, at most 4. Two draw laws: gentle (P(M=3) 0.25, P(M=4) 0.75, a random subset of roles) and Poisson (Poisson(3.5) clamped to [1, 4], roles 0 to M-1). Merged as `tul.cell_count` (default `fixed`, merge 33113fb6).
- Why: Wolfe asked, 2026-10-05, "can I train at 4 and inference at 128 fine?" The draw is also a FLOP lever and a downward knob.
- Result (two seeds each; control is the fixed four cells, 3.9050 and +0.00550). At M = 4: gentle CE@6 3.9033 (-0.0018) with K1-K6 97 %, Poisson 3.9052 (+0.0002) with K1-K6 95.5 %. Weighted wall 0.981x (gentle) and 0.921x (Poisson). Gap to M = 4 for the Poisson arm: M = 1 +0.0051, M = 2 +0.0002, M = 3 -0.0001. Gentle arm: M = 1 +0.0899, M = 2 +0.0252, M = 3 +0.0004. Upward: every arm loses at M = 8 (+0.06, cyclic) and M = 16 (+0.10 to +0.14). Source: `lab/experiments/successes/2026-10-13-fan-variable-cell-count-v1.md`.
- Verdict: success on its decision rules. The downward knob is live: a Poisson-trained model reads M = 1 within 0.005 of M = 4 (the fixed-four control read +0.135). No arm gains above 4 cells. Wolfe's "train at 4, infer at 128" question is not answered by v1, because the cells have roles. The design note (`.agents/notes/proposed/architecture/2026-10-12-fan-variable-cell-count-design.md`) asks for v2: exchangeable cells and a pooled coda read.
- Led to: the speed base variant with Poisson draws. The cell mean 2 diet cut (next entry).

#### 2026-10-13: FLOP diet stack, one-block decoder and cell mean 2 (`tul.spandec_layers: 1`, `tul.cell_count_mean: 2.0`, `..._cellp2_dec1`)

- Idea: two config-only cuts to the speed base. The training-only span decoder keeps its full-span target with one block instead of two. The Poisson cell mean drops from 3.5 to 2.0, so mean M goes from about 3.0 to about 2.06.
- Why: the speed base spent 14.24 TFLOP per step. The decoder was 2.775 TFLOP. The cheaper cut `tul.spandec_max_tokens: 8` was already rejected (CE@6 +0.026 at 5k).
- Result: the stack (both cuts) at seeds 1 to 3: CE@6 3.8999, 3.8939, 3.9058 (mean 3.8999, -0.0014 against the pooled control 3.9013). K1-K6 mean +0.0064. Bench 157.4 ms (speed base) against 139.6 ms (stack), 0.887x. Cell mean 2 alone passes at 3 seeds (3.9007, +0.0059). One-layer decoder alone fails by the letter at 2 seeds (+0.0061). At mean 2, M = 1 and M = 2 read the same as M = 4 (|gap| at most 0.0007). Source: `lab/experiments/successes/2026-10-13-flop-diet-5k.md`.
- Verdict: success for the stack. Stability note: stack seed 1 had grad-norm max/median 45.5, one spike at step 240 in warmup, which the filing judged recovered. Wolfe can overrule. The decision unit is the stack, not the single cuts.
- Led to: round 2, the speed base v2 keys, and the Parcae plan's lean recipe (see open threads).

#### 2026-10-14: Hinge every four steps (`model.slot_gain_every: 4`, `..._cellp2_dec1_ge4`)

- Idea: the one-pass gain hinge runs on every 4th step, with 4 times its lambda. It costs 1.069 TFLOP per step on a side stream.
- Why: the hinge is about 4.6 % of the stack's step. The hinge holds the map's gain near 0.89. Less frequent pushes may let the loop use more depth.
- Result: three seeds. CE@6 mean 3.9043 (+0.0030 against control, on the GREY line by 0.00003). Paired against the stack: +0.0094, +0.0089, -0.0049 (mean +0.0045). K1-K6 mean +0.0083 (160 % of control). Bench 123.9 ms against the stack's 129.96. Source: `lab/experiments/failures/2026-10-14-flop-diet-round-2-5k.md`.
- Verdict: GREY. Wolfe's call. Not in the speed base. The filing's reading: a lighter hinge lets the loop use depth more, and the same change costs token CE. This matches the older view that the loop-holding terms hold the loop still.
- Led to: open for Wolfe.

#### 2026-10-14: Half-slot span decoder (`tul.spandec_slot_keep: 0.5`, `..._cellp2_keep05`)

- Idea: the two-block packed decoder trains on a random token-budget half of each row's slots, each with its full next-span target.
- Why: a cheaper decoder with the same expected gradient to the cells.
- Result: two seeds, CE@6 mean 3.9075 (+0.0062, FAIL by the letter). K1-K6 mean +0.0049 (94 % of control). Bench 123.1 ms, about the same speed as hinge every 4, and 0.011 worse CE than the one-block stack. Source: same filing as above.
- Verdict: failed. The filing's reading: the decoder needs every slot's target each step. Subsampling costs about as much as cutting a block, and the block cut is the cheaper of the two at equal speed.
- Led to: the one-block decoder stays in the stack.

#### 2026-10-14: Speed base v2 rounding keys (`model.hc_bwd_fused`, `model.glue_fold`, on the stack)

- Idea: two rounding-only speed keys on the stack, merged the same night. `hc_bwd_fused` moves the HC entry backward projection to tensor cores. `glue_fold` compiles the span-decoder norms and SwiGLU, pointer mass, tied-head table, injection term, and an exact pad and cast.
- Why: the speed base needs more speed. The keys are gated as rounding-only.
- Result: three seeds, paired against the stack. CE@6 difference +0.0034 (sd 0.0026, over the +0.003 line). K1-K6 difference -0.0001, 95 % CI [-0.0075, +0.0073]. Bench 129.0 ms to 122.1 ms, 1.056x. Source: `lab/experiments/failures/2026-10-14-speed-base-v2-levers-5k.md`.
- Verdict: FAIL by the letter. The filing says the data neither show a cost nor rule one out of about 0.003. It matches the K1 case that Wolfe accepted under his paired reading. The keys stay default off, and the speed base stays the stack until Wolfe calls it.
- Led to: a call for more seeds on both arms (six per arm).

### Threads left open in this era

- Wolfe's calls pending: hinge every 4 (GREY, +0.00303 against a +0.003 line), the speed base v2 keys (+0.0034 paired), and whether the speed base becomes the stack.
- The 2B-token knee. Wolfe expects 4 cells to win later. The planned 20k run (`lab/experiments/planned/2026-10-10-onecell-vs-lxtul-20k.md`) is not queued. Wolfe said on 2026-10-10: "Do not queue the 20k steps right now." Its prediction: the one-cell gap near -0.02 or widening.
- Variable cell count v2 (exchangeable cells, pooled coda read) is the only route to Wolfe's "train at 4, infer at 128" question. Tracker items B1 to B4 are open.
- DITTO redesign: rows on top of the batch, fewer copies per pseudo row, a longer or low-rate DITTO. The junk-subword artifact is not counted. DITTO on the fan is untested.
- Plain plus pointer ruler on MORPH (owed since 2026-10-06). A pointer seed twin is also owed. The winner note lists both.
- `spandec_read: winner` needs the paired continuation test (three draws per arm). Not run. The each-cell read needs two more runs before any recipe call.
- Fan comparisons at the 0.004 level rest on seed 1 runs that the eval regime note says are not typical. A multi-seed check of the fan arms is open.
- The stratified-only arm (depth draw without the frozen-cell cache) would split compaction's cause. Not run.
- `model.core_gain_lambda` in `_core_region` still reads a frozen hinge pass.
- `model.slot_gain_rows` (hinge on k rows, rotating) is proposed, not built.
- DiscoTUL and LX-DiscoTUL (`tul.disco`): built, not merged, not trained. Tracker item D2b waits for the optimization items.
- JEPA-style factored fan (tracker item D1): optional ablation, gated by a cheap pick-agreement probe.
- The Parcae 40k handoff (`.agents/notes/proposed/process/2026-10-14-parcae-lxtul-speed-and-40k-plan.md`) is another agent's job. Its Step 3 "lean recipe" lists the ungraded fan at cell mean 2, the set reader, one decoder block, half-slot decoder and hinge every 4. Two of those (half-slot and hinge every 4) failed or stayed grey on MORPH. The plan does not mention this. Flag for whoever runs it.
- Two numbers for one gap disagree in scope. The planned 20k filing says the strict one-cell loop (fp01) led LXTUL by 0.007 at 5k and 0.029 at 10k against plain. The Parcae plan says "MORPH's one-cell pair went from 0.265 at 5k to 0.333 at 10k." These may be two different gaps (to LXTUL, and to plain). Not reconciled in this range.

### Retractions and corrections in this range

- The winner's K1-K6 of +0.0155 (2026-10-06) was a high draw. The production rerun read +0.0095 (2026-10-08 noise floor). Any "the winner's loop is +0.0155" claim should cite the rerun too.
- The 2026-10-10 no-decoder LXTUL filing first overstated K1-K6. It was corrected the same day: the drop is 0.0071 to 0.0084, about 1.3 times the independent-run spread.
- The 2026-10-07 coverage sweep rows were contaminated by `init_from` restarting the train stream at doc 0. The phase comparison used matched sweep rows and clean val. The control sweep gap (-0.209) is not comparable with the 5k gap (-0.150).
- The 2026-10-12 eval regime check revised the fan seed 1 family. Six runs at CE@6 3.8949 to 3.8996 are "one family of near-copied trajectories." Fresh runs read 3.903 to 3.908. This touches the 2026-10-10 and 2026-10-11 fan arm numbers in that range (set reader, master, no decoder).
- The 2026-10-08 paired noise's first pair (|delta K1-K6| 0.0005) was a low draw. Three draws give a spread of 0.0012.
- The DITTO phase cost +0.071 trainer val against its control. Its filing lists the cost as open and proposes a redesign. The winner note, written the same day, declares the recipe with that cost recorded. The redesign was not run in this range.

### Disagreements between files

- Decoder effect on LXTUL and on the fan points in opposite directions. On LXTUL, removing the span decoder costs CE by 0.07 (`lab/experiments/failures/2026-10-10-lxtul-fast-no-spandec-5k.md`). On the ungraded fan, the mean-read decoder costs CE by 0.022 (`lab/experiments/failures/2026-10-10-credit-panel-round-3.md`, `2026-10-11-fan-no-decoder-replicate.md`). The filings do not reconcile these. The set-reader result is the fan's current answer.
- The 2026-10-06 pointer filing calls the CE result "a strong positive." The 2026-10-07 coverage filing reads the same pointer model as repeating more under sampling: seq_rep_4 0.291 at T 0.7 against plain's 0.215. Both hold. They measure different things, and the winner recipe takes both.

---

## TUL history, part 8: the Parcae port (2026-09-05 to 2026-10-14)

Sources: Parcae repo `/home/wolfe/parcae` (branch `lxtul-testbed`, read only), the filings under
`docs/experiments/{successes,failures,planned}/`, `lxtul/configs/arm/*.yaml`, `lxtul/SPEC.md` (intro and
section headers), the git log, the drawing-board ledger and README, `pz-bench/bench4.md`, and the run
directories in `/home/wolfe/parcae-runs/` and `/home/wolfe/parcae/outputs/TUL-frontier/`. I did not read
`parcae-speed/REPORT.md` in full. Speed numbers come from `bench4.md` and the filings.

Numbers are CE in nats per token unless stated. K1-K6 is CE at depth 1 minus CE at depth 6 of the same
model (positive means later passes pay). Parcae 5k runs use 480 paired held-out rows.

### Era summary

- September 2026 had two Parcae tracks. The first was a clean engineering port of serial, boundary-only
  TUL with full cross-span gradients. It cost a lot of work and produced no quality result yet. The
  second was a slot-only conditioner panel on stock Parcae at mean recurrence 16. It found no useful depth.
- On 2026-10-06 the LXTUL loop was ported to Parcae. Its loop earned K1-K6 +0.0585, 2.7 times MORPH's
  two-seed mean. The bare one-cell loop earned nothing, so the LX machinery was what made the loop earn.
- The CE gap to plain (+0.29) came from the strict mask. A copy channel closed it. An exact-copy cache closed
  it to +0.02 with the loop kept. A learned pointer head did the same job, and a normalised version fixed a
  training bug. Like for like, the pointer gap is +0.045.
- Round 2 showed that K1-K6 rises as the depth-1 CE gets worse. Seed spread of K1-K6 is about 0.02, so
  single-seed K1-K6 differences under about 0.02 cannot be read.
- On 2026-10-14 Wolfe set the paper on the slot loop itself ("Slot loop is the paper"). The lean arm was
  withdrawn. The one-cell minimal loop was 1.78 times faster than LXTUL + pointer, and it earned nothing.
- The 4-cell minimal fan also earned nothing at seed 1 (K1-K6 0.0000). That contradicts the 10-14 filing's
  premise that the fan buys depth. A 5k latent-selected fan run was still training at step 3800 when I read it.
- Flat reference arms with the same pointer head are planned and not run. The publishing claim needs them.

### Entries

#### 2026-09-05: Stock Parcae loop contribution, the positive control (`parcae-small-140m`, mean 8, BPTT 4)

- Idea: no TUL. The author's Parcae 140M (2/2/2 layers, 144M params) on OWT, with its own mean-8 recurrence,
  BPTT 4, and Muon. Kept as a reference for "does a recurrent Parcae learn depth on these rows".
- Why: to check that the Parcae recipe earns depth before building TUL on it.
- Result: at 10k, K1-K8 = 0.294 nats [0.286, 0.303]; K3-K8 = 0.0397; K8 minus K16 = 0.0002. At 5k, K1-K8 =
  0.261. K8 CE improves 0.393 from 5k to 10k. Source:
  `docs/experiments/successes/2026-09-05-parcae-owt-loop-contribution.md`.
- Verdict: success. All five predictions held.
- Led to: nothing direct. The LXTUL testbed uses a bigger shape (d 1024, 4/6/4 layers) and its own plain arm
  (K1-K6 +0.046 at 5k, different recipe).

#### 2026-09-05: Selective compile throughput for Parcae 140M (`selective block compile`, no config key)

- Idea: compile the blocks selectively. Not a TUL change.
- Why: to decide whether a custom Parcae kernel was worth writing.
- Result: +16.8 to +49.3 percent tokens per second over eager blocks. Best case: 91.9k tok/s at seq 2048,
  batch 4, 8.36 GB peak. v1 was invalid because the Dynamo recompile limit fell back to eager. Source:
  `docs/experiments/successes/2026-09-05-parcae-140m-throughput.md`.
- Verdict: success. No custom Parcae kernel. The filing names the next check: profile at the production shape.
- Led to: no direct follow-up in the filings. Later benches time every arm in one harness (`bench4`).

#### 2026-09-05: Slot-only conditioner panel on stock Parcae (`slot16-b4`, `slot16-b8`, `cond4-16-b4`, `cond4-16-b8`, `coda4-16-b4`, `coda4-16-b8`)

- Idea: slot-only TUL on Parcae. Tokens never enter the recurrent core. One slot per completed span, built from
  the contextual prelude state at the boundary. Mean recurrence 16. Three placements of 4 extra blocks:
  none extra (`slot16`), after the slot loop (`cond4`), and after the token coda (`coda4`). BPTT 4 or 8.
  Source: `outputs/tul-queue/<arm>/` and the filing, 10k steps, batch 12, seq 2048, 140M model.
- Why: the old MORPH panel bypassed the slot by compressing it. This tested the placement idea on a clean
  backbone.
- Result: K3-K16 is +0.00011 to +0.00018 in all six arms (CI excludes zero, far under the 0.005 threshold).
  CE at K16: slot 3.590 (BPTT 4) and 3.591 (BPTT 8); `cond4` 3.591 and 3.594; `coda4` 3.472 and 3.473.
  Training tok/s: 146k (slot), 134k (cond4), 106k (coda4). `cond4` loses 0.118 and 0.121 nats to `coda4` at
  matched BPTT, and is 26 percent faster. Zeroing all thoughts at K16 adds 0.049 to 0.051 nats for slot and
  cond4, and 0.089 to 0.143 for coda4. Shuffling adds 0.008 and 0.016 to 0.019. BPTT 8 is worse at K16 in
  all three placements. Source: `docs/experiments/failures/2026-09-05-parcae-tul-cond4-panel.md`, and the
  ledger table 1b.
- Verdict: failure of the useful-depth hypothesis, not of execution. The filing says the thought reader
  "can use a nearly depth-independent slot feature". It also says the zero and shuffle tests are not
  independently trained no-thought controls, so they do not show a training advantage.
- Led to: the question "is slot input quality or post-loop processing the limit". Nothing ran on it. The
  October LXTUL port took a different route: a copy channel, not more blocks.

#### 2026-09-05: Serial boundary-only TUL, 16 arms (`embedding-all_tokens-attached-s1-k8` and 15 siblings, `serial_tul_setup`)

- Idea: causal serial TUL. The core runs once per completed span (boundary-only). One thought per span is
  published and never changed. Tokens are decoded from the fixed thought with full cross-span gradients and
  full BPTT 8, K8 fixed. The 16 arms cross four axes: decoder input (`embedding` or `prelude`), thinking
  (`all_tokens` or `fixed_tokens`), MUX head (`attached` or `detached`), and slots (1 or 3). Loss is one
  combined token CE plus a next-span geometric MUX target (beta 0.1, rho 0.9). Source:
  `docs/experiments/planned/2026-09-05-serial-tul-15k-panel.md`.
- Why: the old slot-only panel compressed the thought. Serial TUL keeps the causal history and the gradient
  path through the thought.
- Result: no quality result. Only engineering. Preregistered gates passed: causality, immutable thoughts, one
  backward, and 50 CPU tests. Full-context batch-one smoke at seq 2048 ran 4 updates with 12.28 GiB peak.
  Launched 2026-09-05 as one user service. The first arm was stopped at update 511 for an execution repair.
  The 2026-09-06 frontier queue reached 11,965 of 15,000 updates in its first arm, then a stop command ended it
  (`outputs/TUL-frontier/monitor.json`). The other 15 arms are pending. The planned file's verdict is "Pending".
- Verdict: pending. No scientific claim is made.
- Led to: the throughput and packing entries below. Their filings call them the path to the 15k panel.

#### 2026-09-05: Serial TUL stop and exact resume (`run_control`, `serial_tul_setup` probe configs)

- Idea: update-boundary checkpoints, SIGTERM and SIGINT handlers that set a flag, a writer flock, a checksummed
  latest pointer, and exact RNG and cursor restore.
- Why: a 15k run must survive stops, kills and restarts without a changed trajectory.
- Result: six four-update CPU continuations matched an uninterrupted reference exactly (parameters,
  optimizer, RNG, loss, cursor). Two SIGKILL cases were covered. 78 tests and 209 subtests passed. Source:
  `docs/experiments/successes/2026-09-05-serial-tul-stop-resume.md`.
- Verdict: success, engineering only. Power loss, GPU bitwise replay and remote W&B exactly-once delivery were
  not tested.
- Led to: the full trainer and queue checks.

#### 2026-09-05: Serial TUL full trainer, queue and capacity (`serial_tul_setup action=train`)

- Idea: run the real serial trainer through a sequential queue of 16 arms, with milestones at 5k, 10k and 15k.
- Why: the stop and resume path had to work on the real trainer.
- Result: the batch-two capacity prediction failed. Original compiled attention peaked at 24.13 GiB on an
  ordinary batch-two smoke. Worst-row batch one (row 13246 has 381 thinking episodes; mean is 110) peaked at
  18.26 GiB after the partition-saving attention fix. A shared K/V allocation per boundary brought the
  fixed-token three-slot case to 20.83 GiB. Batch one was fixed for all 16 arms. Source:
  `docs/experiments/failures/2026-09-05-serial-tul-full-launch-engineering.md`.
- Verdict: failure of the prediction, measured fallback to batch one. Panel can compare arms at equal updates
  and tokens, but cannot claim token-matched gains over the dense reference.
- Led to: packed coda and cross-row packing, both aimed at batch one's GPU gaps.

#### 2026-09-05: Serial TUL throughput repair (`serial_execution` K/V publication and compiled segments)

- Idea: CPU-prepared shifted targets, batched MUX, K/V-only publication, larger compiled recurrence segments.
  Same math, same batch, same boundary order.
- Why: batch-one updates were slow and launch-bound.
- Result: median warmed update 2.4515 s to 1.6307 s (1.503 times, a narrow pass on the 1.5 target). Mean 1.612
  times. Peak allocated 7.58 GiB to 6.23 GiB. Migration of the stopped run into `outputs/TUL-optimized` was exact.
  Source: `docs/experiments/successes/2026-09-05-serial-tul-throughput-repair.md`.
- Verdict: success. The filing says Python and GPU synchronisation caused a material part of the gap, and the
  rest is structural: K8 recurrence launches many small matrix products.
- Led to: packed coda.

#### 2026-09-05: Packed coda, one variable-length coda pass (`serial_execution` packed coda, `max-autotune-no-cudagraphs`, scoped shared-gradient fusion)

- Idea: keep the prelude and core serial. Run the coda once over concatenated episode queries, with
  variable-length FlashAttention per episode and lower-right causality over the published prefix. Pack QKV
  and KV casts once per update.
- Why: the coda launched one small attention sequence per span.
- Result: fresh unpacked control 1.3241 s median, 48.36 percent GPU use. Packed 1.01680 s (1.302 times), GPU
  42.22 percent. Coda attention calls fell from 228 to 2. `max-autotune` was slower (1.07475 s) and took 662 s to
  compile. The scoped shared-gradient fusion matched the graphs, but Muon amplified the reduction-order change
  to 35 to 40 percent of MLP update norms. The fusion was removed. Source:
  `docs/experiments/failures/2026-09-05-serial-tul-packed-coda.md`.
- Verdict: failure on the 1.00 s and 50 percent GPU targets. The packed coda itself was kept.
- Led to: row concurrency, then cross-row packing.

#### 2026-09-05: Row concurrency on CUDA streams (`serial_execution` row streams, removed)

- Idea: independent rows run on separate CUDA streams, each row's boundary order unchanged.
- Why: the core within one row is a chain, but rows are independent.
- Result: batch-two median 2.5248 s (serial) to 2.2390 s (streams), 1.128 times. Mean GPU 45.3 to 50.9 percent.
  The largest Muon parameter update differed by 61.8 percent of the reference update norm, against a 10 percent
  target. Source: `docs/experiments/failures/2026-09-05-serial-tul-row-concurrency.md`.
- Verdict: failure. Stream code and its compile lock were removed.
- Led to: the packed thinking frontier, which shares matrix products instead of launching them on streams.

#### 2026-09-05: Packed thinking frontier (`serial_execution` frontier, index-map custom op)

- Idea: at each round, take one ready boundary from each unfinished row. Concatenate them into one packed core
  pass. Each row keeps its own published history through variable-length attention. Gradients return to each
  original partition. Index maps are built on CPU and uploaded once per batch.
- Why: rows are independent, so their core work can share matrix products. Stream concurrency did not do this.
- Result: batch-two median 2.525 s to 1.637 s (1.542 times). Mean GPU 45.34 to 61.39 percent (target 65).
  Peak 10.70 GiB against 10.78 GiB for row-serial. Batch-two capacity passed for three production shapes at
  9.64, 7.26 and 7.88 GiB. About 1.82 rows are active per round on average. Source:
  `docs/experiments/failures/2026-09-05-serial-tul-packed-frontier.md`.
- Verdict: failure on the GPU target by 3.6 points. Filing says the frontier is still used, because it cuts
  median update time by 35 percent with no added peak memory.
- Led to: the 2026-09-06 decision to start all 16 arms fresh at batch two. The batch-one checkpoint at update 750
  is not resumed. The 15k panel is the run in the entry above. The frontier profile planned for 2026-09-06
  (`docs/experiments/planned/2026-09-06-serial-tul-frontier-profile.md`) has no results.

#### 2026-10-06: LXTUL port panel on Parcae (`arm=lxtul`, `arm=plain`, `arm=lxtul_ctxinj`, `arm=bare`)

- Idea: the faithful port of MORPH's `lxtul.yaml` (4 cells, latent-selected loop, fixed-point term 0.1, gain
  hinge 100, cotangent clip 4, span decoder, epivol 0.1) on Parcae's single-stream backbone. `lxtul_ctxinj`
  uses MORPH's channel-restricted injection. `bare` keeps only the strict geometry, one cell, and the hinge,
  fixed-point term and clip. Port spec: `lxtul/SPEC.md`. Shape d 1024, layers 4/6/4, batch 6, seq 1024.
- Why: MORPH's LXTUL earned K1-K6 +0.0215 and +0.0172 and trailed plain by about 0.27. Was it the backbone?
- Result: plain K1-K6 +0.0462 (predicted at least +0.10, failed), CE 3.8303. LXTUL K1-K6 +0.0585 [0.0564, 0.0607],
  CE 4.1242, gap to plain +0.294. MORPH injection K1-K6 +0.0703, CE 4.1248. Bare K1-K6 -0.0001, CE 4.0826.
  Source: `docs/experiments/failures/2026-10-06-parcae-lxtul-5k-panel.md`.
- Verdict: failure by the frozen rule (one of seven predictions, and it was the baseline's depth). The hypothesis
  held: the port earns 2.7 times MORPH's loop contribution. MORPH's injection is not the reason. The LX
  machinery is what makes the loop earn here. The filing adds a 300-step LR warmup amendment after the first
  launch: Wolfe said "you need a short warmup at least". The plain arm was stopped and not reused.
- Led to: round 2, and the gap decomposition.

#### 2026-10-06: Round 2, seed twin and LX ablations (`arm=lxtul` with `run.seed=2`, `arm=nospandec`, `arm=ungraded`)

- Idea: seed twin of full LXTUL. Full LXTUL with `spandec_weight 0` (no span decoder). Ungraded fan with
  `select none` (no router, no reset, no EMA target). The 10k plain and LXTUL runs were deferred.
- Why: is the round-1 earning a seed accident, and which part makes the loop earn?
- Result: seed twin K1-K6 +0.0794 [0.0771, 0.0817], CE 4.1252, gap +0.295. No span decoder K1-K6 +0.1325,
  CE 4.1304 (CE 0.006 worse than full). Ungraded fan K1-K6 +0.0346, CE 4.1100. Source:
  `docs/experiments/failures/2026-10-06-parcae-lxtul-round2.md`.
- Verdict: failure of predictions 1 and 2. Filing says K1-K6 spread between seeds is 0.021 while CE agrees to 0.001.
  Ordered by K1-K6, the arms are also ordered by depth-1 CE (4.083, 4.145, 4.183, 4.205, 4.263). Each LX part that
  raises K1-K6 raises depth-1 CE more than depth wins back. Filing proposes scoring against a fixed anchor, not each
  arm's own depth 1.
- Led to: the gap decomposition. The 10k runs were never run.

#### 2026-10-06: Gap decomposition: strict mask versus depth (`arm=gpt8`, `arm=lxtul_open`, `arm=lxtul` probe)

- Idea: a flat 8-layer GPT with full context (`gpt8`), and LXTUL with open geometry (`lxtul_open`: tokens read
  every earlier token; cells stay strict). Per-token gap probe `lxtul/gap_probe.py`.
- Why: is the 0.29 gap per-token depth (cost) or cross-span information (the mask)?
- Result: gpt8 CE 3.8086, better than plain (predicted in [3.90, 4.10], failed). Open LXTUL CE 3.8508, K1-K6
  +0.0042. Strict minus open = +0.2734, 93 percent of the gap. Per-token depth = -0.022. Probe: far-bigram tokens
  are 17.9 percent of tokens and 85.7 percent of the gap. Source:
  `docs/experiments/failures/2026-10-06-parcae-gap-decomposition.md`.
- Verdict: the predictions 1 and 2 failed, but the question was answered: the gap is the strict mask, and inside
  it, copying.
- Led to: the copy cache.

#### 2026-10-06: Exact-copy cache (`arm=lxtul_cache`, `lxtul/cache.py`)

- Idea: an output-only, non-parametric copy mixture. For token i, earlier tokens with the same bigram or unigram
  key propose their next token. A trained gate mixes the proposals with the model's own distribution. No hidden
  state carries identity.
- Why: the gap probe said the strict model cannot copy across spans.
- Result: untrained cache probe on the strict checkpoint recovers 0.197 nats (plain 0.030). Trained strict + cache:
  CE 3.8518, gap to plain +0.0215 [0.011, 0.031], K1-K6 +0.0491. Leak check: the cache-off score is CE 4.1182 and
  K1-K6 +0.057, so the loop kept its use. Source: `docs/experiments/failures/2026-10-06-parcae-copy-cache.md`.
- Verdict: the main hypothesis held. Predictions 2, 5 and 8 failed. Prediction 8 failed for an informative
  reason: after opening, open-geometry tokens do not copy well by themselves within 2500 steps.
- Led to: the seed twin, the strict-then-open switch, and the learned copy channels.

#### 2026-10-06: Copy cache seed twin (`arm=lxtul_cache`, `run.seed=2`)

- Idea: the same arm, seed 2.
- Why: strict + cache closed the gap. Was that a seed draw?
- Result: CE 3.8533 (seed 1: 3.8518), gap +0.0230 [0.013, 0.032], K1-K6 +0.0690. Source:
  `docs/experiments/successes/2026-10-06-parcae-copy-cache-seed2.md`.
- Verdict: success. Two seeds agree on CE and gap to 0.0015. K1-K6 spreads 0.020, the same as strict LXTUL.
- Led to: the copy cache is the Parcae recipe for CE at 5k, and the loop stays at strict LXTUL's level.

#### 2026-10-06: Strict then open, with and without the cache (`arm=lxtul_switch`, `arm=lxtul_cache_switch`)

- Idea: train strict to step 2500, then open to step 5000. The cache variant keeps the cache on for both halves.
- Why: Wolfe asked for it: "start it without open geometry so it is forced to learn to handle closed geometry
  then move to open geo and see what happens."
- Result: strict then open CE 4.0600 (open eval), K1-K6 +0.0304, gap +0.230. Under the strict mask at the end,
  4.2224. With the cache, CE 3.8287, gap -0.0017 [-0.012, +0.007] (ties plain), K1-K6 +0.0290, strict re-score
  3.9212. Source: the copy-cache filing.
- Verdict: the switch without the cache costs about 0.21 nats at 5k, and the open half does not recover it. With the
  cache the switched model ties plain.
- Led to: nothing directly. The strict-then-open idea was not pursued further. Not a 10k test.

#### 2026-10-06: Learned pointer head, unnormalised (`arm=lxtul_pointer`, `arm=plain_pointer`, unnormalised first pass)

- Idea: a learned copy head, output only. 4 heads, dh 64. Keys and queries come from coda outputs (span-local in
  strict geometry). Each head sums its attention onto the targets of the attended tokens, gated with the model.
  `lxtul/copy_heads.py`.
- Why: Wolfe: build a less hacky option that "I would be willing to include in the final version".
- Result (unnormalised, first filing): strict + pointer CE 3.8047, K1-K6 +0.0579, head off 4.1661 (0.042 worse than
  strict LXTUL). The head is worth 0.361 nats. Plain + pointer CE 3.7072. Source:
  `docs/experiments/failures/2026-10-06-parcae-copy-heads.md` and `...plain-pointer.md`.
- Verdict: the head is the result, but the filing's headline ("0.026 ahead of plain") compared a pointer model with a
  plain model that had no pointer. The like-for-like gap was +0.098 (unnormalised). The filing's amendment says the
  pointer scores were pessimistic, because the mass on the null key was dropped. Those numbers are superseded by the
  normalised entry below.
- Led to: the normalised pointer and the cell-key test.

#### 2026-10-06: Identity reach, raw embedding copy (`arm=lxtul_reach`, `lxtul/copy_heads.py` IdentityReach)

- Idea: before the coda, each token attends over earlier tokens' raw embedding windows [e(x_{j-1}), e(x_j), e(x_{j+1})].
  8 heads, zero-initialised output projection. The loop and cells are not touched.
- Why: test whether copied identity can enter the hidden state without breaking the loop.
- Result: CE 4.1089, gap +0.2786, K1-K6 +0.0446, cell write worth +0.257. Far-bigram gap falls only from 1.41 to
  1.32. Source: `docs/experiments/failures/2026-10-06-parcae-copy-heads.md`.
- Verdict: failure. Filing says this is a statement about this module, not identity in the hidden state in general.
- Led to: pointer heads whose keys are hidden states.

#### 2026-10-06: Normalised pointer head and cell keys (`arm=lxtul_pointer`, `arm=lxtul_pointer_cellkey`, `arm=plain_pointer`, normalised)

- Idea: the null mass is returned to the model (a normalised mixture). Cell-key variant: keys for earlier spans also
  read that span's loop cells (`cell_key`).
- Why: the pointer was pessimistic. Like-for-like gap needed a fair head on both sides.
- Result: strict + pointer normalised CE 3.7608, K1-K6 +0.0468 [0.0452, 0.0485], gap to plain + pointer +0.0453
  [0.0427, 0.0478]. Plain + pointer normalised CE 3.7155 (the unnormalised plain + pointer was 3.7072; the
  normalised training made plain 0.008 worse). Cell keys: CE 3.7534, K1-K6 +0.0452, gap +0.0379 (-0.0074 paired).
  Source: `docs/experiments/failures/2026-10-06-parcae-pointer-cellkey.md`.
- Verdict: failure on two predictions. The fix mattered through training, not scoring: a head that can abstain is
  worth -0.044 to strict LXTUL and +0.008 to plain. Strict LXTUL + pointer now recovers 85 percent of the 0.294 gap
  with no head on either side. Cell keys add a small real gain. Wolfe: "I am interested in that next lever. I think
  its worth running."
- Led to: generation checks. The 2026-10-14 flat references use the same normalised head.

#### 2026-10-06: Pointer generation check (`lxtul/generate.py`, 32 prompts, 128 new tokens)

- Idea: a measurement, not a training arm. Free generation with five arms and the real continuations. Metrics:
  seq_rep_4, ctx_copy_4, max_copy, distinct_4, gen_ppl (scored by plain-5k).
- Why: Wolfe: "We need generation samples. This method of scoring bigrams may cause repetition."
- Result: under nucleus sampling the head raises copy and repetition only a little. Strict + pointer seq_rep_4
  0.0105 against 0.0013 without the head (real text 0.0258). Under greedy, strict + pointer loops at 0.79, plain at
  0.78. Strict without the head loops less only because it cannot copy. Max verbatim copy rises by 2.3 tokens over
  the twin for strict + pointer (the prediction for this failed). Source:
  `docs/experiments/failures/2026-10-06-parcae-pointer-generation.md`.
- Verdict: failure on predictions 2 and 6 as written. The decision rule (guard only if sampled repetition is more
  than 0.02 above the twin) did not trigger.
- Led to: a watch item for longer training: the longest verbatim run.

#### 2026-10-14: Lean arm (withdrawn) (`arm=lean`, `lxtul/configs/arm/lean.yaml`)

- Idea: the levers that held on MORPH at 5k (ungraded fan, Poisson cell count with mean 2, set-reader span decoder
  with one block) stacked on Parcae with the exact speed keys.
- Why: Wolfe wanted a sizable speed gain, little CE cost, and loop contribution, shown against plain.
- Result: none. Withdrawn before any 5k run. Bench: 230.8 ms per step and 13.83 TFLOP, against 269.7 ms and 21.49
  TFLOP for LXTUL + pointer. Source: `docs/experiments/failures/2026-10-14-parcae-lean-5k.md`.
- Verdict: withdrawn. The file says the lean arm only slims machinery that Parcae does not need.
- Led to: the minimal slot loop.

#### 2026-10-14: Minimal slot loop, one cell (`arm=minimal`, `arm=minimal_noptr`)

- Idea: strict geometry, one cell per slot (rows of 1024 + 64 = 1088 positions), Parcae's diagonal injection,
  token-state dropout 0.15, and the normalised pointer head (`minimal_noptr` drops the head). Removed with no compute
  and no parameters: span decoder, gain hinge, fixed-point term, cotangent clip, latent selection, epivol. Speed
  keys on: pad-slot compaction, static depth, CE on loss rows, no pass checkpoints. Wolfe: "Slot loop is the paper."
- Why: LXTUL carries MORPH machinery that costs FLOPs (21.5 TFLOP per step, of which about 3.3 is useful loop work).
- Result: CE@6 3.7647 and 3.7615 (two seeds). K1-K6 -0.0001 and -0.0000. No pointer: CE 4.1266, K1-K6 -0.0000. Paired
  against plain + pointer seed 2: CE +0.037. Trainer tok/s 39.4k and 39.8k (1.78 times LXTUL + pointer; step bench
  153.4 against 269.5 ms, 1.76 times). Cell change per pass: 8.5 percent, then 0.45, 0.07, 0.05, 0.05. Zeroing every
  slot write costs 0.28 nats, all in pass 1. Source:
  `docs/experiments/failures/2026-10-14-parcae-minimal-slot-loop-5k.md`.
- Verdict: failure of predictions 3 and 4 (the loop does not earn). The filing says the process error was the author's:
  the one-cell `bare` arm from 2026-10-06, which had the stabilisers, was already flat (4.0826 at every depth). Prediction
  3 was written without reading it. The head-off check had crashed and was fixed in e8ede0a before it was read.
- Led to: the minimal fan.

#### 2026-10-14: Minimal fan, four ungraded cells (`arm=minimal_fan`, run dir `minfan-5k`)

- Idea: the minimal loop with 4 cells per slot back. No router, no reset, every cell written to its own prefix
  position, so the coda reads all 4. The stabilisers and span decoder stay off.
- Why: the filing's stated premise was that every 4-cell arm earned, so the fan buys depth.
- Result: seed 1 CE@6 3.7446 (predicted in [3.735, 3.800], holds), K1-K6 +0.0000 [-0.0000, +0.0001] (predicted at least
  +0.040, fails). Trainer tok/s 32.7k. Source: `/home/wolfe/parcae-runs/minfan-5k/summary.json`. No result filing yet
  (the planned file's Results section is empty). Seed 2 and the two-cell fan have no run dir.
- Verdict: the run contradicts the premise. The one 4-cell strict arm with every stabiliser off reads 0.0000. The
  drawing-board note `hypothesis-before-lsel.md` says so before the run ended: "'the fan buys depth' (my morning read)
  was wrong; the fan was a passenger."
- Led to: the latent-selected fan (below). The hypothesis note proposes that a target reading one slot (span decoder or
  latent head) is what buys depth.

#### 2026-10-14: Minimal fan with latent selection (`arm=minimal_fan_lsel`, run dir `minfan-lsel-5k`)

- Idea: the minimal fan with the router, winner-only write, and EMA target put back.
- Why: test whether latent selection alone buys the large K1-K6 once the span decoder and stabilisers are gone.
- Result: not filed. At my read it was at step 3800 of 5000 (`metrics.jsonl`). Per-step train CE is noisy (3.96 at step 3775, 3.59 at step 3800), and
  there is no validation sweep yet. No `summary.json`. Do not treat any number from it as final.
- Verdict: pending.
- Led to: the open question in the hypothesis note: is K1-K6 at least +0.02 under the minimal fan with latent selection?

#### 2026-10-14: Flat references with the pointer head (`arm=gpt8_pointer`, `arm=flat14_pointer`)

- Idea: the flat models with the same normalised pointer head that every slot arm carries. `gpt8_pointer` matches the
  loop's prelude and coda depth (403M). `flat14_pointer` matches the unrolled loop depth (630M).
- Why: Wolfe's claim is Parcae against LXTUL-Parcae with a flat transformer as reference. The only flat number was gpt8
  without a pointer, 3.8086.
- Result: none. Planned only. Bench step times at batch 6: gpt8 + pointer 79.8 ms, flat14 + pointer 123.3 ms. Source:
  `docs/experiments/planned/2026-10-14-parcae-flat-pointer-references-5k.md`.
- Verdict: not run.
- Led to: nothing yet. Whether a deep loop beats flat at equal width at 5k is still unknown.

#### 2026-10-14: Speed ladder (`bench4`, every arm one harness, `bench_all.py`)

- Idea: a measurement of step time, tokens per second and TFLOP for every arm, batch 6 and 12, one harness.
- Why: to decide whether the LX machinery is worth its cost.
- Result, training step at batch 6 (ms, from `pz-bench/bench4.md`): gpt8 75.9; gpt8 + pointer 79.8; flat14 119.6; flat14 +
  pointer 123.3; minimal no pointer 131.3; minimal 153.4; plain + pointer 281.8 (240.5 with row compaction); LXTUL + pointer
  269.5; LXTUL + pointer with exact speed keys 242.1 (233.8 without pass checkpoints). Batch 12: plain + pointer ran out of
  memory without checkpointing (28 GiB), and the checkpointed run is 475 ms. The lean bench (230.8 ms) is in its filing.
  The `REPORT.md` figures (plain + pointer 283.5, LXTUL + pointer 270.7) differ by under 1 percent from these.
- Verdict: a measurement. The filing and the bench agree within noise. Speed keys (`compact_slots`, `static_depth`,
  `ce_rows`, `spandec_packed`, `ckpt_passes`) are exact or rounding-only.
- Led to: the minimal arm, and the lean arm's withdrawal.

### Disagreements and retractions

- Fan and depth. `minimal-slot-loop` filing (2026-10-14): "the fan is the element that buys depth." The same-day
  hypothesis note: "the fan was a passenger." The `minfan-5k` run (K1-K6 0.0000) supports the note. The planned fan
  file's premise, which the filing's conclusion relied on, is contradicted by its own run.
- Pointer scores. Unnormalised pointer numbers (strict 3.8047, plain 3.7072) are superseded by the normalised numbers
  (3.7608, 3.7155). The copy-heads headline "0.026 ahead of plain" is superseded by the like-for-like gap (+0.098
  unnormalised, +0.045 normalised).
- Plain + pointer normalised training is worse than unnormalised by 0.008 nats (3.7155 against 3.7072). The filing
  reads this as the head abstaining cleanly. Not a retraction, but a number that moves the other way.
- Single-seed K1-K6. The round-1 headline (+0.0585) and the seed twin (+0.0794) differ by 0.021. Any K1-K6 difference
  under about 0.02 is inside seed noise for the LXTUL family.
- Round 2 and K1-K6 against depth-1 CE. Filing says K1-K6 rises with depth-1 CE. Not a retraction. It changes how the
  metric should be read, and the filing proposes a fixed anchor.
- Ledger item 3. `bare-5k` ran on a dirty tree (`git_dirty: true` at bc03f38). The round-1 filing says "code bc03f38
  for every arm". Not reconciled.
- Ledger item 2. The minimal fan latent run was at step 650 when the ledger was compiled. At my read it was at 3800.
  Any ledger number for it is stale.
- Speed claims. The minimal filing's 1.78 times comes from trainer logs. The step bench gives 1.76 times.

### Threads left open in this era

- Minimal fan with latent selection (`minfan-lsel-5k`): summary not written at read time. Predictions 2 and 3 in its
  prereg depend on it. Its planned file has no results.
- Minimal fan seed 2 and the two-cell fan: no run dirs, no filing.
- Flat references with the pointer head, seeds 1 and 2 each: planned, not run. The publishing claim needs them.
- Normalised pointer, seed twin: the normalised strict pointer arm has one seed. The like-for-like gap (+0.045) is single seed.
- Plain + trained cache (the fair gap with the cache on both sides): untested. The probe suggests about +0.05.
- 10k runs for plain and LXTUL: deferred in round 2 and never run.
- Strict-then-open switch at 10k, and the MORPH port of the copy cache: not run.
- The emit-label direction (move the first-token label to the slot's own cell, `emit_weight` moved to the cell):
  proposed in the 2026-10-14 README as direction A. The Parcae port has no emit key yet. Not run.
- Small token loop (direction C, mean depth 2 or 3 on tokens): a question for Wolfe, not run.
- Serial TUL 15k panel: 15 arms pending, first arm stopped at 11,965 of 15,000 updates. The frontier profile
  (`2026-09-06-serial-tul-frontier-profile.md`) has no results.
- Serial TUL: the batch-two capacity is OK, but the GPU use target (65 percent) is still missed. Next steps named in
  the filings: batch independent rows at one ready boundary, and a fused recurrent weight gradient.
- Reach module: a different identity channel was not tried after the raw-embedding reach failed.
- Variable cell count at train and test (Wolfe's idea) is not on the Parcae port. Not run.
