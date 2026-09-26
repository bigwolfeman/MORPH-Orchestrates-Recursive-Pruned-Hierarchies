# Planned: does a wider slot channel (a) or a previous-span read (b) close the gap to plain

Status: success

Date: 2026-09-26 05:35 (frozen before any arm's first 5000-step GPU step; 30-step smokes ran
during the build and read nothing about CE at 5k).
Parent: [`../failures/2026-09-26-lxtul-fp01-vs-plain-10k.md`](../failures/2026-09-26-lxtul-fp01-vs-plain-10k.md).
Note: [`../../../.agents/notes/proposed/architecture/2026-09-26-slot-channel-width-and-reach.md`](../../../.agents/notes/proposed/architecture/2026-09-26-slot-channel-width-and-reach.md).
Wolfe's go: 2026-09-26 ("we should test a and b"; "Make sure this all runs end to end").

## Question

fp01 trails the plain model by 0.265 nats at 5k on the same coda tokens, and the gap widened
to 0.333 at 10k. The slot channel is the only cross-span route under the strict geometry.
Does widening it (a) or adding a direct read of the previous span's tokens (b) close the gap?

## Arms (all 5000 steps, seed 1, built at 839b4e0)

| arm | config | differs from fp01 |
|---|---|---|
| a1 | `tul_slot_spandec_strict_e4probe_fp01_pk4` | `tul.prefix_k: 4` (4 cells, each `W_prefix[k] z` of ONE exit state; L_total 1280) |
| b | `tul_slot_spandec_strict_e4probe_fp01_reach1` | `tul.tg_coda_token_reach: 1` (a coda token reads the previous span's tokens; token-path receptive field 4 spans through the 4 coda layers) |
| a2 | `tul_slot_spandec_strict_fan4_all_fp01` | NOT one-factor: fan4 write-all with the fixed-point term 0.1; lacks fp01's code rollouts and parallel head (both refuse the fan), and differs in `spandec`, `slot_gain_target` 0.9, `slot_gain_tail_lambda` 0 |

## Instruments

- Gap to plain: `paired_vs_ruler.py` of each arm's `core_depth_sweep` (480 rows) against
  the plain panel's 5k sweep at depth 6, token-paired by stream index (fp01 reads +0.2650
  [+0.2503, +0.2819] this way, the same as the notul pairing's +0.2651). a1 and b also get
  `lxtul_e_stage2_score.py` vs fp01 and the notul pairing; a2 cannot (no parallel head).
- Channel worth: `worth_profile.py` (192 rows), zero and shuffle totals (fp01 5k: 0.197 /
  0.165).
- Coda K1−K6 from the sweep (fp01 5k: +0.0125).
- The tripwire on each run's per-step probe.

## Predictions

Cross-arm CE at one seed has a measured spread of 0.0137 on this family, so "closes the gap"
means by at least 0.02.

- **A1-1.** a1's gap <= 0.245: **30 %.**
- **A1-2.** a1's coda K1−K6 >= 0.005: **60 %.**
- **B-1.** b's gap <= 0.15: **55 %.**
- **B-2.** b's gap <= 0.10: **30 %.**
- **B-3 (the bypass).** b's channel worth (zero) below fp01's 0.197: **60 %.**
- **B-4.** b's coda K1−K6 >= 0.005: **40 %.**
- **A2-1.** a2's gap <= 0.245: **40 %.**
- **H.** Each arm's tripwire HEALTHY: **65 %** each.

## Verdict rules

Per arm: success if its gap clause (A1-1, B-1, A2-1) holds. b additionally reports B-3 and
B-4: a b that closes the gap by bypassing the loop (B-3 holds, B-4 fails) is filed as a
success on the gap and a failure for the loop, both named. An arm that succeeds at 5k is
resumed to 10k and paired against plain 10k (the test fp01 failed). No clause passes on a
cosine.

## Results (filed 2026-09-26 11:17)

All three arms trained 5000 steps, exit 0, through the queue runner
(`/home/wolfe/morph-scratch/abc/runner.sh`, log in
[`../results/2026-09-26-slot-channel-arms-ab/chain.runlog.txt`](../results/2026-09-26-slot-channel-arms-ab/chain.runlog.txt)).
Artifacts: [`../results/2026-09-26-slot-channel-arms-ab/`](../results/2026-09-26-slot-channel-arms-ab/).

| arm | gap to plain 5k (coda @6) | arm − fp01 (same tokens) | coda K1−K6 | worth zero / shuffle | tripwire | tok/s, peak |
|---|---|---|---|---|---|---|
| fp01 (reference) | +0.2650 [+0.2503, +0.2819] | 0 | +0.0125 | 0.197 / 0.165 | HEALTHY | 5851 |
| a1 pk4 | +0.2729 [+0.2580, +0.2897] | +0.0074 [+0.0048, +0.0102] | +0.0112 [+0.0103, +0.0120] | 0.2104 / 0.1736 | HEALTHY, max 27.9 @4328 | 5723, 18.98 GB |
| b reach1 | +0.2501 [+0.2350, +0.2668] | −0.0148 [−0.0172, −0.0125] | +0.0085 [+0.0077, +0.0092] | 0.0818 / 0.0245 | HEALTHY, max 47.3 @2832 | 6077, 17.03 GB |
| a2 fan4-all-fp01 | **+0.2181 [+0.2027, +0.2353]** | **−0.0472 [−0.0501, −0.0445]** | +0.0057 [+0.0052, +0.0063] | 0.2003 / 0.1903 | HEALTHY, max 30.1 @1846 | 7783, 18.61 GB |

- Gap: `paired_vs_ruler.py` against the plain panel's 5k sweep at depth 6, 501,106 tokens.
  b's notul pairing agrees: +0.2501 [+0.2463, +0.2538].
- arm − fp01: `paired_vs_ruler.py` of each arm's sweep against fp01's 5k sweep
  (`lxtul-fu/sweep_lxtul-e4probe-fp01_5000.json`) at depth 6, on the same 501,106 tokens.
  This replaces the stage-2 scorer's fp01 pairing, which ran only for b: the scorer
  exits 1 on a1 (it builds the batches from fp01's prefix_k 2 layout) and on a2 (no
  parallel head, as the prereg said).
- Worth by offset inside the span (zero ablation, bins 0, 1, 2, 3, 4-7, 8-15, 16+):
  a1 1.008 / 0.402 / 0.303 / 0.265 / 0.203 / 0.148 / 0.103; b 0.088 / 0.124 / 0.106 /
  0.103 / 0.092 / 0.078 / 0.066; a2 0.785 / 0.419 / 0.323 / 0.278 / 0.214 / 0.147 / 0.093.
  b's direct read carries the span's first token, which is where the slot channel's worth
  sits on every other arm.
- Final trainer val loss (not an instrument here): a1 4.4919, b 4.4122, a2 4.4365, fp01
  4.4199.

## Clause by clause

- **A1-1 FAILS.** a1's gap is +0.2729; it is 0.0074 WORSE than fp01 on the same tokens.
- **A1-2 HOLDS.** +0.0112.
- **B-1 FAILS, B-2 FAILS.** +0.2501, 0.015 better than fp01.
- **B-3 HOLDS.** b's channel worth fell to 0.082 (shuffle 0.025).
- **B-4 HOLDS.** +0.0085.
- **A2-1 HOLDS.** +0.2181, with the whole CI below 0.245.
- **H HOLDS** for all three arms.

## Verdict

- **a1: failure.** Four cells that are four projections of ONE exit state do not help. The
  channel's worth barely moved (0.197 → 0.210), and the gap is 0.007 worse.
- **b: failure.** A direct read of the previous span's tokens REPLACED the slot channel
  (worth 0.197 → 0.082, shuffle 0.165 → 0.025) and closed only 0.015 of the gap. The
  previous span's tokens and the slot channel carry nearly the same information. The loop
  still earns a little (+0.0085).
- **a2: success on the gap.** fan4 write-all with the fixed-point term 0.1 is 0.047 nats
  better than fp01 on the same tokens (3.4x the seed spread) and 0.047 closer to plain.
  a2 is NOT one-factor: it differs from fp01 in the fan, the missing code rollouts and
  parallel head, `spandec`, and two gain knobs. Its loop earns the least of the three
  (+0.0057, half of fp01's), so the gain is in what the coda reads, not in depth. One seed.

## Updated hypothesis

The channel's limit is not the number of cells (a1) and not access to the previous span's
tokens (b). What helped is four DIFFERENT states, each written to its own cell and read by
the coda (a2: shuffle worth 0.190 is 95 % of its zero worth, so the content is specific to
the slot). Plan C tests that directly on fp01's recipe: xHC's 16 streams give each of the
4 cells different content, with a1 as the control
([`../planned/2026-09-26-plan-c-xhc-arms.md`](../planned/2026-09-26-plan-c-xhc-arms.md)).
a2's 10k resume, paired against plain 10k, is queued
([`../planned/2026-09-26-fan4-all-fp01-10k.md`](../planned/2026-09-26-fan4-all-fp01-10k.md)).
Not measured: a second seed of any arm, and a2 with fp01's code rollouts (the fan refuses
them).
