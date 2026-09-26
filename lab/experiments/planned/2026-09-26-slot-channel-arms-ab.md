# Planned: does a wider slot channel (a) or a previous-span read (b) close the gap to plain

Status: planned

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
