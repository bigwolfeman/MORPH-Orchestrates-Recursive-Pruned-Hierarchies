# Agent Note: Stp Label Shift Bug

Status: implemented

Origin: Ai-notes/06-23-2026/STP-AB-Results/LABEL_SHIFT_BUG.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# SFT collapse ROOT CAUSE: label-shift bug in sft_data.py (2026-06-23)

## TL;DR
The catastrophic SFT generation collapse (single-token loops) on the MORPH checkpoints was **NOT**
STP, pruning, ternary-QAT, packing, or data-volume. It was a **missing next-token label shift** in
`morph/training/sft_data.build_dolly_examples`. Every SFT result from 2026-06-22/23 (prediction-2
STP-SFT, the no-STP control, the padded runs) was computed on a **broken training objective** and is
VOID. Fixed + gated; re-run trains healthy.

## The bug
- **Pretraining** (`morph/training/data.py:57`): `labels[t] = input_ids[t+1]` — labels are PRE-SHIFTED
 (next-token). The model and the fused CE do **NOT** shift internally (grep: no shift in
 `transformer.py` / `fused_ce.py`). So the model CONTRACT is: feed pre-shifted (next-token) labels.
- **SFT** (`sft_data.py`, old): `ids = p_ids + c_ids; labels = [-100]*len(p_ids) + c_ids`. This makes
 `labels[t] == ids[t]` on response positions — the CURRENT token, not the next. UNSHIFTED.
- Consequence: the model was trained to predict the token it had **already consumed** — a trivial copy
 task. → teacher-forced loss collapses to **ppl ≈ 1.0**, and at generation the model emits the token
 it just saw → **degenerate single-token loops** ("lowerlower…", "....", "\n\n\n…").

### Proof (real Dolly example, `ignore/gate_sft_label_shift.py`)
Over 64 examples / 5349 supervised positions, with the OLD code: **495/496 response labels == ids[t]
(current)**, only ~0.8% == ids[t+1] (coincidental token repeats). With the FIX: **100% == ids[t+1]**,
copy-current down to 0.8%.

### Why this explains every prior observation
- ppl→1.0 on **unique** data (not memorization — it's a trivial copy task).
- Single-token generation loops (the model literally learned "emit current token").
- Packed ppl≈4 vs padded ppl≈1.0: packing's cross-document boundaries are the ONLY positions where
 "copy current" is wrong (the token after an EOS boundary is unpredictable) → they hold avg ppl up;
 padded (one clean example/row) has almost none → ppl≈1.0.
- STP exonerated (no-STP control collapsed identically — CE was broken regardless of STP).
- Dose-response (more fitting of the copy task → worse free generation).
- Retroactively explains task **#164** ("EOS-failure looping from SFT data") — same class of bug.

## The fix (`sft_data.build_dolly_examples`)
NEXT-TOKEN labels: `labels[t] = ids[t+1]` supervised iff `t+1 >= len(p_ids)` (target is a response/eos
token), else -100; final position masked. This makes the last PROMPT token learn to emit the first
response token, the pre-eos token learn to STOP, and the final position carry no target. Supervised
count == len(response+eos), unchanged. Affects BOTH packed and padded paths (shared builder).

## Gates (no-theater)
- NEW `ignore/gate_sft_label_shift.py` → **SFT_LABEL_SHIFT_GATE_PASS**: on real Dolly, 100% supervised
 labels == ids[t+1]; copy-current 0.8%; no target in prompt region; final masked; supervised count ==
 response+eos length per example.
- `ignore/gate_sft_padded.py` was a **LYING GATE** — its `build_example` + check #3 encoded the UNSHIFTED
 (buggy) contract as correct (would pass on broken data, fail on the fix). FIXED to the shifted contract;
 re-run → **SFT_PADDED_GATE_PASS** 7/7 (incl. the no-theater broken-pad-mask catch).

## Validation — ✅ CONFIRMED end-to-end (2026-06-23)
Re-ran proper-data padded SFT (n_examples=12000, epochs=1, batch=8, ~1500 unique steps, flat LR 2e-5,
pruned off-base) with the fix. Loss oscillates **ppl ~6–30 (real NTP)**, final ppl ~15 — no crash to 1.0.

**Generation RECOVERED** (sampled OWT continuation + held-out instruction probe):
| metric | off-base(noSFT) | BROKEN padded | FIXED no-STP | FIXED STP |
|---|---|---|---|---|
| OWT rep-2 ↓ | 0.138 | 0.991 | **0.243** | 0.251 |
| OWT loop_frac ↓ | 0.067 | 0.981 | **0.077** | 0.078 |
| OWT distinct-2 ↑ | 0.862 | 0.009 | **0.757** | 0.749 |
| instruction EOS-stop ↑ | 8/13 | 0/13 | **11/13** | 11/13 |

- Free generation is back in the **base-like healthy regime** (loop_frac 0.077 ≈ base 0.067, NOT 0.98).
- The SFT'd model now **STOPS on instructions 11/13** (base 8/13, broken-SFT 0/13) and answers in the
 right format — genuine instruction-following emerged. RAW (greedy): "The capital of France is France."
 / "The following are the primary colors:" + bullets / "1. Alexander Ringhoffer" (stops @7 tok) / haiku
 attempt (stops w/ EOS).
- ⚠️ **Residual SENTENCE-level repetition + factual errors** ("2 + 2 is 2 + 2", "ocean is beautiful"×N).
 This is the genuine ceiling of a 276M ternary-0.25-density-looped model on light SFT — the *real*
 "how dumb it is" , NOT the catastrophic token-loop. Addressable via more data / stronger base /
 anti-repetition decoding; not a bug. The label bug had been MASKING the model's true (weak) behavior.
- **STP on≈off now** (rep-2 0.251 vs 0.243; both EOS 11/13) → prediction-2's "STP-SFT collapses generation"
 was 100% the label bug. RETRACTED.
- Outputs: `ignore/bench_off_1ep_{nostp,stp}.json`, `ignore/ig_off_1ep_*.json`; ckpts
 `checkpoints/morph/sft_off_1ep_{nostp,stp}/step_1500.pt`; wandb `sft_off_1ep_nostp`/`sft_off_1ep_stp`.

## Fallout / TODO
- ALL contaminated ckpts (sft_{off,on}_{stp,nostp,padded}, the 50k→SFT step_42/192) are garbage — delete.
- Prediction-2 verdict ("STP-SFT collapses generation") is RETRACTED — it was the label bug.
- Once generation confirmed coherent: re-run the clean prediction-2 (STP on/off) AND the dense-base
 prune/QAT control (`dense15k_kernels_b4`, `baseline_diag_epsout_noprune_8k`) — those experiments are
 only meaningful now that SFT actually works.
- The STP loss itself was ALSO fixed today (s<r<t single-triplet, paper-faithful; `prediction.py`,
 `STP_TRIPLET_GATE_PASS`) — independent correctness fix.
