# Planned: the norm-match rule on the shipped forward and at the 20k horizon

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "run the paid TUL loop under norm_match and
the 20k horizon then we do 20k norm match. We ignore bf16."). Arc:
`2026-09-04-loop-contribution-arc.md`. Follows the per-pass-strength panel
(`successes/2026-09-09-arc-per-pass-strength.md`) and the rule change it bound
(`base.yaml` `ternary_scale_mode: norm_match`, commit `2f3a128`).

## Question

Two things the per-pass-strength panel left unmeasured about the rule it shipped:

1. **The shipped forward.** Every strength arm ran the plain model. The recipe runs the
   paid TUL loop (tokens and slots through one core). Does the loop's contribution under
   norm_match (plain: K1−K6 0.185, K3−K6 0.014, core MLP branch 0.86–1.10) carry to the
   TUL forward, and what does the slot apparatus add or cost on top of it?
2. **The horizon.** At 5,000 steps the norm-matched plain model reads 0.08 nats above the
   absmean base on paired tokens, and the learnable-scale control says the loss gradient at
   that horizon prefers the weaker map. The absmean horizon run
   (`failures/2026-09-09-arc-horizon-ternary-25k.md`) showed the absmean loop's
   contribution flat from 5k to 10k. Does the norm-matched model close the CE gap by 20k,
   and does its contribution hold, grow or decay over training?

## Hypothesis

H-rec-1: the rule acts on the core, not on the slot apparatus, so the TUL arm under
norm_match reads a K-curve within 0.03 of the plain norm-match arm and the paid loop's
own earning (TUL minus plain, both norm_match) is of the same sign and size as under
absmean. H-rec-2: the 0.08 gap is the price of a deeper map and shrinks with training
(the absmean loop stopped contributing past pass 3 because its map was weak; a strong map
keeps a reason to loop), so the gap at 20k is smaller than at 5k and the contribution does
not decay. H-rec-2′: the gap does not close and the contribution decays: the stronger map
is a transient the optimizer trains away, and the rule would need a longer-horizon
justification than these runs give.

## Method

All arms on the Parcae-entry recipe (`notul_parcae_entry`: noise state init, all-dim
carry with learned B, no fixed-point term, seq 1024, batch 6, ramp 1000 then flat 1e-4,
ternary backbone), one factor each. Runner `arc/run_recipe.sh`, commit pinned in
`arc/RECIPE_COMMIT`; per arm a 12-step smoke, the draw with the sustained tripwire, then the
readouts. Order as listed.

| arm | config | the one change | steps | readouts |
| --- | --- | --- | --- | --- |
| `tul-norm-match` | `tul_norm_match` | the paid TUL loop on (`tul.activate_at 0.0`), norm_match | 5,000 | sweeps at 2,500 / 5,000; anatomy + init probe at 5,000 (the plain forward of the paid-loop model, bit-identical to plain by §6b) |
| `tul-absmean` | `tul_absmean` | the same with `ternary_scale_mode: symmetric` (the control) | 5,000 | the same |
| `horizon-ternary-25k` (resumed) | `notul_horizon_ternary_25k` + `training.resume=<step_10000.pt> training.steps=20000` | the absmean plain run continued from its step-10,000 checkpoint (full resume: model, optimizer, RNG, data position, wandb id) to 20,000 | 10,000 more | sweeps at 15,000 / 20,000; anatomy + init probe at 20,000 (5k and 10k already filed) |
| `norm-match-20k` | `notul_norm_match_20k` | `scale-norm-match` for 20,000 steps, checkpoints every 5,000 | 20,000 | sweeps at 5,000 / 10,000 / 15,000 / 20,000; anatomy + init probe at 20,000 |

Sweeps: `core_depth_sweep.py`, 480 rows, depths 0,1,2,3,6,9,12,16, per-token files (TUL
arms pack the stream with the trainer's cut; the token index pairs them with the plain
arms). Anatomy `--rows 3 --depth 8`; init probe `--rows 96`. Results to
`lab/experiments/results/2026-09-09-norm-match-recipe-reads/`; scored with the shared
readers against `scale-norm-match` and `parcae-entry` (5k) and the horizon run's own 5k/10k
readouts. Loop CONTRIBUTION is the reading; CE at a step is a horizon reading, and the
20k pair is the first matched-horizon comparison of the two rules.

`precision-bf16-all` (the ceiling arm of the strength panel) is dropped on Wolfe's word.

**Amendment 2026-09-09 20:05 (during the first arm's readout).** The sweep's depth forcing
went through the slot knobs, which the paid loop ignores, so `tul-norm-match`'s first
sweeps read a flat curve (4.1640 at every depth). Fixed in `core_depth_sweep.py`
(commit `d5e6d37`: a paid-loop model is forced through `model.cfg.mean_depth` like the
plain one); the runner's worktree moved to that commit before the control arm's readout,
and `tul-norm-match` was re-swept at 2,500 / 5,000 beside the running control arm (eval
only). Predictions untouched. The `paid_loop` flag is recorded in every sweep header.

**Amendment 2026-09-09 20:40 (Wolfe's correction).** The two "TUL" arms of this file are
the PAID loop (`tokens_through_core: true`: every position loops), which Wolfe identified
as a hallucinated variant: "that is a hallucinated TUL you made in a previous session."
The real TUL is the slot loop. `tul-norm-match` completed and is kept as a SIDE RESULT
(its readouts are on disk; P-rec-b/c score against it as written); `tul-absmean` was
killed at step ~800 and is not re-run. The paid-loop pair is replaced by the slot-loop
panel, `2026-09-09-arc-slot-loop-norm-match.md`, which runs FIRST; the horizon resume and
`norm-match-20k` (P-rec-d/e) stay on this file and run after it on the same runner
(`arc/run_slotloop.sh`). Predictions untouched.

## Predictions (frozen)

- **P-rec-a (survival).** HEALTHY to the end: tul-norm-match **75 %** (the paid loop on
  a 1.5× stronger map under the ramp), tul-absmean **85 %**, the horizon resume **90 %**,
  norm-match-20k **80 %**.
- **P-rec-b (the shipped forward carries the rule).** tul-norm-match K1−K6 > 0.12 and
  K3−K6 > 0.008 with the CI above 0: **60 %**. tul-norm-match core MLP branch out/in at
  iteration 6 above 0.6: **70 %**. tul-absmean K1−K6 < 0.06: **80 %**.
- **P-rec-c (the paid loop's own earning under norm_match).** TUL minus plain at 5,000,
  token-paired, both norm_match: within ±0.03: **50 %**; TUL better by more than 0.03:
  **25 %**; worse by more than 0.03: **25 %**. Under absmean the same difference within
  ±0.03: **60 %**.
- **P-rec-d (the horizon).** norm-match-20k minus the absmean horizon run at 20,000,
  token-paired at depth 6: smaller than the +0.08 read at 5,000: **65 %**; within ±0.02:
  **30 %**; norm_match ahead (CI high below 0): **15 %**.
- **P-rec-e (contribution over training).** norm-match-20k K1−K6 at 20,000 within 0.05 of
  its 5,000 value (0.185): **55 %**; below 0.10 (decay): **20 %**; above 0.23 (growth):
  **15 %**. K3−K6 at 20,000 above 0.008: **60 %**. The absmean horizon run's K1−K6 at
  20,000 within 0.01 of its 10,000 value (0.038): **80 %**.
- **P-rec-f (cost).** TUL arms within 1.4× the plain arm's wall clock: **80 %**; peaks under
  16 GB: **85 %**.

## Binding

- P-rec-b TRUE ⇒ the rule is confirmed on the shipped forward; no further plain-model
  ternary arms. P-rec-b FALSE with the plain arm at 0.185 ⇒ the slot apparatus interferes
  with the core's map under norm_match; the next read is the TUL arm's anatomy against the
  plain arm's, block by block.
- P-rec-d: the gap smaller at 20k than at 5k ⇒ H-rec-2 holds and the 0.08 stays filed as a
  price paid early; the gap unchanged or larger ⇒ H-rec-2′ and the rule's justification is
  the contribution alone, which Wolfe decides on. norm_match ahead at 20k ⇒ the rule wins
  on loss too; record it and stop arguing about the 5k reading.
- P-rec-e decay ⇒ the map's strength is transient; the next lever is holding the branch
  ratio over training (a norm-match target on the branch, not the weight).
- P-rec-a FALSE on an arm ⇒ its trip step and probe go to the divergence README; a
  norm_match arm that trips re-runs ONCE with the gain ramped in over the first 1,000 steps.
- NO run beyond 20,000 steps from this experiment.

## Not verified before launch

The paid-loop anatomy path (`core_anatomy.py` / `core_init_probe.py` now build the TUL
runtime and read the plain forward of a paid-loop model; tested only by the queue's own
readout). The faithful resume of the horizon run across a `training.steps` change (the
trainer's resume restores model, optimizer, scaler, RNG and fast-forwards the data
stream; the run length does not enter the LR schedule, which is flat, nor AdEMAMix's
pinned horizons). Whether the resumed run's wandb history stitches (the sidecar id is
reused).

## Results

Filed 2026-09-10 16:05. Four arms as amended: `tul-norm-match` (the paid loop, kept as a
side result), `tul-absmean` (killed at ~800 on Wolfe's word, not scored), `norm-match-20k`
(20,000 steps, 06:57 → 10:33, HEALTHY, tripwire max 55 at 757, peak 10.17 GB, final
val_loss 3.5037), `horizon-resume` (the absmean run resumed from step 10,000 to 20,000,
13:49 → 15:44, HEALTHY, tripwire max 27.8 at 10,001, final val_loss 3.4575; the resume
restored model, optimizer, RNG and data position and the wandb id was reused).

The 20k pair, 480 rows, the SAME 491,520 token positions (`tok_index` identical):

| step | norm_match K1−K6 | norm_match K3−K6 | absmean K1−K6 | absmean K3−K6 | norm_match − absmean CE@6, token-paired |
| --- | --- | --- | --- | --- | --- |
| 5,000 | +0.1363 | +0.0107 | (10k: +0.0383) | | +0.08 (the strength panel's read) |
| 10,000 | +0.1562 | +0.0130 | +0.0383 [+0.0372, +0.0395] | | |
| 15,000 | +0.1665 [+0.1638, +0.1690] | +0.0146 | +0.0503 [+0.0491, +0.0515] | +0.0038 | −0.0036 [−0.0063, −0.0009] |
| 20,000 | +0.1702 [+0.1675, +0.1729] | +0.0156 [+0.0150, +0.0163] | +0.0529 [+0.0517, +0.0542] | +0.0042 | **−0.0100 [−0.0125, −0.0073]** |

At depth 1 the norm_match model is 0.107 WORSE than absmean at 20k (paired CI [+0.104,
+0.111]) and at depth 6 it is 0.010 BETTER: the whole gap and more is the loop. norm-match-20k
anatomy at 20k: consecutive movement 0.60 / 0.32 / 0.19 / 0.11 / 0.07 / 0.04 / 0.03, MLP
out/in at the last iteration 1.50–1.85, attention 0.36–0.78; init probe: prelude, noise,
zero and small-noise entries all land on 3.413 by pass 6 (entry-independent fixed point).
horizon-resume anatomy at 20k: movement 0.40 / 0.20 / 0.10 / 0.06 / 0.04 / 0.03 / 0.03, MLP
1.10–1.56.

The paid-loop side result (`tul-norm-match`, 5,000 steps, 18:40 → 19:54 = 74 min vs the
plain arm's 53.5, 1.38x; peak 11.48 GB; 7,779 tok/s vs 9,799): tokens K1−K6 +0.0852 at
2,500, **+0.1200 [+0.1177, +0.1223]** at 5,000, K3−K6 +0.0116 [+0.0110, +0.0122]; core MLP
out/in at the last iteration 0.76–0.93; paid minus plain (`scale-norm-match`) at depth 6,
token-paired on 491,520 positions: **+0.1252 [+0.1222, +0.1278]** (the paid loop is worse).
Artifacts: `lab/experiments/results/2026-09-09-norm-match-recipe-reads/`; npz under
`ignored/experiment-artifacts/2026-09-09-norm-match-recipe-reads/`.

## Verdict

P-rec-a TRUE on the three arms that ran (tul-absmean not run to completion, unscored).
P-rec-b: K1−K6 0.1200 with CI [0.1177, 0.1223] is NOT above 0.12 (FALSE by the letter),
K3−K6 > 0.008 TRUE, MLP branch > 0.6 TRUE; the absmean clause unscored. P-rec-c FALSE on the
50 % clause: the paid loop is worse than plain by 0.125, the 25 % clause. P-rec-d: the 65 %
clause TRUE (−0.010 < +0.08) and the 15 % clause TRUE (norm_match AHEAD, CI high below 0);
the 30 % clause (within ±0.02) also TRUE by magnitude. P-rec-e: norm_match K1−K6 at 20k
within 0.05 of 0.185 TRUE (0.170); K3−K6 > 0.008 TRUE (0.0156); the absmean run within 0.01
of its 10k value FALSE (0.038 → 0.053, it grew). P-rec-f TRUE (1.38x, 11.5 GB).

Filed under failures because two predictions missed (P-rec-c's central clause; P-rec-e's
absmean clause) and one is false by the letter (P-rec-b's first bar). The reading that
matters is positive and was the 15 % branch: H-rec-2 holds and more, the norm_match rule's
0.08 deficit at 5k becomes a 0.010 lead at 20k, and its loop contribution grows over
training (0.136 → 0.170) while the absmean loop's grows too but three times smaller (0.038
→ 0.053). Binding: the rule wins on loss too at 20k; record it and stop arguing about the
5k reading; no further plain-model ternary arms.

## Updated hypothesis

A stronger map costs early and pays later: the loop share grows with training under both
rules, the norm_match loop is three times the absmean loop's, and by 20k that share more
than covers the depth-1 deficit. The paid loop is a side result on Wolfe's 2026-09-09 call
(a hallucinated TUL); its numbers stay on disk. The slot loop under the same rule reads flat
on every arm to date (`failures/2026-09-10-arc-slot-mnext-*.md`), so the rule is confirmed
on the plain loop and is not what the slot loop is missing.
