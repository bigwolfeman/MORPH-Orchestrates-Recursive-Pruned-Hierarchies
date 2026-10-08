# Parcae LXTUL 5k panel: does the slot loop contribute on a clean backbone?

Status: failure (prediction 1's depth half failed: plain K1-K6 +0.046, not >= +0.10; predictions
2-7 held, including the hypothesis, prediction 3). Preregistered 2026-10-06 10:45 CDT, filed 12:20.

## Question

MORPH's LXTUL loop earns K1-K6 +0.0215 / +0.0172 (two seeds, 5k) and trails a plain model
by +0.272 / +0.282 nats. On MORPH, 98-99.8 % of the loop's carrier after pass 1 is one
slot-shared vector that lives in the Hyper-Connection stream differences, and every cut of
it was rebuilt by training. Parcae has one residual stream, plain pre-norm blocks, bf16
weights and Muon. Does the same LXTUL mechanism earn more loop contribution there? And on
this backbone, which parts are needed: MORPH's partial injection, and the LX machinery
(fan, latent selection, span decoder, epivol) at all?

## Hypothesis

The small MORPH loop contribution is partly a MORPH backbone effect (the HC carrier and
ternary core). On a clean backbone the faithful port earns at least as much as MORPH.

## Arms (one seed each, 5000 steps, same rows, same recipe)

| arm | config | what it is |
| --- | --- | --- |
| plain Parcae | `arm=plain` | stock Parcae, every token loops, Poisson(6) total, BPTT 6. Baseline. |
| Parcae-LXTUL | `arm=lxtul` | the faithful port of MORPH lxtul.yaml (lxtul/SPEC.md), Parcae's full-width injection |
| LXTUL, MORPH injection | `arm=lxtul_ctxinj` | as above, injection on channels [512, 832) only (gain floor 0.865, as MORPH) |
| bare slot loop | `arm=bare` | strict loop, 1 cell, no fan / latent selection / epivol / span decoder |

Shape d 1024, 8 heads (kv 8), 4/6/4 layers, ReLU2 4096, vocab 49169 (StarCoder2), batch 6,
MORPH's OWT train stream and packer (seq 1024, prefix_k 4, max_slots 64). Recipe: Parcae's
MuonAdamW, lr 0.008 * sqrt(6/256) = 0.00122, constant to step 2500 then linear to 0 at 5000,
Muon momentum 0.85 -> 0.95 over 300 steps, Muon wd 0.2 decayed to 0, clip 1.0, no LR warmup
(amended below: 300-step linear warmup).
Code: branch `lxtul-testbed`, `lxtul/train.py` at the commit that adds this file.

## Predictions

Rows: the first 480 TUL rows of MORPH's validation stream (OWT validation, 50k docs
skipped). The plain arm scores the same (token, target) pairs with slots removed, so every
comparison is paired per row (MORPH `_stats.paired_bootstrap_ci`, 2000 draws).
"CE" = token CE at eval depth 6. "K1-K6" = CE(depth 1) - CE(depth 6), same model.

1. Plain Parcae CE in [3.70, 4.30]; plain K1-K6 >= +0.10 with the CI above 0.
2. Parcae-LXTUL K1-K6 > 0 with the CI above 0.
3. Parcae-LXTUL K1-K6 >= +0.019 (the MORPH two-seed mean). This is the hypothesis; I give
   it about 35 %.
4. Gap, Parcae-LXTUL CE minus plain CE (paired): in [+0.10, +0.40].
5. LXTUL with MORPH injection: K1-K6 within a factor of 2 of Parcae-LXTUL's (injection mode
   is not the lever).
6. Bare slot loop: K1-K6 <= +0.010 and below Parcae-LXTUL's point estimate.
7. No arm produces a non-finite loss (the trainer raises on one). Training tok/s: Parcae-LXTUL
   and plain Parcae >= 12,250 (MORPH LXTUL measured the same day).

## Method

`for a in plain lxtul lxtul_ctxinj bare; do python -m lxtul.train arm=$a; done`, one job at
a time under the GPU lock. Each run: 5000 steps, val CE at depth 6 on 120 rows every 500
steps, final model state saved to `/home/wolfe/parcae-runs/<arm>-5k/final.pt`, then the
depth sweep at K = 1, 2, 3, 4, 6, 8 on 480 rows (`sweep.json`, `summary.json`). wandb
project `parcae-lxtul`. Readout: the table of CE, K1-K6 with CI, gap to plain with CI,
training tok/s and peak memory per arm.

Method amendment, 2026-10-06 10:49 CDT, before any result was read: the first launch
(10:45) ran Parcae's recipe with NO LR warmup. Wolfe: "you need a short warmup at least".
The plain arm was stopped at step ~525 (only its step-500 val line was seen, 5.42; no
sweep). All four arms now add a linear LR warmup over 300 steps on top of Parcae's
schedule (`training.warmup_steps: 300`). The same launch hit Dynamo's recompile limit of 8
on the shared Parcae block forward at the first val pass, so some block variants ran
eager; `training.recompile_limit: 64` (Parcae's own trainer's value) fixes that. The
stopped run's wandb entry `plain-5k` (failed, step ~525) is not part of the panel.

What each outcome changes:
- 3 holds: the backbone was suppressing the loop; new loop arms run on Parcae.
- 2 holds, 3 fails: Parcae reproduces MORPH's small earning at 1.9x the speed; the limit is in
  the mechanism, and Parcae is the faster place to test mechanism changes.
- 2 fails: the port or the recipe kills the loop; diagnose before any new arm (compare the
  ctx-injection arm, read the gain and the fixed-point term).
- 6 fails (bare earns as much as LXTUL): the LX machinery is not what makes the loop earn on
  this backbone.

## Results

Recorded 2026-10-06 after the panel finished (12:11). Code `bc03f38` for every arm (the
warmup amendment above). One seed each. 480 held-out rows, paired per row; the token counts
per row are checked identical across arms (`lxtul/readout.py`). Artifacts:
`docs/experiments/results/2026-10-06-parcae-lxtul-5k-panel/` (per-arm sweep.json, summary.json,
metrics.jsonl; readout.json). Final weights: `/home/wolfe/parcae-runs/<arm>-5k/final.pt`.
wandb `adew-me/parcae-lxtul`: plain `oytqpu97`, lxtul `n8nejrhm`, ctxinj `2o6tgmio`, bare `npsq3m7q`.

| arm | CE@6 | K1-K6 [95% CI] | gap to plain [95% CI] | train tok/s | peak GiB |
| --- | --- | --- | --- | --- | --- |
| plain Parcae | 3.8303 | +0.0462 [+0.0448, +0.0475] | - | 21,819 | 21.1 |
| Parcae-LXTUL | 4.1242 | +0.0585 [+0.0564, +0.0607] | +0.2939 [+0.2756, +0.3151] | 24,403 | 13.8 |
| LXTUL, MORPH injection | 4.1248 | +0.0703 [+0.0679, +0.0727] | +0.2945 [+0.2758, +0.3163] | 24,516 | 13.8 |
| bare slot loop | 4.0826 | -0.0001 [-0.0001, -0.0000] | +0.2523 [+0.2345, +0.2730] | 37,176 | 11.6 |

CE by depth (1 / 2 / 3 / 4 / 6 / 8):
- plain 3.8765 / 3.8402 / 3.8332 / 3.8312 / 3.8303 / 3.8303
- Parcae-LXTUL 4.1827 / 4.1401 / 4.1252 / 4.1241 / 4.1242 / 4.1242
- MORPH injection 4.1952 / 4.1432 / 4.1266 / 4.1248 / 4.1248 / 4.1251
- bare 4.0826 / 4.0826 / 4.0826 / 4.0826 / 4.0826 / 4.0826

MORPH references (its own 480 rows, which overlap training steps 0-80; different optimizer,
flat LR, ternary, Hyper-Connections): LXTUL 5k K1-K6 +0.0215 / +0.0172, gap +0.272 / +0.282.

No arm raised (no non-finite loss or grad norm). No Dynamo recompile-limit warning after the
amendment.

Prediction by prediction:
1. Plain CE 3.830 in [3.70, 4.30]: holds. Plain K1-K6 >= +0.10: FAILS (+0.046; the plain
   loop's gain is mostly pass 2 and flat after pass 4).
2. Parcae-LXTUL K1-K6 CI above 0: holds.
3. Parcae-LXTUL K1-K6 >= +0.019: holds (+0.0585, 2.7x the MORPH two-seed mean).
4. Gap in [+0.10, +0.40]: holds (+0.294).
5. MORPH injection within 2x of Parcae-LXTUL: holds (1.2x; it earns slightly more).
6. Bare K1-K6 <= +0.010 and below Parcae-LXTUL: holds (-0.0001).
7. No non-finite loss; plain and LXTUL train >= 12,250 tok/s: holds (21.8k, 24.4k).

## Verdict

Failure by the frozen rule (one of seven predictions failed, and it was the baseline's depth,
not the hypothesis). The hypothesis held: on a single-stream Parcae backbone the faithful
LXTUL port earns K1-K6 +0.0585, 2.7x MORPH's, and 3.3x with MORPH's own injection. So
MORPH's injection is not why MORPH earns less. The LX machinery is what makes the loop earn
here: the bare strict loop is flat. Two readings that the numbers do not let me separate:
the loop earning did not buy CE (LXTUL trails bare by 0.042 at depth 6, and the gap to
plain, +0.29, is the size MORPH had), and the comparison to MORPH is confounded by optimizer,
LR schedule (Parcae's cools to 0 at 5000), weights and attention, not by HC alone.

## Updated hypothesis

The slot loop's small earning on MORPH is mostly a MORPH backbone effect. The suspect with a
mechanism is the Hyper-Connection carrier: the JPmHC write puts a zero-sum per-stream pattern
into the stream differences and the orthogonal mixer does not damp it (HC audit, filed in
MORPH `lab/divergence/2026-10-06-hc-jpmhc-audit.md`). Within LXTUL, the fan + latent
selection + span decoder make the loop earn, but at 5k that earning costs CE against the
bare loop instead of closing the gap to plain.

Next experiments (each needs its own prereg):
- Seed twin of Parcae-LXTUL (one seed per arm here).
- MORPH with a single-stream core (HC off in the slot loop only): does MORPH's K1-K6 rise
  toward Parcae's? This is the direct HC test.
- On Parcae, drop one LX part at a time (span decoder; latent selection with 4 cells kept)
  to find which part makes the loop earn, and whether a cheaper one earns without the CE cost.
- 10k on Parcae-LXTUL vs plain: does the gap narrow as MORPH's did not?
