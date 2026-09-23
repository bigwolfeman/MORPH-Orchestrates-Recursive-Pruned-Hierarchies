# TUL run history: every positive loop-contribution reading (IMPORTANT)

Saved 2026-09-22 at Wolfe's direction under this exact name so it can be found again.
This is the ledger of every run on which a loop (plain, paid, slot, fan, chain, LCTUL)
read a POSITIVE contribution, on any instrument, across the 2026-08 and 2026-09 campaigns.
It is the companion of the negative-framed synthesis
[`.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md`](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md).
Five load-bearing numbers were checked by hand against their source lines on 2026-09-22
(plain vs d1 twin +0.0674, plain reach1 +0.0554, coretok aux +0.0102, strict norecur twin
+0.0020, oly-spandec-strict +0.0111); the rest are transcribed from the filings by the
reading agents. This file is a record: append dated corrections, do not rewrite rows.

Wolfe's 2026-09-22 calls on reading it: math and Sudoku were red herrings, ignore them;
the horizon result (item 1 of "never built on") and a token-like read of the slot state
are the two directions to build on.


Compiled 2026-09-22 at HEAD a558005 by Positives-opus. Read-only. Every number is copied
from a committed file. I did not recompute any number from wandb or raw JSON. I spot-checked
twelve of the load-bearing numbers against their source lines (marked "checked" below). The
rest are transcribed by four reading agents that read every file in
`lab/experiments/successes/` (43), `lab/experiments/failures/` (134), `lab/experiments/mixed/`,
all 28 `lab/experiments/results/*/README.md`, `lab/toy_slot_loop/WRITEUP.md`, the divergence
docs, the architecture notes and eight vlt threads.

Paths are relative to `lab/experiments/` unless they start with `lab/`, `.agents/` or `vlt:`.

Terms used here:

- **K1-K6** is token CE at forced loop depth 1 minus token CE at forced depth 6, same rows,
  paired bootstrap CI. Positive means later passes pay. K3-K6 is the same past pass 3.
- **Forecast K1-K6** is the same difference on the slot's forecast head (`mux_local`), not on tokens.
- **Depth-1 twin** is the same config trained at depth 1. The paired CE gap to it is a
  training-depth reading, not an eval-depth reading.
- **Status.** *standing*: no later file retracts it. *retracted*: a later file shows it was
  an artefact. *suspect*: the file or a later file names a mechanism that is not loop work
  (dependence without value, relay tax, genericity, pre-onset instability, width). A suspect
  positive can still be a real signal about WHERE the loop acts.
- Slot-loop floor: twelve slot arms sit in [-0.0001, +0.0033]; the campaign calls it about +0.002.

## 1. The table

### 1a. Plain loop (tokens run through the core, no slots)

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| scale-norm-match (shipped rule) | 09-09 | K1-K6 / K3-K6 | +0.1849 [+0.1816, +0.1883] / +0.0139 [+0.0131, +0.0146] | Parcae noise entry, norm_match ternary, Poisson mean 6, 5k, OWT seq 1024 | absmean base +0.0332; bf16-core +0.1682 | standing. CE@6 is +0.0817 worse than absmean at 5k (horizon reading) | successes/2026-09-09-arc-per-pass-strength.md |
| norm-match-20k (top rung) | 09-09 to 09-19 | K1-K6 by step | 0.1363 (5k), 0.1562 (10k), 0.1665 (15k), 0.1702 [+0.1672, +0.1729] (20k); K3-K6 +0.0156 at 20k | same recipe, 20k | absmean same recipe at 20k +0.0529 | standing (checked) | failures/2026-09-09-arc-norm-match-recipe-reads.md; failures/2026-09-13-arc-depth-ladder-ship.md |
| norm-match-20k vs its depth-1 twin | 09-19 | paired CE, d1@20k minus top@20k | **+0.0674 [+0.0647, +0.0701]** (was about 0.004 at 5k) | matched steps, 20k, identical rows | d1 (`notul_norm_match_20k_d1`) | standing at matched steps (checked). At matched wall clock it loses: d1@20k beats top@10k by 0.1728 | failures/2026-09-13-arc-depth-ladder-ship.md |
| depth ladder, sampled family | 09-19 | own K1-K6 at 20k | mean 2 +0.0018, mean 3 clamped +0.0387, Poisson 3 +0.0478, mean 4 +0.0705, mean 6 +0.1702 | same recipe, 20k | d1 -0.1417 | standing as a depth-axis dial (checked). Final CE flat inside 0.024 seed spread | failures/2026-09-13-arc-depth-ladder-ship.md |
| d6fixed / d3fixed | 09-19 | CE at own depth, K1-K6 | d6fixed CE@6 3.4246 vs top 3.4516; d3fixed -0.0069 [-0.0094, -0.0044] vs top at depth 3 | `depth_fixed`, 20k | top rung | suspect by standing call: fixed depth is a tied-weight deep net, never ranked against the sampled loop; d3fixed seed twin is +0.0168 | failures/2026-09-13-arc-depth-ladder-ship.md |
| E6 notul_deep16 | 09-07 | K3-K6 / K1-K3 / K6-K12 | +0.277 [+0.271, +0.284] / +0.541 / +0.0405 | mean 16 max 24, bptt 8, ramp, 5k | mean-6 K3-K6 0.0016 | suspect: model is 0.104 [+0.101, +0.107] WORSE than mean 6 at its own depth. Dependence, not value | successes/2026-09-07-arc-e6-deep-recurrence-draw.md |
| plain loop under span reach 1 | 09-21 | K1-K6 | **+0.0554** | plain model, attention reaches only the previous span, norm_match, 5k | reach 0 +0.0325, reach all +0.0378 | standing, unscored observation (checked) | failures/2026-09-21-span-reach-split.md |
| budget-web-full / -span | 09-11 | K1-K6 | +0.0279 [+0.0269, +0.0290] / +0.0301 [+0.0288, +0.0313] | plain Parcae core, prelude entry, 5k | none | standing, reference | failures/2026-09-11-arc-span-budget.md |
| plain-panel-norm-match (prelude entry) | 09-10 | K1-K6 / K3-K6 | +0.033 / +0.0055 | prelude entry, fixed-point 1.0, 5k | noise entry 0.185 | standing, the prelude-entry reference | failures/2026-09-09-arc-slot-loop-norm-match.md |
| same checkpoint entered from zero | 09-10 | depth 1 to 6 CE drop | 0.47 | eval-time entry change only | prelude entry | suspect: ends 0.2 worse than the prelude entry. Dependence | failures/2026-09-09-arc-slot-loop-norm-match.md |
| depthcand-dense-core (bf16 core) | 09-09 | K1-K6 / K3-K6; CE vs depth-1 | +0.168 / +0.0119; beats depth-1 by 0.0336 | Parcae entry, 5k | parcae-entry ternary 0.033 | standing as a diagnostic; Wolfe forbids dense-then-ternary in production | failures/2026-09-09-arc-e20-loop-depth-candidates.md |
| density-half / density-quarter | 09-09 | K1-K6 / K3-K6 | +0.043 / +0.0028; +0.074 / +0.0054 | pruned core, 5k | dense +0.033 | suspect: "dependence without computation", carrier rank 8.5 and 4.4 | failures/2026-09-09-arc-density-panel.md |
| honest rebuild BG0C0 / notul-20k | 08-31 | K1-K6 / K3-K6 | +0.220 / +0.017; 20k +0.207 / +0.016 | carry=none (leak fixed), GLA off, cap off, flat LR | notul-l2nc 0.120 | standing but pre-ramp, pre-norm_match; saturates by K4 | successes/2026-08-31-loop-killer-bisect.md; successes/2026-08-31-tul-vs-notul-20k.md |
| notul under 1000-step ramp | 09-07 | K1-K6 | +0.0367 [+0.0356, +0.0379] (20k +0.0414) | absmean, ramp | none | standing; the ramp and absmean cut the curve | successes/2026-09-07-arc-e11-fixed-point-ramped.md; failures/2026-09-04-arc-e0-where-depth-earns.md |
| E8 notul_mtp4 | 09-07 | next-token K1-K6; t+2/t+3/t+4 K1-K6 | +0.0526; +0.0374 / +0.0238 / +0.0171 | 4 MTP heads, mean 6 | none | suspect: next-token CE 0.345 worse | failures/2026-09-07-arc-e8-multi-token-coda.md |
| oly-notul-nm (Olympiad) | 09-12 | K1-K6 | +0.0647 [+0.0602, +0.0691] | plain loop, norm_match, 6k, Olympiad holdout_clean | absmean +0.0238 | standing (checked) | failures/2026-09-12-arc-math-under-norm-match.md |
| plain loop attractor | 09-14 | AA(noise) / AA(swap) | 0.9998 / 0.9777 | norm_match 20k run at step 5000 | strict slot loop 0.82 / 0.86 | standing: the earning loop is the attractor | failures/2026-09-14-arc-loop-diagnostics.md |
| l2cap | 08-29 | K1-K6 | +0.2328 | full BPTT + sigma cap, retention_carry on | flat arms | **retracted**: carry-off flips it to -1.1198 | successes/2026-08-31-carry-leak-audit.md |

### 1b. Paid loop (tokens and slots through the core)

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| A2 | 09-01 | K1-K6 | **+0.1685** at 5k (0.119 at 2.5k) | tokens_through_core, no warmup, absmean, 5k | free-ride slot arms <= 0.0113; K6 CE beats R0 by 0.298 | standing; leak probe 0 % in two modes; 2 of 4 paid draws detonated | successes/2026-09-01-a2-paid-loop.md; successes/2026-09-02-a2-future-leak-probe.md |
| tul-a2-20k-wu | 09-02 | K1-K6 by step | 0.041 (2.5k) rising to **0.104** (20k, 480 rows) | paid loop + 1000 ramp, 20k | notul-20k-wu flat 0.041 | standing (checked); still 0.022 behind plain on CE at 20k, gap closing 0.132 to 0.012-0.022 | failures/2026-09-02-warmup-20k-pair.md |
| tul-norm-match (paid) | 09-09 | K1-K6 / K3-K6 | +0.1200 / +0.0116 | norm_match, 5k | none | suspect: 0.1252 worse than plain on CE at 5k | failures/2026-09-09-arc-norm-match-recipe-reads.md |

### 1c. Slot loop, training-depth and depth-draw effects

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| slot-spandec-mask vs depth-1 twin | 09-12 | paired token CE, arm@6 minus norecur@1 | **-0.0235 [-0.0263, -0.0209]**, all at offsets 8+, nil at offset 0 | bypass (mask) geometry, span decoder, Poisson 6, 5k | slot-spandec-norecur (trained at depth 1) | suspect as a general law: the STRICT twin reads only +0.0020 [-0.0006, +0.0044] (checked). Memory entry still says the strict twin is "queued" | failures/2026-09-12-arc-latent-z-gradient.md |
| E13 mask arm, mean 12 | 09-07 | token K1-K6; forecast K6-K12 | +0.0489 [+0.0472, +0.0507]; +0.0172 (clears the think bar) | mask, Poisson 12, full BPTT 16, absmean | mask at mean 6: +0.0209 | suspect: same final CE as mean 6 (-0.0009). The loop takes the prelude's share | failures/2026-09-07-arc-e13-m12-panel.md |
| E4 mask under constraint | 09-04 | token / forecast K1-K6 | +0.0209 / +0.187 at 5k | Y2 constraint + tg_restrict mask, absmean | Y2 unmasked +0.0003 / +0.0135 | suspect: 0.13 CE tax; gone under norm_match (+0.0009) | successes/2026-09-04-arc-e4-mask-under-constraint.md; failures/2026-09-10-arc-slot-mux-mask-norm-match.md |
| E14 g102 / g102-rn | 09-07 | token K1-K6 at 2500 | +0.0425 [+0.0408, +0.0440] / +0.0601 [+0.0578, +0.0623] | gain hinge 1.02, mean 12 mask | hinge 0.90: +0.0326 | suspect: both detonated later; K3-K6 flat 0.0007 | failures/2026-09-07-arc-e14-expansive-dial.md |
| E2 iteration conditioning | 09-04 | forecast K3-K6 | +0.0077 [+0.0068, +0.0087] | per-iteration AdaLN | Y2 | **retracted**: pre-onset, the stable rerun reads +0.0001 | failures/2026-09-04-arc-e2-iteration-conditioning.md |
| E7 mask deep 16 / X1 / Olympiad mask | 09-04 to 09-08 | token or own K-curves | +0.0263; +0.4185; +0.406 | various | various | **retracted**: pre-onset or a broken depth-1 hole | failures/2026-09-07-arc-e7-block-loop.md; failures/2026-09-04-tul-clip-through-time.md; failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md |
| slot-unpack fixed depth 6 | 09-10 | token K1-K6 | +0.1168 [+0.1138, +0.1198] | `slot_depth_fixed` 6 | unpack +0.0006 | suspect: CE@6 +0.052 worse, a bowl | failures/2026-09-10-arc-slot-map-levers.md |

### 1d. Slot loop, content moving through passes (strict geometry and the chain)

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| slot-spandec-strict-prev-reach1 | 09-12 | K1-K6 / K3-K6 | +0.0163 [+0.0153, +0.0173] / +0.0042 | coda reads cell j-1 only, loop reach 1, 5k | strict-prev; CE parity +0.0023 | suspect (restriction); better than its own depth-1 read at every offset (-0.014 to -0.028) | failures/2026-09-12-arc-strict-geometry.md |
| slot-spandec-strict-prev-reach2 | 09-12 | K1-K6 / K3-K6; paired CE | +0.0127 / +0.0028; **-0.0086 [-0.0109, -0.0063] vs prev** | loop reach 2 | prev, prev-reach1 (-0.0109) | suspect (restriction) but the one restriction arm where reach turned into CE value; parity with strict (-0.0004) (checked) | failures/2026-09-12-arc-strict-geometry.md |
| hop staircase on prev-reach1 | 09-19 | per-hop K1-K6 | h3 +0.0640 [+0.0610, +0.0671] at pass 2, h4 +0.0535 at pass 3, h5 +0.0288 by pass 6; replication h3 +0.0665 | 480 rows, then a disjoint 480 | strict null h3 +0.0025; reach1 control flat | standing as a mechanism reading; h2 and h6 negative (dilution) | failures/2026-09-18-hop-distance-earning.md; failures/2026-09-19-hop-distance-plateau-and-dilution.md |
| planted copy, prev-reach1 | 09-19 | benefit by depth | exactly 0.000 before pass g-1; two-token pair g3 +0.083 at pass 2, g4 +0.046 at pass 3 | planted rare ids | g0 control | standing: an exact induction test | same two files |
| own-span content under cut-after-1 | 09-19 | planted g=1 benefit d1 to d6 | +0.003, +0.090, +0.170 (growing) | re-supplied by injection each pass | uncut g1 decays 0.181 to 0.137 | standing: passes REFINE what the injection re-supplies | failures/2026-09-19-hop-distance-plateau-and-dilution.md |
| carry-sum / carry-gate | 09-20 | whole K1-K6; per-hop | +0.0404 (h1 +0.1033, h2 +0.0846) / +0.0086 (h3 +0.0359) | prev-reach1 + loop_carry | ruler +0.0163 | suspect: a worse depth 1, far hops go negative, state unbounded | failures/2026-09-19-loop-carry-prev-reach1.md |
| LXTUL-R Step 1b | 09-22 | K1-K6 / K3-K6; planted arrival | **+0.0261 [+0.0251, +0.0273] / +0.0057**; natural tokens +0.0255; g3 0 at d1, +0.048 at d2 | fan4-all, one slot per pass chain, leak fixed | fp0 +0.0102 | suspect: a tax refund, 0.0265 behind fp0 at depth 6 and 0.0527 at depth 1. Wolfe: "a mixed result" | failures/2026-09-22-lxtul-r-step1b.md |
| Step 1 leaky / C1 persist / C2 hist1 | 09-22 | K1-K6 | +0.0202 / +0.0234 / +0.0204 | chain variants | Step 1b | suspect (relay); lane closed | failures/2026-09-22-lxtul-r-step1.md; failures/2026-09-22-lxtul-r-step2-panel.md |
| trajectory prefix | 09-13 | K1-K6 | +0.0342 | six trajectory cells | trajrep +0.0019 | **retracted**: forced-depth pad cells | failures/2026-09-13-arc-trajectory-prefix.md |

### 1e. Slot loop, per-pass targets, forecasts and the exit

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| slot-mnext-staged | 09-10 | forecast K1-K6; own-span ladder; pass-6 grad cosine | +0.0670 [+0.0628, +0.0714]; own CE 6.25 to 3.36 at pass 3; -0.22 | own span at pass 3, next at exit, norm_match | ruler forecast +0.0067 | suspect: depth-1 got worse, exit forecast equals the ruler (6.7698 vs 6.7724) | successes/2026-09-10-arc-slot-mnext-staged.md |
| staged-20k | 09-11 | forecast K3-K6 (next) | +1.850 (5k) to +2.080 (20k); tokens +0.0018 to +0.0025 | 20k | plain 20k | suspect: 0.16 behind plain, flat 10k to 20k; z-opt worth shrinks | successes/2026-09-10-arc-slot-mnext-staged-20k.md |
| staged-fullread / staged-mask | 09-10 | exit forecast; z-opt exit minus entry | 0.018 better than ruler; +0.0235 (mask) | per-stream readout / mask | ruler +0.0015 | suspect: token K-curve flat; mask costs 0.108 | failures/2026-09-10-arc-slot-mnext-staged-fullread.md; failures/2026-09-10-arc-slot-mnext-staged-mask.md |
| slot-mnext-gradpass | 09-11 | forecast K1-K6; exit forecast; own ladder | +0.0236 [+0.0207, +0.0269]; 0.017 better than ruler; falls 0.366 at pass 1 | loss gradient injected before each pass | ruler | standing as the best exit on the ruler family, one pass deep | failures/2026-09-10-arc-slot-mnext-gradpass.md |
| slot-mux-fixed-point-off | 09-11 | forecast K1-K6 | +0.0145 [+0.0127, +0.0166] | term off | ruler +0.0067 | suspect: K3-K6 0.0000, state expands 2.2x | failures/2026-09-10-arc-slot-mux-fixed-point-off.md |
| Y1 / Y2 | 09-04 | forecast K1-K6 | +0.0128 / +0.0135 | constrained slot loop | none | suspect: "stable and empty", K3-K6 <= 0.0002 | successes/2026-09-04-tul-forward-levers.md |
| R3 M-own | 09-03 | own-loss K1-K6 | +0.0371 [+0.0355, +0.0388] | own-span MUX target | none | suspect: token +0.0002, front-loaded | failures/2026-09-03-tul-think-once-panel.md |
| tul_v1a2b MUX detached | 08-27 | loop worth on ce_main | +0.0058 to +0.0107 (4 seeds), p 0.0286 | detached MUX head, beta 0.1 | tul_a1 -0.0002 to +0.0042 | standing, small, post-hoc | failures/2026-08-27-warmup-sigreg-ntpdrop.md |
| TG2 | 08-27 | loop worth on ce_main | +0.0237 to +0.0378 | TG restriction | tul_a1 | suspect: seed-inconsistent | failures/2026-08-27-tg-restriction.md |
| grad_pass critic / egrad | 09-12 | critic gap; probe AUC rise | +0.00145; AUC 0.611 to 0.638 at pass 1 | strict | shuffle null | standing, tiny, one pass | failures/2026-09-12-arc-core-token-and-critic.md; failures/2026-09-12-arc-latent-z-gradient.md |

### 1f. Slot-trained core read on tokens, the reader, the fan, LCTUL

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| coretok, aux token path through the core | 09-13 | aux K1-K6 / K3-K6 | **+0.0102 [+0.0093, +0.0110]** / +0.0012 | same six blocks, tokens routed through them at eval, strict, 5k | shipped slot path +0.0005 | standing (checked); passes 1 to 2 give 0.0071 | failures/2026-09-13-arc-loop-reads-tokens.md |
| fan4-all | 09-20 | oracle value recovered by the per-token read; paired CE | 0.056 of 0.099; -0.0342 [-0.0367, -0.0316] vs pk4 | four streams, write all, WTA | select-gate 0.014 of 0.113 | standing as a reader result; token K1-K6 +0.0049 | successes/2026-09-20-lxtul-fan4-all.md |
| fan4-all-fp0 / trig / epi / select | 09-20 to 09-21 | K1-K6 | +0.0102 / +0.0089 / +0.0061 / +0.0076 | fan variants | fan4-all +0.0049 | suspect: the extra is pass 1's; K3-K6 <= 0.0016 | failures/2026-09-21-lxtul-fan4-all-fp0.md; -trig.md; failures/2026-09-20-lxtul-fan4-epi.md; -select.md |
| fan4-epivol | 09-20 | oracle minus mixed CE; stream rank | 0.0182; 2.823 of 3 at pass 1 | volume term | fan4 0.008 | standing; Wolfe: "the thing to work on" | failures/2026-09-20-lxtul-fan4-epivol.md |
| spandec-chain-mask | 09-11 | long-range worth bin 8-15 | 0.053 to 0.061, disjoint CIs | W_chain hop recurrence | spandec-mask | standing, small, +0.0068 CE cost | successes/2026-09-11-arc-span-decoder.md |
| code-target-uf (adapted reader) | 09-17 | cell worth | -4.6 nats (frozen reader) to +0.177 (adapted); K1-K6 +0.0047 | unfrozen coda 10k | frozen coda | standing as a reader result | successes/2026-09-17-lctul-target-unfreeze.md |
| code_target arm A / denoise | 09-17 / 09-21 | K1-K6 | +0.0323 / +0.5575 | frozen coda | none | **retracted**: genericity through a frozen coda | successes/2026-09-17-lctul-target-unfreeze.md; failures/2026-09-21-lxtul-loop-denoise.md |
| code_grade per-pass cosine | 09-17 | cos l0 / l1 / l6 | 0.4012 / 0.5235 / 0.5310 | graded target | pass 1 | suspect: met in one pass | failures/2026-09-17-lctul-graded-target.md |
| lejepa / dplan / thinker-only | 09-15 to 09-16 | sampler depth curves | k1 4.585 to k2 4.486; k4-k1 -0.0136; k16-k1 -1.88 | LCTUL sampler | k1 | suspect: sampler depth on a trusting coda, not tokens | failures/2026-09-16-tul-code-lejepa.md; -lctul-dplan-semantic.md; failures/2026-09-15-tul-code-thinker-only.md |

### 1g. Other corpora, toys, external and instruments

| arm | date | instrument | magnitude [CI] | conditions | control | status | source |
|---|---|---|---|---|---|---|---|
| oly-spandec-strict | 09-12 | K1-K6 / K3-K6 | **+0.0111 [+0.0085, +0.0136]** / +0.0024 | strict slot loop, Olympiad | all web slot arms <= 0.005 | standing; gap to plain widened (+0.216) | failures/2026-09-12-arc-math-under-norm-match.md |
| sud-spandec-strict | 09-12 | K1-K6 | +0.0049 [+0.0047, +0.0052] | Sudoku | plain sud flat +0.0011 | standing, small; the only corpus where the slot loop beats plain on K | same file |
| toy slot loop, staged | 09-10 | chain solved | 5/5 seeds (exit-only 2/5, progressive 0/5); fixed-point 1.0 5/5 | strict toy, scan task | dense 1/5 | standing (checked) | lab/toy_slot_loop/WRITEUP.md |
| toy eliminate6 | 09-19 | probe accuracy at hop 4; solved cells | trained 0.999 vs random 0.668; d192 8/15 vs d96 2/15; fixed-point 2/5 vs 0/5 | six hops | random init | standing (toy) | successes/2026-09-19-toy-eliminate6-hop-distance.md |
| toy eliminate, fixed-point | 09-18 | candidate entropy | 0.989 of 1.0986 | two hops | exit 0.352 | standing (toy) | failures/2026-09-18-toy-eliminate-deferred-commitment.md |
| Huginn-0125 | 09-04 | K3-K6 / K6-K16 / K1-K6 | +0.566 [+0.559, +0.573] / +0.206 / +1.906 | eval only, 480 OWT rows | MORPH 0.041 | standing (external); no depth-1 control, so dependence | failures/2026-09-04-huginn-loop-contribution.md |
| sample oracle, entry noise | 09-19 | gain(16) at sigma 1.0 | +0.0246 frozen / +0.0388 adapted | strict ruler / code-target-uf | deterministic | suspect: fixed-point contraction, reader effect | failures/2026-09-18-sample-oracle-gate.md; failures/2026-09-19-sample-oracle-adapted-reader.md |
| strict slot loop basin map | 09-14 | settling entropy | 1.6 nats at 2 of 6 pairs | entry perturbation grid | flat elsewhere | standing, not followed | results/2026-09-14-loop-diagnostics/README.md |
| fitted z | 09-10 | CE vs loop z | 0.9 to 2.6 nats better | hindsight fit | loop z | **retracted** as a ceiling: used the answer; causal fit +0.25 worse | failures/2026-09-12-arc-latent-z-gradient.md |
| region Shapley, TUL A1 | 08-25 | core Shapley on ce_emit | 0.2274 to 0.3296 | healthy A1 checkpoints | ce_main 0.0007 | standing: the core does work on the one target it is paid on | results/2026-08-25-region-shapley/README.md |

Count: 66 table rows (several rows group two to four readings of one arm family). 6 rows are
retracted (l2cap, E2, the E7/X1/Olympiad-mask group, trajectory prefix, arm A plus denoise, fitted z).
The other 60 are standing or suspect positives.

## 2. The strongest standing positives, ranked

1. **The plain loop earns, and training with the loop beats training at depth 1.** Under the
   noise entry and norm_match, K1-K6 is 0.185 at 5k, and it keeps growing with steps (0.136 to
   0.170 on the 20k run). At 20k the looped model beats its own depth-1-trained twin by 0.0674
   [0.0647, 0.0701] at matched steps. At 5k that gap was about 0.004. The loop's value is
   growing with horizon. Where the work happens: on tokens, at eval AND at train time. The
   caveat: at matched wall clock the depth-1 model wins by 0.173. This is the one loop
   positive that already has a matched-step value reading, not only a dependence reading.
2. **The paid loop earns, and its earning grows with steps.** A2 reads 0.1685 at 5k with no
   warmup, leak-free. Under the ramp it reads 0.041 at 2.5k and 0.104 at 20k, rising at every
   checkpoint. Its CE deficit to plain closed from 0.132 to 0.012-0.022 over 5k to 20k. Where:
   on tokens, through the same core the slots use. This says the slot loop's core CAN earn when
   the token loss reads its output.
3. **Content moves through passes on schedule.** On prev-reach1, content h spans back arrives
   at pass h-1 (h3 +0.064 at pass 2, h4 +0.054 at pass 3), and the planted copy reads exactly
   zero before its arrival pass. The staircase replicates on disjoint rows. Own-span content
   that the injection re-supplies is REFINED across passes (+0.003 to +0.170 under the cut).
   Where: on the slot state, read by the coda. It is a relay, and Step 1b showed the relay
   is a refund of a tax, but the mechanism is proven: the passes carry and refine content.
4. **The slot-trained core earns on tokens.** The same six blocks that give the shipped slot
   path +0.0005 give +0.0102 when tokens run through them. Where: the blocks can compute on
   token states. The slot path, not the core, is where earning is lost.
5. **Math corpora lift the slot loop off the floor.** oly-spandec-strict reads +0.0111
   [0.0085, 0.0136] with K3-K6 +0.0024, equal to the plain model's own K3-K6 on that corpus.
   Sudoku reads +0.0049 while the plain loop is flat there.
6. **prev-reach2 turned reach into CE value** (-0.0086 vs prev) while earning +0.0127. It is
   the one restriction arm whose depth came with a CE gain against its own cut, though only at
   parity with unrestricted strict.
7. **The reader cashes candidates.** fan4-all's per-token read recovers 0.056 of 0.099 nats of
   oracle value. The adapted coda turns a -4.6 nat cell into +0.177. Where: on the reader. These
   are not depth, but they say that when a loop produces distinct candidates, the reader can use them.
8. **Training-depth effect on the slot loop (bypass geometry only).** spandec-mask trained at
   Poisson 6 beats its depth-1-trained twin by 0.0235 at offsets 8+, with an eval K-curve of
   0.0007. Where: at training time only. The strict twin reads +0.0020, CI crossing zero, so
   this is geometry-dependent, not a law.

## 3. What the positives have in common

- **Gradient path to the token loss.** Every positive above 0.03 on tokens has the token CE
  reading the core's output directly: the plain loop, the paid loop, the aux token path. Every
  slot-state positive is small (<= 0.026) or sits on a forecast or own-span head the token loss
  does not read. The campaign synthesis calls this Condition B (consumed). The ledger agrees.
- **No shallower route.** The slot-loop positives all come from geometries that force content
  through the passes: reach 1, reach 2, the mask, the chain. The plain loop under reach 1
  earns more (+0.055) than under reach all (+0.038). Removing the shallow route raises the
  loop's share, on the plain loop too.
- **Entry.** The noise entry earns 0.185. The prelude entry earns 0.033. A state near the fixed
  point leaves the passes nothing to build. On the slot loop, removing the prelude did not help
  (+0.0045), so the entry lesson has not transferred there.
- **Depth draw.** The sampled mean is a dial on the loop's share: 0.002 to 0.170 from mean 2 to
  mean 6, 0.021 to 0.049 at mean 12 on the mask arm. It moves dependence, not final CE.
- **Horizon.** The token-path positives grow with steps (plain 0.136 to 0.170, paid 0.041 to
  0.104, the d1 gap 0.004 to 0.067). No slot-state positive grows with steps (staged-20k flat,
  strict 0.0001 to 0.0014 at 4x).
- **Corpus.** Web text carries the plain-loop positives. Olympiad and Sudoku carry the largest
  clean slot-loop K-curves.
- **Pass 1.** Almost every slot positive lives in passes 1 to 3. The exceptions are the hop
  staircase (arrival at h-1) and the refinement of re-supplied own-span content. Both need a
  job that pass 1 cannot finish because the input has not arrived yet or keeps arriving.
- **Stability.** The earning plain loop is an attractor (AA 0.9998). Several slot "earnings"
  (E2, E7, E14, X1) came with instability and were retracted.

## 4. Positives never built on

- **The plain loop's matched-step win over its depth-1 twin, growing with horizon** (0.004 at
  5k, 0.067 at 20k). No arm extended it to 40k or asked where it crosses the wall-clock line.
  No slot design used "the loop's value appears at long horizon" as its premise.
- **The paid loop's closing CE gap** (0.132 to 0.012-0.022, projected crossing 22-25k). Never
  run past 20k. The paid loop was then cut as "not TUL".
- **The plain loop under reach 1** (+0.055 vs +0.033). A one-line aside in the Step 0 filing.
  No arm asked whether a shallow-route cut on a TOKEN-path loop buys value.
- **The aux token path** (+0.0102, 20x the slot path, same blocks). No slot design gave the
  slot state a token-like job on the same blocks, apart from the paid loop that was cut.
- **oly-spandec-strict +0.0111 and sud-spandec-strict +0.0049.** No later slot arm ran on a
  math corpus. The fan, the chain and LCTUL all ran on web text.
- **prev-reach2's CE value.** The chain went to reach 1, the reach that gave no value.
- **Own-span refinement under re-supply** (+0.003 to +0.170). The carry arms tried re-supply
  and ran unbounded; no arm tried a bounded re-supply of the own span each pass.
- **gradpass second-order form and its tg_restrict twin.** Proposed 2026-09-11, never run.
- **staged at 40k.** Left as Wolfe's call.
- **The basin map's two multistable slot pairs.** Not investigated.
- **The spandec-mask training-depth effect** on the bypass geometry. Only the strict twin
  followed, and it read zero; nobody asked why the bypass geometry showed it.
- **E13 / E14 share movers** (mean 12, hinge 1.02). Recorded, then not combined with a
  geometry that has no shallower route.
- **Toy staged 5/5.** Built on (slot-mnext-staged), but the tree could not supervise every
  non-final pass as the toy did. The exact toy arm was never run.

Built on, for the record: Huginn to E6; the toy fixed-point result to the shipped term; the
hop staircase to the LXTUL-R chain; fan select to fan4-all to the LXTUL-R base; the adapted
reader to the LCTUL EMA target; norm_match to `base.yaml`.

## 5. Memory entries checked against lab files

Confirmed in lab files: training-depth-moves-ce (with a correction below),
mean-depth-is-a-depth-axis-dial, hop-staircase-on-prev-reach1, loop-contracts-entry-noise,
write-all-read-cashes, toy-slot-loop-staged-targets-win, morph-loop-earns-when-paid,
huginn-loop-earns, loop-diagnostics-attractor, morph-iteration-conditioning-earns,
gain-target-moves-loop-share, mean12-draw-changes-share, depth-earning-is-depth-dependence,
set-is-carried-at-random-init, gradpass-one-step-optimiser, prelude-entry-flattens-the-loop,
morph-depth1-control-instrument, l2cap-depth-earning-was-the-leak,
trajectory-prefix-is-width-plus-pad-artefact, morph-tul-plan-is-empty (in
`results/2026-08-25-region-shapley/README.md` and `lab/divergence/takeover-campaign.md`, not in a
filed success or failure), overnight-batch-2026-09-11.

Not confirmed or stale:

- training-depth-moves-ce-eval-depth-does-not says the strict twin is "queued". The strict twin
  ran and reads +0.0020 [-0.0006, +0.0044] (failures/2026-09-12-arc-latent-z-gradient.md). The
  entry's "0.054 behind the looped plain at 5k, 44 vs 14 passes" figure does not appear in the
  E19 or ladder filings; E19 says the depth-1 model is within 0.02 (+0.019 in its favour).
