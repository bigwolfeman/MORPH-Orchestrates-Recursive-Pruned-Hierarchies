# The TUL core-takeover campaign — running index

**Read this first.** Every hypothesis tried, its verdict, the number that decided it, and
where the detail lives. One line per idea. If an idea is not here, it has not been tried.

Last updated 2026-08-25. **This is a LAB REPORT** — trial and error, kept in
`lab/` on purpose. Settled claims graduate to `lab/experiments/` and
`.agents/notes/`; this file indexes them AND everything that did not settle.
The instruments it refers to are its neighbours in this directory.

## The problem, in four lines

TUL arm A1 loops the weight-shared 6-block core over ~57 SLOT positions (one per span)
instead of 1024 token positions. Validation CE reaches a minimum at step 500–2000 and then
RISES by +1.3 to +2.4 nats — a turnaround, not a spike. 4 of 4 runs fail at
`ademamix_alpha_cap` 3.5; at 1.0 it is a coin flip. Arm A0 (loop over tokens) and arm A3
(no core) are healthy at the same settings, so the loop over slots is the whole difference.

**It blocks TUL only.** `base.yaml` has `tul.activate_at: never`; nothing shipping is
affected.

## Status right now

Updated 2026-08-25 21:00. Nothing is running; the GPU is idle.

## >>> THE CAMPAIGN HAS PIVOTED. READ THIS BEFORE ANYTHING BELOW. <<<

**The takeover is a GRADIENT FLOW failure, not a stability failure.** Everything in the
hypothesis ledger below was aimed at stabilising the looped core. It is now measured that
there was almost nothing holding those parameters anywhere to begin with.

Measured 2026-08-25, `onset-capture/ROLL_step_1750`, 8 fixed val batches, eager kernels.
Full writeup: [`lab/experiments/results/2026-08-25-region-shapley/README.md`](../../lab/experiments/results/2026-08-25-region-shapley/README.md).

1. **Exact Shapley over prelude / core / coda.** Three players, 8 coalitions, no sampling;
   the values sum to `v(all)` at every checkpoint, which is the arithmetic check.

   | ckpt | core on total loss | **core on `ce_main`** | core on `ce_emit` |
   |---|---:|---:|---:|
   | 1625 healthy | 0.0065 | 0.0015 | 0.2274 |
   | 1750 healthy | 0.0080 | **0.0007** | 0.3296 |
   | 1850 taken over | −0.0015 | −0.0010 | −0.0314 |

   Prelude and coda score 2.5–2.7 throughout. **Shapley, not leave-one-out, is what refutes
   redundancy** — a core the coda covered for would still score high. It scores 0.0007.

2. **The plan is EMPTY, not outcompeted.** Ablating the core by identity removes the LOOP,
   not the SLOT (the slot keeps `E_slot + mean(embed(span))`). Zeroing what
   `prefix_project` writes separates them: the loop is worth 0.0051 on `ce_main`, the WHOLE
   plan 0.0191. Sweeping `token_state_dropout` at eval, paired and seeded, the plan is worth
   0.0191 (p=0) → 0.0344 (p=0.15, shipped) → **0.0699 (p=1.0**, where `ce_main` collapses to
   7.53). Mask every token and the plan still carries 0.07 nats of its span.

3. **Why.** The core's only DIRECT supervision is `ce_emit` — predict ONE token, the next
   span's first, worth 2.6564 nats to that target. Its INDIRECT route through the coda is
   worth 0.0191, so almost no gradient arrives that way. So the plan learns to be a
   one-token predictor and never becomes a span summary. And it loses even that job to the
   free token path: `ce_plast − ce_emit` is −0.2191 at 1750 and −0.4842 at 1850. That is the
   `cf` field in every VAL line and it is negative in every run on disk (−0.103 to −1.047).

   Beside H14 (slot label = 2.8 % of loss weight, ~50 % of the gradient) and >90 % gradient
   share at takeover: **gradient share and value are decoupled by three orders of magnitude.**

### The next focus is gradient flow

The question is no longer "how do we stop the takeover". It is **"how does the looped core
earn gradient in proportion to the value it provides"**. Candidates, none pre-registered:

| lever | one-line? | why | why not |
|---|---|---|---|
| Train at high `token_state_dropout` | yes, `tul.token_state_dropout` | it is the spec's own collapse tax, shipped at 0.15. Raising it makes the coda's `ce_main` depend on the plan, which is the only route that would give the core real gradient | the sweep is EVAL-ONLY on a model trained at 0.15. It proves the plan is empty NOW, not that it cannot be FILLED by training at high p. Also: a coda that cannot see tokens may just get worse |
| Re-weight `emit` / `plast` | yes, `_tul_half_weights` | today they split 0.5/0.5 on the SAME target, so the core's only direct route is one it loses. `emit 1.0 / plast 0.0` forces the plan to carry it; `emit 0.0 / plast 1.0` removes the private route so the core earns only by helping the coda | the second may leave the core with no gradient at all — which at 0.0007 nats may be the honest answer, but it is not TUL working |
| A span-content objective for `z` | no | gives the core a job only it can do | the spec forbids decoding a span from one vector with no token path (Huginn collapse 2026-08-16, MegaByte T7, Bowman T2). Needs design, not a bolt-on |

**Do not open another stability arm.** Every one below failed and now there is a measured
reason why.

### Instruments built for this (do not rebuild)

| tool | measures |
|---|---|
| `lab/divergence/region_shapley.py` | exact Shapley over prelude/core/coda, per loss group |
| `lab/divergence/slot_path_worth.py` | loop vs whole-plan, four conditions, one eval set |
| `lab/divergence/token_tax_sweep.py` | plan worth against `token_state_dropout`, paired and seeded |
| `lab/divergence/delta_ablation.py` | nats a region is worth; `--ablate prelude/coda` is its own control (+3.22 / +3.11) |
| `lab/divergence/scale_probe.py` | whether the core map's output depends on its input size (it does not: pre-norm) |

### Two traps that each cost an experiment today

- **`ademamix_t_beta3` is `null` in `base.yaml` and falls back to `training.steps`.** Run
  length silently sets the AdEMAMix beta3 horizon. It flipped SCSE arm C from "survives to
  2506" to "dies at 1800". PIN IT in every arm script and check the
  `[opt] re-applied config hyperparameters` line.
- **A forward-only probe ranks arms backwards.** `delta_ladder.py` ranked arm A best and C
  second-worst; in training A was catastrophic and C was the only one that helped. A frozen
  probe measures how big a quantity is; the takeover is about where the gradient goes.

| | |
|---|---|
| Cures found | **none, 22 hypotheses in.** H5 `per_slot_embed` is still the only lever: doubles time-to-failure, cuts harm 78 %, 0.46-0.78 nats better CE on both seeds — and still fails seed 0 at step 2225 |
| Closed this session | H18 NOT SUPPORTED (attention is diffuse, not a sink). H24 defect CONFIRMED but REFUTED as a cure. H23 REFUTED. The full SCSE method's refuter fired — it stalls at step ~250 |
| **The blocking problem** | **This machine has no regime where the CONTROL diverges 2 of 2, so every binary arm is unreadable.** The RCA's 2-of-2 aborts (3240, 4540) were at batch 14; batch 12 gives 1 of 2. Three arm designs have been spent on this. Next is a standalone pre-registered CALIBRATION over batch size and `alpha_cap`, not another arm |
| Still open, untouched | [iteration-0 mediation](../../lab/experiments/planned/2026-08-23-tul-iteration0-mediation.md) — arms written, never run. It is the only pre-registered causal test of the one arm-specific forward asymmetry left (lead 4 below) |
| Blocked on a decision | **is TUL on the near path?** If not, stop and ship the mitigations. TUL is already OFF by default (`tul.activate_at: never`), so nothing shipping is exposed |

## Hypothesis ledger

Verdict is the outcome of a PRE-REGISTERED test unless marked otherwise.

| # | Hypothesis | Verdict | The deciding number | Record |
|---|---|---|---|---|
| H1 | Backward power iteration through the weight-shared core drives it | **partly — descriptive** | per-block gain 2.4–2.8, r² 0.97, but gain is the WRONG severity measure: `bptt_depth` 2 has a HIGHER gain (2.784 vs 2.445) with 64 % LESS harm | [takeover-cure](../../lab/experiments/failures/2026-08-24-tul-takeover-cure.md) |
| H2 | The core's weight spectrum runs away; cap `sigma_max` | **REFUTED** | 4 arms (soft 1.5/3.0, hard 1.5 MLP and MLP+attn). ALL FOUR had a worse CE rise than the control they protected | same |
| H3 | A spectral gap opens in the core weights | **REFUTED** | median sigma1/sigma2 1.069 → 1.132; the worst case FALLS | same |
| H4 | The forward slot states collapse in rank | **CONFIRMED, not causal** | rank ratio across the loop 1.23–1.48 healthy, 0.67–1.06 sick; flip precedes the gradient-share flip by ~50 steps | [core-takeover-is-positional](../../.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md) |
| H5 | `per_slot_embed` cures it by adding input diversity | **best lever, not a cure** | holds seed 1, fails seed 0 at 2225. Doubles time-to-failure, cuts harm 78 %, 0.46–0.78 nats better CE on BOTH seeds | [takeover-cure](../../lab/experiments/failures/2026-08-24-tul-takeover-cure.md) |
| H6 | Levers stack | **REFUTED** | `bptt2` + `per_slot_embed` at seed 0: highest gain of any arm, and 0.42 nats WORSE than `per_slot_embed` alone | same |
| H7 | `alpha * m_slow` explains the seed dependence | **REFUTED** | 17–24 % of the `g/sqrt(v)` channel throughout; spearman −0.393 against harm over 7 arms — anti-correlated | [optstate](../../lab/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md) |
| H8 | `\|\|dW_core\|\| x` directional autocorrelation is the right severity measure | **REFUTED** | clip pins `\|\|dW\|\|`; ac is +0.434 to +0.574 at EVERY rung. Spread 1.577x across an onset that moves core share 0.017 → 0.961 | same |
| H9 | Core/non-core gradient coherence is a better severity measure | **CONFIRMED, post-hoc** | 1.001 → 2.103 across the ladder, ~90 steps of lead. Spearman +0.857 vs harm on 7 arms where per-block gain gives +0.536. **Untested out of sample** | same |
| H10 | The core map is under-determined on the slot manifold | **REFUTED** | the input span is nearly full rank — 935 of 1024 directions hold 99 % of input energy | [under-determined](../../lab/experiments/failures/2026-08-24-tul-core-underdetermined.md) |
| H11 | More looped positions would fix it | **UNDERCUT** | the same sick weights collapse on 1024 TOKEN positions too: input eff rank 75.16 → 27.50 | same |
| H12 | The per-slot embedding rows re-converge, which is why seed 0 fails | **REFUTED, opposite direction** | the arm that FAILED had MORE row diversity: centred eff rank 42.92 vs 27.15 | same |
| H13 | Span mean-pooling dilutes slot diversity as `1/sqrt(L)` | **CONFIRMED** | slope −0.473 / −0.504 / −0.527 at three caps, r² up to 0.93, and deviation depends on L ALONE not on the config | [pooling-law](../../lab/experiments/successes/2026-08-24-tul-span-pooling-law.md) |
| H14 | Explorative Modeling applies: the plan is a compromise between conflicting readers | **REFUTED** | reader alignment 1.36–1.45 where 1.0 is random, FLAT across the onset. The slot's own label is 2.8 % of loss weight but ~50 % of the gradient | [reader-conflict](../../lab/experiments/failures/2026-08-24-tul-reader-gradient-conflict.md), [rejected note](../../.agents/notes/rejected/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md) |
| H15 | The per-iteration input re-injection decays, so the loop forgets each slot | **REFUTED** | anchor `dt/(1-A)` is 1.8015 → 1.8047 across all 11 rungs. Flat to four decimals | this file, 2026-08-24 |
| H16 | Finer spans (`span_cap` 8) prevent it | **partial — delays, does not cure** | `g6-ctrl` takes over at step 650, `g6-fine` at 975. 1.5x delay. Confounded: `g6-fine` also has 2.5x the core positions | this file, 2026-08-24 |
| H17 | The LEARNABLE attention sink absorbs the mass | **REFUTED** | core `sink_logits` are 0.0036 → 0.0053 across all 11 rungs (sigmoid 0.5009 → 0.5013). The explicit sink parameter never engages | this file, 2026-08-24 |
| H18 | A POSITIONAL attention sink — mass concentrating on one slot — compounds over T | **NOT SUPPORTED** | the core's forward attention is DIFFUSE at every rung: window participation ratio 0.59–0.68 of 57 valid slots, top-1 key 6.4–8.4 % of the mass, `argmax` always slot 0 or 1 (the causal+XSA artifact) with rows agreeing only 0.47–0.81. 1 of 5 pre-registered predictions held and it does not discriminate. The loop DOES compress attention entropy more at the sick rungs (−2 % at 1625 → −13 % at 1800/1850 → −5 % at 1866, the same shape and the same non-monotone tail as H4) but that is a weak correlate inside a diffuse regime, not a sink | [h18](../../lab/experiments/failures/2026-08-25-h18-positional-attention-sink.md) |
| H24 | The HCA compressed branch is DEAD on the slot path, so three of six core blocks run at half attention capacity — and that is the A1/A0 asymmetry | **DEFECT CONFIRMED, REFUTED as a cure** (arm 2026-08-25: 1 of 2 arm seeds diverged, same as control; aborts 2880 vs 2940, 2.1 % apart, on OPPOSITE seeds — [arm](../../lab/experiments/failures/2026-08-25-h24-hca-branch-arm-binary.md)) | the branch is real: `hca_compress_ratio` 256 against 64 slot positions gives `n_blocks = 0` and `\|out_comp\|` is exactly 0.0000 at core blocks 1/3/5, while the token path at the same weights has 4 blocks and `\|out_comp\|` ~ 1030. But the no-training screen (validity gate reproduces the published H4 ratios to 0.004, iteration 0 identical to 0.000e+00) shows reviving it lifts the loop's rank ratio by **+0.0939 on healthy rungs and +0.0947 on sick ones** — a UNIFORM capacity effect, not a targeted repair. It crosses 1.0 at rungs 1800 and 1825 only because they sit near the threshold; rung 1850 goes 0.714 -> 0.780 and 1866 falls 0.006. Expect a LEVER of the H5 class, not a cure | [h24 screen](../../lab/experiments/failures/2026-08-25-h24-hca-branch-screen.md), [h18 Phase 0](../../lab/experiments/failures/2026-08-25-h18-positional-attention-sink.md) |
| H19 | The zero-deviation forcing bias (SCSE) drives it: `e` re-injected every iteration leaves `G_theta(0) != 0`, so the state drifts and the drift compounds over T | **REFUTED, opposite direction** | the shared component of the per-iteration displacement DECAYS ~100x across the loop (`C_last/C_first` = 0.0076 where the prediction needed >= 3), zeroing the repeated additive injection moves it from 1.13 to 1.12, and the faithful once-only counterfactual (source at iteration 0 only, no decay) RAISES state diversity at 10 of 11 rungs, so the "SCSE would remove a de-correlator" argument is withdrawn — the port fails on its diagnosis, not on its effect | [forcing-bias](../../lab/experiments/failures/2026-08-24-tul-zero-deviation-forcing-bias.md) |
| H20 | The paper's own primitive `b_t(e) = T_t(0;e)` — the shared map's response AT the anchor — is large and arm-linked | **SURVIVES its first real control** | separately TRAINED A0 and A1, one build, 7 matched rungs: A1 carries 18-30% more anchor response at EVERY rung (1.177-1.301) and grows twice as fast (+7.5% vs +3.7%). Refuter did not fire. BUT neither arm turned around in 1900 steps, so this is healthy-vs-healthy — the gap is intrinsic to the arm, NOT yet shown to cause the failure | [arm-control](../../lab/experiments/successes/2026-08-24-tul-forcing-bias-arm-control.md) |

## The two numbers that still point somewhere

1. ~~**The loop, not the input, is where the diversity dies.**~~ **WITHDRAWN 2026-08-24 —
   this was a comparison between two different measures.** The ~28 came from `pooling_probe`
   on the slot INPUT, CENTRED, over at most 57 spans; the 1.7–4.8 came from
   `state_geometry` on the in-loop carrier, UNCENTRED and norm-weighted, averaged per row
   over ~50 slots. Different tensors, different normalisations, different group sizes — the
   last of which this file's own trap list forbids. Measured consistently on a FIXED set of
   96 slot positions, the loop RAISES unit-direction effective rank at all 11 rungs:
   3.01 → 7.06 uncentred and 10.67 → 18.79 centred at rung 1625, 2.41 → 9.18 and
   11.88 → 25.02 at TAKEOVER. `jac_ladder --state-probe` agrees on its own numbers — 3.18 →
   2.77, flat, never falling. Diversity rises MOST at the rung where the model is worst, so
   state diversity is not the failing quantity, which is consistent with every
   diversity-targeting arm having failed. The real drop, ~28 of 57 down to ~10.7 of 96
   centred, happens UPSTREAM of the core, in the embedding-to-prelude path.
2. **Only 320 of 1024 channels are re-anchored** to the slot's own input each iteration
   (`channel_dims: [512, 320, 192]`, `DiagonalInjection` covers the middle band). The other
   704 iterate the shared map with no per-example anchor. Structural, read off the model,
   not inferred.

3. **The loop stops settling at the onset — on BOTH arms.** The last iteration's
   displacement relative to the state, `rel_last`, rises from 0.702 to 1.081 on the slot
   path across the onset ladder, and from 0.682 to 1.077 on the token path. At takeover
   every core iteration moves the state by more than its own norm. This is the
   forward-trajectory form of `rho(J_core) >= 1`. It agrees to three decimals between the
   FAILING arm and the HEALTHY one, so it is a background condition of the recipe, not a
   sufficient cause — and any cure that only restores contractivity has to explain why A0
   never needed it.
4. **The one arm-specific forward asymmetry left is the FIRST iteration.** Half of A1's
   first-step displacement energy lies in one direction shared by all 342 slots and all 6
   rows (`C_first/P` = 0.44-0.58); A0's is 0.17-0.19. It is flat across the onset ladder, so
   it is a property of the slot construction rather than of the failure — but nothing else
   measured in the forward pass separates the arms this cleanly.

## Open ideas, ranked

| Idea | Cost | Why it might work | Why it might not |
|---|---|---|---|
| ~~**SCSE — anchored deviation recurrence**~~ (arXiv:2607.27656, Kim, Hayashi, Kamiya, Koyama, Iwasawa, Matsuo, 2026-07-30) — **DOWNGRADED by H19** | days | learn an anchor `h*(e)`, evolve the DEVIATION `Δ_t = h_t − h*(e)` through a zero-preserving core so `T_t(0;e) = 0` exactly. MORPH's `h = A·h + dt·e` is precisely the additive-injection form the paper says carries a non-zero forcing bias that accumulates over depth. Survives the entire ruled-out list: it re-parameterises the recurrence, it does not cap weights or touch the optimizer | H19 measured the mechanism and it is absent: the drift does not accumulate, and the injection SCSE deletes is the only measured de-correlator in MORPH's loop (removing the DiagonalInjection raises shared concentration 1.2-3.3x at every rung on both paths). Porting it unmodified would remove that. Also evaluated at 139M on WikiText, without stochastic depth or truncated BPTT |
| ~~Positional attention-sink probe~~ DONE 2026-08-25 | ~1 h, no training | — | ran; the attention IS diffuse, so this lead died exactly as the risk column said. It paid for itself by turning up H24 |
| ~~**The H24 arm: train with the core's HCA branch alive**~~ DONE 2026-08-25, **FAILED** — panel refused on V1 (control seed 1 never aborted), and P1 failed on its own: the arm diverged on seed 1 at 2940. An arm that diverges cannot be a cure whatever the control did | 1 config line + 2.6 h, 2 seeds | it fixes a real defect whatever it does to the divergence, and the no-training screen says it lifts the loop's rank ratio +0.09. H5, the best lever this campaign has, is exactly that shape of intervention | the screen already showed the lift is UNIFORM across healthy and sick rungs, so expect a lever that buys time, not a cure. Size the arm for time-to-failure and CE, not for a rescue |
| Make CSA's selection actually select on the short schedule | a log line now, its own arm later | `top_k` 256 exceeds `n_blocks` at `seq_len 1024` (144) and on the 64-slot core (8), so `tk == n_blocks` and CSA is DENSE pooled attention on every TUL arm run to date. It fires at the deploy 4096. [note](../../.agents/notes/proposed/bug-fix/2026-08-25-csa-sparse-selection-never-fires-on-the-short-schedule.md) | it would change every short-arm number, so the ablation table needs a re-baseline. Explicitly NOT folded into the H24 arm |
| Widen `DiagonalInjection` to all 1024 channels | 1 line + 40 min | targets the 69 % unanchored fraction; holds depth, FLOPs, sharing, capacity fixed; causal | H15 says the existing anchor is healthy, so more of it may change nothing. It is ALSO a weaker version of SCSE, which fixes the forcing bias rather than the channel count |
| `span_cap` 8 / `max_slots` 160 (RUNNING) | 40 min | measured 1.97x on slot-input eff rank | not compute-matched: `L_total` 1152 → 1344 and core positions 57 → 142. Granularity and slot count are inseparable at fixed `seq_len` |
| Wire H9's coherence ratio into the trainer | ~1 h | gives H9 its out-of-sample test AND ~90 steps of abort warning; makes every later arm cheaper | not a cure |
| Multi-token slot labels (next span's token SET) | half a day | raises the direct route above 2.8 % of loss weight | H14 showed loss weight is not gradient share; the direct route is already ~50 % of the gradient |
| XM on the plan: sample K candidate `h_i`, train the best | 1–2 days | the only untested form of the XM reading — across-dataset one-to-many coupling | K forward passes of the core per step; the within-example form is already refuted |
| Formal proof in a DSL (Z3) of what forces the rank ratio below 1 | days | four hypotheses have died to measurement; the surviving description is coincident, not causal. The recorded ladder can check a proof | the map is 6 transformer blocks with ternary QAT and hyper-connections; a tractable abstraction may prove something about a different system |

## Vetoed

* **Varying loop depth (`slot_mean_depth`).** Vetoed by Wolfe 2026-08-24. It removes the
  mechanism rather than fixing it and confounds the result with compute — the same trap as
  `bptt_depth` 2, which "helped" by truncating the loop. An arm was launched and killed.
* **Bounding core weight norms.** H2/H3. Four arms, one control, a pre-fixed rule and a
  spectral-gap measurement that says why.
* **`max_slots` 64 → 128 at batch 10–14.** OOMs. It fits at batch 6 (14.49 GB measured).

## Instruments built — do not rebuild these

| Tool | Measures |
|---|---|
| `morph/training/core_jacobian.py` | `sigma_max(J_core)` and typical gain, by power iteration on the real map |
| `lab/divergence/jac_ladder.py` | the above across a checkpoint ladder; `--state-probe` forward rank per iteration; `--gap-probe`; cap sweeps |
| `lab/divergence/optstate_probe.py` | AdEMAMix state decomposition from checkpoints; slow/fast channel, coherence, drift |
| `lab/divergence/score_optstate.py` | ranks candidate severity measures against measured harm |
| `lab/divergence/subspace_probe.py` | input-energy eigenbasis and g/u/m2 energy curves over k |
| `lab/divergence/pooling_probe.py` | span-length law, slot-input diversity, `1/sqrt(L)` slope |
| `lab/divergence/reader_conflict_probe.py` | per-reader gradients at the plan, `sqrt(K)`-normalised alignment |
| `lab/divergence/slot_rows_probe.py` | centred diversity of the per-slot embedding rows |
| `lab/divergence/drift_probe.py` | per-iteration displacement geometry off the captured core trajectory: shared concentration `C`, relative displacement `rel`, participation ratio over positions, and the two injection ablations. Has a trajectory gate that RAISES unless the replayed step reproduces the next captured state |
| `lab/divergence/score_arms.py` | the arm verdict rule; refuses a firing step when the probe cadence is too coarse |

## Traps this campaign has already fallen into

* `ademamix_eps_inside` is **false** — eps is OUTSIDE the sqrt. Assuming the floored form is
  a ~100x error and reports 99 % of coordinates on the epsilon floor.
* `preclip/core_share` is **contaminated** by any region-local loss term. The soft-penalty
  arms read 0.998 on the PENALTY's gradient.
* Loss weight is **not** gradient share. 2.8 % of the weight delivered ~50 % of the gradient.
* Effective rank is bounded by (rows − 1). Comparing it between groups of different size
  compares the group sizes.
* `tul.boundary_chars` cannot be a Hydra command-line override: the escaped comma arrives as
  the literal string `.;!?\,`. Use a config file.
* Two arms were lost to a control that refused to fail. **Run the control first.**
- **A replay probe in `model.train()` is not replaying the same map.** Every core block runs `nn.Dropout(0.1)`, so a replayed step draws a different mask than the captured one and lands 24 % away from the captured next state. `drift_probe.py`'s trajectory gate caught it; `core_jacobian.py`'s own replay has the same exposure and no gate. Zero the rates, do not switch to `eval()` — that also changes Poisson depths to a uniform `mean_depth`, i.e. a different operating point.
- **A concentration measure near 1 does not mean isotropic.** One position holding all the energy drives `P*||mean||^2/mean||.||^2` to 1 just as hard. Always pair it with the participation ratio over positions.
- **An ablation that REMOVES a term is not a model of an alternative that REPLACES it.** Setting `dt = 0` with the decay left running erases the ctx band toward zero; SCSE holds the same information in a persistent anchor. The first reading of that ablation produced a confident, published, wrong argument against the SCSE port. Build the counterfactual, do not infer it.
- **Never compare an effective rank across normalisations.** Centred and uncentred ranks of the same states differ by 3-4x here (10.7 against 3.0 at loop entry), because the states carry a mean pairwise cosine near +0.5. The campaign's headline finding survived for days on exactly this mistake.
- **Never compare an effective rank across ACTIVE-SET sizes.** The Poisson depth draw shrinks the slot path from 342 positions to 96 across the loop. Measure the trend on the intersection.
- **Test the source's OWN primitive, not your operationalisation of it.** H19 pre-registered predictions about coherent accumulation along the trajectory and refuted them, then reported the paper refuted. The paper's quantity is `T_t(0;e)`, evaluated AT the anchor, and it had never been computed. It turned out to be large and arm-linked.
- **1900 steps is not long enough to reproduce the takeover from scratch.** The campaign's own window for the CE minimum is step 500-2000, so a 1900-step arm can end AT the minimum and show nothing. Budget past 3000.
- **Adding evaluation changes the trajectory.** `onset-capture`'s replay recipe sets `eval_every=999999` for a reason: eval consumes RNG, and MORPH decorrelates within 11 steps of any perturbation at a fixed seed. A run with evals is a fresh sample, not a replay — you cannot have both the val curve and the exact trajectory.

- **The core map is NOT iteration-invariant, and this campaign assumed it was for days.** Two scorers shipped a guard asserting `b_rel` is "constant by construction before `route_start`". It raised on real data. Measured on `seedsweep-s1/step_3500.pt`: the probe is bit-exact run to run (`0.00e+00`), pinning `ret_state` to iteration 0's collapses the across-iteration spread to EXACTLY `0.00e+00`, and pinning `iter_idx` leaves it unchanged at `3.87e-03`. `T_t` depends on `t` through the **GLA retention state carried across loop iterations**. The effect is small here (worst `4.25e-03`) but the assumption was wrong, and only the guard caught it. SCSE is unaffected — Theorem 2 already indexes the bias as `b_k`.
- **A turnaround threshold picked without measuring the metric's noise floor is not a threshold.** H21 pre-registered "0.1 nats" for a validation CE whose within-run recovered rise reaches **0.168 nats**. Three healthy seeds with peak rises of 0.168 / 0.151 / 0.156 — the same behaviour — were split into different classes by where the final eval happened to land. Measure the floor first, and require several consecutive evals with no new minimum instead of reading one point.
- **All predictions HELD plus the refuter firing is one seed carrying the panel, not a success.** In H21 every HELD verdict came from seed 0, whose `b` is 8x-40x every other seed. Removing it took P3 from 5.193x to **1.002x**. Always re-run the group contrast with the extreme member removed before believing it.

- **A ratio diagnostic can be driven the "right" way by the wrong term.** SCSE's `R_t = ||b||^2/||realised update||^2` FELL sharply under Stage 1 (seed 3 at 3500: 0.520 -> 0.006), which by the paper's metric reads as success. `b_t` had gone UP 1.6x-5.5x and CE was 0.815 nats worse; `R` fell because the denominator inflated. Never report a ratio's direction without reporting which term moved.
- **MORPH is not in SCSE's regime, measured.** Their looped baseline runs `R_0 = 1.000` rising to `R_47 = 4.35-5.44`; MORPH's runs `R_0 = 1.000` FALLING to 0.056-1.906 at the last iteration. Before importing a cure, check the disease is the same one.
- **A control that survives answers nothing.** Two H24 arm designs were rejected before scoring because both ran where the control mostly lives (batch 6, 3500-6000 steps, 1 seed in 4 diverges). When the question is binary, pick the regime where the CONTROL reliably fails first, then vary one thing. `docs/tul-divergence-rca.md` §1 had that regime written down the whole time.
- **`training.steps` is an OPTIMIZER setting here, not just a stopping rule.** `optimizer.py:152` falls back to `training.steps` for `ademamix_t_beta3` when the key is null, which `base.yaml` ships. Raising the budget from 3500 to 6000 to get a better base rate lengthened the beta3 warmup by 71 % and the control stopped diverging entirely. Pin the horizon, do not move the budget. The replication writeup named this confound in writing and it was read the same day.
- **A prediction keyed on ONE rung is a prediction keyed on one draw.** The H24 screen put two of its three substantive predictions on rung 1850, called the "cleanest sick rung" because the campaign's two rank measures agreed there. It turned out to be the HARDEST rung, and both predictions failed while the effect was real everywhere else. Key predictions on the CLASS, not on its best-behaved member.
- **Ask whether an intervention is selective before calling it a mechanism.** Reviving the dead HCA branch moved the sick rungs by +0.0947 and the healthy rungs by +0.0939. Identical. Two sign crossings looked like a repair until the healthy column was put beside them. Always report the effect on the CONTROLS, in the same units, in the same table.
- **Audit the geometry before you measure the phenomenon.** H18's Phase 0 was meant to be a 30-minute sanity check on `n_blocks`. It found that three of six core blocks have an identically-zero attention branch on the slot path — a bigger result than the hypothesis it was clearing the way for, and one that four months of weight-spectrum probes had walked past. Measure what the module ACTUALLY computes at the shapes it actually runs at.
- **Write the predictions relative when the structure guarantees the absolute.** Phase 0 showed causal+XSA makes early slots absorb mass by construction, so an absolute "slot 0 is the top key" prediction would have HELD and meant nothing. Every H18 prediction was a ratio across iterations or across rungs for that reason.
- **An intervention that engages and makes things worse is worth more than another correlation study.** H21 spent a sweep establishing the forcing bias does not predict failure. H23 spent one sweep intervening on it and got a decisive answer. When a correlation study comes back null, intervene next.

## Loose ends

* `lab/experiments/planned/2026-08-23-tul-iteration0-mediation.md` — pre-registered, never
  resolved. Either run it or move it to `failures/` with the reason.
* H9's coherence measure is post-hoc and has never been tested on a run it was not fitted to.

- **Calibrate the control BEFORE the arm, as its own run.** The H24 binary arm was moved to
  the RCA regime precisely because a surviving control answers nothing — and the control
  still survived on 1 of 2 seeds. The RCA's 2-of-2 aborts were measured at **batch 14**;
  the arm ran at **batch 12**, and batch 12 does not reproduce them. That caveat was stated
  before the run and it is what broke the experiment. Three arm designs have now been spent
  on this. The next step is not another arm: it is a standalone, pre-registered calibration
  that finds the batch size and `alpha_cap` where the control aborts 2 of 2, and nothing
  else.

- **A validity gate is not always load-bearing in both directions.** V1 (both controls must
  abort) exists so that an arm SURVIVING is meaningful. It is not needed for an arm
  DIVERGING to be meaningful, because a diverging arm cannot be a cure whatever the control
  did. Read a refused panel for the conclusions that do not depend on the failed gate,
  say which ones those are, and do not quietly promote the rest.
