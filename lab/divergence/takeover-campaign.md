# The TUL core-takeover campaign — running index

**Read this first.** Every hypothesis tried, its verdict, the number that decided it, and
where the detail lives. One line per idea. If an idea is not here, it has not been tried.

Last updated 2026-08-24. **This is a LAB REPORT** — trial and error, kept in
`lab/` on purpose. Settled claims graduate to `docs/experiments/` and
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

| | |
|---|---|
| Nothing running | `g6-ctrl` took over at step 650; `g6-fine` at 975 and aborted 19:01. Finer spans DELAY 1.5x, do not cure |
| Next | A1 seeds run past 3000 steps, scoring `b_t` against whether each seed turns around. H20 is arm-linked on trained models but the link to FAILURE is untested, because neither Stage 0 arm diverged (A1 `C_first/P` 0.44-0.58 vs A0's 0.17-0.19) — see H19 and the untested [iteration-0 mediation](../../docs/experiments/planned/2026-08-23-tul-iteration0-mediation.md) |
| Blocked on a decision | is TUL on the near path? If not, stop after `g6-fine` and ship the mitigations |

## Hypothesis ledger

Verdict is the outcome of a PRE-REGISTERED test unless marked otherwise.

| # | Hypothesis | Verdict | The deciding number | Record |
|---|---|---|---|---|
| H1 | Backward power iteration through the weight-shared core drives it | **partly — descriptive** | per-block gain 2.4–2.8, r² 0.97, but gain is the WRONG severity measure: `bptt_depth` 2 has a HIGHER gain (2.784 vs 2.445) with 64 % LESS harm | [takeover-cure](../../docs/experiments/failures/2026-08-24-tul-takeover-cure.md) |
| H2 | The core's weight spectrum runs away; cap `sigma_max` | **REFUTED** | 4 arms (soft 1.5/3.0, hard 1.5 MLP and MLP+attn). ALL FOUR had a worse CE rise than the control they protected | same |
| H3 | A spectral gap opens in the core weights | **REFUTED** | median sigma1/sigma2 1.069 → 1.132; the worst case FALLS | same |
| H4 | The forward slot states collapse in rank | **CONFIRMED, not causal** | rank ratio across the loop 1.23–1.48 healthy, 0.67–1.06 sick; flip precedes the gradient-share flip by ~50 steps | [core-takeover-is-positional](../../.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md) |
| H5 | `per_slot_embed` cures it by adding input diversity | **best lever, not a cure** | holds seed 1, fails seed 0 at 2225. Doubles time-to-failure, cuts harm 78 %, 0.46–0.78 nats better CE on BOTH seeds | [takeover-cure](../../docs/experiments/failures/2026-08-24-tul-takeover-cure.md) |
| H6 | Levers stack | **REFUTED** | `bptt2` + `per_slot_embed` at seed 0: highest gain of any arm, and 0.42 nats WORSE than `per_slot_embed` alone | same |
| H7 | `alpha * m_slow` explains the seed dependence | **REFUTED** | 17–24 % of the `g/sqrt(v)` channel throughout; spearman −0.393 against harm over 7 arms — anti-correlated | [optstate](../../docs/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md) |
| H8 | `\|\|dW_core\|\| x` directional autocorrelation is the right severity measure | **REFUTED** | clip pins `\|\|dW\|\|`; ac is +0.434 to +0.574 at EVERY rung. Spread 1.577x across an onset that moves core share 0.017 → 0.961 | same |
| H9 | Core/non-core gradient coherence is a better severity measure | **CONFIRMED, post-hoc** | 1.001 → 2.103 across the ladder, ~90 steps of lead. Spearman +0.857 vs harm on 7 arms where per-block gain gives +0.536. **Untested out of sample** | same |
| H10 | The core map is under-determined on the slot manifold | **REFUTED** | the input span is nearly full rank — 935 of 1024 directions hold 99 % of input energy | [under-determined](../../docs/experiments/failures/2026-08-24-tul-core-underdetermined.md) |
| H11 | More looped positions would fix it | **UNDERCUT** | the same sick weights collapse on 1024 TOKEN positions too: input eff rank 75.16 → 27.50 | same |
| H12 | The per-slot embedding rows re-converge, which is why seed 0 fails | **REFUTED, opposite direction** | the arm that FAILED had MORE row diversity: centred eff rank 42.92 vs 27.15 | same |
| H13 | Span mean-pooling dilutes slot diversity as `1/sqrt(L)` | **CONFIRMED** | slope −0.473 / −0.504 / −0.527 at three caps, r² up to 0.93, and deviation depends on L ALONE not on the config | [pooling-law](../../docs/experiments/successes/2026-08-24-tul-span-pooling-law.md) |
| H14 | Explorative Modeling applies: the plan is a compromise between conflicting readers | **REFUTED** | reader alignment 1.36–1.45 where 1.0 is random, FLAT across the onset. The slot's own label is 2.8 % of loss weight but ~50 % of the gradient | [reader-conflict](../../docs/experiments/failures/2026-08-24-tul-reader-gradient-conflict.md), [rejected note](../../.agents/notes/rejected/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md) |
| H15 | The per-iteration input re-injection decays, so the loop forgets each slot | **REFUTED** | anchor `dt/(1-A)` is 1.8015 → 1.8047 across all 11 rungs. Flat to four decimals | this file, 2026-08-24 |
| H16 | Finer spans (`span_cap` 8) prevent it | **partial — delays, does not cure** | `g6-ctrl` takes over at step 650, `g6-fine` at 975. 1.5x delay. Confounded: `g6-fine` also has 2.5x the core positions | this file, 2026-08-24 |
| H17 | The LEARNABLE attention sink absorbs the mass | **REFUTED** | core `sink_logits` are 0.0036 → 0.0053 across all 11 rungs (sigmoid 0.5009 → 0.5013). The explicit sink parameter never engages | this file, 2026-08-24 |
| H18 | A POSITIONAL attention sink — mass concentrating on one slot — compounds over T | **UNTESTED, top open lead** | the cotangent already sits on the same top-3 slots at every core block with the top slot's share rising 0.18 → 0.54. That is a sink signature, measured, never followed up | — |
| H19 | The zero-deviation forcing bias (SCSE) drives it: `e` re-injected every iteration leaves `G_theta(0) != 0`, so the state drifts and the drift compounds over T | **REFUTED, opposite direction** | the shared component of the per-iteration displacement DECAYS ~100x across the loop (`C_last/C_first` = 0.0076 where the prediction needed >= 3), zeroing the repeated additive injection moves it from 1.13 to 1.12, and the faithful once-only counterfactual (source at iteration 0 only, no decay) RAISES state diversity at 10 of 11 rungs, so the "SCSE would remove a de-correlator" argument is withdrawn — the port fails on its diagnosis, not on its effect | [forcing-bias](../../docs/experiments/failures/2026-08-24-tul-zero-deviation-forcing-bias.md) |
| H20 | The paper's own primitive `b_t(e) = T_t(0;e)` — the shared map's response AT the anchor — is large and arm-linked | **SURVIVES its first real control** | separately TRAINED A0 and A1, one build, 7 matched rungs: A1 carries 18-30% more anchor response at EVERY rung (1.177-1.301) and grows twice as fast (+7.5% vs +3.7%). Refuter did not fire. BUT neither arm turned around in 1900 steps, so this is healthy-vs-healthy — the gap is intrinsic to the arm, NOT yet shown to cause the failure | [arm-control](../../docs/experiments/successes/2026-08-24-tul-forcing-bias-arm-control.md) |

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
| Positional attention-sink probe | ~1 h, no training | Wolfe's hypothesis. The learnable sink is refuted (H17) but positional concentration is not measured. Would explain rank collapse, its compounding over T, and its reproduction on the token path — all without a weight-spectrum change | if attention is diffuse, the last cheap forward hypothesis dies |
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

## Loose ends

* `docs/experiments/planned/2026-08-23-tul-iteration0-mediation.md` — pre-registered, never
  resolved. Either run it or move it to `failures/` with the reason.
* H9's coherence measure is post-hoc and has never been tested on a run it was not fitted to.
