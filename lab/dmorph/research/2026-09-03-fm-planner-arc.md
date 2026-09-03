# FM planner arc — audit for dmorph

Work tree audited: `/home/wolfe/morph-perf` at `223cf85` (branch
`perf/throughput-lever-stack`). Read-only; no repo file touched.

## A. The FM objective exactly as implemented

**Two objectives, one body.** `FMPlannerConfig.objective` is `"edm"` or `"cfm"`
(`morph/model/fm_planner.py:572`). Both share the same `_PlannerLayer` stack,
masks, targets and probe — "one-variable arms" (`fm_planner.py:571`).

**Target `y`.** Pooled span-(i+1) representation, unit-L2 normalized, live
(not detached) — this is SIGReg's gradient path into the backbone.

Two implementations exist, gated equal by a test (`fm_planner.py:181`,
`tul_fm.py:180`):

- Standalone P1 (`fm_planner.pool_targets`, `fm_planner.py:247-266`):
  `y_i = mean(h[b, s_{i+1} : e_{i+1}+1])` then `y_i ← y_i / ‖y_i‖₂`, over a
  contiguous token-index range.
- Shipped model path (`tul_fm.fm_span_targets`, `tul_fm.py:173-203`): pooled
  from the loader's own `bag_id` map via `scatter_add_`, `y[:, :-1] =
  span_mean[:, 1:]` ("slot i -> span i+1"), then unit-L2 normalized.

Quote (`fm_planner.py:38-42`): *"WHAT IS DELIBERATELY NOT REUSED. `SliceScaler`.
Per-component-std target scaling is the known scar: it pushed σ* to 3.3 and put
77-98% of training into trivial autoencoding. Targets here are **unit L2 norm**
(`‖y‖₂ = 1`)."*

Context features = target features (`fm_planner.py:70-73`): `Hpre =
input_norm(prelude(input_ids))`, i.e. the exact tensor A3's coda consumes
(`n_core == 0` makes `_core_region` reduce to `input_norm`). In the shipped
model path, the context is `xn.mean(dim=2)` (HC stream mean, `transformer.py:1822`),
detached before the planner call.

**Source distribution.**
- CFM: `x0 ~ N(0, source_std² · I)` in TARGET space (`fm_planner.py:928-929`).
- EDM: `z = y + σ·ε`, `ε ~ N(0, I)` (`fm_planner.py:892-893`).

`source_std` is load-bearing, not cosmetic (`fm_planner.py:590-595`, quoted):
*"DeepWeightFlow App. H found the source variance decides whether a
low-capacity velocity net learns anything... Match it to the target's
per-component std: 1/sqrt(d) for raw unit-L2 targets (0.031 at d=1024), 1.0 for
whitened targets."* `tul_fm1.yaml` ships `source_std: 1.0` (spec-literal
`N(0,I)`) while printing a loud MISMATCHED warning
(`transformer.py:1784-1785`, `1.0` vs matched `1/√d`), because *"the arm spec
says N(0, I); `fm.source_std=0.03125` is the one-token sweep"* (config comment).

**Flow path / t schedule.**
- CFM: straight-line (rectified) interpolation, `x_t = (1−t)·x0 + t·y`, `t ~
  U(0,1)` per slot independently (`fm_planner.py:930-932`). `dx/dt = y − x0`
  exactly — a *constant* in `t` for a given `(x0, y)` pair (docstring,
  `fm_planner.py:919-922`).
- EDM: `σ ~ p(σ)` — truncated log-normal `p_mean=-1.2, p_std=1.2`, restricted
  to `[0.002, 80]` and renormalized in CDF space (`DBSchedule`,
  `build_schedule`, `fm_planner.py:462-479`), drawn per slot independently.

**What the planner predicts.**
- CFM (`FMPlanner.velocity`, `fm_planner.py:784-797`): raw velocity, NO
  preconditioning — *"the conditioning input is `t` in place of `c_noise` and
  the in/out maps are the identity, so the network output IS the velocity
  estimate."*
- EDM (`FMPlanner.denoise`, `fm_planner.py:769-782`): `D̂ = c_skip(σ)·z +
  c_out(σ)·F_θ(c_in(σ)·z, c_noise(σ), ctx)` — audited `EDMPrecond` coefficients
  from `morph/model/diffusion_blocks.py`, reused verbatim.

**Loss and its scale.**
- CFM: `loss = mean_valid[ ‖v̂ − (y − x0)‖² ]` — no σ-weighting
  (`fm_planner.py:915-948`).
- EDM: `loss = mean_valid[ w(σ)·‖D̂ − y‖² ]` with the audited EDM weight `w(σ) =
  (σ² + σ_d²)/(σ·σ_d)²` (`fm_planner.py:885-912`).
- `loss_scale: auto` divides the loss by `analytic_null_floor` (the expected
  loss of the zero network, closed-form for CFM `E‖v‖² = E‖y‖² + d·s²`, Monte
  Carlo for EDM) so the term "starts near 1.0 and is commensurate with a token
  CE of ~4-11 nats" (`tul_fm.py:89-93`). Without it, at `source_std=1.0, d=1024`
  the raw CFM term is `~1025`, which "would drown the CE entirely" (same
  docstring).

**Conditioning `ctx`.** `FMPlanner.encode_ctx` (`fm_planner.py:740-754`):
projects the (already-frozen-in-P1 / detached-in-model-path) context features
`h_ctx` through `ctx_proj`, adds a fixed sinusoidal position table, RMSNorms.
`detach()` is called again inside `encode_ctx` itself — *"belt-and-braces: the
backbone is already frozen and the features already arrive under `no_grad`. It
costs nothing and it makes 'no gradient reaches the backbone' a property of
THIS function rather than of the caller's discipline"* (`fm_planner.py:743-746`).
In the shipped model path (`transformer.py:1826-1831`), `generate_plans` runs
under `with torch.no_grad()` on `h_ctx.detach().float()`, and the returned plan
is detached AGAIN (`h_slots = z.detach().to(xn.dtype)`) — belt-and-braces at
two levels.

**The Euler ladder at inference** (`generate_plans`, `fm_planner.py:954-1010`,
called with `n_steps = cfg.fm.infer_steps`, default/shipped 6, "a FIXED
inference constant — no loop-depth variation in this arm — Wolfe veto, arc
note"):
- CFM: `x(0) ~ N(0, s²·I)`; `n_steps` forward-Euler steps of fixed size `dt =
  1/n_steps` on `dx/dt = v̂(x,t)` from `t=0` to `t=1`; the plan is `x(1)`.
- EDM: `z ~ N(0, (1+σ_max²)·I)`; `n_steps−1` Euler steps down a strictly
  descending equi-probability σ ladder (`euler_step`, sign convention `dt =
  σ_next − σ < 0`, taken from the authors' code not the paper's rendered
  equations); then one **final read** at `σ_min` (`final_read=True`) — *"the
  loop above stops one short, so without it the lowest-noise block never gets
  to speak"* (quoted from the audited sampler, `fm_planner.py:62-65`).

**Reaching the coda.** `_tul_fm_core` (`transformer.py:1797-1836`) detaches the
generated plan, then `_tul_plan_ablate` (no-op in `normal` mode) passes it to
`TULSlots.prefix_project` (`tul.py:309-336`): value `k` of slot `s` is `h_s ·
W_k` where `W_prefix` is `[prefix_k, d, d]`, **identity-initialized**
(`tul.py:242-243`, `eye = torch.eye(d_model).unsqueeze(0).repeat(prefix_k,1,1)`).
Those values are written at `slot_index[s] + k` via `scatter_positions` into
the coda's input (`transformer.py:1996-1997`). `W_prefix` is described as *"the
one remaining consumer of that parameter"* on the FM path — the paid loop
never builds or reads it (`transformer.py:1970-1972`, `tul.py:220-225`).

**The sigreg term** (`morph/model/sigreg.py`, LeJEPA / Epps-Pulley,
arXiv 2511.08544). `fm_sigreg_loss` (`tul_fm.py:206-231`) runs
`sigreg_epps_pulley(√d · y, num_slices)` on the VALID rows of the live
(undetached) pooled targets `y`. Loss returned is the mean over `num_slices`
random 1-D projections of the weighted-L2 gap between the empirical
characteristic function and `exp(-0.5t²)` (N(0,1)'s CF):

```
ecf = mean_n exp(i · (z@a) · t)          # [M, T], complex
err = |ecf - exp(-0.5 t^2)|^2 * exp(-0.5 t^2)
per_slice = trapz(err, t) * n
loss = mean_M(per_slice)
```
(`sigreg.py:62-76`, transcribed from Algorithm 1 of the paper, DDP
`all_reduce` dropped for single-GPU.)

The `√d` rescale is explained (`tul_fm.py:206-231`, quoted): *"SIGReg tests
each 1-D projection against N(0,1), i.e. it asks for E‖z‖² = d. The targets are
UNIT L2 norm, so E‖y‖² = 1. At d=1024 those two are incompatible by a factor of
1024... So the statistic runs on √d · y."* Role: `groups["fm_sigreg"]`
(`transformer.py:2035-2039`) is added to the total loss with weight
`sigreg_lambda` (default `0.02`), and it is the ONLY term that shapes the
targets — *"the JEPA lesson: if the FM term could move the targets it would
take the cheapest available route and make them predictable rather than
informative"* (`tul_fm.py:29-32`).

**The three gradient paths** — quoted verbatim from `tul_fm.py:21-32`:

| term | gradient reaches |
|---|---|
| `ce`     | backbone (prelude, coda, embeddings), `E_slot`, `E_mask`, `W_prefix` |
| `fm`     | the FMPlanner ONLY — its context input is detached and its target is detached |
| `sigreg` | the backbone, through the pooled targets. This is the term that SHAPES them |

*"The split is deliberate and is the JEPA lesson: if the FM term could move the
targets it would take the cheapest available route and make them predictable
rather than informative."*

Total loss assembly (`transformer.py:2018-2045`): `total = groups["loss"] +
fmc.fm_weight * fm_val [+ fmc.sigreg_lambda * sig]`; `groups["loss_tokens_only"]`
keeps the pure-CE tensor for the no-planner-gradient test
(`test_coda_ce_never_reaches_the_planner`).

## B. Planner architecture and cost

`FMPlannerConfig` (`fm_planner.py:562-624`) / `FMArmConfig` (`tul_fm.py:57-119`),
shipped defaults (`tul_fm1.yaml` and `FMArmConfig` defaults):

- `d_p=512`, `n_layers=4`, `n_heads=8`, `d_ff=1408` (SwiGLU: hidden bank
  `2*d_ff` wide; ratio `2.75× d_p`, matching `base.yaml`'s d_model→d_ff ratio;
  "at `d_ff=2048` the planner weighs 25.97 M, outside the declared 15-25 M
  band" — `fm_planner.py:66-69`).
- `cond_dim=256`, `sigma_n_freq` = `min(128, 2*cond_dim)` = 128.
- `_PlannerLayer` (`fm_planner.py:639-698`): pre-norm masked slot
  self-attention → masked cross-attention to context → SwiGLU MLP, each
  sub-block gated by a zero-init `AdaLNGate` (DiT AdaLN-Zero: `F_θ ≡ 0` at
  init).
- `n_params()` (`fm_planner.py:799-800`) sums `.parameters()`.

**Measured parameter counts** (from filed experiments, same config family):
- `lab/experiments/failures/2026-08-28-tulfm-p1.md`: "Planner: **21.94 M
  params**, `d_p=512`, 4 pre-norm layers, 8 heads, SwiGLU `d_ff=1408`."
- Shipped model-path banner (`transformer.py:1786-1789`) prints `~22.0 M`
  planner params plus, at full-model construction, the TUL slot machinery
  (`E_slot`/`E_mask`/`W_prefix`: `prefix_k × d² = 2 × 1024² ≈ 2.1 M`).
- `tul_fm1.yaml` header: A3 (n_core=0, no slots) is 207.7 M; FM1 expected
  ~232 M total (207.7 + 2.1 + ~22.0 M).

**Wall-clock / throughput measured** (all on the 5090, batch 6, seq 4096
panel arms unless noted; from the filed experiments):

| Arm | steps/s | peak VRAM | Reference |
|---|---|---|---|
| A1 (paid loop, aux) | 1.9 | — | tul-fm1 filing |
| A3 (n_core=0, no slots, coreless baseline) | ~4.5 | — | tul-fm1 filing |
| FM1 | 3.30 | 16.71 GB | tul-fm1-live-arm results |
| FM1-CW (576-token cut) | 3.27–3.30 | 17.03 GB | tul-fm1-cw results |
| FM2 (emit CE on) | 3.28 | 16.71 GB | tul-fm2 results |
| GL1 (restricted, tg_restrict, no planner) | 3.44 | — | tul-gl1 results |
| P1/P1b (planner-only, frozen A3 backbone) | ~3.2 min wall for 4000 steps | — | tulfm-p1 results |

FM1 is summarized in its own verdict: *"FM1 = A1's final CE at 1.74× A1's
speed (4.3501 vs 4.3472; 3.30 vs 1.9 sps), but ~0.35 nats behind A3 at matched
steps"* (`tul-fm1-live-arm.md`). No dedicated per-planner-forward FLOP count is
recorded anywhere in the audited files — only end-to-end steps/s and peak VRAM.

## C. Results — every FM filing's verdict with the deciding number

1. **`lab/experiments/failures/2026-08-28-tulfm-p1.md`** (P1/P1b, frozen A3
   backbone, EDM objective). Gate: within-row retrieval top-1 ≥ 0.06, MRR ≥
   0.12. P1 (`sigma_data=0.5`): top-1 **0.0235**, all bands sat at `rel≈0.51`
   (learned only the unconditional prior). P1b (`sigma_data=1/√d=0.03125`):
   top-1 **0.0423**, MRR **0.1336** (MRR gate passed, top-1 did not). Verdict:
   FAILURE by the compound rule; but real, context-conditional, sub-gate
   content was written (controls at chance, shuffle kills the signal).

2. **`lab/experiments/failures/2026-08-28-tulfm-p1c-objective-and-whitening.md`**
   (whitening + true-CFM sweep). Five configurations (sigma_data×2, objective
   family×2, whitening×2, steps×3) all land within **top-1 0.037–0.052**.
   `edm_white` (0.0421) ≈ P1b's 0.0423 — *"Whitening moved the EDM arm by
   nothing... the ANISOTROPY HYPOTHESIS IS DEAD as the binding constraint."*
   Best arm `cfm_white` at 12k steps: top-1 **0.0516**, MRR 0.1517 — plateaued.
   Post-verdict copy-baseline addendum: the zero-parameter "copy the current
   span" heuristic scores top-1 **0.0678**, MRR 0.1632 — *"The trained planner
   is approximately equal to the copy heuristic."*

3. **`lab/experiments/failures/2026-08-28-tul-fm1-live-arm.md`** (FM1, live
   model, co-trained). Retrieval jumped to **top-1 0.66–0.77** within-row
   (SIGReg decorrelated the target geometry, cos 0.63→~0.00 in 250 steps —
   *"The frozen-backbone information cap is DEAD"*) — but `val/plan_worth_shuffle
   ≈ 0.0000` at all 17 evals. Verdict: *"An accurate plan the decoder will not
   read. The bottleneck was never the planner — it is the coda's incentive to
   consume the prefix when full token context is available."*

4. **`lab/experiments/failures/2026-08-28-tul-fm1-cw.md`** (compaction window,
   coda cannot see tokens before position 576). `worth_shuffle` stayed in
   **[-0.0004, +0.0006]** at all 17 evals even with the cheap channel deleted.
   Unregistered finding: `plan_nats` (CE cost of removing the slot POSITIONS)
   grew **0.27 → 0.81 nats**, while zeroing plan CONTENT cost only 0.0053 and
   shuffling cost 0.0001 — *"The coda built 0.8 nats of machinery ON the slot
   positions... while reading nothing FROM them."*

5. **Oracle-prefix probe** (`successes/2026-08-28-oracle-prefix-probe.md`,
   referenced by the reader-bottleneck note): even the TRUE next-span target
   planted at the prefix scores **0.0000** — *"the content channel's entire
   signal is an is-something-there energy cue (+0.0003, identical for shuffled
   plans and scaled noise)."*

6. **`lab/experiments/failures/2026-08-29-tul-fm2.md`** (emit CE turned back
   on). `first_tok_counterfactual` ended at **−0.0898 @4250 / −0.0568 final**
   (gate: >0 at ≥2 of last 4 evals — 0 of 4 fired). `worth_shuffle` stayed at
   **−0.0002**. Second finding: the writer DEGRADED — `copy_gap` fell **0.47 →
   0.26**, `plan_top1` fell **0.62 → 0.44** — *"Emit gradients cannot reach the
   planner's WEIGHTS — but they reach its TASK: they reshape the prelude
   features, which define the pooled targets y."* Verdict: *"tax (dropout),
   deletion (CW), rerouting (TG), and now direct supervision (emit) are ALL
   null on the additive prefix interface. The training-pressure family is
   exhausted."*

7. **GL1 line** (`lab/experiments/failures/2026-08-29-tul-gl1.md`,
   `2026-08-29-tul-gl1-line2.md`, `successes/2026-08-29-tul-gl1b-gl1c.md`) —
   NOT the FM planner; this is the interface-surgery follow-up that dropped the
   planner in favor of a gradient-carrying tap write + `tg_restrict` masking
   (gisting configuration). GL1 fired the first-ever load-bearing slot content
   gate (`worth_shuffle` 0.0556–0.0696 at three consecutive evals, vs ≤0.0006
   for every FM/prefix-interface arm) but at a CE cost (0.3654–0.3377 nats)
   above its own 0.15-nat price gate. Round 2 (gl1-line2) found the reliance
   signal seed-dependent (gl1b-s2 never reached the gate) and the "mask earns
   CE" claim DEAD (unmasked+mux beats masked+mux, 4.3102 vs 4.4047); only the
   curriculum composition `gl1bc` held reliance robustly (0.05–0.097 at every
   eval) at a ~0.09-nat price against the properly matched (mux'd) ruler. This
   line is separate from and downstream of the FM-planner rejection — it does
   not use `fm_planner.py`/`tul_fm.py` at all.

8. **`lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md`** — the terminal
   FM-line gate, rerunning the exact P1 pipeline on the best available
   substrate (l2cap slot states, whose worth_profile is span-wide: "shuffle
   +0.22 at offset 0, +0.11 at offset 16+"). Binding decision rule: RV1
   (within-row top-1 ≥ 0.06, MRR ≥ 0.12) passes ⇒ FM earns a revival prereg;
   fails with controls clean ⇒ line stays dead. Result: **top-1 0.0167** (below
   even the P1-on-a3 readings of 0.0235/0.0423), **MRR 0.0858**. Controls held
   clean (trained ≈ untrained ≈ shuffled — *"the planner learned nothing
   retrievable at all"*). *"Binding rule fires: the FM planner line stays
   DEAD."*

**Why the arc was rejected** — the rejection note
(`.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md`, status line
and the `2026-08-30-objective-lines-vs-l2cap.md` annex), quoted:

> "Rejection (2026-08-30): the P1 revival gate on the l2cap substrate failed
> with clean controls (`lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md`)
> — the FM planner line is dead by binding rule."

> "**The common failure shape.** Both lines died the same way the gate did:
> they gave the optimization an alternative to building composition through
> the iterated map — FM by routing plan formation around the loop, DB by
> replacing iteration with conditioning-plus-inference-recurrence."

> "Boundary: every P1-family design targeted PRE-core prelude features
> (pairwise cos 0.50, eff-rank 40/1024 — clustered, hard targets); post-core
> carriers were never tested by any FM design. That is the one live opening,
> parked as a NEW design, not a revival."

> "**Consequences.** Any future FM-flavored proposal must target post-core
> carriers or it is pre-refuted; any DB-flavored proposal must be loop-free
> (dmorph) or it is pre-refuted at this scale."

Note: I could not find a `2026-08-30-dmorph-handoff.md` file anywhere in the
repo despite the `objective-lines-vs-l2cap` note citing it
(`.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md`) — it is
referenced but not present in this work tree at `223cf85`. Flag this for the
architect: the handoff note the objective-lines note points to does not exist
in `/home/wolfe/morph-perf`.

## D. The probing doctrine's rules (`docs/tul-fm-probing.md`)

Numbered rules from §4, "Rules written in blood (each cites its scar)":

1. Report the shuffle COST, not the specificity FRACTION, whenever the zero
   cost is not comfortably positive (fraction's denominator collapses through
   zero — tg3b at p=1.0 read −55.4%).
2. `exit=0` is not "no takeover" — only `score_arms.py` decides; both a1noaux
   seeds exited 0 and had taken over.
3. Block backward gain is the MECHANISM (gain > 1, r² ≥ 0.5); core share is
   the SYMPTOM. Gain separates arms at matched steps; share verdicts are
   run-length confounded.
4. Worth metric is `ce_main`, never `loss` — `loss` folds in the §5
   half-weighted emit/plast positions. Use `val/ce_tokens` for cross-arm
   curves.
5. Pin `training.ademamix_t_beta3` explicitly on every run — null falls back
   to `training.steps` and silently changes the optimizer between run lengths.
   Verify from Hydra's frozen `config.yaml`, not the CLI.
6. n=1 comparisons are unreadable (11-step decorrelation, 6.5% median spread).
   Two seeds minimum; within-run measurements beat across-run wherever
   possible.
7. Eval-step val jumps are correlated ACROSS arms and seeds — an eval-batch
   sampling artifact. Compare arms at the same step only.
8. Forcing new inputs at eval measures OOD shock, not worth. A condition is
   comparable only if the model TRAINED with that input distribution, or a
   positive control absorbs the shock.
9. A probe must run on the shipped path. Every arm runs `_tul_core`; a
   token-path guard protects nothing.
10. Never edit a running script; write a new one. Detached scripts need
    `setsid`, verify with a later `pgrep`.
11. Generation claims need the diversity guard (gen-PPL alone is fooled by
    repetition loops) — always pair gen-PPL with rep4/distinct-3.
12. Pre-register before the run; predictions frozen; a failed prediction is
    withdrawn, not defended.

Plus §5 phase gates (P1/P2/P3) and §6-7 model-level rules:
- §6: the ONLY cross-arm gate at the planner level (P1x) is the retrieval
  probe — it is objective-agnostic. Planner-level losses (EDM weighted-denoise
  vs CFM velocity vs token CE) are incommensurable and must never be ranked
  against each other, even rescaled. At the model level (P2/P3), compare
  against the non-TUL baseline via held-out token CE + AR generation
  (gen-PPL + diversity guard). **Never MAUVE**, and never invent a "planner
  perplexity."
- §7: a true CFM arm (straight-line interpolation, velocity target, uniform t)
  is a REQUIRED arm alongside any EDM/denoising arm before any objective-level
  conclusion, not an optional variant — the two differ in weighting, source
  distribution and conditioning variable, and the source-scale knob is
  load-bearing at low capacity.

## E. What is reusable for a no-loop design applying FM/diffusion to the model's own hidden states

**Directly reusable modules (objective mechanics, not the side-planner
wiring):**

- `morph/model/fm_planner.py`: `DBSchedule`/`build_schedule`,
  `band_edges`/`band_edges_for`/`band_of_sigma` (σ or t equi-probability
  banding — objective-agnostic reporting), `analytic_null_floor` (closed-form
  CFM null, Monte-Carlo EDM null — needed for any `loss_scale: auto` pattern),
  `_cfm_loss`/`_edm_loss`/`fm_loss` (the two-objective loss dispatch, reusable
  almost verbatim if the "context" and "target" become the model's own hidden
  states rather than a side planner's inputs), `effective_rank` /
  `mean_pairwise_cos` (the collapse-guard pair — SIGReg diagnostics that
  generalize to any latent, not just plan states), `TargetWhitener`/
  `fit_whitener` (PCA whitening — measured to NOT move the retrieval gate here,
  §C item 2, so port with that null result attached, not as an assumed win).
- `morph/model/sigreg.py`: `sigreg_epps_pulley` is a general isotropy
  regularizer with no dependency on the FM/TUL machinery at all — directly
  applicable to any hidden-state population a no-loop model wants to keep from
  collapsing.
- `morph/model/tul_fm.py`: `copy_gap_scores` — the zero-parameter continuity
  baseline is the single most important reusable INSTRUMENT in this whole
  arc; the P1c addendum shows a trained 22M-param planner was statistically
  indistinguishable from "guess the next thing looks like the current thing."
  Any diffusion-on-hidden-states objective needs this baseline from day one.
- `docs/tul-fm-probing.md` doctrine (§D above) generalizes directly: rule 4
  (worth metric is `ce_main`/held-out CE, never the diffusion loss itself),
  §6 (never rank objectives by their own training loss — only by a shared,
  objective-agnostic gate), §7 (CFM is a required arm alongside EDM, not
  optional) all apply unchanged to a no-loop diffusion-on-hidden-states design.

**Tests worth reusing as gate templates** (`tests/test_tul_fm1.py`,
`tests/test_tulfm_p1.py`): `test_coda_ce_never_reaches_the_planner` /
`test_the_generated_plan_carries_no_graph_at_all` (the detach-boundary
pattern, directly relevant if dmorph keeps ANY detached side-channel);
`test_sigreg_scale_fix_is_what_makes_the_statistic_achievable` (the `√d`
rescale is a generic unit-norm-vs-isotropic-Gaussian fix, not FM-specific);
`test_plans_are_causal_in_the_span_axis` / `test_slot_end_is_the_last_token_
before_the_slot` (causal-masking correctness pattern for any span-conditioned
objective); `test_copy_gap_recovers_a_known_answer` /
`test_copy_gap_excludes_the_self_match_candidate` (baseline-instrument
correctness).

**What does NOT transfer, and why — the load-bearing assumptions tied to
`SlotLayout` / the TUL apparatus:**

- The entire write/read SEPARATION this arc lived and died on (`W_prefix`
  additive-prefix interface, `TULSlots.prefix_project`, the detach boundary at
  `h_slots = z.detach()`) is specific to a model that has SLOT positions
  distinct from token positions. A no-loop design applying diffusion to the
  model's OWN hidden states (every position, not a side-channel slot) has no
  analogous "prefix injection" site and must decide fresh whether/where a
  detach boundary belongs.
- `fm_geometry`/`SpanGeometry` and `pool_targets`/`fm_span_targets` are built
  on `SlotLayout` fields specifically: `layout.slot_index` (row position of
  each slot, used to define `slot_end = slot_index − 1`), `layout.slot_valid`
  (real vs. padded slots — `SlotLayout` guarantees pads sort last, which
  `_tul_plan_ablate`'s shuffle relies on), `layout.bag_id` (token→span
  membership map, used by `fm_span_targets`'s `scatter_add_` pooling —
  `layout.max_slots` is the dump bin), `layout.prefix_k` (coda positions per
  slot — meaningless without a slot/token position split), and the derived
  properties `layout.max_slots` / `layout.l_total` (used to size the
  planner's slot-index embedding and position table). None of these exist on
  a plain token sequence; a hidden-state-diffusion design over ordinary
  positions needs an entirely different geometry object (most likely just
  `[B, L]` position indices, no slot/token distinction, no bag-mean pooling).
- **The decisive finding this arc leaves for a hidden-state design**: the
  planner-level retrieval gate topped out at ~2.5× chance (top-1 ≈0.05, at or
  below the zero-parameter copy baseline of 0.0678) on EVERY tested target
  geometry (raw, whitened, EDM, CFM) when the targets were PRE-core prelude
  features. The one untested cell — POST-core carriers — is the specific gap
  `2026-08-30-objective-lines-vs-l2cap.md` names as the live opening. If
  dmorph applies FM/diffusion to hidden states that are themselves the output
  of some processing (not raw prelude pooling), it is testing exactly that
  unexplored cell, not repeating a refuted one — but the retrieval-probe
  infrastructure (`copy_gap_scores`, `effective_rank`, `mean_pairwise_cos`,
  the doctrine's §6 "never rank by training loss" rule) is the correct
  reusable gate to apply to it.

## Files read (for reference)

- `morph/model/fm_planner.py` (1055 lines, full read)
- `morph/model/tul_fm.py` (292 lines, full read)
- `morph/training/fm_setup.py` (66 lines, full read)
- `morph/model/sigreg.py` (77 lines, full read)
- `morph/model/transformer.py` lines 900-1000, 1737-2160 (`_build_fm`,
  `_tul_fm_core`, `_forward_tul` fm branch, `_tul_plan_ablate`,
  `tul_forward_ablated`/`tul_fm_forward`, `fm_eval_probe`)
- `morph/model/tul.py` lines 200-345 (`TULSlots.__init__`, `prefix_project`)
- `morph/model/tul_layout.py` lines 379-450 (`SlotLayout`)
- `morph/configs/tul_fm1.yaml`, `tul_fm2.yaml`, `tulfm_p1.yaml`,
  `tulfm_p1c_cfm_raw.yaml`, `tulfm_p1c_edm_white.yaml`,
  `tulfm_p1c_cfm_white.yaml`
- `docs/tul-fm-probing.md` (full)
- `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md` (full)
- `.agents/notes/proposed/architecture/2026-08-28-fm-reader-bottleneck-cw.md`
  (full)
- `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md`
  (full)
- `lab/experiments/failures/2026-08-28-tulfm-p1.md`,
  `2026-08-28-tulfm-p1c-objective-and-whitening.md`,
  `2026-08-28-tul-fm1-live-arm.md`, `2026-08-28-tul-fm1-cw.md`,
  `2026-08-29-tul-fm2.md`, `2026-08-29-tul-gl1.md`,
  `2026-08-29-tul-gl1-line2.md` (all full)
- `lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md` (full)
- `lab/tulfm/train_p1.py` header + imports (no README present in `lab/tulfm/`)
- `tests/test_tul_fm1.py`, `tests/test_tulfm_p1.py` (test names enumerated)

Not found despite being cited by a read note: `.agents/notes/proposed/
architecture/2026-08-30-dmorph-handoff.md` (referenced by
`2026-08-30-objective-lines-vs-l2cap.md` as living on branch
`feat/db-objective-l2`, not present on `perf/throughput-lever-stack` at
`223cf85`).
