# DiffusionBlocks-on-MORPH audit for dmorph

Work tree: `/home/wolfe/morph-perf`, branch `perf/throughput-lever-stack`, HEAD `223cf85` (= master). No files modified.

## A. Every DB objective variant implemented in MORPH

### A0. Whole-model DB (`morph/model/diffusion_blocks.py`, modes `b1`/`b3`) — REMOVED from current tree, code lives on `park/db-master-line` and `feat/db-objective-l2`

The module still exists on the current tree (reintroduced at `cf41f8b` to serve TUL-FM/iter_cond, see part C), but nothing on the current tree builds a whole-model prelude|core|coda denoiser from it — `DBConfig(mode="b1"/"b3")` is never constructed except inside `fm_planner.py`'s own `_db1_schedule` helper (`mode="b1"`, used only to reuse `DBSchedule`'s CDF machinery, not a training path). `morph/training/db_setup.py` and `morph/inference/db_generate.py` are **gone** from the current tree (`git log --oneline --diff-filter=D` confirms deletion at `938d2e9`, "Remove DiffusionBlocks from master: tested to a verdict, rejected"). The four `docs/diffusionblocks-*.md` files that `diffusion_blocks.py`'s own docstring still points to are also gone (deleted in the same commit) — see part D.

**What was noised.** The clean target `y = HybridEmbedding output` of the labels (i.e. `embed(labels[t]) = embed(input_ids[t+1])`, the next token), scaled by `SliceScaler` so each embedding slice (euclidean, Lorentz-tangent) independently reaches per-component std `σ_data` (`morph/model/diffusion_blocks.py:424-478`). `z_σ = y + σ·ε`, `ε ~ N(0, I)`.

**EDM preconditioning** (`EDMPrecond.coeffs`, `diffusion_blocks.py:401-411`), `sd = σ_data`:
```
c_skip = sd² / (σ² + sd²)
c_out  = σ·sd / sqrt(σ² + sd²)
c_in   = 1 / sqrt(σ² + sd²)
c_noise = 0.25·log(σ)
```
Network input `c_in·z_σ`; denoised output `D̂ = c_skip·z_σ + c_out·F_θ(c_in·z_σ, cond)`.

**EDM loss weight** (`EDMPrecond.weight`): `w(σ) = (σ² + σ_d²)/(σ·σ_d)²`.

**σ distribution.** `log σ ~ N(P_mean, P_std²)`, truncated to `[σ_min, σ_max] = [0.002, 80.0]`, `P_mean=-1.2, P_std=1.2` (paper defaults) or a reshaped `p_mean/p_std` (config-swept). Equi-probability σ-partition across `B` blocks in CDF space (`DBSchedule`, `diffusion_blocks.py:257-388`); block `b` (layer order) owns `[σ_lo, σ_hi]` with γ=0.1 overlap extension.

**Loss (`DBConfig.loss_kind`):**
- `"l2"` (Option A, paper's AR method, App. B/C): `w(σ)·‖D̂ − y‖²` in embedding space, no readout used in training.
- `"ce"` (Option B, MORPH extension): CE on the tied LM head applied to `D̂` — `unweighted` by default (`loss_weighting="unweighted"`); applying EDM's `w(σ)` to CE was tested and causes collapse (see B).

**Conditioning (`DBConfig.conditioning`):** `"concat"` — clean and noisy streams concatenated under a modified causal mask, `clean_noisy_mask()` (`diffusion_blocks.py:541-565`): clean→clean ordinary causal, clean→noisy never, noisy→clean causal with cutoff `j≤i` (NOT `j≤i+1`, which would leak `clean_{i+1}=embed(labels[i])`, the target), noisy→noisy causal. `"x0_inject"` — additive injection of `x0[t]=embed(input_ids[t])` through the existing `ChannelInject` path (cheaper, one sequence).

**Inference:** `DBSchedule.inference_sigmas(n_steps)` gives `n_steps` DESCENDING equi-probability σ; `euler_step()` (`diffusion_blocks.py:568-582`) does the EDM-form probability-flow Euler step `z ← α·z + (1−α)·D̂`, `α=σ_next/σ`. For a CE-trained model, the sampler bridge from logits back to embedding space is `expected_embedding()` = `softmax(logits) @ E` (soft) or argmax-row (hard) — see B for why hard wins.

**Measured result (clean-room testbed, `11-DiffusionBlocks-Testing`, 124M 12L/768d Llama, matched 143.4M tokens):** plain AR CE **4.0010** vs best DB arm (`db_b1_oracle`, σ_data=0.5) **5.0801** at σ_max; `db_b4` (B=4 blocks) at 4x tokens (573M) still 4.6740, +0.67 nats. Generation at matched diversity: AR gen-PPL 162.07 vs best DB gen-PPL 325.85 (`db_b4`). Source: `.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md`.

### A1. `db_b1_l2` — Option A, x0_inject, B=1

**What is noised:** the euclidean+Lorentz embedding `y=embed(labels[t])`, `SliceScaler`-scaled. **Target:** same `y` (L2 regression). **Loss:** `w(σ)·‖D̂ − y‖²` (EDM-weighted L2), no readout in training. **σ sampling:** paper defaults then reshaped (`p_mean=0.0, p_std=1.6`) to push mass toward high σ. **Conditioning:** x0_inject (additive, one stream). **Inference:** Euler ladder + argmax/soft bridge for eval-only decode.

**Result — FAILS the context gate (autoencoding).** Falsifier (matched vs scrambled context L2 at 8 batches): gap ≈ 0.000–0.005 at every σ tested (0.05→9.79); pre-registered pass bar was gap > 0.3. The model ignores x0 context entirely; it only learns the low-σ copy + marginal-mean regression. AR generation from a greedy 4-Euler-step decode: word salad from token 1 ("The history of the Roman Empire the\nousal, under dileg International al0 he 2..."). Source: `.agents/notes/archived/testing/2026-08-19-db-b1-l2-pathfinder.md`.

A follow-up fix run (`db-b1-l2-fix`: detached target + `ce_anchor_lambda=1.0` Diffusion-LM rounding term + σ-reshape) **still fails**: matched-vs-scrambled gap stays ~0 at every high σ (σ=1: gap 0.0019; σ=3: 0.0242; σ=9.79: 0.0117). Same file. **Root cause found to be structural, not fixable by conditioning:** `z_σ = y + σ·ε` is a noised copy of the target itself, so denoising `z_σ→y` is strictly easier than reading x0 context — the model always has a shortcut and takes it.

### A2. `concat` conditioning ablation (`db_b1_concat_l2`)

Same objective as A1 (L2), only conditioning swapped x0_inject → concat. **Result:** concat ≡ x0_inject on every metric (falsifier gap @σ=9.79: 0.01170 x0_inject vs 0.01163 concat; gen-PPL ~50k vs 45,832 against an English baseline of 13). **The real diagnosis (not a leak):** `shape_decomp.py` compared model output to the EDM zero-network identity `D̂=c_skip·z` and to chance (`σ_d²=0.25`). At low σ the model IS the identity (z passes straight through, nothing learned); at high σ the model is WORSE than the chance predictor (0.371 vs 0.25 chance at σ=9.79) because the L2-optimal prediction of a many-valid-next-token target is a MEAN embedding — not decodable to any token. This is intrinsic to embedding-space L2 regression, proven independent of conditioning (x0≡concat) and independent of leakage (leak_probe.py / leak_watcher.py: exactly 0.0 movement on perturbed targets, positive control confirms the probe can detect leaks). Source: same pathfinder note, "CONCAT VERDICT" section.

### A3. `db_b1_ce` — Option B, x0_inject, CE, unweighted, reshaped σ (the "breakthrough" arm)

**What is noised:** same embedding target. **Target for the loss:** the ORIGINAL token id (CE), read off `D̂ @ E.T` (tied weight-tied head). **Loss:** unweighted CE (EDM `w(σ)` applied to CE was separately tested and causes collapse — train loss→0, val≈ln V; see B). **σ:** `p_mean=0, p_std=1.6`.

**Result — the first arm to use context.** `scrambled_control` @σ=9.79: matched CE 4.87 ≪ scrambled 8.89 (every prior arm was matched≈scrambled≈ln V=10.82). It escapes the L2 regression-to-mean collapse because CE degrades gracefully to a high-entropy distribution rather than to an undecodable mean vector. At 10k steps it generates grammatical English (16-Euler-step sample, gen-PPL 380–775 vs real-text baseline 13) — this was initially misread as "still gibberish" because the sampler eval defaulted to `n_steps=mean_depth=6` Euler steps, far too few (see B, "low-σ collapse" is unrelated; this was a separate sampler bug). Source: memory `db-ce-breakthrough.md`; corroborated in `db_b1_l2_pathfinder.md`.

This CE arm's B=1-recurrent-depth analysis on the clean testbed is the same family scored in A0's table (`db_b1`, `db_b1_oracle` rows) — B=1 is measured to be DB's **weakest** partition setting; B=4 nearly triples σ_max quality (4.67 vs 7.30 CE) — see B.

### A4. `db_loop` / `l3` — TUL loop-side target scheduling, no conditioning

Applies DB-style σ-band target scheduling to MORPH's existing TUL slot core loop, but with **no σ/AdaLN conditioning at all** (a detached, unconditioned 6-iteration unroll). **Result:** depth-inert — CE @4250 = 4.3519 (66-70 min wall clock), depth curve flat within `|Δ|≤0.001` across K=1..8. Same CE quality as the loop doing nothing extra. Source: `.agents/notes/archived/architecture/2026-08-30-dbfix-program.md`, `.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md` reference table.

### A5. `dbfix` — faithful one-pass DB training step on TUL slots (`_tul_core_db1`, `d9e04e6:morph/model/transformer.py:2428-2472`)

**What is noised:** the slot CORE-SEED state `h_0 = core_init(e)`, i.e. exactly the tensor `_tul_core` calls `h` before its first loop iteration — NOT a separately-observed clean target (MORPH has no directly-observed "clean state"; supervision comes only through the coda's downstream CE/mux loss). `h_0` is used BOTH as the noised variable `z_σ = h_0 + σ·ε` AND as the clean conditioning input the core layer's `DiagonalInjection`/x0-injection reads (unchanged path).

**σ sampling:** per BATCH SAMPLE `[B]` (not per-slot — flagged as an unresolved design choice), drawn from `DB1Sampler` (`iter_cond.py`), a TUL-local equal-mass log-normal restricted to TUL's own `(σ_min, σ_max, p_mean, p_std, σ_data)` config keys, deliberately NOT reusing `diffusion_blocks.DBSchedule`/`DBConfig` (whose σ_min/σ_max are module-level GLOBALS shared with the unrelated whole-model DB arm).

**Core application:** the core layer stack runs EXACTLY ONCE (`_apply_core_step` called once, not T times) — the entire point of the paper's recurrent-depth mode (App. B: "reducing computational cost by factor K"). `stage_cond = tul_stage_cond.stage_embed(c_noise)` feeds AdaLN-Zero modulation (`CoreStageConditioning`, reusing `SigmaConditioning`/`AdaLNGate` from `diffusion_blocks.py` verbatim) into `_apply_core_step`.

**Loss:** NO explicit denoising loss (no L2, no separate CE-on-D̂). The returned `h` (the EDM-combined `c_skip·z_σ + c_out·f_out`) lands exactly where `_tul_core`'s `h_slots` lands, and `_forward_tul`'s pre-existing `elif db_traj is None:` branch supervises it with the SAME weighted-CE/mux machinery every other TUL arm uses. Supervision is entirely implicit/downstream.

**Inference:** `_tul_core_db1_ladder` (`d9e04e6:transformer.py:2499-2547`) — K conditioned applications, σ stepping σ_max→σ_min via `euler_step` (reused from `diffusion_blocks.py`), fixed-seed(0) initial noise `z = h_0 + σ_max·ε` (NOT pure noise `z_0=σ_max·ε` — deliberately keeps the seed's real per-span content, since training never saw a σ_max sample with zero seed signal).

**Result — FAILS depth gate, inverted.** Pre-registered bar: `CE(K=6) ≤ CE(K=1) − 0.02`. Measured: K=1 **4.4652**, K=6 4.4992 — monotonically WORSE with more Euler steps (K=8 4.5101). CE@4250 = 4.4521 (passed its own bar of ≤4.46), wall-clock 37.7 min (vs l2cap's 66 min — the fast-and-stable-but-inert profile). Source: `lab/experiments/successes/2026-08-30-tul-dbfix-pair.md`.

### A6. `db_cond` — old `db_loop`/l3 + iter-indexed AdaLN conditioning (the minimal conditioning falsifier)

Same as A4 but with `CoreStageConditioning` in "iter" mode (`stage=t`, the raw loop-iteration index, not σ) added to the existing unconditioned unrolled loop — isolates whether conditioning ALONE (without the one-pass σ/EDM training-step change) can wake a dead loop.

**Result — FAILS, worse than dbfix.** Pre-registered bar: `CE(K=1) − CE(K=6) ≥ 0.02`. Measured: 4.4102 − 4.4061 = 0.0041; full spread K=1..8 only 0.005 nats — dead flat (4.4102 / 4.4065 / 4.4061 / 4.4052 / 4.4055 / 4.4061 / 4.4060 / 4.4060). CE@4250 = 4.3584 (within 0.010 of l2cap's 4.3489, inside replicate spread), 68.5 min. Source: same success filing.

**Combined verdict (both A5+A6 fail → binding rule fired):** "faithful DB does not transfer to TUL slot geometry at this budget. Interleave CANCELLED." Only gradient-through-the-iterated-map under contractivity control (`l2cap`: full BPTT + hard σ≤1.5 spectral projection, unrelated to DB) has ever earned depth on TUL (0.233 nats). Three DB-flavored mechanisms (l3/db_loop target scheduling, dbfix σ+EDM one-pass, db_cond iter-AdaLN) all produce stable good-CE-but-inert-or-inverted loops. Source: `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md`.

### A7 (adjacent, not DB but reuses its machinery). FM1 planner — flow matching, no loop, `n_core=0` (`morph/model/fm_planner.py`, `tul_fm.py`, current tree, off by default)

Not a DB arm (conditional flow matching, `objective: cfm`, not EDM denoising) but the closest thing on the current tree to a "no-loop, diffusion/flow-machinery, slot-level" design — the direct precursor substrate for dmorph. Reuses `DBSchedule`, `EDMPrecond`, `SigmaConditioning`/`AdaLNGate`, `euler_step` VERBATIM from `diffusion_blocks.py`; explicitly does NOT reuse `SliceScaler` (unit-L2 targets instead, to avoid the σ* autoencoding trap — see B). A small 4-layer SwiGLU transformer denoises the POOLED next-span representation from noise, conditioned via cross-attention on frozen `A3` (`n_core=0`) backbone features, trained with true CFM. Requires `tul.activate_at != never` and `model.n_core == 0` at build (`fm: enabled: false` in `base.yaml:526-549` builds nothing).

**Result — dead, closed by binding rule.** Even retrained on the best-available substrate (`l2cap` slot states, which independently measure as span-wide, informative carriers: `worth_profile` shuffle-worth +0.22 at offset 0), the P1 planner's retrieval probe reads within-row top-1 **0.0167** (bar ≥0.06) and MRR **0.0858** (bar ≥0.12) — BELOW even its earlier a3-substrate readings (0.0235/0.0423) — and indistinguishable from the untrained (0.0205) and shuffled-context (~chance 0.0196) controls. "The planner learned nothing retrievable at all." Source: `lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md`; closing note `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md`; arc `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md`.

---

## B. Failure modes and derived rules, each with its number

1. **σ\* autoencoding trap.** `σ* ≈ ‖y‖/4.2` is the noise level above which the target can no longer be nearest-neighbour-decoded straight out of `z=y+σ·ε`; below it the model reaches ~0 loss by autoencoding its own input and learns no language. MORPH's `SliceScaler` (per-component-std scaling to `σ_data=0.5`) pushed `σ*` to **3.30**, putting **77–98%** of the training σ-mass into pure autoencoding (measured: unit-norm+σ_data=0.5 gives σ*=0.235, 42% mass below it; MORPH's RMS-slice scaling gives σ*=3.30, 98%; the actual `db_b1_ce` reshaped schedule still gives σ*=3.30, 77%). Self-consistent fix found but never adopted on the whole-model line: unit-norm target with `σ_data = 1/√d_model`. Source: `.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md` post-audit addendum; memory `db-sigma-star-autoencoding.md`.

2. **The argmax bridge, not the softmax/expected-embedding bridge.** `softmax(logits) @ E` is a convex combination of embedding rows; an uncertain model returns a vector whose norm SHRINKS toward the table mean (`‖D̂‖=0.266` at σ=80 against the training-time `‖y‖=1.0`, an ~4x-too-weak, off-manifold signal). Bridging through the ARGMAX row instead (always full norm, on-manifold) cut gen-PPL by an order of magnitude: soft bridge 4/16/32 Euler steps → 1610/584/596; hard bridge → **55.5/22.9/17.1** (real-text anchor 34.77). Source: memory `db-bridge-is-the-bug.md`; also in `expected_embedding()`'s docstring risk-R9 note, `diffusion_blocks.py:585-602`.

3. **σ_max is the metric, not the σ-grid mean.** The grid mean is dominated by low-σ rows a model can win by autoencoding (rule 1); it ranks arms BACKWARDS. `db_b1_rms` has the BEST grid mean (2.8400) in the whole testbed and is the WORST model — its σ_max CE (10.3916) sits on the uniform floor `ln V=10.8249`. `db_b4` has the worst grid mean of the good arms (2.8487) and the best σ_max CE (4.6740). Rule: always report CE at σ_max against the AR baseline and `ln V`; treat a grid-mean "win" as autoencoding until σ_max agrees. Source: memory `db-high-sigma-is-the-metric.md`.

4. **Low-σ / EDM-weighted-CE collapse.** Applying EDM's `w(σ)=1/c_out²` (derived for L2) to cross-entropy over-weights the trivial low-σ region (`c_skip≈1` → `z` passes through → CE≈0) by up to **~62,000x** (config comment) / measured **45,560x** (commit message `6ebb4e5`), so training collapses to the low-σ copy and never learns high-σ prediction (train loss→0, val≈ln V). Rule (`DBConfig.loss_weighting="unweighted"` default): plain unweighted mean CE, EDM `w(σ)` reserved for L2 only. Source: `diffusion_blocks.py:147-155`, checklist §2.

5. **High-σ regression-to-mean (L2-specific, distinct from #1).** Under embedding-space L2, the L2-optimal prediction of a many-valid-next-token target is `E[y|context]` — a MEAN embedding, decodable to no token — so at high σ the model is measurably WORSE than a zero (chance) predictor (model rawL2 0.371 vs chance 0.25 at σ=9.79). This is intrinsic to L2-in-embedding-space, independent of conditioning (x0_inject≡concat) and independent of leakage (ruled out 3 ways: isolated-mask perturbation probe → 0.0 movement; full-forward flip-and-measure watcher → 0.0 (x0_inject) / ~1e-5 bf16 noise (concat), against a positive-control leak of +16). **This finding REVERSED the Option-A(L2)-over-Option-B(CE) call**: CE degrades gracefully toward `ln V` at high σ instead of collapsing to a garbage mean, so distributional CE (Option B / A3) is the objective that escapes L2's high-σ failure. Source: `db-b1-l2-pathfinder.md`, "The real diagnosis" section.

6. **x0_inject is structurally incapable of conditional prediction — objective tweaks cannot fix it.** `z_σ = y + σ·ε` IS a noised copy of the target the model reads; denoising `z_σ→y` is strictly easier than reading x0 context, so the model always has and takes the shortcut, regardless of detach fixes, CE-anchor terms, or σ-reshaping (all tried and all still gave matched≈scrambled gap ≈0 at every σ). Only a structurally separate stream that WITHHOLDS `y` from the query (concat, masked) can even attempt conditional prediction — and even concat measured identically to x0_inject (rule 5's finding subsumed this once the L2 objective itself was shown to be the cause, not the conditioning delivery mechanism). Source: `db-b1-l2-pathfinder.md` "FIX RUN" section.

7. **"Conditioning banned" (dmorph binding rule).** Neither σ-conditioning delivered as iter-AdaLN (`db_cond`, A6) nor as the paper-faithful σ+EDM one-pass (`dbfix`, A5) rescues TUL's slot loop: `db_cond`'s depth spread is 0.005 nats (dead flat), `dbfix`'s Euler ladder is monotonically WORSE with more steps (K=1 4.4652 → K=8 4.5101, inverted). Rule derived: "conditioning is not the missing ingredient" for depth-earning on TUL slot geometry at this budget. Source: `2026-08-30-tul-dbfix-pair.md` Verdict.

8. **"Interleave cancelled" (binding sequencing rule).** The dbfix program's rule was: interleave (`tul_ilv50`, mixing db1-style one-pass steps with full-BPTT l2cap steps) may only run if the faithful DB step first validates a depth curve (≥0.02 nats). Both D2 (dbfix) and C2 (db_cond) failed their depth gates, so the both-fail branch fired: `tul_ilv50` and `tul_l2cap_cond` were built but **never run**; they remain as built-but-unrun config artifacts (`morph/configs/`, though on the CURRENT tree those configs were later deleted along with the rest of `iter_cond.py`'s call sites — see part C/D). Source: `.agents/notes/archived/architecture/2026-08-30-dbfix-program.md`.

9. **DB sampling is inert / point-mass collapse.** At the position generation actually draws from (the end of the Euler walk), every DB arm's next-token distribution is a point mass: top-1 prob 1.0000, entropy 0.0000 nats, nucleus@0.95 = 1 token (AR baseline: 0.2920 / 4.7613 nats / 8422 tokens). This is NOT trajectory-wide collapse — `db_b1` measures real entropy along the walk (3.32/4.74/5.04/7.72 nats at σ=0.3/1/10/80) — it is exactly what EDM prescribes as `σ→0, c_skip→1` (denoiser becomes the identity). Consequence: temperature and top-p are inert for any DB-style readout; a diversity-guard (distinct-2) must always accompany a gen-PPL number, since a repetition loop can score gen-PPL 1.46 (better than real text's 32.44) by emitting `"the the the..."` (distinct-2 0.016 vs real text's 0.983). Source: memory `db-sampling-is-inert.md`, verdict note's "no token-level randomness" section.

10. **B=1 (recurrent-depth) is DB's weakest setting.** At matched per-block updates (8750), `db_b1`'s CE@σ_max=7.3011 vs `db_b4`'s (B=4 blocks) 4.6740 — B=4's curve is nearly flat from σ=1 to σ_max where B=1 collapses across it. Every earlier finding using only B=1 arms describes the weakest variant of the method, not the method generally — do not generalize "DB doesn't work" from B=1 evidence alone (though the FULL verdict, using B=4 too, still rejects DB against AR). Source: memory `db-b1-is-the-weakest-setting.md`.

11. **DB→AR checkpoint conversion loses, is not a lobotomy.** Best conversion route (noisy-stream fine-tune) reaches CE 4.3042 vs an AR-continued control's 3.9551 — 0.35 nats worse for 2.28x the wall-clock (the two-pass concat forward runs at 59k tok/s vs AR's 135k). The ability is present but not linearly readable: the clean stream alone starts at the uniform floor (10.8405 ≈ `ln V`=10.8249) yet still recovers to 4.8969 within 2000 steps, because context was only ever optimized as K/V material for the noisy stream. Source: memory `db-ar-conversion-loses.md`.

12. **The GLA retention-carry leak risk (concat arms only, unproven exploited but never fully ruled out).** The concat arms seed the noisy stream's GLA scan with the CLEAN stream's FINAL state, which integrates every target token. All formal "no-leak" proofs covered only the attention branches; the GLA path was never separately probed on the shipped `retention: true, retention_layers: [1]` config (the unit tests build `retention=False`). The gate inits near zero and a falsifier showed gap≈0 at 2000 steps (probably unexploited), but this is a code-reading finding, not a runtime-verified one. Source: verdict note's post-audit addendum.

13. **Identity-escape law (the umbrella rule for both objective lines).** Both FM and DB died the same way: they gave the optimizer an alternative to building composition THROUGH the iterated map — FM by routing plan formation around the loop entirely, DB by replacing iteration with conditioning-plus-inference-recurrence trained on a single pass. Only gradient through the realized composition (full BPTT, `l2cap`) ever earned depth. Any future depth recipe must keep the composition inside the training graph — a no-loop design (dmorph) sidesteps this problem entirely by declining to claim depth-earning at all. Source: `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md`.

---

## C. Reusable on the CURRENT tree vs. must be restored from `d9e04e6` / elsewhere

### Present and importable today (`223cf85`)

| Symbol | File | Notes |
|---|---|---|
| `DBConfig`, `DBSchedule`, `EDMPrecond`, `SigmaConditioning`, `AdaLNGate`, `SliceScaler`, `clean_noisy_mask`, `euler_step`, `expected_embedding` | `morph/model/diffusion_blocks.py` (602 lines, unchanged shape) | Whole module intact. Only `DBConfig(mode="b1", ...)` construction on the live tree is inside `fm_planner.py` (schedule reuse, not a training path). `SliceScaler` is explicitly NOT reused by `fm_planner.py` (rule 1). |
| `DBSchedule`, `EDMPrecond`, `SigmaConditioning`/`AdaLNGate`, `euler_step` reused by a live (if disabled) arm | `morph/model/fm_planner.py` (1055 lines), `morph/model/tul_fm.py` (291 lines) | FM1 planner — CFM/EDM-flavored, no-loop (`n_core=0`), σ-conditioned via the same AdaLN machinery. Off by default (`fm.enabled: false`, `base.yaml:530`). Its own objective (A7) is rejected, but the wiring around it (a construction-time no-loop denoiser sitting on TUL slot features, requiring `n_core==0`) is the closest working precedent for dmorph's shape. |
| `n_core == 0` coreless path | `morph/model/transformer.py::_core_region`, lines ~1448-1451 | "prelude output flows straight to the coda... n_core==0 → the whole loop machinery... must NOT run... Used by seed models." This IS the no-loop MORPH baseline dmorph would extend — already exists, already tested (it's the `A3`/"coreless" control arm referenced throughout the notes, CE@4250=4.3102, ~30 min/4500 steps). |
| `lab/divergence/core_depth_sweep.py`, `lab/divergence/worth_profile.py`, `scripts/tul_samples.py` | `lab/`, `scripts/` | Depth-sweep harness (varies Euler/loop steps and reads CE), stratified per-token-offset worth scorer with bootstrap CI, and the slot-budget-aware sampler — all reusable measurement tooling regardless of objective. |
| `tests/test_diffusion_blocks.py`, `tests/test_tulfm_p1.py` | `tests/` | Unit coverage for the still-present modules. |
| `docs/tul-fm-probing.md`, `docs/tul-paid-loop-recipe.md` | `docs/` | Present, current. |

### Deleted from the current tree — must be restored from `d9e04e6` (or earlier) if dmorph wants them

| Symbol | Where it lived at `d9e04e6` | What it did |
|---|---|---|
| `CoreStageConditioning` | `morph/model/iter_cond.py` (whole file, 187 lines, GONE) | σ- or iteration-conditioned AdaLN-Zero modulation, one gate per core layer; zero-init bit-identity. Wraps `diffusion_blocks.SigmaConditioning`/`AdaLNGate`. |
| `DB1Sampler` | same file | TUL-local equal-mass log-normal σ sampler + EDM precond, deliberately scoped away from the whole-model `DBSchedule`'s global σ_min/σ_max. |
| `iter_stage_value()` | same file | `[1]`-tensor helper for the "iter" conditioning mode. |
| `_tul_core_db1` | `morph/model/transformer.py:2428-2472` at `d9e04e6` | The one-pass DB training step (A5 above). |
| `_tul_core_db1_ladder` | `morph/model/transformer.py:2499-2547` at `d9e04e6` | The deterministic Euler-ladder eval (A5's inference side). |
| `_tul_db1_precheck` | `morph/model/transformer.py:2396-2422` at `d9e04e6` | Shared guard raising on db1 × SCSE / db1 × gate / db1 × core_gain_clip combinations left undefined. |
| `tul_step_mode` (forward arg, values `None`/`"bptt"`/`"db1"`) | `morph/model/transformer.py`, `_forward_tul` dispatch, `d9e04e6:2790-2932` | Per-forward selector between the plain loop, the db1 one-pass step, and (at eval on a σ-conditioned model) the auto-firing Euler ladder. GONE — current `_forward_tul` (`transformer.py:1958`) has no such parameter. |
| `training.step_mix` / `build_step_mix_cycle` | `morph/training/train.py` at `d9e04e6` (and earlier) | General interleave-schedule primitive (not DB-specific) used to alternate db1-style steps with full-BPTT l2cap steps; never actually exercised beyond smoke because the interleave was cancelled (rule 8) before it ran live. GONE from current `train.py` (`grep` returns nothing). |
| `tul_dbfix.yaml`, `tul_db_cond.yaml`, `tul_l2cap_cond.yaml`, `tul_ilv50.yaml` | `morph/configs/` at `81a9674`/`d9e04e6` | Four configs; only `wordbridge_d512.yaml` with no DB relation survives in `morph/configs/` today — none of the four exist. |
| `core_stage_cond` / `db1_sigma_min` / `db1_sigma_max` / `db1_p_mean` / `db1_p_std` / `db1_sigma_data` / `db1_cond_dim` / `db1_ladder_steps` (TULConfig fields) | `morph/model/tul.py` at `d9e04e6` | `grep` on current `tul.py` for any of these returns nothing — TULConfig no longer has a `core_stage_cond` field at all. |
| Whole-model DB training/inference (`morph/training/db_setup.py`, `morph/inference/db_generate.py`, `db_b1_fidelity.yaml`, `db_b1.yaml`, `db_b3.yaml`, `db_b3_massvisit.yaml`, 4 `posttrain/` bridge modules) | `park/db-master-line` (master at `e72f84c`), `feat/db-objective-l2` | Never present on `perf/throughput-lever-stack`; would need a cross-branch cherry-pick, and the dmorph handoff note itself flags this as likely to conflict given how far TUL has moved. |

**Bottom line for a no-loop dmorph build:** the reusable substrate is (1) `diffusion_blocks.py`'s primitives (untouched, importable today), (2) the `n_core=0` coreless forward path (untouched, the actual "no loop" mechanism, already load-bearing for FM1), and (3) FM1's wiring pattern (`fm_planner.py`/`tul_fm.py`) as a worked example of "σ/CFM-conditioned no-loop module bolted onto TUL slot features" — even though FM1's own write-objective failed the retrieval gate (A7). Everything DB-specific that targeted the LOOP itself (`iter_cond.py`, `_tul_core_db1*`, `tul_step_mode`, `step_mix`, the four dbfix-family configs) is gone and would need restoration from `d9e04e6` — but a no-loop design has no loop to condition, so most of that machinery (which exists specifically to inject σ into iterated core applications) may not even be the right shape for dmorph; only `CoreStageConditioning`'s AdaLN pattern and `DB1Sampler`'s TUL-scoped σ sampler are architecture-agnostic enough to be directly reusable.

---

## D. What the dmorph handoff note assumed that is no longer true on the current tree

Handoff note: `git show feat/db-objective-l2:.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md` (Status: proposed, never merged to any branch that reached `223cf85`).

1. **"Reusable machinery" table entries `morph/model/iter_cond.py`, `_tul_core_db1`/`_tul_core_db1_ladder` in `transformer.py`, `training.step_mix` in `train.py` (commit `81a9674`) — ALL GONE.** The handoff note was written 2026-08-30 pointing at commit `81a9674` on `perf/throughput-lever-stack`; the SAME branch deleted all three at `223cf85` (2026-09-03, "Ship the paid loop as the TUL recipe; cut the slot-only arms"). Confirmed by `grep -rn "tul_step_mode|step_mix|CoreStageConditioning|DB1Sampler|_tul_core_db1"  morph/` returning nothing, and `morph/model/iter_cond.py` not existing as a file.

2. **"dmorph arm: TUL slot geometry, no core loop (n_core=0 path or single conditioned pass)" — the slot-only TUL geometry itself is gone.** `morph/model/tul.py`'s own header (current tree) states: "2026-09-03: the slot-only loop (arms A0/A1/A3), the span-length gate, the MUX head, SIGReg, the DB1 one-pass step, the GRT recurrence gate, the compaction-window arm, arm A4, the TG restriction and the `e_slot`/`content`/`bound` seed modes were removed after the 20k pair." The shipped TUL forward (`_forward_tul`, `transformer.py:1958`) is now "the PAID loop... tokens and slots are ordinary positions of ONE sequence... the SAME per-sample core loop `_core_region` runs over all of them." So "TUL slot geometry, no core loop" as the handoff describes it (gather slots out → apply something → scatter back, skipping tokens) no longer exists as a code path; a coreless dmorph today would use `n_core=0` over the WHOLE packed sequence (tokens AND slots), not a slot-only subset. Note `_forward_tul` still supports `n_core=0` — but only via the FM-planner branch (`fm_planner is not None`), which replaces the (now-absent) core loop entirely; there is no longer a bare "no planner, no core" TUL path with slots — TULConfig with `cfg.fm is None` and `n_core=0` and no planner is untested territory on current code (worth checking before assuming it "just works").

3. **`halt=True` / eval-time halting — also gone.** The `d9e04e6` `_forward_tul` took a `halt` kwarg (gated the db1×halt interaction). Current `_forward_tul` signature (`transformer.py:1958`) has no `halt` parameter at all.

4. **Docstring-level staleness inside `diffusion_blocks.py` itself.** The module's own header still points to `docs/diffusionblocks-morph-assessment.md`, `docs/diffusionblocks-experiment-sheet.md`, `docs/diffusionblocks-plan-of-action.md`, `docs/diffusionblocks-reference-audit.md` — all four were deleted at `938d2e9` (the whole-model DB removal) and never restored, even though the module file itself was reintroduced later (`cf41f8b`, "TUL-FM P1... on a frozen A3 backbone") to serve FM1. Anyone reading `diffusion_blocks.py` today and following its own doc pointers will hit missing files.

5. **The reference-numbers table (batch 6, 4500 steps, seed 1, eager) is still numerically valid as a historical record** — nothing in `223cf85` recomputed those CE/wall-clock figures — but the ARMS it names (`dbfix`, `db_cond`, `l3`) are no longer runnable configs on the current tree (their YAMLs are deleted, per part C), so the table is now read-only history, not a rerunnable baseline.

6. **What DID survive and is still accurate:** the "coreless nomask" flat-compute bar concept (CE@4250 4.3102, ~30 min) maps to today's `n_core=0` path and remains the right control arm conceptually, even though the exact old TUL-arm names (A0/A1/A3) that produced that number are gone from the code (config for reproducing it would need to be rebuilt against the paid-loop-era TULConfig). The adjacent-testbed pointer (`/home/wolfe/11-DiffusionBlocks-Testing`, clean DB-on-Llama ladder) is unaffected by any of MORPH's internal churn and remains fully valid — argmax-bridge, sigma_max-is-the-metric, B=1-is-weakest, σ*-autoencoding all still hold as reusable lessons for any future denoising objective on MORPH, loop or no loop.

---

## Key file/commit index

- `morph/model/diffusion_blocks.py` — current tree, 602 lines, intact.
- `morph/model/fm_planner.py`, `morph/model/tul_fm.py` — current tree, FM1 (off by default).
- `morph/model/transformer.py::_core_region` (n_core=0 branch), `::_forward_tul` — current tree.
- `morph/model/tul.py` header — 2026-09-03 deletion note, current tree.
- `d9e04e6:morph/model/iter_cond.py` — deleted-file recovery, `git show d9e04e6:morph/model/iter_cond.py`.
- `d9e04e6:morph/model/transformer.py` lines 2396-2547, 2790-2932 — `_tul_core_db1*`, dispatch.
- `.agents/notes/rejected/feature/2026-08-19-diffusionblocks-checklist.md` — the campaign plan (rejected as a plan, not as false).
- `.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md` — the whole-model verdict, deciding numbers.
- `.agents/notes/archived/testing/2026-08-19-db-b1-l2-pathfinder.md` — A1/A2/A3 findings (rules 1, 5, 6).
- `.agents/notes/archived/architecture/2026-08-30-dbfix-program.md` — A4/A5/A6 program summary.
- `lab/experiments/successes/2026-08-30-tul-dbfix-pair.md` — A5/A6 exact numbers.
- `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md` — closes both DB and FM lines, points to dmorph.
- `feat/db-objective-l2:.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md` — the handoff (part D subject).
- `lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md`, `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md` — A7 (FM1) closure.
- Memory files `db-*.md` in `~/.claude/projects/.../memory/` — rules 1-3, 5, 9-11.

## What I did NOT verify

I did not run any code (no `torch` import, no test execution) — this was a pure read/grep audit against git history and markdown, per the task's own framing ("audit... do not modify any file"). All numeric claims are quoted from committed notes/wandb-derived tables, not independently recomputed. I did not check whether `morph/model/tul.py` + `n_core=0` + no FM planner currently constructs a valid coreless-with-slots forward on `223cf85` (point 2 above flags this as untested territory, not confirmed broken or confirmed working) — that would require actually building a `MORPHConfig` and running a forward pass, which is outside a read-only audit's scope. I did not open `lab/experiments/results/2026-08-30-tul-dbfix-pair/*.json` (the raw depth-sweep/eval-history JSON); the numbers I cite come from the success-note's already-transcribed tables, which I take as accurate but did not cross-check against the raw JSON byte-for-byte.
