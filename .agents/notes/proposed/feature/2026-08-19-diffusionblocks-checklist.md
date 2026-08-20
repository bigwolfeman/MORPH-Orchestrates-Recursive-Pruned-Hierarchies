# Agent Note: Diffusionblocks Checklist

Status: proposed

Origin: Ai-notes/diffusionblocks-checklist.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# DiffusionBlocks-on-MORPH task list

## The objective fork — decide before anything else

The paper's autoregressive-text objective is **L2 in embedding space, not
cross-entropy on a readout** (App. B line 1275 + App. C). The target is
`y = normalize(embed(token))`; the loss is `w(σ)·‖D_θ(y+σε) − y‖²`; tokens appear
only at generation, by mapping the denoised embedding back to a token (App. E.4:
"4 diffusion steps with greedy sampling"). The paper uses CE only for the ViT
classifier and the masked-diffusion LM — never for AR text.

The killed run used **CE on the weight-tied readout** (`logits = denoised @ E.T`).
That is a MORPH choice borrowed from the ViT case. The **free-ride pathology is a
direct consequence of that choice**, not a property of the method: EDM
preconditioning makes the network's raw output regress a unit-variance target at
every σ, so under L2 no σ decodes for free. CE + a tied readout lets `c_skip·z`
(≈ the clean target at low σ) pass into the readout and score CE≈0.

Two options. Pick one before writing code.

| Option | Objective | Free ride | Notes |
| --- | --- | --- | --- |
| **A (faithful)** | L2 embedding denoising (the paper's AR method) | none, by construction | matches the setting the paper validated on a looped model (Huginn LM1B); readout is unused in training; PPL still non-computable |
| **B (MORPH extension)** | CE on an **untied** denoising head | must be engineered away (untie head + reshape σ) | richer per-token signal, keeps a CE number; fully off-paper — label it a MORPH variant, not "DiffusionBlocks reproduced" |

**Order: A first, then B.** A is the pathfinder because it removes the free ride
with no head surgery and reproduces a validated result. B is the follow-on for
signal richness, gated on A working. The A→B transition is section 14. The
execution order across all sections is section 15.

## 0. Freeze the fidelity contract

- [ ] Pin the paper version, authors’ repository commit, and Huginn version.
- [ ] Create a conformance matrix: **paper exact**, **necessary MORPH adaptation**, and **later experiment**.
- [ ] Resolve four missing-reference questions before implementation:
  - What exactly runs once in recurrent-depth training: core only or prelude+core+coda?
  - Is the clean stream σ-conditioned?
  - Is the AR output head tied or independent?
  - How do Huginn’s inference iterations map onto MORPH’s prelude/core/coda?
- [ ] Use the paper’s Block Diffusion reference to freeze the clean/noisy mask.
- [ ] **Archive Block Diffusion [2] (Arriola et al.) locally and read its mask before building any mask.** The paper outsources the exact rule (`≤i` vs `<i`, whether `clean_{i+1}` is withheld, whether noisy positions self-attend) to that reference — our paper text does not pin it (re-read §5). Writing the mask from our paper alone is guesswork.
- [ ] Treat TUL integration as a MORPH extension, not part of paper fidelity.
- [ ] **Stand up the bridge-metric harness (MAUVE + generative PPL from a teacher model) and produce the A0 reference row BEFORE the first arm.** This is a precondition, not a falsifier. Adopting the diffusion objective makes val-CE non-comparable to the entire A0 ledger (assessment §4.1: "not recoverable by cleverness"). Without the bridge and its A0 baseline, no arm can be judged at all. (Duplicated as a hard gate in section 15, Phase 0.)

## 1. Configuration and isolation

- [ ] Add DiffusionBlocks as a construction-time model variant, off by default.
- [ ] Add a dedicated `db_b1_fidelity.yaml`.
- [ ] Keep `tul.activate_at: never` in the first fidelity run.
- [ ] Preserve the existing non-TUL and TUL model paths.
- [ ] Reject unsupported DB+TUL combinations explicitly until implemented.
- [ ] Ensure DB-off remains bit-identical and constructs no DB parameters.
- [ ] Keep all MORPH features available. Disable features only in isolated falsifier configs.

## 2. Implement the paper’s exact diffusion math

- [ ] Implement the exact log-normal σ distribution and constants (`P_mean=−1.2`, `P_std=1.2`, `σ_min=0.002`, `σ_max=80`, `σ_data=0.5`, `γ=0.1` for text).
- [ ] Implement equi-probability σ partitioning.
- [ ] Implement EDM `c_skip`, `c_out`, `c_in`, and `c_noise` (from Karras EDM [29]; the paper does not restate them — re-read §1).
- [ ] Use one sampled σ per sample.
- [ ] Use the same σ throughout the selected block’s forward pass.
- [ ] Implement the authors’ descending Euler update and inference initialization (`z_0 = randn·√(1+σ_max²)`, sign settled to `α=σ_b/σ_{b-1}∈(0,1)` — audit §2).
- [ ] **Objective loss follows the fork, not a single fixed rule (re-read §1):**
  - Option A: **L2** `w(σ)·‖D_θ−y‖²` with the EDM weight `w(σ)=(σ²+σ_data²)/(σ·σ_data)²`. Here the EDM weight is correct — it is derived for L2.
  - Option B: **CE on an untied head**, default **UNWEIGHTED**. Do NOT apply `w(σ)` to CE by default: `w(σ)` equalizes an L2 loss, has no such meaning on CE, and empirically collapsed the v1 run (train→0, val≈lnV — `db_setup.db_loss`). Keep `w(σ)·CE` only as a labeled ablation.
- [ ] Log unweighted CE (or raw L2) separately, and **per block index**, so one failing block is visible instead of averaged away.
- [ ] **Embedding scale = per-slice scaling to `σ_data·√(slice_dim)`, NOT whole-vector unit-L2.** Whole-vector unit-L2 at `d=1024` gives per-component std ≈0.031 while `σ_data=0.5` — a ~16× mismatch (audit §5) that keeps `c_skip≈1` across most of the sampled σ range and is a co-cause of the free ride. The per-slice rule (euclidean and Lorentz-tangent slices scaled independently) is already decided (audit §6 / plan O1) and makes `σ_data=0.5` literally true. Apply the same transform to the tied/bridge head weight.
- [ ] Keep custom temperatures out of the fidelity baseline.

## 3. Remove the tied-head shortcut (Option B only — Option A has no head)

Under Option A (L2) there is no readout in training, so this whole section is
moot: the free ride cannot occur. This section applies only if we choose Option B
(CE). Note the paper does NOT state whether the AR readout is tied or independent
(the defining line 1275 is truncated in our markdown — re-read §3); there is
nothing to "reconstruct," so this is a design decision we make, not a fidelity
match.

- [ ] Implement an **independent** learned denoising/output head. Do not tie it to the diffusion target embedding — the tie is what created the free ride.
- [ ] Keep the diffusion target embedding separate from the classifier parameters.
- [ ] Implement the sampler bridge as `softmax(logits) @ normalized_input_embeddings` (audit §4 — this maps CE logits back to embedding space for the next Euler step).
- [ ] Verify that low-σ targets cannot be decoded for free at initialization (fixed-σ CE probe, section 9).
- [ ] Test for embedding collapse with pairwise cosine statistics.
- [ ] **Record that x0_inject's earlier "autoencoding" was the free ride, not an x0 leak.** The loader gives `labels[t]=input[t+1] ≠ x0[t]=embed(input[t])`, so x0_inject never leaked; it failed at high σ because the free ride starved the context signal. This is why the objective fix (this section + section 2) must be validated on the cheap x0_inject path BEFORE any concat/two-source work (section 6/7) — the concat machinery does not fix the objective.

## 4. Implement correct σ conditioning

Fidelity caveat: the paper only says σ-conditioning is done "e.g., via AdaLN"
(DiT [42]) and gives **no per-layer detail and no gate-init values** (re-read §4).
The DiT specifics below are a reasonable imported design, not a fidelity
requirement — label them as our choice.

- [ ] Build one timestep embedding from `c_noise`.
- [ ] Send that same embedding to every layer in the active denoiser block.
- [ ] Implement DiT-style modulation:
  - attention norm shift and scale;
  - attention residual gate;
  - MLP norm shift and scale;
  - MLP residual gate.
- [ ] Initialize modulation and residual gates DiT-style (adaLN-Zero: gates start at 0 → block starts as identity).
- [ ] **SCOPED RESEARCH TASK, not a checkbox: resolve the residual-gate ↔ Cayley-HC conflict.** DiT's residual gate scales the residual branch, but MORPH's `HyperConnectionResidual` (Cayley n=4) was chosen for exact ρ=1 dynamical isometry (assessment §4.4/§6). A learned gate inside the block fights that design. Options to evaluate in a separate note: (a) gate only the norm shift/scale, not the residual branch; (b) apply the σ-blend contraction only at the block seam (`α<1`), leaving HC isometric inside; (c) accept the gate and re-measure `ρ(J_core)`. Do not wire a residual gate onto the Cayley carrier until this is decided.
- [ ] RMSNorm has no mean-centering, so an adaLN "shift" acts differently than on LayerNorm — verify the shift still helps rather than just re-introducing a bias RMSNorm deliberately drops.
- [ ] Do not assign different σ values to different layers.
- [ ] Do not advance σ inside a training forward.
- [ ] **v1 fallback is honest but weaker:** σ-conditioning at section boundaries only (plan-of-action §3b) is strictly weaker than per-layer modulation. If used first, never report it as "the paper's method," and upgrade it the moment σ-conditioning looks like the active ingredient.

## 5. Implement recurrent-depth B=1 first

- [ ] Implement one denoiser pass during training.
- [ ] Remove the recurrent core loop and BPTT only from this training path.
- [ ] Preserve the normal MORPH path unchanged.
- [ ] Confirm which prelude/core/coda sections belong to the B=1 denoiser.
- [ ] Restore recurrence only through Euler steps during generation.
- [ ] Verify gradients reach every parameter that belongs to the B=1 denoiser.

## 6. Implement autoregressive clean/noisy conditioning

- [ ] Build `clean_i = embed(input_i)`.
- [ ] Build `noisy_i = normalize(embed(target_i)) + σ·ε`.
- [ ] Permit noisy position `i` to use only legal clean context.
- [ ] Prevent access to `clean_{i+1}`, which contains its target.
- [ ] Decide from the references whether noisy self-attention includes `i` or only `<i`.
- [ ] Use MORPH’s required two-pass realization:
  - normal clean stream;
  - normal noisy stream;
  - merge clean and noisy K/V during noisy attention.
- [ ] Keep CCA convolution, value shift, and RoPE local to each stream.
- [ ] Keep the clean pass gradient-bearing.
- [ ] Checkpoint both streams without detaching clean context.

## 7. Adapt every MORPH context mechanism (Phase 2 — only after the objective works)

Scope note: use the paper's **two-pass** form (App. E.4 alternative: "compute
key-value pairs separately for clean and noisy … two forward passes … standard
sequence memory"). That form is **memory-safe today** — the `[B,H,S,2S]` OOM the
killed run hit was an SDPA math-backend fallback under an explicit additive mask,
not a fundamental cost. A **fused** two-source kernel is a later throughput
optimization, NOT a gate on correctness.

- [ ] Add two-source support to CSA compressed attention (merged block bank).
- [ ] Add two-source support to HCA compressed attention.
- [ ] Never construct an `[S, 2S]` dense score matrix. Reference the two-pass form; do not call SDPA with an explicit `[S,2S]` additive mask (that forces the quadratic math backend).
- [ ] **Deferred optimization, not a prerequisite:** a fused two-source local-window kernel. The eager/two-pass window is memory-safe; build the fused kernel only if the window becomes the throughput bottleneck, and profile before writing it.
- [ ] Audit XSA self-exclusion under the two-source layout.
- [ ] Audit x0, value, and bigram injections for target leakage.
- [ ] Implement causal clean-prefix GLA state per noisy position.
- [ ] Never seed noisy GLA with the final all-token clean state.
- [ ] Add fused/eager numerical parity tests.

## 8. Loss and memory path

- [ ] (Option B) Route training through chunked/fused CE. (Option A has no CE — the loss is L2 in embedding space.)
- [ ] (Option B) Avoid materializing `[B, S, V]` logits during training.
- [ ] Use the reduction the fork prescribes (section 2): Option A `mean(w·‖D_θ−y‖²)`; Option B `mean(CE)` unweighted by default, `mean(w·CE)` only as a labeled ablation. Do NOT copy the authors' `mean(w·CE)` blindly — their w is for L2 (re-read §1).
- [ ] Support ignored and padded labels without changing weighting.
- [ ] Measure forward/backward peak after compilation warmup.
- [ ] Profile SDPA/Triton backends to prove no quadratic fallback.
- [ ] Compare memory only at the same commit, config, batch, and checkpoint policy.

## 9. Correctness gates

- [ ] Unit-test σ sampling against analytic quantiles.
- [ ] Unit-test EDM coefficients and weights against the authors’ code.
- [ ] Unit-test Euler steps with a perfect denoiser.
- [ ] Unit-test target alignment with next-token labels.
- [ ] Mutation-test every future clean position against every earlier noisy output.
- [ ] Run no-leak tests with GLA, bigrams, value embeddings, and fused kernels enabled.
- [ ] Require zero logical leakage; do not accept baseline mask bleed as normal.
- [ ] Verify gradient flow through clean K/V and every σ-conditioning module.
- [ ] Verify DB-off output, loss, parameters, and gradients are bit-identical.
- [ ] Verify useful gradient across predefined σ bins — **this is the real check.** A near-zero **denoiser** gradient at low σ is CORRECT diffusion behavior (`c_out→0`: there is nothing to denoise), not a bug. Do not gate on low-σ gradient being nonzero. Gate on: the model learns in the mid-to-high-σ regime where prediction actually happens, and (Option B) the readout does NOT decode the target for free at low σ.
- [ ] (Option B only) Verify low-σ initialization CE does not collapse to zero through the readout — the free-ride check. This is a readout property, distinct from the denoiser-gradient point above.

## 10. Training falsifiers

- [ ] Overfit one tiny sequence with resampled σ.
- [ ] Overfit a tiny language corpus.
- [ ] Compare matched clean context against batch-scrambled context.
- [ ] Log fixed-σ CE and gradient norm at several σ values.
- [ ] Run the complete Euler chain during evaluation.
- [ ] Confirm that additional Euler steps improve rather than degrade generation.
- [ ] Evaluate MAUVE and teacher-model generative perplexity.
- [ ] Do not compare diffusion reconstruction CE directly with ordinary LM perplexity.
- [ ] Run equal-token and equal-compute comparisons against MORPH.

### Pre-registered kill criteria (write the numbers in the sheet before the arm starts)

An arm without a stated stop condition is not a gate. Minimum set:

- [ ] **Objective gate (Phase 1):** the matched-vs-scrambled falsifier must show clear context use **at high σ** — matched CE (or L2) clearly below scrambled AND below the chance baseline. If matched ≈ scrambled at high σ, the objective is still autoencoding; stop and fix the objective, do not proceed to concat.
- [ ] **Bridge gate (Phase 1/2):** DB-B1 bridge quality (MAUVE + gen-PPL) must be within **15% of the A0 reference row at equal tokens**. Remember the paper's headline Huginn win used 3× the epochs (assessment §2) — equal-token is the honest bar. Outside the band is a finding to record, not a band to widen.
- [ ] **Block-independence gate (Phase 3, B>1):** DB-B3 quality must be within **10% of DB-B1**, else block independence is not free.
- [ ] **Slot-target gate (TUL):** pre-register a number (e.g. slot-position denoising loss at σ_min must improve on its step-500 value by step 5k); if not, T-a slot targets give no signal → fall back.

## 11. Restore the full MORPH stack

After B=1 clears its gates:

- [ ] Enable HyperConnections and verify conditioned residual gates.
- [ ] Enable production CCA/CSA/HCA/XSA kernels.
- [ ] Enable causal GLA prefix-state integration.
- [ ] Enable hybrid embedding quantization and ternary QAT.
- [ ] Enable TST and verify target alignment.
- [ ] Enable pruning, carve, and routing.
- [ ] Adjust step-counted schedules only where measured update frequency requires it.
- [ ] Repeat no-leak, gradient, generation, memory, and throughput gates after each addition.

## 12. Add block-wise B>1

- [ ] Start only after B=1 generation works.
- [ ] Use the paper’s equal-probability σ ranges and uniform block visits first.
- [ ] Do not use the earlier custom `1:6:1` partition in the fidelity arm.
- [ ] Define MORPH block boundaries without untying the recurrent core.
- [ ] Prove only the selected block receives gradients.
- [ ] Ensure optimizer state and activation memory reflect the claimed saving.
- [ ] Validate block transitions with complete Euler generation.
- [ ] Add overlap `γ=0.1` only as specified for text.

## 13. Add TUL support

- [ ] Keep a fully supported no-TUL DB path.
- [ ] Define diffusion targets for token and slot positions.
- [ ] Ensure slot targets remain predictive rather than span autoencoding.
- [ ] Build clean/noisy layouts containing identical causal slot placement.
- [ ] Define clean-token, clean-slot, noisy-token, and noisy-slot visibility.
- [ ] Preserve TUL’s “core over slots only” contract.
- [ ] Add mutation tests for token↔slot leakage.
- [ ] Compare DB without TUL against DB+TUL using complete generation metrics.
- [ ] Keep DB+TUL disabled with an explicit error until all gates pass.

## 14. Transition from A to B (only after A is verified)

Do this **only after Option A (faithful L2) has cleared its objective gate and its
bridge gate** (section 10). A is the proof that the diffusion objective learns
language on MORPH at all. B is the extension that trades fidelity for a richer
per-token signal and a CE number. Do not start B to "skip ahead" — if A never
passes, B inherits A's problem plus new ones.

Entry condition (all must hold):

- [ ] A's matched-vs-scrambled falsifier passed at high σ (real context use).
- [ ] A's DB-B1 bridge quality is within 15% of the A0 reference row at equal tokens.
- [ ] A's full Euler generation chain runs and additional steps help, not hurt.
- [ ] The A run and its config are logged to wandb and recorded in the sheet.

The transition itself:

- [ ] Freeze A as a named baseline arm (`db_b1_L2`) — its checkpoint and bridge row become B's comparison anchor. B must beat or match A on the bridge, or B is not worth its extra complexity.
- [ ] Build the **independent** denoising/output head (section 3). Keep the diffusion target embedding separate from it. Do not tie.
- [ ] Switch the loss to **unweighted CE** through the readout (section 2, Option B). Keep the L2 loss available behind the construction switch so A stays runnable.
- [ ] Re-run the fixed-σ probe and confirm the free ride is absent with the untied head (section 9, Option B check).
- [ ] Wire the sampler bridge `softmax(logits) @ normalized_embeddings` (audit §4) so generation still produces tokens; verify the Euler chain still works.
- [ ] Reshape / verify the σ sampling if the fixed-σ probe shows most mass in the trivial low-σ region — the informative-σ fraction is the lever, and P_mean=−1.2 was tuned for images + L2, not AR + CE.
- [ ] Re-run the objective gate and the bridge gate for B (`db_b1_ce`). Compare B vs A head-to-head on the bridge; keep whichever wins as the DB-B1 arm going into section 11+.
- [ ] Record honestly that B is a MORPH extension, not a reproduction of the paper's AR method.

## 15. Execution phases (the order to actually run this)

This is the sequencing across all sections above. It exists because the killed run
front-loaded the expensive concat/two-source work before the objective was proven.
Fix the objective on the cheapest path first; add machinery only after each gate.

- [ ] **Phase 0 — decide and prepare (no GPU training).**
  - Pick the objective fork (A first; the fork table at the top).
  - Archive and read Block Diffusion [2]; write the exact mask rule down (section 0).
  - Build the bridge harness (MAUVE + gen-PPL) and produce the **A0 reference row**. Precondition for judging anything.
  - Land sections 1 (config isolation), 2 (diffusion math), 4 (σ-conditioning, v1 section-boundary form allowed), 5 (recurrent-depth B=1), on the **cheap x0_inject conditioning** — no two-source kernel yet.
  - `CUDA_VISIBLE_DEVICES="" pytest tests/` green; DB-off CPU forward-parity test passes.

- [ ] **Phase 1 — prove the objective on x0_inject (the gating experiment).**
  - Run Option A (L2) DB-B1 on x0_inject. Fixed-σ diagnostic (section 9) + matched-vs-scrambled falsifier at high σ (section 10).
  - **Objective gate + bridge gate (section 10).** If it fails here, the concat rebuild would not have saved it — fix the objective, do not add machinery.

- [ ] **Phase 2 — richer context, only if Phase 1 passes.**
  - Upgrade x0_inject → clean|noisy concat using the **two-pass** (separate K/V) form (section 6, 7). Memory-safe; defer any fused kernel.
  - Re-run the no-leak gates (section 9) and the falsifier. Confirm concat beats x0_inject on the bridge, else keep x0_inject.

- [ ] **Phase 3 — A→B transition, then faithfulness and scale.**
  - Run section 14 (A→B) if we want the CE variant.
  - Full per-layer σ-conditioning, resolving the adaLN-gate ↔ Cayley-HC conflict as its own note (section 4).
  - Restore the full MORPH stack (section 11), then B>1 (section 12), then TUL (section 13).

The first implementation milestone ends when Phase 1 clears its gates on x0_inject.
Concat (Phase 2), B>1, and TUL do not enter the critical path until DB-B1 produces
a working Euler generation chain that demonstrably uses context.
