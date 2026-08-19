# DiffusionBlocks — plan of action

**Scope.** Bring DiffusionBlocks ([arXiv:2506.14202](https://arxiv.org/abs/2506.14202), ICLR 2026)
into MORPH as `B = 3` over the existing prelude / core / coda seams, cross it with TUL, and measure
tok/s, FLOP efficiency and VRAM against the recorded A0 / A1 anchors.

**Companion documents — do not duplicate their content here:**

| Document | Role |
| --- | --- |
| [`diffusionblocks-morph-assessment.md`](diffusionblocks-morph-assessment.md) | design rationale; what the paper does and does not say |
| [`diffusionblocks-experiment-sheet.md`](diffusionblocks-experiment-sheet.md) | **pre-registration**: arms, expected numbers, metric contract, kill criteria |
| [`references/training-objectives/2506.14202.md`](references/training-objectives/2506.14202.md) | full paper text + PDF |
| [`ablation-ledger.md`](ablation-ledger.md) | "Planned — DiffusionBlocks" rows; A0 / A1 / A3 measured anchors |

---

## 0. Hard constraints

1. **THE GPU IS BUSY.** Another project owns the 5090 (~25 GB resident, ~94 % util) and Wolfe is
   maximising its time over a two-week window. **Start no training. Kill nothing.** Read-only
   `nvidia-smi` queries are fine. Every item in Phase A is CPU-only or CPU-light by design.
2. Wolfe is on a 10 A circuit with a dying UPS. Heavy CPU load while the GPU is at 100 % trips it.
   No large parallel builds, no multi-core sweeps, while the other job runs.
3. **No arm runs until every Phase-A gate passes and the R3 scale rule is written down.**
   **Reviewer note (2026-08-19):** R3b as first written over-stated the blocker. `morph/model/embeddings.py`
   log-maps the Lorentz point to the origin tangent space INSIDE the embedding module
   (`LorentzEmbedding.forward = log_map_origin(project_to_hyperboloid(·))`), so the network — and
   therefore the target `y` — lives in ambient Euclidean coordinates and `y + σ·ε` never touches a
   hyperboloid point. The live blocker is R3a: per-slice scale. The euclidean slice inits near unit
   norm; the Lorentz tangent slice inits at std 0.005; one global `σ_data = 0.5` drowns the Lorentz
   slice at every σ. See O1.
4. `slot_layout=None` and DB-off must stay bit-identical to today: step-0 loss **11.2379**.
   **Reviewer note:** checking this number needs a GPU forward at the arm shape with the Triton
   kernels — it is a **Phase-B step-0 gate**. The Phase-A parity gate is a CPU forward-parity test
   in the `tests/test_tul_forward.py` pattern (see A5). Do not claim the marker "holds" before a GPU
   window has actually produced the number.
5. Do not edit an `Expected` cell in the experiment sheet after its arm has started.
6. `pytest tests/` is **not** GPU-free: `test_kernel_compile_fences.py` takes CUDA when visible.
   During the busy window run `CUDA_VISIBLE_DEVICES="" pytest tests/` (the CUDA tests skip cleanly);
   the full suite on GPU is a Phase-B precondition, not a Phase-A gate.

---

## 1. Decisions already made

| # | Decision | Basis |
| --- | --- | --- |
| D1 | `B = 3` over prelude \| core \| coda. Core stays weight-tied; no layers added. | MORPH's own seams; matches the paper's best cell (equi-probability σ + even layer split, Table 7) |
| D2 | **DB-B1 runs before DB-B3.** | B=1 is the setting the paper validated on Huginn; it isolates "does the objective work on MORPH at all" and dodges the per-block schedule stretch |
| D3 | Training samples σ **continuously**; only inference discretises into `1 + T + 1` Euler steps, `T ~ Poisson(6)` capped 8. | The denoiser conditions on σ, not Δσ. Preserves Poisson depth and TUL's per-slot depth. **Verified against the paper**: App. E.5 trains the recurrent model single-pass with σ sampled per step; App. E.1 discretises at inference only (50 / 1000 / B steps in different sections) |
| D4 | σ interval mass **1/8 : 6/8 : 1/8**; block visit frequency **uniform 1/3**, a separate knob. | Mass-proportional visits starve prelude/coda to 1/8 of updates. In the paper the two rules coincide (equal-mass partition + uniform visits); they separate here because our mass split is unequal — see the note below |
| D5 | Core conditioned on `x0` (post-embed, pre-prelude, `transformer.py:924`) — no prelude forward. | That is the paper's AR conditioning. 320 of 1024 ctx dims is wide enough (Wolfe's call). **Reviewer verified:** `x0 = x.clone()` sits after `embed_drop(embed(ids))` and before HC expansion / prelude (`_front_tail`) — note it is post-dropout; ctx slice width 320 confirmed (`channel_dims: [512, 320, 192]`) |
| D6 | HC stays **inside** a block; the σ-blend is the seam **between** blocks. | Eq (5) is an inter-block connection (App. C). The n=4 Cayley carrier is untouched |
| D7 | Slot denoising target = **next span's first token embedding** (T-a). | Keeps it a prediction target; span autoencoding is forbidden by the TUL spec |
| D8 | Use the **EDM** Euler sign, `α = σ_b/σ_{b-1} ∈ (0,1)`. | v4's rendered Eq (3)–(5) makes noise increase down the schedule. **Reviewer re-derived and confirmed:** the sign error starts at Eq (3) — a correct Euler step of Eq (1) from σ_{b-1} down to σ_b reads `z + Δσ·σ_{b-1}∇log p`, the paper renders `−`. Decisive self-check: a perfect denoiser must land at `y + σ_b·ε`; only `α = σ_b/σ_{b-1}` does. Authors' code exists ([github.com/SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)) but the README suggests ViT-only — its sampler still settles the seam sign |
| D9 | Loss is cross-entropy, not L2; EDM weighting `σ_data = 0.5`; overlap `γ = 0.1`. | App. B (AR case), App. C |

**Reviewer note on D4 (arithmetic verified, framing corrected).** The arithmetic holds:
1/8 : 6/8 : 1/8 gives the `1 + T̄ + 1 = 8` inference steps equal mass at T̄ = 6, and
`E[passes/token] = (4 + 6 + 4)/3 = 4.67`. But uniform block visits are **not** a departure from the
paper — App. E.1 states *"blocks are sampled uniformly at random for each iteration."* The paper's
partition is equal-mass, so uniform and mass-proportional visits coincide there; they separate here
only because our mass split is unequal. **The actual departure from the paper's stated rule is the
1/8 : 6/8 : 1/8 mass split itself** (the paper gives every block 1/B of the mass). DB-12 still
isolates the visit knob; DB-9 covers the partition.

## 2. Decisions still open

| # | Question | Owner | Blocks |
| --- | --- | --- | --- |
| O1 | **R3 — the scale rule, not the manifold.** The manifold half is closed by the code: `embeddings.py` log-maps Lorentz to the origin tangent space inside the module, so "ambient Euclidean with Lorentz at readout" is not an option to choose — it is what MORPH already does by construction (reviewer, 2026-08-19). Still open and still blocking every arm: pin the per-slice scale so `σ_data` is a known number. Euclidean slice ≈ unit norm; Lorentz tangent slice inits at std 0.005; bigram adds on top. Per-slice L2 vs per-slice `σ_data`; Wolfe signs off. | Wolfe + A2 | **every arm** |
| O2 | Shifted-`x0` conditioning instead of the clean\|noisy concatenation? If it works, `pos/tok` halves everywhere and the new Triton mask disappears. Wolfe: "may work, not sure." | Wolfe, after A4 | scope of A5 |
| O3 | Do we run any DB arm on `base.yaml` (prune / carve / route live), or stay on `tul_short.yaml` (dense) for the whole campaign? | Wolfe | whether §4 mitigations are needed at all |

---

## 3. Phase A — no GPU training. Do all of this first.

| ID | Work | Done when | Depends on |
| --- | --- | --- | --- |
| A1 | **EDM sign check.** Read the authors' released code ([github.com/SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks) — README suggests ViT-only, but the sampler settles the seam sign); confirm `α = σ_b/σ_{b-1}`. Record the file and line in the sheet §4.1 G3. If the repo's sampler is absent or ambiguous, the gate falls back to the Eq (1) derivation plus a unit test in OUR code asserting a perfect denoiser lands at `y + σ_b·ε` — the maths is decisive on its own; the code is confirmatory. | sheet G3 filled with a citation (file+line) or the derivation + passing landing test | — |
| A2 | **R3 scale decision (O1).** The manifold question is closed (`embeddings.py` outputs tangent-space coordinates; see O1). Remaining work: measure the per-slice norms of the `HybridEmbedding` output (CPU forward of the module alone — minutes, no GPU), then pick the scale rule (per-slice L2 vs per-slice `σ_data`) and record it. The rule must be measured on the **quantised** embedding output — `tul_short.yaml` inherits int6 `embed_quant` from base, and the anchors ran with it ON. | the sheet records the rule AND the measured slice norms it came from | — |
| A3 | **FLOP instrumentation (sheet G1).** Hand-written analytic FLOP model covering the Triton kernels. Promote `layer_passes_per_token` to every arm. Add `positions_per_token`, `flop_proxy`, `model_tflops`, `mfu`. Version the model and log the version. **Reviewer correction:** the counter must distinguish NOMINAL from REALIZED. Depth is `clamp(Poisson(6), 1, 8)` (`transformer.py::_sample_depths`), whose mean is **5.67, not 6** — a realized counter reports A0 ≈ 42.0, never "exactly 44.0", and A1's measured 10.68 IS a realized number. `perf/flop_proxy` stays nominal (analytic from the config, T̄ = 6 ⇒ A0 = 44.0 exactly); `perf/layer_passes_per_token` is realized. | nominal: A0 = 44.0 / 1.0 positions, A1 = 12.0 / 1.125 exactly; realized: A0 = 42.0 ± 0.5, A1 ≈ 10.68 | — |
| A4 | **Attention-mask scoping (R1).** Read CCA / CSA / HCA / XSA and their Triton kernels. Answer: what does a clean\|noisy causal mask actually cost to add, and is shifted-`x0` (O2) a real alternative or does it break causality? | a written scoping note with an estimate, feeding O2 | — |
| A5 | **Implement the conversion, DB off by default.** `TULConfig`-style construction-time switches. AdaLN σ-conditioning, VE noising in embedding space with scale pinned per A2, equi-probability σ partition, EDM weighting + preconditioning, the D4 two-knob sampler, the D8 seam. Plus the reviewer additions below this table. | `CUDA_VISIBLE_DEVICES="" pytest tests/` green now, full suite green in the first GPU window; a DB-off CPU forward-parity test (the `test_tul_forward.py` pattern) passes; the 11.2379 marker is confirmed at Phase-B step 0 | A1, A2, A3, A4 |
| A6 | **Schedule-stretch mitigations (§4).** Only if O3 says a DB arm touches `base.yaml`. | asserts in place, not comments | A5, O3 |
| A7 | **Bridge-metric harness.** Post-hoc, separate process: gen-PPL(GPT2-XL), MAUVE, rep4@512 from a checkpoint. Never in the training job. **Reviewer fix:** this does NOT depend on A5 — it needs only generation from an existing checkpoint plus teacher scoring. Build it early, in parallel: it also produces the A0 reference bridge row, which must exist before any DB arm can be judged (sheet §4.4). | runs on an existing A0-era checkpoint on CPU or in a short GPU window | — |

**A5 must also cover (reviewer additions, 2026-08-19) — none of these were in the original plan:**

- **Static CUDA graph capture.** `train.py` captures the front (embed+prelude) and back (coda+head)
  regions as CUDA graphs ("2 fwd + 2 bwd graph replays/step"). DB changes both regions' structure,
  and B=3 varies the live subgraph per step. Decide per arm: extend capture or disable it — and if
  disabled, say so next to every s/step comparison, because A0's 0.947 s (and the `a ≈ 0.156 s`
  floor fitted from it) includes graph replay.
- **torch.compile.** σ-conditioning must enter as a tensor, never a Python scalar (a scalar bakes
  into the guard set → recompile per σ). Keep `mode="default"`, no fullgraph, donated_buffer False.
  Expect one guard set per sampled block under B=3; verify no per-step recompiles in a smoke run.
- **Fused/chunked CE.** `use_kernels: true` routes loss through `fused_ce.py`. Under the clean|noisy
  concat, CE lands only on the noisy half's positions — reuse TUL's position/logit masking. The
  denoiser readout is the same tied `lm_weight()` projection the CE host already builds.
- **QAT stays ON.** `tul_short.yaml` inherits the ternary backbone + int6 `embed_quant` from base;
  the A0/A1/A3 anchors ran with them ON, so DB arms keep them ON or the pairing breaks. (Scale rule
  on the quantised embedding: see A2.)
- **Checkpoint compatibility.** DB off builds zero new parameters (the TUL `activate_at: never`
  pattern), so existing checkpoints load bit-for-bit. New DB parameters get their own namespace so
  `interop/checkpoint.py` and the posttrain mask tooling fail loudly, not silently.
- **Hydra configs.** One file per arm (`db_1.yaml` … `db_4.yaml`) composing `tul_short.yaml`, named
  in the ledger rows. The full resolved `db:` block — including constructor-hardcoded values — goes
  to wandb.
- **Val protocol.** DB arms log val CE at a FIXED σ grid (σ_min at minimum). A sampled-σ val CE is
  noise, not a curve.

**A3 is the gate on the whole campaign.** We cannot judge any arm on FLOP efficiency today —
`perf/mfu` does not exist and `layer_passes_per_token` is TUL-only. `FlopCounterMode` is present but
**blind to Triton**, so it cannot be the logged number; it only cross-checks the aten half. `ncu` is
installed for one-off validation. `nvidia-smi` has no FLOP counter.

## 4. Hazards to neutralise before any `B = 3` arm on `base.yaml`

A block sampled 1/3 of steps stretches every step-counted schedule we own. `tul_short.yaml` runs
dense with prune/carve/route **off**, which hides two of these until a DB arm hits `base.yaml`.

| Knob | Today | Risk | Action |
| --- | --- | --- | --- |
| `ademamix_t_alpha` | 8000 | ~1/3 the effective per-block updates | scale, or record that it was not and why |
| `ademamix_beta3_warmup_start` | 0.9 | same stretch, divergence-prone direction | re-derive per block |
| MORTAR `prune_start` / `prune_interval` | 3000 / 167 | Taylor saliency accumulates 3× slower | scale cadence **and** hard-assert logged `[prune] density=…` before carve |
| `compact_step` | 29000 | would carve a still-dense model → **K/C = 1.0** | hard assert (the CLAUDE.md gotcha) |
| `route_start` | 30000 | inherits the stretch | scale |

## 5. Phase B — the runs, when the GPU frees up

Order matters. Each arm answers one question. Expected numbers, bands and kill criteria are in the
sheet §4; do not restate them here.

| Order | Arm | Question | Stop if |
| --- | --- | --- | --- |
| B1 | DB-1 (`B=1`) | does the diffusion objective work on MORPH at all? | bridge quality > 15 % worse than A0 |
| B2 | DB-2 (`B=3`, `x0`) | is block independence free? (P2) | quality > 10 % worse than DB-1 |
| B3 | DB-3 (`B=3` + TUL) | the TUL cross; does the T-a slot target train? | T-a gives no slot signal → fall back to T-c (DB-10). **Reviewer note:** "no slot signal" must be pre-registered as a number in the sheet before B3 starts (e.g. slot-position denoising CE at σ_min not improving on its step-500 value by step 5k). An undefined stop condition is not a gate |
| B4 | DB-4 (`B=1` + TUL) | TUL cross on the cheaper conversion | — |
| B5 | DB-6 batch sweep 14 → 28 | **P1**: turn the memory win into tok/s | no DB arm came in ≤ 16 GB |

All Phase-B arms run at `tul_short.yaml` shape (seq 1024 × batch 14 × 20k steps) so every cell pairs
with the measured anchors. Budget at the expected step times: B1 ≈ 3.9 h, B2 ≈ 2.2 h, B3 ≈ 1.9 h,
B4 ≈ 3.0 h. **~11 h of GPU for the four arms**, plus the batch sweep.

**Reviewer note on the budget (2026-08-19).** The `a + b·proxy` cost model is a two-point fit
(A0, A3) and it MISSES the one held-out measured point: it predicts A1 at 0.37 s; A1 measured
0.544 s (+46 %). Cause: the proxy counts positions, launch overhead counts layer applications —
A1 still launches the full T×6 core stack on tiny slot tensors, plus TUL pack/gather work. DB
training is single-pass, so DB-1/DB-2 should track the fit better; DB-3/DB-4 inherit the TUL
overhead, and their sheet bands are widened accordingly (sheet §4.2). The static-graph question
(A5) moves `a` too. Treat the first ~200 measured steps of each arm as the band check before
extrapolating; the truthful budget range is **11–15 h**, not 11.

Phase C arms (DB-5, DB-7 … DB-13) are conditional; see the sheet §5. **DB-11 is independent of all of
this** — the σ-blend contraction with the DB objective off, which keeps ordinary CE and val PPL and
tests the `ρ(J_core) ≤ 1` handle as a Task #276 cure. It can run in any spare window and lands on
the existing ledger.

## 6. Reporting rules

- Fill sheet cells from wandb, not from memory. Cite the run name.
- **Never** put a DB arm's CE next to A0's CE. Two metric families, one bridge (sheet §1.3).
- A result outside its pre-registered band is a finding to write down, not a band to widen.
- Log the full config to wandb including anything hardcoded in constructors.
- Every claim of "faster" cites tok/s **and** flop_proxy **and** peak alloc together. Any one alone
  is misleading on a launch-bound model.
