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
3. **No arm runs until every Phase-A gate passes and R3b is decided.** R3b is not a tuning question.
4. `slot_layout=None` and DB-off must stay bit-identical to today: step-0 loss **11.2379**.
5. Do not edit an `Expected` cell in the experiment sheet after its arm has started.

---

## 1. Decisions already made

| # | Decision | Basis |
| --- | --- | --- |
| D1 | `B = 3` over prelude \| core \| coda. Core stays weight-tied; no layers added. | MORPH's own seams; matches the paper's best cell (equi-probability σ + even layer split, Table 7) |
| D2 | **DB-B1 runs before DB-B3.** | B=1 is the setting the paper validated on Huginn; it isolates "does the objective work on MORPH at all" and dodges the per-block schedule stretch |
| D3 | Training samples σ **continuously**; only inference discretises into `1 + T + 1` Euler steps, `T ~ Poisson(6)` capped 8. | The denoiser conditions on σ, not Δσ. Preserves Poisson depth and TUL's per-slot depth |
| D4 | σ interval mass **1/8 : 6/8 : 1/8**; block visit frequency **uniform 1/3**, a separate knob. | Mass-proportional visits starve prelude/coda to 1/8 of updates. The paper conflates the two because it has no weight tying |
| D5 | Core conditioned on `x0` (post-embed, pre-prelude, `transformer.py:924`) — no prelude forward. | That is the paper's AR conditioning. 320 of 1024 ctx dims is wide enough (Wolfe's call) |
| D6 | HC stays **inside** a block; the σ-blend is the seam **between** blocks. | Eq (5) is an inter-block connection (App. C). The n=4 Cayley carrier is untouched |
| D7 | Slot denoising target = **next span's first token embedding** (T-a). | Keeps it a prediction target; span autoencoding is forbidden by the TUL spec |
| D8 | Use the **EDM** Euler sign, `α = σ_b/σ_{b-1} ∈ (0,1)`. | v4's rendered Eq (3)–(5) makes noise increase down the schedule |
| D9 | Loss is cross-entropy, not L2; EDM weighting `σ_data = 0.5`; overlap `γ = 0.1`. | App. B (AR case), App. C |

## 2. Decisions still open

| # | Question | Owner | Blocks |
| --- | --- | --- | --- |
| O1 | **R3b — the manifold question.** `y + σ·ε` leaves the hyperboloid. Noise in tangent space and exp-map back, or run the diffusion in ambient Euclidean coordinates with Lorentz only at readout? | Wolfe + A2 | **every arm** |
| O2 | Shifted-`x0` conditioning instead of the clean\|noisy concatenation? If it works, `pos/tok` halves everywhere and the new Triton mask disappears. Wolfe: "may work, not sure." | Wolfe, after A4 | scope of A5 |
| O3 | Do we run any DB arm on `base.yaml` (prune / carve / route live), or stay on `tul_short.yaml` (dense) for the whole campaign? | Wolfe | whether §4 mitigations are needed at all |

---

## 3. Phase A — no GPU training. Do all of this first.

| ID | Work | Done when | Depends on |
| --- | --- | --- | --- |
| A1 | **EDM sign check.** Read the authors' released code; confirm `α = σ_b/σ_{b-1}`. Record the file and line in the sheet §4.1 G3. | sheet G3 filled with a citation, not an opinion | — |
| A2 | **R3b manifold decision (O1).** Write up both options with the maths, pick one, record it. Cheap analytic work: what does adding isotropic noise to a Lorentz point do to `⟨x,x⟩_L`, and what does the tangent-space alternative cost per step? | a decision recorded in the sheet with its reasoning | — |
| A3 | **FLOP instrumentation (sheet G1).** Hand-written analytic FLOP model covering the Triton kernels. Promote `layer_passes_per_token` to every arm. Add `positions_per_token`, `flop_proxy`, `model_tflops`, `mfu`. Version the model and log the version. | A0 reports **exactly 44.0** passes/token and 1.0 positions/token; A1 reports 10.68 / 1.125 | — |
| A4 | **Attention-mask scoping (R1).** Read CCA / CSA / HCA / XSA and their Triton kernels. Answer: what does a clean\|noisy causal mask actually cost to add, and is shifted-`x0` (O2) a real alternative or does it break causality? | a written scoping note with an estimate, feeding O2 | — |
| A5 | **Implement the conversion, DB off by default.** `TULConfig`-style construction-time switches. AdaLN σ-conditioning, VE noising in embedding space with scale pinned per A2, equi-probability σ partition, EDM weighting + preconditioning, the D4 two-knob sampler, the D8 seam. | `pytest tests/` green **and** the 11.2379 parity marker holds with DB off | A1, A2, A3, A4 |
| A6 | **Schedule-stretch mitigations (§4).** Only if O3 says a DB arm touches `base.yaml`. | asserts in place, not comments | A5, O3 |
| A7 | **Bridge-metric harness.** Post-hoc, separate process: gen-PPL(GPT2-XL), MAUVE, rep4@512 from a checkpoint. Never in the training job. | runs on an existing A0-era checkpoint on CPU or in a short GPU window | A5 |

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
| B3 | DB-3 (`B=3` + TUL) | the TUL cross; does the T-a slot target train? | T-a gives no slot signal → fall back to T-c (DB-10) |
| B4 | DB-4 (`B=1` + TUL) | TUL cross on the cheaper conversion | — |
| B5 | DB-6 batch sweep 14 → 28 | **P1**: turn the memory win into tok/s | no DB arm came in ≤ 16 GB |

All Phase-B arms run at `tul_short.yaml` shape (seq 1024 × batch 14 × 20k steps) so every cell pairs
with the measured anchors. Budget at the expected step times: B1 ≈ 3.9 h, B2 ≈ 2.2 h, B3 ≈ 1.9 h,
B4 ≈ 3.0 h. **~11 h of GPU for the four arms**, plus the batch sweep.

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
