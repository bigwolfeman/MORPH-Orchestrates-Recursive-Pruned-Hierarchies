# DiffusionBlocks, flow matching, and LeJEPA — source extraction for dmorph

Repo: `/home/wolfe/morph-perf`. No file in the repo was modified. All line numbers below are
from the files as they stand in the working tree at HEAD (`master`, matching the parent
session's git status).

## Sources found and read

- `docs/references/training-objectives/diffusionblocks/diffusionblocks.md` — the mirrored
  DiffusionBlocks paper (arXiv 2506.14202 v4), full text including all appendices. Read in
  full (1616 lines). The PDF beside it is marked authoritative for figures/equations; the
  `.md` states it is "machine-converted from the arXiv HTML (v4)."
- `morph/model/fm_planner.py` (1055 lines) and `morph/model/diffusion_blocks.py` — the
  repo's OWN code implementing both objectives (EDM/DiffusionBlocks-style and true
  conditional flow matching), built and run against a frozen MORPH backbone. This is the
  most exact ground truth in the repo for "flow matching as MORPH implements it" and is
  the direct precedent for a dmorph design.
- `docs/references.md` — searched for every entry on flow matching, rectified flow, LeJEPA,
  SIGReg, Coconut, Block Transformer, and diffusion LMs. No standalone "flow matching" or
  "rectified flow" reference entry exists in this file (see note in §B below). Relevant
  sections found: §7 (LeJEPA/SIGReg/LLM-JEPA, all "removed"), §9 (DiffusionBlocks,
  "TESTED, REJECTED"), §13 TUL-lineage entries (Coconut, Block Transformer,
  non-autoregressive/diffusion decoding: LLaDA, Block Diffusion/BD3-LM, Latent Diffusion
  for Language).
- `docs/references/regularization-objectives/lejepa/lejepa.md` — the mirrored LeJEPA paper
  (Balestriero & LeCun, arXiv 2511.08544), read for §4.2 (SIGReg / Epps-Pulley).
- `docs/references/tul-latent-emission/bd3-lm/bd3-lm.md` — Block Diffusion (Arriola et al.
  2025, arXiv 2503.09573). Mirrored, present, not the paper this task centers on, but the
  closest other diffusion-LM mirror. Skimmed for its own objective (discrete masking
  diffusion interpolating AR and diffusion via block-causal attention), not read in full.
- `docs/references/tul-latent-emission/llada/llada.md` — LLaDA (Nie et al. 2025, arXiv
  2502.09992), a masked-diffusion LM. Mirrored, present, skimmed only (not the task's
  target paper).
- `docs/references/tul-latent-emission/ld4lg/ld4lg.md` — Latent Diffusion for Language
  Generation (Lovelace et al. 2023, arXiv 2212.09462). Mirrored, present, not read in full.
- `docs/references/tul-latent-emission/explorative-modeling/explorative-modeling.md` —
  cites Lipman et al. 2022 (Flow Matching, arXiv 2210.02747) directly and states "every
  Diffusion model over continuous data in this work is trained with the Flow Matching
  objective" (line 105) — the one place the repo's mirrored corpus discusses Flow Matching
  as a named method, though only as background for a different paper (Explorative
  Modeling, arXiv 2607.27372), not as its own mirrored source.
- `docs/tul-fm-probing.md`, `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md`,
  `.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md`,
  `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md`,
  `lab/experiments/failures/2026-08-28-tulfm-p1c-objective-and-whitening.md`,
  `lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md` — the repo's own campaign
  against these two objectives, with measured numbers. This is where item D's answer
  actually lives.
- **Not mirrored, not in repo:** Lipman et al. 2022 "Flow Matching for Generative
  Modeling" (arXiv 2210.02747) and Liu et al. "Rectified Flow" (arXiv 2209.03003) have
  no dedicated file anywhere under `docs/references/`. Neither does a general "flow
  matching" entry exist in `docs/references.md`. I fetched the Lipman abstract from
  arxiv.org directly (quoted in §B); I did not fetch Liu et al., since the repo's own
  audited CFM code (§B) already gives the exact linear-path formulation the task asks
  about, and it matches Liu's rectified-flow formulation exactly.
- **Not in repo:** `.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md` is
  referenced BY NAME from two other notes (`docs/tul-paid-loop-recipe.md` line 263-266 and
  `.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md` line
  55-57) as living "on branch `feat/db-objective-l2`", but does not exist on the checked-
  out branch (`master`) or anywhere in the current working tree. State this to whoever is
  designing dmorph: the handoff note that should scope this exact task has not landed on
  `master` yet — it needs to be pulled from `feat/db-objective-l2` or rewritten.
- `docs/diffusionblocks-reference-audit.md`, referenced from `fm_planner.py`'s docstring as
  the audit of the vendored DB code, is also NOT in the current tree — it was removed with
  the rest of the `db_*` docs when DB was parked (per the verdict note's table, those four
  `docs/diffusionblocks-*.md` files live on `park/db-master-line`, not `master`). Only the
  surviving code module `morph/model/diffusion_blocks.py` (kept because `fm_planner.py`
  imports it) is still on `master`.

---

## A. DiffusionBlocks (arXiv 2506.14202 v4), exact mechanics

### A.1 Block partition — how many blocks, how the σ range is split

The formulation is Variance-Exploding: clean data `y ~ p_data` is perturbed as
`z_σ = y + σε`, `ε ~ N(0, I)` (diffusionblocks.md:111). Training minimizes the standard
EDM-style score-matching loss (Eq. 2, line 124):

```
L(θ) := E_{z0~p_data, σ~p_noise, ε~N(0,I)} [ w(σ) · ‖D_θ(y + σε, σ) − y‖² ]
```

**Step 1 — layer partition** (line 178): an `L`-layer network `F = {f_θℓ | ℓ∈[L]}` is cut
into `B` disjoint blocks `F = ⊎_{b=1}^B F_b`, block `b` owning layers
`{ℓ_{b-1}+1, …, ℓ_b}`; `f̄_θb` is the composition of the layers inside block `b`.

**Step 2 — noise range assignment** (line 181, formalized §3.3, line 221-234): a noise
distribution `p_noise` is chosen — log-normal, `log σ ~ N(P_mean, P_std²)`, following
Karras et al. 2022 (EDM) — over range `[σ_min, σ_max]`. The paper explicitly rejects the
naive uniform split `σ_b = σ_min + b·(σ_max − σ_min)/B` (line 223) because it "fails to
account for the varying difficulty of denoising at different noise levels." Instead it
uses **equi-probability partitioning**: choose boundaries `{σ_b}` such that each block
covers exactly `1/B` of the cumulative probability mass of `p_noise`:

```
∫_{σ_{b-1}}^{σ_b} p_noise(σ) dσ = 1/B                          (implicit, §3.3)
σ_b = exp( P_mean + P_std · Φ⁻¹(q_b) )                          (line 232)
q_b = q_min + (b/B)·(q_max − q_min)
q_min/max = Φ( (log σ_min/max − P_mean) / P_std )
```

`Φ⁻¹` is the inverse standard-normal CDF. Boundaries are narrower where denoising is
hardest (mid-range σ) and wider at the extremes (Fig. 4). §5.6.2 (Table 8, line 456-529)
ablates this against uniform: on CIFAR-10, equi-probability [4,4,4] gets FID 38.03 vs.
uniform [4,4,4]'s 43.53 — equi-probability wins across every layer split tried.

**Step 3 — noise conditioning** (line 185, Eq. 5, line 192-199): each block's input is
extended from `x` to `x̃ = (x, z)`, plus AdaLN noise-level conditioning, giving the
diffusion-block update rule

```
z_b = z_{b-1} + (Δσ_b / σ_{b-1}) · ( z_{b-1} − [f̄_{θb|σ_{b-1}}(x, z_{b-1})]_z )     (Eq. 5)
```

which is a rewritable convex combination `z_b = α·z_{b-1} + β·f̄_{θb|σ_{b-1}}(x, z_{b-1})`
with `α, β` functions of the σ ratio (line 199).

### A.2 What each block predicts, and the per-block loss/weighting

Each block predicts the CLEAN target `y` (not the noise, not a velocity) — `D_θb` in
Eq. 2's shape. The per-block objective is Eq. 6 (line 209):

```
L_b(θ_b) := E_{(x,y)~p_data, σ~p_noise^(b), ε~N(0,I)} [ w(σ) · Loss( f̄_{θb|σ}(x, y+σε), y ) ]
```

`p_noise^(b)` is `p_noise` restricted to `[σ_b, σ_{b-1}]` and renormalized; `Loss` is
typically L2 (line 213). The global loss decomposes exactly as `L = Σ_b L_b` (line 1381,
in the masked-diffusion derivation, which is the same additivity argument). Weighting
`w(σ)` (App. C, line 1299): `w(σ) = (σ² + σ_data²) / (σ · σ_data)²`, `σ_data = 0.5` for
every experiment — the paper calls this weighting "crucial for equi-probability
partitioning to work effectively" (line 1300), and it's literally the EDM weight (Karras
et al. 2022).

### A.3 Independent block training — the memory claim

"This independence enables training with gradients for only one block at a time" (line
71); "training with memory requirements for only `L/B` layers, storing activations only
for the active block" (line 216); "gradients are computed for only one block at a time"
(line 219). Training procedure (Appendix C, line 1308-1310): "blocks are randomly sampled
per iteration, requiring memory for only `L/B` layers. Blocks can alternatively be trained
in parallel across multiple GPUs when available." So yes — literally "train one block at
a time," with no gradient computed for the other `B-1` blocks on that step. Blocks are
sampled UNIFORMLY at random per training iteration (App. E preamble, line 1394), not
visited in a fixed cycle.

**Appendix G — vs. activation checkpointing** (line 1563-1573): the paper is explicit that
this is a DIFFERENT axis of savings than gradient checkpointing. Checkpointing only cuts
activation memory `A`; parameters/gradients/optimizer state (`4P` per layer under Adam:
`2P` for the weight+grad, `2P` for Adam's two moments) are untouched. Standard training:
`(4P + A)·L`. Checkpointing: `4P·L + A`. DiffusionBlocks: `(4P + A)·(L/B)` — it shrinks
ALL FOUR components (params, grads, optimizer state, activations) by `B`, and the two
combine multiplicatively: `(4P + A)·(L/B)` when stacked with checkpointing, "the least
memory among these four patterns" — because it deletes the other `B-1` blocks' parameters
and optimizer state from the step entirely, something checkpointing structurally cannot do.

### A.4 Train-time input of block `k` vs. inference-time input

**Training** (Eq. 5/6, Fig. 5): `z_{b-1} = y + σ·ε` for a `σ` sampled fresh from block
`b`'s own noise band, on the SAME clean `y` — each block sees noised GROUND TRUTH at its
assigned σ, independent of what any other block would have output. There is no chaining
through the other blocks during training.

**Inference** (Fig. 3 right / Fig. 6, App. C "Training and inference details," line 1308):
blocks are applied SEQUENTIALLY from `σ_max` down to `σ_min`, each block's OUTPUT `z_b`
feeding the next block's INPUT `z_{b-1}` per Eq. 5's Euler update — `z_0 = σ_max·ε` (pure
noise) is the starting point (line 200). So inference chains through the sequence of
blocks exactly like a discretized reverse-diffusion trajectory; training decouples them
completely. This train/inference mismatch (each block trained on ground-truth-derived
noise, but run at inference on the PREVIOUS BLOCK'S actual output) is not separately
discussed as a risk in the paper text — the repo's own audit (§A.6 below) treats it as
the standard EDM train/sample gap, nothing DiffusionBlocks-specific.

### A.5 The sampler

Euler discretization of the probability-flow ODE (§2.2, Eq. 3-4) is the ONLY sampler used
in every experiment; the paper states elsewhere (Future Works, line 543) that "other
diffusion samplers [DPM-solver, UniPC, DDIM] could be employed within blocks with modified
inter-block connections" but this was not tried. Concretely: DiT image experiments use
"Euler sampling with 50 steps and classifier-free guidance (scale 2.0)" (line 327, App.
E.2 line 1415); the ViT classification experiment uses "4 denoising steps during
inference (matching L/B = 12/3)" (App. E.1, line 1401); NoProp comparison uses "1000
Euler sampling steps instead of our default 50" (App. E.6.1, line 1448); Huginn recurrent-
depth inference keeps the ORIGINAL K-iteration procedure unchanged (App. B, line 1282 —
see A.7).

### A.6 Memory / FLOP / wall-clock claims — the actual numbers

**Training compute is claimed EQUAL to standard training, not cheaper** (Appendix H, line
1575-1581): "Standard end-to-end backpropagation performs `K×L` layer evaluations.
DiffusionBlocks trains only `L/B` layers at a time; training all `B` blocks for `K`
iterations each performs `(L/B)×B×K = L×K` layer evaluations. Thus, DiffusionBlocks
requires the same total amount of computation as standard training, while reducing
memory usage by a factor of `B`." Measured wall-time on a 12-layer ViT, H100 80GB (Table
12, line 1583-1604): baseline `0.0507 s/iter`; per-block DiffusionBlocks (4 layers)
`0.0181 s/iter`; aggregated (`0.0181 × 3`) `= 0.0543 s/iter` — i.e. essentially the SAME
wall time per equivalent unit of training, confirming the claim empirically. **The only
gain in this mode is memory (params+grads+optstate) by `B`, not FLOPs, not wall clock —
for training.**

**Inference compute DOES fall by `B`** for diffusion image models: "With 50 denoising
steps, a 12-layer DiT requires `12×50` layer evaluations. In DiffusionBlocks, each
denoising step applies only the block responsible for that noise level, which contains
4 layers when `B=3`. This reduces the total compute to `4×50`" (line 1611-1613) — a
`B`-fold reduction, because each denoising step only needs ONE block's parameters
resident, not the whole network. For AR/classification tasks the total inference compute
is held EQUAL to the baseline by construction: "the baseline performs a single forward
pass through all 12 layers. Under DiffusionBlocks with `B=3`, we perform three denoising
steps, each invoking the corresponding 4-layer block once. The total compute therefore
corresponds to the same 12 layer evaluations as in standard inference" (line 1606-1609).

**Recurrent-depth mode's claim is different and specific to §5.5/App. E.5** — see A.7.

### A.7 Metric caveats — exactly what was measured, per task

- **Vision classification (ViT/CIFAR-100, §5.1):** accuracy. ViT 60.25% vs. +DB 59.30%
  (Table 1, line 281-293).
- **Image generation (DiT/CIFAR-10, ImageNet-256, §5.2):** FID, train/test split, "minimum
  of three evaluations" (App. E.2, line 1415); train-set FID via the official ADM eval
  suite, test-set FID via clean-fid.
- **Masked diffusion LM (MD4/text8, §5.3):** bits-per-character (BPC).
- **Autoregressive text (Llama-2-style, LM1B/OWT, §5.4):** explicitly NOT perplexity —
  "computing traditional perplexity is non-trivial for our diffusion framework as it is
  not derived from ELBO" (line 427). Instead: **MAUVE**, following SEDD, plus **generative
  perplexity** scored by two teacher models (Llama-2-7B and GPT2-XL) on GENERATED text
  (App. E.4, line 1425-1432: 5 continuations × 50 tokens × 1K prompts, MAUVE scaling
  factor 2, top-p 0.95 baseline vs. "4 diffusion steps with greedy sampling" for DB).
  Table 4: LM1B AR 0.50 MAUVE / 14.58 (Llama-2 PPL) / 38.87 (GPT2-XL PPL) →
  +DB 0.71 / 12.32 / 30.99. OWT: AR 0.85 / 15.05 / 25.24 → +DB 0.82 / 14.99 / 26.33.
- **NoProp comparison (§5.6.1):** accuracy only, on CIFAR-100.

### A.8 Appendix E.5 — the recurrent-depth (Huginn) variant, in full

This is what MORPH's TUL work actually built and rejected, so it needs to be exact.

**Setup** (line 1436-1441): Huginn config — 2 prelude layers, 4-layer recurrent block, 2
coda layers, Pythia-70M-style architecture, 512 hidden dims, 8 attention heads. Trained on
LM1B.

**The mechanism, stated explicitly as NOT block partitioning** (line 1436-1441, and
Appendix B line 1282): *"Unlike other architectures, recurrent-depth models do not
require block partitioning since the entire network is applied recurrently. Instead, we
train the full network as a denoiser by sampling different noise levels σ at each
training step."* Appendix B's version: *"For recurrent-depth architectures that apply the
same network K times… we interpret the entire recurrence as a diffusion process. Instead
of training with K forward passes through recurrent iterations, we train the network as a
denoiser `D_θ(z_σ, x, σ)` by sampling `σ ~ p_σ` and performing a single forward pass to
map noisy input to clean output, reducing computational cost by factor K while maintaining
the original K-iteration inference procedure."*

So: **one conditioned pass per TRAINING step** (σ sampled once, single forward pass, no
BPTT); **recurrence is kept ONLY at inference**, unchanged from the K-iteration Huginn
sampling procedure. This is exactly the split the MORPH repo's own docs (CLAUDE.md,
diffusionblocks.md:6-17 provenance block) already describe correctly.

**Compute claim, specific numbers** (line 1441): baseline Huginn uses "stochastic
recurrence depth (average 32 iterations) with truncated BPTT (8 steps)"; DiffusionBlocks
trains with single-pass diffusion, on LM1B for **15 epochs vs. Huginn's 5 epochs**, and
"despite this, our approach uses approximately **10× less total computation** since we
avoid the 32× recurrent iterations during training." (i.e. 15 epochs of 1-pass training
vs. 5 epochs of ~32-iteration-average training with 8-step truncated BPTT: the paper's own
math is roughly `5 epochs × (8 backward-relevant iters, though forward still runs ~32)`
against `15 epochs × 1 pass` — the paper states the net factor as ~10×, does not show the
arithmetic that produces exactly 10 beyond this sentence).

**Result** (Table 5, line 372-388): Huginn baseline MAUVE 0.49 / Llama-2 PPL 17.04 /
GPT2-XL PPL 46.73 → +DiffusionBlocks 0.70 / 16.08 / 42.43. Text: "better performance on
LM1B for text generation while eliminating 32 iterations… demonstrates that our
framework enables fundamental training transformations beyond block-wise training" (line
434-435).

**What Appendix E.5 does NOT contain, and this is the load-bearing gap for dmorph's
design:** there is no depth-vs-quality curve, no ablation over the number of inference
iterations K, and no comparison against flat (non-recurrent) compute-matched baselines
anywhere in §5.5 or Appendix E.5. The section reports exactly one Huginn config, one
DiffusionBlocks-trained variant of it, and one epoch count comparison — a single-cell
result, not a sweep. This matches the MORPH repo's own finding, stated independently in
`.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md:50-51`:
*"the paper audit matters for the record: arXiv 2506.14202 never measured a depth-vs-steps
curve for its Huginn variant and never compared against flat compute — our K-sweep was the
first measurement, and it ran backwards."* I confirmed this by reading all of §5.5 and
Appendix E.5, B, C, E — no such curve exists anywhere in the paper.

---

## B. Flow matching / rectified flow as an objective

### B.1 What is and isn't mirrored

No dedicated file for Lipman et al. 2022 ("Flow Matching for Generative Modeling," arXiv
2210.02747) or Liu et al. ("Rectified Flow," arXiv 2209.03003) exists under
`docs/references/`, and neither has an entry in `docs/references.md`. I fetched the
Lipman abstract directly from arxiv.org (quoted below); the PDF's full text exceeded the
fetch tool's size limit, so the formulas below for the ORIGINAL Lipman paper come from
general knowledge of the paper, clearly marked as such, not from a local or fetched
source. Lipman abstract, verbatim:

> "We introduce a new paradigm for generative modeling built on Continuous Normalizing
> Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we
> present the notion of Flow Matching (FM), a simulation-free approach for training CNFs
> based on regressing vector fields of fixed conditional probability paths. Flow Matching
> is compatible with a general family of Gaussian probability paths for transforming
> between noise and data samples — which subsumes existing diffusion paths as specific
> instances. Interestingly, we find that employing FM with diffusion paths results in a
> more robust and stable alternative for training diffusion models. Furthermore, Flow
> Matching opens the door to training CNFs with other, non-diffusion probability paths.
> An instance of particular interest is using Optimal Transport (OT) displacement
> interpolation to define the conditional probability paths."

**What IS a fully-sourced, exact, code-level implementation of the same objective**: the
repo's own `morph/model/fm_planner.py`, built by a prior MORPH session specifically to
compare EDM-style DiffusionBlocks denoising against true conditional flow matching under
one shared harness (`docs/tul-fm-probing.md` §7: *"a genuine conditional flow-matching
arm (straight-line interpolation, velocity target, uniform t) must run beside it under
the SAME retrieval gate"*). I'm quoting this code verbatim below because it IS the
"conditional FM loss" and "linear path" the task asks for, already reduced to exact
formulas and matched against MORPH's own EDM implementation.

### B.2 The conditional FM loss, as implemented (`morph/model/fm_planner.py:915-948`)

```python
def _cfm_loss(planner, h_ctx, geom, *, generator, edges, y, loss_scale):
    """Conditional flow matching, straight-line (rectified) probability path.

    x_t = (1-t)*x0 + t*y has dx/dt = y - x0 exactly, so the regression target is
    a CONSTANT in t for a given (x0, y) pair.
    """
    x0 = torch.randn(y.shape, ...) * planner.cfg.source_std
    t = torch.rand((B, S), ...)                       # t ~ U(0, 1), independent per slot
    x_t = (1.0 - t[..., None]) * x0 + t[..., None] * y
    v_target = y - x0

    v_hat = planner.velocity(x_t, t, ctx, geom, self_mask, cross_mask)
    per_slot = (v_hat - v_target).pow(2).sum(-1)       # ‖v̂ - v‖²
    loss_raw = mean over valid slots of per_slot
```

So, exactly as the task frames it: `x_t = (1-t)·x_0 + t·x_1` (with the repo's `y` playing
the role of `x_1`, the data endpoint, and `x0 ~ N(0, source_std²·I)` the noise endpoint);
velocity target `v = x_1 − x_0` (a CONSTANT along the straight-line path — this is the
defining property of the linear/rectified path noted in the docstring); loss
`‖v̂ − (y − x0)‖²`, no EDM-style `w(σ)` weighting — the code comment says why: *"CFM's
target is already scale-uniform in t, so there is nothing for a w to equalise"*
(fm_planner.py:820-821). **t-sampling used: uniform, `t ~ U(0,1)` per slot independently**
(fm_planner.py:930), NOT logit-normal — the code comment at `band_edges_for`
(fm_planner.py:511-514) states this is deliberate so CFM's bands hold equal probability
mass by construction, matching the EDM arm's equi-probability σ bands one-for-one for a
fair side-by-side. (Logit-normal t-sampling for flow matching, e.g. in Stable Diffusion 3
/ Esser et al. 2024, is NOT what this repo implements or references anywhere — noting this
from general knowledge since the task asked to distinguish uniform vs. logit-normal, and
the repo's choice is uniform.)

### B.3 Relation to EDM denoising (x0-prediction ↔ velocity) — as measured in-repo

Both objectives are implemented side by side against the identical planner body, masks,
targets, and probe (`FMPlannerConfig.objective: "edm" | "cfm"`, fm_planner.py:566-572),
which is precisely how you'd A/B them for a dmorph design. The EDM branch
(fm_planner.py:885-912):

```python
def _edm_loss(planner, h_ctx, geom, schedule, *, generator, edges, y, loss_scale):
    sigma = schedule.sample_sigma(...)                  # truncated log-normal, per DBSchedule
    eps = torch.randn(y.shape, ...)
    z = y + sigma[..., None] * eps
    d_hat = planner.denoise(z, sigma, ctx, geom, self_mask, cross_mask)
    sq = (d_hat - y).pow(2).sum(-1)                      # ‖D̂ - y‖²
    w = planner.precond.weight(sigma)                    # EDM weight
    loss_raw = mean over valid slots of (w * sq)
```

`EDMPrecond` (`morph/model/diffusion_blocks.py:391-421`) is "Written as a tiny class
rather than four loose functions so σ_data cannot drift between the coefficients and the
weight," and gives the exact standard EDM preconditioning:

```python
c_skip = sigma_data² / (sigma² + sigma_data²)
c_out  = sigma * sigma_data / sqrt(sigma² + sigma_data²)
c_in   = 1 / sqrt(sigma² + sigma_data²)
c_noise = 0.25 * log(sigma)
weight(sigma) = (sigma² + sigma_data²) / (sigma * sigma_data)²      # matches paper's Eq. w(σ)
```

The module docstring (fm_planner.py:84-101) works out the "x0-prediction ↔ velocity"
correspondence explicitly by writing EDM in its preconditioned residual form
`loss = ‖F_θ − F_target‖²` with `F_target = (y − c_skip·z)/c_out`: at `σ → 0`,
`F_target → −ε` (the network is regressing pure noise, an untrainable floor of `d`); at
`σ → ∞`, `F_target → y/σ_data` (the network is regressing a rescaled copy of the clean
target — a floor of `‖y‖²/σ_data²`). This is functionally the same "the target rotates
from noise-like to data-like as σ/t moves across its range" structure that underlies why
EDM's `x0`-prediction and CFM's velocity prediction are two parameterizations of one
underlying probability-flow ODE — the repo's own analysis derives this from first
principles rather than citing the "Diffusion meets Flow Matching" equivalence paper
directly (that paper — Gao et al., "Diffusion meets flow matching: Two sides of the same
coin" — is cited only inside `explorative-modeling.md:452`, not independently mirrored).

**Repo's own measured answer to "does true FM match EDM":**
`lab/experiments/failures/2026-08-28-tulfm-p1c-objective-and-whitening.md` ran EDM and CFM
side by side under an identical retrieval-probe gate (frozen backbone, 4000-12000 steps).
Result table (within-row top-1 retrieval, chance = 0.0201):

| arm | top-1 | top-5 | MRR |
|---|---|---|---|
| edm_white (4k) | 0.0421 | 0.1568 | 0.1264 |
| cfm_raw (4k) | 0.0371 | 0.1694 | 0.1275 |
| cfm_white (4k) | 0.0510 | 0.1935 | 0.1452 |
| cfm_white (12k) | 0.0516 | 0.2048 | 0.1517 |

Verdict, quoted: *"C2 HELD: |0.0510 − 0.0421| = 0.0089 ≤ 0.0128. True CFM and EDM find
the same signal — the objective family is second-order."* i.e. at MORPH's scale, in this
harness, the choice between EDM-style denoising and true rectified-flow CFM did NOT
materially change what the objective could learn — both plateaued at the same ceiling
(§D below explains what that ceiling was and was not).

### B.4 Sampler / Euler integration

Both objectives are sampled with Euler integration in `generate_plans`
(fm_planner.py:953-1010). CFM: `x(0) ~ N(0, s²I)`, then `n_steps` forward-Euler steps of
fixed size `1/n_steps` on `dx/dt = v̂(x,t)` from `t=0` to `t=1`, plan = `x(1)`
(fm_planner.py:966-969, 982-990) — this is the standard K-step Euler ODE integrator for
flow matching, exactly matching the task's "K Euler steps" framing. EDM: the audited
`euler_step` from `diffusion_blocks.py:568-582`, `z ← α·z + (1−α)·D` with
`α = σ_next/σ`, over a strictly-descending equi-probability σ ladder, "NOT the paper's
rendered Eq (3)-(5), whose sign makes noise increase down the schedule" (a documented sign
bug the MORPH team found in the DiffusionBlocks paper's typeset equations, worked around
using the DiffusionBlocks AUTHORS' OWN reference code instead).

### B.5 Standard FLOP accounting: one training step vs. K-step inference

This is architectural, not paper-cited, but stated plainly by the repo's own docstrings
and matches the general FM/DB literature: training is **one forward pass per step, no
ladder** — `fm_loss` docstring: *"One P1 training step's loss. NO loop, NO BPTT — that is
the entire point"* (fm_planner.py:810). Inference costs `K` forward passes (`n_steps`,
a fixed hyperparameter, "Wolfe veto" against varying it per-example — fm_planner.py:961),
each a full pass through the (here, 4-layer) denoiser/velocity network. So the FLOP ratio
for one generated slot is `K : 1` between inference and training, matching the DB paper's
Appendix H accounting exactly for the recurrent-depth mode (§A.6 above): training does NOT
pay for the iteration count at all (single-σ, single-t, no ladder); only inference does.

---

## C. LeJEPA / SIGReg — the Epps-Pulley statistic

Source: `docs/references/regularization-objectives/lejepa/lejepa.md` (Balestriero & LeCun,
arXiv 2511.08544), §4.2 "SIGReg: Sketching the Epps-Pulley Test is Stable and Scalable"
(line 240), §4.2.3 "Characteristic Functions are Stable, Scalable and Identifiable" (line
320). Note: the paper's actual equations for SIGReg (Def. 2) and the Epps-Pulley statistic
are rendered as images in this arXiv-HTML-derived mirror and are marked "picture...
intentionally omitted" — so I cannot quote the boxed LaTeX formula verbatim; what follows
is the surrounding prose, which the mirror DOES carry in full and which states the
statistic precisely in words.

**Definition** (line 320-324): "The third family of tests is concerned with Empirical
Characteristic Functions (ECF) which are the Fourier transform of the density function.
The Epps–Pulley test [Epps and Pulley, 1983] is one of the most popular test and simply
compares in weighted ℓ2-norm the ECF of the data against a target CF... The ECF being
defined as `φ̂_X(t) = (1/n) Σ_{j=1}^n e^{itX_j}` is naturally differentiable and easily
computed in distributed settings via efficient `all_reduce` operations, as the ECF is a
simple average of complex exponentials. The weight function is typically Gaussian, such as
`w(t) = e^{−t²/σ²}` with σ commonly set to 1."

**Its role, claimed by the paper** — three properties argued to justify picking it over
moment-based (Jarque-Bera) or CDF-based (Cramér-von Mises, Anderson-Darling) tests (line
368): "(i) DDP-friendly and scalable, (ii) uniformly bounded gradients and curvature
regardless of input distribution, and (iii) hyper-parameter free implementation... our
implementation has a linear memory and computational complexity of `O(N)`, with `N` the
minibatch size." Stability claim (Theorem 4, cited but not quoted since its LaTeX is a
picture in this mirror): a bounded gradient norm `‖∇_θ EP(a)‖ ≤ (4σ²/N) Σ‖aᵀ∇_θ f_θ(x_i)‖`.

**SIGReg's overall role** (Definition 2, line 250): the Epps-Pulley statistic `T` is
sliced over random 1-D projections `a` — SIGReg averages `T` over projections `a ∈ A`
rather than taking the worst-case max (line 252, to avoid sparse gradients over the
directions in `A`), pushing the projected embedding distribution toward an isotropic
Gaussian. The paper's own aside (line 458): recovering VICReg as a degenerate special case
of LeJEPA by swapping in a mean/std statistic instead of Epps-Pulley, which the authors
"strongly advocate against... as it would lead to shortcut solutions."

**MORPH's own use, per `docs/references.md:407-425`:** "SIGReg uses randomized 1D
projections and characteristic-function matching to enforce that learned embeddings
follow an isotropic Gaussian distribution, preventing representation collapse with linear
time and memory complexity. MORPH applied SIGReg to z-latent embeddings; **removed**
together with the LeJEPA objective." The `tul-fm-arc.md` proposal (§ Proposal, line 52-53)
specifies the SAME regularizer as a planned-but-unbuilt guard for the FM planner's
targets: `"regularized with SIGReg (LeJEPA/LeVJEPA, lambda=0.02, M~1024 sketch
directions) so the target space cannot collapse without any EMA teacher or stop-gradient."`
This was never actually wired into `fm_planner.py` — grep confirms no `sigreg` or
`epps_pulley` symbol exists in that file; the FM planner's collapse guard in the code that
DID ship is `effective_rank` / `mean_pairwise_cos` diagnostics only (fm_planner.py:1015-
1055), reported but not optimized against.

---

## D. Does a denoising/FM objective replace depth or recurrence? — direct evidence, and where sources have none

### D.1 The DiffusionBlocks paper itself: NO depth-vs-steps measurement exists

Confirmed by reading all of §5.5 and Appendix E.5, B, C, E in full (§A.8 above). The paper
reports exactly ONE Huginn config trained ONE way (single-pass denoiser) and evaluated at
its ORIGINAL fixed inference depth (32 average iterations, unchanged) against the
baseline's OWN fixed depth. There is no ablation varying the number of inference
iterations `K`, no comparison at matched flat (non-recurrent) compute, and no measurement
of quality as a function of loop depth for the DB-trained Huginn variant. The paper's own
"Future works" section (line 541-548) proposes exactly this kind of investigation as
future work ("understanding why moderate block partitioning sometimes outperforms
end-to-end training warrants theoretical investigation") but for the block-count axis, not
depth-vs-iterations on the recurrent-depth mode specifically — that axis isn't raised at
all.

### D.2 MORPH's own K-sweep — the first depth-vs-steps measurement of this method, and it went backwards

`.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md:49-52`,
quoted in full: *"The loop-side resurrection tried three mechanisms — target scheduling
(l3), the paper-faithful σ+EDM one-pass with Euler-ladder inference (dbfix), iter-AdaLN
conditioning (db_cond) — all stable, all decent CE, all depth-dead or depth-inverted; the
50/50 interleave erased l2cap's curve outright. The paper audit matters for the record:
arXiv 2506.14202 never measured a depth-vs-steps curve for its Huginn variant and never
compared against flat compute — our K-sweep was the first measurement, and it ran
backwards."* "Depth-dead or depth-inverted" means: more inference iterations did not
improve quality, and in at least one arm made it WORSE — the opposite of what a working
recurrent-depth mechanism should show. **Binding rule from that note: "DB does not
transfer to TUL slot geometry at this budget."**

### D.3 The token-path replacement result (not depth, but the closest clean A/B)

`.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md` (full text read,
§A above for the DiffusionBlocks paper text, this is MORPH's clean-room reproduction):
at a matched 143.4M-token budget, plain next-token AR training reached held-out CE
**4.0010**; the best DB arm (`db_b1_oracle`, using the paper's own `σ_data=0.5`) reached
**5.0801** at `σ_max` — the regime where "the EDM skip path vanishes and a DB model is
doing nothing but next-token prediction from clean context," i.e. the fairest possible
comparison point. Quoted verdict: *"it simply does not train it as well as predicting
the next token does."* This measures denoising-as-training-objective vs. plain
autoregression, NOT depth vs. recurrence directly, but it's the decisive prior result the
K-sweep and the FM P1 campaign both build on.

### D.4 The flow-matching-as-write-objective result: it hit an information ceiling, not a depth question

`lab/experiments/failures/2026-08-28-tulfm-p1c-objective-and-whitening.md` and
`lab/experiments/successes/2026-08-30-tulfm-p1-l2cap.md` (both read in full, §B.3 above)
show that flow matching (EDM-style OR true CFM, tested side by side) COULD write real,
non-trivial, causally-conditioned content into a latent — but the amount was capped at
~2.5x chance retrieval regardless of whitening, objective family, or training steps, and
a **zero-parameter copy-the-current-span heuristic matched or beat the best 22M-parameter
trained planner** (top-1 0.0678 for the copy baseline vs. 0.0588 for the best trained
planner — quoted from the P1c file's addendum, §B.3/D.4 source). The final verdict,
quoted: *"The planner learned approximately 'the next sentence resembles this one' and
nothing measurably beyond it except a better rank tail."* This is a statement about
INFORMATION AVAILABLE IN A FROZEN, PRE-CORE, 4.0-nat, 207M, seq-1024 CONTEXT — not a
statement that flow matching cannot in principle replace depth. The `p1-l2cap` rerun on
the campaign's strongest available substrate (post-carry slot states) FAILED even harder
(trained ≈ untrained ≈ shuffled — "the planner learned nothing retrievable at all"),
closing that door for pre-core targets specifically and leaving POST-core carrier targets
as the one explicitly untested opening (`objective-lines-vs-l2cap.md:36-39, 81-90`).

### D.5 Explicit statement of what "no-loop, matched wall-clock" has NOT yet been tested

`docs/tul-paid-loop-recipe.md:263-266`, quoted in full: *"dmorph, the no-loop TUL with a
flow-matching objective at matched wall-clock and matched tokens
(.agents/notes/proposed/architecture/2026-08-30-dmorph-handoff.md on branch
feat/db-objective-l2): a dedicated build session, because the wall-clock claim needs its
kernel first."* And `objective-lines-vs-l2cap.md:54-57`: *"Boundary: DB-as-wall-clock on a
NO-loop TUL is untested — that is Wolfe's dmorph program... scored on token CE at matched
wall-clock AND matched tokens, K≤4 inference."* **This is the one clean, explicitly-scoped
open question in the repo relevant to a dmorph design**, and its own handoff note does not
exist on `master` (see §sources-found). Everything measured so far tested either (a)
denoising-vs-recurrence AT THE TOKEN LEVEL (§D.3, lost cleanly) or (b) flow-matching AS A
LATENT-PLANNER WRITE OBJECTIVE bolted onto the existing looped TUL architecture (§D.4,
capped by context information, not by the objective), never (c) a genuinely no-loop
architecture trained end-to-end with a flow-matching or diffusion objective and compared
at matched wall-clock — which is precisely what "dmorph" as named in this task would be.
