# Audit — the authors' DiffusionBlocks implementation

**Repo:** [github.com/SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks),
cloned 2026-08-19 (`--depth 1`). 5 Python files, 1522 lines: `model.py` (291),
`dblock_modules.py` (47), `vit.py` (855), `main.py` (126), `data.py` (203).

**Why this audit exists.** The plan of action had two items resting on the paper text alone: the
Euler-step sign (D8) and the embedding scale rule (R3a/O1). Both are now settled against code. The
audit also found a scope increase and one apparent inconsistency in their own implementation.

**Nothing was run.** Files were read only. The GPU was not touched.

---

## 1. Headline: the two modes MORPH needs most are NOT in the released code

`grep -rin "causal|autoregress|recurrent|huginn|llama|next_token|lm_head|clean.*noisy"` over all five
files returns **nothing**. `load_model` offers exactly two classes: `ViTModel` and `ViTDBlockModel`.

So the release covers **image classification with a ViT only**. Absent:

| Paper section | What it needs | Released? |
| --- | --- | --- |
| §5.4 / App. E.4 | AR text: clean\|noisy concatenation, modified causal mask | **no** |
| §5.5 / App. E.5 | recurrent-depth (the Huginn mode) | **no** |
| §5.3 / App. D | masked diffusion LM | **no** |
| §5.2 | DiT image generation | **no** |

**Consequence for the plan.** DB-1 (recurrent-depth) and the AR conversion are implemented from
paper prose with no reference to check against. The ViT code still pins the shared mechanics below,
which is most of the risk — but the causal-consistency mask and the recurrent-depth training loop
have **no reference implementation anywhere**. A4 and A5 must budget for that. This is a scope
increase, not a detail.

## 2. SETTLED: the Euler sign. The paper's Eq (3)–(5) is a typo.

`model.py:283-287`:

```python
d = (z - denoised) / sigma[:, None]
dt = next_sigma - sigma          # NEGATIVE: sigmas are descending
euler_step = z + dt[:, None] * d
z = euler_step
```

`get_discrete_sigmas(..., dblock=True)` ends with `torch.flip(sigmas, dims=[0])`, so the buffer is
descending and `dt < 0`. Expanding:

```
z_next = z − (Δσ/σ)·(z − D),    Δσ := σ − σ_next > 0
       = α·z + (1−α)·D,          α = σ_next/σ ∈ (0,1)
```

Substituting a perfect denoiser (`z = y + σ·ε`, `D = y`) lands at `y + σ_next·ε`. Correct.

**Gate A1 / sheet G3 closes: use `α = σ_b/σ_{b-1} ∈ (0,1)`.** The rendered Eq (3)–(5) sign in v4 is
a typo; their code is the EDM form. **Our derivation was right.** The `ρ(J) ≤ 1` argument in the
assessment §5.3 rests on `α ∈ (0,1)` and therefore stands.

## 3. Exact mechanics, copied verbatim from their code

Everything here should be matched exactly rather than re-derived.

**Equi-probability boundaries** (`dblock_modules.py::get_block_sigmas`) — `σ_min = 0.002`,
`σ_max = 80.0`, `P_mean = −1.2`, `P_std = 1.2`; boundaries are ascending, evenly spaced in
`Φ((log σ − P_mean)/P_std)`.

**EDM preconditioning** (`model.py:203-206, 222`), `σ_data = 0.5` hardcoded:

```python
c_skip  = σ_d**2 / (σ**2 + σ_d**2)
c_out   = σ * σ_d / (σ**2 + σ_d**2)**0.5
c_in    = 1 / (σ**2 + σ_d**2)**0.5
c_noise = 0.25 * σ.log()                 # the AdaLN timestep input
model_out = hidden_states * c_out + zt * c_skip
```

**Loss** (`model.py:255-259`) — per-sample CE, then EDM weighting, then mean:

```python
loss = F.cross_entropy(..., reduction="none")
w    = (σ**2 + σ_d**2) / (σ * σ_d)**2
loss = (loss * w).mean()
```
They log the unweighted `ce_loss` alongside, and log both **per block index**. Copy that: a
per-block loss channel is how you see one block failing.

**Block sampling** (`model.py:158`) — `random.choices(range(num_blocks), k=1)[0]`: **one block per
BATCH**, uniform, then σ drawn within that block's range. Uniform visits confirmed (App. E.1).
Note it is per-batch, not per-sample.

**Overlap γ** (`model.py:163-169`) — extend the block's `[log σ_min, log σ_max]` by `γ × log_range`
on both sides, then clamp to the global range. Algebraically the paper's `α_b = (σ_{b-1}/σ_b)^γ`.

**Noising** (`model.py:252`) — `zt = z + σ[:,None] * torch.randn_like(z)`. Plain VE.

**Inference init** (`model.py:272-273`) — `z = randn(...) * sqrt(1 + σ_max**2)`, not `σ_max * randn`.

**Inference step count** (`model.py:119`) — `num_inference_steps or num_blocks`. Default is `B`
steps, i.e. one per block.

## 4. NEW, and we did not have it: how CE-trained logits become an embedding

`model.py:280-281`, inside the sampling loop:

```python
probs    = F.softmax(logits, dim=1)
denoised = F.linear(probs, self.model.get_input_embeddings().weight.t())   # = probs @ E
```

The denoised estimate fed to the next Euler step is the **probability-weighted average of the
embedding table**, not the raw network output. This is the bridge from a CE objective back into
embedding space, and MORPH's AR path needs exactly this. We did not have it from the paper text.

**Two consequences worth pre-registering.** The embeddings are unit-norm, so a convex combination of
them has norm ≤ 1, and **shrinks toward 0 as the model gets less certain**. So the sampling
trajectory has a scale drift that training never sees (training always gets a true unit-norm `y`).
That is a train/inference mismatch built into the method. Second, it makes the tied output weight
part of the sampler, so any normalisation we apply to `y` must be applied to the tied head too.

## 5. FLAGGED: their normalisation and their `σ_data` look inconsistent

`model.py:143-144`:

```python
def normalize_embeddings(self, x):
    return F.normalize(x, p=2, dim=-1)      # ‖y‖ = 1 exactly
```

Applied in `get_embeds` to **both** the input and output embedding paths. Meanwhile
`self.sigma_data = 0.5`.

In EDM, `σ_data` is the **per-component** standard deviation of the data (0.5 is the value for
images in `[-1, 1]`). A vector with `‖y‖ = 1` over `d` dimensions has per-component std `1/√d`. For
`d = 1024` that is `0.031`, not `0.5` — off by ~16×. The two settings are only consistent at `d = 4`.

If `σ_data` overstates the true scale by 16×, then `c_skip = σ_d²/(σ²+σ_d²)` stays near 1 until
`σ ≈ 0.5`, while the signal is already buried by `σ ≈ 0.03`. Most of the schedule would be asking
the model to denoise from nothing.

**Caveat, stated honestly:** the objective here is **cross-entropy on logits**, not L2 on embeddings,
so scale enters through the logit temperature rather than directly through the residual. CE is more
tolerant of this than L2 would be, and the `w(σ)` weighting absorbs some of it. We may be misreading
their intent, and a 12-layer ViT on CIFAR-100 with 100 labels may simply not be sensitive here. **We
have not tested this and cannot resolve it from reading alone.**

## 6. Recommendation for O1 — the scale rule

Their whole-vector L2 does **not** transfer to MORPH, and the audit shows why.
`HybridEmbedding.forward` **concatenates** disjoint slices:

```python
return torch.cat([self.euc_embed(input_ids), self.lor_embed(input_ids)], dim=-1)
```

At `d_model = 1024`, `lorentz_fraction = 0.25` → euclidean 768 dims, Lorentz-tangent 256 dims. The
euclidean slice inits near unit norm; the Lorentz tangent slice inits at std 0.005 (log-map ≈
identity near the origin). `F.normalize(..., dim=-1)` over the **concatenation** rescales the whole
vector and leaves the *ratio* between slices untouched. Because `σ·ε` is isotropic, it lands the
same absolute magnitude on the tiny Lorentz components as on the large euclidean ones — **the
Lorentz slice is pure noise at every useful σ.**

**Recommended (this is the O1 call):**

1. Normalise **per slice**, not per vector — the euclidean slice and the Lorentz-tangent slice each
   independently.
2. Scale each slice so its **per-component std equals `σ_data`**, i.e. slice norm
   `= σ_data · √(slice_dim)`, rather than copying their `‖y‖ = 1`. This makes `σ_data = 0.5`
   literally true instead of inheriting the §5 discrepancy.
3. Apply the identical transform to the tied head weight (`HybridEmbedding.lm_weight()`), because
   §4 puts it inside the sampler.
4. Do this **inside the DB conversion**, not in the embedding module, so the DB-off path stays
   bit-identical (hard constraint 4).
5. **Do not** try to measure per-slice data statistics and set `σ_data` from them. That was
   over-engineering on our side: pinning the scale by construction is what makes `σ_data` known.
   Measurement is a sanity check, not the mechanism.

The bigram signal is an **additive full-width** injection, not a concatenated slice, so it is part of
the conditioning `x0`, not part of the target `y`. `y` is `HybridEmbedding.forward(ids)` per-slice
normalised. Confirm this when wiring A5.

## 7. What this changes in the plan

| Item | Before | After |
| --- | --- | --- |
| A1 / G3 (sign) | open, "check their code" | **closed** — `α = σ_b/σ_{b-1}`, §2 above |
| A2 / O1 (scale) | "per-slice L2 vs per-slice σ_data", unresolved | **decided** — §6, per-slice scaling to `σ_data·√dim` |
| A5 scope | AR mask "from paper prose" | **larger** — AR mask *and* recurrent-depth loop have no reference at all (§1) |
| A5 detail | not specified | `probs @ E` for the sampler bridge (§4); exact preconditioning and per-block loss logging (§3) |
| New risk | — | **R8**: their `‖y‖=1` vs `σ_data=0.5` inconsistency (§5). Unresolved; may or may not matter under CE |
| New risk | — | **R9**: sampler scale drift — `probs @ E` shrinks toward 0 under uncertainty while training always sees unit-scale `y` (§4). A train/inference mismatch inherent to the method |

## 8. Not verified

- Nothing was executed. No pytest, no smoke run, no GPU work.
- `vit.py` (855 lines) was not read line by line — only the interfaces `model.py` calls
  (`forward_block(layer_indices=…, pixel_values=…, noisy_embeds=…, timesteps=…)` and
  `forward_output_embeddings`). The AdaLN wiring inside it is unread.
- `data.py` and `main.py` were read only for the class dispatch.
- The §5 inconsistency is a reading of their code against the EDM definition of `σ_data`. It is not
  confirmed with the authors and we have run no experiment either way.
