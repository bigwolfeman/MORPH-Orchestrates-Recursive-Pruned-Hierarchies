# DiffusionBlocks testbed research report

Repo: `/home/wolfe/11-DiffusionBlocks-Testing` (own git repo, own venv). Not modified.
Head: `a8e92eb Verdict: DiffusionBlocks loses. All 27 gates met, testbed closed out`.

## A. The faithful DB recipe, as implemented

### Backbone (identical across every arm)

12 layers, `d_model=768`, 12 heads, RMSNorm, RoPE, SwiGLU `d_ff=2048`, no biases, tied
readout (`lm_head.weight = embed.weight`), ~124M params, bf16 autocast / fp32 master
weights. `dbref/model.py:26-35` (`ModelConfig`), `dbref/model.py:224-236` (`Llama.__init__`).
GPT-2 BPE vocab 50257 (deviation from the paper's Llama-2 32K tokenizer, documented in
`README.md` "Deviations from the paper").

### Block partition

`DBSchedule._block_sigmas` (`dbref/db.py:115-129`) computes `n_blocks+1` sigma boundaries
that carry **equal log-normal CDF mass** under the training noise distribution
(`p_mean=-1.2, p_std=1.2`, i.e. "equi-probability partitioning", paper Appendix C):

```python
cdf_min = _log_normal_cdf(log(sigma_min), p_mean, p_std)
cdf_max = _log_normal_cdf(log(sigma_max), p_mean, p_std)
for i in range(n_blocks + 1):
    p = cdf_min + (cdf_max - cdf_min) * (i / n_blocks)
    out.append(_log_normal_ppf(p, p_mean, p_std))
```

This is a direct port of the vendored `get_block_sigmas` (`third_party/sakana_dblock_modules.py:25-39`),
and parity is asserted numerically in `tests/test_db_parity.py` (17 tests, gate G2.1).
`n_blocks` layers are then split into contiguous, **equal-size** groups: `layer_assignment`
(`dbref/db.py:210-216`), `n_layer % n_blocks == 0` required.

Each block's training interval is the boundary pair `[block_sigmas[b], block_sigmas[b+1]]`,
widened in **log space** by `gamma` (paper: 0.1 for text, Appendix C) so neighbouring blocks
overlap: `block_range` (`dbref/db.py:144-161`). A block is sampled uniformly at random **per
optimizer step** (paper Appendix E: "blocks are sampled uniformly at random for each
iteration"; `dbref/train.py:254`, fixed across gradient-accumulation micro-steps).

**The sigma→block reversal (load-bearing, and a documented past bug).** HIGH sigma is
handled by the EARLY block, LOW sigma by the LATE block — "it does the coarse work, like an
early transformer layer" (`dbref/db.py:181-192`, `block_of_sigma`, a port of the oracle's
`estimate_target_layer`, `model.py:182-188`):

```python
idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
idx = (n_blocks - 1) - idx
```

There is exactly **one road** from sigma to a running layer group,
`DBSchedule.layer_block(sigma)` (`dbref/db.py:194-208`, majority vote over a batch of sigmas),
used identically by training, evaluation, and the sampler. This was not always true: an
earlier bug kept the sampled *band index* on one path and re-derived the block from sigma on
another, so a 573M-token `db_b4` run trained band `b` on `layers[b]` and was evaluated on
`layers[(B-1)-b]` — "Perfectly inverted" (`notes/2026-08-21-block-reversal-bug.md`). Fixed by
routing everything through `layer_block`; a guard test scans `scripts/` for any remaining
direct call to `block_of_sigma`.

### What each block predicts, and the preconditioning

EDM (Karras et al. 2022) preconditioning, `edm_coeffs` (`dbref/db.py:219-239`, oracle
`model.py:203-206`):

```
denom  = sigma^2 + sigma_data^2
c_skip = sigma_data^2 / denom
c_out  = sigma * sigma_data / sqrt(denom)
c_in   = 1 / sqrt(denom)
c_noise= 0.25 * log(sigma)
```

Limits, quoted from `dbref/db.py:225-230`: "sigma -> 0: c_skip -> 1, c_out -> 0 => the
denoiser IS the identity on z. sigma -> inf: c_skip -> 0, c_out -> sigma_data => the output
is entirely the network's prediction, and z carries no information."

Forward (`Llama.forward_db`, `dbref/model.py:352-410`), order is structural (oracle
`model.py:214-225`):
1. `c_skip, c_out, c_in, c_noise = edm_coeffs(sigma)`
2. Network sees `z * c_in`; `sigma` enters through AdaLN via `c_noise` (a `TimestepEmbedder`,
   `dbref/model.py:117-135`, DiT-style sinusoidal embed → MLP, zero-initialized modulation so
   an untrained DB model starts as the un-modulated backbone).
3. `denoised = hidden * (c_out * out_scale) + zt * c_skip` — the **denoised estimate D̂, in
   target embedding space** (`dbref/model.py:405-406`).
4. A second AdaLN-modulated affine (`out_adaLN`) is applied to `denoised`, then
   `logits = F.linear(out, embed.weight) * logit_scale` (`dbref/model.py:407-409`).

So the target the network is trained to reconstruct is **x0** (`D̂` estimates `y`, the clean
scaled embedding row of the label token), not ε or v — standard EDM/Karras x0-parameterization
routed through a tied classifier. The paper substitutes cross-entropy for L2 (Appendix B:
"We minimize cross-entropy loss instead of L2 loss"), so the actual training signal is CE on
`logits` against the label token, with `logits` derived from `D̂` as above.

**Readout scale (`# CHOICE:`, `dbref/model.py:242-279`).** The paper never states the
embedding→token bridge for text and its oracle ViT code uses an untied, zero-init classifier
that can freely learn its own scale; a **tied** readout has no such freedom. `logit_scale`
and `out_scale` are solved in closed form from a target logit-gap `TARGET_GAP = 15.0`:

```
y_norm = sigma_data*sqrt(d_model)  if target_scale=='rms' else 1.0
e_norm = init_std*sqrt(d_model)
logit_scale = TARGET_GAP / (y_norm * e_norm)
out_scale   = y_norm / (sigma_data * sqrt(d_model))
```

Getting this wrong is invisible in the loss curve but not in the numbers: `sqrt(d)` scaling
put logits at ±213 (saturated softmax) and drove init CE to 8e-5; without `out_scale`,
`target_scale='unit'` starts 14x overscaled and reads CE ≈32 nats — "three times ln(V), i.e.
confidently WRONG rather than merely ignorant" (`dbref/model.py:262-277`).

### Loss / weighting

`db_loss` (`dbref/loss.py:17-38`): per-position CE, mean per sample, mean over the batch —
**unweighted** by default. `loss_weighting: edm` applies
`w(sigma) = (sigma^2+sigma_data^2)/(sigma*sigma_data)^2 == 1/c_out^2` per sample
(`edm_weight`, `dbref/db.py:242-252`). Default is unweighted because the paper's own released
code cannot apply Eq. 6 as a per-sample weight in the first place (see B, "eq6 weighting").

### Independence across blocks

Blocks are **not trained jointly with a shared full-network backward**: one block (one
contiguous layer group) is sampled per optimizer step and only that group's forward/backward
runs — `layers=assign[sched.layer_block(sigma)]` is passed into `forward_db`, and `context()`
/ `forward_db` both accept an explicit `layers` list and iterate only over it
(`dbref/model.py:318-339` `context`, `:370-401` the DB loop). There is no cross-block
gradient in a single step by construction — each step touches exactly one block's parameters
(plus the shared embedding/readout, adaLN embedder, and the `clean` pass through
`context(idx, layers=idxs)` restricted to that block's own layers). Embedding, readout, and
`t_embed`/`clean_cond`/`out_adaLN`/`logit_scale`/`out_scale` are shared and always receive
gradient regardless of which block is sampled.

### Bridge (clean context → block input) — train vs inference

**Train time**, `conditioning='concat'` (the paper's Appendix E.4 scheme, "noisy and clean
sequences are concatenated with a modified causal attention mask"): realized as **two passes
over the same weights** rather than one 2L-sequence pass, so the mask stays plain-causal and
the flash kernel runs (`dbref/model.py:372-393`, `README.md` "Deviations"). The clean pass
(`Llama.context`, sigma-independent) produces per-layer K/V (`clean_kv_source = n1(x)`,
`dbref/model.py:171-179`), and the noisy stream at position `i` cross-attends those clean
keys/values with `is_causal=True`, which realizes exactly "noisy query i sees clean keys
j<=i, never clean i+1 (its own target)" — the no-leak rule, proven by perturbation in
`tests/test_no_leak.py` (gate G2.3, with a positive control). **The one documented
deviation**: "the noisy query does not attend its own noisy key... position i's residual
stream already IS c_in*z_i, so z reaches the MLP and the query projection regardless; only
the (self-)value aggregation is dropped" (`dbref/model.py:382-386`). The literal single-pass
2L form with an explicit mask (`conditioning='concat2l'`) is implemented
(`build_db_masks`, `dbref/model.py:182-221`) but never run to a full arm.
An `add` (single-stream additive injection) mode also exists, measured and rejected (see B).

**Inference time**, the bridge from vocab logits back to embedding space for the Euler walk
(`bridge_embedding`, `dbref/sample.py:22-54`) — "THE PAPER NEVER SPECIFIES THIS. It is the
single largest unstated detail in the text":
- `soft`: `softmax(logits) @ E` (the oracle's ViT choice, `expected_embedding`,
  `dbref/db.py:287-296`). A convex combination of embedding rows — collapses norm when the
  model is uncertain.
- `hard`: the argmax row itself. Always full norm, always on-manifold.
- `topk`: renormalized top-k mixture rescaled to the target norm.

`emb = scale_target(model.embed.weight, target_scale, sigma_data)` — the bridge reads through
the readout table **put in target space**, not the raw table the oracle uses inconsistently
(`dbref/sample.py:103-107`, explicit note that this differs from the oracle's `model.py:280-281`).

### Sampler

`db_denoise` (`dbref/sample.py:86-133`): a **DESCENDING equal-CDF-mass sigma grid**,
`inference_sigmas(n_steps)` (`dbref/db.py:131-142`, port of `get_discrete_sigmas(dblock=True)`,
`third_party/sakana_dblock_modules.py:57-66`). `n_steps` is a **free test-time dial**, no
retraining needed, "never has a hidden default" (`dbref/sample.py:1-6`). Paper uses 4 Euler
steps (Appendix E.4); the testbed evaluates 4/16/32.

Initial `z`: `randn(...) * sqrt(1 + sigma_max^2)` (oracle `model.py:272-273`,
`dbref/sample.py:112-113`). Per step: `layer_block(sigma_i)` picks the running block, forward,
`denoised = bridge_embedding(logits, emb, bridge)`, then a single Euler step
(`euler_step`, `dbref/db.py:274-284`, oracle `model.py:283-287`):

```
d  = (z - denoised) / sigma
dt = next_sigma - sigma        # NEGATIVE on a descending grid
z  = z + dt * d
```

One extra forward call at the final (lowest) sigma after the loop, matching the oracle's
extra call (`model.py:288-290`), because the loop only advances `n_steps-1` times. No KV
cache; every generation step (one new token) re-runs the full walk from scratch — "no KV
cache by design (v1): every step is a fresh forward, which is slow but cannot be subtly
wrong" (`dbref/sample.py:157-166`). K per block: n/a — one block runs per Euler step,
determined by `layer_block(sigma)` at that step's sigma, not a fixed rotation.

## B. Ladder results

All numbers from `results/*.json`, `logs/compare_arms.log` (quoted in `GATES.md` I4/G3.3),
and the notes files, at the shared 143.36M-token budget (`16 batch * 1024 seq * 8750 steps`,
matched to the MORPH `db_b1_ce` run) unless stated otherwise.

**AR baseline** (`configs/ar.yaml`, git `26f9f59`): held-out CE **4.0010**, PPL **54.65**
(`results/ar.json`, `notes/2026-08-20-ar-finetune-from-db.md`). `ln(V)=ln(50257)=10.8249`.

**CE at sigma_max (the decisive column — c_skip→0, z carries no information, pure
next-token prediction from clean context)**, `GATES.md` I4 / `notes/2026-08-21-verdict-and-teardown.md`:

| arm | tokens | CE @ sigma_max | vs AR |
|---|---:|---:|---:|
| `ar` | 143.4M | **4.0010** | — |
| `db_b1_oracle` | 143.4M | 5.0801 | +1.08 |
| `db_b4` (tokmatched) | 143.4M | 5.5329 | +1.53 |
| `db_b4` | 573.4M | 4.6740 | +0.67 |
| `db_b1` | 143.4M | 7.3011 | +3.30 |
| `db_b1_wedm` | 143.4M | 7.6827 | +3.68 |
| `db_b1_rms` | 143.4M | 10.3916 | +6.39 |

No arm at any tested budget reaches the AR baseline.

**B=1 vs B=4 sweep** (`notes/2026-08-21-block-reversal-bug.md`, post-fix): the paper's
protocol multiplies steps by the block count (`configs/db_b4.yaml`, `steps: 35000` vs
8750 base — "the paper's protocol multiplies steps by the block count, so B=4 needs 4x the
tokens to give each block the same number of updates"):

| arm | Mtok | tok/s | wall s | micro | peak VRAM | CE @ sigma=1 | CE @ sigma_max |
|---|---|---|---|---|---|---|---|
| ar (baseline) | 143 | 135,518 | 1,058 | 16 | 18.0 G | — | **4.0010** |
| db_b1 | 143 | 59,318 | 2,417 | 4 | 9.0 G | 4.1236 | 7.3011 |
| db_b1_oracle | 143 | 54,585 | 2,626 | 4 | 9.0 G | 4.5572 | 5.0801 |
| **db_b4** | 573 | 172,329 | 3,328 | 16 | 18.5 G | 4.2044 | **4.6740** |

"Block partitioning works, and every B=1 result here was measuring a handicapped variant of
the paper's actual proposal... B=1 is the Huginn/recurrent-depth mode MORPH runs, and B=1 is
the weakest DB setting." At equal wall clock, plain AR still wins: db_b4's 3,328s buys 451M
tokens of plain AR (already at 4.0010 with only 143M).

**Argmax-vs-softmax bridge** (commit `afcd3d8`, `notes` reproduced from `git log afcd3d8 -1`),
db_b1, greedy decode, 8 held-out prompts, real-text anchor gen-PPL 34.77:

| bridge | 4 steps | 16 steps | 32 steps |
|---|---:|---:|---:|
| soft | 1610.58 | 583.57 | 596.26 |
| hard | 55.54 | 22.87 | **17.07** |
| topk8 | 75.36 | 55.32 | 42.90 |

("gen-PPL 584 → 17" from the commit title refers to the 16-step column, 583.57 → 22.87, and
the 32-step best case 596.26 → 17.07.) Measured `||D̂||` collapse driving this (`bridge_probe.py`,
db_b1, training always showed `||y||=1.0`): sigma=80 → `||D̂||=0.266` ("73% of the signal
gone"), sigma=1.8 → 0.297, sigma=0.2 → 0.836, sigma=0.05 → 1.000 ("only recovers at the very
last step"). "MORPH's gen-PPL 775 sits right on our soft-bridge number (584). Same failure
mode, and it was never MORPH's Lorentz embeddings, ternary QAT or looped core."

**"sigma_max is the metric" finding**: the sigma-grid mean ranks arms backwards.
`db_b1_rms` has the **best** grid mean in the testbed (2.8400) and the **worst** model
(CE 10.3916 at sigma_max, essentially the uniform floor `ln V = 10.8249`)
(`GATES.md` I4, `notes/2026-08-21-verdict-and-teardown.md` "Three ways this testbed could
have lied"). Grid means, `logs/compare_arms.log`: db_b1 3.1434, db_b1_oracle 3.0317,
db_b1_rms 2.8400 — inverted order from sigma_max.

**σ\* / SliceScaler autoencoding trap** (`README.md` "The two things worth knowing",
`scripts/decodability.py`): DB target `z = y + sigma*eps`; nearest-neighbour decode of `z`
recovers the token whenever `sigma < sigma* ≈ ||y||/4.2` — below that, "the model can reach
~0 loss without using context at all."

| target scaling | sigma\* | training mass below sigma\* |
|---|---:|---:|
| unit norm, sigma_data=0.5 (oracle) | 0.235 | 42% |
| per-dim RMS = sigma_data (MORPH's SliceScaler) | 3.30 | **98%** |
| MORPH's actual setting (p_mean=0, p_std=1.6) | 3.30 | **77%** |

("77% of training into autoencoding" is the third row: MORPH's own `p_mean=0, p_std=1.6`
schedule combined with RMS target scaling.) `db_b1_rms.yaml` comment: "Prediction: this arm
reaches a low training loss while failing to use context — the exact signature measured on
MORPH's DB arms," borne out by the CE-at-sigma_max=10.3916 result above.

**DB→AR conversion loses** (`notes/2026-08-20-ar-finetune-from-db.md`,
`scripts/compare_finetunes.py`, commit `f2ffeab`):

```
AR baseline, 8750 steps from scratch: CE 4.0010  ppl 54.65
uniform floor ln(50257) = 10.8249

arm                     start    final       ppl  d(nats)  pre tok/s  ft tok/s  total s
---------------------------------------------------------------------------------------
ar_cont                4.0010   3.9551     52.20  -0.0459    135,518   118,433    1,335
ar_ft_db              10.8405   4.8969    133.87  +0.8959     59,318   120,398    2,689
ar_ft_db_noisy         7.3124   4.3042     74.01  +0.3032     59,318    51,821    3,049
ar_ft_oracle_noisy     5.0793   4.6777    107.52  +0.6767     54,585    52,555    3,250

best DB route: ar_ft_db_noisy  CE 4.3042 in 3,049s
control:       ar_cont  CE 3.9551 in 1,335s
=> +0.3491 nats for 2.28x the compute
   that wall clock as plain AR training = 413M tokens vs the 143M this baseline saw
```
("0.35 nats worse for 2.28x compute" matches these two rows, `ar_ft_db_noisy` vs `ar_cont`.)
Two conversions exist: **clean stream** (`forward_ar`, reuses `Llama.context`, 1x compute,
starts at 10.8405 — "above the uniform floor... literally no information," because the final
norm/readout only ever saw the noisy stream) and **noisy stream** (`ar_sigma=80`, pinning
sigma at max with `z = sigma*eps` not `y+sigma*eps`, so the label cannot leak; 2x compute,
the two-pass concat forward). "CE at sigma_max ranks models, it does NOT predict
convertibility": `db_b1_oracle` had the best sigma_max CE (5.0801) but finished the
fine-tune **worst** of the two noisy arms (4.6777 vs `db_b1`'s 4.3042).

**Sampling is inert** (`GATES.md` G3.6, `scripts/nucleus_probe.py`, `notes/2026-08-21-verdict-and-teardown.md`):
at the position generation actually draws from (end of the Euler walk), every DB arm has
top-1 probability 1.0000, entropy 0.0000 nats, nucleus-@0.95 = 1 token — vs AR's 0.2920 /
4.7613 nats / 8422 tokens. Cause: EDM's `c_skip → 1` as `sigma → 0`, so the denoiser output
IS its input there; top-p 0.95 and pure ancestral sampling return bit-identical text on
every DB arm from the same seed. The same denoisers read *directly* at higher sigma are not
collapsed (db_b1: entropy 3.32 / 4.74 / 5.04 / 7.72 nats at sigma 0.3 / 1 / 10 / 80) — "the
collapse is specific to the low-sigma end... a property of the method, not of this build."

**Eq. 6 weighting is slightly worse than unweighted** (`notes/2026-08-21-eq6-weighting.md`):
db_b1_wedm loses to db_b1 (unweighted) at all seven eval sigmas, worst at sigma_max
(7.6827 vs 7.3011, +0.3816). Explanation: the released oracle code computes `loss * w` as a
`[B]` x `[B,1]` broadcast — an outer product — so `.mean()` of it equals
`mean(loss)*mean(w)`, "a single scalar multiplier on the whole batch, not a per-sample
reweighting." The shipped/oracle code's realized behaviour is therefore close to unweighted;
implementing Eq. 6 literally (this arm) makes the model measurably worse.

**Diversity-matched generative PPL** (`GATES.md` I4/G3.4): only rows within 0.03 distinct-2
of real held-out text are compared (naive gen-PPL alone is gameable — `db_b1_rms` scored
gen-PPL **1.46**, "twenty times better than real English," by emitting "the the the ..." at
distinct-2 0.016). Matched: `ar` 250.17 vs best DB row 412.01 (32 prompts × 50 tokens); `ar`
162.07 vs best DB row 325.85 (12 prompts × 64 tokens, `results/samples_all_arms.md`). Same
ordering at both sample sizes; no DB arm reaches the AR baseline on either axis.

**FLOP / wall-clock accounting.** The testbed did **not** build a FLOPs-per-token formula;
its accounting is empirical tok/s and wall-clock (`dbref/train.py` logs `tok_per_s` every
`log_every` steps; `scripts/compare_finetunes.py` and `scripts/compare_arms.py` read those
logs). Measured throughput: AR 135,518 tok/s; db_b1 (concat, two-pass) 59,318 tok/s (~2.3x
per-token cost, `notes/2026-08-20-ar-finetune-from-db.md` point 4: "db_b1 pretrains at
59,318 tok/s against the AR arm's 135,518, because the two-pass concat forward is ~2.3x per
token"); db_b1_oracle 54,585 tok/s; db_b4 172,329 tok/s (faster per-token than AR because
each step only backprops 3 of 12 layers, but needs 4x the steps under the paper's protocol).
Peak VRAM: AR 18.0 G at micro_batch 16; db_b1/db_b1_oracle 9.0 G at micro_batch 4 (4x
accumulation); db_b4 18.5 G at micro_batch 16 — "level" with AR, not a win, because the
paper's memory claim assumes full end-to-end backprop of the same partitioned model as the
comparison point, "not the same yardstick as a plain AR LM of this size"
(`notes/2026-08-21-block-reversal-bug.md`).

## C. Final verdict and what the testbed says transfers

Verdict, `notes/2026-08-21-verdict-and-teardown.md` and `GATES.md` G3.5 (Wolfe's call,
quoted in the note): **"failure, park it... It changes the training objective too much...
it gives excellent benefits hypothetically, but realistically I think it's far off the
mark."** DiffusionBlocks was removed from MORPH master (commit `938d2e9`) and parked on
`park/db-master-line` / `feat/db-objective-l2`; the MORPH-side record is
`.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md` (not duplicated here
per that note's instruction).

**The deciding number**: CE at sigma_max, AR 4.0010 vs best DB arm 5.0801 at a matched
143.4M-token budget, 4.6740 only at 4x the tokens (573M). Nothing reaches the baseline at
any tested budget.

**Culprit named explicitly**: the DB **mechanism**, not MORPH's complexity. "A plain 124M
Llama with no Lorentz slice, no hyper-connections, no quantization and no sparsity reproduces
the same failure," and G2.4 (overfit-probe gate) shows the objective *does* train the network
through the context path (CE 7.8903 → 0.0084 over 400 steps on one repeated batch, scored
above sigma\* where only clean context can help: 0.0329 at sigma=1, 0.0499 at sigma=80
against a 1.50 threshold) — so it is not a wiring bug either.

**What DB costs, quoted plainly:**
- ~2.3x compute per token in the pretraining two-pass concat forward (measured tok/s).
- The best-configuration (B=4, paper protocol) still loses to AR at equal wall clock: 3,328s
  buys AR 451M tokens (already ahead of DB's best number at 143M).
- Converting a DB checkpoint back to AR costs 2.28x the compute of just continuing AR
  training, and still lands 0.35 nats worse (`ar_ft_db_noisy` 4.3042 vs `ar_cont` 3.9551).
- Sampling loses temperature/top-p control entirely (structural, from EDM's `c_skip→1` limit
  at low sigma) — not a bug, a property of the method as implemented for text.
- Memory is a wash at B=4 against a plain AR LM of this size, not the win the paper's
  B-fold claim implies (different backprop yardstick).

**What the testbed itself says transfers** (i.e., lessons stated as general in the notes,
not MORPH-specific facts):
- "Test the seam, not just the parts" — two individually-correct sigma→block functions
  composed into a silently wrong system (the reversal bug); parity-testing each function in
  isolation was not enough.
- CE at the noise level where the diffusion input is provably uninformative (sigma_max here)
  is the only trustworthy scalar to rank DB-style checkpoints; a schedule-weighted mean
  rewards autoencoding and can rank backwards. This generalizes past this specific paper: any
  denoising objective mixed with a schedule that spends non-trivial mass near-clean needs a
  "held-out at max noise" number, not a training-loss mean.
- "CE at sigma_max ranks models, it does NOT predict convertibility" — best-at-one-metric is
  not best-at-the-downstream-use; do not assume monotonicity between a diagnostic and a
  transfer outcome without measuring the transfer directly.
- Unstated bridge/preconditioning details (the vocab→embedding bridge, the readout scale for
  a tied classifier) are exactly the kind of choice that a loss curve cannot reveal as wrong
  — both were measured as "trains fine, generates garbage" or "confidently wrong at init"
  before being caught by dedicated weights-free probes, not by watching the loss.
- Diversity/degeneration checks (distinct-2, rep-rate) must travel with any generative-PPL
  number, and greedy-only decoding must never be the only decode reported — both caught a
  reversed ranking in this testbed.

**Explicitly UNVERIFIED** (`GATES.md` G3.5, `notes/2026-08-21-verdict-and-teardown.md` "Not
verified"): 143.4M tokens is ~1/70 of the paper's language budget, and `db_b4` DID improve
with 4x tokens (5.5329 → 4.6740) — no crossover past this scale is ruled out. One model
size, one depth, one dataset, one tokenizer; no image experiments (the paper's headline
domain). MAUVE and rep4@512 were never computed. `conditioning: concat2l` (literal 2L form)
was implemented but never run to a full arm — the two-pass `concat` form drops the noisy
query's self-attention to its own noisy key, a documented deviation whose effect was never
isolated. `results/samples_all_arms.md` was generated eager while only the CE numbers moved
to the compiled scoring path.

## D. Reference implementation for a block-wise no-loop MORPH — what transfers directly, what is Llama-specific

### Directly reusable pattern (algorithm, not code, since dmorph has a different backbone)

| Concept | File : function |
|---|---|
| Equal-CDF-mass block sigma boundaries | `dbref/db.py:115-129` `DBSchedule._block_sigmas` |
| gamma-widened, overlapping block sigma ranges | `dbref/db.py:144-161` `DBSchedule.block_range` |
| Per-block sigma sampling (uniform-in-CDF) | `dbref/db.py:163-179` `DBSchedule.sample_sigma` |
| sigma → block index with the REQUIRED reversal | `dbref/db.py:181-192` `DBSchedule.block_of_sigma` |
| The single road from sigma to a running layer group (majority vote) — the seam that must not fork | `dbref/db.py:194-208` `DBSchedule.layer_block` |
| Contiguous equal-size layer→block assignment | `dbref/db.py:210-216` `DBSchedule.layer_assignment` |
| EDM preconditioning (`c_skip/c_out/c_in/c_noise`) and its two limits | `dbref/db.py:219-239` `edm_coeffs` |
| EDM per-sample loss weight, and why it should default OFF for CE | `dbref/db.py:242-252` `edm_weight`; `dbref/loss.py:17-38` `db_loss` |
| The x0-in-target-space forward order (`D̂` before `logits`) | `dbref/model.py:352-410` `Llama.forward_db`, steps 1-4 in the docstring |
| Closed-form tied-readout scale derivation from a target logit gap | `dbref/model.py:242-279` (`logit_scale`, `out_scale`) — needed for ANY tied classifier reading a diffusion `D̂`, backbone-independent |
| Descending equal-CDF sigma inference grid | `dbref/db.py:131-142` `DBSchedule.inference_sigmas` |
| Euler probability-flow step | `dbref/db.py:274-284` `euler_step` |
| Bridge choices (soft/hard/topk) and the norm-collapse failure mode of `soft` | `dbref/sample.py:22-54` `bridge_embedding` |
| One-block-per-optimizer-step training loop shape (block fixed across grad-accum micro-steps) | `dbref/train.py:245-284` `main()` training loop body |
| The overfit/anti-theater gate methodology (drive one batch to ~0 loss ABOVE sigma\*) | `scripts/overfit_probe.py`, gate G2.4 in `gates/l2-db-core.md` |
| The weights-free decodability/σ\* diagnostic | `scripts/decodability.py` |
| No-leak mask proof-by-perturbation methodology (with a positive control) | `tests/test_no_leak.py`, gate G2.3 |
| The diversity-matched gen-PPL comparison protocol | `dbref/genppl.py::text_stats`, `scripts/compare_arms.py` D2_BAND logic |

### Llama/text-specific, not directly portable to a no-loop MORPH block-wise design

- The **concat two-stream realization as two passes over shared weights**
  (`dbref/model.py:372-393`, `Llama.context` reused as both AR body and clean stream) is
  specific to a transformer with `is_causal` SDPA and a token sequence; a no-loop MORPH with
  its own residual/attention machinery (HC-Cayley carrier, CCA/CSA/HCA) would need its own
  realization of "clean context reaches the noisy computation without leaking the target,"
  and MORPH's own `ChannelInject`/`x0_inject` gating (referenced but explicitly **not** used
  by this testbed — PLAN.md finding 1) is the closer MORPH-native analogue to study instead
  of copying `concat`.
- `build_db_masks` (`dbref/model.py:182-221`) encodes the exact causal/no-leak rule for a
  single flat token sequence with RoPE positions duplicated for the 2L layout; a MORPH block
  design (hyperbolic embeddings, hybrid attention) has a different sequence/position
  structure entirely.
- The **tied-embedding-table bridge and logit-scale derivation** assumes a GPT-2-BPE
  Euclidean embedding table; MORPH's hybrid (Euclidean + Lorentz) embedding would need its
  own target-space and bridge derivation — the *method* (solve scale from a target CE floor)
  transfers, the formula's constants do not.
- `TimestepEmbedder` / AdaLN modulation (`dbref/model.py:117-169`) is DiT machinery ported
  for conditioning a plain pre-norm block on `c_noise`; MORPH's block (`MORPHBlock`,
  `HyperConnectionResidual`, GLA branch) would need its own conditioning injection point —
  this file shows a working *pattern* (zero-init modulation collapsing to identity at init)
  but the actual six-way `adaLN` chunk targets are Llama-block-specific (attn/mlp scale-shift-gate).
- The **13.9x unit-vs-rms target scaling inconsistency at d=768** (PLAN.md finding 3b,
  `dbref/db.py:255-271` `scale_target`) is a property of L2-normalized token embeddings vs
  `sigma_data`; it is the exact failure mode already identified in MORPH's `SliceScaler`
  (README "The two things worth knowing", row 3), so this is less "port the code" and more
  "the testbed already diagnosed MORPH's own bug independently — read `scale_target`'s
  docstring as the explanation, not as code to reuse."
- Compute/throughput numbers (135,518 / 59,318 / 172,329 tok/s, 18.0/9.0/18.5 GB VRAM) are
  specific to this 124M Llama on this RTX 5090 at these batch/seq sizes and do not transfer
  as absolute numbers — only the finding that the two-pass concat forward costs ~2.3x per
  token generalizes as a qualitative expectation for any two-stream conditioning scheme.

## Not verified by this research pass

I read every file the task named (`PLAN.md`, `README.md`, `GATES.md`, `notes/*`, `results/*`,
`gates/*`, `dbref/*.py`, `third_party/*.py`, all `configs/*.yaml`) and the specific scripts
whose numbers are quoted above (`decodability.py`, `bridge_probe.py`, `ar_probe.py`,
`compare_finetunes.py`, `compare_arms.py`, plus grepped log lines). I did **not** re-run any
script or re-derive any number myself — every figure above is copied verbatim from the
repo's committed notes, gate evidence, JSON results, or `git log` commit bodies, with its
source file cited inline. I did not read `dbref/data.py`, `dbref/genppl.py`, `dbref/guard.py`,
or the full `tests/` suite line-by-line (only referenced them where a note or gate pointed at
a specific test name); those are unlikely to matter for the dmorph design question but were
not exhaustively reviewed. I did not open `results/samples_all_arms.md` in full (434 KB) —
only its existence and the 162.07/325.85 figure, sourced from `GATES.md`, are used.
