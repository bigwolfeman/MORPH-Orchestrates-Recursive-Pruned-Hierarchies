# Agent Note: DiffusionBlocks on MORPH — verdict
Status: rejected — it replaces next-token prediction with denoising, and denoising loses: 4.0010 CE for AR against 5.0801 for the best DB arm at a matched budget, and 162 vs 326 gen-PPL at matched output diversity

## Problem

MORPH's Parcae core loops `T ~ Poisson(6)` times and pays truncated BPTT for it. DiffusionBlocks
(arXiv:2506.14202, ICLR 2026, Sakana AI) claims that a looped network can instead be trained as a
single-pass denoiser at a sampled noise level `sigma`, deleting the loop and the BPTT from the
training step. Its §5.5 / App. E.5 applies exactly that to **Huginn**, which is MORPH's closest
published relative (2 prelude / 4 recurrent / 2 coda against MORPH's 4:6:4). If it worked, the
training step would get much cheaper and optimizer state would fall by `B`.

The MORPH implementation was built and ran, and it failed — but MORPH is a large, unusual model
(Lorentz slice in the embedding, hyper-connections, ternary QAT, MORTAR sparsity), so a failure
there could not distinguish "the method does not work" from "MORPH's complexity broke it". That
ambiguity is what this note closes.

## Proposal

Reproduce DiffusionBlocks in a clean room and measure it against its own baseline, then decide.

The testbed (`11-DiffusionBlocks-Testing`, pushed to `morph-scratch:dbref-testbed`) is a 124M
12L/768d Llama on OpenWebText — no Lorentz, no hyper-connections, no quantization, no sparsity —
with the authors' released sigma code vendored and executed by the test suite, so our schedule is
numerically equal to theirs rather than paraphrased. Every arm shares data, seed, backbone,
optimizer and token budget; a gate refuses to run if any resolved config differs outside the
objective keys.

**The decisive quantity.** As `sigma -> sigma_max` the EDM skip coefficient `c_skip -> 0`, so the
noised input `z` carries no information about the answer and the DB model is doing nothing but
next-token prediction from clean context. A DB model that works must land there at roughly the AR
baseline's CE. That is the one number the whole ladder exists to produce.

### Held-out CE, matched 143.4M-token budget

Every figure below is reproduced by `scripts/gate_ce_agreement.py` against the run's own wandb
log to 0.0000 nats, on all 15 quantities. Getting there caught a defect worth naming: the
scoring harness ran EAGER while the runs were COMPILED, and `torch.compile` is not
numerically identical to eager. On `db_b1_rms` at sigma_max — degenerate, so the CE is
sensitive — eager read 10.3618 and compiled 10.3916, and the run had logged 10.3916.

| arm | what it changes | CE @ sigma=1 | CE @ sigma_max | vs AR |
|---|---|---:|---:|---:|
| `ar` | baseline, plain next-token | — | **4.0010** | — |
| `db_b1` | the paper's recurrent-depth setting | 4.1236 | 7.3011 | +3.30 |
| `db_b1_oracle` | `sigma_data=0.5`, the authors' own value | 4.5572 | **5.0801** | +1.08 |
| `db_b1_wedm` | the paper's literal Eq. (6) weighting | 4.1705 | 7.6827 | +3.68 |
| `db_b1_rms` | MORPH's SliceScaler target scaling | 0.6733 | 10.3916 | +6.39 |
| `db_b4` @ 143M | `B=4` blocks, token-matched | 4.8625 | 5.5329 | +1.53 |
| `db_b4` @ 573M | `B=4`, the 4x budget the protocol demands | 4.2044 | 4.6740 | +0.67 |

`db_b1_rms` is the trap in one row: it has the **best** sigma-grid mean in the whole testbed
(2.8400) and is the **worst** model in it. Below `sigma* ~= ||y||/4.2` the nearest-neighbour decode
of `z` still identifies the target, so the model autoencodes its own input and the loss falls for
free. MORPH's SliceScaler pushes `sigma*` to 3.299, which puts most of the schedule into that
regime. Grid means hide this; the sigma_max column does not.

### Generation, scored under GPT2-XL

12 held-out prompts, 64 new tokens, three decodes each (greedy, top-p 0.95, pure sampling at t=1),
both embedding->token bridges, Euler steps 4/16/32. Real held-out text anchors at gen-PPL **32.44**
with distinct-2 0.983.

gen-PPL alone cannot rank these, and the testbed has the receipt: **`db_b1_rms` scored gen-PPL
1.46**, better than real English by 20x, by emitting `the the the the ...` for 64 tokens
(distinct-2 **0.016**). A scorer rewards text that is easy to predict and nothing is easier than a
loop, so every gen-PPL here is reported next to distinct-2 and rep-rate, and the ranking that
counts keeps only rows whose variety is within 0.03 of the real corpus:

| arm | decode | gen-PPL | distinct-2 |
|---|---|---:|---:|
| `ar` | top-p 0.95 | **162.07** | 0.979 |
| `ar_ft_db_noisy` | top-p 0.95 | 164.91 | 0.966 |
| `db_b4` @ 573M | greedy, soft bridge, 32 steps | **325.85** | 0.985 |
| `db_b1_oracle` | greedy, soft bridge, 32 steps | 363.24 | 0.985 |
| `db_b1` | greedy, soft bridge, 32 steps | 404.94 | 0.962 |

Every hard-bridge DB row is absent from that table. They post the lowest raw gen-PPL of any DB
setting (down to 15.98) and they get it by repeating themselves: distinct-2 0.61 to 0.82 against
the corpus's 0.983.

### A DiffusionBlocks sample has no token-level randomness

On every DB arm, top-p 0.95 and pure ancestral sampling returned **bit-identical** text from
the same seed; on the AR baseline they diverged at the first token. That is not a decoding
detail, and measuring it produced the most structural result here. At the position generation
actually draws from -- the end of the Euler walk:

| arm | top-1 prob | entropy | nucleus @ 0.95 |
|---|---:|---:|---:|
| `ar` | 0.2920 | 4.7613 nats | 8422 tokens |
| `db_b1` | **1.0000** | **0.0000** | **1** |
| `db_b1_oracle` | 1.0000 | 0.0000 | 1 |
| `db_b4` | 1.0000 | 0.0000 | 1 |

The same denoisers read directly along the trajectory are not collapsed -- `db_b1` measures
entropy 3.32 / 4.74 / 5.04 / 7.72 nats at sigma 0.3 / 1 / 10 / 80. The collapse is specific to
the low-sigma end, and it is exactly what EDM prescribes: as `sigma -> 0`, `c_skip -> 1` and
the denoiser IS the identity on its input.

So all of a DB sample's randomness lives in the initial `z` and the trajectory. Temperature
and top-p are inert. A language model built this way gives up every decode-side diversity
control, and that is a property of the method rather than of our build.

## Alternatives considered

* **Different `sigma_data`.** The authors' 0.5 (`db_b1_oracle`) is the single best DB change
  measured: 7.3011 -> 5.0801 at sigma_max. It closes two thirds of the gap and does not close it.
* **The paper's literal Eq. (6) EDM weighting** (`db_b1_wedm`): worse at all seven sigmas, by 0.38
  nats at sigma_max. The authors' released code computes `loss[B] * w[B,1]`, whose `.mean()` is
  `mean(loss)*mean(w)` — a scalar — so their shipped code is effectively unweighted and the bug is
  favourable. Unweighted is what this testbed uses.
* **More blocks (`B=4`).** The best DB configuration measured, and faster per token than plain AR
  (172k vs 136k tok/s, since 3 layers are backpropped instead of 12). But the protocol demands `B`x
  the tokens, and at 4x the budget it still lands 0.67 nats behind an AR run that saw a quarter of
  the data. Memory is level with AR at the same micro-batch, not a win.
* **Additive conditioning instead of the paper's concat.** Rejected on a measured scale argument,
  not a taste one: EDM's `c_in` holds `||c_in*z||` at `sqrt(d) = 27.71` for *every* sigma, while
  the trained embedding rows have mean norm 0.2076. From sigma >= 0.05 the additive residual
  stream is **0.01% clean context by energy** (SNR 0.0075). Concat gives the clean tokens their own
  positions, where the noise energy is exactly zero. `scripts/injection_scale.py`.
* **The embedding->token bridge**, which the paper never specifies. `softmax(logits) @ E` is a
  convex combination of embedding rows, so an uncertain model returns a SHORT vector pointing where
  no token lives (measured `||Dhat|| = 0.266` at sigma=80 against the `||y|| = 1.0` training always
  showed). The argmax bridge stays on the manifold and cuts gen-PPL by an order of magnitude — and
  then produces the repetition above. Neither choice rescues the method.
* **Converting a trained DB checkpoint back to AR.** Both honest conversions work and both lose.
  The clean stream starts at 10.8405 — literally the uniform floor, `ln(50257) = 10.8249`, because
  the final norm and the tied readout only ever saw the noisy stream — and recovers to 4.8969 in
  2000 steps. The noisy stream, which keeps the readout DB actually trained, starts at 7.3124 and
  reaches **4.3043**, against 3.9551 for the same fine-tune budget spent on the AR checkpoint. So a
  DB checkpoint is **not lobotomised**; it is 0.35 nats behind for 2.28x the compute, and the
  readout is the whole story.
* **Keeping it on `master` behind `db.activate_at: never`.** Rejected. It was genuinely inert — no
  DB parameter is constructed when the flag is off — but it cost a `db:` config block, ~50 lines of
  wiring in `transformer.py` / `train.py` / `flops.py`, four post-training modules, and five test
  files, all of which a reader has to route around forever. Parked branches cost nothing.

## Consequences

The implementation is removed from `master` and preserved in three places:

| where | what |
|---|---|
| `park/db-master-line` | `master` as it stood at `e72f84c`: `morph/model/diffusion_blocks.py`, `training/db_setup.py`, `inference/db_generate.py`, the four `posttrain/` bridge modules, `db_b1`/`db_b3`/`db_b3_massvisit` configs, five test files, four `docs/diffusionblocks-*.md` |
| `feat/db-objective-l2` | the later line: concat conditioning, the L2 objective fix, the CE arm, `morph/model/db_context.py` |
| `morph-scratch:dbref-testbed` | the clean-room reference implementation, every arm's config, and the gates |

`master` keeps the paper in `docs/references.md` §9 and the ledger row, both marked rejected. The
sigma-conditioning ternary guard, the `db:` block, and `db_mode` in `perf_metrics` are gone;
`perf/db_mode` is no longer logged. Test suite after removal: **148 passed**.

**Why this is a rejection of the method and not of our build.** The testbed reproduces the paper's
sigma machinery bit-for-bit against the authors' own code, trains the AR baseline and the DB arms
on identical data with a config-diff gate, and still measures the same failure MORPH did. The two
candidate explanations at the start were "the DB mechanism does not carry language" and "MORPH's
complexity broke it". A plain Llama with none of MORPH's machinery reproduces the failure, so it is
the first.

Nor is it a wiring bug. The anti-theater gate drives `db_b1` on one repeated batch from loss 7.8903
to 0.0084 and -- scored ABOVE `sigma*`, where `z` cannot identify the target and only the clean
context can help -- reaches CE **0.0329** at sigma=1 and 0.0499 at sigma=80, against a 1.50
threshold and `ln V = 10.82`. The objective does train the network through the context path. It
simply does not train it as well as predicting the next token does.

**What this does NOT establish, stated plainly:**

* **Budget.** 143.4M tokens is roughly 1/70 of the paper's language setting. The paper reports its
  recurrent-depth win at 3x the epochs, and `db_b4` did improve with 4x tokens (5.5358 -> 4.6738).
  Nothing here rules out a crossover far past this scale; it rules out one at this scale.
* **Scale and shape.** One model size (124M), one depth (12L), one dataset, one tokenizer.
* **Domain.** The paper's headline results are image generation. Nothing here speaks to that.
* **One deviation.** Our concat forward does not let the noisy query attend its own noisy key
  (position `i`'s residual stream already *is* `c_in*z_i`, so `z` reaches the MLP and the query
  projection regardless; only the self-value aggregation is dropped). The literal 2L single-pass
  form is implemented as `conditioning: concat2l` and was not run to a full arm.
* **Metrics not run.** MAUVE and rep4@512 were never computed; the paper reports MAUVE.
* **The MORPH-side implementation** was never trained to a budget comparable to the testbed's. The
  verdict rests on the clean-room measurements, not on the MORPH arms.
