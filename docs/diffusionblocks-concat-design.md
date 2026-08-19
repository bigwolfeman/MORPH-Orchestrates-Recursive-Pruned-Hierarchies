# DiffusionBlocks — faithful concat design (Option A, b1 pathfinder)

Status: DESIGN. No code written. Supersedes the `x0_inject` conditioning, which was
diagnosed as broken (context reached the target through a near-dead additive side channel;
the coda was starved of prediction; the readout collapsed to the centroid). See
`diffusionblocks-morph-assessment.md` §leak and the session log for the evidence chain.

Decision (Wolfe): build the **faithful** paper mechanism (App. E.4 causal consistency),
not the cheap interleave. Build **b1 recurrent-depth first** as the pathfinder, then extend
the shared machinery to b3.

---

## 1. Pivotal finding — "concat" in MORPH is NOT a single 2L sequence

The paper's AR mechanism (line 1428) concatenates a clean and a noisy copy of the sequence
and applies a modified causal mask. That works for a **plain** transformer (Llama-2), where
"position" lives in exactly one place: the causal bias added to `Q·Kᵀ`. Override the bias,
done.

MORPH encodes position in **three** structural places, none of which is an additive bias:

| mechanism | where | breaks under single-2L concat |
| --- | --- | --- |
| CCA causal conv | `attention.py:480` `_causal_conv` → `fused_cca_conv` | a length-`2L` conv convolves `noisy_0` with `clean_{L-1..}` across the boundary |
| CCA value-shift | `attention.py:513` `W_v_prev(pad(x[:, :-1]))` | `noisy_0`'s "previous token" becomes `clean_{L-1}` |
| CoPE-RoPE | `attention.py:242` position-from-index | `clean_i` (index `i`) and `noisy_i` (index `L+i`) get different phases for the same logical position |

So a faithful concat cannot be one `2L` tensor pushed through the existing kernels. The
conv, the shift, and RoPE all assume **one contiguous causal stream**.

**Consequence:** the faithful form in MORPH is the paper's own *stated alternative*
(line 1428): *"compute key-value pairs separately for clean and noisy sequences, combining
them during attention ... uses standard sequence memory."* Each stream is a normal
length-`L` causal sequence — every existing per-stream mechanism (conv, shift, RoPE, QK-norm,
sinks) runs **unchanged**. The only new thing is that **noisy-stream attention also attends
the clean stream's keys, causally**. This is cheaper than feared (standard sequence memory,
not 2×) but it is real kernel work: a **two-source causal attention**.

This is a clarification of "Option A", not a different choice. It is still the faithful
paper mechanism; MORPH's architecture just forces the 2-pass realization of it.

---

## 2. The causality surface and how each part goes two-source

Layout, MORPH next-token convention (`labels[t] = input_ids[t+1]`):

- `clean_i = embed(input_ids[i])`  — clean current token, a plain causal LM context stream.
- `noisy_i = embed(input_ids[i+1]) + σ·ε`  — the **noised target** (next token). Read out here.
- Visibility: `noisy_i` attends `clean_{≤i}` (context up to and including the current token)
  and `noisy_{≤i}` (its own causal past). It does **not** attend `clean_{i+1}` — that is the
  target `input_ids[i+1]`, so there is no leak. `clean_i` attends only `clean_{≤i}`.

| part | today | two-source change | cost |
| --- | --- | --- | --- |
| CCA conv / shift / RoPE / QK-norm | per-stream causal | **none** — run per stream | — |
| CSA compressed blocks (`fused_csa_attention`, `attention.py:763`) | noisy blocks only | compress clean keys into blocks too; noisy query selects top-k over `union(clean≤i, noisy≤i)` | +clean block compress + kernel takes 2 key sources |
| CSA window (`_window_attn`, `attention.py:602`) | noisy window | window also spans clean keys `≤i` | +1 window pass over clean k/v |
| HCA compressed blocks (`_CCAHCAAttention`) | noisy blocks | same block-union as CSA | as CSA |
| GLA retention (`gla.py`) | per-stream scan | seed the noisy scan with the **clean stream's `final_state`** (existing `initial_state`/`final_state` carry, `gla.py:19`) | none — no kernel change |
| attention sinks | per-head logit | unchanged | — |

The clean stream is a **pure causal LM forward** over clean tokens: standard MORPH, no
changes, produces per-layer `(k, v, C_comp)` that the noisy stream consumes.

---

## 3. b1 recurrent-depth mapping (the pathfinder)

1. **Clean context.** Run the prelude (and, for b1, the shared core weights in "context"
   role) on `clean` tokens as a normal causal sequence. Cache per-layer clean keys/values
   and the clean GLA final-state. This is the paper's `x` in `D_θ(z_σ, x, σ)`.
2. **Noisy state.** `z_σ = y + σ·ε` with `y = embed(next token)` (slice-scaled, as today).
   The core loops on `z` (per-sequence Poisson depth, one block over all σ). Each core
   iteration: noisy attends `clean_{≤i}` (context) + `noisy_{≤i}` (own past); GLA seeded from
   the clean final-state.
3. **Readout.** Coda on noisy positions → logits → EDM-preconditioned denoised embedding
   `D̂ = c_skip·z + c_out·hidden` → CE against `labels`. Readout uses `db_lm_weight()`
   (the scaled tied head) and the learned logit scale (centroid-collapse fix, already in).

**Why the target must be the next-token embedding (and thus why we need the split).**
Faithful EDM needs a continuous target `y` to define `c_skip/c_out` and the denoising
regression. That `y` is `embed(next token)`. It therefore lives in the noisy stream at
position `i`, and the clean answer `clean_{i+1}` must be withheld from `noisy_i` — which is
exactly what the two-source causal visibility enforces. A "state-only, CE-only" variant (no
continuous `y`) would drop EDM preconditioning and is not the paper's method; rejected as
unfaithful.

---

## 4. What is new, what is untouched

**New code:**
- Clean-stream forward that caches per-layer `(k, v, C_comp, gla_state)`.
- Two-source `fused_csa_attention` and the HCA equivalent (2 key sources; block-union top-k).
- Two-source `_window_attn` (clean + noisy within window).
- GLA state seeding across streams (wire existing `initial_state`).
- Noisy-only readout + loss (mostly exists in `db_loss`).
- Mask/visibility builder for the two-source form (the existing `clean_noisy_mask` is the
  2L single-tensor shape; it is replaced by the per-source causal test `j ≤ i`).

**Untouched:** CCA conv/prologue kernels, RoPE/CoPE, embeddings, HyperConnection residual,
AdEMAMix optimizer, ternary/int6 QAT, the σ schedule/EDM precond math, `flops.py`.

---

## 5. Risks (named, not yet resolved)

1. **KV cache / generator.** The eager generator (`inference/tul_generate.py` sibling
   `db_generate.py`) must, per Euler step, run the clean pass once then the noisy denoise.
   The clean context is fixed across the σ chain for a given prefix — cache it once per step,
   not per σ.
2. **Gradient checkpointing over two streams.** The DB path already checkpoints per layer
   (`_db_run_section`). The clean pass adds activations; checkpoint it too, or run it under
   `no_grad` if the clean stream carries no loss (it does not — only noisy positions have CE).
   **Clean stream under `no_grad` is the likely correct and cheap choice** — verify grads to
   the prelude still flow through the noisy stream's attention over clean keys (they do: clean
   k/v are inputs to the noisy softmax, so grad reaches the clean projections via that path
   even if the clean stream's *own* readout is absent). Confirm before assuming.
3. **torch.compile graph.** Two-source attention adds a second key source to the fused
   kernels; keep the graph branch-free (no `if two_source:` in forward — bake it into the DB
   module, per the project's no-runtime-flags rule).
4. **Cost.** ~2× attention compute (clean pass + noisy pass) at standard sequence memory.
   The DB memory win (one block's grads) is preserved; the clean pass is the tax for
   faithfulness. Measure it.

---

## 6. Falsifier (unchanged from the plan)

After a short train on the real mechanism: the **scrambled control** CE must climb clearly
above chance (`ln V ≈ 10.8`) — i.e. the model predicts the next token from clean context,
not from its own noised input. Clean-σ CE must **not** be trivially low (that was the
autoencoding tell). Only then do the arms (b1/b3, ±TUL, tok/s, memory) mean anything.
