# TUL Gate — specification (`TUL-gate`, `TUL-halt`)

Status: **proposed. Nothing in this document is built.**

Extends [tul-spec.md](tul-spec.md); read that first. This file adds a **span-length
gate** and an optional **halting gate** to the Thought Unpack Loop. Arms and the
pre-registered prediction are in [ablation-ledger.md](ablation-ledger.md); the
invariants in §9 belong in [runtime-invariants.md](runtime-invariants.md) §6c once
built, in the same "each row names the test that fails" form as §6b.

This document is a **build contract**. Every section states what the code must do
and, in §9, the check that fails when it does not. If the implementation disagrees
with this file, the implementation is wrong until this file is changed on purpose.

---

## 1. The two gates are separate mechanisms

| | `k > 0` — span length | `k = 0` — loop again |
| --- | --- | --- |
| what it decides | how many tokens this plan covers, 1…`gate_k_max` (32) | whether the core takes another iteration on this slot |
| label at training time | **the data's own span length**, from `BoundaryRule.cut` | none exists (§7) |
| changes the data layout? | yes — the coda is conditioned on it, and truncation re-cuts spans | no |
| built in round 1 | **yes** (`TUL-gate`) | trained, but does **not** drive generation (§7) |

Wolfe's framing is one scalar `g ∈ [0,1]`, `k = round(g · 32)`, `k = 0` meaning
"no tokens yet, keep thinking". §4 keeps that interface. §7 records why round 1
does not let `k = 0` steer generation, with the arithmetic.

**Why the length gate is self-supervised and needs no reward.** In pretraining we own
the segmentation, so the correct span length is already computed, causally, by the
boundary rule. The predecessor project needed a reward only because it was bolted onto
a frozen decoder where "did the span come out well" was the sole available signal.

---

## 2. What the predecessor actually built (`00DeepNet/coconut`, mined 2026-08-21)

Facts, for the parts we inherit or must not repeat. Line references are that repo's.

- The gate existed in the remembered shape: `g = sigmoid(gate(z))`,
  `k = round(g · k_max).clamp(0, k_max)`, `k_max = 32` in every real run, `k = 0`
  = loop again (`tul/heads.py:63-64`, `tul/infer.py:157-160`).
- It was trained by **supervised Huber regression**, not RL: target
  `g* = min(len(thought), k_max) / k_max` (`tul/train.py:331-336`,
  `tul/segment.py:299-307`). A later GRPO stage moved the Llama numbers by nothing
  (EM 0.18 → 0.18).
- **The decoder was never told `k`.** No embedding, no prefix token, no mask; at
  generation `k` was a Python loop bound (`tul/infer.py:442-443`). §5 fixes this.
- **Partial spans were never trained.** `phase_jitter` was implemented
  (`tul/segment.py:86,94-105`) and all 15 saved run configs read `phase_jitter: 0.0`.
  §3 turns it on.
- **The gate was dead for a whole ladder.** Bias −2.00000 → −2.00071 against a required
  travel of 1.88, 57× short of the LR budget; an earlier "the gate chose 9–13 tokens"
  claim is retracted in that repo. §10 is the instrumentation that makes this
  impossible to miss again.
- When it did train, the supervised gate worked: chosen `k` mean 8.71 / p50 9 against
  gold 8.89 / p50 9.
- Outcome: TUL lost to its CoT baseline on hard ProsQA, 0.7080 vs 0.8600 greedy
  (gap −0.152 [−0.216, −0.084]) — with no length conditioning and with `phase_jitter`
  off, i.e. with both of §3 and §5 missing.

---

## 3. Data — two augmentations, and the new layout fields

The boundary rule is unchanged. `BoundaryRule.cut` stays the ONE causal function used
by the loader and the generator (§9 invariant 1 of `runtime-invariants.md` §6b still
holds). Two augmentations are applied **after** `cut`, in the loader only.

### 3.1 Start jitter (`jitter_p`) — the label is kept

A span may begin part-way through a unit and still **end on the unit's boundary**. The
length label is then "how many tokens of this unit remain", which is still exactly
data-derived. This is the predecessor's `phase_jitter`, which was built and never run.

### 3.2 End truncation (`truncate_p`) — the label is masked

A span may be cut **short of** its boundary, so it ends without punctuation and the
NEXT span starts mid-unit. This is the only thing that teaches the coda to stop on a
budget rather than on punctuation, and it is the training-time image of a generation
where the model asks for nine tokens and the ninth is not a period.

The truncation point is **our** RNG, not the data's, so a length head graded on it
would be graded on noise. Those slots are excluded from the length loss (§6).

### 3.3 New `SlotLayout` fields

Added to `morph/model/tul_layout.py`, filled by `pack_tul_row` / `pack_tul_batch`:

| field | shape | meaning |
| --- | --- | --- |
| `span_len` | `[B, max_slots]` int64 | tokens this slot's span covers, 1…`gate_k_max`. 0 for pad slots. |
| `len_supervised` | `[B, max_slots]` bool | True when `span_len` is the data's answer (normal or start-jittered), False when end-truncated or pad. |

`L_total = tokens + prefix_k · slots` stays fixed per curriculum stage. Neither
augmentation may change `L_total`; both change only where boundaries fall inside it.

---

## 4. Forward — training

Teacher forcing on **both** gate quantities: the loop count is the Poisson draw and the
coda is conditioned on the realised span length. The gate never steers training. This
is what keeps per-slot Poisson depth, and with it the depth robustness that
`tul-spec.md` §3.3 relies on for the deferred free KL-exit.

```python
# ── unchanged: front, gather, per-slot Poisson depth ──────────────────────
x                  = self._tul_front(input_ids, layout)      # prelude, ALL positions
xn, h_slots, depths = self._tul_core(x, x0, bigram_emb, layout)
#   depths[b, i] ~ Poisson(slot_mean_depth), clamped [1, slot_max_depth]  ← UNCHANGED
#   the loop is a MASKED update over the full compact slot sequence; frozen slots are
#   still computed and still serve K/V (runtime-invariants §6b).            ← UNCHANGED

# ── NEW: the gate head runs at EVERY iteration, inside the loop ───────────
#   _tul_core additionally returns g_traj [B, max_slots, T_max], the per-iteration
#   gate output. It is threaded out of the checkpointed region exactly like
#   ret_state is (transformer.py `_core_step` returns it), never captured by a
#   side channel.
g_traj = sigmoid(gate_head(h_t))        # [B, max_slots] per iteration t, stacked

# ── NEW: budget conditioning of the coda ──────────────────────────────────
#   TRAINING uses the REALISED length, never the head's own prediction: mixing them
#   makes the LM loss chase the gate's error and the target moves.
h_slots = h_slots + budget_embed(layout.span_len)     # [B, max_slots, 1, C], broadcast
                                                      # over the n HC streams, exactly
                                                      # as _apply_injection broadcasts
carrier = scatter_slots(h_slots, xn, layout)          # unchanged
y       = self.coda(carrier)                          # unchanged; slots are a PREFIX
```

`budget_embed` is `nn.Embedding(gate_k_max + 1, d_model)`, **zero-initialised**, so at
step 0 the arm is bit-identical to `TUL-A1`. Index 0 is reserved for pad slots.

---

## 5. Why the coda must be told the budget

If the coda is not conditioned on `k`, the length head's output changes nothing
downstream, its gradient comes only from its own loss, and the two halves of the model
never have to agree. That is precisely the predecessor's configuration, and it is why
"decode to the period versus stop short" was never actually a trade-off there: `k` was
a `for` bound at inference and absent from every forward.

Conditioning also introduces the **leak** that §10 must watch: told the true length, the
coda learns an easier token task than it will face at generation, where the budget is
the model's own guess. `truncate_p` blunts the leak (the budget stops being a reliable
punctuation oracle). Scheduled sampling — feeding the head's prediction on a rising
fraction of steps — is specified here and **deferred**: it is not in round 1.

---

## 6. Losses

The token cross-entropy is unchanged: ONE weighted `fused_linear_cross_entropy` over
every position, per `tul-spec.md` §5. The gate adds one term.

```python
# g_star[b, i, t] = 0                              for t <  depths[b,i] - 1
#                 = span_len[b,i] / gate_k_max     for t == depths[b,i] - 1
# masked entirely for t >= depths[b,i]  and for pad slots.
loss_gate = huber(g_traj, g_star, reduction="none")

# rows whose length label is OUR rng (end-truncated) supervise the ZEROS but not the
# final VALUE: the timing target is legitimate there, the length target is not.
value_mask = layout.len_supervised.unsqueeze(-1) & at_final_iteration
zero_mask  = slot_valid.unsqueeze(-1) & before_final_iteration
loss_gate  = (loss_gate * (value_mask | zero_mask)).sum() / (value_mask | zero_mask).sum()

loss = loss_tokens + gate_lambda * loss_gate
```

`gate_lambda = 0` must be **bit-identical** to `TUL-A1` (§9).

Huber, not squared error, follows the predecessor — the one part of its objective that
demonstrably worked (gate `p50 9` against gold `p50 9`).

---

## 7. `k = 0` at generation: trained, not trusted, in round 1

Round 1 trains the zeros and does **not** let them choose the depth at generation.
Generation uses `slot_mean_depth`, exactly as eval does today. The reason is arithmetic,
not caution.

The Poisson draw is independent of the input, so a head regressed on it converges to the
hazard, `E[g | t] = P(T = t | T ≥ t) · L / gate_k_max`. With `mean_depth 6` and the
measured mean span of 19.9 tokens (`tul-spec.md` §3.1), `L / 32 ≈ 0.62`, and the stop
test `k ≥ 1` needs only `g > 0.0156`:

| t | hazard | predicted `g` | `k = round(g · 32)` |
| --- | --- | --- | --- |
| 1 | 0.015 | 0.009 | 0 |
| 2 | 0.045 | 0.028 | **1 → halts** |
| 3 | 0.089 | 0.055 | 2 |
| 6 | 0.23 | 0.14 | 5 |

It would halt at `t = 2` and emit one token, every time — at a converged, low gate loss.
Raising the threshold does not save it: a hazard of 0.5 is never reached inside
`max_depth 8`, so the rule flips to always-cap. The defect is the **encoding** — one
scalar meaning both "do not stop" and "emit zero tokens" — so an arbitrarily small
hazard clears the stop test. The fix, when we get there, is to split the stop decision
from the length value, not to retune a threshold.

`TUL-halt` (§8) is the arm that turns `gate_drives_depth` on and measures this directly
instead of arguing about it.

**A second reason the halting arm's ceiling is low.** Per-slot depth is a masked update
and frozen slots are still computed (`runtime-invariants.md` §6b), so variable depth
saves **no** compute in batched inference — the batch runs `max(T)` iterations either
way. Halting only saves real work in single-stream generation
(`morph/inference/tul_generate.py`). Its case therefore rests entirely on quality, and
§7's first half says there is no cheap label to learn a quality policy from.

---

## 8. Generation

```python
while not done:
    h = core_loop(slot_state, T = slot_mean_depth)     # round 1: fixed, as eval today
                                                       # TUL-halt: until k > 0, cap max_depth
    k = clamp(round(sigmoid(gate_head(h)) * gate_k_max), 1, gate_k_max)
    h = h + budget_embed(k)                            # the model's OWN choice here
    emit up to k tokens from the coda with h as prefix
    # stop early if the boundary rule fires inside k — KEEP the boundary token and feed
    # it to the cache BEFORE breaking (the predecessor's infer.py:441,459-461).
    # if token k is not a boundary, the next span simply starts mid-unit — which is
    # exactly what §3.2's end-truncated rows taught.
```

`BoundaryRule.cut` remains the same resumable state machine here as in the loader
(`runtime-invariants.md` §6b invariant 1). A wrong `k` must not desynchronise anything:
the next slot starts at the next token, whatever it is.

---

## 9. Invariants (move to `runtime-invariants.md` §6c when built)

| Invariant | Why / the test that fails |
| --- | --- |
| `gate_lambda = 0` and `gate_budget_cond = False` is **bit-identical** to `TUL-A1`: same loss, same parameters, same gradients. `budget_embed` is zero-initialised and constructed last, so no RNG draw moves. | The reference arm and every pre-gate checkpoint must reproduce. Mirrors the existing `test_tul_params_do_not_perturb_the_plain_path`. |
| **`TUL-gate` puts a loss on the slot core state, and that is a deliberate change to a live invariant.** `runtime-invariants.md` §6b says "slot core states have no loss". The exception is narrow: a **scalar/discrete readout** (length, stop) is allowed; a **vector regression onto a target representation** stays forbidden. | The §6b citation set (MegaByte, H-Net, LD4LG, Pred-Sent, the LTD think-position failure, Block Transformer §4.2) is about reconstructing the latent, not reading a scalar off it. The distinction must be tested, not assumed: a test asserts no loss term has the slot state as a regression **target**. |
| `span_len` is 0 and `len_supervised` is False at every pad slot, and pad slots contribute exactly zero to `loss_gate`. | A pad slot's `slot_index` is 0, so a missing validity mask silently trains on the previous row's last token — the existing §6b pad-slot trap. |
| An end-truncated slot supervises the ZEROS but never the length VALUE. | The truncation point is our RNG. Grading a length head on it injects pure label noise, which is the same defect §7 rejects for the halting target. |
| The coda receives the **realised** length in training and the **predicted** length at generation, and never a mixture. Scheduled sampling is not built and its config key RAISES. | Mixing makes the LM loss chase gate error. A silently ignored key is worse than a missing one (existing §6b rule). |
| Neither augmentation changes `L_total`; token counts per row still vary and are still logged. | Fixed shapes for kernels and graphs (existing §6b). |
| `g_traj` leaves the checkpointed core region as a **return value**, never a side channel. | The existing `ret_capture` lesson: side-channel capture is not checkpoint-safe on its own. |
| `gate_drives_depth = False` in round 1; when True, the loop still caps at `slot_max_depth` and every slot emits `k ≥ 1`. | §7. A slot that never halts must not hang the generator. |

---

## 10. Instruments — required from run 1, not added after

Every one of these exists because the predecessor lost a ladder without them.

| instrument | what it does | source |
| --- | --- | --- |
| `audit_travel` / `required_lr_mult` | **Refuses to start** a run whose parameter groups provably cannot move within the step budget. | `coconut/tul/optim.py` — this is what finally caught the dead gate. |
| gate bias seating | Initialise the gate bias at the corpus base rate instead of making it travel there. | `coconut/segment.gate_target_stats` → `stage1.seat_gate_bias`. |
| `gate_separation` | Mean `g` on final-iteration slots minus mean `g` on earlier iterations. A gate that does not discriminate is dead however low its loss is. | Defined in the predecessor and **never measured**. |
| hazard curve | `mean g` per iteration index, logged every eval. §7's table is the prediction; this is the observation. |  |
| chosen-`k` histogram vs gold | mean / p50 of predicted length against `span_len`. | The predecessor's one clear success signal (8.71/9 vs 8.89/9). |
| generation metrics | rep4@512, distinct-3, mean span length, fraction of spans ending on a boundary. | The teacher-forcing leak in §5 hides behind a good val CE; only generation exposes it. |

---

## 11. Arms and the pre-registered prediction

| arm | depth at generation | gates | isolates |
| --- | --- | --- | --- |
| `TUL-A1` (exists) | `mean_depth`, fixed | none | reference |
| `TUL-gate` | `mean_depth`, fixed | length head + budget-conditioned coda + jitter/truncation | does a model-chosen span length pay? |
| `TUL-halt` | gate-driven | `TUL-gate` + `gate_drives_depth` | does variable depth pay on top? |

**Pre-registered, 2026-08-21, before any arm is run [W]:** `TUL-halt` does not beat
`TUL-gate` on val CE. Fixed depth wins or ties, and we ship fixed depth because it keeps
inference shapes static.

**Falsifier:** `TUL-halt` beats `TUL-gate` on val CE by more than the `A1`/`A1r` retrain
noise floor **and** does not lose on the generation metrics.

A results note is written for these arms whether they win or lose. The predecessor
missed both of its pre-registered numbers and no results note was ever written; that is
the failure this line exists to prevent.

---

## 12. Not built — these config keys RAISE

`gate_scheduled_sampling`, `gate_stop_head` (the split stop/length encoding of §7),
`gate_ponder_lambda` (a true PonderNet term needs the loss at every candidate halting
step, i.e. the coda run `T` times over all positions, and a readout the §9 invariant
restricts). Specified here so the design is on record; not implemented, and a config
key that is silently ignored is worse than a missing one.
