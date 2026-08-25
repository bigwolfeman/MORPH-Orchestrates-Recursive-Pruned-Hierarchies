# SCSE spec — Source-Centered State Evolution in the MORPH core loop

Status: SPECIFICATION. Written 2026-08-25, BEFORE the implementation, so the code can be
checked against it rather than the other way round.

Source: **"Looped Transformers with Source-Centered State Evolution"**, Bum Jun Kim, Kohei
Hayashi, Shunsuke Kamiya, Masanori Koyama, Yusuke Iwasawa, Yutaka Matsuo, arXiv:2607.27656,
30 July 2026. Equation, table and listing numbers below are the paper's.
Decision record: [.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md](../.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md).

## 1. What the paper claims, and where

Quoted so the implementation is checked against the paper and not against my summary of it.

* **The gain comes from two things.** Abstract: the ablations "identify the learned anchor and
  the anchor-coordinate deviation recurrence as the primary contributors to the gain."
  A port that implements neither is not a test of the method. This is exactly the error that
  produced [H23](experiments/failures/2026-08-25-scse-stage1-initial-deviation.md).
* **The gains are not confined to depth extrapolation.** Table 1, WikiText-103, 95.6M, at
  `T = 8` — inside the training loop-depth range — baseline 117.1 -> SCSE **96.9** PPL.
  At 50M: 151.1 -> **123.1**.
* **`b_t` is the paper's own term.** Eq. 1: "Evaluating `T_t` at zero deviation ... gives the
  next deviation `b_t(e) := T_t(0; e)`, which we call the **zero-deviation forcing bias**."
  It is a dynamical-systems forcing term. It is not teacher forcing and not exposure bias.
* **The anchor ablation.** Table 2, WikiText-103: learned anchor `h* = e + a_omega(e)` **200.10**;
  raw embedding anchor `h* = e` 206.16; frozen random projection 203.78; and
  `h* = H_0(e)` **294.37**, which collapses `Delta_0` to zero, freezes the loop under the mask,
  and makes PPL independent of depth. That last row is the failure mode this port must not hit.

## 2. The method, exactly

Per batch element `b`, with the anchor `h*` computed once and held fixed for the whole unroll:

    (Eq. 2)  h*        = e + a_omega(e)
             H_0(e)    = e + init_delta_scale * init_delta_proj(e)
             Delta_0   = H_0(e) - h*
    (Eq. 3)  q_t       = s * G_theta(Delta_t)
             D_{b,t}   = || Delta_t^{(b)} ||_F^2
    (Eq. 4)  m_{b,t}   = 1{ D_{b,t} > eps }
             qbar_t    = m_{b,t} * q_t
    (Eq. 5)  Delta_t+1 = Delta_t + qbar_t
             h_T       = h* + Delta_T
             logits    = C_psi( RMSNorm( h_T ) )

Constants, from the paper's main experiments and Listing 1:

| symbol | value | source |
| --- | --- | --- |
| `s` (residual step scale) | **0.50** | "the residual step scale fixed at s = 0.50" |
| `anchor_scale` | **0.1** | Listing 1 default |
| `init_delta_scale` | **0.1** | Listing 1 default |
| `eps` (mask threshold) | **1e-8** | "the mask threshold is eps = 1e-8" |
| `leak` | **0** | "For SCSE, the deviation-leak coefficient is fixed at lambda_leak = 0" |
| `kappa` / `cond_proj` | **0 / None** | "Set cond_proj=None, kappa=0, and leak=0 for SCSE" |

Listing 1, reproduced from the PDF:

```python
def scse_step(h, anchor, core, *, step_scale=0.5, leak=0.0,
              cond_proj=None, kappa=0.0, eps=1e-8):
    delta = h - anchor
    recurrent_input = delta
    if cond_proj is not None and kappa != 0.0:
        recurrent_input = delta + kappa * cond_proj(anchor)
    q = step_scale * core(recurrent_input)
    delta_norm_sq = delta.pow(2).sum(dim=(1, 2), keepdim=True)
    active = (delta_norm_sq > eps).to(q.dtype)
    q = q * active
    return anchor + (1.0 - leak) * delta + q

def scse_unroll(input_ids, *, embed, anchor_proj, init_delta_proj,
                core, readout, loops, anchor_scale=0.1,
                init_delta_scale=0.1, **step_kwargs):
    e = embed(input_ids)
    anchor = e + anchor_scale * anchor_proj(e)
    h = e + init_delta_scale * init_delta_proj(e)
    states = [h]
    for _ in range(loops):
        h = scse_step(h, anchor, core, **step_kwargs)
        states.append(h)
    return readout(h), states, anchor
```

Three facts to read off that listing, because each is a place an implementation goes wrong:

1. **`core` receives `delta` alone.** Not `h`, not `e`, not the anchor. The recurrence is
   source-free. This is "the anchor-coordinate deviation recurrence" the abstract credits.
2. **The mask is per EXAMPLE**, not per position: `sum(dim=(1,2), keepdim=True)` over a
   `[B, S, C]` tensor gives `[B, 1, 1]`. The paper's text agrees: "The per-example mask".
3. **`anchor` is computed once, outside the loop.** "The line computing anchor is executed
   once before the loop, so anchor_proj defines the fixed reference point; the loop evolves h
   in deviation coordinates around that reference rather than repeatedly regenerating or
   adding the anchor."

Why `Delta = 0` is an exact fixed point (paper, after Eq. 5): `Delta = 0` gives `D = 0`, so
`m = 0`, so `qbar = 0`, so `Delta_{t+1} = Delta_t = 0` and `h_{t+1} = h_t = h*`. Hence
`T_t(0; e) = 0`. The designated anchor is a one-step fixed point **by construction**.
The paper further notes that with a zero-preserving core the raw response is already zero
before masking: "Because RMSNorm maps zero to zero and every attention and multilayer-perceptron
projection in the core is bias-free, `G_theta(0) = 0`; hence `b~_t(e) = 0` even before masking.
Therefore, the source-centered, zero-preserving core is the primary reparameterization. The
mask supplies the exact pointwise boundary condition even if the underlying core is not
zero-preserving."

## 3. Mapping onto MORPH

MORPH today is an additive-source looped Transformer, which is the paper's baseline family.
Its core map (`transformer.py::_apply_core_step`) is

    h_injected = self.injection(h_in, e_in)                    # DiagonalInjection: ctx <- A*ctx + dt*e_ctx
    for i, layer in enumerate(self.core):
        h_injected = self._apply_injection(h_injected, term_i) # + x0/bigram, per core layer
        h_injected = layer(h_injected, ...)
    return h_injected

so the source `e` and the `x0`/bigram terms enter on **every** iteration. Under SCSE both
leave the loop and the source is carried by the anchor instead.

### 3.1 The one design rule that keeps the blast radius small

**The deviation exists only inside the loop.** Entry `Delta_0 = H_0(e) - h*`, exit
`h = h* + Delta_T`. Both loop bodies (`_core_region` for the token path, `_tul_core` for the
TUL slot path) return an ABSOLUTE carrier exactly as today, so the prelude, the coda, the
readout, the scatter and every checkpoint key are untouched.

**One exception, and it is not "untouched":** the TUL halting-gate readout is moved. It reads
the slot STATE, so under SCSE it is fed `h* + Delta` rather than the raw carrier, which is now
the deviation. That is the correct choice — a halting head scoring a deviation is scoring a
different quantity — but it IS a change, and an earlier draft of this section wrongly listed
the gate head as untouched. The arms do not build the gate, so nothing measured here depends
on it.

### 3.2 The SCSE core map

    G_theta(Delta) := stack(Delta) - Delta,
    where stack(Delta) := the n_core shared blocks applied to Delta, NO source injection.

Concretely: skip `self.injection` and skip `_apply_injection` for the core layers, leaving
`for layer in self.core: x = layer(x)` — and then **subtract the input**.

**The subtraction is load-bearing, and the first version of this port omitted it.** Found by
audit on 2026-08-25; the corrected form is `_SCSE.update`.

The paper's `G_theta` carries no top-level identity — its residual has been hoisted to loop
level, which is exactly what the name "residual step scale" means. The paper never says this
in one sentence, so here is the argument from its own text. Its tuned adapter is
`h_{t+1} = h_t + s*B_theta(h_t + alpha*W_in*h*)`. If `B_theta` contained the identity, that
map would gain `(1+s)` every step and reach `1.5^48 ~ 1e8` at the `T = 48` the paper
evaluates at. Its T = 48 numbers are ordinary, so it does not.

MORPH's core blocks are full residual blocks — the HyperConnection carrier passthrough is
INSIDE `stack` — so `Delta + s*stack(Delta)` applies a residual twice. Measured on a real
SCSE checkpoint at the converged operating point (iterations 3-5):

| iteration | `\|\|stack(D)\|\|/\|\|D\|\|` | `cos(stack(D), D)` | `\|\|stack(D)-D\|\|/\|\|D\|\|` |
| --- | --- | --- | --- |
| 3 | 0.921 | 0.780 | 0.642 |
| 4 | 0.874 | 0.838 | 0.546 |
| 5 | 0.902 | 0.883 | 0.470 |

`cos = 0.88` is the identity. One-step norm gain: the doubled form **1.414x per iteration**
(about 16x over eight), the corrected form **0.923x**. Against this repo's standing model of
its own failure mode — `rho(J_core)` crossing 1 is the disease
([iterative-map-dynamics](../.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md))
— the doubled form builds the disease into the recurrence.

Equivalently `Delta_{t+1} = (1 - s)*Delta_t + s*stack(Delta_t)`, so `s` is a damping factor
between "no update" at `s = 0` and MORPH's own core map in deviation coordinates at `s = 1`.
That reading is pinned by a test. Zero-preservation survives: `stack(0) = 0` gives `G(0) = 0`.

`G_theta(0) = 0` for this map is already verified numerically on the real 286.1M `tul_a1`
model, fp32 and bf16 autocast, peak `|out| = 0.000e+00`
(`tests/test_scse_core_init.py::test_zero_carrier_returns_zero_on_the_REAL_model`, which
zeroes carrier, source and injection terms together and therefore measures exactly this
source-free stack). **That test does NOT show MORPH's current core is zero-preserving.** It
is not: `injection(0, e) != 0` whenever `e != 0`. The distinction matters and an earlier
writeup blurred it.

### 3.3 Parameters added

| name | shape | note |
| --- | --- | --- |
| `scse.anchor_proj` | `Linear(d_model, d_model, bias=False)` | `a_omega` |
| `scse.init_proj` | `Linear(d_model, d_model, bias=False)` | `init_delta_proj` |

Both are built **last** in `__init__`, after every other module, so that with SCSE off the RNG
draw sequence for all other parameters is unchanged and the OFF path stays bitwise identical to
master. This is the same rule `_SCSEInit` already follows.

### 3.4 Config

    model.scse_enabled: false        # master switch, construction time
    model.scse_step_scale: 0.5       # s
    model.scse_anchor_scale: 0.1
    model.scse_init_scale: 0.1
    model.scse_eps: 1.0e-8
    model.scse_kappa: 0.0            # 0 -> SCSE proper; > 0 builds cond_proj (SC-Cond control)

`model.core_init_scale` (the Stage 1 field) is IGNORED when `scse_enabled` is true, because
SCSE defines the initial state itself. Setting both must RAISE rather than silently pick one.

## 4. Deviations from the paper, and why each one is taken

Every deviation is listed. If it is not here, it is not intended.

**D1 — the carrier has a stream axis.** MORPH's carrier is `[B, S, n_streams=4, C]`
(HyperConnection, Cayley n=4); the paper's is `[B, S, C]`. `anchor_proj` and `init_proj` apply
over the last axis, so all four streams share one weight matrix and are transformed
independently. The mask sums over `(1, 2, 3)` to stay **per example**, which is what the paper
specifies; summing over fewer axes would silently make it per position or per stream.

**D2 — the projections are `bias=False`, where Listing 1's `nn.Linear` defaults to `bias=True`.**
Reason: under TUL, `gather_valid` zeroes pad slots, so `e = 0` there. A bias would put
`h*` and `Delta_0` off zero at pads and give padding a forward effect. With `bias=False`,
`e = 0` gives `h* = 0` and `Delta_0 = 0` at every pad. This is also strictly closer to the
paper's own zero-preserving design rule. It does not degenerate the method: at real positions
`Delta_0 = init_scale*init_proj(e) - anchor_scale*anchor_proj(e)`, which is non-zero for
independently initialised projections.

**D3 — `DiagonalInjection` is dropped from the SCSE loop, not fed a zero source.**
Feeding `e = 0` would leave `h_ctx <- A * h_ctx` with `A ~= 0.447`, which multiplies the
deviation's context channels by ~0.447 every iteration and annihilates them over 8 steps
(`0.447^8 ~= 1.6e-3` at the INIT value); in the baseline that decay is refilled by
`dt * e_ctx` each iteration, and under SCSE there would be nothing to refill it.
**Caveat, added after audit:** `log_A` is a learnable parameter clamped at 0.9999, so a
zero-fed injection could in principle learn `A -> 1` and become a no-op rather than a decay.
The arithmetic above is the init-value behaviour, not a proof. The deviation is taken on the
stronger ground that the paper's `core` is the bare block stack, not on the decay argument. Dropping the module makes the core the shared
block stack, which is what the paper's `core` is ("RMSNorm pre-normalization, causal
self-attention, and SwiGLU updates"). Consequence: `self.injection` and
`x0_injects[n_prelude : n_prelude+n_core]` receive no gradient when SCSE is on. Their signal is
not lost — bigram and `x0` still reach the model through the prelude injections, the coda
injections, and `e` itself, which is the prelude output.

**D4 — the mask norm accumulates in fp32.** The training path runs bf16 autocast, and
`Delta.pow(2).sum()` over `S*4*C ~= 4096*4*768` elements in bf16 loses the precision the
`eps = 1e-8` comparison needs. The comparison is made against the same `eps`. This makes the
test stricter, never looser.

**D5 — depth is Poisson-sampled per sample, and MORPH freezes finished samples.** The paper
samples loop depth uniformly from `{1,...,8}`; MORPH samples Poisson (mean 6, max 8) and holds
`bptt_depth = 4`. Both are standing user vetoes and are NOT changed for this port. The
deviation update must respect MORPH's existing freezing: on the token path the active set is a
sorted prefix, on the TUL path it is a `torch.where` masked update over the full slot sequence.
`Delta` follows the carrier's freezing rule exactly.

**D6 — the core-gain governor is incompatible and must raise.** `core_gain_clip` caps
`||h_new|| / ||h_old||`. Under SCSE the loop carries `Delta`, whose norm is a different
quantity, so the same tau means something else. The arms run `core_gain_clip: 0.0` (dormant).
Enabling both must raise at build time rather than quietly change meaning.

**D7 — `s = 0.50` is the paper's value for a ONE-block core.** MORPH applies `n_core = 6`
blocks per recurrent step, so the per-step gain is not comparable and `s` may need its own
sweep. The port uses 0.50 as specified; the first arm is not a tuning run.

**D8 — `Delta_0` is formed directly, not as `H_0(e) - h*`.** The two expressions are equal
in real arithmetic because the `e` terms cancel, so the implementation evaluates
`init_scale*init_proj(e) - anchor_scale*anchor_proj(e)`. Forming the literal difference in
bf16 would subtract two tensors of the carrier's magnitude to recover a quantity about 20x
smaller and lose most of its significant bits. It also makes `Delta_0` EXACTLY zero wherever
`e` is zero, which is what invariant S8 (TUL pad slots) depends on.

## 4b. Other behaviour SCSE changes (named, after audit)

These were real changes that lived only in code comments until the 2026-08-25 audit. None is
load-bearing for the arms, and all are listed so no reader has to discover them.

* **The GLA retention state is NOT frozen by the mask.** The anchor is a one-step fixed point
  of `Delta`, but the model's full state is `(Delta, ret_state)` and the retention carry keeps
  evolving even on an example whose gate is 0. The paper has no retention branch, so it says
  nothing about this. In practice a masked example has `Delta = 0`, the core runs on zeros and
  `G(0) = 0`, so the state it feeds forward is the zero-input GLA state rather than a frozen
  one. Worth knowing before anyone reads a "frozen" example as fully frozen.
* **`_diag_corecos` and the PERITER gain log now measure the DEVIATION, not the state.** Same
  numbers, different quantity. Both are off by default.
* **The eval-only trajectory capture reconstructs `h* + Delta`** so the forecastability probe
  keeps receiving absolute states. It relies on eval running a uniform depth, which makes the
  permutation the identity; that assumption is already load-bearing in the existing code.
* **`scse_kappa > 0` is NOT exactly the paper's SC-Cond reference.** It builds `cond_proj` but
  not the `leak = 0.02` term the paper pairs with it in the secondary diagnostic. It is a
  source-conditioned variant, and calling it "the SC-Cond control" without that caveat would
  overclaim. It defaults to 0 and no arm uses it.

## 5. Invariants the implementation must satisfy

Each one names the check that fails when it breaks.

| # | invariant | test in `tests/test_scse.py` |
| --- | --- | --- |
| S1 | `scse_enabled: false` is bitwise identical to master: same weights, same logits | `test_scse_off_builds_nothing`, `test_scse_construction_does_not_move_the_rng`, `test_scse_off_logits_are_unchanged` |
| S2 | the anchor is computed ONCE per forward and held fixed | `test_anchor_is_built_exactly_once_per_forward`, `test_tul_and_token_paths_both_reach_the_anchor` |
| S3 | the core sees the deviation only: no `e`, no `x0`/bigram term, no `DiagonalInjection` | `test_core_is_source_free_under_scse`, `test_core_gets_no_x0_or_bigram_term_under_scse` |
| S4 | `Delta_0 = 0` implies `Delta_t = 0` for all `t` | `test_zero_deviation_is_a_fixed_point`, `test_live_model_is_not_at_the_fixed_point` |
| S4b | the MASK freezes a non-zero deviation below threshold | `test_mask_freezes_a_below_threshold_deviation` |
| S5 | the mask is per example, over the squared Frobenius norm, surviving bf16 | `test_mask_is_per_example_not_per_position`, `test_mask_threshold_is_the_squared_frobenius_norm`, `test_mask_norm_survives_bf16` |
| S6 | the loop exit returns `h* + Delta_T`, not the bare deviation | `test_core_region_reconstructs_the_absolute_state`, `test_tul_core_reconstructs_the_absolute_state`, `test_h_star_plus_delta0_equals_H0_of_e` |
| S7 | `b_t = T_t(0; e) = 0` exactly on the REAL 286M model, fp32 and bf16 | `test_forcing_bias_is_zero_on_the_real_model` |
| S8 | TUL pad slots enter at `h* = 0` and `Delta_0 = 0` exactly | `test_pad_slots_enter_at_zero` |
| S9 | a real optimizer step runs with the now-dead injection params at `grad is None` | `test_optimizer_step_runs_with_dead_injection_params` |
| S10 | both loop bodies are ported, and it fires on the SHIPPED model | `test_tul_slot_path_uses_scse`, `test_real_model_loop_is_source_free_and_anchored` |
| S12 | `G(D) = stack(D) - D`: the block residual is NOT applied twice | `test_update_subtracts_the_deviation_from_the_stack_output` |
| S13 | the first core block receives the BARE deviation, not the anchor | `test_the_core_receives_the_BARE_deviation` |
| S14 | the mask fires on the TUL path, which is the path the arms run | `test_mask_freezes_a_below_threshold_deviation_on_the_TUL_path` |
| S15 | `h*` is added in ORIGINAL batch order, under a non-identity depth sort | `test_h_star_is_aligned_with_the_ORIGINAL_batch_order` |
| S11 | the construction guards raise instead of silently changing meaning | `test_scse_and_stage1_are_mutually_exclusive`, `test_scse_rejects_the_core_gain_governor`, `test_scse_rejects_a_coreless_model`, `test_kappa_zero_builds_no_cond_proj` |

**S12-S15 exist because an independent audit broke the implementation four ways and the
whole 384-test suite stayed green.** Three of those (core fed the anchor, mask deleted from
the TUL path, `h*` permuted) were live gaps; the fourth is the recurrence bug that audit
found. The lesson recorded here: the original S3 tests counted calls to MORPH's LEGACY source
modules rather than reading the tensor the blocks actually receive, and every other test ran
at uniform eval depth where the batch permutation is the identity.

**S4 and S4b are separate on purpose.** A sabotage pass on 2026-08-25 disabled the mask
entirely and S4 still passed: MORPH's core is zero-preserving, so `Delta_1 = 0 + s*G(0) = 0`
by the core alone. That is the paper's own claim ("the source-centered, zero-preserving core
is the primary reparameterization; the mask supplies the exact pointwise boundary condition
even if the underlying core is not zero-preserving"), and it means S4 cannot see the mask.
S4b is the test that can. Every invariant here was checked by breaking the implementation
and confirming the named test fails: 9 sabotages, 9 caught after S4b was added.

**`leak` is deliberately not implemented.** Listing 1 carries a `(1 - leak) * delta` term,
and the paper fixes `lambda_leak = 0` for SCSE — it is non-zero only in the secondary leak
diagnostic, which this port does not run. A parameter whose only supported value is 0 is
better absent than present-and-ignorable. Adding the diagnostic later means adding the term.

## 6. Known risks, carried forward

* **The loop can freeze.** If `Delta_0` is ever exactly zero at every position, the mask holds
  the model at the anchor for the entire unroll and PPL becomes independent of depth — the
  paper's 294.37 row. S4 tests that this is the CORRECT behaviour at zero; S8 plus a non-zero
  `Delta_0` at real positions is what keeps the live model off that fixed point. A depth sweep
  showing PPL responds to `T` is the field check.
* **`s = 0.50` across six blocks.** See D7.
* **Attention runs on deviations.** Inside the loop, slots attend to each other's deviations
  rather than their absolute states. The coda still reads `h* + Delta_T`, so the readout is
  unchanged, but the in-loop semantics are not. This is the method, not a bug, and it is the
  single largest behavioural change in the port.
* **Instruments.** `core_jacobian.py`, `jac_ladder.py` and `drift_probe.py` hook
  `_apply_core_step(h, e, ...)`. Against an SCSE model they must be told the anchor, or they
  will describe the wrong operator. The paper is explicit that the diagnostic is applied
  "separately within each trained model family", each with its own anchor.
* **Checkpoints.** `anchor_proj` and `init_proj` are new parameters. Fresh runs only.
* **Evidence limits.** SCSE is evaluated at 22M-139M on WikiText-2/103, OpenWebText and C4,
  with uniform depth sampling and without truncated BPTT, stochastic depth, ternary QAT or
  structured sparsity. MORPH has all of those. A null on MORPH would not refute the paper.
