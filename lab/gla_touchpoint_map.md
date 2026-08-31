# GLA / retention-branch touchpoint map

Repo: `/home/wolfe/morph-perf`, branch `perf/throughput-lever-stack`, HEAD `3bb46b6`.
Written 2026-08-31 as (a) the verification basis for the `model.retention: false` arm and
(b) the interpretation guide for the GLA-ablation results.

Context: the acausal cross-iteration GLA carry was proven to fake the l2cap depth-earning
([`lab/experiments/successes/2026-08-31-carry-leak-audit.md`](experiments/successes/2026-08-31-carry-leak-audit.md),
[`.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md`](../.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md)).
The next arm sets `model.retention: false`, whose config comment claims
"Off → GLA not constructed → bit-identical to baseline". **That claim is verified below (§8).**

---

## Summary table

| # | Area | Where | Verdict / note |
|---|---|---|---|
| 1 | Module | `morph/model/gla.py:59` `GatedLinearAttention` | 6 bias-free `nn.Linear` + `gate_bias` + `GroupNorm`. No dropout, no RNG at forward. |
| 1 | Construction | `morph/model/transformer.py:900-915` | Attached AFTER every base module → RNG tail; `retention=False` skips the whole `if`. |
| 1 | Attach | `morph/model/mhc.py:197-206` `attach_retention` | Sets `retention`, `norm_ret` (RMSNorm), `ret_gate` (scalar Parameter). |
| 1 | Which layers | `retention_layers=(1,)` applied to **prelude, core AND coda** — `transformer.py:906-908` | 3 GLA modules at base.yaml (4:6:4). |
| 1 | State-dict keys | 11 keys per site × 3 sites = 33 | `{sec}.1.retention.{q,k,v,g,r,o}_proj.weight`, `.gate_bias`, `.gn.{weight,bias}`, `{sec}.1.norm_ret.weight`, `{sec}.1.ret_gate` |
| 1 | Param cost | 3 × (6·d² + 4d + 1) | d=1024 → **18,886,659 params** (≈18.9 M). Measured exactly on a d=64 fixture: 74,499 = 3×24,833. |
| 2 | Combination | `morph/model/mhc.py:244-252` | `a = attn(norm_attn(x)) + sigmoid(ret_gate) · gla(norm_ret(x), state)` — **inside** the HC sublayer fn. |
| 2 | Plain path | `_front_tail` `transformer.py:1390`, `_apply_core_step` `:1163-1167`, `_back_region` `:1433` | prelude/coda get `ret_state=None`; only core carries. |
| 2 | TUL path | `_tul_front` `:1979` → `_front_tail`; `_tul_core` `:2213/2217/2220` → `_apply_core_step`; coda `:2903` | Same three seams. |
| 2 | Modes | `gla.py:268-297` | `recurrent` (oracle) / `chunked` (eager training) / `kernel` (Triton). `mode="kernel" if cfg.use_kernels else "chunked"` (`transformer.py:911`). |
| 2 | Kernel fallback | `gla.py:285-294` | SMEM guard `7·DH² ≤ optin`. **base.yaml d=1024, retention_heads=8 → DH=128 → 114688 > 101376 → the 5090 silently runs eager `_chunked` even with `use_kernels: true`.** |
| 2 | Reset mask | `gla.py:260,272-275,160-164,187-211` | `ret_reset_mask` only from `tg_restrict`; `kernel` mode **raises**. |
| 3 | Carry | `track_ret` at `transformer.py:1680-1687` (plain) and `:2132-2139` (TUL) | `_core_has_retention AND retention_carry_mode == "acausal_final"`. Default `"none"` → `ret_state` is `None` everywhere. |
| 3 | Normalizer | `transformer.py:334-347` `retention_carry_mode` property | Bools mapped; anything else raises. Every read site uses the property. |
| 3 | Decode seed | `morph/inference/kv_cache.py:481-490` | Cross-iteration seeding gated on `retention_carry_mode == "acausal_final"`. |
| 4 | HC | inside `mrr_attn`'s `sublayer_fn` (`mhc.py:257`, `hyper_connections.py:182-202`) | GLA reads the HC-mixed single-stream `x_bar`; its output is mixed by the HC residual. |
| 4 | ChannelInject | before the block (`transformer.py:1389,1160,1432`) | GLA sees the post-injection carrier. Never a separate inject. |
| 4 | Checkpointing | `transformer.py:1746`, `:2216` | `_core_step` **returns** `(h, ret_state)`; the `ret_capture` dict is never the transport (`mhc.py:234-238`). |
| 4 | torch.compile | `morph/training/train.py:1689-1691` | `layer.retention.forward = torch.compiler.disable(...)` under `compile_blocks` — Inductor SplitScan bug on the chunked cumsum. |
| 4 | Spectral penalty | `morph/training/spectral_penalty.py:66-89` | Enumerates `MortarLinear` under `core`, plus (opt-in) `type(sub) is nn.Linear` under `blk.attention`. GLA hangs off `blk.retention`, **not** `blk.attention` → **never touched.** |
| 4 | Ternary QAT | `morph/model/ternary_qat.py:529-537,577-581` | `"Attention" in "GatedLinearAttention"` → all 6 GLA linears categorize as **`attention`**, not `backbone`. Default `ternary_scope: backbone` → GLA stays bf16. |
| 4 | Weight decay | `morph/training/optimizer.py:25-51` | **`ret_gate` and `retention.gn.weight` land in the DECAY group.** See flag F1. |
| 4 | Pruning / MORTAR | `morph/training/pruning.py:36-48` | Enumerates `CMSBlockLinear` / `MortarLinear` only. GLA uses plain `nn.Linear` → **not prunable, not carved, not routed.** |
| 5 | Gate telemetry | `train.py:2947-2955` | `retention/gate_{prelude,core,coda}{i}` = `sigmoid(ret_gate)`, every step, gated on `cfg.retention`. |
| 5 | State-norm telemetry | `transformer.py:2296,2320` + `train.py:783-795` | `loop/ret_state_norm_{max,last,t*}` — **`_tul_core` only**; the plain `_forward_single` loop has no twin. |
| 5 | Jacobian probe | `transformer.py:1729,2198`; `morph/training/core_jacobian.py:210,217,246-253` | Captures + replays `ret_state` at the real operating point. |
| 6 | Config keys | 7 keys, `morph/configs/base.yaml:149-157` | Read once in `build_morph_config` `train.py:417-425`. |
| 7 | Tests | `tests/test_causality_contract.py`, `tests/test_tg_restrict.py`, `tests/test_tul_gl1*.py`, `tests/test_tul_forward.py:200`, `tests/test_tul_loop_ladder.py` | No standalone `test_gla.py`; no chunked-vs-recurrent parity test outside the reset-mask cases. |
| 8 | `retention: false` | `transformer.py:901-903` | **VERIFIED: 0 extra state-dict keys, 0 modules, 0 forward ops, and the 253 shared keys are bitwise identical to a `retention=True` build at the same seed.** |
| 9 | Causality | `gla.py:129-137,186,236-240` | The branch is causal within an iteration: per-token GroupNorm, left-to-right `cumsum`, `tril` mask, carried state only from earlier chunks. |

**Flags** (detail in §10): F1 weight decay opens `ret_gate`; F2 the SM120 kernel never runs at
base.yaml dims; F3 six shipped configs still carry `retention_carry: true` (the leak);
F4 the named verification gates `ignore/verify_gla.py` / `verify_fused_gla.py` /
`verify_retention.py` **do not exist**; F5 `loop/ret_state_norm` telemetry is TUL-only;
F6 GLA is a QAT `attention` module, which is counter-intuitive for anyone reading
`ternary_scope`.

---

## 1. Construction

### 1.1 The module

`morph/model/gla.py:59` `GatedLinearAttention(d_model, n_heads, mode, chunk, gate_logit_bias)`.

* `gla.py:83-90` — `q_proj`, `k_proj`, `v_proj`, `g_proj`, `r_proj`, `o_proj` (all
  `nn.Linear(d, d, bias=False)`), `gate_bias` (`nn.Parameter`, filled with
  `gate_logit_bias`), `gn = nn.GroupNorm(n_heads, d_model)`.
* `gla.py:95-96` — every projection `normal_(std=0.02)`. Deliberately **not** zero-init on
  `o_proj`: identity-at-init comes from the outer branch gate instead, so gradient still
  reaches q/k/v/g on step 0.
* `gla.py:78` — `dh = d_model // n_heads` (`dk == dv == dh`); `gla.py:75` asserts divisibility.
* No dropout inside GLA and no RNG draw at forward → adding the branch does not perturb the
  forward RNG stream.

### 1.2 The attach point

`morph/model/mhc.py:193-206`:

```
self.retention: nn.Module | None = None      # mhc.py:193
self.norm_ret:  nn.Module | None = None      # mhc.py:194
self.ret_gate:  nn.Parameter | None = None   # mhc.py:195

def attach_retention(self, gla, norm, gate_init):   # mhc.py:197
    self.retention = gla
    self.norm_ret  = norm
    self.ret_gate  = nn.Parameter(torch.tensor(float(gate_init)))
```

Post-construction attach is deliberate (`mhc.py:190-192`): it keeps the base init RNG draw
identical between a retention-ON and a retention-OFF build.

### 1.3 The construction site and its conditions

`morph/model/transformer.py:895-915`:

```
self._retention_layers = tuple(cfg.retention_layers)                 # :900
self._core_has_retention = cfg.retention and any(                    # :901
    i in self._retention_layers for i in range(cfg.n_core))          # :902
if cfg.retention:                                                    # :903
    from .gla import GatedLinearAttention                            # :904
    rheads = cfg.retention_heads or cfg.n_heads                      # :905
    for section in (self.prelude, self.core, self.coda):             # :906
        for si, blk in enumerate(section):                           # :907
            if si in self._retention_layers:                         # :908
                blk.attach_retention(                                # :909
                    GatedLinearAttention(
                        d, rheads,
                        mode="kernel" if cfg.use_kernels else "chunked",   # :911
                        chunk=cfg.retention_chunk,                         # :913
                        gate_logit_bias=cfg.retention_gate_bias),          # :914
                    RMSNorm(d), gate_init=cfg.retention_gate_init)         # :915
```

Notes that matter for reading the ablation:

* `retention_layers` is a **within-section** index applied to **all three** sections. At
  base.yaml (`n_prelude=4, n_core=6, n_coda=4`, `retention_layers: [1]`) that is
  `prelude[1]`, `core[1]`, `coda[1]` — three GLA modules, only one of which is inside the loop.
* The import is local to the branch: `retention=False` does not even import `gla`.
* `_core_has_retention` is the only place the CORE membership is decided. With `n_core == 0`
  (the FM1 / A2 arms) it is `False` even when `retention=True`, and the prelude/coda branches
  still exist and still run.
* `mode` is chosen at CONSTRUCTION from `use_kernels`. There is no runtime mode switch.

### 1.4 Parameters as they appear in `state_dict`

Per attached site (`{sec}` ∈ `prelude|core|coda`, index 1 at base.yaml):

```
{sec}.1.retention.q_proj.weight   [d, d]
{sec}.1.retention.k_proj.weight   [d, d]
{sec}.1.retention.v_proj.weight   [d, d]
{sec}.1.retention.g_proj.weight   [d, d]
{sec}.1.retention.r_proj.weight   [d, d]
{sec}.1.retention.o_proj.weight   [d, d]
{sec}.1.retention.gate_bias       [d]
{sec}.1.retention.gn.weight       [d]
{sec}.1.retention.gn.bias         [d]
{sec}.1.norm_ret.weight           [d]
{sec}.1.ret_gate                  []          scalar
```

Count per site `6d² + 4d + 1`. Measured on a d=64 fixture: 33 extra keys, 74,499 params
(3 × 24,833). At base.yaml d=1024: **18,886,659 params**.

### 1.5 The config surface (§6 answered here)

`morph/model/transformer.py:275-294` (dataclass) ← `morph/training/train.py:417-425`
(`build_morph_config`) ← `morph/configs/base.yaml:149-157`.

| Key | Default (dataclass:line) | base.yaml | Read at |
|---|---|---|---|
| `retention` | `True` (`:279`) | `true` (`:149`) | `transformer.py:901,903`; `train.py:417,2951` |
| `retention_layers` | `(1,)` (`:280`) | `[1]` (`:150`) | `transformer.py:900,908,1163`; `kv_cache.py:441`; `core_jacobian.py:246` |
| `retention_heads` | `0` → `n_heads` (`:281`) | `8` (`:151`) | `transformer.py:905,1683,2135`; `engine.py:195` |
| `retention_chunk` | `128` (`:282`) | `256` (`:152`) | `transformer.py:913` only |
| `retention_gate_init` | `-6.0` (`:283`) | `-6.0` (`:153`) | `transformer.py:915` only |
| `retention_carry` | `"none"` (`:293`) | `"none"` (`:154`) | via `retention_carry_mode` (`:335`) at `transformer.py:1026,1681,2133`; `kv_cache.py:484`; `future_leak_probe.py:81` |
| `retention_gate_bias` | `2.0` (`:294`) | `2.0` (`:157`) | `transformer.py:914` only |

Warning print when the leak is opted into: `transformer.py:1026-1028`.

---

## 2. Forward integration

### 2.1 The one combination point

`morph/model/mhc.py:244-252` — the ONLY place the branch output enters the residual stream:

```
def _attn_fn(x: Tensor) -> Tensor:                                        # mhc.py:244
    a = self.attention(self.norm_attn(x), **attn_kwargs)                  # mhc.py:245
    if self.retention is not None:                                        # mhc.py:246
        g_out, s_out = self.retention(self.norm_ret(x),
                                      initial_state=ret_state,
                                      reset_mask=ret_reset_mask)          # mhc.py:247-248
        if ret_capture is not None:
            ret_capture["state"] = s_out                                  # mhc.py:250
        a = a + torch.sigmoid(self.ret_gate).to(a.dtype) * g_out          # mhc.py:251
    return self.drop(a)                                                   # mhc.py:252
```

`_attn_fn` is handed to `self.mrr_attn` (`mhc.py:257`), the `HyperConnectionResidual`.
So the branch is **inside** the HC mixing on both ends (see §4.1). The branch shares
`self.drop` with attention — dropout applies to the sum, not the branch.

`MORPHBlock.forward` signature carrying the three retention arguments: `mhc.py:208-217`
(`ret_state`, `ret_capture`, `ret_reset_mask`), documented `mhc.py:230-238`.

### 2.2 Every call site that can reach a retention block

| Path | Site | `ret_state` | `ret_capture` | `ret_reset_mask` |
|---|---|---|---|---|
| Prelude (plain + TUL) | `transformer.py:1390` (`_front_tail`) | None | None | `tg_reset` or None |
| Core (plain + TUL) | `transformer.py:1167` (`_apply_core_step`) | `ret_state` if `i in _retention_layers` | `ret_cap` if same | **never** (spec: core GLA is not reset) |
| Coda (plain + TUL, full-L) | `transformer.py:1433` (`_back_region`) | None | None | `tg_reset` or None |
| `prelude_states()` readout | `transformer.py:1516` | None | None | `tg_reset` |
| DB1 single-step | `transformer.py:2430` | explicit `None` | — | — |
| DB1 Euler ladder | `transformer.py:2487` | explicit `None` | — | — |
| Jacobian probe replay | `core_jacobian.py:217,253` | replayed capture | `cap` | — |
| Decode (eager KV) | `kv_cache.py:397-400` | `sc.ret_state` per site | dict | not supported |
| Decode (fused engine) | `engine.py:912-921,932-933` | `s.ret_state` in place | — | not supported |

The core gating inside `_apply_core_step`:

```
ret_cap = {} if self._core_has_retention else None            # transformer.py:1149
...
is_ret = ret_cap is not None and (i in self._retention_layers) # transformer.py:1163
rs_arg = ret_state if is_ret else None                         # transformer.py:1164
rc_arg = ret_cap   if is_ret else None                         # transformer.py:1165
h_injected = layer(h_injected, mlp_kwargs=mlp_kw,
                   ret_state=rs_arg, ret_capture=rc_arg)        # transformer.py:1166-1167
new_ret = ret_cap.get("state") if ret_cap is not None else None # transformer.py:1168
```

Note `_apply_core_step` never forwards a reset mask — `docs/tul-tg-spec.md:91-92` says the
core loop's GLA gets no reset by design.

### 2.3 The three modes

`gla.py:259-299` `forward(x, initial_state=None, return_state=True, reset_mask=None)`:

* `gla.py:268` `mode == "recurrent"` → `_recurrent` (`gla.py:149-170`), explicit O(T) scan,
  the parity oracle.
* `gla.py:271` `mode == "kernel"` → fused Triton `fused_gla` (`morph/kernels/triton/fused_gla.py`).
  `gla.py:272-275` raises `NotImplementedError` if a `reset_mask` is supplied.
  `gla.py:285-294` SMEM guard: if `7·DH² > shared_memory_per_block_optin`, fall through to
  `_chunked`. **At base.yaml (d=1024, `retention_heads: 8` → DH=128) `7·128² = 114688 >
  101376` on sm_120, so the fused kernel NEVER runs on the 5090 at production dims — the
  eager `_chunked` path is what every 5090 run has executed.** (Flag F2.)
* `gla.py:295` else → `_chunked` (`gla.py:173-257`), the eager training path.

Shared: `_project` (`gla.py:99-121`) and `_readout` (`gla.py:123-140`).
`_project` batches q/k/v/g/r into ONE GEMM when `MORPH_FUSED_GLA_PROJ` is not `0`
(`gla.py:48`, override `gla.py:51`).

### 2.4 The tg reset path

`docs/tul-tg-spec.md` §4. Built once per forward:

* `transformer.py:2782-2792` (`_forward_tul`) and `transformer.py:1509-1518`
  (`prelude_states`): `tg_reset = tg_reset_mask(layout)` when `self._tg_restrict`.
* Threaded to prelude via `_tul_front` → `_front_tail` (`transformer.py:1979` → `:1390`).
* Threaded to the full-L coda at `transformer.py:2903`.
* Implemented structurally in `_recurrent` (`gla.py:160-164`, exact multiply-by-zero) and in
  `_chunked` (`gla.py:187-211,224-231,239-240,247-252`, per-segment cumulative gate +
  cross-reset pair masking). `gla.py:262-267` explains why a `log_alpha` floor was rejected.
* `tg_restrict` therefore **requires `use_kernels=false`** (`gla.py:273-275`).

---

## 3. The carry wiring

### 3.1 The normalizer

`transformer.py:334-347`:

```
@property
def retention_carry_mode(self) -> str:      # transformer.py:335
    v = self.retention_carry
    if isinstance(v, bool):
        v = "acausal_final" if v else "none"
    if v not in ("none", "acausal_final"):
        raise ValueError(...)
    return v
```

The docstring is explicit that a raw truthiness check reads the string `"none"` as `True`.
Every read site in the tree uses the property — verified by grep: `transformer.py:1026,1681,2133`,
`kv_cache.py:484`, `lab/divergence/future_leak_probe.py:81`. No raw `cfg.retention_carry`
truthiness test exists outside the property itself.

### 3.2 `track_ret` — the two identical gates

Plain path, `transformer.py:1680-1687`:

```
track_ret = (self._core_has_retention
             and self.cfg.retention_carry_mode == "acausal_final")   # :1680-1681
if track_ret:
    _rh  = self.cfg.retention_heads or self.cfg.n_heads              # :1683
    _rdh = self.cfg.d_model // _rh                                   # :1684
    ret_state_s = h_s.new_zeros(h_s.shape[0], _rh, _rdh, _rdh,
                                dtype=torch.float32)                 # :1685
else:
    ret_state_s = None                                               # :1687
```

TUL path, `transformer.py:2132-2139` — the same five lines with `ret_state` in place of
`ret_state_s`.

With the shipped default `retention_carry: "none"`, `track_ret` is `False` on both paths, so
`ret_state` is `None` at every `_core_step` call and each core iteration reseeds the GLA
state to zero. The GLA still runs — it is a within-iteration global-context branch over the
sequence (or over the slot sequence, on the TUL path).

### 3.3 Carry propagation (only when `acausal_final`)

* Plain: `rs_a = ret_state_s[:n_active] if track_ret else None` (`transformer.py:1723`);
  update `transformer.py:1791-1793`, which keeps the sorted active-prefix / frozen-suffix
  order exactly like the carrier.
* TUL: `ret_state=ret_state` at all three call sites (`transformer.py:2213,2217,2220`);
  update `transformer.py:2302-2303` (`.detach()` under `db_loop`).
* The `no_grad` iterations (`t < n_nograd`) return a detached state, which is the
  truncated-BPTT boundary for the carry — noted at `transformer.py:1676-1679`.

### 3.4 Checkpoint safety

`ret_capture` is a **side channel** and is explicitly not the transport. `mhc.py:234-238`
states the rule; `_core_step` returns `(h, new_ret)` (`transformer.py:1618-1633` /
`:2141-2156`), and `checkpoint(...)` at `transformer.py:1746` and `:2216` propagates the
state through the return value.

### 3.5 Inference decode seeding

`morph/inference/kv_cache.py`:

* `AttnSiteCache.ret_state` field, `kv_cache.py:65-69`.
* `_block_step` mirrors `mhc._attn_fn` byte-for-byte, `kv_cache.py:397-400`.
* Prelude / coda: unconditional per-site running state (`kv_cache.py:461-463`, `:500-502`).
  This is the exact streaming form of the training forward's single full-sequence GLA call.
* Core: `ret_layers = set(model._retention_layers) if model._core_has_retention else set()`
  (`kv_cache.py:441`), and the cross-iteration seed is **gated**:

```
if (model.cfg.retention_carry_mode == "acausal_final"
        and sc.ret_state is None and t > 0):                      # kv_cache.py:484-485
    sc.ret_state = cache.site(f"core.{i}.{t - 1}").ret_state      # kv_cache.py:486
```

  Docstring `kv_cache.py:423-430` states this is a causal approximation of an acausal
  training path — there is no bit-exact O(1) form of the leak.
* Fused engine: `engine.py:195` (`self.rh`), `:262-264` (zero-init state per site),
  `:299-305` (stacked `Wret`, `sigmoid(ret_gate)` folded into `o_proj`), `:912-921`
  (`_retention` → `gla_step` kernel, state updated in place), `:932-933` (sum into `a`),
  `:693-695` and `:758-760` (state copy-in for graph capture), `:1189-1190`.
  The fused decode kernel: `morph/kernels/triton/fused_decode_step.py:273-285` `gla_step`.

---

## 4. Interactions

### 4.1 Hyper-Connection Cayley residual — INSIDE

`mhc.py:257` `h = self.mrr_attn(h, _attn_fn)`, and `hyper_connections.py:182-202`
documents `sublayer_fn` as taking a single `[B, S, C]` tensor. The HC pre-map produces
`x_bar` (the stream-mixed single-stream view, `hyper_connections.py:209-212`), which is what
`_attn_fn` — and therefore `norm_ret(x)` — receives. The branch output is added to the
attention output BEFORE the HC post write. So the retention branch is fully inside the HC
mixing on both the input and output side. There is no separate retention stream.

### 4.2 ChannelInject / DiagonalInjection

Injections are applied to the carrier before the block runs (`transformer.py:1389`,
`:1160`, `:1432`), so the GLA input already contains the x0 / value-embed / bigram /
diagonal-injection content. No `ChannelInject` targets the retention branch, and
`DiagonalInjection` (`transformer.py:349+`) does not touch it.

### 4.3 Gradient checkpointing

`checkpoint(_core_step, ..., use_reentrant=False)` at `transformer.py:1746` (plain) and
`:2216` (TUL). The GLA forward runs inside the checkpointed region and is recomputed in
backward. Its `final_state` crosses the boundary as a return value, never as a captured
Python dict.

### 4.4 torch.compile

`morph/training/train.py:1685-1691`:

```
# The GLA retention branch stays a graph break (the same eager code
# runs): Inductor hits an upstream SplitScan codegen bug on its
# chunked cumsum.
if getattr(layer, "retention", None) is not None:          # train.py:1689
    layer.retention.forward = torch.compiler.disable(
        layer.retention.forward)                            # train.py:1690-1691
group[i] = torch.compile(layer, mode=compile_mode, dynamic=dyn)
```

This only fires under `training.compile_blocks`. Consequence for the ablation: the
`retention: false` arm has **one fewer graph break per retention block per forward**, so its
wall-clock advantage is larger than the FLOP saving alone. Anyone attributing a step-time
delta to "GLA FLOPs" must account for that.

`tests/test_kernel_compile_fences.py:17-18,29,35` builds its fixture with `retention=False`
specifically to dodge the GLA kernel's SM120 SMEM OOM.

### 4.5 Spectral penalty / projection — does NOT touch GLA

`morph/training/spectral_penalty.py:66-89` `collect_core_linears`:
* MLP side: `isinstance(sub, MortarLinear)` under `root.core` blocks.
* Attention side (`include_attn` only): `type(sub) is nn.Linear` under `getattr(blk, "attention", None)`.

`blk.retention` is a sibling of `blk.attention`, not a child. GLA's linears are therefore
never enumerated, never power-iterated, and never penalized or projected. A spectral-control
arm and a retention arm are orthogonal.

### 4.6 Ternary QAT — GLA is classified as `attention`

`morph/model/ternary_qat.py:529-537`:

```
def _attention_linear_ids(model):
    for m in model.modules():
        if "Attention" in type(m).__name__:      # ternary_qat.py:532
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    ids.add(id(sub))
```

`"Attention" in "GatedLinearAttention"` is `True`. Verified by running `_categorize` over a
real build: all 18 GLA linears (6 × 3 sites) return `"attention"`. With the shipped
`training.ternary_scope: backbone` (`base.yaml:380`) they stay bf16. Under
`ternary_scope: attention` or `full` they would be ternarized. (Flag F6 — this is
correct behaviour but not what the name suggests.)

`gate_bias`, `gn.*`, `norm_ret.weight` and `ret_gate` are not `nn.Linear`/`nn.Embedding` →
`_categorize` returns `None` → never quantized. `embed_quant.py`, `attn_proj_quant.py`,
`fp8_scope.py` contain no retention references.

### 4.7 Optimizer parameter groups — MEASURED

`morph/training/optimizer.py:25-51` `_NO_DECAY_KEYWORDS`, applied by substring at
`optimizer.py:67`. Measured split on a real build:

| Group | GLA params |
|---|---|
| **decay** | `retention.{q,k,v,g,r,o}_proj.weight`, `retention.gn.weight`, **`ret_gate`** |
| no-decay | `retention.gate_bias` (matches `"bias"`), `retention.gn.bias`, `norm_ret.weight` (matches `"norm"`) |

See flag F1.

### 4.8 Pruning / MORTAR / ReMoE — GLA is not eligible

`morph/training/pruning.py:36-48` enumerates `CMSBlockLinear` and `MortarLinear` only.
GLA's projections are plain `nn.Linear`. Confirmed by the state-dict dump: every GLA weight
is a bare `.weight`, with no `_cms` nesting. So GLA weights are never scored, never pruned,
never carved to BCSR, and never routed by `TileRouter`. A "0.25 density" claim never
applied to the 18.9 M GLA parameters — they were dense for the whole run.

Corollary for the ablation: turning retention off removes 18.9 M **dense** parameters, which
is a larger fraction of the post-carve model than of the pre-carve model.

### 4.9 FLOP accounting

`morph/training/flops.py:10` says FlopCounterMode is blind to the custom GLA kernel. The
analytic model walks `nn.Linear` / `MortarLinear` shapes (`flops.py:17-20`), so the six GLA
GEMMs ARE counted, but the recurrence itself (the `[dk,dv]` state update and readout, which
scales with S) is in the "not counted" bucket (`flops.py:26-30`). `perf/flop_proxy` therefore
understates the retention arm relative to the no-retention arm.

---

## 5. Telemetry and probes

| Signal | Site | Scope |
|---|---|---|
| `retention/gate_{sec}{i}` | `train.py:2947-2955` | every step, all three sections, gated on `cfg.retention` |
| `loop/ret_state_norm_{max,last,t*}` | collected `transformer.py:2296`, packed `:2318-2325`, logged `train.py:783-795`, armed `train.py:1657` | **`_tul_core` only** (flag F5) |
| Jacobian operating point (`ret_state`) | captured `transformer.py:1729` (plain) and `:2196-2200` (TUL); replayed `core_jacobian.py:210,217` | probe-only, detached |
| Per-block σ with retention wired | `core_jacobian.py:246-253` | `is_ret = root._core_has_retention and (i in root._retention_layers)` |
| Future-corruption leak probe | `lab/divergence/future_leak_probe.py:81` reads `retention_carry_mode` | offline |
| Carry CE cost | `ignore/perf/carry_leak_cost.py` | offline, same weights, carry on vs off |
| Causality bisect | `ignore/perf/causality_bisect.py` | offline, hooks every submodule |
| Footprint estimate | `ignore/retention_footprint.py` | CPU-only, pre-implementation sizing |
| KV-cache parity | `ignore/verify_kv_cache.py` | offline |
| Onset plot keys | `ignore/perf/phase1/plot_onset.py:102-103` | reads `loop/ret_state_norm_*` |

There is **no** branch-norm telemetry — nothing logs `‖g_out‖` versus `‖attn_out‖`. The gate
value is the only in-training signal of how much the branch contributes, and §10 F1 explains
why that signal is confounded.

---

## 6. Config surface

Covered in §1.5. Additional facts:

* `retention_chunk` and `retention_gate_init` and `retention_gate_bias` each have **exactly
  one** read site, all inside the construction block. They cannot affect a
  `retention: false` build.
* Configs shipping `retention: false` today: `morph/configs/scale30b.yaml:77` only.
* All other shipped configs set `retention: true`.

---

## 7. Tests

| Test | What it covers | Would it catch a broken `retention: false`? |
|---|---|---|
| `tests/test_causality_contract.py:95-97` `test_causality_holds_when_there_is_no_retention_branch_at_all` | builds `retention=False`, asserts future-corruption delta **exactly 0.0** | Yes, for causality. Not for "constructs nothing". |
| `tests/test_causality_contract.py:79-85` | default config is causal | Yes (the shipped default). |
| `tests/test_causality_contract.py:100-110` | `acausal_final` still reproduces the leak (a live falsifier, not a stale assertion) | n/a |
| `tests/test_causality_contract.py:112-114` | invalid carry mode raises | n/a |
| `tests/test_causality_contract.py:194-210` | `_tul_core` path causal on default, acausal on opt-in | n/a |
| `tests/test_tg_restrict.py:161-251` | 5 GLA reset-mask tests: recurrent vs per-segment oracle, chunked vs oracle, chunk-misaligned resets, all-False mask == no-mask path, `kernel` mode raises | No |
| `tests/test_tul_gl1.py:193-213` | T3 gradient severing **with retention ON** — the version that actually protects GL1 | No |
| `tests/test_tul_gl1b.py:65`, `tests/test_tul_gl1c.py:59` | GL1 variants with `retention=True, retention_carry=True` | No |
| `tests/test_tul_forward.py:200-215` | core runs with the GLA carry, loss finite | No |
| `tests/test_tul_loop_ladder.py:27-30` | `retention_layers=(2,)` puts GLA inside the core so the `db_loop` detach is exercised | No |
| `tests/test_kernel_compile_fences.py:29-35` | uses `retention=False` to build a clean graph | Weakly |
| ~15 other tests | build with `retention=False` purely as a fixture simplification | No |

**Gaps.**

1. There is **no** test asserting that a `retention=False` build adds zero state-dict keys, and
   none asserting that a `retention=True` and a `retention=False` build share bitwise-identical
   base weights at the same seed. Both properties are load-bearing for the ablation and both are
   currently protected only by code reading. (§8 verifies them by measurement, but a measurement
   in a lab note is not a gate.)
2. There is **no** standalone `tests/test_gla.py`. The `chunked` vs `recurrent` parity claim in
   `gla.py:15-20` ("Must match") is gated only inside the reset-mask tests
   (`test_gla_chunked_reset_matches_recurrent_oracle_per_segment`,
   `tests/test_tg_restrict.py:176`), and the `kernel` vs `chunked` parity claim
   (`gla.py:277` "grads cos 1.0 vs the recurrent oracle") is gated by
   `ignore/verify_fused_gla.py`, **which does not exist** (flag F4).
3. `tests/test_ckpt_retention.py` is about the checkpoint retention RING
   (`morph/training/ckpt_retention.py`), an unrelated module that shares the word. It has
   nothing to do with GLA. Do not read it as retention-branch coverage.

---

## 8. The `retention: false` path — VERDICT

**The claim holds. `retention: false` constructs nothing, computes nothing, and stores nothing.**

### 8.1 The guard sites, quoted

Construction — the entire attach loop lives under one `if`:

```
if cfg.retention:                                     # morph/model/transformer.py:903
    from .gla import GatedLinearAttention             # :904   (import is inside the branch)
```

Default attribute state when the branch does not run — `morph/model/mhc.py:193-195`:

```
self.retention: nn.Module | None = None
self.norm_ret:  nn.Module | None = None
self.ret_gate:  nn.Parameter | None = None
```

`None` assigned to an `nn.Module` attribute registers no submodule and no parameter, so no
state-dict key appears.

Forward — every consumer is `None`-guarded:

* `mhc.py:246` `if self.retention is not None:` (the only training forward)
* `transformer.py:901-902` `_core_has_retention = cfg.retention and any(...)` → `False`
* `transformer.py:1149` `ret_cap = {} if self._core_has_retention else None` → `None`
* `transformer.py:1163` `is_ret = ret_cap is not None and (...)` → `False`
* `transformer.py:1680-1687` / `:2132-2139` `track_ret` → `False` → no state tensor allocated
* `kv_cache.py:397` `if block.retention is not None:` ; `kv_cache.py:441` `ret_layers = ... else set()`
* `engine.py:264,299,932` `if s.block.retention is not None` / `if block.retention is not None`
* `train.py:1689` `if getattr(layer, "retention", None) is not None:` (compile fence)
* `train.py:2951` `if getattr(_rm.cfg, "retention", False):` (gate telemetry)
* `core_jacobian.py:246` `is_ret = root._core_has_retention and (...)`

No unguarded site was found anywhere in `morph/`.

### 8.2 Measured verification (CPU, this session)

Two builds at `torch.manual_seed(0)` with identical config except `retention`
(d=64, 2:2:2, `retention_layers=(1,)`, `retention_heads=2`, `dropout=0.0`, eval mode):

```
EXTRA KEYS when retention=True (33)   ← exactly the 11 keys × 3 sites of §1.4
MISSING when True: []
SHARED KEYS DIFFERING: []             ← all 253 shared keys bitwise equal
n shared 253
blocks with .retention not None (retention=False model): []
_core_has_retention  False-model: False   True-model: True
ret_gate params in False model: []
total params True/False: 595520 / 521021   (delta 74499 = 3 × 24833)
```

Independent forward check — set every `ret_gate` to `-40.0` (sigmoid ≈ 4e-18) on the
retention-ON model and compare logits against the retention-OFF model on the same input:

```
gate-closed T vs F max|d|: 0.0
bit-identical: True
```

So the ONLY difference between the two builds is the branch contribution itself. The RNG-tail
discipline (`transformer.py:895-899`) holds: no other weight moves.

### 8.3 What is NOT verified

* This was measured on a **d=64 CPU fixture in eval mode**, not on the 286 M GPU model. The
  RNG argument is structural (the attach loop is the last RNG consumer before `TULSlots`,
  whose inits are deterministic — `transformer.py:918-926`), but the bit-identity check itself
  was not run at production dims.
* Training-mode equality was not checked. It should also hold — GLA has no dropout and draws
  no RNG at forward — but I did not measure it.
* `retention_carry: "acausal_final"` combined with `retention: false` was not exercised;
  by code reading `track_ret` is `False` (it ANDs with `_core_has_retention`) so the carry is
  inert, and no warning fires except `transformer.py:1026-1028`, which prints regardless.
  That print will be misleading in that combination.

---

## 9. Fresh-eyes causality sweep of the branch

### 9.1 Within an iteration, the GLA branch is causal

* **Readout normalization is per token.** `gla.py:137`
  `o = self.gn(o.reshape(B * S, H * dh)).reshape(B, S, H * dh)`. The comment at
  `gla.py:129-136` records the earlier `gn(o.transpose(1, 2))` form, which pooled statistics
  across the whole sequence axis and was a real, exploited leak (fixed 2026-07-03). The current
  form folds S into the batch, so no statistic crosses positions.
* **State flows left to right only.** `_recurrent` (`gla.py:155-166`) is a literal forward scan.
* **The chunked form is an exact refactor of that scan, not an approximation of it.**
  `gla.py:186` `b = la.cumsum(dim=1)` is a prefix sum (position t sees only j ≤ t).
  `gla.py:236-238` builds `torch.tril(...)` and zeroes every non-causal pair.
  `gla.py:231` `o_inter` reads only `state`, which at chunk `c` holds the accumulation over
  chunks `< c`. `gla.py:245-255` writes the chunk-end state that only chunks `> c` will read.
  There is no reverse scan and no full-sequence reduction anywhere in `_chunked`.
* **The fused kernel is causal by the same construction.**
  `morph/kernels/triton/fused_gla.py:20` documents `P[t,j] = qb_t · kb_j (j<=t)`;
  `:115` and `:185` build `causal = rows[:, None] >= rows[None, :]`; `:143` and `:223`
  apply `P = tl.where(causal, P, 0.0)` in forward and backward.
* **The only clamp is a numerical floor**, `gla.py:219` `b = b.clamp(min=-30.0)`, with the
  overflow rationale at `gla.py:214-218`. It does not move information backwards.
* **The reset-mask path is more restrictive, not less** — it removes state flow across segment
  boundaries (`gla.py:160-164`, `:224-231`, `:239-240`, `:247-252`), gated by five tests at
  `tests/test_tg_restrict.py:161-251`.

Empirically: `tests/test_causality_contract.py:79-85` passes on the default
(`retention: true`, `retention_carry: "none"`) with an exact-zero future-corruption delta, on
both the plain and `_tul_core` paths. That test builds retention ON, so it is a live gate on
the within-iteration causality of the branch, not just on the carry.

### 9.2 What the carry did, for the record

`retention_carry: "acausal_final"` seeded iteration `t+1` position 0 with iteration `t`'s
END-OF-SEQUENCE state — a summary of every position, future included. From iteration 2 onward
every position saw the future. That is the ONLY sequence-global path the GLA branch ever had,
and it is now off by default (`transformer.py:293`, `base.yaml:154`).

### 9.3 Other sequence-global operations found in the sweep

I grepped `morph/model/` for `mean(dim=1)`, `sum(dim=1)`, `cumsum`, `transpose(1, 2)`,
`GroupNorm`, `BatchNorm` and read every hit. Nothing else pools across the sequence axis in
the forward:

| Hit | Verdict |
|---|---|
| `morph/model/attention.py:362` CSA block scoring `bmm(q_I, K_I.transpose(1,2))` | `:363` `masked_fill(~causal_mask, -inf)` **before** the relu; mask built at `:370-374` as `block_end < query_pos`. Causal. |
| `morph/model/attention.py:609-657,726-729` | `transpose(1,2)` are head/channel layout moves for the CCA depthwise convs, not reductions. |
| `morph/model/tul.py:570` `n_tok = (~slot_mask).sum(dim=1, keepdim=True)` | Counts mask entries, not data. The layout is an input, not a function of future tokens. |
| `morph/model/transformer.py:3430` `layout.prefix_k * layout.slot_valid.sum(dim=1)` | Same — a budget from the layout. |
| `morph/model/attn_lift.py:158` `mass = (w.mean(1) * slot_mask...).sum(-1)` | Entirely inside `with torch.no_grad()` (`attn_lift.py:145`) and recomputed only to be counted. Instrument, not forward. |
| `morph/model/fused_ce.py:196,219,260-264` | Per-row loss reductions over the label axis, after the logits. Not a feature path. |
| `morph/model/ternary_qat.py:171,214,419` | Per-output-row weight scale `mean(dim=1)` over the INPUT-feature axis of a weight matrix. No sequence axis involved. |
| `morph/model/fm_planner.py:674-678` | Head reshape/transpose. |
| TUL slot bag-mean (`_tul_front`, `transformer.py:1966-1979`) | Means over a span's tokens; the slot sits AFTER its span, so the pooled content is strictly in the past. Documented in `docs/tul-spec.md` §3.2. |

The HC stream reduction `x.mean(dim=2)` (`transformer.py:1449`) is over the STREAM axis, not
the sequence axis.

So: after the carry fix, the GLA branch has no remaining acausal path, and the sweep turned up
no second offender elsewhere in the model forward.

---

## 10. Flags

**F1 — weight decay pulls `ret_gate` toward an OPEN gate. This confounds the gate telemetry.**
`ret_gate` initializes at `-6.0` (`base.yaml:153`) and lands in the DECAY group
(`optimizer.py:25-51`; measured, §4.7). Weight decay pulls a scalar toward **0**, and
`sigmoid(0) = 0.5`. So decay is a constant force OPENING the retention branch, independent of
any gradient signal. `train.py:2948-2949` calls the gate "THE key signal for whether the model
actually USES the retention branch (gate opens from ~0) vs treats it as dead weight (stays ~0)"
— that reading is not safe as written, because part of any observed opening is decay, not
learning. Every other scalar branch gate in the model is excluded from decay by name
(`"log_scale"`, `"alpha_raw"`, `"gamma_raw"`, `"channel_scales"`, `"tul_gate"`); `ret_gate`
matches none of the keywords. `retention.gn.weight` (a GroupNorm scale) is in the same
position and for the same reason: `"gn.weight"` contains no `"norm"`.
This is a real defect, not a naming nit. I did NOT measure how large the effect is at the
shipped `weight_decay` — that needs a run.

**F2 — the fused GLA kernel has never executed at production dims on the 5090.**
`gla.py:290` requires `7·DH² ≤ shared_memory_per_block_optin`. base.yaml is d=1024 with
`retention_heads: 8` → DH = 128 → 114688 > 101376 (sm_120). The `else` at `gla.py:293-294`
silently runs the eager `_chunked` path. The fallback is numerically fine (that is the point of
the guard) but it means `use_kernels: true` does not buy the retention branch anything on this
GPU, and any perf attribution that assumed a fused GLA is wrong. The comment at `gla.py:278-284`
describes the same arithmetic for `retention_heads=6` at d=768 without noting that the shipped
d=1024 / 8-head config also fails it.

**F3 — ten shipped configs still carry the leak.** `retention_carry: true` (→ `acausal_final`)
in `morph/configs/react_bridge_d512.yaml:69`, `react_bridge_d512_v4.yaml:69`,
`react_bridge_d512_v4a_dsl.yaml:69`, `react_bridge_d512_v4b_cot.yaml:69`,
`prepretrain_math_d512.yaml:56`, `prepretrain_math_d512_sft.yaml:58`,
`transition_phase_a_d512.yaml:80`, `wordbridge_d512.yaml:58`,
`control_fresh_init_d512.yaml:52`, `forget_olympiad_seed768.yaml:79`.
(That is ten files, not six.) Only `base.yaml` and `scale30b.yaml` were updated to `"none"`.
Any new run launched from one of those configs reintroduces the leak silently — the only
signal is the `transformer.py:1027` warning print.

**F4 — the named verification gates do not exist.** `gla.py:277` cites
`ignore/verify_fused_gla.py`; `morph/configs/scale30b.yaml:76` cites
`ignore/verify_gla.py + verify_retention.py`.
None of the three is present in `/home/wolfe/morph-perf/ignore/` or in the main tree's
`ignore/`. The kernel-vs-eager parity claim and the retention-construction claim are
currently uncited.

**F5 — `loop/ret_state_norm` telemetry is TUL-only.** The probe collector lives in
`_tul_core` (`transformer.py:2296,2318-2325`). The plain `_forward_single` loop has the
`track_ret` machinery but no `_pr_ret` twin. A no-TUL causal arm therefore gets zero carry
telemetry — which is fine while `retention_carry: "none"` makes the carry inert, but the
instrument will still be missing if anyone re-enables it on the plain path.

**F6 — GLA is a `attention`-scope module for ternary QAT.** `"Attention" in
"GatedLinearAttention"` (`ternary_qat.py:532`) is an accidental match that happens to give the
right answer. It is not documented anywhere, and renaming the class would silently move 18
linears from bf16 into the ternary set under `ternary_scope: full`.

**Not a flag, but read it once:** `morph/training/ckpt_retention.py` and
`tests/test_ckpt_retention.py` are the checkpoint retention RING (how many `step_*.pt` files
to keep). They share no code and no concept with the GLA retention branch. Grepping
`retention` across the tree returns both.

---

## 11. Reading guide for the `retention: false` arm

What the arm removes, in order of likely effect size:

1. **18.9 M dense parameters** (§1.4) that are never pruned, never carved, never routed (§4.8).
   Against a 286 M model that is ~6.6 % of parameters pre-carve and a larger share post-carve.
2. **Three global-context branches**, one per section. Only the CORE one (`core[1]`) sits inside
   the loop; `prelude[1]` and `coda[1]` run exactly once per forward. If the arm loses quality,
   check the gate values from the control run
   (`retention/gate_prelude1`, `retention/gate_core1`, `retention/gate_coda1`) to see which of
   the three the model was actually using — and read F1 before treating an open gate as use.
3. **Three graph breaks per forward** under `compile_blocks` (§4.4). Step-time gains are
   partly compile, not FLOPs.
4. **Nothing acausal.** With `retention_carry: "none"` the control is already causal (§9), so a
   `retention: false` arm is not a causality fix — it is a capacity/architecture ablation.
   Do not attribute any CE change to leak removal.

What it does NOT change: base weights at a fixed seed (§8.2), spectral control (§4.5),
pruning schedule (§4.8), ternary scope under `backbone` (§4.6), TUL layout, or the HC residual.
