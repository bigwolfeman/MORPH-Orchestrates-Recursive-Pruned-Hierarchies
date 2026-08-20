# Agent Note: Ademamix Perf Reclaim

Status: implemented

Origin: Ai-notes/06-21-2026/AdEMAMix-Perf-Reclaim/RESULTS.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# AdEMAMix β1=0 Cure — Perf Reclaim (Task #277, 2026-06-21)

The coord-cap cure shipped on the DE-FUSED path (the fused Triton kernel guards-raise on the
new caps). Measured the regression on the real 276M deploy model (454 param tensors), opt.step
wall (fake constant grads, launch-bound). Bench: `ignore/bench_cure_step.py`, `ignore/bench_optimizer_step.py`.

## Baseline (BEFORE), opt.step ms
- AdamW8bit: 13.7 | bnb AdEMAMix8bit: 15.8
- fork BARE de-fused: 27.5 | fork BARE FUSED: **3.3** (faster than AdamW!) | fork 32bit no-quant: 21.2
- **CURE FULL (deploy) de-fused: 99.4 ms** ← the regression (matches old "~91ms" note)

Knob attribution of the +71.8ms cure overhead (diag ON):
SNR gate +31.9 | per-coord cap +20.9 | update_clip +16.1. Root cause = PER-TENSOR PYTHON
DIAGNOSTIC LOOPS over 454 tensors: snr-gate `float(gt.mean())`+`int((gt<0.5).sum())` (~900 GPU→CPU
syncs/step), update_clip `int((|u|>c).sum())` (454 syncs), per-coord-cap `(o>0).sum()` (454 launches,
sync-free but launch-bound). The cure MATH (foreach) is cheap; the telemetry was the cost.

## TIER 1 ✅ DONE — gate diagnostics + batch clamp (bit-identical, gated DIAG_PARITY_GATE_PASS)
Changes (`ademamix_b1zero.py` + `optimizer.py` + `base.yaml`):
- New `track_diag` flag (default **False**). Gates ALL three per-tensor diag loops. Config key
  `ademamix_track_diag` (default false); plumbed via create_optimizer.
- update_clip clamp: per-tensor `u.clamp_(-c,c)` loop → batched `torch._foreach_clamp_min_/max_`
  (numerics-identical — clamp is elementwise/order-free).
- Gate proves max|Δparam|=0 AND max|Δstate|=0 between diag off↔on, and foreach-clamp==per-tensor
  clamp_ (`ignore/gate_diag_parity.py` → DIAG_PARITY_GATE_PASS).

RESULT (AFTER), opt.step ms:
- **CURE FULL deploy (track_diag=False): 65.5 ms** (was 99.4 → **−34ms reclaimed, bit-exact**)
- CURE FULL watched (track_diag=True): ~115 ms (opt-in telemetry for optimizer-debug runs)
- BARE de-fused: 30.6 | FUSED bare: 3.7

Attribution with diag OFF (this is the REAL cure-math floor, not waste):
SNR gate +13.5 | per-coord cap +16.2 | update_clip +3.2 = +34.9ms of foreach over bare 30.6.
⇒ de-fused path floor ≈ 65ms; can't `_foreach`-tune below the bandwidth of touching fp32 state.

Impact: at ~950ms/step (sps~1.0), opt.step 99→65 = ~7% throughput, lands automatically on the
TST 50k runs (cure config → track_diag defaults false). M2N cusp telemetry (MORPH_DIAG_M2G) is a
SEPARATE `_diag_capture` path (also ~per-tracked-tensor sync, ~188-230 tensors) — not affected by
track_diag; enable it only around the watched cusp.

## TIER 2 ✅ BUILT + GATED (fused-kernel cure port) — opt.step 59.7 → 4.87 ms
Ported the 4 deploy-cure knobs INTO the Triton kernel (`ademamix_b1zero_kernel.py`), all elementwise
in-register, constexpr-gated (off → bit-identical to pre-cure kernel):
- EPS_INSIDE flag (False = eps-OUTSIDE = the cure's √(ν/bc2)+ε denom).
- HAS_SNR_GATE (snr=|m2|/denom; gate=floor+(1-floor)·clamp(snr/κ,0,1)·g_coef; gg=g·gate).
- HAS_COORD_CAP (|α·m2| ≤ c·|gg|, clamp sign-preserved).
- HAS_UPD_CLIP (update ∈ [-clip,clip] post-/denom). Order MATCHES de-fused exactly.
Driver `_fused_step` passes self.{eps_inside,g_coef,g_snr_gate_kappa,g_snr_gate_floor,
stale_push_cap_coord,update_clip}. Constructor: removed the coord-cap fused-guard; ADDED guards so
the UN-ported de-fused-only features (per-tensor stale_push_cap, align_gate, num_beta1, amsgrad,
trust_ratio, flip_clamp, update_rms_clip, per-param eps overrides) RAISE on fused (no silent no-op).
diag counters (track_diag) are de-fused-only by design.

GATE `ignore/gate_fused_cure_parity.py` → FUSED_CURE_PARITY_GATE_PASS:
- PART A: fused step-1 (full cure, eps-outside) vs hand fp32 ref → rel **8e-8** (math bit-exact; step-1
  has zero quant round-trip since m2/ν init at 0). PART B: bare/eps-inside backward-compat rel 8e-8.
- PART C: fused-cure (linear-int8) vs de-fused-cure (bnb dynamic qmap), 30 steps, g-collapse →
  worst rel|Δparam| **2.9e-4**, no divergence (different quant ⇒ small bounded drift).

BENCH (`ignore/bench_cure_step.py`): **FUSED+CURE = 4.87 ms** (bare-fused 3.61, +1.26ms for the cure;
de-fused cure 59.7; AdamW8bit 13.7). END-TO-END 99.4→4.87 ms = **−94.5ms/step**; cured β1=0 now 3×
FASTER than AdamW8bit, ~0.5% of the step (was ~10%).

TO DEPLOY: flip `ademamix_fused=true` (keep `ademamix_eps_inside=false`) in the run scripts. Small
params (<4096) still take the de-fused-cure fallback (also validated). eps-outside-on-int8 contained
by the kernel's sqrt-ν storage + code-1 floor + update_clip=5.

### TIER 2 TRAINING A/B (1600 steps, real grads, same seed/data; `ignore/validate_fused_cure_ab.sh`)
- **STABILITY ✅ RESOLVED:** fused-cure ran 1600 real steps, NO divergence / NO NaN. The eps-outside-
  on-int8 concern (the whole reason a training run was needed) is PUT TO BED — sqrt-ν floor + update_clip
  contain it through real ternary-STE cusps. (Note: prune_start=3000 so this 1600-step run did NOT hit a
  prune event — prune-cusp stability on fused still relies on the de-fused cure's gauntlet pass + the
  guarded TST run.)
- **QUALITY ⚠️ NOT loss-neutral:** fused sits a SYSTEMATIC ~0.10 nats ABOVE de-fused (same seed/data, so
  not noise — the quant scheme: fused linear-int8 sqrt-ν vs de-fused bnb dynamic qmap). Steps 400-1400 Δ
  ≈ 0.075-0.12 (worst incl. early step 0.32). At loss ~4.8 (early training) this is INDISTINGUISHABLE
  between a CLOSING LAG (fused ~50-100 steps behind on the same trajectory, vanishes as curves flatten)
  and a PERSISTENT TAX (coarser int8 state = real convergence cost). 1600 steps too early to tell.
- **VERDICT: Tier 2 is STABLE + 94ms-faster but NOT yet proven loss-neutral.** Tier 1 de-fused (65ms,
  bit-exact) remains the zero-risk fallback for science runs.

### LONG A/B IN PROGRESS (lag-vs-tax, 6000 steps, PRUNE DISABLED to isolate quant; `validate_fused_cure_long.sh`)
Wolfe's call: settle it before deploying fused. Clean isolation (prune_start=999999 → only diff is the
optimizer quant). Prints Δloss trend: first-half mean|Δ| vs second-half mean|Δ| → CLOSING-LAG if it
shrinks, PERSISTENT-TAX if flat. Run `valfcl_master.log` ends in `VALFCL_DONE verdict=...`. PENDING.
DECISION RULE: lag → adopt fused (4.87ms) for deploy; tax → either keep Tier-1 de-fused (65ms, bit-exact)
or fix the fused quant (dynamic qmap in Triton) to close the gap.

## TIER 3 ✅ BUILT + GPU-GATED (the CURE for the ~0.10-nat quality tax) — fused dynamic-qmap
Root cause of the tax (read from the code, not guessed): the fused kernel used UNIFORM linear-int8
state quant (`code*absmax/127`) — coarse near 0 where the heavy-tailed optimizer state keeps its
mass — PLUS a sqrt-ν code-1 floor that biases denom UP on small-ν coords → systematic under-stepping
(one-directional → matches the consistent +sign of the 1600-step gap). De-fused never has this: it
uses bnb's NON-LINEAR dynamic qmap (`create_dynamic_map`), which spends codes near 0.

FIX (#278): ported bnb's dynamic qmap INTO the Triton kernel (`ademamix_b1zero_kernel.py`), gated by
`DYNAMIC_QMAP` constexpr (off → linear path bit-identical). m2 → signed map, ν stored DIRECTLY
(no sqrt, no floor — dynamic map represents small ν faithfully) → unsigned map. dequant = 1KB LUT
gather; requant = 8-step vectorized binary search (`_nearest_code`). Same ~2.03 B/param footprint.
Plumbed: `ademamix_b1zero.py` _fused_step (uint8 codes m2_dcode/nu_dcode + damax; distinct keys so
dead-mask can't confuse code-0≠value-0); `_mask_dead_state` dynamic branch (m2→signed-zero-idx=127,
ν→code 0); `optimizer.py` + `base.yaml` flags `ademamix_fused_dynamic_qmap` / `ademamix_fused_nu_floor`.

GATES (GPU, run on the freed card after killing the long A/B):
- `ignore/gate_fused_dynamic_qmap.py` → **GATE_FUSED_DYNAMIC_QMAP_PASS**. 4-arm controlled (heavy-
  tailed quadratic, 500 steps, shared init/grad): fused-dynamic (C) ↔ de-fused bnb reference (A)
  param drift = **1.9e-5** (reproduces the validated reference); quant error vs fp32-no-quant:
  dynamic **0.056 = de-fused ref**, linear **0.146** (2.6× worse). NOTE: the static-toy *objective*
  ordering is NOT a valid quality arbiter (constant gradient field) — dropped as a criterion;
  parameter-drift-from-no-quant is the right metric and it's decisive.
- `ignore/gate_fused_cure_parity.py` → **FUSED_CURE_PARITY_GATE_PASS** still (rel 8e-8) — the linear
  path is untouched / bit-exact (the dynamic branches are constexpr-off by default). No regression.
- CPU pre-check: `_nearest_code` 8-step bisect == brute-force argmin on both real bnb maps (0 genuine
  errors). signed-map zero @idx 127, unsigned-map zero @idx 0 (confirms dead-mask indices).

PROVEN: kernel correct + reproduces the gauntlet-winning de-fused quantizer at full fused speed
(~4.87ms+LUT/bisect overhead; benchmark TBD). NOT YET DIRECTLY MEASURED (no-theater): the real-
training loss tax is gone — strongly implied (fused-dynamic == de-fused, and de-fused beat fused-
linear by 0.10 nat in the 1600-step A/B) but the definitive check is a short fused-dynamic-vs-defused
real-training A/B (should now OVERLAY, vs linear's +0.10). DEPLOY: `ademamix_fused=true
ademamix_fused_dynamic_qmap=true ademamix_eps_inside=false`. Long lag-vs-tax A/B was KILLED (Wolfe:
"you basically just fixed it") — its verdict is moot now that the fix exists.

## TIER 2 — (original notes) fused-kernel port
de-fused 65ms vs FUSED 3.7ms ⇒ ~60ms more reclaimable. Requires porting the cure INTO the Triton
kernel (`ademamix_b1zero_kernel.py`): per-coord cap + SNR gate + an eps-**outside** option (the
kernel currently HARDCODES eps-inside, which cost ~0.5 nats convergence on de-fused — see the
eps_inside comments). Needs: kernel work (tile-prover/cuda agent), a numerics parity gate vs the
de-fused cure (the validated reference), and re-validating eps placement. Decision pending Wolfe.

## Files
`morph/training/ademamix_b1zero.py` (track_diag flag + gated diags + foreach clamp),
`morph/training/optimizer.py` (plumb), `morph/configs/base.yaml` (ademamix_track_diag),
`ignore/gate_diag_parity.py` (DIAG_PARITY_GATE_PASS), `ignore/bench_cure_step.py`,
`ignore/bench_optimizer_step.py`. All uncommitted on branch migrate/ademamix-coordcap-cure.
