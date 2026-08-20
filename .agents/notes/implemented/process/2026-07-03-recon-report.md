# Agent Note: Recon Report

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Recon-Report.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Recon Report

**Date:** 2026-07-03  
**Mode:** Read-only engineering reconnaissance (no code/config/test mutations)  
**Scope:** Public repo quality vs private research-system quality, runtime invariants, training maintainability, reproducibility, tests, ignored evidence, kernels, configs, docs.

---

## 1. Executive Summary

MORPH is a **high-quality research training system** with unusually strong internal documentation of *why* load-bearing choices exist (BPTT + compile fork-safety, phase-boundary optimizer rebuild, AdEMAMix β1=0 cures, kernel-vs-reference A/B). The private `ignore/` + `Ai-notes/` ecosystem is large, structured, and evidence-rich: gate scripts, parity JSON, optlogs, watchdogs, and dated mental-model writeups. That private system is what makes the architecture a “coherent survivor,” not the public test suite.

As a **public open-source artifact**, the repo is only partially legible. The README and `base.yaml` correctly name the survivor stack and phase schedule, and `CONTRIBUTING.md` states a coherent-architecture policy. But an external engineer cannot reconstruct *which* components were rejected, *what* ppl/throughput deltas justified survivors, or *how* to run the private verification gates. Public tests (48 collected) cover peripheral contracts (data placement, seed-model patches, posttrain schema, one kernel-fence suite) and almost none of the training lifecycle (prune→carve→route, resume, TST, QAT, BPTT).

Process-global kernel/compile/CUDA-graph switches are **mostly justified** by recurrent-loop + autograd + compile constraints, not accidental globals. The main risks are: (1) mutability after init without freeze/guards, (2) dual kill-switches (`MORPH_FORCE_EAGER` / `use_kernels` vs import-time `DISABLE_FUSED_KERNELS` and per-op `MORPH_FUSED_*`), (3) `train.py` as a 2k-line operational monorepo, (4) unpinned deps + machine-local data/W&B paths, (5) `training.seed` applied in code but absent from `base.yaml`.

**Bottom line:** Research-system quality is strong; public-artifact quality is a thin shell over a private evidence mountain. Highest leverage is documentation + curated evidence digest + small contract tests — not architectural rewrites.

---

## 2. Working Tree Safety Check

### Initial `git status --short`

```
 M README.md
 M docs/figures/deployment/morph_deploy_stack.pdf
 M docs/figures/deployment/morph_deploy_stack.tex
 M docs/figures/morph_deploy_stack.png
 M morph/configs/base.yaml
```

### Final `git status --short`

```
 M README.md
 M docs/figures/deployment/morph_deploy_stack.pdf
 M docs/figures/deployment/morph_deploy_stack.tex
 M docs/figures/morph_deploy_stack.png
 M morph/configs/base.yaml
```

**Working tree did not change** as a result of this recon. The only write was under gitignored `Ai-notes/07-03-2026/MORPH-Recon-Report.md`. Pre-existing dirty files (README, deploy-stack figures, `base.yaml`) were left untouched.

---

## 3. Codebase Map

| Path | Ownership / responsibility |
| --- | --- |
| `morph/model/` | Architecture: looped transformer, attention (CCA/CSA/HCA), HC residual, GLA, embeddings, sparsity, routing, QAT, fused CE |
| `morph/kernels/triton/` | Fused Triton kernels + pure-PyTorch references; process-global eager flag |
| `morph/sparse/stk/` | Vendored MORTAR/BCSR sparse backend |
| `morph/training/` | Hydra train entry, pruning schedule, optimizers (AdEMAMix β1=0), data/curriculum, spectral penalty, SFT |
| `morph/inference/` | Generation engine, KV cache, deploy quant |
| `morph/posttrain/` | Deploy artifacts, masks, validation contracts |
| `morph/jax/` + `morph/interop/` | Lagging JAX mirror + PT↔JAX converter (not HC-Cayley parity) |
| `morph/configs/` | Hydra recipes (`base`, `cloud`, curriculum, SFT, scale30b, olympiad seed) |
| `tests/` | Small public contract suite (48 tests) |
| `docs/` | References map, data-placement design, olympiad interop, figures |
| `scripts/` | Pretokenize, mem/throughput probes, posttrain accept |
| `ignore/` (gitignored) | Private gates, benches, repros, logs, optlogs, JSON results (~hundreds of files) |
| `Ai-notes/` (gitignored) | Dated research writeups (mental model, AdEMAMix cures, etc.) |
| `checkpoints/`, `wandb/`, `outputs/`, `data/` | Runtime artifacts (gitignored) |

`train.py` (~2058 lines) is the operational center of gravity. Model code is modular; training orchestration is not.

---

## 4. Runtime Invariants and Global State

### Process-global / env-global inventory

| Switch | Location | Classification | Notes |
| --- | --- | --- | --- |
| `_FORCE_EAGER` / `MORPH_FORCE_EAGER` | `morph/kernels/triton/_eager_flag.py` | **Justified invariant** (with risk) | Set at model build from `cfg.use_kernels` (`transformer.py:576–577`). Enables same-architecture kernel-OFF A/B. Mutable via `set_force_eager()` anytime — **no freeze after init**. |
| `_HC_FORCE_EAGER` / `MORPH_HC_FORCE_EAGER` | `_eager_flag.py` | **Justified A/B tool** | Isolates HC kernel vs other fused kernels. Same mutability risk. |
| `DISABLE_FUSED_KERNELS` | `attention.py:37` (import-time) | **Risky coupling / legacy** | Import-time gate for window path only; runtime path uses `force_eager()`. Dual kill-switch history; partially fixed (see `tests/test_kernel_compile_fences.py`). |
| `MORPH_FUSED_ATTN_PROJ`, `MORPH_FUSED_ATTN_QKCONV`, `MORPH_ROPE_CAST_CACHE` | `attention.py:82–96` | **Convenience / perf toggles** | Default ON; in-process setters exist. Bit-exact claims documented in comments; not config-logged by default. |
| `MORPH_FUSED_GLA_PROJ` | `gla.py:48` | Same as above | Env-only perf toggle. |
| `MORPH_FUSED_ROUTER_TAIL` | `routing.py:43` | Same | Env-only. |
| `MORPH_FUSED_ROUTER` | `inference/engine.py:65` | Inference-only | Default ON. |
| `MORPH_SDD_SPLIT` | `sparse/stk/backend/triton_kernels.py:14` | Perf killswitch | Default ON. |
| `MORPH_STATIC_GRAPHS` + `set_static_graphs` | `transformer.py:51–89` | **Justified optional invariant** | Front/back CUDA graphs; core loop stays eager (variable active-set). Default OFF. Documented bit-exactness class A + failure-must-abort. |
| `MORPH_OPT_CUDA_GRAPH` | `ademamix_b1zero.py:59` | **Justified optional** | Optimizer-step graph; couples with static-graph grad views. Default OFF. |
| `donated_buffer = False` | `train.py:45–46` | **BPTT/compile invariant** | Required for checkpointed looped core under compile. |
| Inductor `worker_start_method=spawn` | `train.py:64–67` | **Compile-safety invariant** | Only when `MORPH_COMPILE_CARVED`; fork-deadlock mitigation. |
| `torch.compiler` stance `eager_on_recompile` | `train.py` (post-warmup) | **Justified invariant** | Prevents mid-loop recompile forks. Generation uses `@set_stance("force_eager")` (`train.py:114`). |
| `MORPH_COMPILE_CARVED` | `train.py:1789+` | **Measured opt-out** | Default OFF at d=768 (carved-eager faster). |
| `MORPH_PROFILE_REGIONS` | `transformer.py:38` | Diagnostic | Zero-cost when off. |
| `MORPH_DIAG_*`, `MORPH_MEM_*`, `MORPH_PERF_*`, `MORPH_EXACT_TRACE`, `MORPH_NSYS_*`, `MORPH_PROF_*`, `MORPH_DEBUG_STEP`, `MORPH_FAULT_TIMEOUT`, `MORPH_DIV_*` | `train.py` | Diagnostic / ops | Env-gated; well-commented; do not belong in hot-path defaults. |
| `MORPH_DATA_*` | `data_placement.py` | Runtime data policy | Overrides `data_runtime` config; documented. |
| `_CUDA = "/opt/cuda/..."` | `kernels/l2_persist.py:19` | **Local path assumption** | Hardcoded CUDA toolkit path; degrades to no-op on failure (documented). |
| `wandb.entity: adew-me` | `base.yaml:293` | **Local identity leak** | Public default points at maintainer entity. |
| `data.dataset: /home/wolfe/.cache/...` | `base.yaml:275` | **Local path** | Blocks third-party runs without override. |

### Classification summary

- **Legitimate BPTT/compile/autograd invariants:** donated buffers, compile stance, warmup-before-threads, force_eager for generation, static-graph capture rules, HC/attn reference fallbacks for Dynamo fences.
- **Convenience shortcuts:** per-op `MORPH_FUSED_*` env toggles not in Hydra config.
- **Hidden coupling:** process-global `_FORCE_EAGER` shared across all models in-process; last-built model wins if multiple configs constructed; tests mutate it without a context manager.
- **Undocumented / insufficiently guarded:** no “freeze kernel mode after init”; `DISABLE_FUSED_KERNELS` still coexists; PruningSchedule *code* defaults (`pruning.py:71–76`) diverge from `base.yaml` (3000/167/29000/30000).

---

## 5. BPTT / Compile / Kernel Coherence

### What the design correctly encodes

The looped core is a **nested dynamical system**: outer optimizer steps and inner `h_{k+1}=f_θ(h_k)` over T iterations with truncated BPTT (`bptt_depth=4`) and selective activation checkpointing (`ckpt_grad_iters`). Comments in `train.py` and `transformer.py` show this is not folklore:

1. **Compile only MLPs**, not attention (Triton/SDPA incompatible with fullgraph) — `train.py:1152–1165`.
2. **Core MLPs use `dynamic=True`** because active-set shrinking changes batch each loop iteration.
3. **Warmup forces every active-set size including `n_active==1`** so Triton size-specializations and Inductor guards compile in a **thread-free window** before wandb/dataloader threads exist — prevents glibc-arena fork deadlock (`train.py:1167–1197`, `warmup_compile_all_shapes` at L906).
4. **`eager_on_recompile` stance** after warmup: leftover guards run eager rather than forking compilers mid-loop.
5. **Generation uses `force_eager` stance** because token-by-token shapes would thrash compiled MLPs (`train.py:114–134`).
6. **Dynamo fences** (`@torch.compiler.disable` on fused autograd Functions) — documented in `tests/test_kernel_compile_fences.py`: tracing into Triton IR mis-launches / feeds fp64 to `tl.dot`.
7. **Static graphs only wrap fixed-shape front/back**; variable-depth core stays eager (`transformer.py:51–56`). Fused CE stays eager (host `.item()` illegal in capture).
8. **Routing aux uses `aux_detach_input: true`** so load-balance grads do not extend BPTT depth into OOM (`base.yaml:269–270`).

These are **weird-but-justified**. Treating them as ordinary “global state cleanup” would break training.

### Missing documentation / guardrails

| Gap | Why it matters |
| --- | --- |
| No public “Runtime Invariants” doc | External contributors will “fix” globals and reintroduce hangs |
| `set_force_eager` not frozen after build | Mid-run flip changes autograd path under live compile stance |
| Kernel mode not in checkpoint metadata | Resume could theoretically disagree with build-time flag if env differs |
| Private hang postmortems (`Ai-notes/06-01-2026/...`) not summarized publicly | Knowledge dies with the machine |
| Mental model (`Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md`) is the real architecture doc | Public README does not link contractivity / ρ(J_core) story |

---

## 6. Training Runner Maintainability

`morph/training/train.py` (~2058 lines) interleaves:

| Concern | Approx. lines | Notes |
| --- | --- | --- |
| Import-time compile safety | 21–67 | donated_buffer, spawn workers |
| Eval / generation | 88–170 | force_eager stance on gen |
| Config → MORPHConfig | 175–228 | |
| Checkpoint save/load/init_from | 230–420 | `_orig_mod` alignment, next_step, topology |
| Diagnostic probes (optstate, m2g, fwd norms) | 421–904 | ~500 lines of research instrumentation |
| Warmup compile | 906–947 | Load-bearing |
| `main()` setup (seed, QAT, compile, wandb, data, resume) | 952–1440 | Ordering is load-bearing |
| Loop: curriculum, static graphs, profilers, exact trace, prune/compact/route, opt rebuild | 1446–end | Operational monorepo |

### Recommended extraction boundaries (future work only)

1. **`morph/training/diagnostics.py`** — `diag_*` functions (L421–904). Pure observation; zero behavior change if env-gated the same way.
2. **`morph/training/checkpointing.py`** — `save_checkpoint`, `load_checkpoint`, `load_weights_only`. Already self-contained.
3. **`morph/training/compile_warmup.py`** — `warmup_compile_all_shapes` + stance arming. Keep comments about fork-safety.
4. **`morph/training/loop_hooks.py`** or config-driven registry — mem probe, nsys, kineto, exact-trace, divergence guard. Today these are inline env reads in `main`.
5. **Leave in `main`:** phase transitions (prune/carve/route), optimizer rebuild + GC, TST bag_size switch, curriculum stage-up — these are the training *semantics*.

Do **not** extract for cleanliness alone if it risks reordering the thread-free compile window or phase-boundary `continue` path.

---

## 7. Reproducibility Assessment

### Already strong

- Hydra configs; `base.yaml` is heavily commented and is the stated source of truth for the local recipe.
- Full resume: model + carve/route topology + optimizer + scaler + RNG + step + pruning phase (`base.yaml:249–257`, `load_checkpoint`).
- `resume_fresh_optimizer` for A/B forks without stale slow-EMA.
- Data-stream replay on resume (deterministic skip of batches).
- `MORPH_EXACT_TRACE` for bit-identical loss hex traces.
- Seed application fixed 2026-07-03 in `train.py:959–973` (was previously logged but not applied).
- Prefetch determinism tests in public suite.
- Phase schedule comments explain prune/carve/route/TST coupling.

### Blocks third-party reproduction

| Blocker | Evidence |
| --- | --- |
| Unpinned dependencies | `pyproject.toml`: `torch>=2.1.0`, `triton>=2.1.0`, no lockfile, no CUDA version pin |
| Local dataset path | `base.yaml:275` `/home/wolfe/.cache/huggingface/datasets/openwebtext/...` |
| W&B entity default | `base.yaml:293` `adew-me` |
| `training.seed` not in `base.yaml` | Applied via `getattr(tr, "seed", 0)` — default 0 is implicit |
| No known-good environment matrix | README mentions 5090/4090 casually; no torch/CUDA/driver table |
| Private evidence for “validated winner” claims | Comments cite phase_c_mortar, A-series, #231, etc. — live in `ignore/` / `Ai-notes/` |
| `bitsandbytes` optional (`[train]`) | Required for default `adam8bit: true` / AdEMAMix 8-bit path |
| JAX path advertised but lagging | `morph/jax/` still MRR residual; CLAUDE.md naming gotcha |
| Hardcoded `/opt/cuda` in `l2_persist.py` | Graceful no-op, but silent on non-standard layouts |

### Stale / contradictory notes

- `PruningSchedule` dataclass defaults (`pruning.py:71–76`: prune_start=6000, compact=66000, route=70000) **do not match** `base.yaml` (3000 / 29000 / 30000). Safe only because `from_cfg` reads YAML; dangerous if someone constructs `PruningSchedule()` bare.
- `base.yaml` AdEMAMix comments still say “only used when optimizer=ademamix” in places while default is `ademamix_b1zero` and stock `ademamix` was removed (`base.yaml:144–145` vs `130–133`).
- README claims “enabling less than 1% ppl regression” without a public evidence pointer.
- CONTRIBUTING requires “z3 or lean” for kernel correctness — private gates use empirical parity; tile-prover exists but is not wired into public CI.

---

## 8. Test and Verification Matrix

### Public tests (`pytest --collect-only`: **48 tests**)

| Category | Public coverage | Files |
| --- | --- | --- |
| Kernel / reference parity | **Partial** | `test_kernel_compile_fences.py` (window force_eager; full-model compile; kernel≈ref logits, CUDA) |
| Shape / property | **Partial** | `test_seed_model_patches.py` (n_core=0, bigram=0) |
| Checkpoint / resume | **None public** | Private: `ignore/gate_checkpoint_next_step.py`, `gate_resume_parity.py`, `gate_resume_8bit_state.py` |
| Seed / determinism | **Partial** | Seed applied in train.py; prefetch determinism in `test_data_placement.py`; no end-to-end train seed test |
| Prune / compact / route phases | **None public** | Private: `verify_compaction.py`, `gate_prune_state_mask.py` |
| Quant / QAT | **None public** | Private: `verify_ternary_qat.py`, `gate_packed_ternary.py`, `verify_fp8*.py` |
| Inference / decode | **None public** | Private: `verify_kv_cache.py`, `verify_static_decode.py`, `bench_decode.py` |
| Data-loader / curriculum | **Partial** | `test_data_placement.py` (strong); private `verify_curriculum_loader.py`, `verify_pretok.py` |
| Post-training / SFT | **Partial** | `test_posttrain_contract.py`; private `gate_sft_*.py`, SFT pipeline logs |
| Math-tile protection | **Yes** | `test_math_tile_protection.py` |
| Optimizer / AdEMAMix | **None public** | Private: many `gate_ademamix_*`, `gate_fused_*`, optlogs |
| BPTT / loop dynamics | **None public** | Private: `loop_k_robustness.py`, mental-model notes |

### High-risk gaps (public)

1. **Phase-transition smoke:** one CPU/GPU-tiny run that hits prune → compact → route and asserts density, param-set change, optimizer rebuild flag.
2. **Checkpoint roundtrip:** save with `next_step`, load, assert step/RNG/topology.
3. **force_eager freeze contract:** building two models with different `use_kernels` must not silently share wrong global (or document single-model-per-process).
4. **Reference parity for HC / CCA / CSA / HCA / GLA** — kernels embed `__main__` parity blocks; not promoted to `tests/`.
5. **TST bag_size switch** — private `gate_tst_mce_wiring.py` only.

### Recommended small smoke / contract tests (future)

- CPU: `PruningSchedule.from_cfg(base)` defaults match documented schedule keys.
- CPU: `set_force_eager` context manager restores prior value (tests currently leave global dirty on failure paths — they use try/finally, good pattern to standardize).
- CUDA-tiny: 2-step train with `compact_step=1`, `route_start=2`, assert no exception + finite loss.
- CUDA-tiny: kernel vs reference max-rel on one block (already partially in `test_kernel_and_reference_paths_agree`).

---

## 9. Ignored Artifact Inventory

`ignore/` is a full private lab (~**191** `.log`, **120** `.py`, **85** `.sh`, **45** `.json`, **22** `.optlog`, plus multi-GB traces/m2g files).

### Categories (representative)

| Category | Examples |
| --- | --- |
| **Parity / gate scripts** | `gate_resume_parity.py`, `gate_fused_cure_parity.py`, `gate_fused_router_parity.py`, `gate_tst_mce_wiring.py`, `parity_audit*.py` |
| **Verify scripts** | `verify_compaction.py`, `verify_hyper_connections.py`, `verify_ternary_qat.py`, `verify_kv_cache.py` |
| **Benchmarks** | `bench_decode.py`, `bench_hyper_connections.py`, `bench_optimizer*.py`, `ignore/perf/*` |
| **Repros / hang hunts** | `repro_compile_hang.py`, `repro_deadlock.py`, `repro_recompute_hang.py`, `stress_kernel_hang.py` |
| **Watchdogs** | `watchdog_*.sh`, `ab_watch.sh`, `_watch_*.sh` |
| **Ablation campaigns** | `ab_kernels_*.log`, `b1zero_*_noprune_*.{log,optlog,sh}`, `ademamix_*.sh` |
| **JSON result digests** | `ab_hc_kernel_result.json` (structured summary: loss fused vs eager, step ms, speedup), `bench_*_50k.json`, `ig_on/off_*.json`, `loop_k_results/*.json` |
| **Heavy traces** | `fullmodel_hc_cayley_trace.json` (~612MB), `fullmodel_standard_trace.json` (~567MB), `tst_stp_*.m2g` (~1.3GB each) |
| **SFT pipelines** | `run_sft_*.sh`, `sft_*_pipeline.log` |

`Ai-notes/` holds dated deep dives (e.g. iterative-map mental model, AdEMAMix divergence cure) that explain *why* defaults exist.

### Can this produce a public ablation ledger?

**Yes.** There is enough structured private evidence (especially small JSON summaries and gate scripts with pass/fail) to curate a public digest **without** dumping multi-GB logs.

### Recommended minimal public artifact format

A single `docs/evidence/ablation-ledger.md` (or `docs/evidence/DIGEST.md`) with rows:

```text
| ID | Decision | Config delta | Metric | Hardware | Commit | Private source |
| #231 TST | accept | tst_bag_size=6, ratio=0.3 | −4% ppl @ d=768/15k quant-only | 5090 | <sha> | ignore/... / Ai-notes/... |
| HC fused | accept | hc_use_kernel=true | ~1.45× HC isolated; loss Δ within noise | 5090 | <sha> | ignore/ab_hc_kernel_result.json |
| STP | reject | … | … | … | … | gate_stp_removal.py |
```

Plus optional tiny committed JSON fixtures (not full traces): final loss, ppl, step_ms, max_abs_diff, commit, torch/cuda versions.

---

## 10. Config and Documentation Hygiene

### Stale / confusing

- **PruningSchedule code defaults ≠ base.yaml** (`pruning.py` vs `base.yaml:111–116,264`).
- **AdEMAMix comment drift** (references removed `ademamix` path).
- **`training.seed` missing from base.yaml** despite being load-bearing for reproducibility.
- **Local paths and W&B entity** in the canonical public config.
- **Experimental toggles mixed with architecture choices** in one flat `training:` block (spectral penalty, FP8, attn_proj_quant, many AdEMAMix cure knobs). Survivors and A/B levers look the same to newcomers.
- **README** asserts ablation discipline and “<1% ppl regression” without ledger links.
- **No public hardware/support matrix** (SM120/5090 tuning is real; other GPUs are “should fit”).
- **JAX** still in repo map without a loud “do not use for training” banner in README (CLAUDE.md is clearer).

### Architecture choices vs experimental toggles (suggested grouping)

| Architecture (survivor) | Experimental / ops knobs |
| --- | --- |
| Loop 3:6:3, HC n=4, CCA+CSA/HCA, GLA retention, MORTAR, ReMoE, ternary+int6, AdEMAMix β1=0 | `core_gain_clip`, spectral penalty, FP8, attn_proj_quant, `ademamix_g_*`, static/opt CUDA graphs, mem probes |

`base.yaml` header already lists the winning stack — promote that into README “Accepted stack” / “Rejected or deferred” sections.

### Docs that need “why,” not “what”

| Topic | Where rationale lives today | Public gap |
| --- | --- | --- |
| BPTT loop invariants | `train.py` comments, CLAUDE.md | No `docs/runtime-invariants.md` |
| Global kernel mode | `_eager_flag.py`, fence test docstring | Not in README |
| Accepted vs rejected components | Scattered Ai-notes + ignore gates | No ablation ledger |
| Phase schedule coupling | `base.yaml` comments | Good in config; weak in README table |
| Exact-trace / repro tools | `train.py` env comments | Undocumented for contributors |
| Known-good environment | Implicit (maintainer machine) | Missing |
| Contractivity / ρ(J_core) | `Ai-notes/06-19-2026/.../MENTAL-MODEL.md` | Not linked publicly |

---

## 11. Risk Register

| Sev | Finding | Evidence | Impact | Suggested remediation |
| --- | --- | --- | --- | --- |
| P0 | Unpinned torch/triton/CUDA; no lockfile | `pyproject.toml` | Third-party “repro” is fiction | Pin known-good versions; document SM/arch |
| P0 | Local dataset + W&B entity in default config | `base.yaml:275,293` | Fresh clone fails or writes to wrong W&B | Placeholder paths; env/example overrides |
| P0 | Process-global kernel mode mutable / multi-model unsafe | `_eager_flag.py`, `transformer.py:576` | Silent wrong kernel path in tests or multi-build processes | Freeze-after-init; warn on conflicting set |
| P1 | `train.py` operational monorepo | `train.py` 2k LOC | High change risk; hard review | Extract diagnostics/checkpoint/warmup only |
| P1 | Public tests miss training lifecycle | `tests/` vs `ignore/gate_*` | Regressions only caught privately | Promote 3–5 gate scripts to smoke tests |
| P1 | PruningSchedule defaults diverge from base.yaml | `pruning.py:71–76` vs `base.yaml` | Bare construction uses wrong schedule | Align defaults or force `from_cfg` only |
| P1 | Private evidence not summarized | `ignore/*.json`, Ai-notes | “Survivor architecture” claim unverifiable | Public ablation ledger |
| P2 | Dual kernel kill-switches | `DISABLE_FUSED_KERNELS` + `force_eager` | Contributor confusion | Deprecate import-time flag in docs; single path |
| P2 | `training.seed` not in base.yaml | `train.py:965` | Invisible reproducibility knob | Add `seed: 0` with comment |
| P2 | JAX lag understated in README | `morph/jax/`, CLAUDE.md | Users attempt PT/JAX parity | README warning box |
| P2 | CONTRIBUTING z3/lean bar vs empirical gates | `CONTRIBUTING.md:83` | Unrealistic contributor bar | Align policy with actual gate practice |
| P3 | Hardcoded `/opt/cuda` in l2_persist | `l2_persist.py:19` | No-op on other layouts | Discover via `CUDA_HOME` |
| P3 | README grammar / marketing claims | `README.md:4–7` | Credibility | Tone down; link evidence |

---

## 12. Recommended Recon-Only Backlog

All items are **future work**. None were implemented in this pass.

### P0 — Correctness / reproducibility risk

1. **Pin a known-good environment**  
   - Finding: floating `torch>=2.1`, `triton>=2.1`, no CUDA pin.  
   - Evidence: `pyproject.toml`.  
   - Remediation: `requirements-lock.txt` or documented versions (torch, CUDA, driver, GPU).  
   - Effort: S–M. Impact: High. Behavior-neutral: **yes**.

2. **Remove machine-local defaults from public config**  
   - Finding: `/home/wolfe/...` dataset path; `wandb.entity: adew-me`.  
   - Evidence: `base.yaml:275,293`.  
   - Remediation: `null` / env-required / `configs/local.yaml` gitignored example.  
   - Effort: S. Impact: High. Behavior-neutral: **yes** (for maintainer, use local override).

3. **Guard / freeze process-global kernel mode**  
   - Finding: `set_force_eager` anytime; last model build wins.  
   - Evidence: `_eager_flag.py`, `transformer.py:576–577`.  
   - Remediation: freeze after first build; raise on conflicting change; optional context manager for tests.  
   - Effort: S. Impact: High. Behavior-neutral: **yes** if default path unchanged.

### P1 — Maintainability risk

4. **Extract diagnostics and checkpointing from `train.py`**  
   - Evidence: `train.py:230–904`.  
   - Remediation: move to modules; keep call order.  
   - Effort: M. Impact: High. Behavior-neutral: **yes** if careful.

5. **Promote 3–5 private gates to public smoke tests**  
   - Candidates: resume `next_step`, prune-state mask, TST MCE wiring, fused-router parity (tiny shapes), compaction density.  
   - Evidence: `ignore/gate_*.py`.  
   - Effort: M. Impact: High. Behavior-neutral: **yes**.

6. **Align `PruningSchedule` defaults with `base.yaml`**  
   - Evidence: `pruning.py` vs `base.yaml`.  
   - Effort: S. Impact: Medium. Behavior-neutral: **yes** if only unused bare defaults change.

### P2 — Public-facing clarity

7. **Publish an ablation ledger / evidence digest**  
   - Curate from `ignore/*.json` + Ai-notes; do not dump logs.  
   - Effort: M. Impact: High for external trust. Behavior-neutral: **yes**.

8. **Add `docs/runtime-invariants.md`**  
   - Cover BPTT, force_eager, compile stance, static graphs, phase boundaries.  
   - Effort: S–M. Impact: High for contributors. Behavior-neutral: **yes**.

9. **Add `training.seed` to `base.yaml`; group config sections**  
   - Architecture vs optimizer-cure vs diagnostic.  
   - Effort: S. Impact: Medium. Behavior-neutral: **yes**.

10. **README: accepted stack, deferred (RSA), rejected pointers, JAX lag warning**  
    - Effort: S. Impact: Medium. Behavior-neutral: **yes**.

### P3 — Polish

11. Discover `CUDA_HOME` in `l2_persist.py`.  
12. Resolve `DISABLE_FUSED_KERNELS` vs `force_eager` documentation.  
13. Align CONTRIBUTING kernel-proof language with empirical + optional formal (tile-prover).  
14. Fix README prose / uncited regression claims.

---

## 13. Non-Goals

Intentionally **not** done in this pass:

- No file edits outside gitignored `Ai-notes/`.
- No formatting, patches, staging, or commits.
- No training, SFT, long benchmarks, or GPU-heavy jobs.
- No full pytest execution (collect-only only).
- No mutation of checkpoints, wandb, datasets, or `ignore/` logs.
- No wholesale reading of multi-GB traces (`*.m2g`, fullmodel traces) — metadata/sample only.
- No deep audit of every `docs/references/` paper note.
- No line-by-line review of all 120 `ignore/*.py` gate scripts (inventory + sampling).
- No verification that private gates currently pass on this machine.
- No JAX parity measurement.
- No task-master expand/costly commands.

---

## Appendix A — Public test inventory (collected)

```
test_data_placement.py          — 24 tests (policy, TokenStore, prefetch determinism, TST bag shapes)
test_kernel_compile_fences.py   — 3 tests (force_eager window, compile+kernels, kernel≈ref)
test_math_tile_protection.py    — 6 tests
test_posttrain_contract.py      — 6 tests
test_seed_model_patches.py      — 9 tests
Total: 48
```

## Appendix B — Key file anchors

| Topic | Path:lines |
| --- | --- |
| Eager flag | `morph/kernels/triton/_eager_flag.py:1–40` |
| Set at build | `morph/model/transformer.py:572–577` |
| Compile safety header | `morph/training/train.py:29–67` |
| Seed apply | `morph/training/train.py:959–973` |
| Warmup | `morph/training/train.py:906–947, 1167–1199` |
| Phase-boundary opt rebuild | `morph/training/train.py:1753–1824` |
| Exact trace | `morph/training/train.py:1529–1535` |
| Canonical recipe | `morph/configs/base.yaml` (full file) |
| Local path / wandb | `morph/configs/base.yaml:275,293` |
| Deps | `pyproject.toml:14–23` |
