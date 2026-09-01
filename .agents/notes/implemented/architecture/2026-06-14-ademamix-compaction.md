# Agent Note: Ademamix Compaction

Status: implemented

Origin: Ai-notes/06-14-2026/AdEMAMix-Optimizer/compaction-22-20.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Context Dump — 2026-06-14 ~22:20 CDT (pre-compaction)

## Current Work
AdEMAMix-vs-AdamW optimizer campaign for MORPH (arXiv:2409.03137). Repo: **official**
`00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`, branch **`ablation/ademamix-optimizer`**,
ALL UNCOMMITTED, NOT pushed. approved working on the official repo "carefully."
**Memory + speed axes DONE + verified. Only the ppl quality A/B remains (deferred — deferred, "will run it soon").**

## What's built & verified (every claim has a cited test, no-theater)
- **bnb falsification (the big finding):** bnb's β1=0 "memory trick" is a NO-OP. `AdEMAMix8bit`
 allocates the stacked m1+m2 buffer regardless of β1 → 3.05 B/param vs AdamW8bit 2.03 (+50%),
 measured for β1=0.9 AND β1=0.0. Quantization (4×/buffer) dominates the buffer-skip.
- **Memory (real 276.5M deploy model):** AdamW8bit 1294.8MB → bnb AdEMAMix8bit 1942.0MB (+647MB,+50%).
- **Speed (isolated opt.step):** AdamW8bit 14.5ms, bnb AdEMAMix8bit 16.5ms (+0.3% step). The early
 "25% slowdown" was a non-steady-state rolling-sps artifact (DEBUNKED via converged smoke: AdEMAMix
 1.28 sps ≈ AdamW). Optimizer is ~2% of the ~780ms step → opt speed barely moves training wall-clock.
- **THE FORK `AdEMAMixB1Zero` (β1=0, 2 buffers, blockwise-8bit) — DONE + END-TO-END VERIFIED:**
 - 2.031 B/param = AdamW8bit parity (recovers the 647MB).
 - Fused Triton kernel (linear-int8, block 256) opt.step **3.47ms** (faster than AdamW 14.5 / bnb 16.5).
 - GATES (independently reproduced by me): core-math vs bnb `_ReferenceAdEMAMix(β1=0)` max|Δ|=2.4e-7;
 fused-int8 vs fp32 cos=1.000000 rel=1.4e-4; memory 2.031; state_dict roundtrip reproduces step;
 determinism torch.equal Δ=0.0; **end-to-end 300-step deploy-config train: loss 11.37→6.84, no NaN,
 sps 1.30 ≈ AdamW 1.27, RC=0.**

## Key Files
- `morph/training/optimizer.py` — `create_optimizer`: added `optimizer=adamw|ademamix|ademamix_b1zero`.
 AdEMAMix uses bnb AdEMAMix8bit/32; b1zero builds `AdEMAMixB1Zero(fused=True)`. b1zero forces
 no-decay group `optim_bits=32` (embeddings stay 32-bit). Schedulers default to `training.steps`.
- `morph/training/ademamix_b1zero.py` — the fork optimizer. `_foreach` fp32 fallback + `fused` path.
 β3-warmup degeneracy at β1=0 fixed via `beta3_warmup_start` (default 0.9; paper's log(β1) → -inf NaN).
- `morph/training/ademamix_b1zero_kernel.py` — fused Triton kernel (built by cuda-kernel-master agent).
- `morph/configs/base.yaml` — added optimizer/beta3/ademamix_alpha/t_alpha/t_beta3/beta3_warmup_start
 (defaulted so `optimizer=adamw` deploy default is bit-identical to before).
- `ignore/` (gitignored): gate_ademamix_mem.py, gate_ademamix_b1zero.py, bench_optimizer_step.py,
 train_smoke_fork.sh, speed_smoke*.sh. All reusable.

## Footguns & Gotchas
- **STABILITY SCHEDULERS ARE ESSENTIAL + must be non-None.** bnb only applies α/β3 warmup when
 t_alpha/t_beta3 are set; else β3=0.9999 from step 0 → divergence (even w/ LR warmup). create_optimizer
 defaults them to training.steps. Do NOT remove.
- **No 8-bit optimizer is bit-exact to fp32** (state is quantized). The fork is deterministic
 (Δ=0.0 run-to-run) but ~1e-4 vs fp32 — that's correct, not a bug.
- **Fused kernel uses linear-int8, NOT bnb dynamic-map** → fused ≠ de-fused bit-for-bit (both valid).
- Pre-existing uncommitted changes on the branch (README, pretrain_curriculum.yaml, pretokenize.py,
 data/) are NOT mine — do not commit them.
- ce_chunk/TST: speed smokes used `tst_bag_size=0` for clean loss curves; deploy default has TST on
 (superposition makes short-run loss look chaotic — NOT a bug; flagged this earlier).

## Decisions Made
- Path B (Triton fused kernel) over rebuilding bnb from source (binary wheel, no csrc; fragile CUDA-13/
 sm_120 build). Fused beats bnb because linear-int8 requant > bnb dynamic-map binary-search.
- Fork built de-fused first (bnb.functional), measured, THEN fused — empirical, no premature opt.
- linear-int8 chosen over dynamic-map in kernel (fidelity gate passed, simpler/faster).

## Incomplete / Next Steps
- [ ] **QUALITY A/B (the only remaining task, #271):** does AdEMAMix lower ppl vs AdamW on MORPH?
 Budget: 100k deploy config ≈ 22h/arm on one 5090; pipeline (prune→carve→route by ~30k) needs
 ≥~35k steps (~9h/arm) to be meaningful. Paper: AdEMAMix shines at long horizons (256k+).
 Arms: AdamW8bit (baseline) vs stock AdEMAMix8bit (β1=0.9 = paper-best quality, +647MB).
 Fork (β1=0) optional 3rd arm (now faster+leaner than AdamW; paper says β1=0 slightly worse quality).
 Launch with: `optimizer=ademamix training.beta2=0.999 training.beta3=0.9999 training.ademamix_alpha=8.0`
 (β2=0.999 is AdEMAMix's recommended, vs our AdamW recipe's 0.95). Use wandb (mandatory).
- [ ] Decide commit/push of the branch (locked decision; nothing committed yet).
- [ ] Optional: switch de-fused reference to linear-int8 so fused≡de-fused bit-exact (cosmetic).

## Context That Won't Survive
- 's priority was MEMORY+SPEED, both answered. He explicitly wanted fork opt.step speed "back a
 little before longer runs" → delivered 3.47ms (overshot). He knows opt is ~2% of step (~1% training impact).
- Recommendation given: don't burn GPU tonight; run paired ~35k A/B when ready. He said "will run it soon."
- Hardware: one RTX 5090 (sm_120, 32GB); DGX Spark available (ssh dgx-spark) but bnb+Triton there unverified.
- AdEMAMix research report (paper hyperparams) is in the transcript — α=8, β2=0.999, β3=0.9999,
 t_alpha=t_beta3=T, lr same as AdamW (1e-4 flat for us), wd=0.1, eps=1e-8.

## Memory written
`memory/morph_ademamix_optimizer.md` + MEMORY.md index line (both saved this session).
