# Agent Note: Tst Recipe And Mce Fix

Status: implemented

Origin: Ai-notes/06-21-2026/TST-50k-Run-Prep/RECIPE_AND_MCE_FIX.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# TST 50k Run Prep — Paper Recipe, Impl Audit, MCE Fix (2026-06-21)

Prep for: run TST + full prune→carve→route pipeline at 50k steps, ademamix (β1=0 coord-cap
cure) first, then AdamW8bit. Goal: extend the TST phase so MOST pruning happens during it.

## 1. Paper recipe (arXiv:2605.06546, Nous Research — "Efficient Pre-Training with Token Superposition")
- **s (bag size):** fixed per run, NOT annealed. 6 for ≤3B (sweet spot [4,8]); s=8 needs power-law output weighting g(i)=1/i (uniform degrades at s≥8). 10B used s=16.
- **r (TST fraction):** [0.2,0.4]; **>~0.5 the recovery phase provably fails to catch up (Fig 4).** Center 0.3.
- **Loss (MCE):** mean of s ordinary CE terms vs the s next-bag targets against the SAME logits (Eq.3/Listing3). `-log s` const dropped. **Init ≈ log(V), NOT log(V)/s** — the wiring tell.
- **Input:** plain mean of s embeddings (÷s), accumulate fp32 then cast. NO 1/√s.
- **Causality:** NTP labels shifted left by s−1, non-overlapping bags.
- **Equal-FLOPs:** multiply data seq-len L by s during TST → latent processed length ⌊L/s⌋ ≈ baseline, effective text s× longer. (MORPH `data.py` already does this: serves s·seq_len tokens, processes seq_len.)
- **Transition:** HARD switch — remove TST input-folding + MCE, resume NTP from the ckpt. Same LR across phases.
- **CRITICAL invariant (Table 2):** share input-embed + LM-head, untied, **unchanged across the boundary. Re-init at boundary makes TST WORSE than baseline** (3B: 2.676 → 2.938).
- **Optimizer (paper):** AdamW β1=.9 β2=.95, WSD (2k warmup, decay last 10%). LR 2e-4 (3B).
- **Eval:** only AFTER recovery completes (mid-TST output is gibberish). No separate de-superposition FT.
- **Failure modes:** transition loss spike (blog: ~1-2 nats, recovers in a few k steps); r>0.5 fails; hinge/BCE bag losses worse; embed/head re-init at boundary kills all gains.

## 2. Impl audit (official repo @ migrate/ademamix-coordcap-cure)
- **MCE bug (#274) was STILL present** — `transformer.py` flattened 3-D `[B,L,s]` bag labels and called single-hot `fused_linear_cross_entropy`; `labels[0:N]` silently truncated → loss ≈ log(V)/s ≈ 1.80 (reproduced V=49152: bug 1.8006 vs correct MCE 10.8021). Retroactively explains "#231 TST neutral/negative" — the objective was never connected.
- **Everything else faithful + built:** data pipeline builds `[B,L,s]` with left-shift causality; input embeds mean-pooled; two-phase scheduler complete (`tst_phase1_steps`, hard switch to NTP at boundary, ckpt-at-switch, loader rebuild; val/gen force bag=0). Equal-FLOPs convention matches paper.
- `fused_linear_cross_entropy_mce` (fused_ce.py:191) existed, verified correct (=log V), but imported/called NOWHERE.

## 3. THE FIX (done + gated, 2026-06-21)
- `transformer.py:26` — import `fused_linear_cross_entropy_mce` + `multi_hot_cross_entropy_reference`.
- `transformer.py` kernel loss branch — `if labels.ndim==3:` route to MCE (`labels.reshape(-1, s)`); else single-hot (unchanged). NTP path byte-identical → cure + sparse pipeline untouched at bag=0.
- eager else branch — same `ndim==3` guard → `multi_hot_cross_entropy_reference` (defensive; eval forces bag=0).
- **Gate `ignore/gate_tst_mce_wiring.py` → TST_MCE_WIRING_GATE_PASS** (live 33.2M model, bf16):
  - TST bag_size=6 loss=8.28 (log V=7.62, bug=5.83) — MCE WIRED, far from bug.
  - NTP bag_size=0 loss=8.90 ≈ log V — single-hot unregressed.
  - MCE backward → finite nonzero embed grads.
  - MCE @ K=1 vs single-hot Δ=0.00e+00 (bit-exact reduction, isolated at fused_ce API).

## 4. Run plan (50k)
- **Config deltas only:** `training.steps=50000`, `training.tst_ratio=0.4`. (tst_bag_size=6 already default.)
- **Schedule UNCHANGED (Wolfe's call — don't front-load):** prune_start 3000 / interval 167 / target_density 0.25 (completes ~27k by geometry, total-independent); compact_step 29000; route_start 30000. LR FLAT 1e-4 (validated mortar winner; do NOT add WSD — confounds the A/B + deviates from validated recipe).
- **r=0.4 → TST boundary @20k:** prune (3k–~27k) SPANS the boundary, ~71% of prune events in TST (= "most pruning during TST"); carve@29k + route@30k fall in RECOVERY → un-stacks the objective-switch cusp. Within paper envelope (r≤0.4).
- **Run 1 (ademamix, FIRST):** coord-cap cure config (stale_push_cap_coord=0.5, eps_inside=false, fused=false, g_snr_gate κ=0.3/floor=0.1, update_clip=5, β2=β3=0.999, α=8/cap3.5/t_alpha=8000) + TST on.
- **Run 2 (adamw):** `optimizer=adamw adam8bit=true` (defaults) + identical TST setup.

## 5. KEY RISK (flagged, no-theater)
The TST→NTP boundary @20k is an OBJECTIVE-landscape cusp the β1=0 cure has NEVER been tested against (cure validated only on prune/carve/route cusps at 35k NTP-only). Structurally it IS the cure's target disease (stale slow-EMA m₂ holds TST-objective grads while fresh g flips to NTP) → the per-coord cap SHOULD cushion the paper's transition spike, but this is untested. Watch step 20k closely (MORPH_DIAG_M2G ratio/cos). Also: extending the validated 35k cure to 50k is new; r>0.4 is outside the paper's validated envelope.

## NOT changed (optional, deferred — minimal-perturbation for a clean A/B)
- fp32 embed accumulation in the bag mean (paper does it; bf16 mean gives healthy ~log V at s=6 — negligible).
- step-0 TST-loss startup assertion in train.py (gate already covers it).
- WSD LR schedule (validated recipe is flat; WSD = future ablation).
