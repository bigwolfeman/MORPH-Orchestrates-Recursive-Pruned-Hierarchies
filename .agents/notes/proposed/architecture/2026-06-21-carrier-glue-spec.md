# Agent Note: Carrier Glue Spec

Status: proposed

Origin: Ai-notes/06-21-2026/AdEMAMix-Perf-Reclaim/CARRIER_GLUE_SPEC.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# ROUGH SPEC (for later) — HC Carrier-Glue Fusion + 32-bit Embedding Pickup

Status 2026-06-21: rough scoping while the fused-cure A/B runs. NOT started. Pick up after the
TST/optimizer tests land + compact. The big lever per the whole-step profile (`jaunty-seeking-petal.md`
Step 1): **~32% of the step (~210ms) is generic `vectorized_elementwise`/`elementwise_kernel`/
`direct_copy` — bandwidth-bound HBM round-trips of the 4-stream HC carrier** (residual-adds,
injection broadcasts, casts, active-set copies). GEMMs (~80ms) are real compute; HC compute kernels
(93ms) already fused. The prize is the GLUE, not idle (step is 99% busy).

⚠️ STALE-PROFILE CAVEAT: that 32% was measured BEFORE the optimizer reclaim (opt.step 99→4.87ms).
Re-run the whole-step profiler FIRST to re-rank — the carrier fraction is now relatively larger.
⚠️ n=4 is LOCKED. Do NOT propose n=2: Wolfe measured a quality regression from n=2 and it's only
~7% compute (not the 20% I wrongly estimated). The carrier win must come from FUSION at n=4.

## What already EXISTS (don't rebuild)
- **Injection MERGE (#226, DONE):** `transformer.py::_build_injection_term` assembles x0+value-embed+
  bigram into ONE single-stream [B,S,C] additive term per layer (was a slice+cat "carrier storm").
- **Carrier-engine FOLD (#228, BUILT but NOT WIRED):** the kernels/blocks accept the NEXT layer's
  inject term and fold its broadcast-add into THIS layer's POST write (carrier touched once):
    - `kernels/triton/fused_hyper_connection.py:890` (POST kernel `term` arg, line ~1233)
    - `model/mhc.py:400,440-442` (MORPHBlock.forward `next_inject_term=` → `mrr_mlp(..., post_inject=)`)
    - `model/hyper_connections.py:225,257` (`post_inject` broadcast into every output stream)
  BUT `transformer.py` forward does NOT pass `next_inject_term` — it still does a SEPARATE
  `_apply_injection(h, term)` broadcast-add per layer (prelude line 694, core line 632, coda ~895).
  ⇒ the fold's plumbing is dead code until the forward is rewired. This is task #229's pending work.

## THE WORK (rough)
### A. Wire the carrier-engine fold through the forward (the main win)
Restructure the prelude/core/coda loops so each layer receives the NEXT layer's injection term and
folds it into its POST write, eliminating the per-layer standalone `_apply_injection` broadcast-add
(one fewer full 4-stream carrier round-trip per layer × ~84 layer-applications/step).
- Precompute all injection terms for a section (already cheap single-stream), then iterate passing
  `next_inject_term=terms[i+1]` to layer i; the FIRST layer's own term still applied up front, the
  LAST layer's POST has no fold (or folds into the coda boundary).
- Config flag `hc_carrier_engine: bool` (default false → current behavior bit-identical) so it's an
  A/B, like every other deploy knob.
- Mind the LOOPED core: the core block is applied T× (Poisson depth). The injection term is rebuilt
  per core-iteration (line 629 inside the iter loop) — the fold must compose with the loop + the
  diagonal injection (`self.injection(h_in,e_in)` at 625) and active-set shrinking. This is the
  fiddly part; get the indexing exactly right or parity breaks.

### B. Other carrier round-trips worth folding (smaller, after A)
- `carrier::expand_contig` (transformer.py:686): `x.unsqueeze(2).expand(...).contiguous()` = a full
  [B,S,n,C] carrier materialization at entry. Can the contiguous copy be deferred/avoided?
- Stream mean-reduce before LM head (`x.mean(dim=2)`, ~line 902): one carrier read.
- dtype casts + active-set copies in the loop (the `direct_copy` kernels) — fold into adjacent ops.

### C. Verification (mandatory, same bar as the optimizer cure)
- **Parity gate:** carrier_engine ON vs OFF → bit-exact (or fp32-roundoff) forward+backward on a
  small model. The merge (#226) was 0.0-diff; this must be too (it's reassociating adds).
- **Re-profile:** confirm the ~32% elementwise/copy fraction actually drops; report the real % won.
- **Short training smoke:** loss trajectory matches OFF for a few hundred steps (no silent drift).
- **#229 deploy validation run** (15k carrier_engine=ON deploy stack) before it becomes default.

## 32-BIT EMBEDDING PICKUP (separate, smaller — Wolfe: "might be a good pickup")
`training/optimizer.py:217` forces `groups[1]["optim_bits"]=32` (the no-decay group = the big
embedding tables, 49152×768 + bigram + value-embeds). bnb-8bit is unstable on sparse embedding grads,
so they take the DE-FUSED fp32 path (no quant, but full fp32 state + per-tensor Python dispatch).
Now that the rest of opt.step is ~4.87ms (fused), these few large fp32 embedding updates may be a
disproportionate slice of the remainder.
- FIRST: measure their actual contribution (bench: opt.step with embeddings in group vs excluded).
  If small, skip. If material:
- OPTION 1: route the embedding group through OUR fused kernel at bits=32-equivalent — but the fused
  kernel is int8-only; would need an fp32/bf16-state fused variant (or just a fused fp32 _foreach with
  no quant round-trip). The instability was bnb's int8 qmap, NOT fusion per se, so a no-quant fused
  path may be safe.
- OPTION 2: leave fp32 state but kill the per-tensor Python dispatch (batch the few embedding tensors
  via _foreach in one shot). Cheaper to try, lower ceiling.
- Gate: param-update parity vs current (must be bit-identical — embeddings are quality-sensitive).

## Priority order
1. Re-profile whole step (re-rank after the optimizer reclaim).  2. Carrier-engine wire (A) — biggest.
3. Measure + maybe fix the 32-bit embedding group (C-fast).  4. Secondary carrier round-trips (B).
