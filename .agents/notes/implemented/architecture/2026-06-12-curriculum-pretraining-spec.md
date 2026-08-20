# Agent Note: Curriculum Pretraining Spec

Status: implemented

Origin: Ai-notes/06-12-2026/MORPH-Curriculum-Pretraining/SPEC.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Curriculum Pretraining — Phase P SPEC (2026-06-12)

**Project:** MORPH (`00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`). The whole
`markovian-rsa-cot` pipeline targets MORPH; Qwen-4B was a midtrain-validation proxy, now retired.
**This doc = the concrete LOCAL implementation spec** for the pretraining stage of that pipeline
(`~/projects/markovian-rsa-cot/PRETRAINING.md` is the conceptual parent).

## 0. Status / scope / non-goals
- **Status:** SPEC (pre-build). Decisions flagged `[DEFAULT]` / `[OPEN]`.
- **Goal:** validate the PRETRAINING.md *mechanics* that have never actually been run —
  multi-source blend, **length-bucketed curriculum with a ramping sequence length**, clean packing,
  RoPE context-ramp, and OOM-safe checkpoint-before-step-up — at the 276M proof scale on the 5090.
- **NON-goals (decoupled by Wolfe 2026-06-12):**
  - ❌ No MORTAR prune / carve / route. The sparse deploy pipeline is **Phase D**, runs later at a
    *fixed* context (the Markovian chunk size ~4–8K, not 32K). Isolating it avoids the
    carve-spike-vs-context-step memory collision.
  - ❌ No TST (`tst_bag_size=0`) in this run — TST is a *data schedule*; overlapping it with the
    context ramp = two simultaneous data-schedule changes = confounded. Add TST in a later phase.
  - ❌ Not the cloud data. Dolma3/Nemotron are multi-TB stream-at-cloud sources, not on disk and not
    downloadable for a proof. The cloud run swaps them into the *same* loader.
- **Precision `[OPEN — recommend dense bf16 for this run]`:** to isolate the *curriculum* mechanics
  with zero confounders, run the model **dense bf16** (ternary / int6 / adam8bit OFF). The quant stack
  is already independently validated at 276M; re-layer it once the curriculum is proven. Alternative:
  keep the full deploy-quant stack ON (more faithful, but adds QAT dynamics to a run whose job is to
  vet data/context machinery). Wolfe's call.

## 1. Model (unchanged architecture, schedules off)
MORPH 276M d=768, 3:6:3 looped (mean_depth 6 / max 8 / bptt 4), HC-Cayley n=4, GLA retention on,
windowed attention (window 128). Identical to `base.yaml` model block EXCEPT `max_seq_len` handling
(§4) and the disabled schedules above.

## 2. Data — "everything local" blend (Wolfe: "everything local")

### 2.1 Inventory (on-disk, verified 2026-06-12)
| Source | Path | Size | Role | In? |
|---|---|---|---|---|
| OpenWebText | `~/.cache/huggingface/datasets/openwebtext/**/*.arrow` | 24 G (~8B tok) | web backbone | ✅ bulk |
| code_search_net | `~/.cache/huggingface/datasets/code_search_net` | 1.5 G | code | ✅ |
| dharma 84000 (Tibetan canon) | `~/projects/datasets/dharma/output/84000_pretrain.jsonl` | 64 M | injection domain | ✅ |
| dharma public_domain | `…/dharma/output/public_domain_pretrain.jsonl` | 40 M | injection (books → long tail) | ✅ |
| dharma youtube | `…/dharma/output/youtube_pretrain.jsonl` | 6.6 M | injection | ✅ |
| dharma flatland | `…/dharma/output/flatland_pretrain.jsonl` | 0.2 M | injection | ✅ |
| book_pretrain | `~/projects/datasets/reddit/book_pretrain.jsonl` | — | books (long tail) | 🟡 `[OPEN]` confirm clean |
| **wikitext** | `~/.cache/huggingface/datasets/wikitext` | 1.3 G | **held-out clean-LM-ppl EVAL** | ⛔ NOT trained |
| synthesis-reasoning gold | `commentary/dharma_text_with_reasoning/cross_tradition_maps.jsonl` | ~3.3 M | **SFT/RL seed** | ⛔ role-split rule |
| reddit comments / bandcamp | — | 1.9 G / 12 G | other-project / music | ⛔ |

Total dharma **injection text ≈ ~110 MB ≈ ~27M tokens** (already processed; the raw 3.9 G
`dharma/sources/` can be processed later to grow this — out of scope for the proof).

### 2.2 ROLE-SPLIT RULE (no-theater landmine from PRETRAINING.md)
The synthesis-**reasoning** jsonls (`commentary`, `dharma_text_with_reasoning`, `cross_tradition_maps`,
`reimagined`) are **SFT/RL gold seed (LIMO-scale), NOT pretrain bulk.** They are **excluded** here.

The later remote-source update intentionally added broad reasoning-shaped LM corpora
(`nemotron_qa`, `reasoning`) as `reasoning_midtrain`, not as post-training gold. That is a
different role: it is next-token/midtraining substrate, not ReAct/RSA prompt grammar or reward data.
The executable rule is now:

- `pretrain_bulk`: ordinary web/code/math/domain/reference LM text.
- `reasoning_midtrain`: broad reasoning-shaped LM data allowed in this curriculum.
- `posttrain_gold`: curated SFT/RL trace, aggregation, local synthesis, or reward data; refused.

The pretokenizer, shard verifier, and runtime loader must all enforce the same role metadata so a
glob cannot silently sweep post-training gold into the LM curriculum.

### 2.3 Blend weights `[DEFAULT]` (sampling probability per source)
| Source | weight | rationale |
|---|---|---|
| OWT | 0.62 | web bulk (Dolma's 28% web + general) |
| code_search_net | 0.20 | code tilt (Dolma's 20%) |
| dharma injection (4 files pooled) | 0.13 | upsampled domain (~27M tok → ~13% ⇒ many epochs; cap epochs, see below) |
| book_pretrain | 0.05 | long-doc diversity `[OPEN]` |

- **Known gap (honest):** Dolma's **math (19%) + QA (14%)** tilt has **no non-eval local source**
  (gsm8k/MATH/etc. are eval — contamination-banned). That tilt genuinely waits for the cloud
  Dolma/Nemotron run. The local blend is web+code+domain only. Documented, not hidden.
- **Injection epoch cap:** ~27M tok at weight 0.13 over a 1–2B-token proof ⇒ injection seen ~5–10×.
  Acceptable for a proof; PRETRAINING.md warns against 1000× upsample — we are far under. Log the
  realized per-source epoch count to wandb.

## 3. Pre-tokenization + length index (the efficient, correct base)
Re-tokenizing 8B tokens every epoch (current loader does) is wasteful and gives no length info to
bucket on. Build a one-time pass:
- **`scripts/pretokenize.py`**: stream each source → StarCoder2 tokenize → append EOS doc-separator →
  write **token shards** (`uint16`/`uint32` `.npy` or memmap) + a **doc-boundary index** and a
  **per-doc token-length array**, tagged by source.
- Output: `data/pretok/<source>/{tokens.bin, doc_offsets.npy, doc_lens.npy}` (gitignored — large).
- This makes bucketing + packing O(index), exact (real token lengths), and re-runnable. Cloud reuses
  the same format with Dolma/Nemotron shards.

## 4. Curriculum + RoPE context ramp (the heart — Wolfe's directives)
**Constraints (verbatim intent):** ramp the *actual* seq length (don't pin max from start) · **no
padding / no wasted compute** (each bucket trains at its native length, real docs packed to fill) ·
**checkpoint before each step-up** (OOM + PE-shift are independent risks).

### 4.1 Stages `[DEFAULT]`
| Stage | doc-len bucket | train `seq_len` | RoPE `context_len` | token share | micro-batch |
|---|---|---|---|---|---|
| 1 | < 4K | 4096 | 4096 | ~55% | bench §5 |
| 2 | 4–8K | 8192 | 8192 | ~25% | bench §5 |
| 3 | 8–16K | 16384 | 16384 | ~12% | bench §5 |
| 4 | 16–32K | 32768 | 32768 | ~8% (final tail) | bench §5 |
- Longest docs (public_domain books / 84000 canon / whole code files / book_pretrain) **naturally
  populate stages 3–4** — the long tail is cross-source, concentrated in the final ~10–20% of tokens.
- Short→long ordering within the run; **best-fit pack** docs of like length to fill the stage's
  `seq_len` with **no pad and no cross-doc bleed** (a doc never spans a sequence boundary; a sequence
  is one or more whole docs that fit, EOS-separated, remainder carried to the next same-bucket seq).

### 4.2 Per-stage transition (ordered — the safety sequence)
```
1. CHECKPOINT model+optimizer+scheduler  → checkpoints/pretrain_curriculum/stage{k}_pre.pt   (recovery point)
2. REBUILD RoPE: CoPEEmbedding.set_context(context_len=L_k)  → re-anchor taper + rebuild cos/sin cache to L_k
3. SET loader seq_len = L_k, switch active bucket → L_k packing
4. RE-FIT micro-batch = bench[L_k]; grad_accum = ceil(target_eff_batch / micro-batch)  (effective batch HELD)
5. continue training
```
- **RoPE rebuild** is cheap (~4 MB cos/sin) but **changes the positional encoding** (the
  `context_len`-anchored wavelength taper at attention.py:88–96 re-anchors; long-wavelength freqs
  un-damp). The model must *re-adapt* — hence the recovery checkpoint and a short post-step LR/loss
  watch. `[OPEN]` whether to also step RoPE `base` (10000 → higher, ZAYA used 1M@32K) or rely on the
  taper re-anchor alone — decide from a stage-3 loss-spike probe.
- **Effective batch is preserved** across stages via grad-accum so the optimization trajectory is
  comparable stage-to-stage (only micro-batch — a memory knob — changes). Per the standing rule, the
  micro-batch reduction is **planned config, not a silent OOM reaction.**

## 5b. MEASURED (2026-06-12, RTX 5090, 276M dense-bf16, subprocess-isolated)
True per-stage ceilings + the throughput policy (replaces the §5 estimates):

| seq_len | single-step max mb | **sustained-safe mb** | grad_accum @ eff-batch 8 |
|---|---|---|---|
| 4096 | 6 (21.7GB) | **4** | 2 |
| 8192 | 3 (21.8GB) | **2** | 4 |
| 16384 | 1 (16.6GB) | **1** | 8 |

- **16K IS feasible** (mb1, 16.6GB) — the curriculum reaches the agreed endpoint. Memory is
  LINEAR in T (fwd-only ~flat 2.4→3.5GB via fused kernels + ckpt; fwd+bwd 5.4→16.6GB). The first
  shared-process probe's "16K OOM + super-linear" was an artifact (persistent AdamW state +
  fragmentation bleeding into later cells). Trustworthy probe = fresh subprocess per cell.
- **Grad-accum is throughput-free** (mb1: ga1 13.8k vs ga4 14.0k tok/s). The throughput lever is
  **micro-batch size** (mb2-3 ~20k vs mb1 ~15k tok/s @4096 — small mb underutilizes the GPU).
- **Policy:** largest sustained-safe micro-batch per stage + grad-accum to a constant eff-batch (8).
  Sustained-safe mb is ~1 below the single-step max (warmup + AdamW + fragmentation), so use [4,2,1].

### 5c. GRID (scripts/throughput_probe.py, 3 sections, subprocess-isolated)
**tok/s × peak-VRAM grid (ga=1):** 4096: mb1 15.7k/7.5GB · mb2 18.4k/11.2 · mb3 19.4k/14.9 · mb4
20.4k/18.7 · mb6 OOM | 8192: mb1 16.4k/11.2 · mb2 17.1k/18.7 · mb3 OOM | 16384: mb1 13.7k/18.7 · mb2 OOM.
- **MEMORY LAW: peak VRAM ≈ f(mb×L) = tokens-per-micro-step.** Every mb×L=16384 cell ≈ 18.7GB
  (4096×4 = 8192×2 = 16384×1). Sustained ceiling = ~16K tokens/micro-step ⇒ max mb [4,2,1].
- **tok/s is set by micro-batch size** (occupancy); at EQUAL token load, longer seq is ~33% less
  efficient (20.4k@4096-mb4 vs 13.7k@16384-mb1, same 16K tokens) — sequential attention, less batch parallelism.
- **ga-invariance:** at fixed mb, accumulation is per-token-FREE (4096 mb2 ga1/2/4 = 17.9/18.7/18.7k)
  and POSITIVE at long ctx (16384 mb1 ga1→ga4 = 13.6k→15.0k, amortizes opt.step).
- **matched eff-batch (no-accum vs accum):** 4096 eff4 mb4×ga1 20.5k (baseline) vs mb2×ga2 18.9k (−8%)
  vs mb1×ga4 15.4k (−25%); 8192 eff2 mb2×ga1 17.9k vs mb1×ga2 15.4k (−14%). The penalty is the
  forced-smaller-micro-batch, NOT accumulation. ⇒ config micro_batch [4,2,1] is ceiling-optimal;
  eff_batch via grad-accum is correct (free at fixed mb; unavoidable & helpful at 16K mb1).

## 5. Memory: bench-then-set (no guessed batch numbers)
- **`scripts/mem_probe.py`** `[STEP 0, before any real run]`: for `L ∈ {4K,8K,16K,32K}`, micro-batch
  from 1 upward, fwd+bwd+opt one step, record `torch.cuda.max_memory_allocated`. Produces the
  `bench[L_k]` micro-batch table that §4.1 references.
- Expectation (to verify, not assert): windowed attention is **O(T·w)** (linear), so attention scales
  gently; the growth is the **looped core × bptt_depth 4 × HC n=4 carrier** activations (~8× per-seq
  at 32K vs 4K). Activation checkpointing stays ON (`checkpointing: true`, `ckpt_grad_iters: -1`).
- If even micro-batch 1 OOMs at 32K → **surface to Wolfe** (options: cap curriculum at 16K for the
  local proof, or `ckpt_grad_iters` deeper, or reduce stage-4 to a token-trickle). Do NOT silently
  drop the stage.

## 6. Components to build
| File | What |
|---|---|
| `scripts/pretokenize.py` | §3 token shards + length index, per source, allowlist-gated |
| `scripts/mem_probe.py` | §5 per-seq_len micro-batch table |
| `morph/training/curriculum_data.py` | `MultiSourceCurriculumLoader` (weighted blend, bucket serve, best-fit no-pad packing, TST-compatible, validation split) |
| `morph/training/curriculum.py` | `CurriculumScheduler` (stage boundaries by token-count, emits transition events) |
| `morph/model/attention.py` | add `CoPEEmbedding.set_context(context_len)` → rebuild taper+cache (currently cache built once in `__init__`) |
| `morph/training/train.py` | curriculum hook: on stage event → checkpoint → RoPE rebuild → loader seq_len + micro-batch/grad-accum swap |
| `morph/configs/pretrain_curriculum.yaml` | §7 — SEPARATE from base.yaml (base.yaml's sparse run stays untouched) |

## 7. Config sketch (`pretrain_curriculum.yaml`)
```yaml
# inherits model block from base.yaml; OVERRIDES:
model:
  max_seq_len: 32768        # RoPE cache covers the MAX; ACTUAL seq ramps via curriculum (§4)
  context_len: 4096         # stage-1 anchor; stepped by the scheduler
training:
  # sparse pipeline OFF (Phase D owns it)
  prune_start: 999999999
  compact_step: 999999999
  tst_bag_size: 0           # TST OFF (isolation)
  # precision: dense bf16 for isolation [OPEN — or full deploy-quant]
  ternary: false
  embed_quant: "off"
  adam8bit: false
  bf16: true
  lr: 1e-4                  # flat; warmup [OPEN for from-scratch — may want a short warmup here]
  target_eff_batch: 16      # held across stages via grad-accum
curriculum:
  enabled: true
  stages:                   # (bucket_max_tok, seq_len, context_len, token_share)
    - {bucket_max: 4096,  seq_len: 4096,  context_len: 4096,  share: 0.55}
    - {bucket_max: 8192,  seq_len: 8192,  context_len: 8192,  share: 0.25}
    - {bucket_max: 16384, seq_len: 16384, context_len: 16384, share: 0.12}
    - {bucket_max: 32768, seq_len: 32768, context_len: 32768, share: 0.08}
  rope_base_step: false     # [OPEN] also raise RoPE base at long stages?
  checkpoint_before_stepup: true
data:
  blend:
    - {name: owt,   path: "~/.cache/huggingface/datasets/openwebtext/**/*.arrow", weight: 0.62}
    - {name: code,  path: "~/.cache/huggingface/datasets/code_search_net",        weight: 0.20}
    - {name: dharma, paths: [84000_pretrain.jsonl, public_domain_pretrain.jsonl,
                             youtube_pretrain.jsonl, flatland_pretrain.jsonl],     weight: 0.13}
    - {name: books, path: "~/projects/datasets/reddit/book_pretrain.jsonl",        weight: 0.05}
  eval_holdout: wikitext    # clean-LM-ppl, NEVER trained
```

## 8. Verification gates (no-theater, per piece)
1. **Pre-tok parity:** decode N random docs from the token shards → byte-identical to source text.
   Doc-boundary index round-trips (offsets[i+1]-offsets[i] == doc_lens[i]).
2. **Role-split gate:** assert the loader's resolved file list contains ZERO reasoning-gold paths
   (`commentary|reasoning|cross_tradition|reimagined`). Fails loudly if a glob sweeps one in.
3. **No-pad / no-bleed gate:** over a stage's served batches, assert (a) zero pad tokens, (b) every
   sequence is a concatenation of *whole* docs (no doc spans a boundary), (c) realized seq_len == L_k.
4. **Blend-weight gate:** over 10k served sequences, realized per-source token fractions within ±2%
   of configured weights; log to wandb.
5. **RoPE rebuild gate:** `set_context(L)` then forward a length-L batch → no index error; cos/sin
   cache shape == L; a fixed length-4K batch gives **identical** output before/after a `set_context`
   that keeps context_len=4096 (idempotence); and the taper actually changes when context_len changes
   (assert `inv_freq` differs). 
6. **Curriculum transition smoke:** tiny run (few hundred steps/stage, seq 256→512→1024 scaled) →
   checkpoint written per stage, RoPE rebuilt, loss finite across all transitions, resume-from
   `stage{k}_pre.pt` works.
7. **Mem probe** (§5) produces the micro-batch table; the real run's stage-1 peak alloc matches the
   probe within ~10%.
8. **End-to-end local run:** launch; confirm density of metrics — per-source epoch counts, realized
   seq_len per stage, peak mem per stage, wikitext held-out ppl falling. Name what's unverified
   (e.g. 32K stage stability, whether the PE re-adaptation costs ppl).

## 9. Open decisions (for Wolfe)
1. **Precision:** dense-bf16 isolation `[recommend]` vs full deploy-quant stack ON.
2. **Endpoint:** ramp to 32K, or cap the *local* proof at 16K (5090 memory) and leave 32K for cloud?
   (Bounded-context insight says reasoning doesn't need 32K; 32K is an input-handling property.)
3. **RoPE base step-up** at long stages, or taper-reanchor alone? (decide from stage-3 spike probe)
4. **Warmup** for a from-scratch run (base.yaml's flat-no-warmup was a *resume* recipe; cold start may
   want a short warmup).
5. **book_pretrain** in/out; **token budget** (total steps) for the proof.
6. **Step budget per stage** — proportional to token_share, or fixed wall-time per stage?

## 10. Sequencing
**Phase P (this):** dense curriculum pretrain 4K→(16K/32K) → checkpoint = the base for everything.
**Phase D (later):** load Phase-P ckpt → sparse deploy pipeline (prune/carve/route + TST) at FIXED
chunk-size context (4–8K). **Midtrain / SFT / RL / RSA** follow per the markovian-rsa-cot docs.
