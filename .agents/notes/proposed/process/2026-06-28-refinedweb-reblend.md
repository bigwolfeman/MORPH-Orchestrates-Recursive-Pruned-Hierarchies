# Agent Note: Refinedweb Reblend

Status: proposed

Origin: Ai-notes/06-28-2026/Pretraining-Dataset-Curriculum/REFINEDWEB_REBLEND.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# RefinedWeb-First Pretraining Reblend

## Decision

Use RefinedWeb as the broad web backbone for MORPH curriculum pretraining, with FineWeb-Edu as the high-signal educational slice and small capped capability sources for code, math, and reasoning-shaped language-model text.

This replaces the older OWT/Dolma-heavy blend in `morph/configs/pretrain_curriculum.yaml`. OWT and Dolma remain available as comparability or fallback sources, but they should not carry the default curriculum unless an ablation proves they beat the cleaner web backbone.

## Active Blend

Token-proportioned target weights:

| Source | Weight | Role |
| --- | ---: | --- |
| `refinedweb` | 0.48 | broad filtered/deduped web backbone |
| `fineweb_edu` | 0.20 | educational/knowledge-heavy web |
| `code` | 0.12 | code fluency and structured syntax |
| `math` | 0.08 | proofs/math text |
| `reasoning` | 0.06 | procedural verified CoT-shaped LM text |
| `nemotron_qa` | 0.04 | grounded synthetic QA, capped |
| `dharma` | 0.02 | small domain text slice |

Synthetic/reasoning-shaped sources are capped at 10% total (`reasoning + nemotron_qa`) to avoid turning pretraining into SFT-shaped distribution learning. Broad FineWeb is added as a pretokenizer source for an A/B arm, but it is not in the default blend because it likely overlaps CommonCrawl content with RefinedWeb and should wait for cross-source dedup or an ablation.

## Why RefinedWeb

RefinedWeb is older than FineWeb but still a strong filtered/deduplicated CommonCrawl base, and the public Falcon RefinedWeb extract is directly available on Hugging Face. It is a much better broad-web default than OpenWebText for a data-constrained local run.

FineWeb-Edu stays in the default because its educational filtering is exactly the kind of signal-density boost we want when the model cannot afford trillions of random web tokens.

## Pretokenization

The pretokenizer now recognizes:

- `refinedweb`: `tiiuae/falcon-refinedweb`, field `content`
- `fineweb`: `HuggingFaceFW/fineweb`, field `text`
- `fineweb_edu`: `HuggingFaceFW/fineweb-edu`, field `text`

Each uses shuffled Parquet file selection with `file_limit: 256` before applying the token cap. That avoids accidentally building a capped shard from only the first alphabetical Common Crawl segment.

Suggested full shard command:

```bash
PYTHONPATH=$PWD python scripts/pretokenize.py \
  --out data/pretok \
  --only refinedweb,fineweb_edu,code,math,reasoning,nemotron_qa,dharma \
  --num-proc 8 \
  --remote-max-tokens 5000000000
```

For a smoke/gate slice:

```bash
PYTHONPATH=$PWD python scripts/pretokenize.py \
  --out data/pretok_gate \
  --only refinedweb,fineweb_edu,code,math,reasoning,nemotron_qa,dharma \
  --limit 200
```

Then verify shards:

```bash
PYTHONPATH=$PWD python scripts/pretok_hub.py verify --pretok-dir data/pretok
```

## Verification Gates

Before claiming this blend is better:

1. Run shard verification and confirm all configured sources exist under `data/pretok`.
2. Run a short curriculum smoke and verify stage transitions plus active source names.
3. Log realized token fractions from `MultiSourceCurriculumLoader.realized_token_fractions()`.
4. Compare against the old OWT/Dolma blend at matched steps, matched effective tokens, and same validation stream.
5. Track source epoch pressure. `dharma` is intentionally tiny; if it repeats too much, drop it or grow the source.

Open edge: cross-source dedup between RefinedWeb and FineWeb/FineWeb-Edu is not implemented here. That is why broad FineWeb is available but not mixed by default.
