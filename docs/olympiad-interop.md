# MORPH ⇄ Olympiad-AI Interop Contract

Both repos publish independently. Everything that crosses the boundary is listed here;
if it isn't listed, it must not be assumed. A copy of this document lives in Olympiad-AI
at `docs/morph-interop.md` — keep them in sync.

## Relationship

Olympiad-AI generates synthetic olympiad-math curriculum data and trains small "seed"
models on it. MORPH consumes both:

1. **Data**: olympiad math as a pretok curriculum source (seed training data AND
   anti-forgetting replay in the pretraining blend).
2. **Weights**: a graduated Olympiad seed as `training.init_from` for a MORPH run
   (weights only; fresh optimizer, step reset to 0).

Olympiad-AI in turn consumes MORPH's cached OWT pretok shard for its own
continual-pretraining / forgetting probes (`NLPCorpusSource`).

## 1. Pretok shard format (the data contract)

A *source shard* is a directory:

| file | contents |
|---|---|
| `tokens.u16.bin` | flat little-endian uint16 token ids, all docs concatenated; every doc ends with `eos_id` |
| `doc_offsets.i64.npy` | int64 `[n_docs + 1]`, `tokens[offsets[i]:offsets[i+1]]` = doc *i* (incl. trailing EOS) |
| `doc_lens.i32.npy` | int32 `[n_docs]`, `lens[i] == offsets[i+1] - offsets[i]` |
| `meta.json` | at minimum `{"eos_id": int, "role": str}` |

`role` gates use: MORPH's `MultiSourceCurriculumLoader` only accepts sources whose role
is in `curriculum.allowed_source_roles` (default `pretrain_bulk`, `reasoning_midtrain`).
Producers: MORPH `scripts/pretokenize.py` (OWT etc.), Olympiad
`scripts/export_morph_shards.py` (math pools; also writes an `eval_holdout.jsonl`
sibling that never enters a shard). Consumers: MORPH `morph/training/curriculum_data.py`,
Olympiad `src/olympiad_data/datasets/nlp_source.py`.

uint16 caps vocab at 65,535 — fine for both tokenizers below.

## 2. Tokenizer / vocab compatibility

- MORPH OWT shards: `bigcode/starcoder2-7b` ids (max id 49,151; vocab 49,152).
- Olympiad shards: same base vocab + 17 Olympiad special tokens → vocab **49,169**.
- 49,152 ⊂ 49,169, so a model built at vocab 49,169 consumes BOTH shard families with
  no remapping. The reverse is not safe: never feed an olympiad shard to a 49,152 model.
- `NLPCorpusSource(vocab_size=...)` fail-fast-checks shard ids against the model vocab.

## 3. Checkpoint contract (`init_from`)

Converted seed checkpoints are `{"model": <native MORPH state_dict>, "step": int}`.
`load_weights_only` in `morph/training/train.py` strips/canonicalizes `_orig_mod.` and
`module.` prefixes on BOTH sides (checkpoint and live model), so compiled/uncompiled
producers and consumers interoperate. Build the consuming model with the SAME quant
stack as the seed (`ternary: true, ternary_scope: backbone, embed_quant: "int6"`) so the
`parametrizations.weight.original` keys exist at load — expect a `matched N/N` line;
anything less means the model config doesn't match the seed dump.

## 4. Environment variables

| var | meaning | used by |
|---|---|---|
| `OLYMPIAD_REPO` | path to an Olympiad-AI checkout | MORPH configs that reference olympiad data/seeds (e.g. `forget_olympiad_seed768.yaml`) |
| `MORPH_REPO` | path to a MORPH checkout | Olympiad `nlp_source.py` default OWT dir (`$MORPH_REPO/data/pretok/owt`), grow_nlp configs |
| `MORPH_OWT_DIR` | direct override of the OWT shard dir | Olympiad `nlp_source.py` |
| `MORPH_DATA_PLACEMENT` / `MORPH_DATA_RAM_FRAC` / `MORPH_DATA_RAM_RESERVE_GB` / `MORPH_DATA_PROBE` / `MORPH_DATA_PREFETCH` | data-placement runtime knobs (see `docs/data-placement-design.md`) | **both repos** (shared names by design) |

Machine-local defaults baked into configs use `${oc.env:VAR,default}` so they run
unchanged on the origin machine and are overridable everywhere else.

## 5. Sync-twin module

`morph/training/data_placement.py` ⇄ `src/olympiad_data/datasets/data_placement.py`
are byte-identical below their docstrings (each repo must be self-contained; neither
imports the other). If you touch one, apply the same diff to the other. Each repo's
test suite gates its own copy (`tests/test_data_placement.py` /
`tests/datasets/test_data_placement.py`).

## 6. Cross-repo eval (the forgetting-curve flow)

Math retention of a MORPH checkpoint is measured OFFLINE by Olympiad's
`scripts/eval_math_checkpoint.py` (MORPH only logs NLP val ppl):

```bash
PYTHONPATH=<morph-repo> <olympiad>/.venv/bin/python scripts/eval_math_checkpoint.py \
  --ckpt <morph ckpt.pt> \
  --config scripts/eval_configs/<model>.py \      # exports target_config() + apply_quant()
  --eval-file data/morph_shards/<pool>/eval_holdout.jsonl \
  --limit 128 --format-mode structured --out-json out.json
```

The script imports `morph.*` from `PYTHONPATH` and adds its own `src/` itself. Eval
config modules under `scripts/eval_configs/` are the reproducibility anchors — one per
seed/model family, matching the training config's MORPHConfig + quant stack exactly.

## 7. Known coupling to keep an eye on

- TST checkpoints saved during the superposition phase are generation-broken by
  construction; never evaluate them as normal models (see Olympiad
  `Ai-notes/07-03-2026/MORPH-Forgetting-Curve/`).
- `training.steps` must equal the curriculum stage-sum in MORPH configs — TST phase
  timing is computed from `training.steps` before the curriculum overrides total steps.
- Olympiad stage-2 generator pools are not yet end-to-end seed-reproducible
  (entropy-seeded generators); a shard is the artifact of record until that lands.
