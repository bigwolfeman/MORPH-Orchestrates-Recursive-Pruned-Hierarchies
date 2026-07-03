# Known-Good Runs

Default local recipe and environment assumptions. Full loss curves and optlogs
live under `ignore/` / W&B; component decisions are listed in
[ablation-ledger.md](ablation-ledger.md).

## Canonical local recipe

| Item | Value |
| --- | --- |
| Config | `morph/configs/base.yaml` (Hydra default) |
| Entry | `python -m morph.training.train` |
| Model | `3 + 6×T + 3`, `d_model=768`, `d_ff=2048`, seq 4096 |
| Loop | Poisson depth mean 6 / max 8, `bptt_depth=4` |
| Steps | 100000 |
| LR | flat `1e-4` (`warmup=0`, `min_lr=lr`) |
| Seed | `training.seed` (default `0`) |
| Kernels | `model.use_kernels: true`, `hc_use_kernel: true` |
| Compile | `training.compile: true`, `compile_mode: default` |
| Phases | prune → carve@29000 → route@30000; TST bag=6 for first 30k steps |
| Quant | ternary backbone, int6 embeddings, 8-bit AdEMAMix (`ademamix_b1zero`) |

Phase coupling and gotchas: [runtime-invariants.md](../runtime-invariants.md).

## Environment assumptions (maintainer-validated class)

Public pins are intentionally loose in `pyproject.toml` (`torch>=2.1`,
`triton>=2.1`). Faithful reproduction of private campaigns used approximately:

| Component | Maintainer class (not a lockfile) |
| --- | --- |
| GPU | NVIDIA RTX 5090 (SM120), 32 GB — primary local target |
| Also expected | RTX 4090-class if fragmentation is controlled; reduce seq/batch if OOM |
| OS | Linux x86_64 |
| Python | ≥3.10 |
| Extras | `pip install -e ".[train]"` for `bitsandbytes` (required by default 8-bit opt) |
| Data | Local OpenWebText **Arrow** shards (datasets≥4 drops script loaders) |

There is no committed lockfile. When comparing runs, record
`torch.__version__`, CUDA driver, GPU name, and `git rev-parse HEAD`.

## Data path

`base.yaml` defaults to a **portable** Hugging Face cache glob:

```yaml
data.dataset: ~/.cache/huggingface/datasets/openwebtext/**/openwebtext-train-*.arrow
```

Override if your shards live elsewhere:

```bash
python -m morph.training.train \
  'data.dataset=/path/to/openwebtext/**/openwebtext-train-*.arrow'
```

Under `datasets>=4`, script ids like `Skylion007/openwebtext` do not load. Use
Arrow/Parquet only — see `morph/training/data.py`.

## W&B

Defaults:

```yaml
wandb.project: morph
wandb.entity: null   # use the logged-in account’s default entity
```

Override entity/project as needed. Set `WANDB_MODE=disabled` for offline smoke.

## Minimal smoke (public CI-shaped)

Does **not** replace a 100k run. Confirms install + lifecycle contracts:

```bash
pip install -e ".[dev,train]"
pytest tests/ -q
```

Lifecycle modules: `tests/test_lifecycle_*.py` (checkpoint step semantics,
prune→carve→route flags, kernel-mode flag, TST/QAT wiring, decode/KV smoke).

## Bit-exact A/B (optional, private-style)

```bash
MORPH_EXACT_TRACE=/tmp/a.trace python -m morph.training.train training.steps=20 ...
MORPH_EXACT_TRACE=/tmp/b.trace python -m morph.training.train training.steps=20 ...
# byte-identical traces ⇒ bit-identical losses for that window
```

Requires identical seed, data placement (`data_runtime.prefetch_batches=0` for
strictest data determinism), and no diagnostic hooks that alter math.

## Resume contract

Full resume (“like nothing happened”):

```bash
python -m morph.training.train training.resume=/path/to/step_N.pt
```

Restores model, carve/route topology, optimizer (unless
`training.resume_fresh_optimizer=true`), scaler, RNG, and `next_step`. See
`save_checkpoint` / `load_checkpoint` in `morph/training/train.py` and
`tests/test_lifecycle_checkpoint.py`.

## Not in git

Longer campaign logs, optlogs, profiler traces, checkpoints, and gate scripts
live under gitignored `ignore/` (and W&B). Lifecycle wiring checks are in
`tests/test_lifecycle_*.py`.
