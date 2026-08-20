# Agent Note: Pretok Runbook

Status: implemented

Origin: Ai-notes/06-12-2026/MORPH-Curriculum-Pretraining/PRETOK_RUNBOOK.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Pretokenize → HF → Train — Runbook

Status (2026-06-13): pretokenizer + hub script BUILT + GATED, multi-source w/ concurrent local∥remote
overlap. Full run HELD for Wolfe's "go". Decisions locked: **full OWT**, Dolma3+Nemotron **~5B tok each**,
repo **`bigwolfe/morph-pretok` (private)**. ⚠ **Nemotron access lapsed — re-request** at
huggingface.co/datasets/nvidia/Nemotron-CC-v2 (logged in as bigwolfe) before the full run includes it.

Run from repo root with `PYTHONPATH=$PWD /home/wolfe/.venv/bin/python`.

## Sources (data/pretok/<name>/)
Nemotron-CC-v2 is GATED/no-access → DROPPED. Its "strong synthetic data" role is replaced by a
sample-read mix (nemotron_qa + reasoning + math). All remote sources ungated, ~5B tok budget each.
| name | role | kind | locality | repo / notes |
|---|---|---|---|---|
| owt | pretrain_bulk | arrow | LOCAL | OpenWebText, ~10.6B tok, file-sharded mp8 |
| code | pretrain_bulk | hf | stream | code_search_net (cached) |
| dharma, books | pretrain_bulk | jsonl | LOCAL | small domain text + books |
| dolma | pretrain_bulk | hf_stream | REMOTE | `allenai/dolma3_mix-5.5T-1125` — web; adult_content shards EXCLUDED + shard-shuffled |
| nemotron_qa | reasoning_midtrain | hf_stream | REMOTE | `fineinstructions-pretraining/nemotron_qa_1T` — grounded synthetic QA (Nemotron-CC replacement); **`.shuffle` REQUIRED** (source-grouped) |
| reasoning | reasoning_midtrain | hf_stream | REMOTE | `reasoning-core/procedural-pretraining-pile` cfg default — verified-correct CoT; text = `prompt`+`answer` concat |
| math | pretrain_bulk | hf_stream | REMOTE | `aslawliet/math-pretraining-corpus` — proof-pile2/cosmopedia/dm_math/etc; **`.shuffle` REQUIRED** (front-loads AMPS) |

Role reconciliation: `reasoning_midtrain` is broad next-token LM substrate. It is not the curated
SFT/RL gold split. Curated local synthesis paths (`commentary`, `dharma_text_with_reasoning`,
`cross_tradition`, `reimagined`) remain banned from pretokenization and are now re-checked by the
shared role guard during upload and runtime loading.

Rejected after sampling: `nvidia/Nemotron-Pretraining-Code-v3` (metadata-only, no text), `nvidia/Nemotron-Pretraining-Legal-v1` (narrow, redundant once broad QA in), `applied-ai-018/pretraining_v1-omega_books` (mislabeled — it's FineWeb CC web, redundant with owt/dolma).

## Curriculum blend ✅ SET in pretrain_curriculum.yaml (8 sources)
owt 0.25, dolma 0.20, nemotron_qa 0.18, code 0.10, reasoning 0.10, math 0.07, dharma 0.07, books 0.03.
Loader REQUIRES every listed source's shard present → if any source fails to tokenize, drop it here.
Ratios are sample-read judgment (synthetic qa/reasoning/math = .45/.35/.20), not ablated.

## 1. Full pretokenize — local∥remote overlap (~20-40 min, watch load)
```bash
PYTHONPATH=$PWD /home/wolfe/.venv/bin/python scripts/pretokenize.py --out data/pretok --num-proc 8
```
- owt (local mp8) runs FOREGROUND while dolma/nemotron/code stream in BACKGROUND processes — the
  network download hides under owt's CPU and fills its idle cores.
- Tune subset: `--remote-max-tokens 5e9` (default). Disable overlap: `--sequential`.
- If Nemotron 403s (gated lapse), it fails LOUD (exit 1) but all other shards are written OK; re-run
  `--only nemotron` after re-gating.

## 2. Verify integrity (hard gate)  →  3. Upload (outward, --yes)  →  4. Local proof train
```bash
PYTHONPATH=$PWD /home/wolfe/.venv/bin/python scripts/pretok_hub.py verify   --pretok-dir data/pretok
PYTHONPATH=$PWD /home/wolfe/.venv/bin/python scripts/pretok_hub.py upload   --pretok-dir data/pretok --repo bigwolfe/morph-pretok --private --yes
PYTHONPATH=$PWD /home/wolfe/.venv/bin/python -m morph.training.train --config-name pretrain_curriculum
```
TODO before step 4: fold dolma/nemotron into the curriculum `blend:` in pretrain_curriculum.yaml
(loader requires every blend source present — only include sources that actually tokenized).

## 5. RTX PRO 6000 (later, BIGGER model)
`pretok_hub.py download --repo bigwolfe/morph-pretok --dest data/pretok`, then a new larger-model
config (dims TBD, own mem_probe) + bigger batch + 32K. Shards are model-size-agnostic.

## Measured perf (9950X3D, pyarrow-direct + vectorized writer)
| np | tok/s | busy-cores | peak RSS |  | remote stream |
|----|------:|-----------:|---------:|--|---|
| 4  | 12.21 | 58% | 10.3 GiB | | ~3.5 M tok/s/stream (network-bound) |
| **8** | **14.29** | 68% | **19.2 GiB** | | 5B ≈ 24 min/source |
| 16 | 15.14 | 77% | 36.9 GiB | | parallel streams → HF 429, DON'T |

**np=8 is the pick** (np=16 only +6% for 2× RAM). Local never saturates all cores — the remote
streams fill the rest. Probes: `ignore/sat_probe.py`, `ignore/stream_probe.py`.

## Verified gates (2026-06-13)
- GATE1 batched==per-doc tokens. GATE3 loader reads shards (NTP correct). GATE4 file-shard mp==single
  doc-set. GATE5 pyarrow==datasets doc-set (401,692/532,893,939). Concurrent test: owt∥dolma+jsonl,
  4 shards + integrity green, dolma budget-stop correct. --limit seq path + nemotron fail-loud verified.
