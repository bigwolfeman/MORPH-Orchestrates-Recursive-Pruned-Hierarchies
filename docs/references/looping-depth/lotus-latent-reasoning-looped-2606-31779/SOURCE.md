# Source

- Repo: https://github.com/yingfan-bot/lotus (MIT; branch main; fetched 2026-10-20 via `ghgrab agent tree` and raw.githubusercontent.com). Code for the paper "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (Fan, Svete, Lee), arXiv 2606.31779 v2 (2026-07-13).
- Paper PDF: https://arxiv.org/pdf/2606.31779 (paper.pdf, paper.txt via `pdftotext -layout`).
- Cached repo files: README.md, lotus.py (scripts/lotus.py, 1620 lines, the model wrapper), lotus.yaml (args/base/lotus.yaml).
- Repo file list (key): scripts/{run.py, lotus.py, eval.py, dataset.py, utils.py}; args/{gsm8k_cot_*.yaml, gsm8k_lotus_{gpt2,llama1b,llama3b}.yaml, base/*}; preprocessing/gsm_icot.*; launch_train.sh. No data shipped; checkpoints on Hugging Face (yingfanbot/gsm-lotus-llama3b).
