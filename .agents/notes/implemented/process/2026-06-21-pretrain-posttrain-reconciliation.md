# Agent Note: Pretrain Posttrain Reconciliation

Status: implemented

Origin: Ai-notes/06-21-2026/Pretrain-Posttrain-Reconciliation/RECONCILIATION.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Pretrain / Post-Train Reconciliation

Status: implementation guard + design reconciliation, not a completed post-training trainer.

## Paper-grounded boundary

ReAct supports interleaved reasoning traces, actions, and observations. It does not imply RSA roles,
bounded tails, or structured carry blocks.

RSA supports test-time candidate-population aggregation: generate candidate chains, sample subsets,
prompt the model with the query plus candidates, produce improved chains, and repeat. The RSA paper's
training contribution is aggregation-aware RL over standard prompts plus aggregation prompts.

ZAYA Markovian RSA is the relevant source for MORPH's target workflow: carry only bounded suffix
tails between aggregation rounds, train on aggregation prompts during SFT, and include expert/self
aggregation prompts during RL. ZAYA applies ordinary verifiable final-answer reward to those prompts;
it does not specify a special XML carry reward or a ROLE/TRACE_MODE state machine.

## Local repo reality

Current implemented trainer:

- `morph/training/train.py`: causal-LM pretraining/curriculum trainer.
- `morph/configs/pretrain_curriculum.yaml`: dense bf16 length curriculum over pretokenized shards.
- `scripts/pretokenize.py`: shard builder for LM curriculum sources.

Not currently implemented:

- SFT trainer for ReAct traces, aggregation prompts, or final-answer formatting.
- RL/GRPO/RLOO trainer for verifiable reasoning rewards.
- Markovian RSA inference harness.
- Observation-loss masking for tool traces.

## Reconciled data roles

Use explicit data roles rather than relying on source names:

- `pretrain_bulk`: ordinary web/code/math/domain/reference LM text.
- `reasoning_midtrain`: broad reasoning-shaped next-token LM data. This may include procedural CoT
  corpora as substrate, but it is not the curated post-training gold split.
- `posttrain_gold`: curated ReAct traces, aggregation examples, local synthesis reasoning maps,
  answer/verifier pairs, or reward data. This must not enter pretraining.

The shared guard is `morph/training/source_roles.py`. It is used by:

- `scripts/pretokenize.py`
- `scripts/pretok_hub.py`
- `morph/training/curriculum_data.py`

## Recommended stage contract

1. Dense LM curriculum / reasoning-aware midtraining:
   Use causal LM loss only. Allow `pretrain_bulk` and optional `reasoning_midtrain`. Do not train
   hard ReAct/RSA schemas here. Do not include local post-training gold.

2. Deploy-shape continuation:
   Load the dense curriculum checkpoint, then run the sparse/quant/TST deploy stack at the intended
   Markovian chunk size. This is still LM training, not post-training.

3. SFT:
   Use full-weight SFT for the final model. Include standard reasoning examples, ReAct
   Thought/Action/Observation traces with observation tokens masked from loss, and ZAYA-style
   aggregation examples built from bounded candidate tails. A light tail footer is an ablation, not
   a paper requirement.

4. RL:
   Use verifiable final-answer rewards for math/code/puzzle prompts. Include both ordinary prompts
   and Markovian RSA aggregation prompts. Track tail usefulness as an auxiliary eval/verifier metric,
   but do not claim ZAYA uses a special tail reward unless a later source proves it.

5. Inference harness:
   Runtime owns rollout count, aggregation-set size, tail length, round count, and truncation. The
   model prompt should be task-shaped, not a global ROLE enum.

## Current guard caveat

The role guard prevents accidental post-training-gold ingestion. It does not prove that the
`reasoning_midtrain` mixture improves the model. That remains an empirical ablation against a
`pretrain_bulk`-only curriculum.
