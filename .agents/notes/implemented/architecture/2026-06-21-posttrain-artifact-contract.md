# Agent Note: Posttrain Artifact Contract

Status: implemented

Origin: Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/POSTTRAIN_ARTIFACT_CONTRACT.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Morpheus Post-Training Artifact Contract

Status: MORPH consumer mirror.

Source-of-truth contract:

`/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset/specs/004-morpheus-posttrain-artifacts/contracts/POSTTRAIN_ARTIFACT_CONTRACT.md`

RPDG task tracker:

`/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset/specs/004-morpheus-posttrain-artifacts/tasks.md`

## Boundary

RPDG produces validated artifacts:

- `morpheus_sft_v1.jsonl`
- `morpheus_rl_prompts_v1.jsonl`
- optional `morpheus_preferences_v1.jsonl`
- `manifest.json`
- `validation_report.md`
- optional `tokenized/` shards if tokenizer hash and special tokens match MORPH exactly

MORPH consumes those artifacts and owns:

- tokenizer and special-token registration
- embedding resize and checkpoint compatibility
- chat template and EOT behavior
- canonical renderer/parser in the trainer
- token-level loss-mask construction or audit
- SFT/RL training entrypoints and Hydra/W&B logging
- model-side acceptance gates before training
- runtime action/observation/RSA semantics

## Required Consumer Checks

MORPH must reject an artifact before training if:

- contract version is unsupported
- tokenizer or special-token set differs from the MORPH checkpoint contract
- EOT token differs from MORPH chat-template expectations
- required artifact files are missing
- any selected training row fails MORPH parsing
- target spans include observations, tips, candidate tails, user/system text, or verifier outputs
- completed answer rows are missing `<final>`
- math rows that require a boxed answer lack `\boxed{...}` inside `<final>`
- action and final appear in the same assistant turn before an observation
- validation report hard gates fail

## Notes

This file is a consumer mirror, not the source-of-truth. Update the RPDG contract first, then update
this mirror and the Morpheus SFT/RL spec if the contract changes.
