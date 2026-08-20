# Agent Note: Morpheus Sft Rl Tasks

Status: proposed

Origin: Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/TASKS.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Tasks: Morpheus Post-Training Consumer

**Input specs**:

- `SPEC.md`
- `POSTTRAIN_ARTIFACT_CONTRACT.md`
- RPDG source contract:
  `/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset/specs/004-morpheus-posttrain-artifacts/contracts/POSTTRAIN_ARTIFACT_CONTRACT.md`

**Goal**: Train Morpheus on validated RPDG artifacts without letting the data project define model
semantics silently.

## Format

- `[ ]` not started
- `[~]` in progress
- `[X]` completed
- `[P]` can run in parallel

## Phase 1: Contract And Tokenizer Decisions

- [X] M001 Write Morpheus SFT/RL prompt/data spec in `SPEC.md`
- [X] M002 Mirror RPDG artifact contract in `POSTTRAIN_ARTIFACT_CONTRACT.md`
- [ ] M003 Verify the inherited tokenizer/chat template and identify native EOT, if any
- [~] M004 Decide exact special-token set and whether context-only tags are decode-blocked
- [ ] M005 Implement special-token registration and embedding resize plan
- [ ] M006 Add checkpoint metadata for contract version, tokenizer hash, special-token list, and EOT token

## Phase 2: Renderer, Parser, And Validators

- [X] M007 Create MORPH-side Morpheus post-train package location
- [ ] M008 Implement canonical renderer for RPDG row schema
- [~] M009 Implement parser for `<think>`, `<action>`, `<final>`, runtime-only tags, and EOT
- [X] M010 [P] Implement validator for mandatory `<final>` on completed rows
- [X] M011 [P] Implement validator for forbidden generated tags
- [X] M012 [P] Implement validator for action-turn nonterminal behavior
- [X] M013 [P] Implement validator for action JSON payloads
- [X] M014 [P] Implement math boxed-final validator
- [~] M015 Add unit tests with valid direct, reasoning, action-first, reason-action, post-observation, aggregation, and invalid rows

## Phase 3: Token Masks And Collator

- [X] M016 Implement char-span to token-span conversion using the final tokenizer
- [X] M017 Implement label mask builder with `-100` on system/user/context/observation/tips/tails/verifier spans
- [ ] M018 Implement SFT collator for rendered RPDG rows
- [~] M019 Add tests proving observations are masked and assistant `<think>`, `<action>`, `<final>`, and EOT are trainable
- [ ] M020 Add a smoke fixture that fails if the model is trained to generate `<observation>`

## Phase 4: Artifact Acceptance Gate

- [X] M021 Implement artifact manifest reader
- [~] M022 Validate artifact contract version, tokenizer, EOT, and special-token list
- [X] M023 Re-run MORPH parser/validator on selected artifact rows
- [~] M024 Rebuild or audit token masks inside MORPH
- [X] M025 Produce acceptance report before training starts
- [X] M026 Reject artifact if hard gates fail

## Phase 5: SFT Training Entry Point

- [ ] M027 Add Hydra config for Morpheus SFT
- [ ] M028 Add SFT dataset loader for `morpheus_sft_v1.jsonl` or accepted tokenized shards
- [ ] M029 Add full-weight SFT training loop or adapt existing trainer without breaking LM pretraining
- [ ] M030 Log full resolved Hydra config to W&B, including tokenizer/special-token metadata
- [ ] M031 Add checkpoint save/resume with tokenizer metadata
- [ ] M032 Run tiny CPU/GPU smoke on RPDG fixture artifact

## Phase 6: RL Prompt Consumer

- [ ] M033 Add loader for `morpheus_rl_prompts_v1.jsonl`
- [ ] M034 Implement rollout renderer for direct, tool, aggregation, and continuation prompt families
- [ ] M035 Implement parser for sampled model outputs
- [ ] M036 Add invalid-output rejection or penalty policy for malformed tags
- [ ] M037 Wire verifier/reward metadata from artifact rows to RL runtime
- [ ] M038 Add small RL prompt smoke without claiming training quality

## Phase 7: Runtime Semantics

- [ ] M039 Implement action-turn stop behavior: after `<action>` and EOT, wait for runtime observation
- [ ] M040 Implement runtime observation insertion as context-only text
- [ ] M041 Implement candidate-tail aggregation prompt rendering
- [ ] M042 Implement loading-screen tip sampler in runtime prompt construction
- [~] M043 Ensure `<tip>`, `<observation>`, `<tail>`, and `<candidate_tails>` are never treated as model target text

## Phase 8: Integration With RPDG

- [X] M044 Consume RPDG tiny fixture artifact
- [ ] M045 Compare MORPH validation report against RPDG validation report
- [ ] M046 File contract drift issues back to RPDG tasks when reports disagree
- [ ] M047 Promote fixture to pilot artifact only after acceptance smoke passes

## Not In MORPH Scope

- teacher rollout generation
- data deduplication and contamination checks
- candidate-tail population construction
- verifier implementation details, except runtime adapters
- provider orchestration and generation budgets

Those tasks belong in RPDG.
