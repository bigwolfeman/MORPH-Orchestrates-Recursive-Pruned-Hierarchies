# Agent Note: Morpheus Sft Rl Audit

Status: implemented

Origin: Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/AUDIT.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Worker C Audit: Morpheus Post-Training Buildout

Timestamp: 2026-06-21.

Scope: independent audit of the Morpheus post-training artifact contract, task lists, and current
producer/consumer implementation state.

## Current Verdict

End-to-end status is blocked at implementation availability. The contract and task lists define the
right hard gates, but the current filesystem snapshot does not contain runnable RPDG Morpheus
producer code, MORPH consumer code, or tests for the post-training artifact path.

This is not an end-to-end pass.

## Minimum Non-Theater Acceptance Gates

These are the smallest gates needed before claiming the post-training handoff works:

1. RPDG artifact export gate:
   - produce `morpheus_sft_v1.jsonl`
   - produce `morpheus_rl_prompts_v1.jsonl`
   - produce `manifest.json`
   - produce `validation_report.md`
   - include at least one tiny fixture row for direct final, reasoned final, math boxed final,
     action-first, reason-then-action, post-observation final, aggregation, and continuation

2. RPDG validation gate:
   - reject missing `<final>` on completed answers
   - reject assistant-generated context-only tags
   - reject malformed tags
   - reject invalid `<action>` JSON
   - reject assistant text after `</action>` before an observation
   - reject same-turn `<action>` plus `<final>` before observation
   - reject text after EOT
   - reject math rows missing `\boxed{...}` inside `<final>`
   - prove observations, tips, tails, and candidate tails are masked

3. MORPH acceptance gate:
   - read the RPDG manifest
   - verify contract version, EOT, tokenizer, and special-token set
   - parse selected artifact rows with the MORPH parser
   - rebuild or audit token masks inside MORPH
   - reject the artifact if hard gates fail

4. Token-mask proof gate:
   - a valid post-observation row has `<observation>...</observation>` masked
   - assistant `<think>`, `<action>`, `<final>`, and assistant EOT are trainable
   - a deliberately bad row that targets `<observation>` fails

5. Invalid-row rejection gate:
   - missing completed `<final>` fails
   - action-plus-final in the same assistant turn fails
   - assistant-generated `<candidate_tails>`, `<tail>`, `<tips>`, `<tip>`, `<problem>`, or
     `<observation>` fails
   - invalid action JSON fails
   - math final without boxed answer fails when the row is math-verifiable

## Contract Consistency Check

No contradiction found in the core design decisions:

- Structural tags are XML-formatted strings and intended to be tokenizer special tokens where
  MORPH supports them.
- Action tags are model-generated but outside `<think>`.
- Observation, problem, candidate-tail, tail, tips, and tip tags are context-only.
- Completed answers require `<final>`.
- Math rows require `\boxed{...}` inside `<final>`.
- Candidate tails are harness input, not model output.
- Tips and observations are context-only and loss-masked.
- Trivial aggregation may emit `<final>` directly.

One contract friction point to preserve as a gate: the spec examples use `<tail id="1">`, while the
RPDG source contract says dynamic strings such as `<tail id="1">` must not become separate special
tokens. This is acceptable only if the parser treats `<tail ...>` as an input wrapper form while
special-token registration includes only `<tail>` / `</tail>`.

## Implementation Snapshot

Observed RPDG files for this scope:

- `specs/004-morpheus-posttrain-artifacts/contracts/POSTTRAIN_ARTIFACT_CONTRACT.md`
- `specs/004-morpheus-posttrain-artifacts/tasks.md`

Observed MORPH files for this scope:

- `Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/SPEC.md`
- `Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/TASKS.md`
- `Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/POSTTRAIN_ARTIFACT_CONTRACT.md`
- `morph/posttrain/` exists but contains no files in the inspected snapshot.

Current blockers:

- RPDG task list still marks schema, validators, mask builders, exporters, hard gates, and fixture
  export as incomplete.
- MORPH task list still marks renderer/parser, validators, token masks, artifact acceptance gate,
  SFT loader, RL prompt loader, and runtime semantics as incomplete.
- No Morpheus-specific tests were found under either repo's `tests/` tree.

## Commands Run

From MORPH:

```bash
rg --files -g 'AGENTS.md' -g 'AUDIT.md' -g 'SPEC.md' -g 'TASKS.md' \
  -g 'POSTTRAIN_ARTIFACT_CONTRACT.md' -g '*morpheus*' -g '*posttrain*' \
  /mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies \
  /mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset
git status --short
find Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt -maxdepth 2 -type f -print
rg -n "Morpheus|morpheus|POSTTRAIN|posttrain|candidate_tails|loss_mask|target_spans|masked_spans|<observation>|<action>|<final>|boxed|\\boxed|<tips>|<tip>|<tail>|special.token|special_tokens|eot" morph scripts tests Ai-notes -S
find morph/posttrain -maxdepth 4 -type f -print
find tests -maxdepth 4 -type f -print | sort | rg 'posttrain|morpheus|artifact|sft|mask'
vlt thread read morph-prepost-reconcile
```

From RPDG:

```bash
git -c safe.directory=/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset status --short
find specs/004-morpheus-posttrain-artifacts -maxdepth 4 -type f -print
rg -n "Morpheus|morpheus|POSTTRAIN|posttrain|candidate_tails|loss_mask|target_spans|masked_spans|<observation>|<action>|<final>|boxed|\\boxed|<tips>|<tip>|<tail>|special.token|special_tokens|eot" rpdg scripts tests specs -S
find . -maxdepth 4 -type f \( -iname '*morpheus*' -o -iname '*posttrain*' -o -path './rpdg/morpheus/*' -o -path './tests/*morpheus*' \) -print
find tests -maxdepth 4 -type f -print | sort | rg 'morpheus|posttrain|artifact|mask'
```

## Unverified Edges

- I did not run producer or consumer tests because no Morpheus-specific producer/consumer test
  files were present in this snapshot.
- I did not run an artifact export because no Morpheus artifact exporter CLI or generator was found.
- I did not run MORPH acceptance because no artifact acceptance implementation was found.
- I did not inspect Worker A or Worker B output beyond filesystem/vlt evidence; if they are still
  running, this audit should be repeated after their writes land.
