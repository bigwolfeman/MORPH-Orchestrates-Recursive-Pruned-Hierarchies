# Agent Note: Morpheus Sft Rl Spec

Status: implemented

Origin: Ai-notes/06-21-2026/Morpheus-SFT-RL-System-Prompt/SPEC.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Morpheus SFT/RL System Prompt and Data Spec

Status: draft for audit before data-generation implementation.

## Goal

Define the prompt grammar, tag ownership, masking policy, and data-generation shape for post-training
Morpheus, a visible-reasoning assistant trained by NoSaaS Labs.

This spec is for SFT/RL data construction and runtime prompts. It is not yet an implementation of
the data generator, SFT trainer, RL trainer, or Markovian RSA harness.

Artifact handoff is governed by:

- MORPH consumer mirror: `POSTTRAIN_ARTIFACT_CONTRACT.md`
- MORPH task list: `TASKS.md`
- RPDG source contract:
  `/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset/specs/004-morpheus-posttrain-artifacts/contracts/POSTTRAIN_ARTIFACT_CONTRACT.md`
- RPDG task list:
  `/mnt/BigAssDrive/00projects/00DeepNet/00Reasoning-Dataset/specs/004-morpheus-posttrain-artifacts/tasks.md`

## Subagent Work Division And Audit

Subagent A: prompt grammar and system prompt.

- Accepted: identity line, optional `<think>`, mandatory `<final>`, input-only tags, natural tail
  state, math boxed answer inside `<final>`, and invalid-output filters.
- Rejected/corrected: the proposed formal grammar allowed `<action>` and `<final>` in the same
  assistant turn. The final spec requires action turns to end after `<action>` and wait for
  `<observation>`.

Subagent B: data schema and masking.

- Accepted: JSONL row fields for split/mode/domain/task family/source/tools/context/messages/reward,
  observation masking, prompt/context masking, action-turn nonterminal behavior, action-first rows,
  direct no-think rows, and validators.
- Adjusted: kept the spec's existing `kind` field while adding `mode` and `task_family`, so future
  data generators can distinguish row format from behavioral family.

Subagent C: runtime/harness, special tokens, EOT, tips, and aggregation IO.

- Accepted: special-token preference, explicit EOT, input-only tips, sparse 0-2 tip sampling,
  candidate tails as harness input, and RL prompt families.
- Open: exact tokenizer-native EOT availability was not verified in this session.

## Settled Decisions

- Model identity: "You are Morpheus, trained by NoSaaS Labs."
- Deployment uses visible reasoning.
- Special tokens are preferred for structural tags.
- Final answers always use `<final>...</final>`.
- Math final answers put `\boxed{...}` inside `<final>`.
- Tool actions are outside `<think>`.
- The model never writes `<observation>`; observations are runtime/context only.
- Tool-action turns end after `<action>...</action>` and wait for runtime observation.
- Trivial direct answers may skip `<think>` and emit `<final>` directly.
- Trivial aggregation may skip `<think>` and emit `<final>` directly.
- Action-first examples are required so the model learns it may skip pre-action reasoning.
- Markovian/RSA tail budget target is 4K tokens.
- Tail state is natural language near the end of `<think>`, not a rigid summary schema.
- `<candidate_tails>` is harness-level input, not a model-generated tag.
- Loading-screen tips are random prompt augmentations, context-only and loss-masked.

## Tag Ownership

Model-generated tags:

- `<think>...</think>`
- `<action>...</action>`
- `<final>...</final>`

Runtime/context-only tags:

- `<observation>...</observation>`
- `<problem>...</problem>`
- `<tail>...</tail>`
- `<tips>...</tips>`
- `<tip>...</tip>`

Harness/input wrapper tags:

- `<candidate_tails>...</candidate_tails>`

The model may read harness/input tags in context, but training targets should filter examples where
the assistant generates runtime-only or harness-only tags.

## End Of Turn

Use the inherited tokenizer/chat-template EOT token if one exists. If it does not, add `<|eot|>` as
a tokenizer special token.

EOT is dialogue control. It is not a semantic replacement for `</final>` or `</action>`.

Examples:

```text
<final>Yes.</final><|eot|>
```

```text
<action>{"tool":"python","arguments":{"code":"print(2+2)"}}</action><|eot|>
```

## Draft System Prompt

```text
You are Morpheus, trained by NoSaaS Labs.

You are a helpful assistant and visible reasoning model. Use clear reasoning when it helps, answer
directly when the task is simple, and use tools when external information, exact computation, code
execution, retrieval, or environment feedback is needed.

Use <think>...</think> for visible reasoning. Use <action>...</action> for tool calls as valid JSON.
Use <final>...</final> for the user-facing answer. Every completed answer must include <final>. For
math, put the boxed answer inside <final>.

When a tool is needed, emit <action>...</action> and end the turn. The runtime supplies
<observation>...</observation>. Never generate or invent observations.

When candidate tails are provided by the harness, treat them as uncertain evidence, not votes. Reuse
correct fragments, reject contradictions, and produce one improved solution. If aggregation is
trivial, answer directly in <final>.

Your reasoning may be truncated and carried forward, so the end of <think> should naturally preserve
useful state: current answer, key support, uncertainty, and traps to avoid.
```

## Canonical Output Patterns

Assistant turn types:

- final turn: optional `<think>...</think>`, then mandatory `<final>...</final>`, then EOT.
- action turn: optional `<think>...</think>`, then mandatory `<action>...</action>`, then EOT.
- action-first turn: `<action>...</action>`, then EOT.

An action turn is nonterminal. The assistant must not emit `<final>` in the same turn after
`<action>`; the runtime must first append `<observation>...</observation>`, then the next assistant
turn may reason and finalize or call another action.

Direct answer:

```text
<final>Yes.</final><|eot|>
```

Reasoned answer:

```text
<think>
Natural visible reasoning.

Current best answer: ...
Key support: ...
Remaining uncertainty: ...
</think>
<final>...</final><|eot|>
```

Math answer:

```text
<think>
Compute and verify the result.
</think>
<final>The answer is \boxed{42}.</final><|eot|>
```

Action-first tool turn:

```text
<action>{"tool":"python","arguments":{"code":"print(2+2)"}}</action><|eot|>
```

Reasoned tool turn:

```text
<think>I need exact execution to avoid guessing.</think>
<action>{"tool":"python","arguments":{"code":"print(2+2)"}}</action><|eot|>
```

Post-observation answer:

```text
<observation>4</observation>
<think>The tool returned 4, so the computation is settled.</think>
<final>The answer is \boxed{4}.</final><|eot|>
```

## Aggregation Prompt Shape

Aggregation input is harness-rendered context:

```text
<problem>
...
</problem>
<candidate_tails>
<tail id="1">
...
</tail>
<tail id="2">
...
</tail>
</candidate_tails>
```

Assistant target:

```text
<think>
Compare candidate tails as uncertain evidence. Reuse correct fragments and reject contradictions.
End with useful natural state if more rounds may follow.
</think>
<final>...</final><|eot|>
```

Trivial aggregation target:

```text
<final>...</final><|eot|>
```

If aggregation needs more evidence, the assistant emits an action turn:

```text
<think>The tails disagree on the computed value, so I need exact execution.</think>
<action>{"tool":"python","arguments":{"code":"..."}}</action><|eot|>
```

The final answer comes only after the runtime provides an observation and the next assistant turn is
rendered.

## Loading-Screen Tips

Tips are context-only prompt augmentations. They are sampled, masked from loss, and never generated
by the assistant.

Default sampling policy:

- 60%: no tip
- 30%: one tip
- 10%: two tips
- 2-5%: one micro-example tip after the model is stable

Example tips:

```text
<tips>
<tip>Every completed answer needs <final>...</final>.</tip>
<tip>If a tool is needed, emit <action> JSON and stop.</tip>
</tips>
```

Micro-example tip:

```text
<tip>
Good tool turn:
<action>{"tool":"python","arguments":{"code":"print(2+2)"}}</action>
Then stop. The runtime will provide <observation>.
</tip>
```

## Special Token Inventory

Required structural specials:

- `<think>`
- `</think>`
- `<action>`
- `</action>`
- `<observation>`
- `</observation>`
- `<final>`
- `</final>`

Likely context/harness specials:

- `<problem>`
- `</problem>`
- `<tail>`
- `</tail>`
- `<tips>`
- `</tips>`
- `<tip>`
- `</tip>`

Turn-control special:

- inherited chat EOT token, or `<|eot|>` if missing.

Do not add dynamic forms like `<tail id="1">` as special tokens. Use `<tail>` as the special token
and render metadata as ordinary text/attributes.

## SFT Row Shape

Canonical generated JSONL row:

```json
{
  "id": "stable-id",
  "split": "train|val|test",
  "mode": "sft_trace|rl_prompt",
  "kind": "direct|reasoning|tool|aggregation|continuation",
  "domain": "math|code|general_reasoning",
  "task_family": "direct_answer|reasoned_answer|tool_action|aggregation|continuation",
  "source": {
    "name": "dataset-or-generator",
    "license": "unknown",
    "provenance": ""
  },
  "tools": [
    {
      "name": "tool-name",
      "schema": {}
    }
  ],
  "context": {
    "problem": "",
    "tail": "",
    "candidate_tails": [],
    "tips": [],
    "aggregation": {
      "tail_source": "harness_natural_4k",
      "candidate_policy": ""
    }
  },
  "messages": [
    {
      "role": "system|user|assistant|observation",
      "content": "",
      "tool_call_id": null
    }
  ],
  "rendered": "full rendered training text",
  "assistant_target": "assistant span for this turn",
  "loss_mask_policy": "assistant_generated_only_observation_and_context_masked",
  "target_spans": [],
  "masked_spans": [],
  "answer": "canonical final answer when available",
  "verifier": "exact_math|unit_tests|python|symbolic|teacher|none",
  "tool_calls": [],
  "tips": [],
  "prompt_template_version": "morpheus_sft_v1",
  "tail_budget_tokens": 4096,
  "quality": {
    "validated": false,
    "validator": null,
    "notes": ""
  }
}
```

For `rl_prompt`, `messages` contains only the prompt/runtime context available before rollout. For
`sft_trace`, `messages` contains the supervised trajectory, including assistant tool actions and
runtime observations.

## Loss Mask Policy

Trainable assistant-generated spans:

- `<think>...</think>`
- `<action>...</action>`
- `<final>...</final>`
- EOT after assistant turn

Masked context spans:

- system prompt
- user prompt
- `<problem>...</problem>`
- `<candidate_tails>...</candidate_tails>`
- `<tail>...</tail>`
- `<observation>...</observation>`
- `<tips>...</tips>`
- prior assistant turns when training single-turn targets, unless the trainer intentionally trains
  multi-turn assistant spans.

Observation masking is mandatory. A row that trains the model to generate tool observations is bad
data.

For action turns, the target span ends immediately after the action-turn EOT. Any observation and
later assistant final answer belong to later rendered context/target segments.

## Dataset Families

Minimum SFT families:

- direct concise assistant answers
- visible reasoning answers
- math reasoning with boxed final inside `<final>`
- code reasoning with tests or execution traces
- tool action-first traces
- tool reason-then-action traces
- post-observation answer traces
- aggregation over natural 4K tails
- continuation from natural tail
- malformed-trace rejection/filter examples for validators, not necessarily model training

Minimum RL prompt families:

- standard math/code/general reasoning prompts
- tool-available prompts with verifiable tool outcomes
- aggregation prompts over teacher or self tails
- continuation prompts from truncated natural tails

## Filters And Validators

Reject SFT targets if:

- missing `<final>` on a completed answer
- malformed XML/tag nesting
- assistant generates `<observation>`
- assistant generates `<problem>`, `<tail>`, `<tips>`, or `<candidate_tails>`
- `<action>` is not valid JSON
- assistant text appears after `</action>` before an observation
- `<action>` and `<final>` appear in the same assistant turn before an observation
- text appears after EOT
- math final lacks `\boxed{...}` when answer is math-verifiable
- aggregation output blindly votes without resolving contradiction when candidates disagree
- final answer disagrees with verifier
- tool result is cited that does not appear in an observation

## Open Questions

- Which base tokenizer/chat template will be inherited, and does it already provide a suitable EOT?
- Which tags should be added before midtraining vs before SFT only?
- What exact SFT mixture percentages should be used for direct/reasoning/tool/aggregation rows?
- What percentage of tool examples should be action-first?
- What percentage of reasoning examples should include natural tail-state paragraphs?
- How strict should XML validity be during RL sampling rejection?
- Should tips be used during RL from the beginning, or introduced after basic tag compliance?
- Which verifiers are available first for math/code/general reasoning data generation?
- Should canonical masks be stored as character spans only, or should finalized shards persist
  tokenizer-derived token masks?
- Should decode-time blocking hard-ban input-only tags (`<observation>`, `<tip>`, `<tail>`) or only
  reject/penalize them during training and validation?
