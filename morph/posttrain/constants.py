"""Canonical Morpheus post-training prompt constants."""

from __future__ import annotations

TAIL_BUDGET_TOKENS = 4096
EOT_TOKEN = "<|eot|>"

SYSTEM_PROMPT = (
    "You are Morpheus, trained by NoSaaS Labs.\n\n"
    "Reason visibly when useful. Use concise reasoning for hard math, code, and general "
    "reasoning tasks; answer directly for trivial tasks. Tool calls are actions, not final "
    "answers. If a tool is needed, emit an <action> block and stop the turn. After an "
    "observation is provided, continue from the observed evidence. Completed answers must "
    "include a <final> block. For math, put the boxed answer inside <final>."
)

MODEL_GENERATED_TAGS = (
    "<think>",
    "</think>",
    "<action>",
    "</action>",
    "<final>",
    "</final>",
)

CONTEXT_ONLY_TAGS = (
    "<observation>",
    "</observation>",
    "<problem>",
    "</problem>",
    "<candidate_tails>",
    "</candidate_tails>",
    "<tail>",
    "</tail>",
    "<tips>",
    "</tips>",
    "<tip>",
    "</tip>",
)

TURN_CONTROL_TOKENS = (EOT_TOKEN,)


def special_tokens() -> list[str]:
    """Return all special tokens MORPH must register before SFT/RL."""
    return [*MODEL_GENERATED_TAGS, *CONTEXT_ONLY_TAGS, *TURN_CONTROL_TOKENS]
