"""Validation for Morpheus SFT/RL artifact rows.

The validators are intentionally strict. Training should refuse data that asks
the model to generate runtime-only wrapper tags or mixes tool actions with final
answers in one assistant turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from morph.posttrain.constants import CONTEXT_ONLY_TAGS, EOT_TOKEN, MODEL_GENERATED_TAGS


class ContractError(ValueError):
    """Raised when an artifact row violates the Morpheus post-training contract."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "Span":
        if isinstance(obj, dict):
            return cls(int(obj["start"]), int(obj["end"]), str(obj.get("kind", "")))
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return cls(int(obj[0]), int(obj[1]), str(obj[2]) if len(obj) > 2 else "")
        raise ContractError(f"invalid span object: {obj!r}")

    def contains(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end


def _require(row: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [f for f in fields if f not in row]
    if missing:
        raise ContractError(f"row {row.get('id', '<unknown>')} missing fields: {missing}")


def _block(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return m.group(1) if m else None


def _count(text: str, token: str) -> int:
    return text.count(token)


def _validate_balanced_model_tags(text: str) -> None:
    for open_tag, close_tag in (("<think>", "</think>"), ("<action>", "</action>"), ("<final>", "</final>")):
        if _count(text, open_tag) != _count(text, close_tag):
            raise ContractError(f"unbalanced tag pair {open_tag}/{close_tag}")
        if close_tag in text and open_tag not in text:
            raise ContractError(f"closing tag {close_tag} without opener")


def _validate_no_context_tags_in_target(target: str) -> None:
    found = [tag for tag in CONTEXT_ONLY_TAGS if tag in target]
    if found:
        raise ContractError(f"assistant target contains context-only tags: {found}")


def _validate_action_json(target: str) -> None:
    action = _block(target, "action")
    if action is None:
        return
    try:
        payload = json.loads(action.strip())
    except json.JSONDecodeError as exc:
        raise ContractError(f"action block is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("action block must decode to an object")
    if not isinstance(payload.get("tool"), str) or not payload["tool"].strip():
        raise ContractError("action block missing non-empty tool")
    if "arguments" in payload and not isinstance(payload["arguments"], dict):
        raise ContractError("action arguments must be an object")


def _spans(row: dict[str, Any], key: str) -> list[Span]:
    return [Span.from_obj(s) for s in row.get(key, [])]


def _validate_spans(row: dict[str, Any]) -> None:
    rendered = str(row.get("rendered", ""))
    target_spans = _spans(row, "target_spans")
    masked_spans = _spans(row, "masked_spans")

    for span in target_spans + masked_spans:
        if span.start < 0 or span.end < span.start or span.end > len(rendered):
            raise ContractError(f"span out of range for row {row.get('id')}: {span}")

    for target in target_spans:
        for masked in masked_spans:
            if max(target.start, masked.start) < min(target.end, masked.end):
                raise ContractError(f"target span overlaps masked span in row {row.get('id')}")

    for tag in CONTEXT_ONLY_TAGS:
        pos = rendered.find(tag)
        while pos != -1:
            end = pos + len(tag)
            if not any(span.contains(pos, end) for span in masked_spans):
                raise ContractError(f"context-only tag {tag} is not covered by masked_spans")
            pos = rendered.find(tag, end)


def _is_completed_target(target: str) -> bool:
    return "<final>" in target


def _is_action_target(target: str) -> bool:
    return "<action>" in target


def _validate_target_semantics(row: dict[str, Any]) -> None:
    target = str(row.get("assistant_target", ""))
    _validate_balanced_model_tags(target)
    _validate_no_context_tags_in_target(target)
    _validate_action_json(target)

    has_action = _is_action_target(target)
    has_final = _is_completed_target(target)
    if has_action and has_final:
        raise ContractError("assistant target must not contain <action> and <final> in the same turn")

    if has_action:
        if row.get("answer") not in (None, ""):
            raise ContractError("nonterminal action row must not carry a final answer")
        return

    if not has_final:
        raise ContractError("completed assistant target must contain <final>")

    if str(row.get("domain")) == "math":
        final = _block(target, "final") or ""
        if "\\boxed{" not in final and "boxed{" not in final:
            raise ContractError("math final must contain a boxed answer inside <final>")


def validate_sft_row(row: dict[str, Any]) -> None:
    """Validate one `morpheus_sft_v1.jsonl` row."""
    _require(
        row,
        (
            "id",
            "split",
            "mode",
            "kind",
            "domain",
            "task_family",
            "source",
            "context",
            "messages",
            "rendered",
            "assistant_target",
            "loss_mask_policy",
            "target_spans",
            "masked_spans",
            "tail_budget_tokens",
        ),
    )
    if row["mode"] != "sft_trace":
        raise ContractError(f"SFT row mode must be sft_trace, got {row['mode']!r}")
    if int(row["tail_budget_tokens"]) != 4096:
        raise ContractError("tail_budget_tokens must be 4096")
    if row.get("loss_mask_policy") != "assistant_generated_only_observation_and_context_masked":
        raise ContractError("unsupported loss_mask_policy")
    if str(row.get("assistant_target", "")).count(EOT_TOKEN) > 1:
        raise ContractError("assistant target contains multiple EOT tokens")

    _validate_target_semantics(row)
    _validate_spans(row)


def validate_rl_prompt_row(row: dict[str, Any]) -> None:
    """Validate one `morpheus_rl_prompts_v1.jsonl` row."""
    _require(
        row,
        (
            "id",
            "split",
            "mode",
            "kind",
            "domain",
            "task_family",
            "source",
            "context",
            "messages",
            "rendered",
            "loss_mask_policy",
            "masked_spans",
            "tail_budget_tokens",
            "verifier",
        ),
    )
    if row["mode"] != "rl_prompt":
        raise ContractError(f"RL prompt row mode must be rl_prompt, got {row['mode']!r}")
    if "assistant_target" in row and row["assistant_target"]:
        raise ContractError("RL prompt row must not include a supervised assistant_target")
    if int(row["tail_budget_tokens"]) != 4096:
        raise ContractError("tail_budget_tokens must be 4096")
    _validate_spans(row)
