from __future__ import annotations

import json

import pytest

from morph.posttrain import ContractError, accept_artifact_dir, build_loss_labels, special_tokens
from morph.posttrain.validation import validate_sft_row


class CharTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        return_tensors=None,
    ):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def _base_row(target: str, *, domain: str = "math", answer: str = "\\boxed{4}") -> dict:
    prompt = "<problem>What is 2+2?</problem>\n"
    rendered = prompt + target
    return {
        "id": "row-1",
        "split": "train",
        "mode": "sft_trace",
        "kind": "reasoning",
        "domain": domain,
        "task_family": "reasoned_answer",
        "source": {"name": "fixture", "license": "test", "provenance": "unit test"},
        "tools": [],
        "context": {"problem": "What is 2+2?", "tail": "", "candidate_tails": [], "tips": []},
        "messages": [
            {"role": "user", "content": prompt, "tool_call_id": None},
            {"role": "assistant", "content": target, "tool_call_id": None},
        ],
        "rendered": rendered,
        "assistant_target": target,
        "loss_mask_policy": "assistant_generated_only_observation_and_context_masked",
        "target_spans": [{"start": len(prompt), "end": len(rendered), "kind": "assistant"}],
        "masked_spans": [{"start": 0, "end": len(prompt), "kind": "context"}],
        "answer": answer,
        "verifier": "exact_math",
        "tool_calls": [],
        "tips": [],
        "prompt_template_version": "morpheus_sft_v1",
        "tail_budget_tokens": 4096,
        "quality": {"validated": True, "validator": "fixture", "notes": ""},
    }


def test_completed_math_sft_row_requires_final_with_boxed_answer() -> None:
    row = _base_row("<think>Add the two integers.</think>\n<final>\\boxed{4}</final><|eot|>")

    validate_sft_row(row)


def test_action_first_row_is_valid_nonterminal() -> None:
    action = '<action>{"tool":"python","arguments":{"code":"print(2+2)"}}</action><|eot|>'
    row = _base_row(action, domain="code", answer="")
    row["kind"] = "tool"
    row["task_family"] = "tool_action"
    row["verifier"] = "unit_tests"

    validate_sft_row(row)


def test_observation_in_assistant_target_is_rejected() -> None:
    row = _base_row("<observation>4</observation><final>\\boxed{4}</final><|eot|>")

    with pytest.raises(ContractError, match="context-only"):
        validate_sft_row(row)


def test_action_and_final_same_turn_is_rejected() -> None:
    row = _base_row(
        '<action>{"tool":"python","arguments":{}}</action><final>done</final><|eot|>',
        domain="code",
    )

    with pytest.raises(ContractError, match="same turn"):
        validate_sft_row(row)


def test_loss_mask_keeps_only_target_span_labels() -> None:
    target = "<final>\\boxed{4}</final><|eot|>"
    row = _base_row(target)

    input_ids, labels = build_loss_labels(row["rendered"], CharTokenizer(), row["target_spans"])

    prompt_len = row["target_spans"][0]["start"]
    assert input_ids.numel() == len(row["rendered"])
    assert labels[:prompt_len].tolist() == [-100] * prompt_len
    assert labels[prompt_len:].tolist() == list(range(prompt_len, len(row["rendered"])))


def test_accept_artifact_dir_validates_required_files(tmp_path) -> None:
    sft_row = _base_row("<final>\\boxed{4}</final><|eot|>")
    rl_prompt = dict(sft_row)
    rl_prompt["id"] = "rl-1"
    rl_prompt["mode"] = "rl_prompt"
    rl_prompt.pop("assistant_target")
    rl_prompt["target_spans"] = []

    (tmp_path / "morpheus_sft_v1.jsonl").write_text(json.dumps(sft_row) + "\n", encoding="utf-8")
    (tmp_path / "morpheus_rl_prompts_v1.jsonl").write_text(
        json.dumps(rl_prompt) + "\n", encoding="utf-8"
    )
    manifest = {
        "contract_version": "morpheus_posttrain_artifact_v1",
        "eot_token": "<|eot|>",
        "special_tokens": special_tokens(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = accept_artifact_dir(tmp_path, raise_on_error=True)

    assert report.ok
    assert report.sft_rows == 1
    assert report.rl_prompt_rows == 1
