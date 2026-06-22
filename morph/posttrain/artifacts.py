"""Artifact-directory acceptance gate for Morpheus post-training data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morph.posttrain.constants import EOT_TOKEN, special_tokens
from morph.posttrain.validation import ContractError, validate_rl_prompt_row, validate_sft_row


@dataclass
class AcceptanceReport:
    artifact_dir: Path
    sft_rows: int = 0
    rl_prompt_rows: int = 0
    errors: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(self.errors[:20])
            raise ContractError(f"Morpheus artifact acceptance failed:\n{joined}")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ContractError(f"{path.name}:{lineno}: row must be a JSON object")
            obj["_line"] = lineno
            rows.append(obj)
    return rows


def accept_artifact_dir(path: str | Path, *, raise_on_error: bool = False) -> AcceptanceReport:
    """Validate the required files in a Morpheus post-training artifact directory."""
    root = Path(path)
    report = AcceptanceReport(artifact_dir=root)
    manifest_path = root / "manifest.json"
    sft_path = root / "morpheus_sft_v1.jsonl"
    rl_path = root / "morpheus_rl_prompts_v1.jsonl"

    for required in (manifest_path, sft_path, rl_path):
        if not required.exists():
            report.errors.append(f"missing required artifact file: {required}")
    if report.errors:
        if raise_on_error:
            report.raise_for_errors()
        return report

    try:
        report.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.errors.append(f"manifest.json invalid JSON: {exc}")
        if raise_on_error:
            report.raise_for_errors()
        return report

    if report.manifest.get("contract_version") != "morpheus_posttrain_artifact_v1":
        report.errors.append("manifest contract_version must be morpheus_posttrain_artifact_v1")
    if report.manifest.get("eot_token") != EOT_TOKEN:
        report.errors.append(f"manifest eot_token must be {EOT_TOKEN!r}")
    manifest_tokens = set(report.manifest.get("special_tokens", []))
    missing_tokens = [tok for tok in special_tokens() if tok not in manifest_tokens]
    if missing_tokens:
        report.errors.append(f"manifest missing required special_tokens: {missing_tokens}")

    for row in _iter_jsonl(sft_path):
        try:
            validate_sft_row(row)
            report.sft_rows += 1
        except ContractError as exc:
            report.errors.append(f"{sft_path.name}:{row.get('_line')}: {exc}")

    for row in _iter_jsonl(rl_path):
        try:
            validate_rl_prompt_row(row)
            report.rl_prompt_rows += 1
        except ContractError as exc:
            report.errors.append(f"{rl_path.name}:{row.get('_line')}: {exc}")

    if raise_on_error:
        report.raise_for_errors()
    return report
