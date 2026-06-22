#!/usr/bin/env python
"""Validate a Morpheus post-training artifact directory before SFT/RL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morph.posttrain import accept_artifact_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Directory containing Morpheus JSONL files")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report JSON")
    args = parser.parse_args(argv)

    report = accept_artifact_dir(args.artifact_dir)
    payload = {
        "artifact_dir": str(report.artifact_dir),
        "ok": report.ok,
        "sft_rows": report.sft_rows,
        "rl_prompt_rows": report.rl_prompt_rows,
        "errors": report.errors,
        "manifest_contract_version": report.manifest.get("contract_version"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"[{status}] {report.artifact_dir}")
        print(f"  SFT rows: {report.sft_rows}")
        print(f"  RL prompt rows: {report.rl_prompt_rows}")
        if report.errors:
            print("  Errors:")
            for error in report.errors:
                print(f"    - {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
