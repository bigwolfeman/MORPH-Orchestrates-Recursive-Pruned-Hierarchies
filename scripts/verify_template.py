#!/usr/bin/env python3
"""Gate Agent Note layout under .agents/notes/.

Imported notes (those with an Origin: Ai-notes/ line) only need a valid
path, # Agent Note: title, and a Status: line that matches the folder.
New notes must satisfy the full skeleton in .agents/notes/README.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTE_ROOT = ROOT / ".agents" / "notes"

NOTE_CLASSES = frozenset(
    {"architecture", "bug-fix", "feature", "process", "simplification", "testing"}
)
NOTE_LIFECYCLES = frozenset({"proposed", "implemented", "rejected", "archived"})
NOTE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
SKIP_NAMES = frozenset({"README.md", "AGENTS.md", ".gitkeep"})

PROPOSED_SECTIONS = (
    "## Problem",
    "## Proposal",
    "## Alternatives considered",
    "## Acceptance criteria",
    "## Risks",
)
IMPLEMENTED_SECTIONS = (
    "## Problem",
    "## Decision",
    "## Alternatives considered",
    "## Consequences",
)
IMPLEMENTED_FORBIDDEN = (
    "## Proposal",
    "## Plan",
    "## Migration plan",
    "## Acceptance criteria",
)
REJECTED_SECTIONS = ("## Problem", "## Proposal", "## Alternatives considered")
IMPORTED_MARKER = "Origin: Ai-notes/"


class Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, msg: str) -> None:
        self.items.append(msg)


def heading_present(text: str, heading: str) -> bool:
    return re.search(rf"^{re.escape(heading)}\s*$", text, re.M) is not None


def section_has_body(text: str, heading: str) -> bool:
    pattern = rf"^{re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.M)
    if not match:
        return False
    rest = text[match.end() :]
    next_h = re.search(r"^## ", rest, re.M)
    body = rest[: next_h.start()] if next_h else rest
    return bool(body.strip())


def check_required_dirs(errors: Errors) -> None:
    if not NOTE_ROOT.is_dir():
        errors.add("missing .agents/notes/")
        return
    for life in sorted(NOTE_LIFECYCLES):
        for cls in sorted(NOTE_CLASSES):
            path = NOTE_ROOT / life / cls
            if not path.is_dir():
                errors.add(f"missing note class folder: {path.relative_to(ROOT)}")


def status_ok(lifecycle: str, status_line: str, text: str) -> str | None:
    if lifecycle == "proposed" and status_line != "Status: proposed":
        return "proposed note Status must be 'Status: proposed'"
    if lifecycle == "implemented" and status_line != "Status: implemented":
        return "implemented note Status must be 'Status: implemented'"
    if lifecycle == "archived":
        if status_line != "Status: implemented":
            return "archived note Status must remain 'Status: implemented'"
        if "Archived:" not in text:
            return "archived note missing Archived: date"
    if lifecycle == "rejected" and not status_line.startswith("Status: rejected — "):
        return "rejected note Status must be 'Status: rejected — <why>'"
    return None


def check_agent_notes(errors: Errors) -> None:
    if not NOTE_ROOT.is_dir():
        return
    for markdown in NOTE_ROOT.rglob("*.md"):
        if markdown.name in SKIP_NAMES:
            continue
        rel = markdown.relative_to(ROOT)
        try:
            lifecycle, kind, name = rel.relative_to(".agents/notes").parts
        except ValueError:
            errors.add(f"note not under lifecycle/class: {rel}")
            continue
        if lifecycle not in NOTE_LIFECYCLES:
            errors.add(f"unknown note lifecycle: {rel}")
            continue
        if kind not in NOTE_CLASSES:
            errors.add(f"unknown note class: {rel}")
            continue
        if not NOTE_NAME.match(name):
            errors.add(f"note filename must be yyyy-mm-dd-slug.md: {rel}")
        text = markdown.read_text(encoding="utf-8")
        if not text.startswith("# Agent Note: "):
            errors.add(f"note must start with '# Agent Note: ': {rel}")
        status_line = ""
        for line in text.splitlines()[:12]:
            if line.startswith("Status:"):
                status_line = line
                break
        if not status_line:
            errors.add(f"missing Status: line: {rel}")
            continue
        problem = status_ok(lifecycle, status_line, text)
        if problem:
            errors.add(f"{problem}: {rel}")
        if IMPORTED_MARKER in text:
            continue
        if lifecycle == "proposed":
            required, forbidden = PROPOSED_SECTIONS, ()
        elif lifecycle in ("implemented", "archived"):
            required, forbidden = IMPLEMENTED_SECTIONS, IMPLEMENTED_FORBIDDEN
        else:
            required, forbidden = REJECTED_SECTIONS, ()
        for heading in required:
            if not heading_present(text, heading):
                errors.add(f"{rel} missing {heading}")
            elif not section_has_body(text, heading):
                errors.add(f"{rel} empty {heading}")
        for heading in forbidden:
            if heading_present(text, heading):
                errors.add(f"{rel} implemented/archived notes must not contain {heading}")


def main() -> int:
    errors = Errors()
    check_required_dirs(errors)
    check_agent_notes(errors)
    if errors.items:
        print("verify_template.py failed:", file=sys.stderr)
        for item in errors.items:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("verify_template.py: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
