"""Shared dataset role guards for MORPH training data.

The pretraining and post-training pipelines intentionally share some broad
reasoning-shaped text, but they must not silently mix in curated post-train gold.
Keeping the role rules here prevents the pretokenizer, shard verifier, and runtime
loader from drifting apart.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

ROLE_PRETRAIN_BULK = "pretrain_bulk"
ROLE_REASONING_MIDTRAIN = "reasoning_midtrain"
ROLE_POSTTRAIN_GOLD = "posttrain_gold"

DEFAULT_ALLOWED_PRETRAIN_ROLES = (ROLE_PRETRAIN_BULK, ROLE_REASONING_MIDTRAIN)

DENY_PATH_RE = re.compile(r"(commentary|dharma_text_with_reasoning|cross_tradition|reimagined)", re.I)

KNOWN_SOURCE_ROLES = {
    "owt": ROLE_PRETRAIN_BULK,
    "dolma": ROLE_PRETRAIN_BULK,
    "code": ROLE_PRETRAIN_BULK,
    "math": ROLE_PRETRAIN_BULK,
    "dharma": ROLE_PRETRAIN_BULK,
    "books": ROLE_PRETRAIN_BULK,
    "nemotron_qa": ROLE_REASONING_MIDTRAIN,
    "reasoning": ROLE_REASONING_MIDTRAIN,
}

VALID_SOURCE_ROLES = {
    ROLE_PRETRAIN_BULK,
    ROLE_REASONING_MIDTRAIN,
    ROLE_POSTTRAIN_GOLD,
}


def flatten_paths(paths: Any) -> list[str]:
    """Return string paths/ids from nested metadata specs."""
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths]
    if isinstance(paths, dict):
        out: list[str] = []
        for v in paths.values():
            out.extend(flatten_paths(v))
        return out
    if isinstance(paths, Iterable):
        out = []
        for v in paths:
            out.extend(flatten_paths(v))
        return out
    return []


def reject_denied_paths(paths: Any, *, context: str) -> None:
    """Fail on local curated reasoning-gold path names."""
    for p in flatten_paths(paths):
        if DENY_PATH_RE.search(os.path.basename(p)):
            raise RuntimeError(f"{context}: post-training gold path is not pretraining data: {p}")


def infer_source_role(name: str, explicit_role: str | None = None) -> str:
    """Resolve a source role, using source-name inference for legacy shards."""
    if explicit_role is not None:
        role = str(explicit_role).strip()
        if role not in VALID_SOURCE_ROLES:
            raise ValueError(f"unknown source role {role!r} for {name!r}")
        return role
    return KNOWN_SOURCE_ROLES.get(str(name), ROLE_POSTTRAIN_GOLD)


def validate_source_for_pretraining(
    name: str,
    *,
    explicit_role: str | None = None,
    allowed_roles: Iterable[str] = DEFAULT_ALLOWED_PRETRAIN_ROLES,
    paths: Any = None,
) -> str:
    """Validate a source before it enters a pretraining/midtraining blend."""
    reject_denied_paths(paths, context=f"[{name}]")
    role = infer_source_role(name, explicit_role)
    allowed = {str(r) for r in allowed_roles}
    if role not in allowed:
        raise RuntimeError(
            f"[{name}] source role {role!r} is not allowed in this training blend "
            f"(allowed={sorted(allowed)})."
        )
    return role
