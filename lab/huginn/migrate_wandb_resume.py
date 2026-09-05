"""One exact logging-only checkpoint migration. Never relax evaluator source checks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = "lab/huginn/huginn_depth_sweep.py"
OLD_COMMIT = "63fe1df"
OLD_SHA256 = "b551e79f3330f0dbdd3811f68867520c6a7e354d3bb1b3986224203a803ade21"
EDITS = (
    ("def execute(cfg):\n",
     'def wandb_model_config(model_cfg):\n'
     '    """Match the JSON key types restored by W&B, without allowing value changes."""\n'
     '    return json.loads(json.dumps(model_cfg.to_dict(), allow_nan=False))\n\n\n'
     'def execute(cfg):\n'),
    ('"effective_model_config": model_cfg.to_dict()',
     '"effective_model_config": wandb_model_config(model_cfg),\n'
     '                           "resume_source_migrations": saved["huginn"].get("resume_migrations", []) if saved else []'),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_source(old: bytes) -> bytes:
    if sha(old) != OLD_SHA256:
        raise ValueError("historical evaluator blob differs from the approved source")
    text = old.decode()
    for before, after in EDITS:
        if text.count(before) != 1:
            raise ValueError("logging-only patch does not have a unique location")
        text = text.replace(before, after)
    return text.encode()


def migrated_payload(original: dict, old: bytes, current: bytes, sources: dict) -> dict:
    if current != expected_source(old):
        raise ValueError("current evaluator differs from the exact logging-only patch")
    prior_sources = original["huginn"]["source_hashes"]
    if prior_sources.get(EVALUATOR) != OLD_SHA256:
        raise ValueError("checkpoint does not reference the approved old evaluator")
    expected_hashes = {**prior_sources, EVALUATOR: sha(current)}
    if sources != expected_hashes:
        raise ValueError("another source file changed; migration is not permitted")
    result = copy.deepcopy(original)
    result["huginn"]["source_hashes"] = expected_hashes
    result["huginn"].setdefault("resume_migrations", []).append({
        "reason": "W&B JSON key normalization only; no evaluation math changes",
        "old_source_sha256": OLD_SHA256, "new_source_sha256": sha(current),
        "old_source_commit": OLD_COMMIT,
    })
    return result


def migrate(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    status = checkpoint.with_name("process.json")
    if status.exists():
        record = json.loads(status.read_text())
        if record.get("pid") and Path(f'/proc/{record["pid"]}').exists():
            raise RuntimeError("evaluator PID still exists; refuse to migrate a live run")
    raw = checkpoint.read_bytes()
    original = json.loads(raw)
    old = subprocess.check_output(["git", "show", f"{OLD_COMMIT}:{EVALUATOR}"], cwd=ROOT)
    sources = {p: sha((ROOT / p).read_bytes()) for p in original["huginn"]["source_hashes"]}
    updated = migrated_payload(original, old, (ROOT / EVALUATOR).read_bytes(), sources)
    audit = checkpoint.parent / "resume_migrations" / str(time.time_ns())
    audit.mkdir(parents=True)
    backup = audit / checkpoint.name
    with backup.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    updated["huginn"]["resume_migrations"][-1].update({
        "original_checkpoint_sha256": sha(raw), "backup": str(backup)})
    temporary = checkpoint.with_suffix(".migration.tmp")
    with temporary.open("x") as stream:
        json.dump(updated, stream, indent=1, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint)
    print(f"Migrated logging identity only. Original checkpoint: {backup}", flush=True)
    return backup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    migrate(parser.parse_args().checkpoint)
