"""Verify / upload / download MORPH pretok shards to a Hugging Face dataset repo.

The pretok shards (scripts/pretokenize.py output) are MODEL-SIZE-AGNOSTIC — tokenization depends
only on the tokenizer, not on d_model/n_layers. So we tokenize once locally, push here, and BOTH
the 5090 small-scale proof and the larger RTX-PRO-6000 run pull the same shards.

Subcommands:
  verify    integrity-check every source shard (size==n_tokens*2, offsets monotone+consistent,
            lens.sum()==n_tokens, no role-split path). Prints a manifest. HARD GATE for upload.
  upload    verify → write README manifest card → create_repo(dataset) → upload_large_folder
            (resumable, multi-threaded). REQUIRES --yes (outward-facing; no accidental publish).
  download  snapshot_download the dataset repo to a local dir (run this on the 6000).

Usage:
  PYTHONPATH=$PWD python scripts/pretok_hub.py verify   --pretok-dir data/pretok
  PYTHONPATH=$PWD python scripts/pretok_hub.py upload    --pretok-dir data/pretok \
       --repo bigwolfe/morph-pretok --private --yes
  PYTHONPATH=$PWD python scripts/pretok_hub.py download  --repo bigwolfe/morph-pretok \
       --dest data/pretok
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np

DENY = re.compile(r"(commentary|reasoning|cross_tradition|reimagined|_qa)", re.I)
REQUIRED = ("tokens.u16.bin", "doc_offsets.i64.npy", "doc_lens.i32.npy", "meta.json")


def verify_source(sdir: str) -> dict:
    """Hard integrity check of one source shard. Raises on any inconsistency."""
    name = os.path.basename(sdir.rstrip("/"))
    for f in REQUIRED:
        p = os.path.join(sdir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{name}] missing {f}")
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    # role-split guard travels with the data: a reasoning-gold path must never have been ingested.
    paths = meta.get("paths")
    flat = paths if isinstance(paths, list) else [paths]
    for p in flat:
        if isinstance(p, str) and DENY.search(os.path.basename(p)):
            raise RuntimeError(f"[{name}] ROLE-SPLIT VIOLATION in meta.paths: {p}")
    offs = np.load(os.path.join(sdir, "doc_offsets.i64.npy"))
    lens = np.load(os.path.join(sdir, "doc_lens.i32.npy"))
    n_tok = int(meta["n_tokens"]); n_doc = int(meta["n_docs"])
    bin_bytes = os.path.getsize(os.path.join(sdir, "tokens.u16.bin"))
    assert meta.get("dtype") == "uint16", f"[{name}] dtype != uint16"
    assert bin_bytes == n_tok * 2, f"[{name}] blob {bin_bytes}B != n_tokens*2 ({n_tok*2})"
    assert len(offs) == n_doc + 1, f"[{name}] offsets len {len(offs)} != n_docs+1 ({n_doc+1})"
    assert len(lens) == n_doc, f"[{name}] lens len {len(lens)} != n_docs ({n_doc})"
    assert int(offs[0]) == 0 and int(offs[-1]) == n_tok, f"[{name}] offsets endpoints wrong"
    assert np.array_equal(np.diff(offs).astype(np.int64), lens.astype(np.int64)), \
        f"[{name}] offsets != cumsum(lens)"
    assert (lens > 0).all(), f"[{name}] zero-length doc present"
    return {"name": name, "n_docs": n_doc, "n_tokens": n_tok, "gib": bin_bytes / 2**30,
            "tokenizer": meta.get("tokenizer"), "eos_id": meta.get("eos_id"),
            "mean_len": n_tok / max(1, n_doc)}


def verify_all(pretok_dir: str) -> list[dict]:
    subs = sorted(d for d in os.listdir(pretok_dir)
                  if os.path.isdir(os.path.join(pretok_dir, d)))
    if not subs:
        raise RuntimeError(f"no source subdirs under {pretok_dir}")
    rows = [verify_source(os.path.join(pretok_dir, d)) for d in subs]
    tok = {r["tokenizer"] for r in rows}
    assert len(tok) == 1, f"mixed tokenizers across sources: {tok} — would corrupt training"
    print(f"{'source':10s} {'docs':>12s} {'tokens':>15s} {'GiB':>7s} {'mean_len':>9s}")
    tot_d = tot_t = tot_g = 0
    for r in rows:
        print(f"{r['name']:10s} {r['n_docs']:>12,} {r['n_tokens']:>15,} "
              f"{r['gib']:>7.2f} {r['mean_len']:>9.0f}")
        tot_d += r["n_docs"]; tot_t += r["n_tokens"]; tot_g += r["gib"]
    print(f"{'TOTAL':10s} {tot_d:>12,} {tot_t:>15,} {tot_g:>7.2f}")
    print(f"tokenizer: {tok.pop()}  (all sources consistent)")
    return rows


def _readme(rows: list[dict], repo: str) -> str:
    tot_t = sum(r["n_tokens"] for r in rows); tot_d = sum(r["n_docs"] for r in rows)
    body = [
        "---", "license: other", "task_categories: [text-generation]",
        "tags: [morph, pretraining, length-curriculum]", "---", "",
        f"# {repo} — MORPH curriculum-pretraining shards", "",
        "Pre-tokenized, length-indexed shards for MORPH curriculum pretraining "
        "(see `scripts/pretokenize.py`). Model-size-agnostic: same shards feed the 5090 proof "
        "and the RTX-PRO-6000 scaled run.", "",
        f"- **Tokenizer:** `{rows[0]['tokenizer']}` (EOS id {rows[0]['eos_id']})",
        f"- **Total:** {tot_d:,} docs, {tot_t:,} tokens", "",
        "## Per-source", "",
        "| source | docs | tokens | GiB | mean tok/doc |", "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        body.append(f"| {r['name']} | {r['n_docs']:,} | {r['n_tokens']:,} | "
                    f"{r['gib']:.2f} | {r['mean_len']:.0f} |")
    body += [
        "", "## Shard format (per source dir)", "",
        "- `tokens.u16.bin` — uint16 token ids, all docs concatenated, each doc ends with EOS.",
        "- `doc_offsets.i64.npy` — `[n_docs+1]` start-token index per doc (`[-1] == n_tokens`).",
        "- `doc_lens.i32.npy` — `[n_docs]` token length per doc (includes trailing EOS).",
        "- `meta.json` — provenance + counts.", "",
        "Memmap `tokens.u16.bin`; bucket on `doc_lens` for the length curriculum. Loader: "
        "`morph/training/curriculum_data.py`.", "",
        "**Role-split:** only domain text — synthesis/reasoning splits are excluded by an "
        "ingest-time denylist and re-checked here.",
    ]
    return "\n".join(body)


def upload(pretok_dir, repo, private, yes):
    rows = verify_all(pretok_dir)                       # HARD GATE — never upload broken shards
    readme = _readme(rows, repo)
    with open(os.path.join(pretok_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"\nwrote {pretok_dir}/README.md ({len(readme)} chars)")
    if not yes:
        print("\n[dry-run] verified + card written. Re-run with --yes to create the repo and push.")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"[upload] repo={repo} private={private} — upload_large_folder (resumable) ...")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=pretok_dir)
    print(f"[upload] DONE → https://huggingface.co/datasets/{repo}")


def download(repo, dest):
    from huggingface_hub import snapshot_download
    os.makedirs(dest, exist_ok=True)
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest)
    print(f"[download] {repo} → {path}")
    verify_all(dest)                                    # integrity-check what we pulled


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--pretok-dir", default="data/pretok")
    u = sub.add_parser("upload")
    u.add_argument("--pretok-dir", default="data/pretok")
    u.add_argument("--repo", required=True)
    u.add_argument("--private", action="store_true")
    u.add_argument("--yes", action="store_true", help="actually create+push (outward-facing)")
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True); d.add_argument("--dest", default="data/pretok")
    args = ap.parse_args()
    if args.cmd == "verify":
        verify_all(args.pretok_dir)
    elif args.cmd == "upload":
        upload(args.pretok_dir, args.repo, args.private, args.yes)
    elif args.cmd == "download":
        download(args.repo, args.dest)


if __name__ == "__main__":
    main()
