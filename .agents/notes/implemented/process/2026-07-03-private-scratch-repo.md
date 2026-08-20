# Agent Note: Private Scratch Repo

Status: implemented

Origin: Ai-notes/07-03-2026/Private-Scratch-Repo/PLAN.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Private scratch repo plan — `ignore/` as git root

**Status:** executed 2026-07-03 — large artifacts deleted; private `morph-scratch` repo.
**Date:** 2026-07-03

## Goal

Keep the public MORPH repo clean, but version-control the important gitignored work product (agent notes, one-off probes, Hydra outputs, wandb run metadata) in a **private** GitHub repo whose working tree is `MORPH/ignore/`.

Agents write to `ai-notes/` or `Ai-notes/` interchangeably; both must land in the same tracked tree.

## Current inventory (this machine)

| Path | Size | Role |
|------|------|------|
| `ignore/` | **6.1 GB** | scripts, probes, logs, **and** huge traces |
| `Ai-notes/` | 580 KB | real research notes (canonical content today) |
| `ai-notes/` | 8.5 KB | only `Incomplete/` (agent typo path) |
| `Incomplete/` | 12 KB | agent incomplete notes at repo root |
| `outputs/` | 7.5 MB | Hydra run dirs (`train.log`, configs) |
| `wandb/` | **11 MB** | ~39 local runs (small today) |
| `checkpoints/` | **473 GB** | never touch |
| `data/` | **59 GB** | never touch |

### What is eating `ignore/`

| Class | Files | Size | Track? |
|-------|------:|-----:|--------|
| `*.m2g` (STP traces) | 2 | 2.5 GB | **No** |
| Large `*.json` (perf/census/fullmodel traces, >1 MB) | 14 | 3.5 GB | **No** |
| `*.optlog` | ~22 | ~80 MB | **No** |
| Scripts / notes / small txt (`*.py`, `*.sh`, `*.md`, …) | ~310 | **1.3 MB** | **Yes** |
| `*.log` (AB campaigns, pretok, etc.) | 265 | 25.5 MB | **Yes** (private; useful history) |
| Weights (`*.pt` etc.) | 13 | 1.2 MB | **No** (policy; keep pattern even if tiny now) |

**Trackable slice of `ignore/` alone (exclude m2g, optlog, files >1 MB, pycache): ~676 files, ~15 MB.**

With notes + outputs + selective wandb: initial private commit lands around **~25–40 MB**, well under GitHub comfort.

## Recommended layout

`ignore/` **is** the private repo root (not a clone elsewhere). Public MORPH keeps those paths gitignored; root paths become **symlinks into `ignore/`**.

```
MORPH/                              # public repo (unchanged tracking)
├── ignore/                         # PRIVATE git root
│   ├── .git/
│   ├── .gitignore                  # junk filters (below)
│   ├── README.md                   # what this repo is / how to use
│   ├── Ai-notes/                   # moved from MORPH/Ai-notes
│   │   └── Incomplete -> ../Incomplete
│   ├── Incomplete/                 # merged: root Incomplete + ai-notes/Incomplete
│   ├── outputs/                    # moved from MORPH/outputs
│   ├── wandb/                      # moved from MORPH/wandb
│   ├── perf/                       # existing; large JSON stay untracked
│   └── …                           # existing scripts, logs, probes
│
├── Ai-notes  -> ignore/Ai-notes    # symlink
├── ai-notes  -> ignore/Ai-notes    # same target (case typo fix)
├── Incomplete -> ignore/Incomplete
├── outputs   -> ignore/outputs
└── wandb     -> ignore/wandb
```

Public `.gitignore` already lists `ignore/`, `ai-notes/`, `Ai-notes/`, `outputs/`, `wandb/`, `Incomplete/` — no public-repo change required for ignore rules. Symlinks at those paths remain ignored by the public repo.

### Why both `ai-notes` and `Ai-notes` point at the same directory

Agents (and older notes) use either spelling. Two symlinks → one inode tree → one private-git history. No merge drama later.

`Ai-notes/Incomplete` → `../Incomplete` so writes to `ai-notes/Incomplete/foo.md` and `Incomplete/foo.md` also share one tree.

## wandb: include the folder, not the bloat

Today wandb is only **11 MB**, so co-locating it is fine. Full 100k-step runs will grow `run-*.wandb` event files a lot.

**Policy:** symlink and track **metadata**, ignore **binary event streams**.

Track (examples):

- `wandb/run-*/files/config.yaml`
- `wandb/run-*/files/wandb-summary.json`
- `wandb/run-*/files/wandb-metadata.json`
- optionally `wandb/run-*/files/output.log` if small

Ignore:

- `wandb/**/run-*.wandb` (binary history — the growth risk)
- `wandb/**/tmp/`
- `wandb/**/logs/`
- `wandb/debug*.log`
- `wandb/latest-run` (symlink)
- `wandb/debug-internal.log`

Cloud wandb remains source of truth for full metrics; private git keeps **reproducible run recipes + summaries** offline.

If even that feels heavy later: drop `wandb/` from the private index entirely and keep only the symlink for local convenience (folder present, fully gitignored in private repo). Easy to flip via `.gitignore`.

## Private `.gitignore` (draft)

```gitignore
# --- never commit ---
.env
*.pem
*.key
credentials*.json

# weights / tensors
*.pt
*.ckpt
*.safetensors
*.bin
*.npy
*.npz
*.pkl
*.pickle

# profiler / megatraces (the 6 GB problem)
*.m2g
*.optlog
*_trace.json
fullmodel_*.json

# known huge perf dumps (keep scripts in perf/, drop results)
perf/*.json
perf/**/*.json

# python / editor junk
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.DS_Store

# wandb bloat — keep files/config + summary via negation if needed
wandb/**/run-*.wandb
wandb/**/tmp/
wandb/**/logs/
wandb/debug*.log
wandb/latest-run
wandb/debug-internal.log

# optional safety: anything that looks like a checkpoint dir
**/checkpoints/
```

Notes:

- `perf/*.json` ignores the multi-hundred-MB census/r1/r4 dumps; probe **scripts** (`perf/*.py`, `perf/*.md`) stay tracked.
- Small result JSON outside `perf/` (e.g. `ab_hc_kernel_result.json` ~30 KB) stays tracked unless we add a size-based pre-commit check later.
- Do **not** blanket-ignore `*.log` or `*.json` — that would drop useful AB logs and small result files.

Optional later: a `pre-commit` hook that rejects any file >5 MB.

## What stays out forever (even private)

| Path | Why |
|------|-----|
| `checkpoints/` (473 GB) | weights; use object storage / local disk |
| `data/` (59 GB) | pretokenized corpora |
| `*.m2g`, large traces, `*.optlog` | regenerable profiler artifacts |
| `.env` | secrets |

These remain only on the public `.gitignore` (already) and never enter the private tree as tracked files. If someone drops a checkpoint into `ignore/`, private `.gitignore` still blocks `*.pt` etc.

## Public repo interaction

- Public MORPH: **no** submodule. Submodules are painful for agents and for “just write a note.”
- Public MORPH: keep listing `ignore/`, notes, `outputs/`, `wandb/` in `.gitignore` (already done).
- Optional one-liner in public `CONTRIBUTING.md` or `CLAUDE.md`: “scratch/notes live in private `ignore/` git; both `ai-notes` and `Ai-notes` symlink there.”
- Private `README.md`: clone instructions, symlink setup for a fresh machine, “do not force-add `*.m2g`.”

## Migration sequence (after approval)

1. **Freeze writes** briefly (no training/agent note dumps mid-move).
2. Write `ignore/.gitignore` (draft above).
3. Move into `ignore/`:
   - `Ai-notes/` → `ignore/Ai-notes/`
   - `outputs/` → `ignore/outputs/`
   - `wandb/` → `ignore/wandb/`
   - `Incomplete/` → `ignore/Incomplete/`
   - merge `ai-notes/Incomplete/*` into `ignore/Incomplete/`
   - remove empty `ai-notes/`
4. Inside `ignore/Ai-notes/`: `ln -s ../Incomplete Incomplete` (if not already present).
5. At MORPH root, create symlinks:
   ```bash
   ln -s ignore/Ai-notes Ai-notes
   ln -s ignore/Ai-notes ai-notes
   ln -s ignore/Incomplete Incomplete
   ln -s ignore/outputs outputs
   ln -s ignore/wandb wandb
   ```
6. Smoke-check paths agents use:
   - `Ai-notes/06-19-2026/.../MENTAL-MODEL.md` resolves
   - `ai-notes/Incomplete/` writable and same as `Incomplete/`
   - `python -m morph.training.train` still writes Hydra to `outputs/` and wandb to `wandb/`
7. `git init` inside `ignore/`, initial commit of tracked set, verify `git status` shows no m2g/optlog/huge json.
8. `gh repo create <name> --private --source=ignore --remote=origin --push`
9. Document remote URL in `ignore/README.md`.

### Suggested private repo name

- `morph-scratch` (short, clear), or
- `MORPH-private-scratch`

Owner: your GH user/org. **Private.** No public fork relationship required.

## Risk notes

| Risk | Mitigation |
|------|------------|
| Accidental commit of 3 GB JSON | private `.gitignore` + optional max-size pre-commit |
| Symlink breaks on Windows | you are on Linux; document “Linux/mac only” |
| Public git tries to track symlinks | paths already in public `.gitignore` |
| Agents create a **new** `ai-notes/` dir if symlink missing | setup script / README checklist on new machines |
| wandb grows later | already ignoring `*.wandb`; only metadata tracked |
| Nested `.git` confuses tools | normal pattern; public never enters `ignore/` |

## Decisions needed from you

1. **Repo name** — `morph-scratch` vs something else?
2. **wandb policy** — metadata-only (recommended) vs full current 11 MB tree vs symlink-only / fully ignored?
3. **Include `Incomplete/`** in the private tree (recommended yes)?
4. **Track `*.log` in `ignore/`** (recommended yes, ~25 MB of campaign history)?
5. **Public docs blurb** — add a short note to `CLAUDE.md` / `CONTRIBUTING.md`?

Once those are confirmed, execution is the migration sequence above (no public code changes beyond optional docs).
