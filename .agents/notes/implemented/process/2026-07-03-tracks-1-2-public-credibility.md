# Agent Note: Tracks 1 2 Public Credibility

Status: implemented

Origin: Ai-notes/07-03-2026/tracks-1-2-public-credibility.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Tracks 1–2 implementation (2026-07-03)

Follow-on to `MORPH-Recon-Report.md`. Track 3 deferred.

## Track 1 (no model behavior change)

- `docs/runtime-invariants.md`
- `docs/evidence/ablation-ledger.md`
- `docs/evidence/known-good-runs.md`
- `docs/MANIFEST.md` links
- `morph/configs/base.yaml`: `training.seed: 0`; dataset `~/.cache/...`; `wandb.entity: null`
- README: “What Is Public Vs Private Evidence?” + repo map entries

Seed default matches prior `getattr(tr, "seed", 0)`. Dataset/entity neutralization is portable; override if needed.

## Track 2 (contract tests)

| Module | Contract |
| --- | --- |
| `tests/test_lifecycle_checkpoint.py` | post/pre/legacy `next_step` |
| `tests/test_lifecycle_phase_transition.py` | prune density↓ → compact → route flags |
| `tests/test_lifecycle_kernel_mode.py` | `use_kernels` ↔ `force_eager`, last-build-wins |
| `tests/test_lifecycle_tst_qat.py` | TST ~log(V) not log(V)/s; ternary STE wired |
| `tests/test_lifecycle_decode_kv.py` | decode last-pos vs full forward; cache.pos |

All CPU-safe. Full `pytest tests/`: 60 passed.
