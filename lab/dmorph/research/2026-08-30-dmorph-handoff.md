# Agent Note: dmorph — the no-loop DB/FM head-to-head (wrap-back handoff)

Status: proposed

## Problem

Wolfe declared DiffusionBlocks a dead end for MORPH's *looping* — but wants to
come back with **dmorph**: a MORPH iteration with NO core loop, keeping the
Thought Unpack Loop (slot geometry), trained with a flow-matching or diffusion
objective. The open question: DB is strong on wall-clock and FLOP count but
token-inefficient — is token efficiency really *that* important? The only way to
answer is head-to-head against a slightly different variant that does not rely on
looping at all.

Evidence chain that killed the loop side (all on `perf/throughput-lever-stack`,
pushed to origin 2026-08-30):

- Whole-model DB rejected 2026-08-21:
  `.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md` — plain
  NTP CE 4.0010 vs best DB arm 5.0801 at sigma_max, matched 143.4M-token budget.
- Loop-side DB, three mechanisms, all inert (filed
  `lab/experiments/successes/2026-08-30-tul-dbfix-pair.md`, commit `cb7da65`):
  target scheduling (db_loop/l3), faithful σ+EDM one-pass (dbfix), iter-AdaLN
  conditioning (db_cond). dbfix's Euler ladder is monotonically WORSE with depth
  (K=1 4.4652 → K=8 4.5101). db_cond's depth curve is flat to 0.005 nats.
- The paper (arXiv 2506.14202) never measured a depth-vs-steps curve for its
  Huginn variant and never compared against flat compute — our K-sweep is the
  first measurement, and it supports Wolfe's read: much of DB's PPL regression
  likely comes from the inference loop count itself making the model worse.
  **dmorph should run K=1 or a shallow ladder, not 50 Euler steps.**
- Only gradient-through-the-iterated-map under contractivity control (l2cap,
  σ≤1.5 post-step projection + full BPTT) ever earned depth: 0.233 nats. That is
  the loop's price of admission, and dmorph deliberately declines to pay it.

## Proposal

Head-to-head at matched wall-clock AND matched tokens (both axes, separately):

- **dmorph arm**: TUL slot geometry, no core loop (n_core=0 path or single
  conditioned pass), FM/diffusion objective on slot states, σ conditioning via
  the shipped AdaLN machinery, K=1-or-shallow inference.
- **Control**: coreless TUL baseline (plain CE) — already measured: CE@4250
  4.3102, ~30 min/4500 steps @ batch 6 (the flat-compute bar).
- Decide by token CE at matched wall-clock, and wall-clock at matched CE. Pair
  with generation metrics (rep4/distinct3 vs real-text anchor, greedy included —
  gen health tracked mechanism, not CE, all campaign).

### Reusable machinery (all commits on `perf/throughput-lever-stack`)

| What | Where | Commit |
|---|---|---|
| `CoreStageConditioning` (AdaLN-Zero, iter/σ keyed, bit-identical at init) + `DB1Sampler` (equal-mass log-normal σ + EDM precond) | `morph/model/iter_cond.py` | `81a9674` |
| One-pass DB training step + deterministic Euler-ladder eval | `_tul_core_db1`, `_tul_core_db1_ladder` in `morph/model/transformer.py` | `81a9674` |
| Per-forward `tul_step_mode` + `training.step_mix` interleave cycle (general primitive, not DB-specific) | `morph/training/train.py` (`build_step_mix_cycle`) | `81a9674` |
| Configs: `tul_dbfix`, `tul_db_cond`, `tul_l2cap_cond` (built, unrun), `tul_ilv50` (built, cancelled) | `morph/configs/` | `81a9674` |
| Depth sweep that varies `db1_ladder_steps` for σ-conditioned models | `lab/divergence/core_depth_sweep.py` | `1e86029` |
| Stratified paired worth scorer (per-token CE deltas by offset-in-span, bootstrap CI) | `lab/divergence/worth_profile.py` | `a40c3cd`, `8aeb62b` |
| Sampler slot-budget auto-widen | `scripts/tul_samples.py` | `0a4e7b7` |
| EDM/SigmaConditioning/euler_step primitives (whole-model era) | `morph/model/diffusion_blocks.py` | this branch + `park/db-master-line` |

### Reference numbers for the head-to-head (batch 6, 4500 steps, seed 1, eager)

| arm | CE@4250 | wall-clock | loop verdict |
|---|---|---|---|
| coreless nomask (flat bar) | 4.3102 | ~30 min | — |
| l2cap (loop champion) | 4.3489 | 66 min | +0.233 nats depth |
| db_cond | 4.3584 | 68.5 min | dead (0.004) |
| l3 / db_loop | 4.3519 | 70 min | dead |
| dbfix (faithful one-pass) | 4.4521 | 37.7 min | inverted (worse with K) |

Artifacts: `lab/experiments/results/2026-08-30-tul-dbfix-pair/` (depth sweeps,
wandb eval histories, sample JSONs). wandb: `adew-me/morph-tul` runs `yr4k0zo0`
(dbfix), `7g864ehl` (db_cond). Checkpoints (LOCAL ONLY, /home/wolfe/morph-perf/
checkpoints/morph/*/step_4500.pt): tul-dbfix, tul-db-cond, tul-l2-cap, coreless
et al.

### Adjacent testbed

`/home/wolfe/11-DiffusionBlocks-Testing/`: clean DB-on-Llama ladder (wandb
dbref-ladder), AR baseline PPL 54.65 @143M tokens. Hard-won DB lessons that
transfer to dmorph: argmax bridge not softmax@E (gen-PPL 584→17); sigma_max is
the metric, the sigma-grid mean ranks arms backwards; B=1 is DB's weakest
setting (B=4 4.67 vs B=1 7.30 at sigma_max); the DB readout at sigma_min is a
point mass — sampling knobs are inert; SliceScaler σ* put 77% of training into
autoencoding — check σ* before trusting any DB training mix.

## Alternatives considered

- **Iterate on the loop further (gate-vs-cap ladder)**: that program continues
  separately on `perf/throughput-lever-stack`; dmorph is the no-loop fork, not a
  replacement for it.
- **Merge the dbfix machinery into this branch now**: rejected — the TUL-side
  code has moved far past this branch's fork point; cherry-picks would conflict
  for no benefit. The perf branch is pushed; SHAs above are durable.
- **Whole-sequence diffusion (SEDD-style) instead of TUL-slot FM**: out of scope
  for dmorph v1; the TUL slot substrate is the part of MORPH being kept.

## Acceptance criteria

- A frozen prereg exists before any dmorph run (predictions on both axes:
  matched-wall-clock CE and matched-token CE vs the coreless bar).
- The dmorph arm trains S1-clean and its inference uses K≤4.
- The head-to-head is scored on token CE (not MAUVE / teacher gen-PPL — the
  paper's metric gap is exactly what we refuse to inherit), with generation
  samples delivered alongside.

## Risks

- The σ*-autoencoding trap (77% of training wasted) can silently recur in any
  new noise schedule; check the σ* number first.
- The coreless bar (4.3102) is an n=1 number with ~0.03 replicate spread; a
  serious head-to-head needs within-run or multi-seed reads.
- The checkpoints backing the reference table are local-only and pruned to
  step_4500; if the 5090's disk goes, only wandb histories survive.
