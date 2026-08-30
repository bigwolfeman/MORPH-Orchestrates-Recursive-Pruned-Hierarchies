# Planned: the DB-fix pair — faithful one-pass DB (dbfix) + the conditioning falsifier (db_cond)

Status: success
Date: 2026-08-30 (frozen before launch). Configs: `tul_dbfix.yaml` (one-pass
σ-conditioned DB, EDM precond, Euler-ladder eval, step_mix {db1:1}),
`tul_db_cond.yaml` (the OLD db_loop + iter-conditioning — deviation #1 of the paper
audit, everything else = tul_l3). Machinery commit 81a9674; program note
`.agents/notes/proposed/architecture/2026-08-30-dbfix-program.md`; paper audit in the
2026-08-29 loop-ladder filing lineage (arXiv 2506.14202 App. E.5).

## Question

The paper's recurrent-depth recipe (one σ-conditioned pass in training, Euler ladder
at inference) demonstrably composes depth on Huginn. Our old db_loop lacked its one
mechanism (stage conditioning). Does the faithful recipe produce a depth-earning loop
on TUL slot geometry — and is conditioning ALONE enough to wake the old db_loop?

## Reference numbers (fixed)

l2cap: depth curve 0.233 nats (4.6220@1 → 4.3892@6), CE @4250 4.3489, 66 min.
l3 (old db_loop): depth-flat (|Δ|≤0.001), CE @4250 4.3519, 70 min. Coreless nomask
4.3102 @4250, ~30 min. Replicate CE spread 0.030–0.036. All eager, batch 6, 4500
steps, seed 1.

## Predictions (frozen)

- **D1 (dbfix stability).** Completes S1-clean: 80% (first GPU run of the db1 path;
  smoke gates VRAM/NaN/step_mix wiring).
- **D2 (dbfix, binding).** Ladder-step sweep (patched core_depth_sweep varying
  db1_ladder_steps, 48 rows) shows CE(K=6) ≤ CE(K=1) − 0.02: 45%. The paper's
  mechanism transfers or it does not; our clean-target adaptation (supervise the
  one-pass output through the existing CE/mux, no state regression) is the novel
  piece.
- **D3 (dbfix CE).** val/ce_tokens @4250 ≤ 4.46: 50%.
- **D4 (dbfix wall-clock).** Full 4500-step run ≤ 45 min (vs l2cap 66): 70%.
- **C1 (db_cond stability).** S1-clean: 90%.
- **C2 (db_cond, the deviation-#1 falsifier).** Depth sweep ≥ 0.02 nats: 30% — my
  prior says conditioning alone does NOT rescue db_loop, because its 4 mux states
  share one target (the same-job problem), and the paper differentiates jobs via the
  σ-dependent Euler update, which db_loop lacks.
- **Decision rules (binding).** D2 passes ⇒ task 9 unlocks (tul_ilv50 +
  tul_l2cap_cond run next). D2 fails but C2 passes ⇒ conditioning is the active
  ingredient and the one-pass objective is the problem — interleave stays LOCKED,
  next arm is db_loop+cond+σ-Euler-update hybrid, new prereg. Both fail ⇒ faithful
  DB does not transfer to TUL slot geometry at this budget; interleave CANCELLED
  (sequencing rule), the loop program proceeds on the gate-vs-cap ladder instead.

## Method

1. Smoke dbfix FIRST (steps=12, eval_every=5 — must exercise the db1 step, the
   ladder eval, and the step_mix logging live; gate on exit 0 + a `train/steps_db1`
   line reaching wandb-disabled stdout or log). Smoke db_cond (same shape; its new
   surface is iter-conditioning, zero-init proven bit-identical on CPU).
2. Run tul-dbfix then tul-db-cond, panel flags, seed 1, each followed by the
   (patched) core_depth_sweep at 48 rows and tul_samples, checkpoints pruned.
3. Artifacts → lab/experiments/results/2026-08-30-tul-dbfix-pair/. Wall-clock read
   from the queue-log timestamps (START→DONE), stated per arm.

**Method amendment (2026-08-30, during smoke, before full runs):** the
`train/steps_db1` gate line is a wandb-dict-only metric and never reaches stdout;
the smoke WARN it produced was a false alarm. The live-wiring gate was satisfied
instead by the startup line `step_mix ON: cycle=['db1']` plus exit 0. No change to
predictions.

## Not verified before launch

The db1 GPU path (bf16 autocast, memory) has never run — CPU tests only (691
passed). The step_mix trainer wiring has never executed in a live main() loop; the
smoke is its first execution. The ladder-K sweep extension was added today
(compile-checked, not yet run).

## Results

Wall-clock from queue-log stamps. Both arms trained clean, seed 1, batch 6, eager.

| cell | bar | measured | verdict |
|---|---|---|---|
| D1 dbfix S1-clean | no eval >0.20 over running min ×2 consecutive after step 1000 | max excursion +0.149 (step 2750), never ≥0.20 | **PASS** |
| D2 dbfix ladder (binding) | CE(K=6) ≤ CE(K=1) − 0.02 = 4.4452 | K=1 4.4652, K=6 4.4992 — monotonically WORSE past K=2 (K=8 4.5101) | **FAIL** |
| D3 dbfix CE | @4250 ≤ 4.46 | 4.4521 | **PASS** (margin 0.008) |
| D4 dbfix wall-clock | ≤ 45 min | 37.7 min (08:47:32→09:25:15) | **PASS** |
| C1 db_cond S1-clean | same as D1 | max excursion +0.136 (step 2750) | **PASS** |
| C2 db_cond depth | CE(K=1) − CE(K=6) ≥ 0.02 | 4.4102 − 4.4061 = 0.0041; spread K=1..8 only 0.005 | **FAIL** |

Supporting numbers:

- tul-dbfix: final val 4.4979@4500, wandb `yr4k0zo0`. Depth sweep (48 rows,
  db1_ladder_steps varied): 4.4652 / 4.4645 / 4.4735 / 4.4850 / 4.4930 / 4.4992 /
  4.5048 / 4.5101 for K=1..8. Best inference is K=1–2; each extra Euler step costs
  CE. Samples: topk50 rep4 0.094±0.21, sample_t1 0.0003, greedy rep4 0.848
  (degenerates under greedy like every non-l2cap arm).
- tul-db-cond: final val 4.4075@4500, CE@4250 **4.3584** (within 0.010 of l2cap's
  4.3489 — inside replicate spread), 68.5 min (09:51:03→10:59:34), wandb `7g864ehl`.
  Depth sweep dead flat: 4.4102 / 4.4065 / 4.4061 / 4.4052 / 4.4055 / 4.4061 /
  4.4060 / 4.4060 for K=1..8. Samples: topk50 rep4 0.126±0.25, sample_t1 0.0026,
  greedy rep4 0.840.
- Artifacts: `lab/experiments/results/2026-08-30-tul-dbfix-pair/` (both depth
  sweeps, both wandb eval histories, both sample JSONs).

## Verdict

Predictions held: D1, D3, D4, C1 passed at their favored priors; D2 (45%) and C2
(30%) both failed on the side my priors leaned. **The binding rule fires on the
both-fail branch: faithful DB does not transfer to TUL slot geometry at this
budget. Interleave CANCELLED** — `tul_ilv50` and `tul_l2cap_cond` do not run; the
loop program proceeds on the gate-vs-cap ladder (Gated Recurrent Transformers,
docs/references.md row 69).

Three mechanisms have now tried to make DB-style depth pay inside TUL — target
scheduling (db_loop/l3), σ+EDM one-pass with Euler-ladder inference (dbfix), and
iter-indexed AdaLN conditioning (db_cond). All three produce stable training and
good CE with an inert (or, for dbfix, depth-hostile) loop. The only recipe that has
ever earned depth here is gradient THROUGH the iterated map under contractivity
control (l2cap, 0.233 nats). Conditioning is not the missing ingredient.

## Updated hypothesis

Depth composition in TUL requires the training graph to contain the composition:
the model must feel ∂CE/∂(iterate k applied after iterate k−1) directly.
Inference-time recurrence over a one-pass-trained map (the paper's App. E.5 trick)
does not survive the transfer from Huginn's token-level recurrence to TUL's
slot-level geometry — plausibly because our slots carry pooled span content whose
σ-noised versions are off-manifold in a way Huginn hidden states are not. Next
discriminating test is the gate-vs-cap ladder: contractivity by architecture
(gated blend, bias +4) vs contractivity by constraint (σ-cap), both with full BPTT.
