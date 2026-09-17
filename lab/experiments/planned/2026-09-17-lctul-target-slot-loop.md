# Experiment: the code target — the TUL slot loop regressed onto the frozen VAE code

Status: planned

Date: 2026-09-17 (frozen before any GPU step beyond a 12-step resume smoke on the Spark).
Note: `.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md`.
Spec: `docs/tul-code-spec.md` §17. Tests: `tests/test_tul_code_target.py` (12, CPU).

## Question

When the slot loop is trained to MATCH the frozen VAE code of the next span (LaDiR with
TUL as the latent generator, no sampler), does the loop's cell carry the predictable part
of the span into the frozen coda's generation, and do the loop's passes move toward the
code?

## Hypothesis

H1: the loop can regress. The exit cosine to the code rises well above the entry state's,
the predicted cell helps the frozen coda write the span (OWN > ZERO, OWN > SHUF), and the
passes climb toward the code (per-pass cosine rises with t; forced depth lowers CE).
H0: the conditional mean is a confident vector at the mean direction; the frozen coda reads
it as a wrong code and OWN ≤ ZERO; the passes meet the target in one step (per-pass cosine
flat after l1), as every per-pass target before it (`per-pass-targets-met-in-one-step`).

## Predictions (frozen)

Arms `tul-code-target` (A, `code_target_detach: true`) and `tul-code-target-ce` (B,
`false`), both `tul_code_target*.yaml`: resume `tul-code-vae/step_10000.pt`, 20k steps,
batch 6, seq 1024, one seed each, E + prelude + embeddings + coda frozen. Controls on the
same 120 cuts: the strict ruler (`slot-spandec-strict-20k@20000`, OWN cosine 0.187), the
plain model (0.230), the VAE oracle (0.64 on the LaDiR chain's coda; this arm's ORACLE
condition re-reads it on the same coda).

- P-T1 (the loop regresses): `val/code_target_cos` ≥ 0.20 at 20k on arm A. 55 %.
- P-T2 (the passes move): at 20k on arm A, `tul/code_target_cos_l{T}` − `tul/code_target_cos_l0`
  ≥ 0.05 averaged over the last 500 logged steps, T the deepest logged pass. 45 %.
- P-T3 (own beats none): semantic probe at 20k, arm A, OWN − ZERO on cos_true > 0 with the
  95 % interval clear of 0. 40 %.
- P-T4 (own beats foreign): OWN − SHUF on cos_true > 0, interval clear of 0. 45 %.
- P-T5 (depth pays on the frozen coda): forced-depth sweep at 20k, arm A, `ce_tokens` at
  depth 6 minus depth 1 ≤ −0.02, paired interval clear of 0. 30 %.
- P-T6 (the bar): OWN within 0.03 of the strict ruler's OWN on cos_true, paired over the
  120 cuts (`code_semantic_pair.py`). 35 %.
- P-T7 (arm B reads the same): arm B's OWN − ZERO within 0.02 of arm A's (the CE route
  adds nothing the regression did not). 50 %.
- P-T8 (rate): both arms ≥ 0.9× the strict ruler's tok/s at batch 6 (the frozen token
  path stores no activations). 70 %.

## Binding

- P-T1 and P-T3 hold → the loop carries the predictable part into generation: next is the
  same arm with the coda UNFROZEN after 10k (does a reader that adapts to the predicted
  cell widen the gap) and the seed lane.
- P-T1 holds, P-T3 fails → the loop matches the code's mean direction and the frozen coda
  cannot use a mean: the refiner probe (draft → encode → regenerate) becomes worth its
  cost, and the note goes to rejected with the reading.
- P-T1 fails → the loop cannot regress onto this code from the past at 20k: the target is
  not reachable and the note goes to rejected.
- P-T2 fails with P-T1 holding → the regression is met in one pass like every per-pass
  target before it; the arm says nothing new about depth.

## Method

Queue lines at the head of `recon_arms.txt` (kind `slot`, so the runner sweeps forced
depths 1..16 and runs `worth_profile` at 20k), EXTRA
`training.resume=/home/wolfe/morph-to/checkpoints/morph/tul-code-vae/step_10000.pt`
plus the panel extras. The runner's 12-step smoke does NOT carry EXTRA, so the resume path
is smoked once on the Spark before queueing (12 steps, the same overrides). Spark watcher
`spark_code_probes_target.sh`: the semantic probe (`--kind target`) at 10k and 20k for
both arms; results under `lab/experiments/results/2026-09-17-lctul-target/`. Pairing
against the ruler and plain controls already saved under
`results/2026-09-16-lctul-dplan/controls/` with `code_semantic_pair.py`.

## Not verified before launch

- The 12 CPU contracts and a CPU generation smoke; the GPU resume smoke is run before the
  queue lines are written and its exit code recorded here.
- The frozen coda was trained on cells with cosine ~0.32 to the truth (noise 3.0 renormed);
  how it reads a cell at cosine 0.2 to 0.5 is not known and is what P-T3 measures.
- One seed per arm.

## Results

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
