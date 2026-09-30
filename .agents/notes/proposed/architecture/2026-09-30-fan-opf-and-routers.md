# Agent Note: three one-factor arms on the no-WTA write-all fan (OPF on the cells, a reader-trained router, a latent-taught router)

Status: proposed

## Problem

The write-all fan with no winner-takes-all term (nowta,
`morph/configs/tul_slot_spandec_strict_fan4_all_fp01_nowta.yaml`) writes all M = 4 loop cells
of a slot into its 4 prefix cells, and the coda's attention picks per token. At 5k steps
(`/home/wolfe/morph-scratch/abc/chain.log`, 2026-09-30) it ran 9774 tok/s against a2's 7783 and
reached val 4.4658 against a2's 4.4365; its paired gap to plain was +0.2543 against a2's
+0.2181, so it sits 0.036 nats behind a2 on the gap. Nothing gives one cell a job the other
cells do not have. a2's job-giver (the coda-graded WTA) costs M + 1 extra
coda passes a step. Its cheap replacement, a head grader (arm hwta,
[2026-09-29-fan-head-graded-wta.md](2026-09-29-fan-head-graded-wta.md)), lost: val 4.4800 at
5k, and the head's pick agreed with the coda's best on 0.266 of slots (chance 0.25), regret
0.0985 against a random pick's 0.1089 nats/token.

Three lessons constrain any new job-giver:

1. A grader that is not the reader picks near chance for the reader (hwta).
2. A latent target built from an encoder over a LIVE front moves under the run. The frozen E
   collapsed (own cosine 0.61 to 0.99 by step 3000, the `code_target_ref` note in
   `CLAUDE.md`); LCTUL-J's EMA target lost rank (77 to 40) and nothing reached the reader
   ([2026-09-22-lctul-ema-target-and-factors.md](2026-09-22-lctul-ema-target-and-factors.md)).
   A collapse guard must sit on the tensor whose EMA is the target.
3. REINFORCE on span CE is too noisy (arm B,
   [2026-09-29-code-policy-one-rollout.md](2026-09-29-code-policy-one-rollout.md)); use pathwise
   gradients.

## Proposal

Three arms. Each composes nowta and changes ONE key. They exclude each other.

**Arm F, `tul.fan_opf: true`** (`tul_slot_spandec_strict_fan4_all_fp01_opf.yaml`). Orthogonal
Predictive Factorization (JEPA-Anything, arXiv 2609.20800, Eqs. 2-10; paper code
`opf.py`, `losses.py`).

* Target. An EMA twin of the prelude (`FanTargetFront`, `morph/model/tul_fan_route.py`) runs
  the same row under the same strict masks (`_tul_front(twin=...)`, no embedding dropout, no
  RNG draw). The target of slot s is its output mean-pooled over span s+1's token positions
  (`span_ce_index`'s bin s+1, HC stream mean), then LayerNorm without affine. Under strict
  geometry the prelude is same-span only, so the target of slot s is a function of span s+1's
  tokens alone (pinned by perturbing every other span). Slots with no scored next span are
  masked out.
* The twin copies the prelude blocks and their x0 / value-embedding projections (45.7M
  parameters, 183 MB fp32 at d = 1024). The lookup tables (token, bigram, value-embedding
  tables, TUL slot inputs) are SHARED and read live under no_grad: they are about 1 GB of
  fp32, and the token table is tied to the LM head. EMA momentum `tul.fan_target_ema` 0.996.
  The paper does not state m; 0.996 is the I-JEPA / BYOL start value and is unmeasured here.
* The twin is not a registered submodule (the `code_target_ref` precedent). The trainer builds
  it right after quantisation (same ternary parametrisation as the live prelude) and before
  `torch.compile`, re-syncs it after `init_from` / resume, EMA-updates it after every optimizer
  step (`tul_fan_after_step`), and saves it under the checkpoint key `fan_target`. The lab
  loader (`scripts/tul_samples.py::load_ckpt`) restores it strictly and refuses a checkpoint
  without it. A labelled forward without a twin raises.
* Projector. `P` [d, d] (rows are the analysis rows, K = M = 4 blocks of r = d/4). It starts at
  the identity and is trained by L_pred and L_fac (the stop-gradient sits on the twin's
  output, not on P). After every optimizer step `P` is replaced by the Q of its QR
  factorisation with the sign canonicalised (the paper's strict mode), so it is orthonormal at
  every forward and L_orth is 0 and is not built.
* Predictors. q_k per cell: LayerNorm (no affine) -> Linear(d, d) -> GELU -> Linear(d, r),
  reading cell k's exit state (HC stream mean) WITH grad. My choice over a linear d -> r: a
  linear head would force the target coordinates to be literal linear coordinates of the
  carrier the coda also reads through `W_prefix`; one hidden layer lets the predictor decode
  while the gradient still reaches the cell. `tul.fan_opf_pred_hidden` (0 = d) sets the width.
* Loss, folded as `opf_weighted` = 1.0 L_pred + 0.05 L_fac + 0.02 L_enc (Table 10). L_pred:
  mean squared error over valid slots, factors and coordinates. L_fac: hinge at 0.1 on the
  batch std of each projected target coordinate (reaches P only). L_enc: hinge at 0.1 on the
  batch std of each coordinate of the ONLINE prelude's pooled span states in the target's
  space (the tensor whose EMA is the target; reaches the online prelude). The 0.1 floors are
  the paper code's `min_std` default; the paper text does not state them. The cells get L_pred
  jointly with the coda's token CE, which is the difference from LCTUL-J.
* Instruments (train `fan/opf_*`, val `val/fan_opf_*`): per-factor R^2 = 1 - MSE/Var (0 =
  predicts the batch mean), target participation ratio, online min / mean coordinate std, the
  smallest projected-target std, `max|P P^T - I|`.

**Arm R, `tul.fan_route: reader`** (`..._rmoe.yaml`). A top-1 router, MoE style.

* Score. `score_i = v^T ReLU(W_c LN(cell_i) + W_x LN(ctx))`, rank 64 (`tul.fan_route_rank`).
  ctx is the slot's loop input (the normed prelude output at the slot's first prefix
  position), detached. My choice over the register seed: the seed differs per cell (the
  register adds a per-cell trigger), so it would hand the router each cell's identity as
  context; the loop input is one vector per slot.
* Selection `argmax(score + b)`; b is a per-cell balance BIAS buffer (DeepSeek-V3
  auxiliary-loss-free balancing): after every optimizer step `b_i += 1e-3 * sign(mean_load -
  load_i)`, load the winner share over the valid slots of that step's training forwards. The
  bias chooses and never enters p. The trainer clears the counts of the compile-warmup forwards
  before the loop.
* Read. The coda reads ONLY the winner, scaled by `p_winner = softmax(score)[winner]` (Switch
  Transformer); the losers' source cells are written as zero. My choice over masking the
  losers out of the coda's attention: a zero source needs no mask surgery, composes with the
  plan ablations, and the conv / value shift inside a slot's segment would still read a
  masked-out key position. Under nowta's `prefix_source: exit` no `E_pass` is built, so a
  loser's prefix cell is exactly zero and carries nothing (pinned). The coda's token CE reaches
  the router and every cell's score through `p_winner`, and reaches the winning cell's value
  alone.
* The losers stay visible to later slots' LOOP: the loop runs to completion over the cell
  sequence before the pick is made; the prefix write feeds only the coda.
* Same selection at train and eval; no noise, no train-only branch.

**Arm T, `tul.fan_route: latent`** (`..._rlat.yaml`). Arm R's router, bias and read, trained by
a latent TEACHER instead of the reader.

* Target: arm F's (the shared `_tul_fan_target`).
* g (LayerNorm -> Linear d -> d) reads each cell DETACHED and is trained by MSE onto the
  target for all M cells. Teacher pick `argmin_i ||g(sg cell_i) - z||^2` under no_grad. The
  router reads the cells and ctx detached and is trained by cross-entropy onto the teacher's
  pick. Folded as `rlat_weighted` = 1.0 * (router CE + g MSE). None of it reaches the cells.
* The coda reads the router's pick HARD (no p), so the reader's CE does not train this router;
  the cells get winner-only credit from the coda.

**Shared.** Refused (`TULConfig._check_fan_opf_route`, near the top of `__post_init__`) with
each other, with `fan_all_wta_lambda > 0`, `code_enum_k > 1`, `fan_mix != 'all'` and `bcast`;
every arm key set while its arm is off is refused. Every new module is built in a forked RNG
stream with its own seed, so the base weights of an arm equal nowta's; arm F's forward draws
no extra RNG, so its dropout masks and depth draws are nowta's. Every key at its default is
bit-identical to 622fb13 (nowta pins, tests/test_tul_fan_opf.py). Val-only oracle readings on
`_tul_fan_oracle`'s per-cell coda table (row i = the router's write with the winner forced to
i): `fan/router_coda_agree` (chance 0.25), `fan/router_pick_regret` against
`fan/rand_pick_regret` (nats/token, the hwta units), `fan/teacher_coda_agree` (T). Train and
val: `route_load_k*`, `route_bias_k*`, `route_entropy`, `route_p_win`, `teacher_router_agree`,
`rlat_ce`, `rlat_g_mse`, `rlat_teacher_share_k*`.

## Alternatives considered

* **The paper's soft `L_orth` (Eq. 7) instead of the QR retraction.** Lost: the retraction
  makes the factors an exact orthogonal split at every step with no weight to tune; the paper's
  strict mode exists and its code implements it (`qr_retraction`).
* **EMA of the whole front, tables included.** Lost on memory (about 1 GB more fp32 state on a
  card with about 1 GB of slack) and because the token table is the tied LM head.
* **A full-model deep copy as the twin (`code_target_ref`'s way).** Lost on memory (304M
  parameters against the prelude's 45.7M) for no gain: only the prelude computes the target.
* **Registering the EMA as buffers of the model.** Lost: quantisation would then have to
  ternarise buffers, and the prune / carve / probe walks enumerate modules.
* **A linear predictor d -> r.** Considered; see the predictor bullet.
* **Losers masked out of the coda's attention (R / T).** See the read bullet.
* **The register seed as the router's context.** See the score bullet.
* **An auxiliary load-balance LOSS (Switch).** Lost to the bias: a loss would be a second term
  pulling on the cells and the router beside the one factor under test; the bias only moves
  the selection.
* **Scaling T's read by p.** Lost: the reader's CE would then train T's router and the arm
  would not isolate the teacher.

## Acceptance criteria

* Built and tested (done in this change, CPU, one core): defaults bit-identical (nowta pins);
  per-mechanism contract tests and a 28-case sabotage pass, 28 caught after one added test (tests
  `tests/test_tul_fan_opf.py`, `tests/test_tul_fan_router.py`). 41-step GPU smokes of nowta,
  F, R, T with finite losses and the new keys present.
* Each arm runs 5000 steps (queue lines `/home/wolfe/morph-scratch/abc/queue_staged_opf.txt`)
  and is read against nowta's 5k run on val/ce_tokens, tok/s and the arm's own instruments.
  F: R^2 above 0 on every factor while `opf_target_rank` does not fall toward 1 and
  `opf_enc_std_min` stays above 0.1. R / T: `router_pick_regret` clearly below
  `rand_pick_regret`; T also `teacher_coda_agree` above 0.25.
* An arm that is only as good as nowta at 5k is filed as a failure with its instruments, not
  extended.

## Risks

* **Target collapse (F, T).** The online prelude trains, so the EMA target moves (lesson 2).
  L_enc guards the online pooled states; `opf_target_rank` is the collapse reading. In F the
  cells also train on the token CE, so a collapsed target costs the loss term, not the cells'
  only signal.
* **R's scale at init.** `p_winner` starts near 0.25, so the written cell is about 4x smaller
  than nowta's cells at step 0. That is the Switch design; it is a real difference in the
  coda's input scale.
* **R's loser still trains through the span decoder.** The span decoder reads the MEAN of the
  raw cells, as on nowta. "Winner-only credit" holds for the coda's CE only.
* **T's teacher agreeing with the coda at chance.** The latent-WTA probe
  ([2026-09-29-latent-wta.md](2026-09-29-latent-wta.md)) read chance agreement for every latent
  scorer it tried. `teacher_coda_agree` and the regret pair say whether T repeats it.
* **RNG neutrality on the GPU.** The first build seeded its modules with `torch.manual_seed`
  inside `fork_rng(devices=[])`. That reseeds every CUDA generator and restores only the CPU
  one, so arm F's step-0 token CE read 11.2827 against nowta's 11.2654 on the 5090 while every
  CPU test stayed bit-equal. The modules now seed the CPU default generator only
  (`_cpu_seeded`), and a test pins that the build makes no CUDA seed call.
* **Structure changes.** The twin pairs tensors by name every step and raises if the live
  prelude changes structure (a carve). These configs never prune (`prune_start` 999999999).
* **Cost.** F and T run a no-grad prelude pass on the twin every labelled forward; the smoke
  tok/s is in the change's report, the 5k tok/s decides.
