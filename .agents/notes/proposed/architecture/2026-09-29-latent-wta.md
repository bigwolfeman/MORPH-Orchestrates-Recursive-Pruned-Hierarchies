# Agent Note: latent-space WTA winner for LX-Fan (`tul.fan_all_wta_winner="latent"`)

Status: proposed

> **Correction and verdict, 2026-09-29 (orchestrator review).** The regret table further
> down is WRONG: the probe multiplied a per-span SUMMED CE by the span's token count again
> (each span counted n_tok times) and pooled batches by slot count. Fixed in
> `lab/divergence/latent_wta_probe.py` (cbc4530), which also reports the expected regret of
> a RANDOM cell. Rerun 18:05 CDT, same checkpoints, 96 rows, batch 6, seed 0, temp 0.1
> (`/home/wolfe/morph-scratch/latwta/probe_lxfan4_wta_v2.json`). Nats per token;
> "vs random" = random regret minus the rule's regret, positive = better than random:
>
> | rule | b5k agree | b5k regret | b5k vs random | ev01_3k agree | ev01_3k regret | ev01_3k vs random |
> |---|---|---|---|---|---|---|
> | a_cos | 0.232 | 0.1458 | -0.0076 | 0.278 | 0.1192 | +0.0158 |
> | a_l2 | 0.208 | 0.1525 | -0.0143 | 0.237 | 0.1433 | -0.0084 |
> | b_cos | 0.240 | 0.1470 | -0.0088 | 0.249 | 0.1353 | -0.0003 |
> | b_l2 | 0.210 | 0.1502 | -0.0120 | 0.240 | 0.1444 | -0.0095 |
> | c_infonce (shipped) | 0.246 | 0.1425 | -0.0043 | 0.258 | 0.1301 | +0.0049 |
> | random cell | 0.25 | 0.1382 | 0 | 0.25 | 0.1349 | 0 |
>
> **Verdict: no zero-shot latent score beats a random cell.** On the mature checkpoint every
> rule, the shipped InfoNCE one included, is worse than random; the one clear positive
> (a_cos, +0.0158 at step 3000) flips to -0.0076 at step 5000. `fan_all_wta_winner:
> latent` is therefore a random-winner WTA with extra steps. Do not queue it as a speed
> path. It stays in the tree default off; removing it is Wolfe's call. The cheap-grader
> question moves to a grader TRAINED on the true span (arm A, `fan_all_wta_grader: head`,
> note `2026-09-29-fan-head-graded-wta.md`). No standard error was computed; differences
> under ~0.01 are not read as effects.

## Problem

`tul.fan_all_wta_winner="per_rollout"` (the shipped default under `fan_mix="all"` +
`fan_all_wta_lambda > 0`) finds each (rollout, slot)'s winning register cell by running
**M no-grad coda passes** — one per stream, on the full K-fold rollout batch — before the
one grad pass that charges the winner-alone span CE. Profiled
(`lab/theory/efficient-latent-exploration-lit-2026-09-29.md`): 16 of the arm's 24
coda-units/step are these picking passes, two thirds of its ~3.4x slowdown against the
plain loop. `"map"` (shipped 2026-09-29, commit `c2d77a3`) cuts this to a B0-row batch by
sharing one winner across all K rollouts, but only when `code_enum_k >= 2`.

Question: can a **cheap, already-computed latent-space signal** replace the M coda passes
entirely — zero coda reads at pick time — without materially changing which cell wins?

## Proposal

`tul.fan_all_wta_winner="latent"`: pick each (rollout, slot)'s winner from a score against
`latent_wta_target` (`morph/model/transformer.py`) — a mean-pool of the TRUE next span's
own prelude states, already computed upstream of `_tul_fan_all`, no new network, no coda
read — scored with `latent_wta_infonce_score`: cell (b,s,m)'s InfoNCE own-target
log-probability against every OTHER valid slot's target in the local (rollout-expanded)
batch as in-batch negatives, temperature `tul.fan_all_wta_latent_temp` (default 0.1).
Both the target and the score are computed inside the method's existing `torch.no_grad()`
block from already-`.detach()`d tensors (`base_d`, `cells_d`); only the integer `choice`
index leaves that block, feeding the SAME `.gather`/`.scatter` write and the SAME
winner-alone grad pass every other winner mode uses. Works at any `code_enum_k`
(each (rollout, slot) scored independently — unlike `"map"`, which needs `>= 2` rollouts
to share a winner across).

### Part 1: offline probe (`lab/divergence/latent_wta_probe.py`)

On two already-trained `lxtul-lxfan4-wta-fp01` checkpoints (b5k = step 5000, the "mature"
snapshot; ev01_3k = step 3000, the "immature" one — both from the same LX-Fan+WTA
lineage), train mode (real dropout, a fixed seed), the probe reproduces the coda's own
`_tul_fan_all` WTA pick (M no-grad coda passes, the ground truth) and compares four cheap
candidates by **REGRET** (nats/token of the cheap pick's span CE vs the coda's own
argmin — 0 if they always agree, never negative) and **agreement** (chance = 1/M = 0.25).
The task's own kill rule: rank by regret, not agreement — a candidate can agree often and
still be expensive when it disagrees, or disagree often and be cheap when it does.

A first pass at N=6 rows (1 checkpoint, 1240 slot instances) suggested the mean-pool
target scored with raw cosine. The full-spec run — N=96 rows, both checkpoints,
**19,928–19,944 independently-scored slot instances per candidate per checkpoint** —
changed the ranking:

| candidate | b5k (step 5000) regret | ev01_3k (step 3000) regret | agree (b5k / ev01_3k) |
|---|---|---|---|
| a_cos (nearest to next slot's pre-loop seed) | 3.266 | **2.534** | 0.232 / 0.278 |
| a_l2 | 3.298 | 3.210 | 0.208 / 0.237 |
| b_cos (mean-pool target, raw cosine) | 3.263 | 2.962 | 0.240 / 0.249 |
| b_l2 (mean-pool target, raw L2) | 3.226 | 3.247 | 0.210 / 0.240 |
| **c_infonce (mean-pool target, InfoNCE vs in-batch negatives)** | **3.145** | 2.793 | 0.246 / 0.258 |

(nats/token, N ≈ 19.9k per cell; chance agreement 0.25.)

`c_infonce` beats every OTHER candidate on b5k (the mature checkpoint) and is 2nd-best on
ev01_3k, losing there only to `a_cos`. `a_cos`'s apparent lead on ev01_3k is not read as a
robust signal: it is BEST on the immature checkpoint and WORST-TIER on the mature one
(3.266, effectively tied with `a_l2` for last) — the direction a cheap "nearest to the
pre-loop seed" proxy is expected to move as training progresses and the register's cells
differentiate from their shared seed (this project's own measured early rank collapse,
`slot_eff_rank` 5.7–7.3 / pairwise cos 0.72–0.77, motivated the register's own existence).
`c_infonce` is the only candidate that is never worst and is best on the checkpoint closer
to what a longer run would actually look like — the basis for picking it under "lowest
regret, not highest agreement, and pick the target that generalizes across checkpoints."

`c_infonce` beats the mean-pool target's OWN raw-cosine scoring (`b_cos`) by 0.118 nats/tok
on b5k and 0.169 nats/tok on ev01_3k — the InfoNCE score is a materially better use of the
SAME target vector, not a different target.

### Part 2: implementation

- `morph/model/tul.py`: `fan_all_wta_winner` gains the `"latent"` value (third, alongside
  `"per_rollout"`/`"map"`); `fan_all_wta_latent_temp: float = 0.1` (InfoNCE temperature,
  read only under `winner="latent"`, must be `> 0`).
- `morph/model/transformer.py`: `latent_wta_target` (the mean-pool target, shared with the
  probe's `candidate_b_pool`) and `latent_wta_infonce_score` (the InfoNCE score table,
  shared with the probe's `candidate_c_pick`) are module-level functions, ONE
  implementation each for the probe and the shipped mode. `_tul_fan_all`'s `"latent"`
  branch: builds the target, scores it, negates for `select_winners`' ARGMIN convention
  (reusing the same `fan_select_eps` random-override logic every mode uses), writes
  `choice`/`forced`/a winner-share-entropy collapse instrument
  (`fan_wta_latent_entropy`), and appends a full diagnostic snapshot to
  `_fan_all_winner_capture` when a test attaches one (`latent_cos` [diagnostic, not
  scored on], `latent_score`, `latent_cells_d`, `latent_cells_score`, `latent_target`,
  `latent_ok`). No coda pass runs, so `wta_oracle_ce`/`wta_single_ce`/`wta_pick0` (present
  under `"per_rollout"`/`"map"`) are absent under `"latent"`.
- `morph/training/tul_setup.py`: `KNOWN_TUL_KEYS`, the runtime constructor, the wandb
  manifest, and the build-time banner all carry `fan_all_wta_latent_temp` /
  the `"latent"` description.
- `morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01_latwta.yaml`: composes
  `..._wta_fp01` (arm b) + `fan_all_wta_winner: latent` +
  `fan_all_wta_latent_temp: 0.1` (explicit, matches the probe's own default) + the same
  `steps=3000`/`lr_decay_steps=5000`/`seed=1` pin as the sibling `..._ev01` control arm
  (`per_rollout`, otherwise identical) — the two configs differ by exactly those two
  `tul.*` keys plus `wandb.name`, confirmed by a resolved-config diff test.

### Teacher-forcing-bypass guard (why this is not "read the answer")

The target is built from the TRUE next span and is used ONLY to pick an index among the
model's OWN M cells, entirely inside `torch.no_grad()`, from already-detached tensors.
Only the integer `choice` leaves that block — never the target itself, never a gradient.
`test_latent_target_never_reaches_the_coda_input` proves this directly: with
`latent_wta_target` monkeypatched to a sentinel value, the sentinel never appears in the
grad pass's actual coda input tensor. This mirrors — and was checked against — this
project's own named failure modes for latent targets built from ground truth
(`teacher-forcing-is-a-bypass`, `fitted-z-used-the-answer`,
`frozen-encoder-on-live-front-is-not-a-fixed-target`).

### Dropout (Shen et al. 2019)

Shen et al.'s finding — compute WTA responsibilities without dropout noise between the M
candidates — applies to `"per_rollout"`/`"map"`, which run M SEPARATE no-grad coda
passes and would otherwise draw a fresh dropout mask per candidate. `"latent"` runs no
such passes: `base`/`cells` carry whatever ONE dropout mask this forward's prelude and
loop already drew, shared identically by the target and every one of the M candidates
compared against it, so there is no per-candidate draw to decorrelate. Re-running the
prelude a second time, dropout-free, to strip even that one shared draw was **not done**
— named here as unverified, not hidden.

### Verified

- Numerical equivalence: `latent_wta_infonce_score` (vectorized) was checked against the
  original per-row Python-loop formula (the one that actually produced the table above)
  on random fixtures, including the degenerate `n < 2` case — max absolute difference
  `0.0`, picks identical.
- Tests: `tests/test_tul_latent_wta.py` (16 tests, CPU) — zero coda pick passes; the pick
  equals an independently recomputed InfoNCE argmax against the model's own captured
  target/cells/mask; the target never reaches the coda input (sentinel probe); perturbing
  the target only moves `choice`, never the written cells; no gradient reaches the scoring
  path (positive control: backward still trains real params); runs under real dropout
  (0.2); works at `code_enum_k=1` (unlike `"map"`); the entropy instrument is present,
  bounded, and absent under other winner modes; the config composes, diffs from the
  control by exactly the intended keys, and reaches `model_cfg`; six refusals (bad value,
  needs `fan_mix='all'`, needs `fan_all_wta_lambda > 0`, needs
  `fan_all_wta_latent_temp > 0`, temp read-only under `"latent"`, does NOT need
  `code_enum_k >= 2`). Full regression: `tests/test_tul_latent_wta.py`
  `tests/test_tul_lx_credit.py` `tests/test_tul_lxfan.py` `tests/test_tul_fan_all.py`
  `tests/test_tul_wta_shared_winner.py` `tests/test_tul_wta_fused_ce.py` — 110 passed.
  Whole tree: `pytest tests/` — 2926 passed, 56 skipped, 1 xfailed, 0 failed
  (587.74s).
- 30-step GPU trace (`training.steps=30`, `WANDB_MODE=offline`, seq 1024, batch 6, both
  checkpoints untouched — a fresh-init smoke, not a resumed run), both arms built with the
  correct `winner=` in their own log banner:
  - `latwta` (`winner='latent'`): step 0 loss 22.3127, sps 0.38, tok/s 2328, peak 17.53GB;
    step 20 loss 21.8536, sps 0.51, tok/s 3140, peak 19.58GB.
  - `ev01` (`winner='per_rollout'`, the control): step 0 loss 22.3023, sps 0.29,
    tok/s 1792, peak 17.53GB; step 20 loss 21.8540, sps 0.40, tok/s 2485, peak 19.58GB.
  - `latwta` is ~26–27% faster (tok/s, sps) at matched steps, as expected from removing
    the M no-grad coda passes; peak memory is IDENTICAL at both checkpoints sampled
    (dominated by the grad pass + the model's own deployed pass, shared by both arms).
    Losses track closely at 30 steps from a fresh init — too short to read as a quality
    signal either way.

## Alternatives considered

- **Raw cosine to the mean-pool target (`b_cos`)** — the first-pass (N=6) pick. Rejected
  once the full-spec N=96 data landed: it loses to `c_infonce` on both checkpoints by
  0.118–0.169 nats/tok. Kept as a live diagnostic (`latent_cos` in the capture dict) so a
  future read can see how far cosine and InfoNCE keep disagreeing (measured `agree` ≈
  0.19–0.28 on the probe, near chance).
- **Raw L2 to the mean-pool target (`b_l2`)** — worst or near-worst on both checkpoints.
  Rejected on regret.
- **Nearest to the NEXT slot's own pooled pre-loop seed (`a_cos`/`a_l2`)** — best regret on
  the immature checkpoint, worst-tier on the mature one. Rejected as checkpoint-dependent
  and likely an artifact of the register's known early rank collapse, not a signal that
  would hold up over a longer run.
- **`"map"`'s shared-mixture approach** — ships already (commit `c2d77a3`), not competing
  with this proposal: `"map"` needs `code_enum_k >= 2` to share a winner across rollouts
  and still runs M no-grad coda passes on a smaller (B0-row) batch. `"latent"` runs NONE
  and works at `code_enum_k=1`. The two are siblings, not alternatives to each other.
- **A second dropout-free prelude pass** (to fully satisfy Shen et al. 2019) — not
  implemented; would cost a full front pass to remove noise this mode's OWN M-pass
  removal already mostly avoids (see Dropout section above). Left as a named risk, not
  built, given the offline probe already measures net-positive regret under the shared
  single draw.

## Acceptance criteria

- [x] Offline probe on real checkpoints, ranked by regret not agreement, target+score
  chosen and justified with real numbers (not invented after the fact).
- [x] Implementation: config-reach test proves `build_tul_runtime(cfg).model_cfg` carries
  the value; teacher-forcing-bypass guard proven by test, not asserted by comment alone;
  detach/no-grad proven by test; dropout interaction named.
- [x] Tests, sabotage-checked (see below), full regression suite green.
- [x] Config composed and diffed against the matched control arm.
- [x] 30-step GPU trace of both arms, build-line evidence of the correct mode, real
    loss/tok-s/peak-GB numbers.
- [ ] **NOT done**: the actual 3000-step queued run (`lxtul-lxfan4-wta-fp01-latwta` vs the
    `..._ev01` control) that would show whether `"latent"`'s cheaper pick costs quality
    over a real training horizon — only a 30-step fresh-init smoke has run. Queued:
    `/home/wolfe/morph-scratch/abc/queue_latwta.txt`.
- [ ] **NOT done**: sensitivity to `fan_all_wta_latent_temp` — shipped at the probe's own
    default (0.1) with no sweep.
- [ ] **NOT done**: a live check of whether the shared-dropout-draw gap from Shen et al.
    2019 (named above, not built) costs anything measurable on this arm specifically.

## Risks

- **Checkpoint generalization.** The probe's ranking rests on two checkpoints from ONE
  lineage (`lxtul-lxfan4-wta-fp01`, steps 3000/5000). If the queued 3000-step run diverges
  meaningfully from that lineage's own trajectory, the InfoNCE target's regret advantage
  is unverified beyond it.
- **Temperature sensitivity.** `fan_all_wta_latent_temp=0.1` matches the probe's own
  default and nothing else; no sweep has been run, live or offline.
- **In-batch negative pool includes rollout duplicates.** For `code_enum_k > 1`, the same
  physical row's K rollout copies share an identical target (the front runs once), so they
  appear as near-duplicate negatives in the InfoNCE pool. This does not hand a cell an
  easier match at its OWN (row, slot) position (argued in
  `latent_wta_infonce_score`'s docstring) but was not separately ablated (e.g. filtering
  same-row negatives) — the measured numbers above already reflect this batch structure,
  so they are faithful to what ships, but a cleaner ablation is unbuilt.
- **Shared-draw dropout gap** (Shen et al. 2019) — named, not measured, not closed.
- **Collapse.** The winner-share entropy instrument (`fan_wta_latent_entropy`) exists but
  has not been read on a real run; nothing yet confirms the InfoNCE pick avoids the
  JEPA-paradox / centroid-collapse failure mode the scoring choice was partly meant to
  guard against (a generic cell reading close to many spans could still win if it also
  reads close to its own span more than the true winner does).
