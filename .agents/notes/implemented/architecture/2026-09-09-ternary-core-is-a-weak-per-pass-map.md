# Agent Note: the ternary core is a weak per-pass map, and the norm-matched scale is the recipe's ternary rule

Status: implemented

## Problem

Two panels on 2026-09-09 closed every other candidate for why MORPH's looped core converges
by pass 3 and contributes 0.03 nats where Parcae's contributes 0.29:

- The Parcae-entry panel (`lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md`)
  rebuilt the loop's entry the way Parcae does it: state from noise, e re-injected on every
  dimension through a learned B, no fixed-point term. It gave a 0.023-nat better model with a
  start-independent fixed point, and the loop still contributed 0.033 past pass 1 and 0.0009
  past pass 3. Its depth-1 control sits within 0.02 nats of the looped plain model at depth 6
  at 5,000 steps; that is a short-horizon reading (deep models converge slower), not a ranking.
- The depth-candidates panel (`lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md`)
  ran Parcae's depth schedule (K3−K8 0.002), a carry that trains at 20× the rate (the carrier
  collapses to rank 6, K3−K6 0.007), and a diagnostic arm with ternary everywhere except the
  looped core. Only that arm moved the loop's contribution: past pass 1 0.033 → 0.168, past
  pass 3 0.0009 → 0.012.

The anatomy named the mechanism. At iteration 6 the bf16 core's MLP branches emit 0.75–0.96
of their input; the ternary core's emit 0.19–0.24. A pass through the ternary core barely
changes the state, so the state sits on its fixed point by pass 3 and the coda has nothing
more to read. On the Parcae-entry checkpoint the core MLPs' latent weights are
Gaussian-shaped (absmean over std 0.80), 30.5 % of them sit in the dead zone, and under the
BitNet b1.58 absmean rule (γ = mean|W| over ALL entries, threshold 0.5) every ternary layer's
Frobenius norm is 0.668 of its latent's, so a gate-up-down MLP emits about 0.44 of its bf16
twin from the scale alone.

Ternary is not negotiable (Wolfe's standing rule: no dense-then-ternary; ternary weights
organize differently). So the question was how a ternary core can be a strong per-pass map.

## Decision

`training.ternary_scale_mode: norm_match` is the recipe's ternary rule (`base.yaml`,
`scale30b.yaml`). The rule: the symmetric codes (`sign(w) · [|w| / mean|W| > 0.5]`, unchanged)
with the per-tensor scale `‖W‖_F / √nnz`, so the ternary weight has the latent weight's
Frobenius norm. No parameters, one scalar per tensor, the same packed format as before. The
nearest prior is Ternary Weight Networks (Li et al. 2016, arXiv:1605.04711, Eq. 5: the optimal
scale is the mean magnitude over the NONZERO set; norm_match is its root-mean-square
analogue, which matches the norm exactly). Reference:
`docs/references/training-objectives/ternary-weight-networks/ternary-weight-networks.md`.

The rule lives in ONE place, `morph/model/ternary_rule.py::ternary_codes_and_scale`, and three
paths call it so they cannot drift:

- the training STE (`TernarySTE._forward_norm_match`; the `symmetric` mode keeps its own
  vectorised grouped path, which the shared function reproduces bit-for-bit),
- the carved MORTAR path (`CMSBlockLinear._mortar_effective_data`). Before this change the
  carve at `compact_step` re-registered a HARDCODED absmean STE on `mortar_data`; a
  norm-matched recipe would have silently reverted to absmean at step 29000. The carve now
  captures the dense STE's threshold and mode, persists them in the `mortar_ternary` buffer
  (`[flag, threshold, mode code]`; a pre-2026-09-09 two-element buffer loads as symmetric), and
  refuses the learnable modes before touching any layer,
- the deploy packer (`inference/deploy_quant.py`). `extract_ternary_from_parametrized` and
  `PackedTernaryLinear.from_parametrized` read the mode and threshold off the parametrization;
  `pack_mortar_ternary` packs a carved layer at the rule it TRAINED under and raises on an
  explicit argument that contradicts it. `to_deploy_inference` / `ensure_deploy_packed` no
  longer pass a threshold to trained layers.

Measured on the per-pass-strength panel
(`lab/experiments/successes/2026-09-09-arc-per-pass-strength.md`, three one-factor arms on the
Parcae-entry recipe, 5,000 steps, seq 1024):

| arm | K1−K6 | K3−K6 | core MLP out/in at pass 6 | state movement per pass (%) | depth-6 CE at 5k |
| --- | --- | --- | --- | --- | --- |
| absmean base (`parcae-entry`) | 0.033 | 0.0009 | 0.19–0.24 | 37/23/7/5/3/2 | 3.957 |
| `threshold-03` (dead zone 0.3, absmean scale) | 0.026 | 0.0004 | 0.19–0.24 | 47/18/8/4/3/2 | 4.019 |
| `scale-ttq` (learnable γ₊/γ₋) | 0.041 | 0.0018 | 0.08–0.14 | 42/17/9/5/3/2 | 3.984 |
| `scale-norm-match` | **0.185** [0.182, 0.188] | **0.0139** [0.0131, 0.0146] | **0.86–1.10** | 61/28/16/9/6/4 | 4.039 |
| bf16 core (`depthcand-dense-core`, diagnostic) | 0.168 | 0.0119 | 0.75–0.96 | 63/38/20/12/9/6 | 3.928 |

Three facts decide it. The number of nonzero weights is not the lever (the dead-zone arm
reads as the base). A free scale is not the lever: the optimizer SHRINKS the learnable scales
(core mean γ over mean|W| 0.874 at 5,000; gate-up 0.50–0.68, down 0.79–1.40), and its core
MLP branch reads 0.08–0.14, below the base. The scale has to be imposed by the rule, and the
imposed norm-matched scale puts a ternary core above the bf16-core diagnostic on every
contribution instrument.

The CE at 5,000 steps is 0.08 higher than the base on 491,520 paired tokens. That is a
horizon reading of a deeper map, not a ranking (`short-horizon-ce-is-not-a-verdict`); the
learnable-scale arm shows the loss gradient at this horizon prefers the weaker map, so the
stronger one is a bet on the long horizon that Wolfe took on 2026-09-09 ("the norm matched
seems to be our solution ... flip base.yaml to norm_match").

## Alternatives considered

- **A dense (bf16) core in production.** Rejected: it was the diagnostic that found the
  mechanism, but ternary is the recipe the project exists for and Wolfe's rule forbids a
  dense-then-ternary path. norm_match now beats it on the contribution instruments anyway.
- **Learnable ternary scales (`ttq`).** Measured and rejected as the rule: the optimizer
  shrinks them, and the parametrization has no positivity constraint (several `x0_injects`
  scales went negative or reached 20× absmean at 5,000). Kept as a training-only mode; the
  carve and the packer refuse it.
- **A narrower dead zone (`ternary_threshold: 0.3`).** Measured and rejected: more nonzero
  codes under the absmean scale change nothing.
- **Finer scale groups on the core (`ternary_scale_group: 128`).** Not run; norm_match per
  tensor already restores the branch, and a per-tensor scalar is the cheapest export.
- **Muon or a higher base learning rate for the core.** Deferred: the carry-rate arm showed a
  faster rate alone collapses the state's rank without adding contribution.
- **Parcae's depth schedule.** Rejected by measurement (K3−K8 0.002).
- **Codes-side fixes (Tequila / SZT dead-zone reactivation).** Not needed: the dead-zone arm
  says the codes are not the limiter at this threshold.

## Consequences

- Every new run under `base.yaml` trains, carves and exports under norm_match. Old
  checkpoints trained under `symmetric` still load and export under `symmetric` (the mode is
  read off the parametrization or the persisted buffer, never assumed).
- `morph/model/ternary_rule.py` is the only place the rule is written. A new scale rule is
  added there, appended to `EXPORTABLE_SCALE_MODES` (the persisted code is its index), and
  covered by `tests/test_ternary_rule_carve_export.py` (STE, carve, checkpoint round trip,
  packer) and `tests/test_ternary_norm_match.py`.
- The learnable modes (`ttq`, `dual`) raise at the carve and at export. A recipe that carves
  cannot use them.
- Unmeasured and open, in priority order: the paid TUL loop under norm_match (every strength
  arm ran the plain model; the recipe runs TUL); the 20k horizon against the existing 20k
  control (the 0.08 gap at 5k is the price of a deeper map until shown otherwise; Wolfe's
  launch); the bf16-everywhere ceiling arm (`precision-bf16-all`, queued); why even a strong
  map converges by pass 6 (K3−K6 0.014 against Parcae's K3−K8 0.040: block and stream count,
  optimizer, tokens are the remaining candidates).
- Any detonation on a norm_match run: the gain is 1.5× on every ternary layer from step 0
  under the 1000-step ramp; the sustained tripwire and `lab/divergence/DIVERGENCE-README.md`
  apply. The three 5k arms were HEALTHY (probe max 22 at step 340 on norm_match).
