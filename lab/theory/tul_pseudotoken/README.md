# tul_pseudotoken

A Lean 4 development of how a PonderLM- or CODI-shaped carrier can be trained on LXTUL
WITHOUT thinking traces, and what such a carrier does to loop contribution.

It sits beside [`../tul_information/`](../tul_information/) (a deterministic pass adds no
information; a loop earns only by a relay or by extractability) and
[`../tul_exploration/`](../tul_exploration/) (when width and depth are necessary; T4's lookup
model). Both are path dependencies and both are reused: `Bypass.lean` is built on
`Joint.mi_map_le` and `Joint.mi_relay_redundant`, and `Teacher.lean` on `Prog.run_congr`. The
note that uses these theorems is
[`.agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md`](../../../.agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md).
The paper reading is
[`docs/references/tul-latent-emission/ponderlm/ponderlm.md`](../../../docs/references/tul-latent-emission/ponderlm/ponderlm.md).

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH
cd lab/theory/tul_pseudotoken
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake build
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake env lean Axioms.lean
```

Recorded 2026-10-05 after deleting `.lake/build` and rebuilding: `lake build` exits 0 with
"Build completed successfully (2098 jobs)", zero warnings and zero errors in the full log.
`lake env lean Axioms.lean` exits 0 and prints 36 lines. 34 end `depends on axioms: [propext,
Classical.choice, Quot.sound]`; 2 (`transcript_replay`, `distilled_carrier_le`) end
`[propext]`. No line mentions `sorryAx`, and `grep -rn sorry TulPseudotoken/` finds nothing.

`taskset -c 0,1` holds the build to two cores: the UPS trips when the GPU is loaded and the
CPU joins it. That is a house rule, not a Lean requirement.

Mathlib `v4.31.0`. `.lake/` is a symlink to `/home/wolfe/lean-build/tul_pseudotoken/.lake`,
whose `packages/` was copied from `tul_exploration`'s. The two path dependencies build into
their own `.lake/` directories, so `../tul_information/.lake` and `../tul_exploration/.lake`
must exist and should live on `/home`, not on the spinning disk. In the main checkout they
are already symlinks. In a fresh worktree, copy or link their `build/` directories from
`/home/wolfe/lean-build/` before the first build.

## The model

Everything is either finite or finite-dimensional linear algebra. Four pieces.

* **A carrier position** (`Carrier.lean`) is a vector `v` in the coda's input space. The coda
  reads it after an RMS norm, so token `i` sees the cosine `strength e v i = ⟪v, e i⟫ / ‖v‖`
  with its tied embedding `e i`. A raw token at that position has strength `1` along itself.
  A span's candidate tokens (at most 32) are modelled as orthonormal. A FREE carrier is any
  `v` (CODI, Coconut, LXTUL's `W_prefix` write). A VOCAB MIXTURE is `∑ p i • e i` with
  `p ≥ 0` (PonderLM). A SNAPPED mixture takes `p = softmax(ℓ)` with logits read from a cell.
* **A copy task** (`Signals.lean`): later reads `r` copy span position `src r`, sit `hz r`
  spans ahead, and save `w r` nats when the carrier holds the source. Each trace-free signal
  in LXTUL is a function of the carried set `S`.
* **A teacher** (`Teacher.lean`) is the coda reading the raw context. For its LOGITS: a
  calibrated conditional law. For its COMPUTATION: a `TulExploration.Prog`, an adaptive
  lookup program over the context table, with the coda's fixed path of lookups.
* **A bypass** (`Bypass.lean`): `tul_information`'s `Joint` with a paired first coordinate,
  `(view, entry)`. The view is what the coda reads whatever the pass count is; the loop state
  after `n` passes is `F^[n] entry`.

## What is proved

| result | file:line | statement |
|---|---|---|
| `abs_strength_le_one` | `TulPseudotoken/Carrier.lean:72` | a position's strength along any token is at most 1 |
| `strength_eq_one_iff` | `Carrier.lean:83` | strength 1 along `i` iff the position is a positive multiple of `e i` |
| **`sum_sq_strength_le_one`** | `Carrier.lean:101` | Bessel: one position's squared strengths over a span sum to at most 1 |
| **`card_strong_mul_sq_le_one`** | `Carrier.lean:117` | at most `1/τ²` tokens are read at strength `τ` or more from one position |
| `vertex_mixture_eq` | `Carrier.lean:155` | a mixture with weight only on `y` is `p y • e y`, the raw token up to scale |
| **`mixture_strength_eq_one_iff`** | `Carrier.lean:165` | a non-negative mixture has strength 1 along `y` iff every other weight is 0 |
| `affine_write_card_le` | `Carrier.lean:203` | an affine write's outputs with independent differences number at most `dim V` |
| **`exact_tuple_write_needs_dim`** | `Carrier.lean:234` | an affine write of `N` exact tokens, each from `r + 1` affinely independent tokens, needs `N r ≤ dim V` |
| `softmax_ge`, `one_sub_softmax_le` | `Carrier.lean:280`, `:308` | margin `Δ` puts at least `1/(1 + (n-1)e^{-Δ})` on `y`; mass off `y` at most `(n-1)e^{-Δ}` |
| **`snap_close`** | `Carrier.lean:327` | the snapped mixture is within `2 (n-1) e^{-Δ}` of `e y` (embeddings of norm ≤ 1) |
| **`exists_scale_commit`** | `Carrier.lean:357` | with a free scale, one snap puts `1 - ε` on the argmax, any `ε > 0` |
| `softmax_le` | `Carrier.lean:392` | logit lead at most `M` caps the mass on `y` at `1/(1 + (n-1)e^{-M})` |
| **`passes_to_commit`** | `Carrier.lean:420` | margin built over `T` passes at most `m` each: `p y ≥ 1-ε` needs `T m ≥ log((n-1)(1-ε)/ε)` |
| `ceValue_eq_next_add_far` | `TulPseudotoken/Signals.lean:78` | the coda-CE value is the next-span part plus the far part |
| **`next_blind`** | `Signals.lean:90` | if no copy is one span ahead, the next-span signals are 0 on every carrier |
| **`rec_misaligned`** | `Signals.lean:105` | the reconstruction optimum can be a token nobody copies (CE value 0) |
| **`aux_price`** | `Signals.lean:126` | an auxiliary in `[0, R]` at weight `λ` costs at most `λ R` of CE value at the optimum |
| `kdLoss_orth` | `Signals.lean:155` | orthogonal shifts: CODI's hidden-state loss is `∑_{i ∉ S} ‖f i‖²` |
| **`kd_misaligned`** | `Signals.lean:165` | hidden-state distillation strictly prefers the token with the bigger shift, CE-worthless or not |
| **`kd_mean_eq`** | `TulPseudotoken/Teacher.lean:50` | logit distillation from a calibrated teacher is CE in expectation |
| **`kd_variance_le`** | `Teacher.lean:75` | its spread around any centre is at most CE's (loss or gradient coordinate) |
| **`transcript_replay`** | `Teacher.lean:110` | holding the `cost f p` entries a teacher program read reproduces its output with zero passes |
| `distilled_carrier_le` | `Teacher.lean:118` | a teacher with at most `Lc` lookups needs a carrier of at most `Lc` entries |
| **`bypass_zeroes_passes`** | `TulPseudotoken/Bypass.lean:47` | a sufficient pass-independent view makes the loop state after any passes worth exactly 0 |
| **`deeper_never_more_informative`** | `Bypass.lean:61` | beside any view, the state after `k + n` passes carries at most the state after `n` |
| `raw_span_sufficient` | `Bypass.lean:74` | `Joint.mi_relay_redundant` for the carrier: raw span plus view beats any computed summary |

Helper lemmas (`inner_mixture`, `norm_sq_mixture`, `slot_change_linearIndependent`,
`softmax_nonneg`, `sum_softmax`, `sum_exp_split`, `ceValue_empty`, `norm_sq_sum_orth`,
`sq_wsum_le`) are listed in `Axioms.lean`.

## The theorems in plain English

**P1, one position holds one exact token, and only a snap gets several out of one cell
(`Carrier.lean`).** The coda normalises what it reads, so a carrier position is a direction.
Bessel says its squared cosines with a span's tokens sum to at most one. So one position
carries ONE token at full strength. Superposing `k` tokens leaves each at `1/√k` at best,
free vector or mixture alike. For a non-negative mixture the strength along `y` is
`p y / √(∑ p i²)`, which is one exactly at the vertex `p = δ_y`, and the vertex IS the raw
token's input up to scale. The exact-copy optimum of the free class is that same vertex, so
for an exact copy the vocab mixture loses nothing against a free vector, and the extractive
cell (A3) is the vertex of the pondering mixture (A2). The decisive part is the write.
`exact_tuple_write_needs_dim`: if one cell feeds `N` coda positions through any affine map,
and each position must show an exact token chosen from `r + 1` tokens with independent
differences, then `N r ≤ dim cell`. A real tied table spans its space, so `r = d` and a
`d`-wide cell gives at most one exact arbitrary token through a linear write. That covers
`W_prefix`, the wider read-out (`tul.prefix_per_cell`) and every CODI/Coconut hidden-state
carrier. The snapped mixture escapes it: `snap_close` puts the written vector within
`2 (n-1) e^{-Δ}` of the exact embedding as soon as the logits put the right token first by a
margin `Δ`, whatever else the cell holds. The tied table is a free codebook for a
nearest-token clean-up. Loses: orthonormal candidates for the strength statements (real
embeddings are only near-orthogonal); exactness, not approximate exactness, in the write
theorem; the coda's attention is not modelled beyond "it reads a normalised input".

**P1b, commitment is a one-pass job unless the scale is bounded (`Carrier.lean`).** With a
learnable logit scale, one snap puts `1 - ε` on the argmax (`exists_scale_commit`). If the
scale per pass is bounded (a fixed-gain RMS norm on the cell, no temperature) and the logits
accumulate over passes, commitment needs `T m ≥ log((n-1)(1-ε)/ε)` (`passes_to_commit`). For
a 16-token span and `ε = 0.05`, that is `T m ≥ 5.65`. So depth is necessary for commitment
only if it is made so by construction.

**P2, the coda's own CE is the only trace-free signal that pays for far copies
(`Signals.lean`).** The CE through the carrier is summed over every later span that reads
it. The span decoder and the latent rank head score against the next span only, so they see
the horizon-1 part of that value and are flat on the rest (`next_blind`). The exact-recall
probe puts 41 % of LXTUL's gap at 65 to 256 tokens back, beyond the next span. Own-span
reconstruction ignores the future: its optimum can be a token nobody copies
(`rec_misaligned`). CODI's hidden-state loss ranks a token by the size of the hidden-state
shift it causes, not by the CE it saves (`kd_misaligned`). MORPH's residual stream has
massive channels, so a magnitude ranking is a real risk here. An auxiliary bounded in
`[0, R]` at weight `λ` costs at most `λ R` of CE value at the optimum (`aux_price`): helpers
are safe when `λ R` is small, and dangerous as main signals. Loses: reads are separable
(each needs one source token and saves a fixed amount); a set-valued carrier, justified by
P1, not derived from training.

**P3, a raw-context teacher can change the speed, never the target, and never the depth
(`Teacher.lean`).** Logit distillation from a calibrated teacher has the CE objective in
expectation (`kd_mean_eq`), so the same optimum, and never a larger variance
(`kd_variance_le`, for the loss and for any gradient coordinate). Its only possible gain is
optimisation speed. For depth, `transcript_replay`: whatever a teacher program computes over
the raw context, a student whose carrier holds the `cost f p` table entries the teacher read
reproduces it exactly with zero loop passes. A teacher with the coda's own fixed path asks
the carrier to HOLD entries, never to COMPUTE (`distilled_carrier_le`). So a CODI-style
teacher in LXTUL cannot make the loop's depth necessary. Loses: the teacher is idealised as
calibrated (a real teacher pulls toward its own errors); the transcript must be selected
without knowing the future query, which is P2's selection problem.

**P4, a carrier that delivers content beside the loop takes that content's share of loop
contribution (`Bypass.lean`).** If the coda's pass-independent view is sufficient for a
target, the loop state after any number of passes is worth exactly zero beside it
(`bypass_zeroes_passes`). Without sufficiency, a deeper state never carries more than a
shallower one beside the same view (`deeper_never_more_informative`). So a positive K1−K6 is
never information: it is the bounded coda extracting better from the deeper state. A snapped
pseudo token makes copy content as extractable as a raw token. Once pass 1's pseudo tokens are
right, the deeper states have nothing left to make easier on that content, and K1−K6 on copy
tokens should fall toward zero. Loop contribution can survive only on content the view does
not make as extractable. `raw_span_sufficient` adds that raw copies of the span dominate any
computed summary of it once the carrier is wide enough. Loses: information is not
extractability. These theorems bound what a Bayes reader gains; the prediction for the real
coda's K-curve is a reading, tested by the note's Stage 0 and the arm's per-bucket K-sweep.

## What is assumed, and what is NOT proved here

* **The bridge to MORPH's code is by description.** Nothing here reads
  `morph/model/transformer.py`. That the coda's read is a normalised linear read, that the
  span decoder and the latent rank head look one span ahead, and that a copy read saves a
  fixed amount are modelling claims in the note.
* **Copy selection is shallow, by argument, not by proof.** P3 shows that what a same-depth
  teacher needs is held, not computed. Whether the RULE that picks which tokens to carry
  composes context relations deeper than one pass is not proved. If the rule is a fixed
  function of the token (`TulExploration.fixed_table_no_lookups`) or one lookup into earlier
  cells, it is shallow. If it composes deeper relations, T4's `passes_needed` makes depth
  necessary. Web text was not measured for this.
* **No attention model.** "Retrieved by attention" is reduced to "read after a norm through a
  linear key and value". The softmax over many carrier positions, and the margin a copy head
  needs against distractors, are not modelled.
* **No optimisation theory.** Nothing here says that gradient descent finds the CE optimum.
  The gradient argument for the mixture (`∂L/∂p_i = ⟪∇_t L, e_i⟫`, every candidate's
  first-order value) is calculus in the note, not a theorem here.
* **Finite laws, one context at a time**, as in both parent packages.
