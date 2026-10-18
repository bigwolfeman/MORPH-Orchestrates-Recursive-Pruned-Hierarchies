# The Dynamics of Continuous Mixture Collapse: reading and local cache

Read 2026-10-18, in full: all 13 pages, including Appendix A (benchmark) and Appendix B
(the proofs of Theorems 1 to 3 and Remark 1). Read for the coarse-level particle design on
the Parcae spiral testbed (Lean claims in
[`lab/theory/tul_exploration/TulExploration/Particles.lean`](../../../../../lab/theory/tul_exploration/TulExploration/Particles.lean)).

## Citation (verified)

Ali Backour. *The Dynamics of Continuous Mixture Collapse in Language Models*.
[arXiv 2609.02049v1](https://arxiv.org/abs/2609.02049), posted 2026-09-02. Single author,
MIT. Categories cs.LG (primary) and cs.CL. Marked "Preprint. Under review." No venue.

Verified 2026-10-18 against the arXiv export API
(`https://export.arxiv.org/api/query?id_list=2609.02049`): the title and the one author,
Ali Backour, match. The PDF title page matches.

## Local cache

- [PDF in this folder](continuous-mixture-collapse-2609-02049.pdf)
- [Private PDF](../../../../../ignore/papers/2609.02049v1-continuous-mixture-collapse.pdf)
- [Private extracted text](../../../../../ignore/papers/2609.02049v1-continuous-mixture-collapse.txt)
  (`pdftotext -layout`)
- 630,317 bytes, 13 pages, 671 text lines.
- SHA256: `e46d7fce670154c32b4e431d8e1b4d58dcae1af69ee05196c41967ce061b1e1d`.

## What it actually does

It asks why pretrained LLMs fail to keep a soft token (a convex mix of token embeddings) as
a mix. It names three independent causes:

    cause                          where            evidence
    architecture distorts mixes    Section 2        random-init controls, Table 1
    training amplifies it          Section 2        trained vs matched controls, Table 1
    softmax readout + AR feedback  Sections 3, 4    Theorems 1, 2; Figures 3, 4
    (even with linear transport)
    K-way correction is costly     Section 5        Theorem 3 (no experiment)

## The key results

### Section 2: geometry (measured)

- A 50 / 25 / 25 blend of three colour words in one slot gets 99.5 % "red" averaged over six
  triples and ten wordings (Section 2). Figure 1: over 325 three-way mixes, Qwen3.5-4B and
  Gemma-4-E4B-it map most of the simplex interior to one near-pure answer, with narrow
  boundaries.
- Recovered weight `alpha(w)` (Eq. 1) and the score CALIB (Eq. 2: 1 for exact
  preservation, 0 for a hard threshold at w = 0.5). Table 1, final layer, trained against
  the mean of five random inits of the same architecture: Qwen3.5-0.8B 0.632 / 0.818,
  Qwen3.5-4B 0.332 / 0.732, Qwen3.5-27B 0.130 / 0.611, gemma-4-E4B-it 0.475 / 0.765,
  gemma-4-31B-it 0.160 / 0.569. Trained is always worse and the gap grows with scale.
  Across depth, Qwen loses it toward the output; Gemma is non-monotone.

### Section 3: the recursion (theory)

Two components A and B, two tokens. Mixture weight `w_t` on A, recentred as
`u_t = 2 w_t - 1` (Eq. 4). Each pure branch's log-odds between the two successors are
`Delta_A,t` and `Delta_B,t` (Eq. 5). Then (Eq. 6):

    L_t = (Delta_A,t - Delta_B,t) / 2      coupling: how differently the branches rank
    b_t = (Delta_A,t + Delta_B,t) / 2      field: the offset both branches share

If the mixed final hidden state is carried EXACTLY linearly, the logits mix linearly, the
softmax turns that into a geometric mixture `q_w ∝ q_A^w q_B^(1-w)` (Eq. 3), and feeding it
back gives (Eq. 7):

    u_{t+1} = tanh( (b_t + L_t u_t) / 2 )

For fixed `b, L` this is the Curie-Weiss mean-field iteration. With `b = 0`, the slope at
`u = 0` is `L / 2`, so the balanced state is stable for `L < 2` and unstable for `L > 2`.
Above 2 there are two stable states `+-u*(L)`, `u* = tanh(L u* / 2)` (Eq. 9). `L = 4` gives
`u* ≈ 0.96`, `L = 8` gives `0.9993`.

**Theorem 1 (uniform coupling implies persistent polarisation).** Let `u_0 ≠ 0`,
`b_t = 0` for all `t`, and `L_t >= L_min > 2` for all `t`. Then `sign(u_t) = sign(u_0)` for
all `t`, and `liminf |u_t| >= u*(L_min)` (Eq. 10). Proof: monotone comparison with the
constant-`L_min` system (Appendix B).

**Theorem 2 (subcritical coupling washes out the mixture).** If `0 <= L_t <= L_max < 2`
along every rollout, then

    |u_T| <= (L_max/2)^T |u_0|  +  (1/2) sum_{t=0}^{T-1} (L_max/2)^(T-1-t) |b_t|

and if `b_t = 0`, `u_t -> 0` from any start. Proof: `|tanh x| <= |x|` (Appendix B).
**Remark 1:** if `|b_t| <= b_max`, then `limsup |u_T| <= b_max / (2 - L_max)`. So a
persistent field holds the state away from 50 / 50, but where it holds it is set by the
field, not by `u_0`.

Note what `u -> 0` means. It is the 50 / 50 state, not one component winning. Below the
threshold every starting mix ends at the same place, so 70 / 30 and 30 / 70 become
indistinguishable (Section 3.2). Above it the initial majority wins. Both lose the
information in the mix.

### Section 4: the recursion in real rollouts (measured)

The two pure branches run under the model with greedy successors (Eq. 11). The mix exists
only in logit space between them, so transport is exactly linear by construction.
Figure 3 (1000-item benchmark, 50 steps, medians): starts just above and below 1/2 split
to opposite branches. `L_t` stays mostly above 2 in both families and `|b_t|` is much
smaller. Figure 4: a feedback temperature `tau` sets `L_eff = L_t / tau`. Sweeping it moves
the same items from polarisation to washout, and the turn is near `L_eff = 2`.

### Section 5: the K-way correction cost (theory)

Log-ratio coordinates `a = psi(p)` (Eq. 12) and one decoding step `a_{t+1} = Phi(x_t, a_t)`
(Eq. 13), with `x_t` the context representation. A corrector `f(a, r(x))` with
`r(x) in R^m` must satisfy `Phi(x, f(a, r(x))) = a` (Eq. 16). **Theorem 3:** if it keeps even
one interior mix `a_bar` fixed for all `x` near `x0` (Eq. 17), with `Phi, f, r` all C^1, then
`m >= rank D_x Phi` (Eq. 18), and `m >= K - 1` when the context moves the output in all
`K - 1` mixture directions (Eq. 19). Chain-rule proof (Appendix B). The full-rank condition
is not measured (Section 7).

### Section 6: what the author draws from it

Coconut-style hidden-state feedback removes the recursive-softmax source but "is not by
itself sufficient". Damping the gain (SeLaR, Mixture of Inputs) trades polarisation for
washout, and Theorem 2 says that is the trade to manage. A general K-way fix likely needs
a context-conditioned controller whose capacity grows with K.

## Where it sits for the coarse-level particles

### Does it apply to a hidden-state loop with injection?

Not as stated. Eq. 7 comes from the vocabulary softmax plus autoregressive feedback, with
the hidden state assumed linear. Our coarse loop `x^{t+1} = F(D x^t + Delta B z)` feeds the
hidden state back with no vocabulary softmax in the loop. Section 6 names that route
(Coconut) as outside this source of collapse. Section 2's geometry result does apply to any
trained transformer stack in the loop (inference).

What transfers is the shape of the Theorem 2 proof (inference). It is a contraction with a
bounded additive forcing. For K particles run through the SAME map with the SAME injection
`Delta B z`, the injection cancels in the difference between two particles, so their
spread shrinks by the map's Lipschitz constant per pass. That is Lean C1. The injection
plays the role of the field `b_t`: it sets where the particles go together, and it cannot
hold them apart (Remark 1). The coupling `L_t` maps to the loop's gain along the direction
that separates two basins.

### Collapse to one, or one component takes over?

The paper's theorems are about ONE state that holds a two-way mix. They say nothing direct
about K separate states. For one mixed state: below threshold it washes out to the
balanced state; above threshold with zero field, the initial majority takes over.

For K independent particles (inference): the paper's takeover mechanism needs the mix
weights to feed back through a softmax. Our particles do not read each other, and the coda
reads them teacher-forced, so nothing feeds the coda's attention weights back into the
loop. That channel is absent by design. What remains is the per-particle analogue: under
contraction all particles fall into one common state, and with an expansive direction plus
saturation each particle falls into the basin its random start points at, as in Theorem
1's "the sign of `u_0` decides". That basin choice comes from the start, not from the
context, unless the field breaks the symmetry.

### Multiple stable mixtures?

Only in one case: two components, `b = 0`, `L > 2`. Then exactly two stable states `+-u*`
exist and the balanced state is unstable (Eq. 9, Theorem 1). Both are polarised. For `L`
just above 2, `u*` is small, so the stable states are mixes near 50 / 50, but their weight
is set by `L` and they carry one bit about the start. Nothing is proved for a nonzero field
above threshold. In the standard Curie-Weiss picture (not in this paper) a large enough
field removes one of the two states. Nothing is proved for K > 2 stable states. Figure 1
is a one-shot injection with K = 3 and shows three near-pure basins.

## What it does NOT establish for us

- It does not analyse a hidden-state loop, an injected input, or K separate states.
- Theorem 1 needs `b_t = 0` exactly. Theorem 2 needs `L_t >= 0`. Negative coupling and
  nonzero field above threshold are not covered.
- The empirics are binary, on two model families, with the mix formed in logit space. No
  downstream accuracy is measured (Section 7).
- Theorem 3 is a local, exact-preservation bound. It says nothing about approximate
  preservation or about how much a learned controller actually needs.
