# Agent Note: A trace-free pseudo-token carrier for LXTUL, and what it will do to loop contribution

Status: proposed

## Problem

LXTUL ([decision record](../../implemented/architecture/2026-10-04-lxtul-primary-candidate.md))
trails a plain model at the same step by +0.272 nats at 5k and +0.356 at 10k. The
[exact-recall probe](../../../../lab/experiments/successes/2026-10-04-exact-recall-gap-probe.md)
put 65 % of the 10k gap on bigram repeats 33 to 256 tokens back: 15 % of the tokens, the ones
plain copies through its raw 256-token window. The coda reads ONE live vector per earlier
span (the final winner through `W_prefix[winner]`; the three losers' prefix positions are
exactly zero), and a vector is a poor container for exact tokens.

Wolfe's next line, in order: A2, a pseudo-token write (PonderLM-style probability-weighted
embedding mixtures); B, pack more into cells; A3, extractive cells; A1, a raw local window.
He asked for something "ponder/codi shaped ... that isn't trained on thinking traces", worked
out in Lean, and warned: "Testing the A line there needs to be very careful to measure loop
contribution, we may kill it." Loop contribution (K1−K6, CE at one pass minus CE at six) is
+0.0215 / +0.0172 at 5k (two seeds) and +0.0236 at 10k, and it is the first priority. An arm
that closes the CE gap and kills K1−K6 is a failure.

The seed the orchestrator brought was CODI-shaped: use the coda that reads raw cross-span
context as a teacher and distil it into the coda that reads only the carrier. Wolfe's verdict
on that: "a massive pain in the asshole to use". This note treats it as the fallback and asks
first whether the simplest trace-free signal is enough.

## Proposal

### What was proved

Lean 4, Mathlib v4.31.0, in [`lab/theory/tul_pseudotoken/`](../../../../lab/theory/tul_pseudotoken/).
After a clean rebuild `lake build` exits 0, 2098 jobs, zero warnings. `lake env lean
Axioms.lean` prints 36 lines: 34 depend on `[propext, Classical.choice, Quot.sound]`, two on
`[propext]` alone, none on `sorryAx`. The package reuses `tul_information`
(`Joint.mi_map_le`, `Joint.mi_relay_redundant`) and `tul_exploration` (`Prog.run_congr`). Its
README gives file:line for every result and what each model loses. The reading of PonderLM is
in [`docs/references/tul-latent-emission/ponderlm/ponderlm.md`](../../../../docs/references/tul-latent-emission/ponderlm/ponderlm.md).

The results that decide the design, in one line each:

| theorem (file) | what it says | what it decides |
| --- | --- | --- |
| `sum_sq_strength_le_one`, `card_strong_mul_sq_le_one` (`Carrier.lean`) | one normalised position holds at most one token at full strength; `k` superposed tokens get at most `1/√k` each, free vector or mixture | N exact copies need N positions |
| `mixture_strength_eq_one_iff`, `vertex_mixture_eq` (`Carrier.lean`) | a vocab mixture is exact only at a vertex, and the vertex IS the raw token's input | A3 is the vertex of A2: build one design, not two |
| `exact_tuple_write_needs_dim` (`Carrier.lean`) | an affine write of one cell into N positions showing exact arbitrary tokens needs `N·d ≤ dim(cell)` | no linear read-out (`W_prefix`, `tul.prefix_per_cell`, a CODI/Coconut hidden state) gives a coda two exact tokens from one cell |
| `snap_close` (`Carrier.lean`) | a softmax over the tied table writes a vector within `2(n−1)e^{−Δ}` of the exact embedding once the argmax is right by margin `Δ` | the PonderLM mixture is the delivery mechanism the linear write lacks |
| `exists_scale_commit`, `passes_to_commit` (`Carrier.lean`) | a free scale commits in one snap; a bounded per-pass scale needs `T·m ≥ log((n−1)(1−ε)/ε)` | keep the scale learnable; do not bound it to manufacture depth |
| `ceValue_eq_next_add_far`, `next_blind` (`Signals.lean`) | the span decoder and the latent rank head see only copies one span ahead | they cannot pay for 65-256-token copies (41 % of the gap) |
| `rec_misaligned` (`Signals.lean`) | own-span reconstruction's optimum can be a token nobody copies | reconstruction is not the main signal |
| `kd_misaligned` (`Signals.lean`) | CODI's hidden-state loss ranks tokens by the size of their hidden-state shift | no hidden-state distillation (MORPH has massive channels) |
| `aux_price` (`Signals.lean`) | an auxiliary in `[0, R]` at weight `λ` costs at most `λR` of CE value at the optimum | any helper term stays small |
| `kd_mean_eq`, `kd_variance_le` (`Teacher.lean`) | logit distillation from a calibrated teacher is CE in expectation, with no more variance | a teacher can only speed learning; it never changes what to carry |
| `transcript_replay`, `distilled_carrier_le` (`Teacher.lean`) | a carrier holding the entries a same-depth teacher read reproduces it with zero passes | a raw-context teacher cannot make loop depth necessary |
| `bypass_zeroes_passes`, `deeper_never_more_informative` (`Bypass.lean`) | a sufficient pass-independent view makes every pass worth exactly 0; beside any view, deeper never carries more | K1−K6 is extractability only, and a copy carrier takes the copy share of it |

### The answers to the three questions

**(a) Is the plain trace-free signal enough for copy content?** Yes, at the level these models
reach. The coda's own CE through the carrier is summed over every later span that reads it,
so it is the only signal already in LXTUL that pays for a copy 4 to 17 spans ahead
(`ceValue_eq_next_add_far`, `next_blind`). PonderLM trains with nothing else (plain CE on the
last step) and beats a Coconut-style hidden-state feedback at equal steps (Pile perplexity
14.16 against 15.64 at three steps, PonderLM Table 2). A teacher adds nothing to the target
(`kd_mean_eq`). Capacity: one exact token per position (`card_strong_mul_sq_le_one`), so a
three-position carrier holds three exact tokens per span. For exact retrieval the vocab
mixture loses nothing to a free vector (`mixture_strength_eq_one_iff`), and the snapped
mixture can deliver several exact tokens out of one cell where any linear write cannot
(`snap_close` against `exact_tuple_write_needs_dim`). A peaked mixture is a token embedding,
so the CE optimum, given span-restricted candidates, is a set of the span's own tokens: copies,
not next-token guesses. With full-vocabulary candidates `raw_span_sufficient` still makes raw
copies optimal once the carrier can hold the span; below that capacity the CE trades a copy
against a computed token slot by slot.

What the plain signal may lack is speed, not direction. CODI and Coconut needed a teacher or
a curriculum because their latents must COMPUTE a chain of results. A copy carrier only has to
SELECT among visible candidates, and the gradient on each candidate's weight is
`∂L/∂p_i = ⟪∇_t L, e_i⟫`: the first-order value of placing token `i` there. As soon as the
coda has a copy head that reads token embeddings, that gradient points at the copied token.
This is calculus, not a theorem in the package.

**(b) What does the signal do to loop contribution?** This is the uncomfortable part.

- Copy content does not need depth. `transcript_replay`: anything a coda-depth reader computes
  from raw context, a carrier that HOLDS the entries it read reproduces with zero passes. The
  rule that picks which tokens to hold is shallow if it depends on the token itself or on one
  lookup into earlier cells (`TulExploration.fixed_table_no_lookups`). It needs passes only if
  it composes context relations deeper than one pass (`TulExploration.passes_needed`). Web
  text was not measured for that.
- Commitment does not need depth either, with a learnable scale (`exists_scale_commit`). A
  bounded scale per pass would make it need passes (`passes_to_commit`: about `T·m ≥ 5.65`
  for a 16-token span at `ε = 0.05`). That would be a K-curve made by construction, which
  Wolfe has ruled out (depth use is emergent, not forced).
- A carrier that delivers copies beside the loop takes the copy share of K1−K6.
  `deeper_never_more_informative`: a positive K1−K6 is never information, only the bounded
  coda reading the deeper state better. `bypass_zeroes_passes`: once the coda's view is
  sufficient for a target, every pass is worth zero on it. Snapped pseudo tokens make copy
  content as easy to read as a raw token, so on copy tokens there is nothing left for deeper
  passes to make easier.

So the expected effect is: K1−K6 on far-repeat tokens falls toward zero, K1−K6 on the rest is
unaffected, and the total falls by the share of today's K1−K6 that lives on copies. Nothing in
the theory says what that share is. Stage 0 below measures it on existing files, on CPU,
before anything is built.

The teacher makes this worse, not better: a same-depth teacher asks for a held transcript, so
its student is the shallowest possible carrier (`distilled_carrier_le`).

**(c) Which trace-free signals reward copy content and keep depth necessary?** None of them
keep depth necessary FOR COPY CONTENT, because copy content is held, not computed. Among the
signals, only the coda's CE pays for copies at every horizon. The span decoder and the latent
rank head pay for horizon 1 only. Reconstruction pays for any token, copied or not. The
router's selection follows the latent rank head, so it is also horizon 1. Hidden-state
distillation pays by magnitude. Logit distillation pays the same as CE. The loop's
contribution must therefore be PROTECTED by the design (keep the loop in the path of
everything cross-span, keep the carrier from absorbing computed content), and MEASURED
bucket by bucket. It will not be generated by the copy carrier.

### Stage 0 (CPU, existing files, before any build)

K1−K6 by recall bucket on the three LXTUL sweeps that already have per-token files (5k
seed 1, 5k seed 2, 10k; `*.tokens.npz` with `ce_1` to `ce_6` under
`/home/wolfe/morph-scratch/abc/slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm*`).
Classify tokens exactly as `lab/divergence/exact_recall_gap.py` does (far bigram repeat by
distance 33-64, 65-256, 257+; far token repeat; novel). Report per bucket the mean
`ce_1 − ce_6` with a 95 % row bootstrap and the bucket's share of the total K1−K6 against its
share of tokens. File it as its own planned experiment before it runs.

My predictions, for that prereg: the far-bigram-repeat buckets carry a share of K1−K6 at least
1.5x their token share (0.6); they carry at least 40 % of the total K1−K6 (0.4). Decision: if
they carry 50 % or more, the arm below is predicted to fall under the kill line, and it should
run with that prediction written down, or be replaced by the fed-back variant (Alternatives).

### The arm: snapped pseudo tokens (`lxtul_snap.yaml`, composes `lxtul.yaml`)

Everything in LXTUL stays. The arm changes ONLY the content of the three coda prefix positions
per slot that are exactly zero on LXTUL (the losers' positions after `_fan_route_cells`). The
packer, the mask, the coda's sequence length and the winner write do not change, so LXTUL is
exactly this arm with those three positions at zero.

Per slot `s` and pseudo position `n ∈ {1, 2, 3}`:

```
c   = the winner cell after the LAST pass, as W_prefix reads it (stream mean)   # not detached
q_n = W_q[n] · RMSNorm(c)                                                       # W_q[n]: d x d
ℓ_ni = exp(β_n) · ⟪q_n, RMSNorm(E[x_i])⟫ / √d     for the span's own tokens x_i (pads masked)
p_n = softmax_i(ℓ_n)
pseudo_n = g_n · Σ_i p_ni · E[x_i]               E = lm_weight().detach()  (the tied table)
write pseudo_n into prefix position (winner + n) mod 4 of slot s
```

Keys (all new; they must enter `KNOWN_TUL_KEYS` in `morph/training/tul_setup.py`):

| key | value | why (theorem) |
| --- | --- | --- |
| `tul.pseudo_k` | 3 | one exact token per position (`card_strong_mul_sq_le_one`); the three zero positions cost no sequence length |
| `tul.pseudo_source` | `exit` | the K-sweep truncates the pseudo tokens with the loop, so K1−K6 measures the whole cross-span path; `entry` is the matched twin (`bypass_zeroes_passes`) |
| `tul.pseudo_support` | `span` | candidates are the span's own tokens: copies (`raw_span_sufficient`, `mixture_strength_eq_one_iff`), and a support the cell cannot use to write arbitrary computed content past the loop |
| `tul.pseudo_scale_init` | 0.0 (`β_n`, learnable, `exp` scale) | free scale, one snap commits (`exists_scale_commit`); bounding it would force depth (`passes_to_commit`) |
| `tul.pseudo_gate_init` | 0.0 (`g_n`, learnable) | step 0 is byte-identical to LXTUL; `g_n` grows on the CE gradient, then `W_q` trains |

Losses: none added. The coda's token CE, already the main loss, is the only signal that
reaches `W_q`, `β`, `g` and, through `c`, the loop (`ceValue_eq_next_add_far`,
`kd_mean_eq`). Every LXTUL term stays at its value (span decoder, latent rank head, fixed
point 0.1, epivol).

Detached: the tied table `E` in both the logits and the mixture, so this channel never trains
the input embedding or the LM head. Not detached: the winner cell `c`; the loop learns what its
pseudo tokens do for the coda, as in PonderLM. `W_q` is a plain parameter like `W_prefix` and
gets `W_prefix`'s precision treatment (not ternarised), so the two writes differ only in
mechanism.

Cost: three `d x d` matmuls and a softmax over at most 32 candidates per slot, about 68 slots
per 1024-token row. Small next to one core block. Not measured.

The matched twin, `lxtul_snap_entry`: the same keys with `tul.pseudo_source: entry`, the
logits read from the mean of the four seed cells before pass 1. Its pseudo tokens are
pass-independent, a bypass in the sense of `bypass_zeroes_passes`. It runs only if Stage 0
says the copy share of K1−K6 is under 50 %, as the instrument for "does the loop shape the
pseudo tokens".

### The fallback teacher, if the plain signal is too slow

Only if the arm's pseudo tokens do not become token-like (vertex mass under 0.5 at 2000 steps)
AND the copy hit rate stays at chance. Then the minimal teacher is LOGIT distillation, never
hidden states (`kd_misaligned`, CODI's own L1 on hidden states is the misaligned one): the same
coda, run a second time with a raw cross-span window (A1's mask), trained on its own CE, and
the student coda pulled toward its token distributions by a KL at a small weight. By
`kd_mean_eq` and `kd_variance_le` it changes only the variance. Cost estimate, not measured: a
second coda forward and backward (four blocks over every token) plus a second LM head and CE
over 49,152 entries, roughly +25 to 40 % step time, and more activation memory than the
~1 GB of slack the TUL arms have on the 5090. It also trains the shared coda on raw-window
inputs it never sees at deploy.

### Interaction with the B line now being built

`lxtul_recon.yaml` (own-span reconstruction at weight 1.0 on the winner cell) and
`lxtul_pool16.yaml` (16-head seed pool) put more span content INTO the cell.
`exact_tuple_write_needs_dim` says the linear `W_prefix` write still delivers at most one
exact token per cell to the coda, and `card_strong_mul_sq_le_one` says superposed tokens arrive
at strength `1/√k`. So a small effect of either on the far-bigram buckets is the predicted
outcome, not evidence that span content does not matter. B fills the cell; the snap delivers
it. They compose. `rec_misaligned` and `aux_price` add that reconstruction at weight 1.0 is a
main-sized signal whose optimum ignores what later spans copy.

## Alternatives considered

- **The CODI-shaped seed: distil a raw-context coda into the carrier-only coda.** Demoted to
  the fallback. Hidden-state matching selects by shift magnitude (`kd_misaligned`); logit
  matching has CE's optimum and only lowers variance (`kd_mean_eq`, `kd_variance_le`); a
  same-depth teacher can never make depth necessary (`transcript_replay`). It costs a second
  coda pass and memory the 5090 does not have.
- **Full-vocabulary PonderLM support** (`tul.pseudo_support: vocab_topk`, top 100 as in
  PonderLM §4.4). Kept as the next arm, not the first. It can carry computed tokens (guesses),
  which may suit the loop better (PonderLM's steps refine a prediction), but at step 0 a
  softmax over 49,152 tokens is near the table's mean vector, and its optimum is copies anyway
  once the carrier can hold the span (`raw_span_sufficient`).
- **A state pointer**: candidates are the span's prelude-output states, not its embeddings.
  A vertex is then exactly the coda's input at that token position, previous-token feature
  included, so one pseudo token can serve an induction-style bigram copy. The first switch if
  the vocab arm's copy hit rate is high but the far-bigram gap stays.
- **Fed-back pseudo tokens** (PonderLM-faithful `E_s = E_0 + Σ T_i` on the cell): each pass
  emits pseudo tokens and re-injects them into the cells, so the loop selects with its own
  current selection in view. It is the variant to try if Stage 0 says the copy share of
  K1−K6 is large: it keeps copy selection inside the loop and gives the passes an exact memory
  to compose over. No theorem here says it earns depth; T4's `passes_needed` says it can only
  if the selection composes context relations deeper than one pass.
- **A bounded scale per pass, to force a K-curve** (`passes_to_commit`). Rejected: a K-curve
  made by construction, which Wolfe has ruled out.
- **Hard top-k extraction (A3 as a separate build).** Subsumed: A3 is A2's vertex
  (`mixture_strength_eq_one_iff`), and the hard version has no gradient through the choice.
  Kept as an eval instrument (hard snap) on this arm.
- **Wider linear read-outs** (`tul.prefix_per_cell`, items 2 and 11). Not a way to exact
  tokens: `exact_tuple_write_needs_dim`. They may still help the 257+ and novel buckets.

## Acceptance criteria

Every reading on the 480 sweep rows, paired with LXTUL at the same seed and step where paired.

- Stage 0 is filed (prereg first) with per-bucket K1−K6 and shares on all three LXTUL sweeps.
- **Loop contribution, read first.** K1−K6 with its 95 % CI at 5k. Survives: lower bound at
  least +0.0130 (0.75 x the weaker LXTUL seed, +0.0172). Killed: point under +0.0086 (half of
  it). In between: a seed twin before any verdict.
- Directionality: mean-ablating the cells' shared direction moves K6−K1 by at least 1 %, as on
  all three LXTUL runs.
- K1−K6 per recall bucket against Stage 0. Predicted: it falls on far bigram repeats and does
  not fall on novel tokens.
- Gap to plain at the same step, 5k and 10k. CE success: at least 0.05 below LXTUL's
  (+0.272 at 5k seed 1, +0.356 at 10k) with the paired CI excluding zero.
- Exact-recall buckets against plain: the far-bigram 33-64 and 65-256 gaps (+1.78 and +1.60 on
  LXTUL 10k) fall; the 257+ and novel gaps do not rise.
- Pseudo-token instruments, eval: vertex mass (mean `max_i p_ni`); argmax agreement between
  the pass-1 and pass-6 emissions; copy hit rate (share of far-bigram-repeat targets whose
  source token is the argmax of a pseudo token of its source span); zero-ablation worth of the
  pseudo positions against the winner write (takeover check); hard-snap CE change; tok/s.
- `python scripts/verify_template.py` passes after the filings.

## Risks

- **The A line removes the copy share of loop contribution.** Predicted by
  `bypass_zeroes_passes` and `deeper_never_more_informative`. Its size is Stage 0's number.
  This is Wolfe's "we may kill it", stated as a measurable share before the run.
- **The bigram key.** A token embedding carries identity, not its predecessor. The coda must
  rebuild the induction key over consecutive pseudo positions, two of its four blocks. The
  state pointer is the fix if this binds.
- **Takeover of the winner cell.** The CE gradient through the pseudo tokens reaches the cell
  (not detached). If the cell's zero-ablation worth falls as the pseudo tokens' rises, the
  carrier is absorbing the cell's job; span-restricted support limits what it can absorb to
  raw span tokens and the choice among them.
- **Slow start.** `g_n` starts at zero, so `W_q` gets no gradient until `g_n` moves (the
  register's zero-init `W_o` precedent). If vertex mass stays near uniform for 2000 steps, see
  the fallback teacher.
- **The theory is idealised.** Orthonormal candidates, a normalised linear read, separable
  copy reads, a calibrated teacher, an ideal composer for depth. Every prediction above is
  tested by a measurement in the acceptance list; none is a verdict by itself.
- **n = 1 per arm.** LXTUL's seed spread on the gap is about 0.01 at 5k and on K1−K6 about
  0.004. Differences under those need the seed twin.

## Stage 0 result (2026-10-05, orchestrator, CPU only)

Run with `lab/divergence/exact_recall_gap.py` (per-bucket loop columns added): LXTUL's own
K1-K6 per recall bucket, each run gated on reproducing its own paired gap. Share of the
total K1-K6 (token share in brackets):

| bucket | 5k seed 1 | 5k seed 2 | 10k |
| --- | --- | --- | --- |
| far bigram repeat (19 %) | 0.236 | 0.266 | 0.257 |
| far token repeat, new bigram (35 %) | 0.462 | 0.418 | 0.528 |
| novel (46 %) | 0.303 | 0.317 | 0.215 |

The far-bigram buckets hold about a quarter of K1-K6, under the 50 % line Stage 0 set, so
the snap arm is predicted to keep the loop above the kill line. Per token the loop earns
2-3x more on repeated tokens than on novel ones (far bigram +0.027 to +0.032 against novel
+0.011 to +0.014): the loop already does part of the recall work. If a snap carrier takes
over the far-token bucket as well, up to about 70 % of K1-K6 is exposed. Artifacts:
`/home/wolfe/morph-scratch/recall/stage0_{s1,s2,10k}.json`.
