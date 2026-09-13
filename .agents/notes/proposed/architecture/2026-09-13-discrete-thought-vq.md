# Agent Note: the discrete thought — K vector-quantized codes per span

Status: proposed

## Problem

A span's looped state is ONE continuous vector, and a row's slot states are near copies of
each other. Measured:

| reading | value | source |
| --- | --- | --- |
| `val/slot_eff_rank`, strict ruler | **5.7598** in 1024 dims | `run_slot-spandec-strict.log` |
| `val/slot_pairwise_cos`, same run | **0.7104** | same |
| the same pair, slot family | 5.7 – 7.3 / 0.72 – 0.77 | `lab/experiments/successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md` |
| geometry audit | 1.7 – 4.8 | 2026-09-10 audit |
| 4,906 slots at step 0 | 18 – 23 | `results/2026-09-12-latent-z-gradient/step0-*.json` |

Twelve arms read token K1−K6 inside [−0.0001, +0.0033] (the strict ruler +0.0016, K3−K6
+0.0002) across every lever tried — core shape, stability terms, entry, seed, width, HCA,
geometry, and ten targets. None of them changed what KIND of object a span hands forward.

The other half of the problem is a contrast the tree already contains: **the coda reads
TOKENS well and reads this channel badly.** A token is a discrete symbol from a large
alphabet with its own embedding row, and the coda's whole job is reading sequences of those.
The slot cell is the one input it gets that is a raw point in R^1024.

## Proposal

`tul.vq_codes: K` (default 0 = today's forward, bit-identical). The loop's exit state is
vector-quantized into K symbols, and the K symbols are lifted into the K prefix cells the
coda already reads.

* **Head.** `W_vq: d -> K·d_c` (`vq_dim`, default `d/K`), applied at the SAME seam every
  reader of `z` already sits at — after the loop, after any `cond_layers` stack, before the
  span decoder, the MUX, SIGReg and the write.
* **Quantize.** Each sub-vector is snapped to the nearest entry of ONE shared codebook
  `vq_E: [C, d_c/G]` (`vq_codebook`, default 512). `vq_groups: G` splits a code position
  into G product-quantization groups, so a cell carries G symbols out of the same book.
* **Lift.** `W_vq_out: [K, d_c, d]`, one map per CODE POSITION, so position 0 and position 3
  may read the same symbol differently.
* **Exit.** Cell k is written 1:1 into prefix cell k through the shared `W_prefix[k]`, so
  `prefix_k` must equal `vq_codes` (it raises otherwise). The coda is UNCHANGED.
* **Target.** The span decoder grades the DEQUANTIZED thought, the mean of the K lifted
  codes, so the write is trained by the same target it is trained by today.
* **Loss.** The two VQ-VAE terms, codebook `||sg[u] − e||²` and commitment
  `vq_beta · ||u − sg[e]||²`, exposed as `vq` / `vq_weighted` on the `spandec_weighted`
  contract so `train.py` subtracts them and `train/loss` stays the model's CE.

Arms: `slot-spandec-strict-vq8`, `slot-spandec-strict-vq4`.

Instruments: `tul/vq_perplexity` and `tul/vq_used` (codebook usage over the batch's VALID
assignments) — the first numbers to read; `val/slot_eff_rank` / `val/slot_pairwise_cos` over
all 64·K LIFTED CODES and `val/slot_cell_eff_rank` / `val/slot_cell_pairwise_cos` WITHIN a
slot across its K codes. **The rank probe reads the lifted cells**, which is what the coda
holds; reading the continuous exit state would report the rank of a tensor no reader on this
arm sees, and reading the dequantized mean would report one number per slot and hide the
mechanism.

### Five decisions, each a choice with a cost

1. **Cosine (l2-normalized) codes, not raw L2.** The encoder output's scale has never been
   swept, and a codebook drawn at any fixed std either sits inside the encoder's cloud or
   far outside it — in which case every vector snaps to one code and the arm dies for a
   reason that has nothing to do with the question. Normalising both sides puts the match on
   the unit sphere where the scale cannot decide it. ViT-VQGAN (Yu et al. 2022) §3.2. Cost:
   the dequantized sub-vectors are unit-norm, so the scale the coda needs comes from
   `W_vq_out` instead of from the encoder.
2. **The VQ-VAE loss, not an EMA codebook.** An EMA book is updated INSIDE the forward under
   `no_grad`, and this tree runs the forward many times with no optimiser step behind it —
   `core_depth_sweep.py` at seven forced depths, `worth_profile.py` at four modes, the eval
   loop at 20 batches. Every one of those would step an EMA book and every instrument in
   this arc would stop meaning what it says. Cost: the loss form is the slower-converging of
   the two in the literature.
3. **The dequantized thought is the MEAN of the K lifted codes.** Every reader between the
   loop and the write takes one state per slot; the mean keeps each of those mechanisms the
   shipped one, so the arm differs from its ruler by the quantizer alone. The mean and not
   the sum: a sum scales with K and the vq8/vq4 pair would differ in magnitude as well as
   content. Cost: the decoder's gradient reaching the K positions is one target per span,
   the same limitation the register carries.
4. **One shared codebook** across the K positions and the four Hyper-Connection streams, so
   a symbol useful at one position can be reused at another. Cost: a per-position book would
   let positions specialise without `W_vq` having to arrange it.
5. **`vq_reset_after: 0`.** The dead-code reset is built, tested and OFF on both arms, so a
   collapsed codebook is a RESULT about the mechanism rather than a number the reset
   manufactured. Cost: if the book collapses the panel reads nothing and the arms are spent.

### Step 0 is not the ruler, and this arm is the only one that cannot be

Every other TUL arm zero-inits its new path so step 0 is its ruler exactly. A quantizer
cannot: a zero `W_vq` gives the encoder no direction to normalise, and a zero `W_vq_out`
writes an all-zero cell so the coda reads nothing. At step 0 the coda reads
`W_vq_out[k] @ e_k`, a near-arbitrary unit code picked by a random projection of `z`, at a
per-component rms of 1.0 (the lift's init std is `G^-0.5`, chosen so a lifted cell matches
the scale of the `input_norm`'d field it is scattered into). Measured on a CPU build at the
shipped `d_model` 1024: step-0 loss 13.5196 (ruler), 13.5499 (vq4), 13.7079 (vq8). The arm
starts behind its ruler, its early CE says nothing, and the comparison is at 5,000 steps.

## Alternatives considered

* **The continuous Thought Register (`tul.slot_cells`, the sibling build).** M mutable cells
  per span, seeded apart, looped together. It attacks the same defect and gives the loop MORE
  STATE; this gives the write a different KIND of state and leaves the loop's state alone.
  Not a substitute either way, and the two are refused together at construction — two
  multi-cell mechanisms at once is not one factor. If the register's cells collapse (its
  prediction P-1 failing), the collapse is caused by the loop and the shared write, and a
  quantizer is immune to exactly that: distinct codes cannot collapse onto each other
  however hard the loop pulls.
* **Gumbel-softmax / straight-through categorical codes.** A learned categorical
  distribution over C codes per position with a relaxed sample, instead of nearest-neighbour
  plus STE. It gives an unbiased low-variance gradient through the discrete choice and it
  can be annealed. Rejected for V1 on two grounds: it adds a temperature schedule nobody in
  this tree has swept (one more unmeasured knob on an arm that already has four), and its
  soft phase means the coda reads a MIXTURE of codes for most of training, which is a
  continuous write with extra steps — precisely the thing the arm exists to stop being.
  It is the first thing to try if the STE's bias turns out to be what keeps perplexity low.
* **FSQ, finite scalar quantization** (Mentzer et al. 2023): round each of d_c scalars to
  one of L levels, no codebook at all, no commitment term, no dead codes by construction.
  Genuinely simpler and it removes the failure mode P-1 is about. Rejected for V1 because
  the implicit codebook is `L^d_c`, which at any useful d_c is astronomically larger than
  512 and makes "K symbols out of an alphabet" stop being the right description of what the
  span hands forward — the perplexity instrument, which is the headline reading here, has
  no meaning. FSQ is the right follow-up if the codebook dies and the reset does not save
  it, and then the arm's claim changes from "K symbols" to "a quantized write".
* **Product quantization against one book versus one code per cell.** Built, as
  `vq_groups`, and set to 1 on both arms. G > 1 raises the per-cell alphabet to `C^G` at the
  same C rows, which is the cheap way to more capacity if P-7 says the count is the lever.
  Held at 1 for V1 so a cell is one symbol and the perplexity reading is over the same
  alphabet the description names.
* **K tokens emitted by the LM head itself ("MUX tokens").** Let the slot emit K real
  VOCABULARY tokens through the tied head, and let the coda read their embedding rows. It is
  the most literal reading of "the coda reads tokens well", it needs no new codebook, and
  the symbols would be interpretable — you could print a span's thought. Rejected for V1,
  with regret, because the argmax over 49,169 rows has no usable gradient at all (the STE
  through a tied head that is also the output head trains the embedding table from two
  directions at once — see the measured `lm_weight()` aliasing hazard), and because it is a
  strictly larger change: a new decoding stage, a new sampling decision at generation, and
  a second consumer of the tied table. It is the interpretable version of this arm and it
  belongs after the cheap version has said whether a discrete write is worth anything.
* **An orthogonality or decorrelation penalty on the continuous slot states.** Cheaper and
  it targets the measured symptom directly. Rejected on the same grounds the register
  rejected it: the 2026-08 campaign showed that penalising a geometry the map produces does
  not change what the map does (four spectral interventions, two worse than nothing).

## Acceptance criteria

1. `vq_codes: 0` builds no module, draws no RNG, adds no state-dict key and is byte-identical
   to the pre-work tree. **Met**: proved by running `f89256d` and this tree on four fixtures
   and comparing loss, logit sum, grad sum, `state_dict` key count, parameter count and the
   whole slot probe. All match to the last printed digit; three are pinned in the gate.
2. Cell k of a valid slot is exactly `W_prefix[k]` applied to `W_vq_out[k] @ e_n` for the
   recorded code index, read off the REAL coda input. **Met**, guarded with a spy.
3. The straight-through estimator carries gradient into the loop, and cutting it drives the
   core's gradient to exactly zero **at `vq_weight: 0`**. **Met**, guarded — and the
   qualifier is load-bearing, see Risks.
4. The two VQ-VAE terms are positive, in the loss, and exposed as a detached weighted twin
   `train.py` subtracts. **Met**, guarded.
5. The span decoder grades the dequantized thought and not the continuous exit state.
   **Met**, guarded with a spy on the real call.
6. Pads: index −1, cells exactly zero, out of both loss terms and out of the usage
   histogram, and no VQ parameter reads a NaN gradient. **Met**, guarded two-sided.
7. The discrete assignment is made in fp32 whatever the carrier's dtype, so a bf16 rounding
   cannot flip a code — under autocast the `argmax` is over cosine similarities and two
   close codes are one rounding apart. **Met**, guarded through CPU autocast (same index
   tensor as the fp32 twin, both loss terms fp32); the CUDA path is unverified.
8. `tul/vq_perplexity` above 32 at 5,000 steps on `slot-spandec-strict-vq8`. **Not met —
   unrun.** This is the number the arm is readable through.
9. Token K3−K6 below 0.002, against the ruler's +0.0002. **Not met — unrun.** Predicted to
   HOLD at 85 %: this arm is a test of the information account, not an attempt to beat it.

Prereg: `lab/experiments/planned/2026-09-13-arc-discrete-thought-vq.md`.

## Risks

* **The codebook may collapse, and then the panel reads nothing.** The thing being quantized
  is the same near-degenerate state the rank probe measures at 5.76 effective dimensions,
  and a book fitted to a six-dimensional cloud has little to distinguish. The cosine match
  removes the scale-mismatch failure mode but not this one. `vq_reset_after` is built and off
  against exactly this, and turning it on is the named follow-up.
* **Rank by construction is nearly a tautology.** P-2 (the row rank moves) is close to true
  the moment distinct codes are used, and it is listed in the prereg so the panel cannot
  claim a win on a number that was never in doubt. The claim worth testing is whether a
  channel with rank changes what the READER does with it, and the information account says
  it should not.
* **A claim in the source was wrong, and a sabotage caught it.** The first draft said the STE
  is "the only edge by which the loss reaches the loop". It is not: the COMMITMENT term also
  reaches the encoder, hence the loop, so at `vq_weight > 0` the loop learns from two edges.
  Sabotage S2 (detach the STE) MISSED because the core still read a nonzero gradient through
  the commitment term. The claim is corrected in three places and the test now runs at
  `vq_weight: 0`. The practical consequence is real and not only editorial: on this arm the
  commitment term is a training signal on the LOOP, not only on the quantizer, and a
  `vq_weight` sweep would be sweeping two things.
* **Neither comparison in the panel is one factor.** A 1:1 write forces
  `prefix_k == vq_codes`, so vq8 against the ruler differs by the quantizer AND the prefix
  width (`L_total` 1536 against 1152), and vq8 against vq4 differs by the count AND the code
  width AND the width again. The arm that would make it clean — a strict ruler at `prefix_k`
  8 with no quantizer — is not built. Named in the config header and in every prereg row.
* **The arm starts behind its ruler and cannot be zero-init.** Stated above and measured.
  Anyone reading a 200-step CE on this arm is reading the bottleneck's init, not the arm.
* **Eager-only nothing, but a longer coda.** The quantizer adds no attention mask and no new
  relation, so unlike the register it does not force the eager path. What it does add is
  `L_total` 1536 against 1152, a 33 % longer prelude+coda sequence, and `slot-mux-prefix4` at
  1280 already cost +2.93 GB. vq8 is the most likely arm in this panel to die on memory.
* **Nothing has run on a GPU.** No smoke, no wall clock, no memory figure. The forward and
  backward have run on CPU at the shipped `d_model` 1024 with no non-finite gradient, at a
  sequence and a vocabulary that are not the arm's, and the bf16 path is exercised only
  through CPU autocast.
* **A VQ checkpoint carries three keys no other model has**, and loading one into a non-VQ
  model raises on the homeless keys (the tree's contract, `train.py`). Both arms train from
  scratch, so no loader path was tested.
