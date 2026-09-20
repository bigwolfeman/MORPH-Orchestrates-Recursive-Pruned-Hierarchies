# Agent Note: an epiplexity diversity term for the fan, in place of the cosine repulsion

Status: proposed

## Problem

`slot-spandec-strict-fan4` (K = 4 latent streams per span, `tul.fan_repel_lambda 0.1` on a
mean pairwise cosine at passes 1 and 2) met its repulsion at the floor by step 500 and
stayed there: `fan/stream_cos_t1` −0.326, `fan/stream_rank_t1` 1.05 of 4 for the whole
run. The mean pairwise cosine of K unit vectors is bounded below by −1/(K − 1) = −1/3, and
two shapes sit on that floor: a regular simplex (centered rank 3) and a rank-1 split of
two copies against two negated copies. The stream probe on the step-5000 checkpoint
(`lab/divergence/fan_stream_probe.py`,
`lab/experiments/results/2026-09-19-lxtul-fan4/fan_geom_fan4_5000_d6.json`) found the
second shape, and worse: the ONE direction is the same in 96 % of slots (mean |cos| of
consecutive slots' top directions 0.96) with fixed stream identities (streams 0 and 1 on
the plus side, 2 and 3 on the minus side in 2536 of 2573 slots) and unequal norms (35 to
39 against 20 to 26). The repulsion was satisfied by a constant sign split on a fixed
axis, which carries no content. The coda's reading agrees: at the final val every single
stream scored 0.03 to 0.10 nats above the mixture the coda trained on
(`fan/stream_ce_k{i}` 4.5735 / 4.5439 / 4.5090 / 4.5038 against `fan/mixed_ce` 4.4744)
and the oracle over streams beat the mixture by 0.008, so the 0.107 single-minus-oracle
gap that P-1 of the fan4 prereg reads is 0.099 of a constant offset plus 0.008 of choice.

A diversity term that can be paid off by a constant offset on one axis cannot test the
K-stream question. The term has to be blind to that shape.

## Proposal

`tul.fan_repel_mode: cos | epi` (default `cos`, bit-identical to today; unknown values
raise in `TULConfig` and `tul_setup`). Under `epi` the charged term is minus the
epiplexity score of EpiJEPA (github.com/the-puzzler/epijepa; Zhang & Levin, "Intelligence
from Learnable Novelty", arXiv 2607.18433), applied to the streams' between-stream
deviations:

- `d_i = z_i − mean_j z_j` per stream, over the batch's valid slots, divided by the slot's
  mean stream norm (live, so the score is scale-free and cannot be raised by growing the
  streams; a looped map under a gain constraint must not be handed a term that pays for
  norm).
- A FROZEN random MLP (`FanReservoir`, `C → 256 → F`, ELU, buffers from a private
  generator, `F = tul.fan_epi_features` 64) reads the slot's SEED (mean over streams of
  the trajectory's entry 0, detached). A ridge readout `A = (HᵀH + ρI)⁻¹Hᵀ`
  (`tul.fan_epi_ridge` 3, QR, double, under `no_grad`) maps reservoir features to each
  `d_i`; the score is `S_i = ½ log₂ det(I_F + η W_i W_iᵀ)` with `W_i = A d̂_i` and
  `η = tul.fan_epi_eta` 30, taken on the F × F side by Sylvester's identity.
- The term is `−mean_i S_i / F` (bits per reservoir feature) averaged over passes
  `1 .. fan_repel_passes`, weighted by `fan_repel_lambda` exactly as the cosine was; every
  pass is reported as `fan/epi_t{t}` (train) and `val/fan_epi_t{t}` (val). The cosines
  and ranks stay instruments under `epi`, computed with no gradient.

Why this shape is blind to the failure: centering over the batch removes any constant
offset, so the ±u split scores zero; the log-det rewards magnitude spread over MORE
directions, so rank is what is paid for; and the readout is from a frozen random function
of the input, so only variation that is a linear function of the seed counts, not noise.

## Alternatives considered

- **Raise `fan_repel_lambda` on the cosine.** The floor is a property of the cosine, not
  of its weight; a larger weight reaches the same degenerate split faster.
- **A participation-ratio rank term directly** (maximise `fan_stream_rank_t1`). Rewards
  spread but not input dependence: isotropic noise in the streams would satisfy it, and
  the coda cannot read noise. The epiplexity readout is the part that asks the spread to
  be a function of the input.
- **SIGReg on the deviations** (the tree already has it on the FM planner and the code
  target). Asks for an isotropic Gaussian, which noise also satisfies; EpiJEPA's own
  comparison puts it about 1 to 4 points ahead of epiplexity on image probes at lower
  embedding rank, so it is a second arm if this one moves the rank and not the oracle.
- **K = 2.** Removes the two-against-two split by construction but leaves the constant
  ±u axis available; the term, not K, is the lever.

## Acceptance criteria

The prereg `lab/experiments/planned/2026-09-20-lxtul-fan4-epi.md` freezes them. In short,
on `slot-spandec-strict-fan4-epi` at 5000 steps: `fan/stream_rank_t1` above 2.0 of 4 and
no global axis in the stream probe (mechanism); every `fan/stream_ce_k{i}` within 0.03 of
`fan/mixed_ce` (no constant offset); and the falsifier, `fan/oracle_ce` below
`fan/mixed_ce` by more than 0.022 (the streams give the coda something to choose
between). Bit-identity at `cos` and the pass contract are `tests/test_tul_fan_epi.py`.

## Amendment, 2026-09-20 (the epi arm's loophole, and the volume term)

`slot-spandec-strict-fan4-epi` at step 1000 (val): `val/fan_epi_t1` 1.67 bits per feature
and rising, `fan/stream_rank_t1` **1.001** of 4, `val/fan_stream_cos_t1` 0.30, cos_t6 0.55.
The score is high and the streams of a slot lie on one line. The epi score is computed
per stream ACROSS slots and never looks inside a slot, so it is paid off by
`d_i(n) = c_i · v(n)`: one input-dependent direction per slot (learnable from the seed,
hence the score) with fixed per-stream scalars (hence rank 1). The proposal above named
"input-dependent, spread-out" as one property; it is two, and the term measured one.

Added: `tul.fan_repel_mode: vol | epivol`. `vol` is the WITHIN-slot volume, `½ log₂ det(I +
η G)` of the K × K Gram of a slot's normalised deviations (rank ≤ K − 1 since they sum to
zero), in bits per (K − 1): near zero for a line, largest for K − 1 orthogonal equal
deviations, scale-free through the same normalisation, reported as `fan/vol_t{t}`. `vol`
alone can be met by fixed orthogonal axes (content-free, the cosine's failure one rank up);
`epivol` sums the two so the spread must also be input-dependent, since a constant
deviation scores exactly 0 on epi. Tests: a line scores low and a spread high on `vol` at
equal magnitude; the line family keeps most of its epi score and loses on the sum; fixed
axes score 0 on epi and high on `vol`. The next arm is `epivol` against `epi`
(`lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md`).

The epi arm at 5000 (filed: `lab/experiments/failures/2026-09-20-lxtul-fan4-epi.md`): the
loophole was realised as a ONE-HOT stream per slot, not a shared line. The stream probe reads
per-stream norms 14.5 / 0.1 / 0.1 / 18.9 at pass 1, rank 1.000, `axis_cos` 0.125, and two
sign-pattern families of about equal size (`+---` 1347 slots, `+++-` 1226): stream 0 carries
the slot in half the slots and stream 3 in the rest, chosen by the input, the other streams
near zero. Which stream fires is a learnable function of the seed, so each stream's
epiplexity is high (`val/fan_epi_t1` 1.67 bits per feature) while the within-slot rank is
exactly 1. Oracle − mixed 0.011, inside the 0.008 to 0.016 every fan arm reads. The volume
term scores a one-hot slot near zero (three of four deviations are the negated mean and the
fourth is their sum: rank 1 Gram), so it is blind to this shape as well as to the line.

The epivol arm at 5000 (filed: `lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md`):
the term is NOT gamed. Rank 2.82 of 4, `axis_cos` 0.14, four live streams at norms 12.3 /
10.4 / 9.3 / 11.5, raw cosines near −1/3, directions that change slot to slot. And the
falsifier still fails: oracle − mixed 0.018 (bar 0.022), the softmax mix concentrated on
stream 0 (entropy 0.41, pick0 0.71), streams 1 to 3 read 0.14 to 0.42 nats worse than the
mixture. Three terms read; the K-stream branch closes. This note stays `proposed` as the
record of the term that works, for any future fan; it does not ship.

## Risks

- The term's scale differs from the cosine's (bits per feature, a few at most, against a
  cosine in [−1/3, 1]); `fan_repel_lambda 0.1` is kept and `fan/repel_weighted` is read
  against the total loss. If the term dominates, the fallback is λ, not the shape.
- At step 0 the streams are identical (`W_o` zero-init), so the score and its gradient
  are exactly zero there, as the cosine's were; the register's distinct queries break the
  symmetry through the CE, as they did on fan4.
- The reservoir reads the seed, which trains. It is detached, so the term cannot move the
  seed, but the ridge target drifts with it; EpiJEPA's reservoir reads raw pixels and has
  no such drift. If the score wanders without the streams moving, read `val/fan_epi_t0`
  (the seed's own score, identically 0 while the streams are identical at entry).
