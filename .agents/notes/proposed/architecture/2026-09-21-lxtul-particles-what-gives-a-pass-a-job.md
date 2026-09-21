# Agent Note: LXTUL-P — what gives a pass a job, and the design that falls out of the 2026-09-21 discussion

Status: proposed

Written 2026-09-21 at Wolfe's direction ("save all of this into a file to remember it later,
the why we are doing this") after a three-way discussion: Wolfe, this session, and an outside
agent that had not seen the tree. The outside agent's two replies are summarised and scored
here; the tree's own numbers are cited by their records. Companion readings filed the same
night:
[`docs/references/tul-latent-emission/lcm/2026-09-21-lcm-reading.md`](../../../../docs/references/tul-latent-emission/lcm/2026-09-21-lcm-reading.md)
and
[`docs/references/looping-depth/perceiver/2026-09-21-perceiver-reading.md`](../../../../docs/references/looping-depth/perceiver/2026-09-21-perceiver-reading.md).

## Problem

The strict slot loop does not earn depth. Fourteen arms now read token K3−K6 inside
[−0.0002, +0.002] and pass 1 does 89–95 % of the work. The two levers of this week both
FORCED the written state to have rank and both left the passes flat:

| lever | rank of the written state | K3−K6 | record |
|---|---|---|---|
| strict partner (one vector) | 13.85 | +0.0002 | `lab/experiments/results/2026-09-12-strict/` |
| fan4 + within-slot volume term (K=4 streams) | 2.8 of a ceiling of 3 at pass 1, 2.0 at pass 6 | +0.0049 (K1−K6) | `lab/experiments/successes/2026-09-20-lxtul-fan4-all.md` |
| vq8 / vq4 (K=8 / 4 codes from 512) | 35 / 17 | 0.0000 / +0.0002 | `lab/experiments/mixed/2026-09-13-arc-discrete-thought-vq.md` |

So "the states are copies" is not what stops passes 2–6. The one arm that beats its width
partner is fan4-all (−0.034 nats paired vs pk4): every stream reaches the coda in its own
cell and the coda's per-token attention selects. That fixed the EXIT of the loop, not the loop.

**Wolfe's premise going in:** LXTUL is the most likely direction because of the loop's learning
target (winner-takes-all over K streams: each stream may commit to a mode, none must predict
the mean). **The outside agent's correction, which this note accepts:** winner-takes-all gives
K separate "best prediction given the context" problems, each solvable in one pass by a map of
enough capacity. Contractivity does not bound how much context computation a map does in one
pass (its counterexample: a contractive linear map `A_C` with `A_C (R e_k) = g_k(C)` produces
every candidate in one pass for large `R`). Nothing in the WTA loss asks pass 2 to do what pass
1 could not. The fan's flat K-curve at rank 2.8 is that statement measured.

**The mechanism that DOES give a step a job, in print:** LCM's denoising steps earn (their MI
rises with steps, Figure 10) because each step receives a noise level and has the clean target
at that level as its own trained target; Quant-LCM's iterations earn because code k predicts what
codes 1..k−1 left out. Our passes 2–6 receive nothing that distinguishes them, and when we gave
them fitted per-pass targets, the targets were met in one step
(`per-pass-targets-met-in-one-step`). Our own sampler (the LCTUL flow thinker) had the
mechanism and failed for two reasons LCM did not have: its target was a VERBATIM
reconstruction code of which the context explains one tenth (`lab/theory/lctul_euler_depth/`,
`explained_of_residual_ratio`), and its loss read 1 % of the past (context-blind,
`lctul-thinker-is-context-blind`). LCM's peaked noise schedule reproduced exactly that
failure inside the diffusion family (their Table 5: best ℓ2, worst CA and MI, "akin to a
Base-LCM"), and their wide schedule plus guidance at sample time fixed it.

## Proposal

Two parts: the probes that decide which lever is live (no 5090 time), then ONE design, staged
one factor at a time.

### Part 1 — probes on the fan4-all checkpoint (3070), before any arm

1. **The mixture read** (`lab/divergence/fan_mixture_probe.py`, to build). From the K coda
   passes the oracle instrument already runs (stream i alone in its cell), collect per-span,
   per-stream, per-token log-probs of the next span, and read on the same rows: the oracle
   (per-span min of summed NLL), the best single stream, the deployed all-cell read, the
   UNIFORM mixture `−log((1/K) Σ_k exp(−ℓ_k))`, and the PREFIX-WEIGHTED mixture whose stream
   weights update token by token from the tokens already decoded (no leakage: the prefix is
   available at inference). The outside agent's bound: the uniform mixture is at most `log K`
   nats per SPAN worse than the oracle, so part of "selector regret" (fan4-all: 0.099 − 0.056 =
   0.043 per token) is the price of not being told which branch happened, and no selector can
   remove it. Also from the same tables: **winner persistence**, P(winner of span n+1 = winner
   of span n) against the marginal share, which says whether stream identities persist across
   spans (the precondition for Part 2, stage 4).
2. **Anisotropic gain** (extend `morph/training/core_jacobian.py`). On the fan4-all trajectory,
   per pass, the realised gain `‖J v‖/‖v‖` for `v` in the stream-MEAN direction (the same
   perturbation on all K streams of a slot) against `v` in the DEVIATION subspace (a
   perturbation with zero stream-sum). Uniform contraction cannot change rank; the fall from
   2.8 to 2.0 across passes means the core contracts deviations faster than the mean, and this
   probe says by how much.
3. **The fixed-point term** (`morph/configs/tul_slot_spandec_strict_fan4_all_fp0.yaml`, built
   2026-09-21). `model.core_fixed_point_lambda 1.0` asks the LAST pass to be a fixed point on
   the whole state, deviations included: it pushes the streams to stop moving relative to each
   other. Under the 1000-step ramp the term is measured free, so a one-factor arm at 0 is safe.
   This is a 5090 arm (5k steps, ~65 min) and the only arm this note queues before the probes.

### Part 2 — the design: LXTUL-P (particles)

What the discussion converges on, kept separate from what it does not settle. FIXED, because
measured to work: the span unit; the strict geometry (the next span reaches the past only
through the slot state); the read (the coda attends over K cells per token, fan4-all); the
token path (the coda keeps its token CE; nothing decodes a span from one vector). CHANGED,
because measured to be missing:

1. **Each pass has a job: a noise level.** The K stream states enter the loop as noised
   targets at level β_1, and pass t maps level β_t to β_{t+1} with the clean target at that
   level as its own trained target (LCM's x0-prediction, wide schedule over log-SNR, T
   training levels, S sampling steps = the loop depth). The per-pass loss exists by
   construction, not by a fitted per-pass target. Every pass receives its level (AdaLN on the
   core, as LCM's Two-Tower) so pass t is not pass 1 with a different input.
2. **K streams are K samples.** The seeds are K independent noise draws plus the shared
   trigger. Distinctness comes from the noise; no volume term. The oracle, mixture and
   stream-CE instruments stay as instruments. At eval the K samples are read by the same
   per-token coda.
3. **The target is semantic, not verbatim.** The clean target is a frozen, paraphrase-invariant
   embedding of the NEXT span, so the context explains more than one tenth of it. First
   candidate: SONAR (LCM retrieves the true next sentence among in-batch alternatives 75–80 %
   of the time in that space; the encoder is a Meta pip package, not verified on this
   machine). Control: our own frozen E with LCM's noised-decoder training (their Table 7,
   +8.5 to +15 AutoBLEU). Guidance at sample time (context dropped 15 % at train, scale 3 at
   eval) and the context-blind probe (`code_cfg_drop` 0.999 vs 0.001) are the checks that the
   sampler reads the context at all.
4. **Persistence across spans (stage 4, after 1–3 read).** Stream k of span n+1 seeds from
   stream k of span n plus noise, and the per-stream NLL of span n AFTER it is observed
   reweights the lineages (a filtering posterior; the outside agent's answer 6). The
   winner-persistence probe is its precondition.

**The ladder, one factor per arm over fan4-all:**

| rung | delta over the rung below | what it isolates |
|---|---|---|
| P0 | fan4-all with `core_fixed_point_lambda 0` | whether the terminal fixed-point term is part of the pass-6 collapse |
| P1 | K noise-drawn seeds, no volume term | whether noise alone keeps K streams apart and the reader cashes it (vs the term) |
| P2 | P1 + per-pass denoising target on our frozen E, wide schedule, guidance | whether a pass with a job moves the reader's CE beyond P1 |
| P3 | P2 with the target swapped to SONAR embeddings of the next span (precomputed) | whether a semantic target beats a verbatim one (Wolfe's "is SONAR better", one factor) |
| P4 | P3 + stream lineages across spans with post-hoc reweighting | the filtering job |

**Score.** Not the K-curve alone: a denoiser's K-curve rises trivially as the noise falls. The
score of every rung is the READER's paired CE against the rung below on the same rows, beside
the oracle/mixture gap and the sampler's CA/MI (their §2.4 metrics, to port). The seed spread
on this tree is 0.024 nats (`mean-depth-is-a-depth-axis-dial`); a rung must clear it.

**Falsifier for the whole design.** If P2 does not beat P1 at the reader by more than the seed
spread, a pass with a defined job still adds nothing at this scale, and the loop's job was not
the issue. That closes the sampler lane on this tree for good, with the mechanism named.

## Alternatives considered

The outside agent's six answers, each scored against the tree:

1. **A restricted per-pass operation on a frozen problem** (one gradient step per pass on a
   frozen context-conditioned energy; "the restriction earns depth, not the loss"). Sound, and
   untested here: our gradient-conditioned pass (`gradpass-one-step-optimiser`) used the LIVE
   decoder as the energy and descended in one pass. Held as the alternative to LXTUL-P if P2
   fails: a frozen energy is the other way to give a pass a job.
2. **Change the map, not the penalty**: a split map that contracts the stream mean and rotates
   the deviations by an orthogonal `O(C)` (Jacobian exactly `a` on the mean, 1 on the
   deviations). A diagnostic ablation, not an architecture: it preserves wrong hypotheses as
   faithfully as right ones. Its qualification about the fixed-point term is rung P0. Feeding
   the per-stream trigger every pass keeps streams apart by re-injection with Jacobian `aI`;
   cheap, and second in line after P0.
3. **Particles with a target-blind density-ratio scorer** (NCE: positives = encoded true
   continuations, negatives = proposals for the same context; deployed scorer reads only
   `(C, z)`), tempered SMC over the passes with Langevin moves. Not the fitted-z trap: the
   target supplies training EXAMPLES, and the scorer's input is the same at train and eval.
   Its own decisive limitation: without particle MOVES the weight updates telescope and five
   passes equal one. Heavier than LXTUL-P (a frozen encoder, a scorer, moves with a learned
   score gradient); held behind it. The outside agent's own confidence: 6.5 of 10.
4. **Marginalise, do not select.** The log-sum-exp mixture and the prefix-weighted mixture
   are the principled reader objects; a verifier that predicts the decoder's loss estimates
   risk, not which future happens, and can reward certainty. Adopted as Part 1 probe 1 and as
   the reader of every rung.
5. **The falsifier is semantic specialisation under intervention** (`I(J; M | C)` with a
   semantic branch label). Correct that none of our four instruments (oracle − mixed, stream-CE
   spread, centred rank, sign families) shows the streams are semantic modes, and that sign
   families are the weakest. Needs a judge we do not have in the loop; the cheap proxy
   (decode each stream alone, compare by a sentence embedding) is a later instrument.
6. **Short emission units, persistent document-level hypotheses** with weights updated after
   each span is observed. Adopted as rung P4, gated on the winner-persistence probe.

**Copy LCM's whole recipe, SONAR included** (Wolfe's question). Rejected as a whole: SONAR's
DECODER is the part LCM paid for (no token path; fluency 0.68–0.77 against 0.92–0.97, their
Tables 10 and 12). SONAR's ENCODER as the target is rung P3, one factor. The schedule,
guidance and CA/MI instruments are rung P2.

**Another rank lever** (repulsion on every pass, a bigger codebook). Rejected: two rank levers
this week moved nothing on the passes.

## Acceptance criteria

- Probe 1 reads the five numbers on ≥ 48 rows with a block-bootstrap interval, and the
  prefix-weighted mixture is reported beside the deployed all-cell read. If they agree within
  the interval, the selector question is closed and the note says so.
- Probe 2 reports per-pass mean-direction and deviation-subspace gains with their ratio, on
  the same checkpoint and rows as the stream-rank probe.
- Every rung has a prereg committed before launch with frozen predictions, is scored on the
  reader's paired CE against the rung below, and states the seed spread it must clear.

## Risks

- **Theatre risk named up front:** a denoiser's K-curve rises because early passes are noisy.
  Any rung that quotes K1−K6 without the paired reader CE against the rung below is reading
  the noise schedule, not depth.
- The Lean result stands: on a Gaussian-like conditional the Euler field is affine and the
  step count sets one scalar. P2 earns only if the conditional in the target space is not
  Gaussian-like and guidance makes the field curve. The CA/MI port is the instrument for that.
- SONAR embeddings of 2 M spans must be precomputed; cost unmeasured. Their fragility (LCM
  §2.5.2) applies to any frozen target: small moves decode to different text.
- Five rungs at 5k steps each is a week of 5090 time; the ladder stops at the first rung that
  fails its falsifier.
