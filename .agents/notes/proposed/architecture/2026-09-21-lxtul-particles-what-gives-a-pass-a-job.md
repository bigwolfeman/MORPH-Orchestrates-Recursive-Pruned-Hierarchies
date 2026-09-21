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

### If the ladder fails: remove the prelude (Wolfe, 2026-09-21)

Wolfe's direction, verbatim: "If these fail we should test removing the prelude." Recorded
here so it is not lost. The arm is `model.n_prelude: 0` on the strict slot loop (and on the
LCTUL thinker if that lane is still open): the core reads the embeddings directly, for the
tokens and for the slot seed, with no prelude blocks in front of it. It is NOT the same as
the noise entry already measured (`core_state_init`, `2026-09-10-arc-slot-mnext-noise-entry`:
the slot loop read K1−K6 0.002 under a noise entry with the prelude still present). The prior
fact that motivates it (`prelude-entry-flattens-the-loop`): the PLAIN norm_match model earns
K1−K6 0.033 under the prelude entry and 0.185 under Parcae's noise entry, and the core's
blocks are near-inert on slot states under the prelude entry (MLP out/in 1–2 %), because the
prelude hands the loop a state near its fixed point. Removing the prelude removes the thing
that produces that state. Queue it as one factor over the strict partner FIRST (the ruler must
move before a fan or a sampler is read on it), with the hinge re-checked (a state that is not
near the fixed point at entry reads a gain the hinge was not tuned for; the noise-entry filing
found `gain_est` 8.6 and a 6,000-nat penalty, and ran on the free base).

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

## Outcome, running (dated entries; the note is the record the ladder writes to)

**2026-09-21, Part 1 probe 1 (the mixture read) — DONE, on fan4-all @ 5000, 48 rows, depth 6,
`lab/divergence/fan_mixture_probe.py` (17 tests), artifacts
`lab/experiments/results/2026-09-19-lxtul-fan4/fan_mixture_fan4-all_5000_d6.{json,txt}`.**
Process slip, named: this probe ran with no frozen numeric prediction beyond the bound stated
above (uniform mixture at most `log K` per span above the oracle); the acceptance criterion
was written before the run (commit c707c29), the prediction was not. The reading (nats per
token, 95 % row-block bootstrap; the probe reproduces the model's own `fan/oracle_ce` and
`fan/mixed_ce` to 1e-8 on the same rows):

| reading | value |
|---|---|
| oracle (hindsight per span) | 4.2082 |
| deployed all-cell read | 4.2485 |
| uniform mixture = prefix-weighted mixture (telescoping identity, asserted) | 4.2583 |
| best single stream (k0) | 4.2874 |
| deployed − prefix mixture | **−0.0098 [−0.0119, −0.0077]** |
| deployed − oracle | +0.0404 [+0.0375, +0.0434] |
| uniform mixture − oracle | +0.0502 [+0.0484, +0.0520]; the bound `log K / mean span length` is 0.0705 |
| mean span length | 19.67 tokens |

**What it settles.** The selector question closes. The deployed read (the coda's per-token
attention over the four cells) BEATS the principled branch-blind marginaliser by 0.0098, in
every offset bin (by 0.008–0.021), and the marginaliser itself pays 71 % of the `log K` bound:
the branch is genuinely ambiguous before the span, and no selector that does not see the
future recovers the oracle's 0.040. The 0.042 "selector regret" in the fan4-all filing is the
price of hindsight, not a defect of the gate. The lever left for the read is BETTER CANDIDATES
(rungs P1–P3), not a better selector. Not explained: why a learned joint read beats every
convex combination of the per-stream predictive distributions (a joint read can combine
features the streams hold separately; hypothesis, untested).

**Winner persistence** (the precondition for rung P4): P(winner of span n+1 = winner of span n)
0.3039 [0.2854, 0.3250] against chance 0.2676 [0.2623, 0.2752], excess +0.036 [+0.017,
+0.057]; shares 0.354 / 0.259 / 0.203 / 0.183. Identities persist a little more than chance;
a thin base for lineages. Rung P4 stays last.

**2026-09-21, rungs P1 and P4 BUILT (no GPU step yet).** Appended rather than rewritten,
because two agents were editing this tree at once.

**P1 — `tul.fan_seed_noise`** (`morph/model/tul.py`, `transformer.py`,
`training/tul_setup.py`; 12 tests in `tests/test_tul_fan_seed_noise.py`; config
`tul_slot_spandec_strict_fan4_all_noise.yaml`; prereg
[`2026-09-21-lxtul-fan4-all-noise.md`](../../../../lab/experiments/planned/2026-09-21-lxtul-fan4-all-noise.md)).
One `torch.randn` per forward, per stream and per slot, added to the ENTRY STATE
`h = core_init(e)` — NOT to `e`. That distinction is the one design decision here and it is
a test, not a comment: `e` is bound once and handed to every pass as the injection source,
so noise in `e` would be one draw re-injected at every pass, i.e. a random per-stream
trigger rather than a sample. Scale: the seed lives in the `input_norm`'d field, so
`fan_seed_noise: 1.0` is unit per-channel RMS there (a pair of streams then differs by RMS
sqrt(2)); a fixed std, never a rescale of the live state's RMS, which would correlate the K
draws through one norm and shrink the spread exactly when the map contracts. Drawn at train
AND at eval, so it moves the global RNG stream — a one-factor comparison against fan4-all is
one factor in the MECHANISM, not in the draw order. The arm pairs it with
`fan_repel_lambda: 0`: the volume term's CHARGE goes, its instruments (`fan/vol_t*`,
`fan/epi_t*`, `fan/stream_cos_t*`) keep reporting on the same axes.

**P4 — `tul.fan_lineage: "relation"` only, and the reason the rest is not built.** The
relation half is built (`slot_cell_relation(..., lineage=True)` narrows the CROSS-SLOT half
to the own stream index; within a slot the register's all-to-all relation is untouched; 7
tests in `tests/test_tul_fan_lineage.py` assert the mask pair by pair AND prove the
narrowing EXECUTES, two-sided against an independently written mask — the 2026-09-13
`tg_allow` bug one level down). **The filtering posterior is NOT built, and it is not a
matter of effort: in this forward it is circular.** `_tul_core` advances every slot's pass
`t` together (one `_apply_core_step` over `[B, S*M, *carrier, C]`, the per-slot depth
applied as a masked `where` at the foot of the loop), so slot n's exit state does not exist
when slot n+1's pass 1 runs; and `l_{n,k}` is not a loop quantity at all — `_tul_fan_all`
builds it from K CODA replays AFTER `_tul_core` returns, so the weight that would gate pass
1 is a function of the coda's output, which is a function of the whole loop. Rejected
workarounds, each for a stated reason: the previous step's table (a weight learned on
another document), a second full forward (double cost, still needs a loop before the loop),
a slot-sequential loop (S times the sequential depth, S up to 64). A third `fan_lineage`
value RAISES with that explanation rather than shipping a weighting that reads span n+1.
**So rung P4 as written in Part 2 needs a forward that is sequential over spans; the
relation arm tests only whether a per-stream channel is worth anything before any
reweighting exists to sit on it.**

**The fallback (`model.n_prelude: 0`) is configured and smoked on CPU:**
`morph/configs/tul_slot_spandec_strict_np0.yaml`, prereg
[`2026-09-21-strict-np0.md`](../../../../lab/experiments/planned/2026-09-21-strict-np0.md).
No model code change; `model.n_layers` is documentation in this tree (nothing reads it).
The block budget falls 14 -> 10, so it is NOT a matched-compute partner to
`slot-spandec-strict`. CPU smoke on the tiny fan fixture with the span decoder on and the
hinge at 100 @ 0.9: builds, finite loss and grads, `gain_est` 0.8734 / max 0.8823 with the
penalty exactly 0.0 at `n_prelude 0` against 0.8756 / 0.8860 / 0.0 at `n_prelude 2`. That
is a 64-wide random-init model with two core blocks — it says the arm BUILDS and nothing
about what a 1024-wide trained core reads at step 200, which is why the prereg makes
reading `loop/slot_gain_pen` in the first 500 steps a precondition.

**2026-09-21, Part 1 probe 2 (anisotropic gain) — DONE, on fan4-all @ 5000, 24 rows, depth 6,
16 draws per slot, `lab/divergence/fan_gain_probe.py` (12 tests; the JVP checked against a
central finite difference on the real map, relative error 1.8e-4 at its minimum), artifacts
`lab/experiments/results/2026-09-19-lxtul-fan4/fan_gain_fan4-all_5000_d6.{json,txt}`.
Process slip as for probe 1: no frozen numeric prediction; the hypothesis was written (the
core contracts the deviations faster than the mean).**

| pass | centred rank | gain, random mean-direction | gain, random deviation-direction | dev / mean | gain on the state's OWN mean | on its OWN deviation |
|---|---|---|---|---|---|---|
| 1 | 2.29 | 0.976 | 0.973 | 0.997 | 1.000 | 1.174 |
| 2 | 2.83 | 0.897 | 0.896 | 0.999 | 0.969 | 0.987 |
| 3 | 2.79 | 0.890 | 0.891 | 1.000 | 0.984 | 1.002 |
| 4 | 2.58 | 0.888 | 0.888 | 1.000 | 0.988 | 1.004 |
| 5 | 2.33 | 0.887 | 0.887 | 1.001 | 0.988 | 1.003 |
| 6 | 2.15 | 0.887 | 0.887 | 0.999 | 0.988 | 1.003 |

**The hypothesis is REFUTED.** One core pass contracts the stream mean and the stream
deviations by the same factor (0.887 on random directions, ratio 0.997–1.001, SE ≤ 0.006),
and the state's own realised directions are barely contracted at all (0.99–1.00), the
deviation slightly LESS than the mean. Both subspaces are near-isotropic (effective
dimension ~7,500 of 12,288 deviation directions). So the rank fall 2.83 → 2.15 across
passes is not a Jacobian effect on the K axis, and the split-map construction (Jacobian 1
on the deviations) would change nothing the map is not already doing. The remaining
candidate is the AFFINE part of the pass, what each pass ADDS: the stream probe on the same
arm reads stream 0's norm 25 → 56 across passes with the `+−−−` sign family hardening 0.47
→ 0.95 (`fan_geom_fan4-all_5000_d6.txt`), i.e. a drive that pushes one stream along a
shared axis. Not measured by this probe; the trig arm (re-inject the trigger every pass)
and the fp0 arm (the terminal fixed-point term is also a drive on the last pass) are the
two arms already queued that act on the drive rather than the map. Caveat: the numbers are
the EAGER map's (`tg_scoped_kernels` forced off, the fused path has no second derivative);
the eager trajectory's ranks match the fused stream probe's to 0.01.

**2026-09-21, the three LCM instruments — DONE (probes, not preregs; process slip as above),
`lab/divergence/{code_roundtrip_probe,code_context_mi_probe,fan_stream_decode_probe}.py` on
`_span_decode.py`, 17 tests, artifacts `lab/experiments/results/2026-09-21-lcm-instruments/`.**

1. Round-trip gap on `tul-code-20k` @ 20k (Spark, 48 rows). Chance cosine between two spans'
   true codes 0.102. The k = 8 sample sits AT chance (cos to truth 0.098), so its round-trip
   gap is 0 for the wrong reason. The k = 1 sample carries signal (0.222) and decoding then
   re-encoding destroys all of it (0.109): gap +0.113 [+0.108, +0.120], Base-LCM's signature.
   The floor that makes it legible: the TRUE code decoded and re-encoded lands at cos 0.291, a
   gap of 0.709. This encoder/decoder pair loses most of a code on one round trip even when the
   code is exactly right. Quote that floor beside any future round-trip reading.
2. Context MI (model-internal form: the coda's CE on a span with the earlier cells present vs
   zeroed; the strict geometry makes the cut exact). Decoded span: +0.013 [+0.001, +0.025] and
   a shuffle control at 0. True span: +0.102 [+0.078, +0.129], shuffle +0.044. The sampler's
   own span gets no cheaper from its context: the LCM MI verdict on Base-LCM, on our decoder.
3. Fan stream decodes on fan4-all @ 5000 (3070, 32 rows). Between-stream decode distance
   0.474 against 0.248 within one stream under dropout: ratio **1.91 [1.84, 1.99]**; 90 % of
   slots have all four decodes pairwise different (75 % for the nuisance control). On text the
   four streams are not four spellings of one thing. The outside agent's null ("the deviations
   have no stable causal semantic effect once surface variation is controlled") is rejected on
   this proxy; the semantic-branch labelling it asked for is still not built.

Together: the LCTUL sampler at 20k is Base-LCM by two independent instruments, and the fan's
exit carries K distinguishable hypotheses while its passes stay flat. Both support the ladder
as ordered: fix the proposer (P1–P3) before asking the loop for depth.

**2026-09-21, rung P0 (fp0) FILED: `lab/experiments/failures/2026-09-21-lxtul-fan4-all-fp0.md`.**
The terminal fixed-point term does not contract the K deviations (rank_t6 2.01 vs 2.06
with it on). Off, it was holding the exit's SCALE (norms 2.3–4.2x through the loop,
pre-clip max 522 vs 29.4), and the coda reads the exit 0.006 better with pass 1 doing
twice the work (K1−K6 +0.0102). The rungs keep the term at 1.0. The trig arm (rank_t6
1.47, one stream at 95 %) says re-supplying identity every pass collapses harder, not
softer; its filing follows the runner's readouts.

**2026-09-21, alternative 2 (trig) FILED: `lab/experiments/failures/2026-09-21-lxtul-fan4-all-trig.md`.**
Re-supplying the per-stream trigger at every pass collapses the streams HARDER (rank_t6
1.43 vs 2.03, stream 0 at 4.1x its siblings, `+---` 95 % from pass 3), with the read,
the K-curve and the CE unchanged inside the seed spread (paired −0.005). Both Part-1
state levers are now closed (fp0, trig); the ladder's claim that a pass needs a JOB is
what remains, and P1 (noise) then P2 (denoise) are queued behind tlow and pk8.

**2026-09-21, the schedule arm (tlow) FILED: `lab/experiments/failures/2026-09-21-lctul-cfg-tlow.md`.**
E[t] 0.30 on the flow thinker doubles the field's context dependence (1 % → 1.9 %), reads
a sampled code at 2.7 × chance in-batch (new instrument), one draw 0.010 better than the
parent, k-curve −0.0068; every clause moved the predicted way by about half its bar and
none reached it. The ceiling vs the strict ruler stays +0.64. Risk added: the schedule
is not a lever on a lossless code. The sonar arm (P3) should be re-cut over the uniform
schedule with the shift as a second factor; it is held on its C-1 miss regardless.

**2026-09-21, pk8 FILED (success, `lab/experiments/successes/2026-09-21-strict-pk8-ruler.md`):**
eight continuous cells on one stream are worth 0.035 nats to the reader (strict partner
ruler) with the loop flat (K1−K6 +0.0010); vq8 sits 0.059 above that width ruler, so the
discrete write's deficit is the quantizer's own. Width is a reader lever, not a depth
lever, at 2, 4 and 8 cells.
