# Agent Note: Slot-loop campaign synthesis, 2026-09-09 to 2026-09-22

Status: implemented

## Problem

The TUL slot loop is MORPH's "think once, decode cheap" design. Its passes earn no depth.
From 2026-09-09 to 2026-09-22 about forty experimental arms asked why. This note is the
record of that campaign in one place. It states the design filter that every future
slot-loop proposal must now pass on paper.

The question: why does the slot loop's K-curve sit near zero while the plain looped model
earns a large one?

The yardsticks, as the campaign used them:

- **Token K1−K6.** Token CE at forced loop depth 1 minus token CE at forced depth 6, on the
  same 480 validation rows, read by `lab/divergence/core_depth_sweep.py` with a paired
  bootstrap CI. K3−K6 is the same number past pass 3. Positive means the later passes pay.
- **Paired depth-6 CE.** Token CE at depth 6, paired token by token against a named control
  on the same rows. A negative gap favours the arm. A 5k gap is context, not a ranking.
- **Plain-loop reference.** The plain norm_match loop earns K1−K6 0.185 under the Parcae
  noise entry and 0.033 under the prelude entry
  (`successes/2026-09-09-arc-per-pass-strength.md`,
  `failures/2026-09-09-arc-slot-loop-norm-match.md`).
- **Slot-loop floor.** Twelve slot arms read token K1−K6 inside [−0.0001, +0.0033]. The
  strict partner reads +0.0016 [+0.0013, +0.0019]
  (`failures/2026-09-13-arc-trajectory-prefix.md`). The campaign calls this floor "about
  +0.002".
- **Paid loop.** The paid loop earns 0.1685 with slots present
  (`successes/2026-09-01-a2-paid-loop.md`). Wolfe rejects it as "not TUL": tokens run
  through the core, so it is the plain loop with slot positions added.
- **Relay chain.** The LXTUL-R chain arms read +0.020 to +0.026, the largest slot-loop
  K-curves on the ledger. The number is a tax, not a gain. Step 1b sits 0.0265 behind the
  direct-read control fp0 at depth 6 and 0.053 behind it at depth 1
  (`failures/2026-09-22-lxtul-r-step1b.md`).

Two budget facts frame the problem. With no slots, web text loses 0.40 nats at 5k when
nothing crosses a span boundary (`failures/2026-09-11-arc-span-budget.md`). The whole slot
channel, cells zeroed, is worth about 0.19 nats at 5k, and K1−K6 is about 1 % of that
channel (`.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md`).
A 4x horizon moves the plain loop from 0.136 to 0.170 and the strict slot loop from 0.0001
to 0.0014 (same note).

## Decision

Every future slot-loop proposal states on paper, before any run, how it meets the two
conditions below. A proposal that cannot state both does not run.

**Condition A, no shallower route.** The coda's direct token read must not be able to do
the pass's job. If a shallower path can do the job, the loop is a free ride and the passes
learn nothing. Teacher forcing, an unrestricted coda read, a prelude that attends across
spans and a slot cell the coda reads directly are all shallower routes that this campaign
measured.

**Condition B, consumed.** The loss must pay for the pass's output. A pass whose output
no loss term reads with a steep gradient does no work, however well it can move the state.

**Helper, not a condition.** A job that pass 1 cannot finish alone makes contribution
easier to force. Per-pass targets that pass 1 meets in one step left passes 2 to 6 idle on
every arm that tried them. Wolfe, 2026-09-22: "this is not a universal truth". A design is
never rejected on paper because pass 1 could do the job.

**Standing call, Wolfe 2026-09-22.** No more restriction geometries whose mechanism is to
cut the coda's read so that the loop must matter. Strict geometry, the mask arm and the
relay chain are three such geometries, and they read the same answer: the loop carries what
the cut forces through it, at a CE tax, and earns no computation.

**Standing calls already in force.**

- Never argue that NLP or web text does not need loop depth. Loop flatness is not a
  verdict on the task.
- TUL means the slot loop. The paid loop is not TUL.
- Score arms on the token K-curve and on paired depth-6 CE. Do not score them on
  matched-compute nats.
- A short-horizon CE gap is not a ranking.
- Fixed depth is not sampled depth. A `depth_fixed` rung is a tied-weight deep net and is
  never ranked against the sampled loop.

The depth law the campaign converged on is the reason for the filter. Depth is earned in
proportion to the loss share that has no shallower route: plain loop with noise entry
0.185, plain loop with prelude entry 0.033, slot side channel 0.002
(`failures/2026-09-21-lxtul-loop-denoise.md`, Updated hypothesis).

## Alternatives considered

The table lists every slot-loop arm family of the campaign. Paths are relative to
`lab/experiments/` unless they start with `.agents/`. K1−K6 is token K1−K6 at 5k on 480
rows unless the row says otherwise. "Paired" is the depth-6 CE gap, arm minus control. The
filter column names the condition the arm failed, or the restriction it used. "R" marks a
restriction geometry. "H" marks an arm whose per-pass job pass 1 met alone.

| Family | Arm | Token K1−K6 | Paired depth-6 CE vs control | Filter reading | Source |
|---|---|---|---|---|---|
| Ternary rule | plain loop, norm_match vs bf16 core | 0.185 vs 0.168 | not paired | reference, not a slot arm | `successes/2026-09-09-arc-per-pass-strength.md` |
| Ternary rule | slot ruler under norm_match | 0.0000; forecast 0.0067 | +0.11 to +0.26 behind plain | A | `failures/2026-09-09-arc-slot-loop-norm-match.md` |
| Entry | plain loop, Parcae entry rebuilt | 0.033 | depth-1 model 0.019 ahead at 5k | reference | `failures/2026-09-09-arc-e19-parcae-loop-entry.md` |
| Entry | slot M-next, noise entry | 0.0004 | not paired | A | `failures/2026-09-10-arc-slot-mnext-noise-entry.md` |
| Entry | strict, no prelude (np0) | +0.0045 | +0.1652 vs strict | A | `failures/2026-09-21-strict-np0.md` |
| Fixed-point term | slot MUX, term off | +0.0005 | not paired | A; term holds the exit still, depth indifferent | `failures/2026-09-10-arc-slot-mux-fixed-point-off.md` |
| Fixed-point term | fan4-all fp0 | +0.0102 | −0.0058 vs fan4-all | A; term holds scale, not rank | `failures/2026-09-21-lxtul-fan4-all-fp0.md` |
| Depth draw | slot map levers, depth draw | 0.117 | CE@6 +0.052, unpaired | depth dependence without value | `failures/2026-09-10-arc-slot-map-levers.md` |
| Depth ladder | plain mean 2 to 6 | 0.002 to 0.170 | final CE flat inside 0.024 seed spread | reference; clamped and fixed rungs out | `failures/2026-09-13-arc-depth-ladder-ship.md` |
| Per-pass target | staged M-next | 0.0020 | 0.081 better than ruler, one seed | H | `successes/2026-09-10-arc-slot-mnext-staged.md` |
| Per-pass target | staged M-next, 20k | 0.0023 | gap to plain 0.161, flat 10k to 20k | H | `successes/2026-09-10-arc-slot-mnext-staged-20k.md` |
| Per-pass target | staged, full read | 0.0014 | 0.070 behind ruler | H | `failures/2026-09-10-arc-slot-mnext-staged-fullread.md` |
| Per-pass target | MUX every pass | 0.0001 | 0.023 better than ruler | H; passes became one pass | `failures/2026-09-10-arc-slot-mnext-mux-every-pass.md` |
| Per-pass target | progressive loss | 0.0006 | 0.036 better than ruler | H | `failures/2026-09-10-arc-slot-mnext-progressive.md` |
| Per-pass target | per-pass LoRA | 0.0000 | 0.023 better than ruler | H; every `B_t` left zero | `failures/2026-09-10-arc-slot-mnext-per-pass-lora.md` |
| Per-pass target | per-pass plan / codaspan | +0.0008 / +0.0012 | +0.017 / +0.042 | H | `failures/2026-09-12-arc-objective-arms.md` |
| Per-pass target | LoopMTP horizon port | +0.0040 | −0.0001 vs fixed6 | H; six span-mean targets degenerate | `failures/2026-09-14-arc-horizon-passes.md` |
| Spandec | whole-span decoder, mask family | ≤ 0.0013 on every arm | −0.072 CE, channel win | A; prelude cells bypass the loop | `successes/2026-09-11-arc-span-decoder.md` |
| Mask | slot loop, mask, no MUX | −0.0001 | not paired | R | `failures/2026-09-10-arc-slot-loop-mask-norm-match.md` |
| Mask | slot MUX, mask | +0.0009 | not paired | R | `failures/2026-09-10-arc-slot-mux-mask-norm-match.md` |
| Mask | staged, mask | 0.0033 | 0.108 behind ruler | R, H | `failures/2026-09-10-arc-slot-mnext-staged-mask.md` |
| Strict geometry | coda reach all | ≤ 0.0016 | parity with bypass arm | R | `failures/2026-09-12-arc-strict-geometry.md` |
| Strict geometry | reach 1 / reach 2 | 0.0163 / 0.0127 | parity with full reach | R; carries history, not a thought | `failures/2026-09-12-arc-strict-geometry.md` |
| Strict geometry | Olympiad / Sudoku spandec strict | +0.0111 / +0.0049 | slot gap to plain +0.216 / +0.250 | R | `failures/2026-09-12-arc-math-under-norm-match.md` |
| Gradpass | gradient-conditioned pass | +0.0014 | exit 0.052 better than ruler | H; one descent step | `failures/2026-09-10-arc-slot-mnext-gradpass.md` |
| z-optimize | fitted z, hindsight | instrument | −0.944 and −2.626 vs loop z | instrument; used the answer | `results/2026-09-10-slot-z-optimize/README.md` |
| z-optimize | fitted z, causal | instrument | +0.2517 worse than loop z | instrument | `failures/2026-09-12-arc-latent-z-gradient.md` |
| Latent-z energy | egrad recon / egrad disc | +0.0007 / +0.0019 | +0.050 / +0.008 | B; the coda's loss is not steep in the moved state | `failures/2026-09-12-arc-latent-z-gradient.md` |
| Core-token objective | coretok | +0.0005 | −0.0684 vs strict, shared-block effect | B | `failures/2026-09-12-arc-core-token-and-critic.md` |
| Core-token objective | within-context critic | +0.0012 | −0.0025 vs strict | B; label is a near-tie | `failures/2026-09-12-arc-core-token-and-critic.md` |
| Loop on tokens | aux token path through the core | 0.0102 vs shipped 0.0005 | not paired | instrument; same blocks earn on tokens | `failures/2026-09-13-arc-loop-reads-tokens.md` |
| cond4 reader depth | strict cond4 / register cond4 | +0.00026 / +0.00062 | +0.0057 / +0.0159 vs partner | A | `failures/2026-09-13-arc-cond4-reader.md` |
| Reader capacity | ultralight one-block coda | +0.00169 | +0.0073 vs partner | A; weaker coda leans on z no harder | `failures/2026-09-13-arc-ultralight-coda.md` |
| Bootstrap | plain 5k then slot loop | +0.0019 | −0.2164 vs strict, seed effect | A | `failures/2026-09-13-arc-plain-then-tul-bootstrap.md` |
| Thought register | m4 | +0.00197 | −0.0217 vs strict, width effect | A; four cells collapse to rank 1.24 | `failures/2026-09-13-arc-thought-register.md` |
| Trajectory prefix | traj / trajrep / entryexit | +0.0342 pad artefact / +0.0019 / +0.0014 | width, six cells beat two by 0.030 | A | `failures/2026-09-13-arc-trajectory-prefix.md` |
| Write width | prefix 4 / bag_mean seed / HCA fix | +0.0001 / +0.0005 / +0.0004 | 0.039 better / 0.025 behind / 0.030 better, vs ruler | A | `successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md`, `successes/2026-09-10-arc-slot-mux-bagmean-norm-match.md`, `successes/2026-09-10-arc-slot-mux-hca-fix.md` |
| Write width | strict pk8 ruler | ≤ 0.005 | 0.035 better than strict pk2 | A | `successes/2026-09-21-strict-pk8-ruler.md` |
| Core swap | slot M-next on a Parcae core | +0.0001 | 0.035 better than ruler | A; core exonerated | `successes/2026-09-10-arc-slot-mnext-parcae-core.md` |
| Discrete write | vq8 / vq4 | +0.0015 / not filed | +0.0250 / +0.058 vs strict | A | `mixed/2026-09-13-arc-discrete-thought-vq.md` |
| Fan | fan4, cosine repulsion | +0.0023 | +0.0073 vs pk4 | A; rank-1 split on one axis | `failures/2026-09-19-lxtul-fan4.md` |
| Fan | fan4 epi / epivol | +0.0061 / +0.0047 | not paired; d6 4.3495 / 4.3306 vs pk4 4.3398 | A; epi gamed, epivol not | `failures/2026-09-20-lxtul-fan4-epi.md`, `failures/2026-09-20-lxtul-fan4-epivol.md` |
| Fan | select / select-gate | +0.0076 / +0.0030 | +0.0861 / +0.0205 vs pk4 | A; gate cashes 0.014 of 0.113 | `failures/2026-09-20-lxtul-fan4-select.md`, `failures/2026-09-20-lxtul-fan4-select-gate.md` |
| Fan | fan4-all, write all K cells | +0.0049 | −0.034 vs pk4 | A; fixes the reader, not the loop | `successes/2026-09-20-lxtul-fan4-all.md` |
| Fan | noise / trig / lineage | +0.0045 / +0.0089 / +0.0029 | +0.0017 / −0.0051 / +0.0057 vs fan4-all | A; state levers closed | `failures/2026-09-21-lxtul-fan4-all-noise.md`, `failures/2026-09-21-lxtul-fan4-all-trig.md`, `failures/2026-09-21-lxtul-fan4-all-lineage.md` |
| Carry | sum / gate on prev-reach1 | +0.0404 / +0.0086 | d6 +0.098 / +0.050 behind ruler, unpaired | R; the sum inverts the hop staircase | `failures/2026-09-19-loop-carry-prev-reach1.md` |
| Carry | persist, C1 | +0.0234 | +0.0248 vs fp0 | R; relay | `failures/2026-09-22-lxtul-r-step2-panel.md` |
| History streams | hist1, C2 | +0.0204 | +0.0255 vs fp0 | R; relay | `failures/2026-09-22-lxtul-r-step2-panel.md` |
| Hop distance | prev-reach1 staircase | per-hop, content h back arrives at pass h−1 | not paired | R; carried content decays per pass | `failures/2026-09-18-hop-distance-earning.md`, `failures/2026-09-19-hop-distance-plateau-and-dilution.md` |
| LXTUL-R | Step 0 reach split | instrument | far budget 0.126 [0.117, 0.136] | instrument; brackets fixed by the seed twin | `failures/2026-09-21-span-reach-split.md`, `failures/2026-09-22-coda-seed-twin.md` |
| LXTUL-R | Step 1, leaky chain | +0.0202 | +0.0194 vs fp0 | R | `failures/2026-09-22-lxtul-r-step1.md` |
| LXTUL-R | Step 1b, one-slot chain | +0.0261 | +0.0265 vs fp0; depth 1 +0.0527 | R; a tax, not a gain | `failures/2026-09-22-lxtul-r-step1b.md` |
| LCTUL | tul-code, sampled code, 5k | about −0.01, wrong sign | 0.7 nats behind ruler | B; the sample is an unconditional draw | `failures/2026-09-14-arc-tul-code.md` |
| LCTUL | tul-code, 20k | sampler k16−k1 +0.014, flat after phase 2 | one draw +0.626 vs ruler; sampled code +0.27 worse than no code | B | `failures/2026-09-14-arc-tul-code-20k.md` |
| LCTUL | conditioned thinker / thinker only / xm / lejepa / cfg-tlow | not filed as token K1−K6 | lejepa +0.68, cfg-tlow +0.64 vs ruler | B; the thinker is context-blind | `failures/2026-09-15-tul-code-conditioned-thinker.md`, `failures/2026-09-15-tul-code-thinker-only.md`, `failures/2026-09-15-tul-code-xm.md`, `failures/2026-09-16-tul-code-lejepa.md`, `failures/2026-09-21-lctul-cfg-tlow.md` |
| LCTUL | code_discrete, masked denoiser | k-curve inverts, round 1 best | not filed | B | `failures/2026-09-16-lctul-d-first-arm.md` |
| LCTUL | dplan semantic / LaDiR chain | 0.012 on CE / not filed | 0.030 below strict / 7.7 to 8.0 vs ruler 4.4 | B | `failures/2026-09-16-lctul-dplan-semantic.md`, `failures/2026-09-16-lctul-ladir-chain.md` |
| LCTUL | code_target arm A / unfrozen reader | +0.0323 frozen coda / +0.0047 | semantic OWN +0.0290 after unfreeze | H; arm A's curve is genericity | `successes/2026-09-17-lctul-target-unfreeze.md` |
| LCTUL | code_target progressive | −0.1173 | below arm A at every depth | H; one pass reaches the target | `failures/2026-09-17-lctul-target-progressive.md` |
| LCTUL | code-only with frozen ref | sign flips, void | void, reader worse than uniform | K-curve void (reader above uniform); the frozen twin itself is sound and reads H0: the predictable part of the fixed code is about 0.15 cosine whatever the front, found in one pass | `failures/2026-09-17-lctul-code-only-ref.md` |
| LCTUL | code_grade, graded target | void | winner code no closer to truth, −0.004 | B; ranking text does not rank codes | `failures/2026-09-17-lctul-graded-target.md` |
| Denoise | loop denoise, teacher forced | +0.5575 at 20k, genericity | +0.1260 worse than arm A | A; teacher forcing is a bypass | `failures/2026-09-21-lxtul-loop-denoise.md` |

Three alternatives to the filter itself were weighed and declined.

1. **A stability or spectrum target.** The pre-campaign reading said the cure was
   contractivity. The fixed-point term, the gain hinge and the spectral caps all hold the
   map. None of them moves the K-curve, and the fp0 rows show the term holds the exit's
   scale, not its depth.
2. **More restriction geometry.** Strict, mask and chain each forced content through the
   loop. Each read either the floor or a relay tax. Wolfe closed the lane on 2026-09-22.
3. **The helper as a hard condition.** A rule that rejects any job pass 1 can finish would
   have rejected the plain loop, which earns 0.185 with a strong pass 1. Wolfe's 2026-09-22
   qualification keeps it a helper.

## Consequences

**Closed lanes.**

- The restriction lane: strict geometry, the mask arm and the relay chain. No new arm cuts
  the coda's read to force the loop to matter.
- The relay lane: LXTUL-R is a measurement, not a design. Reaching g spans back costs g
  core passes where one attention hop is free. Step 1b, C1 and C2 share one depth-6 CE
  within 0.002. No C3, no C4.
- The per-pass-target lane, as tried: staged, every-pass, progressive, per-pass LoRA,
  objective arms, LoopMTP and the progressive code target. Every per-pass target was met in
  one step.
- The state-lever lane in the fan: repulsion, epiplexity, volume, noise entry, trigger
  re-supply, fp0 and lineage. Re-supply collapses harder. The exit rank follows the
  objective.
- The input, width and core axes: seed rank, prefix width 2, 4 and 8, the HCA fix, the
  Parcae core, cond4 and the ultralight coda all leave the floor in place.

**Open lanes.**

- The fan's "build candidates, not gates" result. fan4-all's per-token read recovers
  0.056 of 0.099 nats of oracle value and leaves 0.042 of selector regret
  (`successes/2026-09-20-lxtul-fan4-all.md`). The selector problem is solved by reading all
  candidates. The loop still does not make the candidates.
- The LCTUL code target. Its collapse has a named cause: a frozen encoder fed by a live
  front is not a fixed target. The next arm is an EMA target encoder with per-coordinate
  variance floors, from JEPA-Anything, arXiv 2609.20800:
  [`2026-09-22-lctul-ema-target-and-factors.md`](../../proposed/architecture/2026-09-22-lctul-ema-target-and-factors.md)
  (Stage 1 the EMA target and the online floor, Stage 2 the per-pass factor split). Its
  premise from the ledger: every fixed code target so far had a predictable part of about
  0.15 cosine whatever the front (`failures/2026-09-17-lctul-code-only-ref.md`, H0), and
  a target that cannot move cannot leave that ceiling.

**Instruments.**

Trusted:

- Token K1−K6 and K3−K6 from `core_depth_sweep.py` on 480 rows with a paired CI.
- Paired depth-6 CE on identical rows against a named control.
- The planted probe as an exact induction test. It reads content arrival with exact zeros
  before the arrival pass.
- The depth-1 control as a contribution instrument.

Downgraded:

- The span-swap probe is a corruption. Language is causal, so a swapped span is an
  off-distribution input. Read it only as a leak check, for its exact zeros and its arrival
  staircase.
- A K-curve read through a frozen coda measures genericity. It tracks how bland the cell
  gets with depth, not a match to the code. Arm A's +0.0323 and denoise's +0.5575 are this
  artefact.
- A fitted z that saw the tokens it is scored on used the answer. The hindsight fit is
  worth −0.94 to −2.63 nats; the causal fit is +0.2517 worse than the loop's own z.
- A CE read above uniform, ln 49152 = 10.80, is not a reader. Check it first.

**Cost of this decision.** The filter adds a paper step before every slot-loop run. It
does not forbid any mechanism. It forbids running a mechanism whose route to the loss has
not been stated.

Related notes:
[`2026-09-21-lxtul-r-reach-composition.md`](../../proposed/architecture/2026-09-21-lxtul-r-reach-composition.md),
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
[`2026-09-22-lxtul-r-step2-speed-and-carry.md`](../../proposed/architecture/2026-09-22-lxtul-r-step2-speed-and-carry.md),
[`2026-09-03-tul-loop-contribution-drawing-board.md`](../../proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md),
[`2026-09-09-ternary-core-is-a-weak-per-pass-map.md`](2026-09-09-ternary-core-is-a-weak-per-pass-map.md).
