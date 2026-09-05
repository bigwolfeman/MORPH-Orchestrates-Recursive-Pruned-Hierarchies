# Agent Note: prove TUL on Parcae, then select MORPH depth

Status: proposed

## Problem

Wolfe wants a measured loop-contribution reference, a useful cheap TUL mechanism,
and a defensible depth choice for MORPH. These are different questions.

The completed [Parcae OWT experiment](/home/wolfe/parcae/docs/experiments/successes/2026-09-05-parcae-owt-loop-contribution.md)
owns all measured CE, throughput, and training-trend results. It is the positive
control for this plan. Its earning grows between 5k and 10k under its schedule.
This does not isolate training duration from LR cooldown and weight-decay changes,
nor show that its optimal inference depth grows. No TUL arm has passed yet.

The [new paper reading](../../../../docs/references/looping-depth/virtual-logical-depth.md)
supports separating reasoning from memorization and testing where recurrence
acts. It provides no universal optimal K. The local
[Parcae scaling paper](../../../../docs/references/looping-depth/parcae/parcae.md)
adds evidence for jointly choosing training tokens, recurrence, and BPTT.

This proposal continues the
[think-once drawing board](2026-09-03-tul-loop-contribution-drawing-board.md).
It changes the first test platform to Parcae. It does not replace the production
recipe or erase the MORPH failure records. The
[contractivity proposal](2026-09-04-loop-contractivity-as-design.md) remains a
record of tested hypotheses, not proof that every useful map must contract.
No existing implemented decision is fully superseded by this unbuilt proposal.

## Proposal

### Current authorization, 2026-09-05

Wolfe replaces the initial mean8/S1/C panel below with a mean16 panel at BPTT4
and BPTT8. Do not retrain P8 or train separate K1 controls. The recovered
`/home/wolfe/morph-to` panel supplies the post-loop four-slot-block conditioner
and its four-extra-token-block comparison. These are not extra prelude layers.

The current [Parcae experiment record](/home/wolfe/parcae/docs/experiments/failures/2026-09-05-parcae-tul-cond4-panel.md)
owns the six full-gradient arm definitions and preparation gates. The
[Parcae implementation note](/home/wolfe/parcae/.agents/notes/implemented/architecture/2026-09-05-slot-only-tul-panel.md)
owns the model and queue decisions. Wolfe authorizes serial launch after the
correctness and memory gates. The user-set460W cap stays in place throughout.
The queue stops on failure. Detached/MUX variants are not part of this queue.

The remaining sections preserve the earlier proposal and possible confirmation
work. They do not authorize its superseded first panel or its conditional depth
screen. In particular, this queue cannot claim an advantage over independently
trained cheap/S1 controls, because those controls are not being run. Migration
and production depth selection remain contingent on actual results.

The six-arm queue is now complete. Its experiment record rejects useful late
recurrence and conditioner quality parity under this recipe. This does not
authorize migration or a new production depth. The implementation works; the
scientific promotion gate does not pass. The owning record contains the numbers
and proposed next distinctions. No follow-up training is launched on closeout.

### 1. Freeze the reference and define success

Keep the existing checkpoint weights, tokenizer revision, token-cache identities,
optimizer, and data order. Preserve 2048 real target tokens per row across arms.
Auxiliary slot positions must not displace examples, add target loss, or change
the real-token denominator. Rebuild boundaries on this tokenizer causally.

Use the recorded K1-to-K8 and K3-to-K8 reductions as targets, not isolated scores
to maximize. A larger gap obtained by worsening K1 is not progress. Report:

- Absolute CE at each K and a trained cheap baseline.
- Within-checkpoint CE reduction as K increases.
- Improvement over the independently trained K1 slot control.
- Fraction of the plain deep model's advantage over the cheap control recovered.
  This ratio is meaningful only if the denominator is positive.
- Real training and cached-decoding time, memory, real tokens, and executed blocks.

Keep the existing 480 rows as development data. Before confirmation, freeze a
fresh document-separated holdout that has never entered any training cache.
Preserve document IDs and cluster uncertainty by source document. Keep the
original packed-row statistic for continuity, with its caveat. Check duplicates
between training and the confirmation holdout.

### 2. Build one slot-only Parcae candidate

Keep the author's 2/2/2 block layout, width, initialization, diagonal injection,
native optimizer, and per-sequence total-depth sampler. Start at training mean
K8 and BPTT4. Do not start with per-slot halting or a new depth curriculum.

The computation is:

```text
real tokens -> causal prelude -> contextual token states -> token-fed coda -> CE
                                  |
                         completed boundary states
                                  |
                         compact slot-only core
                                  |
                    frozen thought prefixes for next span -> coda
```

One slot corresponds to each completed span. Seed its injection input from the
completed boundary token's contextual prelude state. Preserve the author's
separate random recurrent initialization and injected input. Start with a learned
slot marker and projection only where dimensional alignment requires it.

Only the compact slot sequence enters the recurrent core. Project each final
thought into two attended coda prefix positions before the following span.
Keep full causal token access in the initial coda. Do not damage the token path
to manufacture thought dependence. Keep the decoder autoregressive and token-fed.
Use ordinary target-token CE. No auxiliary forecast, latent regression, decoder
dropout tax, gain hinge, or iteration embedding in this first candidate.

The prefix made after token b may condition the prediction of token b+1, never
an earlier target. The prefix remains fixed until the next boundary closes.
The first span uses an explicitly defined empty-context/BOS state. A prefix must
not change what has already been scored. Treat real-token positions and prefix
positions explicitly rather than silently changing RoPE distances.

The author model's `_current_input_ids` value-embedding lookup needs compact
boundary IDs aligned to slot positions. Reusing the full token-axis IDs inside
the slot core is an implementation error. Mask padded slots in every operation.

The first real inference implementation must cache slot attention K/V separately
for each recurrent iteration and each layer. Final-iteration K/V cannot stand in
for every iteration. Prelude and coda keep their own causal caches. Appending a
future span must leave prior slot initial states, positions, and logits unchanged.

### 3. Cross correctness gates before training

- TUL disabled reproduces native Parcae logits and next optimizer update.
- Full-sequence and cached incremental logits agree within frozen bf16 tolerances
  at K1, K4, and K8, including first-token emission and boundary runs.
- Future-token perturbations leave earlier logits unchanged. Padding and batch
  regrouping cannot expose future slots or alter the loss denominator.
- Instrumented core inputs contain slot positions only. Increasing K must not
  increase token-axis recurrent work. Report actual slot count and padded work.
- Future token CE produces gradients through the prefix, slot state, and core.
- Checkpoint continuation restores optimizer state, RNG, token cursor, and caches
  where applicable. Preserve the already measured CPU continuation contract.

Use Hydra and online W&B with the full resolved model and runtime configuration.
Keep one GPU training process at a time. Reuse the temporary 460 W service cap
and automatic restoration. No gradient accumulation or activation checkpointing.

### 4. Run the mechanism panel at one fixed recipe

Write and commit predictions before running any new arm.

| Arm | Purpose |
|---|---|
| P8, native Parcae mean8/BPTT4 | Existing positive control, re-evaluated by the same instruments |
| C, prelude/coda with no core or thoughts | Independently trained cheap reference |
| S1, identical slot architecture trained at K1 | Separates compression and extra parameters from recurrence |
| S8, slot architecture trained at mean8/BPTT4 | Candidate recurrent thought computation |

Use batch12, the same 245.76M real-token budget, optimizer schedule, and 5k/10k
scientific checkpoints for the first comparison. If an arm cannot fit, choose a
common batch and explicitly rerun the paired reference. Do not silently change
its LR or token budget. The existing P8 may serve as the seed42 reference only
when its data and training contract match the comparison.

At both checkpoints, evaluate fixed K1/2/3/4/6/8/12/16 on identical examples and
paired initial-state seeds. Keep target-token offset-in-span profiles. Compare
intact thoughts with zeroed, shuffled, and reader-blocked thoughts while keeping
masks, positions, and prefix counts fixed. Those interventions test dependence,
not usefulness by themselves. Include a sample-exact cached decode benchmark.

Suggested engineering promotion gates, to freeze in that preregistration:

1. S8 at its selected K beats both trained C and trained S1 in absolute CE,
   with paired 95% intervals above zero for their CE reductions.
2. S8 has at least 0.005 nats of K3-to-K8 contribution with a positive paired CI.
   This is a proposed minimum useful late-loop gain, not a paper result.
3. S8's selected operating point is within 0.02 nats of P8 at K8. It also improves
   the measured plain-model CE/latency frontier: at least 1.2x faster cached decode
   at quality matched within 0.005 nats. Compare against plain K4/K6/K8, not only
   its most expensive depth. These tolerances are proposed engineering choices.
4. The result survives a second independent training seed and the fresh holdout.
   Repeat paired controls, not only the winning arm. Report training-seed results
   separately from row-bootstrap intervals.

Record partial outcomes honestly. Useful slots with no late-loop gain establish
compression, not recurrent TUL. A large shuffle penalty with worse absolute CE
fails. A model that saves FLOPs but loses its wall-clock advantage fails the
efficiency claim. If the slot reader ignores useful input, test a declared
causal stronger-prefix control before another training campaign. Do not provide
future target text as a deployable oracle.

### 5. Select training and inference depth separately

For the current plain model, keep training mean8. Treat inference K4 as a
speed-focused candidate, K6 as near-full-quality, and K8 as the reference.
Measure cached latency before calling any one of them optimal.

After S8 shows useful recurrence, screen training recipes mean4/BPTT2,
mean8/BPTT4, and mean12/BPTT6 at equal real-token budgets. Preserve the author
sampler: sample total depth before splitting forward-only and gradient steps.
This compares joint depth/BPTT recipes, not an isolated forward-depth change.
Add mean16/BPTT8 only if mean12 remains competitive. Measure batch-maximum loop
work; the configured Poisson mean is not the GPU's executed maximum.

Confirm the leading pair at equal training compute as a separate comparison.
Longer loops consume more compute per token. Count forward-only and backward
work separately, then measure actual wall time under the same cap. Schedule LR
by the declared token or compute budget, not a silently changed step horizon.
Keep cost-matched results separate from token-matched results.

Select the lowest measured inference K within 0.005 nats of that model's best
development CE, subject to measured latency and reasoning requirements. This is
a local selection rule, not a universal scaling law. Use a locked confirmation
set after selecting K. The current Parcae CE curve nominates K6 under this rule.

Test the longer-training hypothesis in a separate preregistered continuation
from the preserved 5k checkpoint. Compare branches with explicit, common LR/WD
schedules and a longer declared horizon. Do not extend the cooled-down 10k run
by silently resetting its LR. Retain 5k/10k and add an approved later checkpoint
for this new experiment. No later checkpoint is authorized or launched here.

### 6. Check reasoning, then migrate the mechanism

The new paper motivates an iGSM-style controlled reasoning panel. OWT CE remains
the first gate. Before any reasoning claim, give P8 and the winning TUL arm the
same separately declared reasoning training budget. Split by dependency-template
identity and difficulty. Measure exact answers at trained operation counts and
held-out harder counts. Freeze prompts, decoding, and output-token budgets.
Separate final-answer accuracy from chain-of-thought length and wall time.
Do not interpret a tiny OWT-only model's untrained math failure as a depth verdict.

Then reproduce the Parcae computation in a minimal MORPH integration and verify
its parity contracts. Add MORPH-specific attention, residuals, embeddings,
quantization, and sparsity features one at a time. At each addition compare a
plain model and its TUL counterpart under the same configuration. Keep the
reference tokenizer and target rows through this causal migration ladder.
Changing tokenizer requires a new reference or a byte-normalized comparison.

Do not copy K across architectures:

| Current configuration | Block applications at K | Examples |
|---|---|---|
| Parcae 2/2/2 | 4 + 2K | K4=12, K8=20 |
| MORPH base/notul/tul_a2 4/6/4 | 8 + 6K | K2=20, K3=26, K6=44 |

These counts are not equal FLOPs. MORPH also caps its Poisson draw at8 and uses
full BPTT8, unlike this Parcae recipe. Read the instantiated model configuration
of each historical checkpoint rather than applying today's base YAML to it.

For MORPH, first measure the plain and migrated TUL inference frontier at
K1/2/3/4/6/8 with a fixed physical core. Train-depth selection follows that
baseline and uses matched token and compute panels. Only then choose a MORPH
operating depth for a named workload and budget. No production default changes
on the strength of this plan alone.

## Alternatives considered

Immediate MORPH mean16 training changes a costly six-block loop before proving
the slot mechanism. Increasing inference K beyond8 on the current Parcae gives
almost no OWT CE benefit. Neither is the first experiment.

Repeating all tokens is the paid-loop design, not the requested cheap TUL goal.
Restricting decoder access or adding auxiliary objectives first repeats past
confounds. A read-only token-memory bank is a possible later slot-encoder arm,
but adds attention and cache changes before the contextual-boundary candidate
is measured. Parameter reuse order is also held fixed rather than adding a new
sequence/cycle/inverse-cycle search.

## Acceptance criteria

This turn completes the plan, cache, full reading, and Parcae result record only.
The proposed mechanism is accepted only after correctness, absolute-quality,
late-depth contribution, real efficiency, replication, and confirmation gates.
Migration needs its own parity evidence and paired feature ladder. A single CE
number or a successful training process does not satisfy these criteria.

## Risks

Compression can remove information that token-level recurrence uses. Full causal
token access can let the decoder ignore slots. Initial-state and cache mismatches
can manufacture a depth effect. A changed cheap reference can make a recovered-
gain ratio misleading. Readout cost and padding can erase nominal savings.

The VLD paper does not test TUL and contains metric and recipe ambiguities.
Two checkpoints do not establish a training scaling law. OWT CE, reasoning
accuracy, and latency can prefer different depths. The proposed thresholds and
budgets must be frozen before experiments, not adjusted after seeing a winner.
