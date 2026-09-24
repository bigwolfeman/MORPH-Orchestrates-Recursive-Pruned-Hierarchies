# Emergent Search and Backtracking in Latent Reasoning Models: reading note

Read 2026-09-23, for the controls of the LXTUL-G build
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../../.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md)).
The brief asked for instruments only.

## Citation (verified)

Jasmine Cui, Charles Ye (equal contribution, independent). *Emergent Search and
Backtracking in Latent Reasoning Models*.
[arXiv 2602.08100v1](https://arxiv.org/abs/2602.08100), 2026-02-08. Preprint dated
2026-02-10. No venue. No code link in the PDF.

Read in full: the paper is 4 pages, all of it read.

## Local cache

Not cached into `ignore/papers/` in this pass (the brief allowed only the note files).
Fetched from `https://arxiv.org/pdf/2602.08100` on 2026-09-23: 684,191 bytes, 4 pages,
SHA256 `1d7011ebd02df6be671df48ba7592648d16f58c427fde56fdad8b7df158c7457`. Suggested cache
name: `ignore/papers/2602.08100v1-emergent-search-backtracking.pdf`.

## What it actually does

The model is Huginn-0125 (Geiping et al. 2025), 3.5B parameters: a prelude (embedding + 2
layers), a recurrent block of 4 layers iterated K times, and a coda (2 layers + LM head).
Huginn was trained with randomly sampled recurrence depth, so the coda has seen the state
at every depth. The paper uses that to decode the model's prediction at every step:

    h_0 = P(x),   h_i = R(h_{i-1}),   p_i = softmax( C(h_i) ),   i = 1..K,  K = 30

and tracks the probability of each of the four answer options across the 30 steps.

**The benchmark.** 260 four-choice questions (factual recall, definitions, multi-step
logic, arithmetic, adversarially misleading items). Each stem has three answer-set
variants, which are causal manipulations of difficulty:

- Base: all distractors plausible and related to the stem.
- Easy: distractors obviously unrelated.
- No correct answer: the correct option is replaced by another distractor.

Each question-variant pair runs under 25 random permutations of the answer order, to
control position bias. Shaded regions are 95 % bootstrap intervals over permutations.

## The instruments, with their exact definitions

**E-1. Per-step belief decode.** `p_i` restricted to the answer options, every step. The
authors stress this is an exact readout only because the coda was trained on every depth.

**E-2. Exploration length.** Exploration ends at the first step where the step-to-step KL
stays small:

    D_KL( p_{i+1} || p_i ) <= 0.01   for 3 consecutive steps

Base questions explore 54 % longer than their Easy twins (same stem, same correct answer).

**E-3. Backtracking event.** The argmax option is `a` for at least 3 consecutive steps,
later becomes `b != a` for at least 3 consecutive steps, and the final answer is `b`.

- 32 % of Base instances have at least one backtracking event.
- Instances that backtrack are 34 % more accurate than those that do not.
- 52 % of backtracks land on the correct answer.

**E-4. Direction of backtracking.** Sentence-embedding cosine between the stem and each
option ranks the distractors. The abandoned answer is:

    most similar distractor     72 %
    2nd most similar            22 %
    least similar                6 %      (chance for the top one: 25 %)

**E-5. Entropy by variant.** Mean entropy of `p_i` per step: Easy falls fastest, Base
slower, No-correct-answer stays high for all 30 steps and never enters a low-entropy
state.

**E-6. The phase description.** Exploration (mass spread across options), shallow
commitment (usually to the most stem-similar distractor), then convergence or backtracking.

## Limits the paper states or shows

One model, one 260-item synthetic benchmark, four-choice questions only. The authors say
open-ended generation is untested. No statistics beyond bootstrap intervals over answer
permutations; the 34 % accuracy gain is a correlation between backtracking and accuracy,
not an intervention. No control that separates search from a readout that is simply
noisier at shallow depth.

## What LXTUL-G should take

- **E-1 on the slot loop's passes.** The slot loop draws its depth per sample (Poisson,
  or a fixed rung), so the coda has consumed intermediate pass states and a per-pass decode
  is a real readout, not a projection. Decode the next span's first-token distribution and
  its span CE from every pass, per prior sample. MORPH already has the aggregate version
  (the K-curve); E-1 is its per-slot trajectory.
- **E-2 as the exploration length per slot.** Use the same rule on the first-token
  distribution (KL <= 0.01 for 3 consecutive passes), at an eval depth of 16 so there is
  room for the rule to fire late. Sort slots by the next span's difficulty (the span's
  mean token entropy under the model, or teacher entropy) and ask whether exploration
  length rises with difficulty, as E-2's 54 % does. A loop that commits on pass 1 has
  exploration length 1 everywhere.
- **E-3 as the backtracking rate, and its direction.** Count backtracking events on the
  first-token argmax per slot and prior sample; report the fraction that end on the true
  first token (E-3's 52 %). A backtrack that moves toward the realized token is evidence
  of correction; a backtrack that is a coin flip is noise.
- **E-5 as the no-answer control.** The analogue of "no correct option" is a span whose
  continuation is genuinely open (high teacher entropy). A searching loop should stay
  spread across prior samples and across passes there, and converge on low-entropy spans.
  Diversity that does not track entropy is noise.
- **Their difficulty manipulation as a design.** Same stem, only distractor plausibility
  changed. For LXTUL-G the cleanest equivalent is to hold the context fixed and compare
  slots by the entropy of their next span, measured independently of the loop.

## What does not transfer

- **A four-option readout.** Their beliefs live on four labelled options; a text span has
  no option set. The first-token distribution over the vocabulary is the nearest proxy,
  and it is much noisier. The 0.01 KL threshold was set for a 4-way distribution and will
  need recalibration on a vocabulary-sized one.
- **Thirty steps.** The rules need at least 6 steps to register a backtrack. MORPH trains
  at mean depth 6; eval at depth 16 is required, and beyond the training range the readout
  may degrade.
- **A frozen, pretrained reader.** Huginn's coda is the trained reader of its own loop. On
  MORPH a per-pass decode through a reader that is not trained on that pass's
  distribution measures the reader's taste, not the loop: the frozen-coda K-curve on the
  code-target arms tracked how generic the cell was, not how well it matched
  ([`2026-09-18-frozen-coda-k-curve-measures-genericity.md`](../../../../../.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md)).
  Per-pass decodes are valid only on arms whose coda trains on sampled depths.
- **No causal link to accuracy.** Backtracking correlates with accuracy; the paper never
  intervenes. It gives definitions, not a verdict rule.
